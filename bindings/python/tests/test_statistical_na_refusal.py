# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Require a rebuilt pyo3 candidate; the src/ historical oracle is not this API."""
import pytest

from nirs4all_io import _native

A = "Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;10.0;3.0\n"
B = "Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;30.0;3.0\n"


def _spec(path, na):
    return {
        "schema_version": 1,
        "name": "statistical-na-mean-heldout-probe",
        "task_type": "regression",
        "sample_index": {"by": "id", "key": "Sample_ID"},
        "params": {"delimiter": ";", "decimal": ".", "has_header": True, "na": na},
        "sources": [{
            "id": "x", "role": "mixed", "input": str(path), "key": "Sample_ID",
            "params": {"header_unit": "nm"},
            "columns": {"features": ["1100"], "targets": ["target"], "metadata": ["set"]},
        }],
        "partitions": {
            "by": "column", "column": "set", "train_values": ["train"],
            "test_values": ["test"], "unknown_policy": "error",
        },
    }


@pytest.mark.parametrize("method", ["mean", "median", " Median ", "MEAN"])
@pytest.mark.parametrize("policy", ["replace", "ignore", "remove_sample", "remove_feature"])
@pytest.mark.parametrize("per_column", [True, False])
@pytest.mark.parametrize("source_override", [True, False])
def test_native_statistical_na_refusal(tmp_path, method, policy, per_column, source_override):
    path = tmp_path / "literal.csv"
    path.write_text(A, encoding="utf-8")
    na = {"policy": policy, "fill": {"method": method, "per_column": per_column}}
    value = _spec(path, {"policy": "ignore"} if source_override else na)
    if source_override:
        value["sources"][0]["params"]["na"] = na
    for operation in (_native.validate, _native.assembled_full):
        with pytest.raises(ValueError, match=r"na\.fill\.method: .* is refused: .*fitted preprocessing"):
            operation(value)


def test_native_ignore_preserves_literal_train_bytes_masks_and_ids(tmp_path):
    path = tmp_path / "literal.csv"
    value = _spec(path, {"policy": "ignore"})
    path.write_text(A, encoding="utf-8")
    _native.validate(value)
    a = _native.assembled_full(value)
    path.write_text(B, encoding="utf-8")
    b = _native.assembled_full(value)
    assert a["blocks"]["train"] == b["blocks"]["train"]
    precision = {
        "dtype": "float64", "origin": "default", "source_dtype": None,
        "requested_dtype": None, "consumer": None,
    }
    for full in (a, b):
        assert full["assembled_schema_version"] == 3
        assert full["partition_order"] == ["train", "test"]
        assert full["identity"]["provenance"]["sample_id"] == "Sample_ID"
        assert full["identity"]["provenance"]["source_ids"] == ["x"]
        block = full["blocks"]["train"]
        ids = next(column["values"] for column in block["metadata"]["columns"] if column["name"] == "Sample_ID")
        assert ids == ["S_missing", "S_finite"]
        assert block["x"][0]["data_bytes_hex"] == "00000000000000000000000000000040"
        assert block["x"][0]["validity"] == [False, True]
        assert block["x"][0]["precision"] == precision
        target = block["y"]["columns"][0]["block"]
        assert target["data_bytes_hex"] == "000000000000f03f0000000000000040"
        assert target["validity"] == [True, True]
    assert a["blocks"]["test"]["x"][0]["data_bytes_hex"] == "0000000000002440"
    assert b["blocks"]["test"]["x"][0]["data_bytes_hex"] == "0000000000003e40"
