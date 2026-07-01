# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""The idiomatic Python surface: native inputs (str / Path / list / dict) in,
typed DatasetPlan / DatasetSpec mappings out."""

import json
from pathlib import Path

import pytest

import nirs4all_io as nio

CORPUS = Path(__file__).resolve().parents[3] / "tests/goldens/contract/corpus"


def test_to_spec_accepts_pathlib_and_returns_typed_spec():
    spec = nio.to_spec(CORPUS / "train_test")  # a pathlib.Path, not a str
    assert isinstance(spec, nio.DatasetSpec)
    assert isinstance(spec, dict)  # still a dict: subscriptable, serializable
    assert spec.schema_version == 1 == spec["schema_version"]
    assert spec.name == "train_test"
    assert len(spec.sources) >= 1
    json.dumps(spec)  # JSON-serializable
    nio.validate(spec)  # the typed spec round-trips through validate
    assert "DatasetSpec" in repr(spec) and "train_test" in repr(spec)


def test_infer_accepts_pathlib_list_and_returns_typed_plan():
    plan = nio.infer([CORPUS / "x_y_separate" / "X.csv", CORPUS / "x_y_separate" / "Y.csv"])
    assert isinstance(plan, nio.DatasetPlan)
    assert "resolved_spec" in plan


def test_plan_exposes_decisions_and_resolved_spec():
    plan = nio.infer(CORPUS / "single_combined")
    decisions = plan.decisions()
    assert {"structure", "task_type", "signal_type"} <= decisions.keys()
    for d in decisions.values():
        assert {"value", "score", "ambiguous"} <= d.keys()
    assert isinstance(plan.overall_score, float)
    rs = plan.resolved_spec
    assert isinstance(rs, nio.DatasetSpec)
    nio.validate(rs)
    assert "DatasetPlan" in repr(plan)


def test_load_accepts_pathlib():
    assembled = nio.load(CORPUS / "x_y_separate", target="assembled")
    assert "blocks" in assembled


def test_load_accepts_reference_object_with_to_io_spec():
    class ReferenceDatasetDouble:
        def to_io_spec(self):
            return (
                {
                    "sources": [
                        {"id": "x", "role": "features", "input": "X.csv"},
                        {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}},
                    ]
                },
                CORPUS / "x_y_separate",
            )

    spec = nio.to_spec(ReferenceDatasetDouble(), name="reference")
    assert isinstance(spec, nio.DatasetSpec)
    assert spec.name == "reference"
    assert all(Path(source["input"]).is_absolute() for source in spec.sources)
    nio.validate(spec)

    assembled = nio.load(ReferenceDatasetDouble(), target="assembled", name="reference")
    assert assembled["name"] == "reference"
    assert "train" in assembled["blocks"]


def test_to_io_spec_base_dir_applies_to_secondary_file_refs(tmp_path):
    (tmp_path / "X.csv").write_text("1000;1002\n0.1;0.2\n0.3;0.4\n0.5;0.6\n0.7;0.8\n", encoding="utf-8")
    (tmp_path / "X_snv.csv").write_text("1000;1002\n-1.0;1.0\n-1.0;1.0\n-1.0;1.0\n-1.0;1.0\n", encoding="utf-8")
    (tmp_path / "Y.csv").write_text("y\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")
    (tmp_path / "train_idx.txt").write_text("0\n1\n", encoding="utf-8")
    (tmp_path / "test_idx.txt").write_text("2\n3\n", encoding="utf-8")
    (tmp_path / "folds.csv").write_text("fold_0\n0\n1\n", encoding="utf-8")

    class ReferenceDatasetDouble:
        def to_io_spec(self):
            return (
                {
                    "sources": [
                        {
                            "id": "x",
                            "role": "features",
                            "input": "X.csv",
                            "variations": [{"name": "snv", "input": "X_snv.csv"}],
                        },
                        {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}},
                    ],
                    "partitions": {"by": "index_file", "train_file": "train_idx.txt", "test_file": "test_idx.txt"},
                    "folds": {"file": "folds.csv", "format": "csv"},
                },
                tmp_path,
            )

    spec = nio.to_spec(ReferenceDatasetDouble(), name="reference")

    assert Path(spec.sources[0]["input"]).is_absolute()
    assert Path(spec.sources[0]["variations"][0]["input"]).is_absolute()
    assert Path(spec["partitions"]["train_file"]).is_absolute()
    assert Path(spec["partitions"]["test_file"]).is_absolute()
    assert Path(spec["folds"]["file"]).is_absolute()

    assembled = nio.load(ReferenceDatasetDouble(), target="assembled", name="reference")
    assert set(assembled["blocks"]) == {"train", "test"}


def test_load_parquet_reference_object_fails_with_actionable_binding_error(tmp_path):
    (tmp_path / "X.parquet").write_bytes(b"PAR1")

    class ReferenceDatasetDouble:
        def to_io_spec(self):
            return ({"sources": [{"id": "x", "role": "features", "input": "X.parquet"}]}, tmp_path)

    with pytest.raises(ValueError, match="Parquet inputs are not supported"):
        nio.load(ReferenceDatasetDouble(), target="assembled")

    with pytest.raises(ValueError, match="Parquet inputs are not supported"):
        nio.load([tmp_path / "X.parquet"], target="assembled")


def test_validate_rejects_invalid_typed_path():
    with pytest.raises(ValueError):
        nio.validate({"partitions": {"by": "random"}})


def test_public_surface_unchanged():
    # The historical names plus the new typed classes are all exported.
    for name in ("infer", "to_spec", "validate", "load", "to_spectrodataset", "__version__"):
        assert name in nio.__all__
    assert "DatasetPlan" in nio.__all__ and "DatasetSpec" in nio.__all__
