# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Behavioural gates of the Float64 owner contract on the Python oracle (A-J).

Exact F64 execution IR, source-dtype capture before widening, one quantization
point, explicit masks, target kinds and exactness, independent base identities,
strict full-wire readers and the SpectroDataset consumer policy (via a recording
double; the real SDK controls live in ``test_sdk_consumer.py``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.materialize import (
    DatasetPackage,
    ExactBlock,
    NumericContractError,
    NumericStorageOptions,
)
from nirs4all_io.materialize.assemble import AssembledDataset, PartitionBlock
from nirs4all_io.materialize.package import FeatureMatrix, TargetTable, WeightsVector
from nirs4all_io.materialize.projection import ExecutedProjectionIdentityV1
from nirs4all_io.materialize.spectrodataset import project_spectrodataset
from nirs4all_io.public_content import canonical_content_bytes


def _feature(package: DatasetPackage, partition: str = "train", index: int = 0) -> FeatureMatrix:
    payload = next(p for pid, p in package.partitions[partition].payloads if pid == f"{partition}/x{index}")
    assert isinstance(payload, FeatureMatrix)
    return payload


def _targets(package: DatasetPackage, partition: str = "train") -> TargetTable:
    payload = next(p for pid, p in package.partitions[partition].payloads if pid == f"{partition}/y")
    assert isinstance(payload, TargetTable)
    return payload


def _entry(package: DatasetPackage, payload_id: str):
    return next(e for e in package.manifest().entries if e.id == payload_id)


# --------------------------------------------------------------------------- #
# A/D: exact F64 IR, ULP collapse only on explicit F32                          #
# --------------------------------------------------------------------------- #
def test_loader_and_in_memory_ir_is_exact_float64() -> None:
    x = np.array([[1.0, 1.0 + 2.0**-30], [2.0, 3.0]])
    assembled = nio.load((x, np.arange(2.0)), target="assembled")
    block = assembled.blocks["train"]
    assert block.X[0].dtype == np.float64 and np.array_equal(block.X[0], x)
    assert block.X[0][0, 0] != block.X[0][0, 1], "a sub-F32-ulp difference must survive the IR"


def test_f64_ulp_difference_changes_content_and_requested_f32_collapses_it() -> None:
    near = np.array([[1.0, 1.0 + 2.0**-30]])
    exact = np.array([[1.0, 1.0]])
    y = np.array([0.0])
    f64_near, f64_exact = nio.to_dataset_package((near, y)), nio.to_dataset_package((exact, y))
    assert _entry(f64_near, "train/x0").content_hash != _entry(f64_exact, "train/x0").content_hash
    assert f64_near.manifest().root != f64_exact.manifest().root
    f32_near = nio.to_dataset_package((near, y), features="float32")
    f32_exact = nio.to_dataset_package((exact, y), features="float32")
    assert _entry(f32_near, "train/x0").content_hash == _entry(f32_exact, "train/x0").content_hash
    precision = _feature(f32_near).matrix.precision
    assert (precision.dtype, precision.origin, precision.requested_dtype, precision.source_dtype) == ("float32", "requested", "float32", "float64")


def test_summary_is_not_the_payload_certificate() -> None:
    a = nio.to_dataset_package((np.array([[1.0, 1.0 + 2.0**-30]]), np.array([0.0])))
    summary = a.to_summary_dict()
    assert "data_bytes_hex" not in canonical_json(summary)
    assert summary["schema_version"] == 4 and summary["manifest"]["entries"][0]["dtype"] == "float64"


# --------------------------------------------------------------------------- #
# Source dtype is captured before widening                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("array", "dtype", "origin", "source"),
    [
        (np.array([[0.5, 1.5]], dtype=np.float32), "float32", "source-declared", "float32"),
        (np.array([[0.5, 1.5]], dtype=np.float64), "float64", "source-declared", "float64"),
        (np.array([[1, 2]], dtype=np.int64), "float64", "default", "int64"),
        (np.array([[1, 2]], dtype=np.uint8), "float64", "default", "uint8"),
    ],
)
def test_typed_ndarray_declares_its_exact_dtype(array: np.ndarray, dtype: str, origin: str, source: str) -> None:
    precision = _feature(nio.to_dataset_package((array, np.array([0.0])))).matrix.precision
    assert (precision.dtype, precision.origin, precision.source_dtype, precision.requested_dtype, precision.consumer) == (dtype, origin, source, None, None)


def test_untyped_list_and_csv_declare_nothing(tmp_path) -> None:
    listed = nio.to_dataset_package({"X": [[0.25, 0.5]], "y": np.array([1.0])})
    precision = _feature(listed).matrix.precision
    assert (precision.origin, precision.source_dtype) == ("default", None)
    (tmp_path / "d.csv").write_text("a;b;y\n0.25;0.5;1\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": ["a", "b"]}, {"role": "targets", "select": ["y"]}], "params": {"delimiter": ";"}}]}
    csv_precision = _feature(nio.to_dataset_package(spec, base_dir=tmp_path)).matrix.precision
    assert (csv_precision.dtype, csv_precision.origin, csv_precision.source_dtype) == ("float64", "default", None)


def test_declared_f32_source_is_not_recovered_by_a_later_f64_request() -> None:
    value = np.array([[0.1]], dtype=np.float32)
    kept = nio.to_dataset_package((value, np.array([0.0])))
    widened = nio.to_dataset_package((value, np.array([0.0])), features="float64")
    assert _feature(kept).matrix.dtype == "float32"
    block = _feature(widened).matrix
    assert (block.dtype, block.precision.origin, block.precision.source_dtype) == ("float64", "requested", "float32")
    assert block.values[0, 0] == np.float64(np.float32(0.1)) != 0.1, "widening keeps the normalized source value"


def test_typed_parquet_and_npy_declare_but_csv_text_does_not(tmp_path) -> None:
    frame = pd.DataFrame({"a": np.array([0.5], dtype=np.float32), "b": np.array([1.5], dtype=np.float32), "y": np.array([3], dtype=np.int64)})
    frame.to_parquet(tmp_path / "d.parquet")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.parquet", "columns": [{"role": "features", "select": ["a", "b"]}, {"role": "targets", "select": ["y"]}]}]}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert _feature(package).matrix.precision.source_dtype == "float32"
    assert _feature(package).matrix.dtype == "float32"
    assert _targets(package).columns[0].block.precision.source_dtype == "int64"
    assert _targets(package).columns[0].block.dtype == "float64"
    np.save(tmp_path / "x.npy", np.array([[0.5, 1.5]], dtype=np.float32))
    npy = nio.to_dataset_package(
        {"sources": [{"id": "x", "role": "features", "input": "x.npy", "params": {"header_unit": "index"}}]}, base_dir=tmp_path
    )
    assert _feature(npy).matrix.precision.source_dtype == "float32"


def test_mixed_declared_dtypes_refuse_instead_of_guessing(tmp_path) -> None:
    pd.DataFrame({"a": np.array([0.5], dtype=np.float32), "b": np.array([1.5], dtype=np.float64)}).to_parquet(tmp_path / "m.parquet")
    spec = {"sources": [{"id": "m", "role": "features", "input": "m.parquet"}]}
    with pytest.raises(NumericContractError, match="source dtype"):
        nio.to_dataset_package(spec, base_dir=tmp_path)


def test_unsupported_source_dtypes_refuse() -> None:
    for array in (np.array([[True]]), np.array([[0.5]], dtype=np.float16)):
        with pytest.raises(NumericContractError, match="source dtype"):
            nio.to_dataset_package((array, np.array([0.0])))


# --------------------------------------------------------------------------- #
# D: nearest-even, subnormal, overflow, underflow, signed zero                  #
# --------------------------------------------------------------------------- #
def test_f32_overflow_underflow_and_subnormals() -> None:
    y = np.array([0.0])
    for bad in (1e40, 1e-50):
        with pytest.raises(NumericContractError):
            nio.to_dataset_package((np.array([[bad]]), y), features="float32")
    subnormal = nio.to_dataset_package((np.array([[1e-40]]), y), features="float32")
    value = _feature(subnormal).matrix.values[0, 0]
    assert value != 0 and abs(float(value)) < np.finfo(np.float32).tiny


def test_declared_f32_normalization_happens_once_and_serialization_is_idempotent() -> None:
    package = nio.to_dataset_package((np.array([[0.1]], dtype=np.float32), np.array([0.0])))
    wire = package.to_full_dict()
    again = DatasetPackage.from_full_dict(json.loads(canonical_json(wire)))
    assert again.manifest().to_dict() == package.manifest().to_dict()
    assert _feature(again).matrix.to_bytes() == _feature(package).matrix.to_bytes()


def test_observed_signed_zero_is_preserved_and_distinct() -> None:
    y = np.array([0.0])
    negative = nio.to_dataset_package((np.array([[-0.0]]), y))
    positive = nio.to_dataset_package((np.array([[0.0]]), y))
    assert _feature(negative).matrix.values_bytes() == bytes.fromhex("0000000000000080")
    assert _entry(negative, "train/x0").content_hash != _entry(positive, "train/x0").content_hash
    # the OLD whole-dataset tagged-binary64 profile still folds zero (unchanged contract)
    assert canonical_content_bytes([-0.0]) == canonical_content_bytes([0.0])
    f32 = nio.to_dataset_package((np.array([[-0.0]]), y), features="float32")
    assert _feature(f32).matrix.values_bytes() == bytes.fromhex("00000080")


# --------------------------------------------------------------------------- #
# C: explicit masks                                                             #
# --------------------------------------------------------------------------- #
def test_missing_is_an_explicit_false_mask_never_an_observed_zero() -> None:
    y = np.array([0.0, 1.0])
    missing = nio.to_dataset_package((np.array([[np.nan], [2.0]]), y))
    zero = nio.to_dataset_package((np.array([[0.0], [2.0]]), y))
    block = _feature(missing).matrix
    assert block.validity.ravel().tolist() == [False, True] and block.values_bytes()[:8] == b"\x00" * 8
    assert _entry(missing, "train/x0").content_hash != _entry(zero, "train/x0").content_hash
    wire = missing.to_full_dict()["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"]
    assert wire["validity"] == [False, True]
    restored = missing.to_assembled().blocks["train"].X[0]
    assert np.isnan(restored[0, 0]) and restored[1, 0] == 2.0


def test_missing_target_is_never_observed_zero() -> None:
    x = np.array([[1.0], [2.0]])
    missing = nio.to_dataset_package((x, np.array([np.nan, 1.0])))
    zero = nio.to_dataset_package((x, np.array([0.0, 1.0])))
    assert _targets(missing).columns[0].block.validity.ravel().tolist() == [False, True]
    assert _entry(missing, "train/y").content_hash != _entry(zero, "train/y").content_hash
    y_ir = missing.to_assembled().blocks["train"].y
    assert y_ir is not None and np.isnan(y_ir[0, 0])


def test_infinity_refuses_at_the_package_boundary() -> None:
    with pytest.raises(NumericContractError, match="finite"):
        nio.to_dataset_package((np.array([[np.inf]]), np.array([0.0])))


def test_every_block_carries_a_mask_even_when_all_true() -> None:
    package = nio.to_dataset_package((np.array([[1.0]]), np.array([1.0])), weights=None)
    wire = package.to_full_dict()["partitions"]["train"]["payloads"]
    assert wire["train/x0"]["value"]["matrix"]["validity"] == [True]
    assert wire["train/y"]["value"]["columns"][0]["block"]["validity"] == [True]


# --------------------------------------------------------------------------- #
# E: target kinds, caps, exact labels, unsupported types                        #
# --------------------------------------------------------------------------- #
def _classification(labels: list[int], dtype=np.int64, task="multiclass") -> AssembledDataset:
    assembled = AssembledDataset(name="c", task_type=task, signal_type="auto", n_sources=1)
    block = PartitionBlock(n_samples=len(labels), source_ids=["X"], X=[np.zeros((len(labels), 1))], feature_headers=[["0"]], header_units=["index"], signal_types=[None])
    block.y, block.y_headers, block.y_source_dtype = np.array(labels, dtype=dtype).reshape(-1, 1), ["y"], np.dtype(dtype).name
    assembled.blocks["train"] = block
    return assembled


def test_f32_label_exactness_uses_the_integer_domain() -> None:
    ok = DatasetPackage.from_assembled(_classification([16777218, 16777216]), targets="float32")
    assert _targets(ok).columns[0].kind == "classification_integer"
    assert _targets(ok).columns[0].block.dtype == "float32"
    with pytest.raises(NumericContractError, match="not exact in float32"):
        DatasetPackage.from_assembled(_classification([16777217, 16777216]), targets="float32")
    exact_f64 = DatasetPackage.from_assembled(_classification([16777217, 2**53 - 1]))
    assert _targets(exact_f64).columns[0].block.values.ravel().tolist() == [16777217.0, float(2**53 - 1)]
    for bad in (2**53, -(2**53)):
        with pytest.raises(NumericContractError, match="portable integer"):
            DatasetPackage.from_assembled(_classification([bad, 0]))
    with pytest.raises(NumericContractError, match="not exact in float32"):
        DatasetPackage.from_assembled(_classification([2**53 - 1, 0]), targets="float32")


def test_generic_integers_keep_every_exact_f64_value_and_refuse_only_real_loss(tmp_path) -> None:
    """The +/-(2**53 - 1) cap is a classification-label rule, not a numeric-integer rule."""
    for value in (2**53, 2**54, -(2**54), 2**53 - 1):
        package = DatasetPackage.from_assembled(_classification([value, 0], task="regression"))
        assert _targets(package).columns[0].kind == "numeric"
        assert _targets(package).columns[0].block.values[0, 0] == float(value)
        arrays = nio.to_dataset_package((np.array([[value]], dtype=np.int64), np.array([0.0])))
        assert _feature(arrays).matrix.values[0, 0] == float(value) and _feature(arrays).matrix.precision.source_dtype == "int64"
    for value in (2**53 + 1, 2**63 - 1):
        with pytest.raises(NumericContractError, match="cannot represent exactly"):
            DatasetPackage.from_assembled(_classification([value, 0], task="regression"))
        with pytest.raises(Exception, match="cannot represent exactly"):
            nio.load((np.array([[value]], dtype=np.int64), np.array([0.0])), target="assembled")
    (tmp_path / "d.csv").write_text(f"a;y\n{2**54};1\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}], "params": {"delimiter": ";"}}]}
    assert nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"].X[0][0, 0] == float(2**54)
    with pytest.raises(NumericContractError, match="portable integer"):
        DatasetPackage.from_assembled(_classification([2**53, 0], task="binary"))


def test_classification_is_a_distinct_kind_from_numeric_with_identical_values() -> None:
    numeric = DatasetPackage.from_assembled(_classification([1, 2], task="regression"))
    classification = DatasetPackage.from_assembled(_classification([1, 2], task="multiclass"))
    assert _targets(numeric).columns[0].kind == "numeric"
    assert _entry(numeric, "train/y").content_hash != _entry(classification, "train/y").content_hash


def test_non_integer_classification_labels_refuse() -> None:
    assembled = _classification([1, 2], dtype=np.float64)
    assembled.blocks["train"].y = np.array([[0.5], [1.0]])
    with pytest.raises(NumericContractError, match="integer labels"):
        DatasetPackage.from_assembled(assembled)


def test_unsupported_target_dtypes_refuse_without_reencoding() -> None:
    for y in (np.array([True, False]), np.array(["a", "b"]), np.array([1, "a"], dtype=object)):
        with pytest.raises(NumericContractError, match="Unsupported target dtype"):
            nio.to_dataset_package((np.zeros((2, 1)), y))


def test_string_targets_use_the_existing_codebook_and_refuse_numeric_requests(tmp_path) -> None:
    (tmp_path / "d.csv").write_text("a;y\n1;red\n2;blue\n3;red\n4;\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}], "params": {"delimiter": ";"}}]}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    column = _targets(package).columns[0]
    assert column.kind == "classification_string" and column.block.dtype == "string-label"
    assert column.block.values.ravel().tolist() == ["red", "blue", "red", ""]
    assert column.block.validity.ravel().tolist() == [True, True, True, False]
    assert (column.block.precision.origin, column.block.precision.source_dtype, column.block.precision.requested_dtype) == ("default", None, None)
    restored = package.to_assembled().blocks["train"]
    assert restored.y is not None and restored.y[:3, 0].tolist() == [0.0, 1.0, 0.0] and np.isnan(restored.y[3, 0])
    assert restored.y_categorical == {"y": {"categories": ["red", "blue"]}}
    with pytest.raises(NumericContractError, match="string targets"):
        nio.to_dataset_package(spec, base_dir=tmp_path, targets="float32")


def test_integer_target_beyond_the_cap_refuses_before_widening(tmp_path) -> None:
    (tmp_path / "d.csv").write_text(f"a;y\n1;{2**53 + 1}\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}], "params": {"delimiter": ";"}}]}
    with pytest.raises(Exception, match="cannot represent exactly"):
        nio.load(spec, target="assembled", base_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Weights and named variations                                                  #
# --------------------------------------------------------------------------- #
def _weights_variations_spec(tmp_path, *, weight=0.1, variation=1.0 + 2.0**-30) -> dict:
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [5.0, 6.0], "w": [weight, 2.0]}).to_csv(tmp_path / "x.csv", sep=";", index=False)
    pd.DataFrame({"a": [variation, 2.0], "b": [3.0, 4.0]}).to_csv(tmp_path / "snv.csv", sep=";", index=False)
    return {
        "sources": [
            {
                "id": "x",
                "role": "mixed",
                "input": "x.csv",
                "columns": [{"role": "features", "select": ["a", "b"]}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}],
                "variations": [{"name": "snv", "input": "snv.csv"}],
                "params": {"delimiter": ";"},
            }
        ]
    }


def test_weights_and_variations_stay_exact_f64(tmp_path) -> None:
    spec = _weights_variations_spec(tmp_path)
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    block = assembled.blocks["train"]
    assert block.weights.dtype == np.float64 and block.weights[0] == 0.1
    assert block.processings[0][0][1].dtype == np.float64 and block.processings[0][0][1][0, 0] == 1.0 + 2.0**-30
    package = DatasetPackage.from_assembled(assembled)
    weights = next(p for pid, p in package.partitions["train"].payloads if pid == "train/weights")
    assert isinstance(weights, WeightsVector) and weights.matrix.dtype == "float64" and weights.matrix.values[0, 0] == 0.1
    procs = _feature(package).processings
    assert procs[0][0] == "snv" and procs[0][1].dtype == "float64"
    near = package.manifest().entries
    other = DatasetPackage.from_assembled(nio.load(_weights_variations_spec(tmp_path, variation=1.0), target="assembled", base_dir=tmp_path))
    assert _entry(other, "train/x0").content_hash != next(e for e in near if e.id == "train/x0").content_hash, "a processing ULP must bind the feature payload"
    f32 = DatasetPackage.from_assembled(assembled, features="float32", weights="float32")
    assert _feature(f32).processings[0][1].dtype == "float32"
    f32_weights = next(p for pid, p in f32.partitions["train"].payloads if pid == "train/weights")
    assert isinstance(f32_weights, WeightsVector) and f32_weights.matrix.dtype == "float32"


# --------------------------------------------------------------------------- #
# F-I: independent base identities                                              #
# --------------------------------------------------------------------------- #
def _identity_spec(tmp_path, *, y=(1.0, 2.0, 3.0, 4.0), unit="nm", value=0.5, folds=True, group=("g1", "g1", "g2", "g2")) -> dict:
    pd.DataFrame({"sid": ["s1", "s2", "s3", "s4"], "grp": list(group), "1100": [value, 0.25, 0.75, 1.0], "1102": [0.5, 0.5, 0.5, 0.5], "y": list(y)}).to_csv(tmp_path / "d.csv", sep=";", index=False)
    spec = {
        "sample_index": {"by": "id", "key": "sid", "group_id": "grp"},
        "sources": [
            {
                "id": "d",
                "role": "mixed",
                "input": "d.csv",
                "key": "sid",
                "columns": [{"role": "features", "select": ["1100", "1102"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["grp"]}],
                "params": {"delimiter": ";", "header_unit": unit},
            }
        ],
    }
    if folds:
        spec["folds"] = {"inline": [{"train": [0, 1], "val": [2, 3]}]}
    return spec


def _projection(spec, tmp_path, **options):
    package = nio.to_dataset_package(spec, base_dir=tmp_path, **options)
    return package, package.executed_projection(["d"]).identity


def test_feature_and_target_digests_are_independent(tmp_path) -> None:
    _, base = _projection(_identity_spec(tmp_path), tmp_path)
    _, target_only = _projection(_identity_spec(tmp_path, y=(9.0, 2.0, 3.0, 4.0)), tmp_path)
    assert target_only.feature_content_fingerprint == base.feature_content_fingerprint
    assert target_only.target_content_fingerprint != base.target_content_fingerprint
    _, feature_only = _projection(_identity_spec(tmp_path, value=0.5 + 2.0**-40), tmp_path)
    assert feature_only.target_content_fingerprint == base.target_content_fingerprint
    assert feature_only.feature_content_fingerprint != base.feature_content_fingerprint


def test_raw_units_folds_and_ids_bind_both_sides(tmp_path) -> None:
    _, base = _projection(_identity_spec(tmp_path), tmp_path)
    for label, changed in (
        ("unit", {"unit": "cm-1"}),
        ("folds", {"folds": False}),
        ("groups", {"group": ("g1", "g1", "g3", "g2")}),
    ):
        spec = _identity_spec(tmp_path, **changed)  # written lazily: every case rewrites d.csv
        _, other = _projection(spec, tmp_path)
        if label == "unit":
            assert other.feature_content_fingerprint != base.feature_content_fingerprint
            assert other.target_content_fingerprint == base.target_content_fingerprint, "the unit is a feature-axis fact"
        else:
            assert other.feature_content_fingerprint != base.feature_content_fingerprint, label
            assert other.target_content_fingerprint != base.target_content_fingerprint, label


def test_requested_precision_and_masks_bind_the_digests(tmp_path) -> None:
    spec = _identity_spec(tmp_path)
    _, base = _projection(spec, tmp_path)
    _, f32 = _projection(spec, tmp_path, features="float32")
    assert f32.feature_content_fingerprint != base.feature_content_fingerprint
    assert f32.target_content_fingerprint == base.target_content_fingerprint
    _, t32 = _projection(spec, tmp_path, targets="float32")
    assert t32.target_content_fingerprint != base.target_content_fingerprint
    assert t32.feature_content_fingerprint == base.feature_content_fingerprint
    nan_spec = _identity_spec(tmp_path, y=(float("nan"), 2.0, 3.0, 4.0))
    _, missing = _projection(nan_spec, tmp_path)
    _, zero = _projection(_identity_spec(tmp_path, y=(0.0, 2.0, 3.0, 4.0)), tmp_path)
    assert missing.target_content_fingerprint != zero.target_content_fingerprint


def test_receipt_shape_validation_and_domain_separation(tmp_path) -> None:
    package, receipt = _projection(_identity_spec(tmp_path), tmp_path)
    value = receipt.to_dict()
    assert list(value) == [
        "profile", "version", "scope", "algorithm", "feature_content_fingerprint", "target_content_fingerprint", "selected_source_ids", "sample_ids",
        "observation_ids", "target_names", "target_kinds", "features", "targets", "mask_policy", "package_root", "dataset_content_identity",
    ]  # fmt: skip
    assert (value["profile"], value["version"], value["scope"], value["algorithm"], value["mask_policy"]) == ("nirs4all.io.executed-projection.v1", 1, "base", "sha256", "explicit-mask-positive-zero.v1")
    assert value["package_root"] == package.manifest().root
    assert value["feature_content_fingerprint"] != value["package_root"] != value["target_content_fingerprint"]
    assert value["sample_ids"] == ["s1", "s2", "s3", "s4"] and value["target_kinds"] == ["numeric"] and value["dataset_content_identity"] is None
    assert value["features"][0]["delivered_dtype"] == "float64" and value["features"][0]["axes"][0]["unit"] == "nm"
    ExecutedProjectionIdentityV1.from_dict(json.loads(json.dumps(value))).validate(package)
    tampered = dict(value, feature_content_fingerprint="0" * 64)
    with pytest.raises(NumericContractError, match="disagrees"):
        ExecutedProjectionIdentityV1.from_dict(tampered).validate(package)
    for key, bad in (("version", 2), ("scope", "fold"), ("profile", "other"), ("mask_policy", "x")):
        with pytest.raises(NumericContractError, match="Unsupported"):
            ExecutedProjectionIdentityV1.from_dict({**value, key: bad})
    with pytest.raises(NumericContractError, match="missing or unknown"):
        ExecutedProjectionIdentityV1.from_dict({k: v for k, v in value.items() if k != "targets"})


def test_projection_refuses_unsupported_cohorts(tmp_path) -> None:
    package, _ = _projection(_identity_spec(tmp_path), tmp_path)
    for selected in ([], ["d", "d"], ["d", "e"]):
        with pytest.raises(NumericContractError):
            package.executed_projection(selected)
    with pytest.raises(NumericContractError, match="source missing"):
        package.executed_projection(["nope"])
    leak = _identity_spec(tmp_path, group=("g1", "g2", "g1", "g2"))
    with pytest.raises(NumericContractError, match="group leakage"):
        nio.to_dataset_package(leak, base_dir=tmp_path).executed_projection(["d"])
    arrays = nio.to_dataset_package((np.zeros((2, 1)), np.zeros(2)))
    with pytest.raises(NumericContractError, match="sample identity"):
        arrays.executed_projection(["X"])


def test_primary_ids_are_never_sanitized(tmp_path) -> None:
    spec = _identity_spec(tmp_path)
    frame = pd.read_csv(tmp_path / "d.csv", sep=";")
    frame["sid"] = ["s 1", "s2", "s3", "s4"]
    frame.to_csv(tmp_path / "d.csv", sep=";", index=False)
    with pytest.raises(NumericContractError, match="Invalid primary identity"):
        nio.to_dataset_package(spec, base_dir=tmp_path).executed_projection(["d"])


def test_old_whole_dataset_and_x_only_identities_are_untouched() -> None:
    """The new base digests are separate domains; the old tagged-binary64 bytes still fold zero."""
    assert canonical_content_bytes({"v": -0.0}) == canonical_content_bytes({"v": 0.0})
    pos = ExactBlock.from_matrix(np.array([[0.0]]), source_dtype=None, request=None, consumer=None, feature=True)
    neg = ExactBlock.from_matrix(np.array([[-0.0]]), source_dtype=None, request=None, consumer=None, feature=True)
    assert pos.to_bytes() != neg.to_bytes()


# --------------------------------------------------------------------------- #
# Strict versions and full-wire roundtrips                                      #
# --------------------------------------------------------------------------- #
def _roundtrip(wire: dict) -> dict:
    return json.loads(canonical_json(wire))  # canonical JSON sorts every object's keys


def _rich_package(tmp_path) -> DatasetPackage:
    spec = _weights_variations_spec(tmp_path)
    frame = pd.read_csv(tmp_path / "x.csv", sep=";")
    frame.loc[1, "a"] = np.nan
    frame.to_csv(tmp_path / "x.csv", sep=";", index=False)
    return nio.to_dataset_package({**spec, "params": {"na": {"policy": "ignore"}}}, base_dir=tmp_path)


def test_package_full_roundtrip_is_exact(tmp_path) -> None:
    package = _rich_package(tmp_path)
    assert np.isnan(package.to_assembled().blocks["train"].X[0][1, 0])
    restored = DatasetPackage.from_full_dict(_roundtrip(package.to_full_dict()))
    assert restored.manifest().to_dict() == package.manifest().to_dict()
    assert restored.to_full_dict() == package.to_full_dict()
    for before, after in zip(package.to_assembled().blocks["train"].X, restored.to_assembled().blocks["train"].X, strict=True):
        assert np.array_equal(before, after, equal_nan=True)


def test_assembled_full_roundtrip_is_exact(tmp_path) -> None:
    assembled = nio.load(_weights_variations_spec(tmp_path), target="assembled", base_dir=tmp_path)
    wire = assembled.to_full_dict()
    assert wire["assembled_schema_version"] == 3 and wire["partition_order"] == ["train"]
    block = wire["blocks"]["train"]
    assert set(block["x"][0]) == {"dtype", "n_rows", "n_cols", "data_bytes_hex", "validity", "precision"}
    assert set(block["y"]["columns"][0]) == {"name", "kind", "block"}
    restored = AssembledDataset.from_full_dict(_roundtrip(wire))
    assert restored.to_full_dict() == wire
    assert np.array_equal(restored.blocks["train"].X[0], assembled.blocks["train"].X[0])
    assert np.array_equal(np.asarray(restored.blocks["train"].weights), np.asarray(assembled.blocks["train"].weights))


def _ordered_assembled() -> AssembledDataset:
    assembled = AssembledDataset(name="order", task_type="regression", signal_type="auto", n_sources=1)
    for index, name in enumerate(("train", "predict", "test")):  # deliberately not alphabetical
        block = PartitionBlock(n_samples=1, source_ids=["X"], X=[np.array([[float(index)]])], feature_headers=[["0"]], header_units=["index"], signal_types=[None])
        block.y, block.y_headers = np.array([[float(index)]]), ["y"]
        assembled.blocks[name] = block
    return assembled


def test_partition_order_survives_canonical_key_sorting_and_pins_the_bytes() -> None:
    assembled = _ordered_assembled()
    wire = assembled.to_full_dict()
    assert wire["partition_order"] == ["train", "predict", "test"]
    canonical = canonical_json(wire)
    assert list(json.loads(canonical)["blocks"]) == ["predict", "test", "train"], "canonical JSON sorts keys"
    restored = AssembledDataset.from_full_dict(json.loads(canonical))
    assert list(restored.blocks) == ["train", "predict", "test"]
    package = DatasetPackage.from_assembled(assembled)
    package_wire = package.to_full_dict()
    again = DatasetPackage.from_full_dict(json.loads(canonical_json(package_wire)))
    assert list(again.partitions) == ["train", "predict", "test"] and again.manifest().to_dict() == package.manifest().to_dict()
    # byte fingerprints of the exact full wires (regression pins for the shared goldens)
    assert hashlib.sha256(canonical.encode()).hexdigest() == "73a337cf2a44ed5dcc61574b196cccc290c27517335b854aaed140d1574f73a6"
    assert hashlib.sha256(canonical_json(package_wire).encode()).hexdigest() == "3edf1ae3b3d6b12e9db05d15767767f7c33ad787a22f26d6bf3e3dccae0c9e05"


def test_partition_order_must_be_an_exact_unique_cover() -> None:
    wire = _roundtrip(DatasetPackage.from_assembled(_ordered_assembled()).to_full_dict())
    for bad in (["train", "predict"], ["train", "predict", "test", "test"], ["train", "predict", "other"], "train"):
        with pytest.raises(NumericContractError, match="partition_order"):
            DatasetPackage.from_full_dict({**wire, "partition_order": bad})
    swapped = {**wire, "partition_order": ["predict", "train", "test"]}
    with pytest.raises(NumericContractError, match="Manifest partition order"):
        DatasetPackage.from_full_dict(swapped)
    awire = _roundtrip(_ordered_assembled().to_full_dict())
    with pytest.raises(NumericContractError, match="partition_order"):
        AssembledDataset.from_full_dict({**awire, "partition_order": ["train"]})
    del awire["partition_order"]
    with pytest.raises(NumericContractError, match="missing or unknown"):
        AssembledDataset.from_full_dict(awire)


def test_readers_accept_only_the_exact_current_versions(tmp_path) -> None:
    package = _rich_package(tmp_path)
    wire = _roundtrip(package.to_full_dict())
    for version in (3, 2, 5, "4", True):
        with pytest.raises(NumericContractError, match="Unsupported package version"):
            DatasetPackage.from_full_dict({**wire, "schema_version": version})
    with pytest.raises(NumericContractError, match="missing or unknown"):
        DatasetPackage.from_full_dict({**wire, "extra": 1})
    awire = _roundtrip(nio.load(_weights_variations_spec(tmp_path), target="assembled", base_dir=tmp_path).to_full_dict())
    for version in (2, 4, "3"):
        with pytest.raises(NumericContractError, match="Unsupported assembled version"):
            AssembledDataset.from_full_dict({**awire, "assembled_schema_version": version})


def _first_matrix(wire: dict) -> dict:
    return wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"]


def test_block_reader_refuses_malformed_bytes_masks_dtypes_and_precision(tmp_path) -> None:
    wire = _roundtrip(_rich_package(tmp_path).to_full_dict())
    cases = {
        "truncated": lambda m: m.update(data_bytes_hex=m["data_bytes_hex"][:-2]),
        "extra": lambda m: m.update(data_bytes_hex=m["data_bytes_hex"] + "00"),
        "uppercase": lambda m: m.update(data_bytes_hex=m["data_bytes_hex"].upper().replace("E", "e").replace("A", "a") + "F"),
        "odd": lambda m: m.update(data_bytes_hex=m["data_bytes_hex"] + "0"),
        "nonbool mask": lambda m: m["validity"].__setitem__(0, 1),
        "short mask": lambda m: m.update(validity=m["validity"][:-1]),
        "dtype mismatch": lambda m: m.update(dtype="float32"),
        "shape mismatch": lambda m: m.update(n_rows=m["n_rows"] + 1),
        "unknown field": lambda m: m.update(values=[1]),
        "missing precision field": lambda m: m["precision"].pop("consumer"),
        "origin lies": lambda m: m["precision"].update(origin="requested"),
        "false cell not +0": lambda m: m.update(data_bytes_hex="0000000000000080" + m["data_bytes_hex"][16:]) if not m["validity"][0] else m.update(data_bytes_hex=m["data_bytes_hex"][:16] + "0000000000000080" + m["data_bytes_hex"][32:]),
        "nonfinite": lambda m: m.update(data_bytes_hex="000000000000f07f" + m["data_bytes_hex"][16:]),
    }  # fmt: skip
    for label, mutate in cases.items():
        broken = json.loads(json.dumps(wire))
        mutate(_first_matrix(broken))
        with pytest.raises(NumericContractError):
            DatasetPackage.from_full_dict(broken)
        assert label


def test_flipped_payload_byte_is_caught_by_the_manifest(tmp_path) -> None:
    package = nio.to_dataset_package((np.array([[1.0, 2.0]]), np.array([1.0])))
    wire = _roundtrip(package.to_full_dict())
    matrix = wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"]
    first = int(matrix["data_bytes_hex"][:2], 16) ^ 1
    matrix["data_bytes_hex"] = f"{first:02x}" + matrix["data_bytes_hex"][2:]
    with pytest.raises(NumericContractError, match="manifest disagrees"):
        DatasetPackage.from_full_dict(wire)


def test_target_kind_axis_and_order_are_strict_in_the_wire(tmp_path) -> None:
    wire = _roundtrip(_rich_package(tmp_path).to_full_dict())
    bad_kind = json.loads(json.dumps(wire))
    bad_kind["partitions"]["train"]["payloads"]["train/y"]["value"]["columns"][0]["kind"] = "classification_string"
    with pytest.raises(NumericContractError, match="kind disagrees"):
        DatasetPackage.from_full_dict(bad_kind)
    axis_wire = _roundtrip(nio.to_dataset_package(_identity_spec(tmp_path), base_dir=tmp_path).to_full_dict())
    value = axis_wire["partitions"]["train"]["payloads"]["train/x0"]["value"]
    assert set(value["axes"][0]) == {"name", "unit", "coordinates"} and value["axes"][0]["coordinates"]["kind"] == "numeric"
    for coordinates in ({"kind": "numeric", "count": 3, "data_bytes_hex": "00" * 24}, {"kind": "string", "values": ["a"]}, {"kind": "other"}):
        broken = json.loads(json.dumps(axis_wire))
        broken["partitions"]["train"]["payloads"]["train/x0"]["value"]["axes"][0]["coordinates"] = coordinates
        with pytest.raises(NumericContractError):
            DatasetPackage.from_full_dict(broken)  # wrong extent / kind must not roundtrip silently
    assert wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["axes"][0]["coordinates"] is None, "non-numeric headers declare no coordinates"


def test_numeric_axis_coordinates_are_exact_f64_bits(tmp_path) -> None:
    spec = _identity_spec(tmp_path)
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    axis = _feature(package).axes[0]
    assert axis["coordinates"] == [1100.0, 1102.0] and axis["unit"] == "nm" and axis["name"] == "wavelength"
    wire = package.to_full_dict()["partitions"]["train"]["payloads"]["train/x0"]["value"]["axes"][0]
    assert wire["coordinates"] == {"kind": "numeric", "count": 2, "data_bytes_hex": np.array([1100.0, 1102.0], "<f8").tobytes().hex()}


def test_built_package_cannot_be_rerequested_and_references_are_preserved() -> None:
    package = nio.to_dataset_package((np.zeros((1, 1)), np.zeros(1)))
    assert nio.to_dataset_package(package) is package
    assert nio.load(package, target="dataset_package") is package
    with pytest.raises(Exception, match="already fixed"):
        nio.to_dataset_package(package, features="float32")


def test_options_validate_roles_and_consumers() -> None:
    assert NumericStorageOptions() == NumericStorageOptions(None, None, None, None)
    for bad in ({"features": "float16"}, {"targets": 32}, {"consumer": "Other"}):
        with pytest.raises(NumericContractError):
            NumericStorageOptions(**bad)


# --------------------------------------------------------------------------- #
# B: SpectroDataset consumer policy (recording double)                          #
# --------------------------------------------------------------------------- #
class _Double:
    def __init__(self, name="ds"):
        self.samples, self.targets, self.metadata, self.features_calls = [], [], [], []

    def add_samples(self, data, indexes, headers=None, header_unit=None):
        self.samples.append((indexes.get("partition"), data))

    def add_features(self, features, processings, source=-1):
        self.features_calls.append((np.asarray(features[0]), list(processings)))

    def add_targets(self, y):
        self.targets.append(y)

    def add_metadata(self, data, headers=None):
        self.metadata.append(data)

    def set_signal_type(self, *a, **k): ...
    def set_task_type(self, *a, **k): ...
    def set_folds(self, *a, **k): ...
    def set_repetition(self, *a, **k): ...


def test_consumer_default_projects_features_to_f32_with_a_consumer_receipt() -> None:
    assembled = nio.load((np.array([[1.0, 1.0 + 2.0**-30]]), np.array([1.0 + 2.0**-30])), target="assembled")
    ds, receipt = project_spectrodataset(assembled, spectro_dataset_cls=_Double)
    (_, data), = ds.samples
    assert data.dtype == np.float32 and data[0, 0] == data[0, 1]
    assert receipt["features"][0]["precision"] == {"dtype": "float32", "origin": "consumer", "source_dtype": "float64", "requested_dtype": None, "consumer": "SpectroDataset"}
    assert ds.targets[0].dtype == np.float64 and ds.targets[0][0, 0] == 1.0 + 2.0**-30, "raw regression targets are preserved"
    assert receipt["numerical_target_view"] is None and "SDK" in receipt["numerical_target_view_declined"]


def test_consumer_keeps_raw_integer_and_string_targets() -> None:
    ints = nio.load((np.zeros((2, 1)), np.array([16777217, 16777218])), target="assembled")
    ds, _ = project_spectrodataset(ints, spectro_dataset_cls=_Double)
    assert ds.targets[0].dtype == np.int64 and ds.targets[0].ravel().tolist() == [16777217, 16777218]


def test_consumer_refuses_f64_feature_and_f64_target_requests_before_the_sdk() -> None:
    assembled = nio.load((np.zeros((1, 1)), np.zeros(1)), target="assembled")
    for options in ({"features": "float64"}, {"targets": "float64"}):
        with pytest.raises(NumericContractError):
            project_spectrodataset(assembled, **options)  # type: ignore[arg-type]  # no injected class: the SDK is never imported


def test_consumer_f32_target_request_obeys_owner_exactness() -> None:
    ok = _classification([16777218, 16777216])
    ds, receipt = project_spectrodataset(ok, spectro_dataset_cls=_Double, targets="float32")
    assert ds.targets[0].dtype == np.float32 and ds.targets[0].ravel().tolist() == [16777218.0, 16777216.0]
    assert receipt["targets"]["precision"][0]["requested_dtype"] == "float32"
    bad = _classification([16777217, 16777216])
    with pytest.raises(NumericContractError, match="not exact in float32"):
        project_spectrodataset(bad, spectro_dataset_cls=_Double, targets="float32")
    regression = _classification([16777217, 16777216], task="regression")  # numeric targets may round: no label rule
    ds, _ = project_spectrodataset(regression, spectro_dataset_cls=_Double, targets="float32")
    assert ds.targets[0].ravel().tolist() == [16777216.0, 16777216.0]
    with pytest.raises(NumericContractError):
        project_spectrodataset(_classification([2**53 - 1, 0]), spectro_dataset_cls=_Double, targets="float32")


def test_consumer_string_targets_refuse_numeric_requests_and_missing_stays_nan(tmp_path) -> None:
    (tmp_path / "d.csv").write_text("a;y\n1;red\n2;blue\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}], "params": {"delimiter": ";"}}]}
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    ds, _ = project_spectrodataset(assembled, spectro_dataset_cls=_Double)
    assert ds.targets[0].ravel().tolist() == ["red", "blue"]
    with pytest.raises(NumericContractError, match="string"):
        project_spectrodataset(assembled, spectro_dataset_cls=_Double, targets="float32")
    nan = nio.load((np.array([[np.nan], [1.0]]), np.zeros(2)), target="assembled")
    ds, _ = project_spectrodataset(nan, spectro_dataset_cls=_Double)
    assert np.isnan(ds.samples[0][1][0, 0]), "missing X stays NaN: transported, never imputed"


def test_consumer_processings_follow_the_feature_consumer_policy(tmp_path) -> None:
    assembled = nio.load(_weights_variations_spec(tmp_path), target="assembled", base_dir=tmp_path)
    ds, receipt = project_spectrodataset(assembled, spectro_dataset_cls=_Double)
    assert ds.features_calls[0][0].dtype == np.float32 and ds.features_calls[0][1] == ["snv"]
    assert receipt["processings"][0]["precision"]["origin"] == "consumer"
    meta = ds.metadata[0]
    assert meta["__sample_weight__"].dtype == np.float64, "weights keep their raw precision"


def test_new_modules_keep_the_sdk_import_lazy() -> None:
    code = (
        "import sys, numpy as np, nirs4all_io as nio\n"
        "import nirs4all_io.materialize.projection, nirs4all_io.materialize.numeric\n"
        "p = nio.to_dataset_package((np.zeros((2, 1)), np.zeros(2)))\n"
        "DatasetPackage = type(p); DatasetPackage.from_full_dict(p.to_full_dict())\n"
        "assert not [m for m in sys.modules if m == 'nirs4all' or m.startswith('nirs4all.')]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _wire_paths(value, path=()):
    yield path
    if isinstance(value, dict):
        for key in value:
            yield from _wire_paths(value[key], (*path, key))
    elif isinstance(value, list):
        for index in range(len(value)):
            yield from _wire_paths(value[index], (*path, index))


def _mutated(wire, path, mode):
    wire = json.loads(json.dumps(wire))
    parent = wire
    for key in path[:-1]:
        parent = parent[key]
    if mode == "delete":
        parent.pop(path[-1])
    else:
        parent[path[-1]] = {"null": None, "int": 7, "str": "zz"}[mode]
    return wire


@pytest.mark.parametrize("reader", ["package", "assembled"])
def test_every_single_field_corruption_is_a_contract_error_never_a_crash(tmp_path, reader) -> None:
    read: Callable[[Any], object]
    if reader == "package":
        wire, read = _roundtrip(_rich_package(tmp_path).to_full_dict()), DatasetPackage.from_full_dict
    else:
        wire, read = _roundtrip(nio.load(_weights_variations_spec(tmp_path), target="assembled", base_dir=tmp_path).to_full_dict()), AssembledDataset.from_full_dict
    attempts = 0
    for path in _wire_paths(wire):
        for mode in ("delete", "null", "int", "str"):
            if not path:
                continue
            broken = _mutated(wire, path, mode)
            if broken == wire:
                continue
            attempts += 1
            try:
                read(broken)
            except NumericContractError:
                pass  # refused, or (free-form labels only) accepted below without crashing
    assert attempts > 300


def test_package_feature_payloads_must_match_partition_source_ids(tmp_path) -> None:
    wire = _roundtrip(_rich_package(tmp_path).to_full_dict())
    wire["partitions"]["train"]["source_ids"] = ["other"]
    with pytest.raises(NumericContractError, match="source_ids disagree"):
        DatasetPackage.from_full_dict(wire)


def test_a_caller_supplied_dataset_content_identity_is_never_accepted(tmp_path) -> None:
    package, receipt = _projection(_identity_spec(tmp_path), tmp_path)
    assert receipt.dataset_content_identity is None, "declined: no validated original public record exists in the oracle"
    with pytest.raises(TypeError):
        package.executed_projection(["d"], dataset_content_identity={"profile": "x"})
    claimed = {**receipt.to_dict(), "dataset_content_identity": {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256", "fingerprint": "0" * 64}}
    with pytest.raises(NumericContractError, match="disagrees"):
        ExecutedProjectionIdentityV1.from_dict(claimed).validate(package)
