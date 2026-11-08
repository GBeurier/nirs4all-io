# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Nullable typed (Parquet) integer columns keep their exact cells until the role's selection.

``to_pandas`` turns an integer column holding nulls into float64 (NaN) and rounds integers above ``2**53`` before any
selection can look. The loader rebuilds such columns from the original Arrow cells (pandas nullable ``Int64`` /
``UInt64``); the numeric roles then use the same checked selection as every other source: default float64 refuses an
inexact integer, an explicit float32 request rounds the original integer directly, NA becomes a false mask cell with +0,
and genuine floating columns keep their bits. Non-nullable integers and ordinary strings are unchanged.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize import NumericContractError
from nirs4all_io.materialize.package import FeatureMatrix, MetadataTable, TargetTable, WeightsVector

N = 2**54 + 2**30 + 1


def _write(path: Path, **columns: Any) -> None:
    pq.write_table(pa.table(columns), path / "data.parquet")


def _package(path: Path, spec_columns: dict[str, Any] | None = None, **options: Any):
    spec: dict[str, Any] = {"params": {"na": {"policy": "ignore"}}, "sources": [{"id": "s", "input": "data.parquet", "columns": spec_columns or {"features": "x", "targets": "y"}}]}
    if "task_type" in options:
        spec["task_type"] = options.pop("task_type")
    return nio.to_dataset_package(spec, base_dir=path, **options)


def _y_block(package):
    table = dict(package.partitions["train"].payloads)["train/y"]
    assert isinstance(table, TargetTable)
    return table.columns[0].block


def _x_block(package):
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload.matrix


def test_nullable_int64_target_keeps_the_original_integer(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([N, None], pa.int64()))
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path)  # default float64 must not round the integer
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, targets="float64")
    direct = _y_block(_package(tmp_path, targets="float32"))
    assert direct.values_bytes().hex() == "0100805a00000000", "requested float32 rounds the original integer directly (upper)"
    assert direct.validity.ravel().tolist() == [True, False], "the null is a false mask cell holding +0"
    assert direct.precision.to_dict() == {"dtype": "float32", "origin": "requested", "source_dtype": "int64", "requested_dtype": "float32", "consumer": None}


def test_exact_large_values_and_missing_cells(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([2**54, None], pa.int64()))
    block = _y_block(_package(tmp_path))
    assert block.values.ravel().tolist() == [float(2**54), 0.0] and block.validity.ravel().tolist() == [True, False]
    assert block.precision.source_dtype == "int64" and block.dtype == "float64", "the integer source is declared, never a pre-rounded float"
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([2**53 + 1, None], pa.int64()))
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path)


def test_nullable_features_and_weights_follow_the_same_selection(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([N, None], pa.int64()), y=pa.array([1.0, 2.0], pa.float64()), w=pa.array([None, N], pa.int64()))
    columns = {"features": "x", "targets": "y", "weights": "w"}
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, columns)
    package = _package(tmp_path, columns, features="float32", weights="float32")
    x = _x_block(package)
    assert x.values_bytes().hex() == "0100805a00000000" and x.validity.ravel().tolist() == [True, False] and x.precision.source_dtype == "int64"
    weights = dict(package.partitions["train"].payloads)["train/weights"]
    assert isinstance(weights, WeightsVector)
    assert weights.matrix.values_bytes().hex() == "000000000100805a" and weights.matrix.validity.ravel().tolist() == [False, True]


def test_nullable_uint64_keeps_values_up_to_the_int64_maximum(tmp_path: Path) -> None:
    top = 2**63 - 1  # the Parquet reader is an i64 domain (C5): a UInt64 above this refuses, see test_owner_corrective_c5.py
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([top, None], pa.uint64()))
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path)
    block = _y_block(_package(tmp_path, targets="float32"))
    assert block.values_bytes()[:4].hex() == struct.pack("<f", float(2**63)).hex()
    assert block.precision.source_dtype == "uint64"
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([2**64 - 1, None], pa.uint64()))
    with pytest.raises(Exception, match="outside i64 range"):
        _package(tmp_path, targets="float32")


def test_genuine_floating_nullable_columns_and_plain_integers_are_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([1.5, None], pa.float64()), y=pa.array([3, 4], pa.int64()))
    package = _package(tmp_path)
    x, y = _x_block(package), _y_block(package)
    assert x.values.ravel().tolist() == [1.5, 0.0] and x.validity.ravel().tolist() == [True, False] and x.precision.source_dtype == "float64"
    assert y.values.ravel().tolist() == [3.0, 4.0] and y.precision.source_dtype == "int64"
    _write(tmp_path, x=pa.array([1.5, 2.5], pa.float32()), y=pa.array([N, N + 2**31], pa.int64()))  # non-nullable int64: already exact
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path)
    direct = _y_block(_package(tmp_path, targets="float32")).values_bytes().hex()
    assert direct[:8] == "0100805a" and len(direct) == 16, "already exact non-nullable int64 rounds directly too"


def test_classification_cap_and_f32_exactness_hold_for_nullable_labels(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([2**53, None], pa.int64()))
    with pytest.raises(NumericContractError, match="portable integer"):
        _package(tmp_path, task_type="binary")
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([16777217, None], pa.int64()))
    with pytest.raises(NumericContractError, match="not exact in float32"):
        _package(tmp_path, task_type="binary", targets="float32")
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([16777218, None], pa.int64()))
    block = _y_block(_package(tmp_path, task_type="binary", targets="float32"))
    assert block.values_bytes()[:4].hex() == struct.pack("<f", 16777218.0).hex() and block.validity.ravel().tolist() == [True, False]


def test_ordinary_strings_next_to_nullable_integers_are_untouched(tmp_path: Path) -> None:
    _write(tmp_path, x=pa.array([1.0, 2.0], pa.float64()), y=pa.array([1.0, 2.0], pa.float64()), note=pa.array(["1e-400", None], pa.string()), count=pa.array([5, None], pa.int64()))
    package = _package(tmp_path, {"features": "x", "targets": "y", "metadata": ["note", "count"]})
    metadata = dict(package.partitions["train"].payloads)["train/metadata"]
    assert isinstance(metadata, MetadataTable)
    assert metadata.frame["note"].tolist()[0] == "1e-400"
