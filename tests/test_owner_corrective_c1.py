# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C1: one finalizer seam for every assembly exit that promises numeric storage options.

``assemble`` / ``load(target="assembled")`` / ``_assemble_in_memory`` select storage for exactly the
REQUESTED roles through the shared ``select_block`` helpers: X/processings when features or a consumer is
requested, Y only when targets are requested, weights only when weights are requested. Unrequested raw
targets and weights are never rewritten (an SDK int64 label keeps its dtype).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize import DatasetPackage, NumericContractError
from nirs4all_io.materialize.assemble import assemble
from nirs4all_io.materialize.numeric import NumericStorageOptions
from nirs4all_io.spec import DatasetSpec
from nirs4all_io.spec.enums import SpecError

ULP = 1.0 + 2.0**-52  # 1.0000000000000002
BIG = 18014399583223809  # 2**54 + 1: rounds to 0x5a800001 directly, 0x5a800000 through F64 first


def _csv_spec(tmp_path: Path) -> dict:
    (tmp_path / "d.csv").write_text(f"x,z,y\n{ULP!r},{BIG},1.0\n2.5,3.5,2.0\n")
    return {"sources": [{"id": "t", "input": "d.csv", "columns": {"features": ["x", "z"], "targets": "y"}}], "params": {"delimiter": ","}}


def _x_hex(assembled) -> str:
    return np.ascontiguousarray(assembled.blocks["train"].X[0], dtype="<f8").tobytes().hex()[:16]


def _package_x(assembled, **options) -> dict:
    value = DatasetPackage.from_assembled(assembled, **options).to_full_dict()["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"]
    return {"hex": value["data_bytes_hex"][:16], "precision": value["precision"]}


def test_explicit_float32_is_committed_then_float64_cannot_recover_the_lost_bits(tmp_path) -> None:
    with pytest.raises(Exception, match="float64 cannot represent exactly"):  # default exact F64 refuses the inexact integer
        nio.load(_csv_spec(tmp_path), target="assembled", base_dir=tmp_path)
    assembled = nio.load(_csv_spec(tmp_path), target="assembled", base_dir=tmp_path, features="float32")
    precision = assembled.blocks["train"].x_precisions[0]
    assert (precision.dtype, precision.origin, precision.requested_dtype, precision.consumer) == ("float32", "requested", "float32", None)
    assert _x_hex(assembled) == "000000000000f03f"  # the stored F32 value widened exactly
    with pytest.raises(NumericContractError, match="Cannot re-select float64"):
        DatasetPackage.from_assembled(assembled, features="float64")
    assert _package_x(assembled, features="float32")["hex"] == "0000803f0100805a"  # BIG rounded directly, once
    retained = _package_x(assembled)  # no new request retains the carried explicit one
    assert retained["hex"] == "0000803f0100805a" and retained["precision"]["origin"] == "requested"


def test_consumer_only_is_a_consumer_selection_not_an_explicit_request(tmp_path) -> None:
    assembled = nio.load(_csv_spec(tmp_path), target="assembled", base_dir=tmp_path, consumer="SpectroDataset")
    precision = assembled.blocks["train"].x_precisions[0]
    assert (precision.dtype, precision.origin, precision.requested_dtype, precision.consumer) == ("float32", "consumer", None, "SpectroDataset")
    assert _package_x(assembled)["precision"]["origin"] == "consumer"
    with pytest.raises(NumericContractError, match="SpectroDataset does not support requested float64"):
        nio.load((np.ones((2, 2)), np.ones(2)), target="assembled", features="float64", consumer="SpectroDataset")


def test_direct_public_assemble_applies_the_same_finalizer(tmp_path) -> None:
    (tmp_path / "x.csv").write_text(f"x\n{ULP!r}\n2.0\n")
    spec = DatasetSpec.from_dict({"sources": [{"id": "x", "input": "x.csv", "role": "features", "params": {"delimiter": ","}}]})
    assembled = assemble(spec, tmp_path, NumericStorageOptions(features="float32"))
    block = assembled.blocks["train"]
    assert np.ascontiguousarray(block.X[0], dtype="<f8").tobytes()[:8].hex() == "000000000000f03f"
    assert [p.to_dict() for p in block.x_precisions] == [{"dtype": "float32", "origin": "requested", "source_dtype": None, "requested_dtype": "float32", "consumer": None}]
    plain = assemble(spec, tmp_path)
    assert np.ascontiguousarray(plain.blocks["train"].X[0], dtype="<f8").tobytes()[:8].hex() == "010000000000f03f" and plain.blocks["train"].x_precisions == []


def test_in_memory_input_is_finalized_too() -> None:
    x = np.array([[ULP, float(2**24)], [2.0, 3.0]])
    assembled = nio.load((x, np.array([1.0, 2.0])), target="assembled", features="float32")
    block = assembled.blocks["train"]
    assert block.x_precisions[0].requested_dtype == "float32" and np.ascontiguousarray(block.X[0], dtype="<f8").tobytes()[:8].hex() == "000000000000f03f"
    assert block.y_precisions == []  # targets were not requested
    with pytest.raises(NumericContractError, match="Cannot re-select float64"):
        DatasetPackage.from_assembled(assembled, features="float64")


def test_unrequested_raw_targets_and_weights_are_untouched(tmp_path) -> None:
    y = np.array([16777218, 16777216, 16777218, 16777216], dtype=np.int64)
    assembled = nio.load((np.ones((4, 2)), y), target="assembled", features="float32")
    block = assembled.blocks["train"]
    assert block.y.dtype == np.int64 and block.y_precisions == [] and np.array_equal(block.y.ravel(), y)
    # an explicit targets request selects Y and records its precision; features stay unselected
    selected = nio.load((np.ones((4, 2)), y), target="assembled", targets="float32")
    assert selected.blocks["train"].y.dtype == np.float64 and [p.requested_dtype for p in selected.blocks["train"].y_precisions] == ["float32"]
    assert selected.blocks["train"].x_precisions == []
    # weights: only a weights request commits the weight column
    (tmp_path / "w.csv").write_text("x,w,y\n1.5,1.0000000000000002,1.0\n2.5,2.0,2.0\n")
    spec = {"sources": [{"id": "t", "input": "w.csv", "columns": {"features": "x", "targets": "y", "weights": "w"}}], "params": {"delimiter": ","}}
    untouched = nio.load(spec, target="assembled", base_dir=tmp_path, features="float32")
    assert untouched.blocks["train"].weights_precision is None and untouched.blocks["train"].weights[0] == ULP
    requested = nio.load(spec, target="assembled", base_dir=tmp_path, weights="float32")
    assert requested.blocks["train"].weights_precision.requested_dtype == "float32" and requested.blocks["train"].weights[0] == 1.0


def test_carried_float32_from_a_non_float32_source_cannot_widen_labels_either(tmp_path) -> None:
    y = np.array([16777218, 16777216, 16777218, 16777216], dtype=np.int64)
    assembled = nio.load((np.ones((4, 2)), y), target="assembled", targets="float32")
    assembled.task_type = "binary"
    with pytest.raises(NumericContractError):
        DatasetPackage.from_assembled(assembled, targets="float64")


def test_declared_float32_source_keeps_its_legitimate_widening(tmp_path) -> None:
    np.save(tmp_path / "x.npy", np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
    spec = {"sources": [{"id": "x", "role": "features", "input": "x.npy", "params": {"header_unit": "index"}}]}
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled", features="float64")
    assert assembled.blocks["train"].x_precisions[0].dtype == "float64" and assembled.blocks["train"].x_precisions[0].source_dtype == "float32"


def test_variations_are_selected_with_their_own_records(tmp_path) -> None:
    np.save(tmp_path / "x.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.save(tmp_path / "v.npy", np.array([[ULP, 2.0], [3.0, 4.0]], dtype=np.float64))
    spec = {"sources": [{"id": "x", "role": "features", "input": "x.npy", "params": {"header_unit": "index"}, "variations": [{"name": "v", "input": "v.npy"}]}]}
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled", features="float32")
    block = assembled.blocks["train"]
    assert [p.source_dtype for p in block.processing_precisions[0]] == ["float64"] and block.processing_precisions[0][0].requested_dtype == "float32"
    assert block.processings[0][0][1][0, 0] == 1.0


def test_a_package_input_cannot_be_reselected_through_any_assembly_exit() -> None:
    package = nio.to_dataset_package((np.ones((2, 2)), np.array([1.0, 2.0])))
    message = "already fixed its numeric storage"
    options: list[dict[str, Any]] = [{"features": "float32"}, {"targets": "float32"}, {"weights": "float32"}, {"consumer": "SpectroDataset"}]
    for option in options:
        for target in ("package", "assembled"):
            with pytest.raises(SpecError, match=message):
                nio.load(package, target=target, **option)
    assert nio.load(package, target="package") is package
    assert isinstance(nio.load(package, target="assembled").blocks["train"].X[0], np.ndarray)


def test_in_memory_dataframe_metadata_survives_finalization() -> None:
    meta = pd.DataFrame({"g": ["a", "b"]})
    assembled = nio.load({"X": np.ones((2, 2)), "metadata": meta}, target="assembled", features="float32")
    assert assembled.blocks["predict"].metadata["g"].tolist() == ["a", "b"]
