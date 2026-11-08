# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Residual A and C of the Opus 5.6 work review (Python side), on the exact reviewer inputs ``neg7`` / ``neg8`` / ``neg9``.

A: the approved rule has no NA-only exception - a row removed by a JOIN never refuses either, a kept bad cell still does.
C: every LOADED input keeps its own typed Bool/Integer cells through ``concat_samples`` (both file orders); the stacked
   column is an object column, never a string or float coercion. Values are asserted by type, not by Python ``==``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

import nirs4all_io as nio
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.materialize import DatasetPackage
from nirs4all_io.materialize.package import MetadataTable, _metadata_body, assembled_from_full_dict, assembled_to_full_dict
from nirs4all_io.strict_json import NumberDomainError

X = "18446744073709551616"  # 2**64
P = {"delimiter": ","}


def typed(expected: list[Any], actual: list[Any]) -> None:
    assert [type(v) for v in expected] == [type(v) for v in actual] and expected == actual, (expected, actual)


def _block(package: DatasetPackage, name: str) -> Any:
    return dict(package.partitions["train"].payloads)[f"train/{name}"]


def _x(package: DatasetPackage) -> list[float]:
    return _block(package, "x0").matrix.values.ravel().tolist()


def _y(package: DatasetPackage) -> list[float]:
    return _block(package, "y").columns[0].block.values.ravel().tolist()


def _metadata(package: DatasetPackage) -> dict[str, Any]:
    payload = _block(package, "metadata")
    assert isinstance(payload, MetadataTable)
    return _metadata_body(payload.frame)


def _write(tmp_path: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)


# --------------------------------------------------------------------------- #
# A: JOIN-removed rows never refuse                                              #
# --------------------------------------------------------------------------- #
def _lookup() -> dict[str, Any]:
    return {
        "sources": [
            {"id": "x", "role": "mixed", "input": "a.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["site"]}]},
            {"id": "s", "kind": "lookup", "input": "s.csv", "columns": [{"role": "metadata", "select": ["r"]}], "join": {"to": "x", "left_on": "site", "right_on": "site", "how": "m:1", "coverage": "drop"}},
        ],
        "params": P,
    }


@pytest.mark.parametrize(("token", "peer"), [(X, "2.5"), (X, "2"), ("1e-400", "2.5")])
def test_a_lookup_coverage_drop_never_refuses_the_dropped_row(tmp_path: Path, token: str, peer: str) -> None:
    _write(tmp_path, {"a.csv": f"x,site\n{token},S9\n{peer},S1\n", "s.csv": "site,r\nS1,0.5\n"})
    package = nio.to_dataset_package(_lookup(), base_dir=tmp_path)
    assert package.partitions["train"].n_samples == 1
    typed([float(peer)], _x(package))
    columns = {c["name"]: c["values"] for c in _metadata(package)["columns"]}
    typed(["S1"], columns["site"])
    typed([0.5], columns["r"])


@pytest.mark.parametrize("token", [X, "1e-400"])
def test_a_the_same_cell_in_a_kept_lookup_row_still_refuses(tmp_path: Path, token: str) -> None:
    _write(tmp_path, {"a.csv": f"x,site\n{token},S1\n2.5,S1\n", "s.csv": "site,r\nS1,0.5\n"})
    with pytest.raises(NumberDomainError):
        nio.to_dataset_package(_lookup(), base_dir=tmp_path)


def _right() -> dict[str, Any]:
    return {
        "sample_index": {"by": "id", "key": "id"},
        "sources": [
            {"id": "x", "input": "x.csv", "key": "id", "columns": {"features": "x"}},
            {"id": "y", "input": "y.csv", "key": "id", "columns": {"targets": "y"}, "join": {"to": "x", "on": "id", "how": "m:1", "coverage": "warn"}},
        ],
    }


@pytest.mark.parametrize("unmatched", ["2e-324", X])
@pytest.mark.parametrize("peer", ["20.5", "20"])
def test_a_right_unmatched_row_never_refuses(tmp_path: Path, unmatched: str, peer: str) -> None:
    _write(tmp_path, {"x.csv": "id;x\ns0;1\ns1;2\n", "y.csv": f"id;y\ns1;{peer}\ns0;10\ns9;{unmatched}\n"})
    package = nio.to_dataset_package(_right(), base_dir=tmp_path)
    typed([1.0, 2.0], _x(package))
    typed([10.0, float(peer)], _y(package))


@pytest.mark.parametrize("token", ["2e-324", X])
def test_a_right_matched_kept_row_still_refuses(tmp_path: Path, token: str) -> None:
    _write(tmp_path, {"x.csv": "id;x\ns0;1\ns1;2\n", "y.csv": f"id;y\ns1;{token}\ns0;10.5\n"})
    with pytest.raises(NumberDomainError):
        nio.to_dataset_package(_right(), base_dir=tmp_path)


# --------------------------------------------------------------------------- #
# C: typed cells of each loaded input survive concat_samples                     #
# --------------------------------------------------------------------------- #
BOOL = "m,x,y\nTRUE,1.5,1.0\nFALSE,2.5,2.0\n"
INT = "m,x,y\n3,3.5,3.0\n4,4.5,4.0\n"
FLOAT = "m,x,y\n1.5,3.5,3.0\n2.5,4.5,4.0\n"
TEXT = "m,x,y\nabc,3.5,3.0\nde,4.5,4.0\n"


def _concat(tmp_path: Path, files: dict[str, str], role: str = "metadata", sample: bool = False, **params: Any) -> DatasetPackage:
    _write(tmp_path, files)
    columns = {
        "metadata": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}],
        "features": [{"role": "features", "select": ["m"]}, {"role": "targets", "select": ["y"]}, {"role": "ignore", "select": ["x"]}],
        "targets": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["m"]}, {"role": "ignore", "select": ["y"]}],
        "weights": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["m"]}],
    }[role]
    source: dict[str, Any] = {"id": "d", "role": "mixed", "input": list(files), "merge": "concat_samples", "columns": columns}
    spec: dict[str, Any] = {"sources": [source], "params": {**P, **params}}
    if sample:
        source["key"] = "m"
        spec["sample_index"] = {"by": "id", "key": "m"}
    return nio.to_dataset_package(spec, base_dir=tmp_path)


def test_c_bool_then_integer_keeps_typed_cells_in_an_object_column(tmp_path: Path) -> None:
    package = _concat(tmp_path, {"a.csv": BOOL, "b.csv": INT})
    column = _metadata(package)["columns"][0]
    typed([True, False, 3, 4], column["values"])
    assert column["dtype"] == "string", "the existing object-column label rule (an object column with a boolean is not 'numeric')"
    frame = _block(package, "metadata").frame
    assert frame["m"].dtype == object and [type(v).__name__ for v in frame["m"]] == ["bool", "bool", "int", "int"]


def test_c_integer_then_bool_keeps_typed_cells_in_an_object_column(tmp_path: Path) -> None:
    package = _concat(tmp_path, {"a.csv": INT, "b.csv": BOOL})
    typed([3, 4, True, False], _metadata(package)["columns"][0]["values"])


@pytest.mark.parametrize(("files", "expected"), [
    ({"a.csv": BOOL, "b.csv": FLOAT}, [True, False, 1.5, 2.5]),
    ({"a.csv": FLOAT, "b.csv": BOOL}, [1.5, 2.5, True, False]),
    ({"a.csv": BOOL, "b.csv": TEXT}, [True, False, "abc", "de"]),
    ({"a.csv": TEXT, "b.csv": BOOL}, ["abc", "de", True, False]),
    ({"a.csv": BOOL, "b.csv": INT, "c.csv": FLOAT}, [True, False, 3, 4, 1.5, 2.5]),
    ({"a.csv": FLOAT, "b.csv": BOOL, "c.csv": INT}, [1.5, 2.5, True, False, 3, 4]),
])
def test_c_other_input_mixes_keep_each_inputs_own_types(tmp_path: Path, files: dict[str, str], expected: list[Any]) -> None:
    typed(expected, _metadata(_concat(tmp_path, files))["columns"][0]["values"])


def test_c_bool_na_and_row_removal_keep_the_surviving_typed_cells(tmp_path: Path) -> None:
    na = _concat(tmp_path, {"a.csv": "m,x,y\nTRUE,1.5,1.0\n,2.5,2.0\n", "b.csv": INT}, na={"policy": "ignore"})
    typed([True, None, 3, 4], _metadata(na)["columns"][0]["values"])
    removed = _concat(tmp_path, {"a.csv": "m,x,y\nTRUE,1.5,\nFALSE,2.5,2.0\n", "b.csv": INT}, na={"policy": "remove_sample"})
    typed([False, 3, 4], _metadata(removed)["columns"][0]["values"])


def test_c_per_input_numeric_authority_is_unchanged_next_to_a_bool_input(tmp_path: Path) -> None:
    bad = f"m,x,y\n{X},3.5,3.0\n4,4.5,4.0\n"
    for files in ({"a.csv": BOOL, "b.csv": bad}, {"a.csv": bad, "b.csv": BOOL}):
        with pytest.raises(NumberDomainError, match="outside the supported range"):
            _concat(tmp_path, files)


@pytest.mark.parametrize("role", ["features", "targets", "weights"])
def test_c_numeric_roles_keep_the_established_bool_as_one_or_zero(tmp_path: Path, role: str) -> None:
    for files, expected in (({"a.csv": BOOL, "b.csv": INT}, [1.0, 0.0, 3.0, 4.0]), ({"a.csv": INT, "b.csv": BOOL}, [3.0, 4.0, 1.0, 0.0])):
        package = _concat(tmp_path, files, role)
        if role == "features":
            typed(expected, _x(package))
        elif role == "targets":
            typed(expected, _y(package))
        else:
            typed(expected, _block(package, "weights").matrix.values.ravel().tolist())


def test_c_sample_ids_preimages_hashes_and_roots_follow_the_typed_cells(tmp_path: Path) -> None:
    (tmp_path / "1").mkdir()
    bool_first = _concat(tmp_path / "1", {"a.csv": BOOL, "b.csv": INT}, sample=True)
    ids_one = bool_first.executed_projection(["d"])
    typed(["True", "False", "3", "4"], list(ids_one.identity.sample_ids))
    (tmp_path / "2").mkdir()
    int_first = _concat(tmp_path / "2", {"a.csv": INT, "b.csv": BOOL}, sample=True)
    ids_two = int_first.executed_projection(["d"])
    typed(["3", "4", "True", "False"], list(ids_two.identity.sample_ids))
    assert struct.pack("<Q", 4) + b"True" in ids_one.feature_preimage and b"true" not in ids_one.feature_preimage
    assert ids_one.identity.feature_content_fingerprint != ids_two.identity.feature_content_fingerprint
    assert bool_first.manifest().root != int_first.manifest().root
    # the metadata payload hash is a function of the typed cells: True/False are not the integers 1/0
    (tmp_path / "3").mkdir()
    ints = _concat(tmp_path / "3", {"a.csv": BOOL.replace("TRUE", "1").replace("FALSE", "0"), "b.csv": INT}, sample=False)
    plain = _concat(tmp_path / "3", {"a.csv": BOOL, "b.csv": INT})
    ints_meta = next(e for e in ints.manifest().entries if e.id.endswith("/metadata"))
    plain_meta = next(e for e in plain.manifest().entries if e.id.endswith("/metadata"))
    assert ints_meta.content_hash != plain_meta.content_hash


def test_c_the_typed_column_survives_the_full_wires(tmp_path: Path) -> None:
    package = _concat(tmp_path, {"a.csv": BOOL, "b.csv": INT})
    wire = json.loads(canonical_json(package.to_full_dict()))
    restored = DatasetPackage.from_full_dict(wire)
    assert canonical_json(restored.to_full_dict()) == canonical_json(package.to_full_dict())
    typed([True, False, 3, 4], _metadata(restored)["columns"][0]["values"])
    assembled = assembled_to_full_dict(package.to_assembled())
    again = assembled_from_full_dict(json.loads(canonical_json(assembled)))
    assert canonical_json(assembled_to_full_dict(again)) == canonical_json(assembled)
    typed([True, False, 3, 4], assembled["blocks"]["train"]["metadata"]["columns"][0]["values"])
