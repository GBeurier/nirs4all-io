# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Typed cells through ``concat_samples`` - the three C-cause corrections of the Opus 5.6 work review, on its exact inputs.

``neg10`` / ``neg11`` / ``neg12`` / ``concat_shapes`` (preserved in ``post-b4-residual-abc/opus-abc-before-evidence``):

1. numeric or all-NA inputs that need no retention/refusal/text/bool authority keep the plain MERGED column: ``[3, 4]`` beside
   ``[1.5, 2.5]`` is the float column ``3.0, 4.0, 1.5, 2.5`` in both file orders (an all-NA or NA-holding peer also makes integers
   floats), integer-only inputs stay integers, and an ordinary numeric ID follows the same cause (``'3.0'`` versus ``'3'``);
2. an all-NA input beside booleans keeps the typed ``Bool`` cells in BOTH file orders and the object (String) label, never ``1.0/0.0``;
3. retained / refused tokens, text and bool authority and the numeric-role X/Y/weights controls are unchanged.

Values are compared by type as well as by value (``3 == 3.0`` must never hide the defect).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

import nirs4all_io as nio
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.materialize import DatasetPackage, NumericContractError
from nirs4all_io.materialize.package import FeatureMatrix
from nirs4all_io.strict_json import NumberDomainError

B, U, X = "9007199254740993", "18446744073709551615", "18446744073709551616"
P = {"delimiter": ","}
IGN: dict[str, Any] = {"na": {"policy": "ignore"}}
RM: dict[str, Any] = {"na": {"policy": "remove_sample"}}
MC = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]
TBL = ["TRUE", "FALSE"]


def _f(values: list[str], first: int = 1) -> str:
    return "m,x,y\n" + "".join(f"{v},{i + first}.5,{i + first}.0\n" for i, v in enumerate(values))


def _spec(inputs: tuple[str, ...], *, sample_id: bool = False, **params: Any) -> dict[str, Any]:
    source: dict[str, Any] = {"id": "d", "role": "mixed", "input": list(inputs), "merge": "concat_samples", "columns": MC[:2] if sample_id else MC}
    spec: dict[str, Any] = {"sources": [source], "params": {**P, **params}}
    if sample_id:
        source["key"] = "m"
        spec["sample_index"] = {"by": "id", "key": "m"}
    return spec


def _package(tmp_path: Path, files: dict[str, str], spec: dict[str, Any]) -> DatasetPackage:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return nio.to_dataset_package(spec, base_dir=tmp_path)


def _metadata(package: DatasetPackage) -> dict[str, Any]:
    return package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]


def _typed(expected: list[Any], actual: list[Any]) -> None:
    assert [type(v) for v in expected] == [type(v) for v in actual] and expected == actual, (expected, actual)


NAN = None  # a missing cell is JSON null in the metadata body
PLAIN = [
    # name, first input, second input, expected values, label
    ("int-float", ["3", "4"], ["1.5", "2.5"], [3.0, 4.0, 1.5, 2.5], "numeric"),
    ("float-int", ["1.5", "2.5"], ["3", "4"], [1.5, 2.5, 3.0, 4.0], "numeric"),
    ("int-int", ["3", "4"], ["5", "6"], [3, 4, 5, 6], "numeric"),
    ("intna-int", ["3", ""], ["5", "6"], [3.0, NAN, 5.0, 6.0], "numeric"),
    ("int-intna", ["5", "6"], ["3", ""], [5.0, 6.0, 3.0, NAN], "numeric"),
    ("allna-int", ["", ""], ["3", "4"], [NAN, NAN, 3.0, 4.0], "numeric"),
    ("int-allna", ["3", "4"], ["", ""], [3.0, 4.0, NAN, NAN], "numeric"),
    ("allna-float", ["", ""], ["1.5", "2.5"], [NAN, NAN, 1.5, 2.5], "numeric"),
    ("allna-big", ["", ""], [B, "3"], [NAN, NAN, 9007199254740993, 3], "numeric"),  # retained integers stay exact
    ("allna-u64", ["", ""], [U, "3"], [NAN, NAN, 18446744073709551615, 3], "numeric"),
]


@pytest.mark.parametrize(("name", "first", "second", "expected", "label"), PLAIN, ids=[row[0] for row in PLAIN])
def test_plain_numeric_inputs_keep_the_merged_column(tmp_path: Path, name: str, first: list[str], second: list[str], expected: list[Any], label: str) -> None:
    column = _metadata(_package(tmp_path, {"a.csv": _f(first), "b.csv": _f(second, 3)}, _spec(("a.csv", "b.csv"), **IGN)))
    _typed(expected, column["values"])
    assert column["dtype"] == label


def _ids(package: DatasetPackage) -> list[str]:
    sources: list[str] = []
    for partition in package.partitions.values():
        sources.extend(source for source in partition.source_ids if source not in sources)
    return package.executed_projection(sources).identity.sample_ids


@pytest.mark.parametrize(
    ("first", "second", "ids"),
    [(["3", "4"], ["1.5", "2.5"], ["3.0", "4.0", "1.5", "2.5"]), (["1.5", "2.5"], ["3", "4"], ["1.5", "2.5", "3.0", "4.0"]), (["3", "4"], ["5", "6"], ["3", "4", "5", "6"])],
    ids=["int-float", "float-int", "int-int"],
)
def test_ordinary_numeric_ids_follow_the_same_cause(tmp_path: Path, first: list[str], second: list[str], ids: list[str]) -> None:
    package = _package(tmp_path, {"a.csv": _f(first), "b.csv": _f(second, 3)}, _spec(("a.csv", "b.csv"), sample_id=True))
    assert _ids(package) == ids


@pytest.mark.parametrize(("first", "second"), [(["3", ""], ["5", "6"]), (["5", "6"], ["3", ""]), (["", ""], ["3", "4"]), (["3", "4"], ["", ""])], ids=["intna-int", "int-intna", "allna-int", "int-allna"])
def test_missing_ids_refuse_the_whole_block_never_trim(tmp_path: Path, first: list[str], second: list[str]) -> None:
    package = _package(tmp_path, {"a.csv": _f(first), "b.csv": _f(second, 3)}, _spec(("a.csv", "b.csv"), sample_id=True, **IGN))
    with pytest.raises(NumericContractError, match="identity"):
        _ids(package)


def test_row_removal_classifies_once_over_the_original_rows(tmp_path: Path) -> None:
    # the removed first row is an integer: the survivors are still the plain float column of the float peer, never reclassified
    package = _package(tmp_path, {"a.csv": "m,x,y\n3,1.5,\n4,2.5,2.0\n", "b.csv": _f(["1.5", "2.5"], 3)}, _spec(("a.csv", "b.csv"), **RM))
    _typed([4.0, 1.5, 2.5], _metadata(package)["values"])
    package = _package(tmp_path, {"a.csv": _f(["1.5", "2.5"]), "b.csv": "m,x,y\n3,3.5,\n4,4.5,4.0\n"}, _spec(("a.csv", "b.csv"), **RM))
    _typed([1.5, 2.5, 4.0], _metadata(package)["values"])
    package = _package(tmp_path, {"a.csv": "m,x,y\n3,1.5,\n4,2.5,2.0\n", "b.csv": _f(["5", "6"], 3)}, _spec(("a.csv", "b.csv"), **RM))
    _typed([4, 5, 6], _metadata(package)["values"])
    package = _package(tmp_path, {"a.csv": "m,x,y\n3,1.5,\n4,2.5,2.0\n", "b.csv": _f(["1.5", "2.5"], 3)}, _spec(("a.csv", "b.csv"), sample_id=True, **RM))
    assert _ids(package) == ["4.0", "1.5", "2.5"]


def test_three_inputs_keep_the_plain_merged_column(tmp_path: Path) -> None:
    files = {"a.csv": _f(["3", "4"]), "b.csv": _f(["1.5", "2.5"], 3), "c.csv": _f(["5", ""], 5)}
    _typed([3.0, 4.0, 1.5, 2.5, 5.0, NAN], _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv", "c.csv"), **IGN)))["values"])
    files = {"a.csv": _f(["", ""]), "b.csv": _f(["3", "4"], 3), "c.csv": _f(["1.5", "2.5"], 5)}
    _typed([NAN, NAN, 3.0, 4.0, 1.5, 2.5], _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv", "c.csv"), **IGN)))["values"])


# --------------------------------------------------------------------------- #
# 2. all-NA beside booleans keeps the typed Bool cells in both orders          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(["", ""], TBL, [NAN, NAN, True, False]), (TBL, ["", ""], [True, False, NAN, NAN]), (["", ""], ["TRUE", ""], [NAN, NAN, True, NAN]), (["TRUE", ""], ["", ""], [True, NAN, NAN, NAN])],
    ids=["allna-bool", "bool-allna", "allna-boolna", "boolna-allna"],
)
def test_all_na_beside_booleans_keeps_typed_bool_cells(tmp_path: Path, first: list[str], second: list[str], expected: list[Any]) -> None:
    column = _metadata(_package(tmp_path, {"a.csv": _f(first), "b.csv": _f(second, 3)}, _spec(("a.csv", "b.csv"), **IGN)))
    _typed(expected, column["values"])
    assert column["dtype"] == "string"  # the established object rule, never numeric 1.0 / 0.0


def test_all_na_bool_int_three_inputs_keep_every_input_typed(tmp_path: Path) -> None:
    files = {"a.csv": _f(["", ""]), "b.csv": _f(TBL, 3), "c.csv": _f(["3", "4"], 5)}
    column = _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv", "c.csv"), **IGN)))
    _typed([NAN, NAN, True, False, 3, 4], column["values"])
    assert column["dtype"] == "string"


# --------------------------------------------------------------------------- #
# 3. unchanged authority                                                       #
# --------------------------------------------------------------------------- #
def test_retained_refused_text_and_bool_authority_are_unchanged(tmp_path: Path) -> None:
    files = {"a.csv": _f(["1.5", "2.5"]), "b.csv": _f(["3", "4"], 3), "c.csv": _f([B, "3"], 5)}
    _typed([1.5, 2.5, 3, 4, 9007199254740993, 3], _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv", "c.csv"))))["values"])
    _typed([True, False, 3, 4], _metadata(_package(tmp_path, {"a.csv": _f(TBL), "b.csv": _f(["3", "4"], 3)}, _spec(("a.csv", "b.csv"))))["values"])
    _typed([3, 4, True, False], _metadata(_package(tmp_path, {"a.csv": _f(["3", "4"]), "b.csv": _f(TBL, 3)}, _spec(("a.csv", "b.csv"))))["values"])
    _typed(["abc", "def", 3, 4], _metadata(_package(tmp_path, {"a.csv": _f(["abc", "def"]), "b.csv": _f(["3", "4"], 3)}, _spec(("a.csv", "b.csv"))))["values"])
    with pytest.raises(NumberDomainError, match="outside"):
        _package(tmp_path, {"a.csv": _f(["1.5", "2.5"]), "b.csv": _f([X, "4"], 3)}, _spec(("a.csv", "b.csv")))


def test_numeric_roles_with_bool_and_int_inputs_stay_supported(tmp_path: Path) -> None:
    files = {"a.csv": "m,x,y\nTRUE,1.5,1.0\nFALSE,2.5,2.0\n", "b.csv": "m,x,y\n3,3.5,3.0\n4,4.5,4.0\n"}
    columns = [{"role": "features", "select": ["x", "m"]}, {"role": "targets", "select": ["y"]}]
    package = _package(tmp_path, files, {"sources": [{"id": "d", "role": "mixed", "input": ["a.csv", "b.csv"], "merge": "concat_samples", "columns": columns}], "params": P})
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    block = payload.matrix.values
    assert block.tolist() == [[1.0, 1.5], [0.0, 2.5], [3.0, 3.5], [4.0, 4.5]]  # the Bool column is numeric under a numeric role


def test_column_missing_from_one_input_is_labelled_string_beside_booleans(tmp_path: Path) -> None:
    for files, order in (({"a.csv": _f(TBL), "b.csv": "x,y\n3.5,3.0\n4.5,4.0\n"}, [True, False, NAN, NAN]), ({"a.csv": "x,y\n3.5,3.0\n4.5,4.0\n", "b.csv": _f(TBL, 3)}, [NAN, NAN, True, False])):
        column = _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv"), **IGN)))
        _typed(order, column["values"])
        assert column["dtype"] == "string"


# --------------------------------------------------------------------------- #
# Precision / sign through the public path: the ordinary merged column keeps the literal's bits                  #
# --------------------------------------------------------------------------- #
_FINITE = json.loads((Path(__file__).parent / "fixtures" / "finite_float_vectors.json").read_text(encoding="utf-8"))["cases"]
_CONTROLS = [{"id": "decimal-rounding", "repr": "0.12345678901234567", "bits_hex": struct.pack(">d", float("0.12345678901234567")).hex()}, {"id": "negative-zero-underflow", "repr": "-0.000e-999", "bits_hex": "8000000000000000"}]


@pytest.mark.parametrize("case", _FINITE + _CONTROLS, ids=[c["id"] for c in _FINITE + _CONTROLS])
@pytest.mark.parametrize("order", ["token-second", "token-first"])
def test_ordinary_concat_keeps_the_literal_bits_kinds_and_ids(tmp_path: Path, case: dict[str, str], order: str) -> None:
    ints, floats = _f(["3", "4"]), _f([case["repr"], "1.5"], 3)
    files = {"a.csv": ints, "b.csv": floats} if order == "token-second" else {"a.csv": floats, "b.csv": ints}
    expected = [3.0, 4.0, float(case["repr"]), 1.5] if order == "token-second" else [float(case["repr"]), 1.5, 3.0, 4.0]
    column = _metadata(_package(tmp_path, files, _spec(("a.csv", "b.csv"))))
    assert column["dtype"] == "numeric"
    _typed(expected, column["values"])
    assert [struct.pack(">d", v).hex() for v in column["values"]] == [struct.pack(">d", v).hex() for v in expected]
    assert case["bits_hex"] in {struct.pack(">d", v).hex() for v in column["values"]}
    (tmp_path / "id").mkdir()
    package = _package(tmp_path / "id", dict(files), _spec(("a.csv", "b.csv"), sample_id=True))
    if "+" in case["repr"]:
        # the primary-identity grammar refuses an exponent sign in both owners (Rust and Python): the column itself is still exact above
        with pytest.raises(NumericContractError, match="Invalid primary identity"):
            _ids(package)
    else:
        assert _ids(package) == [repr(v) for v in expected]


_SDK_PRECISION = json.loads((Path(__file__).parent / "fixtures" / "sdk_csv_precision_vectors.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _SDK_PRECISION, ids=[case["id"] for case in _SDK_PRECISION])
def test_sdk_csv_precision_matches_retained_rust_owner(tmp_path: Path, case: dict[str, Any]) -> None:
    """The same physical CSV inputs keep the retained owner's values and bits."""
    for name, text in case["files"].items():
        (tmp_path / name).write_bytes(text.encode("utf-8"))
    package = nio.to_dataset_package(case["spec"], base_dir=tmp_path)
    wire = package.to_full_dict()
    expected = case["expected"]
    assert canonical_json(wire).encode("utf-8") == expected["package_full_canonical"].encode("utf-8")
    assert package.manifest().root == expected["root"]
    for cells in expected["cells"]:
        actual: Any = wire
        for part in cells["path"]:
            actual = actual[part]
        assert actual["dtype"] == cells["dtype"]
        if "data_bytes_hex" in cells:
            assert actual["data_bytes_hex"] == cells["data_bytes_hex"]
            values = [cell[0] for cell in struct.iter_unpack("<d", bytes.fromhex(actual["data_bytes_hex"]))]
        else:
            values = actual["values"]
        _typed(cells["values"], values)
        assert [type(value).__name__ for value in values] == cells["types"]
        assert [struct.pack(">d", value).hex() if type(value) is float else None for value in values] == cells["binary64_hex"]
    if expected["sample_ids"] is not None:
        assert _ids(package) == expected["sample_ids"]
