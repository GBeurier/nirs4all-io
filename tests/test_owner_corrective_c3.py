# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C3: numeric-only CSV metadata/ID columns are numeric authority.

A selected metadata/ID column whose every non-NA token is numeric text keeps exact integers (never a float
rounding of an inexact integer, so ``[2**53 + 1, 1.5]`` keeps two distinct values), leaves F64-exact
integers beside float/NA peers as the unchanged float column, and refuses an integer outside
``[-2**63, 2**64 - 1]`` or a decimal that overflows/underflows binary64. A genuine text column keeps
its literal strings, uncapped. The rule is checked after NA row removal and does not cover spreadsheet cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import nirs4all_io as nio
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.materialize.package import DatasetPackage, MetadataTable, _metadata_body
from nirs4all_io.strict_json import NumberDomainError

N53 = 9007199254740993  # 2**53 + 1: in domain, not exact in binary64
U64MAX = 18446744073709551615
OUT_OF_DOMAIN = ("18446744073709551616", "18446744073709551617", "-9223372036854775809", "340282366920938463463374607431768211456")
_COMMON = {"delimiter": ",", "na": {"policy": "ignore"}}


def _spec(role: str) -> dict[str, Any]:
    if role == "id":
        return {"sample_index": {"by": "id", "key": "id"}, "sources": [{"id": "t", "input": "d.csv", "key": "id", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}], "params": _COMMON}
    columns = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]
    return {"sources": [{"id": "t", "input": "d.csv", "columns": columns}], "params": _COMMON}


def _package(tmp_path: Path, token: str, peer: str, role: str, spec: dict[str, Any] | None = None) -> DatasetPackage:
    column = "id" if role == "id" else "m"
    (tmp_path / "d.csv").write_text(f"{column},x,y\n{token},1.5,1.0\n{peer},2.5,2.0\n")
    return nio.to_dataset_package(spec or _spec(role), base_dir=tmp_path)


def _column(package: DatasetPackage, name: str) -> tuple[str, list[Any]]:
    part = next(iter(package.partitions))
    payload = dict(package.partitions[part].payloads)[f"{part}/metadata"]
    assert isinstance(payload, MetadataTable)
    body = _metadata_body(payload.frame)
    column = next(c for c in body["columns"] if c["name"] == name)
    return column["dtype"], column["values"]


@pytest.mark.parametrize("role", ["meta", "id"])
@pytest.mark.parametrize(
    ("token", "peer", "expected"),
    [
        ("9007199254740993", "42", [N53, 42]),
        ("9007199254740993", "1.5", [N53, 1.5]),
        ("9007199254740993", "", [N53, None]),
        ("18446744073709551615", "42", [U64MAX, 42]),
        ("18446744073709551615", "1.5", [U64MAX, 1.5]),
        ("18446744073709551615", "", [U64MAX, None]),  # pandas can only read this column as text
    ],
)
def test_in_domain_integers_stay_exact_beside_every_peer(tmp_path, role: str, token: str, peer: str, expected: list[Any]) -> None:
    dtype, values = _column(_package(tmp_path, token, peer, role), "id" if role == "id" else "m")
    assert dtype == "numeric" and values == expected
    assert [type(v) for v in values] == [type(v) for v in expected], "ints stay ints and floats stay floats"


@pytest.mark.parametrize("role", ["meta", "id"])
@pytest.mark.parametrize("token", OUT_OF_DOMAIN)
@pytest.mark.parametrize("peer", ["42", "1.5", ""])
def test_out_of_domain_integers_refuse_for_selected_metadata_and_ids(tmp_path, role: str, token: str, peer: str) -> None:
    with pytest.raises(NumberDomainError, match="outside the supported range"):
        _package(tmp_path, token, peer, role)


@pytest.mark.parametrize(("token", "match"), [("1e-400", "underflows"), ("1e400", "overflows")])
def test_decimal_overflow_and_underflow_refuse_for_metadata(tmp_path, token: str, match: str) -> None:
    with pytest.raises(NumberDomainError, match=match):
        _package(tmp_path, token, "1.5", "meta")


@pytest.mark.parametrize(
    ("token", "peer", "expected"),
    [
        ("18014398509481984", "1.5", [18014398509481984.0, 1.5]),  # 2**54 is exact in binary64: the unchanged float column
        ("-9223372036854775808", "1.5", [-9223372036854775808.0, 1.5]),
        ("-0", "1.5", [-0.0, 1.5]),
        ("18014398509481984", "42", [18014398509481984, 42]),  # all integers: exact integers
        ("+0009007199254740993", "1.5", [N53, 1.5]),  # canonical integer, leading plus/zeros dropped
        ("+0009007199254740993", "42", [N53, 42]),
    ],
)
def test_controls_that_must_not_change(tmp_path, token: str, peer: str, expected: list[Any]) -> None:
    dtype, values = _column(_package(tmp_path, token, peer, "meta"), "m")
    assert dtype == "numeric" and values == expected and [type(v) for v in values] == [type(v) for v in expected]
    if token == "-0":
        assert str(values[0]) == "-0.0"


def test_two_distinct_inexact_ids_never_merge(tmp_path) -> None:
    (tmp_path / "d.csv").write_text("id,x,y\n9007199254740993,1.5,1.0\n9007199254740992,2.5,2.0\n1.5,3.5,3.0\n")
    _, values = _column(nio.to_dataset_package(_spec("id"), base_dir=tmp_path), "id")
    assert values == [N53, 9007199254740992, 1.5] and values[0] != values[1]


def test_refusal_applies_after_na_row_removal(tmp_path) -> None:
    spec = {
        "sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]}],
        "params": {"delimiter": ",", "na": {"policy": "remove_sample"}},
    }
    # the out-of-domain token sits in a row that NA removal drops (y is NA): the selected block never sees it
    (tmp_path / "d.csv").write_text("m,x,y\n18446744073709551616,1.5,NA\n7,2.5,2.0\n")
    assert _column(nio.to_dataset_package(spec, base_dir=tmp_path), "m") == ("numeric", [7])
    (tmp_path / "d.csv").write_text("m,x,y\n18446744073709551616,1.5,3.0\n7,2.5,2.0\n")
    with pytest.raises(NumberDomainError):
        nio.to_dataset_package(spec, base_dir=tmp_path)


def test_unselected_column_is_not_inspected(tmp_path) -> None:
    (tmp_path / "d.csv").write_text("m,x,y\n18446744073709551616,1.5,1.0\n7,2.5,2.0\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "ignore", "select": ["m"]}]}], "params": _COMMON}
    assert nio.to_dataset_package(spec, base_dir=tmp_path).partitions


@pytest.mark.parametrize("token", ["1e-400", "18446744073709551616", "340282366920938463463374607431768211456", "9007199254740993"])
def test_genuine_text_columns_keep_their_literal_strings_uncapped(tmp_path, token: str) -> None:
    for role in ("meta", "id"):
        package = _package(tmp_path, token, "sampleB", role)
        assert _column(package, "id" if role == "id" else "m") == ("string", [token, "sampleB"])


def test_numeric_only_barcode_above_u64_refuses_with_an_explicit_reason(tmp_path) -> None:
    with pytest.raises(NumberDomainError, match="numeric authority"):
        _package(tmp_path, "123456789012345678901234", "1", "meta")


def test_full_wire_roundtrip_keeps_exact_integers(tmp_path) -> None:
    package = _package(tmp_path, "9007199254740993", "1.5", "meta")
    wire = json.loads(canonical_json(package.to_full_dict()))
    restored = DatasetPackage.from_full_dict(wire)
    assert canonical_json(restored.to_full_dict()) == canonical_json(package.to_full_dict())
    assert _column(restored, "m") == ("numeric", [N53, 1.5])
    again = DatasetPackage.from_assembled(restored.to_assembled())
    assert again.manifest().root == package.manifest().root
    assembled_wire = json.loads(canonical_json(__import__("nirs4all_io.materialize.package", fromlist=["x"]).assembled_to_full_dict(package.to_assembled())))
    back = __import__("nirs4all_io.materialize.package", fromlist=["x"]).assembled_from_full_dict(assembled_wire)
    assert back.blocks["train"].metadata["m"].tolist() == [N53, 1.5]


def test_spreadsheet_text_cells_are_not_covered_by_the_csv_rule(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["m", "x", "y"])
    sheet.append(["18446744073709551616", 1.5, 1.0])  # a TEXT cell above 2**64
    sheet.append(["7", 2.5, 2.0])
    book.save(tmp_path / "d.xlsx")
    spec = {"sources": [{"id": "t", "input": "d.xlsx", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"na": {"policy": "ignore"}}}
    # the CSV refusal does not run: the spreadsheet path keeps whatever its reader produced
    assert nio.to_dataset_package(spec, base_dir=tmp_path).partitions


# --------------------------------------------------------------------------- #
# Classification once over ALL original rows; validation only on survivors      #
# --------------------------------------------------------------------------- #
def _same(expected: list[Any], actual: list[Any]) -> None:
    """Typed equality: ``3``, ``3.0`` and ``True`` are different values; floats compare by IEEE bits."""
    import struct

    assert len(expected) == len(actual), (expected, actual)
    for e, a in zip(expected, actual, strict=True):
        assert type(e) is type(a), f"{e!r} ({type(e).__name__}) vs {a!r} ({type(a).__name__})"
        if type(e) is float and e == e:
            assert struct.pack(">d", e) == struct.pack(">d", a), (e, a)
        elif type(e) is float:
            assert a != a
        else:
            assert e == a, (e, a)


def _removal_package(tmp_path: Path, cells: list[str], removed: list[bool], role: str, policy: str = "remove_sample") -> DatasetPackage:
    """A one-column CSV where a ``removed`` row has a missing Y (``remove_sample`` drops it, ``ignore`` keeps it)."""
    column = "id" if role == "id" else "m"
    rows = [f"{cell},{0.5 + index},{'' if gone else index + 1}" for index, (cell, gone) in enumerate(zip(cells, removed, strict=True))]
    (tmp_path / "d.csv").write_text(f"{column},x,y\n" + "\n".join(rows) + "\n")
    spec = _spec(role)
    spec["params"] = {"delimiter": ",", "na": {"policy": policy}}
    return nio.to_dataset_package(spec, base_dir=tmp_path)


_N54 = "18014398509481984"
_REMOVAL: dict[str, tuple[list[str], list[bool], list[Any], list[str] | None]] = {  # name: (cells, removed flags, surviving metadata cells, surviving sample ids)
    "exact-int-float-removed": (["3", "1.5"], [False, True], [3.0], ["3.0"]),
    "inexact-float-removed": (["9007199254740993", "1.5"], [False, True], [N53], ["9007199254740993"]),
    "int-text-removed": (["3", "text"], [False, True], ["3"], ["3"]),
    "NA-removed": (["", "3"], [True, False], [3.0], ["3.0"]),
    "A-inexact-removed": (["9007199254740993", "3", "1.5"], [True, False, False], [3, 1.5], ["3", "1.5"]),
    "B-outdomain-removed": (["18446744073709551616", "3", "1.5"], [True, False, False], [3, 1.5], ["3", "1.5"]),
    "E-inexact-survives-NA-removed": (["9007199254740993", "", "3"], [False, True, False], [N53, 3], ["9007199254740993", "3"]),
    "F-outdomain-removed-all-int": (["18446744073709551616", "3", "4"], [True, False, False], [3, 4], ["3", "4"]),
    "G-outdomain-text-peer-removed": (["18446744073709551616", "site"], [False, True], ["18446744073709551616"], ["18446744073709551616"]),
    "H-text-survives": (["18446744073709551616", "site"], [True, False], ["site"], ["site"]),
    "J-underflow-removed-int-survives": (["1e-400", "3"], [True, False], [3.0], ["3.0"]),
    "J2-overflow-removed-int-survives": (["1e400", "3"], [True, False], [3.0], ["3.0"]),
    "J3-underflow-removed-exact-ints": (["1e-400", "3", "4"], [True, False, False], [3.0, 4.0], ["3.0", "4.0"]),
    "J4-underflow-removed-inexact-int-survives": (["1e-400", "9007199254740993", "3"], [True, False, False], [N53, 3], ["9007199254740993", "3"]),
    "L-exact-2p54-float-removed": ([_N54, "1.5"], [False, True], [1.8014398509481984e16], None),
}


@pytest.mark.parametrize("role", ["meta", "id"])
@pytest.mark.parametrize("name", sorted(_REMOVAL))
def test_classification_is_fixed_over_original_rows_and_survivors_keep_their_type(tmp_path, role: str, name: str) -> None:
    cells, removed, values, ids = _REMOVAL[name]
    package = _removal_package(tmp_path, cells, removed, role)
    _dtype, held = _column(package, "id" if role == "id" else "m")
    _same(values, held)
    if role == "id":
        selected = [next(iter(package.partitions.values())).source_ids[0]]
        if ids is None:  # the survivor 1.8014398509481984e+16 spells '1.8014398509481984e+16': '+' is not an identifier byte
            from nirs4all_io.materialize.numeric import NumericContractError

            with pytest.raises(NumericContractError, match="Invalid primary identity"):
                package.executed_projection(selected)
        else:
            _same(ids, package.executed_projection(selected).identity.sample_ids)


@pytest.mark.parametrize("role", ["meta", "id"])
@pytest.mark.parametrize(
    ("cells", "removed", "match"),
    [
        (["18446744073709551616", "1.5"], [False, True], "outside the supported range"),  # I: the out-of-domain row survives
        (["1e-400", "1.5"], [False, True], "underflows"),  # K
        (["1e400", "1.5"], [False, True], "overflows"),
        (["9007199254740993", "1e-400"], [False, False], "underflows"),
    ],
)
def test_a_surviving_bad_cell_refuses(tmp_path, role: str, cells: list[str], removed: list[bool], match: str) -> None:
    """I / K: the bad cell's row survives, so it refuses (its removed twins never refuse: see the table above)."""
    with pytest.raises(NumberDomainError, match=match):
        _removal_package(tmp_path, cells, removed, role)


def test_ignored_na_keeps_its_row_and_the_identity_projection_refuses_the_whole_block(tmp_path) -> None:
    """NA kept under ``ignore``: the cell stays missing and the typed float peer stays float; the ID projection never trims rows."""
    from nirs4all_io.materialize.numeric import NumericContractError

    package = _removal_package(tmp_path, ["", "3"], [False, False], "meta", policy="ignore")
    _same([None, 3.0], _column(package, "m")[1])
    package = _removal_package(tmp_path, ["", "3"], [False, False], "id", policy="ignore")
    _same([None, 3.0], _column(package, "id")[1])
    with pytest.raises(NumericContractError, match="Missing projection identity"):
        package.executed_projection([next(iter(package.partitions.values())).source_ids[0]])
    assert next(iter(package.partitions.values())).n_samples == 2, "the package keeps both rows; nothing is trimmed"
