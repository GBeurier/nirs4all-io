# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Numeric text keeps its original lexemes until the NUMERIC ROLE conversion (class F, CSV and U07 metadata).

Pandas parses ``1e-400`` to ``0.0`` and rounds a large integer that shares a column with floats before any storage
selection can look. The CSV loader therefore keeps the original text, and the numeric-role conversion (features,
targets, weights, variations) runs every numeric-looking token through the shared checked primitives
(``strict_json.decimal_float`` / ``check_integer``). Metadata and categorical string fields are never converted.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io.dataset_facade import to_u07_raw_sources
from nirs4all_io.materialize.package import FeatureMatrix, TargetTable, WeightsVector
from nirs4all_io.public_content import dataset_content_bytes, metadata_number
from nirs4all_io.strict_json import NumberDomainError, decimal_float

N = 2**54 + 2**30 + 1


def _spec(**columns: Any) -> dict[str, Any]:
    return {"sources": [{"id": "table", "input": "data.csv", "columns": columns or {"features": "x", "targets": "y"}}]}


def _package(tmp_path: Path, csv: str, spec: dict[str, Any] | None = None, **options: Any):
    (tmp_path / "data.csv").write_text(csv, encoding="utf-8")
    return nio.to_dataset_package(spec or _spec(), base_dir=tmp_path, **options)


def _x_hex(package) -> str:
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload.matrix.values_bytes()[:8].hex()


# --------------------------------------------------------------------------- #
# The shared primitive                                                          #
# --------------------------------------------------------------------------- #
def test_the_shared_decimal_primitive() -> None:
    assert struct.pack(">d", decimal_float("-0.000e-999")).hex() == "8000000000000000"
    assert decimal_float("4.9e-324") == 5e-324 and decimal_float("1e-300") == 1e-300
    for token in ("1e-400", "2e-324", "-1e-400", "-2e-324"):
        with pytest.raises(NumberDomainError, match="underflows"):
            decimal_float(token)
    with pytest.raises(NumberDomainError, match="overflows"):
        decimal_float("1e400")


# --------------------------------------------------------------------------- #
# Actual CSV files through the public facade                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token", ["1e-400", "2e-324", "-1e-400", "-2e-324"])
def test_decimal_underflow_in_numeric_roles_refuses_before_a_payload_exists(tmp_path: Path, token: str) -> None:
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, f"x;y\n{token};1\n")  # features
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, f"x;y\n1;{token}\n")  # targets
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, f"x;y;w\n1;2;{token}\n", _spec(features="x", targets="y", weights="w"))
    (tmp_path / "data.csv").write_text(f"x;y\n{token};1\n", encoding="utf-8")
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.load(_spec(), target="assembled", base_dir=tmp_path)


def test_true_zero_subnormal_and_ordinary_values_are_kept(tmp_path: Path) -> None:
    assert _x_hex(_package(tmp_path, "x;y\n0.000e-999;1\n")) == "0000000000000000"
    assert _x_hex(_package(tmp_path, "x;y\n-0.000e-999;1\n")) == "0000000000000080", "the true zero keeps its sign"
    assert _x_hex(_package(tmp_path, "x;y\n-0e5;1\n")) == "0000000000000080"
    assert _x_hex(_package(tmp_path, "x;y\n4.9e-324;1\n")) == "0100000000000000", "the minimum subnormal is a real value"
    assert _x_hex(_package(tmp_path, "x;y\n1e-300;1\n")) == struct.pack("<d", 1e-300).hex()
    package = _package(tmp_path, "x;y\n0e5;1\n2.5;3\n")
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix) and payload.matrix.validity.ravel().tolist() == [True, True]


def test_a_genuine_host_zero_has_no_lexeme_to_inspect() -> None:
    package = nio.to_dataset_package((np.array([[0.0], [-0.0]]), np.array([1.0, 2.0])))
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix) and payload.matrix.values_bytes().hex() == "0000000000000000" + "0000000000000080"


def test_mixed_integer_and_float_column_keeps_its_integer_cells(tmp_path: Path) -> None:
    csv = "x;y\n9007199254740993;1\n0.5;2\n"
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, csv)  # default float64: 2**53 + 1 would be rounded
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, csv, features="float64")
    direct = _package(tmp_path, f"x;y\n{N};1\n0.5;2\n", features="float32")
    assert _x_hex(direct)[:8] == "0100805a", "requested float32 rounds the original integer directly (upper), not through float64"
    assert _package(tmp_path, "x;y\n18014398509481984;1\n0.5;2\n").partitions["train"].payloads, "an exact large integer is a valid float64"
    # an NA next to the integers must not hide them either
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package_na(tmp_path)


def _package_na(tmp_path: Path):
    (tmp_path / "data.csv").write_text("x;y\n9007199254740993;1\nNA;2\n", encoding="utf-8")
    spec = {"params": {"na": {"policy": "ignore"}}, **_spec()}
    return nio.to_dataset_package(spec, base_dir=tmp_path)


def test_integers_beyond_the_shared_domain_refuse_in_text(tmp_path: Path) -> None:
    with pytest.raises(NumberDomainError):
        _package(tmp_path, "x;y\n18446744073709551617;1\n0.5;2\n")


def test_pure_integer_columns_and_ordinary_files_are_unchanged(tmp_path: Path) -> None:
    package = _package(tmp_path, "x;y\n1;2\n3;4\n")
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix) and payload.matrix.values.ravel().tolist() == [1.0, 3.0] and payload.matrix.precision.source_dtype is None
    quoted = _package(tmp_path, 'x;y\n"1.5";"2"\n"2.5";"3"\n')
    assert dict(quoted.partitions["train"].payloads)["train/x0"].matrix.values.ravel().tolist() == [1.5, 2.5]
    comma = _package(tmp_path, "x;y\n1,5;2\n2,25;3\n", {"sources": [{"id": "table", "input": "data.csv", "columns": {"features": "x", "targets": "y"}, "params": {"decimal_separator": ","}}]})
    assert dict(comma.partitions["train"].payloads)["train/x0"].matrix.values.ravel().tolist() == [1.5, 2.25]
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "x;y\n1e-400;2\n", {"sources": [{"id": "table", "input": "data.csv", "columns": {"features": "x", "targets": "y"}, "params": {"decimal_separator": ","}}]})
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "x;y\n1,5e-400;2\n", {"sources": [{"id": "table", "input": "data.csv", "columns": {"features": "x", "targets": "y"}, "params": {"decimal_separator": ","}}]})


def test_metadata_and_class_strings_with_numeric_lookalikes_are_preserved(tmp_path: Path) -> None:
    csv = "sid;x;site;y\ns1;1.5;1e-400;red\ns2;2.5;2e-324;blue\ns3;3.5;north;red\n"
    spec = _spec(features="x", targets="y", metadata=["sid", "site"])
    package = _package(tmp_path, csv, {"sample_index": {"by": "id", "key": "sid"}, **spec})
    metadata = dict(package.partitions["train"].payloads)["train/metadata"].frame
    assert metadata["site"].tolist() == ["1e-400", "2e-324", "north"], "ordinary metadata strings are untouched"
    targets = dict(package.partitions["train"].payloads)["train/y"]
    assert isinstance(targets, TargetTable) and targets.columns[0].block.values.ravel().tolist() == ["red", "blue", "red"]
    # a metadata column whose every token is numeric is numeric authority (C3, tests/test_owner_corrective_c3.py):
    # the underflowing token refuses instead of staying a string; the genuine text column above stays literal
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "sid;x;site;y\ns1;1.5;1e-400;1\ns2;2.5;2;2\n", {"sample_index": {"by": "id", "key": "sid"}, **_spec(features="x", targets="y", metadata=["sid", "site"])})


def test_variations_and_multi_file_concatenation_share_the_check(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a;y\n1;1\n2;2\n", encoding="utf-8")
    (tmp_path / "v.csv").write_text("a\n1e-400\n2\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "x.csv", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}], "variations": [{"name": "v", "input": "v.csv"}], "params": {"delimiter": ";"}}]}
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.to_dataset_package(spec, base_dir=tmp_path)
    (tmp_path / "v.csv").write_text(f"a\n{N}\n2\n", encoding="utf-8")
    ok = nio.to_dataset_package(spec, base_dir=tmp_path, features="float32")
    payload = dict(ok.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix) and payload.processings[0][1].values_bytes()[:4].hex() == "0100805a"
    (tmp_path / "a.csv").write_text("x;y\n1;1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x;y\n1e-400;2\n", encoding="utf-8")
    merged = {"sources": [{"id": "m", "input": ["a.csv", "b.csv"], "merge": "concat_samples", "columns": {"features": "x", "targets": "y"}}]}
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.to_dataset_package(merged, base_dir=tmp_path)


def test_weights_follow_the_same_numeric_text_rules(tmp_path: Path) -> None:
    ok = _package(tmp_path, "x;y;w\n1;2;-0.000e-999\n3;4;0.5\n", _spec(features="x", targets="y", weights="w"))
    weights = dict(ok.partitions["train"].payloads)["train/weights"]
    assert isinstance(weights, WeightsVector) and weights.matrix.values_bytes()[:8].hex() == "0000000000000080"


# --------------------------------------------------------------------------- #
# U07 metadata numeric column (the other numeric-text caller)                    #
# --------------------------------------------------------------------------- #
_U07 = json.loads((Path(__file__).parent / "fixtures" / "runtime-u07-groups.json").read_text(encoding="utf-8"))


def _u07(numeric_cell: Any, label: Any = "A") -> dict[str, Any]:
    record = json.loads(json.dumps(_U07))
    source = next(s for s in record["dataset"]["sources"] if s["name"] == "metadata")
    source["array"]["values"][0] = [numeric_cell, label]
    return record


@pytest.mark.parametrize("token", ["1e-400", "2e-324", "-1e-400", "-2e-324"])
def test_u07_numeric_metadata_column_refuses_decimal_underflow(token: str) -> None:
    record = _u07(token)
    with pytest.raises(ValueError, match="underflows"):
        dataset_content_bytes(record)
    with pytest.raises(ValueError, match="incompatible value"):
        to_u07_raw_sources(record["dataset"])
    with pytest.raises(ValueError, match="underflows"):
        metadata_number(token)


def test_u07_genuine_zero_and_ordinary_numbers_keep_their_bytes() -> None:
    reference = dataset_content_bytes(_u07(0.0))
    assert dataset_content_bytes(_u07("0.000e-999")) == reference, "a genuine zero spelled as text is the same identity as 0.0 (zero folding)"
    assert dataset_content_bytes(_u07("-0e5")) == reference
    assert dataset_content_bytes(_u07("20.5")) == dataset_content_bytes(_u07(20.5))
    assert dataset_content_bytes(_u07("4.9e-324")) == dataset_content_bytes(_u07(5e-324)) != reference, "the minimum subnormal is a real value"
    assert dataset_content_bytes(_U07)  # the untouched fixture still hashes
    assert to_u07_raw_sources(_u07("0.000e-999")["dataset"])["sources"]["metadata"]["rows"][0][1] == "A"
    with pytest.raises(ValueError, match="exactly representable"):
        metadata_number("9007199254740993")
    with pytest.raises(ValueError, match="decimal"):
        metadata_number("0x10")


def test_u07_categorical_string_with_a_numeric_lookalike_is_preserved() -> None:
    record = _u07(20.5, "1e-400")
    assert b"1e-400" in dataset_content_bytes(record), "the categorical column is an ordinary string field"
    assert to_u07_raw_sources(record["dataset"])["sources"]["metadata"]["rows"][0][1] == "1e-400"
    # a host number in the numeric column is already decoded: there is no lexeme to inspect
    assert dataset_content_bytes(_u07(5e-324)) != dataset_content_bytes(_u07(0.0))


# --------------------------------------------------------------------------- #
# U07 numeric column: integer-looking tokens stay integers until the exact gate  #
# --------------------------------------------------------------------------- #
def test_u07_integer_text_is_never_rounded_to_a_float() -> None:
    assert metadata_number("18014398509481984") == float(2**54), "an exact integer string is a valid float64"
    assert struct.pack(">d", metadata_number("9007199254740992.0")).hex() == "4340000000000000", "a float string keeps its parsed binary64 bits"
    for token in ("9007199254740993", "18014399583223809", "-9007199254740993", "18446744073709551615"):
        with pytest.raises(ValueError, match="exactly representable"):
            metadata_number(token)
    for token in ("18446744073709551616", "18446744073709551617", "123456789012345678901234567890", "-9223372036854775809"):
        with pytest.raises(NumberDomainError, match="outside the supported range"):
            metadata_number(token)
    assert struct.pack(">d", metadata_number("-0")).hex() == "8000000000000000", "a true -0 keeps its sign"
    assert metadata_number("-9223372036854775808") == -(2.0**63)
    for token, expected in (("+007", 7.0), ("1.5e+3", 1500.0), (".5", 0.5), ("5.", 5.0), (" 12 ", 12.0), ("007.50", 7.5)):
        assert metadata_number(token) == expected, "the explicit decimal grammar (sign, zeros, point, exponent) is unchanged"
    for token in ("0x10", "1_000", "1e", "--1", "٣", "1,5"):
        with pytest.raises(ValueError, match="decimal"):
            metadata_number(token)


def test_u07_whole_identity_uses_the_exact_integer_gate_on_the_actual_numeric_column() -> None:
    # (a) the frozen bug: the integer token used to be rounded to the same bits as a legitimate float string
    legitimate = dataset_content_bytes(_u07("9007199254740992.0"))
    with pytest.raises(ValueError, match="exactly representable"):
        dataset_content_bytes(_u07("9007199254740993"))
    with pytest.raises(ValueError, match="incompatible value"):
        to_u07_raw_sources(_u07("9007199254740993")["dataset"])
    assert dataset_content_bytes(_u07("9007199254740992")) == legitimate, "the exact integer 2**53 is the same value as 2**53.0"
    # (b) 2**54 is a valid float64 as TEXT; a host number in the object-dtype column keeps the public object cap
    assert dataset_content_bytes(_u07("18014398509481984")) != legitimate
    assert dataset_content_bytes(_u07("18014398509481984")) == dataset_content_bytes(_u07("1.8014398509481984e16"))
    with pytest.raises(ValueError, match="JavaScript"):
        dataset_content_bytes(_u07(float(2**54)))
    # (c) a true zero keeps its sign into the zero folding: '-0' and '-0e5' hash like 0.0
    zero = dataset_content_bytes(_u07(0.0))
    assert dataset_content_bytes(_u07("-0")) == dataset_content_bytes(_u07("0")) == dataset_content_bytes(_u07("-0e5")) == zero
    # (d) outside the shared integer domain: refused even when the number happens to be a float64
    with pytest.raises(NumberDomainError, match="outside the supported range"):
        dataset_content_bytes(_u07("18446744073709551616"))
    # (e) ordinary metadata strings stay literal, and the categorical column is not a numeric field
    record = _u07("20.5", "9007199254740993")
    assert b"9007199254740993" in dataset_content_bytes(record)
    assert to_u07_raw_sources(record["dataset"])["sources"]["metadata"]["rows"][0][1] == "9007199254740993"
