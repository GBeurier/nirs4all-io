# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Original cells, physical CSV width, and typed relational identity at public ingestion."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.join import join_tables, merge_by_key
from nirs4all_io.materialize.loaders import NAError, NumericRefusal
from nirs4all_io.spec.enums import Cardinality, Coverage, SpecError
from nirs4all_io.strict_json import loads as strict_loads


def _package(tmp_path: Path, files: dict[str, str], spec: dict[str, Any], **storage: Any) -> Any:
    for name, text in files.items():
        (tmp_path / name).write_bytes(text.encode())
    return nio.to_dataset_package(spec, base_dir=tmp_path, **storage)


def _one(role: str = "features", **params: Any) -> dict[str, Any]:
    columns = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]
    if role == "weights":
        columns.append({"role": "weights", "select": ["m"]})
    return {"sources": [{"id": "t", "input": "d.csv", "columns": columns}], "params": {"delimiter": ",", "na": {"policy": "ignore"}, **params}}


def _matrix(package: Any, role: str = "x0") -> dict[str, Any]:
    payload = package.to_full_dict()["partitions"]["train"]["payloads"][f"train/{role}"]["value"]
    return payload["columns"][0]["block"] if role == "y" else payload["matrix"]


@pytest.mark.parametrize(("token", "bits"), [("0.12345678901234567", "3fbf9add3746f65e"), ("-0.000e-999", "8000000000000000"), ("+.5", "3fe0000000000000"), ("5.", "4014000000000000"), ("1E5", "40f86a0000000000")])
def test_text_peer_numeric_cells_keep_checked_bits(tmp_path: Path, token: str, bits: str) -> None:
    block = _matrix(_package(tmp_path, {"d.csv": f"x,y\n{token},1.0\nabc,2.0\n1.5,3.0\n"}, _one()))
    assert block["validity"] == [True, False, True]
    assert block["data_bytes_hex"] == bytes.fromhex(bits)[::-1].hex() + "0000000000000000" + "000000000000f83f"
    assert block["precision"] == {"dtype": "float64", "origin": "default", "source_dtype": None, "requested_dtype": None, "consumer": None}


@pytest.mark.parametrize("token", ["9007199254740993", "18446744073709551616", "1e-999", "1e999", "inf", "INFINITY"])
@pytest.mark.parametrize("role", ["features", "targets", "weights"])
def test_numeric_roles_never_accept_lost_domain_evidence(tmp_path: Path, token: str, role: str) -> None:
    column = {"features": "x", "targets": "y", "weights": "m"}[role]
    rows = [{"x": "1.5", "y": "1.0", "m": "1.0"}, {"x": "2.5", "y": "2.0", "m": "2.0"}]
    rows[0][column], rows[1][column] = token, "abc"
    text = "x,y,m\n" + "".join(",".join(row[name] for name in ("x", "y", "m")) + "\n" for row in rows)
    spec = _one("weights", categorical="none")
    with pytest.raises((ValueError, SpecError)):
        _package(tmp_path, {"d.csv": text}, spec)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("storage", [None, "float32"])
def test_retained_integer_survives_text_concat_before_storage_selection(tmp_path: Path, reverse: bool, storage: str | None) -> None:
    files = {"a.csv": "x,y\nabc,1.0\n1.5,2.0\n", "b.csv": "x,y\n9007199254740993,3.0\n2.5,4.0\n"}
    spec = _one()
    spec["sources"][0].update(input=["b.csv", "a.csv"] if reverse else list(files), merge="concat_samples")
    if storage is None:
        with pytest.raises((ValueError, SpecError)):
            _package(tmp_path, files, spec)
    else:
        package = _package(tmp_path, files, spec, features=storage)
        block = _matrix(package)
        observed = list(struct.iter_unpack("<f", bytes.fromhex(block["data_bytes_hex"])))
        assert observed[0 if reverse else 2] == (9007199254740992.0,)
        assert block["validity"] == ([True, True, False, True] if reverse else [False, True, True, True])
        assert block["precision"]["requested_dtype"] == "float32"


@pytest.mark.parametrize("text", ["x,y,m\n0.5,1.0,7,9\n1.5,2.0,8\n", "x,y,m\n1,2,3,\n4,5,6,\n", "x,y,m\n1,2,3\n4,5,6,\n", "x,y,m\n1,2,3\n4,5,6,7\n"])
@pytest.mark.parametrize("rest", [False, True])
def test_physical_csv_width_refuses_before_column_selection(tmp_path: Path, text: str, rest: bool) -> None:
    spec = _one()
    spec["sources"][0]["columns"].append({"role": "metadata", "select": "rest" if rest else ["m"]})
    with pytest.raises(SpecError, match=r"Expected \d+ fields"):
        _package(tmp_path, {"d.csv": text}, spec)


def _lookup(coverage: str = "drop", metadata: bool = True) -> dict[str, Any]:
    return {"sources": [
        {"id": "a", "role": "mixed", "input": "a.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata" if metadata else "ignore", "select": ["k"]}]},
        {"id": "s", "kind": "lookup", "input": "s.csv", "columns": [{"role": "targets", "select": ["r"]}], "join": {"to": "a", "left_on": "k", "right_on": "k", "how": "m:1", "coverage": coverage}},
    ], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}


@pytest.mark.parametrize(("left", "right"), [("9007199254740992", "9007199254740993"), ("0", "1e-400"), ("9007199254740993", "9007199254740992")])
def test_all_key_operands_retain_exact_authority(tmp_path: Path, left: str, right: str) -> None:
    package = _package(tmp_path, {"a.csv": f"k,x\n{left},1.5\n1.5,2.5\n", "s.csv": f"k,r\n{right},0.5\n1.5,0.25\n"}, _lookup(metadata=False))
    assert package.partitions["train"].n_samples == 1
    assert _matrix(package)["data_bytes_hex"] == "0000000000000440"
    assert _matrix(package, "y")["data_bytes_hex"] == "000000000000d03f"


@pytest.mark.parametrize(("left", "right"), [("TRUE", "1"), ("", "")])
def test_boolean_and_missing_keys_never_match_integer_or_missing(tmp_path: Path, left: str, right: str) -> None:
    package = _package(tmp_path, {"a.csv": f"k,x\n{left},1.5\nA,2.5\n", "s.csv": f"k,r\n{right},0.5\nA,0.25\n"}, _lookup("warn"))
    assert _matrix(package, "y")["validity"] == [False, True]


@pytest.mark.parametrize("separator", [",", ":"])
def test_numeric_decimal_view_does_not_rewrite_metadata_or_category_lexemes(tmp_path: Path, separator: str) -> None:
    text = f"x;y;m\n1{separator}5;0{separator}125;0{separator}125\nabc;abc;abc\n"
    spec = _one(delimiter=";", decimal=separator)
    spec["sources"][0]["columns"].append({"role": "metadata", "select": ["m"]})
    package = _package(tmp_path, {"d.csv": text}, spec)
    wire = package.to_full_dict()["partitions"]["train"]["payloads"]
    assert _matrix(package)["data_bytes_hex"] == "000000000000f83f0000000000000000"
    assert _matrix(package)["validity"] == [True, False]
    assert wire["train/metadata"]["value"]["columns"][0]["values"] == [f"0{separator}125", "abc"]
    assert wire["train/y"]["value"]["categorical"]["y"]["categories"] == [f"0{separator}125", "abc"]


@pytest.mark.parametrize("reverse", [False, True])
def test_key_restoration_keeps_ordinary_concat_numeric_promotion(tmp_path: Path, reverse: bool) -> None:
    files = {"a.csv": "id,x,y\n3,1.5,1.0\n4,2.5,2.0\n", "b.csv": "id,x,y\n1.5,3.5,3.0\n2.5,4.5,4.0\n"}
    spec = _one()
    spec["sample_index"] = {"by": "id", "key": "id"}
    spec["sources"][0].update(input=["b.csv", "a.csv"] if reverse else list(files), merge="concat_samples", key="id")
    package = _package(tmp_path, files, spec)
    assert package.executed_projection(["t"]).identity.sample_ids == (["1.5", "2.5", "3.0", "4.0"] if reverse else ["3.0", "4.0", "1.5", "2.5"])


def _strict_equal(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected), (actual, expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _strict_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for value, wanted in zip(actual, expected, strict=True):
            _strict_equal(value, wanted)
    elif isinstance(expected, float):
        assert struct.pack(">d", actual) == struct.pack(">d", expected)
    else:
        assert actual == expected


def _pointer(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    assert pointer.startswith("/")
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _refusal_kind(error: Exception) -> tuple[str, str]:
    message = str(error)
    if type(error) is pd.errors.EmptyDataError and message == "No columns to parse from file":
        return "empty_data", "reader"
    if type(error) is SpecError and re.fullmatch(r"source '[^']+' \(\d+ rows\) is not row-aligned with '[^']+' \(\d+ rows\); supply a join key", message):
        return "alignment", "alignment"
    if type(error) is NAError and re.fullmatch(r"NA values detected and na_policy is 'abort'\. First NA in column '.+' \(row: \d+\)\.", message):
        return "na_policy", "na_policy"
    text = message.lower()
    if "expected " in text and " fields" in text:
        return "csv_width", "reader"
    if "coverage '" in text:
        return "join_coverage", "coverage"
    if "duplicate key" in text:
        return "join_cardinality", "cardinality"
    if any(token in text for token in ("exact", "out of", "outside", "overflow", "underflow", "rounds to zero", "finite", "nonzero numeric")):
        return "numeric_domain", "selected_numeric"
    if "source" in text and "dtype" in text:
        return "source_declaration", "selected_numeric"
    if "selector" in text or "unassigned" in text:
        return "invalid_selector", "selected_numeric"
    if "no numeric feature" in text or "no feature" in text:
        return "no_numeric_features", "selected_numeric"
    raise AssertionError(f"Unclassified owner refusal: {type(error).__name__}: {error}")


def _shared_cases() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "fixtures" / "residual_ingestion_vectors.json"
    if not path.exists():
        return [{"id": "shared-bank-not-published"}]
    bank = json.loads(path.read_text(encoding="utf-8"))
    assert set(bank) == {"schema", "cases"}
    assert bank["schema"] == "nirs4all.io.owner-review-vectors.v1"
    assert len({case["id"] for case in bank["cases"]}) == len(bank["cases"])
    assert all(case["hosts"] and set(case["hosts"]) <= {"python", "rust"} for case in bank["cases"])
    return [case for case in bank["cases"] if "python" in case["hosts"]]


@pytest.mark.parametrize("case", _shared_cases(), ids=lambda case: case["id"])
def test_shared_residual_ingestion_vectors(tmp_path: Path, case: dict[str, Any]) -> None:
    """Consume the sole writer's exact physical inputs and independent typed expectations."""
    assert case["id"] != "shared-bank-not-published", "Shared residual-ingestion bank is not yet published"
    assert set(case) <= {"id", "condition", "operation", "hosts", "input", "expected", "storage"}
    assert isinstance(case["condition"], str)
    assert case["operation"] == "file_package"
    assert set(case["hosts"]) <= {"python", "rust"} and "python" in case["hosts"]
    incoming = case["input"]
    assert set(incoming) == {"spec_text", "files"}
    for name, carrier in incoming["files"].items():
        destination = tmp_path / name
        assert destination.is_relative_to(tmp_path) and not Path(name).is_absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(carrier, str):
            content = carrier.encode("utf-8")
        else:
            assert set(carrier) == {"format", "hex", "sha256"} and carrier["format"] == "binary"
            content = bytes.fromhex(carrier["hex"])
            assert hashlib.sha256(content).hexdigest() == carrier["sha256"]
        destination.write_bytes(content)
    expected = case["expected"]
    assert set(expected) <= {"outcome", "category", "assertions", "package_assertions", "projection", "refusal_stage", "data_bytes_hex", "validity", "precision", "metadata"}
    assert expected["outcome"] in {"accept", "refuse"}
    spec = strict_loads(incoming["spec_text"])
    if expected["outcome"] == "refuse":
        assert "projection" not in expected
        assert not set(expected) - {"outcome", "category", "refusal_stage"}
        with pytest.raises((ValueError, SpecError)) as caught:
            nio.to_dataset_package(spec, base_dir=tmp_path, **case.get("storage", {}))
        category, stage = _refusal_kind(caught.value)
        assert category == expected["category"], str(caught.value)
        assert stage == expected["refusal_stage"], str(caught.value)
        return
    assert "refusal_stage" not in expected
    assert expected["category"] == "accepted"
    package = nio.to_dataset_package(spec, base_dir=tmp_path, **case.get("storage", {}))
    wire, assembled = package.to_full_dict(), package.to_assembled().to_full_dict()
    assert expected.get("assertions") or expected.get("package_assertions") or set(expected) & {"data_bytes_hex", "validity", "precision", "metadata"}
    for name, actual in [("assertions", assembled), ("package_assertions", wire)]:
        for assertion in expected.get(name, []):
            assert set(assertion) == {"pointer", "value"}
            _strict_equal(_pointer(actual, assertion["pointer"]), assertion["value"])
    for name in ("data_bytes_hex", "validity", "precision"):
        if name in expected:
            _strict_equal(_matrix(package)[name], expected[name])
    if "metadata" in expected:
        _strict_equal(assembled["blocks"]["train"]["metadata"], expected["metadata"])
    if "projection" in expected:
        projection = expected["projection"]
        sources = list(dict.fromkeys(source for partition in package.partitions.values() for source in partition.source_ids))
        if projection["outcome"] == "accept":
            assert set(projection) == {"outcome", "sample_ids"}
            _strict_equal(package.executed_projection(sources).identity.sample_ids, projection["sample_ids"])
        else:
            assert projection["outcome"] == "refuse" and set(projection) == {"outcome", "category"}
            with pytest.raises(ValueError) as caught_projection:
                package.executed_projection(sources)
            text = str(caught_projection.value)
            if text.startswith("Invalid primary identity"):
                category = "invalid_primary_identity"
            elif text.startswith("Missing projection identity"):
                category = "missing_identity"
            elif text == "Projection requires stable sample identity":
                category = "unstable_identity"
            else:
                raise AssertionError(f"Unclassified projection refusal: {text}") from caught_projection.value
            assert category == projection["category"], text


def _assert_missing_rows(package: Any, x: list[float | None], y: list[float | None]) -> None:
    assert package.partitions["train"].n_samples == len(x) == len(y)
    for role, values in [("x0", x), ("y", y)]:
        block = _matrix(package, role)
        assert block["dtype"] == "float64" and block["n_rows"] == len(values)
        assert block["validity"] == [value is not None for value in values]
        assert block["data_bytes_hex"] == b"".join(struct.pack("<d", 0.0 if value is None else value) for value in values).hex()


@pytest.mark.parametrize(
    ("name", "text", "x", "y"),
    [
        ("first-blank", "x,y\n\n1,2\n", [None, 1.0], [None, 2.0]),
        ("two-first", "x,y\n\n\n1,2\n", [None, None, 1.0], [None, None, 2.0]),
        ("short-next", "x,y\n\n1\n", [None, 1.0], [None, None]),
        ("crlf-first", "x,y\r\n\r\n1,2\r\n", [None, 1.0], [None, 2.0]),
        ("cr-first", "x,y\r\r1,2\r", [None, 1.0], [None, 2.0]),
        ("first-then-more", "x,y\n\n1,2\n3,4\n", [None, 1.0, 3.0], [None, 2.0, 4.0]),
        ("mid", "x,y\n1,2\n\n3,4\n", [1.0, None, 3.0], [2.0, None, 4.0]),
        ("trail", "x,y\n1,2\n\n", [1.0, None], [2.0, None]),
        ("only-blank", "x,y\n\n", [None], [None]),
        ("quoted-first", 'x,y\n""\n1,2\n', [None, 1.0], [None, 2.0]),
        ("quoted2-first", 'x,y\n"",""\n1,2\n', [None, 1.0], [None, 2.0]),
        ("comma-first", "x,y\n,\n1,2\n", [None, 1.0], [None, 2.0]),
        ("space-first", "x,y\n \n1,2\n", [None, 1.0], [None, 2.0]),
        ("one-col-first", "y,x\n\n2,1\n", [None, 1.0], [None, 2.0]),
    ],
)
def test_decided_empty_records_preserve_observation_positions(tmp_path: Path, name: str, text: str, x: list[float | None], y: list[float | None]) -> None:
    package = _package(tmp_path, {"d.csv": text}, _one())
    _assert_missing_rows(package, x, y)
    if name == "first-blank":
        assert package.manifest().root == "d6993aca33f08a1bc79406b7b0bd5ee513bed0fa46f5dc97e1bad4b0ebbaa124"


def test_decided_blank_decimal_observation_preserves_locale(tmp_path: Path) -> None:
    package = _package(tmp_path, {"d.csv": "x;y\n\n1,5;2,5\n"}, _one(delimiter=";", decimal=","))
    _assert_missing_rows(package, [None, 1.5], [None, 2.5])


@pytest.mark.parametrize("policy", ["abort", "remove_sample", "replace"])
def test_decided_first_blank_obeys_existing_na_policy(tmp_path: Path, policy: str) -> None:
    text = "x,y\n\n1,2\n3,4\n" if policy == "replace" else "x,y\n\n1,2\n"
    spec = _one(na={"policy": policy, **({"fill": {"method": "mean"}} if policy == "replace" else {})})
    if policy == "abort":
        with pytest.raises(SpecError, match="na_policy is 'abort'"):
            _package(tmp_path, {"d.csv": text}, spec)
    else:
        package = _package(tmp_path, {"d.csv": text}, spec)
        _assert_missing_rows(package, [2.0, 1.0, 3.0] if policy == "replace" else [1.0], [None, 2.0, 4.0] if policy == "replace" else [2.0])


@pytest.mark.parametrize(("text", "count"), [("id,x,y\n\nA,1,2\n", 2), ("id,x,y\nA,1,2\n\nB,3,4\n", 3)])
def test_decided_missing_identity_row_is_preserved_then_projection_refuses(tmp_path: Path, text: str, count: int) -> None:
    spec = _one()
    spec["sample_index"] = {"by": "id", "key": "id"}
    spec["sources"][0]["key"] = "id"
    package = _package(tmp_path, {"d.csv": text}, spec)
    assert package.partitions["train"].n_samples == count
    with pytest.raises(ValueError, match="Missing projection identity"):
        package.executed_projection(["t"])


@pytest.mark.parametrize("extra_blank", [False, True])
def test_decided_target_alignment_uses_physical_positions(tmp_path: Path, extra_blank: bool) -> None:
    spec = {"sources": [{"id": "x", "role": "features", "input": "x.csv", "columns": {"features": "x"}}, {"id": "y", "role": "targets", "input": "y.csv", "columns": {"targets": "y"}}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    files = {"x.csv": "x\n1.5\n2.5\n3.5\n", "y.csv": "y\n\n20\n30\n" + ("\n" if extra_blank else "")}
    if extra_blank:
        with pytest.raises(SpecError, match="not row-aligned"):
            _package(tmp_path, files, spec)
    else:
        _assert_missing_rows(_package(tmp_path, files, spec), [1.5, 2.5, 3.5], [None, 20.0, 30.0])


def test_decided_headerless_middle_blank_checks_only_data_and_masks(tmp_path: Path) -> None:
    spec = _one(has_header=False)
    spec["sources"][0]["columns"] = {"features": "0", "targets": "1"}
    _assert_missing_rows(_package(tmp_path, {"d.csv": "1,2\n\n3,4\n"}, spec), [1.0, None, 3.0], [2.0, None, 4.0])


@pytest.mark.parametrize(("text", "header"), [("\n\nx,y\n1,2\n", True), ("\nx,y\n1,2\n", True), ("\n1,2\n3,4\n", False)])
def test_leading_blank_controls_keep_explicit_python_empty_data_refusal(tmp_path: Path, text: str, header: bool) -> None:
    spec = _one(has_header=header)
    if not header:
        spec["sources"][0]["columns"] = {"features": "0", "targets": "1"}
    with pytest.raises(pd.errors.EmptyDataError, match="No columns to parse from file"):
        _package(tmp_path, {"d.csv": text}, spec)


@pytest.mark.parametrize("text", ["x,y,m\n1,2,3\n4,5,6,7\n", "x,y,m\n1,2,3\n4,5,6,\n", "x,y,m\n1,2,3,\n4,5,6,\n", "x,y\n\n1,2,3\n"])
def test_decided_width_guard_keeps_its_index_inference_independent(tmp_path: Path, text: str) -> None:
    spec = _one()
    spec["sources"][0]["columns"].append({"role": "metadata", "select": "rest"})
    with pytest.raises(SpecError, match=r"Expected \d+ fields"):
        _package(tmp_path, {"d.csv": text}, spec)


@pytest.mark.parametrize(("right_key", "expected_public_keys", "matched"), [(1, [True, "C", "keep"], 1), ("keep", [True, 1, "C"], 1)])
def test_public_join_audit_keeps_original_scalar_kinds(tmp_path: Path, right_key: Any, expected_public_keys: list[Any], matched: int) -> None:
    left = pd.DataFrame({"k": pd.Series([True, 1, "C", "keep"], dtype=object), "x": [10, 20, 30, 40]})
    right = pd.DataFrame({"k": pd.Series([right_key], dtype=object), "y": [99]})
    result, audit = join_tables(left, right, left_on="k", right_on="k", cardinality=Cardinality.MANY_TO_ONE, coverage=Coverage.DROP)
    _strict_equal([row["key"] for row in audit.dropped_rows], expected_public_keys)
    assert len(result) == audit.n_matched == matched
    assert result["x"].tolist() == ([20] if type(right_key) is int else [40])
    assert all(row["reason"] == "unmatched_left" for row in audit.dropped_rows)


def test_public_join_audit_keeps_compound_keys_while_matching_typed_cells() -> None:
    left = pd.DataFrame({"a": ["A", "A", "B"], "b": pd.Series([True, 1, 3.5], dtype=object), "x": [10, 20, 30]})
    right = pd.DataFrame({"a": ["A"], "b": [1], "y": [99]})
    result, audit = join_tables(left, right, left_on=["a", "b"], right_on=["a", "b"], cardinality=Cardinality.MANY_TO_ONE, coverage=Coverage.DROP)
    assert result["x"].tolist() == [20]
    keys = [row["key"] for row in audit.dropped_rows]
    assert keys == [("A", True), ("B", 3.5)]
    assert type(keys[0]) is tuple and type(keys[0][1]) is bool and type(keys[1][1]) is float


def test_public_merge_audit_propagates_scalar_keys_and_numeric_token_text() -> None:
    marker = NumericRefusal("k", " 1e-400", "Nonzero numeric text rounds to zero", False, " 1e-400")
    left = pd.DataFrame({"k": pd.Series([marker, "A", "C"], dtype=object), "x": [1, 2, 3]})
    middle = pd.DataFrame({"k": ["A"], "y": [4]})
    right = pd.DataFrame({"k": ["A"], "z": [5]})
    result, audit = merge_by_key([left, middle, right], ["left", "middle", "right"], key="k", coverage=Coverage.DROP)
    assert result["k"].tolist() == ["A"]
    _strict_equal(audit.dropped_rows, [{"key": " 1e-400", "reason": "unmatched_left"}, {"key": "C", "reason": "unmatched_left"}])


@pytest.mark.parametrize("first", [" ", "\t", '" "', '""'])
@pytest.mark.parametrize("header", [True, False])
def test_work_width_guard_rejects_narrow_inferred_first_record(tmp_path: Path, first: str, header: bool) -> None:
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": "rest"}] if header else {"features": "0"}}], "params": {"delimiter": ",", "has_header": header, "na": {"policy": "ignore"}}}
    with pytest.raises((SpecError, pd.errors.ParserError), match=r"Expected \d+ fields"):
        _package(tmp_path, {"d.csv": first + "\n1,2\n3,4\n"}, spec)


def test_work_width_guard_preserves_explicit_usecols(tmp_path: Path) -> None:
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": {"features": "x"}}], "params": {"delimiter": ",", "format": {"usecols": ["x"]}, "na": {"policy": "ignore"}}}
    block = _matrix(_package(tmp_path, {"d.csv": "x,y\n1,2\n3,4\n"}, spec))
    assert block["data_bytes_hex"] == "000000000000f03f0000000000000840"
    assert block["validity"] == [True, True]


@pytest.mark.parametrize("join_to", ["a", "s1"])
def test_work_chained_key_restoration_follows_column_origin(tmp_path: Path, join_to: str) -> None:
    files = {"a.csv": "k1,x\nK,1.5\nL,2.5\n", "s1.csv": "k1,k2\nK,9007199254740993\nL,1.5\n", "s2.csv": "k2,r\n9007199254740992,0.5\n1.5,0.25\n"}
    spec = {"sources": [
        {"id": "a", "role": "mixed", "input": "a.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "ignore", "select": ["k1"]}]},
        {"id": "s1", "kind": "lookup", "input": "s1.csv", "columns": [{"role": "ignore", "select": ["k2"]}], "join": {"to": "a", "left_on": "k1", "right_on": "k1", "how": "m:1", "coverage": "warn"}},
        {"id": "s2", "kind": "lookup", "input": "s2.csv", "columns": [{"role": "targets", "select": ["r"]}], "join": {"to": join_to, "left_on": "k2", "right_on": "k2", "how": "m:1", "coverage": "warn"}},
    ], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = _package(tmp_path, files, spec)
    block = _matrix(package, "y")
    assert block["validity"] == [False, True]
    assert block["data_bytes_hex"] == "0000000000000000000000000000d03f"
    assert package.partitions["train"].n_samples == 2


@pytest.mark.parametrize(("left", "right", "valid"), [("inf", "Infinity", False), ("inf", "inf", True), ("-inf", "-inf", True)])
def test_work_nonfinite_text_keys_compare_original_lexemes(tmp_path: Path, left: str, right: str, valid: bool) -> None:
    package = _package(tmp_path, {"a.csv": f"k,x\n{left},1.5\n2.5,2.5\n", "s.csv": f"k,r\n{right},0.5\n2.5,0.25\n"}, _lookup("warn", metadata=False))
    block = _matrix(package, "y")
    assert block["validity"] == [valid, True]
    assert block["data_bytes_hex"] == ("000000000000e03f" if valid else "0000000000000000") + "000000000000d03f"


@pytest.mark.parametrize("role", ["features", "targets"])
@pytest.mark.parametrize("storage", ["float64", "float32"])
@pytest.mark.parametrize("token", ["-0", "+0"])
def test_work_negative_integer_zero_text_preserves_selected_bits(tmp_path: Path, role: str, storage: str, token: str) -> None:
    files = {"d.csv": f"x,y\n{token},1.0\nabc,2.0\n1.5,3.0\n" if role == "features" else f"x,y\n1.5,{token}\n2.5,abc\n3.5,1.5\n"}
    package = _package(tmp_path, files, _one(categorical="none"), **{role: storage})
    block = _matrix(package, "x0" if role == "features" else "y")
    assert block["dtype"] == storage
    assert block["validity"] == [True, False, True]
    size = 8 if storage == "float64" else 4
    assert bytes.fromhex(block["data_bytes_hex"])[:size] == struct.pack("<d" if size == 8 else "<f", -0.0 if token.startswith("-") else 0.0)


def test_work_negative_zero_text_keeps_metadata_and_category_labels(tmp_path: Path) -> None:
    spec = _one()
    spec["sources"][0]["columns"].append({"role": "metadata", "select": ["m"]})
    package = _package(tmp_path, {"d.csv": "x,y,m\n1.5,-0,-0\n2.5,abc,abc\n"}, spec)
    payloads = package.to_full_dict()["partitions"]["train"]["payloads"]
    assert payloads["train/y"]["value"]["categorical"]["y"]["categories"] == ["-0", "abc"]
    _strict_equal(payloads["train/metadata"]["value"]["columns"][0]["values"], ["-0", "abc"])


@pytest.mark.parametrize("token", ["1e999", "18446744073709551616"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("storage", ["float64", "float32"])
def test_work_variation_refuses_only_selected_numeric_columns(tmp_path: Path, token: str, selected: bool, storage: str) -> None:
    spec = _one()
    spec["sources"][0]["variations"] = [{"name": "v", "input": "v.csv"}]
    files = {"d.csv": "x,y\n1,1\n2,2\n", "v.csv": f"x,q\n{token if selected else '1.5'},{'1.5' if selected else token}\n2.5,3\n"}
    if selected:
        with pytest.raises(ValueError, match="column 'x'"):
            _package(tmp_path, files, spec, features=storage)
    else:
        package = _package(tmp_path, files, spec, features=storage)
        processings = package.to_assembled().blocks["train"].processings[0]
        assert [name for name, _ in processings] == ["v"]
        assert processings[0][1].tolist() == [[1.5], [2.5]]


@pytest.mark.parametrize(("peer", "lookup", "matched"), [
    ("9007199254740993", "9007199254740992", False),
    ("9007199254740993", "9007199254740993", True),
    ("18446744073709551615", "18446744073709551615", True),
    ("18446744073709551616", "18446744073709551616", True),
    ("1e-400", "0", False),
    ("1e999", "1e999", True),
    ("3", "3", True),
    ("2.5", "2.5", True),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_residual_nonfinite_key_peer_uses_checked_raw_token(tmp_path: Path, peer: str, lookup: str, matched: bool, reverse: bool) -> None:
    files = {"a.csv": f"k,x\ninf,1.5\n{peer},2.5\n", "s.csv": f"k,r\n{lookup},0.5\n"}
    if reverse:
        files = {"a.csv": f"k,x\n{lookup},2.5\n", "s.csv": f"k,r\ninf,0.25\n{peer},0.5\n"}
    block = _matrix(_package(tmp_path, files, _lookup("warn", metadata=False)), "y")
    assert block["validity"] == ([matched] if reverse else [False, matched])
    assert block["data_bytes_hex"] == ("" if reverse else "0000000000000000") + ("000000000000e03f" if matched else "0000000000000000")


@pytest.mark.parametrize(("peer", "zero_hex"), [("1.5", "00000080"), ("3", "00000000")])
def test_residual_retained_zero_sign_distinguishes_integer_only_column(tmp_path: Path, peer: str, zero_hex: str) -> None:
    package = _package(tmp_path, {"d.csv": f"x,y\n-0,1\n9007199254740993,2\n{peer},3\n"}, _one(), features="float32")
    block = _matrix(package)
    assert block["dtype"] == "float32" and block["validity"] == [True, True, True]
    assert block["data_bytes_hex"][:8] == zero_hex


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_residual_retained_zero_sign_survives_row_removal_and_text_concat(tmp_path: Path, concat: bool, reverse: bool) -> None:
    files = {"a.csv": "x,z,y\n-0,1,1\n9007199254740993,,2\n1.5,3,3\n"}
    spec = _one(na={"policy": "remove_sample"})
    spec["sources"][0]["input"] = "a.csv"
    spec["sources"][0]["columns"][0]["select"] = ["x", "z"]
    if concat:
        files["b.csv"] = "x,z,y\nabc,4,4\n2.5,5,5\n"
        spec["sources"][0].update(input=["b.csv", "a.csv"] if reverse else ["a.csv", "b.csv"], merge="concat_samples")
    package = _package(tmp_path, files, spec)
    block = _matrix(package)
    offset = 2 if concat and reverse else 0
    raw = bytes.fromhex(block["data_bytes_hex"])
    assert raw[offset * 16:offset * 16 + 8].hex() == "0000000000000080"
    assert block["validity"][offset * 2:offset * 2 + 2] == [True, True]
    assert package.partitions["train"].n_samples == (4 if concat else 2)


@pytest.mark.parametrize("concat", [False, True])
def test_residual_retained_metadata_zero_is_float_with_negative_bits(tmp_path: Path, concat: bool) -> None:
    spec: dict[str, Any] = {"sources": [{"id": "d", "input": "a.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    files = {"a.csv": "x,m\n1,-0\n2,9007199254740993\n3,1.5\n"}
    if concat:
        files["b.csv"] = "x,m\n4,abc\n5,def\n"
        spec["sources"][0].update(input=["a.csv", "b.csv"], merge="concat_samples")
    wire = _package(tmp_path, files, spec).to_full_dict()
    values = wire["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]
    _strict_equal(values, [-0.0, 9007199254740993, 1.5] + (["abc", "def"] if concat else []))


@pytest.mark.parametrize("reverse", [False, True])
def test_residual_retained_variation_zero_keeps_sign_beside_text(tmp_path: Path, reverse: bool) -> None:
    files = {"d.csv": "x,y\n1,1\n2,2\n3,3\n4,4\n5,5\n", "v1.csv": "x\n-0\n9007199254740993\n1.5\n", "v2.csv": "x\nabc\n2.5\n"}
    spec = _one()
    spec["sources"][0]["variations"] = [{"name": "v", "input": ["v2.csv", "v1.csv"] if reverse else ["v1.csv", "v2.csv"]}]
    package = _package(tmp_path, files, spec, features="float32")
    array = package.to_assembled().blocks["train"].processings[0][0][1]
    assert struct.pack(">d", float(array[2 if reverse else 0, 0])).hex() == "8000000000000000"


@pytest.mark.parametrize("role", ["features", "metadata"])
def test_residual_retained_excel_text_zero_keeps_original_sign(tmp_path: Path, role: str) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["x", "y"] if role == "features" else ["x", "m"])
    for index, token in enumerate(["-0", "9007199254740993", "1.5"], start=1):
        sheet.append([token, index] if role == "features" else [index, token])
    workbook.save(tmp_path / "d.xlsx")
    spec = _one()
    spec["sources"][0]["input"] = "d.xlsx"
    if role == "metadata":
        spec["sources"][0]["columns"] = [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]
    package = _package(tmp_path, {}, spec, features="float32")
    if role == "features":
        assert _matrix(package)["data_bytes_hex"][:8] == "00000080"
    else:
        column = package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]
        assert column["dtype"] == "string"
        _strict_equal(column["values"], ["-0", "9007199254740993", "1.5"])


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_excel_metadata_numeric_text_preserves_binary_peer_and_decimal_scope(tmp_path: Path, concat: bool, reverse: bool) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["x", "m"], [1, "-0"], [2, "9007199254740993"], [3, 0.1234567890123457]]:
        sheet.append(row)
    workbook.save(tmp_path / "a.xlsx")
    spec: dict[str, Any] = {"sources": [{"id": "d", "input": "a.xlsx", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ";", "decimal": ",", "na": {"policy": "ignore"}}}
    expected: list[Any] = ["-0", "9007199254740993", 0.1234567890123457]
    if concat:
        (tmp_path / "b.csv").write_text("x;m\n4,5;0,25\n5,5;0,5\n")
        spec["sources"][0].update(input=["b.csv", "a.xlsx"] if reverse else ["a.xlsx", "b.csv"], merge="concat_samples")
        expected = [0.25, 0.5] + expected if reverse else expected + [0.25, 0.5]
    package = _package(tmp_path, {}, spec, features="float32")
    assert _matrix(package)["dtype"] == "float32"
    column = package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]
    assert column["dtype"] == "string"
    values = column["values"]
    _strict_equal(values, expected)
    assert struct.pack(">d", values[4 if concat and reverse else 2]).hex() == "3fbf9add3746f661"


def test_excel_metadata_physical_boolean_and_missing_keep_json_kinds(tmp_path: Path) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["x", "m"], [1.5, True], [2.5, False], [3.5, None]]:
        sheet.append(row)
    workbook.save(tmp_path / "a.xlsx")
    spec = {"sources": [{"id": "d", "input": "a.xlsx", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"na": {"policy": "ignore"}}}
    values = _package(tmp_path, {}, spec).to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]
    _strict_equal(values, [True, False, None])


def test_excel_literal_metadata_and_category_do_not_use_csv_decimal(tmp_path: Path) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["x", "y", "m"], [1.5, "-0", "0,125"], [2.5, "abc", "abc"]]:
        sheet.append(row)
    workbook.save(tmp_path / "a.xlsx")
    spec = _one(decimal=",")
    spec["sources"][0]["input"] = "a.xlsx"
    spec["sources"][0]["columns"].append({"role": "metadata", "select": ["m"]})
    payloads = _package(tmp_path, {}, spec).to_full_dict()["partitions"]["train"]["payloads"]
    _strict_equal(payloads["train/metadata"]["value"]["columns"][0]["values"], ["0,125", "abc"])
    _strict_equal(payloads["train/y"]["value"]["categorical"]["y"]["categories"], ["-0", "abc"])


def test_excel_selected_metadata_key_keeps_decoded_text_comparison_class(tmp_path: Path) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["k", "x"], ["1.5", 1.5], ["2.5", 2.5]]:
        sheet.append(row)
    workbook.save(tmp_path / "a.xlsx")
    spec = _lookup("warn")
    spec["sources"][0]["input"] = "a.xlsx"
    block = _matrix(_package(tmp_path, {"s.csv": "k,r\n1.5,0.5\n2.5,0.25\n"}, spec), "y")
    assert block["validity"] == [False, False]
    assert block["data_bytes_hex"] == "00000000000000000000000000000000"


@pytest.mark.parametrize("token", ["inf", "-inf", "+inf", "Infinity", "INF", "iNf", "inFinity"])
def test_metadata_numeric_only_infinity_refuses_at_factory(tmp_path: Path, token: str) -> None:
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    with pytest.raises(ValueError, match="finite"):
        _package(tmp_path, {"d.csv": f"x,m\n1.5,{token}\n2.5,2.5\n"}, spec)


@pytest.mark.parametrize("peer", ["abc", "2.5"])
@pytest.mark.parametrize("remove", [False, True])
def test_metadata_infinity_authority_uses_original_input_and_surviving_rows(tmp_path: Path, peer: str, remove: bool) -> None:
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x", "z"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "remove_sample" if remove else "ignore"}}}
    files = {"d.csv": f"x,z,m\n1.5,1,inf\n2.5,,{peer}\n"}
    if peer != "abc":
        with pytest.raises(ValueError, match="finite"):
            _package(tmp_path, files, spec)
    else:
        values = _package(tmp_path, files, spec).to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]
        _strict_equal(values, ["inf"] if remove else ["inf", "abc"])


@pytest.mark.parametrize("join", [False, True])
def test_metadata_removed_infinity_cannot_round_retained_peer(tmp_path: Path, join: bool) -> None:
    if join:
        spec = _lookup("drop", metadata=False)
        spec["sources"][0]["columns"].append({"role": "metadata", "select": ["m"]})
        files = {"a.csv": "k,x,m\nZ,1.5,inf\nA,2.5,9007199254740993\n", "s.csv": "k,r\nA,0.5\n"}
    else:
        spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x", "z"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "remove_sample"}}}
        files = {"d.csv": "x,z,m\n1.5,,inf\n2.5,2,9007199254740993\n"}
    values = _package(tmp_path, files, spec).to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]
    _strict_equal(values, [9007199254740993])


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("removed", [False, True])
def test_typed_parquet_metadata_infinity_refuses_only_selected_survivor(tmp_path: Path, selected: bool, removed: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"x": [1.5, 2.5], "z": [None, 2.0], "m": [float("inf"), 2.5]}), tmp_path / "d.parquet")
    spec = {"sources": [{"id": "t", "input": "d.parquet", "columns": [{"role": "features", "select": ["x", "z"]}, {"role": "metadata" if selected else "ignore", "select": ["m"]}]}], "params": {"na": {"policy": "remove_sample" if removed else "ignore"}}}
    if selected and not removed:
        with pytest.raises(ValueError, match="finite"):
            _package(tmp_path, {}, spec)
    else:
        package = _package(tmp_path, {}, spec)
        assert package.partitions["train"].n_samples == (1 if removed else 2)
        if selected:
            _strict_equal(package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"], [2.5])


@pytest.mark.parametrize("binary_kind", ["text", "number"])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_metadata_retains_csv_authority_and_decoded_excel_types(tmp_path: Path, binary_kind: str, reverse: bool) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    binary_values: list[Any] = ["abc", "def"] if binary_kind == "text" else [5.5, 6.5]
    sheet.append(["x", "m"])
    for x, value in zip([3.5, 4.5], binary_values, strict=True):
        sheet.append([x, value])
    workbook.save(tmp_path / "d.xlsx")
    spec = {"sources": [{"id": "d", "input": ["d.xlsx", "a.csv"] if reverse else ["a.csv", "d.xlsx"], "merge": "concat_samples", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = _package(tmp_path, {"a.csv": "x,m\n1.5,9007199254740993\n2.5,1.5\n"}, spec)
    values = package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]
    _strict_equal(values, binary_values + [9007199254740993, 1.5] if reverse else [9007199254740993, 1.5] + binary_values)


@pytest.mark.parametrize("merge", ["by_key", "concat_features"])
def test_merged_metadata_uses_column_file_origin(tmp_path: Path, merge: str) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["id", "t"], [1, "3"], [2, "4"]]:
        sheet.append(row)
    workbook.save(tmp_path / "d.xlsx")
    spec = {"sources": [{"id": "d", "input": ["a.csv", "d.xlsx"], "merge": merge, "key": "id", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m", "t"]}, {"role": "ignore", "select": ["id"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = _package(tmp_path, {"a.csv": "id,x,m\n1,1.5,9007199254740993\n2,2.5,1.5\n"}, spec)
    columns = package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"]
    _strict_equal({column["name"]: column["values"] for column in columns}, {"m": [9007199254740993, 1.5], "t": ["3", "4"]})


@pytest.mark.parametrize("words", [["TRUE", "FALSE"], ["True", "False"], ["true", "false"], ["TRUE", "TRUE"]])
def test_w1_excel_boolean_words_keep_boolean_metadata(tmp_path: Path, words: list[str]) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["x", "m"], [1.5, words[0]], [2.5, words[1]]]:
        sheet.append(row)
    workbook.save(tmp_path / "d.xlsx")
    spec = {"sources": [{"id": "t", "input": "d.xlsx", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"na": {"policy": "ignore"}}}
    column = _package(tmp_path, {}, spec).to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]
    assert column["dtype"] == "bool"
    _strict_equal(column["values"], [True, words[1] == "TRUE"])


@pytest.mark.parametrize("role", ["features", "targets", "weights"])
def test_w1_excel_boolean_words_are_numeric_in_numeric_roles(tmp_path: Path, role: str) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    column = {"features": 0, "targets": 1, "weights": 2}[role]
    sheet.append(["x", "y", "m"])
    for index, word in enumerate(["TRUE", "FALSE"]):
        row: list[Any] = [1.5 + index, 1.0 + index, 1.0 + index]
        row[column] = word
        sheet.append(row)
    workbook.save(tmp_path / "d.xlsx")
    spec = _one("weights")
    spec["sources"][0]["input"] = "d.xlsx"
    package = _package(tmp_path, {}, spec)
    wire = package.to_full_dict()["partitions"]["train"]["payloads"]
    block = _matrix(package, "y" if role == "targets" else "x0") if role != "weights" else wire["train/weights"]["value"]["matrix"]
    assert block["data_bytes_hex"] == "000000000000f03f0000000000000000"
    assert block["validity"] == [True, True]
    assert wire["train/y"]["value"]["categorical"] == {}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("role", ["features", "targets", "weights", "variation", "all_na_peer"])
def test_w2_integer_only_input_keeps_unsigned_zero_when_promoted(tmp_path: Path, reverse: bool, role: str) -> None:
    spec = _one("weights")
    files = {"a.csv": "x,y,m\n1,1,1\n2,2,2\n", "b.csv": "x,y,m\n3,3,3\n4,4,4\n"}
    names = ["b.csv", "a.csv"] if reverse else ["a.csv", "b.csv"]
    values: list[float | None] = [0.0, 3.0, 1.5, 2.5]
    if role == "variation":
        files.update({"d.csv": "x,y,m\n1,1,1\n2,2,2\n3,3,3\n4,4,4\n", "a.csv": "x\n-0\n3\n", "b.csv": "x\n1.5\n2.5\n"})
        spec["sources"][0]["variations"] = [{"name": "v", "input": names}]
    else:
        column = {"features": "x", "targets": "y", "weights": "m", "all_na_peer": "x"}[role]
        for name, tokens in [("a.csv", ["-0", "3"]), ("b.csv", ["", ""] if role == "all_na_peer" else ["1.5", "2.5"])]:
            rows = [{"x": str(i + 1), "y": str(i + 1), "m": str(i + 1)} for i in range(2)]
            for row, token in zip(rows, tokens, strict=True):
                row[column] = token
            files[name] = "x,y,m\n" + "".join(",".join(row[key] for key in ["x", "y", "m"]) + "\n" for row in rows)
        spec["sources"][0].update(input=names, merge="concat_samples")
    package = _package(tmp_path, files, spec)
    train = package.to_assembled().to_full_dict()["blocks"]["train"]
    if role == "variation":
        block = train["processings"][0][0]["matrix"]
    elif role == "weights":
        block = train["weights"]
    else:
        block = _matrix(package, "y" if role == "targets" else "x0")
    expected: list[float | None] = [0.0, 3.0, None, None] if role == "all_na_peer" else values
    if reverse:
        expected = expected[2:] + expected[:2]
    assert block["data_bytes_hex"] == b"".join(struct.pack("<d", 0.0 if value is None else value) for value in expected).hex()
    assert block["validity"] == [value is not None for value in expected]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("role", ["features", "targets", "weights", "metadata"])
def test_w3_removed_integer_refusal_does_not_create_float_peer(tmp_path: Path, reverse: bool, role: str) -> None:
    column = {"features": "x", "targets": "y", "weights": "m", "metadata": "m"}[role]
    rows = [{"x": "1", "y": "1", "m": "1"}, {"x": "2", "y": "2", "m": "2"}]
    rows[0][column], rows[1][column] = "-0", "18446744073709551616"
    rows[1]["x" if role in {"targets", "weights"} else "y"] = ""
    files = {
        "a.csv": "x,y,m\n" + "".join(",".join(row[name] for name in ["x", "y", "m"]) + "\n" for row in rows),
        "b.csv": "x,y,m\n1.5,1.5,1.5\n2.5,2.5,2.5\n",
    }
    spec = _one("weights", na={"policy": "remove_sample"})
    spec["sources"][0].update(input=["b.csv", "a.csv"] if reverse else ["a.csv", "b.csv"], merge="concat_samples")
    if role == "metadata":
        spec["sources"][0]["columns"][-1]["role"] = "metadata"
    package = _package(tmp_path, files, spec)
    train = package.to_assembled().to_full_dict()["blocks"]["train"]
    assert train["n_samples"] == 3
    if role == "metadata":
        expected: list[int | float] = [1.5, 2.5, 0] if reverse else [0, 1.5, 2.5]
        _strict_equal(train["metadata"]["columns"][0]["values"], expected)
    else:
        block = train["weights"] if role == "weights" else _matrix(package, "y" if role == "targets" else "x0")
        values = [1.5, 2.5, 0.0] if reverse else [0.0, 1.5, 2.5]
        assert block["data_bytes_hex"] == b"".join(struct.pack("<d", value) for value in values).hex()
        assert block["validity"] == [True, True, True]


@pytest.mark.parametrize("order", ["single", "xc", "cx"])
@pytest.mark.parametrize("storage", [None, "float32"])
def test_spreadsheet_origin_keeps_physical_text_zero_sign(tmp_path: Path, order: str, storage: str | None) -> None:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in [["x", "y"], ["-0", 1.0], ["3", 2.0], ["18446744073709551616", None]]:
        sheet.append(row)
    workbook.save(tmp_path / "a.xlsx")
    spec = _one(na={"policy": "remove_sample"})
    spec["sources"][0]["input"] = "a.xlsx"
    values = [-0.0, 3.0]
    if order != "single":
        spec["sources"][0].update(input=["b.csv", "a.xlsx"] if order == "cx" else ["a.xlsx", "b.csv"], merge="concat_samples")
        values = [1.5, 2.5] + values if order == "cx" else values + [1.5, 2.5]
    package = _package(tmp_path, {"b.csv": "x,y\n1.5,3\n2.5,4\n"}, spec, **({"features": storage} if storage else {}))
    block = _matrix(package)
    assert block["dtype"] == (storage or "float64")
    assert block["precision"]["requested_dtype"] == storage
    assert block["validity"] == [True] * len(values)
    assert block["data_bytes_hex"] == b"".join(struct.pack("<f" if storage else "<d", value) for value in values).hex()
