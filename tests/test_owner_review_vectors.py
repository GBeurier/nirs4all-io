# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Consume the shared additive ``owner_review_vectors.json`` (single writer: the Rust owner) on the Python oracle.

The fixture is only read. Every row whose host list includes ``python`` (default ``rust`` + ``python``) executes its
named operation through the REAL owner entry points; a row is never counted as passing without having run, and an
unknown operation, unknown field or unknown refusal category fails instead of being ignored:

* ``expected.assertions`` / ``first_assertions`` are ``[{pointer, value}]`` JSON-pointer assertions (RFC 6901, ``~0``/``~1``)
  into the ACTUAL owner assembled-v3 full wire (``assembled_to_full_dict``) of the produced package's ``to_assembled()``
  (``assertions``) or of the first assembly stage (``first_assertions``);
* values are compared by a type-sensitive comparator: ``null``/bool/string/integer/float are distinct kinds, an integer
  never equals a float or a bool, floats compare by their IEEE binary64 bits (negative zero included);
* ``expected.projection`` is the separate ``executed_projection`` outcome of an accepted package;
* ``expected.package_assertions`` (post-B4 bank) are the same ``[{pointer, value}]`` shape applied to the produced package's
  ``to_full_dict()`` (manifest root, audits, identity), compared with the same strict comparator.

Two banks are consumed with the one protocol: ``owner_review_vectors.json`` (current 246) and, when present,
``owner_final_work_vectors.json`` (post-B4 corrections; its publication state is the Rust author's
``SHARED_FIXTURE_READY.json``). Neither file is ever written here.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

import nirs4all_io as nio
from nirs4all_io import MultimodalDataset, strict_json
from nirs4all_io.dataset_facade import to_u07_raw_sources
from nirs4all_io.materialize import DatasetPackage, NumericContractError
from nirs4all_io.materialize.loaders import LoaderError
from nirs4all_io.materialize.numeric import NumericStorageOptions
from nirs4all_io.materialize.package import assembled_to_full_dict
from nirs4all_io.public_content import ContentNotRepresentableError, labelled_dataset_content_identity, metadata_number
from nirs4all_io.public_dataset import dataset as public_dataset
from nirs4all_io.spec import DatasetSpec
from nirs4all_io.spec.enums import SpecError
from nirs4all_io.strict_json import NumberDomainError

_BANKS = [name for name in ("owner_review_vectors.json", "owner_final_work_vectors.json") if (Path(__file__).parent / "fixtures" / name).is_file()]
_DOCUMENTS = {name: json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")) for name in _BANKS}
_ALL = [(name, case) for name, document in _DOCUMENTS.items() for case in document["cases"]]
_CASES = [(name, case) for name, case in _ALL if "python" in case.get("hosts", ["rust", "python"])]
_NOT_PYTHON = [case for _name, case in _ALL if "python" not in case.get("hosts", ["rust", "python"])]
_TOP = {"id", "condition", "operation", "input", "expected", "storage", "hosts"}
_STORAGE = {"features", "targets", "weights", "consumer"}
_COMMON = {"outcome", "category"}
_EXPECTED_FIELDS = {
    "file_package": _COMMON | {"assertions", "package_assertions", "data_bytes_hex", "validity", "precision", "metadata", "projection"},
    "assembly_reselect": _COMMON | {"first_assertions", "assertions"},
    "public_masked": _COMMON | {"assertions", "package_assertions", "projection", "whole_link", "whole_link_unavailable_reason"},
    "public_package": _COMMON | {"assertions", "package_assertions", "projection"},
    "whole": _COMMON | {"u07_first_numeric_bits"},
    "spec_validate": _COMMON | {"aggregate_threshold_bits"},
    "python_from_dict": _COMMON | {"first_source_f32_bytes_hex"},
}
_INPUT_FIELDS = {
    "file_package": {"spec_text", "files"},
    "assembly_reselect": {"spec_text", "files", "stages"},
    "public_masked": {"text", "source_id"},
    "public_package": {"text", "source_id"},
    "whole": {"text", "source_id"},
    "spec_validate": {"text"},
    "python_from_dict": {"text"},
}
_CATEGORIES = {
    "accepted", "missing_projection_identity", "numeric_domain", "no_numeric_features", "precision_reselection", "invalid_threshold",
    "invalid_selector", "source_declaration", "unsupported_integer_source", "invalid_observed_float32",
}
_PROJECTION_CATEGORIES = {"missing_identity", "unstable_identity", "invalid_primary_identity"}


# --------------------------------------------------------------------------- #
# Strict comparator and JSON pointer                                           #
# --------------------------------------------------------------------------- #
def same_json(expected: Any, actual: Any, where: str = "$") -> None:
    """Type-sensitive JSON equality; no canonical writer, no coercion: ``1``, ``1.0`` and ``true`` are three different values."""
    if expected is None:
        assert actual is None, f"{where}: expected null, got {actual!r}"
    elif type(expected) is bool:
        assert type(actual) is bool and actual == expected, f"{where}: expected bool {expected!r}, got {type(actual).__name__} {actual!r}"
    elif type(expected) is int:
        assert type(actual) is int and actual == expected, f"{where}: expected integer {expected!r}, got {type(actual).__name__} {actual!r}"
    elif type(expected) is float:
        assert type(actual) is float, f"{where}: expected float {expected!r}, got {type(actual).__name__} {actual!r}"
        assert struct.pack(">d", actual) == struct.pack(">d", expected), f"{where}: float bits differ: expected {expected!r}, got {actual!r}"
    elif type(expected) is str:
        assert type(actual) is str and actual == expected, f"{where}: expected string {expected!r}, got {type(actual).__name__} {actual!r}"
    elif type(expected) is list:
        assert type(actual) is list and len(actual) == len(expected), f"{where}: expected a list of {len(expected)}, got {actual!r}"
        for index, (e, a) in enumerate(zip(expected, actual, strict=True)):
            same_json(e, a, f"{where}[{index}]")
    elif type(expected) is dict:
        assert type(actual) is dict and set(actual) == set(expected), f"{where}: expected keys {sorted(expected)}, got {sorted(actual) if type(actual) is dict else actual!r}"
        for key in expected:
            same_json(expected[key], actual[key], f"{where}.{key}")
    else:  # pragma: no cover - the fixture is JSON
        raise AssertionError(f"{where}: unsupported expected literal {expected!r}")


def resolve_pointer(document: Any, pointer: str) -> Any:
    """RFC 6901 resolution (``~1`` -> ``/``, then ``~0`` -> ``~``); a missing member is a failure, never ``None``."""
    assert pointer == "" or pointer.startswith("/"), f"invalid JSON pointer {pointer!r}"
    node = document
    for token in ([] if pointer == "" else pointer[1:].split("/")):
        token = token.replace("~1", "/").replace("~0", "~")
        if type(node) is list:
            assert token.isdigit() and (token == "0" or not token.startswith("0")) and int(token) < len(node), f"pointer {pointer!r}: bad index {token!r}"
            node = node[int(token)]
        else:
            assert type(node) is dict and token in node, f"pointer {pointer!r}: no member {token!r}"
            node = node[token]
    return node


def assert_pointers(assertions: list[dict[str, Any]], document: Any, label: str) -> None:
    assert assertions, f"{label}: empty assertion list"
    for assertion in assertions:
        assert set(assertion) == {"pointer", "value"}, f"{label}: unknown assertion fields {sorted(assertion)}"
        same_json(assertion["value"], resolve_pointer(document, assertion["pointer"]), f"{label}:{assertion['pointer']}")


def test_comparator_is_type_sensitive() -> None:
    same_json({"a": [1, 1.5, None, "x", True, -0.0]}, {"a": [1, 1.5, None, "x", True, -0.0]})
    for expected, actual in ((1, 1.0), (1.0, 1), (1, True), (True, 1), (0.0, -0.0), (-0.0, 0.0), ("1", 1), (None, 0), (None, ""), ([1], [1.0]), ({"a": 1}, {"a": 1, "b": 2}), (float("nan"), 0.0)):
        with pytest.raises(AssertionError):
            same_json(expected, actual)
    assert resolve_pointer({"a/b": {"c~d": [10, 20]}}, "/a~1b/c~0d/1") == 20
    for pointer in ("/missing", "/a/9", "/a/01", "a"):
        with pytest.raises(AssertionError):
            resolve_pointer({"a": [1, 2]}, pointer)


def test_fixture_envelope_and_protocol_coverage() -> None:
    assert _BANKS and _BANKS[0] == "owner_review_vectors.json"
    seen: set[str] = set()
    for name, document in _DOCUMENTS.items():
        assert document["schema"] == "nirs4all.io.owner-review-vectors.v1" and set(document) == {"schema", "cases"}, name
        for case in document["cases"]:
            assert case["id"] not in seen, f"duplicate id {case['id']}"
            seen.add(case["id"])
    for _name, case in _ALL:
        operation = case["operation"]
        assert operation in _EXPECTED_FIELDS, f"{case['id']}: unknown operation {operation!r}"
        assert set(case) <= _TOP and {"id", "condition", "operation", "input", "expected"} <= set(case), f"{case['id']}: unknown or missing top-level fields"
        assert set(case["input"]) <= _INPUT_FIELDS[operation], f"{case['id']}: unknown input fields {sorted(set(case['input']) - _INPUT_FIELDS[operation])}"
        assert set(case.get("storage", {})) <= _STORAGE and set(case["expected"]) <= _EXPECTED_FIELDS[operation], f"{case['id']}: unknown storage/expected fields {sorted(set(case['expected']) - _EXPECTED_FIELDS[operation])}"
        assert case["expected"]["outcome"] in ("accept", "refuse") and case["expected"]["category"] in _CATEGORIES, f"{case['id']}: unknown outcome/category"
        assert case.get("hosts", ["rust", "python"]) in (["rust", "python"], ["python"]), f"{case['id']}: unknown hosts"
        for stage in case["input"].get("stages", []):
            assert set(stage) == {"operation", "storage"} and stage["operation"] in ("assemble", "package") and set(stage["storage"]) <= _STORAGE
        projection = case["expected"].get("projection")
        if projection is not None:
            assert projection["outcome"] in ("accept", "refuse")
            assert set(projection) == ({"outcome", "sample_ids"} if projection["outcome"] == "accept" else {"outcome", "category"}), f"{case['id']}: unknown projection fields"
            assert projection["outcome"] == "accept" or projection["category"] in _PROJECTION_CATEGORIES
    assert not _NOT_PYTHON, "a row that excludes the Python host needs its own explicit Python test; none exist today"
    assert Counter(case["operation"] for _n, case in _CASES) == Counter(case["operation"] for _n, case in _ALL)


def test_every_accepted_metadata_and_identity_row_carries_typed_assertions() -> None:
    """3 == 3.0 must never hide a type error: an accepted metadata/ID/identity row without typed values is a fixture defect."""
    weak = [
        case["id"]
        for _name, case in _CASES
        if case["expected"]["outcome"] == "accept"
        and case["operation"] == "file_package"
        and case["condition"].startswith(("numeric_meta", "numeric_id", "original_column", "id-", "metadata-", "removed-"))
        and not any(case["expected"].get(field) for field in ("assertions", "metadata"))
    ]
    assert not weak, f"accepted rows without typed metadata assertions: {weak}"


# --------------------------------------------------------------------------- #
# Operations                                                                   #
# --------------------------------------------------------------------------- #
def _classify(error: BaseException, operation: str) -> str:
    text = str(error)
    if "Cannot re-select" in text:
        return "precision_reselection"
    if "outlier_threshold" in text:
        return "invalid_threshold"
    if isinstance(error, LoaderError) and "outside i64 range" in text:
        return "unsupported_integer_source"
    if operation.startswith("public_") and ("declared float64 array" in text or "Unsupported public dataset dtype" in text):
        return "source_declaration"
    if operation.startswith("public_") and "overflows float32" in text:
        return "invalid_observed_float32"
    if isinstance(error, SpecError) and "selector" in text and "no unassigned numeric" not in text:
        return "invalid_selector"
    if isinstance(error, SpecError) and ("has no feature source" in text or ("numeric" in text and "found no" in text)):
        return "no_numeric_features"
    if isinstance(error, (NumberDomainError, NumericContractError, ContentNotRepresentableError, ValueError)) and any(
        word in text.lower() for word in ("exactly", "range", "underflow", "overflow", "cap", "rounds to zero", "domain", "finite", "numeric authority", "outside", "represent")
    ):
        return "numeric_domain"
    return f"other:{type(error).__name__}:{text[:80]}"


def _projection_category(error: BaseException) -> str:
    text = str(error)
    for needle, category in (("Missing projection identity", "missing_identity"), ("stable sample identity", "unstable_identity"), ("Invalid primary identity", "invalid_primary_identity")):
        if needle in text:
            return category
    return f"other:{type(error).__name__}:{text[:80]}"


def _write_files(root: Path, files: dict[str, Any]) -> None:
    """Create the input files; PyArrow is imported only for a Parquet descriptor (CSV text is written verbatim, never coerced)."""
    for name, content in files.items():
        if isinstance(content, str):
            (root / name).write_text(content, encoding="utf-8")
            continue
        assert set(content) == {"format", "columns"} and content["format"] == "parquet", f"unknown file descriptor for {name!r}"
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        kinds = {"float32": pa.float32(), "float64": pa.float64(), "uint64": pa.uint64(), "int64": pa.int64()}
        columns = {}
        for column in content["columns"]:
            assert set(column) == {"name", "dtype", "values"}
            integer = column["dtype"] in ("uint64", "int64")
            cells = [None if value is None else (int(value) if integer else float(value)) for value in column["values"]]
            assert all(value is None or (isinstance(value, str) if integer else type(value) is float) for value in column["values"]), "descriptor cells: decimal strings for integers, native floats otherwise"
            columns[column["name"]] = pa.array(cells, kinds[column["dtype"]])
        pq.write_table(pa.table(columns), root / name)


def _stage_run(case: dict[str, Any], root: Path) -> tuple[dict[str, Any] | None, DatasetPackage | None]:
    """Run the stages in order. Returns (first stage's assembled wire, last package); a failure carries its stage index."""
    inp = case["input"]
    _write_files(root, inp["files"])
    spec, assembled, package, first = json.loads(inp["spec_text"]), None, None, None
    stages = inp["stages"]
    assert stages[0]["operation"] == "assemble", f"{case['id']}: the first stage must assemble"
    for index, stage in enumerate(stages):
        try:
            if stage["operation"] == "assemble":
                assembled = nio.load(spec, target="assembled", base_dir=root, **stage["storage"])
                if first is None:
                    first = assembled_to_full_dict(assembled)
            else:
                assert assembled is not None
                package = DatasetPackage.from_assembled(assembled, **stage["storage"])
        except Exception as error:
            raise _StageError(index, len(stages), first, error) from error
    return first, package


class _StageError(Exception):
    def __init__(self, index: int, count: int, first: dict[str, Any] | None, error: Exception) -> None:
        super().__init__(f"stage {index} of {count} failed: {error!r}")
        self.index, self.count, self.first, self.error = index, count, first, error


def _default_sources(package: DatasetPackage) -> list[str]:
    """The owner's default projection selection: every source of the package, in first-seen order (no arbitrary first source)."""
    seen: list[str] = []
    for partition in package.partitions.values():
        seen.extend(source for source in partition.source_ids if source not in seen)
    return seen


def _check_package(case: dict[str, Any], package: DatasetPackage, expected: dict[str, Any]) -> None:
    wire = assembled_to_full_dict(package.to_assembled())
    first = wire["partition_order"][0]
    flat = {"data_bytes_hex": f"/blocks/{first}/x/0/data_bytes_hex", "validity": f"/blocks/{first}/x/0/validity", "precision": f"/blocks/{first}/x/0/precision", "metadata": f"/blocks/{first}/metadata"}
    for field, pointer in flat.items():
        if field in expected:
            assert_pointers([{"pointer": pointer, "value": expected[field]}], wire, f"{case['id']}:{field}")
    if "assertions" in expected:
        assert_pointers(expected["assertions"], wire, case["id"])
    if "package_assertions" in expected:
        assert_pointers(expected["package_assertions"], package.to_full_dict(), f"{case['id']}:package")
    projection = expected.get("projection")
    if projection is None:
        return
    selected = _default_sources(package)
    if projection["outcome"] == "accept":
        result = package.executed_projection(selected)
        same_json(projection["sample_ids"], result.identity.sample_ids, f"{case['id']}:projection.sample_ids")
    else:
        with pytest.raises(NumericContractError) as caught:
            package.executed_projection(selected)
        assert _projection_category(caught.value) == projection["category"], f"{case['id']}: projection refusal {caught.value!r}"
    if expected["category"] == "missing_projection_identity":
        assert projection == {"outcome": "refuse", "category": "missing_identity"}, f"{case['id']}: category/projection disagree"


def _u07_bits(record: Any) -> str:
    """Big-endian IEEE binary64 pattern of the designated U07 metadata numeric column's first cell, via the owner helpers."""
    projected = to_u07_raw_sources(public_dataset(record).multimodal)
    first = projected["sources"]["metadata"]["rows"][0][0]
    return struct.pack(">d", metadata_number(first)).hex()


def _check_accepted(case: dict[str, Any], result: Any, expected: dict[str, Any]) -> None:
    operation = case["operation"]
    if operation in ("file_package", "public_package", "public_masked", "assembly_reselect"):
        package = result
        assert isinstance(package, DatasetPackage), case["id"]
        _check_package(case, package, expected)
        if operation == "public_masked" and "whole_link" in expected:
            audit = package.audits[0]
            if expected["whole_link"] is None:
                assert audit["dataset_content_identity"] is None, case["id"]
            else:
                same_json(expected["whole_link"], audit["dataset_content_identity"], f"{case['id']}:whole_link")
            if "whole_link_unavailable_reason" in expected:
                reason = expected["whole_link_unavailable_reason"]
                if reason is None:
                    assert "dataset_content_identity_unavailable" not in audit, f"{case['id']}: the unavailable-reason key must be absent"
                else:
                    assert audit["dataset_content_identity_unavailable"]["reason"] == reason, case["id"]
    elif operation == "whole":
        assert result["algorithm"] == "sha256", case["id"]
        if "u07_first_numeric_bits" in expected:
            same_json(expected["u07_first_numeric_bits"], _u07_bits(strict_json.loads(case["input"]["text"])), f"{case['id']}:u07_first_numeric_bits")
    elif operation == "spec_validate":
        if "aggregate_threshold_bits" in expected:
            assert result.aggregate is not None, case["id"]
            same_json(expected["aggregate_threshold_bits"], struct.pack(">d", result.aggregate.outlier_threshold).hex(), f"{case['id']}:aggregate_threshold_bits")
    elif operation == "python_from_dict":
        if "first_source_f32_bytes_hex" in expected:
            import numpy as np

            values = np.asarray(next(iter(result.sources.values())).values)
            assert values.dtype == np.float32, f"{case['id']}: the first source must carry float32 storage, got {values.dtype}"
            same_json(expected["first_source_f32_bytes_hex"], np.ascontiguousarray(values).tobytes()[:4].hex(), f"{case['id']}:first_source_f32_bytes_hex")
    else:  # pragma: no cover - guarded by the protocol test
        raise AssertionError(f"unsupported operation {operation!r}")


def _run(case: dict[str, Any], root: Path) -> Any:
    operation, inp, storage = case["operation"], case["input"], case.get("storage") or {}
    if operation == "file_package":
        _write_files(root, inp["files"])
        return nio.to_dataset_package(json.loads(inp["spec_text"]), base_dir=root, **storage)
    if operation == "assembly_reselect":
        first, package = _stage_run(case, root)
        expected = case["expected"]
        if "first_assertions" in expected:
            assert first is not None
            assert_pointers(expected["first_assertions"], first, f"{case['id']}:first")
        return package
    if operation == "spec_validate":
        spec = DatasetSpec.from_dict(strict_json.loads(inp["text"]))
        spec.validate()
        return spec
    if operation == "whole":
        record = strict_json.loads(inp["text"])
        identity = labelled_dataset_content_identity(record)
        if case["condition"] == "u07_numeric_column":
            _u07_bits(record)  # the designated numeric column must also pass the U07 helper
        return identity
    if operation == "python_from_dict":
        return MultimodalDataset.from_dict(strict_json.loads(inp["text"]))
    if operation == "public_masked":
        return nio.masked_matrix_dataset_package_with_storage(inp["text"], inp["source_id"], NumericStorageOptions(**storage))[0]
    if operation == "public_package":
        return nio.matrix_dataset_package_with_storage(inp["text"], inp["source_id"], NumericStorageOptions(**storage))
    raise AssertionError(f"unsupported operation {operation!r}")


@pytest.mark.parametrize("case", [case for _name, case in _CASES], ids=[case["id"] for _name, case in _CASES])
def test_owner_review_vector(case: dict[str, Any], tmp_path: Path) -> None:
    expected = case["expected"]
    try:
        result = _run(case, tmp_path)
    except _StageError as stage:
        # the first assembly's assertions ran before any later stage; an EARLIER stage failure is never the expected refusal
        assert expected["outcome"] == "refuse", f"{case['id']}: unexpected refusal {stage}"
        assert stage.index == stage.count - 1, f"{case['id']}: refused at stage {stage.index}, before the final stage"
        if "first_assertions" in expected:
            assert stage.first is not None, f"{case['id']}: the first stage did not complete"
            assert_pointers(expected["first_assertions"], stage.first, f"{case['id']}:first")
        assert _classify(stage.error, case["operation"]) == expected["category"], f"{case['id']}: {stage.error!r}"
        return
    except Exception as error:  # the exact refusal CATEGORY is part of the vector
        assert expected["outcome"] == "refuse", f"{case['id']}: unexpected refusal {error!r}"
        assert _classify(error, case["operation"]) == expected["category"], f"{case['id']}: {error!r}"
        return
    assert expected["outcome"] == "accept", f"{case['id']}: accepted but the vector refuses ({expected['category']})"
    _check_accepted(case, result, expected)
