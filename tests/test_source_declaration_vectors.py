# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Driver of the common source-declaration review table (70 literal cases, 12 semantic groups).

``fixtures/source_declaration_vectors.json`` is controller-owned and shared byte for byte with the Rust crate's
test data; this file only consumes it. Every case feeds its ``raw_token`` to the ACTUAL strict text decoder (no
``json.loads`` first), calls the actual public factory and, independently, the whole-dataset operation, and
compares admission, the whole link (``present`` / ``not_representable`` / ``invalid``) and the exact output F64 bits.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize import NumericContractError, NumericStorageOptions
from nirs4all_io.materialize.package import FeatureMatrix, TargetTable
from nirs4all_io.public_content import ContentNotRepresentableError, labelled_dataset_content_identity
from nirs4all_io.public_dataset import Dataset
from nirs4all_io.strict_json import NumberDomainError

_HERE = Path(__file__).parent
_FIXTURE = _HERE / "fixtures" / "source_declaration_vectors.json"
_CRATE_FIXTURES = _HERE.parent / "crates" / "nirs4all-io-core" / "tests" / "fixtures"
_PINS = {
    "source_declaration_vectors.json": "881312a5950b3b207c00fa8be6662eb3f603ddf90037ee5cef8fc53b97ced35f",
    "finite_float_vectors.json": "f3b0d0199ab96da843eedd4b93cf9d38db5c1c97dea03a2398b8297cb5fdc576",
}
DOCUMENT = json.loads(_FIXTURE.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = DOCUMENT["cases"]
_WHOLE_REASON = {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}


# --------------------------------------------------------------------------- #
# Fixture identity and the Cargo-packaged copies                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_PINS))
def test_common_fixtures_are_pinned_and_byte_equal_to_the_crate_copy(name: str) -> None:
    data = (_HERE / "fixtures" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == _PINS[name], f"{name} drifted from the controller-owned bytes"
    crate = _CRATE_FIXTURES / name
    if not crate.parent.is_dir():
        pytest.skip("the Rust crate test data is not part of this checkout")
    assert crate.read_bytes() == data, f"{name}: the Cargo-packaged copy differs"


def test_the_table_is_complete() -> None:
    assert DOCUMENT["schema"] == "nirs4all.io.source-declaration-vectors.v1" and len(CASES) == 70
    assert len({case["group"] for case in CASES}) == 12 and len({case["id"] for case in CASES}) == 70
    assert {case["whole_link"] for case in CASES} == {"present", "not_representable", "invalid"}
    assert all(("raw_token" in case) != ("host_bits_hex" in case) for case in CASES)


# --------------------------------------------------------------------------- #
# Case construction                                                             #
# --------------------------------------------------------------------------- #
def _record_text(case: dict[str, Any]) -> str:
    """A small valid public record whose first retained cell carries the case's raw token verbatim."""
    integer_source = case["source_dtype"] in ("int64", "uint64")
    cell = "@@"
    rows = [[cell, 1 if integer_source else 0.5], [2 if integer_source else 2.0, 3 if integer_source else 3.0]]
    classification = case["role"] == "targets"
    source_dtype = "float64" if classification else case["source_dtype"]
    source_rows = [[1.5, 0.5], [2.0, 3.0]] if classification else rows
    y = {"dtype": "int64", "shape": [2], "values": [cell, 0]} if classification else {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]}
    record = {
        "schema": "nirs4all.dataset.v1", "schema_version": 1,
        "dataset": {
            "schema": "nirs4all.multimodal-dataset", "schema_version": 1, "name": "m", "sample_ids": ["s0", "s1"], "source_alignment": "strict",
            "sources": [{"name": "nir", "sample_ids": ["s0", "s1"], "representation_id": "signal_1d", "axes": ["sample", "wavelength"], "feature_names": None,
                         "axis_units": {"wavelength": "nm"}, "axis_coordinates": {"wavelength": [1.0, 2.0]},
                         "array": {"dtype": source_dtype, "shape": [2, 2], "values": source_rows},
                         "presence_mask": {"dtype": "bool", "shape": [2], "values": [True, True]}}],
            "y": y, "target_names": ["y"], "target_mask": {"dtype": "bool", "shape": [2], "values": [True, True]},
            "task_type": "classification" if classification else "regression", "groups": None,
            "partitions": {"dtype": "<U5", "shape": [2], "values": ["train", "train"]},
        },
        "origin_ids": ["s0", "s1"], "fold_ids": [None, None],
    }  # fmt: skip
    return json.dumps(record).replace('"@@"', case["raw_token"])


def _value(case: dict[str, Any]) -> Any:
    """The factory input: a raw JSON text (strict decoder) or an actual already-decoded host F64 value."""
    if "raw_token" in case:
        return _record_text(case)
    bits = bytes.fromhex(case["host_bits_hex"])
    host = struct.unpack(">d", bits)[0]
    values = np.array([[host, 0.5], [2.0, 3.0]], dtype=np.float64)
    assert values.view(np.uint64)[0, 0] == int.from_bytes(bits, "big"), "the host control carries the literal bits"
    return Dataset.from_sources({"nir": values}, sample_ids=["s0", "s1"], y=np.array([1.0, 2.0]))


def _options(case: dict[str, Any]) -> NumericStorageOptions:
    role = case["role"]
    return NumericStorageOptions(**{role: case["request"]})


def _category(error: BaseException) -> str:
    message = str(error)
    if isinstance(error, NumberDomainError):
        return "text_ingress"
    if "not exact in float32" in message:
        return "classification_label"
    if isinstance(error, NumericContractError) and "cannot represent exactly" in message:
        return "selected_storage"
    return "source_declaration"


def _first_cell(case: dict[str, Any], package: Any) -> float:
    payloads = dict(package.partitions["train"].payloads)
    if case["role"] == "targets":
        table = payloads["train/y"]
        assert isinstance(table, TargetTable)
        return float(table.columns[0].block.values[0, 0])
    feature = payloads["train/x0"]
    assert isinstance(feature, FeatureMatrix)
    return float(feature.matrix.values[0, 0])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_review_case(case: dict[str, Any]) -> None:
    value = _value(case)
    # the whole operation runs on its own, with no storage options
    whole: dict[str, str] | None
    try:
        whole, whole_state = labelled_dataset_content_identity(value), "present"
    except ContentNotRepresentableError:
        whole, whole_state = None, "not_representable"
    except ValueError:
        whole, whole_state = None, "invalid"
    assert whole_state == case["whole_link"], f"{case['id']}: whole operation is {whole_state}"
    try:
        package = nio.matrix_dataset_package_with_storage(value, "nir", _options(case))
    except ValueError as error:
        assert case["expected"] == "refuse", f"{case['id']}: unexpected refusal: {error}"
        assert _category(error) == case["refusal_category"], f"{case['id']}: refused as {_category(error)}: {error}"
        return
    assert case["expected"] == "accept", f"{case['id']}: accepted but the table refuses"
    stored = _first_cell(case, package)
    assert struct.pack(">d", stored).hex() == case["expected_f64_bits_hex"], f"{case['id']}: stored {stored!r}"
    audit = package.audits[0]
    if case["whole_link"] == "present":
        assert audit["dataset_content_identity"] == whole and "dataset_content_identity_unavailable" not in audit
    else:
        assert audit["dataset_content_identity"] is None and audit["dataset_content_identity_unavailable"] == _WHOLE_REASON
    assert package.warnings == []


def test_every_semantic_group_is_exercised_in_both_directions_where_the_table_has_them() -> None:
    by_group: dict[str, set[str]] = {}
    for case in CASES:
        by_group.setdefault(case["group"], set()).add(case["expected"])
    assert by_group["02-declared-f64-inexact-integer"] == {"refuse"} and by_group["09-typed-inexact-requested-f32"] == {"accept"}
    assert by_group["10-decimal-underflow-ingress"] == {"accept", "refuse"} and by_group["12-genuine-host-f64"] == {"accept"}
