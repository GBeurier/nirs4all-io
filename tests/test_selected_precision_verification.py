# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""``verify=True`` means the IR already holds the selected STORED values; nothing is rounded again.

``ExactBlock.with_precision(..., verify=False)`` is the first selection: raw cells go through the owner's
source-declaration-first selector (direct integer -> F32 rounding). With ``verify=True`` (a carried precision
retained on repackaging) the original values go through the exact-F64 conversion only and the stored result is
compared with them, so a retained integer (list, object/integer array) that the selected storage would change
refuses exactly like the equivalent float array. This supersedes the carried-precision statement of READY-13.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.numeric import ExactBlock, NumericContractError, PrecisionRecord
from nirs4all_io.materialize.package import DatasetPackage, FeatureMatrix

INEXACT = 16777217  # 2**24 + 1: not a binary32 value
STORED = 16777216  # 2**24: exactly stored
STORED_HEX = "0000804b"


def _record(source: str | None, request: str | None) -> PrecisionRecord:
    return PrecisionRecord.select(source, request, None, feature=True)


def _hex(matrix: Any, record: PrecisionRecord, *, verify: bool) -> str:
    return ExactBlock.with_precision(matrix, record, verify=verify).values_bytes().hex()


RETAINED_FORMS = {
    "list": lambda v: [[v]],
    "tuple": lambda v: ((v,),),
    "object-array": lambda v: np.array([[v]], dtype=object),
    "int-array": lambda v: np.array([[v]], dtype=np.int64),
    "float-array": lambda v: np.array([[float(v)]]),
}


@pytest.mark.parametrize("form", RETAINED_FORMS)
@pytest.mark.parametrize("source", [None, "float32", "float64", "int64"])
def test_verify_refuses_a_retained_integer_the_float32_storage_would_change(form: str, source: str | None) -> None:
    record = _record(source, "float32")
    with pytest.raises(NumericContractError, match="no longer matches"):
        _hex(RETAINED_FORMS[form](INEXACT), record, verify=True)
    assert _hex(RETAINED_FORMS[form](STORED), record, verify=True) == STORED_HEX, "an exactly stored value is accepted"


def test_the_first_selection_still_rounds_the_original_integer_once() -> None:
    for source in (None, "float32", "int64"):
        record = _record(source, "float32")
        assert _hex([[INEXACT]], record, verify=False) == STORED_HEX
        assert _hex(np.array([[INEXACT]], dtype=np.int64), record, verify=False) == STORED_HEX
    n = 2**54 + 2**30 + 1
    assert _hex([[n]], _record(None, "float32"), verify=False) == "0100805a", "direct upper rounding, not the lower tie"
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        _hex([[n]], _record("float64", "float32"), verify=False)  # source declaration first


def test_verify_with_default_and_float32_source_precisions() -> None:
    default = _record(None, None)
    assert _hex([[2**54]], default, verify=True) == np.float64(2**54).tobytes().hex()
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        _hex([[2**53 + 1]], default, verify=True)
    declared32 = _record("float32", None)
    assert declared32.dtype == "float32"
    with pytest.raises(NumericContractError, match="no longer matches"):
        _hex([[INEXACT]], declared32, verify=True)
    assert _hex([[STORED]], declared32, verify=True) == STORED_HEX
    declared64 = _record("float64", None)
    assert _hex([[2**54]], declared64, verify=True) == np.float64(2**54).tobytes().hex()
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        _hex([[2**53 + 1]], declared64, verify=True)


def test_verify_keeps_partial_missing_masks_and_host_float64_bits() -> None:
    record = _record(None, "float32")
    block = ExactBlock.with_precision([[STORED, float("nan")], [0.5, 2.5]], record, verify=True)
    assert block.validity.tolist() == [[True, False], [True, True]] and block.values[0, 1] == 0
    with pytest.raises(NumericContractError, match="no longer matches"):
        ExactBlock.with_precision([[INEXACT, float("nan")]], record, verify=True)
    with pytest.raises(NumericContractError, match="no longer matches"):
        ExactBlock.with_precision([[INEXACT, None]], record, verify=True)
    host = ExactBlock.with_precision(np.array([[0.5, np.nan]]), record, verify=True)
    assert host.validity.tolist() == [[True, False]]
    stored = ExactBlock.from_matrix([[INEXACT, 0.5]], source_dtype=None, request="float32", consumer=None, feature=True)
    again = ExactBlock.with_precision(stored.to_ir(), stored.precision, verify=True)
    assert again.values_bytes() == stored.values_bytes(), "re-verifying the genuinely stored IR is the identity"


# --------------------------------------------------------------------------- #
# The public package repackaging route                                         #
# --------------------------------------------------------------------------- #
def _package(features: str | None = "float32") -> DatasetPackage:
    return nio.to_dataset_package({"X": [[1.0]], "y": np.array([1.0])}, features=features)


def _set_x(assembled: Any, value: Any) -> None:
    """Replace the carried feature IR of the first source with raw (non-array) cells."""
    blocks: Any = assembled.blocks["train"].X
    blocks[0] = value


def _x(package: DatasetPackage) -> FeatureMatrix:
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload


def test_repackaging_refuses_a_modified_carried_integer_and_keeps_an_exact_one() -> None:
    package = _package()
    builders: list[Callable[[int], Any]] = [lambda v: [[v]], lambda v: np.array([[v]], dtype=object), lambda v: np.array([[v]], dtype=np.int64)]
    for forms in builders:
        bad = package.to_assembled()
        _set_x(bad, forms(INEXACT))
        with pytest.raises(NumericContractError, match="no longer matches"):
            DatasetPackage.from_assembled(bad)
        good = package.to_assembled()
        _set_x(good, forms(STORED))
        assert _x(DatasetPackage.from_assembled(good)).matrix.values_bytes().hex() == STORED_HEX
    floats = package.to_assembled()
    _set_x(floats, np.array([[float(INEXACT)]]))
    with pytest.raises(NumericContractError, match="no longer matches"):
        DatasetPackage.from_assembled(floats)


def test_repackaging_an_unmodified_carried_package_and_first_selection_controls() -> None:
    package = _package()
    again = DatasetPackage.from_assembled(package.to_assembled())
    assert again.to_full_dict() == package.to_full_dict()
    first = nio.to_dataset_package({"X": [[INEXACT]], "y": np.array([1.0])}, features="float32")
    assert _x(first).matrix.values_bytes().hex() == STORED_HEX, "the first selection of a raw integer may round once"
    default = _package(None)
    wide = default.to_assembled()
    _set_x(wide, [[2**54]])
    assert _x(DatasetPackage.from_assembled(wide)).matrix.values.ravel().tolist() == [float(2**54)]
    _set_x(wide, [[2**53 + 1]])
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        DatasetPackage.from_assembled(wide)


def test_repackaging_keeps_partial_masks_and_an_explicit_new_request() -> None:
    package = nio.to_dataset_package({"X": np.array([[1.0, np.nan], [2.0, 3.0]]), "y": np.array([1.0, 2.0])}, features="float32", targets=None)
    assembled = package.to_assembled()
    _set_x(assembled, np.array([[float(STORED), np.nan], [2.0, 3.0]]))
    kept = DatasetPackage.from_assembled(assembled)
    assert _x(kept).matrix.validity.tolist() == [[True, False], [True, True]]
    _set_x(assembled, [[INEXACT, np.nan], [2.0, 3.0]])
    with pytest.raises(NumericContractError, match="no longer matches"):
        DatasetPackage.from_assembled(assembled)
    # an explicit NEW request narrowing exact F64 storage is a genuine first rounding (unchanged behavior)
    wide = _package(None).to_assembled()
    _set_x(wide, [[INEXACT]])
    narrowed = DatasetPackage.from_assembled(wide, features="float32")
    assert _x(narrowed).matrix.values_bytes().hex() == STORED_HEX and _x(narrowed).matrix.precision.requested_dtype == "float32"
