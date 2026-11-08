# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Owner numeric storage selection and exact masked numeric transport (Python oracle).

Independent mirror of the accepted Float64 contract
(``PHASE3_IO_FLOAT64_CONTRACT.md`` sections 2-5): one exact F64 execution IR, a
single quantization point for explicit/declared/consumer F32 storage, explicit
validity masks on every block, and the normative little-endian framing
(``B``/``S``/``O``/``P``/``N``). The IR keeps NaN for a missing tabular cell; the
package boundary turns NaN into ``validity=False`` with a canonical +0 / empty
placeholder *before* any payload hashing.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ..strict_json import check_integer

PUBLIC_INTEGER_CAP = 2**53 - 1
CONSUMER_SPECTRODATASET = "SpectroDataset"
STRING_LABEL = "string-label"
SOURCE_DTYPES = frozenset({"float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"})
ORIGINS = ("default", "source-declared", "requested", "consumer")
_ORIGIN_TAGS = {name: tag for tag, name in enumerate(ORIGINS)}
_DTYPE_TAGS = {"float32": 1, "float64": 2, STRING_LABEL: 3}
_PRECISION_KEYS = frozenset({"dtype", "origin", "source_dtype", "requested_dtype", "consumer"})
_BLOCK_KEYS = frozenset({"dtype", "n_rows", "n_cols", "data_bytes_hex", "validity", "precision"})
_HEX = re.compile(r"(?:[0-9a-f]{2})*\Z")


class NumericContractError(ValueError):
    """A request, value or wire object violates the Float64 owner contract."""


class NumericStorage(StrEnum):
    """Supported numeric storage dtypes."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"


def _storage(value: Any, label: str) -> str | None:
    if value is None:
        return None
    text = value.value if isinstance(value, NumericStorage) else value
    if text not in ("float32", "float64"):
        raise NumericContractError(f"Unsupported {label} numeric storage {value!r}")
    return str(text)


@dataclass(frozen=True)
class NumericStorageOptions:
    """Per-role storage requests. ``None`` means no request (source/default policy).

    A role request applies independently to every supplied block of that role;
    there is no per-source or per-column override. ``consumer`` names an existing
    consumer boundary (currently only ``SpectroDataset``).
    """

    features: NumericStorage | str | None = None
    targets: NumericStorage | str | None = None
    weights: NumericStorage | str | None = None
    consumer: str | None = None

    def __post_init__(self) -> None:
        for role in ("features", "targets", "weights"):
            object.__setattr__(self, role, _storage(getattr(self, role), role))
        if self.consumer is not None and self.consumer != CONSUMER_SPECTRODATASET:
            raise NumericContractError("Unsupported numeric consumer")


def effective_coercion(options: NumericStorageOptions | None) -> NumericStorageOptions:
    """The role requests the RAW numeric coercion needs, derived from the real request.

    The SpectroDataset consumer stores features as float32, so it converts integer sources
    directly like an explicit float32 request. This only steers the raw coercion; it never
    replaces the caller's request or consumer in the precision records.
    """
    options = options or NumericStorageOptions()
    if options.features is None and options.consumer == CONSUMER_SPECTRODATASET:
        return NumericStorageOptions("float32", options.targets, options.weights)
    return NumericStorageOptions(options.features, options.targets, options.weights)


# --------------------------------------------------------------------------- #
# Normative primitive framing                                                  #
# --------------------------------------------------------------------------- #
def u64(value: int) -> bytes:
    """Unsigned 64-bit little-endian count."""
    if type(value) is not int or not 0 <= value < 2**64:
        raise NumericContractError("Count exceeds U64")
    return value.to_bytes(8, "little")


def blob(data: bytes) -> bytes:
    """``B(b)``: U64 length prefix then bytes."""
    return u64(len(data)) + data


def string(text: str) -> bytes:
    """``S(s)``: strict UTF-8, no normalization."""
    try:
        return blob(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise NumericContractError("String is not valid UTF-8") from error


def option_string(text: str | None) -> bytes:
    """``O_S``: ``00`` or ``01 || S``."""
    return b"\x00" if text is None else b"\x01" + string(text)


def strings(items: list[str]) -> bytes:
    """``L_S``: U64 count then each ``S``."""
    return u64(len(items)) + b"".join(string(item) for item in items)


# --------------------------------------------------------------------------- #
# Conversion                                                                   #
# --------------------------------------------------------------------------- #
def round_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    """Quantize finite F64 values once into ``dtype`` storage.

    Binary32 uses nearest-even rounding with subnormals and the sign of zero
    preserved; finite overflow and any nonzero value rounding to zero refuse.
    """
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise NumericContractError("Observed numeric values must be finite")
    if dtype == "float64":
        return array
    with np.errstate(over="ignore", under="ignore"):
        rounded = array.astype(np.float32)
    if not np.isfinite(rounded).all():
        raise NumericContractError("Value overflows float32 storage")
    if np.any((array != 0) & (rounded == 0)):
        raise NumericContractError("Nonzero value rounds to zero in float32 storage")
    return rounded


def f32_integer(value: Any) -> int | None:
    """Decode a finite binary32 integer exactly in the integer domain (no float compare)."""
    bits = struct.unpack("<I", struct.pack("<f", float(np.float32(value))))[0]
    exponent = (bits >> 23) & 255
    if exponent == 255:
        return None
    if bits & 0x7FFFFFFF == 0:
        return 0
    significand = bits & 0x7FFFFF if exponent == 0 else (bits & 0x7FFFFF) | (1 << 23)
    shift = -149 if exponent == 0 else exponent - 127 - 23
    if shift >= 0:
        magnitude = significand << shift
    else:
        if significand & ((1 << -shift) - 1):
            return None
        magnitude = significand >> -shift
    return -magnitude if bits >> 31 else magnitude


def integers_to_f64(values: Any) -> np.ndarray:
    """Widen an integer array to the F64 IR, refusing only genuine precision loss.

    Generic numeric integers keep every exactly representable value (2**53 and
    2**54 are exact); only an integer F64 cannot hold (e.g. 2**53 + 1) refuses.
    The public +/-(2**53 - 1) cap is a *classification label* rule and lives in
    :func:`validate_label`, not here.
    """
    array = np.asarray(values)
    wide = array.astype(np.float64)
    if array.dtype.kind not in "iu":
        return wide  # an already decoded float array (NaN = missing) has no integer cells to inspect
    if array.size and (int(array.max()) > 2**53 or int(array.min()) < -(2**53)):
        # Only values beyond 2**53 can lose bits; compare in the integer domain.
        if any(int(a) != int(b) for a, b in zip(array.ravel().tolist(), wide.ravel().tolist(), strict=True)):
            raise NumericContractError("Integers that float64 cannot represent exactly refuse instead of rounding")
    return wide


def int_to_f32(value: int) -> float:
    """One nearest-even rounding of a Python integer (i64/u64 domain) to binary32, widened exactly.

    ``np.float32(int)`` and ``np.array(list, dtype=float32)`` round through binary64 first;
    only an integer ARRAY ``astype`` converts directly.
    """
    return float(np.array([value], dtype=np.int64 if value < 2**63 else np.uint64).astype(np.float32)[0])


def _cells_to_ir(cells: np.ndarray, request: Any) -> np.ndarray:
    """Cell-wise IR of an object array: the decision is per cell (integer vs float), never per inferred dtype.

    An integer cell converts directly to the selected storage (F32: one rounding; otherwise it must be exactly
    representable in binary64); a float cell is already a binary64 value.
    """
    to_f32 = _storage(request, "requested") == "float32"
    out = np.empty(cells.shape, dtype=np.float64)
    for position, cell in np.ndenumerate(cells):
        if isinstance(cell, (int, np.integer)) and not isinstance(cell, (bool, np.bool_)):
            number = check_integer(int(cell))  # the shared Python-int domain [-2**63, 2**64 - 1], before any conversion
            if to_f32:
                out[position] = int_to_f32(number)
                continue
            exact = float(number)
            if int(exact) != number:
                raise NumericContractError("Integers that float64 cannot represent exactly refuse instead of rounding")
            out[position] = exact
        else:
            out[position] = np.nan if cell is None else float(cell)
    return out


def to_ir_f64(values: Any, request: Any = None) -> np.ndarray:
    """The F64 execution IR of a raw numeric array, honouring the role's storage request.

    Integers convert to the selected storage DIRECTLY: with a ``float32`` request one
    nearest-even rounding of the original integer (never integer -> F64 -> F32, whose
    first rounding can turn a near-tie into a tie and flip the result); otherwise the
    exact-F64 gate of :func:`integers_to_f64` applies. The F32 result widens exactly
    into the F64 IR. Floating inputs are already their own declared bits and are never
    reinterpreted.
    """
    # A raw list is inspected cell by cell BEFORE numpy infers a float64 dtype for a mixed int/float list.
    array = np.asarray(values, dtype=object) if isinstance(values, (list, tuple)) else np.asarray(values)
    if array.dtype.kind == "O":
        return _cells_to_ir(array, request)
    if array.dtype.kind in "iu":
        if _storage(request, "requested") == "float32":
            return array.astype(np.float32).astype(np.float64)
        return integers_to_f64(array)
    return np.asarray(array, dtype=np.float64)


def _selected_ir(matrix: Any, precision: PrecisionRecord, *, verify: bool = False) -> np.ndarray:
    """The F64 IR of a matrix under an already selected precision record.

    Raw cells (a list/tuple, an object array or an integer array) are converted cell by cell through
    :func:`to_ir_f64` BEFORE any float cast, following the owner's source-declaration-first order: a declared
    float64 source holds exact F64 values whatever the request (an inexact integer refuses); a declared float32
    source normalizes the original integer once, directly; an integer or untyped source rounds the original integer
    once, directly, when the selected storage is float32. A genuinely typed float array already holds its own
    bits (possibly already rounded by its host) and is never reinterpreted.

    With ``verify`` the matrix claims to ALREADY hold the selected stored values, so the original values are kept
    through the exact-F64 conversion only (an integer F64 cannot hold refuses) and no storage selection or direct
    integer -> F32 rounding runs: the stored result is later compared with these originals.
    """
    array = np.asarray(matrix, dtype=object) if isinstance(matrix, (list, tuple)) else np.asarray(matrix)
    if array.dtype.kind in "Oiu":
        direct_f32 = not verify and precision.source_dtype != "float64" and (precision.source_dtype == "float32" or precision.dtype == "float32")
        return to_ir_f64(array, "float32" if direct_f32 else None)
    return np.asarray(array, dtype=np.float64)


def validate_label(label: int, dtype: str) -> None:
    """Public classification label gate: cap, then exact decoded F32 integer equality."""
    if abs(label) > PUBLIC_INTEGER_CAP:
        raise NumericContractError("Classification label exceeds portable integer cap")
    if dtype == "float32" and f32_integer(np.float32(label)) != label:
        raise NumericContractError("Classification label is not exact in float32 storage")


# --------------------------------------------------------------------------- #
# Precision record P                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrecisionRecord:
    """The five-field precision record ``P`` (nullable fields serialized as null)."""

    dtype: str = "float64"
    origin: str = "default"
    source_dtype: str | None = None
    requested_dtype: str | None = None
    consumer: str | None = None

    @classmethod
    def select(cls, source: str | None, request: str | None, consumer: str | None, *, feature: bool) -> PrecisionRecord:
        """Owner selection: explicit request, else consumer (features only), else declared, else F64."""
        request = _storage(request, "requested")
        if source is not None and source not in SOURCE_DTYPES:
            raise NumericContractError(f"Unsupported numeric source dtype {source!r}")
        if consumer is not None and consumer != CONSUMER_SPECTRODATASET:
            raise NumericContractError("Unsupported numeric consumer")
        consumer_features = feature and consumer == CONSUMER_SPECTRODATASET
        if consumer_features and request == "float64":
            raise NumericContractError("SpectroDataset does not support requested float64 features/processings")
        if request is not None:
            dtype, origin = request, "requested"
        elif consumer_features:
            dtype, origin = "float32", "consumer"
        elif source in ("float32", "float64"):
            dtype, origin = str(source), "source-declared"
        else:
            dtype, origin = "float64", "default"
        return cls(dtype, origin, source, request, consumer)

    @classmethod
    def string_label(cls, source_dtype: str | None, consumer: str | None) -> PrecisionRecord:
        """Record for existing codebook-decoded string target labels."""
        return cls(STRING_LABEL, "default", source_dtype, None, consumer)

    def validate(self, *, feature: bool) -> None:
        """Refuse a record that disagrees with the owner selection rules."""
        if self.dtype == STRING_LABEL:
            if self != PrecisionRecord.string_label(self.source_dtype, self.consumer) or self.consumer not in (None, CONSUMER_SPECTRODATASET):
                raise NumericContractError("String-label precision record is invalid")
            return
        expected = PrecisionRecord.select(self.source_dtype, self.requested_dtype, self.consumer, feature=feature)
        if self != expected:
            raise NumericContractError("Precision record disagrees with owner selection")

    def to_bytes(self) -> bytes:
        """``P = D || U8(origin) || O_S(source) || O_D(requested) || O_S(consumer)``."""
        requested = b"\x00" if self.requested_dtype is None else b"\x01" + bytes([_DTYPE_TAGS[self.requested_dtype]])
        return bytes([_DTYPE_TAGS[self.dtype], _ORIGIN_TAGS[self.origin]]) + option_string(self.source_dtype) + requested + option_string(self.consumer)

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "origin": self.origin, "source_dtype": self.source_dtype, "requested_dtype": self.requested_dtype, "consumer": self.consumer}

    @classmethod
    def from_dict(cls, value: Any) -> PrecisionRecord:
        if not isinstance(value, dict) or set(value) != _PRECISION_KEYS:
            raise NumericContractError("Precision record needs exactly dtype, origin, source_dtype, requested_dtype, consumer")
        dtype, origin = value["dtype"], value["origin"]
        if dtype not in _DTYPE_TAGS or origin not in _ORIGIN_TAGS:
            raise NumericContractError("Unknown precision dtype/origin")
        if value["requested_dtype"] not in (None, "float32", "float64"):
            raise NumericContractError("Unknown requested dtype")
        for key in ("source_dtype", "consumer"):
            if value[key] is not None and not isinstance(value[key], str):
                raise NumericContractError(f"Precision {key} must be a string or null")
        return cls(dtype, origin, value["source_dtype"], value["requested_dtype"], value["consumer"])


# --------------------------------------------------------------------------- #
# Exact numeric / string block N                                               #
# --------------------------------------------------------------------------- #
def _two_d(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim != 2:
        raise NumericContractError(f"{label} must be a two-dimensional matrix")
    return array


@dataclass(frozen=True, eq=False)
class ExactBlock:
    """``N``: dtype, shape, canonical values and an explicit validity mask.

    Numeric values are float32/float64 arrays whose false cells are +0; string
    blocks hold ``str`` cells whose false cells are the empty string.
    """

    precision: PrecisionRecord
    values: np.ndarray
    validity: np.ndarray

    def __post_init__(self) -> None:
        values, validity = _two_d(self.values, "Block values"), _two_d(self.validity, "Block validity")
        if values.shape != validity.shape or validity.dtype != np.bool_:
            raise NumericContractError("Block values/validity shape or mask dtype mismatch")
        if self.precision.dtype == STRING_LABEL:
            if values.dtype != object or any(cell != "" for cell in values[~validity]) or any(not isinstance(cell, str) or cell == "" for cell in values[validity]):
                raise NumericContractError("String-label block needs nonempty observed labels and empty false cells")
        else:
            if values.dtype != np.dtype(self.precision.dtype):
                raise NumericContractError("Block values dtype disagrees with its precision record")
            if not np.isfinite(values).all():
                raise NumericContractError("Block numeric payload must be finite")
            if any(np.ascontiguousarray(values[~validity]).tobytes()):
                raise NumericContractError("False validity cells must hold canonical +0")

    @property
    def dtype(self) -> str:
        return self.precision.dtype

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def from_matrix(cls, matrix: Any, *, source_dtype: str | None, request: Any, consumer: str | None, feature: bool) -> ExactBlock:
        """Owner storage selection and the single quantization of an F64 IR matrix.

        NaN is the tabular missing marker: it becomes an explicit false mask cell
        with +0 *before* quantization. Infinity refuses. Declared F32 sources
        are normalized once, then the selected storage is applied once.
        """
        precision = PrecisionRecord.select(source_dtype, request, consumer, feature=feature)
        return cls.with_precision(matrix, precision, verify=False)

    @classmethod
    def with_precision(cls, matrix: Any, precision: PrecisionRecord, *, verify: bool) -> ExactBlock:
        """Store an F64 IR matrix under an ALREADY selected precision record.

        With ``verify`` the IR must already be the stored values (it came out of a
        previous selection): re-quantizing would be a second rounding, so any value
        the selected storage would change refuses instead of silently merging again.
        """
        if precision.dtype == STRING_LABEL:
            raise NumericContractError("String-label records have no numeric storage")
        values = _two_d(_selected_ir(matrix, precision, verify=verify), "Numeric block")
        if np.isinf(values).any():
            raise NumericContractError("Observed numeric values must be finite")
        validity = ~np.isnan(values)
        canonical = np.where(validity, values, 0.0)
        stored = canonical
        if precision.source_dtype == "float32":
            stored = round_storage(stored, "float32").astype(np.float64)
        stored = round_storage(stored, precision.dtype)
        if verify and np.ascontiguousarray(stored.astype(np.float64)).tobytes() != np.ascontiguousarray(canonical).tobytes():
            raise NumericContractError("The execution IR no longer matches its selected storage; select the precision from the original source")
        return cls(precision, stored.astype(precision.dtype, copy=False), validity)

    @classmethod
    def from_labels(cls, labels: list[list[str | None]], precision: PrecisionRecord) -> ExactBlock:
        """String-label block from decoded labels (``None`` = missing -> empty string)."""
        rows = len(labels)
        cols = len(labels[0]) if rows else 0
        values = np.empty((rows, cols), dtype=object)
        validity = np.zeros((rows, cols), dtype=bool)
        for r, row in enumerate(labels):
            for c, label in enumerate(row):
                if label is None:
                    values[r, c] = ""
                    continue
                if not isinstance(label, str) or label == "":
                    raise NumericContractError("Observed string labels must be nonempty strings")
                values[r, c], validity[r, c] = label, True
        return cls(precision, values, validity)

    def to_ir(self) -> np.ndarray:
        """The F64 execution IR view: NaN marks a false cell (tabular missing)."""
        if self.dtype == STRING_LABEL:
            raise NumericContractError("String-label blocks have no numeric IR")
        return np.where(self.validity, self.values.astype(np.float64), np.nan)

    def column(self, index: int) -> ExactBlock:
        return ExactBlock(self.precision, self.values[:, index : index + 1], self.validity[:, index : index + 1])

    def values_bytes(self) -> bytes:
        """The exact ``N.values`` bytes (IEEE little-endian or concatenated ``S`` labels)."""
        if self.dtype == STRING_LABEL:
            return b"".join(string(cell) for cell in self.values.ravel(order="C"))
        return np.ascontiguousarray(self.values, dtype="<f4" if self.dtype == "float32" else "<f8").tobytes(order="C")

    def to_bytes(self) -> bytes:
        """``N = D || U64(rows) || U64(cols) || B(values) || B(mask)``."""
        mask = np.ascontiguousarray(self.validity, dtype=np.uint8).tobytes(order="C")
        return bytes([_DTYPE_TAGS[self.dtype]]) + u64(self.n_rows) + u64(self.n_cols) + blob(self.values_bytes()) + blob(mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "data_bytes_hex": self.values_bytes().hex(),
            "validity": self.validity.ravel(order="C").tolist(),
            "precision": self.precision.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any, *, feature: bool) -> ExactBlock:
        """Strict full-wire reader: exact fields, canonical bytes, validated precision."""
        if not isinstance(value, dict) or set(value) != _BLOCK_KEYS:
            raise NumericContractError("Exact block needs exactly dtype, n_rows, n_cols, data_bytes_hex, validity, precision")
        rows, cols = value["n_rows"], value["n_cols"]
        if type(rows) is not int or type(cols) is not int or rows < 0 or cols < 0:
            raise NumericContractError("Block dimensions must be nonnegative integers")
        precision = PrecisionRecord.from_dict(value["precision"])
        if value["dtype"] != precision.dtype:
            raise NumericContractError("Block dtype disagrees with its precision record")
        precision.validate(feature=feature)
        mask = value["validity"]
        if not isinstance(mask, list) or len(mask) != rows * cols or any(type(cell) is not bool for cell in mask):
            raise NumericContractError("Validity must be a flat boolean list of rows*cols entries")
        text = value["data_bytes_hex"]
        if not isinstance(text, str) or not _HEX.fullmatch(text):
            raise NumericContractError("data_bytes_hex must be lowercase hexadecimal bytes")
        data = bytes.fromhex(text)
        validity = np.array(mask, dtype=bool).reshape(rows, cols)
        if precision.dtype == STRING_LABEL:
            cells: list[str] = []
            position = 0
            while position < len(data):
                if position + 8 > len(data):
                    raise NumericContractError("Truncated string-label length")
                length = int.from_bytes(data[position : position + 8], "little")
                position += 8
                if position + length > len(data):
                    raise NumericContractError("Truncated string-label bytes")
                try:
                    cells.append(data[position : position + length].decode("utf-8", errors="strict"))
                except UnicodeDecodeError as error:
                    raise NumericContractError("String label is not valid UTF-8") from error
                position += length
            if len(cells) != rows * cols:
                raise NumericContractError("String-label count disagrees with block shape")
            for cell, valid in zip(cells, mask, strict=True):
                if valid != (cell != ""):
                    raise NumericContractError("Observed labels must be nonempty and false cells must be empty")
            grid = np.empty((rows, cols), dtype=object)
            grid.ravel()[:] = cells
            return cls(precision, grid, validity)
        width = 4 if precision.dtype == "float32" else 8
        if len(data) != rows * cols * width:
            raise NumericContractError("Block byte length disagrees with its shape and dtype")
        grid = np.frombuffer(data, dtype="<f4" if width == 4 else "<f8").astype(precision.dtype).reshape(rows, cols)
        return cls(precision, grid, validity)


def select_block(matrix: Any, *, carried: PrecisionRecord | None, source_dtype: str | None, request: Any, consumer: str | None, feature: bool) -> ExactBlock:
    """Select (or retain) the storage of one numeric block.

    Without a carried record the IR is raw and the normal fallible owner selection
    applies. A carried record is the selection a previous owner step already made: it
    is retained exactly unless the caller asks again (an explicit request or a
    consumer). A new selection derives from the ORIGINAL source dtype and refuses to
    leave already F32-normalized data that the source did not declare as F32: those
    lost bits are never pretended recovered.
    """
    if carried is None:
        return ExactBlock.from_matrix(matrix, source_dtype=source_dtype, request=request, consumer=consumer, feature=feature)
    carried.validate(feature=feature)
    if request is None and consumer is None:
        return ExactBlock.with_precision(matrix, carried, verify=True)
    # An omitted new request keeps the carried explicit one; a new explicit request overrides it.
    inherited = request if request is not None else carried.requested_dtype
    chosen = PrecisionRecord.select(carried.source_dtype, inherited, consumer, feature=feature)
    if carried.dtype == "float32" and carried.source_dtype != "float32" and chosen.dtype != "float32":
        raise NumericContractError("Cannot re-select float64 storage for data already normalized to float32 from a different source")
    # Narrowing exact F64 storage is the one genuine first rounding; everything else re-stores bits.
    return ExactBlock.with_precision(matrix, chosen, verify=not (carried.dtype == "float64" and chosen.dtype == "float32"))


def concat_rows(blocks: list[ExactBlock]) -> ExactBlock:
    """Stack same-precision blocks along rows (base-cohort concatenation)."""
    if not blocks or any(block.precision != blocks[0].precision or block.n_cols != blocks[0].n_cols for block in blocks):
        raise NumericContractError("Blocks differ in precision or width across partitions")
    return ExactBlock(blocks[0].precision, np.concatenate([b.values for b in blocks], axis=0), np.concatenate([b.validity for b in blocks], axis=0))
