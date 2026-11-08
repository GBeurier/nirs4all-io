# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Portable raw-content identity, independent of a host JSON emitter.

The tagged tree encodes every finite number as big-endian IEEE754 binary64.
Objects sort Unicode code points; arrays keep their order. Negative zero equals
zero. This is a separate content contract, not the existing IR canonical JSON.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any

from .strict_json import check_integer, decimal_float

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class ContentNotRepresentableError(ValueError):
    """The record is VALID but its whole-dataset identity cannot be expressed in the tagged-binary64 grammar.

    Distinct from an invalid source (shape/cap/declaration errors): only an integer binary64 cannot hold exactly
    raises it, and a caller may then report the optional whole link as unavailable. It is never a license to
    compute an identity from a rounded stand-in.
    """


def metadata_number(value: Any) -> float:
    """Read the profile's explicit decimal numeric column, never hex or underscores.

    Text goes through the SAME checked decimal primitive as every numeric text conversion
    (``strict_json.decimal_float``): overflow and a nonzero mantissa that underflows to zero refuse, genuine zeros
    keep their sign. An integer token binary64 cannot hold exactly also refuses. Ordinary metadata / class string
    fields never reach this function: only the designated numeric column is converted.
    """
    if isinstance(value, str):
        value = value.strip(" \t\n\r\v\f")
        if not _DECIMAL.fullmatch(value):
            raise ValueError("Metadata numeric column requires finite decimal numbers")
        if re.fullmatch(r"[+-]?[0-9]+", value):
            # an integer token stays an integer until the exact-F64 gate: it is never rounded to the nearest float
            integer = check_integer(int(value))  # the shared integer domain [-2**63, 2**64 - 1]
            exact = float(integer)
            if int(exact) != integer:
                raise ValueError("Metadata numeric column integers must be exactly representable in binary64")
            return -0.0 if integer == 0 and value.startswith("-") else exact  # a true '-0' keeps its sign for the zero folding
        return decimal_float(value)  # raises NumberDomainError (a ValueError) on overflow / underflow
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Metadata numeric column requires finite decimal numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Metadata numeric column requires finite decimal numbers")
    return number


def canonical_content_tree(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Content numbers must be finite")
        if type(value) is int and int(number) != value:
            raise ContentNotRepresentableError("Content integers must be exactly representable in binary64")
        return ["number", struct.pack(">d", 0.0 if number == 0 else number).hex()]
    if isinstance(value, str):
        value.encode("utf-8")
        return ["string", value]
    if isinstance(value, list):
        return ["array", [canonical_content_tree(cell) for cell in value]]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return ["object", [[key, canonical_content_tree(value[key])] for key in sorted(value)]]
    raise ValueError("Content must be a JSON value")


def canonical_content_bytes(value: Any) -> bytes:
    return json.dumps(canonical_content_tree(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _cells(value: Any):
    if isinstance(value, list):
        for cell in value:
            yield from _cells(cell)
    else:
        yield value


def dataset_content_bytes(record: Mapping[str, Any]) -> bytes:
    """Hash logical raw values; text storage widths are not logical types."""
    from .public_dataset import dataset
    value = dataset(record).to_dict()
    raw = value["dataset"]
    y = raw.get("y")
    if raw.get("task_type") == "classification" and y is not None and y["dtype"] == "int64":
        from .materialize.numeric import validate_label

        # the owner's whole operation applies its label gate (no storage option) to every int64 class label,
        # observed or withheld; regression targets and group/unit arrays are values, not labels
        for label in _cells(y["values"]):
            validate_label(label, "float64")
    u07 = [source["name"] for source in raw["sources"]] == ["nir", "image", "series", "metadata"]
    for source in raw["sources"]:
        source["axis_units"] = {key: unit for key, unit in source["axis_units"].items() if unit is not None}
        if source["representation_id"] == "tabular_mixed":
            source["array"]["dtype"] = "object"
            if u07:
                for row in source["array"]["values"]:
                    row[0] = metadata_number(row[0])
    for field in ("partitions", "groups", "y"):
        array = raw.get(field)
        if array is not None and (re.fullmatch(r"<U[0-9]+", array["dtype"]) or array["dtype"] == "object" and all(isinstance(cell, str) for cell in _cells(array["values"]))):
            array["dtype"] = "string"
    return canonical_content_bytes(value)


DATASET_CONTENT_PROFILE = "nirs4all.io.dataset-content.v1"


def labelled_dataset_content_bytes(value: Any) -> bytes:
    """The whole-dataset tagged-binary64 bytes: no label prefix; framing, SHA-256 and zero folding are retained, and the corrected V1 normalization has documented pre-V1 behavior deltas (classes A-F in COPY_PROVENANCE.md)."""
    return dataset_content_bytes(value)


def labelled_dataset_content_identity(value: Any) -> dict[str, str]:
    """Label the SHA-256 of the whole-dataset bytes (never a receipt, root or caller hash); framing, SHA-256 and zero folding are retained and the corrected V1 normalization has documented pre-V1 behavior deltas (classes A-F in COPY_PROVENANCE.md)."""
    return {"profile": DATASET_CONTENT_PROFILE, "algorithm": "sha256", "fingerprint": hashlib.sha256(labelled_dataset_content_bytes(value)).hexdigest()}


def canonical_source_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(schema))
    descriptor = json.loads(value["identity"])
    descriptor["axis_units"] = {key: unit for key, unit in descriptor["axis_units"].items() if unit is not None}
    if value["representation_id"] == "tabular_mixed":
        value["dtype"] = descriptor["dtype"] = "object"
    value["identity"] = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return value


def compatible_source_schemas(current: Mapping[str, Any], saved: Mapping[str, Any]) -> dict[str, Any]:
    if set(current) != set(saved):
        raise ValueError("Source schema names differ")
    for name in current:
        a, b = canonical_source_schema(current[name]), canonical_source_schema(saved[name])
        # Canonical content removes differences in integer/float spelling too.
        a["identity"], b["identity"] = json.loads(a["identity"]), json.loads(b["identity"])
        if canonical_content_bytes(a) != canonical_content_bytes(b):
            raise ValueError(f"Source {name} schema differs (shape, dtype, axes, units, coordinates or columns)")
    return copy.deepcopy(dict(saved))
