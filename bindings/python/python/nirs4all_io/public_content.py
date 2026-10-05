# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Portable raw-content identity, independent of a host JSON emitter.

The tagged tree encodes every finite number as big-endian IEEE754 binary64.
Objects sort Unicode code points; arrays keep their order. Negative zero equals
zero. This is a separate content contract, not the existing IR canonical JSON.
"""
from __future__ import annotations

import copy
import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def metadata_number(value: Any) -> float:
    """Read the profile's decimal numeric column, never hex or underscores."""
    if isinstance(value, str):
        value = value.strip(" \t\n\r\v\f")
        if not _DECIMAL.fullmatch(value):
            raise ValueError("Metadata numeric column requires finite decimal numbers")
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
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
