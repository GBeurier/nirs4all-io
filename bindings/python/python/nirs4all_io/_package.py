# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Target-agnostic DatasetPackage surface for the pyo3 binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ._adapter import _decode_targets, _matrix
from ._assembled import require_assembled_dataset

__all__ = [
    "DatasetPackage",
    "PayloadManifest",
    "PayloadManifestEntry",
    "PayloadStorageKind",
    "RowPositionFallback",
    "repr_ids",
]


class repr_ids:
    """Representation-id strings shared by the DatasetPackage surfaces."""

    SIGNAL_1D = "signal_1d"
    SIGNAL_WITH_PROCESSINGS = "signal_with_processings"
    FEATURE_BLOCK_SET = "feature_block_set"
    TARGET_NUMERIC = "target_numeric"
    TARGET_CATEGORICAL = "target_categorical"
    TARGET_NUMERIC_MATRIX = "target_numeric_matrix"
    TARGET_CATEGORICAL_MATRIX = "target_categorical_matrix"
    SAMPLE_METADATA = "sample_metadata"
    GRAY_IMAGE = "gray_image"
    RGB_IMAGE = "rgb_image"
    MC_IMAGE = "mc_image"
    MULTISPECTRAL_IMAGE = "multispectral_image"


class PayloadStorageKind(StrEnum):
    """Where a package payload is stored."""

    INLINE = "inline"
    URI = "uri"


@dataclass(frozen=True)
class PayloadManifestEntry:
    """A bytes-free manifest row for one package payload."""

    id: str
    partition: str
    role: str
    payload_kind: str
    representation_id: str | None
    dtype: str
    shape: list[int]
    axes: list[str]
    content_hash: str
    byte_len: int
    storage: PayloadStorageKind = PayloadStorageKind.INLINE
    uri: str | None = None
    codec: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "partition": self.partition,
            "role": self.role,
            "payload_kind": self.payload_kind,
            "representation_id": self.representation_id,
            "dtype": self.dtype,
            "shape": self.shape,
            "axes": self.axes,
            "content_hash": self.content_hash,
            "byte_len": self.byte_len,
            "storage": self.storage.value,
            "uri": self.uri,
            "codec": self.codec,
        }


@dataclass(frozen=True)
class PayloadManifest:
    """A package payload manifest and its root fingerprint."""

    root: str
    entries: list[PayloadManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "entries": [entry.to_dict() for entry in self.entries]}


@dataclass(frozen=True)
class RowPositionFallback:
    """Whether sample identity fell back to row position."""

    used: bool
    reason: str
    partitions: list[str]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "reason": self.reason,
            "partitions": self.partitions,
            "fingerprint": self.fingerprint,
        }


@dataclass
class _PartitionBlock:
    n_samples: int
    source_ids: list[str]
    X: list[np.ndarray]
    feature_headers: list[list[str]]
    header_units: list[str]
    signal_types: list[str | None]
    y: np.ndarray | None
    y_headers: list[str]
    y_categorical: dict[str, Any]
    metadata: pd.DataFrame | None
    weights: np.ndarray | None
    weights_header: str | None
    processings: list[list[tuple[str, np.ndarray]]]


@dataclass
class _AssembledDataset:
    name: str
    task_type: str
    signal_type: str
    n_sources: int
    blocks: dict[str, _PartitionBlock]
    folds: list[tuple[list[int], list[int]]]
    fold_provenance: list[dict[str, list[str]]]
    identity: dict[str, Any]
    repetition: str | None
    aggregate: dict[str, Any] | None


class DatasetPackage:
    """Target-agnostic package built from the native full assembled export.

    The pyo3 binding does not reimplement resolving, loading or joining. It asks
    the Rust core for ``assembled_full`` and wraps that result with the same
    public helper names as the Python MVP package API.
    """

    def __init__(self, full: dict[str, Any]) -> None:
        require_assembled_dataset(full)
        self._full = full
        self.name = str(full.get("name") or "dataset")
        self.task_type = str(full.get("task_type") or "auto")
        self.signal_type = str(full.get("signal_type") or "auto")
        self.n_sources = int(full.get("n_sources") or 0)
        self.repetition = full.get("repetition")
        self.aggregate = full.get("aggregate")
        self.folds = [(list(train), list(val)) for train, val in full.get("folds", [])]
        self.fold_provenance = [dict(fold) for fold in full.get("fold_provenance", [])]
        identity = full.get("identity") or {}
        self.identity = dict(identity.get("provenance") or {})
        self.row_position_fallback = _row_position_fallback(full)

    def manifest(self) -> PayloadManifest:
        entries: list[PayloadManifestEntry] = []
        for partition in self._full["partition_order"]:
            block = self._full["blocks"][partition]
            entries.extend(_manifest_entries(str(partition), block, task_type=self.task_type))
        root = _sha256_json([entry.to_dict() for entry in entries])
        return PayloadManifest(root=root, entries=entries)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "repetition": self.repetition,
            "folds": [[list(train), list(val)] for train, val in self.folds],
            "fold_provenance": [dict(fold) for fold in self.fold_provenance],
            "aggregate": self.aggregate,
            "partitions": {
                str(name): {
                    "n_samples": int(block.get("n_samples") or 0),
                    "source_ids": list(block.get("source_ids") or []),
                }
                for name, block in self._full.get("blocks", {}).items()
            },
            "manifest": self.manifest().to_dict(),
            "identity": {
                "provenance": dict(self.identity),
                "row_position_fallback": self.row_position_fallback.to_dict(),
            },
        }

    def to_canonical_summary(self) -> str:
        return _canonical_json(self.to_summary_dict())

    def to_assembled(self) -> _AssembledDataset:
        blocks = {
            str(partition): _block_from_full(block)
            for partition in self._full["partition_order"]
            for block in [self._full["blocks"][partition]]
        }
        return _AssembledDataset(
            name=self.name,
            task_type=self.task_type,
            signal_type=self.signal_type,
            n_sources=self.n_sources,
            blocks=blocks,
            folds=self.folds,
            fold_provenance=self.fold_provenance,
            identity=self.identity,
            repetition=self.repetition,
            aggregate=self.aggregate,
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ": "),
    ) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return value


def _blob(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _string(value: str) -> bytes:
    return _blob(value.encode("utf-8"))


def _option_string(value: str | None) -> bytes:
    return b"\0" if value is None else b"\1" + _string(value)


def _strings(values: list[str]) -> bytes:
    return len(values).to_bytes(8, "little") + b"".join(_string(value) for value in values)


def _precision(matrix: dict[str, Any]) -> bytes:
    record = matrix["precision"]
    tags = {"float32": 1, "float64": 2, "string-label": 3}
    origins = {"default": 0, "source-declared": 1, "requested": 2, "consumer": 3}
    request = record["requested_dtype"]
    return (bytes([tags[record["dtype"]], origins[record["origin"]]])
            + _option_string(record["source_dtype"])
            + (b"\0" if request is None else bytes([1, tags[request]]))
            + _option_string(record["consumer"]))


def _matrix_bytes(matrix: dict[str, Any]) -> bytes:
    # Serialize declared owner bytes, including masks; never choose precision here.
    return (bytes([{"float32": 1, "float64": 2, "string-label": 3}[matrix["dtype"]]])
            + matrix["n_rows"].to_bytes(8, "little") + matrix["n_cols"].to_bytes(8, "little")
            + _blob(bytes.fromhex(matrix["data_bytes_hex"])) + _blob(bytes(matrix["validity"])))


def _payload_head(role: str) -> bytes:
    return _string("nirs4all.io.package-payload.v4") + (4).to_bytes(4, "little") + _string(role)


def _feature_content(block: dict[str, Any], index: int) -> bytes:
    matrix = block["x"][index]
    names = block["feature_names"][index]
    axes = block["feature_axes"][index]
    content = (_payload_head("features") + _string(block["source_ids"][index])
               + _string(block["representation_ids"][index]) + _precision(matrix) + _matrix_bytes(matrix)
               + (b"\0" if names is None else b"\1" + _strings(names)) + len(axes).to_bytes(8, "little"))
    for axis in axes:
        content += _string(axis["name"]) + _option_string(axis["unit"])
        coordinates = axis["coordinates"]
        if coordinates is None:
            content += b"\0"
        elif coordinates["kind"] == "numeric":
            content += b"\1\1" + coordinates["count"].to_bytes(8, "little") + _blob(bytes.fromhex(coordinates["data_bytes_hex"]))
        else:
            content += b"\1\2" + _strings(coordinates["values"])
    processings = block["processings"][index]
    content += len(processings).to_bytes(8, "little")
    for processing in processings:
        content += _string(processing["name"]) + _matrix_bytes(processing["matrix"])
    return content


def _target_content(y: dict[str, Any]) -> bytes:
    content = _payload_head("targets") + len(y["columns"]).to_bytes(8, "little")
    for column in y["columns"]:
        content += (_string(column["name"]) + bytes([{"numeric": 0, "classification_integer": 1, "classification_string": 2}[column["kind"]]])
                    + _precision(column["block"]) + _matrix_bytes(column["block"]))
    return content


def _frame(meta: dict[str, Any] | None) -> pd.DataFrame | None:
    if meta is None:
        return None
    if any(len(column["values"]) != meta["n_rows"] for column in meta.get("columns", [])):
        raise ValueError("Package metadata rows differ from declared row count")
    return pd.DataFrame({str(column["name"]): pd.Series(column["values"], dtype=object) for column in meta.get("columns", [])}, index=range(meta["n_rows"]))


def _frame_content(meta: dict[str, Any]) -> bytes:
    frame = _frame(meta)
    if frame is not None:
        return _metadata_content_bytes(frame)
    return _canonical_json({"n_rows": int(meta.get("n_rows") or 0), "columns": []}).encode("utf-8")


def _metadata_content_bytes(frame: pd.DataFrame) -> bytes:
    columns = []
    for name in frame.columns:
        series = frame[name]
        columns.append(
            {
                "name": str(name),
                "dtype": _series_dtype_label(pd.Series(series.tolist())),
                "values": [_json_safe(value) for value in series.tolist()],
            }
        )
    return _canonical_json({"n_rows": int(len(frame)), "columns": columns}).encode("utf-8")


def _series_dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _target_representation_id(task_type: str, headers: list[str], categorical: dict[str, Any]) -> str | None:
    if not headers:
        return None
    kinds = [header in categorical or task_type in {"binary", "multiclass"} for header in headers]
    if len(headers) == 1:
        return repr_ids.TARGET_CATEGORICAL if kinds[0] else repr_ids.TARGET_NUMERIC
    if all(kind == kinds[0] for kind in kinds):
        return repr_ids.TARGET_CATEGORICAL_MATRIX if kinds[0] else repr_ids.TARGET_NUMERIC_MATRIX
    return None


def _feature_axis_name(unit: str | None) -> str:
    text = (unit or "").lower()
    if "nm" in text or "nanomet" in text or "wavelength" in text:
        return "wavelength"
    if "cm-1" in text or "cm^-1" in text or "1/cm" in text or "wavenumber" in text or "cm\u207b\u00b9" in text:
        return "wavenumber"
    return "feature"


def _entry(
    *,
    partition: str,
    payload_id: str,
    role: str,
    payload_kind: str,
    representation_id: str | None,
    dtype: str,
    shape: list[int],
    axes: list[str],
    content: bytes,
) -> PayloadManifestEntry:
    return PayloadManifestEntry(
        id=payload_id,
        partition=partition,
        role=role,
        payload_kind=payload_kind,
        representation_id=representation_id,
        dtype=dtype,
        shape=shape,
        axes=axes,
        content_hash=_sha256_bytes(content),
        byte_len=len(content),
    )


def _manifest_entries(partition: str, block: dict[str, Any], *, task_type: str) -> list[PayloadManifestEntry]:
    entries: list[PayloadManifestEntry] = []
    processings = block.get("processings") or []
    for idx, matrix in enumerate(block.get("x") or []):
        headers = block.get("feature_headers") or []
        units = block.get("header_units") or []
        proc = processings[idx] if idx < len(processings) else []
        content = _feature_content(block, idx)
        axes = ["sample", "processing", _feature_axis_name(units[idx] if idx < len(units) else None)] if proc else [
            "sample",
            _feature_axis_name(units[idx] if idx < len(units) else None),
        ]
        entries.append(
            _entry(
                partition=partition,
                payload_id=f"{partition}/x{idx}",
                role="features",
                payload_kind="feature_matrix",
                representation_id=repr_ids.SIGNAL_WITH_PROCESSINGS if proc else repr_ids.SIGNAL_1D,
                dtype=matrix["dtype"],
                shape=[int(matrix["n_rows"]), int(matrix["n_cols"])],
                axes=axes,
                content=bytes(content),
            )
        )
    if block.get("y") is not None:
        y = block["y"]
        headers = [str(h) for h in block.get("y_headers", [])]
        categorical = dict(block.get("y_categorical") or {})
        content = _target_content(y)
        columns = y["columns"]
        dtypes = {column["block"]["dtype"] for column in columns}
        dtype = next(iter(dtypes)) if len(dtypes) == 1 else "mixed"
        n_rows = columns[0]["block"]["n_rows"]
        n_cols = len(columns)
        entries.append(
            _entry(
                partition=partition,
                payload_id=f"{partition}/y",
                role="targets",
                payload_kind="target_table",
                representation_id=_target_representation_id(task_type, headers, categorical),
                dtype=dtype,
                shape=[n_rows, n_cols],
                axes=["sample", "target"] if n_cols > 1 else ["sample"],
                content=bytes(content),
            )
        )
    if block.get("metadata") is not None:
        meta = block["metadata"]
        entries.append(
            _entry(
                partition=partition,
                payload_id=f"{partition}/metadata",
                role="metadata",
                payload_kind="metadata_table",
                representation_id=repr_ids.SAMPLE_METADATA,
                dtype="mixed",
                shape=[int(meta.get("n_rows") or 0), len(meta.get("columns", []))],
                axes=["sample", "field"],
                content=_frame_content(meta),
            )
        )
    if block.get("weights") is not None:
        weights = block["weights"]
        content = _payload_head("weights") + _precision(weights) + _matrix_bytes(weights)
        entries.append(
            _entry(
                partition=partition,
                payload_id=f"{partition}/weights",
                role="weights",
                payload_kind="weights",
                representation_id=None,
                dtype=weights["dtype"],
                shape=[weights["n_rows"]],
                axes=["sample"],
                content=bytes(content),
            )
        )
    return entries


def _row_position_fallback(full: dict[str, Any]) -> RowPositionFallback:
    partitions = list(full["partition_order"])
    identity = full.get("identity") or {}
    provenance = identity.get("provenance") or {}
    sample_id = provenance.get("sample_id")
    if isinstance(sample_id, str):
        has_sample_id = all(
            block.get("metadata") is not None
            and sample_id in {str(column.get("name")) for column in block["metadata"].get("columns", [])}
            and any(
                str(column.get("name")) == sample_id
                and len(column.get("values", [])) == int(block.get("n_samples") or 0)
                for column in block["metadata"].get("columns", [])
            )
            for block in full.get("blocks", {}).values()
        )
        if has_sample_id:
            used = False
            reason = f"stable sample-id key '{sample_id}' is aligned in every partition"
        else:
            used = True
            reason = f"sample-id key '{sample_id}' is absent or not aligned in every partition; sample identity falls back to row position"
    else:
        used = True
        reason = "no stable sample-id key declared; sample identity falls back to row position"
    fingerprint = _sha256_json({"used": used, "reason": reason, "partitions": partitions})
    return RowPositionFallback(used=used, reason=reason, partitions=partitions, fingerprint=fingerprint)


def _block_from_full(block: dict[str, Any]) -> _PartitionBlock:
    return _PartitionBlock(
        n_samples=int(block.get("n_samples") or 0),
        source_ids=list(block.get("source_ids") or []),
        X=[_matrix(matrix) for matrix in block.get("x") or []],
        feature_headers=[list(headers) for headers in block.get("feature_headers", [])],
        header_units=list(block.get("header_units", [])),
        signal_types=list(block.get("signal_types", [])),
        y=_decode_targets(block["y"]) if block.get("y") is not None else None,
        y_headers=list(block.get("y_headers", [])),
        y_categorical=dict(block.get("y_categorical") or {}),
        metadata=_frame(block.get("metadata")),
        weights=_matrix(block["weights"]).reshape(-1) if block.get("weights") is not None else None,
        weights_header=block.get("weights_header"),
        processings=[
            [(str(item.get("name")), _matrix(item["matrix"])) for item in source_processings]
            for source_processings in block.get("processings", [])
        ],
    )
