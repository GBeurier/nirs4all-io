# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""DatasetPackage v4 over the explicitly versioned AssembledDataset v3 wire.

The package wraps v3 assembly feature, target, metadata, and weight payloads in
exact typed blocks (dtype, shape, canonical bytes, explicit validity mask and
precision record), exposes a bytes-free manifest, and records row-position
fallback. Package v4 and assembled-wire v3 are separate strict contracts:
readers require the exact current version and never dispatch to or fall back on
a retired predecessor.

Payload ``content_hash`` bytes use the normative ``H4`` framing (contract
section 7); metadata and URI payloads and the manifest-root algorithm are
unchanged. The base-cohort feature/target identity of contract section 6 lives
in :mod:`nirs4all_io.materialize.projection`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from ..canonical_json import canonical_json
from ..spec.dataset_spec import AggregateSpec, outlier_threshold_number
from ..spec.enums import AggregateMethod, SpecError
from ..strict_json import NumberDomainError
from .assemble import ASSEMBLED_DATASET_VERSION, AssembledDataset, PartitionBlock, feature_axis_name, tabular_axis
from .numeric import (
    CONSUMER_SPECTRODATASET,
    PUBLIC_INTEGER_CAP,
    STRING_LABEL,
    ExactBlock,
    NumericContractError,
    NumericStorageOptions,
    PrecisionRecord,
    blob,
    integers_to_f64,
    option_string,
    select_block,
    string,
    strings,
    to_ir_f64,
    u64,
    validate_label,
)

DATASET_PACKAGE_VERSION = 4
TARGET_KINDS = ("numeric", "classification_integer", "classification_string")
_CLASSIFICATION_TASKS = frozenset({"classification", "binary", "multiclass"})
_FINITE_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_SUMMARY_KEYS = frozenset({"schema_version", "name", "task_type", "signal_type", "n_sources", "repetition", "folds", "fold_provenance", "aggregate", "partitions", "manifest", "identity"})
_IDENTITY_KEYS = ("source_ids", "sample_id", "observation_id", "repetition_id", "group_id")


class repr_ids:
    """Representation-id strings mirrored from the Rust DatasetPackage model."""

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
    """Where payload bytes live."""

    INLINE = "inline"
    URI = "uri"


@dataclass(frozen=True)
class PayloadManifestEntry:
    """A bytes-free manifest row for one payload block."""

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
        """Return the stable JSON-serializable manifest row."""
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
    """The package payload manifest plus its root fingerprint."""

    root: str
    entries: list[PayloadManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-serializable manifest."""
        return {"root": self.root, "entries": [entry.to_dict() for entry in self.entries]}


@dataclass(frozen=True)
class UriRef:
    """A URI-backed payload pointer."""

    uri: str
    dtype: str
    shape: list[int]
    axes: list[str]
    content_hash: str
    byte_len: int
    codec: str | None = None


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """A dense exact feature matrix payload with its source, representation and axes.

    ``feature_names`` are the semantic original names (``None`` = a genuinely
    unnamed source), distinct from the presentation ``headers``. ``axes`` are the
    source's non-sample axes: ``{name, unit, coordinates}`` with the raw configured
    unit spelling (``None`` when absent), ``coordinates`` being ``None``, a list of
    floats (raw F64 bits) or a list of strings.
    """

    matrix: ExactBlock
    headers: list[str]
    header_unit: str
    signal_type: str | None
    source_id: str
    representation_id: str
    feature_names: list[str] | None
    axes: list[dict[str, Any]]
    processings: list[tuple[str, ExactBlock]] = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class TargetColumn:
    """One scalar target column: original name, kind, and its exact one-column block."""

    name: str
    kind: str
    block: ExactBlock


@dataclass(frozen=True, eq=False)
class TargetTable:
    """A target table payload: ordered exact columns plus the existing codebooks."""

    columns: list[TargetColumn]
    categorical: dict[str, Any] = field(default_factory=dict)
    representation_id: str | None = None

    @property
    def headers(self) -> list[str]:
        return [column.name for column in self.columns]


@dataclass(frozen=True)
class MetadataTable:
    """A sample-aligned metadata table payload."""

    frame: pd.DataFrame


@dataclass(frozen=True, eq=False)
class WeightsVector:
    """Per-sample training weights as an exact ``n x 1`` block."""

    matrix: ExactBlock
    header: str | None = None


@dataclass(frozen=True)
class SpectralRecordSet:
    """Decoded spectroscopy records carried as a package payload."""

    records: list[dict[str, Any]]


@dataclass(frozen=True)
class NdTensor:
    """A URI-backed N-D tensor payload."""

    representation_id: str
    dtype: str
    shape: list[int]
    axis_names: list[str]
    observation_ids: list[str]
    uri: UriRef


@dataclass(frozen=True)
class SequenceBlock:
    """A URI-backed time-series payload."""

    representation_id: str
    dtype: str
    n_channels: int
    lengths: list[int]
    uri: UriRef


@dataclass(frozen=True)
class GenotypeMatrix:
    """A URI-backed genotype matrix descriptor."""

    representation_id: str
    n_variants: int
    n_samples: int
    encoding: str
    uri: UriRef


@dataclass(frozen=True)
class MaskBlock:
    """A URI-backed mask payload."""

    representation_id: str
    dtype: str
    shape: list[int]
    uri: UriRef


PayloadBlock = (
    FeatureMatrix
    | TargetTable
    | MetadataTable
    | WeightsVector
    | SpectralRecordSet
    | NdTensor
    | SequenceBlock
    | GenotypeMatrix
    | MaskBlock
    | UriRef
)


@dataclass(frozen=True)
class PackagePartition:
    """Typed payloads for one dataset partition."""

    n_samples: int
    payloads: list[tuple[str, PayloadBlock]]
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RowPositionFallback:
    """Whether sample identity fell back to row position."""

    used: bool
    reason: str
    partitions: list[str]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-serializable diagnostic."""
        return {
            "used": self.used,
            "reason": self.reason,
            "partitions": self.partitions,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class DatasetPackage:
    """The public v4 dataset package built from an :class:`AssembledDataset`."""

    name: str
    task_type: str
    signal_type: str
    n_sources: int
    repetition: str | None
    aggregate: AggregateSpec | None
    folds: list[tuple[list[int], list[int]]]
    fold_provenance: list[dict[str, list[str]]]
    identity: dict[str, object]
    warnings: list[str]
    audits: list[dict[str, Any]]
    partitions: dict[str, PackagePartition]
    row_position_fallback: RowPositionFallback
    # Process-local evidence of the public record this package was built from
    # (never serialized, never compared): ``(record, source_id, options, masked)``.
    # It is the only authority for the optional whole-dataset identity link.
    source_evidence: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def from_assembled(
        cls,
        assembled: AssembledDataset,
        *,
        features: Any = None,
        targets: Any = None,
        weights: Any = None,
        consumer: str | None = None,
    ) -> DatasetPackage:
        """Build a v4 package from assembly output.

        ``features`` / ``targets`` / ``weights`` are optional per-role storage
        requests (``"float32"`` / ``"float64"``) applied to every block of that
        role; with no request a genuine source declaration is retained, otherwise
        storage is exact F64. ``consumer`` names an existing consumer boundary.
        """
        options = NumericStorageOptions(features, targets, weights, consumer)
        for block in assembled.blocks.values():
            _check_block(block)
        partitions: dict[str, PackagePartition] = {}
        for partition_name, block in assembled.blocks.items():
            payloads: list[tuple[str, PayloadBlock]] = []
            for idx in range(len(block.X)):
                exact, processings = _select_features(block, idx, options)
                headers = list(block.feature_headers[idx]) if idx < len(block.feature_headers) else []
                if len(headers) != exact.n_cols:
                    raise NumericContractError("Feature header count differs from matrix width")
                names, axes, representation_id = _feature_metadata(block, idx, headers, exact.n_cols)
                payloads.append(
                    (
                        f"{partition_name}/x{idx}",
                        FeatureMatrix(
                            matrix=exact,
                            headers=headers,
                            header_unit=block.header_units[idx] if idx < len(block.header_units) else "",
                            signal_type=block.signal_types[idx] if idx < len(block.signal_types) else None,
                            source_id=block.source_ids[idx],
                            representation_id=representation_id,
                            feature_names=names,
                            axes=axes,
                            processings=processings,
                        ),
                    )
                )
            if block.y is not None:
                payloads.append((f"{partition_name}/y", _target_table(block, assembled.task_type, options)))
            if block.metadata is not None:
                payloads.append((f"{partition_name}/metadata", MetadataTable(frame=block.metadata.copy())))
            if block.weights is not None:
                payloads.append((f"{partition_name}/weights", WeightsVector(matrix=_select_weights(block, options), header=block.weights_header)))
            partitions[partition_name] = PackagePartition(n_samples=block.n_samples, payloads=payloads, source_ids=list(block.source_ids))
        return cls(
            name=assembled.name,
            task_type=assembled.task_type,
            signal_type=assembled.signal_type,
            n_sources=assembled.n_sources,
            repetition=assembled.repetition,
            aggregate=assembled.aggregate,
            folds=[(list(train), list(val)) for train, val in assembled.folds],
            fold_provenance=[dict(fold) for fold in assembled.fold_provenance],
            identity=dict(assembled.identity),
            warnings=list(assembled.warnings),
            audits=list(assembled.audits),
            partitions=partitions,
            row_position_fallback=_compute_row_position_fallback(assembled),
        )

    def to_assembled(self) -> AssembledDataset:
        """Reconstruct the v3-expressible :class:`AssembledDataset` (F64 IR, NaN = false mask)."""
        assembled = AssembledDataset(
            name=self.name,
            task_type=self.task_type,
            signal_type=self.signal_type,
            n_sources=self.n_sources,
            repetition=self.repetition,
            aggregate=self.aggregate,
            warnings=list(self.warnings),
            audits=list(self.audits),
        )
        assembled.folds = [(list(train), list(val)) for train, val in self.folds]
        assembled.fold_provenance = [dict(fold) for fold in self.fold_provenance]
        assembled.identity = dict(self.identity)
        for partition_name, partition in self.partitions.items():
            block = PartitionBlock(n_samples=partition.n_samples)
            block.source_ids = list(partition.source_ids)
            for _payload_id, payload in partition.payloads:
                if isinstance(payload, FeatureMatrix):
                    block.X.append(payload.matrix.to_ir())
                    block.x_source_dtypes.append(payload.matrix.precision.source_dtype)
                    block.x_precisions.append(payload.matrix.precision)
                    block.feature_headers.append(list(payload.headers))
                    block.feature_names.append(None if payload.feature_names is None else list(payload.feature_names))
                    block.feature_axes.append(json.loads(json.dumps(payload.axes)))
                    block.representation_ids.append(payload.representation_id)
                    block.header_units.append(payload.header_unit)
                    block.signal_types.append(payload.signal_type)
                    block.processings.append([(name, matrix.to_ir()) for name, matrix in payload.processings])
                    block.processing_precisions.append([matrix.precision for _, matrix in payload.processings])
                    block.processing_source_dtypes.append([matrix.precision.source_dtype for _, matrix in payload.processings])
                elif isinstance(payload, TargetTable):
                    block.y = _target_matrix(payload)
                    block.y_headers = payload.headers
                    block.y_categorical = json.loads(json.dumps(payload.categorical))
                    numeric = [c.block.precision.source_dtype for c in payload.columns if c.kind != TARGET_KINDS[2]]
                    block.y_source_dtype = numeric[0] if numeric else None
                    block.y_precisions = [c.block.precision for c in payload.columns]
                elif isinstance(payload, MetadataTable):
                    block.metadata = payload.frame.copy()
                elif isinstance(payload, WeightsVector):
                    block.weights = payload.matrix.to_ir()[:, 0]
                    block.weights_header = payload.header
                    block.weights_source_dtype = payload.matrix.precision.source_dtype
                    block.weights_precision = payload.matrix.precision
            assembled.blocks[partition_name] = block
        return assembled

    def manifest(self) -> PayloadManifest:
        """Build the payload manifest and root fingerprint (algorithm unchanged)."""
        entries: list[PayloadManifestEntry] = []
        for partition_name, partition in self.partitions.items():
            for payload_id, payload in partition.payloads:
                entries.append(_manifest_entry(payload, partition=partition_name, payload_id=payload_id))
        root = _sha256_json([entry.to_dict() for entry in entries])
        return PayloadManifest(root=root, entries=entries)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a stable, bytes-free package summary (a structural summary only)."""
        aggregate = None
        if self.aggregate is not None:
            aggregate = {
                "by": self.aggregate.by,
                "method": self.aggregate.method.value,
                "exclude_outliers": self.aggregate.exclude_outliers,
                "outlier_threshold": self.aggregate.outlier_threshold,
            }
        return {
            "schema_version": DATASET_PACKAGE_VERSION,
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "repetition": self.repetition,
            "folds": [[list(train), list(val)] for train, val in self.folds],
            "fold_provenance": [dict(fold) for fold in self.fold_provenance],
            "aggregate": aggregate,
            "partitions": {
                name: {"n_samples": partition.n_samples, "source_ids": list(partition.source_ids)}
                for name, partition in self.partitions.items()
            },
            "manifest": self.manifest().to_dict(),
            "identity": {
                "provenance": dict(self.identity),
                "row_position_fallback": self.row_position_fallback.to_dict(),
            },
        }

    def to_canonical_summary(self) -> str:
        """Return the canonical JSON form of :meth:`to_summary_dict`."""
        return canonical_json(self.to_summary_dict())

    def to_full_dict(self) -> dict[str, Any]:
        """Exact v4 full wire: summary outer fields, ``partition_order`` and exact payload bodies.

        ``partition_order`` is required because canonical JSON sorts object keys;
        it is the only carrier of the base cohort's block order.
        """
        value = self.to_summary_dict()
        value["partition_order"] = list(self.partitions)
        value["audits"] = _audits_wire(self.audits)
        value["identity"]["provenance"] = _identity_wire(self.identity)
        value["partitions"] = {
            name: {
                "n_samples": partition.n_samples,
                "source_ids": list(partition.source_ids),
                "payloads": {payload_id: _payload_wire(payload) for payload_id, payload in partition.payloads},
            }
            for name, partition in self.partitions.items()
        }
        return value

    @classmethod
    def from_full_dict(cls, value: Any) -> DatasetPackage:
        """Strict reader of the exact current wire (version 4); nothing older is accepted.

        Block order comes from ``partition_order`` and payload order from the
        manifest entries; the manifest (root, hashes, shapes, axes) is recomputed
        from the decoded payload bytes and must match what was supplied.
        """
        if not isinstance(value, dict) or set(value) != _SUMMARY_KEYS | {"partition_order", "audits"}:
            raise NumericContractError("Package full wire has missing or unknown fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != DATASET_PACKAGE_VERSION:
            raise NumericContractError(f"Unsupported package version {value['schema_version']!r}; exactly {DATASET_PACKAGE_VERSION} is required")
        order = _partition_order(value["partition_order"], value["partitions"])
        manifest_wire = value["manifest"]
        if not isinstance(manifest_wire, dict) or set(manifest_wire) != {"root", "entries"} or not isinstance(manifest_wire["entries"], list):
            raise NumericContractError("Package manifest is malformed")
        entry_order = [(entry["partition"], entry["id"]) for entry in manifest_wire["entries"] if isinstance(entry, dict) and isinstance(entry.get("partition"), str) and isinstance(entry.get("id"), str)]
        if len(entry_order) != len(manifest_wire["entries"]) or len(set(entry_order)) != len(entry_order):
            raise NumericContractError("Package manifest entries are malformed or duplicated")
        if any(not isinstance(value["partitions"][name], dict) or not isinstance(value["partitions"][name].get("payloads"), dict) for name in order):
            raise NumericContractError("Partitions must be objects with a payloads object")
        populated = [name for name in order if value["partitions"][name]["payloads"]]
        seen_partitions: list[str] = []
        for partition_name, _ in entry_order:
            if not seen_partitions or seen_partitions[-1] != partition_name:
                seen_partitions.append(partition_name)
        if seen_partitions != populated:
            raise NumericContractError("Manifest partition order disagrees with partition_order")
        partitions: dict[str, PackagePartition] = {}
        for name in order:
            body = value["partitions"][name]
            if not isinstance(body, dict) or set(body) != {"n_samples", "source_ids", "payloads"}:
                raise NumericContractError(f"Partition {name!r} has missing or unknown fields")
            if type(body["n_samples"]) is not int or body["n_samples"] < 0:
                raise NumericContractError("n_samples must be a nonnegative integer")
            if not isinstance(body["source_ids"], list) or not all(isinstance(s, str) for s in body["source_ids"]) or not isinstance(body["payloads"], dict):
                raise NumericContractError("Partition source_ids/payloads are malformed")
            ids_in_manifest = [payload_id for partition_name, payload_id in entry_order if partition_name == name]
            if set(ids_in_manifest) != set(body["payloads"]):
                raise NumericContractError("Manifest entries and payload IDs disagree")
            payloads = [(payload_id, _payload_from_wire(body["payloads"][payload_id])) for payload_id in ids_in_manifest]
            declared_sources = [p.source_id for _, p in payloads if isinstance(p, FeatureMatrix)]
            if declared_sources != body["source_ids"]:
                raise NumericContractError("Partition source_ids disagree with its feature payloads")
            for _, payload in payloads:
                rows = _payload_rows(payload)
                if rows is not None and rows != body["n_samples"]:
                    raise NumericContractError("Payload rows disagree with n_samples")
            partitions[name] = PackagePartition(n_samples=body["n_samples"], payloads=payloads, source_ids=list(body["source_ids"]))
        identity = value["identity"]
        if not isinstance(identity, dict) or set(identity) != {"provenance", "row_position_fallback"}:
            raise NumericContractError("Package identity wrapper is malformed")
        fallback_wire = identity["row_position_fallback"]
        if not isinstance(fallback_wire, dict) or set(fallback_wire) != {"used", "reason", "partitions", "fingerprint"}:
            raise NumericContractError("Row-position fallback is malformed")
        package = cls(
            name=_text(value["name"], "name"),
            task_type=_text(value["task_type"], "task_type"),
            signal_type=_text(value["signal_type"], "signal_type"),
            n_sources=_count(value["n_sources"], "n_sources"),
            repetition=None if value["repetition"] is None else _text(value["repetition"], "repetition"),
            aggregate=_aggregate_from_wire(value["aggregate"]),
            folds=_folds_from_wire(value["folds"]),
            fold_provenance=_fold_provenance_from_wire(value["fold_provenance"]),
            identity=_identity_from_wire(identity["provenance"]),
            warnings=[],
            audits=_audits_from_wire(value["audits"]),
            partitions=partitions,
            row_position_fallback=RowPositionFallback(
                used=fallback_wire["used"] if type(fallback_wire["used"]) is bool else _bad("fallback used must be a boolean"), reason=_text(fallback_wire["reason"], "reason"), partitions=_string_list(fallback_wire["partitions"], "fallback partitions"), fingerprint=_text(fallback_wire["fingerprint"], "fingerprint")
            ),
        )
        if package.fold_provenance and (
            len(package.fold_provenance) != len(package.folds)
            or any(
                len(train) != len(p["train_observation_ids"]) or len(val) != len(p["validation_observation_ids"])
                for (train, val), p in zip(package.folds, package.fold_provenance, strict=True)
            )
        ):
            raise NumericContractError("folds disagree with fold_provenance")
        if package.row_position_fallback.to_dict() != _compute_row_position_fallback(package.to_assembled()).to_dict():
            raise NumericContractError("Row-position fallback disagrees with the package payloads")
        if package.manifest().to_dict() != manifest_wire:
            raise NumericContractError("Package manifest disagrees with the payload bytes")
        return package

    def executed_projection(self, selected_source_ids: list[str]) -> Any:
        """Base-cohort feature/target preimages, digests and receipt (contract section 6)."""
        from .projection import executed_projection

        return executed_projection(self, selected_source_ids)


def _check_block(block: PartitionBlock) -> None:
    """Structural alignment every selection (package or finalizer) relies on."""
    if len(block.source_ids) != len(block.X) or len(set(block.source_ids)) != len(block.source_ids):
        raise NumericContractError("Missing/duplicate feature source IDs")
    if any(np.asarray(matrix).shape[0] != block.n_samples for matrix in block.X):
        raise NumericContractError("Feature row count mismatch")
    if block.weights is not None and len(block.weights) != block.n_samples:
        raise NumericContractError("Weight row count mismatch")


def _select_features(block: PartitionBlock, idx: int, options: NumericStorageOptions) -> tuple[ExactBlock, list[tuple[str, ExactBlock]]]:
    """Storage selection of one feature source and its named processings (shared by the package and the finalizer)."""
    matrix = block.X[idx]
    source_dtype = block.x_source_dtypes[idx] if idx < len(block.x_source_dtypes) else None
    carried = _carried(block.x_precisions, idx, len(block.X), "x_precisions")
    exact = select_block(matrix, carried=carried, source_dtype=source_dtype, request=options.features, consumer=options.consumer, feature=True)
    processings: list[tuple[str, ExactBlock]] = []
    children = block.processings[idx] if idx < len(block.processings) else []
    child_dtypes = block.processing_source_dtypes[idx] if idx < len(block.processing_source_dtypes) else []
    child_records = _carried_row(block.processing_precisions, idx, len(block.X))
    if (child_dtypes and len(child_dtypes) != len(children)) or (child_records and len(child_records) != len(children)):
        raise NumericContractError("Processing source dtypes / precision records must align with the processings")
    for position, (name, array) in enumerate(children):
        if np.asarray(array).shape != np.asarray(matrix).shape:
            raise NumericContractError("Processing shape differs from feature matrix")
        # Each processing is an independently loaded input: it is selected under ITS OWN
        # declared source dtype (None when undeclared), the same role request/consumer, and
        # must land on the governing feature storage dtype.
        child = select_block(
            array,
            carried=child_records[position] if child_records else None,
            source_dtype=child_dtypes[position] if child_dtypes else None,
            request=options.features,
            consumer=options.consumer,
            feature=True,
        )
        _check_child_precision(exact.precision, child.precision, name)
        processings.append((name, child))
    return exact, processings


def _select_weights(block: PartitionBlock, options: NumericStorageOptions) -> ExactBlock:
    return select_block(
        np.asarray(block.weights).reshape(-1, 1),
        carried=block.weights_precision,
        source_dtype=block.weights_source_dtype,
        request=options.weights,
        consumer=options.consumer,
        feature=False,
    )


def finalize_selected_storage(assembled: AssembledDataset, options: NumericStorageOptions | None) -> AssembledDataset:
    """Commit the REQUESTED storage selection into an assembled dataset (in place) and return it.

    The one seam behind every assembly exit that promises storage options: it runs the
    same role selection as :meth:`DatasetPackage.from_assembled` (shared helpers over
    :func:`select_block`, no second numeric loop) and writes back only what was selected:

    * X and the named processings, with their precision records, when ``features`` or a
      consumer is set;
    * Y and ``y_precisions`` only when ``targets`` was requested;
    * weights only when ``weights`` was requested.

    Unrequested raw targets and weights stay untouched (an SDK int64 label keeps its dtype).
    """
    if options is None or all(value is None for value in (options.features, options.targets, options.weights, options.consumer)):
        return assembled
    select_features = options.features is not None or options.consumer is not None
    for block in assembled.blocks.values():
        _check_block(block)
        if select_features:
            selected = [_select_features(block, idx, options) for idx in range(len(block.X))]
            block.X = [exact.to_ir() for exact, _ in selected]
            block.x_precisions = [exact.precision for exact, _ in selected]
            block.x_source_dtypes = [exact.precision.source_dtype for exact, _ in selected]
            block.processings = [[(name, child.to_ir()) for name, child in children] for _, children in selected]
            block.processing_precisions = [[child.precision for _, child in children] for _, children in selected]
            block.processing_source_dtypes = [[child.precision.source_dtype for _, child in children] for _, children in selected]
        if options.targets is not None and block.y is not None:
            table = _target_table(block, assembled.task_type, options)
            block.y = _target_matrix(table)
            block.y_precisions = [column.block.precision for column in table.columns]
        if options.weights is not None and block.weights is not None:
            weight_block = _select_weights(block, options)
            block.weights = weight_block.to_ir()[:, 0]
            block.weights_precision = weight_block.precision
    return assembled


# --------------------------------------------------------------------------- #
# Target tables, axes and content bytes                                        #
# --------------------------------------------------------------------------- #
def _feature_metadata(block: PartitionBlock, idx: int, headers: list[str], n_cols: int) -> tuple[list[str] | None, list[dict[str, Any]], str]:
    """Exact names / axes / representation of one source, validated against its matrix.

    The three per-source lists are either all present (genuine assembly) or all
    absent on a manually built IR, in which case the current owned headers and
    presentation unit derive them once, before serialization.
    """
    lengths = {len(block.feature_names), len(block.feature_axes), len(block.representation_ids)}
    if lengths == {0}:
        unit = block.header_units[idx] if idx < len(block.header_units) else ""
        raw_unit = unit if unit in ("nm", "cm-1") else None
        return list(headers), [tabular_axis(headers, unit or "text", raw_unit)], repr_ids.SIGNAL_1D
    if lengths != {len(block.X)}:
        raise NumericContractError("feature_names, feature_axes and representation_ids must align with the feature sources")
    names, axes, representation_id = block.feature_names[idx], block.feature_axes[idx], block.representation_ids[idx]
    if names is not None and (not isinstance(names, list) or len(names) != n_cols or not all(isinstance(n, str) for n in names)):
        raise NumericContractError("feature_names must be null or one string per feature column")
    if not isinstance(representation_id, str) or not representation_id:
        raise NumericContractError("representation_ids must be nonempty strings")
    checked: list[dict[str, Any]] = []
    for axis in axes:
        if not isinstance(axis, dict) or set(axis) != {"name", "unit", "coordinates"} or not isinstance(axis["name"], str):
            raise NumericContractError("Axis needs exactly name, unit, coordinates")
        if axis["unit"] is not None and not isinstance(axis["unit"], str):
            raise NumericContractError("Axis unit must be a string or null")
        coordinates = axis["coordinates"]
        if coordinates is not None:
            coordinates = _checked_coordinates(coordinates, n_cols)
        checked.append({"name": axis["name"], "unit": axis["unit"], "coordinates": coordinates})
    return (None if names is None else list(names)), checked, representation_id


def _checked_coordinates(coordinates: Any, n_cols: int) -> list[Any]:
    """Public IR axis coordinates: n_cols finite F64 numbers or n_cols strings (never a mix).

    A numeric coordinate converts through the exact-F64 selector (:func:`to_ir_f64`) BEFORE any float cast:
    an integer F64 holds exactly (2**54) is a valid coordinate and one it cannot (2**53 + 1) refuses; a genuine
    float keeps its bits (the sign of zero included). The portable PublicDatasetV1 axis cap is a different,
    stricter contract and stays in ``public_dataset``.
    """
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != n_cols:
        raise NumericContractError("Axis coordinates must be n_cols finite floats or strings")
    cells = list(coordinates)
    if cells and all(isinstance(c, (int, float, np.integer)) and not isinstance(c, (bool, np.bool_)) for c in cells):
        try:
            values = to_ir_f64(cells)
        except NumberDomainError as error:
            raise NumericContractError(f"Axis coordinate: {error}") from error
        if not np.isfinite(values).all():
            raise NumericContractError("Axis coordinates must be finite")
        return [float(v) for v in values]
    if all(isinstance(c, str) for c in cells):
        return cells
    raise NumericContractError("Axis coordinates must be n_cols finite floats or strings")


def _target_kind(name: str, task_type: str, categorical: dict[str, Any]) -> str:
    if name in categorical:
        return TARGET_KINDS[2]
    return TARGET_KINDS[1] if task_type in _CLASSIFICATION_TASKS else TARGET_KINDS[0]


def _carried_row(rows: list[list[PrecisionRecord]], index: int, expected: int) -> list[PrecisionRecord]:
    if not rows:
        return []
    if len(rows) != expected:
        raise NumericContractError("processing_precisions must align with the feature sources")
    return rows[index]


def _check_child_precision(base: PrecisionRecord, child: PrecisionRecord, name: str) -> None:
    """A named processing shares its source's selected storage dtype and role request/consumer.

    Origins and source dtypes may differ (they describe the child's own input); the H4 frame
    hashes the base ``P`` and the child ``N`` only, so the child's source metadata is carried
    by the full-wire ``ExactBlock`` and is not part of that payload hash.
    """
    if (child.dtype, child.requested_dtype, child.consumer) != (base.dtype, base.requested_dtype, base.consumer):
        raise NumericContractError(
            f"Processing {name!r} stores {child.dtype} (request {child.requested_dtype}) but its feature source stores {base.dtype} "
            f"(request {base.requested_dtype}); pass a common features request so every block resolves to one storage dtype"
        )


def _carried(records: list[PrecisionRecord], index: int, expected: int, label: str) -> PrecisionRecord | None:
    """The carried selected record of one aligned block, or ``None`` on a raw IR."""
    if not records:
        return None
    if len(records) != expected:
        raise NumericContractError(f"{label} must align with the blocks")
    return records[index]


def _target_table(block: PartitionBlock, task_type: str, options: NumericStorageOptions) -> TargetTable:
    y = np.asarray(block.y)
    headers = list(block.y_headers)
    if y.ndim != 2 or y.shape != (block.n_samples, len(headers)):
        raise NumericContractError("Target shape/header mismatch")
    if len(set(headers)) != len(headers):
        raise NumericContractError("Duplicate target names")
    if y.dtype.kind not in "fiu":
        raise NumericContractError(f"Unsupported target dtype {y.dtype}: bool/object/string targets need an existing codebook, no encoder is applied")
    kinds = [_target_kind(name, task_type, block.y_categorical) for name in headers]
    if TARGET_KINDS[2] in kinds and options.targets is not None:
        raise NumericContractError("Numeric precision request for string targets is unsupported")
    # Class labels stay exact (portable cap + exact-F32 rule); other integer targets convert to
    # the selected storage directly, so a float32 request rounds the original integer once.
    values = np.column_stack(
        [integers_to_f64(y[:, i]) if kind == TARGET_KINDS[1] else to_ir_f64(y[:, i], options.targets) for i, kind in enumerate(kinds)]
    ) if len(kinds) else y.astype(np.float64)
    if block.y_precisions and len(block.y_precisions) != len(headers):
        raise NumericContractError("y_precisions must align with the target columns")
    columns: list[TargetColumn] = []
    for index, (name, kind) in enumerate(zip(headers, kinds, strict=True)):
        carried = block.y_precisions[index] if block.y_precisions else None
        if kind == TARGET_KINDS[2]:
            consumer = options.consumer if options.consumer is not None else (carried.consumer if carried is not None else None)
            string_block = _string_column(name, values[:, index], block.y_categorical[name], consumer)
            if carried is not None and string_block.precision != carried and options.consumer is None:
                raise NumericContractError(f"Carried precision of string target {name!r} disagrees with its codebook")
            columns.append(TargetColumn(name, kind, string_block))
            continue
        column = select_block(values[:, index : index + 1], carried=carried, source_dtype=block.y_source_dtype, request=options.targets, consumer=options.consumer, feature=False)
        if column.precision.consumer == CONSUMER_SPECTRODATASET and column.precision.requested_dtype == "float64":
            raise NumericContractError("SpectroDataset cannot guarantee float64 numerical targets (the SDK narrows targets itself); weights are not numerical targets")
        if kind == TARGET_KINDS[1]:
            for label, valid in zip(values[:, index], column.validity[:, 0], strict=True):
                if not valid:
                    continue
                if label != np.floor(label) or abs(label) > PUBLIC_INTEGER_CAP:
                    raise NumericContractError("Classification requires portable integer labels")
                validate_label(int(label), column.dtype)
        columns.append(TargetColumn(name, kind, column))
    return TargetTable(columns=columns, categorical=json.loads(json.dumps(block.y_categorical)), representation_id=_target_representation_id(task_type, headers, block.y_categorical))


def _string_column(name: str, codes: np.ndarray, codebook: Any, consumer: str | None) -> ExactBlock:
    categories = codebook.get("categories") if isinstance(codebook, dict) else None
    if not isinstance(categories, list) or any(not isinstance(label, str) or label == "" for label in categories) or len(set(categories)) != len(categories):
        raise NumericContractError(f"Unsupported or invalid categorical codebook for target {name!r}")
    labels: list[list[str | None]] = []
    for code in codes:
        if np.isnan(code):
            labels.append([None])
        elif code != np.floor(code) or code < 0 or code >= len(categories):
            raise NumericContractError(f"Categorical code out of range for target {name!r}")
        else:
            labels.append([categories[int(code)]])
    return ExactBlock.from_labels(labels, PrecisionRecord.string_label(codebook.get("source_dtype"), consumer))


def _target_matrix(table: TargetTable) -> np.ndarray:
    """The IR target matrix: numeric columns as values, string columns as codebook codes (NaN = missing)."""
    rows = table.columns[0].block.n_rows if table.columns else 0
    out = np.full((rows, len(table.columns)), np.nan, dtype=np.float64)
    for index, column in enumerate(table.columns):
        if column.kind != TARGET_KINDS[2]:
            out[:, index] = column.block.to_ir()[:, 0]
            continue
        categories = table.categorical.get(column.name, {}).get("categories")
        if not isinstance(categories, list):
            raise NumericContractError(f"String target {column.name!r} has no codebook")
        codes = {label: code for code, label in enumerate(categories)}
        for row in range(rows):
            if column.block.validity[row, 0]:
                label = column.block.values[row, 0]
                if label not in codes:
                    raise NumericContractError(f"Label {label!r} is not in the codebook of target {column.name!r}")
                out[row, index] = codes[label]
    return out


def _axis_bytes(axis: dict[str, Any]) -> bytes:
    out = string(axis["name"]) + option_string(axis["unit"])
    coordinates = axis["coordinates"]
    if coordinates is None:
        return out + b"\x00"
    if all(isinstance(value, float) for value in coordinates):
        return out + b"\x01\x01" + u64(len(coordinates)) + blob(np.asarray(coordinates, dtype="<f8").tobytes())
    return out + b"\x01\x02" + strings(coordinates)


def feature_bytes(payload: FeatureMatrix) -> bytes:
    """``Feature = S(source) || S(representation) || P || N || O_(L_S)(names) || L_Axis(axes)``."""
    return (
        string(payload.source_id)
        + string(payload.representation_id)
        + payload.matrix.precision.to_bytes()
        + payload.matrix.to_bytes()
        + (b"\x00" if payload.feature_names is None else b"\x01" + strings(payload.feature_names))
        + u64(len(payload.axes))
        + b"".join(_axis_bytes(axis) for axis in payload.axes)
    )


def target_bytes(column: TargetColumn) -> bytes:
    """``Target = S(name) || U8(kind) || P || N``."""
    return string(column.name) + bytes([TARGET_KINDS.index(column.kind)]) + column.block.precision.to_bytes() + column.block.to_bytes()


def _payload_head(kind: str) -> bytes:
    return string("nirs4all.io.package-payload.v4") + (4).to_bytes(4, "little") + string(kind)


def _feature_content_bytes(payload: FeatureMatrix) -> bytes:
    out = _payload_head("features") + feature_bytes(payload) + u64(len(payload.processings))
    return out + b"".join(string(name) + block.to_bytes() for name, block in payload.processings)


def _target_content_bytes(payload: TargetTable) -> bytes:
    return _payload_head("targets") + u64(len(payload.columns)) + b"".join(target_bytes(column) for column in payload.columns)


def _weights_content_bytes(payload: WeightsVector) -> bytes:
    return _payload_head("weights") + payload.matrix.precision.to_bytes() + payload.matrix.to_bytes()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _metadata_content_bytes(frame: pd.DataFrame) -> bytes:
    return canonical_json(_metadata_body(frame)).encode("utf-8")


def _metadata_body(frame: pd.DataFrame) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        columns.append(
            {
                "name": str(name),
                "dtype": _series_dtype_label(series),
                "values": [_json_scalar(value) for value in series.tolist()],
            }
        )
    return {"n_rows": int(len(frame)), "columns": columns}


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return _json_scalar(value)


def _series_dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series) or _numeric_object_cells(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _numeric_object_cells(series: pd.Series) -> bool:
    """An object column of exact numbers only (ints F64 cannot hold beside floats / NA): a numeric column."""
    if series.dtype != object:
        return False
    cells = [cell for cell in series.tolist() if not (cell is None or (isinstance(cell, float) and cell != cell))]
    return bool(cells) and all(type(cell) in (int, float) for cell in cells)


def _series_from_wire(values: list[Any]) -> pd.Series:
    """A metadata column read back from its JSON values; an integer F64 cannot hold keeps its exact value."""
    inexact = any(float(v) != v for v in values if type(v) is int)
    if inexact and any(type(v) is not int for v in values):
        return pd.Series(values, dtype=object)
    return pd.Series(values)


def _target_representation_id(task_type: str, headers: list[str], categorical: dict[str, Any]) -> str | None:
    if not headers:
        return None
    # A target is categorical when it is a classification kind (codebook strings or integer labels of a
    # classification task); a same-valued regression target stays numeric. The H4 target bytes are unchanged.
    kinds = [_target_kind(header, task_type, categorical) != TARGET_KINDS[0] for header in headers]
    if len(headers) == 1:
        return repr_ids.TARGET_CATEGORICAL if kinds[0] else repr_ids.TARGET_NUMERIC
    if all(kind == kinds[0] for kind in kinds):
        return repr_ids.TARGET_CATEGORICAL_MATRIX if kinds[0] else repr_ids.TARGET_NUMERIC_MATRIX
    return None


def _inline_entry(
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
        storage=PayloadStorageKind.INLINE,
    )


def _uri_entry(
    *,
    partition: str,
    payload_id: str,
    role: str,
    payload_kind: str,
    representation_id: str | None,
    uri: UriRef,
    dtype: str | None = None,
    shape: list[int] | None = None,
    axes: list[str] | None = None,
) -> PayloadManifestEntry:
    return PayloadManifestEntry(
        id=payload_id,
        partition=partition,
        role=role,
        payload_kind=payload_kind,
        representation_id=representation_id,
        dtype=dtype or uri.dtype,
        shape=list(shape or uri.shape),
        axes=list(axes or uri.axes),
        content_hash=uri.content_hash,
        byte_len=uri.byte_len,
        storage=PayloadStorageKind.URI,
        uri=uri.uri,
        codec=uri.codec,
    )


def _manifest_entry(payload: PayloadBlock, *, partition: str, payload_id: str) -> PayloadManifestEntry:
    if isinstance(payload, FeatureMatrix):
        has_processings = bool(payload.processings)
        axes = ["sample"]
        if has_processings:
            axes.append("processing")
        axes.append(feature_axis_name(payload.header_unit))
        return _inline_entry(
            partition=partition,
            payload_id=payload_id,
            role="features",
            payload_kind="feature_matrix",
            representation_id=repr_ids.SIGNAL_WITH_PROCESSINGS if has_processings else repr_ids.SIGNAL_1D,
            dtype=payload.matrix.dtype,
            shape=[payload.matrix.n_rows, payload.matrix.n_cols],
            axes=axes,
            content=_feature_content_bytes(payload),
        )
    if isinstance(payload, TargetTable):
        n_cols = len(payload.columns)
        dtypes = {column.block.dtype for column in payload.columns}
        return _inline_entry(
            partition=partition,
            payload_id=payload_id,
            role="targets",
            payload_kind="target_table",
            representation_id=payload.representation_id,
            dtype=next(iter(dtypes)) if len(dtypes) == 1 else "mixed",
            shape=[payload.columns[0].block.n_rows if payload.columns else 0, n_cols],
            axes=["sample", "target"] if n_cols > 1 else ["sample"],
            content=_target_content_bytes(payload),
        )
    if isinstance(payload, MetadataTable):
        return _inline_entry(
            partition=partition,
            payload_id=payload_id,
            role="metadata",
            payload_kind="metadata_table",
            representation_id=repr_ids.SAMPLE_METADATA,
            dtype="mixed",
            shape=[int(len(payload.frame)), int(len(payload.frame.columns))],
            axes=["sample", "field"],
            content=_metadata_content_bytes(payload.frame),
        )
    if isinstance(payload, WeightsVector):
        return _inline_entry(
            partition=partition,
            payload_id=payload_id,
            role="weights",
            payload_kind="weights",
            representation_id=None,
            dtype=payload.matrix.dtype,
            shape=[payload.matrix.n_rows],
            axes=["sample"],
            content=_weights_content_bytes(payload),
        )
    if isinstance(payload, SpectralRecordSet):
        content = canonical_json(_json_safe(payload.records)).encode("utf-8")
        return _inline_entry(
            partition=partition,
            payload_id=payload_id,
            role="features",
            payload_kind="spectral_record_set",
            representation_id=None,
            dtype="records",
            shape=[len(payload.records)],
            axes=["sample"],
            content=content,
        )
    if isinstance(payload, NdTensor):
        return _uri_entry(
            partition=partition,
            payload_id=payload_id,
            role="features",
            payload_kind="nd_tensor",
            representation_id=payload.representation_id,
            uri=payload.uri,
            dtype=payload.dtype,
            shape=payload.shape,
            axes=payload.axis_names,
        )
    if isinstance(payload, SequenceBlock):
        return _uri_entry(
            partition=partition,
            payload_id=payload_id,
            role="features",
            payload_kind="sequence_block",
            representation_id=payload.representation_id,
            uri=payload.uri,
        )
    if isinstance(payload, GenotypeMatrix):
        return _uri_entry(
            partition=partition,
            payload_id=payload_id,
            role="features",
            payload_kind="genotype_matrix",
            representation_id=payload.representation_id,
            uri=payload.uri,
            shape=[payload.n_samples, payload.n_variants],
            axes=["sample", "variant"],
        )
    if isinstance(payload, MaskBlock):
        return _uri_entry(
            partition=partition,
            payload_id=payload_id,
            role="mask",
            payload_kind="mask_block",
            representation_id=payload.representation_id,
            uri=payload.uri,
            dtype=payload.dtype,
            shape=payload.shape,
        )
    return _uri_entry(partition=partition, payload_id=payload_id, role="features", payload_kind="uri_backed", representation_id=None, uri=payload)


def _compute_row_position_fallback(assembled: AssembledDataset) -> RowPositionFallback:
    partitions = list(assembled.blocks)
    sample_id = assembled.identity.get("sample_id") if isinstance(assembled.identity, dict) else None
    use_sample_id = isinstance(sample_id, str) and all(
        block.metadata is not None and sample_id in block.metadata.columns and len(block.metadata[sample_id]) == block.n_samples for block in assembled.blocks.values()
    )
    if use_sample_id:
        used = False
        reason = f"stable sample-id key '{sample_id}' is aligned in every partition"
    elif isinstance(sample_id, str):
        used = True
        reason = f"sample-id key '{sample_id}' is absent or not aligned in every partition; sample identity falls back to row position"
    else:
        used = True
        reason = "no stable sample-id key declared; sample identity falls back to row position"
    fingerprint = _sha256_json({"used": used, "reason": reason, "partitions": partitions})
    return RowPositionFallback(used=used, reason=reason, partitions=partitions, fingerprint=fingerprint)


# --------------------------------------------------------------------------- #
# Full-wire encoding / strict decoding                                         #
# --------------------------------------------------------------------------- #
def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise NumericContractError(f"{label} must be a string")
    return value


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise NumericContractError(f"{label} must be a nonnegative integer")
    return value


def _bad(message: str) -> Any:
    raise NumericContractError(message)


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise NumericContractError(f"{label} must be a list of strings")
    return list(value)


def _partition_order(order: Any, partitions: Any) -> list[str]:
    """``partition_order`` must be an exact, unique cover of the partition keys."""
    if not isinstance(partitions, dict) or not isinstance(order, list) or not all(isinstance(n, str) for n in order):
        raise NumericContractError("partition_order must list the partition names")
    if len(set(order)) != len(order) or set(order) != set(partitions):
        raise NumericContractError("partition_order must contain each partition exactly once")
    return list(order)


def _audits_wire(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The owner audits list, verbatim. It is provenance, never numeric authority."""
    try:
        wire = json.loads(json.dumps(audits, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise NumericContractError(f"Owner audits must be JSON objects without nonfinite numbers: {error}") from error
    if not isinstance(wire, list) or not all(isinstance(entry, dict) for entry in wire):
        raise NumericContractError("Owner audits must be a list of objects")
    return wire


def _audits_from_wire(value: Any) -> list[dict[str, Any]]:
    """Restore the supplied owner audits list (an empty list is valid; omission/null refuses)."""
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise NumericContractError("audits must be a list of objects")
    return json.loads(json.dumps(value))


def _identity_wire(identity: dict[str, object]) -> dict[str, Any]:
    wire = {key: identity.get(key) for key in _IDENTITY_KEYS}
    wire["source_ids"] = list(identity.get("source_ids") or [])  # type: ignore[call-overload]
    return wire


def _identity_from_wire(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(_IDENTITY_KEYS):
        raise NumericContractError("Identity provenance needs exactly source_ids, sample_id, observation_id, repetition_id, group_id")
    for key in _IDENTITY_KEYS[1:]:
        if value[key] is not None and not isinstance(value[key], str):
            raise NumericContractError(f"Identity {key} must be a string or null")
    return {"source_ids": _string_list(value["source_ids"], "identity source_ids"), **{key: value[key] for key in _IDENTITY_KEYS[1:]}}


def _aggregate_from_wire(value: Any) -> AggregateSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"by", "method", "exclude_outliers", "outlier_threshold"}:
        raise NumericContractError("Aggregate is malformed")
    if value["by"] is not None and not isinstance(value["by"], str):
        raise NumericContractError("Aggregate by must be a string or null")
    if type(value["exclude_outliers"]) is not bool:
        raise NumericContractError("Aggregate outlier fields are malformed")
    try:
        threshold = outlier_threshold_number(value["outlier_threshold"])
    except SpecError as error:
        raise NumericContractError("Aggregate outlier fields are malformed") from error
    try:
        method = AggregateMethod(value["method"])
    except ValueError as error:
        raise NumericContractError("Unknown aggregate method") from error
    return AggregateSpec(by=value["by"], method=method, exclude_outliers=value["exclude_outliers"], outlier_threshold=threshold)


def _folds_from_wire(value: Any) -> list[tuple[list[int], list[int]]]:
    if not isinstance(value, list):
        raise NumericContractError("folds must be a list")
    folds: list[tuple[list[int], list[int]]] = []
    for fold in value:
        if not isinstance(fold, list) or len(fold) != 2 or any(not isinstance(side, list) or any(type(i) is not int for i in side) for side in fold):
            raise NumericContractError("Each fold must be [train indices, validation indices]")
        folds.append((list(fold[0]), list(fold[1])))
    return folds


def _fold_provenance_from_wire(value: Any) -> list[dict[str, list[str]]]:
    if not isinstance(value, list):
        raise NumericContractError("fold_provenance must be a list")
    out: list[dict[str, list[str]]] = []
    for fold in value:
        if not isinstance(fold, dict) or set(fold) != {"train_observation_ids", "validation_observation_ids"}:
            raise NumericContractError("fold_provenance entry is malformed")
        out.append({key: _string_list(fold[key], key) for key in ("train_observation_ids", "validation_observation_ids")})
    return out


def _axes_wire(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for axis in axes:
        coordinates = axis["coordinates"]
        if coordinates is None:
            wire: Any = None
        elif all(isinstance(value, float) for value in coordinates):
            wire = {"kind": "numeric", "count": len(coordinates), "data_bytes_hex": np.asarray(coordinates, dtype="<f8").tobytes().hex()}
        else:
            wire = {"kind": "string", "values": list(coordinates)}
        out.append({"name": axis["name"], "unit": axis["unit"], "coordinates": wire})
    return out


def _axes_from_wire(value: Any, n_cols: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise NumericContractError("axes must be a list")
    axes: list[dict[str, Any]] = []
    for axis in value:
        if not isinstance(axis, dict) or set(axis) != {"name", "unit", "coordinates"}:
            raise NumericContractError("Axis needs exactly name, unit, coordinates")
        if not isinstance(axis["name"], str) or (axis["unit"] is not None and not isinstance(axis["unit"], str)):
            raise NumericContractError("Axis name/unit must be strings (unit nullable)")
        raw = axis["coordinates"]
        coordinates: list[Any] | None
        if raw is None:
            coordinates = None
        elif isinstance(raw, dict) and raw.get("kind") == "numeric" and set(raw) == {"kind", "count", "data_bytes_hex"}:
            text = raw["data_bytes_hex"]
            if type(raw["count"]) is not int or not isinstance(text, str) or not re.fullmatch(r"(?:[0-9a-f]{2})*", text) or len(text) != raw["count"] * 16:
                raise NumericContractError("Numeric coordinates are malformed")
            coordinates = np.frombuffer(bytes.fromhex(text), dtype="<f8").tolist()
            if not all(np.isfinite(coordinates)):
                raise NumericContractError("Numeric coordinates must be finite")
        elif isinstance(raw, dict) and raw.get("kind") == "string" and set(raw) == {"kind", "values"}:
            coordinates = _string_list(raw["values"], "string coordinates")
        else:
            raise NumericContractError("Unknown coordinate representation")
        if coordinates is not None and len(coordinates) != n_cols:
            raise NumericContractError("Coordinate count differs from the axis extent")
        axes.append({"name": axis["name"], "unit": axis["unit"], "coordinates": coordinates})
    return axes


def _target_columns_wire(table: TargetTable) -> list[dict[str, Any]]:
    return [{"name": column.name, "kind": column.kind, "block": column.block.to_dict()} for column in table.columns]


def _target_columns_from_wire(value: Any) -> list[TargetColumn]:
    if not isinstance(value, list):
        raise NumericContractError("Target columns must be a list")
    columns: list[TargetColumn] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"name", "kind", "block"} or not isinstance(entry["name"], str) or entry["kind"] not in TARGET_KINDS:
            raise NumericContractError("Target column needs exactly name, kind (numeric/classification_integer/classification_string), block")
        block = ExactBlock.from_dict(entry["block"], feature=False)
        if (entry["kind"] == TARGET_KINDS[2]) != (block.dtype == STRING_LABEL) or block.n_cols != 1:
            raise NumericContractError("Target kind disagrees with its block dtype/shape")
        if entry["kind"] == TARGET_KINDS[1]:
            for label, valid in zip(block.to_ir()[:, 0], block.validity[:, 0], strict=True):
                if valid:
                    if label != np.floor(label):
                        raise NumericContractError("Classification requires integer labels")
                    validate_label(int(label), block.dtype)
        columns.append(TargetColumn(entry["name"], entry["kind"], block))
    if len({column.name for column in columns}) != len(columns) or len({column.block.n_rows for column in columns}) > 1:
        raise NumericContractError("Target columns need unique names and a common row count")
    return columns


def _payload_rows(payload: PayloadBlock) -> int | None:
    if isinstance(payload, FeatureMatrix):
        return payload.matrix.n_rows
    if isinstance(payload, TargetTable):
        return payload.columns[0].block.n_rows if payload.columns else None
    if isinstance(payload, WeightsVector):
        return payload.matrix.n_rows
    if isinstance(payload, MetadataTable):
        return int(len(payload.frame))
    return None


def _uri_wire(uri: UriRef) -> dict[str, Any]:
    return {"uri": uri.uri, "dtype": uri.dtype, "shape": list(uri.shape), "axes": list(uri.axes), "content_hash": uri.content_hash, "byte_len": uri.byte_len, "codec": uri.codec}


def _uri_from_wire(value: Any) -> UriRef:
    if not isinstance(value, dict) or set(value) != {"uri", "dtype", "shape", "axes", "content_hash", "byte_len", "codec"}:
        raise NumericContractError("URI reference is malformed")
    if value["codec"] is not None and not isinstance(value["codec"], str):
        raise NumericContractError("URI codec must be a string or null")
    if not isinstance(value["shape"], list) or any(type(n) is not int for n in value["shape"]):
        raise NumericContractError("URI shape must be a list of integers")
    return UriRef(
        uri=_text(value["uri"], "uri"),
        dtype=_text(value["dtype"], "dtype"),
        shape=list(value["shape"]),
        axes=_string_list(value["axes"], "axes"),
        content_hash=_text(value["content_hash"], "content_hash"),
        byte_len=_count(value["byte_len"], "byte_len"),
        codec=value["codec"],
    )


def _payload_wire(payload: PayloadBlock) -> dict[str, Any]:
    if isinstance(payload, FeatureMatrix):
        return {
            "kind": "feature_matrix",
            "value": {
                "matrix": payload.matrix.to_dict(),
                "headers": list(payload.headers),
                "feature_names": None if payload.feature_names is None else list(payload.feature_names),
                "header_unit": payload.header_unit,
                "signal_type": payload.signal_type,
                "source_id": payload.source_id,
                "representation_id": payload.representation_id,
                "axes": _axes_wire(payload.axes),
                "processings": [{"name": name, "matrix": block.to_dict()} for name, block in payload.processings],
            },
        }
    if isinstance(payload, TargetTable):
        return {"kind": "target_table", "value": {"columns": _target_columns_wire(payload), "categorical": _json_safe(payload.categorical), "representation_id": payload.representation_id}}
    if isinstance(payload, WeightsVector):
        return {"kind": "weights", "value": {"matrix": payload.matrix.to_dict(), "header": payload.header}}
    if isinstance(payload, MetadataTable):
        return {"kind": "metadata_table", "value": _metadata_body(payload.frame)}
    if isinstance(payload, SpectralRecordSet):
        return {"kind": "spectral_record_set", "value": {"records": _json_safe(payload.records)}}
    if isinstance(payload, NdTensor):
        return {
            "kind": "nd_tensor",
            "value": {
                "representation_id": payload.representation_id,
                "dtype": payload.dtype,
                "shape": list(payload.shape),
                "axis_names": list(payload.axis_names),
                "observation_ids": list(payload.observation_ids),
                "uri": _uri_wire(payload.uri),
            },
        }
    if isinstance(payload, SequenceBlock):
        return {
            "kind": "sequence_block",
            "value": {"representation_id": payload.representation_id, "dtype": payload.dtype, "n_channels": payload.n_channels, "lengths": list(payload.lengths), "uri": _uri_wire(payload.uri)},
        }
    if isinstance(payload, GenotypeMatrix):
        return {
            "kind": "genotype_matrix",
            "value": {
                "representation_id": payload.representation_id,
                "n_variants": payload.n_variants,
                "n_samples": payload.n_samples,
                "encoding": payload.encoding,
                "uri": _uri_wire(payload.uri),
            },
        }
    if isinstance(payload, MaskBlock):
        return {"kind": "mask_block", "value": {"representation_id": payload.representation_id, "dtype": payload.dtype, "shape": list(payload.shape), "uri": _uri_wire(payload.uri)}}
    return {"kind": "uri_backed", "value": _uri_wire(payload)}


def _frame_from_body(body: Any) -> pd.DataFrame:
    if not isinstance(body, dict) or set(body) != {"n_rows", "columns"} or not isinstance(body["columns"], list):
        raise NumericContractError("Metadata body is malformed")
    n_rows = _count(body["n_rows"], "metadata n_rows")
    data: dict[str, Any] = {}
    for column in body["columns"]:
        if not isinstance(column, dict) or set(column) != {"name", "dtype", "values"} or not isinstance(column["name"], str) or not isinstance(column["values"], list) or len(column["values"]) != n_rows:
            raise NumericContractError("Metadata column is malformed")
        series = _series_from_wire(column["values"])
        series = pd.to_datetime(series) if column["dtype"] == "datetime" else series
        if _series_dtype_label(series) != column["dtype"]:
            raise NumericContractError(f"Metadata column {column['name']!r} dtype label disagrees with its values")
        data[column["name"]] = series
    return pd.DataFrame(data) if data else pd.DataFrame(index=range(n_rows))


def _payload_from_wire(wire: Any) -> PayloadBlock:
    if not isinstance(wire, dict) or set(wire) != {"kind", "value"}:
        raise NumericContractError("Payload needs exactly kind and value")
    kind, value = wire["kind"], wire["value"]
    if kind == "feature_matrix":
        keys = {"matrix", "headers", "feature_names", "header_unit", "signal_type", "source_id", "representation_id", "axes", "processings"}
        if not isinstance(value, dict) or set(value) != keys or not isinstance(value["processings"], list):
            raise NumericContractError("Feature payload has missing or unknown fields")
        matrix = ExactBlock.from_dict(value["matrix"], feature=True)
        headers = _string_list(value["headers"], "headers")
        if len(headers) != matrix.n_cols or matrix.dtype == STRING_LABEL:
            raise NumericContractError("Feature headers disagree with the matrix width")
        names = value["feature_names"]
        if names is not None:
            names = _string_list(names, "feature_names")
            if len(names) != matrix.n_cols:
                raise NumericContractError("feature_names disagree with the matrix width")
        processings: list[tuple[str, ExactBlock]] = []
        for entry in value["processings"]:
            if not isinstance(entry, dict) or set(entry) != {"name", "matrix"} or not isinstance(entry["name"], str):
                raise NumericContractError("Processing entry is malformed")
            block = ExactBlock.from_dict(entry["matrix"], feature=True)
            if (block.n_rows, block.n_cols) != (matrix.n_rows, matrix.n_cols) or block.dtype == STRING_LABEL:
                raise NumericContractError("Processing shape differs from the feature matrix")
            _check_child_precision(matrix.precision, block.precision, entry["name"])
            processings.append((entry["name"], block))
        if value["signal_type"] is not None and not isinstance(value["signal_type"], str):
            raise NumericContractError("signal_type must be a string or null")
        return FeatureMatrix(
            matrix=matrix,
            headers=headers,
            header_unit=_text(value["header_unit"], "header_unit"),
            signal_type=value["signal_type"],
            source_id=_text(value["source_id"], "source_id"),
            representation_id=_text(value["representation_id"], "representation_id"),
            feature_names=names,
            axes=_axes_from_wire(value["axes"], matrix.n_cols),
            processings=processings,
        )
    if kind == "target_table":
        if not isinstance(value, dict) or set(value) != {"columns", "categorical", "representation_id"} or not isinstance(value["categorical"], dict):
            raise NumericContractError("Target payload has missing or unknown fields")
        if value["representation_id"] is not None and not isinstance(value["representation_id"], str):
            raise NumericContractError("representation_id must be a string or null")
        return TargetTable(columns=_target_columns_from_wire(value["columns"]), categorical=value["categorical"], representation_id=value["representation_id"])
    if kind == "weights":
        if not isinstance(value, dict) or set(value) != {"matrix", "header"} or (value["header"] is not None and not isinstance(value["header"], str)):
            raise NumericContractError("Weights payload has missing or unknown fields")
        matrix = ExactBlock.from_dict(value["matrix"], feature=False)
        if matrix.n_cols != 1 or matrix.dtype == STRING_LABEL:
            raise NumericContractError("Weights need one numeric column")
        return WeightsVector(matrix=matrix, header=value["header"])
    if kind == "metadata_table":
        return MetadataTable(frame=_frame_from_body(value))
    if kind == "spectral_record_set":
        if not isinstance(value, dict) or set(value) != {"records"} or not isinstance(value["records"], list):
            raise NumericContractError("Record-set payload is malformed")
        return SpectralRecordSet(records=value["records"])
    if kind == "uri_backed":
        return _uri_from_wire(value)
    uri_kinds: dict[str, set[str]] = {
        "nd_tensor": {"representation_id", "dtype", "shape", "axis_names", "observation_ids", "uri"},
        "sequence_block": {"representation_id", "dtype", "n_channels", "lengths", "uri"},
        "genotype_matrix": {"representation_id", "n_variants", "n_samples", "encoding", "uri"},
        "mask_block": {"representation_id", "dtype", "shape", "uri"},
    }
    if kind not in uri_kinds or not isinstance(value, dict) or set(value) != uri_kinds[kind]:
        raise NumericContractError(f"Unsupported or malformed payload kind {kind!r}")
    uri = _uri_from_wire(value["uri"])
    rep = _text(value["representation_id"], "representation_id")
    if kind == "nd_tensor":
        return NdTensor(rep, _text(value["dtype"], "dtype"), list(value["shape"]), _string_list(value["axis_names"], "axis_names"), _string_list(value["observation_ids"], "observation_ids"), uri)
    if kind == "sequence_block":
        return SequenceBlock(rep, _text(value["dtype"], "dtype"), _count(value["n_channels"], "n_channels"), list(value["lengths"]), uri)
    if kind == "genotype_matrix":
        return GenotypeMatrix(rep, _count(value["n_variants"], "n_variants"), _count(value["n_samples"], "n_samples"), _text(value["encoding"], "encoding"), uri)
    return MaskBlock(rep, _text(value["dtype"], "dtype"), list(value["shape"]), uri)


# --------------------------------------------------------------------------- #
# AssembledDataset v3 full wire                                                #
# --------------------------------------------------------------------------- #
_ASSEMBLED_KEYS = frozenset(
    {"partition_order", "assembled_schema_version", "name", "task_type", "signal_type", "n_sources", "blocks", "folds", "fold_provenance", "identity", "repetition", "aggregate", "warnings", "audits"}
)
_BLOCK_WIRE_KEYS = frozenset(
    {"n_samples", "source_ids", "x", "feature_headers", "feature_names", "feature_axes", "representation_ids", "header_units", "signal_types", "y", "y_headers", "y_categorical", "metadata", "weights", "weights_header", "processings"}
)


def assembled_to_full_dict(assembled: AssembledDataset, *, features: Any = None, targets: Any = None, weights: Any = None, consumer: str | None = None) -> dict[str, Any]:
    """Exact assembled v3 full export (dtype/bytes/mask blocks, required ``partition_order``)."""
    package = DatasetPackage.from_assembled(assembled, features=features, targets=targets, weights=weights, consumer=consumer)
    normalized = package.to_assembled()
    blocks: dict[str, Any] = {}
    for name, block in normalized.blocks.items():
        payloads = [payload for _, payload in package.partitions[name].payloads]
        features_ = [p for p in payloads if isinstance(p, FeatureMatrix)]
        table = next((p for p in payloads if isinstance(p, TargetTable)), None)
        weight = next((p for p in payloads if isinstance(p, WeightsVector)), None)
        blocks[name] = {
            "n_samples": block.n_samples,
            "source_ids": list(block.source_ids),
            "x": [f.matrix.to_dict() for f in features_],
            "feature_headers": [list(h) for h in block.feature_headers],
            "feature_names": [None if f.feature_names is None else list(f.feature_names) for f in features_],
            "feature_axes": [_axes_wire(f.axes) for f in features_],
            "representation_ids": [f.representation_id for f in features_],
            "header_units": list(block.header_units),
            "signal_types": list(block.signal_types),
            "y": None if table is None else {"columns": _target_columns_wire(table)},
            "y_headers": list(block.y_headers),
            "y_categorical": _json_safe(block.y_categorical),
            "metadata": None if block.metadata is None else _frame_full(block.metadata),
            "weights": None if weight is None else weight.matrix.to_dict(),
            "weights_header": block.weights_header,
            "processings": [[{"name": n, "matrix": b.to_dict()} for n, b in f.processings] for f in features_],
        }
    aggregate = None
    if normalized.aggregate is not None:
        aggregate = {
            "by": normalized.aggregate.by,
            "method": normalized.aggregate.method.value,
            "exclude_outliers": normalized.aggregate.exclude_outliers,
            "outlier_threshold": normalized.aggregate.outlier_threshold,
        }
    return {
        "partition_order": list(normalized.blocks),
        "assembled_schema_version": ASSEMBLED_DATASET_VERSION,
        "name": normalized.name,
        "task_type": normalized.task_type,
        "signal_type": normalized.signal_type,
        "n_sources": normalized.n_sources,
        "blocks": blocks,
        "folds": [[list(train), list(val)] for train, val in normalized.folds],
        "fold_provenance": [dict(fold) for fold in normalized.fold_provenance],
        "identity": {"provenance": _identity_wire(normalized.identity)},
        "repetition": normalized.repetition,
        "aggregate": aggregate,
        "warnings": list(normalized.warnings),
        "audits": _audits_wire(normalized.audits),
    }


def _frame_full(frame: pd.DataFrame) -> dict[str, Any]:
    return {"n_rows": int(len(frame)), "columns": [{"name": str(name), "values": [_json_scalar(v) for v in frame[name].tolist()]} for name in frame.columns]}


def assembled_from_full_dict(value: Any) -> AssembledDataset:
    """Strict reader of the exact assembled v3 wire; block order comes from ``partition_order``."""
    if not isinstance(value, dict) or set(value) != _ASSEMBLED_KEYS:
        raise NumericContractError("Assembled full wire has missing or unknown fields")
    if type(value["assembled_schema_version"]) is not int or value["assembled_schema_version"] != ASSEMBLED_DATASET_VERSION:
        raise NumericContractError(f"Unsupported assembled version {value['assembled_schema_version']!r}; exactly {ASSEMBLED_DATASET_VERSION} is required")
    order = _partition_order(value["partition_order"], value["blocks"])
    identity = value["identity"]
    if not isinstance(identity, dict) or set(identity) != {"provenance"}:
        raise NumericContractError("Assembled identity wrapper is malformed")
    assembled = AssembledDataset(
        name=_text(value["name"], "name"),
        task_type=_text(value["task_type"], "task_type"),
        signal_type=_text(value["signal_type"], "signal_type"),
        n_sources=_count(value["n_sources"], "n_sources"),
        repetition=None if value["repetition"] is None else _text(value["repetition"], "repetition"),
        aggregate=_aggregate_from_wire(value["aggregate"]),
        warnings=_string_list(value["warnings"], "warnings"),
        audits=_audits_from_wire(value["audits"]),
    )
    assembled.folds = _folds_from_wire(value["folds"])
    assembled.fold_provenance = _fold_provenance_from_wire(value["fold_provenance"])
    assembled.identity = _identity_from_wire(identity["provenance"])
    for name in order:
        wire = value["blocks"][name]
        if not isinstance(wire, dict) or set(wire) != _BLOCK_WIRE_KEYS:
            raise NumericContractError(f"Block {name!r} has missing or unknown fields")
        block = PartitionBlock(n_samples=_count(wire["n_samples"], "n_samples"))
        block.source_ids = _string_list(wire["source_ids"], "source_ids")
        if not isinstance(wire["x"], list) or not isinstance(wire["processings"], list) or len(wire["x"]) != len(block.source_ids) or len(wire["processings"]) != len(wire["x"]):
            raise NumericContractError("Block x/processings/source_ids lengths disagree")
        headers = wire["feature_headers"]
        if (
            not isinstance(headers, list)
            or not isinstance(wire["header_units"], list)
            or not isinstance(wire["signal_types"], list)
            or not all(isinstance(wire[key], list) and len(wire[key]) == len(wire["x"]) for key in ("feature_names", "feature_axes", "representation_ids"))
            or len(headers) != len(wire["x"])
            or len(wire["header_units"]) != len(wire["x"])
            or len(wire["signal_types"]) != len(wire["x"])
        ):
            raise NumericContractError("Block feature metadata lengths disagree with x")
        for index, matrix_wire in enumerate(wire["x"]):
            matrix = ExactBlock.from_dict(matrix_wire, feature=True)
            if matrix.dtype == STRING_LABEL or matrix.n_rows != block.n_samples:
                raise NumericContractError("Feature block is not numeric or its rows disagree with n_samples")
            block.X.append(matrix.to_ir())
            block.x_source_dtypes.append(matrix.precision.source_dtype)
            block.x_precisions.append(matrix.precision)
            block.feature_headers.append(_string_list(headers[index], "feature_headers"))
            if len(block.feature_headers[-1]) != matrix.n_cols:
                raise NumericContractError("feature_headers disagree with the matrix width")
            names = wire["feature_names"][index]
            if names is not None:
                names = _string_list(names, "feature_names")
                if len(names) != matrix.n_cols:
                    raise NumericContractError("feature_names disagree with the matrix width")
            block.feature_names.append(names)
            block.feature_axes.append(_axes_from_wire(wire["feature_axes"][index], matrix.n_cols))
            block.representation_ids.append(_text(wire["representation_ids"][index], "representation_id"))
            block.header_units.append(_text(wire["header_units"][index], "header_unit"))
            signal = wire["signal_types"][index]
            if signal is not None and not isinstance(signal, str):
                raise NumericContractError("signal_type must be a string or null")
            block.signal_types.append(signal)
            procs: list[tuple[str, np.ndarray]] = []
            child_records: list[PrecisionRecord] = []
            if not isinstance(wire["processings"][index], list):
                raise NumericContractError("Processings of a source must be a list")
            for entry in wire["processings"][index]:
                if not isinstance(entry, dict) or set(entry) != {"name", "matrix"} or not isinstance(entry["name"], str):
                    raise NumericContractError("Processing entry is malformed")
                processing = ExactBlock.from_dict(entry["matrix"], feature=True)
                if (processing.n_rows, processing.n_cols) != (matrix.n_rows, matrix.n_cols):
                    raise NumericContractError("A processing must keep its feature source's shape")
                _check_child_precision(matrix.precision, processing.precision, entry["name"])
                procs.append((entry["name"], processing.to_ir()))
                child_records.append(processing.precision)
            block.processings.append(procs)
            block.processing_precisions.append(child_records)
            block.processing_source_dtypes.append([p.source_dtype for p in child_records])
        if wire["y"] is not None:
            if not isinstance(wire["y"], dict) or set(wire["y"]) != {"columns"}:
                raise NumericContractError("Target wire needs exactly columns")
            columns = _target_columns_from_wire(wire["y"]["columns"])
            if [c.name for c in columns] != _string_list(wire["y_headers"], "y_headers"):
                raise NumericContractError("y_headers disagree with target column order")
            if not isinstance(wire["y_categorical"], dict):
                raise NumericContractError("y_categorical must be an object")
            table = TargetTable(columns=columns, categorical=wire["y_categorical"])
            block.y = _target_matrix(table)
            block.y_headers = table.headers
            block.y_categorical = json.loads(json.dumps(wire["y_categorical"]))
            numeric = [c.block.precision.source_dtype for c in columns if c.kind != TARGET_KINDS[2]]
            block.y_source_dtype = numeric[0] if numeric else None
            block.y_precisions = [c.block.precision for c in columns]
        elif wire["y_headers"] != [] or wire["y_categorical"] != {}:
            raise NumericContractError("Absent targets cannot carry headers or codebooks")
        if wire["metadata"] is not None:
            block.metadata = _frame_from_wire_full(wire["metadata"])
        if wire["weights"] is not None:
            weights = ExactBlock.from_dict(wire["weights"], feature=False)
            if weights.n_cols != 1 or weights.dtype == STRING_LABEL:
                raise NumericContractError("Weights need one numeric column")
            block.weights = weights.to_ir()[:, 0]
            block.weights_source_dtype = weights.precision.source_dtype
            block.weights_precision = weights.precision
        if wire["weights_header"] is not None and not isinstance(wire["weights_header"], str):
            raise NumericContractError("weights_header must be a string or null")
        if wire["weights"] is None and wire["weights_header"] is not None:
            raise NumericContractError("weights_header requires weights")
        block.weights_header = wire["weights_header"]
        assembled.blocks[name] = block
    return assembled


def _frame_from_wire_full(body: Any) -> pd.DataFrame:
    if not isinstance(body, dict) or set(body) != {"n_rows", "columns"} or not isinstance(body["columns"], list):
        raise NumericContractError("Metadata wire is malformed")
    n_rows = _count(body["n_rows"], "metadata n_rows")
    data: dict[str, Any] = {}
    for column in body["columns"]:
        if not isinstance(column, dict) or set(column) != {"name", "values"} or not isinstance(column["name"], str) or not isinstance(column["values"], list) or len(column["values"]) != n_rows:
            raise NumericContractError("Metadata column is malformed")
        data[column["name"]] = _series_from_wire(column["values"])
    return pd.DataFrame(data) if data else pd.DataFrame(index=range(n_rows))


__all__ = [
    "DATASET_PACKAGE_VERSION",
    "DatasetPackage",
    "FeatureMatrix",
    "GenotypeMatrix",
    "MaskBlock",
    "MetadataTable",
    "NdTensor",
    "PackagePartition",
    "PayloadBlock",
    "PayloadManifest",
    "PayloadManifestEntry",
    "PayloadStorageKind",
    "RowPositionFallback",
    "SequenceBlock",
    "SpectralRecordSet",
    "TargetColumn",
    "TargetTable",
    "UriRef",
    "WeightsVector",
    "assembled_from_full_dict",
    "assembled_to_full_dict",
    "repr_ids",
]
