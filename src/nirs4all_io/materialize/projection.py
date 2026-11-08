# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Base-cohort feature/target content identity (contract section 6), Python oracle.

``nirs4all.io.executed-projection.v1`` describes the **base materialized
cohort**: two separate SHA-256 digests (features, targets) over byte-identical
shared cohort metadata (ordered sample/observation IDs, partitions, group /
origin / repetition IDs, observation-keyed folds). It is independent of the
package root, of the old whole-dataset tagged-binary64 bytes and of the fixed
X-only ``n4a-matrix-f64-le.v1`` prediction profile, and it does not certify exact
per-fold buffers. The receipt carries no self-hash.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .numeric import NumericContractError, concat_rows, option_string, string, strings, u64
from .package import (
    DatasetPackage,
    FeatureMatrix,
    TargetColumn,
    TargetTable,
    _axes_wire,
    feature_bytes,
    target_bytes,
)

EXECUTED_PROJECTION_PROFILE = "nirs4all.io.executed-projection.v1"
MASK_POLICY = "explicit-mask-positive-zero.v1"
DATASET_CONTENT_PROFILE = "nirs4all.io.dataset-content.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.\-]{1,128}\Z")
_RECEIPT_KEYS = (
    "profile",
    "version",
    "scope",
    "algorithm",
    "feature_content_fingerprint",
    "target_content_fingerprint",
    "selected_source_ids",
    "sample_ids",
    "observation_ids",
    "target_names",
    "target_kinds",
    "features",
    "targets",
    "mask_policy",
    "package_root",
    "dataset_content_identity",
)


@dataclass(frozen=True)
class ExecutedProjectionIdentityV1:
    """The base-cohort receipt: separate digests plus the data they describe (no self-hash)."""

    profile: str
    version: int
    scope: str
    algorithm: str
    feature_content_fingerprint: str
    target_content_fingerprint: str
    selected_source_ids: list[str]
    sample_ids: list[str]
    observation_ids: list[str]
    target_names: list[str]
    target_kinds: list[str]
    features: list[dict[str, Any]]
    targets: list[dict[str, Any]] | None
    mask_policy: str
    package_root: str | None
    dataset_content_identity: Any

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _RECEIPT_KEYS}

    @classmethod
    def from_dict(cls, value: Any) -> ExecutedProjectionIdentityV1:
        """Strict reader: exact fields, profile, version, scope, algorithm and mask policy."""
        if not isinstance(value, dict) or set(value) != set(_RECEIPT_KEYS):
            raise NumericContractError("Executed-projection receipt has missing or unknown fields")
        if (value["profile"], value["scope"], value["algorithm"], value["mask_policy"]) != (EXECUTED_PROJECTION_PROFILE, "base", "sha256", MASK_POLICY) or value["version"] != 1 or type(value["version"]) is not int:
            raise NumericContractError("Unsupported executed-projection profile/version/scope/algorithm/mask policy")
        _check_dataset_identity_shape(value["dataset_content_identity"])
        return cls(**{key: value[key] for key in _RECEIPT_KEYS})

    def validate(self, package: DatasetPackage) -> None:
        """Refuse a receipt that disagrees with the package buffers/metadata."""
        actual = executed_projection(package, self.selected_source_ids).identity
        if actual != self:
            raise NumericContractError("Executed projection disagrees with package buffers/metadata")


@dataclass(frozen=True)
class ExecutedProjection:
    """The receipt together with the exact feature/target preimages that were hashed."""

    identity: ExecutedProjectionIdentityV1
    feature_preimage: bytes
    target_preimage: bytes


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise NumericContractError(f"Invalid primary identity {value!r}")
    return value


def _identity_text(cell: Any) -> str:
    """Render an identity cell. Strings are kept; other scalars use the JSON ``to_string`` form.

    Only representations with a byte-identical Rust rendering are supported: integers (decimal), booleans
    (CPython ``True``/``False``) and finite floats. A float is its Python ``repr`` (the owner's ``str(float)``: shortest
    round trip, ``5.0``, ``1e-07``, ``-0.0``), which then meets the same identifier grammar as every other
    identity: ``1e+16`` fails it because of the ``+``, not because of any numeric cap. Other types refuse.
    """
    if isinstance(cell, np.generic):
        cell = cell.item()
    if isinstance(cell, str):
        return cell
    if isinstance(cell, bool):
        return "True" if cell else "False"  # CPython spelling: the owner renders scalar cells with Python's ``str``
    if isinstance(cell, int):
        return str(cell)
    if isinstance(cell, float):
        if not np.isfinite(cell):
            raise NumericContractError("Missing projection identity")
        return repr(cell)
    raise NumericContractError(f"Unsupported identity representation {type(cell).__name__}")


def _head(side: str, shared: bytes) -> bytes:
    return string(EXECUTED_PROJECTION_PROFILE) + (1).to_bytes(4, "little") + string(side) + shared


def _shared(package: DatasetPackage, assembled: Any) -> tuple[bytes, list[str], list[str]]:
    sample_column = package.identity.get("sample_id")
    if not isinstance(sample_column, str):
        raise NumericContractError("Projection requires stable sample identity")
    observation_column = package.identity.get("observation_id") or sample_column
    group_column, repetition_column = package.identity.get("group_id"), package.identity.get("repetition_id")
    rows: list[bytes] = []
    sample_ids: list[str] = []
    observation_ids: list[str] = []
    row_groups: list[str | None] = []
    for partition, block in assembled.blocks.items():
        frame = block.metadata
        if frame is None:
            raise NumericContractError("Projection identity metadata absent")

        def get(name: Any, row: int, frame: pd.DataFrame = frame) -> str:
            if not isinstance(name, str) or name not in frame.columns or row >= len(frame):
                raise NumericContractError("Projection identity column missing/misaligned")
            cell = frame[name].iloc[row]
            if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
                raise NumericContractError("Missing projection identity")
            return _identifier(_identity_text(cell))

        for row in range(block.n_samples):
            sample, observation = get(sample_column, row), get(observation_column, row)
            group = get(group_column, row) if group_column else None
            origin = get("origin_id", row) if "origin_id" in frame.columns else None
            repetition = get(repetition_column, row) if repetition_column else None
            rows.append(string(sample) + string(observation) + string(partition) + option_string(group) + option_string(origin) + option_string(repetition))
            sample_ids.append(sample)
            observation_ids.append(observation)
            row_groups.append(group)
    if len(set(observation_ids)) != len(observation_ids):
        raise NumericContractError("Duplicate observation identity")
    if repetition_column is None and len(set(sample_ids)) != len(sample_ids):
        raise NumericContractError("Repeated samples require repetition semantics")
    shared = b"\x00" + u64(len(rows)) + b"".join(rows)
    known = set(observation_ids)
    shared += b"\x01" if package.fold_provenance else b"\x00"
    for fold in package.fold_provenance:
        train, validation = fold["train_observation_ids"], fold["validation_observation_ids"]
        if len(set(train)) != len(train) or len(set(validation)) != len(validation) or set(train) & set(validation) or not set(train) <= known or not set(validation) <= known:
            raise NumericContractError("Invalid observation-keyed fold membership")
        train_groups = {g for i, g in zip(observation_ids, row_groups, strict=True) if i in set(train) and g is not None}
        if any(i in set(validation) and g is not None and g in train_groups for i, g in zip(observation_ids, row_groups, strict=True)):
            raise NumericContractError("Fold group leakage")
    if package.fold_provenance:
        shared += u64(len(package.fold_provenance)) + b"".join(strings(f["train_observation_ids"]) + strings(f["validation_observation_ids"]) for f in package.fold_provenance)
    audit = next((a for a in package.audits if "raw_fold_assignments" in a), None)
    shared += b"\x01" if audit is not None else b"\x00"
    if audit is not None:
        input_ids, raw = audit.get("input_sample_ids"), audit.get("raw_fold_assignments")
        if not isinstance(input_ids, list) or not isinstance(raw, list) or len(input_ids) != len(raw):
            raise NumericContractError("Raw fold assignment length mismatch")
        shared += u64(len(rows))
        for sample in sample_ids:
            if sample not in input_ids:
                raise NumericContractError("Raw fold identity mismatch")
            shared += option_string(raw[input_ids.index(sample)])
    return shared, sample_ids, observation_ids


def _whole_feature(package: DatasetPackage, source_id: str) -> FeatureMatrix:
    parts: list[FeatureMatrix] = []
    for partition in package.partitions.values():
        found = next((p for _, p in partition.payloads if isinstance(p, FeatureMatrix) and p.source_id == source_id), None)
        if found is None:
            raise NumericContractError("Selected feature source missing from partition")
        found.matrix.precision.validate(feature=True)
        if parts and (
            found.headers != parts[0].headers
            or found.feature_names != parts[0].feature_names
            or found.axes != parts[0].axes
            or found.representation_id != parts[0].representation_id
        ):
            raise NumericContractError("Feature descriptor differs across partitions")
        parts.append(found)
    if not parts:
        raise NumericContractError("Projection has no feature rows")
    first = parts[0]
    return FeatureMatrix(
        matrix=concat_rows([p.matrix for p in parts]),
        headers=first.headers,
        header_unit=first.header_unit,
        signal_type=first.signal_type,
        source_id=source_id,
        representation_id=first.representation_id,
        feature_names=first.feature_names,
        axes=first.axes,
    )


def _whole_targets(package: DatasetPackage) -> TargetTable | None:
    tables: list[TargetTable | None] = [next((p for _, p in partition.payloads if isinstance(p, TargetTable)), None) for partition in package.partitions.values()]
    present = [t for t in tables if t is not None]
    if not present:
        return None
    if len(present) != len(tables):
        raise NumericContractError("Targets absent in one partition")
    first = present[0]
    if any(t.headers != first.headers or [c.kind for c in t.columns] != [c.kind for c in first.columns] or t.categorical != first.categorical for t in present):
        raise NumericContractError("Target descriptor differs across partitions")
    columns = []
    for index, column in enumerate(first.columns):
        block = concat_rows([t.columns[index].block for t in present])
        block.precision.validate(feature=False)
        columns.append(TargetColumn(column.name, column.kind, block))
    return TargetTable(columns=columns, categorical=first.categorical, representation_id=first.representation_id)


def _verified_dataset_identity(package: DatasetPackage) -> dict[str, str] | None:
    from ..public_package import recover_source_evidence, verified_dataset_identity

    # Process-local evidence, or a serialized record that rebuilds the package exactly; else declined (None).
    evidenced = recover_source_evidence(package)
    return None if evidenced.source_evidence is None else verified_dataset_identity(evidenced)


def _check_dataset_identity_shape(value: Any) -> None:
    """Independent shape check of the optional link (profile / algorithm / fingerprint)."""
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"profile", "algorithm", "fingerprint"}:
        raise NumericContractError("dataset_content_identity needs exactly profile, algorithm, fingerprint")
    if value["profile"] != DATASET_CONTENT_PROFILE or value["algorithm"] != "sha256":
        raise NumericContractError("Unsupported dataset-content identity profile/algorithm")
    fingerprint = value["fingerprint"]
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise NumericContractError("dataset-content fingerprint must be 64 lowercase hex digits")


def executed_projection(package: DatasetPackage, selected: list[str]) -> ExecutedProjection:
    """Exact base-cohort preimages and receipt for explicitly selected feature sources.

    ``dataset_content_identity`` is the unchanged whole-dataset identity, labelled, and
    only when it is genuinely computed from the original public record that built the
    package (process-local ``source_evidence``, re-verified by rebuilding the package
    and comparing every byte); otherwise ``None``. A caller hash, an audit echo or a
    receipt field is never authority. Bounded operation: the current provider limit is
    one selected source (order is explicit). Targets absent from every partition leave
    the target digest slot's ``None`` option; a present target list is ``Some(columns)``.
    """
    if not selected or len(set(selected)) != len(selected):
        raise NumericContractError("Explicit unique feature source selection required")
    if len(selected) != 1:
        raise NumericContractError("The bounded base projection requires exactly one selected feature source (no fusion)")
    assembled = package.to_assembled()
    shared, sample_ids, observation_ids = _shared(package, assembled)
    feature_preimage = _head("features", shared) + u64(len(selected))
    descriptors: list[dict[str, Any]] = []
    for source in selected:
        whole = _whole_feature(package, source)
        feature_preimage += feature_bytes(whole)
        descriptors.append(
            {
                "source_id": source,
                "representation_id": whole.representation_id,
                "precision": whole.matrix.precision.to_dict(),
                "shape": [whole.matrix.n_rows, whole.matrix.n_cols],
                "feature_names": None if whole.feature_names is None else list(whole.feature_names),
                "axes": _axes_wire(whole.axes),
                "storage_dtype": whole.matrix.dtype,
                "delivered_dtype": "float64",
            }
        )
    table = _whole_targets(package)
    target_preimage = _head("targets", shared) + (b"\x01" if table is not None else b"\x00")
    target_names: list[str] = []
    target_kinds: list[str] = []
    target_descriptors: list[dict[str, Any]] | None = None
    if table is not None:
        target_preimage += u64(len(table.columns)) + b"".join(target_bytes(c) for c in table.columns)
        target_names = table.headers
        target_kinds = [c.kind for c in table.columns]
        target_descriptors = [{"name": c.name, "kind": c.kind, "precision": c.block.precision.to_dict(), "shape": [c.block.n_rows, 1]} for c in table.columns]
    identity = ExecutedProjectionIdentityV1(
        profile=EXECUTED_PROJECTION_PROFILE,
        version=1,
        scope="base",
        algorithm="sha256",
        feature_content_fingerprint=hashlib.sha256(feature_preimage).hexdigest(),
        target_content_fingerprint=hashlib.sha256(target_preimage).hexdigest(),
        selected_source_ids=list(selected),
        sample_ids=sample_ids,
        observation_ids=observation_ids,
        target_names=target_names,
        target_kinds=target_kinds,
        features=descriptors,
        targets=target_descriptors,
        mask_policy=MASK_POLICY,
        package_root=package.manifest().root,
        dataset_content_identity=_verified_dataset_identity(package),
    )
    return ExecutedProjection(identity, feature_preimage, target_preimage)


__all__ = ["EXECUTED_PROJECTION_PROFILE", "MASK_POLICY", "ExecutedProjection", "ExecutedProjectionIdentityV1", "executed_projection"]
