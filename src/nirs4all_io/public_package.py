# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Public ``Dataset`` -> exact F64 ``DatasetPackage`` factories (Python oracle).

Independent counterparts of the owner's ``matrix_dataset_package[_with_storage]``,
``masked_matrix_dataset_package[_with_storage]`` and
``dense_dataset_package[_with_storage]``. They consume the existing validated public
``Dataset`` / ``TensorSource`` record and the same fallible numeric owner selection as
the tabular path (source-record dtype counts as a declaration, F64 by default). The
actual sample/observation IDs, partitions, origins, groups, repetitions, declared fixed
folds, original feature names, raw axes/units/coordinates and target masks are carried;
primary IDs are never sanitized, folds are never generated, no label is invented and no
encoder exists. Unsupported shapes, dtypes and source selections refuse explicitly.

The optional whole-dataset identity link of the receipt is computed only from the
original record this module saw (process-local ``source_evidence``) and re-verified by
rebuilding the package; it is never read back from an audit or a caller-supplied hash.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .dataset_facade import load_dataset_document
from .materialize.assemble import AssembledDataset, PartitionBlock
from .materialize.numeric import NumericContractError, NumericStorageOptions, integers_to_f64, to_ir_f64
from .materialize.package import DatasetPackage, FeatureMatrix, TargetTable
from .multimodal import MultimodalDataset, TensorSource
from .public_content import ContentNotRepresentableError, canonical_content_bytes, labelled_dataset_content_identity
from .public_dataset import Dataset
from .public_dataset import dataset as public_dataset

_PARTITIONS = ("train", "test", "predict")


def _render_identity(value: Any) -> str:
    """Render a public group/unit ID like the owner's JSON ``to_string``; unsupported forms refuse."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and "e" not in repr(value) and (value == 0 or 1e-4 <= abs(value) < 1e15):
        return repr(value)
    raise NumericContractError(f"Unsupported public group representation {value!r}: its JSON rendering is not byte-identical across owners")


def _axes(source: TensorSource) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    for axis in (source.axes or ())[1:]:
        coordinates = (source.axis_coordinates or {}).get(axis)
        wire: list[Any] | None = None
        if coordinates is not None:
            wire = [str(c) for c in coordinates] if any(isinstance(c, str) for c in coordinates) else [float(c) for c in coordinates]
        axes.append({"name": axis, "unit": (source.axis_units or {}).get(axis), "coordinates": wire})
    return axes


def _array_wire(array: np.ndarray) -> dict[str, Any]:
    return {"dtype": "bool", "shape": list(array.shape), "values": array.tolist()}


def _source_schema(source: TensorSource, source_id: str) -> dict[str, Any]:
    """The cohort-independent source schema (sample dimension ``None``), like ``TensorSource.schema_descriptor``."""
    return {
        "name": source_id,
        "representation_id": source.representation_id,
        "axes": list(source.axes or ()),
        "shape": [None, *source.values.shape[1:]],
        "dtype": str(source.values.dtype),
        "feature_names": None if source.feature_names is None else list(source.feature_names),
        "axis_units": {axis: unit for axis, unit in (source.axis_units or {}).items() if unit is not None},
        "axis_coordinates": {axis: list(coordinates) for axis, coordinates in (source.axis_coordinates or {}).items()},
    }


def _origin(value: Any) -> Any:
    """The isolated original input: a caller ``Dataset`` is kept as is; any other document
    (mapping, JSON/YAML text or path, read through the existing ``load_dataset_document``)
    is deep-copied BEFORE normalization so provenance and buffers come from one input."""
    if isinstance(value, Dataset):
        return value
    document = load_dataset_document(value)
    return public_dataset(document) if isinstance(document, MultimodalDataset) else copy.deepcopy(document)


def _provenance(origin: Any, ds: Dataset) -> dict[str, Any]:
    """The actual supplied record for ``owner_dataset_record`` (never numeric authority).

    A decoded JSON-compatible document keeps its supplied scalar values and optional
    fields; an already validated ``Dataset`` object (or a non-JSON mapping) exposes only
    its validated ``to_dict``, so no prior bytes are invented.
    """
    if isinstance(origin, Dataset):
        return ds.to_dict()
    try:
        json.dumps(origin, allow_nan=False)
    except (TypeError, ValueError):
        return ds.to_dict()
    return copy.deepcopy(dict(origin))


def _masked_cells(values: Any, flags: Any, shape: list[int], dtype: str) -> Any:
    """Replace every literally false mask cell of a target array by its canonical placeholder.

    Recursion follows the declared SHAPE (never ``isinstance(list)``): a false cell admits only ``null``, a bool
    or a finite number (anything else refuses) and becomes ``false`` for a bool array, ``0.0`` for a float array
    and ``0`` otherwise. Observed cells are returned unchanged and validate under their declared dtype later.
    """
    if not shape:
        if flags is not False:
            return values
        if not (values is None or type(values) is bool or (type(values) in (int, float) and math.isfinite(values))):
            raise ValueError("Masked target storage requires null or finite numeric values")
        return False if dtype == "bool" else 0.0 if dtype.startswith("float") else 0
    if not isinstance(values, list) or not isinstance(flags, list) or len(values) != shape[0] or len(flags) != shape[0]:
        raise ValueError("Target mask mismatch")
    return [_masked_cells(value, flag, shape[1:], dtype) for value, flag in zip(values, flags, strict=True)]


def _masked_materialization(document: Any) -> Any:
    """The explicit masked route's working copy: false target cells carry canonical placeholders.

    A withheld label is not a value, so its stored placeholder is discarded BEFORE any dtype, label or
    declared-array gate runs; observed cells stay strict. Only the working copy is rewritten: the ORIGINAL
    record keeps providing the audit, ``source_evidence`` and the optional whole-dataset identity link.
    """
    working = copy.deepcopy(document)
    if isinstance(working, Dataset) or not isinstance(working, dict):
        return working
    raw = working["dataset"] if isinstance(working.get("dataset"), dict) else working
    y, mask = raw.get("y"), raw.get("target_mask")
    if y is None or mask is None:
        return working
    if not isinstance(y, dict) or not isinstance(mask, dict) or y.get("shape") != mask.get("shape") or mask.get("dtype") != "bool":
        raise ValueError("Target mask mismatch")
    shape, dtype = y["shape"], y.get("dtype")
    if not isinstance(shape, list) or not all(type(size) is int and size >= 0 for size in shape) or not isinstance(dtype, str):
        raise ValueError("Target mask mismatch")
    y["values"] = _masked_cells(y.get("values"), mask.get("values"), shape, dtype)
    return working


def _assemble(ds: Dataset, source_id: str, *, allow_masked: bool, dense: bool, origin: Any, options: NumericStorageOptions) -> AssembledDataset:
    mm = ds.multimodal
    source = mm.sources.get(source_id)
    if source is None:
        raise NumericContractError("Unknown selected source")
    if not isinstance(source, TensorSource) or source.values.ndim != 2 or source.values.dtype.kind not in "biuf" or not bool(np.all(source.presence_mask)):
        raise NumericContractError("Dense workflow requires a complete numeric rank-2 source")
    y, mask = mm.y, mm.target_mask
    classification = mm.task_type == "classification"
    if dense and (classification or (y is not None and y.ndim != 1)):
        raise NumericContractError("Dense regression requires one numeric target")
    if y is not None:
        if y.dtype.kind not in "fiu":
            raise NumericContractError(f"Unsupported public target dtype {y.dtype}: bool/string/object targets need an existing codebook and no encoder is applied")
        if not allow_masked and not bool(np.all(mask)):
            raise NumericContractError("Matrix projection requires observed targets")
        if classification and ((y.ndim != 1 and not allow_masked) or y.dtype != np.int64):
            raise NumericContractError("Matrix classification requires one int64 target vector")
    record = ds.to_dict()  # the normalized record: the selected source's raw_projection_source derives from it
    supplied = _provenance(origin, ds)  # the original input snapshot: provenance only
    samples, origins, partitions = list(mm.sample_ids), list(ds.origin_ids), [str(p) for p in mm.partitions]
    group_key = "group_id" if mm.groups is not None else "independent_unit_id" if mm.independent_unit_ids is not None else "origin_id"
    assembled = AssembledDataset(name=mm.name, task_type=mm.task_type or "regression", signal_type="unknown", n_sources=1)
    assembled.identity = {
        "source_ids": [source_id],
        "sample_id": "sample_id",
        "observation_id": "sample_id",
        "repetition_id": "repetition_id" if mm.repetition_ids is not None else None,
        "group_id": group_key,
    }
    unavailable: dict[str, str] | None = None
    try:
        identity: dict[str, str] | None = labelled_dataset_content_identity(origin)
    except ContentNotRepresentableError:
        # a valid record whose request-free binary64 identity cannot exist (e.g. an int64 not exact in binary64):
        # the optional whole link is None with an explicit reason; execution may still proceed under a valid selection
        identity = None
        unavailable = {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}  # static spelling, never the exception text
    except ValueError:
        if not allow_masked:
            raise
        # masked route only: the ORIGINAL record (with its withheld-cell placeholders) is not a valid v1 whole record,
        # so the optional link is simply absent, with no reason (the owner's `.ok()`); the package itself is valid
        identity = None
    assembled.audits = [
        {
            "source": ds.to_dict()["schema"],
            "selected_source": source_id,
            "owner_dataset_record": supplied,
            "raw_projection_source": next(entry for entry in record["dataset"]["sources"] if entry["name"] == source_id),
            "raw_fold_assignments": list(ds.fold_ids),
            "dataset_content_identity": identity,
            **({"dataset_content_identity_unavailable": unavailable} if unavailable is not None else {}),
            "input_sample_ids": list(samples),
            "raw_source_schema": _source_schema(source, source_id),
            "target_mask": None if mask is None else _array_wire(np.asarray(mask)),
            "target_names": list(mm.target_names),
        }
    ]
    headers = (
        list(source.feature_names)
        if source.feature_names is not None
        else [c if isinstance(c, str) else str(c) for c in (source.axis_coordinates or {}).get("wavelength", ())] or [f"x{i}" for i in range(source.values.shape[1])]
    )
    if len(headers) != source.values.shape[1]:
        raise NumericContractError("Presentation headers disagree with the source width")
    # The role request is applied to the ORIGINAL cells (integer -> selected storage directly), never after an F64 cast.
    values = to_ir_f64(source.values, options.features)
    y64: np.ndarray | None = None
    if y is not None:
        y64 = integers_to_f64(y) if classification else to_ir_f64(y, options.targets)  # class labels stay exact
        y64 = np.where(np.asarray(mask), y64, np.nan)  # a withheld label is a false mask cell, never a value
        y64 = y64.reshape(-1, 1) if y64.ndim == 1 else y64
    for partition in _PARTITIONS:
        rows = [i for i, p in enumerate(partitions) if p == partition]
        if not rows:
            continue
        frame: dict[str, list[str]] = {"sample_id": [samples[i] for i in rows], "origin_id": [origins[i] for i in rows]}
        if mm.groups is not None:
            frame["group_id"] = [_render_identity(mm.groups[i]) for i in rows]
        if mm.independent_unit_ids is not None:
            frame["independent_unit_id"] = [_render_identity(mm.independent_unit_ids[i]) for i in rows]
        if mm.repetition_ids is not None:
            frame["repetition_id"] = [_render_identity(mm.repetition_ids[i]) for i in rows]
        block = PartitionBlock(n_samples=len(rows), source_ids=[source_id])
        block.X = [values[rows]]
        block.x_source_dtypes = [source.values.dtype.name]
        block.feature_headers = [list(headers)]
        block.header_units = [(source.axis_units or {}).get("wavelength") or "text"]
        block.signal_types = [None]
        block.processings = [[]]
        block.feature_names = [None if source.feature_names is None else list(source.feature_names)]
        block.feature_axes = [_axes(source)]
        block.representation_ids = [source.representation_id]
        if y64 is not None and y is not None:
            block.y = y64[rows]
            block.y_headers = list(mm.target_names)
            block.y_source_dtype = y.dtype.name
        block.metadata = pd.DataFrame(frame)
        assembled.blocks[partition] = block
    train = [i for i, p in enumerate(partitions) if p == "train"]
    fold_ids = list(ds.fold_ids)
    labels = sorted({str(fold_ids[i]) for i in train if fold_ids[i] is not None})
    if labels and any(fold_ids[i] is None for i in train):
        raise NumericContractError("Every training row must declare a fold when CV folds are supplied")
    for label in labels:
        validation = [i for i in train if fold_ids[i] == label]
        fitting = [i for i in train if fold_ids[i] != label]
        assembled.folds.append(([train.index(i) for i in fitting], [train.index(i) for i in validation]))
        assembled.fold_provenance.append({"train_observation_ids": [samples[i] for i in fitting], "validation_observation_ids": [samples[i] for i in validation]})
    return assembled


def _options(options: NumericStorageOptions | None) -> NumericStorageOptions:
    return options if options is not None else NumericStorageOptions()


def _package(value: Any, source_id: str, options: NumericStorageOptions | None, *, allow_masked: bool, dense: bool) -> tuple[DatasetPackage, Dataset]:
    origin = _origin(value)
    working = _masked_materialization(origin) if allow_masked else origin
    ds = public_dataset(working if isinstance(working, Dataset) else copy.deepcopy(working))  # the working copy is validated; ORIGIN stays the provenance
    chosen = _options(options)
    assembled = _assemble(ds, source_id, allow_masked=allow_masked, dense=dense, origin=origin, options=chosen)
    package = DatasetPackage.from_assembled(assembled, features=chosen.features, targets=chosen.targets, weights=chosen.weights, consumer=chosen.consumer)
    return dataclasses.replace(package, source_evidence=(origin, source_id, chosen, allow_masked)), ds


def matrix_dataset_package_with_storage(value: Any, source_id: str, options: NumericStorageOptions | None = None) -> DatasetPackage:
    """Complete matrix-source projection with ordered numeric target columns (observed targets only)."""
    return _package(value, source_id, options, allow_masked=False, dense=False)[0]


def matrix_dataset_package(value: Any, source_id: str) -> DatasetPackage:
    """:func:`matrix_dataset_package_with_storage` with default owner selection."""
    return matrix_dataset_package_with_storage(value, source_id)


def dense_dataset_package_with_storage(value: Any, source_id: str, options: NumericStorageOptions | None = None) -> DatasetPackage:
    """Historical one-target regression projection (a classification or wide target refuses)."""
    return _package(value, source_id, options, allow_masked=False, dense=True)[0]


def dense_dataset_package(value: Any, source_id: str) -> DatasetPackage:
    """:func:`dense_dataset_package_with_storage` with default owner selection."""
    return dense_dataset_package_with_storage(value, source_id)


def masked_matrix_dataset_package_with_storage(value: Any, source_id: str, options: NumericStorageOptions | None = None) -> tuple[DatasetPackage, dict[str, Any]]:
    """Explicit masked projection: false target cells stay false mask cells, never observed zeros."""
    package, ds = _package(value, source_id, options, allow_masked=True, dense=False)
    mm = ds.multimodal
    projection: dict[str, Any] = {
        "sample_ids": list(mm.sample_ids),
        "target_names": list(mm.target_names),
        "target_mask": None if mm.target_mask is None else _array_wire(np.asarray(mm.target_mask)),
    }
    projection["mask_content_fingerprint"] = hashlib.sha256(canonical_content_bytes(projection)).hexdigest()
    return package, projection


def masked_matrix_dataset_package(value: Any, source_id: str) -> tuple[DatasetPackage, dict[str, Any]]:
    """:func:`masked_matrix_dataset_package_with_storage` with default owner selection."""
    return masked_matrix_dataset_package_with_storage(value, source_id)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)  # keeps the sign of -0.0


def verified_dataset_identity(package: DatasetPackage) -> dict[str, str] | None:
    """The unchanged whole-dataset identity, only after the package is proven to derive from the record.

    The package is rebuilt from its retained original record and compared in full
    (payload bytes, masks, axes, folds, audits; signed zero included). Any difference refuses.
    """
    if package.source_evidence is None:
        raise NumericContractError("No original public record is available")
    record, source_id, options, masked = package.source_evidence
    rebuilt, _ = _package(record, source_id, options, allow_masked=masked, dense=False)
    if _canonical(rebuilt.to_full_dict()) != _canonical(package.to_full_dict()):
        raise NumericContractError("The package does not derive from its claimed public record")
    try:
        return labelled_dataset_content_identity(record)
    except ContentNotRepresentableError:
        return None  # valid record, identity not representable: the link is None, not an error
    except ValueError:
        if masked:
            return None  # withheld-cell placeholders made the original an invalid whole record: no link, no reason
        raise


def _requests_of(package: DatasetPackage, source_id: str) -> NumericStorageOptions:
    """The role requests a package's own precision records say were made (rebuild inputs, never proof)."""
    feature = next((p for part in package.partitions.values() for _, p in part.payloads if isinstance(p, FeatureMatrix) and p.source_id == source_id), None)
    table = next((p for part in package.partitions.values() for _, p in part.payloads if isinstance(p, TargetTable)), None)
    numeric = [c.block.precision for c in table.columns if c.block.dtype != "string-label"] if table is not None else []
    return NumericStorageOptions(
        features=feature.matrix.precision.requested_dtype if feature else None,
        targets=numeric[0].requested_dtype if numeric else None,
        consumer=feature.matrix.precision.consumer if feature else None,
    )


def recover_source_evidence(package: DatasetPackage) -> DatasetPackage:
    """Bind the serialized original record as evidence ONLY if the package rebuilds from it exactly.

    A serialized record is a candidate, not a certificate: it must pass the existing
    ``Dataset`` owner validation and an exact rebuild (typed blocks, P, masks, axes, folds,
    audits). Anything else declines the proof (the package is returned unchanged, with no
    evidence) and never changes the package's exact data.
    """
    if package.source_evidence is not None:
        return package
    audit = next((a for a in package.audits if isinstance(a, dict) and "owner_dataset_record" in a and "selected_source" in a), None)
    if audit is None or not isinstance(audit["selected_source"], str):
        return package
    try:
        source_id = audit["selected_source"]
        candidate = dataclasses.replace(package, source_evidence=(_origin(audit["owner_dataset_record"]), source_id, _requests_of(package, source_id), True))
        verified_dataset_identity(candidate)
    except (ValueError, TypeError, KeyError, StopIteration):
        return package
    return candidate


def attach_source_record(package: DatasetPackage, record: Any, source_id: str, options: NumericStorageOptions | None = None, *, masked: bool = False) -> DatasetPackage:
    """Bind (and verify) the original public record to a package, e.g. one read from the full wire."""
    candidate = dataclasses.replace(package, source_evidence=(_origin(record), source_id, _options(options), masked))
    verified_dataset_identity(candidate)
    return candidate


__all__ = [
    "attach_source_record",
    "recover_source_evidence",
    "dense_dataset_package",
    "dense_dataset_package_with_storage",
    "masked_matrix_dataset_package",
    "masked_matrix_dataset_package_with_storage",
    "matrix_dataset_package",
    "matrix_dataset_package_with_storage",
    "verified_dataset_identity",
]
