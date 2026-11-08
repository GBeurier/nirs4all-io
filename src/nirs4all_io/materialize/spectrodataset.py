# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Adapt an :class:`AssembledDataset` to a nirs4all ``SpectroDataset`` (Epic 4.1b).

This is the **sole** nirs4all touch-point: the ``SpectroDataset`` class is
imported lazily here (never at module import â€” see the import-boundary test).
The build flow mirrors ``DatasetConfigs._load_dataset``: construct empty, then
per partition ``add_samples({"partition": p})`` -> ``add_targets`` (appends) ->
``add_metadata``, then ``set_signal_type/task_type/folds/repetition/aggregate``.

``spectro_dataset_cls`` can be injected (a recording double) so the adapter is
testable without nirs4all installed.

SpectroDataset is an existing **F32-only feature/processing consumer**: unless an
explicit float32 request is given, X and named processings are quantized once to
binary32 (nearest-even, overflow / nonzero-underflow refused) and the projection
is recorded with consumer origin; an explicit float64 feature request refuses
before the SDK is touched. Targets and weights are NOT blindly cast: default
targets keep their genuine raw values/dtypes (regression float64, integer labels,
strings through the explicit codebook) and the SDK's own encoder owns any numeric
view; the IO base target receipt stays the raw semantic labels. An explicit
float64 numerical-target guarantee refuses (the SDK narrows targets itself);
an explicit float32 target request obeys the owner integer cap and exact-label
rules. Missing cells stay NaN, never imputed.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pandas as pd

from .assemble import AssembledDataset, PartitionBlock
from .numeric import (
    CONSUMER_SPECTRODATASET,
    ExactBlock,
    NumericContractError,
    NumericStorageOptions,
    PrecisionRecord,
    integers_to_f64,
    select_block,
    to_ir_f64,
    validate_label,
)

_PARTITION_ORDER = ("train", "test", "val", "predict")
# Conventional metadata column name carrying per-sample training weights
# (`role: weights` columns). The SpectroDataset has no first-class weight slot,
# so we surface them as metadata; the user/pipeline can read them off as
# `dataset.metadata({...})[_WEIGHT_COL]` and pass to `sample_weight` at fit.
_WEIGHT_COL = "__sample_weight__"


def _decode_targets(y: np.ndarray, headers: list[str], categorical: dict) -> np.ndarray:
    """Restore categorical labels from the assembled wire's explicit codebooks."""
    if not categorical:
        return y
    restored = y.astype(object)
    for index, header in enumerate(headers):
        if header not in categorical:
            continue
        categories = categorical[header].get("categories")
        codes = y[:, index]
        observed = ~np.isnan(codes)
        present_codes = codes[observed]
        if (
            not isinstance(categories, list)
            or not all(isinstance(label, str) for label in categories)
            or not np.all(np.isfinite(present_codes))
            or not np.all(present_codes == np.floor(present_codes))
            or np.any(present_codes < 0)
            or np.any(present_codes >= len(categories))
        ):
            raise ValueError(f"Invalid categorical target codebook for {header!r}")
        restored[:, index] = np.nan
        restored[observed, index] = np.asarray(categories, dtype=object)[present_codes.astype(np.intp)]
    return restored


def _one_or_list(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def _consumer_features(matrix: Any, block: PartitionBlock, idx: int, options: NumericStorageOptions) -> tuple[np.ndarray, dict[str, Any], PrecisionRecord]:
    """The consumer F32 projection of one feature matrix (NaN = missing); a carried selection is retained or honestly re-selected."""
    carried = block.x_precisions[idx] if block.x_precisions else None
    source_dtype = block.x_source_dtypes[idx] if idx < len(block.x_source_dtypes) else None
    exact = select_block(matrix, carried=carried, source_dtype=source_dtype, request=options.features, consumer=CONSUMER_SPECTRODATASET, feature=True)
    return exact.to_ir().astype(exact.dtype), exact.precision.to_dict(), exact.precision


def _requested_targets(block: PartitionBlock, task_type: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Explicit float32 targets: owner quantization plus exact-label checks (no strings)."""
    from .package import TARGET_KINDS, _target_kind

    y = np.asarray(block.y)
    if y.dtype.kind not in "fiu" or y.ndim != 2:
        raise NumericContractError("A numeric precision request cannot apply to string/non-numeric targets")
    kinds = [_target_kind(name, task_type, block.y_categorical) for name in block.y_headers]
    if TARGET_KINDS[2] in kinds:
        raise NumericContractError("A numeric precision request cannot apply to string targets")
    ir = np.column_stack([integers_to_f64(y[:, i]) if kind == TARGET_KINDS[1] else to_ir_f64(y[:, i], "float32") for i, kind in enumerate(kinds)]) if kinds else y.astype(np.float64)
    exact = ExactBlock.from_matrix(ir, source_dtype=block.y_source_dtype, request="float32", consumer=None, feature=False)
    # Exact-label rules belong to classification labels only: a numeric (regression)
    # target may round to F32 even when its source dtype is an integer.
    for column, kind in enumerate(kinds):
        if kind != TARGET_KINDS[1]:
            continue
        for label, valid in zip(y[:, column], exact.validity[:, column], strict=True):
            if valid:
                if float(label) != np.floor(float(label)):
                    raise NumericContractError("Integer-label targets must be integers")
                validate_label(int(label), "float32")
    return exact.to_ir().astype(np.float32), [exact.precision.to_dict()]


def to_spectrodataset(
    assembled: AssembledDataset,
    *,
    spectro_dataset_cls: type | None = None,
    features: Any = None,
    targets: Any = None,
    weights: Any = None,
) -> Any:
    """Materialize ``assembled`` into a SpectroDataset (lazy import unless injected)."""
    return project_spectrodataset(assembled, spectro_dataset_cls=spectro_dataset_cls, features=features, targets=targets, weights=weights)[0]


def _permutation(authority: list[Any], current: list[Any], label: str) -> list[int]:
    """Keep identical layouts; otherwise require an unambiguous identity map."""
    if authority == current:
        return list(range(len(authority)))
    if len(set(authority)) != len(authority) or len(set(current)) != len(current) or set(authority) != set(current):
        raise NumericContractError(f"SpectroDataset {label} layout differs or is ambiguous")
    return [current.index(key) for key in authority]


def _exact_layout(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _exact_layout(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_exact_layout(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return ("float", struct.pack(">d", float(value)))
    return (type(value).__name__, value)


def _feature_identity(block: PartitionBlock, index: int) -> tuple[list[Any], list[Any]]:
    headers = block.feature_headers[index]
    width = block.X[index].shape[1]
    if len(headers) != width:
        raise NumericContractError("SpectroDataset feature header count differs from matrix width")
    axes = block.feature_axes[index] if block.feature_axes else []
    numeric = [axis["coordinates"] for axis in axes if isinstance(axis.get("coordinates"), (list, tuple)) and len(axis["coordinates"]) == width and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in axis["coordinates"])]
    keys = [tuple(struct.pack(">d", float(coordinates[column])) for coordinates in numeric) for column in range(width)] if numeric else list(headers)
    metadata: list[Any] = []
    for axis in axes:
        coordinates = axis.get("coordinates")
        description = _exact_layout({key: value for key, value in axis.items() if key != "coordinates"})
        if coordinates is None:
            metadata.append((description, None))
        elif len(coordinates) == width:
            values = [struct.pack(">d", float(v)) for v in coordinates] if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in coordinates) else [_exact_layout(v) for v in coordinates]
            metadata.append((description, values))
        else:
            raise NumericContractError("SpectroDataset feature axis length differs from matrix width")
    return keys, metadata


def _source_alignment(assembled: AssembledDataset, order: list[str]) -> dict[str, list[tuple[int, list[int]]]]:
    """Map declared source slots, then align each slot's scientific columns."""
    first = assembled.blocks[order[0]]
    count = len(first.X)
    ids = [assembled.blocks[part].source_ids for part in order]
    if any(len(assembled.blocks[part].X) != count for part in order):
        raise NumericContractError("SpectroDataset feature source count differs across partitions")
    # Old directly constructed, unnamed IR remains positional. Public assembled
    # inputs carry source IDs; partially identified blocks are never guessed.
    unnamed = all(not names for names in ids)
    if not unnamed and any(len(names) != count or len(set(names)) != count for names in ids):
        raise NumericContractError("SpectroDataset source identities are missing or duplicated")
    common = unnamed or all(set(names) == set(ids[0]) for names in ids)
    disjoint = all(not set(left) & set(right) for i, left in enumerate(ids) for right in ids[i + 1:])
    if not common and not disjoint:
        raise NumericContractError("SpectroDataset source identities partially overlap")
    aligned = {}
    for part in order:
        block = assembled.blocks[part]
        slots = list(range(count)) if unnamed or not common else [block.source_ids.index(name) for name in first.source_ids]
        entries = []
        for slot, index in enumerate(slots):
            authoritative, expected_axes = _feature_identity(first, slot)
            current, axes = _feature_identity(block, index)
            permutation = _permutation(authoritative, current, "feature")
            actual_axes = [(description, None if coordinates is None else [coordinates[c] for c in permutation]) for description, coordinates in axes]
            expected_procs = [name for name, _ in first.processings[slot]] if first.processings else []
            actual_procs = [name for name, _ in block.processings[index]] if block.processings else []
            if len(set(actual_procs)) != len(actual_procs) or set(actual_procs) != set(expected_procs):
                raise NumericContractError("SpectroDataset processing layout differs or is ambiguous")
            for field in ("header_units", "signal_types", "representation_ids"):
                left, right = getattr(first, field), getattr(block, field)
                if (left[slot] if left else None) != (right[index] if right else None):
                    raise NumericContractError(f"SpectroDataset {field} layout differs")
            if actual_axes != expected_axes:
                raise NumericContractError("SpectroDataset exact feature axes differ")
            if block.X[index].shape[0] != block.n_samples or any(matrix.shape != block.X[index].shape for _, matrix in (block.processings[index] if block.processings else [])):
                raise NumericContractError("SpectroDataset feature/processing rows or columns are unaligned")
            entries.append((index, permutation))
        aligned[part] = entries
    return aligned


def _target_table(assembled: AssembledDataset, order: list[str], options: NumericStorageOptions, receipt: dict[str, Any]) -> np.ndarray | None:
    labelled = [i for i, part in enumerate(order) if assembled.blocks[part].y is not None]
    if not labelled:
        return None
    headers = assembled.blocks[order[labelled[0]]].y_headers
    kinds: list[str | None] = [None] * len(headers)
    decoded: dict[str, np.ndarray] = {}
    for part in order:
        block = assembled.blocks[part]
        if block.y is None:
            continue
        y = block.y
        if y.shape != (block.n_samples, len(block.y_headers)):
            raise NumericContractError("SpectroDataset target shape differs from its header/row layout")
        permutation = _permutation(headers, block.y_headers, "target")
        if len(set(block.y_headers)) != len(block.y_headers) and block.y_categorical:
            raise NumericContractError("SpectroDataset categorical target headers are ambiguous")
        if options.targets is not None:
            y, precision = _requested_targets(block, assembled.task_type)
            receipt["targets"]["precision"] = precision
        restored = _decode_targets(y, block.y_headers, block.y_categorical)
        for position, column in enumerate(permutation):
            if np.any(~pd.isna(restored[:, column])):
                kind = "categorical" if block.y_headers[column] in block.y_categorical else "numeric"
                if kinds[position] is not None and kinds[position] != kind:
                    raise NumericContractError(f"SpectroDataset target kind differs for {headers[position]!r}")
                kinds[position] = kind
        decoded[part] = restored[:, permutation]
    parts = [decoded.get(part, np.full((assembled.blocks[part].n_samples, len(headers)), np.nan)) for part in order[:labelled[-1] + 1]]
    receipt["targets"]["headers"] = list(headers)
    return np.concatenate(parts, axis=0)



def _metadata_tables(assembled: AssembledDataset, order: list[str], options: NumericStorageOptions) -> pd.DataFrame | None:
    """Build each metadata column once, preserving its cells across missing blocks."""
    frames: dict[str, pd.DataFrame] = {}
    columns: list[str] = []
    last = -1
    for index, part in enumerate(order):
        block = assembled.blocks[part]
        frame = block.metadata.copy() if block.metadata is not None else pd.DataFrame(index=range(block.n_samples))
        if len(frame) != block.n_samples:
            raise NumericContractError("SpectroDataset metadata rows differ from sample rows")
        if block.weights is not None:
            weights_block = block.weights
            if options.weights is not None:
                exact = ExactBlock.from_matrix(np.asarray(weights_block).reshape(-1, 1), source_dtype=block.weights_source_dtype, request=options.weights, consumer=None, feature=False)
                weights_block = exact.to_ir()[:, 0].astype(exact.dtype)
            frame[_WEIGHT_COL] = weights_block
        if len(frame.columns):
            last = index
            columns.extend(column for column in frame.columns if column not in columns)
            frames[part] = frame
    if last < 0:
        return None
    result = {}
    for column in columns:
        cells = [cell for part in order[:last + 1] for cell in (frames[part][column].tolist() if part in frames and column in frames[part] else [None] * assembled.blocks[part].n_samples)]
        dtype: Any = object
        if column == _WEIGHT_COL and any(assembled.blocks[part].weights is not None for part in order):
            dtype = options.weights or "float64"
        else:
            original_dtypes = [frame[column].dtype for frame in frames.values() if column in frame]
            if original_dtypes and all(pd.api.types.is_float_dtype(original) for original in original_dtypes) and all(pd.isna(cell) for cell in cells):
                dtype = "float64" if any(original.itemsize > 4 for original in original_dtypes) else "float32"
        if dtype is not object:
            cells = [np.nan if pd.isna(cell) else cell for cell in cells]
        result[column] = pd.Series(cells, dtype=dtype)
    return pd.DataFrame(result)


def project_spectrodataset(
    assembled: AssembledDataset,
    *,
    spectro_dataset_cls: type | None = None,
    features: Any = None,
    targets: Any = None,
    weights: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the SpectroDataset and the truthful consumer-projection receipt."""
    options = NumericStorageOptions(features, targets, weights, CONSUMER_SPECTRODATASET)
    if options.features == "float64":
        raise NumericContractError("SpectroDataset does not support requested float64 features/processings")
    inherited_f64_targets = any(
        precision.requested_dtype == "float64" and precision.dtype != "string-label"
        for block in assembled.blocks.values()
        for precision in block.y_precisions
    )
    if options.targets == "float64" or (options.targets is None and inherited_f64_targets):
        raise NumericContractError("SpectroDataset cannot guarantee float64 numerical targets (the SDK narrows targets itself)")
    receipt: dict[str, Any] = {
        "consumer": CONSUMER_SPECTRODATASET,
        "features": [],
        "processings": [],
        "targets": {"policy": "raw-preserved" if options.targets is None else "float32-requested"},
        "weights": {"policy": "raw-preserved" if options.weights is None else options.weights},
        "numerical_target_view": None,
        "numerical_target_view_declined": "the SDK-owned target encoder/dtype is not an IO claim; the base target receipt stays the raw semantic labels",
    }
    order = [p for p in _PARTITION_ORDER if p in assembled.blocks and assembled.blocks[p].X]
    alignment = _source_alignment(assembled, order) if order else {}
    y_table = _target_table(assembled, order, options, receipt)
    metadata_table = _metadata_tables(assembled, order, options)
    canonical = assembled.blocks[order[0]] if order else None
    if spectro_dataset_cls is None:
        from nirs4all.data import SpectroDataset  # lazy: the only nirs4all runtime touch-point

        spectro_dataset_cls = SpectroDataset

    ds = spectro_dataset_cls(name=assembled.name)
    first = True
    # Accumulate per-source variation arrays across partitions; once all
    # partitions are added (samples are flat across partitions in the
    # SpectroDataset), call add_features once per (source, processing_name).
    accumulated_processings: list[dict[str, list[np.ndarray]]] = []
    for part in order:
        block = assembled.blocks[part]
        if not block.X:
            continue
        projected_x: list[np.ndarray] = []
        projected_procs: list[list[tuple[str, np.ndarray]]] = []
        for slot, (src_idx, permutation) in enumerate(alignment[part]):
            matrix = block.X[src_idx]
            array, precision, governing = _consumer_features(matrix, block, src_idx, options)
            projected_x.append(array[:, permutation])
            receipt["features"].append({"partition": part, "source_id": block.source_ids[src_idx] if src_idx < len(block.source_ids) else None, "slot": slot, "precision": precision})
            projected_list: list[tuple[str, np.ndarray]] = []
            children = block.processings[src_idx] if src_idx < len(block.processings) else []
            child_dtypes = block.processing_source_dtypes[src_idx] if src_idx < len(block.processing_source_dtypes) else []
            child_records = block.processing_precisions[src_idx] if src_idx < len(block.processing_precisions) else []
            positions = {name: position for position, (name, _) in enumerate(children)}
            names = [name for name, _ in canonical.processings[slot]] if canonical is not None and canonical.processings else []
            for name in names:
                position = positions[name]
                _, proc = children[position]
                # Each processing keeps ITS OWN source dtype / selected record; only the F32 storage is common.
                stored = select_block(
                    proc,
                    carried=child_records[position] if child_records else None,
                    source_dtype=child_dtypes[position] if child_dtypes else None,
                    request=options.features,
                    consumer=CONSUMER_SPECTRODATASET,
                    feature=True,
                )
                if stored.dtype != governing.dtype:
                    raise NumericContractError(f"Processing {name!r} does not resolve to the feature source's {governing.dtype} consumer storage")
                projected_list.append((name, stored.to_ir().astype(stored.dtype)[:, permutation]))
                receipt["processings"].append({"partition": part, "name": name, "precision": stored.precision.to_dict()})
            projected_procs.append(projected_list)
        ds.add_samples(
            _one_or_list(projected_x),
            {"partition": part},
            headers=_one_or_list(canonical.feature_headers if canonical is not None else block.feature_headers),
            header_unit=_one_or_list(canonical.header_units if canonical is not None else block.header_units),
        )
        if first:
            for src_idx, sig in enumerate(canonical.signal_types if canonical is not None else block.signal_types):
                if sig:
                    ds.set_signal_type(sig, src=src_idx, forced=False)
            accumulated_processings = [{} for _ in block.X]
        for src_idx, src_procs in enumerate(projected_procs):
            for name, arr in src_procs:
                accumulated_processings[src_idx].setdefault(name, []).append(arr)
        first = False

    if y_table is not None:
        ds.add_targets(y_table)
    if metadata_table is not None:
        ds.add_metadata(metadata_table)

    # Now that all partitions are added, attach the named preprocessing variants
    # as additional processings on each source (parallel to the raw features).
    # `add_features` is wrapped in a list so the inner `if not features:` check
    # in nirs4all's features.update_features stays well-defined for numpy input.
    for src_idx, procs in enumerate(accumulated_processings):
        for name, parts in procs.items():
            arr = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            ds.add_features([arr], [name], source=src_idx)

    if assembled.task_type and assembled.task_type != "auto":
        ds.set_task_type(assembled.task_type, forced=True)
    if assembled.folds:
        ds.set_folds(assembled.folds)
    if assembled.repetition:
        ds.set_repetition(assembled.repetition)
    if assembled.aggregate is not None:
        ds.set_aggregate(assembled.aggregate.by if assembled.aggregate.by else True)
        ds.set_aggregate_method(assembled.aggregate.method.value)
        if assembled.aggregate.exclude_outliers:
            ds.set_aggregate_exclude_outliers(True, assembled.aggregate.outlier_threshold)
    return ds, receipt


__all__ = ["to_spectrodataset", "project_spectrodataset", "AssembledDataset", "PartitionBlock"]
