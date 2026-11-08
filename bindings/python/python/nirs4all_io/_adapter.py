# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Lazy SpectroDataset adapter (EPIC 11.1 / 12.1).

Consumes the full-precision dict from the native ``assembled_full`` and rebuilds a
nirs4all ``SpectroDataset``. The build flow mirrors the Python MVP's
``materialize/spectrodataset.py`` (which mirrors ``DatasetConfigs._load_dataset``)
so the binding's output equals nirs4all's own ``DatasetConfigs`` (parity oracle,
``tests/test_parity.py``). ``nirs4all`` is imported lazily INSIDE
``to_spectrodataset`` — never at module import — so importing ``nirs4all_io``
never imports ``nirs4all`` (import-boundary test). numpy/pandas are not nirs4all.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._assembled import require_assembled_dataset

_PARTITION_ORDER = ("train", "test", "val", "predict")
_WEIGHT_COL = "__sample_weight__"


def _matrix(d: dict[str, Any]) -> np.ndarray:
    """Decode owner-selected little-endian numeric bytes, restoring missing cells."""
    dtype = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}.get(d["dtype"])
    if dtype is None:
        raise ValueError("Unsupported assembled numeric dtype")
    array = np.frombuffer(bytes.fromhex(d["data_bytes_hex"]), dtype=dtype).copy()
    validity = d["validity"]
    if len(validity) != array.size or any(type(value) is not bool for value in validity):
        raise ValueError("Invalid assembled numeric validity")
    array[~np.asarray(validity, dtype=bool)] = np.nan
    return array.reshape(d["n_rows"], d["n_cols"])


def _frame(meta: dict) -> pd.DataFrame:
    """Keep exact wire cells: never let pandas infer float, bool or text types."""
    if any(len(c["values"]) != meta["n_rows"] for c in meta["columns"]):
        raise ValueError("SpectroDataset metadata rows differ from sample rows")
    return pd.DataFrame({c["name"]: pd.Series(c["values"], dtype=object) for c in meta["columns"]}, index=range(meta["n_rows"]))


def _string_labels(block: dict[str, Any]) -> np.ndarray:
    """Unframe the owner's U64-length-prefixed UTF-8 labels, without interpreting values."""
    if block["dtype"] != "string-label" or block["n_cols"] != 1:
        raise ValueError("Invalid assembled string-label block")
    data = bytes.fromhex(block["data_bytes_hex"])
    validity = block["validity"]
    if len(validity) != block["n_rows"] or any(type(value) is not bool for value in validity):
        raise ValueError("Invalid assembled string-label validity")
    values: list[Any] = []
    offset = 0
    for valid in validity:
        if offset + 8 > len(data):
            raise ValueError("Truncated assembled string-label length")
        length = int.from_bytes(data[offset:offset + 8], "little")
        offset += 8
        end = offset + length
        if end > len(data):
            raise ValueError("Truncated assembled string-label value")
        label = data[offset:end].decode("utf-8")
        values.append(label if valid else np.nan)
        offset = end
    if offset != len(data):
        raise ValueError("Trailing assembled string-label bytes")
    return np.asarray(values, dtype=object).reshape(block["n_rows"], 1)


def _decode_targets(y: dict[str, Any]) -> np.ndarray:
    """Decode each declared target kind; heterogeneous columns retain their types."""
    columns = []
    for column in y["columns"]:
        kind = column["kind"]
        if kind == "classification_string":
            columns.append(_string_labels(column["block"]))
        elif kind in {"numeric", "classification_integer"}:
            columns.append(_matrix(column["block"]))
        else:
            raise ValueError(f"Unsupported assembled target kind {kind!r}")
    if not columns:
        raise ValueError("Assembled target table requires columns")
    # Object storage prevents NumPy from converting numeric peers into text.
    if any(column.dtype == object for column in columns):
        result = np.empty((columns[0].shape[0], len(columns)), dtype=object)
        for index, column in enumerate(columns):
            result[:, index] = column[:, 0]
        return result
    return np.concatenate(columns, axis=1)


def _one_or_list(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def _permutation(authority: list[Any], current: list[Any], label: str) -> list[int]:
    if authority == current:
        return list(range(len(authority)))
    if len(set(authority)) != len(authority) or len(set(current)) != len(current) or set(authority) != set(current):
        raise ValueError(f"SpectroDataset {label} layout differs or is ambiguous")
    return [current.index(key) for key in authority]


def _axis_cells(axis: dict[str, Any], width: int) -> list[Any] | None:
    coordinates = axis.get("coordinates")
    if coordinates is None:
        return None
    if coordinates["kind"] == "numeric":
        data = bytes.fromhex(coordinates["data_bytes_hex"])
        if coordinates["count"] != width or len(data) != 8 * width:
            raise ValueError("SpectroDataset feature axis length differs from matrix width")
        return [data[8 * column:8 * column + 8] for column in range(width)]
    if coordinates["kind"] != "string":
        raise ValueError("SpectroDataset unsupported feature axis coordinates")
    values = coordinates["values"]
    if len(values) != width:
        raise ValueError("SpectroDataset feature axis length differs from matrix width")
    return list(values)


def _feature_identity(block: dict[str, Any], index: int) -> tuple[list[Any], list[Any]]:
    headers = block["feature_headers"][index]
    width = block["x"][index]["n_cols"]
    if len(headers) != width:
        raise ValueError("SpectroDataset feature header count differs from matrix width")
    axes = block["feature_axes"][index]
    cells = [_axis_cells(axis, width) for axis in axes]
    numeric = [axis_cells for axis, axis_cells in zip(axes, cells, strict=True) if axis_cells is not None and axis["coordinates"]["kind"] == "numeric"]
    keys = [tuple(axis_cells[column] for axis_cells in numeric) for column in range(width)] if numeric else list(headers)
    return keys, [({key: value for key, value in axis.items() if key != "coordinates"}, axis_cells) for axis, axis_cells in zip(axes, cells, strict=True)]


def _source_alignment(blocks: dict[str, Any], order: list[str]) -> dict[str, list[tuple[int, list[int]]]]:
    first = blocks[order[0]]
    count = len(first["x"])
    ids = [blocks[part]["source_ids"] for part in order]
    if any(len(blocks[part]["x"]) != count for part in order):
        raise ValueError("SpectroDataset feature source count differs across partitions")
    if any(len(names) != count or len(set(names)) != count for names in ids):
        raise ValueError("SpectroDataset source identities are missing or duplicated")
    common = all(set(names) == set(ids[0]) for names in ids)
    disjoint = all(not set(left) & set(right) for i, left in enumerate(ids) for right in ids[i + 1:])
    if not common and not disjoint:
        raise ValueError("SpectroDataset source identities partially overlap")
    aligned = {}
    for part in order:
        block = blocks[part]
        if any(len(block[field]) != count for field in ("feature_headers", "header_units", "signal_types", "representation_ids", "feature_axes", "processings")):
            raise ValueError("SpectroDataset feature source descriptors are unaligned")
        slots = [block["source_ids"].index(name) for name in first["source_ids"]] if common else list(range(count))
        entries = []
        for slot, index in enumerate(slots):
            authoritative, expected_axes = _feature_identity(first, slot)
            current, axes = _feature_identity(block, index)
            permutation = _permutation(authoritative, current, "feature")
            actual_axes = [(description, None if cells is None else [cells[column] for column in permutation]) for description, cells in axes]
            expected_procs = [item["name"] for item in first["processings"][slot]]
            actual_procs = [item["name"] for item in block["processings"][index]]
            if len(set(actual_procs)) != len(actual_procs) or set(actual_procs) != set(expected_procs):
                raise ValueError("SpectroDataset processing layout differs or is ambiguous")
            for field in ("header_units", "signal_types", "representation_ids"):
                if first[field][slot] != block[field][index]:
                    raise ValueError(f"SpectroDataset {field} layout differs")
            if actual_axes != expected_axes:
                raise ValueError("SpectroDataset exact feature axes differ")
            matrix = block["x"][index]
            if matrix["n_rows"] != block["n_samples"] or any((item["matrix"]["n_rows"], item["matrix"]["n_cols"]) != (matrix["n_rows"], matrix["n_cols"]) for item in block["processings"][index]):
                raise ValueError("SpectroDataset feature/processing rows or columns are unaligned")
            entries.append((index, permutation))
        aligned[part] = entries
    return aligned


def _consumer_matrix(block: dict[str, Any]) -> np.ndarray:
    """Ask the storage owner to select consumer bytes, preserving its carried record."""
    if block["dtype"] != "float32":
        from ._native import select_spectrodataset_features

        block = select_spectrodataset_features(block)
    return _matrix(block)


def _target_table(blocks: dict[str, Any], order: list[str]) -> np.ndarray | None:
    labelled = [index for index, part in enumerate(order) if blocks[part]["y"] is not None]
    if not labelled:
        return None
    headers = blocks[order[labelled[0]]]["y_headers"]
    kinds: list[str | None] = [None] * len(headers)
    decoded: dict[str, np.ndarray] = {}
    for part in order:
        block = blocks[part]
        if block["y"] is None:
            continue
        columns = block["y"]["columns"]
        names = [column["name"] for column in columns]
        if names != block["y_headers"] or any(column["block"]["n_rows"] != block["n_samples"] or column["block"]["n_cols"] != 1 for column in columns):
            raise ValueError("SpectroDataset target shape differs from its header/row layout")
        if any(column["block"]["precision"]["requested_dtype"] == "float64" for column in columns):
            raise ValueError("SpectroDataset cannot guarantee an explicitly requested float64 numerical target view")
        permutation = _permutation(headers, names, "target")
        table = _decode_targets(block["y"])
        for position, column in enumerate(permutation):
            if np.any(~pd.isna(table[:, column])):
                kind = "categorical" if columns[column]["kind"] == "classification_string" else "numeric"
                if kinds[position] is not None and kinds[position] != kind:
                    raise ValueError(f"SpectroDataset target kind differs for {headers[position]!r}")
                kinds[position] = kind
        decoded[part] = table[:, permutation]
    parts = [decoded.get(part, np.full((blocks[part]["n_samples"], len(headers)), np.nan)) for part in order[:labelled[-1] + 1]]
    return np.concatenate(parts, axis=0)


def _metadata_table(blocks: dict[str, Any], order: list[str]) -> pd.DataFrame | None:
    frames: dict[str, pd.DataFrame] = {}
    columns: list[str] = []
    last = -1
    for index, part in enumerate(order):
        block = blocks[part]
        frame = _frame(block["metadata"]) if block["metadata"] is not None else pd.DataFrame(index=range(block["n_samples"]))
        if len(frame) != block["n_samples"]:
            raise ValueError("SpectroDataset metadata rows differ from sample rows")
        if block["weights"] is not None:
            if (block["weights"]["n_rows"], block["weights"]["n_cols"]) != (block["n_samples"], 1):
                raise ValueError("SpectroDataset weights rows or columns are unaligned")
            frame[_WEIGHT_COL] = _matrix(block["weights"]).reshape(-1)
        if len(frame.columns):
            last = index
            columns.extend(column for column in frame.columns if column not in columns)
            frames[part] = frame
    if last < 0:
        return None
    result = {}
    for column in columns:
        cells = [cell for part in order[:last + 1] for cell in (frames[part][column].tolist() if part in frames and column in frames[part] else [None] * blocks[part]["n_samples"])]
        if column == _WEIGHT_COL:
            result[column] = pd.Series([np.nan if pd.isna(cell) else cell for cell in cells], dtype="float64")
        else:
            result[column] = pd.Series(cells, dtype=object)
    return pd.DataFrame(result)


def to_spectrodataset(full: dict, *, spectro_dataset_cls: type | None = None) -> Any:
    """Build a SpectroDataset from the full-precision assembled dict."""
    require_assembled_dataset(full)
    blocks = full["blocks"]
    order = [p for p in _PARTITION_ORDER if p in blocks and blocks[p]["x"]]
    alignment = _source_alignment(blocks, order) if order else {}
    targets = _target_table(blocks, order)
    metadata = _metadata_table(blocks, order)
    canonical = blocks[order[0]] if order else {}
    prepared: dict[str, tuple[list[np.ndarray], list[list[tuple[str, np.ndarray]]]]] = {}
    names = [[item["name"] for item in source] for source in canonical.get("processings", [])]
    # Complete decoding and owner selection before the SDK constructor is touched.
    for part in order:
        block = blocks[part]
        xs = []
        procs = []
        for slot, (index, permutation) in enumerate(alignment[part]):
            xs.append(_consumer_matrix(block["x"][index])[:, permutation])
            by_name = {item["name"]: item["matrix"] for item in block["processings"][index]}
            procs.append([(name, _consumer_matrix(by_name[name])[:, permutation]) for name in names[slot]])
        prepared[part] = xs, procs
    if spectro_dataset_cls is None:
        from nirs4all.data import SpectroDataset  # lazy: the only nirs4all touch-point

        spectro_dataset_cls = SpectroDataset

    ds = spectro_dataset_cls(name=full["name"])
    accumulated: list[dict[str, list[np.ndarray]]] = []
    first = True
    for part in order:
        xs, procs = prepared[part]
        ds.add_samples(
            _one_or_list(xs),
            {"partition": part},
            headers=_one_or_list(canonical["feature_headers"]),
            header_unit=_one_or_list(canonical["header_units"]),
        )
        if first:
            for src_idx, sig in enumerate(canonical["signal_types"]):
                if sig:
                    ds.set_signal_type(sig, src=src_idx, forced=False)
            accumulated = [{} for _ in xs]
        for src_idx, source_procs in enumerate(procs):
            for name, array in source_procs:
                accumulated[src_idx].setdefault(name, []).append(array)
        first = False

    if targets is not None:
        ds.add_targets(targets)
    if metadata is not None:
        ds.add_metadata(metadata)
    for src_idx, accumulated_procs in enumerate(accumulated):
        for name, parts in accumulated_procs.items():
            array = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            ds.add_features([array], [name], source=src_idx)

    if full["task_type"] and full["task_type"] != "auto":
        ds.set_task_type(full["task_type"], forced=True)
    if full["folds"]:
        ds.set_folds([tuple(f) for f in full["folds"]])
    if full["repetition"]:
        ds.set_repetition(full["repetition"])
    agg = full["aggregate"]
    if agg is not None:
        ds.set_aggregate(agg["by"] if agg["by"] else True)
        ds.set_aggregate_method(agg["method"])
        if agg["exclude_outliers"]:
            ds.set_aggregate_exclude_outliers(True, agg["outlier_threshold"])
    return ds


__all__ = ["to_spectrodataset"]
