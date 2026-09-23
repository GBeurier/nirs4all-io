# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Assemble explicitly identified, in-memory sources without flattening tensors.

This host-owned Python surface preserves raw arrays for the pipeline's source
adapters. It performs only deterministic identity alignment and row selection:
encoding, imputation, feature extraction and fitting belong to the ML runtime.
Representation IDs and semantic axes use the published ``dag-ml-data`` registry.
The tabular DatasetSpec / AssembledDataset wire formats remain independent.

Example::

    dataset = MultimodalDataset(
        {"nir": TensorSource(spectra, ids, representation_id="signal_1d"),
         "image": TensorSource(images, image_ids, representation_id="rgb_image"),
         "weather": TensorSource(series, ids, representation_id="series_mv"),
         "metadata": TensorSource(table, ids, representation_id="tabular_mixed")},
        sample_ids=ids, y=targets, groups=plant_ids, partitions=partitions,
    )

Strict alignment requires exactly the declared sample IDs in every source;
input row order may differ. Explicit left alignment permits source subsets and
marks absent rows with presence masks. Targets, groups and partition labels
follow the supplied canonical ``sample_ids`` order. ``take`` accepts IDs, never
positional row labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .ragged import RaggedSeriesBatch, RaggedSeriesSource

# Frozen representation vocabulary from dag-ml-data. These are the fixed-size
# array representations supported by this in-memory assembly surface.
_AXES: dict[str, tuple[str, ...]] = {
    "signal_1d": ("sample", "wavelength"),
    "tabular_numeric": ("sample", "feature"),
    "tabular_mixed": ("sample", "column"),
    "sample_metadata": ("sample", "field"),
    "gray_image": ("sample", "height", "width"),
    "rgb_image": ("sample", "height", "width", "channel"),
    "mc_image": ("sample", "height", "width", "channel"),
    "multispectral_image": ("sample", "height", "width", "band"),
    "series_mv": ("sample", "time", "variable"),
}
_MIXED = {"tabular_mixed", "sample_metadata"}
_SOURCE_TYPES = {
    "signal_1d": ("dense_signal", "nirs"),
    "tabular_numeric": ("table", "tabular"),
    "tabular_mixed": ("table", "tabular"),
    "sample_metadata": ("metadata", "metadata"),
    "gray_image": ("gray_image", "image"),
    "rgb_image": ("image_rgb", "image"),
    "mc_image": ("multichannel_image", "image"),
    "multispectral_image": ("multichannel_image", "image"),
    "series_mv": ("time_series", "time_series"),
}
_AXIS_KINDS = {"column": "feature", "field": "feature", "variable": "feature", "band": "channel"}
_JSON_SCHEMA = "nirs4all.multimodal-dataset"
_JSON_SCHEMA_VERSION = 1


def _identities(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of sample IDs, not a string")
    ids = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{label} must contain non-empty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate sample IDs")
    return ids


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _row_positions(values: Sequence[int], size: int) -> np.ndarray:
    indices = np.asarray(values)
    if indices.size == 0 and indices.ndim == 1:
        indices = indices.astype(np.intp)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("row_indices must be a one-dimensional sequence of integer positions")
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("row_indices contains an out-of-range position")
    return indices


def _closed_fields(value: Any, fields: set[str], label: str, *, optional: frozenset[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing, unknown = fields - set(value), set(value) - fields - optional
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={sorted(missing)}, unknown={sorted(map(str, unknown))}")
    return value


def _json_values(value: Any, *, decode: bool = False) -> Any:
    """Represent non-finite scalars explicitly instead of emitting invalid JSON."""
    if isinstance(value, list):
        return [_json_values(item, decode=decode) for item in value]
    if not decode and isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, np.generic):
            raise ValueError(f"Unsupported multimodal JSON scalar dtype: {value.dtype}; no lossless JSON scalar representation")
    if decode and isinstance(value, Mapping):
        tagged = _closed_fields(value, {"nonfinite"}, "non-finite scalar")
        token = tagged["nonfinite"]
        if not isinstance(token, str) or token not in {"nan", "inf", "-inf"}:
            raise ValueError("nonfinite must be 'nan', 'inf' or '-inf'")
        return float(token)
    if type(value) is float and not isfinite(value):
        if decode:
            raise ValueError("Non-finite JSON values require an explicit nonfinite scalar object")
        return {"nonfinite": "nan" if np.isnan(value) else "inf" if value > 0 else "-inf"}
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"Multimodal JSON array cells must be scalar numbers, strings, booleans or null; got {type(value).__name__}")


def _array_to_dict(array: np.ndarray) -> dict[str, Any]:
    if array.dtype.kind not in "biufUO" or (array.dtype.kind == "f" and array.dtype.itemsize > 8):
        raise ValueError(f"Unsupported multimodal JSON dtype: {array.dtype}")
    return {"dtype": str(array.dtype), "shape": list(array.shape), "values": _json_values(array.tolist())}


def _array_from_dict(value: Any, label: str) -> np.ndarray:
    record = _closed_fields(value, {"dtype", "shape", "values"}, label)
    if not isinstance(record["dtype"], str):
        raise ValueError(f"{label}.dtype must be a dtype string")
    try:
        dtype = np.dtype(record["dtype"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}.dtype: {record['dtype']!r}") from exc
    if dtype.kind not in "biufUO" or (dtype.kind == "f" and dtype.itemsize > 8):
        raise ValueError(f"Unsupported {label}.dtype: {dtype}")
    shape = record["shape"]
    if not isinstance(shape, list) or not shape or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError(f"{label}.shape must be a non-empty list of nonnegative integer dimensions")
    if not isinstance(record["values"], list):
        raise ValueError(f"{label}.values must be a nested JSON array")
    decoded = _json_values(record["values"], decode=True)
    try:
        cells = np.asarray(decoded, dtype=object)
    except ValueError as exc:
        raise ValueError(f"{label}.values must form a rectangular array") from exc
    for cell in cells.flat:
        if cell is not None and type(cell) not in (bool, int, float, str):
            raise ValueError(f"{label}.values must form a rectangular array of scalar cells")
        if dtype.kind in "iu":
            bounds = np.iinfo(dtype)
            if type(cell) is not int or cell < bounds.min or cell > bounds.max:
                raise ValueError(f"{label}.values contains a cell outside dtype {dtype}")
        elif dtype.kind == "b" and type(cell) is not bool:
            raise ValueError(f"{label}.values must contain booleans for dtype {dtype}")
        elif dtype.kind == "f" and type(cell) not in (int, float):
            raise ValueError(f"{label}.values must contain numbers for dtype {dtype}")
        elif dtype.kind == "U" and (not isinstance(cell, str) or len(cell) > dtype.itemsize // 4):
            raise ValueError(f"{label}.values contains a string incompatible with dtype {dtype}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(cells, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}.values cannot be represented by dtype {dtype}") from exc
    if dtype.kind == "f" and any(
        (type(cell) is int or isfinite(cell)) and not np.isfinite(converted)
        for cell, converted in zip(cells.flat, array.flat, strict=True)
    ):
        raise ValueError(f"{label}.values overflows dtype {dtype}")
    if array.shape != tuple(shape):
        # JSON [] cannot carry the trailing dimensions after its first empty
        # axis. Preserve those explicit dimensions without reshaping populated
        # arrays or accepting a different leading row structure.
        empty_axis = shape.index(0) if 0 in shape else None
        if empty_axis is None or array.shape != tuple(shape[:empty_axis + 1]):
            raise ValueError(f"{label}.shape {shape} does not match its values shape {array.shape}")
        array = array.reshape(shape)
    return array


def _axis_metadata(
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    axis_units: Mapping[str, str | None] | None,
    axis_coordinates: Mapping[str, Sequence[Any] | np.ndarray] | None,
) -> tuple[Mapping[str, str | None], Mapping[str, tuple[str | int | float, ...]]]:
    """Validate declared axis information without sorting or unit conversion."""
    sizes = dict(zip(axes[1:], shape[1:], strict=True))
    units: dict[str, str | None] = dict.fromkeys(sizes)
    coordinates: dict[str, tuple[str | int | float, ...]] = {}
    for label, mapping in (("axis_units", axis_units), ("axis_coordinates", axis_coordinates)):
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{label} must be a mapping keyed by non-sample axis names")
            if any(axis not in sizes for axis in mapping):
                raise ValueError(f"{label} keys must be non-sample axes from {tuple(sizes)}")
    for axis, unit in (axis_units or {}).items():
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise ValueError(f"Unit for axis {axis!r} must be a non-empty string or None")
        units[axis] = unit
    for axis, raw_values in (axis_coordinates or {}).items():
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, (Sequence, np.ndarray)):
            raise ValueError(f"Coordinates for axis {axis!r} must be a one-dimensional sequence")
        if isinstance(raw_values, np.ndarray) and raw_values.ndim != 1:
            raise ValueError(f"Coordinates for axis {axis!r} must be a one-dimensional sequence")
        if len(raw_values) != sizes[axis]:
            raise ValueError(f"Axis {axis!r} has size {sizes[axis]} but {len(raw_values)} coordinates")
        scalars: list[str | int | float] = []
        for value in raw_values:
            value = value.item() if isinstance(value, np.generic) else value
            if type(value) not in (str, int, float):
                raise ValueError(f"Coordinates for axis {axis!r} must be real numbers or non-empty strings")
            if (isinstance(value, str) and not value.strip()) or (isinstance(value, float) and not isfinite(value)):
                raise ValueError(f"Coordinates for axis {axis!r} must be finite numbers or non-empty strings")
            scalars.append(value)
        numeric = all(not isinstance(value, str) for value in scalars)
        if not numeric and any(not isinstance(value, str) for value in scalars):
            raise ValueError(f"Coordinates for axis {axis!r} cannot mix numeric and string labels")
        if len(set(scalars)) != len(scalars):
            raise ValueError(f"Coordinates for axis {axis!r} must be unique")
        if axis in ("wavelength", "time"):
            if not numeric:
                raise ValueError(f"Coordinates for axis {axis!r} must be numeric")
            numbers = [value for value in scalars if not isinstance(value, str)]
            increasing = all(left < right for left, right in zip(numbers, numbers[1:], strict=False))
            decreasing = all(left > right for left, right in zip(numbers, numbers[1:], strict=False))
            if axis == "time" and not increasing:
                raise ValueError("Coordinates for axis 'time' must be strictly increasing")
            if axis == "wavelength" and not (increasing or decreasing):
                raise ValueError("Coordinates for axis 'wavelength' must be strictly monotonic")
        coordinates[axis] = tuple(scalars)
    ordered_coordinates = {axis: coordinates[axis] for axis in sizes if axis in coordinates}
    return MappingProxyType(units), MappingProxyType(ordered_coordinates)


@dataclass(frozen=True)
class TensorSource:
    """One raw source with explicit row identity and semantic axes.

    ``values`` may be a NumPy array or an array-compatible table. Mixed tables
    retain strings and numeric objects for fold-scoped encoding by the runtime.
    Numeric tensors retain their rank and dtype. Ragged arrays are not accepted.
    Input buffers are copied and marked read-only to isolate assembly from the
    caller's subsequent mutations.

    ``axis_units`` and ``axis_coordinates`` describe non-sample axes explicitly;
    undeclared units stay ``None`` and coordinates are never inferred. Coordinates
    must match their axis size and be unique finite numbers or string labels.
    Wavelength coordinates may increase or decrease strictly; time coordinates
    must increase strictly. Neither source values nor coordinates are reordered
    to repair an invalid axis. Numeric coordinates are never parsed from text.

    ``presence_mask`` is a boolean vector of exactly one flag per sample ID;
    True means the modality is present, as in dag-ml-data's ``PresenceMask``.
    The default marks every supplied row present. An absent row's buffer is
    retained as opaque storage, never treated as an observed measurement.
    An empty source requires an array with its complete non-sample shape, such
    as ``np.empty((0, height, width, 3))`` for an absent RGB source.
    NumPy masked arrays are rejected; use a plain array and ``presence_mask``
    so that missingness cannot be lost during array conversion.
    """

    values: np.ndarray = field(repr=False)
    sample_ids: Sequence[str]
    representation_id: str = field(kw_only=True)
    axes: Sequence[str] | None = field(default=None, kw_only=True)
    feature_names: Sequence[str] | None = field(default=None, kw_only=True)
    axis_units: Mapping[str, str | None] | None = field(default=None, kw_only=True)
    axis_coordinates: Mapping[str, Sequence[Any] | np.ndarray] | None = field(default=None, kw_only=True)
    presence_mask: Sequence[bool] | np.ndarray | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        ids = _identities(self.sample_ids, "source sample_ids")
        if self.representation_id not in _AXES:
            raise ValueError(f"Unsupported in-memory representation_id: {self.representation_id!r}")
        expected_axes = _AXES[self.representation_id]
        raw_values = self.values
        if np.ma.isMaskedArray(raw_values):
            raise ValueError("Source values must be a plain array with explicit presence_mask, not a NumPy masked array")
        if self.representation_id in _MIXED and isinstance(raw_values, (list, tuple)):
            # NumPy otherwise promotes [[20.0, "cultivar-a"]] to strings,
            # silently losing the numeric column before the pipeline sees it.
            raw_values = np.asarray(raw_values, dtype=object)
        values = _readonly(raw_values)
        if values.ndim != len(expected_axes):
            raise ValueError(f"{self.representation_id} requires rank {len(expected_axes)} ({expected_axes}), got shape {values.shape}")
        if values.shape[0] != len(ids):
            raise ValueError(f"Source has {values.shape[0]} rows but {len(ids)} sample IDs")
        if np.ma.isMaskedArray(self.presence_mask):
            raise ValueError("presence_mask must be an explicit boolean array, not a NumPy masked array")
        presence = _readonly(np.ones(len(ids), dtype=bool) if self.presence_mask is None else self.presence_mask)
        if presence.dtype.kind != "b" or presence.shape != (len(ids),):
            raise ValueError("presence_mask must have boolean dtype and exactly one dimension matching source sample_ids")
        if any(size == 0 for size in values.shape[1:]):
            raise ValueError("Source feature, spatial, time and channel axes must be non-empty")
        axes = expected_axes if self.axes is None else tuple(self.axes)
        if axes != expected_axes:
            raise ValueError(f"{self.representation_id} requires axes {expected_axes}, got {axes}")
        if self.representation_id == "rgb_image" and values.shape[-1] != 3:
            raise ValueError("rgb_image requires exactly three channels on the final axis")
        if self.representation_id in _MIXED:
            if values.dtype.kind not in "biufUSO":
                raise ValueError("Mixed table cells must contain numeric, string, boolean or null scalars")
            if values.dtype.kind == "O" and any(
                value is not None and not isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)) for value in values.flat
            ):
                raise ValueError("Mixed table cells must contain numeric, string, boolean or null scalars")
        elif values.dtype.kind not in "biuf":
            raise ValueError(f"{self.representation_id} requires a real numeric tensor, got dtype {values.dtype}")
        names = None if self.feature_names is None else _identities(self.feature_names, "feature_names")
        if names is not None and (values.ndim != 2 or len(names) != values.shape[1]):
            raise ValueError("feature_names must match the columns of a rank-2 source")
        units, coordinates = _axis_metadata(axes, values.shape, self.axis_units, self.axis_coordinates)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "axis_units", units)
        object.__setattr__(self, "axis_coordinates", coordinates)
        object.__setattr__(self, "presence_mask", presence)

    def take(self, sample_ids: Sequence[str]) -> TensorSource:
        """Select and reorder rows by ID, preserving every non-sample axis."""
        ids = _identities(sample_ids, "selection sample_ids")
        lookup = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        missing = set(ids) - lookup.keys()
        if missing:
            raise ValueError(f"Unknown sample IDs in source selection: {sorted(missing)}")
        positions = np.asarray([lookup[sample_id] for sample_id in ids], dtype=np.intp)
        return TensorSource(
            self.values[positions], ids, representation_id=self.representation_id,
            axes=self.axes, feature_names=self.feature_names,
            axis_units=self.axis_units, axis_coordinates=self.axis_coordinates,
            presence_mask=np.asarray(self.presence_mask)[positions],
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Revalidate arrays and restore read-only ownership when unpickling."""
        return (
            _restore_tensor_source,
            (self.values, self.sample_ids, self.representation_id, self.axes, self.feature_names,
             dict(self.axis_units or {}), dict(self.axis_coordinates or {}), self.presence_mask, str(self.values.dtype)),
        )

    def descriptor(self, source_id: str) -> dict[str, Any]:
        """Describe a source without feature values or variable presence flags."""
        type_id, modality = _SOURCE_TYPES[self.representation_id]
        return {
            "source_id": source_id,
            "representation_id": self.representation_id,
            "type_id": type_id,
            "modality": modality,
            "axes": list(self.axes or ()),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "feature_names": None if self.feature_names is None else list(self.feature_names),
            "axis_units": dict(self.axis_units or {}),
            "axis_coordinates": {axis: list(coordinates) for axis, coordinates in (self.axis_coordinates or {}).items()},
            "native_representation": {
                "id": self.representation_id,
                "type_id": type_id,
                "rank": self.values.ndim,
                "axes": [
                    {
                        "name": axis,
                        "kind": _AXIS_KINDS.get(axis, axis),
                        "unit": (self.axis_units or {}).get(axis),
                        "size": size,
                        "variable": False,
                    }
                    for axis, size in zip(self.axes or (), self.values.shape, strict=True)
                ],
                "container": "ndarray",
                "dtype": None if self.values.dtype.kind in "OUS" else str(self.values.dtype),
                "sparse": False,
                "ragged": False,
            },
        }

    def schema_descriptor(self, source_id: str) -> dict[str, Any]:
        """Describe the replay input contract independently of cohort length.

        Values and sample identities are absent. Only the sample-axis size is
        replaced with ``None``; representation, raw dtype, all other dimensions,
        semantic axes, declared units, coordinates and feature names stay exact.
        A runtime can persist this JSON-compatible record at fit time and compare
        it with a new source before replay. No unit conversion or coordinate
        interpolation is implied by equality of tensor shapes.
        """
        descriptor = self.descriptor(source_id)
        descriptor["shape"][0] = None
        descriptor["native_representation"]["axes"][0]["size"] = None
        return descriptor


class MultimodalDataset:
    """Named, sample-aligned raw sources with explicit targets and partitions.

    Sources may be dense ``TensorSource`` instances or ``RaggedSeriesSource``
    instances with packed variable-length series. Ragged sources retain their
    typed batch through ``source_values``; selection never pads their values.

    With default ``source_alignment='strict'``, every source must have the same
    set of sample IDs. Assembly reorders source rows to ``sample_ids`` and
    rejects missing, extra or duplicated identities. Explicit ``'left'`` allows
    each source to contain a subset of the canonical IDs, including zero rows
    when its non-sample shape is known. Extra IDs are always errors. Missing
    source rows receive dtype-preserving placeholders (zero, empty string or
    None for object cells) and presence=False. These are storage placeholders,
    never imputed measurements. Existing rows retain their values and presence.
    ``partitions`` is an aligned sequence of ``train``, ``test`` or ``predict``;
    omitting it assigns every row to ``train``. Groups may repeat within a
    partition but cannot cross a partition boundary. This container never
    chooses train/test membership, constructs folds, or fits transformations.

    ``y`` retains its sample-major rank: ``(n_samples,)`` or
    ``(n_samples, n_targets)``. ``target_names`` defaults to ``("y",)`` for
    one target and ``("y0", "y1", ...)`` for multiple targets. A boolean
    ``target_mask`` has exactly the same shape as ``y``: ``True`` means an
    observed label, matching dag-ml-data's target validity-mask convention.
    None/NaN/Inf labels require an explicit mask and cannot be marked observed.
    False may also deliberately withhold a finite label; its value is preserved.
    Complete targets default to an all-True mask. Masking never imputes labels,
    drops rows, or grants a downstream estimator support for partial targets.
    ``y=None`` denotes absent targets; names may still declare the prediction
    outputs, while ``target_mask`` must remain None. Values and masks follow the
    canonical ``sample_ids`` order; target names follow the target column order.
    ``task_type`` optionally declares regression or classification, including
    for prediction-only inputs. Assembly preserves this declaration without
    inferring a task or encoding labels.
    """

    def __init__(
        self,
        sources: Mapping[str, TensorSource | RaggedSeriesSource],
        *,
        sample_ids: Sequence[str],
        y: Any = None,
        target_names: Sequence[str] | None = None,
        target_mask: Any = None,
        task_type: Literal["regression", "classification"] | None = None,
        source_alignment: Literal["strict", "left"] = "strict",
        groups: Any = None,
        partitions: Sequence[str] | np.ndarray | None = None,
        name: str = "multimodal",
    ) -> None:
        ids = _identities(sample_ids, "dataset sample_ids")
        if not sources:
            raise ValueError("At least one named source is required")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string")
        if task_type is not None and (not isinstance(task_type, str) or task_type not in ("regression", "classification")):
            raise ValueError("task_type must be 'regression', 'classification' or None")
        if not isinstance(source_alignment, str) or source_alignment not in ("strict", "left"):
            raise ValueError("source_alignment must be 'strict' or 'left'")
        expected = set(ids)
        cohort_positions = {sample_id: index for index, sample_id in enumerate(ids)}
        aligned: dict[str, TensorSource | RaggedSeriesSource] = {}
        for source_id, source in sources.items():
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError("Source names must be non-empty strings")
            if not isinstance(source, (TensorSource, RaggedSeriesSource)):
                raise TypeError(f"Source {source_id!r} must be a TensorSource or RaggedSeriesSource with explicit sample_ids")
            actual = set(source.sample_ids)
            if actual - expected or (source_alignment == "strict" and actual != expected):
                raise ValueError(f"Source {source_id!r} sample IDs do not match the cohort: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
            if actual == expected:
                aligned[source_id] = source if tuple(source.sample_ids) == ids else source.take(ids)
            elif isinstance(source, RaggedSeriesSource):
                aligned[source_id] = source._aligned(ids, allow_missing=True)
            else:
                values = np.zeros((len(ids), *source.values.shape[1:]), dtype=source.values.dtype)
                if values.dtype.kind == "O":
                    values.fill(None)
                presence = np.zeros(len(ids), dtype=bool)
                positions = np.asarray([cohort_positions[sample_id] for sample_id in source.sample_ids], dtype=np.intp)
                values[positions] = source.values
                presence[positions] = np.asarray(source.presence_mask)
                aligned[source_id] = TensorSource(
                    values, ids, representation_id=source.representation_id,
                    axes=source.axes, feature_names=source.feature_names,
                    axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
                    presence_mask=presence,
                )
        if np.ma.isMaskedArray(y) or np.ma.isMaskedArray(target_mask):
            raise ValueError("Use plain y values and an explicit target_mask instead of NumPy masked arrays")
        target = None if y is None else _readonly(y)
        if target is not None and target.dtype.kind == "U" and isinstance(y, (list, tuple)):
            original_cells = np.asarray(y, dtype=object)
            if any(not isinstance(value, str) for value in original_cells.flat):
                # Preserve numeric/missing cells beside categorical strings;
                # otherwise np.asarray(["A", np.nan]) erases missingness.
                target = _readonly(original_cells)
        if target is not None and (target.ndim not in (1, 2) or target.shape[0] != len(ids)):
            raise ValueError("y must be a rank-1 or rank-2 array aligned to dataset sample_ids")
        target_count = 0 if target is None else 1 if target.ndim == 1 else target.shape[1]
        if target is not None and target_count == 0:
            raise ValueError("y must contain at least one target column")
        if target_names is None:
            names = ("y",) if target_count == 1 else tuple(f"y{index}" for index in range(target_count))
        else:
            if isinstance(target_names, (str, bytes)):
                raise ValueError("target_names must be a sequence of non-empty unique strings")
            names = tuple(target_names)
            if any(not isinstance(value, str) or not value.strip() for value in names) or len(set(names)) != len(names):
                raise ValueError("target_names must contain non-empty unique strings")
        if target is not None and len(names) != target_count:
            raise ValueError(f"target_names has {len(names)} entries but y has {target_count} targets")
        mask = None
        if target is None:
            if target_mask is not None:
                raise ValueError("target_mask must be None when y is absent")
        else:
            if target.dtype.kind not in "biufUO":
                raise ValueError("y must contain real numeric, boolean or string labels, with None permitted for missing labels")
            missing = np.zeros(target.shape, dtype=bool)
            if target.dtype.kind == "f":
                missing = ~np.isfinite(target)
            elif target.dtype.kind == "O":
                for position, raw_value in np.ndenumerate(target):
                    value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                    if value is not None and type(value) not in (bool, int, float, str):
                        raise ValueError("y cells must be scalar numbers, booleans, strings or None")
                    missing[position] = value is None or (type(value) is float and not isfinite(value))
            if target_mask is None:
                if missing.any():
                    raise ValueError("An explicit target_mask is required when y contains None or non-finite labels")
                mask = _readonly(np.ones(target.shape, dtype=bool))
            else:
                mask = _readonly(target_mask)
                if mask.dtype.kind != "b" or mask.shape != target.shape:
                    raise ValueError("target_mask must have boolean dtype and exactly y.shape; broadcasting is not allowed")
                if np.any(mask & missing):
                    raise ValueError("target_mask marks a missing or non-finite label as observed (True)")
        partition_values = _readonly(["train"] * len(ids) if partitions is None else partitions)
        if partition_values.ndim != 1 or len(partition_values) != len(ids):
            raise ValueError("partitions must have one label per dataset sample ID")
        if any(value not in ("train", "test", "predict") for value in partition_values):
            raise ValueError("partitions may contain only 'train', 'test' or 'predict'")
        group_values = None if groups is None else _readonly(groups)
        if group_values is not None:
            if group_values.ndim != 1 or len(group_values) != len(ids):
                raise ValueError("groups must have one group ID per dataset sample ID")
            ownership: dict[Any, str] = {}
            for group, partition in zip(group_values, partition_values, strict=True):
                if not isinstance(group, (str, int, float, np.integer, np.floating)) or isinstance(group, (bool, np.bool_)):
                    raise ValueError("Group IDs must be non-empty strings or finite numbers")
                if (isinstance(group, str) and not group.strip()) or (not isinstance(group, str) and not np.isfinite(group)):
                    raise ValueError("Group IDs must be non-empty strings or finite numbers")
                previous = ownership.setdefault(group, str(partition))
                if previous != partition:
                    raise ValueError(f"Group {group!r} crosses partitions {previous!r} and {str(partition)!r}")
        self.sources: Mapping[str, TensorSource | RaggedSeriesSource] = MappingProxyType(aligned)
        self.sample_ids = ids
        self.y = target
        self.target_names = names
        self.target_mask = mask
        self.task_type = task_type
        self.source_alignment = source_alignment
        self.groups = group_values
        self.partitions = partition_values
        self.name = name

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Persist host inputs without serializing the read-only mapping proxy."""
        # NumPy's older pickle protocols normalize byte order. Carry the target
        # dtype separately so reconstruction preserves the declared contract.
        return (_restore_dataset, (
            dict(self.sources), self.sample_ids, self.y, self.groups, self.partitions, self.name,
            self.target_names, self.target_mask, self.task_type, None if self.y is None else str(self.y.dtype),
            self.source_alignment,
        ))

    def source_values(
        self, row_indices: Sequence[int] | None = None, *, source_names: Sequence[str] | None = None,
    ) -> list[np.ndarray | RaggedSeriesBatch]:
        """Return read-only source blocks in canonical or explicitly requested order.

        ``source_names`` selects a unique ordered subset before reading any row
        buffers. Unselected modalities are not materialized. Positional row
        selection preserves order and duplicate rows; ``take`` selects by ID.
        """
        if source_names is None:
            names = tuple(self.sources)
        else:
            if isinstance(source_names, (str, bytes)):
                raise ValueError("source_names must be a sequence of names, not a string")
            names = tuple(source_names)
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("source_names must contain non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError("source_names contains duplicate names")
            unknown = set(names) - self.sources.keys()
            if unknown:
                raise ValueError(f"Unknown source names: {sorted(unknown)}")
        if row_indices is None:
            return [self.sources[name].values for name in names]
        indices = _row_positions(row_indices, len(self))
        blocks: list[np.ndarray | RaggedSeriesBatch] = []
        for name in names:
            source = self.sources[name]
            if isinstance(source, RaggedSeriesSource):
                blocks.append(source.values.take_rows(indices))
            else:
                # Integer advanced indexing already owns a new array; do not
                # copy large image blocks a second time just to freeze them.
                selected = source.values[indices]
                selected.setflags(write=False)
                blocks.append(selected)
        return blocks

    def source_presence(self, rows: Sequence[int] | None = None) -> dict[str, np.ndarray]:
        """Return read-only modality-presence flags in canonical sample order.

        ``rows`` selects positional rows like ``source_values``; use ``take``
        for selection by ID. Each vector corresponds to dag-ml-data's
        ``PresenceMask.present`` for this source and these sample IDs. These
        flags describe whole modalities, separately from target validity.
        """
        if rows is None:
            return {name: np.asarray(source.presence_mask) for name, source in self.sources.items()}
        indices = _row_positions(rows, len(self))
        return {name: _readonly(np.asarray(source.presence_mask)[indices]) for name, source in self.sources.items()}

    def take(self, sample_ids: Sequence[str]) -> MultimodalDataset:
        """Build a cohort subset by IDs without fitting or changing raw values."""
        ids = _identities(sample_ids, "selection sample_ids")
        lookup = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        missing = set(ids) - lookup.keys()
        if missing:
            raise ValueError(f"Unknown sample IDs in dataset selection: {sorted(missing)}")
        positions = np.asarray([lookup[sample_id] for sample_id in ids], dtype=np.intp)
        return MultimodalDataset(
            {name: source.take(ids) for name, source in self.sources.items()},
            sample_ids=ids,
            y=None if self.y is None else self.y[positions],
            target_names=self.target_names,
            target_mask=None if self.target_mask is None else self.target_mask[positions],
            task_type=self.task_type,
            source_alignment=self.source_alignment,
            groups=None if self.groups is None else self.groups[positions],
            partitions=self.partitions[positions],
            name=self.name,
        )

    def descriptors(self) -> list[dict[str, Any]]:
        """Return value-free source descriptions in the source traversal order."""
        return [source.descriptor(name) for name, source in self.sources.items()]

    def schema_descriptors(self) -> list[dict[str, Any]]:
        """Return source replay contracts with sample-axis sizes left unbound."""
        return [source.schema_descriptor(name) for name, source in self.sources.items()]

    def target_descriptor(self) -> dict[str, Any]:
        """Describe target axes and mask semantics without exposing label values.

        The host container keeps sample-major arrays. Transposing them to native
        target-major blocks is a runtime adapter's responsibility. A missing
        payload remains absent even when its output names have been declared.
        """
        return {
            "target_names": list(self.target_names),
            "task_type": self.task_type,
            "layout": "sample_major",
            "axes": None if self.y is None else ["sample"] if self.y.ndim == 1 else ["sample", "target"],
            "shape": None if self.y is None else list(self.y.shape),
            "dtype": None if self.y is None else str(self.y.dtype),
            "target_mask": None if self.target_mask is None else {
                "shape": list(self.target_mask.shape), "dtype": "bool", "true_means": "observed",
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned, closed JSON representation of this raw cohort.

        Buffers use nested arrays plus explicit dtype and shape; source order,
        identity and all declared axis information are retained. Non-finite
        numeric cells use ``{"nonfinite": "nan" | "inf" | "-inf"}`` so callers
        can write standards-compliant JSON with ``json.dumps(..., allow_nan=False)``.
        Bytes and extended-precision floating dtypes are not supported by this
        JSON format and raise ``ValueError`` without decoding or rounding them.
        This is an in-memory interchange surface, not a file parser or an ML
        artifact. No source is encoded, imputed or flattened during export.
        Ragged source records use ``source_kind='ragged_series'`` and carry
        packed values, offsets and optional per-point time coordinates. Dense
        source records and their existing version-1 fingerprints stay unchanged.
        """
        return {
            "schema": _JSON_SCHEMA,
            "schema_version": _JSON_SCHEMA_VERSION,
            "name": self.name,
            "sample_ids": list(self.sample_ids),
            "source_alignment": self.source_alignment,
            "sources": [
                source._to_record(name) if isinstance(source, RaggedSeriesSource) else {
                    "name": name, "sample_ids": list(source.sample_ids),
                    "representation_id": source.representation_id,
                    "axes": list(source.axes or ()),
                    "feature_names": None if source.feature_names is None else list(source.feature_names),
                    "axis_units": dict(source.axis_units or {}),
                    "axis_coordinates": {axis: list(values) for axis, values in (source.axis_coordinates or {}).items()},
                    "array": _array_to_dict(source.values),
                    "presence_mask": _array_to_dict(np.asarray(source.presence_mask)),
                }
                for name, source in self.sources.items()
            ],
            "y": None if self.y is None else _array_to_dict(self.y),
            "target_names": list(self.target_names),
            "target_mask": None if self.target_mask is None else _array_to_dict(self.target_mask),
            "task_type": self.task_type,
            "groups": None if self.groups is None else _array_to_dict(self.groups),
            "partitions": _array_to_dict(self.partitions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MultimodalDataset:
        """Restore the declared JSON cohort through the normal validation path.

        Unknown fields, unsupported schema versions, shape/dtype disagreement,
        malformed identities and invalid axis/group contracts are errors. Values
        are never truncated to fit an integer/string dtype, sources are never
        matched by row position. Only explicit left alignment creates missing
        source rows, paired with False presence flags and storage placeholders.

        Version 1 permits omitted ``target_names``, ``target_mask`` and
        ``task_type`` with the same defaults as the constructor. A mask is still mandatory when labels
        contain missing values; loading never infers which labels to withhold.
        Omitted ``source_alignment`` defaults to strict; omitted source
        ``presence_mask`` marks all its supplied rows present.
        Explicit ragged records are validated with ``RaggedSeriesSource``;
        their variable lengths never relax the existing dense source contract.
        """
        record = _closed_fields(
            payload, {"schema", "schema_version", "name", "sample_ids", "sources", "y", "groups", "partitions"},
            "multimodal dataset", optional=frozenset({"target_names", "target_mask", "task_type", "source_alignment"}),
        )
        if record["schema"] != _JSON_SCHEMA or type(record["schema_version"]) is not int or record["schema_version"] != _JSON_SCHEMA_VERSION:
            raise ValueError("Unsupported multimodal dataset schema or schema_version; expected nirs4all.multimodal-dataset version 1")
        if not isinstance(record["sample_ids"], list):
            raise ValueError("multimodal dataset.sample_ids must be an array")
        if record.get("target_names") is not None and not isinstance(record["target_names"], list):
            raise ValueError("multimodal dataset.target_names must be an array or null")
        source_records = record["sources"]
        if not isinstance(source_records, list) or not source_records:
            raise ValueError("multimodal dataset.sources must be a non-empty array")
        sources: dict[str, TensorSource | RaggedSeriesSource] = {}
        fields = {"name", "sample_ids", "representation_id", "axes", "feature_names", "axis_units", "axis_coordinates", "array"}
        for index, raw_source in enumerate(source_records):
            if isinstance(raw_source, Mapping) and raw_source.get("source_kind") == "ragged_series":
                ragged = RaggedSeriesSource._from_record(raw_source)
                name = raw_source["name"]
                if not isinstance(name, str) or not name.strip() or name in sources:
                    raise ValueError("Multimodal JSON source names must be non-empty and unique")
                sources[name] = ragged
                continue
            source = _closed_fields(raw_source, fields, f"sources[{index}]", optional=frozenset({"presence_mask"}))
            name = source["name"]
            if not isinstance(name, str) or not name.strip() or name in sources:
                raise ValueError("Multimodal JSON source names must be non-empty and unique")
            if not isinstance(source["sample_ids"], list):
                raise ValueError(f"Source {name!r}.sample_ids must be an array")
            sources[name] = TensorSource(
                _array_from_dict(source["array"], f"source {name!r}"), source["sample_ids"],
                representation_id=source["representation_id"], axes=source["axes"],
                feature_names=source["feature_names"], axis_units=source["axis_units"], axis_coordinates=source["axis_coordinates"],
                presence_mask=None if source.get("presence_mask") is None else _array_from_dict(source["presence_mask"], f"source {name!r}.presence_mask"),
            )
        return cls(
            sources, sample_ids=record["sample_ids"], name=record["name"],
            y=None if record["y"] is None else _array_from_dict(record["y"], "y"),
            target_names=record.get("target_names"),
            target_mask=None if record.get("target_mask") is None else _array_from_dict(record["target_mask"], "target_mask"),
            task_type=record.get("task_type"),
            source_alignment=record.get("source_alignment", "strict"),
            groups=None if record["groups"] is None else _array_from_dict(record["groups"], "groups"),
            partitions=_array_from_dict(record["partitions"], "partitions"),
        )


def _restore_tensor_source(
    values: np.ndarray,
    sample_ids: Sequence[str],
    representation_id: str,
    axes: Sequence[str],
    feature_names: Sequence[str] | None,
    axis_units: Mapping[str, str | None],
    axis_coordinates: Mapping[str, Sequence[Any]],
    presence_mask: np.ndarray,
    dtype: str,
) -> TensorSource:
    return TensorSource(
        np.asarray(values, dtype=dtype), sample_ids, representation_id=representation_id, axes=axes,
        feature_names=feature_names, axis_units=axis_units, axis_coordinates=axis_coordinates,
        presence_mask=presence_mask,
    )


def _restore_dataset(
    sources: Mapping[str, TensorSource | RaggedSeriesSource], sample_ids: Sequence[str], y: Any, groups: Any, partitions: np.ndarray, name: str,
    target_names: Sequence[str], target_mask: np.ndarray | None, task_type: Literal["regression", "classification"] | None, target_dtype: str | None,
    source_alignment: Literal["strict", "left"],
) -> MultimodalDataset:
    return MultimodalDataset(
        sources, sample_ids=sample_ids, y=None if y is None else np.asarray(y, dtype=target_dtype), groups=groups, partitions=partitions, name=name,
        target_names=target_names, target_mask=target_mask, task_type=task_type,
        source_alignment=source_alignment,
    )


__all__ = ["MultimodalDataset", "TensorSource", "RaggedSeriesBatch", "RaggedSeriesSource"]
