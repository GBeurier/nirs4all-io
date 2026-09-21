# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Packed, immutable time-series payloads with explicit sample boundaries.

No padding, truncation, interpolation or statistical encoding happens here.
The existing dense TensorSource contract remains separate from this profile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RaggedSeriesBatch:
    """Packed rows of variable-length, fixed-channel numeric series.

    ``values`` has shape ``(total_points, channels)``. Integral ``offsets``
    starts at zero, ends at ``total_points`` and has one boundary per sample
    plus the final boundary. Repeated boundaries denote empty series. Offsets
    normalize exactly to int64; values/time coordinates retain their dtype.
    Optional flat time coordinates match all points and increase strictly
    within each sample, without sorting or converting units.

    All buffers are copied and read-only. ``take_rows`` projects sample rows;
    integer indexing returns one raw ``(length, channels)`` array. Implicit
    NumPy coercion is refused so an encoder must explicitly support this type.
    """

    values: np.ndarray = field(repr=False)
    offsets: np.ndarray
    time_coordinates: np.ndarray | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        from .multimodal import _readonly

        if any(np.ma.isMaskedArray(value) for value in (self.values, self.offsets, self.time_coordinates)):
            raise ValueError("Ragged series require plain arrays and explicit source presence, not NumPy masked arrays")
        values = _readonly(self.values)
        if values.ndim != 2 or values.shape[1] == 0 or values.dtype.kind not in "biuf":
            raise ValueError("Ragged values must be a real numeric rank-2 array with fixed nonempty channels")
        offsets = np.asarray(self.offsets)
        if offsets.ndim != 1 or offsets.size == 0 or offsets.dtype.kind not in "iu":
            raise ValueError("Ragged offsets must be a nonempty one-dimensional integer array")
        if offsets[0] != 0 or offsets[-1] != values.shape[0] or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Ragged offsets must start at zero, be nondecreasing and end at the number of points")
        if np.any(offsets > np.iinfo(np.int64).max):
            raise ValueError("Ragged offsets exceed int64 storage")
        offsets = _readonly(offsets.astype(np.int64))
        coordinates = None
        if self.time_coordinates is not None:
            coordinates = _readonly(self.time_coordinates)
            if coordinates.ndim != 1 or coordinates.shape[0] != values.shape[0] or coordinates.dtype.kind not in "iuf":
                raise ValueError("Ragged time_coordinates must be a real numeric vector with one coordinate per point")
            if not np.isfinite(coordinates).all():
                raise ValueError("Ragged time_coordinates must be finite")
            for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
                times = coordinates[begin:end]
                if np.any(times[1:] <= times[:-1]):
                    raise ValueError("Ragged time_coordinates must be strictly increasing within each sample")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "time_coordinates", coordinates)

    def __len__(self) -> int:
        return self.offsets.size - 1

    @property
    def shape(self) -> tuple[int, None, int]:
        return len(self), None, self.values.shape[1]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @property
    def lengths(self) -> np.ndarray:
        lengths = np.diff(self.offsets)
        lengths.setflags(write=False)
        return lengths

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        raise TypeError("RaggedSeriesBatch cannot be coerced to a dense array; use an explicit ragged-series encoder")

    def __getitem__(self, index: int) -> np.ndarray:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("RaggedSeriesBatch indexing requires an integer; use take_rows for sample selections")
        if index < 0 or index >= len(self):
            raise IndexError("RaggedSeriesBatch sample index out of range")
        return self.values[self.offsets[index]:self.offsets[index + 1]]

    def take_rows(self, rows: Sequence[int] | np.ndarray) -> RaggedSeriesBatch:
        """Project integer or exact-length boolean sample positions in order."""
        from .multimodal import _row_positions

        if np.ma.isMaskedArray(rows):
            raise ValueError("Ragged row selection cannot be a NumPy masked array")
        selected = np.asarray(rows)
        if selected.dtype.kind == "b":
            if selected.shape != (len(self),):
                raise ValueError("Boolean ragged row selection must match the sample count")
            indices = np.flatnonzero(selected)
        else:
            indices = _row_positions(selected.tolist(), len(self))
        return self._project([int(index) for index in indices])

    def _project(self, rows: Sequence[int | None]) -> RaggedSeriesBatch:
        """Project known rows; None inserts an empty row for explicit alignment."""
        lengths = [0 if row is None else int(self.offsets[row + 1] - self.offsets[row]) for row in rows]
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
        arrays = [self[row] for row in rows if row is not None]
        values = np.concatenate(arrays, axis=0, dtype=self.dtype) if arrays else np.empty((0, self.values.shape[1]), dtype=self.dtype)
        coordinates = None
        if self.time_coordinates is not None:
            times = [self.time_coordinates[self.offsets[row]:self.offsets[row + 1]] for row in rows if row is not None]
            coordinates = np.concatenate(times, dtype=self.time_coordinates.dtype) if times else np.empty(0, dtype=self.time_coordinates.dtype)
        return RaggedSeriesBatch(values, offsets, time_coordinates=coordinates)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _restore_batch, (
            self.values, self.offsets, self.time_coordinates, str(self.dtype),
            None if self.time_coordinates is None else str(self.time_coordinates.dtype),
        )


@dataclass(frozen=True, init=False)
class RaggedSeriesSource:
    """A sample-identified ``series_mv`` source with variable time lengths.

    Values are packed in a :class:`RaggedSeriesBatch` exposed as ``.values``.
    Present samples must have at least one point and finite measurements.
    Absent samples may have zero points or retain opaque hidden measurements;
    ``presence_mask`` remains independent from sequence length and target masks.
    Channel names and time units are optional declarations, never inferred.
    """

    values: RaggedSeriesBatch = field(repr=False)
    sample_ids: tuple[str, ...]
    channel_names: tuple[str, ...] | None
    time_unit: str | None
    presence_mask: np.ndarray = field(repr=False)
    representation_id = "series_mv"
    axes = ("sample", "time", "variable")

    def __init__(
        self, values: Any, offsets: Any, sample_ids: Sequence[str], *,
        time_coordinates: Any = None, channel_names: Sequence[str] | None = None,
        time_unit: str | None = None, presence_mask: Any = None,
    ) -> None:
        from .multimodal import _identities, _readonly

        ids = _identities(sample_ids, "source sample_ids")
        batch = RaggedSeriesBatch(values, offsets, time_coordinates=time_coordinates)
        if len(batch) != len(ids):
            raise ValueError("Ragged offsets must provide exactly one series per source sample ID")
        if np.ma.isMaskedArray(presence_mask):
            raise ValueError("presence_mask must be an explicit boolean array, not a NumPy masked array")
        presence = _readonly(np.ones(len(ids), dtype=bool) if presence_mask is None else presence_mask)
        if presence.dtype.kind != "b" or presence.shape != (len(ids),):
            raise ValueError("presence_mask must have boolean dtype and one dimension matching source sample_ids")
        if np.any(presence & (batch.lengths == 0)):
            raise ValueError("Present ragged series must contain at least one point; declare absent samples explicitly")
        for index in np.flatnonzero(presence):
            if not np.isfinite(batch[int(index)]).all():
                raise ValueError("Present ragged series values must be finite; per-cell missingness is not supported")
        names = None if channel_names is None else _identities(channel_names, "channel_names")
        if names is not None and len(names) != batch.values.shape[1]:
            raise ValueError("channel_names must match the fixed number of ragged channels")
        if time_unit is not None and (not isinstance(time_unit, str) or not time_unit.strip()):
            raise ValueError("time_unit must be a non-empty string or None")
        object.__setattr__(self, "values", batch)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "presence_mask", presence)

    @property
    def offsets(self) -> np.ndarray:
        return self.values.offsets

    @property
    def lengths(self) -> np.ndarray:
        return self.values.lengths

    @property
    def time_coordinates(self) -> np.ndarray | None:
        return self.values.time_coordinates

    @property
    def axis_units(self) -> Mapping[str, str | None]:
        return MappingProxyType({"time": self.time_unit, "variable": None})

    @property
    def axis_coordinates(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({} if self.channel_names is None else {"variable": self.channel_names})

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Declared channel names; this does not imply a flattened feature width."""
        return self.channel_names

    def take(self, sample_ids: Sequence[str]) -> RaggedSeriesSource:
        """Select unique sample IDs without flattening or padding sequences."""
        return self._aligned(sample_ids, allow_missing=False)

    def _aligned(self, sample_ids: Sequence[str], *, allow_missing: bool) -> RaggedSeriesSource:
        from .multimodal import _identities

        ids = _identities(sample_ids, "selection sample_ids")
        lookup = {sample_id: row for row, sample_id in enumerate(self.sample_ids)}
        missing = set(ids) - lookup.keys()
        if missing and not allow_missing:
            raise ValueError(f"Unknown sample IDs in source selection: {sorted(missing)}")
        rows = [lookup.get(sample_id) for sample_id in ids]
        batch = self.values._project(rows)
        return RaggedSeriesSource(
            batch.values, batch.offsets, ids, time_coordinates=batch.time_coordinates,
            channel_names=self.channel_names, time_unit=self.time_unit,
            presence_mask=np.asarray([False if row is None else bool(self.presence_mask[row]) for row in rows], dtype=bool),
        )

    def descriptor(self, source_id: str) -> dict[str, Any]:
        """Describe a variable time axis without exposing its concrete lengths."""
        return {
            "source_id": source_id, "representation_id": self.representation_id,
            "type_id": "time_series", "modality": "time_series", "axes": list(self.axes),
            "shape": list(self.values.shape), "dtype": str(self.values.dtype),
            "feature_names": None if self.channel_names is None else list(self.channel_names),
            "axis_units": dict(self.axis_units),
            "axis_coordinates": {axis: list(coordinates) for axis, coordinates in self.axis_coordinates.items()},
            "time_coordinates": None if self.time_coordinates is None else {
                "dtype": str(self.time_coordinates.dtype), "ordering": "strictly_increasing_per_sample",
            },
            "native_representation": {
                "id": self.representation_id, "type_id": "time_series", "rank": 3,
                "axes": [
                    {"name": "sample", "kind": "sample", "unit": None, "size": len(self.sample_ids), "variable": False},
                    {"name": "time", "kind": "time", "unit": self.time_unit, "size": None, "variable": True},
                    {"name": "variable", "kind": "feature", "unit": None, "size": self.values.shape[2], "variable": False},
                ],
                "container": "ragged_array", "dtype": str(self.values.dtype), "sparse": False, "ragged": True,
            },
        }

    def schema_descriptor(self, source_id: str) -> dict[str, Any]:
        descriptor = self.descriptor(source_id)
        descriptor["shape"][0] = None
        descriptor["native_representation"]["axes"][0]["size"] = None
        return descriptor

    def _to_record(self, source_id: str) -> dict[str, Any]:
        from .multimodal import _array_to_dict

        return {
            "source_kind": "ragged_series", "name": source_id, "sample_ids": list(self.sample_ids),
            "representation_id": self.representation_id, "axes": list(self.axes),
            "array": _array_to_dict(self.values.values), "offsets": _array_to_dict(self.offsets),
            "time_coordinates": None if self.time_coordinates is None else _array_to_dict(self.time_coordinates),
            "channel_names": None if self.channel_names is None else list(self.channel_names),
            "time_unit": self.time_unit, "presence_mask": _array_to_dict(self.presence_mask),
        }

    @classmethod
    def _from_record(cls, record: Any) -> RaggedSeriesSource:
        from .multimodal import _array_from_dict, _closed_fields

        source = _closed_fields(record, {
            "source_kind", "name", "sample_ids", "representation_id", "axes", "array", "offsets",
            "time_coordinates", "channel_names", "time_unit", "presence_mask",
        }, "ragged series source")
        if source["source_kind"] != "ragged_series" or source["representation_id"] != "series_mv" or source["axes"] != list(cls.axes):
            raise ValueError("Ragged series require source_kind='ragged_series', series_mv and sample/time/variable axes")
        if not isinstance(source["sample_ids"], list) or (source["channel_names"] is not None and not isinstance(source["channel_names"], list)):
            raise ValueError("Ragged sample_ids and channel_names must be JSON arrays")
        return cls(
            _array_from_dict(source["array"], "ragged array"), _array_from_dict(source["offsets"], "ragged offsets"), source["sample_ids"],
            time_coordinates=None if source["time_coordinates"] is None else _array_from_dict(source["time_coordinates"], "ragged time_coordinates"),
            channel_names=source["channel_names"], time_unit=source["time_unit"],
            presence_mask=_array_from_dict(source["presence_mask"], "ragged presence_mask"),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _restore_source, (self.values, self.sample_ids, self.channel_names, self.time_unit, self.presence_mask)


def _restore_batch(values: np.ndarray, offsets: np.ndarray, coordinates: np.ndarray | None, dtype: str, time_dtype: str | None) -> RaggedSeriesBatch:
    return RaggedSeriesBatch(np.asarray(values, dtype=dtype), offsets, time_coordinates=None if coordinates is None else np.asarray(coordinates, dtype=time_dtype))


def _restore_source(batch: RaggedSeriesBatch, ids: Sequence[str], names: Sequence[str] | None, unit: str | None, presence: np.ndarray) -> RaggedSeriesSource:
    return RaggedSeriesSource(batch.values, batch.offsets, ids, time_coordinates=batch.time_coordinates, channel_names=names, time_unit=unit, presence_mask=presence)


__all__ = ["RaggedSeriesBatch", "RaggedSeriesSource"]
