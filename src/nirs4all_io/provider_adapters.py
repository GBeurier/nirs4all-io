# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Data-only framework adapters over an already materialized finite cohort.

``SklearnProviderAdapter`` supplies arrays for ordinary ``fit(X, y)`` or batches
for an estimator's actual ``partial_fit`` method. It never fits an estimator.
Select a source explicitly for a 2-D matrix; otherwise X remains a source dict.
Ragged dictionary entries retain their typed batches and time coordinates for
an explicit ragged-aware encoder.

Torch classes are imported lazily and return source dictionaries without changing
tensor axes or encoding categories. Use ``collate_provider_samples`` with mixed
numeric/categorical sources: numeric arrays become tensors, mixed arrays remain
Python rows. Labels are never encoded. ``return_metadata=True`` preserves IDs,
groups, partitions and masks; tuple mode refuses to discard missingness masks.
Torch adapters reject ragged sources by default. Explicit ``ragged_policy='packed'``
with metadata enabled preserves sequence boundaries, times and source declarations
in ``TorchRaggedSeriesBatch``; it never pads or sorts series.

Adapters capture the materialized cohort, not the generator callback. Iterable
workers shard sample positions without duplication. Their order can depend on
the worker count; DataLoader prefetch state and mid-epoch resume are not provided.
ProviderBatches owns the independent provider-cursor checkpoint contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from operator import index as integer_index
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .multimodal import MultimodalDataset, _identities
from .ragged import RaggedSeriesBatch, RaggedSeriesSource

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, eq=False)
class TorchRaggedSeriesBatch:
    """Torch tensors in sample order, produced by ``collate_provider_samples``.

    ``values`` has shape (total_points, channels); int64 ``offsets`` and
    ``lengths`` retain every sample, including absent empty series. Presence is
    independent of length: hidden measurements remain opaque behind a false
    mask. Optional times retain their original numeric dtype and units.
    Tensors own storage separate from the IO cohort. This is a packed row
    container, not a Torch RNN ``PackedSequence`` or a padded dense tensor.
    """

    values: Tensor
    offsets: Tensor
    lengths: Tensor
    time_coordinates: Tensor | None
    channel_names: tuple[str, ...] | None
    time_unit: str | None
    presence_mask: Tensor

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def pin_memory(self) -> TorchRaggedSeriesBatch:
        """Return the same structure with pinned tensors, using Torch's allocator."""
        return replace(
            self, values=self.values.pin_memory(), offsets=self.offsets.pin_memory(),
            lengths=self.lengths.pin_memory(), presence_mask=self.presence_mask.pin_memory(),
            time_coordinates=None if self.time_coordinates is None else self.time_coordinates.pin_memory(),
        )

    def to(self, device: Any, non_blocking: bool = False) -> TorchRaggedSeriesBatch:
        """Move tensors to a device without offering dtype conversion overloads."""
        from torch import device as torch_device

        destination = torch_device(device)
        return replace(
            self, values=self.values.to(device=destination, non_blocking=non_blocking),
            offsets=self.offsets.to(device=destination, non_blocking=non_blocking),
            lengths=self.lengths.to(device=destination, non_blocking=non_blocking),
            presence_mask=self.presence_mask.to(device=destination, non_blocking=non_blocking),
            time_coordinates=None if self.time_coordinates is None else self.time_coordinates.to(device=destination, non_blocking=non_blocking),
        )


def _validate_torch_ragged_dtype(source: RaggedSeriesSource, name: str) -> None:
    """Refuse unsupported storage before NumPy concatenation can change dtype."""
    import torch

    arrays = [source.values.values]
    if source.time_coordinates is not None:
        arrays.append(source.time_coordinates)
    for array in arrays:
        if not array.dtype.isnative:
            raise ValueError(f"Ragged source {name!r} dtype {array.dtype} has non-native byte order; Torch packed collation never converts dtypes")
        try:
            torch.from_numpy(np.empty(0, dtype=array.dtype))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ragged source {name!r} dtype {array.dtype} is not representable by Torch packed collation") from exc


def _cohort(data: Any, sample_ids: Sequence[str] | None = None) -> MultimodalDataset:
    cohort = data if isinstance(data, MultimodalDataset) else getattr(data, "cohort", None)
    if not isinstance(cohort, MultimodalDataset):
        raise TypeError("Expected a MultimodalDataset or a materialized DataProvider")
    return cohort if sample_ids is None else cohort.take(sample_ids)


def _require_complete(cohort: MultimodalDataset, sources: Sequence[str], positions: np.ndarray | None = None) -> None:
    for name in sources:
        source_mask = np.asarray(cohort.sources[name].presence_mask)
        if not np.all(source_mask if positions is None else source_mask[positions]):
            raise ValueError("Missing sources require return_metadata=True to preserve source masks")
    if cohort.target_mask is not None:
        target_mask = np.asarray(cohort.target_mask)
        if not np.all(target_mask if positions is None else target_mask[positions]):
            raise ValueError("Missing targets require return_metadata=True to preserve target_mask")


class SklearnProviderAdapter:
    """Expose explicit ID selections as arrays, without generation or training.

    ``source=None`` returns a dict for an estimator accepting named sources.
    Ragged entries remain ``RaggedSeriesBatch`` objects, including their times.
    ``source='nir'`` returns that source's matrix and rejects N-D flattening.
    Ordinary sklearn estimators do not generally accept a dict or an iterator.
    """

    def __init__(self, data: Any, *, source: str | None = None) -> None:
        self.cohort = _cohort(data)
        self.source = source
        if source is not None:
            if source not in self.cohort.sources:
                raise ValueError(f"Unknown source {source!r}")
            if self.cohort.sources[source].values.ndim != 2:
                raise ValueError("A sklearn matrix source must be rank 2; provide an explicit encoder for N-D sources")

    def arrays(self, sample_ids: Sequence[str] | None = None, *, return_metadata: bool = False) -> Any:
        """Return ``(X, y)`` or a dict retaining IDs and missingness metadata."""
        ids, positions = self._selection(sample_ids)
        return self._arrays(ids, positions, return_metadata=return_metadata)

    def _selection(self, sample_ids: Sequence[str] | None) -> tuple[tuple[str, ...], np.ndarray]:
        if sample_ids is None:
            return self.cohort.sample_ids, np.arange(len(self.cohort), dtype=np.intp)
        ids = _identities(sample_ids, "selection sample_ids")
        lookup = {sample_id: index for index, sample_id in enumerate(self.cohort.sample_ids)}
        missing = set(ids) - lookup.keys()
        if missing:
            raise ValueError(f"Unknown sample IDs in dataset selection: {sorted(missing)}")
        return ids, np.asarray([lookup[sample_id] for sample_id in ids], dtype=np.intp)

    def _arrays(self, ids: tuple[str, ...], positions: np.ndarray, *, return_metadata: bool) -> Any:
        cohort = self.cohort
        names = [self.source] if self.source is not None else list(cohort.sources)
        if not return_metadata:
            _require_complete(cohort, names, positions)
        values: dict[str, np.ndarray | RaggedSeriesBatch] = {}
        for name in names:
            block = cohort.sources[name].values
            # Advanced indexing owns a mutable copy; unrequested sources are
            # never selected or copied through an intermediate cohort.
            values[name] = block.take_rows(positions) if isinstance(block, RaggedSeriesBatch) else block[positions]
        x = values[self.source] if self.source is not None else values
        y = None if cohort.y is None else cohort.y[positions]
        if not return_metadata:
            return x, y
        return {
            "X": x, "y": y, "sample_ids": ids,
            "source_masks": {name: np.asarray(cohort.sources[name].presence_mask)[positions] for name in names},
            "target_mask": None if cohort.target_mask is None else cohort.target_mask[positions],
            "target_names": cohort.target_names, "task_type": cohort.task_type,
            "groups": None if cohort.groups is None else cohort.groups[positions],
            "partitions": cohort.partitions[positions],
        }

    def batches(
        self, batch_size: int, *, sample_ids: Sequence[str] | None = None,
        start: int = 0, drop_last: bool = False, return_metadata: bool = False,
    ) -> Iterator[Any]:
        """Yield arrays for caller-managed ``partial_fit``; never call ``fit``.

        ``start`` is a sample offset within the explicitly selected ID sequence.
        No cursor or estimator checkpoint is implied by this adapter iterator.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        ids, positions = self._selection(sample_ids)
        if type(start) is not int or not 0 <= start <= len(ids):
            raise ValueError("start must be an integer sample offset within the selection")
        for offset in range(start, len(ids), batch_size):
            batch_ids = ids[offset:offset + batch_size]
            if drop_last and len(batch_ids) < batch_size:
                break
            yield self._arrays(batch_ids, positions[offset:offset + batch_size], return_metadata=return_metadata)


class _TorchSamples:
    """Finite cohort view; opt-in packed ragged data requires metadata and its collator."""

    def __init__(
        self, data: Any, *, sample_ids: Sequence[str] | None = None,
        return_metadata: bool = False, ragged_policy: Literal["error", "packed"] = "error",
    ) -> None:
        if not isinstance(ragged_policy, str) or ragged_policy not in {"error", "packed"}:
            raise ValueError("ragged_policy must be 'error' or 'packed'")
        self.cohort = _cohort(data, sample_ids)
        self.return_metadata = return_metadata
        self.ragged_policy = ragged_policy
        for name, source in self.cohort.sources.items():
            if isinstance(source, RaggedSeriesSource):
                if ragged_policy == "error":
                    raise ValueError("Torch provider adapters reject ragged sources by default; use ragged_policy='packed' with typed collation to preserve time coordinates")
                if not return_metadata:
                    raise ValueError("Packed ragged sources require return_metadata=True to preserve IDs and masks")
                _validate_torch_ragged_dtype(source, name)
        if not return_metadata:
            _require_complete(self.cohort, list(self.cohort.sources))

    def __len__(self) -> int:
        return len(self.cohort)

    def _sample(self, position: int) -> Any:
        if isinstance(position, (bool, np.bool_)):
            raise TypeError("Sample position must be an integer, not bool")
        position = integer_index(position)
        if not 0 <= position < len(self):
            raise IndexError("Sample position is outside the materialized cohort")
        cohort = self.cohort
        x: dict[str, Any] = {}
        for name, source in cohort.sources.items():
            if isinstance(source, RaggedSeriesSource):
                # The source constructor owns the resulting buffers. Slice the
                # aligned storage directly, avoiding a copied intermediate batch.
                begin, end = source.offsets[position:position + 2]
                coordinates = None if source.time_coordinates is None else source.time_coordinates[begin:end]
                x[name] = RaggedSeriesSource(
                    source.values[position], [0, end - begin], [cohort.sample_ids[position]], time_coordinates=coordinates,
                    channel_names=source.channel_names, time_unit=source.time_unit,
                    presence_mask=source.presence_mask[position:position + 1],
                )
            else:
                x[name] = np.array(source.values[position], copy=True)
        y = None if cohort.y is None else np.array(cohort.y[position], copy=True)
        if not self.return_metadata:
            return x if y is None else (x, y)
        item: dict[str, Any] = {
            "X": x, "sample_id": cohort.sample_ids[position],
            "source_masks": {name: bool(np.asarray(source.presence_mask)[position]) for name, source in cohort.sources.items()},
            "partition": str(cohort.partitions[position]),
        }
        if y is not None:
            assert cohort.target_mask is not None
            item["y"] = y
            item["target_mask"] = np.array(cohort.target_mask[position], copy=True)
        if cohort.groups is not None:
            group = cohort.groups[position]
            item["group"] = group.item() if isinstance(group, np.generic) else group
        return item


class _TorchMapSamples(_TorchSamples):
    def __getitem__(self, position: int) -> Any:
        return self._sample(position)


class _TorchIterableSamples(_TorchSamples):
    def __iter__(self) -> Iterator[Any]:
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        start, step = (0, 1) if worker is None else (worker.id, worker.num_workers)
        for position in range(start, len(self), step):
            yield self._sample(position)


def collate_provider_samples(samples: list[Any]) -> Any:
    """Collate dense arrays and explicitly packed ragged sources into Torch tensors.

    Pass as ``DataLoader(..., collate_fn=collate_provider_samples)``. String
    labels and mixed metadata remain Python values for caller-owned encoders.
    Missing targets are retained with their mask, never imputed or dropped.
    Ragged entries become ``TorchRaggedSeriesBatch`` with homogeneous source
    declarations. Sample order and repetitions are retained without padding.
    """
    from torch.utils.data import default_collate

    if not samples:
        raise ValueError("Cannot collate an empty list of provider samples")
    first = samples[0]
    if isinstance(first, RaggedSeriesSource):
        import torch

        schema = first.schema_descriptor("source")
        if any(not isinstance(item, RaggedSeriesSource) or len(item.sample_ids) != 1 or item.schema_descriptor("source") != schema for item in samples):
            raise ValueError("Packed collation requires one-row ragged sources with identical channel, unit, coordinate and dtype schemas")
        _validate_torch_ragged_dtype(first, "source")
        lengths = np.asarray([item.lengths[0] for item in samples], dtype=np.int64)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
        coordinates = None if first.time_coordinates is None else torch.from_numpy(np.concatenate([item.time_coordinates for item in samples]))
        return TorchRaggedSeriesBatch(
            values=torch.from_numpy(np.concatenate([item.values.values for item in samples])),
            offsets=torch.from_numpy(offsets), lengths=torch.from_numpy(lengths),
            time_coordinates=coordinates, channel_names=first.channel_names, time_unit=first.time_unit,
            presence_mask=torch.from_numpy(np.asarray([item.presence_mask[0] for item in samples], dtype=bool)),
        )
    if isinstance(first, dict):
        if any(not isinstance(item, dict) or item.keys() != first.keys() for item in samples):
            raise ValueError("Provider samples must have identical metadata and source keys")
        if "sample_id" in first and isinstance(first.get("X"), dict):
            for item in samples:
                for name, source in item["X"].items():
                    if isinstance(source, RaggedSeriesSource) and (
                        source.sample_ids != (item["sample_id"],) or
                        item.get("source_masks", {}).get(name) != bool(source.presence_mask[0])
                    ):
                        raise ValueError("Packed source identity and presence must match the sample metadata")
        return {key: collate_provider_samples([item[key] for item in samples]) for key in first}
    if isinstance(first, tuple):
        return tuple(collate_provider_samples(list(items)) for items in zip(*samples, strict=True))
    if isinstance(first, np.ndarray) and first.dtype.kind in "OUS":
        return [item.tolist() for item in samples]
    return default_collate(samples)


def __getattr__(name: str) -> Any:
    """Import Torch only when a Torch dataset class is explicitly requested."""
    bases = {"TorchMapDataset": _TorchMapSamples, "TorchIterableDataset": _TorchIterableSamples}
    if name not in bases:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from torch.utils.data import Dataset, IterableDataset
    except ImportError as exc:
        raise ImportError(f"{name} requires the optional PyTorch dependency") from exc
    framework_base = Dataset if name == "TorchMapDataset" else IterableDataset
    # A module-level name/qualname makes the lazy class importable by spawn workers.
    dataset_class = type(name, (bases[name], framework_base), {"__module__": __name__, "__doc__": _TorchSamples.__doc__})
    globals()[name] = dataset_class
    return dataset_class


if TYPE_CHECKING:
    from torch.utils.data import Dataset, IterableDataset

    class TorchMapDataset(_TorchMapSamples, Dataset):
        """A finite map-style dataset over a materialized provider cohort."""

    class TorchIterableDataset(_TorchIterableSamples, IterableDataset):
        """A finite stream with disjoint worker shards over a materialized cohort."""


__all__ = ["SklearnProviderAdapter", "TorchMapDataset", "TorchIterableDataset", "TorchRaggedSeriesBatch", "collate_provider_samples"]
