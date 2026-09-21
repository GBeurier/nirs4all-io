# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Data-only framework adapters over an already materialized finite cohort.

``SklearnProviderAdapter`` supplies arrays for ordinary ``fit(X, y)`` or batches
for an estimator's actual ``partial_fit`` method. It never fits an estimator.
Select a source explicitly for a 2-D matrix; otherwise X remains a source dict.

Torch classes are imported lazily and return source dictionaries without changing
tensor axes or encoding categories. Use ``collate_provider_samples`` with mixed
numeric/categorical sources: numeric arrays become tensors, mixed arrays remain
Python rows. Labels are never encoded. ``return_metadata=True`` preserves IDs,
groups, partitions and masks; tuple mode refuses to discard missingness masks.

Adapters capture the materialized cohort, not the generator callback. Iterable
workers shard sample positions without duplication. Their order can depend on
the worker count; DataLoader prefetch state and mid-epoch resume are not provided.
ProviderBatches owns the independent provider-cursor checkpoint contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from operator import index as integer_index
from typing import TYPE_CHECKING, Any

import numpy as np

from .multimodal import MultimodalDataset


def _cohort(data: Any, sample_ids: Sequence[str] | None = None) -> MultimodalDataset:
    cohort = data if isinstance(data, MultimodalDataset) else getattr(data, "cohort", None)
    if not isinstance(cohort, MultimodalDataset):
        raise TypeError("Expected a MultimodalDataset or a materialized DataProvider")
    return cohort if sample_ids is None else cohort.take(sample_ids)


def _require_complete(cohort: MultimodalDataset, sources: Sequence[str]) -> None:
    if any(not np.all(cohort.sources[name].presence_mask) for name in sources):
        raise ValueError("Missing sources require return_metadata=True to preserve source masks")
    if cohort.target_mask is not None and not np.all(cohort.target_mask):
        raise ValueError("Missing targets require return_metadata=True to preserve target_mask")


class SklearnProviderAdapter:
    """Expose explicit ID selections as arrays, without generation or training.

    ``source=None`` returns a dict for an estimator accepting named sources.
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
        cohort = _cohort(self.cohort, sample_ids)
        names = [self.source] if self.source is not None else list(cohort.sources)
        if not return_metadata:
            _require_complete(cohort, names)
        values = {name: np.array(cohort.sources[name].values, copy=True) for name in names}
        x = values[self.source] if self.source is not None else values
        y = None if cohort.y is None else np.array(cohort.y, copy=True)
        if not return_metadata:
            return x, y
        return {
            "X": x, "y": y, "sample_ids": cohort.sample_ids,
            "source_masks": {name: np.array(cohort.sources[name].presence_mask, copy=True) for name in names},
            "target_mask": None if cohort.target_mask is None else np.array(cohort.target_mask, copy=True),
            "target_names": cohort.target_names, "task_type": cohort.task_type,
            "groups": None if cohort.groups is None else np.array(cohort.groups, copy=True),
            "partitions": np.array(cohort.partitions, copy=True),
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
        cohort = _cohort(self.cohort, sample_ids)
        if type(start) is not int or not 0 <= start <= len(cohort):
            raise ValueError("start must be an integer sample offset within the selection")
        for offset in range(start, len(cohort), batch_size):
            ids = cohort.sample_ids[offset:offset + batch_size]
            if drop_last and len(ids) < batch_size:
                break
            yield self.arrays(ids, return_metadata=return_metadata)


class _TorchSamples:
    """Implementation shared by the lazily constructed real Torch subclasses."""

    def __init__(
        self, data: Any, *, sample_ids: Sequence[str] | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.cohort = _cohort(data, sample_ids)
        self.return_metadata = return_metadata
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
        x = {name: np.array(source.values[position], copy=True) for name, source in cohort.sources.items()}
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
    """Collate numeric arrays into Torch tensors while retaining mixed values.

    Pass as ``DataLoader(..., collate_fn=collate_provider_samples)``. String
    labels and mixed metadata remain Python values for caller-owned encoders.
    Missing targets are retained with their mask, never imputed or dropped.
    """
    from torch.utils.data import default_collate

    if not samples:
        raise ValueError("Cannot collate an empty list of provider samples")
    first = samples[0]
    if isinstance(first, dict):
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


__all__ = ["SklearnProviderAdapter", "TorchMapDataset", "TorchIterableDataset", "collate_provider_samples"]
