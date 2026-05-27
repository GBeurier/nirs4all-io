# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Adapt an :class:`AssembledDataset` to a nirs4all ``SpectroDataset`` (Epic 4.1b).

This is the **sole** nirs4all touch-point: the ``SpectroDataset`` class is
imported lazily here (never at module import — see the import-boundary test).
The build flow mirrors ``DatasetConfigs._load_dataset``: construct empty, then
per partition ``add_samples({"partition": p})`` -> ``add_targets`` (appends) ->
``add_metadata``, then ``set_signal_type/task_type/folds/repetition/aggregate``.

``spectro_dataset_cls`` can be injected (a recording double) so the adapter is
testable without nirs4all installed.
"""

from __future__ import annotations

from typing import Any

from .assemble import AssembledDataset, PartitionBlock

_PARTITION_ORDER = ("train", "test", "val", "predict")


def _one_or_list(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def to_spectrodataset(assembled: AssembledDataset, *, spectro_dataset_cls: type | None = None) -> Any:
    """Materialize ``assembled`` into a SpectroDataset (lazy import unless injected)."""
    if spectro_dataset_cls is None:
        from nirs4all.data import SpectroDataset  # lazy: the only nirs4all runtime touch-point

        spectro_dataset_cls = SpectroDataset

    ds = spectro_dataset_cls(name=assembled.name)
    order = [p for p in _PARTITION_ORDER if p in assembled.blocks]
    first = True
    for part in order:
        block = assembled.blocks[part]
        if not block.X:
            continue
        nirs_partition = "test" if part == "val" else part  # nirs4all build has no 'val' partition
        ds.add_samples(
            _one_or_list(block.X),
            {"partition": nirs_partition},
            headers=_one_or_list(block.feature_headers),
            header_unit=_one_or_list(block.header_units),
        )
        if first:
            for src_idx, sig in enumerate(block.signal_types):
                if sig:
                    ds.set_signal_type(sig, src=src_idx, forced=False)
        if block.y is not None:
            ds.add_targets(block.y)
        if block.metadata is not None:
            ds.add_metadata(block.metadata)
        first = False

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
    return ds


__all__ = ["to_spectrodataset", "AssembledDataset", "PartitionBlock"]
