# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""nirs4all-io — dataset-assembly bridge (Rust core, pyo3 binding).

The JSON surface (``infer`` / ``to_spec`` / ``validate``) comes straight from the
native extension. ``load`` wraps the materialized exports: ``target="assembled"``
returns the structural summary dict; ``target="spectrodataset"`` builds a real
nirs4all ``SpectroDataset`` via the lazy adapter (the only nirs4all touch-point).
"""
from __future__ import annotations

from typing import Any

from ._adapter import to_spectrodataset
from ._native import (
    __version__,
    assembled_full,
    infer,
    load_summary,
    to_spec,
    validate,
)

__all__ = ["infer", "to_spec", "validate", "load", "to_spectrodataset", "__version__"]


def load(
    input: Any,
    *,
    target: str = "assembled",
    conventions: list[str] | None = None,
    name: str | None = None,
    spectro_dataset_cls: type | None = None,
) -> Any:
    """Materialize ``input``.

    ``target="assembled"`` → the rounded structural summary dict.
    ``target="spectrodataset"`` → a nirs4all ``SpectroDataset`` (lazy import; a
    recording double can be injected via ``spectro_dataset_cls`` for testing
    without nirs4all).
    """
    if target == "assembled":
        return load_summary(input, conventions, name)
    if target == "spectrodataset":
        full = assembled_full(input, conventions, name)
        return to_spectrodataset(full, spectro_dataset_cls=spectro_dataset_cls)
    raise ValueError(f"unknown target {target!r}; expected 'assembled' | 'spectrodataset'")
