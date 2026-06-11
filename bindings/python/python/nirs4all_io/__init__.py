# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""nirs4all-io — dataset-assembly bridge (Rust core, pyo3 binding).

The inference surface (:func:`infer` / :func:`to_spec` / :func:`validate`) is
backed by the native extension. ``infer`` / ``to_spec`` accept native inputs — a
path (``str`` or :class:`~pathlib.Path`), a list of files, or a config ``dict`` —
and return ergonomic typed mappings (:class:`DatasetPlan` / :class:`DatasetSpec`)
that subclass ``dict`` (so they stay subscriptable, JSON-serializable, and valid
inputs to :func:`validate` / :func:`load`) while adding a readable ``repr`` and a
few convenience accessors.

:func:`load` wraps the materialized exports: ``target="assembled"`` returns the
structural summary dict; ``target="spectrodataset"`` builds a real nirs4all
``SpectroDataset`` via the lazy adapter (the only nirs4all touch-point).
"""

from __future__ import annotations

import os
from typing import Any

from ._adapter import to_spectrodataset
from ._native import __version__, assembled_full, load_summary
from ._native import infer as _native_infer
from ._native import to_spec as _native_to_spec
from ._native import validate as _native_validate

__all__ = [
    "infer",
    "to_spec",
    "validate",
    "load",
    "to_spectrodataset",
    "DatasetPlan",
    "DatasetSpec",
    "__version__",
]


def _normalize_input(input: Any) -> Any:
    """Coerce ``os.PathLike`` (and lists thereof) to plain ``str`` for the native
    call, which accepts ``str`` paths, ``list[str]`` file lists, and ``dict``
    specs. Mappings and plain strings pass through unchanged."""
    if isinstance(input, os.PathLike):
        return os.fspath(input)
    if isinstance(input, (list, tuple)):
        return [os.fspath(p) if isinstance(p, os.PathLike) else p for p in input]
    return input


class DatasetSpec(dict):
    """A canonical ``DatasetSpec`` returned by :func:`to_spec`.

    A plain ``dict`` (subscriptable, JSON-serializable, accepted by
    :func:`validate` and :func:`load`) with a readable ``repr`` and shortcut
    accessors for the common top-level fields.
    """

    __slots__ = ()

    @property
    def name(self) -> str | None:
        return self.get("name")

    @property
    def schema_version(self) -> int | None:
        return self.get("schema_version")

    @property
    def sources(self) -> list[dict[str, Any]]:
        return self.get("sources", [])

    def __repr__(self) -> str:
        srcs = self.sources
        head = ", ".join(f"{s.get('id', '?')}:{s.get('role', '?')}" for s in srcs[:4])
        if len(srcs) > 4:
            head += ", ..."
        return f"DatasetSpec(name={self.name!r}, schema_version={self.schema_version}, sources=[{head}])"


class DatasetPlan(dict):
    """A scored ``DatasetPlan`` returned by :func:`infer`.

    A plain ``dict`` with a readable ``repr``, a :attr:`resolved_spec` accessor
    (returns a :class:`DatasetSpec`), and :meth:`decisions` for the scored
    inference decisions (structure / signal type / task type / ...).
    """

    __slots__ = ()

    @property
    def overall_score(self) -> float | None:
        return self.get("overall_score")

    @property
    def resolved_spec(self) -> DatasetSpec:
        """The editable spec produced by inference, as a :class:`DatasetSpec`."""
        return DatasetSpec(self.get("resolved_spec", {}))

    @property
    def recommendations(self) -> list[str]:
        return self.get("recommendations", [])

    @property
    def warnings(self) -> list[str]:
        return self.get("warnings", [])

    def decisions(self) -> dict[str, dict[str, Any]]:
        """The scored decisions keyed by kind.

        A decision is any top-level field shaped like
        ``{value, score, evidence, alternatives, ambiguous}`` (structure,
        signal_type, task_type, ...)."""
        return {k: v for k, v in self.items() if isinstance(v, dict) and {"value", "score", "ambiguous"} <= v.keys()}

    def __repr__(self) -> str:
        decs = ", ".join(f"{k}={v.get('value')!r}({v.get('score')})" for k, v in self.decisions().items())
        return f"DatasetPlan(overall_score={self.overall_score}, {decs})"


def infer(input: Any, conventions: list[str] | None = None) -> DatasetPlan:
    """Inspect a data input and return a scored :class:`DatasetPlan`.

    Args:
        input: A path (``str`` or :class:`~pathlib.Path`) or a list of files.
        conventions: Optional convention names; ``None`` uses the default list.

    Returns:
        A :class:`DatasetPlan` (a ``dict`` subclass) with ``resolved_spec``,
        the scored decisions, ``recommendations`` and ``warnings``.
    """
    plan = _native_infer(_normalize_input(input), conventions)
    return DatasetPlan(plan)


def to_spec(input: Any, conventions: list[str] | None = None, name: str | None = None) -> DatasetSpec:
    """Normalize a data input into a canonical :class:`DatasetSpec`.

    Args:
        input: A path (``str`` or :class:`~pathlib.Path`), a list of files, or a
            config ``dict``.
        conventions: Optional convention names; ``None`` uses the default list.
        name: Optional dataset name override.

    Returns:
        A :class:`DatasetSpec` (a ``dict`` subclass) that round-trips through
        :func:`validate` and :func:`load`.
    """
    spec = _native_to_spec(_normalize_input(input), conventions, name)
    return DatasetSpec(spec)


def validate(spec: Any) -> None:
    """Validate a ``DatasetSpec`` (a mapping or a JSON string).

    Raises:
        ValueError: if the spec is malformed or fails validation.
    """
    _native_validate(spec)


def load(
    input: Any,
    *,
    target: str = "assembled",
    conventions: list[str] | None = None,
    name: str | None = None,
    spectro_dataset_cls: type | None = None,
) -> Any:
    """Materialize ``input``.

    Args:
        input: A path (``str`` or :class:`~pathlib.Path`), a list of files, or a
            config ``dict``.
        target: ``"assembled"`` → the rounded structural summary ``dict``;
            ``"spectrodataset"`` → a nirs4all ``SpectroDataset`` (lazy import).
        conventions: Optional convention names.
        name: Optional dataset name override.
        spectro_dataset_cls: A recording double can be injected here to exercise
            the ``"spectrodataset"`` adapter without nirs4all installed.

    Returns:
        The assembled summary ``dict`` or a ``SpectroDataset``.
    """
    native_input = _normalize_input(input)
    if target == "assembled":
        return load_summary(native_input, conventions, name)
    if target == "spectrodataset":
        full = assembled_full(native_input, conventions, name)
        return to_spectrodataset(full, spectro_dataset_cls=spectro_dataset_cls)
    raise ValueError(f"unknown target {target!r}; expected 'assembled' | 'spectrodataset'")
