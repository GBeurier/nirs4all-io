# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Finite host data providers, explicit assembly and resumable cohort views.

Providers generate a fixed universe once per explicit ``materialize`` call.
They do not split data, learn transformations, schedule folds, or invoke a
framework. A DAG controller can supply its task seed; standalone callers use
the provider's own seed. Callback code remains in the host and is never loaded
from a recipe or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np

from .multimodal import MultimodalDataset

_DATA_FIELDS = {
    "sources", "sample_ids", "y", "target_names", "target_mask", "task_type",
    "source_alignment", "groups", "partitions", "name",
}
_TARGET_FIELDS = {"y", "target_names", "target_mask", "task_type"}
_STATE_FIELDS = {"schema", "version", "recipe", "seed", "context", "fingerprint"}
_BATCH_FIELDS = {"schema", "version", "provider", "sample_ids", "batch_size", "drop_last", "position"}


def _json_copy(value: Any, label: str) -> Any:
    """Validate JSON without silently stringifying keys or non-finite floats."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} must have string JSON keys")
        return {key: _json_copy(item, label) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item, label) for item in value]
    raise ValueError(f"{label} must contain only finite JSON values")


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON mapping")
    result: dict[str, Any] = _json_copy(value, label)
    return result


def _unsigned(value: Any, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be a non-negative integer" + (f" no greater than {maximum}" if maximum is not None else ""))
    return value


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cohort_fingerprint(cohort: MultimodalDataset) -> str:
    # The typed JSON payload includes values, masks, source order, IDs and axes.
    # Unsupported JSON dtypes fail explicitly rather than losing precision.
    return _digest(cohort.to_dict())


def _closed_state(state: Any, *, fields: set[str], schema: str) -> dict[str, Any]:
    result = _json_mapping(state, "checkpoint")
    if set(result) != fields or result.get("schema") != schema or type(result.get("version")) is not int or result["version"] != 1:
        raise ValueError(f"Invalid {schema} checkpoint fields or version")
    return result


class DataProvider:
    """Generate and expose a finite, identity-aligned multimodal cohort.

    ``generate`` is called as ``generate(seed=..., params=..., context=...)``.
    It returns a :class:`MultimodalDataset` or a constructor-field mapping.
    Without ``base``, the mapping must describe a complete cohort. With an
    explicit fixed ``base``, it can supply only targets or selected sources;
    ``sample_ids`` is always required and must describe the same universe.
    Source rows may be absent under explicit ``source_alignment='left'``.
    Existing sources/targets can be replaced only with the corresponding
    opt-in. Groups and partitions of a base can never be changed by generation.

    Recipe params reserve ``_io_assembly`` for the fixed-base fingerprint and
    replacement rules. Callback params contain only the caller's parameters.
    Parameters/context must be finite JSON values. This initial profile is
    run-scoped, finite and unlearned. It does not promise lazy out-of-core
    generation: all indexed/batch views read the already materialized cohort.

    Checkpoints contain data identities and configuration, never executable
    callback code. To resume, construct the same provider and call
    ``load_state_dict``; it explicitly regenerates and checks the entire
    payload before making the restored cohort available.
    """

    def __init__(
        self,
        generate: Callable[..., MultimodalDataset | Mapping[str, Any]],
        *,
        provider_id: str,
        provider_version: str = "1",
        params: Mapping[str, Any] | None = None,
        seed: int = 0,
        context: Mapping[str, Any] | None = None,
        base: MultimodalDataset | None = None,
        replace_sources: Sequence[str] = (),
        replace_targets: bool = False,
    ) -> None:
        if not callable(generate):
            raise TypeError("generate must be callable")
        for label, value in (("provider_id", provider_id), ("provider_version", provider_version)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if base is not None and not isinstance(base, MultimodalDataset):
            raise TypeError("base must be a MultimodalDataset")
        if isinstance(replace_sources, (str, bytes)):
            raise ValueError("replace_sources must contain unique source names")
        replacements = tuple(replace_sources)
        if any(not isinstance(name, str) or not name.strip() for name in replacements) or len(set(replacements)) != len(replacements):
            raise ValueError("replace_sources must contain unique source names")
        if type(replace_targets) is not bool:
            raise ValueError("replace_targets must be boolean")
        if (replacements or replace_targets) and base is None:
            raise ValueError("Replacement rules require an explicit base cohort")
        if base is not None and set(replacements) - base.sources.keys():
            raise ValueError("replace_sources names must exist in the fixed base")
        self._generate = generate
        self._provider_id = provider_id
        self._provider_version = provider_version
        self._params = _json_mapping({} if params is None else params, "params")
        if "_io_assembly" in self._params:
            raise ValueError("params._io_assembly is reserved for provider assembly provenance")
        self._seed = _unsigned(seed, "seed", maximum=2**64 - 1)
        self._context = _json_mapping({} if context is None else context, "context")
        # Isolate a fixed base from changes to the caller's container/buffers.
        self._base = None if base is None else base.take(base.sample_ids)
        self._base_fingerprint = None if self._base is None else _cohort_fingerprint(self._base)
        self._replace_sources = replacements
        self._replace_targets = replace_targets
        self._cohort: MultimodalDataset | None = None
        self._state: dict[str, Any] | None = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def cohort(self) -> MultimodalDataset:
        """The materialized cohort; accessing it never invokes the callback."""
        if self._cohort is None:
            raise RuntimeError("DataProvider must be materialized before accessing its cohort")
        return self._cohort

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the complete typed cohort payload after materialization."""
        return str(self.state_dict()["fingerprint"])

    def recipe(self) -> dict[str, Any]:
        """Return the closed run-scoped recipe accepted by a DAG controller."""
        params = _json_mapping(self._params, "params")
        params["_io_assembly"] = {
            "base_fingerprint": self._base_fingerprint,
            "replace_sources": list(self._replace_sources),
            "replace_targets": self._replace_targets,
        }
        return {
            "provider_id": self.provider_id, "provider_version": self.provider_version,
            "params": params, "seed": self.seed, "scope": "run", "finite": True,
            "learned": False, "context": _json_mapping(self._context, "context"),
        }

    def _assemble(self, result: MultimodalDataset | Mapping[str, Any]) -> MultimodalDataset:
        if isinstance(result, MultimodalDataset):
            if self._base is not None:
                raise ValueError("A provider with a fixed base must return an explicit partial mapping")
            return result.take(result.sample_ids)
        if not isinstance(result, Mapping):
            raise TypeError("generate must return a MultimodalDataset or a constructor-field mapping")
        unknown = set(result) - _DATA_FIELDS
        if unknown or "sample_ids" not in result:
            raise ValueError(f"Provider output requires sample_ids and only dataset fields; unknown={sorted(unknown, key=str)}")
        payload = dict(result)
        if self._base is None:
            if "sources" not in payload:
                raise ValueError("A complete provider output requires sources when no base is given")
            return MultimodalDataset(**payload)
        base = self._base
        ordered = base.take(payload["sample_ids"])
        if len(ordered) != len(base):
            raise ValueError("Provider output sample_ids must equal the fixed base universe")
        sources = payload.get("sources", {})
        if not isinstance(sources, Mapping):
            raise ValueError("Provider sources must be a mapping of names to TensorSource")
        collisions = sources.keys() & base.sources.keys()
        if collisions - set(self._replace_sources):
            raise ValueError(f"Replacing sources requires replace_sources: {sorted(collisions - set(self._replace_sources))}")
        if set(self._replace_sources) - sources.keys():
            raise ValueError("Provider output must supply every declared replace_sources entry")
        if base.y is not None and payload.keys() & _TARGET_FIELDS and not self._replace_targets:
            raise ValueError("Replacing existing targets requires replace_targets=True")
        for field in ("groups", "partitions"):
            if field in payload:
                supplied, expected = payload[field], getattr(ordered, field)
                if (supplied is None) != (expected is None) or (supplied is not None and not np.array_equal(supplied, expected)):
                    raise ValueError(f"Provider output cannot change fixed base {field}")
        assembled = MultimodalDataset(
            {**ordered.sources, **sources}, sample_ids=ordered.sample_ids,
            y=payload.get("y", ordered.y),
            target_names=payload.get("target_names", None if "y" in payload else ordered.target_names),
            target_mask=payload.get("target_mask", None if "y" in payload else ordered.target_mask),
            task_type=payload.get("task_type", ordered.task_type),
            source_alignment=payload.get("source_alignment", ordered.source_alignment),
            groups=ordered.groups, partitions=ordered.partitions,
            name=payload.get("name", ordered.name),
        )
        return assembled.take(base.sample_ids)

    def _produce(self, seed: int, context: dict[str, Any]) -> tuple[MultimodalDataset, dict[str, Any]]:
        result = self._generate(seed=seed, params=_json_mapping(self._params, "params"), context=_json_mapping(context, "context"))
        cohort = self._assemble(result)
        state = {
            "schema": "nirs4all.data-provider", "version": 1, "recipe": self.recipe(),
            "seed": seed, "context": context, "fingerprint": _cohort_fingerprint(cohort),
        }
        return cohort, state

    def materialize(self, *, seed: int | None = None, context: Mapping[str, Any] | None = None) -> MultimodalDataset:
        """Explicitly generate a cohort; failure preserves any previous cohort.

        An explicit context replaces the recipe context for this materialization
        and is recorded in checkpoints. The effective seed may be a native task
        seed without changing the provider's root recipe seed.
        """
        effective_seed = self.seed if seed is None else _unsigned(seed, "seed", maximum=2**64 - 1)
        effective_context = _json_mapping(self._context if context is None else context, "context")
        cohort, state = self._produce(effective_seed, effective_context)
        self._cohort, self._state = cohort, state
        return cohort

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON checkpoint, rejecting mutations of the stored cohort."""
        cohort = self.cohort
        assert self._state is not None
        if _cohort_fingerprint(cohort) != self._state["fingerprint"]:
            raise ValueError("Materialized provider content changed after generation")
        return _json_mapping(self._state, "checkpoint")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Regenerate from a checkpoint, then atomically verify and restore it."""
        saved = _closed_state(state, fields=_STATE_FIELDS, schema="nirs4all.data-provider")
        if _digest(saved["recipe"]) != _digest(self.recipe()):
            raise ValueError("Provider checkpoint recipe does not match this provider")
        seed = _unsigned(saved["seed"], "checkpoint seed", maximum=2**64 - 1)
        context = _json_mapping(saved["context"], "checkpoint context")
        fingerprint = saved["fingerprint"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("Provider checkpoint fingerprint must be a SHA-256 hex digest")
        cohort, restored = self._produce(seed, context)
        if restored["fingerprint"] != fingerprint:
            raise ValueError("Regenerated provider content does not match checkpoint fingerprint")
        self._cohort, self._state = cohort, restored

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, index: int) -> MultimodalDataset:
        """Select one positional sample as a one-row cohort, retaining axes."""
        cohort = self.cohort
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("Provider index must be an integer")
        if index < 0 or index >= len(cohort):
            raise IndexError("Provider index out of range")
        return cohort.take([cohort.sample_ids[index]])

    def get(self, sample_ids: Sequence[str]) -> MultimodalDataset:
        """Select an explicitly ordered ID view without generating or fitting."""
        return self.cohort.take(sample_ids)

    def batches(
        self, batch_size: int, *, sample_ids: Sequence[str] | None = None,
        start: int = 0, drop_last: bool = False,
    ) -> ProviderBatches:
        """Iterate a bounded view in caller-specified order, without shuffling.

        ``start`` is a sample position within that view. CV, epochs, shuffling
        and worker sharding belong to callers/frameworks and must provide their
        own explicit sample-ID views.
        """
        return ProviderBatches(self, batch_size, sample_ids=sample_ids, start=start, drop_last=drop_last)


class ProviderBatches(Iterator[MultimodalDataset]):
    """A bounded batch cursor over a materialized provider cohort.

    Resume with ``provider.load_state_dict(state['provider'])``, recreate the
    same batch view, then call ``batches.load_state_dict(state)``. Regeneration
    is explicit and happens once before cursor restoration. Loading batch state
    itself only verifies the already materialized content and configuration.
    """

    def __init__(
        self, provider: DataProvider, batch_size: int, *,
        sample_ids: Sequence[str] | None = None, start: int = 0, drop_last: bool = False,
    ) -> None:
        self._batch_size = _unsigned(batch_size, "batch_size")
        if self._batch_size == 0:
            raise ValueError("batch_size must be positive")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be boolean")
        self._drop_last = drop_last
        self._cohort = provider.cohort
        self._sample_ids = self._cohort.take(self._cohort.sample_ids if sample_ids is None else sample_ids).sample_ids
        self._provider_state = provider.state_dict()
        self._position = _unsigned(start, "start", maximum=len(self._sample_ids))

    def __iter__(self) -> ProviderBatches:
        return self

    def __next__(self) -> MultimodalDataset:
        remaining = len(self._sample_ids) - self._position
        if remaining == 0 or (self._drop_last and remaining < self._batch_size):
            self._position = len(self._sample_ids)
            raise StopIteration
        end = min(len(self._sample_ids), self._position + self._batch_size)
        batch = self._cohort.take(self._sample_ids[self._position:end])
        self._position = end
        return batch

    def state_dict(self) -> dict[str, Any]:
        """Serialize cursor and full view/provider identity as finite JSON."""
        if _cohort_fingerprint(self._cohort) != self._provider_state["fingerprint"]:
            raise ValueError("Materialized provider content changed after batch creation")
        return {
            "schema": "nirs4all.provider-batches", "version": 1,
            "provider": _json_mapping(self._provider_state, "provider checkpoint"),
            "sample_ids": list(self._sample_ids), "batch_size": self._batch_size,
            "drop_last": self._drop_last, "position": self._position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Verify the regenerated cohort/view before changing the cursor."""
        saved = _closed_state(state, fields=_BATCH_FIELDS, schema="nirs4all.provider-batches")
        current = self.state_dict()
        position = _unsigned(saved["position"], "checkpoint position", maximum=len(self._sample_ids))
        if _digest({key: saved[key] for key in _BATCH_FIELDS - {"position"}}) != _digest({key: current[key] for key in _BATCH_FIELDS - {"position"}}):
            raise ValueError("Batch checkpoint provider or view configuration does not match")
        self._position = position
