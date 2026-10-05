# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Public dataset transport with explicit observation origins and fixed folds."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dataset_facade import load_dataset_document, load_multimodal_definition, to_dense_regression
from .multimodal import MultimodalDataset, TensorSource


def _portable_storage(values: np.ndarray, label: str) -> None:
    if values.dtype.kind not in "biufUO" or (values.dtype.kind == "f" and values.dtype.itemsize > 8):
        raise ValueError(f"Public dataset has unsupported {label} dtype")
    if values.dtype.kind == "f" and not np.isfinite(values).all():
        raise ValueError(f"Public dataset transport requires finite {label} values")
    if values.dtype.kind in "iu" and np.any(np.abs(values.astype(object)) > 2**53 - 1):
        raise ValueError("Public dataset integers must be exactly representable in JavaScript")
    if values.dtype.kind == "O":
        for cell in values.flat:
            if isinstance(cell, (float, np.floating)) and not np.isfinite(cell):
                raise ValueError(f"Public dataset transport requires finite {label} values")
            if isinstance(cell, (int, float, np.integer, np.floating)) and float(cell).is_integer() and abs(float(cell)) > 2**53 - 1:
                raise ValueError("Public dataset integers must be exactly representable in JavaScript")


class Dataset:
    """IO-owned raw cohort; origins and folds are supplied, never generated.

    Repeated origin IDs cannot cross partitions or folds. A fold label may be
    null for a row outside CV. Origins default to the explicit sample IDs.
    """

    multimodal: MultimodalDataset
    origin_ids: tuple[str, ...]
    fold_ids: tuple[str | None, ...]

    def __init__(self, definition: Any, *, origin_ids: Sequence[str] | None = None,
                 fold_ids: Sequence[str | None] | None = None) -> None:
        definition = load_dataset_document(definition)
        if isinstance(definition, Mapping) and definition.get("schema") == "nirs4all.dataset.v1":
            if origin_ids is not None or fold_ids is not None:
                raise ValueError("Dataset options cannot override an envelope")
            parsed = type(self).from_dict(definition)
            self.multimodal, self.origin_ids, self.fold_ids = parsed.multimodal, parsed.origin_ids, parsed.fold_ids
            return
        self.multimodal = load_multimodal_definition(definition)
        for source in self.multimodal.sources.values():
            if not isinstance(source, TensorSource):
                raise ValueError("Public cross-language Dataset currently requires fixed-size tensor sources")
            _portable_storage(source.values, "source")
            for coordinates in (source.axis_coordinates or {}).values():
                for coordinate in coordinates:
                    if isinstance(coordinate, (int, float, np.integer, np.floating)) and float(coordinate).is_integer() and abs(float(coordinate)) > 2**53 - 1:
                        raise ValueError("Public axis coordinates must be exactly representable in JavaScript")
        for label, values in (("target", self.multimodal.y), ("group", self.multimodal.groups)):
            if values is not None:
                _portable_storage(values, label)
        ids = self.multimodal.sample_ids
        self.origin_ids = tuple(ids if origin_ids is None else origin_ids)
        self.fold_ids = tuple([None] * len(ids) if fold_ids is None else fold_ids)
        if len(self.origin_ids) != len(ids) or any(not isinstance(x, str) or not x.strip() for x in self.origin_ids):
            raise ValueError("origin_ids must contain one nonempty string per sample")
        if len(self.fold_ids) != len(ids) or any(x is not None and (not isinstance(x, str) or not x.strip()) for x in self.fold_ids):
            raise ValueError("fold_ids must contain one nonempty string or null per sample")
        assignments: dict[str, tuple[str, str | None]] = {}
        for origin, partition, fold in zip(self.origin_ids, self.multimodal.partitions, self.fold_ids, strict=True):
            if partition != "train" and fold is not None:
                raise ValueError("Only training samples may declare CV fold_ids")
            membership = (str(partition), fold)
            if origin in assignments and assignments[origin] != membership:
                raise ValueError("An origin cannot cross partitions or folds")
            assignments[origin] = membership
        for label, unit_values in (("group", self.multimodal.groups), ("independent unit", self.multimodal.independent_unit_ids)):
            if unit_values is None:
                continue
            units: dict[Any, str | None] = {}
            for unit, fold in zip(unit_values, self.fold_ids, strict=True):
                if unit in units and units[unit] != fold:
                    raise ValueError(f"A {label} cannot cross folds")
                units[unit] = fold

    @classmethod
    def from_sources(cls, sources: Mapping[str, Any], *, sample_ids: Sequence[str],
                     y: Any = None, representations: Mapping[str, str] | None = None,
                     axis_units: Mapping[str, Any] | None = None,
                     axis_coordinates: Mapping[str, Any] | None = None,
                     feature_names: Mapping[str, Any] | None = None,
                     origin_ids: Sequence[str] | None = None,
                     fold_ids: Sequence[str | None] | None = None, **options: Any) -> Dataset:
        """Construct named host arrays through existing IO tensor validation.

        Numeric matrices default to tabular_numeric (spectra/nir to signal_1d).
        Other ranks require an explicit representation declaration. TensorSource
        objects retain their own row IDs, axes, units and presence mask.
        """
        declarations = {"spectra": "signal_1d", "nir": "signal_1d", "image": "rgb_image",
                        "series": "series_mv", "metadata": "tabular_mixed"}
        built = {}
        for name, source in sources.items():
            if isinstance(source, TensorSource):
                built[name] = source
                continue
            representation = (representations or {}).get(name, declarations.get(name, "tabular_numeric"))
            built[name] = TensorSource(source, sample_ids, representation_id=representation,
                                      axis_units=(axis_units or {}).get(name),
                                      axis_coordinates=(axis_coordinates or {}).get(name),
                                      feature_names=(feature_names or {}).get(name))
        if y is None and options.get("partitions") is None:
            options["partitions"] = ["predict"] * len(sample_ids)
        return cls(MultimodalDataset(built, sample_ids=sample_ids, y=y, **options),
                   origin_ids=origin_ids, fold_ids=fold_ids)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.multimodal.sample_ids

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Dataset:
        if set(value) != {"schema", "schema_version", "dataset", "origin_ids", "fold_ids"}:
            raise ValueError("Dataset envelope has invalid fields")
        if value["schema"] != "nirs4all.dataset.v1" or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported public dataset schema")
        if not isinstance(value["origin_ids"], list) or not isinstance(value["fold_ids"], list):
            raise ValueError("origin_ids and fold_ids must be arrays")
        return cls(value["dataset"], origin_ids=value["origin_ids"], fold_ids=value["fold_ids"])

    def to_dict(self) -> dict[str, Any]:
        raw = self.multimodal.to_dict()
        for source in raw["sources"]:
            source["axis_units"] = {axis: unit for axis, unit in source["axis_units"].items() if unit is not None}
        return {"schema": "nirs4all.dataset.v1", "schema_version": 1,
                "dataset": raw, "origin_ids": list(self.origin_ids), "fold_ids": list(self.fold_ids)}

    def take(self, sample_ids: Sequence[str]) -> Dataset:
        positions = {sample: i for i, sample in enumerate(self.sample_ids)}
        selected = self.multimodal.take(sample_ids)
        return type(self)(selected, origin_ids=[self.origin_ids[positions[s]] for s in sample_ids],
                          fold_ids=[self.fold_ids[positions[s]] for s in sample_ids])

    def to_dense_regression(self, source_id: str) -> dict[str, Any]:
        return {**to_dense_regression(self.multimodal, source_id=source_id), "origin_ids": list(self.origin_ids), "fold_ids": list(self.fold_ids)}


def dataset(value: Any, **kwargs: Any) -> Dataset:
    """Construct or validate the public dataset using the IO contract."""
    if isinstance(value, Dataset):
        if kwargs:
            raise ValueError("Dataset options cannot override an existing Dataset")
        return value
    value = load_dataset_document(value)
    if isinstance(value, Mapping) and value.get("schema") == "nirs4all.dataset.v1":
        if kwargs:
            raise ValueError("Dataset options cannot override a dataset envelope")
        return Dataset.from_dict(value)
    if isinstance(value, Mapping) and "sample_ids" in kwargs:
        return Dataset.from_sources(value, **kwargs)
    return Dataset(value, **kwargs)
