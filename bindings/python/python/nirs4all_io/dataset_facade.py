# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Small user-facing projections of IO's versioned multimodal dataset record.

These functions assemble and validate raw sources only. Learned encoders,
prediction and archive verification belong to Methods, DAG-ML and Core.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .multimodal import MultimodalDataset, TensorSource
from .public_content import canonical_source_schema, metadata_number

U07_SOURCE_ORDER = ("nir", "image", "series", "metadata")
_U07_TYPES = {"nir": "signal_1d", "image": "rgb_image", "series": "series_mv", "metadata": "tabular_mixed"}


def load_dataset_document(definition: Any) -> Any:
    """Read IO JSON/YAML input without discarding its outer public envelope."""
    if isinstance(definition, (Mapping, MultimodalDataset)):
        return definition
    if not isinstance(definition, (str, Path)):
        raise TypeError("dataset definition must be a MultimodalDataset, mapping, JSON/YAML text or path")
    source = str(definition)
    path = Path(source)
    is_path = isinstance(definition, Path)
    if not is_path and len(source) < 4096 and "\n" not in source and not source.lstrip().startswith(("{", "[")):
        try:
            is_path = path.is_file()
        except OSError:
            is_path = False
    if is_path:
        text = path.read_text(encoding="utf-8")
    else:
        text = source
    try:
        document = json.loads(text) if text.lstrip().startswith(("{", "[")) else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError("dataset definition must contain valid JSON or safe YAML") from error
    if not isinstance(document, Mapping):
        raise ValueError("dataset definition must contain an object")
    return document


def load_multimodal_definition(definition: Mapping[str, Any] | str | Path | MultimodalDataset) -> MultimodalDataset:
    """Load the same closed IO dataset declaration from a dict, JSON or YAML.

    A path is read as UTF-8; text is parsed directly. YAML uses ``safe_load``.
    The existing ``MultimodalDataset`` constructor owns all shape, identity and
    alignment checks. It never reads an original training workspace.
    """
    document = load_dataset_document(definition)
    if isinstance(document, MultimodalDataset):
        return document
    return MultimodalDataset.from_dict(document)


def to_dense_regression(
    definition: Mapping[str, Any] | str | Path | MultimodalDataset,
    *, source_id: str,
) -> dict[str, Any]:
    """Project one explicitly chosen numeric matrix for a dense workflow.

    A multimodal cohort is never silently flattened or fused. The workflow
    receives exactly ``X``, ``y`` and aligned ``sample_ids``.
    """
    cohort = load_multimodal_definition(definition)
    if source_id not in cohort.sources:
        raise ValueError(f"Unknown source {source_id!r}; choose one of {list(cohort.sources)}")
    source = cohort.sources[source_id]
    if not isinstance(source, TensorSource) or source.values.ndim != 2 or source.values.dtype.kind not in "biuf":
        raise ValueError(f"Source {source_id!r} must be a dense numeric rank-2 matrix")
    if source.presence_mask is None or not np.all(source.presence_mask):
        raise ValueError(f"Source {source_id!r} has missing rows; dense workflows require complete input")
    if not np.isfinite(source.values).all():
        raise ValueError(f"Source {source_id!r} contains non-finite values")
    if cohort.y is None or cohort.y.ndim != 1 or cohort.y.dtype.kind not in "biuf":
        raise ValueError("Dense regression requires one numeric target vector")
    if cohort.target_mask is not None and not np.all(cohort.target_mask):
        raise ValueError("Dense regression requires all selected targets observed")
    if not np.isfinite(cohort.y).all():
        raise ValueError("Dense regression target contains non-finite values")
    return {"X": source.values.astype(float).tolist(), "y": cohort.y.astype(float).tolist(), "sample_ids": list(cohort.sample_ids),
            "partitions": cohort.partitions.tolist(), "target_names": list(cohort.target_names),
            "groups": None if cohort.groups is None else cohort.groups.tolist(),
            "independent_unit_ids": None if cohort.independent_unit_ids is None else list(cohort.independent_unit_ids),
            "repetition_ids": None if cohort.repetition_ids is None else list(cohort.repetition_ids)}


def to_u07_raw_sources(
    definition: Mapping[str, Any] | str | Path | MultimodalDataset,
) -> dict[str, Any]:
    """Project the closed four-source U07 raw input for native PREDICT replay.

    Schema identities preserve shape, units, coordinates and columns.
    This function creates no
    training request, signature, fitted state or controller trust manifest.
    """
    cohort = load_multimodal_definition(definition)
    if tuple(cohort.sources) != U07_SOURCE_ORDER:
        raise ValueError(f"U07 requires ordered sources {U07_SOURCE_ORDER}")
    if len(cohort.sample_ids) == 0:
        raise ValueError("U07 prediction input requires sample IDs")
    if cohort.source_alignment != "strict":
        raise ValueError("U07 replay requires strict source alignment")
    result: dict[str, Any] = {}
    schemas: dict[str, Any] = {}
    for name, descriptor in zip(U07_SOURCE_ORDER, cohort.schema_descriptors(), strict=True):
        source = cohort.sources[name]
        if not isinstance(source, TensorSource) or descriptor["representation_id"] != _U07_TYPES[name]:
            raise ValueError(f"U07 source {name!r} has an incompatible representation")
        if source.presence_mask is None or not np.all(source.presence_mask):
            raise ValueError(f"U07 source {name!r} has missing rows")
        shape = list(source.values.shape)
        if shape[0] != len(cohort.sample_ids) or any(size < 1 for size in shape[1:]):
            raise ValueError(f"U07 source {name!r} shape is incompatible with sample IDs")
        if name == "metadata":
            if shape[1:] != [2] or source.feature_names is None or len(source.feature_names) != 2:
                raise ValueError("U07 metadata needs declared numeric and categorical columns")
            if any(not isinstance(cell, str) for cell in source.values[:, 1]):
                raise ValueError("U07 metadata categories must be strings")
            try:
                numeric = np.asarray([metadata_number(cell.item() if isinstance(cell, np.generic) else cell) for cell in source.values[:, 0]])
            except (TypeError, ValueError) as error:
                raise ValueError("U07 metadata numeric column contains an incompatible value") from error
            if not np.isfinite(numeric).all():
                raise ValueError("U07 metadata numeric column contains non-finite values")
        elif source.values.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(source.values).all():
            raise ValueError(f"U07 source {name!r} needs finite float32/float64 raw values")
        identity = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        schema = canonical_source_schema({"representation_id": source.representation_id, "input_shape": shape[1:], "dtype": str(source.values.dtype), "identity": identity})
        schemas[name] = schema
        entry = {"sample_ids": list(cohort.sample_ids), "descriptor": schema, "shape": shape}
        if name == "metadata":
            entry["rows"] = source.values.tolist()
        else:
            entry["data"] = source.values.reshape(-1, order="C").tolist()
        result[name] = entry
    return {"sample_ids": list(cohort.sample_ids), "source_schemas": schemas, "sources": result}
