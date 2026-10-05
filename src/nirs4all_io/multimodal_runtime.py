# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""IO-to-DAG-ML-Data bridge for a current, target-free raw multimodal cohort."""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from .dataset_facade import to_u07_raw_sources
from .public_content import dataset_content_bytes
from .public_dataset import dataset


def _runtime_group_ids(groups: Any) -> list[str] | None:
    if groups is None:
        return None
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Multimodal replay group IDs must be nonempty strings")
    return list(groups)


def multimodal_runtime_input(value: Any) -> dict[str, Any]:
    """Build independently identified raw values and native data relations.

    DAG-ML-Data validates descriptors and derives relation fingerprints. This
    adapter never receives a trained graph, artifact, request or trust manifest.
    Explicit input origins are metadata; they do not declare augmentation.
    Declared replay group IDs must be nonempty strings, as native identities,
    without host-specific numeric formatting. Generic IO groups may be numeric.
    """
    import dag_ml_data

    ds = dataset(value)
    cohort = ds.multimodal
    if cohort.y is not None or any(partition != "predict" for partition in cohort.partitions):
        raise ValueError("Raw replay requires a target-free predict cohort")
    if tuple(cohort.target_names) not in ((), ("y",)):
        raise ValueError("U07 replay requires absent target names or ['y']")
    group_ids = _runtime_group_ids(cohort.groups)
    raw = to_u07_raw_sources(cohort)
    source_ids = [f"src{i}" for i in range(len(cohort.sources))]
    sources = []
    for source_id, descriptor in zip(source_ids, cohort.descriptors(), strict=True):
        sources.append({"id": source_id, "name": descriptor["source_id"],
                        "type_id": descriptor["type_id"], "modality": descriptor["modality"],
                        "native_representation": copy.deepcopy(descriptor["native_representation"]),
                        "sample_key": "sample_id", "granularity": "per_sample", "schema": {}, "tags": {}})
    target = {"id": "target_numeric_matrix", "type_id": "target", "rank": 2,
              "axes": [{"name": "sample", "kind": "sample", "unit": None, "size": len(cohort), "variable": False},
                       {"name": "target", "kind": "target", "unit": None, "size": 1, "variable": False}],
              "container": "array", "dtype": "float64", "sparse": False, "ragged": False}
    schema = {"dataset_id": f"nirs4all.{cohort.name}", "sample_ids": list(cohort.sample_ids),
              "sources": sources, "targets": {"y": target}, "metadata": {}}
    steps: list[dict[str, Any]] = [{"kind": "materialize", "source_id": source_id, "adapter_id": None,
              "input_representation": None, "output_representation": source.representation_id,
              "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"output": f"src:{source_id}"}}
             for source_id, source in zip(source_ids, cohort.sources.values(), strict=True)]
    steps.append({"kind": "join", "source_id": None, "adapter_id": None, "input_representation": None,
                  "output_representation": "feature_block_set", "fit_scope": "stateless", "requires_user_choice": False,
                  "metadata": {"inputs": [f"src:{s}" for s in source_ids], "output": "port:X"}})
    plan = {"id": f"plan.{cohort.name}", "steps": steps, "output_representation": "feature_block_set", "issues": []}
    rows = []
    for i, sample_id in enumerate(cohort.sample_ids):
        metadata = {"input_origin_id": ds.origin_ids[i]}
        if cohort.independent_unit_ids is not None:
            metadata["independent_unit_id"] = cohort.independent_unit_ids[i]
        if cohort.repetition_ids is not None:
            metadata["repetition_id"] = cohort.repetition_ids[i]
        rows.append({"observation_id": sample_id, "sample_id": sample_id, "source_id": None, "target_id": "y",
                     "group_id": None if group_ids is None else group_ids[i], "origin_id": None,
                     "repetition_id": None if cohort.repetition_ids is None else cohort.repetition_ids[i],
                     "augmented": False, "excluded": False, "metadata": metadata})
    envelope = dag_ml_data.build_coordinator_data_plan_envelope(schema, plan, {"rows": rows}).to_dict()
    # IO owns the current observation annotations passed to native DAG.
    by_id = {row["observation_id"]: row for row in rows}
    for record in envelope["coordinator_relations"]["records"]:
        record["metadata"] = copy.deepcopy(by_id[record["observation_id"]]["metadata"])
    content = dataset_content_bytes(ds.to_dict())
    return {**raw, "source_ids": source_ids, "data_envelope": envelope,
            "data_content_fingerprint": hashlib.sha256(content).hexdigest()}
