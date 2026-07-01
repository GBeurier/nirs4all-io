# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Public DatasetPackage API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

import nirs4all_io as nio
from nirs4all_io.materialize import DatasetPackage, PayloadStorageKind, repr_ids


def _csv(path, df):
    df.to_csv(path, sep=";", index=False)


def test_to_dataset_package_exposes_manifest_and_summary(tmp_path):
    _csv(
        tmp_path / "data.csv",
        pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "rep": ["a", "b", "c"],
                "400": [0.1, 0.2, 0.3],
                "401": [0.4, 0.5, 0.6],
                "y": [1.0, 2.0, 3.0],
                "weight": [1.0, 0.5, 2.0],
                "site": ["north", "south", "north"],
            }
        ),
    )
    spec = {
        "sample_index": {"by": "id", "key": "sample_id"},
        "repetition": "rep",
        "sources": [
                {
                    "id": "data",
                    "role": "mixed",
                    "input": "data.csv",
                    "key": "sample_id",
                    "columns": [
                    {"role": "features", "select": ["400", "401"]},
                    {"role": "targets", "select": ["y"]},
                    {"role": "weights", "select": ["weight"]},
                    {"role": "metadata", "select": ["rep", "site"]},
                ],
            }
        ],
    }

    package = nio.to_dataset_package(spec, base_dir=tmp_path, name="pkg")

    assert isinstance(package, DatasetPackage)
    assert package.name == "pkg"
    manifest = package.manifest()
    by_id = {entry.id: entry for entry in manifest.entries}
    assert set(by_id) == {"train/x0", "train/y", "train/metadata", "train/weights"}
    assert by_id["train/x0"].representation_id == repr_ids.SIGNAL_1D
    assert by_id["train/x0"].axes == ["sample", "feature"]
    assert by_id["train/y"].representation_id == repr_ids.TARGET_NUMERIC
    assert by_id["train/metadata"].representation_id == repr_ids.SAMPLE_METADATA
    assert by_id["train/weights"].role == "weights"
    assert all(entry.storage is PayloadStorageKind.INLINE for entry in manifest.entries)
    assert all(len(entry.content_hash) == 64 for entry in manifest.entries)
    assert len(manifest.root) == 64

    fallback = package.row_position_fallback
    assert fallback.used is False
    assert "repetition key 'rep'" in fallback.reason
    assert len(fallback.fingerprint) == 64

    summary = package.to_summary_dict()
    assert summary["schema_version"] == 2
    assert summary["partitions"] == {"train": {"n_samples": 3}}
    assert summary["manifest"]["root"] == manifest.root

    canonical = package.to_canonical_summary()
    assert canonical.endswith("\n")
    assert "\"schema_version\": 2" in canonical
    assert "\"data\"" not in canonical


def test_load_dataset_package_target_and_describe_accept_in_memory_input():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = np.arange(4, dtype=np.float32)

    package = nio.load((X, y), target="dataset_package", name="arrays")
    assert isinstance(package, DatasetPackage)
    assert package.name == "arrays"

    described = nio.describe_dataset_package(package)
    assert isinstance(described, dict)
    assert described["name"] == "arrays"
    assert described["partitions"]["train"]["n_samples"] == 4


def test_dataset_package_round_trips_v1_assembled_payloads():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = np.arange(4, dtype=np.float32)
    assembled = nio.load((X, y), target="assembled", name="roundtrip")
    package = nio.to_dataset_package(assembled)

    restored = package.to_assembled()

    assert restored.name == assembled.name
    assert restored.task_type == assembled.task_type
    assert restored.signal_type == assembled.signal_type
    assert restored.n_sources == assembled.n_sources
    np.testing.assert_allclose(restored.blocks["train"].X[0], assembled.blocks["train"].X[0])
    np.testing.assert_allclose(restored.blocks["train"].y, assembled.blocks["train"].y)
