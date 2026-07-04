# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Ecosystem E2E entrypoint for the formats -> io -> datasets -> methods scenario."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import nirs4all_io as nio

SCENARIO_ID = "e2e-formats-io-datasets-methods-language-bindings"
REFERENCE_DATASET_IDS = (
    "cgl_nir_grain_eigenvector",
    "ohpl_beer_nir",
)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _datasets_root() -> Path:
    root = _workspace_root() / "nirs4all-datasets"
    if not root.exists():
        raise FileNotFoundError(f"missing sibling repository: {root}")
    return root


def _import_nirs4all_datasets() -> Any:
    datasets_root = _datasets_root()
    datasets_src = datasets_root / "src"
    if str(datasets_src) not in sys.path:
        sys.path.insert(0, str(datasets_src))
    import nirs4all_datasets as n4ad

    return n4ad


def _expected_metadata(reference: Any) -> tuple[list[str], np.ndarray | None]:
    frames: list[Any] = []
    metadata = reference.metadata()
    split = reference.split()
    if metadata is not None:
        frames.append(metadata)
    if split is not None:
        frames.append(split)
    if not frames:
        return [], None

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="sample_id", how="outer", validate="1:1")

    sample_order = [str(value) for value in reference.sample_ids().tolist()]
    aligned = combined.set_index("sample_id").loc[sample_order]
    return aligned.columns.tolist(), aligned.to_numpy()


def _validate_reference_dataset(reference: Any) -> dict[str, Any]:
    package = nio.load(reference, target="dataset_package")
    summary = nio.describe_dataset_package(reference)
    assembled = package.to_assembled()
    manifest = package.manifest()

    assert package.name == reference.id
    assert isinstance(summary, dict)
    assert summary["name"] == reference.id
    assert summary["manifest"]["root"] == manifest.root
    assert set(assembled.blocks) == {"train"}
    assert summary["identity"]["row_position_fallback"] == package.row_position_fallback.to_dict()
    assert len(package.row_position_fallback.fingerprint) == 64
    assert package.row_position_fallback.reason

    block = assembled.blocks["train"]
    sample_order = [str(value) for value in reference.sample_ids().tolist()]
    assert block.n_samples == len(sample_order)
    assert summary["partitions"] == {"train": {"n_samples": len(sample_order)}}
    assert len(block.X) == len(reference.sources())

    for index, source_id in enumerate(reference.sources()):
        expected_x = reference.x(source=source_id)
        expected_wavelengths = reference.wavelengths(source_id)
        np.testing.assert_allclose(block.X[index], expected_x, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(block.feature_headers[index], dtype=float),
            expected_wavelengths,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )

    expected_y = reference.y()
    assert expected_y is not None
    expected_y_aligned = expected_y.set_index("sample_id").loc[sample_order, block.y_headers]
    np.testing.assert_allclose(block.y, expected_y_aligned.to_numpy(), rtol=1e-6, atol=1e-8)

    expected_metadata_columns, expected_metadata_values = _expected_metadata(reference)
    if expected_metadata_values is None:
        assert block.metadata is None
    else:
        assert block.metadata is not None
        assert block.metadata.columns.tolist() == expected_metadata_columns
        np.testing.assert_array_equal(block.metadata.to_numpy(), expected_metadata_values)

    expected_payload_count = len(block.X) + 1 + (0 if block.metadata is None else 1) + (0 if block.weights is None else 1)
    assert len(manifest.entries) == expected_payload_count

    return {
        "dataset_id": reference.id,
        "sources": reference.sources(),
        "summary": summary,
        "payload_ids": [entry.id for entry in manifest.entries],
        "target_headers": list(block.y_headers),
        "metadata_columns": [] if block.metadata is None else block.metadata.columns.tolist(),
    }


def test_assemble_reference_datasets(artifacts_dir: Path) -> None:
    n4ad = _import_nirs4all_datasets()
    datasets_root = _datasets_root()

    validated = [
        _validate_reference_dataset(n4ad.get(dataset_id, root=datasets_root))
        for dataset_id in REFERENCE_DATASET_IDS
    ]

    out = artifacts_dir / "assembled-datasets.json"
    out.write_text(
        json.dumps(
            {
                "scenario": SCENARIO_ID,
                "workspace_root": str(_workspace_root()),
                "datasets_root": str(datasets_root),
                "datasets": validated,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert out.exists()
