# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Native public inputs: no oracle assembly or parser retries."""

import json

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio


def test_array_tuple_preserves_values_partition_order_and_package():
    x = np.arange(18, dtype=np.float32).reshape(6, 3) / 7
    y = np.arange(6, dtype=np.float32) * 0.7
    split = np.array(["test", "train", "predict", "train", "test", "train"])
    package = nio.load((x, y, split), target="dataset_package")
    blocks = package.to_assembled().blocks
    for partition in ("train", "test", "predict"):
        np.testing.assert_array_equal(blocks[partition].X[0], x[split == partition])
        np.testing.assert_array_equal(blocks[partition].y.ravel(), y[split == partition])


def test_arrays_with_metadata_and_x_only_prediction_have_no_synthetic_targets():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    package = nio.load({"X": x, "metadata": pd.DataFrame({"subject": ["b", "a", "b", "c"]})}, target="package")
    block = package.to_assembled().blocks["predict"]
    np.testing.assert_array_equal(block.X[0], x)
    assert block.y is None
    assert block.metadata["subject"].tolist() == ["b", "a", "b", "c"]


@pytest.mark.parametrize(
    "inp,limits",
    [
        ((np.ones((4, 3)), np.ones(3)), None),
        ((np.ones((4, 3)), np.ones(4), np.array(["train", "typo", "test", "test"])), None),
        (np.ones((4, 3)), {"max_cells": 2}),
        (np.ones((4, 3)), {"max_decoded_total_bytes": 16}),
    ],
)
def test_array_admission_rejects_misalignment_and_small_budgets(inp, limits):
    with pytest.raises(ValueError):
        nio.load(inp, limits=limits)


def test_native_frames_direct_entry_enforces_budget_before_copy():
    from nirs4all_io._native import assemble_frames

    spec = {"sources": [{"id": "x", "role": "features", "input": "x"}]}
    frames = [{"name": "x", "columns": ["a"], "rows": [["longer"]]}]
    with pytest.raises(ValueError, match="field.*limit"):
        assemble_frames(spec, frames, limits={"max_field_bytes": 2})


def test_missing_array_values_are_preserved_without_dropping_rows_or_features():
    x = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    block = nio.load(x, target="package").to_assembled().blocks["predict"]
    np.testing.assert_array_equal(block.X[0], x)


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_infinite_array_values_are_not_silently_replaced_by_missing_values(value):
    with pytest.raises(ValueError, match="infinite"):
        nio.load(np.array([[1.0, value]]), target="package")


def test_yaml_relative_refs_false_header_and_aggregate_budget(tmp_path):
    (tmp_path / "X.csv").write_text("1;2\n3;4\n")
    (tmp_path / "Y.csv").write_text("10\n20\n")
    config = tmp_path / "dataset.yaml"
    config.write_text("train_x: X.csv\ntrain_y: Y.csv\nglobal_params:\n  has_header: false\n  delimiter: ';'\n")
    spec = nio.to_spec(config)
    assert all(source["input"].startswith(str(tmp_path)) for source in spec.sources)
    block = nio.load(config, target="package").to_assembled().blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(block.y.ravel(), [10, 20])
    with pytest.raises(ValueError, match="budget|limit"):
        nio.load(config, limits={"max_total_bytes": config.stat().st_size + 1})


@pytest.mark.parametrize("document", ["sources: &cycle [*cycle]\n", "!!python/object/apply:os.system ['echo forbidden']\n"])
def test_yaml_cycles_and_unsafe_tags_fail_before_assembly(tmp_path, document):
    config = tmp_path / "unsafe.yaml"
    config.write_text(document)
    with pytest.raises((ValueError, __import__("yaml").YAMLError)):
        nio.load(config)


def test_scored_plan_mapping_attributes_and_direct_load_match_resolved_spec(tmp_path):
    (tmp_path / "Xcal.csv").write_text("1100;1102\n1.5;2.5\n3.5;4.5\n5.5;6.5\n")
    (tmp_path / "Ycal.csv").write_text("protein\n1.1\n2.3\n3.7\n")
    plan = nio.infer(tmp_path, hints=None)
    assert plan.structure.value == plan["structure"]["value"]
    assert plan.calibration["method"] == "none"
    assert json.loads(json.dumps(plan)) == plan.to_dict()
    assert nio.load(plan) == nio.load(plan.resolved_spec)
    assert plan.accept(name="reviewed").name == "reviewed"
    assert nio.DatasetPlan().resolved_spec is None
    with pytest.raises(ValueError, match="Non-empty"):
        nio.infer(tmp_path / "missing", hints={"task_type": "regression"})


def test_native_bare_convention_gate_covers_oracle_specific_exception_namespace_case(tmp_path):
    (tmp_path / "spectra.csv").write_text("1100;1102\n1.5;2.5\n3.5;4.5\n")
    (tmp_path / "target.csv").write_text("protein\n1.1\n2.3\n")
    with pytest.raises(ValueError, match="no dataset files recognized"):
        nio.load(tmp_path)
    block = nio.load(tmp_path, conventions=["bare"], target="package").to_assembled().blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[1.5, 2.5], [3.5, 4.5]])
    np.testing.assert_allclose(block.y.ravel(), [1.1, 2.3])


def test_array_categorical_targets_keep_labels_across_partitions():
    pytest.importorskip("nirs4all.data", reason="optional downstream SpectroDataset integration")
    x = np.arange(12).reshape(6, 2)
    labels = np.array(["b", "a", "b", "a", "b", "a"])
    split = np.array(["train", "train", "train", "test", "test", "test"])
    ds = nio.load((x, labels, split), target="spectrodataset")
    np.testing.assert_array_equal(ds.y({"y": "raw"}).ravel(), labels)
    np.testing.assert_array_equal(ds.y({"partition": "test"}).ravel(), [0, 1, 0])


@pytest.mark.parametrize("code", [-1, 0.5, 2, float("nan")])
def test_categorical_codebook_refuses_invalid_codes(code):
    from nirs4all_io._adapter import _decode_targets

    with pytest.raises(ValueError, match="codebook"):
        _decode_targets(np.array([[code]]), ["label"], {"label": {"categories": ["a", "b"]}})


def test_public_v2_ragged_masked_transport_and_matrix_projection():
    import json
    from pathlib import Path

    from nirs4all_io.public_dataset import Dataset

    fixture = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
    value = json.loads((fixture / "public-dataset-v2.json").read_text())
    expected = json.loads((fixture / "public-dataset-v2-normalized.json").read_text())
    cohort = Dataset.from_dict(value)
    assert cohort.to_dict() == expected
    assert cohort.take(["c", "a"]).to_dict()["dataset"]["sources"][1]["offsets"]["values"] == [0, 2, 3]
    with pytest.raises(ValueError, match="observed"):
        cohort.to_matrix_regression("matrix")
    projected = cohort.to_masked_matrix_regression("matrix")
    assert projected["y"] == [[1.0, 0.0], [2.0, 4.0], [3.0, 6.0], [0.0, 8.0]]
    assert projected["target_mask"] == expected["dataset"]["target_mask"]["values"]
    for kind in ("offsets", "times", "target", "v1"):
        wrong = json.loads(json.dumps(value))
        if kind == "offsets":
            wrong["dataset"]["sources"][1]["offsets"]["values"][2] = 4
        elif kind == "times":
            wrong["dataset"]["sources"][1]["time_coordinates"]["values"][1] = 0
        elif kind == "target":
            wrong["dataset"]["target_mask"]["values"][0][1] = True
        else:
            wrong["schema"] = "nirs4all.dataset.v1"
            wrong["schema_version"] = 1
        with pytest.raises(ValueError):
            Dataset.from_dict(wrong)


def test_projected_matrix_uses_native_owner_and_preserves_masks():
    import json
    from pathlib import Path

    from nirs4all_io import Dataset, projected_matrix_dataset

    value = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/public-dataset-v2.json").read_text())
    projections = [
        {"source_id": "matrix", "sample_ids": ["d", "c", "b", "a"], "array": {"dtype": "float64", "shape": [4, 1], "values": [[4.0], [3.0], [2.0], [1.0]]}, "feature_names": ["mean"], "presence_encoded": False},
        {
            "source_id": "series",
            "sample_ids": ["a", "b", "c", "d"],
            "array": {"dtype": "float64", "shape": [4, 2], "values": [[0.0, 0.0], [0.0, 0.0], [4.0, 1.0], [0.0, 0.0]]},
            "feature_names": ["mean", "present"],
            "presence_encoded": False,
        },
    ]
    with pytest.raises(ValueError, match="presence"):
        projected_matrix_dataset(value, projections)
    projections[1]["presence_encoded"] = True
    output = projected_matrix_dataset(value, projections)
    assert output["record"]["dataset"]["sources"][0]["array"]["values"] == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 4.0, 1.0], [4.0, 0.0, 0.0]]
    assert output["record"]["dataset"]["target_mask"] == value["dataset"]["target_mask"]
    assert len(output["provenance"]["source_projections"][1]["projection_content_fingerprint"]) == 64
    value["dataset"]["y"]["dtype"] = "float32"
    value["dataset"]["y"]["values"][0][1] = 1e99
    assert Dataset.from_dict(value).to_masked_matrix_regression("matrix")["y"][0][1] == 0


def test_masked_classifier_columns_keep_labels_names_and_observed_truth():
    import json
    from pathlib import Path

    from nirs4all_io.public_dataset import Dataset

    value = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/public-dataset-v2.json").read_text())
    value["dataset"]["task_type"] = "classification"
    value["dataset"]["target_names"] = ["class_a", "class_b"]
    value["dataset"]["y"]["dtype"] = "int64"
    value["dataset"]["y"]["values"] = [[0, None], [1, 3], [0, 7], [1e99, 3]]
    cohort = Dataset.from_dict(value)
    projected = cohort.to_masked_matrix_regression("matrix")
    assert projected["y"] == [[0, 0], [1, 3], [0, 7], [0, 3]]
    assert projected["target_names"] == ["class_a", "class_b"]
    assert projected["target_mask"] == value["dataset"]["target_mask"]["values"]
    with pytest.raises(ValueError):
        cohort.to_matrix_regression("matrix")
    value["dataset"]["y"]["values"][1][1] = 16777217
    with pytest.raises(ValueError, match="float32"):
        Dataset.from_dict(value).to_masked_matrix_regression("matrix")
    value["dataset"]["y"]["values"][1][1] = 3
    value["dataset"]["target_names"] = ["class_a", "class_a"]
    with pytest.raises(ValueError):
        Dataset.from_dict(value)
    value["dataset"]["target_names"] = ["class_a", "class_b"]
    value["dataset"]["target_mask"]["values"][0][1] = True
    with pytest.raises(ValueError):
        Dataset.from_dict(value)
    value["dataset"]["y"]["values"] = [[0,3],[1,3],[0,7],[1,3]]
    value["dataset"]["target_mask"]["values"] = [[True,True]] * 4
    cohort = Dataset.from_dict(value)
    assert cohort.to_masked_matrix_regression("matrix")["y"] == value["dataset"]["y"]["values"]
    with pytest.raises(ValueError, match="int64 target vector"):
        cohort.to_matrix_regression("matrix")
