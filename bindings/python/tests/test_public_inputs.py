# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Native public inputs: no oracle assembly or parser retries."""

import json
import os

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


@pytest.mark.parametrize("storage", [None, {"consumer": "SpectroDataset"}])
def test_native_frames_direct_entry_enforces_budget_before_copy(storage):
    from nirs4all_io._native import assemble_frames

    spec = {"sources": [{"id": "x", "role": "features", "input": "x"}]}
    frames = [{"name": "x", "columns": ["a"], "rows": [["longer"]]}]
    with pytest.raises(ValueError, match="field.*limit"):
        assemble_frames(spec, frames, limits={"max_field_bytes": 2}, storage=storage)


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


@pytest.mark.parametrize("code", [-1, 0.5, 2, float("inf")])
def test_string_target_kind_refuses_numeric_code_carriers(code):
    from nirs4all_io._adapter import _decode_targets
    from test_partition_projection import matrix

    # V3 carries string labels directly; old numeric code carriers are refused.
    target = {"columns": [{"name": "label", "kind": "classification_string", "block": matrix([code], 1, 1)}]}
    with pytest.raises(ValueError, match="string-label"):
        _decode_targets(target)


def test_categorical_codebook_preserves_missing_targets():
    import struct

    from nirs4all_io._adapter import _decode_targets

    values = b"".join(struct.pack("<Q", len(value)) + value for value in [b"a", b"", b"b"])
    block = {"dtype": "string-label", "n_rows": 3, "n_cols": 1,
             "data_bytes_hex": values.hex(), "validity": [True, False, True]}
    restored = _decode_targets({"columns": [{"name": "label", "kind": "classification_string", "block": block}]})
    assert restored[0, 0] == "a" and restored[2, 0] == "b"
    assert np.isnan(restored[1, 0])


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


@pytest.mark.parametrize("via", ["base_dir", "pair", "yaml"])
def test_path_reference_public_pattern_under_literal_base(tmp_path, via):
    import glob
    from pathlib import Path
    base = tmp_path / "run[2]"
    base.mkdir()
    peer = tmp_path / "run2"
    peer.mkdir()
    for directory, value in [(base, "0.1"), (peer, "7.1")]:
        (directory / "part-a.csv").write_text(f"x;y\n{value};1.5\n")
    spec = {"sources": [{"id": "x", "role": "features", "input": "part-*.csv", "merge": "concat_samples"}]}
    if via == "base_dir":
        from nirs4all_io import _absolutize_input_ref
        actual = _absolutize_input_ref("part-*.csv", base)
    elif via == "pair":
        class Provider:
            def to_io_spec(self):
                return spec, base
        actual = nio.to_spec(Provider()).sources[0]["input"]
    else:
        import yaml
        config = base / "dataset.yaml"
        config.write_text(yaml.safe_dump(spec))
        actual = nio.to_spec(config).sources[0]["input"]
    assert [Path(p) for p in glob.glob(actual)] == [base / "part-a.csv"]


@pytest.mark.skipif(os.name != "posix", reason="literal wildcard filenames require Unix")
def test_path_reference_public_exact_wildcard_name_is_literal(tmp_path):
    from nirs4all_io import _absolutize_input_ref
    literal = tmp_path / "part-*.csv"
    literal.write_text("x\n0.1\n")
    (tmp_path / "part-a.csv").write_text("x\n7.1\n")
    assert _absolutize_input_ref("part-*.csv", tmp_path) == str(literal)


@pytest.mark.parametrize("via", ["base_dir", "pair", "yaml"])
def test_path_reference_public_dotted_scoped_pattern(tmp_path, monkeypatch, via):
    import glob
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    base = Path("./run[2]")
    base.mkdir()
    Path("run2").mkdir()
    (base / "part-a.csv").write_text("x;y\n0.1;1.5\n")
    Path("run2/part-a.csv").write_text("x;y\n7.1;9.5\n")
    spec = {"sources": [{"id": "x", "role": "features", "input": "part-*.csv", "merge": "concat_samples"}]}
    if via == "base_dir":
        from nirs4all_io import _absolutize_input_ref
        actual = _absolutize_input_ref("part-*.csv", base)
    elif via == "pair":
        class Provider:
            def to_io_spec(self):
                return spec, base
        actual = nio.to_spec(Provider()).sources[0]["input"]
    else:
        import yaml
        config = base / "dataset.yaml"
        config.write_text(yaml.safe_dump(spec))
        actual = nio.to_spec(config).sources[0]["input"]
    selected = [Path(path) for path in glob.glob(actual)]
    assert [path.resolve() for path in selected] == [(base / "part-a.csv").resolve()]
    assert selected[0].read_text() == "x;y\n0.1;1.5\n"


@pytest.fixture
def source_array_transport(monkeypatch):
    """Spy on transport only; no native selection or assembly is simulated."""
    from copy import deepcopy

    from test_partition_projection import packet

    from nirs4all_io import _inputs, _native

    policy = {"max_rows": 100, "max_columns": 100, "max_cells": 1000,
              "max_field_bytes": 1024, "max_decoded_total_bytes": 100000, "max_files": 10}
    # Explicit unit limits exercise the real Python admission, not the native resolver.
    monkeypatch.setattr(_inputs, "resolved_load_limits", lambda limits: {**policy, **(limits or {})})
    state = {"calls": [], "full": packet()}

    def transport(spec, frames, **options):
        state["calls"].append({"spec": deepcopy(spec), "frames": deepcopy(frames), "options": deepcopy(options)})
        return deepcopy(state["full"])

    monkeypatch.setattr(_native, "assemble_frames", transport)
    return state


def test_source_array_public_consumer_options_and_default_routes(source_array_transport):
    from test_partition_projection import Recorder

    x = np.array([[1., -0.], [3., np.nan]])
    y = np.array([.25, np.nan])
    state = source_array_transport
    full = state["full"]
    full["blocks"] = {"train": full["blocks"]["train"]}
    full["partition_order"] = ["train"]
    limits = {"max_cells": 20}
    ds = nio.load((x, y), target="spectrodataset", spectro_dataset_cls=Recorder, limits=limits)
    call = state["calls"][-1]
    assert call["options"] == {"limits": limits, "summary": False, "storage": {"consumer": "SpectroDataset"}}
    assert call["frames"] == [
        {"name": "array_features", "columns": ["0", "1"], "rows": [[1., -0.], [3., None]]},
        {"name": "array_targets", "columns": ["y0"], "rows": [[.25], [None]]},
    ]
    assert np.signbit(call["frames"][0]["rows"][0][1])
    assert call["spec"]["params"] == {"na": {"policy": "ignore"}}
    assert ds.samples[0][1].dtype == np.float32
    assert ds.targets[0].dtype == np.float64
    assert np.isnan(ds.samples[0][1][1, 1]) and np.isnan(ds.targets[0][1, 0])
    # Raw/default routes do not receive a consumer request.
    assert nio.load((x, y), target="assembled")["assembled_schema_version"] == 3
    package = nio.load((x, y), target="package")
    assert package.to_assembled().blocks["train"].n_samples == 2
    assert [call["options"] for call in state["calls"][1:]] == [
        {"limits": None, "summary": True}, {"limits": None, "summary": False}]


def test_source_array_public_metadata_identity_is_forwarded(source_array_transport):
    from test_partition_projection import Recorder

    state = source_array_transport
    full = state["full"]
    full["blocks"] = {"train": full["blocks"]["train"]}
    full["partition_order"] = ["train"]
    metadata = pd.DataFrame({"sid": ["train0", "train1"]})
    ds = nio.load({"X": np.array([[1., -0.], [3., np.nan]]), "y": np.array([.25, np.nan]), "metadata": metadata},
                  target="spectrodataset", spectro_dataset_cls=Recorder)
    assert state["calls"][0]["frames"][2] == {"name": "array_metadata", "columns": ["sid"], "rows": [["train0"], ["train1"]]}
    assert ds.metadata[0]["sid"].tolist() == ["train0", "train1"]


def _declared_memory_spec_frames():
    """Existing raw-frame contract: deterministic partitions and inline folds."""
    spec = {"task_type": "regression", "params": {"na": {"policy": "ignore"}}, "sources": [{"id": "nir", "role": "mixed", "input": "x", "key": "sid",
            "columns": [{"role": "features", "select": ["a", "b"]}, {"role": "targets", "select": ["y"]},
                        {"role": "metadata", "select": ["sid"]}]}],
            "partitions": {"by": "index", "train": [0, 1], "test": [2, 3], "val": [4, 5], "predict": [6, 7]},
            "sample_index": {"by": "id", "key": "sid"}, "folds": {"inline": [{"train": [0], "val": [1]}]}}
    rows = [[1., -0., .25, "train0"], [3., None, None, "train1"],
            [11., -0., 10.25, "test0"], [13., None, None, "test1"],
            [21., -0., 20.25, "val0"], [23., None, None, "val1"],
            [31., -0., 30.25, "predict0"], [33., None, None, "predict1"]]
    return spec, [{"name": "x", "columns": ["a", "b", "y", "sid"], "rows": rows}]


def test_source_memory_declared_spec_keeps_four_partitions_ids_and_folds(source_array_transport, monkeypatch):
    from copy import deepcopy

    from test_partition_projection import Recorder

    from nirs4all_io import _inputs

    spec, frames = _declared_memory_spec_frames()
    for block in source_array_transport["full"]["blocks"].values():
        block["weights"] = None
        block["processings"] = [[]]
    # This is a declared-spec transport test, not new NumPy tuple syntax.
    monkeypatch.setattr(_inputs, "array_payload", lambda *args: (deepcopy(spec), deepcopy(frames)))
    ds = nio.load(object(), target="spectrodataset", spectro_dataset_cls=Recorder)
    call = source_array_transport["calls"][0]
    assert call["spec"] == spec and call["frames"] == frames
    assert [part for part, _ in ds.samples] == ["train", "test", "val", "predict"]
    assert [len(values) for _, values in ds.samples] == [2, 2, 2, 2]
    assert len(ds.metadata) == len(ds.targets) == 1
    assert ds.metadata[0]["sid"].tolist() == [part + suffix for part in ["train", "test", "val", "predict"] for suffix in ["0", "1"]]
    np.testing.assert_equal(ds.targets[0].reshape(-1), [.25, np.nan, 10.25, np.nan, 20.25, np.nan, 30.25, np.nan])
    assert ds.folds == [([0], [1])]


@pytest.mark.parametrize("limits", [{"max_cells": 1}, {"max_rows": 1}, {"max_columns": 1},
                                    {"max_decoded_total_bytes": 1}, {"max_files": 1}, {"max_field_bytes": 1}])
def test_source_array_public_limits_refuse_before_native_transport(source_array_transport, limits):
    x = np.ones((2, 2))
    y = np.ones(2)
    metadata = pd.DataFrame({"sid": ["long", "text"]})
    with pytest.raises(ValueError, match="budget"):
        nio.load({"X": x, "y": y, "metadata": metadata}, target="spectrodataset", limits=limits)
    assert source_array_transport["calls"] == []


@pytest.mark.parametrize("inp", [(np.ones((2, 2)), np.ones(3)),
                                  (np.ones((2, 2)), np.ones(2), np.array(["train", "val"]))])
def test_source_array_public_alignment_and_existing_split_vocabulary(source_array_transport, inp):
    with pytest.raises(ValueError):
        nio.load(inp, target="spectrodataset")
    assert source_array_transport["calls"] == []


def test_source_array_public_sdk_constructor_matches_direct(source_array_transport):
    """Real SDK constructor, with explicitly source-only native transport."""
    import inspect
    from pathlib import Path

    pytest.importorskip("polars", reason="Real SDK constructor unavailable without polars")
    from nirs4all.data import SpectroDataset

    origin = str(Path(inspect.getfile(SpectroDataset)).resolve())
    if os.environ.get("L2_SDK_EXPECTED"):
        assert origin == os.environ["L2_SDK_EXPECTED"]
    state = source_array_transport
    full = state["full"]
    del full["blocks"]["val"]
    full["partition_order"] = ["train", "test", "predict"]
    full["folds"] = []
    for block in full["blocks"].values():
        block["metadata"] = None
        block["weights"] = None
        block["processings"] = [[]]
        block["feature_headers"] = [["0", "1"]]
    split = np.array(["test", "train", "predict", "train", "test", "predict"])
    x = np.array([[11., -0.], [1., -0.], [31., -0.], [3., np.nan], [13., np.nan], [33., np.nan]], dtype=np.float64)
    y = np.array([10.25, .25, 30.25, np.nan, np.nan, np.nan], dtype=np.float64)
    projected = nio.load((x, y, split), target="spectrodataset")
    call = state["calls"][0]
    assert call["options"]["storage"] == {"consumer": "SpectroDataset"}
    assert call["spec"]["partitions"] == {"by": "index", "train": [1, 3], "test": [0, 4], "predict": [2, 5]}
    np.testing.assert_array_equal(np.asarray(call["frames"][0]["rows"], dtype=float), x)
    direct = SpectroDataset(name=full["name"])
    for part in ["train", "test", "predict"]:
        rows = split == part
        # The direct SDK receives the same independent original arrays.
        direct.add_samples(x[rows], {"partition": part}, headers=["0", "1"], header_unit="text")
        direct.add_targets(y[rows].reshape(-1, 1))
    direct.set_task_type("regression", forced=True)
    for part in ["train", "test", "predict"]:
        selector = {"partition": part}
        left, right = np.asarray(projected.x(selector, layout="3d")), np.asarray(direct.x(selector, layout="3d"))
        assert left.shape == right.shape == (2, 1, 2)
        assert left.dtype == right.dtype == np.float32 and left.tobytes() == right.tobytes()
        assert np.isnan(left).tolist() == np.isnan(right).tolist()
        yl, yr = np.asarray(projected.y(selector)), np.asarray(direct.y(selector))
        assert yl.dtype == yr.dtype and yl.tobytes() == yr.tobytes()
    print(json.dumps({"real_sdk_origin": origin, "source_only_native_transport": True,
                      "counts": {part: 2 for part in ["train", "test", "predict"]}}))


def test_native_memory_storage_consumer_and_explicit_request():
    """Requires the future coherent compiled native gate; never run via a spy."""
    from nirs4all_io._adapter import _matrix
    from nirs4all_io._native import assemble_frames

    spec = {"params": {"na": {"policy": "ignore"}}, "sources": [{"id": "x", "role": "features", "input": "x"}]}
    frames = [{"name": "x", "columns": ["a"], "rows": [[1 + 2**-30], [None]]}]
    default = assemble_frames(spec, frames)["blocks"]["train"]["x"][0]
    consumer = assemble_frames(spec, frames, storage={"consumer": "SpectroDataset"})["blocks"]["train"]["x"][0]
    requested = assemble_frames(spec, frames, storage={"features": "float64"})["blocks"]["train"]["x"][0]
    assert default["dtype"] == requested["dtype"] == "float64"
    assert _matrix(default)[0, 0] == _matrix(requested)[0, 0] == 1 + 2**-30
    assert consumer["dtype"] == "float32" and consumer["precision"]["consumer"] == "SpectroDataset"
    assert _matrix(consumer)[0, 0] == 1. and np.isnan(_matrix(consumer)[1, 0])
    with pytest.raises(ValueError, match="float64"):
        assemble_frames(spec, frames, storage={"consumer": "SpectroDataset", "features": "float64"})



def test_native_memory_declared_partitions_identity_and_folds():
    """Compiled transport control retained for the later native wheel gate."""
    from nirs4all_io._native import assemble_frames
    from test_partition_projection import Recorder

    spec, frames = _declared_memory_spec_frames()
    full = assemble_frames(spec, frames, storage={"consumer": "SpectroDataset"})
    ds = nio.to_spectrodataset(full, spectro_dataset_cls=Recorder)
    assert [part for part, _ in ds.samples] == ["train", "test", "val", "predict"]
    assert [len(values) for _, values in ds.samples] == [2, 2, 2, 2]
    for index, (part, values) in enumerate(ds.samples):
        assert values.dtype == np.float32 and values[0, 0] == 1 + 10 * index
        assert np.signbit(values[0, 1]) and np.isnan(values[1, 1])
        assert len(ds.metadata) == len(ds.targets) == 1
        assert ds.metadata[0]["sid"].tolist()[2 * index:2 * index + 2] == [part + "0", part + "1"]
        assert ds.targets[0].dtype == np.float64 and np.isnan(ds.targets[0][2 * index + 1, 0])
    assert ds.folds == [([0], [1])]


def test_native_projection_selection_roundtrip_and_package_sdk_route():
    """Prepared for a coherent native build; UNRUN in source-only qualification."""
    from nirs4all_io._adapter import _matrix
    from nirs4all_io._native import select_spectrodataset_features
    from nirs4all_io._package import DatasetPackage
    from test_partition_projection import Recorder, matrix, packet

    original = matrix([1.5, -0., 0., float(np.nextafter(np.float32(0), np.float32(1)))], 2, 2, validity=[True, True, False, True])
    selected = select_spectrodataset_features(original)
    values = _matrix(selected)
    assert selected["dtype"] == "float32"
    assert selected["precision"] == {"dtype": "float32", "origin": "consumer", "source_dtype": None, "requested_dtype": None, "consumer": "SpectroDataset"}
    assert values.dtype == np.float32 and np.signbit(values[0, 1]) and np.isnan(values[1, 0])
    assert values[1, 1].tobytes() == bytes.fromhex("01000000")
    full = packet()
    full["blocks"]["train"]["x"] = [original]
    ds = nio.load(DatasetPackage(full), target="spectrodataset", spectro_dataset_cls=Recorder)
    np.testing.assert_equal(ds.samples[0][1], values)
    assert original["dtype"] == "float64" and original["precision"]["consumer"] is None
    for value, message in [(1e300, "overflows float32"), (1e-320, "rounds to zero")]:
        with pytest.raises(ValueError, match=message):
            select_spectrodataset_features(matrix([value], 1, 1))
    original["precision"].update(origin="requested", requested_dtype="float64")
    with pytest.raises(ValueError, match="float64"):
        select_spectrodataset_features(original)


def test_native_projection_wire_admission_and_owner_errors():
    """Prepared raw PyO3 boundary checks, including admission before depythonize."""
    from copy import deepcopy

    from nirs4all_io._native import select_spectrodataset_features
    from test_partition_projection import matrix

    good = matrix([1.5], 1, 1)
    variants = [
        ("data_bytes_hex", "000000000000f8"), ("data_bytes_hex", "000000000000f83g"),
        ("validity", [1]), ("validity", []), ("dtype", "float32"),
        ("n_rows", True), ("n_rows", -1), ("n_rows", 1.0), ("n_rows", 2**64 - 1),
    ]
    for field, value in variants:
        bad = deepcopy(good)
        bad[field] = value
        with pytest.raises((ValueError, TypeError, OverflowError)):
            select_spectrodataset_features(bad)
    bad = deepcopy(good)
    bad["precision"]["source_dtype"] = [object()]
    with pytest.raises(TypeError):
        select_spectrodataset_features(bad)
    bad = deepcopy(good)
    bad["n_rows"] = 10_000_001
    # A non-serializable payload is deliberately behind an already refused shape.
    bad["data_bytes_hex"] = object()
    with pytest.raises(ValueError, match="limit"):
        select_spectrodataset_features(bad)


# Independent little-endian literals from the adopted G1 cell contract. These
# tests require the newly compiled endpoint; no reference owner is injected.
@pytest.mark.parametrize(
    "dtype,cell,expected_hex",
    [
        ("float64", 0.0, "0000000000000000"),
        ("float64", -0.0, "0000000000000080"),
        ("float64", 18014398509481984, "0000000000005043"),
        ("float64", 9007199254740993.0, "0000000000004043"),
        ("float64", -(2**63), "000000000000e0c3"),
        ("float64", 2**63, "000000000000e043"),
        ("float64", 5e-324, "0100000000000000"),
        ("float32", 2**54 + 2**30 + 1, "0100805a"),
        ("float32", 16777217, "0000804b"),
        ("float32", 2**64 - 1, "0000805f"),
        ("float32", 1.401298464324817e-45, "01000000"),
        ("float32", 0.0, "00000000"),
        ("float32", -0.0, "00000080"),
        ("float32", 1.5, "0000c03f"),
    ],
)
def test_declared_float_native_callback_and_array_literal_bits(dtype, cell, expected_hex):
    import struct

    from nirs4all_io._native import declared_float_cell

    from nirs4all_io.dataset_facade import _declared_float_cells
    from nirs4all_io.multimodal import _array_from_dict

    record = {"dtype": dtype, "shape": [1, 1], "values": [[cell]]}
    checked = declared_float_cell(cell, dtype)
    normalized = _declared_float_cells(record)
    fmt, little_dtype = ("<f", "<f4") if dtype == "float32" else ("<d", "<f8")
    assert struct.pack(fmt, checked).hex() == expected_hex
    assert struct.pack(fmt, normalized["values"][0][0]).hex() == expected_hex
    array = _array_from_dict(record, "literal")
    assert array.dtype == np.dtype(dtype) and array.shape == (1, 1)
    assert array.astype(little_dtype).tobytes().hex() == expected_hex
    assert normalized is not record and normalized["values"] is not record["values"]
    assert normalized["values"][0] is not record["values"][0]
    assert record["values"][0][0] is cell and normalized["dtype"] == dtype
    if dtype == "float64":
        assert type(checked) is type(cell) and type(normalized["values"][0][0]) is type(cell)
        assert checked is cell and normalized["values"][0][0] is cell
    else:
        assert type(checked) is float and type(normalized["values"][0][0]) is float


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cell", [2**64 + 1, 10**400, -(2**63) - 1])
def test_declared_float_integer_domain_refuses_before_any_float_retry(dtype, cell):
    from nirs4all_io._native import declared_float_cell

    from nirs4all_io.dataset_facade import _declared_float_cells
    from nirs4all_io.multimodal import _array_from_dict

    record = {"dtype": dtype, "shape": [1], "values": [cell]}
    # All are invalid integers even where a rounded binary64 would be finite.
    with pytest.raises(ValueError, match="i64/u64 domain"):
        declared_float_cell(cell, dtype)
    with pytest.raises(ValueError, match="i64/u64 domain"):
        _declared_float_cells(record)
    with pytest.raises(ValueError, match="i64/u64 domain"):
        _array_from_dict(record, "domain")
    assert record["values"] == [cell] and type(record["values"][0]) is int


@pytest.mark.parametrize(
    "dtype,cell,message",
    [
        ("float64", 2**53 + 1, "not exactly representable in float64"),
        ("float64", 2**64 - 1, "not exactly representable in float64"),
        ("float32", 5e-324, "rounds to zero in float32"),
        ("float32", 1e40, "overflows float32"),
    ],
)
def test_declared_float_owner_refusals_survive_all_three_boundaries(dtype, cell, message):
    from nirs4all_io._native import declared_float_cell

    from nirs4all_io.dataset_facade import _declared_float_cells
    from nirs4all_io.multimodal import _array_from_dict

    record = {"dtype": dtype, "shape": [1], "values": [cell]}
    with pytest.raises(ValueError, match=message):
        declared_float_cell(cell, dtype)
    with pytest.raises(ValueError, match=message):
        _declared_float_cells(record)
    with pytest.raises(ValueError, match=message):
        _array_from_dict(record, "refusal")


@pytest.mark.parametrize("cell", [True, False, "1", None, object(), np.int64(1), np.float64(1)])
def test_declared_float_native_rejects_inappropriate_types_explicitly(cell):
    from nirs4all_io._native import declared_float_cell

    with pytest.raises(TypeError, match="exact builtin int or finite float"):
        declared_float_cell(cell, "float64")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cell", [float("nan"), float("inf"), -float("inf")])
def test_declared_float_native_requires_finite_floats(dtype, cell):
    from nirs4all_io._native import declared_float_cell

    with pytest.raises(ValueError, match="Observed numeric values must be finite"):
        declared_float_cell(cell, dtype)


@pytest.mark.parametrize("dtype", ["double", "<f8", ">f4", "float16", "int64"])
def test_declared_float_native_requires_canonical_dtype_name(dtype):
    from nirs4all_io._native import declared_float_cell

    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        declared_float_cell(1, dtype)


def test_declared_float_native_does_not_coerce_numeric_subclasses():
    from nirs4all_io._native import declared_float_cell

    from nirs4all_io.dataset_facade import _declared_float_cells

    class Integer(int):
        def __float__(self):
            raise AssertionError("no float coercion permitted")

    class Real(float):
        def __float__(self):
            raise AssertionError("no float coercion permitted")

    for cell in (Integer(1), Real(1.0)):
        with pytest.raises(TypeError, match="exact builtin"):
            declared_float_cell(cell, "float64")
        record = {"dtype": "float64", "shape": [1], "values": [cell]}
        assert _declared_float_cells(record)["values"][0] is cell


@pytest.mark.parametrize(
    "dtype,width,cell,expected_hex",
    [
        ("<f8", 64, 18014398509481984, "0000000000005043"),
        (">f4", 32, 16777217, "0000804b"),
        ("double", 64, -0.0, "0000000000000080"),
        ("single", 32, 16777217, "0000804b"),
        ("f", 32, -0.0, "00000080"),
    ],
)
def test_declared_float_alias_and_endian_metadata_are_retained(dtype, width, cell, expected_hex):
    from nirs4all_io.dataset_facade import _declared_float_cells, _float_width
    from nirs4all_io.multimodal import _array_from_dict

    record = {"dtype": dtype, "shape": [1], "values": [cell]}
    normalized = _declared_float_cells(record)
    assert _float_width(dtype) == width and normalized["dtype"] == dtype
    assert record["dtype"] == dtype and record["values"][0] is cell
    array = _array_from_dict(record, "alias")
    assert array.dtype == np.dtype(dtype)
    assert array.astype("<f4" if width == 32 else "<f8").tobytes().hex() == expected_hex
    if width == 64:
        assert type(normalized["values"][0]) is type(cell)


@pytest.mark.parametrize("dtype", ["float16", "int64", "bool", "object", "U4", "not-a-dtype"])
def test_declared_float_unrecognized_width_or_nonfloat_record_is_unchanged(dtype):
    from nirs4all_io.dataset_facade import _declared_float_cells, _float_width

    record = {"dtype": dtype, "shape": [1], "values": [16777217]}
    assert _float_width(dtype) is None
    assert _declared_float_cells(record) is record


@pytest.mark.parametrize("values", [None, 1, (1,), "1", {"value": 1}])
def test_declared_float_nonlist_record_is_unchanged(values):
    from nirs4all_io.dataset_facade import _declared_float_cells

    record = {"dtype": "float64", "shape": [1], "values": values}
    assert _declared_float_cells(record) is record


@pytest.mark.parametrize("cell", [float("nan"), float("inf"), -float("inf"),
                                 {"nonfinite": "nan"}, {"nonfinite": "inf"}, {"nonfinite": "-inf"},
                                 True, False, "1", None, object(), np.int64(1), np.float64(1.0)])
def test_declared_float_callback_leaves_non_dispatched_cells_untouched(monkeypatch, cell):
    from nirs4all_io import _native
    from nirs4all_io.dataset_facade import _declared_float_cells

    def forbidden(*args):
        raise AssertionError("callback must not dispatch this cell")

    monkeypatch.setattr(_native, "declared_float_cell", forbidden)
    record = {"dtype": "float32", "shape": [1, 1], "values": [[cell]]}
    normalized = _declared_float_cells(record)
    assert normalized["values"][0][0] is cell and record["values"][0][0] is cell
    assert normalized["values"] is not record["values"] and normalized["values"][0] is not record["values"][0]


@pytest.mark.parametrize("cell", [float("nan"), float("inf"), -float("inf"), True, "1", None, object()])
def test_declared_float_decoder_keeps_its_nonfinite_and_cell_type_refusals(cell):
    from nirs4all_io.multimodal import _array_from_dict

    with pytest.raises(ValueError):
        _array_from_dict({"dtype": "float32", "shape": [1], "values": [cell]}, "cell")


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_declared_float_decoder_still_decodes_tagged_nonfinite(token):
    from nirs4all_io.dataset_facade import _declared_float_cells
    from nirs4all_io.multimodal import _array_from_dict

    tagged = {"nonfinite": token}
    record = {"dtype": "float32", "shape": [1], "values": [tagged]}
    assert _declared_float_cells(record)["values"][0] is tagged
    result = _array_from_dict(record, "tagged")
    assert result.dtype == np.dtype("float32")
    assert {"nan": np.isnan, "inf": np.isposinf, "-inf": np.isneginf}[token](result[0])


def test_declared_float_list_copy_dispatches_exact_cells_without_mutating_caller(monkeypatch):
    from copy import deepcopy

    from nirs4all_io import _native
    from nirs4all_io.dataset_facade import _declared_float_cells

    original = {"dtype": ">f4", "shape": [2, 2], "values": [[16777217, -0.0], [1, None]]}
    before = deepcopy(original)
    calls = []
    owner = _native.declared_float_cell

    def observed(cell, dtype):
        calls.append((cell, dtype))
        return owner(cell, dtype)

    monkeypatch.setattr(_native, "declared_float_cell", observed)
    normalized = _declared_float_cells(original)
    assert original == before and type(original["values"][0][0]) is int
    assert normalized["values"] == [[16777216.0, -0.0], [1.0, None]] and normalized["dtype"] == ">f4"
    assert calls == [(16777217, "float32"), (-0.0, "float32"), (1, "float32")]
    assert np.signbit(normalized["values"][0][1])
    assert normalized["values"] is not original["values"]
    assert all(new is not old for new, old in zip(normalized["values"], original["values"], strict=True))


def test_declared_float_gate_does_not_add_rank_eight_admission():
    from nirs4all_io.multimodal import _array_from_dict

    values = 1
    for _ in range(9):
        values = [values]
    array = _array_from_dict({"dtype": "float32", "shape": [1] * 9, "values": values}, "rank9")
    assert array.shape == (1,) * 9 and array.tobytes().hex() == "0000803f"


def test_declared_float_mask_normalizes_false_target_before_native_callback(monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from nirs4all_io import _native
    from nirs4all_io.public_dataset import Dataset

    record = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/public-dataset-v2.json").read_text())
    record["dataset"]["y"]["dtype"] = "float32"
    record["dataset"]["y"]["values"][0][1] = 1e99
    assert record["dataset"]["target_mask"]["values"][0][1] is False
    before = deepcopy(record)
    seen = []
    owner = _native.declared_float_cell

    def observed(cell, dtype):
        seen.append(cell)
        return owner(cell, dtype)

    monkeypatch.setattr(_native, "declared_float_cell", observed)
    assert Dataset.from_dict(record).to_masked_matrix_regression("matrix")["y"][0][1] == 0.0
    assert record == before and 1e99 not in seen and 0.0 in seen
    seen.clear()
    record["dataset"]["target_mask"]["values"][0][1] = True
    with pytest.raises(ValueError, match="overflows float32"):
        Dataset.from_dict(record)
    assert 1e99 in seen


def test_declared_float_gate_does_not_reinterpret_typed_host_or_pickle(monkeypatch):
    import pickle

    from nirs4all_io import _native
    from nirs4all_io.multimodal import TensorSource

    def forbidden(*args):
        raise AssertionError("typed host and pickle routes must not reinterpret JSON integers")

    monkeypatch.setattr(_native, "declared_float_cell", forbidden)
    values = np.array([[16777217, -0.0]], dtype=np.float32)
    source = TensorSource(values, ["sample"], representation_id="tabular_numeric")
    restored = pickle.loads(pickle.dumps(source))
    assert source.values.dtype == restored.values.dtype == np.dtype("float32")
    assert source.values.tobytes().hex() == restored.values.tobytes().hex() == "0000804b00000080"
