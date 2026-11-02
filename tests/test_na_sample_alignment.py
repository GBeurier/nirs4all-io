"""Removal is a dataset operation: every sample-aligned value moves together."""
import numpy as np
import pytest

import nirs4all_io as nio


@pytest.mark.parametrize("target", ["10\n\n30\n40\n", "a\nNA\nb\na\n", "10;100\n20;NA\n30;300\n40;400\n"])
@pytest.mark.parametrize("partition", [None, "train", "test"])
def test_remove_target_samples_preserves_correspondence(tmp_path, target, partition):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n4\n")
    header = "y;z\n" if ";" in target else "y\n"
    (tmp_path / "y.csv").write_text(header + target)
    (tmp_path / "meta.csv").write_text("label\none\ntwo\nthree\nfour\n")
    sources = [
        {"id": "x", "role": "features", "input": "x.csv"},
        {"id": "y", "role": "targets", "input": "y.csv", "params": {"na": {"policy": "remove_sample"}}},
        {"id": "meta", "role": "metadata", "input": "meta.csv"},
    ]
    if partition:
        for source in sources:
            source["partition"] = partition
    block = nio.load({"sources": sources}, base_dir=tmp_path, target="assembled").blocks[partition or "train"]
    assert block.n_samples == 3
    np.testing.assert_array_equal(block.X[0].ravel(), [1, 3, 4])
    assert list(block.metadata["label"]) == ["one", "three", "four"]
    assert np.isfinite(block.y).all()
    if target.startswith("10"):
        np.testing.assert_array_equal(block.y[:, 0], [10, 30, 40])
    else:
        assert block.y_categorical["y"]["categories"] == ["a", "b"]


def test_remove_samples_unions_sources_and_variations(tmp_path):
    for name, contents in {
        "x": "w\n1\nNA\n3\n4\n5\n6\n",
        "x2": "z\n10\n20\n30\n40\n50\n60\n",
        "v": "w\n100\n200\nNA\n400\n500\n600\n",
        "y": "y\n1\n2\n3\nNA\n5\n6\n",
        "m": "label\na\nb\nc\nd\nNA\nf\n",
        "weights": "weight\n1\n2\n3\n4\n5\n6\n",
    }.items():
        (tmp_path / f"{name}.csv").write_text(contents)
    spec = {"params": {"na": {"policy": "remove_sample"}}, "sources": [
        {"id": "x", "role": "features", "input": "x.csv", "variations": [{"name": "processed", "input": "v.csv"}]},
        {"id": "x2", "role": "features", "input": "x2.csv"},
        {"id": "y", "role": "targets", "input": "y.csv"},
        {"id": "m", "role": "metadata", "input": "m.csv"},
        {"id": "weights", "role": "weights", "input": "weights.csv"},
    ]}
    block = nio.load(spec, base_dir=tmp_path, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.X[0].ravel(), [1, 6])
    np.testing.assert_array_equal(block.X[1].ravel(), [10, 60])
    np.testing.assert_array_equal(block.processings[0][0][1].ravel(), [100, 600])
    np.testing.assert_array_equal(block.y.ravel(), [1, 6])
    np.testing.assert_array_equal(block.weights, [1, 6])
    assert list(block.metadata["label"]) == ["a", "f"]


def test_remove_target_samples_after_key_join_reordering(tmp_path):
    (tmp_path / "x.csv").write_text("id;w\na;1\nb;2\nc;3\n")
    (tmp_path / "y.csv").write_text("id;y\nc;30\na;10\nb;NA\n")
    block = nio.load({"sources": [
        {"id": "x", "role": "features", "key": "id", "input": "x.csv"},
        {"id": "y", "role": "targets", "key": "id", "input": "y.csv", "params": {"na": {"policy": "remove_sample"}}},
    ]}, base_dir=tmp_path, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.X[0].ravel(), [1, 3])
    np.testing.assert_array_equal(block.y.ravel(), [10, 30])


def test_blank_target_is_not_silently_skipped_under_abort(tmp_path):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n")
    (tmp_path / "y.csv").write_text("y\n10\n\n30\n")
    with pytest.raises(ValueError, match="NA values detected"):
        nio.load({"params": {"na": {"policy": "abort"}}, "sources": [
            {"id": "x", "role": "features", "input": "x.csv"},
            {"id": "y", "role": "targets", "input": "y.csv"},
        ]}, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("policy", ["auto", "remove_feature", "replace"])
def test_missing_target_excludes_its_sample(tmp_path, policy):
    # A missing target is never imputed, dropped as a column or made a class.
    (tmp_path / "x.csv").write_text("w;v\n1;10\n2;20\n3;30\n4;40\n")
    (tmp_path / "y.csv").write_text("y;c\n10;a\n\n30;NA\n40;b\n")
    assembled = nio.load({"params": {"na": {"policy": policy}}, "sources": [
        {"id": "x", "role": "features", "input": "x.csv"},
        {"id": "y", "role": "targets", "input": "y.csv"},
    ]}, base_dir=tmp_path, target="assembled")
    block = assembled.blocks["train"]
    assert block.y_headers == ["y", "c"]
    np.testing.assert_array_equal(block.X[0][:, 0], [1, 4])
    np.testing.assert_array_equal(block.y[:, 0], [10, 40])
    assert block.y_categorical["c"]["categories"] == ["a", "b"]
    np.testing.assert_array_equal(block.y[:, 1], [0, 1])
    assert assembled.warnings == ["2 sample(s) with a missing target value were excluded; set na.policy to 'abort' to refuse them or 'ignore' to keep them as NaN"]


def test_ignore_keeps_missing_target_as_nan(tmp_path):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n")
    (tmp_path / "y.csv").write_text("y\n10\n\n30\n")
    block = nio.load({"params": {"na": {"policy": "ignore"}}, "sources": [
        {"id": "x", "role": "features", "input": "x.csv"},
        {"id": "y", "role": "targets", "input": "y.csv"},
    ]}, base_dir=tmp_path, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.y[:, 0], [10, np.nan, 30])


@pytest.mark.parametrize("policy", ["auto", "remove_feature", "replace"])
def test_feature_policy_skips_samples_excluded_for_their_target(tmp_path, policy):
    # One table holds X and y: a blank row is a missing target, so the sample is
    # excluded rather than aborting, dropping or filling its feature cells.
    (tmp_path / "data.csv").write_text("y;w;v\n10;1;5\n\n30;3;7\n")
    block = nio.load({"params": {"na": {"policy": policy}}, "sources": [{
        "id": "d", "role": "mixed", "input": "data.csv",
        "columns": [{"select": "y", "role": "targets"}, {"select": "rest", "role": "features"}],
    }]}, base_dir=tmp_path, target="assembled").blocks["train"]
    assert block.feature_headers == [["w", "v"]]
    np.testing.assert_array_equal(block.X[0], [[1, 5], [3, 7]])
    np.testing.assert_array_equal(block.y[:, 0], [10, 30])


def test_remove_feature_only_drops_feature_columns(tmp_path):
    (tmp_path / "x.csv").write_text("w;v\n1;10\n2;NA\n3;30\n")
    (tmp_path / "y.csv").write_text("y\n10\n20\n30\n")
    block = nio.load({"params": {"na": {"policy": "remove_feature"}}, "sources": [
        {"id": "x", "role": "features", "input": "x.csv"},
        {"id": "y", "role": "targets", "input": "y.csv"},
    ]}, base_dir=tmp_path, target="assembled").blocks["train"]
    assert block.feature_headers == [["w"]]
    np.testing.assert_array_equal(block.y[:, 0], [10, 20, 30])


def test_auto_still_aborts_on_missing_feature(tmp_path):
    (tmp_path / "x.csv").write_text("w\n1\nNA\n3\n")
    (tmp_path / "y.csv").write_text("y\n10\n20\n30\n")
    with pytest.raises(ValueError, match="First NA in column 'w' \\(row: 1\\)"):
        nio.load({"sources": [
            {"id": "x", "role": "features", "input": "x.csv"},
            {"id": "y", "role": "targets", "input": "y.csv"},
        ]}, base_dir=tmp_path, target="assembled")


def test_metadata_keeps_missing_values_under_the_dataset_policy(tmp_path):
    # nirs4all parity: metadata ignores the dataset NA policy unless it is remove_sample.
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n")
    (tmp_path / "y.csv").write_text("y\n1\n2\n3\n")
    (tmp_path / "m.csv").write_text("location;site\n;s1\nfield;\n;s3\n")
    sources = [
        {"id": "x", "role": "features", "input": "x.csv"},
        {"id": "y", "role": "targets", "input": "y.csv"},
        {"id": "m", "role": "metadata", "input": "m.csv"},
    ]
    block = nio.load({"params": {"na": {"policy": "abort"}}, "sources": sources}, base_dir=tmp_path, target="assembled").blocks["train"]
    assert block.n_samples == 3
    assert list(block.metadata["site"].isna()) == [False, True, False]
    (tmp_path / "x.csv").write_text("w\n1\nNA\n3\n")
    with pytest.raises(nio.SpecError, match="na_policy is 'abort'"):
        nio.load({"sources": sources}, base_dir=tmp_path, target="assembled")
