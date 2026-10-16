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


@pytest.mark.parametrize("policy", ["abort", "auto"])
def test_blank_target_is_not_silently_skipped_under_strict_policy(tmp_path, policy):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n")
    (tmp_path / "y.csv").write_text("y\n10\n\n30\n")
    with pytest.raises(ValueError, match="NA values detected"):
        nio.load({"params": {"na": {"policy": policy}}, "sources": [
            {"id": "x", "role": "features", "input": "x.csv"},
            {"id": "y", "role": "targets", "input": "y.csv"},
        ]}, base_dir=tmp_path, target="assembled")
