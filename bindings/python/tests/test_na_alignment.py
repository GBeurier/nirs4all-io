"""Exercise public Rust-backed loading, including physical blank records."""
import numpy as np
import pytest
from nirs4all_io._adapter import to_spectrodataset

import nirs4all_io as nio


@pytest.mark.parametrize("target", ["10\n\n30\n40\n", "a\nNA\nb\na\n", "10;100\n20;NA\n30;300\n40;400\n"])
@pytest.mark.parametrize("partition", ["train", "test"])
def test_removing_target_rows_keeps_original_spectral_samples(tmp_path, target, partition):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n4\n")
    (tmp_path / "y.csv").write_text(("y;z\n" if ";" in target else "y\n") + target)
    spec = {"sources": [
        {"id": "x", "role": "features", "input": str(tmp_path / "x.csv"), "partition": partition},
        {"id": "y", "role": "targets", "input": str(tmp_path / "y.csv"), "partition": partition,
         "params": {"na": {"policy": "remove_sample"}}},
    ]}
    block = nio.assembled_full(spec)["blocks"][partition]
    assert block["n_samples"] == 3
    assert block["x"][0]["data"] == [1.0, 3.0, 4.0]
    assert block["y"]["n_rows"] == 3


def _xy_spec(tmp_path, policy):
    (tmp_path / "x.csv").write_text("w\n1\n2\n3\n")
    (tmp_path / "y.csv").write_text("y\n10\n\n30\n")
    return {"params": {"na": {"policy": policy}}, "sources": [
        {"id": "x", "role": "features", "input": str(tmp_path / "x.csv")},
        {"id": "y", "role": "targets", "input": str(tmp_path / "y.csv")},
    ]}


class _TargetRecorder:
    def __init__(self, name):
        self.targets = []

    def add_samples(self, *args, **kwargs):
        pass

    def add_targets(self, y):
        self.targets.append(y)

    def set_signal_type(self, *args, **kwargs):
        pass


def test_missing_target_is_none_on_the_wire_and_nan_in_the_dataset(tmp_path):
    full = nio.assembled_full(_xy_spec(tmp_path, "ignore"))
    assert full["blocks"]["train"]["y"]["data"] == [10.0, None, 30.0]
    ds = to_spectrodataset(full, spectro_dataset_cls=_TargetRecorder)
    np.testing.assert_array_equal(ds.targets[0][:, 0], [10.0, np.nan, 30.0])


def test_missing_target_excludes_its_sample_by_default(tmp_path):
    full = nio.assembled_full(_xy_spec(tmp_path, "auto"))
    block = full["blocks"]["train"]
    assert block["x"][0]["data"] == [1.0, 3.0]
    assert block["y"]["data"] == [10.0, 30.0]
    assert len(full["warnings"]) == 1
