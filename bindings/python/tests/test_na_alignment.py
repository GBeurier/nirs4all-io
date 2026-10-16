"""Exercise public Rust-backed loading, including physical blank records."""
import pytest

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
