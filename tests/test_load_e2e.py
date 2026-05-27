# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""End-to-end load() tests: directory conventions, combined files, flagship lookup,
multi-source, in-memory, and the lazy SpectroDataset adapter (via a recording double)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import nirs4all_io as nio


# --------------------------------------------------------------------------- #
# Recording SpectroDataset double (lets us test the adapter without nirs4all)  #
# --------------------------------------------------------------------------- #
class FakeSpectroDataset:
    def __init__(self, name="ds"):
        self.name = name
        self.samples = []  # (partition, X)
        self.targets = []
        self.metadata = []
        self.signal_types = []
        self.task_type = None
        self.folds = None
        self.repetition = None

    def add_samples(self, data, indexes, headers=None, header_unit=None):
        self.samples.append((indexes.get("partition"), data, headers, header_unit))

    def add_targets(self, y):
        self.targets.append(np.asarray(y))

    def add_metadata(self, data, headers=None):
        self.metadata.append(data)

    def set_signal_type(self, sig, src=0, forced=True):
        self.signal_types.append((sig, src, forced))

    def set_task_type(self, t, forced=True):
        self.task_type = t

    def set_folds(self, folds):
        self.folds = list(folds)

    def set_repetition(self, col):
        self.repetition = col

    def set_aggregate(self, v):
        self.aggregate = v

    def set_aggregate_method(self, m):
        self.aggregate_method = m

    def set_aggregate_exclude_outliers(self, v, t=0.95):
        self.aggregate_outliers = (v, t)


def _csv(path, df, sep=";"):
    df.to_csv(path, sep=sep, index=False)


# --------------------------------------------------------------------------- #
# Directory conventions (Xcal/Ycal/Xval/Yval)                                 #
# --------------------------------------------------------------------------- #
def test_load_classic_folder_to_assembled(tmp_path):
    _csv(tmp_path / "Xcal.csv", pd.DataFrame(np.random.rand(6, 4), columns=["400", "401", "402", "403"]))
    _csv(tmp_path / "Ycal.csv", pd.DataFrame({"y": np.arange(6.0)}))
    _csv(tmp_path / "Xval.csv", pd.DataFrame(np.random.rand(3, 4), columns=["400", "401", "402", "403"]))
    _csv(tmp_path / "Yval.csv", pd.DataFrame({"y": np.arange(3.0)}))
    asm = nio.load(tmp_path, target="assembled")
    assert set(asm.blocks) == {"train", "test"}
    assert asm.blocks["train"].X[0].shape == (6, 4)
    assert asm.blocks["test"].X[0].shape == (3, 4)
    assert asm.blocks["train"].y.shape == (6, 1)


def test_load_classic_folder_to_fake_spectrodataset(tmp_path):
    _csv(tmp_path / "Xcal.csv", pd.DataFrame(np.random.rand(5, 3), columns=["a", "b", "c"]))
    _csv(tmp_path / "Ycal.csv", pd.DataFrame({"y": np.arange(5.0)}))
    ds = nio.load(tmp_path, target="spectrodataset", spectro_dataset_cls=FakeSpectroDataset)
    assert isinstance(ds, FakeSpectroDataset)
    assert ds.samples[0][0] == "train"
    assert ds.samples[0][1].shape == (5, 3)
    assert len(ds.targets) == 1


# --------------------------------------------------------------------------- #
# Combined file + column roles + percentage split                             #
# --------------------------------------------------------------------------- #
def test_load_combined_file_with_percentage_split(tmp_path):
    df = pd.DataFrame(np.random.rand(10, 3), columns=["400", "401", "402"])
    df["y"] = np.arange(10.0)
    _csv(tmp_path / "all.csv", df)
    spec = {
        "sources": [{"id": "data", "role": "mixed", "input": "all.csv", "columns": {"features": "0:3", "targets": ["y"]}}],
        "partitions": {"by": "percentage", "train": "70%", "shuffle": True, "random_state": 0},
    }
    asm = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert asm.blocks["train"].X[0].shape == (7, 3)
    assert asm.blocks["test"].X[0].shape == (3, 3)


# --------------------------------------------------------------------------- #
# Flagship L.8: concat_samples + mixed columns + m:1 lookup broadcast          #
# --------------------------------------------------------------------------- #
def test_load_flagship_merged_with_lookup(tmp_path):
    for batch, sites in [("a", ["S1", "S2"]), ("b", ["S1", "S2"]), ("c", ["S2", "S2"])]:
        df = pd.DataFrame({"400": [0.1, 0.2], "401": [0.3, 0.4], "protein": [1.0, 2.0], "site_code": sites})
        _csv(tmp_path / f"batch_{batch}.csv", df)
    _csv(tmp_path / "sites.csv", pd.DataFrame({"site_code": ["S1", "S2"], "region": ["north", "south"]}))
    spec = {
        "name": "mango",
        "sample_index": {"by": "row"},
        "sources": [
            {
                "id": "measurements",
                "role": "mixed",
                "input": ["batch_a.csv", "batch_b.csv", "batch_c.csv"],
                "merge": "concat_samples",
                "columns": [
                    {"role": "features", "select": {"regex": r"^\d+$"}},
                    {"role": "targets", "select": ["protein"]},
                    {"role": "metadata", "select": ["site_code"]},
                ],
            },
            {
                "id": "sites",
                "kind": "lookup",
                "input": "sites.csv",
                "columns": [{"role": "metadata", "select": ["region"]}],
                "join": {"to": "measurements", "on": "site_code", "how": "m:1", "coverage": "complete"},
            },
        ],
    }
    asm = nio.load(spec, base_dir=tmp_path, target="assembled")
    block = asm.blocks["train"]
    assert block.X[0].shape == (6, 2)  # 3 batches x 2 rows, 2 wavelength cols
    assert block.y.shape == (6, 1)
    # the lookup's 'region' was broadcast (m:1) into metadata
    assert "region" in block.metadata.columns
    assert set(block.metadata["region"]) == {"north", "south"}


# --------------------------------------------------------------------------- #
# Multi-source features joined by key                                          #
# --------------------------------------------------------------------------- #
def test_load_multisource_by_key(tmp_path):
    _csv(tmp_path / "nir.csv", pd.DataFrame({"id": [1, 2, 3], "n1": [0.1, 0.2, 0.3], "n2": [0.4, 0.5, 0.6]}))
    _csv(tmp_path / "markers.csv", pd.DataFrame({"id": [1, 2, 3], "m1": [9.0, 8.0, 7.0]}))
    _csv(tmp_path / "y.csv", pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 2.0, 3.0]}))
    spec = {
        "sample_index": {"by": "id", "key": "id"},
        "sources": [
            {"id": "nir", "role": "features", "modality": "spectroscopy", "input": "nir.csv", "key": "id"},
            {"id": "markers", "role": "features", "modality": "markers", "input": "markers.csv", "key": "id", "join": {"to": "nir", "on": "id", "how": "1:1"}},
            {"id": "y", "role": "targets", "input": "y.csv", "key": "id", "join": {"to": "nir", "on": "id", "how": "1:1", "coverage": "complete"}},
        ],
    }
    asm = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert asm.n_sources == 2
    assert len(asm.blocks["train"].X) == 2  # multi-source: nir + markers
    assert asm.blocks["train"].y.shape == (3, 1)


# --------------------------------------------------------------------------- #
# In-memory                                                                    #
# --------------------------------------------------------------------------- #
def test_load_in_memory_xy():
    X = np.random.rand(8, 5).astype("float32")
    y = np.arange(8.0)
    asm = nio.load((X, y), target="assembled")
    assert asm.blocks["train"].X[0].shape == (8, 5)
    assert asm.blocks["train"].y.shape == (8, 1)


def test_load_in_memory_x_only_is_predict():
    asm = nio.load(np.random.rand(4, 3), target="assembled")
    assert "predict" in asm.blocks
