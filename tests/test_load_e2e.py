# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""End-to-end load() tests: directory conventions, combined files, flagship lookup,
multi-source, in-memory, and the lazy SpectroDataset adapter (via a recording double)."""

from __future__ import annotations

import json as _path_json
import os
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio


# --------------------------------------------------------------------------- #
# Recording SpectroDataset double (lets us test the adapter without nirs4all)  #
# --------------------------------------------------------------------------- #
class FakeSpectroDataset:
    def __init__(self, name="ds"):
        self.name = name
        self.samples = []  # (partition, X, headers, header_unit)
        self.targets = []
        self.metadata = []
        self.signal_types = []
        self.task_type = None
        self.folds = None
        self.repetition = None
        self.features_calls = []  # (array, processings, source)

    def add_samples(self, data, indexes, headers=None, header_unit=None):
        self.samples.append((indexes.get("partition"), data, headers, header_unit))

    def add_features(self, features, processings, source=-1):
        # adapter passes features wrapped in a one-element list (real nirs4all
        # expects InputFeatures = list[ndarray] | ndarray); unwrap for assertions
        arr = features[0] if isinstance(features, list) and len(features) == 1 else features
        self.features_calls.append((np.asarray(arr), list(processings), source))

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
# Combined file + column roles + explicit row-index split                     #
# --------------------------------------------------------------------------- #
def test_load_combined_file_with_index_split(tmp_path):
    df = pd.DataFrame(np.random.rand(10, 3), columns=["400", "401", "402"])
    df["y"] = np.arange(10.0)
    _csv(tmp_path / "all.csv", df)
    spec = {
        "sources": [{"id": "data", "role": "mixed", "input": "all.csv", "columns": {"features": "0:3", "targets": ["y"]}}],
        "partitions": {"by": "index", "train": [0, 1, 2, 3, 4, 5, 6], "test": [7, 8, 9]},
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


def test_load_dag_ml_data_target_points_to_rust_bridge():
    with pytest.raises(NotImplementedError, match="nirs4all-io-dagml"):
        nio.load(np.random.rand(4, 3), target="dag-ml-data")


# --------------------------------------------------------------------------- #
# Variations -> SpectroDataset processings (via add_features)                  #
# --------------------------------------------------------------------------- #
def test_load_variations_become_processings(tmp_path):
    wl = ["400", "401", "402"]
    raw = np.random.rand(5, 3).astype("float32")
    snv = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    _csv(tmp_path / "X.csv", pd.DataFrame(raw, columns=wl))
    _csv(tmp_path / "X_snv.csv", pd.DataFrame(snv, columns=wl))
    _csv(tmp_path / "Y.csv", pd.DataFrame({"y": np.arange(5.0)}))
    spec = {
        "sources": [
            {"id": "x", "role": "features", "input": "X.csv", "variations": [{"name": "snv", "input": "X_snv.csv"}]},
            {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}},
        ],
    }
    ds = nio.load(spec, base_dir=tmp_path, target="spectrodataset", spectro_dataset_cls=FakeSpectroDataset)
    # exactly one add_features call: one source × one variation
    assert len(ds.features_calls) == 1
    arr, names, src = ds.features_calls[0]
    assert names == ["snv"]
    assert src == 0
    assert arr.shape == (5, 3)
    np.testing.assert_allclose(arr, snv, rtol=1e-4)


def test_load_variations_survive_partition_split(tmp_path):
    """Variations must follow the parent's row reordering through an explicit index split."""
    wl = ["400", "401", "402"]
    raw = np.arange(30, dtype="float32").reshape(10, 3)
    snv = raw + 100.0  # easy-to-detect distinct values
    df = pd.DataFrame(raw, columns=wl)
    df["y"] = np.arange(10.0)
    _csv(tmp_path / "all.csv", df)
    _csv(tmp_path / "snv.csv", pd.DataFrame(snv, columns=wl))
    spec = {
        "sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": "0:3"}, {"role": "targets", "select": ["y"]}], "variations": [{"name": "snv", "input": "snv.csv"}]}],
        "partitions": {"by": "index", "train": [0, 1, 2, 3, 4, 5, 6], "test": [7, 8, 9]},
    }
    ds = nio.load(spec, base_dir=tmp_path, target="spectrodataset", spectro_dataset_cls=FakeSpectroDataset)
    # the variation array is added once with the concatenated train+test rows in their final order
    assert len(ds.features_calls) == 1
    arr, _names, _src = ds.features_calls[0]
    # train rows 0..6 come first, then test rows 7..9; snv[0] follows row 0
    np.testing.assert_allclose(arr[0], snv[0], rtol=1e-5)
    assert arr.shape == (10, 3)


# --------------------------------------------------------------------------- #
# Weights column -> SpectroDataset metadata `__sample_weight__`                #
# --------------------------------------------------------------------------- #
def test_load_weights_become_metadata_column(tmp_path):
    df = pd.DataFrame(np.random.rand(5, 3), columns=["400", "401", "402"])
    df["y"] = np.arange(5.0)
    df["w"] = [1.0, 1.0, 2.0, 0.5, 1.5]
    _csv(tmp_path / "data.csv", df)
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "data.csv", "columns": [{"role": "features", "select": "0:3"}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}]}]}
    ds = nio.load(spec, base_dir=tmp_path, target="spectrodataset", spectro_dataset_cls=FakeSpectroDataset)
    # exactly one metadata frame, with the `__sample_weight__` column
    assert len(ds.metadata) == 1
    meta = ds.metadata[0]
    assert "__sample_weight__" in meta.columns
    np.testing.assert_allclose(meta["__sample_weight__"].to_numpy(), [1.0, 1.0, 2.0, 0.5, 1.5], rtol=1e-5)


def _l1_four_partition_spec(tmp_path):
    sources = []
    for index, part in enumerate(["train", "val", "test", "predict"]):
        x = index * 10 + 1.25
        target = "" if part == "predict" else str(index * 10 + 4.5)
        (tmp_path / f"{part}.csv").write_text(f"id,x,z,y,w\n{part}-0,{x},{x + 1},{index * 10 + 3.5},0.25\n{part}-1,{x + 0.25},,{target},0.5\n")
        (tmp_path / f"{part}-v.csv").write_text(f"x,z\n{x + 100},{x + 101}\n{x + 100.25},{x + 101.25}\n")
        sources.append({"id": "s_" + part, "partition": part, "input": part + ".csv", "columns": [{"role": "features", "select": ["x", "z"]}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}, {"role": "metadata", "select": ["id"]}], "variations": [{"name": "v", "input": part + "-v.csv"}]})
    spec = {"name": "four-declared-partitions", "task_type": "regression", "sample_index": {"by": "id", "key": "id"}, "sources": sources, "folds": {"inline": [{"train": [0], "val": [1]}]}, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    return spec


def test_l1_declared_validation_partition_recording_constructor(tmp_path):
    spec = _l1_four_partition_spec(tmp_path)
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    ds = nio.load(spec, target="spectrodataset", spectro_dataset_cls=FakeSpectroDataset, base_dir=tmp_path)
    assert [part for part, *_ in ds.samples] == ["train", "test", "val", "predict"]
    assert all(np.asarray(x).shape == (2, 2) for _, x, *_ in ds.samples)
    assert ds.folds == [(np.array([0]), np.array([1]))]
    assert len(ds.features_calls) == 1 and ds.features_calls[0][1] == ["v"]
    for position, (part, data, *_rest) in enumerate(ds.samples):
        np.testing.assert_array_equal(np.isnan(data), np.isnan(assembled.blocks[part].X[0]))
        assert ds.metadata[0].iloc[position * 2:(position + 1) * 2]["id"].tolist() == [part + "-0", part + "-1"]
        assert ds.metadata[0].iloc[position * 2:(position + 1) * 2]["__sample_weight__"].tolist() == [0.25, 0.5]


def test_l1_four_declared_partitions_real_sdk(tmp_path):
    import json

    pytest.importorskip("polars", reason="Real SDK constructor unavailable without polars")
    from nirs4all.data import SpectroDataset

    spec = _l1_four_partition_spec(tmp_path)
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    projected = nio.load(spec, target="spectrodataset", base_dir=tmp_path)
    direct = SpectroDataset(name=assembled.name)
    order = ["train", "test", "val", "predict"]
    for part in order:
        block = assembled.blocks[part]
        direct.add_samples(block.X[0].astype(np.float32), {"partition": part}, headers=block.feature_headers[0], header_unit=block.header_units[0])
        direct.add_targets(block.y)
        metadata = block.metadata.copy()
        metadata["__sample_weight__"] = block.weights
        direct.add_metadata(metadata)
    direct.add_features([np.concatenate([assembled.blocks[p].processings[0][0][1] for p in order]).astype(np.float32)], ["v"], source=0)
    direct.set_folds(assembled.folds)
    assert projected.num_samples == direct.num_samples == 8
    facts = {}
    for part in ["train", "val", "test", "predict"]:
        selector = {"partition": part}
        left = np.asarray(projected.x(selector, layout="3d"))
        right = np.asarray(direct.x(selector, layout="3d"))
        assert left.shape == right.shape == (2, 2, 2)
        assert left.dtype == right.dtype == np.dtype("float32")
        assert left.tobytes() == right.tobytes()
        assert np.isnan(left).tolist() == np.isnan(right).tolist()
        yleft, yright = np.asarray(projected.y(selector)), np.asarray(direct.y(selector))
        assert yleft.dtype == yright.dtype and yleft.tobytes() == yright.tobytes()
        ids = projected.metadata_column("id", selector).tolist()
        assert ids == direct.metadata_column("id", selector).tolist() == [part + "-0", part + "-1"]
        assert projected.metadata_column("__sample_weight__", selector).tolist() == [0.25, 0.5]
        facts[part] = {"count": left.shape[0], "x_dtype": str(left.dtype), "x_shape": list(left.shape), "x_bytes": left.tobytes().hex(), "x_validity": (~np.isnan(left)).tolist(), "y_dtype": str(yleft.dtype), "y_bytes": yleft.tobytes().hex(), "y_validity": (~np.isnan(yleft)).tolist(), "ids": ids, "weights": [0.25, 0.5]}
    assert [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in projected.folds] == [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in direct.folds] == [([0], [1])]
    assert projected.features_processings(0) == direct.features_processings(0)
    expected_ids = ["train-0", "train-1", "test-0", "test-1", "val-0", "val-1", "predict-0", "predict-1"]
    with pytest.raises(ValueError, match="Selected feature source missing from partition"):
        package.executed_projection(["s_train"])
    print(json.dumps({"real_sdk_constructor_only": True, "partitions": facts, "folds": [([0], [1])], "processings": projected.features_processings(0), "stable_ids": expected_ids, "stable_projection_executed": False}))


@pytest.fixture(params=["recording", "sdk"])
def correction_constructor(request):
    if request.param == "recording":
        return FakeSpectroDataset
    pytest.importorskip("polars", reason="Actual SDK construction requires Polars")
    from nirs4all.data import SpectroDataset
    return SpectroDataset


def _correction_spec(tmp_path, train, test, *, targets=("t1", "t2"), features=("x",), test_features=None, val=None, variations=False):
    sources = []
    for part, text in [("train", train), ("test", test), ("val", val)]:
        if text is None:
            continue
        (tmp_path / (part + ".csv")).write_text(text)
        headers = text.split("\n", 1)[0].split(",")
        fs = list(test_features if part == "test" and test_features is not None else features)
        columns = [{"role": "features", "select": fs}, {"role": "metadata", "select": ["id"]}]
        present = [name for name in targets if name in headers]
        if present:
            columns.append({"role": "targets", "select": present})
        source = {"id": part, "partition": part, "input": part + ".csv", "columns": columns}
        if variations:
            values = [[101.5, 102.5], [103.5, 104.5]] if part == "train" else [[105.5, 106.5], [107.5, 108.5]]
            (tmp_path / (part + "-v.csv")).write_text(",".join(fs) + "\n" + "\n".join(",".join(str(v) for v in row) for row in values) + "\n")
            source["variations"] = [{"name": "v", "input": part + "-v.csv"}]
        sources.append(source)
    return {"sources": sources, "sample_index": {"by": "id", "key": "id"}, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}


def _correction_raw_targets(ds, part):
    if isinstance(ds, FakeSpectroDataset):
        table = np.concatenate(ds.targets) if ds.targets else np.empty((0, 0))
        offset = 0
        for partition, data, *_ in ds.samples:
            rows = len(data[0] if isinstance(data, list) else data)
            if partition == part:
                return table[offset:offset + rows]
            offset += rows
        return np.empty((0, table.shape[1]))
    return np.asarray(ds.y({"partition": part, "y": "raw"}))


@pytest.mark.parametrize("categorical,reverse_authority", [(False, False), (True, False), (False, True)])
def test_correction_target_permutation_keeps_named_meaning(tmp_path, correction_constructor, categorical, reverse_authority):
    train = "id,x,t1,t2\nA,1.5,10.25,20.75\nB,2.5,11.25,21.75\n"
    test = "id,x,t2,t1\nC,3.5,22.75,12.25\nD,4.5,23.75,13.25\n"
    expected = [[12.25, 22.75], [13.25, 23.75]]
    if categorical:
        train = "id,x,t1,t2\nA,1.5,red,20.5\nB,2.5,blue,21.5\n"
        test = "id,x,t2,t1\nC,3.5,22.5,red\nD,4.5,23.5,blue\n"
        expected = [["red", 22.5], ["blue", 23.5]]
    if reverse_authority:
        train, test = test.replace("C,", "A,").replace("D,", "B,"), train.replace("A,", "C,").replace("B,", "D,")
        expected = [[20.75, 10.25], [21.75, 11.25]]
    spec = _correction_spec(tmp_path, train, test)
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    assert assembled.blocks["train"].y_headers != assembled.blocks["test"].y_headers
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    assert _correction_raw_targets(ds, "test").tolist() == expected


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
@pytest.mark.parametrize("gap", ["leading", "interior", "trailing"])
def test_correction_missing_target_rows_keep_sample_origins(tmp_path, correction_constructor, kind, gap):
    header, a, b = ("t1", "10.5", "12.5") if kind == "numeric" else (("t1", "red", "blue") if kind == "categorical" else ("t1,t2", "red,10.5", "blue,12.5"))
    def labelled(ids, values):
        return "id,x," + header + "\n" + "\n".join(f"{i},{j + 1.5},{values}" for j, i in enumerate(ids)) + "\n"
    unlabelled = "id,x\nE,5.5\nF,6.5\n"
    tr, te, va = labelled("AB", a), labelled("CD", b), None
    if gap == "leading":
        tr, te = unlabelled, labelled("CD", b)
    elif gap == "interior":
        te, va = unlabelled, labelled("CD", b)
    else:
        te = unlabelled
    spec = _correction_spec(tmp_path, tr, te, val=va)
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    missing_part = "train" if gap == "leading" else "test"
    missing = _correction_raw_targets(ds, missing_part)
    if gap == "trailing":
        assert missing.shape[0] == 0
    else:
        assert missing.shape == (2, len(header.split(","))) and pd.isna(missing).all()
        part = "val" if gap == "interior" else "test"
        expected = [[float(b)]] * 2 if kind == "numeric" else ([[b]] * 2 if kind == "categorical" else [["blue", 12.5]] * 2)
        assert _correction_raw_targets(ds, part).tolist() == expected


def test_correction_allmissing_kind_is_neutral(tmp_path, correction_constructor):
    spec = _correction_spec(tmp_path, "id,x,t1\nA,1.5,red\nB,2.5,blue\n", "id,x,t1\nC,3.5,\nD,4.5,\n")
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    assert pd.isna(_correction_raw_targets(ds, "test")).all()


def test_correction_observed_kind_conflict_refuses_before_constructor(tmp_path):
    spec = _correction_spec(tmp_path, "id,x,t1\nA,1.5,red\nB,2.5,1\n", "id,x,t1\nC,3.5,1\nD,4.5,2\n")
    class NeverConstruct:
        def __init__(self, **kwargs):
            pytest.fail("Layout refusal must precede SDK construction")
    with pytest.raises(ValueError, match="target kind"):
        nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=NeverConstruct)


def test_correction_all_nan_preserves_sdk_refusal(tmp_path):
    pytest.importorskip("polars")
    spec = _correction_spec(tmp_path, "id,x,t1\nA,1.5,\nB,2.5,\n", "id,x,t1\nC,3.5,\nD,4.5,\n")
    with pytest.raises(ValueError, match="only NaN"):
        nio.load(spec, target="spectrodataset", base_dir=tmp_path)


@pytest.mark.parametrize("numeric_axis", [False, True])
def test_correction_feature_permutation_aligns_processing_and_headers(tmp_path, correction_constructor, numeric_axis):
    a, b, aa, bb = ("1000", "1100", "1000.0", "1100.0") if numeric_axis else ("f1", "f2", "f1", "f2")
    train = f"id,{a},{b},t1\nA,1.5,2.5,10.5\nB,3.5,4.5,11.5\n"
    test = f"id,{bb},{aa},t1\nC,6.5,5.5,12.5\nD,8.5,7.5,13.5\n"
    spec = _correction_spec(tmp_path, train, test, features=(a, b), test_features=(bb, aa), variations=True)
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    if isinstance(ds, FakeSpectroDataset):
        assert ds.samples[1][2] == [a, b]
        np.testing.assert_array_equal(ds.samples[1][1], [[5.5, 6.5], [7.5, 8.5]])
        np.testing.assert_array_equal(ds.features_calls[0][0][2:], [[106.5, 105.5], [108.5, 107.5]])
    else:
        assert ds.headers(0) == [a, b]
        values = np.asarray(ds.x({"partition": "test"}, layout="3d"))
        np.testing.assert_array_equal(values[:, 0], [[5.5, 6.5], [7.5, 8.5]])
        np.testing.assert_array_equal(values[:, 1], [[106.5, 105.5], [108.5, 107.5]])


@pytest.mark.parametrize("swapped", [False, True])
def test_correction_disjoint_source_ids_keep_declared_slots(tmp_path, correction_constructor, swapped):
    sources = []
    for part in ["train", "test"]:
        for kind in (["m", "n"] if swapped and part == "test" else ["n", "m"]):
            header = "1000,1100" if kind == "n" else "2000,2100"
            value = "0.1,0.2\n0.3,0.4\n" if kind == "n" else "0.5,0.6\n0.7,0.8\n"
            name = part + kind
            (tmp_path / (name + ".csv")).write_text(header + "\n" + value)
            sources.append({"id": name, "partition": part, "role": "features", "input": name + ".csv"})
    spec = {"sources": sources, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    if swapped:
        with pytest.raises(ValueError, match="feature|layout|axes"):
            nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    else:
        from nirs4all_io.materialize.spectrodataset import project_spectrodataset
        assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
        ds, receipt = project_spectrodataset(assembled, spectro_dataset_cls=correction_constructor)
        assert [(entry["partition"], entry["source_id"], entry["slot"]) for entry in receipt["features"]] == [("train", "trainn", 0), ("train", "trainm", 1), ("test", "testn", 0), ("test", "testm", 1)]
        if isinstance(ds, FakeSpectroDataset):
            np.testing.assert_allclose(ds.samples[1][1][0], [[.1, .2], [.3, .4]])
        else:
            assert len(ds.x({"partition": "test"}, layout="2d", concat_source=False)) == 2


def test_correction_legacy_train_x_list_remains_supported(tmp_path, correction_constructor):
    files = {"n1.csv": "1000,1100\n0.1,0.2\n0.3,0.4\n", "m1.csv": "2000,2100\n0.5,0.6\n0.7,0.8\n", "n2.csv": "1000,1100\n1.1,1.2\n1.3,1.4\n", "m2.csv": "2000,2100\n1.5,1.6\n1.7,1.8\n", "y1.csv": "y\n1.5\n2.5\n", "y2.csv": "y\n3.5\n4.5\n"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    config = {"train_x": [str(tmp_path / "n1.csv"), str(tmp_path / "m1.csv")], "test_x": [str(tmp_path / "n2.csv"), str(tmp_path / "m2.csv")], "train_y": str(tmp_path / "y1.csv"), "test_y": str(tmp_path / "y2.csv"), "global_params": {"delimiter": ",", "has_header": True}}
    ds = nio.load(config, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    matrices = ds.samples[1][1] if isinstance(ds, FakeSpectroDataset) else ds.x({"partition": "test"}, layout="2d", concat_source=False)
    np.testing.assert_allclose(matrices[0], [[1.1, 1.2], [1.3, 1.4]])
    np.testing.assert_allclose(matrices[1], [[1.5, 1.6], [1.7, 1.8]])


def _correction_ir():
    from nirs4all_io.materialize.assemble import AssembledDataset, PartitionBlock
    def block(ids, offset):
        arrays = [np.array([[offset + 1.5], [offset + 2.5]]), np.array([[offset + 3.5], [offset + 4.5]])]
        return PartitionBlock(n_samples=2, source_ids=ids, X=arrays, feature_headers=[["n"], ["m"]], header_units=["text", "text"], signal_types=[None, None], processings=[[("v", arrays[0] + 10)], [("v", arrays[1] + 20)]])
    return AssembledDataset("d", "auto", "auto", 2, blocks={"train": block(["n", "m"], 0), "test": block(["n", "m"], 100)})


def test_correction_common_source_identity_moves_processing_together():
    from nirs4all_io.materialize.spectrodataset import project_spectrodataset
    assembled = _correction_ir()
    test = assembled.blocks["test"]
    for field in ["source_ids", "X", "feature_headers", "header_units", "signal_types", "processings"]:
        setattr(test, field, list(reversed(getattr(test, field))))
    ds, receipt = project_spectrodataset(assembled, spectro_dataset_cls=FakeSpectroDataset)
    np.testing.assert_array_equal(ds.samples[1][1][0], [[101.5], [102.5]])
    np.testing.assert_array_equal(ds.features_calls[0][0][2:], [[111.5], [112.5]])
    assert [(r["source_id"], r["slot"]) for r in receipt["features"][-2:]] == [("n", 0), ("m", 1)]


@pytest.mark.parametrize("defect", ["overlap", "duplicate", "unit", "axis", "processing", "different-target", "duplicate-target"])
def test_correction_ambiguous_layouts_refuse_before_sdk_mutation(defect):
    from nirs4all_io.materialize.spectrodataset import project_spectrodataset
    assembled = _correction_ir()
    train, test = assembled.blocks["train"], assembled.blocks["test"]
    if defect == "overlap":
        test.source_ids = ["n", "other"]
    elif defect == "duplicate":
        test.source_ids = ["n", "n"]
    elif defect == "unit":
        test.header_units[0] = "nm"
    elif defect == "axis":
        train.feature_axes = [[{"name": "a", "unit": "nm", "coordinates": [1.0]}], []]
        test.feature_axes = [[{"name": "a", "unit": "nm", "coordinates": [2.0]}], []]
    elif defect == "processing":
        test.processings[0] = [("other", test.X[0])]
    else:
        train.y_headers = ["a", "a", "b"] if defect == "duplicate-target" else ["a"]
        test.y_headers = ["a", "b", "a"] if defect == "duplicate-target" else ["b"]
        train.y = np.ones((2, len(train.y_headers)))
        test.y = np.ones((2, len(test.y_headers)))
    class NeverConstruct:
        def __init__(self, **kwargs):
            pytest.fail("preflight must precede SDK construction")
    with pytest.raises(ValueError, match="SpectroDataset"):
        project_spectrodataset(assembled, spectro_dataset_cls=NeverConstruct)


def test_correction_identical_ordered_duplicate_targets_keep_position():
    from nirs4all_io.materialize.spectrodataset import project_spectrodataset
    assembled = _correction_ir()
    for block in assembled.blocks.values():
        block.y_headers = ["a", "a"]
        block.y = np.array([[1.5, 2.5], [3.5, 4.5]])
    ds, _ = project_spectrodataset(assembled, spectro_dataset_cls=FakeSpectroDataset)
    np.testing.assert_array_equal(ds.targets[0], [[1.5, 2.5], [3.5, 4.5]] * 2)


def test_label_content_sdk_constructor_preserves_exact_categories(tmp_path, correction_constructor):
    import csv
    labels = [" ", "\t", "\u00a0", "  ", "é", "e\u0301"]
    with (tmp_path / "label.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "label"])
        writer.writerows([["S" + str(i), i + .5, label] for i, label in enumerate(labels)])
    spec = {"sources": [{"id": "s", "input": "label.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["label"]}, {"role": "metadata", "select": ["id"]}]}], "sample_index": {"by": "id", "key": "id"}, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    # Exercise the package contract as well as the SDK's own label encoder.
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert package.executed_projection(["s"]).identity.sample_ids == ["S" + str(i) for i in range(len(labels))]
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    assert _correction_raw_targets(ds, "train")[:, 0].tolist() == labels
    if not isinstance(ds, FakeSpectroDataset):
        numeric = np.asarray(ds.y({"partition": "train"}))
        assert np.isfinite(numeric).all() and len(np.unique(numeric)) == len(labels)
        assert ds.metadata_column("id", {"partition": "train"}).tolist() == ["S" + str(i) for i in range(len(labels))]


def _r12_metadata_values(ds, name, part):
    if not isinstance(ds, FakeSpectroDataset):
        return ds.metadata_column(name, {"partition": part}).tolist()
    table = pd.concat(ds.metadata, ignore_index=True)
    offset = 0
    for partition, data, *_ in ds.samples:
        rows = len(data[0] if isinstance(data, list) else data)
        if partition == part:
            return table.reindex(range(offset, offset + rows))[name].tolist()
        offset += rows
    raise AssertionError(part)


@pytest.mark.parametrize("kind", ["metadata", "weights", "mixed"])
@pytest.mark.parametrize("gap", ["leading", "interior", "trailing"])
@pytest.mark.parametrize("reverse_declaration", [False, True])
def test_r12_partition_metadata_and_weights_keep_sample_rows(tmp_path, correction_constructor, kind, gap, reverse_declaration):
    parts = ["train", "test", "val"]
    missing = {"leading": "train", "interior": "test", "trailing": "val"}[gap]
    sources = []
    for index, part in enumerate(parts):
        headers = ["x", "y"]
        values = [[index * 10 + 1.5, index * 10 + 2.5], [index * 10 + 3.5, index * 10 + 4.5]]
        columns = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]
        if part != missing:
            if kind != "weights":
                # Distinct headers and order also require padding absent columns.
                meta = ["m", "extra"] if index % 2 == 0 else ["extra", "m"]
                headers += meta
                for row, suffix in zip(values, ["a", "b"], strict=True):
                    row += [part + "-" + suffix if name == "m" else part + "-extra" for name in meta]
                columns.append({"role": "metadata", "select": meta})
            if kind != "metadata":
                headers.append("w")
                for row, weight in zip(values, [index + .25, index + .75], strict=True):
                    row.append(weight)
                columns.append({"role": "weights", "select": ["w"]})
        text = ",".join(headers) + "\n" + "\n".join(",".join(map(str, row)) for row in values) + "\n"
        (tmp_path / (part + ".csv")).write_text(text)
        sources.append({"id": part, "partition": part, "input": part + ".csv", "columns": columns})
    spec = {"sources": list(reversed(sources)) if reverse_declaration else sources, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    for index, part in enumerate(parts):
        # Payload checks ensure metadata is attached to its original sample block.
        raw_y = _correction_raw_targets(ds, part)
        np.testing.assert_array_equal(raw_y, assembled.blocks[part].y)
        for column in (["m", "extra"] if kind == "metadata" else (["__sample_weight__"] if kind == "weights" else ["m", "extra", "__sample_weight__"])):
            actual = _r12_metadata_values(ds, column, part)
            if part == missing:
                assert len(actual) == 2 and pd.isna(actual).all()
            else:
                expected = [index + .25, index + .75] if column == "__sample_weight__" else ([part + "-a", part + "-b"] if column == "m" else [part + "-extra"] * 2)
                assert actual == expected


def test_r12_public_test_group_stays_in_test(tmp_path, correction_constructor):
    files = {"x1.csv": "x\n1.5\n2.5\n", "x2.csv": "x\n3.5\n4.5\n", "y1.csv": "y\n10.5\n11.5\n", "y2.csv": "y\n12.5\n13.5\n", "g.csv": "site\nS3\nS4\n"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    spec = {"train_x": "x1.csv", "test_x": "x2.csv", "train_y": "y1.csv", "test_y": "y2.csv", "test_group": "g.csv", "global_params": {"delimiter": ",", "has_header": True}}
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    assert pd.isna(_r12_metadata_values(ds, "site", "train")).all()
    assert _r12_metadata_values(ds, "site", "test") == ["S3", "S4"]
    np.testing.assert_array_equal(_correction_raw_targets(ds, "train"), [[10.5], [11.5]])
    np.testing.assert_array_equal(_correction_raw_targets(ds, "test"), [[12.5], [13.5]])


def test_r12_processing_names_permute_with_each_source_payload(tmp_path, correction_constructor):
    sources = []
    for partition_index, part in enumerate(["train", "test"]):
        for source_index, headers in enumerate([("p", "q"), ("r", "s")]):
            ordered = list(reversed(headers)) if part == "test" else list(headers)
            base = 100 * source_index + 10 * partition_index
            raw = {headers[0]: [base + 1.5, base + 3.5], headers[1]: [base + 2.5, base + 4.5]}
            name = part + str(source_index)
            (tmp_path / (name + ".csv")).write_text(",".join(ordered) + "\n" + "\n".join(",".join(str(raw[h][row]) for h in ordered) for row in range(2)) + "\n")
            variations = []
            for processing, delta in ([("w", 200), ("v", 100)] if part == "test" else [("v", 100), ("w", 200)]):
                filename = name + processing + ".csv"
                (tmp_path / filename).write_text(",".join(ordered) + "\n" + "\n".join(",".join(str(raw[h][row] + delta) for h in ordered) for row in range(2)) + "\n")
                variations.append({"name": processing, "input": filename})
            sources.append({"id": name, "partition": part, "role": "features", "input": name + ".csv", "variations": variations})
    spec = {"sources": sources, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    for source in [0, 1]:
        for name, delta in [("v", 100), ("w", 200)]:
            expected = np.array([[source * 100 + 11.5 + delta, source * 100 + 12.5 + delta], [source * 100 + 13.5 + delta, source * 100 + 14.5 + delta]], dtype=np.float32)
            if isinstance(ds, FakeSpectroDataset):
                arrays = [array for array, names, slot in ds.features_calls if names == [name] and slot == source]
                assert len(arrays) == 1
                actual = arrays[0][2:]
            else:
                position = ds.features_processings(source).index(name)
                actual = ds.x({"partition": "test"}, layout="3d", concat_source=False)[source][:, position]
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["bool", "int", "bigint", "date", "string", "float", "mixed"])
@pytest.mark.parametrize("gap", ["leading", "interior", "trailing", "disjoint"])
def test_r3_metadata_full_columns_preserve_values_and_sdk_kinds(tmp_path, correction_constructor, kind, gap):
    import datetime

    import openpyxl
    values = {"bool": [True, False], "int": [7, 5], "bigint": [9007199254740993, 5], "date": [datetime.datetime(2026, 1, 2), datetime.datetime(2026, 1, 3)], "string": ["red", "blue"], "float": [3.5, 4.5], "mixed": [7, "abc"]}[kind]
    missing = {"leading": "train", "interior": "test", "trailing": "val", "disjoint": "test"}[gap]
    sources = []
    for index, part in enumerate(["train", "test", "val"]):
        present = part != missing
        headers = ["id", "x", "y"] + (["m"] if present else (["q"] if gap == "disjoint" else []))
        rows = [[part + "-" + str(row), index * 10 + row + .5, index * 10 + row + .25] + ([values[row]] if present else ([7] if gap == "disjoint" else [])) for row in range(2)]
        if kind in ("bool", "date", "mixed"):
            filename = part + ".xlsx"
            book = openpyxl.Workbook()
            sheet = book.active
            sheet.append(headers)
            for row in rows:
                sheet.append(row)
            book.save(tmp_path / filename)
            book.close()
        else:
            filename = part + ".csv"
            # CSV is essential for exact integer >2**53: openpyxl's writer rounds it.
            (tmp_path / filename).write_text(",".join(headers) + "\n" + "\n".join(",".join(map(str, row)) for row in rows) + "\n")
        cols = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "ignore", "select": ["id"]}]
        if present:
            cols.append({"role": "metadata", "select": ["m"]})
        elif gap == "disjoint":
            cols.append({"role": "metadata", "select": ["q"]})
        sources.append({"id": part, "partition": part, "input": filename, "columns": cols})
    spec = {"sources": list(reversed(sources)), "sample_index": {"by": "id", "key": "id"}, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    def metadata_values(name, part):
        return _r12_metadata_values(ds, name, part) if isinstance(ds, FakeSpectroDataset) else ds.metadata({"partition": part})[name].to_list()
    for part in ["train", "test", "val"]:
        actual = metadata_values("m", part)
        if part == missing:
            assert actual == [None, None]
        else:
            expected = ["7", "abc"] if kind == "mixed" and not isinstance(ds, FakeSpectroDataset) else values
            assert actual == expected
            for actual_value, expected_value in zip(actual, expected, strict=True):
                if kind == "date":
                    assert isinstance(actual_value, datetime.datetime)
                else:
                    assert type(actual_value) is type(expected_value)
        assert metadata_values("id", part) == [part + "-0", part + "-1"]
        np.testing.assert_array_equal(_correction_raw_targets(ds, part), assembled.blocks[part].y)
    if isinstance(ds, FakeSpectroDataset):
        assert len(ds.metadata) == 1
        assert all(dtype.kind == "O" for dtype in ds.metadata[0].dtypes)
    else:
        column = ds.metadata({"partition": "train"})["m"]
        expected_kind = {"bool": "Boolean", "int": "Int64", "bigint": "Int64", "date": "Datetime", "string": "String", "float": "Float64", "mixed": "String"}[kind]
        assert str(column.dtype).startswith(expected_kind)
        if gap == "disjoint":
            assert str(ds.metadata({"partition": "test"})["q"].dtype) == "Int64"
            assert metadata_values("q", "test") == [7, 7]


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("weight_storage", [None, "float32"])
def test_r3_c3_weight_storage_keeps_requested_numeric_kind(tmp_path, correction_constructor, gap, missing, weight_storage):
    sources = []
    for part in (["train", "test"] if gap else ["train"]):
        weighted = not gap or part == "test"
        text = "x,y" + (",w" if weighted else "") + "\n"
        text += "1.5,10.5" + (("," + ("" if missing else "1.5")) if weighted else "") + "\n2.5,11.5" + (("," + ("" if missing else "2.5")) if weighted else "") + "\n"
        (tmp_path / (part + ".csv")).write_text(text)
        columns = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]
        if weighted:
            columns.append({"role": "weights", "select": ["w"]})
        sources.append({"id": part, "partition": part, "input": part + ".csv", "columns": columns})
    spec = {"sources": sources, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor, weights=weight_storage)
    expected = ([None, None] if gap else []) + ([None, None] if missing else [1.5, 2.5])
    if isinstance(ds, FakeSpectroDataset):
        column = ds.metadata[0]["__sample_weight__"]
        assert column.dtype == np.dtype(weight_storage or "float64")
        actual = column.tolist()
    else:
        column = ds.metadata({})["__sample_weight__"]
        assert str(column.dtype) == ("Float32" if weight_storage == "float32" else "Float64")
        actual = column.to_list()
    assert len(actual) == len(expected)
    for value, wanted in zip(actual, expected, strict=True):
        assert pd.isna(value) if wanted is None else value == wanted
    for part in (["train", "test"] if gap else ["train"]):
        np.testing.assert_array_equal(_correction_raw_targets(ds, part), [[10.5], [11.5]])


@pytest.mark.parametrize("gap", [False, True])
def test_r3_all_null_numeric_metadata_retains_demonstrated_float_kind(tmp_path, correction_constructor, gap):
    sources = []
    for part in (["train", "test"] if gap else ["train"]):
        selected = not gap or part == "test"
        (tmp_path / (part + ".csv")).write_text("x" + (",m" if selected else "") + "\n1.5" + ("," if selected else "") + "\n2.5" + ("," if selected else "") + "\n")
        columns = [{"role": "features", "select": ["x"]}] + ([{"role": "metadata", "select": ["m"]}] if selected else [])
        sources.append({"id": part, "partition": part, "input": part + ".csv", "columns": columns})
    spec = {"sources": sources, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    column = ds.metadata[0]["m"] if isinstance(ds, FakeSpectroDataset) else ds.metadata({})["m"]
    assert str(column.dtype) == ("float64" if isinstance(ds, FakeSpectroDataset) else "Float64")
    assert len(column) == (4 if gap else 2)
    assert all(pd.isna(value) for value in (column.tolist() if isinstance(ds, FakeSpectroDataset) else column.to_list()))


@pytest.mark.parametrize("weight_storage", [None, "float32"])
@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("blank", [False, True])
def test_qw_real_temporal_weights_keep_c3_numeric_storage(tmp_path, correction_constructor, weight_storage, gap, blank):
    import datetime

    import openpyxl
    sources = []
    for part in (["train", "test"] if gap else ["train"]):
        weighted = not gap or part == "test"
        filename = part + ".xlsx"
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(["x", "y"] + (["w"] if weighted else []))
        for index in range(2):
            sheet.append([index + 1.5, index + 10.5] + ([None if blank and index == 1 else datetime.datetime(2026, 1, 2)] if weighted else []))
        book.save(tmp_path / filename)
        book.close()
        columns = [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}] + ([{"role": "weights", "select": ["w"]}] if weighted else [])
        sources.append({"id": part, "partition": part, "input": filename, "columns": columns})
    spec = {"sources": sources, "params": {"na": {"policy": "ignore"}}}
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor, weights=weight_storage)
    if isinstance(ds, FakeSpectroDataset):
        column = ds.metadata[0]["__sample_weight__"]
        assert column.dtype == np.dtype(weight_storage or "float64")
        assert column.isna().all()
    else:
        column = ds.metadata({})["__sample_weight__"]
        assert str(column.dtype) == ("Float32" if weight_storage == "float32" else "Float64")
        assert column.to_list() == [None] * (4 if gap else 2)
    for part in (["train", "test"] if gap else ["train"]):
        np.testing.assert_array_equal(_correction_raw_targets(ds, part), [[10.5], [11.5]])



def test_qw_parquet_default_sdk_raw_admission_is_labelled_qp(tmp_path, correction_constructor):
    import datetime

    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"x": [1.5, 2.5, 3.5], "m": pa.array([datetime.datetime(2026, 1, 2), datetime.datetime(2026, 1, 3), None], type=pa.timestamp("us"))}), tmp_path / "w.parquet")
    spec = {"sources": [{"id": "s", "input": "w.parquet", "columns": [{"role": "features", "select": ["x"]}, {"role": "weights", "select": ["m"]}]}], "params": {"na": {"policy": "ignore"}}}
    # Q-P: default raw SDK admits this input while Rust refuses its decoder.
    ds = nio.load(spec, target="spectrodataset", base_dir=tmp_path, spectro_dataset_cls=correction_constructor)
    column = ds.metadata[0]["__sample_weight__"] if isinstance(ds, FakeSpectroDataset) else ds.metadata({})["__sample_weight__"]
    assert str(column.dtype) == ("float64" if isinstance(ds, FakeSpectroDataset) else "Float64")
    assert len(column) == 3
    assert all(pd.isna(value) for value in (column.tolist() if isinstance(ds, FakeSpectroDataset) else column.to_list()))


_PATH_VECTORS = _path_json.loads((_Path(__file__).parent / "fixtures/path_reference_vectors.json").read_text())["rows"]


def _path_reference_case(tmp_path, case):
    if case.get("platform") == "unix" and os.name != "posix":
        pytest.skip("literal wildcard filenames require Unix")
    for name in case.get("directories", []):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    for name, text in case["files"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for name, target in case.get("links", {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(tmp_path / target, target_is_directory=(tmp_path / target).is_dir())
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
    value = case["input"]
    return [str(tmp_path / x) for x in value] if isinstance(value, list) else str(tmp_path / value)


@pytest.mark.parametrize("case", [x for x in _PATH_VECTORS if x["operation"] == "load"], ids=lambda x: x["id"])
def test_path_reference_materialization_vectors(tmp_path, monkeypatch, case):
    import nirs4all_io as nio
    inp = _path_reference_case(tmp_path, case)
    base = tmp_path / case.get("base", ".")
    cwd = tmp_path / case.get("cwd", "elsewhere")
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)
    source = {"id": "data", "role": "mixed", "input": inp if "base" not in case else case["input"], "merge": "concat_samples", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}
    if "variation" in case:
        source["variations"] = [{"name": "filtered", "input": str(tmp_path / case["variation"])}]
    spec = {"name": "path-case", "params": {"delimiter": ","}, "sources": [source]}
    assembled = nio.load(spec, base_dir=base, target="assembled")
    block = assembled.blocks["train"]
    expected = case["expected"]
    assert block.n_samples == expected["n_samples"]
    assert float(block.X[0][0, 0]) == expected["x00"]
    assert float(block.y[0, 0]) == expected["y0"]
    if "x_values" in expected:
        np.testing.assert_array_equal(block.X[0][:, 0], expected["x_values"])
        np.testing.assert_array_equal(block.y[:, 0], [1.5, 2.5] * (expected["n_samples"] // 2))
    assert block.source_ids == ["data"]
    assert block.feature_headers == [["x"]]
    if "processing_x00" in expected:
        assert block.processings[0][0][0] == "filtered"
        assert float(block.processings[0][0][1][0, 0]) == expected["processing_x00"]
    wire = assembled.to_full_dict()["blocks"]["train"]
    assert all(wire["x"][0]["validity"])
    assert all(wire["y"]["columns"][0]["block"]["validity"])


def test_path_reference_actual_sdk_constructor(tmp_path):
    pytest.importorskip("polars", reason="real SDK construction requires optional Polars")
    import numpy as np

    import nirs4all_io as nio
    case = next(x for x in _PATH_VECTORS if x["id"] == "alias-store")
    inp = _path_reference_case(tmp_path, case)
    ds = nio.load(nio.infer(inp).accept(), target="spectrodataset")
    values = ds.x({"partition": "train"})
    assert len(values) == 2
    assert np.asarray(values).dtype == np.dtype("float32")
    assert np.asarray(values).reshape(2, -1)[0, 0].view(np.uint32) == np.float32(0.1).view(np.uint32)
    np.testing.assert_array_equal(ds.y({"y": "raw"}).reshape(-1), [1.5, 2.5])


@pytest.mark.parametrize("cwd_decoy", [False, True], ids=["without-cwd-file", "with-cwd-csv"])
def test_path_reference_scoped_directory_preserves_vendor_refusal(tmp_path, monkeypatch, cwd_decoy):
    from nirs4all_io.materialize import loaders

    base = tmp_path / "base"
    directory = base / "dd"
    directory.mkdir(parents=True)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    if cwd_decoy:
        (cwd / "dd").write_text(",".join(str(1000 + 5 * i) for i in range(12)) + "\n" + ",".join(["9.9"] * 12) + "\n")
    monkeypatch.chdir(cwd)
    spec = {"params": {"delimiter": ","}, "sources": [{"id": "spectra", "role": "features", "input": "dd"}]}
    # The actual public route must refuse at the vendor reader, even with a valid CWD CSV.
    with pytest.raises(loaders.LoaderError, match="vendor|formats"):
        nio.load(spec, base_dir=base, target="assembled")
    calls = []

    def refuse_vendor(path, params):
        calls.append(path)
        raise loaders.LoaderError(f"vendor directory refused: {path}")

    # A controlled owner-boundary probe additionally pins the exact directory path.
    monkeypatch.setattr(loaders, "_read_vendor", refuse_vendor)
    with pytest.raises(loaders.LoaderError, match="vendor directory refused"):
        nio.load(spec, base_dir=base, target="assembled")
    assert calls == [directory]


@pytest.mark.parametrize("case", [x for x in _PATH_VECTORS if x["operation"] == "directory-refusal"], ids=lambda x: x["id"])
def test_path_reference_a1_literal_directory_preserves_vendor_path(tmp_path, monkeypatch, case):
    from nirs4all_io.materialize import loaders

    _path_reference_case(tmp_path, case)
    monkeypatch.chdir(tmp_path / case["cwd"])
    spec = {"params": {"delimiter": ","}, "sources": [{"id": "spectra", "role": "features", "input": case["input"]}]}
    with pytest.raises(loaders.LoaderError, match="vendor|formats"):
        nio.load(spec, base_dir=tmp_path / case["base"], target="assembled")
    calls = []

    def refuse_vendor(path, params):
        calls.append(path)
        raise loaders.LoaderError(f"vendor directory refused: {path}")

    monkeypatch.setattr(loaders, "_read_vendor", refuse_vendor)
    with pytest.raises(loaders.LoaderError, match="vendor directory refused"):
        nio.load(spec, base_dir=tmp_path / case["base"], target="assembled")
    assert calls == [tmp_path / case["expected"]["vendor_path"]]


_DELIMITER_VECTORS = _path_json.loads((_Path(__file__).parent / "fixtures/delimiter_resolution_vectors.json").read_text())["rows"]


def _delimiter_case(tmp_path, case):
    import gzip
    import zipfile

    for name, text in case["files"].items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    route = case["route"]
    spec = {"sources": [{"id": "train_x", "role": "features", "input": "Xcal.csv"}]}
    if case.get("params"):
        spec["params"] = case["params"]
    if route == "dir":
        return tmp_path
    if route == "list":
        return list(tmp_path.glob("*.csv"))
    if route == "file":
        return tmp_path / "data.csv"
    if route == "legacy":
        return {"train_x": "Xcal.csv"}
    if route == "spec":
        return nio.DatasetSpec.from_dict(spec)
    if route == "config":
        path = tmp_path / "config.json"
        path.write_text(_path_json.dumps(spec))
        return path
    if route == "variation":
        (tmp_path / "alternate.csv").write_text(case["files"]["Xcal.csv"])
        (tmp_path / "Xcal.csv").write_text(case["files"]["Xcal.csv"].replace(",", ";"))
        spec["sources"][0]["variations"] = [{"name": "alternate", "input": "alternate.csv"}]
    if route in {"gzip", "zip"}:
        name = "Xcal.csv" + (".gz" if route == "gzip" else ".zip")
        data = case["files"]["Xcal.csv"].encode()
        if route == "gzip":
            (tmp_path / name).write_bytes(gzip.compress(data, mtime=0))
        else:
            with zipfile.ZipFile(tmp_path / name, "w") as archive:
                archive.writestr("Xcal.csv", data)
        spec["sources"][0]["input"] = name
    return case.get("spec", spec)


@pytest.mark.parametrize("case", [row for row in _DELIMITER_VECTORS if row["operation"] == "load-assembled"], ids=lambda x: x["id"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_delimiter_resolution_public(tmp_path, case, target):
    """Pin values and masks through the public owner, including caller-owned refusal."""
    from nirs4all_io.spec.enums import SpecError

    inp = _delimiter_case(tmp_path, case)
    expected = case["expected"]
    if "refuse" in expected:
        # to_spec stays byte-free for caller-owned declarations.
        if case["route"] not in {"dir", "list", "file"}:
            nio.to_spec(inp, base_dir=tmp_path)
        with pytest.raises(SpecError, match=expected["refuse"]) as error:
            nio.load(inp, base_dir=tmp_path, target=target)
        assert expected["file"] in str(error.value)
        assert expected.get("source", "train_x") in str(error.value)
        if expected.get("delimiter"):
            assert repr(expected["delimiter"]) in str(error.value)
        return
    spec, base = nio.to_spec(inp, base_dir=tmp_path)
    if case["route"] in {"dir", "list", "file"}:
        assert spec.sources[0].params.delimiter == expected.get("delimiter")
        if len(spec.sources) > 1:
            assert spec.sources[1].params.delimiter is None
    result = nio.load(spec, base_dir=base, target=target)
    block = (result.to_assembled() if target == "dataset_package" else result).blocks["train"]
    values = np.array(expected["x"], dtype=float)
    if "shape" in expected:
        values = values.reshape(expected["shape"])
    np.testing.assert_array_equal(block.X[0], values)
    np.testing.assert_array_equal(np.isfinite(block.X[0]), np.isfinite(values))
    assert block.feature_headers[0] == expected["headers"]
    if expected.get("y") is not None:
        np.testing.assert_array_equal(block.y, expected["y"])
    if "y_headers" in expected:
        assert block.y_headers == expected["y_headers"]


def test_delimiter_resolution_raw_reader_and_infer_keep_contracts(tmp_path):
    """Raw summaries remain permissive; inspection may declare its own dialect."""
    from nirs4all_io.materialize.loaders import read_table
    from nirs4all_io.spec.dataset_spec import LoadingParams

    case = _DELIMITER_VECTORS[0]
    _delimiter_case(tmp_path, case)
    table = read_table(tmp_path / "Xcal.csv", LoadingParams())
    assert table.df.shape == (3, 1)
    assert table.headers == [",".join(case["expected"]["headers"])]
    block = nio.load(nio.infer(tmp_path).accept(), target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.X[0], case["expected"]["x"])


@pytest.mark.parametrize("encoding", ["utf-8-sig", "latin-1", "utf-16"])
def test_delimiter_resolution_encoding_and_quoted_records(tmp_path, encoding):
    """Use the existing reader's explicit encoding and multiline CSV quoting."""
    from nirs4all_io.spec.enums import SpecError

    path = tmp_path / "Xcal.csv"
    path.write_bytes('"café,\naxis";b\n1;2\n3;4\n'.encode(encoding))
    spec = {"params": {"encoding": encoding}, "sources": [{"id": "x", "input": path.name, "role": "features"}]}
    block = nio.load(spec, base_dir=tmp_path, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[1, 2], [3, 4]])
    assert block.feature_headers[0] == ["café,\naxis", "b"]
    path.write_bytes('"café,\naxis",b\n1,2\n3,4\n'.encode(encoding))
    with pytest.raises(SpecError, match="declare params.delimiter"):
        nio.load(spec, base_dir=tmp_path, target="assembled")
    spec["params"]["delimiter"] = ","
    np.testing.assert_array_equal(nio.load(spec, base_dir=tmp_path, target="assembled").blocks["train"].X[0], [[1, 2], [3, 4]])


def test_delimiter_resolution_bounded_prefix_preserves_later_reader_data(tmp_path):
    """The proposal is bounded; materialization still consumes every record."""
    from nirs4all_io.materialize.loaders import _DIALECT_SAMPLE_BYTES, _delimiter_sample
    from nirs4all_io.spec.dataset_spec import LoadingParams

    path = tmp_path / "data.csv"
    # Over 8 MiB of nonblank, quote-aware one-column text, with no alternate
    # candidate. A cap in the final quoted record must not create new evidence.
    path.write_text('"a,b"\n' + ('"Smith, J"\n' * 50) + ('"' + 'z' * _DIALECT_SAMPLE_BYTES + '"\n'))
    text, truncated = _delimiter_sample(path, LoadingParams())
    assert len(text.encode()) == _DIALECT_SAMPLE_BYTES
    assert truncated
    # A source with more than 50 valid numeric rows still retains the tail.
    path.write_text("x,y\n" + "1.0,2.0\n" * 51 + "3.0,4.0\n")
    block = nio.load(path, target="assembled").blocks["train"]
    assert block.X[0].shape == (52, 2)
    np.testing.assert_array_equal(block.X[0][-1], [3, 4])


def test_delimiter_resolution_large_quoted_fields_do_not_change_parser_limits(tmp_path):
    """The 8 MiB sampler must not inherit csv.reader's smaller field limit."""
    import csv

    field_limit = csv.field_size_limit()
    header = "axis," * 40000
    (tmp_path / "data.csv").write_text(f'"{header}",y\n1.0,2.0\n3.0,4.0\n')
    block = nio.load(tmp_path / "data.csv", target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[1, 2], [3, 4]])
    assert block.feature_headers[0] == [header, "y"]
    # The same quoted commas are one legitimate text column when no second
    # field exists, even beyond the standard csv module's field-size limit.
    (tmp_path / "data.csv").write_text(f'"{header}"\n1\n2\n')
    np.testing.assert_array_equal(nio.load(tmp_path / "data.csv", target="assembled").blocks["train"].X[0], [[1], [2]])
    assert csv.field_size_limit() == field_limit


def test_delimiter_resolution_cap_keeps_complete_evidence(tmp_path):
    """A partial whitespace tail must not erase a preceding complete record."""
    from nirs4all_io.materialize.loaders import _DIALECT_SAMPLE_BYTES, proposed_delimiter
    from nirs4all_io.spec.dataset_spec import LoadingParams

    path = tmp_path / "data.csv"
    path.write_text("x,y\n1.0,2.0\n" + " " * _DIALECT_SAMPLE_BYTES)
    assert proposed_delimiter(path, LoadingParams()) == ","


_DECIMAL_VECTORS = _path_json.loads((_Path(__file__).parent / "fixtures/decimal_inference_vectors.json").read_text())["rows"]
_DECIMAL_ROUTES = [(case, form, method, target) for case in _DECIMAL_VECTORS for form in case["forms"] for method in case["methods"] for target in ("assembled", "dataset_package")]


def _decimal_input(tmp_path, case, form):
    for name, text in case["files"].items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    if form == "dir":
        return tmp_path
    if form == "list":
        return [tmp_path / name for name in case["files"]]
    if form == "single":
        return tmp_path / next(iter(case["files"]))
    return case["spec"]


def _decimal_execute(tmp_path, case, form, method, target):
    inp = _decimal_input(tmp_path, case, form)
    plan = nio.infer(inp) if method == "infer" else None
    if plan is not None:
        spec, base = plan.accept(), tmp_path
    else:
        spec, base = nio.to_spec(inp, base_dir=tmp_path)
    for source in spec.sources:
        for key, value in case.get("declare_after_infer", {}).get(source.id, {}).items():
            setattr(source.params, key, value)
    if case.get("post_params"):
        from nirs4all_io.spec.dataset_spec import LoadingParams
        spec.params = LoadingParams.from_dict(case["post_params"])
    result = nio.load(spec, base_dir=base, target=target)
    assembled = result.to_assembled() if target == "dataset_package" else result
    block = assembled.blocks["train"]
    print(_path_json.dumps({"decimal_trace": case["id"], "method": method, "target": target, "shape": list(block.X[0].shape), "valid": int(np.isfinite(block.X[0]).sum()), "x_first": block.X[0][0].tolist(), "x_last": block.X[0][-1].tolist(), "headers": block.feature_headers, "y": None if block.y is None else block.y.tolist(), "y_headers": block.y_headers, "global_params": spec.params.to_dict(), "source_params": {source.id: source.params.to_dict() for source in spec.sources}}))
    return assembled, spec, plan


@pytest.mark.parametrize("case,form,method,target", _DECIMAL_ROUTES, ids=[f"{c['id']}-{f}-{m}-{t}" for c, f, m, t in _DECIMAL_ROUTES])
def test_decimal_inference_vectors(tmp_path, case, form, method, target):
    """Public values, masks and diagnostics from literal bytes, without fitting."""
    from nirs4all_io.materialize.join import JoinError
    from nirs4all_io.spec.enums import SpecError

    expected = case["expected"]
    if "refuse" in expected:
        error_type = JoinError if expected.get("error") == "JoinError" else SpecError
        with pytest.raises(error_type, match=expected["refuse"]) as error:
            _decimal_execute(tmp_path, case, form, method, target)
        print(_path_json.dumps({"decimal_trace": case["id"], "method": method, "target": target, "error": type(error.value).__name__, "message": str(error.value)}))
        for key in ("source", "file"):
            if key in expected:
                assert expected[key] in str(error.value)
        if method == "infer" and case.get("unresolved"):
            plan = nio.infer(_decimal_input(tmp_path, case, form))
            assert plan.params[case["unresolved"]]["delimiter"]["value"] is None
            assert any(case["unresolved"] in warning and expected["source"] in warning and "candidates" in warning for warning in plan.warnings)
        return
    assembled, spec, plan = _decimal_execute(tmp_path, case, form, method, target)
    if method != "explicit":
        assert spec.params.decimal_separator is None
        assert all(source.params.decimal_separator != "." for source in spec.sources)
    def check(block, exp):
        values = np.array(exp["x"], dtype=float)
        np.testing.assert_array_equal(block.X[0], values)
        np.testing.assert_array_equal(np.isfinite(block.X[0]), exp["x_mask"])
        assert block.feature_headers[0] == exp["headers"]
        if "y" in exp:
            np.testing.assert_array_equal(block.y, np.array(exp["y"], dtype=float))
            np.testing.assert_array_equal(np.isfinite(block.y), exp["y_mask"])
            assert block.y_headers == exp["y_headers"]
    check(assembled.blocks["train"], expected)
    if case.get("test_block"):
        check(assembled.blocks["test"], case["test_block"])
    for source_id, decimal in case.get("decimals", {}).items():
        source = next(s for s in spec.sources if s.id == source_id)
        assert source.params.decimal_separator == decimal
        if plan is not None and decimal == ",":
            assert plan.params[_Path(source.input).name]["decimal_separator"]["value"] == ","
    if case.get("ids"):
        assert spec.sample_index.key == "id"
