# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""V3 byte transport and declared partitions; doubles do not qualify native IO."""
import struct

import numpy as np
import pandas as pd
import pytest
from nirs4all_io._adapter import _matrix, to_spectrodataset
from nirs4all_io._package import DatasetPackage

import nirs4all_io as nio


def matrix(values, rows, cols, dtype="float64", validity=None):
    """Encode explicit test literals using the owner wire layout, without rounding policy."""
    return {"dtype": dtype, "n_rows": rows, "n_cols": cols,
            "data_bytes_hex": struct.pack("<" + ("d" if dtype == "float64" else "f") * len(values), *values).hex(),
            "validity": [True] * len(values) if validity is None else validity,
            "precision": {"dtype": dtype, "origin": "source-declared" if dtype == "float32" else "default", "source_dtype": "float32" if dtype == "float32" else None,
                          "requested_dtype": None, "consumer": None}}


def packet():
    blocks = {}
    for index, part in enumerate(("train", "test", "val", "predict")):
        offset = 10 * index
        blocks[part] = {
            "n_samples": 2, "source_ids": ["nir"],
            "x": [matrix([offset + 1, -0.0, offset + 3, 0], 2, 2, "float32", [True, True, True, False])],
            "feature_headers": [["a", "b"]], "feature_names": [["a", "b"]],
            "feature_axes": [[{"name": "feature", "unit": None, "coordinates": None}]],
            "representation_ids": ["signal_1d"], "header_units": ["text"], "signal_types": [None],
            "y": {"columns": [{"name": "y", "kind": "numeric", "block": matrix([offset + .25, 0], 2, 1, validity=[True, False])}]},
            "y_headers": ["y"], "y_categorical": {},
            "metadata": {"n_rows": 2, "columns": [{"name": "sid", "values": [part + "0", part + "1"]}]},
            "weights": matrix([1 + index, 2 + index], 2, 1), "weights_header": "w",
            "processings": [[{"name": "raw_copy", "matrix": matrix([offset + 4, 5, 6, 7], 2, 2, "float32")}]],
        }
    return {"assembled_schema_version": 3, "partition_order": list(blocks), "name": "partitions",
            "task_type": "regression", "signal_type": "auto", "n_sources": 1, "blocks": blocks,
            "identity": {"provenance": {"source_ids": ["nir"], "sample_id": "sid"}},
            "folds": [[[0], [1]]], "fold_provenance": [], "repetition": None, "aggregate": None}


class Recorder:
    def __init__(self, name):
        self.samples, self.targets, self.metadata, self.features = [], [], [], []
    def add_samples(self, x, selector, **kwargs):
        self.samples.append((selector["partition"], x))
    def add_targets(self, y):
        self.targets.append(y)
    def add_metadata(self, meta):
        self.metadata.append(meta)
    def add_features(self, arrays, names, source):
        self.features.append((arrays, names, source))
    def set_task_type(self, *args, **kwargs):
        pass
    def set_folds(self, folds):
        self.folds = folds


@pytest.mark.parametrize("version", [None, 1, 2, 3.0, True, "3", 4])
def test_v3_gate_refuses_other_versions(version):
    full = packet()
    full["assembled_schema_version"] = version
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        DatasetPackage(full)
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        to_spectrodataset(full, spectro_dataset_cls=Recorder)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_declared_numeric_bytes_validity_and_bits(dtype):
    values = [1 + 2**-30, -0.0, 0, .125]
    wire = matrix(values, 2, 2, dtype, [True, True, False, True])
    result = _matrix(wire)
    assert result.dtype == np.dtype(dtype)
    assert result[:1].tobytes() == bytes.fromhex(wire["data_bytes_hex"])[:result.dtype.itemsize * 2]
    assert np.isnan(result[1, 0])
    assert result[1, 1] == .125


@pytest.mark.parametrize("shape", [(0, 2), (1, 1), (2, 0)])
def test_empty_and_singleton_matrices_keep_shape(shape):
    rows, cols = shape
    assert _matrix(matrix([1.] * (rows * cols), rows, cols)).shape == shape


def test_four_partitions_payloads_masks_ids_processings_weights_folds():
    full = packet()
    ds = to_spectrodataset(full, spectro_dataset_cls=Recorder)
    assert [p for p, _ in ds.samples] == ["train", "test", "val", "predict"]
    for index, (part, values) in enumerate(ds.samples):
        assert values.dtype == np.float32
        assert values[0, 0] == 10 * index + 1
        assert np.signbit(values[0, 1]) and np.isnan(values[1, 1])
        assert len(ds.targets) == len(ds.metadata) == 1
        assert ds.targets[0].dtype == np.float64
        assert ds.targets[0][2 * index, 0] == 10 * index + .25
        assert np.isnan(ds.targets[0][2 * index + 1, 0])
        assert ds.metadata[0]["sid"].tolist()[2 * index:2 * index + 2] == [part + "0", part + "1"]
        assert ds.metadata[0]["__sample_weight__"].tolist()[2 * index:2 * index + 2] == [1 + index, 2 + index]
    assert ds.features[0][0][0][:, 0].tolist() == [4, 6, 14, 6, 24, 6, 34, 6]
    assert ds.folds == [([0], [1])]
    assembled = DatasetPackage(full).to_assembled()
    assert list(assembled.blocks) == full["partition_order"]
    assert assembled.blocks["val"].X[0][0, 0] == 21


def test_target_kinds_preserve_labels_codebooks_and_numeric_precision():
    full = packet()
    block = full["blocks"]["train"]
    labels = {"dtype": "string-label", "n_rows": 2, "n_cols": 1,
              "data_bytes_hex": (struct.pack("<Q", 3) + b"cat" + struct.pack("<Q", 0)).hex(),
              "validity": [True, False], "precision": {"dtype": "string-label", "origin": "default", "source_dtype": None, "requested_dtype": None, "consumer": None}}
    block["y"]["columns"] += [
        {"name": "class", "kind": "classification_integer", "block": matrix([16777217, 2], 2, 1)},
        {"name": "label", "kind": "classification_string", "block": labels}]
    block["y_headers"] = ["y", "class", "label"]
    block["y_categorical"] = {"label": {"categories": ["cat"], "source_dtype": None}}
    y = DatasetPackage(full).to_assembled().blocks["train"].y
    assert y.dtype == object
    assert y[0].tolist() == [0.25, 16777217., "cat"]
    assert isinstance(y[0, 1], float)
    assert np.isnan(y[1, 2])
    assert DatasetPackage(full).to_assembled().blocks["train"].y_categorical == block["y_categorical"]


def test_multi_source_package_arrays_keep_each_declared_dtype():
    full = packet()
    full["n_sources"] = 2
    full["identity"]["provenance"]["source_ids"] = ["nir", "aux"]
    for block in full["blocks"].values():
        block["source_ids"].append("aux")
        block["x"].append(matrix([1 + 2**-30, 2], 2, 1))
        block["feature_headers"].append(["c"])
        block["feature_names"].append(["c"])
        block["feature_axes"].append([{"name": "feature", "unit": None, "coordinates": None}])
        block["representation_ids"].append("signal_1d")
        block["header_units"].append("text")
        block["signal_types"].append(None)
        block["processings"].append([])
    assembled = DatasetPackage(full).to_assembled().blocks["train"]
    assert [x.dtype for x in assembled.X] == [np.dtype("float32"), np.dtype("float64")]
    assert assembled.X[1][0, 0] == 1 + 2**-30


def test_public_routes_delegate_storage_selection_to_native(monkeypatch):
    calls = []
    def transport(*args, **kwargs):
        calls.append(kwargs)
        return packet()
    monkeypatch.setattr(nio, "assembled_full", transport)
    monkeypatch.setattr(nio, "load_summary", transport)
    nio.load("transport.csv", target="assembled")
    package = nio.load("transport.csv", target="package")
    nio.load("transport.csv", target="spectrodataset", spectro_dataset_cls=Recorder)
    assert calls == [{}, {}, {"storage": {"consumer": "SpectroDataset"}}]
    assert isinstance(package, DatasetPackage)


# Retained Rust exporter SHA-256: 602960e1f6b09a417a590b1213bd37fd23d89445ff1c31d48065f22769bdf7c4.
# These packets test the codec only; no current extension was built for them.
RETAINED_RUST_V3 = {'N.float-1e30.ft': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['t'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 2, 'data_bytes_hex': '000000000000f83f000000000000f03f0000000000000440ea8ca039593e2946', 'validity': [True, True, True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x', 'm']], 'feature_names': [['x', 'm']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': None, 'y_headers': [], 'y_categorical': {}, 'metadata': None, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': []}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': True, 'reason': 'no stable sample-id key declared; sample identity falls back to row position', 'partitions': ['train'], 'fingerprint': 'bbcfe123f1cf4496f899b231030b40f787859e305707b5e0fef2a99c2be05624'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['t']}}, 'manifest': {'root': 'f896455bdecdd317fb446773a3ef7241cc753ae4eb6ff1846bc15055038d4919', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 2], 'axes': ['sample', 'feature'], 'content_hash': '683c38b89950faf62bbd6b107bb85b87b32c09c568f25112841d3633804c3a19', 'byte_len': 218, 'storage': 'inline', 'uri': None, 'codec': None}]}}}, 'N.float-1e30.tg': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['t'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f83f0000000000000440', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x']], 'feature_names': [['x']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': {'columns': [{'name': 'm', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f03fea8ca039593e2946', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['m'], 'y_categorical': {}, 'metadata': None, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': []}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': True, 'reason': 'no stable sample-id key declared; sample identity falls back to row position', 'partitions': ['train'], 'fingerprint': 'bbcfe123f1cf4496f899b231030b40f787859e305707b5e0fef2a99c2be05624'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['t']}}, 'manifest': {'root': 'a172aa7e883645d9cf550d3e7cce0e6ff6ccc452cbc23d3f47da98a94beb67d9', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample', 'feature'], 'content_hash': '4c1657c7212c6bed0e5110ad01f961ed09a8c72d5266b9cb26e133b519b46658', 'byte_len': 191, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/y', 'partition': 'train', 'role': 'targets', 'payload_kind': 'target_table', 'representation_id': 'target_numeric', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample'], 'content_hash': 'aef7c331f38a8863dc6e6292da53e008d853a832548ac1787092e91f13e6e50f', 'byte_len': 131, 'storage': 'inline', 'uri': None, 'codec': None}]}}}, 'N.float-missing-peer.tg': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['t'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f83f0000000000000440', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x']], 'feature_names': [['x']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': {'columns': [{'name': 'm', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '0000000000000000ea8ca039593e2946', 'validity': [False, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['m'], 'y_categorical': {}, 'metadata': None, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': [{'source': 't', 'na': {'strategy': 'ignore', 'na_detected': True, 'na_count': 1, 'na_samples': 1, 'na_features': 1, 'removed_samples': [], 'removed_features': [], 'fill_method': None, 'na_preserved': True}}]}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['t'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': True, 'reason': 'no stable sample-id key declared; sample identity falls back to row position', 'partitions': ['train'], 'fingerprint': 'bbcfe123f1cf4496f899b231030b40f787859e305707b5e0fef2a99c2be05624'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['t']}}, 'manifest': {'root': 'cc89ee00565f9a5bdf5b6cbdb0bd426c33734e1e25e2acd0aee4ab27b8c97ceb', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample', 'feature'], 'content_hash': '4c1657c7212c6bed0e5110ad01f961ed09a8c72d5266b9cb26e133b519b46658', 'byte_len': 191, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/y', 'partition': 'train', 'role': 'targets', 'payload_kind': 'target_table', 'representation_id': 'target_numeric', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample'], 'content_hash': '8f1965846b89778149bc1cf4faa9e4a97a8f928c2887fe68149616de755e8cc1', 'byte_len': 131, 'storage': 'inline', 'uri': None, 'codec': None}]}}}, 'T.mixed-categorical': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['d'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f83f0000000000000440', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x']], 'feature_names': [['x']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': {'columns': [{'name': 'a', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f03f0000000000000840', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}, {'name': 'c', 'kind': 'classification_string', 'block': {'dtype': 'string-label', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '03000000000000007265640000000000000000', 'validity': [True, False], 'precision': {'dtype': 'string-label', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['a', 'c'], 'y_categorical': {'c': {'categories': ['red']}}, 'metadata': None, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['d'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': [{'source': 'd', 'na': {'strategy': 'ignore', 'na_detected': True, 'na_count': 1, 'na_samples': 1, 'na_features': 1, 'removed_samples': [], 'removed_features': [], 'fill_method': None, 'na_preserved': True}}]}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['d'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': True, 'reason': 'no stable sample-id key declared; sample identity falls back to row position', 'partitions': ['train'], 'fingerprint': 'bbcfe123f1cf4496f899b231030b40f787859e305707b5e0fef2a99c2be05624'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['d']}}, 'manifest': {'root': '0bdc2b9386c02a7f9cdfc8450d9dd52a54b4001f8d81a58d53f8398cb27045c0', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample', 'feature'], 'content_hash': '18d8797e8b7fdf6d29607962217dbeb3ffba58120938b032a70b409ccf0ee601', 'byte_len': 191, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/y', 'partition': 'train', 'role': 'targets', 'payload_kind': 'target_table', 'representation_id': None, 'dtype': 'mixed', 'shape': [2, 2], 'axes': ['sample', 'target'], 'content_hash': 'f447dff5830a4c99b13388262860bb4419609e25f525ab3e2f855088c100e78f', 'byte_len': 200, 'storage': 'inline', 'uri': None, 'codec': None}]}}}, 'T.sel-ba': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['d'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f83f0000000000000440', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x']], 'feature_names': [['x']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': {'columns': [{'name': 'a', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f03f0000000000000840', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}, {'name': 'b', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '00000000000000400000000000001040', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['a', 'b'], 'y_categorical': {}, 'metadata': None, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['d'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': []}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['d'], 'sample_id': None, 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': True, 'reason': 'no stable sample-id key declared; sample identity falls back to row position', 'partitions': ['train'], 'fingerprint': 'bbcfe123f1cf4496f899b231030b40f787859e305707b5e0fef2a99c2be05624'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['d']}}, 'manifest': {'root': 'd4dd67b0fe6af1881f4d42f7a588870fea692ea337cb26f667b10c2d08d53645', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample', 'feature'], 'content_hash': '18d8797e8b7fdf6d29607962217dbeb3ffba58120938b032a70b409ccf0ee601', 'byte_len': 191, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/y', 'partition': 'train', 'role': 'targets', 'payload_kind': 'target_table', 'representation_id': 'target_numeric_matrix', 'dtype': 'float64', 'shape': [2, 2], 'axes': ['sample', 'target'], 'content_hash': '8777cee22748f6b1e7a79515aa38552316c5d3e15dcfcf47c0aa9b2c9ed5c4a8', 'byte_len': 197, 'storage': 'inline', 'uri': None, 'codec': None}]}}}, 'T.two-sources': {'full': {'partition_order': ['train'], 'assembled_schema_version': 3, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'blocks': {'train': {'n_samples': 2, 'source_ids': ['p'], 'x': [{'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f83f0000000000000440', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}], 'feature_headers': [['x']], 'feature_names': [['x']], 'feature_axes': [[{'name': 'feature', 'unit': None, 'coordinates': None}]], 'representation_ids': ['signal_1d'], 'header_units': ['text'], 'signal_types': [None], 'y': {'columns': [{'name': 'b', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '00000000000000400000000000001040', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}, {'name': 'a', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 2, 'n_cols': 1, 'data_bytes_hex': '000000000000f03f0000000000000840', 'validity': [True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['b', 'a'], 'y_categorical': {}, 'metadata': {'n_rows': 2, 'columns': [{'name': 'k', 'values': ['A', 'B']}]}, 'weights': None, 'weights_header': None, 'processings': [[]]}}, 'folds': [], 'fold_provenance': [], 'identity': {'provenance': {'source_ids': ['p'], 'sample_id': 'k', 'observation_id': None, 'repetition_id': None, 'group_id': None}}, 'repetition': None, 'aggregate': None, 'warnings': [], 'audits': []}, 'summary': {'schema_version': 4, 'name': 'dataset', 'task_type': 'auto', 'signal_type': 'auto', 'n_sources': 1, 'repetition': None, 'identity': {'provenance': {'source_ids': ['p'], 'sample_id': 'k', 'observation_id': None, 'repetition_id': None, 'group_id': None}, 'row_position_fallback': {'used': False, 'reason': "stable sample-id key 'k' is aligned in every partition", 'partitions': ['train'], 'fingerprint': '1e7e75a33bdb983f5bdaf33db04b128ff32904b25b5ecdf6f39a61b44ac0fdb3'}}, 'folds': [], 'fold_provenance': [], 'aggregate': None, 'partitions': {'train': {'n_samples': 2, 'source_ids': ['p']}}, 'manifest': {'root': 'debf16a1439505da611167f97b4658a58cbf532836b5db3d4e660d7c07c36a24', 'entries': [{'id': 'train/x0', 'partition': 'train', 'role': 'features', 'payload_kind': 'feature_matrix', 'representation_id': 'signal_1d', 'dtype': 'float64', 'shape': [2, 1], 'axes': ['sample', 'feature'], 'content_hash': 'bf017db7a0aa68f9cbe471c5302797a3918d87c1260207f614511ab808b9d3e6', 'byte_len': 191, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/y', 'partition': 'train', 'role': 'targets', 'payload_kind': 'target_table', 'representation_id': 'target_numeric_matrix', 'dtype': 'float64', 'shape': [2, 2], 'axes': ['sample', 'target'], 'content_hash': '05be30e451d8264567f0ca7e7934c81f84ea6c3fc28c17e604ac79265fcc6ec7', 'byte_len': 197, 'storage': 'inline', 'uri': None, 'codec': None}, {'id': 'train/metadata', 'partition': 'train', 'role': 'metadata', 'payload_kind': 'metadata_table', 'representation_id': 'sample_metadata', 'dtype': 'mixed', 'shape': [2, 1], 'axes': ['sample', 'field'], 'content_hash': '431614a8d7cb844f10804725783fce9c5d0ff9c8deed7324065f105bbe7cdb9b', 'byte_len': 145, 'storage': 'inline', 'uri': None, 'codec': None}]}}}}


@pytest.mark.parametrize("case", list(RETAINED_RUST_V3))
def test_retained_genuine_rust_v3_packet_and_owner_package_manifest(case):
    retained = RETAINED_RUST_V3[case]
    package = DatasetPackage(retained["full"])
    assert package.to_summary_dict() == retained["summary"]
    assembled = package.to_assembled()
    for partition, block in retained["full"]["blocks"].items():
        for index, matrix in enumerate(block["x"]):
            values = assembled.blocks[partition].X[index]
            observed = np.asarray(matrix["validity"], dtype=bool)
            original = np.frombuffer(bytes.fromhex(matrix["data_bytes_hex"]), dtype="<f8")
            assert values.dtype == np.float64
            assert values.reshape(-1)[observed].tobytes() == original[observed].tobytes()


def test_real_sdk_four_partitions_matches_direct_constructor():
    import inspect
    import json
    import os
    from pathlib import Path

    pytest.importorskip("polars", reason="Real SDK constructor unavailable without polars")
    from nirs4all.data import SpectroDataset

    origin = str(Path(inspect.getfile(SpectroDataset)).resolve())
    expected = os.environ.get("L2_SDK_EXPECTED")
    if expected is not None:
        assert origin == expected
    full = packet()
    projected = to_spectrodataset(full)
    direct = SpectroDataset(name="partitions")
    order = ["train", "test", "val", "predict"]
    # Direct inputs are independent literals, not the codec's decoded arrays.
    for index, part in enumerate(order):
        offset = index * 10
        direct.add_samples(np.array([[offset + 1, -0.0], [offset + 3, np.nan]], dtype=np.float32),
                           {"partition": part}, headers=["a", "b"], header_unit="text")
        direct.add_targets(np.array([[offset + .25], [np.nan]], dtype=np.float64))
        direct.add_metadata(pd.DataFrame({"sid": [part + "0", part + "1"], "__sample_weight__": [1. + index, 2. + index]}))
    direct.add_features([np.array([[4, 5], [6, 7], [14, 5], [6, 7], [24, 5], [6, 7], [34, 5], [6, 7]], dtype=np.float32)], ["raw_copy"], source=0)
    direct.set_task_type("regression", forced=True)
    direct.set_folds([([0], [1])])
    facts = {}
    assert projected.num_samples == direct.num_samples == 8
    for index, part in enumerate(order):
        selector = {"partition": part}
        left, right = np.asarray(projected.x(selector, layout="3d")), np.asarray(direct.x(selector, layout="3d"))
        assert left.shape == right.shape == (2, 2, 2)
        assert left.dtype == right.dtype == np.float32
        assert left.tobytes() == right.tobytes()
        assert np.isnan(left).tolist() == np.isnan(right).tolist()
        yl, yr = np.asarray(projected.y(selector)), np.asarray(direct.y(selector))
        assert yl.dtype == yr.dtype and yl.tobytes() == yr.tobytes()
        assert np.isnan(yl).tolist() == np.isnan(yr).tolist()
        assert projected.metadata_column("sid", selector).tolist() == direct.metadata_column("sid", selector).tolist() == [part + "0", part + "1"]
        assert projected.metadata_column("__sample_weight__", selector).tolist() == [1. + index, 2. + index]
        facts[part] = {"count": left.shape[0], "dtype": str(left.dtype), "x_bytes": left.tobytes().hex(), "ids": [part + "0", part + "1"]}
    assert [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in projected.folds] == [([0], [1])]
    assert projected.features_processings(0) == direct.features_processings(0)
    print(json.dumps({"real_sdk_origin": origin, "constructor_only": True, "source_only_io_transport": True, "partitions": facts}))


def test_memory_summary_uses_the_same_v3_gate(monkeypatch):
    from nirs4all_io import _inputs, _native

    monkeypatch.setattr(_inputs, "array_payload", lambda *args: ({}, []))
    monkeypatch.setattr(_native, "assemble_frames", lambda *args, **kwargs: {"assembled_schema_version": 2})
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        nio.load(object(), target="assembled")
    monkeypatch.setattr(_native, "assemble_frames", lambda *args, **kwargs: packet())
    assert nio.load(object(), target="assembled")["assembled_schema_version"] == 3


def test_declared_weight_validity_is_retained_without_narrowing():
    full = packet()
    full["blocks"]["train"]["weights"] = matrix([1 + 2**-30, 0], 2, 1, validity=[True, False])
    weights = DatasetPackage(full).to_assembled().blocks["train"].weights
    assert weights.dtype == np.float64
    assert weights[0] == 1 + 2**-30 and np.isnan(weights[1])
    ds = to_spectrodataset(full, spectro_dataset_cls=Recorder)
    assert ds.metadata[0]["__sample_weight__"].iloc[0] == 1 + 2**-30
    assert np.isnan(ds.metadata[0]["__sample_weight__"].iloc[1])


def test_malformed_byte_blocks_refuse_instead_of_using_legacy_data():
    block = matrix([1.], 1, 1)
    block["data"] = [42.]
    block["data_bytes_hex"] = "00"
    with pytest.raises(ValueError):
        _matrix(block)
    block = matrix([1.], 1, 1)
    block["validity"] = [1]
    with pytest.raises(ValueError, match="validity"):
        _matrix(block)


# Corrective B1-B4 cases below use literal v3 packets, never native assembly.
@pytest.fixture(params=["recorder", "sdk"])
def projection_constructor(request):
    if request.param == "recorder":
        return Recorder
    pytest.importorskip("polars", reason="Real SDK constructor unavailable without polars")
    import inspect
    import os
    from pathlib import Path

    from nirs4all.data import SpectroDataset

    assert str(Path(inspect.getfile(SpectroDataset)).resolve()) == os.environ["L2_SDK_EXPECTED"]
    return SpectroDataset


def _projection_packet():
    full = packet()
    full["folds"] = []
    full["task_type"] = "auto"
    for block in full["blocks"].values():
        block["metadata"] = None
        block["weights"] = None
        block["processings"] = [[]]
    return full


def _projected_targets(ds):
    return ds.targets[0] if isinstance(ds, Recorder) else np.asarray(ds.y({"y": "raw"}))


def _metadata_cells(ds, column):
    # Polars -> pandas inference can itself round nullable integer columns.
    return ds.metadata[0][column].tolist() if isinstance(ds, Recorder) else ds.metadata().get_column(column).to_list()


@pytest.mark.parametrize("missing", ["train", "test"])
def test_projection_internal_unlabelled_rows_are_nan(projection_constructor, missing):
    full = _projection_packet()
    full["blocks"][missing]["y"] = None
    full["blocks"][missing]["y_headers"] = []
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    actual = _projected_targets(ds).reshape(-1)
    expected = [.25, np.nan, 10.25, np.nan, 20.25, np.nan, 30.25, np.nan]
    start = 0 if missing == "train" else 2
    expected[start:start + 2] = [np.nan, np.nan]
    np.testing.assert_equal(actual, expected)
    if isinstance(ds, Recorder):
        assert len(ds.targets) == 1


def test_projection_trailing_unlabelled_keeps_oracle_extent(projection_constructor):
    full = _projection_packet()
    for part in ("val", "predict"):
        full["blocks"][part]["y"] = None
        full["blocks"][part]["y_headers"] = []
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    if isinstance(ds, Recorder):
        assert ds.targets[0].shape == (4, 1)
    else:
        np.testing.assert_equal(np.asarray(ds.y({"partition": "test"})).reshape(-1), [10.25, np.nan])
        assert np.asarray(ds.y({"y": "raw"})).shape[0] == 4
        assert np.asarray(ds.y({"partition": "predict", "y": "raw"})).shape[0] == 0


def test_projection_metadata_and_weights_share_one_aligned_table(projection_constructor):
    full = _projection_packet()
    full["blocks"]["val"]["metadata"] = {"n_rows": 2, "columns": [{"name": "sid", "values": ["v0", "v1"]}]}
    full["blocks"]["test"]["weights"] = matrix([7., 8.], 2, 1)
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    assert _metadata_cells(ds, "sid")[:6] == [None, None, None, None, "v0", "v1"]
    weights = _metadata_cells(ds, "__sample_weight__")[:6]
    assert weights[2:4] == [7., 8.]
    assert all(pd.isna(weights[index]) for index in (0, 1, 4, 5))
    if isinstance(ds, Recorder):
        assert len(ds.metadata) == 1


def test_projection_exact_integer_metadata_with_null(projection_constructor):
    full = _projection_packet()
    full["blocks"]["train"]["metadata"] = {"n_rows": 2, "columns": [{"name": "exact", "values": [9007199254740993, None]}]}
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    value = _metadata_cells(ds, "exact")[0]
    assert int(value) == 9007199254740993
    assert isinstance(value, (int, np.integer))


@pytest.mark.parametrize("axis", [False, True])
def test_projection_feature_and_processing_permutation(projection_constructor, axis):
    full = _projection_packet()
    full["blocks"] = {part: full["blocks"][part] for part in ("train", "test")}
    for part, values, proc in (("train", [1., 100., 2., 200.], [5., 500., 6., 501.]),
                               ("test", [300., 3., 400., 4.], [600., 7., 700., 8.])):
        b = full["blocks"][part]
        b["x"] = [matrix(values, 2, 2, "float32")]
        b["processings"] = [[{"name": "smooth", "matrix": matrix(proc, 2, 2, "float32")}]]
    full["blocks"]["test"]["feature_headers"] = [["b", "a"]]
    if axis:
        full["blocks"]["test"]["feature_headers"] = [["display-left", "display-right"]]
        for part, coords in (("train", [1000., 1005.]), ("test", [1005., 1000.])):
            full["blocks"][part]["feature_axes"] = [[{"name": "wavelength", "unit": "nm", "coordinates": {
                "kind": "numeric", "count": 2, "data_bytes_hex": struct.pack("<dd", *coords).hex()}}]]
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    if isinstance(ds, Recorder):
        actual = ds.samples[1][1]
        processing = ds.features[0][0][0][2:]
    else:
        table = np.asarray(ds.x({"partition": "test"}, layout="3d"))
        actual, processing = table[:, 0], table[:, 1]
    np.testing.assert_equal(actual, [[3., 300.], [4., 400.]])
    np.testing.assert_equal(processing, [[7., 600.], [8., 700.]])


def _string_target(name, values):
    raw = b"".join(struct.pack("<Q", len(value.encode())) + value.encode() for value in values)
    return {"name": name, "kind": "classification_string", "block": {"dtype": "string-label", "n_rows": len(values), "n_cols": 1,
            "data_bytes_hex": raw.hex(), "validity": [True] * len(values), "precision": {
                "dtype": "string-label", "origin": "default", "source_dtype": None, "requested_dtype": None, "consumer": None}}}


@pytest.mark.parametrize("mixed", [False, True])
def test_projection_target_columns_align_by_name(projection_constructor, mixed):
    full = _projection_packet()
    full["blocks"] = {part: full["blocks"][part] for part in ("train", "test")}
    for part, numeric, second in (("train", [1.5, 2.5], [100.5, 200.5]), ("test", [3.5, 4.5], [300.5, 400.5])):
        columns = [{"name": "p", "kind": "numeric", "block": matrix(numeric, 2, 1)},
                   _string_target("q", ["A", "B"]) if mixed else {"name": "q", "kind": "numeric", "block": matrix(second, 2, 1)}]
        if part == "test":
            columns.reverse()
        full["blocks"][part]["y"] = {"columns": columns}
        full["blocks"][part]["y_headers"] = [c["name"] for c in columns]
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    actual = _projected_targets(ds)
    np.testing.assert_equal(actual[:, 0].astype(float), [1.5, 2.5, 3.5, 4.5])
    assert actual[:, 1].tolist() == (["A", "B", "A", "B"] if mixed else [100.5, 200.5, 300.5, 400.5])


def test_projection_reordered_source_ids(projection_constructor):
    from copy import deepcopy

    full = _projection_packet()
    full["n_sources"] = 2
    full["blocks"] = {part: full["blocks"][part] for part in ("train", "test")}
    fields = ("x", "feature_headers", "feature_names", "feature_axes", "representation_ids", "header_units", "signal_types", "processings")
    for part, block in full["blocks"].items():
        block["source_ids"] = ["nir", "mir"]
        for field in fields:
            block[field].append(deepcopy(block[field][0]))
        block["x"][1] = matrix([50., 51., 52., 53.], 2, 2, "float32")
        block["feature_headers"][1] = ["c", "d"]
        if part == "test":
            block["source_ids"].reverse()
            for field in fields:
                block[field].reverse()
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    arrays = ds.samples[1][1] if isinstance(ds, Recorder) else ds.x({"partition": "test"}, concat_source=False)
    np.testing.assert_equal(np.asarray(arrays[0]).reshape(2, -1)[:, 0], [11., 13.])
    np.testing.assert_equal(np.asarray(arrays[1]).reshape(2, -1), [[50., 51.], [52., 53.]])


@pytest.mark.parametrize("defect", ["names", "duplicate_reorder", "unit", "axis_unit", "signal", "representation", "processing_missing", "processing_rename", "source_duplicate", "source_partial", "source_count", "rows", "target_kind", "target_names", "target_width", "metadata_rows", "weight_rows"])
def test_projection_layout_refuses_before_sdk(defect):
    from copy import deepcopy

    full = _projection_packet()
    a, b = full["blocks"]["train"], full["blocks"]["test"]
    if defect == "names":
        b["feature_headers"] = [["c", "d"]]
    elif defect == "duplicate_reorder":
        a["feature_headers"], b["feature_headers"] = [["a", "a"]], [["a", "b"]]
    elif defect in {"unit", "signal", "representation"}:
        field, value = {"unit": ("header_units", "nm"), "signal": ("signal_types", "absorbance"), "representation": ("representation_ids", "feature_block_set")}[defect]
        b[field][0] = value
    elif defect == "axis_unit":
        b["feature_axes"][0][0]["unit"] = "nm"
    elif defect in {"processing_missing", "processing_rename"}:
        for block in full["blocks"].values():
            block["processings"] = [[{"name": "smooth", "matrix": deepcopy(block["x"][0])}]]
        b["processings"] = [[]] if defect == "processing_missing" else [[{"name": "other", "matrix": deepcopy(b["x"][0])}]]
    elif defect == "source_duplicate":
        b["source_ids"] = ["nir", "nir"]
    elif defect == "source_partial":
        for block in full["blocks"].values():
            block["source_ids"] = ["nir", "mir"]
            for field in ("x", "feature_headers", "feature_names", "feature_axes", "representation_ids", "header_units", "signal_types", "processings"):
                block[field].append(deepcopy(block[field][0]))
        b["source_ids"] = ["nir", "uv"]
    elif defect == "source_count":
        b["x"].append(deepcopy(b["x"][0]))
    elif defect == "rows":
        b["n_samples"] = 3
    elif defect == "target_kind":
        b["y"] = {"columns": [_string_target("y", ["A", "B"])]}
    elif defect == "target_names":
        b["y_headers"] = ["other"]
    elif defect == "target_width":
        b["y"]["columns"][0]["block"] = matrix([1., 2., 3., 4.], 2, 2)
    elif defect == "metadata_rows":
        b["metadata"] = {"n_rows": 2, "columns": [{"name": "id", "values": [1]}]}
    elif defect == "weight_rows":
        b["weights"] = matrix([1.], 1, 1)
    class NeverConstruct:
        def __init__(self, *args, **kwargs):
            pytest.fail("SDK was constructed before complete layout validation")
    with pytest.raises(ValueError):
        to_spectrodataset(full, spectro_dataset_cls=NeverConstruct)


@pytest.mark.parametrize("role", ["x", "processing"])
def test_projection_default_float64_delegates_owner_selection(monkeypatch, role):
    """Transport only: the selected response is an independent literal, not a kernel."""
    from nirs4all_io import _native

    full = _projection_packet()
    original = matrix([1.5, -0., 0., 2.], 2, 2, validity=[True, True, False, True])
    selected = matrix([1.5, -0., 0., 2.], 2, 2, "float32", [True, True, False, True])
    selected["precision"] = {"dtype": "float32", "origin": "consumer", "source_dtype": None, "requested_dtype": None, "consumer": "SpectroDataset"}
    if role == "x":
        full["blocks"]["train"]["x"] = [original]
    else:
        for b in full["blocks"].values():
            b["processings"] = [[{"name": "selected", "matrix": b["x"][0]}]]
        full["blocks"]["train"]["processings"][0][0]["matrix"] = original
    calls = []
    def select(wire):
        assert wire is original
        calls.append(wire)
        return selected
    monkeypatch.setattr(_native, "select_spectrodataset_features", select, raising=False)
    ds = nio.load(DatasetPackage(full), target="spectrodataset", spectro_dataset_cls=Recorder)
    assert calls == [original]
    actual = ds.samples[0][1] if role == "x" else ds.features[0][0][0][:2]
    np.testing.assert_equal(actual, [[1.5, -0.], [np.nan, 2.]])
    assert np.signbit(actual[0, 1]) and actual.dtype == np.float32
    assert original["dtype"] == "float64" and original["precision"]["consumer"] is None


@pytest.mark.parametrize("role", ["x", "processing"])
@pytest.mark.parametrize("value,message", [(1e300, "Value overflows float32 storage"), (1e-320, "Nonzero value rounds to zero in float32 storage"),
                                         (1.5, "SpectroDataset does not support an explicit float64 numerical guarantee")])
def test_projection_owner_selection_error_propagates_before_sdk(monkeypatch, role, value, message):
    """A labelled owner error double proves delegation and propagation only."""
    from nirs4all_io import _native

    full = _projection_packet()
    original = matrix([value, 2., 3., 4.], 2, 2)
    if value == 1.5:
        original["precision"].update(origin="requested", requested_dtype="float64")
    if role == "x":
        full["blocks"]["val"]["x"] = [original]
    else:
        for b in full["blocks"].values():
            b["processings"] = [[{"name": "selected", "matrix": b["x"][0]}]]
        full["blocks"]["val"]["processings"][0][0]["matrix"] = original
    error = ValueError(message)
    calls = []
    def select(wire):
        assert wire is original
        calls.append(wire)
        raise error
    monkeypatch.setattr(_native, "select_spectrodataset_features", select, raising=False)
    class NeverConstruct:
        def __init__(self, *args, **kwargs):
            pytest.fail("SDK constructed before owner selection completed")
    with pytest.raises(ValueError, match=message) as caught:
        to_spectrodataset(full, spectro_dataset_cls=NeverConstruct)
    assert caught.value is error and calls == [original]


def test_projection_requested_float64_target_refuses_before_sdk():
    full = _projection_packet()
    record = full["blocks"]["val"]["y"]["columns"][0]["block"]["precision"]
    record.update(origin="requested", requested_dtype="float64")
    class NeverConstruct:
        def __init__(self, *args, **kwargs):
            pytest.fail("SDK constructed before requested target precision refusal")
    with pytest.raises(ValueError, match="float64"):
        to_spectrodataset(full, spectro_dataset_cls=NeverConstruct)


def test_projection_string_targets_keep_codes_across_internal_missing_partition(projection_constructor):
    full = _projection_packet()
    full["task_type"] = "binary"
    for part, labels in (("train", ["A", "B"]), ("val", ["B", "A"]), ("predict", ["A", "B"])):
        full["blocks"][part]["y"] = {"columns": [_string_target("label", labels)]}
        full["blocks"][part]["y_headers"] = ["label"]
        full["blocks"][part]["y_categorical"] = {"label": {"categories": sorted(set(labels))}}
    full["blocks"]["test"]["y"] = None
    full["blocks"]["test"]["y_headers"] = []
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    raw = _projected_targets(ds).reshape(-1)
    assert raw[[0, 1, 4, 5, 6, 7]].tolist() == ["A", "B", "B", "A", "A", "B"]
    assert all(pd.isna(raw[index]) for index in [2, 3])
    if not isinstance(ds, Recorder):
        train = np.asarray(ds.y({"partition": "train"})).reshape(-1)
        val = np.asarray(ds.y({"partition": "val"})).reshape(-1)
        assert train[0] == val[1] and train[1] == val[0]


def test_projection_disjoint_declared_source_ids_follow_validated_positional_contract():
    full = _projection_packet()
    for index, block in enumerate(full["blocks"].values()):
        block["source_ids"] = ["partition-source-" + str(index)]
    ds = to_spectrodataset(full, spectro_dataset_cls=Recorder)
    assert [values[0, 0] for _, values in ds.samples] == [1., 11., 21., 31.]


def test_projection_processing_order_is_canonical(projection_constructor):
    full = _projection_packet()
    for part, block in full["blocks"].items():
        entries = [{"name": name, "matrix": matrix(values, 2, 2, "float32")}
                   for name, values in (("first", [1., 2., 3., 4.]), ("second", [5., 6., 7., 8.]))]
        block["processings"] = [entries if part == "train" else entries[::-1]]
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    if isinstance(ds, Recorder):
        assert [entry[1][0] for entry in ds.features] == ["first", "second"]
        np.testing.assert_equal(ds.features[0][0][0], [[1., 2.], [3., 4.]] * 4)
    else:
        assert ds.features_processings(0)[-2:] == ["first", "second"]
        np.testing.assert_equal(np.asarray(ds.x({"partition": "test"}, layout="3d"))[:, 1], [[1., 2.], [3., 4.]])


def test_projection_val_only_metadata_alignment(projection_constructor):
    full = _projection_packet()
    full["blocks"]["val"]["metadata"] = {"n_rows": 2, "columns": [{"name": "id", "values": ["v0", "v1"]}]}
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    assert _metadata_cells(ds, "id")[:6] == [None, None, None, None, "v0", "v1"]


def test_projection_test_only_weights_alignment(projection_constructor):
    full = _projection_packet()
    full["blocks"]["test"]["weights"] = matrix([7., 8.], 2, 1)
    ds = to_spectrodataset(full, spectro_dataset_cls=projection_constructor)
    values = _metadata_cells(ds, "__sample_weight__")[:4]
    assert len(values) == 4 and all(pd.isna(value) for value in values[:2]) and values[2:] == [7., 8.]


@pytest.mark.parametrize("values,error", [
    ([9007199254740993, None], None), ([9007199254740993, 1.5], "ArrowInvalid"),
    ([18446744073709551615, None], "OverflowError"), ([True, False], None),
    ([True, None], None), (["a", None], None), ([1.5, None], None),
    ([None, None], None), ([9007199254740993, "x"], None), ([9007199254740993, 2], None),
])
def test_projection_metadata_shapes_match_direct_sdk_constructor(values, error):
    """SDK-owned conversion/refusals must match direct exact-object input."""
    pytest.importorskip("polars", reason="Real SDK constructor unavailable without polars")
    import inspect
    import os
    from pathlib import Path

    import pyarrow
    from nirs4all.data import SpectroDataset

    assert str(Path(inspect.getfile(SpectroDataset)).resolve()) == os.environ["L2_SDK_EXPECTED"]
    full = _projection_packet()
    full["blocks"] = {"train": full["blocks"]["train"]}
    full["blocks"]["train"]["metadata"] = {"n_rows": 2, "columns": [{"name": "m", "values": values}]}
    def direct():
        ds = SpectroDataset(name="direct")
        ds.add_samples(np.array([[1., -0.], [3., np.nan]], dtype=np.float32), {"partition": "train"}, headers=["a", "b"], header_unit="text")
        ds.add_metadata(pd.DataFrame({"m": pd.Series(values, dtype=object)}))
        return ds
    if error:
        exception = {"ArrowInvalid": pyarrow.ArrowInvalid, "OverflowError": OverflowError}[error]
        with pytest.raises(exception):
            direct()
        with pytest.raises(exception):
            to_spectrodataset(full)
    else:
        assert to_spectrodataset(full).metadata().to_dicts() == direct().metadata().to_dicts()
