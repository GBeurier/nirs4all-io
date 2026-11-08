# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Controls of the SpectroDataset consumer policy against the REAL nirs4all SDK.

Dev/test-only (``pytest -m parity``): skipped unless ``nirs4all`` is importable.
The policy under test: features/processings are an explicit F32 consumer
projection; default targets keep their genuine raw values/dtypes and the SDK's
own encoder owns any numeric view (never relabelled as the IO base receipt);
explicit float64 feature/numerical-target guarantees refuse; explicit float32
targets obey the owner integer cap and exact-label rules.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nirs4all.data", reason="the consumer controls need the real nirs4all SDK")
pytestmark = pytest.mark.parity

import nirs4all_io as nio  # noqa: E402
from nirs4all_io.materialize import NumericContractError  # noqa: E402
from nirs4all_io.materialize.assemble import AssembledDataset, PartitionBlock  # noqa: E402
from nirs4all_io.materialize.spectrodataset import project_spectrodataset  # noqa: E402


def _assembled(y: np.ndarray, task: str, x: np.ndarray | None = None) -> AssembledDataset:
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]]) if x is None else x
    assembled = AssembledDataset(name="sdk-controls", task_type=task, signal_type="auto", n_sources=1)
    block = PartitionBlock(n_samples=len(x), source_ids=["X"], X=[x], feature_headers=[["0", "1"]], header_units=["index"], signal_types=[None])
    block.y, block.y_headers, block.y_source_dtype = y, ["y"], y.dtype.name
    assembled.blocks["train"] = block
    return assembled


def _y(ds, view: str) -> np.ndarray:
    return np.asarray(ds.y({"partition": "train", "y": view}))


def test_regression_f64_targets_stay_raw_while_features_are_the_f32_consumer() -> None:
    y = np.array([[1.0], [1.0 + 2.0**-30], [2.0], [2.0 + 2.0**-29]])
    x = np.array([[1.0 + 2.0**-30, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]])
    ds, receipt = project_spectrodataset(_assembled(y, "regression", x))
    raw = _y(ds, "raw")
    assert raw.dtype == np.float64 and raw.tobytes() == y.tobytes()
    numeric = _y(ds, "numeric")  # existing SDK behaviour, not an IO claim
    assert numeric.dtype == np.float32 and numeric[0, 0] == numeric[1, 0]
    assert np.asarray(ds.x({"partition": "train"})).dtype == np.float32
    assert receipt["features"][0]["precision"]["origin"] == "consumer" and receipt["numerical_target_view"] is None


def test_integer_and_string_classification_keep_raw_labels_and_the_sdk_encoder() -> None:
    ints = np.array([[16777217], [16777218], [16777217], [16777218]], dtype=np.int64)
    ds, _ = project_spectrodataset(_assembled(ints, "binary"))
    raw = _y(ds, "raw")
    assert raw.dtype == np.int64 and np.array_equal(raw, ints)
    assert _y(ds, "numeric").dtype == np.float32 and _y(ds, "numeric").ravel().tolist() == [0.0, 1.0, 0.0, 1.0]
    strings = np.array([["alpha"], ["beta"], ["alpha"], ["beta"]])
    ds, _ = project_spectrodataset(_assembled(strings, "binary"))
    assert _y(ds, "raw").ravel().tolist() == ["alpha", "beta", "alpha", "beta"]


def test_explicit_f64_requests_refuse_before_the_sdk_is_used() -> None:
    assembled = _assembled(np.zeros((4, 1)), "regression")
    for options in ({"features": "float64"}, {"targets": "float64"}):
        with pytest.raises(NumericContractError):
            project_spectrodataset(assembled, **options)  # type: ignore[arg-type]


def test_explicit_f32_targets_obey_exact_label_rules_on_the_real_sdk() -> None:
    ok = _assembled(np.array([[16777218], [16777216], [16777218], [16777216]], dtype=np.int64), "binary")
    ds, receipt = project_spectrodataset(ok, targets="float32")
    raw = _y(ds, "raw")
    assert raw.dtype == np.float32 and raw.ravel().tolist() == [16777218.0, 16777216.0, 16777218.0, 16777216.0]
    assert receipt["targets"]["precision"][0]["origin"] == "requested"
    bad = _assembled(np.array([[16777217], [16777216], [16777217], [16777216]], dtype=np.int64), "binary")
    with pytest.raises(NumericContractError, match="not exact in float32"):
        project_spectrodataset(bad, targets="float32")
    regression = _assembled(np.array([[16777217], [16777216], [16777217], [16777216]], dtype=np.int64), "regression")
    ds, _ = project_spectrodataset(regression, targets="float32")  # numeric regression may round
    assert _y(ds, "raw").dtype == np.float32
    with pytest.raises(NumericContractError, match="numeric precision request"):
        project_spectrodataset(_assembled(np.array([["a"], ["b"], ["a"], ["b"]]), "binary"), targets="float32")


def test_load_default_target_is_the_real_spectrodataset() -> None:
    ds = nio.load((np.array([[0.5, 1.5], [1.5, 2.5]]), np.array([1.0 + 2.0**-30, 2.0])))
    assert type(ds).__name__ == "SpectroDataset"
    assert np.asarray(ds.x({"partition": "train"})).dtype == np.float32


def test_independently_typed_processing_keeps_its_own_source_on_the_real_sdk(tmp_path) -> None:
    ulp = 1.0 + 2.0**-30
    np.save(tmp_path / "x.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.save(tmp_path / "v.npy", np.array([[ulp, 2.0], [3.0, 4.0]], dtype=np.float64))
    spec = {"sources": [{"id": "x", "role": "features", "input": "x.npy", "params": {"header_unit": "index"}, "variations": [{"name": "v", "input": "v.npy"}]}]}
    # No explicit request: an explicit features request is committed into the assembled IR and then stays a
    # "requested" origin at the consumer boundary (tests/test_owner_corrective_c1.py), so this consumer-origin
    # receipt starts from the raw assembly.
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    ds, receipt = project_spectrodataset(assembled)
    assert receipt["features"][0]["precision"]["source_dtype"] == "float32" and receipt["features"][0]["precision"]["origin"] == "consumer"
    assert receipt["processings"][0]["precision"] == {"dtype": "float32", "origin": "consumer", "source_dtype": "float64", "requested_dtype": None, "consumer": "SpectroDataset"}
    cube = np.asarray(ds.x({"partition": "train"}, layout="3d", concat_source=True))
    assert cube.dtype == np.float32 and cube[0, 1, 0] == np.float32(ulp) == np.float32(1.0), "the real SDK stores the F32 consumer projection"
    default = nio.load(spec, base_dir=tmp_path)  # the F32 consumer is the common request
    assert np.array_equal(np.asarray(default.x({"partition": "train"}, layout="3d", concat_source=True)), cube)
    with pytest.raises(NumericContractError, match="float64"):
        project_spectrodataset(assembled, features="float64")


def test_inherited_explicit_f64_guarantees_refuse_on_the_real_sdk_route_while_raw_targets_stay_supported() -> None:
    from nirs4all_io.materialize import DatasetPackage

    x, y = np.array([[1.0 + 2.0**-30, 2.0], [3.0, 4.0]]), np.array([2.0 + 2.0**-30, 3.0])
    explicit_targets = nio.to_dataset_package((x, y), targets="float64")
    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        project_spectrodataset(explicit_targets.to_assembled())
    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        DatasetPackage.from_assembled(explicit_targets.to_assembled(), consumer="SpectroDataset")
    explicit_features = nio.to_dataset_package((x, y), features="float64")
    with pytest.raises(NumericContractError, match="SpectroDataset does not support requested float64"):
        project_spectrodataset(explicit_features.to_assembled())
    # the default raw F64 targets remain a supported SDK operation: the raw values survive, the SDK owns its numeric view
    default = nio.to_dataset_package((x, y))
    ds, receipt = project_spectrodataset(default.to_assembled())
    raw = _y(ds, "raw")
    assert raw.dtype == np.float64 and raw.tobytes() == y.reshape(-1, 1).tobytes() and receipt["numerical_target_view"] is None
    assert np.asarray(ds.x({"partition": "train"})).dtype == np.float32


def test_array_route_raw_targets_and_mixed_cells_on_the_real_sdk() -> None:
    n = 2**54 + 2**30 + 1
    x = np.ones((3, 2))
    ints = nio.load((x, [1, 2, 1]))  # raw integer list: no blanket cast, the SDK owns its encoder
    assert _y(ints, "raw").dtype.kind == "i" and _y(ints, "raw").ravel().tolist() == [1, 2, 1]
    strings = nio.load((x, ["a", "b", "a"]))
    assert _y(strings, "raw").ravel().tolist() == ["a", "b", "a"]
    mixed = nio.load((np.ones((2, 2)), [n, 0.5]), targets="float32")
    raw = _y(mixed, "raw")
    assert raw.dtype == np.float32 and raw.tobytes()[:4].hex() == "0100805a", "explicit F32: direct integer rounding (upper), through the real SDK"
    with pytest.raises(NumericContractError):
        nio.load((np.ones((2, 2)), [2**53 + 1, 0.5]))  # default storage refuses the inexact retained integer


def test_finalized_assembly_keeps_raw_int64_labels_on_the_real_sdk() -> None:
    """C1: committing the requested feature storage never rewrites unrequested raw labels (int64 stays int64)."""
    y = np.array([16777218, 16777216, 16777218, 16777216], dtype=np.int64)
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 8.0]])
    before, _ = project_spectrodataset(nio.load((x, y), target="assembled"))
    finalized = nio.load((x, y), target="assembled", features="float32")
    after, receipt = project_spectrodataset(finalized)
    assert _y(before, "raw").dtype == _y(after, "raw").dtype == np.int64
    assert _y(before, "raw").tobytes() == _y(after, "raw").tobytes() and receipt["numerical_target_view"] is None
    assert receipt["features"][0]["precision"]["origin"] == "requested"
    reg = np.array([1.0, 1.0 + 2.0**-30, 2.0, 2.0 + 2.0**-29])
    plain, _ = project_spectrodataset(nio.load((x, reg), target="assembled"))
    kept, _ = project_spectrodataset(nio.load((x, reg), target="assembled", features="float32"))
    assert _y(plain, "raw").tobytes() == _y(kept, "raw").tobytes() and _y(kept, "raw").dtype == np.float64


def test_sdk_inline_multimodal_adapter_refuses_the_serialized_declared_float_bypass() -> None:
    """E4/B3: the SDK's only direct ``MultimodalDataset.from_dict`` caller now meets the one array gate."""
    from nirs4all.api.studio_scientific_general import STUDIO_MULTIMODAL_DATASET_SCHEMA, StudioScientificJobError, _inline_dataset_arrays

    from nirs4all_io import strict_json

    def cohort(dtype: str, cell: str) -> dict:
        return strict_json.loads(
            '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"m","sample_ids":["a"],'
            '"sources":[{"name":"s","sample_ids":["a"],"representation_id":"tabular_numeric","axes":["sample","feature"],"feature_names":null,'
            '"axis_units":{"feature":null},"axis_coordinates":{},"array":{"dtype":"' + dtype + '","shape":[1,1],"values":[[' + cell + ']]},'
            '"presence_mask":{"dtype":"bool","shape":[1],"values":[true]}}],'
            '"y":{"dtype":"float64","shape":[1],"values":[1.0]},"target_names":["y"],"target_mask":{"dtype":"bool","shape":[1],"values":[true]},'
            '"groups":null,"partitions":{"dtype":"<U5","shape":[1],"values":["train"]}}'
        )

    for dtype, cell in (("float32", "5e-324"), ("float64", "9007199254740993")):
        with pytest.raises(StudioScientificJobError) as caught:
            _inline_dataset_arrays({"schema": STUDIO_MULTIMODAL_DATASET_SCHEMA, "cohort": cohort(dtype, cell)})
        assert caught.value.__cause__ is not None and isinstance(caught.value.__cause__, ValueError)
    ok = _inline_dataset_arrays({"schema": STUDIO_MULTIMODAL_DATASET_SCHEMA, "cohort": cohort("float32", "18014399583223809")})
    assert np.ascontiguousarray(next(iter(ok.sources.values())).values).tobytes().hex() == "0100805a"
