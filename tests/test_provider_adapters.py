# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Real sklearn and optional Torch consumers of materialized provider views."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nirs4all_io import MultimodalDataset, RaggedSeriesBatch, RaggedSeriesSource, TensorSource
from nirs4all_io.provider import DataProvider
from nirs4all_io.provider_adapters import SklearnProviderAdapter, TorchRaggedSeriesBatch, collate_provider_samples


def _cohort(*, masked=False, prediction=False, mixed=True, labels=False):
    ids = tuple(f"row-{i}" for i in range(11))
    rng = np.random.default_rng(17)
    nir = rng.normal(size=(11, 4))
    image = rng.normal(size=(11, 2, 3, 3))
    order = np.arange(11)[::-1]
    presence = np.ones(11, dtype=bool)
    if masked:
        presence[5] = False
    sources = {
        "nir": TensorSource(nir, ids, representation_id="signal_1d"),
        "image": TensorSource(image[order], [ids[i] for i in order], representation_id="rgb_image", presence_mask=presence[order]),
        "series": TensorSource(rng.normal(size=(11, 7, 2)), ids, representation_id="series_mv"),
    }
    if mixed:
        sources["metadata"] = TensorSource([[float(i), "A" if i % 2 else "B"] for i in range(11)], ids, representation_id="tabular_mixed")
    y = np.column_stack([nir[:, 0] - 0.3 * nir[:, 1], nir[:, 2] + 0.1 * nir[:, 3]])
    if labels:
        y = np.array(["alpha" if i % 2 else "beta" for i in range(11)])
    mask = np.ones(y.shape, dtype=bool)
    if masked:
        mask[-1, 1] = False
        y[-1, 1] = np.nan
    return MultimodalDataset(
        sources, sample_ids=ids, y=None if prediction else y,
        target_mask=None if prediction else mask,
        groups=[f"unit-{i}" for i in range(11)], partitions=["train"] * 8 + ["test"] * 3,
        task_type="classification" if labels else "regression",
    )


def _provider(cohort):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return cohort

    return DataProvider(generate, provider_id="tests.frameworks", seed=17), calls


def test_sklearn_import_does_not_load_torch_or_nirs4all():
    script = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, RejectTorch())
from nirs4all_io.provider_adapters import SklearnProviderAdapter
assert 'torch' not in sys.modules
assert 'nirs4all' not in sys.modules
try:
    from nirs4all_io.provider_adapters import TorchMapDataset
except ImportError as error:
    assert 'optional PyTorch' in str(error)
else:
    raise AssertionError('Torch dependency refusal missing')
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_torch_ragged_batch_import_does_not_load_torch_or_nirs4all():
    script = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately unavailable')
sys.meta_path.insert(0, RejectTorch())
from nirs4all_io.provider_adapters import TorchRaggedSeriesBatch
assert TorchRaggedSeriesBatch.__name__ == 'TorchRaggedSeriesBatch'
assert 'torch' not in sys.modules
assert 'nirs4all' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_adapters_require_materialization_and_never_call_generator():
    cohort = _cohort()
    provider, calls = _provider(cohort)
    with pytest.raises(RuntimeError, match="materializ"):
        SklearnProviderAdapter(provider)
    assert calls == []
    provider.materialize()
    adapter = SklearnProviderAdapter(provider)
    x, y = adapter.arrays(["row-8", "row-1"])
    assert len(calls) == 1
    assert list(x) == list(cohort.sources)
    assert x["image"].shape == (2, 2, 3, 3)
    assert x["series"].shape == (2, 7, 2)
    assert x["metadata"].tolist() == [[8.0, "B"], [1.0, "A"]]
    np.testing.assert_array_equal(y, cohort.y[[8, 1]])
    list(adapter.batches(3))
    assert len(calls) == 1


def test_generated_view_is_consumable_by_sklearn_and_torch_without_regeneration():
    torch = pytest.importorskip("torch")
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    from nirs4all_io.provider_adapters import TorchMapDataset

    base = _cohort()
    callbacks = []

    def generate(**_):
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids, **_):
        callbacks.append(tuple(sample_ids))
        source = base.take(sample_ids).sources["nir"]
        return {"sample_ids": list(sample_ids), "sources": {"nir": TensorSource(
            np.asarray(source.values) + 10.0, sample_ids, representation_id=source.representation_id,
        )}}

    provider = DataProvider(
        generate, generate_view=generate_view, provider_id="tests.frameworks.view",
        base=base, replace_sources=["nir"],
    )
    provider.materialize()
    ids = ["row-7", "row-2", "row-5"]
    view = provider.materialize_view(ids, view_key="view:v1:" + "a" * 64)
    sklearn_x, sklearn_y = SklearnProviderAdapter(view).arrays()
    assert set(sklearn_x) == set(base.sources)
    np.testing.assert_array_equal(sklearn_x["nir"], base.take(ids).sources["nir"].values + 10.0)
    np.testing.assert_array_equal(sklearn_x["image"], base.take(ids).sources["image"].values)
    np.testing.assert_array_equal(sklearn_y, base.take(ids).y)
    Ridge().fit(SklearnProviderAdapter(view, source="nir").arrays()[0], sklearn_y)

    loader = torch.utils.data.DataLoader(
        TorchMapDataset(view, return_metadata=True), batch_size=len(ids), collate_fn=collate_provider_samples,
    )
    batch = next(iter(loader))
    assert batch["sample_id"] == ids
    np.testing.assert_array_equal(batch["X"]["nir"].numpy(), sklearn_x["nir"])
    np.testing.assert_array_equal(batch["X"]["image"].numpy(), sklearn_x["image"])
    np.testing.assert_array_equal(batch["y"].numpy(), sklearn_y)
    assert callbacks == [tuple(ids)]


def test_sklearn_matrix_fit_is_the_same_real_ridge_fit():
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge
    cohort = _cohort()
    selected = [f"row-{i}" for i in range(8)]
    x, y = SklearnProviderAdapter(cohort, source="nir").arrays(selected)
    fitted = Ridge(alpha=0.7).fit(x, y)
    reference = Ridge(alpha=0.7).fit(cohort.sources["nir"].values[:8], cohort.y[:8])
    np.testing.assert_array_equal(fitted.predict(x), reference.predict(x))
    x[0, 0] = 999
    assert cohort.sources["nir"].values[0, 0] != 999


def test_sklearn_partial_fit_uses_actual_incremental_estimator_and_batches():
    SGDRegressor = pytest.importorskip("sklearn.linear_model").SGDRegressor
    cohort = _cohort()
    adapter = SklearnProviderAdapter(cohort, source="nir")
    kwargs = dict(random_state=17, shuffle=False, learning_rate="constant", eta0=0.01)
    fitted, reference = SGDRegressor(**kwargs), SGDRegressor(**kwargs)
    for x, y in adapter.batches(3, sample_ids=cohort.sample_ids[:8]):
        fitted.partial_fit(x, y[:, 0])
    for start in range(0, 8, 3):
        reference.partial_fit(cohort.sources["nir"].values[start:min(start + 3, 8)], cohort.y[start:min(start + 3, 8), 0])
    np.testing.assert_array_equal(fitted.predict(cohort.sources["nir"].values), reference.predict(cohort.sources["nir"].values))


def test_partial_fit_classifier_keeps_declared_training_classes():
    SGDClassifier = pytest.importorskip("sklearn.linear_model").SGDClassifier
    cohort = _cohort(labels=True)
    adapter = SklearnProviderAdapter(cohort, source="nir")
    model = SGDClassifier(random_state=17, shuffle=False)
    for x, y in adapter.batches(3, sample_ids=cohort.sample_ids[:8]):
        model.partial_fit(x, y, classes=np.array(["alpha", "beta"]))
    assert model.classes_.tolist() == ["alpha", "beta"]
    assert model.predict(adapter.arrays()[0]).shape == (11,)


def test_explicit_source_refuses_implicit_flattening_and_unknown_source():
    with pytest.raises(ValueError, match="rank 2"):
        SklearnProviderAdapter(_cohort(), source="image")
    with pytest.raises(ValueError, match="Unknown source"):
        SklearnProviderAdapter(_cohort(), source="missing")
    with pytest.raises(TypeError, match="materialized"):
        SklearnProviderAdapter(iter([]))


def test_sklearn_masks_ids_and_selection_are_explicit():
    cohort = _cohort(masked=True)
    adapter = SklearnProviderAdapter(cohort)
    with pytest.raises(ValueError, match="source masks"):
        adapter.arrays()
    with pytest.raises(ValueError, match="target_mask"):
        SklearnProviderAdapter(cohort, source="nir").arrays()
    item = adapter.arrays(["row-10", "row-5"], return_metadata=True)
    assert item["sample_ids"] == ("row-10", "row-5")
    np.testing.assert_array_equal(item["target_mask"], [[True, False], [True, True]])
    np.testing.assert_array_equal(item["source_masks"]["image"], [True, False])
    assert item["groups"].tolist() == ["unit-10", "unit-5"]
    assert item["partitions"].tolist() == ["test", "train"]
    # Missingness outside the explicitly selected samples does not reject a view.
    assert adapter.arrays(["row-0"])[1].shape == (1, 2)


@pytest.mark.parametrize("size,start", [(0, 0), (True, 0), (2, -1), (2, 12), (2, True)])
def test_invalid_batch_offsets_are_refused(size, start):
    with pytest.raises(ValueError):
        list(SklearnProviderAdapter(_cohort()).batches(size, start=start))


def test_batch_start_drop_last_empty_selection_and_prediction_only():
    adapter = SklearnProviderAdapter(_cohort(), source="nir")
    result = list(adapter.batches(2, sample_ids=["row-9", "row-1", "row-7", "row-2"], start=1, drop_last=True, return_metadata=True))
    assert [item["sample_ids"] for item in result] == [("row-1", "row-7")]
    assert list(adapter.batches(2, sample_ids=[])) == []
    assert SklearnProviderAdapter(_cohort(prediction=True)).arrays()[1] is None


@pytest.mark.parametrize("kind", ["TorchMapDataset", "TorchIterableDataset"])
@pytest.mark.parametrize("workers", [0, 2])
def test_real_torch_loader_preserves_all_ids_modalities_and_masks(kind, workers):
    torch = pytest.importorskip("torch")
    from nirs4all_io import provider_adapters

    cohort = _cohort(masked=True)
    provider, calls = _provider(cohort)
    provider.materialize()
    dataset = getattr(provider_adapters, kind)(provider, return_metadata=True)
    assert isinstance(dataset, torch.utils.data.IterableDataset if "Iterable" in kind else torch.utils.data.Dataset)
    # Generator is local/unpicklable; capturing only its cohort permits spawn.
    restored = pickle.loads(pickle.dumps(dataset))
    kwargs = {"multiprocessing_context": "spawn"} if workers else {}
    loader = torch.utils.data.DataLoader(
        restored, batch_size=3, num_workers=workers, drop_last=False,
        collate_fn=collate_provider_samples, generator=torch.Generator().manual_seed(17), **kwargs,
    )
    records = {}
    for batch in loader:
        assert tuple(batch["X"]["image"].shape[1:]) == (2, 3, 3)
        assert tuple(batch["X"]["series"].shape[1:]) == (7, 2)
        assert batch["y"].dtype == torch.float64
        for index, sample_id in enumerate(batch["sample_id"]):
            assert sample_id not in records
            records[sample_id] = (batch["X"]["nir"][index].numpy(), batch["X"]["metadata"][index], batch["source_masks"]["image"][index].item(), batch["target_mask"][index].numpy())
    assert set(records) == set(cohort.sample_ids)
    assert len(calls) == 1
    for index, sample_id in enumerate(cohort.sample_ids):
        nir, metadata, presence, mask = records[sample_id]
        np.testing.assert_array_equal(nir, cohort.sources["nir"].values[index])
        assert metadata == cohort.sources["metadata"].values[index].tolist()
        assert presence == bool(cohort.sources["image"].presence_mask[index])
        np.testing.assert_array_equal(mask, cohort.target_mask[index])


def test_torch_tuple_mode_string_labels_and_prediction_only():
    torch = pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    cohort = _cohort(labels=True, mixed=False)
    dataset = TorchMapDataset(cohort, sample_ids=["row-3", "row-1", "row-4"])
    x, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=collate_provider_samples)))
    assert y == ["alpha", "alpha", "beta"]
    assert x["image"].shape == (3, 2, 3, 3)
    with pytest.raises(ValueError, match="source masks"):
        TorchMapDataset(_cohort(masked=True))
    assert len(TorchMapDataset(cohort, sample_ids=[])) == 0
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(TypeError):
        dataset[True]
    prediction = TorchMapDataset(_cohort(prediction=True), return_metadata=True)[0]
    assert "y" not in prediction and "target_mask" not in prediction


def test_binding_adapter_mirror_is_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/nirs4all_io/provider_adapters.py").read_bytes() == (root / "bindings/python/python/nirs4all_io/provider_adapters.py").read_bytes()


def _ragged_cohort(*, equal_length=False, shifted_time=False, masked=False):
    ids = ["row-0", "row-1"]
    times = np.array([0., 1., 2., 4.])
    if shifted_time:
        times = 100 + 10 * times
    return MultimodalDataset(
        {
            "nir": TensorSource(np.arange(4).reshape(2, 2), ids, representation_id="signal_1d"),
            "series": RaggedSeriesSource(np.arange(8, dtype=np.float32).reshape(4, 2), [0, 2 if equal_length else 1, 4], ids,
                                         time_coordinates=times, channel_names=["one", "two"], time_unit="h", presence_mask=[not masked, True]),
        }, sample_ids=ids, y=[1., np.nan if masked else 2.], target_mask=[True, not masked], task_type="regression",
    )


@pytest.mark.parametrize("equal_length", [False, True])
def test_sklearn_named_ragged_keeps_times_and_lengths_without_regenerating(equal_length):
    first = _ragged_cohort(equal_length=equal_length)
    provider, calls = _provider(first)
    provider.materialize()
    adapter = SklearnProviderAdapter(provider)
    x, y = adapter.arrays(["row-1", "row-0"])
    batch = x["series"]
    expected = first.sources["series"].take(["row-1", "row-0"]).values
    assert isinstance(batch, RaggedSeriesBatch)
    assert batch is not expected and not np.shares_memory(batch.values, provider.cohort.sources["series"].values.values)
    np.testing.assert_array_equal(batch.values, expected.values)
    np.testing.assert_array_equal(batch.time_coordinates, expected.time_coordinates)
    np.testing.assert_array_equal(batch.offsets, expected.offsets)
    assert not batch.values.flags.writeable and not batch.time_coordinates.flags.writeable
    np.testing.assert_array_equal(y, [2., 1.])
    changed, _ = SklearnProviderAdapter(_ragged_cohort(equal_length=equal_length, shifted_time=True)).arrays(["row-1", "row-0"])
    np.testing.assert_array_equal(batch.values, changed["series"].values)
    assert not np.array_equal(batch.time_coordinates, changed["series"].time_coordinates)
    singles = list(adapter.batches(1, sample_ids=["row-1", "row-0"]))
    assert all(isinstance(item[0]["series"], RaggedSeriesBatch) for item in singles)
    assert singles[0][0]["series"].time_coordinates.tolist() == expected.time_coordinates[:expected.lengths[0]].tolist()
    assert len(calls) == 1


def test_sklearn_ragged_metadata_keeps_both_masks_and_matrix_selection_stays_explicit():
    cohort = _ragged_cohort(masked=True)
    adapter = SklearnProviderAdapter(cohort)
    with pytest.raises(ValueError, match="source masks"):
        adapter.arrays()
    item = adapter.arrays(["row-1", "row-0"], return_metadata=True)
    assert isinstance(item["X"]["series"], RaggedSeriesBatch)
    assert item["source_masks"]["series"].tolist() == [True, False]
    assert item["target_mask"].tolist() == [False, True]
    assert item["sample_ids"] == ("row-1", "row-0")
    assert item["X"]["series"].lengths.tolist() == [3, 1]
    with pytest.raises(ValueError, match="rank 2"):
        SklearnProviderAdapter(cohort, source="series")
    matrix, y = SklearnProviderAdapter(cohort, source="nir").arrays(["row-0"])
    np.testing.assert_array_equal(matrix, [[0, 1]])
    np.testing.assert_array_equal(y, [1.])


def test_sklearn_ragged_can_feed_an_explicit_transformer_and_real_estimator():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    cohort = _ragged_cohort()
    x, y = SklearnProviderAdapter(cohort).arrays()

    def summarize(batch):
        assert isinstance(batch, RaggedSeriesBatch)
        return np.vstack([batch[index].mean(axis=0) for index in range(len(batch))])

    estimator = make_pipeline(FunctionTransformer(summarize, validate=False), Ridge(alpha=0.3))
    estimator.fit(x["series"], y)
    reference = Ridge(alpha=0.3).fit(summarize(cohort.sources["series"].values), y)
    np.testing.assert_allclose(estimator.predict(x["series"]), reference.predict(summarize(x["series"])))


@pytest.mark.parametrize("kind", ["TorchMapDataset", "TorchIterableDataset"])
@pytest.mark.parametrize("equal_length", [False, True])
@pytest.mark.parametrize("return_metadata", [False, True])
def test_torch_ragged_refuses_before_time_coordinates_can_be_discarded(kind, equal_length, return_metadata, monkeypatch):
    pytest.importorskip("torch")
    from nirs4all_io import provider_adapters

    def forbid_sample(*args, **kwargs):
        pytest.fail("ragged data reached Torch sample extraction")

    monkeypatch.setattr(provider_adapters._TorchSamples, "_sample", forbid_sample)
    for shifted_time in (False, True):
        provider, calls = _provider(_ragged_cohort(equal_length=equal_length, shifted_time=shifted_time))
        provider.materialize()
        for selection in (None, ["row-0"]):
            with pytest.raises(ValueError, match="ragged sources.*typed collation.*time coordinates"):
                getattr(provider_adapters, kind)(provider, sample_ids=selection, return_metadata=return_metadata)
        assert len(calls) == 1


@pytest.mark.parametrize("kind", ["TorchMapDataset", "TorchIterableDataset"])
def test_torch_ragged_refuses_even_empty_or_entirely_absent_sources(kind):
    pytest.importorskip("torch")
    from nirs4all_io import provider_adapters

    cohort = MultimodalDataset(
        {"series": RaggedSeriesSource(np.empty((0, 2)), [0], [])},
        sample_ids=["row-0", "row-1"], source_alignment="left",
    )
    for selection in (None, []):
        with pytest.raises(ValueError, match="ragged sources.*typed collation"):
            getattr(provider_adapters, kind)(cohort, sample_ids=selection, return_metadata=True)


def _packed_cohort(*, with_times=True, prediction=False):
    ids = ["a", "b", "hidden", "empty", "c"]
    values = np.arange(16, dtype=np.float32).reshape(8, 2)
    values[3:5] = np.nan
    source = RaggedSeriesSource(
        values, [0, 2, 3, 3, 5, 8], ["c", "a", "empty", "hidden", "b"],
        time_coordinates=np.array([0., 1.5, 7., 4., 5., 1., 3., 9.]) if with_times else None,
        channel_names=["temperature", "humidity"], time_unit="h", presence_mask=[True, True, False, False, True],
    )
    y = np.arange(10, dtype=np.float64).reshape(5, 2)
    y[2, 1] = np.nan
    mask = np.ones((5, 2), dtype=bool)
    mask[2, 1] = False
    return MultimodalDataset(
        {"series": source, "metadata": TensorSource([[float(i), name] for i, name in enumerate(ids)], ids, representation_id="tabular_mixed")},
        sample_ids=ids, y=None if prediction else y, target_mask=None if prediction else mask,
        groups=[f"unit-{name}" for name in ids], partitions=["train"] * 3 + ["test"] * 2,
        task_type="regression",
    )


def _assert_packed_records(batch, cohort):
    torch = pytest.importorskip("torch")
    packed = batch["X"]["series"]
    assert isinstance(packed, TorchRaggedSeriesBatch)
    rows = [cohort.sample_ids.index(sample_id) for sample_id in batch["sample_id"]]
    source = cohort.sources["series"]
    expected = source.values.take_rows(rows)
    np.testing.assert_array_equal(packed.values.numpy(), expected.values)
    np.testing.assert_array_equal(packed.offsets.numpy(), expected.offsets)
    np.testing.assert_array_equal(packed.lengths.numpy(), expected.lengths)
    np.testing.assert_array_equal(packed.presence_mask.numpy(), source.presence_mask[rows])
    np.testing.assert_array_equal(packed.presence_mask.numpy(), batch["source_masks"]["series"].numpy())
    assert packed.values.dtype == torch.float32
    assert packed.offsets.dtype == packed.lengths.dtype == torch.int64
    assert packed.presence_mask.dtype == torch.bool
    assert packed.channel_names == source.channel_names and packed.time_unit == source.time_unit
    assert len(packed) == len(rows)
    if expected.time_coordinates is None:
        assert packed.time_coordinates is None
    else:
        assert packed.time_coordinates.dtype == torch.float64
        np.testing.assert_array_equal(packed.time_coordinates.numpy(), expected.time_coordinates)
    assert batch["partition"] == cohort.partitions[rows].tolist()
    assert batch["group"] == cohort.groups[rows].tolist()
    assert batch["X"]["metadata"] == cohort.sources["metadata"].values[rows].tolist()
    if cohort.y is None:
        assert "y" not in batch and "target_mask" not in batch
    else:
        np.testing.assert_array_equal(batch["y"].numpy(), cohort.y[rows])
        np.testing.assert_array_equal(batch["target_mask"].numpy(), cohort.target_mask[rows])


@pytest.mark.parametrize("equal_length", [False, True])
def test_torch_packed_preserves_distinct_times_and_supports_device_only_moves(equal_length):
    torch = pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    outputs = []
    for shifted in (False, True):
        cohort = _ragged_cohort(equal_length=equal_length, shifted_time=shifted)
        dataset = TorchMapDataset(cohort, sample_ids=["row-1", "row-0"], ragged_policy="packed", return_metadata=True)
        assert isinstance(dataset[0]["X"]["series"], RaggedSeriesSource)
        item = collate_provider_samples([dataset[0], dataset[1]])
        packed = item["X"]["series"]
        expected = dataset.cohort.sources["series"]
        np.testing.assert_array_equal(packed.values.numpy(), expected.values.values)
        np.testing.assert_array_equal(packed.time_coordinates.numpy(), expected.time_coordinates)
        np.testing.assert_array_equal(packed.offsets.numpy(), expected.offsets)
        assert item["sample_id"] == ["row-1", "row-0"]
        assert packed.channel_names == ("one", "two") and packed.time_unit == "h"
        moved = pickle.loads(pickle.dumps(packed)).to(torch.device("cpu"), non_blocking=True)
        torch.testing.assert_close(moved.values, packed.values)
        assert moved.values.dtype == torch.float32 and moved.time_coordinates.dtype == torch.float64
        assert moved.offsets.dtype == moved.lengths.dtype == torch.int64 and moved.presence_mask.dtype == torch.bool
        meta = packed.to("meta")
        assert meta.values.device.type == meta.offsets.device.type == meta.time_coordinates.device.type == "meta"
        assert meta.channel_names == packed.channel_names and meta.time_unit == packed.time_unit
        with pytest.raises(TypeError):
            packed.to(torch.float64)
        outputs.append(packed)
        packed.values[0, 0] = -99
        assert dataset.cohort.sources["series"].values.values[0, 0] != -99
    torch.testing.assert_close(outputs[0].values, outputs[1].values)
    assert not torch.equal(outputs[0].time_coordinates, outputs[1].time_coordinates)


def test_torch_packed_sample_owns_storage_without_an_intermediate_batch(monkeypatch):
    pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    cohort = _packed_cohort()
    source = cohort.sources["series"]

    def forbidden(*args, **kwargs):
        raise AssertionError("one sample must not allocate an intermediate ragged batch")

    monkeypatch.setattr(type(source.values), "take_rows", forbidden)
    dataset = TorchMapDataset(cohort, ragged_policy="packed", return_metadata=True)
    for position, sample_id in enumerate(cohort.sample_ids):
        sample = dataset[position]
        row = sample["X"]["series"]
        begin, end = source.offsets[position:position + 2]
        assert row.sample_ids == (sample_id,)
        assert row.offsets.tolist() == [0, end - begin]
        assert row.presence_mask.tolist() == [source.presence_mask[position]]
        np.testing.assert_array_equal(row.values.values, source.values[position])
        np.testing.assert_array_equal(row.time_coordinates, source.time_coordinates[begin:end])
        assert not np.shares_memory(row.values.values, source.values.values)
        assert not np.shares_memory(row.time_coordinates, source.time_coordinates)
        assert not row.values.values.flags.writeable and not row.time_coordinates.flags.writeable


@pytest.mark.parametrize("with_times", [False, True])
@pytest.mark.parametrize("prediction", [False, True])
def test_torch_packed_keeps_empty_and_hidden_series_and_target_masks(with_times, prediction):
    torch = pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    cohort = _packed_cohort(with_times=with_times, prediction=prediction)
    dataset = TorchMapDataset(cohort, ragged_policy="packed", return_metadata=True)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=collate_provider_samples)))
    _assert_packed_records(batch, cohort)
    packed = batch["X"]["series"]
    assert packed.lengths.tolist() == [1, 3, 2, 0, 2]
    assert packed.presence_mask.tolist() == [True, True, False, False, True]
    assert torch.isnan(packed.values[4:6]).all()
    absent = MultimodalDataset({"series": RaggedSeriesSource(np.empty((0, 2), dtype=np.float32), [0], [],
        time_coordinates=np.empty(0, dtype=np.float64) if with_times else None)}, sample_ids=["x", "y"], source_alignment="left")
    empty_dataset = TorchMapDataset(absent, ragged_policy="packed", return_metadata=True)
    empty = collate_provider_samples([empty_dataset[1], empty_dataset[0]])["X"]["series"]
    assert empty.values.shape == (0, 2) and empty.offsets.tolist() == [0, 0, 0]
    assert empty.lengths.tolist() == [0, 0] and empty.presence_mask.tolist() == [False, False]
    assert (empty.time_coordinates is None) == (not with_times)
    assert list(torch.utils.data.DataLoader(TorchMapDataset(absent, sample_ids=[], ragged_policy="packed", return_metadata=True), collate_fn=collate_provider_samples)) == []


@pytest.mark.parametrize("kind", ["TorchMapDataset", "TorchIterableDataset"])
@pytest.mark.parametrize("workers", [0, 2])
def test_torch_packed_real_workers_preserve_sampler_identity_and_never_regenerate(kind, workers):
    torch = pytest.importorskip("torch")
    from nirs4all_io import provider_adapters

    provider, calls = _provider(_packed_cohort())
    provider.materialize()
    dataset = getattr(provider_adapters, kind)(provider, sample_ids=list(reversed(provider.cohort.sample_ids)), ragged_policy="packed", return_metadata=True)
    restored = pickle.loads(pickle.dumps(dataset))
    kwargs = {"multiprocessing_context": "spawn"} if workers else {}
    if kind == "TorchMapDataset":
        kwargs["sampler"] = [4, 0, 2, 1, 2, 3]
    loader = torch.utils.data.DataLoader(restored, batch_size=2, num_workers=workers, pin_memory=True,
        collate_fn=collate_provider_samples, generator=torch.Generator().manual_seed(17), **kwargs)
    seen = []
    for batch in loader:
        _assert_packed_records(batch, dataset.cohort)
        seen.extend(batch["sample_id"])
    if kind == "TorchMapDataset":
        assert seen == [dataset.cohort.sample_ids[i] for i in [4, 0, 2, 1, 2, 3]]
    else:
        assert len(seen) == len(set(seen)) == len(dataset)
        assert set(seen) == set(dataset.cohort.sample_ids)
        if workers == 0:
            assert seen == list(dataset.cohort.sample_ids)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["TorchMapDataset", "TorchIterableDataset"])
def test_torch_packed_requires_explicit_metadata_and_supported_dtypes_before_sampling(kind, monkeypatch):
    pytest.importorskip("torch")
    from nirs4all_io import provider_adapters

    monkeypatch.setattr(provider_adapters._TorchSamples, "_sample", lambda *args: pytest.fail("Invalid input reached sample extraction"))
    factory = getattr(provider_adapters, kind)
    for policy in (None, True, "pad", [], {}):
        with pytest.raises(ValueError, match="ragged_policy"):
            factory(_ragged_cohort(), ragged_policy=policy, return_metadata=True)
    with pytest.raises(ValueError, match="return_metadata=True"):
        factory(_ragged_cohort(), ragged_policy="packed")
    non_native = np.dtype("float32").newbyteorder("S")
    dtypes = [(non_native, np.dtype("float64")), (np.dtype("float32"), non_native)]
    if np.dtype(np.longdouble).itemsize > 8:
        dtypes.extend([(np.dtype(np.longdouble), np.dtype("float64")), (np.dtype("float32"), np.dtype(np.longdouble))])
    for dtype, time_dtype in dtypes:
        source = RaggedSeriesSource(np.ones((2, 1), dtype=dtype), [0, 2], ["a"], time_coordinates=np.array([0, 1], dtype=time_dtype))
        cohort = MultimodalDataset({"series": source}, sample_ids=["a"])
        with pytest.raises(ValueError, match="dtype"):
            factory(cohort, ragged_policy="packed", return_metadata=True)


@pytest.mark.parametrize("change", ["dtype", "time_dtype", "coordinates", "unit", "names", "channels", "metadata_keys", "identity", "presence"])
def test_torch_packed_rejects_inconsistent_collation_without_dtype_promotion(change):
    pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    dataset = TorchMapDataset(_ragged_cohort(), ragged_policy="packed", return_metadata=True)
    samples = [dataset[0], dataset[1]]
    source = samples[1]["X"]["series"]
    kwargs = dict(values=source.values.values, offsets=source.offsets, sample_ids=source.sample_ids,
                  time_coordinates=source.time_coordinates, time_unit=source.time_unit, channel_names=source.channel_names)
    if change == "dtype":
        kwargs["values"] = source.values.values.astype(np.float64)
    elif change == "time_dtype":
        kwargs["time_coordinates"] = source.time_coordinates.astype(np.float32)
    elif change == "coordinates":
        kwargs["time_coordinates"] = None
    elif change == "unit":
        kwargs["time_unit"] = "s"
    elif change == "names":
        kwargs["channel_names"] = list(reversed(source.channel_names))
    elif change == "channels":
        kwargs["values"] = source.values.values[:, :1]
        kwargs["channel_names"] = ["one"]
    elif change == "metadata_keys":
        samples[1].pop("target_mask")
    elif change == "identity":
        samples[1]["sample_id"] = "another"
    else:
        samples[1]["source_masks"]["series"] = False
    samples[1]["X"]["series"] = RaggedSeriesSource(**kwargs)
    with pytest.raises(ValueError, match="schema|metadata"):
        collate_provider_samples(samples)


def test_torch_packed_pin_memory_on_real_cuda_allocator():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Physical pinned memory qualification requires a CUDA allocator")
    from nirs4all_io.provider_adapters import TorchMapDataset

    dataset = TorchMapDataset(_packed_cohort(), ragged_policy="packed", return_metadata=True)
    original = collate_provider_samples([dataset[0], dataset[1]])["X"]["series"]
    pinned = original.pin_memory()
    assert pinned is not original and not original.values.is_pinned()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=collate_provider_samples, pin_memory=True)))
    for packed in (pinned, batch["X"]["series"]):
        assert all(tensor.is_pinned() for tensor in (packed.values, packed.offsets, packed.lengths, packed.presence_mask, packed.time_coordinates))
        moved = packed.to("cuda", non_blocking=True).to("cpu")
        torch.testing.assert_close(moved.values, packed.values, equal_nan=True)
        assert moved.values.dtype == packed.values.dtype and moved.channel_names == packed.channel_names


@pytest.mark.parametrize("value_dtype,time_dtype", [(np.bool_, np.int64), (np.int16, np.float32), (np.float64, np.float64)])
def test_torch_packed_preserves_numeric_dtypes_and_unencoded_labels(value_dtype, time_dtype):
    torch = pytest.importorskip("torch")
    from nirs4all_io.provider_adapters import TorchMapDataset

    values = np.arange(4).reshape(2, 2).astype(value_dtype)
    times = np.array([0, 2], dtype=time_dtype)
    cohort = MultimodalDataset(
        {"series": RaggedSeriesSource(values, [0, 1, 2], ["a", "b"], time_coordinates=times)},
        sample_ids=["a", "b"], y=["red", "green"], task_type="classification", groups=np.array([11, 12], dtype=np.int32),
    )
    dataset = TorchMapDataset(cohort, ragged_policy="packed", return_metadata=True)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_provider_samples)))
    packed = batch["X"]["series"]
    assert packed.values.numpy().dtype == values.dtype
    assert packed.time_coordinates.numpy().dtype == times.dtype
    np.testing.assert_array_equal(packed.values.numpy(), values)
    np.testing.assert_array_equal(packed.time_coordinates.numpy(), times)
    assert packed.channel_names is None and packed.time_unit is None
    assert batch["y"] == ["red", "green"] and batch["group"].tolist() == [11, 12]


def test_sklearn_matrix_selection_never_materializes_unrequested_modalities(monkeypatch):
    original = _cohort(masked=True)
    sources = dict(original.sources)
    sources["series"] = RaggedSeriesSource(
        np.arange(44, dtype=np.int16).reshape(22, 2), np.arange(0, 23, 2), original.sample_ids,
        time_coordinates=np.tile([0., 1.5], 11), channel_names=["a", "b"], time_unit="s",
    )
    cohort = MultimodalDataset(sources, sample_ids=original.sample_ids, y=original.y, target_mask=original.target_mask,
        groups=original.groups, partitions=original.partitions, task_type="regression")
    adapter = SklearnProviderAdapter(cohort, source="nir")

    def forbidden(*args, **kwargs):
        pytest.fail("Selecting nir must not materialize a cohort or project another modality")

    monkeypatch.setattr(MultimodalDataset, "take", forbidden)
    monkeypatch.setattr(TensorSource, "take", forbidden)
    monkeypatch.setattr(RaggedSeriesSource, "take", forbidden)
    monkeypatch.setattr(RaggedSeriesBatch, "take_rows", forbidden)
    ids = ["row-8", "row-1", "row-5", "row-3"]
    rows = [8, 1, 5, 3]
    # row-5's absent image and missing targets outside this view are irrelevant.
    x, y = adapter.arrays(ids)
    np.testing.assert_array_equal(x, cohort.sources["nir"].values[rows])
    np.testing.assert_array_equal(y, cohort.y[rows])
    assert x.flags.writeable and y.flags.writeable
    assert not np.shares_memory(x, cohort.sources["nir"].values)
    x[0, 0], y[0, 0] = -999, -999
    assert cohort.sources["nir"].values[8, 0] != -999 and cohort.y[8, 0] != -999
    item = adapter.arrays(ids, return_metadata=True)
    assert item["sample_ids"] == tuple(ids) and set(item["source_masks"]) == {"nir"}
    np.testing.assert_array_equal(item["target_mask"], cohort.target_mask[rows])
    np.testing.assert_array_equal(item["groups"], cohort.groups[rows])
    np.testing.assert_array_equal(item["partitions"], cohort.partitions[rows])
    for array in (item["source_masks"]["nir"], item["target_mask"], item["groups"], item["partitions"]):
        assert array.flags.writeable
    batches = list(adapter.batches(2, sample_ids=iter(ids), start=1, drop_last=True, return_metadata=True))
    assert [batch["sample_ids"] for batch in batches] == [("row-1", "row-5")]
    np.testing.assert_array_equal(batches[0]["X"], cohort.sources["nir"].values[[1, 5]])
    empty_x, empty_y = adapter.arrays([])
    assert empty_x.shape == (0, 4) and empty_y.shape == (0, 2)
    assert empty_x.dtype == x.dtype and empty_y.dtype == y.dtype


def test_sklearn_named_selection_keeps_ragged_times_and_masks_without_cohort_copy(monkeypatch):
    cohort = _packed_cohort()
    ids = ["c", "empty", "hidden", "a"]
    rows = [cohort.sample_ids.index(sample_id) for sample_id in ids]
    expected = cohort.sources["series"].values.take_rows(rows)

    def forbidden(*args, **kwargs):
        pytest.fail("Named array selection must project requested buffers directly")

    monkeypatch.setattr(MultimodalDataset, "take", forbidden)
    monkeypatch.setattr(TensorSource, "take", forbidden)
    monkeypatch.setattr(RaggedSeriesSource, "take", forbidden)
    adapter = SklearnProviderAdapter(cohort)
    item = adapter.arrays(ids, return_metadata=True)
    batch = item["X"]["series"]
    assert isinstance(batch, RaggedSeriesBatch) and not batch.values.flags.writeable
    np.testing.assert_array_equal(batch.values, expected.values)
    np.testing.assert_array_equal(batch.offsets, expected.offsets)
    np.testing.assert_array_equal(batch.time_coordinates, expected.time_coordinates)
    np.testing.assert_array_equal(item["y"], cohort.y[rows])
    np.testing.assert_array_equal(item["target_mask"], cohort.target_mask[rows])
    np.testing.assert_array_equal(item["source_masks"]["series"], [True, False, False, True])
    assert item["sample_ids"] == tuple(ids) and list(item["X"]) == list(cohort.sources)
    assert item["X"]["metadata"].flags.writeable and item["X"]["metadata"].dtype == object
    batches = list(adapter.batches(2, sample_ids=iter(ids), return_metadata=True))
    assert [part["sample_ids"] for part in batches] == [("c", "empty"), ("hidden", "a")]
    np.testing.assert_array_equal(np.concatenate([part["X"]["series"].time_coordinates for part in batches]), expected.time_coordinates)


@pytest.mark.parametrize("ids", ["row-0", b"row-0", ["row-0", "row-0"], [""], [1], ["missing"], ["row-0", "missing"]])
def test_sklearn_selection_validates_identity_before_any_source_projection(ids, monkeypatch):
    adapter = SklearnProviderAdapter(_cohort(), source="nir")
    monkeypatch.setattr(MultimodalDataset, "take", lambda *args: pytest.fail("Selection rebuilt a cohort"))
    with pytest.raises(ValueError, match="sample IDs|sample_ids|ID"):
        adapter.arrays(ids)
    with pytest.raises(ValueError, match="sample IDs|sample_ids|ID"):
        list(adapter.batches(2, sample_ids=ids, start=0, drop_last=True))


def test_sklearn_batches_validate_completeness_only_for_emitted_rows():
    cohort = _cohort(masked=True)
    adapter = SklearnProviderAdapter(cohort)
    batches = list(adapter.batches(2, sample_ids=["row-10", "row-0", "row-1", "row-5"], start=1, drop_last=True))
    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0][0]["nir"], cohort.sources["nir"].values[:2])
    np.testing.assert_array_equal(batches[0][1], cohort.y[:2])
