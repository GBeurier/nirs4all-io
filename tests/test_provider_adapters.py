# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Real sklearn and optional Torch consumers of materialized provider views."""

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nirs4all_io import MultimodalDataset, TensorSource
from nirs4all_io.provider import DataProvider
from nirs4all_io.provider_adapters import SklearnProviderAdapter, collate_provider_samples


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
