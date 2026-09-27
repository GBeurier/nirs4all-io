# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Finite provider generation, identity assembly and checked cursor resume."""

import json
from pathlib import Path

import numpy as np
import pytest

from nirs4all_io import DataProvider, MultimodalDataset, TensorSource


def synthetic(*, seed, params, context):
    rng = np.random.default_rng(seed)
    count = params.get("count", 7)
    ids = tuple(f"sample-{index}" for index in range(count))
    y = rng.normal(size=(count, 2))
    target_mask = np.ones(y.shape, dtype=bool)
    if count:
        target_mask[-1, 1] = False
        y[-1, 1] = np.nan
    presence = np.ones(count, dtype=bool)
    presence[::3] = False
    return MultimodalDataset(
        {
            "nir": TensorSource(rng.normal(size=(count, 3)), ids, representation_id="signal_1d",
                                axis_units={"wavelength": "nm"}, axis_coordinates={"wavelength": [900, 1000, 1100]}),
            "image": TensorSource(rng.integers(0, 255, size=(count, 2, 2, 3), dtype=np.uint8), ids,
                                  representation_id="rgb_image", presence_mask=presence),
            "temporal": TensorSource(rng.normal(size=(count, 4, 2)), ids, representation_id="series_mv",
                                     axis_coordinates={"time": [0, 1, 2, 3]}),
            "metadata": TensorSource(np.array([[index, "A" if index % 2 else "B"] for index in range(count)], dtype=object).reshape(count, 2),
                                     ids, representation_id="tabular_mixed", feature_names=["index", "class"]),
        }, sample_ids=ids, y=y, target_mask=target_mask, target_names=["one", "two"], task_type="regression",
        groups=[f"group-{index}" for index in range(count)],
        partitions=["train" if index < count - 2 else "test" for index in range(count)],
        name=context.get("name", "synthetic"),
    )


def provider(generate=synthetic, **kwargs):
    return DataProvider(generate, provider_id="tests.synthetic", provider_version="1", seed=23, **kwargs)


def test_generation_is_explicit_and_views_preserve_four_raw_modalities():
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return synthetic(**kwargs)

    data = provider(generate)
    for access in (lambda: data.cohort, lambda: len(data), lambda: data[0], lambda: data.get([]), lambda: data.batches(2), lambda: data.fingerprint):
        with pytest.raises(RuntimeError, match="materialized"):
            access()
    cohort = data.materialize()
    assert len(data) == 7
    assert data[0].sample_ids == ("sample-0",)
    assert data[np.int64(1)].sample_ids == ("sample-1",)
    view = data.get(["sample-4", "sample-1"])
    assert view.sources["image"].values.shape == (2, 2, 2, 3)
    assert view.sources["temporal"].values.shape == (2, 4, 2)
    np.testing.assert_array_equal(view.y, cohort.y[[4, 1]])
    assert view.groups.tolist() == ["group-4", "group-1"]
    assert view.sources["nir"].axis_units["wavelength"] == "nm"
    assert len(list(data.batches(3))) == 3
    assert len(calls) == 1
    assert calls[0] == {"seed": 23, "params": {}, "context": {}}


def test_seed_and_context_are_isolated_from_global_rng_and_recipe_mutation():
    params, context = {"count": 7}, {"name": "recipe-name"}
    data = provider(params=params, context=context)
    params["count"], context["name"] = 100, "mutated"
    np.random.seed(77)
    before = np.random.get_state()
    first = data.materialize(seed=2**64 - 1, context={"name": "task-name"}).to_dict()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert first == data.materialize(seed=2**64 - 1, context={"name": "task-name"}).to_dict()
    assert data.recipe()["seed"] == 23
    assert data.recipe()["context"] == {"name": "recipe-name"}
    assert data.state_dict()["context"] == {"name": "task-name"}
    assert data.recipe()["params"]["count"] == 7
    copied = data.recipe()
    copied["params"]["count"] = 99
    assert data.recipe()["params"]["count"] == 7
    assert data.materialize().to_dict() != first


def test_callback_cannot_mutate_recipe_parameters_or_effective_context():
    def mutate(**kwargs):
        result = synthetic(**kwargs)
        kwargs["params"]["nested"][0] = "changed"
        kwargs["context"]["nested"][0] = "changed"
        return result

    data = provider(mutate, params={"nested": ["original"]}, context={"nested": ["original"]})
    data.materialize()
    assert data.recipe()["params"]["nested"] == ["original"]
    assert data.state_dict()["context"]["nested"] == ["original"]


def test_partial_y_and_source_assembly_aligns_by_identity_preserving_fixed_boundaries():
    ids = ("b", "a", "c")
    base = MultimodalDataset(
        {"nir": TensorSource(np.arange(9).reshape(3, 3), ids, representation_id="signal_1d")},
        sample_ids=ids, groups=["gb", "ga", "gc"], partitions=["train", "train", "test"],
    )

    def partial(**kwargs):
        return {
            "sample_ids": ["c", "b", "a"], "y": [30.0, np.nan, 10.0], "target_mask": [True, False, True],
            "task_type": "regression", "source_alignment": "left",
            "sources": {"image": TensorSource(np.full((1, 2, 2, 3), 17, dtype=np.uint8), ["a"], representation_id="rgb_image")},
        }

    data = provider(partial, base=base)
    cohort = data.materialize()
    assert cohort.sample_ids == ids
    np.testing.assert_array_equal(cohort.y, [np.nan, 10.0, 30.0])
    np.testing.assert_array_equal(cohort.target_mask, [False, True, True])
    assert cohort.source_presence()["image"].tolist() == [False, True, False]
    assert cohort.partitions.tolist() == ["train", "train", "test"]
    np.testing.assert_array_equal(cohort.sources["nir"].values, base.sources["nir"].values)
    assert base.y is None and list(base.sources) == ["nir"]
    assert data.recipe()["params"]["_io_assembly"]["base_fingerprint"] is not None


def test_target_only_generation_and_explicit_replacement_rules():
    base = synthetic(seed=0, params={}, context={})

    def partial(**kwargs):
        return {"sample_ids": base.sample_ids, "y": np.arange(len(base)), "task_type": "regression"}

    with pytest.raises(ValueError, match="replace_targets"):
        provider(partial, base=base).materialize()
    cohort = provider(partial, base=base, replace_targets=True).materialize()
    assert cohort.target_names == ("y",)
    assert cohort.target_mask.all()
    np.testing.assert_array_equal(cohort.y, np.arange(len(base)))
    source_output = {"sample_ids": base.sample_ids, "sources": {"nir": base.sources["nir"]}}
    with pytest.raises(ValueError, match="replace_sources"):
        provider(lambda **_: source_output, base=base).materialize()
    replaced = provider(lambda **_: source_output, base=base, replace_sources=["nir"])
    assert replaced.materialize().to_dict() == base.to_dict()
    assert replaced.recipe()["params"]["_io_assembly"]["replace_sources"] == ["nir"]


@pytest.mark.parametrize("change", [
    {"sample_ids": ["sample-0"]},
    {"sample_ids": ["foreign"]},
    {"partitions": ["train"] * 7},
    {"groups": ["changed"] * 7},
    {"ignored": True},
])
def test_partial_generation_cannot_change_identity_or_fixed_partition_contract(change):
    base = synthetic(seed=0, params={}, context={})
    output = {"sample_ids": base.sample_ids, **change}
    with pytest.raises(ValueError):
        provider(lambda **_: output, base=base).materialize()


def test_base_is_snapshotted_and_recipe_changes_if_fixed_data_changes():
    base = synthetic(seed=0, params={}, context={})
    data = provider(lambda **_: {"sample_ids": base.sample_ids}, base=base)
    original = data.recipe()
    base.name = "changed-after-construction"
    assert data.recipe() == original
    assert data.materialize().name == "synthetic"
    changed = provider(lambda **_: {"sample_ids": base.sample_ids}, base=base)
    assert changed.recipe() != original


def test_complete_mapping_output_and_prediction_only_targets():
    ids = ["one", "two"]
    output = {"sources": {"x": TensorSource(np.ones((2, 3)), ids, representation_id="signal_1d")},
              "sample_ids": ids, "target_names": ["future"], "task_type": "regression", "partitions": ["predict"] * 2}
    cohort = provider(lambda **_: output).materialize()
    assert cohort.y is None and cohort.target_mask is None
    assert cohort.target_names == ("future",)


def test_batches_are_bounded_independent_views_preserving_masks_and_order():
    data = provider()
    cohort = data.materialize()
    ids = ["sample-6", "sample-0", "sample-3", "sample-1", "sample-4"]
    first, second = data.batches(2, sample_ids=ids), data.batches(2, sample_ids=ids)
    assert next(first).sample_ids == next(second).sample_ids == tuple(ids[:2])
    batches = list(first)
    assert [item.sample_ids for item in batches] == [tuple(ids[2:4]), tuple(ids[4:])]
    assert next(second).sample_ids == tuple(ids[2:4])
    assert [len(item) for item in data.batches(2, sample_ids=ids, drop_last=True)] == [2, 2]
    assert [len(item) for item in data.batches(2, sample_ids=ids, start=3)] == [2]
    assert list(data.batches(2, sample_ids=[])) == []
    all_batches = list(data.batches(3))
    np.testing.assert_array_equal(np.concatenate([item.y for item in all_batches]), cohort.y)
    np.testing.assert_array_equal(np.concatenate([item.target_mask for item in all_batches]), cohort.target_mask)
    np.testing.assert_array_equal(np.concatenate([item.source_presence()["image"] for item in all_batches]), cohort.source_presence()["image"])


@pytest.mark.parametrize("ids", [None, [], ["sample-6", "sample-0", "sample-3"]])
def test_batch_cursor_construction_does_not_copy_the_selected_cohort(ids, monkeypatch):
    data = provider()
    cohort = data.materialize()
    original_take = MultimodalDataset.take
    materialized = []

    def tracked_take(self, sample_ids):
        materialized.append(tuple(sample_ids))
        return original_take(self, sample_ids)

    monkeypatch.setattr(MultimodalDataset, "take", tracked_take)
    batches = data.batches(2, sample_ids=ids)
    expected = cohort.sample_ids if ids is None else tuple(ids)
    assert batches.state_dict()["sample_ids"] == list(expected)
    assert materialized == []
    if expected:
        assert next(batches).sample_ids == expected[:2]
        assert materialized == [expected[:2]]
    else:
        assert list(batches) == [] and materialized == []


def test_json_checkpoint_regenerates_once_and_resumes_exact_unconsumed_batches():
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["seed"])
        return synthetic(**kwargs)

    data = provider(generate)
    data.materialize(seed=91, context={"name": "effective"})
    batches = data.batches(2, sample_ids=["sample-4", "sample-2", "sample-6", "sample-1", "sample-0"])
    next(batches)
    checkpoint = json.loads(json.dumps(batches.state_dict(), allow_nan=False))
    expected = [batch.to_dict() for batch in batches]
    resumed = provider(generate)
    resumed.load_state_dict(checkpoint["provider"])
    restored_batches = resumed.batches(2, sample_ids=checkpoint["sample_ids"])
    restored_batches.load_state_dict(checkpoint)
    assert calls == [91, 91]
    assert [batch.to_dict() for batch in restored_batches] == expected
    assert calls == [91, 91]
    assert resumed.fingerprint == data.fingerprint
    exhausted = restored_batches.state_dict()
    restored_batches.load_state_dict(exhausted)
    assert list(restored_batches) == []


@pytest.mark.parametrize("change", [
    {"version": 2}, {"extra": 0}, {"seed": True}, {"context": []}, {"fingerprint": "invalid"},
])
def test_bad_provider_checkpoint_is_rejected_before_generation(change):
    data = provider()
    data.materialize()
    state = {**data.state_dict(), **change}

    def forbidden(**kwargs):
        pytest.fail("malformed checkpoint invoked generation")

    with pytest.raises(ValueError):
        provider(forbidden).load_state_dict(state)


@pytest.mark.parametrize("params", [{"count": 8}, {"count": 7.0}])
def test_checkpoint_recipe_changes_are_rejected_before_generation(params):
    data = provider(params={"count": 7})
    data.materialize()

    def forbidden(**kwargs):
        pytest.fail("changed recipe invoked generation")

    with pytest.raises(ValueError, match="recipe"):
        provider(forbidden, params=params).load_state_dict(data.state_dict())


def test_nondeterministic_generation_refuses_restore_without_replacing_existing_cohort():
    original = provider()
    original.materialize()

    def changed(**kwargs):
        cohort = synthetic(**kwargs)
        cohort.name = "different-content"
        return cohort

    target = provider(changed)
    before = target.materialize(seed=9)
    with pytest.raises(ValueError, match="Regenerated provider content"):
        target.load_state_dict(original.state_dict())
    assert target.cohort is before


@pytest.mark.parametrize("field, value", [
    ("position", -1), ("position", 8), ("position", True),
    ("batch_size", 3), ("drop_last", 0), ("sample_ids", ["sample-0"]), ("version", 2),
])
def test_bad_batch_checkpoint_preserves_cursor(field, value):
    data = provider()
    data.materialize()
    batches = data.batches(2)
    next(batches)
    before = batches.state_dict()
    invalid = {**before, field: value}
    with pytest.raises(ValueError):
        batches.load_state_dict(invalid)
    assert batches.state_dict() == before


def test_changed_masks_fail_checkpoint_even_with_unchanged_visible_payload():
    data = provider()
    data.materialize()
    checkpoint = data.state_dict()

    def changed(**kwargs):
        cohort = synthetic(**kwargs)
        payload = cohort.to_dict()
        payload["sources"][1]["presence_mask"]["values"][0] = True
        return MultimodalDataset.from_dict(payload)

    with pytest.raises(ValueError, match="fingerprint"):
        provider(changed).load_state_dict(checkpoint)


def test_materialized_mutations_are_not_accepted_as_original_checkpoints():
    data = provider()
    data.materialize()
    batches = data.batches(2)
    data.cohort.name = "mutated"
    with pytest.raises(ValueError, match="content changed"):
        data.state_dict()
    with pytest.raises(ValueError, match="content changed"):
        batches.state_dict()


@pytest.mark.parametrize("kwargs", [
    {"params": {"_io_assembly": {}}}, {"params": {"bad": np.nan}}, {"params": {1: "key"}},
    {"context": {"bad": float("inf")}}, {"seed": -1}, {"seed": True}, {"seed": 2**64},
    {"replace_sources": ["nir"]}, {"replace_targets": True},
])
def test_invalid_provider_configuration_fails_before_generation(kwargs):
    with pytest.raises(ValueError):
        DataProvider(lambda **_: pytest.fail("must not generate"), provider_id="invalid", **kwargs)


@pytest.mark.parametrize("index, error", [(-1, IndexError), (7, IndexError), (True, TypeError), (0.0, TypeError)])
def test_index_validation(index, error):
    data = provider()
    data.materialize()
    with pytest.raises(error):
        data[index]


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0},
    {"batch_size": True},
    {"batch_size": 2, "start": 8},
    {"batch_size": 2, "drop_last": 1},
    {"batch_size": 2, "sample_ids": ["sample-0", "sample-0"]},
    {"batch_size": 2, "sample_ids": ["unknown"]},
    {"batch_size": 2, "sample_ids": "sample-0"},
])
def test_batch_validation(kwargs):
    data = provider()
    data.materialize()
    with pytest.raises(ValueError):
        data.batches(**kwargs)


def test_empty_provider_cohort_is_finite_and_checkpointable():
    data = provider(params={"count": 0})
    data.materialize()
    assert len(data) == 0 and list(data.batches(2)) == []
    restored = provider(params={"count": 0})
    restored.load_state_dict(data.state_dict())
    assert restored.fingerprint == data.fingerprint


def test_source_and_native_binding_provider_modules_are_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/nirs4all_io/provider.py").read_bytes() == (root / "bindings/python/python/nirs4all_io/provider.py").read_bytes()
