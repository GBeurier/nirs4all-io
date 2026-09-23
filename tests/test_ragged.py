# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Raw variable-length source identity, replay schema and checkpoint contracts."""

import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from nirs4all_io import DataProvider, MultimodalDataset, RaggedSeriesBatch, RaggedSeriesSource, TensorSource


def make_source(*, dtype="float32", times_dtype="float64", hidden=np.nan):
    # Source order c,a,b; b is absent despite retaining two opaque stored points.
    return RaggedSeriesSource(
        np.asarray([[30, 31], [32, 33], [34, 35], [10, 11], [hidden, hidden], [hidden, hidden]], dtype=dtype),
        [0, 3, 4, 6], ["c", "a", "b"],
        time_coordinates=np.asarray([1, 2, 4, 8, 0, 1], dtype=times_dtype),
        channel_names=["temperature", "humidity"], time_unit="h", presence_mask=[True, True, False],
    )


def make_cohort(**kwargs):
    return MultimodalDataset(
        {
            "nir": TensorSource(np.arange(12).reshape(3, 4), ["a", "b", "c"], representation_id="signal_1d"),
            "series": make_source(**kwargs),
        }, sample_ids=["a", "b", "c"], y=[[1, 2], [np.nan, 4], [5, 6]],
        target_mask=[[True, True], [False, True], [True, True]], target_names=["one", "two"],
        task_type="regression", groups=["g1", "g1", "g2"], partitions=["train", "train", "test"],
    )


def test_source_keeps_raw_points_times_axes_and_presence_after_identity_alignment():
    cohort = make_cohort()
    source = cohort.sources["series"]
    assert isinstance(source, RaggedSeriesSource)
    assert isinstance(cohort.source_values()[1], RaggedSeriesBatch)
    assert source.sample_ids == ("a", "b", "c")
    assert source.lengths.tolist() == [1, 2, 3]
    assert source.offsets.tolist() == [0, 1, 3, 6]
    assert source.values.shape == (3, None, 2)
    assert source.values.ndim == 3 and source.values.dtype == np.dtype("float32")
    assert source.feature_names == source.channel_names == ("temperature", "humidity")
    np.testing.assert_array_equal(source.values[0], [[10, 11]])
    np.testing.assert_array_equal(source.values[2], [[30, 31], [32, 33], [34, 35]])
    assert np.isnan(source.values[1]).all()
    assert source.time_coordinates.tolist() == [8, 0, 1, 1, 2, 4]
    assert cohort.source_presence()["series"].tolist() == [True, False, True]
    assert source.axis_units == {"time": "h", "variable": None}
    assert source.axis_coordinates == {"variable": ("temperature", "humidity")}
    with pytest.raises(TypeError, match="explicit ragged-series encoder"):
        np.asarray(source.values)


def test_cohort_id_and_position_views_keep_labels_groups_and_raw_time_boundaries():
    cohort = make_cohort()
    view = cohort.take(["c", "a"])
    assert view.sample_ids == ("c", "a")
    assert view.sources["series"].lengths.tolist() == [3, 1]
    assert view.sources["series"].time_coordinates.tolist() == [1, 2, 4, 8]
    np.testing.assert_array_equal(view.y, [[5, 6], [1, 2]])
    assert view.groups.tolist() == ["g2", "g1"]
    assert view.partitions.tolist() == ["test", "train"]
    batch = cohort.source_values([2, 0])[1]
    np.testing.assert_array_equal(batch.values, view.sources["series"].values.values)
    assert cohort.source_presence([2, 1])["series"].tolist() == [True, False]


def test_selected_sources_keep_requested_order_and_own_readonly_row_buffers():
    cohort = make_cohort()
    assert cohort.source_values(source_names=[]) == []
    assert cohort.source_values([2, 0], source_names=[]) == []
    assert cohort.source_values(source_names=["series"])[0] is cohort.sources["series"].values
    series, nir = cohort.source_values([2, 0, 2], source_names=["series", "nir"])
    assert isinstance(series, RaggedSeriesBatch)
    assert series.lengths.tolist() == [3, 1, 3]
    assert series.time_coordinates.tolist() == [1, 2, 4, 8, 1, 2, 4]
    assert cohort.sources["series"].channel_names == ("temperature", "humidity")
    assert cohort.sources["series"].time_unit == "h"
    assert cohort.source_presence([2, 0, 2])["series"].tolist() == [True, True, True]
    np.testing.assert_array_equal(nir, cohort.sources["nir"].values[[2, 0, 2]])
    assert nir.dtype == cohort.sources["nir"].values.dtype
    assert not nir.flags.writeable
    assert not np.shares_memory(nir, cohort.sources["nir"].values)
    assert cohort.source_values([], source_names=["nir"])[0].shape == (0, 4)


def test_selecting_dense_source_does_not_materialize_unselected_ragged_rows(monkeypatch):
    cohort = make_cohort()

    def forbidden(*args, **kwargs):
        raise AssertionError("unselected series must not be materialized")

    monkeypatch.setattr(RaggedSeriesBatch, "take_rows", forbidden)
    nir, = cohort.source_values([2, 0], source_names=["nir"])
    np.testing.assert_array_equal(nir, [[8, 9, 10, 11], [0, 1, 2, 3]])


@pytest.mark.parametrize("names, message", [
    ("nir", "sequence"), (["nir", "nir"], "duplicate"), (["missing"], "Unknown source"),
    ([""], "non-empty"), ([2], "non-empty"),
])
def test_invalid_source_selections_are_refused(names, message):
    with pytest.raises(ValueError, match=message):
        make_cohort().source_values([0], source_names=names)


def test_batch_row_selections_preserve_order_duplicates_and_empty_axes():
    batch = make_source().values
    selected = batch.take_rows([1, 0, 1])
    assert selected.lengths.tolist() == [1, 3, 1]
    assert selected.time_coordinates.tolist() == [8, 1, 2, 4, 8]
    boolean = batch.take_rows(np.array([True, False, True]))
    assert boolean.lengths.tolist() == [3, 2]
    empty = batch.take_rows([])
    assert empty.shape == (0, None, 2)
    assert empty.values.shape == (0, 2) and empty.offsets.tolist() == [0]
    assert empty.time_coordinates.shape == (0,)
    assert make_cohort().take([]).sources["series"].values.shape == (0, None, 2)


def test_explicit_left_alignment_creates_empty_absent_series_without_padding_points():
    source = RaggedSeriesSource([[3, 4], [5, 6]], [0, 2], ["c"], time_coordinates=[1, 2])
    with pytest.raises(ValueError, match="missing"):
        MultimodalDataset({"series": source}, sample_ids=["a", "b", "c"])
    cohort = MultimodalDataset({"series": source}, sample_ids=["a", "b", "c"], source_alignment="left")
    aligned = cohort.sources["series"]
    assert aligned.offsets.tolist() == [0, 0, 0, 2]
    assert aligned.presence_mask.tolist() == [False, False, True]
    np.testing.assert_array_equal(aligned.values.values, source.values.values)
    assert aligned.time_coordinates.tolist() == [1, 2]
    with pytest.raises(ValueError, match="extra"):
        MultimodalDataset({"series": source}, sample_ids=["a"], source_alignment="left")


def test_absent_sources_and_empty_cohorts_require_fixed_channels_not_fake_points():
    empty = RaggedSeriesSource(np.empty((0, 2), dtype=np.float32), [0], [])
    cohort = MultimodalDataset({"series": empty}, sample_ids=["a", "b"], source_alignment="left")
    assert cohort.sources["series"].lengths.tolist() == [0, 0]
    assert not cohort.source_presence()["series"].any()
    assert cohort.sources["series"].values.values.shape == (0, 2)
    explicit = RaggedSeriesSource(np.empty((0, 2)), [0, 0], ["a"], presence_mask=[False])
    assert len(explicit.values) == 1
    with pytest.raises(ValueError, match="at least one point"):
        RaggedSeriesSource(np.empty((0, 2)), [0, 0], ["a"])
    with pytest.raises(ValueError, match="nonempty channels"):
        RaggedSeriesBatch(np.empty((0, 0)), [0])


def test_buffers_are_readonly_and_copied():
    values, offsets, times, presence = np.ones((2, 2)), np.array([0, 2]), np.array([0., 1.]), np.array([True])
    source = RaggedSeriesSource(values, offsets, ["a"], time_coordinates=times, presence_mask=presence)
    values[:] = 9
    offsets[:] = 0
    times[:] = -9
    presence[:] = False
    np.testing.assert_array_equal(source.values.values, np.ones((2, 2)))
    assert source.offsets.tolist() == [0, 2] and source.time_coordinates.tolist() == [0, 1]
    assert source.presence_mask.tolist() == [True]
    for array in (source.values.values, source.offsets, source.lengths, source.time_coordinates, source.presence_mask, source.values[0]):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1


@pytest.mark.parametrize("offsets", [[], [1, 2], [0, 1], [0, 3, 2], [0, -1, 2], [False, True], [0., 2.], [[0, 2]], [0, 2**64 - 1]])
def test_malformed_offsets_are_never_repaired(offsets):
    with pytest.raises(ValueError, match="offsets"):
        RaggedSeriesBatch(np.ones((2, 2)), offsets)


@pytest.mark.parametrize("times", [[0], [0, 0], [2, 1], [0, np.nan], [0, np.inf], ["0", "1"], [[0, 1]], [False, True]])
def test_bad_time_coordinates_fail_without_sorting_or_numeric_conversion(times):
    with pytest.raises(ValueError, match="time_coordinates"):
        RaggedSeriesBatch(np.ones((2, 2)), [0, 2], time_coordinates=times)


@pytest.mark.parametrize("field", ["values", "offsets", "time_coordinates", "presence_mask"])
def test_masked_arrays_cannot_lose_their_mask(field):
    kwargs = dict(values=np.ones((2, 2)), offsets=[0, 2], sample_ids=["a"], time_coordinates=[0, 1], presence_mask=[True])
    kwargs[field] = np.ma.array(kwargs[field], mask=False)
    with pytest.raises(ValueError, match="masked array"):
        RaggedSeriesSource(**kwargs)


@pytest.mark.parametrize("presence", [[1], [[True]], [], [True, False]])
def test_presence_requires_exact_boolean_sample_shape(presence):
    with pytest.raises(ValueError, match="presence_mask"):
        RaggedSeriesSource([[1, 2]], [0, 1], ["a"], presence_mask=presence)


def test_present_nonfinite_values_refused_but_absent_opaque_points_preserved():
    with pytest.raises(ValueError, match="finite"):
        RaggedSeriesSource([[np.nan, np.inf]], [0, 1], ["a"])
    source = RaggedSeriesSource([[np.nan, np.inf]], [0, 1], ["a"], presence_mask=[False])
    assert np.isnan(source.values.values[0, 0]) and np.isinf(source.values.values[0, 1])


@pytest.mark.parametrize("rows", [[-1], [3], [0.0], [[0]], [True], np.ma.array([0], mask=False)])
def test_bad_row_selections_fail(rows):
    with pytest.raises(ValueError):
        make_source().values.take_rows(rows)


def test_dense_tensor_source_stays_dense_and_does_not_accept_ragged_carriers():
    with pytest.raises((ValueError, TypeError)):
        TensorSource([np.ones((2, 2)), np.ones((3, 2))], ["a", "b"], representation_id="series_mv")
    with pytest.raises(TypeError, match="RaggedSeriesBatch"):
        TensorSource(make_source().values, ["a", "b", "c"], representation_id="series_mv")
    dense = TensorSource(np.ones((2, 3, 2)), ["a", "b"], representation_id="series_mv")
    assert not dense.descriptor("series")["native_representation"]["ragged"]
    assert dense.schema_descriptor("series")["shape"] == [None, 3, 2]


def test_replay_schema_allows_new_lengths_and_times_but_binds_dtype_channels_and_units():
    source = make_source()
    fresh = RaggedSeriesSource(np.ones((7, 2), dtype=np.float32), [0, 5, 7], ["new-1", "new-2"],
                               time_coordinates=np.array([10, 11, 12, 14, 18, 4, 7], dtype=np.float64),
                               channel_names=source.channel_names, time_unit=source.time_unit)
    expected = source.schema_descriptor("series")
    assert fresh.schema_descriptor("series") == expected
    assert expected["shape"] == [None, None, 2]
    representation = expected["native_representation"]
    assert representation["id"] == "series_mv" and representation["ragged"]
    assert representation["axes"][1]["variable"] is True and representation["axes"][1]["size"] is None
    assert "lengths" not in json.dumps(expected) and "offsets" not in json.dumps(expected)
    variants = [
        {"values": np.ones((7, 2), dtype=np.float64)}, {"time_unit": "s"},
        {"channel_names": ["humidity", "temperature"]}, {"time_coordinates": None},
        {"time_coordinates": fresh.time_coordinates.astype(np.float32)},
    ]
    for changes in variants:
        kwargs = dict(values=fresh.values.values, offsets=fresh.offsets, sample_ids=fresh.sample_ids,
                      time_coordinates=fresh.time_coordinates, channel_names=fresh.channel_names, time_unit=fresh.time_unit)
        changed = RaggedSeriesSource(**{**kwargs, **changes})
        assert changed.schema_descriptor("series") != expected


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_restores_readonly_dtype_and_offsets_without_endian_normalization(protocol):
    original = make_cohort(dtype=">f4", times_dtype=">f8")
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert restored.to_dict() == original.to_dict()
    source = restored.sources["series"]
    assert source.values.dtype == np.dtype(">f4") and source.time_coordinates.dtype == np.dtype(">f8")
    assert not source.values.values.flags.writeable and not source.offsets.flags.writeable
    batch = pickle.loads(pickle.dumps(source.values, protocol=protocol))
    assert batch.dtype == source.values.dtype
    np.testing.assert_array_equal(batch.values, source.values.values)
    assert not batch.values.flags.writeable


def test_strict_json_roundtrip_preserves_dense_records_and_hidden_nonfinite_ragged_values():
    original = make_cohort()
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    assert payload["schema_version"] == 1
    assert "source_kind" not in payload["sources"][0]
    assert payload["sources"][1]["source_kind"] == "ragged_series"
    restored = MultimodalDataset.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.schema_descriptors() == original.schema_descriptors()
    assert restored.sources["series"].sample_ids == original.sample_ids
    assert restored.source_presence()["series"].tolist() == [True, False, True]
    assert not restored.sources["series"].values.values.flags.writeable


@pytest.mark.parametrize("field,value", [("source_kind", "unknown"), ("representation_id", "signal_1d"),
                                        ("axes", ["sample", "variable", "time"]), ("extra", 1),
                                        ("channel_names", ["one"]), ("sample_ids", ["a", "a", "c"])])
def test_malformed_json_ragged_records_are_refused(field, value):
    payload = make_cohort().to_dict()
    payload["sources"][1][field] = value
    with pytest.raises(ValueError):
        MultimodalDataset.from_dict(payload)


def test_json_requires_integral_valid_offsets_and_exact_values_shape():
    payload = make_cohort().to_dict()
    broken = copy.deepcopy(payload)
    broken["sources"][1]["offsets"]["values"][1] = 0.5
    with pytest.raises(ValueError, match="dtype"):
        MultimodalDataset.from_dict(broken)
    broken = copy.deepcopy(payload)
    broken["sources"][1]["array"]["shape"][1] = 3
    with pytest.raises(ValueError, match="shape"):
        MultimodalDataset.from_dict(broken)


def test_provider_checkpoint_regenerates_typed_ragged_and_restores_sample_batches():
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["seed"])
        return make_cohort()

    provider = DataProvider(generate, provider_id="test.ragged", seed=19)
    provider.materialize()
    batches = provider.batches(2)
    first = next(batches)
    assert first.sources["series"].lengths.tolist() == [1, 2]
    state = json.loads(json.dumps(batches.state_dict(), allow_nan=False))
    expected = next(batches).to_dict()
    resumed = DataProvider(generate, provider_id="test.ragged", seed=19)
    resumed.load_state_dict(state["provider"])
    restored = resumed.batches(2)
    restored.load_state_dict(state)
    assert next(restored).to_dict() == expected
    assert calls == [19, 19] and list(restored) == []
    assert provider.fingerprint == resumed.fingerprint
    direct = hashlib.sha256(json.dumps(provider.cohort.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert provider.fingerprint == direct


@pytest.mark.parametrize("change", ["hidden", "offsets", "time", "presence"])
def test_provider_resume_checks_full_ragged_content_even_hidden_buffers(change):
    def fixture(**kwargs):
        payload = make_cohort(**kwargs).to_dict()
        # Distinct coordinates around the movable b/c boundary keep both
        # original and changed sample sequences valid without any time repair.
        payload["sources"][1]["time_coordinates"]["values"] = [8, 0, 1, 2, 3, 4]
        return payload

    original = DataProvider(lambda **_: MultimodalDataset.from_dict(fixture()), provider_id="test.ragged")
    original.materialize()

    def changed(**kwargs):
        payload = fixture(hidden=999 if change == "hidden" else np.nan)
        record = payload["sources"][1]
        if change == "offsets":
            record["offsets"]["values"][2] = 4
            # The newly present c rows still have increasing times; b stays absent.
        elif change == "time":
            record["time_coordinates"]["values"][0] += 1
        elif change == "presence":
            record["presence_mask"]["values"][0] = False
        return MultimodalDataset.from_dict(payload)

    with pytest.raises(ValueError, match="fingerprint"):
        DataProvider(changed, provider_id="test.ragged").load_state_dict(original.state_dict())


def test_provider_partial_assembly_can_add_ragged_to_an_explicit_fixed_base():
    cohort = make_cohort()
    base = MultimodalDataset({"nir": cohort.sources["nir"]}, sample_ids=cohort.sample_ids)
    provider = DataProvider(lambda **_: {"sample_ids": ["c", "a", "b"], "sources": {"series": make_source()}},
                            provider_id="test.ragged-partial", base=base)
    assert provider.materialize().sources["series"].lengths.tolist() == [1, 2, 3]


def test_ragged_and_cohort_modules_match_native_binding_mirrors():
    root = Path(__file__).resolve().parents[1]
    for module in ("ragged.py", "multimodal.py"):
        assert (root / "src/nirs4all_io" / module).read_bytes() == (root / "bindings/python/python/nirs4all_io" / module).read_bytes()


# Original smoke contracts retained alongside the expanded qualification.

@pytest.fixture
def source():
    return RaggedSeriesSource(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        [0, 2, 3, 6],
        ["sample-b", "sample-a", "sample-c"],
        time_coordinates=np.array([0.0, 0.5, 2.0, 1.0, 2.0, 4.0]),
        channel_names=["temperature", "humidity"],
        time_unit="h",
        presence_mask=[True, True, True],
    )


def test_batch_preserves_variable_lengths_dtype_time_and_read_only_buffers(source):
    batch = source.values
    assert batch.shape == (3, None, 2)
    assert batch.ndim == 3 and batch.dtype == np.float32
    np.testing.assert_array_equal(batch.lengths, [2, 1, 3])
    np.testing.assert_array_equal(batch[1], [[4.0, 5.0]])
    assert not batch.values.flags.writeable
    assert not batch.offsets.flags.writeable
    assert batch.time_coordinates is not None and not batch.time_coordinates.flags.writeable
    with pytest.raises(TypeError, match="cannot be coerced"):
        np.asarray(batch)


def test_row_projection_keeps_order_duplicates_and_empty_selection(source):
    projected = source.values.take_rows([2, 0, 2])
    np.testing.assert_array_equal(projected.lengths, [3, 2, 3])
    np.testing.assert_array_equal(projected[1], source.values[0])
    empty = source.values.take_rows([])
    assert empty.shape == (0, None, 2)
    assert empty.values.shape == (0, 2)
    assert empty.time_coordinates is not None and empty.time_coordinates.shape == (0,)
    boolean = source.values.take_rows([True, False, True])
    np.testing.assert_array_equal(boolean.lengths, [2, 3])


@pytest.mark.parametrize(
    ("values", "offsets", "time_coordinates"),
    [
        (np.ones((2, 1)), [1, 2], None),
        (np.ones((2, 1)), [0, 3], None),
        (np.ones((2, 1)), [0, 2, 1, 2], None),
        (np.ones((2, 1)), [0.0, 2.0], None),
        (np.ones((2, 1)), [0, 2], [0.0]),
        (np.ones((2, 1)), [0, 2], [1.0, 1.0]),
        (np.ones((2, 1)), [0, 2], [0.0, np.nan]),
        (np.array([["bad"]], dtype=object), [0, 1], None),
    ],
)
def test_invalid_packed_arrays_are_rejected(values, offsets, time_coordinates):
    with pytest.raises(ValueError):
        RaggedSeriesBatch(values, offsets, time_coordinates=time_coordinates)


def test_source_rejects_present_empty_or_nonfinite_series_and_bad_metadata():
    with pytest.raises(ValueError, match="at least one point"):
        RaggedSeriesSource(np.ones((1, 2)), [0, 0, 1], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        RaggedSeriesSource([[np.nan]], [0, 1], ["a"])
    with pytest.raises(ValueError, match="channel_names"):
        RaggedSeriesSource(np.ones((1, 2)), [0, 1], ["a"], channel_names=["only-one"])
    with pytest.raises(ValueError, match="time_unit"):
        RaggedSeriesSource(np.ones((1, 1)), [0, 1], ["a"], time_unit="")


def test_absent_series_can_be_empty_and_left_alignment_inserts_absent_rows():
    sparse = RaggedSeriesSource(
        np.array([[3.0], [4.0]]), [0, 1, 2], ["sample-c", "sample-a"],
        presence_mask=[True, True],
    )
    cohort = MultimodalDataset(
        {"sensor": sparse}, sample_ids=["sample-a", "sample-b", "sample-c"],
        source_alignment="left",
    )
    aligned = cohort.sources["sensor"]
    assert aligned.sample_ids == cohort.sample_ids
    assert aligned.presence_mask.tolist() == [True, False, True]
    np.testing.assert_array_equal(aligned.lengths, [1, 0, 1])
    np.testing.assert_array_equal(aligned.values[0], [[4.0]])
    np.testing.assert_array_equal(aligned.values[2], [[3.0]])


def test_cohort_selection_descriptors_json_and_pickle_preserve_contract(source):
    cohort = MultimodalDataset({"sensor": source}, sample_ids=["sample-a", "sample-b", "sample-c"])
    selected = cohort.take(["sample-c", "sample-a"])
    assert selected.sources["sensor"].lengths.tolist() == [3, 1]
    assert selected.source_values()[0].shape == (2, None, 2)
    descriptor = selected.descriptors()[0]
    assert descriptor["native_representation"]["ragged"] is True
    assert descriptor["native_representation"]["axes"][1]["size"] is None
    assert descriptor["axis_coordinates"]["variable"] == ["temperature", "humidity"]
    payload = selected.to_dict()
    json.dumps(payload, allow_nan=False)
    restored = MultimodalDataset.from_dict(payload)
    roundtripped = pickle.loads(pickle.dumps(restored))
    assert roundtripped.to_dict() == payload


@pytest.mark.parametrize("selection", [[-1], [3], [True], [0.0], [[0]]])
def test_invalid_batch_selection_is_rejected(source, selection):
    with pytest.raises((IndexError, TypeError, ValueError)):
        source.values.take_rows(selection)


def test_ragged_source_and_binding_module_are_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/nirs4all_io/ragged.py").read_bytes() == (
        root / "bindings/python/python/nirs4all_io/ragged.py"
    ).read_bytes()
