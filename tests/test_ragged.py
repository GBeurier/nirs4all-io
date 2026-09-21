# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Variable-length series contracts, validation and cohort integration."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from nirs4all_io import MultimodalDataset, RaggedSeriesBatch, RaggedSeriesSource


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
