# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Identity and raw tensor contracts for in-memory multimodal assembly."""

import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nirs4all_io import MultimodalDataset, TensorSource


@pytest.fixture
def dataset():
    ids = ("plant-a", "plant-b", "plant-c", "plant-d")
    permutation = [2, 0, 3, 1]
    images = np.arange(4 * 3 * 2 * 3, dtype=np.uint8).reshape(4, 3, 2, 3)
    return MultimodalDataset(
        {
            "nir": TensorSource(
                np.arange(20).reshape(4, 5), ids, representation_id="signal_1d",
                axis_units={"wavelength": "nm"}, axis_coordinates={"wavelength": [1000, 1010, 1020, 1030, 1040]},
            ),
            "image": TensorSource(
                images[permutation], [ids[i] for i in permutation], representation_id="rgb_image",
                axis_units={"height": "px", "width": "px"}, axis_coordinates={"channel": ["red", "green", "blue"]},
            ),
            "temporal": TensorSource(
                np.arange(48, dtype=np.float32).reshape(4, 6, 2), ids, representation_id="series_mv",
                axis_units={"time": "h"}, axis_coordinates={"time": [0, 1, 2, 3, 4, 5], "variable": ["temperature", "humidity"]},
            ),
            "metadata": TensorSource(
                pd.DataFrame({"temperature": [21.0, 22.0, 23.0, 24.0], "cultivar": ["A", "B", "A", "C"]}),
                ids, representation_id="tabular_mixed", feature_names=["temperature", "cultivar"],
                axis_coordinates={"column": ["temperature", "cultivar"]},
            ),
        },
        sample_ids=ids,
        y=np.array([1.0, 2.0, 3.0, np.nan]),
        target_mask=[True, True, True, False],
        partitions=["train", "train", "test", "predict"],
        groups=["plot-a", "plot-a", "plot-b", "plot-c"],
    )


def test_aligns_permuted_sources_without_losing_raw_axes(dataset):
    assert list(dataset.sources) == ["nir", "image", "temporal", "metadata"]
    assert [array.shape for array in dataset.source_values()] == [(4, 5), (4, 3, 2, 3), (4, 6, 2), (4, 2)]
    np.testing.assert_array_equal(dataset.sources["image"].values, np.arange(72, dtype=np.uint8).reshape(4, 3, 2, 3))
    assert all(source.sample_ids == dataset.sample_ids for source in dataset.sources.values())
    assert dataset.sources["temporal"].values.dtype == np.float32
    assert dataset.sources["metadata"].values[0].tolist() == [21.0, "A"]
    assert np.isnan(dataset.y[-1])


def test_id_views_keep_targets_groups_and_partitions_aligned(dataset):
    view = dataset.take(["plant-c", "plant-a"])
    assert view.sample_ids == ("plant-c", "plant-a")
    np.testing.assert_array_equal(view.y, [3.0, 1.0])
    assert view.groups.tolist() == ["plot-b", "plot-a"]
    assert view.partitions.tolist() == ["test", "train"]
    for name, source in view.sources.items():
        np.testing.assert_array_equal(source.values, dataset.sources[name].values[[2, 0]])
    assert view.sources["metadata"].feature_names == ("temperature", "cultivar")
    assert len(dataset) == 4


def test_empty_cohort_view_keeps_non_sample_axes(dataset):
    view = dataset.take([])
    assert view.sample_ids == ()
    assert view.sources["image"].values.shape == (0, 3, 2, 3)
    assert view.sources["temporal"].values.shape == (0, 6, 2)
    assert dataset.source_values([])[1].shape == (0, 3, 2, 3)


def test_source_arrays_isolate_the_input_buffer():
    values = np.ones((2, 3))
    source = TensorSource(values, ["a", "b"], representation_id="signal_1d")
    values[0, 0] = -999
    assert source.values[0, 0] == 1
    with pytest.raises(ValueError, match="read-only"):
        source.values[0, 0] = 0


def test_mixed_python_rows_preserve_numeric_and_categorical_scalars():
    source = TensorSource([[20.0, "A"], [21.0, "B"]], ["a", "b"], representation_id="tabular_mixed")
    assert isinstance(source.values[0, 0], float)
    assert source.values[0, 1] == "A"


@pytest.mark.parametrize("sample_ids", [["a", "a"], ["a", ""], ["a", 1], "ab"])
def test_invalid_source_identity_is_rejected(sample_ids):
    with pytest.raises(ValueError, match="IDs"):
        TensorSource(np.ones((2, 3)), sample_ids, representation_id="signal_1d")


def test_missing_and_extra_identity_are_never_positionally_aligned():
    source = TensorSource(np.ones((2, 3)), ["b", "c"], representation_id="signal_1d")
    with pytest.raises(ValueError, match="missing=.*a.*extra=.*c"):
        MultimodalDataset({"nir": source}, sample_ids=["a", "b"])


@pytest.mark.parametrize("selection", [["plant-a", "missing"], ["plant-a", "plant-a"]])
def test_invalid_id_views_fail_before_projection(dataset, selection):
    with pytest.raises(ValueError, match="sample IDs"):
        dataset.take(selection)


@pytest.mark.parametrize("indices", [[-1], [4], [0.0], [True], [[0]], "0"])
def test_invalid_position_views_are_rejected(dataset, indices):
    with pytest.raises(ValueError, match="row_indices"):
        dataset.source_values(indices)


def test_position_view_keeps_each_tensor_rank(dataset):
    arrays = dataset.source_values([3, 1])
    assert [array.shape for array in arrays] == [(2, 5), (2, 3, 2, 3), (2, 6, 2), (2, 2)]
    assert arrays[-1].tolist() == [[24.0, "C"], [22.0, "B"]]


@pytest.mark.parametrize(
    ("representation", "shape"),
    [("signal_1d", (2, 3, 1)), ("rgb_image", (2, 3, 3)), ("series_mv", (2, 3)), ("rgb_image", (2, 3, 3, 4)), ("signal_1d", (2, 0))],
)
def test_wrong_rank_channel_or_empty_axis_rejected(representation, shape):
    with pytest.raises(ValueError):
        TensorSource(np.zeros(shape), ["a", "b"], representation_id=representation)


def test_source_axes_are_semantic_not_just_shapes():
    with pytest.raises(ValueError, match="requires axes"):
        TensorSource(np.zeros((2, 3, 2)), ["a", "b"], representation_id="series_mv", axes=["sample", "channel", "time"])


def test_object_image_is_rejected():
    with pytest.raises(ValueError, match="numeric tensor"):
        TensorSource(np.full((2, 3, 3, 3), "pixel", dtype=object), ["a", "b"], representation_id="rgb_image")


def test_nested_objects_are_not_mistaken_for_mixed_table_cells():
    values = np.empty((2, 1), dtype=object)
    values[:, 0] = [{"nested": 1}, [1, 2]]
    with pytest.raises(ValueError, match="scalars"):
        TensorSource(values, ["a", "b"], representation_id="tabular_mixed")


@pytest.mark.parametrize("partitions", [["train", "test"], ["train", "predict"], ["test", "predict"]])
def test_group_leakage_across_partitions_is_rejected(partitions):
    source = TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d")
    with pytest.raises(ValueError, match="crosses partitions"):
        MultimodalDataset({"nir": source}, sample_ids=["a", "b"], groups=["plant", "plant"], partitions=partitions)


@pytest.mark.parametrize("kwargs", [{"y": [1]}, {"partitions": ["train"]}, {"partitions": ["train", "val"]}, {"groups": ["a"]}, {"groups": ["a", None]}])
def test_cohort_side_inputs_must_be_explicitly_aligned(kwargs):
    source = TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d")
    with pytest.raises(ValueError):
        MultimodalDataset({"nir": source}, sample_ids=["a", "b"], **kwargs)


def test_source_descriptors_have_typed_axes_and_no_feature_payload(dataset):
    descriptors = dataset.descriptors()
    json.dumps(descriptors, allow_nan=False)
    assert [entry["type_id"] for entry in descriptors] == ["dense_signal", "image_rgb", "time_series", "table"]
    temporal = descriptors[2]["native_representation"]
    assert temporal["rank"] == 3
    assert [(axis["name"], axis["kind"], axis["size"]) for axis in temporal["axes"]] == [
        ("sample", "sample", 4), ("time", "time", 6), ("variable", "feature", 2),
    ]
    assert descriptors[-1]["native_representation"]["dtype"] is None
    assert all("values" not in descriptor for descriptor in descriptors)


def test_undeclared_axis_units_and_coordinates_are_never_invented():
    source = TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d")
    descriptor = source.descriptor("nir")
    assert source.axis_units == {"wavelength": None}
    assert descriptor["axis_units"] == {"wavelength": None}
    assert descriptor["axis_coordinates"] == {}
    assert all(axis["unit"] is None for axis in descriptor["native_representation"]["axes"])
    image = TensorSource(np.ones((2, 3, 4, 3)), ["a", "b"], representation_id="rgb_image")
    assert all(axis["unit"] is None for axis in image.descriptor("rgb")["native_representation"]["axes"])


def test_declared_axis_metadata_survives_selection_and_pickle(dataset):
    view = pickle.loads(pickle.dumps(dataset.take(["plant-c", "plant-a"])))
    for source_id, source in view.sources.items():
        original = dataset.sources[source_id]
        assert source.axis_units == original.axis_units
        assert source.axis_coordinates == original.axis_coordinates
        assert source.schema_descriptor(source_id) == original.schema_descriptor(source_id)
        json.dumps(source.schema_descriptor(source_id), allow_nan=False)
    assert view.sources["temporal"].axis_units == {"time": "h", "variable": None}
    assert view.sources["image"].axis_coordinates["channel"] == ("red", "green", "blue")
    assert view.sources["nir"].descriptor("nir")["native_representation"]["axes"][1]["unit"] == "nm"


def test_axis_metadata_has_independent_immutable_ownership():
    units = {"wavelength": "nm"}
    coordinates = {"wavelength": np.array([1000.0, 1010.0, 1020.0])}
    source = TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d", axis_units=units, axis_coordinates=coordinates)
    units["wavelength"] = "um"
    coordinates["wavelength"][0] = 1.0
    assert source.axis_units["wavelength"] == "nm"
    assert source.axis_coordinates["wavelength"] == (1000.0, 1010.0, 1020.0)
    with pytest.raises(TypeError):
        source.axis_units["wavelength"] = "um"
    with pytest.raises(TypeError):
        source.axis_coordinates["wavelength"] = (1.0, 2.0, 3.0)
    descriptor = source.descriptor("nir")
    descriptor["axis_coordinates"]["wavelength"][0] = 99
    assert source.axis_coordinates["wavelength"][0] == 1000.0


@pytest.mark.parametrize(
    "metadata",
    [
        {"axis_units": {"time": "s"}},
        {"axis_units": {"sample": "plant"}},
        {"axis_units": {"wavelength": ""}},
        {"axis_units": {"wavelength": 1}},
        {"axis_coordinates": {"time": [0, 1, 2]}},
        {"axis_coordinates": {"sample": ["a", "b"]}},
        {"axis_coordinates": {"wavelength": [1000, 1010]}},
        {"axis_coordinates": {"wavelength": [[1000], [1010], [1020]]}},
        {"axis_coordinates": {"wavelength": np.array([[1000, 1010, 1020]])}},
        {"axis_coordinates": {"wavelength": "abc"}},
        {"axis_coordinates": {"wavelength": [1000, np.nan, 1020]}},
        {"axis_coordinates": {"wavelength": [1000, np.inf, 1020]}},
        {"axis_coordinates": {"wavelength": [1000, 1000, 1020]}},
        {"axis_coordinates": {"wavelength": [1000, 1020, 1010]}},
        {"axis_coordinates": {"wavelength": ["1000", "1010", "1020"]}},
        {"axis_coordinates": {"wavelength": [1000, "1010", 1020]}},
        {"axis_coordinates": {"wavelength": [True, False, True]}},
        {"axis_coordinates": {"wavelength": [1000, None, 1020]}},
    ],
)
def test_invalid_axis_metadata_is_rejected_without_coercion(metadata):
    with pytest.raises(ValueError):
        TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d", **metadata)


@pytest.mark.parametrize("time", [[0, 0, 1], [2, 1, 0], [0, 2, 1], [0, np.nan, 2], ["0", "1", "2"]])
def test_invalid_time_grid_is_rejected_without_sorting(time):
    with pytest.raises(ValueError, match="time"):
        TensorSource(np.ones((2, 3, 1)), ["a", "b"], representation_id="series_mv", axis_coordinates={"time": time})


def test_descending_wavelength_grid_is_kept_exactly():
    values = np.array([[1, 2, 3], [4, 5, 6]])
    source = TensorSource(values, ["a", "b"], representation_id="signal_1d", axis_coordinates={"wavelength": [1020, 1010, 1000]})
    assert source.axis_coordinates["wavelength"] == (1020, 1010, 1000)
    np.testing.assert_array_equal(source.values, values)


def test_source_schema_excludes_only_sample_population(dataset):
    training_schema = dataset.schema_descriptors()
    assert dataset.take(["plant-b"]).schema_descriptors() == training_schema
    assert dataset.take([]).schema_descriptors() == training_schema
    assert all(schema["shape"][0] is None for schema in training_schema)
    assert all(schema["native_representation"]["axes"][0]["size"] is None for schema in training_schema)
    assert dataset.descriptors()[0]["shape"][0] == 4


def test_axis_schema_serialization_is_independent_of_mapping_insertion_order():
    kwargs = {"values": np.ones((2, 3, 2)), "sample_ids": ["a", "b"], "representation_id": "series_mv"}
    left = TensorSource(**kwargs, axis_units={"time": "h", "variable": None}, axis_coordinates={"time": [0, 1, 2], "variable": ["a", "b"]})
    right = TensorSource(**kwargs, axis_units={"variable": None, "time": "h"}, axis_coordinates={"variable": ["a", "b"], "time": [0, 1, 2]})
    assert json.dumps(left.schema_descriptor("time")) == json.dumps(right.schema_descriptor("time"))


@pytest.mark.parametrize(
    "changed",
    [
        {"axis_units": {"wavelength": "um"}},
        {"axis_coordinates": {"wavelength": [1000, 1010, 1021, 1030, 1040]}},
        {"axis_coordinates": {"wavelength": [1040, 1030, 1020, 1010, 1000]}},
        {"axis_coordinates": {}},
        {"feature_names": ["a", "b", "c", "d", "e"]},
    ],
)
def test_source_schema_detects_equal_shape_incompatible_axes(dataset, changed):
    original = dataset.sources["nir"]
    metadata = {"axis_units": original.axis_units, "axis_coordinates": original.axis_coordinates, "feature_names": original.feature_names}
    metadata.update(changed)
    altered = TensorSource(original.values, original.sample_ids, representation_id=original.representation_id, **metadata)
    assert altered.values.shape == original.values.shape
    assert altered.schema_descriptor("nir") != original.schema_descriptor("nir")


def test_metadata_column_reorder_is_visible_in_replay_schema(dataset):
    original = dataset.sources["metadata"]
    reordered = TensorSource(
        original.values[:, ::-1], original.sample_ids, representation_id=original.representation_id,
        feature_names=["cultivar", "temperature"], axis_coordinates={"column": ["cultivar", "temperature"]},
    )
    assert reordered.values.shape == original.values.shape
    assert reordered.schema_descriptor("metadata") != original.schema_descriptor("metadata")


def test_pickle_roundtrip_revalidates_host_arrays_and_keeps_ownership(dataset):
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.sample_ids == dataset.sample_ids
    assert restored.descriptors() == dataset.descriptors()
    np.testing.assert_array_equal(restored.y, dataset.y)
    np.testing.assert_array_equal(restored.groups, dataset.groups)
    np.testing.assert_array_equal(restored.partitions, dataset.partitions)
    assert not restored.y.flags.writeable
    assert not restored.groups.flags.writeable
    assert not restored.partitions.flags.writeable
    for name, source in restored.sources.items():
        np.testing.assert_array_equal(source.values, dataset.sources[name].values)
        assert not source.values.flags.writeable
    assert not pickle.loads(pickle.dumps(dataset.sources["image"])).values.flags.writeable


def test_source_vocabulary_matches_published_registry_when_sibling_is_present(dataset):
    registry_path = Path(__file__).resolve().parents[2] / "dag-ml-data/docs/contracts/representation_registry.v1.json"
    if not registry_path.exists():
        pytest.skip("The optional sibling registry is not checked out")
    registry = {entry["representation_id"]: entry for entry in json.loads(registry_path.read_text())["representations"]}
    for descriptor in dataset.descriptors():
        entry = registry[descriptor["representation_id"]]
        native = descriptor["native_representation"]
        expected = entry["representation"]
        assert native["id"] == expected["id"]
        assert native["type_id"] == expected["type_id"]
        assert native["rank"] == expected["rank"]
        assert descriptor["modality"] == entry["modality"]
        assert [(axis["name"], axis["kind"]) for axis in native["axes"]] == [
            (axis["name"], axis["kind"]) for axis in expected["axes"]
        ]


def test_python_oracle_and_native_wheel_share_the_same_host_container():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/nirs4all_io/multimodal.py").read_bytes() == (root / "bindings/python/python/nirs4all_io/multimodal.py").read_bytes()


@pytest.mark.parametrize("empty", [False, True])
def test_json_roundtrip_preserves_raw_buffers_axes_and_identity(dataset, empty):
    original = dataset.take([]) if empty else dataset
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    restored = MultimodalDataset.from_dict(payload)
    assert restored.name == original.name
    assert restored.sample_ids == original.sample_ids
    assert restored.descriptors() == original.descriptors()
    assert restored.target_descriptor() == original.target_descriptor()
    assert list(restored.sources) == list(original.sources)
    for name, source in restored.sources.items():
        expected = original.sources[name]
        assert source.values.shape == expected.values.shape
        assert source.values.dtype == expected.values.dtype
        np.testing.assert_array_equal(source.values, expected.values)
        assert not source.values.flags.writeable
    for name in ("y", "target_mask", "groups", "partitions"):
        actual, expected = getattr(restored, name), getattr(original, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert not actual.flags.writeable
    if not empty:
        payload["sources"][0]["array"]["values"][0][0] = -999
        assert restored.sources["nir"].values[0, 0] == original.sources["nir"].values[0, 0]


def test_json_handles_optional_targets_and_nonfinite_object_cells():
    ids = ["a", "b"]
    original = MultimodalDataset(
        {
            "nir": TensorSource(np.array([[np.nan, np.inf], [-np.inf, 2]], dtype=">f4"), ids, representation_id="signal_1d"),
            "metadata": TensorSource([[None, "A", np.nan], [True, "B", 1.0]], ids, representation_id="tabular_mixed"),
        },
        sample_ids=ids, partitions=["predict", "predict"],
    )
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    assert payload["sources"][0]["array"]["values"][0] == [{"nonfinite": "nan"}, {"nonfinite": "inf"}]
    restored = MultimodalDataset.from_dict(payload)
    assert restored.y is None and restored.groups is None
    assert restored.sources["nir"].values.dtype == np.dtype(">f4")
    np.testing.assert_array_equal(restored.sources["nir"].values, original.sources["nir"].values)
    mixed = restored.sources["metadata"].values
    assert mixed[0, 0] is None and mixed[1, 0] is True
    assert mixed[0, 1] == "A" and mixed[1, 1] == "B"
    assert np.isnan(mixed[0, 2]) and mixed[1, 2] == 1.0


def test_json_realigns_each_explicit_source_identity(dataset):
    payload = dataset.to_dict()
    image = payload["sources"][1]
    image["sample_ids"] = image["sample_ids"][::-1]
    image["array"]["values"] = image["array"]["values"][::-1]
    restored = MultimodalDataset.from_dict(payload)
    np.testing.assert_array_equal(restored.sources["image"].values, dataset.sources["image"].values)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "nirs4all.other"),
        (("schema_version",), 0),
        (("schema_version",), 2),
        (("schema_version",), True),
        (("unknown",), None),
        (("sample_ids",), "plant-a"),
        (("sample_ids",), ["plant-a"] * 4),
        (("sources",), []),
        (("sources", 1, "name"), "nir"),
        (("sources", 0, "unknown"), None),
        (("sources", 0, "sample_ids"), ["plant-a", "plant-b", "plant-c", "missing"]),
        (("sources", 0, "array", "unknown"), None),
        (("sources", 0, "array", "shape"), [4, 1, 5]),
        (("sources", 0, "array", "shape"), [4, True]),
        (("sources", 0, "array", "dtype"), "unknown"),
        (("sources", 0, "array", "dtype"), "complex128"),
        (("sources", 0, "array", "values", 0, 0), 0.5),
        (("sources", 0, "array", "values", 0, 0), "0"),
        (("sources", 2, "array", "values", 0, 0, 0), 1e100),
        (("sources", 2, "array", "values", 0, 0, 0), float("nan")),
        (("sources", 2, "array", "values", 0, 0, 0), {"nonfinite": "unknown"}),
        (("sources", 1, "array", "values", 0, 0, 0, 0), 256),
        (("sources", 1, "array", "values", 0, 0, 0, 0), -1),
        (("sources", 2, "axis_coordinates", "time"), [0, 2, 1, 3, 4, 5]),
        (("sources", 3, "feature_names"), ["only-one"]),
        (("groups", "values", 2), "plot-a"),
        (("groups", "dtype"), "<U1"),
        (("y", "dtype"), "bool"),
        (("y",), {"dtype": "object", "shape": [4], "values": [[1, 2], [1], [1], [1]]}),
    ],
)
def test_json_rejects_unknown_contracts_and_malformed_payloads(dataset, path, value):
    payload = dataset.to_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        MultimodalDataset.from_dict(payload)


def test_json_requires_all_declared_fields(dataset):
    payload = dataset.to_dict()
    del payload["sources"][0]["array"]["dtype"]
    with pytest.raises(ValueError, match="missing=.*dtype"):
        MultimodalDataset.from_dict(payload)


def test_named_partial_targets_keep_sample_major_identity_and_raw_values(dataset):
    y = np.array([[1, np.nan], [2, 20], [np.inf, 30], [4, -np.inf]], dtype=np.float32)
    mask = np.array([[True, False], [False, True], [False, True], [True, False]])
    names = ["sugar", "protein"]
    partial = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_names=names, target_mask=mask)
    assert partial.target_names == ("sugar", "protein")
    assert partial.y.dtype == np.float32
    np.testing.assert_array_equal(partial.y, y)
    np.testing.assert_array_equal(partial.target_mask, mask)
    assert partial.target_descriptor() == {
        "target_names": ["sugar", "protein"], "task_type": None, "layout": "sample_major", "axes": ["sample", "target"],
        "shape": [4, 2], "dtype": "float32", "target_mask": {"shape": [4, 2], "dtype": "bool", "true_means": "observed"},
    }
    assert not partial.y.flags.writeable and not partial.target_mask.flags.writeable
    y[0, 0], mask[0, 0], names[0] = -999, False, "changed"
    assert partial.y[0, 0] == 1 and partial.target_mask[0, 0]
    assert partial.target_names[0] == "sugar"
    selected = partial.take(["plant-d", "plant-b", "plant-a"])
    assert selected.sample_ids == ("plant-d", "plant-b", "plant-a")
    np.testing.assert_array_equal(selected.y, partial.y[[3, 1, 0]])
    np.testing.assert_array_equal(selected.target_mask, partial.target_mask[[3, 1, 0]])
    assert selected.target_names == partial.target_names
    assert selected.y[1, 0] == 2 and not selected.target_mask[1, 0]  # Withholding never overwrites a finite label.
    assert selected.sources["image"].values.shape == (3, 3, 2, 3)


@pytest.mark.parametrize("empty", [False, True])
def test_partial_multitarget_roundtrips_json_and_pickle(dataset, empty):
    partial = MultimodalDataset(
        dataset.sources, sample_ids=dataset.sample_ids,
        y=np.array([[1, np.nan], [2, 20], [np.inf, 30], [4, -np.inf]], dtype=">f4"),
        target_names=["sugar", "protein"], target_mask=[[True, False], [False, True], [False, True], [True, False]],
        groups=dataset.groups, partitions=dataset.partitions,
    )
    if empty:
        partial = partial.take([])
    payload = json.loads(json.dumps(partial.to_dict(), allow_nan=False))
    assert payload["schema_version"] == 1
    assert payload["target_mask"]["dtype"] == "bool"
    for restored in [MultimodalDataset.from_dict(payload), pickle.loads(pickle.dumps(partial))]:
        assert restored.target_descriptor() == partial.target_descriptor()
        assert restored.sample_ids == partial.sample_ids
        np.testing.assert_array_equal(restored.y, partial.y)
        np.testing.assert_array_equal(restored.target_mask, partial.target_mask)
        assert not restored.target_mask.flags.writeable
        assert not restored.y.flags.writeable
        assert restored.y.shape == (0 if empty else 4, 2)


@pytest.mark.parametrize("shape", [(4,), (4, 1), (4, 3)])
def test_complete_targets_have_explicit_default_validity_and_names(dataset, shape):
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.ones(shape, dtype=np.int16))
    expected_names = ("y0", "y1", "y2") if shape == (4, 3) else ("y",)
    assert cohort.target_names == expected_names
    assert cohort.y.shape == shape and cohort.y.dtype == np.int16
    assert cohort.target_mask.shape == shape and cohort.target_mask.dtype == bool
    assert cohort.target_mask.all() and not cohort.target_mask.flags.writeable


@pytest.mark.parametrize("names", [None, [], ["sugar", "protein"]])
def test_absent_targets_can_declare_output_names_without_fabricating_values(dataset, names):
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, target_names=names, partitions=["predict"] * 4)
    for restored in [cohort.take(["plant-b"]), pickle.loads(pickle.dumps(cohort)), MultimodalDataset.from_dict(cohort.to_dict())]:
        assert restored.y is None and restored.target_mask is None
        assert restored.target_names == tuple(names or ())
        assert restored.target_descriptor() == {
            "target_names": list(names or []), "task_type": None, "layout": "sample_major", "axes": None,
            "shape": None, "dtype": None, "target_mask": None,
        }


def test_object_targets_keep_missing_numeric_and_categorical_labels_distinct(dataset):
    y = np.array([[1, "A"], [None, "B"], [np.nan, None], [np.inf, "nan"]], dtype=object)
    mask = [[True, True], [False, True], [False, False], [False, True]]
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_mask=mask)
    restored = MultimodalDataset.from_dict(json.loads(json.dumps(cohort.to_dict(), allow_nan=False)))
    assert restored.y.dtype == object and restored.y[0].tolist() == [1, "A"]
    assert restored.y[1].tolist() == [None, "B"]
    assert np.isnan(restored.y[2, 0]) and restored.y[2, 1] is None
    assert np.isposinf(restored.y[3, 0]) and restored.y[3, 1] == "nan"
    np.testing.assert_array_equal(restored.target_mask, mask)


def test_fully_unlabelled_output_is_representable_without_imputation(dataset):
    y = np.column_stack([np.arange(4), np.full(4, np.nan)])
    mask = np.column_stack([np.ones(4, dtype=bool), np.zeros(4, dtype=bool)])
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_mask=mask)
    assert np.isnan(cohort.y[:, 1]).all() and not cohort.target_mask[:, 1].any()
    assert len(cohort) == 4 and cohort.target_names == ("y0", "y1")


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf])
def test_missing_labels_require_explicit_masks_and_cannot_be_marked_observed(dataset, missing):
    y = np.array([1, 2, 3, missing], dtype=object)
    with pytest.raises(ValueError, match="explicit target_mask"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y)
    with pytest.raises(ValueError, match="as observed"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_mask=[True] * 4)


@pytest.mark.parametrize(
    "mask",
    [True, [True] * 4, [[True, True]], [[True], [True], [True], [True]], [[1, 0]] * 4, [["true", "false"]] * 4, np.ones((4, 2), dtype=object)],
)
def test_target_masks_are_boolean_exact_shape_and_never_broadcast(dataset, mask):
    with pytest.raises(ValueError, match="boolean dtype and exactly y.shape"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.ones((4, 2)), target_mask=mask)


@pytest.mark.parametrize("names", ["protein", [], ["protein"], ["protein", "protein"], ["protein", ""], ["protein", 1]])
def test_target_names_are_unique_explicit_axis_labels(dataset, names):
    with pytest.raises(ValueError, match="target_names"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.ones((4, 2)), target_names=names)


def test_target_axes_and_masked_array_inputs_are_validated_before_use(dataset):
    with pytest.raises(ValueError, match="at least one target"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.empty((4, 0)))
    with pytest.raises(ValueError, match="when y is absent"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, target_mask=[False] * 4)
    with pytest.raises(ValueError, match="NumPy masked arrays"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.ma.array([1, 2, 3, 4], mask=[False, False, True, False]))
    with pytest.raises(ValueError, match="NumPy masked arrays"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.arange(4), target_mask=np.ma.array([True] * 4))


def test_version_one_optional_target_fields_use_constructor_defaults(dataset):
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.ones((4, 2)))
    payload = cohort.to_dict()
    del payload["target_names"], payload["target_mask"], payload["task_type"]
    restored = MultimodalDataset.from_dict(payload)
    assert restored.target_names == ("y0", "y1") and restored.target_mask.all()
    assert restored.task_type is None
    payload["y"] = None
    restored = MultimodalDataset.from_dict(payload)
    assert restored.y is None and restored.target_mask is None and restored.target_names == ()


@pytest.mark.parametrize("change", ["missing_mask", "observed_missing", "int_mask", "unknown_mask_field", "transposed_mask", "bad_names", "scalar_names"])
def test_json_partial_target_contract_rejects_malformed_masks(dataset, change):
    payload = dataset.to_dict()
    if change == "missing_mask":
        del payload["target_mask"]
    elif change == "observed_missing":
        payload["target_mask"]["values"][-1] = True
    elif change == "int_mask":
        payload["target_mask"] = {"dtype": "int64", "shape": [4], "values": [1, 1, 1, 0]}
    elif change == "unknown_mask_field":
        payload["target_mask"]["auto"] = True
    elif change == "transposed_mask":
        payload["target_mask"] = {"dtype": "bool", "shape": [1, 4], "values": [[True, True, True, False]]}
    elif change == "bad_names":
        payload["target_names"] = ["y", "extra"]
    else:
        payload["target_names"] = "y"
    with pytest.raises(ValueError):
        MultimodalDataset.from_dict(payload)


@pytest.mark.parametrize("task_type", [None, "regression", "classification"])
@pytest.mark.parametrize("absent", [False, True])
def test_declared_task_survives_views_json_pickle_without_label_encoding(dataset, task_type, absent):
    y = None if absent else np.array([1, 2, 3, 4], dtype=np.int16)
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_names=["target"], task_type=task_type)
    for restored in [cohort.take(dataset.sample_ids), pickle.loads(pickle.dumps(cohort)), MultimodalDataset.from_dict(cohort.to_dict())]:
        assert restored.task_type == task_type
        assert restored.target_descriptor()["task_type"] == task_type
        if absent:
            assert restored.y is None and restored.target_mask is None
        else:
            np.testing.assert_array_equal(restored.y, y)
            assert restored.y.dtype == np.int16


@pytest.mark.parametrize("task_type", ["auto", "Regression", "multiclass", "", 0, False, {}, []])
def test_unknown_task_declarations_are_rejected_in_constructor_and_json(dataset, task_type):
    with pytest.raises(ValueError, match="task_type"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=np.arange(4), task_type=task_type)
    payload = dataset.to_dict()
    payload["task_type"] = task_type
    with pytest.raises(ValueError, match="task_type"):
        MultimodalDataset.from_dict(payload)


def test_python_mixed_labels_do_not_turn_missing_values_into_string_classes(dataset):
    y = ["A", "B", np.nan, "nan"]
    with pytest.raises(ValueError, match="explicit target_mask"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y)
    cohort = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=y, target_mask=[True, True, False, True])
    restored = MultimodalDataset.from_dict(cohort.to_dict())
    assert restored.y.dtype == object and np.isnan(restored.y[2])
    assert restored.y[3] == "nan" and restored.target_mask[3]
    mixed = MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, y=[[1, "A"], [2, "B"], [3, "C"], [4, "D"]])
    assert mixed.y[0].tolist() == [1, "A"]


@pytest.fixture
def partial_sources(dataset):
    sources = {
        "nir": replace(dataset.sources["nir"].take(["plant-c", "plant-a"]), presence_mask=[False, True]),
        "image": dataset.sources["image"].take(["plant-d", "plant-b"]),
        "temporal": dataset.sources["temporal"].take([]),
        "metadata": dataset.sources["metadata"].take(["plant-b", "plant-a", "plant-d"]),
    }
    return MultimodalDataset(
        sources, sample_ids=dataset.sample_ids, source_alignment="left", y=dataset.y,
        target_names=dataset.target_names, target_mask=dataset.target_mask, task_type=dataset.task_type,
        groups=dataset.groups, partitions=dataset.partitions, name=dataset.name,
    )


def test_left_alignment_copies_partial_sources_by_ids_and_preserves_explicit_absence(dataset, partial_sources):
    assert partial_sources.source_alignment == "left"
    assert partial_sources.sample_ids == dataset.sample_ids
    assert list(partial_sources.sources) == list(dataset.sources)
    masks = partial_sources.source_presence()
    assert {name: mask.tolist() for name, mask in masks.items()} == {
        "nir": [True, False, False, False], "image": [False, True, False, True],
        "temporal": [False, False, False, False], "metadata": [True, True, False, True],
    }
    for name, rows in {"nir": [0, 2], "image": [1, 3], "metadata": [0, 1, 3]}.items():
        np.testing.assert_array_equal(partial_sources.sources[name].values[rows], dataset.sources[name].values[rows])
    assert np.all(partial_sources.sources["nir"].values[[1, 3]] == 0)
    assert np.all(partial_sources.sources["image"].values[[0, 2]] == 0)
    assert np.all(partial_sources.sources["temporal"].values == 0)
    assert partial_sources.sources["metadata"].values[2].tolist() == [None, None]
    assert all(source.sample_ids == dataset.sample_ids for source in partial_sources.sources.values())
    assert all(not mask.flags.writeable for mask in masks.values())
    assert all(not source.values.flags.writeable for source in partial_sources.sources.values())
    np.testing.assert_array_equal(partial_sources.y, dataset.y)
    np.testing.assert_array_equal(partial_sources.target_mask, dataset.target_mask)
    assert len(partial_sources) == 4  # A row with no present modality is not silently removed.


def test_strict_alignment_is_default_and_never_infers_missing_ids(dataset):
    assert dataset.source_alignment == "strict"
    assert all(mask.all() for mask in dataset.source_presence().values())
    sources = {"nir": dataset.sources["nir"].take(["plant-c", "plant-a"])}
    with pytest.raises(ValueError, match="missing=.*plant-b.*plant-d"):
        MultimodalDataset(sources, sample_ids=dataset.sample_ids)
    with pytest.raises(ValueError, match="missing="):
        MultimodalDataset(sources, sample_ids=dataset.sample_ids, source_alignment="strict")
    aligned = MultimodalDataset(sources, sample_ids=dataset.sample_ids, source_alignment="left")
    assert aligned.source_presence()["nir"].tolist() == [True, False, True, False]


@pytest.mark.parametrize("alignment", ["strict", "left"])
def test_foreign_source_ids_are_rejected_even_if_flagged_absent(alignment):
    source = TensorSource(np.ones((2, 3)), ["a", "foreign"], representation_id="signal_1d", presence_mask=[True, False])
    with pytest.raises(ValueError, match="extra=.*foreign"):
        MultimodalDataset({"nir": source}, sample_ids=["a", "b"], source_alignment=alignment)


@pytest.mark.parametrize("alignment", [None, "outer", "inner", "LEFT", "", False, {}, []])
def test_unknown_source_alignment_policy_is_refused(dataset, alignment):
    with pytest.raises(ValueError, match="source_alignment"):
        MultimodalDataset(dataset.sources, sample_ids=dataset.sample_ids, source_alignment=alignment)


@pytest.mark.parametrize(
    "mask",
    [True, False, [True], [True, False, True], [[True], [False]], [1, 0], ["true", "false"],
     np.array([True, False], dtype=object), np.ma.array([True, False])],
)
def test_source_presence_requires_exact_boolean_sample_vector(mask):
    with pytest.raises(ValueError, match="presence_mask"):
        TensorSource(np.ones((2, 3)), ["a", "b"], representation_id="signal_1d", presence_mask=mask)


def test_presence_mask_is_owned_and_reordered_with_source_ids_without_touching_hidden_values():
    values = np.array([[np.nan, 9], [3, 4]], dtype=np.float32)
    mask = np.array([False, True])
    source = TensorSource(values, ["b", "a"], representation_id="signal_1d", presence_mask=mask)
    values[0, 1], mask[0] = 999, True
    aligned = MultimodalDataset({"nir": source}, sample_ids=["a", "b"])
    assert aligned.sources["nir"].presence_mask.tolist() == [True, False]
    assert aligned.sources["nir"].values[0].tolist() == [3, 4]
    assert np.isnan(aligned.sources["nir"].values[1, 0]) and aligned.sources["nir"].values[1, 1] == 9
    with pytest.raises(ValueError, match="read-only"):
        source.presence_mask[0] = True
    default = TensorSource([[np.nan], [0.0]], ["a", "b"], representation_id="signal_1d")
    assert default.presence_mask.all()  # Non-finite cells never imply an absent whole modality.


@pytest.mark.parametrize("rows", [[3, 1, 0], [1, 1], []])
def test_positional_presence_projection_matches_source_buffers(partial_sources, rows):
    masks = partial_sources.source_presence(rows)
    assert list(masks) == list(partial_sources.sources)
    for name, mask in masks.items():
        assert mask.dtype == bool and mask.shape == (len(rows),)
        np.testing.assert_array_equal(mask, partial_sources.sources[name].presence_mask[rows])
        assert not mask.flags.writeable


@pytest.mark.parametrize("rows", [[-1], [4], [0.5], [True], [[0]], "0"])
def test_presence_projection_refuses_invalid_rows(partial_sources, rows):
    with pytest.raises(ValueError, match="row_indices"):
        partial_sources.source_presence(rows)


@pytest.mark.parametrize("ids", [["plant-d", "plant-a"], []])
def test_left_aligned_views_and_roundtrips_preserve_source_and_target_masks(partial_sources, ids):
    selected = partial_sources.take(ids)
    positions = [partial_sources.sample_ids.index(sample) for sample in ids]
    payload = json.loads(json.dumps(selected.to_dict(), allow_nan=False))
    assert payload["source_alignment"] == "left"
    assert all(source["presence_mask"]["dtype"] == "bool" for source in payload["sources"])
    for restored in [selected, MultimodalDataset.from_dict(payload), pickle.loads(pickle.dumps(selected))]:
        assert restored.source_alignment == "left" and restored.sample_ids == tuple(ids)
        np.testing.assert_array_equal(restored.target_mask, partial_sources.target_mask[positions])
        np.testing.assert_array_equal(restored.groups, partial_sources.groups[positions])
        np.testing.assert_array_equal(restored.partitions, partial_sources.partitions[positions])
        for name, source in restored.sources.items():
            np.testing.assert_array_equal(source.values, partial_sources.sources[name].values[positions])
            np.testing.assert_array_equal(source.presence_mask, partial_sources.sources[name].presence_mask[positions])
            assert not source.presence_mask.flags.writeable and not source.values.flags.writeable
        assert restored.schema_descriptors() == partial_sources.schema_descriptors()


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, ">f4", bool, "<U5", object])
def test_left_alignment_placeholders_preserve_dtype_and_observed_scalars(dtype):
    values = np.array([[1, 2]], dtype=dtype)
    source = TensorSource(values, ["b"], representation_id="tabular_mixed")
    cohort = MultimodalDataset({"table": source}, sample_ids=["a", "b"], source_alignment="left")
    aligned = cohort.sources["table"]
    assert aligned.values.dtype == source.values.dtype
    np.testing.assert_array_equal(aligned.values[1], source.values[0])
    if aligned.values.dtype.kind == "O":
        assert aligned.values[0].tolist() == [None, None]
    elif aligned.values.dtype.kind == "U":
        assert aligned.values[0].tolist() == ["", ""]
    else:
        assert not aligned.values[0].any()
    for restored in [MultimodalDataset.from_dict(cohort.to_dict()), pickle.loads(pickle.dumps(cohort))]:
        assert restored.sources["table"].values.dtype == source.values.dtype
        np.testing.assert_array_equal(restored.sources["table"].values, aligned.values)
        assert restored.source_presence()["table"].tolist() == [False, True]


def test_fully_absent_source_keeps_explicit_tensor_shape_axes_and_dtype(dataset):
    empty = dataset.sources["image"].take([])
    assert empty.values.shape == (0, 3, 2, 3) and empty.presence_mask.shape == (0,)
    with pytest.raises(ValueError, match="missing="):
        MultimodalDataset({"image": empty}, sample_ids=["new-1", "new-2"])
    prediction = MultimodalDataset({"image": empty}, sample_ids=["new-1", "new-2"], source_alignment="left", partitions=["predict"] * 2)
    for restored in [prediction, MultimodalDataset.from_dict(prediction.to_dict()), pickle.loads(pickle.dumps(prediction))]:
        assert restored.y is None and restored.target_mask is None
        assert restored.sources["image"].values.shape == (2, 3, 2, 3)
        assert restored.sources["image"].values.dtype == np.uint8
        assert not restored.source_presence()["image"].any()
        assert restored.sources["image"].schema_descriptor("image") == empty.schema_descriptor("image")
    with pytest.raises(ValueError, match="rank"):
        TensorSource([], [], representation_id="rgb_image")
    with pytest.raises(ValueError, match="non-empty"):
        TensorSource(np.empty((0, 0, 2, 3)), [], representation_id="rgb_image")


def test_presence_patterns_and_alignment_policy_do_not_change_replay_source_schema(dataset, partial_sources):
    assert partial_sources.schema_descriptors() == dataset.schema_descriptors()
    assert partial_sources.descriptors() == dataset.descriptors()
    altered = replace(partial_sources.sources["nir"], presence_mask=[False, True, True, True])
    assert altered.schema_descriptor("nir") == dataset.sources["nir"].schema_descriptor("nir")
    assert "presence_mask" not in altered.schema_descriptor("nir")


def test_version_one_omitted_presence_fields_have_strict_complete_defaults(dataset):
    payload = dataset.to_dict()
    del payload["source_alignment"]
    for source in payload["sources"]:
        del source["presence_mask"]
    restored = MultimodalDataset.from_dict(payload)
    assert restored.source_alignment == "strict"
    assert all(mask.all() for mask in restored.source_presence().values())
    np.testing.assert_array_equal(restored.target_mask, dataset.target_mask)


@pytest.mark.parametrize("change", ["alignment", "mask_dtype", "mask_rank", "mask_length", "unknown_mask_field"])
def test_json_source_presence_is_validated_through_normal_constructors(dataset, change):
    payload = dataset.to_dict()
    source = payload["sources"][0]
    if change == "alignment":
        payload["source_alignment"] = "outer"
    elif change == "mask_dtype":
        source["presence_mask"] = {"dtype": "uint8", "shape": [4], "values": [1, 1, 1, 0]}
    elif change == "mask_rank":
        source["presence_mask"] = {"dtype": "bool", "shape": [1, 4], "values": [[True, True, True, False]]}
    elif change == "mask_length":
        source["presence_mask"] = {"dtype": "bool", "shape": [3], "values": [True, True, False]}
    else:
        source["presence_mask"]["fill"] = True
    with pytest.raises(ValueError):
        MultimodalDataset.from_dict(payload)


def test_json_left_alignment_reads_unaligned_subsets_without_losing_mask_order(dataset):
    payload = dataset.to_dict()
    payload["source_alignment"] = "left"
    source = payload["sources"][0]
    source["sample_ids"] = ["plant-c", "plant-a"]
    source["array"]["shape"][0] = 2
    source["array"]["values"] = [source["array"]["values"][2], source["array"]["values"][0]]
    source["presence_mask"] = {"dtype": "bool", "shape": [2], "values": [False, True]}
    restored = MultimodalDataset.from_dict(payload)
    assert restored.source_presence()["nir"].tolist() == [True, False, False, False]
    np.testing.assert_array_equal(restored.sources["nir"].values[[0, 2]], dataset.sources["nir"].values[[0, 2]])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rows", [0, 2])
def test_source_numpy_masked_arrays_require_explicit_plain_buffer(masked, rows):
    # np.array(masked_array) previously exposed hidden measurements as present.
    values = np.ma.array(np.full((rows, 2), 999.0), mask=masked)
    with pytest.raises(ValueError, match="plain array with explicit presence_mask"):
        TensorSource(values, [f"s{i}" for i in range(rows)], representation_id="signal_1d", presence_mask=[True] * rows)


@pytest.mark.parametrize("dtype", ["S1", np.longdouble])
@pytest.mark.parametrize("empty", [False, True])
def test_json_rejects_nonrepresentable_dtypes_without_decoding_or_rounding(dtype, empty):
    if np.dtype(dtype).kind == "f" and np.dtype(dtype).itemsize <= 8:
        pytest.skip("Platform longdouble has no extended precision")
    values = np.ones((0 if empty else 1, 2), dtype=dtype)
    source = TensorSource(values, [] if empty else ["b"], representation_id="tabular_mixed")
    cohort = MultimodalDataset({"table": source}, sample_ids=["a", "b"], source_alignment="left")
    with pytest.raises(ValueError, match="Unsupported multimodal JSON dtype"):
        cohort.to_dict()
    assert cohort.sources["table"].values.dtype == values.dtype
    np.testing.assert_array_equal(cohort.sources["table"].values[1:1 if empty else 2], values)


def test_json_rejects_extended_float_object_cells_and_input_dtype(dataset):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("Platform longdouble has no extended precision")
    extended = np.longdouble(1) + np.finfo(np.longdouble).eps
    source = TensorSource(np.array([[extended]], dtype=object), ["a"], representation_id="tabular_mixed")
    cohort = MultimodalDataset({"table": source}, sample_ids=["a"])
    with pytest.raises(ValueError, match="Unsupported multimodal JSON scalar dtype"):
        cohort.to_dict()
    assert cohort.sources["table"].values[0, 0] == extended
    payload = dataset.to_dict()
    payload["sources"][0]["array"]["dtype"] = str(np.dtype(np.longdouble))
    with pytest.raises(ValueError, match="Unsupported .*dtype"):
        MultimodalDataset.from_dict(payload)
