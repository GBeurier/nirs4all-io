"""Experimental units are explicit IO identities, independent of split groups."""

from __future__ import annotations

import json
import pickle
from collections import UserDict
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from nirs4all_io import MultimodalDataset, TensorSource


def _cohort(**kwargs: Any) -> MultimodalDataset:
    ids = ["row_a", "row_b", "row_c", "row_d"]
    return MultimodalDataset({"nir": TensorSource(np.arange(12).reshape(4, 3), ids, representation_id="signal_1d")},
                             sample_ids=ids, y=[1, 1, 2, 2], **kwargs)


def test_explicit_units_repetitions_are_copied_and_read_only() -> None:
    units, repetitions = ["plant_a", "plant_a", "plant_b", "plant_b"], ["scan_0", "scan_1", "scan_0", "scan_1"]
    cohort = _cohort(independent_unit_ids=units, repetition_ids=repetitions, groups=["batch_a"] * 4)
    units[0], repetitions[0] = "changed", "changed"
    assert cohort.independent_unit_ids == ("plant_a", "plant_a", "plant_b", "plant_b")
    assert cohort.repetition_ids == ("scan_0", "scan_1", "scan_0", "scan_1")
    assert cohort.sample_ids == ("row_a", "row_b", "row_c", "row_d")
    assert cohort.groups.tolist() == ["batch_a"] * 4
    with pytest.raises(AttributeError):
        cohort.independent_unit_ids = ("changed",) * 4
    with pytest.raises(AttributeError):
        cohort.repetition_ids = ("changed",) * 4


def test_declared_units_survive_selection_json_pickle_and_python_mirror() -> None:
    cohort = _cohort(independent_unit_ids=["a", "a", "b", "b"], repetition_ids=["0", "1", "0", "1"])
    selected = cohort.take(["row_d", "row_a"])
    for restored in (selected, MultimodalDataset.from_dict(json.loads(json.dumps(selected.to_dict()))), pickle.loads(pickle.dumps(selected))):
        assert restored.independent_unit_ids == ("b", "a") and restored.repetition_ids == ("1", "0")
        assert restored.sample_ids == ("row_d", "row_a")
        assert not restored.sources["nir"].values.flags.writeable
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/nirs4all_io/multimodal.py").read_bytes() == (root / "bindings/python/python/nirs4all_io/multimodal.py").read_bytes()


@pytest.mark.parametrize("options", [
    {"repetition_ids": ["0"] * 4},
    {"independent_unit_ids": ["a"] * 4},
    {"independent_unit_ids": ["a", "a", "b", "b"], "repetition_ids": ["0", "0", "0", "1"]},
    {"independent_unit_ids": ["a", "a", "b", "b"], "repetition_ids": ["0", "1", "0", "1"], "partitions": ["train", "test", "train", "train"]},
    {"independent_unit_ids": ["a", "b", "c"]},
    {"independent_unit_ids": ["a", "b", "c", ""]},
    {"independent_unit_ids": ["a", "b", "c", None]},
    {"independent_unit_ids": ["a", "b", "c", "nul\x00"]},
    {"independent_unit_ids": {"a": 0, "b": 1, "c": 2, "d": 3}},
])
def test_bad_unit_or_repetition_contract_refused(options: Any) -> None:
    with pytest.raises(ValueError):
        _cohort(**options)


@pytest.mark.parametrize("label", ["independent_unit_ids", "repetition_ids"])
@pytest.mark.parametrize("bad", [
    "abcd",
    b"abcd",
    {"a": 0, "b": 1, "c": 2, "d": 3},
    MappingProxyType({"a": 0, "b": 1, "c": 2, "d": 3}),
    UserDict({"a": 0, "b": 1, "c": 2, "d": 3}),
    [0, 1, 2, 3],
    [True, False, True, False],
])
@pytest.mark.parametrize("construction", ["constructor", "from_dict"])
def test_experimental_labels_reject_wrong_types_on_both_public_paths(label: str, bad: Any, construction: str) -> None:
    valid = {"independent_unit_ids": ["a", "b", "c", "d"], "repetition_ids": ["scan"] * 4}
    if construction == "constructor":
        options: dict[str, Any] = {**valid, label: bad}
        with pytest.raises(ValueError, match=label):
            _cohort(**options)
    else:
        wire = _cohort(**valid).to_dict()
        wire[label] = bad
        with pytest.raises(ValueError, match=label):
            MultimodalDataset.from_dict(wire)


def test_groups_alone_do_not_create_units_or_change_old_json_wire() -> None:
    cohort = _cohort(groups=["a", "a", "b", "b"])
    assert cohort.independent_unit_ids is None and cohort.repetition_ids is None
    wire = cohort.to_dict()
    assert "independent_unit_ids" not in wire and "repetition_ids" not in wire
    restored = MultimodalDataset.from_dict(wire)
    assert restored.to_dict() == wire
    assert restored.independent_unit_ids is None


def test_unique_declared_units_do_not_require_repetition_ids() -> None:
    cohort = _cohort(independent_unit_ids=["a", "b", "c", "d"])
    assert cohort.repetition_ids is None
    assert cohort.independent_unit_ids == ("a", "b", "c", "d")
