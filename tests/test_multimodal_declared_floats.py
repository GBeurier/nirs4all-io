# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""E4/B3: the serialized-array gate lives in ``MultimodalDataset._array_from_dict``.

Every route that reaches a serialized array (raw ``from_dict``, the facade, the public ``Dataset``) normalizes the
ORIGINAL JSON cells against the declared dtype in one place, for sources, targets, groups, partitions, presence
masks and ragged series. Typed host arrays, pickle, raw strings and object metadata are never reinterpreted.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nirs4all_io import MultimodalDataset, TensorSource, strict_json
from nirs4all_io.dataset_facade import load_multimodal_definition
from nirs4all_io.public_dataset import dataset as public_dataset
from nirs4all_io.ragged import RaggedSeriesBatch
from nirs4all_io.strict_json import NumberDomainError

_DOC = (
    '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"m","sample_ids":["a"],'
    '"sources":[{"name":"s","sample_ids":["a"],"representation_id":"%(rep)s","axes":["sample","%(axis)s"],"feature_names":null,'
    '"axis_units":{"%(axis)s":null},"axis_coordinates":{},"array":{"dtype":"%(sd)s","shape":[1,1],"values":[[%(sv)s]]},'
    '"presence_mask":{"dtype":"bool","shape":[1],"values":[true]}}],'
    '"y":{"dtype":"%(yd)s","shape":[1],"values":[%(yv)s]},"target_names":["y"],"target_mask":{"dtype":"bool","shape":[1],"values":[true]},'
    '"groups":%(groups)s,"partitions":{"dtype":"<U5","shape":[1],"values":["train"]}}'
)
_RAGGED = (
    '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"m","sample_ids":["a"],"sources":[{"source_kind":"ragged_series","name":"r",'
    '"sample_ids":["a"],"representation_id":"series_mv","axes":["sample","time","variable"],"array":{"dtype":"%(ad)s","shape":[2,1],"values":[[%(av)s],[2.0]]},'
    '"offsets":{"dtype":"int64","shape":[2],"values":[0,2]},"time_coordinates":null,"channel_names":null,"time_unit":null,'
    '"presence_mask":{"dtype":"bool","shape":[1],"values":[true]}}],"y":{"dtype":"float64","shape":[1],"values":[1.0]},"target_names":["y"],'
    '"target_mask":{"dtype":"bool","shape":[1],"values":[true]},"groups":null,"partitions":{"dtype":"<U5","shape":[1],"values":["train"]}}'
)
_ROUTES: dict[str, Callable[[str], Any]] = {
    "from_dict": lambda text: MultimodalDataset.from_dict(strict_json.loads(text)),
    "facade": load_multimodal_definition,
    "public": public_dataset,
}
_UNDERFLOW = "Nonzero value rounds to zero in float32 storage"
_RETAINED = "cannot be represented exactly by a declared float64 array"


def _doc(sd="float64", sv="1.0", yd="float64", yv="1.0", rep="tabular_numeric", groups="null") -> str:
    return _DOC % {"sd": sd, "sv": sv, "yd": yd, "yv": yv, "rep": rep, "groups": groups, "axis": "column" if rep == "tabular_mixed" else "feature"}


def _cohort(value) -> MultimodalDataset:
    return getattr(value, "multimodal", value)


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_doc(sd="float32", sv="5e-324"), _UNDERFLOW),
        (_doc(sd="float64", sv="9007199254740993"), _RETAINED),
        (_doc(yd="float64", yv="9007199254740993"), _RETAINED),
        (_doc(yd="float32", yv="5e-324"), _UNDERFLOW),
        (_doc(groups='{"dtype":"float64","shape":[1],"values":[9007199254740993]}'), _RETAINED),
        (_RAGGED % {"ad": "float32", "av": "5e-324"}, _UNDERFLOW),
        (_RAGGED % {"ad": "float64", "av": "9007199254740993"}, _RETAINED),
        (_doc(sd="float32", sv="1e40"), "overflows float32|overflows dtype"),
    ],
)
def test_serialized_arrays_of_every_role_refuse_in_every_route(route: str, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _ROUTES[route](text)


@pytest.mark.parametrize("big", [str(2**64 + 1), str(10**400)])
def test_integer_domain_is_checked_first_and_never_leaks_overflow_error(big: str) -> None:
    for sd in ("float64", "float32"):
        record = strict_json.loads(_doc(sd=sd, sv="1"))
        record["sources"][0]["array"]["values"] = [[int(big)]]
        with pytest.raises(NumberDomainError):
            MultimodalDataset.from_dict(record)
    assert issubclass(NumberDomainError, ValueError) and not issubclass(NumberDomainError, OverflowError)


def test_valid_controls_keep_their_exact_bits() -> None:
    def source_hex(text: str, route: str = "from_dict") -> str:
        source = next(iter(_cohort(_ROUTES[route](text)).sources.values()))
        return np.ascontiguousarray(source.values).tobytes().hex()

    for route in _ROUTES:
        assert source_hex(_doc(sd="float32", sv=str(2**54 + 2**30 + 1)), route) == "0100805a", "direct nearest-even rounding of the original integer"
        assert source_hex(_doc(sd="float64", sv="18014398509481984"), route) == "0000000000005043"
        assert source_hex(_doc(sd="float64", sv="-0.0"), route) == "0000000000000080"
        assert source_hex(_doc(sd="float32", sv="1.401298464324817e-45"), route) == "01000000"
        assert source_hex(_doc(sd="float64", sv="9007199254740993.0"), route) == "0000000000004043", "a decimal token is an F64 value"
        assert source_hex(_doc(sd="float32", sv="16777217"), route) == "0000804b"
    ragged = MultimodalDataset.from_dict(strict_json.loads(_RAGGED % {"ad": "float64", "av": "18014398509481984"}))
    series = next(iter(ragged.sources.values())).values
    assert isinstance(series, RaggedSeriesBatch)
    assert np.ascontiguousarray(series.values).tobytes().hex().startswith("0000000000005043")


def test_raw_strings_object_cells_and_integer_labels_are_not_reinterpreted() -> None:
    mixed = MultimodalDataset.from_dict(strict_json.loads(_doc(sd="object", sv='"1e-400", 9007199254740993', rep="tabular_mixed").replace('"shape":[1,1]', '"shape":[1,2]')))
    cells = np.asarray(next(iter(mixed.sources.values())).values).tolist()
    assert cells == [["1e-400", 9007199254740993]]
    text_y = MultimodalDataset.from_dict(strict_json.loads(_doc(yd="<U6", yv='"1e-400"')))
    assert text_y.y is not None and text_y.y.tolist() == ["1e-400"]
    int_y = MultimodalDataset.from_dict(strict_json.loads(_doc(yd="int64", yv="9007199254740992")))
    assert int_y.y is not None and int(int_y.y[0]) == 2**53 and int_y.y.dtype == np.int64


def test_typed_host_arrays_keep_their_bits_through_pickle_and_roundtrip() -> None:
    host = MultimodalDataset(
        {"s": TensorSource(np.array([[float(2**53 + 1)]]), ["a"], representation_id="tabular_numeric")},
        sample_ids=["a"], y=np.array([1.0]), partitions=["train"],
    )
    wanted = np.array([float(2**53 + 1)]).tobytes().hex()  # 0000000000004043: a genuine host F64 already holds rounded bits
    for restored in (pickle.loads(pickle.dumps(host)), MultimodalDataset.from_dict(host.to_dict())):
        assert np.ascontiguousarray(next(iter(restored.sources.values())).values).tobytes().hex() == wanted
    tiny = MultimodalDataset(
        {"s": TensorSource(np.array([[np.float32(5e-324)]]), ["a"], representation_id="tabular_numeric")},
        sample_ids=["a"], y=np.array([1.0]), partitions=["train"],
    )
    assert np.ascontiguousarray(next(iter(MultimodalDataset.from_dict(tiny.to_dict()).sources.values())).values).tobytes().hex() == "00000000"
