# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C2: the explicit masked route discards withheld-cell placeholders BEFORE any dtype/label/declared-array gate.

A false mask cell holds no value: ``null``, a bool or a finite number are admitted there and replaced by the
canonical placeholder (false / 0.0 / 0); anything else refuses. Observed cells stay strict, the matrix/dense
routes stay strict, and the ORIGINAL record keeps the audit, ``source_evidence`` and the whole link (absent,
with no reason, when that original is not a valid whole record).
"""

from __future__ import annotations

import copy
import json

import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.numeric import NumericStorageOptions
from nirs4all_io.public_package import verified_dataset_identity

_TEMPLATE = (
    '{"schema":"nirs4all.dataset.v1","schema_version":1,"origin_ids":["a","b","c"],"fold_ids":[null,null,null],'
    '"dataset":{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"p","sample_ids":["a","b","c"],"source_alignment":"strict",'
    '"sources":[{"name":"nir","sample_ids":["a","b","c"],"representation_id":"signal_1d","axes":["sample","wavelength"],"feature_names":null,'
    '"axis_units":{"wavelength":"nm"},"axis_coordinates":{"wavelength":[900.0,1000.0]},"array":{"dtype":"%(xd)s","shape":[3,2],"values":%(x)s}}],'
    '"y":%(y)s,%(extra)s"groups":null,"partitions":{"dtype":"<U7","shape":[3],"values":["train","train","train"]}}}'
)
_FALSE_MIDDLE = '"target_mask":{"dtype":"bool","shape":[3],"values":[true,false,true]},'
_ALL_TRUE = '"target_mask":{"dtype":"bool","shape":[3],"values":[true,true,true]},'
_CLASSIFICATION = '"task_type":"classification",'


def _y(values: str, dtype: str = "float64") -> str:
    return f'{{"dtype":"{dtype}","shape":[3],"values":{values}}}'


def _record(y: str, extra: str = _FALSE_MIDDLE, x: str = "[[1.5,2.0],[3.0,4.0],[5.0,6.0]]", xd: str = "float64") -> str:
    return _TEMPLATE % {"xd": xd, "x": x, "y": y, "extra": extra}


def _masked(text: str, options: NumericStorageOptions | None = None):
    return nio.masked_matrix_dataset_package_with_storage(text, "nir", options)[0]


def _y_block(package):
    payload = dict(package.partitions["train"].payloads)["train/y"]
    return payload.columns[0].block


@pytest.mark.parametrize(
    ("name", "text", "y_hex"),
    [
        ("null", _record(_y("[1.0,null,3.0]")), "000000000000f03f00000000000000000000000000000840"),
        ("float32-overflow", _record(_y("[1.0,1e40,3.0]", "float32")), "0000803f0000000000004040"),
        ("float32-underflow", _record(_y("[1.0,5e-324,3.0]", "float32")), "0000803f0000000000004040"),
        ("float64-retained-integer", _record(_y("[1.0,9007199254740993,3.0]")), "000000000000f03f00000000000000000000000000000840"),
        ("bool", _record(_y("[1.0,true,3.0]")), "000000000000f03f00000000000000000000000000000840"),
        ("classification-1.5", _record(_y("[1,1.5,0]", "int64"), _CLASSIFICATION + _FALSE_MIDDLE), "000000000000f03f00000000000000000000000000000000"),
        ("classification-null", _record(_y("[1,null,0]", "int64"), _CLASSIFICATION + _FALSE_MIDDLE), "000000000000f03f00000000000000000000000000000000"),
        ("classification-bool", _record(_y("[1,true,0]", "int64"), _CLASSIFICATION + _FALSE_MIDDLE), "000000000000f03f00000000000000000000000000000000"),
    ],
)
def test_false_cell_placeholders_are_discarded_before_the_gates(name: str, text: str, y_hex: str) -> None:
    package = _masked(text)
    block = _y_block(package)
    assert block.values_bytes().hex() == y_hex and block.validity.ravel().tolist() == [True, False, True], name
    audit = package.audits[0]
    assert audit["dataset_content_identity"] is None and "dataset_content_identity_unavailable" not in audit
    assert audit["target_mask"]["values"] == [True, False, True]
    # the audit keeps the ORIGINAL record, placeholder included
    assert audit["owner_dataset_record"] == json.loads(text)


def test_a_valid_placeholder_keeps_the_whole_link() -> None:
    package = _masked(_record(_y("[1.0,99.0,3.0]")))
    assert package.audits[0]["dataset_content_identity"] is not None
    assert verified_dataset_identity(package) == package.audits[0]["dataset_content_identity"]


@pytest.mark.parametrize("placeholder", ['"x"', "[1]", '{"a":1}'])
def test_unsupported_placeholder_kinds_refuse(placeholder: str) -> None:
    with pytest.raises(ValueError, match="Masked target storage requires null or finite numeric values"):
        _masked(_record(_y(f"[1.0,{placeholder},3.0]")))


def test_nonfinite_placeholder_in_a_decoded_mapping_refuses() -> None:
    record = json.loads(_record(_y("[1.0,0.0,3.0]")))
    for bad in (float("nan"), float("inf")):
        mapping = copy.deepcopy(record)
        mapping["dataset"]["y"]["values"][1] = bad
        with pytest.raises(ValueError):
            nio.masked_matrix_dataset_package(mapping, "nir")


@pytest.mark.parametrize(
    "text",
    [_record(_y("[1.0,1e40,3.0]", "float32"), _ALL_TRUE), _record(_y("[1.0,5e-324,3.0]", "float32"), _ALL_TRUE), _record(_y("[1.0,9007199254740993,3.0]"), _ALL_TRUE)],
)
def test_observed_twins_of_the_placeholders_still_refuse(text: str) -> None:
    with pytest.raises(ValueError):
        _masked(text)


def test_non_masked_routes_stay_strict() -> None:
    for text in (_record(_y("[1.0,1e40,3.0]", "float32")), _record(_y("[1.0,null,3.0]"))):
        with pytest.raises(ValueError):
            nio.matrix_dataset_package(text, "nir")
        with pytest.raises(ValueError):
            nio.dense_dataset_package(text, "nir")


def test_mask_shape_and_dtype_are_checked() -> None:
    with pytest.raises(ValueError, match="Target mask mismatch"):
        _masked(_record(_y("[1.0,2.0,3.0]"), '"target_mask":{"dtype":"bool","shape":[2],"values":[true,false]},'))
    with pytest.raises(ValueError, match="Target mask mismatch"):
        _masked(_record(_y("[1.0,2.0,3.0]"), '"target_mask":{"dtype":"int64","shape":[3],"values":[1,0,1]},'))


def test_false_placeholder_with_an_inexact_integer_in_another_array_keeps_no_reason() -> None:
    """The previously unexecuted combination: the Rust predicate wins (link None, no ``unavailable``)."""
    x = "[[9007199254740993,2],[3,4],[5,6]]"
    options = NumericStorageOptions(features="float32")
    combo = _masked(_record(_y("[1.0,1e40,3.0]", "float32"), x=x, xd="int64"), options)
    assert combo.audits[0]["dataset_content_identity"] is None and "dataset_content_identity_unavailable" not in combo.audits[0]
    control = _masked(_record(_y("[1.0,2.0,3.0]"), x=x, xd="int64"), options)
    assert control.audits[0]["dataset_content_identity_unavailable"] == {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}


def test_source_evidence_keeps_the_original_record_and_verifies() -> None:
    package = _masked(_record(_y("[1.0,null,3.0]")))
    record, _source, _options, masked = package.source_evidence
    assert masked and record["dataset"]["y"]["values"][1] is None
    assert verified_dataset_identity(package) is None
