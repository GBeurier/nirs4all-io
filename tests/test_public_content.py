"""One raw-content byte contract shared with Rust and JavaScript."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from nirs4all_io.public_content import canonical_content_bytes, dataset_content_bytes, metadata_number
from nirs4all_io.public_dataset import dataset


def test_shared_byte_golden_and_numeric_spellings() -> None:
    golden = json.loads((Path(__file__).parent / "fixtures/public-content-v1.json").read_text())
    actual = canonical_content_bytes(golden["input"])
    assert actual.decode() == golden["canonical_content_utf8"]
    assert hashlib.sha256(actual).hexdigest() == golden["sha256"]
    reordered = dict(reversed(list(golden["input"].items())))
    assert canonical_content_bytes(reordered) == actual
    assert canonical_content_bytes([0, 1, 1000]) == canonical_content_bytes([-0.0, 1.0, 1e3])
    assert canonical_content_bytes([1]) != canonical_content_bytes(["1"])


def test_dataset_units_and_width_are_logical_and_roles_are_kept() -> None:
    ds = dataset({'spectra': [[1., 2.], [3., 4.]]}, y=[5., 6.], sample_ids=['a', 'b'],
                 partitions=['train', 'test'], origin_ids=['origin.a', 'origin.b'])
    a = ds.to_dict()
    b = deepcopy(a)
    b['dataset']['sources'][0]['axis_units']['wavelength'] = None
    b['dataset']['partitions']['dtype'] = '<U40'
    assert dataset_content_bytes(dataset(b).to_dict()) == dataset_content_bytes(a)
    projection = ds.to_dense_regression('spectra')
    assert projection['partitions'] == ['train', 'test']
    assert projection['origin_ids'] == ['origin.a', 'origin.b']
    assert projection['fold_ids'] == [None, None]


def test_metadata_decimal_grammar_is_explicit() -> None:
    for whitespace in [' ', '\t', '\n', '\r', '\v', '\f']:
        assert metadata_number(f'{whitespace}20.5{whitespace}') == 20.5
    assert metadata_number('1e3') == 1000
    for invalid in ['0x10', '1_000', '', 'NaN', True]:
        with pytest.raises(ValueError, match='decimal'):
            metadata_number(invalid)


def test_runtime_groups_preserve_string_ids_and_refuse_numeric_labels() -> None:
    from nirs4all_io.multimodal_runtime import multimodal_runtime_input

    record = json.loads((Path(__file__).parent / "fixtures/runtime-u07-groups.json").read_text())
    result = multimodal_runtime_input(record)
    assert [row["group_id"] for row in result["data_envelope"]["coordinator_relations"]["records"]] == ["1.0", "1"]
    for labels in [[1.0, 1.5], [1, 2], ["1", 2.0]]:
        numeric = deepcopy(record)
        numeric["dataset"]["groups"] = {"dtype": "object", "shape": [2], "values": labels}
        groups = dataset(numeric).multimodal.groups
        assert groups is not None
        assert groups.tolist() == labels
        with pytest.raises(ValueError, match="group IDs must be nonempty strings"):
            multimodal_runtime_input(numeric)
