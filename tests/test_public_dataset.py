"""Public dataset identity, folds and host-array construction contracts."""
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from nirs4all_io.public_dataset import Dataset, dataset


def test_host_arrays_keep_values_identity_and_fold_origin() -> None:
    ds = dataset({'spectra': [[1., 2.], [3., 4.], [5., 6.]]}, y=[7., 8., 9.],
                 sample_ids=['a', 'b', 'c'], origin_ids=['plant', 'plant', 'other'],
                 fold_ids=['f0', 'f0', 'f1'])
    restored = Dataset.from_dict(ds.to_dict()).take(['c', 'a'])
    assert restored.sample_ids == ('c', 'a')
    assert restored.origin_ids == ('other', 'plant')
    assert restored.fold_ids == ('f1', 'f0')
    np.testing.assert_array_equal(restored.multimodal.sources['spectra'].values, [[5., 6.], [1., 2.]])


def test_origins_and_user_groups_cannot_cross_folds() -> None:
    with pytest.raises(ValueError, match='origin'):
        dataset({'spectra': [[1.], [2.]]}, y=[1., 2.], sample_ids=['a', 'b'], origin_ids=['same', 'same'], fold_ids=['0', '1'])
    with pytest.raises(ValueError, match='group'):
        dataset({'spectra': [[1.], [2.]]}, y=[1., 2.], sample_ids=['a', 'b'], groups=['same', 'same'], fold_ids=['0', '1'])


def test_envelope_refuses_unknowns_and_nonportable_storage() -> None:
    ds = dataset({'spectra': [[1.], [2.]]}, sample_ids=['a', 'b'])
    record = deepcopy(ds.to_dict())
    record['untrusted'] = True
    with pytest.raises(ValueError, match='fields'):
        Dataset.from_dict(record)
    with pytest.raises(ValueError, match='finite'):
        dataset({'spectra': [[float('nan')]]}, sample_ids=['a'])
    with pytest.raises(ValueError, match='JavaScript'):
        dataset({'spectra': np.asarray([[2**60]], dtype='int64')}, sample_ids=['a'])
    with pytest.raises(ValueError, match='coordinates.*JavaScript'):
        dataset({'spectra': [[1., 2.]]}, sample_ids=['a'],
                axis_coordinates={'spectra': {'wavelength': [2**53, 2**53 + 2]}})


def test_public_envelope_paths_and_json_yaml_keep_origin_and_folds(tmp_path: Path) -> None:
    record = dataset({'spectra': [[1.], [2.]]}, y=[4., 5.], sample_ids=['a', 'b'],
                     origin_ids=['origin-a', 'origin-b'], fold_ids=['0', '1']).to_dict()
    for name, text in [('dataset.json', json.dumps(record)), ('dataset.yaml', yaml.safe_dump(record))]:
        path = tmp_path / name
        path.write_text(text)
        loaders: tuple[Callable[..., Dataset], ...] = (dataset, Dataset)
        for value in [path, str(path), text]:
            for load in loaders:
                assert load(value).to_dict() == record
