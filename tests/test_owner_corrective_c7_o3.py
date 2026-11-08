# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C7 whole-content classification label cap and O3 aggregate outlier threshold."""

from __future__ import annotations

import copy
import json

import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.numeric import NumericContractError
from nirs4all_io.public_content import ContentNotRepresentableError, labelled_dataset_content_identity
from nirs4all_io.spec import DatasetSpec
from nirs4all_io.spec.enums import SpecError
from nirs4all_io.strict_json import loads

_TEMPLATE = (
    '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"p","sample_ids":["a","b","c"],"source_alignment":"strict",'
    '"sources":[{"name":"nir","sample_ids":["a","b","c"],"representation_id":"signal_1d","axes":["sample","wavelength"],"feature_names":null,'
    '"axis_units":{"wavelength":"nm"},"axis_coordinates":{"wavelength":[900.0,1000.0]},"array":{"dtype":"float64","shape":[3,2],"values":[[1.5,2.0],[3.0,4.0],[5.0,6.0]]}}],'
    '"y":{"dtype":"%(dtype)s","shape":[3],"values":%(y)s},"task_type":"%(task)s","groups":%(groups)s,"partitions":{"dtype":"<U7","shape":[3],"values":["train","train","train"]}}'
)


def _whole(y: str, dtype: str, task: str, groups: str = "null") -> dict[str, str]:
    return labelled_dataset_content_identity(loads(_TEMPLATE % {"y": y, "dtype": dtype, "task": task, "groups": groups}))


@pytest.mark.parametrize("label", ["9007199254740992", "-9007199254740992", "18014398509481984"])
def test_int64_classification_labels_beyond_the_portable_cap_refuse(label: str) -> None:
    with pytest.raises(NumericContractError, match="exceeds portable integer cap"):
        _whole(f"[{label},1,0]", "int64", "classification")


@pytest.mark.parametrize("label", ["9007199254740991", "-9007199254740991"])
def test_labels_at_the_cap_minus_one_are_valid(label: str) -> None:
    assert _whole(f"[{label},1,0]", "int64", "classification")["algorithm"] == "sha256"


def test_the_cap_is_a_label_rule_only() -> None:
    assert _whole("[18014398509481984,1,0]", "int64", "regression")  # generic exact int64 regression target
    assert _whole("[9007199254740992.0,1.0,0.0]", "float64", "classification")  # a float dtype is no integer label
    assert _whole("[1,0,1]", "int64", "classification", '{"dtype":"int64","shape":[3],"values":[18014398509481984,1,2]}')  # groups are not labels
    with pytest.raises(ContentNotRepresentableError):  # a regression 2**53 + 1 is a content refusal, not a label cap
        _whole("[9007199254740993,1,0]", "int64", "regression")


# --------------------------------------------------------------------------- #
# O3                                                                           #
# --------------------------------------------------------------------------- #
def _aggregate(**fields):
    return DatasetSpec.from_dict({"sources": [{"id": "s", "input": "d.csv", "role": "features"}], "aggregate": {"by": "g", "exclude_outliers": True, **fields}})


def test_present_threshold_must_be_a_finite_json_number() -> None:
    def threshold(**fields):
        aggregate = _aggregate(**fields).aggregate
        assert aggregate is not None
        return aggregate.outlier_threshold

    assert threshold() == 0.95  # absent keeps the default
    assert threshold(outlier_threshold=2) == 2.0
    assert threshold(outlier_threshold=0.5) == 0.5
    for bad in (True, False, "0.9", None, float("nan"), float("inf"), [1], {"v": 1}):
        with pytest.raises(SpecError, match="finite JSON number"):
            _aggregate(outlier_threshold=bad)


def test_package_wire_reader_uses_the_same_seam(tmp_path) -> None:
    (tmp_path / "d.csv").write_text("g,x,y\na,1.5,1.0\na,2.5,2.0\n")
    spec = {
        "sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["g"]}]}],
        "aggregate": {"by": "g", "exclude_outliers": True, "outlier_threshold": 0.75},
        "params": {"delimiter": ","},
    }
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    from nirs4all_io.canonical_json import canonical_json
    from nirs4all_io.materialize import DatasetPackage

    wire = json.loads(canonical_json(package.to_full_dict()))
    restored = DatasetPackage.from_full_dict(wire).aggregate
    assert restored is not None and restored.outlier_threshold == 0.75
    for bad in (True, "0.75", None):
        broken = copy.deepcopy(wire)
        broken["aggregate"]["outlier_threshold"] = bad
        with pytest.raises(NumericContractError, match="Aggregate outlier fields are malformed"):
            DatasetPackage.from_full_dict(broken)
