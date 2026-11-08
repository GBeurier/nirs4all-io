# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Public ``Dataset`` -> package factories, owner carriers and the direct integer->F32 rule.

Covers the follow-up gates: exact F64 public-to-package roundtrips through BOTH the
assembled v3 and the package v4 full-wire readers (equal preimages and digests), the
required owner ``audits`` carrier, exact feature names / raw units / coordinate bits,
unsupported-shape refusals, the labelled whole-content wrappers over the unchanged
bytes, the optional dataset-content identity (evidence-only, never an echo), and the
generic-integer storage precedence (direct integer -> F32 rounding).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.materialize import DatasetPackage, NumericContractError, NumericStorageOptions
from nirs4all_io.materialize.assemble import AssembledDataset
from nirs4all_io.materialize.package import FeatureMatrix, MetadataTable, TargetTable, WeightsVector
from nirs4all_io.materialize.projection import ExecutedProjectionIdentityV1
from nirs4all_io.multimodal import TensorSource
from nirs4all_io.public_content import canonical_content_bytes, dataset_content_bytes, labelled_dataset_content_bytes, labelled_dataset_content_identity
from nirs4all_io.public_dataset import Dataset
from nirs4all_io.public_package import attach_source_record, verified_dataset_identity

IDS = ["s1", "s2", "s3", "s4"]


def _dataset(**changes: Any) -> Dataset:
    options: dict[str, Any] = {
        "sample_ids": IDS,
        "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "partitions": ["train", "train", "train", "test"],
        "fold_ids": ["a", "a", "b", None],
        "groups": ["g1", "g1", "g2", "g3"],
        "axis_units": {"nir": {"wavelength": "nm"}},
        "axis_coordinates": {"nir": {"wavelength": [1100.0, 1102.0]}},
        "feature_names": {"nir": ["w1100", "w1102"]},
    }
    x = changes.pop("x", np.array([[1.0, 1.0 + 2.0**-30], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    options.update(changes)
    return Dataset.from_sources({"nir": x}, **options)


def _digests(package: DatasetPackage, source: str = "nir"):
    receipt = package.executed_projection([source])
    return receipt.identity, receipt.feature_preimage, receipt.target_preimage


def _feature(package: DatasetPackage, partition: str = "train") -> FeatureMatrix:
    payload = dict(package.partitions[partition].payloads)[f"{partition}/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload


def _y(package: DatasetPackage, partition: str = "train") -> TargetTable:
    payload = dict(package.partitions[partition].payloads)[f"{partition}/y"]
    assert isinstance(payload, TargetTable)
    return payload


def _meta(package: DatasetPackage, partition: str = "train") -> MetadataTable:
    payload = dict(package.partitions[partition].payloads)[f"{partition}/metadata"]
    assert isinstance(payload, MetadataTable)
    return payload


def _both_readers(package: DatasetPackage):
    """The package restored through the package v4 AND the assembled v3 full-wire readers."""
    via_package = DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict())))
    wire = json.loads(canonical_json(package.to_assembled().to_full_dict()))
    via_assembled = DatasetPackage.from_assembled(AssembledDataset.from_full_dict(wire))
    return via_package, via_assembled


def _assert_same_projection(package: DatasetPackage, *others: DatasetPackage) -> None:
    identity, features, targets = _digests(package)
    for other in others:
        again, other_features, other_targets = _digests(other)
        assert other_features == features and other_targets == targets
        assert (again.feature_content_fingerprint, again.target_content_fingerprint) == (identity.feature_content_fingerprint, identity.target_content_fingerprint)
        assert again.observation_ids == identity.observation_ids and again.features == identity.features


# --------------------------------------------------------------------------- #
# Public Dataset -> package                                                     #
# --------------------------------------------------------------------------- #
def test_public_f64_package_preserves_ids_partitions_folds_axes_and_bits() -> None:
    package = nio.matrix_dataset_package(_dataset(), "nir")
    assert list(package.partitions) == ["train", "test"], "owner partition order, not alphabetical"
    feature = _feature(package)
    assert feature.matrix.dtype == "float64" and feature.matrix.precision.origin == "source-declared" and feature.matrix.precision.source_dtype == "float64"
    assert feature.matrix.values[0, 1] == 1.0 + 2.0**-30, "the sub-F32-ulp difference survives"
    assert feature.feature_names == ["w1100", "w1102"] and feature.representation_id == "signal_1d"
    assert feature.axes == [{"name": "wavelength", "unit": "nm", "coordinates": [1100.0, 1102.0]}]
    assert package.fold_provenance == [
        {"train_observation_ids": ["s3"], "validation_observation_ids": ["s1", "s2"]},
        {"train_observation_ids": ["s1", "s2"], "validation_observation_ids": ["s3"]},
    ]
    identity, _, _ = _digests(package)
    assert identity.sample_ids == IDS and identity.observation_ids == IDS and identity.target_kinds == ["numeric"]
    audit = package.audits[0]
    assert {"source", "selected_source", "owner_dataset_record", "raw_projection_source", "raw_fold_assignments", "dataset_content_identity", "input_sample_ids", "raw_source_schema", "target_mask", "target_names"} == set(audit)
    assert audit["raw_fold_assignments"] == ["a", "a", "b", None] and audit["input_sample_ids"] == IDS


def test_unnamed_public_sources_stay_none_and_differ_from_named() -> None:
    unnamed = nio.matrix_dataset_package(_dataset(feature_names=None), "nir")
    named = nio.matrix_dataset_package(_dataset(), "nir")
    assert _feature(unnamed).feature_names is None and _feature(unnamed).headers == ["1100.0", "1102.0"], "presentation headers are generated, names stay null"
    assert _digests(unnamed)[0].feature_content_fingerprint != _digests(named)[0].feature_content_fingerprint
    assert _digests(unnamed)[0].features[0]["feature_names"] is None


def test_public_record_dtype_is_the_source_declaration_and_f32_requests_are_honest() -> None:
    f32 = _dataset(x=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32))
    declared = nio.matrix_dataset_package(f32.to_dict(), "nir")  # a JSON-shaped record, not a live Dataset
    block = _feature(declared).matrix
    assert (block.dtype, block.precision.origin, block.precision.source_dtype) == ("float32", "source-declared", "float32")
    widened = nio.matrix_dataset_package_with_storage(f32, "nir", NumericStorageOptions(features="float64"))
    wide = _feature(widened).matrix
    assert (wide.dtype, wide.precision.origin, wide.precision.requested_dtype) == ("float64", "requested", "float64")
    assert wide.values[0, 0] == np.float64(np.float32(0.1)) != 0.1, "normalized source, no restored bits"
    requested = nio.matrix_dataset_package_with_storage(_dataset(), "nir", NumericStorageOptions(features="float32", targets="float32"))
    assert _feature(requested).matrix.values[0, 0] == _feature(requested).matrix.values[0, 1], "an explicit F32 request merges the ULP once"
    assert _digests(requested)[0].feature_content_fingerprint != _digests(nio.matrix_dataset_package(_dataset(), "nir"))[0].feature_content_fingerprint


def test_target_free_prediction_and_classification_integer() -> None:
    predict = nio.matrix_dataset_package(_dataset(y=None, partitions=None, fold_ids=None, groups=None), "nir")
    identity, _, target_preimage = _digests(predict)
    assert identity.targets is None and identity.target_names == [] and target_preimage.endswith(b"\x00")
    ds = _dataset(y=np.array([0, 1, 1, 0], dtype=np.int64), task_type="classification")
    package = nio.matrix_dataset_package(ds, "nir")
    column = _y(package)
    assert isinstance(column, TargetTable) and column.columns[0].kind == "classification_integer"
    assert column.columns[0].block.precision.source_dtype == "int64" and column.columns[0].block.dtype == "float64"
    f32 = nio.matrix_dataset_package_with_storage(_dataset(y=np.array([16777218, 1, 1, 0], dtype=np.int64), task_type="classification"), "nir", NumericStorageOptions(targets="float32"))
    assert _y(f32).columns[0].block.dtype == "float32"
    with pytest.raises(NumericContractError, match="not exact in float32"):
        nio.matrix_dataset_package_with_storage(_dataset(y=np.array([16777217, 1, 1, 0], dtype=np.int64), task_type="classification"), "nir", NumericStorageOptions(targets="float32"))
    regression_ints = _dataset(y=np.array([16777217, 1, 1, 0], dtype=np.int64))
    rounded = nio.matrix_dataset_package_with_storage(regression_ints, "nir", NumericStorageOptions(targets="float32"))
    assert _y(rounded).columns[0].kind == "numeric", "a regression integer source never becomes classification"


def test_dense_matrix_and_masked_factories_and_their_refusals() -> None:
    dense = nio.dense_dataset_package(_dataset(), "nir")
    assert nio.dense_dataset_package_with_storage(_dataset(), "nir").manifest().root == dense.manifest().root == nio.matrix_dataset_package(_dataset(), "nir").manifest().root
    matrix_targets = _dataset(y=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]), target_names=["a", "b"])
    wide = nio.matrix_dataset_package(matrix_targets, "nir")
    assert _digests(wide)[0].target_names == ["a", "b"]
    with pytest.raises(NumericContractError, match="one numeric target"):
        nio.dense_dataset_package(matrix_targets, "nir")
    with pytest.raises(NumericContractError, match="one numeric target"):
        nio.dense_dataset_package(_dataset(y=np.array([0, 1, 1, 0], dtype=np.int64), task_type="classification"), "nir")
    mask = np.array([True, False, True, True])
    masked_ds = _dataset(y=np.array([1.0, 99.0, 3.0, 4.0]), target_mask=mask)
    with pytest.raises(NumericContractError, match="observed targets"):
        nio.matrix_dataset_package(masked_ds, "nir")
    package, projection = nio.masked_matrix_dataset_package(masked_ds, "nir")
    block = _y(package).columns[0].block
    assert block.validity.ravel().tolist() == [True, False, True] and block.values_bytes()[8:16] == b"\x00" * 8, "a withheld label is a false cell, never an observed zero"
    assert set(projection) == {"sample_ids", "target_names", "target_mask", "mask_content_fingerprint"} and len(projection["mask_content_fingerprint"]) == 64
    zero = nio.masked_matrix_dataset_package(_dataset(y=np.array([1.0, 0.0, 3.0, 4.0]), target_mask=mask), "nir")[0]
    assert _digests(package)[0].target_content_fingerprint == _digests(zero)[0].target_content_fingerprint, "a masked cell's stored value is not truth"
    assert _digests(package)[0].target_content_fingerprint != _digests(nio.matrix_dataset_package(_dataset(y=np.array([1.0, 0.0, 3.0, 4.0])), "nir"))[0].target_content_fingerprint


def test_unsupported_public_shapes_and_types_refuse() -> None:
    with pytest.raises(NumericContractError, match="Unknown selected source"):
        nio.matrix_dataset_package(_dataset(), "missing")
    for bad_y in (np.array(["a", "b", "a", "b"]), np.array([True, False, True, False])):
        with pytest.raises(NumericContractError, match="Unsupported public target dtype"):
            nio.matrix_dataset_package(_dataset(y=bad_y), "nir")
    with pytest.raises(NumericContractError, match="int64"):
        nio.matrix_dataset_package(_dataset(y=np.array([0.0, 1.0, 1.0, 0.0]), task_type="classification"), "nir")
    cube = Dataset.from_sources({"img": np.zeros((4, 2, 2, 3))}, sample_ids=IDS, y=np.arange(4.0), representations={"img": "rgb_image"})
    with pytest.raises(NumericContractError, match="rank-2"):
        nio.matrix_dataset_package(cube, "img")
    partial = TensorSource(np.zeros((4, 2)), IDS, representation_id="tabular_numeric", presence_mask=np.array([True, True, False, True]))
    with pytest.raises(NumericContractError, match="complete numeric"):
        nio.matrix_dataset_package(Dataset.from_sources({"t": partial}, sample_ids=IDS, y=np.arange(4.0)), "t")
    with pytest.raises(NumericContractError, match="source dtype"):
        nio.matrix_dataset_package(_dataset(x=np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)), "nir")
    with pytest.raises(NumericContractError, match="Every training row"):
        nio.matrix_dataset_package(_dataset(fold_ids=["a", None, "b", None], groups=["g1", "g2", "g3", "g4"]), "nir")


def test_group_rendering_matches_the_owner_or_refuses() -> None:
    package = nio.matrix_dataset_package(_dataset(groups=[1, 1, 2, 3]), "nir")
    assert list(_meta(package).frame["group_id"]) == ["1", "1", "2"]
    with pytest.raises(NumericContractError, match="JSON rendering"):
        nio.matrix_dataset_package(_dataset(groups=[1e22, 1e22, 2.5, 3.5]), "nir")


def test_primary_ids_are_not_sanitized_by_the_factory() -> None:
    package = nio.matrix_dataset_package(_dataset(sample_ids=["a b", "s2", "s3", "s4"]), "nir")
    assert _meta(package).frame["sample_id"].iloc[0] == "a b"
    with pytest.raises(NumericContractError, match="Invalid primary identity"):
        package.executed_projection(["nir"])


# --------------------------------------------------------------------------- #
# Full-wire roundtrips: equal preimages and digests through BOTH readers        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"feature_names": None},
        {"axis_units": {"nir": {"wavelength": "nanometre"}}},
        {"axis_units": {"nir": {"wavelength": None}}},
        {"axis_coordinates": {"nir": {"wavelength": [-0.0, 1.0]}}},
        {"axis_coordinates": {"nir": {"wavelength": [1100.0, 1100.0 + 2.0**-40]}}},
        {"partitions": ["predict", "test", "train", "train"], "fold_ids": [None, None, "a", "b"], "groups": ["g1", "g2", "g3", "g4"]},
        {"origin_ids": ["o1", "o1", "o2", "o3"]},
        {"x": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)},
        {"y": np.array([1.0, -0.0, 3.0, 4.0])},
    ],
    ids=["base", "unnamed", "raw-nanometre", "raw-unit-null", "signed-zero-coordinate", "coordinate-ulp", "nonalphabetic-partitions", "origins", "source-f32", "target-signed-zero"],
)
def test_public_package_roundtrips_to_identical_preimages_through_both_readers(changes: dict) -> None:
    package = nio.matrix_dataset_package(_dataset(**changes), "nir")
    via_package, via_assembled = _both_readers(package)
    _assert_same_projection(package, via_package, via_assembled)
    assert via_package.manifest().to_dict() == package.manifest().to_dict()
    assert list(via_package.partitions) == list(package.partitions) == list(via_assembled.to_assembled().blocks)


def test_masked_and_requested_f32_roundtrip_through_the_package_wire() -> None:
    masked, _ = nio.masked_matrix_dataset_package(_dataset(y=np.array([1.0, 99.0, 3.0, 4.0]), target_mask=np.array([True, False, True, True])), "nir")
    restored = DatasetPackage.from_full_dict(json.loads(canonical_json(masked.to_full_dict())))
    _assert_same_projection(masked, restored)
    requested = nio.matrix_dataset_package_with_storage(_dataset(), "nir", NumericStorageOptions(features="float32"))
    _assert_same_projection(requested, DatasetPackage.from_full_dict(json.loads(canonical_json(requested.to_full_dict()))))


def test_tabular_names_axes_and_audits_roundtrip(tmp_path) -> None:
    pd.DataFrame({"sid": ["s1", "s2"], "1100": [0.5, 0.25], "1102": [1.5, 2.0], "y": [1.0, 2.0]}).to_csv(tmp_path / "d.csv", sep=";", index=False)
    spec = {
        "sample_index": {"by": "id", "key": "sid"},
        "sources": [{"id": "d", "role": "mixed", "input": "d.csv", "key": "sid", "columns": [{"role": "features", "select": ["1100", "1102"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["sid"]}], "params": {"delimiter": ";", "header_unit": "nm"}}],
    }
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    feature = _feature(package)
    assert feature.feature_names == ["1100", "1102"] and feature.axes == [{"name": "wavelength", "unit": "nm", "coordinates": [1100.0, 1102.0]}]
    no_unit = json.loads(json.dumps(spec))
    del no_unit["sources"][0]["params"]["header_unit"]
    assert _feature(nio.to_dataset_package(no_unit, base_dir=tmp_path)).axes[0]["unit"] is None, "an absent raw unit stays null, not 'text'"
    arrays = nio.to_dataset_package((np.zeros((2, 2)), np.zeros(2)))
    assert _feature(arrays).feature_names is None and _feature(arrays).axes == [{"name": "feature", "unit": None, "coordinates": None}]
    for item in (package, arrays):
        wire = json.loads(canonical_json(item.to_full_dict()))
        assert wire["audits"] == item.audits
        assert DatasetPackage.from_full_dict(wire).audits == item.audits
        assert AssembledDataset.from_full_dict(json.loads(canonical_json(item.to_assembled().to_full_dict()))).audits == item.audits


# --------------------------------------------------------------------------- #
# Strict readers: audits, new metadata, versions                                #
# --------------------------------------------------------------------------- #
def _wires():
    package = nio.matrix_dataset_package(_dataset(), "nir")
    return package, json.loads(canonical_json(package.to_full_dict())), json.loads(canonical_json(package.to_assembled().to_full_dict()))


def test_audits_are_required_exact_lists() -> None:
    _, pkg, asm = _wires()
    assert nio.matrix_dataset_package(_dataset(), "nir").audits, "the public factory supplies owner provenance"
    empty_pkg, empty_asm = {**pkg, "audits": []}, {**asm, "audits": []}
    assert DatasetPackage.from_full_dict(empty_pkg).audits == [] and AssembledDataset.from_full_dict(empty_asm).audits == [], "an empty list is valid and restored as empty"
    for wire, read in ((pkg, DatasetPackage.from_full_dict), (asm, AssembledDataset.from_full_dict)):
        omitted = {k: v for k, v in wire.items() if k != "audits"}
        with pytest.raises(NumericContractError, match="missing or unknown"):
            read(omitted)
        bad: Any
        for bad in (None, {}, "audits", [1], [None], [[]], 3):
            with pytest.raises(NumericContractError, match="audits"):
                read({**wire, "audits": bad})


def test_new_feature_metadata_is_required_aligned_and_typed() -> None:
    _, pkg, asm = _wires()
    value = pkg["partitions"]["train"]["payloads"]["train/x0"]["value"]
    for key in ("feature_names", "axes", "representation_id", "source_id"):
        broken = json.loads(json.dumps(pkg))
        del broken["partitions"]["train"]["payloads"]["train/x0"]["value"][key]
        with pytest.raises(NumericContractError):
            DatasetPackage.from_full_dict(broken)
    bad: Any
    for bad in (["only-one"], [1, 2], "names", {}):
        broken = json.loads(json.dumps(pkg))
        broken["partitions"]["train"]["payloads"]["train/x0"]["value"]["feature_names"] = bad
        with pytest.raises(NumericContractError):
            DatasetPackage.from_full_dict(broken)
    assert value["feature_names"] == ["w1100", "w1102"]
    block = asm["blocks"]["train"]
    for key in ("feature_names", "feature_axes", "representation_ids"):
        omitted = json.loads(json.dumps(asm))
        del omitted["blocks"]["train"][key]
        with pytest.raises(NumericContractError, match="missing or unknown"):
            AssembledDataset.from_full_dict(omitted)
        for bad in ([], [None, None], "x", None):  # noqa: B007
            broken = json.loads(json.dumps(asm))
            broken["blocks"]["train"][key] = bad
            with pytest.raises(NumericContractError):
                AssembledDataset.from_full_dict(broken)
    for key, replacement in (("feature_names", [["a"]]), ("representation_ids", [3]), ("feature_axes", [[{"name": "w", "unit": None, "coordinates": {"kind": "numeric", "count": 1, "data_bytes_hex": "00" * 8}}]])):
        broken = json.loads(json.dumps(asm))
        broken["blocks"]["train"][key] = replacement
        with pytest.raises(NumericContractError):
            AssembledDataset.from_full_dict(broken)
    assert block["feature_names"] == [["w1100", "w1102"]] and block["representation_ids"] == ["signal_1d"]


def test_future_and_retired_versions_have_no_reader_dispatch() -> None:
    _, pkg, asm = _wires()
    for version in (3, 5, 99):
        with pytest.raises(NumericContractError, match="Unsupported package version"):
            DatasetPackage.from_full_dict({**pkg, "schema_version": version})
    for version in (2, 4, 99):
        with pytest.raises(NumericContractError, match="Unsupported assembled version"):
            AssembledDataset.from_full_dict({**asm, "assembled_schema_version": version})


def test_audit_numbers_never_override_the_exact_numeric_authority() -> None:
    package, pkg, asm = _wires()
    for wire in (pkg, asm):
        tampered = json.loads(json.dumps(wire))
        raw = tampered["audits"][0]["raw_projection_source"]
        raw["axis_coordinates"] = {"wavelength": [0.0, 0.0]}  # e.g. JS-normalized -0 / lost ULPs
        raw["axis_units"] = {"wavelength": "tampered"}
        tampered["audits"][0]["raw_source_schema"]["feature_names"] = ["x", "y"]
        restored = DatasetPackage.from_full_dict(tampered) if wire is pkg else DatasetPackage.from_assembled(AssembledDataset.from_full_dict(tampered))
        _assert_same_projection(package, restored)
        assert _feature(restored).axes == _feature(package).axes and _feature(restored).feature_names == ["w1100", "w1102"]


def test_axis_bits_inconsistent_with_the_block_refuse_instead_of_reconstructing() -> None:
    _, pkg, _ = _wires()
    broken = json.loads(json.dumps(pkg))
    axis = broken["partitions"]["train"]["payloads"]["train/x0"]["value"]["axes"][0]
    axis["coordinates"] = {"kind": "numeric", "count": 3, "data_bytes_hex": "00" * 24}
    with pytest.raises(NumericContractError):
        DatasetPackage.from_full_dict(broken)
    flipped = json.loads(json.dumps(pkg))
    flipped["partitions"]["train"]["payloads"]["train/x0"]["value"]["axes"][0]["unit"] = "nanometre"
    # the unit is feature-payload content: the manifest hash no longer matches the bytes
    with pytest.raises(NumericContractError, match="manifest disagrees"):
        DatasetPackage.from_full_dict(flipped)


# --------------------------------------------------------------------------- #
# Whole-dataset identity: labelled wrappers, evidence-only optional link        #
# --------------------------------------------------------------------------- #
def test_labelled_wrappers_delegate_the_unchanged_bytes() -> None:
    ds = _dataset(x=np.array([[-0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    assert labelled_dataset_content_bytes(ds) == dataset_content_bytes(ds.to_dict())
    zero = _dataset(x=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]))
    assert labelled_dataset_content_bytes(ds) == labelled_dataset_content_bytes(zero), "the old zero-folding policy is unchanged"
    assert not labelled_dataset_content_bytes(ds).startswith(b"nirs4all.io.dataset-content")
    identity = labelled_dataset_content_identity(ds)
    assert set(identity) == {"profile", "algorithm", "fingerprint"} and identity["profile"] == "nirs4all.io.dataset-content.v1" and identity["algorithm"] == "sha256"
    import hashlib

    assert identity["fingerprint"] == hashlib.sha256(dataset_content_bytes(ds.to_dict())).hexdigest()
    assert canonical_content_bytes({"a": [-0.0]}) == canonical_content_bytes({"a": [0.0]})


def test_dataset_identity_is_evidence_only_and_fakes_refuse() -> None:
    ds = _dataset()
    package = nio.matrix_dataset_package(ds, "nir")
    identity, _, _ = _digests(package)
    assert identity.dataset_content_identity == labelled_dataset_content_identity(ds) == verified_dataset_identity(package)
    identity.validate(package)
    receipt = identity.to_dict()
    for fake in (
        {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256", "fingerprint": "0" * 64},
        {"profile": "other", "algorithm": "sha256", "fingerprint": "0" * 64},
        {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "md5", "fingerprint": "0" * 64},
        {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256", "fingerprint": "ABC"},
        {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256"},
        "0" * 64,
    ):
        with pytest.raises(NumericContractError):
            ExecutedProjectionIdentityV1.from_dict({**receipt, "dataset_content_identity": fake}).validate(package)
    # a serialized record is a candidate, not a certificate: the full wire carries the original record and
    # the identity is only recovered after an exact owner rebuild; an audit claim alone is never authority
    restored = DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict())))
    assert restored.source_evidence is None and restored.audits[0]["dataset_content_identity"] == identity.dataset_content_identity
    assert _digests(restored)[0].dataset_content_identity == identity.dataset_content_identity
    ExecutedProjectionIdentityV1.from_dict(receipt).validate(restored)
    forged = json.loads(canonical_json(package.to_full_dict()))
    forged["audits"][0]["dataset_content_identity"]["fingerprint"] = "0" * 64
    assert _digests(DatasetPackage.from_full_dict(forged))[0].dataset_content_identity is None, "inconsistent audits decline the proof"
    no_record = json.loads(canonical_json(package.to_full_dict()))
    del no_record["audits"][0]["owner_dataset_record"]
    assert _digests(DatasetPackage.from_full_dict(no_record))[0].dataset_content_identity is None, "absent evidence is None"
    with pytest.raises(NumericContractError, match="disagrees"):
        ExecutedProjectionIdentityV1.from_dict(receipt).validate(DatasetPackage.from_full_dict(no_record))
    # evidence can be re-bound, but only to the record the package really derives from
    rebound = attach_source_record(restored, ds, "nir")
    assert _digests(rebound)[0].dataset_content_identity == identity.dataset_content_identity
    with pytest.raises(NumericContractError, match="does not derive"):
        attach_source_record(restored, _dataset(x=np.zeros((4, 2))), "nir")
    tampered = DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict())))
    import dataclasses

    wrong = dataclasses.replace(tampered, name="tampered", source_evidence=package.source_evidence)
    with pytest.raises(NumericContractError, match="does not derive"):
        wrong.executed_projection(["nir"])


def test_public_object_cap_coordinate_cap_and_class_cap_stay_while_typed_integers_are_generic() -> None:
    typed = Dataset.from_sources({"nir": np.array([[2**53, 1], [1, 1], [1, 1], [1, 1]], dtype=np.int64)}, sample_ids=IDS, y=np.arange(4.0))
    typed_source = typed.multimodal.sources["nir"]
    assert isinstance(typed_source, TensorSource) and typed_source.values[0, 0] == 2**53, "typed int64 is generic numeric data, not capped"
    with pytest.raises(ValueError, match="exactly representable"):
        Dataset.from_sources({"nir": np.array([[2**53, 1], [1, 1], [1, 1], [1, 1]], dtype=object)}, sample_ids=IDS, y=np.arange(4.0), representations={"nir": "tabular_mixed"})
    with pytest.raises(ValueError, match="coordinates"):
        Dataset.from_sources({"nir": np.zeros((4, 2))}, sample_ids=IDS, y=np.arange(4.0), axis_coordinates={"nir": {"wavelength": [2**53, 2**53 + 2]}})
    generic = nio.to_dataset_package((np.array([[2**53, 2**54]], dtype=np.int64), np.array([0.0])))
    assert [float(v) for v in _feature(generic).matrix.values.ravel()] == [float(2**53), float(2**54)]
    package = nio.matrix_dataset_package(typed, "nir")
    assert _feature(package).matrix.values[0, 0] == float(2**53) and _feature(package).matrix.precision.source_dtype == "int64"
    with pytest.raises(NumericContractError, match="portable integer"):
        nio.matrix_dataset_package(_dataset(y=np.array([2**53, 0, 0, 1], dtype=np.int64), task_type="classification"), "nir")


# --------------------------------------------------------------------------- #
# Storage precedence: direct integer -> F32 rounding                            #
# --------------------------------------------------------------------------- #
N = 2**54 + 2**30 + 1  # a hair above the F32 tie: direct rounding goes UP, int -> F64 -> F32 goes DOWN
UPPER, LOWER = 2**54 + 2**31, 2**54


def _ref_f32(value: int) -> int:
    """Independent exact nearest-even rounding of an integer to 24 significant bits."""
    shift = abs(value).bit_length() - 24
    if shift <= 0:
        return value
    quotient, remainder = divmod(abs(value), 1 << shift)
    half = 1 << (shift - 1)
    quotient += int(remainder > half or (remainder == half and quotient & 1))
    return (quotient << shift) * (1 if value > 0 else -1)


def test_reference_rounding_documents_the_regression() -> None:
    assert _ref_f32(N) == UPPER and _ref_f32(int(float(N))) == LOWER


def test_in_memory_integer_features_round_directly_to_f32() -> None:
    package = nio.to_dataset_package((np.array([[N]], dtype=np.int64), np.array([0.0])), features="float32")
    block = _feature(package).matrix
    assert float(block.values[0, 0]) == float(UPPER) and block.dtype == "float32"
    assert (block.precision.origin, block.precision.requested_dtype, block.precision.source_dtype) == ("requested", "float32", "int64")
    assert block.values_bytes() == np.float32(UPPER).tobytes()
    with pytest.raises(Exception, match="cannot represent exactly"):
        nio.to_dataset_package((np.array([[N]], dtype=np.int64), np.array([0.0])))  # default F64 storage refuses the inexact integer
    declared = nio.to_dataset_package((np.array([[float(N)]], dtype=np.float64), np.array([0.0])), features="float32")
    assert float(_feature(declared).matrix.values[0, 0]) == float(LOWER), "a declared F64 source keeps its already-rounded bits: nothing is recovered"
    assert _feature(nio.to_dataset_package((np.array([[float(N)]]), np.array([0.0])))).matrix.values[0, 0] == float(N)


def test_csv_parquet_and_processing_integers_round_directly(tmp_path) -> None:
    (tmp_path / "x.csv").write_text(f"a;y;w\n{N};1;{N}\n", encoding="utf-8")
    (tmp_path / "v.csv").write_text(f"a\n{N}\n", encoding="utf-8")
    spec = {
        "sources": [
            {
                "id": "d",
                "role": "mixed",
                "input": "x.csv",
                "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}],
                "variations": [{"name": "v", "input": "v.csv"}],
                "params": {"delimiter": ";"},
            }
        ]
    }
    package = nio.to_dataset_package(spec, base_dir=tmp_path, features="float32", weights="float32")
    assert float(_feature(package).matrix.values[0, 0]) == float(UPPER)
    assert float(_feature(package).processings[0][1].values[0, 0]) == float(UPPER)
    weights = dict(package.partitions["train"].payloads)["train/weights"]
    assert isinstance(weights, WeightsVector) and float(weights.matrix.values[0, 0]) == float(UPPER)
    for requests in ({}, {"features": "float64"}):
        options: dict[str, Any] = requests
        with pytest.raises(Exception, match="cannot represent exactly"):
            nio.to_dataset_package(spec, base_dir=tmp_path, **options)
    pd.DataFrame({"a": np.array([N], dtype=np.int64), "y": [1.0]}).to_parquet(tmp_path / "p.parquet")
    parquet = {"sources": [{"id": "p", "role": "mixed", "input": "p.parquet", "columns": [{"role": "features", "select": ["a"]}, {"role": "targets", "select": ["y"]}]}]}
    typed = nio.to_dataset_package(parquet, base_dir=tmp_path, features="float32")
    assert float(_feature(typed).matrix.values[0, 0]) == float(UPPER) and _feature(typed).matrix.precision.source_dtype == "int64"


def test_integer_targets_round_directly_but_class_labels_never_round() -> None:
    regression = nio.to_dataset_package((np.zeros((1, 1)), np.array([N], dtype=np.int64)), targets="float32")
    assert float(_y(regression).columns[0].block.values[0, 0]) == float(UPPER)
    from nirs4all_io.materialize.assemble import PartitionBlock

    assembled = AssembledDataset(name="c", task_type="multiclass", signal_type="auto", n_sources=1)
    block = PartitionBlock(n_samples=1, source_ids=["X"], X=[np.zeros((1, 1))], feature_headers=[["0"]], header_units=["index"], signal_types=[None])
    block.y, block.y_headers, block.y_source_dtype = np.array([[N]], dtype=np.int64), ["y"], "int64"
    assembled.blocks["train"] = block
    with pytest.raises(NumericContractError):
        DatasetPackage.from_assembled(assembled, targets="float32")  # beyond the portable class cap: never merged
    with pytest.raises(NumericContractError):
        DatasetPackage.from_assembled(assembled)


def test_spectrodataset_consumer_converts_integer_features_directly() -> None:
    assembled = nio.load((np.array([[N]], dtype=np.int64), np.array([0.0])), target="assembled", features="float32")
    assert assembled.blocks["train"].X[0][0, 0] == float(UPPER)

    class _Double:
        def __init__(self, name="d"):
            self.samples: list[Any] = []

        def add_samples(self, data, indexes, headers=None, header_unit=None):
            self.samples.append(data)

        def add_targets(self, y): ...
        def add_metadata(self, *a, **k): ...
        def set_signal_type(self, *a, **k): ...
        def set_task_type(self, *a, **k): ...

    ds = nio.load((np.array([[N]], dtype=np.int64), np.array([0.0])), spectro_dataset_cls=_Double)  # the F32 consumer request reaches coercion
    assert float(ds.samples[0][0, 0]) == float(UPPER)


def test_metadata_labels_and_fold_counts_must_agree_with_their_values() -> None:
    _, pkg, _ = _wires()
    broken = json.loads(json.dumps(pkg))
    broken["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["dtype"] = "numeric"
    with pytest.raises(NumericContractError, match="dtype label"):
        DatasetPackage.from_full_dict(broken)
    broken = json.loads(json.dumps(pkg))
    broken["folds"][0][0] = broken["folds"][0][0][:-1]
    with pytest.raises(NumericContractError, match="fold_provenance"):
        DatasetPackage.from_full_dict(broken)


# --------------------------------------------------------------------------- #
# Selected precision survives the F64 IR (no re-selection, no lost P)           #
# --------------------------------------------------------------------------- #
def _selected_spec(tmp_path, *, string_target: bool = False, task: str = "auto") -> dict[str, Any]:
    ulp = 1.0 + 2.0**-30
    frame = pd.DataFrame(
        {
            "sid": ["s1", "s2", "s3"],
            "a": [ulp, -0.0, 0.5],
            "b": [np.nan, 2.0, 3.0],
            "y": ["red", "blue", "red"] if string_target else [ulp, 0.0, np.nan],
            "w": [ulp, 2.0, np.nan],
        }
    )
    frame.to_csv(tmp_path / "x.csv", sep=";", index=False)
    pd.DataFrame({"a": [ulp + 2.0**-29, 1.0, -0.0], "b": [4.0, 5.0, 6.0]}).to_csv(tmp_path / "v.csv", sep=";", index=False)
    return {
        "task_type": task,
        "params": {"na": {"policy": "ignore"}},
        "sample_index": {"by": "id", "key": "sid"},
        "sources": [
            {
                "id": "d",
                "role": "mixed",
                "input": "x.csv",
                "key": "sid",
                "columns": [
                    {"role": "features", "select": ["a", "b"]},
                    {"role": "targets", "select": ["y"]},
                    {"role": "weights", "select": ["w"]},
                    {"role": "metadata", "select": ["sid"]},
                ],
                "variations": [{"name": "v", "input": "v.csv"}],
                "params": {"delimiter": ";", "header_unit": "nm"},
            }
        ],
    }


def _records(package: DatasetPackage) -> dict[str, Any]:
    """Every role's selected P (features, processings, targets, weights) of the train block."""
    payloads = dict(package.partitions["train"].payloads)
    feature, targets, weights = payloads["train/x0"], payloads["train/y"], payloads["train/weights"]
    assert isinstance(feature, FeatureMatrix) and isinstance(targets, TargetTable) and isinstance(weights, WeightsVector)
    return {
        "features": feature.matrix.precision.to_dict(),
        "processings": [block.precision.to_dict() for _, block in feature.processings],
        "targets": [column.block.precision.to_dict() for column in targets.columns],
        "weights": weights.matrix.precision.to_dict(),
    }


_SELECTIONS: dict[str, dict[str, Any]] = {
    "requested-f32": {"features": "float32", "targets": "float32", "weights": "float32"},
    "requested-f64": {"features": "float64", "targets": "float64", "weights": "float64"},
    "consumer-f32": {"consumer": "SpectroDataset"},
    "default": {},
    "mixed-roles": {"features": "float32", "weights": "float64"},
}


@pytest.mark.parametrize("selection", list(_SELECTIONS), ids=list(_SELECTIONS))
def test_every_roundtrip_retains_the_exact_selected_precision_bytes_and_root(tmp_path, selection) -> None:
    package = nio.to_dataset_package(_selected_spec(tmp_path), base_dir=tmp_path, **_SELECTIONS[selection])
    records = _records(package)
    if selection == "requested-f32":
        assert records["features"] == {"dtype": "float32", "origin": "requested", "source_dtype": None, "requested_dtype": "float32", "consumer": None}
        assert records["processings"] == [records["features"]] and records["weights"]["dtype"] == "float32"
    expected = (package.manifest().to_dict(), _digests(package, "d")[1:], records)
    block = package.to_assembled().blocks["train"]
    assert block.x_precisions and block.y_precisions and block.weights_precision is not None, "the selected P travels beside the F64 IR"
    via_assembled_ir = DatasetPackage.from_assembled(package.to_assembled())
    via_package_wire = DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict())))
    via_assembled_wire = DatasetPackage.from_assembled(AssembledDataset.from_full_dict(json.loads(canonical_json(package.to_assembled().to_full_dict()))))
    twice = DatasetPackage.from_assembled(via_assembled_wire.to_assembled())
    for restored in (via_assembled_ir, via_package_wire, via_assembled_wire, twice):
        assert (restored.manifest().to_dict(), _digests(restored, "d")[1:], _records(restored)) == expected
        assert restored.manifest().root == package.manifest().root
    # the F32-selected values really are the once-rounded ones (never approximate comparison)
    feature = _feature(package)
    assert feature.matrix.values.tobytes() == _feature(via_assembled_wire).matrix.values.tobytes()


def test_classification_and_string_targets_keep_their_selected_records(tmp_path) -> None:
    for string_target, task in ((True, "auto"), (False, "binary")):
        spec = _selected_spec(tmp_path, string_target=string_target, task=task)
        if not string_target:
            pd.read_csv(tmp_path / "x.csv", sep=";").assign(y=[1, 0, 1]).to_csv(tmp_path / "x.csv", sep=";", index=False)
        extra: dict[str, Any] = {} if string_target else {"targets": "float32"}
        package = nio.to_dataset_package(spec, base_dir=tmp_path, features="float32", weights="float32", **extra)
        for restored in _both_readers(package):
            assert restored.manifest().root == package.manifest().root
            assert _records(restored) == _records(package)
        kind = dict(package.partitions["train"].payloads)["train/y"].columns[0].kind  # type: ignore[union-attr]
        assert kind == ("classification_string" if string_target else "classification_integer")


def test_masks_signed_zero_and_ulps_survive_selected_f32(tmp_path) -> None:
    package = nio.to_dataset_package(_selected_spec(tmp_path), base_dir=tmp_path, features="float32", weights="float32")
    feature = _feature(package)
    assert feature.matrix.validity.ravel().tolist() == [True, False, True, True, True, True]
    assert feature.matrix.values_bytes()[4:8] == b"\x00\x00\x00\x00" and feature.matrix.values_bytes()[8:12] == b"\x00\x00\x00\x80"
    assert feature.matrix.values[0, 0] == np.float32(1.0), "the F64 ULP merged once, at selection"
    restored = DatasetPackage.from_assembled(package.to_assembled())
    assert _feature(restored).matrix.values_bytes() == feature.matrix.values_bytes()
    assert _feature(restored).matrix.validity.tolist() == feature.matrix.validity.tolist()


def test_conflicting_reselection_of_normalized_data_refuses_and_honest_ones_work(tmp_path) -> None:
    requested = nio.to_dataset_package((np.array([[1.0 + 2.0**-30]]), np.array([0.0])), features="float32")
    with pytest.raises(NumericContractError, match="Cannot re-select float64"):
        DatasetPackage.from_assembled(requested.to_assembled(), features="float64")
    assert DatasetPackage.from_assembled(requested.to_assembled(), features="float32").manifest().root == requested.manifest().root
    declared = nio.to_dataset_package((np.array([[0.1]], dtype=np.float32), np.array([0.0])))
    widened = DatasetPackage.from_assembled(declared.to_assembled(), features="float64")  # declared F32 source: widen the normalized source
    assert _feature(widened).matrix.precision.to_dict() == {"dtype": "float64", "origin": "requested", "source_dtype": "float32", "requested_dtype": "float64", "consumer": None}
    exact = nio.to_dataset_package((np.array([[1.0 + 2.0**-30]]), np.array([0.0])))
    narrowed = DatasetPackage.from_assembled(exact.to_assembled(), features="float32")
    assert _feature(narrowed).matrix.precision.origin == "requested" and _feature(narrowed).matrix.values[0, 0] == np.float32(1.0)
    assembled = requested.to_assembled()
    assembled.blocks["train"].X[0][0, 0] = 1.0 + 2.0**-30  # the IR was edited after its storage was selected
    with pytest.raises(NumericContractError, match="no longer matches its selected storage"):
        DatasetPackage.from_assembled(assembled)


def test_processing_and_weight_records_must_match_in_the_readers(tmp_path) -> None:
    package = nio.to_dataset_package(_selected_spec(tmp_path), base_dir=tmp_path, features="float32")
    wire = json.loads(canonical_json(package.to_full_dict()))
    processing = wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["processings"][0]["matrix"]
    assert processing["precision"] == wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"]["precision"]
    processing["precision"] = {"dtype": "float64", "origin": "default", "source_dtype": None, "requested_dtype": None, "consumer": None}
    processing["dtype"] = "float64"
    processing["data_bytes_hex"] = np.zeros(6, dtype="<f8").tobytes().hex()
    with pytest.raises(NumericContractError, match="common features request"):
        DatasetPackage.from_full_dict(wire)
    asm = json.loads(canonical_json(package.to_assembled().to_full_dict()))
    asm_processing = asm["blocks"]["train"]["processings"][0][0]["matrix"]
    asm_processing.update(dtype="float64", precision=processing["precision"], data_bytes_hex=processing["data_bytes_hex"])
    with pytest.raises(NumericContractError, match="common features request"):
        AssembledDataset.from_full_dict(asm)


@pytest.mark.parametrize("selection", ["requested-f32", "default", "consumer-f32"])
def test_public_package_selected_precision_roundtrips_through_the_ir(selection) -> None:
    options = NumericStorageOptions(**_SELECTIONS[selection])
    for x in (np.array([[1.0, 1.0 + 2.0**-30], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]), np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)):
        package = nio.matrix_dataset_package_with_storage(_dataset(x=x), "nir", options)
        again = DatasetPackage.from_assembled(package.to_assembled())
        assert again.manifest().to_dict() == package.manifest().to_dict()
        assert _feature(again).matrix.precision == _feature(package).matrix.precision
        _assert_same_projection(package, again, *_both_readers(package))
        assert again.manifest().root == package.manifest().root


# --------------------------------------------------------------------------- #
# Independently typed processings keep their own source provenance              #
# --------------------------------------------------------------------------- #
ULP = 1.0 + 2.0**-30


def _typed_pair(tmp_path, base: str | None, child: str | None) -> dict[str, Any]:
    """A feature source and one variation loaded from independently typed .npy / CSV files."""
    base_values = np.array([[ULP, -0.0, np.nan], [2.0, 3.0, 4.0]])
    child_values = np.array([[ULP + 2.0**-29, 5.0, -0.0], [6.0, np.nan, 7.0]])
    for stem, values, dtype in (("x", base_values, base), ("v", child_values, child)):
        if dtype is None:
            pd.DataFrame(values, columns=["0", "1", "2"]).to_csv(tmp_path / f"{stem}.csv", sep=";", index=False)
        else:
            np.save(tmp_path / f"{stem}.npy", (np.nan_to_num(values) if np.dtype(dtype).kind == "i" else values).astype(dtype))
    suffix = {"x": "csv" if base is None else "npy", "v": "csv" if child is None else "npy"}
    return {
        "params": {"na": {"policy": "ignore"}},
        "sources": [
            {
                "id": "x",
                "role": "features",
                "input": f"x.{suffix['x']}",
                "variations": [{"name": "v", "input": f"v.{suffix['v']}"}],
                "params": {"header_unit": "index", "delimiter": ";"},
            }
        ],
    }


def _provenance(package: DatasetPackage) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature = _feature(package)
    return feature.matrix.precision.to_dict(), [block.precision.to_dict() for _, block in feature.processings]


@pytest.mark.parametrize(("base", "child"), [("float32", "float64"), ("float64", "float32")])
def test_conflicting_default_storage_of_independent_children_refuses(tmp_path, base, child) -> None:
    spec = _typed_pair(tmp_path, base, child)
    with pytest.raises(NumericContractError, match="common features request"):
        nio.to_dataset_package(spec, base_dir=tmp_path)
    with pytest.raises(NumericContractError, match="common features request"):
        DatasetPackage.from_assembled(nio.load(spec, base_dir=tmp_path, target="assembled"))


def test_explicit_f32_resolves_each_input_with_its_own_source_dtype(tmp_path) -> None:
    for base, child, base_source, child_source in (("float32", "float64", "float32", "float64"), ("float64", "float32", "float64", "float32"), (None, "float64", None, "float64")):
        package = nio.to_dataset_package(_typed_pair(tmp_path, base, child), base_dir=tmp_path, features="float32")
        feature_p, child_p = _provenance(package)
        assert feature_p == {"dtype": "float32", "origin": "requested", "source_dtype": base_source, "requested_dtype": "float32", "consumer": None}
        assert child_p == [{"dtype": "float32", "origin": "requested", "source_dtype": child_source, "requested_dtype": "float32", "consumer": None}]
        feature = _feature(package)
        assert feature.matrix.validity.ravel().tolist() == [True, True, False, True, True, True]
        assert feature.matrix.values[0, 0] == np.float32(1.0), "the F64 ULP merged once at selection"
        child_block = feature.processings[0][1]
        assert child_block.validity.ravel().tolist() == [True, True, True, True, False, True] and child_block.values_bytes()[8:12] == b"\x00\x00\x00\x80", "signed zero and the false mask survive"


def test_explicit_f64_widens_a_declared_f32_input_and_preserves_f64_ulps(tmp_path) -> None:
    package = nio.to_dataset_package(_typed_pair(tmp_path, "float32", "float64"), base_dir=tmp_path, features="float64")
    feature_p, child_p = _provenance(package)
    assert feature_p == {"dtype": "float64", "origin": "requested", "source_dtype": "float32", "requested_dtype": "float64", "consumer": None}
    assert child_p == [{"dtype": "float64", "origin": "requested", "source_dtype": "float64", "requested_dtype": "float64", "consumer": None}]
    assert _feature(package).matrix.values[0, 0] == np.float64(np.float32(ULP)), "widened normalized F32 source: no restored bits"
    assert _feature(package).processings[0][1].values[0, 0] == ULP + 2.0**-29, "a declared F64 child keeps its ULPs"


def test_same_storage_defaults_keep_their_own_truthful_origins(tmp_path) -> None:
    for base, child, base_expected, child_expected in (
        ("float64", None, ("source-declared", "float64"), ("default", None)),
        (None, "float64", ("default", None), ("source-declared", "float64")),
        ("float32", "float32", ("source-declared", "float32"), ("source-declared", "float32")),
        ("float64", "int64", ("source-declared", "float64"), ("default", "int64")),
    ):
        package = nio.to_dataset_package(_typed_pair(tmp_path, base, child), base_dir=tmp_path)
        feature_p, child_p = _provenance(package)
        assert (feature_p["origin"], feature_p["source_dtype"]) == base_expected
        assert (child_p[0]["origin"], child_p[0]["source_dtype"]) == child_expected and child_p[0]["dtype"] == feature_p["dtype"]


def test_independent_child_provenance_roundtrips_exactly_through_every_path(tmp_path) -> None:
    from nirs4all_io.materialize.package import _feature_content_bytes

    cases: tuple[tuple[tuple[str | None, str | None], dict[str, Any]], ...] = (
        (("float32", "float64"), {"features": "float32"}),
        (("float64", None), {}),
        ((None, "float64"), {}),
        (("float32", "float64"), {"features": "float64"}),
        (("float32", "float64"), {"consumer": "SpectroDataset"}),
    )
    for (base, child), options in cases:
        package = nio.to_dataset_package(_typed_pair(tmp_path, base, child), base_dir=tmp_path, **options)
        expected = (package.manifest().to_dict(), _feature_content_bytes(_feature(package)), _provenance(package), package.to_full_dict())
        restored = (
            DatasetPackage.from_assembled(package.to_assembled()),
            DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict()))),
            *_both_readers(package),
        )
        for again in restored:
            assert (again.manifest().to_dict(), _feature_content_bytes(_feature(again)), _provenance(again), again.to_full_dict()) == expected
        carried = package.to_assembled().blocks["train"].processing_precisions
        assert [[p.source_dtype for p in row] for row in carried] == [[_provenance(package)[1][0]["source_dtype"]]]
        # intentional H4 limitation: the frame hashes the base P and each child N only, so the child's
        # source metadata lives in its full-wire ExactBlock.P (validated and preserved), not in the payload hash.


def test_child_with_a_different_request_or_consumer_is_refused_by_the_readers(tmp_path) -> None:
    package = nio.to_dataset_package(_typed_pair(tmp_path, None, None), base_dir=tmp_path, features="float32")
    wire = json.loads(canonical_json(package.to_full_dict()))
    child = wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["processings"][0]["matrix"]
    child["precision"]["requested_dtype"] = None
    child["precision"]["origin"] = "source-declared"
    child["precision"]["source_dtype"] = "float32"
    with pytest.raises(NumericContractError, match="common features request"):
        DatasetPackage.from_full_dict(wire)
    wire = json.loads(canonical_json(package.to_full_dict()))
    child = wire["partitions"]["train"]["payloads"]["train/x0"]["value"]["processings"][0]["matrix"]
    child["precision"] = {"dtype": "float32", "origin": "consumer", "source_dtype": None, "requested_dtype": None, "consumer": "SpectroDataset"}
    with pytest.raises(NumericContractError, match="common features request"):
        DatasetPackage.from_full_dict(wire)


def test_consumer_records_each_processing_own_source_dtype(tmp_path) -> None:
    # No request: an explicit features request is now committed into the assembled IR (and then rightly stays a
    # "requested" origin at the consumer boundary), so the consumer-origin receipt starts from the raw assembly.
    assembled = nio.load(_typed_pair(tmp_path, "float32", "float64"), base_dir=tmp_path, target="assembled")
    from nirs4all_io.materialize.spectrodataset import project_spectrodataset

    class _Recorder:
        def __init__(self, name="d"):
            self.features: list[Any] = []

        def add_samples(self, data, indexes, headers=None, header_unit=None): ...
        def add_features(self, features, processings, source=-1):
            self.features.append(np.asarray(features[0]))

        def add_targets(self, y): ...
        def add_metadata(self, *a, **k): ...
        def set_signal_type(self, *a, **k): ...
        def set_task_type(self, *a, **k): ...

    ds, receipt = project_spectrodataset(assembled, spectro_dataset_cls=_Recorder)
    assert receipt["features"][0]["precision"] == {"dtype": "float32", "origin": "consumer", "source_dtype": "float32", "requested_dtype": None, "consumer": "SpectroDataset"}
    assert receipt["processings"][0]["precision"] == {"dtype": "float32", "origin": "consumer", "source_dtype": "float64", "requested_dtype": None, "consumer": "SpectroDataset"}
    assert ds.features[0].dtype == np.float32
    # the same independently typed inputs through the default load(): the consumer is the common F32 request
    default = nio.load(_typed_pair(tmp_path, "float32", "float64"), base_dir=tmp_path, spectro_dataset_cls=_Recorder)
    assert default.features[0].dtype == np.float32 and default.features[0][0, 0] == np.float32(1.0 + 2.0**-29)


# --------------------------------------------------------------------------- #
# Cross-owner interop: the recorded DEV public inputs and target metadata       #
# --------------------------------------------------------------------------- #
def _dev_record(**overrides: Any) -> dict[str, Any]:
    """The public record of the controller's DEV parity case (feature names null, +-0 / ULP axes, folds, groups)."""
    record: dict[str, Any] = {
        "schema": "nirs4all.dataset.v1",
        "schema_version": 1,
        "dataset": {
            "schema": "nirs4all.multimodal-dataset",
            "schema_version": 1,
            "name": "multimodal",
            "sample_ids": ["s0", "s1", "s2", "s3"],
            "source_alignment": "strict",
            "sources": [
                {
                    "name": "nir",
                    "sample_ids": ["s0", "s1", "s2", "s3"],
                    "representation_id": "signal_1d",
                    "axes": ["sample", "wavelength"],
                    "feature_names": None,
                    "axis_units": {"wavelength": "nm"},
                    "axis_coordinates": {"wavelength": [-0.0, 1.0000000000009095]},
                    "array": {"dtype": "float64", "shape": [4, 2], "values": [[1.0, 1.0000000009313226], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]},
                    "presence_mask": {"dtype": "bool", "shape": [4], "values": [True, True, True, True]},
                }
            ],
            "y": {"dtype": "float64", "shape": [4], "values": [1.0000000009313226, 2.0, 3.0, 4.0]},
            "target_names": ["y"],
            "target_mask": {"dtype": "bool", "shape": [4], "values": [True, True, True, True]},
            "task_type": "regression",
            "groups": {"dtype": "<U2", "shape": [4], "values": ["g0", "g0", "g1", "g1"]},
            "partitions": {"dtype": "<U5", "shape": [4], "values": ["train", "train", "train", "train"]},
        },
        "origin_ids": ["s0", "s1", "s2", "s3"],
        "fold_ids": ["f0", "f0", "f1", "f1"],
    }
    record["dataset"].update(overrides)
    return record


def test_dev_public_record_carries_the_complete_original_record_and_cohort_independent_schema() -> None:
    record = _dev_record()
    package = nio.matrix_dataset_package(record, "nir")
    audit = package.audits[0]
    assert audit["owner_dataset_record"] == Dataset.from_dict(record).to_dict()
    source = audit["raw_projection_source"]
    assert set(source) == {"name", "sample_ids", "representation_id", "axes", "feature_names", "axis_units", "axis_coordinates", "array", "presence_mask"}
    assert source["sample_ids"] == ["s0", "s1", "s2", "s3"] and source["array"]["shape"] == [4, 2] and source["array"]["dtype"] == "float64"
    assert source["presence_mask"]["values"] == [True] * 4 and source["feature_names"] is None
    assert json.dumps(source["axis_coordinates"]) == '{"wavelength": [-0.0, 1.0000000000009095]}'
    assert audit["raw_source_schema"] == {
        "name": "nir", "representation_id": "signal_1d", "axes": ["sample", "wavelength"], "shape": [None, 2], "dtype": "float64",
        "feature_names": None, "axis_units": {"wavelength": "nm"}, "axis_coordinates": {"wavelength": [-0.0, 1.0000000000009095]},
    }  # fmt: skip
    feature = _feature(package)
    assert feature.feature_names is None and feature.axes[0]["unit"] == "nm"
    assert np.asarray(feature.axes[0]["coordinates"], dtype="<f8").tobytes() == np.array([-0.0, 1.0000000000009095], dtype="<f8").tobytes()
    identity, _, _ = _digests(package)
    assert identity.sample_ids == ["s0", "s1", "s2", "s3"] and package.fold_provenance[0]["validation_observation_ids"] == ["s0", "s1"]
    assert dict(package.partitions["train"].payloads)["train/metadata"].frame["group_id"].tolist() == ["g0", "g0", "g1", "g1"]  # type: ignore[union-attr]
    assert identity.dataset_content_identity == labelled_dataset_content_identity(record)


def test_serialized_record_is_a_candidate_proof_by_exact_rebuild_only() -> None:
    record = _dev_record()
    package = nio.matrix_dataset_package(record, "nir")
    expected = labelled_dataset_content_identity(record)
    for restored in _both_readers(package):
        _assert_same_projection(package, restored)
        assert _digests(restored)[0].dataset_content_identity == expected
    wire = json.loads(canonical_json(package.to_full_dict()))

    def decline(mutate) -> None:
        broken = json.loads(json.dumps(wire))
        mutate(broken["audits"][0])
        restored = DatasetPackage.from_full_dict(broken)
        assert _digests(restored)[0].dataset_content_identity is None
        _assert_same_projection(package, restored)  # the exact typed data/preimages never change

    coordinates = lambda audit: audit["owner_dataset_record"]["dataset"]["sources"][0]  # noqa: E731
    decline(lambda a: coordinates(a)["axis_coordinates"].__setitem__("wavelength", [0.0, 1.0000000000009095]))  # JS-normalized -0
    decline(lambda a: coordinates(a)["axis_coordinates"].__setitem__("wavelength", [-0.0, 1.0000000000009097]))
    decline(lambda a: coordinates(a).__setitem__("name", "other"))
    decline(lambda a: coordinates(a)["array"]["values"][1].__setitem__(0, 2.5))
    decline(lambda a: coordinates(a)["axis_units"].__setitem__("wavelength", "nanometre"))
    decline(lambda a: coordinates(a).__setitem__("feature_names", ["a", "b"]))
    decline(lambda a: a["owner_dataset_record"]["dataset"].__setitem__("sample_ids", ["s1", "s0", "s2", "s3"]))
    decline(lambda a: a["owner_dataset_record"].__setitem__("fold_ids", ["f0", "f1", "f1", "f0"]))
    decline(lambda a: a["owner_dataset_record"]["dataset"]["y"]["values"].__setitem__(0, 9.0))
    decline(lambda a: a.__setitem__("dataset_content_identity", {"profile": "other", "algorithm": "sha256", "fingerprint": "0" * 64}))
    decline(lambda a: a["raw_projection_source"]["axis_coordinates"].__setitem__("wavelength", [0.0, 1.0000000000009095]))  # an audit-only edit: strict equality declines
    decline(lambda a: a.__setitem__("owner_dataset_record", "not a record"))
    decline(lambda a: a.__setitem__("selected_source", 3))
    assert _digests(DatasetPackage.from_full_dict(json.loads(json.dumps(wire))))[0].dataset_content_identity == expected


def test_masked_false_cell_keeps_the_original_record_identity_while_storage_is_canonical_zero() -> None:
    withheld = _dev_record(
        y={"dtype": "float64", "shape": [4], "values": [1.0, 99.0, 3.0, 4.0]},
        target_mask={"dtype": "bool", "shape": [4], "values": [True, False, True, True]},
    )
    zero = _dev_record(
        y={"dtype": "float64", "shape": [4], "values": [1.0, 0.0, 3.0, 4.0]},
        target_mask={"dtype": "bool", "shape": [4], "values": [True, False, True, True]},
    )
    first, _ = nio.masked_matrix_dataset_package(withheld, "nir")
    second, _ = nio.masked_matrix_dataset_package(zero, "nir")
    a, b = _digests(first)[0], _digests(second)[0]
    assert a.target_content_fingerprint == b.target_content_fingerprint, "the new N stores a false cell as +0 whatever the withheld value"
    assert a.dataset_content_identity == labelled_dataset_content_identity(withheld) != b.dataset_content_identity == labelled_dataset_content_identity(zero)
    assert dict(first.partitions["train"].payloads)["train/y"].columns[0].block.values_bytes()[8:16] == b"\x00" * 8  # type: ignore[union-attr]
    for restored in _both_readers(first):
        assert _digests(restored)[0].dataset_content_identity == labelled_dataset_content_identity(withheld)


def test_integer_classification_manifest_representation_is_categorical() -> None:
    classification = _dev_record(
        y={"dtype": "int64", "shape": [4], "values": [16777218, 0, 16777218, 0]},
        task_type="classification",
    )
    regression = _dev_record(y={"dtype": "int64", "shape": [4], "values": [16777218, 0, 16777218, 0]})
    options = NumericStorageOptions(features="float32", targets="float32")
    cls_package = nio.matrix_dataset_package_with_storage(classification, "nir", options)
    reg_package = nio.matrix_dataset_package_with_storage(regression, "nir", options)
    entry = {e.id: e for e in cls_package.manifest().entries}["train/y"]
    assert entry.representation_id == "target_categorical" and dict(cls_package.partitions["train"].payloads)["train/y"].representation_id == "target_categorical"  # type: ignore[union-attr]
    assert {e.id: e for e in reg_package.manifest().entries}["train/y"].representation_id == "target_numeric"
    cls_y = dict(cls_package.partitions["train"].payloads)["train/y"].columns[0]  # type: ignore[union-attr]
    reg_y = dict(reg_package.partitions["train"].payloads)["train/y"].columns[0]  # type: ignore[union-attr]
    assert (cls_y.kind, reg_y.kind) == ("classification_integer", "numeric")
    assert cls_y.block.to_bytes() == reg_y.block.to_bytes(), "identical label bytes: only the kind and the manifest metadata differ"
    # manual / tabular flags: classification, binary and multiclass are categorical; same values as regression are not
    for task, expected in (("classification", "target_categorical"), ("binary", "target_categorical"), ("multiclass", "target_categorical"), ("regression", "target_numeric"), ("auto", "target_numeric")):
        assembled = _classification([1, 0, 1], task=task)
        manifest_entry = {e.id: e for e in DatasetPackage.from_assembled(assembled).manifest().entries}["train/y"]
        assert manifest_entry.representation_id == expected, task
    assert {e.id: e for e in nio.to_dataset_package((np.zeros((3, 1)), np.array([1, 0, 1], dtype=np.int64))).manifest().entries}["train/y"].representation_id == "target_numeric"


def _classification(labels: list[int], task: str) -> AssembledDataset:
    from nirs4all_io.materialize.assemble import PartitionBlock

    assembled = AssembledDataset(name="c", task_type=task, signal_type="auto", n_sources=1)
    block = PartitionBlock(n_samples=len(labels), source_ids=["X"], X=[np.zeros((len(labels), 1))], feature_headers=[["0"]], header_units=["index"], signal_types=[None])
    block.y, block.y_headers, block.y_source_dtype = np.array(labels, dtype=np.int64).reshape(-1, 1), ["y"], "int64"
    assembled.blocks["train"] = block
    return assembled


# --------------------------------------------------------------------------- #
# Provenance snapshot of the ACTUAL input (before Dataset normalization)        #
# --------------------------------------------------------------------------- #
def _declared_f32_raw_record() -> dict[str, Any]:
    """A serialized record that declares float32 but carries raw 1+2**-30 JSON numbers (no F32 array is built first)."""
    record = _dev_record()
    source = record["dataset"]["sources"][0]
    source["array"]["dtype"] = "float32"
    assert source["array"]["values"][0][1] == 1.0000000009313226
    return record


def test_declared_f32_raw_json_is_retained_as_provenance_but_stored_normalized() -> None:
    import copy

    record = _declared_f32_raw_record()
    pristine = copy.deepcopy(record)
    package = nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(features="float64"))
    audit = package.audits[0]
    assert audit["owner_dataset_record"] == pristine, "the supplied snapshot, raw scalar values included"
    assert audit["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][1] == 1.0000000009313226
    normalized = Dataset.from_dict(pristine).to_dict()
    assert normalized["dataset"]["sources"][0]["array"]["values"][0][1] == 1.0, "Dataset normalization owns the F32 value"
    assert audit["raw_projection_source"]["array"]["values"][0][1] == 1.0, "the selected source record derives from the normalized source"
    block = _feature(package).matrix
    assert block.values[0, 1] == 1.0 and block.dtype == "float64"
    assert block.precision.to_dict() == {"dtype": "float64", "origin": "requested", "source_dtype": "float32", "requested_dtype": "float64", "consumer": None}
    # the old semantic identity is the old normalizer over the original record == over the normalized record
    expected = labelled_dataset_content_identity(pristine)
    assert expected == labelled_dataset_content_identity(normalized) == audit["dataset_content_identity"]
    assert _digests(package)[0].dataset_content_identity == expected
    # a caller mutation after the factory changes neither the stored witness nor the buffers
    record["dataset"]["sources"][0]["array"]["values"][0][1] = 123.0
    record["dataset"]["y"]["values"][0] = -5.0
    assert package.audits[0]["owner_dataset_record"] == pristine and _feature(package).matrix.values[0, 1] == 1.0
    assert _digests(package)[0].dataset_content_identity == expected


def test_declared_f32_snapshot_survives_both_readers_and_rebuilds_the_proof() -> None:
    record = _declared_f32_raw_record()
    package = nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(features="float64"))
    expected = labelled_dataset_content_identity(record)
    for restored in _both_readers(package):
        _assert_same_projection(package, restored)
        assert restored.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][1] == 1.0000000009313226
        assert _digests(restored)[0].dataset_content_identity == expected
        assert _feature(restored).matrix.values[0, 1] == 1.0
    wire = json.loads(canonical_json(package.to_full_dict()))

    def declined(mutate) -> None:
        broken = json.loads(json.dumps(wire))
        mutate(broken["audits"][0]["owner_dataset_record"]["dataset"])
        restored = DatasetPackage.from_full_dict(broken)
        assert _digests(restored)[0].dataset_content_identity is None
        assert _feature(restored).matrix.values_bytes() == _feature(package).matrix.values_bytes(), "the witness never overwrites numeric fields"

    source = lambda d: d["sources"][0]  # noqa: E731
    declined(lambda d: source(d)["array"].__setitem__("dtype", "float64"))  # a witness that declares another dtype
    declined(lambda d: source(d)["array"]["values"][0].__setitem__(1, 2.5))
    declined(lambda d: source(d)["presence_mask"]["values"].__setitem__(0, False))
    declined(lambda d: source(d)["axis_coordinates"].__setitem__("wavelength", [0.0, 1.0000000000009095]))
    declined(lambda d: source(d)["axis_units"].__setitem__("wavelength", "cm-1"))
    declined(lambda d: d["target_mask"]["values"].__setitem__(1, False))
    declined(lambda d: d.__setitem__("partitions", {"dtype": "<U5", "shape": [4], "values": ["train", "train", "train", "test"]}))
    tampered = json.loads(json.dumps(wire))
    tampered["audits"][0]["dataset_content_identity"]["fingerprint"] = "0" * 64
    assert _digests(DatasetPackage.from_full_dict(tampered))[0].dataset_content_identity is None
    # the raw-snapshot extra precision never becomes numeric authority: re-binding a normalized record to the same package is the same proof
    rebound = attach_source_record(DatasetPackage.from_full_dict(wire), record, "nir", NumericStorageOptions(features="float64"))
    assert _digests(rebound)[0].dataset_content_identity == expected


def test_caller_dataset_object_provenance_is_its_validated_record_and_json_text_is_snapshotted() -> None:
    record = _declared_f32_raw_record()
    ds = Dataset.from_dict(record)
    from_object = nio.matrix_dataset_package(ds, "nir")
    assert from_object.audits[0]["owner_dataset_record"] == ds.to_dict(), "no prior bytes are invented for an already normalized object"
    from_text = nio.matrix_dataset_package(json.dumps(record), "nir")
    assert from_text.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][1] == 1.0000000009313226, "decoded JSON text keeps its supplied scalars"
    assert _digests(from_text)[0].dataset_content_identity == _digests(from_object)[0].dataset_content_identity
    assert _feature(from_text).matrix.values_bytes() == _feature(from_object).matrix.values_bytes()
    assert from_object.to_full_dict() != from_text.to_full_dict() and json.dumps(from_text.to_summary_dict()["manifest"]) == json.dumps(from_object.to_summary_dict()["manifest"])


def test_false_finite_target_identity_stays_original_with_a_raw_snapshot() -> None:
    masked = _dev_record(
        y={"dtype": "float64", "shape": [4], "values": [1.0, 99.0, 3.0, 4.0]},
        target_mask={"dtype": "bool", "shape": [4], "values": [True, False, True, True]},
    )
    package, _ = nio.masked_matrix_dataset_package(masked, "nir")
    assert package.audits[0]["owner_dataset_record"]["dataset"]["y"]["values"][1] == 99.0
    assert _digests(package)[0].dataset_content_identity == labelled_dataset_content_identity(masked)
    assert dict(package.partitions["train"].payloads)["train/y"].columns[0].block.values_bytes()[8:16] == b"\x00" * 8  # type: ignore[union-attr]
