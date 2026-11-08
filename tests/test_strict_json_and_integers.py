# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Strict JSON number domain, per-cell integer conversion, V1 whole-content classes and carried requests.

Covers the canonical-float follow-up: shared decode/write number domain
(``[-2**63, 2**64 - 1]`` integers, ``-0`` token, finite floats), integer cells inspected
before numpy infers a float dtype, the corrected V1 whole-dataset normalizer (classes A-D),
and an explicit storage request that survives repackaging.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import nirs4all_io as nio
from nirs4all_io import strict_json
from nirs4all_io.canonical_json import canonical_json
from nirs4all_io.dataset_facade import load_dataset_document
from nirs4all_io.materialize import DatasetPackage, NumericContractError, NumericStorageOptions
from nirs4all_io.materialize.assemble import AssembledDataset, PartitionBlock
from nirs4all_io.materialize.numeric import int_to_f32, to_ir_f64
from nirs4all_io.materialize.package import FeatureMatrix
from nirs4all_io.public_content import dataset_content_bytes, labelled_dataset_content_identity
from nirs4all_io.public_dataset import Dataset
from nirs4all_io.spec.dataset_spec import DatasetSpec
from nirs4all_io.spec.enums import SpecError
from nirs4all_io.strict_json import NumberDomainError

N = 2**54 + 2**30 + 1
UPPER, LOWER = 2**54 + 2**31, 2**54


# --------------------------------------------------------------------------- #
# Strict JSON helper                                                            #
# --------------------------------------------------------------------------- #
def test_integer_domain_is_inclusive_and_checked_before_any_float_conversion() -> None:
    assert strict_json.loads("[-9223372036854775808, 18446744073709551615]") == [-(2**63), 2**64 - 1]
    for token in ("-9223372036854775809", "18446744073709551616", "123456789012345678901234567890", "-123456789012345678901234567890"):
        with pytest.raises(NumberDomainError, match="outside the supported range"):
            strict_json.loads(f"[{token}]")
    assert strict_json.loads("[9007199254740993]") == [9007199254740993], "an in-range integer stays an exact integer"


def test_minus_zero_token_is_negative_zero_bits_and_other_tokens_keep_their_kind() -> None:
    (value,) = strict_json.loads("[-0]")
    assert isinstance(value, float) and struct.pack(">d", value).hex() == "8000000000000000"
    for text, expected in (("[0]", 0), ("[-0.0]", -0.0), ("[1E5]", 1e5), ("[1.0]", 1.0), ("[-1]", -1)):
        (decoded,) = strict_json.loads(text)
        assert decoded == expected and type(decoded) is type(expected), text
    assert struct.pack(">d", strict_json.loads("[-0.0]")[0]).hex() == "8000000000000000"


def test_non_finite_tokens_and_exponent_overflow_refuse() -> None:
    for text in ("[NaN]", "[Infinity]", "[-Infinity]", "[1e400]", "[-1e400]", '{"a": 1e999}'):
        with pytest.raises(NumberDomainError):
            strict_json.loads(text)
    assert strict_json.loads("[1e308]") == [1e308]


def test_numeric_looking_strings_and_escapes_are_left_alone() -> None:
    document = strict_json.loads('{"a": "18446744073709551616", "b": "-0", "c": "\\u0031\\u0032", "d": ["NaN", "1e999", "\\"-0\\""]}')
    assert document == {"a": "18446744073709551616", "b": "-0", "c": "12", "d": ["NaN", "1e999", '"-0"']}
    assert strict_json.loads('{"nested": {"x": [{"y": -0}]}}')["nested"]["x"][0]["y"] == 0.0


def test_object_documents_get_the_same_integer_domain() -> None:
    assert strict_json.check_numbers({"a": [2**64 - 1, -(2**63), True, 1.5, "x"]})
    for bad in (2**64, -(2**63) - 1):
        with pytest.raises(NumberDomainError):
            strict_json.check_numbers({"a": [{"b": bad}]})
    strict_json.check_numbers({"flag": True, "n": np.int64(3)})  # bool is not an integer here; numpy scalars are not ingress ints


def test_every_oracle_ingress_site_uses_the_guard(tmp_path: Path) -> None:
    big = "18446744073709551616"
    with pytest.raises(NumberDomainError):
        DatasetSpec.from_json(f'{{"sources": [{{"id": "d", "role": "features", "input": "x.csv", "n": {big}}}]}}')
    with pytest.raises(NumberDomainError):
        DatasetSpec.from_yaml(f"sources:\n  - id: d\n    role: features\n    input: x.csv\n    n: {big}\n")
    path = tmp_path / "spec.json"
    path.write_text(f'{{"sources": [{{"id": "d", "role": "features", "input": "x.csv", "k": {big}}}]}}', encoding="utf-8")
    with pytest.raises(NumberDomainError):
        nio.to_spec(path)
    yml = tmp_path / "spec.yaml"
    yml.write_text(f"sources:\n  - id: d\n    role: features\n    input: x.csv\n    k: {big}\n", encoding="utf-8")
    with pytest.raises(NumberDomainError):
        nio.to_spec(yml)
    with pytest.raises(NumberDomainError):
        load_dataset_document(f'{{"a": {big}}}')
    with pytest.raises(NumberDomainError):
        load_dataset_document(f"a: {big}\n")
    with pytest.raises(NumberDomainError):
        load_dataset_document({"nested": [{"k": 2**64}]})
    with pytest.raises(NumberDomainError):
        nio.to_spec({"sources": [{"id": "d", "role": "features", "input": "x.csv", "k": 2**64}]})
    from nirs4all_io.materialize.folds import parse_fold_file

    fold = tmp_path / "folds.json"
    fold.write_text(f'[{{"train": [0, {big}], "val": [1]}}]', encoding="utf-8")
    with pytest.raises(NumberDomainError):
        parse_fold_file(fold, "json")
    from nirs4all_io.conventions.profiles import resolve_profiles

    profile = tmp_path / "profile.json"
    profile.write_text(f'{{"name": "p", "n": {big}}}', encoding="utf-8")
    with pytest.raises((NumberDomainError, SpecError, ValueError, KeyError, TypeError)):
        resolve_profiles([profile])
    ok = tmp_path / "ok.json"
    ok.write_text("[0, 1, 2, -0]", encoding="utf-8")
    assert strict_json.loads(ok.read_text()) == [0, 1, 2, -0.0]


# --------------------------------------------------------------------------- #
# Canonical writer                                                              #
# --------------------------------------------------------------------------- #
def test_writer_uses_the_shortest_repr_and_the_shared_integer_domain() -> None:
    assert sys.float_repr_style == "short"
    assert canonical_json({"a": 1, "b": 1.0, "c": 1e5, "z": [2**64 - 1, -(2**63)]}).splitlines()[1:3] == ['  "a": 1,', '  "b": 1.0,']
    for bad in (2**64, -(2**63) - 1):
        with pytest.raises(NumberDomainError):
            canonical_json({"x": [bad]})
    assert '"x": true' in canonical_json({"x": True})  # bool is not an integer
    with pytest.raises(ValueError, match="Out of range|nan"):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})


_VECTORS = json.loads((Path(__file__).parent / "fixtures" / "finite_float_vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _VECTORS["cases"], ids=lambda case: case["id"])
def test_writer_matches_the_common_literal_finite_float_vectors(case: dict[str, str]) -> None:
    value = struct.unpack(">d", bytes.fromhex(case["bits_hex"]))[0]
    text = canonical_json([value])
    assert text == f"[\n  {case['repr']}\n]\n"  # the literal token, byte for byte, with the existing layout
    assert struct.pack(">d", json.loads(text)[0]).hex() == case["bits_hex"], "the token round-trips to the same bits"


def test_writer_leaves_containers_unicode_and_numeric_looking_strings_alone() -> None:
    value = {"é": ["1e-07", "-0", {"z": 1e-07, "a": "日本語"}], "a": None}
    text = canonical_json(value)
    assert text == '{\n  "a": null,\n  "é": [\n    "1e-07",\n    "-0",\n    {\n      "a": "日本語",\n      "z": 1e-07\n    }\n  ]\n}\n'
    assert text.endswith("}\n") and "\\u" not in text


def test_the_four_pinned_tie_patterns_are_independent_of_core_formatting() -> None:
    for bits, token in (("42625d3cf82b2500", "630989504857.1562"), ("42708949efe72e80", "1136364813938.9062"), ("42745aaaa5465b80", "1398727660645.7188"), ("4289dc37530281c0", "3554201460816.2188")):
        value = struct.unpack(">d", bytes.fromhex(bits))[0]
        assert canonical_json([value]) == f"[\n  {token}\n]\n"


# --------------------------------------------------------------------------- #
# Integer cells: per cell, before numpy infers a dtype                          #
# --------------------------------------------------------------------------- #
def test_direct_integer_to_f32_and_exact_f64_gate_per_cell() -> None:
    assert to_ir_f64([N, 0.5], "float32")[0] == float(UPPER), "a mixed list no longer double-rounds"
    assert to_ir_f64([N, 1], "float32")[0] == float(UPPER) and to_ir_f64(np.array([N], dtype=np.int64), "float32")[0] == float(UPPER)
    assert int_to_f32(N) == float(UPPER) and int_to_f32(2**64 - 1) == float(2**64)
    assert to_ir_f64([2**54, 0.5], "float64")[0] == float(2**54)
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        to_ir_f64([2**53 + 1, 0.5], "float64")
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        to_ir_f64([2**53 + 1, 0.5])
    assert to_ir_f64(np.array([float(N), 0.5], dtype=np.float64), "float32")[0] == float(N), "a genuine host float64 keeps its already-rounded bits"
    assert to_ir_f64(np.array([float(N)]), "float32")[0] == float(N)  # the later storage rounding sees F64 bits, never the integer
    assert np.isnan(to_ir_f64([None, 1.5])[0])


def _text_record(token: str, dtype: str) -> str:
    record = {
        "schema": "nirs4all.dataset.v1", "schema_version": 1,
        "dataset": {
            "schema": "nirs4all.multimodal-dataset", "schema_version": 1, "name": "m", "sample_ids": ["s0", "s1"], "source_alignment": "strict",
            "sources": [{"name": "nir", "sample_ids": ["s0", "s1"], "representation_id": "signal_1d", "axes": ["sample", "wavelength"], "feature_names": None,
                         "axis_units": {"wavelength": "nm"}, "axis_coordinates": {"wavelength": [1.0, 2.0]},
                         "array": {"dtype": dtype, "shape": [2, 2], "values": [["@@", 0.5], [2.0, 3.0]]},
                         "presence_mask": {"dtype": "bool", "shape": [2], "values": [True, True]}}],
            "y": {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]}, "target_names": ["y"],
            "target_mask": {"dtype": "bool", "shape": [2], "values": [True, True]}, "task_type": "regression",
            "groups": None, "partitions": {"dtype": "<U5", "shape": [2], "values": ["train", "train"]},
        },
        "origin_ids": ["s0", "s1"], "fold_ids": [None, None],
    }  # fmt: skip
    return json.dumps(record).replace('"@@"', token)


def test_raw_json_integer_tokens_in_declared_float_arrays_five_case_receipt() -> None:
    def feature(token: str, dtype: str = "float64", **options: Any) -> float:
        package = nio.matrix_dataset_package_with_storage(_text_record(token, dtype), "nir", NumericStorageOptions(**options))
        payload = dict(package.partitions["train"].payloads)["train/x0"]
        assert isinstance(payload, FeatureMatrix)
        return float(payload.matrix.values[0, 0])

    with pytest.raises(ValueError, match="cannot be represented exactly"):
        feature("9007199254740993")  # int-inexact-u64 in a declared float64 array
    assert feature("18014398509481984") == float(2**54)  # int-exact-u64
    assert feature("9007199254740992.0") == float(2**53)  # float-already-rounded-u64
    with pytest.raises(NumberDomainError, match="outside the supported range"):
        feature("18446744073709551617")  # int-inexact-over-u64: refused at ingress
    assert feature("1.8446744073709552e+19") == 1.8446744073709552e19  # float-exact-over-u64
    assert struct.pack(">d", feature("-0")).hex() == "8000000000000000", "the -0 token keeps its negative-zero bits"


def test_declared_float32_integer_cell_rounds_once_from_the_integer() -> None:
    record = _text_record(str(N), "float32")
    package = nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(features="float64"))
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    assert payload.matrix.values[0, 0] == float(UPPER), "class A: the normalized F32 value is the direct rounding (upper), not the double-rounded lower"
    assert payload.matrix.precision.source_dtype == "float32"
    assert package.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][0] == N, "the raw integer token is retained as provenance"


def test_typed_int64_cases_and_whole_content_classes_a_to_d() -> None:
    def _typed(values: list[list[int]], dtype: str) -> dict[str, Any]:
        record = json.loads(_text_record("1", "float64"))
        record["dataset"]["sources"][0]["array"] = {"dtype": dtype, "shape": [2, 2], "values": values}
        return record

    # D: exact integers above 2**53 - 1 are admitted by the public Dataset and by the whole-content identity
    exact = _typed([[2**54, 1], [2, 3]], "int64")
    package = nio.matrix_dataset_package(exact, "nir")
    assert dict(package.partitions["train"].payloads)["train/x0"].matrix.values[0, 0] == float(2**54)  # type: ignore[union-attr]
    assert package.audits[0]["dataset_content_identity"] == labelled_dataset_content_identity(exact)
    # typed int64 inexact in float64: refused by the package (selected F64) and by the whole-content identity (no silent rounding)
    inexact = _typed([[2**53 + 1, 1], [2, 3]], "int64")
    with pytest.raises(ValueError, match="exactly"):
        nio.matrix_dataset_package(inexact, "nir")  # default F64 storage
    with pytest.raises(ValueError, match="exactly representable in binary64"):
        dataset_content_bytes(inexact)  # the tagged-binary64 identity never rounds an integer
    # (typed int64 inexact + explicit F32 through the public factory also needs the whole-content identity: reported
    # to the controller as an open policy question, not asserted here; the direct integer->F32 rule is covered above)
    # A: declared float32 + integer token: the old identity changes with the corrected normalization
    mixed = json.loads(_text_record(str(N), "float32"))
    assert dataset_content_bytes(mixed) == dataset_content_bytes(json.loads(_text_record(repr(float(UPPER)), "float32")))
    assert dataset_content_bytes(mixed) != dataset_content_bytes(json.loads(_text_record(repr(float(LOWER)), "float32")))
    # frozen unaffected input: a plain float record is byte-stable across the V1 normalizer
    plain = json.loads(_text_record("1.5", "float64"))
    assert dataset_content_bytes(plain) == dataset_content_bytes(Dataset.from_dict(plain).to_dict())
    # B/C: refused, no identity
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        dataset_content_bytes(json.loads(_text_record("9007199254740993", "float64")))
    with pytest.raises(NumberDomainError):
        strict_json.loads(_text_record("18446744073709551617", "float64"))


# --------------------------------------------------------------------------- #
# A carried explicit request is preserved through repackaging                   #
# --------------------------------------------------------------------------- #
def _arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[1.0 + 2.0**-30]]), np.array([2.0 + 2.0**-30])


def test_carried_explicit_f64_feature_request_refuses_the_f32_consumer() -> None:
    package = nio.to_dataset_package(_arrays(), features="float64")
    assert dict(package.partitions["train"].payloads)["train/x0"].matrix.precision.requested_dtype == "float64"  # type: ignore[union-attr]
    with pytest.raises(NumericContractError, match="SpectroDataset does not support requested float64"):
        DatasetPackage.from_assembled(package.to_assembled(), consumer="SpectroDataset")
    # a new explicit request may override it; omitting the request inherits it
    overridden = DatasetPackage.from_assembled(package.to_assembled(), features="float32", consumer="SpectroDataset")
    block = dict(overridden.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert block.dtype == "float32" and block.precision.requested_dtype == "float32"
    assert DatasetPackage.from_assembled(package.to_assembled()).manifest().root == package.manifest().root


def test_carried_explicit_f64_target_request_refuses_the_consumer_but_raw_f64_targets_stay_supported() -> None:
    package = nio.to_dataset_package(_arrays(), targets="float64")
    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        DatasetPackage.from_assembled(package.to_assembled(), consumer="SpectroDataset")
    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        nio.to_dataset_package(_arrays(), targets="float64", consumer="SpectroDataset")
    default = nio.to_dataset_package(_arrays())  # default raw F64 targets: no explicit guarantee
    assert DatasetPackage.from_assembled(default.to_assembled(), consumer="SpectroDataset") is not None
    assert nio.to_dataset_package(_arrays(), weights="float64", consumer="SpectroDataset") is not None, "weights are not numerical targets"
    from nirs4all_io.materialize.spectrodataset import project_spectrodataset

    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        project_spectrodataset(package.to_assembled())  # inherited explicit F64 target: refused before the SDK
    with pytest.raises(NumericContractError, match="float64 numerical targets"):
        project_spectrodataset(package.to_assembled(), targets="float64")


def test_carried_explicit_f32_request_survives_a_consumer_only_selection() -> None:
    package = nio.to_dataset_package(_arrays(), features="float32", targets="float32")
    again = DatasetPackage.from_assembled(package.to_assembled(), consumer="SpectroDataset")
    feature = dict(again.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert feature.precision.requested_dtype == "float32" and feature.precision.consumer == "SpectroDataset" and feature.values[0, 0] == np.float32(1.0)


def test_assembled_ir_without_a_carried_record_keeps_normal_selection() -> None:
    assembled = AssembledDataset(name="a", task_type="regression", signal_type="auto", n_sources=1)
    block = PartitionBlock(n_samples=1, source_ids=["X"], X=[np.array([[0.5]])], feature_headers=[["0"]], header_units=["index"], signal_types=[None])
    assembled.blocks["train"] = block
    assert DatasetPackage.from_assembled(assembled, consumer="SpectroDataset") is not None


def test_yaml_object_documents_are_domain_checked_before_the_public_factory() -> None:
    record = json.loads(_text_record("1.0", "float64"))
    record["dataset"]["name"] = 2**64
    with pytest.raises(NumberDomainError):
        nio.matrix_dataset_package(record, "nir")
    document = yaml.safe_dump(json.loads(_text_record("1.0", "float64")))
    assert nio.matrix_dataset_package(document, "nir") is not None


# --------------------------------------------------------------------------- #
# Declared float32 cells: one checked normalization (overflow, underflow, ±0)   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token", ["5e-324", "-5e-324", "1e-46", "-7.006492321624085e-46"])
def test_declared_float32_underflow_refuses_in_storage_and_identity(token: str) -> None:
    record = json.loads(_text_record(token, "float32"))
    for options in (NumericStorageOptions(), NumericStorageOptions(features="float64"), NumericStorageOptions(features="float32")):
        with pytest.raises(ValueError, match="rounds to zero in float32"):
            nio.matrix_dataset_package_with_storage(record, "nir", options)
    with pytest.raises(ValueError, match="rounds to zero in float32"):
        dataset_content_bytes(record)  # the whole-dataset identity uses the same corrected normalizer
    with pytest.raises(ValueError, match="rounds to zero in float32"):
        Dataset.from_dict(record)
    # the same cell under a declared float64 array is a perfectly valid binary64 subnormal
    assert nio.matrix_dataset_package(json.loads(_text_record(token, "float64")), "nir") is not None


@pytest.mark.parametrize(("token", "bits"), [("1.401298464324817e-45", "01000000"), ("-1.401298464324817e-45", "01000080"), ("0.0", "00000000"), ("-0.0", "00000080"), ("3.4028234663852886e+38", "ffff7f7f")])
def test_declared_float32_subnormal_zero_and_max_controls_are_kept(token: str, bits: str) -> None:
    record = json.loads(_text_record(token, "float32"))
    package = nio.matrix_dataset_package(record, "nir")
    block = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert block.dtype == "float32" and block.values_bytes()[:4].hex() == bits
    widened = nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(features="float64"))
    wide = dict(widened.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert struct.pack("<d", float(np.frombuffer(bytes.fromhex(bits), dtype="<f4")[0])) == wide.values_bytes()[:8], "widening keeps the signed zero / subnormal exactly"
    assert dataset_content_bytes(record) == dataset_content_bytes(Dataset.from_dict(record).to_dict())


def test_declared_float32_overflow_still_refuses_and_every_numeric_kind_is_covered() -> None:
    with pytest.raises(ValueError, match="overflows float32"):
        nio.matrix_dataset_package(json.loads(_text_record("1e39", "float32")), "nir")
    # float cells, integer cells and mixed lists all pass through the same rounding in one array
    package = nio.matrix_dataset_package_with_storage(json.loads(_text_record("1.0000000009313226", "float32")), "nir", NumericStorageOptions(features="float64"))
    assert dict(package.partitions["train"].payloads)["train/x0"].matrix.values[0, 0] == 1.0  # type: ignore[union-attr]
    # a declared-float32 TARGET array is checked too
    record = json.loads(_text_record("1.0", "float64"))
    record["dataset"]["y"] = {"dtype": "float32", "shape": [2], "values": [5e-324, 2.0]}
    with pytest.raises(ValueError, match="rounds to zero in float32"):
        nio.matrix_dataset_package(record, "nir")


def test_genuine_host_float32_array_keeps_its_already_normalized_values() -> None:
    values = np.array([[1.401298464324817e-45, -0.0], [0.5, 1.0]], dtype=np.float32)
    dataset = Dataset.from_sources({"nir": values}, sample_ids=["a", "b"], y=np.arange(2.0))
    package = nio.matrix_dataset_package_with_storage(dataset, "nir", NumericStorageOptions(features="float64"))
    block = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert block.values.astype(np.float32).tobytes() == values.tobytes() and block.precision.source_dtype == "float32"


# --------------------------------------------------------------------------- #
# Decimal underflow guard (class F)                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token", ["1e-400", "2e-324", "-1e-400", "-2e-324", "5e-325", "0.0001e-400", "1.5E-999"])
def test_decimal_tokens_that_lose_a_nonzero_mantissa_refuse(token: str) -> None:
    with pytest.raises(NumberDomainError, match="underflows to zero"):
        strict_json.loads(f"[{token}]")


@pytest.mark.parametrize(("token", "bits"), [("0e5", "0000000000000000"), ("-0e5", "8000000000000000"), ("0.000e-999", "0000000000000000"), ("-0.0E-9999", "8000000000000000"), ("0.0", "0000000000000000"), ("-0.0", "8000000000000000"), ("0e-999999", "0000000000000000")])
def test_true_zero_tokens_are_accepted_with_their_sign(token: str, bits: str) -> None:
    (value,) = strict_json.loads(f"[{token}]")
    assert struct.pack(">d", value).hex() == bits


def test_exponent_digits_never_count_as_mantissa_and_overflow_still_refuses() -> None:
    assert strict_json.loads("[1e-300, 5e-324, 4.9e-324]")[1] == 5e-324  # subnormals are real values
    for token in ("1e400", "-1e999"):
        with pytest.raises(NumberDomainError, match="overflows"):
            strict_json.loads(f"[{token}]")
    with pytest.raises(NumberDomainError, match="underflows"):
        strict_json.loads('{"a": {"b": [2e-324]}}')


def test_public_factory_refuses_decimal_underflow_in_serialized_records_but_not_host_zero() -> None:
    for token in ("1e-400", "2e-324", "-2e-324"):
        with pytest.raises(NumberDomainError, match="underflows"):
            nio.matrix_dataset_package(_text_record(token, "float64"), "nir")
    package = nio.matrix_dataset_package(_text_record("-0e5", "float64"), "nir")
    block = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert block.values_bytes()[:8].hex() == "0000000000000080", "the true zero keeps its sign"
    host = Dataset.from_sources({"nir": np.array([[0.0, -0.0], [1.0, 2.0]])}, sample_ids=["a", "b"], y=np.arange(2.0))
    assert nio.matrix_dataset_package(host, "nir") is not None, "a genuine host F64 zero has no lexeme to inspect"


# --------------------------------------------------------------------------- #
# Retained cells on the actual target API route; Python int domain              #
# --------------------------------------------------------------------------- #
def _y_block(package: DatasetPackage):
    return dict(package.partitions["train"].payloads)["train/y"].columns[0].block  # type: ignore[union-attr]


def test_mixed_target_list_keeps_its_integer_cells_through_the_selector() -> None:
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        nio.to_dataset_package((np.ones((2, 1)), [2**53 + 1, 0.5]))  # default F64
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        nio.to_dataset_package((np.ones((2, 1)), [2**53 + 1, 0.5]), targets="float64")
    direct = nio.to_dataset_package((np.ones((2, 1)), [N, 0.5]), targets="float32")
    assert _y_block(direct).values_bytes()[:4].hex() == "0100805a", "direct integer -> F32 bits (upper), not the double-rounded 0000805a"
    exact = nio.to_dataset_package((np.ones((2, 1)), [2**54, 0.5]))
    assert float(_y_block(exact).values[0, 0]) == float(2**54)
    # controls: an ndarray / list of already-rounded floats keeps its genuine F64 bits and rounds once under F32
    rounded = np.array([float(N), 0.5])
    assert nio.to_dataset_package((np.ones((2, 1)), rounded), targets="float32") is not None
    assert _y_block(nio.to_dataset_package((np.ones((2, 1)), rounded), targets="float32")).values_bytes()[:4].hex() == "0000805a"
    assert _y_block(nio.to_dataset_package((np.ones((2, 1)), [float(N), 0.5]), targets="float32")).values_bytes()[:4].hex() == "0000805a"


def test_raw_integer_and_string_targets_keep_their_semantics_on_the_array_route() -> None:
    ints = nio.load((np.ones((3, 1)), [1, 2, 3]), target="assembled")
    assert ints.blocks["train"].y is not None and ints.blocks["train"].y.dtype.kind == "i"
    strings = nio.load((np.ones((2, 1)), ["a", "b"]), target="assembled")
    assert strings.blocks["train"].y is not None and strings.blocks["train"].y.dtype.kind == "U"
    big = nio.load((np.ones((2, 1)), [2**64 - 1, 1]), target="assembled")
    assert big.blocks["train"].y is not None
    with pytest.raises(NumberDomainError):
        nio.load((np.ones((2, 1)), [2**64, 1]), target="assembled")


def test_python_int_domain_is_checked_before_conversion_everywhere() -> None:
    for bad in (2**64, -(2**63) - 1):
        with pytest.raises(NumberDomainError):
            to_ir_f64([bad], "float64")
        with pytest.raises(NumberDomainError):
            to_ir_f64([bad, 0.5], "float32")
        with pytest.raises(NumberDomainError):
            nio.to_dataset_package({"X": [[bad]], "y": np.array([0.0])})
    assert to_ir_f64([2**64 - 1], "float32")[0] == float(2**64)  # in-domain integers convert directly
    assert to_ir_f64(np.array([2.0**64]), "float64")[0] == 2.0**64, "a genuine already-rounded host F64 2**64 is legitimate"
    host = nio.to_dataset_package((np.array([[2.0**64]]), np.array([0.0])))
    assert host.partitions["train"].payloads


# --------------------------------------------------------------------------- #
# Whole link: valid record, representable selection, unrepresentable identity    #
# --------------------------------------------------------------------------- #
def _typed_int_record(value: int, dtype: str = "int64") -> dict[str, Any]:
    record = json.loads(_text_record("1", "float64"))
    record["dataset"]["sources"][0]["array"] = {"dtype": dtype, "shape": [2, 2], "values": [[value, 1], [2, 3]]}
    return record


def test_typed_int64_not_exact_in_binary64_accepts_direct_f32_with_an_honest_none_link() -> None:
    record = _typed_int_record(N)
    package = nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(features="float32"))
    block = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert block.values_bytes()[:4].hex() == "0100805a" and block.precision.to_dict() == {"dtype": "float32", "origin": "requested", "source_dtype": "int64", "requested_dtype": "float32", "consumer": None}
    assert package.audits[0]["dataset_content_identity"] is None
    assert package.audits[0]["dataset_content_identity_unavailable"] == {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}
    assert package.warnings == [], "the reason lives in the owner audit, never in warnings"
    assert package.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][0] == N, "the original record is retained as provenance"
    assert package.to_assembled().audits == package.audits
    for restored in (DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict()))), *_both(package)):
        assert restored.audits[0]["dataset_content_identity"] is None
        assert restored.audits[0]["dataset_content_identity_unavailable"] == {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}
        assert restored.executed_projection(["nir"]).identity.dataset_content_identity is None
    for restored in (DatasetPackage.from_full_dict(json.loads(canonical_json(package.to_full_dict()))), DatasetPackage.from_assembled(package.to_assembled())):
        assert restored.manifest().to_dict() == package.manifest().to_dict()
    from nirs4all_io.materialize.projection import ExecutedProjectionIdentityV1
    from nirs4all_io.public_package import verified_dataset_identity

    identity = package.executed_projection(["nir"]).identity
    assert identity.dataset_content_identity is None
    identity.validate(package)
    with pytest.raises(NumericContractError, match="disagrees"):
        ExecutedProjectionIdentityV1.from_dict({**identity.to_dict(), "dataset_content_identity": {"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256", "fingerprint": "0" * 64}}).validate(package)
    assert verified_dataset_identity(package) is None
    with pytest.raises(ValueError, match="exactly representable in binary64"):
        dataset_content_bytes(record)  # the whole operation itself never rounds
    for options in (NumericStorageOptions(), NumericStorageOptions(features="float64")):
        with pytest.raises(NumericContractError, match="cannot represent exactly"):
            nio.matrix_dataset_package_with_storage(record, "nir", options)


def _both(package: DatasetPackage) -> tuple[DatasetPackage, DatasetPackage]:
    wire = json.loads(canonical_json(package.to_assembled().to_full_dict()))
    return DatasetPackage.from_assembled(package.to_assembled()), DatasetPackage.from_assembled(AssembledDataset.from_full_dict(wire))


def test_a_supplied_reason_is_never_proof() -> None:
    from nirs4all_io.public_package import recover_source_evidence

    reason = {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}
    # a representable record whose audit CLAIMS unrepresentability: the rebuild denies the proof, the link stays None
    honest = nio.matrix_dataset_package(_typed_int_record(2**54), "nir")
    wire = json.loads(canonical_json(honest.to_full_dict()))
    wire["audits"][0]["dataset_content_identity"] = None
    wire["audits"][0]["dataset_content_identity_unavailable"] = reason
    forged = DatasetPackage.from_full_dict(wire)
    assert recover_source_evidence(forged).source_evidence is None
    assert forged.executed_projection(["nir"]).identity.dataset_content_identity is None
    # an unrepresentable record whose reason was removed or altered: no certificate either
    unrepresentable = nio.matrix_dataset_package_with_storage(_typed_int_record(N), "nir", NumericStorageOptions(features="float32"))
    for mutate in (lambda a: a.pop("dataset_content_identity_unavailable"), lambda a: a["dataset_content_identity_unavailable"].update(reason="other"), lambda a: a.update(dataset_content_identity={"profile": "nirs4all.io.dataset-content.v1", "algorithm": "sha256", "fingerprint": "0" * 64})):
        wire = json.loads(canonical_json(unrepresentable.to_full_dict()))
        mutate(wire["audits"][0])
        restored = DatasetPackage.from_full_dict(wire)
        assert restored.executed_projection(["nir"]).identity.dataset_content_identity is None
        assert recover_source_evidence(restored).source_evidence is None
    kept = DatasetPackage.from_full_dict(json.loads(canonical_json(unrepresentable.to_full_dict())))
    assert recover_source_evidence(kept).source_evidence is not None, "the genuine package rebuilds exactly, reason included"


def test_only_unrepresentability_yields_none_never_an_invalid_source() -> None:
    # exact int64 above 2**53 - 1: computed (class D)
    exact = nio.matrix_dataset_package(_typed_int_record(2**54), "nir")
    assert exact.audits[0]["dataset_content_identity"] == labelled_dataset_content_identity(_typed_int_record(2**54)) and "dataset_content_identity_unavailable" not in exact.audits[0]
    # invalid sources are refused, never reported as an unavailable link
    bad_dtype = _typed_int_record(2**53 + 1, "float64")
    with pytest.raises(ValueError, match="cannot be represented exactly by a declared float64"):
        nio.matrix_dataset_package_with_storage(bad_dtype, "nir", NumericStorageOptions(features="float32"))
    shape = _typed_int_record(1)
    shape["dataset"]["sources"][0]["array"]["shape"] = [3, 2]
    with pytest.raises(ValueError):
        nio.matrix_dataset_package_with_storage(shape, "nir", NumericStorageOptions(features="float32"))
    capped = _typed_int_record(2**60)
    capped["dataset"]["sources"][0]["array"]["dtype"] = "object"
    with pytest.raises(ValueError):
        nio.matrix_dataset_package_with_storage(capped, "nir", NumericStorageOptions(features="float32"))


# --------------------------------------------------------------------------- #
# The twelve source-declaration review rows (admission and whole link)           #
# --------------------------------------------------------------------------- #
_ROWS: list[tuple[str, str, str, str, str]] = [
    # (id, raw token, source dtype, request, outcome) ; outcome: link / nolink / refuse
    ("int-2^54-f64", "18014398509481984", "float64", "none", "link"),
    ("int-2^54-f64", "18014398509481984", "float64", "float64", "link"),
    ("int-2^54-f64", "18014398509481984", "float64", "float32", "link"),
    ("int-2^53+1-f64", "9007199254740993", "float64", "none", "refuse"),
    ("int-2^53+1-f64", "9007199254740993", "float64", "float64", "refuse"),
    ("int-2^53+1-f64", "9007199254740993", "float64", "float32", "refuse"),
    ("int-n-f64", str(N), "float64", "float32", "refuse"),
    ("float-literal-f64", "0.1", "float64", "none", "link"),
    ("float-literal-f64", "0.1", "float64", "float32", "link"),
    ("float-integral-literal-f64", "9007199254740993.0", "float64", "float32", "link"),
    ("int-n-f32", str(N), "float32", "none", "link"),
    ("int-n-f32", str(N), "float32", "float32", "link"),
    ("int-n-f32", str(N), "float32", "float64", "link"),
    ("underflow-f32", "5e-324", "float32", "none", "refuse"),
    ("underflow-f32", "5e-324", "float32", "float64", "refuse"),
    ("overflow-f32", "1e40", "float32", "none", "refuse"),
    ("overflow-f32", "1e40", "float32", "float32", "refuse"),
    ("int-2^54-int64", "18014398509481984", "int64", "none", "link"),
    ("int-2^54-int64", "18014398509481984", "int64", "float32", "link"),
    ("int-2^53+1-int64", "9007199254740993", "int64", "none", "refuse"),
    ("int-2^53+1-int64", "9007199254740993", "int64", "float64", "refuse"),
    ("int-2^53+1-int64", "9007199254740993", "int64", "float32", "nolink"),
    ("int-2^53+1-uint64", "9007199254740993", "uint64", "float32", "nolink"),
    ("decimal-underflow", "1e-400", "float64", "none", "refuse"),
    ("decimal-underflow", "2e-324", "float64", "float32", "refuse"),
    ("zero-token", "-0e5", "float64", "float32", "link"),
]


@pytest.mark.parametrize(("case", "token", "dtype", "request_", "outcome"), _ROWS, ids=[f"{r[0]}-{r[2]}-{r[3]}" for r in _ROWS])
def test_source_declaration_review_rows(case: str, token: str, dtype: str, request_: str, outcome: str) -> None:
    options = NumericStorageOptions(features=None if request_ == "none" else request_)
    text = json.dumps(_typed_int_record(int(token), dtype)) if dtype in ("int64", "uint64") else _text_record(token, dtype)
    if outcome == "refuse":
        with pytest.raises(ValueError):
            nio.matrix_dataset_package_with_storage(text, "nir", options)
        return
    package = nio.matrix_dataset_package_with_storage(text, "nir", options)
    audit_identity = package.audits[0]["dataset_content_identity"]
    block = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    if outcome == "link":
        assert audit_identity == labelled_dataset_content_identity(text)
    else:
        assert audit_identity is None and package.audits[0]["dataset_content_identity_unavailable"] == {"profile": "nirs4all.io.dataset-content.v1", "reason": "not_representable"}
    expected_storage = options.features or (dtype if dtype in ("float32", "float64") else "float64")
    assert block.dtype == expected_storage
    value = int(token) if token.lstrip("-").isdigit() else float(token)
    if dtype == "float32" and isinstance(value, int):
        assert block.values[0, 0] == int_to_f32(value), "declared float32: the integer normalizes directly first"
    if request_ == "float32" and dtype in ("int64", "uint64") and isinstance(value, int):
        assert block.values[0, 0] == int_to_f32(value), "a generic integer source rounds DIRECTLY from the integer"
    if request_ == "float32" and dtype == "float64" and isinstance(value, float):
        assert block.values[0, 0] == np.float32(value), "a float64 literal rounds from its binary64 value"


def test_class_label_cap_and_f32_exactness_stay_with_the_review_rows() -> None:
    for labels, request_ in (([2**53, 0], None), ([2**53 - 1, 0], "float32"), ([16777217, 0], "float32")):
        with pytest.raises(NumericContractError):
            record = _typed_int_record(1)
            record["dataset"]["y"] = {"dtype": "int64", "shape": [2], "values": labels}
            record["dataset"]["task_type"] = "classification"
            nio.matrix_dataset_package_with_storage(record, "nir", NumericStorageOptions(targets=request_))
    ok = _typed_int_record(1)
    ok["dataset"]["y"] = {"dtype": "int64", "shape": [2], "values": [16777218, 0]}
    ok["dataset"]["task_type"] = "classification"
    assert nio.matrix_dataset_package_with_storage(ok, "nir", NumericStorageOptions(targets="float32")) is not None
