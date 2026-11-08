# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Post-B4 owner corrections (Python items 1, 2, 3, 5, 6, 7) on the exact reviewer inputs (Opus ``neg1``-``neg5``).

Output types, bytes, audits, IDs and roots are asserted, not only accept/refuse counts. Expected literals that depend on the
other owner were taken from the frozen correct Rust exporter (anchors noted inline), never from a Python output.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io import MultimodalDataset, strict_json
from nirs4all_io.materialize import NumericContractError
from nirs4all_io.materialize.package import MetadataTable, _metadata_body
from nirs4all_io.public_content import labelled_dataset_content_identity
from nirs4all_io.public_dataset import dataset as public_dataset
from nirs4all_io.strict_json import NumberDomainError

INT = "9007199254740993"  # 2**53 + 1
N = "18014399583223809"  # 2**54 + 2**30 + 1: nearest-even float32 0x5a800001 directly, 0x5a800000 through binary64
X = "18446744073709551616"  # 2**64

# --------------------------------------------------------------------------- #
# Item 1: the E4 gate keys on the RESOLVED float dtype                          #
# --------------------------------------------------------------------------- #
_RECORD = (
    '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"p","sample_ids":["a","b"],"source_alignment":"strict","sources":[{"name":"nir","sample_ids":["a","b"],'
    '"representation_id":"signal_1d","axes":["sample","wavelength"],"feature_names":null,"axis_units":{"wavelength":"nm"},"axis_coordinates":{"wavelength":[900.0,1000.0]},'
    '"array":{"dtype":"%(xd)s","shape":[2,2],"values":%(x)s}}],"y":%(y)s,"groups":%(g)s,"partitions":{"dtype":"<U5","shape":[2],"values":["train","train"]}}'
)
_ENVELOPE = '{"schema":"nirs4all.dataset.v1","schema_version":1,"origin_ids":["a","b"],"fold_ids":[null,null],"dataset":%s}'


def _record(xd: str = "float64", x: str = "[[1.5,2.0],[3.0,4.0]]", y: str = '{"dtype":"float64","shape":[2],"values":[1.0,2.0]}', g: str = "null") -> str:
    return _RECORD % {"xd": xd, "x": x, "y": y, "g": g}


def _array(dtype: str, values: str) -> str:
    return f'{{"dtype":"{dtype}","shape":[2],"values":{values}}}'


F64_SPELLINGS = ("float64", "<f8", "=f8", ">f8", "f8", "float", "double", "d")
F32_SPELLINGS = ("float32", "<f4", "=f4", ">f4", "f4", "single")


def _first_bytes(array: Any, width: str) -> str:
    return np.ascontiguousarray(np.asarray(array), dtype=width).tobytes()[:4 if width == "<f4" else 8].hex()


@pytest.mark.parametrize("dtype", F64_SPELLINGS)
def test_declared_f64_spellings_refuse_an_inexact_integer_in_every_array(dtype: str) -> None:
    refusals = {
        "source": _record(dtype, f"[[{INT},2],[3,4]]"),
        "target": _record(y=_array(dtype, f"[{INT},1]")),
        "groups": _record(g=_array(dtype, f"[{INT},1]")),
    }
    for role, text in refusals.items():
        with pytest.raises(ValueError, match="cannot be represented exactly by a declared float64 array"):
            MultimodalDataset.from_dict(strict_json.loads(text))
        assert role


@pytest.mark.parametrize("dtype", F32_SPELLINGS)
def test_declared_f32_spellings_round_the_integer_directly_and_refuse_underflow(dtype: str) -> None:
    cohort = MultimodalDataset.from_dict(strict_json.loads(_record(dtype, f"[[{N},2],[3,4]]", y=_array(dtype, f"[{N},1]"))))
    assert _first_bytes(next(iter(cohort.sources.values())).values, "<f4") == "0100805a", "direct nearest-even upper, never 0000805a"
    assert cohort.y is not None and _first_bytes(cohort.y, "<f4") == "0100805a"
    for text in (_record(dtype, "[[5e-324,2.0],[3.0,4.0]]"), _record(y=_array(dtype, "[5e-324,1.0]")), _record(g=_array(dtype, "[5e-324,1.0]"))):
        with pytest.raises(ValueError, match="rounds to zero in float32"):
            MultimodalDataset.from_dict(strict_json.loads(text))


@pytest.mark.parametrize("dtype", F64_SPELLINGS)
def test_declared_f64_spellings_keep_a_genuine_denormal_and_exact_integers(dtype: str) -> None:
    cohort = MultimodalDataset.from_dict(strict_json.loads(_record(dtype, "[[5e-324,18014398509481984],[3,4]]")))
    values = np.ascontiguousarray(next(iter(cohort.sources.values())).values, dtype="<f8")
    assert values.tobytes()[:16].hex() == "01000000000000000000000000005043"


def test_ragged_arrays_are_gated_under_every_spelling() -> None:
    ragged = (
        '{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"m","sample_ids":["a"],"sources":[{"source_kind":"ragged_series","name":"r","sample_ids":["a"],'
        '"representation_id":"series_mv","axes":["sample","time","variable"],"array":{"dtype":"%s","shape":[2,1],"values":[[%s],[2.0]]},"offsets":{"dtype":"int64","shape":[2],"values":[0,2]},'
        '"time_coordinates":null,"channel_names":null,"time_unit":null,"presence_mask":{"dtype":"bool","shape":[1],"values":[true]}}],'
        '"y":{"dtype":"float64","shape":[1],"values":[1.0]},"target_names":["y"],"target_mask":{"dtype":"bool","shape":[1],"values":[true]},'
        '"groups":null,"partitions":{"dtype":"<U5","shape":[1],"values":["train"]}}'
    )
    for dtype in F64_SPELLINGS:
        with pytest.raises(ValueError, match="declared float64 array"):
            MultimodalDataset.from_dict(strict_json.loads(ragged % (dtype, INT)))
    for dtype in F32_SPELLINGS:
        with pytest.raises(ValueError, match="rounds to zero in float32"):
            MultimodalDataset.from_dict(strict_json.loads(ragged % (dtype, "5e-324")))
        ok = MultimodalDataset.from_dict(strict_json.loads(ragged % (dtype, N)))
        series = next(iter(ok.sources.values())).values
        assert _first_bytes(series.values, "<f4") == "0100805a"  # type: ignore[union-attr]


def test_typed_arrays_and_nonfinite_pass_through_are_unchanged() -> None:
    host = MultimodalDataset.from_dict(strict_json.loads(_record("<f4", '[[{"nonfinite":"nan"},1.5],[2.5,3.5]]')))
    values = np.asarray(next(iter(host.sources.values())).values)
    assert np.isnan(values[0, 0]) and values.dtype == np.dtype("<f4")


_PUBLIC_REFUSED = ("<f8", "=f8", ">f8", "f8", "float", "double", "d", "<f4", "=f4", ">f4", "f4", "single", "float16", "<f2", "e", "half", "longdouble", "float128", "<i8", "i8", "int", "<u8", "?", "O", "U5")
_PUBLIC_ACCEPTED = ("float64", "float32", "int64", "uint64", "int32", "uint32", "int16", "uint16", "int8", "uint8")


@pytest.mark.parametrize("dtype", _PUBLIC_REFUSED)
def test_the_public_record_enforces_the_closed_dtype_vocabulary_independently(dtype: str) -> None:
    for text in (_record(dtype, "[[1.5,2.0],[3.0,4.0]]"), _record(y=_array(dtype, "[1,2]")), _record(g=_array(dtype, "[1,2]"))):
        for build in (lambda t: public_dataset(_ENVELOPE % t), lambda t: nio.matrix_dataset_package(_ENVELOPE % t, "nir"), lambda t: nio.masked_matrix_dataset_package(_ENVELOPE % t, "nir")):
            with pytest.raises(ValueError):
                build(text)
        with pytest.raises(ValueError):
            labelled_dataset_content_identity(strict_json.loads(_ENVELOPE % text))


@pytest.mark.parametrize("dtype", _PUBLIC_ACCEPTED)
def test_the_closed_public_vocabulary_keeps_its_members(dtype: str) -> None:
    group = _array(dtype, "[1,2]") if dtype not in ("float64", "float32") else _array(dtype, "[1.0,2.0]")
    assert public_dataset(_ENVELOPE % _record(g=group)).multimodal.groups is not None


# --------------------------------------------------------------------------- #
# Item 7: inexact integer groups                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_inexact_integer_groups_refuse_in_the_public_record_and_the_whole_grammar(dtype: str) -> None:
    text = _ENVELOPE % _record(g=_array(dtype, f"[{INT},1]"))
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        public_dataset(text)
    with pytest.raises(NumericContractError):
        labelled_dataset_content_identity(strict_json.loads(text))
    for factory in (nio.matrix_dataset_package, nio.masked_matrix_dataset_package):
        with pytest.raises(NumericContractError):
            factory(text, "nir")


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_exact_integer_groups_and_text_groups_are_not_capped(dtype: str) -> None:
    exact = _ENVELOPE % _record(g=_array(dtype, "[18014398509481984,1]"))
    assert public_dataset(exact).multimodal.groups is not None
    assert labelled_dataset_content_identity(strict_json.loads(exact))["algorithm"] == "sha256"
    text_groups = _ENVELOPE % _record(g=_array("<U16", f'["{INT}","b"]'))  # a text group is never numeric authority
    assert labelled_dataset_content_identity(strict_json.loads(text_groups))["algorithm"] == "sha256"


# --------------------------------------------------------------------------- #
# Item 2: removed rows never refuse a numeric-role cell with a floating peer      #
# --------------------------------------------------------------------------- #
def _numeric_spec() -> dict[str, Any]:
    columns = [{"role": "features", "select": ["x", "z"]}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}, {"role": "metadata", "select": ["m"]}]
    return {"sources": [{"id": "t", "input": "d.csv", "columns": columns}], "params": {"delimiter": ",", "na": {"policy": "remove_sample"}}}


def _write_row_pair(tmp_path: Path, token_column: str, na_column: str | None, token: str) -> None:
    columns = ["x", "z", "y", "w", "m"]
    first = {c: "1.5" for c in columns}
    first[token_column] = token
    if na_column is not None:
        first[na_column] = ""
    second = {c: "2.5" for c in columns}
    (tmp_path / "d.csv").write_text(",".join(columns) + "\n" + ",".join(first[c] for c in columns) + "\n" + ",".join(second[c] for c in columns) + "\n")


_REMOVAL_PAIRS = [(token_column, na_column) for token_column in ("x", "y", "w", "m") for na_column in ("x", "z", "y", "w") if token_column != na_column]  # the 13 distinct-cell pairs


@pytest.mark.parametrize("token", [X, "1e-400"])
@pytest.mark.parametrize(("token_column", "na_column"), _REMOVAL_PAIRS)
def test_a_removed_row_never_refuses_and_the_survivor_is_exact(tmp_path: Path, token_column: str, na_column: str, token: str) -> None:
    _write_row_pair(tmp_path, token_column, na_column, token)
    package = nio.to_dataset_package(_numeric_spec(), base_dir=tmp_path)
    partition = package.partitions["train"]
    assert partition.n_samples == 1
    payload = dict(partition.payloads)
    features = payload["train/x0"].matrix  # type: ignore[union-attr]
    assert features.values.ravel().tolist() == [2.5, 2.5] and features.validity.ravel().tolist() == [True, True]
    assert payload["train/weights"].matrix.values.ravel().tolist() == [2.5]  # type: ignore[union-attr]


@pytest.mark.parametrize("token", [X, "1e-400"])
@pytest.mark.parametrize("token_column", ["x", "y", "w", "m"])
def test_the_same_cell_in_a_kept_row_still_refuses(tmp_path: Path, token_column: str, token: str) -> None:
    _write_row_pair(tmp_path, token_column, None, token)
    with pytest.raises(NumberDomainError, match="outside the supported range|underflows"):
        nio.to_dataset_package(_numeric_spec(), base_dir=tmp_path)


def test_classification_stays_the_original_source_column_not_the_survivors(tmp_path: Path) -> None:
    """An out-of-domain removed token beside floats must not turn the surviving exact integer into anything else."""
    (tmp_path / "d.csv").write_text(f"x,y\n{X},\n3,1.0\n1.5,2.0\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}], "params": {"delimiter": ",", "na": {"policy": "remove_sample"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    features = dict(package.partitions["train"].payloads)["train/x0"].matrix  # type: ignore[union-attr]
    assert features.values.ravel().tolist() == [3.0, 1.5] and features.dtype == "float64"


# --------------------------------------------------------------------------- #
# Item 3: boolean sample IDs are CPython True/False                              #
# --------------------------------------------------------------------------- #
_BOOL_FEATURE = "0e8a36504cae0257af80563e375c01a7520c4205a165533ea8df67b19edb2b60"  # frozen Rust exporter anchor
_BOOL_TARGET = "c204499826f2809212442296df6f7b5dca89086d64adc0f9828d8db6c4421ae7"
_BOOL_ROOT = "42e7b78734bf9201927090ed74a6a6ba08427414963f6409b9a8d7fe2d6fd67a"


@pytest.mark.parametrize(("first", "second"), [("TRUE", "FALSE"), ("true", "false"), ("True", "False")])
def test_boolean_ids_render_as_cpython_true_false_with_the_rust_evidence(tmp_path: Path, first: str, second: str) -> None:
    (tmp_path / "d.csv").write_text(f"id,x,y\n{first},1.5,1.0\n{second},2.5,2.0\n")
    spec = {
        "sample_index": {"by": "id", "key": "id"},
        "sources": [{"id": "t", "input": "d.csv", "key": "id", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}],
        "params": {"delimiter": ",", "na": {"policy": "ignore"}},
    }
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    projection = package.executed_projection(["t"])
    assert projection.identity.sample_ids == ["True", "False"] and projection.identity.observation_ids == ["True", "False"]
    assert (projection.identity.feature_content_fingerprint, projection.identity.target_content_fingerprint) == (_BOOL_FEATURE, _BOOL_TARGET)
    assert package.manifest().root == _BOOL_ROOT
    assert struct.pack("<Q", 4) + b"True" in projection.feature_preimage and b"true" not in projection.feature_preimage
    metadata = dict(package.partitions["train"].payloads)["train/metadata"]
    assert isinstance(metadata, MetadataTable) and _metadata_body(metadata.frame)["columns"][0]["values"] == [True, False], "the metadata cells stay genuine booleans"


# --------------------------------------------------------------------------- #
# Item 5: classification per LOADED INPUT before concat_samples                  #
# --------------------------------------------------------------------------- #
def _concat(tmp_path: Path, first: str, second: str, role: str = "metadata") -> Any:
    (tmp_path / "a.csv").write_text(first)
    (tmp_path / "b.csv").write_text(second)
    column = {"role": role, "select": ["m"]}
    spec = {"sources": [{"id": "d", "role": "mixed", "input": ["a.csv", "b.csv"], "merge": "concat_samples", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, column]}], "params": {"delimiter": ","}}
    return nio.to_dataset_package(spec, base_dir=tmp_path)


def _metadata_column(package: Any) -> dict[str, Any]:
    payload = dict(package.partitions["train"].payloads)["train/metadata"]
    assert isinstance(payload, MetadataTable)
    return _metadata_body(payload.frame)["columns"][0]


def _metadata_values(package: Any) -> list[Any]:
    return _metadata_column(package)["values"]


def _m(package: Any) -> list[Any]:
    return _metadata_values(package)


def _typed(expected: list[Any], actual: list[Any]) -> None:
    assert [type(v) for v in expected] == [type(v) for v in actual] and expected == actual, (expected, actual)


def test_a_float_input_contributes_floats_even_when_another_input_retains_huge_integers(tmp_path: Path) -> None:
    both = ("m,x,y\n%s,1.5,1.0\n3,2.5,2.0\n", "m,x,y\n1.5,3.5,3.0\n4,4.5,4.0\n")
    package = _concat(tmp_path, both[0] % INT, both[1])
    _typed([9007199254740993, 3, 1.5, 4.0], _m(package))  # the second input's 4 is the float 4.0 (Rust output, neg1 concat-retained)
    reversed_order = _concat(tmp_path, both[1], both[0] % INT)
    _typed([1.5, 4.0, 9007199254740993, 3], _m(reversed_order))


def test_a_text_input_cannot_erase_the_numeric_authority_of_another_input(tmp_path: Path) -> None:
    text, numeric = "m,x,y\n3,1.5,1.0\nabc,2.5,2.0\n", f"m,x,y\n{X},3.5,3.0\n4,4.5,4.0\n"
    with pytest.raises(NumberDomainError, match="outside the supported range"):
        _concat(tmp_path, text, numeric)
    with pytest.raises(NumberDomainError, match="outside the supported range"):
        _concat(tmp_path, numeric, text)
    package = _concat(tmp_path, text, "m,x,y\n7,3.5,3.0\n4,4.5,4.0\n")  # a numeric-only twin without the bad token stays valid
    _typed(["3", "abc", 7, 4], _m(package))


@pytest.mark.parametrize("role", ["targets", "weights"])
def test_a_text_input_cannot_erase_numeric_authority_in_numeric_roles(tmp_path: Path, role: str) -> None:
    for first, second in (("m,x,y\n3,1.5,1.0\nabc,2.5,2.0\n", f"m,x,y\n{X},3.5,3.0\n4,4.5,4.0\n"), (f"m,x,y\n{X},3.5,3.0\n4,4.5,4.0\n", "m,x,y\n3,1.5,1.0\nabc,2.5,2.0\n")):
        (tmp_path / "a.csv").write_text(first)
        (tmp_path / "b.csv").write_text(second)
        columns = [{"role": "features", "select": ["x"]}, {"role": role, "select": ["m"]}, {"role": "ignore", "select": ["y"]}]
        spec = {"sources": [{"id": "d", "role": "mixed", "input": ["a.csv", "b.csv"], "merge": "concat_samples", "columns": columns}], "params": {"delimiter": ","}}
        with pytest.raises(NumberDomainError, match="outside the supported range"):
            nio.to_dataset_package(spec, base_dir=tmp_path)


def test_per_input_classification_for_integer_inputs_and_numeric_roles(tmp_path: Path) -> None:
    package = _concat(tmp_path, "m,x,y\n3,1.5,1.0\n4,2.5,2.0\n", "m,x,y\n1.5,3.5,3.0\n2.5,4.5,4.0\n")
    # No input needs retention, refusal, text or bool authority: the stacked column is the plain MERGED float column (Rust, pandas and
    # final15 agree; the former [3, 4, 1.5, 2.5] expectation encoded the C-cause defect found by the Opus 5.6 re-check, see test_concat_typed_cells.py).
    _typed([3.0, 4.0, 1.5, 2.5], _m(package))
    features = dict(package.partitions["train"].payloads)["train/x0"].matrix
    assert features.values.ravel().tolist() == [1.5, 2.5, 3.5, 4.5]


# --------------------------------------------------------------------------- #
# Item 6: the NA audit of [u64max, NA] metadata/ID columns                       #
# --------------------------------------------------------------------------- #
_NA_AUDIT = [{"source": "t", "na": {"strategy": "ignore", "na_detected": True, "na_count": 1, "na_samples": 1, "na_features": 1, "removed_samples": [], "removed_features": [], "fill_method": None, "na_preserved": True}}]  # Rust anchor


def test_u64max_with_na_records_the_exact_na_audit_in_metadata_mode(tmp_path: Path) -> None:
    (tmp_path / "d.csv").write_text("m,x,y\n18446744073709551615,1.5,1.0\n,2.5,2.0\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert package.audits == _NA_AUDIT
    body = _metadata_column(package)
    assert body["dtype"] == "numeric"
    _typed([18446744073709551615, None], body["values"])


def test_u64max_with_na_records_the_exact_na_audit_in_id_mode_and_projection_refuses_separately(tmp_path: Path) -> None:
    (tmp_path / "d.csv").write_text("id,x,y\n18446744073709551615,1.5,1.0\n,2.5,2.0\n")
    spec = {
        "sample_index": {"by": "id", "key": "id"},
        "sources": [{"id": "t", "input": "d.csv", "key": "id", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}],
        "params": {"delimiter": ",", "na": {"policy": "ignore"}},
    }
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert package.audits == _NA_AUDIT, "the package keeps both rows and the audit"
    assert package.partitions["train"].n_samples == 2
    with pytest.raises(NumericContractError, match="Missing projection identity"):
        package.executed_projection(["t"])


@pytest.mark.parametrize("token", ["NA", "N/A", "", "null", "NaN", "None", "n/a", "#N/A", "<NA>"])
def test_na_spellings_beside_a_retained_unsigned_integer_are_all_missing(tmp_path: Path, token: str) -> None:
    (tmp_path / "d.csv").write_text(f"m,x,y\n18446744073709551615,1.5,1.0\n{token},2.5,2.0\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert package.audits[0]["na"]["na_count"] == 1
    _typed([18446744073709551615, None], _metadata_values(package))


def test_genuine_text_with_a_na_token_is_unchanged(tmp_path: Path) -> None:
    (tmp_path / "d.csv").write_text("m,x,y\nabc,1.5,1.0\nNA,2.5,2.0\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    _typed(["abc", None], _metadata_values(package))
    assert json.dumps(package.audits[0]["na"]["na_count"]) == "1"
