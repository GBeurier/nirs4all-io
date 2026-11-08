# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Original cells reach the numeric-role selection: spreadsheets, list blocks, axis coordinates and numeric headers.

Four places used to convert known original values (text, integers) to float64 BEFORE the owner's storage
selection could look at them:

* ``.xlsx`` text cells (pandas' text parser turned ``1e-400`` into 0.0 and rounded large integer text);
* the public ``ExactBlock`` primitive over raw lists (a mixed integer/float list was cast to float64 first);
* integer coordinates of the public ``Assembled.feature_axes`` (only floats were accepted);
* numeric feature headers (``float(header)`` erased integer/decimal text).

Every case below runs through real files / the real public primitive, with genuine already decoded F64 values
(including the sign of zero) kept as controls.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.numeric import CONSUMER_SPECTRODATASET, ExactBlock, NumericContractError
from nirs4all_io.materialize.package import DatasetPackage, FeatureMatrix, MetadataTable
from nirs4all_io.spec.enums import SpecError
from nirs4all_io.strict_json import NumberDomainError

openpyxl = pytest.importorskip("openpyxl")

N = 2**54 + 2**30 + 1


def _xlsx(path: Path, rows: list[list[Any]], sheet: str | None = None, extra: dict[str, list[list[Any]]] | None = None) -> Path:
    book = openpyxl.Workbook()
    sheet_obj = book.active
    if sheet:
        sheet_obj.title = sheet
    for row in rows:
        sheet_obj.append(row)
    for title, other in (extra or {}).items():
        extra_sheet = book.create_sheet(title)
        for row in other:
            extra_sheet.append(row)
    book.save(path)
    return path


def _spec(columns: dict[str, Any] | None = None, **top: Any) -> dict[str, Any]:
    return {"sources": [{"id": "table", "input": "data.xlsx", "columns": columns or {"features": "x", "targets": "y"}}], **top}


def _package(tmp_path: Path, rows: list[list[Any]], spec: dict[str, Any] | None = None, **options: Any) -> DatasetPackage:
    _xlsx(tmp_path / "data.xlsx", rows)
    return nio.to_dataset_package(spec or _spec(), base_dir=tmp_path, **options)


def _x(package: DatasetPackage) -> FeatureMatrix:
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload


def _x_hex(package: DatasetPackage) -> str:
    return _x(package).matrix.values_bytes().hex()


# --------------------------------------------------------------------------- #
# 4. Excel text cells                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("token", ["1e-400", "2e-324", "-1e-400"])
def test_excel_text_underflow_refuses_in_every_numeric_role(tmp_path: Path, token: str) -> None:
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, [["x", "y"], [token, 1]])  # features
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, [["x", "y"], [1, token]])  # targets
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, [["x", "y", "w"], [1, 2, token]], _spec({"features": "x", "targets": "y", "weights": "w"}))


def test_excel_true_text_zero_and_physical_zero_are_valid(tmp_path: Path) -> None:
    assert _x_hex(_package(tmp_path, [["x", "y"], ["0.000e-999", 1]])) == "0000000000000000"
    assert _x_hex(_package(tmp_path, [["x", "y"], ["-0.000e-999", 1]])) == "0000000000000080", "the true text zero keeps its sign"
    assert _x_hex(_package(tmp_path, [["x", "y"], [0.0, 1]])) == "0000000000000000", "a physical numeric zero is a valid observation"
    assert _x_hex(_package(tmp_path, [["x", "y"], ["4.9e-324", 1]])) == "0100000000000000", "the minimum subnormal text is a real value"


def test_excel_mixed_text_integer_keeps_the_original_integer_until_the_role_selection(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["x", "y"], [str(N), 1], ["0.5", 1]]
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, [["x", "y"], ["9007199254740993", 1], ["0.5", 1]])
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, rows, features="float64")
    direct = _package(tmp_path, rows, features="float32")
    assert _x_hex(direct) == "0100805a0000003f", "requested float32 rounds the original integer once, directly (upper), not 0000805a"
    assert _x(direct).matrix.precision.dtype == "float32"
    exact = _package(tmp_path, [["x", "y"], ["18014398509481984", 1], ["0.5", 1]])
    assert _x(exact).matrix.values.ravel().tolist() == [float(2**54), 0.5], "an exactly representable large integer is valid"
    with pytest.raises(NumberDomainError):
        _package(tmp_path, [["x", "y"], ["18446744073709551617", 1], ["0.5", 1]])


def test_excel_genuine_numeric_cells_keep_their_bits(tmp_path: Path) -> None:
    values = [0.1, 1e-300, 5e-324, 123.456, -2.5, 0.0]
    package = _package(tmp_path, [["x", "y"], *[[v, 1] for v in values]])
    assert _x(package).matrix.values.ravel().tolist() == values
    assert _x(package).matrix.precision.dtype == "float64"
    ints = _package(tmp_path, [["x", "y"], [1, 2], [3, 4]])
    assert _x(ints).matrix.values.ravel().tolist() == [1.0, 3.0] and _x(ints).matrix.precision.source_dtype is None


def test_excel_metadata_and_class_labels_stay_ordinary_values(tmp_path: Path) -> None:
    rows: list[list[Any]] = [["x", "y", "note", "count"], [1, "a", "1e-400", 5], [2, "b", "north", 6]]
    spec = _spec({"features": "x", "targets": "y", "metadata": ["note", "count"]})
    package = _package(tmp_path, rows, spec)
    metadata = dict(package.partitions["train"].payloads)["train/metadata"]
    assert isinstance(metadata, MetadataTable)
    assert metadata.frame["note"].tolist() == ["1e-400", "north"], "metadata text is never converted"
    assert metadata.frame["count"].tolist() == [5, 6] and metadata.frame["count"].dtype.kind == "i", "integer metadata is not turned into text"
    assembled = nio.load(spec, target="assembled", base_dir=tmp_path)
    assert assembled.blocks["train"].metadata["count"].dtype.kind == "i"


def test_excel_na_blank_and_text_rules_are_unchanged(tmp_path: Path) -> None:
    ignore = _spec(params={"na": {"policy": "ignore"}})
    package = _package(tmp_path, [["x", "y"], [str(N), 1], [None, 2], [0.5, 3]], ignore, features="float32")
    matrix = _x(package).matrix
    assert matrix.validity.ravel().tolist() == [True, False, True] and matrix.values_bytes().hex() == "0100805a" + "00000000" + "0000003f"
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, [["x", "y"], [str(2**53 + 1), 1], ["NA", 2]], ignore)
    with pytest.raises(Exception, match="NA values detected"):
        _package(tmp_path, [["x", "y"], [1, 1], [None, 2]])


def test_excel_configured_sheet_and_usecols_keep_the_same_rules(tmp_path: Path) -> None:
    _xlsx(tmp_path / "data.xlsx", [["x", "y"], [1, 1]], extra={"second": [["skip", "x", "y"], ["z", "1e-400", 1]]})
    spec = {"sources": [{"id": "table", "input": "data.xlsx", "columns": {"features": "x", "targets": "y"}, "params": {"format": {"sheet_name": "second", "usecols": [1, 2]}}}]}
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.to_dataset_package(spec, base_dir=tmp_path)
    _xlsx(tmp_path / "data.xlsx", [["x", "y"], [1, 1]], extra={"second": [["skip", "x", "y"], ["z", "1e-300", 1]]})
    assert _x_hex(nio.to_dataset_package(spec, base_dir=tmp_path)) == struct.pack("<d", 1e-300).hex()


# --------------------------------------------------------------------------- #
# 3. ExactBlock over raw lists                                                 #
# --------------------------------------------------------------------------- #
def _block(matrix: Any, source: str | None = None, request: str | None = None, consumer: str | None = None) -> ExactBlock:
    return ExactBlock.from_matrix(matrix, source_dtype=source, request=request, consumer=consumer, feature=True)


def test_public_exact_block_refuses_an_inexact_integer_in_a_raw_list() -> None:
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        _block([[2**53 + 1, 0.5]])
    assert _block([[2**54, 0.5]]).values.ravel().tolist() == [float(2**54), 0.5]
    with pytest.raises(NumericContractError, match="cannot represent exactly"):
        _block(((2**53 + 1, 0.5),))


def test_public_exact_block_rounds_the_original_integer_once_for_requested_float32() -> None:
    assert _block([[N, 0.5]], request="float32").values_bytes().hex() == "0100805a0000003f", "upper neighbour, not the lower 0000805a"
    assert _block([[N, 0.5]], source="int64", request="float32").values_bytes().hex() == "0100805a0000003f"
    assert _block(np.array([[N, 1]], dtype=object), request="float32").values_bytes().hex() == "0100805a0000803f"
    consumer = _block([[N, 0.5]], consumer=CONSUMER_SPECTRODATASET)
    assert consumer.dtype == "float32" and consumer.values_bytes().hex() == "0100805a0000003f"
    # an integer ARRAY converts directly as well (never integer -> F64 -> F32)
    assert _block(np.array([[N]], dtype=np.int64), request="float32").values_bytes().hex() == "0100805a"


def test_public_exact_block_follows_source_declaration_first() -> None:
    declared64 = {"source": "float64"}
    for request in (None, "float32", "float64"):
        with pytest.raises(NumericContractError, match="cannot represent exactly"):
            _block([[N, 0.5]], request=request, **declared64)
    assert _block([[2**54, 0.5]], request="float32", **declared64).dtype == "float32"
    declared32 = _block([[N, 0.5]], source="float32", request="float64")
    assert declared32.dtype == "float64" and declared32.values_bytes().hex() == "0000002000005043000000000000e03f", "float32-declared: normalized directly once, then widened"
    assert _block([[N, 0.5]], source="float32").values_bytes().hex() == "0100805a0000003f"


def test_public_exact_block_keeps_host_float64_bits_and_a_carried_precision_recovers_nothing() -> None:
    host = np.array([[float(N), 0.5]], dtype=np.float64)  # a typed F64 array: its bits are already the host's rounding
    assert _block(host, request="float32").values_bytes().hex() == "0000805a0000003f"
    assert _block(host).values.ravel().tolist() == [float(N), 0.5]
    rounded = _block([[N, 0.5]], request="float32")
    with pytest.raises(NumericContractError, match="no longer matches"):
        ExactBlock.with_precision(np.array([[float(N), 0.5]]), rounded.precision, verify=True)
    again = ExactBlock.with_precision(rounded.to_ir(), rounded.precision, verify=True)
    assert again.values_bytes() == rounded.values_bytes()


def test_package_ir_path_keeps_the_list_selection(tmp_path: Path) -> None:
    """The in-memory public path with a raw integer/float list reaches the same selection."""
    with pytest.raises(Exception, match="cannot represent exactly"):
        nio.to_dataset_package({"X": [[2**53 + 1, 0.5]], "y": np.array([1.0])})
    direct = nio.to_dataset_package({"X": [[N, 0.5]], "y": np.array([1.0])}, features="float32")
    assert _x_hex(direct) == "0100805a0000003f"


# --------------------------------------------------------------------------- #
# 1. Public IR numeric axis coordinates                                        #
# --------------------------------------------------------------------------- #
def _axis_package(coordinates: list[Any]) -> DatasetPackage:
    assembled = nio.load({"X": np.ones((1, len(coordinates))), "y": np.array([1.0])}, target="assembled")
    assembled.blocks["train"].feature_axes = [[{"name": "wavelength", "unit": "nm", "coordinates": coordinates}]]
    return DatasetPackage.from_assembled(assembled)


def _axes(package: DatasetPackage) -> list[dict[str, Any]]:
    return _x(package).axes


def test_representable_integer_axis_coordinates_are_accepted_and_inexact_ones_refuse() -> None:
    assert _axes(_axis_package([2**54]))[0]["coordinates"] == [float(2**54)]
    assert _axes(_axis_package([2**53, -(2**53), 0, 5]))[0]["coordinates"] == [float(2**53), -float(2**53), 0.0, 5.0]
    for inexact in (2**53 + 1, N, -(2**53 + 1)):
        with pytest.raises(NumericContractError, match="cannot represent exactly"):
            _axis_package([inexact])
    with pytest.raises(NumericContractError, match="outside the supported range"):
        _axis_package([2**64])


def test_genuine_float_coordinates_keep_bits_and_strings_stay_valid() -> None:
    coordinates = _axes(_axis_package([-0.0, 0.1, 5e-324, 1100.5]))[0]["coordinates"]
    assert [struct.pack("<d", c).hex() for c in coordinates] == [struct.pack("<d", c).hex() for c in (-0.0, 0.1, 5e-324, 1100.5)]
    assert _axes(_axis_package(["a", "b"]))[0]["coordinates"] == ["a", "b"]
    assert _axes(_axis_package([1, 2.5]))[0]["coordinates"] == [1.0, 2.5], "an integer beside a float is the same numeric kind"


def test_axis_kind_mixes_bools_and_non_finite_values_still_refuse() -> None:
    for bad in ([1, "a"], ["a", 1.0], [True], [float("nan")], [float("inf")]):
        with pytest.raises(NumericContractError):
            _axis_package(bad)


def test_integer_axis_coordinates_round_trip_the_full_wire() -> None:
    package = _axis_package([2**54, 1100.5])
    again = DatasetPackage.from_full_dict(package.to_full_dict())
    assert _axes(again) == _axes(package) and _axes(again)[0]["coordinates"][0] == float(2**54)
    assert again.to_full_dict() == package.to_full_dict()


# --------------------------------------------------------------------------- #
# 2. Numeric header coordinates                                                #
# --------------------------------------------------------------------------- #
def _header_package(tmp_path: Path, headers: list[str]) -> DatasetPackage:
    (tmp_path / "data.csv").write_text(";".join([*headers, "y"]) + "\n" + ";".join(["1"] * len(headers) + ["2"]) + "\n", encoding="utf-8")
    spec = {"sources": [{"id": "t", "input": "data.csv", "columns": {"features": list(headers), "targets": "y"}}]}
    return nio.to_dataset_package(spec, base_dir=tmp_path)


def test_integer_header_text_is_selected_exactly(tmp_path: Path) -> None:
    assert _axes(_header_package(tmp_path, ["18014398509481984", "9007199254740992"]))[0]["coordinates"] == [float(2**54), float(2**53)]
    with pytest.raises(SpecError, match="cannot represent exactly"):
        _header_package(tmp_path, ["9007199254740993", "9007199254740994"])
    with pytest.raises(NumberDomainError):
        _header_package(tmp_path, ["18446744073709551617", "5"])


def test_decimal_header_text_underflow_refuses_and_true_zero_is_valid(tmp_path: Path) -> None:
    with pytest.raises(NumberDomainError, match="underflows"):
        _header_package(tmp_path, ["1e-400", "1"])
    assert _axes(_header_package(tmp_path, ["0.000e-999", "-0.000e-999", "1"]))[0]["coordinates"] == [0.0, -0.0, 1.0]
    zero = _axes(_header_package(tmp_path, ["0.000e-999", "-0.000e-999"]))[0]["coordinates"]
    assert [struct.pack("<d", c).hex() for c in zero] == ["0000000000000000", "0000000000000080"]


def test_ordinary_headers_and_labels_are_unchanged(tmp_path: Path) -> None:
    assert _axes(_header_package(tmp_path, ["1100", "1102.5", "1105"]))[0]["coordinates"] == [1100.0, 1102.5, 1105.0]
    assert _axes(_header_package(tmp_path, ["w1100", "w1102"]))[0]["coordinates"] is None, "label headers declare no coordinates"
    assert _axes(_header_package(tmp_path, ["1100", "w1102"]))[0]["coordinates"] is None
    (tmp_path / "idx.csv").write_text("1;2;3\n4;5;6\n", encoding="utf-8")
    spec = {"sources": [{"id": "t", "input": "idx.csv", "columns": {"features": [0, 1], "targets": 2}, "params": {"has_header": False}}]}
    assert _axes(nio.to_dataset_package(spec, base_dir=tmp_path))[0]["coordinates"] is None, "generated index headers declare no coordinates"
