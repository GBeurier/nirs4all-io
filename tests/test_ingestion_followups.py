# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Target order and original spreadsheet Float authority at public ingestion."""
from __future__ import annotations

import datetime
import hashlib
import io
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import openpyxl
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize import loaders


def _sheet(path: Path, rows: list[list[Any]]) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    book.save(path)
    book.close()
    # openpyxl's writer formats numbers with 16 significant digits. These controls
    # require the intended physical INTEGER token, rather than that lossy fixture.
    with zipfile.ZipFile(path) as archive:
        members = [(item, archive.read(item.filename)) for item in archive.infolist()]
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    physical = ET.fromstring(dict((item.filename, data) for item, data in members)["xl/worksheets/sheet1.xml"])
    cells = {cell.attrib["r"]: cell for cell in physical.iter(namespace + "c")}
    for r, row in enumerate(rows, 1):
        for c, value in enumerate(row, 1):
            if type(value) is int or (type(value) is float and value.is_integer() and abs(value) >= 2**53):
                node = cells[f"{openpyxl.utils.get_column_letter(c)}{r}"].find(namespace + "v")
                assert node is not None
                node.text = str(value)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for item, data in members:
            archive.writestr(item, ET.tostring(physical) if item.filename == "xl/worksheets/sheet1.xml" else data)
    path.write_bytes(buffer.getvalue())


def _spec(**params: Any) -> dict[str, Any]:
    return {"sources": [{"id": "t", "input": "d.xlsx", "columns": [{"role": "features", "select": ["x"]}, {"role": "metadata", "select": ["m"]}]}], "params": {"na": {"policy": "ignore"}, **params}}


def _values(package: Any) -> list[Any]:
    return package.to_full_dict()["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]["values"]


@pytest.mark.parametrize("headers", [["a", "b"], ["b", "a"], ["c", "a", "b"]])
def test_targets_follow_final_physical_order(tmp_path: Path, headers: list[str]) -> None:
    (tmp_path / "d.csv").write_text("x," + ",".join(headers) + "\n1.5," + ",".join(str(i + 1) for i in range(len(headers))) + "\n")
    spec = {"sources": [{"id": "t", "input": "d.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": sorted(headers, reverse=True)}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    block = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    assert block.y_headers == headers
    np.testing.assert_array_equal(block.y, np.arange(1, len(headers) + 1).reshape(1, -1))


@pytest.mark.parametrize("remove", [False, True])
def test_physical_float_and_exact_peer_keep_loaded_column_mode(tmp_path: Path, remove: bool) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1], [None if remove else 2.5, 1e30]])
    package = nio.to_dataset_package(_spec(na={"policy": "remove_sample" if remove else "ignore"}), base_dir=tmp_path)
    values = _values(package)
    assert type(values[0]) is float and values[0] == 1.0
    assert package.partitions["train"].n_samples == (1 if remove else 2)
    if not remove:
        assert type(values[1]) is float and struct.pack(">d", values[1]).hex() == "46293e5939a08cea"


@pytest.mark.parametrize("peer", [9007199254740993, True, "abc"])
def test_physical_float_never_retypes_original_peer(tmp_path: Path, peer: Any) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, peer], [2.5, 1e30]])
    values = _values(nio.to_dataset_package(_spec(), base_dir=tmp_path))
    expected = float(peer) if type(peer) is bool else peer
    assert type(values[0]) is type(expected) and values[0] == expected
    assert struct.pack(">d", values[1]).hex() == "46293e5939a08cea"


def test_integer_lexemes_keep_exact_metadata_and_distinct_ids(tmp_path: Path) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 9007199254740993], [2.5, 9007199254740992]])
    spec = _spec()
    spec["sample_index"] = {"by": "id", "key": "m"}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    values = _values(package)
    assert all(type(v) is int for v in values) and values == [9007199254740993, 9007199254740992]
    assert package.executed_projection(["t"]).identity.sample_ids == ["9007199254740993", "9007199254740992"]


def test_out_of_domain_integer_is_not_a_physical_float(tmp_path: Path) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1], [2.5, 1000000000000000000000000000000]])
    package = nio.to_dataset_package(_spec(), base_dir=tmp_path)
    with pytest.raises(ValueError, match="integer|Integer"):
        package.to_full_dict()


def test_physical_text_zero_and_large_integer_remain_literal_metadata(tmp_path: Path) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, "-0"], [2.5, "9007199254740993"], [3.5, "1.5"]])
    assert _values(nio.to_dataset_package(_spec(), base_dir=tmp_path, features="float32")) == ["-0", "9007199254740993", "1.5"]


@pytest.mark.parametrize("selection", [{"sheet_name": 0, "header": 0}, {"sheet_name": 0, "header": None}, {"sheet_name": "second", "header": 0}, {"sheet_name": 1, "header": None}, {"sheet_name": 0, "header": 0, "skiprows": 1}, {"sheet_name": 0, "header": 0, "skiprows": [0]}, {"sheet_name": 0, "header": 0, "skiprows": 1, "usecols": "A:C"}, {"sheet_name": 0, "header": None, "usecols": [0, 2]}, {"sheet_name": 0, "header": 0, "skiprows": 1, "usecols": ["x", "m"]}])
def test_pandas_workbook_selection_matches_path(tmp_path: Path, selection: dict[str, Any]) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "first"
    for row in [["title"], ["x", "d", "m", "u"], [1.5, datetime.date(2020, 1, 2), 1, None], [], [2.5, datetime.date(2020, 1, 3), 1e30, "k"]]:
        sheet.append(row)
    sheet.merge_cells("A1:C1")
    sheet.row_dimensions[3].hidden = True
    second = book.create_sheet("second")
    second.append(["x", "m"])
    second.append([1, 2])
    path = tmp_path / "grid.xlsx"
    book.save(path)
    book.close()
    book = openpyxl.load_workbook(path, data_only=True, keep_links=False)
    try:
        left = pd.read_excel(path, **selection, dtype=object, na_filter=False, keep_default_na=False)
        right = pd.read_excel(book, engine="openpyxl", **selection, dtype=object, na_filter=False, keep_default_na=False)
        assert left.shape == right.shape and list(left.columns) == list(right.columns)
        assert [repr(v) for v in left.to_numpy().flat] == [repr(v) for v in right.to_numpy().flat]
    finally:
        book.close()


@pytest.mark.parametrize("option", ["usecols", "skiprows"])
def test_selected_physical_float_uses_declared_pandas_selection(tmp_path: Path, option: str) -> None:
    rows: list[list[Any]] = [["junk", "x", "m"], ["q", 1.5, 1], ["q", 2.5, 1e30]] if option == "usecols" else [["title"], ["x", "m"], [1.5, 1], [2.5, 1e30]]
    _sheet(tmp_path / "d.xlsx", rows)
    values = _values(nio.to_dataset_package(_spec(format={option: "B:C" if option == "usecols" else 1}), base_dir=tmp_path))
    assert all(type(v) is float for v in values) and values == [1.0, 1e30]


def test_exact_registry_avoids_literal_and_unselected_sheet_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = UUID(int=0)
    namespace = "nirs4allfloat" + fixed.hex
    literal = namespace + "v0"
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    for row in [["x", "m"], [1.5, literal], [2.5, 1e30]]:
        sheet.append(row)
    book.create_sheet("unselected").append([namespace + "xv0"])
    path = tmp_path / "d.xlsx"
    book.save(path)
    book.close()
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(loaders, "uuid4", lambda: fixed)
    values = _values(nio.to_dataset_package(_spec(), base_dir=tmp_path))
    assert type(values[0]) is str and values[0] == literal
    assert type(values[1]) is float and struct.pack(">d", values[1]).hex() == "46293e5939a08cea"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


@pytest.mark.parametrize("failure", ["read", "shape", "column"])
def test_tagged_provenance_failures_refuse_outside_optional_raw_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1], [2.5, 1e30]])
    reader = pd.read_excel

    def selected(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, openpyxl.Workbook):
            if failure == "read":
                raise RuntimeError("tagged-reader-control")
            frame = reader(value, *args, **kwargs)
            if failure == "shape":
                return frame.iloc[:-1]
            frame.columns = ["m", "x"]
            return frame
        return reader(value, *args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", selected)
    with pytest.raises(loaders.LoaderError, match="physical Excel"):
        nio.to_dataset_package(_spec(), base_dir=tmp_path)


def test_physical_float_does_not_relax_classification_label_cap(tmp_path: Path) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1], [2.5, 1e30]])
    spec = _spec()
    spec["task_type"] = "multiclass"
    spec["sources"][0]["columns"][1]["role"] = "targets"
    with pytest.raises(ValueError):
        nio.to_dataset_package(spec, base_dir=tmp_path)




@pytest.mark.parametrize("peer", ["abc", "5", " ", datetime.datetime(2026, 1, 2), datetime.time(1, 2, 3), datetime.timedelta(hours=30)])
def test_correction_loaded_text_peer_preserves_integer(tmp_path: Path, peer: Any) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1e30], [2.5, 7], [3.5, peer]])
    values = _values(nio.to_dataset_package(_spec(), base_dir=tmp_path))
    assert type(values[0]) is float and struct.pack(">d", values[0]).hex() == "46293e5939a08cea"
    assert type(values[1]) is int and values[1] == 7


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("key", [False, True])
def test_correction_removed_text_keeps_loaded_integer_authority(tmp_path: Path, first: bool, key: bool) -> None:
    rows: list[list[Any]] = [[1.5, 9007199254740992.0], [2.5, 7]]
    rows.insert(0 if first else len(rows), [None, "abc"])
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *rows])
    spec = _spec(na={"policy": "remove_sample"})
    if key:
        spec["sample_index"] = {"by": "id", "key": "m"}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    values = _values(package)
    assert type(values[1]) is int and values == [9007199254740992.0, 7]
    if key:
        assert package.executed_projection(["t"]).identity.sample_ids == ["9007199254740992.0", "7"]
    # Rust's retained String dtype is a separately recorded package-label gap.


@pytest.mark.parametrize("peers", [[7, True], [7, False], [True, False]])
@pytest.mark.parametrize("remove", [False, True])
def test_correction_physical_bool_never_becomes_missing(tmp_path: Path, peers: list[Any], remove: bool) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1e30], [2.5, peers[0]], [3.5, peers[1]]])
    package = nio.to_dataset_package(_spec(na={"policy": "remove_sample" if remove else "ignore"}), base_dir=tmp_path)
    values = _values(package)
    expected = [1e30, *[float(v) if type(v) in (int, bool) else v for v in peers]]
    assert len(values) == package.partitions["train"].n_samples == 3
    assert [type(v) for v in values] == [type(v) for v in expected] and values == expected
    # Bool-vs-Rust-float is Q1; this verifies no physical observation is lost.


def _correction_workbook_parts(path: Path, *, wrong_dimension: bool = False) -> None:
    """Tiny actual XML declarations; never ask a writer to expand a merge."""
    with zipfile.ZipFile(path) as archive:
        parts = [(info, archive.read(info.filename)) for info in archive.infolist()]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for info, data in parts:
            if info.filename.startswith("xl/worksheets/sheet"):
                ref = "D1:D1048576" if info.filename.endswith("sheet1.xml") else "A1:CV200000"
                data = data.replace(b"</worksheet>", f'<mergeCells count="1"><mergeCell ref="{ref}"/></mergeCells></worksheet>'.encode())
                if wrong_dimension and info.filename.endswith("sheet1.xml"):
                    data, count = re.subn(rb'<dimension ref="[^"]+"\s*/>', b'<dimension ref="A1:B2"/>', data)
                    assert count == 1
            archive.writestr(info, data)
    path.write_bytes(buffer.getvalue())


@pytest.mark.parametrize("selection", [0, "selected", None, ["selected"]])
@pytest.mark.parametrize("wrong_dimension", [False, True])
@pytest.mark.parametrize("far", [False, True])
def test_correction_readonly_merge_stream_bounds_and_borrowed_lifetime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: Any, wrong_dimension: bool, far: bool, chartsheet: bool = False) -> None:
    from openpyxl.worksheet._read_only import ReadOnlyWorksheet
    from openpyxl.worksheet._reader import WorksheetReader
    from openpyxl.worksheet.worksheet import Worksheet

    fixed = UUID(int=0)
    namespace = "nirs4allfloat" + fixed.hex
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "selected"
    for row in [["x", "m"], [1.5, 1], [2.5, 2], [3.5, 1e30]]:
        sheet.append(row)
    sheet.cell(1, 3, namespace + "v0")  # selected but excluded header must force retry
    if far:
        sheet.cell(500, 30, "excluded far physical Text")
    # An unreachable collision must not trigger unselected row streaming.
    book.create_sheet("unselected").cell(20000, 100, namespace + "v0")
    if chartsheet:
        from openpyxl.chart import BarChart
        book.create_chartsheet("chart", index=0).add_chart(BarChart())
    path = tmp_path / "d.xlsx"
    book.save(path)
    book.close()
    _correction_workbook_parts(path, wrong_dimension=wrong_dimension)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    observations = {"bind": 0, "normal_cell": 0, "selected_rows": 0, "unselected_rows": 0, "borrowed_reads": 0, "in_helper": False}
    original_rows = ReadOnlyWorksheet._cells_by_row
    read_excel = pd.read_excel

    def forbidden_bind(reader: Any) -> None:
        observations["bind"] += 1
        pytest.fail("normal merge binding before allocation is a resource regression")

    def forbidden_cell(sheet: Any, *args: Any, **kwargs: Any) -> None:
        observations["normal_cell"] += 1
        pytest.fail("normal worksheet cell allocation is forbidden")

    def observed_rows(sheet: Any, *args: Any, **kwargs: Any) -> Any:
        name = "unselected_rows" if sheet.title == "unselected" else "selected_rows"
        if observations["in_helper"]:
            observations[name] += 1
            assert observations["unselected_rows"] == 0
        yield from original_rows(sheet, *args, **kwargs)

    class ObservedExcelFile(pd.ExcelFile):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            observations["in_helper"] = True
            self.observed_dimensions = [(s._max_row, s._max_column) for s in self.book.worksheets]
        def close(self) -> None:
            assert self.book._archive.fp is not None, "borrowed close must not close the source archive"
            assert [(s._max_row, s._max_column) for s in self.book.worksheets] == self.observed_dimensions
            assert all("_cells_by_row" not in s.__dict__ for s in self.book.worksheets)
            super().close()
            observations["in_helper"] = False

    def observed_read(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, openpyxl.Workbook):
            assert value.read_only
            observations["borrowed_reads"] += 1
            # The public values_only form must preserve the stream wrapper contract.
            rows = list(value["selected"].iter_rows(values_only=True))
            assert len(rows) == (500 if far else 4)
            assert rows[3][1] != namespace + "v0"
        return read_excel(value, *args, **kwargs)

    monkeypatch.setattr(WorksheetReader, "bind_merged_cells", forbidden_bind)
    monkeypatch.setattr(Worksheet, "cell", forbidden_cell)
    monkeypatch.setattr(ReadOnlyWorksheet, "_cells_by_row", observed_rows)
    monkeypatch.setattr(pd, "ExcelFile", ObservedExcelFile)
    monkeypatch.setattr(pd, "read_excel", observed_read)
    monkeypatch.setattr(loaders, "uuid4", lambda: fixed)
    package = nio.to_dataset_package(_spec(format={"sheet_name": selection, "usecols": "A:B"}, na={"policy": "remove_sample" if far else "ignore"}), base_dir=tmp_path)
    values = _values(package)
    assert values == [1.0, 2.0, 1e30] and all(type(v) is float for v in values)
    assert struct.pack(">d", values[2]).hex() == "46293e5939a08cea"
    assert observations["bind"] == observations["normal_cell"] == observations["unselected_rows"] == 0
    assert observations["borrowed_reads"] == 1 and observations["selected_rows"] <= 8
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_correction_chartsheet_index_uses_worksheets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_correction_readonly_merge_stream_bounds_and_borrowed_lifetime(tmp_path, monkeypatch, 0, True, False, True)


@pytest.mark.parametrize("labels", [[" ", "x"], ["\t", "x"], ["\u00a0", "x"], ["\u3000", "\u2003", "\u0085"], [" leading", "trailing ", " "], ["é", "e\u0301"], [" ", "  "]])
def test_label_content_public_package_is_exact(tmp_path: Path, labels: list[str]) -> None:
    import csv

    from nirs4all_io.materialize.package import DatasetPackage, TargetTable
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "label"])
        writer.writerows([["S" + str(i), i + .5, label] for i, label in enumerate(labels)])
    spec = {"sources": [{"id": "s", "input": "labels.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["label"]}, {"role": "metadata", "select": ["id"]}]}], "sample_index": {"by": "id", "key": "id"}, "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    wire = package.to_full_dict()
    payload = next(value for name, value in package.partitions["train"].payloads if name == "train/y")
    assert isinstance(payload, TargetTable)
    block = payload.columns[0].block
    assert block.values[:, 0].tolist() == labels and block.validity[:, 0].tolist() == [True] * len(labels)
    assert block.values_bytes() == b"".join(len(label.encode("utf-8")).to_bytes(8, "little") + label.encode("utf-8") for label in labels)
    assert DatasetPackage.from_full_dict(wire).to_full_dict()["manifest"]["root"] == wire["manifest"]["root"]
    assert package.executed_projection(["s"]).identity.sample_ids == ["S" + str(i) for i in range(len(labels))]


def _label_rust_literals() -> tuple[dict[str, Any], dict[str, Any]]:
    import json
    bank = json.loads((Path(__file__).parent / "fixtures/residual_ingestion_vectors.json").read_text())
    row = next(row for row in bank["cases"] if row["id"] == "ingest.correction.B2.space-target")
    return row["expected"]["assertions"][0]["value"], row["expected"]["package_assertions"][0]["value"]


def test_label_rust_wire_all_three_strict_readers_keep_root() -> None:
    from nirs4all_io.materialize.numeric import ExactBlock
    from nirs4all_io.materialize.package import DatasetPackage
    assembled, package = _label_rust_literals()
    target = package["partitions"]["train"]["payloads"]["train/y"]["value"]["columns"][0]["block"]
    assert ExactBlock.from_dict(target, feature=False).to_dict() == target
    assert nio.AssembledDataset.from_full_dict(assembled).to_full_dict() == assembled
    assert DatasetPackage.from_full_dict(package).to_full_dict() == package
    assert DatasetPackage.from_full_dict(package).to_full_dict()["manifest"]["root"] == package["manifest"]["root"]


@pytest.mark.parametrize("categories", [["a", "a"], [" ", " "], [""], ["a", ""], [None], [1], [True], ("a",), None])
def test_label_malformed_or_empty_codebooks_refuse(categories: Any) -> None:
    from nirs4all_io.materialize.package import _string_column
    with pytest.raises(ValueError):
        _string_column("label", np.array([0.0]), {"categories": categories}, None)


@pytest.mark.parametrize("code", [-1.0, .5, 2.0, float("inf")])
def test_label_code_bounds_stay_strict(code: float) -> None:
    from nirs4all_io.materialize.package import _string_column
    with pytest.raises(ValueError, match="code out of range"):
        _string_column("label", np.array([code]), {"categories": [" ", "x"]}, None)


def test_label_nan_code_and_default_na_remain_missing(tmp_path: Path) -> None:
    from nirs4all_io.materialize.package import _string_column
    block = _string_column("label", np.array([0.0, np.nan]), {"categories": [" "]}, None)
    assert block.values[:, 0].tolist() == [" ", ""] and block.validity[:, 0].tolist() == [True, False]
    (tmp_path / "labels.csv").write_text('x,label\n1.5,NA\n2.5,\n3.5,""\n4.5," "\n')
    spec = {"sources": [{"id": "s", "input": "labels.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["label"]}]}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    target = package.to_full_dict()["partitions"]["train"]["payloads"]["train/y"]["value"]["columns"][0]["block"]
    assert target["validity"] == [False, False, False, True]


@pytest.mark.parametrize("reader", ["constructor", "wire"])
def test_label_false_space_never_admitted(reader: str) -> None:
    from nirs4all_io.materialize.numeric import ExactBlock, PrecisionRecord
    precision = PrecisionRecord.string_label(None, None)
    with pytest.raises(ValueError):
        if reader == "constructor":
            ExactBlock(precision, np.array([[" "]], dtype=object), np.array([[False]]))
        else:
            wire = {"dtype": "string-label", "n_rows": 1, "n_cols": 1, "precision": precision.to_dict(), "data_bytes_hex": "010000000000000020", "validity": [False]}
            ExactBlock.from_dict(wire, feature=False)


@pytest.mark.parametrize("payload,valid", [("0000000000000000", True), ("0100000000000000ff", True), ("020000000000000020", True), ("00", False)])
def test_label_invalid_wire_refuses(payload: str, valid: bool) -> None:
    from nirs4all_io.materialize.numeric import ExactBlock, PrecisionRecord
    wire = {"dtype": "string-label", "n_rows": 1, "n_cols": 1, "precision": PrecisionRecord.string_label(None, None).to_dict(), "data_bytes_hex": payload, "validity": [valid]}
    with pytest.raises(ValueError):
        ExactBlock.from_dict(wire, feature=False)


def test_label_surrogate_is_not_utf8() -> None:
    from nirs4all_io.materialize.numeric import ExactBlock, PrecisionRecord
    block = ExactBlock.from_labels([["\ud800"]], PrecisionRecord.string_label(None, None))
    with pytest.raises(ValueError, match="UTF-8"):
        block.to_dict()


@pytest.mark.parametrize("codebook", [None, 1, True, ["a"], ("a",)])
def test_label_nondict_codebook_refuses(codebook: Any) -> None:
    from nirs4all_io.materialize.package import _string_column
    with pytest.raises(ValueError):
        _string_column("label", np.array([0.0]), codebook, None)


@pytest.mark.parametrize("role", ["metadata", "key", "remove"])
@pytest.mark.parametrize("missing", [None, "NA"])
def test_r12_tagged_numeric_bool_keeps_numeric_mode(tmp_path: Path, role: str, missing: Any) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1e30], [2.5, True], [3.5, missing]])
    spec = _spec(na={"policy": "remove_sample" if role == "remove" else "ignore"})
    if role == "key":
        spec["sample_index"] = {"by": "id", "key": "m"}
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    values = _values(package)
    assert len(values) == (2 if role == "remove" else 3)
    assert type(values[0]) is float and struct.pack(">d", values[0]).hex() == "46293e5939a08cea"
    assert type(values[1]) is float and values[1] == 1.0
    if role != "remove":
        assert values[2] is None
    if role == "key":
        assert package.partitions["train"].n_samples == 3


def test_r12_text_peer_keeps_physical_bool(tmp_path: Path) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], [1.5, 1e30], [2.5, True], [3.5, "abc"]])
    values = _values(nio.to_dataset_package(_spec(), base_dir=tmp_path))
    assert type(values[1]) is bool and values[1] is True and values[2] == "abc"


def _temporal_cells(kind: str) -> list[Any]:
    date = datetime.datetime(2026, 1, 2)
    date2 = datetime.datetime(2026, 1, 3, 4, 5, 6)
    duration = datetime.timedelta(hours=30)
    values: dict[str, list[Any]] = {"mix-date": [1e30, 7, date], "small-date": [5, 7, date], "date-first": [date, 7, 5.5], "all-date": [date, date2, date], "date-blank": [date, None, date2], "time": [5, 7, datetime.time(1, 2, 3)], "dur": [5, 7, duration], "all-dur": [duration] * 3, "text-ctl": [5, 7, "abc"], "bool-ctl": [5, 7, True]}
    return values[kind]


def _temporal_spec(role: str = "features", policy: str = "ignore") -> dict[str, Any]:
    cols = [{"role": "features", "select": ["x", "m"]}] if role == "features" else [{"role": "features", "select": ["x"]}, {"role": role, "select": ["m"]}]
    na: dict[str, Any] = {"policy": policy}
    if policy == "replace":
        na["fill"] = {"value": 0}
    return {"sources": [{"id": "t", "input": "d.xlsx", "columns": cols}], "params": {"na": na}}


@pytest.mark.parametrize("kind", ["mix-date", "small-date", "date-first", "all-date", "date-blank", "time", "dur", "all-dur", "text-ctl", "bool-ctl"])
@pytest.mark.parametrize("policy", ["ignore", "remove_sample", "replace", "remove_feature", "abort", "auto"])
def test_date_features_use_nonnumeric_masks_after_existing_na_stage(tmp_path: Path, kind: str, policy: str) -> None:
    cells = _temporal_cells(kind)
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    if kind == "date-blank" and policy in ("abort", "auto"):
        with pytest.raises(loaders.NAError):
            nio.load(_temporal_spec(policy=policy), target="assembled", base_dir=tmp_path)
        return
    block = nio.load(_temporal_spec(policy=policy), target="assembled", base_dir=tmp_path).blocks["train"]
    rows = [0, 2] if kind == "date-blank" and policy == "remove_sample" else [0, 1, 2]
    assert block.n_samples == len(rows)
    np.testing.assert_array_equal(block.X[0][:, 0], [i + 1.5 for i in rows])
    if kind == "date-blank" and policy == "remove_feature":
        assert block.feature_headers == [["x"]]
        return
    expected = []
    for row in rows:
        cell = cells[row]
        expected.append(0.0 if cell is None and policy == "replace" else float(cell) if isinstance(cell, (int, float, bool)) else np.nan)
    np.testing.assert_array_equal(np.isnan(block.X[0][:, 1]), np.isnan(expected))
    np.testing.assert_array_equal(block.X[0][:, 1], expected)
    if kind == "mix-date":
        assert struct.pack(">d", block.X[0][0, 1]).hex() == "46293e5939a08cea"


@pytest.mark.parametrize("kind", ["mix-date", "small-date", "date-first", "all-date", "date-blank", "dur", "all-dur"])
def test_date_named_processings_use_feature_temporal_mask(tmp_path: Path, kind: str) -> None:
    cells = _temporal_cells(kind)
    _sheet(tmp_path / "v.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    (tmp_path / "d.csv").write_text("x,m\n1.5,2.5\n2.5,3.5\n3.5,4.5\n")
    spec = _temporal_spec()
    spec["sources"][0]["input"] = "d.csv"
    spec["sources"][0]["variations"] = [{"name": "v", "input": "v.xlsx"}]
    spec["params"]["delimiter"] = ","
    block = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    assert block.processings[0][0][0] == "v"
    actual = block.processings[0][0][1]
    expected = [float(cell) if isinstance(cell, (int, float, bool)) else np.nan for cell in cells]
    np.testing.assert_array_equal(actual[:, 1], expected)
    np.testing.assert_array_equal(np.isnan(actual[:, 1]), np.isnan(expected))
    np.testing.assert_array_equal(block.X[0][:, 1], [2.5, 3.5, 4.5])


@pytest.mark.parametrize("series", [pd.Series(pd.to_datetime(["2026-01-02", None])), pd.Series(pd.to_timedelta(["30h", None])), pd.Series(pd.to_datetime(["2026-01-02", None], utc=True)), pd.Series([pd.Timestamp("2026-01-02", tz="UTC"), pd.Timedelta("30h"), pd.NaT, datetime.date(2026, 1, 2), datetime.time(1, 2), datetime.timedelta(hours=1), np.datetime64("2026-01-02"), np.timedelta64(1, "D"), 7, True, -0.0, "abc"], dtype=object)])
def test_date_direct_typed_temporals_never_become_epoch_features(series: pd.Series) -> None:
    frame = pd.DataFrame({"m": series})
    actual = loaders.coerce_numeric(frame)[:, 0]
    expected = [np.nan] * len(series) if series.dtype != object else [np.nan] * 8 + [7.0, 1.0, -0.0, np.nan]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    if series.dtype == object:
        assert struct.pack(">d", actual[-2]).hex() == "8000000000000000"


def test_date_parquet_temporal_feature_keeps_existing_mixed_precision_refusal(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"x": [1.5, 2.5], "m": pa.array([datetime.datetime(2026, 1, 2), None], type=pa.timestamp("us"))}), tmp_path / "d.parquet")
    spec = _temporal_spec()
    spec["sources"][0]["input"] = "d.parquet"
    with pytest.raises(ValueError, match="mixed"):
        nio.to_dataset_package(spec, base_dir=tmp_path)


# These three target pins are intentionally NONCONFORMING with Rust (Q-T).
# Q-W supersedes only the seven former weight pins listed in C2_SUPERSESSION.json.
_TEMPORAL_UNCHANGED_ROLE_PINS = {'targets/all-date': {'y': {'columns': [{'name': 'm', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 3, 'n_cols': 1, 'data_bytes_hex': '0080de8f711d194300026bb9cf1d19430080de8f711d1943', 'validity': [True, True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['m'], 'y_categorical': {}}, 'targets/date-blank': {'y': {'columns': [{'name': 'm', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 3, 'n_cols': 1, 'data_bytes_hex': '0080de8f711d1943000000000000e0c300026bb9cf1d1943', 'validity': [True, True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['m'], 'y_categorical': {}}, 'targets/all-dur': {'y': {'columns': [{'name': 'm', 'kind': 'numeric', 'block': {'dtype': 'float64', 'n_rows': 3, 'n_cols': 1, 'data_bytes_hex': '000000384d253942000000384d253942000000384d253942', 'validity': [True, True, True], 'precision': {'dtype': 'float64', 'origin': 'default', 'source_dtype': None, 'requested_dtype': None, 'consumer': None}}}]}, 'y_headers': ['m'], 'y_categorical': {}}}


@pytest.mark.parametrize("case", list(_TEMPORAL_UNCHANGED_ROLE_PINS))
def test_temporal_targets_keep_nonconforming_byte_guards(tmp_path: Path, case: str) -> None:
    role, kind = case.split("/")
    cells = _temporal_cells(kind)
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    block = nio.load(_temporal_spec(role), target="assembled", base_dir=tmp_path).to_full_dict()["blocks"]["train"]
    expected = _TEMPORAL_UNCHANGED_ROLE_PINS[case]
    assert {key: block[key] for key in expected} == expected


@pytest.mark.parametrize("kind", ["mix-date", "small-date", "date-first", "all-date", "date-blank", "dur", "all-dur"])
@pytest.mark.parametrize("policy", ["ignore", "remove_sample", "replace", "remove_feature", "abort", "auto"])
def test_qw_temporal_weights_follow_existing_na_stage(tmp_path: Path, kind: str, policy: str) -> None:
    cells = _temporal_cells(kind)
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    if kind == "date-blank" and policy in ("abort", "auto"):
        with pytest.raises(loaders.NAError):
            nio.load(_temporal_spec("weights", policy), target="assembled", base_dir=tmp_path)
        return
    block = nio.load(_temporal_spec("weights", policy), target="assembled", base_dir=tmp_path).blocks["train"]
    rows = [0, 2] if kind == "date-blank" and policy == "remove_sample" else [0, 1, 2]
    assert block.n_samples == len(rows)
    np.testing.assert_array_equal(block.X[0][:, 0], [row + 1.5 for row in rows])
    if kind == "date-blank" and policy == "remove_feature":
        assert block.weights is None
        return
    expected = [0.0 if cells[row] is None and policy == "replace" else float(cells[row]) if isinstance(cells[row], (int, float, bool)) else np.nan for row in rows]
    assert block.weights_header == "m"
    np.testing.assert_array_equal(block.weights, expected)
    np.testing.assert_array_equal(np.isnan(block.weights), np.isnan(expected))


@pytest.mark.parametrize("peer", [None, "abc"])
@pytest.mark.parametrize("storage", [None, "float32", "float64"])
def test_qw_original_big_integer_obeys_the_actual_weight_request(tmp_path: Path, peer: str | None, storage: str | None) -> None:
    (tmp_path / "w.csv").write_text("x,m\n1.5,9007199254740993\n2.5," + (peer or "5") + "\n3.5,7\n")
    spec = _temporal_spec("weights")
    spec["sources"][0]["input"] = "w.csv"
    spec["params"]["delimiter"] = ","
    if storage != "float32":
        with pytest.raises(ValueError, match="float64"):
            nio.to_dataset_package(spec, base_dir=tmp_path, weights=storage)
        return
    block = nio.load(spec, target="assembled", base_dir=tmp_path, weights=storage).blocks["train"]
    # Assembled IR widens selected F32 values exactly; the package owns F32 storage.
    assert block.weights.dtype == np.dtype("float64")
    assert block.weights[0] == 9007199254740992.0
    stored = nio.to_dataset_package(spec, base_dir=tmp_path, weights=storage).to_full_dict()["partitions"]["train"]["payloads"]["train/weights"]["value"]["matrix"]
    assert stored["dtype"] == "float32" and stored["data_bytes_hex"].startswith("0000005a")
    assert np.isnan(block.weights[1]) if peer else block.weights[1] == 5.0


@pytest.mark.parametrize("namespaced", [False, True])
def test_qw_declared_decimal_is_used_on_real_text_weight_cells(tmp_path: Path, namespaced: bool) -> None:
    (tmp_path / "w.csv").write_text("m\n0,25\nabc\n0,5\n")
    spec: dict[str, Any]
    if namespaced:
        (tmp_path / "a.csv").write_text("x,m\n1.5,2.5\n3.5,4.5\n5.5,6.5\n")
        spec = {"sources": [{"id": "a", "input": "a.csv", "columns": [{"role": "features", "select": ["x", "m"]}]}, {"id": "w", "role": "weights", "input": "w.csv", "params": {"delimiter": ";", "decimal": ","}}], "params": {"delimiter": ",", "na": {"policy": "ignore"}}}
    else:
        (tmp_path / "w.csv").write_text("x;m\n1,5;0,25\n2,5;abc\n3,5;0,5\n")
        spec = _temporal_spec("weights")
        spec["sources"][0]["input"] = "w.csv"
        spec["params"].update(delimiter=";", decimal=",")
    block = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    np.testing.assert_array_equal(block.weights, [.25, np.nan, .5])
    assert block.weights_header == ("w__m" if namespaced else "m")


@pytest.mark.parametrize("reverse", [False, True])
def test_qw_each_input_decimal_scope_survives_a_text_peer(tmp_path: Path, reverse: bool) -> None:
    _sheet(tmp_path / "a.xlsx", [["x", "m"], [1, "0.75"], [2, datetime.datetime(2026, 1, 2)], [3, "abc"]])
    (tmp_path / "b.csv").write_text("x;m\n4,5;0,25\n5,5;abc\n")
    spec = _temporal_spec("weights")
    spec["sources"][0].update(input=["b.csv", "a.xlsx"] if reverse else ["a.xlsx", "b.csv"], merge="concat_samples")
    spec["params"].update(delimiter=";", decimal=",")
    block = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    expected = [.25, np.nan, .75, np.nan, np.nan] if reverse else [.75, np.nan, np.nan, .25, np.nan]
    np.testing.assert_array_equal(block.weights, expected)


def test_qw_csv_negative_zero_has_real_signed_bits(tmp_path: Path) -> None:
    (tmp_path / "w.csv").write_text("x,m\n1.5,-0\n2.5,0.0\n3.5,-0.0\n")
    spec = _temporal_spec("weights")
    spec["sources"][0]["input"] = "w.csv"
    spec["params"]["delimiter"] = ","
    block = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    assert block.weights.view(np.uint64).tolist() == [1 << 63, 0, 1 << 63]


@pytest.mark.parametrize("kind", ["time", "text-ctl", "bool-ctl"])
def test_qw_nontemporal_controls_keep_original_weight_meaning(tmp_path: Path, kind: str) -> None:
    cells = _temporal_cells(kind)
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    block = nio.load(_temporal_spec("weights"), target="assembled", base_dir=tmp_path).blocks["train"]
    np.testing.assert_array_equal(block.weights, [5.0, 7.0, 1.0 if kind == "bool-ctl" else np.nan])


@pytest.mark.parametrize("dtype", [object, "datetime64[ns]"])
@pytest.mark.parametrize("storage", [None, "float32", "float64"])
def test_qw_empty_numeric_column_slice_keeps_vector_shape(dtype: Any, storage: str | None) -> None:
    result = loaders.coerce_numeric(pd.DataFrame({"w": pd.Series([], dtype=dtype)}), storage)[:, 0]
    assert result.shape == (0,) and result.dtype == np.dtype("float64")


def test_qw_parquet_weight_refusal_is_route_specific_not_universal(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"x": [1.5, 2.5, 3.5], "m": pa.array([datetime.datetime(2026, 1, 2), datetime.datetime(2026, 1, 3), None], type=pa.timestamp("us"))}), tmp_path / "d.parquet")
    spec = _temporal_spec("weights")
    spec["sources"][0]["input"] = "d.parquet"
    # Q-P: these raw routes remain admitted while Rust refuses the decoder input.
    raw = nio.load(spec, target="assembled", base_dir=tmp_path).blocks["train"]
    assert raw.n_samples == 3 and np.isnan(raw.weights).all()
    with pytest.raises(ValueError, match="object"):
        nio.to_dataset_package(spec, base_dir=tmp_path)
    with pytest.raises(ValueError, match="object"):
        nio.load(spec, target="spectrodataset", base_dir=tmp_path, weights="float32")


@pytest.mark.parametrize("cells,expected", [([1e30, True, datetime.datetime(2026, 1, 2)], [1e30, 1.0, np.nan]), ([5, True, datetime.datetime(2026, 1, 2)], [5.0, 1.0, np.nan]), ([5, "abc", datetime.datetime(2026, 1, 2)], [5.0, np.nan, np.nan]), ([datetime.datetime(2026, 1, 2), True, "abc"], [np.nan, 1.0, np.nan]), ([True, False, datetime.datetime(2026, 1, 2)], [1.0, 0.0, np.nan])])
def test_qw_feature_bool_date_peers_remain_feature_authority(tmp_path: Path, cells: list[Any], expected: list[float]) -> None:
    _sheet(tmp_path / "d.xlsx", [["x", "m"], *[[i + 1.5, cell] for i, cell in enumerate(cells)]])
    block = nio.load(_temporal_spec(), target="assembled", base_dir=tmp_path).blocks["train"]
    np.testing.assert_array_equal(block.X[0][:, 1], expected)
