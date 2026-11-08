# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Smoke tests for the pyo3 binding's JSON surface against the contract corpus."""
from pathlib import Path

import pytest
from nirs4all_io._adapter import to_spectrodataset
from nirs4all_io._package import DatasetPackage

import nirs4all_io as nio

CORPUS = Path(__file__).resolve().parents[3] / "tests/goldens/contract/corpus"


def test_infer_returns_plan_dict():
    plan = nio.infer(str(CORPUS / "single_combined"))
    assert isinstance(plan, dict)
    assert "resolved_spec" in plan


def test_to_spec_returns_validatable_spec():
    spec = nio.to_spec(str(CORPUS / "train_test"))
    assert spec["schema_version"] == 1
    # the produced spec round-trips through validate()
    nio.validate(spec)


def test_load_assembled_summary():
    assembled = nio.load(str(CORPUS / "x_y_separate"), target="assembled")
    assert "blocks" in assembled
    assert type(assembled["assembled_schema_version"]) is int
    assert assembled["assembled_schema_version"] == 3


@pytest.mark.parametrize("wire_version", [None, 2, 3.0, True, "3"])
def test_public_assembled_paths_reject_non_integer_or_retired_wire(monkeypatch, wire_version):
    """Fresh-wheel public readers accept only an exact integer v3 marker."""
    full = {"assembled_schema_version": wire_version, "blocks": {}}
    monkeypatch.setattr(nio, "load_summary", lambda *_args: full)
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        nio.load("legacy.csv", target="assembled")
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        DatasetPackage(full)
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        nio.to_spectrodataset(full, spectro_dataset_cls=_RecordingDataset)


def test_validate_rejects_invalid_spec():
    with pytest.raises(ValueError):
        nio.validate({"partitions": {"by": "random"}})


class _RecordingDataset:
    """Recording double so the SpectroDataset adapter is testable without nirs4all."""

    def __init__(self, name):
        self.name = name
        self.add_samples_calls = []
        self.targets = 0

    def add_samples(self, x, meta, headers=None, header_unit=None):
        self.add_samples_calls.append(meta)

    def add_targets(self, y):
        self.targets += 1

    def set_signal_type(self, *a, **k):
        pass

    def add_metadata(self, df):
        pass

    def add_features(self, *a, **k):
        pass

    def set_task_type(self, *a, **k):
        pass

    def set_folds(self, f):
        pass

    def set_repetition(self, r):
        pass

    def set_aggregate(self, *a, **k):
        pass

    def set_aggregate_method(self, *a):
        pass

    def set_aggregate_exclude_outliers(self, *a):
        pass


def test_spectrodataset_adapter_drives_the_builder():
    # nirs4all is absent in this clean venv; inject a recording double to exercise
    # the lazy SpectroDataset adapter end-to-end (full-array reconstruction).
    ds = nio.load(str(CORPUS / "train_test"), target="spectrodataset", spectro_dataset_cls=_RecordingDataset)
    assert ds.name == "train_test"
    assert {m["partition"] for m in ds.add_samples_calls} == {"train", "test"}
    assert ds.targets >= 1


def test_spectrodataset_adapter_rejects_retired_unversioned_wire():
    """Wheel-level guard: direct adapter calls cannot revive v1 fallback."""
    with pytest.raises(ValueError, match="assembled_schema_version=3"):
        to_spectrodataset({}, spectro_dataset_cls=_RecordingDataset)

# These cases require the rebuilt native owner; never run against an old wheel.
def _partition_label_spec(path, **partitions):
    return {"task_type": "regression", "sources": [{"id": "d", "input": str(path), "role": "mixed", "params": {"delimiter": ";", "has_header": True}, "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["set", "row"]}]}], "partitions": {"by": "column", "column": "set", **partitions}}


@pytest.mark.parametrize("policy", [None, "train", "test", "drop", "error"])
def test_partition_labels_csv_literal_nan_is_separate_from_missing(tmp_path, policy):
    path = tmp_path / "missing-labels.csv"
    path.write_text('x;y;set;row\n1;101;cal;r0\n2;102;NAN;r1\n3;103;;r2\n4;104;NA;r3\n5;105;none;r4\n6;106;val;r5\n7;107;"nan";r6\n', encoding="utf-8")
    options = {"train_values": ["cal"], "test_values": ["nan"]}
    if policy is not None:
        options["unknown_policy"] = policy
    spec = _partition_label_spec(path, **options)
    if policy == "error":
        with pytest.raises(ValueError, match="set"):
            nio.load(spec, target="dataset_package")
        return
    expected = {"train": ["r0"], "test": ["r1"]}
    if policy is None or policy == "train":
        expected["train"] += ["r2", "r3", "r4", "r5", "r6"]
    elif policy == "test":
        expected["test"] += ["r2", "r3", "r4", "r5", "r6"]
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == expected
    for block in assembled.blocks.values():
        assert block.y[:, 0].tolist() == [101 + int(row[1:]) for row in block.metadata["row"]]


@pytest.mark.parametrize("part", ["train", "test", "val", "predict"])
@pytest.mark.parametrize("bad", [None, [], {}])
def test_partition_labels_invalid_declaration_refuses_with_list_name(tmp_path, part, bad):
    path = tmp_path / "labels.csv"
    path.write_text("x;y;set;row\n1;101;cal;r0\n")
    with pytest.raises(ValueError, match=part + "_values"):
        nio.load(_partition_label_spec(path, **{part + "_values": [bad]}), target="assembled")


@pytest.mark.parametrize("values,rows", [(["nan"], ["r1", "r5"]), (["none", ""], ["r3", "r4"])])
def test_partition_labels_parquet_literals_and_null_keep_exact_rows(tmp_path, values, rows):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "literals.parquet"
    pd.DataFrame({"x": [float(i) for i in range(9)], "y": [float(101+i) for i in range(9)], "set": ["cal", "nan", None, "None", "", "NaN", "NA", "val", None], "row": [f"r{i}" for i in range(9)]}).to_parquet(path, index=False)
    spec = _partition_label_spec(path, train_values=["cal"], test_values=values, unknown_policy="drop")
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"train": ["r0"], "test": rows}


@pytest.mark.parametrize("observed,declared,matches", [("ΣΑΣ", "σας", True), ("İ", "i\u0307", True), ("ß", "SS", False)])
def test_partition_labels_unicode_test_side_under_default_train(tmp_path, observed, declared, matches):
    path = tmp_path / "unicode.csv"
    path.write_text(f"x;y;set;row\n1;101;{observed};r0\n2;102;unassigned;r1\n", encoding="utf-8")
    assembled = nio.load(_partition_label_spec(path, test_values=[declared]), target="dataset_package").to_assembled()
    actual = {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()}
    assert actual == ({"train": ["r1"], "test": ["r0"]} if matches else {"train": ["r0", "r1"]})


class _PartitionRecorder(_RecordingDataset):
    def add_targets(self, y):
        self.target_values = y.reshape(-1).tolist()

    def add_metadata(self, df):
        self.metadata = df.copy()


def test_partition_labels_adapter_retains_exact_selected_rows(tmp_path):
    path = tmp_path / "adapter.csv"
    path.write_text("x;y;set;row\n1;101;cal;r0\n2;102;NAN;r1\n3;103;;r2\n")
    spec = _partition_label_spec(path, train_values=["cal"], test_values=["nan"], unknown_policy="drop")
    result = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_PartitionRecorder)
    assert [meta["partition"] for meta in result.add_samples_calls] == ["train", "test"]
    assert result.target_values == [101, 102]
    assert result.metadata["row"].tolist() == ["r0", "r1"]


@pytest.mark.parametrize("value", [1,"1",1.0,"1.0","1e0"])
def test_c8_native_csv_integer_missing_numeric_aliases(tmp_path,value):
    path=tmp_path/"numeric.csv"
    path.write_text("x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;;r2\n4;104;1;r3\n5;105;2;r4\n")
    spec=_partition_label_spec(path,test_values=[value],unknown_policy="drop")
    assembled=nio.load(spec,target="dataset_package").to_assembled()
    assert {part:block.metadata["row"].tolist() for part,block in assembled.blocks.items()} == {"test":["r0","r3"]}
    default=_partition_label_spec(path,test_values=[2])
    result=nio.load(default,target="dataset_package").to_assembled()
    assert {part:block.metadata["row"].tolist() for part,block in result.blocks.items()} == {"train":["r0","r2","r3"],"test":["r1","r4"]}


@pytest.mark.parametrize("value,rows", [(0,["r0","r2"]),(0.0,["r0","r2"]),("0",["r0","r2"]),(-0.0,["r1","r3"]),("-0",["r1","r3"]),("-0.0",["r1","r3"])])
def test_c8_native_signed_zero_partition_identity(tmp_path,value,rows):
    path=tmp_path/"zeros.csv"
    path.write_text("x;y;set;row\n1;101;0;r0\n2;102;-0;r1\n3;103;0.0;r2\n4;104;-0.0;r3\n5;105;1.5;r4\n")
    assembled=nio.load(_partition_label_spec(path,test_values=[value],unknown_policy="drop"),target="dataset_package").to_assembled()
    assert assembled.blocks["test"].metadata["row"].tolist() == rows


@pytest.mark.parametrize("value", [0.1,"0.1",0.10000000149011612,0.10000000000001])
@pytest.mark.parametrize("keyed", [False,True])
@pytest.mark.parametrize("role", ["ignore","metadata"])
def test_c8_native_float32_secondary_join_retains_dtype(tmp_path,value,keyed,role):
    pd=pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    left=tmp_path/"left.csv"
    right=tmp_path/"right.parquet"
    left.write_text("x;y;sid;row\n1;101;s0;r0\n2;102;s1;r1\n3;103;s2;r2\n4;104;s3;r3\n")
    frame=pd.DataFrame({"set":pd.Series([0.1,0.2,None,0.1],dtype="float32")})
    if keyed:
        frame["sid"]=["s0","s1","s2","s3"]
        frame=frame.iloc[::-1]
    frame.to_parquet(right,index=False)
    spec={"task_type":"regression","sources":[{"id":"left","input":str(left),"role":"mixed","params":{"delimiter":";","has_header":True},"columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["sid","row"]}]},{"id":"labels","input":str(right),"role":"mixed","params":{"na":{"policy":"ignore"}},"columns":[{"role":role,"select":["set"]}]}],"partitions":{"by":"column","column":"set","test_values":[value],"unknown_policy":"drop"}}
    if keyed:
        spec["sources"][0]["key"]="sid"
        spec["sources"][1]["key"]="sid"
        spec["sources"][1]["join"]={"to":"left","on":"sid","how":"1:1"}
    block=nio.load(spec,target="dataset_package").to_assembled().blocks["test"]
    assert block.metadata["row"].tolist()==["r0","r3"]
    assert block.y[:,0].tolist()==[101.,104.]


@pytest.mark.parametrize("values,rows", [(["none"],[]),(["inf"],["r1"]),(["-inf"],["r2"])])
def test_c8_native_parquet_ignored_infinity_is_local_lexical(tmp_path,values,rows):
    pd=pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path=tmp_path/"infinity.parquet"
    pd.DataFrame({"x":[1.,2.,3.,4.],"y":[101.,102.,103.,104.],"set":[1.5,float("inf"),-float("inf"),None],"row":["r0","r1","r2","r3"]}).to_parquet(path,index=False)
    spec=_partition_label_spec(path,test_values=values,unknown_policy="drop")
    spec["sources"][0]["columns"]=[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["row"]},{"role":"ignore","select":["set"]}]
    spec["sources"][0]["params"]["na"]={"policy":"ignore"}
    assembled=nio.load(spec,target="dataset_package").to_assembled()
    assert {part:block.metadata["row"].tolist() for part,block in assembled.blocks.items()} == ({"test":rows} if rows else {})


def test_c8_native_numeric_runtime_overlap_text_positive_and_absent_alias(tmp_path):
    for name,values,refuses,expected in [("numeric",["1","2",""],True,None),("text",["1","1.0","a"],False,{"train":["r0"],"test":["r1"]}),("absent",["2","3",""],False,{})]:
        path=tmp_path/(name+".csv")
        path.write_text("x;y;set;row\n"+"".join(f"{i+1};{i+101};{value};r{i}\n" for i,value in enumerate(values)))
        spec=_partition_label_spec(path,train_values=[1],test_values=["1.0"],unknown_policy="drop")
        if refuses:
            with pytest.raises(ValueError,match="set.*train_values.*test_values"):
                nio.load(spec,target="dataset_package")
        else:
            assembled=nio.load(spec,target="dataset_package").to_assembled()
            assert {part:block.metadata["row"].tolist() for part,block in assembled.blocks.items()} == expected


def _write_scientific_xlsx(path, headers, rows, *, numeric_columns=(), date_columns=(), epoch_1904=False):
    """Write physical XML tokens directly, without converting them through floats."""
    from xml.sax.saxutils import escape
    from zipfile import ZipFile

    def cell(column, value, header=False):
        if not header and column in numeric_columns:
            style = ' s="1"' if column in date_columns else ""
            return f"<c{style}><v>{escape(str(value))}</v></c>"
        return f'<c t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    sheet_rows = ["<row r=\"1\">" + "".join(cell(i, value, True) for i, value in enumerate(headers)) + "</row>"]
    sheet_rows += [f'<row r="{r}">' + "".join(cell(i, value) for i, value in enumerate(row)) + "</row>" for r, row in enumerate(rows, 2)]
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        "_rels/.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr date1904="{int(epoch_1904)}"/><sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "xl/styles.xml": '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14" applyNumberFormat="1"/></cellXfs></styleSheet>',
        "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>",
    }
    with ZipFile(path, "w") as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)


def _scientific_ignored_partition_spec(path, declared, policy=None):
    spec = _partition_label_spec(path, train_values=[1, 2], test_values=[declared])
    spec["sample_index"] = {"by": "id", "key": "row"}
    spec["sources"][0]["columns"] = [
        {"role": "features", "select": ["x"]},
        {"role": "targets", "select": ["y"]},
        {"role": "metadata", "select": ["row"]},
        {"role": "ignore", "select": ["set"]},
    ]
    spec["sources"][0]["params"]["na"] = {"policy": "ignore"}
    if policy is not None:
        spec["partitions"]["unknown_policy"] = policy
    return spec


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
@pytest.mark.parametrize("token", ["1000000000000000000000000000000", "-9999999999999999999999999999999", "18446744073709551616"])
@pytest.mark.parametrize("policy", [None, "drop", "error"])
def test_scientific_ignored_outside_domain_partition_selects_declared_test(tmp_path, extension, token, policy):
    path = tmp_path / ("labels." + extension)
    if extension == "csv":
        path.write_text(f"x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;{token};r2\n", encoding="utf-8")
    else:
        _write_scientific_xlsx(path, ["x", "y", "set", "row"], [(1, 101, "1", "r0"), (2, 102, "2", "r1"), (3, 103, token, "r2")], numeric_columns=(0, 1, 2))
    spec = _scientific_ignored_partition_spec(path, token, policy)
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert set(assembled.blocks) == {"train", "test"}
    for part, x, y, ids in [("train", [1, 2], [101, 102], ["r0", "r1"]), ("test", [3], [103], ["r2"])]:
        block = assembled.blocks[part]
        assert block.n_samples == len(ids)
        assert block.X[0][:, 0].tolist() == x
        assert block.y[:, 0].tolist() == y
        assert block.metadata["row"].tolist() == ids
    # Selecting the same numeric token as metadata still refuses its domain.
    spec["sources"][0]["columns"][-1]["role"] = "metadata"
    with pytest.raises(ValueError, match="outside the supported i64/u64 domain"):
        nio.load(spec, target="dataset_package")


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_scientific_outside_domain_partition_has_no_float_or_spelling_alias(tmp_path, extension):
    token = "1000000000000000000000000000000"
    path = tmp_path / ("lexical." + extension)
    if extension == "csv":
        path.write_text(f"x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;{token};r2\n", encoding="utf-8")
    else:
        _write_scientific_xlsx(path, ["x", "y", "set", "row"], [(1, 101, "1", "r0"), (2, 102, "2", "r1"), (3, 103, token, "r2")], numeric_columns=(0, 1, 2))
    for alias in ["1e30", "+" + token, "0" + token, "1000000000000000019884624838656"]:
        assembled = nio.load(_scientific_ignored_partition_spec(path, alias, "drop"), target="dataset_package").to_assembled()
        assert set(assembled.blocks) == {"train"}
        block = assembled.blocks["train"]
        assert block.metadata["row"].tolist() == ["r0", "r1"]
        assert block.X[0][:, 0].tolist() == [1, 2]
        assert block.y[:, 0].tolist() == [101, 102]


def test_scientific_ignored_csv_1e999_keeps_local_lexical_match_and_selected_refusal(tmp_path):
    path = tmp_path / "overflow.csv"
    path.write_text("x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;1e999;r2\n", encoding="utf-8")
    for declared, matches in [("1e999", True), ("inf", False), ("1e998", False)]:
        spec = _scientific_ignored_partition_spec(path, declared, "drop")
        assembled = nio.load(spec, target="dataset_package").to_assembled()
        assert set(assembled.blocks) == ({"train", "test"} if matches else {"train"})
        assert assembled.blocks["train"].metadata["row"].tolist() == ["r0", "r1"]
        if matches:
            block = assembled.blocks["test"]
            assert block.metadata["row"].tolist() == ["r2"]
            assert block.X[0][:, 0].tolist() == [3]
            assert block.y[:, 0].tolist() == [103]
    spec = _scientific_ignored_partition_spec(path, "1e999")
    spec["sources"][0]["columns"][-1]["role"] = "metadata"
    with pytest.raises(ValueError, match="finite"):
        nio.load(spec, target="dataset_package")


@pytest.mark.parametrize("date_style", [False, True])
def test_scientific_xlsx_integer_identity_joins_exactly_and_refuses_rounded_neighbor(tmp_path, date_style):
    path, targets = tmp_path / "x.xlsx", tmp_path / "y.csv"
    dates = (0,) if date_style else ()
    _write_scientific_xlsx(path, ["k", "x"], [("9007199254740993", 1)], numeric_columns=(0, 1), date_columns=dates)
    spec = {"task_type": "regression", "sample_index": {"by": "id", "key": "k"}, "sources": [
        {"id": "x", "input": str(path), "key": "k", "columns": {"features": "x", "metadata": "k"}},
        {"id": "y", "input": str(targets), "key": "k", "columns": {"targets": "y"}, "join": {"left": "x", "right": "y", "left_on": "k", "right_on": "k", "how": "1:1", "coverage": "complete"}},
    ], "params": {"delimiter": ";", "na": {"policy": "ignore"}}}
    # This single-row wrong-target case silently succeeded before the date fix.
    targets.write_text("k;y\n9007199254740992;20\n", encoding="utf-8")
    with pytest.raises(ValueError):
        nio.load(spec, target="dataset_package")
    targets.write_text("k;y\n9007199254740993;10\n", encoding="utf-8")
    block = nio.load(spec, target="dataset_package").to_assembled().blocks["train"]
    assert block.n_samples == 1
    assert block.metadata["k"].tolist() == [9007199254740993]
    assert block.X[0][:, 0].tolist() == [1]
    assert block.y[:, 0].tolist() == [10]
    _write_scientific_xlsx(path, ["k", "x"], [("9007199254740993", 1), ("9007199254740992", 2)], numeric_columns=(0, 1), date_columns=dates)
    targets.write_text("k;y\n9007199254740992;20\n9007199254740993;10\n", encoding="utf-8")
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert set(assembled.blocks) == {"train"}
    block = assembled.blocks["train"]
    assert block.n_samples == 2
    assert block.metadata["k"].tolist() == [9007199254740993, 9007199254740992]
    assert block.X[0][:, 0].tolist() == [1, 2]
    assert block.y[:, 0].tolist() == [10, 20]


@pytest.mark.parametrize("token,epoch,expected", [
    ("45292", False, "2024-01-01 00:00:00"),
    ("0", True, "1904-01-01 00:00:00"),
    ("9007199254740993.0", False, 9007199254740992.0),
    ("9.007199254740993E15", False, 9007199254740992.0),
    ("1000000000.25", False, 1000000000.25),
    ("1E+30", False, 1e30),
    ("-9007199254740993", False, -9007199254740993),
])
def test_scientific_date_rendering_and_noninteger_serial_semantics(tmp_path, token, epoch, expected):
    path = tmp_path / "dates.xlsx"
    _write_scientific_xlsx(path, ["k", "x"], [(token, 7)], numeric_columns=(0, 1), date_columns=(0,), epoch_1904=epoch)
    spec = {"sources": [{"id": "d", "input": str(path), "columns": {"features": "x", "metadata": "k"}}]}
    block = nio.load(spec, target="dataset_package").to_assembled().blocks["train"]
    assert block.n_samples == 1
    assert block.metadata["k"].tolist() == [expected]
    assert block.X[0][:, 0].tolist() == [7]


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
def test_scientific_u64_spelling_preserves_canonical_identity_and_same_fold(tmp_path, extension):
    path = tmp_path / ("u64." + extension)
    ids = ["+18446744073709551615", "0018446744073709551614", "18446744073709551613", "18446744073709551612"]
    labels = ["18446744073709551615", "+18446744073709551615", "0018446744073709551615", "18446744073709551614"]
    rows = [(ids[i], i + 1, i + 101, labels[i]) for i in range(4)]
    if extension == "csv":
        path.write_text("id;x;y;cv\n" + "".join(";".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")
    else:
        _write_scientific_xlsx(path, ["id", "x", "y", "cv"], rows, numeric_columns=(0, 1, 2, 3))
    spec = {"task_type": "regression", "sample_index": {"by": "id", "key": "id"}, "sources": [{"id": "d", "input": str(path), "columns": {"features": "x", "targets": "y", "metadata": ["id", "cv"]}}], "params": {"delimiter": ";", "na": {"policy": "ignore"}}, "folds": {"column": "cv"}}
    full = nio.assembled_full(spec)
    metadata = {column["name"]: column["values"] for column in full["blocks"]["train"]["metadata"]["columns"]}
    expected = [18446744073709551615 - i for i in range(4)]
    assert metadata["id"] == expected
    assert all(type(value) is int for value in metadata["id"])
    assert metadata["cv"] == [18446744073709551615] * 3 + [18446744073709551614]
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert set(assembled.blocks) == {"train"}
    assert assembled.folds == [([0, 1, 2], [3]), ([3], [0, 1, 2])]
    block = assembled.blocks["train"]
    assert block.metadata["id"].tolist() == expected
    assert block.X[0][:, 0].tolist() == [1, 2, 3, 4]
    assert block.y[:, 0].tolist() == [101, 102, 103, 104]


@pytest.mark.parametrize("token,error", [
    ("9007199254740993", "exactly representable"),
    ("18446744073709551616", "outside the supported i64/u64 domain"),
    ("1000000000000000000000000000000", "outside the supported i64/u64 domain"),
])
def test_scientific_unrenderable_date_integer_keeps_numeric_role_refusal(tmp_path, token, error):
    path = tmp_path / "numeric-date.xlsx"
    _write_scientific_xlsx(path, ["k", "x"], [(token, 7)], numeric_columns=(0, 1), date_columns=(0,))
    spec = {"sources": [{"id": "d", "input": str(path), "columns": {"features": "k", "ignore": "x"}}]}
    with pytest.raises(ValueError, match=error):
        nio.load(spec, target="dataset_package")
    if token != "9007199254740993":
        spec["sources"][0]["columns"] = {"features": "x", "metadata": "k"}
        with pytest.raises(ValueError, match="outside the supported i64/u64 domain"):
            nio.load(spec, target="dataset_package")


@pytest.mark.parametrize("date_style", [False, True])
def test_scientific_ignored_xlsx_1e999_follows_existing_infinity_label(tmp_path, date_style):
    path = tmp_path / "overflow.xlsx"
    _write_scientific_xlsx(path, ["x", "y", "set", "row"], [(1, 101, "1", "r0"), (2, 102, "2", "r1"), (3, 103, "1e999", "r2")], numeric_columns=(0, 1, 2), date_columns=(2,) if date_style else ())
    for declared, matches in [("1e999", False), ("inf", True)]:
        spec = _scientific_ignored_partition_spec(path, declared, "drop")
        if date_style:
            spec["partitions"]["train_values"] = ["1900-01-01 00:00:00", "1900-01-02 00:00:00"]
        assembled = nio.load(spec, target="dataset_package").to_assembled()
        assert set(assembled.blocks) == ({"train", "test"} if matches else {"train"})
        assert assembled.blocks["train"].metadata["row"].tolist() == ["r0", "r1"]
        if matches:
            block = assembled.blocks["test"]
            assert block.metadata["row"].tolist() == ["r2"]
            assert block.X[0][:, 0].tolist() == [3]
            assert block.y[:, 0].tolist() == [103]
