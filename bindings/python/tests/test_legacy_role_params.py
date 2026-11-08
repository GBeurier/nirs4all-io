"""Installed wheel regression: no dropped first target/metadata row."""

import numpy as np
import pytest

import nirs4all_io as nio


def test_headerless_role_params_keep_first_targets_and_metadata(tmp_path):
    pytest.importorskip("nirs4all.data", reason="optional downstream SpectroDataset integration")
    config = {"task_type": "regression", "global_params": {"delimiter": ";", "has_header": True}}
    for partition in ("train", "test"):
        for role, content in {"x": "1;2\n3;4\n5;6\n", "y": "101\n102\n103\n", "group": "first\nsecond\nthird\n"}.items():
            path = tmp_path / f"{partition}_{role}.csv"
            path.write_text(content)
            config[f"{partition}_{role}"] = str(path)
            config[f"{partition}_{role}_params"] = {"has_header": False}
    dataset = nio.load(config, target="spectrodataset")
    for partition in ("train", "test"):
        np.testing.assert_array_equal(dataset.x({"partition": partition}), [[1, 2], [3, 4], [5, 6]])
        # This boundary preserves loaded values. Numeric target encoding is a
        # separate library-owned processing, not IO's file-row contract.
        np.testing.assert_array_equal(np.asarray(dataset.y({"partition": partition, "y": "raw"})).reshape(-1), [101, 102, 103])
        # Headerless metadata receives source-qualified column identities.
        assert dataset.metadata_columns == ["train_m__0", "test_m__0"]
        column = dataset.metadata_columns.index(f"{partition}_m__0")
        assert dataset.metadata({"partition": partition})[0, column] == "first"



class _NativeTargetRecorder:
    def __init__(self, name="dataset"):
        self.name = name
        self.samples = []
        self.targets = None
        self.metadata = None

    def add_samples(self, data, indexes, headers=None, header_unit=None):
        self.samples.append((indexes["partition"], np.asarray(data).copy(), headers))

    def add_targets(self, y):
        self.targets = np.asarray(y).copy()

    def add_metadata(self, data, headers=None):
        self.metadata = data.copy()

    def set_signal_type(self, value, src=0, forced=True):
        pass

    def set_task_type(self, value, forced=True):
        self.task_type = value


def _native_target_source(tmp_path, source_id, text, role, partition=None, *, header=True, **kwargs):
    path = tmp_path / (source_id + ".csv")
    path.write_text(text, encoding="utf-8")
    result = {"id": source_id, "input": str(path), "role": role, "params": {"has_header": header, "delimiter": ";"}, **kwargs}
    if partition is not None:
        result["partition"] = partition
    return result


def _native_target_pair(tmp_path, x_header, train_y, test_y, *, y_header=True, task="regression"):
    sources = []
    for partition, x, y in [("train", "1;2\n3;4\n5;6\n", train_y), ("test", "7;8\n9;10\n11;12\n", test_y)]:
        sources.append(_native_target_source(tmp_path, partition + "_x", (x_header or "") + x, "features", partition, header=x_header is not None))
        sources.append(_native_target_source(tmp_path, partition + "_y", y, "targets", partition, header=y_header))
    return {"name": "target-label-pair", "task_type": task, "sources": sources}


def _native_duplicate_targets(tmp_path, case):
    x = "x0;x1\n1;2\n3;4\n5;6\n"
    y1, y2 = "y\n1\n2\n3\n", "y\n4\n5\n6\n"
    header, task = True, "regression"
    if case == "L":
        x, y1, y2, header = "1;2\n3;4\n5;6\n", "1\n2\n3\n", "4\n5\n6\n", False
    if case == "M":
        y1, y2, task = "y\ncat\ndog\ncat\n", "y\nred\nblue\nred\n", "multiclass"
    if case == "N":
        first = _native_target_source(tmp_path, "d", "x0;y\n1;7\n3;8\n5;9\n", "mixed", columns=[{"role":"features","select":["x0"]},{"role":"targets","select":["y"]}])
        sources = [first, _native_target_source(tmp_path, "y2", y2, "targets")]
    else:
        sources = [_native_target_source(tmp_path, "x", x, "features", header=header), _native_target_source(tmp_path, "y1", y1, "targets", header=header), _native_target_source(tmp_path, "y2", y2, "targets", header=header)]
    return {"name": "duplicate-target-" + case, "task_type": task, "sources": sources}


@pytest.mark.parametrize("x_header,train_y,test_y,y_header,headers,train_values,test_values", [
    (None, "101\n102\n103\n", "201\n202\n203\n", False, ["0"], [[101],[102],[103]], [[201],[202],[203]]),
    ("x0;x1\n", "101\n102\n103\n", "201\n202\n203\n", False, ["0"], [[101],[102],[103]], [[201],[202],[203]]),
    (None, "1;9\n2;8\n3;7\n", "4;6\n5;5\n6;4\n", False, ["0","1"], [[1,9],[2,8],[3,7]], [[4,6],[5,5],[6,4]]),
    ("a;b\n", "b;a\n1;9\n2;8\n3;7\n", "a;b\n6;4\n5;5\n4;6\n", True, ["b","a"], [[1,9],[2,8],[3,7]], [[4,6],[5,5],[6,4]]),
    ("x0;x1\n", "0__train_y\n101\n102\n103\n", "0__train_y\n201\n202\n203\n", True, ["0__train_y"], [[101],[102],[103]], [[201],[202],[203]]),
])
def test_native_target_labels_and_recorder_keep_values(tmp_path, x_header, train_y, test_y, y_header, headers, train_values, test_values):
    spec = _native_target_pair(tmp_path, x_header, train_y, test_y, y_header=y_header)
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers == headers
    expected_test_headers = ["a","b"] if headers == ["b","a"] else headers
    assert assembled.blocks["test"].y_headers == expected_test_headers
    np.testing.assert_array_equal(assembled.blocks["train"].y, train_values)
    raw_test = [[6,4],[5,5],[4,6]] if headers == ["b","a"] else test_values
    np.testing.assert_array_equal(assembled.blocks["test"].y, raw_test)
    record = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
    np.testing.assert_array_equal(record.targets, train_values + test_values)
    assert [part for part,_,_ in record.samples] == ["train","test"]
    np.testing.assert_array_equal(record.samples[0][1], [[1,2],[3,4],[5,6]])
    np.testing.assert_array_equal(record.samples[1][1], [[7,8],[9,10],[11,12]])


@pytest.mark.parametrize("x_header,y_header", [(None,False),("x0;x1\n",False),("x0;x1\n",True)])
def test_native_target_categorical_keys_decode_reversed_partition_categories(tmp_path, x_header, y_header):
    header = "label\n" if y_header else ""
    spec = _native_target_pair(tmp_path, x_header, header+"cat\ndog\ncat\n", header+"dog\ncat\ncat\n", y_header=y_header, task="multiclass")
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    label = "label" if y_header else "0"
    for block in assembled.blocks.values():
        assert block.y_headers == [label]
        assert list(block.y_categorical) == [label]
    record = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
    assert record.targets.reshape(-1).tolist() == ["cat","dog","cat","dog","cat","cat"]


@pytest.mark.parametrize("header,expected", [(False,"0"),(True,"a")])
def test_native_target_single_partition_uses_actual_native_name(tmp_path, header, expected):
    x = "a;b\n1;2\n3;4\n5;6\n" if header else "1;2\n3;4\n5;6\n"
    y = "a\n101\n102\n103\n" if header else "101\n102\n103\n"
    spec = {"task_type":"regression","sources":[_native_target_source(tmp_path,"x",x,"features",header=header),_native_target_source(tmp_path,"y",y,"targets",header=header)]}
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers == [expected]
    np.testing.assert_array_equal(assembled.blocks["train"].y[:,0], [101,102,103])


@pytest.mark.parametrize("case,headers", [("K",["y","y2__y"]),("L",["y1__0","y2__0"]),("M",["y","y2__y"]),("N",["y","y2__y"])])
def test_native_target_duplicate_native_names_keep_combined_labels(tmp_path, case, headers):
    spec = _native_duplicate_targets(tmp_path, case)
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    block = assembled.blocks["train"]
    assert block.y_headers == headers
    if case == "M":
        assert list(block.y_categorical) == headers
        record = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
        assert record.targets.tolist() == [["cat","red"],["dog","blue"],["cat","red"]]
    else:
        expected = [[7,4],[8,5],[9,6]] if case == "N" else [[1,4],[2,5],[3,6]]
        np.testing.assert_array_equal(block.y, expected)


@pytest.mark.parametrize("train_header,test_header", [("a","b"),("train_y__0","test_y__0")])
def test_native_target_different_literal_labels_still_refuse_adapter(tmp_path, train_header, test_header):
    spec = _native_target_pair(tmp_path,"x0;x1\n",train_header+"\n101\n102\n103\n",test_header+"\n201\n202\n203\n")
    assembled = nio.load(spec,target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers == [train_header]
    assert assembled.blocks["test"].y_headers == [test_header]
    np.testing.assert_array_equal(assembled.blocks["test"].y[:,0],[201,202,203])
    with pytest.raises(ValueError,match="target.*layout"):
        nio.load(spec,target="spectrodataset",spectro_dataset_cls=_NativeTargetRecorder)


def test_native_target_final_export_collision_refuses_before_category_overwrite(tmp_path):
    spec = _native_duplicate_targets(tmp_path,"M")
    spec["sources"].append(_native_target_source(tmp_path,"y3","y2__y\nblack\nwhite\nblack\n","targets"))
    with pytest.raises(ValueError,match="target export labels must be unique"):
        nio.load(spec,target="dataset_package")


def test_native_target_metadata_and_weights_keep_qualified_labels(tmp_path):
    config = {"task_type":"regression","global_params":{"delimiter":";","has_header":True}}
    for part,offset,first in [("train",0,"first"),("test",100,"fourth")]:
        for role,text in {"x":("1;2\n3;4\n5;6\n" if part=="train" else "7;8\n9;10\n11;12\n"),"y":f"{101+offset}\n{102+offset}\n{103+offset}\n","group":first+"\nsecond\nthird\n"}.items():
            path=tmp_path/f"{part}_{role}.csv"
            path.write_text(text,encoding="utf-8")
            config[f"{part}_{role}"]=str(path)
            config[f"{part}_{role}_params"]={"has_header":False}
    assembled=nio.load(config,target="dataset_package").to_assembled()
    for part in ["train","test"]:
        assert assembled.blocks[part].y_headers==["0"]
        assert list(assembled.blocks[part].metadata)==[part+"_m__0"]
    record=nio.load(config,target="spectrodataset",spectro_dataset_cls=_NativeTargetRecorder)
    np.testing.assert_array_equal(record.targets[:,0],[101,102,103,201,202,203])
    assert list(record.metadata)==["train_m__0","test_m__0"]
    assert record.metadata.iloc[0,0]=="first" and record.metadata.iloc[3,1]=="fourth"
    weighted={"sources":[_native_target_source(tmp_path,"xw","m;n\n1;2\n3;4\n5;6\n","features"),_native_target_source(tmp_path,"w","m\n0.25\n0.5\n0.75\n","weights")]}
    block=nio.load(weighted,target="dataset_package").to_assembled().blocks["train"]
    assert block.weights_header=="w__m"
    np.testing.assert_array_equal(block.weights,[0.25,0.5,0.75])


def test_native_target_unassigned_metadata_guard_stays_strict(tmp_path):
    source=_native_target_source(tmp_path,"mixed","x;y;metadata\n1;101;a\n2;102;b\n3;103;c\n","mixed",columns=[{"role":"features","select":["x"]},{"role":"targets","select":["y"]}])
    with pytest.raises(ValueError,match="unassigned"):
        nio.load({"sources":[source]},target="dataset_package")



def test_native_target_keyed_suffix_keeps_distinct_partition_rows(tmp_path):
    sources=[]
    for part,ids,offset in [("train",["s1","s2","s3"],100),("test",["t1","t2","t3"],200)]:
        x="id;a;b\n"+"".join(f"{key};{i+1};{i+2}\n" for i,key in enumerate(ids))
        y="id;a\n"+"".join(f"{key};{offset+i+1}\n" for i,key in enumerate(ids))
        sources.append(_native_target_source(tmp_path,part+"_x",x,"mixed",part,key="id",columns=[{"role":"features","select":["a","b"]},{"role":"metadata","select":["id"]}]))
        sources.append(_native_target_source(tmp_path,part+"_y",y,"mixed",part,key="id",columns=[{"role":"targets","select":["a"]},{"role":"ignore","select":["id"]}],join={"to":part+"_x","on":"id","how":"1:1"}))
    spec={"task_type":"regression","sources":sources}
    assembled=nio.load(spec,target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers==["a"] and assembled.blocks["test"].y_headers==["a"]
    np.testing.assert_array_equal(assembled.blocks["train"].y[:,0],[101,102,103])
    np.testing.assert_array_equal(assembled.blocks["test"].y[:,0],[201,202,203])
    record=nio.load(spec,target="spectrodataset",spectro_dataset_cls=_NativeTargetRecorder)
    np.testing.assert_array_equal(record.targets[:,0],[101,102,103,201,202,203])


@pytest.mark.parametrize("case,headers,values", [
    ("c1", [["0", "train_x__0"]], [[[10, 11], [20, 21]]]),
    ("c2", [["train_x__0", "0"]], [[[11, 10], [21, 20]]]),
    ("c3", [["0", "1"], ["0", "1"]], [[[100, 101], [200, 201]], [[10, 11], [20, 21]]]),
    ("c3b", [["0"], ["0"], ["0"]], [[[100], [200]], [[10], [20]], [[30], [40]]]),
    ("c4", [["0"], ["0", "train_x__0"]], [[[100], [200]], [[10, 11], [20, 21]]]),
])
def test_native_merge_frame_order_preserves_independent_values(tmp_path, case, headers, values):
    tables = {
        "c1": [("train_x", "0;train_x__0\n10;11\n20;21\n")],
        "c2": [("train_x", "train_x__0;0\n11;10\n21;20\n")],
        "c3": [("anchor", "0;1\n100;101\n200;201\n"), ("b", "0;1\n10;11\n20;21\n")],
        "c3b": [("anchor", "0\n100\n200\n"), ("b", "0\n10\n20\n"), ("c", "0\n30\n40\n")],
        "c4": [("anchor", "0\n100\n200\n"), ("train_x", "0;train_x__0\n10;11\n20;21\n")],
    }
    sources = [_native_target_source(tmp_path, name, text, "features") for name, text in tables[case]]
    sources.append(_native_target_source(tmp_path, "y", "t\n70\n80\n", "targets"))
    block = nio.load({"task_type": "regression", "sources": sources}, target="dataset_package").to_assembled().blocks["train"]
    assert block.source_ids == [name for name, _ in tables[case]]
    assert block.feature_headers == headers
    for actual, expected in zip(block.X, values, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(block.y[:, 0], [70, 80])
    if case == "c4":
        assert nio.load({"task_type": "regression", "sources": sources}, target="dataset_package")._full["blocks"]["train"]["feature_axes"][1][0]["coordinates"] is None


@pytest.mark.parametrize("role", ["features", "metadata"])
@pytest.mark.parametrize("literal_first", [True, False])
def test_native_merge_selector_order_does_not_claim_other_columns(tmp_path, role, literal_first):
    rules = [{"role": role, "select": ["train_x__0"]}, {"role": "targets", "select": ["0"]}]
    if not literal_first:
        rules.reverse()
    sources = [
        _native_target_source(tmp_path, "anchor", "0\n100\n200\n", "features"),
        _native_target_source(tmp_path, "train_x", "0;train_x__0\n10;11\n20;21\n", "mixed", columns=rules),
    ]
    block = nio.load({"task_type": "regression", "sources": sources}, target="dataset_package").to_assembled().blocks["train"]
    assert block.y_headers == ["0"]
    np.testing.assert_array_equal(block.y[:, 0], [10, 20])
    if role == "features":
        assert block.source_ids == ["anchor", "train_x"]
        assert block.feature_headers[1] == ["train_x__0"]
        np.testing.assert_array_equal(block.X[1], [[11], [21]])
    else:
        assert list(block.metadata) == ["train_x__train_x__0"]
        assert block.metadata.iloc[:, 0].tolist() == [11, 21]


@pytest.mark.parametrize("case", ["c5", "c9", "c10", "c11", "c12", "c14", "c15b", "right_literal"])
def test_native_merge_generated_output_collision_refuses_typed(tmp_path, case):
    calls = []
    def source(name, text, role="features", **kwargs):
        directory = tmp_path / str(len(calls))
        directory.mkdir()
        calls.append(directory)
        return _native_target_source(directory, name, text, role, **kwargs)
    cases = {
        "c5": [source("anchor", "0\n100\n200\n"), source("train_x", "train_x__0;0\n11;10\n21;20\n")],
        "c9": [source("anchor", "0;train_x__0\n100;101\n200;201\n"), source("train_x", "0\n10\n20\n")],
        "c10": [source("x", "0\n1\n2\n"), source("y", "y__0;0\n11;10\n21;20\n", "targets")],
        "c15b": [source("x2", "0\n50\n60\n"), source("y", "x__0\n70\n80\n", "targets"), source("x", "0\n1\n2\n")],
    }
    if case == "c11":
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        first.write_text("x;b__x\n1;2\n3;4\n")
        second.write_text("x\n5\n6\n")
        sources = [{"id": "d", "input": [str(first), str(second)], "role": "features", "merge": "concat_features", "params": {"delimiter": ";", "has_header": True}}]
    elif case in ("c12", "c14", "right_literal"):
        role = "targets" if case != "c14" else "features"
        name = "z" if case == "c14" else "y"
        suffix = "a__" + name
        left_text = "id;a;" + suffix + "\ns1;1;2\ns2;3;4\n" if case != "right_literal" else "id;a\ns1;1\ns2;3\n"
        right_text = "id;a\ns1;9\ns2;8\n" if case != "right_literal" else "id;a;a__y\ns1;9;90\ns2;8;80\n"
        sources = [source("x", left_text, "mixed", key="id", columns=[{"role": "features", "select": ["a", suffix] if case != "right_literal" else ["a"]}]), source(name, right_text, "mixed", key="id", join={"to": "x", "on": "id", "how": "1:1"}, columns=[{"role": role, "select": ["a", "a__y"] if case == "right_literal" else ["a"]}])]
    else:
        sources = cases[case]
    with pytest.raises(ValueError, match="source .*column .*duplicate output column") as error:
        nio.load({"task_type": "regression", "sources": sources}, target="dataset_package")
    assert type(error.value) is ValueError


@pytest.mark.parametrize("case", ["c15", "c13", "c16"])
def test_native_merge_unassigned_keys_and_feature_primary_own_their_values(tmp_path, case):
    feature = _native_target_source(tmp_path, "x", "0;1\n1;2\n3;4\n", "features")
    other = _native_target_source(tmp_path, "y", "0\n70\n80\n", "targets")
    if case != "c15":
        feature = _native_target_source(tmp_path, "x", "id;a\n1;5\n2;6\n", "mixed", key="id", columns=[{"role": "features", "select": ["a"]}])
        other = _native_target_source(tmp_path, "y", "id\n70\n80\n", "metadata" if case == "c16" else "targets")
    sources = [other, feature] if case == "c15" else [feature, other]
    block = nio.load({"task_type": "regression", "sources": sources}, target="dataset_package").to_assembled().blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[1, 2], [3, 4]] if case == "c15" else [[5], [6]])
    if case == "c16":
        assert list(block.metadata) == ["y__id"]
        assert block.metadata.iloc[:, 0].tolist() == [70, 80]
    else:
        assert block.y_headers == ["0" if case == "c15" else "id"]
        np.testing.assert_array_equal(block.y[:, 0], [70, 80])


def test_native_merge_dropped_right_key_does_not_claim_later_source(tmp_path):
    sources = [
        _native_target_source(tmp_path, "x", "id;a\ns1;1\ns2;2\n", "mixed", key="id", columns=[{"role": "features", "select": ["a"]}]),
        _native_target_source(tmp_path, "lookup", "rid;z\ns1;10\ns2;20\n", "mixed", kind="lookup", key="rid", columns=[{"role": "metadata", "select": ["z"]}], join={"to": "x", "left_on": "id", "right_on": "rid", "how": "1:1"}),
        _native_target_source(tmp_path, "y", "rid\n70\n80\n", "targets"),
    ]
    block = nio.load({"task_type": "regression", "sources": sources}, target="dataset_package").to_assembled().blocks["train"]
    assert block.y_headers == ["rid"]
    np.testing.assert_array_equal(block.y[:, 0], [70, 80])
    assert list(block.metadata) == ["z"]
    assert block.metadata.iloc[:, 0].tolist() == [10, 20]


def test_native_merge_forward_literal_target_header_is_not_reversed(tmp_path):
    spec = _native_target_pair(tmp_path, "0;1\n", "0;train_y__0\n101;109\n102;108\n103;107\n", "0;train_y__0\n201;209\n202;208\n203;207\n")
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers == ["0", "train_y__0"]
    assert assembled.blocks["test"].y_headers == ["0", "train_y__0"]
    record = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
    np.testing.assert_array_equal(record.targets, [[101,109],[102,108],[103,107],[201,209],[202,208],[203,207]])


def test_native_target_duplicate_sources_in_two_partitions_still_refuse_adapter(tmp_path):
    sources = []
    for part, ids, offset in [("train", ("ta", "tb"), 100), ("test", ("ea", "eb"), 200)]:
        sources.append(_native_target_source(tmp_path, part+"_x", "x\n1\n2\n3\n", "features", part))
        for index, source_id in enumerate(ids):
            sources.append(_native_target_source(tmp_path, source_id, f"y\n{offset+index*10+1}\n{offset+index*10+2}\n{offset+index*10+3}\n", "targets", part))
    spec = {"task_type": "regression", "sources": sources}
    assembled = nio.load(spec, target="dataset_package").to_assembled()
    assert assembled.blocks["train"].y_headers == ["y", "tb__y"]
    assert assembled.blocks["test"].y_headers == ["y", "eb__y"]
    np.testing.assert_array_equal(assembled.blocks["test"].y, [[201,211],[202,212],[203,213]])
    with pytest.raises(ValueError, match="target.*layout"):
        nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
