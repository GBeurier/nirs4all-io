# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Literal owner regressions for the accepted Q-ROW / Q-DEC A1-A10 batch."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import nirs4all_io as nio

_FIXTURE = json.loads((Path(__file__).parent / "fixtures/q_row_key_policy_v1.json").read_text())
_CASES = [(row, method, form, target) for row in _FIXTURE["rows"] for method in row["methods"] for form in ("dir", "list") for target in ("assembled", "dataset_package")]


@pytest.mark.parametrize("case,method,form,target", _CASES, ids=[f"{r['id']}/{m}/{f}/{t}" for r, m, f, t in _CASES])
def test_literal_owner_routes(tmp_path, case, method, form, target):
    for name, text in {**case["files"], **case.get("extra", {})}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    inp = tmp_path if form == "dir" else list(reversed(sorted(tmp_path.glob("*.csv"))))
    kwargs = {}
    explicit = method.startswith("explicit")
    original = None
    if explicit:
        inp = {"sources": [{"id": "x", "role": "features", "input": "Xcal.csv"}, {"id": "y", "role": "targets", "input": "Ycal.csv", "params": case.get("params", {})}]}
        if "source_mode" in case:
            inp["params"] = {"categorical": case["global_mode"]}
            inp["sources"][1]["params"]["categorical"] = case["source_mode"]
        if method == "explicit-m1":
            inp["sample_index"] = {"by": "id", "key": "id"}
            for source in inp["sources"]:
                source["key"] = "id"
            inp["sources"][1]["join"] = {"to": "x", "on": "id", "how": "m:1"}
        original = copy.deepcopy(inp)
        kwargs["base_dir"] = tmp_path
    elif method == "infer":
        inp = nio.infer(inp)
    if case.get("provisional"):
        spec, _ = nio.to_spec(inp)
        assert isinstance(spec, nio.DatasetSpec), "configuration must remain editable"
    if case["status"] == "refused":
        with pytest.raises(ValueError):
            nio.load(inp, target=target, **kwargs)
        return
    result = nio.load(inp, target=target, **kwargs)
    assembled = result if target == "assembled" else result.to_assembled()
    block = assembled.blocks["train"]
    expected = np.asarray([[np.nan if value is None else value for value in row] for row in case["targets"]])
    np.testing.assert_allclose(block.y, expected, rtol=0, atol=0, equal_nan=True)
    assert block.n_samples == len(expected)
    if case.get("headers"):
        assert block.y_headers == case["headers"]
    if "identity" in case:
        assert block.feature_headers == [["400", "500"]]
        assert block.metadata["id"].tolist() == case["identity"]
    if case.get("no_identity"):
        assert (result.identity["sample_id"] if target == "dataset_package" else assembled.identity["sample_id"]) is None
    if case.get("repetition"):
        assert inp.repetition is not None if hasattr(inp, "repetition") else inp.resolved_spec.repetition is not None
    if original is not None:
        assert inp == original


def test_fixture_mirrors_are_literal():
    root = Path(__file__).parents[1]
    expected = (root / "tests/fixtures/q_row_key_policy_v1.json").read_bytes()
    for crate in ("nirs4all-io", "nirs4all-io-core"):
        assert (root / f"crates/{crate}/tests/fixtures/q_row_key_policy_v1.json").read_bytes() == expected


@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_duplicate_beyond_discovery_still_refuses(tmp_path, target):
    control = _FIXTURE["controls"]["bounded"]["later_duplicate"]
    x = control["x"] + "".join(f"s{i};{i}.25\n" for i in range(control["rows"]))
    y = control["y"] + "".join(f"s{i if i != control['duplicate_row'] else 0};{i}.5\n" for i in range(control["rows"]))
    (tmp_path / "Xcal.csv").write_text(x)
    (tmp_path / "Ycal.csv").write_text(y)
    assert isinstance(nio.to_spec(tmp_path)[0], nio.DatasetSpec)
    with pytest.raises(ValueError, match="duplicate"):
        nio.load(tmp_path, target=target)


def test_multi_input_discovery_never_uses_only_first(tmp_path):
    from nirs4all_io.infer.identity import source_preview
    from nirs4all_io.resolve.resolver import _file_reference_table, resolve

    control = _FIXTURE["controls"]["multi_input"]
    for name, text in zip(control["inputs"], [control["first"], control["second"]], strict=True):
        (tmp_path / name).write_text(text)
    spec = {"sources": [{"id": "x", "role": "features", "input": control["inputs"], "merge": "concat_samples"}]}
    refs = _file_reference_table(resolve(tmp_path), spec)
    assert source_preview(spec["sources"][0], refs) is None


@pytest.mark.parametrize("lock", _FIXTURE["controls"]["locks"], ids=[v["value"].get("mode", "identity") for v in _FIXTURE["controls"]["locks"]])
def test_explicit_edited_proposal_choices_remain_authoritative(tmp_path, lock):
    # Oracle equivalent of edited owner specs. The actual native confirmed-lock
    # API is tested in Rust and remains UNRUN in this source-only lane.
    row = next(row for row in _FIXTURE["rows"] if row["id"] == "shuffled")
    for name, text in row["files"].items():
        (tmp_path / name).write_text(text)
    spec, base = nio.to_spec(tmp_path)
    value = spec.to_dict()
    if lock["value"].get("mode") == "row_order":
        value["sources"][1].pop("key", None)
        value["sources"][1].pop("join", None)
        value["sources"][1]["role"] = "mixed"
        # Explicit column selection keeps identity out of the target payload.
        value["sources"][1]["columns"] = [{"role": "targets", "select": ["Protein, %"]}, {"role": "metadata", "select": "rest"}]
    elif lock["kind"] == "identity":
        value["sample_index"] = lock["value"]
    else:
        value["sources"][1]["join"] = {"to": value["sources"][0]["id"], "left_on": "id", "right_on": "id", "cardinality": "1:1"}
    saved = copy.deepcopy(value)
    a = nio.load(value, base_dir=base, target="assembled")
    np.testing.assert_array_equal(a.blocks["train"].y[:, -1], lock["target_values"])
    assert value == saved


@pytest.mark.parametrize("case", _FIXTURE["controls"]["positive_counterexamples"], ids=[v["id"] for v in _FIXTURE["controls"]["positive_counterexamples"]])
@pytest.mark.parametrize("method", ["auto", "infer"])
def test_positive_decimal_and_width_counterexamples(tmp_path, case, method):
    (tmp_path / "Xcal.csv").write_text(case["x"])
    (tmp_path / "Ycal.csv").write_text(case["y"])
    inp = tmp_path if method == "auto" else nio.infer(tmp_path)
    for target in ("assembled", "dataset_package"):
        out = nio.load(inp, target=target)
        block = (out if target == "assembled" else out.to_assembled()).blocks["train"]
        np.testing.assert_array_equal(block.y, case["targets"])


def test_lone_x_identity_is_real_metadata(tmp_path):
    case = _FIXTURE["controls"]["lone_x"]
    for name, text in case["files"].items():
        (tmp_path / name).write_text(text)
    block = nio.load(tmp_path, target="assembled").blocks["train"]
    assert block.metadata["id"].tolist() == case["identity"]
    assert block.feature_headers == [["400"]]
    np.testing.assert_array_equal(block.X[0], case["features"])


def test_explicit_changed_identity_keeps_separate_join_keys(tmp_path):
    case = _FIXTURE["controls"]["identity_lock_change"]
    for name, text in case["files"].items():
        (tmp_path / name).write_text(text)
    spec = nio.infer(tmp_path).accept().to_dict()
    spec["sample_index"] = case["value"]
    spec["sources"][0]["key"] = case["value"]["key"]
    saved = copy.deepcopy(spec)
    block = nio.load(spec, base_dir=tmp_path, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.y[:, 0], case["targets"])
    assert block.metadata["sample_alt"].tolist() == ["a", "b", "c"]
    assert spec == saved


_A5_X = "400;500\n0.1;0.2\n0.3;0.4\n0.5;0.6\n"
_A5_Y = "y\n1.5\n2.5\n3.5\n"


def _a5_files(root, files):
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")


def _a5_warning(partition, column, source):
    return f"partition {partition}: metadata label '{column}' in source '{source}' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity"


@pytest.mark.parametrize("method", ["automatic", "infer"])
@pytest.mark.parametrize("form", ["dir", "list"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
@pytest.mark.parametrize("labels,other,values", [
    (["Smith, J", "Doe, B", "Lee, C"], "site", ["A", "B", "C"]),
    (["12,1", "12,10", "14,2"], "site", ["A", "B", "C"]),
    (["A", "B", "C"], "ph", ["6,5", "7,5", "8,5"]),
], ids=["text-labels", "distinct-comma-labels", "numeric-ph"])
def test_a5_public_metadata_labels_remain_unverified_rows(tmp_path, method, form, target, labels, other, values):
    from nirs4all_io.materialize.package import DatasetPackage

    _a5_files(tmp_path, {"Xcal.csv": _A5_X, "Ycal.csv": _A5_Y, "Mcal.csv": f"name;{other}\n" + "".join(f"{label};{value}\n" for label, value in zip(labels, values, strict=True))})
    inp = tmp_path if form == "dir" else sorted(tmp_path.glob("*.csv"))
    plan = nio.infer(inp) if method == "infer" else None
    selected = plan if plan is not None else inp
    spec, _ = nio.to_spec(selected)
    assert spec.sample_index.by.value == "row"
    if plan is not None:
        assert plan.identity is None
        assert plan.alignment == []
        assert "identity_repetition" not in plan.params
        assert _a5_warning("train", "name", "train_m") in plan.warnings
    for source in spec.to_dict()["sources"]:
        assert source.get("key") is None
        assert not any(source.get("join", {}).get(field) for field in ("on", "left_on", "right_on"))
    result = nio.load(selected, target=target)
    assembled = result if target == "assembled" else result.to_assembled()
    block = assembled.blocks["train"]
    np.testing.assert_array_equal(block.X[0], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_equal(block.y, [[1.5], [2.5], [3.5]])
    assert block.metadata["name"].tolist() == labels
    assert block.metadata[other].tolist() == ([6.5, 7.5, 8.5] if other == "ph" else values)
    package = result if target == "dataset_package" else DatasetPackage.from_assembled(assembled)
    assert package.identity["sample_id"] is None
    assert package.row_position_fallback.used is True


def _a5_generate(root, sources, files):
    from nirs4all_io.infer.identity import key_sources_by_id
    from nirs4all_io.infer.plan import DatasetPlan
    from nirs4all_io.resolve.resolver import _file_reference_table, resolve

    _a5_files(root, files)
    value = {"sources": sources}
    refs = _file_reference_table(resolve(root), value)
    plan = DatasetPlan()
    key_sources_by_id(value, refs, plan)
    return value, plan


def _a5_sources(partition="train", suffix=""):
    return [
        {"id": "x" + suffix, "input": "X" + suffix + ".csv", "role": "features", "partition": partition},
        {"id": "y" + suffix, "input": "Y" + suffix + ".csv", "role": "targets", "partition": partition, "join": {"to": "x" + suffix}},
        {"id": "m" + suffix, "input": "M" + suffix + ".csv", "role": "metadata", "partition": partition, "join": {"to": "x" + suffix}},
    ]


@pytest.mark.parametrize("case", ["shared", "anchor-id", "target-code", "target-name", "column-presence", "metadata-sharing", "mixed-partitions", "two-local-partitions", "reordered", "unequal"])
def test_a5_causal_identity_controls(tmp_path, case):
    sources = _a5_sources()
    files = {"X.csv": _A5_X, "Y.csv": _A5_Y, "M.csv": "name;site\ns1;A\ns2;B\ns3;C\n"}
    key = None
    refused = False
    warnings = 1
    if case == "shared":
        files.update({"X.csv": "name;400\ns1;0.1\ns2;0.3\ns3;0.5\n", "Y.csv": "name;y\ns3;3.5\ns1;1.5\ns2;2.5\n"})
        key, warnings = "name", 0
    elif case == "anchor-id":
        files.update({"X.csv": "id;400\ns1;0.1\ns2;0.3\ns3;0.5\n", "Y.csv": "id;y\ns3;3.5\ns1;1.5\ns2;2.5\n"})
        key, refused = "id", True
    elif case == "target-code":
        files["Y.csv"] = "code;y\ns1;1.5\ns2;2.5\ns3;3.5\n"
        key, refused = "code", True
    elif case == "target-name":
        files["Y.csv"] = "name\ns1\ns2\ns3\n"
        key, refused, warnings = "name", True, 0
    elif case == "column-presence":
        files.update({"X.csv": "code;400\n1.5;0.1\n1.5;0.3\n2.5;0.5\n", "M.csv": "code;site\n1;A\n2;B\n3;C\n"})
        key, refused, warnings = "code", True, 0
    elif case == "metadata-sharing":
        sources.append({"id": "m2", "input": "M2.csv", "role": "metadata", "partition": "train", "join": {"to": "x"}})
        files["M2.csv"] = "name;batch\ns1;a\ns2;b\ns3;c\n"
        warnings = 2
    elif case in ("mixed-partitions", "two-local-partitions"):
        sources += _a5_sources("test", "t")
        files.update({"Xt.csv": _A5_X, "Yt.csv": _A5_Y, "Mt.csv": "sample_id;site\na;A\nb;B\nc;C\n"})
        warnings = 2
        if case == "mixed-partitions":
            # No metadata table in the independently keyed test partition.
            sources.pop()
            files.pop("Mt.csv")
            files.update({"Xt.csv": "id;400\na;0.1\nb;0.3\nc;0.5\n", "Yt.csv": "id;y\na;1.5\nb;2.5\nc;3.5\n"})
            key, refused, warnings = "id", True, 1
    elif case == "reordered":
        files["M.csv"] = "name;site\ns3;C\ns1;A\ns2;B\n"
    elif case == "unequal":
        files["M.csv"] += "s4;D\ns5;E\n"
        refused = True
    value, plan = _a5_generate(tmp_path, sources, files)
    if refused:
        with pytest.raises(ValueError):
            nio.load(value, base_dir=tmp_path, target="assembled")
    else:
        package = nio.load(value, base_dir=tmp_path, target="dataset_package")
        block = package.to_assembled().blocks["train"]
        np.testing.assert_array_equal(block.y, [[1.5], [2.5], [3.5]])
        if case == "reordered":
            assert block.metadata["name"].tolist() == ["s3", "s1", "s2"]
        assert package.row_position_fallback.used is (key is None)
    assert value.get("sample_index", {}).get("key") == key
    assert sum("not shared with features or targets" in warning for warning in plan.warnings) == warnings
    if key is None:
        assert plan.identity is None and not plan.alignment
        assert "identity_repetition" not in plan.params


def test_a5_explicit_and_accepted_missing_identity_still_refuse(tmp_path):
    _a5_files(tmp_path, {"Xcal.csv": _A5_X, "Ycal.csv": _A5_Y, "Mcal.csv": "name;site\ns1;A\ns2;B\ns3;C\n"})
    explicit = {"sample_index": {"by": "id", "key": "name"}, "sources": [{"id": "x", "role": "features", "input": "Xcal.csv", "key": "name"}, {"id": "y", "role": "targets", "input": "Ycal.csv", "join": {"to": "x", "on": "name"}}]}
    saved = copy.deepcopy(explicit)
    with pytest.raises(ValueError):
        nio.load(explicit, base_dir=tmp_path, target="assembled")
    assert explicit == saved
    plan = nio.infer(tmp_path)
    absent = copy.deepcopy(explicit)
    absent["sample_index"]["key"] = "absent"
    accepted = plan.accept(sample_index=nio.DatasetSpec.from_dict(absent).sample_index)
    before = accepted.to_dict()
    with pytest.raises(ValueError):
        nio.load(accepted, target="assembled")
    assert accepted.to_dict() == before


@pytest.mark.parametrize("method", ["automatic", "infer"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
@pytest.mark.parametrize("column,values,expected,comma", [
    ("name", ["12,5", "13,1", "14,2"], ["12,5", "13,1", "14,2"], False),
    ("index", ["1,33", "2,44", "3,55"], ["1,33", "2,44", "3,55"], False),
    ("ph", ["6,5", "7,5", "8,5"], [6.5, 7.5, 8.5], True),
    ("id", ["1.1", "2.2", "3.3"], [1.1, 2.2, 3.3], False),
])
def test_a5_i6_typing_and_float_gate(tmp_path, method, target, column, values, expected, comma):
    _a5_files(tmp_path, {"Xcal.csv": _A5_X, "Ycal.csv": _A5_Y, "Mcal.csv": f"{column};site\n" + "".join(f"{v};A\n" for v in values)})
    selected = tmp_path if method == "automatic" else nio.infer(tmp_path)
    spec, _ = nio.to_spec(selected)
    metadata = next(source for source in spec.sources if source.role.value == "metadata")
    assert (metadata.params.decimal_separator == ",") is comma
    if method == "infer":
        assert ("decimal_separator" in selected.params["Mcal.csv"]) is comma
        assert selected.identity is None
        assert any("not shared with features or targets" in warning for warning in selected.warnings) is (column in ("name", "index"))
    result = nio.load(selected, target=target)
    block = (result if target == "assembled" else result.to_assembled()).blocks["train"]
    assert block.metadata[column].tolist() == expected


def _a5_combined(root, method, target):
    _a5_files(root, {"Xcal.csv": _A5_X, "Ycal.csv": _A5_Y, "Mcal.csv": "name;ph\n12,1;6,5\n12,10;7,5\n14,2;8,5\n"})
    selected = root if method == "automatic" else nio.infer(root)
    spec, _ = nio.to_spec(selected)
    metadata = next(source for source in spec.sources if source.role.value == "metadata")
    assert metadata.params.decimal_separator == ","
    result = nio.load(selected, target=target)
    block = (result if target == "assembled" else result.to_assembled()).blocks["train"]
    np.testing.assert_array_equal(block.metadata["ph"], [6.5, 7.5, 8.5])
    if method == "infer":
        assert "partition train: metadata label 'name' in source 'train_m' also reads as numbers under decimal ','; source-wide decimals can rewrite or merge its values; declare decimal_separator '.' to keep comma-form values as text in this source" in selected.warnings
        assert selected.params["Mcal.csv"]["decimal_separator"]["value"] == ","
    return block


@pytest.mark.parametrize("method", ["automatic", "infer"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_a5_i6_combined_keeps_numeric_ph_and_discloses_limit(tmp_path, method, target):
    _a5_combined(tmp_path, method, target)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q-META-LEXEME: source-wide comma decimal cannot preserve distinct numeric-looking metadata labels")
@pytest.mark.parametrize("method", ["automatic", "infer"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_a5_i6_combined_distinct_labels_remain_unqualified(tmp_path, method, target):
    block = _a5_combined(tmp_path, method, target)
    assert block.metadata["name"].tolist() == ["12,1", "12,10", "14,2"]


@pytest.mark.parametrize("separator", [",", "."])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_a5_i6_explicit_decimals_are_caller_owned(tmp_path, separator, target):
    _a5_files(tmp_path, {"X.csv": _A5_X, "Y.csv": _A5_Y, "M.csv": "name;ph\n12,1;6,5\n12,10;7,5\n14,2;8,5\n"})
    spec = {"sources": _a5_sources()}
    spec["sources"][2]["params"] = {"decimal_separator": separator}
    before = copy.deepcopy(spec)
    result = nio.load(spec, base_dir=tmp_path, target=target)
    block = (result if target == "assembled" else result.to_assembled()).blocks["train"]
    if separator == ".":
        assert block.metadata["name"].tolist() == ["12,1", "12,10", "14,2"]
        assert block.metadata["ph"].tolist() == ["6,5", "7,5", "8,5"]
    else:
        np.testing.assert_array_equal(block.metadata["ph"], [6.5, 7.5, 8.5])
    assert spec == before


@pytest.mark.parametrize("case", ["shared-column", "multi-input", "global-key"])
def test_a5_i6_neutral_profile_cannot_promote_identity(tmp_path, case):
    sources = _a5_sources()
    sources[2]["params"] = {"decimal_separator": ","}
    files = {"X.csv": _A5_X, "Y.csv": _A5_Y, "M.csv": "name;site\n12,1;A\n12,10;B\n14,2;C\n"}
    if case == "shared-column":
        files["Y.csv"] = "name\n1.5\n2.5\n3.5\n"
    elif case == "multi-input":
        files["M2.csv"] = files["M.csv"]
        sources[2]["input"] = ["M.csv", "M2.csv"]
        sources[2]["merge"] = "concat_samples"
    else:
        sources += _a5_sources("test", "t")[:2]
        files.update({"Xt.csv": "id;400\na;0.1\nb;0.3\nc;0.5\n", "Yt.csv": "id;y\na;1.5\nb;2.5\nc;3.5\n"})
    value, plan = _a5_generate(tmp_path, sources, files)
    assert value["sources"][2]["params"]["decimal_separator"] == ","
    assert not any("metadata label" in warning for warning in plan.warnings)
    if case == "global-key":
        assert value["sample_index"]["key"] == "id"
        with pytest.raises(ValueError):
            nio.load(value, base_dir=tmp_path, target="assembled")
    else:
        assert "sample_index" not in value and plan.identity is None


@pytest.mark.parametrize("method", ["automatic", "infer"])
def test_a5_low_score_group_keeps_its_real_join_refusal(tmp_path, method):
    _a5_files(tmp_path, {
        "Xcal.csv": "id;400\n1.5;0.1\n1.5;0.3\n2.5;0.5\n",
        "Ycal.csv": "id;y\n2.5;30\n1.5;10\n1.5;20\n",
        "Xval.csv": "id;400\na;0.1\nb;0.3\nc;0.5\n",
        "Yval.csv": "id;y\na;1.5\nb;2.5\nc;3.5\n",
    })
    selected = tmp_path if method == "automatic" else nio.infer(tmp_path)
    for target in ("assembled", "dataset_package"):
        with pytest.raises(ValueError):
            nio.load(selected, target=target)


@pytest.mark.parametrize("method", ["automatic", "infer"])
def test_a5_i6_numeric_targets_never_become_metadata_labels(tmp_path, method):
    _a5_files(tmp_path, {"Xcal.csv": _A5_X, "Ycal.csv": "code;y\n1,33;10\n2,44;20\n3,55;30\n"})
    selected = tmp_path if method == "automatic" else nio.infer(tmp_path)
    for target in ("assembled", "dataset_package"):
        result = nio.load(selected, target=target)
        block = (result if target == "assembled" else result.to_assembled()).blocks["train"]
        np.testing.assert_array_equal(block.y, [[1.33, 10], [2.44, 20], [3.55, 30]])
