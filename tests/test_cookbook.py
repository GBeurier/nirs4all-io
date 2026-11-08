# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Cookbook coverage matrix (Epic 2.4 gate).

Each case is a real on-disk situation + its DatasetSpec. The vocabulary each
spec exercises is **introspected from the spec** (not hand-asserted), and the
final test fails if any load-supported E.1/E.2 vocabulary element has zero
fixtures -- undocumented/untested vocabulary is treated as unshipped.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.spec import DatasetSpec
from nirs4all_io.spec.enums import MergeMode, SourceKind

# load-supported vocabulary that MUST each have >=1 cookbook fixture
CATALOGUE = {
    "selector:positional", "selector:slice", "selector:names", "selector:name_range", "selector:regex", "selector:dtype", "selector:rest",
    "merge:concat_samples", "merge:concat_features", "merge:by_key",
    "cardinality:1:1", "cardinality:m:1", "cardinality:1:m",
    "coverage:complete", "coverage:warn", "coverage:drop", "coverage:error",
    "partition:column", "partition:index", "partition:index_file",
    "folds:inline", "folds:file", "folds:column",
    "lookup", "composite_key",
    "variations", "role:weights", "selector:auto",
}


def vocab_elements(spec: DatasetSpec) -> set[str]:
    """Introspect which vocabulary elements a spec exercises (generates the matrix)."""
    els: set[str] = set()
    if spec.partitions and spec.partitions.by:
        els.add(f"partition:{spec.partitions.by.value}")
    if spec.folds:
        if spec.folds.inline:
            els.add("folds:inline")
        if spec.folds.file:
            els.add("folds:file")
        if spec.folds.column:
            els.add("folds:column")
    for s in spec.sources:
        if s.merge is not MergeMode.NONE:
            els.add(f"merge:{s.merge.value}")
        if s.kind is SourceKind.LOOKUP:
            els.add("lookup")
        if isinstance(s.key, list):
            els.add("composite_key")
        if s.key == "filename_stem":
            els.add("virtual_key")
        for c in s.columns:
            els.add(f"selector:{c.select.kind}")
            if c.role.value == "weights":
                els.add("role:weights")
        if s.join:
            els.add(f"cardinality:{s.join.cardinality.value}")
            els.add(f"coverage:{s.join.coverage.value}")
        if s.variations:
            els.add("variations")
    return els


def _wl(n=6, start=1000, step=5):
    return [str(start + i * step) for i in range(n)]


def _build(tmp, files):
    for name, df in files.items():
        df.to_csv(tmp / name, sep=";", index=False)


# (id, files, spec) — files written into a per-case tmp dir
def _cases():
    wl = _wl()
    X6 = lambda n=6: pd.DataFrame(np.random.rand(n, len(wl)), columns=wl)  # noqa: E731

    cases: list[tuple[str, dict, dict]] = []

    # L.2 — slice features + positional last target
    df = X6()
    df["y"] = np.arange(6.0)
    cases.append(("L2_slice_positional", {"data.csv": df}, {"sources": [{"id": "d", "role": "mixed", "input": "data.csv", "columns": [{"role": "features", "select": "0:6"}, {"role": "targets", "select": -1}]}]}))

    # L.3 — regex + names + dtype + rest
    df = X6()
    df["protein"] = np.arange(6.0)
    df["site"] = ["a", "b", "a", "b", "a", "b"]
    df["notes"] = ["n"] * 6
    cases.append(("L3_regex_names_dtype_rest", {"d.csv": df}, {"sources": [{"id": "d", "role": "mixed", "input": "d.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["protein"]}, {"role": "metadata", "select": {"dtype": "string"}}, {"role": "ignore", "select": "rest"}]}]}))

    # L.18a — name_range + rest
    df = X6()
    df["protein"] = np.arange(6.0)
    cases.append(("L18a_name_range_rest", {"w.csv": df}, {"sources": [{"id": "a", "role": "mixed", "input": "w.csv", "columns": [{"role": "features", "select": {"name_range": [wl[0], wl[-1]]}}, {"role": "targets", "select": ["protein"]}, {"role": "metadata", "select": "rest"}]}]}))

    # L.6 — concat_samples glob
    cases.append(("L6_concat_samples", {"b1.csv": X6(3).assign(y=[1.0, 2, 3]), "b2.csv": X6(3).assign(y=[4.0, 5, 6])}, {"sources": [{"id": "d", "role": "mixed", "input": ["b1.csv", "b2.csv"], "merge": "concat_samples", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}]}]}))

    # L.7 — concat_features by key
    cases.append(("L7_concat_features", {"nir.csv": pd.DataFrame({"id": [1, 2, 3], "n1": [0.1, 0.2, 0.3]}), "mir.csv": pd.DataFrame({"id": [1, 2, 3], "m1": [0.4, 0.5, 0.6]})}, {"sources": [{"id": "x", "role": "features", "input": ["nir.csv", "mir.csv"], "merge": "concat_features", "key": "id"}]}))

    # L.18c — by_key merge of 3 files
    cases.append(("L18c_by_key", {"a.csv": pd.DataFrame({"id": [1, 2], "a": [1.0, 2]}), "b.csv": pd.DataFrame({"id": [1, 2], "b": [3.0, 4]}), "c.csv": pd.DataFrame({"id": [1, 2], "c": [5.0, 6]})}, {"sources": [{"id": "d", "role": "features", "input": ["a.csv", "b.csv", "c.csv"], "merge": "by_key", "key": "id"}]}))

    # L.4 — 3 files 1:1 (default complete)
    cases.append(("L4_1to1", {"X.csv": X6(), "Y.csv": pd.DataFrame({"y": np.arange(6.0)})}, {"sources": [{"id": "x", "role": "features", "input": "X.csv"}, {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}}]}))

    # L.8 flagship — concat_samples + m:1 lookup complete
    meas = lambda site: pd.DataFrame({wl[0]: [0.1, 0.2], wl[1]: [0.3, 0.4], "protein": [1.0, 2.0], "site": site})  # noqa: E731
    cases.append(("L8_flagship_m1_complete", {"ba.csv": meas(["S1", "S2"]), "bb.csv": meas(["S2", "S1"]), "sites.csv": pd.DataFrame({"site": ["S1", "S2"], "region": ["n", "s"]})}, {"sources": [{"id": "m", "role": "mixed", "input": ["ba.csv", "bb.csv"], "merge": "concat_samples", "columns": [{"role": "features", "select": [wl[0], wl[1]]}, {"role": "targets", "select": ["protein"]}, {"role": "metadata", "select": ["site"]}]}, {"id": "sites", "kind": "lookup", "input": "sites.csv", "columns": [{"role": "metadata", "select": ["region"]}], "join": {"to": "m", "on": "site", "how": "m:1", "coverage": "complete"}}]}))

    # m:1 coverage warn (missing key kept + null-filled)
    cases.append(("m1_warn", {"x.csv": pd.DataFrame({wl[0]: [0.1, 0.2, 0.3], "site": ["S1", "S2", "S3"]}), "s.csv": pd.DataFrame({"site": ["S1", "S2"], "region": ["n", "s"]})}, {"sources": [{"id": "x", "role": "mixed", "input": "x.csv", "columns": [{"role": "features", "select": [wl[0]]}, {"role": "metadata", "select": ["site"]}]}, {"id": "s", "kind": "lookup", "input": "s.csv", "columns": [{"role": "metadata", "select": ["region"]}], "join": {"to": "x", "on": "site", "how": "m:1", "coverage": "warn"}}]}))

    # m:1 coverage drop
    cases.append(("m1_drop", {"x.csv": pd.DataFrame({wl[0]: [0.1, 0.2, 0.3], "site": ["S1", "S2", "S3"]}), "s.csv": pd.DataFrame({"site": ["S1", "S2"], "region": ["n", "s"]})}, {"sources": [{"id": "x", "role": "mixed", "input": "x.csv", "columns": [{"role": "features", "select": [wl[0]]}, {"role": "metadata", "select": ["site"]}]}, {"id": "s", "kind": "lookup", "input": "s.csv", "columns": [{"role": "metadata", "select": ["region"]}], "join": {"to": "x", "on": "site", "how": "m:1", "coverage": "drop"}}]}))

    # 1:1 keyed join, coverage error (all keys present -> ok; exercises 'error')
    cases.append(("c1to1_error", {"X.csv": X6().assign(id=list(range(6))), "Y.csv": pd.DataFrame({"id": list(range(6)), "y": np.arange(6.0)})}, {"sample_index": {"by": "id", "key": "id"}, "sources": [{"id": "x", "role": "mixed", "input": "X.csv", "key": "id", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}]}, {"id": "y", "role": "targets", "input": "Y.csv", "key": "id", "columns": [{"role": "targets", "select": ["y"]}], "join": {"to": "x", "on": "id", "how": "1:1", "coverage": "error"}}]}))

    # 1:m expansion
    cases.append(("c1tom", {"x.csv": pd.DataFrame({"sid": [1, 2], wl[0]: [0.1, 0.2]}), "t.csv": pd.DataFrame({"sid": [1, 1, 2], "y": [10.0, 11, 20]})}, {"sources": [{"id": "x", "role": "mixed", "input": "x.csv", "key": "sid", "columns": [{"role": "features", "select": [wl[0]]}]}, {"id": "t", "role": "targets", "input": "t.csv", "columns": [{"role": "targets", "select": ["y"]}], "join": {"to": "x", "left_on": "sid", "right_on": "sid", "how": "1:m"}}]}))

    # composite key 1:1
    cases.append(("composite", {"x.csv": pd.DataFrame({"a": [1, 1], "b": ["p", "q"], wl[0]: [0.1, 0.2]}), "y.csv": pd.DataFrame({"a": [1, 1], "b": ["p", "q"], "y": [1.0, 2.0]})}, {"sample_index": {"by": "id", "key": ["a", "b"]}, "sources": [{"id": "x", "role": "mixed", "input": "x.csv", "key": ["a", "b"], "columns": [{"role": "features", "select": [wl[0]]}]}, {"id": "y", "role": "targets", "input": "y.csv", "key": ["a", "b"], "columns": [{"role": "targets", "select": ["y"]}], "join": {"left": "x", "right": "y", "left_on": ["a", "b"], "right_on": ["a", "b"], "how": "1:1", "coverage": "complete"}}]}))

    # L.10 — column split
    df = X6(8)
    df["y"] = np.arange(8.0)
    df["set"] = ["cal", "cal", "cal", "cal", "val", "val", "val", "val"]
    cases.append(("L10_column_split", {"all.csv": df}, {"sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["set"]}]}], "partitions": {"by": "column", "column": "set", "train_values": ["cal"], "test_values": ["val"]}}))

    # L.11 — explicit row-index split (deterministic by construction; no shuffle at load time)
    df = X6(10)
    df["y"] = np.arange(10.0)
    cases.append(("L11_index", {"all.csv": df}, {"sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}]}], "partitions": {"by": "index", "train": [0, 1, 2, 3, 4, 5, 6], "test": [7, 8, 9]}}))

    # auto selector for targets (deterministic at load time: last numeric column)
    df_auto = X6(6).assign(y=np.arange(6.0))
    cases.append(("auto_targets", {"data.csv": df_auto}, {"sources": [{"id": "d", "role": "mixed", "input": "data.csv", "columns": [{"role": "targets", "select": "auto"}, {"role": "features", "select": "rest"}]}]}))

    # folds inline
    cases.append(("folds_inline", {"X.csv": X6(), "Y.csv": pd.DataFrame({"y": np.arange(6.0)})}, {"sources": [{"id": "x", "role": "features", "input": "X.csv"}, {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}}], "folds": {"inline": [{"train": [0, 1, 2, 3], "val": [4, 5]}]}}))

    # folds by column (each distinct value of cv_fold -> one fold)
    df_cv = X6(6).assign(y=np.arange(6.0), cv_fold=[0, 1, 0, 1, 0, 1])
    cases.append(("folds_column", {"all.csv": df_cv}, {"sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["cv_fold"]}]}], "folds": {"column": "cv_fold"}}))

    # weights as a column role: surfaced on the SpectroDataset as a `__sample_weight__` metadata column
    df_w = X6(6).assign(y=np.arange(6.0), w=[1.0, 1.0, 2.0, 0.5, 1.0, 1.5])
    cases.append(("role_weights", {"data.csv": df_w}, {"sources": [{"id": "d", "role": "mixed", "input": "data.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}, {"role": "weights", "select": ["w"]}]}]}))

    # variations: a feature source with a pre-computed preprocessing variant (e.g. SNV) -> processings on the SpectroDataset
    snv_df = X6(6)  # second file with identical column headers, different values
    cases.append(("variations_snv", {"X.csv": X6(6), "X_snv.csv": snv_df, "Y.csv": pd.DataFrame({"y": np.arange(6.0)})}, {"sources": [{"id": "x", "role": "features", "input": "X.csv", "variations": [{"name": "snv", "input": "X_snv.csv"}]}, {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}}]}))

    return cases


@pytest.fixture(params=_cases(), ids=lambda c: c[0])
def cookbook_case(request, tmp_path):
    case_id, files, spec_dict = request.param
    _build(tmp_path, files)
    return case_id, spec_dict, tmp_path


def test_cookbook_case_loads(cookbook_case):
    _, spec_dict, base = cookbook_case
    asm = nio.load(spec_dict, base_dir=base, target="assembled")
    assert asm.blocks  # produced at least one partition block
    for block in asm.blocks.values():
        assert block.X and block.X[0].shape[0] == block.n_samples


def test_folds_file_case(tmp_path):
    # folds:file needs an extra fold file written alongside
    pd.DataFrame(np.random.rand(6, 6), columns=_wl()).to_csv(tmp_path / "X.csv", sep=";", index=False)
    pd.DataFrame({"y": np.arange(6.0)}).to_csv(tmp_path / "Y.csv", sep=";", index=False)
    (tmp_path / "folds.csv").write_text("fold_0,fold_1\n0,2\n1,3\n,4\n,5\n", encoding="utf-8")
    spec = {"sources": [{"id": "x", "role": "features", "input": "X.csv"}, {"id": "y", "role": "targets", "input": "Y.csv", "join": {"to": "x", "how": "1:1"}}], "folds": {"file": "folds.csv", "format": "csv"}}
    asm = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert len(asm.folds) == 2


def test_index_file_case(tmp_path):
    # partitions:index_file needs index files written alongside
    df = pd.DataFrame(np.random.rand(8, 6), columns=_wl())
    df["y"] = np.arange(8.0)
    df.to_csv(tmp_path / "all.csv", sep=";", index=False)
    (tmp_path / "train_idx.txt").write_text("0\n1\n2\n3\n4\n", encoding="utf-8")
    (tmp_path / "test_idx.txt").write_text("[5, 6, 7]\n", encoding="utf-8")
    spec = {"sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": {"regex": r"^\d+$"}}, {"role": "targets", "select": ["y"]}]}], "partitions": {"by": "index_file", "train_file": "train_idx.txt", "test_file": "test_idx.txt"}}
    asm = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert asm.blocks["train"].n_samples == 5
    assert asm.blocks["test"].n_samples == 3


def test_coverage_matrix_complete():
    """Every load-supported E.1/E.2 vocabulary element must be exercised (Epic 2.4 gate)."""
    covered: set[str] = set()
    for _id, _files, spec_dict in _cases():
        covered |= vocab_elements(DatasetSpec.from_dict(spec_dict))
    # folds:file + partition:index_file are exercised by their dedicated tests
    covered.add("folds:file")
    covered.add("partition:index_file")
    missing = CATALOGUE - covered
    assert not missing, f"cookbook does not exercise these vocabulary elements: {sorted(missing)}"


def _validation_partition_spec(tmp_path, partitions, *, unknown=False):
    rows = ["train", "test", "val", "predict"] * 2
    if unknown:
        rows.append("other")
    pd.DataFrame({"x": [10.0 + i for i in range(len(rows))], "y": [100.0 + i for i in range(len(rows))], "set": rows, "row": [f"r{i}" for i in range(len(rows))]}).to_csv(tmp_path / "all.csv", sep=";", index=False)
    return {"task_type": "regression", "sources": [{"id": "d", "role": "mixed", "input": "all.csv", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["set", "row"]}]}], "partitions": partitions}


def _four_partitions(tmp_path, by):
    if by == "column":
        return {"by": by, "column": "set", "train_values": ["TRAIN"], "test_values": ["test"], "val_values": ["VaL"], "predict_values": ["predict"]}
    lists = {"train": [0, 4], "test": [1, 5], "val": [2, 6], "predict": [3, 7]}
    if by == "index":
        return {"by": by, **lists}
    for name, rows in lists.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    return {"by": by, **{name + "_file": name + ".json" for name in lists}}


@pytest.mark.parametrize("by", ["column", "index", "index_file"])
@pytest.mark.parametrize("target", ["assembled", "dataset_package"])
def test_validation_partitions_conserve_rows_and_targets(tmp_path, by, target):
    spec = _validation_partition_spec(tmp_path, _four_partitions(tmp_path, by))
    result = nio.load(spec, base_dir=tmp_path, target=target)
    assembled = result.to_assembled() if target == "dataset_package" else result
    assert list(assembled.blocks) == ["train", "test", "val", "predict"]
    all_rows = []
    for part, indices in {"train": [0, 4], "test": [1, 5], "val": [2, 6], "predict": [3, 7]}.items():
        block = assembled.blocks[part]
        assert block.n_samples == 2
        np.testing.assert_array_equal(block.X[0][:, 0], [10.0 + i for i in indices])
        np.testing.assert_array_equal(block.y[:, 0], [100.0 + i for i in indices])
        assert block.metadata["row"].tolist() == [f"r{i}" for i in indices]
        assert block.source_ids == ["d"]
        all_rows.extend(block.metadata["row"].tolist())
    assert sorted(all_rows) == [f"r{i}" for i in range(8)]


@pytest.mark.parametrize("by", ["index", "index_file"])
def test_validation_only_partition_is_valid(tmp_path, by):
    partitions = {"by": "index", "val": [2, 6]}
    if by == "index_file":
        (tmp_path / "val.json").write_text("[2, 6]", encoding="utf-8")
        partitions = {"by": by, "val_file": "val.json"}
    spec = _validation_partition_spec(tmp_path, partitions)
    DatasetSpec.from_dict(spec).validate()
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert list(assembled.blocks) == ["val"]
    np.testing.assert_array_equal(assembled.blocks["val"].y[:, 0], [102.0, 106.0])


@pytest.mark.parametrize("policy", ["train", "test", "drop", "error"])
def test_validation_values_are_known_before_unknown_policy(tmp_path, policy):
    partitions = {**_four_partitions(tmp_path, "column"), "unknown_policy": policy}
    spec = _validation_partition_spec(tmp_path, partitions)
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert [b.n_samples for b in assembled.blocks.values()] == [2, 2, 2, 2]
    spec = _validation_partition_spec(tmp_path, partitions, unknown=True)
    if policy == "error":
        with pytest.raises(ValueError, match="unknown values"):
            nio.load(spec, base_dir=tmp_path, target="assembled")
    else:
        assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
        assert assembled.blocks["val"].metadata["row"].tolist() == ["r2", "r6"]
        assert sum(b.n_samples for b in assembled.blocks.values()) == (8 if policy == "drop" else 9)
        if policy != "drop":
            assert assembled.blocks[policy].metadata["row"].tolist()[-1] == "r8"


@pytest.mark.parametrize("overlap", ["TRAIN", "TEST", "PREDICT"])
def test_validation_column_overlap_refuses_case_insensitively(tmp_path, overlap):
    spec = _validation_partition_spec(tmp_path, {**_four_partitions(tmp_path, "column"), "val_values": [overlap]})
    with pytest.raises(ValueError, match="appears in both.*val_values"):
        DatasetSpec.from_dict(spec).validate()
    with pytest.raises(ValueError, match="appears in both.*val_values"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("bad, diagnostic", [([2, 2], "duplicate"), ([8], "out-of-range"), ([-1], "out-of-range"), (2, "must be a list"), ([0], "disjoint"), (["bad"], "invalid literal")])
def test_validation_indices_reuse_owner_refusals(tmp_path, bad, diagnostic):
    spec = _validation_partition_spec(tmp_path, {"by": "index", "train": [0], "val": bad})
    with pytest.raises(ValueError, match=diagnostic):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("bad, diagnostic", [([2, 2], "duplicate"), ([8], "out-of-range"), ([0], "disjoint"), (["bad"], "invalid literal")])
def test_validation_file_indices_reuse_owner_refusals(tmp_path, bad, diagnostic):
    (tmp_path / "train.json").write_text("[0]", encoding="utf-8")
    (tmp_path / "val.json").write_text(json.dumps(bad), encoding="utf-8")
    spec = _validation_partition_spec(tmp_path, {"by": "index_file", "train_file": "train.json", "val_file": "val.json"})
    with pytest.raises(ValueError, match=diagnostic):
        nio.load(spec, base_dir=tmp_path, target="assembled")


def test_validation_file_missing_refuses(tmp_path):
    spec = _validation_partition_spec(tmp_path, {"by": "index_file", "val_file": "missing-val.txt"})
    with pytest.raises(ValueError, match="missing-val.txt"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


def test_validation_file_relative_to_yaml_base(tmp_path, monkeypatch):
    import yaml

    base = tmp_path / "source"
    base.mkdir()
    spec = _validation_partition_spec(base, _four_partitions(base, "index_file"))
    path = base / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "val.json").write_text("[0]", encoding="utf-8")
    monkeypatch.chdir(decoy)
    assembled = nio.load(path, target="assembled")
    assert list(assembled.blocks) == ["train", "test", "val", "predict"]
    np.testing.assert_array_equal(assembled.blocks["val"].y[:, 0], [102.0, 106.0])


def test_partitions_without_validation_keep_literal_emission():
    from nirs4all_io.spec.dataset_spec import PartitionsSpec

    for original in [{"by": "index", "train": [0]}, {"by": "index_file", "train_file": "train.json"}, {"by": "column", "column": "set", "train_values": ["train"]}]:
        assert json.dumps(PartitionsSpec.from_dict(original).to_dict(), separators=(",", ":")) == json.dumps(original, separators=(",", ":"))



@pytest.mark.parametrize("left,right", [("train_values","test_values"),("train_values","val_values"),("train_values","predict_values"),("test_values","val_values"),("test_values","predict_values"),("val_values","predict_values")])
def test_all_column_overlap_pairs_refuse_at_validation_and_direct_mask(tmp_path, left, right):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    partitions = {"by": "column", "column": "set", left: ["Absent"], right: ["ABSENT"]}
    spec = _validation_partition_spec(tmp_path, partitions)
    parsed = DatasetSpec.from_dict(spec)
    for call in [parsed.validate, lambda: _split_partition_mask(pd.DataFrame({"set": ["train"]}), parsed, tmp_path)]:
        with pytest.raises(ValueError, match=f"value 'absent' appears in both '{left}' and '{right}'"):
            call()


@pytest.mark.parametrize("left,right,refused", [(1,"1",True),(True,"true",True),("TRAIN","train",True),(1,1.0,False),(" 1","1",False)])
def test_partition_normalization_keeps_existing_matching(tmp_path, left, right, refused):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    spec = DatasetSpec.from_dict(_validation_partition_spec(tmp_path, {"by":"column","column":"set","train_values":[left,left],"test_values":[right]}))
    if refused:
        with pytest.raises(ValueError, match="appears in both"):
            spec.validate()
        with pytest.raises(ValueError, match="appears in both"):
            _split_partition_mask(pd.DataFrame({"set":["unseen"]}), spec, tmp_path)
    else:
        spec.validate()
        frame = pd.DataFrame({"set": pd.Series([left,right],dtype=object)})
        if left == 1 and right == 1.0:
            # C8 keeps lexical declarations distinct, but numeric rows now conflict.
            with pytest.raises(ValueError, match="set.*train_values.*test_values"):
                _split_partition_mask(frame, spec, tmp_path)
        else:
            masks = _split_partition_mask(frame, spec, tmp_path)
            assert masks["train"].tolist() == [True,False]
            assert masks["test"].tolist() == [False,True]


@pytest.mark.parametrize("key", ["validation","valid","cal","tset","unknown_polcy"])
@pytest.mark.parametrize("route", ["validate","to_spec","load"])
def test_partition_unknown_keys_refuse_before_loss(tmp_path, key, route):
    spec = _validation_partition_spec(tmp_path, {"by":"index","train":[0],key:[1]})
    with pytest.raises(ValueError, match="partitions: unknown keys"):
        if route == "validate":
            DatasetSpec.from_dict(spec).validate()
        elif route == "to_spec":
            nio.to_spec(spec)
        else:
            nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("partitions", [{"by":"column","column":"set","train":[0]}, {"by":"index","column":"set","train":[0]}, {"by":"index","train_values":["train"]}, {"by":"index_file","train":[0]}, {"by":"files","train_file":"train.txt"}, {"train":[0]}, {"unknown_policy":"train"}])
def test_partition_declarations_must_match_mode(tmp_path, partitions):
    spec = _validation_partition_spec(tmp_path, partitions)
    with pytest.raises(ValueError, match="not applicable"):
        DatasetSpec.from_dict(spec)
    with pytest.raises(ValueError, match="not applicable"):
        nio.to_spec(spec)
    with pytest.raises(ValueError, match="not applicable"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("field", ["train_values","test_values","val_values","predict_values"])
@pytest.mark.parametrize("bad", [None,"train",1,{}])
def test_partition_value_declarations_must_be_arrays(tmp_path, field, bad):
    spec = _validation_partition_spec(tmp_path, {"by":"column","column":"set",field:bad})
    with pytest.raises(ValueError, match=f"partitions.{field} must be an array"):
        DatasetSpec.from_dict(spec)


@pytest.mark.parametrize("bad", [[],"index",1,False])
def test_partition_object_must_be_a_mapping(tmp_path, bad):
    spec = _validation_partition_spec(tmp_path, bad)
    with pytest.raises(ValueError, match="partitions must be a mapping"):
        DatasetSpec.from_dict(spec)


def test_empty_partitions_and_input_adapter_empty_indices_stay_admitted(tmp_path):
    for partitions in [{}, {"by":"index","train":[]}, {"by":"index","train":[],"test":[]}]:
        spec = _validation_partition_spec(tmp_path, partitions)
        DatasetSpec.from_dict(spec).validate()
        nio.to_spec(spec)
    import jsonschema

    from nirs4all_io.spec.json_schema import validate_against_schema

    spec = _validation_partition_spec(tmp_path, {"by":"index","val":[2]})
    validate_against_schema(spec)
    spec["partitions"]["validation"] = [1]
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        validate_against_schema(spec)


def _fold_training_spec(tmp_path, folds):
    spec = _validation_partition_spec(tmp_path, {"by":"index","train":[1,3,5],"test":[0,2],"val":[4,6],"predict":[7]})
    spec["sample_index"] = {"by":"id","key":"row","observation_id":"row"}
    spec["folds"] = folds
    return spec


@pytest.mark.parametrize("target", ["assembled","dataset_package"])
def test_nonprefix_folds_emit_train_positions_with_frame_identity(tmp_path, target):
    spec = _fold_training_spec(tmp_path, {"inline":[{"train":[3,1],"val":[5]}]})
    data = nio.load(spec, base_dir=tmp_path, target=target)
    assembled = data.to_assembled() if target == "dataset_package" else data
    assert [(list(train),list(val)) for train,val in assembled.folds] == [([1,0],[2])]
    assert assembled.fold_provenance == [{"train_observation_ids":["r3","r1"],"validation_observation_ids":["r5"]}]
    np.testing.assert_array_equal(assembled.blocks["train"].X[0][:,0], [11.0,13.0,15.0])
    np.testing.assert_array_equal(assembled.blocks["train"].y[:,0], [101.0,103.0,105.0])


@pytest.mark.parametrize("bad", [7,0,2,99,-1,True,1.0,"1"])
def test_inline_folds_refuse_outside_training_namespace_and_nonintegers(tmp_path, bad):
    spec = _fold_training_spec(tmp_path, {"inline":[{"train":[1],"val":[bad]}]})
    with pytest.raises(ValueError, match="fold"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("fold", [{"train":[0,1],"val":[2]}, {"train":[1,1],"val":[5]}, {"train":[1],"val":[5,5]}, {"train":[1],"val":[1]}, {"train":1,"val":[5]}])
def test_fold_positions_are_not_input_indices_and_sides_are_strict(tmp_path, fold):
    spec = _fold_training_spec(tmp_path, {"inline":[fold]})
    with pytest.raises(ValueError, match="fold"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("name,text,expected", [
    ("folds.json", '[{"train":[3,1],"val":[5]}]', [([1,0],[2])]),
    ("folds.txt", '3,1\n5\n', [([1,0],[2])]),
    ("train.csv", 'fold_0,fold_1\n3,5\n1,\n', [([1,0],[2]),([2],[0,1])]),
    ("assignment.csv", 'sample_id,fold\n3,0\n1,0\n5,1\n', [([2],[1,0]),([0,1],[2])]),
])
def test_four_fold_file_forms_use_training_frame_namespace(tmp_path, name, text, expected):
    spec = _fold_training_spec(tmp_path, {"file":name})
    (tmp_path/name).write_text(text, encoding="utf-8")
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert [(list(train),list(val)) for train,val in assembled.folds] == expected
    for (train,val), provenance in zip(expected,assembled.fold_provenance,strict=True):
        ids = ["r1","r3","r5"]
        assert provenance == {"train_observation_ids":[ids[i] for i in train],"validation_observation_ids":[ids[i] for i in val]}


@pytest.mark.parametrize("name,text", [("folds.json",'[{"train":[1],"val":[7]}]'),("folds.txt",'1\n7\n'),("train.csv",'fold_0,fold_1\n1,7\n'),("assignment.csv",'sample_id,fold\n1,0\n7,1\n')])
def test_every_fold_file_form_refuses_nontraining_rows(tmp_path, name, text):
    spec = _fold_training_spec(tmp_path, {"file":name})
    (tmp_path/name).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="training rows"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


@pytest.mark.parametrize("name,text", [("folds.json",'[{"train":[true],"val":[5]}]'),("folds.json",'[{"train":[1.0],"val":[5]}]'),("folds.json",'[{"train":["1"],"val":[5]}]'),("folds.txt",'1.0\n5\n'),("train.csv",'fold_0\n1.0\n'),("assignment.csv",'sample_id,fold\n1.0,0\n5,1\n'),("train.csv",'fold_0\n1\n1\n')])
def test_fold_file_parsers_do_not_coerce_drop_or_deduplicate_bad_indices(tmp_path, name, text):
    spec = _fold_training_spec(tmp_path, {"file":name})
    (tmp_path/name).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        nio.load(spec, base_dir=tmp_path, target="assembled")


def test_fold_empty_sides_stay_valid_including_csv_complement(tmp_path):
    spec = _fold_training_spec(tmp_path, {"inline":[{"train":[],"val":[5]}]})
    assert nio.load(spec, base_dir=tmp_path, target="assembled").folds == [([],[2])]
    (tmp_path/'all-train.csv').write_text('fold_0\n1\n3\n5\n', encoding="utf-8")
    spec['folds'] = {"file":"all-train.csv"}
    assert nio.load(spec, base_dir=tmp_path, target="assembled").folds == [([0,1,2],[])]


def test_fold_column_only_uses_train_and_keeps_unlabelled_training_rows_fitting(tmp_path):
    spec = _fold_training_spec(tmp_path, {"column":"cv"})
    df = pd.read_csv(tmp_path/'all.csv', sep=';')
    df['cv'] = [None,'a',None,None,None,'b',None,None]
    df.to_csv(tmp_path/'all.csv',sep=';',index=False)
    spec['sources'][0]['columns'][-1]['select'].append('cv')
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert assembled.folds == [([1,2],[0]),([0,1],[2])]
    assert assembled.fold_provenance == [
        {"train_observation_ids":["r3","r5"],"validation_observation_ids":["r1"]},
        {"train_observation_ids":["r1","r3"],"validation_observation_ids":["r5"]},
    ]
    df.loc[2,'cv'] = 'leak'
    df.to_csv(tmp_path/'all.csv',sep=';',index=False)
    with pytest.raises(ValueError, match="only training rows may have fold labels"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


def test_per_source_folds_use_train_block_namespace_and_no_train_refuses(tmp_path):
    spec = _fold_training_spec(tmp_path, {"inline":[{"train":[0],"val":[2]}]})
    spec.pop('partitions')
    spec['sources'][0]['partition'] = 'train'
    pd.read_csv(tmp_path/'all.csv',sep=';').iloc[:3].to_csv(tmp_path/'all.csv',sep=';',index=False)
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert assembled.folds == [([0],[2])]
    assert assembled.fold_provenance == [{"train_observation_ids":["r0"],"validation_observation_ids":["r2"]}]
    spec['folds']['inline'][0]['val'] = [4]
    with pytest.raises(ValueError, match="out of range"):
        nio.load(spec, base_dir=tmp_path, target="assembled")
    spec['sources'][0]['partition'] = 'test'
    spec['folds'] = {"inline":[{"train":[],"val":[]}]}
    with pytest.raises(ValueError, match="require a train block"):
        nio.load(spec, base_dir=tmp_path, target="assembled")


def test_unpartitioned_fold_positions_stay_frame_positions(tmp_path):
    spec = _fold_training_spec(tmp_path, {"inline":[{"train":[3,1],"val":[5]}]})
    spec.pop('partitions')
    assembled = nio.load(spec, base_dir=tmp_path, target="assembled")
    assert assembled.folds == [([3,1],[5])]
    assert assembled.fold_provenance == [{"train_observation_ids":["r3","r1"],"validation_observation_ids":["r5"]}]



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
        assert block.feature_axes[1][0]["coordinates"] is None


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
    assert type(error.value).__name__ in ("JoinError", "SpecError")


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



def test_native_merge_decimal_and_dtype_use_collected_native_columns():
    from nirs4all_io.materialize.assemble import SourceTable, _assemble_block
    from nirs4all_io.materialize.numeric import NumericStorageOptions

    def table(name, frame, roles, **kwargs):
        frame = frame.copy()
        frame[f"__src_row_idx__{name}__"] = [0, 1]
        return SourceTable(name, frame, roles, None, None, "table", None, None, None, "text", **kwargs)
    anchor = table("anchor", pd.DataFrame({"0": [100, 200]}), {"0": "features"})
    mixed = table("train_x", pd.DataFrame({"0": ["10,5", "20,5"], "train_x__0": ["11.5", "21.5"]}), {"train_x__0": "features", "0": "features"}, numeric_scope={"0": ",", "train_x__0": "."}, source_dtypes={"0": "float32", "train_x__0": "float64"})
    spec = DatasetSpec.from_dict({"sources": [{"id": "anchor", "input": "a.csv", "role": "features"}, {"id": "train_x", "input": "b.csv", "role": "features"}]})
    block, _ = _assemble_block({"anchor": anchor, "train_x": mixed}, spec, [], [], NumericStorageOptions())
    assert block.feature_headers[1] == ["0", "train_x__0"]
    np.testing.assert_array_equal(block.X[1], [[10.5, 11.5], [20.5, 21.5]])
    assert block.x_source_dtypes[1] == "mixed"
    assert block.feature_axes[1][0]["coordinates"] is None


def test_native_merge_lookup_filename_stem_and_na_dropped_columns_keep_ownership(tmp_path):
    first = tmp_path / "sample1.csv"
    second = tmp_path / "sample2.csv"
    first.write_text("x;gone\n1;NA\n")
    second.write_text("x;gone\n2;NA\n")
    lookup = _native_target_source(tmp_path, "lookup", "stem;x;label\nsample2;80;second\nsample1;70;first\n", "mixed", kind="lookup", key="stem", columns=[{"role": "targets", "select": ["x"]}, {"role": "metadata", "select": ["label"]}], join={"to": "spectra", "left_on": "filename_stem", "right_on": "stem", "how": "m:1"})
    sources = [lookup, {"id": "spectra", "input": [str(first), str(second)], "role": "features", "merge": "concat_samples", "params": {"delimiter": ";", "has_header": True, "na": {"policy": "remove_feature"}}}]
    block = nio.load({"task_type": "regression", "sources": sources}, target="dataset_package").to_assembled().blocks["train"]
    assert block.feature_headers == [["x"]]
    np.testing.assert_array_equal(block.X[0], [[1], [2]])
    assert block.y_headers == ["x"]
    np.testing.assert_array_equal(block.y[:, 0], [70, 80])
    assert block.metadata["label"].tolist() == ["first", "second"]


# D2/D1/D3 partition keys: decoded missingness is separate from literal labels.
def _partition_contract_spec(**partitions):
    return DatasetSpec.from_dict({"sources": [{"id": "d", "input": "unused.csv", "role": "features"}], "partitions": {"by": "column", "column": "set", **partitions}})


@pytest.mark.parametrize("part", ["train", "test", "val", "predict"])
@pytest.mark.parametrize("policy", [None, "train", "test", "drop", "error"])
def test_partition_contract_missing_is_unknown_for_every_list_and_policy(tmp_path, part, policy):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    options = {part + "_values": ["nan"]}
    if policy is not None:
        options["unknown_policy"] = policy
    spec = _partition_contract_spec(**options)
    frame = pd.DataFrame({"set": pd.Series(["NAN", None, float("nan"), "other"], dtype=object)})
    if policy == "error":
        with pytest.raises(SpecError, match="set.*missing rows: 2"):
            _split_partition_mask(frame, spec, tmp_path)
    else:
        actual = {key: np.flatnonzero(mask).tolist() for key, mask in _split_partition_mask(frame, spec, tmp_path).items()}
        expected = {part: [0]}
        destination = "train" if policy is None else policy
        if destination != "drop":
            expected[destination] = sorted(expected.get(destination, []) + [1, 2, 3])
        assert actual == {key: expected[key] for key in ("train", "test", "val", "predict") if key in expected}


@pytest.mark.parametrize("part", ["train", "test", "val", "predict"])
@pytest.mark.parametrize("bad", [None, [], {}, float("nan"), float("inf"), -float("inf"), ("nested",)])
@pytest.mark.parametrize("boundary", ["validate", "mask"])
def test_partition_contract_non_scalar_declarations_refuse_with_list_name(tmp_path, part, bad, boundary):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(**{part + "_values": [bad]})
    with pytest.raises(SpecError, match=part + "_values"):
        if boundary == "validate":
            spec.partitions.validate_column_values()
        else:
            _split_partition_mask(pd.DataFrame({"set": ["anything"]}), spec, tmp_path)


@pytest.mark.parametrize("values", [[None, float("nan")], [None, float("nan"), "ZZ", "aa"]])
def test_partition_contract_error_sorts_only_observed_keys_and_counts_missing(tmp_path, values):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(test_values=["nan", "none"], unknown_policy="error")
    with pytest.raises(SpecError) as error:
        _split_partition_mask(pd.DataFrame({"set": pd.Series(values, dtype=object)}), spec, tmp_path)
    message = str(error.value)
    assert "'set'" in message and "missing rows: 2" in message
    assert ("['aa', 'zz']" if len(values) == 4 else "[]") in message


@pytest.mark.parametrize("kind,label", [("float32", "0.1"), ("datetime", "2020-01-02"), ("nullable", "1")])
def test_partition_contract_keeps_original_series_astype_string_keys(tmp_path, kind, label):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    series = {"float32": pd.Series([0.1, None], dtype="float32"), "datetime": pd.Series(pd.to_datetime(["2020-01-02", None])), "nullable": pd.Series([1, None], dtype="Int64")}[kind]
    assert series.astype(str).iloc[0] == label
    spec = _partition_contract_spec(test_values=[label], unknown_policy="drop")
    masks = _split_partition_mask(pd.DataFrame({"set": series}), spec, tmp_path)
    assert {key: np.flatnonzero(mask).tolist() for key, mask in masks.items()} == {"test": [0]}


@pytest.mark.parametrize("observed,declared,expected", [("ΣΑΣ", "σας", [0]), ("İ", "i\u0307", [0]), ("ß", "SS", []), (" SS ", "ss", [])])
def test_partition_contract_scalar_unicode_lower_and_default_train(tmp_path, observed, declared, expected):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    spec = _partition_contract_spec(test_values=[declared])
    frame = pd.DataFrame({"set": pd.Series([observed, "unassigned"], dtype="str")})
    masks = _split_partition_mask(frame, spec, tmp_path)
    assert np.flatnonzero(masks.get("test", np.zeros(2, dtype=bool))).tolist() == expected
    assert np.flatnonzero(masks["train"]).tolist() == ([1] if expected else [0, 1])


@pytest.mark.parametrize("first,second,collision", [("ΣΑΣ", "σας", True), ("İ", "i\u0307", True), ("ß", "SS", False), (1, "1", True), (1, 1.0, False), (True, "true", True)])
def test_partition_contract_existing_scalar_overlap_rules_stay(tmp_path, first, second, collision):
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(train_values=[first, first], test_values=[second])
    if collision:
        with pytest.raises(SpecError, match="both 'train_values' and 'test_values'"):
            spec.partitions.validate_column_values()
    else:
        spec.partitions.validate_column_values()


def _partition_contract_public_spec(path, **partitions):
    return {"task_type": "regression", "sources": [{"id": "d", "input": str(path), "role": "mixed", "params": {"delimiter": ";", "has_header": True}, "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}, {"role": "metadata", "select": ["set", "row"]}]}], "partitions": {"by": "column", "column": "set", **partitions}}


@pytest.mark.parametrize("policy", [None, "train", "test", "drop", "error"])
def test_partition_contract_public_csv_literal_nan_and_decoded_missing(tmp_path, policy):
    from nirs4all_io.spec.enums import SpecError

    path = tmp_path / "partition-missing.csv"
    path.write_text('x;y;set;row\n1;101;cal;r0\n2;102;NAN;r1\n3;103;;r2\n4;104;NA;r3\n5;105;none;r4\n6;106;val;r5\n7;107;"nan";r6\n', encoding="utf-8")
    options = {"train_values": ["cal"], "test_values": ["nan"]}
    if policy is not None:
        options["unknown_policy"] = policy
    spec = _partition_contract_public_spec(path, **options)
    if policy == "error":
        with pytest.raises(SpecError, match="set.*missing rows: 3"):
            nio.load(spec, target="assembled")
        return
    expected = {"train": ["r0"], "test": ["r1"]}
    if policy is None or policy == "train":
        expected["train"] += ["r2", "r3", "r4", "r5", "r6"]
    elif policy == "test":
        expected["test"] += ["r2", "r3", "r4", "r5", "r6"]
    for target in ("assembled", "dataset_package"):
        result = nio.load(spec, target=target)
        assembled = result.to_assembled() if target == "dataset_package" else result
        assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == expected
        for block in assembled.blocks.values():
            rows = [int(row[1:]) for row in block.metadata["row"]]
            np.testing.assert_array_equal(block.y[:, 0], [101 + row for row in rows])


@pytest.mark.parametrize("test_values,expected_test", [(["nan"], ["r1", "r5"]), (["none", ""], ["r3", "r4"])])
def test_partition_contract_public_parquet_literal_text_and_null(tmp_path, test_values, expected_test):
    pytest.importorskip("pyarrow")
    path = tmp_path / "partition-literals.parquet"
    pd.DataFrame({"x": np.arange(9, dtype=float), "y": np.arange(101, 110, dtype=float), "set": ["cal", "nan", None, "None", "", "NaN", "NA", "val", None], "row": [f"r{i}" for i in range(9)]}).to_parquet(path, index=False)
    spec = _partition_contract_public_spec(path, train_values=["cal"], test_values=test_values, unknown_policy="drop")
    for target in ("assembled", "dataset_package"):
        result = nio.load(spec, target=target)
        assembled = result.to_assembled() if target == "dataset_package" else result
        assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"train": ["r0"], "test": expected_test}


@pytest.mark.parametrize("dtype", ["float64", "Int64"])
def test_partition_contract_numeric_missing_does_not_match_none(tmp_path, dtype):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    frame = pd.DataFrame({"set": pd.Series([1, None, 2], dtype=dtype)})
    spec = _partition_contract_spec(test_values=["none"], unknown_policy="drop")
    assert _split_partition_mask(frame, spec, tmp_path) == {}


@pytest.mark.parametrize("bad", [None, [], {}])
def test_partition_contract_public_invalid_label_declarations_refuse(tmp_path, bad):
    from nirs4all_io.spec.enums import SpecError

    path = tmp_path / "declared.csv"
    path.write_text("x;y;set;row\n1;101;cal;r0\n")
    with pytest.raises(SpecError, match="test_values"):
        nio.load(_partition_contract_public_spec(path, test_values=[bad]), target="assembled")


@pytest.mark.parametrize("observed,declared", [("ΣΑΣ", "σας"), ("İ", "i\u0307")])
def test_partition_contract_public_unicode_test_label_under_default_train(tmp_path, observed, declared):
    path = tmp_path / "unicode.csv"
    path.write_text(f"x;y;set;row\n1;101;{observed};r0\n2;102;unassigned;r1\n", encoding="utf-8")
    spec = _partition_contract_public_spec(path, test_values=[declared])
    assembled = nio.load(spec, target="assembled")
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"train": ["r1"], "test": ["r0"]}
    np.testing.assert_array_equal(assembled.blocks["test"].y[:, 0], [101])


def test_partition_contract_public_adapter_retains_exact_selected_rows(tmp_path):
    path = tmp_path / "adapter.csv"
    path.write_text("x;y;set;row\n1;101;cal;r0\n2;102;NAN;r1\n3;103;;r2\n")
    spec = _partition_contract_public_spec(path, train_values=["cal"], test_values=["nan"], unknown_policy="drop")
    record = nio.load(spec, target="spectrodataset", spectro_dataset_cls=_NativeTargetRecorder)
    assert [part for part, _, _ in record.samples] == ["train", "test"]
    np.testing.assert_array_equal(record.targets[:, 0], [101, 102])
    assert record.metadata["row"].tolist() == ["r0", "r1"]



def _c8_partition_rows(tmp_path, series, values, **partitions):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    spec = _partition_contract_spec(test_values=values, unknown_policy="drop", **partitions)
    masks = _split_partition_mask(pd.DataFrame({"set": series}), spec, tmp_path)
    return {part: [f"r{i}" for i in np.flatnonzero(mask)] for part, mask in masks.items()}


@pytest.mark.parametrize("value", [1, "1", 1.0, "1.0", "1e0"])
@pytest.mark.parametrize("dtype", ["float64", "Int64", "object"])
def test_c8_integer_missing_numeric_identity_keeps_exact_rows(tmp_path, value, dtype):
    series = pd.Series([1, 2, None, 1, 2], dtype=dtype)
    assert _c8_partition_rows(tmp_path, series, [value]) == {"test": ["r0", "r3"]}


@pytest.mark.parametrize("value", ["01", "+1", ".5", " 1", "1 ", "1,5", True, "1e999", "1e-999"])
def test_c8_numeric_rows_do_not_coerce_invalid_or_refused_tokens(tmp_path, value):
    assert _c8_partition_rows(tmp_path, pd.Series([1, 2, None, 1, 2]), [value]) == {}


def test_c8_default_test_numeric_label_conserves_all_five_rows(tmp_path):
    from nirs4all_io.materialize.assemble import _split_partition_mask

    frame = pd.DataFrame({"set": [1, 2, None, 1, 2]})
    masks = _split_partition_mask(frame, _partition_contract_spec(test_values=[2]), tmp_path)
    assert {part: [f"r{i}" for i in np.flatnonzero(mask)] for part, mask in masks.items()} == {"train": ["r0", "r2", "r3"], "test": ["r1", "r4"]}


@pytest.mark.parametrize("series,values,expected", [
    (pd.Series([1, 2, 3, 1], dtype="int64"), [1.0], ["r0", "r3"]),
    (pd.Series([1.0, 1.5, 2.0, 1.0]), [1], ["r0", "r3"]),
    (pd.Series([1.0, 1.5, 2.0, 1.0]), ["1.50"], ["r1"]),
    (pd.Series([2**53+1, 2**53, 5], dtype="int64"), [9007199254740992.0], ["r1"]),
    (pd.Series([2**53+1, 2**53, 5], dtype="int64"), ["9007199254740993"], ["r0"]),
    (pd.Series([2**64-1, 2**53+1, None, 5], dtype="UInt64"), [2**64-1], ["r0"]),
    (pd.Series([2**53+1, None, 7, 7], dtype="Int64"), [7.0], ["r2", "r3"]),
    (pd.Series([True, False, None], dtype=object), [1], []),
    (pd.Series([True, False, None], dtype=object), ["true"], ["r0"]),
])
def test_c8_exact_integer_float_bool_and_nullable_rows(tmp_path, series, values, expected):
    result = _c8_partition_rows(tmp_path, series, values)
    assert result == ({"test": expected} if expected else {})


@pytest.mark.parametrize("value,expected", [(0, ["r0", "r2"]), (0.0, ["r0", "r2"]), ("0", ["r0", "r2"]), (-0.0, ["r1", "r3"]), ("-0", ["r1", "r3"]), ("-0.0", ["r1", "r3"])])
def test_c8_signed_zero_keeps_its_own_numeric_identity(tmp_path, value, expected):
    assert _c8_partition_rows(tmp_path, pd.Series([0.0, -0.0, 0.0, -0.0, 1.5]), [value]) == {"test": expected}


@pytest.mark.parametrize("value", [0.1, "0.1", 0.10000000149011612, 0.10000000000001])
def test_c8_float32_declarations_round_once_to_owner_dtype(tmp_path, value):
    assert _c8_partition_rows(tmp_path, pd.Series([0.1, 0.2, None, 0.1], dtype="float32"), [value]) == {"test": ["r0", "r3"]}


@pytest.mark.parametrize("value", [1e100, 1e-100])
def test_c8_refused_float32_storage_conversion_means_no_match(tmp_path, value):
    assert _c8_partition_rows(tmp_path, pd.Series([0.0, 0.1], dtype="float32"), [value]) == {}


@pytest.mark.parametrize("value", [9223372586610589697, "9223372586610589697"])
def test_c8_float32_integer_declaration_avoids_binary64_double_rounding(tmp_path, value):
    series = pd.Series([float(2**63), float(2**63+2**40)], dtype="float32")
    assert _c8_partition_rows(tmp_path, series, [value]) == {"test": ["r1"]}


@pytest.mark.parametrize("value,expected", [(1, ["r0"]), (1.0, ["r1"]), ("1.0", ["r1"]), ("001", ["r2"]), ("1e999", ["r4"]), (" 1", ["r5"]), ("+1", ["r6"]), ("1,5", ["r7"])])
def test_c8_text_tokens_remain_lexical(tmp_path, value, expected):
    series = pd.Series(["1", "1.0", "001", "a", "1e999", " 1", "+1", "1,5"], dtype=object)
    assert _c8_partition_rows(tmp_path, series, [value]) == {"test": expected}


@pytest.mark.parametrize("part", ["test", "val", "predict"])
def test_c8_runtime_numeric_alias_overlap_refuses_only_when_present(tmp_path, part):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(train_values=[1], **{part+"_values": ["1.0"]}, unknown_policy="drop")
    spec.partitions.validate_column_values()  # Lexical declarations remain disjoint.
    with pytest.raises(SpecError, match=f"set.*train_values.*{part}_values"):
        _split_partition_mask(pd.DataFrame({"set": [1, 2, None]}), spec, tmp_path)
    masks = _split_partition_mask(pd.DataFrame({"set": ["1", "1.0", "001", "a"]}), spec, tmp_path)
    assert {key: np.flatnonzero(mask).tolist() for key, mask in masks.items()} == {"train": [0], part: [1]}
    assert _split_partition_mask(pd.DataFrame({"set": [2, 3, None]}), spec, tmp_path) == {}


def test_c8_object_per_cell_numeric_text_and_timestamp_conversion(tmp_path):
    series = pd.Series([1, "1", None, 1.0, "1.0"], dtype=object)
    assert _c8_partition_rows(tmp_path, series, ["1e0"]) == {"test": ["r0", "r3"]}
    stamps = pd.Series(pd.to_datetime(["2020-01-02", None, "2021-03-04 05:06", "2020-01-02"], format="mixed"))
    assert _c8_partition_rows(tmp_path, stamps, ["2020-01-02 00:00:00"]) == {"test": ["r0", "r3"]}


@pytest.mark.parametrize("values,expected", [(["none"], []), (["inf"], ["r1"]), (["-inf"], ["r2"]), ([1.5], ["r0"])])
def test_c8_public_parquet_infinity_ignore_column_is_local_lexical(tmp_path, values, expected):
    pytest.importorskip("pyarrow")
    path = tmp_path / "infinity.parquet"
    pd.DataFrame({"x": [1., 2., 3., 4.], "y": [101., 102., 103., 104.], "set": [1.5, float("inf"), -float("inf"), None], "row": ["r0", "r1", "r2", "r3"]}).to_parquet(path, index=False)
    spec = _partition_contract_public_spec(path, test_values=values, unknown_policy="drop")
    spec["sources"][0]["columns"] = [{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["row"]},{"role":"ignore","select":["set"]}]
    spec["sources"][0]["params"]["na"] = {"policy": "ignore"}
    assembled = nio.load(spec, target="assembled")
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == ({"test": expected} if expected else {})


@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("role", ["ignore", "metadata"])
@pytest.mark.parametrize("value", [0.1, "0.1", 0.10000000149011612, 0.10000000000001])
def test_c8_float32_secondary_source_and_keyed_join_retain_owner_dtype(tmp_path, keyed, role, value):
    pytest.importorskip("pyarrow")
    left = tmp_path / "left.csv"
    right = tmp_path / "labels.parquet"
    left.write_text("x;y;sid;row\n1;101;s0;r0\n2;102;s1;r1\n3;103;s2;r2\n4;104;s3;r3\n")
    labels = pd.DataFrame({"set": pd.Series([0.1, 0.2, None, 0.1], dtype="float32")})
    if keyed:
        labels["sid"] = ["s0", "s1", "s2", "s3"]
        labels = labels.iloc[::-1]
    labels.to_parquet(right, index=False)
    spec = {"task_type":"regression","sources":[{"id":"left","input":str(left),"role":"mixed","params":{"delimiter":";","has_header":True},"columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["sid","row"]}]},{"id":"labels","input":str(right),"role":"mixed","params":{"na":{"policy":"ignore"}},"columns":[{"role":"ignore","select":["set"]}]}],"partitions":{"by":"column","column":"set","test_values":[value],"unknown_policy":"drop"}}
    spec["sources"][1]["columns"][0]["role"] = role
    if keyed:
        spec["sources"][0]["key"] = "sid"
        spec["sources"][1]["key"] = "sid"
        spec["sources"][1]["join"] = {"to":"left","on":"sid","how":"1:1"}
    for target in ("assembled", "dataset_package"):
        result = nio.load(spec, target=target)
        assembled = result.to_assembled() if target == "dataset_package" else result
        assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"test": ["r0", "r3"]}
        np.testing.assert_array_equal(assembled.blocks["test"].y[:, 0], [101., 104.])


def test_c8_public_csv_default_policy_corrects_measured_all_train_result(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text("x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;;r2\n4;104;1;r3\n5;105;2;r4\n")
    assembled = nio.load(_partition_contract_public_spec(path, test_values=[2]), target="assembled")
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"train": ["r0", "r2", "r3"], "test": ["r1", "r4"]}


@pytest.mark.parametrize("values,expected", [([1], ["r0", "r3"]), (["1"], ["r0", "r3"]), ([1.0], ["r0", "r3"]), (["1.0"], ["r0", "r3"]), (["1e0"], ["r0", "r3"])])
def test_c8_public_csv_integer_missing_aliases(tmp_path, values, expected):
    path = tmp_path / "aliases.csv"
    path.write_text("x;y;set;row\n1;101;1;r0\n2;102;2;r1\n3;103;;r2\n4;104;1;r3\n5;105;2;r4\n")
    assembled = nio.load(_partition_contract_public_spec(path, test_values=values, unknown_policy="drop"), target="assembled")
    assert {part: block.metadata["row"].tolist() for part, block in assembled.blocks.items()} == {"test": expected}


@pytest.mark.parametrize("dtype,cells,values,expected", [
    ("Int64", [2**53+1, None, 7, 7], [7.0], ["r2", "r3"]),
    ("UInt64", [2**64-1, 2**53+1, None, 5], [2**64-1], ["r0"]),
    ("int64", [2**53+1, 2**53, 5, 6], [9007199254740992.0], ["r1"]),
    ("boolean", [True, False, None, True], [1], []),
])
def test_c8_public_parquet_numeric_identity_preserves_typed_cells(tmp_path, dtype, cells, values, expected):
    pytest.importorskip("pyarrow")
    path = tmp_path / "exact-cells.parquet"
    pd.DataFrame({"x":[1.,2.,3.,4.],"y":[101.,102.,103.,104.],"row":["r0","r1","r2","r3"],"set":pd.Series(cells,dtype=dtype)}).to_parquet(path,index=False)
    spec = _partition_contract_public_spec(path, test_values=values, unknown_policy="drop")
    if dtype == "UInt64":
        # This reader has an existing signed-i64 admission boundary, before C8.
        # Direct UInt64 owner identity is tested separately, without a decoder change.
        from nirs4all_io.materialize.loaders import LoaderError

        with pytest.raises(LoaderError, match="outside i64 range"):
            nio.load(spec, target="assembled")
        return
    assembled = nio.load(spec, target="assembled")
    assert {part:block.metadata["row"].tolist() for part,block in assembled.blocks.items()} == ({"test":expected} if expected else {})


@pytest.mark.parametrize("left,right", [("train_values","test_values"),("train_values","val_values"),("train_values","predict_values"),("test_values","val_values"),("test_values","predict_values"),("val_values","predict_values")])
def test_c8_runtime_numeric_overlap_checks_all_six_pairs(tmp_path, left, right):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(**{left:[1],right:["1.0"]},unknown_policy="drop")
    with pytest.raises(SpecError,match=f"set.*{left}.*{right}"):
        _split_partition_mask(pd.DataFrame({"set":[1.,2.,None]}),spec,tmp_path)


def test_c8_float32_alias_overlap_is_runtime_and_int32_boundary_is_exact(tmp_path):
    from nirs4all_io.materialize.assemble import _split_partition_mask
    from nirs4all_io.spec.enums import SpecError

    spec = _partition_contract_spec(train_values=[0.1],test_values=[0.10000000000001],unknown_policy="drop")
    spec.partitions.validate_column_values()
    with pytest.raises(SpecError,match="set.*train_values.*test_values"):
        _split_partition_mask(pd.DataFrame({"set":pd.Series([0.1],dtype="float32")}),spec,tmp_path)
    assert _split_partition_mask(pd.DataFrame({"set":pd.Series([0.2],dtype="float32")}),spec,tmp_path) == {}
    assert _c8_partition_rows(tmp_path,pd.Series([16777216.,16777218.],dtype="float32"),[16777217]) == {"test":["r0"]}
