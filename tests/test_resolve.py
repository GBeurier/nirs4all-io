# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Tests for the resolver: identity, hashing, ordering, sidecars, in-memory."""

from __future__ import annotations

import numpy as np

from nirs4all_io.resolve import resolve


def _write(path, text="a,b\n1,2\n"):
    path.write_text(text, encoding="utf-8")
    return path


def test_resolve_directory_ordered_with_hashes(tmp_path):
    _write(tmp_path / "Xcal.csv")
    _write(tmp_path / "Ycal.csv", "y\n3\n")
    iset = resolve(tmp_path)
    assert iset.origin["kind"] == "directory"
    assert iset.names == ["Xcal.csv", "Ycal.csv"]  # deterministic (sorted)
    for item in iset.items:
        assert item.kind == "file"
        assert item.identity.endswith(item.ref) or item.ref in item.identity
        assert item.content_hash and len(item.content_hash) == 64
        assert item.extension == ".csv"


def test_resolve_glob_and_file_list(tmp_path):
    _write(tmp_path / "a.csv")
    _write(tmp_path / "b.csv")
    _write(tmp_path / "note.txt", "hello")
    glob_set = resolve(str(tmp_path / "*.csv"))
    assert sorted(i.hints["basename"] for i in glob_set.items) == ["a.csv", "b.csv"]
    list_set = resolve([str(tmp_path / "a.csv"), str(tmp_path / "note.txt")])
    assert {i.hints["basename"] for i in list_set.items} == {"a.csv", "note.txt"}


def test_content_hash_distinguishes_content(tmp_path):
    _write(tmp_path / "x.csv", "a\n1\n")
    h1 = resolve(tmp_path).items[0].content_hash
    _write(tmp_path / "x.csv", "a\n2\n")
    h2 = resolve(tmp_path).items[0].content_hash
    assert h1 != h2


def test_sidecar_grouping_envi(tmp_path):
    (tmp_path / "cube.img").write_bytes(b"\x00\x01\x02")
    (tmp_path / "cube.hdr").write_text("ENVI\nsamples = 2\n", encoding="utf-8")
    iset = resolve(tmp_path)
    # the .hdr is attached to the .img primary, not a standalone item
    assert iset.names == ["cube.img"]
    assert iset.items[0].sidecars and iset.items[0].sidecars[0].endswith("cube.hdr")


def test_recursive_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(tmp_path / "top.csv")
    _write(sub / "deep.csv")
    flat = resolve(tmp_path, recursive=False)
    deep = resolve(tmp_path, recursive=True)
    assert flat.names == ["top.csv"]
    assert set(deep.names) == {"top.csv", "sub/deep.csv"}


def test_resolve_in_memory_arrays():
    X = np.zeros((4, 3))
    y = np.arange(4)
    seq = resolve([X, y])
    assert [i.kind for i in seq.items] == ["array", "array"]
    assert seq.items[0].hints["shape"] == [4, 3]
    mapping = resolve({"X": X, "y": y})
    assert {i.ref for i in mapping.items} == {"X", "y"}
    assert all(i.content_hash for i in mapping.items)


def test_resolve_single_ndarray():
    iset = resolve(np.ones((2, 2)))
    assert len(iset.items) == 1 and iset.items[0].kind == "array"


def test_path_reference_fixture_mirrors_are_exact():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "tests/fixtures/path_reference_vectors.json").read_bytes() == (root / "crates/nirs4all-io/tests/fixtures/path_reference_vectors.json").read_bytes()


def test_path_reference_literal_file_and_logical_alias(tmp_path):
    import pytest

    from nirs4all_io.resolve.resolver import resolve
    actual = tmp_path / "raw"
    actual.write_text("x\n1\n")
    alias = tmp_path / "Xcal.csv"
    try:
        alias.symlink_to(actual)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    item = next(x for x in resolve(tmp_path).items if x.hints["basename"] == "Xcal.csv")
    assert item.ref == "Xcal.csv"
    assert item.identity == str(actual.resolve())
    literal = tmp_path / "scan[2].csv"
    literal.write_text("x\n0.1\n")
    (tmp_path / "scan2.csv").write_text("x\n7.1\n")
    assert resolve(str(literal)).items[0].identity == str(literal)
    assert resolve([str(literal)]).items[0].identity == str(literal)


def test_path_reference_alias_ties_use_logical_order(tmp_path):
    import pytest

    from nirs4all_io.resolve.resolver import resolve
    target = tmp_path / "store"
    target.write_text("x\n0.1\n")
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    for name in ["m.csv", "c.csv", "z.csv", "b.csv", "q.csv", "a.csv"]:
        try:
            (aliases / name).symlink_to(target)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
    assert resolve(aliases).names == ["a.csv", "b.csv", "c.csv", "m.csv", "q.csv", "z.csv"]
    listed = resolve([str(aliases / "z.csv"), str(aliases / "a.csv")])
    assert listed.names == ["z.csv", "a.csv"]


def test_path_reference_literal_list_keeps_orphan_and_input_order(tmp_path):
    paths = [tmp_path / "literal[2].csv", tmp_path / "orphan[3].csv"]
    for path in paths:
        _write(path)
    _write(tmp_path / "literal2.csv", "x\n7.2\n")
    items = resolve([str(path) for path in paths]).items
    assert [item.ref for item in items] == ["literal[2].csv", "orphan[3].csv"]
    assert [item.identity for item in items] == [str(path) for path in paths]
