# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Original CSV text follows ``by_key`` / ``concat_features`` merges (class F and large integers across joins).

A positional marker rides each CSV frame through the EXISTING join machinery, so the text of every result cell is
re-attached by the join's own row semantics (reordered keys, renamed duplicate headers, typed files without text).
These tests run the actual public facade on real files and check payload bits, masks and row order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.package import FeatureMatrix, MetadataTable, TargetTable
from nirs4all_io.strict_json import NumberDomainError

N = 2**54 + 2**30 + 1
MERGES = ["by_key", "concat_features"]


def _package(tmp_path: Path, a: str, b: str, merge: str, *, columns: dict[str, Any] | None = None, **options: Any):
    (tmp_path / "a.csv").write_text(a, encoding="utf-8")
    (tmp_path / "b.csv").write_text(b, encoding="utf-8")
    source = {"id": "s", "input": ["a.csv", "b.csv"], "merge": merge, "key": "id", "columns": columns or {"features": "x", "targets": "y"}}
    return nio.to_dataset_package({"sample_index": {"by": "id", "key": "id"}, "sources": [source]}, base_dir=tmp_path, **options)


def _x(package) -> list[float]:
    payload = dict(package.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    return payload.matrix.values.ravel().tolist()


def _y(package) -> list[float]:
    payload = dict(package.partitions["train"].payloads)["train/y"]
    assert isinstance(payload, TargetTable)
    return payload.columns[0].block.values.ravel().tolist()


@pytest.mark.parametrize("merge", MERGES)
@pytest.mark.parametrize("token", ["1e-400", "2e-324", "-1e-400", "-2e-324"])
def test_underflow_refuses_in_either_source_and_any_row(tmp_path: Path, merge: str, token: str) -> None:
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, f"id;x\ns0;{token}\n", "id;y\ns0;1\n", merge)  # the exact counterexample: feature in file a
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "id;x\ns0;1\n", f"id;y\ns0;{token}\n", merge)  # target in the LATER source
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, f"id;x\ns0;1\ns1;{token}\n", "id;y\ns1;20\ns0;10\n", merge)  # later row of file a, file b reversed
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "id;x\ns0;1\ns1;2\n", f"id;y\ns1;{token}\ns0;10\n", merge)  # bad cell is the FIRST row of the reversed later file


@pytest.mark.parametrize("merge", MERGES)
def test_row_order_is_followed_not_assumed(tmp_path: Path, merge: str) -> None:
    package = _package(tmp_path, "id;x\ns0;1\ns1;2\ns2;3\n", "id;y\ns2;30\ns0;10\ns1;20\n", merge)
    assert _x(package) == [1.0, 2.0, 3.0] and _y(package) == [10.0, 20.0, 30.0], "values are matched by key, in the left file's row order"
    zero = _package(tmp_path, "id;x\ns0;0.000e-999\ns1;-0.000e-999\n", "id;y\ns1;2\ns0;1\n", merge)
    payload = dict(zero.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix)
    assert payload.matrix.values_bytes().hex() == "0000000000000000" + "0000000000000080" and payload.matrix.validity.ravel().tolist() == [True, True]
    assert _y(zero) == [1.0, 2.0]
    tiny = _package(tmp_path, "id;x\ns0;4.9e-324\ns1;1e-300\n", "id;y\ns1;2\ns0;1\n", merge)
    assert _x(tiny) == [5e-324, 1e-300], "genuine subnormal and ordinary values are legitimate observations"


@pytest.mark.parametrize("merge", MERGES)
def test_a_wrong_cell_is_never_validated_a_lookalike_in_another_role_is_untouched(tmp_path: Path, merge: str) -> None:
    columns = {"features": "x", "targets": "y", "metadata": ["note"]}
    package = _package(tmp_path, "id;x\ns0;1\ns1;2\n", "id;y;note\ns1;20;1e-400\ns0;10;north\n", merge, columns=columns)
    metadata = dict(package.partitions["train"].payloads)["train/metadata"]
    assert isinstance(metadata, MetadataTable)
    assert metadata.frame["note"].tolist() == ["north", "1e-400"], "ordinary metadata strings stay literal, aligned to their own rows"
    assert _x(package) == [1.0, 2.0] and _y(package) == [10.0, 20.0]
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "id;x\ns0;1\ns1;2\n", "id;y;note\ns1;2e-324;north\ns0;10;south\n", merge, columns=columns)


@pytest.mark.parametrize("merge", MERGES)
def test_large_integers_keep_their_kind_across_the_join(tmp_path: Path, merge: str) -> None:
    a, b = f"id;x\ns0;{N}\ns1;0.5\n", "id;y\ns1;2\ns0;1\n"
    with pytest.raises(Exception, match="cannot represent exactly"):
        _package(tmp_path, "id;x\ns0;9007199254740993\ns1;0.5\n", b, merge)
    direct = _package(tmp_path, a, b, merge, features="float32")
    payload = dict(direct.partitions["train"].payloads)["train/x0"]
    assert isinstance(payload, FeatureMatrix) and payload.matrix.values_bytes().hex() == "0100805a0000003f"
    assert _y(direct) == [1.0, 2.0]
    exact = _package(tmp_path, "id;x\ns0;18014398509481984\ns1;0.5\n", b, merge)
    assert _x(exact) == [float(2**54), 0.5]
    with pytest.raises(NumberDomainError):
        _package(tmp_path, "id;x\ns0;18446744073709551617\ns1;0.5\n", b, merge)


@pytest.mark.parametrize("merge", MERGES)
def test_duplicate_headers_map_each_result_column_to_its_own_file(tmp_path: Path, merge: str) -> None:
    columns = {"features": ["x", "x__b"], "targets": "y"}  # a key join suffixes the later file's duplicate header
    package = _package(tmp_path, "id;x\ns0;1\ns1;2\n", "id;x;y\ns1;5;20\ns0;6;10\n", merge, columns=columns)
    assert _x(package) == [1.0, 6.0, 2.0, 5.0] and _y(package) == [10.0, 20.0]
    with pytest.raises(NumberDomainError, match="underflows"):
        _package(tmp_path, "id;x\ns0;1\ns1;2\n", "id;x;y\ns1;2e-324;20\ns0;6;10\n", merge, columns=columns)


def test_row_order_concatenation_prefixes_duplicate_headers(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x\n1\n2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x;y\n1e-400;10\n6;20\n", encoding="utf-8")
    spec = {"sources": [{"id": "s", "input": ["a.csv", "b.csv"], "merge": "concat_features", "columns": {"features": ["x", "b__x"], "targets": "y"}}]}
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.to_dataset_package(spec, base_dir=tmp_path)
    (tmp_path / "b.csv").write_text("x;y\n5;10\n6;20\n", encoding="utf-8")
    assert _x(nio.to_dataset_package(spec, base_dir=tmp_path)) == [1.0, 5.0, 2.0, 6.0]


@pytest.mark.parametrize("merge", MERGES)
def test_typed_sources_keep_their_decoded_cells_and_declared_precision(tmp_path: Path, merge: str) -> None:
    pd.DataFrame({"id": ["s1", "s0"], "y": np.array([2.5, 1.5], dtype=np.float32)}).to_parquet(tmp_path / "t.parquet")
    (tmp_path / "a.csv").write_text("id;x\ns0;0.000e-999\ns1;3\n", encoding="utf-8")
    source = {"id": "s", "input": ["a.csv", "t.parquet"], "merge": merge, "key": "id", "columns": {"features": "x", "targets": "y"}}
    package = nio.to_dataset_package({"sample_index": {"by": "id", "key": "id"}, "sources": [source]}, base_dir=tmp_path)
    targets = dict(package.partitions["train"].payloads)["train/y"]
    assert isinstance(targets, TargetTable)
    block = targets.columns[0].block
    assert block.values.ravel().tolist() == [1.5, 2.5] and block.precision.source_dtype == "float32" and block.dtype == "float32", "the genuine typed cells and declared dtype survive"
    assert _x(package) == [0.0, 3.0]
    (tmp_path / "a.csv").write_text("id;x\ns0;1e-400\ns1;3\n", encoding="utf-8")
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.to_dataset_package({"sample_index": {"by": "id", "key": "id"}, "sources": [source]}, base_dir=tmp_path)


def test_separate_sources_and_concat_samples_with_a_typed_file(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("id;x\ns0;1\ns1;2\n", encoding="utf-8")
    (tmp_path / "y.csv").write_text("id;y\ns1;20\ns0;10\ns9;2e-324\n", encoding="utf-8")
    spec = {"sample_index": {"by": "id", "key": "id"}, "sources": [
        {"id": "x", "input": "x.csv", "key": "id", "columns": {"features": "x"}},
        {"id": "y", "input": "y.csv", "key": "id", "columns": {"targets": "y"}, "join": {"to": "x", "on": "id", "how": "m:1", "coverage": "warn"}},
    ]}  # fmt: skip
    # a row the JOIN drops never refuses (the approved rule has no NA-only exception): the unmatched ``s9`` cell is a removed cell
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert _x(package) == [1.0, 2.0] and _y(package) == [10.0, 20.0]
    (tmp_path / "y.csv").write_text("id;y\ns1;2e-324\ns0;10\n", encoding="utf-8")
    with pytest.raises(NumberDomainError, match="underflows"):  # the same cell in a KEPT (matched) row still refuses
        nio.to_dataset_package(spec, base_dir=tmp_path)
    (tmp_path / "y.csv").write_text("id;y\ns1;20\ns0;10\n", encoding="utf-8")
    package = nio.to_dataset_package(spec, base_dir=tmp_path)
    assert _x(package) == [1.0, 2.0] and _y(package) == [10.0, 20.0]
    np.save(tmp_path / "t.npy", np.array([[2.5], [3.5]], dtype=np.float32))
    (tmp_path / "c.csv").write_text("0\n1.5\n", encoding="utf-8")
    spec = {"sources": [{"id": "m", "input": ["c.csv", "t.npy"], "merge": "concat_samples", "columns": {"features": "0"}, "params": {"header_unit": "index"}}]}
    stacked = nio.load(spec, target="assembled", base_dir=tmp_path)
    assert stacked.blocks["train"].X[0].ravel().tolist() == [1.5, 2.5, 3.5], "the typed file's decoded cells are untouched next to CSV text"
    (tmp_path / "c.csv").write_text("0\n1e-400\n", encoding="utf-8")
    with pytest.raises(NumberDomainError, match="underflows"):
        nio.load(spec, target="assembled", base_dir=tmp_path)
