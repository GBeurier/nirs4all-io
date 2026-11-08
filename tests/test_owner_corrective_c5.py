# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C5: a Parquet UInt64 above ``i64::MAX`` refuses at the typed reader, for any role of any READ column.

Public JSON ``u64``, generic ``uint64`` arrays and ``.npy`` files are other readers and stay valid; a column
excluded by ``format.columns`` is never read and never inspected; nullable cells within range are preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.loaders import LoaderError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

BIG = 2**63 + 2048


def _spec(columns: dict[str, str], fmt: dict[str, Any] | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {"id": "t", "input": "d.parquet", "columns": columns}
    if fmt:
        source["params"] = {"format": fmt}
    return {"sources": [source], "params": {"na": {"policy": "ignore"}}}


def _build(tmp_path, table, columns: dict[str, str], fmt: dict[str, Any] | None = None, **storage: Any):
    pq.write_table(table, tmp_path / "d.parquet")
    return nio.to_dataset_package(_spec(columns, fmt), base_dir=tmp_path, **storage)


def _u64(values) -> Any:
    return pa.array(values, pa.uint64())


@pytest.mark.parametrize(
    ("name", "table", "columns", "storage"),
    [
        ("features", lambda: pa.table({"x": _u64([BIG, 1]), "y": pa.array([1.0, 2.0])}), {"features": "x", "targets": "y"}, {}),
        ("features-float32", lambda: pa.table({"x": _u64([BIG, 1]), "y": pa.array([1.0, 2.0])}), {"features": "x", "targets": "y"}, {"features": "float32"}),
        ("u64max-float32", lambda: pa.table({"x": _u64([2**64 - 1, 1]), "y": pa.array([1.0, 2.0])}), {"features": "x", "targets": "y"}, {"features": "float32"}),
        ("targets", lambda: pa.table({"x": pa.array([1.5, 2.5]), "y": _u64([BIG, 1])}), {"features": "x", "targets": "y"}, {}),
        ("unselected", lambda: pa.table({"x": pa.array([1.5, 2.5]), "y": pa.array([1.0, 2.0]), "z": _u64([BIG, 1])}), {"features": "x", "targets": "y"}, {}),
        ("metadata", lambda: pa.table({"x": pa.array([1.5, 2.5]), "y": pa.array([1.0, 2.0]), "m": _u64([BIG, 1])}), {"features": "x", "targets": "y", "metadata": "m"}, {}),
        ("nullable-big", lambda: pa.table({"x": _u64([BIG, None]), "y": pa.array([1.0, 2.0])}), {"features": "x", "targets": "y"}, {}),
    ],
)
def test_uint64_above_i64_max_refuses_for_any_role(tmp_path, name: str, table, columns: dict[str, str], storage: dict[str, Any]) -> None:
    with pytest.raises(LoaderError, match="outside i64 range"):
        _build(tmp_path, table(), columns, **storage)


def test_a_column_excluded_by_format_columns_is_never_read(tmp_path) -> None:
    table = pa.table({"x": pa.array([1.5, 2.5]), "y": pa.array([1.0, 2.0]), "z": _u64([BIG, 1])})
    assert _build(tmp_path, table, {"features": "x", "targets": "y"}, {"columns": ["x", "y"]}).partitions


def test_in_range_unsigned_and_signed_controls_stay_valid(tmp_path) -> None:
    columns = {"features": "x", "targets": "y"}
    for table in (
        pa.table({"x": _u64([2**63 - 1, 1]), "y": pa.array([1.0, 2.0])}),  # exactly i64::MAX
        pa.table({"x": _u64([5, 1]), "y": pa.array([1.0, 2.0])}),
        pa.table({"x": _u64([7, None]), "y": pa.array([1.0, 2.0])}),  # nullable, within range
        pa.table({"x": pa.array([2**62, 1], pa.int64()), "y": pa.array([1.0, 2.0])}),
    ):
        assert _build(tmp_path, table, columns, features="float32").partitions
    nullable = _build(tmp_path, pa.table({"x": _u64([7, None]), "y": pa.array([1.0, 2.0])}), columns)
    matrix = dict(nullable.partitions["train"].payloads)["train/x0"].matrix
    assert matrix.validity.ravel().tolist() == [True, False] and matrix.values.ravel().tolist() == [7.0, 0.0]


def test_other_unsigned_readers_keep_their_own_domain(tmp_path) -> None:
    np.save(tmp_path / "x.npy", np.array([[2**63 + 2048, 1]], dtype=np.uint64))
    spec = {"sources": [{"id": "x", "role": "features", "input": "x.npy", "params": {"header_unit": "index"}}]}
    assert nio.to_dataset_package(spec, base_dir=tmp_path, features="float32").partitions  # .npy is not a Parquet reader
