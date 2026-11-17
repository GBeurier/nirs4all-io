# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Selecting some columns of a file: the others are ignored and never drive the NA policy."""
import numpy as np
import pytest
from nirs4all_io._adapter import _matrix

import nirs4all_io as nio

ROWS, TARGETS = 300, 49


def _small(tmp_path, part, policy, **filters):
    """Six samples; ``b`` misses row 0, ``a`` and ``c`` miss rows 3 and 1, ``note`` rows 0, 2 and 4."""
    (tmp_path / "X.csv").write_text("1000;1100\n" + "".join(f"{i};{i + 10}\n" for i in range(6)))
    (tmp_path / "Y.csv").write_text("a;b;c\n1;;1\n2;2;\n3;3;3\n;4;4\n5;5;5\n6;6;6\n")
    (tmp_path / "M.csv").write_text("sid;site;note\ns0;u;\ns1;v;k\ns2;u;\ns3;v;k\ns4;u;\ns5;v;k\n")
    config = {
        f"{part}_x": str(tmp_path / "X.csv"), f"{part}_y": str(tmp_path / "Y.csv"), f"{part}_group": str(tmp_path / "M.csv"),
        f"{part}_group_filter": ["sid", "site"],
        "global_params": {"delimiter": ";", "has_header": True, "na": {"policy": policy}},
    }
    config.update({f"{part}_{key}": value for key, value in filters.items()})
    return nio.assembled_full(config)["blocks"][part]


def _metadata(block):
    return {column["name"]: column["values"] for column in block["metadata"]["columns"]}


def _targets(block):
    """Target names and values, with ``nan`` where the validity mask says missing."""
    columns = block["y"]["columns"]
    values = np.column_stack([np.where(column["block"]["validity"], _matrix(column["block"])[:, 0], np.nan) for column in columns])
    return [column["name"] for column in columns], values


def _rows(block):
    return _matrix(block["x"][0])[:, 0].tolist(), _targets(block)[1], _metadata(block)["sid"]


@pytest.mark.parametrize("part", ["train", "test"])
@pytest.mark.parametrize("y_filter", [[1], ["b"]])
def test_target_filter_selects_its_column_and_ignores_the_rest(tmp_path, part, y_filter):
    block = _small(tmp_path, part, "ignore", y_filter=y_filter)
    x, y, sid = _rows(block)
    assert _targets(block)[0] == ["b"]
    assert x == [0, 1, 2, 3, 4, 5] and sid == ["s0", "s1", "s2", "s3", "s4", "s5"]
    np.testing.assert_array_equal(y[:, 0], [np.nan, 2, 3, 4, 5, 6])
    assert list(_metadata(block)) == ["sid", "site"]


@pytest.mark.parametrize("part", ["train", "test"])
def test_several_targets_and_a_feature_filter(tmp_path, part):
    block = _small(tmp_path, part, "ignore", y_filter=["c", "a"], x_filter=[1])
    assert _targets(block)[0] == ["a", "c"]
    np.testing.assert_array_equal(_matrix(block["x"][0]), [[10], [11], [12], [13], [14], [15]])


def test_feature_filter_beside_targets_carried_by_the_feature_file(tmp_path):
    (tmp_path / "XY.csv").write_text("1000;1100;y;z\n1;2;3;\n4;5;6;7\n")
    config = {"train_x": str(tmp_path / "XY.csv"), "train_x_filter": [0, 1], "train_y_filter": ["y"], "global_params": {"delimiter": ";", "has_header": True}}
    block = nio.assembled_full(config)["blocks"]["train"]
    np.testing.assert_array_equal(_matrix(block["x"][0]), [[1, 2], [4, 5]])
    np.testing.assert_array_equal(_targets(block)[1][:, 0], [3, 6])


@pytest.mark.parametrize("part", ["train", "test"])
def test_missing_values_in_ignored_columns_never_remove_a_sample_or_abort(tmp_path, part):
    # only b is selected: its single missing row goes, rows 1 and 3 (missing in the ignored a and c) stay
    x, y, sid = _rows(_small(tmp_path, part, "remove_sample", y_filter=[1]))
    assert x == [1, 2, 3, 4, 5] and sid == ["s1", "s2", "s3", "s4", "s5"]
    np.testing.assert_array_equal(y[:, 0], [2, 3, 4, 5, 6])
    # selecting a and c removes exactly their missing rows 1 and 3
    x, y, sid = _rows(_small(tmp_path, part, "remove_sample", y_filter=["a", "c"]))
    assert x == [0, 2, 4, 5] and sid == ["s0", "s2", "s4", "s5"]
    np.testing.assert_array_equal(y, [[1, 1], [3, 3], [5, 5], [6, 6]])
    # the default policy keeps a missing target and ignores unselected columns
    for policy in ("auto", "ignore"):
        x, y, _ = _rows(_small(tmp_path, part, policy, y_filter=[1]))
        assert len(x) == 6 and np.isnan(y[0, 0])
    # abort refuses the selected target's own missing value only
    with pytest.raises(ValueError, match="column 'b'"):
        _small(tmp_path, part, "abort", y_filter=[1])


def _missing(row, column):
    """49 targets: ``t00`` complete, ``tNN`` missing 3% to 90% of the rows."""
    return column > 0 and (row * 37 + column * 11) % 100 < 3 + (column - 1) * 87 // 47


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    root = tmp_path_factory.mktemp("wide")
    (root / "X.csv").write_text("1000;1100\n" + "".join(f"{i};{i * 2}\n" for i in range(ROWS)))
    lines = [";".join(f"t{c:02d}" for c in range(TARGETS))]
    lines += [";".join("" if _missing(row, c) else str(row * 100 + c) for c in range(TARGETS)) for row in range(ROWS)]
    (root / "Y.csv").write_text("\n".join(lines) + "\n")
    (root / "M.csv").write_text("sid\n" + "".join(f"s{i:03d}\n" for i in range(ROWS)))
    return root


@pytest.mark.parametrize("column", [0, 1, 24, 48])
@pytest.mark.parametrize("by_name", [False, True])
def test_one_target_of_a_wide_sparse_file_keeps_every_sample_it_observes(wide, column, by_name):
    def load(policy):
        config = {
            "train_x": str(wide / "X.csv"), "train_y": str(wide / "Y.csv"), "train_group": str(wide / "M.csv"),
            "train_y_filter": [f"t{column:02d}" if by_name else column],
            "global_params": {"delimiter": ";", "has_header": True, "na": {"policy": policy}},
        }
        return _rows(nio.assembled_full(config)["blocks"]["train"])

    observed = [row for row in range(ROWS) if not _missing(row, column)]
    assert (len(observed) == ROWS) == (column == 0)
    x, y, sid = load("ignore")
    assert len(x) == ROWS == len(sid)
    np.testing.assert_array_equal(np.isnan(y[:, 0]), [_missing(row, column) for row in range(ROWS)])
    x, y, sid = load("remove_sample")
    assert x == observed
    np.testing.assert_array_equal(y[:, 0], [row * 100 + column for row in observed])
    assert sid == [f"s{row:03d}" for row in observed]
