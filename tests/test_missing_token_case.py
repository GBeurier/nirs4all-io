"""Missing-value tokens match without regard to ASCII case, in every role (oracle side of the Rust rule)."""
import numpy as np
import pandas as pd
import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.loaders import NAError

MISSING = [
    "", "NA", "na", "Na", "nA", "N/A", "n/a", "N/a", "NaN", "nan", "Nan", "NAN", "naN", "nAn", "null", "NULL", "Null",
    "None", "none", "NONE", "#N/A", "#n/a", "#NA", "#na", "<NA>", "<na>", "-NaN", "-nan", "-NAN",
]
TEXT = [" ", "-", "?", ".", "missing", " NA"]
ROW = 3


def _spec(tmp_path, role, token):
    rows = ["id,x1,x2,x3,site,batch,y"]
    for row in range(8):
        cells = {"spectral": f"{0.25 * row}", "metadata": f"{row % 3 + 1}", "target": f"{10 + 1.5 * row}"}
        if row == ROW:
            cells[role] = token
        rows.append(f"s{row},{float(row)},{cells['spectral']},{0.5 * row},{'AB'[row % 2]},{cells['metadata']},{cells['target']}")
    path = tmp_path / "data.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "task_type": "regression",
        "sources": [{"id": "t", "input": str(path), "columns": {"features": ["x1", "x2", "x3"], "targets": "y", "metadata": ["id", "site", "batch"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"delimiter": ",", "na": {"policy": "ignore"}},
    }


def _train(tmp_path, role, token):
    return nio.load(_spec(tmp_path, role, token), target="assembled").blocks["train"]


@pytest.mark.parametrize("token", MISSING)
def test_target_token_is_missing_and_the_column_stays_numeric(tmp_path, token):
    y = _train(tmp_path, "target", token).y[:, 0]
    np.testing.assert_array_equal(y, [np.nan if row == ROW else 10 + 1.5 * row for row in range(8)])
    # inspection sees the same missing cell: the default inference policy refuses it instead of reading a text column
    with pytest.raises(NAError, match="column 'y'"):
        nio.infer(str(tmp_path))


@pytest.mark.parametrize("token", MISSING)
def test_spectral_token_is_missing(tmp_path, token):
    x = _train(tmp_path, "spectral", token).X[0]
    np.testing.assert_array_equal(np.isnan(x), [[False, row == ROW, False] for row in range(8)])


@pytest.mark.parametrize("token", MISSING)
def test_metadata_token_is_missing_and_the_column_stays_numeric(tmp_path, token):
    batch = _train(tmp_path, "metadata", token).metadata["batch"]
    assert pd.api.types.is_float_dtype(batch)
    np.testing.assert_array_equal(batch, [np.nan if row == ROW else row % 3 + 1 for row in range(8)])


@pytest.mark.parametrize("token", TEXT)
def test_spelling_outside_the_vocabulary_stays_text(tmp_path, token):
    assert _train(tmp_path, "metadata", token).metadata["batch"].iloc[ROW] == token
    _spec(tmp_path, "target", token)
    columns = nio.infer(str(tmp_path)).resolved_spec.to_dict()["sources"][0]["columns"]
    assert next(entry["role"] for entry in columns if "y" in entry["select"]) == "metadata"


@pytest.mark.parametrize("token", ["na", "Nan", "NONE"])
def test_text_column_keeps_its_labels_and_only_the_token_cell_is_missing(tmp_path, token):
    path = tmp_path / "labels.csv"
    path.write_text(f"id,x1,site,y\ns0,1,north,low\ns1,2,{token},high\ns2,3,south,{token}\ns3,4,north,low\n", encoding="utf-8")
    spec = {
        "sources": [{"id": "t", "input": str(path), "columns": {"features": ["x1"], "targets": "y", "metadata": ["id", "site"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"delimiter": ",", "na": {"policy": "ignore"}},
    }
    block = nio.load(spec, target="assembled").blocks["train"]
    site = block.metadata["site"]
    assert site.isna().tolist() == [False, True, False, False]
    assert site.dropna().tolist() == ["north", "south", "north"]
    np.testing.assert_array_equal(np.isnan(block.y[:, 0]), [False, False, True, False])


@pytest.mark.parametrize("token", ["na", "Nan", "NULL"])
def test_excel_text_cell_token_is_missing(tmp_path, token):
    pytest.importorskip("openpyxl")
    path = tmp_path / "data.xlsx"
    pd.DataFrame({"id": ["s0", "s1", "s2", "s3"], "x1": [0.1, 0.2, 0.3, 0.4], "batch": [1, token, 3, 1], "y": [1.5, 2.5, token, 4.5]}).to_excel(path, index=False)
    spec = {
        "task_type": "regression",
        "sources": [{"id": "t", "input": str(path), "columns": {"features": ["x1"], "targets": "y", "metadata": ["id", "batch"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"na": {"policy": "ignore"}},
    }
    block = nio.load(spec, target="assembled").blocks["train"]
    np.testing.assert_array_equal(block.y[:, 0], [1.5, 2.5, np.nan, 4.5])
    np.testing.assert_array_equal(block.metadata["batch"], [1.0, np.nan, 3.0, 1.0])
