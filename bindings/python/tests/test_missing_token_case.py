# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Missing-value tokens match without regard to ASCII case, in every role."""
import pytest

import nirs4all_io as nio

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
    return nio.assembled_full(_spec(tmp_path, role, token))["blocks"]["train"]


def _inferred_role(tmp_path, column):
    plan = nio.infer(str(tmp_path), params={"delimiter": ",", "na": {"policy": "ignore"}})
    return next(entry["role"] for entry in plan["resolved_spec"]["sources"][0]["columns"] if column in entry["select"])


@pytest.mark.parametrize("token", MISSING)
def test_target_token_is_missing_and_the_column_stays_numeric(tmp_path, token):
    column = _train(tmp_path, "target", token)["y"]["columns"][0]
    assert column["kind"] == "numeric"
    assert column["block"]["validity"] == [row != ROW for row in range(8)]
    assert _inferred_role(tmp_path, "y") == "targets"


@pytest.mark.parametrize("token", MISSING)
def test_spectral_token_is_missing(tmp_path, token):
    validity = _train(tmp_path, "spectral", token)["x"][0]["validity"]
    assert validity == [index != ROW * 3 + 1 for index in range(24)]
    assert _inferred_role(tmp_path, "x2") == "features"


@pytest.mark.parametrize("token", MISSING)
def test_metadata_token_is_missing_and_the_column_stays_numeric(tmp_path, token):
    columns = _train(tmp_path, "metadata", token)["metadata"]["columns"]
    values = next(column["values"] for column in columns if column["name"] == "batch")
    assert values == [None if row == ROW else float(row % 3 + 1) for row in range(8)]


@pytest.mark.parametrize("token", TEXT)
def test_spelling_outside_the_vocabulary_stays_text(tmp_path, token):
    assert _train(tmp_path, "target", token)["y"]["columns"][0]["kind"] == "classification_string"
    assert _inferred_role(tmp_path, "y") == "metadata"
    columns = _train(tmp_path, "metadata", token)["metadata"]["columns"]
    assert next(column["values"] for column in columns if column["name"] == "batch")[ROW] == token


@pytest.mark.parametrize("token", ["na", "Nan", "NONE"])
def test_text_column_keeps_its_labels_and_only_the_token_cell_is_missing(tmp_path, token):
    path = tmp_path / "labels.csv"
    path.write_text(f"id,x1,site,y\ns0,1,north,low\ns1,2,{token},high\ns2,3,south,{token}\ns3,4,north,low\n", encoding="utf-8")
    spec = {
        "sources": [{"id": "t", "input": str(path), "columns": {"features": ["x1"], "targets": "y", "metadata": ["id", "site"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"delimiter": ",", "na": {"policy": "ignore"}},
    }
    block = nio.assembled_full(spec)["blocks"]["train"]
    site = next(column["values"] for column in block["metadata"]["columns"] if column["name"] == "site")
    assert site == ["north", None, "south", "north"]
    target = block["y"]["columns"][0]
    assert target["kind"] == "classification_string"
    assert target["block"]["validity"] == [True, True, False, True]
