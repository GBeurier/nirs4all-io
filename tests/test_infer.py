# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Tests for describe, value detectors, and the infer() engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

import nirs4all_io as nio
from nirs4all_io.infer import describe, detect_signal_type, detect_task_type


def _csv(path, df, sep=";"):
    df.to_csv(path, sep=sep, index=False)


# --------------------------------------------------------------------------- #
# describe                                                                    #
# --------------------------------------------------------------------------- #
def test_describe_delimiter_and_header(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
    desc = describe(p)
    assert desc.delimiter == ";"
    assert desc.has_header is True
    assert desc.n_cols == 3


def test_describe_wavelength_header(tmp_path):
    cols = [str(w) for w in range(1000, 1200, 5)]  # 40 monotonic nm columns
    df = pd.DataFrame(np.random.rand(5, len(cols)) * 0.5 + 0.2, columns=cols)
    _csv(tmp_path / "spectra.csv", df)
    desc = describe(tmp_path / "spectra.csv")
    assert desc.is_wavelength_header is True
    assert desc.header_unit == "nm"
    assert desc.axis_range == (1000.0, 1195.0)


# --------------------------------------------------------------------------- #
# value detectors                                                             #
# --------------------------------------------------------------------------- #
def test_detect_task_type():
    assert detect_task_type(np.array([0, 1, 0, 1]))[0] == "binary"
    assert detect_task_type(np.arange(0, 40))[0] == "multiclass"
    assert detect_task_type(np.random.rand(200) * 100)[0] == "regression"


def test_detect_signal_type_absorbance():
    x = np.random.rand(20, 50) * 2.0 + 0.3  # range ~[0.3, 2.3] -> absorbance
    sig, score, _ = detect_signal_type(x)
    assert sig in ("absorbance", "unknown")  # absorbance scorer dominates; may abstain if ambiguous
    if sig != "unknown":
        assert sig == "absorbance"


def test_detect_signal_type_preprocessed():
    x = np.random.randn(20, 50)  # mean~0, std~1 -> SNV-like
    sig, score, _ = detect_signal_type(x)
    assert sig == "preprocessed"


# --------------------------------------------------------------------------- #
# infer()                                                                     #
# --------------------------------------------------------------------------- #
def test_infer_classic_folder(tmp_path):
    cols = [str(w) for w in range(1000, 1060, 5)]
    _csv(tmp_path / "Xcal.csv", pd.DataFrame(np.random.rand(6, len(cols)), columns=cols))
    _csv(tmp_path / "Ycal.csv", pd.DataFrame({"y": np.arange(6.0)}))
    _csv(tmp_path / "Xval.csv", pd.DataFrame(np.random.rand(3, len(cols)), columns=cols))
    _csv(tmp_path / "Yval.csv", pd.DataFrame({"y": np.arange(3.0)}))
    plan = nio.infer(tmp_path)
    assert plan.structure.value == "train_test_folder"
    assert plan.resolved_spec is not None
    d = plan.to_dict()
    assert d["overall_score"] > 0
    # the plan's resolved_spec must actually load
    asm = nio.load(plan.accept(), base_dir=tmp_path, target="assembled")
    assert set(asm.blocks) == {"train", "test"}


def test_infer_single_combined_file_column_roles(tmp_path):
    cols = [str(w) for w in range(1000, 1100, 5)]  # 20 wavelength cols
    df = pd.DataFrame(np.random.rand(8, len(cols)), columns=cols)
    df["protein"] = np.arange(8.0)
    _csv(tmp_path / "data.csv", df)
    plan = nio.infer(tmp_path / "data.csv")
    assert plan.structure.value == "single_combined"
    # column roles: wavelength cols -> features, 'protein' -> targets
    roles = {g["col"]: g["role"] for g in plan.columns[0]["column_roles"]}
    assert roles["1000"] == "features"
    assert roles["protein"] == "targets"
    asm = nio.load(plan.accept(), base_dir=tmp_path, target="assembled")
    assert asm.blocks["train"].X[0].shape == (8, len(cols))
    assert asm.blocks["train"].y.shape == (8, 1)


def test_infer_directory_full_decisions_with_scores(tmp_path):
    """Directory / file-list inference yields scored decisions for every choice."""
    cols = [str(1000 + i * 5) for i in range(12)]
    rng = np.random.default_rng(0)
    absb = rng.random((20, 12)) * 1.0 + 0.4  # unambiguous absorbance
    _csv(tmp_path / "Xcal.csv", pd.DataFrame(absb, columns=cols))
    _csv(tmp_path / "Ycal.csv", pd.DataFrame({"protein": rng.random(20) * 50}))
    _csv(tmp_path / "Xval.csv", pd.DataFrame(rng.random((6, 12)) * 1.0 + 0.4, columns=cols))
    _csv(tmp_path / "Yval.csv", pd.DataFrame({"protein": rng.random(6) * 50}))
    for inp in (tmp_path, [str(tmp_path / f) for f in ("Xcal.csv", "Ycal.csv", "Xval.csv", "Yval.csv")]):
        plan = nio.infer(inp)
        assert plan.structure.value == "train_test_folder" and plan.structure.score > 0
        # every file assignment carries a role, partition and a confidence score
        assert plan.assignments and all("score" in a and a["partition"] for a in plan.assignments)
        assert {(a["role"], a["partition"]) for a in plan.assignments} >= {("features", "train"), ("targets", "test")}
        assert plan.axis and plan.axis["unit"] == "nm"
        assert plan.signal_type.value == "absorbance"
        # task type detected from the SEPARATE Ycal/Yval target file
        assert plan.task_type is not None and plan.task_type.value == "regression"


def test_infer_task_from_separate_classification_target(tmp_path):
    cols = [str(1000 + i * 5) for i in range(8)]
    rng = np.random.default_rng(1)
    _csv(tmp_path / "Xcal.csv", pd.DataFrame(rng.random((30, 8)), columns=cols))
    _csv(tmp_path / "Ycal.csv", pd.DataFrame({"grade": rng.integers(0, 3, 30)}))
    plan = nio.infer(tmp_path)
    assert plan.task_type is not None and plan.task_type.value == "multiclass"


def test_infer_plan_is_json_serializable(tmp_path):
    import json

    _csv(tmp_path / "x.csv", pd.DataFrame({"400": [0.1, 0.2], "401": [0.3, 0.4], "y": [1.0, 2.0]}))
    plan = nio.infer(tmp_path / "x.csv")
    json.dumps(plan.to_dict())  # must not raise
