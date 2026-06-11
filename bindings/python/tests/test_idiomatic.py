# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""The idiomatic Python surface: native inputs (str / Path / list / dict) in,
typed DatasetPlan / DatasetSpec mappings out."""

import json
from pathlib import Path

import pytest

import nirs4all_io as nio

CORPUS = Path(__file__).resolve().parents[3] / "tests/goldens/contract/corpus"


def test_to_spec_accepts_pathlib_and_returns_typed_spec():
    spec = nio.to_spec(CORPUS / "train_test")  # a pathlib.Path, not a str
    assert isinstance(spec, nio.DatasetSpec)
    assert isinstance(spec, dict)  # still a dict: subscriptable, serializable
    assert spec.schema_version == 1 == spec["schema_version"]
    assert spec.name == "train_test"
    assert len(spec.sources) >= 1
    json.dumps(spec)  # JSON-serializable
    nio.validate(spec)  # the typed spec round-trips through validate
    assert "DatasetSpec" in repr(spec) and "train_test" in repr(spec)


def test_infer_accepts_pathlib_list_and_returns_typed_plan():
    plan = nio.infer([CORPUS / "x_y_separate" / "X.csv", CORPUS / "x_y_separate" / "Y.csv"])
    assert isinstance(plan, nio.DatasetPlan)
    assert "resolved_spec" in plan


def test_plan_exposes_decisions_and_resolved_spec():
    plan = nio.infer(CORPUS / "single_combined")
    decisions = plan.decisions()
    assert {"structure", "task_type", "signal_type"} <= decisions.keys()
    for d in decisions.values():
        assert {"value", "score", "ambiguous"} <= d.keys()
    assert isinstance(plan.overall_score, float)
    rs = plan.resolved_spec
    assert isinstance(rs, nio.DatasetSpec)
    nio.validate(rs)
    assert "DatasetPlan" in repr(plan)


def test_load_accepts_pathlib():
    assembled = nio.load(CORPUS / "x_y_separate", target="assembled")
    assert "blocks" in assembled


def test_validate_rejects_invalid_typed_path():
    with pytest.raises(ValueError):
        nio.validate({"partitions": {"by": "random"}})


def test_public_surface_unchanged():
    # The historical names plus the new typed classes are all exported.
    for name in ("infer", "to_spec", "validate", "load", "to_spectrodataset", "__version__"):
        assert name in nio.__all__
    assert "DatasetPlan" in nio.__all__ and "DatasetSpec" in nio.__all__
