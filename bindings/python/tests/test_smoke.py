# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Smoke tests for the pyo3 binding's JSON surface against the contract corpus."""
from pathlib import Path

import pytest

import nirs4all_io as nio

CORPUS = Path(__file__).resolve().parents[3] / "tests/goldens/contract/corpus"


def test_infer_returns_plan_dict():
    plan = nio.infer(str(CORPUS / "single_combined"))
    assert isinstance(plan, dict)
    assert "resolved_spec" in plan


def test_to_spec_returns_validatable_spec():
    spec = nio.to_spec(str(CORPUS / "train_test"))
    assert spec["schema_version"] == 1
    # the produced spec round-trips through validate()
    nio.validate(spec)


def test_load_assembled_summary():
    assembled = nio.load(str(CORPUS / "x_y_separate"), target="assembled")
    assert "blocks" in assembled


def test_validate_rejects_invalid_spec():
    with pytest.raises(ValueError):
        nio.validate({"partitions": {"by": "random"}})


def test_spectrodataset_target_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        nio.load(str(CORPUS / "train_test"), target="spectrodataset")
