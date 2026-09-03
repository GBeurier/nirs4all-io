# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("qualification.py")
SPEC = importlib.util.spec_from_file_location("io_xlg_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def _rows(disposition: str = "passed") -> list[dict[str, object]]:
    return [
        {"surface": surface, "disposition": disposition, "reason": "test", "artifacts": []}
        for surface in qualification.SURFACES
    ]


def test_complete_requires_every_surface_passed() -> None:
    assert qualification.finalize_report(_rows())["overall_complete"] is True
    rows = _rows()
    rows[3]["disposition"] = "unavailable"
    assert qualification.finalize_report(rows)["overall_complete"] is False
    rows = _rows()
    rows[2]["disposition"] = "refused"
    assert qualification.finalize_report(rows)["overall_complete"] is False


def test_frozen_summary_carries_every_identity_axis() -> None:
    summary = json.loads(qualification.GOLDEN.read_text(encoding="utf-8"))
    qualification.validate_identity(summary)
    assert summary["assembled_schema_version"] == 2
    assert summary["blocks"]["train"]["source_ids"] == ["data"]
    assert summary["folds"] == [[[0, 2], [1]]]


def test_strict_runner_never_installs_r_dependencies() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "install.packages" not in source
    assert "requireNamespace('jsonlite',quietly=TRUE)" in source


def test_isolated_r_library_preserves_explicit_host_closure(tmp_path: Path) -> None:
    host_library = os.pathsep.join(("/opt/r/closure-a", "/opt/r/closure-b"))
    combined = qualification.prepend_search_path(tmp_path / "package", host_library)

    assert combined.split(os.pathsep) == [str(tmp_path / "package"), "/opt/r/closure-a", "/opt/r/closure-b"]


def test_r_configure_uses_disposable_exact_source_copy(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    (source / "bindings" / "r").mkdir(parents=True)
    (source / "bindings" / "r" / "configure").write_text("original\n", encoding="utf-8")
    for crate in qualification.R_VENDOR_CRATES:
        crate_root = source / "crates" / crate
        crate_root.mkdir(parents=True)
        (crate_root / "Cargo.toml").write_text(f"# {crate}\n", encoding="utf-8")
    monkeypatch.setattr(qualification, "ROOT", source)

    package = qualification.prepare_r_source_tree(tmp_path / "work")
    (package / "configure").write_text("mutated\n", encoding="utf-8")

    assert (source / "bindings" / "r" / "configure").read_text(encoding="utf-8") == "original\n"
    assert (package / "configure").read_text(encoding="utf-8") == "mutated\n"
    for crate in qualification.R_VENDOR_CRATES:
        assert (tmp_path / "work" / "r-source" / "crates" / crate / "Cargo.toml").is_file()
