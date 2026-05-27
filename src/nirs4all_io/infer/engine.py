# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Inference engine: any input -> a scored DatasetPlan (Epic 3.2/3.3/3.5).

Composes the convention match (structure + file roles), :func:`describe` (neutral
per-file params + axis), value detectors (signal/task type), and column-role
inference (the genuinely-new bit vs studio) into a :class:`DatasetPlan` whose
``resolved_spec`` ``load`` can execute. Scores are uncalibrated (C5).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..conventions.engine import match_items
from ..conventions.profiles import resolve_profiles
from ..conventions.to_spec import assignments_to_spec_dict
from ..materialize.loaders import coerce_numeric, read_table
from ..resolve import resolve
from ..spec import DatasetSpec
from ..spec.enums import Role
from .describe import FileDescription, describe
from .detectors import detect_signal_type, detect_task_type
from .plan import DatasetPlan, Decision

_WL_RE = re.compile(r"^\d+(\.\d+)?(nm|cm-1)?$", re.IGNORECASE)
_DEFAULT_CONVENTIONS = ["nirs4all-classic"]


def infer(inp: object, *, conventions: list[str] | None = None, hints: dict | None = None) -> DatasetPlan:
    """Inspect ``inp`` and propose a confidence-scored :class:`DatasetPlan`."""
    iset = resolve(inp)
    profiles = resolve_profiles(conventions or _DEFAULT_CONVENTIONS)
    result = match_items(iset.names, profiles)
    base_dir = Path(iset.items[0].identity).parent if iset.items else Path(".")
    plan = DatasetPlan(input={"kind": iset.origin.get("kind"), "ref": iset.origin.get("ref")})

    file_items = [it for it in iset.items if it.kind == "file"]
    if result.vendor is not None and result.vendor.spectra:
        plan.structure = Decision("vendor_corpus", 0.9, [f"{len(result.vendor.spectra)} vendor spectra + {len(result.vendor.reference)} reference file(s)"])
        spec_dict = assignments_to_spec_dict(result, name="dataset")
    elif result.assignments:
        kind, score = _structure_kind(result)
        plan.structure = Decision(kind, score, [f"matched {len(result.assignments)} file(s) via conventions"])
        plan.assignments = [
            {
                "ref": a.name,
                "role": a.role.value,
                "partition": a.partition.value if a.partition else None,
                "source_index": a.source_index,
                "score": 0.75 if a.matched_pattern.startswith("bare:") else 0.9,  # bare-stem match is weaker than a role pattern
                "evidence": [f"filename matches {a.matched_pattern}"],
            }
            for a in result.assignments
        ]
        spec_dict = assignments_to_spec_dict(result, name="dataset")
    elif len(file_items) == 1:
        plan.structure = Decision("single_combined", 0.7, ["one tabular file; column roles inferred"])
        spec_dict = _infer_single_file(file_items[0].identity, plan)
    else:
        plan.structure = Decision("unknown", 0.2, [f"{len(file_items)} file(s), no convention match"])
        plan.warnings.append("no convention matched; provide an explicit DatasetSpec or conventions")
        spec_dict = {"name": "dataset", "sources": []}

    _enrich_params(spec_dict, base_dir, plan)
    _infer_signal_and_task(spec_dict, base_dir, plan)
    _absolutize_inputs(spec_dict, base_dir, iset)  # so load(plan.accept()) works anywhere

    if result.unmatched:
        plan.warnings.append(f"unassigned files: {result.unmatched}")
    plan.warnings.extend(result.warnings)
    plan.resolved_spec = DatasetSpec.from_dict(spec_dict)
    decisions = [d for d in (plan.structure, plan.signal_type, plan.task_type) if d is not None]
    plan.overall_score = round(float(np.mean([d.score for d in decisions])) if decisions else 0.0, 3)
    plan.recommendations.append("review the resolved_spec; pass it (or plan.accept()) to load()")
    return plan


def _structure_kind(result) -> tuple[str, float]:
    partitions = {a.partition for a in result.assignments if a.partition}
    roles = {a.role for a in result.assignments}
    if len(partitions) > 1:
        return "train_test_folder", 0.92
    if Role.FEATURES in roles and Role.TARGETS in roles:
        return "x_y_separate", 0.85
    return "single_partition_folder", 0.8


def infer_column_roles(desc: FileDescription, df: pd.DataFrame) -> list[dict]:
    """Assign a role to each column (features/targets/metadata) with evidence (heuristics A,C,I)."""
    headers = [str(c) for c in df.columns]
    wl_cols = {h for h in headers if _WL_RE.match(h)} if desc.is_wavelength_header else set()
    numeric_non_wl = [h for h in headers if h not in wl_cols and pd.api.types.is_numeric_dtype(df[h])]
    guesses: list[dict] = []
    for h in headers:
        if h in wl_cols:
            guesses.append({"col": h, "role": "features", "score": 0.9, "evidence": ["monotonic nm/cm-1 wavelength column"]})
        elif not pd.api.types.is_numeric_dtype(df[h]):
            guesses.append({"col": h, "role": "metadata", "score": 0.7, "evidence": ["non-numeric / id-like column"]})
        elif not wl_cols and h == numeric_non_wl[-1] and len(numeric_non_wl) > 1:
            guesses.append({"col": h, "role": "targets", "score": 0.6, "evidence": ["last numeric column (likely target)"]})
        elif wl_cols and h in numeric_non_wl and len(numeric_non_wl) <= 3:
            guesses.append({"col": h, "role": "targets", "score": 0.65, "evidence": ["numeric non-wavelength column alongside a spectral block"]})
        else:
            guesses.append({"col": h, "role": "features", "score": 0.55, "evidence": ["numeric column"]})
    return guesses


def _infer_single_file(path: str, plan: DatasetPlan) -> dict:
    from ..spec.dataset_spec import LoadingParams

    desc = describe(path)
    # Read WITH a header using the detected delimiter so the inferred name-based
    # column roles match what load() will read. Like nirs4all's loader, default
    # has_header=True for tabular data (the speculative no-header verdict on a
    # numeric wavelength header is unreliable).
    table = read_table(path, LoadingParams(delimiter=desc.delimiter, has_header=True))
    desc.column_names = list(table.df.columns)
    guesses = infer_column_roles(desc, table.df)
    plan.columns = [{"ref": Path(path).name, "column_roles": guesses}]
    columns: list[dict] = []
    for role in ("features", "targets", "metadata"):
        cols = [g["col"] for g in guesses if g["role"] == role]
        if cols:
            columns.append({"role": role, "select": cols})
    return {"name": Path(path).stem, "sources": [{"id": "data", "role": "mixed", "input": Path(path).name, "columns": columns, "params": {"delimiter": desc.delimiter, "has_header": True}}]}


def _enrich_params(spec_dict: dict, base_dir: Path, plan: DatasetPlan) -> None:
    for src in spec_dict.get("sources", []):
        inp = src.get("input")
        first = inp[0] if isinstance(inp, list) else inp
        if not isinstance(first, str):
            continue
        path = Path(first) if Path(first).is_absolute() else base_dir / first
        if not path.exists() or path.suffix.lower() not in (".csv", ".tsv", ".txt"):
            continue
        desc = describe(path)
        params = src.setdefault("params", {})
        params.setdefault("delimiter", desc.delimiter)
        # only assert a header when detected positively; never speculatively set
        # has_header=False (spectral CSVs default to having a header, like nirs4all)
        if desc.has_header:
            params.setdefault("has_header", True)
        if src.get("role") in ("features", "mixed") and desc.header_unit in ("nm", "cm-1"):
            params.setdefault("header_unit", desc.header_unit)
        plan.params[Path(first).name] = {k: {"value": v, "score": round(desc.confidence.get(k, 0.5), 3)} for k, v in (("delimiter", desc.delimiter), ("has_header", desc.has_header))}
        if desc.is_wavelength_header and desc.axis_range:
            plan.axis = {"unit": desc.header_unit, "n": desc.n_cols, "range": list(desc.axis_range), "score": round(desc.confidence.get("header_unit", 0.8), 3)}


def _infer_signal_and_task(spec_dict: dict, base_dir: Path, plan: DatasetPlan) -> None:
    feature_src = next((s for s in spec_dict.get("sources", []) if s.get("role") in ("features", "mixed")), None)
    if feature_src is None:
        return
    inp = feature_src.get("input")
    first = inp[0] if isinstance(inp, list) else inp
    if not isinstance(first, str):
        return
    path = Path(first) if Path(first).is_absolute() else base_dir / first
    if not path.exists():
        return
    try:
        table = read_table(path, _params_from(feature_src))
    except Exception:  # noqa: BLE001 - inference is best-effort
        return
    df = table.df
    # use ONLY the columns marked 'features' (declared roles); a contaminating
    # target/metadata column would wreck signal detection. Fall back to numeric.
    feature_cols = _resolve_role_columns(feature_src, df, "features")
    if not feature_cols:
        feature_cols = [str(c) for c in df.columns if _WL_RE.match(str(c)) or pd.api.types.is_numeric_dtype(df[c])]
    if feature_cols:
        x = coerce_numeric(df[feature_cols])
        wl = None
        if plan.axis and all(_WL_RE.match(c) for c in feature_cols):
            try:
                wl = np.array([float(re.sub(r"(nm|cm-1)$", "", c, flags=re.I)) for c in feature_cols])
                if plan.axis["unit"] == "cm-1":
                    wl = 1e7 / wl
            except ValueError:
                wl = None
        sig, sscore, reason = detect_signal_type(x, wl)
        plan.signal_type = Decision(sig, sscore, [reason], ambiguous=sig == "unknown")
    # task type: from a target column in the feature source, OR a separate targets file
    values, where = _first_target_values(spec_dict, feature_src, df, base_dir)
    if values is not None and values.size:
        task, tscore = detect_task_type(values)
        plan.task_type = Decision(task, tscore, [f"target {where} value distribution"])


def _first_target_values(spec_dict: dict, feature_src: dict, feature_df: pd.DataFrame, base_dir: Path):
    """Find a target column's values: inside the feature source, else a separate targets source."""
    in_feature = _resolve_role_columns(feature_src, feature_df, "targets")
    if in_feature and in_feature[0] in feature_df.columns:
        return pd.to_numeric(feature_df[in_feature[0]], errors="coerce").to_numpy(), f"'{in_feature[0]}'"
    for src in spec_dict.get("sources", []):
        if src.get("id") == feature_src.get("id"):
            continue
        is_targets = src.get("role") == "targets" or any(isinstance(c, dict) and c.get("role") == "targets" for c in (src.get("columns") or []))
        if not is_targets:
            continue
        inp = src.get("input")
        first = inp[0] if isinstance(inp, list) else inp
        if not isinstance(first, str):
            continue
        p = Path(first) if Path(first).is_absolute() else base_dir / first
        if not p.exists():
            continue
        try:
            tdf = read_table(p, _params_from(src)).df
        except Exception:  # noqa: BLE001 - best-effort
            continue
        cols = _resolve_role_columns(src, tdf, "targets")
        col = cols[0] if cols else (str(tdf.columns[0]) if len(tdf.columns) else None)
        if col and col in tdf.columns:
            return pd.to_numeric(tdf[col], errors="coerce").to_numpy(), f"source '{src.get('id')}'"
    return None, ""


def _absolutize_inputs(spec_dict: dict, base_dir: Path, iset) -> None:
    """Rewrite source inputs to absolute paths so the resolved_spec is location-independent."""
    name_to_abs = {it.ref: it.identity for it in iset.items}

    def _abs(name: str) -> str:
        return name_to_abs.get(name) or str(base_dir / name)

    for src in spec_dict.get("sources", []):
        value = src.get("input")
        if isinstance(value, list):
            src["input"] = [_abs(n) for n in value]
        elif isinstance(value, str):
            src["input"] = _abs(value)


def _params_from(src: dict):
    from ..spec.dataset_spec import LoadingParams

    return LoadingParams.from_dict(src.get("params"))


def _resolve_role_columns(src: dict, df: pd.DataFrame, role: str) -> list[str]:
    """Resolve the declared column selectors of a given role to concrete column names."""
    cols_spec = src.get("columns")
    if not cols_spec:
        return []
    from ..materialize.loaders import infer_dtypes
    from ..spec.selectors import parse_selector

    headers = [str(c) for c in df.columns]
    dtypes = infer_dtypes(df)
    entries = cols_spec if isinstance(cols_spec, list) else [{"role": r, "select": s} for r, s in cols_spec.items()]
    out: list[str] = []
    for e in entries:
        if e.get("role") == role:
            try:
                out.extend(headers[i] for i in parse_selector(e["select"]).resolve(headers, dtypes, set()))
            except Exception:  # noqa: BLE001 - best-effort during inference
                continue
    return out
