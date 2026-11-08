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

from ..conventions.engine import get_stem, match_items
from ..conventions.profiles import resolve_profiles
from ..conventions.to_spec import assignments_to_spec_dict
from ..materialize.loaders import coerce_numeric, discovery_table, effective_params, read_table
from ..resolve import resolve
from ..resolve.resolver import InputItem, _absolutize_file_references, _file_reference_table
from ..spec import DatasetSpec
from ..spec.dataset_spec import LoadingParams
from ..spec.enums import Role
from .describe import FileDescription, describe
from .detectors import detect_signal_type, detect_task_type
from .identity import classify_metadata_relation, coverage_audit, detect_replicate_grouping, detect_sample_id, key_sources_by_id, repetition_profile
from .plan import DatasetPlan, Decision

_WL_RE = re.compile(r"^\d+(\.\d+)?(nm|cm-1)?$", re.IGNORECASE)
_DEFAULT_CONVENTIONS = ["nirs4all-classic"]


def infer(inp: object, *, conventions: list[str] | None = None, hints: dict | None = None) -> DatasetPlan:
    """Inspect ``inp`` and propose a confidence-scored :class:`DatasetPlan`."""
    if hints is not None and not isinstance(hints, dict):
        raise TypeError("hints must be a mapping or None")
    if hints:
        raise ValueError("Non-empty inference hints are not implemented; edit the resolved_spec explicitly")
    iset = resolve(inp)
    profiles = resolve_profiles(conventions or _DEFAULT_CONVENTIONS)
    result = match_items(iset.names, profiles)
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

    refs = _file_reference_table(iset, spec_dict)
    _enrich_params(spec_dict, plan, refs)
    _infer_identity_and_alignment(spec_dict, plan, refs)  # detect sample id + audit y/metadata coverage
    _infer_signal_and_task(spec_dict, plan, refs)
    # a proposed aggregation over repetitions uses 'vote' for classification targets
    if plan.task_type and plan.task_type.value in ("binary", "multiclass") and isinstance(spec_dict.get("aggregate"), dict):
        spec_dict["aggregate"]["method"] = "vote"
    _absolutize_file_references(spec_dict, refs)  # so load(plan.accept()) works anywhere

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


def _select_description_params(desc: FileDescription, params: dict, single: bool, global_params: dict | None = None) -> bool:
    if desc.delimiter_candidates:
        return False
    params.setdefault("delimiter", desc.delimiter)
    effective = effective_params(LoadingParams.from_dict(global_params), LoadingParams.from_dict(params))
    if single and desc.decimal_separator == "," and (effective.delimiter or ";") == desc.delimiter and not effective.decimal_separator:
        params.setdefault("decimal_separator", ",")
    return True


def _record_description(desc: FileDescription, params: dict, source_id: str, filename: str, plan: DatasetPlan) -> None:
    resolved = not desc.delimiter_candidates
    entry = {"delimiter": {"value": desc.delimiter if resolved else None, "score": round(desc.confidence.get("delimiter", 0.5), 3)}, "has_header": {"value": desc.has_header, "score": round(desc.confidence.get("has_header", 0.5), 3)}}
    if resolved and params.get("decimal_separator") == ",":
        entry["decimal_separator"] = {"value": ",", "score": round(desc.confidence.get("decimal_separator", 0.5), 3)}
    plan.params[filename] = entry
    if not resolved:
        warning = f"source '{source_id}': file '{filename}' delimiter unresolved: candidates {list(desc.delimiter_candidates)!r}; declare params.delimiter (and params.decimal_separator if ',' is the decimal mark)"
        if warning not in plan.warnings:
            plan.warnings.append(warning)


def _infer_single_file(path: str, plan: DatasetPlan) -> dict:
    from ..spec.dataset_spec import LoadingParams

    desc = describe(path)
    # Read WITH a header using the detected delimiter so the inferred name-based
    # column roles match what load() will read. Like nirs4all's loader, default
    # has_header=True for tabular data (the speculative no-header verdict on a
    # numeric wavelength header is unreliable).
    params: dict = {"has_header": True}
    _select_description_params(desc, params, True)
    table = read_table(path, LoadingParams.from_dict(params))
    _record_description(desc, params, "data", Path(path).name, plan)
    desc.column_names = list(table.df.columns)
    guesses = infer_column_roles(desc, table.df)
    # detect a sample-id column; it is the identity key, not a feature/target/metadata role
    id_guess = detect_sample_id(table.df)
    id_col = id_guess.column if id_guess else None
    if id_guess:
        plan.identity = Decision(id_guess.column, id_guess.score, id_guess.evidence, ambiguous=not id_guess.unique)
        for g in guesses:
            if g["col"] == id_col:
                g["role"], g["evidence"] = "id", ["detected sample-id column"]
    plan.columns = [{"ref": Path(path).name, "column_roles": guesses}]
    columns: list[dict] = []
    for role in ("features", "targets", "metadata"):
        cols = [g["col"] for g in guesses if g["role"] == role and g["col"] != id_col]
        if cols:
            columns.append({"role": role, "select": cols})
    source: dict = {"id": "data", "role": "mixed", "input": path, "columns": columns, "params": params}
    spec: dict = {"name": Path(path).stem, "sources": [source]}
    if id_col:
        source["key"] = id_col
        spec["sample_index"] = {"by": "id", "key": id_col}
        _apply_repetition(spec, plan, id_col, table.df[id_col])
    return spec


def _enrich_params(spec_dict: dict, plan: DatasetPlan, refs: dict[str, InputItem]) -> None:
    for src in spec_dict.get("sources", []):
        inp = src.get("input")
        first = inp[0] if isinstance(inp, list) else inp
        if not isinstance(first, str):
            continue
        path = Path(refs[first].identity)
        if not path.exists() or refs[first].extension not in (".csv", ".tsv", ".txt"):
            continue
        desc = describe(path)
        params = src.setdefault("params", {})
        _select_description_params(desc, params, not isinstance(inp, list) or len(inp) == 1, spec_dict.get("params"))
        # only assert a header when detected positively; never speculatively set
        # has_header=False (spectral CSVs default to having a header, like nirs4all)
        if desc.has_header:
            params.setdefault("has_header", True)
        if src.get("role") in ("features", "mixed") and desc.header_unit in ("nm", "cm-1"):
            params.setdefault("header_unit", desc.header_unit)
        _record_description(desc, params, src["id"], Path(first).name, plan)
        if desc.is_wavelength_header and desc.axis_range:
            plan.axis = {"unit": desc.header_unit, "n": desc.n_cols, "range": list(desc.axis_range), "score": round(desc.confidence.get("header_unit", 0.8), 3)}


def _infer_signal_and_task(spec_dict: dict, plan: DatasetPlan, refs: dict[str, InputItem]) -> None:
    feature_src = next((s for s in spec_dict.get("sources", []) if s.get("role") in ("features", "mixed")), None)
    if feature_src is None:
        return
    inp = feature_src.get("input")
    first = inp[0] if isinstance(inp, list) else inp
    if not isinstance(first, str):
        return
    path = Path(refs[first].identity)
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
    values, where = _first_target_values(spec_dict, feature_src, df, refs)
    if values is not None and values.size:
        task, tscore = detect_task_type(values)
        plan.task_type = Decision(task, tscore, [f"target {where} value distribution"])


def _first_target_values(spec_dict: dict, feature_src: dict, feature_df: pd.DataFrame, refs: dict[str, InputItem]):
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
        p = Path(refs[first].identity)
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


def _params_from(src: dict):
    from ..spec.dataset_spec import LoadingParams

    return LoadingParams.from_dict(src.get("params"))


def _source_paths(src: dict, refs: dict[str, InputItem]) -> list[Path]:
    value = src.get("input")
    # A canonical identity or exact logical name must never be interpreted as a pattern.
    return [Path(refs[item].identity) for item in (value if isinstance(value, list) else [value]) if isinstance(item, str)]


def _read_source_df(src: dict, refs: dict[str, InputItem]) -> pd.DataFrame | None:
    """Read + concat ALL of a source's input files for id detection / coverage.

    A multi-file source also gets a per-row `filename_stem` (so vendor corpora,
    one file per sample, expose a usable identity).
    """
    paths = _source_paths(src, refs)
    if not paths:
        return None
    params = _params_from(src)
    frames, stems = [], []
    for p in paths:
        try:
            if p.suffix.lower() in {".csv", ".tsv", ".txt"}:
                preview = discovery_table(p, params)
                if preview is None:
                    return None
                df = preview[0]
            else:
                # Preserve the existing Formats/compressed-reader delegation;
                # the new bounded CSV proposal never parses vendor bytes.
                df = read_table(p, params).df
        except Exception:  # noqa: BLE001 - best-effort
            return None
        frames.append(df)
        stems.extend([get_stem(p.name)] * len(df))
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    out["filename_stem"] = stems
    return out


def _infer_identity_and_alignment(spec_dict: dict, plan: DatasetPlan, refs: dict[str, InputItem]) -> None:
    """Detect a sample-id column; key the sources by it; audit y/metadata coverage."""
    if plan.identity is not None:  # already handled (single combined file)
        return
    sources = spec_dict.get("sources", [])
    feature_src = next((s for s in sources if s.get("role") in ("features", "mixed") and s.get("kind") != "lookup"), None)
    if feature_src is None:
        return
    fdf = _read_source_df(feature_src, refs)
    if fdf is None:
        return
    guess = detect_sample_id(fdf)
    if guess is not None:
        id_col, x_ids = guess.column, fdf[guess.column]
        plan.identity = Decision(id_col, guess.score, list(guess.evidence), ambiguous=not guess.unique)
        spec_dict["sample_index"] = {"by": "id", "key": id_col}
        _apply_repetition(spec_dict, plan, id_col, fdf[id_col])
    elif "filename_stem" in fdf.columns:  # multi-file X with no id column
        stems = [str(v) for v in fdf["filename_stem"].tolist()]
        grouping = detect_replicate_grouping(stems)
        if grouping is not None:  # vendor replicates: mango_001_a/_b/_c -> sample mango_001
            id_col = "sample_id"
            plan.identity = Decision(id_col, 0.7, [*grouping.evidence, "derived from filename_stem"], ambiguous=False)
            spec_dict["sample_index"] = {"by": "id", "key": id_col, "repetition_id": "filename_stem", "derive": {"from": "filename_stem", "strip_suffix": grouping.pattern}}
            spec_dict["repetition"] = id_col
            spec_dict.setdefault("aggregate", {"by": id_col, "method": "median"})
            plan.recommendations.append(f"replicate files detected -> group into '{id_col}' (strip /{grouping.pattern}/) + repetition + aggregate")
            x_ids = pd.Series(grouping.sample_ids, name=id_col)
        else:  # one file per sample
            id_col, x_ids = "filename_stem", fdf["filename_stem"]
            plan.identity = Decision(id_col, 0.7, ["one file per sample -> identity = filename stem"], ambiguous=not bool(fdf["filename_stem"].is_unique))
            spec_dict["sample_index"] = {"by": "id", "key": id_col}
    else:
        key_sources_by_id(spec_dict, refs, plan)
        return  # id-free groups retain convention row order

    if id_col == "filename_stem" or spec_dict.get("sample_index", {}).get("derive"):
        feature_src.setdefault("key", id_col)
        for s in sources:
            if s.get("role") in ("features", "mixed") and s.get("kind") != "lookup":
                sdf_feat = fdf if s.get("id") == feature_src.get("id") else _read_source_df(s, refs)
                if sdf_feat is not None and id_col in sdf_feat.columns:
                    s.setdefault("key", id_col)
                continue
            sdf = _read_source_df(s, refs)
            if sdf is None:
                continue
            if s.get("role") == "targets":
                if id_col in sdf.columns:
                    s["key"] = id_col
                    _key_join(s, id_col, "1:1")
                    audit = coverage_audit(x_ids, sdf[id_col], other_name=s.get("id", "targets"))
                    audit["role"] = "targets"
                    plan.alignment.append(audit)
                    if audit["n_missing"]:
                        plan.warnings.append(f"{audit['n_missing']} sample(s) have NO target in '{s.get('id')}' (e.g. {audit['missing'][:3]})")
            elif s.get("role") == "metadata" or s.get("kind") == "lookup":
                rel = classify_metadata_relation(fdf, sdf, id_col)
                if rel is None:
                    continue
                card, key, ev = rel
                s["key"] = key
                if card == "m:1":
                    s["kind"] = "lookup"
                _key_join(s, key, card)
                if key in fdf.columns:
                    audit = coverage_audit(fdf[key], sdf[key], other_name=s.get("id", "metadata"))
                    audit.update({"role": "metadata", "relation": card, "evidence": ev})
                    plan.alignment.append(audit)
                    if audit["n_missing"]:
                        plan.warnings.append(f"{audit['n_missing']} {key}(s) have NO metadata in '{s.get('id')}' (e.g. {audit['missing'][:3]})")
        return

    key_sources_by_id(spec_dict, refs, plan)
    # Keep the existing infer-only shared-dimension classifier.
    for s in sources:
        sdf = _read_source_df(s, refs)
        if sdf is None:
            continue
        if s.get("role") == "metadata" or s.get("kind") == "lookup":
            rel = classify_metadata_relation(fdf, sdf, id_col)
            if rel is None:
                continue
            card, key, ev = rel
            if card == "1:1" and key == id_col and repetition_profile(fdf[id_col]).is_repetition:
                card = "m:1"
            s["key"] = key
            if card == "m:1":
                s["kind"] = "lookup"
            _key_join(s, key, card)
            if key in fdf.columns:
                audit = coverage_audit(fdf[key], sdf[key], other_name=s.get("id", "metadata"))
                audit.update({"role": "metadata", "relation": card, "evidence": ev})
                plan.alignment.append(audit)
                if audit["n_missing"]:
                    plan.warnings.append(f"{audit['n_missing']} {key}(s) have NO metadata in '{s.get('id')}' (e.g. {audit['missing'][:3]})")


def _apply_repetition(spec_dict: dict, plan: DatasetPlan, id_col: str, ids: pd.Series) -> None:
    """Non-unique id -> repetition + aggregation (systematic), else a duplicate-id warning."""
    prof = repetition_profile(ids)
    if prof.is_repetition:
        spec_dict["repetition"] = id_col
        spec_dict.setdefault("sample_index", {})["repetition_id"] = id_col
        spec_dict.setdefault("aggregate", {"by": id_col, "method": "median"})
        if plan.identity is not None:
            plan.identity.evidence.append(f"repeated measurements: {prof.n_rows} rows / {prof.n_unique} samples (avg {prof.avg_reps} reps)")
        plan.recommendations.append(f"repeated measurements detected -> set repetition='{id_col}' + aggregate by it (method default 'median')")
    elif prof.n_unique < prof.n_rows:
        plan.warnings.append(f"sample id '{id_col}' has {prof.n_rows - prof.n_unique} duplicate value(s) (not unique per row, not a clean repetition pattern)")


def _key_join(src: dict, key: str, how: str) -> None:
    join = src.get("join")
    if isinstance(join, dict):
        src["join"] = {**join, "on": key, "how": how}
    elif join is None and src.get("role") != "features":
        pass  # leave row-order if no join was declared


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
