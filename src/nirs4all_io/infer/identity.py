# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Sample-identity inference + cross-file alignment audit (Epic 3.3, R-IDX).

Detects a **sample-id column** (so the loader can align by id, not row order),
classifies a metadata source as **per-sample (1:1)** vs **shared/dimension (m:1)**,
and audits coverage: does every sample have a target and metadata? Duplicate ids?
These feed the inference engine's resolved spec (keys + joins) and plan warnings.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..spec import DatasetSpec
    from .plan import DatasetPlan

# id-like column names (case-insensitive): sample_id, id, name, code, ref, key, *_id, ...
_ID_NAME_RE = re.compile(r"^(sample[_ ]?id|sampleid|sample|id|name|code|ref(erence)?|key|index|spectrum[_ ]?id|scan[_ ]?id|.*_id|id_.*)$", re.IGNORECASE)
_WL_RE = re.compile(r"^\d+(\.\d+)?(nm|cm-1)?$", re.IGNORECASE)


@dataclass
class SampleIdGuess:
    column: str
    score: float
    unique: bool
    evidence: list[str] = field(default_factory=list)


@dataclass
class RepetitionProfile:
    n_rows: int
    n_unique: int
    avg_reps: float
    is_repetition: bool  # systematic repeats (vs sporadic duplicates)


# Replicate-suffix patterns on a filename stem (mango_001_a / sample-rep2 / scan_3 / (2)).
# A bare trailing number (_1, _2) is deliberately EXCLUDED: it is usually the sample
# number, not a replicate index, and would over-group distinct samples.
_REPLICATE_SUFFIX_PATTERNS = [
    r"[_\-. ]rep(?:licate)?[_\-. ]?\d+$",
    r"[_\-. ]r\d+$",
    r"[_\-. ]scan[_\-. ]?\d+$",
    r"[_\-. ]dup(?:licate)?\d*$",
    r"\s*\(\d+\)$",
    r"[_\-. ][a-z]$",  # single trailing letter: mango_001_a / _b / _c
]


@dataclass
class ReplicateGrouping:
    pattern: str
    sample_ids: list[str]  # the grouped sample id per file (suffix stripped)
    n_files: int
    n_samples: int
    avg_reps: float
    evidence: list[str] = field(default_factory=list)


def detect_replicate_grouping(stems: list[str]) -> ReplicateGrouping | None:
    """Detect replicate files of the same sample (mango_001_a/_b/_c -> mango_001).

    Tries replicate-suffix patterns; accepts one only if a majority of stems carry
    it AND stripping it merges files into fewer samples with avg >= 1.5 reps.
    """
    n = len(stems)
    if n < 2:
        return None
    best: tuple[float, ReplicateGrouping] | None = None
    for pat in _REPLICATE_SUFFIX_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        stripped = [rx.sub("", s) for s in stems]
        matched = sum(1 for s, st in zip(stripped, stems, strict=True) if s != st)
        if matched < n * 0.5:  # most files must carry the suffix
            continue
        n_samples = len(set(stripped))
        if n_samples >= n:  # nothing merged
            continue
        avg = n / n_samples
        if avg < 1.5:  # not a real repetition pattern
            continue
        score = matched / n
        if best is None or score > best[0]:
            ev = [f"replicate suffix /{pat}/ on {matched}/{n} files -> {n_samples} sample(s), avg {round(avg, 2)} reps"]
            best = (score, ReplicateGrouping(pat, stripped, n, n_samples, round(avg, 2), ev))
    return best[1] if best else None


def repetition_profile(ids: pd.Series) -> RepetitionProfile:
    """Decide whether a non-unique id column is *repeated measurements* (avg >=1.5
    rows/id) or just sporadic duplicate ids (a data-quality issue)."""
    n = len(ids)
    u = int(ids.nunique(dropna=False)) or 1
    avg = n / u
    return RepetitionProfile(n_rows=n, n_unique=u, avg_reps=round(avg, 2), is_repetition=(u < n and avg >= 1.5))


def detect_sample_id(df: pd.DataFrame) -> SampleIdGuess | None:
    """Pick the most likely sample-id column, or None. Requires score >= 0.5."""
    n = len(df)
    if n == 0:
        return None
    best: SampleIdGuess | None = None
    for col in df.columns:
        name = str(col)
        if _WL_RE.match(name):  # a wavelength column is never the id
            continue
        s = df[col]
        nunique = int(s.nunique(dropna=False))
        unique = nunique == n
        score, ev = 0.0, []
        if _ID_NAME_RE.match(name):
            score += 0.5
            ev.append(f"id-like name '{name}'")
        if unique:
            score += 0.35
            ev.append("unique per row")
        elif nunique >= 0.9 * n:
            score += 0.15
            ev.append("near-unique values")
        if pd.api.types.is_float_dtype(s):
            score -= 0.25  # continuous float is unlikely to be an identifier
        elif not pd.api.types.is_numeric_dtype(s):
            score += 0.1
            ev.append("string identifier")
        if score > 0 and (best is None or score > best.score):
            best = SampleIdGuess(name, min(score, 1.0), unique, ev)
    return best if best and best.score >= 0.5 else None


def coverage_audit(x_ids: pd.Series, other_ids: pd.Series, *, other_name: str) -> dict:
    """Does every sample id have a match in `other_ids`? Report missing/extra/duplicates."""
    x_list = [str(v) for v in x_ids.tolist()]
    o_set = {str(v) for v in other_ids.tolist()}
    x_set = set(x_list)
    missing = sorted(x_set - o_set)
    extra = sorted(o_set - x_set)
    dup_x = sorted(k for k, c in Counter(x_list).items() if c > 1)
    return {
        "against": other_name,
        "n_samples": len(x_set),
        "n_missing": len(missing),
        "missing": missing[:20],
        "n_extra": len(extra),
        "extra": extra[:20],
        "duplicate_sample_ids": dup_x[:20],
    }


def classify_metadata_relation(x_df: pd.DataFrame, meta_df: pd.DataFrame, x_id: str) -> tuple[str, str, list[str]] | None:
    """Classify a metadata source vs the samples: ('1:1'|'m:1', key, evidence).

    1:1 = per-sample metadata keyed by the sample id; m:1 = a shared dimension
    table keyed by a grouping column present in X (fewer rows, broadcast).
    """
    # per-sample: metadata carries the sample id, one row per sample
    if x_id in meta_df.columns and meta_df[x_id].is_unique and len(meta_df) >= len(x_df) * 0.5:
        return "1:1", x_id, [f"keyed by sample id '{x_id}' (one row per sample)"]
    # shared/dimension: a column present in both, unique in metadata, fewer rows than samples
    for col in meta_df.columns:
        if col == x_id or col not in x_df.columns:
            continue
        if meta_df[col].is_unique and len(meta_df) < len(x_df):
            return "m:1", col, [f"shared dimension table keyed by '{col}' ({len(meta_df)} rows broadcast to {len(x_df)} samples)"]
    return None


def source_preview(source: dict, refs: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Project every input through the same bounded owner reader, or none."""
    from pathlib import Path

    from ..materialize.loaders import discovery_table
    from ..spec.dataset_spec import LoadingParams

    inputs = source["input"] if isinstance(source["input"], list) else [source["input"]]
    frames, raw_frames = [], []
    for name in inputs:
        item = refs[name]
        if item.extension not in {".csv", ".tsv", ".txt"}:
            return None
        table = discovery_table(Path(item.identity), LoadingParams.from_dict(source.get("params")))
        if table is None:
            return None
        frames.append(table[0])
        raw_frames.append(table[1])
    if not frames or any(list(frame.columns) != list(frames[0].columns) for frame in frames):
        return None
    return pd.concat(frames, ignore_index=True), pd.concat(raw_frames, ignore_index=True)


def key_sources_by_id(spec: dict, refs: dict, plan: DatasetPlan | None = None) -> None:
    """Propose generated, partition-local keys without comparing or joining ids.

    Missing operands are deliberately provisional. Full typed cardinality and
    coverage are enforced by the existing materializer after user edits.
    """
    from pathlib import Path

    from ..spec.dataset_spec import LoadingParams
    from .describe import _detect_decimal, proposed_decimal
    from .plan import Decision

    sources = spec.get("sources", [])
    # Vendor/declared lookup relations already have a deliberate identity policy.
    if any(source.get("kind") == "lookup" for source in sources):
        return
    partitions = list(dict.fromkeys(source.get("partition") for source in sources))
    groups = []
    for partition in partitions:
        group = [source for source in sources if source.get("partition") == partition]
        anchor = next((source for source in group if source.get("role") in ("features", "mixed")), None)
        if anchor is None:
            continue
        previews = {source["id"]: source_preview(source, refs) for source in group}
        if any(value is None for value in previews.values()):
            continue  # Never decide from a partially readable group/source.
        profiles = {name: value for name, value in previews.items() if value is not None}
        guesses = {}
        for source in group:
            frame = profiles[source["id"]][0]
            guess = detect_sample_id(frame) if len(frame.columns) > 1 else None
            if source is not anchor and guess is not None and (not guess.unique or pd.api.types.is_float_dtype(frame[guess.column])):
                guess = None
            if source.get("role") == "metadata" and guess is not None and not any(other.get("role") != "metadata" and guess.column in profiles[other["id"]][0].columns for other in group):
                if plan is not None:
                    name = "null" if partition is None else str(partition)
                    plan.warnings.append(f"partition {name}: metadata label '{guess.column}' in source '{source['id']}' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity")
                guess = None
            guesses[source["id"]] = guess
        guess = guesses[anchor["id"]] or next((value for value in guesses.values() if value is not None), None)
        groups.append((group, anchor, profiles, guess))
    selected = next((guess for _, _, _, guess in groups if guess is not None), None)
    if selected is None:
        # Neutral discovery can recover a comma label already converted by a
        # provisional decimal. It never participates in identity selection.
        for group, anchor, profiles, _ in groups:
            partition = anchor.get("partition")
            for source in group:
                inputs = source["input"] if isinstance(source["input"], list) else [source["input"]]
                if source.get("role") != "metadata" or len(inputs) != 1 or source.get("params", {}).get("decimal_separator") != ",":
                    continue
                neutral = source_preview({**source, "params": {**source["params"], "decimal_separator": None}}, refs)
                if neutral is None or len(neutral[0].columns) <= 1:
                    continue
                label = detect_sample_id(neutral[0])
                if label is None or not label.unique or pd.api.types.is_float_dtype(neutral[0][label.column]):
                    continue
                if any(other.get("role") != "metadata" and label.column in profiles[other["id"]][0].columns for other in group):
                    continue
                params = source["params"]
                decimal = proposed_decimal(Path(refs[inputs[0]].identity), LoadingParams.from_dict(params), key=label.column)
                params.pop("decimal_separator", None)
                if decimal == ",":
                    params["decimal_separator"] = decimal
                if plan is not None:
                    name = "null" if partition is None else str(partition)
                    warning = f"partition {name}: metadata label '{label.column}' in source '{source['id']}' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity"
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
                    entry = plan.params.get(Path(inputs[0]).name, {})
                    entry.pop("decimal_separator", None)
                    if decimal == ",":
                        raw = profiles[source["id"]][1].drop(columns=[label.column], errors="ignore")
                        _, score = _detect_decimal([list(raw.columns), *raw.values.tolist()])
                        entry["decimal_separator"] = {"value": ",", "score": round(score, 3)}
                        if pd.api.types.is_numeric_dtype(profiles[source["id"]][0][label.column]):
                            plan.warnings.append(f"partition {name}: metadata label '{label.column}' in source '{source['id']}' also reads as numbers under decimal ','; source-wide decimals can rewrite or merge its values; declare decimal_separator '.' to keep comma-form values as text in this source")
        return
    key = selected.column
    spec["sample_index"] = {"by": "id", "key": key}
    if plan is not None and plan.identity is None:
        plan.identity = Decision(key, selected.score, list(selected.evidence), ambiguous=not selected.unique)
    for group, anchor, profiles, guess in groups:
        # A single dataset identity cannot truthfully describe an id-free or
        # differently keyed partition. Keep the required real operand visible.
        frame = profiles[anchor["id"]][0]
        profile = repetition_profile(frame[key]) if key in frame else None
        repeated = bool(plan is not None and profile is not None and profile.is_repetition)
        if plan is not None and profile is not None:
            plan.params.setdefault("identity_repetition", {})[anchor["id"]] = {
                "n_rows": profile.n_rows, "n_unique": profile.n_unique,
                "avg_reps": profile.avg_reps, "is_repetition": profile.is_repetition,
            }
        if plan is not None and (guess is None or guess.column != key or any(key not in profiles[s["id"]][0] for s in group)):
            plan.warnings.append(f"partition {anchor.get('partition')!r}: incomplete identity '{key}'; declare the join/identity before loading")
        for source in group:
            source["key"] = key
            if source is not anchor:
                join = source.setdefault("join", {"to": anchor["id"]})
                join.update({"on": key, "how": "m:1" if repeated and source.get("role") in ("targets", "metadata") else "1:1", "coverage": "complete"})
            inputs = source["input"] if isinstance(source["input"], list) else [source["input"]]
            if len(inputs) == 1:
                params = source.setdefault("params", {})
                decimal = proposed_decimal(Path(refs[inputs[0]].identity), LoadingParams.from_dict(params), key=key)
                params.pop("decimal_separator", None)
                if decimal == ",":
                    params["decimal_separator"] = decimal
                if plan is not None:
                    entry = plan.params.get(Path(inputs[0]).name, {})
                    entry.pop("decimal_separator", None)
                    if decimal == ",":
                        raw = profiles[source["id"]][1].drop(columns=[key], errors="ignore")
                        _, score = _detect_decimal([list(raw.columns), *raw.values.tolist()])
                        entry["decimal_separator"] = {"value": ",", "score": round(score, 3)}
            if plan is not None and source.get("role") in ("targets", "metadata") and key in frame and key in profiles[source["id"]][0]:
                audit = coverage_audit(frame[key], profiles[source["id"]][0][key], other_name=source["id"])
                audit["role"] = source.get("role")
                audit["scope"] = "bounded_discovery_prefix"
                audit["complete"] = False
                plan.alignment.append(audit)
                if audit["n_missing"]:
                    plan.warnings.append(f"{audit['n_missing']} sample(s) have NO target in the sampled prefix of '{source['id']}'")


def refuse_generated_identity(spec: DatasetSpec, refs: dict) -> None:
    """Refuse unsafe generated identity at materialization, never in to_spec."""
    from ..materialize.join import JoinError
    from ..materialize.loaders import _COMMA_PAIR

    if spec.sample_index.by.value != "id" or not isinstance(spec.sample_index.key, str):
        return
    key = spec.sample_index.key
    for source in spec.to_dict()["sources"]:
        if source.get("kind") == "lookup":
            continue
        preview = source_preview(source, refs)
        if preview is None:
            continue  # The existing reader owns unresolved dialect refusals.
        frame, raw = preview
        if key not in frame:
            raise JoinError(f"source '{source['id']}': file '{source['input']}' missing identity key '{key}'; declare compatible keys/partitions")
        if source.get("params", {}).get("decimal_separator") == "," and any(_COMMA_PAIR.fullmatch(str(value).strip()) for value in raw[key]):
            raise JoinError(f"source '{source['id']}': file '{source['input']}' key '{key}' contains comma pairs under params.decimal_separator=','; declare an explicit identity/decimal policy")
