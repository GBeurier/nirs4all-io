# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Assemble a DatasetSpec into a target-agnostic dataset (Epic 4.1b).

Flow per source: load (loaders) -> merge multi-file input (join engine) ->
split columns by role (selectors). Then, per partition: align feature sources
(multi-source X), join targets/metadata/lookup sources onto the sample rows, and
collect X / y / metadata. The result is an :class:`AssembledDataset` IR that
``to_spectrodataset`` (and, later, ``to_dag_ml_data``) adapt to a concrete target.
Keeping the IR target-agnostic lets the assembly logic be tested without nirs4all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..conventions.engine import get_stem
from ..spec.dataset_spec import AggregateSpec, DatasetSpec, SourceSpec
from ..spec.enums import Cardinality, Coverage, MergeMode, PartitionBy, Role, SourceKind, SpecError
from ..spec.selectors import RestSelector
from .join import concat_features, concat_samples, join_tables, merge_by_key
from .loaders import coerce_numeric, effective_params, encode_categorical, read_table

_FEATURE_PARTITION_DEFAULT = "train"


@dataclass
class SourceTable:
    source_id: str
    df: pd.DataFrame
    roles: dict[str, str]  # column -> role
    key: list[str] | None
    partition: str | None
    kind: str
    join: Any
    modality: str | None
    signal_type: str | None
    header_unit: str
    origins: list[str] | None = None


@dataclass
class PartitionBlock:
    n_samples: int
    X: list[np.ndarray] = field(default_factory=list)
    feature_headers: list[list[str]] = field(default_factory=list)
    header_units: list[str] = field(default_factory=list)
    signal_types: list[str | None] = field(default_factory=list)
    y: np.ndarray | None = None
    y_headers: list[str] = field(default_factory=list)
    y_categorical: dict[str, dict] = field(default_factory=dict)
    metadata: pd.DataFrame | None = None


@dataclass
class AssembledDataset:
    name: str
    task_type: str
    signal_type: str
    n_sources: int
    blocks: dict[str, PartitionBlock] = field(default_factory=dict)
    folds: list[tuple[list[int], list[int]]] = field(default_factory=list)
    repetition: str | None = None
    aggregate: AggregateSpec | None = None
    warnings: list[str] = field(default_factory=list)
    audits: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-source: load + merge + role-split                                       #
# --------------------------------------------------------------------------- #
def _resolve_inputs(source: SourceSpec, base_dir: Path) -> list[Path]:
    import glob as _glob

    raw = source.input
    paths: list[str] = []
    for item in raw if isinstance(raw, list) else [raw]:
        text = str(item)
        if any(c in text for c in "*?["):
            paths.extend(sorted(_glob.glob(str(base_dir / text)) or _glob.glob(text)))
        else:
            p = base_dir / text
            paths.append(str(p if p.exists() else Path(text)))
    return [Path(p) for p in paths]


def _load_source_frame(source: SourceSpec, spec: DatasetSpec, base_dir: Path, audits: list[dict]) -> tuple[pd.DataFrame, str, str | None, list[str] | None]:
    params = effective_params(spec.params, source.params)
    paths = _resolve_inputs(source, base_dir)
    if not paths:
        raise SpecError(f"source '{source.id}': no input files resolved from {source.input!r}")
    tables = [read_table(p, params) for p in paths]
    for t in tables:
        if t.na_report.get("na_detected"):
            audits.append({"source": source.id, "na": t.na_report})
    header_unit = tables[0].header_unit
    signal = params.signal_type.value if params.signal_type else None
    origins: list[str] | None = None
    if len(tables) == 1:
        return tables[0].df, header_unit, signal, None
    frames = [t.df for t in tables]
    # use the convention engine's compound-extension-aware stem so `filename_stem`
    # is consistent (s1.csv.gz -> "s1", matching a reference keyed by stem), not "s1.csv"
    names = [get_stem(p.name) for p in paths]
    if source.merge is MergeMode.CONCAT_SAMPLES:
        df, origins, audit = concat_samples(frames, names)
    elif source.merge is MergeMode.CONCAT_FEATURES:
        df, audit = concat_features(frames, names, key=source.key)
    elif source.merge is MergeMode.BY_KEY:
        if source.key is None:
            raise SpecError(f"source '{source.id}': merge='by_key' requires a 'key'")
        df, audit = merge_by_key(frames, names, key=source.key)
    else:
        raise SpecError(f"source '{source.id}': multi-file input requires a merge mode")
    audits.append({"source": source.id, "merge": audit.operation, "warnings": audit.warnings})
    return df, header_unit, signal, origins


def _join_key_columns(source: SourceSpec) -> set[str]:
    cols: set[str] = set()
    if source.join is not None:
        for k in (source.join.left_on, source.join.right_on):
            if isinstance(k, str):
                cols.add(k)
            elif isinstance(k, list):
                cols.update(k)
    return cols


def _split_roles(source: SourceSpec, df: pd.DataFrame, dtypes: list[str]) -> dict[str, str]:
    """Map each column to a role per E.1 (ordered selectors / whole-source role)."""
    headers = list(df.columns)
    key_cols = set(source.key if isinstance(source.key, list) else [source.key] if source.key else [])
    # join keys are identity, not a role -> exempt from role coverage
    key_cols |= _join_key_columns(source)
    key_cols.add("filename_stem")  # reserved virtual key (vendor-corpus); never a role
    roles: dict[str, str] = {}
    if source.role is not Role.MIXED and not source.columns:
        for col in headers:
            if col not in key_cols:
                roles[col] = source.role.value
        return roles

    assigned: set[int] = set()
    has_rest = any(isinstance(cr.select, RestSelector) for cr in source.columns)
    for cr in source.columns:
        idxs = cr.select.resolve(headers, dtypes, assigned)
        for i in idxs:
            if i in assigned:
                if source.strict_columns:
                    raise SpecError(f"source '{source.id}': column '{headers[i]}' matched by multiple selectors (set strict_columns:false for first-match-wins)")
                continue
            assigned.add(i)
            if headers[i] not in key_cols:
                roles[headers[i]] = cr.role.value
    unmatched = [headers[i] for i in range(len(headers)) if i not in assigned and headers[i] not in key_cols]
    if unmatched and not has_rest:
        raise SpecError(f"source '{source.id}': columns {unmatched} are unassigned and no 'rest' selector is present")
    return roles


def _build_source_table(source: SourceSpec, spec: DatasetSpec, base_dir: Path, audits: list[dict]) -> SourceTable:
    df, header_unit, signal, origins = _load_source_frame(source, spec, base_dir, audits)
    from .loaders import infer_dtypes

    # materialize the `filename_stem` virtual key from concat_samples origins so a
    # vendor-corpus lookup can join spectra to a reference by file stem (E.2).
    if origins is not None and "filename_stem" not in df.columns:
        df = df.copy()
        df["filename_stem"] = origins
    # derive the grouped sample id (e.g. mango_001_a -> mango_001) for vendor replicates
    si = spec.sample_index
    if si.derive_from and si.derive_from in df.columns and isinstance(si.key, str) and si.key not in df.columns:
        df = df.copy()
        derived = df[si.derive_from].astype(str)
        df[si.key] = derived.str.replace(si.derive_pattern, "", regex=True) if si.derive_pattern else derived
    roles = _split_roles(source, df, infer_dtypes(df))
    if isinstance(si.key, str):
        roles.pop(si.key, None)  # the sample identity column is never a feature/target/metadata role
    key = source.key if isinstance(source.key, list) else [source.key] if source.key else None
    partition = source.partition.value if source.partition else None
    return SourceTable(
        source_id=source.id,
        df=df,
        roles=roles,
        key=key,
        partition=partition,
        kind=source.kind.value,
        join=source.join,
        modality=source.modality.value if source.modality else None,
        signal_type=signal or (spec.signal_type.value if spec.signal_type.value != "auto" else None),
        header_unit=header_unit,
        origins=origins,
    )


# --------------------------------------------------------------------------- #
# Partition splitting (PartitionAssigner subset: column / percentage)         #
# --------------------------------------------------------------------------- #
def _split_partition_mask(df: pd.DataFrame, spec: DatasetSpec) -> dict[str, np.ndarray]:
    """Return a {partition -> boolean row mask} for a single combined frame."""
    p = spec.partitions
    n = len(df)
    if p is None or p.by is None:
        return {"train": np.ones(n, dtype=bool)}
    if p.by is PartitionBy.COLUMN:
        if p.column not in df.columns:
            raise SpecError(f"partitions.column '{p.column}' not found in the assembled data (columns: {list(df.columns)})")
        col = df[p.column].astype(str).str.lower()
        train_vals = {str(v).lower() for v in p.train_values}
        test_vals = {str(v).lower() for v in p.test_values}
        predict_vals = {str(v).lower() for v in p.predict_values}
        known = train_vals | test_vals | predict_vals
        train_mask = col.isin(train_vals).to_numpy()
        test_mask = col.isin(test_vals).to_numpy()
        unknown_mask = (~col.isin(known)).to_numpy()
        if unknown_mask.any():
            from ..spec.enums import UnknownPolicy

            if p.unknown_policy is UnknownPolicy.ERROR:
                raise SpecError(f"partitions: unknown values in '{p.column}': {sorted(set(col[unknown_mask]))}")
            if p.unknown_policy is UnknownPolicy.TRAIN:
                train_mask = train_mask | unknown_mask
            elif p.unknown_policy is UnknownPolicy.TEST:
                test_mask = test_mask | unknown_mask
            # DROP: leave unknown rows out of every partition
        masks = {"train": train_mask, "test": test_mask}
        if predict_vals:
            masks["predict"] = col.isin(predict_vals).to_numpy()
        return {k: v for k, v in masks.items() if v.any()}
    if p.by is PartitionBy.PERCENTAGE:
        idx = np.arange(n)
        if p.shuffle:
            rng = np.random.RandomState(p.random_state)
            rng.shuffle(idx)
        train_frac = _parse_pct(p.train, n)
        train_idx = idx[:train_frac]
        test_idx = idx[train_frac:]
        m_train = np.zeros(n, dtype=bool)
        m_test = np.zeros(n, dtype=bool)
        m_train[train_idx] = True
        m_test[test_idx] = True
        return {"train": m_train, "test": m_test}
    raise SpecError(f"partitions.by={p.by.value} is not yet supported in the MVP assembler")


def _parse_pct(spec_val: Any, n: int) -> int:
    if isinstance(spec_val, str) and spec_val.endswith("%"):
        return int(n * float(spec_val[:-1]) / 100)
    if isinstance(spec_val, (int, float)):
        return int(spec_val if spec_val > 1 else n * spec_val)
    return int(n * 0.8)


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def assemble(spec: DatasetSpec, base_dir: str | Path = ".") -> AssembledDataset:
    """Load, role-split, join and partition ``spec`` into an AssembledDataset."""
    base_dir = Path(base_dir)
    audits: list[dict] = []
    tables = {s.id: _build_source_table(s, spec, base_dir, audits) for s in spec.sources}

    # group sources by partition; sources without a partition are single-partition ("train" unless split)
    partitions = sorted({t.partition for t in tables.values() if t.partition} - {None})
    assembled = AssembledDataset(
        name=spec.name or "dataset",
        task_type=spec.task_type.value,
        signal_type=spec.signal_type.value,
        n_sources=sum(1 for t in tables.values() if t.kind != SourceKind.LOOKUP.value and _has_role(t, "features")),
        repetition=spec.repetition or (spec.sample_index.repetition_id if spec.sample_index.repetition_id and spec.sample_index.repetition_id != "auto" else None),
        aggregate=spec.aggregate,
    )

    combined_df: pd.DataFrame | None = None
    if partitions:
        for part in partitions:
            part_tables = {sid: t for sid, t in tables.items() if t.partition == part or t.kind == SourceKind.LOOKUP.value}
            assembled.blocks[part], _ = _assemble_block(part_tables, spec, audits, assembled.warnings)
    else:
        block, combined_df = _assemble_block(dict(tables), spec, audits, assembled.warnings)
        if spec.partitions is None:
            assembled.blocks["train"] = block
        else:
            # split on the FINAL (post-join) frame so masks line up with the assembled rows
            for part, mask in _split_partition_mask(combined_df, spec).items():
                assembled.blocks[part] = _slice_block(block, mask)

    assembled.audits = audits
    _attach_folds(spec, base_dir, assembled, combined_df)
    return assembled


def _has_role(table: SourceTable, role: str) -> bool:
    return any(r == role for r in table.roles.values())


def _assemble_block(tables: dict[str, SourceTable], spec: DatasetSpec, audits: list[dict], warnings: list[str]) -> tuple[PartitionBlock, pd.DataFrame]:
    feature_tables = [t for t in tables.values() if t.kind != SourceKind.LOOKUP.value and _has_role(t, "features")]
    if not feature_tables:
        raise SpecError("partition has no feature source")
    primary = feature_tables[0]
    combined = primary.df.copy()

    # join every non-primary source (extra features, targets, metadata, lookups) onto the primary rows
    for t in tables.values():
        if t.source_id == primary.source_id:
            continue
        combined = _join_onto(combined, primary, t, audits, warnings)

    # role -> columns across all joined sources
    role_cols = _collect_roles(combined, tables)

    block = PartitionBlock(n_samples=len(combined))
    # multi-source X: each feature source contributes its own array (aligned to combined rows)
    for ft in feature_tables:
        cols = [c for c in combined.columns if role_cols.get(c) == ("features", ft.source_id)]
        if not cols:
            continue
        block.X.append(coerce_numeric(combined[cols]))
        block.feature_headers.append(cols)
        block.header_units.append(ft.header_unit)
        block.signal_types.append(ft.signal_type)

    target_cols = [c for c, rs in role_cols.items() if rs[0] == "targets"]
    if target_cols:
        y_series, cats = [], {}
        for c in target_cols:
            mode = (spec.params.categorical.value if spec.params.categorical else "auto")
            enc, mapping = encode_categorical(combined[c], mode)
            y_series.append(enc.to_numpy())
            if mapping:
                cats[c] = mapping
        block.y = np.column_stack(y_series)
        block.y_headers = target_cols
        block.y_categorical = cats

    meta_cols = [c for c, rs in role_cols.items() if rs[0] == "metadata"]
    if meta_cols:
        block.metadata = combined[meta_cols].reset_index(drop=True)
    return block, combined.reset_index(drop=True)


def _row_align(combined: pd.DataFrame, t: SourceTable, primary: SourceTable) -> pd.DataFrame:
    if len(t.df) != len(combined):
        raise SpecError(f"source '{t.source_id}' ({len(t.df)} rows) is not row-aligned with '{primary.source_id}' ({len(combined)} rows); supply a join key")
    merged, _ = concat_features([combined.reset_index(drop=True), t.df.reset_index(drop=True)], [primary.source_id, t.source_id])
    return merged


def _as_key_list(on: str | list[str] | None) -> list[str] | None:
    if on is None:
        return None
    return [on] if isinstance(on, str) else list(on)


def _join_onto(combined: pd.DataFrame, primary: SourceTable, t: SourceTable, audits: list[dict], warnings: list[str]) -> pd.DataFrame:
    if t.join is not None:
        j = t.join
        left_on = _as_key_list(j.left_on)  # left keys reference the accumulating combined frame
        right_on = _as_key_list(j.right_on) or _as_key_list(t.key)
        if left_on is not None and right_on is not None:
            missing_l = [c for c in left_on if c not in combined.columns]
            missing_r = [c for c in right_on if c not in t.df.columns]
            if missing_l or missing_r:
                # keys were declared but are absent -> hard error, never a silent row-align (Codex)
                raise SpecError(f"source '{t.source_id}': join keys not found (left missing {missing_l} in assembled data, right missing {missing_r} in '{t.source_id}')")
            out, audit = join_tables(combined, t.df, left_on=left_on, right_on=right_on, cardinality=j.cardinality, coverage=j.coverage, left_name=j.left or primary.source_id, right_name=t.source_id)
            audits.append({"join": audit.operation, "dropped": len(audit.dropped_rows), "warnings": audit.warnings})
            warnings.extend(audit.warnings)
            return out
        if j.cardinality is Cardinality.ONE_TO_ONE:
            return _row_align(combined, t, primary)  # 1:1 with no declared keys -> row order
        raise SpecError(f"source '{t.source_id}': {j.cardinality.value} join needs explicit left_on/right_on")
    # no join spec: align by a shared key (1:1) if both declare one, else row order
    lkeys, rkeys = _as_key_list(primary.key), _as_key_list(t.key)
    if lkeys and rkeys and all(c in combined.columns for c in lkeys) and all(c in t.df.columns for c in rkeys):
        out, audit = join_tables(combined, t.df, left_on=lkeys, right_on=rkeys, cardinality=Cardinality.ONE_TO_ONE, coverage=Coverage.COMPLETE, left_name=primary.source_id, right_name=t.source_id)
        warnings.extend(audit.warnings)
        return out
    return _row_align(combined, t, primary)


def _collect_roles(combined: pd.DataFrame, tables: dict[str, SourceTable]) -> dict[str, tuple[str, str]]:
    """Map each combined column -> (role, owning_source_id)."""
    role_cols: dict[str, tuple[str, str]] = {}
    for t in tables.values():
        for col, role in t.roles.items():
            if col in combined.columns and col not in role_cols:
                role_cols[col] = (role, t.source_id)
            else:
                ns = f"{col}__{t.source_id}"
                if ns in combined.columns:
                    role_cols[ns] = (role, t.source_id)
    return role_cols


def _slice_block(block: PartitionBlock, mask: np.ndarray) -> PartitionBlock:
    out = PartitionBlock(n_samples=int(mask.sum()))
    out.X = [x[mask] for x in block.X]
    out.feature_headers = list(block.feature_headers)
    out.header_units = list(block.header_units)
    out.signal_types = list(block.signal_types)
    out.y = block.y[mask] if block.y is not None else None
    out.y_headers = list(block.y_headers)
    out.y_categorical = dict(block.y_categorical)
    out.metadata = block.metadata.iloc[mask].reset_index(drop=True) if block.metadata is not None else None
    return out


def _attach_folds(spec: DatasetSpec, base_dir: Path, assembled: AssembledDataset, combined_df: pd.DataFrame | None = None) -> None:
    if spec.folds is None:
        return
    if spec.folds.inline:
        assembled.folds = [(list(f.get("train", [])), list(f.get("val", f.get("test", [])))) for f in spec.folds.inline]
    elif spec.folds.column:
        # each distinct value of the fold column defines one fold: val = its rows, train = rest
        if combined_df is None:
            raise SpecError(f"folds.column='{spec.folds.column}' requires a single combined frame (it does not apply with explicit per-source partitions)")
        if spec.folds.column not in combined_df.columns:
            raise SpecError(f"folds.column='{spec.folds.column}' not found in the assembled data (columns: {list(combined_df.columns)})")
        col = combined_df[spec.folds.column]
        all_idx = list(range(len(col)))
        folds: list[tuple[list[int], list[int]]] = []
        for value in sorted(col.dropna().unique(), key=lambda v: str(v)):
            val_set = set(col.index[col == value].tolist())
            val_idx = sorted(val_set)
            train_idx = [i for i in all_idx if i not in val_set]
            folds.append((train_idx, val_idx))
        if not folds:
            raise SpecError(f"folds.column='{spec.folds.column}' has no non-NaN values")
        assembled.folds = folds
    elif spec.folds.file:
        from .folds import parse_fold_file

        assembled.folds = parse_fold_file(base_dir / spec.folds.file, spec.folds.format.value)
