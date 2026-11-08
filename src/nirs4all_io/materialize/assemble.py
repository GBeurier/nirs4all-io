# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Assemble a DatasetSpec into a target-agnostic dataset (Epic 4.1b).

Flow per source: load (loaders) -> merge multi-file input (join engine) ->
split columns by role (selectors). Then, per partition: align feature sources
(multi-source X), join targets/metadata/lookup sources onto the sample rows, and
collect X / y / metadata. The result is an :class:`AssembledDataset` IR that
``to_spectrodataset`` and ``DatasetPackage`` adapt on the Python MVP side, while
the Rust bridge crate adapts the same IR to ``dag-ml-data``. Keeping the IR
target-agnostic lets the assembly logic be tested without nirs4all.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import strict_json
from ..conventions.engine import get_stem
from ..spec.dataset_spec import AggregateSpec, ColumnRole, DatasetSpec, NaConfig, SourceSpec, VariationSpec
from ..spec.enums import Cardinality, Coverage, MergeMode, NaPolicy, PartitionBy, Role, SourceKind, SpecError
from ..spec.selectors import AutoSelector, RestSelector
from .join import JoinError, _typed_key, concat_features, concat_samples, join_tables, merge_by_key
from .loaders import (
    _COMMA_PAIR,
    apply_na_policy,
    coerce_numeric,
    effective_params,
    encode_categorical,
    exact_float64,
    read_table,
    refuse_numeric_cells,
    refuse_undeclared_delimiter,
    restore_numeric_cells,
    restore_numeric_keys,
    restore_numeric_metadata,
)
from .numeric import NumericContractError, NumericStorageOptions, PrecisionRecord, effective_coercion, int_to_f32, round_storage, to_ir_f64

_NumericScope = str | dict[str, str | tuple[str, ...]]
_MetadataScope = bool | list[bool] | dict[str, bool]

_FEATURE_PARTITION_DEFAULT = "train"
# Sentinel column carrying the source's original positional row index so that
# variations (loaded once per source, before any join) can be re-aligned to the
# combined frame after joins/reordering. Stripped before exposing to the user.
_ROW_IDX_PREFIX = "__src_row_idx__"

# Version of the JSON-compatible AssembledDataset summaries/full exports emitted
# by the Rust facade. Version 3 carries the exact F64 execution IR: explicit
# dtype/precision/validity blocks and a required ``partition_order``. Earlier
# versions are retired and never read as v3.
ASSEMBLED_DATASET_VERSION = 3


_DECIMAL = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


def feature_axis_name(unit: str) -> str:
    """The inferred axis bucket of a presentation unit (never the raw unit spelling)."""
    text = unit.lower()
    if "nm" in text or "nanomet" in text or "wavelength" in text:
        return "wavelength"
    if "cm-1" in text or "cm^-1" in text or "1/cm" in text or "wavenumber" in text or "cm\u207b\u00b9" in text:
        return "wavenumber"
    return "feature"


def tabular_axis(headers: list[str], header_unit: str, raw_unit: str | None) -> dict[str, Any]:
    """The non-sample axis of a tabular feature source.

    ``unit`` is the raw configured spelling (``None`` when none was configured);
    ``name`` is the inferred bucket of the presentation unit. Coordinates are the
    exact F64 values of genuine finite-decimal file headers; generated index
    headers declare none.
    """
    coordinates = None
    if header_unit != "index" and headers and all(_DECIMAL.fullmatch(header) for header in headers):
        # the header text is the original cell: integer text through the exact-F64 gate (2**54 is exact, 2**53 + 1
        # refuses), decimal text through the checked decimal primitive (``1e-400`` refuses, a true zero keeps its sign)
        cells: list[Any] = []
        for header in headers:
            if _INTEGER.fullmatch(header):
                number = strict_json.check_integer(int(header))
                cells.append(-0.0 if number == 0 and header.startswith("-") else number)
            else:
                cells.append(strict_json.decimal_float(header))
        try:
            values = to_ir_f64(cells).tolist()
        except NumericContractError as error:
            raise SpecError(f"feature header: {error}") from error
        coordinates = values if all(np.isfinite(values)) else None
    return {"name": feature_axis_name(header_unit), "unit": raw_unit, "coordinates": coordinates}


def _row_idx_col(source_id: str) -> str:
    return f"{_ROW_IDX_PREFIX}{source_id}__"


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
    # Pre-loaded variation matrices for this (feature) source: rows align with
    # df's original row order; columns are already aligned to the feature
    # columns (either by name or positionally).
    variations: list[tuple[str, np.ndarray]] = field(default_factory=list)
    # Each variation's OWN declared storage dtype (typed readers only), captured before
    # the F64 widening; ``None`` for an undeclared (CSV/untyped) file. A named processing
    # is an independently loaded input, never assumed to share its base source's dtype.
    variation_dtypes: list[str | None] = field(default_factory=list)
    remove_rows: np.ndarray | None = None
    # Declared storage dtype per original column (typed readers only; see LoadedTable).
    source_dtypes: dict[str, str] = field(default_factory=dict)
    # Raw configured unit spelling, captured before any inferred unit bucket is
    # formed; ``None`` when the spec configured no unit (never a presentation default).
    raw_unit: str | None = None
    numeric_scope: _NumericScope = "."


@dataclass
class PartitionBlock:
    n_samples: int
    source_ids: list[str] = field(default_factory=list)
    X: list[np.ndarray] = field(default_factory=list)
    feature_headers: list[list[str]] = field(default_factory=list)
    header_units: list[str] = field(default_factory=list)
    signal_types: list[str | None] = field(default_factory=list)
    # Per feature source, the list of named preprocessing variants (additional
    # processings on the SpectroDataset side, parallel to X[src_idx]). Empty
    # outer list when no source declares variations.
    processings: list[list[tuple[str, np.ndarray]]] = field(default_factory=list)
    # Declared source dtype per X source / for the numeric target columns / for
    # the weights, captured BEFORE the exact F64 widening (None = undeclared,
    # e.g. CSV or untyped lists). Never inferred from the widened F64 IR.
    x_source_dtypes: list[str | None] = field(default_factory=list)
    # Exact per-source feature metadata, aligned with ``X`` / ``source_ids``:
    # ``feature_names`` holds the semantic original names (``None`` = a genuinely
    # unnamed source; distinct from the presentation ``feature_headers``),
    # ``feature_axes`` the ordered non-sample ``{name, unit, coordinates}`` axes
    # (raw unit spelling, exact F64 coordinates or string labels) and
    # ``representation_ids`` the actual source representation IDs.
    feature_names: list[list[str] | None] = field(default_factory=list)
    feature_axes: list[list[dict[str, Any]]] = field(default_factory=list)
    representation_ids: list[str] = field(default_factory=list)
    y_source_dtype: str | None = None
    weights_source_dtype: str | None = None
    # The ACTUAL selected precision records of an already stored block (set when the IR
    # came out of a package / full wire; empty on a raw IR whose selection is pending).
    # They travel beside the F64 IR so a repackaging retains the exact P and bytes.
    x_precisions: list[PrecisionRecord] = field(default_factory=list)
    # Per source, the actual declared source dtype / selected record of each named processing
    # (aligned with ``processings``); they differ from the base source's when the child file
    # was typed independently.
    processing_source_dtypes: list[list[str | None]] = field(default_factory=list)
    processing_precisions: list[list[PrecisionRecord]] = field(default_factory=list)
    y_precisions: list[PrecisionRecord] = field(default_factory=list)
    weights_precision: PrecisionRecord | None = None
    y: np.ndarray | None = None
    y_headers: list[str] = field(default_factory=list)
    y_categorical: dict[str, dict] = field(default_factory=dict)
    metadata: pd.DataFrame | None = None
    # Per-sample training weights (sklearn-style `sample_weight`), surfaced on
    # the SpectroDataset side as a metadata column.
    weights: np.ndarray | None = None
    weights_header: str | None = None


@dataclass
class AssembledDataset:
    name: str
    task_type: str
    signal_type: str
    n_sources: int
    blocks: dict[str, PartitionBlock] = field(default_factory=dict)
    folds: list[tuple[list[int], list[int]]] = field(default_factory=list)
    fold_provenance: list[dict[str, list[str]]] = field(default_factory=list)
    repetition: str | None = None
    identity: dict[str, object] = field(default_factory=dict)
    aggregate: AggregateSpec | None = None
    warnings: list[str] = field(default_factory=list)
    audits: list[dict] = field(default_factory=list)

    def to_full_dict(self, *, features: Any = None, targets: Any = None, weights: Any = None, consumer: str | None = None) -> dict[str, Any]:
        """Exact assembled v3 full wire (exact dtype/bytes/mask blocks, required ``partition_order``)."""
        from .package import assembled_to_full_dict

        return assembled_to_full_dict(self, features=features, targets=targets, weights=weights, consumer=consumer)

    @classmethod
    def from_full_dict(cls, value: Any) -> AssembledDataset:
        """Strict reader of the exact assembled v3 wire; any other version refuses."""
        from .package import assembled_from_full_dict

        return assembled_from_full_dict(value)


# --------------------------------------------------------------------------- #
# Per-source: load + merge + role-split                                       #
# --------------------------------------------------------------------------- #
def _resolve_input_values(raw: Any, base_dir: Path) -> list[Path]:
    import glob as _glob

    paths: list[Path] = []
    for item in raw if isinstance(raw, list) else [raw]:
        text = str(item)
        scoped = base_dir / text
        if scoped.is_file():
            paths.append(scoped)
            continue
        pattern = any(c in text for c in "*?[")
        if pattern:
            scoped_pattern = text if Path(text).is_absolute() else str(Path(_glob.escape(str(base_dir))) / text)
            matches = sorted(_glob.glob(scoped_pattern))
            if matches:
                paths.extend(Path(p) for p in matches)
                continue
        if scoped.exists():
            paths.append(scoped)
            continue
        direct = Path(text)
        if direct.is_file():
            paths.append(direct)
        elif pattern:
            paths.extend(Path(p) for p in sorted(_glob.glob(text)))
        else:
            paths.append(direct)
    return paths


def _resolve_inputs(source: SourceSpec, base_dir: Path) -> list[Path]:
    return _resolve_input_values(source.input, base_dir)


def _combine_dtypes(values: list[str | None]) -> str | None:
    """One declared dtype for a column group; disagreement (or partial declaration) is ``mixed``."""
    distinct = set(values)
    if len(distinct) > 1:
        return "mixed"
    return next(iter(distinct), None)


def _input_decimal(table: Any) -> str:
    return table.decimal if table.text_tokens else "."


def _numeric_scope(tables: list[Any], columns: list[str]) -> _NumericScope:
    input_decimals = {_input_decimal(table) for table in tables}
    if len(input_decimals) == 1:
        return next(iter(input_decimals))
    scope: dict[str, str | tuple[str, ...]] = {}
    for column in columns:
        declared = {_input_decimal(table) for table in tables if column in table.df.columns}
        if len(declared) == 1:
            scope[column] = next(iter(declared))
        else:
            scope[column] = tuple(separator for table in tables for separator in [_input_decimal(table) if column in table.df.columns else "."] * len(table.df))
    distinct = {value for value in scope.values() if isinstance(value, str)}
    return next(iter(distinct)) if len(distinct) == 1 and all(isinstance(value, str) for value in scope.values()) else scope


def _scope_entry(scope: _NumericScope, column: str) -> str | tuple[str, ...]:
    return scope if isinstance(scope, str) else scope.get(column, ".")


def _source_key_columns(source: SourceSpec, spec: DatasetSpec) -> set[str]:
    keys = set(source.key if isinstance(source.key, list) else [source.key] if source.key else [])
    if source.join is not None:
        keys.update(_as_key_list(source.join.right_on) or [])
    for peer in spec.sources:
        if peer.join is not None:
            keys.update(_as_key_list(peer.join.left_on) or [])
    return keys


def _load_source_frame(
    source: SourceSpec, spec: DatasetSpec, base_dir: Path, audits: list[dict]
) -> tuple[pd.DataFrame, str, str | None, list[str] | None, dict[str, str], pd.DataFrame | None, str | list[str], _MetadataScope, list[int] | None, pd.DataFrame | None, _NumericScope]:
    params = effective_params(spec.params, source.params)
    # The NA policy depends on each column's role, so it is applied once roles
    # are split (`_apply_role_na_policy`); rows are removed only after joining.
    params.na = replace(params.na, policy=NaPolicy.IGNORE)
    paths = _resolve_inputs(source, base_dir)
    if not paths:
        raise SpecError(f"source '{source.id}': no input files resolved from {source.input!r}")
    tables = [read_table(p, params) for p in paths]
    for path, table in zip(paths, tables, strict=True):
        if table.text_tokens:
            refuse_undeclared_delimiter(path, params, source.id)
    for table in tables:
        table.df = restore_numeric_keys(table.df, table.raw, list(_source_key_columns(source, spec)), table.decimal, table.text_tokens)
    for t in tables:
        if t.na_report.get("na_detected"):
            audits.append({"source": source.id, "na": t.na_report})
    header_unit = tables[0].header_unit
    signal = params.signal_type.value if params.signal_type else None
    origins: list[str] | None = None
    if len(tables) == 1:
        return tables[0].df, header_unit, signal, None, dict(tables[0].source_dtypes), tables[0].raw, [_input_decimal(tables[0])], tables[0].text_tokens, None, None, _numeric_scope(tables, list(tables[0].df.columns))
    # use the convention engine's compound-extension-aware stem so `filename_stem`
    # is consistent (s1.csv.gz -> "s1", matching a reference keyed by stem), not "s1.csv"
    names = [get_stem(p.name) for p in paths]
    # A positional marker rides each CSV frame through the EXISTING join machinery, so the original cell text can be
    # re-attached to whatever row the join produced (reordering, unmatched rows, renamed duplicates included).
    carry_text = source.merge in (MergeMode.CONCAT_FEATURES, MergeMode.BY_KEY) and any(t.raw is not None for t in tables)
    frames = [_with_row_marker(t.df, i) if carry_text and t.raw is not None else t.df for i, t in enumerate(tables)]
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
    declared: dict[str, str] = {}
    for column in df.columns:
        combined = _combine_dtypes([t.source_dtypes.get(column) for t in tables if column in t.df.columns])
        if combined is not None:
            declared[column] = combined
    raw: pd.DataFrame | None = None
    metadata_scope: _MetadataScope = False
    column_scope: _NumericScope = "."
    if source.merge is MergeMode.CONCAT_SAMPLES and any(t.raw is not None for t in tables):
        # rows stack in file order, so the original text stacks the same way; a typed (binary) file has no text:
        # its cells keep their genuine decoded values (empty tokens are never inspected)
        pieces = [t.raw if t.raw is not None else pd.DataFrame("", index=range(len(t.df)), columns=t.df.columns) for t in tables]
        stacked = pd.concat(pieces, ignore_index=True)
        raw = stacked if len(stacked) == len(df) else None
        metadata_scope = [table.text_tokens for table in tables]
    elif carry_text:
        df, raw, column_scope, metadata_scope = _carry_text(df, tables, names)
    # the stacked inputs' row counts: a concat_samples column is classified per loaded input (see ``restore_numeric_metadata``)
    row_counts = [len(t.df) for t in tables] if source.merge is MergeMode.CONCAT_SAMPLES else None
    # Every loaded input's cells as that input loaded them (object dtype, so pandas cannot unify a boolean input with an
    # integer one while stacking): the typed cells of a non-numeric input survive ``concat_samples`` unchanged.
    typed_cells = pd.concat([t.df.astype(object) for t in tables], ignore_index=True) if row_counts is not None else None
    if typed_cells is not None and len(typed_cells) != len(df):
        typed_cells = None
    scope = _numeric_scope(tables, list(df.columns)) if source.merge is MergeMode.CONCAT_SAMPLES else column_scope if carry_text else "."
    return df, header_unit, signal, origins, declared, raw, [_input_decimal(table) for table in tables], metadata_scope, row_counts, typed_cells, scope


_ROW_MARKER = "__nio_text_row_{}__"


def _with_row_marker(frame: pd.DataFrame, index: int) -> pd.DataFrame:
    tagged = frame.copy()
    tagged[_ROW_MARKER.format(index)] = np.arange(len(frame), dtype=np.int64)
    return tagged


def _carry_text(merged: pd.DataFrame, tables: list[Any], names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame | None, _NumericScope, dict[str, bool]]:
    """Re-attach the original CSV text to the rows a by_key / concat_features merge produced.

    Each result row knows, per file, which original row it came from (the marker column rode the join); every result
    column is mapped to the file column it came from by the join's own naming rules (shared name, ``col__file``
    suffix, ``file__col`` prefix). Unmatched rows and files without text (typed sources) contribute empty tokens,
    which the numeric conversion never inspects.
    """
    markers = {i: _ROW_MARKER.format(i) for i, t in enumerate(tables) if t.raw is not None}
    positions = {i: merged[m].to_numpy() for i, m in markers.items() if m in merged.columns}
    result = merged.drop(columns=[m for m in markers.values() if m in merged.columns])
    claimed: set[str] = set()
    text: dict[str, np.ndarray] = {}
    scope: dict[str, str | tuple[str, ...]] = {}
    metadata_scope: dict[str, bool] = {}
    for i, table in enumerate(tables):
        for column in table.df.columns:
            candidates = [column, f"{column}__{names[i]}", f"{names[i]}__{column}"]
            target = next((c for c in candidates if c in result.columns and c not in claimed), None)
            if target is None:
                continue
            claimed.add(target)
            scope[target] = _input_decimal(table)
            metadata_scope[target] = table.text_tokens
            if i not in positions or column not in table.raw.columns:
                continue
            tokens = table.raw[column].to_numpy(dtype=object)
            where = positions[i]
            text[target] = np.array([tokens[int(w)] if w == w else "" for w in where], dtype=object)
    if not text:
        return result, None, scope, metadata_scope
    frame = pd.DataFrame({c: text.get(c, np.full(len(result), "", dtype=object)) for c in result.columns}, index=result.index)
    return result, frame, scope, metadata_scope


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
    key_cols.add(_row_idx_col(source.id))  # reserved virtual key (variation row tracking); never a role
    roles: dict[str, str] = {}
    if source.role is not Role.MIXED and not source.columns:
        for col in headers:
            if col not in key_cols:
                roles[col] = source.role.value
        return roles

    assigned: set[int] = set()
    has_rest = any(isinstance(cr.select, RestSelector) for cr in source.columns)
    for cr in source.columns:
        if isinstance(cr.select, AutoSelector):
            idxs = _resolve_auto_selector(source.id, cr, headers, dtypes, assigned, key_cols)
        else:
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


def _resolve_auto_selector(
    source_id: str,
    cr: ColumnRole,
    headers: list[str],
    dtypes: list[str],
    assigned: set[int],
    key_cols: set[str],
) -> list[int]:
    """Resolve an `auto` selector at load time using deterministic, role-aware heuristics.

    - Explicit ``candidates`` always win when one of them is present in the
      headers (case-sensitive); the first matching candidate is used.
    - Otherwise the role decides:
        * ``features``        -> every unassigned numeric column.
        * ``targets``         -> the last unassigned numeric column (a single column).
        * ``metadata``        -> every unassigned non-numeric column.
        * ``ignore``          -> every unassigned column.
        * ``weights``         -> requires explicit ``candidates`` (no safe default).

    The resolution is **deterministic** (no shuffling, no model in the loop):
    it inspects only the loaded frame's column names and dtypes. A user who
    wants a more nuanced choice should run ``infer()`` and store its
    accepted plan as a concrete spec.
    """
    selector = cr.select
    assert isinstance(selector, AutoSelector)
    candidates = list(selector.candidates)
    if candidates:
        for name in candidates:
            if name in headers and name not in key_cols:
                idx = headers.index(name)
                if idx not in assigned:
                    return [idx]
        raise SpecError(f"source '{source_id}': auto selector with candidates {candidates} matched no available column (headers: {headers})")
    role = cr.role
    free = [i for i, h in enumerate(headers) if i not in assigned and h not in key_cols]
    numeric = [i for i in free if dtypes[i] == "numeric"]
    non_numeric = [i for i in free if dtypes[i] != "numeric"]
    if role is Role.FEATURES:
        if not numeric:
            raise SpecError(f"source '{source_id}': auto features selector found no unassigned numeric columns")
        return numeric
    if role is Role.TARGETS:
        if not numeric:
            raise SpecError(f"source '{source_id}': auto targets selector found no unassigned numeric columns (provide explicit candidates for a categorical target)")
        return [numeric[-1]]
    if role is Role.METADATA:
        if not non_numeric:
            raise SpecError(f"source '{source_id}': auto metadata selector found no unassigned non-numeric columns")
        return non_numeric
    if role is Role.IGNORE:
        return free
    raise SpecError(f"source '{source_id}': auto selector for role '{role.value}' requires explicit candidates")


def _build_source_table(source: SourceSpec, spec: DatasetSpec, base_dir: Path, audits: list[dict], options: NumericStorageOptions) -> SourceTable:
    df, header_unit, signal, origins, source_dtypes, raw_text, decimal, text_tokens, row_counts, typed_cells, numeric_scope = _load_source_frame(source, spec, base_dir, audits)
    for column in df.columns:
        if column.startswith(_ROW_IDX_PREFIX):
            raise SpecError(f"source '{source.id}': reserved column name '{column}' collides with a user-supplied column; rename it")
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
    for identity_column in (si.key, si.observation_id, si.repetition_id, si.group_id):
        if isinstance(identity_column, str):
            roles.pop(identity_column, None)
            if identity_column in df.columns:
                # Preserve declared identity as sample-aligned metadata while
                # preventing it from becoming an X/y role.
                roles[identity_column] = "metadata"
    na = effective_params(spec.params, source.params).na
    # nirs4all parity: metadata permits missing values and joins sample removal
    # only when that policy is explicitly requested.
    if source.role is Role.METADATA and na.policy is not NaPolicy.REMOVE_SAMPLE:
        na = replace(na, policy=NaPolicy.IGNORE)
    df, roles, remove_rows = _apply_role_na_policy(df, roles, na)
    # numeric roles keep their original CSV text until conversion: decimal underflow refuses, large integers stay exact
    # (a refusal is deferred to the surviving selected rows: see ``refuse_numeric_cells`` in ``_assemble_block``)
    numeric_columns = [c for c, role in roles.items() if role in ("features", "targets", "weights")]
    delimited = text_tokens if isinstance(text_tokens, list) else [text_tokens] if isinstance(text_tokens, bool) else None
    if row_counts is not None:
        df = restore_numeric_cells(df, raw_text, numeric_columns, decimal, defer=True, segments=row_counts, delimited=delimited)
    else:
        for column in numeric_columns:
            entry = _scope_entry(numeric_scope, column)
            origin = [text_tokens.get(column, False)] if isinstance(text_tokens, dict) else delimited
            df = restore_numeric_cells(df, raw_text, [column], entry if isinstance(entry, str) else list(entry), defer=True, delimited=origin)
    # Only delimited-origin metadata has numeric-text authority; decoded spreadsheet cells keep their types.
    for column, role in roles.items():
        metadata_origin = text_tokens.get(column, False) if isinstance(text_tokens, dict) else text_tokens
        if role == "metadata" and (any(metadata_origin) if isinstance(metadata_origin, list) else metadata_origin):
            metadata_decimal = decimal if row_counts is not None else _scope_entry(numeric_scope, column)
            df = restore_numeric_metadata(df, raw_text, [column], metadata_decimal if isinstance(metadata_decimal, (str, list)) else list(metadata_decimal), row_counts, typed_cells, delimited=metadata_origin)
    # stamp a per-source positional row index AFTER role-splitting so it does
    # not shift positional selectors (e.g. `-1` keeps pointing at the last user
    # column, not at the row-idx sentinel). The column survives joins as any
    # other column and is used to re-align variations to the combined frame.
    row_idx_name = _row_idx_col(source.id)
    if row_idx_name in df.columns:
        raise SpecError(f"source '{source.id}': reserved column name '{row_idx_name}' collides with a user-supplied column; rename it")
    df = df.copy()
    df[row_idx_name] = np.arange(len(df), dtype=np.int64)
    key = source.key if isinstance(source.key, list) else [source.key] if source.key else None
    partition = source.partition.value if source.partition else None
    variations, variation_dtypes, remove_rows = _load_variations(source, spec, base_dir, df, roles, remove_rows, options)
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
        variations=variations,
        variation_dtypes=variation_dtypes,
        remove_rows=remove_rows,
        source_dtypes=source_dtypes,
        numeric_scope=numeric_scope,
        raw_unit=(lambda unit: unit.value if unit else None)(effective_params(spec.params, source.params).header_unit),
    )


def _resolve_variation_inputs(input_val: Any, base_dir: Path) -> list[Path]:
    return _resolve_input_values(input_val, base_dir)


def _read_variation_frame(variation: VariationSpec, spec: DatasetSpec, source: SourceSpec, base_dir: Path) -> tuple[pd.DataFrame, dict[str, str], _NumericScope]:
    """Load (and concatenate, if multi-file) a variation into a single frame.

    A variation file has the same row order as the parent source's loaded frame
    (post-merge). Multiple files concatenate by samples (rows) so the row count
    matches the parent's post-merge row count -- same semantics as the parent's
    own ``concat_samples`` flow.
    """
    params = effective_params(spec.params, source.params)
    if variation.params.to_dict():
        params = effective_params(params, variation.params)
    if params.na.policy is NaPolicy.REMOVE_SAMPLE:
        params.na = replace(params.na, policy=NaPolicy.IGNORE)
    paths = _resolve_variation_inputs(variation.input, base_dir)
    if not paths:
        raise SpecError(f"source '{source.id}': variation '{variation.name}': no input files resolved from {variation.input!r}")
    tables = [read_table(p, params) for p in paths]
    for path, table in zip(paths, tables, strict=True):
        if table.text_tokens:
            refuse_undeclared_delimiter(path, params, source.id)
    frames = [t.df for t in tables]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if any(t.raw is not None for t in tables):
        raws = [t.raw if t.raw is not None else pd.DataFrame("", index=range(len(t.df)), columns=t.df.columns) for t in tables]
        raw = pd.concat(raws, ignore_index=True) if len(raws) > 1 else raws[0]
        frame = restore_numeric_cells(frame, raw if len(raw) == len(frame) else None, list(frame.columns), [_input_decimal(table) for table in tables], defer=True, segments=[len(table.df) for table in tables] if len(tables) > 1 else None, delimited=[table.text_tokens for table in tables])
    declared: dict[str, str] = {}
    for column in frame.columns:
        combined = _combine_dtypes([t.source_dtypes.get(column) for t in tables if column in t.df.columns])
        if combined is not None:
            declared[column] = combined
    return frame, declared, _numeric_scope(tables, list(frame.columns))


def _load_variations(
    source: SourceSpec, spec: DatasetSpec, base_dir: Path, parent_df: pd.DataFrame, roles: dict[str, str], remove_rows: np.ndarray | None, options: NumericStorageOptions
) -> tuple[list[tuple[str, np.ndarray]], list[str | None], np.ndarray | None]:
    """Materialize a feature source's named preprocessing variants into arrays.

    Each variation must row-align with the parent source (post-merge) and provide
    one column per parent *feature* column. Two unambiguous modes are accepted:
    (a) *full name match* -- every parent feature column is present in the
    variation frame (vendor exports usually preserve wavelength headers); or
    (b) *full positional* -- no parent feature header appears in the variation
    frame and ``shape[1] == len(feature_cols)``. Anything between (partial name
    match) raises rather than silently realigning, because a renamed/missing
    column would otherwise be quietly swapped for an unrelated value column.

    The resulting (name, array) tuples are later re-indexed by ``__src_row_idx__``
    so joins/partitions on the parent propagate automatically to the variations.
    """
    if not source.variations:
        return [], [], remove_rows
    if source.role is not Role.FEATURES and not any(r == "features" for r in roles.values()):
        raise SpecError(f"source '{source.id}': variations are only meaningful on a feature source (role/columns must declare 'features')")
    feature_cols = [c for c, r in roles.items() if r == "features"]
    if not feature_cols:
        raise SpecError(f"source '{source.id}': has variations but no feature columns to attach them to")
    parent_n = len(parent_df)
    out: list[tuple[str, np.ndarray]] = []
    dtypes: list[str | None] = []
    seen: set[str] = set()
    for variation in source.variations:
        if variation.name in seen:
            raise SpecError(f"source '{source.id}': duplicate variation name '{variation.name}'")
        seen.add(variation.name)
        vdf, vdf_dtypes, vdf_scope = _read_variation_frame(variation, spec, source, base_dir)
        if len(vdf) != parent_n:
            raise SpecError(f"source '{source.id}': variation '{variation.name}' has {len(vdf)} rows, expected {parent_n} (must row-align with the parent source)")
        params = effective_params(effective_params(spec.params, source.params), variation.params)
        if params.na.policy is NaPolicy.REMOVE_SAMPLE:
            missing = vdf.isna().any(axis=1).to_numpy()
            remove_rows = missing if remove_rows is None else remove_rows | missing
        named = sum(1 for c in feature_cols if c in vdf.columns)
        if named == len(feature_cols):
            refuse_numeric_cells(vdf, feature_cols)
            arr = coerce_numeric(vdf[feature_cols], options.features, {column: _scope_entry(vdf_scope, column) for column in feature_cols})
            child_dtype = _combine_dtypes([vdf_dtypes.get(c) for c in feature_cols])
        elif named == 0 and vdf.shape[1] == len(feature_cols):
            refuse_numeric_cells(vdf, list(vdf.columns))
            arr = coerce_numeric(vdf, options.features, {column: _scope_entry(vdf_scope, column) for column in vdf.columns})
            child_dtype = _combine_dtypes([vdf_dtypes.get(c) for c in vdf.columns])
        else:
            raise SpecError(
                f"source '{source.id}': variation '{variation.name}' is ambiguous: "
                f"{named}/{len(feature_cols)} parent feature columns matched by name "
                f"and the frame has {vdf.shape[1]} columns. Either include all "
                f"parent feature headers or supply exactly {len(feature_cols)} unnamed (positional) columns."
            )
        out.append((variation.name, arr))
        dtypes.append(child_dtype)
    return out, dtypes, remove_rows


# --------------------------------------------------------------------------- #
# Partition splitting (PartitionAssigner subset: column / index / index_file) #
# --------------------------------------------------------------------------- #
_PARTITION_NUMBER_TOKEN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _partition_numeric_key(value: Any, float32: bool = False) -> tuple | None:
    """Use owner identity and storage, keeping the sign that join keys collapse."""
    if float32:
        try:
            value = int_to_f32(int(value)) if isinstance(value, (int, np.integer)) else round_storage(np.asarray([value], dtype=np.float64), "float32")[0]
        except (ValueError, OverflowError):
            return None  # A refused storage conversion cannot match this owner.
    if isinstance(value, (float, np.floating)) and value == 0 and np.signbit(value):
        return ("float", struct.pack(">d", float(value)))
    return _typed_key(value)


def _partition_declared_keys(values: list[Any], float32: bool) -> set[tuple]:
    keys: set[tuple] = set()
    for value in values:
        keys.add(("text", str(value).lower()))
        number = value
        if isinstance(value, str):
            if not _PARTITION_NUMBER_TOKEN.fullmatch(value):
                continue
            try:
                number = strict_json.loads(value)
            except ValueError:
                continue  # Refused numeric tokens remain valid lexical labels.
        if isinstance(number, bool):
            continue
        key = _partition_numeric_key(number, float32)
        if key is not None:
            keys.add(key)
    return keys


def _split_partition_mask(df: pd.DataFrame, spec: DatasetSpec, base_dir: Path) -> dict[str, np.ndarray]:
    """Return a {partition -> boolean row mask} for a single combined frame.

    Only deterministic-by-construction modes are supported (no shuffling at load
    time): ``by: column`` (split on a column's values), ``by: index`` (explicit
    row-index lists), ``by: index_file`` (lists read from a file). A user who
    wants a stratified or percentage split should pre-compute the indices once
    and feed them as an index list / column, or do the split inside the
    pipeline's CV layer.
    """
    p = spec.partitions
    n = len(df)
    if p is None or p.by is None:
        return {"train": np.ones(n, dtype=bool)}
    if p.by is PartitionBy.COLUMN:
        if p.column not in df.columns:
            raise SpecError(f"partitions.column '{p.column}' not found in the assembled data (columns: {list(df.columns)})")
        p.validate_column_values()
        missing = df[p.column].isna().to_numpy()
        converted = df[p.column].astype(str)
        # Preserve pandas' scalar spelling, then use Python's Unicode lowercase.
        # Missingness comes from the original series, never its string spelling.
        col = pd.Series([None if absent else value.lower() for value, absent in zip(converted, missing, strict=True)], index=df.index, dtype=object)
        float32 = str(df[p.column].dtype).lower() == "float32"
        names = ("train_values", "test_values", "val_values", "predict_values")
        declared = [_partition_declared_keys(getattr(p, name), float32) for name in names]
        observed: list[tuple | None] = []
        for value, spelling, absent in zip(df[p.column], converted, missing, strict=True):
            if absent:
                observed.append(None)
            elif isinstance(value, (bool, np.bool_)):
                observed.append(("text", spelling.lower()))
            elif isinstance(value, (int, np.integer, float, np.floating)):
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    observed.append(("text", "-inf" if value < 0 else "inf"))
                else:
                    observed.append(_partition_numeric_key(value))
            else:
                observed.append(("text", spelling.lower()))
        matches = [np.asarray([key is not None and key in keys for key in observed], dtype=bool) for keys in declared]
        conflict = np.flatnonzero(np.sum(matches, axis=0) > 1)
        if len(conflict):
            row = int(conflict[0])
            owners = [name for name, mask in zip(names, matches, strict=True) if mask[row]]
            raise SpecError(f"partitions.column '{p.column}': row {row} matches both '{owners[0]}' and '{owners[1]}'")
        train_mask, test_mask, val_mask, predict_mask = matches
        unknown_mask = ~np.logical_or.reduce(matches)
        if unknown_mask.any():
            from ..spec.enums import UnknownPolicy

            if p.unknown_policy is UnknownPolicy.ERROR:
                observed = sorted(set(col[unknown_mask & ~missing]))
                raise SpecError(f"partitions: unknown values in '{p.column}': {observed}; missing rows: {int(missing.sum())}")
            if p.unknown_policy is UnknownPolicy.TRAIN:
                train_mask = train_mask | unknown_mask
            elif p.unknown_policy is UnknownPolicy.TEST:
                test_mask = test_mask | unknown_mask
            # DROP: leave unknown rows out of every partition
        masks = {"train": train_mask, "test": test_mask}
        if p.val_values:
            masks["val"] = val_mask
        if p.predict_values:
            masks["predict"] = predict_mask
        return {k: v for k, v in masks.items() if v.any()}
    if p.by is PartitionBy.INDEX:
        return _mask_from_index_lists({"train": p.train, "test": p.test, "val": p.val, "predict": p.predict}, n, source="partitions")
    if p.by is PartitionBy.INDEX_FILE:
        loaded = {
            "train": _read_index_file(base_dir / p.train_file) if p.train_file else None,
            "test": _read_index_file(base_dir / p.test_file) if p.test_file else None,
            "val": _read_index_file(base_dir / p.val_file) if p.val_file else None,
            "predict": _read_index_file(base_dir / p.predict_file) if p.predict_file else None,
        }
        return _mask_from_index_lists(loaded, n, source="index file")
    raise SpecError(f"partitions.by={p.by.value} is not supported by the assembler")


def _mask_from_index_lists(by_part: dict[str, Any], n: int, *, source: str) -> dict[str, np.ndarray]:
    """Validate per-partition row-index lists and turn them into boolean masks.

    Indices must be in [0, n), unique within a partition, and disjoint across
    partitions; a row may belong to at most one partition (rows not listed
    anywhere are simply omitted).
    """
    masks: dict[str, np.ndarray] = {}
    seen: dict[int, str] = {}
    for part, raw in by_part.items():
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raise SpecError(f"{source}: '{part}' must be a list of row indices, got {type(raw).__name__}")
        idx_list = [int(v) for v in raw]
        if any(i < 0 or i >= n for i in idx_list):
            raise SpecError(f"{source}: '{part}' contains out-of-range indices (n={n}): {[i for i in idx_list if i < 0 or i >= n]}")
        if len(set(idx_list)) != len(idx_list):
            raise SpecError(f"{source}: '{part}' contains duplicate indices")
        for i in idx_list:
            if i in seen and seen[i] != part:
                raise SpecError(f"{source}: row {i} listed in both '{seen[i]}' and '{part}' (partitions must be disjoint)")
            seen[i] = part
        m = np.zeros(n, dtype=bool)
        m[idx_list] = True
        masks[part] = m
    if not masks:
        raise SpecError(f"{source}: no partition lists provided")
    return masks


def _read_index_file(path: Path) -> list[int]:
    """Read a row-index list from a file. Accepts JSON arrays, CSV/TXT one-per-line, or comma-separated."""
    if not path.exists():
        raise SpecError(f"partitions.index_file: file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return [int(v) for v in strict_json.loads(text)]
    # csv/txt: whitespace and/or comma separated; ignore blanks
    tokens = [t.strip() for t in text.replace(",", "\n").splitlines() if t.strip()]
    return [int(t) for t in tokens]


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def assemble(spec: DatasetSpec, base_dir: str | Path = ".", options: NumericStorageOptions | None = None) -> AssembledDataset:
    """Load, role-split, join and partition ``spec``, then commit the REQUESTED numeric storage.

    ``options`` is the caller's real request (per-role storage and consumer). The raw
    assembly is :func:`_assemble_raw`; the shared finalizer then selects storage for exactly
    the requested roles (features/processings, targets, weights), so the returned IR carries
    truthful precision records and unrequested raw targets/weights stay untouched.
    """
    from .package import finalize_selected_storage

    return finalize_selected_storage(_assemble_raw(spec, base_dir, options), options)


def _assemble_raw(spec: DatasetSpec, base_dir: str | Path = ".", options: NumericStorageOptions | None = None) -> AssembledDataset:
    """The raw F64 assembly, with no storage selection.

    The REAL request stays in ``options``; the effective integer coercion is derived here:
    an explicit float32 request or the SpectroDataset F32 feature consumer converts integer
    sources directly (see ``to_ir_f64``). The IR stays F64 (F32 values widen exactly).
    """
    options = effective_coercion(options)
    base_dir = Path(base_dir)
    audits: list[dict] = []
    tables = {s.id: _build_source_table(s, spec, base_dir, audits, options) for s in spec.sources}

    # group sources by partition; sources without a partition are single-partition ("train" unless split)
    partitions = sorted({t.partition for t in tables.values() if t.partition} - {None})
    assembled = AssembledDataset(
        name=spec.name or "dataset",
        task_type=spec.task_type.value,
        signal_type=spec.signal_type.value,
        n_sources=sum(1 for t in tables.values() if t.kind != SourceKind.LOOKUP.value and _has_role(t, "features")),
        # `repetition` groups repeated measurements of one sample and is opt-in;
        # `sample_index.repetition_id` is a per-scan index and never stands in for it.
        repetition=spec.repetition,
        identity={
            "source_ids": [t.source_id for t in tables.values() if t.kind != SourceKind.LOOKUP.value and _has_role(t, "features")],
            "sample_id": spec.sample_index.key if spec.sample_index.by.value == "id" and isinstance(spec.sample_index.key, str) else None,
            "observation_id": spec.sample_index.observation_id,
            "repetition_id": spec.sample_index.repetition_id,
            "group_id": spec.sample_index.group_id,
        },
        aggregate=spec.aggregate,
    )

    combined_df: pd.DataFrame | None = None
    train_rows: list[int] | None = None
    if partitions:
        for part in partitions:
            part_tables = {sid: t for sid, t in tables.items() if t.partition == part or t.kind == SourceKind.LOOKUP.value}
            assembled.blocks[part], _ = _assemble_block(part_tables, spec, audits, assembled.warnings, options)
    else:
        block, combined_df = _assemble_block(dict(tables), spec, audits, assembled.warnings, options)
        if spec.partitions is None:
            assembled.blocks["train"] = block
            train_rows = list(range(len(combined_df)))
        else:
            # split on the FINAL (post-join) frame so masks line up with the assembled rows
            for part, mask in _split_partition_mask(combined_df, spec, base_dir).items():
                assembled.blocks[part] = _slice_block(block, mask)
                if part == "train":
                    train_rows = np.flatnonzero(mask).tolist()

    assembled.audits = audits
    _attach_folds(spec, base_dir, assembled, combined_df, train_rows)
    return assembled


def _has_role(table: SourceTable, role: str) -> bool:
    return any(r == role for r in table.roles.values())


def _apply_role_na_policy(df: pd.DataFrame, roles: dict[str, str], na: NaConfig) -> tuple[pd.DataFrame, dict[str, str], np.ndarray | None]:
    """Apply a source's NA policy once its column roles are known.

    Loading is lossless for targets and metadata: a missing value stays NaN and
    its sample is kept, because training and scoring exclude target-missing
    samples downstream. Only ``remove_sample`` removes such a sample and only
    ``abort`` refuses a missing target; a target is never imputed, dropped as a
    column or encoded as a class. Every other column follows the policy
    (``auto`` = ``abort``).

    Returns the treated frame and roles, and the rows ``remove_sample`` removes
    once all sources are joined.
    """
    if na.policy is NaPolicy.IGNORE:
        return df, roles, None
    if na.policy is NaPolicy.REMOVE_SAMPLE:
        return df, roles, df.isna().any(axis=1).to_numpy()
    if na.policy is NaPolicy.ABORT:
        apply_na_policy(df[[c for c in df.columns if roles.get(c) == "targets"]], na)
    policy_cols = [c for c in df.columns if roles.get(c) not in ("targets", "metadata")]
    if not policy_cols:
        return df, roles, None
    treated, _ = apply_na_policy(df[policy_cols], na)
    dropped = [c for c in policy_cols if c not in treated.columns]
    if dropped:
        df = df.drop(columns=dropped)
        roles = {c: r for c, r in roles.items() if c not in dropped}
    if na.policy is NaPolicy.REPLACE:
        df = df.copy()
        df[list(treated.columns)] = treated
    return df, roles, None


def _role_decimals(combined: pd.DataFrame, owner: SourceTable, column: str, native_cols: dict[str, str]) -> str | list[str]:
    entry = _scope_entry(owner.numeric_scope, native_cols[column])
    if isinstance(entry, str):
        return entry
    positions = combined[_row_idx_col(owner.source_id)].tolist()
    return [entry[int(position)] if pd.notna(position) else "." for position in positions]


def _assemble_block(tables: dict[str, SourceTable], spec: DatasetSpec, audits: list[dict], warnings: list[str], options: NumericStorageOptions) -> tuple[PartitionBlock, pd.DataFrame]:
    feature_tables = [t for t in tables.values() if t.kind != SourceKind.LOOKUP.value and _has_role(t, "features")]
    if not feature_tables:
        raise SpecError("partition has no feature source")
    primary = feature_tables[0]
    combined = primary.df.copy()
    merge_steps: list[tuple[SourceTable, list[str] | None]] = [(primary, None)]

    # join every non-primary source (extra features, targets, metadata, lookups) onto the primary rows
    for t in tables.values():
        if t.source_id == primary.source_id:
            continue
        combined, dropped_keys = _join_onto(combined, primary, t, audits, warnings)
        merge_steps.append((t, dropped_keys))

    if spec.sample_index.by.value == "id" and isinstance(spec.sample_index.key, str) and spec.sample_index.key not in combined:
        raise JoinError(f"source '{primary.source_id}': declared sample identity '{spec.sample_index.key}' missing in this partition")

    keep = np.ones(len(combined), dtype=bool)
    for table in tables.values():
        if table.remove_rows is not None:
            positions = combined[_row_idx_col(table.source_id)]
            present = positions.notna().to_numpy()
            source_keep = np.zeros(len(combined), dtype=bool)
            source_keep[present] = ~table.remove_rows[positions[present].to_numpy(dtype=np.int64)]
            keep &= source_keep
    if not keep.all():
        audits.append({"operation": "remove_na_samples", "removed": int((~keep).sum())})
        combined = combined.loc[keep].reset_index(drop=True)

    # role -> columns across all joined sources
    role_cols, native_cols = _collect_roles(combined, merge_steps)
    # a numeric-role cell the integer/decimal domain refuses was kept as a marker: only a surviving row refuses
    refuse_numeric_cells(combined, [c for c, rs in role_cols.items() if rs[0] in ("features", "targets", "weights")])

    block = PartitionBlock(n_samples=len(combined))
    # multi-source X: each feature source contributes its own array (aligned to combined rows)
    for ft in feature_tables:
        cols, headers = _source_feature_columns(combined, role_cols, native_cols, ft)
        if not cols:
            continue
        block.source_ids.append(ft.source_id)
        block.X.append(coerce_numeric(combined[cols], options.features, {column: _role_decimals(combined, ft, column, native_cols) for column in cols}))
        block.x_source_dtypes.append(_combine_dtypes([ft.source_dtypes.get(native_cols[c]) for c in cols]))
        block.feature_names.append(list(headers))
        block.feature_axes.append([tabular_axis(headers, ft.header_unit, ft.raw_unit)])
        block.representation_ids.append("signal_1d")
        block.feature_headers.append(headers)
        block.header_units.append(ft.header_unit)
        block.signal_types.append(ft.signal_type)
        # variations: re-align each pre-loaded array to the combined frame using
        # the parent source's positional row index (which survives joins as a
        # regular column).
        src_processings: list[tuple[str, np.ndarray]] = []
        block.processing_source_dtypes.append(list(ft.variation_dtypes))
        if ft.variations:
            idx_col = _row_idx_col(ft.source_id)
            if idx_col not in combined.columns:
                raise SpecError(f"source '{ft.source_id}': row-index column '{idx_col}' lost during assembly; variations cannot be aligned")
            row_idx = combined[idx_col].to_numpy(dtype=np.int64)
            for name, arr in ft.variations:
                src_processings.append((name, arr[row_idx, :]))
        block.processings.append(src_processings)

    target_cols = [c for c in combined.columns if c in role_cols and role_cols[c][0] == "targets"]
    if target_cols:
        counts = Counter(native_cols[c] for c in target_cols)
        target_headers = [native_cols[c] if counts[native_cols[c]] == 1 else c for c in target_cols]
        if len(set(target_headers)) != len(target_headers):
            raise SpecError(f"target export labels must be unique, got {target_headers}")
        y_series, cats, numeric_dtypes = [], {}, []
        for c, header in zip(target_cols, target_headers, strict=True):
            mode = (spec.params.categorical.value if spec.params.categorical else "auto")
            owner = tables[role_cols[c][1]]
            declared = owner.source_dtypes.get(native_cols[c])
            decimal_view = _role_decimals(combined, owner, c, native_cols)
            source = next(source for source in spec.sources if source.id == owner.source_id)
            target_params = effective_params(spec.params, source.params)
            target_mode = target_params.categorical.value if target_params.categorical else mode
            cells = [str(cell).strip() for cell in combined[c].dropna()]
            if target_mode == "auto" and not target_params.decimal_separator and cells and any(_COMMA_PAIR.fullmatch(cell) for cell in cells) and all(_COMMA_PAIR.fullmatch(cell) or _DECIMAL.fullmatch(cell) for cell in cells):
                raise SpecError(f"source '{source.id}': file '{source.input}' target column '{native_cols[c]}' contains numeric comma data; declare params.decimal_separator")
            enc, mapping = encode_categorical(combined[c], target_mode, decimal_view)
            # Class labels are never rounded by a numeric storage request: they stay exact
            # here and the package applies the portable cap and exact-F32 rule.
            classification = mapping is not None or spec.task_type.value in ("binary", "multiclass")
            y_series.append(exact_float64(enc, None if classification else options.targets, decimal_view))
            if mapping:
                cats[header] = {**mapping, "source_dtype": declared} if declared is not None else mapping
            else:
                numeric_dtypes.append(declared)
        block.y = np.column_stack(y_series)
        block.y_headers = target_headers
        block.y_categorical = cats
        block.y_source_dtype = _combine_dtypes(numeric_dtypes) if numeric_dtypes else None

    weight_cols = [c for c, rs in role_cols.items() if rs[0] == "weights"]
    if weight_cols:
        if len(weight_cols) > 1:
            raise SpecError(f"only one 'weights' column is supported, got {weight_cols} (combine them upstream before declaring)")
        weight_owner = tables[role_cols[weight_cols[0]][1]]
        block.weights = coerce_numeric(combined[[weight_cols[0]]], options.weights, {weight_cols[0]: _role_decimals(combined, weight_owner, weight_cols[0], native_cols)})[:, 0]
        block.weights_header = weight_cols[0]
        weight_owner = tables[role_cols[weight_cols[0]][1]]
        block.weights_source_dtype = weight_owner.source_dtypes.get(native_cols[weight_cols[0]])

    meta_cols = [c for c in combined.columns if c in role_cols and role_cols[c][0] == "metadata"]
    if meta_cols:
        # the numeric-authority refusal applies to the rows that survived NA removal (see ``restore_numeric_metadata``)
        refuse_numeric_cells(combined, meta_cols)
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


def _join_onto(combined: pd.DataFrame, primary: SourceTable, t: SourceTable, audits: list[dict], warnings: list[str]) -> tuple[pd.DataFrame, list[str] | None]:
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
            return out, right_on
        if j.cardinality is Cardinality.ONE_TO_ONE:
            return _row_align(combined, t, primary), None  # 1:1 with no declared keys -> row order
        raise SpecError(f"source '{t.source_id}': {j.cardinality.value} join needs explicit left_on/right_on")
    # no join spec: align by a shared key (1:1) if both declare one, else row order
    lkeys, rkeys = _as_key_list(primary.key), _as_key_list(t.key)
    if lkeys and rkeys and all(c in combined.columns for c in lkeys) and all(c in t.df.columns for c in rkeys):
        out, audit = join_tables(combined, t.df, left_on=lkeys, right_on=rkeys, cardinality=Cardinality.ONE_TO_ONE, coverage=Coverage.COMPLETE, left_name=primary.source_id, right_name=t.source_id)
        warnings.extend(audit.warnings)
        return out, rkeys
    return _row_align(combined, t, primary), None


def _collect_roles(combined: pd.DataFrame, merge_steps: list[tuple[SourceTable, list[str] | None]]) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Collect role/owner and native names in the exact frame merge order.

    Keys and internal columns claim their names even when they have no role.
    A keyed join drops its right keys, so those never claim a payload name.
    """
    role_cols: dict[str, tuple[str, str]] = {}
    native_cols: dict[str, str] = {}
    seen: set[str] = set()
    for table, dropped_keys in merge_steps:
        left_names = set(seen)
        for col in table.df.columns:
            if dropped_keys is not None and col in dropped_keys:
                continue
            if dropped_keys is not None:
                name = f"{col}__{table.source_id}" if col in left_names else col
            else:
                name = f"{table.source_id}__{col}" if col in seen else col
            seen.add(name)
            if name not in combined.columns:
                continue
            native_cols[name] = col
            if col in table.roles:
                role_cols[name] = (table.roles[col], table.source_id)
    return role_cols, native_cols


def _source_feature_columns(combined: pd.DataFrame, role_cols: dict[str, tuple[str, str]], native_cols: dict[str, str], source: SourceTable) -> tuple[list[str], list[str]]:
    """Return actual combined columns and their collected source-native names."""
    cols = [col for col in combined.columns if role_cols.get(col) == ("features", source.source_id)]
    return cols, [native_cols[col] for col in cols]


def _slice_block(block: PartitionBlock, mask: np.ndarray) -> PartitionBlock:
    out = PartitionBlock(n_samples=int(mask.sum()))
    out.source_ids = list(block.source_ids)
    out.X = [x[mask] for x in block.X]
    out.feature_headers = list(block.feature_headers)
    out.header_units = list(block.header_units)
    out.signal_types = list(block.signal_types)
    out.processings = [[(name, arr[mask]) for name, arr in src] for src in block.processings]
    out.x_source_dtypes = list(block.x_source_dtypes)
    out.x_precisions = list(block.x_precisions)
    out.processing_source_dtypes = [list(row) for row in block.processing_source_dtypes]
    out.processing_precisions = [list(row) for row in block.processing_precisions]
    out.y_precisions = list(block.y_precisions)
    out.weights_precision = block.weights_precision
    out.feature_names = [None if names is None else list(names) for names in block.feature_names]
    out.feature_axes = [[dict(axis) for axis in axes] for axes in block.feature_axes]
    out.representation_ids = list(block.representation_ids)
    out.y_source_dtype = block.y_source_dtype
    out.weights_source_dtype = block.weights_source_dtype
    out.y = block.y[mask] if block.y is not None else None
    out.y_headers = list(block.y_headers)
    out.y_categorical = dict(block.y_categorical)
    out.metadata = block.metadata.iloc[mask].reset_index(drop=True) if block.metadata is not None else None
    out.weights = block.weights[mask] if block.weights is not None else None
    out.weights_header = block.weights_header
    return out


def _attach_folds(spec: DatasetSpec, base_dir: Path, assembled: AssembledDataset, combined_df: pd.DataFrame | None = None, train_rows: list[int] | None = None) -> None:
    if spec.folds is None:
        return
    if "train" not in assembled.blocks:
        raise SpecError("folds require a train block; only training rows may declare folds")
    train_block = assembled.blocks["train"]
    n_rows = len(combined_df) if combined_df is not None else train_block.n_samples
    training = list(train_rows or []) if combined_df is not None else list(range(n_rows))
    positions = {row: index for index, row in enumerate(training)}
    from .folds import _parse_json, parse_fold_file

    raw: list[tuple[list[int], list[int]]] = []
    if spec.folds.inline:
        raw = _parse_json(spec.folds.inline)
    elif spec.folds.column:
        if combined_df is None:
            raise SpecError(f"folds.column='{spec.folds.column}' requires a single combined frame (it does not apply with explicit per-source partitions)")
        if spec.folds.column not in combined_df.columns:
            raise SpecError(f"folds.column='{spec.folds.column}' not found in the assembled data (columns: {list(combined_df.columns)})")
        col = combined_df[spec.folds.column]
        if any(not pd.isna(col.iloc[i]) for i in range(n_rows) if i not in positions):
            raise SpecError(f"folds.column='{spec.folds.column}': only training rows may have fold labels")
        for value in sorted(col.iloc[training].dropna().unique(), key=lambda v: str(v)):
            val_idx = [i for i in training if not pd.isna(col.iloc[i]) and col.iloc[i] == value]
            val_set = set(val_idx)
            raw.append(([i for i in training if i not in val_set], val_idx))
        if not raw:
            raise SpecError(f"folds.column='{spec.folds.column}' has no non-NaN values")
    elif spec.folds.file:
        raw = parse_fold_file(base_dir / spec.folds.file, spec.folds.format.value)

    for train, validation in raw:
        for side, indices in (("train", train), ("val", validation)):
            if any(type(index) is not int for index in indices):
                raise SpecError("fold indices must be integers (not bool, float or string)")
            if any(index < 0 or index >= n_rows for index in indices):
                raise SpecError(f"folds.{side}: index out of range for {n_rows} frame rows")
            if any(index not in positions for index in indices):
                raise SpecError(f"folds.{side}: only training rows may declare folds")
            if len(set(indices)) != len(indices):
                raise SpecError(f"folds.{side}: duplicate indices")
        if set(train) & set(validation):
            raise SpecError("fold train and validation sides must be disjoint")

    identity_column = spec.sample_index.observation_id or (
        spec.sample_index.key if spec.sample_index.by.value == "id" and isinstance(spec.sample_index.key, str) else None
    )
    identity_frame = combined_df if combined_df is not None else train_block.metadata
    if identity_frame is not None and identity_column in identity_frame.columns:
        values = identity_frame[identity_column]

        def translate(indices: list[int]) -> list[str] | None:
            out: list[str] = []
            for index in indices:
                if pd.isna(values.iloc[index]):
                    return None
                value = str(values.iloc[index])
                if not value.strip():
                    return None
                out.append(value)
            return out

        provenance = [
            {"train_observation_ids": train_ids, "validation_observation_ids": validation_ids}
            for train_idx, validation_idx in raw
            if (train_ids := translate(train_idx)) is not None and (validation_ids := translate(validation_idx)) is not None
        ]
        if len(provenance) == len(raw):
            assembled.fold_provenance = provenance
    # Provenance used final frame rows; public folds now use train-block positions.
    assembled.folds = [([positions[i] for i in train], [positions[i] for i in val]) for train, val in raw]
