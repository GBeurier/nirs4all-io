# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Relational merge/join engine (Epic 4.0).

``nirs4all-io`` performs every join/broadcast itself (dag-ml-data does not join,
Critique C13). This module operates on pandas ``DataFrame``s and is **role-
agnostic** — column roles are tracked by the builder; here we only align rows.

- ``concat_samples`` — vertically stack rows (schema-union), tracking origin.
- ``concat_features`` — horizontally stack column-blocks (by key or row order).
- ``merge_by_key`` — relational join of a list of frames on a shared key.
- ``join_tables`` — a single cross-source join with **cardinality** (1:1/m:1/1:m),
  **duplicate-key** rules, **coverage** (complete/warn/drop/error) and a
  **dropped-row audit** (Critique C8). Cardinality/duplicate violations are
  always errors, independent of ``coverage`` (E.2).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..spec.enums import Cardinality, Coverage, SpecError
from .loaders import NumericRefusal


class JoinError(SpecError):
    """A cardinality, duplicate-key, or coverage violation."""


@dataclass
class JoinAudit:
    operation: str
    n_left: int = 0
    n_right: int = 0
    n_result: int = 0
    n_matched: int = 0
    dropped_rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _keys(on: str | list[str]) -> list[str]:
    return [on] if isinstance(on, str) else list(on)


def _require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise JoinError(f"{where}: key column(s) {missing} not found (have {list(df.columns)})")


def _typed_key(cell: Any) -> tuple[Any, ...]:
    """The admitted key domain, independent of pandas column-wide dtype inference."""
    if isinstance(cell, NumericRefusal):
        return ("lexeme", cell.lexeme if cell.lexeme is not None else cell.token)
    if cell is None or pd.isna(cell):
        return ("missing",)
    if isinstance(cell, (bool, np.bool_)):
        return ("bool", bool(cell))
    if isinstance(cell, (int, np.integer)):
        value = int(cell)
        if -(2**63) <= value < 2**64:
            return ("integer", value)
        return ("lexeme", str(value))
    if isinstance(cell, (float, np.floating)):
        floating = float(cell)
        if floating.is_integer() and -(2**63) <= floating < 2**64:
            return ("integer", int(floating))
        return ("float", struct.pack(">d", floating))
    return ("text", str(cell))


def _key_tuples(df: pd.DataFrame, keys: list[str]) -> list[tuple]:
    # Column iteration avoids itertuples' row-wide numeric promotion.
    columns = [df[key].tolist() for key in keys]
    return [tuple(_typed_key(cell) for cell in row) for row in zip(*columns, strict=True)]


def _missing_key(key: tuple) -> bool:
    return any(cell[0] == "missing" for cell in key)


def _diagnostic_keys(frame: pd.DataFrame, columns: list[str], typed_keys: list[tuple]) -> dict[tuple, Any]:
    """Public scalar/compound diagnostics retain cells, never private comparison tags."""
    result: dict[tuple, Any] = {}
    cells = [frame[column].tolist() for column in columns]
    for key, row in zip(typed_keys, zip(*cells, strict=True), strict=True):
        values = tuple(cell.token if isinstance(cell, NumericRefusal) else cell for cell in row)
        result.setdefault(key, values[0] if len(values) == 1 else values)
    return result


def concat_samples(frames: list[pd.DataFrame], names: list[str]) -> tuple[pd.DataFrame, list[str], JoinAudit]:
    """Vertically stack rows with a schema-union of columns; track origin per row."""
    if not frames:
        return pd.DataFrame(), [], JoinAudit(operation="concat_samples")
    origins: list[str] = []
    for frame, name in zip(frames, names, strict=False):
        origins.extend([name] * len(frame))
    stacked = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    audit = JoinAudit(operation="concat_samples", n_result=len(stacked))
    union_cols = set().union(*(set(f.columns) for f in frames))
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    if union_cols != common:
        audit.warnings.append(f"concat_samples schema-union: {len(union_cols - common)} column(s) not shared across files were null-filled")
    return stacked, origins, audit


def concat_features(frames: list[pd.DataFrame], names: list[str], *, key: str | list[str] | None = None) -> tuple[pd.DataFrame, JoinAudit]:
    """Horizontally stack column-blocks for the same samples (by key or row order)."""
    if not frames:
        return pd.DataFrame(), JoinAudit(operation="concat_features")
    audit = JoinAudit(operation="concat_features")
    if key is not None:
        # same samples in each block -> validated 1:1 join (unique keys, complete coverage),
        # NOT a raw inner merge (which would silently multiply dup keys / drop missing keys).
        out = frames[0]
        for frame, name in zip(frames[1:], names[1:], strict=False):
            out, step = join_tables(out, frame, left_on=key, right_on=key, cardinality=Cardinality.ONE_TO_ONE, coverage=Coverage.COMPLETE, left_name=names[0], right_name=name)
            audit.warnings.extend(step.warnings)
        audit.n_result = len(out)
        return out, audit
    lengths = {len(f) for f in frames}
    if len(lengths) > 1:
        raise JoinError(f"concat_features by row order needs equal row counts, got {sorted(lengths)} (supply a 'key')")
    renamed = []
    seen: set[str] = set()
    for frame, name in zip(frames, names, strict=False):
        cols = {}
        for c in frame.columns:
            target = c if c not in seen else f"{name}__{c}"
            if target in seen:
                raise JoinError(f"source '{name}': column '{c}' generates duplicate output column '{target}'")
            seen.add(target)
            cols[c] = target
        renamed.append(frame.rename(columns=cols).reset_index(drop=True))
    out = pd.concat(renamed, axis=1)
    audit.n_result = len(out)
    return out, audit


def join_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str | list[str],
    right_on: str | list[str],
    cardinality: Cardinality,
    coverage: Coverage,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[pd.DataFrame, JoinAudit]:
    """Join ``right`` onto ``left`` honoring cardinality, duplicate rules, coverage."""
    lkeys, rkeys = _keys(left_on), _keys(right_on)
    _require_columns(left, lkeys, left_name)
    _require_columns(right, rkeys, right_name)
    if len(lkeys) != len(rkeys):
        raise JoinError(f"join keys must have equal arity: {lkeys} vs {rkeys}")

    # Validate payload names before pandas can merge or overwrite colliding columns.
    seen: set[str] = set()
    left_names = set(left.columns)
    for c in left.columns:
        if c in seen:
            raise JoinError(f"source '{left_name}': duplicate output column '{c}'")
        seen.add(c)
    for c in right.columns:
        if c in rkeys:
            continue
        target = f"{c}__{right_name}" if c in left_names else c
        if target in seen:
            raise JoinError(f"source '{right_name}': column '{c}' generates duplicate output column '{target}'")
        seen.add(target)

    left_tuples, right_tuples = _key_tuples(left, lkeys), _key_tuples(right, rkeys)
    left_dup = len(set(left_tuples)) != len(left_tuples)
    right_dup = len(set(right_tuples)) != len(right_tuples)
    if cardinality is Cardinality.ONE_TO_ONE:
        if left_dup:
            raise JoinError(f"1:1 join: duplicate keys on left '{left_name}'")
        if right_dup:
            raise JoinError(f"1:1 join: duplicate keys on right '{right_name}'")
    elif cardinality is Cardinality.MANY_TO_ONE:
        if right_dup:
            raise JoinError(f"m:1 join: duplicate keys on the right (lookup) '{right_name}' — a dimension table must be unique on {rkeys}")
    elif cardinality is Cardinality.ONE_TO_MANY:
        if left_dup:
            raise JoinError(f"1:m join: duplicate keys on left '{left_name}' on {lkeys}")

    right_keyset = {key for key in right_tuples if not _missing_key(key)}
    missing = [key for key in left_tuples if _missing_key(key) or key not in right_keyset]
    missing_unique = list(dict.fromkeys(missing))
    diagnostics = _diagnostic_keys(left, lkeys, left_tuples)

    if missing and coverage is Coverage.COMPLETE:
        preview = [diagnostics[key] for key in missing_unique[:10]]
        raise JoinError(f"coverage 'complete': {len(missing_unique)} key(s) in '{left_name}' have no match in '{right_name}': {preview}{' ...' if len(missing_unique) > 10 else ''}")
    if missing and coverage is Coverage.ERROR:
        raise JoinError(f"coverage 'error': key {diagnostics[missing[0]]!r} from '{left_name}' not found in '{right_name}'")

    how = "inner" if coverage is Coverage.DROP else "left"
    name = "__nio_join_code__"
    while name in left.columns or name in right.columns:
        name += "_"
    matched_keys = {key for key in left_tuples if not _missing_key(key)} & right_keyset
    codes: dict[tuple, int] = {}
    def code(key: tuple, missing_code: int) -> int:
        if key not in matched_keys:
            return missing_code
        if key not in codes:
            codes[key] = len(codes)
        return codes[key]
    left_codes = [code(key, -1 - row) for row, key in enumerate(left_tuples)]
    right_codes = [code(key, -1 - len(left_tuples) - row) for row, key in enumerate(right_tuples)]
    left_view = left.assign(**{name: np.asarray(left_codes, dtype=np.int64)})
    # The right key is a comparison operand only; the left key remains the original payload.
    right_view = right.drop(columns=rkeys).assign(**{name: np.asarray(right_codes, dtype=np.int64)})
    merged = left_view.merge(right_view, how=how, on=name, suffixes=("", f"__{right_name}"), sort=False).drop(columns=name)

    audit = JoinAudit(
        operation=f"{cardinality.value} join '{left_name}' <- '{right_name}'",
        n_left=len(left),
        n_right=len(right),
        n_result=len(merged),
        n_matched=len(left_tuples) - len(missing),
    )
    if missing and coverage is Coverage.DROP:
        audit.dropped_rows = [{"key": diagnostics[key], "reason": "unmatched_left"} for key in missing_unique]
        audit.warnings.append(f"coverage 'drop': removed {len(missing)} unmatched row(s) from '{left_name}' ({len(missing_unique)} distinct key(s))")
    if missing and coverage is Coverage.WARN:
        audit.warnings.append(f"coverage 'warn': {len(missing_unique)} key(s) in '{left_name}' unmatched in '{right_name}'; right columns null-filled")
    return merged, audit


def merge_by_key(frames: list[pd.DataFrame], names: list[str], *, key: str | list[str], coverage: Coverage = Coverage.COMPLETE) -> tuple[pd.DataFrame, JoinAudit]:
    """Relational 1:1 join of several frames on a shared key (the ``by_key`` merge)."""
    if not frames:
        return pd.DataFrame(), JoinAudit(operation="by_key")
    out, audit = frames[0], JoinAudit(operation="by_key", n_result=len(frames[0]))
    for frame, name in zip(frames[1:], names[1:], strict=False):
        out, step = join_tables(out, frame, left_on=key, right_on=key, cardinality=Cardinality.ONE_TO_ONE, coverage=coverage, left_name=names[0], right_name=name)
        audit.warnings.extend(step.warnings)
        audit.dropped_rows.extend(step.dropped_rows)
        audit.n_result = step.n_result
    return out, audit
