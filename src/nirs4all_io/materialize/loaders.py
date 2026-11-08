# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Tabular loaders + NA policy + param precedence (Epic 4.1a / 4.2).

Copied and re-orchestrated from ``nirs4all`` ``data/loaders/*`` and
``data/loaders/base.py::apply_na_policy`` (COPY_PROVENANCE #5/#6); io stays
self-contained (no nirs4all import). A loader reads a file into a pandas
``DataFrame`` with string column names and pandas-inferred dtypes; column-role
numeric coercion / categorical encoding happen later (at role split). Vendor
spectroscopy files are decoded through ``nirs4all-formats`` — never reparsed here.
"""

from __future__ import annotations

import codecs
import datetime
import gzip
import io
import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..spec.dataset_spec import LoadingParams, NaConfig
from ..spec.enums import FillMethod, NaPolicy, SpecError
from ..strict_json import NumberDomainError, check_integer, decimal_float
from .numeric import NumericContractError, to_ir_f64

_Decimals = str | tuple[str, ...] | list[str]

_CSV_EXTS = {".csv", ".tsv", ".txt"}
_CSV_COMPOUND = (".csv.gz", ".csv.zip")
_NA_VALUES = ["NA", "N/A", ""]
# pandas' own default missing-value tokens (``keep_default_na=True``); the reader passes ``_NA_VALUES`` on top of them
_PANDAS_NA_TOKENS = frozenset({"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"}) | frozenset(_NA_VALUES)
_VENDOR_HINT_EXTS = {".0", ".1", ".2", ".dx", ".jdx", ".spc", ".spa", ".asd", ".sed", ".sig", ".sli", ".0a"}
_VENDOR_META_COLUMNS = {"sample_id"}
_VENDOR_META_PREFIXES = ("nirs4all_formats.",)
_VENDOR_FEATURE_PREFIX = "x_"


class NAError(SpecError):
    """NA values were found and the policy is 'abort'."""


class LoaderError(SpecError):
    """A file could not be read by any loader."""


@dataclass
class LoadedTable:
    df: pd.DataFrame
    headers: list[str]
    header_unit: str
    dtypes: list[str]  # per column: "numeric" | "string" | "datetime" | "bool"
    na_report: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    # The ORIGINAL cell text of a CSV file (same shape and headers as ``df``), kept so the numeric role conversion
    # can inspect the lexemes pandas already parsed (decimal underflow, large integers in a float-inferred column).
    raw: pd.DataFrame | None = None
    decimal: str = "."
    # True only for a delimited-text file: its ``raw`` cells are the original lexemes, so a selected metadata/ID column can
    # be inspected as numeric text. Spreadsheet cells (physical numbers, text cells) are not covered by that CSV rule.
    text_tokens: bool = False
    # Declared storage dtype per column, captured from typed readers (numpy,
    # parquet) BEFORE any widening. Text formats (CSV/Excel) declare nothing:
    # their dtypes are parser inference, never a source declaration.
    source_dtypes: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Param precedence (global -> source); source wins on explicitly-set fields    #
# --------------------------------------------------------------------------- #
def _merge_na(base: NaConfig, over: NaConfig) -> NaConfig:
    """Field-wise NA merge: the source configures NA but inherits unset fill bits."""
    if over.policy is NaPolicy.AUTO and over.fill_method is None:
        return base
    return NaConfig(
        policy=over.policy if over.policy is not NaPolicy.AUTO else base.policy,
        fill_method=over.fill_method or base.fill_method,
        fill_value=over.fill_value if over.fill_value is not None else base.fill_value,
        fill_per_column=over.fill_per_column,
    )


def effective_params(global_params: LoadingParams | None, source_params: LoadingParams | None) -> LoadingParams:
    """Merge global and source loading params (source wins on set fields)."""
    base = global_params or LoadingParams()
    over = source_params or LoadingParams()
    na = _merge_na(base.na, over.na)
    fmt_values = {**base.format.values, **over.format.values}
    from ..spec.dataset_spec import FormatParams

    merged = LoadingParams(
        delimiter=over.delimiter or base.delimiter,
        decimal_separator=over.decimal_separator or base.decimal_separator,
        has_header=over.has_header if over.has_header is not None else base.has_header,
        header_unit=over.header_unit or base.header_unit,
        signal_type=over.signal_type or base.signal_type,
        encoding=over.encoding or base.encoding,
        na=na,
        categorical=over.categorical or base.categorical,
        format=FormatParams(values=fmt_values),
    )
    return merged


# --------------------------------------------------------------------------- #
# NA policy (ported from nirs4all base.py::apply_na_policy)                    #
# --------------------------------------------------------------------------- #
def apply_na_policy(df: pd.DataFrame, na: NaConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = NaPolicy.ABORT if na.policy is NaPolicy.AUTO else na.policy
    na_mask = df.isna()
    na_any_row = na_mask.any(axis=1)
    na_any_col = na_mask.any(axis=0)
    report: dict[str, Any] = {
        "strategy": policy.value,
        "na_detected": bool(na_any_row.any()),
        "na_count": int(na_mask.to_numpy().sum()),
        "na_samples": int(na_any_row.sum()),
        "na_features": int(na_any_col.sum()),
        "removed_samples": [],
        "removed_features": [],
        "fill_method": None,
        "na_preserved": False,
    }
    if not report["na_detected"]:
        return df, report

    if policy is NaPolicy.ABORT:
        first_row = df.index[na_any_row][0]
        first_col = df.loc[first_row].isna().idxmax()
        raise NAError(f"NA values detected and na_policy is 'abort'. First NA in column '{first_col}' (row: {first_row}).")
    if policy is NaPolicy.REMOVE_SAMPLE:
        report["removed_samples"] = df.index[na_any_row].tolist()
        return df[~na_any_row].copy().reset_index(drop=True), report
    if policy is NaPolicy.REMOVE_FEATURE:
        removed = df.columns[na_any_col].tolist()
        report["removed_features"] = removed
        return df.drop(columns=removed), report
    if policy is NaPolicy.REPLACE:
        method = na.fill_method or FillMethod.VALUE
        report["fill_method"] = method.value
        return _fill(df, na, method), report
    # IGNORE
    report["na_preserved"] = True
    return df, report


def _fill(df: pd.DataFrame, na: NaConfig, method: FillMethod) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").columns
    if method is FillMethod.VALUE:
        return df.fillna(na.fill_value if na.fill_value is not None else 0.0)
    if method is FillMethod.MEAN:
        return df.fillna(df[numeric].mean()) if na.fill_per_column else df.fillna(float(np.nanmean(df[numeric].to_numpy(dtype=float))))
    if method is FillMethod.MEDIAN:
        return df.fillna(df[numeric].median()) if na.fill_per_column else df.fillna(float(np.nanmedian(df[numeric].to_numpy(dtype=float))))
    if method is FillMethod.FORWARD_FILL:
        return df.ffill(axis=1)
    return df.bfill(axis=1)


# --------------------------------------------------------------------------- #
# dtype inference                                                             #
# --------------------------------------------------------------------------- #
def infer_dtypes(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_bool_dtype(dt):
            out.append("bool")
        elif pd.api.types.is_numeric_dtype(dt) or (pd.api.types.is_object_dtype(dt) and not df[col].dropna().empty and all(_is_exact_number(cell) or isinstance(cell, NumericRefusal) for cell in df[col].dropna())):
            out.append("numeric")
        elif pd.api.types.is_datetime64_any_dtype(dt):
            out.append("datetime")
        else:
            out.append("string")
    return out


# --------------------------------------------------------------------------- #
# Per-format readers                                                          #
# --------------------------------------------------------------------------- #
def _ext(path: Path) -> str:
    low = path.name.lower()
    for compound in _CSV_COMPOUND + (".tar.gz",):
        if low.endswith(compound):
            return compound
    return path.suffix.lower()


_DIALECT_SAMPLE_BYTES = 8 * 1024 * 1024
_DIALECT_SAMPLE_RECORDS = 50


_COMMA_PAIR = re.compile(r"[+-]?[0-9]+,[0-9]+(?:[eE][+-]?[0-9]+)?\Z")


def _dialect_records(text: str, delimiter: str, truncated: bool = False) -> list[tuple[int, str]]:
    # Count widths only, without decoding fields or changing csv's global
    # field-size limit. Quoting follows the existing reader: quotes open
    # at a field start, doubled quotes escape, and quoted newlines stay data.
    rows: list[tuple[int, str]] = []
    start = i = 0
    width = 1
    quoted, field_start = False, True
    while i < len(text):
        char = text[i]
        if quoted:
            if char == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    i += 2
                    continue
                quoted = False
        elif char == '"' and field_start:
            quoted, field_start = True, False
        elif char == delimiter:
            width += 1
            field_start = True
        elif char in "\r\n":
            raw = text[start:i]
            if raw.strip(" \t\r\n\f"):
                rows.append((width, raw))
                if len(rows) == _DIALECT_SAMPLE_RECORDS:
                    return rows
            if char == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
            start, width, field_start = i + 1, 1, True
        else:
            field_start = False
        i += 1
    raw = text[start:]
    # A byte cap may bisect a field. Retain every complete record, but
    # exclude an unfinished tail; full inputs may end without a newline.
    if not truncated and raw.strip(" \t\r\n\f"):
        rows.append((width, raw))
    return rows


def _is_na_record(raw: str) -> bool:
    """Recognize a missing-only CSV record, including its quoted spelling."""
    token = raw.strip()
    return token in _PANDAS_NA_TOKENS or (token.startswith('"') and token.endswith('"') and token[1:-1] in _PANDAS_NA_TOKENS)


def _undeclared_delimiter(text: str, *, truncated: bool = False) -> tuple[str, tuple[str, ...]]:
    """Classify only consistent alternatives to the intentional semicolon default.

    This is deliberately narrower than inspection's delimiter detector. Each
    candidate uses CSV quoting and includes the header in its width evidence.
    """

    default_rows = _dialect_records(text, ";", truncated)
    if len(default_rows) < 2 or any(row[0] > 1 for row in default_rows):
        return "default", ()
    candidates = []
    for delimiter in (",", "\t", "|"):
        rows = _dialect_records(text, delimiter, truncated)
        if len(rows) >= 2 and rows[0][0] >= 2 and all(row[0] == rows[0][0] for row in rows):
            candidates.append(delimiter)
    if not candidates:
        rows = _dialect_records(text, ",", truncated)
        data = [row for row in rows[1:] if row[0] != 1]
        if rows and rows[0][0] >= 2 and data and all(row[0] == 2 and _COMMA_PAIR.fullmatch(row[1].strip()) for row in data):
            return "ambiguous", (",",)
        return "default", ()
    comma_pair = candidates == [","] and all(_COMMA_PAIR.fullmatch(row[1].strip()) for row in default_rows[1:])
    return ("detected" if len(candidates) == 1 and not comma_pair else "ambiguous"), tuple(candidates)


def _delimiter_sample(path: Path, params: LoadingParams) -> tuple[str, bool]:
    """Read at most 8 MiB of decoded CSV payload, including gzip/single-file ZIP."""
    def prefix(stream: Any) -> bytes:
        return stream.read(_DIALECT_SAMPLE_BYTES)

    lower = path.name.lower()
    if lower.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            data = prefix(stream)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise LoaderError(f"delimiter sample requires one ZIP member: {path}")
            with archive.open(names[0]) as member:
                data = prefix(member)
    else:
        with path.open("rb") as plain:
            data = prefix(plain)
    try:
        decoder = codecs.getincrementaldecoder(params.encoding or "utf-8")()
        text = decoder.decode(data, final=len(data) < _DIALECT_SAMPLE_BYTES)
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return text.removeprefix("\ufeff"), len(data) == _DIALECT_SAMPLE_BYTES


def proposed_delimiter(path: Path, params: LoadingParams) -> str | None:
    """Propose an unambiguous alternative for an automatically generated source."""
    text, truncated = _delimiter_sample(path, params)
    verdict, delimiters = _undeclared_delimiter(text, truncated=truncated)
    return delimiters[0] if verdict == "detected" else None


def refuse_undeclared_delimiter(path: Path, params: LoadingParams, source_id: str) -> None:
    """Refuse alternate dialect evidence before assembly coerces text to numbers."""
    if params.delimiter:
        return
    text, truncated = _delimiter_sample(path, params)
    verdict, delimiters = _undeclared_delimiter(text, truncated=truncated)
    if verdict != "default":
        detail = f"splits consistently on {delimiters[0]!r}" if verdict == "detected" else f"has ambiguous delimiters {list(delimiters)!r} (and params.decimal_separator if ',' is the decimal mark)"
        raise LoaderError(f"source '{source_id}': file '{path.name}' has no ';' but {detail}; declare params.delimiter")


def discovery_table(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Project at most 50 complete CSV records from the existing 8 MiB sample.

    Profiles select column names only. Full typed-key validation and NA policy
    remain with materialization; no key equality is decided from this preview.
    """
    text, truncated = _delimiter_sample(path, params)
    if not params.delimiter and _undeclared_delimiter(text, truncated=truncated)[0] != "default":
        return None
    records = _dialect_records(text, params.delimiter or ";", truncated)
    if not records:
        return None
    prefix = "\n".join(raw for _, raw in records)
    options = _csv_options(params)
    options["skip_blank_lines"] = True
    try:
        df = pd.read_csv(io.StringIO(prefix), **options)
        raw = pd.read_csv(io.StringIO(prefix), **{**options, "dtype": str, "na_filter": False, "keep_default_na": False, "na_values": None})
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        options["engine"] = "c"
        try:
            df = pd.read_csv(io.StringIO(prefix), **options)
            raw = pd.read_csv(io.StringIO(prefix), **{**options, "dtype": str, "na_filter": False, "keep_default_na": False, "na_values": None})
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None
    df.columns = [str(c) for c in df.columns]
    raw.columns = [str(c) for c in raw.columns]
    return (df, raw) if df.shape == raw.shape else None


def _csv_options(params: LoadingParams) -> dict[str, Any]:
    options: dict[str, Any] = {
        "sep": params.delimiter or ";", "decimal": params.decimal_separator or ".",
        "header": 0 if params.has_header is not False else None,
        "na_values": _NA_VALUES, "keep_default_na": True,
        "skip_blank_lines": False, "index_col": False, "engine": "python",
    }
    if params.format.values.get("usecols") is not None:
        options["usecols"] = params.format.values["usecols"]
    return options


def _read_csv(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    decimal = params.decimal_separator or "."
    has_header = True if params.has_header is None else params.has_header
    encoding = params.encoding or "utf-8"
    fmt = params.format.values
    kwargs = _csv_options(params)
    compression = "gzip" if path.name.lower().endswith(".gz") else "zip" if path.name.lower().endswith(".zip") else None
    if compression:
        kwargs["compression"] = compression
    guarded_width = None
    try:
        guarded_width = _guard_csv_width(path, encoding, kwargs)
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        encoding = "latin-1"
        guarded_width = _guard_csv_width(path, encoding, kwargs)
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    except LoaderError:
        raise
    except Exception:
        kwargs["engine"] = "c"
        df = pd.read_csv(path, encoding=encoding, **kwargs)
    if df.shape[1] == 0:
        raise pd.errors.EmptyDataError("No columns to parse from file")
    if fmt.get("usecols") is None and guarded_width is not None and guarded_width > df.shape[1]:
        raise LoaderError(f"Expected {df.shape[1]} fields, found {guarded_width} in physical CSV records")
    df.columns = [str(c) for c in df.columns]
    header_unit = params.header_unit.value if params.header_unit else ("text" if has_header else "index")
    # the same file again as untouched text (no dtype inference, no NA conversion): same parser, same shape
    try:
        raw = pd.read_csv(path, encoding=encoding, **{**kwargs, "dtype": str, "na_filter": False, "keep_default_na": False, "na_values": None})
        raw.columns = [str(c) for c in raw.columns]
        if raw.shape != df.shape or list(raw.columns) != list(df.columns):
            raw = None
    except Exception:
        raw = None
    if raw is not None:
        df = _exact_decimal_columns(_restore_csv_text(_restore_na_tokens(df, raw), raw, decimal), raw, decimal)
    return df, header_unit, raw



def _guard_csv_width(path: Path, encoding: str, kwargs: dict[str, Any]) -> int | None:
    """Use the existing readers to reject a later physical record wider than the first."""
    options = {k: v for k, v in kwargs.items() if k not in {"header", "usecols", "index_col", "engine", "na_values", "keep_default_na", "skip_blank_lines"}}
    options.update(header=None, dtype=str, na_filter=False, keep_default_na=False, skip_blank_lines=True)
    for engine in ("python", "c"):
        try:
            return pd.read_csv(path, encoding=encoding, engine=engine, **options).shape[1]
        except UnicodeDecodeError:
            raise
        except pd.errors.ParserError as error:
            if re.search(r"Expected \d+ fields", str(error)):
                raise LoaderError(str(error)) from error
        except Exception:
            pass  # No structural verdict: preserve the reader's existing fallback (for example, large fields).
    return None


def _restore_csv_text(df: pd.DataFrame, raw: pd.DataFrame, decimal: str) -> pd.DataFrame:
    """A decimal view must not replace original string/category lexemes."""
    if decimal == ".":
        return df
    out = df
    for column in df.columns:
        current = df[column]
        if column not in raw.columns or not isinstance(current, pd.Series):
            continue
        strings = current.map(lambda value: isinstance(value, str))
        if strings.any():
            if out is df:
                out = df.copy()
            out[column] = current.where(~strings, raw[column])
    return out


def _exact_decimal_columns(df: pd.DataFrame, raw: pd.DataFrame, decimal: str) -> pd.DataFrame:
    """Correct inferred float64 cells from their original tokens before NA fill."""
    out = df
    for column in df.columns:
        series = df[column]
        if series.dtype.kind != "f" or series.dtype.itemsize != 8 or column not in raw.columns:
            continue
        tokens = raw[column]
        if not isinstance(tokens, pd.Series):
            continue
        values = series.to_numpy(dtype=np.float64, copy=True)
        changed = False
        for row, text in enumerate(tokens.tolist()):
            if np.isnan(values[row]) or not isinstance(text, str):
                continue
            token = text.strip()
            if decimal != ".":
                token = token.replace(decimal, ".")
            if _INTEGER_TEXT.fullmatch(token) or _DECIMAL_TEXT.fullmatch(token):
                exact = float(token)
                if struct.pack(">d", exact) != struct.pack(">d", values[row]):
                    values[row] = exact
                    changed = True
        if changed:
            if out is df:
                out = df.copy()
            out[column] = values
    return out


def _restore_na_tokens(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Missing tokens pandas left as text stay missing.

    A column pandas can only read as text after a numeric attempt failed (e.g. ``[18446744073709551615, ""]``: the
    integer overflows the signed parse and the fallback keeps every token verbatim) keeps its ``""`` / ``NA`` tokens as
    strings, so the NA audit and every later missing-value rule would never see them. Only a token the reader's own NA
    vocabulary names, in a non-numeric column, is turned back into a missing cell.
    """
    out = df
    for column in df.columns:
        series = df[column]
        if series.dtype.kind in "biufc" or column not in raw.columns:
            continue
        tokens = raw[column]
        if not isinstance(tokens, pd.Series):
            continue
        stale = (tokens.isin(_PANDAS_NA_TOKENS) & series.notna()).to_numpy(dtype=bool)
        if stale.any():
            if out is df:
                out = df.copy()
            out[column] = series.mask(stale)
    return out


def _read_numpy(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, str]:
    fmt = params.format.values
    obj = np.load(path, allow_pickle=bool(fmt.get("allow_pickle", False)))
    if path.suffix.lower() == ".npz":
        key = fmt.get("key") or fmt.get("variable")
        arr = obj[key] if key else obj[list(obj.keys())[0]]
    else:
        arr = obj
    arr = np.asarray(arr)
    arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
    unit = params.header_unit.value if params.header_unit else "index"
    headers = [str(i) for i in range(arr.shape[1])] if unit == "index" else [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=headers), unit


def _read_parquet(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, str]:
    fmt = params.format.values
    read_kwargs: dict[str, Any] = {}
    if fmt.get("columns") is not None:
        read_kwargs["columns"] = fmt["columns"]
    _refuse_oversized_unsigned(path, read_kwargs.get("columns"))
    df = pd.read_parquet(path, **read_kwargs)
    df = _exact_nullable_integers(path, df, read_kwargs.get("columns"))
    df.columns = [str(c) for c in df.columns]
    return df, params.header_unit.value if params.header_unit else "text"


def _refuse_oversized_unsigned(path: Path, columns: list[str] | None) -> None:
    """Refuse a UInt64 cell above ``i64::MAX`` in ANY column the reader actually reads (any role).

    The Parquet table is an i64 integer domain (the owner converts every integer cell to a signed 64-bit cell);
    the check runs on the typed Arrow cells, before any pandas coercion could round or retype them. A column
    excluded by ``format.columns`` is never read and never inspected. Public JSON ``u64``, generic ``uint64``
    arrays and ``.npy`` files are different readers with their own, unchanged domain.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    names = [f.name for f in pq.read_schema(path) if pa.types.is_uint64(f.type) and (columns is None or f.name in columns)]
    if not names:
        return
    table = pq.read_table(path, columns=names)
    for name in names:
        largest = pc.max(table.column(name)).as_py()
        if largest is not None and largest > 2**63 - 1:
            raise LoaderError(f"parquet column {name!r} contains integer value {largest} outside i64 range")


def _exact_nullable_integers(path: Path, df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    """Keep nullable integer columns exact.

    ``to_pandas`` turns an integer column that holds nulls into float64 (NaN) and rounds every integer above
    ``2**53`` before any role selection can see it. Such columns are rebuilt from the original Arrow cells as
    pandas' nullable ``Int64`` / ``UInt64``; non-nullable integers and genuine floating columns are untouched.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    integer_names = [f.name for f in pq.read_schema(path) if pa.types.is_integer(f.type) and f.name in df.columns and (columns is None or f.name in columns)]
    if not integer_names:
        return df
    table = pq.read_table(path, columns=integer_names)
    out = df
    for name in integer_names:
        column = table.column(name)
        if column.null_count == 0 or df[name].dtype.kind != "f":
            continue
        dtype = pd.UInt64Dtype() if pa.types.is_unsigned_integer(column.type) else pd.Int64Dtype()
        if out is df:
            out = df.copy()
        out[name] = pd.Series(pd.array(column.to_pylist(), dtype=dtype), index=df.index)
    return out


def _boxed_excel_cell(cell: Any) -> tuple[str, Any]:
    """Keep hash-equal physical booleans and integers distinct during pandas object inference."""
    return type(cell).__name__, cell


def _restore_excel_floats(path: Path, kwargs: dict[str, Any], df: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Carry physical Float provenance through a borrowed read-only selected view.

    The private openpyxl view is exercised with openpyxl 3.1.5 and pandas
    3.0.5/3.1.0. It never binds merges or builds a normal worksheet rectangle.
    """
    if not any(isinstance(cell, (int, np.integer)) and not isinstance(cell, (bool, np.bool_)) and abs(int(cell)) >= 2**53 for column in raw for cell in raw[column]):
        return
    from copy import copy

    from openpyxl.cell.read_only import ReadOnlyCell

    try:
        with pd.ExcelFile(path) as excel:
            if excel.engine != "openpyxl":
                return
            book = excel.book
            selection = kwargs["sheet_name"]
            source = book[selection] if isinstance(selection, str) else book.worksheets[selection]
            view, sheet = copy(book), copy(source)
            view._sheets = [sheet if candidate is source else candidate for candidate in book._sheets]
            view.close = lambda: None  # Borrowed archive: only ExcelFile owns its lifetime.
            sheet.reset_dimensions()
            strings: list[str] = []
            physical: dict[tuple[int, int], float] = {}
            for row in sheet.rows:
                for cell in row:
                    value = cell.value
                    if isinstance(value, str):
                        strings.append(value)
                    elif cell.data_type == "n" and type(value) is float and value.is_integer() and abs(value) >= 2**53:
                        physical[(cell.row, cell.column)] = value
            if not physical:
                return
            namespace = "nirs4allfloat" + uuid4().hex
            while any(value.startswith(namespace) for value in strings):
                namespace += "x"
            tags = {coordinate: namespace + "v" + str(index) for index, coordinate in enumerate(physical)}
            registry = {tags[coordinate]: value for coordinate, value in physical.items()}
            original_rows = sheet._cells_by_row

            def tagged_rows(min_col: int, min_row: int, max_col: int | None, max_row: int | None, values_only: bool = False) -> Any:
                for row in original_rows(min_col, min_row, max_col, max_row, False):
                    selected = []
                    for cell in row:
                        coordinate = (getattr(cell, "row", None), getattr(cell, "column", None))
                        if coordinate in tags:
                            cell = ReadOnlyCell(sheet, cell.row, cell.column, tags[coordinate], "s")
                        selected.append(cell.value if values_only else cell)
                    yield tuple(selected)

            sheet._cells_by_row = tagged_rows
            tagged = pd.read_excel(view, engine="openpyxl", **kwargs, dtype=object, na_filter=False, keep_default_na=False)
            if tagged.shape != raw.shape or tagged.shape != df.shape:
                raise LoaderError(f"physical Excel selection shape differs: {tagged.shape} != {raw.shape}")
            for position, column in enumerate(df.columns):
                label = str(tagged.columns[position])
                if label != column and label not in registry:
                    raise LoaderError(f"physical Excel selection column differs at position {position}")
                selected = tagged.iloc[:, position].tolist()
                if not any(isinstance(cell, str) and cell in registry for cell in selected):
                    continue
                original = raw.iloc[:, position].tolist()
                # Cell::Str includes decoded dates/times/durations, not only Python str.
                text_peer = any(not pd.isna(cell) and not (isinstance(cell, str) and cell in _PANDAS_NA_TOKENS) and not isinstance(cell, (int, float, bool, np.number)) for cell in original)
                restored: list[Any] = []
                for tag, cell in zip(selected, original, strict=True):
                    if isinstance(tag, str) and tag in registry:
                        cell = registry[tag]
                    elif not text_peer and isinstance(cell, (bool, np.bool_)):
                        cell = float(cell)
                    elif not text_peer and isinstance(cell, (int, np.integer)):
                        try:
                            integer = check_integer(int(cell))
                        except NumberDomainError:
                            pass  # The existing selected-cell boundary owns this refusal.
                        else:
                            if float(integer) == integer:
                                cell = float(integer)
                    restored.append(cell)
                missing = df[column].isna().tolist()
                raw[column] = pd.Series(restored, index=raw.index, dtype=object)
                # Pandas can blank a real Bool. Only physical missing/non-numeric
                # NA Text stays missing; never erase a decoded non-Text cell.
                df[column] = pd.Series([None if absent and (isinstance(cell, str) or pd.isna(cell)) else cell for cell, absent in zip(restored, missing, strict=True)], index=df.index, dtype=object)
    except LoaderError:
        raise
    except Exception as error:
        raise LoaderError(f"physical Excel provenance read failed: {error}") from error


def _read_excel(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    fmt = params.format.values
    has_header = True if params.has_header is None else params.has_header
    kwargs: dict[str, Any] = {"sheet_name": fmt.get("sheet_name", 0), "header": 0 if has_header else None}
    if fmt.get("usecols") is not None:
        kwargs["usecols"] = fmt["usecols"]
    if fmt.get("skiprows") is not None:
        kwargs["skiprows"] = fmt["skiprows"]
    df = pd.read_excel(path, **kwargs)
    if isinstance(df, dict):
        kwargs["sheet_name"], df = next(iter(df.items()))
    labels = list(df.columns)
    df.columns = [str(c) for c in df.columns]
    # the same sheet again with the original cell values (text cells stay text, numeric cells keep their physical
    # value): pandas' text parser turns a text cell such as ``1e-400`` into 0.0 and rounds a large integer text
    try:
        raw_kwargs = {key: value for key, value in kwargs.items() if key not in {"na_filter", "keep_default_na", "na_values"}}
        converters = {label: _boxed_excel_cell for label in labels}
        converters.update({position: _boxed_excel_cell for position in range(len(labels))})
        raw = pd.read_excel(path, **raw_kwargs, converters=converters, na_filter=False, keep_default_na=False)
        raw = pd.DataFrame({column: pd.Series([cell[1] for cell in raw[column]], index=raw.index, dtype=object) for column in raw.columns}, index=raw.index)
        raw.columns = [str(c) for c in raw.columns]
        if raw.shape != df.shape or list(raw.columns) != list(df.columns):
            raw = None
    except Exception:
        raw = None
    if raw is not None:
        for column in df.columns:
            missing = df[column].isna()
            unrecognised = raw[column].map(lambda cell: isinstance(cell, str) and cell in _PANDAS_NA_TOKENS) & ~missing
            if unrecognised.any():
                df[column] = pd.Series([None if absent else cell for cell, absent in zip(df[column].astype(object), unrecognised, strict=True)], index=df.index, dtype=object)
                missing = missing | unrecognised
            cells = raw.loc[~missing, column].tolist()
            if any((isinstance(cell, str) and _is_exact_number(value)) or (df[column].dtype == object and isinstance(cell, (bool, np.bool_)) != isinstance(value, (bool, np.bool_)) and not isinstance(cell, str) and not isinstance(value, str)) for cell, value in zip(cells, df.loc[~missing, column].tolist(), strict=True)):
                df[column] = pd.Series([None if is_missing and isinstance(cell, str) else cell for cell, is_missing in zip(raw[column], missing, strict=True)], index=df.index, dtype=object)
            elif missing.any() and cells and all(isinstance(cell, (bool, np.bool_)) for cell in cells):
                df[column] = pd.Series([None if is_missing else bool(cell) for cell, is_missing in zip(raw[column], missing, strict=True)], index=df.index, dtype=object)
        _restore_excel_floats(path, kwargs, df, raw)
    return df, params.header_unit.value if params.header_unit else "text", raw


def _read_vendor(path: Path, params: LoadingParams) -> tuple[pd.DataFrame, str]:
    """Decode a vendor spectroscopy file through nirs4all-formats (never reparsed here)."""
    try:
        import nirs4all_formats as nfmt  # lazy: optional dependency
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise LoaderError(f"reading vendor file {path.name!r} requires the 'nirs4all-formats' package (pip install nirs4all-io[formats])") from exc

    open_recordset = getattr(nfmt, "open_recordset", None)
    if not callable(open_recordset):
        raise LoaderError("installed 'nirs4all-formats' package does not expose open_recordset()")
    try:
        records = open_recordset(str(path))
    except Exception as exc:
        raise LoaderError(f"nirs4all-formats could not read vendor file {path.name!r}: {exc}") from exc
    if not hasattr(records, "to_pandas"):
        raise LoaderError("nirs4all-formats open_recordset() returned an object without to_pandas()")

    df = records.to_pandas()
    df.columns = [str(c) for c in df.columns]
    spectral_cols = [c for c in df.columns if c.startswith(_VENDOR_FEATURE_PREFIX)]
    if not spectral_cols:
        spectral_cols = [
            c
            for c in df.columns
            if c not in _VENDOR_META_COLUMNS
            and not any(c.startswith(prefix) for prefix in _VENDOR_META_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not spectral_cols:
        raise LoaderError(f"nirs4all-formats produced no spectral columns for {path.name!r}")
    df = df.loc[:, spectral_cols].copy()
    unit = params.header_unit.value if params.header_unit else "nm"
    return df, unit


def read_table(path: str | Path, params: LoadingParams | None = None) -> LoadedTable:
    """Read a tabular/vendor file into a :class:`LoadedTable` (dtypes + NA policy applied)."""
    path = Path(path)
    params = params or LoadingParams()
    if not path.exists():
        raise LoaderError(f"file not found: {path}")
    ext = _ext(path)
    raw: pd.DataFrame | None = None
    if ext in _CSV_EXTS or ext in _CSV_COMPOUND or (ext in {".gz", ".zip"} and ".csv" in path.name.lower()):
        df, unit, raw = _read_csv(path, params)
    elif ext in {".npy", ".npz"}:
        df, unit = _read_numpy(path, params)
    elif ext in {".parquet", ".pq"}:
        df, unit = _read_parquet(path, params)
    elif ext in {".xlsx", ".xls", ".xlsm"}:
        df, unit, raw = _read_excel(path, params)
    elif ext in _VENDOR_HINT_EXTS:
        df, unit = _read_vendor(path, params)
    else:
        # unknown extension: try CSV (nirs4all fallback), then vendor formats.
        try:
            df, unit, raw = _read_csv(path, params)
        except Exception:
            df, unit = _read_vendor(path, params)
    source_dtypes = _declared_dtypes(path, ext, df)
    df, na_report = apply_na_policy(df, params.na)
    return LoadedTable(
        df=df,
        headers=list(df.columns),
        header_unit=unit,
        dtypes=infer_dtypes(df),
        na_report=na_report,
        source=str(path),
        source_dtypes=source_dtypes,
        raw=raw if raw is not None and len(raw) == len(df) else None,
        decimal="." if ext in {".xlsx", ".xls", ".xlsm"} else params.decimal_separator or ".",
        text_tokens=raw is not None and ext not in {".xlsx", ".xls", ".xlsm"},
    )


def _arrow_dtype(arrow_type: Any) -> str:
    import pyarrow as pa

    if pa.types.is_float32(arrow_type):
        return "float32"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_integer(arrow_type):
        return f"{'u' if pa.types.is_unsigned_integer(arrow_type) else ''}int{arrow_type.bit_width}"
    return "bool" if pa.types.is_boolean(arrow_type) else "object"


def _declared_dtypes(path: Path, ext: str, df: pd.DataFrame) -> dict[str, str]:
    """Storage dtypes a typed file genuinely declares (numpy / parquet only)."""
    if ext in {".npy", ".npz"}:
        return {str(column): ("bool" if dtype.kind == "b" else dtype.name if dtype.kind in "fiu" else "object") for column, dtype in df.dtypes.items()}
    if ext in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        return {field.name: _arrow_dtype(field.type) for field in schema if field.name in df.columns}
    return {}


# --------------------------------------------------------------------------- #
# Role coercion helpers (used by the builder when splitting columns by role)  #
# --------------------------------------------------------------------------- #
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+\Z")
_DECIMAL_TEXT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_NONFINITE_TEXT = re.compile(r"[+-]?(?:inf|infinity)\Z", re.IGNORECASE)


def restore_numeric_cells(df: pd.DataFrame, raw: pd.DataFrame | None, columns: list[str], decimal: str | list[str] = ".", *, defer: bool = False, segments: list[int] | None = None, delimited: list[bool] | None = None) -> pd.DataFrame:
    """Inspect the original cells of the NUMERIC-ROLE columns of a CSV / spreadsheet before they are converted.

    Pandas already parsed these columns, which is lossy in two ways the selector can no longer see:
    a decimal that underflows to zero (``1e-400``) and an integer above ``2**53`` inside a column that also
    holds floats or NA. Every numeric-looking token goes through the shared checked primitives
    (``decimal_float`` / ``check_integer``); a column whose integer cells float64 cannot hold is rebuilt as
    exact Python numbers so the role's storage selection decides (direct integer→F32, exact-F64 gate).
    Metadata/categorical columns are never passed here, and non-numeric tokens (NA, labels) keep pandas' value.

    With ``defer`` an out-of-domain integer or an overflowing/underflowing decimal is not raised here: it becomes a
    :class:`NumericRefusal` cell of an object column (the classification of the original column is unchanged) and
    :func:`refuse_numeric_cells` raises it only if its row survives NA row removal. A removed cell never refuses.
    ``segments`` (the stacked inputs' row counts of a ``concat_samples`` source) additionally gives every loaded input its
    own numeric authority: a numeric-only input's refused token is kept as a marker even when a text input made the merged
    column non-numeric (a text input cannot erase it).
    """
    if raw is None:
        return df
    out = df
    for column in columns:
        if column not in raw.columns or column not in df.columns:
            continue
        current, tokens = df[column], raw[column]
        if defer and isinstance(current, pd.Series) and isinstance(tokens, pd.Series) and current.dtype.kind not in "fiu" and len(tokens) == len(current):
            marked = _mark_stacked_refusals(column, current, tokens, decimal, segments, delimited)
            if marked is not None:
                if out is df:
                    out = df.copy()
                out[column] = marked
            continue
        if not isinstance(current, pd.Series) or not isinstance(tokens, pd.Series) or current.dtype.kind not in "fiu" or len(tokens) != len(current):
            continue
        cells: list[Any] = []
        refused = False
        large = False
        row_decimals = _row_decimals(decimal, segments, len(current))
        integer_rows: set[int] = set()
        bounds = _segment_bounds(segments, len(current))
        if len(bounds) > 1:
            missing = current.isna().tolist()
            for (start, stop), is_delimited in zip(bounds, delimited or [True] * len(bounds), strict=True):
                if not is_delimited:
                    continue
                part = tokens.iloc[start:stop].tolist()
                if not part or not all(isinstance(token, str) for token in part):
                    continue
                original = _numeric_text_cells(column, part, missing[start:stop], row_decimals[start])
                if original is not None and not _has_float_peer(original):
                    # Promotion by another input does not give an integer-only input a signed zero.
                    integer_rows.update(range(start, stop))
        for row, (token, value) in enumerate(zip(tokens.tolist(), current.tolist(), strict=True)):
            if isinstance(token, (bool, np.bool_)):
                cells.append(value)
                continue
            if isinstance(token, (float, np.floating)):
                # a physical (already decoded) numeric spreadsheet cell: its bits are the value
                cells.append(value if token != token else float(token))
                continue
            if isinstance(token, (int, np.integer)):
                text = str(int(token))
            elif isinstance(token, str):
                text = token.strip()
            else:
                cells.append(value)
                continue
            separator = row_decimals[row]
            lexeme = str(token).replace(separator, ".") if separator != "." else str(token)
            if separator != ".":
                text = text.replace(separator, ".")
            try:
                if _INTEGER_TEXT.fullmatch(text):
                    number = check_integer(int(text))
                    large = large or abs(number) > 2**53
                    cells.append(-0.0 if number == 0 and text.startswith("-") and row not in integer_rows else number)
                elif _DECIMAL_TEXT.fullmatch(text):
                    cells.append(decimal_float(text))
                else:
                    cells.append(value)
            except NumberDomainError as error:
                if not defer:
                    raise
                refused = True
                cells.append(NumericRefusal(column, text, str(error), bool(_INTEGER_TEXT.fullmatch(text)), lexeme))
        if refused:
            if out is df:
                out = df.copy()
            out[column] = pd.Series(cells, index=df.index, dtype=object)
        elif large and current.dtype.kind == "f":
            if out is df:
                out = df.copy()
            out[column] = pd.Series(cells, index=df.index, dtype=object)
        elif current.dtype.kind == "f":
            # a genuine zero's sign is part of the value: some pandas versions parse "-0.000e-999" as +0.0
            fixed = current.to_numpy(dtype=np.float64, copy=True)
            changed = False
            for position, (cell, value) in enumerate(zip(cells, fixed.tolist(), strict=True)):
                if isinstance(cell, float) and cell == 0.0 and value == 0.0 and np.signbit(cell) != np.signbit(value):
                    fixed[position] = cell
                    changed = True
            if changed:
                if out is df:
                    out = df.copy()
                out[column] = fixed
    return out


@dataclass(frozen=True)
class NumericRefusal:
    """A numeric cell (selected metadata/ID column, or a deferred numeric-role cell) the integer/decimal domain refuses.

    It rides the frame through joins untouched; the refusal is raised only if the cell's row is still part of the
    selected block (after row removal), by :func:`refuse_numeric_cells`. ``integer`` tells an out-of-domain INTEGER token
    (which selects the retained-integer mode of its column) from an overflowing/underflowing DECIMAL (which never does).
    """

    column: str
    token: str
    reason: str
    integer: bool
    lexeme: str | None = None


def restore_numeric_metadata(df: pd.DataFrame, raw: pd.DataFrame | None, columns: list[str], decimal: str | list[str] = ".", segments: list[int] | None = None, typed: pd.DataFrame | None = None, *, delimited: bool | list[bool] = True) -> pd.DataFrame:
    """Numeric-authority rule for SELECTED metadata/ID columns of a delimited-text file.

    The column is classified ONCE per LOADED INPUT, here, over every original row of that input (rows that NA removal later
    drops included); the survivors are never reclassified, and only a surviving cell can refuse (:func:`refuse_numeric_cells`).
    ``segments`` are the row counts of the inputs a ``concat_samples`` source stacked and ``typed`` their cells as each input
    loaded them: each input is classified on its own tokens before the rows are joined (a non-numeric input keeps its own
    typed cells, ``TRUE`` stays ``True``), so a float input contributes ``4.0`` even when another input retains huge integers,
    and a text input cannot erase the numeric authority of another input's out-of-domain token (and vice versa).
    A column (input) whose every non-NA token is numeric text follows the rule below; any other keeps its literal
    strings, uncapped, even when its text peer row is later removed. (Integers live in the shared domain
    ``[-2**63, 2**64 - 1]``; a decimal token is a float token.)

    * all tokens integers and no NA: exact integers (pandas already holds them);
    * no integer trigger: the plain float column of the correctly rounded token values (F64-exact integers beside a
      float/NA/decimal peer become floats);
    * RETAINED-INTEGER mode, triggered only by an INTEGER token outside the domain, or an in-domain integer F64 cannot
      hold beside a float/NA/decimal peer: integer tokens stay exact integers, float tokens stay floats;
    * an out-of-domain integer or an overflowing/underflowing decimal becomes a :class:`NumericRefusal` cell.

    The original lexemes are inspected (never a float cast of the pandas column), so ``+0009007199254740993`` is the
    integer ``9007199254740993`` and ``[2**53 + 1, 1.5]`` keeps two distinct values.
    """
    if raw is None or delimited is False:
        return df
    out = df
    for column in columns:
        if column not in raw.columns or column not in df.columns:
            continue
        current, tokens = df[column], raw[column]
        if not isinstance(current, pd.Series) or not isinstance(tokens, pd.Series) or len(tokens) != len(current):
            continue
        texts = tokens.tolist()
        missing = current.isna().tolist()
        bounds = _segment_bounds(segments, len(texts))
        if len(bounds) > 1:
            replacement = _stacked_inputs(column, texts, missing, decimal if isinstance(decimal, str) else decimal[0], bounds, df.index, typed[column] if typed is not None and column in typed.columns else current, delimited if isinstance(delimited, list) else None) if isinstance(decimal, str) or len(set(decimal)) == 1 else _stacked_metadata(column, texts, missing, decimal, bounds, df.index, typed[column] if typed is not None and column in typed.columns else current, delimited if isinstance(delimited, list) else None)
            if replacement is None:
                continue
        else:
            if not all(isinstance(text, str) for text in texts):
                continue
            cells = _numeric_text_cells(column, texts, missing, decimal if isinstance(decimal, str) else decimal[0])
            if cells is None:
                continue
            mode, resolved = _typed_cells(cells, texts)
            if mode == "integer":
                continue  # all integers (pandas holds them exactly)
            if mode == "float" and not any(isinstance(c, NumericRefusal) for c in resolved):
                values = np.array(resolved, dtype=np.float64)
                if current.dtype.kind == "f":
                    held = current.to_numpy(dtype=np.float64)
                    if np.array_equal(held, values, equal_nan=True) and np.array_equal(np.signbit(held), np.signbit(values)):
                        continue
                replacement = pd.Series(values, index=df.index)
            else:
                replacement = pd.Series(resolved, index=df.index, dtype=object)
        if out is df:
            out = df.copy()
        out[column] = replacement
    return out


def _row_decimals(decimal: str | list[str], segments: list[int] | None, length: int) -> list[str]:
    if isinstance(decimal, str):
        return [decimal] * length
    if not segments:
        return [decimal[0]] * length if len(decimal) == 1 else decimal
    bounds = _segment_bounds(segments, length)
    return [separator for separator, (start, stop) in zip(decimal, bounds, strict=True) for _ in range(start, stop)]


def _mark_stacked_refusals(column: str, current: pd.Series, tokens: pd.Series, decimal: str | list[str], segments: list[int] | None, delimited: list[bool] | None = None) -> pd.Series | None:
    """Keep checked numeric cells and literal spreadsheet zero signs in object columns."""
    texts = tokens.tolist()
    bounds = _segment_bounds(segments, len(texts))
    flags = delimited or [True] * len(bounds)
    if len(bounds) < 2 and all(flags):
        return None
    separators = [decimal] * len(bounds) if isinstance(decimal, str) else decimal
    missing = current.isna().tolist()
    marks: dict[int, Any] = {}
    for (start, stop), separator, is_delimited in zip(bounds, separators, flags, strict=True):
        part = texts[start:stop]
        if not all(isinstance(text, str) for text in part):
            continue
        cells = _numeric_text_cells(column, part, missing[start:stop], separator)
        if cells is None:
            continue
        mode, resolved = _typed_cells(cells, part)
        if not is_delimited:
            # A physical text cell has no CSV integer-column authority, including in a single sheet.
            resolved = [-0.0 if type(cell) is int and cell == 0 and text.strip().startswith("-") else cell for cell, text in zip(resolved, part, strict=True)]
        for offset, cell in enumerate(resolved):
            negative_zero = type(cell) is float and cell == 0.0 and np.signbit(cell)
            if isinstance(cell, NumericRefusal) or (mode == "retained" and (type(cell) is int or negative_zero)) or (not is_delimited and negative_zero):
                marks[start + offset] = cell
    if not marks:
        return None
    marked = current.astype(object).copy()
    for position, marker in marks.items():
        marked.iloc[position] = marker
    return marked


def _stacked_metadata(column: str, texts: list[Any], missing: list[bool], decimals: list[str], bounds: list[tuple[int, int]], index: pd.Index, typed: pd.Series, delimited: list[bool] | None = None) -> pd.Series | None:
    """The existing per-input authority rule with distinct input decimal characters."""
    # Normalize the temporary numeric view, while nonnumeric inputs retain their already typed cells.
    view = texts.copy()
    flags = delimited or [True] * len(bounds)
    for (start, stop), separator, is_delimited in zip(bounds, decimals, flags, strict=True):
        if is_delimited and separator != ".":
            view[start:stop] = [text.replace(separator, ".") for text in texts[start:stop]]
    return _stacked_inputs(column, view, missing, ".", bounds, index, typed, flags)


def restore_numeric_keys(df: pd.DataFrame, raw: pd.DataFrame | None, columns: list[str], decimal: str, text_tokens: bool) -> pd.DataFrame:
    """Restore admitted typed authority for every relational key, independent of its payload role."""
    if raw is None:
        return df
    out = df
    for column in columns:
        if column not in raw.columns or column not in df.columns:
            continue
        current, tokens = df[column], raw[column]
        if len(tokens) != len(current):
            continue
        if text_tokens:
            texts = tokens.tolist()
            if not all(isinstance(text, str) for text in texts):
                continue
            cells = _numeric_text_cells(column, texts, current.isna().tolist(), decimal)
            if cells is None:
                continue
            mode, resolved = _typed_cells(cells, texts)
            if mode == "float" and not any(isinstance(cell, NumericRefusal) for cell in resolved):
                replacement = pd.Series(resolved, index=df.index, dtype=np.float64)
            elif mode == "integer" and all(type(cell) is int for cell in resolved):
                low, high = min(resolved, default=0), max(resolved, default=0)
                dtype = object if any(float(cell) != cell for cell in resolved) else np.int64 if -(2**63) <= low and high < 2**63 else np.uint64 if low >= 0 and high < 2**64 else object
                replacement = pd.Series(resolved, index=df.index, dtype=dtype)
            else:
                replacement = pd.Series(resolved, index=df.index, dtype=object)
        else:
            resolved = [None if missing else original if isinstance(original, str) and not isinstance(held, (bool, np.bool_)) else held for original, held, missing in zip(tokens.tolist(), current.tolist(), current.isna(), strict=True)]
            replacement = pd.Series(resolved, index=df.index, dtype=object)
        if out is df:
            out = df.copy()
        out[column] = replacement
    return out


def _segment_bounds(segments: list[int] | None, total: int) -> list[tuple[int, int]]:
    if not segments or sum(segments) != total:
        return [(0, total)]
    bounds, start = [], 0
    for count in segments:
        bounds.append((start, start + count))
        start += count
    return bounds


def _has_float_peer(cells: list[Any]) -> bool:
    """Whether one original input contains a float, NA, or refused decimal peer."""
    return any(type(c) is float or c is None or (isinstance(c, NumericRefusal) and not c.integer) for c in cells)


def _typed_cells(cells: list[Any], texts: list[str]) -> tuple[str, list[Any]]:
    """Mode and final cells of ONE classified numeric input (``None`` in ``cells`` = NA, floats may carry refusals).

    ``integer``: only integers; ``float``: the plain float cells (a refused decimal stays a marker); ``retained``: the
    retained-integer mode. Only an INTEGER token triggers the last one.
    """
    refusals = [c for c in cells if isinstance(c, NumericRefusal)]
    # a float peer is a float token, a missing cell or a decimal that overflows/underflows
    has_peer = _has_float_peer(cells)
    if any(c.integer for c in refusals) or (has_peer and any(float(n) != n for n in cells if type(n) is int)):
        return "retained", [np.nan if c is None else -0.0 if has_peer and type(c) is int and c == 0 and text.strip().startswith("-") else c
                            for c, text in zip(cells, texts, strict=True)]
    if has_peer:
        # The values are the correctly rounded token values (pandas' fast float parser is not guaranteed to round
        # correctly); a true '-0' keeps its sign.
        return "float", [c if isinstance(c, NumericRefusal) else np.nan if c is None else (-0.0 if type(c) is int and c == 0 and text.strip().startswith("-") else float(c)) for c, text in zip(cells, texts, strict=True)]
    return "integer", cells


def _stacked_inputs(column: str, texts: list[Any], missing: list[bool], decimal: str, bounds: list[tuple[int, int]], index: pd.Index, typed: pd.Series, delimited: list[bool] | None = None) -> pd.Series | None:
    """The column of a ``concat_samples`` source, each loaded input classified on its OWN tokens (``None``: keep the merged column).

    Per-input cells replace the merged column only where an input needs its own authority: a retained or refused token, an
    integer outside the F64/int64 domain, a text input or booleans. When every input is numeric or all-NA and none needs that,
    the merged column already holds the values (``[3, 4]`` beside ``[1.5, 2.5]`` is the float column ``3.0, 4.0, 1.5, 2.5``).
    """
    final: list[Any] = []
    numeric_input = False
    plain = True
    all_na_input = False
    typed_bool = False
    for (start, stop), is_delimited in zip(bounds, delimited or [True] * len(bounds), strict=True):
        part_texts, part_missing = texts[start:stop], missing[start:stop]
        cells = _numeric_text_cells(column, part_texts, part_missing, decimal) if is_delimited else None
        if cells is None:
            # A non-numeric input (genuine text, booleans, all-NA) keeps its OWN typed cells exactly as that input loaded them
            # (strings stay literal, ``TRUE`` stays the boolean ``True``): the stacked column is an object column, never
            # a string or float coercion of a typed input.
            part = typed.iloc[start:stop].tolist()
            final.extend(part)
            all_na_input = all_na_input or all(part_missing)
            binary_plain = not is_delimited and all(absent or (_is_exact_number(cell) and (type(cell) is not int or float(cell) == cell)) for cell, absent in zip(part, part_missing, strict=True))
            plain = plain and (all(part_missing) or binary_plain)
            typed_bool = typed_bool or any(isinstance(cell, (bool, np.bool_)) for cell in part)
            continue
        numeric_input = True
        mode, resolved = _typed_cells(cells, part_texts)
        # an integer F64 or int64 cannot hold, a refused token and the retained-integer mode are exact only in the per-input cells
        if mode == "retained" or any(isinstance(c, NumericRefusal) or (type(c) is int and (float(c) != c or not -(2**63) <= c < 2**63)) for c in resolved):
            plain = False
        final.extend(resolved)
    if numeric_input and plain:
        return None
    if numeric_input or (all_na_input and typed_bool):
        return pd.Series(final, index=index, dtype=object)
    return None



def _numeric_text_cells(column: str, texts: list[str], missing: list[bool], decimal: str) -> list[Any] | None:
    """Typed cells of a numeric-only text column (``None`` = NA), or ``None`` when it holds any non-numeric token."""
    cells: list[Any] = []
    numeric = 0
    for text, is_missing in zip(texts, missing, strict=True):
        if is_missing:
            cells.append(None)
            continue
        token = text.strip()
        if decimal != ".":
            token = token.replace(decimal, ".")
        try:
            if _INTEGER_TEXT.fullmatch(token):
                cells.append(check_integer(int(token)))
            elif _DECIMAL_TEXT.fullmatch(token):
                cells.append(decimal_float(token))
            elif _NONFINITE_TEXT.fullmatch(token):
                cells.append(NumericRefusal(column, text, "Non-finite numeric text", False, text.replace(decimal, ".") if decimal != "." else text))
            else:
                return None
        except NumberDomainError as error:
            cells.append(NumericRefusal(column, text, f"{error}; a numeric-only column is numeric authority, so the value cannot be kept as text", bool(_INTEGER_TEXT.fullmatch(token)), text.replace(decimal, ".") if decimal != "." else text))
        numeric += 1
    return cells if numeric else None


def refuse_numeric_cells(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise for any surviving :class:`NumericRefusal` cell of the selected metadata/ID and numeric-role columns.

    A float-mode column kept as an object column only to carry such markers becomes a plain float64 column once the
    survivors hold no marker (a removed row never refuses and never changes the type of a surviving cell).
    """
    for column in columns:
        series = frame[column]
        cells = series.tolist()
        for cell in cells:
            if isinstance(cell, NumericRefusal):
                raise NumberDomainError(f"column {cell.column!r}: value {cell.token!r}: {cell.reason}")
            if isinstance(cell, (float, np.floating)) and np.isinf(cell):
                raise NumberDomainError(f"column {column!r}: Observed numeric values must be finite")
        if series.dtype == object and cells and all(type(cell) is float for cell in cells):
            frame[column] = series.astype(np.float64)


def _is_exact_number(cell: Any) -> bool:
    return isinstance(cell, (int, float, np.integer, np.floating)) and not isinstance(cell, (bool, np.bool_))


def _object_cells_to_ir(cells: np.ndarray, request: Any, label: Any, decimal: _Decimals = ".") -> np.ndarray:
    """An object column of numbers/text, cell by cell: integers stay integers until the selected conversion."""
    prepared = np.empty(cells.shape, dtype=object)
    for position, cell in np.ndenumerate(cells):
        if isinstance(cell, str):
            separator = decimal if isinstance(decimal, str) else decimal[position[0]]
            text = cell.strip().replace(separator, ".") if separator != "." else cell.strip()
            if _INTEGER_TEXT.fullmatch(text):
                number = check_integer(int(text))
                cell = -0.0 if number == 0 and text.startswith("-") else number
            elif _DECIMAL_TEXT.fullmatch(text):
                cell = decimal_float(text)
            else:
                cell = pd.to_numeric(pd.Series([cell]), errors="coerce").iloc[0]
        elif not _is_exact_number(cell):
            cell = pd.to_numeric(pd.Series([cell]), errors="coerce").iloc[0]
        prepared[position] = cell
    try:
        return to_ir_f64(prepared, request)
    except NumericContractError as error:
        raise LoaderError(f"column {label!r}: {error}") from error


def exact_float64(series: pd.Series, request: Any = None, decimal: _Decimals = ".") -> np.ndarray:
    """Exact F64 IR of a numeric column (non-numeric -> NaN) under the role's storage request.

    Integer columns convert to the selected storage directly (:func:`to_ir_f64`): an
    integer float64 cannot hold refuses only when F64 storage is selected, while an
    explicit float32 request rounds the original integer once.
    """
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        return _object_cells_to_ir(series.astype(object).where(series.notna(), None).to_numpy(), request, series.name, decimal)
    if isinstance(series.dtype, pd.api.extensions.ExtensionDtype) and series.dtype.kind in "iu":
        # nullable integers: the original cells (exact Python ints, None for NA) go to the role's selection
        return _object_cells_to_ir(series.astype(object).where(series.notna(), None).to_numpy(), request, series.name)
    if isinstance(series.dtype, pd.api.extensions.ExtensionDtype) and series.dtype.kind == "f":
        return series.to_numpy(dtype=np.float64, na_value=np.nan)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dtype.kind in "iu":
        try:
            return to_ir_f64(numeric.to_numpy(), request)
        except NumericContractError as error:
            raise LoaderError(f"column {series.name!r}: {error}") from error
    return numeric.to_numpy(dtype=np.float64)


def coerce_numeric(df: pd.DataFrame, request: Any = None, decimals: dict[str, _Decimals] | None = None) -> np.ndarray:
    """Coerce a numeric-role frame to exact float64 IR (non-numeric and temporal cells -> NaN)."""
    if df.shape[1] == 0:
        return np.empty((len(df), 0), dtype=np.float64)
    columns = []
    for i in range(df.shape[1]):
        series = df.iloc[:, i]
        if series.dtype.kind in "Mm":
            series = pd.Series(np.nan, index=series.index, name=series.name, dtype=np.float64)
        elif series.dtype == object:
            series = pd.Series([None if isinstance(cell, (datetime.date, datetime.time, datetime.timedelta, np.datetime64, np.timedelta64)) or cell is pd.NaT else cell for cell in series], index=series.index, name=series.name, dtype=object)
        columns.append(exact_float64(series, request, (decimals or {}).get(str(df.columns[i]), ".")))
    return np.column_stack(columns)


def encode_categorical(series: pd.Series, mode: str, decimal: _Decimals = ".") -> tuple[pd.Series, dict | None]:
    """Encode a target column. 'auto' factorizes non-numeric columns; else numeric coercion."""
    if series.dtype == object and any(_is_exact_number(cell) for cell in series.dropna()) and all(_is_exact_number(cell) for cell in series.dropna()):
        return series, None  # exact Python numbers (a CSV column that kept its large integers): never coerced through float64
    view = series
    separators = [decimal] * len(series) if isinstance(decimal, str) else decimal
    if any(separator != "." for separator in separators):
        view = pd.Series([cell.replace(separator, ".") if isinstance(cell, str) and separator != "." else cell for cell, separator in zip(series, separators, strict=True)], index=series.index)
    numeric = pd.to_numeric(view, errors="coerce")
    is_objectish = not pd.api.types.is_numeric_dtype(series.dtype)
    treat_categorical = mode == "auto" and is_objectish and numeric.isna().sum() > series.isna().sum()
    if treat_categorical:
        codes, categories = pd.factorize(series.astype(str).where(series.notna()))
        encoded = pd.Series(codes, index=series.index, dtype=float).mask(codes < 0)
        return encoded, {"categories": list(categories)}
    return (series if is_objectish else numeric), None
