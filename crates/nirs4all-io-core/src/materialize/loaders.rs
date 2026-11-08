// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Tabular loaders (ports the pure CSV-from-bytes path of `materialize/loaders.py`).
//!
//! Decodes delimited bytes into a typed [`Frame`] whose per-column dtype
//! inference mirrors pandas: a column is int64 only if every non-NA cell is an
//! integer literal and there is no NA; float64 if every non-NA cell parses as a
//! float (NA promotes int→float); object/string otherwise.
//!
//! This module is fs-free: it operates on bytes already read (and decompressed)
//! by the caller. The facade owns the file-read + gzip/zip decompression and
//! then delegates here, so the same decoder backs both the native (path) and the
//! in-memory paths.

use std::sync::LazyLock;

use crate::infer::table::{ColDtype, NumericKind};
use crate::spec::dataset_spec::{FormatParams, LoadingParams, NaConfig};
use crate::spec::enums::{FillMethod, NaPolicy};
use crate::spec::SpecError;
use serde_json::Value;

use super::frame::{Cell, Column, Frame};

/// Back-compat alias for the inference path.
pub type LoadedTable = Frame;

/// Streaming shape budgets applied while decoding one tabular byte source.
///
/// Record and field sizes count decoded CSV payload bytes (delimiters and CSV
/// quoting are not included). Rows exclude the optional header. The cell cap
/// covers both decoded data fields and the rectangular frame that will be
/// materialized, so ragged input cannot hide a large allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TabularReadLimits {
    pub max_record_bytes: u64,
    pub max_field_bytes: u64,
    pub max_rows: u64,
    pub max_columns: u64,
    pub max_cells: u64,
}

impl TabularReadLimits {
    /// Check a rectangular allocation before creating/copying its cells.
    pub fn check_shape(self, rows: u64, columns: u64) -> Result<(), SpecError> {
        if rows > self.max_rows {
            return Err(SpecError::new(format!(
                "load limit exceeded: rows (maximum {})",
                self.max_rows
            )));
        }
        if columns > self.max_columns {
            return Err(SpecError::new(format!(
                "load limit exceeded: columns (maximum {})",
                self.max_columns
            )));
        }
        let cells = rows
            .checked_mul(columns)
            .ok_or_else(|| SpecError::new("load limit exceeded: shape overflow"))?;
        if cells > self.max_cells {
            return Err(SpecError::new(format!(
                "load limit exceeded: cells (maximum {})",
                self.max_cells
            )));
        }
        Ok(())
    }

    pub const fn new(
        max_record_bytes: u64,
        max_field_bytes: u64,
        max_rows: u64,
        max_columns: u64,
        max_cells: u64,
    ) -> Self {
        Self {
            max_record_bytes,
            max_field_bytes,
            max_rows,
            max_columns,
            max_cells,
        }
    }
}

fn merge_na(base: &NaConfig, over: &NaConfig) -> NaConfig {
    if over.policy == NaPolicy::Auto && over.fill_method.is_none() {
        return base.clone();
    }
    NaConfig {
        policy: if over.policy != NaPolicy::Auto {
            over.policy
        } else {
            base.policy
        },
        fill_method: over.fill_method.or(base.fill_method),
        fill_value: if !over.fill_value.is_null() {
            over.fill_value.clone()
        } else {
            base.fill_value.clone()
        },
        fill_per_column: over.fill_per_column,
    }
}

/// Merge global + source loading params (source wins on explicitly-set fields).
pub fn effective_params(global: &LoadingParams, source: &LoadingParams) -> LoadingParams {
    let or_str = |o: &Option<String>, b: &Option<String>| {
        o.clone().filter(|s| !s.is_empty()).or_else(|| b.clone())
    };
    let mut fmt = global.format.values.clone();
    for (k, v) in &source.format.values {
        fmt.insert(k.clone(), v.clone());
    }
    LoadingParams {
        delimiter: or_str(&source.delimiter, &global.delimiter),
        decimal_separator: or_str(&source.decimal_separator, &global.decimal_separator),
        has_header: source.has_header.or(global.has_header),
        header_unit: source.header_unit.or(global.header_unit),
        signal_type: source.signal_type.or(global.signal_type),
        encoding: or_str(&source.encoding, &global.encoding),
        na: merge_na(&global.na, &source.na),
        categorical: source.categorical.or(global.categorical),
        format: FormatParams { values: fmt },
    }
}

/// pandas `keep_default_na=True` default set ∪ io's `_NA_VALUES`.
pub(crate) static NA_TOKENS: LazyLock<std::collections::HashSet<&'static str>> =
    LazyLock::new(|| {
        [
            "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
            "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null", "",
        ]
        .into_iter()
        .collect()
    });

fn is_na(cell: &str) -> bool {
    NA_TOKENS.contains(cell)
}

fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

fn infer_column(name: &str, raw: &[String], decimal: char) -> Column {
    let norm = |c: &str| super::frame::numeric_view(c, decimal).into_owned();
    let mut has_na = false;
    let mut all_int = true;
    let mut all_numeric = true;
    for cell in raw {
        if is_na(cell) {
            has_na = true;
            continue;
        }
        let n = norm(cell);
        let body = n.trim().trim_start_matches(['+', '-']);
        if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
            all_int = false;
        }
        if parse_float(&n).is_none() {
            all_numeric = false;
        }
    }

    if raw.iter().any(|cell| !is_na(cell))
        && raw
            .iter()
            .all(|cell| is_na(cell) || sheet_bool_word(cell).is_some())
    {
        return Column {
            numeric_text: vec![],
            source_dtype: None,
            name: name.into(),
            dtype: if has_na {
                ColDtype::String
            } else {
                ColDtype::Bool
            },
            numeric_kind: if has_na {
                NumericKind::NonNumeric
            } else {
                NumericKind::NonFloatNumeric
            },
            values: raw
                .iter()
                .map(|cell| sheet_bool_word(cell).map(Cell::Bool).unwrap_or(Cell::Na))
                .collect(),
        };
    }

    if all_numeric {
        let keep_integer_peers = raw.iter().filter(|c| !is_na(c)).any(|cell| {
            let text = norm(cell);
            let text = text.trim();
            let body = text.trim_start_matches(['+', '-']);
            !body.is_empty()
                && body.bytes().all(|b| b.is_ascii_digit())
                && (!super::numeric::integer_in_domain(text)
                    || text
                        .parse::<i128>()
                        .map_or(true, |n| !super::numeric::integer_exact_f64(n)))
        });
        Column {
            numeric_text: vec![],
            source_dtype: None,
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: if all_int && !has_na {
                NumericKind::NonFloatNumeric
            } else {
                NumericKind::Float
            },
            values: raw
                .iter()
                .map(|c| {
                    if is_na(c) {
                        Cell::Float(f64::NAN)
                    } else {
                        let text = norm(c);
                        let body = text.trim().trim_start_matches(['+', '-']);
                        let integer_literal =
                            !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit());
                        if integer_literal && (keep_integer_peers || (all_int && !has_na)) {
                            if let Ok(value) = text.trim().parse::<i64>() {
                                if value == 0
                                    && text.trim_start().starts_with('-')
                                    && (!all_int || has_na)
                                {
                                    Cell::Float(-0.0)
                                } else {
                                    Cell::Int(value)
                                }
                            } else if super::numeric::integer_in_domain(&text) {
                                let digits = body.trim_start_matches('0');
                                Cell::Str(if digits.is_empty() {
                                    "0".into()
                                } else {
                                    digits.into()
                                })
                            } else {
                                Cell::Str(text)
                            }
                        } else if super::numeric::checked_decimal_f64(text.trim()).is_err() {
                            Cell::Str(text)
                        } else {
                            Cell::Float(parse_float(&text).unwrap())
                        }
                    }
                })
                .collect(),
        }
    } else {
        Column {
            numeric_text: vec![],
            source_dtype: None,
            name: name.into(),
            dtype: ColDtype::String,
            numeric_kind: NumericKind::NonNumeric,
            values: raw
                .iter()
                .map(|c| {
                    if is_na(c) {
                        Cell::Na
                    } else {
                        Cell::Str(c.clone())
                    }
                })
                .collect(),
        }
    }
}

fn header_unit(params: &LoadingParams, has_header: bool) -> String {
    match params.header_unit {
        Some(u) => u.value().to_string(),
        None => if has_header { "text" } else { "index" }.to_string(),
    }
}

fn cell_is_na(cell: &Cell) -> bool {
    matches!(cell, Cell::Na) || matches!(cell, Cell::Float(f) if f.is_nan())
}

/// Report missing input cells at the deferred-NA reader boundary, before joins
/// or role-specific policies can introduce or remove cells.
pub(crate) fn ignored_input_na_report(frame: &Frame) -> Option<Value> {
    let mut samples = vec![false; frame.n_rows];
    let mut count = 0_usize;
    let mut features = 0_usize;
    for column in &frame.columns {
        let mut missing_column = false;
        for (row, cell) in column.values.iter().enumerate() {
            if cell_is_na(cell) {
                count += 1;
                missing_column = true;
                if let Some(sample) = samples.get_mut(row) {
                    *sample = true;
                }
            }
        }
        features += usize::from(missing_column);
    }
    (count != 0).then(|| {
        serde_json::json!({
            "strategy": "ignore", "na_detected": true, "na_count": count,
            "na_samples": samples.iter().filter(|missing| **missing).count(),
            "na_features": features, "removed_samples": [], "removed_features": [],
            "fill_method": null, "na_preserved": true
        })
    })
}

fn first_na(frame: &Frame) -> Option<(usize, String)> {
    for row in 0..frame.n_rows {
        for col in &frame.columns {
            if col.values.get(row).is_some_and(cell_is_na) {
                return Some((row, col.name.clone()));
            }
        }
    }
    None
}

fn row_has_na(frame: &Frame, row: usize) -> bool {
    frame
        .columns
        .iter()
        .any(|col| col.values.get(row).is_some_and(cell_is_na))
}

fn column_has_na(col: &Column) -> bool {
    col.values.iter().any(cell_is_na)
}

fn frame_from_column_values(frame: &Frame, values_by_column: Vec<Vec<Cell>>) -> Frame {
    let columns = frame
        .columns
        .iter()
        .zip(values_by_column)
        .map(|(col, values)| col.aligned_cells(&col.name, values, col.numeric_text.clone()))
        .collect();
    Frame {
        columns,
        n_rows: frame.n_rows,
        header_unit: frame.header_unit.clone(),
    }
}

fn fill_value_cell(fill_value: &Value) -> Result<Cell, SpecError> {
    match fill_value {
        Value::Null => Ok(Cell::Float(0.0)),
        Value::Bool(value) => Ok(Cell::Bool(*value)),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Cell::Int(value))
            } else if let Some(value) = value.as_u64() {
                Ok(i64::try_from(value)
                    .map(Cell::Int)
                    .unwrap_or_else(|_| Cell::Str(value.to_string())))
            } else if let Some(value) = value.as_f64() {
                Ok(Cell::Float(value))
            } else {
                Err(SpecError::new(
                    "na.fill.fill_value must be a number, string, bool, or null",
                ))
            }
        }
        Value::String(value) => Ok(Cell::Str(value.clone())),
        _ => Err(SpecError::new(
            "na.fill.fill_value must be a number, string, bool, or null",
        )),
    }
}

fn replace_na_value(frame: &Frame, fill_value: &Value) -> Result<Frame, SpecError> {
    let replacement = fill_value_cell(fill_value)?;
    let values_by_column = frame
        .columns
        .iter()
        .map(|col| {
            col.values
                .iter()
                .map(|cell| {
                    if cell_is_na(cell) {
                        replacement.clone()
                    } else {
                        cell.clone()
                    }
                })
                .collect()
        })
        .collect();
    Ok(frame_from_column_values(frame, values_by_column))
}

fn fill_across_columns(frame: &Frame, forward: bool) -> Frame {
    let mut rows: Vec<Vec<Cell>> = vec![vec![Cell::Na; frame.columns.len()]; frame.n_rows];
    for (row_idx, row) in rows.iter_mut().enumerate() {
        for (col_idx, col) in frame.columns.iter().enumerate() {
            row[col_idx] = col.values.get(row_idx).cloned().unwrap_or(Cell::Na);
        }
        let mut carry: Option<Cell> = None;
        if forward {
            for cell in row.iter_mut() {
                if cell_is_na(cell) {
                    if let Some(value) = &carry {
                        *cell = value.clone();
                    }
                } else {
                    carry = Some(cell.clone());
                }
            }
        } else {
            for cell in row.iter_mut().rev() {
                if cell_is_na(cell) {
                    if let Some(value) = &carry {
                        *cell = value.clone();
                    }
                } else {
                    carry = Some(cell.clone());
                }
            }
        }
    }
    let values_by_column = (0..frame.columns.len())
        .map(|col_idx| rows.iter().map(|row| row[col_idx].clone()).collect())
        .collect();
    frame_from_column_values(frame, values_by_column)
}

fn replace_na(frame: &Frame, na: &NaConfig, method: FillMethod) -> Result<Frame, SpecError> {
    match method {
        FillMethod::Value => replace_na_value(frame, &na.fill_value),
        FillMethod::ForwardFill => Ok(fill_across_columns(frame, true)),
        FillMethod::BackwardFill => Ok(fill_across_columns(frame, false)),
    }
}

/// Apply the configured NA policy to an already-decoded [`Frame`].
///
/// The Python MVP treats `auto` as `abort`; the Rust core follows that contract
/// for both CSV bytes and facade-decoded frames such as Parquet.
pub fn apply_na_policy(frame: &Frame, na: &NaConfig) -> Result<Frame, SpecError> {
    let policy = if na.policy == NaPolicy::Auto {
        NaPolicy::Abort
    } else {
        na.policy
    };
    let Some((row, col)) = first_na(frame) else {
        return Ok(frame.clone());
    };

    match policy {
        NaPolicy::Abort => Err(SpecError::new(format!(
            "NA values detected and na_policy is 'abort'. First NA in column '{col}' (row: {row})."
        ))),
        NaPolicy::Ignore => Ok(frame.clone()),
        NaPolicy::RemoveSample => {
            let mask: Vec<bool> = (0..frame.n_rows)
                .map(|row| !row_has_na(frame, row))
                .collect();
            Ok(frame.mask_rows(&mask))
        }
        NaPolicy::RemoveFeature => Ok(Frame {
            columns: frame
                .columns
                .iter()
                .filter(|col| !column_has_na(col))
                .cloned()
                .collect(),
            n_rows: frame.n_rows,
            header_unit: frame.header_unit.clone(),
        }),
        NaPolicy::Replace => {
            let method = na.fill_method.unwrap_or(FillMethod::Value);
            replace_na(frame, na, method)
        }
        NaPolicy::Auto => unreachable!("auto is normalized to abort"),
    }
}

/// pandas `mangle_dupe_cols`: `a, a.1, a.2`; stringify + pad to `ncols`.
fn mangle_dupes(header: &[String], ncols: usize) -> Vec<String> {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(ncols);
    for j in 0..ncols {
        let base = header.get(j).cloned().unwrap_or_else(|| j.to_string());
        let count = seen.entry(base.clone()).or_insert(0);
        let name = if *count == 0 {
            base.clone()
        } else {
            format!("{base}.{count}")
        };
        *count += 1;
        out.push(name);
    }
    out
}

/// IO-owned spreadsheet values. Integer tokens retain physical numeric origin
/// until role and survivor admission has taken place.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum SheetCell {
    Cell(Cell),
    IntegerLexeme(String),
    Float64 { bits: u64, token: String },
}

impl From<Cell> for SheetCell {
    fn from(cell: Cell) -> Self { Self::Cell(cell) }
}

/// A worksheet, parameterized so existing callers can supply ordinary cells.
pub type Sheet<C = SheetCell> = (String, Vec<Vec<C>>);

fn canonical_integer(token: &str) -> String {
    let negative = token.starts_with('-');
    let digits = token.trim_start_matches(['+', '-']).trim_start_matches('0');
    if digits.is_empty() { return "0".to_string(); }
    if negative { format!("-{digits}") } else { digits.to_string() }
}

fn sheet_cell_text(cell: &SheetCell) -> String {
    match cell {
        SheetCell::IntegerLexeme(token) => canonical_integer(token),
        SheetCell::Float64 { bits, .. } => {
            let value = f64::from_bits(*bits);
            if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
                (value as i64).to_string()
            } else { value.to_string() }
        }
        SheetCell::Cell(cell) => match cell {
            Cell::Str(text) => text.clone(),
            Cell::Int(value) => value.to_string(),
            Cell::Float(value) => value.to_string(),
            Cell::Bool(value) => if *value { "True" } else { "False" }.to_string(),
            Cell::Na => String::new(),
        },
    }
}

fn project_sheet_cell(cell: SheetCell) -> (Cell, bool) {
    match cell {
        SheetCell::Cell(Cell::Str(text)) if is_na(&text) => (Cell::Na, false),
        SheetCell::Cell(cell) => (cell, false),
        SheetCell::IntegerLexeme(token) => match token.parse::<i64>() {
            Ok(value) => (Cell::Int(value), false),
            Err(_) if super::numeric::integer_in_domain(&token) => {
                (Cell::Str(canonical_integer(&token)), true)
            }
            Err(_) => (Cell::Str(token), true),
        },
        SheetCell::Float64 { bits, .. } => {
            let value = f64::from_bits(bits);
            if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
                (Cell::Int(value as i64), false)
            } else { (Cell::Float(value), false) }
        }
    }
}

/// pandas' default boolean words.
fn sheet_bool_word(text: &str) -> Option<bool> {
    match text {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    }
}

/// Bytes a cell counts against the record and field budgets: its text, or the
/// width of an 8-byte value.
fn sheet_cell_bytes(cell: &SheetCell) -> u64 {
    match cell {
        SheetCell::IntegerLexeme(token) | SheetCell::Float64 { token, .. } => (token.len() as u64).max(8),
        SheetCell::Cell(Cell::Str(text)) => text.len() as u64,
        SheetCell::Cell(Cell::Na) => 0,
        SheetCell::Cell(_) => 8,
    }
}

/// Decode the worksheet named by `format.sheet_name` (a 0-based index or a
/// sheet name; the first sheet by default) into a [`Frame`], as pandas reads
/// it. A spreadsheet has a header row unless `has_header` is false; header
/// cells name the columns as text (a numeric wavelength `1000` names column
/// `"1000"`, a blank one `"Unnamed: <j>"`). Typed cells keep their type, a text
/// cell holding an NA token is missing, and a column of boolean words is
/// boolean.
/// `usecols` and `skiprows` are refused rather than ignored: select columns
/// with source column selectors instead.
pub fn read_sheet<C: Clone + Into<SheetCell>>(
    sheets: &[Sheet<C>],
    params: &LoadingParams,
    limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    for option in ["usecols", "skiprows"] {
        if params
            .format
            .values
            .get(option)
            .is_some_and(|value| !value.is_null())
        {
            return Err(SpecError::new(format!(
                "loading.format.{option} is not supported for spreadsheets; select columns with the source `columns` selectors"
            )));
        }
    }
    let sheet = match params.format.values.get("sheet_name") {
        None | Some(Value::Null) => sheets.first(),
        Some(Value::Number(index)) => index
            .as_u64()
            .and_then(|index| sheets.get(usize::try_from(index).ok()?)),
        Some(Value::String(name)) => sheets.iter().find(|(sheet, _)| sheet == name),
        Some(_) => {
            return Err(SpecError::new(
                "loading.format.sheet_name must be a sheet index or a sheet name",
            ))
        }
    };
    let Some((_, records)) = sheet else {
        let names: Vec<&str> = sheets.iter().map(|(name, _)| name.as_str()).collect();
        return Err(SpecError::new(format!(
            "worksheet {} not found; available: {names:?}",
            params
                .format
                .values
                .get("sheet_name")
                .unwrap_or(&Value::from(0))
        )));
    };
    let has_header = params.has_header.unwrap_or(true) && !records.is_empty();
    let ncols = records.iter().map(Vec::len).max().unwrap_or(0);
    let data = &records[usize::from(has_header)..];
    if let Some(limits) = limits {
        limits.check_shape(data.len() as u64, ncols as u64)?;
        for record in records {
            if record.iter().map(|cell| sheet_cell_bytes(&cell.clone().into())).sum::<u64>() > limits.max_record_bytes {
                return Err(SpecError::new(format!(
                    "tabular record exceeds the {}-byte record budget",
                    limits.max_record_bytes
                )));
            }
            if record
                .iter()
                .any(|cell| sheet_cell_bytes(&cell.clone().into()) > limits.max_field_bytes)
            {
                return Err(SpecError::new(format!(
                    "tabular field exceeds the {}-byte field budget",
                    limits.max_field_bytes
                )));
            }
        }
    }
    let header: Vec<String> = if has_header {
        records[0]
            .iter()
            .enumerate()
            .map(|(j, cell)| match sheet_cell_text(&cell.clone().into()) {
                text if text.is_empty() => format!("Unnamed: {j}"),
                text => text,
            })
            .collect()
    } else {
        (0..ncols).map(|j| j.to_string()).collect()
    };
    let columns = mangle_dupes(&header, ncols)
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let physical_text = data.iter().any(|record| {
                matches!(record.get(j).cloned().map(Into::into), Some(SheetCell::Cell(Cell::Str(_))))
            });
            let projected: Vec<(Cell, bool)> = data
                .iter()
                .map(|record| record.get(j).cloned().map(Into::into)
                    .map(project_sheet_cell).unwrap_or((Cell::Na, false)))
                .collect();
            let (mut cells, numeric_text): (Vec<Cell>, Vec<bool>) = projected.into_iter().unzip();
            // pandas reads a column of boolean words as booleans.
            if cells.iter().any(|cell| matches!(cell, Cell::Str(_)))
                && cells.iter().all(|cell| match cell {
                    Cell::Str(text) => sheet_bool_word(text).is_some(),
                    Cell::Bool(_) | Cell::Na => true,
                    _ => false,
                })
            {
                for cell in &mut cells {
                    if let Cell::Str(text) = cell {
                        *cell = Cell::Bool(sheet_bool_word(text) == Some(true));
                    }
                }
            }
            let mut column = Column::from_cells(name, cells);
            if numeric_text.iter().any(|value| *value) {
                if !physical_text {
                    column.dtype = ColDtype::Numeric;
                    column.numeric_kind = if column.values.iter().any(|cell| matches!(cell, Cell::Float(_) | Cell::Na)) {
                        NumericKind::Float
                    } else {
                        NumericKind::NonFloatNumeric
                    };
                }
                column.numeric_text = numeric_text;
            }
            column
        })
        .collect();
    apply_na_policy(
        &Frame {
            columns,
            n_rows: data.len(),
            header_unit: header_unit(params, has_header),
        },
        &params.na,
    )
}

/// Maximum decoded prefix inspected for an undeclared CSV dialect.
pub const DELIMITER_SAMPLE_BYTES: usize = 8 * 1024 * 1024;
const DELIMITER_SAMPLE_RECORDS: usize = 50;

/// Narrow evidence about alternatives to the intentional semicolon default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelimiterVerdict {
    Default,
    Detected(u8),
    Ambiguous(Vec<u8>),
}

/// Inspect at most 50 nonblank CSV records and 8 MiB, including the header.
/// This is not the inference detector: single-column files retain the default.
pub(crate) fn dialect_records(bytes: &[u8], delimiter: u8, truncated: bool) -> Vec<(usize, &[u8])> {
    // Width evidence only: fields remain opaque to the scientific decoder.
    let mut rows = Vec::new();
    let (mut start, mut i, mut width) = (0, 0, 1);
    let (mut quoted, mut field_start) = (false, true);
    while i < bytes.len() {
        let byte = bytes[i];
        if quoted {
            if byte == b'"' {
                if bytes.get(i + 1) == Some(&b'"') {
                    i += 2;
                    continue;
                }
                quoted = false;
            }
        } else if byte == b'"' && field_start {
            quoted = true;
            field_start = false;
        } else if byte == delimiter {
            width += 1;
            field_start = true;
        } else if matches!(byte, b'\r' | b'\n') {
            let raw = &bytes[start..i];
            if !raw.iter().all(u8::is_ascii_whitespace) {
                rows.push((width, raw));
                if rows.len() == DELIMITER_SAMPLE_RECORDS {
                    return rows;
                }
            }
            if byte == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                i += 1;
            }
            start = i + 1;
            width = 1;
            field_start = true;
        } else {
            field_start = false;
        }
        i += 1;
    }
    let raw = &bytes[start..];
    // Keep complete records at the cap; only an unfinished tail is omitted.
    if !truncated && !raw.iter().all(u8::is_ascii_whitespace) {
        rows.push((width, raw));
    }
    rows
}

pub(crate) fn dialect_text(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    match std::str::from_utf8(bytes) {
        Ok(text) => std::borrow::Cow::Borrowed(text),
        Err(_) => std::borrow::Cow::Owned(bytes.iter().map(|&byte| byte as char).collect()),
    }
}

pub(crate) fn is_na_record(raw: &str) -> bool {
    let token = raw.trim();
    NA_TOKENS.contains(token)
        || token
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .is_some_and(|s| NA_TOKENS.contains(s))
}

pub(crate) fn is_comma_pair(value: &str) -> bool {
    static PAIR: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^[+-]?[0-9]+,[0-9]+(?:[eE][+-]?[0-9]+)?$").unwrap());
    PAIR.is_match(value.trim())
}

pub fn undeclared_delimiter(bytes: &[u8]) -> DelimiterVerdict {
    let truncated = bytes.len() >= DELIMITER_SAMPLE_BYTES;
    let bytes = &bytes[..bytes.len().min(DELIMITER_SAMPLE_BYTES)];
    let bytes = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes);

    let default_rows = dialect_records(bytes, b';', truncated);
    if default_rows.len() < 2 || default_rows.iter().any(|row| row.0 > 1) {
        return DelimiterVerdict::Default;
    }
    let candidates: Vec<u8> = [b',', b'\t', b'|']
        .into_iter()
        .filter(|&delimiter| {
            let rows = dialect_records(bytes, delimiter, truncated);
            rows.len() >= 2 && rows[0].0 >= 2 && rows.iter().all(|r| r.0 == rows[0].0)
        })
        .collect();
    if candidates.is_empty() {
        let rows = dialect_records(bytes, b',', truncated);
        let data: Vec<_> = rows.iter().skip(1).filter(|row| row.0 != 1).collect();
        if !rows.is_empty()
            && rows[0].0 >= 2
            && !data.is_empty()
            && data
                .iter()
                .all(|row| row.0 == 2 && is_comma_pair(&dialect_text(row.1)))
        {
            return DelimiterVerdict::Ambiguous(vec![b',']);
        }
        return DelimiterVerdict::Default;
    }
    let comma_pair = candidates == [b',']
        && default_rows
            .iter()
            .skip(1)
            .all(|row| is_comma_pair(&dialect_text(row.1)));
    if candidates.len() == 1 && !comma_pair {
        DelimiterVerdict::Detected(candidates[0])
    } else {
        DelimiterVerdict::Ambiguous(candidates)
    }
}

/// Assembly-only guard; raw read/inspection callers keep their own contracts.
pub fn refuse_undeclared_delimiter(
    bytes: &[u8],
    params: &LoadingParams,
    source_id: &str,
    name: &str,
) -> Result<(), SpecError> {
    if params.delimiter.as_deref().is_some_and(|d| !d.is_empty()) {
        return Ok(());
    }
    let detail = match undeclared_delimiter(bytes) {
        DelimiterVerdict::Default => return Ok(()),
        DelimiterVerdict::Detected(delimiter) => format!(
            "splits consistently on {:?}",
            (delimiter as char).to_string()
        ),
        DelimiterVerdict::Ambiguous(delimiters) => format!(
            "has ambiguous delimiters {:?} (and params.decimal_separator if ',' is the decimal mark)",
            delimiters
                .iter()
                .map(|&d| (d as char).to_string())
                .collect::<Vec<_>>()
        ),
    };
    Err(SpecError::new(format!(
        "source '{source_id}': file '{name}' has no ';' but {detail}; declare params.delimiter"
    )))
}

/// Decode already-read (and decompressed) tabular bytes into a [`Frame`].
///
/// v0: the CSV family. UTF-8 with a latin-1 fallback, mirroring the native
/// loader's decoding. The facade reads the file and handles gzip/zip before
/// calling this so a single decoder backs both paths.
pub fn discovery_prefix(bytes: &[u8], params: &LoadingParams) -> Option<Vec<u8>> {
    let sample = &bytes[..bytes.len().min(DELIMITER_SAMPLE_BYTES)];
    let verdict = undeclared_delimiter(sample);
    let sample = sample.strip_prefix(b"\xef\xbb\xbf").unwrap_or(sample);
    if params.delimiter.as_deref().is_none_or(str::is_empty)
        && !matches!(verdict, DelimiterVerdict::Default)
    {
        return None;
    }
    let delimiter = params
        .delimiter
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(";")
        .as_bytes()[0];
    let records = dialect_records(sample, delimiter, bytes.len() >= DELIMITER_SAMPLE_BYTES);
    if records.is_empty() {
        return None;
    }
    let mut prefix = Vec::new();
    for (_, record) in records {
        prefix.extend_from_slice(record);
        prefix.push(b'\n');
    }
    Some(prefix)
}

/// Discovery never applies destructive NA policy or validates key equality.
pub fn discovery_table(bytes: &[u8], params: &LoadingParams) -> Option<Frame> {
    let prefix = discovery_prefix(bytes, params)?;
    let mut preview_params = params.clone();
    preview_params.na.policy = NaPolicy::Ignore;
    read_table_bytes(&prefix, &preview_params).ok()
}

pub fn read_table_bytes(bytes: &[u8], params: &LoadingParams) -> Result<Frame, SpecError> {
    read_table_bytes_impl(bytes, params, None)
}

/// Decode tabular bytes while enforcing streaming record, field, and frame
/// shape budgets before copying an accepted record into owned cell strings.
pub fn read_table_bytes_with_limits(
    bytes: &[u8],
    params: &LoadingParams,
    limits: TabularReadLimits,
) -> Result<Frame, SpecError> {
    read_table_bytes_impl(bytes, params, Some(limits))
}

fn read_table_bytes_impl(
    bytes: &[u8],
    params: &LoadingParams,
    limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    let delimiter = params
        .delimiter
        .as_deref()
        .unwrap_or(";")
        .chars()
        .next()
        .unwrap_or(';');
    let decimal = params
        .decimal_separator
        .as_deref()
        .unwrap_or(".")
        .chars()
        .next()
        .unwrap_or('.');
    let has_header = params.has_header.unwrap_or(true);
    let latin1;
    let text = match std::str::from_utf8(bytes) {
        Ok(_) => bytes,
        Err(_) => {
            latin1 = bytes.iter().map(|&byte| byte as char).collect::<String>();
            latin1.as_bytes()
        }
    };

    // csv's reader unconditionally skips physically empty lines. These are
    // missing observations under every NA policy, including in
    // single-column target files. Quote empty physical records before parsing;
    // quoted multiline fields and CRLF bytes retain their original meaning.
    let preserved = preserve_empty_csv_records(text, delimiter as u8, has_header);
    let text = preserved.as_slice();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(delimiter as u8)
        .from_reader(text);
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record = csv::ByteRecord::new();
    let mut data_cells = 0_u64;
    let mut max_columns_seen = 0_u64;
    while rdr
        .read_byte_record(&mut record)
        .map_err(|e| SpecError::new(format!("csv parse error: {e}")))?
    {
        if let Some(limits) = limits {
            let record_bytes = record.iter().fold(0_u64, |total, field| {
                total.saturating_add(field.len() as u64)
            });
            if record_bytes > limits.max_record_bytes {
                return Err(SpecError::new(format!(
                    "tabular record exceeds the {}-byte record budget",
                    limits.max_record_bytes
                )));
            }
            if record
                .iter()
                .any(|field| field.len() as u64 > limits.max_field_bytes)
            {
                return Err(SpecError::new(format!(
                    "tabular field exceeds the {}-byte field budget",
                    limits.max_field_bytes
                )));
            }

            let next_record_count = records.len() as u64 + 1;
            let data_rows = next_record_count.saturating_sub(u64::from(has_header));
            if data_rows > limits.max_rows {
                return Err(SpecError::new(format!(
                    "tabular input exceeds the {}-row budget",
                    limits.max_rows
                )));
            }
            let columns = max_columns_seen.max(record.len() as u64);
            if columns > limits.max_columns {
                return Err(SpecError::new(format!(
                    "tabular input exceeds the {}-column budget",
                    limits.max_columns
                )));
            }
            let next_data_cells = if has_header && records.is_empty() {
                data_cells
            } else {
                data_cells.saturating_add(record.len() as u64)
            };
            let rectangular_cells = data_rows.saturating_mul(columns);
            if next_data_cells.max(rectangular_cells) > limits.max_cells {
                return Err(SpecError::new(format!(
                    "tabular input exceeds the {}-cell budget",
                    limits.max_cells
                )));
            }
            data_cells = next_data_cells;
            max_columns_seen = columns;
        }
        records.push(
            record
                .iter()
                .map(|field| {
                    std::str::from_utf8(field)
                        .expect("CSV input was normalized to UTF-8")
                        .to_string()
                })
                .collect(),
        );
    }
    let first_width = if has_header {
        records.first().map(|row| (0, row.len()))
    } else {
        records
            .iter()
            .enumerate()
            .find(|(_, row)| !(row.len() == 1 && row[0].is_empty()))
            .map(|(index, row)| (index, row.len()))
    };
    if let Some((first, width)) = first_width {
        for (index, row) in records.iter().enumerate().skip(first + 1) {
            if row.len() > width {
                return Err(SpecError::new(format!(
                    "csv width error: record {} has {} fields, expected {width}",
                    index + 1,
                    row.len()
                )));
            }
        }
    }
    if records.is_empty() {
        return apply_na_policy(
            &Frame {
                columns: vec![],
                n_rows: 0,
                header_unit: header_unit(params, has_header),
            },
            &params.na,
        );
    }

    let ncols = records.iter().map(|r| r.len()).max().unwrap_or(0);
    let (header, data_start) = if has_header {
        (
            records[0]
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    if name.is_empty() {
                        format!("Unnamed: {index}")
                    } else {
                        name.clone()
                    }
                })
                .collect::<Vec<_>>(),
            1,
        )
    } else {
        ((0..ncols).map(|i| i.to_string()).collect(), 0)
    };
    let names = mangle_dupes(&header, ncols);

    let columns: Vec<Column> = names
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let raw: Vec<String> = records[data_start..]
                .iter()
                .map(|r| r.get(j).cloned().unwrap_or_default())
                .collect();
            infer_column(name, &raw, decimal)
        })
        .collect();
    apply_na_policy(
        &Frame {
            columns,
            n_rows: records.len() - data_start,
            header_unit: header_unit(params, has_header),
        },
        &params.na,
    )
}

fn preserve_empty_csv_records(input: &[u8], delimiter: u8, has_header: bool) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut quoted = false;
    let mut field_start = true;
    let mut line_start = true;
    let mut seen_content = !has_header;
    let mut index = 0;
    while index < input.len() {
        let byte = input[index];
        if quoted && byte == b'"' && input.get(index + 1) == Some(&b'"') {
            output.extend_from_slice(b"\"\"");
            index += 2;
            continue;
        }
        if byte == b'"' && (quoted || field_start) {
            quoted = !quoted;
        }
        if !quoted && matches!(byte, b'\r' | b'\n') {
            // CRLF is one terminator; do not manufacture a second record.
            let crlf_tail = byte == b'\n' && index > 0 && input[index - 1] == b'\r';
            if line_start && !crlf_tail && seen_content {
                output.extend_from_slice(b"\"\"");
            }
            line_start = true;
            field_start = true;
        } else {
            line_start = false;
            seen_content = true;
            field_start = !quoted && byte == delimiter;
        }
        output.push(byte);
        index += 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_negative_zero_preserves_bits_without_retyping_integer_columns() {
        for tokens in [
            vec!["-0", "9007199254740993", "1.5"],
            vec!["-00", "18446744073709551616", "1.5"],
            vec![" -0 ", "9007199254740993", ""],
        ] {
            let raw = tokens.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let column = infer_column("x", &raw, '.');
            assert!(
                matches!(column.values[0], Cell::Float(value) if value.to_bits() == (-0.0_f64).to_bits())
            );
            let metadata = super::super::frame::metadata_cell_value(&column, 0, &column.values[0]);
            assert!(metadata.is_f64());
            assert_eq!(metadata.as_f64().unwrap().to_bits(), (-0.0_f64).to_bits());
        }
        for tokens in [vec!["-0", "3"], vec!["-0", "9007199254740993", "3"]] {
            let raw = tokens.into_iter().map(str::to_owned).collect::<Vec<_>>();
            let column = infer_column("x", &raw, '.');
            assert_eq!(column.values[0], Cell::Int(0));
        }
        let raw = ["-0", "9007199254740993", "1.5"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let frame = Frame::from_columns(vec![infer_column("x", &raw, '.')], "text");
        let matrix = frame
            .coerce_numeric_with_storage(
                &["x".into()],
                Some(super::super::numeric::NumericStorage::Float32),
                None,
            )
            .unwrap();
        assert_eq!(matrix.data[0].to_bits(), (-0.0_f64).to_bits());
        let raw = ["-0", "abc"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(
            infer_column("m", &raw, '.').values[0],
            Cell::Str("-0".into())
        );
    }

    #[test]
    fn ignore_keeps_empty_target_records_with_lf_crlf_and_quoted_cells() {
        for newline in ["\n", "\r\n", "\r"] {
            let text = ["class", "pass", "", "fail", "\"\"", ""].join(newline) + newline;
            let result =
                read_table_bytes(text.as_bytes(), &params_with_na(na(NaPolicy::Ignore))).unwrap();
            assert_eq!(result.n_rows, 5);
            assert!(matches!(result.columns[0].values[1], Cell::Na));
            assert!(matches!(result.columns[0].values[3], Cell::Na));
            assert!(matches!(result.columns[0].values[4], Cell::Na));
        }
    }

    #[test]
    fn empty_record_preservation_respects_csv_quoting_and_headerless_rows() {
        let input = b"name;value\n\"a\n\nb\";1\n\"a\"\"b\";2\n;\n";
        let result = read_table_bytes(input, &params_with_na(na(NaPolicy::Ignore))).unwrap();
        assert_eq!(result.n_rows, 3);
        assert_eq!(result.columns[0].values[0].to_str_scalar(), "a\n\nb");
        assert_eq!(result.columns[0].values[1].to_str_scalar(), "a\"b");
        assert!(matches!(result.columns[0].values[2], Cell::Na));
        let mut params = params_with_na(na(NaPolicy::Ignore));
        params.has_header = Some(false);
        let result = read_table_bytes(b"\n1\n\n", &params).unwrap();
        assert_eq!(result.n_rows, 3);
        assert!(result.numeric_column_f64("0")[0].is_nan());
    }

    #[test]
    fn empty_observations_count_towards_row_budgets() {
        let params = params_with_na(na(NaPolicy::Ignore));
        assert!(read_table_bytes_with_limits(
            b"target\n1\n\n3\n",
            &params,
            TabularReadLimits::new(64, 64, 2, 1, 3)
        )
        .unwrap_err()
        .to_string()
        .contains("row budget"));
        assert_eq!(
            read_table_bytes(b"\n\ntarget\n1\n", &params)
                .unwrap()
                .n_rows,
            1
        );
    }

    fn read(text: &str, params: &LoadingParams) -> Frame {
        read_table_bytes(text.as_bytes(), params).unwrap()
    }

    fn params_with_na(na: NaConfig) -> LoadingParams {
        LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            na,
            ..Default::default()
        }
    }

    #[test]
    fn bounded_reader_accepts_every_limit_exactly() {
        let limits = TabularReadLimits::new(5, 3, 2, 2, 4);
        let frame = read_table_bytes_with_limits(
            b"abc;de\n1;2\n3;4\n",
            &LoadingParams {
                delimiter: Some(";".into()),
                has_header: Some(true),
                ..Default::default()
            },
            limits,
        )
        .unwrap();

        assert_eq!(frame.n_rows, 2);
        assert_eq!(frame.columns.len(), 2);
    }

    #[test]
    fn bounded_reader_counts_latin1_after_utf8_normalization() {
        let error = read_table_bytes_with_limits(
            b"\xe9\n",
            &LoadingParams {
                has_header: Some(false),
                ..Default::default()
            },
            TabularReadLimits::new(1, 1, 1, 1, 1),
        )
        .unwrap_err();

        assert!(error.message.contains("1-byte record budget"));
        assert!(read_table_bytes(
            b"\xe9\n",
            &LoadingParams {
                has_header: Some(false),
                ..Default::default()
            }
        )
        .is_ok());
    }

    fn na(policy: NaPolicy) -> NaConfig {
        NaConfig {
            policy,
            ..Default::default()
        }
    }

    #[test]
    fn reads_combined_float_columns() {
        let mut text = String::from("1000;1005;protein\n");
        for y in ["12.5", "8.3", "15.1"] {
            text.push_str(&format!("0.40;1.30;{y}\n"));
        }
        let params = LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            ..Default::default()
        };
        let t = read(&text, &params);
        assert_eq!(t.column_names(), vec!["1000", "1005", "protein"]);
        assert_eq!(t.dtype_labels(), vec!["numeric", "numeric", "numeric"]);
        let prof = t.to_table_profile();
        assert_eq!(prof.column("1000").unwrap().nunique_with_na, 1);
        assert!(prof.column("protein").unwrap().is_unique);
        assert!(prof.column("protein").unwrap().is_float_dtype());
        assert_eq!(t.numeric_column_f64("protein"), vec![12.5, 8.3, 15.1]);
    }

    #[test]
    fn int_column_stays_nonfloat() {
        let params = LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            ..Default::default()
        };
        let prof = read("id;v\n1;0.5\n2;0.6\n", &params).to_table_profile();
        let idc = prof.column("id").unwrap();
        assert_eq!(idc.numeric_kind, NumericKind::NonFloatNumeric);
        assert!(!idc.is_float_dtype());
        assert!(idc.is_numeric_dtype());
        assert_eq!(idc.str_values, vec!["1", "2"]);
    }

    #[test]
    fn csv_auto_and_abort_error_on_first_na() {
        let text = "a;b\n1;\n2;3\n";

        let auto = read_table_bytes(text.as_bytes(), &params_with_na(na(NaPolicy::Auto)))
            .expect_err("auto must abort on NA");
        assert!(auto.message.contains("First NA in column 'b' (row: 0)"));

        let abort = read_table_bytes(text.as_bytes(), &params_with_na(na(NaPolicy::Abort)))
            .expect_err("abort must abort on NA");
        assert!(abort.message.contains("na_policy is 'abort'"));
    }

    #[test]
    fn csv_ignore_preserves_na() {
        let frame = read("a;b\n1;\n2;3\n", &params_with_na(na(NaPolicy::Ignore)));

        assert_eq!(frame.n_rows, 2);
        let b = frame.numeric_column_f64("b");
        assert!(b[0].is_nan());
        assert_eq!(b[1], 3.0);
    }

    #[test]
    fn csv_remove_sample_drops_rows_with_any_na() {
        let frame = read(
            "a;b;c\n1;;x\n2;3;y\n;4;z\n",
            &params_with_na(na(NaPolicy::RemoveSample)),
        );

        assert_eq!(frame.n_rows, 1);
        assert_eq!(frame.numeric_column_f64("a"), vec![2.0]);
        assert_eq!(frame.str_column("c"), vec!["y"]);
    }

    #[test]
    fn csv_remove_feature_drops_columns_with_any_na() {
        let frame = read(
            "a;b;c\n1;;x\n2;3;y\n;4;z\n",
            &params_with_na(na(NaPolicy::RemoveFeature)),
        );

        assert_eq!(frame.column_names(), vec!["c"]);
        assert_eq!(frame.n_rows, 3);
    }

    #[test]
    fn csv_replace_value_fills_na() {
        let mut cfg = na(NaPolicy::Replace);
        cfg.fill_method = Some(FillMethod::Value);
        cfg.fill_value = serde_json::json!(9.5);

        let frame = read("a;b\n1;\n2;3\n", &params_with_na(cfg));

        assert_eq!(frame.numeric_column_f64("b"), vec![9.5, 3.0]);

        let default_fill = read("a;b\n1;\n2;3\n", &params_with_na(na(NaPolicy::Replace)));
        assert_eq!(default_fill.numeric_column_f64("b"), vec![0.0, 3.0]);
    }

    #[test]
    fn replace_value_supports_numeric_string_and_bool_fill_values() {
        let replace = |fill_value: Value| NaConfig {
            policy: NaPolicy::Replace,
            fill_method: Some(FillMethod::Value),
            fill_value,
            ..Default::default()
        };

        let numeric = Frame::from_columns(
            vec![Column::from_cells(
                "x",
                vec![Cell::Float(1.0), Cell::Float(f64::NAN)],
            )],
            "text",
        );
        let numeric = apply_na_policy(&numeric, &replace(serde_json::json!(7.5))).unwrap();
        assert_eq!(numeric.numeric_column_f64("x"), vec![1.0, 7.5]);

        let string = Frame::from_columns(
            vec![Column::from_cells(
                "label",
                vec![Cell::Str("a".into()), Cell::Na],
            )],
            "text",
        );
        let string = apply_na_policy(&string, &replace(serde_json::json!("missing"))).unwrap();
        assert_eq!(string.str_column("label"), vec!["a", "missing"]);

        let bools = Frame::from_columns(
            vec![Column::from_cells("flag", vec![Cell::Bool(true), Cell::Na])],
            "text",
        );
        let bools = apply_na_policy(&bools, &replace(serde_json::json!(false))).unwrap();
        assert_eq!(bools.str_column("flag"), vec!["True", "False"]);
    }

    #[test]
    fn csv_replace_forward_and_backward_fill_across_columns() {
        let mut cfg = na(NaPolicy::Replace);
        cfg.fill_method = Some(FillMethod::ForwardFill);
        let ffill = read("a;b;c\n1;;3\n;2;3\n", &params_with_na(cfg.clone()));
        assert_eq!(ffill.numeric_column_f64("b"), vec![1.0, 2.0]);
        assert!(ffill.numeric_column_f64("a")[1].is_nan());

        cfg.fill_method = Some(FillMethod::BackwardFill);
        let bfill = read("a;b;c\n1;;3\n;2;3\n", &params_with_na(cfg));
        assert_eq!(bfill.numeric_column_f64("b"), vec![3.0, 2.0]);
        assert_eq!(bfill.numeric_column_f64("a")[1], 2.0);
    }
}

#[cfg(test)]
mod residual_arity_tests {
    use super::*;

    #[test]
    fn literal_empty_or_whitespace_header_sets_physical_width() {
        let params = LoadingParams {
            delimiter: Some(",".into()),
            has_header: Some(true),
            na: NaConfig {
                policy: NaPolicy::Ignore,
                ..Default::default()
            },
            ..Default::default()
        };
        for header in [" ", "\t", "\" \"", "\"\""] {
            let text = format!("{header}\n1,2\n3,4\n");
            let error = read_table_bytes(text.as_bytes(), &params).unwrap_err();
            assert_eq!(
                error.message,
                "csv width error: record 2 has 2 fields, expected 1"
            );
        }
        let empty_header = read_table_bytes(b"\"\"\n1\n2\n", &params).unwrap();
        assert_eq!(empty_header.n_rows, 2);
        assert_eq!(empty_header.column_names(), vec!["Unnamed: 0"]);
        let headerless = LoadingParams {
            has_header: Some(false),
            ..params
        };
        let error = read_table_bytes(b" \n1,2\n3,4\n", &headerless).unwrap_err();
        assert!(error.message.contains("expected 1"));
        let missing_first = read_table_bytes(b"\n1,2\n3,4\n", &headerless).unwrap();
        assert_eq!(missing_first.n_rows, 3);
        assert!(missing_first.numeric_column_f64("0")[0].is_nan());
    }

    #[test]
    fn physical_width_valid_controls_keep_names_and_blank_observation_positions() {
        let params = LoadingParams {
            delimiter: Some(",".into()),
            has_header: Some(true),
            na: NaConfig {
                policy: NaPolicy::Ignore,
                ..Default::default()
            },
            ..Default::default()
        };
        let trailing = read_table_bytes(b"x,y,\n1,2,\n", &params).unwrap();
        assert_eq!(trailing.column_names(), vec!["x", "y", "Unnamed: 2"]);
        let first = read_table_bytes(b"x,y\n\n1,2\n", &params).unwrap();
        assert_eq!(first.n_rows, 2);
        assert!(first.numeric_column_f64("x")[0].is_nan());
        assert_eq!(first.numeric_column_f64("x")[1], 1.0);
        let middle = read_table_bytes(b"x,y\n1,2\n\n3,4\n", &params).unwrap();
        assert_eq!(middle.n_rows, 3);
        assert!(middle.numeric_column_f64("x")[1].is_nan());
        let single = read_table_bytes(b"x\n1\n\n2\n", &params).unwrap();
        assert_eq!(single.n_rows, 3);
        assert!(single.numeric_column_f64("x")[1].is_nan());
        let semicolon = LoadingParams {
            delimiter: Some(";".into()),
            ..params.clone()
        };
        for (text, rows) in [
            (b"x;y\n\n1;2\n".as_slice(), 2),
            (b"x;y\n\n\n1;2\n".as_slice(), 3),
            (b"x;y\r\n\r\n1;2\r\n".as_slice(), 2),
            (b"x;y\r\r1;2\r".as_slice(), 2),
        ] {
            let frame = read_table_bytes(text, &semicolon).unwrap();
            assert_eq!(frame.n_rows, rows);
            assert!(frame.numeric_column_f64("x")[0].is_nan());
            assert_eq!(frame.numeric_column_f64("x")[rows - 1], 1.0);
        }
        assert_eq!(
            read_table_bytes(b"\n\ntarget\n1\n", &semicolon)
                .unwrap()
                .n_rows,
            1
        );
    }

    #[test]
    fn physical_width_refuses_overflow_including_trailing_empty_fields() {
        for has_header in [true, false] {
            let params = LoadingParams {
                delimiter: Some(",".into()),
                has_header: Some(has_header),
                na: NaConfig {
                    policy: NaPolicy::Ignore,
                    ..Default::default()
                },
                ..Default::default()
            };
            for text in [
                "x,y,m\n0.5,1.0,7,9\n1.5,2.0,8\n",
                "x,y,m\n1,2,3,4\n5,6,7,8\n",
                "x,y,m\n1,2,3,\n4,5,6,\n",
                "x,y,m\n1,2,3,\n4,5,6\n",
                "x,y,m\n1,2,3\n4,5,6,\n",
                "x,y,m\n1,2,3\n4,5,6,7\n",
                "x,y,m\n1,2,3,,\n",
                "x,y,m\n1\n2,3,4,5\n",
            ] {
                let error = read_table_bytes(text.as_bytes(), &params).unwrap_err();
                assert!(error.message.contains("csv width error:"), "{error:?}");
            }
            for text in [
                "x,y,m\n1,2\n4,5,6\n",
                "x,y,\n1,2,\n",
                "x,y\n\"a\nb,c\",2\n",
                "x,y\r\n1,2\r\n3,4\r\n",
                "x,y\n\n1,2\n",
            ] {
                assert!(
                    read_table_bytes(text.as_bytes(), &params).is_ok(),
                    "{text:?}"
                );
            }
            let large = format!("x,y\n{},2\n", "a".repeat(140_000));
            assert!(read_table_bytes(large.as_bytes(), &params).is_ok());
        }
    }
}

#[cfg(test)]
mod delimiter_resolution_tests {
    use super::*;

    #[test]
    fn narrow_rule_retains_quotes_defaults_and_ambiguity() {
        for (bytes, expected) in [
            (&b"x,y\n1,2\n"[..], DelimiterVerdict::Ambiguous(vec![b','])),
            (&b"x\ty\n1\t2\n"[..], DelimiterVerdict::Detected(b'\t')),
            (&b"x|y\n1|2\n"[..], DelimiterVerdict::Detected(b'|')),
            (&b"1,5\n2,5\n"[..], DelimiterVerdict::Ambiguous(vec![b','])),
            (
                &b"a,b|c\n1,2|3\n"[..],
                DelimiterVerdict::Ambiguous(vec![b',', b'|']),
            ),
            (&b"x;m\n1.5;\n2.5;\n"[..], DelimiterVerdict::Default),
            (&b"1,5;2,5\n3,5;4,5\n"[..], DelimiterVerdict::Default),
            (
                &b"name\n\"Smith, J\"\n\"Doe, A\"\n"[..],
                DelimiterVerdict::Default,
            ),
            (&b"name\nSmith, J\nDoe, A\n"[..], DelimiterVerdict::Default),
            (&b"y\n1,5\n2,5\n"[..], DelimiterVerdict::Default),
            (
                &b"Full Name\nJohn Smith\nAnn Lee\n"[..],
                DelimiterVerdict::Default,
            ),
            (&b"a,b\n"[..], DelimiterVerdict::Default),
            (
                &b"Protein, %\n12,5\n13,1\n14,2\n"[..],
                DelimiterVerdict::Ambiguous(vec![b',']),
            ),
            (
                &b"id,y\n1,5\n2,7\n"[..],
                DelimiterVerdict::Ambiguous(vec![b',']),
            ),
            (
                &b"x,y\n1.5,2.5\n3.5,4.5\n"[..],
                DelimiterVerdict::Detected(b','),
            ),
            (
                &b"1,5 \n2,5 \n3,5 "[..],
                DelimiterVerdict::Ambiguous(vec![b',']),
            ),
        ] {
            assert_eq!(undeclared_delimiter(bytes), expected);
        }
        let header = "axis,".repeat(40000);
        assert_eq!(
            undeclared_delimiter(format!("\"{header}\",y\n1.0,2.0\n3.0,4.0\n").as_bytes()),
            DelimiterVerdict::Detected(b',')
        );
        assert_eq!(
            undeclared_delimiter(format!("\"{header}\"\n1\n2\n").as_bytes()),
            DelimiterVerdict::Default
        );
        let mut bounded = b"x,y\n1.0,2.0\n".repeat(25);
        bounded.extend_from_slice(b"3;4\n");
        assert_eq!(
            undeclared_delimiter(&bounded),
            DelimiterVerdict::Detected(b',')
        );
    }
}
