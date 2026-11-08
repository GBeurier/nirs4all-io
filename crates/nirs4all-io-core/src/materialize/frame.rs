// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! A minimal pandas-faithful typed frame for materialization.
//!
//! Columns carry a uniform pandas-like dtype (int64 / float64 / object) inferred
//! the way pandas infers a CSV column. The frame supports the operations the
//! join engine and assembler need (select, mask, add-column, coerce-to-matrix)
//! and produces the neutral `TableProfile` the core inference consumes.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::infer::table::{ColDtype, ColumnProfile, NumericKind, TableProfile};
use crate::pyfmt::py_str_scalar;
use serde_json::Value;

/// A temporary numerical view. The original cell remains the wire/key authority.
pub(crate) fn numeric_view(text: &str, decimal: char) -> Cow<'_, str> {
    if decimal != '.' && text.contains(decimal) {
        Cow::Owned(text.replace(decimal, "."))
    } else {
        Cow::Borrowed(text)
    }
}

/// A pandas-typed cell. Within a column all cells share the column dtype
/// (`Bool`, `Int` for int64, `Float` for float64 with `NaN`=NA, `Str`/`Na`
/// for object).
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Na,
}

impl Cell {
    /// `to_numeric(errors="coerce")` → f64 (NaN for non-numeric / NA).
    pub fn to_numeric(&self) -> f64 {
        self.to_numeric_view('.')
    }
    pub(crate) fn to_numeric_view(&self, decimal: char) -> f64 {
        match self {
            Cell::Bool(b) => i64::from(*b) as f64,
            Cell::Int(i) => *i as f64,
            Cell::Float(f) => *f,
            Cell::Str(s) => numeric_view(s, decimal)
                .trim()
                .parse::<f64>()
                .unwrap_or(f64::NAN),
            Cell::Na => f64::NAN,
        }
    }
    pub fn to_numeric_checked(&self) -> Result<f64, String> {
        self.to_numeric_with_storage(super::numeric::NumericStorage::Float64, None)
    }
    pub fn to_numeric_with_storage(
        &self,
        storage: super::numeric::NumericStorage,
        source_dtype: Option<&str>,
    ) -> Result<f64, String> {
        self.to_numeric_with_decimal(storage, source_dtype, '.')
    }
    pub(crate) fn to_numeric_with_decimal(
        &self,
        storage: super::numeric::NumericStorage,
        source_dtype: Option<&str>,
        decimal: char,
    ) -> Result<f64, String> {
        use super::numeric::NumericStorage;
        if source_dtype == Some("float32") {
            let normalized =
                self.to_numeric_with_decimal(NumericStorage::Float32, None, decimal)?;
            return if normalized.is_nan() {
                Ok(normalized)
            } else {
                storage.round(normalized)
            };
        }
        if source_dtype == Some("float64") {
            let source = self.to_numeric_with_decimal(NumericStorage::Float64, None, decimal)?;
            return if source.is_nan() {
                Ok(source)
            } else {
                storage.round(source)
            };
        }
        match self {
            Cell::Int(value) => storage.integer(i128::from(*value)),
            Cell::Str(value) => {
                let view = numeric_view(value, decimal);
                let text = view.trim();
                match text.parse::<f64>() {
                    Ok(value) if value.is_nan() => Ok(value),
                    Ok(_) => storage.decimal(text),
                    Err(_) => Ok(f64::NAN),
                }
            }
            _ => {
                let value = self.to_numeric();
                if value.is_nan() {
                    Ok(value)
                } else {
                    storage.round(value)
                }
            }
        }
    }
    /// `py_str_scalar` of the pandas-typed value.
    pub fn to_str_scalar(&self) -> String {
        match self {
            Cell::Bool(b) => {
                if *b {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            Cell::Int(i) => i.to_string(),
            Cell::Float(f) => py_str_scalar(&Value::from(*f)),
            Cell::Str(s) => s.clone(),
            Cell::Na => "nan".to_string(),
        }
    }
    /// A hashable identity for `nunique`/grouping (pandas: all NaN are one group).
    fn group_key(&self) -> GroupKey {
        match self.join_key() {
            Some(key) => GroupKey::Value(key),
            None => GroupKey::Na,
        }
    }

    /// A join-match key: `Int(1)` and `Float(1.0)` unify; NaN/NA never match
    /// (returns `None`), mirroring pandas merge (NaN keys don't join).
    pub fn join_key(&self) -> Option<JoinKey> {
        match self {
            Cell::Bool(b) => Some(JoinKey::Bool(*b)),
            Cell::Int(i) => Some(JoinKey::Int(*i)),
            Cell::Float(f) if f.is_nan() => None,
            // The upper bound is exclusive: i64::MAX rounds up to 2^63 in
            // f64. A saturating cast must never equate that float with MAX.
            Cell::Float(f)
                if f.fract() == 0.0 && *f >= i64::MIN as f64 && *f < -(i64::MIN as f64) =>
            {
                Some(JoinKey::Int(*f as i64))
            }
            Cell::Float(f)
                if f.fract() == 0.0
                    && *f >= 9_223_372_036_854_775_808.0
                    && *f < 18_446_744_073_709_551_616.0 =>
            {
                Some(JoinKey::Uint(*f as u64))
            }
            Cell::Float(f) => Some(JoinKey::Num(f.to_bits())),
            Cell::Str(s) => Some(JoinKey::Str(s.clone())),
            Cell::Na => None,
        }
    }
}

/// Exact numeric identity: integral f64 values within i64 unify with integers;
/// other floats retain their bits. Both signed zeros represent integer zero.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JoinKey {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Num(u64),
    Lexeme(String),
    Str(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey {
    Value(JoinKey),
    Na,
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn typed_nonfinite_keys_keep_bits_and_never_fabricate_text_lexemes() {
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            let key = Cell::Float(value).join_key();
            assert_eq!(key, Some(JoinKey::Num(value.to_bits())));
            assert_eq!(key, Cell::Float(value).join_key());
            assert_ne!(key, Some(JoinKey::Lexeme(value.to_string())));
        }
        assert_ne!(
            Cell::Float(f64::INFINITY).join_key(),
            Cell::Float(f64::NEG_INFINITY).join_key()
        );
        assert_eq!(Cell::Float(f64::NAN).join_key(), None);
        assert_eq!(
            Cell::Float(f64::from_bits(0x7ff8000000000001)).join_key(),
            None
        );
        assert_eq!(Cell::Na.join_key(), None);
    }

    #[test]
    fn integer_identity_is_exact_at_float_and_i64_boundaries() {
        for integer in [
            i64::MIN,
            i64::MIN + 1,
            -(1_i64 << 53) - 1,
            (1_i64 << 53) - 1,
            1_i64 << 53,
            i64::MAX - 1,
        ] {
            assert_ne!(
                Cell::Int(integer).join_key(),
                Cell::Int(integer + 1).join_key()
            );
            assert_ne!(
                Cell::Int(integer).group_key(),
                Cell::Int(integer + 1).group_key()
            );
        }
        assert_ne!(
            Cell::Int(i64::MAX).join_key(),
            Cell::Float(i64::MAX as f64).join_key()
        );
        assert_ne!(
            Cell::Int((1_i64 << 53) + 1).join_key(),
            Cell::Float((1_i64 << 53) as f64).join_key()
        );
    }

    #[test]
    fn exactly_equal_numeric_keys_unify_without_losing_other_types() {
        for integer in [i64::MIN, -(1_i64 << 53), -1, 0, 1, 1_i64 << 53] {
            assert_eq!(
                Cell::Int(integer).join_key(),
                Cell::Float(integer as f64).join_key()
            );
            assert_eq!(
                Cell::Int(integer).group_key(),
                Cell::Float(integer as f64).group_key()
            );
        }
        assert_eq!(Cell::Float(-0.0).join_key(), Cell::Float(0.0).join_key());
        assert_ne!(Cell::Float(1.5).join_key(), Cell::Int(1).join_key());
        assert_ne!(
            Cell::Float(f64::INFINITY).join_key(),
            Cell::Int(i64::MAX).join_key()
        );
        assert_ne!(Cell::Str("1".into()).join_key(), Cell::Int(1).join_key());
        assert_ne!(Cell::Bool(true).join_key(), Cell::Int(1).join_key());
        assert_eq!(Cell::Float(f64::NAN).join_key(), None);
        assert_eq!(Cell::Na.join_key(), None);
        assert_eq!(Cell::Float(f64::NAN).group_key(), Cell::Na.group_key());
    }

    #[test]
    fn column_authority_keeps_unsigned_integer_lexeme_and_text_keys_distinct() {
        let mut numeric = Column::from_cells(
            "k",
            vec![
                Cell::Str("18446744073709549568".into()),
                Cell::Str("9223372036854775808".into()),
                Cell::Str("1e999".into()),
                Cell::Str(" 1e999".into()),
                Cell::Int(9_007_199_254_740_993),
                Cell::Float(9_007_199_254_740_993.0),
            ],
        );
        numeric.dtype = ColDtype::Numeric;
        assert_eq!(
            numeric.join_key_at(0),
            Cell::Float(18_446_744_073_709_549_568.0).join_key()
        );
        assert_eq!(
            numeric.join_key_at(1),
            Cell::Float(9_223_372_036_854_775_808.0).join_key()
        );
        assert_eq!(
            numeric.join_key_at(2),
            Some(JoinKey::Lexeme("1e999".into()))
        );
        assert_ne!(numeric.join_key_at(2), numeric.join_key_at(3));
        assert_ne!(numeric.join_key_at(4), numeric.join_key_at(5));
        assert_ne!(
            numeric.join_key_at(0),
            Cell::Str("18446744073709549568".into()).join_key()
        );
        let missing = Frame::from_columns(
            vec![Column::from_cells("k", vec![Cell::Na, Cell::Na])],
            "text",
        );
        assert_eq!(missing.join_key_tuples(&["k".into()]), vec![None, None]);
        assert!(missing.has_duplicate_keys(&["k".into()]));
    }

    #[test]
    fn numeric_view_changes_only_requested_text_and_uses_existing_storage() {
        let text = Cell::Str("0:12345678901234567".into());
        assert!(text.to_numeric_checked().unwrap().is_nan());
        assert_eq!(
            text.to_numeric_with_decimal(super::super::numeric::NumericStorage::Float64, None, ':')
                .unwrap()
                .to_bits(),
            0x3fbf9add3746f65e
        );
        assert_eq!(text, Cell::Str("0:12345678901234567".into()));
        let typed = Cell::Float(-0.0);
        assert_eq!(
            typed
                .to_numeric_with_decimal(super::super::numeric::NumericStorage::Float64, None, ':')
                .unwrap()
                .to_bits(),
            (-0.0_f64).to_bits()
        );
        let frame = Frame::from_columns(
            vec![Column::from_cells(
                "x",
                vec![Cell::Str("1:5".into()), Cell::Str("abc".into())],
            )],
            "text",
        );
        let matrix = frame
            .coerce_numeric_with_decimal(&["x".into()], None, None, |_, _| ':')
            .unwrap();
        assert_eq!(matrix.data[0].to_bits(), 1.5_f64.to_bits());
        assert_eq!(matrix.validity, vec![true, false]);
        assert_eq!(
            frame
                .feature_matrix_f32_with_decimal(&["x".into()], |_, _| ':')
                .0[0],
            1.5
        );
    }

    #[test]
    fn empty_alignment_keeps_loaded_cell_class() {
        let mut retained = Column::from_cells(
            "k",
            vec![Cell::Str("18446744073709551615".into()), Cell::Int(5)],
        );
        retained.dtype = ColDtype::Numeric;
        retained.numeric_kind = NumericKind::NonFloatNumeric;
        let empty = retained.aligned_cells("k", vec![], vec![]);
        assert_eq!(empty.dtype, ColDtype::String);
        assert_eq!(empty.numeric_kind, NumericKind::NonNumeric);
        assert!(empty.values.is_empty());
        let integer = Column::from_cells("k", vec![Cell::Int(5)]);
        assert_eq!(
            integer.aligned_cells("k", vec![], vec![]).dtype,
            ColDtype::Numeric
        );
        let text = Column::from_cells("k", vec![Cell::Str("five".into())]);
        assert_eq!(
            text.aligned_cells("k", vec![], vec![]).dtype,
            ColDtype::String
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub source_dtype: Option<String>,
    pub name: String,
    pub dtype: ColDtype,
    pub numeric_kind: NumericKind,
    pub values: Vec<Cell>,
    // Original numeric-column authority for retained text, aligned with rows.
    // Not part of the assembled or package wire representation.
    pub(crate) numeric_text: Vec<bool>,
}

pub(crate) fn metadata_cell_value(column: &Column, row: usize, cell: &Cell) -> Value {
    match cell {
        Cell::Bool(v) => Value::from(*v),
        Cell::Int(v) => Value::from(*v),
        Cell::Float(v) => Value::from(*v),
        Cell::Str(v) if column.numeric_text_at(row) => v
            .parse::<u64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::from(v.clone())),
        Cell::Str(v) => Value::from(v.clone()),
        Cell::Na => Value::Null,
    }
}

impl Column {
    fn join_key_at(&self, row: usize) -> Option<JoinKey> {
        let cell = &self.values[row];
        if let Cell::Str(text) = cell {
            if self.numeric_text_at(row) {
                let token = text.trim();
                let digits = token.trim_start_matches(['+', '-']);
                if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
                    if super::numeric::integer_in_domain(token) {
                        if let Ok(value) = token.parse::<i64>() {
                            return Some(JoinKey::Int(value));
                        }
                        return Some(JoinKey::Uint(
                            token
                                .parse::<u64>()
                                .expect("admitted positive integer fits u64"),
                        ));
                    }
                } else if let Ok(value) = super::numeric::checked_decimal_f64(token) {
                    return Cell::Float(value).join_key();
                }
                return Some(JoinKey::Lexeme(text.clone()));
            }
        }
        cell.join_key()
    }
    pub(crate) fn numeric_text_at(&self, row: usize) -> bool {
        self.dtype == ColDtype::Numeric || self.numeric_text.get(row).copied().unwrap_or(false)
    }
    /// Align existing cells without losing a numeric column's inference merely
    /// because an unsafe numeric lexeme is retained as text until role conversion.
    pub(crate) fn aligned_cells(
        &self,
        name: &str,
        cells: Vec<Cell>,
        numeric_text: Vec<bool>,
    ) -> Self {
        let mut column = Self::from_cells(name, cells);
        if column.values.is_empty() {
            // No survivor can re-infer a cell class. Retained numeric text is
            // an object carrier; ordinary typed columns keep their loaded dtype.
            column.dtype = if self.values.iter().any(|cell| matches!(cell, Cell::Str(_))) {
                ColDtype::String
            } else {
                self.dtype
            };
            column.numeric_kind = if column.dtype == ColDtype::String {
                NumericKind::NonNumeric
            } else {
                self.numeric_kind
            };
        } else if self.is_numeric_dtype()
            && !column.is_numeric_dtype()
            && column.values.iter().all(|cell| match cell {
                Cell::Str(text) => text.trim().parse::<f64>().is_ok(),
                _ => true,
            })
        {
            column.dtype = self.dtype;
            column.numeric_kind = self.numeric_kind;
        }
        column.source_dtype = self.source_dtype.clone();
        if numeric_text.iter().any(|numeric| *numeric) {
            column.numeric_text = numeric_text;
        }
        column
    }
    pub fn with_source_dtype(mut self, dtype: Option<String>) -> Self {
        self.source_dtype = dtype;
        self
    }
    pub fn is_float_dtype(&self) -> bool {
        self.numeric_kind == NumericKind::Float
    }
    pub fn is_numeric_dtype(&self) -> bool {
        self.numeric_kind != NumericKind::NonNumeric
    }
    pub fn nunique_with_na(&self) -> usize {
        self.values
            .iter()
            .map(Cell::group_key)
            .collect::<HashSet<_>>()
            .len()
    }
    /// Build a column from cells, inferring the pandas dtype: object if any
    /// string; bool if all-bool; int64 if all-int and no NA; else float64
    /// (with int/bool→float and NA→NaN promotion, as pandas does on a join/concat).
    pub fn from_cells(name: &str, cells: Vec<Cell>) -> Column {
        let mut any_str = false;
        let mut any_bool = false;
        let mut any_na = false;
        let mut all_int = true;
        let mut all_bool = true;
        for c in &cells {
            match c {
                Cell::Str(_) => {
                    any_str = true;
                    all_int = false;
                    all_bool = false;
                }
                Cell::Bool(_) => {
                    any_bool = true;
                    all_int = false;
                }
                Cell::Float(f) => {
                    all_int = false;
                    all_bool = false;
                    if f.is_nan() {
                        any_na = true;
                    }
                }
                Cell::Na => {
                    any_na = true;
                    all_int = false;
                }
                Cell::Int(_) => {
                    all_bool = false;
                }
            }
        }
        if any_str {
            Column {
                numeric_text: vec![],
                source_dtype: None,
                name: name.into(),
                dtype: ColDtype::String,
                numeric_kind: NumericKind::NonNumeric,
                values: cells,
            }
        } else if any_bool && all_bool {
            Column {
                numeric_text: vec![],
                source_dtype: None,
                name: name.into(),
                dtype: if any_na {
                    ColDtype::String
                } else {
                    ColDtype::Bool
                },
                numeric_kind: if any_na {
                    NumericKind::NonNumeric
                } else {
                    NumericKind::NonFloatNumeric
                },
                values: cells,
            }
        } else if all_int && !any_na {
            Column {
                numeric_text: vec![],
                source_dtype: None,
                name: name.into(),
                dtype: ColDtype::Numeric,
                numeric_kind: NumericKind::NonFloatNumeric,
                values: cells,
            }
        } else {
            let keep_integers = cells.iter().any(
                |c| matches!(c, Cell::Int(i) if !super::numeric::integer_exact_f64(i128::from(*i))),
            );
            let values = cells
                .into_iter()
                .map(|c| match c {
                    Cell::Bool(b) => Cell::Float(i64::from(b) as f64),
                    Cell::Int(i)
                        if !keep_integers && super::numeric::integer_exact_f64(i128::from(i)) =>
                    {
                        Cell::Float(i as f64)
                    }
                    Cell::Na => Cell::Float(f64::NAN),
                    other => other,
                })
                .collect();
            Column {
                numeric_text: vec![],
                source_dtype: None,
                name: name.into(),
                dtype: ColDtype::Numeric,
                numeric_kind: NumericKind::Float,
                values,
            }
        }
    }

    fn to_profile(&self, n_rows: usize) -> ColumnProfile {
        let nunique = self.nunique_with_na();
        ColumnProfile {
            name: self.name.clone(),
            dtype: self.dtype,
            numeric_kind: self.numeric_kind,
            nunique_with_na: nunique,
            is_unique: nunique == n_rows,
            str_values: self.values.iter().map(Cell::to_str_scalar).collect(),
            numeric_values: self.values.iter().map(Cell::to_numeric).collect(),
        }
    }
}

/// A column-major typed frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub columns: Vec<Column>,
    pub n_rows: usize,
    pub header_unit: String,
}

/// A row-major dense `f64` matrix (an assembled feature/variation block).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub validity: Vec<bool>,
    pub precision: super::numeric::PrecisionRecord,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Frame {
    pub fn empty() -> Self {
        Frame {
            columns: vec![],
            n_rows: 0,
            header_unit: "text".into(),
        }
    }

    /// Build a frame from columns (n_rows from the first column).
    pub fn from_columns(columns: Vec<Column>, header_unit: &str) -> Frame {
        let n_rows = columns.first().map(|c| c.values.len()).unwrap_or(0);
        Frame {
            columns,
            n_rows,
            header_unit: header_unit.into(),
        }
    }

    /// Per-row join keys over `keys` (`None` if any key cell is NA — never matches).
    pub fn join_key_tuples(&self, keys: &[String]) -> Vec<Option<Vec<JoinKey>>> {
        let cols: Vec<&Column> = keys.iter().filter_map(|k| self.column(k)).collect();
        (0..self.n_rows)
            .map(|row| {
                let mut tuple = Vec::with_capacity(cols.len());
                for c in &cols {
                    match c.join_key_at(row) {
                        Some(k) => tuple.push(k),
                        None => return None,
                    }
                }
                Some(tuple)
            })
            .collect()
    }

    /// Whether any two rows share a key (pandas `duplicated(subset=keys)`; NA groups).
    pub fn has_duplicate_keys(&self, keys: &[String]) -> bool {
        let cols: Vec<&Column> = keys.iter().filter_map(|k| self.column(k)).collect();
        let mut seen: HashSet<Vec<GroupKey>> = HashSet::new();
        for row in 0..self.n_rows {
            let tuple: Vec<GroupKey> = cols
                .iter()
                .map(|c| {
                    c.join_key_at(row)
                        .map(GroupKey::Value)
                        .unwrap_or(GroupKey::Na)
                })
                .collect();
            if !seen.insert(tuple) {
                return true;
            }
        }
        false
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    pub fn dtype_labels(&self) -> Vec<String> {
        self.columns
            .iter()
            .map(|c| c.dtype.label().to_string())
            .collect()
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.column(name).is_some()
    }

    pub fn to_table_profile(&self) -> TableProfile {
        TableProfile {
            n_rows: self.n_rows,
            columns: self
                .columns
                .iter()
                .map(|c| c.to_profile(self.n_rows))
                .collect(),
        }
    }

    /// `to_numeric(errors="coerce")` of a column as f64 (for task detection).
    pub fn numeric_column_f64(&self, name: &str) -> Vec<f64> {
        self.column(name)
            .map(|c| c.values.iter().map(Cell::to_numeric).collect())
            .unwrap_or_default()
    }

    /// `py_str_scalar` rendering of a column (for ids / coverage).
    pub fn str_column(&self, name: &str) -> Vec<String> {
        self.column(name)
            .map(|c| c.values.iter().map(Cell::to_str_scalar).collect())
            .unwrap_or_default()
    }

    /// Row-major feature matrix over `cols`, f32-widened to f64 (mirrors
    /// `coerce_numeric` → float32 → widen) for `detect_signal_type`.
    pub fn feature_matrix_f32(&self, cols: &[String]) -> (Vec<f64>, usize) {
        self.feature_matrix_f32_with_decimal(cols, |_, _| '.')
    }
    pub(crate) fn feature_matrix_f32_with_decimal(
        &self,
        cols: &[String],
        decimal: impl Fn(&str, usize) -> char,
    ) -> (Vec<f64>, usize) {
        let columns = cols
            .iter()
            .map(|name| self.column(name))
            .collect::<Vec<_>>();
        let mut values = Vec::with_capacity(self.n_rows * cols.len());
        for row in 0..self.n_rows {
            for (name, column) in cols.iter().zip(&columns) {
                values.push(column.map_or(f64::NAN, |c| {
                    c.values[row].to_numeric_view(decimal(name, row))
                }) as f32 as f64);
            }
        }
        (values, cols.len())
    }

    /// `coerce_numeric(df[cols])` → row-major `f64` matrix (non-numeric → NaN).
    pub fn coerce_numeric(&self, cols: &[String]) -> Result<Matrix, String> {
        self.coerce_numeric_with_storage(cols, None, None)
    }
    pub fn coerce_numeric_with_storage(
        &self,
        cols: &[String],
        request: Option<super::numeric::NumericStorage>,
        consumer: Option<&str>,
    ) -> Result<Matrix, String> {
        self.coerce_numeric_with_decimal(cols, request, consumer, |_, _| '.')
    }
    pub(crate) fn coerce_numeric_with_decimal(
        &self,
        cols: &[String],
        request: Option<super::numeric::NumericStorage>,
        consumer: Option<&str>,
        decimal: impl Fn(&str, usize) -> char,
    ) -> Result<Matrix, String> {
        let precision = super::numeric::PrecisionRecord::select(
            self.numeric_source_dtype(cols).as_deref(),
            request,
            consumer,
            true,
        )?;
        let n_cols = cols.len();
        // Resolve each column once: a per-cell name lookup is quadratic in the
        // column count, which dominates wide spectral tables.
        let columns: Vec<Option<&Column>> = cols.iter().map(|name| self.column(name)).collect();
        let mut data = Vec::with_capacity(self.n_rows * n_cols);
        for row in 0..self.n_rows {
            for (name, column) in cols.iter().zip(&columns) {
                let v = column
                    .map(|c| {
                        c.values[row].to_numeric_with_decimal(
                            precision.dtype,
                            c.source_dtype.as_deref(),
                            decimal(name, row),
                        )
                    })
                    .transpose()?
                    .unwrap_or(f64::NAN);
                data.push(v);
            }
        }
        let mut matrix = Matrix::new(data, self.n_rows, n_cols);
        matrix.precision = precision;
        Ok(matrix)
    }

    pub fn numeric_source_dtype(&self, names: &[String]) -> Option<String> {
        let dtypes = names
            .iter()
            .map(|name| self.column(name).and_then(|c| c.source_dtype.clone()))
            .collect::<HashSet<_>>();
        if dtypes.len() > 1 {
            Some("mixed".into())
        } else {
            dtypes.into_iter().next().flatten()
        }
    }

    /// Append an int64 column (e.g. the row-index sentinel).
    pub fn add_int_column(&mut self, name: &str, values: Vec<i64>) {
        self.columns.push(Column {
            numeric_text: vec![],
            source_dtype: None,
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: NumericKind::NonFloatNumeric,
            values: values.into_iter().map(Cell::Int).collect(),
        });
    }

    /// Append an object (string) column.
    pub fn add_str_column(&mut self, name: &str, values: Vec<String>) {
        self.columns.push(Column {
            numeric_text: vec![],
            source_dtype: None,
            name: name.into(),
            dtype: ColDtype::String,
            numeric_kind: NumericKind::NonNumeric,
            values: values.into_iter().map(Cell::Str).collect(),
        });
    }

    /// A new frame with only `cols` (preserving order), rows unchanged.
    pub fn select(&self, cols: &[String]) -> Frame {
        let columns = cols
            .iter()
            .filter_map(|name| self.column(name).cloned())
            .collect();
        Frame {
            columns,
            n_rows: self.n_rows,
            header_unit: self.header_unit.clone(),
        }
    }

    /// A new frame keeping only rows where `mask[i]` is true.
    pub fn mask_rows(&self, mask: &[bool]) -> Frame {
        let columns = self
            .columns
            .iter()
            .map(|c| Column {
                numeric_text: c
                    .numeric_text
                    .iter()
                    .zip(mask)
                    .filter(|(_, keep)| **keep)
                    .map(|(numeric, _)| *numeric)
                    .collect(),
                source_dtype: c.source_dtype.clone(),
                name: c.name.clone(),
                dtype: c.dtype,
                numeric_kind: c.numeric_kind,
                values: c
                    .values
                    .iter()
                    .zip(mask)
                    .filter(|(_, &m)| m)
                    .map(|(v, _)| v.clone())
                    .collect(),
            })
            .collect();
        Frame {
            columns,
            n_rows: mask.iter().filter(|&&m| m).count(),
            header_unit: self.header_unit.clone(),
        }
    }
}

impl Matrix {
    /// F64 execution IR; NaN denotes tabular missingness until owner normalization.
    pub fn new(data: Vec<f64>, n_rows: usize, n_cols: usize) -> Self {
        let validity = data.iter().map(|v| !v.is_nan()).collect();
        Self {
            data,
            n_rows,
            n_cols,
            validity,
            precision: Default::default(),
        }
    }
    pub fn with_storage(
        &self,
        request: Option<super::numeric::NumericStorage>,
        consumer: Option<&str>,
        feature: bool,
    ) -> Result<Self, String> {
        use super::numeric::{NumericStorage, PrecisionRecord};
        let count = self
            .n_rows
            .checked_mul(self.n_cols)
            .ok_or("Matrix shape overflow")?;
        if count != self.data.len() || count != self.validity.len() {
            return Err("Matrix shape/validity length mismatch".into());
        }
        self.precision.validate(feature)?;
        // This IR already declares selected storage. Verify it before a new
        // request can round anything; default repackaging must not repair edits.
        for (&value, &valid) in self.data.iter().zip(&self.validity) {
            if valid && self.precision.dtype.round(value)?.to_bits() != value.to_bits() {
                return Err("Numeric values no longer match carried storage precision".into());
            }
        }
        let supplied_request = request;
        let request = request.or(self.precision.requested_dtype);
        let consumer = consumer.or_else(|| {
            if supplied_request.is_none() {
                self.precision.consumer.as_deref()
            } else {
                None
            }
        });
        let precision = PrecisionRecord::select(
            self.precision.source_dtype.as_deref(),
            request,
            consumer,
            feature,
        )?;
        if self.precision.dtype == NumericStorage::Float32
            && precision.dtype == NumericStorage::Float64
            && self.precision.source_dtype.as_deref() != Some("float32")
        {
            return Err("Cannot re-select float64 storage for data already normalized to float32 from a different source".into());
        }
        let mut data = Vec::with_capacity(count);
        for (value, valid) in self.data.iter().zip(&self.validity) {
            if !valid {
                data.push(0.0);
                continue;
            }
            let source = if precision.source_dtype.as_deref() == Some("float32") {
                NumericStorage::Float32.round(*value)?
            } else {
                *value
            };
            data.push(precision.dtype.round(source)?);
        }
        Ok(Self {
            data,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            validity: self.validity.clone(),
            precision,
        })
    }
    pub fn validate_storage(&self, feature: bool) -> Result<(), String> {
        self.precision.validate(feature)?;
        let count = self
            .n_rows
            .checked_mul(self.n_cols)
            .ok_or("Matrix shape overflow")?;
        if count != self.data.len() || count != self.validity.len() {
            return Err("Matrix shape/validity mismatch".into());
        }
        for (&value, &valid) in self.data.iter().zip(&self.validity) {
            if !valid {
                if value.to_bits() != 0 {
                    return Err("False numeric placeholder must be positive zero".into());
                }
            } else if self.precision.dtype.round(value)?.to_bits() != value.to_bits() {
                return Err("Numeric values disagree with storage dtype".into());
            }
        }
        Ok(())
    }
    pub fn full_value(&self) -> serde_json::Value {
        super::numeric::numeric_full(
            &self.data,
            self.n_rows,
            self.n_cols,
            &self.validity,
            &self.precision,
        )
    }
    /// Strict inverse of the full numeric wire (fixed little-endian storage).
    ///
    /// Shape arithmetic, encoded length and bitmap admission precede allocation.
    /// Carried precision and observed/placeholder values use the existing owner
    /// validators; decoding never repairs a record or re-selects its storage.
    pub fn from_full_value(value: &Value) -> Result<Self, String> {
        use super::numeric::{decode_hex, NumericStorage, PrecisionRecord};
        let fields = [
            "dtype",
            "n_rows",
            "n_cols",
            "data_bytes_hex",
            "validity",
            "precision",
        ];
        let object = value.as_object().ok_or("Numeric wire object required")?;
        if object.len() != fields.len() || fields.iter().any(|key| !object.contains_key(*key)) {
            return Err("Numeric wire requires exactly six fields".into());
        }
        let dimension = |name: &str| -> Result<usize, String> {
            let integer = value[name]
                .as_u64()
                .ok_or("Numeric wire shape must contain unsigned integers")?;
            usize::try_from(integer).map_err(|_| "Numeric wire shape exceeds usize".into())
        };
        let n_rows = dimension("n_rows")?;
        let n_cols = dimension("n_cols")?;
        let count = n_rows.checked_mul(n_cols).ok_or("Matrix shape overflow")?;
        if !value["dtype"].is_string() {
            return Err("Numeric wire dtype string required".into());
        }
        let dtype: NumericStorage =
            serde_json::from_value(value["dtype"].clone()).map_err(|e| e.to_string())?;
        let width = match dtype {
            NumericStorage::Float32 => 4,
            NumericStorage::Float64 => 8,
        };
        let byte_count = count
            .checked_mul(width)
            .ok_or("Numeric wire byte count overflow")?;
        // Matrix uses f64 internally even for a carried f32 block.
        if count
            .checked_mul(std::mem::size_of::<f64>())
            .is_none_or(|size| size > isize::MAX as usize)
        {
            return Err("Numeric wire allocation exceeds capacity".into());
        }
        let encoded = value["data_bytes_hex"]
            .as_str()
            .ok_or("Numeric wire hex string required")?;
        if byte_count.checked_mul(2) != Some(encoded.len()) {
            return Err("Numeric wire byte count mismatch".into());
        }
        let bitmap = value["validity"]
            .as_array()
            .ok_or("Numeric wire validity array required")?;
        if bitmap.len() != count || bitmap.iter().any(|flag| !flag.is_boolean()) {
            return Err("Numeric wire validity must contain exactly one boolean per cell".into());
        }
        if value["precision"].as_object().is_none_or(|record| {
            record.len() != 5
                || record
                    .values()
                    .any(|field| !field.is_null() && !field.is_string())
        }) {
            return Err("Numeric wire precision requires five scalar fields".into());
        }
        let precision: PrecisionRecord =
            serde_json::from_value(value["precision"].clone()).map_err(|e| e.to_string())?;
        if dtype != precision.dtype {
            return Err("Numeric wire dtype disagrees with precision record".into());
        }
        let bytes = decode_hex(encoded, byte_count)?;
        let mut data = Vec::new();
        data.try_reserve_exact(count)
            .map_err(|_| "Numeric wire allocation exceeds capacity")?;
        for cell in bytes.chunks_exact(width) {
            data.push(match dtype {
                NumericStorage::Float32 => f64::from(f32::from_le_bytes(cell.try_into().unwrap())),
                NumericStorage::Float64 => f64::from_le_bytes(cell.try_into().unwrap()),
            });
        }
        let mut validity = Vec::new();
        validity
            .try_reserve_exact(count)
            .map_err(|_| "Numeric wire allocation exceeds capacity")?;
        validity.extend(bitmap.iter().map(|flag| flag.as_bool().unwrap()));
        let matrix = Self {
            data,
            n_rows,
            n_cols,
            validity,
            precision,
        };
        // The decoder is generic: consumer precision may govern a feature or a
        // target. A feature-only record must still be valid in its actual role.
        let feature = matrix.precision.validate(false).is_err();
        matrix.validate_storage(feature)?;
        Ok(matrix)
    }
    pub fn numeric_bytes(&self) -> Vec<u8> {
        use super::numeric::{bytes, u64_bytes, NumericStorage};
        let mut values = Vec::new();
        for (value, valid) in self.data.iter().zip(&self.validity) {
            let v = if *valid { *value } else { 0.0 };
            match self.precision.dtype {
                NumericStorage::Float32 => values.extend_from_slice(&(v as f32).to_le_bytes()),
                NumericStorage::Float64 => values.extend_from_slice(&v.to_le_bytes()),
            }
        }
        let mut out = vec![self.precision.dtype.tag()];
        u64_bytes(&mut out, self.n_rows);
        u64_bytes(&mut out, self.n_cols);
        bytes(&mut out, &values);
        bytes(
            &mut out,
            &self
                .validity
                .iter()
                .map(|v| u8::from(*v))
                .collect::<Vec<_>>(),
        );
        out
    }
    pub fn column_matrix(&self, column: usize) -> Self {
        Self {
            data: (0..self.n_rows)
                .map(|r| self.data[r * self.n_cols + column])
                .collect(),
            validity: (0..self.n_rows)
                .map(|r| self.validity[r * self.n_cols + column])
                .collect(),
            n_rows: self.n_rows,
            n_cols: 1,
            precision: self.precision.clone(),
        }
    }
}
