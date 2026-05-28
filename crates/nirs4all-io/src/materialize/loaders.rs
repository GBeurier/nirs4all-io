// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Tabular loaders (ports `materialize/loaders.py`).
//!
//! Reads a delimited file into a typed table whose per-column dtype inference
//! mirrors pandas: a column is int64 only if every non-NA cell is an integer
//! literal and there is no NA; float64 if every non-NA cell parses as a float
//! (NA promotes int→float); object/string otherwise. The facade is the single
//! place that mirrors pandas, then hands neutral `TableProfile`s + numeric arrays
//! to the pure core (D-R4).
//!
//! v0 reads the CSV family (the infer/SYNC-#1 path); numpy/parquet/excel/vendor
//! readers land with the materialize/load path.

use std::path::Path;
use std::sync::LazyLock;

use nirs4all_io_core::infer::table::{ColDtype, ColumnProfile, NumericKind, TableProfile};
use nirs4all_io_core::pyfmt::py_str_scalar;
use nirs4all_io_core::spec::dataset_spec::LoadingParams;
use nirs4all_io_core::spec::SpecError;
use serde_json::Value;

/// pandas `keep_default_na=True` default set ∪ io's `_NA_VALUES`.
static NA_TOKENS: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
    [
        "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
        "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null", "",
    ]
    .into_iter()
    .collect()
});

#[derive(Debug, Clone)]
enum PandasValue {
    Int(i64),
    Float(f64),
    Str(String),
    Na,
}

#[derive(Debug, Clone)]
struct LoadedColumn {
    name: String,
    dtype: ColDtype,
    numeric_kind: NumericKind,
    values: Vec<PandasValue>,
}

#[derive(Debug, Clone)]
pub struct LoadedTable {
    columns: Vec<LoadedColumn>,
    pub n_rows: usize,
    pub header_unit: String,
}

fn is_na(cell: &str) -> bool {
    NA_TOKENS.contains(cell)
}

/// pandas integer-literal test: optional sign + ASCII digits (no `.`/`e`).
fn parse_int_literal(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
        t.parse::<i64>().ok()
    } else {
        None
    }
}

/// pandas float parse (after decimal normalization).
fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    // pandas accepts inf/-inf/nan (case-insensitive) — but those are NA tokens here.
    t.parse::<f64>().ok()
}

fn infer_column(name: &str, raw: &[String], decimal: char) -> LoadedColumn {
    let norm = |c: &str| -> String {
        if decimal == ',' {
            c.replace(',', ".")
        } else {
            c.to_string()
        }
    };
    let mut has_na = false;
    let mut all_int = true;
    let mut all_numeric = true;
    for cell in raw {
        if is_na(cell) {
            has_na = true;
            continue;
        }
        let n = norm(cell);
        if parse_int_literal(&n).is_none() {
            all_int = false;
        }
        if parse_float(&n).is_none() {
            all_numeric = false;
        }
    }

    if all_numeric && all_int && !has_na {
        let values = raw
            .iter()
            .map(|c| PandasValue::Int(parse_int_literal(&norm(c)).unwrap()))
            .collect();
        LoadedColumn {
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: NumericKind::NonFloatNumeric,
            values,
        }
    } else if all_numeric {
        let values = raw
            .iter()
            .map(|c| {
                if is_na(c) {
                    PandasValue::Float(f64::NAN)
                } else {
                    PandasValue::Float(parse_float(&norm(c)).unwrap())
                }
            })
            .collect();
        LoadedColumn {
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: NumericKind::Float,
            values,
        }
    } else {
        let values = raw
            .iter()
            .map(|c| {
                if is_na(c) {
                    PandasValue::Na
                } else {
                    PandasValue::Str(c.clone())
                }
            })
            .collect();
        LoadedColumn {
            name: name.into(),
            dtype: ColDtype::String,
            numeric_kind: NumericKind::NonNumeric,
            values,
        }
    }
}

impl PandasValue {
    /// `to_numeric(errors="coerce")` → f64 (NaN for non-numeric/NA).
    fn to_numeric(&self) -> f64 {
        match self {
            PandasValue::Int(i) => *i as f64,
            PandasValue::Float(f) => *f,
            PandasValue::Str(s) => s.trim().parse::<f64>().unwrap_or(f64::NAN),
            PandasValue::Na => f64::NAN,
        }
    }
    /// `py_str_scalar` of the pandas-typed value.
    fn to_str_scalar(&self) -> String {
        match self {
            PandasValue::Int(i) => i.to_string(),
            PandasValue::Float(f) => py_str_scalar(&Value::from(*f)),
            PandasValue::Str(s) => s.clone(),
            PandasValue::Na => "nan".to_string(),
        }
    }
}

impl LoadedColumn {
    fn nunique_with_na(&self) -> usize {
        let mut distinct: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut strs: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut has_na = false;
        for v in &self.values {
            match v {
                PandasValue::Int(i) => {
                    distinct.insert((*i as f64).to_bits());
                }
                PandasValue::Float(f) => {
                    if f.is_nan() {
                        has_na = true;
                    } else {
                        distinct.insert(f.to_bits());
                    }
                }
                PandasValue::Str(s) => {
                    strs.insert(s.as_str());
                }
                PandasValue::Na => has_na = true,
            }
        }
        distinct.len() + strs.len() + usize::from(has_na)
    }

    fn to_profile(&self, n_rows: usize) -> ColumnProfile {
        let nunique = self.nunique_with_na();
        ColumnProfile {
            name: self.name.clone(),
            dtype: self.dtype,
            numeric_kind: self.numeric_kind,
            nunique_with_na: nunique,
            is_unique: nunique == n_rows,
            str_values: self.values.iter().map(PandasValue::to_str_scalar).collect(),
            numeric_values: self.values.iter().map(PandasValue::to_numeric).collect(),
        }
    }
}

impl LoadedTable {
    pub fn column_names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// Per-column dtype labels for selector resolution.
    pub fn dtype_labels(&self) -> Vec<String> {
        self.columns
            .iter()
            .map(|c| c.dtype.label().to_string())
            .collect()
    }

    /// Build the neutral [`TableProfile`] core consumes.
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

    fn column(&self, name: &str) -> Option<&LoadedColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// `to_numeric(errors="coerce")` of a column as f64 (for task detection).
    pub fn numeric_column_f64(&self, name: &str) -> Vec<f64> {
        self.column(name)
            .map(|c| c.values.iter().map(PandasValue::to_numeric).collect())
            .unwrap_or_default()
    }

    /// Row-major feature matrix over `cols`, f32-widened to f64 (mirrors
    /// `coerce_numeric` → float32 → widen) for `detect_signal_type`.
    pub fn feature_matrix_f32(&self, cols: &[String]) -> (Vec<f64>, usize) {
        let ncols = cols.len();
        let mut out = Vec::with_capacity(self.n_rows * ncols);
        for row in 0..self.n_rows {
            for name in cols {
                let v = self
                    .column(name)
                    .map(|c| c.values[row].to_numeric())
                    .unwrap_or(f64::NAN);
                out.push(v as f32 as f64);
            }
        }
        (out, ncols)
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.column(name).is_some()
    }
}

fn ext_lower(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    for compound in [".csv.gz", ".csv.zip"] {
        if name.ends_with(compound) {
            return compound.to_string();
        }
    }
    match name.rfind('.') {
        Some(idx) => name[idx..].to_string(),
        None => String::new(),
    }
}

/// Read a file, transparently decompressing `.gz` (gzip) and `.zip` (first
/// entry) so compressed CSVs (`.csv.gz`/`.csv.zip`) parse correctly.
fn read_maybe_compressed(path: &Path) -> Result<Vec<u8>, SpecError> {
    use std::io::Read;
    let raw = std::fs::read(path)
        .map_err(|e| SpecError::new(format!("file not found: {} ({e})", path.display())))?;
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".gz") {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .map_err(|e| {
                SpecError::new(format!("gzip decode failed for {}: {e}", path.display()))
            })?;
        Ok(out)
    } else if lower.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(raw))
            .map_err(|e| SpecError::new(format!("zip open failed for {}: {e}", path.display())))?;
        if archive.is_empty() {
            return Err(SpecError::new(format!(
                "empty zip archive: {}",
                path.display()
            )));
        }
        let mut entry = archive.by_index(0).map_err(|e| {
            SpecError::new(format!("zip entry read failed for {}: {e}", path.display()))
        })?;
        let mut out = Vec::new();
        entry.read_to_end(&mut out).map_err(|e| {
            SpecError::new(format!("zip decompress failed for {}: {e}", path.display()))
        })?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

fn read_csv(path: &Path, params: &LoadingParams) -> Result<LoadedTable, SpecError> {
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
    let bytes = read_maybe_compressed(path)?;
    let text = match String::from_utf8(bytes.clone()) {
        Ok(t) => t,
        Err(_) => bytes.iter().map(|&b| b as char).collect(), // latin-1 fallback
    };

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .delimiter(delimiter as u8)
        .from_reader(text.as_bytes());
    let mut records: Vec<Vec<String>> = Vec::new();
    for rec in rdr.records() {
        let rec =
            rec.map_err(|e| SpecError::new(format!("csv parse error in {}: {e}", path.display())))?;
        // skip_blank_lines: drop fully-empty rows
        if rec.iter().all(|c| c.is_empty()) {
            continue;
        }
        records.push(rec.iter().map(str::to_string).collect());
    }
    if records.is_empty() {
        return Ok(LoadedTable {
            columns: vec![],
            n_rows: 0,
            header_unit: header_unit(params, has_header),
        });
    }

    let ncols = records.iter().map(|r| r.len()).max().unwrap_or(0);
    let (header, data_start) = if has_header {
        (records[0].clone(), 1)
    } else {
        ((0..ncols).map(|i| i.to_string()).collect(), 0)
    };
    let names: Vec<String> = mangle_dupes(&header, ncols);

    let mut columns: Vec<LoadedColumn> = Vec::with_capacity(ncols);
    for (j, name) in names.iter().enumerate() {
        let raw: Vec<String> = records[data_start..]
            .iter()
            .map(|r| r.get(j).cloned().unwrap_or_default())
            .collect();
        columns.push(infer_column(name, &raw, decimal));
    }
    let n_rows = records.len() - data_start;
    Ok(LoadedTable {
        columns,
        n_rows,
        header_unit: header_unit(params, has_header),
    })
}

fn header_unit(params: &LoadingParams, has_header: bool) -> String {
    match params.header_unit {
        Some(u) => u.value().to_string(),
        None => if has_header { "text" } else { "index" }.to_string(),
    }
}

/// pandas `mangle_dupe_cols`: `a, a.1, a.2`. Also stringifies + pads to `ncols`.
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

/// Read a tabular file into a [`LoadedTable`]. v0: CSV family.
pub fn read_table(path: &Path, params: &LoadingParams) -> Result<LoadedTable, SpecError> {
    let ext = ext_lower(path);
    if matches!(
        ext.as_str(),
        ".csv" | ".tsv" | ".txt" | ".csv.gz" | ".csv.zip"
    ) || (matches!(ext.as_str(), ".gz" | ".zip")
        && path.to_string_lossy().to_lowercase().contains(".csv"))
    {
        read_csv(path, params)
    } else {
        // Unknown: try CSV (nirs4all fallback). numpy/parquet/excel/vendor land later.
        read_csv(path, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::File::create(&p)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        p
    }

    #[test]
    fn reads_combined_float_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let mut text = String::from("1000;1005;protein\n");
        for y in ["12.5", "8.3", "15.1"] {
            text.push_str(&format!("0.40;1.30;{y}\n"));
        }
        let p = write(tmp.path(), "data.csv", &text);
        let params = LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            ..Default::default()
        };
        let t = read_table(&p, &params).unwrap();
        assert_eq!(t.column_names(), vec!["1000", "1005", "protein"]);
        assert_eq!(t.dtype_labels(), vec!["numeric", "numeric", "numeric"]);
        let prof = t.to_table_profile();
        // wl col constant -> nunique 1; protein -> 3 distinct (unique)
        assert_eq!(prof.column("1000").unwrap().nunique_with_na, 1);
        assert!(prof.column("protein").unwrap().is_unique);
        assert!(prof.column("protein").unwrap().is_float_dtype());
        let task = t.numeric_column_f64("protein");
        assert_eq!(task, vec![12.5, 8.3, 15.1]);
    }

    #[test]
    fn int_column_stays_nonfloat() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "ids.csv", "id;v\n1;0.5\n2;0.6\n");
        let params = LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            ..Default::default()
        };
        let t = read_table(&p, &params).unwrap();
        let prof = t.to_table_profile();
        let idc = prof.column("id").unwrap();
        assert_eq!(idc.numeric_kind, NumericKind::NonFloatNumeric);
        assert!(!idc.is_float_dtype());
        assert!(idc.is_numeric_dtype());
        assert_eq!(idc.str_values, vec!["1", "2"]);
    }

    #[test]
    fn reads_gzip_csv() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("X.csv.gz");
        let mut enc = GzEncoder::new(std::fs::File::create(&p).unwrap(), Compression::default());
        enc.write_all(b"1000;1005\n0.4;1.3\n0.5;1.2\n").unwrap();
        enc.finish().unwrap();
        let params = LoadingParams {
            delimiter: Some(";".into()),
            has_header: Some(true),
            ..Default::default()
        };
        let t = read_table(&p, &params).unwrap();
        assert_eq!(t.column_names(), vec!["1000", "1005"]);
        assert_eq!(t.n_rows, 2);
        assert_eq!(t.numeric_column_f64("1000"), vec![0.4, 0.5]);
    }
}
