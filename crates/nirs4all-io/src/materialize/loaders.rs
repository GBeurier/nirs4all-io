// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Tabular loaders (ports `materialize/loaders.py`).
//!
//! Reads a delimited file into a typed [`Frame`] whose per-column dtype
//! inference mirrors pandas: a column is int64 only if every non-NA cell is an
//! integer literal and there is no NA; float64 if every non-NA cell parses as a
//! float (NA promotes int→float); object/string otherwise. The facade is the
//! single place that mirrors pandas, then hands neutral `TableProfile`s + numeric
//! arrays to the pure core (D-R4).
//!
//! v0 reads the CSV family (+ gzip/zip); numpy/parquet/excel/vendor readers land
//! with the broader load path.

use std::path::Path;
use std::sync::LazyLock;

use nirs4all_io_core::infer::table::{ColDtype, NumericKind};
use nirs4all_io_core::spec::dataset_spec::{FormatParams, LoadingParams, NaConfig};
use nirs4all_io_core::spec::enums::NaPolicy;
use nirs4all_io_core::spec::SpecError;

use super::frame::{Cell, Column, Frame};

/// Back-compat alias for the inference path.
pub type LoadedTable = Frame;

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
static NA_TOKENS: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
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

fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

fn infer_column(name: &str, raw: &[String], decimal: char) -> Column {
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
        Column {
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: NumericKind::NonFloatNumeric,
            values: raw
                .iter()
                .map(|c| Cell::Int(parse_int_literal(&norm(c)).unwrap()))
                .collect(),
        }
    } else if all_numeric {
        Column {
            name: name.into(),
            dtype: ColDtype::Numeric,
            numeric_kind: NumericKind::Float,
            values: raw
                .iter()
                .map(|c| {
                    if is_na(c) {
                        Cell::Float(f64::NAN)
                    } else {
                        Cell::Float(parse_float(&norm(c)).unwrap())
                    }
                })
                .collect(),
        }
    } else {
        Column {
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

/// Read a file, transparently decompressing `.gz` / `.zip` so compressed CSVs parse.
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

fn header_unit(params: &LoadingParams, has_header: bool) -> String {
    match params.header_unit {
        Some(u) => u.value().to_string(),
        None => if has_header { "text" } else { "index" }.to_string(),
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

fn read_csv(path: &Path, params: &LoadingParams) -> Result<Frame, SpecError> {
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
        if rec.iter().all(|c| c.is_empty()) {
            continue; // skip_blank_lines
        }
        records.push(rec.iter().map(str::to_string).collect());
    }
    if records.is_empty() {
        return Ok(Frame {
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
    Ok(Frame {
        columns,
        n_rows: records.len() - data_start,
        header_unit: header_unit(params, has_header),
    })
}

/// Read a tabular file into a [`Frame`]. v0: CSV family (+ gzip/zip).
pub fn read_table(path: &Path, params: &LoadingParams) -> Result<Frame, SpecError> {
    // numpy/parquet/excel/vendor readers land with the broader load path; until
    // then unknown extensions fall back to CSV (nirs4all's own fallback).
    read_csv(path, params)
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
        assert_eq!(prof.column("1000").unwrap().nunique_with_na, 1);
        assert!(prof.column("protein").unwrap().is_unique);
        assert!(prof.column("protein").unwrap().is_float_dtype());
        assert_eq!(t.numeric_column_f64("protein"), vec![12.5, 8.3, 15.1]);
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
        let prof = read_table(&p, &params).unwrap().to_table_profile();
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
