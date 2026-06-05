// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Tabular loaders — the facade's file-reading entry (ports the IO half of
//! `materialize/loaders.py`).
//!
//! The pure CSV-from-bytes decoder, NA policy, dtype inference and param merge
//! moved into `nirs4all-io-core` so the WASM binding can reach them. This module
//! keeps only the filesystem side: read a file, transparently decompress
//! `.gz`/`.zip`, then delegate to the shared core decoder. A single decoder thus
//! backs both the native (path) and the in-memory paths (D-R4).

use std::path::Path;

use nirs4all_io_core::materialize::loaders::read_table_bytes;
use nirs4all_io_core::spec::dataset_spec::LoadingParams;
use nirs4all_io_core::spec::SpecError;

use super::frame::Frame;

pub use nirs4all_io_core::materialize::loaders::{effective_params, LoadedTable};

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

/// Read a tabular file into a [`Frame`]: read bytes (+ gzip/zip), then run the
/// shared core CSV decoder. v0: the CSV family. numpy/parquet/excel/vendor
/// readers land with the broader load path; until then unknown extensions fall
/// back to CSV (nirs4all's own fallback).
pub fn read_table(path: &Path, params: &LoadingParams) -> Result<Frame, SpecError> {
    let bytes = read_maybe_compressed(path)?;
    read_table_bytes(&bytes, params)
        .map_err(|e| SpecError::new(format!("{} in {}", e.message, path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nirs4all_io_core::infer::table::NumericKind;
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
