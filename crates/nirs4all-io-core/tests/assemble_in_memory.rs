// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! fs-free assemble parity (Phase C).
//!
//! Proves the in-memory assembly core agrees with the filesystem path:
//!
//! 1. `Bytes` payloads: the 3 contract corpus cases, inferred fs-free via
//!    `infer_named_bytes`, assembled via `assemble_in_memory`, must reproduce the
//!    blessed `*.assembled.canonical` goldens byte-for-byte. Those same goldens
//!    are what the facade's `assemble(spec, base_dir)` matches (the facade now
//!    delegates to `assemble_in_memory`), so this is the "two paths agree" guard.
//!
//! 2. `Records` payloads: a dataset built from pre-decoded nirs4all-formats
//!    records assembles to the *same* `to_summary_value()` as the equivalent CSV
//!    bytes, proving `records_to_frame` produces the same typed `Frame` as the
//!    proven CSV decoder.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nirs4all_io_core::canonical_json;
use nirs4all_io_core::infer::memory::{infer_named_bytes, NamedBytes};
use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
use nirs4all_io_core::spec::{validate_spec, DatasetSpec};
use serde_json::{json, Value};

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .join("tests/goldens/contract")
}

/// Read every file in a corpus case directory into named byte buffers
/// (basename-keyed, matching what the in-memory resolver re-matches against).
fn read_case_files(case_dir: &Path) -> Vec<NamedBytes> {
    let mut files: Vec<NamedBytes> = std::fs::read_dir(case_dir)
        .expect("read case dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .map(|e| NamedBytes {
            name: e.file_name().to_string_lossy().into_owned(),
            bytes: std::fs::read(e.path()).expect("read fixture"),
        })
        .collect();
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files
}

fn to_sources(files: &[NamedBytes]) -> Vec<InMemorySource> {
    files
        .iter()
        .map(|f| InMemorySource {
            name: f.name.clone(),
            payload: SourcePayload::Bytes(f.bytes.clone()),
        })
        .collect()
}

#[test]
fn bytes_path_reproduces_assembled_goldens() {
    let dir = contract_dir();
    for case in ["single_combined", "train_test", "x_y_separate"] {
        let case_dir = dir.join("corpus").join(case);
        let files = read_case_files(&case_dir);

        // fs-free infer → resolved spec (the browser-native counterpart of the
        // facade's infer_path used by the assembled_goldens.rs reference test).
        let plan = infer_named_bytes(&files, None).expect("infer_named_bytes");
        let spec = plan.resolved_spec.expect("resolved_spec");
        validate_spec(&spec).expect("valid spec");

        let sources = to_sources(&files);
        let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None)
            .unwrap_or_else(|e| panic!("assemble_in_memory `{case}`: {}", e.message));
        let produced = canonical_json(&assembled.to_summary_value()).unwrap();
        let golden =
            std::fs::read_to_string(dir.join(format!("{case}.assembled.canonical"))).unwrap();
        assert_eq!(
            produced, golden,
            "in-memory assembled drift for `{case}` (Bytes path)"
        );
    }
}

/// `;`-CSV from `(header, column-of-strings)` pairs.
fn csv(columns: &[(&str, Vec<String>)]) -> String {
    let header = columns
        .iter()
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join(";");
    let nrows = columns.first().map(|(_, v)| v.len()).unwrap_or(0);
    let mut out = String::from(&header);
    out.push('\n');
    for r in 0..nrows {
        let line = columns
            .iter()
            .map(|(_, v)| v[r].clone())
            .collect::<Vec<_>>()
            .join(";");
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn fnum(vals: &[f64]) -> Vec<String> {
    vals.iter().map(|v| format!("{v}")).collect()
}

fn strs(vals: &[&str]) -> Vec<String> {
    vals.iter().map(|s| s.to_string()).collect()
}

/// One decoded record: an absorbance signal over `axis` (nm), a `protein`
/// target, a `site` metadata field.
fn record(values: [f64; 3], axis: &[f64; 3], protein: f64, site: &str) -> Value {
    json!({
        "signals": {"absorbance": {"values": values, "axis": {"values": axis, "unit": "nm"}}},
        "targets": {"protein": protein},
        "metadata": {"site": site},
    })
}

#[test]
fn records_path_matches_equivalent_csv() {
    // The SAME logical dataset two ways: a single mixed CSV source, and a single
    // mixed source backed by decoded records. One explicit spec drives both, so
    // any difference is `records_to_frame` vs the CSV decoder.
    let axis = [1000.0_f64, 1005.0, 1010.0];
    let feats = [
        [0.10_f64, 0.20, 0.30],
        [0.40, 0.50, 0.60],
        [0.70, 0.80, 0.90],
        [1.00, 1.10, 1.20],
    ];
    let proteins = [12.5_f64, 8.3, 15.1, 9.9];
    let sites = ["a", "b", "a", "b"];

    // header_unit is pinned to nm so the CSV path (whose header would default to
    // "text") matches the records path (whose axis carries the nm unit). The unit
    // difference is the one legitimate semantic gap between the two input forms.
    let spec = DatasetSpec::from_value(&json!({
        "name": "rec-parity",
        "sources": [{
            "id": "data",
            "role": "mixed",
            "input": "scan",
            "params": {"header_unit": "nm"},
            "columns": [
                {"role": "features", "select": ["1000", "1005", "1010"]},
                {"role": "targets", "select": ["protein"]},
                {"role": "metadata", "select": ["site"]},
            ],
        }],
    }))
    .expect("spec");
    validate_spec(&spec).expect("valid spec");

    // CSV bytes form.
    let csv_text = csv(&[
        (
            "1000",
            fnum(&feats.iter().map(|r| r[0]).collect::<Vec<_>>()),
        ),
        (
            "1005",
            fnum(&feats.iter().map(|r| r[1]).collect::<Vec<_>>()),
        ),
        (
            "1010",
            fnum(&feats.iter().map(|r| r[2]).collect::<Vec<_>>()),
        ),
        ("protein", fnum(&proteins)),
        ("site", strs(&sites)),
    ]);
    let csv_sources = vec![InMemorySource {
        name: "scan".into(),
        payload: SourcePayload::Bytes(csv_text.into_bytes()),
    }];
    let from_csv = assemble_in_memory(&spec, &csv_sources, &HashMap::new(), None)
        .expect("assemble from CSV bytes");

    // Records form.
    let records: Vec<Value> = (0..4)
        .map(|i| record(feats[i], &axis, proteins[i], sites[i]))
        .collect();
    let rec_sources = vec![InMemorySource {
        name: "scan".into(),
        payload: SourcePayload::Records(records),
    }];
    let from_records = assemble_in_memory(&spec, &rec_sources, &HashMap::new(), None)
        .expect("assemble from records");

    assert_eq!(
        canonical_json(&from_records.to_summary_value()).unwrap(),
        canonical_json(&from_csv.to_summary_value()).unwrap(),
        "records path diverged from the equivalent CSV path"
    );
}

#[test]
fn records_path_full_value_carries_x_y_metadata() {
    // A direct structural check on the records path independent of the CSV path:
    // header_unit from axis.unit (nm), feature headers from axis labels, y from
    // targets, metadata column present.
    let axis = [1000.0_f64, 1005.0, 1010.0];
    let spec = DatasetSpec::from_value(&json!({
        "name": "rec",
        "sources": [{
            "id": "data",
            "role": "mixed",
            "input": "scan",
            "columns": [
                {"role": "features", "select": ["1000", "1005", "1010"]},
                {"role": "targets", "select": ["protein"]},
                {"role": "metadata", "select": ["site"]},
            ],
        }],
    }))
    .expect("spec");
    let records = vec![
        record([0.1, 0.2, 0.3], &axis, 12.5, "a"),
        record([0.4, 0.5, 0.6], &axis, 8.3, "b"),
    ];
    let sources = vec![InMemorySource {
        name: "scan".into(),
        payload: SourcePayload::Records(records),
    }];
    let assembled =
        assemble_in_memory(&spec, &sources, &HashMap::new(), None).expect("assemble records");
    let block = &assembled.blocks["train"];
    assert_eq!(block.n_samples, 2);
    assert_eq!(block.feature_headers, vec![vec!["1000", "1005", "1010"]]);
    assert_eq!(block.header_units, vec!["nm"]);
    assert_eq!(block.y_headers, vec!["protein"]);
    let y = block.y.as_ref().expect("y");
    assert_eq!((y.n_rows, y.n_cols), (2, 1));
    assert_eq!(
        block
            .metadata
            .as_ref()
            .map(|f| f.column_names())
            .unwrap_or_default(),
        vec!["site"]
    );
}
