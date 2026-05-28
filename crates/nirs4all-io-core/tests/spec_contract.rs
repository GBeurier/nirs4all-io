// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Partial SYNC #1: dict-input `to_spec` cases must reproduce the blessed
//! Python contract goldens byte-for-byte through the Rust IR.
//!
//! Covers the `{"spec": {...}}` cases (pure IR round-trip; no normalize aliases
//! in the corpus, so `from_value` == `from_value(normalize(...))`). Directory /
//! config-file / infer cases need the facade (EPIC 8 SYNC #1) and are gated
//! there. Re-bless via the Python harness (`NIRS4ALL_IO_ACCEPT_GOLDENS=1`).

use std::path::PathBuf;

use nirs4all_io_core::canonical_json::canonical_json;
use nirs4all_io_core::spec::{normalize_to_spec_dict, DatasetSpec};
use serde_json::Value;

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .join("tests/goldens/contract")
}

#[test]
fn dict_to_spec_cases_match_goldens() {
    let dir = contract_dir();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut checked = 0;
    for case in manifest["cases"].as_array().unwrap() {
        if case["op"] != "to_spec" {
            continue;
        }
        let Some(spec_input) = case["input"].get("spec") else {
            continue;
        };
        let name = case["name"].as_str().unwrap();
        // to_spec(dict) == DatasetSpec.from_dict(normalize_to_spec_dict(inp))
        let normalized = normalize_to_spec_dict(spec_input);
        let produced =
            canonical_json(&DatasetSpec::from_value(&normalized).unwrap().to_value()).unwrap();
        let golden =
            std::fs::read_to_string(dir.join(format!("{name}.to_spec.canonical"))).unwrap();
        assert_eq!(produced, golden, "IR drift for {name}");
        checked += 1;
    }
    assert!(checked > 0, "no dict to_spec cases found");
}
