// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! SYNC #1 (infer): the facade `infer()` reproduces the blessed Python contract
//! goldens byte-for-byte. Mirrors `tests/test_contract_goldens.py`: directory
//! inputs, output path-normalized to `<CORPUS>` (prefix-only — never rewrites
//! selector regex backslashes), canonicalized, diffed against the blessed files.

use std::path::PathBuf;

use nirs4all_io::canonical_json;
use nirs4all_io::infer::infer_path;
use serde_json::Value;

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .join("tests/goldens/contract")
}

fn normalize_paths(value: &Value, base_abs: &str) -> Value {
    match value {
        Value::String(s) => {
            let replaced = s
                .replace(&format!("{base_abs}/"), "<CORPUS>/")
                .replace(base_abs, "<CORPUS>");
            Value::String(replaced)
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| normalize_paths(v, base_abs)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), normalize_paths(v, base_abs)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[test]
fn infer_cases_match_goldens() {
    let dir = contract_dir();
    let base_abs = std::fs::canonicalize(&dir)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut checked = 0;
    for case in manifest["cases"].as_array().unwrap() {
        if case["op"] != "infer" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let Some(dir_rel) = case["input"].get("dir").and_then(|v| v.as_str()) else {
            continue;
        };
        let case_dir = dir.join(dir_rel);
        let plan = infer_path(&case_dir.to_string_lossy(), None).expect("infer");
        let produced = canonical_json(&normalize_paths(&plan.to_value(), &base_abs)).unwrap();
        let golden = std::fs::read_to_string(dir.join(format!("{name}.infer.canonical"))).unwrap();
        assert_eq!(produced, golden, "infer drift for {name}");
        checked += 1;
    }
    assert_eq!(checked, 3, "expected 3 infer dir cases");
}
