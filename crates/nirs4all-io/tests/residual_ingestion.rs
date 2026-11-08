//! Approved residual ingestion cases through the actual public file facade.
use nirs4all_io::core::canonical_json::parse_json_value;
use nirs4all_io::core::materialize::NumericStorageOptions;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{package_with_limits_and_storage, LoadLimits};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

// Frozen Python-host expectations remain historical; these Rust requests now
// refuse at spec admission instead of reaching their historical numeric checks.
const HISTORICAL_STATISTICAL_NA_CASES: [&str; 16] = [
    "ingest.work-text.X-D-mean",
    "ingest.work-text.X-Z-mean",
    "ingest.work-text.X-H-mean",
    "ingest.work-text.X-B-mean",
    "ingest.work-text.X-U-mean",
    "ingest.work-text.X-ovf-mean",
    "ingest.work-text.X-unf-mean",
    "ingest.work-text.X-int-mean",
    "ingest.work-final.c1-mean",
    "ingest.work-final.c2-mean",
    "ingest.work-final.c3-mean",
    "ingest.work-final.c1-median",
    "ingest.work-final.c2-median",
    "ingest.work-final.c3-median",
    "ingest.work-final.comma-mean",
    "ingest.blank-decision.fill-first",
];

fn requests_statistical_na(spec: &Value) -> bool {
    let params = std::iter::once(spec.get("params")).chain(
        spec["sources"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|source| source.get("params")),
    );
    params.flatten().any(|params| {
        params
            .pointer("/na/fill/method")
            .and_then(Value::as_str)
            .is_some_and(|method| {
                matches!(
                    method.trim().to_lowercase().replace('-', "_").as_str(),
                    "mean" | "median"
                )
            })
    })
}

fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            a.as_f64().unwrap().to_bits() == b.as_f64().unwrap().to_bits()
        }
        (Value::Number(a), Value::Number(b)) if !a.is_f64() && !b.is_f64() => a == b,
        (Value::Number(_), _) | (_, Value::Number(_)) => false,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_json(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| same_json(v, other)))
        }
        _ => a == b,
    }
}

fn assertions(wire: &Value, assertions: &Value) -> Result<(), String> {
    let assertions = assertions.as_array().ok_or("assertions must be an array")?;
    if assertions.is_empty() {
        return Err("empty assertion list".into());
    }
    for assertion in assertions {
        let object = assertion.as_object().ok_or("assertion object required")?;
        if object.len() != 2 || !object.contains_key("pointer") || !object.contains_key("value") {
            return Err("assertion requires exactly pointer and value".into());
        }
        let pointer = assertion["pointer"].as_str().ok_or("pointer required")?;
        let actual = wire
            .pointer(pointer)
            .ok_or_else(|| format!("missing pointer {pointer}"))?;
        if !same_json(actual, &assertion["value"]) {
            return Err(format!(
                "{pointer}: expected {}, got {actual}",
                assertion["value"]
            ));
        }
    }
    Ok(())
}

fn refusal_stage(category: &str, error: &str) -> Option<&'static str> {
    match category {
        "csv_width" if error.contains("csv width error:") => Some("reader"),
        "join_coverage"
            if error.contains("coverage 'complete'") || error.contains("coverage 'error'") =>
        {
            Some("coverage")
        }
        "join_cardinality" if error.contains("duplicate keys") => Some("cardinality"),
        "alignment"
            if error.contains("is not row-aligned with") && error.contains("supply a join key") =>
        {
            Some("alignment")
        }
        "na_policy" if error.contains("NA values detected and na_policy is 'abort'") => {
            Some("na_policy")
        }
        "numeric_domain"
            if [
                "exactly representable",
                "outside the supported",
                "rounds to zero",
                "must be finite",
                "overflows",
                "Classification requires portable",
                "Classification label",
            ]
            .iter()
            .any(|part| error.contains(part)) =>
        {
            Some("selected_numeric")
        }
        _ => None,
    }
}

fn verify_fixture_context(raw: &[u8]) {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let parent = manifest.parent().expect("facade manifest has a parent");
    if parent.file_name() == Some(std::ffi::OsStr::new("crates")) {
        let workspace = parent.parent().expect("crates directory has a parent");
        assert!(
            workspace.join("Cargo.toml").is_file(),
            "workspace context requires its root Cargo.toml"
        );
        for relative in [
            "tests/fixtures/residual_ingestion_vectors.json",
            "crates/nirs4all-io-core/tests/fixtures/residual_ingestion_vectors.json",
            "crates/nirs4all-io/tests/fixtures/residual_ingestion_vectors.json",
        ] {
            let path = workspace.join(relative);
            let bytes = std::fs::read(&path).unwrap_or_else(|error| {
                panic!("mandatory workspace fixture {}: {error}", path.display())
            });
            assert_eq!(raw, bytes.as_slice(), "fixture drift: {}", path.display());
        }
        eprintln!("residual ingestion fixture context: workspace; root/core/facade copies checked");
    } else if manifest.join("Cargo.toml.orig").is_file() {
        eprintln!("residual ingestion fixture context: packaged facade (Cargo.toml.orig present); crate-local fixture loaded; workspace root/core drift checks unavailable in this package context");
    } else {
        panic!(
            "unrecognized facade fixture context {}; expected workspace crates directory or packaged Cargo.toml.orig; workspace checks were not skipped",
            manifest.display()
        );
    }
}

#[test]
fn residual_ingestion_shared_vectors() {
    let raw = include_bytes!("fixtures/residual_ingestion_vectors.json");
    verify_fixture_context(raw);
    let document: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(document["schema"], "nirs4all.io.owner-review-vectors.v1");
    assert_eq!(document.as_object().unwrap().len(), 2);
    let pinned: BTreeSet<&str> = HISTORICAL_STATISTICAL_NA_CASES.into_iter().collect();
    assert_eq!(pinned.len(), 16, "historical request IDs must be unique");
    let actual: BTreeSet<&str> = document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| {
            let spec = parse_json_value(case["input"]["spec_text"].as_str().unwrap()).unwrap();
            requests_statistical_na(&spec)
        })
        .map(|case| case["id"].as_str().unwrap())
        .collect();
    assert_eq!(pinned, actual, "frozen statistical NA request set drifted");
    let mut failures = Vec::new();
    let mut executed = 0;
    let mut excluded = 0;
    let mut admissions = 0;
    let mut refusals = 0;
    let mut statistical_spec_refusals = 0;
    let mut refusal_stages = std::collections::BTreeMap::new();
    let mut projection_admissions = 0;
    let mut projection_refusals = 0;
    for case in document["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let hosts = case["hosts"].as_array().unwrap();
        assert!(
            case["hosts"] == serde_json::json!(["rust", "python"])
                || case["hosts"] == serde_json::json!(["python"])
        );
        if !hosts.iter().any(|host| host == "rust") {
            excluded += 1;
            continue;
        }
        executed += 1;
        assert_eq!(case["operation"], "file_package", "{id}");
        for key in case.as_object().unwrap().keys() {
            assert!(
                [
                    "id",
                    "condition",
                    "operation",
                    "hosts",
                    "input",
                    "expected",
                    "storage"
                ]
                .contains(&key.as_str()),
                "{id}: unknown key {key}"
            );
        }
        for key in case["input"].as_object().unwrap().keys() {
            assert!(
                ["spec_text", "files"].contains(&key.as_str()),
                "{id}: unknown input key {key}"
            );
        }
        let expected = &case["expected"];
        for key in expected.as_object().unwrap().keys() {
            assert!(
                [
                    "outcome",
                    "category",
                    "refusal_stage",
                    "assertions",
                    "package_assertions",
                    "projection",
                    "data_bytes_hex",
                    "validity",
                    "precision",
                    "metadata"
                ]
                .contains(&key.as_str()),
                "{id}: unknown expected key {key}"
            );
        }
        let result = (|| -> Result<(), String> {
            match expected["outcome"].as_str() {
                Some("accept") => {
                    if expected["category"] != "accepted" || expected.get("refusal_stage").is_some()
                    {
                        return Err("invalid accepted expectation".into());
                    }
                }
                Some("refuse") => {
                    if expected.as_object().unwrap().keys().any(|key| {
                        !["outcome", "category", "refusal_stage"].contains(&key.as_str())
                    }) {
                        return Err("refused case has unexecuted assertions or projection".into());
                    }
                }
                _ => return Err("unknown expected outcome".into()),
            }
            if let Some(projection) = expected.get("projection") {
                let fields = projection.as_object().ok_or("projection object required")?;
                let required = match projection["outcome"].as_str() {
                    Some("accept") => ["outcome", "sample_ids"],
                    Some("refuse") => ["outcome", "category"],
                    _ => return Err("unknown projection outcome".into()),
                };
                if fields.len() != required.len()
                    || required.iter().any(|field| !fields.contains_key(*field))
                {
                    return Err("unknown or missing projection fields".into());
                }
            }
            if pinned.contains(id) {
                let value = parse_json_value(
                    case["input"]["spec_text"].as_str().ok_or("spec_text required")?,
                )
                .map_err(|error| error.to_string())?;
                let error = DatasetSpec::from_value(&value)
                    .expect_err("historical statistical NA request must refuse at spec admission");
                if !error.message.contains("na.fill.method") || !error.message.contains("is refused") {
                    return Err(format!("unexpected statistical NA refusal: {}", error.message));
                }
                statistical_spec_refusals += 1;
                return Ok(());
            }
            let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
            for (name, carrier) in case["input"]["files"]
                .as_object()
                .ok_or("files object required")?
            {
                let bytes = if let Some(text) = carrier.as_str() {
                    text.as_bytes().to_vec()
                } else {
                    let fields = carrier
                        .as_object()
                        .ok_or("binary carrier object required")?;
                    if fields.len() != 3
                        || carrier["format"] != "binary"
                        || !fields.contains_key("hex")
                        || !fields.contains_key("sha256")
                    {
                        return Err("unknown binary carrier".into());
                    }
                    let text = carrier["hex"].as_str().ok_or("binary hex required")?;
                    if text.len() % 2 != 0
                        || !text
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    {
                        return Err("invalid binary hex".into());
                    }
                    let bytes = (0..text.len())
                        .step_by(2)
                        .map(|offset| u8::from_str_radix(&text[offset..offset + 2], 16).unwrap())
                        .collect::<Vec<_>>();
                    if format!("{:x}", Sha256::digest(&bytes))
                        != carrier["sha256"].as_str().ok_or("binary hash required")?
                    {
                        return Err("physical file hash mismatch".into());
                    }
                    bytes
                };
                let path = directory.path().join(name);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                }
                std::fs::write(path, bytes).map_err(|error| error.to_string())?;
            }
            let spec = DatasetSpec::from_value(
                &parse_json_value(
                    case["input"]["spec_text"]
                        .as_str()
                        .ok_or("spec_text required")?,
                )
                .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.message)?;
            let storage: NumericStorageOptions = if let Some(storage) = case.get("storage") {
                serde_json::from_value(storage.clone()).map_err(|error| error.to_string())?
            } else {
                Default::default()
            };
            match package_with_limits_and_storage(
                &spec,
                directory.path(),
                LoadLimits::default(),
                &storage,
            ) {
                Err(error) => {
                    refusals += 1;
                    if expected["outcome"] != "refuse" {
                        return Err(format!("unexpected refusal: {}", error.message));
                    }
                    let observed = refusal_stage(
                        expected["category"].as_str().ok_or("category required")?,
                        &error.message,
                    )
                    .ok_or_else(|| format!("unexpected refusal category: {}", error.message))?;
                    *refusal_stages.entry(observed).or_insert(0usize) += 1;
                    if expected["refusal_stage"] != observed {
                        return Err(format!(
                            "expected stage {}, got {observed}",
                            expected["refusal_stage"]
                        ));
                    }
                }
                Ok(package) => {
                    admissions += 1;
                    if expected["outcome"] != "accept" || expected["category"] != "accepted" {
                        return Err("unexpected package admission".into());
                    }
                    if expected.get("refusal_stage").is_some() {
                        return Err("accepted case carries refusal stage".into());
                    }
                    let full = package.to_assembled().to_full_value()?;
                    let wire = package.to_full_value()?;
                    let checks = expected
                        .get("assertions")
                        .ok_or("accepted case needs meaningful assertions")?;
                    assertions(&full, checks)?;
                    if let Some(checks) = expected.get("package_assertions") {
                        assertions(&wire, checks)?;
                    }
                    for (field, pointer) in [
                        ("data_bytes_hex", "/blocks/train/x/0/data_bytes_hex"),
                        ("validity", "/blocks/train/x/0/validity"),
                        ("precision", "/blocks/train/x/0/precision"),
                        ("metadata", "/blocks/train/metadata"),
                    ] {
                        if let Some(value) = expected.get(field) {
                            assertions(
                                &full,
                                &serde_json::json!([{"pointer":pointer,"value":value}]),
                            )?;
                        }
                    }
                    if let Some(projection) = expected.get("projection") {
                        match package.executed_projection(&package.identity.source_ids) {
                            Ok((identity, _, _)) if projection["outcome"] == "accept" => {
                                projection_admissions += 1;
                                if !same_json(
                                    &serde_json::json!(identity.sample_ids),
                                    &projection["sample_ids"],
                                ) {
                                    return Err("sample IDs differ".into());
                                }
                            }
                            Err(error) if projection["outcome"] == "refuse" => {
                                projection_refusals += 1;
                                let category = if error.contains("stable sample identity") {
                                    "unstable_identity"
                                } else if error.contains("Missing projection identity") {
                                    "missing_identity"
                                } else if error.contains("Invalid primary identity") {
                                    "invalid_primary_identity"
                                } else {
                                    return Err(error);
                                };
                                if projection["category"] != category {
                                    return Err(format!("projection category differs: {error}"));
                                }
                            }
                            other => {
                                return Err(format!("unexpected projection result: {other:?}"))
                            }
                        }
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            failures.push(format!("{id}: {error}"));
        }
    }
    eprintln!("residual ingestion fixture: executed {executed} Rust rows; excluded {excluded} Python-only rows");
    eprintln!("actual factory: {admissions} admissions, {refusals} refusals; refusal stages: {refusal_stages:?}; actual projection: {projection_admissions} admissions, {projection_refusals} refusals");
    eprintln!("statistical NA: {statistical_spec_refusals} explicit Rust spec refusals; frozen Python-host expectations unchanged; no production parity claimed for these requests");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(statistical_spec_refusals, 16);
}
