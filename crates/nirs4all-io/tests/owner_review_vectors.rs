//! Execute the shared owner-review rows against the actual file/package APIs.
use nirs4all_io::core::canonical_json::parse_json_value;
use nirs4all_io::core::materialize::{DatasetPackage, NumericStorageOptions};
use nirs4all_io::core::public_dataset::dataset_content_bytes;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{
    assemble_with_storage, package_with_limits_and_storage, LoadLimits,
};
use serde_json::Value;

// JSON's numeric tokens carry semantic type here: integer, float and boolean
// must never compare equal merely because their mathematical values coincide.
fn strict_value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            a.as_f64().unwrap().to_bits() == b.as_f64().unwrap().to_bits()
        }
        (Value::Number(a), Value::Number(b)) if !a.is_f64() && !b.is_f64() => a == b,
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| strict_value_eq(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| strict_value_eq(v, w)))
        }
        (Value::Number(_), _) | (_, Value::Number(_)) => false,
        _ => a == b,
    }
}

fn check_assertions(value: &Value, assertions: &Value) -> Result<(), String> {
    for assertion in assertions.as_array().ok_or("assertions must be an array")? {
        let fields = assertion.as_object().ok_or("assertion must be an object")?;
        if fields.len() != 2 || !fields.contains_key("pointer") || !fields.contains_key("value") {
            return Err("assertion must contain exactly pointer and value".into());
        }
        let pointer = assertion["pointer"]
            .as_str()
            .ok_or("assertion pointer required")?;
        let expected = assertion.get("value").ok_or("assertion value required")?;
        let actual = value
            .pointer(pointer)
            .ok_or_else(|| format!("missing {pointer}"))?;
        if !strict_value_eq(actual, expected) {
            return Err(format!("{pointer}: expected {expected}, got {actual}"));
        }
    }
    Ok(())
}

fn tree_member<'a>(object: &'a Value, name: &str) -> Result<&'a Value, String> {
    object[1]
        .as_array()
        .ok_or("tagged object required")?
        .iter()
        .find(|pair| pair[0] == name)
        .map(|pair| &pair[1])
        .ok_or_else(|| format!("tagged member {name} required"))
}

fn refusal_matches(category: &str, error: &str) -> bool {
    match category {
        "no_numeric_features" => error.contains("no feature source"),
        "numeric_domain" => [
            "exactly representable",
            "rounds to zero",
            "outside the supported",
            "must be finite",
        ]
        .iter()
        .any(|s| error.contains(s)),
        "invalid_selector" => error.contains("selectors") || error.contains("unassigned"),
        "invalid_threshold" => error.contains("outlier_threshold"),
        "precision_reselection" => error.contains("re-select float64"),
        "source_declaration" => {
            error.contains("exactly representable")
                || error.contains("Array scalar incompatible with dtype")
                || error.contains("Unsupported numeric source dtype")
        }
        "unsupported_integer_source" => error.contains("outside i64 range"),
        "invalid_observed_float32" => error.contains("overflows float32"),
        _ => false,
    }
}

#[test]
fn metadata_comparison_is_type_and_float_bit_sensitive() {
    assert!(!strict_value_eq(
        &serde_json::json!(42),
        &serde_json::json!(42.0)
    ));
    assert!(!strict_value_eq(
        &serde_json::json!(true),
        &serde_json::json!(1)
    ));
    assert!(!strict_value_eq(
        &serde_json::json!(-0.0),
        &serde_json::json!(0.0)
    ));
    assert!(strict_value_eq(
        &serde_json::json!([42, 0.0, true]),
        &serde_json::json!([42, 0.0, true])
    ));
}

fn write_fixture_file(path: &std::path::Path, value: &Value) -> Result<(), String> {
    if let Some(text) = value.as_str() {
        return std::fs::write(path, text).map_err(|e| e.to_string());
    }
    use arrow_array::{ArrayRef, Float32Array, Int64Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    let columns = value["columns"]
        .as_array()
        .ok_or("typed columns required")?;
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for column in columns {
        let values = column["values"].as_array().ok_or("typed values required")?;
        let name = column["name"].as_str().ok_or("typed name required")?;
        match column["dtype"].as_str() {
            Some("int64") => {
                let data = values
                    .iter()
                    .map(|v| {
                        if v.is_null() {
                            Ok(None)
                        } else {
                            v.as_str()
                                .ok_or("integer string required")?
                                .parse::<i64>()
                                .map(Some)
                                .map_err(|_| "invalid integer")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                fields.push(Field::new(name, DataType::Int64, true));
                arrays.push(Arc::new(Int64Array::from(data)));
            }
            Some("float32") => {
                let data = values
                    .iter()
                    .map(|v| {
                        if v.is_null() {
                            Ok(None)
                        } else {
                            v.as_f64()
                                .map(|n| Some(n as f32))
                                .ok_or("numeric float32 required")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                fields.push(Field::new(name, DataType::Float32, true));
                arrays.push(Arc::new(Float32Array::from(data)));
            }
            Some("uint64") => {
                let data = values
                    .iter()
                    .map(|v| {
                        if v.is_null() {
                            Ok(None)
                        } else {
                            v.as_str()
                                .ok_or("integer string required")?
                                .parse::<u64>()
                                .map(Some)
                                .map_err(|_| "invalid unsigned integer")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                fields.push(Field::new(name, DataType::UInt64, true));
                arrays.push(Arc::new(UInt64Array::from(data)));
            }
            _ => return Err("unsupported typed fixture column".into()),
        }
    }
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(|e| e.to_string())?;
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = parquet::arrow::arrow_writer::ArrowWriter::try_new(file, batch.schema(), None)
        .map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.close().map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
fn executable_owner_review_fixture() {
    assert_eq!(
        include_bytes!("../../../tests/fixtures/owner_review_vectors.json").as_slice(),
        include_bytes!("../../nirs4all-io-core/tests/fixtures/owner_review_vectors.json")
            .as_slice()
    );
    execute_fixture(include_str!(
        "../../../tests/fixtures/owner_review_vectors.json"
    ));
}

#[test]
fn executable_owner_final_work_fixture() {
    assert_eq!(
        include_bytes!("../../../tests/fixtures/owner_final_work_vectors.json").as_slice(),
        include_bytes!("../../nirs4all-io-core/tests/fixtures/owner_final_work_vectors.json")
            .as_slice()
    );
    execute_fixture(include_str!(
        "../../../tests/fixtures/owner_final_work_vectors.json"
    ));
}

fn execute_fixture(text: &str) {
    let fixture: Value = serde_json::from_str(text).unwrap();
    assert_eq!(fixture["schema"], "nirs4all.io.owner-review-vectors.v1");
    let mut failures = Vec::new();
    let mut executed = 0;
    let mut excluded = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if let Some(hosts) = case.get("hosts") {
            let hosts = hosts.as_array().expect("hosts must be an array");
            assert!(!hosts.is_empty(), "{id}: hosts must not be empty");
            assert!(
                hosts.iter().all(|host| host == "rust" || host == "python"),
                "{id}: unknown host"
            );
        }
        if case["hosts"]
            .as_array()
            .is_some_and(|h| !h.iter().any(|v| v == "rust"))
        {
            excluded += 1;
            continue;
        }
        executed += 1;
        let input = &case["input"];
        let mut full = None;
        let mut package_full = None;
        let mut projection = None;
        let mut whole_link = None;
        let mut whole_link_reason_present = None;
        let mut first_checked = false;
        for key in case["expected"].as_object().unwrap().keys() {
            assert!(
                [
                    "outcome",
                    "category",
                    "data_bytes_hex",
                    "validity",
                    "precision",
                    "metadata",
                    "sample_ids",
                    "projection_error_category",
                    "whole_link",
                    "whole_link_unavailable_reason",
                    "assertions",
                    "package_assertions",
                    "first_assertions",
                    "projection",
                    "aggregate_threshold_bits",
                    "u07_first_numeric_bits"
                ]
                .contains(&key.as_str()),
                "{id}: unsupported expectation {key}"
            );
        }
        if let Some(p) = case["expected"].get("projection") {
            let p = p
                .as_object()
                .expect("projection expectation must be an object");
            assert!(
                p.len() == 2
                    && p.contains_key("outcome")
                    && ((p["outcome"] == "accept" && p.contains_key("sample_ids"))
                        || (p["outcome"] == "refuse" && p.contains_key("category"))),
                "{id}: unsupported projection expectation"
            );
        }
        let result: Result<(), String> = (|| match case["operation"].as_str().unwrap() {
            "file_package" | "assembly_reselect" => {
                let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
                for (name, text) in input["files"].as_object().ok_or("files required")? {
                    write_fixture_file(&dir.path().join(name), text)?;
                }
                let raw =
                    parse_json_value(input["spec_text"].as_str().ok_or("spec text required")?)
                        .map_err(|e| e.to_string())?;
                let spec = DatasetSpec::from_value(&raw).map_err(|e| e.message)?;
                if case["operation"] == "assembly_reselect" {
                    let stages = input["stages"].as_array().ok_or("stages required")?;
                    let first: NumericStorageOptions =
                        serde_json::from_value(stages[0]["storage"].clone())
                            .map_err(|e| e.to_string())?;
                    let second: NumericStorageOptions =
                        serde_json::from_value(stages[1]["storage"].clone())
                            .map_err(|e| e.to_string())?;
                    let assembled =
                        assemble_with_storage(&spec, dir.path(), &first).map_err(|e| e.message)?;
                    if let Some(assertions) = case["expected"].get("first_assertions") {
                        check_assertions(&assembled.to_full_value()?, assertions)?;
                        first_checked = true;
                    }
                    let package = DatasetPackage::from_assembled_with_storage(&assembled, &second)?;
                    package_full = Some(package.to_full_value()?);
                    full = Some(package.to_assembled().to_full_value()?);
                } else {
                    let options: NumericStorageOptions = serde_json::from_value(
                        case.get("storage")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({})),
                    )
                    .map_err(|e| e.to_string())?;
                    let package = package_with_limits_and_storage(
                        &spec,
                        dir.path(),
                        LoadLimits::default(),
                        &options,
                    )
                    .map_err(|e| e.message)?;
                    if case["expected"].get("projection").is_some()
                        || case["expected"].get("sample_ids").is_some()
                        || case["expected"].get("projection_error_category").is_some()
                    {
                        projection = Some(
                            package
                                .executed_projection(&package.identity.source_ids)
                                .map(|v| v.0.sample_ids),
                        );
                    }
                    full = Some(package.to_assembled().to_full_value()?);
                    package_full = Some(package.to_full_value()?);
                }
                Ok(())
            }
            "whole" => {
                let value = parse_json_value(input["text"].as_str().ok_or("text required")?)
                    .map_err(|e| e.to_string())?;
                let bytes = dataset_content_bytes(&value)?;
                if let Some(expected) = case["expected"].get("u07_first_numeric_bits") {
                    let tree: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                    let dataset = tree_member(&tree, "dataset")?;
                    let sources = tree_member(dataset, "sources")?;
                    let array = tree_member(&sources[1][3], "array")?;
                    let values = tree_member(array, "values")?;
                    let first_row = &values[1][0][1];
                    if first_row[0] != serde_json::json!(["number", expected])
                        || first_row[1] != serde_json::json!(["string", "A"])
                    {
                        return Err("U07 numeric or string tagged content differs".into());
                    }
                }
                Ok(())
            }
            "public_masked" | "public_package" => {
                let value = parse_json_value(input["text"].as_str().ok_or("text required")?)
                    .map_err(|e| e.to_string())?;
                let options: NumericStorageOptions = serde_json::from_value(
                    case.get("storage")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({})),
                )
                .map_err(|e| e.to_string())?;
                let source = input["source_id"].as_str().ok_or("source id required")?;
                let package = if case["operation"] == "public_masked" {
                    nirs4all_io::core::public_dataset::masked_matrix_dataset_package_with_storage(
                        &value, source, &options,
                    )?
                    .0
                } else {
                    nirs4all_io::core::public_dataset::matrix_dataset_package_with_storage(
                        &value, source, &options,
                    )?
                };
                if case["expected"].get("projection").is_some() {
                    projection = Some(
                        package
                            .executed_projection(&[source.to_owned()])
                            .map(|value| value.0.sample_ids),
                    );
                }
                if case["expected"].get("whole_link").is_some() {
                    whole_link = Some(
                        package
                            .executed_projection(&[source.to_owned()])
                            .map(|p| p.0.dataset_content_identity),
                    );
                }
                if case["expected"]
                    .get("whole_link_unavailable_reason")
                    .is_some()
                {
                    whole_link_reason_present =
                        Some(package.audits.iter().any(|audit| {
                            audit.get("dataset_content_identity_unavailable").is_some()
                        }));
                }
                full = Some(package.to_assembled().to_full_value()?);
                package_full = Some(package.to_full_value()?);
                Ok(())
            }
            "spec_validate" => {
                let value = parse_json_value(input["text"].as_str().ok_or("text required")?)
                    .map_err(|e| e.to_string())?;
                let spec = DatasetSpec::from_value(&value).map_err(|e| e.message)?;
                if let Some(bits) = case["expected"].get("aggregate_threshold_bits") {
                    let actual = spec
                        .aggregate
                        .as_ref()
                        .ok_or("aggregate required")?
                        .outlier_threshold
                        .to_bits();
                    if format!("{actual:016x}") != bits.as_str().ok_or("threshold bits required")? {
                        return Err("aggregate default threshold differs".into());
                    }
                }
                Ok(())
            }
            other => Err(format!("unsupported fixture operation {other}")),
        })();
        let accept = case["expected"]["outcome"] == "accept";
        if case["expected"].get("first_assertions").is_some() && !first_checked {
            failures.push(format!(
                "{id}: first assembly failed or its assertions did not match: {result:?}"
            ));
        }
        if case["expected"].get("projection").is_some() && projection.is_none() {
            failures.push(format!("{id}: projection expectation was not executed"));
        }
        if case["expected"].get("assertions").is_some() && full.is_none() {
            failures.push(format!("{id}: full-value assertions were not executed"));
        }
        if let Some(assertions) = case["expected"].get("package_assertions") {
            match package_full.as_ref() {
                Some(value) => {
                    if let Err(error) = check_assertions(value, assertions) {
                        failures.push(format!("{id}: package {error}"));
                    }
                }
                None => failures.push(format!("{id}: package assertions were not executed")),
            }
        }
        if result.is_ok() != accept {
            failures.push(format!(
                "{id}: expected {}, got {result:?}",
                case["expected"]["outcome"]
            ));
        }
        if let Err(error) = &result {
            let category = case["expected"]["category"].as_str().unwrap();
            if !accept && !refusal_matches(category, error) {
                failures.push(format!("{id}: unexpected refusal for {category}: {error}"));
            }
        }
        if case["expected"].get("whole_link").is_some() && !matches!(whole_link, Some(Ok(None))) {
            failures.push(format!(
                "{id}: whole-link projection did not return an absent identity: {whole_link:?}"
            ));
        }
        if case["expected"]
            .get("whole_link_unavailable_reason")
            .is_some()
            && whole_link_reason_present != Some(false)
        {
            failures.push(format!("{id}: unexpected whole-link unavailability reason"));
        }
        if let Some(result) = projection {
            let expectation = case["expected"].get("projection");
            match result {
                Ok(ids)
                    if expectation.is_some_and(|e| {
                        e["outcome"] == "accept" && serde_json::json!(ids) == e["sample_ids"]
                    }) => {}
                Err(error)
                    if expectation.is_some_and(|e| {
                        e["outcome"] == "refuse" && e["category"] == "missing_identity"
                    }) && error.contains("Missing projection identity") => {}
                Err(error)
                    if expectation.is_some_and(|e| {
                        e["outcome"] == "refuse" && e["category"] == "invalid_primary_identity"
                    }) && error == "Invalid primary identity \"1.8014398509481984e+16\"" => {}
                Err(error)
                    if expectation.is_some_and(|e| {
                        e["outcome"] == "refuse" && e["category"] == "unstable_identity"
                    }) && error == "Projection requires stable sample identity" => {}
                Ok(ids)
                    if case["expected"].get("sample_ids").is_some()
                        && serde_json::json!(ids) == case["expected"]["sample_ids"] => {}
                Err(error)
                    if case["expected"]["projection_error_category"] == "missing_identity"
                        && error.contains("Missing projection identity") => {}
                other => failures.push(format!("{id}: projection identity mismatch: {other:?}")),
            }
        }
        if let Some(full) = full {
            if let Some(assertions) = case["expected"].get("assertions") {
                if let Err(error) = check_assertions(&full, assertions) {
                    failures.push(format!("{id}: {error}"));
                }
            }
            if let Some(expected) = case["expected"].get("data_bytes_hex") {
                let block = &full["blocks"]["train"];
                for field in ["data_bytes_hex", "validity", "precision"] {
                    if block["x"][0][field] != case["expected"][field] {
                        failures.push(format!("{id}: feature {field} differs"));
                    }
                }
                if !strict_value_eq(&block["metadata"], &case["expected"]["metadata"]) {
                    failures.push(format!("{id}: metadata differs"));
                }
                assert!(expected.is_string());
            }
        }
    }
    eprintln!(
        "owner fixture: executed {executed} Rust rows; excluded {excluded} rows for other hosts"
    );
    assert!(executed > 0, "fixture executed no Rust rows");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
