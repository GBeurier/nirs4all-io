// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Literal owner vectors; facade/core/proposal execution requires the separate compiler grant.
use nirs4all_io::api::{to_spec, Input};
use nirs4all_io::core::infer::memory::{infer_named_bytes, NamedBytes};
use nirs4all_io::core::infer::plan::DatasetPlan;
use nirs4all_io::core::infer::propose::propose_browser_dataset;
use nirs4all_io::core::materialize::{
    assemble_in_memory, AssembledDataset, InMemorySource, NumericStorageOptions, SourcePayload,
};
use nirs4all_io::core::spec::{DatasetSpec, SpecError};
use nirs4all_io::infer::{infer_path, infer_paths};
use nirs4all_io::materialize::{assemble, package_with_limits_and_storage, LoadLimits};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
enum MaterializationPhase {
    Raw,
    Package,
}

fn vectors() -> Value {
    serde_json::from_str(include_str!("fixtures/decimal_inference_vectors.json")).unwrap()
}
fn write_files(case: &Value, root: &Path) {
    for (name, text) in case["files"].as_object().unwrap() {
        std::fs::write(root.join(name), text.as_str().unwrap()).unwrap();
    }
}
fn apply_choices(spec: DatasetSpec, case: &Value) -> DatasetSpec {
    let mut value = spec.to_value();
    if let Some(params) = case.get("post_params") {
        value["params"] = params.clone();
    }
    if let Some(choices) = case["declare_after_infer"].as_object() {
        for source in value["sources"].as_array_mut().unwrap() {
            if let Some(params) = choices.get(source["id"].as_str().unwrap()) {
                if source.get("params").is_none() {
                    source["params"] = json!({});
                }
                for (key, v) in params.as_object().unwrap() {
                    source["params"][key] = v.clone();
                }
            }
        }
    }
    DatasetSpec::from_value(&value).unwrap()
}
fn check_error(error: SpecError, case: &Value) {
    let exp = &case["expected"];
    assert!(
        error
            .message
            .to_lowercase()
            .contains(&exp["refuse"].as_str().unwrap().to_lowercase()),
        "{}: {}",
        case["id"],
        error.message
    );
    for key in ["source", "file"] {
        if let Some(part) = exp[key].as_str() {
            assert!(
                error.message.contains(part),
                "{}: {}",
                case["id"],
                error.message
            );
        }
    }
}
fn check_values(assembled: &AssembledDataset, case: &Value, phase: MaterializationPhase) {
    for (partition, expected) in [("train", &case["expected"]), ("test", &case["test_block"])] {
        if expected.is_null() {
            continue;
        }
        let b = &assembled.blocks[partition];
        let rows = expected["x"].as_array().unwrap();
        let width = expected["headers"].as_array().unwrap().len();
        assert_eq!(
            (b.x[0].n_rows, b.x[0].n_cols),
            (rows.len(), width),
            "{}",
            case["id"]
        );
        for (index, (actual, value)) in b.x[0]
            .data
            .iter()
            .zip(rows.iter().flat_map(|row| row.as_array().unwrap()))
            .enumerate()
        {
            if value.is_null() {
                assert!(!b.x[0].validity[index]);
                match phase {
                    MaterializationPhase::Raw => assert!(actual.is_nan()),
                    MaterializationPhase::Package => assert_eq!(actual.to_bits(), 0),
                }
            } else {
                assert!(b.x[0].validity[index]);
                assert_eq!(
                    actual.to_bits(),
                    value.as_f64().unwrap().to_bits(),
                    "{}",
                    case["id"]
                );
            }
        }
        let mask: Vec<bool> = expected["x_mask"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_bool().unwrap()))
            .collect();
        assert_eq!(
            rows.iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|value| !value.is_null()))
                .collect::<Vec<_>>(),
            mask
        );
        assert_eq!(b.x[0].validity, mask);
        assert_eq!(json!(b.feature_headers[0]), expected["headers"]);
        if let Some(y) = expected["y"].as_array() {
            let values = y
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .collect::<Vec<_>>();
            let matrix = b.y.as_ref().unwrap();
            let actual = &matrix.data;
            assert_eq!(actual.len(), values.len());
            assert_eq!(matrix.validity.len(), actual.len());
            for (index, (actual, value)) in actual.iter().zip(values).enumerate() {
                if value.is_null() {
                    assert!(!matrix.validity[index]);
                    match phase {
                        MaterializationPhase::Raw => assert!(actual.is_nan()),
                        MaterializationPhase::Package => assert_eq!(actual.to_bits(), 0),
                    }
                } else {
                    assert!(matrix.validity[index]);
                    assert_eq!(
                        actual.to_bits(),
                        value.as_f64().unwrap().to_bits(),
                        "{}",
                        case["id"]
                    );
                }
            }
            let mask: Vec<bool> = expected["y_mask"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_bool().unwrap()))
                .collect();
            assert_eq!(
                y.iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|value| !value.is_null()))
                    .collect::<Vec<_>>(),
                mask
            );
            assert_eq!(matrix.validity, mask);
            assert_eq!(json!(b.y_headers), expected["y_headers"]);
        }
    }
}
fn check_plan(plan: &DatasetPlan, case: &Value) {
    if let Some(file) = case["unresolved"].as_str() {
        assert!(plan.params[file]["delimiter"]["value"].is_null());
        assert_eq!(plan.params[file]["delimiter"]["score"], json!(0.0));
        assert!(plan
            .warnings
            .iter()
            .any(|w| w.contains(file) && w.contains("candidates")));
    }
}
#[test]
fn facade_vectors_preserve_values_masks_and_explicit_choices() {
    for case in vectors()["rows"].as_array().unwrap() {
        for form in case["forms"].as_array().unwrap() {
            for method in case["methods"].as_array().unwrap() {
                let tmp = tempfile::tempdir().unwrap();
                write_files(case, tmp.path());
                let selected: Result<(DatasetSpec, PathBuf), SpecError> = (|| {
                    let paths = case["files"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(|n| tmp.path().join(n).to_string_lossy().into_owned())
                        .collect::<Vec<_>>();
                    if method == "infer" {
                        let plan = if form == "list" {
                            infer_paths(&paths, &Default::default())?
                        } else {
                            infer_path(
                                &if form == "single" {
                                    paths[0].clone()
                                } else {
                                    tmp.path().to_string_lossy().into_owned()
                                },
                                &Default::default(),
                            )?
                        };
                        check_plan(&plan, case);
                        Ok((
                            plan.resolved_spec
                                .clone()
                                .ok_or_else(|| SpecError::new("missing inferred spec"))?,
                            tmp.path().to_path_buf(),
                        ))
                    } else if form == "explicit" {
                        Ok((
                            to_spec(&Input::Spec(case["spec"].clone()), None, None)?.0,
                            tmp.path().to_path_buf(),
                        ))
                    } else {
                        to_spec(
                            &if form == "list" {
                                Input::Paths(paths)
                            } else if form == "single" {
                                Input::Path(paths[0].clone())
                            } else {
                                Input::Path(tmp.path().to_string_lossy().into_owned())
                            },
                            None,
                            None,
                        )
                    }
                })();
                let (spec, base) = match selected {
                    Ok(v) => v,
                    Err(e) => {
                        assert!(case["expected"].get("refuse").is_some());
                        check_error(e, case);
                        continue;
                    }
                };
                let spec = apply_choices(spec, case);
                for package in [false, true] {
                    let result = if package {
                        package_with_limits_and_storage(
                            &spec,
                            &base,
                            LoadLimits::default(),
                            &NumericStorageOptions::default(),
                        )
                        .map(|p| p.to_assembled().clone())
                    } else {
                        assemble(&spec, &base)
                    };
                    if case["expected"].get("refuse").is_some() {
                        check_error(result.unwrap_err(), case);
                    } else {
                        check_values(
                            &result.unwrap(),
                            case,
                            if package {
                                MaterializationPhase::Package
                            } else {
                                MaterializationPhase::Raw
                            },
                        );
                    }
                }
                if let Some(decimals) = case["decimals"].as_object() {
                    for (id, v) in decimals {
                        let source = spec.sources.iter().find(|s| s.id == *id).unwrap();
                        assert_eq!(json!(source.params.decimal_separator), *v);
                    }
                }
            }
        }
    }
}
#[test]
fn named_bytes_inherit_the_same_inference_and_target_guard() {
    for case in vectors()["rows"].as_array().unwrap() {
        if !case["methods"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m == "infer")
        {
            continue;
        }
        let files = case["files"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, text)| NamedBytes {
                name: name.clone(),
                bytes: text.as_str().unwrap().as_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        let plan = match infer_named_bytes(&files, &Default::default()) {
            Ok(p) => p,
            Err(e) => {
                assert!(case["expected"].get("refuse").is_some());
                check_error(e, case);
                continue;
            }
        };
        check_plan(&plan, case);
        let spec = apply_choices(plan.resolved_spec.clone().unwrap(), case);
        let sources = files
            .iter()
            .map(|file| InMemorySource {
                name: file.name.clone(),
                payload: SourcePayload::Bytes(file.bytes.clone()),
            })
            .collect::<Vec<_>>();
        let result = assemble_in_memory(&spec, &sources, &HashMap::new(), None);
        if case["expected"].get("refuse").is_some() {
            check_error(result.unwrap_err(), case);
        } else {
            check_values(&result.unwrap(), case, MaterializationPhase::Raw);
        }
    }
}
#[test]
fn provisional_sources_carry_decimal_only_when_resolved() {
    let files = [
        NamedBytes {
            name: "arbitrary-a.csv".into(),
            bytes: b"x;y\n1,5;2,5\n3,5;4,5\n".to_vec(),
        },
        NamedBytes {
            name: "arbitrary-b.csv".into(),
            bytes: b"Protein, %\n1,25e1\n1,31e1\n".to_vec(),
        },
    ];
    let result = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let a = result
        .spec
        .sources
        .iter()
        .find(|s| s.input == json!("arbitrary-a.csv"))
        .unwrap();
    let b = result
        .spec
        .sources
        .iter()
        .find(|s| s.input == json!("arbitrary-b.csv"))
        .unwrap();
    assert_eq!(a.params.decimal_separator.as_deref(), Some(","));
    assert!(b.params.delimiter.is_none());
    assert!(b.params.decimal_separator.is_none());
}
