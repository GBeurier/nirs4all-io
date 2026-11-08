//! Immutable accepted DESIGN vectors plus independent production-boundary counterexamples.
use indexmap::IndexMap;
use nirs4all_io_core::materialize::{
    numeric::{f32_integer, validate_label},
    package::{feature_content_bytes, target_content_bytes, weights_content_bytes},
    Cell, Column, DatasetPackage, FeatureMatrix, FoldProvenance, Frame, IdentityProvenance, Matrix,
    NumericStorage, PackagePartition, PayloadBlock, PrecisionOrigin, PrecisionRecord,
    RowPositionFallback, TargetTable, WeightsVector,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn vectors() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/accepted_float64_design_vectors.json"
    ))
    .unwrap()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn precision(value: &Value) -> PrecisionRecord {
    let dtype = |tag| {
        if tag == 1 {
            NumericStorage::Float32
        } else {
            NumericStorage::Float64
        }
    };
    PrecisionRecord {
        dtype: dtype(value["dtype"].as_u64().unwrap()),
        origin: match value["origin"].as_u64().unwrap() {
            0 => PrecisionOrigin::Default,
            1 => PrecisionOrigin::SourceDeclared,
            2 => PrecisionOrigin::Requested,
            _ => PrecisionOrigin::Consumer,
        },
        source_dtype: value["source_dtype"].as_str().map(str::to_owned),
        requested_dtype: value["requested_dtype"].as_u64().map(dtype),
        consumer: value["consumer"].as_str().map(str::to_owned),
    }
}
fn block(value: &Value, p: &Value) -> Matrix {
    let dtype = value["dtype"].as_u64().unwrap();
    let data = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            if dtype == 1 {
                f64::from(f32::from_bits(
                    u32::from_str_radix(v.as_str().unwrap(), 16).unwrap(),
                ))
            } else {
                f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            }
        })
        .collect();
    let mut matrix = Matrix::new(
        data,
        value["rows"].as_u64().unwrap() as usize,
        value["cols"].as_u64().unwrap() as usize,
    );
    matrix.validity = value["mask"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    matrix.precision = precision(p);
    for (v, valid) in matrix.data.iter_mut().zip(&matrix.validity) {
        if !valid {
            *v = 0.0;
        }
    }
    matrix
}
fn feature(value: &Value) -> FeatureMatrix {
    FeatureMatrix {
        feature_names: value["names"].as_array().map(|names| {
            names
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect()
        }),
        source_id: value["source_id"].as_str().unwrap().into(),
        representation_id: value["representation_id"].as_str().unwrap().into(),
        matrix: block(&value["block"], &value["precision"]),
        headers: value["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().into())
            .collect(),
        header_unit: "nm".into(),
        signal_type: None,
        processings: vec![],
        axes: value["axes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|axis| {
                let coords = if axis["coordinates"].is_null() {
                    Value::Null
                } else if axis["coordinates"]["kind"] == 1 {
                    Value::Array(
                        axis["coordinates"]["values"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| {
                                json!(f64::from_bits(
                                    u64::from_str_radix(v.as_str().unwrap(), 16).unwrap()
                                ))
                            })
                            .collect(),
                    )
                } else {
                    axis["coordinates"]["values"].clone()
                };
                json!({"name":axis["name"],"unit":axis["unit"],"coordinates":coords})
            })
            .collect(),
    }
}
fn target(value: &Value) -> TargetTable {
    let inputs = value.as_array().unwrap();
    let rows = inputs
        .first()
        .map_or(2, |v| v["block"]["rows"].as_u64().unwrap() as usize);
    let cols = inputs.len();
    let mut matrix = Matrix::new(vec![0.0; rows * cols], rows, cols);
    let mut categorical = IndexMap::new();
    let headers = inputs
        .iter()
        .map(|v| v["name"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let mut kinds = Vec::new();
    for (c, t) in inputs.iter().enumerate() {
        let kind = t["kind"].as_u64().unwrap() as u8;
        kinds.push(kind);
        if kind == 2 {
            let values = t["block"]["values"].as_array().unwrap().clone();
            categorical.insert(
                headers[c].clone(),
                json!({"categories":values,"source_dtype":t["precision"]["source_dtype"]}),
            );
            for r in 0..rows {
                matrix.data[r * cols + c] = r as f64;
                matrix.validity[r * cols + c] = t["block"]["mask"][r].as_bool().unwrap();
            }
        } else {
            let m = block(&t["block"], &t["precision"]);
            matrix.precision = m.precision;
            for r in 0..rows {
                matrix.data[r * cols + c] = m.data[r];
                matrix.validity[r * cols + c] = m.validity[r];
            }
        }
    }
    TargetTable {
        matrix,
        headers,
        categorical,
        kinds,
        representation_id: None,
    }
}
fn package(input: &Value) -> DatasetPackage {
    let rows = input["rows"].as_array().unwrap();
    let columns = [
        ("sample_id", "sample_id"),
        ("observation_id", "observation_id"),
        ("group_id", "group_id"),
        ("origin_id", "origin_id"),
        ("repetition_id", "repetition_id"),
    ]
    .into_iter()
    .filter(|(_, key)| rows.iter().all(|r| r[*key].is_string()))
    .map(|(name, key)| {
        Column::from_cells(
            name,
            rows.iter()
                .map(|r| Cell::Str(r[key].as_str().unwrap().into()))
                .collect(),
        )
    })
    .collect::<Vec<_>>();
    let metadata = Frame::from_columns(columns, "text");
    let partition = rows[0]["partition"].as_str().unwrap();
    let features = input["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(feature)
        .collect::<Vec<_>>();
    let source_ids = features
        .iter()
        .map(|f| f.source_id.clone())
        .collect::<Vec<_>>();
    let mut payloads = features
        .into_iter()
        .enumerate()
        .map(|(i, f)| (format!("{partition}/x{i}"), PayloadBlock::FeatureMatrix(f)))
        .collect::<Vec<_>>();
    if input["targets"].is_array() {
        payloads.push((
            format!("{partition}/y"),
            PayloadBlock::TargetTable(target(&input["targets"])),
        ));
    }
    payloads.push((
        format!("{partition}/metadata"),
        PayloadBlock::MetadataTable(nirs4all_io_core::materialize::MetadataTable {
            frame: metadata.clone(),
        }),
    ));
    let folds = input["folds"]
        .as_array()
        .map(|fs| {
            fs.iter()
                .map(|f| FoldProvenance {
                    train_observation_ids: f["train"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().into())
                        .collect(),
                    validation_observation_ids: f["validation"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().into())
                        .collect(),
                })
                .collect()
        })
        .unwrap_or_default();
    let audits = if input["fold_assignments"].is_array() {
        vec![
            json!({"input_sample_ids":rows.iter().map(|r|r["sample_id"].clone()).collect::<Vec<_>>(),"raw_fold_assignments":input["fold_assignments"]}),
        ]
    } else {
        vec![]
    };
    let mut pkg = DatasetPackage {
        name: "design".into(),
        task_type: "regression".into(),
        signal_type: "unknown".into(),
        n_sources: source_ids.len(),
        repetition: None,
        identity: IdentityProvenance {
            source_ids: source_ids.clone(),
            sample_id: Some("sample_id".into()),
            observation_id: Some("observation_id".into()),
            group_id: metadata.has_column("group_id").then(|| "group_id".into()),
            repetition_id: metadata
                .has_column("repetition_id")
                .then(|| "repetition_id".into()),
        },
        aggregate: None,
        folds: vec![],
        fold_provenance: folds,
        warnings: vec![],
        audits,
        partitions: IndexMap::from([(
            partition.into(),
            PackagePartition {
                n_samples: rows.len(),
                source_ids,
                payloads,
            },
        )]),
        row_position_fallback: RowPositionFallback {
            used: false,
            reason: "design".into(),
            partitions: vec![partition.into()],
            fingerprint: "unused".into(),
        },
    };
    let original = pkg.partitions.shift_remove(partition).unwrap();
    for row in rows {
        let name = row["partition"].as_str().unwrap();
        if pkg.partitions.contains_key(name) {
            continue;
        }
        let mask = rows
            .iter()
            .map(|r| r["partition"] == name)
            .collect::<Vec<_>>();
        let select = |matrix: &Matrix| {
            let mut m = matrix.clone();
            m.n_rows = mask.iter().filter(|b| **b).count();
            m.data = (0..matrix.n_rows)
                .filter(|r| mask[*r])
                .flat_map(|r| {
                    matrix.data[r * matrix.n_cols..(r + 1) * matrix.n_cols]
                        .iter()
                        .copied()
                })
                .collect();
            m.validity = (0..matrix.n_rows)
                .filter(|r| mask[*r])
                .flat_map(|r| {
                    matrix.validity[r * matrix.n_cols..(r + 1) * matrix.n_cols]
                        .iter()
                        .copied()
                })
                .collect();
            m
        };
        let payloads = original
            .payloads
            .iter()
            .map(|(id, payload)| {
                (
                    id.clone(),
                    match payload {
                        PayloadBlock::FeatureMatrix(f) => {
                            let mut f = f.clone();
                            f.matrix = select(&f.matrix);
                            PayloadBlock::FeatureMatrix(f)
                        }
                        PayloadBlock::TargetTable(t) => {
                            let mut t = t.clone();
                            t.matrix = select(&t.matrix);
                            PayloadBlock::TargetTable(t)
                        }
                        PayloadBlock::MetadataTable(m) => PayloadBlock::MetadataTable(
                            nirs4all_io_core::materialize::MetadataTable {
                                frame: m.frame.mask_rows(&mask),
                            },
                        ),
                        _ => unreachable!(),
                    },
                )
            })
            .collect();
        pkg.partitions.insert(
            name.into(),
            PackagePartition {
                n_samples: mask.iter().filter(|b| **b).count(),
                source_ids: original.source_ids.clone(),
                payloads,
            },
        );
    }
    pkg
}
#[test]
fn accepted_design_projection_preimages_match_exactly() {
    for vector in vectors()["projection_vectors"].as_array().unwrap() {
        // The assembled owner has Vec<Fold>, so absent and present-empty folds
        // are the same owner state. It has no Some(empty) input operation.
        if vector["name"] == "folds_present_empty" {
            continue;
        }
        let package = package(&vector["input"]);
        let selected = package.identity.source_ids.clone();
        let (receipt, features, targets) = package.executed_projection(&selected).unwrap();
        assert_eq!(
            hex(&features),
            vector["features"]["preimage_hex"],
            "{} features",
            vector["name"]
        );
        assert_eq!(
            hex(&targets),
            vector["targets"]["preimage_hex"],
            "{} targets",
            vector["name"]
        );
        assert_eq!(
            receipt.feature_content_fingerprint,
            vector["features"]["sha256"]
        );
        assert_eq!(
            receipt.target_content_fingerprint,
            vector["targets"]["sha256"]
        );
        receipt.validate(&package).unwrap();
    }
}
#[test]
fn accepted_design_payload_preimages_match_exactly() {
    let all = vectors();
    let base = &all["projection_vectors"][0]["input"];
    let f = feature(&base["features"][0]);
    let t = target(&base["targets"]);
    for vector in all["package_vectors"].as_array().unwrap() {
        let output = match vector["name"].as_str().unwrap() {
            "feature_payload_v4" => feature_content_bytes(&f),
            "target_payload_v4" => target_content_bytes(&t),
            "weights_payload_v4" => {
                let m = Matrix::new(vec![1.0, 2.0], 2, 1);
                weights_content_bytes(&WeightsVector {
                    values: m.data,
                    validity: m.validity,
                    precision: m.precision,
                    header: None,
                })
            }
            name => panic!("Unknown design payload {name}"),
        };
        assert_eq!(hex(&output), vector["preimage_hex"]);
        assert_eq!(format!("{:x}", Sha256::digest(&output)), vector["sha256"]);
    }
}
#[test]
fn literal_ieee_conversion_and_label_cases() {
    for vector in vectors()["conversion_vectors"].as_array().unwrap() {
        let value = f64::from_bits(
            u64::from_str_radix(vector["input_ieee_f64_hex"].as_str().unwrap(), 16).unwrap(),
        );
        let result = NumericStorage::Float32.round(value);
        if let Some(expected) = vector["result"]["value"].as_str() {
            assert_eq!(
                format!("{:08x}", (result.unwrap() as f32).to_bits()),
                expected,
                "{}",
                vector["name"]
            );
        } else {
            assert!(result.is_err(), "{}", vector["name"]);
        }
    }
    assert_eq!(f32_integer(16_777_218.0), Some(16_777_218));
    assert!(validate_label(16_777_218, NumericStorage::Float32).is_ok());
    assert!(validate_label(16_777_217, NumericStorage::Float32).is_err());
    for label in [-9_007_199_254_740_991, 9_007_199_254_740_991] {
        assert!(validate_label(label, NumericStorage::Float64).is_ok());
        assert!(validate_label(label, NumericStorage::Float32).is_err());
    }
    for label in [i64::MAX, -9_007_199_254_740_992, 9_007_199_254_740_992] {
        assert!(validate_label(label, NumericStorage::Float64).is_err());
    }
}
#[test]
fn f64_execution_ir_masks_and_consumer_policy() {
    let value = f64::from_bits(0x3ff0000000000001);
    let frame = Frame::from_columns(
        vec![Column::from_cells(
            "x",
            vec![Cell::Float(value), Cell::Float(-0.0), Cell::Na],
        )],
        "nm",
    );
    let matrix = frame.coerce_numeric(&["x".into()]).unwrap();
    assert_eq!(matrix.data[0].to_bits(), 0x3ff0000000000001);
    assert_eq!(
        frame.feature_matrix_f32(&["x".into()]).0[0].to_bits(),
        1.0f64.to_bits()
    );
    let normalized = matrix.with_storage(None, None, true).unwrap();
    assert_eq!(normalized.data[1].to_bits(), 0x8000000000000000);
    assert_eq!(normalized.data[2].to_bits(), 0);
    assert_eq!(normalized.validity, vec![true, true, false]);
    let consumer = matrix
        .with_storage(None, Some("SpectroDataset"), true)
        .unwrap();
    assert_eq!(consumer.precision.origin, PrecisionOrigin::Consumer);
    assert!(matrix
        .with_storage(Some(NumericStorage::Float64), Some("SpectroDataset"), true)
        .is_err());
    assert_eq!(
        matrix
            .with_storage(None, Some("SpectroDataset"), false)
            .unwrap()
            .precision
            .dtype,
        NumericStorage::Float64
    );
    let requested = matrix
        .with_storage(Some(NumericStorage::Float64), None, true)
        .unwrap();
    assert!(requested
        .with_storage(None, Some("SpectroDataset"), true)
        .is_err());
    let mut declared = Matrix::new(vec![value], 1, 1);
    declared.precision.source_dtype = Some("float32".into());
    assert!(declared
        .with_storage(Some(NumericStorage::Float64), None, true)
        .is_err());
    let declared = Frame::from_columns(
        vec![Column::from_cells("x", vec![Cell::Float(value)])
            .with_source_dtype(Some("float32".into()))],
        "nm",
    )
    .coerce_numeric(&["x".into()])
    .unwrap();
    assert_eq!(
        declared
            .with_storage(Some(NumericStorage::Float64), None, true)
            .unwrap()
            .data[0],
        1.0
    );
    let mut invalid = normalized;
    invalid.validity.pop();
    assert!(invalid.with_storage(None, None, true).is_err());
}

#[test]
fn exact_full_wire_and_real_precision_receipt_readers_refuse_mismatches() {
    let all = vectors();
    let package = package(&all["projection_vectors"][0]["input"]);
    let full = package.to_full_value().unwrap();
    assert_eq!(full["schema_version"], 4);
    assert_eq!(full["partition_order"], json!(["train"]));
    let block = &full["partitions"]["train"]["payloads"]["train/x0"]["value"]["matrix"];
    assert_eq!(block["dtype"], "float64");
    assert_eq!(
        block["data_bytes_hex"],
        "000000000000f03f010000000000f03f00000000000000000000000000000080"
    );
    assert!(block.get("data").is_none());
    assert_eq!(block["validity"], json!([true, true, true, true]));
    let assembled = package.to_assembled().to_full_value().unwrap();
    assert_eq!(assembled["assembled_schema_version"], 3);
    assert_eq!(assembled["partition_order"], json!(["train"]));
    assert_eq!(
        assembled["blocks"]["train"]["y"]["columns"][0]["kind"],
        "numeric"
    );
    let p = serde_json::to_value(PrecisionRecord::default()).unwrap();
    for field in [
        "dtype",
        "origin",
        "source_dtype",
        "requested_dtype",
        "consumer",
    ] {
        let mut bad = p.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<PrecisionRecord>(bad).is_err());
    }
    for (key, value) in [
        ("dtype", json!("float16")),
        ("origin", json!("requested")),
        ("requested_dtype", json!("float32")),
        ("source_dtype", json!("float32")),
        ("consumer", json!("unknown")),
    ] {
        let mut bad = p.clone();
        bad[key] = value;
        assert!(serde_json::from_value::<PrecisionRecord>(bad).is_err());
    }
    let (receipt, _, _) = package.executed_projection(&["nir".into()]).unwrap();
    let json = serde_json::to_value(&receipt).unwrap();
    for (key, value) in [
        ("profile", json!("n4a-matrix-f64-le.v1")),
        ("version", json!(0)),
        ("scope", json!("fold")),
        ("algorithm", json!("sha512")),
        ("mask_policy", json!("implicit")),
    ] {
        let mut bad = json.clone();
        bad[key] = value;
        assert!(serde_json::from_value::<
            nirs4all_io_core::materialize::ExecutedProjectionIdentityV1,
        >(bad)
        .is_err());
    }
    let mut missing = json.clone();
    missing
        .as_object_mut()
        .unwrap()
        .remove("dataset_content_identity");
    assert!(
        serde_json::from_value::<nirs4all_io_core::materialize::ExecutedProjectionIdentityV1>(
            missing
        )
        .is_err()
    );
    let mut mismatch = receipt;
    mismatch.feature_content_fingerprint = "a".repeat(64);
    assert!(mismatch.validate(&package).is_err());
    let mut corrupt = package;
    if let PayloadBlock::FeatureMatrix(f) = &mut corrupt.partitions["train"].payloads[0].1 {
        f.matrix.validity[0] = false;
    }
    assert!(corrupt
        .to_full_value()
        .unwrap_err()
        .contains("positive zero"));
}

#[test]
fn original_integer_numeric_roles_are_checked_before_any_execution_cast() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use std::collections::HashMap;
    let spec=DatasetSpec::from_value(&json!({"sources":[{"id":"table","input":"data.csv","columns":{"features":"x","targets":"y","weights":"w","metadata":"id"}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":",","na":{"policy":"ignore"}}})).unwrap();
    let run = |text: &str| {
        assemble_in_memory(
            &spec,
            &[InMemorySource {
                name: "data.csv".into(),
                payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
            }],
            &HashMap::new(),
            None,
        )
    };
    let exact=run("id,x,y,w\ns1,9007199254740992,18014398509481984,9007199254740992\ns2,18014398509481984,9007199254740992,18014398509481984\n").unwrap();
    let package = DatasetPackage::from_assembled(&exact).unwrap();
    assert_eq!(
        package.to_assembled().blocks["train"].x[0].data,
        vec![9_007_199_254_740_992.0, 18_014_398_509_481_984.0]
    );
    for (x, y, w) in [
        ("9007199254740993", "1", "1"),
        ("1", "9007199254740993", "1"),
        ("1", "1", "9007199254740993"),
    ] {
        let text = format!("id,x,y,w\ns1,{x},{y},{w}\ns2,1,,1\n");
        assert!(run(&text)
            .unwrap_err()
            .message
            .contains("exactly representable"));
    }
    let untyped = Column::from_cells("x", vec![Cell::Int(9_007_199_254_740_993), Cell::Na]);
    assert!(Frame::from_columns(vec![untyped], "text")
        .coerce_numeric(&["x".into()])
        .is_err());
    let mut classification = exact;
    classification.task_type = "classification".into();
    assert!(DatasetPackage::from_assembled(&classification)
        .unwrap_err()
        .contains("Classification"));
    let options = nirs4all_io_core::materialize::NumericStorageOptions {
        consumer: Some("SpectroDataset".into()),
        targets: Some(NumericStorage::Float64),
        ..Default::default()
    };
    assert!(
        DatasetPackage::from_assembled_with_storage(&classification, &options)
            .unwrap_err()
            .contains("float64 numerical target")
    );
}

#[test]
fn nonalphabetic_partition_order_is_explicit_in_both_full_wires_and_shared() {
    let all = vectors();
    let mut input = all["projection_vectors"][0]["input"].clone();
    input["rows"][1]["partition"] = json!("test");
    input["folds"] = Value::Null;
    let package = package(&input);
    let full = package.to_full_value().unwrap();
    let assembled = package.to_assembled().to_full_value().unwrap();
    assert_eq!(full["partition_order"], json!(["train", "test"]));
    assert_eq!(assembled["partition_order"], json!(["train", "test"]));
    let canonical = nirs4all_io_core::canonical_json(&full).unwrap();
    let sorted: Value = serde_json::from_str(&canonical).unwrap();
    assert_eq!(
        sorted["partitions"]
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap(),
        "test"
    );
    assert_eq!(sorted["partition_order"], json!(["train", "test"]));
    let receipt = package.executed_projection(&["nir".into()]).unwrap().0;
    assert_eq!(receipt.sample_ids, vec!["s1", "s2"]);
    assert_eq!(
        package
            .manifest()
            .entries
            .iter()
            .map(|e| e.partition.as_str())
            .collect::<Vec<_>>(),
        vec!["train", "train", "train", "test", "test", "test"]
    );
}

#[test]
fn public_numeric_integers_and_classification_cap_have_distinct_role_rules() {
    use nirs4all_io_core::public_dataset::{
        matrix_dataset_package, matrix_dataset_package_with_storage,
    };
    let mut input = numeric_public_record();
    let package = matrix_dataset_package(&input, "nir").unwrap();
    assert_eq!(
        package.to_assembled().blocks["train"].x[0]
            .precision
            .source_dtype
            .as_deref(),
        Some("int64")
    );
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(9007199254740993_i64);
    assert!(matrix_dataset_package(&input, "nir")
        .unwrap_err()
        .contains("exactly representable"));
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(9007199254740992_i64);
    input["dataset"]["task_type"] = json!("classification");
    assert!(matrix_dataset_package(&input, "nir")
        .unwrap_err()
        .contains("portable integer cap"));
    input["dataset"]["y"]["values"] = json!([16777217, 16777218]);
    assert!(matrix_dataset_package(&input, "nir").is_ok());
    let request = nirs4all_io_core::NumericStorageOptions {
        targets: Some(NumericStorage::Float32),
        ..Default::default()
    };
    assert!(matrix_dataset_package_with_storage(&input, "nir", &request)
        .unwrap_err()
        .contains("exact in float32"));
    input["dataset"]["task_type"] = json!("regression");
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(integer);
    input["dataset"]["y"]["values"][0] = json!(integer);
    let request = nirs4all_io_core::NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        ..Default::default()
    };
    let direct = matrix_dataset_package_with_storage(&input, "nir", &request).unwrap();
    assert_eq!(
        (direct.to_assembled().blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5a800001
    );
    assert_eq!(
        (direct.to_assembled().blocks["train"]
            .y
            .as_ref()
            .unwrap()
            .data[0] as f32)
            .to_bits(),
        0x5a800001
    );
    assert!(direct
        .executed_projection(&["nir".into()])
        .unwrap()
        .0
        .dataset_content_identity
        .is_none());
    input["dataset"]["sources"][0]["array"]["dtype"] = json!("float64");
    assert!(matrix_dataset_package_with_storage(&input, "nir", &request).is_err());
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(integer as f64);
    let host_f64 = matrix_dataset_package_with_storage(&input, "nir", &request).unwrap();
    assert_eq!(
        (host_f64.to_assembled().blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5a800000
    );
}

#[test]
fn integer_f32_selection_precedes_f64_and_avoids_double_rounding() {
    use nirs4all_io_core::materialize::{
        assemble_in_memory_with_storage, InMemorySource, SourcePayload,
    };
    use nirs4all_io_core::spec::DatasetSpec;
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    assert_eq!((integer as f32).to_bits(), 0x5a800001);
    assert_eq!(((integer as f64) as f32).to_bits(), 0x5a800000);
    let options = nirs4all_io_core::NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    let spec=DatasetSpec::from_value(&json!({"sources":[{"id":"table","input":"data.csv","columns":{"features":"x","targets":"y","weights":"w","metadata":"id"}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":",","na":{"policy":"ignore"}}})).unwrap();
    let csv = format!("id,x,y,w\ns1,{integer},{integer},{integer}\ns2,1,,1\n");
    let assembled = assemble_in_memory_with_storage(
        &spec,
        &[InMemorySource {
            name: "data.csv".into(),
            payload: SourcePayload::Bytes(csv.into_bytes()),
        }],
        &Default::default(),
        None,
        &options,
    )
    .unwrap();
    let package = DatasetPackage::from_assembled_with_storage(&assembled, &options).unwrap();
    let block = &package.to_assembled().blocks["train"];
    for value in [
        block.x[0].data[0],
        block.y.as_ref().unwrap().data[0],
        block.weights.as_ref().unwrap()[0],
    ] {
        assert_eq!((value as f32).to_bits(), 0x5a800001);
    }
    let original = Cell::Int(integer);
    assert!(original
        .to_numeric_with_storage(NumericStorage::Float32, Some("float64"))
        .is_err());
    assert_eq!(
        (Cell::Float(integer as f64)
            .to_numeric_with_storage(NumericStorage::Float32, Some("float64"))
            .unwrap() as f32)
            .to_bits(),
        0x5a800000
    );
    assert_eq!(
        (original
            .to_numeric_with_storage(NumericStorage::Float32, Some("int64"))
            .unwrap() as f32)
            .to_bits(),
        0x5a800001
    );
    assert_eq!(
        (NumericStorage::Float32
            .json_number(&json!(integer), Some("float32"))
            .unwrap() as f32)
            .to_bits(),
        0x5a800001
    );
}

#[test]
fn optional_names_axes_and_audits_survive_package_assembled_roundtrip() {
    let all = vectors();
    let mut package = package(&all["projection_vectors"][0]["input"]);
    if let PayloadBlock::FeatureMatrix(f) = &mut package.partitions["train"].payloads[0].1 {
        f.feature_names = None;
        f.axes[0]["unit"] = json!("nanometre");
        f.axes[0]["coordinates"] = json!([-0.0, f64::from_bits(0x4079000000000001)]);
    }
    package
        .audits
        .push(json!({"diagnostic":"retained owner provenance"}));
    let before = package.executed_projection(&["nir".into()]).unwrap();
    let assembled = package.to_assembled();
    assert_eq!(assembled.audits, package.audits);
    let full = assembled.to_full_value().unwrap();
    assert!(full["blocks"]["train"]["feature_names"][0].is_null());
    assert_eq!(
        full["blocks"]["train"]["feature_axes"][0][0]["coordinates"]["data_bytes_hex"],
        "00000000000000800100000000007940"
    );
    assert_eq!(full["audits"], json!(package.audits));
    let restored = DatasetPackage::from_assembled(&assembled).unwrap();
    let after = restored.executed_projection(&["nir".into()]).unwrap();
    assert_eq!(before.1, after.1);
    assert_eq!(before.2, after.2);
    let mut named = restored;
    if let PayloadBlock::FeatureMatrix(f) = &mut named.partitions["train"].payloads[0].1 {
        f.feature_names = Some(f.headers.clone());
    }
    assert_ne!(
        after.1,
        named.executed_projection(&["nir".into()]).unwrap().1
    );
}

#[test]
fn optional_whole_dataset_identity_cannot_echo_an_unsupported_audit_claim() {
    let all = vectors();
    let mut package = package(&all["projection_vectors"][0]["input"]);
    assert!(package
        .executed_projection(&["nir".into()])
        .unwrap()
        .0
        .dataset_content_identity
        .is_none());
    package.audits.push(json!({"dataset_content_identity":{"profile":"nirs4all.io.dataset-content.v1","algorithm":"sha256","fingerprint":"0".repeat(64)}}));
    assert!(package
        .executed_projection(&["nir".into()])
        .unwrap_err()
        .contains("actual owner dataset record"));
}

fn numeric_public_record() -> Value {
    json!({"schema":"nirs4all.dataset.v1","schema_version":1,"origin_ids":["s1","s2"],"fold_ids":[null,null],"dataset":{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"large","sample_ids":["s1","s2"],"source_alignment":"strict","sources":[{"name":"nir","sample_ids":["s1","s2"],"representation_id":"signal_1d","axes":["sample","wavelength"],"feature_names":null,"axis_units":{"wavelength":"nm"},"axis_coordinates":{"wavelength":[900]},"array":{"dtype":"int64","shape":[2,1],"values":[[9007199254740992_i64],[18014398509481984_i64]]}}],"task_type":"regression","y":{"dtype":"int64","shape":[2],"values":[9007199254740992_i64,18014398509481984_i64]},"groups":null,"partitions":{"dtype":"<U5","shape":[2],"values":["train","train"]}}})
}

#[test]
fn whole_dataset_link_is_recomputed_and_checked_against_actual_owner_input() {
    use nirs4all_io_core::public_dataset::{
        labelled_dataset_content_identity, matrix_dataset_package,
    };
    let input = numeric_public_record();
    let package = matrix_dataset_package(&input, "nir").unwrap();
    let receipt = package.executed_projection(&["nir".into()]).unwrap().0;
    assert_eq!(
        receipt.dataset_content_identity,
        Some(labelled_dataset_content_identity(&input).unwrap())
    );
    for (field, value) in [
        ("profile", json!("unknown")),
        ("algorithm", json!("sha512")),
        ("fingerprint", json!("0".repeat(64))),
        ("version", json!(99)),
    ] {
        let mut changed = package.clone();
        changed.audits[0]["dataset_content_identity"][field] = value;
        assert!(changed
            .executed_projection(&["nir".into()])
            .unwrap_err()
            .contains("owner-derived"));
    }
    let mut changed = package;
    changed.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][0] =
        json!(18_014_398_509_481_984_i64);
    changed.audits[0]["dataset_content_identity"] =
        labelled_dataset_content_identity(&changed.audits[0]["owner_dataset_record"]).unwrap();
    assert!(changed
        .executed_projection(&["nir".into()])
        .unwrap_err()
        .contains("does not correspond"));
}

#[test]
fn masked_public_projection_keeps_original_whole_content_evidence_separate() {
    use nirs4all_io_core::public_dataset::{
        labelled_dataset_content_identity, masked_matrix_dataset_package,
    };
    let mut input = numeric_public_record();
    input["dataset"]["y"] = json!({"dtype":"float64","shape":[2],"values":[123.0,4.0]});
    input["dataset"]["target_mask"] = json!({"dtype":"bool","shape":[2],"values":[false,true]});
    let old = labelled_dataset_content_identity(&input).unwrap();
    let (package, _) = masked_matrix_dataset_package(&input, "nir").unwrap();
    let receipt = package.executed_projection(&["nir".into()]).unwrap().0;
    assert_eq!(receipt.dataset_content_identity, Some(old));
    assert_eq!(package.audits[0]["owner_dataset_record"], input);
    let assembled = package.to_assembled();
    let y = assembled.blocks["train"].y.as_ref().unwrap();
    assert_eq!(y.data[0].to_bits(), 0);
    assert!(!y.validity[0]);
    input["dataset"]["y"]["values"][0] = Value::Null;
    let (package, _) = masked_matrix_dataset_package(&input, "nir").unwrap();
    assert!(package
        .executed_projection(&["nir".into()])
        .unwrap()
        .0
        .dataset_content_identity
        .is_none());
}

#[test]
fn original_unsigned_public_values_use_role_and_selected_storage_before_casting() {
    use nirs4all_io_core::materialize::NumericStorageOptions;
    use nirs4all_io_core::public_dataset::matrix_dataset_package_with_storage;
    let mut input = numeric_public_record();
    input["dataset"]["sources"][0]["array"]["dtype"] = json!("uint64");
    input["dataset"]["y"]["dtype"] = json!("uint64");
    for role in ["feature", "target"] {
        let mut case = input.clone();
        if role == "feature" {
            case["dataset"]["sources"][0]["array"]["values"][0][0] = json!(1_u64 << 63);
        } else {
            case["dataset"]["y"]["values"][0] = json!(1_u64 << 63);
        }
        let package =
            matrix_dataset_package_with_storage(&case, "nir", &NumericStorageOptions::default())
                .unwrap();
        let ir = package.to_assembled();
        let matrix = if role == "feature" {
            &ir.blocks["train"].x[0]
        } else {
            ir.blocks["train"].y.as_ref().unwrap()
        };
        assert_eq!(matrix.data[0].to_bits(), ((1_u64 << 63) as f64).to_bits());
        if role == "feature" {
            case["dataset"]["sources"][0]["array"]["values"][0][0] = json!((1_u64 << 63) + 1);
        } else {
            case["dataset"]["y"]["values"][0] = json!((1_u64 << 63) + 1);
        }
        assert!(matrix_dataset_package_with_storage(
            &case,
            "nir",
            &NumericStorageOptions::default()
        )
        .unwrap_err()
        .contains("exactly representable"));
    }
    let integer = (1_u64 << 63) + (1_u64 << 39) + 1;
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(integer);
    input["dataset"]["y"]["values"][0] = json!(integer);
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        ..Default::default()
    };
    let package = matrix_dataset_package_with_storage(&input, "nir", &options).unwrap();
    let ir = package.to_assembled();
    assert_eq!(
        (ir.blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5f000001
    );
    assert_eq!(
        (ir.blocks["train"].y.as_ref().unwrap().data[0] as f32).to_bits(),
        0x5f000001
    );
    assert_eq!(((integer as f64) as f32).to_bits(), 0x5f000000);
    input["dataset"]["task_type"] = json!("classification");
    assert!(matrix_dataset_package_with_storage(&input, "nir", &options).is_err());
}

#[test]
fn attaching_consumer_cannot_relabel_an_inherited_explicit_f64_target_guarantee() {
    use nirs4all_io_core::materialize::NumericStorageOptions;
    use nirs4all_io_core::public_dataset::matrix_dataset_package_with_storage;
    let input = numeric_public_record();
    let selected = matrix_dataset_package_with_storage(
        &input,
        "nir",
        &NumericStorageOptions {
            targets: Some(NumericStorage::Float64),
            ..Default::default()
        },
    )
    .unwrap();
    let consumer = NumericStorageOptions {
        consumer: Some("SpectroDataset".into()),
        ..Default::default()
    };
    assert!(
        DatasetPackage::from_assembled_with_storage(&selected.to_assembled(), &consumer).is_err()
    );
    let raw = matrix_dataset_package_with_storage(&input, "nir", &NumericStorageOptions::default())
        .unwrap();
    let projected =
        DatasetPackage::from_assembled_with_storage(&raw.to_assembled(), &consumer).unwrap();
    let ir = projected.to_assembled();
    assert_eq!(
        ir.blocks["train"].y.as_ref().unwrap().data[0],
        9_007_199_254_740_992.0
    );
    assert_eq!(
        ir.blocks["train"].y.as_ref().unwrap().precision.dtype,
        NumericStorage::Float64
    );
    assert_eq!(
        ir.blocks["train"]
            .y
            .as_ref()
            .unwrap()
            .precision
            .requested_dtype,
        None
    );
    let mut forged = selected;
    for (_, payload) in &mut forged.partitions["train"].payloads {
        if let PayloadBlock::TargetTable(table) = payload {
            table.matrix.precision.consumer = Some("SpectroDataset".into());
        }
    }
    assert!(forged.validate_storage().is_err());
}

#[test]
fn known_integer_public_ingress_cannot_hide_behind_float64_declaration() {
    use nirs4all_io_core::materialize::NumericStorageOptions;
    use nirs4all_io_core::public_dataset::matrix_dataset_package_with_storage;
    let mut base = numeric_public_record();
    base["dataset"]["sources"][0]["array"]["dtype"] = json!("float64");
    base["dataset"]["y"]["dtype"] = json!("float64");
    for target in [false, true] {
        for request in [None, Some(NumericStorage::Float64)] {
            let options = NumericStorageOptions {
                features: request,
                targets: request,
                ..Default::default()
            };
            for integer in [
                9_007_199_254_740_992_u64,
                18_014_398_509_481_984,
                1_u64 << 63,
            ] {
                let mut input = base.clone();
                let value = if target {
                    &mut input["dataset"]["y"]["values"][0]
                } else {
                    &mut input["dataset"]["sources"][0]["array"]["values"][0][0]
                };
                *value = json!(integer);
                assert!(value.is_u64());
                assert!(matrix_dataset_package_with_storage(&input, "nir", &options).is_ok());
            }
            for integer in [9_007_199_254_740_993_u64, (1_u64 << 63) + 1] {
                let mut input = base.clone();
                let value = if target {
                    &mut input["dataset"]["y"]["values"][0]
                } else {
                    &mut input["dataset"]["sources"][0]["array"]["values"][0][0]
                };
                *value = json!(integer);
                assert!(value.is_u64());
                assert!(matrix_dataset_package_with_storage(&input, "nir", &options)
                    .unwrap_err()
                    .contains("exactly representable"));
                // A caller's actual F64 value has already lost the integer bit;
                // this separate host-F64 input is admitted as its real bits.
                let rounded = json!(integer as f64);
                assert!(rounded.is_f64());
                if target {
                    input["dataset"]["y"]["values"][0] = rounded;
                } else {
                    input["dataset"]["sources"][0]["array"]["values"][0][0] = rounded;
                }
                assert!(matrix_dataset_package_with_storage(&input, "nir", &options).is_ok());
            }
        }
    }
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    base["dataset"]["sources"][0]["array"]["values"][0][0] = json!(integer);
    base["dataset"]["y"]["values"][0] = json!(integer);
    for (source, request) in [
        ("float64", NumericStorage::Float32),
        ("float32", NumericStorage::Float64),
    ] {
        base["dataset"]["sources"][0]["array"]["dtype"] = json!(source);
        base["dataset"]["y"]["dtype"] = json!(source);
        if source == "float64" {
            assert!(matrix_dataset_package_with_storage(
                &base,
                "nir",
                &NumericStorageOptions {
                    features: Some(request),
                    targets: Some(request),
                    ..Default::default()
                }
            )
            .is_err());
            continue;
        }
        let selected = matrix_dataset_package_with_storage(
            &base,
            "nir",
            &NumericStorageOptions {
                features: Some(request),
                targets: Some(request),
                ..Default::default()
            },
        )
        .unwrap()
        .to_assembled();
        assert_eq!(
            (selected.blocks["train"].x[0].data[0] as f32).to_bits(),
            0x5a800001
        );
        assert_eq!(
            (selected.blocks["train"].y.as_ref().unwrap().data[0] as f32).to_bits(),
            0x5a800001
        );
    }
}

#[test]
fn known_integer_decoded_records_keep_integer_bits_until_storage_selection() {
    use nirs4all_io_core::materialize::{
        assemble_in_memory_with_storage, InMemorySource, NumericStorageOptions, SourcePayload,
    };
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(
        &json!({"sources":[{"id":"x","role":"features","input":"decoded"}]}),
    )
    .unwrap();
    let assemble = |value: Value, request| {
        let records = vec![
            json!({"signals":{"raw":{"dtype":"float64","values":[value],"axis":{"values":[900.0],"unit":"nm"}}}}),
            json!({"signals":{"raw":{"dtype":"float64","values":[0.5],"axis":{"values":[900.0],"unit":"nm"}}}}),
        ];
        assemble_in_memory_with_storage(
            &spec,
            &[InMemorySource {
                name: "decoded".into(),
                payload: SourcePayload::Records(records),
            }],
            &Default::default(),
            None,
            &NumericStorageOptions {
                features: request,
                ..Default::default()
            },
        )
    };
    assert!(assemble(json!(9_007_199_254_740_993_i64), None)
        .unwrap_err()
        .message
        .contains("exactly representable"));
    assert!(
        assemble(json!((1_u64 << 63) + 1), Some(NumericStorage::Float64))
            .unwrap_err()
            .message
            .contains("exactly representable")
    );
    for integer in [
        9_007_199_254_740_992_u64,
        18_014_398_509_481_984,
        1_u64 << 63,
    ] {
        assert!(assemble(json!(integer), None).is_ok());
    }
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    assert!(assemble(json!(integer), Some(NumericStorage::Float32)).is_err());
    let host = assemble(json!(integer as f64), Some(NumericStorage::Float32)).unwrap();
    assert_eq!(
        (host.blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5a800000
    );
    assert!(assemble(json!(9_007_199_254_740_993_i64 as f64), None).is_ok());
}

#[test]
fn known_integer_typed_frame_admission_covers_features_targets_and_weights() {
    use nirs4all_io_core::materialize::{
        assemble_in_memory_with_storage, InMemorySource, NumericStorageOptions, SourcePayload,
    };
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"table","input":"frame","columns":{"features":"x","targets":"y","weights":"w"}}]})).unwrap();
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    for role in ["x", "y", "w"] {
        let frame = Frame::from_columns(
            ["x", "y", "w"]
                .iter()
                .map(|name| {
                    Column::from_cells(
                        name,
                        vec![Cell::Int(if *name == role { integer } else { 1 })],
                    )
                    .with_source_dtype(Some("float64".into()))
                })
                .collect(),
            "text",
        );
        let sources = [InMemorySource {
            name: "frame".into(),
            payload: SourcePayload::Frame(frame),
        }];
        for request in [None, Some(NumericStorage::Float64)] {
            let options = NumericStorageOptions {
                features: request,
                targets: request,
                weights: request,
                consumer: None,
            };
            assert!(assemble_in_memory_with_storage(
                &spec,
                &sources,
                &Default::default(),
                None,
                &options
            )
            .unwrap_err()
            .message
            .contains("exactly representable"));
        }
        assert!(assemble_in_memory_with_storage(
            &spec,
            &sources,
            &Default::default(),
            None,
            &NumericStorageOptions {
                features: Some(NumericStorage::Float32),
                targets: Some(NumericStorage::Float32),
                weights: Some(NumericStorage::Float32),
                consumer: None
            }
        )
        .is_err());
        let SourcePayload::Frame(mut integer_frame) = sources.into_iter().next().unwrap().payload
        else {
            unreachable!()
        };
        for column in &mut integer_frame.columns {
            column.source_dtype = Some("int64".into());
        }
        let sources = [InMemorySource {
            name: "frame".into(),
            payload: SourcePayload::Frame(integer_frame),
        }];
        let mut options = NumericStorageOptions::default();
        match role {
            "x" => options.features = Some(NumericStorage::Float32),
            "y" => options.targets = Some(NumericStorage::Float32),
            _ => options.weights = Some(NumericStorage::Float32),
        }
        let ir =
            assemble_in_memory_with_storage(&spec, &sources, &Default::default(), None, &options)
                .unwrap();
        let b = &ir.blocks["train"];
        let value = match role {
            "x" => b.x[0].data[0],
            "y" => b.y.as_ref().unwrap().data[0],
            _ => b.weights.as_ref().unwrap()[0],
        };
        assert_eq!((value as f32).to_bits(), 0x5a800001);
    }
}

#[test]
fn ordinary_csv_geometry_does_not_create_transport_audits() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"table","input":"data.csv","columns":{"features":"x","targets":"y"}}]})).unwrap();
    let ir = assemble_in_memory(
        &spec,
        &[InMemorySource {
            name: "data.csv".into(),
            payload: SourcePayload::Bytes(b"x;y\n1;2\n3;4\n".to_vec()),
        }],
        &Default::default(),
        None,
    )
    .unwrap();
    assert!(ir.audits.is_empty());
    let block = &ir.blocks["train"];
    assert_eq!(block.header_units, ["text"]);
    assert_eq!(block.feature_names, [Some(vec!["x".into()])]);
    assert_eq!(
        block.feature_axes,
        vec![vec![
            json!({"name":"feature","unit":null,"coordinates":null})
        ]]
    );
    let package = DatasetPackage::from_assembled(&ir).unwrap();
    // Actual pre-correction same-file owner root; only its transport audit changes.
    assert_eq!(
        package.manifest().root,
        "31515e93c7f716ed47ea89285436b2f44064246de5cba2b01ccb0ce5c6ec1ca1"
    );
    assert_eq!(package.to_full_value().unwrap()["audits"], json!([]));
    assert_eq!(
        package.to_assembled().to_full_value().unwrap()["audits"],
        json!([])
    );
}

#[test]
fn decoded_geometry_tracks_selected_columns_merges_and_exact_axis_bits() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    let upper = f64::from_bits(0x408c200000000001);
    let records = || {
        vec![
            json!({"signals":{"raw":{"dtype":"float64","values":[1.0,2.0],"axis":{"values":[-0.0,upper],"unit":"nanometre"}}},"metadata":{"id":"s1"}}),
            json!({"signals":{"raw":{"dtype":"float64","values":[3.0,4.0],"axis":{"values":[-0.0,upper],"unit":"nanometre"}}},"metadata":{"id":"s2"}}),
        ]
    };
    for merged in [false, true] {
        let (input, merge, features) = if merged {
            (
                json!(["a.rec", "b.rec"]),
                "concat_features",
                json!(["0", "b__900"]),
            )
        } else {
            (json!("a.rec"), "none", json!(["900"]))
        };
        let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"table","input":input,"merge":merge,"columns":{"features":features,"metadata":"rest"}}],"sample_index":{"by":"id","key":"id"}})).unwrap();
        let ir = assemble_in_memory(
            &spec,
            &[
                InMemorySource {
                    name: "a.rec".into(),
                    payload: SourcePayload::Records(records()),
                },
                InMemorySource {
                    name: "b.rec".into(),
                    payload: SourcePayload::Records(records()),
                },
            ],
            &Default::default(),
            None,
        )
        .unwrap();
        assert!(!ir
            .audits
            .iter()
            .any(|a| a.get("projection_raw_unit").is_some()));
        if merged {
            assert!(ir.audits.iter().any(|a| a["merge"] == "concat_features"));
        } else {
            assert!(ir.audits.is_empty());
        }
        let b = &ir.blocks["train"];
        let axis = &b.feature_axes[0][0];
        assert_eq!(axis["unit"], "nanometre");
        for (header, coordinate) in b.feature_headers[0]
            .iter()
            .zip(axis["coordinates"].as_array().unwrap())
        {
            assert_eq!(
                coordinate.as_f64().unwrap().to_bits(),
                if header.ends_with("900") {
                    upper.to_bits()
                } else {
                    (-0.0_f64).to_bits()
                }
            );
        }
        let package = DatasetPackage::from_assembled(&ir).unwrap();
        let before = package.executed_projection(&["table".into()]).unwrap();
        let rebuilt = DatasetPackage::from_assembled(&package.to_assembled()).unwrap();
        assert_eq!(
            before,
            rebuilt.executed_projection(&["table".into()]).unwrap()
        );
        assert_eq!(
            package.to_full_value().unwrap(),
            rebuilt.to_full_value().unwrap()
        );
        assert_eq!(package.audits, ir.audits);
    }
}

#[test]
fn declared_f32_nonzero_underflow_refuses_factory_and_whole_content_identity() {
    use nirs4all_io_core::public_dataset::{
        labelled_dataset_content_identity, matrix_dataset_package,
    };
    let mut input = numeric_public_record();
    input["dataset"]["sources"][0]["array"]["dtype"] = json!("float32");
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(f64::from_bits(1));
    assert!(matrix_dataset_package(&input, "nir")
        .unwrap_err()
        .contains("rounds to zero"));
    assert!(labelled_dataset_content_identity(&input)
        .unwrap_err()
        .contains("rounds to zero"));
}

#[test]
fn negative_zero_text_keeps_f64_bits_and_refuses_integer_classification_schema() {
    use nirs4all_io_core::{
        canonical_json::parse_json_value, public_dataset::matrix_dataset_package,
    };
    let negative_zero = parse_json_value("-0").unwrap();
    assert!(negative_zero.is_f64());
    let mut input = numeric_public_record();
    input["dataset"]["sources"][0]["array"]["dtype"] = json!("float64");
    input["dataset"]["sources"][0]["array"]["values"][0][0] = negative_zero.clone();
    let package = matrix_dataset_package(&input, "nir").unwrap();
    assert_eq!(
        package.to_assembled().blocks["train"].x[0].data[0].to_bits(),
        0x8000000000000000
    );
    input["dataset"]["task_type"] = json!("classification");
    input["dataset"]["y"]["values"] = json!([negative_zero, 1]);
    assert!(matrix_dataset_package(&input, "nir").is_err());
    input["dataset"]["y"]["values"][0] = json!(0);
    assert!(matrix_dataset_package(&input, "nir").is_ok());
}

#[test]
fn shared_source_declaration_vectors_drive_real_parser_factories_and_whole_identity() {
    use nirs4all_io_core::public_dataset::{
        labelled_dataset_content_identity, matrix_dataset_package_with_storage,
    };
    use nirs4all_io_core::{canonical_json::parse_json_value, NumericStorageOptions};
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/source_declaration_vectors.json")).unwrap();
    assert_eq!(
        fixture["schema"],
        "nirs4all.io.source-declaration-vectors.v1"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let cell = if let Some(bits) = case["host_bits_hex"].as_str() {
            Ok(json!(f64::from_bits(
                u64::from_str_radix(bits, 16).unwrap()
            )))
        } else {
            parse_json_value(case["raw_token"].as_str().unwrap())
        };
        let cell = match cell {
            Ok(v) => v,
            Err(_) => {
                assert_eq!(case["expected"], "refuse", "{}", case["id"]);
                assert_eq!(case["refusal_category"], "text_ingress", "{}", case["id"]);
                continue;
            }
        };
        let request: Option<NumericStorage> =
            serde_json::from_value(case["request"].clone()).unwrap();
        let target = case["role"] == "targets";
        let mut input = numeric_public_record();
        input["dataset"]["sources"][0]["array"] =
            json!({"dtype":"float64","shape":[2,1],"values":[[1.0],[2.0]]});
        if target {
            input["dataset"]["y"] =
                json!({"dtype":case["source_dtype"],"shape":[2],"values":[cell,1]});
        } else {
            input["dataset"]["sources"][0]["array"] =
                json!({"dtype":case["source_dtype"],"shape":[2,1],"values":[[cell],[1]]});
        }
        if case["group"] == "11-classification-label" {
            input["dataset"]["task_type"] = json!("classification");
        }
        let options = if target {
            NumericStorageOptions {
                targets: request,
                ..Default::default()
            }
        } else {
            NumericStorageOptions {
                features: request,
                ..Default::default()
            }
        };
        let result = matrix_dataset_package_with_storage(&input, "nir", &options);
        let whole = labelled_dataset_content_identity(&input);
        assert_eq!(
            whole.is_ok(),
            case["whole_link"] == "present",
            "whole {}: {whole:?}",
            case["id"]
        );
        if case["expected"] == "refuse" {
            assert!(result.is_err(), "{}", case["id"]);
            continue;
        }
        let package = result.unwrap_or_else(|e| panic!("{}: {e}", case["id"]));
        let ir = package.to_assembled();
        let matrix = if target {
            ir.blocks["train"].y.as_ref().unwrap()
        } else {
            &ir.blocks["train"].x[0]
        };
        assert_eq!(
            format!("{:016x}", matrix.data[0].to_bits()),
            case["expected_f64_bits_hex"],
            "{}",
            case["id"]
        );
        let receipt = package.executed_projection(&["nir".into()]).unwrap().0;
        if case["whole_link"] == "present" {
            assert_eq!(
                receipt.dataset_content_identity,
                Some(whole.unwrap()),
                "{}",
                case["id"]
            );
        } else {
            assert!(receipt.dataset_content_identity.is_none(), "{}", case["id"]);
            assert_eq!(
                package.audits[0]["dataset_content_identity_unavailable"],
                json!({"profile":"nirs4all.io.dataset-content.v1","reason":"not_representable"})
            );
        }
    }
}

#[test]
fn unavailable_whole_identity_reason_is_derived_and_cannot_echo_a_claim() {
    use nirs4all_io_core::{
        public_dataset::matrix_dataset_package_with_storage, NumericStorageOptions,
    };
    let mut input = numeric_public_record();
    input["dataset"]["sources"][0]["array"]["values"][0][0] = json!(9_007_199_254_740_993_i64);
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        ..Default::default()
    };
    let package = matrix_dataset_package_with_storage(&input, "nir", &options).unwrap();
    let receipt = package.executed_projection(&["nir".into()]).unwrap().0;
    assert!(receipt.dataset_content_identity.is_none());
    let rebuilt = DatasetPackage::from_assembled(&package.to_assembled()).unwrap();
    assert_eq!(
        package.to_full_value().unwrap(),
        rebuilt.to_full_value().unwrap()
    );
    assert_eq!(
        receipt,
        rebuilt.executed_projection(&["nir".into()]).unwrap().0
    );
    for status in [
        Value::Null,
        json!({"profile":"wrong","reason":"not_representable"}),
        json!({"profile":"nirs4all.io.dataset-content.v1","reason":"other"}),
        json!({"profile":"nirs4all.io.dataset-content.v1","reason":"not_representable","extra":true}),
    ] {
        let mut changed = package.clone();
        changed.audits[0]["dataset_content_identity_unavailable"] = status;
        assert!(changed.executed_projection(&["nir".into()]).is_err());
    }
    let mut removed = package.clone();
    removed.audits[0]
        .as_object_mut()
        .unwrap()
        .remove("dataset_content_identity_unavailable");
    assert!(removed.executed_projection(&["nir".into()]).is_err());
    let mut missing_record = package.clone();
    missing_record.audits[0]
        .as_object_mut()
        .unwrap()
        .remove("owner_dataset_record");
    assert!(missing_record.executed_projection(&["nir".into()]).is_err());
    let mut malformed = package.clone();
    malformed.audits[0]["owner_dataset_record"]["dataset"]["sources"][0]["array"]["shape"] =
        json!([999, 1]);
    assert!(malformed.executed_projection(&["nir".into()]).is_err());
    let mut fabricated = matrix_dataset_package_with_storage(
        &numeric_public_record(),
        "nir",
        &NumericStorageOptions::default(),
    )
    .unwrap();
    fabricated.audits[0]["dataset_content_identity_unavailable"] =
        package.audits[0]["dataset_content_identity_unavailable"].clone();
    assert!(fabricated.executed_projection(&["nir".into()]).is_err());
}

#[test]
fn whole_only_dataset_name_changes_whole_identity_without_changing_base_payloads() {
    use nirs4all_io_core::public_dataset::matrix_dataset_package;
    let mut input = numeric_public_record();
    let a = matrix_dataset_package(&input, "nir").unwrap();
    let before = a.executed_projection(&["nir".into()]).unwrap();
    input["dataset"]["name"] = json!("renamed whole dataset");
    let b = matrix_dataset_package(&input, "nir").unwrap();
    let after = b.executed_projection(&["nir".into()]).unwrap();
    assert_ne!(
        before.0.dataset_content_identity,
        after.0.dataset_content_identity
    );
    assert_eq!(before.1, after.1);
    assert_eq!(before.2, after.2);
    assert_eq!(a.manifest(), b.manifest());
    let whole = before.0.dataset_content_identity.unwrap();
    assert_ne!(whole["fingerprint"], before.0.feature_content_fingerprint);
    assert_ne!(whole["fingerprint"], before.0.target_content_fingerprint);
}

#[test]
fn decoded_records_cannot_silently_replace_an_axis_signed_zero() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(
        &json!({"sources":[{"id":"x","input":"records","role":"features"}]}),
    )
    .unwrap();
    let records = vec![
        json!({"signals":{"raw":{"values":[1.0],"axis":{"unit":"nm","values":[-0.0]}}}}),
        json!({"signals":{"raw":{"values":[2.0],"axis":{"unit":"nm","values":[0.0]}}}}),
    ];
    assert!(assemble_in_memory(
        &spec,
        &[InMemorySource {
            name: "records".into(),
            payload: SourcePayload::Records(records)
        }],
        &Default::default(),
        None
    )
    .unwrap_err()
    .message
    .contains("axes differ"));
}

#[test]
fn decoded_numeric_axes_check_retained_integers_but_accept_actual_f64_bits() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(
        &json!({"sources":[{"id":"x","input":"records","role":"features"}]}),
    )
    .unwrap();
    let run = |first: Value, second: Value| {
        assemble_in_memory(
            &spec,
            &[InMemorySource {
                name: "records".into(),
                payload: SourcePayload::Records(vec![
                    json!({"signals":{"raw":{"values":[1.0],"axis":{"unit":"nm","values":[first]}}}}),
                    json!({"signals":{"raw":{"values":[2.0],"axis":{"unit":"nm","values":[second]}}}}),
                ]),
            }],
            &Default::default(),
            None,
        )
    };
    let inexact = json!(9_007_199_254_740_993_u64);
    assert!(run(json!(9_007_199_254_740_992_u64), inexact.clone())
        .unwrap_err()
        .message
        .contains("exactly representable"));
    assert!(run(inexact.clone(), inexact)
        .unwrap_err()
        .message
        .contains("exactly representable"));
    for coordinate in [
        json!(1_u64 << 54),
        json!(9_007_199_254_740_993_u64 as f64),
        json!(-0.0),
        json!(0.0),
    ] {
        let assembled = run(coordinate.clone(), coordinate.clone()).unwrap();
        let package = DatasetPackage::from_assembled(&assembled).unwrap();
        let full = package.to_full_value().unwrap();
        assert!(!full.is_null());
        let axis = &assembled.blocks["train"].feature_axes[0][0]["coordinates"][0];
        assert_eq!(
            axis.as_f64().unwrap().to_bits(),
            coordinate.as_f64().unwrap().to_bits()
        );
    }
    // Integer zero and genuine float zero are the same numeric coordinate bits.
    assert!(run(json!(0), json!(0.0)).is_ok());
    assert!(run(json!(-0.0), json!(0.0)).is_err());
    // The separate public Dataset portable-coordinate contract stays narrower.
    let mut public = numeric_public_record();
    public["dataset"]["sources"][0]["axis_coordinates"]["wavelength"] = json!([1_u64 << 54]);
    assert!(nirs4all_io_core::public_dataset::normalize_dataset(&public).is_err());
}

#[test]
fn retained_numeric_strings_use_selected_storage_before_loss_and_keep_true_zero_bits() {
    for text in ["1e-400", "2e-324", "-2e-324", "9007199254740993"] {
        assert!(
            Cell::Str(text.into()).to_numeric_checked().is_err(),
            "{text}"
        );
    }
    let text = Cell::Str("18014399583223809".into());
    assert_eq!(
        (text
            .to_numeric_with_storage(NumericStorage::Float32, None)
            .unwrap() as f32)
            .to_bits(),
        0x5a800001
    );
    assert!(text
        .to_numeric_with_storage(NumericStorage::Float32, Some("float64"))
        .is_err());
    for (text, bits) in [("0e5", 0), ("0.000e-999", 0), ("-0.000e-999", 1_u64 << 63)] {
        assert_eq!(
            Cell::Str(text.into())
                .to_numeric_checked()
                .unwrap()
                .to_bits(),
            bits
        );
    }
    assert_eq!(Cell::Float(0.0).to_numeric_checked().unwrap().to_bits(), 0);
}

#[test]
fn default_repackaging_verifies_carried_f32_values_before_rounding_every_role() {
    use nirs4all_io_core::materialize::{
        assemble_in_memory_with_storage, InMemorySource, NumericStorageOptions, SourcePayload,
    };
    use nirs4all_io_core::spec::DatasetSpec;
    let spec = DatasetSpec::from_value(&json!({"task_type":"regression","params":{"delimiter":","},"sources":[{"id":"table","input":"base.csv","columns":{"features":"x","targets":"y","weights":"w"},"variations":[{"name":"child","input":"child.csv"}]}]})).unwrap();
    let sources: Vec<_> = ["base.csv", "child.csv"]
        .iter()
        .map(|name| InMemorySource {
            name: (*name).into(),
            payload: SourcePayload::Bytes(b"x,y,w\n1.0,2.0,3.0\n4.0,5.0,6.0\n".to_vec()),
        })
        .collect();
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    let assembled =
        assemble_in_memory_with_storage(&spec, &sources, &Default::default(), None, &options)
            .unwrap();
    let package = DatasetPackage::from_assembled(&assembled).unwrap();
    assert_eq!(
        package.to_full_value().unwrap(),
        DatasetPackage::from_assembled(&package.to_assembled())
            .unwrap()
            .to_full_value()
            .unwrap()
    );
    for role in ["feature", "target", "weight", "processing"] {
        let mut carried = package.to_assembled();
        let b = carried.blocks.values_mut().next().unwrap();
        match role {
            "feature" => b.x[0].data[0] = 1.0 + 2_f64.powi(-30),
            "target" => b.y.as_mut().unwrap().data[0] = 1.0 + 2_f64.powi(-30),
            "weight" => b.weights.as_mut().unwrap()[0] = 1.0 + 2_f64.powi(-30),
            _ => b.processings[0][0].1.data[0] = 1.0 + 2_f64.powi(-30),
        }
        assert!(
            DatasetPackage::from_assembled(&carried)
                .unwrap_err()
                .contains("carried storage"),
            "{role}"
        );
    }
    // A new F32 request rounds initial F64 data; its lost bits cannot be recovered.
    let initial = Matrix::new(vec![1.0 + 2_f64.powi(-30)], 1, 1);
    let selected = initial
        .with_storage(Some(NumericStorage::Float32), None, true)
        .unwrap();
    assert_eq!(selected.data, [1.0]);
    assert!(selected
        .with_storage(Some(NumericStorage::Float64), None, true)
        .is_err());
    let mut declared = Matrix::new(vec![1.0], 1, 1);
    declared.precision = PrecisionRecord::select(Some("float32"), None, None, true).unwrap();
    assert_eq!(
        declared
            .with_storage(Some(NumericStorage::Float64), None, true)
            .unwrap()
            .data,
        [1.0]
    );
}

#[test]
fn public_factory_initial_f64_to_requested_f32_rounds_once_and_repacks() {
    use nirs4all_io_core::materialize::NumericStorageOptions;
    use nirs4all_io_core::public_dataset::matrix_dataset_package_with_storage;
    let mut input = numeric_public_record();
    let array = &mut input["dataset"]["sources"][0]["array"];
    array["dtype"] = json!("float64");
    array["values"] = json!([[1.0 + 2_f64.powi(-30)], [-0.0]]);
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        ..Default::default()
    };
    let package = matrix_dataset_package_with_storage(&input, "nir", &options).unwrap();
    let assembled = package.to_assembled();
    let x = &assembled.blocks["train"].x[0];
    assert_eq!(x.data[0].to_bits(), 1.0_f64.to_bits());
    assert_eq!(x.data[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(x.precision.source_dtype.as_deref(), Some("float64"));
    assert_eq!(x.precision.requested_dtype, Some(NumericStorage::Float32));
    assert_eq!(
        package.to_full_value().unwrap(),
        DatasetPackage::from_assembled(&assembled)
            .unwrap()
            .to_full_value()
            .unwrap()
    );
}

#[test]
fn declared_f64_original_integer_snapshot_and_normalized_source_remain_distinct() {
    use nirs4all_io_core::public_dataset::{
        labelled_dataset_content_identity, matrix_dataset_package,
    };
    let mut input = numeric_public_record();
    input["dataset"]["sources"][0]["array"]["dtype"] = json!("float64");
    let identity = labelled_dataset_content_identity(&input).unwrap();
    let package = matrix_dataset_package(&input, "nir").unwrap();
    let audit = &package.audits[0];
    assert_eq!(audit["owner_dataset_record"], input);
    assert!(
        audit["owner_dataset_record"]["dataset"]["sources"][0]["array"]["values"][0][0].is_i64()
    );
    assert!(audit["raw_projection_source"]["array"]["values"][0][0].is_f64());
    assert_eq!(
        audit["raw_projection_source"]["array"]["values"][0][0]
            .as_f64()
            .unwrap()
            .to_bits(),
        9_007_199_254_740_992.0_f64.to_bits()
    );
    assert_eq!(
        package
            .executed_projection(&["nir".into()])
            .unwrap()
            .0
            .dataset_content_identity,
        Some(identity)
    );
}

#[test]
fn normalized_source_and_fold_audit_facts_cannot_replace_actual_owner_evidence() {
    use nirs4all_io_core::public_dataset::matrix_dataset_package;
    let input = numeric_public_record();
    let package = matrix_dataset_package(&input, "nir").unwrap();
    for (field, value) in [
        ("raw_projection_source", json!({})),
        ("raw_source_schema", json!({})),
        ("raw_fold_assignments", json!(["invented", "invented"])),
        ("input_sample_ids", json!(["s2", "s1"])),
        ("target_names", json!(["invented"])),
    ] {
        let mut changed = package.clone();
        changed.audits[0][field] = value;
        assert!(
            changed.executed_projection(&["nir".into()]).is_err(),
            "{field}"
        );
    }
}

// Projection-correction decoder/selection tests: prepared source, compilation UNRUN.
#[test]
fn wire_inverse_preserves_f32_f64_bits_missingness_and_precision() {
    for dtype in [NumericStorage::Float32, NumericStorage::Float64] {
        let tiny = if dtype == NumericStorage::Float32 {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        let source = Matrix::new(vec![1.5, -0.0, tiny, f64::NAN], 2, 2)
            .with_storage(Some(dtype), None, true)
            .unwrap();
        let full = source.full_value();
        let decoded = Matrix::from_full_value(&full).unwrap();
        assert_eq!(decoded.full_value(), full);
        assert_eq!(decoded.numeric_bytes(), source.numeric_bytes());
        assert_eq!(decoded.precision, source.precision);
        assert_eq!(decoded.data[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(decoded.data[2].to_bits(), tiny.to_bits());
        assert_eq!(decoded.data[3].to_bits(), 0);
        assert_eq!(decoded.validity, vec![true, true, true, false]);
    }
    for (rows, cols) in [(0, 2), (2, 0), (0, 0)] {
        let source = Matrix::new(vec![], rows, cols);
        assert_eq!(
            Matrix::from_full_value(&source.full_value())
                .unwrap()
                .full_value(),
            source.full_value()
        );
    }
}

#[test]
fn wire_inverse_reselects_only_through_existing_storage_owner() {
    let source = Matrix::new(
        vec![1.5, -0.0, f64::from(f32::from_bits(1)), f64::NAN],
        2,
        2,
    )
    .with_storage(None, None, true)
    .unwrap();
    let selected = Matrix::from_full_value(&source.full_value())
        .unwrap()
        .with_storage(None, Some("SpectroDataset"), true)
        .unwrap();
    assert_eq!(selected.precision.dtype, NumericStorage::Float32);
    assert_eq!(selected.precision.origin, PrecisionOrigin::Consumer);
    assert_eq!(selected.precision.requested_dtype, None);
    assert_eq!(
        selected.precision.consumer.as_deref(),
        Some("SpectroDataset")
    );
    assert_eq!(selected.validity, source.validity);
    assert_eq!(selected.data[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        Matrix::from_full_value(&selected.full_value())
            .unwrap()
            .full_value(),
        selected.full_value()
    );
    for (value, message) in [(1e300, "overflows float32"), (1e-320, "rounds to zero")] {
        let original = Matrix::new(vec![value], 1, 1);
        let decoded = Matrix::from_full_value(&original.full_value()).unwrap();
        assert_eq!(decoded.data[0].to_bits(), value.to_bits());
        assert!(decoded
            .with_storage(None, Some("SpectroDataset"), true)
            .unwrap_err()
            .contains(message));
    }
    let requested = source
        .with_storage(Some(NumericStorage::Float64), None, true)
        .unwrap();
    assert!(Matrix::from_full_value(&requested.full_value())
        .unwrap()
        .with_storage(None, Some("SpectroDataset"), true)
        .unwrap_err()
        .contains("explicit float64"));
}

#[test]
fn wire_inverse_keeps_each_carried_precision_role() {
    let mut source = Matrix::new(vec![1.5], 1, 1);
    source.precision = PrecisionRecord::select(Some("float32"), None, None, true).unwrap();
    let decoded = Matrix::from_full_value(&source.full_value()).unwrap();
    assert_eq!(decoded.precision, source.precision);
    let selected = decoded
        .with_storage(None, Some("SpectroDataset"), true)
        .unwrap();
    assert_eq!(selected.precision.source_dtype.as_deref(), Some("float32"));
    let target = source
        .with_storage(Some(NumericStorage::Float64), Some("SpectroDataset"), false)
        .unwrap();
    assert_eq!(
        Matrix::from_full_value(&target.full_value())
            .unwrap()
            .precision,
        target.precision
    );
}

#[test]
fn wire_inverse_rejects_corrupt_layout_before_buffer_allocation() {
    let good = Matrix::new(vec![1.5], 1, 1).full_value();
    for bad_dimension in [json!(-1), json!(1.0), json!(true), json!("1"), Value::Null] {
        for field in ["n_rows", "n_cols"] {
            let mut bad = good.clone();
            bad[field] = bad_dimension.clone();
            assert!(
                Matrix::from_full_value(&bad).is_err(),
                "{field}: {bad_dimension}"
            );
        }
    }
    for (rows, cols) in [(u64::MAX, 2_u64), (u64::MAX / 8 + 1, 1)] {
        let mut bad = good.clone();
        bad["n_rows"] = json!(rows);
        bad["n_cols"] = json!(cols);
        assert!(Matrix::from_full_value(&bad).is_err());
    }
    for text in [
        "0",
        "000000000000f8",
        "000000000000f83f00",
        "000000000000f8gg",
        "000000000000F83F",
        "000000000000f8é",
    ] {
        let mut bad = good.clone();
        bad["data_bytes_hex"] = json!(text);
        assert!(Matrix::from_full_value(&bad).is_err(), "{text}");
    }
    for bitmap in [
        json!([]),
        json!([true, false]),
        json!([1]),
        json!([null]),
        json!(true),
    ] {
        let mut bad = good.clone();
        bad["validity"] = bitmap.clone();
        assert!(Matrix::from_full_value(&bad).is_err(), "{bitmap}");
    }
    for dtype in [
        json!("float32"),
        json!("<f8"),
        json!("big-endian-float64"),
        json!(true),
    ] {
        let mut bad = good.clone();
        bad["dtype"] = dtype;
        assert!(Matrix::from_full_value(&bad).is_err());
    }
    let mut mismatched = good.clone();
    mismatched["precision"] =
        serde_json::to_value(PrecisionRecord::select(Some("float32"), None, None, true).unwrap())
            .unwrap();
    assert!(Matrix::from_full_value(&mismatched)
        .unwrap_err()
        .contains("dtype disagrees"));
    for field in ["endianness", "data", "unknown"] {
        let mut bad = good.clone();
        bad[field] = json!("big");
        assert!(Matrix::from_full_value(&bad).is_err());
    }
    let mut missing = good.clone();
    missing.as_object_mut().unwrap().remove("precision");
    assert!(Matrix::from_full_value(&missing).is_err());
    let mut bad_precision = good.clone();
    bad_precision["precision"]["origin"] = json!("consumer");
    assert!(Matrix::from_full_value(&bad_precision).is_err());
    bad_precision["precision"]["origin"] = json!(["default"]);
    assert!(Matrix::from_full_value(&bad_precision).is_err());
}

#[test]
fn wire_inverse_refuses_observed_nonfinite_and_noncanonical_missing_placeholders() {
    for dtype in [NumericStorage::Float32, NumericStorage::Float64] {
        let base = Matrix::new(vec![0.0], 1, 1)
            .with_storage(Some(dtype), None, true)
            .unwrap()
            .full_value();
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0, 1.0] {
            let mut bad = base.clone();
            bad["data_bytes_hex"] = json!(match dtype {
                NumericStorage::Float32 => hex(&(value as f32).to_le_bytes()),
                NumericStorage::Float64 => hex(&value.to_le_bytes()),
            });
            bad["validity"] = json!([false]);
            assert!(Matrix::from_full_value(&bad)
                .unwrap_err()
                .contains("placeholder"));
            if !value.is_finite() {
                bad["validity"] = json!([true]);
                assert!(Matrix::from_full_value(&bad)
                    .unwrap_err()
                    .contains("finite"));
            }
        }
    }
    // Independent LE bytes: this must decode as 1.5, never as host-endian data.
    let mut little = Matrix::new(vec![0.0], 1, 1).full_value();
    little["data_bytes_hex"] = json!("000000000000f83f");
    assert_eq!(Matrix::from_full_value(&little).unwrap().data[0], 1.5);
}
