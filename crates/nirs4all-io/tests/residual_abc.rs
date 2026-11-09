//! Residual B/C cases from the final owner review, through actual file loading.
use nirs4all_io::core::infer::table::ColDtype;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{
    package_with_limits_and_storage, DatasetPackage, LoadLimits, NumericStorageOptions,
};
use serde_json::{json, Value};

const TEXT: &str = "m,x,y\n3,3.5,3.0\nabc,4.5,4.0\n";
const U64: &str = "m,x,y\n18446744073709551615,1.5,1.0\n4,2.5,2.0\n";
const BOOL: &str = "m,x,y\nTRUE,1.5,1.0\nFALSE,2.5,2.0\n";
const INT: &str = "m,x,y\n3,3.5,3.0\n4,4.5,4.0\n";

fn package(first: &str, second: &str, reverse: bool, role: &str) -> Result<DatasetPackage, String> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.csv"), first).unwrap();
    std::fs::write(dir.path().join("b.csv"), second).unwrap();
    let order = if reverse {
        ["b.csv", "a.csv"]
    } else {
        ["a.csv", "b.csv"]
    };
    let mut spec = json!({"sources":[{"id":"d","role":"mixed","input":order,"merge":"concat_samples","columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":role,"select":["m"]}]}],"params":{"delimiter":","}});
    if role == "id" {
        spec["sources"][0]["columns"][2]["role"] = json!("metadata");
        spec["sources"][0]["key"] = json!("m");
        spec["sample_index"] = json!({"by":"id","key":"m"});
    } else if role == "features" || role == "targets" {
        let index = usize::from(role == "targets");
        let peer = if role == "features" { "x" } else { "y" };
        spec["sources"][0]["columns"][index]["select"] = json!([peer, "m"]);
        spec["sources"][0]["columns"].as_array_mut().unwrap().pop();
    }
    package_with_limits_and_storage(
        &DatasetSpec::from_value(&spec).unwrap(),
        dir.path(),
        LoadLimits::default(),
        &NumericStorageOptions::default(),
    )
    .map_err(|error| error.message)
}

fn assert_typed(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) if e.is_f64() => {
            assert!(a.is_f64());
            assert_eq!(a.as_f64().unwrap().to_bits(), e.as_f64().unwrap().to_bits());
        }
        (Value::Number(a), Value::Number(e)) => {
            assert!(!a.is_f64());
            assert_eq!(a, e);
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                assert_typed(a, e);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

fn assert_metadata(package: &DatasetPackage, expected: &Value) {
    let assembled = package.to_assembled();
    let column = assembled.blocks["train"]
        .metadata
        .as_ref()
        .unwrap()
        .column("m")
        .unwrap();
    assert_eq!(column.dtype, ColDtype::String);
    let full = assembled.to_full_value().unwrap();
    assert_typed(
        &full["blocks"]["train"]["metadata"]["columns"][0]["values"],
        expected,
    );
    let full_package = package.to_full_value().unwrap();
    assert_typed(
        &full_package["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]
            ["values"],
        expected,
    );
    assert_eq!(
        full["audits"],
        json!([{"merge":"concat_samples","source":"d","warnings":[]}])
    );
}

#[test]
fn reviewer_u64_and_text_keep_per_input_json_types_in_both_orders() {
    for reverse in [false, true] {
        let package = package(U64, TEXT, reverse, "metadata").unwrap();
        let expected = if reverse {
            json!(["3", "abc", u64::MAX, 4])
        } else {
            json!([u64::MAX, 4, "3", "abc"])
        };
        assert_metadata(&package, &expected);
    }
}

#[test]
fn reviewer_bool_and_integer_keep_typed_metadata_and_exact_ids_in_both_orders() {
    for reverse in [false, true] {
        for role in ["metadata", "id"] {
            let package = package(BOOL, INT, reverse, role).unwrap();
            let expected = if reverse {
                json!([3, 4, true, false])
            } else {
                json!([true, false, 3, 4])
            };
            assert_metadata(&package, &expected);
            if role == "id" {
                let ids = package
                    .executed_projection(&package.identity.source_ids)
                    .unwrap()
                    .0
                    .sample_ids;
                assert_eq!(
                    ids,
                    if reverse {
                        ["3", "4", "True", "False"]
                    } else {
                        ["True", "False", "3", "4"]
                    }
                );
            }
        }
    }
}

#[test]
fn reviewer_concat_numeric_role_keeps_checked_conversion_and_exact_bits() {
    for reverse in [false, true] {
        let p = package(BOOL, INT, reverse, "weights").unwrap();
        let assembled = p.to_assembled();
        let block = &assembled.blocks["train"];
        let expected: [f64; 4] = if reverse {
            [3.0, 4.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 3.0, 4.0]
        };
        assert_eq!(
            block
                .weights
                .as_ref()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected.map(f64::to_bits)
        );
        assert_eq!(block.weights_validity, Some(vec![true; 4]));
        let full = assembled.to_full_value().unwrap();
        assert_eq!(
            full["blocks"]["train"]["weights"]["precision"],
            json!({"dtype":"float64","origin":"default","source_dtype":null,"requested_dtype":null,"consumer":null})
        );
        let error = package(U64, TEXT, reverse, "weights").unwrap_err();
        assert!(error.contains("exactly representable"));
        for role in ["features", "targets"] {
            let p = package(BOOL, INT, reverse, role).unwrap();
            let assembled = p.to_assembled();
            let block = &assembled.blocks["train"];
            let matrix = if role == "features" {
                &block.x[0]
            } else {
                block.y.as_ref().unwrap()
            };
            let headers = if role == "features" {
                &block.feature_headers[0]
            } else {
                &block.y_headers
            };
            let column = headers.iter().position(|name| name == "m").unwrap();
            assert_eq!(matrix.n_cols, 2);
            assert_eq!(
                matrix
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|row| row[column].to_bits())
                    .collect::<Vec<_>>(),
                expected.map(f64::to_bits)
            );
            assert_eq!(matrix.validity, vec![true; 8]);
            if role == "features" {
                assert!(package(U64, TEXT, reverse, role)
                    .unwrap_err()
                    .contains("exactly representable"));
            }
        }
    }
}

fn concat_closure_package(first: &str, second: &str, role: &str) -> DatasetPackage {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.csv"), first).unwrap();
    std::fs::write(dir.path().join("b.csv"), second).unwrap();
    let mut columns = json!([{"role":"features","select":["x"]},{"role":"targets","select":["y"]}]);
    if role == "features" {
        columns[0]["select"] = json!(["x", "m"]);
    } else if role == "targets" {
        columns[1]["select"] = json!(["m"]);
        columns
            .as_array_mut()
            .unwrap()
            .push(json!({"role":"ignore","select":["y"]}));
    } else {
        columns
            .as_array_mut()
            .unwrap()
            .push(json!({"role":role,"select":["m"]}));
    }
    let spec = json!({"sources":[{"id":"d","role":"mixed","input":["a.csv","b.csv"],"merge":"concat_samples","columns":columns}],"params":{"delimiter":",","na":{"policy":"ignore"}}});
    package_with_limits_and_storage(
        &DatasetSpec::from_value(&spec).unwrap(),
        dir.path(),
        LoadLimits::default(),
        &NumericStorageOptions::default(),
    )
    .unwrap()
}

#[test]
fn reviewer_bool_missing_column_keeps_string_dtype_and_literal_identity_both_orders() {
    // Literal hashes come from the unchanged Python owner on the preserved
    // inspect.py missing-col-bool inputs; raw outputs are in the closure notes.
    for reverse in [false, true] {
        let missing = "x,y\n3.5,3.0\n4.5,4.0\n";
        let (first, second) = if reverse {
            (missing, BOOL)
        } else {
            (BOOL, missing)
        };
        let package = concat_closure_package(first, second, "metadata");
        let assembled = package.to_assembled();
        assert_eq!(
            assembled.blocks["train"]
                .metadata
                .as_ref()
                .unwrap()
                .column("m")
                .unwrap()
                .dtype,
            ColDtype::String
        );
        let full = assembled.to_full_value().unwrap();
        let expected = if reverse {
            json!([null, null, true, false])
        } else {
            json!([true, false, null, null])
        };
        assert_typed(
            &full["blocks"]["train"]["metadata"]["columns"][0]["values"],
            &expected,
        );
        let wire = package.to_full_value().unwrap();
        assert_typed(
            &wire["partitions"]["train"]["payloads"]["train/metadata"]["value"],
            &json!({"n_rows":4,"columns":[{"name":"m","dtype":"string","values":expected}]}),
        );
        assert_eq!(
            full["audits"],
            json!([{"source":"d","merge":"concat_samples","warnings":["concat_samples schema-union: 1 column(s) not shared across files were null-filled"]}])
        );
        assert_eq!(wire["audits"], full["audits"]);
        let (root, hashes) = if reverse {
            (
                "a9346f9f694bc0895d020ae1fc43c9125738559d6158191255c0890899581648",
                [
                    "cc2c8b2e07574f55459a5a513b7a49b58d8aa10552c5dfbbe17bbe9f23d2e18a",
                    "6c0315be98e49009eb701f5293fa8e3b68df139cddde450bd78dad054844bada",
                    "840739b1f312c77c9259df9115accb887fc9a4467b7cfaf152820c0bbae0f583",
                ],
            )
        } else {
            (
                "411ea6742f7d0ca65bc0ab5b7142b148468d9e2182496e8c30050c475277e0f9",
                [
                    "756204b703c6413e69c5ae7a73b0342f6dadd152d8f72cfeaa98f2db3de5d879",
                    "eb27625670fa15a2f1eedb47dc5fbda24407f6ceed610fe5b72c51e1ed5abd0b",
                    "46dd54c91299dafc9b0944fe9eabe6832e92c80c32affea35ab052587f3dc053",
                ],
            )
        };
        assert_eq!(wire["manifest"]["root"], root);
        for (entry, hash) in wire["manifest"]["entries"]
            .as_array()
            .unwrap()
            .iter()
            .zip(hashes)
        {
            assert_eq!(entry["content_hash"], hash);
        }
        assert_eq!(
            wire["manifest"]["entries"].as_array().unwrap().len(),
            hashes.len()
        );
    }
}

#[test]
fn reviewer_bool_missing_column_numeric_roles_keep_exact_wire_both_orders() {
    for reverse in [false, true] {
        for role in ["features", "targets", "weights"] {
            let missing = "x,y\n3.5,3.0\n4.5,4.0\n";
            let (first, second) = if reverse {
                (missing, BOOL)
            } else {
                (BOOL, missing)
            };
            let package = concat_closure_package(first, second, role);
            let full = package.to_assembled().to_full_value().unwrap();
            let block = &full["blocks"]["train"];
            let matrix = match role {
                "features" => &block["x"][0],
                "targets" => &block["y"]["columns"][0]["block"],
                _ => &block["weights"],
            };
            let (bytes, validity) = if role == "features" {
                if reverse {
                    ("0000000000000c40000000000000000000000000000012400000000000000000000000000000f83f000000000000f03f00000000000004400000000000000000", json!([true, false, true, false, true, true, true, true]))
                } else {
                    ("000000000000f03f000000000000f83f0000000000000000000000000000044000000000000000000000000000000c4000000000000000000000000000001240", json!([true, true, true, true, false, true, false, true]))
                }
            } else if reverse {
                (
                    "00000000000000000000000000000000000000000000f03f0000000000000000",
                    json!([false, false, true, true]),
                )
            } else {
                (
                    "000000000000f03f000000000000000000000000000000000000000000000000",
                    json!([true, true, false, false]),
                )
            };
            assert_typed(
                matrix,
                &json!({"dtype":"float64","n_rows":4,"n_cols":if role == "features" {2} else {1},"data_bytes_hex":bytes,"validity":validity,"precision":{"dtype":"float64","origin":"default","source_dtype":null,"requested_dtype":null,"consumer":null}}),
            );
            let root = match (reverse, role) {
                (false, "features") => {
                    "4e458d55cb5128674576a94162ef135142f1c0962e010767ba43d8853082b73d"
                }
                (true, "features") => {
                    "d9657218a435a4b77f5e5b020fdcd47937a450b534f648763a01420f23fdaf69"
                }
                (false, "targets") => {
                    "c87bd8b6bfd54837ddcfd3f800e8176d0556e31873d2d743cb726c12d82617da"
                }
                (true, "targets") => {
                    "8bd2ee26757cd380051a85041aa82cdd3aa910860f3b2f7da1f52e6e03d370ca"
                }
                (false, _) => "94a6850dffb51f3869db66a20531b800cf8c10cbaede2766f95829fb043e89dc",
                (true, _) => "dadd20cdfa8ee532a3173b253820b25eba498673972951382c02b52c4191f127",
            };
            assert_eq!(package.to_full_value().unwrap()["manifest"]["root"], root);
        }
    }
}

#[test]
fn reviewer_concat_numeric_promotion_and_all_na_bool_controls_stay_unchanged() {
    // These exact concat_shapes.py/inspect.py controls were already correct
    // in the preserved Rust owner. Expectations are literal owner outputs.
    let cases = [
        (
            "int-float",
            "m,x,y\n3,1.5,1.0\n4,2.5,2.0\n",
            "m,x,y\n1.5,3.5,3.0\n2.5,4.5,4.0\n",
            json!([3.0, 4.0, 1.5, 2.5]),
            ColDtype::Numeric,
            "24828be616b697ab75cf29a465c9eb632b9e27c1297ebd3e5c8acb57279bb4da",
        ),
        (
            "float-int",
            "m,x,y\n1.5,1.5,1.0\n2.5,2.5,2.0\n",
            "m,x,y\n3,3.5,3.0\n4,4.5,4.0\n",
            json!([1.5, 2.5, 3.0, 4.0]),
            ColDtype::Numeric,
            "2b1ba68f920820620b7913fbb29f32f74d56bb1cde8d0071a02e1c64175e9f5f",
        ),
        (
            "intna-int",
            "m,x,y\n3,1.5,1.0\n,2.5,2.0\n",
            "m,x,y\n5,3.5,3.0\n6,4.5,4.0\n",
            json!([3.0, null, 5.0, 6.0]),
            ColDtype::Numeric,
            "92bba889d4d1d17f67b60a95b866c2ce1d2ca7e24bd22c15818ac8fc8bdd8553",
        ),
        (
            "int-intna",
            "m,x,y\n5,1.5,1.0\n6,2.5,2.0\n",
            "m,x,y\n3,3.5,3.0\n,4.5,4.0\n",
            json!([5.0, 6.0, 3.0, null]),
            ColDtype::Numeric,
            "9b4404a2afced75c9397cdbaf5ccfdd8818147b812747d7082be4e03e3422810",
        ),
        (
            "allna-int",
            "m,x,y\n,1.5,1.0\n,2.5,2.0\n",
            "m,x,y\n3,3.5,3.0\n4,4.5,4.0\n",
            json!([null, null, 3.0, 4.0]),
            ColDtype::Numeric,
            "b3563d292fd034d6b20f15d27186b0d2d9b69c91f185c9ab4ed3d6e1aa696298",
        ),
        (
            "int-allna",
            "m,x,y\n3,1.5,1.0\n4,2.5,2.0\n",
            "m,x,y\n,3.5,3.0\n,4.5,4.0\n",
            json!([3.0, 4.0, null, null]),
            ColDtype::Numeric,
            "0f32d400748e7bfbf398262eca0caff58f0e268488f260112b5401833045fce5",
        ),
        (
            "allna-bool",
            "m,x,y\n,1.5,1.0\n,2.5,2.0\n",
            "m,x,y\nTRUE,3.5,3.0\nFALSE,4.5,4.0\n",
            json!([null, null, true, false]),
            ColDtype::String,
            "435d241864ef76aef897498e216965590d2601ddcc37d17284ae371b142375c0",
        ),
        (
            "bool-allna",
            "m,x,y\nTRUE,3.5,3.0\nFALSE,4.5,4.0\n",
            "m,x,y\n,1.5,1.0\n,2.5,2.0\n",
            json!([true, false, null, null]),
            ColDtype::String,
            "f589dc5ff2ad5ac9297968d830ca326b388863016a4a13a5f0a67df524ba0a49",
        ),
    ];
    for (name, first, second, expected, dtype, root) in cases {
        let package = concat_closure_package(first, second, "metadata");
        let assembled = package.to_assembled();
        assert_eq!(
            assembled.blocks["train"]
                .metadata
                .as_ref()
                .unwrap()
                .column("m")
                .unwrap()
                .dtype,
            dtype,
            "{name}"
        );
        let full = assembled.to_full_value().unwrap();
        assert_typed(
            &full["blocks"]["train"]["metadata"]["columns"][0]["values"],
            &expected,
        );
        let wire = package.to_full_value().unwrap();
        assert_typed(
            &wire["partitions"]["train"]["payloads"]["train/metadata"]["value"]["columns"][0]
                ["values"],
            &expected,
        );
        assert_eq!(wire["manifest"]["root"], root, "{name}");
    }
}
