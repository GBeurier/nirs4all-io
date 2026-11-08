// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Literal source-schema identities from the unchanged Python public owner.
use nirs4all_io_core::public_dataset::{
    canonical_source_schema, compatible_source_schemas, u07_sources,
};
use serde_json::{json, Value};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/source_schema_canonical_vectors.json"
    ))
    .unwrap()
}

fn case<'a>(fixture: &'a Value, id: &str) -> &'a Value {
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == id)
        .unwrap()
}

// Value equality alone does not assert float bits, notably the sign of zero.
fn assert_typed(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            assert_eq!(a.is_f64(), b.is_f64(), "integer/float kind");
            if a.is_f64() {
                assert_eq!(a.as_f64().unwrap().to_bits(), b.as_f64().unwrap().to_bits());
            } else {
                assert_eq!(a, b);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in b {
                assert_typed(a.get(key).unwrap(), value);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_typed(a, b);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn shared_source_schema_vectors_keep_exact_identity_bytes_and_outer_order() {
    let fixture = fixture();
    assert_eq!(
        fixture["schema"],
        "nirs4all.io.source-schema-canonical-vectors.v1"
    );
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 40);
    for case in fixture["cases"].as_array().unwrap() {
        let actual = canonical_source_schema(&case["input"]).unwrap();
        let expected = &case["expected"];
        assert_eq!(
            actual["identity"].as_str().unwrap().as_bytes(),
            expected["identity"].as_str().unwrap().as_bytes(),
            "{}",
            case["id"]
        );
        assert_typed(&actual, expected);
        // The frozen canonical input has absent null units; keep it byte-stable too.
        assert_typed(&canonical_source_schema(expected).unwrap(), expected);
        assert_eq!(
            actual.as_object().unwrap().keys().collect::<Vec<_>>(),
            case["input"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>()
        );
        let actual_inner: Value =
            serde_json::from_str(actual["identity"].as_str().unwrap()).unwrap();
        let expected_inner: Value =
            serde_json::from_str(expected["identity"].as_str().unwrap()).unwrap();
        assert_typed(&actual_inner, &expected_inner);
    }
}

#[test]
fn literal_coordinate_anchors() {
    let fixture = fixture();
    for (id, literal) in [
        ("integer.nir", "\"wavelength\":[900,910,920]"),
        ("integral_float.nir", "\"wavelength\":[900.0,910.0,920.0]"),
        ("mixed.nir", "\"wavelength\":[900,910.0,920]"),
        (
            "adjacent.nir",
            "\"wavelength\":[1e-07,1.0000000000000001e-07,2e-07]",
        ),
        ("geometry.nir", "\"wavelength\":[-0.0,1.5e-300,1e-07]"),
    ] {
        assert!(case(&fixture, id)["expected"]["identity"]
            .as_str()
            .unwrap()
            .contains(literal));
    }
    assert!(case(&fixture, "geometry.metadata")["expected"]["identity"]
        .as_str()
        .unwrap()
        .contains("quote\\\" backslash\\\\ newline\\n control\\u0001 separator\u{2028}"));
}

#[test]
fn compatibility_preserves_saved_literal_and_semantic_controls() {
    let fixture = fixture();
    let current = json!({"nir": case(&fixture, "base.nir")["input"]});
    let saved = json!({"nir": case(&fixture, "permuted.nir")["input"]});
    assert_typed(
        &compatible_source_schemas(&current, &saved).unwrap(),
        &saved,
    );
    let mismatch = json!({"nir": case(&fixture, "integer.nir")["input"]});
    assert_eq!(
        compatible_source_schemas(&current, &mismatch).unwrap_err(),
        "Source nir schema differs (shape, dtype, axes, units, coordinates or columns)"
    );
    let saved = json!({"nir": case(&fixture, "geometry.nir")["input"]});
    let mut current = saved.clone();
    let mut descriptor: Value =
        serde_json::from_str(current["nir"]["identity"].as_str().unwrap()).unwrap();
    descriptor["axis_coordinates"]["wavelength"][0] = json!(0);
    current["nir"]["identity"] = json!(serde_json::to_string(&descriptor).unwrap());
    assert_typed(
        &compatible_source_schemas(&current, &saved).unwrap(),
        &saved,
    );
}

#[test]
fn saved_identity_shapes_return_results_across_direct_current_and_saved_routes() {
    let identities = [
        "[1]",
        "[]",
        "\"x\"",
        "1",
        "1.5",
        "true",
        "false",
        "null",
        "{}",
        "{\"axis_units\":null}",
        "{\"axis_units\":[]}",
        "{\"axis_units\":1}",
        "{\"axis_units\":\"x\"}",
        "{\"axis_units\":{}}",
        "{\"axis_units\":{\"a\":null}}",
    ];
    let mut calls = 0;
    let mut accepted = 0;
    let mut refused = 0;
    let mut panics = 0;
    let mut failures = Vec::new();
    for (index, identity) in identities.iter().enumerate() {
        for representation in ["signal_1d", "tabular_mixed"] {
            let schema = json!({"representation_id":representation,"input_shape":[2],"dtype":"float64","identity":identity});
            let good_schema = json!({"representation_id":representation,"input_shape":[2],"dtype":"float64","identity":"{\"axis_units\":{}}"});
            let current = json!({"n":schema});
            let good = json!({"n":good_schema});
            for route in ["direct", "current", "saved"] {
                calls += 1;
                // Catch only to record every actual owner call; any panic fails the matrix.
                let result = std::panic::catch_unwind(|| match route {
                    "direct" => canonical_source_schema(&schema),
                    "current" => compatible_source_schemas(&current, &good),
                    "saved" => compatible_source_schemas(&good, &current),
                    _ => unreachable!(),
                });
                let (outcome, error, output) = match &result {
                    Ok(Ok(value)) => {
                        accepted += 1;
                        ("accept", None, Some(value))
                    }
                    Ok(Err(error)) => {
                        refused += 1;
                        ("refuse", Some(error.as_str()), None)
                    }
                    Err(_) => {
                        panics += 1;
                        ("panic", None, None)
                    }
                };
                eprintln!(
                    "P9_ROW {}",
                    json!({"identity":identity,"representation":representation,"route":route,"outcome":outcome,"error":error,"output":output})
                );
                if index < 13 {
                    if !matches!(&result, Ok(Err(error)) if error == "Source axis units required") {
                        failures.push(format!("{representation}/{route}/{identity}: expected Source axis units required, got {outcome}"));
                    }
                } else {
                    let mut expected = match route {
                        "direct" => good_schema.clone(),
                        "current" => good.clone(),
                        "saved" => current.clone(),
                        _ => unreachable!(),
                    };
                    if route == "direct" && representation == "tabular_mixed" {
                        expected["dtype"] = json!("object");
                        expected["identity"] = json!("{\"axis_units\":{},\"dtype\":\"object\"}");
                    }
                    match result {
                        Ok(Ok(value)) => assert_typed(&value, &expected),
                        _ => failures.push(format!("{representation}/{route}/{identity}: expected acceptance, got {outcome}")),
                    }
                }
            }
        }
    }
    eprintln!("P9_MATRIX calls={calls} accepted={accepted} refused={refused} panics={panics}");
    assert_eq!(calls, 90);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!((accepted, refused, panics), (12, 78, 0));
}

fn flatten(value: &Value, out: &mut Vec<Value>) {
    if let Some(array) = value.as_array() {
        for cell in array {
            flatten(cell, out);
        }
    } else {
        out.push(value.clone());
    }
}

#[test]
fn original_public_wrapper_composes_through_rust_u07_owner() {
    let fixture = fixture();
    for group in [
        "base",
        "permuted",
        "integer",
        "integral_float",
        "mixed",
        "adjacent",
        "geometry",
        "null_unit",
        "absent_unit",
        "tabular_text",
    ] {
        let mut original: Value = serde_json::from_str(ORIGINAL_PUBLIC_WRAPPER).unwrap();
        let mut schemas = serde_json::Map::new();
        for source in original["dataset"]["sources"].as_array_mut().unwrap() {
            let name = source["name"].as_str().unwrap().to_owned();
            let row = case(&fixture, &format!("{group}.{name}"));
            let descriptor: Value =
                serde_json::from_str(row["input"]["identity"].as_str().unwrap()).unwrap();
            for field in ["axis_units", "axis_coordinates", "feature_names"] {
                source[field] = descriptor[field].clone();
            }
            source["array"]["dtype"] = row["input"]["dtype"].clone();
            if group == "tabular_text" && name == "metadata" {
                source["array"]["values"] = json!([["20.5", "A"], ["21.5", "B"]]);
            }
            schemas.insert(name, row["expected"].clone());
        }
        let mut sources = serde_json::Map::new();
        for source in original["dataset"]["sources"].as_array().unwrap() {
            let name = source["name"].as_str().unwrap();
            let mut entry = json!({"sample_ids": original["dataset"]["sample_ids"], "descriptor": schemas[name], "shape": source["array"]["shape"]});
            if name == "metadata" {
                entry["rows"] = source["array"]["values"].clone();
            } else {
                let mut data = Vec::new();
                flatten(&source["array"]["values"], &mut data);
                entry["data"] = json!(data);
            }
            sources.insert(name.to_owned(), entry);
        }
        let expected = json!({"sample_ids": original["dataset"]["sample_ids"], "source_schemas": schemas, "sources": sources});
        // u07_sources accepts the original public wrapper and normalizes internally.
        assert_typed(&u07_sources(&original).unwrap(), &expected);
    }
}

// Actual runtime-u07-groups.json wrapper, embedded to keep this crate test package-safe.
const ORIGINAL_PUBLIC_WRAPPER: &str = r#"{
  "schema": "nirs4all.dataset.v1",
  "schema_version": 1,
  "dataset": {
    "schema": "nirs4all.multimodal-dataset",
    "schema_version": 1,
    "name": "multimodal",
    "sample_ids": [
      "p0",
      "p1"
    ],
    "source_alignment": "strict",
    "sources": [
      {
        "name": "nir",
        "sample_ids": [
          "p0",
          "p1"
        ],
        "representation_id": "signal_1d",
        "axes": [
          "sample",
          "wavelength"
        ],
        "feature_names": null,
        "axis_units": {},
        "axis_coordinates": {},
        "array": {
          "dtype": "float64",
          "shape": [
            2,
            3
          ],
          "values": [
            [
              1.0,
              2.0,
              3.0
            ],
            [
              4.0,
              5.0,
              6.0
            ]
          ]
        },
        "presence_mask": {
          "dtype": "bool",
          "shape": [
            2
          ],
          "values": [
            true,
            true
          ]
        }
      },
      {
        "name": "image",
        "sample_ids": [
          "p0",
          "p1"
        ],
        "representation_id": "rgb_image",
        "axes": [
          "sample",
          "height",
          "width",
          "channel"
        ],
        "feature_names": null,
        "axis_units": {},
        "axis_coordinates": {},
        "array": {
          "dtype": "float64",
          "shape": [
            2,
            1,
            1,
            3
          ],
          "values": [
            [
              [
                [
                  0.0,
                  1.0,
                  2.0
                ]
              ]
            ],
            [
              [
                [
                  3.0,
                  4.0,
                  5.0
                ]
              ]
            ]
          ]
        },
        "presence_mask": {
          "dtype": "bool",
          "shape": [
            2
          ],
          "values": [
            true,
            true
          ]
        }
      },
      {
        "name": "series",
        "sample_ids": [
          "p0",
          "p1"
        ],
        "representation_id": "series_mv",
        "axes": [
          "sample",
          "time",
          "variable"
        ],
        "feature_names": null,
        "axis_units": {},
        "axis_coordinates": {},
        "array": {
          "dtype": "float64",
          "shape": [
            2,
            2,
            1
          ],
          "values": [
            [
              [
                1.0
              ],
              [
                2.0
              ]
            ],
            [
              [
                3.0
              ],
              [
                4.0
              ]
            ]
          ]
        },
        "presence_mask": {
          "dtype": "bool",
          "shape": [
            2
          ],
          "values": [
            true,
            true
          ]
        }
      },
      {
        "name": "metadata",
        "sample_ids": [
          "p0",
          "p1"
        ],
        "representation_id": "tabular_mixed",
        "axes": [
          "sample",
          "column"
        ],
        "feature_names": [
          "temperature",
          "cultivar"
        ],
        "axis_units": {},
        "axis_coordinates": {},
        "array": {
          "dtype": "object",
          "shape": [
            2,
            2
          ],
          "values": [
            [
              20.5,
              "A"
            ],
            [
              21.5,
              "B"
            ]
          ]
        },
        "presence_mask": {
          "dtype": "bool",
          "shape": [
            2
          ],
          "values": [
            true,
            true
          ]
        }
      }
    ],
    "y": null,
    "target_names": [],
    "target_mask": null,
    "task_type": null,
    "groups": {
      "dtype": "<U3",
      "shape": [
        2
      ],
      "values": [
        "1.0",
        "1"
      ]
    },
    "partitions": {
      "dtype": "<U7",
      "shape": [
        2
      ],
      "values": [
        "predict",
        "predict"
      ]
    }
  },
  "origin_ids": [
    "p0",
    "p1"
  ],
  "fold_ids": [
    null,
    null
  ]
}"#;
