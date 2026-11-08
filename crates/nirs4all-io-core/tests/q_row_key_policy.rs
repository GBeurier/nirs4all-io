// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Source-only vectors; native execution requires the coherent qualification grant.
use nirs4all_io_core::conventions::{assignments_to_spec_dict, match_items, resolve_profiles};
use nirs4all_io_core::infer::identity::{configure_generated_keys, refuse_generated_identity};
use nirs4all_io_core::infer::memory::{infer_named_bytes, NamedBytes};
use nirs4all_io_core::infer::plan::DatasetPlan;
use nirs4all_io_core::infer::propose::{propose_browser_dataset, ConfirmedLock};
use nirs4all_io_core::materialize::{
    assemble_in_memory, AssembledDataset, DatasetPackage, InMemorySource, SourcePayload,
};
use nirs4all_io_core::spec::{DatasetSpec, SpecError};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
enum MaterializationPhase {
    Raw,
    Package,
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/q_row_key_policy_v1.json")).unwrap()
}
fn bytes(source: &Value, files: &[NamedBytes]) -> Option<Vec<Vec<u8>>> {
    let names: Vec<&str> = match &source["input"] {
        Value::String(name) => vec![name],
        Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    names
        .iter()
        .map(|name| {
            files
                .iter()
                .find(|file| file.name == *name)
                .map(|file| file.bytes.clone())
        })
        .collect()
}
fn files(case: &Value) -> Vec<NamedBytes> {
    case["files"]
        .as_object()
        .unwrap()
        .iter()
        .chain(case["extra"].as_object().into_iter().flatten())
        .map(|(name, text)| NamedBytes {
            name: name.clone(),
            bytes: text.as_str().unwrap().as_bytes().to_vec(),
        })
        .collect()
}
fn explicit(case: &Value, method: &str) -> Value {
    let mut spec = json!({"sources":[{"id":"x","role":"features","input":"Xcal.csv"},{"id":"y","role":"targets","input":"Ycal.csv","params":case.get("params").cloned().unwrap_or(json!({}))}]});
    if !case["source_mode"].is_null() {
        spec["params"] = json!({"categorical":case["global_mode"]});
        spec["sources"][1]["params"]["categorical"] = case["source_mode"].clone();
    }
    if method == "explicit-m1" {
        spec["sample_index"] = json!({"by":"id","key":"id"});
        spec["sources"][0]["key"] = json!("id");
        spec["sources"][1]["key"] = json!("id");
        spec["sources"][1]["join"] = json!({"to":"x","on":"id","how":"m:1"});
    }
    spec
}
fn assert_targets(a: &AssembledDataset, case: &Value, phase: MaterializationPhase) {
    let block = &a.blocks["train"];
    let y = block.y.as_ref().unwrap();
    let rows = case["targets"].as_array().unwrap();
    assert_eq!(y.n_rows, rows.len(), "{}", case["id"]);
    assert_eq!(
        y.n_cols,
        rows[0].as_array().unwrap().len(),
        "{}",
        case["id"]
    );
    assert_eq!(y.validity.len(), y.data.len());
    for ((actual, valid), expected) in y
        .data
        .iter()
        .zip(&y.validity)
        .zip(rows.iter().flat_map(|r| r.as_array().unwrap()))
    {
        assert_eq!(*valid, !expected.is_null(), "{}", case["id"]);
        if expected.is_null() {
            match phase {
                MaterializationPhase::Raw => assert!(actual.is_nan(), "{}", case["id"]),
                MaterializationPhase::Package => assert_eq!(actual.to_bits(), 0, "{}", case["id"]),
            }
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.as_f64().unwrap().to_bits(),
                "{}",
                case["id"]
            );
        }
    }
    if let Some(headers) = case["headers"].as_array() {
        assert_eq!(
            block.y_headers,
            headers
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        );
    }
    if let Some(identity) = case["identity"].as_array() {
        assert_eq!(
            block.metadata.as_ref().unwrap().str_column("id"),
            identity
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            block.feature_headers,
            vec![vec!["400".to_string(), "500".to_string()]]
        );
    }
}

#[test]
fn literal_byte_owner_vectors_and_package() {
    for case in fixture()["rows"].as_array().unwrap() {
        let inputs = files(case);
        let payloads: Vec<_> = inputs
            .iter()
            .map(|f| InMemorySource {
                name: f.name.clone(),
                payload: SourcePayload::Bytes(f.bytes.clone()),
            })
            .collect();
        for method in case["methods"].as_array().unwrap() {
            let method = method.as_str().unwrap();
            let spec = if method == "infer" {
                infer_named_bytes(&inputs, &Default::default())
                    .unwrap()
                    .resolved_spec
                    .unwrap()
            } else if method.starts_with("explicit") {
                DatasetSpec::from_value(&explicit(case, method)).unwrap()
            } else {
                let names: Vec<_> = inputs.iter().map(|f| f.name.clone()).collect();
                let profiles = resolve_profiles(&["nirs4all-classic"]).unwrap();
                let matched = match_items(&names, &profiles, None, None);
                let mut value = assignments_to_spec_dict(&matched, "literal");
                configure_generated_keys(
                    &mut value,
                    &mut DatasetPlan::default(),
                    |source| bytes(source, &inputs),
                    false,
                );
                DatasetSpec::from_value(&value).unwrap()
            };
            let before = spec.to_value();
            // This is the actual owner generated-proposal guard followed by
            // actual SourcePayload::Bytes assembly, never a model of the join.
            let result: Result<_, SpecError> = (|| {
                if !method.starts_with("explicit") {
                    refuse_generated_identity(&spec, |source| bytes(source, &inputs))?;
                }
                assemble_in_memory(&spec, &payloads, &HashMap::new(), None)
            })();
            if case["status"] == "refused" {
                assert!(result.is_err(), "{} {method}", case["id"]);
                continue;
            }
            let a = result.unwrap_or_else(|e| panic!("{} {method}: {}", case["id"], e.message));
            assert_targets(&a, case, MaterializationPhase::Raw);
            let package = DatasetPackage::from_assembled(&a).unwrap();
            assert_targets(&package.to_assembled(), case, MaterializationPhase::Package);
            assert_eq!(spec.to_value(), before);
        }
    }
}

#[test]
fn confirmed_owner_locks_keep_their_real_call_path() {
    let f = fixture();
    let case = f["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "shuffled")
        .unwrap();
    let inputs = files(case);
    let payloads: Vec<_> = inputs
        .iter()
        .map(|f| InMemorySource {
            name: f.name.clone(),
            payload: SourcePayload::Bytes(f.bytes.clone()),
        })
        .collect();
    for lock in f["controls"]["locks"].as_array().unwrap() {
        let confirmed = ConfirmedLock {
            kind: lock["kind"].as_str().unwrap().into(),
            target: lock["target"].as_str().unwrap().into(),
            value: lock["value"].clone(),
            status: None,
        };
        let proposal =
            propose_browser_dataset(&inputs, &[], &[confirmed], &Default::default()).unwrap();
        let a = assemble_in_memory(&proposal.spec, &payloads, &HashMap::new(), None).unwrap();
        let y = a.blocks["train"].y.as_ref().unwrap();
        let actual: Vec<_> = y
            .data
            .chunks(y.n_cols)
            .map(|row| *row.last().unwrap())
            .collect();
        let expected: Vec<_> = lock["target_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(actual, expected);
        if lock["value"]["mode"] == "row_order" {
            assert!(proposal.spec.to_value()["sources"][1].get("join").is_none());
        }
    }
}

#[test]
fn every_input_participates_in_bounded_selection() {
    let f = fixture();
    let c = &f["controls"]["multi_input"];
    let names = c["inputs"].as_array().unwrap();
    let inputs = vec![
        NamedBytes {
            name: names[0].as_str().unwrap().into(),
            bytes: c["first"].as_str().unwrap().as_bytes().to_vec(),
        },
        NamedBytes {
            name: names[1].as_str().unwrap().into(),
            bytes: c["second"].as_str().unwrap().as_bytes().to_vec(),
        },
    ];
    let mut spec =
        json!({"sources":[{"id":"x","role":"features","input":names,"merge":"concat_samples"}]});
    let before = spec.clone();
    configure_generated_keys(
        &mut spec,
        &mut DatasetPlan::default(),
        |source| bytes(source, &inputs),
        false,
    );
    assert_eq!(spec, before);
}

#[test]
fn changed_identity_lock_preserves_the_independent_join() {
    let f = fixture();
    let case = &f["controls"]["identity_lock_change"];
    let inputs = files(case);
    let lock = ConfirmedLock {
        kind: case["kind"].as_str().unwrap().into(),
        target: case["target"].as_str().unwrap().into(),
        value: case["value"].clone(),
        status: None,
    };
    let proposal = propose_browser_dataset(&inputs, &[], &[lock], &Default::default()).unwrap();
    assert_eq!(
        proposal.spec.to_value()["sample_index"]["key"],
        "sample_alt"
    );
    let payloads: Vec<_> = inputs
        .iter()
        .map(|file| InMemorySource {
            name: file.name.clone(),
            payload: SourcePayload::Bytes(file.bytes.clone()),
        })
        .collect();
    let a = assemble_in_memory(&proposal.spec, &payloads, &HashMap::new(), None).unwrap();
    assert_eq!(
        a.blocks["train"].y.as_ref().unwrap().data,
        vec![10.5, 20.5, 30.5]
    );
    assert_eq!(
        a.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .str_column("sample_alt"),
        vec!["a", "b", "c"]
    );
}

#[test]
fn decimal_and_three_column_positive_counterexamples() {
    for case in fixture()["controls"]["positive_counterexamples"]
        .as_array()
        .unwrap()
    {
        let inputs = vec![
            NamedBytes {
                name: "Xcal.csv".into(),
                bytes: case["x"].as_str().unwrap().as_bytes().to_vec(),
            },
            NamedBytes {
                name: "Ycal.csv".into(),
                bytes: case["y"].as_str().unwrap().as_bytes().to_vec(),
            },
        ];
        let spec = infer_named_bytes(&inputs, &Default::default())
            .unwrap()
            .resolved_spec
            .unwrap();
        let payloads: Vec<_> = inputs
            .iter()
            .map(|file| InMemorySource {
                name: file.name.clone(),
                payload: SourcePayload::Bytes(file.bytes.clone()),
            })
            .collect();
        let a = assemble_in_memory(&spec, &payloads, &HashMap::new(), None).unwrap();
        assert_targets(&a, case, MaterializationPhase::Raw);
        assert_targets(
            &DatasetPackage::from_assembled(&a).unwrap().to_assembled(),
            case,
            MaterializationPhase::Package,
        );
    }
}

fn a5_inputs(metadata: &str) -> Vec<NamedBytes> {
    [
        ("Xcal.csv", "400;500\n0.1;0.2\n0.3;0.4\n0.5;0.6\n"),
        ("Ycal.csv", "y\n1.5\n2.5\n3.5\n"),
        ("Mcal.csv", metadata),
    ]
    .into_iter()
    .map(|(name, text)| NamedBytes {
        name: name.into(),
        bytes: text.as_bytes().to_vec(),
    })
    .collect()
}

#[test]
fn a5_metadata_labels_keep_unverified_positional_values() {
    for (metadata, labels, other, values) in [
        (
            "name;site\nSmith, J;A\nDoe, B;B\nLee, C;C\n",
            vec!["Smith, J", "Doe, B", "Lee, C"],
            "site",
            vec!["A", "B", "C"],
        ),
        (
            "name;site\n12,1;A\n12,10;B\n14,2;C\n",
            vec!["12,1", "12,10", "14,2"],
            "site",
            vec!["A", "B", "C"],
        ),
        (
            "name;ph\nA;6,5\nB;7,5\nC;8,5\n",
            vec!["A", "B", "C"],
            "ph",
            vec!["6.5", "7.5", "8.5"],
        ),
    ] {
        let inputs = a5_inputs(metadata);
        let plan = infer_named_bytes(&inputs, &Default::default()).unwrap();
        let spec = plan.resolved_spec.as_ref().unwrap();
        assert_eq!(
            spec.to_value()["sample_index"]["by"]
                .as_str()
                .unwrap_or("row"),
            "row"
        );
        assert!(plan.identity.is_none());
        assert!(plan.alignment.is_empty());
        assert!(plan.params.get("identity_repetition").is_none());
        assert!(plan.warnings.iter().any(|warning| warning == "partition train: metadata label 'name' in source 'train_m' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity"));
        let sources = inputs
            .iter()
            .map(|input| InMemorySource {
                name: input.name.clone(),
                payload: SourcePayload::Bytes(input.bytes.clone()),
            })
            .collect::<Vec<_>>();
        let assembled = assemble_in_memory(spec, &sources, &HashMap::new(), None).unwrap();
        assert_eq!(
            assembled.blocks["train"].x[0].data,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        );
        assert_eq!(
            assembled.blocks["train"].y.as_ref().unwrap().data,
            [1.5, 2.5, 3.5]
        );
        let metadata = assembled.blocks["train"].metadata.as_ref().unwrap();
        assert_eq!(metadata.str_column("name"), labels);
        assert_eq!(metadata.str_column(other), values);
        let package = DatasetPackage::from_assembled(&assembled).unwrap();
        assert!(package.identity.sample_id.is_none());
        assert!(package.row_position_fallback.used);
    }
}

#[test]
fn a5_column_presence_and_global_partition_refusals_stay_real() {
    for case in [
        "metadata-sharing",
        "column-presence",
        "mixed",
        "two-local",
        "unequal",
        "reordered",
    ] {
        let mut inputs = a5_inputs("name;site\ns1;A\ns2;B\ns3;C\n");
        let mut value = json!({"sources":[
            {"id":"x","input":"Xcal.csv","role":"features","partition":"train"},
            {"id":"y","input":"Ycal.csv","role":"targets","partition":"train","join":{"to":"x"}},
            {"id":"m","input":"Mcal.csv","role":"metadata","partition":"train","join":{"to":"x"}}
        ]});
        match case {
            "metadata-sharing" => {
                inputs.push(NamedBytes {
                    name: "M2.csv".into(),
                    bytes: b"name;batch\ns1;a\ns2;b\ns3;c\n".to_vec(),
                });
                value["sources"].as_array_mut().unwrap().push(json!({"id":"m2","input":"M2.csv","role":"metadata","partition":"train","join":{"to":"x"}}));
            }
            "column-presence" => {
                inputs[0].bytes = b"code;400\n1.5;0.1\n1.5;0.3\n2.5;0.5\n".to_vec();
                inputs[2].bytes = b"code;site\n1;A\n2;B\n3;C\n".to_vec();
            }
            "mixed" | "two-local" => {
                let (x, y) = if case == "mixed" {
                    (
                        "id;400\na;0.1\nb;0.3\nc;0.5\n",
                        "id;y\na;1.5\nb;2.5\nc;3.5\n",
                    )
                } else {
                    ("400;500\n0.1;0.2\n0.3;0.4\n0.5;0.6\n", "y\n1.5\n2.5\n3.5\n")
                };
                inputs.extend([
                    NamedBytes {
                        name: "Xt.csv".into(),
                        bytes: x.as_bytes().to_vec(),
                    },
                    NamedBytes {
                        name: "Yt.csv".into(),
                        bytes: y.as_bytes().to_vec(),
                    },
                ]);
                value["sources"].as_array_mut().unwrap().extend([json!({"id":"xt","input":"Xt.csv","role":"features","partition":"test"}),json!({"id":"yt","input":"Yt.csv","role":"targets","partition":"test","join":{"to":"xt"}})]);
                if case == "two-local" {
                    inputs.push(NamedBytes {
                        name: "Mt.csv".into(),
                        bytes: b"sample_id;site\na;A\nb;B\nc;C\n".to_vec(),
                    });
                    value["sources"].as_array_mut().unwrap().push(json!({"id":"mt","input":"Mt.csv","role":"metadata","partition":"test","join":{"to":"xt"}}));
                }
            }
            "unequal" => inputs[2].bytes.extend_from_slice(b"s4;D\ns5;E\n"),
            "reordered" => inputs[2].bytes = b"name;site\ns3;C\ns1;A\ns2;B\n".to_vec(),
            _ => unreachable!(),
        }
        let mut plan = DatasetPlan::default();
        configure_generated_keys(&mut value, &mut plan, |source| bytes(source, &inputs), true);
        let expected_key = match case {
            "column-presence" => Some("code"),
            "mixed" => Some("id"),
            _ => None,
        };
        assert_eq!(
            value["sample_index"]["key"].as_str(),
            expected_key,
            "{case}"
        );
        if expected_key.is_none() {
            assert!(
                plan.identity.is_none() && plan.alignment.is_empty(),
                "{case}"
            );
            assert!(plan.params.get("identity_repetition").is_none());
        }
        let spec = DatasetSpec::from_value(&value).unwrap();
        let sources = inputs
            .iter()
            .map(|input| InMemorySource {
                name: input.name.clone(),
                payload: SourcePayload::Bytes(input.bytes.clone()),
            })
            .collect::<Vec<_>>();
        let result = assemble_in_memory(&spec, &sources, &HashMap::new(), None);
        if matches!(case, "column-presence" | "mixed" | "unequal") {
            assert!(result.is_err(), "{case}");
        } else {
            let assembled = result.unwrap();
            assert_eq!(
                assembled.blocks["train"].y.as_ref().unwrap().data,
                [1.5, 2.5, 3.5]
            );
            if case == "reordered" {
                assert_eq!(
                    assembled.blocks["train"]
                        .metadata
                        .as_ref()
                        .unwrap()
                        .str_column("name"),
                    ["s3", "s1", "s2"]
                );
            }
        }
    }
}

#[test]
fn a5_i6_only_reprofiles_metadata_without_any_global_identity() {
    for (metadata, column, expected, comma, warning) in [
        (
            "name;site\n12,5;A\n13,1;B\n14,2;C\n",
            "name",
            vec!["12,5", "13,1", "14,2"],
            false,
            true,
        ),
        (
            "index;site\n1,33;A\n2,44;B\n3,55;C\n",
            "index",
            vec!["1,33", "2,44", "3,55"],
            false,
            true,
        ),
        (
            "ph;site\n6,5;A\n7,5;B\n8,5;C\n",
            "ph",
            vec!["6.5", "7.5", "8.5"],
            true,
            false,
        ),
        (
            "id;site\n1.1;A\n2.2;B\n3.3;C\n",
            "id",
            vec!["1.1", "2.2", "3.3"],
            false,
            false,
        ),
    ] {
        let inputs = a5_inputs(metadata);
        let plan = infer_named_bytes(&inputs, &Default::default()).unwrap();
        let spec = plan.resolved_spec.as_ref().unwrap();
        let source = spec
            .sources
            .iter()
            .find(|source| source.role.value() == "metadata")
            .unwrap();
        assert_eq!(
            source.params.decimal_separator.as_deref() == Some(","),
            comma
        );
        assert_eq!(
            plan.params["Mcal.csv"].get("decimal_separator").is_some(),
            comma
        );
        assert_eq!(
            plan.warnings
                .iter()
                .any(|text| text.contains("not shared with features or targets")),
            warning
        );
        assert!(plan.identity.is_none());
        let sources = inputs
            .iter()
            .map(|input| InMemorySource {
                name: input.name.clone(),
                payload: SourcePayload::Bytes(input.bytes.clone()),
            })
            .collect::<Vec<_>>();
        let assembled = assemble_in_memory(spec, &sources, &HashMap::new(), None).unwrap();
        assert_eq!(
            assembled.blocks["train"]
                .metadata
                .as_ref()
                .unwrap()
                .str_column(column),
            expected
        );
    }
    let inputs = a5_inputs("name;ph\n12,1;6,5\n12,10;7,5\n14,2;8,5\n");
    let plan = infer_named_bytes(&inputs, &Default::default()).unwrap();
    assert!(plan.warnings.iter().any(|text|text=="partition train: metadata label 'name' in source 'train_m' also reads as numbers under decimal ','; source-wide decimals can rewrite or merge its values; declare decimal_separator '.' to keep comma-form values as text in this source"));
    assert_eq!(plan.params["Mcal.csv"]["decimal_separator"]["value"], ",");
    let sources = inputs
        .iter()
        .map(|input| InMemorySource {
            name: input.name.clone(),
            payload: SourcePayload::Bytes(input.bytes.clone()),
        })
        .collect::<Vec<_>>();
    let assembled = assemble_in_memory(
        plan.resolved_spec.as_ref().unwrap(),
        &sources,
        &HashMap::new(),
        None,
    )
    .unwrap();
    assert_eq!(
        assembled.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .numeric_column_f64("ph"),
        [6.5, 7.5, 8.5]
    );
    // Deliberately no golden assertion endorses merged label values.
}
