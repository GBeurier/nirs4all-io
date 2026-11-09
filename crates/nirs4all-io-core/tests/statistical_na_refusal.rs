// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Native spec refusal and supported controls for the literal held-out probe.
use nirs4all_io_core::materialize::{
    assemble_in_memory, AssembledDataset, InMemorySource, SourcePayload,
};
use nirs4all_io_core::spec::enums::FillMethod;
use nirs4all_io_core::spec::{role_tagged_config_to_spec, DatasetSpec, SpecError};
use serde_json::{json, Value};

const A: &str = "Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;10.0;3.0\n";
const B: &str = "Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;30.0;3.0\n";

fn probe_spec(na: Value) -> Value {
    json!({
        "schema_version": 1,
        "name": "statistical-na-mean-heldout-probe",
        "task_type": "regression",
        "sample_index": {"by": "id", "key": "Sample_ID"},
        "params": {"delimiter": ";", "decimal": ".", "has_header": true, "na": na},
        "sources": [{
            "id": "x", "role": "mixed", "input": "literal.csv", "key": "Sample_ID",
            "params": {"header_unit": "nm"},
            "columns": {"features": ["1100"], "targets": ["target"], "metadata": ["set"]}
        }],
        "partitions": {
            "by": "column", "column": "set", "train_values": ["train"],
            "test_values": ["test"], "unknown_policy": "error"
        }
    })
}

fn assemble(value: &Value, csv: &str) -> Result<AssembledDataset, SpecError> {
    let spec = DatasetSpec::from_value(value)?;
    assemble_in_memory(
        &spec,
        &[InMemorySource {
            name: "literal.csv".into(),
            payload: SourcePayload::Bytes(csv.as_bytes().to_vec()),
        }],
        &Default::default(),
        None,
    )
}

fn assert_refusal(error: SpecError) {
    assert!(error.message.contains("na.fill.method:"), "{error}");
    assert!(error.message.contains("is refused"), "{error}");
    assert!(error.message.contains("fitted preprocessing"), "{error}");
    assert!(error.message.contains("within each sample"), "{error}");
}

#[test]
fn explicit_statistics_refuse_for_global_and_source_params_under_every_policy() {
    for method in ["mean", "median", " Median ", "MEAN"] {
        for per_column in [true, false] {
            for policy in [
                "auto",
                "abort",
                "ignore",
                "remove_sample",
                "remove_feature",
                "replace",
            ] {
                for source_override in [false, true] {
                    let na = json!({"policy": policy, "fill": {
                        "method": method, "per_column": per_column
                    }});
                    let mut value = probe_spec(json!({"policy": "ignore"}));
                    if source_override {
                        value["sources"][0]["params"]["na"] = na;
                    } else {
                        value["params"]["na"] = na;
                    }
                    assert_refusal(DatasetSpec::from_value(&value).unwrap_err());
                }
            }
        }
    }
}

#[test]
fn probe_assembly_request_refuses_before_missingness_is_considered() {
    for method in ["mean", "median"] {
        let value = probe_spec(json!({"policy": "replace", "fill": {"method": method}}));
        assert_refusal(assemble(&value, A).unwrap_err());
        let complete = A.replace("S_missing;train;;", "S_missing;train;1.0;");
        assert_refusal(assemble(&value, &complete).unwrap_err());
    }
}

#[test]
fn role_tagged_request_reaches_the_same_owner_refusal() {
    let value = json!({
        "files": [
            {"path": "/data/X.csv", "type": "X", "split": "train", "source": 0},
            {"path": "/data/Y.csv", "type": "Y", "split": "train", "source": null}
        ],
        "na_policy": "replace", "na_fill_config": {"method": "forward_fill"}
    });
    let control = role_tagged_config_to_spec(&value).unwrap().to_value();
    assert_eq!(control["params"]["na"]["fill"]["method"], "forward_fill");

    for method in ["mean", "median"] {
        // Only the method differs from the owner-valid control above.
        let mut statistical = value.clone();
        statistical["na_fill_config"]["method"] = json!(method);
        let error = role_tagged_config_to_spec(&statistical).unwrap_err();
        assert!(
            error
                .message
                .starts_with(&format!("na.fill.method: '{method}' is refused:")),
            "{error}"
        );
        assert_refusal(error);
    }
}

#[test]
fn accepted_fill_vocabulary_and_serialization_stay_canonical() {
    assert_eq!(
        FillMethod::VALUES,
        &["value", "forward_fill", "backward_fill"]
    );
    for (method, canonical) in [
        ("value", "value"),
        (" Forward-Fill ", "forward_fill"),
        ("BACKWARD_FILL", "backward_fill"),
    ] {
        for per_column in [true, false] {
            let value = probe_spec(json!({"policy": "replace", "fill": {
                "method": method, "fill_value": 0, "per_column": per_column
            }}));
            let parsed = DatasetSpec::from_value(&value).unwrap();
            let mut expected = json!({"policy": "replace", "fill": {
                "method": canonical, "fill_value": 0
            }});
            if per_column {
                expected["fill"]["per_column"] = json!(true);
            }
            assert_eq!(parsed.to_value()["params"]["na"], expected);
        }
    }
}

#[test]
fn supported_probe_controls_preserve_train_bytes_masks_identity_and_partitions() {
    let precision = json!({
        "dtype": "float64", "origin": "default", "source_dtype": null,
        "requested_dtype": null, "consumer": null
    });
    for (label, na) in [
        ("ignore", json!({"policy": "ignore"})),
        (
            "value",
            json!({"policy": "replace", "fill": {"method": "value", "fill_value": 0}}),
        ),
        ("remove_sample", json!({"policy": "remove_sample"})),
        (
            "forward_fill",
            json!({"policy": "replace", "fill": {"method": "forward_fill"}}),
        ),
        (
            "backward_fill",
            json!({"policy": "replace", "fill": {"method": "backward_fill"}}),
        ),
    ] {
        let spec = probe_spec(na);
        let a = assemble(&spec, A).unwrap();
        let b = assemble(&spec, B).unwrap();
        let wa = a.to_full_value().unwrap();
        let wb = b.to_full_value().unwrap();
        assert_eq!(wa["blocks"]["train"], wb["blocks"]["train"], "{label}");
        let removed = label == "remove_sample";
        let ids = if removed {
            vec!["S_finite"]
        } else {
            vec!["S_missing", "S_finite"]
        };
        let targets = if removed { vec![2.0] } else { vec![1.0, 2.0] };
        let validity = if removed {
            vec![true]
        } else {
            vec![label == "value", true]
        };
        let train_bytes = if removed {
            "0000000000000040"
        } else {
            "00000000000000000000000000000040"
        };
        for (assembled, wire, test_value, test_bytes) in [
            (&a, &wa, 10.0, "0000000000002440"),
            (&b, &wb, 30.0, "0000000000003e40"),
        ] {
            assert_eq!(wire["partition_order"], json!(["train", "test"]));
            assert_eq!(wire["assembled_schema_version"], 3);
            assert_eq!(assembled.identity.source_ids, ["x"]);
            assert_eq!(assembled.identity.sample_id.as_deref(), Some("Sample_ID"));
            let train = &assembled.blocks["train"];
            let test = &assembled.blocks["test"];
            assert_eq!(
                train.metadata.as_ref().unwrap().str_column("Sample_ID"),
                ids
            );
            assert_eq!(
                test.metadata.as_ref().unwrap().str_column("Sample_ID"),
                ["S_test"]
            );
            assert_eq!(train.y.as_ref().unwrap().data, targets);
            assert_eq!(train.y.as_ref().unwrap().validity, vec![true; ids.len()]);
            assert_eq!(test.y.as_ref().unwrap().data, [3.0]);
            assert_eq!(test.y.as_ref().unwrap().validity, [true]);
            assert_eq!(train.n_samples, ids.len());
            assert_eq!(
                wire["blocks"]["train"]["x"][0]["data_bytes_hex"],
                train_bytes
            );
            assert_eq!(train.x[0].validity, validity);
            assert_eq!(wire["blocks"]["train"]["x"][0]["validity"], json!(validity));
            assert_eq!(train.x[0].data.last(), Some(&2.0));
            assert_eq!(test.x[0].data, [test_value]);
            assert_eq!(test.x[0].validity, [true]);
            assert_eq!(wire["blocks"]["test"]["x"][0]["data_bytes_hex"], test_bytes);
            for partition in ["train", "test"] {
                let block = &wire["blocks"][partition];
                assert_eq!(block["source_ids"], json!(["x"]));
                assert_eq!(block["feature_headers"], json!([["1100"]]));
                assert_eq!(block["header_units"], json!(["nm"]));
                assert_eq!(block["x"][0]["n_cols"], 1);
                assert_eq!(block["x"][0]["precision"], precision);
                assert_eq!(block["y"]["columns"][0]["block"]["precision"], precision);
            }
        }
        assert_eq!(a.identity, b.identity);
        assert_eq!(a.audits, b.audits);
        assert_eq!(a.fold_provenance, b.fold_provenance);
    }
}
