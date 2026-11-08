// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Literal Q-CSV vectors shared with the Python oracle; native execution is a separate gate.
use nirs4all_io::api::{to_spec, Input};
use nirs4all_io::core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::assemble;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;

fn vectors() -> Value {
    serde_json::from_str(include_str!("fixtures/delimiter_resolution_vectors.json")).unwrap()
}

#[test]
fn public_delimiter_vectors_preserve_values_masks_targets_and_refusals() {
    for case in vectors()["rows"].as_array().unwrap() {
        if case["operation"] != "load-assembled" {
            continue;
        }
        let tmp = tempfile::tempdir().unwrap();
        for (name, text) in case["files"].as_object().unwrap() {
            std::fs::write(tmp.path().join(name), text.as_str().unwrap()).unwrap();
        }
        let route = case["route"].as_str().unwrap();
        let mut value = case.get("spec").cloned().unwrap_or_else(
            || json!({"sources":[{"id":"train_x","role":"features","input":"Xcal.csv"}]}),
        );
        if let Some(params) = case.get("params") {
            value["params"] = params.clone();
        }
        if route == "variation" {
            let text = case["files"]["Xcal.csv"].as_str().unwrap();
            std::fs::write(tmp.path().join("alternate.csv"), text).unwrap();
            std::fs::write(tmp.path().join("Xcal.csv"), text.replace(',', ";")).unwrap();
            value["sources"][0]["variations"] =
                json!([{"name":"alternate","input":"alternate.csv"}]);
        }
        if route == "gzip" || route == "zip" {
            let text = case["files"]["Xcal.csv"].as_str().unwrap();
            if route == "gzip" {
                let mut writer =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                writer.write_all(text.as_bytes()).unwrap();
                std::fs::write(tmp.path().join("Xcal.csv.gz"), writer.finish().unwrap()).unwrap();
                value["sources"][0]["input"] = json!("Xcal.csv.gz");
            } else {
                let file = std::fs::File::create(tmp.path().join("Xcal.csv.zip")).unwrap();
                let mut writer = zip::ZipWriter::new(file);
                writer
                    .start_file("Xcal.csv", zip::write::SimpleFileOptions::default())
                    .unwrap();
                writer.write_all(text.as_bytes()).unwrap();
                writer.finish().unwrap();
                value["sources"][0]["input"] = json!("Xcal.csv.zip");
            }
        }
        let dir = tmp.path().to_string_lossy().into_owned();
        let (spec, base) = match route {
            "dir" => to_spec(&Input::Path(dir), None, None).unwrap(),
            "list" => {
                let files = case["files"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(|n| tmp.path().join(n).to_string_lossy().into_owned())
                    .collect();
                to_spec(&Input::Paths(files), None, None).unwrap()
            }
            "file" => to_spec(
                &Input::Path(tmp.path().join("data.csv").to_string_lossy().into_owned()),
                None,
                None,
            )
            .unwrap(),
            "legacy" => (
                to_spec(&Input::Spec(json!({"train_x":"Xcal.csv"})), None, None)
                    .unwrap()
                    .0,
                tmp.path().to_path_buf(),
            ),
            "config" => {
                let path = tmp.path().join("config.json");
                std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
                to_spec(
                    &Input::Path(path.to_string_lossy().into_owned()),
                    None,
                    None,
                )
                .unwrap()
            }
            _ => (
                DatasetSpec::from_value(&value).unwrap(),
                tmp.path().to_path_buf(),
            ),
        };
        let result = assemble(&spec, &base);
        let expected = &case["expected"];
        if let Some(refusal) = expected["refuse"].as_str() {
            let error = result.unwrap_err().message;
            assert!(error.contains(refusal), "{}: {error}", case["id"]);
            assert!(
                error.contains(expected["file"].as_str().unwrap()),
                "{}: {error}",
                case["id"]
            );
            assert!(error.contains(expected["source"].as_str().unwrap_or("train_x")));
            continue;
        }
        let assembled = result.unwrap();
        let block = &assembled.blocks["train"];
        let rows = expected["x"].as_array().unwrap();
        let width = expected["headers"].as_array().unwrap().len();
        assert_eq!(
            (block.x[0].n_rows, block.x[0].n_cols),
            (rows.len(), width),
            "{}",
            case["id"]
        );
        for (actual, expected) in block.x[0]
            .data
            .iter()
            .zip(rows.iter().flat_map(|r| r.as_array().unwrap()))
        {
            if expected.is_null() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(
                    actual.to_bits(),
                    expected.as_f64().unwrap().to_bits(),
                    "{}",
                    case["id"]
                );
            }
        }
        assert_eq!(json!(block.feature_headers[0]), expected["headers"]);
        if let Some(y) = expected["y"].as_array() {
            let values: Vec<f64> = y
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .collect();
            assert_eq!(block.y.as_ref().unwrap().data, values);
        }
        if let Some(y_headers) = expected.get("y_headers") {
            assert_eq!(json!(block.y_headers), *y_headers);
        }
        if matches!(route, "dir" | "list" | "file") {
            assert_eq!(
                json!(spec.sources[0].params.delimiter),
                expected.get("delimiter").cloned().unwrap_or(Value::Null)
            );
            if spec.sources.len() > 1 {
                assert!(spec.sources[1].params.delimiter.is_none());
            }
        }
    }
}

#[test]
fn shared_refusal_vectors_reach_named_bytes_without_a_filesystem_adapter() {
    for case in vectors()["rows"].as_array().unwrap() {
        if case["operation"] != "assemble-in-memory" {
            continue;
        }
        let spec = DatasetSpec::from_value(
            &json!({"sources":[{"id":"train_x","input":"Xcal.csv","role":"features"}]}),
        )
        .unwrap();
        let sources = [InMemorySource {
            name: "Xcal.csv".into(),
            payload: SourcePayload::Bytes(
                case["files"]["Xcal.csv"]
                    .as_str()
                    .unwrap()
                    .as_bytes()
                    .to_vec(),
            ),
        }];
        assert!(assemble_in_memory(&spec, &sources, &HashMap::new(), None)
            .unwrap_err()
            .message
            .contains("declare params.delimiter"));
    }
}

#[test]
fn raw_reader_keeps_summary_contract_separate_from_assembly_guard() {
    use nirs4all_io::core::spec::dataset_spec::LoadingParams;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("raw.csv");
    std::fs::write(&path, "x,y\n1,2\n3,4\n").unwrap();
    let frame =
        nirs4all_io::materialize::loaders::read_table(&path, &LoadingParams::default()).unwrap();
    assert_eq!(frame.columns.len(), 1);
    assert_eq!(frame.n_rows, 2);
}
