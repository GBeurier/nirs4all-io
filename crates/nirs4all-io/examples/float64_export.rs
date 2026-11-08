// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Owner qualification exporter. Input is one JSON object on stdin.
use nirs4all_io::core::materialize::{
    package_in_memory_with_storage_and_tabular_limits, InMemorySource, NumericStorageOptions,
    SourcePayload,
};
use nirs4all_io::core::public_dataset::masked_matrix_dataset_package_with_storage;
use nirs4all_io::core::spec::DatasetSpec;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn export(input: &Value) -> Result<Value, String> {
    if input["operation"] == "dataset_content" {
        let value = &input["public_dataset"];
        return Ok(json!({
            "dataset_content_identity": nirs4all_io::core::public_dataset::labelled_dataset_content_identity(value)?,
            "dataset_content_bytes_hex": hex(&nirs4all_io::core::public_dataset::labelled_dataset_content_bytes(value)?)
        }));
    }
    let options = if input["storage"].is_null() {
        NumericStorageOptions::default()
    } else {
        serde_json::from_value(input["storage"].clone()).map_err(|e| e.to_string())?
    };
    let package = if !input["public_dataset"].is_null() {
        masked_matrix_dataset_package_with_storage(
            &input["public_dataset"],
            input["source_id"].as_str().ok_or("source_id required")?,
            &options,
        )?
        .0
    } else {
        let spec = DatasetSpec::from_value(&input["spec"]).map_err(|e| e.message)?;
        if input["sources"].is_null() {
            nirs4all_io::materialize::package_with_limits_and_storage(
                &spec,
                Path::new(input["base_dir"].as_str().ok_or("base_dir required")?),
                nirs4all_io::materialize::LoadLimits::default(),
                &options,
            )
            .map_err(|e| e.message)?
        } else {
            let sources = input["sources"]
                .as_array()
                .ok_or("sources array required")?
                .iter()
                .map(|source| {
                    let payload = if let Some(csv) = source["csv"].as_str() {
                        SourcePayload::Bytes(csv.as_bytes().to_vec())
                    } else {
                        SourcePayload::Records(
                            source["records"]
                                .as_array()
                                .ok_or("csv or records required")?
                                .clone(),
                        )
                    };
                    Ok(InMemorySource {
                        name: source["name"]
                            .as_str()
                            .ok_or("source name required")?
                            .into(),
                        payload,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            package_in_memory_with_storage_and_tabular_limits(
                &spec,
                &sources,
                &HashMap::new(),
                None,
                None,
                &options,
            )
            .map_err(|e| e.message)?
        }
    };
    let selected = if let Some(sources) = input["selected_source_ids"].as_array() {
        sources
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or("Selected source string required".into())
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        package.identity.source_ids.clone()
    };
    let (receipt, features, targets, error) = match package.executed_projection(&selected) {
        Ok((receipt, features, targets)) => (
            Some(receipt),
            Some(hex(&features)),
            Some(hex(&targets)),
            None,
        ),
        Err(error) => (None, None, None, Some(error)),
    };
    let assembled_full = package.to_assembled().to_full_value()?;
    let package_full = package.to_full_value()?;
    let assembled_full_canonical =
        nirs4all_io::core::canonical_json::canonical_json(&assembled_full)
            .map_err(|e| e.to_string())?;
    let package_full_canonical = nirs4all_io::core::canonical_json::canonical_json(&package_full)
        .map_err(|e| e.to_string())?;
    Ok(
        json!({"assembled_full":assembled_full,"package_full":package_full,
        "assembled_full_canonical":assembled_full_canonical,"package_full_canonical":package_full_canonical,
        "assembled_summary":package.to_assembled().to_summary_value(),"package_summary":package.to_summary_value(),
        "executed_projection":receipt,"feature_preimage_hex":features,"target_preimage_hex":targets,"projection_error":error}),
    )
}
fn main() -> Result<(), String> {
    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    let input =
        nirs4all_io::core::canonical_json::parse_json_value(&text).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string(&export(&input)?).map_err(|e| e.to_string())?
    );
    Ok(())
}
