// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! wasm-bindgen binding for `nirs4all-io` (EPIC 11.4).
//!
//! WASM has no filesystem (D-R7), so this binding exposes the pure, fs-free
//! JSON surface backed by `nirs4all-io-core`: normalize a spec/config dict into
//! the canonical `DatasetSpec` (`to_spec`), validate a spec (`validate`), and
//! infer a dataset plan from named in-memory files (`inferFiles`). Strings cross
//! as canonical JSON where that is the cross-binding contract; `inferFiles`
//! returns a plain JS object for browser UIs.

use nirs4all_io_core::canonical_json;
use nirs4all_io_core::infer::memory::{
    describe_tables as describe_named_tables, infer_browser_dataset, infer_decoded_records,
    infer_named_bytes, DecodedRecordSet, DescribeOptions, NamedBytes,
};
use nirs4all_io_core::infer::propose::{propose_browser_dataset, ConfirmedLock};
use nirs4all_io_core::infer::InferOptions;
use nirs4all_io_core::materialize::{
    assemble_in_memory, AssembledDataset, InMemorySource, SourcePayload,
};
use nirs4all_io_core::spec::{normalize_to_spec_dict, validate_spec, DatasetSpec};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct WasmFile {
    name: String,
    /// One copy of the `Uint8Array` (a generic sequence would cross the
    /// JS/WASM boundary once per byte).
    #[serde(with = "serde_bytes")]
    bytes: Vec<u8>,
}

#[derive(Deserialize)]
struct WasmRecordSet {
    source: String,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    records: Vec<Value>,
}

#[derive(Deserialize)]
struct WasmConfirmedLock {
    kind: String,
    target: String,
    #[serde(default)]
    value: Value,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Deserialize, Default)]
struct ProposeOptions {
    #[serde(flatten)]
    infer: InferOptions,
    #[serde(default)]
    confirmed: Vec<WasmConfirmedLock>,
}

/// Decoded record sets read through one `JSON.stringify` and a native parse,
/// instead of one boundary call per value (thousands of spectra hold millions).
/// The replacer keeps what the per-value conversion produced: an `undefined`
/// field is `null`, and a `BigInt` (u64 fields such as `row_index`) is its
/// exact integer (`JSON.rawJSON`, or `Number` where that is unavailable, exact
/// below 2^53); non-finite numbers become `null` in both.
fn record_sets_from_js(record_sets: &JsValue) -> Result<Vec<WasmRecordSet>, JsError> {
    let replacer = js_sys::Function::new_with_args(
        "key, value",
        "if (value === undefined) return null; \
         if (typeof value === 'bigint') return JSON.rawJSON ? JSON.rawJSON(value.toString()) : Number(value); \
         return value;",
    );
    let text = js_sys::JSON::stringify_with_replacer(record_sets, &replacer)
        .map_err(|_| JsError::new("recordSets must be an array: not serializable"))?
        .as_string()
        .ok_or_else(|| JsError::new("recordSets must be an array"))?;
    canonical_json::parse_json_value(&text)
        .and_then(serde_json::from_value)
        .map_err(|e| JsError::new(&format!("recordSets must be an array: {e}")))
}

/// A JSON value as a plain JS object: one `JSON.parse` of its text instead of
/// one boundary call per value.
fn to_js(value: &Value) -> Result<JsValue, JsError> {
    let text = serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))?;
    js_sys::JSON::parse(&text).map_err(|_| JsError::new("JSON.parse of a serialized value failed"))
}

/// Normalize a spec/config JSON string into the canonical `DatasetSpec` JSON.
#[wasm_bindgen]
pub fn to_spec(spec_json: &str) -> Result<String, JsError> {
    let value: Value = canonical_json::parse_json_value(spec_json)
        .map_err(|e| JsError::new(&format!("input JSON: {e}")))?;
    let spec = DatasetSpec::from_value(&normalize_to_spec_dict(&value))
        .map_err(|e| JsError::new(&e.message))?;
    canonical_json(&spec.to_value()).map_err(|e| JsError::new(&e.to_string()))
}

/// Validate a `DatasetSpec` JSON string; throws (rejects) when invalid.
#[wasm_bindgen]
pub fn validate(spec_json: &str) -> Result<(), JsError> {
    let value: Value = canonical_json::parse_json_value(spec_json)
        .map_err(|e| JsError::new(&format!("input JSON: {e}")))?;
    let spec = DatasetSpec::from_value(&value).map_err(|e| JsError::new(&e.message))?;
    validate_spec(&spec).map_err(|e| JsError::new(&e.message))?;
    Ok(())
}

/// Infer a dataset plan from browser-provided files.
///
/// `files` must be an array of `{name, bytes}` where `bytes` is a
/// `Uint8Array`. `options.conventions` can override the default
/// `nirs4all-classic` convention list.
/// `options.params` and exact-name `options.file_params` apply before inference reads.
#[wasm_bindgen(js_name = inferFiles)]
pub fn infer_files(files: JsValue, options: JsValue) -> Result<JsValue, JsError> {
    let files: Vec<WasmFile> = serde_wasm_bindgen::from_value(files)
        .map_err(|e| JsError::new(&format!("files must be an array of {{name, bytes}}: {e}")))?;
    let options = parse_options(options)?;
    let named = files
        .into_iter()
        .map(|f| NamedBytes {
            name: f.name,
            bytes: f.bytes,
        })
        .collect::<Vec<_>>();
    let plan = infer_named_bytes(&named, &options).map_err(|e| JsError::new(&e.message))?;
    to_js(&plan.to_value())
}

/// Inspect each original table independently, retaining NA and assembler column names.
/// Per-file read refusals are returned in the descriptions; invalid arguments throw.
#[wasm_bindgen(js_name = describeTables)]
pub fn describe_tables(files: JsValue, options: JsValue) -> Result<JsValue, JsError> {
    let files: Vec<WasmFile> = serde_wasm_bindgen::from_value(files)
        .map_err(|e| JsError::new(&format!("files must be an array of {{name, bytes}}: {e}")))?;
    let options: DescribeOptions = if options.is_null() || options.is_undefined() {
        DescribeOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid describe options: {e}")))?
    };
    let named = files
        .into_iter()
        .map(|file| NamedBytes {
            name: file.name,
            bytes: file.bytes,
        })
        .collect::<Vec<_>>();
    let descriptions = describe_named_tables(&named, &options);
    let value = serde_json::to_value(descriptions).map_err(|e| JsError::new(&e.to_string()))?;
    to_js(&value)
}

/// Infer a browser dataset plan from raw files and decoded spectral records.
///
/// `files` must be an array of `{name, bytes}`. `recordSets` must be an array
/// of `{source, format?, records}` where records follow `nirs4all-formats`.
#[wasm_bindgen(js_name = inferDataset)]
pub fn infer_dataset(
    files: JsValue,
    record_sets: JsValue,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let files: Vec<WasmFile> = serde_wasm_bindgen::from_value(files)
        .map_err(|e| JsError::new(&format!("files must be an array of {{name, bytes}}: {e}")))?;
    let record_sets = record_sets_from_js(&record_sets)?;
    let options = parse_options(options)?;
    let named = files
        .into_iter()
        .map(|f| NamedBytes {
            name: f.name,
            bytes: f.bytes,
        })
        .collect::<Vec<_>>();
    let sets = record_sets
        .into_iter()
        .map(|set| DecodedRecordSet {
            source: set.source,
            format: set.format,
            records: set.records,
        })
        .collect::<Vec<_>>();
    let plan =
        infer_browser_dataset(&named, &sets, &options).map_err(|e| JsError::new(&e.message))?;
    to_js(&plan.to_value())
}

/// Iterative, user-validatable dataset inference.
///
/// Inputs mirror [`infer_dataset`] plus `options.confirmed` — an array of
/// `{kind, target, value, status?}` locks the user has accepted/overridden,
/// fed back so re-inference honours them. Returns a plain JS object
/// `{plan, proposals, spec, valid, validation_errors}` where `proposals` is the
/// list of still-open (and echoed confirmed) decisions to validate.
#[wasm_bindgen(js_name = proposeDataset)]
pub fn propose_dataset(
    files: JsValue,
    record_sets: JsValue,
    options: JsValue,
) -> Result<JsValue, JsError> {
    let files: Vec<WasmFile> = serde_wasm_bindgen::from_value(files)
        .map_err(|e| JsError::new(&format!("files must be an array of {{name, bytes}}: {e}")))?;
    let record_sets = record_sets_from_js(&record_sets)?;
    let opts: ProposeOptions = if options.is_null() || options.is_undefined() {
        ProposeOptions::default()
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid propose options: {e}")))?
    };
    let named = files
        .into_iter()
        .map(|f| NamedBytes {
            name: f.name,
            bytes: f.bytes,
        })
        .collect::<Vec<_>>();
    let sets = record_sets
        .into_iter()
        .map(|set| DecodedRecordSet {
            source: set.source,
            format: set.format,
            records: set.records,
        })
        .collect::<Vec<_>>();
    let confirmed = opts
        .confirmed
        .into_iter()
        .map(|l| ConfirmedLock {
            kind: l.kind,
            target: l.target,
            value: l.value,
            status: l.status,
        })
        .collect::<Vec<_>>();
    let result = propose_browser_dataset(&named, &sets, &confirmed, &opts.infer)
        .map_err(|e| JsError::new(&e.message))?;
    to_js(&result.to_value())
}

/// Infer a dataset plan from decoded spectral records.
///
/// `recordSets` must be an array of `{source, format?, records}` where
/// `records` follows the JSON shape emitted by `nirs4all-formats`.
#[wasm_bindgen(js_name = inferRecords)]
pub fn infer_records(record_sets: JsValue) -> Result<JsValue, JsError> {
    let record_sets = record_sets_from_js(&record_sets)?;
    let sets = record_sets
        .into_iter()
        .map(|set| DecodedRecordSet {
            source: set.source,
            format: set.format,
            records: set.records,
        })
        .collect::<Vec<_>>();
    let plan = infer_decoded_records(&sets).map_err(|e| JsError::new(&e.message))?;
    to_js(&plan.to_value())
}

/// Assemble a `DatasetSpec` into a materialized, target-agnostic dataset, fs-free.
///
/// `files` = array of `{name, bytes}` — CSV/tabular bytes the formats layer won't
/// decode (y / metadata tables). `recordSets` = array of `{source, format?,
/// records}` — pre-decoded `nirs4all-formats` records (signals → X, targets,
/// metadata). `spec` = a `DatasetSpec` JSON string (typically `plan.resolved_spec`);
/// its source `input` strings must match the `name`/`source` of the supplied
/// sources (exact, then file-stem, then glob). Returns `AssembledDataset` as a
/// plain JS object (per-partition `blocks` with `x`/`y`/headers/metadata + `folds`).
///
/// The owner full value carries exact assembled3 bytes and explicit masks.
/// The existing typed-array view is retained; its DTO/mask interoperability
/// requires separate stage4 qualification beyond this Rust type propagation.
///
/// `index_file`/`folds.file` partitions are not browser-reachable here (no fs) and
/// raise an error — column/inline partitions and folds work.
#[wasm_bindgen(js_name = assembleDataset)]
pub fn assemble_dataset(
    files: JsValue,
    record_sets: JsValue,
    spec: &str,
) -> Result<JsValue, JsError> {
    let assembled = assemble_from_js(files, record_sets, spec)?;
    let mut full: Value = assembled
        .to_full_value()
        .map_err(|error| JsError::new(&error))?;
    // The numbers cross as typed arrays below, not as JSON text.
    for block in full["blocks"]
        .as_object_mut()
        .into_iter()
        .flat_map(|b| b.values_mut())
    {
        for matrix in block["x"].as_array_mut().into_iter().flatten() {
            matrix["data"] = Value::Null;
        }
        if block["y"].is_object() {
            block["y"]["data"] = Value::Null;
        }
        for source in block["processings"].as_array_mut().into_iter().flatten() {
            for processing in source.as_array_mut().into_iter().flatten() {
                processing["matrix"]["data"] = Value::Null;
            }
        }
        if block["weights"].is_array() {
            block["weights"] = Value::Null;
        }
    }
    let js = to_js(&full)?;
    let js_blocks = get(&js, "blocks")?;
    for (part, block) in &assembled.blocks {
        let js_block = get(&js_blocks, part)?;
        let js_x = get(&js_block, "x")?;
        for (index, matrix) in block.x.iter().enumerate() {
            set_data(
                &js_sys::Reflect::get_u32(&js_x, index as u32).map_err(js_err)?,
                &matrix.data,
            )?;
        }
        if let Some(y) = &block.y {
            set_data(&get(&js_block, "y")?, &y.data)?;
        }
        let js_processings = get(&js_block, "processings")?;
        for (source, processings) in block.processings.iter().enumerate() {
            let js_source =
                js_sys::Reflect::get_u32(&js_processings, source as u32).map_err(js_err)?;
            for (index, (_, matrix)) in processings.iter().enumerate() {
                let js_processing =
                    js_sys::Reflect::get_u32(&js_source, index as u32).map_err(js_err)?;
                set_data(&get(&js_processing, "matrix")?, &matrix.data)?;
            }
        }
        if let Some(weights) = &block.weights {
            js_sys::Reflect::set(&js_block, &"weights".into(), &float64_array(weights))
                .map_err(js_err)?;
        }
    }
    Ok(js)
}

fn js_err(_: JsValue) -> JsError {
    JsError::new("could not build the assembled dataset object")
}

fn get(object: &JsValue, key: &str) -> Result<JsValue, JsError> {
    js_sys::Reflect::get(object, &key.into()).map_err(js_err)
}

fn float64_array(values: &[f64]) -> js_sys::Float64Array {
    js_sys::Float64Array::from(values)
}

fn set_data(matrix: &JsValue, data: &[f64]) -> Result<(), JsError> {
    js_sys::Reflect::set(matrix, &"data".into(), &float64_array(data)).map_err(js_err)?;
    Ok(())
}

/// Materialize the same fs-free inputs as [`assemble_dataset`] and return the
/// canonical structural summary used by the native CLI and C ABI.
///
/// Keeping this as canonical JSON (rather than a re-serialized JS object) makes
/// the cross-language qualification byte-exact and preserves source, sample,
/// observation, repetition, group, and fold provenance without host-side logic.
#[wasm_bindgen(js_name = loadSummary)]
pub fn load_summary(files: JsValue, record_sets: JsValue, spec: &str) -> Result<String, JsError> {
    let assembled = assemble_from_js(files, record_sets, spec)?;
    canonical_json(&assembled.to_summary_value()).map_err(|e| JsError::new(&e.to_string()))
}

fn assemble_from_js(
    files: JsValue,
    record_sets: JsValue,
    spec: &str,
) -> Result<AssembledDataset, JsError> {
    let files: Vec<WasmFile> = serde_wasm_bindgen::from_value(files)
        .map_err(|e| JsError::new(&format!("files must be an array of {{name, bytes}}: {e}")))?;
    let record_sets = record_sets_from_js(&record_sets)?;
    let value: Value = canonical_json::parse_json_value(spec)
        .map_err(|e| JsError::new(&format!("spec JSON: {e}")))?;
    let spec = DatasetSpec::from_value(&value).map_err(|e| JsError::new(&e.message))?;

    let mut sources: Vec<InMemorySource> = Vec::with_capacity(files.len() + record_sets.len());
    for f in files {
        sources.push(InMemorySource {
            name: f.name,
            payload: SourcePayload::Bytes(f.bytes),
        });
    }
    for set in record_sets {
        sources.push(InMemorySource {
            name: set.source,
            payload: SourcePayload::Records(set.records),
        });
    }

    assemble_in_memory(&spec, &sources, &HashMap::new(), None).map_err(|e| JsError::new(&e.message))
}

/// The wire-contract (crate) version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn parse_options(options: JsValue) -> Result<InferOptions, JsError> {
    if options.is_null() || options.is_undefined() {
        Ok(InferOptions::default())
    } else {
        serde_wasm_bindgen::from_value(options)
            .map_err(|e| JsError::new(&format!("invalid infer options: {e}")))
    }
}
