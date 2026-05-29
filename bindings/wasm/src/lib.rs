// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! wasm-bindgen binding for `nirs4all-io` (EPIC 11.4).
//!
//! WASM has no filesystem (D-R7), so this binding exposes only the pure,
//! fs-free JSON surface backed by `nirs4all-io-core`: normalize a spec/config
//! dict into the canonical `DatasetSpec` (`to_spec`) and validate a spec
//! (`validate`). Path-based `infer`/`load` need file IO and stay in the native
//! facade. Strings cross as canonical JSON, identical to every other binding.

use nirs4all_io_core::canonical_json;
use nirs4all_io_core::spec::{normalize_to_spec_dict, validate_spec, DatasetSpec};
use serde_json::Value;
use wasm_bindgen::prelude::*;

/// Normalize a spec/config JSON string into the canonical `DatasetSpec` JSON.
#[wasm_bindgen]
pub fn to_spec(spec_json: &str) -> Result<String, JsError> {
    let value: Value =
        serde_json::from_str(spec_json).map_err(|e| JsError::new(&format!("input JSON: {e}")))?;
    let spec = DatasetSpec::from_value(&normalize_to_spec_dict(&value))
        .map_err(|e| JsError::new(&e.message))?;
    canonical_json(&spec.to_value()).map_err(|e| JsError::new(&e.to_string()))
}

/// Validate a `DatasetSpec` JSON string; throws (rejects) when invalid.
#[wasm_bindgen]
pub fn validate(spec_json: &str) -> Result<(), JsError> {
    let value: Value =
        serde_json::from_str(spec_json).map_err(|e| JsError::new(&format!("input JSON: {e}")))?;
    let spec = DatasetSpec::from_value(&value).map_err(|e| JsError::new(&e.message))?;
    validate_spec(&spec).map_err(|e| JsError::new(&e.message))?;
    Ok(())
}

/// The wire-contract (crate) version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
