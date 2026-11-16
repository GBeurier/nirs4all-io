//! Native facade for the same declared multimodal import used by WASM.
use std::collections::BTreeMap;

use nirs4all_io_core::materialize::loaders::TabularReadLimits;
use nirs4all_io_core::multimodal_import::{
    assemble_multimodal_files, multimodal_decode_requests, DecodedAsset, ImportManifest,
    ImportedDataset,
};
use nirs4all_io_core::SpecError;

/// Import supplied file bytes; callers retain filesystem and resource policy.
pub fn load_multimodal_files(
    manifest: &ImportManifest,
    files: &BTreeMap<String, Vec<u8>>,
    limits: TabularReadLimits,
    max_decoded_bytes: usize,
) -> Result<ImportedDataset, SpecError> {
    let requests = multimodal_decode_requests(manifest, files, limits)?;
    let mut decoded = BTreeMap::<String, DecodedAsset>::new();
    let mut decoded_total = 0_usize;
    for (name, profile) in requests {
        let bytes = files
            .get(&name)
            .ok_or_else(|| SpecError::new(format!("declared file '{name}' is missing")))?;
        let asset = nirs4all_formats::decode_multimodal_bytes(&profile, bytes, max_decoded_bytes)
            .map_err(SpecError::new)?;
        let neutral: DecodedAsset = serde_json::to_value(asset)
            .and_then(serde_json::from_value)
            .map_err(|error| SpecError::new(error.to_string()))?;
        decoded_total = neutral
            .decoded_bytes()
            .and_then(|bytes| decoded_total.checked_add(bytes))
            .ok_or_else(|| SpecError::new("aggregate decoded payload size overflow"))?;
        if decoded_total > max_decoded_bytes {
            return Err(SpecError::new(
                "aggregate decoded payloads exceed host byte budget",
            ));
        }
        decoded.insert(name, neutral);
    }
    assemble_multimodal_files(manifest, files, &decoded, limits)
}
