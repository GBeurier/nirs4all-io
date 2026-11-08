// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! `DatasetPackage` — the target-agnostic v4 package (`IO-002` / `IO-MM-002`).
//!
//! `AssembledDataset` stores its matrix payloads inline as dense `f64`
//! [`Matrix`]. That is fine for 1-D spectra but wrong for images / cubes (wrong
//! dtype, wrong rank, pathological JSON size). [`DatasetPackage`] is versioned
//! contract the `LOCK-IO` spec ratifies: a container of **typed payload blocks**
//! ([`PayloadBlock`]) plus a **payload manifest** ([`PayloadManifest`]) whose rows
//! carry a `content_hash`, a representation-ID hint and either inline bytes or a
//! large-payload **URI reference** ([`UriRef`]) — bytes are never embedded in the
//! canonical summary. Identity's **row-position fallback** is recorded explicitly
//! and fingerprinted ([`RowPositionFallback`]), not left silent.
//!
//! `DatasetPackage` v4 is distinct from the explicitly versioned
//! `AssembledDataset` v3 summary/full wire. [`DatasetPackage::from_assembled`]
//! preserves every supported assembled payload (proven by round-tripping through
//! [`DatasetPackage::to_assembled`]); URI/tensor variants remain package-only.
//! Neither wire silently accepts its retired predecessor.
//!
//! Like the rest of `nirs4all-io-core`, this module is pure logic: no file IO,
//! no `nirs4all-formats`, no `dag-ml-data` dependency. The representation-ID
//! strings mirror the published `dag-ml-data` `builtin_models.rs` registry
//! (`DMD-001`); the `nirs4all-io-dagml` bridge owns the drift guard that asserts
//! [`repr_ids`] equals the real `dag_ml_data::REPRESENTATION_*` constants.

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::assemble::{
    identity_provenance_value, AssembledDataset, FoldProvenance, IdentityProvenance, PartitionBlock,
};
use super::folds::Fold;
#[cfg(test)]
use super::frame::Cell;
use super::frame::{Frame, Matrix};
use super::numeric::{NumericStorageOptions, PrecisionRecord};
use crate::canonical_json::canonical_json;
use crate::spec::dataset_spec::AggregateSpec;

/// `DatasetPackage` wire-schema version. Distinct from
/// `DATASET_SPEC_SCHEMA_VERSION` (1): this is a separate v4 wire and does not
/// imply compatibility with older packages.
pub const DATASET_PACKAGE_VERSION: u32 = 4;

/// Frozen representation-ID strings — a verbatim mirror of the published
/// `dag-ml-data` `builtin_models.rs` registry (`DMD-001`). Kept here as plain
/// strings so the pure core stays free of a `dag-ml-data` dependency; the
/// `nirs4all-io-dagml` bridge asserts these equal the real
/// `dag_ml_data::REPRESENTATION_*` constants (drift guard).
pub mod repr_ids {
    /// `sample × wavelength` — a single 1-D signal source.
    pub const SIGNAL_1D: &str = "signal_1d";
    /// `sample × processing × wavelength` — a signal with preprocessing layers.
    pub const SIGNAL_WITH_PROCESSINGS: &str = "signal_with_processings";
    /// `sample × block × feature` — multi-source feature blocks (join output).
    pub const FEATURE_BLOCK_SET: &str = "feature_block_set";
    /// `sample` rank-1 numeric target.
    pub const TARGET_NUMERIC: &str = "target_numeric";
    /// `sample` rank-1 categorical target.
    pub const TARGET_CATEGORICAL: &str = "target_categorical";
    /// `sample × target` rank-2 numeric targets.
    pub const TARGET_NUMERIC_MATRIX: &str = "target_numeric_matrix";
    /// `sample × target` rank-2 categorical targets.
    pub const TARGET_CATEGORICAL_MATRIX: &str = "target_categorical_matrix";
    /// `sample × field` metadata table.
    pub const SAMPLE_METADATA: &str = "sample_metadata";
    /// `sample × height × width` grayscale image (net-new, declared).
    pub const GRAY_IMAGE: &str = "gray_image";
    /// `sample × height × width × channel=3` RGB image (net-new, declared).
    pub const RGB_IMAGE: &str = "rgb_image";
    /// `sample × height × width × channel` multichannel image (net-new, declared).
    pub const MC_IMAGE: &str = "mc_image";
    /// `sample × height × width × band` multispectral image (net-new, declared).
    pub const MULTISPECTRAL_IMAGE: &str = "multispectral_image";
}

/// Assemble and package with the same initial storage selection and shape limits.
pub fn package_in_memory_with_storage_and_tabular_limits(
    spec: &crate::spec::DatasetSpec,
    sources: &[super::assemble::InMemorySource],
    index_lists: &std::collections::HashMap<String, Vec<i64>>,
    fold_inline: Option<&[super::folds::Fold]>,
    limits: Option<super::loaders::TabularReadLimits>,
    options: &NumericStorageOptions,
) -> Result<DatasetPackage, crate::spec::SpecError> {
    let assembled = super::assemble::assemble_in_memory_with_storage_and_tabular_limits(
        spec,
        sources,
        index_lists,
        fold_inline,
        limits,
        options,
    )?;
    DatasetPackage::from_assembled_with_storage(&assembled, options)
        .map_err(crate::spec::SpecError::new)
}

// --------------------------------------------------------------------------- //
// Payload manifest                                                            //
// --------------------------------------------------------------------------- //

/// Where a payload's bytes live: inline in the package, or referenced by URI so
/// large image/cube bytes are never embedded in the canonical summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadStorageKind {
    /// Bytes are carried in the in-memory payload block (small numeric payloads).
    Inline,
    /// Bytes live outside the package, referenced by a [`UriRef`] (large payloads).
    Uri,
}

/// One manifest row: a bytes-free description of a single payload block. Ties the
/// payload to a representation-ID hint, its shape/dtype, a `content_hash` (SHA-256
/// of the payload's canonical bytes) and its storage mode. Tampering with any
/// field changes the manifest root hash; tampering with the payload bytes changes
/// `content_hash`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PayloadManifestEntry {
    /// Stable payload id within the package (e.g. `train/x0`, `train/y`).
    pub id: String,
    /// Owning partition name.
    pub partition: String,
    /// io role: `features` | `targets` | `metadata` | `weights` | `mask`.
    pub role: String,
    /// Payload block kind (`feature_matrix`, `target_table`, `nd_tensor`, …).
    pub payload_kind: String,
    /// `dag-ml-data` representation-ID hint (`DMD-001`), when known.
    pub representation_id: Option<String>,
    /// Element dtype (`float32`, `mixed`, or the tensor/mask dtype).
    pub dtype: String,
    /// Payload shape (`[n_samples, n_features]`, tensor shape, …).
    pub shape: Vec<usize>,
    /// Axis names, best-effort (`sample`, `wavelength`, `processing`, …).
    pub axes: Vec<String>,
    /// SHA-256 (lowercase hex) of the payload's canonical bytes.
    pub content_hash: String,
    /// Byte length of the hashed payload serialization.
    pub byte_len: u64,
    /// Inline vs URI-referenced.
    pub storage: PayloadStorageKind,
    /// The URI, when `storage == Uri`.
    pub uri: Option<String>,
    /// The codec, when `storage == Uri`.
    pub codec: Option<String>,
}

/// The package payload manifest: every payload's manifest row plus a `root`
/// fingerprint over all rows (canonical-JSON, so key order is stable). The root
/// is what a provider checks to detect tampering with payloads or the manifest.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PayloadManifest {
    /// SHA-256 (lowercase hex) over the canonical JSON of `entries`.
    pub root: String,
    /// One row per payload block, in package order.
    pub entries: Vec<PayloadManifestEntry>,
}

/// A large-payload manifest pointer: the `IO-MM-002` `UriBackedPayload` contract.
/// Carries everything a provider needs to fetch and verify external bytes without
/// embedding them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UriRef {
    /// The location of the bytes (file path, object-store key, …).
    pub uri: String,
    /// Element dtype of the referenced payload.
    pub dtype: String,
    /// Shape of the referenced payload.
    pub shape: Vec<usize>,
    /// Axis names of the referenced payload.
    pub axes: Vec<String>,
    /// SHA-256 (lowercase hex) of the referenced bytes.
    pub content_hash: String,
    /// Byte length of the referenced payload.
    pub byte_len: u64,
    /// Optional codec / compression (e.g. `npy`, `zstd`).
    pub codec: Option<String>,
}

// --------------------------------------------------------------------------- //
// Typed payload blocks                                                        //
// --------------------------------------------------------------------------- //

/// A dense numeric feature source (the current `X` path), with its headers, unit,
/// signal type and any named preprocessing variants.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    pub feature_names: Option<Vec<String>>,
    pub source_id: String,
    pub representation_id: String,
    pub axes: Vec<Value>,
    /// Native `sample × feature` matrix.
    pub matrix: Matrix,
    /// Per-feature headers (axis labels / column names).
    pub headers: Vec<String>,
    /// Header unit hint (`nm`, `cm-1`, `index`, …).
    pub header_unit: String,
    /// Signal type tag (`absorbance`, `reflectance`, …), when known.
    pub signal_type: Option<String>,
    /// Named preprocessing variants aligned to `matrix` rows.
    pub processings: Vec<(String, Matrix)>,
}

/// A target table: a `sample × target` numeric matrix plus categorical mappings.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetTable {
    pub kinds: Vec<u8>,
    /// `sample × target` (encoded) matrix.
    pub matrix: Matrix,
    /// Target column headers.
    pub headers: Vec<String>,
    /// Per-column categorical mapping (`{"categories": [...]}`), when encoded.
    pub categorical: IndexMap<String, Value>,
    /// Representation-ID hint decided at construction (`target_numeric`, …).
    pub representation_id: Option<String>,
}

/// A metadata table (typed columns, sample-aligned).
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataTable {
    /// The sample-aligned metadata frame.
    pub frame: Frame,
}

/// Per-sample training weights (sklearn-style `sample_weight`).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightsVector {
    pub validity: Vec<bool>,
    pub precision: PrecisionRecord,
    /// One weight per sample.
    pub values: Vec<f64>,
    /// The source column header, when known.
    pub header: Option<String>,
}

/// Decoded `nirs4all-formats` records carried as a payload (net-new as a package
/// payload; the input form already exists as `SourcePayload::Records`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralRecordSet {
    /// The decoded records (one JSON object per observation).
    pub records: Vec<Value>,
}

/// An N-D tensor payload (image / cube / tensor). Manifest-/URI-backed in this
/// first slice — there is no in-core image decoder yet (OQ-1), so bytes live
/// behind a [`UriRef`].
#[derive(Debug, Clone, PartialEq)]
pub struct NdTensor {
    /// `dag-ml-data` representation-ID (`rgb_image`, `gray_image`, `cube_hwb`, …).
    pub representation_id: String,
    /// Element dtype (`uint8`, `int32`, `float32`, …).
    pub dtype: String,
    /// Tensor shape, e.g. `[sample, height, width, channel]`.
    pub shape: Vec<usize>,
    /// Axis names.
    pub axis_names: Vec<String>,
    /// Observation ids (one per leading-axis element).
    pub observation_ids: Vec<String>,
    /// The bytes reference.
    pub uri: UriRef,
}

/// A time-series payload (fixed or ragged length). URI-backed (net-new).
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceBlock {
    /// `dag-ml-data` representation-ID (`series_mv`, …).
    pub representation_id: String,
    /// Element dtype.
    pub dtype: String,
    /// Number of channels.
    pub n_channels: usize,
    /// Per-observation sequence lengths (a single value ⇒ fixed length).
    pub lengths: Vec<usize>,
    /// The bytes reference.
    pub uri: UriRef,
}

/// A genotype variant/dosage matrix, descriptor-first (net-new). URI-backed; no
/// genotype-byte parsing is claimed here.
#[derive(Debug, Clone, PartialEq)]
pub struct GenotypeMatrix {
    /// `dag-ml-data` representation-ID (`variant_matrix` / `dosage_matrix`).
    pub representation_id: String,
    /// Number of variants.
    pub n_variants: usize,
    /// Number of samples.
    pub n_samples: usize,
    /// Encoding descriptor (`dosage`, `onehot`, …).
    pub encoding: String,
    /// The bytes reference.
    pub uri: UriRef,
}

/// A segmentation / ROI / label mask (net-new). URI-backed.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskBlock {
    /// `dag-ml-data` representation-ID (`segmentation_mask` / `roi_mask`).
    pub representation_id: String,
    /// Element dtype (`bool`, `uint8`, …).
    pub dtype: String,
    /// Mask shape.
    pub shape: Vec<usize>,
    /// The bytes reference.
    pub uri: UriRef,
}

/// The tagged union of typed payload variants (`IO-MM-002`). Small numeric
/// payloads are inline; large image/cube/series/genotype payloads are carried as
/// URI-backed descriptors so no bytes land in the canonical summary.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadBlock {
    /// Dense numeric feature source (inline).
    FeatureMatrix(FeatureMatrix),
    /// Target table (inline).
    TargetTable(TargetTable),
    /// Metadata table (inline).
    MetadataTable(MetadataTable),
    /// Per-sample weights (inline).
    Weights(WeightsVector),
    /// Decoded formats records (inline).
    SpectralRecordSet(SpectralRecordSet),
    /// N-D tensor (URI-backed).
    NdTensor(NdTensor),
    /// Time series (URI-backed).
    SequenceBlock(SequenceBlock),
    /// Genotype matrix (URI-backed).
    GenotypeMatrix(GenotypeMatrix),
    /// Mask (URI-backed).
    MaskBlock(MaskBlock),
    /// A bare large-payload manifest pointer (URI-backed).
    UriBackedPayload(UriRef),
}

impl PayloadBlock {
    /// Build this payload's manifest row. Inline variants hash their canonical
    /// bytes; URI-backed variants trust the [`UriRef`]'s pre-computed hash.
    pub fn manifest_entry(&self, partition: &str, id: &str) -> PayloadManifestEntry {
        let base = |role: &str, kind: &str| PayloadManifestEntry {
            id: id.to_string(),
            partition: partition.to_string(),
            role: role.to_string(),
            payload_kind: kind.to_string(),
            representation_id: None,
            dtype: String::new(),
            shape: vec![],
            axes: vec![],
            content_hash: String::new(),
            byte_len: 0,
            storage: PayloadStorageKind::Inline,
            uri: None,
            codec: None,
        };
        let inline = |mut e: PayloadManifestEntry, bytes: &[u8]| {
            e.content_hash = sha256_hex(bytes);
            e.byte_len = bytes.len() as u64;
            e.storage = PayloadStorageKind::Inline;
            e
        };
        let from_uri = |mut e: PayloadManifestEntry, uri: &UriRef| {
            e.content_hash = uri.content_hash.clone();
            e.byte_len = uri.byte_len;
            e.storage = PayloadStorageKind::Uri;
            e.uri = Some(uri.uri.clone());
            e.codec = uri.codec.clone();
            e
        };
        match self {
            PayloadBlock::FeatureMatrix(fm) => {
                let has_proc = !fm.processings.is_empty();
                let mut axes = vec!["sample".to_string()];
                if has_proc {
                    axes.push("processing".to_string());
                }
                axes.push(feature_axis_name(&fm.header_unit).to_string());
                let mut e = base("features", "feature_matrix");
                e.representation_id = Some(
                    if has_proc {
                        repr_ids::SIGNAL_WITH_PROCESSINGS
                    } else {
                        repr_ids::SIGNAL_1D
                    }
                    .to_string(),
                );
                e.dtype = fm.matrix.precision.dtype.dtype().into();
                e.shape = vec![fm.matrix.n_rows, fm.matrix.n_cols];
                e.axes = axes;
                inline(e, &feature_content_bytes(fm))
            }
            PayloadBlock::TargetTable(tt) => {
                let mut e = base("targets", "target_table");
                e.representation_id = tt.representation_id.clone();
                e.dtype = if tt.kinds.iter().all(|kind| *kind == 2) {
                    "string-label".into()
                } else if tt.kinds.contains(&2) {
                    "mixed".into()
                } else {
                    tt.matrix.precision.dtype.dtype().into()
                };
                e.shape = vec![tt.matrix.n_rows, tt.matrix.n_cols];
                e.axes = if tt.matrix.n_cols > 1 {
                    vec!["sample".into(), "target".into()]
                } else {
                    vec!["sample".into()]
                };
                inline(e, &target_content_bytes(tt))
            }
            PayloadBlock::MetadataTable(mt) => {
                let mut e = base("metadata", "metadata_table");
                e.representation_id = Some(repr_ids::SAMPLE_METADATA.to_string());
                e.dtype = "mixed".into();
                e.shape = vec![mt.frame.n_rows, mt.frame.columns.len()];
                e.axes = vec!["sample".into(), "field".into()];
                inline(e, frame_canonical_bytes(&mt.frame).as_bytes())
            }
            PayloadBlock::Weights(w) => {
                let mut e = base("weights", "weights");
                e.dtype = w.precision.dtype.dtype().into();
                e.shape = vec![w.values.len()];
                e.axes = vec!["sample".into()];
                inline(e, &weights_content_bytes(w))
            }
            PayloadBlock::SpectralRecordSet(rs) => {
                let mut e = base("features", "spectral_record_set");
                e.dtype = "records".into();
                e.shape = vec![rs.records.len()];
                e.axes = vec!["sample".into()];
                let bytes = canonical_json(&Value::Array(rs.records.clone()))
                    .expect("records serialize")
                    .into_bytes();
                inline(e, &bytes)
            }
            PayloadBlock::NdTensor(t) => {
                let mut e = base("features", "nd_tensor");
                e.representation_id = Some(t.representation_id.clone());
                e.dtype = t.dtype.clone();
                e.shape = t.shape.clone();
                e.axes = t.axis_names.clone();
                from_uri(e, &t.uri)
            }
            PayloadBlock::SequenceBlock(s) => {
                let mut e = base("features", "sequence_block");
                e.representation_id = Some(s.representation_id.clone());
                e.dtype = s.dtype.clone();
                e.shape = s.uri.shape.clone();
                e.axes = s.uri.axes.clone();
                from_uri(e, &s.uri)
            }
            PayloadBlock::GenotypeMatrix(g) => {
                let mut e = base("features", "genotype_matrix");
                e.representation_id = Some(g.representation_id.clone());
                e.dtype = g.uri.dtype.clone();
                e.shape = vec![g.n_samples, g.n_variants];
                e.axes = vec!["sample".into(), "variant".into()];
                from_uri(e, &g.uri)
            }
            PayloadBlock::MaskBlock(m) => {
                let mut e = base("mask", "mask_block");
                e.representation_id = Some(m.representation_id.clone());
                e.dtype = m.dtype.clone();
                e.shape = m.shape.clone();
                e.axes = m.uri.axes.clone();
                from_uri(e, &m.uri)
            }
            PayloadBlock::UriBackedPayload(u) => {
                let mut e = base("features", "uri_backed");
                e.dtype = u.dtype.clone();
                e.shape = u.shape.clone();
                e.axes = u.axes.clone();
                from_uri(e, u)
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// The package                                                                 //
// --------------------------------------------------------------------------- //

/// A partition's typed payloads, in package order.
#[derive(Debug, Clone, PartialEq)]
pub struct PackagePartition {
    /// Sample count for this partition.
    pub n_samples: usize,
    /// Feature-source ids aligned with `feature_matrix` payloads.
    pub source_ids: Vec<String>,
    /// `(payload id, block)` pairs.
    pub payloads: Vec<(String, PayloadBlock)>,
}

/// An explicit, fingerprinted record of whether sample identity fell back to row
/// position. `used = true` means no aligned sample-id key existed, so identity
/// was derived from row order — a leakage/traceability hazard that must never be
/// silent (`LOCK-IO` / IO-MM-003).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RowPositionFallback {
    /// Whether identity fell back to row position.
    pub used: bool,
    /// Human-readable reason.
    pub reason: String,
    /// Partitions the decision applies to.
    pub partitions: Vec<String>,
    /// SHA-256 (lowercase hex) over the `{used, reason, partitions}` descriptor.
    pub fingerprint: String,
}

/// The target-agnostic v4 dataset package (`AssembledDataset v2`).
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetPackage {
    /// Dataset name.
    pub name: String,
    /// Task type (`regression`, `binary`, `multiclass`, …).
    pub task_type: String,
    /// Dataset-level signal type.
    pub signal_type: String,
    /// Number of logical feature sources.
    pub n_sources: usize,
    /// Repetition/leakage key column, when declared.
    pub repetition: Option<String>,
    /// Scientific identity column names retained from `sample_index`.
    pub identity: IdentityProvenance,
    /// Aggregation policy, carried for lossless round-trip.
    pub aggregate: Option<AggregateSpec>,
    /// Cross-validation folds, carried for lossless round-trip.
    pub folds: Vec<Fold>,
    /// Stable observation IDs for folds, captured before partition reordering.
    pub fold_provenance: Vec<FoldProvenance>,
    /// Assembly warnings, carried for lossless round-trip.
    pub warnings: Vec<String>,
    /// Assembly audits, carried for lossless round-trip.
    pub audits: Vec<Value>,
    /// Per-partition typed payload blocks.
    pub partitions: IndexMap<String, PackagePartition>,
    /// Explicit row-position fallback diagnostic.
    pub row_position_fallback: RowPositionFallback,
}

impl DatasetPackage {
    /// Represent an [`AssembledDataset`] as a v4 package, losslessly. Each `X`
    /// source becomes a [`FeatureMatrix`] (carrying its processings), `y` a
    /// [`TargetTable`], metadata a [`MetadataTable`], weights a [`WeightsVector`].
    /// The row-position fallback decision is computed with the same rule the
    /// bridge uses for identity (aligned sample-id key in every partition).
    pub fn from_assembled(assembled: &AssembledDataset) -> Result<DatasetPackage, String> {
        Self::from_assembled_with_storage(assembled, &NumericStorageOptions::default())
    }

    pub fn from_assembled_with_storage(
        assembled: &AssembledDataset,
        options: &NumericStorageOptions,
    ) -> Result<DatasetPackage, String> {
        if options.consumer.as_deref() == Some("SpectroDataset")
            && options.targets == Some(super::numeric::NumericStorage::Float64)
            && assembled.blocks.values().any(|b| b.y.is_some())
        {
            return Err("SpectroDataset cannot guarantee an explicitly requested float64 numerical target view".into());
        }
        for block in assembled.blocks.values() {
            if block.source_ids.len() != block.x.len()
                || block
                    .source_ids
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != block.source_ids.len()
            {
                return Err("Missing/duplicate feature source IDs".into());
            }
            if block.x.iter().any(|m| m.n_rows != block.n_samples) {
                return Err("Feature row count mismatch".into());
            }
            if block
                .weights
                .as_ref()
                .is_some_and(|w| w.len() != block.n_samples)
            {
                return Err("Weight row count mismatch".into());
            }
        }
        let row_position_fallback = compute_row_position_fallback(assembled);
        let mut partitions: IndexMap<String, PackagePartition> = IndexMap::new();
        for (part, b) in &assembled.blocks {
            let mut payloads: Vec<(String, PayloadBlock)> = Vec::new();
            for k in 0..b.x.len() {
                let matrix =
                    b.x[k].with_storage(options.features, options.consumer.as_deref(), true)?;
                let headers = b.feature_headers.get(k).cloned().unwrap_or_default();
                if headers.len() != matrix.n_cols {
                    return Err("Feature header count differs from matrix width".into());
                }
                let source_id = b
                    .source_ids
                    .get(k)
                    .ok_or("Missing feature source id")?
                    .clone();
                let (representation_id, axes) = feature_descriptor(assembled, b, k)?;
                let fm = FeatureMatrix {
                    feature_names: b
                        .feature_names
                        .get(k)
                        .cloned()
                        .unwrap_or_else(|| Some(headers.clone())),
                    source_id,
                    representation_id,
                    axes,
                    matrix,
                    headers: b.feature_headers.get(k).cloned().unwrap_or_default(),
                    header_unit: b.header_units.get(k).cloned().unwrap_or_default(),
                    signal_type: b.signal_types.get(k).cloned().flatten(),
                    processings: b
                        .processings
                        .get(k)
                        .into_iter()
                        .flatten()
                        .map(|(name, m)| {
                            if m.n_rows != b.n_samples || m.n_cols != b.x[k].n_cols {
                                return Err("Processing shape differs from feature matrix".into());
                            }
                            Ok((
                                name.clone(),
                                m.with_storage(
                                    options.features,
                                    options.consumer.as_deref(),
                                    true,
                                )?,
                            ))
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                };
                payloads.push((format!("{part}/x{k}"), PayloadBlock::FeatureMatrix(fm)));
            }
            if let Some(y) = &b.y {
                let representation_id =
                    target_representation_id(&assembled.task_type, &b.y_headers, &b.y_categorical);
                if b.y_headers.len() != y.n_cols || y.n_rows != b.n_samples {
                    return Err("Target shape/header mismatch".into());
                }
                let kinds = b
                    .y_headers
                    .iter()
                    .map(|name| {
                        if b.y_categorical.contains_key(name) {
                            2
                        } else if matches!(
                            assembled.task_type.as_str(),
                            "classification" | "binary" | "multiclass"
                        ) {
                            1
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<_>>();
                if kinds.contains(&2) && options.targets.is_some() {
                    return Err(
                        "Numeric precision request for string targets is unsupported".into(),
                    );
                }
                let mut numeric_ir = y.clone();
                if kinds.iter().all(|kind| *kind == 2) {
                    numeric_ir.precision.source_dtype = None;
                }
                let matrix =
                    numeric_ir.with_storage(options.targets, options.consumer.as_deref(), false)?;
                for (column, kind) in kinds.iter().enumerate() {
                    if *kind == 1 {
                        for row in 0..matrix.n_rows {
                            let pos = row * matrix.n_cols + column;
                            if !matrix.validity[pos] {
                                continue;
                            }
                            let label = y.data[pos];
                            if label.fract() != 0.0 || label.abs() > 9_007_199_254_740_991.0 {
                                return Err(
                                    "Classification requires portable integer labels".into()
                                );
                            }
                            super::numeric::validate_label(label as i64, matrix.precision.dtype)?;
                        }
                    }
                }
                for (column, kind) in kinds.iter().enumerate() {
                    if *kind == 2 {
                        let categories = b.y_categorical[&b.y_headers[column]]["categories"]
                            .as_array()
                            .ok_or("Invalid categorical codebook")?;
                        if categories
                            .iter()
                            .any(|v| v.as_str().is_none_or(|s| s.is_empty()))
                        {
                            return Err("Unsupported string categorical labels".into());
                        }
                        for row in 0..matrix.n_rows {
                            let pos = row * matrix.n_cols + column;
                            let value = matrix.data[pos];
                            if matrix.validity[pos] && matrix.data[pos] != y.data[pos] {
                                return Err("Categorical codes cannot change during numeric storage projection".into());
                            }
                            if matrix.validity[pos]
                                && (value < 0.0
                                    || value.fract() != 0.0
                                    || value >= categories.len() as f64)
                            {
                                return Err("Categorical code out of range".into());
                            }
                        }
                    }
                }
                let tt = TargetTable {
                    kinds,
                    matrix,
                    headers: b.y_headers.clone(),
                    categorical: b.y_categorical.clone(),
                    representation_id,
                };
                payloads.push((format!("{part}/y"), PayloadBlock::TargetTable(tt)));
            }
            if let Some(meta) = &b.metadata {
                payloads.push((
                    format!("{part}/metadata"),
                    PayloadBlock::MetadataTable(MetadataTable {
                        frame: meta.clone(),
                    }),
                ));
            }
            if let Some(w) = &b.weights {
                let mut matrix = Matrix::new(w.clone(), w.len(), 1);
                matrix.precision = b.weights_precision.clone();
                if let Some(validity) = &b.weights_validity {
                    matrix.validity = validity.clone();
                }
                let matrix =
                    matrix.with_storage(options.weights, options.consumer.as_deref(), false)?;
                payloads.push((
                    format!("{part}/weights"),
                    PayloadBlock::Weights(WeightsVector {
                        values: matrix.data,
                        validity: matrix.validity,
                        precision: matrix.precision,
                        header: b.weights_header.clone(),
                    }),
                ));
            }
            partitions.insert(
                part.clone(),
                PackagePartition {
                    n_samples: b.n_samples,
                    source_ids: b.source_ids.clone(),
                    payloads,
                },
            );
        }
        let package = DatasetPackage {
            name: assembled.name.clone(),
            task_type: assembled.task_type.clone(),
            signal_type: assembled.signal_type.clone(),
            n_sources: assembled.n_sources,
            repetition: assembled.repetition.clone(),
            identity: assembled.identity.clone(),
            aggregate: assembled.aggregate.clone(),
            folds: assembled.folds.clone(),
            fold_provenance: assembled.fold_provenance.clone(),
            warnings: assembled.warnings.clone(),
            audits: assembled.audits.clone(),
            partitions,
            row_position_fallback,
        };
        package.validate_storage()?;
        Ok(package)
    }

    /// Reconstruct the [`AssembledDataset`] this package was built from. The
    /// inverse of [`from_assembled`](Self::from_assembled) over the supported assembled
    /// payloads after storage normalization. Raw Rust execution IR from
    /// `Matrix::new` / assembly carries NaN with a false validity bit; it need
    /// not satisfy canonical storage validation. This Rust package view instead
    /// retains exact positive-zero placeholders and their false masks for X,
    /// processings, targets and weights. Observed values retain their selected
    /// precision, including signed zero and subnormals. Read data with validity;
    /// data alone is never a logical view or the original masked sentinel.
    /// Whole-content evidence remains in the owner record/content identity.
    /// Logical presentation is explicit at consumers: the wire carries bytes
    /// plus masks, the Python adapter copies them to NaN, and DMD enforces its
    /// mask support. Python's in-memory NaN contract is unchanged.
    pub fn to_assembled(&self) -> AssembledDataset {
        let mut blocks: IndexMap<String, PartitionBlock> = IndexMap::new();
        for (part, p) in &self.partitions {
            let mut block = PartitionBlock {
                n_samples: p.n_samples,
                source_ids: p.source_ids.clone(),
                ..Default::default()
            };
            for (_id, payload) in &p.payloads {
                match payload {
                    PayloadBlock::FeatureMatrix(fm) => {
                        block.x.push(fm.matrix.clone());
                        block.feature_headers.push(fm.headers.clone());
                        block.feature_names.push(fm.feature_names.clone());
                        block.feature_axes.push(fm.axes.clone());
                        block.representation_ids.push(fm.representation_id.clone());
                        block.header_units.push(fm.header_unit.clone());
                        block.signal_types.push(fm.signal_type.clone());
                        block.processings.push(fm.processings.clone());
                    }
                    PayloadBlock::TargetTable(tt) => {
                        block.y = Some(tt.matrix.clone());
                        block.y_headers = tt.headers.clone();
                        block.y_categorical = tt.categorical.clone();
                    }
                    PayloadBlock::MetadataTable(mt) => {
                        block.metadata = Some(mt.frame.clone());
                    }
                    PayloadBlock::Weights(w) => {
                        block.weights = Some(w.values.clone());
                        block.weights_validity = Some(w.validity.clone());
                        block.weights_precision = w.precision.clone();
                        block.weights_header = w.header.clone();
                    }
                    // Package-only payloads have no `PartitionBlock` slot.
                    _ => {}
                }
            }
            blocks.insert(part.clone(), block);
        }
        AssembledDataset {
            name: self.name.clone(),
            task_type: self.task_type.clone(),
            signal_type: self.signal_type.clone(),
            n_sources: self.n_sources,
            blocks,
            folds: self.folds.clone(),
            fold_provenance: self.fold_provenance.clone(),
            repetition: self.repetition.clone(),
            identity: self.identity.clone(),
            aggregate: self.aggregate.clone(),
            warnings: self.warnings.clone(),
            audits: self.audits.clone(),
        }
    }

    /// Build the payload manifest: one row per payload, plus a `root` fingerprint
    /// over the canonical JSON of all rows.
    pub fn manifest(&self) -> PayloadManifest {
        let mut entries: Vec<PayloadManifestEntry> = Vec::new();
        for (part, p) in &self.partitions {
            for (id, payload) in &p.payloads {
                entries.push(payload.manifest_entry(part, id));
            }
        }
        let entries_value = serde_json::to_value(&entries).expect("manifest entries serialize");
        let root = sha256_hex(
            canonical_json(&entries_value)
                .expect("manifest serializes")
                .as_bytes(),
        );
        PayloadManifest { root, entries }
    }

    /// A stable, **bytes-free** canonical summary for goldens. Emits dataset-level
    /// metadata, per-partition sample counts, the payload manifest (hashes /
    /// shapes / storage only — never the payload bytes) and the row-position
    /// fallback diagnostic.
    pub fn to_summary_value(&self) -> Value {
        let mut parts = Map::new();
        for (name, p) in &self.partitions {
            parts.insert(
                name.clone(),
                json!({ "n_samples": p.n_samples, "source_ids": p.source_ids }),
            );
        }
        let aggregate = self.aggregate.as_ref().map(|a| {
            json!({
                "by": a.by,
                "method": a.method.value(),
                "exclude_outliers": a.exclude_outliers,
                "outlier_threshold": a.outlier_threshold,
            })
        });
        let manifest = serde_json::to_value(self.manifest()).expect("manifest serializes");
        let row_position_fallback =
            serde_json::to_value(&self.row_position_fallback).expect("fallback serializes");
        json!({
            "schema_version": DATASET_PACKAGE_VERSION,
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "repetition": self.repetition,
            "identity": {
                "provenance": identity_provenance_value(&self.identity),
                "row_position_fallback": row_position_fallback,
            },
            "folds": self.folds.iter().map(|(tr, vl)| vec![tr.clone(), vl.clone()]).collect::<Vec<_>>(),
            "fold_provenance": self.fold_provenance.iter().map(|fold| json!({
                "train_observation_ids": fold.train_observation_ids,
                "validation_observation_ids": fold.validation_observation_ids,
            })).collect::<Vec<_>>(),
            "aggregate": aggregate,
            "partitions": Value::Object(parts),
            "manifest": manifest,
        })
    }

    pub fn validate_storage(&self) -> Result<(), String> {
        for partition in self.partitions.values() {
            let ordered_sources = partition
                .payloads
                .iter()
                .filter_map(|(_, payload)| match payload {
                    PayloadBlock::FeatureMatrix(feature) => Some(&feature.source_id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if ordered_sources != partition.source_ids.iter().collect::<Vec<_>>() {
                return Err("Ordered feature sources disagree with partition source IDs".into());
            }
            for kind in ["targets", "metadata", "weights"] {
                let count = partition
                    .payloads
                    .iter()
                    .filter(|(_, payload)| {
                        matches!(
                            (kind, payload),
                            ("targets", PayloadBlock::TargetTable(_))
                                | ("metadata", PayloadBlock::MetadataTable(_))
                                | ("weights", PayloadBlock::Weights(_))
                        )
                    })
                    .count();
                if count > 1 {
                    return Err(format!("Multiple {kind} payloads cannot be represented by the assembled owner seam"));
                }
            }
            let mut ids = std::collections::HashSet::new();
            let mut sources = std::collections::HashSet::new();
            for (id, payload) in &partition.payloads {
                if !ids.insert(id) {
                    return Err("Duplicate payload ID".into());
                }
                match payload {
                    PayloadBlock::FeatureMatrix(f) => {
                        f.matrix.validate_storage(true)?;
                        for audit in self.audits.iter().filter(|audit| {
                            audit["source"] == f.source_id && audit.get("projection_axes").is_some()
                        }) {
                            let axes = super::projection::axes_from_full(
                                &audit["projection_axes"],
                                f.matrix.n_cols,
                            )?;
                            if super::projection::axes_full(&axes)?
                                != super::projection::axes_full(&f.axes)?
                                || audit["projection_representation_id"].as_str()
                                    != Some(f.representation_id.as_str())
                            {
                                return Err(
                                    "Exact feature axes/representation disagree with owner audits"
                                        .into(),
                                );
                            }
                        }
                        if f.source_id.trim().is_empty()
                            || !sources.insert(&f.source_id)
                            || f.matrix.n_rows != partition.n_samples
                            || f.headers.len() != f.matrix.n_cols
                            || f.feature_names
                                .as_ref()
                                .is_some_and(|names| names.len() != f.matrix.n_cols)
                        {
                            return Err("Feature shape/source/header mismatch".into());
                        }
                        for axis in &f.axes {
                            if axis["name"].as_str().is_none_or(str::is_empty)
                                || (!axis["unit"].is_null() && !axis["unit"].is_string())
                            {
                                return Err("Invalid feature axis".into());
                            }
                            if let Some(coords) = axis["coordinates"].as_array() {
                                if coords.len() != f.matrix.n_cols
                                    || !(coords
                                        .iter()
                                        .all(|v| super::numeric::exact_json_number_f64(v).is_ok())
                                        || coords.iter().all(Value::is_string))
                                {
                                    return Err("Invalid feature axis coordinates".into());
                                }
                            } else if !axis["coordinates"].is_null() {
                                return Err("Invalid feature axis coordinates".into());
                            }
                        }
                        for (_, matrix) in &f.processings {
                            matrix.validate_storage(true)?;
                            if matrix.n_rows != f.matrix.n_rows
                                || matrix.n_cols != f.matrix.n_cols
                                || matrix.precision.dtype != f.matrix.precision.dtype
                            {
                                return Err("Processing storage/shape differs from features; select an explicit common features storage request".into());
                            }
                        }
                    }
                    PayloadBlock::TargetTable(t) => {
                        t.matrix.validate_storage(false)?;
                        if t.matrix.precision.consumer.as_deref() == Some("SpectroDataset")
                            && t.matrix.precision.requested_dtype
                                == Some(super::numeric::NumericStorage::Float64)
                        {
                            return Err("SpectroDataset cannot guarantee an explicitly requested float64 numerical target view".into());
                        }
                        if t.kinds.len() != t.headers.len()
                            || t.matrix.n_cols != t.headers.len()
                            || t.matrix.n_rows != partition.n_samples
                            || t.headers
                                .iter()
                                .collect::<std::collections::HashSet<_>>()
                                .len()
                                != t.headers.len()
                        {
                            return Err("Target kind/shape/header mismatch".into());
                        }
                        for (column, &kind) in t.kinds.iter().enumerate() {
                            if kind > 2 {
                                return Err("Unknown target kind".into());
                            }
                            if kind == 2 && t.matrix.precision.requested_dtype.is_some() {
                                return Err("Numeric precision request for string target".into());
                            }
                            for row in 0..t.matrix.n_rows {
                                let pos = row * t.matrix.n_cols + column;
                                if !t.matrix.validity[pos] {
                                    continue;
                                }
                                if kind == 2 {
                                    if super::projection::target_string(t, row, column)
                                        .is_none_or(str::is_empty)
                                    {
                                        return Err("Invalid categorical code/label".into());
                                    }
                                } else if kind == 1 {
                                    let value = t.matrix.data[pos];
                                    if value.fract() != 0.0 || value.abs() > 9_007_199_254_740_991.0
                                    {
                                        return Err(
                                            "Classification label outside portable integer domain"
                                                .into(),
                                        );
                                    }
                                    super::numeric::validate_label(
                                        value as i64,
                                        t.matrix.precision.dtype,
                                    )?;
                                }
                            }
                        }
                    }
                    PayloadBlock::Weights(w) => {
                        let mut matrix = Matrix::new(w.values.clone(), w.values.len(), 1);
                        matrix.validity = w.validity.clone();
                        matrix.precision = w.precision.clone();
                        matrix.validate_storage(false)?;
                        if matrix.n_rows != partition.n_samples {
                            return Err("Weight row count mismatch".into());
                        }
                    }
                    PayloadBlock::MetadataTable(metadata)
                        if metadata.frame.n_rows != partition.n_samples
                            || metadata
                                .frame
                                .columns
                                .iter()
                                .any(|column| column.values.len() != partition.n_samples) =>
                    {
                        return Err("Metadata shape differs from partition rows".into());
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Exact v4 payload export. There is intentionally no whole-package Rust reader.
    pub fn to_full_value(&self) -> Result<Value, String> {
        self.validate_storage()?;
        let mut value = self.to_summary_value();
        value["partition_order"] = json!(self.partitions.keys().collect::<Vec<_>>());
        value["audits"] = json!(self.audits);
        let mut partitions = Map::new();
        for (name, partition) in &self.partitions {
            let mut payloads = Map::new();
            for (id, payload) in &partition.payloads {
                let (kind, body) = match payload {
                    PayloadBlock::FeatureMatrix(feature) => (
                        "feature_matrix",
                        json!({
                        "matrix":feature.matrix.full_value(),"headers":feature.headers,"header_unit":feature.header_unit,"signal_type":feature.signal_type,
                        "source_id":feature.source_id,"feature_names":feature.feature_names,"representation_id":feature.representation_id,"axes":super::projection::axes_full(&feature.axes)?,
                        "processings":feature.processings.iter().map(|(name,matrix)|json!({"name":name,"matrix":matrix.full_value()})).collect::<Vec<_>>() }),
                    ),
                    PayloadBlock::TargetTable(table) => (
                        "target_table",
                        json!({"columns":super::projection::target_columns(table),"categorical":table.categorical,"representation_id":table.representation_id}),
                    ),
                    PayloadBlock::Weights(weights) => {
                        let mut matrix =
                            Matrix::new(weights.values.clone(), weights.values.len(), 1);
                        matrix.validity = weights.validity.clone();
                        matrix.precision = weights.precision.clone();
                        (
                            "weights",
                            json!({"matrix":matrix.full_value(),"header":weights.header}),
                        )
                    }
                    PayloadBlock::MetadataTable(metadata) => (
                        "metadata_table",
                        serde_json::from_str(&frame_canonical_bytes(&metadata.frame))
                            .map_err(|e| e.to_string())?,
                    ),
                    PayloadBlock::SpectralRecordSet(records) => {
                        ("spectral_record_set", json!({"records":records.records}))
                    }
                    PayloadBlock::NdTensor(tensor) => (
                        "nd_tensor",
                        json!({"representation_id":tensor.representation_id,"dtype":tensor.dtype,"shape":tensor.shape,"axis_names":tensor.axis_names,"observation_ids":tensor.observation_ids,"uri":tensor.uri}),
                    ),
                    PayloadBlock::SequenceBlock(sequence) => (
                        "sequence_block",
                        json!({"representation_id":sequence.representation_id,"dtype":sequence.dtype,"n_channels":sequence.n_channels,"lengths":sequence.lengths,"uri":sequence.uri}),
                    ),
                    PayloadBlock::GenotypeMatrix(genotype) => (
                        "genotype_matrix",
                        json!({"representation_id":genotype.representation_id,"n_variants":genotype.n_variants,"n_samples":genotype.n_samples,"encoding":genotype.encoding,"uri":genotype.uri}),
                    ),
                    PayloadBlock::MaskBlock(mask) => (
                        "mask_block",
                        json!({"representation_id":mask.representation_id,"dtype":mask.dtype,"shape":mask.shape,"uri":mask.uri}),
                    ),
                    PayloadBlock::UriBackedPayload(uri) => ("uri_backed_payload", json!(uri)),
                };
                if payloads
                    .insert(id.clone(), json!({"kind":kind,"value":body}))
                    .is_some()
                {
                    return Err("Duplicate payload ID".into());
                }
            }
            partitions.insert(name.clone(),json!({"n_samples":partition.n_samples,"source_ids":partition.source_ids,"payloads":payloads}));
        }
        value["partitions"] = Value::Object(partitions);
        Ok(value)
    }

    /// The canonical-JSON string of [`to_summary_value`](Self::to_summary_value).
    pub fn to_canonical_summary(&self) -> String {
        canonical_json(&self.to_summary_value()).expect("summary serializes")
    }
}

// --------------------------------------------------------------------------- //
// Helpers                                                                     //
// --------------------------------------------------------------------------- //

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(crate) fn payload_head(kind: &str) -> Vec<u8> {
    let mut out = Vec::new();
    super::numeric::string(&mut out, "nirs4all.io.package-payload.v4");
    out.extend_from_slice(&4_u32.to_le_bytes());
    super::numeric::string(&mut out, kind);
    out
}

pub fn feature_content_bytes(fm: &FeatureMatrix) -> Vec<u8> {
    let mut out = payload_head("features");
    out.extend_from_slice(&super::projection::feature_bytes(fm));
    super::numeric::u64_bytes(&mut out, fm.processings.len());
    for (name, matrix) in &fm.processings {
        super::numeric::string(&mut out, name);
        out.extend_from_slice(&matrix.numeric_bytes());
    }
    out
}

pub fn target_content_bytes(tt: &TargetTable) -> Vec<u8> {
    let mut out = payload_head("targets");
    super::numeric::u64_bytes(&mut out, tt.headers.len());
    for column in 0..tt.headers.len() {
        out.extend_from_slice(&super::projection::target_bytes(tt, column));
    }
    out
}

pub fn weights_content_bytes(w: &WeightsVector) -> Vec<u8> {
    let mut out = payload_head("weights");
    out.extend_from_slice(&w.precision.bytes());
    let mut matrix = Matrix::new(w.values.clone(), w.values.len(), 1);
    matrix.validity = w.validity.clone();
    matrix.precision = w.precision.clone();
    out.extend_from_slice(&matrix.numeric_bytes());
    out
}

pub(crate) fn feature_descriptor(
    assembled: &AssembledDataset,
    block: &PartitionBlock,
    index: usize,
) -> Result<(String, Vec<Value>), String> {
    let source_id = &block.source_ids[index];
    if !block.feature_axes.is_empty() || !block.representation_ids.is_empty() {
        if block.feature_axes.len() != block.x.len()
            || block.representation_ids.len() != block.x.len()
            || block.feature_names.len() != block.x.len()
        {
            return Err("Feature metadata must align exactly with source matrices".into());
        }
        return Ok((
            block.representation_ids[index].clone(),
            block.feature_axes[index].clone(),
        ));
    }

    if let Some(audit) = assembled
        .audits
        .iter()
        .find(|audit| audit["source"] == *source_id && audit.get("projection_axes").is_some())
    {
        return Ok((
            audit["projection_representation_id"]
                .as_str()
                .ok_or("Projection representation missing")?
                .to_owned(),
            super::projection::axes_from_full(&audit["projection_axes"], block.x[index].n_cols)?,
        ));
    }
    if let Some(source) = assembled.audits.iter().find_map(|audit| {
        (audit["selected_source"] == *source_id)
            .then(|| audit.get("raw_projection_source"))
            .flatten()
    }) {
        let axes = source["axes"].as_array().into_iter().flatten().filter_map(|name| {
            let name = name.as_str()?;
            if name == "sample" { return None; }
            Some(json!({"name":name,"unit":source["axis_units"][name],"coordinates":source["axis_coordinates"][name]}))
        }).collect();
        return Ok((
            source["representation_id"]
                .as_str()
                .unwrap_or(repr_ids::SIGNAL_1D)
                .to_owned(),
            axes,
        ));
    }
    let unit = block
        .header_units
        .get(index)
        .map(String::as_str)
        .unwrap_or("text");
    let name = feature_axis_name(unit);
    let headers = block
        .feature_headers
        .get(index)
        .cloned()
        .unwrap_or_default();
    let coords = header_coordinates(&headers)?;
    let raw_unit = if matches!(unit, "text" | "index" | "") {
        Value::Null
    } else {
        json!(unit)
    };
    let coordinates = json!(coords);
    Ok((
        repr_ids::SIGNAL_1D.into(),
        vec![json!({"name":name,"unit":raw_unit,"coordinates":coordinates})],
    ))
}

/// A frame's deterministic value (name/dtype/values per column) for hashing.
fn frame_canonical_bytes(f: &Frame) -> String {
    let columns: Vec<Value> = f
        .columns
        .iter()
        .map(|c| {
            json!({
                "name": c.name,
                "dtype": c.dtype.label(),
                "values": c.values.iter().enumerate().map(|(row, v)| super::frame::metadata_cell_value(c, row, v)).collect::<Vec<_>>(),
            })
        })
        .collect();
    let v = json!({ "n_rows": f.n_rows, "columns": columns });
    canonical_json(&v).expect("frame serializes")
}

/// Spectral axis name from a header unit — mirrors the bridge's `feature_axis`.
pub(crate) fn header_coordinates(headers: &[String]) -> Result<Option<Vec<Value>>, String> {
    if headers
        .iter()
        .any(|header| !header.parse::<f64>().is_ok_and(f64::is_finite))
    {
        return Ok(None);
    }
    headers
        .iter()
        .map(|header| {
            super::numeric::NumericStorage::Float64
                .decimal(header)
                .map(Value::from)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub(crate) fn feature_axis_name(unit: &str) -> &'static str {
    let u = unit.to_ascii_lowercase();
    if u.contains("nm") || u.contains("nanomet") || u.contains("wavelength") {
        "wavelength"
    } else if u.contains("cm-1")
        || u.contains("cm^-1")
        || u.contains("1/cm")
        || u.contains("wavenumber")
        || u.contains("cm\u{207b}\u{b9}")
    {
        "wavenumber"
    } else {
        "feature"
    }
}

/// Pick a target representation-ID with the same rule the bridge uses: a single
/// target ⇒ `target_numeric` / `target_categorical`; uniform-kind multivariate ⇒
/// the matrix variant; mixed kinds ⇒ `None` (the bridge emits per-target specs).
fn target_representation_id(
    task_type: &str,
    headers: &[String],
    categorical: &IndexMap<String, Value>,
) -> Option<String> {
    if headers.is_empty() {
        return None;
    }
    let is_cat = |h: &str| {
        categorical.contains_key(h)
            || matches!(task_type, "classification" | "binary" | "multiclass")
    };
    let kinds: Vec<bool> = headers.iter().map(|h| is_cat(h)).collect();
    let all_same = kinds.iter().all(|k| *k == kinds[0]);
    if headers.len() == 1 {
        Some(
            if kinds[0] {
                repr_ids::TARGET_CATEGORICAL
            } else {
                repr_ids::TARGET_NUMERIC
            }
            .to_string(),
        )
    } else if all_same {
        Some(
            if kinds[0] {
                repr_ids::TARGET_CATEGORICAL_MATRIX
            } else {
                repr_ids::TARGET_NUMERIC_MATRIX
            }
            .to_string(),
        )
    } else {
        None
    }
}

/// Decide + describe the row-position fallback with the bridge's identity rule:
/// identity is key-based iff a stable sample-id key is declared and aligned (present,
/// length == `n_samples`) in every partition; otherwise it falls back to row
/// position.
fn compute_row_position_fallback(a: &AssembledDataset) -> RowPositionFallback {
    let partitions: Vec<String> = a.blocks.keys().cloned().collect();
    let sample_id = a.identity.sample_id.as_deref();
    let has_sample_id = sample_id.is_some_and(|col| {
        a.blocks.values().all(|b| {
            b.metadata
                .as_ref()
                .is_some_and(|f| f.has_column(col) && f.str_column(col).len() == b.n_samples)
        })
    });
    let (used, reason) = if has_sample_id {
        (
            false,
            format!(
                "stable sample-id key '{}' is aligned in every partition",
                sample_id.expect("has_sample_id implies a key")
            ),
        )
    } else if let Some(col) = sample_id {
        (
            true,
            format!(
                "sample-id key '{col}' is absent or not aligned in every partition; sample identity falls back to row position"
            ),
        )
    } else {
        (
            true,
            "no stable sample-id key declared; sample identity falls back to row position"
                .to_string(),
        )
    };
    let descriptor = json!({ "used": used, "reason": reason, "partitions": partitions });
    let fingerprint = sha256_hex(
        canonical_json(&descriptor)
            .expect("descriptor serializes")
            .as_bytes(),
    );
    RowPositionFallback {
        used,
        reason,
        partitions,
        fingerprint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize::frame::Column;

    fn matrix(rows: usize, cols: usize) -> Matrix {
        Matrix::new((0..rows * cols).map(|v| v as f64).collect(), rows, cols)
    }

    fn base_block() -> PartitionBlock {
        PartitionBlock {
            n_samples: 2,
            source_ids: vec!["spectra".into()],
            x: vec![matrix(2, 3)],
            feature_headers: vec![vec!["1000".into(), "1010".into(), "1020".into()]],
            feature_names: vec![],
            feature_axes: vec![],
            representation_ids: vec![],
            header_units: vec!["nm".into()],
            signal_types: vec![Some("absorbance".into())],
            processings: vec![vec![]],
            y: Some(matrix(2, 1)),
            y_headers: vec!["protein".into()],
            y_categorical: Default::default(),
            metadata: Some(Frame::from_columns(
                vec![
                    Column::from_cells("batch", vec![Cell::Str("a".into()), Cell::Str("b".into())]),
                    Column::from_cells("rep", vec![Cell::Str("r1".into()), Cell::Str("r2".into())]),
                ],
                "text",
            )),
            weights: Some(vec![1.0, 2.0]),
            weights_header: Some("w".into()),
            weights_validity: None,
            weights_precision: Default::default(),
        }
    }

    fn assembled(block: PartitionBlock) -> AssembledDataset {
        let mut a = AssembledDataset {
            name: "demo".into(),
            task_type: "regression".into(),
            signal_type: "absorbance".into(),
            n_sources: 1,
            blocks: IndexMap::new(),
            folds: vec![],
            fold_provenance: vec![],
            repetition: None,
            identity: IdentityProvenance::default(),
            aggregate: None,
            warnings: vec![],
            audits: vec![],
        };
        a.blocks.insert("train".into(), block);
        a
    }

    fn uri_ref(name: &str, dtype: &str, shape: Vec<usize>, axes: Vec<&str>) -> UriRef {
        let bytes = format!("{name}:{dtype}:{shape:?}");
        UriRef {
            uri: format!("s3://bucket/{name}.bin"),
            dtype: dtype.into(),
            shape,
            axes: axes.into_iter().map(str::to_string).collect(),
            content_hash: sha256_hex(bytes.as_bytes()),
            byte_len: bytes.len() as u64,
            codec: Some("raw".into()),
        }
    }

    #[test]
    fn from_assembled_round_trips_losslessly() {
        let a = assembled(base_block());
        let pkg = DatasetPackage::from_assembled(&a).unwrap();
        // The v2-expressible payload is reconstructed byte-for-byte.
        assert_eq!(pkg.to_assembled().to_full_value(), a.to_full_value());
    }

    #[test]
    fn canonical_package_view_preserves_masks_and_storage() {
        let mut block = base_block();
        block.x[0] = Matrix::new(vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0], 2, 3);
        block.y = Some(Matrix::new(vec![1.5, f64::NAN], 2, 1));
        block.processings[0].push(("raw".into(), block.x[0].clone()));
        block.weights = Some(vec![1.0, f64::NAN]);
        block.weights_validity = Some(vec![true, false]);
        let original = assembled(block);
        let raw = &original.blocks["train"];
        assert!(raw.x[0].data[1].is_nan());
        assert!(raw.y.as_ref().unwrap().data[1].is_nan());
        assert!(raw.processings[0][0].1.data[1].is_nan());
        assert!(raw.weights.as_ref().unwrap()[1].is_nan());
        assert!(!raw.x[0].validity[1]);
        assert!(!raw.y.as_ref().unwrap().validity[1]);
        assert!(!raw.processings[0][0].1.validity[1]);
        assert!(!raw.weights_validity.as_ref().unwrap()[1]);
        assert!(raw.x[0].validate_storage(true).is_err());
        let package = DatasetPackage::from_assembled(&original).unwrap();
        let package_wire = package.to_full_value().unwrap();
        let view = package.to_assembled();
        let block = &view.blocks["train"];
        assert_eq!(block.x[0].data[1].to_bits(), 0);
        assert_eq!(block.y.as_ref().unwrap().data[1].to_bits(), 0);
        assert_eq!(block.processings[0][0].1.data[1].to_bits(), 0);
        assert_eq!(block.weights.as_ref().unwrap()[1].to_bits(), 0);
        assert!(!block.x[0].validity[1]);
        assert!(!block.y.as_ref().unwrap().validity[1]);
        assert!(!block.processings[0][0].1.validity[1]);
        assert_eq!(block.weights_validity, Some(vec![true, false]));
        for (raw, stored) in [
            (&raw.x[0], &block.x[0]),
            (raw.y.as_ref().unwrap(), block.y.as_ref().unwrap()),
            (&raw.processings[0][0].1, &block.processings[0][0].1),
        ] {
            assert_eq!(raw.validity, stored.validity);
            for ((&expected, &observed), &valid) in
                raw.data.iter().zip(&stored.data).zip(&stored.validity)
            {
                if valid {
                    assert_eq!(expected.to_bits(), observed.to_bits());
                }
            }
        }
        assert_eq!(
            raw.weights.as_ref().unwrap()[0].to_bits(),
            block.weights.as_ref().unwrap()[0].to_bits()
        );
        let full = view.to_full_value().unwrap();
        assert_eq!(full, original.to_full_value().unwrap());
        let feature = Matrix::from_full_value(&full["blocks"]["train"]["x"][0]).unwrap();
        assert_eq!(feature.data[1].to_bits(), 0);
        let processing =
            Matrix::from_full_value(&full["blocks"]["train"]["processings"][0][0]["matrix"])
                .unwrap();
        assert_eq!(processing.data[1].to_bits(), 0);
        let weights = Matrix::from_full_value(&full["blocks"]["train"]["weights"]).unwrap();
        assert_eq!(weights.data[1].to_bits(), 0);
        assert!(!weights.validity[1]);
        assert_eq!(package.to_full_value().unwrap(), package_wire);
        let repacked = DatasetPackage::from_assembled(&view).unwrap();
        assert_eq!(repacked.to_full_value().unwrap(), package_wire);
        assert_eq!(repacked.to_assembled().to_full_value().unwrap(), full);
        for (_, payload) in &package.partitions["train"].payloads {
            let matrix = match payload {
                PayloadBlock::FeatureMatrix(feature) => &feature.matrix,
                PayloadBlock::TargetTable(target) => &target.matrix,
                _ => continue,
            };
            matrix.validate_storage(false).unwrap();
            for (&value, &valid) in matrix.data.iter().zip(&matrix.validity) {
                if !valid {
                    assert_eq!(value.to_bits(), 0);
                }
            }
            let mut invalid = matrix.clone();
            let missing = invalid.validity.iter().position(|valid| !valid).unwrap();
            invalid.data[missing] = f64::NAN;
            assert!(invalid.validate_storage(false).is_err());
            let mut invalid_wire = matrix.full_value();
            let mut hex = invalid_wire["data_bytes_hex"].as_str().unwrap().to_string();
            let offset = missing * hex.len() / matrix.data.len();
            hex.replace_range(offset..offset + 2, "01");
            invalid_wire["data_bytes_hex"] = json!(hex);
            assert!(Matrix::from_full_value(&invalid_wire).is_err());
        }
    }

    #[test]
    fn canonical_view_keeps_observed_signed_zero_and_subnormal_next_to_missing() {
        let tiny = f64::from_bits(1);
        let mut block = base_block();
        block.x[0] = Matrix::new(vec![-0.0, f64::NAN, tiny, 4.0, 5.0, 6.0], 2, 3);
        block.y = Some(Matrix::new(vec![-0.0, f64::NAN], 2, 1));
        block.processings[0].push(("raw".into(), block.x[0].clone()));
        block.weights = Some(vec![-0.0, f64::NAN]);
        block.weights_validity = Some(vec![true, false]);
        let raw = assembled(block);
        let float64 = super::super::numeric::NumericStorage::Float64;
        let options = NumericStorageOptions {
            features: Some(float64),
            targets: Some(float64),
            weights: Some(float64),
            ..Default::default()
        };
        let package = DatasetPackage::from_assembled_with_storage(&raw, &options).unwrap();
        let view = package.to_assembled();
        let block = &view.blocks["train"];
        for matrix in [
            &block.x[0],
            &block.processings[0][0].1,
            block.y.as_ref().unwrap(),
        ] {
            assert_eq!(matrix.data[0].to_bits(), (-0.0_f64).to_bits());
            assert!(matrix.validity[0]);
            assert_eq!(matrix.data[1].to_bits(), 0);
            assert!(!matrix.validity[1]);
            matrix.validate_storage(false).unwrap();
        }
        for matrix in [&block.x[0], &block.processings[0][0].1] {
            assert_eq!(matrix.data[2].to_bits(), tiny.to_bits());
            assert!(matrix.validity[2]);
        }
        assert_eq!(
            block.weights.as_ref().unwrap()[0].to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(block.weights.as_ref().unwrap()[1].to_bits(), 0);
        assert_eq!(block.weights_validity, Some(vec![true, false]));
    }

    #[test]
    fn round_trip_preserves_multi_source_and_processings() {
        let mut block = base_block();
        block.x.push(matrix(2, 2));
        block.source_ids.push("spectra2".into());
        block
            .feature_headers
            .push(vec!["1200".into(), "1210".into()]);
        block.header_units.push("cm-1".into());
        block.signal_types.push(Some("reflectance".into()));
        block.processings = vec![vec![("snv".into(), matrix(2, 3))], vec![]];
        let mut a = assembled(block);
        a.n_sources = 2;
        let pkg = DatasetPackage::from_assembled(&a).unwrap();
        assert_eq!(pkg.to_assembled().to_full_value(), a.to_full_value());
    }

    #[test]
    fn summary_is_deterministic_and_carries_no_payload_bytes() {
        let a = assembled(base_block());
        let pkg = DatasetPackage::from_assembled(&a).unwrap();
        let s1 = pkg.to_canonical_summary();
        let s2 = pkg.to_canonical_summary();
        assert_eq!(s1, s2, "canonical summary must be deterministic");
        // Manifest present, hashes look like SHA-256, bytes are not embedded.
        let manifest = pkg.manifest();
        assert!(manifest.entries.iter().all(|e| e.content_hash.len() == 64));
        assert!(manifest.entries.iter().all(|e| e.byte_len > 0));
        assert_eq!(manifest.root.len(), 64);
        // The feature matrix has raw values 0..6; a payload-carrying JSON would
        // serialize them. The bytes-free summary must not.
        assert!(
            !s1.contains("\"data\""),
            "summary must not embed raw matrix data arrays"
        );
    }

    #[test]
    fn manifest_maps_feature_target_metadata_weights() {
        let a = assembled(base_block());
        let pkg = DatasetPackage::from_assembled(&a).unwrap();
        let m = pkg.manifest();
        let by_id = |id: &str| m.entries.iter().find(|e| e.id == id).unwrap();
        assert_eq!(
            by_id("train/x0").representation_id.as_deref(),
            Some("signal_1d")
        );
        assert_eq!(by_id("train/x0").axes, vec!["sample", "wavelength"]);
        assert_eq!(
            by_id("train/y").representation_id.as_deref(),
            Some("target_numeric")
        );
        assert_eq!(
            by_id("train/metadata").representation_id.as_deref(),
            Some("sample_metadata")
        );
        assert_eq!(by_id("train/weights").role, "weights");
        assert!(m
            .entries
            .iter()
            .all(|e| e.storage == PayloadStorageKind::Inline));
    }

    #[test]
    fn processing_stack_lifts_to_signal_with_processings() {
        let mut block = base_block();
        block.processings = vec![vec![("snv".into(), matrix(2, 3))]];
        let pkg = DatasetPackage::from_assembled(&assembled(block)).unwrap();
        let e = pkg
            .manifest()
            .entries
            .into_iter()
            .find(|e| e.id == "train/x0")
            .unwrap();
        assert_eq!(
            e.representation_id.as_deref(),
            Some("signal_with_processings")
        );
        assert_eq!(e.axes, vec!["sample", "processing", "wavelength"]);
    }

    #[test]
    fn uniform_multivariate_targets_lift_to_matrix_representation() {
        let mut block = base_block();
        block.y = Some(matrix(2, 2));
        block.y_headers = vec!["protein".into(), "moisture".into()];
        let pkg = DatasetPackage::from_assembled(&assembled(block)).unwrap();
        let e = pkg
            .manifest()
            .entries
            .into_iter()
            .find(|e| e.id == "train/y")
            .unwrap();
        assert_eq!(
            e.representation_id.as_deref(),
            Some("target_numeric_matrix")
        );
        assert_eq!(e.axes, vec!["sample", "target"]);
    }

    #[test]
    fn content_hash_and_root_detect_payload_tampering() {
        let a = assembled(base_block());
        let root0 = DatasetPackage::from_assembled(&a).unwrap().manifest().root;
        let x0 = DatasetPackage::from_assembled(&a)
            .unwrap()
            .manifest()
            .entries
            .into_iter()
            .find(|e| e.id == "train/x0")
            .unwrap()
            .content_hash;

        // Flip a single feature value.
        let mut tampered = a.clone();
        tampered.blocks["train"].x[0].data[0] += 1.0;
        let m1 = DatasetPackage::from_assembled(&tampered)
            .unwrap()
            .manifest();
        let x1 = m1.entries.iter().find(|e| e.id == "train/x0").unwrap();
        assert_ne!(x0, x1.content_hash, "feature hash must change on tamper");
        assert_ne!(root0, m1.root, "manifest root must change on tamper");
    }

    #[test]
    fn row_position_fallback_is_recorded_when_no_key() {
        let pkg = DatasetPackage::from_assembled(&assembled(base_block())).unwrap();
        let f = &pkg.row_position_fallback;
        assert!(f.used, "no sample-id key ⇒ row-position fallback");
        assert!(f.reason.contains("no stable sample-id key"));
        assert_eq!(f.fingerprint.len(), 64);
        assert_eq!(f.partitions, vec!["train"]);
    }

    #[test]
    fn row_position_fallback_off_when_key_aligned() {
        let mut a = assembled(base_block());
        a.identity.sample_id = Some("rep".into()); // aligned metadata column of len 2
        let pkg = DatasetPackage::from_assembled(&a).unwrap();
        assert!(!pkg.row_position_fallback.used);
        assert!(pkg
            .row_position_fallback
            .reason
            .contains("sample-id key 'rep'"));
    }

    #[test]
    fn summary_has_one_versioned_identity_object_and_round_trips_provenance() {
        let mut a = assembled(base_block());
        a.identity = IdentityProvenance {
            source_ids: vec!["spectra".into()],
            sample_id: Some("sample_id".into()),
            observation_id: Some("scan_id".into()),
            repetition_id: Some("rep".into()),
            group_id: Some("batch".into()),
        };
        a.fold_provenance = vec![FoldProvenance {
            train_observation_ids: vec!["O1".into()],
            validation_observation_ids: vec!["O2".into()],
        }];
        let package = DatasetPackage::from_assembled(&a).unwrap();
        let value = package.to_summary_value();
        assert_eq!(value["schema_version"], Value::from(4));
        assert_eq!(
            package.to_assembled().to_full_value().unwrap()["assembled_schema_version"],
            Value::from(3),
            "the binding-facing assembled wire is explicitly versioned"
        );
        assert_eq!(value["identity"].as_object().unwrap().len(), 2);
        assert_eq!(
            value["identity"]["provenance"]["sample_id"],
            Value::String("sample_id".into())
        );
        assert_eq!(
            package.to_assembled().identity,
            a.identity,
            "package round-trip keeps identity provenance"
        );
        assert_eq!(package.to_assembled().fold_provenance, a.fold_provenance);
    }

    #[test]
    fn v4_cross_language_canonical_summary_matches_golden() {
        let mut a = assembled(base_block());
        a.identity = IdentityProvenance {
            source_ids: vec!["spectra".into()],
            sample_id: Some("sample_id".into()),
            observation_id: Some("observation_id".into()),
            repetition_id: Some("rep".into()),
            group_id: Some("batch".into()),
        };
        a.folds = vec![(vec![0], vec![1])];
        a.fold_provenance = vec![FoldProvenance {
            train_observation_ids: vec!["O1".into()],
            validation_observation_ids: vec!["O2".into()],
        }];
        let metadata = a.blocks["train"].metadata.as_mut().unwrap();
        metadata.columns.push(Column::from_cells(
            "sample_id",
            vec![Cell::Str("S1".into()), Cell::Str("S2".into())],
        ));
        metadata.columns.push(Column::from_cells(
            "observation_id",
            vec![Cell::Str("O1".into()), Cell::Str("O2".into())],
        ));

        assert_eq!(
            DatasetPackage::from_assembled(&a)
                .unwrap()
                .to_canonical_summary(),
            include_str!("../../../../tests/goldens/dataset_package_v4.cross_language.canonical")
        );
    }

    #[test]
    fn repetition_does_not_disable_row_position_fallback() {
        let mut a = assembled(base_block());
        a.repetition = Some("rep".into());
        assert!(
            DatasetPackage::from_assembled(&a)
                .unwrap()
                .row_position_fallback
                .used
        );
    }

    #[test]
    fn uri_backed_payload_references_bytes_without_embedding() {
        let uri = UriRef {
            uri: "s3://bucket/cube.npy".into(),
            dtype: "float32".into(),
            shape: vec![4, 128, 128, 3],
            axes: vec![
                "sample".into(),
                "height".into(),
                "width".into(),
                "channel".into(),
            ],
            content_hash: "a".repeat(64),
            byte_len: 4 * 128 * 128 * 3 * 4,
            codec: Some("npy".into()),
        };
        let tensor = PayloadBlock::NdTensor(NdTensor {
            representation_id: repr_ids::RGB_IMAGE.into(),
            dtype: "uint8".into(),
            shape: vec![4, 128, 128, 3],
            axis_names: vec![
                "sample".into(),
                "height".into(),
                "width".into(),
                "channel".into(),
            ],
            observation_ids: vec!["o0".into(), "o1".into(), "o2".into(), "o3".into()],
            uri: uri.clone(),
        });
        let e = tensor.manifest_entry("train", "train/img");
        assert_eq!(e.storage, PayloadStorageKind::Uri);
        assert_eq!(e.uri.as_deref(), Some("s3://bucket/cube.npy"));
        assert_eq!(e.content_hash, uri.content_hash);
        assert_eq!(e.byte_len, uri.byte_len);
        assert_eq!(e.representation_id.as_deref(), Some("rgb_image"));

        // A package carrying it summarizes without embedding any tensor bytes.
        let mut pkg = DatasetPackage::from_assembled(&assembled(base_block())).unwrap();
        pkg.partitions["train"]
            .payloads
            .push(("train/img".into(), tensor));
        let summary = pkg.to_canonical_summary();
        assert!(summary.contains("s3://bucket/cube.npy"));
        assert!(summary.contains("\"uri\""));
        assert!(!summary.contains("\"data\""));
    }

    #[test]
    fn spectral_record_set_manifest_hashes_inline_records() {
        let records = PayloadBlock::SpectralRecordSet(SpectralRecordSet {
            records: vec![
                json!({
                    "metadata": {"sample_id": "s1"},
                    "signals": {"absorbance": {"values": [0.1, 0.2]}}
                }),
                json!({
                    "metadata": {"sample_id": "s2"},
                    "signals": {"absorbance": {"values": [0.3, 0.4]}}
                }),
            ],
        });
        let entry = records.manifest_entry("train", "train/records");
        assert_eq!(entry.role, "features");
        assert_eq!(entry.payload_kind, "spectral_record_set");
        assert_eq!(entry.dtype, "records");
        assert_eq!(entry.shape, vec![2]);
        assert_eq!(entry.axes, vec!["sample"]);
        assert_eq!(entry.storage, PayloadStorageKind::Inline);
        assert!(entry.byte_len > 0);

        let tampered = PayloadBlock::SpectralRecordSet(SpectralRecordSet {
            records: vec![json!({
                "metadata": {"sample_id": "s1"},
                "signals": {"absorbance": {"values": [0.1, 0.9]}}
            })],
        });
        let tampered_entry = tampered.manifest_entry("train", "train/records");
        assert_ne!(entry.content_hash, tampered_entry.content_hash);
    }

    #[test]
    fn uri_backed_declared_variants_have_distinct_manifest_shapes() {
        let sequence_uri = uri_ref(
            "series",
            "float32",
            vec![2, 10, 3],
            vec!["sample", "time", "channel"],
        );
        let genotype_uri = uri_ref("dosage", "uint8", vec![2, 4], vec!["sample", "variant"]);
        let mask_uri = uri_ref(
            "mask",
            "bool",
            vec![2, 64, 64],
            vec!["sample", "height", "width"],
        );
        let bare_uri = uri_ref(
            "cube",
            "uint16",
            vec![2, 8, 8, 16],
            vec!["sample", "height", "width", "band"],
        );

        let cases = vec![
            (
                PayloadBlock::SequenceBlock(SequenceBlock {
                    representation_id: "series_mv".into(),
                    dtype: "float32".into(),
                    n_channels: 3,
                    lengths: vec![10, 10],
                    uri: sequence_uri.clone(),
                }),
                "sequence_block",
                Some("series_mv"),
                "features",
                vec![2, 10, 3],
                vec!["sample", "time", "channel"],
                sequence_uri,
            ),
            (
                PayloadBlock::GenotypeMatrix(GenotypeMatrix {
                    representation_id: "dosage_matrix".into(),
                    n_variants: 4,
                    n_samples: 2,
                    encoding: "dosage".into(),
                    uri: genotype_uri.clone(),
                }),
                "genotype_matrix",
                Some("dosage_matrix"),
                "features",
                vec![2, 4],
                vec!["sample", "variant"],
                genotype_uri,
            ),
            (
                PayloadBlock::MaskBlock(MaskBlock {
                    representation_id: "segmentation_mask".into(),
                    dtype: "bool".into(),
                    shape: vec![2, 64, 64],
                    uri: mask_uri.clone(),
                }),
                "mask_block",
                Some("segmentation_mask"),
                "mask",
                vec![2, 64, 64],
                vec!["sample", "height", "width"],
                mask_uri,
            ),
            (
                PayloadBlock::UriBackedPayload(bare_uri.clone()),
                "uri_backed",
                None,
                "features",
                vec![2, 8, 8, 16],
                vec!["sample", "height", "width", "band"],
                bare_uri,
            ),
        ];

        for (payload, kind, repr, role, shape, axes, uri) in cases {
            let entry = payload.manifest_entry("train", kind);
            assert_eq!(entry.payload_kind, kind);
            assert_eq!(entry.representation_id.as_deref(), repr);
            assert_eq!(entry.role, role);
            assert_eq!(entry.shape, shape);
            assert_eq!(entry.axes, axes);
            assert_eq!(entry.storage, PayloadStorageKind::Uri);
            assert_eq!(entry.uri.as_deref(), Some(uri.uri.as_str()));
            assert_eq!(entry.content_hash, uri.content_hash);
            assert_eq!(entry.byte_len, uri.byte_len);
            assert_eq!(entry.codec, uri.codec);
        }
    }
}
