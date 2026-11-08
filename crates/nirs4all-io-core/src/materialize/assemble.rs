// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Assemble a DatasetSpec into a target-agnostic dataset (ports `materialize/assemble.py`).
//!
//! Per source: load → merge multi-file → split columns by role. Then per
//! partition: align feature sources (multi-source X), join targets/metadata/
//! lookups onto the sample rows, collect X/y/metadata/weights. The result is an
//! `AssembledDataset` IR that `to_spectrodataset`/`to_dag_ml_data` adapt — kept
//! target-agnostic so assembly is testable without nirs4all.
//!
//! This module is fs-free. Inputs arrive as named in-memory payloads
//! ([`InMemorySource`]): tabular bytes the facade already read (and decompressed)
//! or pre-decoded `nirs4all-formats` records. The facade resolves paths, reads
//! files, reads index files and fold files, then delegates to
//! [`assemble_in_memory`] so a single assembly core backs both the native (path)
//! and the browser (no-fs) paths (D-R4 / D-R7).

use std::collections::{HashMap, HashSet};

use crate::conventions::engine::get_stem;
use crate::spec::dataset_spec::{
    AggregateSpec, DatasetSpec, JoinSpec, LoadingParams, NaConfig, SourceSpec, VariationSpec,
};
use crate::spec::enums::{
    Cardinality, Coverage, MergeMode, NaPolicy, PartitionBy, Role, SourceKind, UnknownPolicy,
};
use crate::spec::selectors::Selector;
use crate::spec::SpecError;
use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::folds::Fold;
use super::frame::{Cell, Column, Frame, JoinKey, Matrix};
use super::join::{
    concat_features_with_limits, concat_samples_with_limits, join_tables_with_limits,
    merge_by_key_with_limits,
};
use super::loaders::{
    apply_na_policy, effective_params, is_comma_pair, read_sheet, read_table_bytes,
    read_table_bytes_with_limits, refuse_undeclared_delimiter, Sheet, TabularReadLimits,
};
use super::numeric::{NumericStorageOptions, PrecisionRecord};

/// A source payload supplied in-memory instead of a file path.
///
/// Adding `Rows` is a breaking Rust source API change for external exhaustive
/// matches. It changes no C ABI, JSON transport or artifact schema.
pub enum SourcePayload {
    /// Raw CSV/tabular bytes the formats layer won't decode (y/metadata tables).
    Bytes(Vec<u8>),
    /// A facade-decoded typed table for formats that need filesystem-backed
    /// native loaders (for example Parquet).
    Frame(Frame),
    /// A facade-decoded spreadsheet; each source selects its worksheet.
    Sheets(Vec<Sheet>),
    /// Explicit scalar row objects, distinct from decoded Formats records.
    /// Columns follow first-seen key order; under `preserve_order`, different
    /// input key orders can produce different tables and fingerprints. Declare
    /// `format.columns` to fix column order. `tabular_limits=None` is uncapped.
    Rows(Vec<Value>),
    /// Pre-decoded nirs4all-formats records (signals[].values=X, targets{}, metadata{}).
    /// Plain row objects and malformed sections are refused, never reinterpreted.
    Records(Vec<Value>),
}

/// A named in-memory source. `name` matches a `source.input` string / list /
/// glob-expansion entry (exact name first; otherwise a pattern, or a non-pattern file-stem fallback).
pub struct InMemorySource {
    pub name: String,
    pub payload: SourcePayload,
}

fn row_idx_col(source_id: &str) -> String {
    format!("__src_row_idx__{source_id}__")
}

#[derive(Clone)]
enum NumericScope {
    Uniform(char),
    Rows(Vec<char>),
    Columns(IndexMap<String, char>),
}

impl NumericScope {
    fn decimal(&self, column: &str, row: usize) -> char {
        match self {
            Self::Uniform(decimal) => *decimal,
            Self::Rows(rows) => rows.get(row).copied().unwrap_or('.'),
            Self::Columns(columns) => columns.get(column).copied().unwrap_or('.'),
        }
    }

    fn inputs(
        frames: &[Frame],
        decimals: &[char],
        merge: MergeMode,
        names: &[String],
        keys: &[String],
    ) -> Self {
        let first = decimals.first().copied().unwrap_or('.');
        if decimals.iter().all(|decimal| *decimal == first) {
            return Self::Uniform(first);
        }
        if merge == MergeMode::ConcatSamples {
            return Self::Rows(
                frames
                    .iter()
                    .zip(decimals)
                    .flat_map(|(frame, decimal)| std::iter::repeat_n(*decimal, frame.n_rows))
                    .collect(),
            );
        }
        let mut columns = IndexMap::new();
        for (index, (frame, decimal)) in frames.iter().zip(decimals).enumerate() {
            for column in &frame.columns {
                if index > 0 && keys.contains(&column.name) {
                    continue;
                }
                let name = if columns.contains_key(&column.name) {
                    if merge == MergeMode::ConcatFeatures && keys.is_empty() {
                        format!("{}__{}", names[index], column.name)
                    } else {
                        format!("{}__{}", column.name, names[index])
                    }
                } else {
                    column.name.clone()
                };
                columns.insert(name, *decimal);
            }
        }
        Self::Columns(columns)
    }
}

fn input_decimal(source: &InMemorySource, params: &LoadingParams) -> char {
    if matches!(
        &source.payload,
        SourcePayload::Bytes(_) | SourcePayload::Rows(_)
    ) {
        params
            .decimal_separator
            .as_deref()
            .unwrap_or(".")
            .chars()
            .next()
            .unwrap_or('.')
    } else {
        '.'
    }
}

/// Actual loaded geometry stays internal until selected feature columns are known.
struct LoadedSource {
    frame: Frame,
    numeric_scope: NumericScope,
    header_unit: String,
    signal_type: Option<String>,
    origins: Option<Vec<String>>,
    raw_unit: Option<String>,
    raw_coordinates: Option<IndexMap<String, Value>>,
}

/// Version of the JSON-compatible `AssembledDataset` summary/full wire.
///
/// Version 1 was unversioned and did not preserve source ids, scientific
/// identity, or stable fold provenance. Consumers must reject it rather than
/// silently interpreting it as this version.
pub const ASSEMBLED_DATASET_VERSION: u32 = 3;

#[derive(Debug, Clone, Default)]
pub struct PartitionBlock {
    pub n_samples: usize,
    /// Feature-source ids aligned with `x` and its per-source layout fields.
    pub source_ids: Vec<String>,
    pub x: Vec<Matrix>,
    pub feature_headers: Vec<Vec<String>>,
    pub feature_names: Vec<Option<Vec<String>>>,
    pub feature_axes: Vec<Vec<Value>>,
    pub representation_ids: Vec<String>,
    pub header_units: Vec<String>,
    pub signal_types: Vec<Option<String>>,
    pub processings: Vec<Vec<(String, Matrix)>>,
    pub y: Option<Matrix>,
    pub y_headers: Vec<String>,
    pub y_categorical: IndexMap<String, Value>,
    pub metadata: Option<Frame>,
    pub weights: Option<Vec<f64>>,
    pub weights_validity: Option<Vec<bool>>,
    pub weights_precision: super::numeric::PrecisionRecord,
    pub weights_header: Option<String>,
}

/// Column-level scientific identity provenance retained from `sample_index`.
///
/// The assembled matrices do not otherwise retain the `DatasetSpec`, so the
/// dag-ml bridge needs these names to recover the corresponding, sample-aligned
/// metadata values without guessing from row order. `sample_id` is only set for
/// a scalar `sample_index.key`; composite keys are deliberately left unsupported
/// at this v1 seam rather than serialised into an invented identifier.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdentityProvenance {
    /// Ordered feature-source ids contributing to every assembled row.
    pub source_ids: Vec<String>,
    pub sample_id: Option<String>,
    pub observation_id: Option<String>,
    pub repetition_id: Option<String>,
    pub group_id: Option<String>,
}

/// Fold membership translated to stable observation ids before partitioning can
/// reorder assembled rows. Empty provenance with non-empty positional folds
/// means the v1 assembler could not establish that translation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoldProvenance {
    pub train_observation_ids: Vec<String>,
    pub validation_observation_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AssembledDataset {
    pub name: String,
    pub task_type: String,
    pub signal_type: String,
    pub n_sources: usize,
    pub blocks: IndexMap<String, PartitionBlock>,
    pub folds: Vec<Fold>,
    pub fold_provenance: Vec<FoldProvenance>,
    pub repetition: Option<String>,
    /// Scientific identity columns carried from `DatasetSpec.sample_index`.
    pub identity: IdentityProvenance,
    pub aggregate: Option<AggregateSpec>,
    pub warnings: Vec<String>,
    pub audits: Vec<Value>,
}

impl AssembledDataset {
    /// A canonical structural summary (shapes, headers, partitions, rounded
    /// X/y values, folds) for cross-language golden comparison. The explicit
    /// `assembled_schema_version` makes this public CLI/C-ABI artifact distinct
    /// from the `DatasetSpec` schema and rejects the former unversioned wire.
    /// Mirrors
    /// `tests/test_assembled_goldens.py`.
    pub fn to_summary_value(&self) -> Value {
        use crate::pyfmt::py_round;
        let mat_rows = |m: &Matrix| -> Value {
            Value::Array(
                (0..m.n_rows)
                    .map(|r| {
                        Value::Array(
                            m.data[r * m.n_cols..(r + 1) * m.n_cols]
                                .iter()
                                .map(|&v| Value::from(py_round(v, 4)))
                                .collect(),
                        )
                    })
                    .collect(),
            )
        };
        let mut blocks = Map::new();
        for (part, b) in &self.blocks {
            blocks.insert(
                part.clone(),
                serde_json::json!({
                    "n_samples": b.n_samples,
                    "source_ids": b.source_ids,
                    "x_shapes": b.x.iter().map(|m| vec![m.n_rows, m.n_cols]).collect::<Vec<_>>(),
                    "x": b.x.iter().map(mat_rows).collect::<Vec<_>>(),
                    "feature_headers": b.feature_headers,
                    "header_units": b.header_units,
                    "signal_types": b.signal_types,
                    "y_shape": b.y.as_ref().map(|m| vec![m.n_rows, m.n_cols]),
                    "y": b.y.as_ref().map(mat_rows),
                    "y_headers": b.y_headers,
                    "y_categorical": b.y_categorical,
                    "metadata_columns": b.metadata.as_ref().map(|f| f.column_names()).unwrap_or_default(),
                    "weights_header": b.weights_header,
                    "processings": b.processings.iter().map(|src| src.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()).collect::<Vec<_>>(),
                }),
            );
        }
        let aggregate = self.aggregate.as_ref().map(|a| {
            serde_json::json!({
                "by": a.by,
                "method": a.method.value(),
                "exclude_outliers": a.exclude_outliers,
                "outlier_threshold": a.outlier_threshold,
            })
        });
        serde_json::json!({
            "assembled_schema_version": ASSEMBLED_DATASET_VERSION,
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "blocks": Value::Object(blocks),
            "folds": self.folds.iter().map(|(tr, vl)| vec![tr.clone(), vl.clone()]).collect::<Vec<_>>(),
            "fold_provenance": self.fold_provenance.iter().map(|fold| serde_json::json!({
                "train_observation_ids": fold.train_observation_ids,
                "validation_observation_ids": fold.validation_observation_ids,
            })).collect::<Vec<_>>(),
            "identity": { "provenance": identity_provenance_value(&self.identity) },
            "repetition": self.repetition,
            "aggregate": aggregate,
        })
    }

    /// Full-precision export for the binding-side SpectroDataset adapter. Unlike
    /// `to_summary_value` (a rounded structural fingerprint), this carries the
    /// complete X/y/processing matrices (flat f64 + shape), metadata column
    /// values, weights, and folds — everything `to_spectrodataset` needs to
    /// rebuild a `SpectroDataset` from the materialized data.
    pub fn to_full_value(&self) -> Result<Value, String> {
        let package = super::package::DatasetPackage::from_assembled(self)?;
        let normalized = package.to_assembled();
        let mut blocks = Map::new();
        for (part, b) in &normalized.blocks {
            let target = package.partitions[part]
                .payloads
                .iter()
                .find_map(|(_, p)| match p {
                    super::package::PayloadBlock::TargetTable(t) => Some(t),
                    _ => None,
                });
            let weight = package.partitions[part]
                .payloads
                .iter()
                .find_map(|(_, p)| match p {
                    super::package::PayloadBlock::Weights(w) => Some(w),
                    _ => None,
                });
            blocks.insert(
                part.clone(),
                serde_json::json!({
                    "n_samples": b.n_samples,
                    "source_ids": b.source_ids,
                    "x": b.x.iter().map(matrix_full).collect::<Vec<_>>(),
                    "feature_headers": b.feature_headers,
                    "feature_names": b.feature_names,
                    "feature_axes": b.feature_axes.iter().map(|axes|super::projection::axes_full(axes)).collect::<Result<Vec<_>,_>>()?,
                    "representation_ids": b.representation_ids,
                    "header_units": b.header_units,
                    "signal_types": b.signal_types,
                    "y": target.map(|t| serde_json::json!({"columns":super::projection::target_columns(t)})),
                    "y_headers": b.y_headers,
                    "y_categorical": b.y_categorical,
                    "metadata": b.metadata.as_ref().map(frame_full),
                    "weights": weight.map(|w| { let mut matrix=Matrix::new(w.values.clone(),w.values.len(),1);matrix.validity=w.validity.clone();matrix.precision=w.precision.clone();matrix.full_value() }),
                    "weights_header": b.weights_header,
                    "processings": b.processings.iter().map(|src| {
                        src.iter()
                            .map(|(name, m)| serde_json::json!({"name": name, "matrix": matrix_full(m)}))
                            .collect::<Vec<_>>()
                    }).collect::<Vec<_>>(),
                }),
            );
        }
        let aggregate = self.aggregate.as_ref().map(|a| {
            serde_json::json!({
                "by": a.by,
                "method": a.method.value(),
                "exclude_outliers": a.exclude_outliers,
                "outlier_threshold": a.outlier_threshold,
            })
        });
        Ok(serde_json::json!({
            "partition_order":self.blocks.keys().collect::<Vec<_>>(),
            "assembled_schema_version": ASSEMBLED_DATASET_VERSION,
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "blocks": Value::Object(blocks),
            "folds": self.folds.iter().map(|(tr, vl)| vec![tr.clone(), vl.clone()]).collect::<Vec<_>>(),
            "fold_provenance": self.fold_provenance.iter().map(|fold| serde_json::json!({
                "train_observation_ids": fold.train_observation_ids,
                "validation_observation_ids": fold.validation_observation_ids,
            })).collect::<Vec<_>>(),
            "identity": { "provenance": identity_provenance_value(&self.identity) },
            "repetition": self.repetition,
            "aggregate": aggregate,
            "warnings": self.warnings,
            "audits": self.audits,
        }))
    }
}

fn matrix_full(m: &Matrix) -> Value {
    m.full_value()
}

pub fn identity_provenance_value(identity: &IdentityProvenance) -> Value {
    serde_json::json!({
        "source_ids": identity.source_ids,
        "sample_id": identity.sample_id,
        "observation_id": identity.observation_id,
        "repetition_id": identity.repetition_id,
        "group_id": identity.group_id,
    })
}

fn frame_full(f: &Frame) -> Value {
    let columns: Vec<Value> = f
        .columns
        .iter()
        .map(|c| serde_json::json!({"name": c.name, "values": c.values.iter().enumerate().map(|(row, v)| super::frame::metadata_cell_value(c, row, v)).collect::<Vec<_>>()}))
        .collect();
    serde_json::json!({ "n_rows": f.n_rows, "columns": columns })
}

struct SourceTable {
    source_id: String,
    df: Frame,
    numeric_scope: NumericScope,
    /// column -> role, in `_split_roles` order.
    roles: Vec<(String, String)>,
    key: Option<Vec<String>>,
    partition: Option<String>,
    kind: String,
    join: Option<JoinSpec>,
    signal_type: Option<String>,
    header_unit: String,
    raw_unit: Option<String>,
    raw_coordinates: Option<IndexMap<String, Value>>,
    variations: Vec<(String, Matrix)>,
    /// Original source positions to remove, applied only after all sources align.
    remove_rows: Option<Vec<bool>>,
}

fn missing_rows(frame: &Frame) -> Vec<bool> {
    (0..frame.n_rows)
        .map(|row| {
            frame.columns.iter().any(|col| {
                matches!(&col.values[row], Cell::Na)
                    || matches!(&col.values[row], Cell::Float(v) if v.is_nan())
            })
        })
        .collect()
}

fn defer_sample_removal(params: &mut LoadingParams) {
    if params.na.policy == NaPolicy::RemoveSample {
        params.na.policy = NaPolicy::Ignore;
    }
}

/// Apply a source's NA policy once its column roles are known.
///
/// Loading is lossless for targets and metadata: a missing value stays NaN and
/// its sample is kept, because training and scoring exclude target-missing
/// samples downstream. Only `remove_sample` removes such a sample and only
/// `abort` refuses a missing target; a target is never imputed, dropped as a
/// column or encoded as a class. Every other column follows the policy
/// (`auto` = `abort`).
///
/// Returns the source rows `remove_sample` removes once all sources are joined.
fn apply_role_na_policy(
    df: &mut Frame,
    roles: &mut Vec<(String, String)>,
    na: &NaConfig,
) -> Result<Option<Vec<bool>>, SpecError> {
    match na.policy {
        NaPolicy::Ignore => return Ok(None),
        NaPolicy::RemoveSample => return Ok(Some(missing_rows(df))),
        _ => {}
    }
    let (target_idx, policy_idx): (Vec<usize>, Vec<usize>) = {
        let role_of: HashMap<&str, &str> = roles
            .iter()
            .map(|(column, role)| (column.as_str(), role.as_str()))
            .collect();
        let role = |idx: &usize| role_of.get(df.columns[*idx].name.as_str()).copied();
        let indices = 0..df.columns.len();
        (
            indices
                .clone()
                .filter(|idx| role(idx) == Some("targets"))
                .collect(),
            indices
                .filter(|idx| !matches!(role(idx), Some("targets" | "metadata")))
                .collect(),
        )
    };
    let sub_frame = |idx: &[usize]| Frame {
        columns: idx.iter().map(|&i| df.columns[i].clone()).collect(),
        n_rows: df.n_rows,
        header_unit: df.header_unit.clone(),
    };

    if na.policy == NaPolicy::Abort {
        apply_na_policy(&sub_frame(&target_idx), na)?;
    }
    let treated = apply_na_policy(&sub_frame(&policy_idx), na)?;
    let mut treated: HashMap<String, Column> = treated
        .columns
        .into_iter()
        .map(|column| (column.name.clone(), column))
        .collect();
    let policy_idx: HashSet<usize> = policy_idx.into_iter().collect();
    let mut dropped: HashSet<String> = HashSet::new();
    let mut columns = Vec::with_capacity(df.columns.len());
    for (idx, column) in std::mem::take(&mut df.columns).into_iter().enumerate() {
        if !policy_idx.contains(&idx) {
            columns.push(column);
        } else if let Some(treated) = treated.remove(&column.name) {
            columns.push(treated);
        } else {
            dropped.insert(column.name);
        }
    }
    df.columns = columns;
    roles.retain(|(column, _)| !dropped.contains(column));
    Ok(None)
}

fn key_list(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => {
            let ks: Vec<String> = a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            if ks.is_empty() {
                None
            } else {
                Some(ks)
            }
        }
        _ => None,
    }
}

fn is_glob(text: &str) -> bool {
    text.chars().any(|c| matches!(c, '*' | '?' | '['))
}

/// Filename portion of a name (path-separator agnostic).
fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// Pure glob match (`*`, `?`, `[...]`) of `pattern` against `text`.
fn glob_match(pattern: &str, text: &str) -> bool {
    fn matches(p: &[char], t: &[char]) -> bool {
        let mut pi = 0;
        let mut ti = 0;
        let mut star: Option<(usize, usize)> = None;
        while ti < t.len() {
            if pi < p.len() {
                match p[pi] {
                    '*' => {
                        star = Some((pi, ti));
                        pi += 1;
                        continue;
                    }
                    '?' => {
                        pi += 1;
                        ti += 1;
                        continue;
                    }
                    '[' => {
                        if let Some((next_pi, ok)) = match_class(p, pi, t[ti]) {
                            if ok {
                                pi = next_pi;
                                ti += 1;
                                continue;
                            }
                        }
                    }
                    c if c == t[ti] => {
                        pi += 1;
                        ti += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            if let Some((sp, st)) = star {
                pi = sp + 1;
                ti = st + 1;
                star = Some((sp, st + 1));
            } else {
                return false;
            }
        }
        while pi < p.len() && p[pi] == '*' {
            pi += 1;
        }
        pi == p.len()
    }
    fn match_class(p: &[char], start: usize, ch: char) -> Option<(usize, bool)> {
        let mut i = start + 1;
        let negate = i < p.len() && (p[i] == '!' || p[i] == '^');
        if negate {
            i += 1;
        }
        let mut matched = false;
        let mut first = true;
        while i < p.len() {
            if p[i] == ']' && !first {
                return Some((i + 1, matched != negate));
            }
            first = false;
            if i + 2 < p.len() && p[i + 1] == '-' && p[i + 2] != ']' {
                if ch >= p[i] && ch <= p[i + 2] {
                    matched = true;
                }
                i += 3;
            } else {
                if ch == p[i] {
                    matched = true;
                }
                i += 1;
            }
        }
        None
    }
    let pc: Vec<char> = pattern.chars().collect();
    let tc: Vec<char> = text.chars().collect();
    matches(&pc, &tc)
}

/// Resolve `input` against the named in-memory sources (replaces `resolve_inputs`).
///
/// Match exact names first. Without an exact name, patterns use glob matching
/// and non-patterns use the existing unique file-stem fallback:
/// - A glob whose pattern contains a path separator matches against the full
///   source name; a separator-free glob matches against the source basename. This
///   mirrors the native `glob(base_dir.join(text))` path-scoping, so a feature
///   `spectra/*.csv` no longer also matches a sibling `meta.csv`. Glob matches are
///   deduped by resolved name (path), and sorted for stable order.
/// - Exact-name and file-stem fallbacks error on ambiguous duplicates instead of
///   silently picking the first source supplied.
fn resolve_inputs_mem<'a>(
    input: &Value,
    sources: &'a [InMemorySource],
) -> Result<Vec<&'a InMemorySource>, SpecError> {
    let items: Vec<&Value> = match input {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut out: Vec<&InMemorySource> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for item in items {
        let Some(text) = item.as_str() else { continue };
        if let Some(s) = unique_match(text, sources, |s| s.name == text, "exact name")? {
            if seen.insert(s.name.as_str()) {
                out.push(s);
            }
        } else if is_glob(text) {
            let path_scoped = text.contains(['/', '\\']);
            let mut matched: Vec<&InMemorySource> = sources
                .iter()
                .filter(|s| {
                    if path_scoped {
                        glob_match(text, &s.name)
                    } else {
                        glob_match(text, basename(&s.name))
                    }
                })
                .collect();
            matched.sort_by(|a, b| a.name.cmp(&b.name));
            for s in matched {
                if seen.insert(s.name.as_str()) {
                    out.push(s);
                }
            }
        } else if let Some(s) = unique_match(
            text,
            sources,
            |s| get_stem(basename(&s.name)) == get_stem(basename(text)),
            "file-stem",
        )? {
            if seen.insert(s.name.as_str()) {
                out.push(s);
            }
        }
    }
    Ok(out)
}

/// Find the single source satisfying `pred`; error if more than one matches so an
/// ambiguous `input` (e.g. `scan` matching both `scan.csv` and `scan.asd`) is a
/// hard error rather than silent first-wins (Codex #4).
fn unique_match<'a>(
    text: &str,
    sources: &'a [InMemorySource],
    pred: impl Fn(&&'a InMemorySource) -> bool,
    how: &str,
) -> Result<Option<&'a InMemorySource>, SpecError> {
    let mut matched = sources.iter().filter(|s| pred(s));
    let Some(first) = matched.next() else {
        return Ok(None);
    };
    if let Some(second) = matched.next() {
        let mut names = vec![first.name.clone(), second.name.clone()];
        names.extend(matched.map(|s| s.name.clone()));
        return Err(SpecError::new(format!(
            "input '{text}' is ambiguous: matches {how} of multiple sources {names:?}"
        )));
    }
    Ok(Some(first))
}

/// Decode one in-memory source into a [`Frame`].
fn frame_from_source(
    src: &InMemorySource,
    source_id: &str,
    params: &LoadingParams,
    tabular_limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    match &src.payload {
        SourcePayload::Bytes(bytes) => {
            refuse_undeclared_delimiter(bytes, params, source_id, &src.name)?;
            match tabular_limits {
                Some(limits) => read_table_bytes_with_limits(bytes, params, limits),
                None => read_table_bytes(bytes, params),
            }
        }
        SourcePayload::Frame(frame) => {
            if let Some(limits) = tabular_limits {
                limits.check_shape(frame.n_rows as u64, frame.columns.len() as u64)?;
            }
            frame_with_params(frame, params)
        }
        SourcePayload::Sheets(sheets) => read_sheet(sheets, params, tabular_limits),
        SourcePayload::Rows(rows) => rows_to_frame(rows, params, tabular_limits),
        SourcePayload::Records(records) => records_to_frame(records, params, tabular_limits),
    }
}

fn frame_requested_columns(params: &LoadingParams) -> Result<Option<Vec<String>>, SpecError> {
    match params.format.values.get("columns") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) => Ok(Some(vec![name.clone()])),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_string).ok_or_else(|| {
                    SpecError::new("loading.format.columns must be a string or a list of strings")
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(SpecError::new(
            "loading.format.columns must be a string or a list of strings",
        )),
    }
}

fn frame_with_params(frame: &Frame, params: &LoadingParams) -> Result<Frame, SpecError> {
    let mut out = if let Some(columns) = frame_requested_columns(params)? {
        for name in &columns {
            if !frame.has_column(name) {
                return Err(SpecError::new(format!(
                    "column name '{name}' not found; available: {:?}",
                    frame.column_names()
                )));
            }
        }
        frame.select(&columns)
    } else {
        frame.clone()
    };
    if let Some(unit) = params.header_unit {
        out.header_unit = unit.value().to_string();
    }
    apply_na_policy(&out, &params.na)
}

fn load_source_frame_mem(
    source: &SourceSpec,
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    audits: &mut Vec<Value>,
    tabular_limits: Option<TabularReadLimits>,
) -> Result<LoadedSource, SpecError> {
    let mut params = effective_params(&spec.params, &source.params);
    // The NA policy depends on each column's role, so it is applied once roles
    // are split (`apply_role_na_policy`); rows are removed only after joining.
    params.na.policy = NaPolicy::Ignore;
    let matched = resolve_inputs_mem(&source.input, sources)?;
    if matched.is_empty() {
        return Err(SpecError::new(format!(
            "source '{}': no input files resolved from {}",
            source.id,
            crate::pyfmt::py_repr(&source.input)
        )));
    }
    let frames: Vec<Frame> = matched
        .iter()
        .map(|s| frame_from_source(s, &source.id, &params, tabular_limits))
        .collect::<Result<_, _>>()?;
    for frame in &frames {
        if let Some(report) = super::loaders::ignored_input_na_report(frame) {
            audits.push(serde_json::json!({"source": source.id, "na": report}));
        }
    }
    let mut raw_axes: Option<Value> = None;
    for input in &matched {
        if let SourcePayload::Records(records) = &input.payload {
            for record in records {
                if let Some((_, signal)) = record
                    .get("signals")
                    .and_then(Value::as_object)
                    .and_then(|signals| {
                        pick_signal(
                            signals,
                            params
                                .format
                                .values
                                .get("signal")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                        )
                    })
                {
                    let axis = serde_json::json!({"unit":signal["axis"]["unit"],"coordinates":signal["axis"]["values"]});
                    for coordinate in axis["coordinates"].as_array().into_iter().flatten() {
                        if coordinate.is_number() {
                            super::numeric::exact_json_number_f64(coordinate)
                                .map_err(SpecError::new)?;
                        }
                    }
                    if raw_axes.as_ref().is_some_and(|known| {
                        known["unit"] != axis["unit"]
                            || match (
                                known["coordinates"].as_array(),
                                axis["coordinates"].as_array(),
                            ) {
                                (Some(left), Some(right)) => {
                                    left.len() != right.len()
                                        || left.iter().zip(right).any(|(a, b)| {
                                            match (a.as_f64(), b.as_f64()) {
                                                (Some(a), Some(b)) => a.to_bits() != b.to_bits(),
                                                _ => a != b,
                                            }
                                        })
                                }
                                _ => known["coordinates"] != axis["coordinates"],
                            }
                    }) {
                        return Err(SpecError::new("Decoded feature axes differ across records"));
                    }
                    raw_axes = Some(axis);
                }
            }
        }
    }
    let raw_unit = params
        .header_unit
        .map(|unit| unit.value().to_owned())
        .or_else(|| {
            raw_axes.as_ref().and_then(|axis| {
                axis["unit"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
            })
        });
    let raw_coordinates = raw_axes.as_ref().map(|axis| {
        axis["coordinates"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
    });
    let header_unit = frames[0].header_unit.clone();
    let signal = params.signal_type.map(|s| s.value().to_string());
    let names: Vec<String> = matched
        .iter()
        .map(|s| get_stem(basename(&s.name)))
        .collect();
    let numeric_scope = NumericScope::inputs(
        &frames,
        &matched
            .iter()
            .map(|input| input_decimal(input, &params))
            .collect::<Vec<_>>(),
        source.merge,
        &names,
        &key_list(&source.key).unwrap_or_default(),
    );
    let (df, origins): (Frame, Option<Vec<String>>) = if frames.len() == 1 {
        (frames[0].clone(), None)
    } else {
        match source.merge {
            MergeMode::ConcatSamples => {
                let (df, origins, a) = concat_samples_with_limits(&frames, &names, tabular_limits)?;
                audits.push(serde_json::json!({"source": source.id, "merge": a.operation, "warnings": a.warnings}));
                (df, Some(origins))
            }
            MergeMode::ConcatFeatures => {
                let (df, a) = concat_features_with_limits(
                    &frames,
                    &names,
                    key_list(&source.key).as_deref(),
                    tabular_limits,
                )?;
                audits.push(serde_json::json!({"source": source.id, "merge": a.operation, "warnings": a.warnings}));
                (df, None)
            }
            MergeMode::ByKey => {
                let key = key_list(&source.key).ok_or_else(|| {
                    SpecError::new(format!(
                        "source '{}': merge='by_key' requires a 'key'",
                        source.id
                    ))
                })?;
                let (df, a) = merge_by_key_with_limits(
                    &frames,
                    &names,
                    &key,
                    Coverage::Complete,
                    tabular_limits,
                )?;
                audits.push(serde_json::json!({"source": source.id, "merge": a.operation, "warnings": a.warnings}));
                (df, None)
            }
            MergeMode::None => {
                return Err(SpecError::new(format!(
                    "source '{}': multi-file input requires a merge mode",
                    source.id
                )))
            }
        }
    };
    // Merge results preserve column order: horizontal merges append right
    // non-key columns, while sample concatenation takes a schema union. Pair
    // geometry with that actual output order rather than rebuilding its names.
    let raw_coordinates = raw_coordinates
        .map(|coordinates| {
            let selected: Vec<Option<Value>> = if frames.len() > 1
                && matches!(source.merge, MergeMode::ConcatFeatures | MergeMode::ByKey)
            {
                let key = key_list(&source.key).unwrap_or_default();
                frames
                    .iter()
                    .enumerate()
                    .flat_map(|(index, frame)| {
                        let key = &key;
                        frame
                            .columns
                            .iter()
                            .enumerate()
                            .filter_map(move |(column, c)| {
                                if index > 0 && key.contains(&c.name) {
                                    None
                                } else {
                                    Some(coordinates.get(column).cloned())
                                }
                            })
                    })
                    .collect()
            } else {
                df.columns
                    .iter()
                    .map(|column| {
                        frames[0]
                            .columns
                            .iter()
                            .position(|original| original.name == column.name)
                            .and_then(|index| coordinates.get(index))
                            .cloned()
                    })
                    .collect()
            };
            if selected.len() != df.columns.len() {
                return Err(SpecError::new(
                    "Raw feature geometry lost column alignment during merge",
                ));
            }
            let mut mapped = IndexMap::new();
            for (column, coordinate) in df.columns.iter().zip(selected) {
                if let Some(coordinate) = coordinate {
                    if mapped.contains_key(&column.name) {
                        return Err(SpecError::new(
                            "Decoded feature column names collide before geometry insertion",
                        ));
                    }
                    mapped.insert(column.name.clone(), coordinate);
                }
            }
            Ok(mapped)
        })
        .transpose()?;
    Ok(LoadedSource {
        frame: df,
        numeric_scope,
        header_unit,
        signal_type: signal,
        origins,
        raw_unit,
        raw_coordinates,
    })
}

/// Admit the entire rectangular row shape before allocating cells or columns.
fn rows_to_frame(
    rows: &[Value],
    params: &LoadingParams,
    limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    if let Some(limits) = limits {
        limits.check_shape(rows.len() as u64, 0)?;
    }
    if rows.is_empty() {
        return Err(SpecError::new("Rows requires at least one row"));
    }
    let mut columns: IndexMap<String, (bool, bool)> = IndexMap::new();
    for (index, row) in rows.iter().enumerate() {
        let row = row
            .as_object()
            .ok_or_else(|| SpecError::new(format!("Rows row {index} must be an object")))?;
        let mut record_bytes = 0_u64;
        for (name, value) in row {
            let context = || format!("Rows row {index}, column '{name}'");
            if name.is_empty() {
                return Err(SpecError::new(format!(
                    "{}: empty column names are not supported",
                    context()
                )));
            }
            let field_bytes = match value {
                Value::String(text) => text.len() as u64,
                Value::Null | Value::Bool(_) => 8,
                Value::Number(number) => {
                    if number.as_i64().is_none()
                        && (number.is_u64()
                            || number.as_f64().is_none_or(|number| !number.is_finite()))
                    {
                        return Err(SpecError::new(format!(
                            "{}: expected an i64 integer or finite float",
                            context()
                        )));
                    }
                    8
                }
                Value::Array(_) | Value::Object(_) => {
                    return Err(SpecError::new(format!(
                        "{}: nested cells are not supported",
                        context()
                    )));
                }
            };
            if let Some(limits) = limits {
                if name.len() as u64 > limits.max_field_bytes
                    || matches!(value, Value::String(_)) && field_bytes > limits.max_field_bytes
                {
                    return Err(SpecError::new(format!(
                        "{}: load limit exceeded: field bytes",
                        context()
                    )));
                }
                record_bytes = record_bytes
                    .checked_add(name.len() as u64)
                    .and_then(|bytes| bytes.checked_add(field_bytes))
                    .filter(|&bytes| bytes <= limits.max_record_bytes)
                    .ok_or_else(|| {
                        SpecError::new(format!("{}: load limit exceeded: record bytes", context()))
                    })?;
            }
            if !columns.contains_key(name) {
                if let Some(limits) = limits {
                    limits.check_shape(rows.len() as u64, columns.len() as u64 + 1)?;
                }
                columns.insert(name.clone(), (false, false));
            }
            let (boolean, numeric) = columns.get_mut(name).unwrap();
            *boolean |= value.is_boolean();
            *numeric |= value.is_number();
            if *boolean && *numeric {
                return Err(SpecError::new(format!(
                    "{}: bool and number cells cannot share a column",
                    context()
                )));
            }
        }
    }
    if columns.is_empty() {
        return Err(SpecError::new("Rows requires at least one column"));
    }
    let columns = columns
        .into_keys()
        .map(|name| {
            let cells = rows
                .iter()
                .map(|row| match row.get(&name) {
                    None | Some(Value::Null) => Cell::Na,
                    Some(Value::Bool(value)) => Cell::Bool(*value),
                    Some(Value::String(value)) => Cell::Str(value.clone()),
                    Some(Value::Number(value)) => match value.as_i64() {
                        Some(value) => Cell::Int(value),
                        None => Cell::Float(value.as_f64().unwrap()),
                    },
                    _ => unreachable!("Rows cells were admitted before allocation"),
                })
                .collect();
            Column::from_cells(&name, cells)
        })
        .collect();
    frame_with_params(
        &Frame {
            columns,
            n_rows: rows.len(),
            header_unit: "text".into(),
        },
        params,
    )
}

/// Build a typed [`Frame`] from pre-decoded nirs4all-formats records.
///
/// - `signals[].values` → numeric feature columns, named by the signal's axis
///   labels (header_unit derived from `axis.unit`: nm / cm-1 / index).
/// - `targets{k:v}` → numeric or categorical columns named `k` (left as raw
///   cells so the assembler's `encode_categorical` applies the same rules as the
///   CSV path).
/// - `metadata{sample_id, partition, id, ...}` → string/key columns, so the
///   existing partition (`PartitionBy::Column`) and identity/join logic runs
///   unchanged. The formats-layer `row_index` provenance counter is dropped: it
///   carries no dataset meaning and its numeric "0","1",… cells would otherwise
///   be picked up as a phantom feature by the `auto` features selector.
fn records_to_frame(
    records: &[Value],
    params: &LoadingParams,
    limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    if records.is_empty() {
        return Ok(Frame {
            columns: vec![],
            n_rows: 0,
            header_unit: header_unit_from_axis(""),
        });
    }
    let n = records.len();
    let signal_name = params
        .format
        .values
        .get("signal")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    for (index, record) in records.iter().enumerate() {
        let record = record
            .as_object()
            .ok_or_else(|| SpecError::new(format!("decoded record {index} must be an object")))?;
        if !["signals", "targets", "metadata"]
            .iter()
            .any(|section| record.contains_key(*section))
        {
            return Err(SpecError::new(format!(
                "decoded record {index} has no Formats sections; use SourcePayload::Rows for plain row objects"
            )));
        }
        for section in ["signals", "targets", "metadata"] {
            if record
                .get(section)
                .is_some_and(|value| !value.is_null() && !value.is_object())
            {
                return Err(SpecError::new(format!(
                    "decoded record {index}: section '{section}' must be an object or null"
                )));
            }
        }
        if let Some(signals) = record.get("signals").and_then(Value::as_object) {
            let selected = pick_signal(signals, &signal_name).or_else(|| {
                signals
                    .iter()
                    .next()
                    .map(|(key, value)| (key.as_str(), value))
            });
            if let Some((name, signal)) = selected {
                if signal.get("values").is_some_and(|value| !value.is_array()) {
                    return Err(SpecError::new(format!(
                        "decoded record {index}: selected signal '{name}' values must be an array"
                    )));
                }
            }
        }
    }

    // Determine the feature axis from the first record that carries a signal.
    let mut feature_headers: Vec<String> = Vec::new();
    let mut axis_unit = String::new();
    for record in records {
        let Some(signals) = record.get("signals").and_then(Value::as_object) else {
            continue;
        };
        let Some((_, signal)) = pick_signal(signals, &signal_name) else {
            continue;
        };
        let values = signal_values(signal);
        if values.is_empty() {
            continue;
        }
        feature_headers = axis_labels_from_values(&signal_axis_values(signal)?, values.len());
        axis_unit = signal
            .get("axis")
            .and_then(|axis| axis.get("unit"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        break;
    }
    let source_dtypes = records
        .iter()
        .filter_map(|record| {
            record
                .get("signals")
                .and_then(Value::as_object)
                .and_then(|signals| {
                    pick_signal(signals, &signal_name)
                        .map(|(_, signal)| signal["dtype"].as_str().map(str::to_owned))
                })
        })
        .collect::<HashSet<_>>();
    if source_dtypes.len() > 1 {
        return Err(SpecError::new(
            "Decoded records have inconsistent source dtypes",
        ));
    }
    let source_dtype = source_dtypes.into_iter().next().flatten();
    let n_features = feature_headers.len();
    if let Some(limits) = limits {
        let extra_columns: HashSet<_> = records
            .iter()
            .flat_map(|record| {
                ["targets", "metadata"].into_iter().flat_map(move |key| {
                    record
                        .get(key)
                        .and_then(Value::as_object)
                        .into_iter()
                        .flat_map(move |values| values.keys().map(move |name| (key, name)))
                })
            })
            .collect();
        limits.check_shape(n as u64, (n_features + extra_columns.len()) as u64)?;
    }

    // Feature columns (one per axis position): coerce each record's signal row.
    let mut feature_cells: Vec<Vec<Cell>> = vec![Vec::with_capacity(n); n_features];
    // Targets / metadata accumulate in first-appearance order across records.
    let mut target_order: Vec<String> = Vec::new();
    let mut target_cells: HashMap<String, Vec<Option<Value>>> = HashMap::new();
    let mut meta_order: Vec<String> = Vec::new();
    let mut meta_cells: HashMap<String, Vec<Option<String>>> = HashMap::new();

    for (row_idx, record) in records.iter().enumerate() {
        // features
        let row = record
            .get("signals")
            .and_then(Value::as_object)
            .and_then(|signals| pick_signal(signals, &signal_name))
            .and_then(|(_, signal)| signal["values"].as_array());
        let width = row.map_or(0, Vec::len);
        if width != 0 && width != n_features {
            return Err(SpecError::new(format!("decoded record {row_idx}: signal width {width} differs from axis width {n_features}")));
        }
        for (column, cells) in feature_cells.iter_mut().enumerate() {
            let value = row.and_then(|values| values.get(column));
            // Keep a retained integer scalar even when its declared destination
            // is F64; actual host-F64 scalars already become Cell::Float.
            cells.push(target_cell(value.cloned()));
        }
        // targets
        if let Some(targets) = record.get("targets").and_then(Value::as_object) {
            for key in targets.keys() {
                if !target_order.contains(key) {
                    target_order.push(key.clone());
                    target_cells.insert(key.clone(), Vec::new());
                }
            }
        }
        // metadata
        if let Some(metadata) = record.get("metadata").and_then(Value::as_object) {
            for key in metadata.keys() {
                // `row_index` is a formats-layer per-row provenance counter (0..n), not
                // a dataset column. Emitting it would let the `auto` features selector
                // (which takes every unassigned *numeric* column) coerce its "0","1",…
                // cells into a phantom feature, widening X by one and shifting the axis.
                if key == "row_index" {
                    continue;
                }
                if !meta_order.contains(key) {
                    meta_order.push(key.clone());
                    meta_cells.insert(key.clone(), Vec::new());
                }
            }
        }
    }
    // Second pass: fill target/metadata per record (None for absent keys).
    for record in records {
        let targets = record.get("targets").and_then(Value::as_object);
        for key in &target_order {
            let v = targets.and_then(|m| m.get(key)).cloned();
            target_cells.get_mut(key).unwrap().push(v);
        }
        let metadata = record.get("metadata").and_then(Value::as_object);
        for key in &meta_order {
            let v = metadata.and_then(|m| m.get(key)).and_then(value_string);
            meta_cells.get_mut(key).unwrap().push(v);
        }
    }

    let mut columns: Vec<Column> = Vec::new();
    for (header, cells) in feature_headers.into_iter().zip(feature_cells) {
        columns.push(Column::from_cells(&header, cells).with_source_dtype(source_dtype.clone()));
    }
    for key in target_order {
        let raw = target_cells.remove(&key).unwrap();
        columns.push(Column::from_cells(
            &key,
            raw.into_iter().map(target_cell).collect(),
        ));
    }
    for key in meta_order {
        let raw = meta_cells.remove(&key).unwrap();
        columns.push(Column::from_cells(
            &key,
            raw.into_iter()
                .map(|o| match o {
                    Some(s) => Cell::Str(s),
                    None => Cell::Na,
                })
                .collect(),
        ));
    }
    Ok(Frame {
        columns,
        n_rows: n,
        header_unit: header_unit_from_axis(&axis_unit),
    })
}

/// Map a decoded record axis unit to a `Frame` header_unit string.
fn header_unit_from_axis(unit: &str) -> String {
    match unit {
        "nm" | "cm-1" => unit.to_string(),
        "" => "index".to_string(),
        other => other.to_string(),
    }
}

/// A target value cell: numbers stay numeric; numeric-looking strings stay
/// strings (the assembler's `encode_categorical` does the to_numeric coercion).
fn target_cell(v: Option<Value>) -> Cell {
    match v {
        None | Some(Value::Null) => Cell::Na,
        Some(Value::Bool(b)) => Cell::Bool(b),
        Some(Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Cell::Int(i)
            } else if n
                .as_u64()
                .is_some_and(|i| !super::numeric::integer_exact_f64(i128::from(i)))
            {
                Cell::Str(n.to_string())
            } else {
                Cell::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Some(Value::String(s)) => Cell::Str(s),
        Some(other) => Cell::Str(other.to_string()),
    }
}

fn value_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn pick_signal<'a>(
    signals: &'a Map<String, Value>,
    preferred: &str,
) -> Option<(&'a str, &'a Value)> {
    if !preferred.is_empty() {
        if let Some((key, signal)) = signals.iter().find(|(key, _)| key.as_str() == preferred) {
            return Some((key.as_str(), signal));
        }
    }
    signals
        .iter()
        .find(|(_, signal)| signal["values"].as_array().is_some_and(|v| !v.is_empty()))
        .map(|(key, signal)| (key.as_str(), signal))
}

fn signal_values(signal: &Value) -> Vec<f64> {
    signal
        .get("values")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .unwrap_or_default()
}

fn signal_axis_values(signal: &Value) -> Result<Vec<f64>, SpecError> {
    signal
        .get("axis")
        .and_then(|axis| axis.get("values"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|value| value.is_number())
        .map(|value| super::numeric::exact_json_number_f64(value).map_err(SpecError::new))
        .collect()
}

/// Axis labels: one per signal value. If the axis has fewer/zero entries, fall
/// back to positional indices so the column count always matches the signal.
fn axis_labels_from_values(axis: &[f64], n_values: usize) -> Vec<String> {
    if axis.len() == n_values {
        axis.iter().copied().map(axis_label).collect()
    } else {
        (0..n_values).map(|i| i.to_string()).collect()
    }
}

fn axis_label(value: f64) -> String {
    if value.is_finite() && (value.fract().abs() < 1e-9) {
        format!("{}", value as i64)
    } else {
        crate::pyfmt::py_round(value, 6).to_string()
    }
}

fn join_key_columns(source: &SourceSpec) -> HashSet<String> {
    let mut cols = HashSet::new();
    if let Some(j) = &source.join {
        for k in [&j.left_on, &j.right_on] {
            if let Some(ks) = key_list(k) {
                cols.extend(ks);
            }
        }
    }
    cols
}

fn split_roles(source: &SourceSpec, df: &Frame) -> Result<Vec<(String, String)>, SpecError> {
    let headers = df.column_names();
    let dtypes = df.dtype_labels();
    let mut key_cols: HashSet<String> = key_list(&source.key)
        .unwrap_or_default()
        .into_iter()
        .collect();
    key_cols.extend(join_key_columns(source));
    key_cols.insert("filename_stem".into());
    key_cols.insert(row_idx_col(&source.id));

    let mut roles: Vec<(String, String)> = Vec::new();
    if source.role != Role::Mixed && source.columns.is_empty() {
        for col in &headers {
            if !key_cols.contains(col) {
                roles.push((col.clone(), source.role.value().to_string()));
            }
        }
        return Ok(roles);
    }

    let mut assigned: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let has_rest = source
        .columns
        .iter()
        .any(|cr| matches!(cr.select, Selector::Rest));
    for cr in &source.columns {
        let idxs = if let Selector::Auto { candidates } = &cr.select {
            resolve_auto_selector(
                &source.id, cr.role, candidates, &headers, &dtypes, &assigned, &key_cols,
            )?
        } else {
            cr.select.resolve(&headers, &dtypes, &assigned)?
        };
        for i in idxs {
            if assigned.contains(&i) {
                if source.strict_columns {
                    return Err(SpecError::new(format!(
                        "source '{}': column '{}' matched by multiple selectors (set strict_columns:false for first-match-wins)",
                        source.id, headers[i]
                    )));
                }
                continue;
            }
            assigned.insert(i);
            if !key_cols.contains(&headers[i]) {
                roles.push((headers[i].clone(), cr.role.value().to_string()));
            }
        }
    }
    let unmatched: Vec<&String> = headers
        .iter()
        .enumerate()
        .filter(|(i, h)| !assigned.contains(i) && !key_cols.contains(*h))
        .map(|(_, h)| h)
        .collect();
    if !unmatched.is_empty() && !has_rest {
        return Err(SpecError::new(format!(
            "source '{}': columns {unmatched:?} are unassigned and no 'rest' selector is present",
            source.id
        )));
    }
    Ok(roles)
}

#[allow(clippy::too_many_arguments)]
fn resolve_auto_selector(
    source_id: &str,
    role: Role,
    candidates: &[String],
    headers: &[String],
    dtypes: &[String],
    assigned: &std::collections::BTreeSet<usize>,
    key_cols: &HashSet<String>,
) -> Result<Vec<usize>, SpecError> {
    if !candidates.is_empty() {
        for name in candidates {
            if let Some(idx) = headers.iter().position(|h| h == name) {
                if !key_cols.contains(name) && !assigned.contains(&idx) {
                    return Ok(vec![idx]);
                }
            }
        }
        return Err(SpecError::new(format!(
            "source '{source_id}': auto selector with candidates {candidates:?} matched no available column (headers: {headers:?})"
        )));
    }
    let free: Vec<usize> = (0..headers.len())
        .filter(|i| !assigned.contains(i) && !key_cols.contains(&headers[*i]))
        .collect();
    let numeric: Vec<usize> = free
        .iter()
        .copied()
        .filter(|i| dtypes[*i] == "numeric")
        .collect();
    let non_numeric: Vec<usize> = free
        .iter()
        .copied()
        .filter(|i| dtypes[*i] != "numeric")
        .collect();
    match role {
        Role::Features => {
            if numeric.is_empty() {
                return Err(SpecError::new(format!("source '{source_id}': auto features selector found no unassigned numeric columns")));
            }
            Ok(numeric)
        }
        Role::Targets => {
            if numeric.is_empty() {
                return Err(SpecError::new(format!("source '{source_id}': auto targets selector found no unassigned numeric columns (provide explicit candidates for a categorical target)")));
            }
            Ok(vec![*numeric.last().unwrap()])
        }
        Role::Metadata => {
            if non_numeric.is_empty() {
                return Err(SpecError::new(format!("source '{source_id}': auto metadata selector found no unassigned non-numeric columns")));
            }
            Ok(non_numeric)
        }
        Role::Ignore => Ok(free),
        other => Err(SpecError::new(format!(
            "source '{source_id}': auto selector for role '{}' requires explicit candidates",
            other.value()
        ))),
    }
}

fn build_source_table(
    source: &SourceSpec,
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    audits: &mut Vec<Value>,
    tabular_limits: Option<TabularReadLimits>,
    options: &NumericStorageOptions,
) -> Result<SourceTable, SpecError> {
    let LoadedSource {
        frame: mut df,
        numeric_scope,
        header_unit,
        signal_type: signal,
        origins,
        raw_unit,
        raw_coordinates,
    } = load_source_frame_mem(source, spec, sources, audits, tabular_limits)?;
    if let Some(column) = df
        .columns
        .iter()
        .find(|col| col.name.starts_with("__src_row_idx__"))
    {
        return Err(SpecError::new(format!("source '{}': reserved column name '{}' collides with a user-supplied column; rename it", source.id, column.name)));
    }

    if let Some(origins) = &origins {
        if !df.has_column("filename_stem") {
            df.add_str_column("filename_stem", origins.clone());
        }
    }
    // derive grouped sample id (e.g. mango_001_a -> mango_001) for vendor replicates
    let si = &spec.sample_index;
    if let (Some(from), Some(key)) = (&si.derive_from, si.key.as_str()) {
        if df.has_column(from) && !df.has_column(key) {
            let src = df.str_column(from);
            let derived: Vec<String> = match &si.derive_pattern {
                Some(pat) => {
                    let rx = regex::Regex::new(pat).map_err(|e| {
                        SpecError::new(format!("invalid derive strip_suffix /{pat}/: {e}"))
                    })?;
                    src.iter()
                        .map(|s| rx.replace_all(s, "").into_owned())
                        .collect()
                }
                None => src,
            };
            df.add_str_column(key, derived);
        }
    }
    let mut roles = split_roles(source, &df)?;
    // Identity is never a feature/target role. It is retained separately as
    // aligned metadata for scientific traceability.
    for identity_column in [
        si.key.as_str(),
        si.observation_id.as_deref(),
        si.repetition_id.as_deref(),
        si.group_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        roles.retain(|(column, _)| column != identity_column);
        // Keep the actual values in the assembled frame as metadata. They are
        // needed to construct stable downstream identity and must never be
        // reconstructed from row position.
        if df.has_column(identity_column) {
            roles.push((identity_column.to_string(), "metadata".to_string()));
        }
    }
    let mut na = effective_params(&spec.params, &source.params).na;
    // nirs4all parity: metadata permits missing values and joins sample removal
    // only when that policy is explicitly requested.
    if source.role == Role::Metadata && na.policy != NaPolicy::RemoveSample {
        na.policy = NaPolicy::Ignore;
    }
    let mut remove_rows = apply_role_na_policy(&mut df, &mut roles, &na)?;
    let row_idx_name = row_idx_col(&source.id);
    if df.has_column(&row_idx_name) {
        return Err(SpecError::new(format!(
            "source '{}': reserved column name '{row_idx_name}' collides with a user-supplied column; rename it",
            source.id
        )));
    }
    df.add_int_column(&row_idx_name, (0..df.n_rows as i64).collect());

    let key = key_list(&source.key);
    let variations = load_variations(
        source,
        spec,
        sources,
        &df,
        &roles,
        tabular_limits,
        &mut remove_rows,
        options,
    )?;
    let spec_signal = if spec.signal_type.value() != "auto" {
        Some(spec.signal_type.value().to_string())
    } else {
        None
    };
    Ok(SourceTable {
        source_id: source.id.clone(),
        df,
        numeric_scope,
        roles,
        key,
        partition: source.partition.map(|p| p.value().to_string()),
        kind: source.kind.value().to_string(),
        join: source.join.clone(),
        signal_type: signal.or(spec_signal),
        header_unit,
        raw_unit,
        raw_coordinates,
        variations,
        remove_rows,
    })
}

fn read_variation_frame_mem(
    variation: &VariationSpec,
    spec: &DatasetSpec,
    source: &SourceSpec,
    sources: &[InMemorySource],
    tabular_limits: Option<TabularReadLimits>,
) -> Result<(Frame, NumericScope), SpecError> {
    let mut params = effective_params(&spec.params, &source.params);
    if !variation.params.is_empty_value() {
        params = effective_params(&params, &variation.params);
    }
    defer_sample_removal(&mut params);
    let matched = resolve_inputs_mem(&variation.input, sources)?;
    if matched.is_empty() {
        return Err(SpecError::new(format!(
            "source '{}': variation '{}': no input files resolved",
            source.id, variation.name
        )));
    }
    let frames: Vec<Frame> = matched
        .iter()
        .map(|s| frame_from_source(s, &source.id, &params, tabular_limits))
        .collect::<Result<_, _>>()?;
    let names: Vec<String> = (0..frames.len()).map(|i| i.to_string()).collect();
    let scope = NumericScope::inputs(
        &frames,
        &matched
            .iter()
            .map(|input| input_decimal(input, &params))
            .collect::<Vec<_>>(),
        MergeMode::ConcatSamples,
        &names,
        &[],
    );
    if frames.len() > 1 {
        Ok((
            concat_samples_with_limits(&frames, &names, tabular_limits)?.0,
            scope,
        ))
    } else {
        Ok((frames.into_iter().next().unwrap(), scope))
    }
}

#[allow(clippy::too_many_arguments)]
fn load_variations(
    source: &SourceSpec,
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    parent_df: &Frame,
    roles: &[(String, String)],
    tabular_limits: Option<TabularReadLimits>,
    remove_rows: &mut Option<Vec<bool>>,
    options: &NumericStorageOptions,
) -> Result<Vec<(String, Matrix)>, SpecError> {
    if source.variations.is_empty() {
        return Ok(vec![]);
    }
    if source.role != Role::Features && !roles.iter().any(|(_, r)| r == "features") {
        return Err(SpecError::new(format!(
            "source '{}': variations are only meaningful on a feature source (role/columns must declare 'features')",
            source.id
        )));
    }
    let feature_cols: Vec<String> = roles
        .iter()
        .filter(|(_, r)| r == "features")
        .map(|(c, _)| c.clone())
        .collect();
    if feature_cols.is_empty() {
        return Err(SpecError::new(format!(
            "source '{}': has variations but no feature columns to attach them to",
            source.id
        )));
    }
    let parent_n = parent_df.n_rows;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for variation in &source.variations {
        if !seen.insert(variation.name.clone()) {
            return Err(SpecError::new(format!(
                "source '{}': duplicate variation name '{}'",
                source.id, variation.name
            )));
        }
        let (vdf, scope) =
            read_variation_frame_mem(variation, spec, source, sources, tabular_limits)?;
        if vdf.n_rows != parent_n {
            return Err(SpecError::new(format!(
                "source '{}': variation '{}' has {} rows, expected {parent_n} (must row-align with the parent source)",
                source.id, variation.name, vdf.n_rows
            )));
        }
        let params = effective_params(
            &effective_params(&spec.params, &source.params),
            &variation.params,
        );
        if params.na.policy == NaPolicy::RemoveSample {
            let rows = remove_rows.get_or_insert_with(|| vec![false; parent_n]);
            for (remove, missing) in rows.iter_mut().zip(missing_rows(&vdf)) {
                *remove |= missing;
            }
        }
        let named = feature_cols.iter().filter(|c| vdf.has_column(c)).count();
        let arr = if named == feature_cols.len() {
            vdf.coerce_numeric_with_decimal(
                &feature_cols,
                options.features,
                options.consumer.as_deref(),
                |column, row| scope.decimal(column, row),
            )
            .map_err(SpecError::new)?
        } else if named == 0 && vdf.columns.len() == feature_cols.len() {
            vdf.coerce_numeric_with_decimal(
                &vdf.column_names(),
                options.features,
                options.consumer.as_deref(),
                |column, row| scope.decimal(column, row),
            )
            .map_err(SpecError::new)?
        } else {
            return Err(SpecError::new(format!(
                "source '{}': variation '{}' is ambiguous: {named}/{} parent feature columns matched by name and the frame has {} columns",
                source.id, variation.name, feature_cols.len(), vdf.columns.len()
            )));
        };
        out.push((variation.name.clone(), arr));
    }
    Ok(out)
}

fn has_role(table: &SourceTable, role: &str) -> bool {
    table.roles.iter().any(|(_, r)| r == role)
}

/// Load, role-split, join and partition `spec` into an [`AssembledDataset`],
/// using named in-memory sources (fs-free). `index_lists` supplies row indices
/// for `partitions.by=index_file` (keys: `train`/`test`/`predict`); `fold_inline`
/// supplies parsed folds for `folds.file`. Both default empty/None.
pub fn assemble_in_memory(
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    index_lists: &HashMap<String, Vec<i64>>,
    fold_inline: Option<&[Fold]>,
) -> Result<AssembledDataset, SpecError> {
    assemble_in_memory_with_tabular_limits(spec, sources, index_lists, fold_inline, None)
}

/// Assemble in-memory sources while applying parser budgets to every tabular
/// byte source and shape budgets before structured-frame and merge/join
/// allocations. Passing `None` is identical to [`assemble_in_memory`].
pub fn assemble_in_memory_with_tabular_limits(
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    index_lists: &HashMap<String, Vec<i64>>,
    fold_inline: Option<&[Fold]>,
    tabular_limits: Option<TabularReadLimits>,
) -> Result<AssembledDataset, SpecError> {
    assemble_in_memory_with_storage_and_tabular_limits(
        spec,
        sources,
        index_lists,
        fold_inline,
        tabular_limits,
        &NumericStorageOptions::default(),
    )
}

pub fn assemble_in_memory_with_storage(
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    index_lists: &HashMap<String, Vec<i64>>,
    fold_inline: Option<&[Fold]>,
    options: &NumericStorageOptions,
) -> Result<AssembledDataset, SpecError> {
    assemble_in_memory_with_storage_and_tabular_limits(
        spec,
        sources,
        index_lists,
        fold_inline,
        None,
        options,
    )
}

pub fn assemble_in_memory_with_storage_and_tabular_limits(
    spec: &DatasetSpec,
    sources: &[InMemorySource],
    index_lists: &HashMap<String, Vec<i64>>,
    fold_inline: Option<&[Fold]>,
    tabular_limits: Option<TabularReadLimits>,
    options: &NumericStorageOptions,
) -> Result<AssembledDataset, SpecError> {
    let mut audits: Vec<Value> = Vec::new();
    let mut tables: IndexMap<String, SourceTable> = IndexMap::new();
    for s in &spec.sources {
        tables.insert(
            s.id.clone(),
            build_source_table(s, spec, sources, &mut audits, tabular_limits, options)?,
        );
    }

    let mut partitions: Vec<String> = tables
        .values()
        .filter_map(|t| t.partition.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    partitions.sort();

    let n_sources = tables
        .values()
        .filter(|t| t.kind != SourceKind::Lookup.value() && has_role(t, "features"))
        .count();
    // `repetition` groups repeated measurements of one sample and is opt-in;
    // `sample_index.repetition_id` is a per-scan index and never stands in for it.
    let repetition = spec.repetition.clone();
    let identity = IdentityProvenance {
        source_ids: tables
            .values()
            .filter(|table| table.kind != SourceKind::Lookup.value() && has_role(table, "features"))
            .map(|table| table.source_id.clone())
            .collect(),
        sample_id: (spec.sample_index.by.value() == "id")
            .then(|| spec.sample_index.key.as_str().map(str::to_string))
            .flatten(),
        observation_id: spec.sample_index.observation_id.clone(),
        repetition_id: spec.sample_index.repetition_id.clone(),
        group_id: spec.sample_index.group_id.clone(),
    };
    let mut assembled = AssembledDataset {
        name: spec.name.clone().unwrap_or_else(|| "dataset".into()),
        task_type: spec.task_type.value().to_string(),
        signal_type: spec.signal_type.value().to_string(),
        n_sources,
        blocks: IndexMap::new(),
        folds: vec![],
        fold_provenance: vec![],
        repetition,
        identity,
        aggregate: spec.aggregate.clone(),
        warnings: vec![],
        audits: vec![],
    };

    let mut combined_df: Option<Frame> = None;
    let mut train_rows: Option<Vec<usize>> = None;
    if !partitions.is_empty() {
        for part in &partitions {
            let part_tables: IndexMap<&String, &SourceTable> = tables
                .iter()
                .filter(|(_, t)| {
                    t.partition.as_deref() == Some(part) || t.kind == SourceKind::Lookup.value()
                })
                .collect();
            let (block, _) = assemble_block(
                &part_tables,
                spec,
                &mut audits,
                &mut assembled.warnings,
                tabular_limits,
                options,
            )?;
            assembled.blocks.insert(part.clone(), block);
        }
    } else {
        let all: IndexMap<&String, &SourceTable> = tables.iter().collect();
        let (block, combined) = assemble_block(
            &all,
            spec,
            &mut audits,
            &mut assembled.warnings,
            tabular_limits,
            options,
        )?;
        if spec.partitions.is_none() {
            train_rows = Some((0..combined.n_rows).collect());
            assembled.blocks.insert("train".into(), block);
        } else {
            for (part, mask) in split_partition_mask(&combined, spec, index_lists)? {
                if part == "train" {
                    train_rows = Some(
                        mask.iter()
                            .enumerate()
                            .filter_map(|(i, keep)| keep.then_some(i))
                            .collect(),
                    );
                }
                assembled.blocks.insert(part, slice_block(&block, &mask));
            }
        }
        combined_df = Some(combined);
    }

    assembled.audits = audits;
    attach_folds(
        spec,
        fold_inline,
        &mut assembled,
        combined_df.as_ref(),
        train_rows.as_deref(),
    )?;
    Ok(assembled)
}

fn row_align(
    combined: &Frame,
    t: &SourceTable,
    primary_id: &str,
    limits: Option<TabularReadLimits>,
) -> Result<Frame, SpecError> {
    if t.df.n_rows != combined.n_rows {
        return Err(SpecError::new(format!(
            "source '{}' ({} rows) is not row-aligned with '{primary_id}' ({} rows); supply a join key",
            t.source_id, t.df.n_rows, combined.n_rows
        )));
    }
    let (merged, _) = concat_features_with_limits(
        &[combined.clone(), t.df.clone()],
        &[primary_id.to_string(), t.source_id.clone()],
        None,
        limits,
    )?;
    Ok(merged)
}

fn join_onto(
    combined: &Frame,
    primary: &SourceTable,
    t: &SourceTable,
    audits: &mut Vec<Value>,
    warnings: &mut Vec<String>,
    limits: Option<TabularReadLimits>,
) -> Result<(Frame, Option<Vec<String>>), SpecError> {
    if let Some(j) = &t.join {
        let left_on = key_list(&j.left_on);
        let right_on = key_list(&j.right_on).or_else(|| t.key.clone());
        if let (Some(left_on), Some(right_on)) = (&left_on, &right_on) {
            let missing_l: Vec<&String> =
                left_on.iter().filter(|c| !combined.has_column(c)).collect();
            let missing_r: Vec<&String> = right_on.iter().filter(|c| !t.df.has_column(c)).collect();
            if !missing_l.is_empty() || !missing_r.is_empty() {
                return Err(SpecError::new(format!(
                    "source '{}': join keys not found (left missing {missing_l:?} in assembled data, right missing {missing_r:?} in '{}')",
                    t.source_id, t.source_id
                )));
            }
            let left_name = j.left.clone().unwrap_or_else(|| primary.source_id.clone());
            let (out, audit) = join_tables_with_limits(
                combined,
                &t.df,
                left_on,
                right_on,
                j.cardinality,
                j.coverage,
                &left_name,
                &t.source_id,
                limits,
            )?;
            audits.push(serde_json::json!({"join": audit.operation, "dropped": audit.dropped_rows.len(), "warnings": audit.warnings}));
            warnings.extend(audit.warnings);
            return Ok((out, Some(right_on.clone())));
        }
        if j.cardinality == Cardinality::OneToOne {
            return row_align(combined, t, &primary.source_id, limits).map(|frame| (frame, None));
        }
        return Err(SpecError::new(format!(
            "source '{}': {} join needs explicit left_on/right_on",
            t.source_id,
            j.cardinality.value()
        )));
    }
    // no join spec: align by a shared key (1:1) if both declare one, else row order
    if let (Some(lkeys), Some(rkeys)) = (&primary.key, &t.key) {
        if lkeys.iter().all(|c| combined.has_column(c)) && rkeys.iter().all(|c| t.df.has_column(c))
        {
            let (out, audit) = join_tables_with_limits(
                combined,
                &t.df,
                lkeys,
                rkeys,
                Cardinality::OneToOne,
                Coverage::Complete,
                &primary.source_id,
                &t.source_id,
                limits,
            )?;
            warnings.extend(audit.warnings);
            return Ok((out, Some(rkeys.clone())));
        }
    }
    row_align(combined, t, &primary.source_id, limits).map(|frame| (frame, None))
}

type RoleColumns = IndexMap<String, (String, String)>;

/// Collect roles and native names in frame merge order, including unassigned
/// keys and internal columns in name ownership. Dropped right keys claim nothing.
fn collect_roles(
    combined: &Frame,
    merge_steps: &[(&SourceTable, Option<Vec<String>>)],
) -> (RoleColumns, IndexMap<String, String>) {
    let mut role_cols = RoleColumns::new();
    let mut native_cols = IndexMap::new();
    let mut seen = HashSet::new();
    for (table, dropped_keys) in merge_steps {
        let left_names = seen.clone();
        for column in &table.df.columns {
            let col = &column.name;
            if dropped_keys.as_ref().is_some_and(|keys| keys.contains(col)) {
                continue;
            }
            let name = if dropped_keys.is_some() {
                if left_names.contains(col) {
                    format!("{col}__{}", table.source_id)
                } else {
                    col.clone()
                }
            } else if seen.contains(col) {
                format!("{}__{col}", table.source_id)
            } else {
                col.clone()
            };
            seen.insert(name.clone());
            if !combined.has_column(&name) {
                continue;
            }
            native_cols.insert(name.clone(), col.clone());
            if let Some((_, role)) = table.roles.iter().find(|(name, _)| name == col) {
                role_cols.insert(name, (role.clone(), table.source_id.clone()));
            }
        }
    }
    (role_cols, native_cols)
}

fn assemble_block(
    tables: &IndexMap<&String, &SourceTable>,
    spec: &DatasetSpec,
    audits: &mut Vec<Value>,
    warnings: &mut Vec<String>,
    limits: Option<TabularReadLimits>,
    options: &NumericStorageOptions,
) -> Result<(PartitionBlock, Frame), SpecError> {
    let feature_tables: Vec<&SourceTable> = tables
        .values()
        .filter(|t| t.kind != SourceKind::Lookup.value() && has_role(t, "features"))
        .copied()
        .collect();
    if feature_tables.is_empty() {
        return Err(SpecError::new("partition has no feature source"));
    }
    let primary = feature_tables[0];
    let mut combined = primary.df.clone();
    let mut merge_steps = vec![(primary, None)];
    for t in tables.values() {
        if t.source_id == primary.source_id {
            continue;
        }
        let (frame, dropped_keys) = join_onto(&combined, primary, t, audits, warnings, limits)?;
        combined = frame;
        merge_steps.push((t, dropped_keys));
    }

    if spec.sample_index.by.value() == "id" {
        if let Some(key) = spec.sample_index.key.as_str() {
            if !combined.has_column(key) {
                return Err(SpecError::new(format!(
                    "source '{}': declared sample identity '{key}' missing in this partition",
                    primary.source_id
                )));
            }
        }
    }

    let mut keep = vec![true; combined.n_rows];
    for table in tables.values() {
        if let Some(remove) = &table.remove_rows {
            let positions = combined.numeric_column_f64(&row_idx_col(&table.source_id));
            if positions.len() != combined.n_rows {
                return Err(SpecError::new(format!("source '{}': row-index column lost during assembly; NA samples cannot be aligned", table.source_id)));
            }
            for (keep_row, original_row) in keep.iter_mut().zip(positions) {
                *keep_row &= original_row.is_finite()
                    && original_row >= 0.0
                    && remove
                        .get(original_row as usize)
                        .is_some_and(|remove| !remove);
            }
        }
    }
    if keep.iter().any(|keep_row| !keep_row) {
        let removed = keep.iter().filter(|keep_row| !**keep_row).count();
        combined = combined.mask_rows(&keep);
        audits.push(serde_json::json!({"operation": "remove_na_samples", "removed": removed}));
    }

    let (role_cols, native_cols) = collect_roles(&combined, &merge_steps);
    let decimal = |name: &str, row: usize| {
        let Some((_, owner)) = role_cols.get(name) else {
            return '.';
        };
        let Some(table) = tables.values().find(|table| &table.source_id == owner) else {
            return '.';
        };
        let original = &native_cols[name];
        let position = combined
            .column(&row_idx_col(&table.source_id))
            .and_then(|column| column.values.get(row))
            .map(Cell::to_numeric)
            .unwrap_or(f64::NAN);
        if position.is_finite() && position >= 0.0 {
            table.numeric_scope.decimal(original, position as usize)
        } else {
            '.'
        }
    };
    for column in &combined.columns {
        let selected = role_cols
            .get(&column.name)
            .is_some_and(|(role, _)| role != "ignore")
            || key_list(&spec.sample_index.key).is_some_and(|keys| keys.contains(&column.name))
            || [
                spec.sample_index.observation_id.as_deref(),
                spec.sample_index.repetition_id.as_deref(),
                spec.sample_index.group_id.as_deref(),
            ]
            .into_iter()
            .flatten()
            .any(|key| key == column.name);
        if selected {
            for (row, cell) in column.values.iter().enumerate() {
                if matches!(cell, Cell::Float(value) if value.is_infinite()) {
                    return Err(SpecError::new("Observed numeric values must be finite"));
                }
                if !column.numeric_text_at(row) {
                    continue;
                }
                if let Cell::Str(text) = cell {
                    let body = text.trim().trim_start_matches(['+', '-']);
                    if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
                        if !super::numeric::integer_in_domain(text) {
                            return Err(SpecError::new(
                                "Integer numeric text is outside the supported i64/u64 domain",
                            ));
                        }
                    } else {
                        super::numeric::checked_decimal_f64(text.trim()).map_err(SpecError::new)?;
                    }
                }
            }
        }
    }
    let combined_names = combined.column_names();
    let mut block = PartitionBlock {
        n_samples: combined.n_rows,
        ..Default::default()
    };

    for ft in &feature_tables {
        let cols: Vec<String> = combined_names
            .iter()
            .filter(|c| {
                role_cols
                    .get(*c)
                    .map(|(r, s)| r == "features" && s == &ft.source_id)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        if cols.is_empty() {
            continue;
        }
        block.source_ids.push(ft.source_id.clone());
        let headers: Vec<String> = cols.iter().map(|c| native_cols[c].clone()).collect();
        block.x.push(
            combined
                .coerce_numeric_with_decimal(
                    &cols,
                    options.features,
                    options.consumer.as_deref(),
                    decimal,
                )
                .map_err(SpecError::new)?,
        );
        let coordinates = if let Some(raw) = &ft.raw_coordinates {
            headers
                .iter()
                .map(|name| raw.get(name).cloned())
                .collect::<Option<Vec<_>>>()
        } else {
            super::package::header_coordinates(&headers).map_err(SpecError::new)?
        };
        block.feature_names.push(Some(headers.clone()));
        block.feature_axes.push(vec![serde_json::json!({
            "name":super::package::feature_axis_name(&ft.header_unit),
            "unit":ft.raw_unit,"coordinates":coordinates
        })]);
        block
            .representation_ids
            .push(super::package::repr_ids::SIGNAL_1D.into());
        block.feature_headers.push(headers);
        block.header_units.push(ft.header_unit.clone());
        block.signal_types.push(ft.signal_type.clone());
        let mut src_proc: Vec<(String, Matrix)> = Vec::new();
        if !ft.variations.is_empty() {
            let idx_col = row_idx_col(&ft.source_id);
            let row_idx: Vec<usize> = combined
                .numeric_column_f64(&idx_col)
                .iter()
                .map(|&v| v as usize)
                .collect();
            for (name, arr) in &ft.variations {
                let data: Vec<f64> = row_idx
                    .iter()
                    .flat_map(|&r| {
                        arr.data[r * arr.n_cols..(r + 1) * arr.n_cols]
                            .iter()
                            .copied()
                    })
                    .collect();
                src_proc.push((
                    name.clone(),
                    Matrix {
                        validity: row_idx
                            .iter()
                            .flat_map(|&r| {
                                arr.validity[r * arr.n_cols..(r + 1) * arr.n_cols]
                                    .iter()
                                    .copied()
                            })
                            .collect(),
                        precision: arr.precision.clone(),
                        data,
                        n_rows: row_idx.len(),
                        n_cols: arr.n_cols,
                    },
                ));
            }
        }
        block.processings.push(src_proc);
    }

    let target_cols: Vec<String> = combined_names
        .iter()
        .filter(|c| {
            role_cols
                .get(*c)
                .map(|(r, _)| r == "targets")
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if !target_cols.is_empty() {
        let mut counts = HashMap::new();
        for col in &target_cols {
            *counts.entry(&native_cols[col]).or_insert(0usize) += 1;
        }
        let target_headers: Vec<String> = target_cols
            .iter()
            .map(|col| {
                if counts[&native_cols[col]] == 1 {
                    native_cols[col].clone()
                } else {
                    col.clone()
                }
            })
            .collect();
        let mut seen = HashSet::new();
        if target_headers.iter().any(|name| !seen.insert(name)) {
            return Err(SpecError::new(format!(
                "target export labels must be unique, got {target_headers:?}"
            )));
        }
        let mode = spec
            .params
            .categorical
            .map(|c| c.value().to_string())
            .unwrap_or_else(|| "auto".into());
        let numeric_columns = target_cols
            .iter()
            .filter(|name| combined.column(name).unwrap().is_numeric_dtype())
            .cloned()
            .collect::<Vec<_>>();
        let numeric_source = combined.numeric_source_dtype(&numeric_columns);
        let precision = PrecisionRecord::select(
            numeric_source.as_deref(),
            options.targets,
            options.consumer.as_deref(),
            false,
        )
        .map_err(SpecError::new)?;
        if options.consumer.as_deref() == Some("SpectroDataset")
            && options.targets == Some(super::numeric::NumericStorage::Float64)
        {
            return Err(SpecError::new("SpectroDataset cannot guarantee an explicitly requested float64 numerical target view"));
        }
        let mut y_cols: Vec<Vec<f64>> = Vec::new();
        let mut cats: IndexMap<String, Value> = IndexMap::new();
        for (c, header) in target_cols.iter().zip(&target_headers) {
            let owner_id = &role_cols[c].1;
            let source = spec
                .sources
                .iter()
                .find(|source| source.id.as_str() == owner_id.as_str())
                .expect("target source exists");
            let params = effective_params(&spec.params, &source.params);
            let target_mode = params.categorical.map(|mode| mode.value()).unwrap_or(&mode);
            let cells: Vec<String> = combined
                .column(c)
                .unwrap()
                .values
                .iter()
                .filter(|cell| {
                    !matches!(cell, Cell::Na) && !matches!(cell, Cell::Float(v) if v.is_nan())
                })
                .map(Cell::to_str_scalar)
                .collect();
            static DOT: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
                regex::Regex::new(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
                    .unwrap()
            });
            if target_mode == "auto"
                && params
                    .decimal_separator
                    .as_deref()
                    .is_none_or(str::is_empty)
                && cells.iter().any(|cell| is_comma_pair(cell))
                && cells
                    .iter()
                    .all(|cell| is_comma_pair(cell) || DOT.is_match(cell.trim()))
            {
                return Err(SpecError::new(format!("source '{}': file {} target column '{}' contains numeric comma data; declare params.decimal_separator",source.id,source.input,c)));
            }
            let (enc, mapping) = encode_categorical_with_storage(
                combined.column(c).unwrap(),
                target_mode,
                &precision,
                matches!(
                    spec.task_type.value(),
                    "classification" | "binary" | "multiclass"
                ),
                |row| decimal(c, row),
            )?;
            y_cols.push(enc);
            if let Some(m) = mapping {
                cats.insert(header.clone(), m);
            }
        }
        let mut matrix = column_stack(&y_cols);
        matrix.precision = precision;
        block.y = Some(matrix);
        block.y_headers = target_headers;
        block.y_categorical = cats;
    }

    let weight_cols: Vec<String> = combined_names
        .iter()
        .filter(|c| {
            role_cols
                .get(*c)
                .map(|(r, _)| r == "weights")
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if !weight_cols.is_empty() {
        if weight_cols.len() > 1 {
            return Err(SpecError::new(format!(
                "only one 'weights' column is supported, got {weight_cols:?}"
            )));
        }
        let precision = PrecisionRecord::select(
            combined.numeric_source_dtype(&weight_cols).as_deref(),
            options.weights,
            options.consumer.as_deref(),
            false,
        )
        .map_err(SpecError::new)?;
        let column = combined.column(&weight_cols[0]).unwrap();
        block.weights = Some(
            column
                .values
                .iter()
                .enumerate()
                .map(|(row, cell)| {
                    cell.to_numeric_with_decimal(
                        precision.dtype,
                        column.source_dtype.as_deref(),
                        decimal(&weight_cols[0], row),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(SpecError::new)?,
        );
        block.weights_validity = block
            .weights
            .as_ref()
            .map(|values| values.iter().map(|value| !value.is_nan()).collect());
        block.weights_precision = precision;
        block.weights_header = Some(weight_cols[0].clone());
    }

    let mut meta_cols: Vec<String> = combined_names
        .iter()
        .filter(|c| {
            role_cols
                .get(*c)
                .map(|(r, _)| r == "metadata")
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    // Identity columns are exempt from ordinary roles so they can drive joins
    // without leaking into X. Preserve them explicitly in the assembled IR:
    // downstream scientific adapters must never recreate IDs from row position.
    // Keep the source order stable and ignore unavailable columns here; the
    // bridge performs a typed preflight and names the missing capability.
    for column in [
        spec.sample_index.key.as_str(),
        spec.sample_index.observation_id.as_deref(),
        spec.sample_index.repetition_id.as_deref(),
        spec.sample_index.group_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if combined.has_column(column) && !meta_cols.iter().any(|name| name == column) {
            meta_cols.push(column.to_string());
        }
    }
    if !meta_cols.is_empty() {
        block.metadata = Some(combined.select(&meta_cols));
    }
    Ok((block, combined))
}

/// `encode_categorical`: factorize an object target when `auto` + more NaNs under
/// numeric coercion; else numeric coercion (f64). The mapping (when categorical)
/// is `{"categories": [...]}`, matching the Python IR.
#[cfg(test)]
fn encode_categorical(col: &Column, mode: &str) -> Result<(Vec<f64>, Option<Value>), SpecError> {
    encode_categorical_with_storage(col, mode, &PrecisionRecord::default(), false, |_| '.')
}
fn encode_categorical_with_storage(
    col: &Column,
    mode: &str,
    precision: &PrecisionRecord,
    classification: bool,
    decimal: impl Fn(usize) -> char,
) -> Result<(Vec<f64>, Option<Value>), SpecError> {
    let numeric: Vec<f64> = col
        .values
        .iter()
        .enumerate()
        .map(|(row, cell)| cell.to_numeric_view(decimal(row)))
        .collect();
    let is_objectish = !col.is_numeric_dtype();
    let numeric_na = numeric.iter().filter(|v| v.is_nan()).count();
    let series_na = col
        .values
        .iter()
        .filter(|c| matches!(c, Cell::Na) || matches!(c, Cell::Float(f) if f.is_nan()))
        .count();
    let treat_categorical = mode == "auto" && is_objectish && numeric_na > series_na;
    if treat_categorical {
        if precision.requested_dtype.is_some() {
            return Err(SpecError::new(
                "Numeric precision request for string targets is unsupported",
            ));
        }
        // pandas factorize over str(values): codes in first-appearance order.
        let mut categories: Vec<String> = Vec::new();
        let mut codes = Vec::with_capacity(col.values.len());
        for v in &col.values {
            if matches!(v, Cell::Na) || matches!(v, Cell::Float(f) if f.is_nan()) {
                codes.push(f64::NAN);
                continue;
            }
            let s = v.to_str_scalar();
            let code = match categories.iter().position(|c| c == &s) {
                Some(i) => i,
                None => {
                    categories.push(s);
                    categories.len() - 1
                }
            };
            codes.push(code as f64);
        }
        Ok((
            codes,
            Some(if let Some(dtype) = &col.source_dtype {
                serde_json::json!({ "categories": categories, "source_dtype": dtype })
            } else {
                serde_json::json!({ "categories": categories })
            }),
        ))
    } else {
        Ok((
            col.values
                .iter()
                .enumerate()
                .map(|(row, cell)| {
                    if classification {
                        let value = cell.to_numeric_with_decimal(
                            super::numeric::NumericStorage::Float64,
                            None,
                            decimal(row),
                        )?;
                        if value.is_nan() {
                            return Ok(value);
                        }
                        if value.fract() != 0.0 || value.abs() > 9_007_199_254_740_991.0 {
                            return Err("Classification requires portable integer labels".into());
                        }
                        super::numeric::validate_label(value as i64, precision.dtype)?;
                    }
                    cell.to_numeric_with_decimal(
                        precision.dtype,
                        col.source_dtype.as_deref(),
                        decimal(row),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(SpecError::new)?,
            None,
        ))
    }
}

fn column_stack(cols: &[Vec<f64>]) -> Matrix {
    let n_cols = cols.len();
    let n_rows = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for row in 0..n_rows {
        for c in cols {
            data.push(c[row]);
        }
    }
    Matrix::new(data, n_rows, n_cols)
}

fn slice_block(block: &PartitionBlock, mask: &[bool]) -> PartitionBlock {
    let take = |m: &Matrix| -> Matrix {
        let data: Vec<f64> = (0..m.n_rows)
            .filter(|&r| mask[r])
            .flat_map(|r| m.data[r * m.n_cols..(r + 1) * m.n_cols].iter().copied())
            .collect();
        Matrix {
            validity: (0..m.n_rows)
                .filter(|&r| mask[r])
                .flat_map(|r| m.validity[r * m.n_cols..(r + 1) * m.n_cols].iter().copied())
                .collect(),
            precision: m.precision.clone(),
            data,
            n_rows: mask.iter().filter(|&&b| b).count(),
            n_cols: m.n_cols,
        }
    };
    PartitionBlock {
        n_samples: mask.iter().filter(|&&b| b).count(),
        source_ids: block.source_ids.clone(),
        x: block.x.iter().map(take).collect(),
        feature_headers: block.feature_headers.clone(),
        feature_names: block.feature_names.clone(),
        feature_axes: block.feature_axes.clone(),
        representation_ids: block.representation_ids.clone(),
        header_units: block.header_units.clone(),
        signal_types: block.signal_types.clone(),
        processings: block
            .processings
            .iter()
            .map(|src| src.iter().map(|(n, m)| (n.clone(), take(m))).collect())
            .collect(),
        y: block.y.as_ref().map(take),
        y_headers: block.y_headers.clone(),
        y_categorical: block.y_categorical.clone(),
        metadata: block.metadata.as_ref().map(|f| f.mask_rows(mask)),
        weights: block.weights.as_ref().map(|w| {
            w.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect()
        }),
        weights_header: block.weights_header.clone(),
        weights_validity: block.weights_validity.as_ref().map(|v| {
            v.iter()
                .zip(mask)
                .filter_map(|(v, m)| m.then_some(*v))
                .collect()
        }),
        weights_precision: block.weights_precision.clone(),
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
enum PartitionLabelKey {
    Text(String),
    Numeric(JoinKey),
}

fn partition_numeric_key(cell: &Cell) -> Option<JoinKey> {
    match cell {
        Cell::Float(value) if *value == 0.0 && value.is_sign_negative() => {
            Some(JoinKey::Num(value.to_bits()))
        }
        _ => cell.join_key(),
    }
}

fn partition_json_numeric_key(value: &Value, float32: bool) -> Option<JoinKey> {
    if float32 {
        let rounded = super::numeric::NumericStorage::Float32
            .json_number(value, None)
            .ok()?;
        return partition_numeric_key(&Cell::Float(rounded));
    }
    if let Some(integer) = value.as_i64() {
        return Cell::Int(integer).join_key();
    }
    if let Some(integer) = value.as_u64() {
        // The existing identity domain has an exact unsigned key; never widen it.
        return Some(JoinKey::Uint(integer));
    }
    let number = value.as_f64()?;
    if !number.is_finite() {
        return None;
    }
    partition_numeric_key(&Cell::Float(number))
}

fn partition_declared_keys(values: &[Value], float32: bool) -> HashSet<PartitionLabelKey> {
    static TOKEN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\A-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\z").unwrap()
    });
    let mut keys = HashSet::new();
    for value in values {
        keys.insert(PartitionLabelKey::Text(
            crate::pyfmt::py_str_scalar(value).to_lowercase(),
        ));
        let number = match value {
            Value::Number(_) => Some(value.clone()),
            Value::String(text) if TOKEN.is_match(text) => {
                crate::canonical_json::parse_json_value(text).ok()
            }
            _ => None,
        };
        if let Some(key) = number
            .as_ref()
            .and_then(|number| partition_json_numeric_key(number, float32))
        {
            keys.insert(PartitionLabelKey::Numeric(key));
        }
    }
    keys
}

fn split_partition_mask(
    df: &Frame,
    spec: &DatasetSpec,
    index_lists: &HashMap<String, Vec<i64>>,
) -> Result<IndexMap<String, Vec<bool>>, SpecError> {
    let n = df.n_rows;
    let p = match &spec.partitions {
        Some(p) if p.by.is_some() => p,
        _ => {
            let mut m = IndexMap::new();
            m.insert("train".into(), vec![true; n]);
            return Ok(m);
        }
    };
    match p.by.unwrap() {
        PartitionBy::Column => {
            let column = p.column.as_deref().unwrap_or("");
            if !df.has_column(column) {
                return Err(SpecError::new(format!(
                    "partitions.column '{column}' not found in the assembled data (columns: {:?})",
                    df.column_names()
                )));
            }
            p.validate_column_values()?;
            let source_column = df.column(column).expect("partition column exists");
            let float32 = source_column.source_dtype.as_deref() == Some("float32");
            let values = [
                ("train_values", &p.train_values),
                ("test_values", &p.test_values),
                ("val_values", &p.val_values),
                ("predict_values", &p.predict_values),
            ];
            let declared: Vec<_> = values
                .iter()
                .map(|(_, list)| partition_declared_keys(list, float32))
                .collect();
            // Reuse the frame's identity projection for retained exact u64 carriers.
            let numeric_keys = df.join_key_tuples(&[column.to_string()]);
            let col: Vec<Option<PartitionLabelKey>> = source_column
                .values
                .iter()
                .enumerate()
                .map(|(row, cell)| match cell {
                    Cell::Na => None,
                    Cell::Float(value) if value.is_nan() => None,
                    Cell::Float(value) if value.is_infinite() => Some(PartitionLabelKey::Text(
                        if value.is_sign_negative() {
                            "-inf"
                        } else {
                            "inf"
                        }
                        .into(),
                    )),
                    Cell::Bool(_) => {
                        Some(PartitionLabelKey::Text(cell.to_str_scalar().to_lowercase()))
                    }
                    Cell::Int(_) | Cell::Float(_) => {
                        partition_numeric_key(cell).map(PartitionLabelKey::Numeric)
                    }
                    Cell::Str(text) if source_column.numeric_text_at(row) => {
                        let negative_zero = text.trim().starts_with('-')
                            && super::numeric::checked_decimal_f64(text.trim())
                                .is_ok_and(|v| v == 0.0 && v.is_sign_negative());
                        let key = if negative_zero {
                            Some(JoinKey::Num((-0.0f64).to_bits()))
                        } else {
                            numeric_keys[row]
                                .as_ref()
                                .and_then(|keys| keys.first())
                                .cloned()
                        };
                        key.map(|key| match key {
                            // Retained refused numerics are local lexical labels
                            // when ignored, never approximate numeric aliases.
                            JoinKey::Lexeme(token) => PartitionLabelKey::Text(token.to_lowercase()),
                            key => PartitionLabelKey::Numeric(key),
                        })
                    }
                    Cell::Str(_) => {
                        Some(PartitionLabelKey::Text(cell.to_str_scalar().to_lowercase()))
                    }
                })
                .collect();
            let mut matches: Vec<Vec<bool>> = declared
                .iter()
                .map(|keys| {
                    col.iter()
                        .map(|key| key.as_ref().is_some_and(|key| keys.contains(key)))
                        .collect()
                })
                .collect();
            for row in 0..n {
                let owners: Vec<_> = values
                    .iter()
                    .zip(&matches)
                    .filter_map(|((name, _), mask)| mask[row].then_some(*name))
                    .collect();
                if owners.len() > 1 {
                    return Err(SpecError::new(format!(
                        "partitions.column '{column}': row {row} matches both '{}' and '{}'",
                        owners[0], owners[1]
                    )));
                }
            }
            let unknown: Vec<bool> = (0..n)
                .map(|row| !matches.iter().any(|mask| mask[row]))
                .collect();
            let predict_mask = matches.pop().expect("four partition lists");
            let val_mask = matches.pop().expect("four partition lists");
            let mut test_mask = matches.pop().expect("four partition lists");
            let mut train_mask = matches.pop().expect("four partition lists");
            if unknown.iter().any(|&u| u) {
                match p.unknown_policy {
                    UnknownPolicy::Error => {
                        return Err(SpecError::new(format!(
                            "partitions: unknown values in '{column}'"
                        )))
                    }
                    UnknownPolicy::Train => {
                        for (m, u) in train_mask.iter_mut().zip(&unknown) {
                            *m |= u;
                        }
                    }
                    UnknownPolicy::Test => {
                        for (m, u) in test_mask.iter_mut().zip(&unknown) {
                            *m |= u;
                        }
                    }
                    UnknownPolicy::Drop => {}
                }
            }
            let mut masks: IndexMap<String, Vec<bool>> = IndexMap::new();
            masks.insert("train".into(), train_mask);
            masks.insert("test".into(), test_mask);
            if !p.val_values.is_empty() {
                masks.insert("val".into(), val_mask);
            }
            if !p.predict_values.is_empty() {
                masks.insert("predict".into(), predict_mask);
            }
            Ok(masks
                .into_iter()
                .filter(|(_, m)| m.iter().any(|&b| b))
                .collect())
        }
        PartitionBy::Index => mask_from_index_lists(
            &[
                ("train", &p.train),
                ("test", &p.test),
                ("val", &p.val),
                ("predict", &p.predict),
            ],
            n,
            "partitions",
        ),
        PartitionBy::IndexFile => {
            // Index lists were read from index_file by the facade and supplied
            // in-memory. A referenced partition file that was not supplied is an
            // error (the facade only omits a key when the spec omits the file).
            let load = |part: &str, file: &Option<String>| -> Result<Value, SpecError> {
                match file {
                    Some(_) => match index_lists.get(part) {
                        Some(v) => Ok(Value::from(v.clone())),
                        None => Err(SpecError::new(format!(
                            "index file: '{part}' partition file content not supplied"
                        ))),
                    },
                    None => Ok(Value::Null),
                }
            };
            let train = load("train", &p.train_file)?;
            let test = load("test", &p.test_file)?;
            let val = load("val", &p.val_file)?;
            let predict = load("predict", &p.predict_file)?;
            mask_from_index_lists(
                &[
                    ("train", &train),
                    ("test", &test),
                    ("val", &val),
                    ("predict", &predict),
                ],
                n,
                "index file",
            )
        }
        PartitionBy::Files => Err(SpecError::new(
            "partitions.by=files is not supported by the assembler",
        )),
    }
}

fn mask_from_index_lists(
    by_part: &[(&str, &Value)],
    n: usize,
    source: &str,
) -> Result<IndexMap<String, Vec<bool>>, SpecError> {
    let mut masks: IndexMap<String, Vec<bool>> = IndexMap::new();
    let mut seen: HashMap<i64, String> = HashMap::new();
    for (part, raw) in by_part {
        let Some(arr) = raw.as_array() else {
            if raw.is_null() {
                continue;
            }
            return Err(SpecError::new(format!(
                "{source}: '{part}' must be a list of row indices"
            )));
        };
        // Python `[int(v) for v in raw]`: coerce ints/floats/int-strings, raise
        // on anything else (don't silently drop, which would omit requested rows).
        let idx_list: Vec<i64> = arr
            .iter()
            .map(|v| {
                crate::pyfmt::py_int(v).ok_or_else(|| {
                    SpecError::new(format!(
                        "{source}: '{part}' contains a non-integer index {}",
                        crate::pyfmt::py_repr(v)
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        if idx_list.iter().any(|&i| i < 0 || i >= n as i64) {
            return Err(SpecError::new(format!(
                "{source}: '{part}' contains out-of-range indices (n={n})"
            )));
        }
        if idx_list.iter().collect::<HashSet<_>>().len() != idx_list.len() {
            return Err(SpecError::new(format!(
                "{source}: '{part}' contains duplicate indices"
            )));
        }
        for &i in &idx_list {
            if let Some(other) = seen.get(&i) {
                if other != part {
                    return Err(SpecError::new(format!(
                        "{source}: row {i} listed in both '{other}' and '{part}' (partitions must be disjoint)"
                    )));
                }
            }
            seen.insert(i, part.to_string());
        }
        let mut m = vec![false; n];
        for &i in &idx_list {
            m[i as usize] = true;
        }
        masks.insert(part.to_string(), m);
    }
    if masks.is_empty() {
        return Err(SpecError::new(format!(
            "{source}: no partition lists provided"
        )));
    }
    Ok(masks)
}

fn attach_folds(
    spec: &DatasetSpec,
    fold_inline: Option<&[Fold]>,
    assembled: &mut AssembledDataset,
    combined_df: Option<&Frame>,
    train_rows: Option<&[usize]>,
) -> Result<(), SpecError> {
    let Some(folds) = &spec.folds else {
        return Ok(());
    };
    let block = assembled.blocks.get("train").ok_or_else(|| {
        SpecError::new("folds require a train block; only training rows may declare folds")
    })?;
    let n_rows = combined_df.map(|f| f.n_rows).unwrap_or(block.n_samples);
    let training: Vec<usize> = if combined_df.is_some() {
        train_rows.unwrap_or_default().to_vec()
    } else {
        (0..n_rows).collect()
    };
    let positions: HashMap<usize, i64> = training
        .iter()
        .enumerate()
        .map(|(pos, row)| (*row, pos as i64))
        .collect();
    let raw: Vec<Fold> = if !folds.inline.is_empty() {
        super::folds::parse_json(&Value::Array(folds.inline.clone()))?
    } else if let Some(column) = folds.column.as_deref().filter(|s| !s.is_empty()) {
        let combined = combined_df.ok_or_else(|| SpecError::new(format!("folds.column='{column}' requires a single combined frame (it does not apply with explicit per-source partitions)")))?;
        let col = combined.column(column).ok_or_else(|| {
            SpecError::new(format!(
                "folds.column='{column}' not found in the assembled data (columns: {:?})",
                combined.column_names()
            ))
        })?;
        let mut distinct: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, cell) in col.values.iter().enumerate() {
            if matches!(cell, Cell::Na) || matches!(cell, Cell::Float(f) if f.is_nan()) {
                continue;
            }
            if !positions.contains_key(&i) {
                return Err(SpecError::new(format!(
                    "folds.column='{column}': only training rows may have fold labels"
                )));
            }
            let key = cell.to_str_scalar();
            match distinct.iter_mut().find(|(k, _)| *k == key) {
                Some((_, rows)) => rows.push(i),
                None => distinct.push((key, vec![i])),
            }
        }
        distinct.sort_by(|a, b| a.0.cmp(&b.0));
        if distinct.is_empty() {
            return Err(SpecError::new(format!(
                "folds.column='{column}' has no non-NaN values"
            )));
        }
        distinct
            .into_iter()
            .map(|(_, val_rows)| {
                let set: HashSet<usize> = val_rows.iter().copied().collect();
                (
                    training
                        .iter()
                        .filter(|i| !set.contains(*i))
                        .map(|i| *i as i64)
                        .collect(),
                    val_rows.iter().map(|i| *i as i64).collect(),
                )
            })
            .collect()
    } else if folds.file.as_deref().filter(|s| !s.is_empty()).is_some() {
        fold_inline
            .ok_or_else(|| {
                SpecError::new("folds.file content not supplied (not browser-reachable)")
            })?
            .to_vec()
    } else {
        vec![]
    };
    for (train, val) in &raw {
        for (name, side) in [("train", train), ("val", val)] {
            let mut seen = HashSet::new();
            for &index in side {
                let row = usize::try_from(index)
                    .ok()
                    .filter(|i| *i < n_rows)
                    .ok_or_else(|| {
                        SpecError::new(format!(
                            "folds.{name}: index out of range for {n_rows} frame rows"
                        ))
                    })?;
                if !positions.contains_key(&row) {
                    return Err(SpecError::new(format!(
                        "folds.{name}: only training rows may declare folds"
                    )));
                }
                if !seen.insert(index) {
                    return Err(SpecError::new(format!("folds.{name}: duplicate indices")));
                }
            }
        }
        if train.iter().any(|i| val.contains(i)) {
            return Err(SpecError::new(
                "fold train and validation sides must be disjoint",
            ));
        }
    }
    let provenance = combined_df
        .or(block.metadata.as_ref())
        .and_then(|frame| fold_provenance_from_frame(spec, frame, &raw))
        .unwrap_or_default();
    let mapped = raw
        .iter()
        .map(|(train, val)| {
            (
                train.iter().map(|i| positions[&(*i as usize)]).collect(),
                val.iter().map(|i| positions[&(*i as usize)]).collect(),
            )
        })
        .collect();
    assembled.fold_provenance = provenance;
    assembled.folds = mapped;
    Ok(())
}

fn fold_provenance_from_frame(
    spec: &DatasetSpec,
    frame: &Frame,
    folds: &[Fold],
) -> Option<Vec<FoldProvenance>> {
    let identity_column = spec.sample_index.observation_id.as_deref().or_else(|| {
        (spec.sample_index.by.value() == "id")
            .then(|| spec.sample_index.key.as_str())
            .flatten()
    })?;
    let values = frame.column(identity_column)?;
    if values.values.len() != frame.n_rows {
        return None;
    }
    folds
        .iter()
        .map(|(train, validation)| {
            let translate = |indices: &[i64]| -> Option<Vec<String>> {
                indices
                    .iter()
                    .map(|index| {
                        usize::try_from(*index)
                            .ok()
                            .and_then(|index| values.values.get(index))
                            .and_then(|cell| match cell {
                                Cell::Na => None,
                                Cell::Float(value) if value.is_nan() => None,
                                _ => Some(cell.to_str_scalar()),
                            })
                            .filter(|value| !value.trim().is_empty())
                    })
                    .collect()
            };
            Some(FoldProvenance {
                train_observation_ids: translate(train)?,
                validation_observation_ids: translate(validation)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize::frame::{Cell, Column};
    use serde_json::json;

    #[test]
    fn selected_typed_infinite_metadata_refuses_without_reclassifying_missing_or_text() {
        for role in ["metadata", "ignore"] {
            let spec = DatasetSpec::from_value(&json!({
                "params": {"na": {"policy": "ignore"}},
                "sources": [{"id": "t", "input": "d.parquet", "columns": [
                    {"role": "features", "select": ["x"]},
                    {"role": role, "select": ["m"]}
                ]}]
            }))
            .unwrap();
            for value in [f64::INFINITY, f64::NEG_INFINITY] {
                let frame = Frame::from_columns(
                    vec![
                        Column::from_cells("x", vec![Cell::Float(1.5), Cell::Float(2.5)]),
                        Column::from_cells("m", vec![Cell::Float(value), Cell::Float(2.5)]),
                    ],
                    "text",
                );
                let result = assemble_in_memory(
                    &spec,
                    &[InMemorySource {
                        name: "d.parquet".into(),
                        payload: SourcePayload::Frame(frame),
                    }],
                    &HashMap::new(),
                    None,
                );
                if role == "metadata" {
                    assert_eq!(
                        result.unwrap_err().message,
                        "Observed numeric values must be finite"
                    );
                } else {
                    assert_eq!(result.unwrap().blocks["train"].n_samples, 2);
                }
            }
        }
        let spec = DatasetSpec::from_value(&json!({
            "params": {"na": {"policy": "ignore"}},
            "sources": [{"id": "t", "input": "d.parquet", "columns": [
                {"role": "features", "select": ["x"]},
                {"role": "metadata", "select": ["m"]}
            ]}]
        }))
        .unwrap();
        for (values, expected) in [
            (vec![Cell::Float(f64::NAN), Cell::Float(2.5)], Value::Null),
            (vec![Cell::Na, Cell::Float(2.5)], Value::Null),
            (
                vec![Cell::Str("inf".into()), Cell::Str("2.5".into())],
                json!("inf"),
            ),
        ] {
            let frame = Frame::from_columns(
                vec![
                    Column::from_cells("x", vec![Cell::Float(1.5), Cell::Float(2.5)]),
                    Column::from_cells("m", values),
                ],
                "text",
            );
            let assembled = assemble_in_memory(
                &spec,
                &[InMemorySource {
                    name: "d.parquet".into(),
                    payload: SourcePayload::Frame(frame),
                }],
                &HashMap::new(),
                None,
            )
            .unwrap();
            let metadata = assembled.blocks["train"].metadata.as_ref().unwrap();
            let column = metadata.column("m").unwrap();
            assert_eq!(
                super::super::frame::metadata_cell_value(column, 0, &column.values[0]),
                expected
            );
        }
    }

    #[test]
    fn typed_infinite_metadata_is_checked_only_after_na_and_join_row_removal() {
        for policy in ["na", "join"] {
            let spec = if policy == "na" {
                json!({
                    "params": {"na": {"policy": "remove_sample"}},
                    "sources": [{"id": "a", "input": "a.parquet", "columns": [
                        {"role": "features", "select": ["x", "z"]},
                        {"role": "metadata", "select": ["m"]},
                        {"role": "ignore", "select": ["k"]}
                    ]}]
                })
            } else {
                json!({
                    "params": {"delimiter": ",", "na": {"policy": "ignore"}},
                    "sources": [
                        {"id": "a", "input": "a.parquet", "columns": [
                            {"role": "features", "select": ["x"]},
                            {"role": "metadata", "select": ["m"]},
                            {"role": "ignore", "select": ["k", "z"]}
                        ]},
                        {"id": "s", "kind": "lookup", "input": "s.csv", "columns": [
                            {"role": "targets", "select": ["r"]}
                        ], "join": {"to": "a", "left_on": "k", "right_on": "k", "how": "m:1", "coverage": "drop"}}
                    ]
                })
            };
            let spec = DatasetSpec::from_value(&spec).unwrap();
            let frame = Frame::from_columns(
                vec![
                    Column::from_cells("k", vec![Cell::Str("Z".into()), Cell::Str("A".into())]),
                    Column::from_cells("x", vec![Cell::Float(1.5), Cell::Float(2.5)]),
                    Column::from_cells("z", vec![Cell::Na, Cell::Float(2.0)]),
                    Column::from_cells("m", vec![Cell::Float(f64::INFINITY), Cell::Float(2.5)]),
                ],
                "text",
            );
            let assembled = assemble_in_memory(
                &spec,
                &[
                    InMemorySource {
                        name: "a.parquet".into(),
                        payload: SourcePayload::Frame(frame),
                    },
                    InMemorySource {
                        name: "s.csv".into(),
                        payload: SourcePayload::Bytes(b"k,r\nA,0.5\n".to_vec()),
                    },
                ],
                &HashMap::new(),
                None,
            )
            .unwrap();
            let block = &assembled.blocks["train"];
            assert_eq!(block.n_samples, 1, "{policy}");
            assert_eq!(
                block.metadata.as_ref().unwrap().column("m").unwrap().values,
                vec![Cell::Float(2.5)],
                "{policy}"
            );
            assert_eq!(block.x[0].data[0], 2.5, "{policy}");
        }
    }

    #[test]
    fn sample_removal_aligns_all_sources_and_variations() {
        let spec = DatasetSpec::from_value(&json!({
            "params": {"na": {"policy": "remove_sample"}},
            "sources": [
                {"id": "x", "role": "features", "input": "x.csv", "variations": [{"name": "processed", "input": "v.csv"}]},
                {"id": "x2", "role": "features", "input": "x2.csv"},
                {"id": "y", "role": "targets", "input": "y.csv"},
                {"id": "m", "role": "metadata", "input": "m.csv"},
                {"id": "weights", "role": "weights", "input": "weights.csv"}
            ]
        })).unwrap();
        let sources: Vec<_> = [
            ("x", "w\n1\nNA\n3\n4\n5\n6\n"),
            ("x2", "z\n10\n20\n30\n40\n50\n60\n"),
            ("v", "w\n100\n200\nNA\n400\n500\n600\n"),
            ("y", "y\n1\n2\n3\n\n5\n6\n"),
            ("m", "label\na\nb\nc\nd\nNA\nf\n"),
            ("weights", "weight\n1\n2\n3\n4\n5\n6\n"),
        ]
        .into_iter()
        .map(|(name, text)| InMemorySource {
            name: format!("{name}.csv"),
            payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
        })
        .collect();
        let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
        let block = &assembled.blocks["train"];
        assert_eq!(block.n_samples, 2);
        assert_eq!(block.x[0].data, vec![1.0, 6.0]);
        assert_eq!(block.x[1].data, vec![10.0, 60.0]);
        assert_eq!(block.y.as_ref().unwrap().data, vec![1.0, 6.0]);
        assert_eq!(block.processings[0][0].1.data, vec![100.0, 600.0]);
        assert_eq!(block.weights.as_ref().unwrap(), &vec![1.0, 6.0]);
        assert_eq!(
            block.metadata.as_ref().unwrap().str_column("label"),
            vec!["a", "f"]
        );
    }

    #[test]
    fn metadata_keeps_missing_values_under_the_dataset_policy() {
        // nirs4all parity: metadata ignores the dataset NA policy unless it is remove_sample.
        let spec = DatasetSpec::from_value(&json!({
            "params": {"na": {"policy": "abort"}},
            "sources": [
                {"id": "x", "role": "features", "input": "x.csv"},
                {"id": "y", "role": "targets", "input": "y.csv"},
                {"id": "m", "role": "metadata", "input": "m.csv"}
            ]
        }))
        .unwrap();
        let sources: Vec<_> = [
            ("x", "w\n1\n2\n3\n"),
            ("y", "y\n1\n2\n3\n"),
            ("m", "location;site\n;s1\nfield;\n;s3\n"),
        ]
        .into_iter()
        .map(|(name, text)| InMemorySource {
            name: format!("{name}.csv"),
            payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
        })
        .collect();
        let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
        assert_eq!(assembled.blocks["train"].n_samples, 3);
    }

    fn csv_sources(files: &[(&str, &str)]) -> Vec<InMemorySource> {
        files
            .iter()
            .map(|(name, text)| InMemorySource {
                name: format!("{name}.csv"),
                payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
            })
            .collect()
    }

    fn assemble_xy(policy: &str, x: &str, y: &str) -> Result<AssembledDataset, SpecError> {
        let spec = DatasetSpec::from_value(&json!({
            "params": {"na": {"policy": policy}},
            "sources": [
                {"id": "x", "role": "features", "input": "x.csv"},
                {"id": "y", "role": "targets", "input": "y.csv"}
            ]
        }))
        .unwrap();
        assemble_in_memory(
            &spec,
            &csv_sources(&[("x", x), ("y", y)]),
            &HashMap::new(),
            None,
        )
    }

    fn assert_nan_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                a == e || (a.is_nan() && e.is_nan()),
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn missing_target_keeps_its_sample() {
        // Loading is lossless: a missing target is never imputed, dropped as a
        // column, made a class, or allowed to remove its sample.
        for policy in ["auto", "remove_feature", "replace"] {
            let assembled = assemble_xy(
                policy,
                "w;v\n1;10\n2;20\n3;30\n4;40\n",
                "y;c\n10;a\n\n30;NA\n40;b\n",
            )
            .unwrap();
            let block = &assembled.blocks["train"];
            assert_eq!(block.n_samples, 4, "{policy}");
            assert_eq!(block.y_headers, vec!["y", "c"], "{policy}");
            assert_eq!(
                block.x[0].data,
                vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
                "{policy}"
            );
            assert_nan_eq(
                &block.y.as_ref().unwrap().data,
                &[10.0, 0.0, f64::NAN, f64::NAN, 30.0, f64::NAN, 40.0, 1.0],
            );
            assert_eq!(block.y_categorical["c"], json!({"categories": ["a", "b"]}));
            assert!(assembled.warnings.is_empty(), "{policy}");
        }
    }

    #[test]
    fn abort_refuses_and_ignore_keeps_a_missing_target() {
        let err = assemble_xy("abort", "w\n1\n2\n3\n", "y\n10\n\n30\n").unwrap_err();
        assert!(err.message.contains("First NA in column 'y' (row: 1)"));
        let assembled = assemble_xy("ignore", "w\n1\n2\n3\n", "y\n10\n\n30\n").unwrap();
        let y = &assembled.blocks["train"].y.as_ref().unwrap().data;
        assert_eq!((y[0], y[2]), (10.0, 30.0));
        assert!(y[1].is_nan());
        assert!(assembled.warnings.is_empty());
    }

    #[test]
    fn remove_feature_only_drops_feature_columns() {
        let assembled = assemble_xy(
            "remove_feature",
            "w;v\n1;10\n2;NA\n3;30\n",
            "y\n10\n20\n30\n",
        )
        .unwrap();
        let block = &assembled.blocks["train"];
        assert_eq!(block.feature_headers, vec![vec!["w".to_string()]]);
        assert_eq!(block.y.as_ref().unwrap().data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn auto_still_aborts_on_a_missing_feature() {
        let err = assemble_xy("auto", "w\n1\nNA\n3\n", "y\n10\n20\n30\n").unwrap_err();
        assert!(err.message.contains("First NA in column 'w' (row: 1)"));
    }

    #[test]
    fn feature_policy_never_reaches_a_target_in_the_same_table() {
        // One table holds X and y: a blank row is a missing target and missing
        // features; the features follow the policy, the target stays missing.
        let assemble = |policy: &str, data: &str| {
            let spec = DatasetSpec::from_value(&json!({
                "params": {"na": {"policy": policy}},
                "sources": [{
                    "id": "d", "role": "mixed", "input": "data.csv",
                    "columns": [{"select": "y", "role": "targets"}, {"select": "rest", "role": "features"}]
                }]
            }))
            .unwrap();
            let sources = csv_sources(&[("data", data)]);
            assemble_in_memory(&spec, &sources, &HashMap::new(), None)
        };
        let blank_row = "y;w;v\n10;1;5\n\n30;3;7\n";
        let assembled = assemble("replace", blank_row).unwrap();
        let block = &assembled.blocks["train"];
        assert_eq!(block.x[0].data, vec![1.0, 5.0, 0.0, 0.0, 3.0, 7.0]);
        assert_nan_eq(&block.y.as_ref().unwrap().data, &[10.0, f64::NAN, 30.0]);
        let assembled = assemble("remove_feature", "y;w;v\n10;1;5\n;2;\n30;3;7\n").unwrap();
        let block = &assembled.blocks["train"];
        assert_eq!(block.feature_headers, vec![vec!["w"]]);
        assert_nan_eq(&block.y.as_ref().unwrap().data, &[10.0, f64::NAN, 30.0]);
        let err = assemble("auto", blank_row).unwrap_err();
        assert!(err.message.contains("First NA in column 'w' (row: 1)"));
    }

    #[test]
    fn scan_index_is_not_the_repetition_grouping_key() {
        let spec = |repetition: Value| {
            let mut spec = json!({
                "sample_index": {"by": "id", "key": "sample_id", "repetition_id": "rep"},
                "sources": [{"id": "x", "role": "mixed", "input": "x.csv", "columns": [
                    {"select": ["sample_id", "rep"], "role": "metadata"},
                    {"select": "rest", "role": "features"}
                ]}]
            });
            if !repetition.is_null() {
                spec["repetition"] = repetition;
            }
            DatasetSpec::from_value(&spec).unwrap()
        };
        let sources = csv_sources(&[("x", "sample_id;rep;w\ns1;1;0.1\ns1;2;0.2\ns2;1;0.3\n")]);
        let implicit =
            assemble_in_memory(&spec(Value::Null), &sources, &HashMap::new(), None).unwrap();
        assert_eq!(implicit.repetition, None);
        assert_eq!(implicit.identity.repetition_id.as_deref(), Some("rep"));
        assert!(implicit.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .has_column("rep"));
        let explicit =
            assemble_in_memory(&spec(json!("sample_id")), &sources, &HashMap::new(), None).unwrap();
        assert_eq!(explicit.repetition.as_deref(), Some("sample_id"));
    }

    #[test]
    fn sample_removal_applies_after_key_alignment() {
        for target in ["id;y\nc;30\na;10\nb;NA\n", "id;y\nc;b\na;a\nb;NA\n"] {
            let spec = DatasetSpec::from_value(&json!({"sources": [
                {"id": "x", "role": "features", "key": "id", "input": "x.csv"},
                {"id": "y", "role": "targets", "key": "id", "input": "y.csv", "params": {"na": {"policy": "remove_sample"}}}
            ]})).unwrap();
            let sources = vec![
                InMemorySource {
                    name: "x.csv".into(),
                    payload: SourcePayload::Bytes(b"id;w\na;1\nb;2\nc;3\n".to_vec()),
                },
                InMemorySource {
                    name: "y.csv".into(),
                    payload: SourcePayload::Bytes(target.as_bytes().to_vec()),
                },
            ];
            let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
            let block = &assembled.blocks["train"];
            assert_eq!(block.x[0].data, vec![1.0, 3.0]);
            assert_eq!(block.y.as_ref().unwrap().n_rows, 2);
            assert!(block.y.as_ref().unwrap().data.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn categorical_mapping_is_categories_dict() {
        // string target auto-encoded -> mapping is {"categories": [...]} (Python IR shape).
        let col = Column::from_cells(
            "class",
            vec![
                Cell::Str("a".into()),
                Cell::Str("b".into()),
                Cell::Str("a".into()),
            ],
        );
        let (codes, mapping) = encode_categorical(&col, "auto").unwrap();
        assert_eq!(codes, vec![0.0, 1.0, 0.0]);
        assert_eq!(mapping, Some(json!({"categories": ["a", "b"]})));
    }

    #[test]
    fn categorical_mapping_preserves_missing_observations() {
        let col = Column::from_cells(
            "class",
            vec![Cell::Str("a".into()), Cell::Na, Cell::Str("b".into())],
        );
        let (codes, mapping) = encode_categorical(&col, "auto").unwrap();
        assert_eq!(codes[0], 0.0);
        assert!(codes[1].is_nan());
        assert_eq!(codes[2], 1.0);
        assert_eq!(mapping, Some(json!({"categories": ["a", "b"]})));
    }

    #[test]
    fn float_target_has_no_mapping() {
        let col = Column::from_cells("y", vec![Cell::Float(12.5), Cell::Float(8.3)]);
        let (vals, mapping) = encode_categorical(&col, "auto").unwrap();
        assert_eq!(vals, vec![12.5, 8.3]);
        assert_eq!(mapping, None);
    }

    #[test]
    fn records_to_frame_drops_row_index_provenance() {
        // formats tags every decoded CSV row with metadata.row_index (0..n). It must
        // NOT survive into the assembled frame: the `auto` features selector takes
        // every unassigned numeric column, so a "0","1",… row_index column would be
        // coerced into a phantom feature, widening X by one and shifting the axis.
        let mk = |vals: [f64; 3], idx: i64| {
            json!({
                "signals": { "signal": {
                    "axis": { "values": [400.0, 402.0, 404.0], "unit": "nm" },
                    "values": vals,
                } },
                "metadata": { "row_index": idx, "sample_id": format!("s{idx}") }
            })
        };
        let records = vec![mk([0.40, 0.41, 0.42], 0), mk([0.50, 0.51, 0.52], 1)];
        let frame = records_to_frame(&records, &LoadingParams::default(), None).unwrap();
        let names = frame.column_names();
        assert!(
            !names.iter().any(|c| c == "row_index"),
            "row_index must be dropped, got {names:?}"
        );
        // exactly the three signal-axis features + the real sample_id metadata key.
        assert_eq!(names, vec!["400", "402", "404", "sample_id"]);
        assert_eq!(frame.n_rows, 2);
    }

    #[test]
    fn index_partition_coerces_and_rejects() {
        // Python int(): "2" and 3.0 coerce; "3.5" raises.
        let train = json!([0, 1, "2"]);
        let test = json!([3.0, 4]);
        let by = [("train", &train), ("test", &test)];
        let masks = mask_from_index_lists(&by, 5, "partitions").unwrap();
        assert_eq!(masks["train"], vec![true, true, true, false, false]);
        assert_eq!(masks["test"], vec![false, false, false, true, true]);

        let bad_v = json!(["3.5"]);
        let bad = [("train", &bad_v)];
        assert!(mask_from_index_lists(&bad, 5, "partitions").is_err());
    }

    #[test]
    fn glob_match_basics() {
        assert!(glob_match("*.csv", "data.csv"));
        assert!(glob_match("X*.csv", "Xcal.csv"));
        assert!(!glob_match("X*.csv", "Ycal.csv"));
        assert!(glob_match("data?.csv", "data1.csv"));
        assert!(glob_match("f[0-9].csv", "f3.csv"));
        assert!(!glob_match("f[0-9].csv", "fa.csv"));
    }
}

#[cfg(test)]
mod path_reference_tests {
    use super::*;
    use serde_json::json;

    fn source(name: &str, value: &str) -> InMemorySource {
        InMemorySource {
            name: name.into(),
            payload: SourcePayload::Bytes(format!("x; y\n{value};1.5\n").into_bytes()),
        }
    }

    #[test]
    fn exact_metacharacter_names_win_and_ambiguous_exact_names_refuse() {
        let sources = vec![source("run[2]/x.csv", "0.1"), source("run2/x.csv", "7.1")];
        let selected = resolve_inputs_mem(&json!("run[2]/x.csv"), &sources).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "run[2]/x.csv");
        let duplicate = vec![source("*.csv", "0.1"), source("*.csv", "7.1")];
        assert!(resolve_inputs_mem(&json!("*.csv"), &duplicate)
            .err()
            .unwrap()
            .message
            .contains("ambiguous"));
    }

    #[test]
    fn genuine_patterns_keep_order_and_name_deduplication() {
        let sources = vec![source("b.csv", "0.2"), source("a.csv", "0.1")];
        let selected = resolve_inputs_mem(&json!(["b.csv", "*.csv", "a.csv"]), &sources).unwrap();
        assert_eq!(
            selected.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["b.csv", "a.csv"]
        );
        let literal = vec![source("*.csv", "0.1"), source("a.csv", "7.1")];
        assert_eq!(
            resolve_inputs_mem(&json!("*.csv"), &literal).unwrap().len(),
            1
        );
    }

    #[test]
    fn public_memory_source_and_variation_select_literal_payloads() {
        let spec = DatasetSpec::from_value(&json!({"params":{"delimiter":";"},"sources":[{
            "id":"x","role":"features","input":"main.csv",
            "variations":[{"name":"filtered","input":"run[2]/x.csv"}]
        }]}))
        .unwrap();
        let sources = vec![
            source("main.csv", "0.5"),
            source("run[2]/x.csv", "0.1"),
            source("run2/x.csv", "7.1"),
        ];
        let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
        let block = &assembled.blocks["train"];
        assert_eq!(block.x[0].data[0].to_bits(), 0.5_f64.to_bits());
        assert_eq!(block.processings[0][0].0, "filtered");
        assert_eq!(
            block.processings[0][0].1.data[0].to_bits(),
            0.1_f64.to_bits()
        );
        assert_eq!(block.x[0].n_rows, 1);
    }
}

#[cfg(test)]
mod delimiter_resolution_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn named_bytes_and_variation_refuse_before_numeric_coercion() {
        for variation in [false, true] {
            let mut value =
                json!({"sources":[{"id":"train_x","role":"features","input":"Xcal.csv"}]});
            let mut sources = vec![InMemorySource {
                name: "Xcal.csv".into(),
                payload: SourcePayload::Bytes(if variation {
                    b"x;y\n1;2\n3;4\n".to_vec()
                } else {
                    b"x,y\n1,2\n3,4\n".to_vec()
                }),
            }];
            if variation {
                value["sources"][0]["variations"] =
                    json!([{"name":"alternate","input":"alternate.csv"}]);
                sources.push(InMemorySource {
                    name: "alternate.csv".into(),
                    payload: SourcePayload::Bytes(b"x,y\n1,2\n3,4\n".to_vec()),
                });
            }
            let error = assemble_in_memory(
                &DatasetSpec::from_value(&value).unwrap(),
                &sources,
                &HashMap::new(),
                None,
            )
            .unwrap_err();
            assert!(error.message.contains("train_x"));
            assert!(error.message.contains(if variation {
                "alternate.csv"
            } else {
                "Xcal.csv"
            }));
            assert!(error.message.contains("declare params.delimiter"));
        }
    }
}

#[cfg(test)]
mod validation_partition_tests {
    use super::*;
    use serde_json::json;

    fn spec(partitions: Value) -> DatasetSpec {
        DatasetSpec::from_value(&json!({"task_type":"regression","sources":[{"id":"d","role":"mixed","input":"all.csv",
            "columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["set","row"]}]}],
            "partitions":partitions})).unwrap()
    }

    fn source(unknown: bool) -> InMemorySource {
        let mut text = "x;y;set;row\n10.0;100.0;train;r0\n11.0;101.0;test;r1\n12.0;102.0;val;r2\n13.0;103.0;predict;r3\n14.0;104.0;train;r4\n15.0;105.0;test;r5\n16.0;106.0;val;r6\n17.0;107.0;predict;r7\n".to_string();
        if unknown {
            text.push_str("18.0;108.0;other;r8\n");
        }
        InMemorySource {
            name: "all.csv".into(),
            payload: SourcePayload::Bytes(text.into_bytes()),
        }
    }

    fn column(policy: &str) -> Value {
        json!({"by":"column","column":"set","train_values":["TRAIN"],"test_values":["test"],"val_values":["VaL"],"predict_values":["predict"],"unknown_policy":policy})
    }

    #[test]
    fn all_three_modes_preserve_eight_rows_values_and_order() {
        let lists: HashMap<String, Vec<i64>> = [
            ("train", vec![0, 4]),
            ("test", vec![1, 5]),
            ("val", vec![2, 6]),
            ("predict", vec![3, 7]),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect();
        for partitions in [
            column("error"),
            json!({"by":"index","train":[0,4],"test":[1,5],"val":[2,6],"predict":[3,7]}),
            json!({"by":"index_file","train_file":"train.json","test_file":"test.json","val_file":"val.json","predict_file":"predict.json"}),
        ] {
            let assembled =
                assemble_in_memory(&spec(partitions), &[source(false)], &lists, None).unwrap();
            assert_eq!(
                assembled
                    .blocks
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec!["train", "test", "val", "predict"]
            );
            assert_eq!(
                assembled
                    .blocks
                    .values()
                    .map(|b| b.n_samples)
                    .sum::<usize>(),
                8
            );
            for (part, x, y, rows) in [
                (
                    "train",
                    vec![10.0, 14.0],
                    vec![100.0, 104.0],
                    vec!["r0", "r4"],
                ),
                (
                    "test",
                    vec![11.0, 15.0],
                    vec![101.0, 105.0],
                    vec!["r1", "r5"],
                ),
                (
                    "val",
                    vec![12.0, 16.0],
                    vec![102.0, 106.0],
                    vec!["r2", "r6"],
                ),
                (
                    "predict",
                    vec![13.0, 17.0],
                    vec![103.0, 107.0],
                    vec!["r3", "r7"],
                ),
            ] {
                let block = &assembled.blocks[part];
                assert_eq!(block.n_samples, 2);
                assert_eq!(block.x[0].data, x);
                assert_eq!(block.x[0].validity, vec![true, true]);
                assert_eq!(block.y.as_ref().unwrap().data, y);
                assert_eq!(block.y.as_ref().unwrap().validity, vec![true, true]);
                assert_eq!(block.metadata.as_ref().unwrap().str_column("row"), rows);
                assert_eq!(block.source_ids, vec!["d"]);
            }
        }
    }

    #[test]
    fn validation_is_known_before_each_unknown_policy() {
        for policy in ["train", "test", "drop", "error"] {
            let value = spec(column(policy));
            let exact =
                assemble_in_memory(&value, &[source(false)], &HashMap::new(), None).unwrap();
            assert_eq!(
                exact
                    .blocks
                    .values()
                    .map(|b| b.n_samples)
                    .collect::<Vec<_>>(),
                vec![2, 2, 2, 2]
            );
            let extra = assemble_in_memory(&value, &[source(true)], &HashMap::new(), None);
            if policy == "error" {
                assert!(extra.unwrap_err().message.contains("unknown values"));
            } else {
                let extra = extra.unwrap();
                assert_eq!(
                    extra.blocks["val"].y.as_ref().unwrap().data,
                    vec![102.0, 106.0]
                );
                assert_eq!(
                    extra.blocks.values().map(|b| b.n_samples).sum::<usize>(),
                    if policy == "drop" { 8 } else { 9 }
                );
                if policy != "drop" {
                    assert_eq!(extra.blocks[policy].n_samples, 3);
                }
            }
        }
    }

    #[test]
    fn validation_only_and_missing_preloaded_file() {
        let lists = HashMap::from([("val".into(), vec![2, 6])]);
        for partitions in [
            json!({"by":"index","val":[2,6]}),
            json!({"by":"index_file","val_file":"val.json"}),
        ] {
            let assembled =
                assemble_in_memory(&spec(partitions), &[source(false)], &lists, None).unwrap();
            assert_eq!(
                assembled
                    .blocks
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec!["val"]
            );
            assert_eq!(
                assembled.blocks["val"].y.as_ref().unwrap().data,
                vec![102.0, 106.0]
            );
        }
        let error = assemble_in_memory(
            &spec(json!({"by":"index_file","val_file":"val.json"})),
            &[source(false)],
            &HashMap::new(),
            None,
        )
        .unwrap_err();
        assert!(error
            .message
            .contains("'val' partition file content not supplied"));
    }

    #[test]
    fn validation_refuses_only_its_new_overlap_and_uses_existing_index_checks() {
        for overlap in ["TRAIN", "TEST", "PREDICT"] {
            let mut partitions = column("error");
            partitions["val_values"] = json!([overlap]);
            let error =
                assemble_in_memory(&spec(partitions), &[source(false)], &HashMap::new(), None)
                    .unwrap_err();
            assert!(error.message.contains("appears in both"));
        }
        for (bad, diagnostic) in [
            (json!([2, 2]), "duplicate"),
            (json!([8]), "out-of-range"),
            (json!([-1]), "out-of-range"),
            (json!(2), "must be a list"),
            (json!([0]), "disjoint"),
            (json!(["bad"]), "non-integer"),
        ] {
            let error = assemble_in_memory(
                &spec(json!({"by":"index","train":[0],"val":bad})),
                &[source(false)],
                &HashMap::new(),
                None,
            )
            .unwrap_err();
            assert!(error.message.contains(diagnostic), "{error:?}");
        }
    }
}

#[cfg(test)]
mod strict_partition_fold_tests {
    use super::*;
    use serde_json::json;

    fn value(partitions: Value) -> Value {
        json!({"task_type":"regression","sample_index":{"by":"id","key":"row","observation_id":"row"},
            "sources":[{"id":"d","role":"mixed","input":"all.csv","columns":[{"role":"features","select":["x"]},
                {"role":"targets","select":["y"]},{"role":"metadata","select":["set","row"]}]}],"partitions":partitions})
    }

    fn source() -> InMemorySource {
        InMemorySource { name:"all.csv".into(), payload:SourcePayload::Bytes(b"x;y;set;row\n10.0;100.0;train;r0\n11.0;101.0;test;r1\n12.0;102.0;val;r2\n13.0;103.0;predict;r3\n14.0;104.0;train;r4\n15.0;105.0;test;r5\n16.0;106.0;val;r6\n17.0;107.0;predict;r7\n".to_vec()) }
    }

    fn fold_value(folds: Value) -> Value {
        let mut v =
            value(json!({"by":"index","train":[1,3,5],"test":[0,2],"val":[4,6],"predict":[7]}));
        v["folds"] = folds;
        v
    }

    #[test]
    fn all_six_column_pairs_refuse_at_validate_and_wasm_shaped_assembly() {
        let names = [
            "train_values",
            "test_values",
            "val_values",
            "predict_values",
        ];
        for i in 0..names.len() {
            for j in i + 1..names.len() {
                let mut p = json!({"by":"column","column":"set"});
                p[names[i]] = json!(["Absent"]);
                p[names[j]] = json!(["ABSENT"]);
                let spec = DatasetSpec::from_value(&value(p)).unwrap();
                for error in [
                    crate::spec::validate_spec(&spec).unwrap_err(),
                    assemble_in_memory(&spec, &[source()], &HashMap::new(), None).unwrap_err(),
                ] {
                    assert!(error.message.contains("'absent'"));
                    assert!(error.message.contains(names[i]) && error.message.contains(names[j]));
                }
            }
        }
    }

    #[test]
    fn column_matching_uses_original_scalar_normalization_without_trimming() {
        for (a, b, refused) in [
            (json!(1), json!("1"), true),
            (json!(true), json!("true"), true),
            (json!("TRAIN"), json!("train"), true),
            (json!(1), json!(1.0), false),
            (json!(" 1"), json!("1"), false),
        ] {
            let spec = DatasetSpec::from_value(&value(json!({"by":"column","column":"set","train_values":[a.clone(),a],"test_values":[b]}))).unwrap();
            assert_eq!(
                spec.partitions
                    .as_ref()
                    .unwrap()
                    .validate_column_values()
                    .is_err(),
                refused
            );
        }
    }

    #[test]
    fn closed_keys_arrays_and_modes_refuse_before_ir_loss() {
        for key in ["validation", "valid", "cal", "tset", "unknown_polcy"] {
            let mut p = json!({"by":"index","train":[0]});
            p[key] = json!([1]);
            assert!(DatasetSpec::from_value(&value(p))
                .unwrap_err()
                .message
                .contains("unknown keys"));
        }
        for p in [
            json!({"by":"column","column":"set","train":[0]}),
            json!({"by":"index","column":"set"}),
            json!({"by":"index","val_values":["val"]}),
            json!({"by":"index_file","val":[0]}),
            json!({"by":"files","val_file":"v.txt"}),
            json!({"train":[0]}),
            json!({"unknown_policy":"train"}),
        ] {
            assert!(DatasetSpec::from_value(&value(p))
                .unwrap_err()
                .message
                .contains("not applicable"));
        }
        for key in [
            "train_values",
            "test_values",
            "val_values",
            "predict_values",
        ] {
            for bad in [Value::Null, json!("train"), json!(1), json!({})] {
                let mut p = json!({"by":"column","column":"set"});
                p[key] = bad;
                assert!(DatasetSpec::from_value(&value(p))
                    .unwrap_err()
                    .message
                    .contains("must be an array"));
            }
        }
        for p in [
            json!({}),
            json!({"by":"index","train":[]}),
            json!({"by":"index","train":[],"test":[]}),
        ] {
            crate::spec::validate_spec(&DatasetSpec::from_value(&value(p)).unwrap()).unwrap();
        }
    }

    #[test]
    fn nonprefix_folds_map_to_training_positions_after_identity_capture() {
        let spec =
            DatasetSpec::from_value(&fold_value(json!({"inline":[{"train":[3,1],"val":[5]}]})))
                .unwrap();
        let data = assemble_in_memory(&spec, &[source()], &HashMap::new(), None).unwrap();
        assert_eq!(data.folds, vec![(vec![1, 0], vec![2])]);
        assert_eq!(
            data.fold_provenance[0].train_observation_ids,
            vec!["r3", "r1"]
        );
        assert_eq!(
            data.fold_provenance[0].validation_observation_ids,
            vec!["r5"]
        );
        assert_eq!(data.blocks["train"].x[0].data, vec![11.0, 13.0, 15.0]);
        assert_eq!(
            data.blocks["train"].y.as_ref().unwrap().data,
            vec![101.0, 103.0, 105.0]
        );
    }

    #[test]
    fn strict_inline_fold_domain_and_sides_refuse() {
        for bad in [
            json!(7),
            json!(0),
            json!(2),
            json!(99),
            json!(-1),
            json!(true),
            json!(1.0),
            json!("1"),
        ] {
            let spec =
                DatasetSpec::from_value(&fold_value(json!({"inline":[{"train":[1],"val":[bad]}]})))
                    .unwrap();
            assert!(assemble_in_memory(&spec, &[source()], &HashMap::new(), None).is_err());
        }
        for fold in [
            json!({"train":[0,1],"val":[2]}),
            json!({"train":[1,1],"val":[5]}),
            json!({"train":[1],"val":[5,5]}),
            json!({"train":[1],"val":[1]}),
            json!({"train":1,"val":[5]}),
        ] {
            let spec = DatasetSpec::from_value(&fold_value(json!({"inline":[fold]}))).unwrap();
            assert!(assemble_in_memory(&spec, &[source()], &HashMap::new(), None).is_err());
        }
    }

    #[test]
    fn all_file_forms_admit_training_rows_only_and_keep_declared_order() {
        for (fmt, text, expected) in [
            (
                "json",
                r#"[{"train":[3,1],"val":[5]}]"#,
                vec![(vec![1_i64, 0], vec![2])],
            ),
            ("txt", "3,1\n5\n", vec![(vec![1, 0], vec![2])]),
            (
                "csv",
                "fold_0,fold_1\n3,5\n1,\n",
                vec![(vec![1, 0], vec![2]), (vec![2], vec![0, 1])],
            ),
            (
                "csv",
                "sample_id,fold\n3,0\n1,0\n5,1\n",
                vec![(vec![2], vec![1, 0]), (vec![0, 1], vec![2])],
            ),
        ] {
            let folds = super::super::folds::parse_fold_str(text, fmt).unwrap();
            let spec =
                DatasetSpec::from_value(&fold_value(json!({"file":"f","format":fmt}))).unwrap();
            let data =
                assemble_in_memory(&spec, &[source()], &HashMap::new(), Some(&folds)).unwrap();
            assert_eq!(data.folds, expected);
        }
        for (fmt, text) in [
            ("json", r#"[{"train":[1],"val":[7]}]"#),
            ("txt", "1\n7\n"),
            ("csv", "fold_0,fold_1\n1,7\n"),
            ("csv", "sample_id,fold\n1,0\n7,1\n"),
        ] {
            let folds = super::super::folds::parse_fold_str(text, fmt).unwrap();
            let spec =
                DatasetSpec::from_value(&fold_value(json!({"file":"f","format":fmt}))).unwrap();
            assert!(
                assemble_in_memory(&spec, &[source()], &HashMap::new(), Some(&folds))
                    .unwrap_err()
                    .message
                    .contains("training rows")
            );
        }
    }

    #[test]
    fn column_folds_exclude_null_test_rows_and_keep_null_train_fitting() {
        let mut v = fold_value(json!({"column":"cv"}));
        v["sources"][0]["columns"][2]["select"] = json!(["set", "row", "cv"]);
        let spec = DatasetSpec::from_value(&v).unwrap();
        let text = "x;y;set;row;cv\n10.0;100.0;train;r0;\n11.0;101.0;test;r1;a\n12.0;102.0;val;r2;\n13.0;103.0;predict;r3;\n14.0;104.0;train;r4;\n15.0;105.0;test;r5;b\n16.0;106.0;val;r6;\n17.0;107.0;predict;r7;\n";
        let input = |text: &str| InMemorySource {
            name: "all.csv".into(),
            payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
        };
        let data = assemble_in_memory(&spec, &[input(text)], &HashMap::new(), None).unwrap();
        assert_eq!(
            data.folds,
            vec![(vec![1, 2], vec![0]), (vec![0, 1], vec![2])]
        );
        assert_eq!(
            data.fold_provenance[0].train_observation_ids,
            vec!["r3", "r5"]
        );
        let leak = text.replace("12.0;102.0;val;r2;", "12.0;102.0;val;r2;leak");
        assert!(
            assemble_in_memory(&spec, &[input(&leak)], &HashMap::new(), None)
                .unwrap_err()
                .message
                .contains("only training rows")
        );
    }

    #[test]
    fn per_source_namespace_empty_sides_and_missing_train_are_checked() {
        let mut v = fold_value(json!({"inline":[{"train":[0],"val":[2]}]}));
        v.as_object_mut().unwrap().remove("partitions");
        v["sources"][0]["partition"] = json!("train");
        let input = InMemorySource {
            name: "all.csv".into(),
            payload: SourcePayload::Bytes(
                b"x;y;set;row\n10.0;100.0;train;r0\n11.0;101.0;train;r1\n12.0;102.0;train;r2\n"
                    .to_vec(),
            ),
        };
        let spec = DatasetSpec::from_value(&v).unwrap();
        let data =
            assemble_in_memory(&spec, std::slice::from_ref(&input), &HashMap::new(), None).unwrap();
        assert_eq!(data.folds, vec![(vec![0], vec![2])]);
        assert_eq!(
            data.fold_provenance[0].validation_observation_ids,
            vec!["r2"]
        );
        v["folds"]["inline"][0]["val"] = json!([4]);
        assert!(assemble_in_memory(
            &DatasetSpec::from_value(&v).unwrap(),
            std::slice::from_ref(&input),
            &HashMap::new(),
            None
        )
        .is_err());
        v["sources"][0]["partition"] = json!("test");
        v["folds"] = json!({"inline":[{"train":[],"val":[]}]});
        assert!(assemble_in_memory(
            &DatasetSpec::from_value(&v).unwrap(),
            &[input],
            &HashMap::new(),
            None
        )
        .unwrap_err()
        .message
        .contains("require a train block"));
        let spec = DatasetSpec::from_value(&fold_value(json!({"inline":[{"train":[],"val":[5]}]})))
            .unwrap();
        assert_eq!(
            assemble_in_memory(&spec, &[source()], &HashMap::new(), None)
                .unwrap()
                .folds,
            vec![(vec![], vec![2])]
        );
    }
}

#[cfg(test)]
mod native_column_ownership_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collected_native_columns_drive_decimal_and_explicit_axis_coordinates() {
        let table = |id: &str, mut columns: Vec<Column>, roles: Vec<(String, String)>| {
            columns.push(Column::from_cells(
                &row_idx_col(id),
                vec![Cell::Int(0), Cell::Int(1)],
            ));
            SourceTable {
                source_id: id.into(),
                df: Frame::from_columns(columns, "text"),
                numeric_scope: NumericScope::Uniform('.'),
                roles,
                key: None,
                partition: None,
                kind: "table".into(),
                join: None,
                signal_type: None,
                header_unit: "text".into(),
                raw_unit: None,
                raw_coordinates: None,
                variations: vec![],
                remove_rows: None,
            }
        };
        let anchor = table(
            "anchor",
            vec![Column::from_cells(
                "0",
                vec![Cell::Int(100), Cell::Int(200)],
            )],
            vec![("0".into(), "features".into())],
        );
        let bare = Column::from_cells(
            "0",
            vec![Cell::Str("10,5".into()), Cell::Str("20,5".into())],
        );
        let literal = Column::from_cells(
            "train_x__0",
            vec![Cell::Str("11.5".into()), Cell::Str("21.5".into())],
        );
        let mut mixed = table(
            "train_x",
            vec![bare, literal],
            vec![
                ("train_x__0".into(), "features".into()),
                ("0".into(), "features".into()),
            ],
        );
        mixed.numeric_scope = NumericScope::Columns(IndexMap::from([
            ("0".into(), ','),
            ("train_x__0".into(), '.'),
        ]));
        mixed.raw_coordinates = Some(IndexMap::from([
            ("0".into(), json!(900.25)),
            ("train_x__0".into(), json!(1000.5)),
        ]));
        mixed.raw_unit = Some("nm".into());
        let tables = IndexMap::from([(&anchor.source_id, &anchor), (&mixed.source_id, &mixed)]);
        let spec=DatasetSpec::from_value(&json!({"sources":[{"id":"anchor","input":"a.csv","role":"features"},{"id":"train_x","input":"b.csv","role":"features"}]})).unwrap();
        let (block, _) = assemble_block(
            &tables,
            &spec,
            &mut vec![],
            &mut vec![],
            None,
            &NumericStorageOptions::default(),
        )
        .unwrap();
        assert_eq!(block.feature_headers[1], vec!["0", "train_x__0"]);
        assert_eq!(block.x[1].data, vec![10.5, 11.5, 20.5, 21.5]);
        assert_eq!(block.x[1].validity, vec![true, true, true, true]);
        assert_eq!(
            block.feature_axes[1][0]["coordinates"],
            json!([900.25, 1000.5])
        );
        assert_eq!(block.feature_axes[1][0]["unit"], "nm");
    }
}

#[cfg(test)]
mod partition_label_contract_tests {
    use super::*;
    use serde_json::json;

    fn spec(part: &str, values: Value, policy: &str) -> DatasetSpec {
        let mut partitions = json!({"by":"column","column":"set"});
        partitions[format!("{part}_values")] = values;
        if policy != "default" {
            partitions["unknown_policy"] = json!(policy);
        }
        DatasetSpec::from_value(&json!({"sources":[{"id":"d","input":"d.csv","role":"features"}],"partitions":partitions})).unwrap()
    }

    #[test]
    fn missing_variants_never_match_any_list_and_follow_all_policies() {
        let frame = Frame::from_columns(
            vec![Column::from_cells(
                "set",
                vec![
                    Cell::Str("NAN".into()),
                    Cell::Na,
                    Cell::Float(f64::NAN),
                    Cell::Str("other".into()),
                ],
            )],
            "text",
        );
        // Keep both real missing variants, rather than testing their global spelling.
        assert!(matches!(frame.columns[0].values[1], Cell::Na));
        assert!(matches!(frame.columns[0].values[2], Cell::Float(v) if v.is_nan()));
        for part in ["train", "test", "val", "predict"] {
            for policy in ["default", "train", "test", "drop", "error"] {
                let spec = spec(part, json!(["nan"]), policy);
                let result = split_partition_mask(&frame, &spec, &HashMap::new());
                if policy == "error" {
                    assert!(result.unwrap_err().message.contains("'set'"));
                    continue;
                }
                let masks = result.unwrap();
                for output in ["train", "test", "val", "predict"] {
                    let destination = if policy == "default" { "train" } else { policy };
                    let mut expected = Vec::new();
                    if output == part {
                        expected.push(0);
                    }
                    if output == destination {
                        expected.extend([1, 2, 3]);
                    }
                    let actual: Vec<usize> = masks
                        .get(output)
                        .into_iter()
                        .flatten()
                        .enumerate()
                        .filter_map(|(row, keep)| keep.then_some(row))
                        .collect();
                    assert_eq!(actual, expected, "{part}/{policy}/{output}");
                }
            }
        }
    }

    #[test]
    fn literal_none_empty_and_unicode_keys_are_observed_not_missing() {
        let frame = Frame::from_columns(
            vec![Column::from_cells(
                "set",
                vec![
                    Cell::Str("None".into()),
                    Cell::Str("".into()),
                    Cell::Na,
                    Cell::Float(f64::NAN),
                    Cell::Str("ΣΑΣ".into()),
                    Cell::Str("İ".into()),
                    Cell::Str("ß".into()),
                    Cell::Str("SS".into()),
                ],
            )],
            "text",
        );
        let masks = split_partition_mask(
            &frame,
            &spec(
                "test",
                json!(["none", "", "σας", "i\u{0307}", "ss"]),
                "default",
            ),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            masks["test"],
            vec![true, true, false, false, true, true, false, true]
        );
        assert_eq!(
            masks["train"],
            vec![false, false, true, true, false, false, true, false]
        );
        for cells in [
            vec![Cell::Float(1.0), Cell::Float(f64::NAN)],
            vec![Cell::Int(1), Cell::Na],
        ] {
            let frame = Frame::from_columns(vec![Column::from_cells("set", cells)], "text");
            assert!(split_partition_mask(
                &frame,
                &spec("test", json!(["none"]), "drop"),
                &HashMap::new()
            )
            .unwrap()
            .is_empty());
        }
    }

    #[test]
    fn invalid_label_declarations_refuse_at_direct_mask_boundary() {
        let frame = Frame::from_columns(
            vec![Column::from_cells(
                "set",
                vec![Cell::Str("anything".into())],
            )],
            "text",
        );
        for part in ["train", "test", "val", "predict"] {
            for bad in [Value::Null, json!([]), json!({})] {
                let error = split_partition_mask(
                    &frame,
                    &spec(part, json!([bad]), "drop"),
                    &HashMap::new(),
                )
                .unwrap_err();
                assert!(error.message.contains(&format!("{part}_values")));
            }
        }
    }

    #[test]
    fn public_csv_nan_literal_matches_but_decoded_missing_obeys_policy() {
        let text = b"x;y;set;row\n1;101;cal;r0\n2;102;NAN;r1\n3;103;;r2\n4;104;NA;r3\n5;105;none;r4\n6;106;val;r5\n7;107;\"nan\";r6\n";
        for policy in ["default", "train", "test", "drop", "error"] {
            let mut value = json!({
                "task_type":"regression",
                "sources":[{"id":"d","input":"d.csv","role":"mixed","params":{"delimiter":";","has_header":true},"columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["set","row"]}]}],
                "partitions":{"by":"column","column":"set","train_values":["cal"],"test_values":["nan"]}
            });
            if policy != "default" {
                value["partitions"]["unknown_policy"] = json!(policy);
            }
            let spec = DatasetSpec::from_value(&value).unwrap();
            let result = assemble_in_memory(
                &spec,
                &[InMemorySource {
                    name: "d.csv".into(),
                    payload: SourcePayload::Bytes(text.to_vec()),
                }],
                &HashMap::new(),
                None,
            );
            if policy == "error" {
                assert!(result.unwrap_err().message.contains("'set'"));
                continue;
            }
            let data = result.unwrap();
            let expected_train = if matches!(policy, "default" | "train") {
                vec!["r0", "r2", "r3", "r4", "r5", "r6"]
            } else {
                vec!["r0"]
            };
            let expected_test = if policy == "test" {
                vec!["r1", "r2", "r3", "r4", "r5", "r6"]
            } else {
                vec!["r1"]
            };
            for (part, expected) in [("train", expected_train), ("test", expected_test)] {
                let block = &data.blocks[part];
                let rows = &block
                    .metadata
                    .as_ref()
                    .unwrap()
                    .column("row")
                    .unwrap()
                    .values;
                assert_eq!(
                    rows.iter().map(Cell::to_str_scalar).collect::<Vec<_>>(),
                    expected
                );
                let expected_y: Vec<f64> = expected
                    .iter()
                    .map(|row| 101.0 + row[1..].parse::<f64>().unwrap())
                    .collect();
                assert_eq!(block.y.as_ref().unwrap().data, expected_y);
            }
        }
    }
}

#[cfg(test)]
mod partition_numeric_identity_tests {
    use super::*;
    use serde_json::json;

    fn matching_rows(column: Column, values: Value) -> Vec<usize> {
        let frame = Frame::from_columns(vec![column], "text");
        let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"d","input":"d.csv","role":"features"}],"partitions":{"by":"column","column":"set","test_values":values,"unknown_policy":"drop"}})).unwrap();
        split_partition_mask(&frame, &spec, &HashMap::new())
            .unwrap()
            .get("test")
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(row, keep)| keep.then_some(row))
            .collect()
    }

    fn floats(values: &[f64]) -> Column {
        Column::from_cells("set", values.iter().copied().map(Cell::Float).collect())
    }

    #[test]
    fn numeric_aliases_missing_default_policy_and_no_missing_integers() {
        for value in [json!(1), json!("1"), json!(1.0), json!("1.0"), json!("1e0")] {
            assert_eq!(
                matching_rows(floats(&[1., 2., f64::NAN, 1., 2.]), json!([value])),
                vec![0, 3]
            );
        }
        let integers = Column::from_cells(
            "set",
            vec![Cell::Int(1), Cell::Int(2), Cell::Int(3), Cell::Int(1)],
        );
        assert_eq!(matching_rows(integers, json!([1.0])), vec![0, 3]);
        let frame = Frame::from_columns(vec![floats(&[1., 2., f64::NAN, 1., 2.])], "text");
        let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"d","input":"d.csv","role":"features"}],"partitions":{"by":"column","column":"set","test_values":[2]}})).unwrap();
        let masks = split_partition_mask(&frame, &spec, &HashMap::new()).unwrap();
        assert_eq!(masks["train"], vec![true, false, true, true, false]);
        assert_eq!(masks["test"], vec![false, true, false, false, true]);
        assert_eq!(
            matching_rows(floats(&[1., 1.5, 2., 1.]), json!(["1.50"])),
            vec![1]
        );
    }

    #[test]
    fn signed_zero_exact_integers_unsigned_carriers_and_bool_are_distinct() {
        for (value, expected) in [
            (json!(0), vec![0, 2]),
            (json!(0.0), vec![0, 2]),
            (json!("0"), vec![0, 2]),
            (json!(-0.0), vec![1, 3]),
            (json!("-0"), vec![1, 3]),
            (json!("-0.0"), vec![1, 3]),
        ] {
            assert_eq!(
                matching_rows(floats(&[0., -0., 0., -0., 1.5]), json!([value])),
                expected
            );
        }
        let big = Column::from_cells(
            "set",
            vec![
                Cell::Int((1_i64 << 53) + 1),
                Cell::Int(1_i64 << 53),
                Cell::Int(5),
            ],
        );
        assert_eq!(matching_rows(big, json!([9007199254740992.0])), vec![1]);
        let mut unsigned = Column::from_cells(
            "set",
            vec![
                Cell::Str(u64::MAX.to_string()),
                Cell::Str("9007199254740993".into()),
                Cell::Na,
                Cell::Str("5".into()),
            ],
        );
        unsigned.numeric_text = vec![true, true, false, true];
        unsigned.source_dtype = Some("uint64".into());
        assert_eq!(matching_rows(unsigned, json!([u64::MAX])), vec![0]);
        let mut nullable = Column::from_cells(
            "set",
            vec![
                Cell::Str("9007199254740993".into()),
                Cell::Na,
                Cell::Str("7".into()),
                Cell::Str("7".into()),
            ],
        );
        nullable.numeric_text = vec![true, false, true, true];
        nullable.source_dtype = Some("int64".into());
        assert_eq!(matching_rows(nullable, json!([7.0])), vec![2, 3]);
        let boolean =
            Column::from_cells("set", vec![Cell::Bool(true), Cell::Bool(false), Cell::Na]);
        assert!(matching_rows(boolean, json!([1])).is_empty());
    }

    #[test]
    fn text_tokens_and_per_cell_frame_authority_do_not_become_numeric() {
        let cells = ["1", "1.0", "001", "a", "1e999", " 1", "+1", "1,5"];
        for (value, expected) in [
            (json!(1), vec![0]),
            (json!(1.0), vec![1]),
            (json!("001"), vec![2]),
            (json!("1e999"), vec![4]),
            (json!(" 1"), vec![5]),
            (json!("+1"), vec![6]),
            (json!("1,5"), vec![7]),
        ] {
            let col = Column::from_cells(
                "set",
                cells
                    .iter()
                    .map(|value| Cell::Str((*value).into()))
                    .collect(),
            );
            assert_eq!(matching_rows(col, json!([value])), expected);
        }
        let mixed = Column::from_cells(
            "set",
            vec![
                Cell::Int(1),
                Cell::Str("1".into()),
                Cell::Na,
                Cell::Float(1.),
                Cell::Str("1.0".into()),
            ],
        );
        assert_eq!(matching_rows(mixed, json!(["1e0"])), vec![0, 3]);
        for value in [
            json!("01"),
            json!("+1"),
            json!(".5"),
            json!(" 1"),
            json!("1 "),
            json!("1,5"),
            json!("1e999"),
            json!("1e-999"),
            json!(true),
        ] {
            assert!(matching_rows(floats(&[1., 2., f64::NAN, 1.]), json!([value])).is_empty());
        }
        assert_eq!(
            matching_rows(
                floats(&[1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
                json!(["inf"])
            ),
            vec![1]
        );
        assert_eq!(
            matching_rows(
                floats(&[1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
                json!(["-inf"])
            ),
            vec![2]
        );
        assert!(matching_rows(
            floats(&[1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]),
            json!(["none"])
        )
        .is_empty());
    }

    #[test]
    fn float32_declaration_rounding_uses_owner_integer_and_float_storage() {
        for value in [
            json!(0.1),
            json!("0.1"),
            json!(0.10000000149011612),
            json!(0.10000000000001),
        ] {
            let mut col = floats(&[
                f64::from(0.1f32),
                f64::from(0.2f32),
                f64::NAN,
                f64::from(0.1f32),
            ]);
            col.source_dtype = Some("float32".into());
            assert_eq!(matching_rows(col, json!([value])), vec![0, 3]);
        }
        for value in [json!(1e100), json!(1e-100)] {
            let mut col = floats(&[0., f64::from(0.1f32)]);
            col.source_dtype = Some("float32".into());
            assert!(matching_rows(col, json!([value])).is_empty());
        }
        let mut col = floats(&[9_223_372_036_854_775_808., 9_223_373_136_366_403_584.]);
        col.source_dtype = Some("float32".into());
        assert_eq!(
            matching_rows(col, json!([9_223_372_586_610_589_697u64])),
            vec![1]
        );
        let mut col = floats(&[16_777_216., 16_777_218.]);
        col.source_dtype = Some("float32".into());
        assert_eq!(matching_rows(col, json!([16_777_217])), vec![0]);
    }

    #[test]
    fn runtime_numeric_overlap_refuses_all_six_pairs_but_text_and_absence_pass() {
        for (left, right) in [
            ("train_values", "test_values"),
            ("train_values", "val_values"),
            ("train_values", "predict_values"),
            ("test_values", "val_values"),
            ("test_values", "predict_values"),
            ("val_values", "predict_values"),
        ] {
            let mut value = json!({"sources":[{"id":"d","input":"d.csv","role":"features"}],"partitions":{"by":"column","column":"set","unknown_policy":"drop"}});
            value["partitions"][left] = json!([1]);
            value["partitions"][right] = json!(["1.0"]);
            let spec = DatasetSpec::from_value(&value).unwrap();
            spec.partitions
                .as_ref()
                .unwrap()
                .validate_column_values()
                .unwrap();
            let numeric = Frame::from_columns(vec![floats(&[1., 2., f64::NAN])], "text");
            let error = split_partition_mask(&numeric, &spec, &HashMap::new()).unwrap_err();
            assert!(
                error.message.contains("set")
                    && error.message.contains(left)
                    && error.message.contains(right)
            );
            let text = Frame::from_columns(
                vec![Column::from_cells(
                    "set",
                    vec![Cell::Str("1".into()), Cell::Str("1.0".into())],
                )],
                "text",
            );
            let masks = split_partition_mask(&text, &spec, &HashMap::new()).unwrap();
            assert_eq!(masks.len(), 2);
            let absent = Frame::from_columns(vec![floats(&[2., 3., f64::NAN])], "text");
            assert!(split_partition_mask(&absent, &spec, &HashMap::new())
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn float32_secondary_source_keyed_join_retains_owner_dtype() {
        for (keyed, role) in [
            (false, "ignore"),
            (true, "ignore"),
            (false, "metadata"),
            (true, "metadata"),
        ] {
            let left = Frame::from_columns(
                vec![
                    Column::from_cells(
                        "x",
                        vec![
                            Cell::Float(1.),
                            Cell::Float(2.),
                            Cell::Float(3.),
                            Cell::Float(4.),
                        ],
                    ),
                    Column::from_cells(
                        "y",
                        vec![
                            Cell::Float(101.),
                            Cell::Float(102.),
                            Cell::Float(103.),
                            Cell::Float(104.),
                        ],
                    ),
                    Column::from_cells(
                        "sid",
                        ["s0", "s1", "s2", "s3"]
                            .iter()
                            .map(|s| Cell::Str((*s).into()))
                            .collect(),
                    ),
                    Column::from_cells(
                        "row",
                        ["r0", "r1", "r2", "r3"]
                            .iter()
                            .map(|s| Cell::Str((*s).into()))
                            .collect(),
                    ),
                ],
                "text",
            );
            let mut col = floats(&[
                f64::from(0.1f32),
                f64::from(0.2f32),
                f64::NAN,
                f64::from(0.1f32),
            ]);
            col.source_dtype = Some("float32".into());
            let mut columns = vec![col];
            if keyed {
                columns.push(Column::from_cells(
                    "sid",
                    ["s0", "s1", "s2", "s3"]
                        .iter()
                        .map(|s| Cell::Str((*s).into()))
                        .collect(),
                ));
            }
            let mut right = Frame::from_columns(columns, "text");
            if keyed {
                for column in &mut right.columns {
                    column.values.reverse();
                    column.numeric_text.reverse();
                }
            }
            for value in [
                json!(0.1),
                json!("0.1"),
                json!(0.10000000149011612),
                json!(0.10000000000001),
            ] {
                let mut spec = json!({"task_type":"regression","sources":[{"id":"left","input":"left.parquet","role":"mixed","columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":"metadata","select":["sid","row"]}]},{"id":"labels","input":"labels.parquet","role":"mixed","params":{"na":{"policy":"ignore"}},"columns":[{"role":role,"select":["set"]}]}],"partitions":{"by":"column","column":"set","test_values":[value],"unknown_policy":"drop"}});
                if keyed {
                    spec["sources"][0]["key"] = json!("sid");
                    spec["sources"][1]["key"] = json!("sid");
                    spec["sources"][1]["join"] = json!({"to":"left","on":"sid","how":"1:1"});
                }
                let data = assemble_in_memory(
                    &DatasetSpec::from_value(&spec).unwrap(),
                    &[
                        InMemorySource {
                            name: "left.parquet".into(),
                            payload: SourcePayload::Frame(left.clone()),
                        },
                        InMemorySource {
                            name: "labels.parquet".into(),
                            payload: SourcePayload::Frame(right.clone()),
                        },
                    ],
                    &HashMap::new(),
                    None,
                )
                .unwrap();
                assert_eq!(
                    data.blocks["test"].y.as_ref().unwrap().data,
                    vec![101., 104.]
                );
                assert_eq!(
                    data.blocks["test"]
                        .metadata
                        .as_ref()
                        .unwrap()
                        .column("row")
                        .unwrap()
                        .values
                        .iter()
                        .map(Cell::to_str_scalar)
                        .collect::<Vec<_>>(),
                    vec!["r0", "r3"]
                );
            }
        }
    }
}
