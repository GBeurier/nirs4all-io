// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Assemble a DatasetSpec into a target-agnostic dataset (ports `materialize/assemble.py`).
//!
//! Per source: load → merge multi-file → split columns by role. Then per
//! partition: align feature sources (multi-source X), join targets/metadata/
//! lookups onto the sample rows, collect X/y/metadata/weights. The result is an
//! `AssembledDataset` IR that `to_spectrodataset`/`to_dag_ml_data` adapt — kept
//! target-agnostic so assembly is testable without nirs4all.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use nirs4all_io_core::conventions::engine::get_stem;
use nirs4all_io_core::spec::dataset_spec::{
    AggregateSpec, DatasetSpec, JoinSpec, SourceSpec, VariationSpec,
};
use nirs4all_io_core::spec::enums::{
    Cardinality, Coverage, MergeMode, PartitionBy, Role, SourceKind, UnknownPolicy,
};
use nirs4all_io_core::spec::selectors::Selector;
use nirs4all_io_core::spec::SpecError;
use serde_json::{Map, Value};

use super::folds::{parse_fold_file, Fold};
use super::frame::{Cell, Column, Frame, Matrix};
use super::join::{concat_features, concat_samples, join_tables, merge_by_key};
use super::loaders::{effective_params, read_table};

fn row_idx_col(source_id: &str) -> String {
    format!("__src_row_idx__{source_id}__")
}

/// `(frame, header_unit, signal_type, origins)` from a source's file load.
type LoadedSource = (Frame, String, Option<String>, Option<Vec<String>>);

#[derive(Debug, Clone, Default)]
pub struct PartitionBlock {
    pub n_samples: usize,
    pub x: Vec<Matrix>,
    pub feature_headers: Vec<Vec<String>>,
    pub header_units: Vec<String>,
    pub signal_types: Vec<Option<String>>,
    pub processings: Vec<Vec<(String, Matrix)>>,
    pub y: Option<Matrix>,
    pub y_headers: Vec<String>,
    pub y_categorical: IndexMap<String, Value>,
    pub metadata: Option<Frame>,
    pub weights: Option<Vec<f32>>,
    pub weights_header: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AssembledDataset {
    pub name: String,
    pub task_type: String,
    pub signal_type: String,
    pub n_sources: usize,
    pub blocks: IndexMap<String, PartitionBlock>,
    pub folds: Vec<Fold>,
    pub repetition: Option<String>,
    pub aggregate: Option<AggregateSpec>,
    pub warnings: Vec<String>,
    pub audits: Vec<Value>,
}

impl AssembledDataset {
    /// A canonical structural summary (shapes, headers, partitions, rounded
    /// X/y values, folds) for cross-language golden comparison. Mirrors
    /// `tests/test_assembled_goldens.py`.
    pub fn to_summary_value(&self) -> Value {
        use nirs4all_io_core::pyfmt::py_round;
        let mat_rows = |m: &Matrix| -> Value {
            Value::Array(
                (0..m.n_rows)
                    .map(|r| {
                        Value::Array(
                            m.data[r * m.n_cols..(r + 1) * m.n_cols]
                                .iter()
                                .map(|&v| Value::from(py_round(v as f64, 4)))
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
            "name": self.name,
            "task_type": self.task_type,
            "signal_type": self.signal_type,
            "n_sources": self.n_sources,
            "blocks": Value::Object(blocks),
            "folds": self.folds.iter().map(|(tr, vl)| vec![tr.clone(), vl.clone()]).collect::<Vec<_>>(),
            "repetition": self.repetition,
            "aggregate": aggregate,
        })
    }
}

struct SourceTable {
    source_id: String,
    df: Frame,
    /// column -> role, in `_split_roles` order.
    roles: Vec<(String, String)>,
    key: Option<Vec<String>>,
    partition: Option<String>,
    kind: String,
    join: Option<JoinSpec>,
    signal_type: Option<String>,
    header_unit: String,
    variations: Vec<(String, Matrix)>,
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

fn resolve_inputs(input: &Value, base_dir: &Path) -> Vec<PathBuf> {
    let items: Vec<&Value> = match input {
        Value::Array(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut out = Vec::new();
    for item in items {
        let Some(text) = item.as_str() else { continue };
        if is_glob(text) {
            let pat = base_dir.join(text);
            let mut g: Vec<PathBuf> = glob::glob(&pat.to_string_lossy())
                .map(|p| p.filter_map(Result::ok).collect())
                .unwrap_or_default();
            if g.is_empty() {
                g = glob::glob(text)
                    .map(|p| p.filter_map(Result::ok).collect())
                    .unwrap_or_default();
            }
            g.sort();
            out.extend(g);
        } else {
            let p = base_dir.join(text);
            out.push(if p.exists() { p } else { PathBuf::from(text) });
        }
    }
    out
}

fn load_source_frame(
    source: &SourceSpec,
    spec: &DatasetSpec,
    base_dir: &Path,
    audits: &mut Vec<Value>,
) -> Result<LoadedSource, SpecError> {
    let params = effective_params(&spec.params, &source.params);
    let paths = resolve_inputs(&source.input, base_dir);
    if paths.is_empty() {
        return Err(SpecError::new(format!(
            "source '{}': no input files resolved from {}",
            source.id,
            nirs4all_io_core::pyfmt::py_repr(&source.input)
        )));
    }
    let frames: Vec<Frame> = paths
        .iter()
        .map(|p| read_table(p, &params))
        .collect::<Result<_, _>>()?;
    let header_unit = frames[0].header_unit.clone();
    let signal = params.signal_type.map(|s| s.value().to_string());
    if frames.len() == 1 {
        return Ok((frames[0].clone(), header_unit, signal, None));
    }
    let names: Vec<String> = paths
        .iter()
        .map(|p| {
            get_stem(
                &p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            )
        })
        .collect();
    let (df, origins): (Frame, Option<Vec<String>>) = match source.merge {
        MergeMode::ConcatSamples => {
            let (df, origins, a) = concat_samples(&frames, &names);
            audits.push(serde_json::json!({"source": source.id, "merge": a.operation, "warnings": a.warnings}));
            (df, Some(origins))
        }
        MergeMode::ConcatFeatures => {
            let (df, a) = concat_features(&frames, &names, key_list(&source.key).as_deref())?;
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
            let (df, a) = merge_by_key(&frames, &names, &key, Coverage::Complete)?;
            audits.push(serde_json::json!({"source": source.id, "merge": a.operation, "warnings": a.warnings}));
            (df, None)
        }
        MergeMode::None => {
            return Err(SpecError::new(format!(
                "source '{}': multi-file input requires a merge mode",
                source.id
            )))
        }
    };
    Ok((df, header_unit, signal, origins))
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
    base_dir: &Path,
    audits: &mut Vec<Value>,
) -> Result<SourceTable, SpecError> {
    let (mut df, header_unit, signal, origins) = load_source_frame(source, spec, base_dir, audits)?;

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
    if let Some(key) = si.key.as_str() {
        roles.retain(|(c, _)| c != key);
    }
    let row_idx_name = row_idx_col(&source.id);
    if df.has_column(&row_idx_name) {
        return Err(SpecError::new(format!(
            "source '{}': reserved column name '{row_idx_name}' collides with a user-supplied column; rename it",
            source.id
        )));
    }
    df.add_int_column(&row_idx_name, (0..df.n_rows as i64).collect());

    let key = key_list(&source.key);
    let variations = load_variations(source, spec, base_dir, &df, &roles)?;
    let spec_signal = if spec.signal_type.value() != "auto" {
        Some(spec.signal_type.value().to_string())
    } else {
        None
    };
    Ok(SourceTable {
        source_id: source.id.clone(),
        df,
        roles,
        key,
        partition: source.partition.map(|p| p.value().to_string()),
        kind: source.kind.value().to_string(),
        join: source.join.clone(),
        signal_type: signal.or(spec_signal),
        header_unit,
        variations,
    })
}

fn read_variation_frame(
    variation: &VariationSpec,
    spec: &DatasetSpec,
    source: &SourceSpec,
    base_dir: &Path,
) -> Result<Frame, SpecError> {
    let mut params = effective_params(&spec.params, &source.params);
    if !variation.params.is_empty_value() {
        params = effective_params(&params, &variation.params);
    }
    let paths = resolve_inputs(&variation.input, base_dir);
    if paths.is_empty() {
        return Err(SpecError::new(format!(
            "source '{}': variation '{}': no input files resolved",
            source.id, variation.name
        )));
    }
    let frames: Vec<Frame> = paths
        .iter()
        .map(|p| read_table(p, &params))
        .collect::<Result<_, _>>()?;
    if frames.len() > 1 {
        let names: Vec<String> = (0..frames.len()).map(|i| i.to_string()).collect();
        Ok(concat_samples(&frames, &names).0)
    } else {
        Ok(frames.into_iter().next().unwrap())
    }
}

fn load_variations(
    source: &SourceSpec,
    spec: &DatasetSpec,
    base_dir: &Path,
    parent_df: &Frame,
    roles: &[(String, String)],
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
        let vdf = read_variation_frame(variation, spec, source, base_dir)?;
        if vdf.n_rows != parent_n {
            return Err(SpecError::new(format!(
                "source '{}': variation '{}' has {} rows, expected {parent_n} (must row-align with the parent source)",
                source.id, variation.name, vdf.n_rows
            )));
        }
        let named = feature_cols.iter().filter(|c| vdf.has_column(c)).count();
        let arr = if named == feature_cols.len() {
            vdf.coerce_numeric(&feature_cols)
        } else if named == 0 && vdf.columns.len() == feature_cols.len() {
            vdf.coerce_numeric(&vdf.column_names())
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

/// Load, role-split, join and partition `spec` into an [`AssembledDataset`].
pub fn assemble(spec: &DatasetSpec, base_dir: &Path) -> Result<AssembledDataset, SpecError> {
    let mut audits: Vec<Value> = Vec::new();
    let mut tables: IndexMap<String, SourceTable> = IndexMap::new();
    for s in &spec.sources {
        tables.insert(
            s.id.clone(),
            build_source_table(s, spec, base_dir, &mut audits)?,
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
    let repetition = spec.repetition.clone().or_else(|| {
        spec.sample_index
            .repetition_id
            .clone()
            .filter(|r| r != "auto")
    });
    let mut assembled = AssembledDataset {
        name: spec.name.clone().unwrap_or_else(|| "dataset".into()),
        task_type: spec.task_type.value().to_string(),
        signal_type: spec.signal_type.value().to_string(),
        n_sources,
        blocks: IndexMap::new(),
        folds: vec![],
        repetition,
        aggregate: spec.aggregate.clone(),
        warnings: vec![],
        audits: vec![],
    };

    let mut combined_df: Option<Frame> = None;
    if !partitions.is_empty() {
        for part in &partitions {
            let part_tables: IndexMap<&String, &SourceTable> = tables
                .iter()
                .filter(|(_, t)| {
                    t.partition.as_deref() == Some(part) || t.kind == SourceKind::Lookup.value()
                })
                .collect();
            let (block, _) =
                assemble_block(&part_tables, spec, &mut audits, &mut assembled.warnings)?;
            assembled.blocks.insert(part.clone(), block);
        }
    } else {
        let all: IndexMap<&String, &SourceTable> = tables.iter().collect();
        let (block, combined) = assemble_block(&all, spec, &mut audits, &mut assembled.warnings)?;
        if spec.partitions.is_none() {
            assembled.blocks.insert("train".into(), block);
        } else {
            for (part, mask) in split_partition_mask(&combined, spec, base_dir)? {
                assembled.blocks.insert(part, slice_block(&block, &mask));
            }
        }
        combined_df = Some(combined);
    }

    assembled.audits = audits;
    attach_folds(spec, base_dir, &mut assembled, combined_df.as_ref())?;
    Ok(assembled)
}

fn row_align(combined: &Frame, t: &SourceTable, primary_id: &str) -> Result<Frame, SpecError> {
    if t.df.n_rows != combined.n_rows {
        return Err(SpecError::new(format!(
            "source '{}' ({} rows) is not row-aligned with '{primary_id}' ({} rows); supply a join key",
            t.source_id, t.df.n_rows, combined.n_rows
        )));
    }
    let (merged, _) = concat_features(
        &[combined.clone(), t.df.clone()],
        &[primary_id.to_string(), t.source_id.clone()],
        None,
    )?;
    Ok(merged)
}

fn join_onto(
    combined: &Frame,
    primary: &SourceTable,
    t: &SourceTable,
    audits: &mut Vec<Value>,
    warnings: &mut Vec<String>,
) -> Result<Frame, SpecError> {
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
            let (out, audit) = join_tables(
                combined,
                &t.df,
                left_on,
                right_on,
                j.cardinality,
                j.coverage,
                &left_name,
                &t.source_id,
            )?;
            audits.push(serde_json::json!({"join": audit.operation, "dropped": audit.dropped_rows.len(), "warnings": audit.warnings}));
            warnings.extend(audit.warnings);
            return Ok(out);
        }
        if j.cardinality == Cardinality::OneToOne {
            return row_align(combined, t, &primary.source_id);
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
            let (out, audit) = join_tables(
                combined,
                &t.df,
                lkeys,
                rkeys,
                Cardinality::OneToOne,
                Coverage::Complete,
                &primary.source_id,
                &t.source_id,
            )?;
            warnings.extend(audit.warnings);
            return Ok(out);
        }
    }
    row_align(combined, t, &primary.source_id)
}

/// column -> (role, owning_source_id), mirroring `_collect_roles` verbatim.
///
/// Edge case (reproduced from Python for parity): if a non-primary source's role
/// column name collides with a bare (unassigned/key) column already in
/// `combined`, the bare name is recorded even though the join suffixed the
/// source's column to `name__<source>`. This can mis-own the column — but it is
/// the Python behavior, so it stays for byte/parity fidelity; do not "fix" here.
fn collect_roles(
    combined: &Frame,
    tables: &IndexMap<&String, &SourceTable>,
) -> IndexMap<String, (String, String)> {
    let mut role_cols: IndexMap<String, (String, String)> = IndexMap::new();
    for t in tables.values() {
        for (col, role) in &t.roles {
            if combined.has_column(col) && !role_cols.contains_key(col) {
                role_cols.insert(col.clone(), (role.clone(), t.source_id.clone()));
            } else {
                let ns = format!("{col}__{}", t.source_id);
                if combined.has_column(&ns) {
                    role_cols.insert(ns, (role.clone(), t.source_id.clone()));
                }
            }
        }
    }
    role_cols
}

fn assemble_block(
    tables: &IndexMap<&String, &SourceTable>,
    spec: &DatasetSpec,
    audits: &mut Vec<Value>,
    warnings: &mut Vec<String>,
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
    for t in tables.values() {
        if t.source_id == primary.source_id {
            continue;
        }
        combined = join_onto(&combined, primary, t, audits, warnings)?;
    }

    let role_cols = collect_roles(&combined, tables);
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
        block.x.push(combined.coerce_numeric(&cols));
        block.feature_headers.push(cols);
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
                let data: Vec<f32> = row_idx
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
        let mode = spec
            .params
            .categorical
            .map(|c| c.value().to_string())
            .unwrap_or_else(|| "auto".into());
        let mut y_cols: Vec<Vec<f64>> = Vec::new();
        let mut cats: IndexMap<String, Value> = IndexMap::new();
        for c in &target_cols {
            let (enc, mapping) = encode_categorical(combined.column(c).unwrap(), &mode);
            y_cols.push(enc);
            if let Some(m) = mapping {
                cats.insert(c.clone(), m);
            }
        }
        block.y = Some(column_stack(&y_cols));
        block.y_headers = target_cols;
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
        block.weights = Some(
            combined
                .numeric_column_f64(&weight_cols[0])
                .iter()
                .map(|&v| v as f32)
                .collect(),
        );
        block.weights_header = Some(weight_cols[0].clone());
    }

    let meta_cols: Vec<String> = combined_names
        .iter()
        .filter(|c| {
            role_cols
                .get(*c)
                .map(|(r, _)| r == "metadata")
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if !meta_cols.is_empty() {
        block.metadata = Some(combined.select(&meta_cols));
    }
    Ok((block, combined))
}

/// `encode_categorical`: factorize an object target when `auto` + more NaNs under
/// numeric coercion; else numeric coercion (f64). The mapping (when categorical)
/// is `{"categories": [...]}`, matching the Python IR.
fn encode_categorical(col: &Column, mode: &str) -> (Vec<f64>, Option<Value>) {
    let numeric: Vec<f64> = col.values.iter().map(Cell::to_numeric).collect();
    let is_objectish = !col.is_numeric_dtype();
    let numeric_na = numeric.iter().filter(|v| v.is_nan()).count();
    let series_na = col
        .values
        .iter()
        .filter(|c| matches!(c, Cell::Na) || matches!(c, Cell::Float(f) if f.is_nan()))
        .count();
    let treat_categorical = mode == "auto" && is_objectish && numeric_na > series_na;
    if treat_categorical {
        // pandas factorize over str(values): codes in first-appearance order.
        let mut categories: Vec<String> = Vec::new();
        let mut codes = Vec::with_capacity(col.values.len());
        for v in &col.values {
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
        (codes, Some(serde_json::json!({ "categories": categories })))
    } else {
        (numeric, None)
    }
}

fn column_stack(cols: &[Vec<f64>]) -> Matrix {
    let n_cols = cols.len();
    let n_rows = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for row in 0..n_rows {
        for c in cols {
            data.push(c[row] as f32);
        }
    }
    Matrix {
        data,
        n_rows,
        n_cols,
    }
}

fn slice_block(block: &PartitionBlock, mask: &[bool]) -> PartitionBlock {
    let take = |m: &Matrix| -> Matrix {
        let data: Vec<f32> = (0..m.n_rows)
            .filter(|&r| mask[r])
            .flat_map(|r| m.data[r * m.n_cols..(r + 1) * m.n_cols].iter().copied())
            .collect();
        Matrix {
            data,
            n_rows: mask.iter().filter(|&&b| b).count(),
            n_cols: m.n_cols,
        }
    };
    PartitionBlock {
        n_samples: mask.iter().filter(|&&b| b).count(),
        x: block.x.iter().map(take).collect(),
        feature_headers: block.feature_headers.clone(),
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
    }
}

fn split_partition_mask(
    df: &Frame,
    spec: &DatasetSpec,
    base_dir: &Path,
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
            let col: Vec<String> = df
                .str_column(column)
                .iter()
                .map(|s| s.to_lowercase())
                .collect();
            let lower_set = |vals: &[Value]| -> HashSet<String> {
                vals.iter()
                    .map(|v| nirs4all_io_core::pyfmt::py_str_scalar(v).to_lowercase())
                    .collect()
            };
            let train_vals = lower_set(&p.train_values);
            let test_vals = lower_set(&p.test_values);
            let predict_vals = lower_set(&p.predict_values);
            let known: HashSet<&String> = train_vals
                .iter()
                .chain(&test_vals)
                .chain(&predict_vals)
                .collect();
            let mut train_mask: Vec<bool> = col.iter().map(|v| train_vals.contains(v)).collect();
            let mut test_mask: Vec<bool> = col.iter().map(|v| test_vals.contains(v)).collect();
            let unknown: Vec<bool> = col.iter().map(|v| !known.contains(v)).collect();
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
            if !predict_vals.is_empty() {
                masks.insert(
                    "predict".into(),
                    col.iter().map(|v| predict_vals.contains(v)).collect(),
                );
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
                ("predict", &p.predict),
            ],
            n,
            "partitions",
        ),
        PartitionBy::IndexFile => {
            let load = |f: &Option<String>| -> Result<Option<Vec<i64>>, SpecError> {
                match f {
                    Some(f) => Ok(Some(read_index_file(&base_dir.join(f))?)),
                    None => Ok(None),
                }
            };
            let train = load(&p.train_file)?;
            let test = load(&p.test_file)?;
            let predict = load(&p.predict_file)?;
            let as_val = |o: &Option<Vec<i64>>| -> Value {
                o.as_ref()
                    .map(|v| Value::from(v.clone()))
                    .unwrap_or(Value::Null)
            };
            mask_from_index_lists(
                &[
                    ("train", &as_val(&train)),
                    ("test", &as_val(&test)),
                    ("predict", &as_val(&predict)),
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
    let mut seen: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
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
                nirs4all_io_core::pyfmt::py_int(v).ok_or_else(|| {
                    SpecError::new(format!(
                        "{source}: '{part}' contains a non-integer index {}",
                        nirs4all_io_core::pyfmt::py_repr(v)
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

fn read_index_file(path: &Path) -> Result<Vec<i64>, SpecError> {
    if !path.exists() {
        return Err(SpecError::new(format!(
            "partitions.index_file: file not found: {}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| SpecError::new(format!("cannot read index file {}: {e}", path.display())))?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(vec![]);
    }
    if text.starts_with('[') {
        let arr: Vec<i64> = serde_json::from_str(text)
            .map_err(|e| SpecError::new(format!("index_file JSON: {e}")))?;
        return Ok(arr);
    }
    text.replace(',', "\n")
        .lines()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<i64>()
                .map_err(|_| SpecError::new(format!("index_file: non-integer token '{t}'")))
        })
        .collect()
}

fn attach_folds(
    spec: &DatasetSpec,
    base_dir: &Path,
    assembled: &mut AssembledDataset,
    combined_df: Option<&Frame>,
) -> Result<(), SpecError> {
    let Some(folds) = &spec.folds else {
        return Ok(());
    };
    if !folds.inline.is_empty() {
        let int_list = |v: Option<&Value>| -> Vec<i64> {
            v.and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default()
        };
        assembled.folds = folds
            .inline
            .iter()
            .map(|f| {
                let m = f.as_object();
                let train = int_list(m.and_then(|m| m.get("train")));
                let val = int_list(m.and_then(|m| m.get("val").or_else(|| m.get("test"))));
                (train, val)
            })
            .collect();
    } else if let Some(column) = folds.column.as_deref().filter(|s| !s.is_empty()) {
        let combined = combined_df.ok_or_else(|| {
            SpecError::new(format!(
                "folds.column='{column}' requires a single combined frame (it does not apply with explicit per-source partitions)"
            ))
        })?;
        if !combined.has_column(column) {
            return Err(SpecError::new(format!(
                "folds.column='{column}' not found in the assembled data (columns: {:?})",
                combined.column_names()
            )));
        }
        let col = combined.column(column).unwrap();
        // distinct non-NA values, sorted by str(value)
        let mut distinct: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, cell) in col.values.iter().enumerate() {
            if matches!(cell, Cell::Na) || matches!(cell, Cell::Float(f) if f.is_nan()) {
                continue;
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
        let n = col.values.len();
        assembled.folds = distinct
            .into_iter()
            .map(|(_, val_rows)| {
                let vset: HashSet<usize> = val_rows.iter().copied().collect();
                let train: Vec<i64> = (0..n)
                    .filter(|i| !vset.contains(i))
                    .map(|i| i as i64)
                    .collect();
                let val: Vec<i64> = val_rows.iter().map(|&i| i as i64).collect();
                (train, val)
            })
            .collect();
    } else if let Some(file) = folds.file.as_deref().filter(|s| !s.is_empty()) {
        assembled.folds = parse_fold_file(&base_dir.join(file), folds.format.value())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize::frame::{Cell, Column};
    use serde_json::json;

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
        let (codes, mapping) = encode_categorical(&col, "auto");
        assert_eq!(codes, vec![0.0, 1.0, 0.0]);
        assert_eq!(mapping, Some(json!({"categories": ["a", "b"]})));
    }

    #[test]
    fn float_target_has_no_mapping() {
        let col = Column::from_cells("y", vec![Cell::Float(12.5), Cell::Float(8.3)]);
        let (vals, mapping) = encode_categorical(&col, "auto");
        assert_eq!(vals, vec![12.5, 8.3]);
        assert_eq!(mapping, None);
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
}
