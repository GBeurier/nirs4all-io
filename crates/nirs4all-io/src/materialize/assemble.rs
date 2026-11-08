// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Assemble a DatasetSpec into a target-agnostic dataset — the facade's
//! filesystem entry (ports the IO half of `materialize/assemble.py`).
//!
//! The whole assembly core — frame, join, role-split, partition masks, fold
//! attach, `AssembledDataset` — moved into `nirs4all-io-core::materialize` so the
//! WASM binding can reach it (D-R7). This module keeps only the filesystem side:
//! resolve `source.input` (glob/dir/list), read each file's bytes (the loaders
//! handle gzip/zip), read `partitions.index_file`s, and parse `folds.file`. It
//! then gathers everything into named in-memory payloads and delegates to
//! [`assemble_in_memory_with_tabular_limits`], so native and browser paths share
//! one assembly core and produce byte-identical `AssembledDataset`s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nirs4all_io_core::materialize::loaders::effective_params;
use nirs4all_io_core::materialize::{
    assemble_in_memory_with_tabular_limits, InMemorySource, SourcePayload,
};
use nirs4all_io_core::spec::dataset_spec::DatasetSpec;
use nirs4all_io_core::spec::dataset_spec::LoadingParams;
use nirs4all_io_core::spec::enums::PartitionBy;
use nirs4all_io_core::spec::SpecError;
use serde_json::Value;

use super::folds::{parse_fold_file_with_budget, Fold};
use super::limits::{LoadLimits, ReadBudget};
use super::loaders::{
    is_spreadsheet_path, read_parquet_frame_with_budget, read_workbook_with_budget,
};

pub use nirs4all_io_core::materialize::{
    AssembledDataset, FoldProvenance, IdentityProvenance, PartitionBlock, ASSEMBLED_DATASET_VERSION,
};

fn is_glob(text: &str) -> bool {
    text.chars().any(|c| matches!(c, '*' | '?' | '['))
}

/// Resolve one `source.input` value against `base_dir` into `(match_name, path)`
/// pairs. `match_name` is what the in-memory resolver re-matches against: the
/// original entry for plain inputs, and — for glob-expanded entries — the path
/// *relative to `base_dir`* (with separators normalized to `/`) so the in-memory
/// glob re-match stays path-scoped and a `spectra/*.csv` source does not also
/// match a sibling `meta.csv` (Codex #2).
fn pattern_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern)
        .map(|paths| paths.filter_map(Result::ok).collect())
        .unwrap_or_default()
}

// Keep scoped glob construction testable independently of direct-path fallback.
fn scoped_pattern(text: &str, base_dir: &Path) -> String {
    if Path::new(text).is_absolute() {
        text.to_string()
    } else {
        let escaped = glob::Pattern::escape(&base_dir.to_string_lossy());
        Path::new(&escaped)
            .join(text)
            .to_string_lossy()
            .into_owned()
    }
}

fn resolve_named(input: &Value, base_dir: &Path) -> Result<Vec<(String, PathBuf)>, SpecError> {
    let items: Vec<&Value> = match input {
        Value::Array(values) => values.iter().collect(),
        other => vec![other],
    };
    let mut out = Vec::new();
    for item in items {
        let Some(text) = item.as_str() else { continue };
        let scoped = base_dir.join(text);
        if scoped.is_file() {
            out.push((text.to_string(), scoped));
            continue;
        }
        if is_glob(text) {
            let pattern = scoped_pattern(text, base_dir);
            let mut matched = pattern_paths(&pattern);
            matched.sort();
            if !matched.is_empty() {
                for path in matched {
                    let name = if Path::new(text).is_absolute() {
                        path.to_string_lossy().replace('\\', "/")
                    } else {
                        let logical = crate::resolve::logical_path(&path);
                        let base = crate::resolve::logical_path(base_dir);
                        logical
                            .strip_prefix(&base)
                            .map_err(|_| {
                                SpecError::new(format!(
                                    "glob path {logical:?} is outside base {base:?}"
                                ))
                            })?
                            .to_string_lossy()
                            .replace('\\', "/")
                    };
                    out.push((name, path));
                }
                continue;
            }
        }
        if scoped.exists() {
            out.push((text.to_string(), scoped));
            continue;
        }
        let direct = PathBuf::from(text);
        if direct.is_file() {
            out.push((text.to_string(), direct));
        } else if is_glob(text) {
            let mut matched = pattern_paths(text);
            matched.sort();
            out.extend(
                matched
                    .into_iter()
                    .map(|path| (path.to_string_lossy().replace('\\', "/"), path)),
            );
        } else {
            out.push((text.to_string(), direct));
        }
    }
    Ok(out)
}

fn path_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn requested_parquet_columns(params: &LoadingParams) -> Result<Option<Vec<String>>, SpecError> {
    match params.format.values.get("columns") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) => Ok(Some(vec![name.clone()])),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_string).ok_or_else(|| {
                    SpecError::new(
                        "parquet loading.format.columns must be a string or a list of strings",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(SpecError::new(
            "parquet loading.format.columns must be a string or a list of strings",
        )),
    }
}

fn update_projection(
    projections: &mut HashMap<PathBuf, Option<Vec<String>>>,
    path: &Path,
    params: &LoadingParams,
) -> Result<(), SpecError> {
    if !is_parquet_path(path) {
        return Ok(());
    }
    let key = path_key(path);
    let Some(requested) = requested_parquet_columns(params)? else {
        projections.insert(key, None);
        return Ok(());
    };
    let entry = projections.entry(key).or_insert_with(|| Some(Vec::new()));
    if let Some(union) = entry {
        for column in requested {
            if !union.contains(&column) {
                union.push(column);
            }
        }
    }
    Ok(())
}

fn gather_parquet_projections(
    spec: &DatasetSpec,
    base_dir: &Path,
) -> Result<HashMap<PathBuf, Option<Vec<String>>>, SpecError> {
    let mut projections = HashMap::new();
    for source in &spec.sources {
        let source_params = effective_params(&spec.params, &source.params);
        for (_, path) in resolve_named(&source.input, base_dir)? {
            update_projection(&mut projections, &path, &source_params)?;
        }
        for variation in &source.variations {
            let mut params = source_params.clone();
            if !variation.params.is_empty_value() {
                params = effective_params(&params, &variation.params);
            }
            for (_, path) in resolve_named(&variation.input, base_dir)? {
                update_projection(&mut projections, &path, &params)?;
            }
        }
    }
    Ok(projections)
}

/// Gather every input referenced by `spec` (sources + variations) into named
/// in-memory payloads, reading each unique file once.
fn gather_sources(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
) -> Result<Vec<InMemorySource>, SpecError> {
    let parquet_projections = gather_parquet_projections(spec, base_dir)?;
    let mut seen: HashMap<String, ()> = HashMap::new();
    let mut out: Vec<InMemorySource> = Vec::new();
    let mut add_input = |input: &Value| -> Result<(), SpecError> {
        for (name, path) in resolve_named(input, base_dir)? {
            if seen.insert(name.clone(), ()).is_some() {
                continue;
            }
            let parquet_columns = parquet_projections
                .get(&path_key(&path))
                .and_then(|columns| columns.as_deref());
            let payload = source_payload(&path, parquet_columns, budget)?;
            out.push(InMemorySource { name, payload });
        }
        Ok(())
    };
    for source in &spec.sources {
        add_input(&source.input)?;
        for variation in &source.variations {
            add_input(&variation.input)?;
        }
    }
    Ok(out)
}

fn is_parquet_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("parquet" | "pq")
    )
}

fn source_payload(
    path: &Path,
    parquet_columns: Option<&[String]>,
    budget: &mut ReadBudget,
) -> Result<SourcePayload, SpecError> {
    if is_spreadsheet_path(path) {
        return Ok(SourcePayload::Sheets(read_workbook_with_budget(
            path, budget,
        )?));
    }
    if is_parquet_path(path) {
        return Ok(SourcePayload::Frame(read_parquet_frame_with_budget(
            path,
            parquet_columns,
            budget,
        )?));
    }
    Ok(SourcePayload::Bytes(budget.read(path)?))
}

/// Read `partitions.index_file` lists referenced by `spec` into `index_lists`.
fn gather_index_lists(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
) -> Result<HashMap<String, Vec<i64>>, SpecError> {
    let mut out = HashMap::new();
    let Some(p) = &spec.partitions else {
        return Ok(out);
    };
    if p.by != Some(PartitionBy::IndexFile) {
        return Ok(out);
    }
    for (part, file) in [
        ("train", &p.train_file),
        ("test", &p.test_file),
        ("val", &p.val_file),
        ("predict", &p.predict_file),
    ] {
        if let Some(file) = file {
            out.insert(
                part.to_string(),
                read_index_file(&base_dir.join(file), budget)?,
            );
        }
    }
    Ok(out)
}

/// Parse `folds.file` (when set) into folds; `None` otherwise.
fn gather_folds(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
) -> Result<Option<Vec<Fold>>, SpecError> {
    let Some(folds) = &spec.folds else {
        return Ok(None);
    };
    // inline / column folds are handled by the core; only the file path needs IO.
    if !folds.inline.is_empty() || folds.column.as_deref().is_some_and(|s| !s.is_empty()) {
        return Ok(None);
    }
    if let Some(file) = folds.file.as_deref().filter(|s| !s.is_empty()) {
        return Ok(Some(parse_fold_file_with_budget(
            &base_dir.join(file),
            folds.format.value(),
            budget,
        )?));
    }
    Ok(None)
}

/// Load, role-split, join and partition `spec` into an [`AssembledDataset`].
///
/// The facade reads every input from `base_dir`, then delegates to the shared
/// fs-free core [`assemble_in_memory_with_tabular_limits`].
pub fn assemble(spec: &DatasetSpec, base_dir: &Path) -> Result<AssembledDataset, SpecError> {
    assemble_with_limits(spec, base_dir, LoadLimits::default())
}

/// Assemble with explicit host-selected read/decompression/shape budgets.
pub fn assemble_with_storage(
    spec: &DatasetSpec,
    base_dir: &Path,
    options: &nirs4all_io_core::materialize::NumericStorageOptions,
) -> Result<AssembledDataset, SpecError> {
    assemble_with_limits_and_storage(spec, base_dir, LoadLimits::default(), options)
}

pub fn assemble_with_limits_and_storage(
    spec: &DatasetSpec,
    base_dir: &Path,
    limits: LoadLimits,
    options: &nirs4all_io_core::materialize::NumericStorageOptions,
) -> Result<AssembledDataset, SpecError> {
    assemble_with_budget_and_storage(spec, base_dir, &mut ReadBudget::new(limits)?, options)
}

pub(crate) fn assemble_with_budget_and_storage(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
    options: &nirs4all_io_core::materialize::NumericStorageOptions,
) -> Result<AssembledDataset, SpecError> {
    let sources = gather_sources(spec, base_dir, budget)?;
    let indices = gather_index_lists(spec, base_dir, budget)?;
    let folds = gather_folds(spec, base_dir, budget)?;
    nirs4all_io_core::materialize::assemble_in_memory_with_storage_and_tabular_limits(
        spec,
        &sources,
        &indices,
        folds.as_deref(),
        Some(budget.limits.tabular()),
        options,
    )
}

pub(crate) fn package_with_budget_and_storage(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
    options: &nirs4all_io_core::materialize::NumericStorageOptions,
) -> Result<nirs4all_io_core::materialize::DatasetPackage, SpecError> {
    let sources = gather_sources(spec, base_dir, budget)?;
    let indices = gather_index_lists(spec, base_dir, budget)?;
    let folds = gather_folds(spec, base_dir, budget)?;
    nirs4all_io_core::materialize::package_in_memory_with_storage_and_tabular_limits(
        spec,
        &sources,
        &indices,
        folds.as_deref(),
        Some(budget.limits.tabular()),
        options,
    )
}

pub fn package_with_limits_and_storage(
    spec: &DatasetSpec,
    base_dir: &Path,
    limits: LoadLimits,
    options: &nirs4all_io_core::materialize::NumericStorageOptions,
) -> Result<nirs4all_io_core::materialize::DatasetPackage, SpecError> {
    package_with_budget_and_storage(spec, base_dir, &mut ReadBudget::new(limits)?, options)
}

pub fn assemble_with_limits(
    spec: &DatasetSpec,
    base_dir: &Path,
    limits: LoadLimits,
) -> Result<AssembledDataset, SpecError> {
    assemble_with_budget(spec, base_dir, &mut ReadBudget::new(limits)?)
}

pub(crate) fn assemble_with_budget(
    spec: &DatasetSpec,
    base_dir: &Path,
    budget: &mut ReadBudget,
) -> Result<AssembledDataset, SpecError> {
    let sources = gather_sources(spec, base_dir, budget)?;
    let index_lists = gather_index_lists(spec, base_dir, budget)?;
    let fold_inline = gather_folds(spec, base_dir, budget)?;
    assemble_in_memory_with_tabular_limits(
        spec,
        &sources,
        &index_lists,
        fold_inline.as_deref(),
        Some(budget.limits.tabular()),
    )
}

fn read_index_file(path: &Path, budget: &mut ReadBudget) -> Result<Vec<i64>, SpecError> {
    if !path.exists() {
        return Err(SpecError::new(format!(
            "partitions.index_file: file not found: {}",
            path.display()
        )));
    }
    let text = String::from_utf8(budget.read(path)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use serde_json::json;
    use std::sync::Arc;

    fn write_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn shared_parquet_path_applies_format_columns_per_source() {
        let tmp = tempfile::tempdir().unwrap();
        write_parquet(&tmp.path().join("data.parquet"));
        let spec = DatasetSpec::from_value(&json!({
            "sources": [
                {
                    "id": "left",
                    "role": "features",
                    "input": "data.parquet",
                    "params": {"format": {"columns": ["a"]}}
                },
                {
                    "id": "right",
                    "role": "features",
                    "input": "data.parquet",
                    "params": {"format": {"columns": ["b"]}}
                }
            ]
        }))
        .unwrap();

        let assembled = assemble(&spec, tmp.path()).unwrap();
        let block = assembled.blocks.get("train").unwrap();

        assert_eq!(block.feature_headers, vec![vec!["a"], vec!["b"]]);
        assert_eq!(block.x[0].data, vec![1.0, 2.0]);
        assert_eq!(block.x[1].data, vec![10.0, 20.0]);
    }
}

#[cfg(test)]
mod path_reference_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_pattern_directory_keeps_its_base_path() {
        let tmp = tempfile::tempdir().unwrap();
        let decoy = tempfile::Builder::new()
            .prefix("n4io-dir-decoy-")
            .tempfile_in(".")
            .unwrap();
        let name = decoy
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        std::fs::create_dir(tmp.path().join(&name)).unwrap();
        let expected = vec![(name.clone(), tmp.path().join(&name))];
        assert_eq!(resolve_named(&json!(name), tmp.path()).unwrap(), expected);
        drop(decoy);
        assert_eq!(resolve_named(&json!(name), tmp.path()).unwrap(), expected);
    }

    #[test]
    fn empty_base_scoped_pattern_remains_relative_before_fallback() {
        assert_eq!(scoped_pattern("*.csv", Path::new("")), "*.csv");
        assert_eq!(
            Path::new(&scoped_pattern("spectra/*.csv", Path::new(""))),
            Path::new("spectra/*.csv")
        );
        assert_eq!(
            Path::new(&scoped_pattern("spectra/*.csv", Path::new("cfg[1]"))),
            Path::new("cfg[[]1[]]/spectra/*.csv")
        );
    }

    #[test]
    fn absolute_pattern_is_independent_of_empty_base() {
        let tmp = tempfile::Builder::new()
            .prefix("n4io-relative-")
            .tempdir_in(".")
            .unwrap();
        std::fs::write(tmp.path().join("part.csv"), "x\n0.1\n").unwrap();
        let pattern = format!("{}/*.csv", tmp.path().display());
        let found = resolve_named(&json!(pattern), Path::new("")).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, tmp.path().join("part.csv"));
    }
    #[test]
    fn interior_curdir_scoped_pattern_preserves_names() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("spectra")).unwrap();
        std::fs::write(tmp.path().join("spectra/a.csv"), "x\n0.1\n").unwrap();
        let base = tmp.path().join(".");
        let names = resolve_named(&json!("spectra/*.csv"), &base).unwrap();
        assert_eq!(names[0].0, "spectra/a.csv");
        let spec = DatasetSpec::from_value(
            &json!({"sources":[{"id":"x","role":"features","input":"spectra/*.csv"}]}),
        )
        .unwrap();
        assert_eq!(
            assemble(&spec, &base).unwrap().blocks["train"].x[0].data[0],
            0.1
        );
    }
}
