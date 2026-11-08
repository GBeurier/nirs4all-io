// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Sample-identity inference + cross-file alignment audit (ports `infer/identity.py`).
//!
//! Detects a sample-id column, classifies a metadata source as per-sample (1:1)
//! vs shared/dimension (m:1), and audits coverage. Operates on the neutral
//! [`TableProfile`] (facade-populated) — no pandas, no file IO.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};

use super::plan::DatasetPlan;
use super::table::TableProfile;
use crate::pyfmt::{py_float_repr, py_repr, py_round};

static ID_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(
        r"^(sample[_ ]?id|sampleid|sample|id|name|code|ref(erence)?|key|index|spectrum[_ ]?id|scan[_ ]?id|.*_id|id_.*)$",
    )
    .case_insensitive(true)
    .build()
    .unwrap()
});

static WL_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r"^\d+(\.\d+)?(nm|cm-1)?$")
        .case_insensitive(true)
        .build()
        .unwrap()
});

// Replicate-suffix patterns (source string leaks verbatim into evidence).
static REPLICATE_PATTERNS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    let build = |p: &'static str| {
        (
            p,
            RegexBuilder::new(p).case_insensitive(true).build().unwrap(),
        )
    };
    vec![
        build(r"[_\-. ]rep(?:licate)?[_\-. ]?\d+$"),
        build(r"[_\-. ]r\d+$"),
        build(r"[_\-. ]scan[_\-. ]?\d+$"),
        build(r"[_\-. ]dup(?:licate)?\d*$"),
        build(r"\s*\(\d+\)$"),
        build(r"[_\-. ][a-z]$"),
    ]
});

#[derive(Debug, Clone)]
pub struct SampleIdGuess {
    pub column: String,
    pub score: f64,
    pub unique: bool,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RepetitionProfile {
    pub n_rows: usize,
    pub n_unique: usize,
    pub avg_reps: f64,
    pub is_repetition: bool,
}

#[derive(Debug, Clone)]
pub struct ReplicateGrouping {
    pub pattern: String,
    pub sample_ids: Vec<String>,
    pub n_files: usize,
    pub n_samples: usize,
    pub avg_reps: f64,
    pub evidence: Vec<String>,
}

/// Pick the most likely sample-id column, or `None` (requires score >= 0.5).
pub fn detect_sample_id(table: &TableProfile) -> Option<SampleIdGuess> {
    let n = table.n_rows;
    if n == 0 {
        return None;
    }
    let mut best: Option<SampleIdGuess> = None;
    for col in &table.columns {
        if WL_RE.is_match(&col.name) {
            continue;
        }
        let nunique = col.nunique_with_na;
        let unique = col.is_unique;
        let mut score = 0.0;
        let mut ev: Vec<String> = vec![];
        if ID_NAME_RE.is_match(&col.name) {
            score += 0.5;
            ev.push(format!("id-like name '{}'", col.name));
        }
        if unique {
            score += 0.35;
            ev.push("unique per row".into());
        } else if nunique as f64 >= 0.9 * n as f64 {
            score += 0.15;
            ev.push("near-unique values".into());
        }
        if col.is_float_dtype() {
            score -= 0.25;
        } else if !col.is_numeric_dtype() {
            score += 0.1;
            ev.push("string identifier".into());
        }
        if score > 0.0 && best.as_ref().is_none_or(|b| score > b.score) {
            best = Some(SampleIdGuess {
                column: col.name.clone(),
                score: score.min(1.0),
                unique,
                evidence: ev,
            });
        }
    }
    best.filter(|b| b.score >= 0.5)
}

/// Non-unique id → repeated measurements (avg >= 1.5) vs sporadic duplicates.
pub fn repetition_profile(n_rows: usize, n_unique: usize) -> RepetitionProfile {
    let u = if n_unique == 0 { 1 } else { n_unique };
    let avg = n_rows as f64 / u as f64;
    RepetitionProfile {
        n_rows,
        n_unique: u,
        avg_reps: py_round(avg, 2),
        is_repetition: u < n_rows && avg >= 1.5,
    }
}

fn unique_count(items: &[String]) -> usize {
    items.iter().collect::<HashSet<&String>>().len()
}

/// Detect replicate files of the same sample (mango_001_a/_b/_c → mango_001).
pub fn detect_replicate_grouping(stems: &[String]) -> Option<ReplicateGrouping> {
    let n = stems.len();
    if n < 2 {
        return None;
    }
    let mut best: Option<(f64, ReplicateGrouping)> = None;
    for (src, rx) in REPLICATE_PATTERNS.iter() {
        let stripped: Vec<String> = stems
            .iter()
            .map(|s| rx.replace_all(s, "").into_owned())
            .collect();
        let matched = stripped.iter().zip(stems).filter(|(st, s)| st != s).count();
        if (matched as f64) < n as f64 * 0.5 {
            continue;
        }
        let n_samples = unique_count(&stripped);
        if n_samples >= n {
            continue;
        }
        let avg = n as f64 / n_samples as f64;
        if avg < 1.5 {
            continue;
        }
        let score = matched as f64 / n as f64;
        if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
            let avg2 = py_round(avg, 2);
            let ev = vec![format!(
                "replicate suffix /{src}/ on {matched}/{n} files -> {n_samples} sample(s), avg {} reps",
                py_float_repr(avg2)
            )];
            best = Some((
                score,
                ReplicateGrouping {
                    pattern: (*src).to_string(),
                    sample_ids: stripped,
                    n_files: n,
                    n_samples,
                    avg_reps: avg2,
                    evidence: ev,
                },
            ));
        }
    }
    best.map(|(_, g)| g)
}

fn sorted_unique_diff(a: &HashSet<&str>, b: &HashSet<&str>) -> Vec<String> {
    let mut v: Vec<String> = a.difference(b).map(|s| s.to_string()).collect();
    v.sort();
    v
}

/// Does every sample id have a match in `other_ids`? Report missing/extra/dupes.
pub fn coverage_audit(x_ids: &[String], other_ids: &[String], other_name: &str) -> Value {
    let o_set: HashSet<&str> = other_ids.iter().map(String::as_str).collect();
    let x_set: HashSet<&str> = x_ids.iter().map(String::as_str).collect();
    let missing = sorted_unique_diff(&x_set, &o_set);
    let extra = sorted_unique_diff(&o_set, &x_set);
    // duplicates in x_ids (count > 1), sorted
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for v in x_ids {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let mut dup_x: Vec<String> = counts
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(k, _)| k.to_string())
        .collect();
    dup_x.sort();

    let take20 = |v: &[String]| -> Vec<Value> {
        v.iter().take(20).map(|s| Value::from(s.clone())).collect()
    };
    serde_json::json!({
        "against": other_name,
        "n_samples": x_set.len(),
        "n_missing": missing.len(),
        "missing": take20(&missing),
        "n_extra": extra.len(),
        "extra": take20(&extra),
        "duplicate_sample_ids": take20(&dup_x),
    })
}

/// Classify a metadata source vs the samples: `("1:1"|"m:1", key, evidence)`.
pub fn classify_metadata_relation(
    x_table: &TableProfile,
    meta_table: &TableProfile,
    x_id: &str,
) -> Option<(String, String, Vec<String>)> {
    if let Some(col) = meta_table.column(x_id) {
        if col.is_unique && meta_table.n_rows as f64 >= x_table.n_rows as f64 * 0.5 {
            return Some((
                "1:1".into(),
                x_id.to_string(),
                vec![format!("keyed by sample id '{x_id}' (one row per sample)")],
            ));
        }
    }
    for col in &meta_table.columns {
        if col.name == x_id || !x_table.has_column(&col.name) {
            continue;
        }
        if col.is_unique && meta_table.n_rows < x_table.n_rows {
            return Some((
                "m:1".into(),
                col.name.clone(),
                vec![format!(
                    "shared dimension table keyed by '{}' ({} rows broadcast to {} samples)",
                    col.name, meta_table.n_rows, x_table.n_rows
                )],
            ));
        }
    }
    None
}

/// Keys the sources by the detected sample id `id_col` of the feature source
/// `feature_id` (`x`), as the oracle `_infer_identity_and_alignment` does after
/// detection: features carrying the id are keyed by it; targets carrying it
/// get a keyed join and a bounded coverage preview. Automatic joins are 1:1;
/// inference proposes m:1 only for a visible systematic repetition profile.
/// Metadata tables get the 1:1 or
/// m:1 (lookup) relation of [`classify_metadata_relation`]. Without these keys
/// a convention join stays positional and rows in a different order are
/// paired wrongly. `profile` reads the table of another source.
pub fn key_sources_by_id(
    spec_dict: &mut Value,
    plan: &mut DatasetPlan,
    id_col: &str,
    feature_id: &str,
    x: &TableProfile,
    mut profile: impl FnMut(&Value) -> Option<TableProfile>,
    inference: bool,
) {
    let x_ids: Vec<String> = x
        .column(id_col)
        .map(|c| c.str_values.clone())
        .unwrap_or_default();
    let repeated = inference
        && x.column(id_col).is_some_and(|column| {
            repetition_profile(x.n_rows, column.nunique_with_na).is_repetition
        });
    let target_cardinality = if repeated { "m:1" } else { "1:1" };
    let Some(sources) = spec_dict.get_mut("sources").and_then(Value::as_array_mut) else {
        return;
    };
    for source in sources.iter_mut() {
        let role = source
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let lookup = source.get("kind").and_then(Value::as_str) == Some("lookup");
        let id = source
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if matches!(role.as_str(), "features" | "mixed") && !lookup {
            let keyed = id == feature_id || profile(source).is_some_and(|t| t.has_column(id_col));
            if keyed {
                if let Some(obj) = source.as_object_mut() {
                    obj.entry("key").or_insert_with(|| Value::from(id_col));
                }
            }
            continue;
        }
        let Some(table) = profile(source) else {
            continue;
        };
        if role == "targets" {
            let Some(column) = table.column(id_col) else {
                continue;
            };
            set_key(source, id_col, target_cardinality, None);
            let mut audit = coverage_audit(&x_ids, &column.str_values, &id);
            audit["role"] = Value::from("targets");
            audit["scope"] = Value::from("bounded_discovery_prefix");
            audit["complete"] = Value::from(false);
            if let Some(missing) = missing_warning(
                &audit,
                "sample(s) have NO target in the sampled prefix",
                &id,
            ) {
                plan.warnings.push(missing);
            }
            plan.alignment.push(audit);
        } else if role == "metadata" || lookup {
            if !inference {
                set_key(source, id_col, "1:1", None);
                continue;
            }
            let Some((mut card, key, evidence)) = classify_metadata_relation(x, &table, id_col)
            else {
                continue;
            };
            if repeated && key == id_col && card == "1:1" {
                card = "m:1".into();
            }
            set_key(source, &key, &card, Some(card == "m:1"));
            if let (Some(xc), Some(mc)) = (x.column(&key), table.column(&key)) {
                let mut audit = coverage_audit(&xc.str_values, &mc.str_values, &id);
                audit["role"] = Value::from("metadata");
                audit["relation"] = Value::from(card.clone());
                audit["evidence"] = serde_json::json!(evidence);
                let label = format!("{key}(s) have NO metadata");
                if let Some(missing) = missing_warning(&audit, &label, &id) {
                    plan.warnings.push(missing);
                }
                plan.alignment.push(audit);
            }
        }
    }
}

/// `src["key"] = key` and a declared join keyed on it (`on`/`how`, as the
/// oracle's `_key_join`); a missing join stays absent. `lookup` marks an m:1
/// dimension table.
fn set_key(source: &mut Value, key: &str, how: &str, lookup: Option<bool>) {
    let Some(obj) = source.as_object_mut() else {
        return;
    };
    obj.insert("key".into(), Value::from(key));
    if lookup == Some(true) {
        obj.insert("kind".into(), Value::from("lookup"));
    }
    if let Some(join) = obj.get_mut("join").and_then(Value::as_object_mut) {
        join.insert("on".into(), Value::from(key));
        join.insert("how".into(), Value::from(how));
    }
}

/// The oracle warning `"<n> <label> in '<id>' (e.g. <first three>)"`.
fn missing_warning(audit: &Value, label: &str, id: &str) -> Option<String> {
    let n = audit["n_missing"].as_u64().unwrap_or(0);
    (n > 0).then(|| {
        let first: Vec<Value> = audit["missing"]
            .as_array()
            .map(|m| m.iter().take(3).cloned().collect())
            .unwrap_or_default();
        format!(
            "{n} {label} in '{id}' (e.g. {})",
            py_repr(&Value::Array(first))
        )
    })
}

/// Propose only partition-local identity, using every member's bounded prefix.
/// No equality, full coverage or duplicate validation belongs to discovery.
pub fn configure_generated_keys(
    spec: &mut Value,
    plan: &mut DatasetPlan,
    mut inputs: impl FnMut(&Value) -> Option<Vec<Vec<u8>>>,
    inference: bool,
) {
    use crate::infer::describe::proposed_decimal_evidence;
    use crate::infer::plan::Decision;
    use crate::materialize::join::concat_samples;
    use crate::materialize::loaders::discovery_table;
    use crate::spec::dataset_spec::LoadingParams;

    fn recompute_decimal(
        source: &mut Value,
        plan: &mut DatasetPlan,
        bytes: &[u8],
        key: &str,
        inference: bool,
    ) {
        let Ok(params) = LoadingParams::from_value(source.get("params")) else {
            return;
        };
        let decimal = proposed_decimal_evidence(bytes, &params, Some(key));
        if !source["params"].is_object() {
            source["params"] = json!({});
        }
        source["params"]
            .as_object_mut()
            .unwrap()
            .remove("decimal_separator");
        if let Some((decimal, _)) = &decimal {
            source["params"]["decimal_separator"] = json!(decimal);
        }
        if inference {
            if let Some(name) = source["input"]
                .as_str()
                .and_then(|name| name.rsplit('/').next())
            {
                if let Some(entry) = plan.params[name].as_object_mut() {
                    entry.remove("decimal_separator");
                    if source["params"]["decimal_separator"] == "," {
                        entry.insert(
                            "decimal_separator".into(),
                            json!({"value":",","score":py_round(decimal.as_ref().unwrap().1,3)}),
                        );
                    }
                }
            }
        }
    }

    let Some(mut sources) = spec["sources"].as_array().cloned() else {
        return;
    };
    if sources.iter().any(|s| s["kind"] == "lookup") {
        return;
    }
    let mut partitions = Vec::new();
    for source in &sources {
        if !partitions.contains(&source["partition"]) {
            partitions.push(source["partition"].clone());
        }
    }
    let mut profiles: HashMap<usize, TableProfile> = HashMap::new();
    let mut samples: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
    let mut groups = Vec::new();
    let mut filtered_labels = Vec::new();
    for partition in partitions {
        let indices: Vec<usize> = (0..sources.len())
            .filter(|&i| sources[i]["partition"] == partition)
            .collect();
        let Some(anchor) = indices
            .iter()
            .copied()
            .find(|&i| matches!(sources[i]["role"].as_str(), Some("features" | "mixed")))
        else {
            continue;
        };
        let complete = indices.iter().all(|&index| {
            let Some(bytes) = inputs(&sources[index]) else {
                return false;
            };
            let Ok(params) = LoadingParams::from_value(sources[index].get("params")) else {
                return false;
            };
            let Some(frames) = bytes
                .iter()
                .map(|bytes| discovery_table(bytes, &params))
                .collect::<Option<Vec<_>>>()
            else {
                return false;
            };
            if frames.is_empty()
                || frames
                    .iter()
                    .any(|frame| frame.column_names() != frames[0].column_names())
            {
                return false;
            }
            let names: Vec<String> = (0..frames.len()).map(|i| i.to_string()).collect();
            profiles.insert(index, concat_samples(&frames, &names).0.to_table_profile());
            samples.insert(index, bytes);
            true
        });
        if !complete {
            continue;
        }
        let mut guesses = HashMap::new();
        for &index in &indices {
            let table = &profiles[&index];
            let mut guess = (table.columns.len() > 1)
                .then(|| detect_sample_id(table))
                .flatten();
            if index != anchor
                && guess.as_ref().is_some_and(|guess| {
                    !guess.unique
                        || table
                            .column(&guess.column)
                            .is_some_and(|column| column.is_float_dtype())
                })
            {
                guess = None;
            }
            if sources[index]["role"] == "metadata" {
                if let Some(label) = guess.clone().filter(|label| {
                    !indices.iter().any(|other| {
                        sources[*other]["role"] != "metadata"
                            && profiles[other].has_column(&label.column)
                    })
                }) {
                    filtered_labels.push((index, label.column, partition.clone()));
                    guess = None;
                }
            }
            guesses.insert(index, guess);
        }
        let selected = guesses[&anchor]
            .clone()
            .or_else(|| indices.iter().find_map(|i| guesses[i].clone()));
        groups.push((indices, anchor, selected));
    }
    for (index, column, partition) in filtered_labels {
        if inference {
            let partition = partition.as_str().unwrap_or("null");
            let id = sources[index]["id"].as_str().unwrap_or("");
            plan.warnings.push(format!("partition {partition}: metadata label '{column}' in source '{id}' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity"));
        }
    }
    let Some(selected) = groups.iter().find_map(|(_, _, guess)| guess.clone()) else {
        // Neutral profiles affect only decimal voting after all groups failed
        // identity selection. They never choose or promote a generated key.
        for (indices, _, _) in &groups {
            for &index in indices {
                if sources[index]["role"] != "metadata"
                    || sources[index]["params"]["decimal_separator"] != ","
                    || samples[&index].len() != 1
                {
                    continue;
                }
                let Ok(mut params) = LoadingParams::from_value(sources[index].get("params")) else {
                    continue;
                };
                params.decimal_separator = None;
                let Some(neutral) = discovery_table(&samples[&index][0], &params) else {
                    continue;
                };
                let neutral = neutral.to_table_profile();
                if neutral.columns.len() <= 1 {
                    continue;
                }
                let Some(label) = detect_sample_id(&neutral).filter(|candidate| {
                    candidate.unique
                        && neutral
                            .column(&candidate.column)
                            .is_some_and(|column| !column.is_float_dtype())
                }) else {
                    continue;
                };
                if indices.iter().any(|other| {
                    sources[*other]["role"] != "metadata"
                        && profiles[other].has_column(&label.column)
                }) {
                    continue;
                }
                recompute_decimal(
                    &mut sources[index],
                    plan,
                    &samples[&index][0],
                    &label.column,
                    inference,
                );
                spec["sources"][index] = sources[index].clone();
                if inference {
                    let partition = sources[index]["partition"].as_str().unwrap_or("null");
                    let id = sources[index]["id"].as_str().unwrap_or("");
                    let column = &label.column;
                    let warning = format!("partition {partition}: metadata label '{column}' in source '{id}' is not shared with features or targets; rows stay positional and are not verified by id; declare a key or join to align by identity");
                    if !plan.warnings.contains(&warning) {
                        plan.warnings.push(warning);
                    }
                    if sources[index]["params"]["decimal_separator"] == ","
                        && profiles[&index]
                            .column(column)
                            .is_some_and(|column| column.is_numeric_dtype())
                    {
                        plan.warnings.push(format!("partition {partition}: metadata label '{column}' in source '{id}' also reads as numbers under decimal ','; source-wide decimals can rewrite or merge its values; declare decimal_separator '.' to keep comma-form values as text in this source"));
                    }
                }
            }
        }
        return;
    };
    let key = selected.column;
    spec["sample_index"] = json!({"by":"id","key":key});
    if inference && plan.identity.is_none() {
        plan.identity = Some(Decision {
            value: json!(key),
            score: selected.score,
            evidence: selected.evidence,
            ambiguous: !selected.unique,
            alternatives: vec![],
        });
    }
    for (indices, anchor, guess) in groups {
        let x = &profiles[&anchor];
        let anchor_id = sources[anchor]["id"].as_str().unwrap_or("");
        if inference {
            if let Some(column) = x.column(&key) {
                let profile = repetition_profile(x.n_rows, column.nunique_with_na);
                plan.params["identity_repetition"][anchor_id] = json!({"n_rows":profile.n_rows,"n_unique":profile.n_unique,"avg_reps":profile.avg_reps,"is_repetition":profile.is_repetition});
            }
            if guess.as_ref().is_none_or(|guess| guess.column != key)
                || indices.iter().any(|i| !profiles[i].has_column(&key))
            {
                plan.warnings.push(format!("partition {}: incomplete identity '{key}'; declare the join/identity before loading", sources[anchor]["partition"]));
            }
        }
        let mut group =
            json!({"sources":indices.iter().map(|&i| sources[i].clone()).collect::<Vec<_>>()});
        for (position, &index) in indices.iter().enumerate() {
            let source = &mut group["sources"][position];
            source["key"] = json!(key);
            if index != anchor {
                if !source["join"].is_object() {
                    source["join"] = json!({"to":anchor_id});
                }
                source["join"]["on"] = json!(key);
                source["join"]["how"] = json!("1:1");
                source["join"]["coverage"] = json!("complete");
            }
        }
        key_sources_by_id(
            &mut group,
            plan,
            &key,
            anchor_id,
            x,
            |source| {
                let index = indices
                    .iter()
                    .find(|&&i| sources[i]["id"] == source["id"])?;
                Some(profiles[index].clone())
            },
            inference,
        );
        for (position, &index) in indices.iter().enumerate() {
            let source = &mut group["sources"][position];
            let bytes = &samples[&index];
            if bytes.len() == 1 {
                recompute_decimal(source, plan, &bytes[0], &key, inference);
            }
            spec["sources"][index] = source.clone();
        }
    }
}

/// Check generated proposals at load, leaving explicit specs authoritative.
pub fn refuse_generated_identity(
    spec: &crate::spec::DatasetSpec,
    mut inputs: impl FnMut(&Value) -> Option<Vec<Vec<u8>>>,
) -> Result<(), crate::spec::SpecError> {
    use crate::infer::describe::csv_rows;
    use crate::materialize::loaders::{
        dialect_text, discovery_prefix, discovery_table, is_comma_pair,
    };
    use crate::spec::dataset_spec::LoadingParams;
    use crate::spec::SpecError;
    let value = spec.to_value();
    if value["sample_index"]["by"] != "id" {
        return Ok(());
    }
    let Some(key) = value["sample_index"]["key"].as_str() else {
        return Ok(());
    };
    for source in value["sources"].as_array().into_iter().flatten() {
        if source["kind"] == "lookup" {
            continue;
        }
        let Some(inputs) = inputs(source) else {
            continue;
        };
        let params = LoadingParams::from_value(source.get("params"))?;
        for bytes in inputs {
            let Some(table) = discovery_table(&bytes, &params) else {
                continue;
            };
            if !table.has_column(key) {
                return Err(SpecError::new(format!("source '{}': file {} missing identity key '{key}'; declare compatible keys/partitions",source["id"],source["input"])));
            }
            if params.decimal_separator.as_deref() == Some(",") {
                let Some(prefix) = discovery_prefix(&bytes, &params) else {
                    continue;
                };
                let delimiter = params
                    .delimiter
                    .as_deref()
                    .unwrap_or(";")
                    .chars()
                    .next()
                    .unwrap_or(';');
                let Ok(rows) = csv_rows(&dialect_text(&prefix), delimiter) else {
                    continue;
                };
                if let Some(index) = rows
                    .first()
                    .and_then(|row| row.iter().position(|name| name == key))
                {
                    if rows
                        .iter()
                        .skip(1)
                        .any(|row| row.get(index).is_some_and(|cell| is_comma_pair(cell)))
                    {
                        return Err(SpecError::new(format!("source '{}': file {} key '{key}' contains comma pairs under params.decimal_separator=','; declare an explicit identity/decimal policy",source["id"],source["input"])));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::table::{ColDtype, ColumnProfile, NumericKind, TableProfile};
    use super::*;

    fn col(
        name: &str,
        dtype: ColDtype,
        nk: NumericKind,
        nunique: usize,
        n: usize,
    ) -> ColumnProfile {
        ColumnProfile {
            name: name.into(),
            dtype,
            numeric_kind: nk,
            nunique_with_na: nunique,
            is_unique: nunique == n,
            str_values: vec![],
            numeric_values: vec![],
        }
    }

    #[test]
    fn detect_id_by_name_and_uniqueness() {
        let t = TableProfile {
            n_rows: 3,
            columns: vec![
                col("sample_id", ColDtype::String, NumericKind::NonNumeric, 3, 3),
                col("1000", ColDtype::Numeric, NumericKind::Float, 3, 3),
            ],
        };
        let g = detect_sample_id(&t).unwrap();
        assert_eq!(g.column, "sample_id");
        // 0.5 (id name) + 0.35 (unique) + 0.1 (string) = 0.95
        assert_eq!(g.score, 0.95);
    }

    #[test]
    fn float_target_is_not_an_id() {
        let t = TableProfile {
            n_rows: 6,
            columns: vec![col("protein", ColDtype::Numeric, NumericKind::Float, 6, 6)],
        };
        // unique 0.35 - float 0.25 = 0.1 < 0.5 -> None
        assert!(detect_sample_id(&t).is_none());
    }

    #[test]
    fn replicate_grouping_single_letter() {
        let stems: Vec<String> = ["m_001_a", "m_001_b", "m_002_a", "m_002_b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let g = detect_replicate_grouping(&stems).unwrap();
        assert_eq!(g.n_samples, 2);
        assert_eq!(g.avg_reps, 2.0);
        assert_eq!(g.sample_ids, vec!["m_001", "m_001", "m_002", "m_002"]);
    }

    #[test]
    fn coverage_audit_missing() {
        let x: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let o: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let audit = coverage_audit(&x, &o, "targets");
        assert_eq!(audit["n_missing"], serde_json::json!(1));
        assert_eq!(audit["missing"], serde_json::json!(["c"]));
        assert_eq!(audit["against"], serde_json::json!("targets"));
    }
}
