// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Exact base-cohort feature and target content, independent of DAG/DMD identities.
use super::frame::Cell;
use super::numeric::{bytes, hex, option_string, string, strings, u64_bytes};
use super::package::{DatasetPackage, FeatureMatrix, PayloadBlock, TargetTable};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const EXECUTED_PROJECTION_PROFILE: &str = "nirs4all.io.executed-projection.v1";
pub const MASK_POLICY: &str = "explicit-mask-positive-zero.v1";

pub fn feature_bytes(feature: &FeatureMatrix) -> Vec<u8> {
    let mut out = Vec::new();
    string(&mut out, &feature.source_id);
    string(&mut out, &feature.representation_id);
    out.extend_from_slice(&feature.matrix.precision.bytes());
    out.extend_from_slice(&feature.matrix.numeric_bytes());
    out.push(u8::from(feature.feature_names.is_some()));
    if let Some(names) = &feature.feature_names {
        strings(&mut out, names);
    }
    u64_bytes(&mut out, feature.axes.len());
    for axis in &feature.axes {
        string(
            &mut out,
            axis["name"].as_str().expect("validated axis name"),
        );
        option_string(&mut out, axis["unit"].as_str());
        let coordinates = axis["coordinates"].as_array();
        out.push(u8::from(coordinates.is_some()));
        if let Some(coordinates) = coordinates {
            if coordinates.iter().all(Value::is_number) {
                out.push(1);
                u64_bytes(&mut out, coordinates.len());
                let values = coordinates
                    .iter()
                    .flat_map(|v| {
                        super::numeric::exact_json_number_f64(v)
                            .expect("validated exact numeric coordinate")
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                bytes(&mut out, &values);
            } else {
                out.push(2);
                strings(
                    &mut out,
                    &coordinates
                        .iter()
                        .map(|v| v.as_str().expect("string coordinate").to_owned())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    out
}

pub fn string_precision(table: &TargetTable, column: usize) -> Value {
    json!({"dtype":"string-label","origin":"default","source_dtype":table.categorical[&table.headers[column]]["source_dtype"],"requested_dtype":null,"consumer":table.matrix.precision.consumer})
}
pub fn target_string(table: &TargetTable, row: usize, column: usize) -> Option<&str> {
    let pos = row * table.matrix.n_cols + column;
    if !table.matrix.validity[pos] {
        return None;
    }
    let value = *table.matrix.data.get(pos)?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    table
        .categorical
        .get(table.headers.get(column)?)?
        .get("categories")?
        .get(value as usize)?
        .as_str()
}
pub fn string_target_full(table: &TargetTable, column: usize) -> Value {
    let mut values = Vec::new();
    for row in 0..table.matrix.n_rows {
        string(&mut values, target_string(table, row, column).unwrap_or(""));
    }
    json!({"dtype":"string-label","n_rows":table.matrix.n_rows,"n_cols":1,"data_bytes_hex":hex(&values),"validity":table.matrix.column_matrix(column).validity,"precision":string_precision(table, column)})
}
pub fn target_bytes(table: &TargetTable, column: usize) -> Vec<u8> {
    let mut out = Vec::new();
    string(&mut out, &table.headers[column]);
    let kind = table.kinds[column];
    out.push(kind);
    if kind == 2 {
        out.extend_from_slice(&[3, 0]);
        option_string(
            &mut out,
            table.categorical[&table.headers[column]]["source_dtype"].as_str(),
        );
        out.push(0);
        option_string(&mut out, table.matrix.precision.consumer.as_deref());
        out.push(3);
        u64_bytes(&mut out, table.matrix.n_rows);
        u64_bytes(&mut out, 1);
        let mut values = Vec::new();
        for row in 0..table.matrix.n_rows {
            string(&mut values, target_string(table, row, column).unwrap_or(""));
        }
        bytes(&mut out, &values);
        bytes(
            &mut out,
            &table
                .matrix
                .column_matrix(column)
                .validity
                .iter()
                .map(|v| u8::from(*v))
                .collect::<Vec<_>>(),
        );
    } else {
        out.extend_from_slice(&table.matrix.precision.bytes());
        out.extend_from_slice(&table.matrix.column_matrix(column).numeric_bytes());
    }
    out
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct ExecutedProjectionIdentityV1 {
    pub profile: String,
    pub version: u32,
    pub scope: String,
    pub algorithm: String,
    pub feature_content_fingerprint: String,
    pub target_content_fingerprint: String,
    pub selected_source_ids: Vec<String>,
    pub sample_ids: Vec<String>,
    pub observation_ids: Vec<String>,
    pub target_names: Vec<String>,
    pub target_kinds: Vec<String>,
    pub features: Vec<Value>,
    pub targets: Option<Vec<Value>>,
    pub mask_policy: String,
    pub package_root: Option<String>,
    pub dataset_content_identity: Option<Value>,
}

fn identifier(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return Err(format!("Invalid primary identity {value:?}"));
    }
    Ok(())
}

impl DatasetPackage {
    fn verified_dataset_content_identity(&self) -> Result<Option<Value>, String> {
        let records = self
            .audits
            .iter()
            .filter(|a| a.get("owner_dataset_record").is_some())
            .collect::<Vec<_>>();
        let claims = self
            .audits
            .iter()
            .filter_map(|a| a.get("dataset_content_identity").filter(|v| !v.is_null()))
            .collect::<Vec<_>>();
        let unavailable = self
            .audits
            .iter()
            .filter_map(|audit| audit.get("dataset_content_identity_unavailable"))
            .collect::<Vec<_>>();
        if records.is_empty() {
            if !claims.is_empty() || !unavailable.is_empty() {
                return Err(
                    "Whole-dataset identity claim requires an actual owner dataset record".into(),
                );
            }
            return Ok(None);
        }
        if records.len() != 1 {
            return Err("Ambiguous whole-dataset owner records".into());
        }
        let record = &records[0]["owner_dataset_record"];
        let source = records[0]["selected_source"]
            .as_str()
            .ok_or("Whole-dataset evidence source absent")?;
        let feature = self
            .partitions
            .values()
            .flat_map(|p| &p.payloads)
            .find_map(|(_, p)| match p {
                PayloadBlock::FeatureMatrix(f) if f.source_id == source => Some(f),
                _ => None,
            })
            .ok_or("Whole-dataset evidence source disagrees with package")?;
        let target =
            self.partitions
                .values()
                .flat_map(|p| &p.payloads)
                .find_map(|(_, p)| match p {
                    PayloadBlock::TargetTable(t) => Some(t),
                    _ => None,
                });
        let options = super::numeric::NumericStorageOptions {
            features: feature.matrix.precision.requested_dtype,
            targets: target.and_then(|t| t.matrix.precision.requested_dtype),
            weights: None,
            consumer: feature.matrix.precision.consumer.clone(),
        };
        let expected = crate::public_dataset::masked_matrix_dataset_package_with_storage(
            record, source, &options,
        )?
        .0;
        if self.to_summary_value() != expected.to_summary_value()
            || self.partitions.keys().collect::<Vec<_>>()
                != expected.partitions.keys().collect::<Vec<_>>()
        {
            return Err(
                "Whole-dataset owner record does not correspond to materialized package".into(),
            );
        }
        let expected_audit = &expected.audits[0];
        for field in [
            "source",
            "selected_source",
            "raw_projection_source",
            "raw_source_schema",
            "raw_fold_assignments",
            "input_sample_ids",
            "target_mask",
            "target_names",
        ] {
            // These are actual normalized owner facts, not caller-provided
            // substitutes for the original record or the exact block buffers.
            if crate::canonical_json::canonical_json(&records[0][field])
                .map_err(|e| e.to_string())?
                != crate::canonical_json::canonical_json(&expected_audit[field])
                    .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "Whole-dataset owner evidence {field} disagrees with actual source"
                ));
            }
        }
        let expected_unavailable = expected
            .audits
            .iter()
            .filter_map(|audit| audit.get("dataset_content_identity_unavailable"))
            .collect::<Vec<_>>();
        if unavailable != expected_unavailable {
            return Err(
                "Whole-dataset unavailability status disagrees with actual owner record".into(),
            );
        }
        let identity = expected
            .audits
            .iter()
            .find_map(|audit| {
                audit
                    .get("dataset_content_identity")
                    .filter(|value| !value.is_null())
            })
            .cloned();
        if claims.iter().any(|claim| Some(*claim) != identity.as_ref()) {
            return Err("Whole-dataset identity claim disagrees with owner-derived content/profile/algorithm".into());
        }
        Ok(identity)
    }

    /// Exact preimages and receipt for the ordered base cohort, selecting existing sources.
    pub fn executed_projection(
        &self,
        selected: &[String],
    ) -> Result<(ExecutedProjectionIdentityV1, Vec<u8>, Vec<u8>), String> {
        self.validate_storage()?;
        if selected.len() != 1 {
            return Err("Base projection requires exactly one selected feature source; fusion is unsupported".into());
        }
        if selected.is_empty() || selected.iter().collect::<BTreeSet<_>>().len() != selected.len() {
            return Err("Explicit unique feature source selection required".into());
        }
        let assembled = self.to_assembled();
        let sample_column = self
            .identity
            .sample_id
            .as_deref()
            .ok_or("Projection requires stable sample identity")?;
        let mut shared = vec![0];
        let count: usize = self.partitions.values().map(|part| part.n_samples).sum();
        u64_bytes(&mut shared, count);
        let mut sample_ids = Vec::new();
        let mut observation_ids = Vec::new();
        let mut row_groups = Vec::new();
        for (partition, block) in &assembled.blocks {
            let frame = block
                .metadata
                .as_ref()
                .ok_or("Projection identity metadata absent")?;
            let get = |name: &str, row: usize| -> Result<String, String> {
                let cell = frame
                    .column(name)
                    .and_then(|c| c.values.get(row))
                    .ok_or("Projection identity column missing/misaligned")?;
                if matches!(cell, Cell::Na) || matches!(cell, Cell::Float(f) if !f.is_finite()) {
                    return Err("Missing projection identity".into());
                }
                let value = cell.to_str_scalar();
                identifier(&value)?;
                Ok(value)
            };
            for row in 0..block.n_samples {
                let sample = get(sample_column, row)?;
                let observation = get(
                    self.identity
                        .observation_id
                        .as_deref()
                        .unwrap_or(sample_column),
                    row,
                )?;
                string(&mut shared, &sample);
                string(&mut shared, &observation);
                string(&mut shared, partition);
                let group = self
                    .identity
                    .group_id
                    .as_deref()
                    .map(|column| get(column, row))
                    .transpose()?;
                let origin = frame
                    .has_column("origin_id")
                    .then(|| get("origin_id", row))
                    .transpose()?;
                let repetition = self
                    .identity
                    .repetition_id
                    .as_deref()
                    .map(|column| get(column, row))
                    .transpose()?;
                for value in [&group, &origin, &repetition] {
                    option_string(&mut shared, value.as_deref());
                }
                row_groups.push(group);
                sample_ids.push(sample);
                observation_ids.push(observation);
            }
        }
        let observations = observation_ids.iter().collect::<BTreeSet<_>>();
        if observations.len() != observation_ids.len() {
            return Err("Duplicate observation identity".into());
        }
        if self.identity.repetition_id.is_none()
            && sample_ids.iter().collect::<BTreeSet<_>>().len() != sample_ids.len()
        {
            return Err("Repeated samples require repetition semantics".into());
        }
        if !self.folds.is_empty() && self.folds.len() != self.fold_provenance.len() {
            return Err("Folds require stable observation provenance".into());
        }
        shared.push(u8::from(!self.fold_provenance.is_empty()));
        if !self.fold_provenance.is_empty() {
            u64_bytes(&mut shared, self.fold_provenance.len());
            for fold in &self.fold_provenance {
                let train = fold.train_observation_ids.iter().collect::<BTreeSet<_>>();
                let validation = fold
                    .validation_observation_ids
                    .iter()
                    .collect::<BTreeSet<_>>();
                if train.len() != fold.train_observation_ids.len()
                    || validation.len() != fold.validation_observation_ids.len()
                    || !train.is_disjoint(&validation)
                    || !train.is_subset(&observations)
                    || !validation.is_subset(&observations)
                {
                    return Err("Invalid observation-keyed fold membership".into());
                }
                let train_samples = observation_ids
                    .iter()
                    .zip(&sample_ids)
                    .filter(|(id, _)| train.contains(id))
                    .map(|(_, sample)| sample)
                    .collect::<BTreeSet<_>>();
                if observation_ids
                    .iter()
                    .zip(&sample_ids)
                    .any(|(id, sample)| validation.contains(id) && train_samples.contains(sample))
                {
                    return Err("Fold sample leakage".into());
                }
                let train_groups = observation_ids
                    .iter()
                    .zip(&row_groups)
                    .filter(|(id, _)| train.contains(id))
                    .filter_map(|(_, group)| group.as_ref())
                    .collect::<BTreeSet<_>>();
                if observation_ids.iter().zip(&row_groups).any(|(id, group)| {
                    validation.contains(id)
                        && group.as_ref().is_some_and(|g| train_groups.contains(g))
                }) {
                    return Err("Fold group leakage".into());
                }
                strings(&mut shared, &fold.train_observation_ids);
                strings(&mut shared, &fold.validation_observation_ids);
            }
        }
        let audit = self
            .audits
            .iter()
            .find(|a| a.get("raw_fold_assignments").is_some());
        shared.push(u8::from(audit.is_some()));
        if let Some(audit) = audit {
            let input = audit["input_sample_ids"]
                .as_array()
                .ok_or("Raw fold sample order absent")?;
            let raw = audit["raw_fold_assignments"]
                .as_array()
                .ok_or("Raw fold assignments absent")?;
            if input.len() != raw.len()
                || raw
                    .iter()
                    .any(|v| !v.is_null() && v.as_str().is_none_or(str::is_empty))
            {
                return Err("Invalid raw fold assignments".into());
            }
            if input.len() != count
                || input
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<BTreeSet<_>>()
                    .len()
                    != count
                || sample_ids
                    .iter()
                    .any(|sample| !input.iter().any(|v| v.as_str() == Some(sample)))
            {
                return Err("Raw fold input identities disagree with cohort".into());
            }
            for fold in &self.fold_provenance {
                let assignment = |observation: &str| -> Result<Option<&str>, String> {
                    let row = observation_ids
                        .iter()
                        .position(|id| id == observation)
                        .ok_or("Fold observation absent")?;
                    let index = input
                        .iter()
                        .position(|v| v.as_str() == Some(&sample_ids[row]))
                        .ok_or("Raw fold sample absent")?;
                    Ok(raw[index].as_str())
                };
                let validation_labels = fold
                    .validation_observation_ids
                    .iter()
                    .map(|id| assignment(id))
                    .collect::<Result<BTreeSet<_>, _>>()?;
                if validation_labels.iter().any(Option::is_some)
                    && (validation_labels.len() != 1
                        || validation_labels.contains(&None)
                        || fold
                            .train_observation_ids
                            .iter()
                            .map(|id| assignment(id))
                            .collect::<Result<Vec<_>, _>>()?
                            .iter()
                            .any(|label| label.is_some() && validation_labels.contains(label)))
                {
                    return Err("Raw folds disagree with fold provenance".into());
                }
            }
            u64_bytes(&mut shared, count);
            for sample in &sample_ids {
                let index = input
                    .iter()
                    .position(|v| v.as_str() == Some(sample))
                    .ok_or("Raw fold identity mismatch")?;
                option_string(&mut shared, raw[index].as_str());
            }
        }
        let head = |side: &str| {
            let mut out = Vec::new();
            string(&mut out, EXECUTED_PROJECTION_PROFILE);
            out.extend_from_slice(&1_u32.to_le_bytes());
            string(&mut out, side);
            out.extend_from_slice(&shared);
            out
        };
        let mut feature_preimage = head("features");
        u64_bytes(&mut feature_preimage, selected.len());
        let mut feature_descriptors = Vec::new();
        for source in selected {
            let mut whole: Option<FeatureMatrix> = None;
            for partition in self.partitions.values() {
                let feature = partition
                    .payloads
                    .iter()
                    .find_map(|(_, p)| match p {
                        PayloadBlock::FeatureMatrix(f) if f.source_id == *source => Some(f),
                        _ => None,
                    })
                    .ok_or("Selected feature source missing from partition")?;
                feature.matrix.precision.validate(true)?;
                if let Some(whole) = &mut whole {
                    if whole.feature_names != feature.feature_names
                        || whole.headers != feature.headers
                        || whole.axes != feature.axes
                        || whole.representation_id != feature.representation_id
                        || whole.matrix.precision != feature.matrix.precision
                    {
                        return Err("Feature descriptor differs across partitions".into());
                    }
                    whole.matrix.n_rows += feature.matrix.n_rows;
                    whole.matrix.data.extend_from_slice(&feature.matrix.data);
                    whole
                        .matrix
                        .validity
                        .extend_from_slice(&feature.matrix.validity);
                } else {
                    whole = Some(feature.clone());
                }
            }
            let whole = whole.ok_or("Projection has no feature rows")?;
            feature_preimage.extend_from_slice(&feature_bytes(&whole));
            feature_descriptors.push(json!({"source_id":source,"representation_id":whole.representation_id,"precision":whole.matrix.precision,"shape":[whole.matrix.n_rows,whole.matrix.n_cols],"feature_names":whole.feature_names,"axes":axes_full(&whole.axes)?,"storage_dtype":whole.matrix.precision.dtype,"delivered_dtype":"float64"}));
        }
        let mut whole_targets: Option<TargetTable> = None;
        let mut absent = false;
        for partition in self.partitions.values() {
            let target = partition.payloads.iter().find_map(|(_, p)| match p {
                PayloadBlock::TargetTable(t) => Some(t),
                _ => None,
            });
            if let Some(target) = target {
                if absent {
                    return Err("Targets absent in one partition".into());
                }
                if let Some(whole) = &mut whole_targets {
                    if whole.headers != target.headers
                        || whole.kinds != target.kinds
                        || whole.categorical != target.categorical
                        || whole.matrix.precision != target.matrix.precision
                    {
                        return Err("Target descriptor differs across partitions".into());
                    }
                    whole.matrix.n_rows += target.matrix.n_rows;
                    whole.matrix.data.extend_from_slice(&target.matrix.data);
                    whole
                        .matrix
                        .validity
                        .extend_from_slice(&target.matrix.validity);
                } else {
                    whole_targets = Some(target.clone());
                }
            } else {
                if whole_targets.is_some() {
                    return Err("Targets absent in one partition".into());
                }
                absent = true;
            }
        }
        let mut target_preimage = head("targets");
        target_preimage.push(u8::from(whole_targets.is_some()));
        let mut target_names = Vec::new();
        let mut target_kinds = Vec::new();
        let mut descriptors = Vec::new();
        if let Some(table) = &whole_targets {
            table.matrix.precision.validate(false)?;
            u64_bytes(&mut target_preimage, table.headers.len());
            for (column, name) in table.headers.iter().enumerate() {
                target_preimage.extend_from_slice(&target_bytes(table, column));
                target_names.push(name.clone());
                let kind = match table.kinds[column] {
                    0 => "numeric",
                    1 => "classification_integer",
                    _ => "classification_string",
                };
                target_kinds.push(kind.into());
                descriptors.push(json!({"name":name,"kind":kind,"precision":if table.kinds[column]==2 { string_precision(table, column) } else { serde_json::to_value(&table.matrix.precision).unwrap() },"shape":[table.matrix.n_rows,1]}));
            }
        }
        let identity = ExecutedProjectionIdentityV1 {
            profile: EXECUTED_PROJECTION_PROFILE.into(),
            version: 1,
            scope: "base".into(),
            algorithm: "sha256".into(),
            feature_content_fingerprint: format!("{:x}", Sha256::digest(&feature_preimage)),
            target_content_fingerprint: format!("{:x}", Sha256::digest(&target_preimage)),
            selected_source_ids: selected.to_vec(),
            sample_ids,
            observation_ids,
            target_names,
            target_kinds,
            features: feature_descriptors,
            targets: whole_targets.as_ref().map(|_| descriptors),
            mask_policy: MASK_POLICY.into(),
            package_root: Some(self.manifest().root),
            dataset_content_identity: self.verified_dataset_content_identity()?,
        };
        Ok((identity, feature_preimage, target_preimage))
    }
}
impl ExecutedProjectionIdentityV1 {
    pub fn validate(&self, package: &DatasetPackage) -> Result<(), String> {
        if self.profile != EXECUTED_PROJECTION_PROFILE
            || self.version != 1
            || self.scope != "base"
            || self.algorithm != "sha256"
            || self.mask_policy != MASK_POLICY
        {
            return Err(
                "Unsupported executed-projection profile/version/scope/algorithm/mask policy"
                    .into(),
            );
        }
        let (actual, _, _) = package.executed_projection(&self.selected_source_ids)?;
        if *self != actual {
            return Err("Executed projection disagrees with package buffers/metadata".into());
        }
        Ok(())
    }
}

/// Agreed exact target-column wire, in the original header order.
pub fn target_columns(table: &TargetTable) -> Vec<Value> {
    table.headers.iter().enumerate().map(|(column,name)| json!({"name":name,
        "kind":match table.kinds[column] { 0=>"numeric",1=>"classification_integer",_=>"classification_string" },
        "block":if table.kinds[column]==2 { string_target_full(table,column) } else { table.matrix.column_matrix(column).full_value() }
    })).collect()
}

pub fn axes_full(axes: &[Value]) -> Result<Vec<Value>, String> {
    axes.iter()
        .map(|axis| {
            let coordinates = axis["coordinates"].as_array().map(|values| -> Result<Value, String> {
                if values.iter().all(Value::is_number) {
                    let data = values
                        .iter()
                        .map(super::numeric::exact_json_number_f64).collect::<Result<Vec<_>,_>>()?
                        .into_iter().flat_map(f64::to_le_bytes).collect::<Vec<_>>();
                    Ok(json!({"kind":"numeric","count":values.len(),"data_bytes_hex":hex(&data)}))
                } else {
                    Ok(json!({"kind":"string","values":values}))
                }
            }).transpose()?;
            Ok(json!({"name":axis["name"],"unit":axis["unit"],"coordinates":coordinates}))
        })
        .collect()
}

impl TryFrom<Value> for ExecutedProjectionIdentityV1 {
    type Error = String;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let fields = [
            "profile",
            "version",
            "scope",
            "algorithm",
            "feature_content_fingerprint",
            "target_content_fingerprint",
            "selected_source_ids",
            "sample_ids",
            "observation_ids",
            "target_names",
            "target_kinds",
            "features",
            "targets",
            "mask_policy",
            "package_root",
            "dataset_content_identity",
        ];
        let object = value
            .as_object()
            .ok_or("Executed projection object required")?;
        if object.len() != fields.len() || fields.iter().any(|name| !object.contains_key(*name)) {
            return Err("Executed projection requires exact receipt fields".into());
        }
        macro_rules! field {
            ($name:literal) => {
                serde_json::from_value(value[$name].clone()).map_err(|e| e.to_string())?
            };
        }
        let receipt = Self {
            profile: field!("profile"),
            version: field!("version"),
            scope: field!("scope"),
            algorithm: field!("algorithm"),
            feature_content_fingerprint: field!("feature_content_fingerprint"),
            target_content_fingerprint: field!("target_content_fingerprint"),
            selected_source_ids: field!("selected_source_ids"),
            sample_ids: field!("sample_ids"),
            observation_ids: field!("observation_ids"),
            target_names: field!("target_names"),
            target_kinds: field!("target_kinds"),
            features: field!("features"),
            targets: field!("targets"),
            mask_policy: field!("mask_policy"),
            package_root: field!("package_root"),
            dataset_content_identity: field!("dataset_content_identity"),
        };
        if receipt.profile != EXECUTED_PROJECTION_PROFILE
            || receipt.version != 1
            || receipt.scope != "base"
            || receipt.algorithm != "sha256"
            || receipt.mask_policy != MASK_POLICY
        {
            return Err(
                "Unsupported executed projection profile/version/scope/algorithm/mask policy"
                    .into(),
            );
        }
        for fingerprint in [
            &receipt.feature_content_fingerprint,
            &receipt.target_content_fingerprint,
        ] {
            if fingerprint.len() != 64
                || !fingerprint
                    .bytes()
                    .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
            {
                return Err("Invalid executed projection digest".into());
            }
        }
        Ok(receipt)
    }
}

/// Read only the agreed exact Axis records retained in owner audits.
pub(crate) fn axes_from_full(value: &Value, columns: usize) -> Result<Vec<Value>, String> {
    value
        .as_array()
        .ok_or("Projection axes array required")?
        .iter()
        .map(|axis| {
            let fields = axis.as_object().ok_or("Projection axis object required")?;
            if fields.len() != 3
                || !["name", "unit", "coordinates"]
                    .iter()
                    .all(|name| fields.contains_key(*name))
                || axis["name"].as_str().is_none_or(str::is_empty)
                || (!axis["unit"].is_null() && !axis["unit"].is_string())
            {
                return Err("Malformed projection Axis".into());
            }
            let raw = &axis["coordinates"];
            let coordinates = if raw.is_null() {
                Value::Null
            } else if raw["kind"] == "numeric" {
                let count = raw["count"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("Invalid coordinate count")?;
                let text = raw["data_bytes_hex"]
                    .as_str()
                    .ok_or("Exact coordinate bytes required")?;
                if raw.as_object().is_none_or(|o| o.len() != 3)
                    || count != columns
                    || count.checked_mul(16) != Some(text.len())
                    || !text
                        .bytes()
                        .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
                {
                    return Err("Coordinate dtype/count/bytes mismatch".into());
                }
                let mut values = Vec::with_capacity(count);
                for chunk in text.as_bytes().as_chunks::<16>().0 {
                    let mut bytes = [0; 8];
                    for i in 0..8 {
                        bytes[i] = u8::from_str_radix(
                            std::str::from_utf8(&chunk[i * 2..i * 2 + 2])
                                .map_err(|e| e.to_string())?,
                            16,
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    let value = f64::from_le_bytes(bytes);
                    if !value.is_finite() {
                        return Err("Observed coordinate is nonfinite".into());
                    }
                    values.push(json!(value));
                }
                Value::Array(values)
            } else if raw["kind"] == "string" {
                let values = raw["values"]
                    .as_array()
                    .ok_or("String coordinates array required")?;
                if raw.as_object().is_none_or(|o| o.len() != 2)
                    || values.len() != columns
                    || values.iter().any(|v| !v.is_string())
                {
                    return Err("String coordinate shape/type mismatch".into());
                }
                Value::Array(values.clone())
            } else {
                return Err("Unsupported coordinate kind".into());
            };
            Ok(json!({"name":axis["name"],"unit":axis["unit"],"coordinates":coordinates}))
        })
        .collect()
}

#[cfg(test)]
mod numeric_coordinate_tests {
    use super::axes_from_full;
    use serde_json::{json, Value};

    #[test]
    fn numeric_coordinates_owner_corpus() {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../../bindings/wasm/tests/fixtures/numeric-coordinates.json"
        ))
        .unwrap();
        assert_eq!(
            corpus["schema"],
            "nirs4all.io.numeric-coordinates-corpus.v1"
        );
        for case in corpus["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let columns = usize::try_from(case["expectedCount"].as_u64().unwrap()).unwrap();
            let axes = json!([{"name":"wavelength","unit":"nm","coordinates":case["packet"]}]);
            let result = axes_from_full(&axes, columns);
            if let Some(refusal) = case.get("refusal") {
                assert_eq!(result.unwrap_err(), refusal.as_str().unwrap(), "{id}");
                continue;
            }
            let decoded = result.unwrap_or_else(|error| panic!("{id}: {error}"));
            assert_eq!(decoded.len(), 1, "{id}");
            let coordinates = decoded[0]["coordinates"].as_array().unwrap();
            let bits = case["bits"].as_array().unwrap();
            let values = case["values"].as_array().unwrap();
            assert_eq!(coordinates.len(), columns, "{id}");
            assert_eq!(bits.len(), columns, "{id}");
            assert_eq!(values.len(), columns, "{id}");
            for (index, coordinate) in coordinates.iter().enumerate() {
                let actual = coordinate.as_f64().unwrap();
                let expected = u64::from_str_radix(bits[index].as_str().unwrap(), 16).unwrap();
                assert!(actual.is_finite(), "{id}[{index}]");
                assert_eq!(actual.to_bits(), expected, "{id}[{index}] literal bits");
                assert_eq!(
                    actual.to_bits(),
                    values[index].as_f64().unwrap().to_bits(),
                    "{id}[{index}] literal value/sign"
                );
            }
        }
    }
}
