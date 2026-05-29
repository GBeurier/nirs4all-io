// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! `to_dag_ml_data`: build a dag-ml-data `CoordinatorDataPlanEnvelope` from a
//! nirs4all-io [`AssembledDataset`] (EPIC 10, D-R8, ADR-0001).
//!
//! io owns the assembled → envelope bridge. It maps the `AssembledDataset` onto a
//! `DatasetSchema` (+ `SourceDescriptor`/`RepresentationSpec`/`AxisSpec`;
//! nm→`Wavelength`, cm⁻¹→`Wavenumber`; signal_type→`tags`), a minimal `DataPlan`,
//! and a `SampleRelationTable` (observation/sample/group/repetition identity from
//! the repetition key, the only leakage unit io knows), then calls
//! `CoordinatorDataPlanEnvelope::from_parts` (it computes the three fingerprints
//! and self-validates). io does **not** build dag-ml `FoldSet`/`DataBinding` —
//! those are dag-ml's domain (folds/campaigns).
//!
//! This crate is **excluded from the nirs4all-io workspace**: it path-depends on
//! the `dag-ml-data` sibling, and an absent optional path dep would break
//! standalone `cargo build` resolution (even with the feature off). It builds
//! only in the ecosystem tree.

use std::collections::{BTreeMap, BTreeSet};

use dag_ml_data::{
    AxisKind, AxisSpec, CoordinatorDataPlanEnvelope, DataPlan, DataPlanStep, DataPlanStepKind,
    DatasetSchema, FitScope, GroupId, ObservationId, RepetitionId, RepresentationId,
    RepresentationSpec, SampleId, SampleRelation, SampleRelationTable, SignalKind,
    SourceDescriptor, SourceGranularity, SourceId, TargetId, TypeId,
};
use nirs4all_io::core::spec::SpecError;
use nirs4all_io::materialize::AssembledDataset;
use serde_json::Value;

fn err<E: std::fmt::Display>(e: E) -> SpecError {
    SpecError::new(e.to_string())
}

/// Coerce an arbitrary label into a dag-ml-data identifier (ASCII alnum / `_-.`,
/// 1..=128 bytes). Unsupported characters collapse to `_`; an all-`_` or empty
/// result falls back to `fallback`.
fn sanitize(raw: &str, fallback: &str) -> String {
    let mut s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.trim_matches('_').is_empty() {
        s = fallback.to_string();
    }
    if s.len() > 128 {
        s.truncate(128);
    }
    s
}

fn signal_kind(signal: &str) -> SignalKind {
    match signal.to_ascii_lowercase().as_str() {
        "absorbance" => SignalKind::Absorbance,
        "reflectance" => SignalKind::Reflectance,
        "transmittance" => SignalKind::Transmittance,
        "log_reflectance" => SignalKind::LogReflectance,
        _ => SignalKind::Unknown,
    }
}

/// Map an io header unit onto a spectral axis kind + canonical unit string.
fn feature_axis(unit: &str) -> (AxisKind, Option<String>, &'static str) {
    let u = unit.to_ascii_lowercase();
    if u.contains("nm") || u.contains("nanomet") || u.contains("wavelength") {
        (AxisKind::Wavelength, Some("nm".to_string()), "wavelength")
    } else if u.contains("cm-1")
        || u.contains("cm^-1")
        || u.contains("1/cm")
        || u.contains("wavenumber")
        || u.contains("cm⁻¹")
    {
        (AxisKind::Wavenumber, Some("cm-1".to_string()), "wavenumber")
    } else {
        (AxisKind::Feature, None, "feature")
    }
}

/// Numeric axis coordinates from feature headers, only when every header parses
/// to a finite number and the count matches the axis size (else `None`, so the
/// `AxisSpec` size/coordinates invariant always holds).
fn numeric_coords(headers: &[String], size: usize) -> Option<Vec<Value>> {
    if headers.len() != size {
        return None;
    }
    let mut out = Vec::with_capacity(size);
    for h in headers {
        let v: f64 = h.trim().parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        out.push(Value::from(v));
    }
    Some(out)
}

/// Map an [`AssembledDataset`] to a dag-ml-data `CoordinatorDataPlanEnvelope`.
pub fn to_dag_ml_data(
    assembled: &AssembledDataset,
) -> Result<CoordinatorDataPlanEnvelope, SpecError> {
    let Some((_first, b0)) = assembled.blocks.iter().next() else {
        return Err(SpecError::new(
            "cannot emit dag-ml-data: dataset has no partitions",
        ));
    };
    if assembled.n_sources == 0 || b0.x.is_empty() {
        return Err(SpecError::new(
            "cannot emit dag-ml-data: dataset has no feature source",
        ));
    }

    // --- sample / observation identity across all partitions ---
    // Use the repetition key (a leakage unit) iff it is present and aligned in
    // every partition; otherwise each observation is its own 1:1 sample with no
    // group_id (roadmap 10.2: group_id only when the key is a leakage unit).
    let rep_col = assembled.repetition.as_deref();
    let use_rep = rep_col.is_some_and(|col| {
        assembled.blocks.values().all(|b| {
            b.metadata
                .as_ref()
                .is_some_and(|f| f.has_column(col) && f.str_column(col).len() == b.n_samples)
        })
    });

    let mut sample_ids: Vec<SampleId> = Vec::new();
    let mut sample_seen: BTreeSet<String> = BTreeSet::new();
    let mut obs_per_sample: BTreeMap<String, usize> = BTreeMap::new();
    let mut rows: Vec<SampleRelation> = Vec::new();
    let mut global = 0usize;
    for b in assembled.blocks.values() {
        let rep_vals = if use_rep {
            Some(
                b.metadata
                    .as_ref()
                    .expect("checked")
                    .str_column(rep_col.expect("checked")),
            )
        } else {
            None
        };
        for r in 0..b.n_samples {
            let (observation, sample, group, repetition) = if let Some(rv) = &rep_vals {
                let key = sanitize(&rv[r], "sample");
                let n = obs_per_sample.entry(key.clone()).or_insert(0);
                let observation = format!("{key}.obs{n}");
                let repetition = format!("rep.{n}");
                *n += 1;
                (observation, key.clone(), Some(key), Some(repetition))
            } else {
                let pair = (format!("obs.{global}"), format!("s.{global}"), None, None);
                global += 1;
                pair
            };
            if sample_seen.insert(sample.clone()) {
                sample_ids.push(SampleId::new(&sample).map_err(err)?);
            }
            rows.push(SampleRelation {
                observation_id: ObservationId::new(&observation).map_err(err)?,
                sample_id: SampleId::new(&sample).map_err(err)?,
                source_id: None,
                target_id: None,
                group_id: group.map(|g| GroupId::new(&g)).transpose().map_err(err)?,
                origin_id: None,
                repetition_id: repetition
                    .map(|r| RepetitionId::new(&r))
                    .transpose()
                    .map_err(err)?,
                augmented: false,
                excluded: false,
                metadata: BTreeMap::new(),
                augmentation: None,
            });
        }
    }
    let n_samples = sample_ids.len();

    // --- sources (each X source → a rank-2 [sample, feature] representation) ---
    let mut sources = Vec::with_capacity(assembled.n_sources);
    for k in 0..assembled.n_sources {
        let n_features = b0.x.get(k).map(|m| m.n_cols).unwrap_or(0);
        let headers = b0.feature_headers.get(k).cloned().unwrap_or_default();
        let unit = b0.header_units.get(k).cloned().unwrap_or_default();
        let signal = b0.signal_types.get(k).and_then(Clone::clone);
        let (kind, axis_unit, axis_name) = feature_axis(&unit);
        let rep_id = sanitize(&format!("src_{k}_native"), "rep");
        let axes = vec![
            AxisSpec {
                name: "sample".into(),
                kind: AxisKind::Sample,
                unit: None,
                size: Some(n_samples),
                variable: false,
                coordinates: None,
            },
            AxisSpec {
                name: axis_name.into(),
                kind,
                unit: axis_unit,
                size: Some(n_features),
                variable: false,
                coordinates: numeric_coords(&headers, n_features),
            },
        ];
        let mut tags = BTreeMap::new();
        if let Some(sig) = &signal {
            tags.insert("signal_type".to_string(), Value::String(sig.clone()));
        }
        sources.push(SourceDescriptor {
            id: SourceId::new(sanitize(&format!("src_{k}"), "src")).map_err(err)?,
            name: format!("source {k}"),
            type_id: TypeId::new("dense_signal").map_err(err)?,
            modality: "spectroscopy".into(),
            native_representation: RepresentationSpec {
                id: RepresentationId::new(&rep_id).map_err(err)?,
                type_id: TypeId::new("dense_signal").map_err(err)?,
                rank: Some(2),
                axes,
                container: "ndarray".into(),
                dtype: Some("float64".into()),
                sparse: false,
                ragged: false,
                signal_type: signal.as_deref().map(signal_kind),
            },
            sample_key: "sample_id".into(),
            granularity: if use_rep {
                SourceGranularity::PerSampleRepeated
            } else {
                SourceGranularity::PerSample
            },
            schema: BTreeMap::new(),
            tags,
            shape_contract: None,
        });
    }

    // --- targets (one representation per y column) ---
    let mut targets: BTreeMap<TargetId, RepresentationSpec> = BTreeMap::new();
    if b0.y.is_some() {
        for (i, header) in b0.y_headers.iter().enumerate() {
            let tid = sanitize(header, &format!("target{i}"));
            let rep_id = sanitize(&format!("target_{tid}"), "target");
            let rep = RepresentationSpec {
                id: RepresentationId::new(&rep_id).map_err(err)?,
                type_id: TypeId::new("tabular_numeric").map_err(err)?,
                rank: Some(2),
                axes: vec![
                    AxisSpec {
                        name: "sample".into(),
                        kind: AxisKind::Sample,
                        unit: None,
                        size: Some(n_samples),
                        variable: false,
                        coordinates: None,
                    },
                    AxisSpec {
                        name: "target".into(),
                        kind: AxisKind::Target,
                        unit: None,
                        size: Some(1),
                        variable: false,
                        coordinates: Some(vec![Value::String(header.clone())]),
                    },
                ],
                container: "dataframe".into(),
                dtype: Some("float64".into()),
                sparse: false,
                ragged: false,
                signal_type: None,
            };
            targets.insert(TargetId::new(&tid).map_err(err)?, rep);
        }
    }

    let schema = DatasetSchema {
        dataset_id: sanitize(&assembled.name, "dataset"),
        sample_ids,
        sources,
        targets,
        metadata: BTreeMap::new(),
        metadata_schema: None,
        groups: vec![],
        folds: vec![],
    };

    // --- plan: materialize each source, join when multi-source ---
    let mut steps: Vec<DataPlanStep> = schema
        .sources
        .iter()
        .map(|s| DataPlanStep {
            kind: DataPlanStepKind::Materialize,
            source_id: Some(s.id.clone()),
            adapter_id: None,
            input_representation: None,
            output_representation: Some(s.native_representation.id.clone()),
            fit_scope: FitScope::Stateless,
            requires_user_choice: false,
            metadata: BTreeMap::new(),
        })
        .collect();
    let output_representation = if schema.sources.len() == 1 {
        schema.sources[0].native_representation.id.clone()
    } else {
        let model_input = RepresentationId::new("model_input").map_err(err)?;
        steps.push(DataPlanStep {
            kind: DataPlanStepKind::Join,
            source_id: None,
            adapter_id: None,
            input_representation: None,
            output_representation: Some(model_input.clone()),
            fit_scope: FitScope::Stateless,
            requires_user_choice: false,
            metadata: BTreeMap::new(),
        });
        model_input
    };
    let plan = DataPlan {
        id: sanitize(&format!("{}_plan", assembled.name), "plan"),
        steps,
        output_representation,
        issues: vec![],
    };

    let relations = if rows.is_empty() {
        None
    } else {
        Some(SampleRelationTable { rows })
    };
    CoordinatorDataPlanEnvelope::from_parts(&schema, plan, relations.as_ref()).map_err(err)
}
