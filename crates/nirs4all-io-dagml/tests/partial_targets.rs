// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Partial target validity across the IO -> DMD bridge.
//!
//! A record whose targets are observed on different rows is bridged as one
//! scalar DMD target per column. Each scalar table holds exactly that column's
//! observed samples; `PackageProvider::multi_target_block` reports the other
//! samples as missing, never as stored values. Complete homogeneous records
//! keep their single DMD matrix target.

use std::collections::BTreeMap;

use dag_ml_data::{CoordinatorDataMaterializationRequest, DataView, SampleId, SourceId, TargetId};
use dag_ml_data_provider::DagMlDataProvider;
use nirs4all_io_core::public_dataset::masked_matrix_dataset_package;
use nirs4all_io_dagml::PackageProvider;
use serde_json::{json, Value};

const SOURCE: &str = "spectra";
const N_FEATURES: usize = 3;

/// Deterministic feature value of one sample row.
fn feature(row: usize, column: usize) -> f64 {
    (row * N_FEATURES + column) as f64 * 0.5 + 0.25
}

/// Deterministic truth of one target cell.
fn truth(row: usize, target: usize) -> f64 {
    (row as f64) * (target as f64 + 1.0) + target as f64 * 0.125
}

/// A `nirs4all.dataset.v2` record. `observed[row][target]` is the target
/// mask; unobserved cells are `null`, never a value.
fn record(names: &[&str], observed: &[Vec<bool>], partitions: &[&str], task_type: &str) -> Value {
    let rows = observed.len();
    let ids: Vec<String> = (0..rows).map(|row| format!("s{row}")).collect();
    let x: Vec<Vec<f64>> = (0..rows)
        .map(|row| (0..N_FEATURES).map(|column| feature(row, column)).collect())
        .collect();
    let classification = task_type == "classification";
    let y: Vec<Value> = observed
        .iter()
        .enumerate()
        .map(|(row, cells)| {
            Value::Array(
                cells
                    .iter()
                    .enumerate()
                    .map(|(target, seen)| match (*seen, classification) {
                        (false, _) => Value::Null,
                        (true, true) => json!(((row + target) % 3) as i64),
                        (true, false) => json!(truth(row, target)),
                    })
                    .collect(),
            )
        })
        .collect();
    json!({
        "schema": "nirs4all.dataset.v2", "schema_version": 2, "origin_ids": ids,
        "fold_ids": vec![Value::Null; rows],
        "dataset": {
            "schema": "nirs4all.multimodal-dataset", "schema_version": 1, "name": "partial-targets",
            "task_type": task_type, "sample_ids": ids,
            "sources": [{"name": SOURCE, "sample_ids": ids, "representation_id": "signal_1d",
                "axes": ["sample", "wavelength"], "feature_names": null, "axis_units": {},
                "axis_coordinates": {},
                "array": {"dtype": "float64", "shape": [rows, N_FEATURES], "values": x}}],
            "y": {"dtype": if classification { "int64" } else { "float64" },
                "shape": [rows, names.len()], "values": y},
            "target_names": names,
            "target_mask": {"dtype": "bool", "shape": [rows, names.len()], "values": observed},
            "groups": null,
            "partitions": {"dtype": "<U7", "shape": [rows], "values": partitions}
        }
    })
}

fn provider(value: &Value) -> PackageProvider {
    let (package, _) = masked_matrix_dataset_package(value, SOURCE).expect("masked package");
    PackageProvider::from_package_source(&package, SOURCE).expect("bridge")
}

/// A scheduler-shaped view over exactly `sample_ids`.
fn view(provider: &PackageProvider, sample_ids: &[String]) -> u64 {
    let envelope = provider.envelope();
    let data = provider
        .materialize(&CoordinatorDataMaterializationRequest {
            run_id: "partial".into(),
            node_id: "model".into(),
            input_name: "X".into(),
            phase: "fit".into(),
            variant_id: None,
            fold_id: None,
            request_id: "partial.request".into(),
            schema_fingerprint: envelope.schema_fingerprint.clone(),
            plan_fingerprint: envelope.plan_fingerprint.clone(),
            relation_fingerprint: envelope.relation_fingerprint.clone(),
            output_representation: envelope.plan.output_representation.clone(),
            source_ids: vec![SourceId::new(SOURCE).unwrap()],
            require_relations: true,
            predict_cohort: None,
        })
        .unwrap();
    provider
        .make_view(
            data.handle.handle,
            &DataView {
                sample_ids: Some(
                    sample_ids
                        .iter()
                        .map(|id| SampleId::new(id).unwrap())
                        .collect(),
                ),
                source_ids: Some(vec![SourceId::new(SOURCE).unwrap()]),
                ..Default::default()
            },
        )
        .unwrap()
        .handle
        .handle
}

fn ids(rows: impl IntoIterator<Item = usize>) -> Vec<String> {
    rows.into_iter().map(|row| format!("s{row}")).collect()
}

fn row_of(sample_id: &SampleId) -> usize {
    sample_id.as_str()[1..].parse().unwrap()
}

/// Checks a multi-target block against the record by sample identity:
/// features, observed values and validity all join on `sample_id`.
fn assert_block_matches_record(
    provider: &PackageProvider,
    view_handle: u64,
    observed: &[Vec<bool>],
    expected_samples: &[String],
) {
    let block = provider.multi_target_block(view_handle).unwrap();
    assert_eq!(block.target_ids, provider.target_ids());
    let mut got = block
        .sample_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>();
    got.sort();
    let mut want = expected_samples.to_vec();
    want.sort();
    assert_eq!(got, want);
    let features = provider.feature_block_f64(view_handle).unwrap();
    let feature_rows = features
        .sample_ids
        .iter()
        .enumerate()
        .map(|(position, id)| (id.clone(), position))
        .collect::<BTreeMap<_, _>>();
    for (position, sample_id) in block.sample_ids.iter().enumerate() {
        let row = row_of(sample_id);
        let feature_position = feature_rows[sample_id];
        assert_eq!(
            &features.values[feature_position * N_FEATURES..(feature_position + 1) * N_FEATURES],
            (0..N_FEATURES)
                .map(|column| feature(row, column))
                .collect::<Vec<_>>()
                .as_slice(),
            "features of {sample_id}"
        );
        for (target, seen) in observed[row].iter().copied().enumerate() {
            assert_eq!(block.validity_masks[target][position], seen);
            let value = &block.values[target][position];
            if seen {
                assert_eq!(
                    value.as_f64(),
                    Some(truth(row, target)),
                    "{sample_id}/{target}"
                );
            } else {
                assert!(
                    value.is_null(),
                    "unobserved {sample_id}/{target} must stay null"
                );
            }
        }
    }
}

#[test]
fn two_targets_observed_on_different_rows_bridge_per_target() {
    // protein is unobserved on s1, moisture on s4 (the reported reproducer).
    let observed: Vec<Vec<bool>> = (0..6).map(|row| vec![row != 1, row != 4]).collect();
    let value = record(
        &["protein", "moisture"],
        &observed,
        &["train"; 6],
        "regression",
    );
    let provider = provider(&value);

    assert_eq!(provider.target_names(), ["protein", "moisture"]);
    let target_ids = provider
        .target_ids()
        .iter()
        .map(TargetId::as_str)
        .collect::<Vec<_>>();
    assert_eq!(target_ids, ["protein", "moisture"], "target-column order");
    let targets = &provider.envelope().metadata["io.dataset_schema"]["targets"];
    for name in ["protein", "moisture"] {
        assert_eq!(targets[name]["id"], "target_numeric");
        assert_eq!(targets[name]["rank"], 1);
    }
    assert!(targets.get("targets").is_none());

    let all = ids(0..6);
    let full = view(&provider, &all);
    assert_block_matches_record(&provider, full, &observed, &all);

    // The single-target DMD block has no validity mask: it serves only views
    // whose samples all observe the target, and refuses the others.
    let protein = TargetId::new("protein").unwrap();
    let refused = provider
        .target_block(full, &protein)
        .unwrap_err()
        .to_string();
    assert!(
        refused.contains("has no value for sample `s1`"),
        "{refused}"
    );
    let protein_rows = ids((0..6).filter(|row| *row != 1));
    let served = provider
        .target_block(view(&provider, &protein_rows), &protein)
        .unwrap();
    assert_eq!(
        served.values,
        (0..6)
            .filter(|row| *row != 1)
            .map(|row| json!(truth(row, 0)))
            .collect::<Vec<_>>()
    );

    // A view in a different requested order still joins by identity.
    let reordered = ids([5, 2, 4, 0]);
    let subset = view(&provider, &reordered);
    assert_block_matches_record(&provider, subset, &observed, &reordered);
}

#[test]
fn three_targets_keep_training_scope_and_unobserved_targets_declared() {
    // 10 samples: s0..s6 train, s7..s9 test.
    //   ash      observed on every train row except s2/s5, and on s8;
    //   fat      observed only on test rows (no observed training row);
    //   fibre    never observed.
    let partitions: Vec<&str> = (0..10)
        .map(|row| if row < 7 { "train" } else { "test" })
        .collect();
    let observed: Vec<Vec<bool>> = (0..10)
        .map(|row| {
            vec![
                if row < 7 {
                    row != 2 && row != 5
                } else {
                    row == 8
                },
                row >= 7,
                false,
            ]
        })
        .collect();
    let value = record(
        &["ash", "fat", "fibre"],
        &observed,
        &partitions,
        "regression",
    );
    let provider = provider(&value);
    assert_eq!(
        provider
            .target_ids()
            .iter()
            .map(TargetId::as_str)
            .collect::<Vec<_>>(),
        ["ash", "fat", "fibre"]
    );

    let train = ids(0..7);
    let train_view = view(&provider, &train);
    assert_block_matches_record(&provider, train_view, &observed, &train);
    let block = provider.multi_target_block(train_view).unwrap();
    assert!(
        block.validity_masks[1].iter().all(|valid| !valid),
        "fat has no observed training row; DAG-ML refuses fitting it"
    );
    assert!(block.validity_masks[2].iter().all(|valid| !valid));

    let all = ids(0..10);
    assert_block_matches_record(&provider, view(&provider, &all), &observed, &all);

    let fibre = TargetId::new("fibre").unwrap();
    let refused = provider
        .target_block(train_view, &fibre)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("has no observed value"), "{refused}");
}

#[test]
fn complete_multi_target_record_keeps_its_matrix_target() {
    let observed = vec![vec![true, true]; 6];
    let value = record(
        &["protein", "moisture"],
        &observed,
        &["train"; 6],
        "regression",
    );
    let provider = provider(&value);
    assert_eq!(provider.target_names(), ["protein", "moisture"]);
    assert_eq!(provider.target_ids(), [TargetId::new("targets").unwrap()]);
    let targets = &provider.envelope().metadata["io.dataset_schema"]["targets"];
    assert_eq!(targets["targets"]["id"], "target_numeric_matrix");
    assert_eq!(targets["targets"]["rank"], 2);

    let all = ids(0..6);
    let full = view(&provider, &all);
    let block = provider
        .target_block(full, &TargetId::new("targets").unwrap())
        .unwrap();
    for (sample_id, value) in block.sample_ids.iter().zip(&block.values) {
        let row = row_of(sample_id);
        assert_eq!(value, &json!([truth(row, 0), truth(row, 1)]));
    }
    let refused = provider.multi_target_block(full).unwrap_err().to_string();
    assert!(refused.contains("matrix target"), "{refused}");
}

#[test]
fn partial_records_offer_no_dense_multi_output_target() {
    // Several targets at once need per-target observed rows. IO offers them
    // only as per-target validity; it never serves a dense matrix row that
    // would hide a missing cell.
    let observed: Vec<Vec<bool>> = (0..6).map(|row| vec![row != 1, row != 4]).collect();
    let value = record(
        &["protein", "moisture"],
        &observed,
        &["train"; 6],
        "regression",
    );
    let provider = provider(&value);
    let full = view(&provider, &ids(0..6));
    let refused = provider
        .target_block(full, &TargetId::new("targets").unwrap())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("unknown target id"), "{refused}");
}

#[test]
fn partial_classification_targets_keep_integer_labels() {
    let observed: Vec<Vec<bool>> = (0..6).map(|row| vec![row != 0, row != 3]).collect();
    let value = record(
        &["grade", "class"],
        &observed,
        &["train"; 6],
        "classification",
    );
    let provider = provider(&value);
    let targets = &provider.envelope().metadata["io.dataset_schema"]["targets"];
    assert_eq!(targets["grade"]["id"], "target_categorical");
    assert_eq!(targets["class"]["id"], "target_categorical");
    let block = provider
        .multi_target_block(view(&provider, &ids(0..6)))
        .unwrap();
    for (position, sample_id) in block.sample_ids.iter().enumerate() {
        let row = row_of(sample_id);
        for (target, seen) in observed[row].iter().copied().enumerate() {
            let expected = if seen {
                json!(((row + target) % 3) as i64)
            } else {
                Value::Null
            };
            assert_eq!(block.values[target][position], expected);
            assert_eq!(block.validity_masks[target][position], seen);
        }
    }
}

#[test]
fn wide_sparse_record_shaped_like_a_real_49_target_dataset() {
    // 800 samples and 49 targets: target 0 is complete, targets 1..48 miss
    // between 3% and 90% of their rows, each on its own deterministic rows.
    const ROWS: usize = 800;
    const TARGETS: usize = 49;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let missing_rate = |target: usize| 0.03 + 0.87 * (target - 1) as f64 / (TARGETS - 2) as f64;
    let observed: Vec<Vec<bool>> = (0..ROWS)
        .map(|_| {
            (0..TARGETS)
                .map(|target| {
                    target == 0 || (next() % 10_000) as f64 / 10_000.0 >= missing_rate(target)
                })
                .collect()
        })
        .collect();
    let names: Vec<String> = (0..TARGETS)
        .map(|target| format!("analyte_{target:02}"))
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let partitions: Vec<&str> = (0..ROWS)
        .map(|row| if row % 5 == 4 { "test" } else { "train" })
        .collect();
    let value = record(&names, &observed, &partitions, "regression");
    let provider = provider(&value);
    assert_eq!(provider.target_ids().len(), TARGETS);
    assert_eq!(provider.target_names(), names.as_slice());

    let train = ids((0..ROWS).filter(|row| row % 5 != 4));
    let train_view = view(&provider, &train);
    assert_block_matches_record(&provider, train_view, &observed, &train);
    let block = provider.multi_target_block(train_view).unwrap();
    let observed_counts = block
        .validity_masks
        .iter()
        .map(|mask| mask.iter().filter(|valid| **valid).count())
        .collect::<Vec<_>>();
    assert_eq!(observed_counts[0], train.len(), "the complete target");
    let sparsest = observed_counts[TARGETS - 1] as f64 / train.len() as f64;
    let densest = observed_counts[1] as f64 / train.len() as f64;
    assert!(
        (0.05..0.15).contains(&sparsest),
        "~10% observed, got {sparsest}"
    );
    assert!(
        (0.94..1.0).contains(&densest),
        "~97% observed, got {densest}"
    );
    // The complete target is still served by the single-target block.
    let complete = provider
        .target_block(train_view, &TargetId::new("analyte_00").unwrap())
        .unwrap();
    assert_eq!(complete.sample_ids.len(), train.len());
}
