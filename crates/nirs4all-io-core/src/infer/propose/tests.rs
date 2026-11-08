// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Tests for the iterative proposal/validation engine.

use super::*;
use crate::infer::memory::infer_browser_dataset;
use crate::materialize::{assemble_in_memory, AssembledDataset, InMemorySource, SourcePayload};
use crate::spec::DatasetSpec;
use std::collections::HashMap;

fn mem_sources(files: &[NamedBytes]) -> Vec<InMemorySource> {
    files
        .iter()
        .map(|f| InMemorySource {
            name: f.name.clone(),
            payload: SourcePayload::Bytes(f.bytes.clone()),
        })
        .collect()
}

fn assemble(spec: &DatasetSpec, files: &[NamedBytes]) -> AssembledDataset {
    assemble_in_memory(spec, &mem_sources(files), &HashMap::new(), None).expect("spec materializes")
}

fn named(name: &str, text: &str) -> NamedBytes {
    NamedBytes {
        name: name.into(),
        bytes: text.as_bytes().to_vec(),
    }
}

#[test]
fn provisional_headerless_source_keeps_twenty_rows_and_warns() {
    let mut text = String::new();
    for row in 0..20 {
        text.push_str(&format!("0.{:07}", 465041 + row * 10000));
        for _ in 1..700 {
            text.push_str(";0.2");
        }
        text.push('\n');
    }
    let files = vec![
        named("data.csv", &text),
        named("auxiliary.csv", "batch;value\na;1\nb;2\n"),
    ];
    let auto = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    assert!(auto
        .plan
        .warnings
        .iter()
        .any(|w| w.contains("data.csv") && w.contains("header_ambiguous")));
    let choices = InferOptions {
        file_params: [("data.csv".into(), json!({"has_header": false}))]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    let result = propose_browser_dataset(&files, &[], &[], &choices).unwrap();
    let value = result.spec.to_value();
    let source = value["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["input"] == "data.csv")
        .unwrap();
    assert_eq!(source["params"]["has_header"], json!(false));
    assert_eq!(
        result
            .plan
            .warnings
            .iter()
            .filter(|w| w.contains("data.csv") && w.contains("has_header=false"))
            .count(),
        0
    );
    // Materialize the actual proposed source independently of unconfirmed joins.
    let spec =
        DatasetSpec::from_value(&json!({"name": "provisional", "sources": [source]})).unwrap();
    let data = assemble(&spec, &files);
    let block = &data.blocks["train"];
    assert_eq!(block.n_samples, 20);
    assert_eq!(block.x[0].n_rows, 20);
    assert_eq!(block.x[0].n_cols, 699);
    assert_eq!(block.x[0].data[0], 0.0465041);
    assert_eq!(block.x[0].data[19 * 699], 0.0655041);
    assert_eq!(block.y.as_ref().unwrap().data, vec![0.2; 20]);
}

fn lock(kind: &str, target: &str, value: Value) -> ConfirmedLock {
    ConfirmedLock {
        kind: kind.into(),
        target: target.into(),
        value,
        status: None,
    }
}

fn of_kind<'a>(r: &'a ProposalResult, kind: &str) -> Vec<&'a Proposal> {
    r.proposals
        .iter()
        .filter(|p| p.kind.as_str() == kind)
        .collect()
}

fn source_id_for(r: &ProposalResult, input: &str) -> String {
    r.spec.to_value()["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["input"] == json!(input))
        .expect("source for input")["id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn provisional_sources_created_for_unmatched_files() {
    // Two files matching no convention -> plain inference yields `sources: []`;
    // the proposal layer must synthesise one source per file (Codex #2).
    let files = vec![
        named("alpha.csv", "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\n"),
        named("beta.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n"),
    ];
    let plain = infer_browser_dataset(&files, &[], &Default::default()).unwrap();
    let plain_sources = plain.resolved_spec.unwrap().to_value()["sources"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(plain_sources, 0, "plain inference drops unmatched files");

    let r = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let inputs: Vec<String> = r.spec.to_value()["sources"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["input"].as_str().map(str::to_string))
        .collect();
    assert_eq!(inputs.len(), 2, "one provisional source per file");
    assert!(inputs.contains(&"alpha.csv".to_string()));
    assert!(inputs.contains(&"beta.csv".to_string()));
}

#[test]
fn equal_count_join_pairing_when_shared_unique_id() {
    let files = vec![
        named(
            "Xcal.csv",
            "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\ns3;0.3;0.4\n",
        ),
        named("Ycal.csv", "sample_id;protein\ns1;11.0\ns2;12.0\ns3;13.0\n"),
    ];
    let r = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let pairing = of_kind(&r, "pairing");
    assert!(!pairing.is_empty(), "a pairing decision is proposed");
    let p = pairing[0];
    assert_eq!(p.value["mode"], json!("join_id"));
    assert_eq!(p.value["on"], json!("sample_id"));
}

#[test]
fn equal_count_row_order_low_score_when_no_shared_id() {
    // Convention names drive the roles; neither file carries an id column.
    let files = vec![
        named("Xcal.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n0.3;0.4\n"),
        named("Ycal.csv", "protein\n11.0\n12.0\n13.0\n"),
    ];
    let r = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let p = of_kind(&r, "pairing");
    assert!(!p.is_empty(), "row-count pairing is proposed");
    let row = p
        .iter()
        .find(|p| p.value["mode"] == json!("row_order"))
        .expect("row_order mode");
    assert!(row.ambiguous, "row-count-only pairing is ambiguous");
    assert!(
        row.score < 0.6,
        "weak evidence => low score, got {}",
        row.score
    );
}

#[test]
fn confirmed_pairing_join_sets_source_join_and_suppresses_proposal() {
    let files = vec![
        named(
            "Xcal.csv",
            "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\ns3;0.3;0.4\n",
        ),
        named("Ycal.csv", "sample_id;protein\ns1;11.0\ns2;12.0\ns3;13.0\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let target = of_kind(&r0, "pairing")[0].target.clone();

    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock(
            "pairing",
            &target,
            json!({"mode": "join_id", "on": "sample_id"}),
        )],
        &Default::default(),
    )
    .unwrap();

    let spec = r.spec.to_value();
    let has_join = spec["sources"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s.get("join").is_some());
    assert!(has_join, "join_id lock writes a SourceSpec.join");

    let open = r
        .proposals
        .iter()
        .filter(|p| {
            p.kind == ProposalKind::Pairing
                && p.target == target
                && p.status == ProposalStatus::Proposed
        })
        .count();
    assert_eq!(open, 0, "the confirmed pairing proposal is suppressed");
}

#[test]
fn confirmed_role_overrides_inferred_role() {
    let files = vec![named("alpha.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n")];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let id = source_id_for(&r0, "alpha.csv");

    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock("role", &id, json!("metadata"))],
        &Default::default(),
    )
    .unwrap();
    let spec = r.spec.to_value();
    let src = &spec["sources"][0];
    assert_eq!(src["role"], json!("metadata"));
    let open_role = of_kind(&r, "role")
        .iter()
        .filter(|p| p.target == id && p.status == ProposalStatus::Proposed)
        .count();
    assert_eq!(open_role, 0, "role proposal suppressed once confirmed");
}

#[test]
fn confirmed_identity_sets_sample_index() {
    let files = vec![named(
        "alpha.csv",
        "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\n",
    )];
    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock(
            "identity",
            "dataset",
            json!({"by": "id", "key": "sample_id"}),
        )],
        &Default::default(),
    )
    .unwrap();
    let spec = r.spec.to_value();
    assert_eq!(spec["sample_index"]["by"], json!("id"));
    assert_eq!(spec["sample_index"]["key"], json!("sample_id"));
}

#[test]
fn identity_lock_keys_only_proven_source_columns_and_preserves_materialized_ids() {
    let files = vec![
        named("Xcal.csv", "1000;1005\n0.1;0.2\n0.3;0.4\n"),
        named("Ycal.csv", "protein\n11\n12\n"),
        named("Mcal.csv", "sample_id;site\ns1;A\ns2;B\n"),
    ];
    let initial = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let x = source_id_for(&initial, "Xcal.csv");
    let y = source_id_for(&initial, "Ycal.csv");
    let m = source_id_for(&initial, "Mcal.csv");
    for identity in [json!("sample_id"), json!({"by": "id", "key": "sample_id"})] {
        let result = propose_browser_dataset(
            &files,
            &[],
            &[
                lock("role", &x, json!("features")),
                lock("role", &y, json!("targets")),
                lock("role", &m, json!("metadata")),
                lock("identity", "dataset", identity),
                lock("pairing", &format!("{x}|{y}"), json!({"mode": "row_order"})),
                lock("pairing", &format!("{x}|{m}"), json!({"mode": "row_order"})),
            ],
            &Default::default(),
        )
        .unwrap();
        let spec = result.spec.to_value();
        assert_eq!(spec["sample_index"]["key"], "sample_id");
        let feature = spec["sources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|source| source["id"].as_str() == Some(x.as_str()))
            .unwrap();
        assert!(feature.get("key").is_none_or(Value::is_null));
        assert!(of_kind(&result, "identity")
            .iter()
            .any(|proposal| proposal.status == ProposalStatus::Confirmed));
        let assembled = assemble(&result.spec, &files);
        let block = &assembled.blocks["train"];
        assert_eq!(
            block.metadata.as_ref().unwrap().str_column("sample_id"),
            ["s1", "s2"]
        );
        assert_eq!(block.x[0].data, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(block.y.as_ref().unwrap().data, [11.0, 12.0]);
    }

    // A real feature-table identity remains keyed and materializes unchanged.
    let keyed = vec![named("alpha.csv", "sample_id;1000\ns1;0.1\ns2;0.3\n")];
    let result = propose_browser_dataset(
        &keyed,
        &[],
        &[lock("identity", "dataset", json!("sample_id"))],
        &Default::default(),
    )
    .unwrap();
    assert_eq!(result.spec.to_value()["sources"][0]["key"], "sample_id");
    assert_eq!(
        assemble(&result.spec, &keyed).blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .str_column("sample_id"),
        ["s1", "s2"]
    );

    // Both orders must inspect the id-free member, not only the first table.
    let mut inputs = keyed.clone();
    inputs.push(named("beta.csv", "1000\n0.2\n0.4\n"));
    let by_name = inputs
        .iter()
        .map(|file| (file.name.clone(), file))
        .collect();
    for (names, expected) in [
        (json!(["alpha.csv", "beta.csv"]), Value::Null),
        (json!(["beta.csv", "alpha.csv"]), Value::Null),
        (json!(["alpha.csv", "alpha.csv"]), json!("sample_id")),
    ] {
        let mut spec = json!({"sources": [{"id": "x", "role": "features", "input": names}]});
        assert!(matches!(
            apply_identity_lock(
                &mut spec,
                &lock("identity", "dataset", json!("sample_id")),
                &by_name,
                &[],
                &[],
            ),
            LockOutcome::Applied
        ));
        assert_eq!(spec["sources"][0]["key"], expected);
    }
    let mut missing =
        json!({"sources": [{"id": "x", "role": "features", "input": ["alpha.csv", "absent.csv"]}]});
    let before = missing.clone();
    assert!(matches!(
        apply_identity_lock(
            &mut missing,
            &lock("identity", "dataset", json!("sample_id")),
            &by_name,
            &[],
            &[],
        ),
        LockOutcome::Unresolved(message) if message.contains("absent.csv")
    ));
    assert_eq!(
        missing, before,
        "missing facts refuse without partial mutation"
    );

    // Existing decoded-record inference supplies bare metadata names, not
    // dotted diagnostic hints. Its supported executable spec must stay intact.
    let records = vec![DecodedRecordSet {
        source: "spectra.spc".into(),
        format: None,
        records: vec![
            json!({"signals": {"nir": {"values": [0.1, 0.2], "axis": {"values": [1000, 1005], "unit": "nm"}}}, "targets": {"protein": 11}, "metadata": {"sample_id": "s1", "row_index": 0}}),
            json!({"signals": {"nir": {"values": [0.3, 0.4], "axis": {"values": [1000, 1005], "unit": "nm"}}}, "targets": {"protein": 12}, "metadata": {"sample_id": "s2", "row_index": 1}}),
        ],
    }];
    let original = propose_browser_dataset(&[], &records, &[], &Default::default()).unwrap();
    let confirmed = propose_browser_dataset(
        &[],
        &records,
        &[lock("identity", "dataset", json!("sample_id"))],
        &Default::default(),
    )
    .unwrap();
    assert_eq!(confirmed.spec.to_value(), original.spec.to_value());
    let payload = vec![InMemorySource {
        name: records[0].source.clone(),
        payload: SourcePayload::Records(records[0].records.clone()),
    }];
    let assembled = assemble_in_memory(&confirmed.spec, &payload, &HashMap::new(), None).unwrap();
    assert_eq!(
        assembled.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .str_column("sample_id"),
        ["s1", "s2"]
    );
    assert_eq!(assembled.blocks["train"].x[0].data, [0.1, 0.2, 0.3, 0.4]);

    // The main browser proposes with both representations, then materializes
    // vendor Records only. Opaque raw bytes must never supply CSV column facts.
    let raw = vec![NamedBytes {
        name: records[0].source.clone(),
        bytes: vec![0xff, 0x00, 0x1a, 0x01],
    }];
    let paired = propose_browser_dataset(
        &raw,
        &records,
        &[lock("identity", "dataset", json!("sample_id"))],
        &Default::default(),
    )
    .unwrap();
    assert!(of_kind(&paired, "identity")
        .iter()
        .any(|proposal| proposal.status == ProposalStatus::Confirmed));
    assert_eq!(paired.spec.to_value()["sources"][0]["key"], "sample_id");
    let assembled = assemble_in_memory(&paired.spec, &payload, &HashMap::new(), None).unwrap();
    assert_eq!(
        assembled.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .str_column("sample_id"),
        ["s1", "s2"]
    );
    assert_eq!(assembled.blocks["train"].x[0].data, [0.1, 0.2, 0.3, 0.4]);

    // Delimited-text is the opposite selection: decoded-only target names
    // cannot establish a column in the authoritative raw table.
    let mut text_records = records.clone();
    text_records[0].source = keyed[0].name.clone();
    text_records[0].format = Some("delimited-text".into());
    let text_plan = infer_browser_dataset(&keyed, &text_records, &Default::default()).unwrap();
    let text_by_name = keyed.iter().map(|file| (file.name.clone(), file)).collect();
    let mut text_spec = json!({"sources": [{"id": "x", "role": "features", "input": "alpha.csv"}]});
    assert!(matches!(
        apply_identity_lock(
            &mut text_spec,
            &lock("identity", "dataset", json!("protein")),
            &text_by_name,
            &text_plan.columns,
            &text_records,
        ),
        LockOutcome::Applied
    ));
    assert!(text_spec["sources"][0].get("key").is_none());

    // Raw WASM transport appends payloads without precedence; genuinely
    // duplicated assembly inputs still refuse instead of silently winning.
    let mut conflicting = mem_sources(&raw);
    conflicting.extend(payload);
    assert!(
        assemble_in_memory(&paired.spec, &conflicting, &HashMap::new(), None)
            .unwrap_err()
            .message
            .contains("ambiguous")
    );
}

#[test]
fn confirmed_signal_and_task_override_detectors() {
    let files = vec![named(
        "Xcal.csv",
        "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\n",
    )];
    let r = propose_browser_dataset(
        &files,
        &[],
        &[
            lock("signal_type", "dataset", json!("absorbance")),
            lock("task_type", "dataset", json!("regression")),
        ],
        &Default::default(),
    )
    .unwrap();
    let spec = r.spec.to_value();
    assert_eq!(spec["params"]["signal_type"], json!("absorbance"));
    assert_eq!(spec["task_type"], json!("regression"));
}

#[test]
fn no_locks_equals_plain_browser_inference() {
    // Convention-named files create their own sources, so no provisional source
    // is added and the proposal layer is a pure superset (spec unchanged).
    let files = vec![
        named(
            "Xcal.csv",
            "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\ns3;0.3;0.4\n",
        ),
        named("Ycal.csv", "sample_id;protein\ns1;11.0\ns2;12.0\ns3;13.0\n"),
    ];
    let proposed = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let plain = infer_browser_dataset(&files, &[], &Default::default()).unwrap();
    assert_eq!(
        proposed.spec.to_value(),
        plain.resolved_spec.unwrap().to_value()
    );
}

#[test]
fn invalid_lock_surfaces_validation_error() {
    // Re-role the only feature source to metadata -> the spec has no features.
    let files = vec![named("alpha.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n")];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let id = source_id_for(&r0, "alpha.csv");
    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock("role", &id, json!("metadata"))],
        &Default::default(),
    )
    .unwrap();
    assert!(!r.valid, "validation runs and flags the invalid spec");
    assert!(!r.validation_errors.is_empty());
}

#[test]
fn sum_pairing_proposes_concat_samples() {
    let files = vec![
        named("part_a.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n"),
        named("part_b.csv", "1000;1005\n0.3;0.4\n"),
        named("whole_y.csv", "protein\n11\n12\n13\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let whole = source_id_for(&r0, "whole_y.csv");
    // Only once the whole is a target does summing X parts make sense.
    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock("role", &whole, json!("targets"))],
        &Default::default(),
    )
    .unwrap();
    let sum = of_kind(&r, "pairing")
        .into_iter()
        .find(|p| p.value["mode"] == json!("concat_samples"));
    assert!(
        sum.is_some(),
        "feature sources summing to the target are paired"
    );
    let parts = sum.unwrap().value["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2, "two feature parts stack into one X");
}

// --- materialization proofs: each lock yields a spec the assembler accepts --- //

#[test]
fn join_id_lock_materializes_y() {
    let files = vec![
        named(
            "Xcal.csv",
            "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\ns3;0.3;0.4\n",
        ),
        named("Ycal.csv", "sample_id;protein\ns1;11.0\ns2;12.0\ns3;13.0\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let target = of_kind(&r0, "pairing")[0].target.clone();
    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock(
            "pairing",
            &target,
            json!({"mode": "join_id", "on": "sample_id"}),
        )],
        &Default::default(),
    )
    .unwrap();
    let a = assemble(&r.spec, &files);
    let block = a.blocks.get("train").expect("train block");
    assert_eq!(block.n_samples, 3);
    assert!(block.y.is_some(), "join_id lock materializes y");
}

#[test]
fn row_order_lock_materializes_y() {
    let files = vec![
        named("Xcal.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n0.3;0.4\n"),
        named("Ycal.csv", "protein\n11.0\n12.0\n13.0\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let target = of_kind(&r0, "pairing")[0].target.clone();
    let r = propose_browser_dataset(
        &files,
        &[],
        &[lock("pairing", &target, json!({"mode": "row_order"}))],
        &Default::default(),
    )
    .unwrap();
    let a = assemble(&r.spec, &files);
    let block = a.blocks.get("train").expect("train block");
    assert_eq!(block.n_samples, 3);
    assert!(block.y.is_some(), "row_order lock materializes y");
}

#[test]
fn concat_samples_lock_materializes_full_x_and_y() {
    let files = vec![
        named("part_a.csv", "1000;1005\n0.1;0.2\n0.2;0.3\n"),
        named("part_b.csv", "1000;1005\n0.3;0.4\n"),
        named("whole_y.csv", "protein\n11\n12\n13\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let a_id = source_id_for(&r0, "part_a.csv");
    let b_id = source_id_for(&r0, "part_b.csv");
    let y_id = source_id_for(&r0, "whole_y.csv");
    let r = propose_browser_dataset(
        &files,
        &[],
        &[
            lock("role", &y_id, json!("targets")),
            lock(
                "pairing",
                "sum",
                json!({"mode": "concat_samples", "parts": [a_id, b_id], "whole": y_id}),
            ),
        ],
        &Default::default(),
    )
    .unwrap();
    // The two feature files folded into one multi-input concat source; y remains.
    let sources = r.spec.to_value()["sources"].as_array().unwrap().clone();
    assert_eq!(sources.len(), 2, "part_b folded into part_a; y remains");
    let a = assemble(&r.spec, &files);
    let block = a.blocks.get("train").expect("train block");
    assert_eq!(block.n_samples, 3, "2 + 1 stacked rows align with y(3)");
    assert!(block.y.is_some(), "concat_samples lock materializes y");
}

#[test]
fn concat_samples_keeps_shared_id_out_of_features() {
    // Both X parts carry `sample_id`; folding must preserve the key so the id
    // stays out of X (else it would be coerced to a NaN feature column).
    let files = vec![
        named(
            "part_a.csv",
            "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\n",
        ),
        named(
            "part_b.csv",
            "sample_id;1000;1005\ns3;0.3;0.4\ns4;0.4;0.5\n",
        ),
        named("whole_y.csv", "protein\n11\n12\n13\n14\n"),
    ];
    let r0 = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let a_id = source_id_for(&r0, "part_a.csv");
    let b_id = source_id_for(&r0, "part_b.csv");
    let y_id = source_id_for(&r0, "whole_y.csv");
    let r = propose_browser_dataset(
        &files,
        &[],
        &[
            lock("role", &y_id, json!("targets")),
            lock(
                "pairing",
                "sum",
                json!({"mode": "concat_samples", "parts": [a_id, b_id], "whole": y_id}),
            ),
        ],
        &Default::default(),
    )
    .unwrap();
    let a = assemble(&r.spec, &files);
    let block = a.blocks.get("train").expect("train block");
    assert_eq!(block.n_samples, 4, "2 + 2 stacked rows align with y(4)");
    let headers: Vec<String> = block.feature_headers.iter().flatten().cloned().collect();
    assert!(
        !headers.contains(&"sample_id".to_string()),
        "the shared id stays out of X, got {headers:?}"
    );
    assert!(block.y.is_some());
}

#[test]
fn proposal_ids_are_stable_across_reinference() {
    let files = vec![
        named("alpha.csv", "sample_id;1000;1005\ns1;0.1;0.2\ns2;0.2;0.3\n"),
        named("beta.csv", "sample_id;protein\ns1;11.0\ns2;12.0\n"),
    ];
    let a = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let b = propose_browser_dataset(&files, &[], &[], &Default::default()).unwrap();
    let ids_a: Vec<&str> = a.proposals.iter().map(|p| p.id.as_str()).collect();
    let ids_b: Vec<&str> = b.proposals.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids_a, ids_b);
}
