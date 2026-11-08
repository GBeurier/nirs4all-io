//! Input-level NA provenance is captured before merges and role policies.
use nirs4all_io_core::materialize::{
    assemble_in_memory, Cell, Column, DatasetPackage, Frame, InMemorySource, SourcePayload,
};
use nirs4all_io_core::spec::DatasetSpec;
use serde_json::{json, Value};

fn input(name: &str, data: &str) -> InMemorySource {
    InMemorySource {
        name: name.into(),
        payload: SourcePayload::Bytes(data.as_bytes().to_vec()),
    }
}
fn run(spec: Value, sources: &[InMemorySource]) -> nirs4all_io_core::materialize::AssembledDataset {
    assemble_in_memory(
        &DatasetSpec::from_value(&spec).unwrap(),
        sources,
        &Default::default(),
        None,
    )
    .unwrap()
}
fn report(count: usize, samples: usize, features: usize) -> Value {
    json!({"strategy":"ignore","na_detected":true,"na_count":count,"na_samples":samples,"na_features":features,"removed_samples":[],"removed_features":[],"fill_method":null,"na_preserved":true})
}

#[test]
fn input_reports_count_original_cells_in_input_order_before_key_merge() {
    let spec = json!({"task_type":"regression","sources":[{"id":"table","input":["a.csv","b.csv"],"merge":"by_key","key":"id","columns":{"features":["x","z"],"targets":"y","metadata":["id","note"]}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":",","na":{"policy":"ignore"}}});
    let assembled = run(
        spec,
        &[
            input(
                "a.csv",
                "id,x,y,note\ns1,1,NA,first\ns2,NA,2,NA\ns3,3,3,last\n",
            ),
            input("b.csv", "id,z\ns3,NA\ns1,4\ns2,NA\n"),
        ],
    );
    assert_eq!(
        assembled.audits,
        vec![
            json!({"source":"table","na":report(3,2,3)}),
            json!({"source":"table","na":report(2,2,1)}),
            json!({"source":"table","merge":"by_key","warnings":[]})
        ]
    );
    let block = &assembled.blocks["train"];
    assert_eq!(block.x[0].validity, [true, true, false, false, true, false]);
    assert_eq!(block.y.as_ref().unwrap().validity, [false, true, true]);
    let package = DatasetPackage::from_assembled(&assembled).unwrap();
    assert_eq!(package.to_assembled().audits, assembled.audits);
    assert_eq!(
        package.to_full_value().unwrap()["audits"],
        json!(assembled.audits)
    );
    assert_eq!(
        assembled.to_full_value().unwrap()["audits"],
        json!(assembled.audits)
    );
}

#[test]
fn union_missing_cells_are_not_reported_as_missing_in_either_input() {
    let spec = json!({"sources":[{"id":"table","input":["a.csv","b.csv"],"merge":"concat_samples","columns":{"features":"rest"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}});
    let assembled = run(
        spec,
        &[input("a.csv", "x\n1\n2\n"), input("b.csv", "z\n3\n4\n")],
    );
    assert_eq!(
        assembled.audits,
        vec![
            json!({"source":"table","merge":"concat_samples","warnings":["concat_samples schema-union: 2 column(s) not shared across files were null-filled"]})
        ]
    );
}

#[test]
fn decoded_frames_report_pre_role_missingness_and_preserve_zero() {
    let frame = Frame::from_columns(
        vec![
            Column::from_cells("x", vec![Cell::Float(-0.0), Cell::Na]),
            Column::from_cells("note", vec![Cell::Str("ok".into()), Cell::Na]),
        ],
        "text",
    );
    let spec = json!({"sources":[{"id":"table","input":"frame","columns":{"features":"x","metadata":"note"}}],"params":{"na":{"policy":"ignore"}}});
    let assembled = run(
        spec,
        &[InMemorySource {
            name: "frame".into(),
            payload: SourcePayload::Frame(frame),
        }],
    );
    assert_eq!(
        assembled.audits,
        vec![json!({"source":"table","na":report(2,1,2)})]
    );
    assert_eq!(
        assembled.blocks["train"].x[0].data[0].to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(assembled.blocks["train"].x[0].validity, [true, false]);
}

#[test]
fn configured_replacement_is_applied_after_the_actual_ignored_reader_report() {
    let spec = json!({"sources":[{"id":"x","input":"x.csv","role":"features"}],"params":{"delimiter":",","na":{"policy":"replace","fill":{"method":"value","fill_value":7.0}}}});
    let assembled = run(spec, &[input("x.csv", "x\n1\nNA\n")]);
    assert_eq!(
        assembled.audits,
        vec![json!({"source":"x","na":report(1,1,1)})]
    );
    assert_eq!(assembled.blocks["train"].x[0].data, [1.0, 7.0]);
    assert_eq!(assembled.blocks["train"].x[0].validity, [true, true]);
}

#[test]
fn decoded_records_have_input_reports_before_role_handling() {
    let records = vec![
        json!({"signals":{"raw":{"values":[1.0],"axis":{"unit":"nm","values":[900.0]}}}}),
        json!({"signals":{"raw":{"values":[null],"axis":{"unit":"nm","values":[900.0]}}}}),
    ];
    let spec = json!({"sources":[{"id":"x","input":"records","role":"features"}],"params":{"na":{"policy":"ignore"}}});
    let assembled = run(
        spec,
        &[InMemorySource {
            name: "records".into(),
            payload: SourcePayload::Records(records),
        }],
    );
    assert_eq!(
        assembled.audits,
        vec![json!({"source":"x","na":report(1,1,1)})]
    );
    assert_eq!(assembled.blocks["train"].x[0].validity, [true, false]);
}
