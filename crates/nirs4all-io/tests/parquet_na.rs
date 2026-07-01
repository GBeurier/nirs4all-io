// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Parquet materialization must decode raw frames first, then apply source params.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use nirs4all_io::core::spec::{normalize_to_spec_dict, DatasetSpec};
use nirs4all_io::materialize::assemble;
use parquet::arrow::arrow_writer::ArrowWriter;
use serde_json::{json, Value};

fn build_spec(spec_json: &Value) -> DatasetSpec {
    DatasetSpec::from_value(&normalize_to_spec_dict(spec_json)).expect("spec parses")
}

fn write_parquet(path: &std::path::Path, batch: RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn parquet_assemble_applies_source_na_policy_without_default_abort() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("1005", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef,
            Arc::new(Float64Array::from(vec![None, Some(3.0)])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-na",
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet",
            "params": {
                "na": {"policy": "replace", "fill": {"method": "value", "fill_value": 7.0}}
            },
        }],
    }));

    let assembled = assemble(&spec, dir.path()).expect("assemble parquet with source NA policy");
    let block = &assembled.blocks["train"];

    assert_eq!(block.feature_headers, vec![vec!["1000", "1005"]]);
    assert_eq!(block.x[0].data, vec![1.0, 7.0, 2.0, 3.0]);
}
