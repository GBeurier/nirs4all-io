// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Strict explicit Rows capability; Formats Records remains a separate contract.
use std::collections::HashMap;

use nirs4all_io_core::materialize::{
    assemble_in_memory_with_storage_and_tabular_limits, AssembledDataset, Cell, DatasetPackage,
    InMemorySource, NumericStorage, NumericStorageOptions, SourcePayload, TabularReadLimits,
};
use nirs4all_io_core::spec::{DatasetSpec, SpecError};
use serde_json::{json, Value};

fn features() -> Value {
    json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"table","input":"table","role":"features"}]})
}

fn run(
    spec: &Value,
    payload: SourcePayload,
    limits: Option<TabularReadLimits>,
) -> Result<AssembledDataset, SpecError> {
    assemble_in_memory_with_storage_and_tabular_limits(
        &DatasetSpec::from_value(spec).unwrap(),
        &[InMemorySource {
            name: "table".into(),
            payload,
        }],
        &HashMap::new(),
        None,
        limits,
        &NumericStorageOptions::default(),
    )
}

fn reject(rows: Vec<Value>, message: &str) {
    let error = run(&features(), SourcePayload::Rows(rows), None).unwrap_err();
    assert!(error.message.contains(message), "{}", error.message);
}

#[test]
fn first_seen_order_binds_values_by_key_and_can_be_declared() {
    // This literal is also a preserve_order feature guard: sorted JSON maps fail.
    let rows: Vec<Value> = serde_json::from_str(r#"[{"b":2,"a":1},{"a":3,"b":4}]"#).unwrap();
    let a = run(&features(), SourcePayload::Rows(rows.clone()), None).unwrap();
    assert_eq!(a.blocks["train"].feature_headers[0], ["b", "a"]);
    assert_eq!(a.blocks["train"].x[0].data, [2.0, 1.0, 4.0, 3.0]);
    let reordered: Vec<Value> = serde_json::from_str(r#"[{"a":1,"b":2},{"a":3,"b":4}]"#).unwrap();
    assert_eq!(rows, reordered);
    let reordered = run(&features(), SourcePayload::Rows(reordered), None).unwrap();
    assert_ne!(
        DatasetPackage::from_assembled(&a).unwrap().manifest().root,
        DatasetPackage::from_assembled(&reordered)
            .unwrap()
            .manifest()
            .root
    );
    let mut spec = features();
    spec["sources"][0]["params"] = json!({"format":{"columns":["a","b"]}});
    let ordered = run(&spec, SourcePayload::Rows(rows), None).unwrap();
    assert_eq!(ordered.blocks["train"].feature_headers[0], ["a", "b"]);
    assert_eq!(ordered.blocks["train"].x[0].data, [1.0, 2.0, 3.0, 4.0]);
    let axes: Vec<Value> = serde_json::from_str(r#"[{"900":1,"1000":2}]"#).unwrap();
    let a = run(&features(), SourcePayload::Rows(axes), None).unwrap();
    assert_eq!(a.blocks["train"].feature_headers[0], ["900", "1000"]);
}

#[test]
fn scalar_metadata_and_missing_targets_remain_literal() {
    let spec = json!({"params":{"na":{"policy":"ignore"}},"sources":[{
        "id":"table","input":"table","role":"mixed","params":{"header_unit":"nm"},
        "columns":{"features":"x","targets":"y","metadata":["flag","note","row_index","id","absent"]}
    }]});
    let rows = vec![
        json!({"x":0.1,"y":1.5,"flag":true,"note":"NA","row_index":7,"id":i64::MAX,"absent":null}),
        json!({"x":0.2,"y":null,"flag":null,"note":"001","row_index":8,"id":null}),
    ];
    let a = run(&spec, SourcePayload::Rows(rows), None).unwrap();
    let b = &a.blocks["train"];
    assert_eq!(b.header_units, ["nm"]);
    assert_eq!(b.x[0].data, [0.1, 0.2]);
    assert_eq!(b.y.as_ref().unwrap().data[0].to_bits(), 1.5_f64.to_bits());
    assert!(b.y.as_ref().unwrap().validity[0]);
    assert!(b.y.as_ref().unwrap().data[1].is_nan());
    assert!(!b.y.as_ref().unwrap().validity[1]);
    let meta = b.metadata.as_ref().unwrap();
    assert_eq!(meta.column("flag").unwrap().values[0], Cell::Bool(true));
    assert_eq!(
        meta.column("note").unwrap().values[0],
        Cell::Str("NA".into())
    );
    assert_eq!(
        meta.column("note").unwrap().values[1],
        Cell::Str("001".into())
    );
    assert_eq!(meta.column("row_index").unwrap().values[0], Cell::Int(7));
    assert_eq!(meta.column("id").unwrap().values[0], Cell::Int(i64::MAX));
    assert!(meta.column("absent").unwrap().values.iter().all(|cell| {
        matches!(cell, Cell::Na) || matches!(cell, Cell::Float(value) if value.is_nan())
    }));
    let view = DatasetPackage::from_assembled(&a).unwrap().to_assembled();
    assert_eq!(
        view.blocks["train"].y.as_ref().unwrap().data[1].to_bits(),
        0
    );
    assert!(!view.blocks["train"].y.as_ref().unwrap().validity[1]);
    assert_eq!(
        view.blocks["train"].y.as_ref().unwrap().data[0].to_bits(),
        1.5_f64.to_bits()
    );
    assert!(view.blocks["train"].y.as_ref().unwrap().validity[0]);
}

#[test]
fn strings_mix_with_other_scalars_without_becoming_numeric_tokens() {
    let spec = json!({"sources":[{"id":"table","input":"table","columns":{"features":"x","metadata":"note"}}]});
    let a = run(
        &spec,
        SourcePayload::Rows(vec![json!({"x":1,"note":true}), json!({"x":2,"note":"NA"})]),
        None,
    )
    .unwrap();
    assert_eq!(
        a.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .column("note")
            .unwrap()
            .values,
        [Cell::Bool(true), Cell::Str("NA".into())]
    );
}

#[test]
fn row_shape_and_scalar_refusals_are_explicit() {
    reject(vec![], "at least one row");
    reject(vec![json!({}), json!({})], "at least one column");
    reject(vec![json!(1)], "row 0 must be an object");
    reject(vec![json!({"":1})], "empty column");
    reject(vec![json!({"x":[1,2]})], "row 0, column 'x': nested");
    reject(vec![json!({"x":{"value":1}})], "row 0, column 'x': nested");
    reject(vec![json!({"signals":{"s":{"values":[1]}}})], "nested");
    for integer in [9_223_372_036_854_775_808_u64, u64::MAX] {
        reject(vec![json!({"id":integer})], "column 'id': expected an i64");
    }
    reject(vec![json!({"a":true}), json!({"a":1.5})], "bool and number");
    reject(vec![json!({"a":1}), json!({"a":false})], "bool and number");
}

#[test]
fn declared_decimal_is_used_instead_of_silent_target_codes() {
    let mut spec = json!({"sources":[{"id":"table","input":"table","columns":{"features":"x","targets":"y"}}]});
    let rows = vec![json!({"x":0.1,"y":"12,5"}), json!({"x":0.2,"y":"13,1"})];
    let error = run(&spec, SourcePayload::Rows(rows.clone()), None).unwrap_err();
    assert!(
        error.message.contains("decimal_separator"),
        "{}",
        error.message
    );
    spec["sources"][0]["params"] = json!({"decimal_separator":","});
    let a = run(&spec, SourcePayload::Rows(rows), None).unwrap();
    assert_eq!(a.blocks["train"].y.as_ref().unwrap().data, [12.5, 13.1]);
    assert!(a.blocks["train"].y_categorical.is_empty());
}

#[test]
fn selection_header_unit_and_owner_na_policy_apply() {
    let spec = json!({"sources":[{"id":"table","input":"table","role":"features","params":{
        "format":{"columns":["x"]},"header_unit":"nm",
        "na":{"policy":"replace","fill":{"method":"value","fill_value":7}}
    }}]});
    let a = run(
        &spec,
        SourcePayload::Rows(vec![
            json!({"x":null,"discard":5}),
            json!({"x":2,"discard":6}),
        ]),
        None,
    )
    .unwrap();
    assert_eq!(a.blocks["train"].feature_headers[0], ["x"]);
    assert_eq!(a.blocks["train"].header_units, ["nm"]);
    assert_eq!(a.blocks["train"].x[0].data, [7.0, 2.0]);
}

#[test]
fn i64_metadata_and_keys_are_exact_but_inexact_numeric_f64_is_refused() {
    let big = 9_007_199_254_740_993_i64;
    let spec = DatasetSpec::from_value(&json!({
        "sample_index":{"by":"id","key":"id"},
        "sources":[
            {"id":"x","input":"x","key":"id","columns":{"features":"x","metadata":"id"}},
            {"id":"y","input":"y","role":"targets","key":"id","join":{"to":"x","on":"id","how":"1:1","coverage":"complete"}}
        ]
    }))
    .unwrap();
    let sources = [
        InMemorySource {
            name: "x".into(),
            payload: SourcePayload::Rows(vec![json!({"id":big,"x":1}), json!({"id":big-1,"x":2})]),
        },
        InMemorySource {
            name: "y".into(),
            payload: SourcePayload::Rows(vec![
                json!({"id":big-1,"y":15}),
                json!({"id":big,"y":25}),
            ]),
        },
    ];
    let a = assemble_in_memory_with_storage_and_tabular_limits(
        &spec,
        &sources,
        &HashMap::new(),
        None,
        None,
        &NumericStorageOptions::default(),
    )
    .unwrap();
    assert_eq!(a.blocks["train"].y.as_ref().unwrap().data, [25.0, 15.0]);
    assert_eq!(
        a.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .column("id")
            .unwrap()
            .values,
        [Cell::Int(big), Cell::Int(big - 1)]
    );

    let metadata = json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"table","input":"table","columns":{"features":"x","metadata":"id"}}]});
    let a = run(
        &metadata,
        SourcePayload::Rows(vec![json!({"x":1,"id":big}), json!({"x":2,"id":null})]),
        None,
    )
    .unwrap();
    assert_eq!(
        a.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .column("id")
            .unwrap()
            .values[0],
        Cell::Int(big)
    );
    let error = assemble_in_memory_with_storage_and_tabular_limits(
        &DatasetSpec::from_value(&features()).unwrap(),
        &[InMemorySource {
            name: "table".into(),
            payload: SourcePayload::Rows(vec![json!({"x":big}), json!({"x":null})]),
        }],
        &HashMap::new(),
        None,
        None,
        &NumericStorageOptions {
            features: Some(NumericStorage::Float64),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.message.contains("exact"), "{}", error.message);
}

#[test]
fn limits_precede_rectangular_allocation_and_validate_utf8_bytes() {
    let limits = TabularReadLimits::new(100, 10, 2, 2, 4);
    for (rows, limit, message) in [
        (
            vec![json!({"x":u64::MAX})],
            TabularReadLimits {
                max_rows: 0,
                ..limits
            },
            "rows",
        ),
        (
            vec![json!({"a":1,"b":2})],
            TabularReadLimits {
                max_columns: 1,
                ..limits
            },
            "columns",
        ),
        (
            vec![json!({"a":1}), json!({"b":2}), json!({"later":[]})],
            TabularReadLimits {
                max_rows: 3,
                max_cells: 3,
                ..limits
            },
            "cells",
        ),
        (
            vec![json!({"x":"long"})],
            TabularReadLimits {
                max_field_bytes: 3,
                ..limits
            },
            "field bytes",
        ),
        (
            vec![json!({"long":1})],
            TabularReadLimits {
                max_field_bytes: 3,
                ..limits
            },
            "field bytes",
        ),
        (
            vec![json!({"é":1})],
            TabularReadLimits {
                max_field_bytes: 1,
                ..limits
            },
            "field bytes",
        ),
        (
            vec![json!({"x":1})],
            TabularReadLimits {
                max_record_bytes: 8,
                ..limits
            },
            "record bytes",
        ),
    ] {
        let error = run(&features(), SourcePayload::Rows(rows), Some(limit)).unwrap_err();
        assert!(error.message.contains(message), "{}", error.message);
    }
    let sparse: Vec<Value> = (0..100_000)
        .map(|index| json!({index.to_string():1}))
        .collect();
    let error = run(
        &features(),
        SourcePayload::Rows(sparse),
        Some(TabularReadLimits::new(100, 10, 100_000, 100_000, 150_000)),
    )
    .unwrap_err();
    assert!(error.message.contains("cells"), "{}", error.message);
    // None is intentionally uncapped; the same small valid table is admitted.
    assert!(run(&features(), SourcePayload::Rows(vec![json!({"x":1})]), None).is_ok());
}

#[test]
fn records_reject_plain_rows_and_malformed_sections_without_reinterpretation() {
    for (record, message) in [
        (json!(1), "must be an object"),
        (json!({"x":0.1,"y":1.5}), "SourcePayload::Rows"),
        (json!({"targets":"bad"}), "section 'targets'"),
        (json!({"metadata":[]}), "section 'metadata'"),
        (json!({"signals":true}), "section 'signals'"),
        (
            json!({"signals":{"s":{"values":"bad"}}}),
            "values must be an array",
        ),
    ] {
        let error = run(&features(), SourcePayload::Records(vec![record]), None).unwrap_err();
        assert!(error.message.contains(message), "{}", error.message);
    }
}

#[test]
fn formats_empty_and_missing_signals_and_unknown_keys_keep_their_meaning() {
    for records in [
        vec![],
        vec![json!({"metadata":{},"unknown_extra":[1,2]})],
        vec![json!({"signals":null,"targets":null,"metadata":null})],
        vec![json!({"signals":{"s":{"values":[]}},"unknown_extra":true})],
    ] {
        // Valid empty/missing-signal conversion still reaches the existing
        // assembly check: a sole source with no feature columns is not a dataset.
        let error = run(&features(), SourcePayload::Records(records), None).unwrap_err();
        assert_eq!(error.message, "partition has no feature source");
    }
    let spec = json!({"sources":[{"id":"table","input":"table","columns":{"features":"1000","metadata":"site"}}]});
    let a = run(
        &spec,
        SourcePayload::Records(vec![json!({
            "signals":{"s":{"values":[0.1],"axis":{"values":[1000],"unit":"nm"}}},
            "metadata":{"row_index":0,"site":"A"},"unknown_extra":{"x":1}
        })]),
        None,
    )
    .unwrap();
    assert_eq!(a.blocks["train"].x[0].data, [0.1]);
    assert_eq!(a.blocks["train"].header_units, ["nm"]);
    assert_eq!(
        a.blocks["train"].metadata.as_ref().unwrap().column_names(),
        ["site"]
    );
}
