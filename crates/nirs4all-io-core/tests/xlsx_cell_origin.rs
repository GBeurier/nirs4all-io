// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
use nirs4all_io_core::infer::table::{ColDtype, NumericKind};
use nirs4all_io_core::materialize::frame::Cell;
use nirs4all_io_core::materialize::loaders::{read_sheet, Sheet, SheetCell, TabularReadLimits};
use nirs4all_io_core::spec::dataset_spec::LoadingParams;

#[test]
fn integer_header_and_mixed_text_keep_their_origins() {
    let sheets: Vec<Sheet<SheetCell>> = vec![("s".into(), vec![
        vec![SheetCell::IntegerLexeme("+0007".into())],
        vec![SheetCell::Cell(Cell::Str("123".into()))],
        vec![SheetCell::IntegerLexeme("18446744073709551615".into())],
    ])];
    let frame = read_sheet(&sheets, &LoadingParams::default(), None).unwrap();
    let column = &frame.columns[0];
    assert_eq!(column.name, "7");
    assert_eq!(column.dtype, ColDtype::String);
    assert_eq!(column.numeric_kind, NumericKind::NonNumeric);
    assert_eq!(column.values, vec![Cell::Str("123".into()), Cell::Str("18446744073709551615".into())]);
}

#[test]
fn numeric_token_counts_against_field_budget() {
    let sheets: Vec<Sheet<SheetCell>> = vec![("s".into(), vec![
        vec![SheetCell::Cell(Cell::Str("x".into()))],
        vec![SheetCell::IntegerLexeme("9".repeat(100_000))],
    ])];
    let exact = TabularReadLimits::new(100_000, 100_000, 10, 10, 100);
    assert!(read_sheet(&sheets, &LoadingParams::default(), Some(exact)).is_ok());
    let limits = TabularReadLimits::new(100_000, 99_999, 10, 10, 100);
    let error = read_sheet(&sheets, &LoadingParams::default(), Some(limits)).unwrap_err();
    assert!(error.to_string().contains("tabular field exceeds"));
}

#[test]
fn integer_headers_are_canonical_without_storage_narrowing() {
    let tokens = ["007", "+007", "+5", "-0", "1234567890123456789012345678901234567890"];
    let rows = vec![tokens.iter().map(|token| SheetCell::IntegerLexeme((*token).into())).collect()];
    let frame = read_sheet(&[("s".into(), rows)], &LoadingParams::default(), None).unwrap();
    assert_eq!(frame.column_names(), vec!["7", "7.1", "5", "0", "1234567890123456789012345678901234567890"]);
}

#[test]
fn physical_float_and_integer_origins_are_projected_separately() {
    let rows = vec![
        vec![SheetCell::Cell(Cell::Str("value".into()))],
        vec![SheetCell::IntegerLexeme("9007199254740993".into())],
        vec![SheetCell::Float64 { bits: 1e30_f64.to_bits(), token: "1E+30".into() }],
    ];
    let frame = read_sheet(&[("s".into(), rows)], &LoadingParams::default(), None).unwrap();
    assert_eq!(frame.columns[0].values[0], Cell::Int(9_007_199_254_740_993));
    assert_eq!(frame.columns[0].values[1], Cell::Float(1e30));
}

#[test]
fn exact_integer_keys_remain_distinct_through_join() {
    use nirs4all_io_core::materialize::join::join_tables;
    use nirs4all_io_core::spec::enums::{Cardinality, Coverage};

    let make = |keys: &[&str]| {
        let mut rows = vec![vec![SheetCell::Cell(Cell::Str("id".into()))]];
        rows.extend(keys.iter().map(|key| vec![SheetCell::IntegerLexeme((*key).into())]));
        read_sheet(&[("s".into(), rows)], &LoadingParams::default(), None).unwrap()
    };
    let left = make(&["9007199254740993", "9007199254740992"]);
    let right = make(&["9007199254740992", "9007199254740993"]);
    let keys = vec!["id".to_string()];
    let (joined, audit) = join_tables(&left, &right, &keys, &keys, Cardinality::OneToOne, Coverage::Complete, "left", "right").unwrap();
    assert_eq!(audit.n_matched, 2);
    assert_eq!(joined.columns[0].values, vec![Cell::Int(9_007_199_254_740_993), Cell::Int(9_007_199_254_740_992)]);
}

#[test]
fn typed_numeric_tokens_obey_record_and_shape_limits() {
    let rows = vec![
        vec![SheetCell::Cell(Cell::Str("x".into()))],
        vec![SheetCell::IntegerLexeme("123456789".into())],
        vec![SheetCell::IntegerLexeme("2".into())],
    ];
    let sheets = [("s".into(), rows)];
    let record_limit = TabularReadLimits::new(8, 100, 10, 10, 100);
    let error = read_sheet(&sheets, &LoadingParams::default(), Some(record_limit)).unwrap_err();
    assert!(error.to_string().contains("tabular record exceeds"));
    let row_limit = TabularReadLimits::new(100, 100, 1, 10, 100);
    assert!(read_sheet(&sheets, &LoadingParams::default(), Some(row_limit)).is_err());
}

#[test]
fn float_and_out_of_domain_integer_are_distinct_join_keys() {
    use nirs4all_io_core::materialize::join::join_tables;
    use nirs4all_io_core::spec::enums::{Cardinality, Coverage};

    let make = |key| {
        let rows = vec![
            vec![SheetCell::Cell(Cell::Str("id".into()))],
            vec![key],
        ];
        read_sheet(&[("s".into(), rows)], &LoadingParams::default(), None).unwrap()
    };
    let integer = make(SheetCell::IntegerLexeme("1000000000000000000000000000000".into()));
    let float = make(SheetCell::Float64 { bits: 1e30_f64.to_bits(), token: "1E+30".into() });
    let keys = vec!["id".to_string()];
    for (left, right) in [(&integer, &float), (&float, &integer)] {
        let (_, audit) = join_tables(left, right, &keys, &keys, Cardinality::OneToOne, Coverage::Warn, "left", "right").unwrap();
        assert_eq!(audit.n_matched, 0);
        assert!(join_tables(left, right, &keys, &keys, Cardinality::OneToOne, Coverage::Complete, "left", "right").is_err());
        assert!(join_tables(left, right, &keys, &keys, Cardinality::OneToOne, Coverage::Error, "left", "right").is_err());
    }
}

#[test]
fn selected_out_of_domain_metadata_refuses_but_unselected_and_removed_rows_are_exempt() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;

    let run = |select_metadata: bool, remove_sample: bool, bad_feature_is_missing: bool| {
        let columns = if select_metadata {
            json!([{"role":"features","select":["x"]}, {"role":"metadata","select":["m"]}])
        } else {
            // Explicitly exclude m while satisfying complete column assignment.
            json!([{"role":"features","select":["x"]}, {"role":"ignore","select":["m"]}])
        };
        let policy = if remove_sample { "remove_sample" } else { "ignore" };
        let spec = DatasetSpec::from_value(&json!({
            "sources":[{"id":"t","input":"a.xlsx","columns":columns}],
            "params":{"na":{"policy":policy}}
        })).unwrap();
        let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
        let rows = vec![
            vec![text("x"), text("m")],
            vec![SheetCell::Cell(Cell::Int(1)), SheetCell::IntegerLexeme("7".into())],
            vec![if bad_feature_is_missing { SheetCell::Cell(Cell::Na) } else { SheetCell::Cell(Cell::Int(2)) },
                 SheetCell::IntegerLexeme("1000000000000000000000000000000".into())],
        ];
        assemble_in_memory(&spec, &[InMemorySource {
            name: "a.xlsx".into(), payload: SourcePayload::Sheets(vec![("s".into(), rows)]),
        }], &HashMap::new(), None)
    };
    let error = run(true, false, false).unwrap_err();
    assert!(error.message.contains("outside the supported i64/u64 domain"));
    assert!(run(false, false, false).is_ok());
    let surviving = run(true, true, true).unwrap();
    assert_eq!(surviving.blocks["train"].n_samples, 1);
}

#[test]
fn typed_sheet_limits_accept_exact_boundaries_and_reject_each_excess() {
    let sheets = [("s".into(), vec![
        vec![SheetCell::Cell(Cell::Str("x".into())), SheetCell::Cell(Cell::Str("y".into()))],
        vec![SheetCell::IntegerLexeme("123456789".into()), SheetCell::IntegerLexeme("2".into())],
    ])];
    let params = LoadingParams::default();
    assert!(read_sheet(&sheets, &params, Some(TabularReadLimits::new(17, 9, 1, 2, 2))).is_ok());
    for limits in [
        TabularReadLimits::new(16, 9, 1, 2, 2),
        TabularReadLimits::new(17, 8, 1, 2, 2),
        TabularReadLimits::new(17, 9, 0, 2, 2),
        TabularReadLimits::new(17, 9, 1, 1, 2),
        TabularReadLimits::new(17, 9, 1, 2, 1),
    ] {
        assert!(read_sheet(&sheets, &params, Some(limits)).is_err());
    }
}

#[test]
fn binary64_integer_join_uses_exact_values_in_both_directions() {
    use nirs4all_io_core::materialize::join::join_tables;
    use nirs4all_io_core::spec::enums::{Cardinality, Coverage};
    let make = |key| read_sheet(&[("s".into(), vec![
        vec![SheetCell::Cell(Cell::Str("id".into()))], vec![key],
    ])], &LoadingParams::default(), None).unwrap();
    let float = make(SheetCell::Float64 { bits: 0x4340_0000_0000_0000, token: "9007199254740992.0".into() });
    let keys = vec!["id".to_string()];
    for (token, expected) in [("9007199254740992", 1), ("9007199254740993", 0)] {
        let integer = make(SheetCell::IntegerLexeme(token.into()));
        for (left, right) in [(&integer, &float), (&float, &integer)] {
            let (_, audit) = join_tables(left, right, &keys, &keys, Cardinality::OneToOne, Coverage::Warn, "left", "right").unwrap();
            assert_eq!(audit.n_matched, expected);
        }
    }
}

#[test]
fn join_dropped_out_of_domain_metadata_does_not_reach_survivor_admission() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let spec = DatasetSpec::from_value(&json!({
        "params":{"delimiter":",","na":{"policy":"ignore"}},
        "sources":[
            {"id":"a","input":"a.xlsx","columns":[
                {"role":"features","select":["x"]}, {"role":"metadata","select":["m"]},
                {"role":"ignore","select":["k"]}]},
            {"id":"s","kind":"lookup","input":"s.csv","columns":[{"role":"targets","select":["r"]}],
             "join":{"to":"a","left_on":"k","right_on":"k","how":"m:1","coverage":"drop"}}
        ]
    })).unwrap();
    let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
    let rows = vec![
        vec![text("k"),text("x"),text("m")],
        vec![text("Z"),SheetCell::Cell(Cell::Int(1)),SheetCell::IntegerLexeme("1000000000000000000000000000000".into())],
        vec![text("A"),SheetCell::Cell(Cell::Int(2)),SheetCell::IntegerLexeme("7".into())],
    ];
    let result = assemble_in_memory(&spec, &[
        InMemorySource { name:"a.xlsx".into(), payload:SourcePayload::Sheets(vec![("s".into(),rows)]) },
        InMemorySource { name:"s.csv".into(), payload:SourcePayload::Bytes(b"k,r\nA,0.5\n".to_vec()) },
    ], &HashMap::new(), None).unwrap();
    assert_eq!(result.blocks["train"].n_samples, 1);
    assert_eq!(result.blocks["train"].metadata.as_ref().unwrap().column("m").unwrap().values, vec![Cell::Int(7)]);
}

#[test]
fn integer_bounds_text_boolean_and_signed_zero_keep_physical_origin() {
    let rows = vec![
        vec!["integer", "text", "boolean", "zero"].into_iter().map(|v| SheetCell::Cell(Cell::Str(v.into()))).collect(),
        vec![SheetCell::IntegerLexeme("-9223372036854775808".into()), SheetCell::Cell(Cell::Str("-0".into())), SheetCell::Cell(Cell::Bool(false)), SheetCell::IntegerLexeme("-0".into())],
        vec![SheetCell::IntegerLexeme("9223372036854775807".into()), SheetCell::Cell(Cell::Str("123".into())), SheetCell::Cell(Cell::Bool(true)), SheetCell::Float64 { bits: (-0.0_f64).to_bits(), token: "-0.0".into() }],
    ];
    let frame = read_sheet(&[("s".into(), rows)], &LoadingParams::default(), None).unwrap();
    assert_eq!(frame.columns[0].values, vec![Cell::Int(i64::MIN), Cell::Int(i64::MAX)]);
    assert_eq!(frame.columns[1].values, vec![Cell::Str("-0".into()), Cell::Str("123".into())]);
    assert_eq!(frame.columns[2].values, vec![Cell::Bool(false), Cell::Bool(true)]);
    assert_eq!(frame.columns[3].values, vec![Cell::Int(0), Cell::Int(0)]);
}

#[test]
fn concat_preserves_integer_metadata_in_both_input_orders_and_partitions() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let source = |name: &str, id: &str, partition: &str| {
        let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
        InMemorySource { name:name.into(), payload:SourcePayload::Sheets(vec![("s".into(),vec![
            vec![text("id"),text("x"),text("set")],
            vec![SheetCell::IntegerLexeme(id.into()), SheetCell::Cell(Cell::Int(1)), text(partition)],
        ])]) }
    };
    for names in [["a.xlsx", "b.xlsx"], ["b.xlsx", "a.xlsx"]] {
        let spec = DatasetSpec::from_value(&json!({
            "sources":[{"id":"t","input":names,"merge":"concat_samples","columns":{
                "features":"x","metadata":["id","set"]}}],
            "sample_index":{"by":"id","key":"id"},
            "partitions":{"by":"column","column":"set","train_values":["cal"],"test_values":["val"]},
            "params":{"na":{"policy":"ignore"}}
        })).unwrap();
        let sources = [source("a.xlsx","9007199254740993","cal"), source("b.xlsx","9007199254740992","val")];
        let assembled = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
        for (part, expected) in [("train", 9_007_199_254_740_993_i64), ("test", 9_007_199_254_740_992_i64)] {
            let block = &assembled.blocks[part];
            assert_eq!(block.n_samples, 1);
            assert_eq!(block.metadata.as_ref().unwrap().column("id").unwrap().values, vec![Cell::Int(expected)]);
        }
    }
}

#[test]
fn spreadsheet_variation_retains_exact_base_identity() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let spec = DatasetSpec::from_value(&json!({
        "sources":[{"id":"t","input":"base.xlsx","columns":{"features":"x","metadata":"id"},
        "variations":[{"name":"child","input":"child.xlsx"}]}],
        "sample_index":{"by":"id","key":"id"},"params":{"na":{"policy":"ignore"}}
    })).unwrap();
    let source = |name: &str, x: i64| InMemorySource { name:name.into(), payload:SourcePayload::Sheets(vec![("s".into(),vec![
        vec![SheetCell::Cell(Cell::Str("id".into())),SheetCell::Cell(Cell::Str("x".into()))],
        vec![SheetCell::IntegerLexeme("9007199254740993".into()),SheetCell::Cell(Cell::Int(x))],
    ])]) };
    let assembled = assemble_in_memory(&spec, &[source("base.xlsx",1),source("child.xlsx",2)], &HashMap::new(), None).unwrap();
    let block = &assembled.blocks["train"];
    assert_eq!(block.metadata.as_ref().unwrap().column("id").unwrap().values, vec![Cell::Int(9_007_199_254_740_993)]);
    assert_eq!(block.processings[0][0].1.data, vec![2.0]);
}

#[test]
fn feature_merge_aligns_exact_ids_and_retains_numeric_authority() {
    use nirs4all_io_core::materialize::join::concat_features;
    let make = |name: &str, rows: &[(&str, &str)]| {
        let mut cells = vec![vec![SheetCell::Cell(Cell::Str("id".into())), SheetCell::Cell(Cell::Str(name.into()))]];
        cells.extend(rows.iter().map(|(id, value)| vec![SheetCell::IntegerLexeme((*id).into()), SheetCell::IntegerLexeme((*value).into())]));
        read_sheet(&[("s".into(), cells)], &LoadingParams::default(), None).unwrap()
    };
    let left = make("a", &[("9007199254740993", "1"), ("9007199254740992", "2")]);
    let right = make("b", &[("9007199254740992", "20"), ("9007199254740993", "10")]);
    let (merged, _) = concat_features(&[left.clone(), right.clone()], &["a".into(), "b".into()], Some(&["id".into()])).unwrap();
    assert_eq!(merged.column("id").unwrap().values, vec![Cell::Int(9_007_199_254_740_993), Cell::Int(9_007_199_254_740_992)]);
    assert_eq!(merged.column("b").unwrap().values, vec![Cell::Int(10), Cell::Int(20)]);
    let (reversed, _) = concat_features(&[right, left], &["b".into(), "a".into()], Some(&["id".into()])).unwrap();
    assert_eq!(reversed.column("id").unwrap().values, vec![Cell::Int(9_007_199_254_740_992), Cell::Int(9_007_199_254_740_993)]);
    assert_eq!(reversed.column("a").unwrap().values, vec![Cell::Int(2), Cell::Int(1)]);
}

#[test]
fn typed_sheet_masks_and_classification_caps_survive_assembly() {
    use nirs4all_io_core::materialize::{assemble_in_memory, DatasetPackage, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let spec = DatasetSpec::from_value(&json!({
        "sources":[{"id":"t","input":"a.xlsx","columns":{"features":"x","targets":"y","metadata":"id"}}],
        "params":{"na":{"policy":"ignore"}}
    })).unwrap();
    let run = |label: &str| {
        let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
        let rows = vec![vec![text("x"), text("y"), text("id")],
            vec![SheetCell::IntegerLexeme("1".into()), SheetCell::IntegerLexeme(label.into()), SheetCell::IntegerLexeme("9007199254740993".into())],
            vec![SheetCell::Cell(Cell::Na), SheetCell::Cell(Cell::Na), SheetCell::IntegerLexeme("9007199254740992".into())]];
        assemble_in_memory(&spec, &[InMemorySource {name:"a.xlsx".into(), payload:SourcePayload::Sheets(vec![("s".into(), rows)])}], &HashMap::new(), None).unwrap()
    };
    let mut admitted = run("9007199254740991");
    assert_eq!(admitted.blocks["train"].x[0].validity, vec![true, false]);
    assert_eq!(admitted.blocks["train"].y.as_ref().unwrap().validity, vec![true, false]);
    admitted.task_type = "classification".into();
    assert!(DatasetPackage::from_assembled(&admitted).is_ok());
    let mut refused = run("9007199254740992");
    refused.task_type = "classification".into();
    assert!(DatasetPackage::from_assembled(&refused).unwrap_err().contains("Classification"));
    let mut f32_refused = run("16777217");
    f32_refused.task_type = "classification".into();
    let options = nirs4all_io_core::materialize::NumericStorageOptions {
        targets: Some(nirs4all_io_core::materialize::NumericStorage::Float32),
        ..Default::default()
    };
    assert!(DatasetPackage::from_assembled_with_storage(&f32_refused, &options).is_err());
}

#[test]
fn surviving_out_of_domain_integer_refuses_each_selected_numeric_role_and_identity() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    for role in ["features", "targets", "weights", "metadata", "identity"] {
        let mut columns = json!({"features":"x"});
        let mut spec = json!({"sources":[{"id":"t","input":"a.xlsx","columns":{}}], "params":{"na":{"policy":"ignore"}}});
        if role == "features" {
            columns["features"] = json!("bad");
            // Exclude the unused control column so numeric admission is reached.
            columns["ignore"] = json!("x");
        }
        else { columns[if role == "identity" { "metadata" } else { role }] = json!("bad"); }
        spec["sources"][0]["columns"] = columns;
        if role == "identity" { spec["sample_index"] = json!({"by":"id","key":"bad"}); }
        let spec = DatasetSpec::from_value(&spec).unwrap();
        let rows = vec![
            vec![SheetCell::Cell(Cell::Str("x".into())), SheetCell::Cell(Cell::Str("bad".into()))],
            vec![SheetCell::Cell(Cell::Int(1)), SheetCell::IntegerLexeme("1000000000000000000000000000000".into())],
        ];
        let error = assemble_in_memory(&spec, &[InMemorySource { name:"a.xlsx".into(), payload:SourcePayload::Sheets(vec![("s".into(),rows)]) }], &HashMap::new(), None).unwrap_err();
        assert!(error.message.contains("outside the supported i64/u64 domain"), "{role}: {}", error.message);
    }
}

#[test]
fn mixed_physical_text_and_u64_metadata_serialize_with_distinct_json_kinds() {
    use nirs4all_io_core::materialize::{assemble_in_memory, DatasetPackage, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"t","input":"a.xlsx","columns":{"features":"x","metadata":"m"}}],"params":{"na":{"policy":"ignore"}}})).unwrap();
    let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
    let rows = vec![vec![text("x"), text("m")],
        vec![SheetCell::Cell(Cell::Int(1)), text("123")],
        vec![SheetCell::Cell(Cell::Int(2)), SheetCell::IntegerLexeme("18446744073709551615".into())]];
    let assembled = assemble_in_memory(&spec, &[InMemorySource {name:"a.xlsx".into(),payload:SourcePayload::Sheets(vec![("s".into(),rows)])}], &HashMap::new(), None).unwrap();
    let full = DatasetPackage::from_assembled(&assembled).unwrap().to_assembled().to_full_value().unwrap();
    assert_eq!(full.pointer("/blocks/train/metadata/columns/0/values").unwrap(), &json!(["123", 18446744073709551615_u64]));
}

#[test]
fn selected_integer_feature_checks_exact_binary64_before_storage() {
    use nirs4all_io_core::materialize::{assemble_in_memory, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    let spec = DatasetSpec::from_value(&json!({"sources":[{"id":"t","input":"a.xlsx","columns":{"features":"x"}}],"params":{"na":{"policy":"ignore"}}})).unwrap();
    let run = |token: &str| assemble_in_memory(&spec, &[InMemorySource {
        name:"a.xlsx".into(), payload:SourcePayload::Sheets(vec![("s".into(), vec![
            vec![SheetCell::Cell(Cell::Str("x".into()))], vec![SheetCell::IntegerLexeme(token.into())],
        ])]),
    }], &HashMap::new(), None);
    assert_eq!(run("9007199254740992").unwrap().blocks["train"].x[0].data, vec![9_007_199_254_740_992.0]);
    assert!(run("9007199254740993").unwrap_err().message.contains("exactly representable"));
}

fn scientific_partition_source(xlsx: bool, token: &str) -> nirs4all_io_core::materialize::InMemorySource {
    use nirs4all_io_core::materialize::{InMemorySource, SourcePayload};
    let name = if xlsx { "labels.xlsx" } else { "labels.csv" };
    let payload = if xlsx {
        let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
        let mut rows = vec![vec![text("x"), text("y"), text("row"), text("set")]];
        for (i, label) in ["1", "2", token].iter().enumerate() {
            rows.push(vec![SheetCell::Cell(Cell::Int(i as i64 + 1)),
                SheetCell::Cell(Cell::Int(i as i64 + 101)), text(&format!("r{i}")),
                SheetCell::IntegerLexeme((*label).into())]);
        }
        SourcePayload::Sheets(vec![("s".into(), rows)])
    } else {
        SourcePayload::Bytes(format!("x;y;row;set\n1;101;r0;1\n2;102;r1;2\n3;103;r2;{token}\n").into_bytes())
    };
    InMemorySource { name: name.into(), payload }
}

fn scientific_partition_spec(name: &str, token: &str) -> serde_json::Value {
    serde_json::json!({
        "task_type":"regression", "sample_index":{"by":"id","key":"row"},
        "sources":[{"id":"d","input":name,"columns":{
            "features":"x","targets":"y","metadata":"row","ignore":"set"}}],
        "params":{"delimiter":";","na":{"policy":"ignore"}},
        "partitions":{"by":"column","column":"set","train_values":[1,2],"test_values":[token]}
    })
}

#[test]
fn ignored_outside_domain_partition_labels_select_exact_samples_in_csv_and_sheets() {
    use nirs4all_io_core::materialize::assemble_in_memory;
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    for xlsx in [false, true] {
        for token in ["1000000000000000000000000000000", "-9999999999999999999999999999999", "18446744073709551616"] {
            let source = scientific_partition_source(xlsx, token);
            for policy in [None, Some("train"), Some("test"), Some("drop"), Some("error")] {
                for part in ["train", "test", "val", "predict"] {
                    let mut value = scientific_partition_spec(&source.name, token);
                    value["partitions"].as_object_mut().unwrap().remove("test_values");
                    value["partitions"][format!("{part}_values")] = if part == "train" {
                        json!([1,2,token])
                    } else { json!([token]) };
                    if let Some(policy) = policy { value["partitions"]["unknown_policy"] = json!(policy); }
                    let spec = DatasetSpec::from_value(&value).unwrap();
                    let assembled = assemble_in_memory(&spec, std::slice::from_ref(&source), &HashMap::new(), None).unwrap();
                    assert_eq!(assembled.blocks.len(), if part == "train" { 1 } else { 2 });
                    for (partition, block) in &assembled.blocks {
                        let selected = if part == "train" { vec![0,1,2] }
                            else if partition == "train" { vec![0,1] } else { assert_eq!(partition, part); vec![2] };
                        assert_eq!(block.n_samples, selected.len());
                        assert_eq!(block.x[0].data, selected.iter().map(|i| (*i + 1) as f64).collect::<Vec<_>>());
                        assert_eq!(block.y.as_ref().unwrap().data, selected.iter().map(|i| (*i + 101) as f64).collect::<Vec<_>>());
                        assert_eq!(block.metadata.as_ref().unwrap().column("row").unwrap().values,
                            selected.iter().map(|i| Cell::Str(format!("r{i}"))).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}

#[test]
fn ignored_refused_numeric_labels_are_lexical_without_approximate_aliases() {
    use nirs4all_io_core::materialize::assemble_in_memory;
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    for xlsx in [false, true] {
        let source = scientific_partition_source(xlsx, "1000000000000000000000000000000");
        for alias in ["1e30", "+1000000000000000000000000000000", "01000000000000000000000000000000", "1000000000000000019884624838656"] {
            let mut value = scientific_partition_spec(&source.name, alias);
            value["partitions"]["unknown_policy"] = json!("drop");
            let spec = DatasetSpec::from_value(&value).unwrap();
            let assembled = assemble_in_memory(&spec, std::slice::from_ref(&source), &HashMap::new(), None).unwrap();
            assert_eq!(assembled.blocks.len(), 1);
            let block = &assembled.blocks["train"];
            assert_eq!(block.x[0].data, vec![1.0,2.0]);
            assert_eq!(block.y.as_ref().unwrap().data, vec![101.0,102.0]);
            assert_eq!(block.metadata.as_ref().unwrap().column("row").unwrap().values, vec![Cell::Str("r0".into()),Cell::Str("r1".into())]);
        }
    }
    // CSV retains an overflowed decimal as numeric text; its ignored label may
    // match its spelling, but must not acquire the numeric identity of infinity.
    let source = scientific_partition_source(false, "1e999");
    for (declared, matches) in [("1e999", true), ("inf", false), ("1e998", false)] {
        let mut value = scientific_partition_spec(&source.name, declared);
        value["partitions"]["unknown_policy"] = json!("drop");
        let spec = DatasetSpec::from_value(&value).unwrap();
        let assembled = assemble_in_memory(&spec, std::slice::from_ref(&source), &HashMap::new(), None).unwrap();
        assert_eq!(assembled.blocks.len(), if matches { 2 } else { 1 });
        if matches {
            let block = &assembled.blocks["test"];
            assert_eq!(block.x[0].data, vec![3.0]);
            assert_eq!(block.y.as_ref().unwrap().data, vec![103.0]);
            assert_eq!(block.metadata.as_ref().unwrap().column("row").unwrap().values, vec![Cell::Str("r2".into())]);
        }
    }
    let mut selected = scientific_partition_spec(&source.name, "1e999");
    selected["sources"][0]["columns"]["metadata"] = json!(["row","set"]);
    selected["sources"][0]["columns"].as_object_mut().unwrap().remove("ignore");
    let error = assemble_in_memory(&DatasetSpec::from_value(&selected).unwrap(), &[source], &HashMap::new(), None).unwrap_err();
    assert!(error.message.contains("finite"), "{}", error.message);
}

#[test]
fn in_domain_u64_spellings_share_folds_and_canonical_sample_identity() {
    use nirs4all_io_core::materialize::{assemble_in_memory, DatasetPackage, InMemorySource, SourcePayload};
    use nirs4all_io_core::spec::DatasetSpec;
    use serde_json::json;
    use std::collections::HashMap;
    for xlsx in [false, true] {
        let ids = ["+18446744073709551615", "0018446744073709551614", "18446744073709551613", "18446744073709551612"];
        let labels = ["18446744073709551615", "+18446744073709551615", "0018446744073709551615", "18446744073709551614"];
        let name = if xlsx { "u64.xlsx" } else { "u64.csv" };
        let payload = if xlsx {
            let text = |s: &str| SheetCell::Cell(Cell::Str(s.into()));
            let mut rows = vec![vec![text("id"),text("x"),text("y"),text("cv")]];
            for i in 0..4 { rows.push(vec![SheetCell::IntegerLexeme(ids[i].into()),SheetCell::Cell(Cell::Int(i as i64 + 1)),SheetCell::Cell(Cell::Int(i as i64 + 101)),SheetCell::IntegerLexeme(labels[i].into())]); }
            SourcePayload::Sheets(vec![("s".into(), rows)])
        } else {
            let mut csv = "id;x;y;cv\n".to_string();
            for i in 0..4 { csv.push_str(&format!("{};{};{};{}\n", ids[i], i+1, i+101, labels[i])); }
            SourcePayload::Bytes(csv.into_bytes())
        };
        let spec = DatasetSpec::from_value(&json!({
            "task_type":"regression","sample_index":{"by":"id","key":"id"},
            "sources":[{"id":"d","input":name,"columns":{"features":"x","targets":"y","metadata":["id","cv"]}}],
            "params":{"delimiter":";","na":{"policy":"ignore"}},"folds":{"column":"cv"}
        })).unwrap();
        let assembled = assemble_in_memory(&spec, &[InMemorySource {name:name.into(),payload}], &HashMap::new(), None).unwrap();
        assert_eq!(assembled.folds, vec![(vec![0,1,2],vec![3]),(vec![3],vec![0,1,2])]);
        let block = &assembled.blocks["train"];
        assert_eq!(block.x[0].data, vec![1.0,2.0,3.0,4.0]);
        assert_eq!(block.y.as_ref().unwrap().data, vec![101.0,102.0,103.0,104.0]);
        let canonical: Vec<String> = (0..4).map(|i| (u64::MAX - i).to_string()).collect();
        assert_eq!(block.metadata.as_ref().unwrap().column("id").unwrap().values, canonical.iter().cloned().map(Cell::Str).collect::<Vec<_>>());
        let package = DatasetPackage::from_assembled(&assembled).unwrap();
        let (identity, _, _) = package.executed_projection(&["d".into()]).unwrap();
        assert_eq!(identity.sample_ids, canonical);
        let full = package.to_assembled().to_full_value().unwrap();
        assert_eq!(full.pointer("/blocks/train/metadata/columns/0/values").unwrap(), &json!([u64::MAX,u64::MAX-1,u64::MAX-2,u64::MAX-3]));
    }
}
