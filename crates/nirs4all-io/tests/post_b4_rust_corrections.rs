//! Exact post-B4 reviewer cases and per-loaded-column controls.
use nirs4all_io::core::infer::table::ColDtype;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{
    package_with_limits_and_storage, DatasetPackage, LoadLimits, NumericStorageOptions,
};
use serde_json::{json, Value};

fn package(files: &[(&str, &str)], spec: Value) -> Result<DatasetPackage, String> {
    let dir = tempfile::tempdir().unwrap();
    for (name, text) in files {
        std::fs::write(dir.path().join(name), text).unwrap();
    }
    package_with_limits_and_storage(
        &DatasetSpec::from_value(&spec).unwrap(),
        dir.path(),
        LoadLimits::default(),
        &NumericStorageOptions::default(),
    )
    .map_err(|e| e.message)
}

fn spec(order: &[&str], role: &str, remove: bool) -> Value {
    let mut value = json!({"sources":[{"id":"d","role":"mixed","input":order,"merge":"concat_samples","columns":[{"role":"features","select":["x"]},{"role":"targets","select":["y"]},{"role":role,"select":["m"]}]}],"params":{"delimiter":",","na":{"policy":if remove {"remove_sample"} else {"ignore"}}}});
    if role == "id" {
        value["sources"][0]["columns"][2]["role"] = json!("metadata");
        value["sources"][0]["key"] = json!("m");
        value["sample_index"] = json!({"by":"id","key":"m"});
    }
    value
}

#[test]
fn reviewer_bool_na_metadata_keeps_values_with_string_dtype() {
    let p = package(&[("d.csv", "m,x,y\nTRUE,1.5,1.0\n,2.5,2.0\n")], json!({"sources":[{"id":"t","input":"d.csv","columns":{"features":"x","targets":"y","metadata":"m"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}})).unwrap();
    let a = p.to_assembled();
    let column = a.blocks["train"]
        .metadata
        .as_ref()
        .unwrap()
        .column("m")
        .unwrap();
    assert_eq!(column.dtype, ColDtype::String);
    let full = a.to_full_value().unwrap();
    assert_eq!(
        full["blocks"]["train"]["metadata"]["columns"][0]["values"],
        json!([true, null])
    );
}

#[test]
fn bool_without_na_and_dtype_selectors_keep_existing_behavior() {
    for (tail, dtype) in [("FALSE", ColDtype::Bool), ("", ColDtype::String)] {
        let csv = format!("m,x,y\nTRUE,1.5,1.0\n{tail},2.5,2.0\n");
        let p = package(&[("d.csv", &csv)], json!({"sources":[{"id":"t","input":"d.csv","strict_columns":false,"columns":[{"role":"targets","select":["y"]},{"role":"features","select":{"dtype":"numeric"}},{"role":"metadata","select":"rest"}]}],"params":{"delimiter":",","na":{"policy":"ignore"}}})).unwrap();
        let a = p.to_assembled();
        assert_eq!(a.blocks["train"].feature_headers, vec![vec!["x"]]);
        assert_eq!(
            a.blocks["train"]
                .metadata
                .as_ref()
                .unwrap()
                .column("m")
                .unwrap()
                .dtype,
            dtype
        );
    }
}

#[test]
fn reviewer_text_concat_cannot_erase_other_loaded_numeric_authority() {
    let files = [
        ("a.csv", "m,x,y\n3,1.5,1.0\nabc,2.5,2.0\n"),
        ("b.csv", "m,x,y\n18446744073709551616,3.5,3.0\n4,4.5,4.0\n"),
    ];
    for order in [["a.csv", "b.csv"], ["b.csv", "a.csv"]] {
        for role in ["metadata", "id", "weights"] {
            let result = package(&files, spec(&order, role, false));
            assert!(
                result.is_err(),
                "{order:?}/{role}: numeric authority disappeared"
            );
            assert!(result.unwrap_err().contains("outside the supported"));
        }
    }
}

#[test]
fn reviewer_retained_integer_concat_preserves_float_peer_and_id_spelling() {
    let files = [
        ("a.csv", "m,x,y\n9007199254740993,1.5,1.0\n3,2.5,2.0\n"),
        ("b.csv", "m,x,y\n1.5,3.5,3.0\n4,4.5,4.0\n"),
    ];
    for order in [["a.csv", "b.csv"], ["b.csv", "a.csv"]] {
        for role in ["metadata", "id"] {
            let p = package(&files, spec(&order, role, false)).unwrap();
            let full = p.to_assembled().to_full_value().unwrap();
            let column = full["blocks"]["train"]["metadata"]["columns"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == "m")
                .unwrap();
            let expected = if order[0] == "a.csv" {
                json!([9007199254740993_u64, 3, 1.5, 4.0])
            } else {
                json!([1.5, 4.0, 9007199254740993_u64, 3])
            };
            assert_eq!(column["values"], expected);
            let offset = if order[0] == "a.csv" { 3 } else { 1 };
            assert!(column["values"][offset].as_number().unwrap().is_f64());
            if role == "id" {
                let ids = p
                    .executed_projection(&p.identity.source_ids)
                    .unwrap()
                    .0
                    .sample_ids;
                assert_eq!(ids[offset], "4.0");
            }
        }
    }
}

#[test]
fn loaded_authority_remains_deferred_until_surviving_rows_and_keeps_text_literal() {
    let files = [
        (
            "a.csv",
            "m,x,y\n18446744073709551616,1.5,1.0\nabc,2.5,2.0\n",
        ),
        ("b.csv", "m,x,y\n18446744073709551616,,3.0\n4,4.5,4.0\n"),
    ];
    for order in [["a.csv", "b.csv"], ["b.csv", "a.csv"]] {
        let p = package(&files, spec(&order, "metadata", true)).unwrap();
        let full = p.to_assembled().to_full_value().unwrap();
        let values = &full["blocks"]["train"]["metadata"]["columns"][0]["values"];
        assert!(values
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "18446744073709551616"));
        assert_eq!(full["blocks"]["train"]["n_samples"], 3);
    }
}

#[test]
fn selected_integer_domain_and_direct_f32_rounding_stay_closed() {
    use nirs4all_io::core::materialize::frame::Cell;
    use nirs4all_io::core::materialize::NumericStorage;
    for storage in [NumericStorage::Float32, NumericStorage::Float64] {
        for text in [
            "18446744073709551616",
            "-9223372036854775809",
            "170141183460469231731687303715884105728",
        ] {
            assert_eq!(
                Cell::Str(text.into())
                    .to_numeric_with_storage(storage, None)
                    .unwrap_err(),
                "Integer numeric text is outside the supported i64/u64 domain"
            );
        }
    }
    assert!(Cell::Str("9007199254740993".into())
        .to_numeric_checked()
        .unwrap_err()
        .contains("exactly representable"));
    let direct = Cell::Str("18014399583223809".into())
        .to_numeric_with_storage(NumericStorage::Float32, None)
        .unwrap();
    assert_eq!((direct as f32).to_bits(), 0x5a800001);
    for (text, expected) in [
        ("+0004", 4.0_f64.to_bits()),
        ("-0", (-0.0_f64).to_bits()),
        (
            "-0000000000000000000000000000000000000",
            (-0.0_f64).to_bits(),
        ),
    ] {
        assert_eq!(
            Cell::Str(text.into())
                .to_numeric_checked()
                .unwrap()
                .to_bits(),
            expected
        );
    }
}
