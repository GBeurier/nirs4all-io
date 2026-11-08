use nirs4all_io_core::materialize::{
    assemble_in_memory_with_storage, DatasetPackage, InMemorySource, NumericStorage,
    NumericStorageOptions, SourcePayload,
};
use nirs4all_io_core::spec::DatasetSpec;
use serde_json::{json, Value};

#[test]
fn shared_fixture_declares_file_operations_as_facade_only() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/owner_review_vectors.json")).unwrap();
    assert_eq!(fixture["schema"], "nirs4all.io.owner-review-vectors.v1");
    let cases = fixture["cases"].as_array().unwrap();
    let unsupported_file_cases = cases
        .iter()
        .filter(|case| {
            matches!(
                case["operation"].as_str(),
                Some("file_package" | "assembly_reselect")
            )
        })
        .count();
    assert!(
        unsupported_file_cases >= 181,
        "pure core has no filesystem route"
    );
    assert!(cases
        .iter()
        .filter(|c| c["operation"] == "python_from_dict")
        .all(|c| c["hosts"] == json!(["python"])));
}

fn run(
    csv: &str,
    spec: Value,
    options: &NumericStorageOptions,
) -> Result<nirs4all_io_core::materialize::AssembledDataset, String> {
    let spec = DatasetSpec::from_value(&spec).map_err(|e| e.message)?;
    assemble_in_memory_with_storage(
        &spec,
        &[InMemorySource {
            name: "a.csv".into(),
            payload: SourcePayload::Bytes(csv.as_bytes().to_vec()),
        }],
        &Default::default(),
        None,
        options,
    )
    .map_err(|e| e.message)
}

#[test]
fn numeric_csv_metadata_retains_exact_integers_and_validates_after_row_removal() {
    let spec = json!({"sources":[{"id":"t","input":"a.csv","columns":{"features":"x","metadata":"m"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}});
    for (csv, expected) in [
        (
            "m,x\n+0009007199254740993,1\n1.5,2\n",
            json!([9007199254740993_u64, 1.5]),
        ),
        (
            "m,x\n18446744073709551615,1\nNA,2\n",
            json!([18446744073709551615_u64, null]),
        ),
        (
            "m,x\n18014398509481984,1\n1.5,2\n",
            json!([18014398509481984_f64, 1.5]),
        ),
    ] {
        let a = run(csv, spec.clone(), &Default::default()).unwrap();
        assert_eq!(
            a.to_full_value().unwrap()["blocks"]["train"]["metadata"]["columns"][0]["values"],
            expected
        );
        let p = DatasetPackage::from_assembled(&a).unwrap();
        assert_eq!(
            p.to_assembled().to_full_value().unwrap()["blocks"]["train"]["metadata"]["columns"][0]
                ["values"],
            expected
        );
    }
    for token in [
        "18446744073709551616",
        "-9223372036854775809",
        "1e-400",
        "1e400",
    ] {
        let csv = format!("m,x\n{token},1\n1.5,2\n");
        assert!(
            run(&csv, spec.clone(), &Default::default()).is_err(),
            "{token}"
        );
        let text = format!("m,x\n{token},1\nsite,2\n");
        assert!(
            run(&text, spec.clone(), &Default::default()).is_ok(),
            "literal {token}"
        );
    }
    let remove = json!({"task_type":"regression","sources":[{"id":"t","input":"a.csv","columns":{"features":"x","targets":"y"}}],"params":{"delimiter":",","na":{"policy":"remove_sample"}}});
    assert!(run(
        "x,y\n18446744073709551616,NA\n1,2\n",
        remove.clone(),
        &Default::default()
    )
    .is_ok());
    assert!(run(
        "x,y\n18446744073709551616,3\n1,2\n",
        remove,
        &Default::default()
    )
    .is_err());
}

#[test]
fn ignored_numeric_columns_are_not_validated_as_selected_data() {
    let spec = json!({"sources":[{"id":"t","input":"a.csv","columns":[{"role":"features","select":"x"},{"role":"ignore","select":"ignored"}]}],"params":{"delimiter":","}});
    assert!(run(
        "x,ignored\n1,18446744073709551616\n2,42\n",
        spec,
        &Default::default()
    )
    .is_ok());
}

#[test]
fn csv_boolean_words_are_observed_and_mixed_text_stays_literal() {
    let spec = json!({"sources":[{"id":"t","input":"a.csv","columns":{"features":"x","metadata":"m"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}});
    let a = run(
        "x,m\nTRUE,TrUe\nfalse,site\nNA,site\n",
        spec,
        &Default::default(),
    )
    .unwrap();
    let b = &a.blocks["train"];
    assert_eq!(&b.x[0].data[..2], &[1.0, 0.0]);
    assert_eq!(b.x[0].validity, [true, true, false]);
    assert_eq!(
        b.metadata.as_ref().unwrap().column("m").unwrap().values[0].to_str_scalar(),
        "TrUe"
    );
}

#[test]
fn boolean_metadata_survives_a_real_auto_target_selection() {
    let spec = json!({"sources":[{"id":"t","role":"mixed","input":"a.csv","strict_columns":false,"columns":[{"role":"targets","select":"auto"},{"role":"features","select":{"dtype":"numeric"}},{"role":"metadata","select":"rest"}]}],"params":{"delimiter":","}});
    let a = run(
        "flag,x,y\nTRUE,1.5,2\nfalse,2.5,3\n",
        spec,
        &Default::default(),
    )
    .unwrap();
    let b = &a.blocks["train"];
    assert_eq!(b.y_headers, ["y"]);
    assert_eq!(b.feature_headers[0], ["x"]);
    assert_eq!(
        b.metadata.as_ref().unwrap().column("flag").unwrap().values,
        [
            nirs4all_io_core::materialize::Cell::Bool(true),
            nirs4all_io_core::materialize::Cell::Bool(false)
        ]
    );
}

#[test]
fn carried_requested_and_consumer_f32_refuse_reselection_but_real_f32_widens() {
    use nirs4all_io_core::materialize::{Matrix, PrecisionRecord};
    for (request, consumer) in [
        (Some(NumericStorage::Float32), None),
        (None, Some("SpectroDataset")),
    ] {
        let mut m = Matrix::new(vec![1.0], 1, 1);
        m.precision = PrecisionRecord::select(Some("float64"), request, consumer, true).unwrap();
        assert!(m
            .with_storage(Some(NumericStorage::Float64), None, true)
            .is_err());
        m.data[0] = 1.0 + 2_f64.powi(-30);
        assert!(m
            .with_storage(None, None, true)
            .unwrap_err()
            .contains("carried storage"));
    }
    let mut source = Matrix::new(vec![1.0], 1, 1);
    source.precision = PrecisionRecord::select(Some("float32"), None, None, true).unwrap();
    assert_eq!(
        source
            .with_storage(Some(NumericStorage::Float64), None, true)
            .unwrap()
            .data,
        [1.0]
    );
}

#[test]
fn aggregate_threshold_rejects_non_numbers_and_preserves_absent_default() {
    let make =
        |v: Value| json!({"sources":[{"id":"x","input":"a.csv","role":"features"}],"aggregate":v});
    assert_eq!(
        DatasetSpec::from_value(&make(json!({})))
            .unwrap()
            .aggregate
            .unwrap()
            .outlier_threshold,
        0.95
    );
    for value in [json!(null), json!(true), json!("0.5")] {
        assert!(DatasetSpec::from_value(&make(json!({"outlier_threshold":value}))).is_err());
    }
    assert_eq!(
        DatasetSpec::from_value(&make(json!({"outlier_threshold":0.7})))
            .unwrap()
            .aggregate
            .unwrap()
            .outlier_threshold,
        0.7
    );
}

#[test]
fn mixed_numeric_ids_keep_original_integer_authority_and_missing_ids_refuse_projection() {
    let spec = json!({"sample_index":{"by":"id","key":"id"},"sources":[{"id":"x","input":"a.csv","columns":{"features":"x","metadata":"id"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}});
    let assembled = run(
        "id,x\n+0009007199254740993,1\n1.5,2\n",
        spec.clone(),
        &Default::default(),
    )
    .unwrap();
    let package = DatasetPackage::from_assembled(&assembled).unwrap();
    let receipt = package.executed_projection(&["x".into()]).unwrap().0;
    assert_eq!(receipt.sample_ids, ["9007199254740993", "1.5"]);
    let missing = run(
        "id,x\n9007199254740993,1\nNA,2\n",
        spec,
        &Default::default(),
    )
    .unwrap();
    assert!(DatasetPackage::from_assembled(&missing)
        .unwrap()
        .executed_projection(&["x".into()])
        .unwrap_err()
        .contains("Missing projection identity"));
}

#[test]
fn in_memory_package_composes_initial_storage_and_shape_limits() {
    use nirs4all_io_core::materialize::{
        package_in_memory_with_storage_and_tabular_limits, TabularReadLimits,
    };
    let spec=DatasetSpec::from_value(&json!({"sources":[{"id":"x","input":"a.csv","role":"features"}],"params":{"delimiter":","}})).unwrap();
    let sources = [InMemorySource {
        name: "a.csv".into(),
        payload: SourcePayload::Bytes(b"x\n18014399583223809\n1\n".to_vec()),
    }];
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        ..Default::default()
    };
    let tight = TabularReadLimits::new(1024, 1024, 1, 2, 4);
    let error = package_in_memory_with_storage_and_tabular_limits(
        &spec,
        &sources,
        &Default::default(),
        None,
        Some(tight),
        &options,
    )
    .unwrap_err();
    assert!(error.message.contains("row budget"), "{}", error.message);
    let package = package_in_memory_with_storage_and_tabular_limits(
        &spec,
        &sources,
        &Default::default(),
        None,
        None,
        &options,
    )
    .unwrap();
    assert_eq!(
        (package.to_assembled().blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5a800001
    );
}
