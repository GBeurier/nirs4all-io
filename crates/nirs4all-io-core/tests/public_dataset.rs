use nirs4all_io_core::public_dataset::{
    canonical_content_bytes, dataset_content_bytes, dense_dataset_package,
    multimodal_runtime_input, normalize_dataset, public_source_schema, u07_sources,
};
use serde_json::{json, Value};
fn input() -> Value {
    json!({"schema":"nirs4all.dataset.v1","schema_version":1,"origin_ids":["a","b","c","d"],"fold_ids":["0","0","1","1"],"dataset":{"schema":"nirs4all.multimodal-dataset","schema_version":1,"name":"public","sample_ids":["a","b","c","d"],"source_alignment":"strict","sources":[{"name":"spectra","sample_ids":["d","c","b","a"],"representation_id":"signal_1d","axes":["sample","wavelength"],"feature_names":null,"axis_units":{"wavelength":"nm"},"axis_coordinates":{"wavelength":[900,1000]},"array":{"dtype":"float64","shape":[4,2],"values":[[7.0,8.0],[5.0,6.0],[3.0,4.0],[1.0,2.0]]}}],"y":{"dtype":"float64","shape":[4],"values":[1.0,2.0,3.0,4.0]},"groups":null,"partitions":{"dtype":"<U5","shape":[4],"values":["train","train","train","train"]}}})
}
#[test]
fn aligns_by_identity_and_preserves_folds_in_existing_package() {
    let normalized = normalize_dataset(&input()).unwrap();
    assert_eq!(
        normalized["dataset"]["sources"][0]["array"]["values"][0],
        json!([1.0, 2.0])
    );
    let package = dense_dataset_package(&input(), "spectra").unwrap();
    let assembled = package.to_assembled();
    assert_eq!(
        assembled.blocks["train"].x[0].data,
        vec![1., 2., 3., 4., 5., 6., 7., 8.]
    );
    assert_eq!(assembled.fold_provenance.len(), 2);
    assert_eq!(
        assembled.fold_provenance[0].validation_observation_ids,
        vec!["a", "b"]
    );
}
#[test]
fn origin_group_and_axis_failures_are_closed() {
    let mut value = input();
    value["origin_ids"][2] = json!("a");
    assert!(normalize_dataset(&value).unwrap_err().contains("origin"));
    let mut value = input();
    value["dataset"]["groups"] = json!({"dtype":"<U1","shape":[4],"values":["x","y","x","z"]});
    assert!(normalize_dataset(&value).unwrap_err().contains("group"));
    let mut value = input();
    value["dataset"]["sources"][0]["axes"] = json!(["sample", "feature"]);
    assert!(normalize_dataset(&value).is_err());
}
#[test]
fn missing_rows_require_explicit_left_alignment() {
    let mut value = input();
    let source = &mut value["dataset"]["sources"][0];
    source["sample_ids"] = json!(["a"]);
    source["array"]["shape"][0] = json!(1);
    source["array"]["values"] = json!([[1., 2.]]);
    assert!(normalize_dataset(&value).is_err());
    value["dataset"]["source_alignment"] = json!("left");
    let normalized = normalize_dataset(&value).unwrap();
    assert_eq!(
        normalized["dataset"]["sources"][0]["presence_mask"]["values"],
        json!([true, false, false, false])
    );
    assert!(dense_dataset_package(&value, "spectra").is_err());
}

#[test]
fn unknown_units_are_equivalent_and_float32_values_follow_declared_storage() {
    let mut omitted = input();
    omitted["dataset"]["sources"][0]["axis_units"] = json!({});
    let mut explicit = omitted.clone();
    explicit["dataset"]["sources"][0]["axis_units"] = json!({"wavelength":null});
    assert_eq!(
        public_source_schema(&omitted, "spectra").unwrap(),
        public_source_schema(&explicit, "spectra").unwrap()
    );
    assert_ne!(
        public_source_schema(&input(), "spectra").unwrap(),
        public_source_schema(&omitted, "spectra").unwrap()
    );
    explicit["dataset"]["sources"][0]["array"]["dtype"] = json!("float32");
    explicit["dataset"]["sources"][0]["array"]["values"][0][0] = json!(1.234567891);
    let normalized = normalize_dataset(&explicit).unwrap();
    assert_eq!(
        normalized["dataset"]["sources"][0]["array"]["values"][3][0],
        json!(1.234567891_f64 as f32 as f64)
    );
}

#[test]
fn unsafe_integer_coordinates_fail_before_transport() {
    let mut value = input();
    value["dataset"]["sources"][0]["axis_coordinates"]["wavelength"] =
        json!([9_007_199_254_740_992_u64, 9_007_199_254_740_994_u64]);
    assert!(normalize_dataset(&value)
        .unwrap_err()
        .contains("exactly representable"));
}

#[test]
fn content_bytes_match_shared_unicode_and_numeric_golden() {
    use sha2::{Digest, Sha256};
    let golden: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/public-content-v1.json"
    ))
    .unwrap();
    let bytes = canonical_content_bytes(&golden["input"]).unwrap();
    assert_eq!(
        String::from_utf8(bytes.clone()).unwrap(),
        golden["canonical_content_utf8"]
    );
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), golden["sha256"]);
    assert_eq!(
        canonical_content_bytes(&json!([0, 1, 1000])).unwrap(),
        canonical_content_bytes(&json!([-0.0, 1.0, 1e3])).unwrap()
    );
}

#[test]
fn portable_validators_do_not_infer_from_invalid_declarations() {
    let mut value = input();
    value["dataset"]["source_alignment"] = Value::Null;
    assert!(normalize_dataset(&value).is_err());
    let mut value = input();
    value["dataset"]["repetition_ids"] = json!(["a", "b", "c", "d"]);
    assert!(normalize_dataset(&value).is_err());
    let mut value = input();
    let source = &mut value["dataset"]["sources"][0];
    source["representation_id"] = json!("tabular_numeric");
    source["axes"] = json!(["sample", "feature"]);
    source["axis_units"] = json!({});
    source["axis_coordinates"] = json!({"feature":[1,"category"]});
    assert!(normalize_dataset(&value).is_err());
    let mut value = input();
    value["dataset"]["sources"][0]["array"]["values"][0][0] = json!(9_007_199_254_740_992_u64);
    assert!(normalize_dataset(&value).is_ok()); // Declared float64, independent of JSON integer spelling.
}

#[test]
fn metadata_ascii_whitespace_includes_vertical_tab() {
    let original: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/runtime-u07-groups.json"
    ))
    .unwrap();
    for whitespace in [" ", "\t", "\n", "\r", "\u{000b}", "\u{000c}"] {
        let mut variant = original.clone();
        variant["dataset"]["sources"][3]["array"]["values"][0][0] =
            json!(format!("{whitespace}20.5{whitespace}"));
        assert!(
            u07_sources(&variant).is_ok(),
            "ASCII whitespace {whitespace:?}"
        );
        assert_eq!(
            dataset_content_bytes(&variant).unwrap(),
            dataset_content_bytes(&original).unwrap()
        );
    }
}

#[test]
fn runtime_groups_preserve_string_ids_and_refuse_numeric_labels() {
    let original: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/runtime-u07-groups.json"
    ))
    .unwrap();
    let output = multimodal_runtime_input(&original).unwrap();
    assert_eq!(
        output["coordinator_relations"]["records"][0]["group_id"],
        json!("1.0")
    );
    assert_eq!(
        output["coordinator_relations"]["records"][1]["group_id"],
        json!("1")
    );
    for labels in [json!([1.0, 1.5]), json!([1, 2]), json!(["1", 2.0])] {
        let mut variant = original.clone();
        variant["dataset"]["groups"] = json!({"dtype":"object","shape":[2],"values":labels});
        assert_eq!(
            normalize_dataset(&variant).unwrap()["dataset"]["groups"]["values"],
            labels
        );
        assert!(multimodal_runtime_input(&variant)
            .unwrap_err()
            .contains("group IDs must be nonempty strings"));
    }
}
