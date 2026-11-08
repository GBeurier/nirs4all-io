use nirs4all_io::api::{load_package_with_limits_and_storage, Input, LoadLimits};
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{
    assemble_with_limits, package_with_limits_and_storage, DatasetPackage, NumericStorage,
    NumericStorageOptions,
};
use serde_json::json;

#[test]
fn package_selection_happens_before_integer_coercion_under_the_shared_read_budget() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "x\n18014399583223809\n1\n";
    std::fs::write(dir.path().join("a.csv"), csv).unwrap();
    let spec=DatasetSpec::from_value(&json!({"sources":[{"id":"x","input":"a.csv","role":"features"}],"params":{"delimiter":","}})).unwrap();
    let opts = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        ..Default::default()
    };
    assert!(package_with_limits_and_storage(
        &spec,
        dir.path(),
        LoadLimits::default(),
        &Default::default()
    )
    .unwrap_err()
    .message
    .contains("exactly representable"));
    let package =
        package_with_limits_and_storage(&spec, dir.path(), LoadLimits::default(), &opts).unwrap();
    assert_eq!(
        (package.to_assembled().blocks["train"].x[0].data[0] as f32).to_bits(),
        0x5a800001
    );
    let tight = LoadLimits {
        max_total_bytes: 2,
        ..Default::default()
    };
    let err = package_with_limits_and_storage(&spec, dir.path(), tight, &opts).unwrap_err();
    let baseline = assemble_with_limits(&spec, dir.path(), tight).unwrap_err();
    assert_eq!(err.message, baseline.message);
    assert!(err.message.contains("limit"));
    let normal = "x\n1\n2\n";
    std::fs::write(dir.path().join("a.csv"), normal).unwrap();
    let assembled = assemble_with_limits(&spec, dir.path(), LoadLimits::default()).unwrap();
    let composed = package_with_limits_and_storage(
        &spec,
        dir.path(),
        LoadLimits::default(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        composed.to_full_value().unwrap(),
        DatasetPackage::from_assembled(&assembled)
            .unwrap()
            .to_full_value()
            .unwrap()
    );
}

#[test]
fn input_config_and_data_share_the_package_budget() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "x\n1\n2\n";
    std::fs::write(dir.path().join("a.csv"), csv).unwrap();
    let config=json!({"sources":[{"id":"x","input":"a.csv","role":"features"}],"params":{"delimiter":","}}).to_string();
    let path = dir.path().join("spec.json");
    std::fs::write(&path, &config).unwrap();
    let input = Input::Path(path.to_string_lossy().into());
    let limits = LoadLimits {
        max_total_bytes: (config.len() + csv.len() - 1) as u64,
        ..Default::default()
    };
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        ..Default::default()
    };
    assert!(
        load_package_with_limits_and_storage(&input, None, None, limits, &options)
            .unwrap_err()
            .message
            .contains("limit")
    );
    assert!(load_package_with_limits_and_storage(
        &input,
        None,
        None,
        LoadLimits::default(),
        &options
    )
    .is_ok());
}

#[test]
fn config_data_and_index_or_fold_share_one_budget_with_storage_requested() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "x\n1\n2\n";
    std::fs::write(dir.path().join("a.csv"), csv).unwrap();
    std::fs::write(dir.path().join("indices.json"), "[0,1]").unwrap();
    std::fs::write(
        dir.path().join("folds.json"),
        "[{\"train\":[0],\"val\":[1]}]",
    )
    .unwrap();
    for (key, value) in [
        (
            "partitions",
            json!({"by":"index_file","train_file":"indices.json"}),
        ),
        ("folds", json!({"file":"folds.json","format":"json"})),
    ] {
        let mut spec = json!({"sources":[{"id":"x","input":"a.csv","role":"features"}],"params":{"delimiter":","}});
        spec[key] = value;
        let config = spec.to_string();
        let path = dir.path().join("config.json");
        std::fs::write(&path, &config).unwrap();
        let input = Input::Path(path.to_string_lossy().into());
        let limits = LoadLimits {
            max_total_bytes: (config.len() + csv.len()) as u64,
            ..Default::default()
        };
        let options = NumericStorageOptions {
            features: Some(NumericStorage::Float32),
            ..Default::default()
        };
        assert!(
            load_package_with_limits_and_storage(&input, None, None, limits, &options)
                .unwrap_err()
                .message
                .contains("limit")
        );
        assert!(load_package_with_limits_and_storage(
            &input,
            None,
            None,
            LoadLimits::default(),
            &options
        )
        .is_ok());
    }
}
