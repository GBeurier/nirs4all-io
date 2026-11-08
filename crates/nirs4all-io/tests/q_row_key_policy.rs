// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Filesystem propagation and real host-limit kwargs. Native execution is UNRUN.
use nirs4all_io::api::{
    load_assembled, load_assembled_with_limits, load_package_with_limits_and_storage, to_spec,
    Input,
};
use nirs4all_io::core::materialize::NumericStorageOptions;
use nirs4all_io::materialize::LoadLimits;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/q_row_key_policy_v1.json")).unwrap()
}
#[test]
fn automatic_directory_and_reversed_list_reach_both_outputs() {
    for case in fixture()["rows"].as_array().unwrap().iter().filter(|case| {
        case["methods"]
            .as_array()
            .unwrap()
            .iter()
            .any(|method| method == "auto")
    }) {
        let dir = tempfile::tempdir().unwrap();
        let mut names = Vec::new();
        for (name, text) in case["files"]
            .as_object()
            .unwrap()
            .iter()
            .chain(case["extra"].as_object().into_iter().flatten())
        {
            let path = dir.path().join(name);
            std::fs::write(&path, text.as_str().unwrap()).unwrap();
            names.push(path.to_string_lossy().to_string());
        }
        names.reverse();
        for input in [
            Input::Path(dir.path().to_string_lossy().to_string()),
            Input::Paths(names),
        ] {
            // Even one-sided/mixed identities must return an editable template.
            assert!(to_spec(&input, None, None).is_ok(), "{}", case["id"]);
            let result = load_assembled(&input, None, None);
            let package = load_package_with_limits_and_storage(
                &input,
                None,
                None,
                LoadLimits::default(),
                &NumericStorageOptions::default(),
            );
            if case["status"] == "refused" {
                assert!(result.is_err(), "{}", case["id"]);
                assert!(package.is_err(), "{}", case["id"]);
                continue;
            }
            for a in [result.unwrap(), package.unwrap().to_assembled()] {
                let y = a.blocks["train"].y.as_ref().unwrap();
                let expected: Vec<_> = case["targets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| row.as_array().unwrap())
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                assert_eq!(y.data, expected, "{}", case["id"]);
            }
        }
    }
}
#[test]
fn load_limits_reject_before_full_discovery_read() {
    let f = fixture();
    let case = f["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["id"] == "shuffled")
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (name, text) in case["files"].as_object().unwrap() {
        let path = dir.path().join(name);
        std::fs::write(&path, text.as_str().unwrap()).unwrap();
        paths.push(path.to_string_lossy().to_string());
    }
    let limits = LoadLimits {
        max_file_bytes: f["controls"]["limits"]["max_file_bytes"].as_u64().unwrap(),
        max_total_bytes: f["controls"]["limits"]["max_total_bytes"].as_u64().unwrap(),
        ..LoadLimits::default()
    };
    for input in [
        Input::Path(dir.path().to_string_lossy().to_string()),
        Input::Paths(paths),
    ] {
        let error = load_assembled_with_limits(&input, None, None, limits).unwrap_err();
        assert!(
            error.message.contains("before discovery"),
            "{}",
            error.message
        );
        assert!(load_package_with_limits_and_storage(
            &input,
            None,
            None,
            limits,
            &NumericStorageOptions::default()
        )
        .is_err());
    }
}

#[test]
fn real_automatic_numeric_ref_and_three_column_data_stay_successful() {
    for case in fixture()["controls"]["positive_counterexamples"]
        .as_array()
        .unwrap()
    {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Xcal.csv"), case["x"].as_str().unwrap()).unwrap();
        std::fs::write(dir.path().join("Ycal.csv"), case["y"].as_str().unwrap()).unwrap();
        let input = Input::Path(dir.path().to_string_lossy().to_string());
        let a = load_assembled(&input, None, None).unwrap();
        let package = load_package_with_limits_and_storage(
            &input,
            None,
            None,
            LoadLimits::default(),
            &NumericStorageOptions::default(),
        )
        .unwrap();
        let expected: Vec<_> = case["targets"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap())
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(a.blocks["train"].y.as_ref().unwrap().data, expected);
        assert_eq!(
            package.to_assembled().blocks["train"]
                .y
                .as_ref()
                .unwrap()
                .data,
            expected
        );
    }
}
