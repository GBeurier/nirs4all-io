//! The file facade reads every case of a missing-value token as missing.
use nirs4all_io::core::infer::InferOptions;
use nirs4all_io::core::materialize::frame::Cell;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::infer::infer_path;
use nirs4all_io::materialize::assemble_with_storage;
use serde_json::json;

const MISSING: &[&str] = &[
    "", "NA", "na", "Na", "nA", "N/A", "n/a", "NaN", "nan", "Nan", "NAN", "naN", "null", "NULL",
    "None", "none", "#N/A", "#NA", "#na",
];
const TEXT: &[&str] = &[" ", "-", "?", "."];

/// `token` is one cell of the spectral `x2`, metadata `batch` and target `y` columns.
fn write_table(dir: &std::path::Path, token: &str) {
    let mut text = String::from("id,x1,x2,site,batch,y\n");
    for row in 0..8 {
        let (x2, batch, y) = if row == 3 {
            (token.to_string(), token.to_string(), token.to_string())
        } else {
            (
                format!("{}", 0.25 * row as f64),
                format!("{}", row % 3 + 1),
                format!("{}", 10.0 + 1.5 * row as f64),
            )
        };
        let site = ["A", "B"][row % 2];
        text.push_str(&format!("s{row},{},{x2},{site},{batch},{y}\n", row as f64));
    }
    std::fs::write(dir.join("data.csv"), text).unwrap();
}

fn inferred_target_role(dir: &std::path::Path) -> String {
    let options = InferOptions {
        params: json!({"delimiter": ",", "na": {"policy": "ignore"}}),
        ..InferOptions::default()
    };
    let plan = infer_path(&dir.to_string_lossy(), &options)
        .unwrap()
        .to_value();
    plan["resolved_spec"]["sources"][0]["columns"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["select"].as_array().unwrap().contains(&json!("y")))
        .map(|entry| entry["role"].as_str().unwrap().to_string())
        .unwrap_or_default()
}

#[test]
fn csv_files_read_every_case_of_a_missing_token_in_each_role() {
    let dir = tempfile::tempdir().unwrap();
    let spec = DatasetSpec::from_value(&json!({
        "task_type": "regression",
        "sources": [{"id": "t", "input": "data.csv", "columns": {
            "features": ["x1", "x2"], "targets": "y", "metadata": ["id", "site", "batch"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"delimiter": ",", "na": {"policy": "ignore"}}
    }))
    .unwrap();
    for token in MISSING {
        write_table(dir.path(), token);
        let data = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
        let block = &data.blocks["train"];
        assert!(block.y_categorical.is_empty(), "{token:?}");
        assert!(block.y.as_ref().unwrap().data[3].is_nan(), "{token:?}");
        assert_eq!(block.y.as_ref().unwrap().data[2], 13.0, "{token:?}");
        assert!(block.x[0].data[3 * 2 + 1].is_nan(), "{token:?}");
        let batch = block.metadata.as_ref().unwrap().column("batch").unwrap();
        assert!(
            matches!(batch.values[3], Cell::Float(v) if v.is_nan()),
            "{token:?}"
        );
        assert_eq!(batch.values[0], Cell::Float(1.0), "{token:?}");
        assert_eq!(inferred_target_role(dir.path()), "targets", "{token:?}");
    }
    for token in TEXT {
        write_table(dir.path(), token);
        let data = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
        assert!(!data.blocks["train"].y_categorical.is_empty(), "{token:?}");
        assert_eq!(inferred_target_role(dir.path()), "metadata", "{token:?}");
    }
}
