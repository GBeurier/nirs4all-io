// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Native infer -> accepted spec -> assembly, without fitting.

use nirs4all_io::core::infer::InferOptions;
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::infer::infer_path;
use nirs4all_io::materialize::assemble;
use serde_json::json;

#[test]
fn unnamed_headerless_native_inference_keeps_twenty_by_seven_hundred_cells() {
    let directory = tempfile::tempdir().unwrap();
    let mut text = String::new();
    for row in 0..20 {
        text.push_str(&format!("0.{:07}", 465041 + row * 10000));
        for _ in 1..700 {
            text.push_str(";0.2");
        }
        text.push('\n');
    }
    for name in ["data.csv", "myspectra.csv"] {
        let path = directory.path().join(name);
        std::fs::write(&path, &text).unwrap();
        let auto = infer_path(path.to_str().unwrap(), &Default::default()).unwrap();
        assert_eq!(auto.params[name]["header_ambiguous"], json!(true));
        assert_eq!(
            auto.resolved_spec.unwrap().to_value()["sources"][0]["params"]["has_header"],
            json!(true)
        );
        let choices = InferOptions {
            params: json!({"has_header": true}),
            file_params: [(name.into(), json!({"has_header": false}))]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let plan = infer_path(path.to_str().unwrap(), &choices).unwrap();
        assert_eq!(plan.params[name]["has_header"]["value"], json!(false));
        assert_eq!(
            plan.warnings
                .iter()
                .filter(|w| w.contains(name) && w.contains("has_header=false"))
                .count(),
            0
        );
        let spec = plan.resolved_spec.unwrap();
        assert_eq!(
            spec.to_value()["sources"][0]["params"]["has_header"],
            json!(false)
        );
        let data = assemble(&spec, directory.path()).unwrap();
        let block = &data.blocks["train"];
        assert_eq!(block.source_ids, vec!["data"]);
        assert_eq!(block.n_samples, 20);
        assert_eq!(block.x[0].n_rows, 20);
        assert_eq!(block.x[0].n_cols, 699);
        assert_eq!(block.x[0].data[0], 0.0465041);
        assert_eq!(block.x[0].data[19 * 699], 0.0655041);
        assert_eq!(block.y.as_ref().unwrap().data, vec![0.2; 20]);
        assert!(data.identity.sample_id.is_none());

        let mut explicit = spec.to_value();
        explicit["sources"][0]["params"]["has_header"] = json!(true);
        explicit["sources"][0]["role"] = json!("features");
        explicit["sources"][0]
            .as_object_mut()
            .unwrap()
            .remove("columns");
        let explicit = DatasetSpec::from_value(&explicit).unwrap();
        let data = assemble(&explicit, directory.path()).unwrap();
        assert_eq!(data.blocks["train"].n_samples, 19);
        assert_eq!(data.blocks["train"].x[0].data[0], 0.0475041);
        assert!(!data.warnings.iter().any(|w| w.contains("has_header=false")));
    }
}

#[test]
fn mixed_header_native_inference_preserves_scientific_contents() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens/contract/corpus/single_combined/data.csv");
    let plan = infer_path(path.to_str().unwrap(), &Default::default()).unwrap();
    assert_eq!(
        plan.params["data.csv"]["has_header"],
        json!({"value": true, "score": 0.85})
    );
    assert!(plan.axis.is_none());
    let spec = plan.resolved_spec.unwrap();
    assert_eq!(
        spec.to_value()["sources"][0]["params"]["has_header"],
        json!(true)
    );
    let data = assemble(&spec, path.parent().unwrap()).unwrap();
    let block = &data.blocks["train"];
    assert_eq!(block.n_samples, 6);
    assert_eq!(block.x[0].n_rows, 6);
    assert_eq!(block.x[0].n_cols, 12);
    assert_eq!(
        block.feature_headers[0],
        (0..12)
            .map(|i| (1000 + i * 5).to_string())
            .collect::<Vec<_>>()
    );
    for row in block.x[0].data.as_chunks::<12>().0 {
        assert_eq!(
            row,
            &[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
        );
    }
    assert_eq!(block.y_headers, vec!["protein"]);
    assert_eq!(
        block.y.as_ref().unwrap().data,
        vec![12.5, 8.3, 15.1, 9.7, 11.2, 13.8]
    );
    assert_eq!(block.header_units, vec!["nm"]);
    assert!(data.identity.sample_id.is_none());
    assert!(data.folds.is_empty());
}

#[test]
fn native_inferred_ambiguous_delimiter_still_refuses() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("Xtest.csv");
    std::fs::write(&path, "id,y\n1,2\n3,4\n").unwrap();
    let plan = infer_path(path.to_str().unwrap(), &Default::default()).unwrap();
    assert!(plan.params["Xtest.csv"]["delimiter"]["value"].is_null());
    assert!(!plan.warnings.iter().any(|w| w.contains("has_header=false")));
    let error = assemble(plan.resolved_spec.as_ref().unwrap(), directory.path()).unwrap_err();
    assert!(error.message.contains("delimiter"), "{}", error.message);
}

#[test]
fn headerless_convention_inference_keeps_all_twenty_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("Xtest-no-header.csv");
    let mut text = String::new();
    for row in 0..20 {
        text.push_str(&format!("0.{:07};0.2;0.1\n", 465041 + row * 10000));
    }
    std::fs::write(&path, text).unwrap();

    let choices = InferOptions {
        params: json!({"has_header": false}),
        ..Default::default()
    };
    let plan = infer_path(path.to_str().unwrap(), &choices).unwrap();
    assert_eq!(
        plan.params["Xtest-no-header.csv"]["has_header"]["value"],
        json!(false)
    );
    let spec = plan.resolved_spec.unwrap();
    assert_eq!(
        spec.to_value()["sources"][0]["params"]["has_header"],
        json!(false)
    );
    let assembled = assemble(&spec, directory.path()).unwrap();
    let block = &assembled.blocks["test"];
    assert_eq!(assembled.n_sources, 1);
    assert_eq!(block.source_ids, vec!["test_x"]);
    assert_eq!(block.n_samples, 20);
    assert_eq!(block.x[0].n_rows, 20);
    assert_eq!(block.x[0].n_cols, 3);
    assert_eq!(block.x[0].data.len(), 60);
    assert_eq!(block.x[0].data[0], 0.0465041);
    assert_eq!(block.x[0].data[57], 0.0655041);
    assert_eq!(block.header_units, vec!["index"]);
    assert!(assembled.identity.sample_id.is_none());
    assert!(block.y.is_none());
}
