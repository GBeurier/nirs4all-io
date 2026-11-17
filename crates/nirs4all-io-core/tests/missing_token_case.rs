//! Missing-value tokens match without regard to ASCII case, in every role.
//!
//! One vocabulary (pandas' default set plus `NA`, `N/A` and the empty cell)
//! serves inspection, inference and materialization: `na`, `Nan` or `NULL` in an
//! otherwise numeric column is a missing value and the column stays numeric.
use nirs4all_io_core::infer::memory::{
    describe_tables, infer_named_bytes, DescribeOptions, NamedBytes,
};
use nirs4all_io_core::infer::table::ColDtype;
use nirs4all_io_core::infer::InferOptions;
use nirs4all_io_core::materialize::frame::Cell;
use nirs4all_io_core::materialize::loaders::TabularReadLimits;
use nirs4all_io_core::materialize::{
    assemble_in_memory, AssembledDataset, InMemorySource, SourcePayload,
};
use nirs4all_io_core::multimodal_import::{assemble_multimodal_files, ImportManifest};
use nirs4all_io_core::spec::DatasetSpec;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// Every declared spelling in several cases, including each case of `na`/`nan`.
const MISSING: &[&str] = &[
    "", "NA", "na", "Na", "nA", "N/A", "n/a", "N/a", "n/A", "NaN", "nan", "Nan", "NAN", "naN",
    "nAn", "NAn", "nAN", "null", "NULL", "Null", "None", "none", "NONE", "#N/A", "#n/a", "#NA",
    "#na", "#N/A N/A", "#n/a n/a", "<NA>", "<na>", "-NaN", "-nan", "-NAN", "1.#IND", "1.#ind",
    "-1.#QNAN", "-1.#qnan",
];

/// Spellings outside the declared vocabulary stay text.
const TEXT: &[&str] = &[" ", "-", "?", ".", "missing", "nd", "n.a.", " NA", "NA "];

const ROLES: [&str; 3] = ["target", "spectral", "metadata"];
const ROW: usize = 3;

/// `token` replaces one cell of the target, spectral or metadata column.
fn table(role: &str, token: &str) -> String {
    let mut text = String::from("id,x1,x2,x3,site,batch,y\n");
    for row in 0..8 {
        let cell = |column: &str, value: String| {
            if row == ROW && column == role {
                token.to_string()
            } else {
                value
            }
        };
        text.push_str(&format!(
            "s{row},{},{},{},{},{},{}\n",
            0.1 * row as f64 + 0.01,
            cell("spectral", format!("{}", 0.2 * row as f64 + 0.02)),
            0.3 * row as f64,
            ["A", "B"][row % 2],
            cell("metadata", format!("{}", row % 3 + 1)),
            cell("target", format!("{}", 10.0 + 1.5 * row as f64)),
        ));
    }
    text
}

fn assemble(text: &str) -> AssembledDataset {
    let spec = DatasetSpec::from_value(&json!({
        "task_type": "regression",
        "sources": [{"id": "t", "input": "data.csv", "columns": {
            "features": ["x1", "x2", "x3"], "targets": "y", "metadata": ["id", "site", "batch"]}}],
        "sample_index": {"by": "id", "key": "id"},
        "params": {"delimiter": ",", "na": {"policy": "ignore"}}
    }))
    .unwrap();
    let sources = [InMemorySource {
        name: "data.csv".into(),
        payload: SourcePayload::Bytes(text.as_bytes().to_vec()),
    }];
    assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap()
}

fn named(text: &str) -> Vec<NamedBytes> {
    vec![NamedBytes {
        name: "data.csv".into(),
        bytes: text.as_bytes().to_vec(),
    }]
}

/// The inspected dtype of `column`, as the browser dataset builder reads it.
fn described_dtype(text: &str, column: &str) -> String {
    let options = DescribeOptions {
        params: BTreeMap::from([("data.csv".to_string(), json!({"delimiter": ","}))]),
        distinct: vec![],
    };
    let description = &describe_tables(&named(text), &options)[0];
    assert!(description.readable, "{:?}", description.refusal);
    let column = description.columns.iter().find(|c| c.name == column);
    column.unwrap().dtype.clone()
}

/// The role inference assigns to `column` in the resolved spec.
fn inferred_role(text: &str, column: &str) -> String {
    let options = InferOptions {
        params: json!({"delimiter": ",", "na": {"policy": "ignore"}}),
        ..InferOptions::default()
    };
    let plan = infer_named_bytes(&named(text), &options)
        .unwrap()
        .to_value();
    plan["resolved_spec"]["sources"][0]["columns"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["select"].as_array().unwrap().contains(&json!(column)))
        .map(|entry| entry["role"].as_str().unwrap().to_string())
        .unwrap_or_default()
}

#[test]
fn every_case_of_a_declared_token_is_missing_in_each_role() {
    for token in MISSING {
        for role in ROLES {
            let text = table(role, token);
            let data = assemble(&text);
            let block = &data.blocks["train"];
            assert_eq!(block.n_samples, 8, "{role}/{token:?}");
            match role {
                "target" => {
                    let y = block.y.as_ref().unwrap();
                    assert!(
                        block.y_categorical.is_empty(),
                        "{token:?} encoded as a class"
                    );
                    assert!(y.data[ROW].is_nan(), "{token:?}");
                    assert_eq!(y.data[ROW - 1], 13.0, "{token:?}");
                    assert_eq!(described_dtype(&text, "y"), "numeric", "{token:?}");
                    assert_eq!(inferred_role(&text, "y"), "targets", "{token:?}");
                }
                "spectral" => {
                    assert!(block.x[0].data[ROW * 3 + 1].is_nan(), "{token:?}");
                    assert_eq!(block.x[0].data[ROW * 3], 0.1 * 3.0 + 0.01, "{token:?}");
                    assert_eq!(described_dtype(&text, "x2"), "numeric", "{token:?}");
                    assert_eq!(inferred_role(&text, "x2"), "features", "{token:?}");
                }
                _ => {
                    let batch = block.metadata.as_ref().unwrap().column("batch").unwrap();
                    assert_eq!(batch.dtype, ColDtype::Numeric, "{token:?}");
                    assert!(
                        matches!(batch.values[ROW], Cell::Float(v) if v.is_nan()),
                        "{token:?}: {:?}",
                        batch.values[ROW]
                    );
                    assert_eq!(batch.values[0], Cell::Float(1.0), "{token:?}");
                    assert_eq!(described_dtype(&text, "batch"), "numeric", "{token:?}");
                }
            }
        }
    }
}

#[test]
fn spellings_outside_the_vocabulary_stay_text() {
    for token in TEXT {
        let text = table("target", token);
        assert!(
            !assemble(&text).blocks["train"].y_categorical.is_empty(),
            "{token:?}"
        );
        assert_eq!(described_dtype(&text, "y"), "string", "{token:?}");
        assert_eq!(inferred_role(&text, "y"), "metadata", "{token:?}");

        let data = assemble(&table("metadata", token));
        let batch = data.blocks["train"].metadata.as_ref().unwrap();
        let batch = batch.column("batch").unwrap();
        assert_eq!(batch.dtype, ColDtype::String, "{token:?}");
        assert_eq!(batch.values[ROW], Cell::Str(token.to_string()), "{token:?}");
    }
}

#[test]
fn a_text_column_keeps_its_labels_and_only_the_token_cell_is_missing() {
    // The ambiguity: `na` among genuine text labels is a missing cell, never a
    // label, and it does not change the column's text type.
    for token in ["na", "Nan", "NONE"] {
        let text = format!(
            "id,x1,site,y\ns0,1,north,low\ns1,2,{token},high\ns2,3,south,{token}\ns3,4,north,low\n"
        );
        let spec = DatasetSpec::from_value(&json!({
            "sources": [{"id": "t", "input": "data.csv", "columns": {
                "features": ["x1"], "targets": "y", "metadata": ["id", "site"]}}],
            "sample_index": {"by": "id", "key": "id"},
            "params": {"delimiter": ",", "na": {"policy": "ignore"}}
        }))
        .unwrap();
        let sources = [InMemorySource {
            name: "data.csv".into(),
            payload: SourcePayload::Bytes(text.into_bytes()),
        }];
        let data = assemble_in_memory(&spec, &sources, &HashMap::new(), None).unwrap();
        let block = &data.blocks["train"];
        let site = block.metadata.as_ref().unwrap().column("site").unwrap();
        assert_eq!(site.dtype, ColDtype::String, "{token}");
        assert_eq!(site.values[0], Cell::Str("north".into()), "{token}");
        assert_eq!(site.values[1], Cell::Na, "{token}");
        // The target stays a two-class label column with one missing sample.
        let classes: &Value = &block.y_categorical["y"];
        assert!(!classes.to_string().contains(token), "{token}: {classes}");
        assert!(block.y.as_ref().unwrap().data[2].is_nan(), "{token}");
    }
}

#[test]
fn multimodal_numeric_csv_missing_tokens_ignore_case() {
    let manifest: ImportManifest = serde_json::from_value(json!({
        "schema": "nirs4all.multimodal-import.v1", "caseId": "genetics", "name": "fixture",
        "metadata": {"file": "meta.csv", "sampleId": "sample_id", "target": "y", "targetUnit": "a.u."},
        "sources": [{"id": "markers", "label": "Markers", "modality": "genetics",
            "profile": "numeric_csv", "file": "x.csv", "sampleId": "sample_id",
            "features": ["m0", "m1", "m2"]}]
    }))
    .unwrap();
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    let load = |cells: &str| {
        let files = BTreeMap::from([
            ("meta.csv".into(), b"sample_id,y\na,1\nb,2\n".to_vec()),
            (
                "x.csv".into(),
                format!("sample_id,m0,m1,m2\na,0,1,2\nb,{cells}\n").into_bytes(),
            ),
        ]);
        assemble_multimodal_files(&manifest, &files, &BTreeMap::new(), limits)
    };
    for cells in ["NA,NaN,N/A", "na,nan,n/a", "Na,Nan,N/a", "nA,NAN,n/A"] {
        let assembled = load(cells).unwrap_or_else(|error| panic!("{cells}: {error}"));
        assert_eq!(assembled.sources[0].validity, [1, 1, 1, 0, 0, 0], "{cells}");
    }
    // This operation keeps its own narrower declared vocabulary.
    assert!(load("null,1,2").is_err());
}
