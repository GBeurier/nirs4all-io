// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Parquet materialization must decode raw frames first, then apply source params.

use std::sync::Arc;

use arrow_array::{ArrayRef, Date32Array, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use nirs4all_io::core::spec::{normalize_to_spec_dict, DatasetSpec};
use nirs4all_io::materialize::assemble;
use parquet::arrow::arrow_writer::ArrowWriter;
use serde_json::{json, Value};

fn build_spec(spec_json: &Value) -> DatasetSpec {
    DatasetSpec::from_value(&normalize_to_spec_dict(spec_json)).expect("spec parses")
}

fn write_parquet(path: &std::path::Path, batch: RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn parquet_assemble_applies_source_na_policy_without_default_abort() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("1005", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0)])) as ArrayRef,
            Arc::new(Float64Array::from(vec![None, Some(3.0)])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-na",
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet",
            "params": {
                "na": {"policy": "replace", "fill": {"method": "value", "fill_value": 7.0}}
            },
        }],
    }));

    let assembled = assemble(&spec, dir.path()).expect("assemble parquet with source NA policy");
    let block = &assembled.blocks["train"];

    assert_eq!(block.feature_headers, vec![vec!["1000", "1005"]]);
    assert_eq!(block.x[0].data, vec![1.0, 7.0, 2.0, 3.0]);
}

#[test]
fn parquet_assemble_skips_unsupported_unselected_column() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("unsupported_date", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Date32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-projected",
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet",
            "params": {"format": {"columns": ["1000"]}},
        }],
    }));

    let assembled = assemble(&spec, dir.path()).expect("assemble projected parquet");
    let block = &assembled.blocks["train"];

    assert_eq!(block.feature_headers, vec![vec!["1000"]]);
    assert_eq!(block.x[0].data, vec![1.0, 2.0]);
}

#[test]
fn parquet_assemble_shared_file_with_different_selected_columns() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("1005", DataType::Float64, false),
        Field::new("unused_date", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
            Arc::new(Date32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-shared-projected",
        "sources": [
            {
                "id": "left",
                "role": "features",
                "input": "scan.parquet",
                "params": {"format": {"columns": ["1000"]}},
            },
            {
                "id": "right",
                "role": "features",
                "input": "scan.parquet",
                "params": {"format": {"columns": ["1005"]}},
            },
        ],
    }));

    let assembled = assemble(&spec, dir.path()).expect("assemble shared projected parquet");
    let block = &assembled.blocks["train"];

    assert_eq!(block.feature_headers, vec![vec!["1000"], vec!["1005"]]);
    assert_eq!(block.x[0].data, vec![1.0, 2.0]);
    assert_eq!(block.x[1].data, vec![10.0, 20.0]);
}

#[test]
fn parquet_projection_union_includes_source_and_variation_columns() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("1005", DataType::Float64, false),
        Field::new("snv1000", DataType::Float64, false),
        Field::new("snv1005", DataType::Float64, false),
        Field::new("unused_date", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 20.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.1, 0.2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.3, 0.4])) as ArrayRef,
            Arc::new(Date32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-variation-projected",
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet",
            "params": {"format": {"columns": ["1000", "1005"]}},
            "variations": [{
                "name": "snv",
                "input": "scan.parquet",
                "params": {"format": {"columns": ["snv1000", "snv1005"]}}
            }]
        }],
    }));

    let assembled = assemble(&spec, dir.path()).expect("assemble variation projected parquet");
    let block = &assembled.blocks["train"];

    assert_eq!(block.feature_headers, vec![vec!["1000", "1005"]]);
    assert_eq!(block.x[0].data, vec![1.0, 10.0, 2.0, 20.0]);
    assert_eq!(block.processings.len(), 1);
    assert_eq!(block.processings[0][0].0, "snv");
    assert_eq!(block.processings[0][0].1.data, vec![0.1, 0.3, 0.2, 0.4]);
}

#[test]
fn parquet_projection_falls_back_to_all_columns_when_any_use_is_unprojected() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("unused_date", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Date32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let spec = build_spec(&json!({
        "name": "parquet-fallback-all",
        "sources": [
            {
                "id": "left",
                "role": "features",
                "input": "scan.parquet",
                "params": {"format": {"columns": ["1000"]}},
            },
            {
                "id": "right",
                "role": "features",
                "input": "scan.parquet"
            },
        ],
    }));

    let err = assemble(&spec, dir.path()).expect_err("unprojected use must force full read");

    assert!(
        err.message
            .contains("parquet column 'unused_date' has unsupported type Date32"),
        "{}",
        err.message
    );
}

#[test]
fn parquet_projection_honors_global_columns_and_null_override() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let path = dir.path().join("scan.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("1000", DataType::Float64, false),
        Field::new("unused_date", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            Arc::new(Date32Array::from(vec![1, 2])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&path, batch);

    let projected = build_spec(&json!({
        "name": "parquet-global-projected",
        "params": {"format": {"columns": ["1000"]}},
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet"
        }],
    }));
    let assembled =
        assemble(&projected, dir.path()).expect("assemble with global parquet projection");
    let block = &assembled.blocks["train"];
    assert_eq!(block.feature_headers, vec![vec!["1000"]]);
    assert_eq!(block.x[0].data, vec![1.0, 2.0]);

    let fallback = build_spec(&json!({
        "name": "parquet-global-null-fallback",
        "params": {"format": {"columns": ["1000"]}},
        "sources": [{
            "id": "x",
            "role": "features",
            "input": "scan.parquet",
            "params": {"format": {"columns": null}}
        }],
    }));
    let err = assemble(&fallback, dir.path()).expect_err("source null columns forces full read");
    assert!(
        err.message
            .contains("parquet column 'unused_date' has unsupported type Date32"),
        "{}",
        err.message
    );
}

#[test]
fn declared_f32_storage_is_captured_before_widening_with_subnormals_and_missingness() {
    use arrow_array::Float32Array;
    use nirs4all_io::materialize::{DatasetPackage, NumericStorage, PrecisionOrigin};
    let dir = tempfile::tempdir().unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float32, true)])),
        vec![Arc::new(Float32Array::from(vec![
            Some(f32::from_bits(1)),
            Some(-0.0),
            None,
        ])) as ArrayRef],
    )
    .unwrap();
    write_parquet(&dir.path().join("x.parquet"), batch);
    let spec = build_spec(
        &json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"x","role":"features","input":"x.parquet"}]}),
    );
    let assembled = assemble(&spec, dir.path()).unwrap();
    let package = DatasetPackage::from_assembled(&assembled).unwrap();
    let block = &package.to_assembled().blocks["train"].x[0];
    assert_eq!(block.precision.source_dtype.as_deref(), Some("float32"));
    assert_eq!(block.precision.dtype, NumericStorage::Float32);
    assert_eq!(block.precision.origin, PrecisionOrigin::SourceDeclared);
    assert_eq!(
        block.data[0].to_bits(),
        f64::from(f32::from_bits(1)).to_bits()
    );
    assert_eq!(block.data[1].to_bits(), 0x8000000000000000);
    assert_eq!(block.data[2].to_bits(), 0);
    assert_eq!(block.validity, vec![true, true, false]);
}

#[test]
fn typed_integer_requests_reach_original_parquet_cells_before_f64_coercion() {
    use arrow_array::{Int64Array, StringArray};
    use nirs4all_io::materialize::{
        assemble_with_storage, DatasetPackage, NumericStorage, NumericStorageOptions,
    };
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    let dir = tempfile::tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Int64, true),
        Field::new("w", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["s1", "s2"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![integer, 1])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(integer), None])) as ArrayRef,
            Arc::new(Int64Array::from(vec![integer, 1])) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&dir.path().join("data.parquet"), batch);
    let spec = build_spec(
        &json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"table","input":"data.parquet","columns":{"metadata":"id","features":"x","targets":"y","weights":"w"}}],"sample_index":{"by":"id","key":"id"}}),
    );
    assert!(assemble(&spec, dir.path())
        .unwrap_err()
        .message
        .contains("exactly representable"));
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    let assembled = assemble_with_storage(&spec, dir.path(), &options).unwrap();
    assert_eq!(
        assembled.audits,
        vec![
            json!({"source":"table","na":{"strategy":"ignore","na_detected":true,"na_count":1,"na_samples":1,"na_features":1,"removed_samples":[],"removed_features":[],"fill_method":null,"na_preserved":true}})
        ]
    );
    let package = DatasetPackage::from_assembled_with_storage(&assembled, &options).unwrap();
    let assembled = package.to_assembled();
    let block = &assembled.blocks["train"];
    for value in [
        block.x[0].data[0],
        block.y.as_ref().unwrap().data[0],
        block.weights.as_ref().unwrap()[0],
    ] {
        assert_eq!((value as f32).to_bits(), 0x5a800001);
    }
    assert_eq!(block.x[0].precision.source_dtype.as_deref(), Some("int64"));
    assert_eq!(
        block.y.as_ref().unwrap().precision.source_dtype.as_deref(),
        Some("int64")
    );
    assert_eq!(
        block.weights_precision.source_dtype.as_deref(),
        Some("int64")
    );
}

#[test]
fn independently_typed_processing_sources_preserve_their_own_precision() {
    use arrow_array::Float32Array;
    use nirs4all_io::materialize::{
        assemble_with_storage, DatasetPackage, NumericStorage, NumericStorageOptions,
        PrecisionOrigin,
    };
    let dir = tempfile::tempdir().unwrap();
    let f32_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float32, false)])),
        vec![Arc::new(Float32Array::from(vec![1.0, -0.0])) as ArrayRef],
    )
    .unwrap();
    let f64_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
        vec![Arc::new(Float64Array::from(vec![
            f64::from_bits(0x3ff0000000000001),
            -0.0,
        ])) as ArrayRef],
    )
    .unwrap();
    write_parquet(&dir.path().join("f32.parquet"), f32_batch);
    write_parquet(&dir.path().join("f64.parquet"), f64_batch);
    for (base, child, base_dtype, child_dtype) in [
        ("f32.parquet", "f64.parquet", "float32", "float64"),
        ("f64.parquet", "f32.parquet", "float64", "float32"),
    ] {
        let spec = build_spec(
            &json!({"sources":[{"id":"x","role":"features","input":base,"variations":[{"name":"processed","input":child}]}]}),
        );
        let raw = assemble(&spec, dir.path()).unwrap();
        assert!(DatasetPackage::from_assembled(&raw)
            .unwrap_err()
            .contains("common features storage request"));
        for dtype in [NumericStorage::Float64, NumericStorage::Float32] {
            let options = NumericStorageOptions {
                features: Some(dtype),
                ..Default::default()
            };
            let assembled = assemble_with_storage(&spec, dir.path(), &options).unwrap();
            let package =
                DatasetPackage::from_assembled_with_storage(&assembled, &options).unwrap();
            let roundtrip = package.to_assembled();
            let block = &roundtrip.blocks["train"];
            assert_eq!(
                block.x[0].precision.source_dtype.as_deref(),
                Some(base_dtype)
            );
            assert_eq!(
                block.processings[0][0].1.precision.source_dtype.as_deref(),
                Some(child_dtype)
            );
            assert_eq!(block.x[0].precision.origin, PrecisionOrigin::Requested);
            assert_eq!(
                block.processings[0][0].1.precision.origin,
                PrecisionOrigin::Requested
            );
            let actual_child = block.processings[0][0].1.data[0];
            let expected = if dtype == NumericStorage::Float64 && child_dtype == "float64" {
                0x3ff0000000000001
            } else {
                0x3ff0000000000000
            };
            assert_eq!(actual_child.to_bits(), expected);
            assert_eq!(
                block.processings[0][0].1.data[1].to_bits(),
                0x8000000000000000
            );
            let full = roundtrip.to_full_value().unwrap();
            assert_eq!(
                full["blocks"]["train"]["processings"][0][0]["matrix"]["precision"]["source_dtype"],
                child_dtype
            );
            let rebuilt = DatasetPackage::from_assembled(&roundtrip).unwrap();
            assert_eq!(rebuilt.manifest(), package.manifest());
        }
        let options = NumericStorageOptions {
            consumer: Some("SpectroDataset".into()),
            ..Default::default()
        };
        let assembled = assemble_with_storage(&spec, dir.path(), &options).unwrap();
        let package = DatasetPackage::from_assembled_with_storage(&assembled, &options).unwrap();
        let ir = package.to_assembled();
        let child = &ir.blocks["train"].processings[0][0].1;
        assert_eq!(child.precision.source_dtype.as_deref(), Some(child_dtype));
        assert_eq!(child.precision.dtype, NumericStorage::Float32);
        assert_eq!(child.precision.origin, PrecisionOrigin::Consumer);
        assert_eq!(child.data[0].to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn same_storage_processing_keeps_declared_and_undeclared_origins_distinct() {
    use nirs4all_io::materialize::{DatasetPackage, PrecisionOrigin};
    let dir = tempfile::tempdir().unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
        vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef],
    )
    .unwrap();
    write_parquet(&dir.path().join("typed.parquet"), batch);
    std::fs::write(dir.path().join("untyped.csv"), "x\n1.0\n2.0\n").unwrap();
    for (base, child, base_origin, child_origin) in [
        (
            "typed.parquet",
            "untyped.csv",
            PrecisionOrigin::SourceDeclared,
            PrecisionOrigin::Default,
        ),
        (
            "untyped.csv",
            "typed.parquet",
            PrecisionOrigin::Default,
            PrecisionOrigin::SourceDeclared,
        ),
    ] {
        let spec = build_spec(
            &json!({"sources":[{"id":"x","role":"features","input":base,"variations":[{"name":"processed","input":child}]}]}),
        );
        let assembled = assemble(&spec, dir.path()).unwrap();
        let package = DatasetPackage::from_assembled(&assembled).unwrap();
        let roundtrip = package.to_assembled();
        let block = &roundtrip.blocks["train"];
        assert_eq!(block.x[0].precision.origin, base_origin);
        assert_eq!(block.processings[0][0].1.precision.origin, child_origin);
        assert_ne!(
            block.x[0].precision.source_dtype,
            block.processings[0][0].1.precision.source_dtype
        );
        assert_eq!(
            package.manifest(),
            DatasetPackage::from_assembled(&roundtrip)
                .unwrap()
                .manifest()
        );
    }
}

#[test]
fn selected_precision_roundtrips_all_roles_processings_and_masks() {
    use arrow_array::StringArray;
    use nirs4all_io::materialize::{
        assemble_with_storage, DatasetPackage, NumericStorage, NumericStorageOptions,
        PrecisionOrigin,
    };
    let dir = tempfile::tempdir().unwrap();
    let values = vec![Some(1.0 + 2_f64.powi(-30)), Some(-0.0), None];
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
            Field::new("w", DataType::Float64, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["s1", "s2", "s3"])) as ArrayRef,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&dir.path().join("data.parquet"), batch);
    let child = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(values)) as ArrayRef],
    )
    .unwrap();
    write_parquet(&dir.path().join("child.parquet"), child);
    let spec = build_spec(
        &json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"table","input":"data.parquet","columns":{"metadata":"id","features":"x","targets":"y","weights":"w"},"variations":[{"name":"child","input":"child.parquet"}]}],"sample_index":{"by":"id","key":"id"}}),
    );
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    let assembled = assemble_with_storage(&spec, dir.path(), &options).unwrap();
    let package = DatasetPackage::from_assembled_with_storage(&assembled, &options).unwrap();
    let ir = package.to_assembled();
    let b = &ir.blocks["train"];
    for matrix in [&b.x[0], &b.processings[0][0].1, b.y.as_ref().unwrap()] {
        assert_eq!(matrix.precision.origin, PrecisionOrigin::Requested);
        assert_eq!(matrix.precision.source_dtype.as_deref(), Some("float64"));
        assert_eq!(
            matrix.precision.requested_dtype,
            Some(NumericStorage::Float32)
        );
        assert_eq!(
            matrix.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![1.0_f64.to_bits(), (-0.0_f64).to_bits(), 0]
        );
        assert_eq!(matrix.validity, vec![true, true, false]);
    }
    assert_eq!(b.weights_precision, b.x[0].precision);
    assert_eq!(
        b.weights_validity.as_ref().unwrap(),
        &vec![true, true, false]
    );
    assert_eq!(
        b.weights
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![1.0_f64.to_bits(), (-0.0_f64).to_bits(), 0]
    );
    let before = package.executed_projection(&["table".into()]).unwrap();
    let rebuilt = DatasetPackage::from_assembled(&ir).unwrap();
    assert_eq!(package.manifest(), rebuilt.manifest());
    assert_eq!(
        package.to_full_value().unwrap(),
        rebuilt.to_full_value().unwrap()
    );
    assert_eq!(
        before,
        rebuilt.executed_projection(&["table".into()]).unwrap()
    );
    assert_eq!(
        ir.to_full_value().unwrap(),
        rebuilt.to_assembled().to_full_value().unwrap()
    );
    // Reselecting a carried F32 block as F64 cannot recover source precision.
    let widened = DatasetPackage::from_assembled_with_storage(
        &ir,
        &NumericStorageOptions {
            features: Some(NumericStorage::Float64),
            targets: Some(NumericStorage::Float64),
            weights: Some(NumericStorage::Float64),
            consumer: None,
        },
    )
    .unwrap_err();
    assert!(widened.contains("Cannot re-select float64"));
}

#[test]
fn unsigned_parquet_outside_existing_cell_range_refuses_before_any_float_cast() {
    use arrow_array::UInt64Array;
    use nirs4all_io::materialize::{assemble_with_storage, NumericStorage, NumericStorageOptions};
    let dir = tempfile::tempdir().unwrap();
    for integer in [1_u64 << 63, (1_u64 << 63) + 1] {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::UInt64, false)])),
            vec![Arc::new(UInt64Array::from(vec![integer])) as ArrayRef],
        )
        .unwrap();
        write_parquet(&dir.path().join("uint.parquet"), batch);
        let spec =
            build_spec(&json!({"sources":[{"id":"x","role":"features","input":"uint.parquet"}]}));
        for request in [
            None,
            Some(NumericStorage::Float32),
            Some(NumericStorage::Float64),
        ] {
            let error = assemble_with_storage(
                &spec,
                dir.path(),
                &NumericStorageOptions {
                    features: request,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(error.message.contains(&integer.to_string()));
            assert!(error.message.contains("outside i64 range"));
        }
    }
}

#[test]
fn genuine_typed_f64_parquet_buffers_keep_their_existing_rounded_bits() {
    use nirs4all_io::materialize::{assemble_with_storage, NumericStorage, NumericStorageOptions};
    let dir = tempfile::tempdir().unwrap();
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    // These are actual Float64Array buffers, not retained integer input cells.
    let values = vec![9_007_199_254_740_993_i64 as f64, integer as f64];
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("w", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
            Arc::new(Float64Array::from(values.clone())) as ArrayRef,
        ],
    )
    .unwrap();
    write_parquet(&dir.path().join("host.parquet"), batch);
    let spec = build_spec(
        &json!({"sources":[{"id":"table","input":"host.parquet","columns":{"features":"x","targets":"y","weights":"w"}}]}),
    );
    let assembled = assemble(&spec, dir.path()).unwrap();
    let b = &assembled.blocks["train"];
    for data in [
        &b.x[0].data,
        &b.y.as_ref().unwrap().data,
        b.weights.as_ref().unwrap(),
    ] {
        assert_eq!(
            data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    let projected = assemble_with_storage(&spec, dir.path(), &options).unwrap();
    let b = &projected.blocks["train"];
    for value in [
        b.x[0].data[1],
        b.y.as_ref().unwrap().data[1],
        b.weights.as_ref().unwrap()[1],
    ] {
        assert_eq!((value as f32).to_bits(), 0x5a800000);
    }
}

#[test]
fn nullable_int64_roles_preserve_original_cells_for_direct_f32_and_missing_masks() {
    use arrow_array::Int64Array;
    use nirs4all_io::materialize::{
        assemble_with_storage, DatasetPackage, NumericStorage, NumericStorageOptions,
    };
    let dir = tempfile::tempdir().unwrap();
    let integer = (1_i64 << 54) + (1_i64 << 30) + 1;
    let options = NumericStorageOptions {
        features: Some(NumericStorage::Float32),
        targets: Some(NumericStorage::Float32),
        weights: Some(NumericStorage::Float32),
        consumer: None,
    };
    for selected_role in ["features", "targets", "weights"] {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int64, true),
                Field::new("y", DataType::Int64, true),
                Field::new("w", DataType::Int64, true),
            ])),
            ["features", "targets", "weights"]
                .iter()
                .map(|role| {
                    Arc::new(Int64Array::from(vec![
                        Some(if *role == selected_role { integer } else { 1 }),
                        None,
                    ])) as ArrayRef
                })
                .collect(),
        )
        .unwrap();
        write_parquet(&dir.path().join("nullable.parquet"), batch);
        let spec = build_spec(
            &json!({"task_type":"regression","params":{"na":{"policy":"ignore"}},"sources":[{"id":"table","input":"nullable.parquet","columns":{"features":"x","targets":"y","weights":"w"}}]}),
        );
        assert!(
            assemble(&spec, dir.path())
                .unwrap_err()
                .message
                .contains("exactly representable"),
            "{selected_role}"
        );
        let assembled = assemble_with_storage(&spec, dir.path(), &options).unwrap();
        let package = DatasetPackage::from_assembled(&assembled).unwrap();
        let rebuilt = package.to_assembled();
        let b = &rebuilt.blocks["train"];
        for (role, values, validity, precision) in [
            (
                "features",
                &b.x[0].data,
                &b.x[0].validity,
                &b.x[0].precision,
            ),
            (
                "targets",
                &b.y.as_ref().unwrap().data,
                &b.y.as_ref().unwrap().validity,
                &b.y.as_ref().unwrap().precision,
            ),
            (
                "weights",
                b.weights.as_ref().unwrap(),
                b.weights_validity.as_ref().unwrap(),
                &b.weights_precision,
            ),
        ] {
            assert_eq!(
                (values[0] as f32).to_bits(),
                if role == selected_role {
                    0x5a800001
                } else {
                    1_f32.to_bits()
                }
            );
            assert_eq!(values[1].to_bits(), 0);
            assert_eq!(validity, &[true, false]);
            assert_eq!(precision.source_dtype.as_deref(), Some("int64"));
            assert_eq!(precision.requested_dtype, Some(NumericStorage::Float32));
        }
    }
    // Already-decoded nullable F64 holds its actual bits, including observed zero.
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(vec![
            Some(-0.0),
            None,
            Some(integer as f64),
        ])) as ArrayRef],
    )
    .unwrap();
    write_parquet(&dir.path().join("binary-zero.parquet"), batch);
    let spec = build_spec(
        &json!({"params":{"na":{"policy":"ignore"}},"sources":[{"id":"x","role":"features","input":"binary-zero.parquet"}]}),
    );
    let assembled = assemble(&spec, dir.path()).unwrap();
    let canonical = DatasetPackage::from_assembled(&assembled)
        .unwrap()
        .to_assembled();
    let x = &canonical.blocks["train"].x[0];
    assert_eq!(x.validity, [true, false, true]);
    assert_eq!(x.data[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(x.data[1].to_bits(), 0);
    assert_eq!(x.data[2].to_bits(), (integer as f64).to_bits());
}
