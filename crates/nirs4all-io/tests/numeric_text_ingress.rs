//! Exercise the file facade and its shared in-memory decoder with original numeric text.
use nirs4all_io::core::materialize::{
    assemble_in_memory_with_storage, InMemorySource, SourcePayload,
};
use nirs4all_io::core::spec::DatasetSpec;
use nirs4all_io::materialize::{assemble_with_storage, NumericStorage, NumericStorageOptions};
use serde_json::{json, Value};

fn spec(value: Value) -> DatasetSpec {
    DatasetSpec::from_value(&value).unwrap()
}

#[test]
fn csv_numeric_roles_refuse_decimal_underflow_without_reclassifying_it_as_a_label() {
    let dir = tempfile::tempdir().unwrap();
    let spec = spec(
        json!({"task_type":"regression","sources":[{"id":"table","input":"data.csv","columns":{"features":"x","targets":"y","weights":"w","metadata":"id"}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":",","na":{"policy":"ignore"}}}),
    );
    for token in ["1e-400", "2e-324", "-1e-400", "-2e-324"] {
        for role in 0..3 {
            let mut values = ["1", "2", "3"];
            values[role] = token;
            let csv = format!(
                "id,x,y,w\ns1,\"{}\",\"{}\",\"{}\"\ns2,NA,NA,NA\n",
                values[0], values[1], values[2]
            );
            std::fs::write(dir.path().join("data.csv"), &csv).unwrap();
            let memory = [InMemorySource {
                name: "data.csv".into(),
                payload: SourcePayload::Bytes(csv.into_bytes()),
            }];
            for storage in [NumericStorage::Float64, NumericStorage::Float32] {
                let options = NumericStorageOptions {
                    features: Some(storage),
                    targets: Some(storage),
                    weights: Some(storage),
                    consumer: None,
                };
                let file_error = assemble_with_storage(&spec, dir.path(), &options).unwrap_err();
                let memory_error = assemble_in_memory_with_storage(
                    &spec,
                    &memory,
                    &Default::default(),
                    None,
                    &options,
                )
                .unwrap_err();
                assert!(
                    file_error.message.contains("rounds to zero"),
                    "{role}/{token}: {file_error}"
                );
                assert_eq!(file_error.message, memory_error.message);
            }
        }
    }
    // A true zero with an arbitrarily small exponent is observed; NA is absent.
    let csv = "id,x,y,w\ns1,\"-0.000e-999\",\"0e5\",\"0.000e-999\"\ns2,NA,NA,NA\n";
    std::fs::write(dir.path().join("data.csv"), csv).unwrap();
    let assembled = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
    let canonical = nirs4all_io::materialize::DatasetPackage::from_assembled(&assembled)
        .unwrap()
        .to_assembled();
    let b = &canonical.blocks["train"];
    assert_eq!(b.x[0].data[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(b.y.as_ref().unwrap().data[0].to_bits(), 0);
    assert_eq!(b.weights.as_ref().unwrap()[0].to_bits(), 0);
    assert_eq!(b.x[0].validity, [true, false]);
    assert_eq!(b.y.as_ref().unwrap().validity, [true, false]);
    assert_eq!(b.weights_validity.as_ref().unwrap(), &[true, false]);
    assert_eq!(b.x[0].data[1].to_bits(), 0);
}

#[test]
fn csv_string_metadata_and_existing_classification_codebook_preserve_numeric_looking_text() {
    let dir = tempfile::tempdir().unwrap();
    let csv = "id,x,label,note\ns1,1,\"1e-400\",\"2e-324\"\ns2,2,control,ordinary\n";
    std::fs::write(dir.path().join("data.csv"), csv).unwrap();
    let spec = spec(
        json!({"task_type":"binary","sources":[{"id":"table","input":"data.csv","columns":{"features":"x","targets":"label","metadata":["id","note"]}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":","}}),
    );
    let assembled = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
    let b = &assembled.blocks["train"];
    assert!(serde_json::to_string(&b.y_categorical)
        .unwrap()
        .contains("1e-400"));
    assert_eq!(
        b.metadata.as_ref().unwrap().column("note").unwrap().values[0].to_str_scalar(),
        "2e-324"
    );
}

#[test]
fn csv_merges_retain_original_text_and_integer_cells_until_selected_conversion() {
    let dir = tempfile::tempdir().unwrap();
    for merge in ["by_key", "concat_features"] {
        let spec = spec(
            json!({"task_type":"regression","sources":[{"id":"table","input":["a.csv","b.csv"],"merge":merge,"key":"id","columns":{"features":"x","targets":"y","weights":"w","metadata":"id"}}],"sample_index":{"by":"id","key":"id"},"params":{"delimiter":",","na":{"policy":"ignore"}}}),
        );
        for token in ["2e-324", "18014399583223809"] {
            let a = format!("id,x\ns1,{token}\ns2,NA\n");
            let b = format!("id,y,w\ns2,NA,NA\ns1,{token},{token}\n");
            std::fs::write(dir.path().join("a.csv"), &a).unwrap();
            std::fs::write(dir.path().join("b.csv"), &b).unwrap();
            let memory = [
                InMemorySource {
                    name: "a.csv".into(),
                    payload: SourcePayload::Bytes(a.into_bytes()),
                },
                InMemorySource {
                    name: "b.csv".into(),
                    payload: SourcePayload::Bytes(b.into_bytes()),
                },
            ];
            assert!(assemble_with_storage(&spec, dir.path(), &Default::default()).is_err());
            let f32 = NumericStorageOptions {
                features: Some(NumericStorage::Float32),
                targets: Some(NumericStorage::Float32),
                weights: Some(NumericStorage::Float32),
                consumer: None,
            };
            let file = assemble_with_storage(&spec, dir.path(), &f32);
            let memory =
                assemble_in_memory_with_storage(&spec, &memory, &Default::default(), None, &f32);
            if token == "2e-324" {
                assert!(file.unwrap_err().message.contains("rounds to zero"));
                assert!(memory.unwrap_err().message.contains("rounds to zero"));
            } else {
                let file = file.unwrap();
                assert_eq!(
                    file.to_full_value().unwrap(),
                    memory.unwrap().to_full_value().unwrap()
                );
                let b = &file.blocks["train"];
                for value in [
                    b.x[0].data[0],
                    b.y.as_ref().unwrap().data[0],
                    b.weights.as_ref().unwrap()[0],
                ] {
                    assert_eq!((value as f32).to_bits(), 0x5a800001);
                }
                assert_eq!(b.x[0].validity, [true, false]);
            }
        }
    }
}

#[test]
fn merged_numeric_dtype_selection_does_not_drop_retained_unsafe_lexemes() {
    let dir = tempfile::tempdir().unwrap();
    for merge in ["by_key", "concat_features", "concat_samples"] {
        let spec = spec(
            json!({"sources":[{"id":"table","input":["a.csv","b.csv"],"merge":merge,"key":"id","columns":{"features":{"dtype":"numeric"},"metadata":"rest"}}],"params":{"delimiter":",","na":{"policy":"ignore"}}}),
        );
        for token in ["2e-324", "18014399583223809"] {
            std::fs::write(
                dir.path().join("a.csv"),
                format!("id,x\ns1,{token}\ns2,NA\n"),
            )
            .unwrap();
            std::fs::write(
                dir.path().join("b.csv"),
                if merge == "concat_samples" {
                    "id,x\ns3,1\ns4,2\n"
                } else {
                    "id,control\ns1,1\ns2,2\n"
                },
            )
            .unwrap();
            let default = assemble_with_storage(&spec, dir.path(), &Default::default());
            assert!(
                default.is_err(),
                "{merge}/{token} was silently excluded by numeric dtype selection"
            );
            if token != "2e-324" {
                let assembled = assemble_with_storage(
                    &spec,
                    dir.path(),
                    &NumericStorageOptions {
                        features: Some(NumericStorage::Float32),
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(
                    (assembled.blocks["train"].x[0].data[0] as f32).to_bits(),
                    0x5a800001
                );
            }
        }
    }
}

fn write_text_workbook(path: &std::path::Path, values: &[&str; 3]) {
    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let cells = ["x", "y", "w"]
        .iter()
        .enumerate()
        .map(|(i, header)| {
            let column = char::from(b'A' + i as u8);
            format!("<c r=\"{column}1\" t=\"inlineStr\"><is><t>{header}</t></is></c>")
        })
        .collect::<String>();
    let row = values
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let column = char::from(b'A' + i as u8);
            format!("<c r=\"{column}2\" t=\"inlineStr\"><is><t>{text}</t></is></c>")
        })
        .collect::<String>();
    let sheet = format!("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData><row r=\"1\">{cells}</row><row r=\"2\">{row}</row></sheetData></worksheet>");
    for (name, text) in [
        (
            "[Content_Types].xml",
            r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        (
            "xl/workbook.xml",
            r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="data" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
        ),
        ("xl/worksheets/sheet1.xml", sheet.as_str()),
    ] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(text.as_bytes()).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn excel_text_numeric_roles_reach_the_same_checked_converter() {
    let dir = tempfile::tempdir().unwrap();
    let spec = spec(
        json!({"task_type":"regression","sources":[{"id":"table","input":"text.xlsx","columns":{"features":"x","targets":"y","weights":"w"}}]}),
    );
    for token in ["1e-400", "-2e-324", "9007199254740993"] {
        for role in 0..3 {
            let mut values = ["1.0", "2.0", "3.0"];
            values[role] = token;
            write_text_workbook(&dir.path().join("text.xlsx"), &values);
            let error = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap_err();
            assert!(
                error.message.contains("rounds to zero")
                    || error.message.contains("exactly representable"),
                "{role}/{token}: {error}"
            );
        }
    }
    write_text_workbook(
        &dir.path().join("text.xlsx"),
        &["-0.000e-999", "0e5", "0.000e-999"],
    );
    let assembled = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
    assert_eq!(
        assembled.blocks["train"].x[0].data[0].to_bits(),
        (-0.0_f64).to_bits()
    );
    write_text_workbook(
        &dir.path().join("text.xlsx"),
        &[
            "18014399583223809",
            "18014399583223809",
            "18014399583223809",
        ],
    );
    let selected = assemble_with_storage(
        &spec,
        dir.path(),
        &NumericStorageOptions {
            features: Some(NumericStorage::Float32),
            targets: Some(NumericStorage::Float32),
            weights: Some(NumericStorage::Float32),
            consumer: None,
        },
    )
    .unwrap();
    let b = &selected.blocks["train"];
    for value in [
        b.x[0].data[0],
        b.y.as_ref().unwrap().data[0],
        b.weights.as_ref().unwrap()[0],
    ] {
        assert_eq!((value as f32).to_bits(), 0x5a800001);
    }
}

#[test]
fn explicit_numeric_na_fill_retains_unsigned_integer_until_storage_selection() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("fill.csv"), "x,control\n1,1\nNA,1\n").unwrap();
    let n = (1_u64 << 63) + (1_u64 << 39) + 1;
    let make_spec = |value| {
        spec(
            json!({"params":{"delimiter":",","na":{"policy":"replace","fill":{"method":"value","fill_value":value}}},"sources":[{"id":"x","input":"fill.csv","columns":{"features":{"dtype":"numeric"}}}]}),
        )
    };
    let inexact = make_spec(n);
    assert!(
        assemble_with_storage(&inexact, dir.path(), &Default::default())
            .unwrap_err()
            .message
            .contains("exactly representable")
    );
    let selected = assemble_with_storage(
        &inexact,
        dir.path(),
        &NumericStorageOptions {
            features: Some(NumericStorage::Float32),
            ..Default::default()
        },
    )
    .unwrap();
    let x = &selected.blocks["train"].x[0];
    assert_eq!((x.data[2] as f32).to_bits(), 0x5f000001);
    assert_eq!(x.validity, [true, true, true, true]);
    let exact =
        assemble_with_storage(&make_spec(1_u64 << 63), dir.path(), &Default::default()).unwrap();
    assert_eq!(
        exact.blocks["train"].x[0].data[2].to_bits(),
        ((1_u64 << 63) as f64).to_bits()
    );
}

#[test]
fn numeric_header_coordinates_check_original_text_before_binary64_axis_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let spec = spec(
        json!({"sources":[{"id":"x","role":"features","input":"axis.csv"}],"params":{"delimiter":","}}),
    );
    for token in ["9007199254740993", "2e-324", "-1e-400"] {
        std::fs::write(dir.path().join("axis.csv"), format!("{token}\n1.0\n2.0\n")).unwrap();
        assert!(
            assemble_with_storage(&spec, dir.path(), &Default::default()).is_err(),
            "{token}"
        );
    }
    for (token, bits) in [
        ("18014398509481984", (18_014_398_509_481_984_f64).to_bits()),
        ("9007199254740993.0", (9_007_199_254_740_992_f64).to_bits()),
        ("-0.000e-999", (-0.0_f64).to_bits()),
        ("0e5", 0),
    ] {
        std::fs::write(dir.path().join("axis.csv"), format!("{token}\n1.0\n2.0\n")).unwrap();
        let assembled = assemble_with_storage(&spec, dir.path(), &Default::default()).unwrap();
        let b = &assembled.blocks["train"];
        assert_eq!(
            b.feature_axes[0][0]["coordinates"][0]
                .as_f64()
                .unwrap()
                .to_bits(),
            bits
        );
        assert_eq!(b.feature_axes[0][0]["unit"], Value::Null);
        assert_eq!(b.header_units[0], "text");
        let package = nirs4all_io::materialize::DatasetPackage::from_assembled(&assembled).unwrap();
        assert!(package.to_full_value().is_ok());
    }
}
