// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! The facade exposes the Formats contract and retains IO's read budgets.

use std::path::Path;

use nirs4all_io::api::LoadLimits;
use nirs4all_io::core::spec::dataset_spec::LoadingParams;
use nirs4all_io::materialize::loaders::{read_table, read_table_with_limits};
use nirs4all_io::{XLSX_CELLS_OPERATION, XLSX_CELLS_VERSION};

#[test]
fn exposes_the_typed_cell_contract() {
    assert_eq!(XLSX_CELLS_OPERATION, "nirs4all.formats.xlsx-cells/v1");
    assert_eq!(XLSX_CELLS_VERSION, 1);
}

#[test]
fn existing_workbook_passes_through_the_bounded_facade() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spectra_refs.xlsx");
    let params = LoadingParams::default();
    let frame = read_table(&path, &params).unwrap();
    assert_eq!(frame.column_names(), vec!["1000", "1002", "1004"]);
    let limits = LoadLimits {
        max_decoded_file_bytes: 1,
        ..LoadLimits::default()
    };
    let error = read_table_with_limits(&path, &params, limits).unwrap_err();
    assert!(error.message.contains("decoded file/total bytes"));
}

#[test]
fn measured_workbook_usage_obeys_exact_raw_and_decoded_boundaries() {
    use nirs4all_formats::{decode_xlsx_cells, XlsxDecodeLimits};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spectra_refs.xlsx");
    let bytes = std::fs::read(&path).unwrap();
    let workbook =
        decode_xlsx_cells(&bytes, "spectra_refs.xlsx", XlsxDecodeLimits::unlimited()).unwrap();
    let decoded = workbook.usage.total_decoded_bytes().unwrap();
    assert!(decoded > 1);
    let exact = LoadLimits {
        max_file_bytes: bytes.len() as u64,
        max_total_bytes: bytes.len() as u64,
        max_decoded_file_bytes: decoded,
        max_decoded_total_bytes: decoded,
        ..LoadLimits::default()
    };
    let params = LoadingParams::default();
    assert!(read_table_with_limits(&path, &params, exact).is_ok());
    for limits in [
        LoadLimits {
            max_file_bytes: bytes.len() as u64 - 1,
            ..exact
        },
        LoadLimits {
            max_total_bytes: bytes.len() as u64 - 1,
            ..exact
        },
        LoadLimits {
            max_decoded_file_bytes: decoded - 1,
            ..exact
        },
        LoadLimits {
            max_decoded_total_bytes: decoded - 1,
            ..exact
        },
    ] {
        assert!(read_table_with_limits(&path, &params, limits).is_err());
    }
}

#[test]
fn malformed_and_truncated_workbooks_refuse_with_path_context() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/spectra_refs.xlsx");
    let bytes = std::fs::read(fixture).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (name, input) in [
        ("invalid.xlsx", b"not an OOXML archive".as_slice()),
        ("truncated.xlsx", &bytes[..bytes.len() / 2]),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, input).unwrap();
        let error = read_table(&path, &LoadingParams::default()).unwrap_err();
        assert!(error.message.contains("spreadsheet read failed for"));
        assert!(error.message.contains(name));
    }
}

/// Literal OOXML: numeric spellings and date style are independent of an XLSX writer's casts.
fn write_scientific_workbook(path: &Path, rows: &str, epoch_1904: bool) {
    use std::io::Write;
    let workbook = format!(
        r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr date1904="{}"/><sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        u8::from(epoch_1904)
    );
    let sheet = format!(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{rows}</sheetData></worksheet>"#
    );
    let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, xml) in [
        (
            "[Content_Types].xml",
            r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        ("xl/workbook.xml", workbook.as_str()),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>"#,
        ),
        (
            "xl/styles.xml",
            r#"<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14" applyNumberFormat="1"/></cellXfs></styleSheet>"#,
        ),
        ("xl/worksheets/sheet1.xml", sheet.as_str()),
    ] {
        archive
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        archive.write_all(xml.as_bytes()).unwrap();
    }
    archive.finish().unwrap();
}

#[test]
fn physical_xlsx_ignored_outside_domain_integer_goes_to_declared_test() {
    use nirs4all_io::core::spec::DatasetSpec;
    use nirs4all_io::materialize::{assemble, Cell};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    for token in [
        "1000000000000000000000000000000",
        "-9999999999999999999999999999999",
        "18446744073709551616",
    ] {
        let rows = format!(
            r#"<row r="1"><c t="inlineStr"><is><t>x</t></is></c><c t="inlineStr"><is><t>y</t></is></c><c t="inlineStr"><is><t>row</t></is></c><c t="inlineStr"><is><t>set</t></is></c></row><row r="2"><c><v>1</v></c><c><v>101</v></c><c t="inlineStr"><is><t>r0</t></is></c><c><v>1</v></c></row><row r="3"><c><v>2</v></c><c><v>102</v></c><c t="inlineStr"><is><t>r1</t></is></c><c><v>2</v></c></row><row r="4"><c><v>3</v></c><c><v>103</v></c><c t="inlineStr"><is><t>r2</t></is></c><c><v>{token}</v></c></row>"#
        );
        write_scientific_workbook(&dir.path().join("labels.xlsx"), &rows, false);
        let spec = DatasetSpec::from_value(&json!({
            "task_type":"regression","sample_index":{"by":"id","key":"row"},
            "sources":[{"id":"d","input":"labels.xlsx","columns":{"features":"x","targets":"y","metadata":"row","ignore":"set"}}],
            "partitions":{"by":"column","column":"set","train_values":[1,2],"test_values":[token]},
            "params":{"na":{"policy":"ignore"}}
        })).unwrap();
        let assembled = assemble(&spec, dir.path()).unwrap();
        assert_eq!(assembled.blocks.len(), 2);
        for (part, x, y, ids) in [
            (
                "train",
                vec![1.0, 2.0],
                vec![101.0, 102.0],
                vec![Cell::Str("r0".into()), Cell::Str("r1".into())],
            ),
            ("test", vec![3.0], vec![103.0], vec![Cell::Str("r2".into())]),
        ] {
            let block = &assembled.blocks[part];
            assert_eq!(block.n_samples, ids.len());
            assert_eq!(block.x[0].data, x);
            assert_eq!(block.y.as_ref().unwrap().data, y);
            assert_eq!(
                block
                    .metadata
                    .as_ref()
                    .unwrap()
                    .column("row")
                    .unwrap()
                    .values,
                ids
            );
        }
    }
}

#[test]
fn unrenderable_date_integer_joins_exact_target_and_refuses_rounded_neighbor() {
    use nirs4all_io::core::spec::DatasetSpec;
    use nirs4all_io::materialize::{assemble, Cell, DatasetPackage};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let header = r#"<row r="1"><c t="inlineStr"><is><t>k</t></is></c><c t="inlineStr"><is><t>x</t></is></c></row>"#;
    let exact_row = r#"<row r="2"><c s="1"><v>9007199254740993</v></c><c><v>1</v></c></row>"#;
    write_scientific_workbook(
        &dir.path().join("x.xlsx"),
        &format!("{header}{exact_row}"),
        false,
    );
    let bytes = std::fs::read(dir.path().join("x.xlsx")).unwrap();
    let decoded = nirs4all_formats::decode_xlsx_cells(
        &bytes,
        "x.xlsx",
        nirs4all_formats::XlsxDecodeLimits::unlimited(),
    )
    .unwrap();
    let physical_key = decoded.sheets[0]
        .cells
        .iter()
        .find(|cell| cell.position == (1, 0))
        .unwrap();
    assert!(
        matches!(&physical_key.value, nirs4all_formats::XlsxCellValue::Date { token, serial_bits, rendered: None, .. }
        if token == "9007199254740993" && *serial_bits == 9_007_199_254_740_992_f64.to_bits())
    );
    let spec = DatasetSpec::from_value(&json!({
        "task_type":"regression","sample_index":{"by":"id","key":"k"},
        "sources":[
            {"id":"x","input":"x.xlsx","key":"k","columns":{"features":"x","metadata":"k"}},
            {"id":"y","input":"y.csv","key":"k","columns":{"targets":"y"},
             "join":{"left":"x","right":"y","left_on":"k","right_on":"k","how":"1:1","coverage":"complete"}}
        ],"params":{"delimiter":";","na":{"policy":"ignore"}}
    })).unwrap();
    // The original scientific counterexample: only the wrong target exists.
    std::fs::write(dir.path().join("y.csv"), "k;y\n9007199254740992;20\n").unwrap();
    assert!(
        assemble(&spec, dir.path()).is_err(),
        "date integer must not join its rounded neighbor"
    );
    std::fs::write(dir.path().join("y.csv"), "k;y\n9007199254740993;10\n").unwrap();
    let exact = assemble(&spec, dir.path()).unwrap();
    let block = &exact.blocks["train"];
    assert_eq!(block.n_samples, 1);
    assert_eq!(block.x[0].data, vec![1.0]);
    assert_eq!(block.y.as_ref().unwrap().data, vec![10.0]);
    assert_eq!(
        block.metadata.as_ref().unwrap().column("k").unwrap().values,
        vec![Cell::Int(9_007_199_254_740_993)]
    );
    let (identity, _, _) = DatasetPackage::from_assembled(&exact)
        .unwrap()
        .executed_projection(&["x".into()])
        .unwrap();
    assert_eq!(identity.sample_ids, vec!["9007199254740993"]);
    // Reversed target order checks both sides of the precision boundary.
    write_scientific_workbook(
        &dir.path().join("x.xlsx"),
        &format!(
            r#"{header}{exact_row}<row r="3"><c s="1"><v>9007199254740992</v></c><c><v>2</v></c></row>"#
        ),
        false,
    );
    std::fs::write(
        dir.path().join("y.csv"),
        "k;y\n9007199254740992;20\n9007199254740993;10\n",
    )
    .unwrap();
    let paired = assemble(&spec, dir.path()).unwrap();
    assert_eq!(paired.blocks.len(), 1);
    let block = &paired.blocks["train"];
    assert_eq!(block.x[0].data, vec![1.0, 2.0]);
    assert_eq!(block.y.as_ref().unwrap().data, vec![10.0, 20.0]);
    assert_eq!(
        block.metadata.as_ref().unwrap().column("k").unwrap().values,
        vec![
            Cell::Int(9_007_199_254_740_993),
            Cell::Int(9_007_199_254_740_992)
        ]
    );
}

#[test]
fn date_projection_preserves_rendering_float_serials_and_integer_domain_refusal() {
    use nirs4all_io::core::spec::DatasetSpec;
    use nirs4all_io::materialize::{assemble, Cell};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dates.xlsx");
    let header = r#"<row r="1"><c t="inlineStr"><is><t>k</t></is></c><c t="inlineStr"><is><t>x</t></is></c></row>"#;
    for (token, epoch, expected) in [
        ("45292", false, Cell::Str("2024-01-01 00:00:00".into())),
        ("0", true, Cell::Str("1904-01-01 00:00:00".into())),
        (
            "9007199254740993.0",
            false,
            Cell::Float(9_007_199_254_740_992.0),
        ),
        (
            "9.007199254740993E15",
            false,
            Cell::Float(9_007_199_254_740_992.0),
        ),
        ("1000000000.25", false, Cell::Float(1_000_000_000.25)),
        ("1E+30", false, Cell::Float(1e30)),
        (
            "-9007199254740993",
            false,
            Cell::Int(-9_007_199_254_740_993),
        ),
        (
            "+0018446744073709551615",
            false,
            Cell::Str("18446744073709551615".into()),
        ),
    ] {
        write_scientific_workbook(
            &path,
            &format!(r#"{header}<row r="2"><c s="1"><v>{token}</v></c><c><v>7</v></c></row>"#),
            epoch,
        );
        let frame = read_table(&path, &LoadingParams::default()).unwrap();
        assert_eq!(
            frame.column("k").unwrap().values,
            vec![expected],
            "{token}, epoch1904={epoch}"
        );
        assert_eq!(frame.column("x").unwrap().values, vec![Cell::Int(7)]);
    }
    for token in ["18446744073709551616", "1000000000000000000000000000000"] {
        write_scientific_workbook(
            &path,
            &format!(r#"{header}<row r="2"><c s="1"><v>{token}</v></c><c><v>7</v></c></row>"#),
            false,
        );
        let frame = read_table(&path, &LoadingParams::default()).unwrap();
        assert_eq!(
            frame.column("k").unwrap().values,
            vec![Cell::Str(token.into())]
        );
        for role in ["metadata", "features", "identity"] {
            let mut value = json!({"sources":[{"id":"d","input":"dates.xlsx","columns":{"features":"x","metadata":"k"}}]});
            if role == "features" {
                value["sources"][0]["columns"] = json!({"features":"k","ignore":"x"});
            }
            if role == "identity" {
                value["sample_index"] = json!({"by":"id","key":"k"});
            }
            let error =
                assemble(&DatasetSpec::from_value(&value).unwrap(), dir.path()).unwrap_err();
            assert!(
                error
                    .message
                    .contains("outside the supported i64/u64 domain"),
                "{role}: {}",
                error.message
            );
        }
    }
    // The same date-styled integer is valid metadata but cannot silently round
    // when selected as a binary64 feature.
    write_scientific_workbook(
        &path,
        &format!(r#"{header}<row r="2"><c s="1"><v>9007199254740993</v></c><c><v>7</v></c></row>"#),
        false,
    );
    let numeric = DatasetSpec::from_value(&json!({"sources":[{"id":"d","input":"dates.xlsx","columns":{"features":"k","ignore":"x"}}]})).unwrap();
    let error = assemble(&numeric, dir.path()).unwrap_err();
    assert!(
        error.message.contains("exactly representable"),
        "{}",
        error.message
    );
    // The ordinary calamine-compatible Sheet<Cell> input still preserves bits.
    let sheets = [(
        "s".into(),
        vec![vec![Cell::Str("k".into())], vec![Cell::Float(-0.0)]],
    )];
    let frame = nirs4all_io::core::materialize::loaders::read_sheet(
        &sheets,
        &LoadingParams::default(),
        None,
    )
    .unwrap();
    assert!(
        matches!(frame.column("k").unwrap().values[0], Cell::Float(value) if value.to_bits() == (-0.0_f64).to_bits())
    );
}

#[test]
fn ignored_xlsx_1e999_keeps_infinity_and_physical_text_labels_separate() {
    use nirs4all_io::core::spec::DatasetSpec;
    use nirs4all_io::materialize::{assemble, Cell};
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    write_scientific_workbook(
        &dir.path().join("overflow.xlsx"),
        r#"<row r="1"><c t="inlineStr"><is><t>x</t></is></c><c t="inlineStr"><is><t>y</t></is></c><c t="inlineStr"><is><t>set</t></is></c><c t="inlineStr"><is><t>row</t></is></c></row><row r="2"><c><v>1</v></c><c><v>101</v></c><c><v>1e999</v></c><c t="inlineStr"><is><t>r0</t></is></c></row><row r="3"><c><v>2</v></c><c><v>102</v></c><c t="inlineStr"><is><t>1e999</t></is></c><c t="inlineStr"><is><t>r1</t></is></c></row><row r="4"><c><v>3</v></c><c><v>103</v></c><c s="1"><v>1e999</v></c><c t="inlineStr"><is><t>r2</t></is></c></row>"#,
        false,
    );
    for (declared, ids, x, y) in [
        (
            "1e999",
            vec![Cell::Str("r1".into())],
            vec![2.0],
            vec![102.0],
        ),
        (
            "inf",
            vec![Cell::Str("r0".into()), Cell::Str("r2".into())],
            vec![1.0, 3.0],
            vec![101.0, 103.0],
        ),
    ] {
        let spec = DatasetSpec::from_value(&json!({
            "task_type":"regression","sample_index":{"by":"id","key":"row"},
            "sources":[{"id":"d","input":"overflow.xlsx","columns":{"features":"x","targets":"y","metadata":"row","ignore":"set"}}],
            "partitions":{"by":"column","column":"set","test_values":[declared],"unknown_policy":"drop"},
            "params":{"na":{"policy":"ignore"}}
        })).unwrap();
        let assembled = assemble(&spec, dir.path()).unwrap();
        assert_eq!(assembled.blocks.len(), 1);
        let block = &assembled.blocks["test"];
        assert_eq!(block.n_samples, ids.len());
        assert_eq!(
            block
                .metadata
                .as_ref()
                .unwrap()
                .column("row")
                .unwrap()
                .values,
            ids
        );
        assert_eq!(block.x[0].data, x);
        assert_eq!(block.y.as_ref().unwrap().data, y);
    }
}
