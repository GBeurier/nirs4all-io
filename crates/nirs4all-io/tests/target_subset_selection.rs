//! Selecting some columns of a file: the others are ignored and never drive the NA policy.
use nirs4all_io::api::{load_assembled, Input};
use nirs4all_io::core::materialize::assemble::PartitionBlock;
use nirs4all_io::core::materialize::frame::Cell;
use serde_json::{json, Value};
use std::path::Path;

/// Six samples; `b` misses row 0, `a` and `c` miss rows 3 and 1, `note` rows 0, 2 and 4.
fn write_small(dir: &Path) {
    let x: String = (0..6).map(|i| format!("{i};{}\n", i + 10)).collect();
    std::fs::write(dir.join("X.csv"), format!("1000;1100\n{x}")).unwrap();
    std::fs::write(
        dir.join("Y.csv"),
        "a;b;c\n1;;1\n2;2;\n3;3;3\n;4;4\n5;5;5\n6;6;6\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("M.csv"),
        "sid;site;note\ns0;u;\ns1;v;k\ns2;u;\ns3;v;k\ns4;u;\ns5;v;k\n",
    )
    .unwrap();
}

fn legacy(dir: &Path, part: &str, policy: &str, extra: Value) -> Input {
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let mut config = json!({
        format!("{part}_x"): path("X.csv"),
        format!("{part}_y"): path("Y.csv"),
        format!("{part}_group"): path("M.csv"),
        format!("{part}_group_filter"): ["sid", "site"],
        "global_params": {"delimiter": ";", "has_header": true, "na": {"policy": policy}},
    });
    for (key, value) in extra.as_object().unwrap() {
        config[format!("{part}_{key}")] = value.clone();
    }
    Input::Spec(config)
}

/// First feature column (the original row number), targets and sample ids.
fn rows(block: &PartitionBlock) -> (Vec<f64>, Vec<f64>, Vec<String>) {
    let x = &block.x[0];
    let y = block.y.as_ref().unwrap();
    let sid = block.metadata.as_ref().unwrap().column("sid").unwrap();
    (
        (0..x.n_rows).map(|row| x.data[row * x.n_cols]).collect(),
        y.data.clone(),
        sid.values.iter().map(Cell::to_str_scalar).collect(),
    )
}

#[test]
fn a_filter_selects_its_columns_and_ignores_the_rest_in_each_partition() {
    let dir = tempfile::tempdir().unwrap();
    write_small(dir.path());
    for part in ["train", "test"] {
        for filter in [json!([1]), json!(["b"])] {
            let input = legacy(dir.path(), part, "ignore", json!({"y_filter": filter}));
            let data = load_assembled(&input, None, None).unwrap();
            let block = &data.blocks[part];
            assert_eq!(block.y_headers, ["b"], "{part}");
            let (x, y, sid) = rows(block);
            assert_eq!(x, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(y[0].is_nan());
            assert_eq!(y[1..], [2.0, 3.0, 4.0, 5.0, 6.0]);
            assert_eq!(sid, ["s0", "s1", "s2", "s3", "s4", "s5"]);
            let metadata = block.metadata.as_ref().unwrap();
            assert!(metadata.column("site").is_some() && metadata.column("note").is_none());
        }
        // Several targets, and a feature filter beside the target filter.
        let input = legacy(
            dir.path(),
            part,
            "ignore",
            json!({"y_filter": ["c", "a"], "x_filter": [1]}),
        );
        let block = &load_assembled(&input, None, None).unwrap().blocks[part];
        assert_eq!(block.y_headers, ["a", "c"], "{part}");
        assert_eq!(block.x[0].n_cols, 1);
        assert_eq!(block.x[0].data, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }
    // Targets carried by the feature file: a feature filter leaves the other columns out.
    std::fs::write(
        dir.path().join("XY.csv"),
        "1000;1100;y;z\n1;2;3;\n4;5;6;7\n",
    )
    .unwrap();
    let input = Input::Spec(json!({
        "train_x": dir.path().join("XY.csv").to_string_lossy(),
        "train_x_filter": [0, 1], "train_y_filter": ["y"],
        "global_params": {"delimiter": ";", "has_header": true},
    }));
    let block = &load_assembled(&input, None, None).unwrap().blocks["train"];
    assert_eq!(block.x[0].data, [1.0, 2.0, 4.0, 5.0]);
    assert_eq!(block.y.as_ref().unwrap().data, [3.0, 6.0]);
}

#[test]
fn missing_values_in_ignored_columns_never_remove_a_sample_or_abort() {
    let dir = tempfile::tempdir().unwrap();
    write_small(dir.path());
    for part in ["train", "test"] {
        // Only `b` is selected: its single missing row goes, rows 1 and 3 stay.
        let input = legacy(dir.path(), part, "remove_sample", json!({"y_filter": [1]}));
        let (x, y, sid) = rows(&load_assembled(&input, None, None).unwrap().blocks[part]);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0, 5.0], "{part}");
        assert_eq!(y, [2.0, 3.0, 4.0, 5.0, 6.0], "{part}");
        assert_eq!(sid, ["s1", "s2", "s3", "s4", "s5"], "{part}");

        // Selecting `a` and `c` removes exactly their missing rows 1 and 3.
        let input = legacy(
            dir.path(),
            part,
            "remove_sample",
            json!({"y_filter": ["a", "c"]}),
        );
        let (x, y, sid) = rows(&load_assembled(&input, None, None).unwrap().blocks[part]);
        assert_eq!(x, [0.0, 2.0, 4.0, 5.0], "{part}");
        assert_eq!(y, [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 6.0, 6.0], "{part}");
        assert_eq!(sid, ["s0", "s2", "s4", "s5"], "{part}");

        // The default policy keeps a missing target and ignores unselected columns.
        for policy in ["auto", "ignore"] {
            let input = legacy(dir.path(), part, policy, json!({"y_filter": [1]}));
            let (x, y, _) = rows(&load_assembled(&input, None, None).unwrap().blocks[part]);
            assert_eq!(x.len(), 6, "{part}/{policy}");
            assert!(y[0].is_nan(), "{part}/{policy}");
        }
        // `abort` refuses the selected target's own missing value only.
        let input = legacy(dir.path(), part, "abort", json!({"y_filter": [1]}));
        let error = load_assembled(&input, None, None).unwrap_err();
        assert!(error.message.contains("column 'b'"), "{}", error.message);
    }
}

/// 49 targets over 300 samples: `t00` complete, `tNN` missing 3% to 90% of rows.
fn missing(row: usize, column: usize) -> bool {
    column > 0 && (row * 37 + column * 11) % 100 < 3 + (column - 1) * 87 / 47
}

#[test]
fn one_target_of_a_wide_sparse_file_keeps_every_sample_it_observes() {
    const ROWS: usize = 300;
    const TARGETS: usize = 49;
    let dir = tempfile::tempdir().unwrap();
    let x: String = (0..ROWS).map(|i| format!("{i};{}\n", i * 2)).collect();
    std::fs::write(dir.path().join("X.csv"), format!("1000;1100\n{x}")).unwrap();
    let mut y = (0..TARGETS)
        .map(|c| format!("t{c:02}"))
        .collect::<Vec<_>>()
        .join(";");
    y.push('\n');
    for row in 0..ROWS {
        let cells: Vec<String> = (0..TARGETS)
            .map(|c| {
                if missing(row, c) {
                    String::new()
                } else {
                    format!("{}", row * 100 + c)
                }
            })
            .collect();
        y.push_str(&cells.join(";"));
        y.push('\n');
    }
    std::fs::write(dir.path().join("Y.csv"), y).unwrap();
    let m: String = (0..ROWS).map(|i| format!("s{i:03}\n")).collect();
    std::fs::write(dir.path().join("M.csv"), format!("sid\n{m}")).unwrap();
    let load = |policy: &str, filter: Value| {
        let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
        let input = Input::Spec(json!({
            "train_x": path("X.csv"), "train_y": path("Y.csv"), "train_y_filter": filter,
            "train_group": path("M.csv"),
            "global_params": {"delimiter": ";", "has_header": true, "na": {"policy": policy}},
        }));
        rows(&load_assembled(&input, None, None).unwrap().blocks["train"])
    };
    for column in [0, 1, 24, 48] {
        let observed: Vec<usize> = (0..ROWS).filter(|&row| !missing(row, column)).collect();
        assert_eq!(column == 0, observed.len() == ROWS);
        for filter in [json!([column]), json!([format!("t{column:02}")])] {
            // Kept: every sample stays, the selected target is missing where the file is.
            let (x, y, sid) = load("ignore", filter.clone());
            assert_eq!(x.len(), ROWS);
            assert_eq!(sid.len(), ROWS);
            for (row, value) in y.iter().enumerate() {
                assert_eq!(value.is_nan(), missing(row, column), "t{column:02}/{row}");
            }
            // Removed: exactly the samples the selected target observes, still aligned.
            let (x, y, sid) = load("remove_sample", filter);
            let rows_f: Vec<f64> = observed.iter().map(|&row| row as f64).collect();
            assert_eq!(x, rows_f, "t{column:02}");
            let expected: Vec<f64> = observed
                .iter()
                .map(|&row| (row * 100 + column) as f64)
                .collect();
            assert_eq!(y, expected, "t{column:02}");
            let ids: Vec<String> = observed.iter().map(|row| format!("s{row:03}")).collect();
            assert_eq!(sid, ids, "t{column:02}");
        }
    }
}
