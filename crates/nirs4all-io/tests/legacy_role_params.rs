use nirs4all_io::api::{load_assembled, Input};
use serde_json::json;

#[test]
fn headerless_targets_and_metadata_keep_first_row_in_both_partitions() {
    let root = tempfile::tempdir().unwrap();
    let mut config = json!({"global_params":{"delimiter":";","has_header":true}});
    for partition in ["train", "test"] {
        for (role, text) in [
            (
                "x",
                if partition == "train" {
                    "1;2\n3;4\n5;6\n"
                } else {
                    "7;8\n9;10\n11;12\n"
                },
            ),
            (
                "y",
                if partition == "train" {
                    "101\n102\n103\n"
                } else {
                    "201\n202\n203\n"
                },
            ),
            ("group", "first\nsecond\nthird\n"),
        ] {
            let path = root.path().join(format!("{partition}_{role}.csv"));
            std::fs::write(&path, text).unwrap();
            config[format!("{partition}_{role}")] = json!(path);
            config[format!("{partition}_{role}_params")] = json!({"has_header":false});
        }
    }
    let assembled = load_assembled(&Input::Spec(config), None, None).unwrap();
    for partition in ["train", "test"] {
        let block = &assembled.blocks[partition];
        assert_eq!(block.n_samples, 3);
        assert_eq!(
            block.x[0].data,
            if partition == "train" {
                vec![1., 2., 3., 4., 5., 6.]
            } else {
                vec![7., 8., 9., 10., 11., 12.]
            }
        );
        assert_eq!(
            block.y.as_ref().unwrap().data,
            if partition == "train" {
                vec![101., 102., 103.]
            } else {
                vec![201., 202., 203.]
            }
        );
        assert_eq!(block.y_headers, vec!["0"]);
        let metadata = block.metadata.as_ref().unwrap();
        assert_eq!(metadata.n_rows, 3);
        assert_eq!(metadata.column_names(), vec![format!("{partition}_m__0")]);
        assert_eq!(metadata.columns[0].values[0].to_str_scalar(), "first");
    }
}

fn table(root: &std::path::Path, id: &str, text: &str, role: &str) -> serde_json::Value {
    let path = root.join(format!("{id}.csv"));
    std::fs::write(&path, text).unwrap();
    json!({"id":id,"input":path,"role":role,"params":{"delimiter":";","has_header":true}})
}

fn assemble(
    sources: Vec<serde_json::Value>,
) -> nirs4all_io::core::materialize::assemble::AssembledDataset {
    load_assembled(
        &Input::Spec(json!({"task_type":"regression","sources":sources})),
        None,
        None,
    )
    .unwrap()
}

#[test]
fn feature_primary_and_unassigned_keys_never_become_targets() {
    for key in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut x = table(
            root.path(),
            "x",
            if key {
                "id;a\n1;5\n2;6\n"
            } else {
                "0;1\n1;2\n3;4\n"
            },
            "features",
        );
        let y = table(
            root.path(),
            "y",
            if key { "id\n70\n80\n" } else { "0\n70\n80\n" },
            "targets",
        );
        if key {
            x["role"] = json!("mixed");
            x["key"] = json!("id");
            x["columns"] = json!([{"role":"features","select":["a"]}]);
        }
        let data = assemble(vec![y, x]);
        let block = &data.blocks["train"];
        assert_eq!(block.y.as_ref().unwrap().data, vec![70., 80.]);
        assert_eq!(block.y_headers, if key { vec!["id"] } else { vec!["0"] });
        assert_eq!(
            block.x[0].data,
            if key {
                vec![5., 6.]
            } else {
                vec![1., 2., 3., 4.]
            }
        );
    }
}

#[test]
fn frame_order_controls_native_names_and_selector_order_cannot_steal_values() {
    for metadata in [false, true] {
        for literal_first in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let anchor = table(root.path(), "anchor", "0\n100\n200\n", "features");
            let mut mixed = table(
                root.path(),
                "train_x",
                "0;train_x__0\n10;11\n20;21\n",
                "mixed",
            );
            let role = if metadata { "metadata" } else { "features" };
            let mut rules = vec![
                json!({"role":role,"select":["train_x__0"]}),
                json!({"role":"targets","select":["0"]}),
            ];
            if !literal_first {
                rules.reverse();
            }
            mixed["columns"] = json!(rules);
            let data = assemble(vec![anchor, mixed]);
            let block = &data.blocks["train"];
            assert_eq!(block.y_headers, vec!["0"]);
            assert_eq!(block.y.as_ref().unwrap().data, vec![10., 20.]);
            if metadata {
                let frame = block.metadata.as_ref().unwrap();
                assert_eq!(frame.column_names(), vec!["train_x__train_x__0"]);
                assert_eq!(
                    frame.numeric_column_f64("train_x__train_x__0"),
                    vec![11., 21.]
                );
            } else {
                assert_eq!(block.source_ids, vec!["anchor", "train_x"]);
                assert_eq!(block.feature_headers[1], vec!["train_x__0"]);
                assert_eq!(block.x[1].data, vec![11., 21.]);
            }
        }
    }
}

#[test]
fn independent_duplicate_features_and_forward_literal_preserve_every_value() {
    for anchored in [false, true] {
        for reverse in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut sources = Vec::new();
            if anchored {
                sources.push(table(root.path(), "anchor", "0\n100\n200\n", "features"));
            }
            sources.push(table(
                root.path(),
                "train_x",
                if reverse {
                    "train_x__0;0\n11;10\n21;20\n"
                } else {
                    "0;train_x__0\n10;11\n20;21\n"
                },
                "features",
            ));
            let result = load_assembled(&Input::Spec(json!({"sources":sources})), None, None);
            if anchored && reverse {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate output column"));
            } else {
                let data = result.unwrap();
                let block = &data.blocks["train"];
                let index = usize::from(anchored);
                assert_eq!(
                    block.feature_headers[index],
                    if reverse {
                        vec!["train_x__0", "0"]
                    } else {
                        vec!["0", "train_x__0"]
                    }
                );
                assert_eq!(
                    block.x[index].data,
                    if reverse {
                        vec![11., 10., 21., 20.]
                    } else {
                        vec![10., 11., 20., 21.]
                    }
                );
                assert!(block.feature_axes[index][0]["coordinates"].is_null());
            }
        }
    }
    let root = tempfile::tempdir().unwrap();
    let sources = vec![
        table(root.path(), "anchor", "0;1\n100;101\n200;201\n", "features"),
        table(root.path(), "b", "0;1\n10;11\n20;21\n", "features"),
        table(root.path(), "c", "0\n30\n40\n", "features"),
    ];
    let data = assemble(sources);
    let block = &data.blocks["train"];
    assert_eq!(
        block.feature_headers,
        vec![vec!["0", "1"], vec!["0", "1"], vec!["0"]]
    );
    assert_eq!(block.x[1].data, vec![10., 11., 20., 21.]);
    assert_eq!(block.x[2].data, vec![30., 40.]);
}

#[test]
fn concat_collision_routes_refuse_instead_of_substituting_values() {
    for case in [
        "cross_source",
        "target_reverse",
        "target_before_feature",
        "in_source",
    ] {
        let root = tempfile::tempdir().unwrap();
        let sources = match case {
            "cross_source" => vec![
                table(
                    root.path(),
                    "anchor",
                    "0;train_x__0\n100;101\n200;201\n",
                    "features",
                ),
                table(root.path(), "train_x", "0\n10\n20\n", "features"),
            ],
            "target_reverse" => vec![
                table(root.path(), "x", "0\n1\n2\n", "features"),
                table(root.path(), "y", "y__0;0\n11;10\n21;20\n", "targets"),
            ],
            "target_before_feature" => vec![
                table(root.path(), "x2", "0\n50\n60\n", "features"),
                table(root.path(), "y", "x__0\n70\n80\n", "targets"),
                table(root.path(), "x", "0\n1\n2\n", "features"),
            ],
            _ => {
                let a = table(root.path(), "a", "x;b__x\n1;2\n3;4\n", "features");
                let b = table(root.path(), "b", "x\n5\n6\n", "features");
                vec![
                    json!({"id":"d","input":[a["input"],b["input"]],"role":"features","merge":"concat_features","params":{"delimiter":";","has_header":true}}),
                ]
            }
        };
        let error =
            load_assembled(&Input::Spec(json!({"sources":sources})), None, None).unwrap_err();
        assert!(
            error.to_string().contains("duplicate output column"),
            "{case}: {error}"
        );
    }
}

#[test]
fn keyed_suffix_collisions_refuse_but_normal_join_and_dropped_key_work() {
    for kind in ["targets", "features"] {
        for collision in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut x = table(
                root.path(),
                "x",
                if collision {
                    "id;a;a__y\ns1;1;2\ns2;3;4\n"
                } else {
                    "id;a;b\ns1;1;2\ns2;3;4\n"
                },
                "mixed",
            );
            x["key"] = json!("id");
            x["columns"] = json!([{"role":"features","select":if collision {vec!["a","a__y"]} else {vec!["a","b"]}}]);
            let mut y = table(root.path(), "y", "id;a\ns1;70\ns2;80\n", "mixed");
            y["key"] = json!("id");
            y["columns"] = json!([{"role":kind,"select":["a"]}]);
            y["join"] = json!({"to":"x","on":"id","how":"1:1"});
            let result = load_assembled(&Input::Spec(json!({"sources":[x,y]})), None, None);
            if collision {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate output column"));
            } else {
                let data = result.unwrap();
                let block = &data.blocks["train"];
                if kind == "targets" {
                    assert_eq!(block.y_headers, vec!["a"]);
                    assert_eq!(block.y.as_ref().unwrap().data, vec![70., 80.]);
                } else {
                    assert_eq!(block.feature_headers[1], vec!["a"]);
                    assert_eq!(block.x[1].data, vec![70., 80.]);
                }
            }
        }
    }
    let root = tempfile::tempdir().unwrap();
    let mut x = table(root.path(), "x", "id;a\ns1;1\ns2;2\n", "mixed");
    x["key"] = json!("id");
    x["columns"] = json!([{"role":"features","select":["a"]}]);
    let mut lookup = table(root.path(), "lookup", "rid;z\ns1;10\ns2;20\n", "mixed");
    lookup["kind"] = json!("lookup");
    lookup["key"] = json!("rid");
    lookup["columns"] = json!([{"role":"metadata","select":["z"]}]);
    lookup["join"] = json!({"to":"x","left_on":"id","right_on":"rid","how":"1:1"});
    let y = table(root.path(), "y", "rid\n70\n80\n", "targets");
    let data = assemble(vec![x, lookup, y]);
    let block = &data.blocks["train"];
    assert_eq!(block.y_headers, vec!["rid"]);
    assert_eq!(block.y.as_ref().unwrap().data, vec![70., 80.]);
    assert_eq!(
        block.metadata.as_ref().unwrap().numeric_column_f64("z"),
        vec![10., 20.]
    );
}

#[test]
fn targets_native_labels_permutation_literals_and_distinct_partitions() {
    for case in ["A", "B", "C", "F", "G", "H", "I", "J"] {
        let root = tempfile::tempdir().unwrap();
        let mut sources = Vec::new();
        for (part, base) in [("train", 100), ("test", 200)] {
            let (xtext, ytext, header, expected_headers, expected_values) = match case {
                "A" | "B" => (
                    if case == "A" {
                        "1;2\n3;4\n".into()
                    } else {
                        "x0;x1\n1;2\n3;4\n".into()
                    },
                    format!("{}\n{}\n", base + 1, base + 2),
                    false,
                    vec!["0"],
                    vec![f64::from(base + 1), f64::from(base + 2)],
                ),
                "C" => (
                    "1;2\n3;4\n".into(),
                    format!("{};{}\n{};{}\n", base + 1, base + 9, base + 2, base + 8),
                    false,
                    vec!["0", "1"],
                    vec![
                        f64::from(base + 1),
                        f64::from(base + 9),
                        f64::from(base + 2),
                        f64::from(base + 8),
                    ],
                ),
                "F" => (
                    "a;b\n1;2\n3;4\n".into(),
                    if part == "train" {
                        "b;a\n101;109\n102;108\n".into()
                    } else {
                        "a;b\n209;201\n208;202\n".into()
                    },
                    true,
                    if part == "train" {
                        vec!["b", "a"]
                    } else {
                        vec!["a", "b"]
                    },
                    if part == "train" {
                        vec![101., 109., 102., 108.]
                    } else {
                        vec![209., 201., 208., 202.]
                    },
                ),
                "H" => (
                    "0;1\n1;2\n3;4\n".into(),
                    format!(
                        "0;train_y__0\n{};{}\n{};{}\n",
                        base + 1,
                        base + 9,
                        base + 2,
                        base + 8
                    ),
                    true,
                    vec!["0", "train_y__0"],
                    vec![
                        f64::from(base + 1),
                        f64::from(base + 9),
                        f64::from(base + 2),
                        f64::from(base + 8),
                    ],
                ),
                "J" => (
                    format!("id;a;b\n{part}1;1;2\n{part}2;3;4\n"),
                    format!("id;a\n{part}1;{}\n{part}2;{}\n", base + 1, base + 2),
                    true,
                    vec!["a"],
                    vec![f64::from(base + 1), f64::from(base + 2)],
                ),
                _ => {
                    let name = if case == "I" {
                        "0__train_y".to_string()
                    } else {
                        format!("{part}_y__0")
                    };
                    (
                        "x0;x1\n1;2\n3;4\n".into(),
                        format!("{name}\n{}\n{}\n", base + 1, base + 2),
                        true,
                        if case == "I" {
                            vec!["0__train_y"]
                        } else if part == "train" {
                            vec!["train_y__0"]
                        } else {
                            vec!["test_y__0"]
                        },
                        vec![f64::from(base + 1), f64::from(base + 2)],
                    )
                }
            };
            let mut x = table(root.path(), &format!("{part}_x"), &xtext, "features");
            x["partition"] = json!(part);
            x["params"]["has_header"] = json!(!matches!(case, "A" | "C"));
            let mut y = table(root.path(), &format!("{part}_y"), &ytext, "targets");
            y["partition"] = json!(part);
            y["params"]["has_header"] = json!(header);
            if case == "J" {
                x["role"] = json!("mixed");
                x["key"] = json!("id");
                x["columns"] = json!([{"role":"features","select":["a","b"]}]);
                y["role"] = json!("mixed");
                y["key"] = json!("id");
                y["columns"] = json!([{"role":"targets","select":["a"]}]);
                y["join"] = json!({"to":format!("{part}_x"),"on":"id","how":"1:1"});
            }
            sources.extend([x, y]);
            let data = assemble(sources.clone());
            let block = &data.blocks[part];
            assert_eq!(block.y_headers, expected_headers, "{case}/{part}");
            assert_eq!(
                block.y.as_ref().unwrap().data,
                expected_values,
                "{case}/{part}"
            );
        }
    }
}

#[test]
fn categorical_native_keys_keep_reversed_categories_and_duplicate_labels() {
    let root = tempfile::tempdir().unwrap();
    let mut sources = Vec::new();
    for (part, text) in [("train", "cat\ndog\ncat\n"), ("test", "dog\ncat\ncat\n")] {
        let mut x = table(
            root.path(),
            &format!("{part}_x"),
            "1;2\n3;4\n5;6\n",
            "features",
        );
        x["partition"] = json!(part);
        x["params"]["has_header"] = json!(false);
        let mut y = table(root.path(), &format!("{part}_y"), text, "targets");
        y["partition"] = json!(part);
        y["params"]["has_header"] = json!(false);
        sources.extend([x, y]);
    }
    let data = load_assembled(
        &Input::Spec(json!({"task_type":"multiclass","sources":sources})),
        None,
        None,
    )
    .unwrap();
    for part in ["train", "test"] {
        let block = &data.blocks[part];
        assert_eq!(block.y_headers, vec!["0"]);
        assert_eq!(
            block
                .y_categorical
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["0"]
        );
        assert_eq!(
            block.y_categorical["0"]["categories"],
            if part == "train" {
                json!(["cat", "dog"])
            } else {
                json!(["dog", "cat"])
            }
        );
        assert_eq!(
            block.y.as_ref().unwrap().data,
            if part == "train" {
                vec![0., 1., 0.]
            } else {
                vec![0., 1., 1.]
            }
        );
    }
    for (case, header, y1, y2, expected) in [
        (
            "K",
            true,
            "y\n1\n2\n3\n",
            "y\n4\n5\n6\n",
            vec!["y", "y2__y"],
        ),
        ("L", false, "1\n2\n3\n", "4\n5\n6\n", vec!["y1__0", "y2__0"]),
        (
            "M",
            true,
            "y\ncat\ndog\ncat\n",
            "y\nred\nblue\nred\n",
            vec!["y", "y2__y"],
        ),
        (
            "N",
            true,
            "x;y\n1;7\n3;8\n5;9\n",
            "y\n4\n5\n6\n",
            vec!["y", "y2__y"],
        ),
    ] {
        let mut x = table(
            root.path(),
            "x",
            if header {
                "x0;x1\n1;2\n3;4\n5;6\n"
            } else {
                "1;2\n3;4\n5;6\n"
            },
            "features",
        );
        x["params"]["has_header"] = json!(header);
        let mut first = table(root.path(), "y1", y1, "targets");
        first["params"]["has_header"] = json!(header);
        let mut second = table(root.path(), "y2", y2, "targets");
        second["params"]["has_header"] = json!(header);
        let sources = if case == "N" {
            first["role"] = json!("mixed");
            first["columns"] =
                json!([{"role":"features","select":["x"]},{"role":"targets","select":["y"]}]);
            vec![first, second]
        } else {
            vec![x, first, second]
        };
        let data=load_assembled(&Input::Spec(json!({"task_type":if case=="M" {"multiclass"} else {"regression"},"sources":sources})),None,None).unwrap();
        let block = &data.blocks["train"];
        assert_eq!(block.y_headers, expected);
        if case == "M" {
            assert_eq!(
                block
                    .y_categorical
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                expected
            );
        } else {
            assert_eq!(
                block.y.as_ref().unwrap().data,
                if case == "N" {
                    vec![7., 4., 8., 5., 9., 6.]
                } else {
                    vec![1., 4., 2., 5., 3., 6.]
                }
            );
        }
    }
}

#[test]
fn final_target_export_collision_refuses_before_categorical_overwrite() {
    let root = tempfile::tempdir().unwrap();
    let sources = vec![
        table(root.path(), "x", "x0\n1\n2\n3\n", "features"),
        table(root.path(), "y1", "y\ncat\ndog\ncat\n", "targets"),
        table(root.path(), "y2", "y\nred\nblue\nred\n", "targets"),
        table(root.path(), "y3", "y2__y\nblack\nwhite\nblack\n", "targets"),
    ];
    let error = load_assembled(
        &Input::Spec(json!({"task_type":"multiclass","sources":sources})),
        None,
        None,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("target export labels must be unique"));
}

#[test]
fn single_partition_targets_metadata_weights_and_unassigned_guard_stay_explicit() {
    for named in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut x = table(
            root.path(),
            "x",
            if named {
                "a;b\n1;2\n3;4\n"
            } else {
                "1;2\n3;4\n"
            },
            "features",
        );
        let mut y = table(
            root.path(),
            "y",
            if named { "a\n70\n80\n" } else { "70\n80\n" },
            "targets",
        );
        x["params"]["has_header"] = json!(named);
        y["params"]["has_header"] = json!(named);
        let data = assemble(vec![x, y]);
        let block = &data.blocks["train"];
        assert_eq!(block.y_headers, if named { vec!["a"] } else { vec!["0"] });
        assert_eq!(block.y.as_ref().unwrap().data, vec![70., 80.]);
    }
    let root = tempfile::tempdir().unwrap();
    let data = assemble(vec![
        table(root.path(), "x", "m;n\n1;2\n3;4\n", "features"),
        table(root.path(), "w", "m\n0.25\n0.75\n", "weights"),
    ]);
    assert_eq!(data.blocks["train"].weights_header, Some("w__m".into()));
    assert_eq!(data.blocks["train"].weights, Some(vec![0.25, 0.75]));
    let mut mixed = table(root.path(), "d", "x;y;metadata\n1;70;a\n2;80;b\n", "mixed");
    mixed["columns"] =
        json!([{"role":"features","select":["x"]},{"role":"targets","select":["y"]}]);
    assert!(
        load_assembled(&Input::Spec(json!({"sources":[mixed]})), None, None)
            .unwrap_err()
            .to_string()
            .contains("unassigned")
    );
    let mut x = table(root.path(), "keys", "id;a\n1;5\n2;6\n", "mixed");
    x["key"] = json!("id");
    x["columns"] = json!([{"role":"features","select":["a"]}]);
    let data = assemble(vec![
        x,
        table(root.path(), "meta", "id\n70\n80\n", "metadata"),
    ]);
    assert_eq!(
        data.blocks["train"]
            .metadata
            .as_ref()
            .unwrap()
            .numeric_column_f64("meta__id"),
        vec![70., 80.]
    );
}
