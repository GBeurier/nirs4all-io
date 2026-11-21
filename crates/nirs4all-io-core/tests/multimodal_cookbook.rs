//! CATALOGUE fixtures for the separate declared multimodal import vocabulary.
use nirs4all_io_core::materialize::loaders::TabularReadLimits;
use nirs4all_io_core::multimodal_import::{assemble_multimodal_files, ImportManifest};
use serde_json::json;
use std::collections::BTreeMap;

const CATALOGUE: &[&str] = &["numeric_csv", "vcf_gt_dosage", "png_manifest"];

#[test]
fn numeric_ids_are_exact_reordered_and_missing_dosage_is_preserved() {
    assert_eq!(CATALOGUE.len(), 3);
    let declaration = json!({"schema":"nirs4all.multimodal-import.v1","caseId":"genetics","name":"fixture",
        "metadata":{"file":"meta.csv","sampleId":"sample_id","target":"y","targetUnit":"a.u.","group":"g","partition":"p"},
        "sources":[{"id":"markers","label":"Markers","modality":"genetics","profile":"numeric_csv","file":"x.csv","sampleId":"sample_id","features":["m0","m1"]}]});
    let manifest: ImportManifest = serde_json::from_value(declaration).unwrap();
    let mut files = BTreeMap::from([
        (
            "meta.csv".into(),
            b"sample_id,y,g,p\n001,1,a,train\n002,2,b,test\n".to_vec(),
        ),
        (
            "x.csv".into(),
            b"sample_id,m0,m1\n002,2,NA\n001,0,1\n".to_vec(),
        ),
    ]);
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    let assembled = assemble_multimodal_files(&manifest, &files, &BTreeMap::new(), limits).unwrap();
    assert_eq!(assembled.sample_ids, ["001", "002"]);
    assert_eq!(assembled.sources[0].data, [0.0, 1.0, 2.0, 0.0]);
    assert_eq!(assembled.sources[0].validity, [1, 1, 1, 0]);
    files.insert(
        "x.csv".into(),
        b"sample_id,m0,m1\n001,1,1\n001,2,2\n".to_vec(),
    );
    assert!(assemble_multimodal_files(&manifest, &files, &BTreeMap::new(), limits).is_err());
}

#[test]
fn genotype_and_png_profiles_join_neutral_decodes_by_exact_id() {
    use nirs4all_io_core::multimodal_import::DecodedAsset;
    use sha2::{Digest, Sha256};
    let files = BTreeMap::from([
        ("meta.csv".into(), b"sample_id,y\na,1\nb,2\n".to_vec()),
        ("variants.vcf".into(), b"retained VCF bytes".to_vec()),
        (
            "images.csv".into(),
            b"sample_id,file\nb,b.png\na,a.png\n".to_vec(),
        ),
        ("a.png".into(), b"retained image a".to_vec()),
        ("b.png".into(), b"retained image b".to_vec()),
    ]);
    let sha = |name: &str| format!("{:x}", Sha256::digest(&files[name]));
    let genotype = DecodedAsset::Genotype {
        schema: "nirs4all.formats.multimodal.v1".into(),
        source_sha256: sha("variants.vcf"),
        sample_ids: vec!["b".into(), "a".into()],
        variant_ids: vec!["1:1:A:G".into()],
        original_variant_ids: vec!["v".into()],
        variant_metadata: vec![],
        data: vec![2.0, 1.0],
        validity: vec![1, 1],
        phased_calls: 1,
        missing_calls: 0,
        warnings: vec![],
    };
    let image = |name: &str, pixels: Vec<u8>| DecodedAsset::Image {
        schema: "nirs4all.formats.multimodal.v1".into(),
        source_sha256: sha(name),
        shape: vec![1, 2, 1],
        pixels,
        layout: "HWC".into(),
        intensity_range: [0, 255],
        warnings: vec![],
    };
    let decoded = BTreeMap::from([
        ("variants.vcf".into(), genotype),
        ("a.png".into(), image("a.png", vec![10, 20])),
        ("b.png".into(), image("b.png", vec![30, 40])),
    ]);
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    for (profile, case, source) in [
        (
            "vcf_gt_dosage",
            "genetics",
            json!({"id":"g","label":"Genotypes","modality":"genetics","profile":"vcf_gt_dosage","file":"variants.vcf"}),
        ),
        (
            "png_manifest",
            "images",
            json!({"id":"im","label":"Image","modality":"image","profile":"png_manifest","file":"images.csv","shape":[1,2,1]}),
        ),
    ] {
        assert!(CATALOGUE.contains(&profile));
        let manifest:ImportManifest=serde_json::from_value(json!({"schema":"nirs4all.multimodal-import.v1","caseId":case,"name":"fixture",
            "metadata":{"file":"meta.csv","sampleId":"sample_id","target":"y","targetUnit":"a.u."},"sources":[source]})).unwrap();
        let assembled = assemble_multimodal_files(&manifest, &files, &decoded, limits).unwrap();
        assert_eq!(assembled.sample_ids, ["a", "b"]);
        assert_eq!(
            assembled.sources[0].data,
            if profile == "png_manifest" {
                vec![10.0, 20.0, 30.0, 40.0]
            } else {
                vec![1.0, 2.0]
            }
        );
    }
}

#[test]
fn categorical_targets_use_the_declared_class_vocabulary() {
    let declaration = |metadata: serde_json::Value| -> ImportManifest {
        serde_json::from_value(json!({"schema":"nirs4all.multimodal-import.v1","caseId":"spectroscopy","name":"fixture",
            "metadata":metadata,
            "sources":[{"id":"nir","label":"NIR","modality":"spectroscopy","profile":"numeric_csv","file":"x.csv","sampleId":"sample_id","features":["1000","1002"],
                "axis":{"values":[1000.0,1002.0],"unit":"nm","kind":"wavelength"}}]}))
        .unwrap()
    };
    let files = BTreeMap::from([
        (
            "meta.csv".into(),
            b"sample_id,variety\na,beta\nb,alpha\nc,beta\n".to_vec(),
        ),
        (
            "x.csv".into(),
            b"sample_id,1000,1002\nc,5,6\na,1,2\nb,3,4\n".to_vec(),
        ),
    ]);
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    let categorical = declaration(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class",
        "targetKind":"categorical","classes":["alpha","beta"]}),
    );
    let assembled =
        assemble_multimodal_files(&categorical, &files, &BTreeMap::new(), limits).unwrap();
    assert_eq!(assembled.target["kind"], "categorical");
    assert_eq!(assembled.target["classes"], json!(["alpha", "beta"]));
    assert_eq!(assembled.target["values"], json!([1.0, 0.0, 1.0]));

    // The vocabulary is declared, never discovered: undeclared labels, missing
    // declarations and classes without the categorical kind refuse.
    let refuse = |metadata: serde_json::Value| {
        assert!(
            assemble_multimodal_files(&declaration(metadata), &files, &BTreeMap::new(), limits)
                .is_err()
        )
    };
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class","targetKind":"categorical","classes":["alpha","gamma"]}),
    );
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class","targetKind":"categorical"}),
    );
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class","targetKind":"categorical","classes":["alpha","alpha"]}),
    );
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class","classes":["alpha","beta"]}),
    );
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class","targetKind":"ordinal","classes":["alpha","beta"]}),
    );
    // Without a declared kind the target stays numeric (and refuses labels).
    refuse(
        json!({"file":"meta.csv","sampleId":"sample_id","target":"variety","targetUnit":"class"}),
    );
}

#[test]
fn continuous_targets_keep_their_output_and_fingerprint() {
    let manifest = |kind: Option<&str>| -> ImportManifest {
        let mut metadata =
            json!({"file":"meta.csv","sampleId":"sample_id","target":"y","targetUnit":"a.u."});
        if let Some(kind) = kind {
            metadata["targetKind"] = json!(kind);
        }
        serde_json::from_value(json!({"schema":"nirs4all.multimodal-import.v1","caseId":"genetics","name":"fixture","metadata":metadata,
            "sources":[{"id":"markers","label":"Markers","modality":"genetics","profile":"numeric_csv","file":"x.csv","sampleId":"sample_id","features":["m0"]}]}))
        .unwrap()
    };
    let files = BTreeMap::from([
        ("meta.csv".into(), b"sample_id,y\na,1.5\nb,2\n".to_vec()),
        ("x.csv".into(), b"sample_id,m0\na,0\nb,2\n".to_vec()),
    ]);
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    let implicit =
        assemble_multimodal_files(&manifest(None), &files, &BTreeMap::new(), limits).unwrap();
    let explicit = assemble_multimodal_files(
        &manifest(Some("continuous")),
        &files,
        &BTreeMap::new(),
        limits,
    )
    .unwrap();
    assert_eq!(
        implicit.target,
        json!({"name":"y","unit":"a.u.","values":[1.5,2.0]})
    );
    assert_eq!(implicit.fingerprint, explicit.fingerprint);
}

#[test]
fn time_series_sources_are_multivariate_series_over_a_declared_time_axis() {
    let manifest = |source: serde_json::Value| -> ImportManifest {
        serde_json::from_value(json!({"schema":"nirs4all.multimodal-import.v1","caseId":"timeseries","name":"fixture",
            "metadata":{"file":"meta.csv","sampleId":"sample_id","target":"y","targetUnit":"%"},
            "sources":[{"id":"nir","label":"NIR","modality":"spectroscopy","profile":"numeric_csv","file":"nir.csv","sampleId":"sample_id",
                "features":["700","710"],"axis":{"values":[700.0,710.0],"unit":"nm","kind":"wavelength"}}, source]}))
        .unwrap()
    };
    let weather = json!({"id":"weather","label":"Daily weather","modality":"timeseries","profile":"timeseries_csv","file":"w.csv",
        "sampleId":"sample_id","time":"day","features":["t_mean","rain"],"axis":{"values":[-2.0,-1.0,0.0],"unit":"day","kind":"time"}});
    // One row per (sample, time step), one column per variable, in any row order.
    let mut files = BTreeMap::from([
        ("meta.csv".into(), b"sample_id,y\na,15.5\nb,17\n".to_vec()),
        ("nir.csv".into(), b"sample_id,700,710\nb,0.2,0.3\na,0.1,0.2\n".to_vec()),
        ("w.csv".into(), b"sample_id,day,t_mean,rain\nb,0,28.5,0\na,-2,25,1.5\na,-1,26.5,NA\nb,-2,30,0\na,0,27,0\nb,-1,29,4\n".to_vec()),
    ]);
    let limits = TabularReadLimits::new(4096, 4096, 100, 100, 1000);
    let assembled =
        assemble_multimodal_files(&manifest(weather.clone()), &files, &BTreeMap::new(), limits)
            .unwrap();
    let series = &assembled.sources[1];
    assert_eq!(
        (series.modality.as_str(), series.representation.as_str()),
        ("timeseries", "series_mv")
    );
    assert_eq!(series.shape, [3, 2]);
    assert_eq!(series.feature_names, ["t_mean", "rain"]);
    // Time-major [time, variable] per sample, in metadata sample order; a missing cell keeps validity=false.
    assert_eq!(
        series.data,
        [25.0, 1.5, 26.5, 0.0, 27.0, 0.0, 30.0, 0.0, 29.0, 4.0, 28.5, 0.0]
    );
    assert_eq!(series.validity, [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]);
    let refuse = |files: &BTreeMap<String, Vec<u8>>, source: serde_json::Value| {
        assert!(
            assemble_multimodal_files(&manifest(source), files, &BTreeMap::new(), limits).is_err()
        );
    };
    // A time series needs its own time axis, and only the timeseries_csv profile reads one.
    let mut no_axis = weather.clone();
    no_axis.as_object_mut().unwrap().remove("axis");
    refuse(&files, no_axis);
    let mut wavelength = weather.clone();
    wavelength["axis"]["kind"] = json!("wavelength");
    refuse(&files, wavelength);
    let mut wide = weather.clone();
    wide["profile"] = json!("numeric_csv");
    refuse(&files, wide);
    // Undeclared, repeated or missing time steps refuse instead of being resampled.
    for bad in [&b"sample_id,day,t_mean,rain\na,-2,25,1\na,-1,26,1\na,0,27,0\nb,-2,30,0\nb,-1,29,4\nb,1,28,0\n"[..],
                &b"sample_id,day,t_mean,rain\na,-2,25,1\na,-2,26,1\na,0,27,0\nb,-2,30,0\nb,-1,29,4\nb,0,28,0\n"[..],
                &b"sample_id,day,t_mean,rain\na,-2,25,1\na,0,27,0\nb,-2,30,0\nb,-1,29,4\nb,0,28,0\n"[..]] {
        files.insert("w.csv".into(), bad.to_vec());
        refuse(&files, weather.clone());
    }
}
