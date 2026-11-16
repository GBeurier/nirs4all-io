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
