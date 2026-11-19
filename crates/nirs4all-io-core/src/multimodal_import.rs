//! Declared multimodal file materialization: exact IDs, no modelling or imputation.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::materialize::loaders::{read_delimited_records, TabularReadLimits};
use crate::SpecError;

pub const MULTIMODAL_IMPORT_PROFILE: &str = "nirs4all.multimodal-import.v1";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ImportManifest {
    pub schema: String,
    pub case_id: String,
    pub name: String,
    pub metadata: MetadataDeclaration,
    pub sources: Vec<SourceDeclaration>,
    #[serde(default)]
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct MetadataDeclaration {
    pub file: String,
    pub sample_id: String,
    pub target: String,
    pub target_unit: String,
    pub group: Option<String>,
    pub partition: Option<String>,
    /// `"categorical"` declares a class target; absent (or `"continuous"`) keeps numeric targets.
    #[serde(default)]
    pub target_kind: Option<String>,
    /// Ordered class vocabulary of a categorical target; cells must match one entry exactly.
    #[serde(default)]
    pub classes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct SourceDeclaration {
    pub id: String,
    pub label: String,
    pub modality: String,
    pub profile: String,
    pub file: String,
    pub sample_id: Option<String>,
    pub features: Option<Vec<String>>,
    pub axis: Option<AxisDeclaration>,
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisDeclaration {
    pub values: Vec<f64>,
    pub unit: String,
    pub kind: String,
}

/// Neutral Formats output is provided by the facade/binding, never parsed here.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecodedAsset {
    Image {
        schema: String,
        source_sha256: String,
        shape: Vec<usize>,
        pixels: Vec<u8>,
        layout: String,
        intensity_range: [u16; 2],
        warnings: Vec<String>,
    },
    Genotype {
        schema: String,
        source_sha256: String,
        sample_ids: Vec<String>,
        variant_ids: Vec<String>,
        original_variant_ids: Vec<String>,
        variant_metadata: Vec<Value>,
        data: Vec<f64>,
        validity: Vec<u8>,
        phased_calls: usize,
        missing_calls: usize,
        warnings: Vec<String>,
    },
}

impl DecodedAsset {
    /// Numeric payload accounting for the caller's aggregate decode budget.
    pub fn decoded_bytes(&self) -> Option<usize> {
        match self {
            Self::Image { pixels, .. } => Some(pixels.len()),
            Self::Genotype { data, validity, .. } => {
                data.len().checked_mul(8)?.checked_add(validity.len())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedSource {
    pub id: String,
    pub label: String,
    pub modality: String,
    pub representation: String,
    pub sample_ids: Vec<String>,
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub feature_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub axis: Option<AxisDeclaration>,
    pub validity: Vec<u8>,
    pub provenance: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedDataset {
    pub case_id: String,
    pub name: String,
    pub fingerprint: String,
    pub sample_ids: Vec<String>,
    pub sources: Vec<ImportedSource>,
    pub target: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partitions: Option<Vec<String>>,
    pub alignment: Vec<Value>,
    pub provenance: Value,
}

fn err(message: impl Into<String>) -> SpecError {
    SpecError::new(message)
}
fn distinct(values: &[String], label: &str) -> Result<(), SpecError> {
    if values.is_empty()
        || values.iter().any(|value| value.trim().is_empty())
        || values.iter().collect::<BTreeSet<_>>().len() != values.len()
    {
        return Err(err(format!(
            "{label} must be nonempty, distinct exact string identities"
        )));
    }
    Ok(())
}

struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}
impl Table {
    fn column(&self, name: &str) -> Result<usize, SpecError> {
        self.headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| err(format!("declared column '{name}' is absent")))
    }
    fn strings(&self, name: &str) -> Result<Vec<String>, SpecError> {
        let column = self.column(name)?;
        Ok(self.rows.iter().map(|row| row[column].clone()).collect())
    }
}

fn table(
    file: &str,
    files: &BTreeMap<String, Vec<u8>>,
    limits: TabularReadLimits,
) -> Result<Table, SpecError> {
    let bytes = files
        .get(file)
        .ok_or_else(|| err(format!("declared file '{file}' is missing")))?;
    std::str::from_utf8(bytes).map_err(|_| err(format!("'{file}' requires UTF-8 CSV text")))?;
    let records = read_delimited_records(bytes, ',', true, Some(limits))?;
    let headers = records
        .first()
        .ok_or_else(|| err(format!("'{file}' has no CSV header")))?
        .clone();
    distinct(&headers, "CSV headers")?;
    if records.len() < 2 || records[1..].iter().any(|row| row.len() != headers.len()) {
        return Err(err(format!(
            "'{file}' requires nonempty rectangular comma-separated UTF-8 records"
        )));
    }
    Ok(Table {
        headers,
        rows: records.into_iter().skip(1).collect(),
    })
}

fn numeric(value: &str) -> Result<Option<f64>, SpecError> {
    if ["", "na", "n/a", "nan"]
        .iter()
        .any(|token| token.eq_ignore_ascii_case(value))
    {
        return Ok(None);
    }
    // Reuse the owner's checked numeric storage conversion, preserving refusal
    // for integer/decimal overflow and underflow instead of silently changing it.
    let cell = crate::materialize::frame::Cell::Str(value.into());
    let number = cell.to_numeric_checked().map_err(err)?;
    if !number.is_finite() {
        return Err(err(format!("nonfinite or nonnumeric cell '{value}'")));
    }
    Ok(Some(number))
}

fn order(canonical: &[String], actual: &[String], source: &str) -> Result<Vec<usize>, SpecError> {
    distinct(actual, "source sample IDs")?;
    let lookup = actual
        .iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let missing = canonical
        .iter()
        .filter(|id| !lookup.contains_key(id))
        .count();
    let extra = actual.iter().filter(|id| !canonical.contains(id)).count();
    if missing != 0 || extra != 0 {
        return Err(err(format!("source '{source}' identity mismatch: {} matched, {missing} missing, {extra} unassigned; positional repair is forbidden",canonical.len()-missing)));
    }
    Ok(canonical.iter().map(|id| lookup[id]).collect())
}

/// Discover the exact declared non-spectral assets a host must decode in Formats.
pub fn multimodal_decode_requests(
    manifest: &ImportManifest,
    files: &BTreeMap<String, Vec<u8>>,
    limits: TabularReadLimits,
) -> Result<Vec<(String, String)>, SpecError> {
    let mut requests = BTreeMap::new();
    for source in &manifest.sources {
        match source.profile.as_str() {
            "numeric_csv" => {}
            "vcf_gt_dosage" => {
                if let Some(existing) = requests.insert(source.file.clone(), "vcf_gt_dosage".into())
                {
                    if existing != "vcf_gt_dosage" {
                        return Err(err(
                            "one file is declared under conflicting decoder profiles",
                        ));
                    }
                }
            }
            "png_manifest" => {
                let mapping = table(&source.file, files, limits)?;
                if mapping.headers != ["sample_id", "file"] {
                    return Err(err("PNG manifest requires exactly sample_id,file columns"));
                }
                distinct(&mapping.strings("sample_id")?, "image manifest sample IDs")?;
                for file in mapping.strings("file")? {
                    if file.is_empty() || !files.contains_key(&file) {
                        return Err(err(format!("image file '{file}' is missing")));
                    }
                    if let Some(existing) = requests.insert(file, "png8".into()) {
                        if existing != "png8" {
                            return Err(err(
                                "one file is declared under conflicting decoder profiles",
                            ));
                        }
                    }
                }
            }
            other => {
                return Err(err(format!(
                    "unsupported declared source import profile '{other}'"
                )))
            }
        }
    }
    Ok(requests.into_iter().collect())
}

pub fn assemble_multimodal_files(
    manifest: &ImportManifest,
    files: &BTreeMap<String, Vec<u8>>,
    decoded: &BTreeMap<String, DecodedAsset>,
    limits: TabularReadLimits,
) -> Result<ImportedDataset, SpecError> {
    if manifest.schema != MULTIMODAL_IMPORT_PROFILE
        || !["spectroscopy", "genetics", "images"].contains(&manifest.case_id.as_str())
        || manifest.name.trim().is_empty()
    {
        return Err(err(
            "unsupported multimodal import schema/case or empty dataset name",
        ));
    }
    let metadata = table(&manifest.metadata.file, files, limits)?;
    let sample_ids = metadata.strings(&manifest.metadata.sample_id)?;
    distinct(&sample_ids, "metadata sample IDs")?;
    let target_cells = metadata.strings(&manifest.metadata.target)?;
    // A categorical target carries its declared, ordered vocabulary: classes are never
    // discovered from the data, and each value is the index of its declared class.
    let categorical = match manifest.metadata.target_kind.as_deref() {
        None | Some("continuous") => {
            if manifest.metadata.classes.is_some() {
                return Err(err("declared classes require targetKind \"categorical\""));
            }
            None
        }
        Some("categorical") => {
            let classes = manifest
                .metadata
                .classes
                .clone()
                .ok_or_else(|| err("a categorical target requires its declared classes"))?;
            distinct(&classes, "declared classes")?;
            if classes.len() < 2 {
                return Err(err(
                    "a categorical target requires at least two declared classes",
                ));
            }
            Some(classes)
        }
        Some(other) => return Err(err(format!("unsupported targetKind '{other}'"))),
    };
    let target_values = match &categorical {
        None => target_cells
            .iter()
            .map(|value| numeric(value)?.ok_or_else(|| err("target values must be complete")))
            .collect::<Result<Vec<_>, _>>()?,
        Some(classes) => target_cells
            .iter()
            .map(|value| {
                if value.trim().is_empty() {
                    return Err(err("target values must be complete"));
                }
                classes
                    .iter()
                    .position(|class| class == value)
                    .map(|index| index as f64)
                    .ok_or_else(|| err(format!("target value '{value}' is not a declared class")))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let groups = manifest
        .metadata
        .group
        .as_ref()
        .map(|name| metadata.strings(name))
        .transpose()?;
    if groups
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| value.trim().is_empty()))
    {
        return Err(err("declared groups must be complete"));
    }
    let partitions = manifest
        .metadata
        .partition
        .as_ref()
        .map(|name| metadata.strings(name))
        .transpose()?;
    if let Some(parts) = &partitions {
        if parts
            .iter()
            .any(|part| !["train", "test"].contains(&part.as_str()))
        {
            return Err(err("partitions must be explicitly train or test"));
        }
        if let Some(groups) = &groups {
            let mut seen = BTreeMap::new();
            for (group, part) in groups.iter().zip(parts) {
                if seen
                    .insert(group, part)
                    .is_some_and(|previous| previous != part)
                {
                    return Err(err("a declared group crosses train/test partitions"));
                }
            }
        }
    }
    distinct(
        &manifest
            .sources
            .iter()
            .map(|source| source.id.clone())
            .collect::<Vec<_>>(),
        "source IDs",
    )?;
    let mut sources = Vec::new();
    let mut alignment = Vec::new();
    let mut total_cells = target_values.len() as u64;
    for source in &manifest.sources {
        if !["spectroscopy", "genetics", "image"].contains(&source.modality.as_str()) {
            return Err(err("unsupported declared modality"));
        }
        let mut result = ImportedSource {
            id: source.id.clone(),
            label: source.label.clone(),
            modality: source.modality.clone(),
            representation: String::new(),
            sample_ids: sample_ids.clone(),
            data: Vec::new(),
            shape: Vec::new(),
            feature_names: Vec::new(),
            axis: source.axis.clone(),
            validity: Vec::new(),
            provenance: Value::Null,
        };
        match source.profile.as_str() {
            "numeric_csv" => {
                if source.modality == "image" {
                    return Err(err("images require the declared PNG manifest profile"));
                }
                let values = table(&source.file, files, limits)?;
                let identity = source
                    .sample_id
                    .as_ref()
                    .ok_or_else(|| err("numeric CSV requires a declared sampleId column"))?;
                let indices = order(&sample_ids, &values.strings(identity)?, &source.id)?;
                let features = source
                    .features
                    .as_ref()
                    .ok_or_else(|| err("numeric CSV requires explicit ordered features"))?;
                distinct(features, "feature headers")?;
                if features.contains(identity) {
                    return Err(err("sample ID cannot be a numeric feature"));
                }
                let columns = features
                    .iter()
                    .map(|name| values.column(name))
                    .collect::<Result<Vec<_>, _>>()?;
                for row in indices {
                    for &column in &columns {
                        let value = numeric(&values.rows[row][column])?;
                        if source.modality == "genetics"
                            && value.is_some_and(|n| ![0.0, 1.0, 2.0].contains(&n))
                        {
                            return Err(err(
                                "declared dosage CSV accepts 0/1/2 or explicit missing calls only",
                            ));
                        }
                        result.data.push(value.unwrap_or(0.0));
                        result.validity.push(u8::from(value.is_some()));
                    }
                }
                result.shape = vec![columns.len()];
                result.feature_names = features.clone();
                result.representation = if source.modality == "spectroscopy" {
                    "signal_1d"
                } else {
                    "tabular_numeric"
                }
                .into();
                result.provenance = json!({"profile":"numeric_csv","sourceSha256":format!("{:x}",Sha256::digest(&files[&source.file])),"missingPolicy":"preserve validity; no imputation"});
            }
            "vcf_gt_dosage" => {
                if source.modality != "genetics" {
                    return Err(err("VCF dosage requires genetics modality"));
                }
                let asset = decoded
                    .get(&source.file)
                    .ok_or_else(|| err("Formats VCF decode is missing"))?;
                if let DecodedAsset::Genotype {
                    schema,
                    source_sha256,
                    sample_ids: ids,
                    variant_ids,
                    original_variant_ids,
                    variant_metadata,
                    data,
                    validity,
                    phased_calls,
                    missing_calls,
                    warnings,
                } = asset
                {
                    if schema != "nirs4all.formats.multimodal.v1"
                        || source_sha256 != &format!("{:x}", Sha256::digest(&files[&source.file]))
                        || ids.len().checked_mul(variant_ids.len()) != Some(data.len())
                        || validity.len() != data.len()
                        || data.iter().any(|value| ![0.0, 1.0, 2.0].contains(value))
                        || validity.iter().any(|&valid| valid > 1)
                    {
                        return Err(err("decoded VCF contract/hash/shape mismatch"));
                    }
                    distinct(variant_ids, "decoded VCF variant IDs")?;
                    let indices = order(&sample_ids, ids, &source.id)?;
                    for row in indices {
                        let start = row * variant_ids.len();
                        result
                            .data
                            .extend_from_slice(&data[start..start + variant_ids.len()]);
                        result
                            .validity
                            .extend_from_slice(&validity[start..start + variant_ids.len()]);
                    }
                    result.shape = vec![variant_ids.len()];
                    result.feature_names = variant_ids.clone();
                    result.representation = "tabular_numeric".into();
                    result.provenance = json!({"profile":"vcf_gt_dosage","sourceSha256":source_sha256,"originalVariantIds":original_variant_ids,"variantMetadata":variant_metadata,"originalHeaderSampleIds":ids,"phasedCalls":phased_calls,"missingCalls":missing_calls,"warnings":warnings});
                } else {
                    return Err(err("VCF declaration received an image decode"));
                }
            }
            "png_manifest" => {
                if source.modality != "image" {
                    return Err(err("PNG manifest requires image modality"));
                }
                if source.shape.is_none() {
                    return Err(err("PNG source requires a declared fixed HWC shape"));
                }
                let mapping = table(&source.file, files, limits)?;
                let indices = order(&sample_ids, &mapping.strings("sample_id")?, &source.id)?;
                let column = mapping.column("file")?;
                let mut hashes = Vec::new();
                for row in indices {
                    let file = &mapping.rows[row][column];
                    let asset = decoded
                        .get(file)
                        .ok_or_else(|| err(format!("Formats PNG decode missing for '{file}'")))?;
                    if let DecodedAsset::Image {
                        schema,
                        source_sha256,
                        shape,
                        pixels,
                        layout,
                        intensity_range,
                        warnings: _,
                    } = asset
                    {
                        if schema != "nirs4all.formats.multimodal.v1"
                            || layout != "HWC"
                            || *intensity_range != [0, 255]
                            || shape.len() != 3
                            || ![1, 3].contains(&shape[2])
                            || shape.contains(&0)
                            || shape
                                .iter()
                                .try_fold(1_usize, |n, size| n.checked_mul(*size))
                                != Some(pixels.len())
                            || source_sha256 != &format!("{:x}", Sha256::digest(&files[file]))
                        {
                            return Err(err("decoded PNG contract/hash/shape mismatch"));
                        }
                        if result.shape.is_empty() {
                            result.shape = shape.clone();
                        }
                        if &result.shape != shape
                            || source
                                .shape
                                .as_ref()
                                .is_some_and(|expected| expected != shape)
                        {
                            return Err(err("image dimensions differ; this profile never resizes"));
                        }
                        result
                            .data
                            .extend(pixels.iter().map(|&value| f64::from(value)));
                        result.validity.extend(std::iter::repeat_n(1, pixels.len()));
                        hashes.push((file.clone(), source_sha256.clone()));
                    } else {
                        return Err(err("PNG declaration received a genotype decode"));
                    }
                }
                result.representation = if result.shape[2] == 1 {
                    "gray_image"
                } else {
                    "rgb_image"
                }
                .into();
                result.feature_names = (0..result.shape.iter().product())
                    .map(|index| format!("pixel:{index}"))
                    .collect();
                result.provenance = json!({"profile":"png_manifest","files":hashes,"layout":"HWC","intensityRange":[0,255],"resize":"none","colourManagement":"none"});
            }
            _ => return Err(err("unsupported multimodal source profile")),
        }
        total_cells = total_cells
            .checked_add(result.data.len() as u64)
            .ok_or_else(|| err("assembled multimodal cell count overflow"))?;
        if total_cells > limits.max_cells {
            return Err(err(
                "assembled multimodal numeric storage exceeds the host cell budget",
            ));
        }
        if let Some(axis) = &source.axis {
            if source.modality != "spectroscopy"
                || axis.values.len() != result.shape[0]
                || axis.values.iter().any(|value| !value.is_finite())
                || axis.unit.is_empty()
                || axis.kind.is_empty()
                || !(axis.values.windows(2).all(|pair| pair[0] < pair[1])
                    || axis.values.windows(2).all(|pair| pair[0] > pair[1]))
            {
                return Err(err("spectral axis must match source features and be finite, monotonic and independently labelled"));
            }
        } else if source.modality == "spectroscopy" {
            return Err(err(
                "spectroscopy source requires its own explicit axis and units",
            ));
        }
        alignment
            .push(json!({"sourceId":source.id,"matched":sample_ids.len(),"missing":0,"refused":0}));
        sources.push(result);
    }
    let provenance = json!({"profile":MULTIMODAL_IMPORT_PROFILE,"metadataSourceSha256":format!("{:x}",Sha256::digest(&files[&manifest.metadata.file])),"declaration":manifest.provenance,"missingPolicy":"complete-data training refuses preserved missing cells; no imputation"});
    let mut target = json!({"name":manifest.metadata.target,"unit":manifest.metadata.target_unit,"values":target_values});
    if let Some(classes) = categorical {
        target["kind"] = json!("categorical");
        target["classes"] = json!(classes);
    }
    let mut result = ImportedDataset {
        case_id: manifest.case_id.clone(),
        name: manifest.name.clone(),
        fingerprint: String::new(),
        sample_ids,
        sources,
        target,
        group_ids: groups,
        partitions,
        alignment,
        provenance,
    };
    result.fingerprint = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&result).map_err(|error| err(error.to_string()))?)
    );
    Ok(result)
}
