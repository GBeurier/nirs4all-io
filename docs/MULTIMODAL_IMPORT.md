# Declared multimodal file import

`nirs4all.multimodal-import.v1` is a separate explicit materialization operation,
not an alternative spelling of DatasetSpec V1. The pure core accepts named CSV
bytes plus neutral decoded assets. The native facade delegates non-spectral
bytes to Formats; the WASM binding calls the supplied synchronous Formats owner
decoder. No model, imputer, random partition, scaling or resize is fitted here.

Native: `multimodal_import::load_multimodal_files(manifest,files,tabular_limits,max_decoded_bytes)`.
WASM: `assembleMultimodalFiles(manifestJson,files,decode,maxDecodedBytes)`, where
`files` is `{name,bytes:Uint8Array}[]` and `decode(profile,bytes,budget)` is the
Formats `decodeMultimodalBytes` function. The caller owns file acquisition and
resource policy. New output C1 DTO has typed Float64 data/target buffers and
Uint8 validity buffers in WASM. Other existing bindings are not claimed here.

```json
{
  "schema": "nirs4all.multimodal-import.v1",
  "caseId": "genetics",
  "name": "my-dataset",
  "metadata": {"file":"metadata.csv","sampleId":"sample_id","target":"y","targetUnit":"a.u.","group":"material","partition":"set"},
  "sources": [
    {"id":"nir","label":"NIRS","modality":"spectroscopy","profile":"numeric_csv","file":"nir.csv","sampleId":"sample_id","features":["1000","1100"],"axis":{"values":[1000,1100],"unit":"nm","kind":"wavelength"}},
    {"id":"markers","label":"Genotypes","modality":"genetics","profile":"vcf_gt_dosage","file":"markers.vcf"}
  ],
  "provenance": {"description":"Declared import"}
}
```

All CSV files use comma-separated UTF-8, an explicit header and exact string ID
cells (including leading zeros). Metadata row order establishes canonical sample
order. Source row/header sample order may differ; assembly joins by identity.
Duplicate IDs, missing/extra source IDs, absent declared columns or rectangular
shape defects refuse. Optional metadata `group` and `partition` declarations must
be complete; partitions are `train` or `test`, and a group cannot cross them.

The numeric profile requires explicit ordered `features` and its `sampleId`
column; spectral sources also require an independent finite monotonic axis,
unit and kind. Genetics dosage CSV uses modality `genetics`, profile
`numeric_csv`, and accepts only 0/1/2 or declared missing tokens (`NA`, `N/A`,
`NaN`, empty). Missing source cells retain zero storage plus validity=false;
targets must be complete. No missing value is silently filled for training.

`png_manifest` declares image modality, a fixed `shape:[H,W,C]`, and a mapping
CSV containing exactly `sample_id,file`. Files decode through Formats `png8`;
every image must match the declared shape and raw HWC layout. Grayscale C=1,
RGB C=3; no resize or colour transform. Decoder SHA-256 is checked against the
provided file bytes. VCF uses the explicit Formats `vcf_gt_dosage` profile and
preserves marker identities, masks and phase-discard provenance.

The import report carries matched/missing/refused identity counts and source
provenance. A complete-data training consumer must reject false validity cells;
an eventual training-fitted imputer belongs in Methods. The browser teaching
samples use this exact same operation. Synthetic examples are CC0 and created
offline in the Python SDK, with their seed/formulas/groups/units recorded.

The new `multimodal_cookbook` fixtures cover this operation's CATALOGUE separately
from the established DatasetSpec cookbook. Source tests are not execution proof;
run the owner and WASM gates before release.
