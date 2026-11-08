# Copy provenance manifest

> Story 5.4 / Critique C12. `nirs4all-io` is built by **copying and
> re-orchestrating** business logic from `nirs4all` (and, by design reference,
> `nirs4all-studio`). `nirs4all`'s default OSS license is **AGPL-3.0-or-later**
> with **CeCILL-2.1** as a variant; `nirs4all-io` is licensed
> **`CeCILL-2.1 OR AGPL-3.0-or-later`** so the copied logic stays
> license-compatible. Both projects are owned by **G. Beurier (GBeurier)**, who
> authorizes this reuse.

This file records every block of logic copied or adapted from `nirs4all` into
`nirs4all-io`, mapping the **source** (path in the `nirs4all` repo) to its
**destination** (module in this repo) and the **action** taken:

- **COPY-LOGIC** — the algorithm/data was reproduced (possibly re-architected).
- **GENERALIZE** — COPY-LOGIC + made declarative / format-agnostic / multi-target.
- **PORT** — translated across languages.
- **REFERENCE** — used only as a design reference (not copied).

| # | Source (`nirs4all/nirs4all/...`) | Destination (`src/nirs4all_io/...`) | Action | Notes |
|---|---|---|---|---|
| 2 | `data/parsers/normalizer.py` (alias map) | `spec/normalize.py` | COPY-LOGIC | partition×role synonym map reproduced verbatim |
| 4 | `data/parsers/folder_parser.py` (`FILE_PATTERNS`, word-boundary match) | `conventions/builtin/*.toml` + `conventions/engine.py` | GENERALIZE | patterns → declarative profiles; extension set widened beyond CSV |

_(rows are appended as each piece is copied)_
| 7,9,10 | `data/detection/detector.py`, `data/signal_type.py`, `core/task_detection.py` | `infer/` (pending) | COPY-LOGIC | detection heuristics distilled (HEADER_PATTERNS, signal scorers, water bands, task thresholds) — used by the inference engine |
| 11,12,13 | `data/selection/{role_assigner,column_selector,row_selector,sample_linker}.py`, `data/partition/partition_assigner.py` | `spec/selectors.py`, `materialize/assemble.py`, `materialize/join.py` | COPY-LOGIC/GENERALIZE | column selectors E.1; per-source alignment; partition (column/percentage) split |
| 14 | `controllers/splitters/fold_file_loader.py` | `materialize/folds.py` | COPY-LOGIC | fold-file parsing (csv-nirs4all/csv-assignment/json/yaml/txt) |
| 5,6 | `data/loaders/*` (`base.py::apply_na_policy`, csv/numpy/parquet/excel) | `materialize/loaders.py` | COPY-LOGIC | tabular reading + NA policy + param precedence + dtype/categorical |
| 15 | `data/config.py::_load_dataset`, `data/dataset.py` add_*/set_* | `materialize/spectrodataset.py` (build flow), `materialize/assemble.py` | COPY-LOGIC + EMIT(lazy) | build orchestration; SpectroDataset class imported lazily only |
| Studio role adapter | `nirs4all-studio/api/shared/dataset_config.py::build_nirs4all_config` at `cdb9f46de1643ab170934481cd9471e54ed3701c` | `crates/nirs4all-io-core/src/spec/role_tagged.rs::studio_legacy_config` | PORT + STRICT | permissive last-write behavior is retained only in a frozen oracle intermediate; the public adapter closes unknown/conflicting input first. Oracle source SHA-256 and generated corpus live in `tests/fixtures/role_tagged/studio_oracle_corpus.json`. |

## Float64 owner contract deltas (3 November 2026)

Intentional departures from the copied `nirs4all` logic, made to satisfy the accepted
IO Float64 owner contract (`PHASE3_IO_FLOAT64_CONTRACT.md`). They are deltas of the
**Python oracle** (`src/nirs4all_io`), not changes to the full `nirs4all` storage.

| Copied block | Delta | Reason |
|---|---|---|
| `data/loaders/*` numeric coercion → `materialize/loaders.py::coerce_numeric` (was `to_numpy(dtype=np.float32)`) | now an exact float64 matrix (`exact_float64`, NaN = missing); an integer that the SELECTED storage cannot hold exactly refuses (exact F64: every exactly representable integer such as 2⁵³ and 2⁵⁴ is valid, 2⁵³+1 refuses; with an explicit float32 request the original integer is rounded once, directly, never integer→F64→F32); the ±(2⁵³−1) cap and exact-F32 rule belong to classification labels only; typed readers (numpy, parquet) additionally return their declared storage dtype per column (`LoadedTable.source_dtypes`), captured before widening | the F32 default silently merged distinct F64 values and made source precision unrecoverable |
| `data/config.py::_load_dataset` weights/targets/in-memory arrays → `materialize/assemble.py`, `api.py::_assemble_in_memory` (weights and `np.asarray(x, dtype=np.float32)`) | F64 execution IR; declared source dtypes (`x_source_dtypes`, `y_source_dtype`, `weights_source_dtype`) travel beside the IR; CSV/untyped lists declare nothing | source-declared F32 stays supported, but undeclared/integer input is exact F64 |
| `data/dataset.py` add_samples/add_features → `materialize/spectrodataset.py` | SpectroDataset remains the existing **F32 feature/processing consumer**: an explicit, checked, once-only F32 projection with a consumer receipt; explicit float64 feature or numerical-target requests refuse; default targets/weights stay raw (the SDK encoder owns any numeric view) | the consumer is F32-only; no SDK/storage migration, no IO encoder |

New oracle-owned modules without a `nirs4all` source: `materialize/numeric.py`
(storage selection, quantization, exact masked blocks), `materialize/projection.py`
(base-cohort feature/target digests), and the package v4 / assembled v3 full wires in
`materialize/package.py`. Their expected bytes come from the reviewed DESIGN vectors
(`tests/fixtures/float64_design_vectors.json`, SHA-256 `b62cad5ac4f6…`), which were
generated independently of this oracle; no expected value is produced by Rust or by the
code under test. Cross-owner (Rust-versus-oracle) comparisons are a separate, later gate
run on actual exporter outputs and are not a source of expected values here.

### Follow-up deltas (owner carriers and public factories)

| Block | Delta | Reason |
|---|---|---|
| Role storage requests → raw coercion (`loaders.exact_float64/coerce_numeric`, `assemble(options)`, `api.load`, `_assemble_in_memory`) | the per-role requests reach the numeric coercion through `numeric.to_ir_f64`; integer sources convert to the selected storage directly, the IR stays F64 (F32 values widen exactly); class labels are never rounded by a numeric request; a declared float64 source keeps its already-rounded bits | int→F64→F32 double rounding flips a near-tie (2⁵⁴+2³⁰+1 must reach the upper F32) |
| `PartitionBlock` / package feature payloads | per-source `feature_names` (null = unnamed, distinct from presentation `feature_headers`), `feature_axes` (raw unit spelling or null, exact F64/string coordinates) and `representation_ids` are carried by the IR; full wires require them and the owner `audits` list | optional names, raw units and raw fold assignments must survive the full-wire roundtrip with equal preimages |
| `public_package.py` (new, no `nirs4all` source) | `matrix/masked_matrix/dense_dataset_package[_with_storage]` over the existing validated public `Dataset`/`TensorSource`; public JS integer cap, whole-content bytes and `public_content` algorithms unchanged | public-record seam of the accepted plan; labelled whole-content wrappers delegate the old bytes exactly |
| `PartitionBlock` precision carriers (`x_precisions`, `y_precisions`, `weights_precision`) | the ACTUAL selected `PrecisionRecord` of an already stored block travels beside the F64 IR (set by `DatasetPackage.to_assembled` and the assembled/package full readers; empty on a raw IR). `numeric.select_block` retains it on repackaging, or re-selects from the ORIGINAL source dtype and refuses to widen data that was normalized to F32 from a source not declared F32; an IR edited after its storage was selected refuses; processings carry their source's governing record | a requested/consumer selection is not recoverable from the source dtype alone; repackaging must keep P, bytes, preimages and the manifest root |
| Named processings (`variations`) → `assemble.py::_read_variation_frame/_load_variations`, `package.py::from_assembled`, `spectrodataset.py` | every variation's OWN declared source dtype is captured before widening (`SourceTable.variation_dtypes` → `PartitionBlock.processing_source_dtypes`) and its selected record travels as `processing_precisions`; a processing is selected under its own source dtype, the same role request/consumer, and must land on the base source's storage dtype (a default conflict refuses and asks for a common `features` request); same-storage defaults keep their own origin/source (declared F64 base + CSV child stays `source_dtype=None`). The H4 frame hashes the base `P` and each child `N` only, so a child's source metadata is carried and validated by its full-wire `ExactBlock.P`, not by the payload hash | an independently typed child must not be described as the base's source; no record-wide declaration exists in the Python IR, so none is fabricated |

### Interop corrections (cross-owner development comparison)

| Block | Delta | Reason |
|---|---|---|
| `public_package.py` audits | `owner_dataset_record` (the complete validated original record), a FULL `raw_projection_source` (sample IDs, array, presence mask, original names/axes/units) and a cohort-independent `raw_source_schema` (sample dimension `None`); all are provenance, never numeric authority | the original record is the only evidence of the optional whole-dataset identity |
| `public_package.recover_source_evidence` / `projection._verified_dataset_identity` | a serialized record is only a candidate: it must pass the existing `Dataset` owner validation and an exact rebuild (typed blocks, P, masks, axes with signed zero, folds, audits) before the unchanged labelled whole-dataset identity is computed from it; any mismatch declines the proof (identity `None`) and never alters the exact package data | a serialized record is not a certificate; the original (not storage-canonical) values keep the old identity (e.g. a withheld label `99` stays `99`) |
| `package._target_representation_id` | the manifest/payload representation is categorical for every classification kind (codebook strings and integer labels of a classification/binary/multiclass task), numeric otherwise; the H4 target bytes are unchanged | an integer classification target was labelled `target_numeric` |
| Child processing provenance | a processing's source dtype/selected P are independent of its base source under one common storage dtype (see the processing row above) | the earlier "processing follows its source" wording is superseded |
| `public_package._origin/_provenance` (supplied-input snapshot) | for a serialized/mapping public record the decoded document (read through the existing `load_dataset_document`) is deep-copied BEFORE `Dataset` normalization, that snapshot is validated, and it is retained verbatim as `owner_dataset_record` provenance (raw scalars such as a declared-float32 `1+2**-30` survive); a caller `Dataset` object, which exposes only normalized values, contributes its validated `to_dict`. Storage and the unchanged whole-dataset identity still come from the existing normalizer; the evidence/recovery rebuild revalidates the snapshot, never a normalized stand-in | the raw input is witness only: it never overrides ExactBlock/Axis authority and invents no prior bytes |

### Canonical float / integer corrections (07)

| Block | Delta | Reason |
|---|---|---|
| `strict_json.py` (new) + ingress sites (`api.to_spec`, `dataset_facade`, `DatasetSpec.from_json/from_yaml`, `folds`, index files, convention profiles) and `canonical_json` | one shared number domain: integers in `[-2**63, 2**64 - 1]` (checked before any float conversion), the `-0` token decodes to `-0.0`, float tokens must be finite, `NaN`/`Infinity` refuse; object/YAML documents get the same integer domain; the canonical writer enforces it, keeps its existing spelling/layout and asserts `sys.float_repr_style == "short"` | Rust cannot hold wider integers; `-0` is a numeric identity difference |
| `numeric.to_ir_f64` / `int_to_f32` | integer cells are decided per cell before numpy infers a float dtype (direct integer→F32 when F32 is selected; exact F64 otherwise); a genuine host float64 keeps its already-rounded bits | mixed `[n, 0.5]` lists double-rounded / silently lost `2**53+1` |
| `dataset_facade._declared_float_cells` | the public normalizer rounds every numeric cell of a declared float32 array once with the existing checked rounding (integer cells directly; overflow and nonzero→zero underflow refuse; subnormals and `±0` kept) and refuses an inexact integer in a declared float64 array; `public_content` refuses inexact integers (no silent tagged-binary64 rounding); the blanket JS cap on TYPED integers is removed (object-dtype and classification caps stay) | one corrected V1 normalizer for storage and the labelled whole-dataset identity; pre-V1 deltas beyond classes A-D include declared-float32 underflow |
| `numeric.select_block` + `package._target_table` / `project_spectrodataset` | an omitted new request preserves the carried explicit request; consumer selection refuses a carried/explicit float64 feature or numerical-target guarantee (weights are not numerical targets) | repackaging forgot `requested_dtype` and accepted the F32 consumer policy |

### Source declaration first (07c / 08)

| Block | Delta | Reason |
|---|---|---|
| `dataset_facade._declared_float_cells` (request-free source validation) | a declaration is validated and normalized once, before and independently of any storage request: declared float64 refuses an integer it cannot hold under EVERY request (including float32); declared float32 normalizes each cell once with the checked rounding (an integer cell directly from the integer), so `5e-324` refuses even under a requested float64; generic int64/uint64/untyped integers keep the direct integer→F32 rule; default/F64 inexact generic integers refuse | the request selects the output precision from already valid source values only |
| `strict_json._parse_float` | a decimal float token that parses to ±0 but has a nonzero MANTISSA digit refuses before its bits are lost (`1e-400`, `2e-324`); `0e5`, `-0e5`, `0.000e-999` stay accepted with their sign; exponent digits never count; a host F64 zero has no lexeme | pre-V1 class F |
| `numeric._cells_to_ir` / `api._target_array` | Python ints are checked against the shared domain before any conversion; a raw LIST mixing integer and float target cells is converted per cell through the shared selector (the old route rounded through numpy first); pure integer/string targets and ndarrays keep their raw semantics (no blanket cast, no encoder) | the actual target API route now yields the direct `0100805a` and refuses `2**53+1` by default |
| `public_content.ContentNotRepresentableError`, `public_package` audits | for a VALID typed-integer record whose request-free binary64 grammar cannot represent an integer (e.g. int64 `2**53+1` under an explicit float32) the audit carries `dataset_content_identity: null` plus the static `dataset_content_identity_unavailable: {profile: "nirs4all.io.dataset-content.v1", reason: "not_representable"}`; it is rebuilt/validated from the original record, survives package/assembled full wires, and a supplied reason is never proof. Invalid sources, caps and shapes still refuse; nothing is computed from a rounded record | pre-V1 classes are now A-F (E: declared-float32 underflow, F: decimal underflow) |
| common fixtures | `tests/fixtures/source_declaration_vectors.json` (70 cases, SHA 881312a5…) and `finite_float_vectors.json` (39 cases, SHA f3b0d019…) are controller-owned, consumed read-only, pinned, and compared byte for byte with the Rust crate's packaged copies | protects Cargo-packaged test data from drift |

### Numeric text keeps its lexemes until conversion (09)

| Block | Delta | Reason |
|---|---|---|
| `loaders._read_csv` / `LoadedTable.raw` / `loaders.restore_numeric_cells` / `loaders._object_cells_to_ir` / `encode_categorical` | the CSV loader also keeps the original cell text; the NUMERIC-ROLE conversion (features, targets, weights, variations; concat-samples multi-file) sends every numeric-looking token through the shared checked primitives (`strict_json.decimal_float`, `check_integer`) BEFORE pandas' lossy values are used: a nonzero mantissa that underflows to ±0 refuses, genuine zeros keep their sign, the minimum subnormal is kept; an integer above `2**53` in a column that also holds floats/NA stays an exact Python integer so the role's selection decides (default float64 refuses, requested float32 rounds directly). Metadata/categorical string fields and typed (parquet/numpy) sources are never touched; spreadsheets follow the (13) rule below. Multi-file `by_key` / `concat_features` sources carry the text through the EXISTING join machinery: a positional marker column rides each CSV frame through `join_tables` / `concat_features`, and after the merge every result row's original text is re-attached from the file row the join actually used (reordered keys, renamed duplicate headers `col__file` / `file__col`, typed files without text contribute only empty tokens); `concat_samples` stacks the text in file order (typed files too) | pandas parsed `1e-400` to `0.0` and rounded the integer cell before any selector could inspect it (pre-V1 class F also reaches CSV) |
| `public_content.metadata_number` (U07 numeric metadata column; callers `dataset_content_bytes`, `dataset_facade.to_u07_raw_sources`) | the designated numeric column's text goes through the same checked primitives: an integer-looking token stays an integer until the exact-F64 gate (the shared domain `[-2**63, 2**64 - 1]` first, then exact representability, so `9007199254740993` refuses instead of rounding to `9007199254740992.0` and `18014398509481984` is valid; `-0` keeps its sign into the zero folding), a token with a point/exponent goes through `decimal_float` (underflow/overflow refuse; `0.000e-999` is a genuine zero); ordinary metadata/class strings (including `"1e-400"`) are untouched; the tagged framing, SHA and zero folding are unchanged | class F admission example for the whole-dataset identity; no new hash or profile |

### Typed nullable integers keep their original cells (12)

| Block | Delta | Reason |
|---|---|---|
| `loaders._read_parquet` / `_exact_nullable_integers` / `exact_float64` (nullable extension dtypes) | an integer Parquet column that holds nulls is rebuilt from the original Arrow cells as pandas' nullable `Int64` / `UInt64` (public `pyarrow` + `pd.array`, within the declared minimums pandas>=2.0 / pyarrow>=14); `exact_float64` sends extension integers to the role's selection cell by cell (exact Python ints, `None` for NA → false mask cell with +0). Non-nullable integers and genuine floating columns are untouched; the declared source dtype (`int64`/`uint64`) stays truthful | `to_pandas` turned `[n, null]` into float64 and rounded `n` before the selector could look (default accepted a rounded value; requested float32 produced the lower tie) |
| `numeric.integers_to_f64` | an already decoded float array (NaN = missing) is returned as is instead of being inspected as integers | classification labels with missing cells crashed on `int(nan)` |

### Original cells reach the selection: spreadsheets, list blocks, IR axes, numeric headers (13)

| Block | Delta | Reason |
|---|---|---|
| `loaders._read_excel` / `read_table` / `restore_numeric_cells` | the sheet is read a second time with the SAME public `pandas.read_excel` call and arguments (`sheet_name`, `usecols`, `skiprows`, header) plus `dtype=object, na_filter=False, keep_default_na=False`, and the original cells ride `LoadedTable.raw` exactly like CSV text. At the NUMERIC ROLE conversion a text cell goes through the checked primitives (`decimal_float`, `check_integer`) while a physical numeric cell (already decoded F64/int) keeps its own value and bits; `1e-400` text refuses, `0.000e-999` text and a physical `0.0` are valid, a large integer text beside a float stays exact so the role decides (default F64 refuses, requested float32 rounds directly). The Excel decimal separator is always `.`; metadata, class labels, NA/blank handling and sheet/usecols selection are unchanged; no XML parser, no new dependency | pandas' Excel text parser turned the text cell `1e-400` into `0.0` and rounded `18014399583223809` before the selector could look |
| `numeric._selected_ir` / `ExactBlock.with_precision` | raw cells (list/tuple, object array, integer array) are converted cell by cell through `to_ir_f64` BEFORE any float cast, source declaration first: declared float64 holds exact values under every request, declared float32 normalizes the original integer directly once, integer/untyped sources round directly once when the selected storage is float32; a typed float array keeps its (possibly host-rounded) bits and a carried precision with `verify=True` still recovers nothing | `ExactBlock.from_matrix([[2**53+1, 0.5]])` accepted a rounded value and a requested float32 gave the lower tie because `np.asarray(dtype=float64)` preceded the selector |
| `package._checked_coordinates` (public IR `feature_axes` boundary) | numeric axis coordinates convert through `to_ir_f64`: an integer F64 holds exactly (`2**54`) is a valid coordinate, an inexact one (`2**53 + 1`) or one outside the shared domain refuses, a genuine float keeps its bits (sign of zero included), strings stay valid, numeric/string mixes, bools and non-finite values refuse; the stored coordinates are F64 so the wire framing is unchanged. The portable `PublicDatasetV1` axis cap (`public_dataset`) is a different contract and is NOT changed | the boundary accepted only `float` instances |
| `assemble.tabular_axis` | numeric header text goes through the same checked conversion instead of `float(header)`: integer text through the exact-F64 gate (`2**54` valid, `2**53+1` refuses), decimal text through `decimal_float` (`1e-400` and overflow refuse, `0.000e-999` is a zero with its sign); label headers and generated index headers are unchanged and the raw `feature_names` strings / raw unit are kept | `float(header)` erased the known integer/decimal text |

### Carried-precision verification keeps the original values (14)

| Block | Delta | Reason |
|---|---|---|
| `numeric._selected_ir(..., verify=)` / `ExactBlock.with_precision` | with `verify=True` (a retained selection, e.g. repackaging) the matrix claims to ALREADY hold the selected stored values: raw cells go through the exact-F64 conversion only (an integer F64 cannot hold refuses) and no storage selection or direct integer→F32 rounding runs before the stored result is compared with those originals, so a retained integer such as `16777217` under float32 refuses exactly like the equivalent float array and `16777216` is accepted. `verify=False` keeps the first-selection behavior (direct integer→F32, source declaration first); typed float arrays keep their bits; masks and explicit new requests are unchanged | the 13 change converted a retained integer list to the selected storage before the comparison, so `with_precision([[16777217]], p, verify=True)` and `DatasetPackage.from_assembled` silently re-rounded a modified carried IR (this supersedes the carried-precision statement of READY-13) |

### Four golden fixtures updated (15, post-parity)

| Block | Delta | Reason |
|---|---|---|
| `tests/goldens/contract/{single_combined,train_test,x_y_separate}.assembled.canonical`, `tests/goldens/dataset_package_v4.cross_language.canonical` (renamed from `dataset_package_v3.cross_language.canonical`; no compatibility alias or reader) | the four goldens carry the exact bytes of the actual Python producers captured from source14 (copied, never recomputed); the helpers `read_v2_golden` are now `read_v3_golden` and keep their strict `assembled_schema_version` guard | both owners produced byte-identical bytes for all four goldens in the controller's paired-owner qualification (`phase3-io-owner/paired-owner-qualification`), so the pending update is now intentional |

### Owner corrective batch (16, Opus corrective-plan closure)

| Block | Delta | Reason |
|---|---|---|
| `package._select_features` / `_select_weights` / `finalize_selected_storage`, `assemble.assemble` / `_assemble_raw`, `numeric.effective_coercion`, `api.load` / `_assemble_in_memory` (C1) | the per-role selection inside `DatasetPackage.from_assembled` moved, unchanged, into shared helpers over `select_block`; `finalize_selected_storage` commits X/processings + records when features or a consumer is requested, Y + `y_precisions` only for a `targets` request, weights only for a `weights` request. The public `assemble` is `_assemble_raw` plus that finalizer, `_assemble_in_memory` finalizes too, and `load(target="assembled")` keeps the REAL request and consumer (the SpectroDataset F32 coercion is now derived inside `_assemble_raw`, the fake `features="float32"` is gone). `target="package"` selects once in `from_assembled`; `target="spectrodataset"` keeps its own boundary. `load(DatasetPackage, target="assembled", <storage option>)` refuses like the package target | an F32 assembly still returned the raw F64 IR with empty records, so a later `float64` request recovered bits F32 had discarded; unrequested raw SDK int64 labels must keep their dtype |
| `public_package._masked_cells` / `_masked_materialization` / `_package` / `_assemble` / `verified_dataset_identity` (C2) | the masked route validates a working copy whose literally false target cells (null, bool or finite number only) carry the canonical placeholder (false / 0.0 / 0) BEFORE the dtype, label and declared-array gates; the audit, `source_evidence` and the whole link keep the ORIGINAL record, and an original that is not a valid whole record gives link `None` with no reason | Rust discards withheld-cell placeholders first; Python refused null, float32 overflow/underflow, retained integers and bool placeholders that Rust accepted. Observed cells and the matrix/dense routes stay strict |
| `loaders.restore_numeric_metadata` / `NumericRefusal` / `refuse_numeric_cells`, `LoadedTable.text_tokens`, `assemble._build_source_table` / `_assemble_block`, `package._numeric_object_cells` / `_series_from_wire` (C3) | a SELECTED metadata/ID column of numeric-only delimited text is numeric authority. The column is classified ONCE over ALL original rows of the loaded input, before any NA removal, and survivors are never reclassified: all-integer columns stay exact integers; otherwise the column is the plain correctly rounded float column, unless RETAINED-INTEGER mode is triggered, and only an INTEGER token triggers it (an integer outside `[-2**63, 2**64 - 1]`, or an in-domain integer F64 cannot hold beside a float/NA/decimal peer): then integer tokens stay exact integers and float tokens stay floats. An overflowing/underflowing DECIMAL is a float token and never triggers it. Validation runs on the SURVIVING selected cells only (an out-of-domain integer or a decimal overflow/underflow refuses; a removed cell never refuses). A genuine text column stays literal even if its text peer row is removed; spreadsheet cells and unselected columns are untouched; an object column of exact numbers is labelled `numeric` and survives the full-wire readers | `[2**53 + 1, 1.5]` was silently rounded so two distinct IDs merged, `[u64max, ""]` stayed text and `1e-400` became `0.0` in metadata; the Opus row-removal decision (`C3_ROW_REMOVAL_DECISION.md`) fixed the classification/validation order and exposed that a removed `1e-400` row wrongly turned a surviving `3` into the integer `3` (now the float `3.0`, ID `3.0`) |
| `loaders._refuse_oversized_unsigned` (C5) | a Parquet `UInt64` above `i64::MAX` refuses on the typed Arrow cells of every column the reader reads, any role; `format.columns` exclusions are not read | the owner's Parquet table is an i64 integer domain; Python accepted and rounded such cells |
| `projection._identity_text` (C6) | a finite float identity is its Python `repr` and then meets the ordinary identifier grammar (`1e+16` fails on `+`, no cap); a non-finite one is `Missing projection identity` | Rust renders `str(float)`; Python refused every float ID |
| `public_content.dataset_content_bytes` (C7) | after normalization every int64 class label of a `classification` record passes the existing `validate_label` (cap, no storage option); regression targets, float labels and groups are not labels | Rust applies the portable cap in the whole operation; Python hashed `2**53` labels |
| `canonical_json` module docstring (C8) | the stale normal-range paragraph is replaced by the accepted float spelling; no code | the docstring contradicted the shipped formatter |
| `multimodal._array_from_dict`, `dataset_facade._declared_float_cells` / `load_multimodal_definition`, `bindings/python/python/nirs4all_io/multimodal.py` (E4/B3) | the one declared-float gate runs on every serialized array record (function-level import, integer domain check first), the facade pre-pass is removed, and the wheel copy is the byte-identical mirror. The mirror needs the Stage 4 native `_declared_float_cells` and raises `ImportError` until then (no stub, no numeric copy) | raw `from_dict` accepted `5e-324` float32 cells and retained float64 integers that every other route refused; the SDK's only direct caller bypassed the gate |
| `dataset_spec.outlier_threshold_number`, `AggregateSpec.from_dict`, `package._aggregate_from_wire` (O3) | a PRESENT `aggregate.outlier_threshold` must be a finite JSON number (bool, string, null, non-finite refuse), absent stays 0.95; the typed spec and the full-wire reader share the one check; only an absent/null `aggregate` is "no aggregation", so an empty object is a present block with the defaults (the Rust `from_value` behaviour) | `float()` coerced strings and `True` silently |

Tests that encoded the superseded behaviour were corrected, not their goldens: `test_typed_nullable.py` (Parquet UInt64 above `i64::MAX` now refuses), `test_csv_numeric_text.py` (a numeric-only metadata column with an underflowing token refuses), and two consumer receipts that loaded `target="assembled", features="float32"` only to feed the consumer (an explicit request is now committed and stays `origin=requested`).

### Post-B4 owner corrections (17, Opus final work review items 1, 2, 3, 5, 6, 7)

| Block | Delta | Reason |
|---|---|---|
| `dataset_facade._float_width` / `_declared_float_cells` (item 1) | the single E4 array gate resolves the declared dtype with NumPy first: any spelling of a 4- or 8-byte float (`<f8`, `>f4`, `f8`, `float`, `double`, `d`, `single`, ...) gets the same cell check as `float32` / `float64` (direct integer->F32 rounding, inexact declared-F64 integer and nonzero F32 underflow refuse). Genuine typed arrays and the non-finite pass-through are unchanged | the gate matched only the literal strings, so `<f8` / `<f4` stored `2**53`, double-rounded `0000805a` and zeroed `5e-324` silently |
| `public_dataset._public_dtype_vocabulary`, `Dataset.__init__` (item 1) | the public record independently enforces the closed dtype vocabulary of the Rust owner (`object`, `bool`, `float64`, `float32`, `int64`/`uint64`/`int32`/`uint32`/`int16`/`uint16`/`int8`/`uint8`, `<U<n>`); every other alias refuses and no Rust alias is added | the low-level array reader accepts any NumPy spelling, the public profile never did |
| `public_dataset.Dataset.__init__` (item 7) | an integer (int/uint) `groups` array runs the existing `integers_to_f64` exactness check: `2**53 + 1` refuses, `2**54` stays valid; no generic text/ID cap | Rust refuses inexact integer groups at ingest; Python accepted them |
| `loaders.restore_numeric_cells(defer=True, segments=...)` / `NumericRefusal` / `refuse_numeric_cells`, `assemble._build_source_table` / `_assemble_block` (items 2, 5; residual A) | an out-of-domain integer or an overflowing/underflowing decimal in a numeric-role column (features, targets, weights) is kept as a marker and raised only if its row survives to the assembled block: a row removed by NA policy OR dropped by a JOIN never refuses (the approved rule has no NA-only exception, `_refuse_lost_numeric_cells` was removed), a kept cell still refuses; the original column classification is unchanged, variation frames stay eager. A `concat_samples` source gives every loaded input its own numeric authority (a text input cannot erase another input's refused token) | with a floating peer a removed row's bad cell refused in Python while Rust kept the surviving row (18 of 26 role combinations) |
| `loaders.restore_numeric_metadata(segments=...)` / `_typed_cells` / `_stacked_inputs` (item 5) | selected metadata/ID columns of a `concat_samples` source are classified per loaded input before the rows are stacked (a float input keeps `4.0`, an all-integer input keeps its integers, a text input keeps literals, a numeric-only input still refuses), both input orders | the merged frame was reclassified as one column |
| `projection._identity_text` (item 3) | a boolean sample ID renders as CPython `True` / `False` | Rust renders `to_str_scalar`; feature and target digests differed |
| `loaders._restore_na_tokens` (item 6) | a CSV column pandas could only read as text (e.g. `[u64max, ""]`) gets its NA-vocabulary tokens back as missing cells before the NA audit | the NA audit and later missing-value rules never saw them; Rust records the audit |

### Residual A and C of the Opus 5.6 work review (18)

| Block | Delta | Reason |
|---|---|---|
| `assemble._assemble_block` (A) | `_refuse_lost_numeric_cells` and its call are deleted; `tests/test_csv_merge_text.py` now pins that an unmatched join row never refuses and a kept matched cell does | the approved rule has no NA-only exception: a lookup `coverage: drop` row or a right-unmatched row with an out-of-domain/underflow cell and a float peer refused in Python and was accepted by Rust |
| `assemble._load_source_frame` / `loaders.restore_numeric_metadata` / `_stacked_inputs` (C) | a `concat_samples` source also returns every loaded input's cells as that input loaded them (object dtype); a non-numeric input (boolean, text, all-NA) contributes those typed cells, so `[TRUE, FALSE]` + `[3, 4]` is the object column `[True, False, 3, 4]` in both file orders, never strings or a float/int coercion; numeric inputs keep the per-input classification and the survivor validation | pandas unified a boolean and an integer input to integers and the previous code replaced a non-numeric input by its raw tokens (`"TRUE"`); the numeric roles keep the established bool = 1.0 / 0.0 |

The CSV reader additionally corrects existing inferred float64 cells from original
integer/decimal tokens with CPython `float`, after NA-token restoration and before
NA fill (`loaders._exact_decimal_columns`). This oracle-owned precision delta
preserves binary64 bits, including signed zero, across pandas versions; integer
and object columns and the later raw-token domain checks retain their rules.

### Residual ingestion owner closure (approved A1–A5)

The Python loader routes StringDtype numeric-role cells through the existing
checked cell-to-IR conversion and preserves original non-categorical targets.
Per-input retained integer/refusal authority includes typed and variation peers;
private decimal scopes follow each source column and original row through joins.
Non-dot CSV text/category lexemes remain literal, while numeric views alone apply
the originating input's decimal. Physical Excel Bool+missing cells retain Bool
identity in the existing object representation. Relational keys retain raw typed
authority and use exact typed comparison codes, with surviving selected payload
validation unchanged. A separate read with the existing pandas engines rejects
physical CSV width overflow before inferred loading; no parser or numeric kernel
is copied, and no binding or SDK compensation is added.
