---
orphan: true
---

# IO operations and contracts

Discover current capabilities from the owner manifests, `API.md`, `VERSIONING.md`,
`../COMPAT.md`, `../bindings/SPEC.md` and per-binding documentation. The architecture
and command examples below are navigation and gate obligations, not proof that every
decoder, binding, task or wire version is implemented or qualified. Claims must name
the profile, operation, artifact/build/version, platform, executed evidence and
skips/refusals; do not freeze transient cohort counts or acceptance IDs into policy.

## Commands

### Rust workspace (the production implementation) — green gate

Run all four before reporting Rust work complete (mirrors `.github/workflows/ci.yml`):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace --no-default-features

# Targeted
cargo test -p nirs4all-io-core            # pure-logic crate
cargo test -p nirs4all-io contract_goldens   # one integration test target in the facade crate
cargo run -p nirs4all-io-cli -- to-spec <dir>   # exercise the CLI binary
```

Discover workspace membership in the current root `Cargo.toml` (`crates/nirs4all-io-core`,
`-io`, `-io-capi`, `-io-cli`, `-io-dagml` are the architecture entrypoints below).
The language bindings build with their own toolchains and are deliberately
workspace-excluded — `cargo test --workspace` will NOT touch them:
- `bindings/python` (maturin/pyo3) and `bindings/wasm` (wasm-pack) — see below.

`crates/nirs4all-io-dagml` is now a workspace member. Its default Cargo dependency resolves
`dag-ml-data` from crates.io for standalone builds, while the cross-CLI conformance harness patches
Cargo to the sibling checkout when one is present.

### Python MVP / oracle — green gate

The root `pyproject.toml` is the **private** `nirs4all-io-oracle` distribution (`Private :: Do Not
Upload`); the published `nirs4all-io` Python package is the pyo3 binding in `bindings/python`. The
oracle is installed editable. Use the ecosystem venv if one exists, else install into the active
interpreter:

```bash
pip install -e ".[dev]"      # ruff, mypy, pytest + pyarrow/openpyxl/scipy + nirs4all & nirs4all-formats (dev oracles)
# CI cannot install the non-PyPI oracles; it uses: pip install -e ".[parquet,excel,dagml]" ... && pytest -m "not parity"

ruff check .                 # lint: E,F,I,W,UP,B; line-length 220; E501 ignored
mypy .                       # type check (py311, ignore_missing_imports); excludes ^bindings/
pytest                       # all tests; parity tests auto-skip if nirs4all is absent

# Targeted
pytest tests/test_spec.py
pytest tests/test_cookbook.py::test_coverage_matrix_complete
pytest -m parity             # ONLY the parity-oracle tests (needs nirs4all installed)
pytest -m "not parity"       # everything except the parity oracle
```

Two registered pytest markers (see `pyproject.toml`): **`parity`** (imports `nirs4all` read-only as
an oracle) and **`formats`** (needs the `nirs4all-formats` reader library).

### Bindings

```bash
cd bindings/python && maturin develop && pytest tests   # pyo3; qualify SpectroDataset support against the current binding contract
cd bindings/wasm   && wasm-pack build --target nodejs    # fs-free, in-memory: toSpec/validate/version + inferFiles/inferDataset/proposeDataset/inferRecords/assembleDataset/loadSummary
cd bindings/r      && ./build_and_test.sh                # .Call over the C ABI
# MATLAB/Octave (bindings/matlab) is CI-gated; needs octave/MEX locally.
```

### Browser dataset builder (`web/`)

Static page (deployed to GitHub Pages by `deploy-pages.yml`) that drives the IO WASM binding together
with the `nirs4all-formats` WASM (vendor decoding) — one-shot `inferDataset` mode and an iterative
`proposeDataset` mode where confirmed decisions are fed back as locks. The WASM bundles are vendored
under `web/vendor/`; rebuild them with `web/build-wasm.sh`. See `web/README.md`.

```bash
cd web && ./build-wasm.sh && python3 -m http.server 8080   # dev page
cd web && node make-standalone.mjs                          # -> dist/nirs4all-io-standalone.html
cd web && node tests/standalone-smoke.mjs                   # Playwright smoke (also -formats-, -all-samples-)
```

`web/dataset-builder.html` is a second, owner-backed page for delimited tables (`describeTables` →
`proposeDataset` → documented `DatasetSpec` overlay → `validate`/`to_spec` → `assembleDataset`). Its column
table is composed from three public `nirs4all-ui/datasetBuilder` components, bundled from the accepted `.1`
package (SHA-256 verified, extracted into a fresh temp directory; no sibling `nirs4all-ui/dist`):

```bash
cd web && node build-dataset-builder.mjs --tgz <nirs4all-ui-....tgz> --node-modules <dir with react, react-dom, esbuild>
cd web && node tests/dataset-builder-smoke.mjs              # served Playwright smoke, non-localhost requests blocked
```

### Versioning

The single source of truth is `[workspace.package] version` in the root `Cargo.toml`;
`scripts/bump_version.sh` syncs every binding manifest (Python/WASM Cargo.toml, R `configure` /
`DESCRIPTION` / `Cargo.lock.rust`, `src/nirs4all_io/_version.py`). `version-sync.yml` fails on drift.

```bash
scripts/bump_version.sh --check            # verify no manifest drifts
scripts/bump_version.sh --bump X.Y.Z[-pre] # rewrite the SoT, then sync
```

## Architecture: the four stages

The same four-stage pipeline exists twice, module-for-module — once in Python (`src/nirs4all_io/`,
the table below) and once in Rust (`crates/`, mapped at the end of this section). Read the Python
description first; the Rust crates own the same concepts.

Each `src/nirs4all_io/` subpackage owns one stage of the pipeline. `api.py` is the public glue.

| Stage | Module | Input → Output | Notes |
|---|---|---|---|
| **RESOLVE** | `resolve/resolver.py` | any input → ordered `InputSet` | stamps a **stable identity** (abspath / `array:<n>` / `object:<id>`), content hash, extension hint, and **sidecar grouping** (ENVI `.hdr` etc.) on every item. Deterministic ordering. Retrofitting identity later would break fingerprints, so it's foundational. |
| **CONFIGURE** | `conventions/` + `spec/normalize.py` | filenames / legacy-dict → `DatasetSpec` dict | `conventions/` matches filenames against declarative TOML **profiles**; `normalize.py` maps legacy `nirs4all` config keys + synonyms onto the canonical spec. |
| **(the IR)** | `spec/` | — | the canonical, validated, serializable `DatasetSpec`. The single source of truth. |
| **INFER** | `infer/` | `InputSet` → scored `DatasetPlan` | composes convention match + per-file `describe` + value detectors (signal/task) + **column-role inference** (the genuinely new bit) into a `DatasetPlan` whose `.resolved_spec` `load` can execute. Scores are **uncalibrated** (ranking/triage only). |
| **MATERIALIZE** | `materialize/` | `DatasetSpec` → `AssembledDataset` → `SpectroDataset` | load → merge multi-file → relational join → role-split columns → partition. |

### The `DatasetSpec` IR (`spec/dataset_spec.py`)
Everything funnels into a `DatasetSpec`; the materializers consume **only** it. It round-trips through
dict/YAML/JSON, is structurally checked by `spec/validate.py`, and has a **versioned JSON Schema**
(`spec/json_schema.py` + `dataset_spec.schema.json`, `SCHEMA_VERSION = 1`) that is the wire contract.
Closed vocabularies live in `spec/enums.py` (all `str` enums, case-insensitive `.coerce()` with a few
aliases). Column selection is a small DSL in `spec/selectors.py` (positional / slice / names /
`name_range` / `regex` / `dtype` / `rest` / `auto`).

### `AssembledDataset` is the target-agnostic seam (`materialize/assemble.py`)
`assemble(spec)` produces an `AssembledDataset` (per-partition `PartitionBlock`s: multi-source `X`,
`y`, `metadata`, headers, units, weights, named processings). `to_spectrodataset(assembled)`
(`materialize/spectrodataset.py`) adapts it to nirs4all; `DatasetPackage` wraps the same IR for
target-agnostic payload manifests; the Rust bridge crate adapts it to `dag-ml-data` via
`to_dag_ml_data(&AssembledDataset)`. Keep assembly logic target-agnostic so it stays testable
without nirs4all — `target="assembled"` is the test seam.

### The Rust workspace mirrors the Python stages, split core ↔ facade

The Rust port deliberately splits along a **pure-logic / file-IO** seam (rule below):

| Crate | Role |
|---|---|
| `nirs4all-io-core` | **Pure logic, no file IO.** Owns the `DatasetSpec` IR (`spec/`), selectors, conventions, normalize, inference over a *neutral* `FileDescription`/`TableProfile` it does **not** populate, validation, `canonical_json`, `pyfmt` (Python-`repr`-faithful float formatting), and the fs-free `materialize` (`assemble`). |
| `nirs4all-io` (the **facade**) | **All file IO.** Resolution, tabular loaders, the join engine, the `nirs4all-formats` vendor reads, and producing the neutral descriptions fed to `-core`. Wires `resolve → infer → configure → materialize`. Re-exports `-core` as `core`. |
| `nirs4all-io-capi` | C ABI (`n4io_to_spec` / `infer` / `validate`, JSON in/out; opaque context + per-context error buffer; ABI versioning). Symbol surface is drift-guarded (`expected_symbols_*.txt`, `tests/abi_surface.rs`). |
| `nirs4all-io-cli` | The `nirs4all-io` binary (`infer` / `to-spec` / `validate` / `load` / `emit-dag-ml-data`). |
| `nirs4all-io-dagml` | `to_dag_ml_data(&AssembledDataset)` → `CoordinatorDataPlanEnvelope`. Uses crates.io by default; the ecosystem conformance harness patches `dag-ml-data` to the sibling checkout. |

**IR canonical JSON has named Python/Rust byte-parity contracts.**
`canonical_json.rs` and `canonical_json.py` share the frozen canonical corpus
(`tests/test_canonical_json.py`, `crates/nirs4all-io-core/tests/canonical_parity.rs`);
`tests/goldens/contract/`, Rust `contract_goldens.rs`/`assembled_goldens.rs` and binding
parity checks retain their specifically declared output profiles. This does not
assert universal cross-language in-memory equality or qualify every binding route.
`serde_json` runs with `preserve_order` (input map order mirrors Python dict order)
**and** `float_roundtrip` (correctly-rounded float parse matching CPython); canonical
JSON sorts keys explicitly rather than relying on the map type. Public-content
identity is a separate tagged-tree contract, not this IR encoding.

The convention TOMLs are **copied** into `-core` and kept in sync by `tests/test_convention_toml_sync.py`.

### Multimodal / provider surface (Python, outside the tabular `DatasetSpec` path)

A second, independent Python surface handles explicitly identified in-memory sources without
flattening tensors:

- `multimodal.py` — `MultimodalDataset` of `TensorSource`s (representation IDs / semantic axes from the
  published `dag-ml-data` registry); only deterministic identity alignment and row selection.
- `ragged.py` — `RaggedSeriesSource`/`RaggedSeriesBatch`: packed variable-length series with explicit
  boundaries; no padding/truncation/interpolation (see `docs/RAGGED_SERIES.md`).
- `provider.py` (+ `provider_adapters.py`) — `DataProvider`/`ProviderBatches`: finite host providers and
  resumable cohort views, with sklearn/PyTorch adapters (CI job `provider-framework-adapters`).
- `dataset_facade.py` / `public_dataset.py` — user-facing projections of the versioned multimodal record.
- `public_content.py` — a **separate** content-identity contract (tagged tree, binary64 numbers); it is
  not the IR canonical JSON, don't merge the two.

These never encode, impute, fit, split into folds or invoke a framework — that belongs to the ML
runtime (Methods / DAG-ML / Core). `to_dataset_package` / `DatasetPackage` (`materialize/package.py`)
is the target-agnostic payload package built from the tabular IR; discover its
current wire version and refusal rules in `VERSIONING.md` and the owner API.

## Load-bearing rules (do not break these)

These are the architectural invariants. PRs that violate them are rejected.

1. **No runtime dependency on `nirs4all`.** `import nirs4all_io` must never import `nirs4all`; only
   `to_spectrodataset` (i.e. `load(..., target="spectrodataset")`) may lazily import it, at call
   time. Enforced by `tests/test_import_boundary.py` (runs in a subprocess and asserts no
   `nirs4all*` modules leaked). `nirs4all` is a **dev/test-only parity oracle**, never a runtime dep.

2. **Never re-parse vendor files.** Vendor byte-decoding (OPUS/JCAMP/SPC/ASD/…) is delegated to
   `nirs4all-formats`, imported lazily. Tabular loading + NA policy in `materialize/loaders.py` is
   **copied** from `nirs4all` (not re-derived) — see `COPY_PROVENANCE.md`, which maps every copied
   block source→destination. Update that manifest when you copy more logic.

3. **It's a loader, not a splitter.** All partition modes are **deterministic by construction**:
   `partitions.by` ∈ `column` / `index` / `index_file`. Percentage / stratified / shuffled splits are
   *intentionally rejected* — they belong in the pipeline's CV layer. Do not add a "random split."

4. **io owns the dataset layer; don't reimplement it up- or downstream.** roles, multi-source, joins,
   merges, partitions, folds, signal/task inference, conventions, the `DatasetSpec` IR, and
   `SpectroDataset` materialization are this repo's responsibility. The host (`nirs4all` /
   `nirs4all-studio`) keeps everything downstream of a built `SpectroDataset` (pipeline, UI, storage).

5. **Parity is the correctness bar.** For supported topologies, `load(...) → SpectroDataset` must
   equal `nirs4all.DatasetConfigs(...)` (`tests/test_parity.py`). The build flow in
   `spectrodataset.py` deliberately mirrors `DatasetConfigs._load_dataset`.

6. **`nirs4all-io-core` is pure logic — no file IO.** It performs inference over *neutral*
   `FileDescription`/`TableProfile` structs it does not populate; the **facade** (`nirs4all-io`) owns
   all filesystem access, vendor reads, and producing those descriptions. Don't pull `std::fs` or
   `nirs4all-formats` into `-core`.

7. **Require byte parity for the named Python/Rust IR canonical-JSON profiles.**
   Keep the shared `tests/goldens/contract/` expectations and the canonical tests
   named above. Keep non-recursive profile resolution and the declared float-text profile.
   Refuse an ambiguous used file reference that matches distinct physical identities
   before reading; unassigned duplicate names retain warnings, repeated one-identity
   inputs keep operation-specific semantics. See [filesystem references](DATASET_CONFIGURATIONS.md#filesystem-references-aliases-and-literal-wildcard-names).
   Update Python + Rust together when changing shared load-bearing logic; retain the Python
   dev/parity oracle. Never re-bless goldens merely to force green: intentional
   contract changes require independently justified expected values and the
   owner's versioning/provenance rules.

   Assembled storage is representation-specific: Rust
   `DatasetPackage::to_assembled` preserves canonical masked `+0`/false storage;
   the Python ndarray view and raw IR retain logical NaN/false. Observed signed-zero and
   subnormal bits remain intact in selected numeric storage and IR canonical JSON.
   Tagged `nirs4all.io.dataset-content.v1` retains its declared zero folding;
   join/group keys preserve numeric identity, including matching `0.0` and `-0.0`.
   F32 selection rounds or refuses as declared. This is not universal
   cross-language in-memory equality. Keep IR JSON, package storage/wires,
   logical views and tagged public-content identity distinct; require independent
   expected-value, named-profile and identity checks for each requested operation.
   Unsupported input/profile/operation must retain explicit errors/refusals;
   decoder or binding availability does not qualify every task or route.

8. **The `dag-ml-data` emit lives in the `nirs4all-io-dagml` bridge crate.** It is a workspace
   member, but default dependency resolution stays ecosystem-free via crates.io. The cross-CLI
   conformance harness patches Cargo to the sibling `dag-ml-data` checkout when present, and CI now
   requires that sibling path. Retain supported bridge operations and explicit refusals;
   check the named profile in `IO_XLG_QUALIFICATION.md` against `VERSIONING.md`; do not extrapolate a rounded assembled-summary qualification to exact full assembled/package payloads or other wires, and never replace an implemented path with a stub.

## Public API (`api.py`, re-exported from `__init__.py`)

These examples describe the Python dev/parity oracle. Discover the published
binding's current entrypoints, targets and refusal rules in `API.md`,
`../COMPAT.md` and its own documentation; do not infer them from oracle examples.

```python
import nirs4all_io as nio
plan = nio.infer(<input>, conventions=[...])     # -> scored DatasetPlan (plan.resolved_spec, .recommendations, .warnings)
ds   = nio.load(<input | spec | plan>, target="spectrodataset" | "assembled", base_dir=, name=)
spec, base = nio.to_spec(<input>)                # resolve only, no materialization
desc = nio.describe(<file>)                      # neutral per-file descriptor (delimiter, header unit, axis)
pkg  = nio.to_dataset_package(<input>)           # versioned target-agnostic DatasetPackage (+ nio.describe_dataset_package)
# IR types: nio.DatasetSpec (.from_dict/.from_yaml/.to_dict/.validate), nio.DatasetPlan, nio.AssembledDataset
# Multimodal: nio.MultimodalDataset, TensorSource, RaggedSeriesSource/Batch, DataProvider, ProviderBatches
```

`load(target="spectrodataset")` accepts `spectro_dataset_cls=` to inject a recording double, so the
adapter is testable with no nirs4all installed (see `test_load_e2e.py`).

## Testing specifics

- **Cookbook coverage gate** — `tests/test_cookbook.py::test_coverage_matrix_complete` introspects
  which vocabulary each fixture spec exercises and **fails if any load-supported element**
  (selector / merge / cardinality / coverage / partition / fold / `lookup` / `variations` /
  `role:weights` / `auto`) **has zero fixtures**. Treat "added load-supported vocabulary" as
  unshipped until it has a cookbook fixture in the `CATALOGUE`.
- **Inference corpus** — `tests/test_inference_corpus.py` is a labelled-corpus per-decision precision
  check for `infer()`.
- Tests cover each stage in isolation: `test_resolve`, `test_normalize`, `test_conventions`,
  `test_spec` / `test_json_schema`, `test_infer`, `test_loaders` / `test_join`, plus `test_load_e2e`
  and `test_hardening` (adversarial inputs → clear `SpecError`s).

## Gotchas

- **The Python package has no CLI; the CLI is the Rust binary.** `crates/nirs4all-io-cli` ships the
  `nirs4all-io` binary (`infer` / `to-spec` / `validate` / `load` / `emit-dag-ml-data`); the dead
  `nirs4all_io.cli:main` console-script entry was removed from `pyproject.toml`. The Python MVP surface
  is library-only (`nio.infer` / `nio.load` / `nio.to_spec`).
- **Two `nirs4all_io` Python packages exist.** The MVP (`src/nirs4all_io/`) and the pyo3 binding's
  mixed-maturin package (`bindings/python/python/nirs4all_io`, wrapping the native `_native` module).
  The root `mypy` run excludes `^bindings/` so they don't collide; don't remove that exclude.
- **`cargo test --workspace` covers the Rust crates, including `nirs4all-io-dagml`, but not the
  bindings** — run their own toolchains (maturin / wasm-pack / R script) to test them.
- **`rtk` can mask a command's exit code.** When verifying a green gate, capture the status explicitly
  (`cmd > log 2>&1; echo $?`) rather than relying on `&&` chains.
- **YAML round-trips lists, not tuples** — specs are dict/JSON/YAML-authorable and stay `str`-enum
  clean precisely so they survive serialization; don't put tuples in a spec.
- The repo is a clean tree: **no dead/deprecated code, no backward-compat shims.** Remove rather than
  deprecate.
- **All project content is in English** (code, comments, docs, test prose, commits, UI text in `web/`) —
  see `AGENTS.md`; conversation with the maintainer may be in French.

## Where to look it up

- `docs/DATASET_CONFIGURATIONS.md` — the **complete reference**: every input form, `DatasetSpec`
  field, selector, merge mode, join, partition, fold, loading param, supported/out-of-scope layout,
  and a use-case cookbook with honest ✅/🟡/📋 status per option. Read this before adding spec vocab.
- `docs/API.md` — the stable integration seam and how a host adopts it.
- `docs/development.md` — contributor entry point, current references, verification and private archive policy.
- `docs/ROADMAP.md` — intentionally deferred work; validate historical items against current source.
- `docs/_private/README.md` — local-only index of historical plans, acceptance records and AI specifications;
  ignored by Git and excluded from Sphinx. Never force-add private records. Historical Phase-2
  STATUS, RUST_REWRITE_ROADMAP, PHASE2_GATE and PORT_BLUEPRINT are archived there.
- `docs/REPLUG.md` — host-adoption sequence and the io-vs-host ownership split.
- `docs/VERSIONING.md` — the three versioned contracts (schema / convention-profile / canonical-JSON).
- `COMPAT.md` — the per-binding operation matrix (which of infer/to-spec/validate/load each supports).
- `bindings/SPEC.md` + each `bindings/*/README.md` — the binding contract and per-language build/test.
- `COPY_PROVENANCE.md` — what was copied from `nirs4all` and how (license-compatibility record).
- `AGENTS.md` at this repository root is the local owner policy; parent-workspace maps are optional navigation, never required policy inputs.
