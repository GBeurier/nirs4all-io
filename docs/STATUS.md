# Status

**Phase 1 (Python MVP) — COMPLETE & Codex-ACCEPTED** (2026-05-27).
200 tests pass; ruff + mypy clean; parity-verified against the real `nirs4all`.

Post-acceptance additions (2026-05-28): `folds.column`, `variations`,
`role: weights`, `partitions.by: index` / `index_file`, `selector: auto` at
load time. Drop of `partitions.by: percentage` (and its `stratify` / `shuffle`
/ `random_state` knobs) on the principle that `nirs4all-io` is a *loader, not a
splitter*; all partition modes are now deterministic by construction. Updated
deferred items live in [`ROADMAP.md`](ROADMAP.md).

**Phase 2 (Rust rewrite) — IN PROGRESS** (branch `feat/rust-rewrite-phase2`,
[`RUST_REWRITE_ROADMAP.md`](RUST_REWRITE_ROADMAP.md); Codex-reviewed per EPIC).

| EPIC | Scope | State |
|---|---|---|
| 7 | Workspace (4 crates) + contract anchor + canonical-JSON parity (the #1 risk) | ✅ + Codex |
| 8 | Rust core+facade port — `resolve→infer→configure→materialize` **byte-identical** to Python (`to_spec`/`infer`/assembled goldens) | ✅ + 2 Codex |
| 9 | C ABI `n4io_*` (status/error model, opaque context, ABI versioning) + symbol governance + `nirs4all-io-cli` | ✅ + Codex (+4 fixes) |
| 10 | dag-ml-data emit (`AssembledDataset`→`CoordinatorDataPlanEnvelope`) in the workspace-excluded `nirs4all-io-dagml`; validated by **both** dag-ml CLIs | ✅ + Codex (+3 fixes) |
| 11 | Bindings: pyo3 (tested), wasm (tested), R (tested), MATLAB/Octave (CI-gated). Import-boundary green | ✅ + Codex (+1 fix) |
| 12.1 | **Direct parity oracle**: the pyo3 binding builds a `SpectroDataset` via the lazy adapter; `tests/test_parity.py` proves Rust→SpectroDataset X/y/task/headers ≡ `nirs4all.DatasetConfigs` (allclose). CI: `parity-oracle.yml` (ecosystem-gated) | ✅ |
| 12.2 | Cookbook coverage gate in Rust — all 28 vocabulary elements driven through `assemble()` (`crates/nirs4all-io/tests/cookbook.rs`) | ✅ |
| 12.3 | Cross-language goldens: pyo3 binding `to_spec`/`infer` byte-identical to the contract goldens | ✅ (binding) |
| 12.5 | Property/fuzz tests (proptest) + `float_roundtrip` (correctly-rounded JSON float parse, matches CPython) | ✅ + Codex |
| 12.4 / 12.6 / 12.7 | Cross-binding behavioral-parity harness (smoke today); release pipelines (cibuildwheel / R / C-ABI archive / npm); Sphinx docs + relegate the Python MVP (kept as the dev-oracle for byte goldens + `test_parity`) | 📋 remaining (packaging/docs; not validation) |

## Per-epic

| Epic | Story | Module / artifact | State |
|---|---|---|---|
| 2.0/2.1 | DatasetSpec IR + selectors + validation | `spec/` (`dataset_spec`, `selectors`, `validate`) | ✅ |
| 2.1/5.1 | Versioned JSON Schema (wire contract) | `spec/json_schema.py` + `dataset_spec.schema.json` | ✅ |
| 2.2 | Alias normalizer (verbatim map) + legacy→spec + root-param shorthand | `spec/normalize.py` | ✅ |
| 2.3 | Declarative conventions (FolderParser parity) + 4 profiles | `conventions/` | ✅ |
| 2.4 | Cookbook coverage matrix (introspected, fails on uncovered vocab) | `tests/test_cookbook.py` | ✅ |
| 3.1 | Resolver → InputSet (identity/hash/sidecars/ordering) | `resolve/` | ✅ |
| 3.2/3.3/3.5 | `infer` + scored `DatasetPlan` + column-role inference + `describe` | `infer/` | ✅ |
| 3.4/3.6 | Labeled corpus + per-decision precision + abstention | `tests/test_inference_corpus.py` | ✅ (Brier/ECE deferred) |
| 4.0 | Relational join engine (cardinality/coverage/duplicate/audit) | `materialize/join.py` | ✅ |
| 4.1a/4.2 | Tabular loaders + NA policy + param precedence | `materialize/loaders.py` | ✅ |
| 4.1b | Assembler + lazy `SpectroDataset` adapter | `materialize/{assemble,spectrodataset}.py` | ✅ |
| 5.2 | Parity oracle vs real nirs4all (`pytest -m parity`) | `tests/test_parity.py` | ✅ |
| 5.3 | Re-plug guide | `docs/REPLUG.md` | ✅ |
| 5.4 | Copy-provenance manifest + dual license | `COPY_PROVENANCE.md`, `LICENSE` | ✅ |
| 6.1 | Hardening (adversarial inputs → clear errors) | `tests/test_hardening.py` | ✅ |
| 6.5 | Import-boundary (no nirs4all at import) | `tests/test_import_boundary.py` | ✅ |

## Deferred (documented, not blocking)

See [`ROADMAP.md`](ROADMAP.md) for the consolidated list. Highlights:

- **Phase 2** (Rust core + `dag-ml-data` target) — **unblocked** (2026-05-28); the
  actionable plan is in [`RUST_REWRITE_ROADMAP.md`](RUST_REWRITE_ROADMAP.md) (gate status
  in [`PHASE2_GATE.md`](PHASE2_GATE.md)).
- **SpectroDataset extension** for `observation_id` / `group_id` / `weights` as
  first-class slots — out of scope here (host-side change on `nirs4all`).
- **Inference calibration** (Brier/ECE, story 3.6) — needs a labelled corpus.
- **Stratified percentage split** — intentionally rejected (loader ≠ splitter).

## Codex review trail

Phase-0 rename · Phase-1 foundation (PASS) · Phase-1 build (2 blockers → fixed) ·
final round 1 (vendor-corpus blocker → fixed) · final round 2 (`filename_stem` stem
semantics → fixed, test-verified) · **final round 3 → VERDICT: ACCEPT**.
