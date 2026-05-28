# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`nirs4all-io` is the **dataset-assembly bridge** of the nirs4all ecosystem. It turns *any*
user input — a directory, a file list, a glob, a config dict/JSON/YAML, in-memory arrays, or a
folder of vendor spectra + a reference table — into a pipeline-ready `SpectroDataset`. It owns the
dataset-level concepts that the low-level reader library (`nirs4all-formats`) deliberately does not:
X/Y/metadata **roles**, train/test/folds **partitions**, **multi-source**, relational **joins**,
signal/task-type **inference**, and a declarative **convention** system.

```
any input ──► RESOLVE ──► INFER ──► CONFIGURE ──► MATERIALIZE ──► SpectroDataset
              (InputSet)  (DatasetPlan, scored)   (DatasetSpec)    (or target-agnostic AssembledDataset)
```

Phase 1 (Python MVP) is complete and parity-verified. Phase 2 (Rust core + a `dag-ml-data` target)
is externally gated — see `docs/PHASE2_GATE.md`. `src/` layout (`src/nirs4all_io/`), Python ≥3.11,
dual-licensed `CeCILL-2.1 OR AGPL-3.0-or-later`.

## Commands

The package is installed editable; it is **not** importable until you do so, and there is no `.venv`
in this repo. Use the ecosystem venv if one exists, else install into the active interpreter:

```bash
pip install -e ".[dev]"      # ruff, mypy, pytest + pyarrow/openpyxl/scipy + nirs4all & nirs4all-formats (dev oracles)

# Green gate (run all three before reporting work complete)
ruff check .                 # lint: E,F,I,W,UP,B; line-length 220; E501 ignored
mypy .                       # type check (py311, ignore_missing_imports)
pytest                       # all tests; parity tests auto-skip if nirs4all is absent

# Targeted test runs
pytest tests/test_spec.py                      # one file
pytest tests/test_cookbook.py::test_coverage_matrix_complete   # one test
pytest -m parity             # ONLY the parity-oracle tests (needs nirs4all installed)
pytest -m "not parity"       # everything except the parity oracle
```

Two registered pytest markers (see `pyproject.toml`): **`parity`** (imports `nirs4all` read-only as
an oracle) and **`formats`** (needs the `nirs4all-formats` reader library).

## Architecture: the four stages

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
(`materialize/spectrodataset.py`) adapts it to nirs4all today; `to_dag_ml_data(assembled)` slots in
beside it when Phase 2 unblocks. Keep assembly logic target-agnostic so it stays testable without
nirs4all — `target="assembled"` is the test seam.

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

6. **Phase 2 is gated.** `load(..., target="dag-ml-data")` raises `NotImplementedError` on purpose.
   Do not implement the dag-ml-data target until `docs/PHASE2_GATE.md` flips green.

## Public API (`api.py`, re-exported from `__init__.py`)

```python
import nirs4all_io as nio
plan = nio.infer(<input>, conventions=[...])     # -> scored DatasetPlan (plan.resolved_spec, .recommendations, .warnings)
ds   = nio.load(<input | spec | plan>, target="spectrodataset" | "assembled", base_dir=, name=)
spec, base = nio.to_spec(<input>)                # resolve only, no materialization
desc = nio.describe(<file>)                      # neutral per-file descriptor (delimiter, header unit, axis)
# IR types: nio.DatasetSpec (.from_dict/.from_yaml/.to_dict/.validate), nio.DatasetPlan, nio.AssembledDataset
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

- **There is no CLI.** `pyproject.toml` declares a `nirs4all-io = "nirs4all_io.cli:main"`
  console-script, but `cli.py` does not exist — the entry point is dead. The API is Python-only.
- **YAML round-trips lists, not tuples** — specs are dict/JSON/YAML-authorable and stay `str`-enum
  clean precisely so they survive serialization; don't put tuples in a spec.
- The repo is a clean tree: **no dead/deprecated code, no backward-compat shims.** Remove rather than
  deprecate.

## Where to look it up

- `docs/DATASET_CONFIGURATIONS.md` — the **complete reference**: every input form, `DatasetSpec`
  field, selector, merge mode, join, partition, fold, loading param, supported/out-of-scope layout,
  and a use-case cookbook with honest ✅/🟡/📋 status per option. Read this before adding spec vocab.
- `docs/API.md` — the stable integration seam and how a host adopts it.
- `docs/STATUS.md` / `docs/ROADMAP.md` — per-epic state and the intentionally-deferred list.
- `docs/REPLUG.md` — host-adoption sequence and the io-vs-host ownership split.
- `COPY_PROVENANCE.md` — what was copied from `nirs4all` and how (license-compatibility record).
- `../nirs4all/CLAUDE.md` — ecosystem map and the cross-repo boundary rules.
