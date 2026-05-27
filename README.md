# nirs4all-io

> **Dataset-assembly bridge.** Turn *any* user input — a directory, a list of
> files, a glob, a config dict/JSON/YAML, in-memory arrays, a folder of vendor
> spectra + a reference table — into a pipeline-ready dataset.

`nirs4all-io` owns the dataset-level concepts that the low-level reader library
[`nirs4all-formats`](https://github.com/GBeurier/nirs4all-formats) deliberately
does **not**: X/Y/metadata roles, train/test/folds, multi-source, relational
joins, signal/task-type inference, and a declarative convention system. It
matches the expressiveness of `nirs4all`'s `DatasetConfig`/`DatasetLoader` and
adds a score-based inference engine.

```
any input ──► RESOLVE ──► INFER ──► CONFIGURE ──► MATERIALIZE ──► SpectroDataset
              (InputSet)  (DatasetPlan, scored)   (DatasetSpec)    (and later: dag-ml-data)
```

## Status

**Phase 1 (Python MVP) — complete and parity-verified.** `load()` and `infer()`
work end-to-end and target `SpectroDataset`; the build is **byte-equivalent to
nirs4all's own `DatasetConfigs`** on the supported topologies (`pytest -m parity`).
178 tests, ruff + mypy clean. See [`docs/STATUS.md`](docs/STATUS.md) for the
per-epic breakdown, [`docs/API.md`](docs/API.md) for the seam, and
[`docs/PHASE2_GATE.md`](docs/PHASE2_GATE.md) for why the Rust / `dag-ml-data`
target (Phase 2) stays gated. Full design:
[`../nirs4all-formats/docs/REDESIGN_FORMATS_AND_IO.md`](../nirs4all-formats/docs/REDESIGN_FORMATS_AND_IO.md).

## Quick start (target API)

```python
import nirs4all_io as nio

# Inspect a directory and get a scored recommendation
plan = nio.infer("data/mango/", conventions=["nirs4all-classic"])
print(plan.recommendations)

# Materialize a spec/plan/input into a SpectroDataset
ds = nio.load(plan, target="spectrodataset")
ds = nio.load({"sources": [{"id": "x", "role": "features", "input": "X.csv"}]})

# Vendor corpus + reference table (headline new capability)
plan = nio.infer(["spectra/*.0", "reference.csv"], conventions=["vendor-corpus"])
```

## What can it load?

[`docs/DATASET_CONFIGURATIONS.md`](docs/DATASET_CONFIGURATIONS.md) is the complete
reference: **every** input form, `DatasetSpec` field, column selector, merge mode,
relational join, partition, fold and loading parameter — with a use-case cookbook
and an honest ✅/🟡/📋 implementation status on each option.

## Design principles

- **Self-contained**: no runtime dependency on `nirs4all`. The only touch-point
  is a lazy import of the `SpectroDataset` class at materialization.
- **Parsers live in `nirs4all-formats`**: vendor byte-decoding is never
  reimplemented here; tabular loading logic is copied from `nirs4all`
  (see [`COPY_PROVENANCE.md`](COPY_PROVENANCE.md)).
- **Versioned, machine-validatable `DatasetSpec`** is the canonical contract.

## License

Dual-licensed `CeCILL-2.1 OR AGPL-3.0-or-later` — see [`LICENSE`](LICENSE).
