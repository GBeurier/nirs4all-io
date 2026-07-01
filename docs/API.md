# Public API & integration seam

```{note}
This page describes the Phase-1 Python MVP surface (the parity oracle). For the
**published PyPI wheel** (the pyo3 binding) — including its actual `load`
default, `to_spec` return type and `nio.validate` — see
[Getting started](getting_started.md). The `DatasetSpec` / `DatasetPlan` JSON
shapes below are identical across both; some Python accessor spellings differ.
```

> Story 5.1. This is the **stable seam** `nirs4all` / `nirs4all-studio` (or any
> host) can adopt later without depending on internals. The contract is the
> `DatasetSpec` / `DatasetPlan` JSON shape + these four entry points.

## Entry points

```python
import nirs4all_io as nio

# 1. infer(input) -> DatasetPlan : inspect anything, get a scored recommendation
plan = nio.infer("data/mango/", conventions=["nirs4all-classic"])
print(plan.recommendations, plan.warnings)
plan.to_dict()                       # JSON-serializable (scores are uncalibrated; see C5)

# 2. load(input | spec | plan) -> target : materialize
ds  = nio.load(plan.accept())                          # DatasetPlan -> SpectroDataset
ds  = nio.load("data/mango/")                          # directory + conventions
ds  = nio.load({"sources": [...]})                     # explicit DatasetSpec (dict)
ds  = nio.load("dataset.yaml")                          # JSON/YAML config (alias-normalized)
ds  = nio.load((X, y))                                  # in-memory arrays
ds  = nio.load(reference_dataset)                       # any object with to_io_spec(), e.g. NirsDataset
asm = nio.load(spec, target="assembled")               # target-agnostic AssembledDataset

# 3. to_spec(input) -> (DatasetSpec, base_dir) : just resolve, no materialization
spec, base = nio.to_spec("data/mango/")

# 4. DatasetSpec : the canonical IR
spec = nio.DatasetSpec.from_yaml("dataset.yaml"); spec.validate()
spec.to_dict(); spec.to_yaml(); spec.to_json()
```

### `load` signature

```python
nio.load(inp, *, target="spectrodataset" | "assembled",
         conventions=None, base_dir=None, name=None, spectro_dataset_cls=None)
```

- `target="spectrodataset"` (default) lazily imports `nirs4all.data.SpectroDataset`.
  Pass `spectro_dataset_cls=` to inject a double (used in tests; no nirs4all needed).
- `target="assembled"` returns the **target-agnostic** `AssembledDataset` (per-partition
  `PartitionBlock`: multi-source `X`, `y`, `metadata`, headers, units) — testable, and the
  hand-off point for the future `to_dag_ml_data` adapter (Phase 2).
- `target="dataset_package"` returns the public target-agnostic `DatasetPackage`
  summary/manifest wrapper.
- `target="dag-ml-data"` raises `NotImplementedError` (Phase 2, gated — see `PHASE2_GATE.md`).

### Reference Dataset Adapter

Catalog/reference dataset objects can be handed to IO without creating a package
dependency cycle by exposing:

```python
def to_io_spec(self) -> dict | tuple[dict, pathlib.Path]: ...
```

`nirs4all-datasets.NirsDataset` uses this seam: it publishes a normal
`DatasetSpec` over its verified local canonical files, then IO owns the usual
load/join/package materialization. Parsing still belongs to `nirs4all-formats`
through IO's vendor loader; datasets only supplies catalog paths and roles.

## Invariants the seam guarantees

- **No runtime `nirs4all` dependency.** `import nirs4all_io` never imports `nirs4all`
  (enforced by `tests/test_import_boundary.py`); only `target="spectrodataset"` lazily does.
- **`DatasetSpec` round-trips** through dict/YAML/JSON and is structurally validated.
- **Parity**: for the supported topologies, `load(...) → SpectroDataset` equals
  `nirs4all.DatasetConfigs(...)` (`tests/test_parity.py`, run with `pytest -m parity`).

## How a host would adopt it (illustrative — not wired here)

```python
# in a host (nirs4all / studio), later, with no change to nirs4all-io:
import nirs4all_io as nio
def load_dataset(user_input):
    plan = nio.infer(user_input)          # show plan.recommendations in the UI
    return nio.load(plan.accept(overrides))  # user-edited spec -> SpectroDataset
```

See `REPLUG.md` for the recommended adoption sequence.
