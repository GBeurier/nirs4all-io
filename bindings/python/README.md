<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# Python binding (pyo3 / maturin)

A thin pyo3 wrapper over the single `nirs4all-io` Rust core (the facade crate) —
all RESOLVE → INFER → CONFIGURE → MATERIALIZE logic lives in Rust; Python only
adapts the results.

## Layout (mixed maturin)

- Native extension `nirs4all_io._native` (the pyo3 module, built from `src/lib.rs`).
- Python package `nirs4all_io` (under `python/`) that wraps `_native` and adds the
  lazy `SpectroDataset` adapter (`_adapter.py`).

`numpy` and `pandas` are wheel runtime dependencies — they back the full-array
`SpectroDataset` reconstruction and are **not** `nirs4all`.

## Build & install

```bash
maturin develop          # build + install into the active venv (dev)
maturin build            # build an abi3 wheel (abi3-py311)
pip install <wheel>      # install the built wheel
```

Requires Python ≥ 3.11. See `PACKAGING.md` for the abi3 wheel / `nirs4all-formats`
reuse details.

## API

All functions are re-exported from `nirs4all_io`.

| Function | Signature | Returns |
|---|---|---|
| `infer` | `infer(input, conventions=None)` | scored `DatasetPlan` dict (data input only) |
| `to_spec` | `to_spec(input, conventions=None, name=None)` | canonical `DatasetSpec` dict |
| `validate` | `validate(spec)` | `None`; raises `ValueError` if invalid |
| `load` | `load(input, *, target="assembled", conventions=None, name=None, spectro_dataset_cls=None)` | summary dict, or a `SpectroDataset` |
| `to_spectrodataset` | `to_spectrodataset(full, *, spectro_dataset_cls=None)` | a `SpectroDataset` |

**Inputs** (`input`) accept a `str` path, a sequence of `str` (file list), or a
`dict` (a spec). `validate` additionally accepts a JSON string.

**`load` targets:**
- `target="assembled"` → the rounded structural summary dict (no `nirs4all`).
- `target="spectrodataset"` → a real nirs4all `SpectroDataset`, built via a **lazy**
  `nirs4all` import inside the adapter. This is the **only** `nirs4all` touch-point;
  `import nirs4all_io` never imports `nirs4all` (enforced by
  `tests/test_import_boundary.py`). Inject `spectro_dataset_cls=` to drive the
  builder with a double (testing without nirs4all).

## Usage

```python
import nirs4all_io as nio

plan = nio.infer("/data/run")                  # scored DatasetPlan dict
spec = nio.to_spec("/data/run")                # canonical DatasetSpec dict
nio.validate(spec)                             # raises ValueError if invalid

summary = nio.load("/data/run")                # target="assembled" (default)
ds = nio.load("/data/run", target="spectrodataset")   # nirs4all SpectroDataset
```

## Test

```bash
pytest bindings/python/tests
```
