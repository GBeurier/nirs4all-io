<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# nirs4all-io binding compatibility

Every binding is a **thin wrapper over the single Rust core** (`nirs4all-io-core`
+ the `nirs4all-io` facade): no language re-implements resolve/infer/configure/
materialize logic. Per **D-R7**, only the **JSON surface** crosses the C ABI in
v0 — no materialized arrays. The pyo3 binding additionally links the facade
directly, so it (and only it) can hand back native Python objects and a real
`SpectroDataset`.

## Operation matrix

| Operation | Rust CLI | Python (pyo3) | R | MATLAB / Octave | WASM / JS |
|---|---|---|---|---|---|
| `infer` | ✅ `infer` | ✅ `infer` | ✅ `n4io_infer` | ✅ `nirs4all_io.infer` | ❌ (needs fs) |
| `to_spec` | ✅ `to-spec` | ✅ `to_spec` | ✅ `n4io_to_spec` | ✅ `nirs4all_io.to_spec` | 🟡 `to_spec` (spec-dict only, fs-free) |
| `validate` | ✅ `validate` | ✅ `validate` | ✅ `n4io_validate` | ✅ `nirs4all_io.validate` | ✅ `validate` |
| `load` → assembled | ✅ `load` | ✅ `load(target="assembled")` / `load_summary` | ❌ | ❌ | ❌ |
| `load` → SpectroDataset | ❌ | ✅ `load(target="spectrodataset")` (lazy adapter) | ❌ | ❌ | ❌ |
| `emit-dag-ml-data` | 🟡 stub → ecosystem crate | ❌ | ❌ | ❌ | ❌ |
| ABI / version | — | `__version__` | `n4io_abi_version` | `nirs4all_io.abi_version` | `version` |

Legend: ✅ supported · 🟡 partial / out-of-process · ❌ out of scope in v0.

### Data form accepted / returned

| Surface | Input form | Output form |
|---|---|---|
| Rust CLI | path / glob args (`--spec FILE` for a spec/config JSON); `--convention`, `--name` | canonical JSON to stdout (`validate` prints `valid: …` to stderr) |
| Python (pyo3) | native: `str` path, `list[str]` file list, or `dict` spec | native Python objects (dicts); `target="spectrodataset"` → nirs4all `SpectroDataset` |
| R | JSON strings (`'"/data/run"'`, `'["a.csv","b.csv"]'`, spec object) | canonical JSON string |
| MATLAB / Octave | JSON strings (quoted path / JSON array / JSON object) | canonical JSON string |
| WASM / JS | JSON string (spec/config dict only) | canonical JSON string |

Notes:
- **`emit-dag-ml-data`** is not built into the in-tree CLI: the subcommand exists
  but bails with a pointer to the workspace-excluded ecosystem crate
  `crates/nirs4all-io-dagml` (`emit-dagml` binary), which carries the
  `dag-ml-data` sibling dependency. No binding exposes it in v0.
- **R / MATLAB / Octave** are C-ABI-first and v0-scoped to the JSON surface
  (`infer` / `to_spec` / `validate` + the version probe); they have no array
  `load`.
- **WASM/JS** is fs-free: `to_spec` here only **normalizes a spec/config dict**
  (it cannot resolve paths), and there is no `infer`/`load`.
- **SpectroDataset** materialization is **pyo3-only**, via the lazy `_adapter`
  that imports `nirs4all` at call time — importing `nirs4all_io` never imports
  `nirs4all`.

## Exact entry points (verified against source)

- **CLI** (`nirs4all-io` binary, `crates/nirs4all-io-cli/src/main.rs`):
  `infer`, `to-spec`, `validate`, `load`, `emit-dag-ml-data`.
- **Python** (`bindings/python`): native module `nirs4all_io._native` exports
  `infer(input, conventions=None)`, `to_spec(input, conventions=None, name=None)`,
  `validate(spec)`, `load_summary(input, conventions=None, name=None)`,
  `assembled_full(input, conventions=None, name=None)`, `__version__`. The
  package adds `load(input, *, target="assembled", conventions=None, name=None,
  spectro_dataset_cls=None)` and `to_spectrodataset(full, *,
  spectro_dataset_cls=None)`.
- **R** (`bindings/r/R/n4io.R`): `n4io_to_spec(input_json, conventions_json=NULL)`,
  `n4io_infer(input_json, conventions_json=NULL)`, `n4io_validate(spec_json)`,
  `n4io_abi_version()`.
- **MATLAB / Octave** (`bindings/matlab/+nirs4all_io`): `to_spec(input_json,
  conventions_json)`, `infer(input_json, conventions_json)`,
  `validate(spec_json)`, `abi_version()` — one `n4io` MEX dispatches on a command
  string.
- **WASM / JS** (`bindings/wasm/src/lib.rs`): `to_spec(spec_json)`,
  `validate(spec_json)`, `version()`.

## Build & install

```bash
# Rust CLI (from the io repo root)
cargo build -p nirs4all-io-cli --release        # → target/release/nirs4all-io

# Python wheel (pyo3 / maturin; import name nirs4all_io)
maturin build -m bindings/python/pyproject.toml # abi3-py311 wheel
# or, for development:
pip install maturin && maturin develop -m bindings/python/pyproject.toml

# R (builds the capi cdylib, installs, runs smoke)
bash bindings/r/build_and_test.sh

# MATLAB / Octave (builds capi + MEX, runs smoke under Octave)
bash bindings/matlab/build_and_test.sh

# WASM (excluded from the cargo workspace; built with wasm-pack)
wasm-pack build bindings/wasm --target nodejs --out-dir pkg
node bindings/wasm/tests/node_smoke.cjs

# emit-dag-ml-data (ecosystem crate, separate dag-ml-data dependency)
cargo run --manifest-path crates/nirs4all-io-dagml/Cargo.toml --bin emit-dagml -- <input>
```

## Usage example

```bash
# CLI — canonical DatasetSpec (sorted keys, 2-space indent, trailing newline)
nirs4all-io to-spec /data/run
nirs4all-io infer /data/run -c nirs4all-classic
nirs4all-io load /data/run --name my_dataset       # assembled summary JSON
```

```python
# Python — native objects, plus the SpectroDataset target
import nirs4all_io as nio
spec = nio.to_spec("/data/run")          # dict
plan = nio.infer("/data/run")            # scored DatasetPlan dict
ds   = nio.load("/data/run", target="spectrodataset")  # nirs4all SpectroDataset
```

```r
# R — JSON in, canonical JSON out
n4io_to_spec('"/data/run"')
n4io_infer('["a.csv","b.csv"]')
```

```js
// WASM/JS — fs-free spec normalize + validate
import { to_spec, validate } from "nirs4all-io-wasm";
const spec = to_spec(JSON.stringify({ name: "d", sources: [{ id: "x", role: "features", input: "x.csv" }] }));
validate(spec);
```

Across every binding the JSON is the **same canonical form** (UTF-8, lexically
sorted keys, two-space indent, one trailing newline) — see `bindings/SPEC.md`.
