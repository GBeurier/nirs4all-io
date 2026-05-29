<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# R binding (EPIC 11.2)

R package `nirs4allio` over the nirs4all-io C ABI (`libnirs4all_io_capi`, the
`n4io_*` JSON surface). The C glue (`src/n4io.c`) drives each call via `.Call`;
on a non-OK status the context error is raised as an R error and owned result
strings are copied into R and freed with `n4io_string_free`.

```r
library(nirs4allio)

n4io_to_spec('"/data/run"')          # canonical DatasetSpec (JSON string)
n4io_infer('"/data/run"')            # scored DatasetPlan (JSON string)
n4io_validate(specJson)              # errors if invalid; returns invisible(NULL)
n4io_abi_version()                   # C ABI version string
```

## Functions

| Function | Signature | Returns |
|---|---|---|
| `n4io_to_spec` | `n4io_to_spec(input_json, conventions_json = NULL)` | canonical `DatasetSpec` as a JSON string |
| `n4io_infer` | `n4io_infer(input_json, conventions_json = NULL)` | scored `DatasetPlan` as a JSON string |
| `n4io_validate` | `n4io_validate(spec_json)` | `invisible(NULL)`; raises an R error if the spec is invalid |
| `n4io_abi_version` | `n4io_abi_version()` | the C ABI version string |

`conventions_json`, when supplied, is a JSON array of convention names.

## Inputs cross as JSON values

Identical to the C ABI / other bindings: a path is a quoted string
(e.g. `'"/data/run"'`), a file list is a JSON array
(e.g. `'["a.csv","b.csv"]'`), and a spec is a JSON object. Results are canonical
JSON strings; the JSON⇄list layer (e.g. `jsonlite`) is the user's.

## Build & install

```bash
bash bindings/r/build_and_test.sh    # builds the capi, installs the package, runs the smoke test
```

`build_and_test.sh` builds `nirs4all-io-capi` (release), then exports
`N4IO_INCLUDE` (dir with `nirs4all_io.h`) and `N4IO_CAPI_DIR` (dir with
`libnirs4all_io_capi.so`) — both required by `src/Makevars` — and runs
`R CMD INSTALL --no-multiarch` followed by `tests/smoke.R` against the contract
corpus (`N4IO_CORPUS`). To install manually, set those two env vars first:

```bash
export N4IO_INCLUDE=.../crates/nirs4all-io-capi/include
export N4IO_CAPI_DIR=.../target/release
R CMD INSTALL --no-multiarch bindings/r
```

The binding is a thin wrapper: it only marshals JSON strings across the ABI and
runs the single shared Rust core — no NIRS/dataset logic lives here.
