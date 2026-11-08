<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# nirs4all-io binding specification (normative)

This is the binding contract for the Rust rewrite of `nirs4all-io`. It is adapted
from `nirs4all-methods/bindings/SPEC.md` (the C-ABI discipline) and
`nirs4all-formats` (the pyo3/maturin model), specialized to io's surface. PRs that
violate a **MUST** here are rejected.

## 1. Scope and layers

io exposes one engine through several language bindings. Two layers:

- **Raw layer** — the stable C ABI (`n4io_` prefix) + the pyo3 native module.
  Hand-written FFI plumbing; conformance-tested. Generated header is committed.
- **Idiomatic layer** — per-language ergonomics (pandas/`data.frame`/typed
  classes). Hand-written, shared, conformance-tested. No per-call codegen.

The v0 surface that crosses the C ABI is **strings (JSON)**: `infer`, `to_spec`,
`validate`, and the bytes-free assembled structural summary. The `dag-ml-data` emit lives in the Rust bridge crate
`nirs4all-io-dagml` (`emit-dagml`), outside the C ABI bindings in v0. **No
materialized arrays cross the C ABI in v0** (D-R7).

## 2. Canonical-JSON contract (the wire format)

Every JSON that crosses any binding boundary, and every golden, MUST be the
canonical form defined once in `nirs4all-io-core/src/canonical_json.rs` and
mirrored in `nirs4all_io/canonical_json.py`:

- UTF-8, no BOM; non-ASCII emitted verbatim (NOT `\uXXXX`-escaped).
- Object keys sorted lexicographically by Unicode code point.
- Two-space pretty indent, `": "` after keys, one element per line.
- `\n` line endings; exactly one trailing `\n`.
- Finite numbers only (NaN/Inf rejected on the Python side, mapped to `null`
  by serde on the Rust side). The `DatasetSpec` IR never emits non-finite
  numbers. Assembled data does: the rounded `load_summary` JSON writes a missing
  value as `null`, which hosts must read back as NaN/NA rather than drop or coerce
  to 0. The exact `assembled_full` wire instead carries numeric bytes and an
  explicit `validity` mask: a false cell stores `+0`, and hosts restore NaN/NA
  from the mask. The WASM `assembleDataset` keeps its typed-array view: matrices
  and weights are `Float64Array`s holding NaN.

The workspace enables serde_json's `preserve_order` so *input* object order
mirrors Python dict insertion order (load-bearing for legacy-config
normalization); `canonical_json` therefore sorts object keys EXPLICITLY rather
than relying on the map type. Parity is gated by `tests/test_canonical_json.py`
and `crates/nirs4all-io-core/tests/canonical_parity.rs` against the same blessed
corpus.

## 3. Status and error model (C ABI)

- Every fallible C ABI call returns an `n4io_status_t` (a `#[repr(C)]` enum;
  `N4IO_OK == 0`). Callers MUST check it.
- On a non-OK status, the human-readable message (for a `SpecError`, the
  CPython-exact diagnostic) lives in a **per-context error buffer**. Bindings copy
  it out **before** the next call on the same context. A structured JSON
  diagnostic (message + JSON-pointer path) is deferred to v0.1.
- `n4io_context_last_error(ctx)` returns a context-owned `const char*` (never
  NULL; empty string when no error). Bindings MUST NOT free it.

## 4. Memory ownership (never free across the boundary)

Ownership is symmetric: the caller frees what it allocated; the core frees what
it allocated. The only core-allocated outputs a caller receives ownership of are
strings returned by value (e.g. `n4io_to_spec`) and opaque handles. Each such
string MUST be released with `n4io_string_free`; each handle with its documented
`n4io_*_free`/`_destroy`. Never free a core pointer with the host allocator.

## 5. Opaque handles

Handles are forward-declared opaque structs. `_create` returns `N4IO_OK` and sets
`*out`; on any failure `*out` is set to NULL and a non-OK status is returned.
`_destroy` is void and no-ops on NULL. v0 exposes one handle, `n4io_context_t`
(carries the error buffer). Result handles (spec/plan) are deferred to v0.1 — v0
returns canonical JSON by value instead.

## 6. ABI versioning

- `n4io_abi_version()` returns the ABI version string (owned; free with
  `n4io_string_free`). The ABI version is **independent of crate semver** (D-R6).
- Bindings MUST call `n4io_check_abi_compatibility(header_major, header_minor)`
  before any other call and fail loudly on skew: MAJOR must match; the library
  MINOR must be `>=` the header MINOR (forward-compatible additive changes).

## 7. Symbol governance

ABI **0.3.0** adds `n4io_load_summary_with_limits`; old signatures and canonical
outputs are unchanged. NULL limits select finite defaults; a separate JSON
object overrides individual positive integer fields, or the explicit JSON
string `"unlimited"` opts out for trusted hosts. Unknown/invalid options return
`N4IO_ERR_INVALID_ARGUMENT`; resource-policy failures return `N4IO_ERR_SPEC`.
The output pointer MUST be NULL on either failure. Policies MUST NOT be sourced
from DatasetSpec or silently ignored by bindings. Python native, R, MATLAB and
CLI forward the same policy to the Rust facade; the Python oracle and WASM
filesystem-free calls do not expose this native file-read policy. See
[API resource budgets](../docs/API.md#native-resource-budgets-published-wheel-and-rust-facade)
for defaults, coverage and memory-accounting limitations. ABI versioning does not
change package versions or authorize a release.

- The cbindgen header (`crates/nirs4all-io-capi/include/nirs4all_io.h`) is
  committed and regenerated by the capi `build.rs`.
- `crates/nirs4all-io-capi/abi/expected_symbols_{linux,macos,windows}.txt`
  snapshots the exported symbol set; the ABI-check CI
  (`.github/workflows/abi-check.yml`) diffs actual vs expected and **fails on
  drift**. Every exported symbol MUST start with `n4io_`. Rust supplies the cdylib
  export list; the capi `build.rs` sets only the Linux SONAME and macOS install
  name and wires no named GNU ld version script. A CI grep rejects other exported
  names on Linux and macOS. The Windows leg diffs only the `n4io_` names it
  extracts from `dumpbin`, so it does not detect a foreign export. A `cargo test` drift guard
  (`tests/abi_surface.rs`) cross-checks source ↔ snapshot ↔ header.
- A **forbidden-runtime-dependency audit** (CI `ldd`) ensures the cdylib does not
  link Python/R/BLAS/etc.
- **Windows**: Rust auto-exports `#[no_mangle]` symbols (no `.def` needed); a
  dumpbin-based MSVC CI leg diffs the surface.

## 8. Per-language FFI policy (D-R3 — NOT one-size-fits-all)

- **Python = pyo3-native** (formats model). Links the facade crate directly so it
  can return numpy/pandas/sklearn objects and a lazy `SpectroDataset` in-process.
  The lazy `SpectroDataset` import is the **only** nirs4all touch-point and MUST
  stay lazy (import-boundary test).
- **R / MATLAB / Octave / C = C-ABI-first** (methods model). v0 scope: the C-ABI
  JSON surface (`infer`/`to_spec`/`validate` plus the assembled structural
  summary); `dag-ml-data` emit remains in the Rust bridge CLI, not these
  bindings. No array `load()` for these hosts in v0.
- **WASM/JS = wasm-bindgen-native**: `infer`/`plan` over bytes/JSON, no fs.

### WASM independent table inspection

Raw `describeTables(files, options?) -> TableDescription[]` accepts the same
`{name, bytes: Uint8Array}[]` as `inferFiles`. Options are
`{params?: {[fileName: string]: LoadingParams}, distinct?: [fileName, columnName][]}`.
Omitted/null options mean empty options. Malformed arguments throw `JsError`;
each file's parameter/reader failure instead returns `readable: false`, a verbatim
owner `refusal`, null `n_rows`/`n_cols`, empty `columns`/`distinct` and null `identity`.
Unresolved delimiter evidence is `{delimiter_unresolved: {candidates: string[]}}`;
no table read occurs until a nonempty delimiter is declared.

The core `infer::memory::describe_tables(&[NamedBytes], &DescribeOptions)` uses
`materialize::loaders::read_table_bytes` over the original bytes with NA forced to
`Ignore`; it never requires joins or assembly and adds no row/column ceiling.
Descriptions retain effective `params`, exact reader shape/column names,
`columns[{name,dtype,numeric_kind,nunique_with_na,is_unique,preview,role}]`,
`identity`, requested `distinct` strings in first-appearance order and `warnings`.
`preview` is the first nonmissing owner string or null. Guesses are uncalibrated.
`distinct` includes the owner's missing-value strings; it is not typed partition
conversion. Missing requested columns warn without a fabricated value list.
The optional `ignored_input_na_report` is copied verbatim from the existing helper
and omitted when it returns `None`; it supplies input counts before roles/joins,
never per-column NA counts or positions. Inspection performs no fill/removal or
statistic and must not replace the user's chosen materialization NA policy.

Results are plain JS objects, not canonical JSON text. Detailed DTO declarations
live in `wasm/types/nirs4all-io.d.ts`; `wasm/idiomatic.d.ts` re-exports those DTO
types only. This source addition does not claim an idiomatic runtime export,
Python-oracle parity, a C ABI addition or executed binding qualification.

## 9. Conformance gates

- Per-crate: `cargo fmt --check`, `cargo clippy --workspace --all-targets -D
  warnings`, `cargo test --workspace`, `--no-default-features` build.
- Contract: Python ≡ Rust ≡ each binding, byte-identical `to_spec`/`infer` JSON.
- IO-XLG-001 cross-binding qualification: Rust CLI, Python wheel, WASM module,
  R package, MATLAB/Octave MEX, and direct C ABI materialize one frozen
  identity-rich fixture to byte-identical assembled-summary-v3 JSON. A runtime
  absence is recorded as `unavailable`, a build/contract failure as `refused`,
  and either disposition prevents closure. This bounded summary gate does not
  qualify full v3 bytes, masks or precision, and C ABI 0.3 does not transport
  materialized arrays.
- ABI: symbol-snapshot diff + Linux/macOS export-prefix grep + forbidden-dep audit + the
  compatibility probe on load; an MSVC/Windows leg.
- Import-boundary: importing the Python binding MUST NOT import `nirs4all`.

## Explicit inference loading choices (native V1)

Rust inference takes `InferOptions { conventions, params, file_params }`.
WASM inferFiles/inferDataset consume that object; proposeDataset adds confirmed
locks. Native Python exposes infer(input, conventions=None, params=None,
file_params=None); CLI infer exposes --params / --file-params JSON objects.
The existing C n4io_infer second JSON pointer is an options object or NULL,
with unchanged symbol/calling convention and no separately versioned options
schema. Loading choices apply before the first table read and owner inference.
Exact file names override global parameters. Invalid/unknown/conflicting inputs
refuse. Default ambiguous headers remain true and expose header_ambiguous through
the plan's existing per-file params/warning carrier and TableDescription's
additive boolean. Explicit declarations suppress ambiguity. The frozen Python
oracle and R's vendored owner copy remain unchanged; no parity for these new
oracle-absent keyword options is claimed. See COMPAT.md for the mixed-header V1
exception and operation-specific qualification.

