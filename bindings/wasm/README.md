<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# WASM binding (EPIC 11.4)

`wasm-bindgen` binding built with `wasm-pack`, backed directly by
`nirs4all-io-core` (the pure core, **not** the facade). WASM has no filesystem
(D-R7), so this binding exposes only the fs-free JSON surface; path-based
`infer`/`load` need file IO and stay in the native facade.

A thin wrapper: every function just translates strings to/from the single Rust
core, identical to the other bindings.

## Exposed functions

```js
to_spec(spec_json)   // String -> canonical DatasetSpec JSON string
validate(spec_json)  // String -> undefined; throws when the spec is invalid
version()            // () -> crate version string (semver)
```

`to_spec` normalizes a spec/config JSON string into the canonical `DatasetSpec`
JSON. `validate` parses a `DatasetSpec` JSON string and throws on an invalid
spec. Strings cross as canonical JSON, identical to every other binding.

## Build & install

```bash
wasm-pack build bindings/wasm --target nodejs --out-dir pkg
```

This emits `bindings/wasm/pkg/` (the `nirs4all_io_wasm.js` module + the `.wasm`).
Require it from Node:

```js
const wasm = require("./bindings/wasm/pkg/nirs4all_io_wasm.js");
```

## Usage

```js
const wasm = require("./pkg/nirs4all_io_wasm.js");

const specJson = wasm.to_spec(JSON.stringify({
  name: "wasm-smoke",
  sources: [{ id: "x", role: "features", input: "x.csv" }],
}));

wasm.validate(specJson);   // ok; throws on an invalid spec
console.log(wasm.version());
```

## Test

```bash
wasm-pack build bindings/wasm --target nodejs --out-dir pkg
node bindings/wasm/tests/node_smoke.cjs
```
