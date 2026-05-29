// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
// Node smoke test for the wasm binding. Build first:
//   wasm-pack build bindings/wasm --target nodejs --out-dir pkg
// then: node bindings/wasm/tests/node_smoke.cjs
const assert = require("node:assert");
const wasm = require("../pkg/nirs4all_io_wasm.js");

// to_spec: normalize a minimal config dict into a canonical DatasetSpec.
const specJson = wasm.to_spec(
  JSON.stringify({ name: "wasm-smoke", sources: [{ id: "x", role: "features", input: "x.csv" }] })
);
assert.ok(specJson.endsWith("\n"), "canonical JSON ends with a newline");
const spec = JSON.parse(specJson);
assert.strictEqual(spec.schema_version, 1, "schema_version is 1");

// validate: the produced spec passes; a bad partition mode throws.
wasm.validate(specJson);
assert.throws(() => wasm.validate(JSON.stringify({ partitions: { by: "random" } })));

assert.match(wasm.version(), /^\d+\.\d+\.\d+/, "version looks like semver");

console.log("wasm node smoke OK");
