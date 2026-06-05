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

// inferFiles: browser/no-fs inference from named Uint8Array files.
const plan = wasm.inferFiles(
  [
    {
      name: "combined.csv",
      bytes: new TextEncoder().encode(
        "sample_id;1000;1005;protein;batch\ns1;0.10;0.20;11.2;A\ns2;0.30;0.40;12.5;B\n"
      ),
    },
  ],
  {}
);
assert.strictEqual(plan.structure.value, "single_combined");
assert.ok(plan.resolved_spec.sources.length >= 1, "resolved spec has sources");
assert.ok(plan.columns[0].column_roles.length >= 1, "column roles were inferred");

// inferDataset: browser orchestration is implemented in Rust, not in the page.
const browserPlan = wasm.inferDataset(
  [{ name: "scan.asd", bytes: new Uint8Array([0, 1, 2, 3]) }],
  [{
    source: "scan.asd",
    format: "asd-fieldspec",
    records: [{
      signals: {
        absorbance: {
          values: [0.1, 0.2, 0.3],
          axis: { values: [1000, 1005, 1010], unit: "nm" },
        },
      },
      targets: { protein: 12.4 },
      metadata: { sample_id: "s1" },
    }],
  }],
  {}
);
assert.strictEqual(browserPlan.input.selected_inference, "decoded_records");
assert.strictEqual(browserPlan.resolved_spec.sources[0].role, "mixed");

console.log("wasm node smoke OK");
