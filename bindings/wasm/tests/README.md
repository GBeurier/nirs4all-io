# IO WASM owner-contract checks

Build the current Node-target binding into `../pkg`, then run the existing
`node_smoke.cjs` and `idiomatic_smoke.mjs` from this directory.

`assert_ingestion.mjs` adds exact CSV ingestion contracts with shared literal
fixtures: reordered target rows, train/validation/test membership, ordered
targets, numeric bytes and validity masks, axes/units and sample identities.
It also covers the exact synthetic Quali repeated-sample smoke inputs and an
explicit user-confirmed target identity witness. Without that witness, both
engines must refuse the unkeyed three-target/twelve-observation join. No fitting
or inferred physical scan identity is part of this gate.

For the named Node24.16/IO0.3 profile, run from the repository root:

```sh
node bindings/wasm/tests/assert_ingestion.mjs --binding "$IO_NODE_BINDING"
"$IO_INSTALLED_PYTHON" -I -B bindings/wasm/tests/dump_native.py --out-dir "$NEW_FIXTURE_OUTPUT"
node bindings/wasm/tests/assert_ingestion.mjs --binding "$IO_NODE_BINDING" --native "$NEW_FIXTURE_OUTPUT/native-full.json"
```

`IO_NODE_BINDING` is the binding directory containing `pkg/nirs4all_io_wasm.js`.
`IO_INSTALLED_PYTHON` must use the installed native IO0.3 wheel, not the Python
parity oracle; its site-packages module path and digest are recorded by the
dumper. `NEW_FIXTURE_OUTPUT` must not exist. Choose either the literal-only Node
command or the installed-native dump plus comparison, depending on the intended
qualification. No dependency installation, network or automatic build occurs.

The comparison establishes behavior of the named artifacts. It does not prove
identical compiler source snapshots from matching versions or equal results.
Keep provenance limits explicit and distinguish the masked canonical positive
zero bytes from logical typed-array NaNs. The fixtures retain their source and
explicit-confirmation provenance; they are synthetic test data, not measured
material or a general identity/decoder certificate.
