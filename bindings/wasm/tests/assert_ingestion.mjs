// Private qualification only. Run against Root's current Node glue, never pkg-node.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, realpathSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
assert.ok(args.length === 2 || args.length === 4,
  'Usage: node assert_ingestion.mjs --binding ABS_BINDING_DIR [--native ABS_NATIVE_FULL_JSON]');
assert.equal(args[0], '--binding');
if (args.length === 4) assert.equal(args[2], '--native');
assert.match(process.versions.node, /^24\.16\./, 'qualified Node profile is 24.16.x');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const fixturePaths = ['case1.json', 'quali-owner-case.json'].map(n => join(here, 'fixtures', n));
const fixtureHashes = fixturePaths.map(p => sha(readFileSync(p)));
const [case1, quali] = fixturePaths.map(p => JSON.parse(readFileSync(p, 'utf8')));
const files = list => list.map(f => ({ name: f.name, bytes: new TextEncoder().encode(f.text) }));

const binding = realpathSync(resolve(args[1]));
for (let dir = binding; ; dir = dirname(dir)) {
  const p = join(dir, 'package.json');
  if (existsSync(p)) assert.notEqual(JSON.parse(readFileSync(p, 'utf8')).type, 'module', p);
  if (dirname(dir) === dir) break;
}
const require = createRequire(import.meta.url);
const glue = realpathSync(require.resolve(join(binding, 'pkg', 'nirs4all_io_wasm.js')));
assert.ok(glue.startsWith(binding + sep), 'glue must resolve inside the explicit private binding');
assert.ok(!glue.split(sep).includes('pkg-node'), 'old pkg-node is forbidden');
const wasm = require(glue);
assert.equal(wasm.version(), '0.3.0');
const wasmPath = join(dirname(glue), 'nirs4all_io_wasm_bg.wasm');

// Decode the owner's numeric wire only. This is not a CSV or dataset parser.
function doubles(hex, count, label) {
  assert.match(hex, /^[0-9a-f]*$/, label);
  assert.equal(hex.length, count * 16, label);
  const bytes = Buffer.from(hex, 'hex');
  return Array.from({ length: count }, (_, i) => bytes.readDoubleLE(i * 8));
}

function matrix(packet, expected, nRows, nCols, typed, label) {
  assert.equal(packet.dtype, 'float64', label);
  assert.equal(packet.n_rows, nRows, label);
  assert.equal(packet.n_cols, nCols, label);
  const mask = expected.map(value => value !== null);
  assert.deepEqual(packet.validity, mask, `${label} explicit mask`);
  const wire = doubles(packet.data_bytes_hex, expected.length, label);
  expected.forEach((value, i) => {
    if (mask[i]) assert.ok(Object.is(wire[i], value), `${label}[${i}] exact wire value`);
    else assert.ok(Object.is(wire[i], +0), `${label}[${i}] canonical masked positive zero`);
  });
  if (typed) {
    assert.ok(packet.data instanceof Float64Array, `${label} logical Float64Array`);
    assert.equal(packet.data.length, expected.length, label);
    expected.forEach((value, i) => {
      if (mask[i]) assert.ok(Object.is(packet.data[i], value), `${label}[${i}] logical value`);
      else assert.ok(Number.isNaN(packet.data[i]), `${label}[${i}] logical NaN, never null/zero`);
    });
  }
  // Deliberately omit only the WASM legacy typed data; never JSON-stringify NaN.
  const { data: logicalData, ...canonical } = packet;
  return canonical;
}

function inspect(full, expected, typed) {
  assert.equal(full.assembled_schema_version, 3);
  assert.equal(full.name, expected.name);
  assert.equal(full.n_sources, 1);
  assert.deepEqual(full.partition_order, expected.partition_order);
  assert.deepEqual(Object.keys(full.blocks).sort(), [...expected.partition_order].sort());
  assert.equal(full.identity.provenance.sample_id, expected.sample_key);
  assert.equal(full.identity.provenance.observation_id, null, 'never fabricate observation identity');
  assert.deepEqual(full.identity.provenance.source_ids, ['x']);
  assert.equal(full.repetition, expected.repetition);
  assert.deepEqual(full.folds, []);
  assert.deepEqual(full.fold_provenance, []);
  const blocks = {};
  for (const part of expected.partition_order) {
    const block = full.blocks[part], want = expected.blocks[part];
    assert.equal(block.n_samples, want.ids.length, part);
    assert.deepEqual(block.source_ids, ['x']);
    assert.deepEqual(block.feature_headers, [expected.headers]);
    assert.deepEqual(block.feature_names, [expected.headers]);
    assert.deepEqual(block.header_units, ['nm']);
    assert.deepEqual(block.representation_ids, ['signal_1d']);
    assert.deepEqual(block.y_headers, expected.targets, 'target order is literal, not alphabetical');
    assert.deepEqual(block.y_categorical, {});
    assert.equal(block.x.length, 1);
    const x = matrix(block.x[0], want.x, want.ids.length, expected.headers.length, typed, `${part}.X`);
    assert.equal(block.feature_axes.length, 1);
    assert.equal(block.feature_axes[0].length, 1);
    const axis = block.feature_axes[0][0];
    assert.equal(axis.name, 'wavelength');
    assert.equal(axis.unit, 'nm');
    assert.equal(axis.coordinates.kind, 'numeric');
    assert.equal(axis.coordinates.count, expected.axis.length);
    assert.deepEqual(doubles(axis.coordinates.data_bytes_hex, expected.axis.length, 'axis'), expected.axis);
    assert.deepEqual(block.y.columns.map(c => c.name), expected.targets);
    const y = block.y.columns.map((column, col) => {
      assert.equal(column.kind, 'numeric');
      const values = want.ids.map((_, row) => want.y[row * expected.targets.length + col]);
      return { name: column.name, kind: column.kind,
        block: matrix(column.block, values, want.ids.length, 1, false, `${part}.${column.name}`) };
    });
    if (typed) {
      // WASM also injects the original logical row-major target matrix at y.data.
      assert.ok(block.y.data instanceof Float64Array);
      assert.equal(block.y.data.length, want.y.length);
      want.y.forEach((value, i) => {
        if (value === null) assert.ok(Number.isNaN(block.y.data[i]), `${part}.y[${i}] logical NaN`);
        else assert.ok(Object.is(block.y.data[i], value), `${part}.y[${i}] logical value`);
      });
    }
    const metadata = Object.fromEntries(block.metadata.columns.map(c => [c.name, c.values]));
    assert.equal(block.metadata.n_rows, want.ids.length);
    assert.deepEqual(Object.keys(metadata).sort(), Object.keys(want.metadata).sort());
    assert.deepEqual(metadata, want.metadata);
    assert.deepEqual(metadata[expected.sample_key], want.ids);
    assert.deepEqual(block.processings, [[]]);
    assert.equal(block.weights, null);
    blocks[part] = { x, y, metadata, source_ids: block.source_ids,
      feature_headers: block.feature_headers, feature_names: block.feature_names,
      feature_axes: block.feature_axes, header_units: block.header_units,
      representation_ids: block.representation_ids, y_headers: block.y_headers };
  }
  return { name: full.name, partition_order: full.partition_order,
    identity: full.identity, repetition: full.repetition, folds: full.folds, blocks };
}

const native = args.length === 4 ? JSON.parse(readFileSync(resolve(args[3]), 'utf8')) : null;
if (native) {
  assert.equal(native.native.version, '0.3.0');
  assert.deepEqual(native.fixture_sha256, fixtureHashes);
}
const rawFiles = files(case1.files);
wasm.validate(JSON.stringify(case1.spec));
const canonical = JSON.parse(wasm.to_spec(JSON.stringify(case1.spec)));
if (native) assert.deepEqual(canonical, native.case1.normalized_spec);
const proposals = wasm.proposeDataset(rawFiles, [], {});
assert.ok(proposals.proposals.some(p => p.kind === 'pairing'
  && p.value.mode === 'join_id' && p.value.on === 'Sample_ID'),
  'literal pairing proposal: join_id on Sample_ID');
const first = inspect(wasm.assembleDataset(rawFiles, [], JSON.stringify(case1.spec)), case1.expected, true);
if (native) assert.deepEqual(first, inspect(native.case1.full, case1.expected, false));

// Supplied user witness only: no ID inference and no change to any original CSV.
assert.deepEqual(quali.explicit_user_confirmation, [
  { target_file: 'Ytrain.csv', data_row: 0, sample_id: 'A' },
  { target_file: 'Ytrain.csv', data_row: 1, sample_id: 'B' },
  { target_file: 'Ytrain.csv', data_row: 2, sample_id: 'C' },
]);
for (const f of [...quali.files, quali.in_memory_witness]) assert.ok(!f.text.endsWith('\n'));
const refusalSpec = structuredClone(quali.spec);
refusalSpec.sources[2] = quali.original_three_file_refusal.target_source_replacement;
const refusalMessage = quali.original_three_file_refusal.message;
assert.throws(() => wasm.assembleDataset(files(quali.files), [], JSON.stringify(refusalSpec)),
  error => error.message === refusalMessage, 'original three files must refuse, never broadcast by position');
if (native) assert.equal(native.case2.original_refusal, refusalMessage);
const q = quali.expected;
const expectedQuali = {
  name: quali.spec.name, partition_order: q.partition_names, sample_key: q.identity_sample_id,
  repetition: q.repetition, headers: q.feature_headers[0], axis: [1000, 1100, 1200], targets: q.y_headers,
  blocks: { train: { ids: q.sample_ids_in_physical_row_order, x: q.x_row_major, y: q.y_row_major,
    metadata: { sample_id: q.sample_ids_in_physical_row_order, site: q.site_in_physical_row_order } } },
};
assert.equal(q.n_observation_rows, 12);
assert.equal(new Set(q.sample_ids_in_physical_row_order).size, 3);
assert.deepEqual(q.x_shape, [12, 3]);
assert.deepEqual(q.y_shape, [12, 1]);
assert.deepEqual(q.x_validity, Array(36).fill(true));
assert.deepEqual(q.y_validity, Array(12).fill(true));
assert.equal(q.identity_observation_id, null);
wasm.validate(JSON.stringify(quali.spec));
const second = inspect(wasm.assembleDataset(files([...quali.files, quali.in_memory_witness]), [],
  JSON.stringify(quali.spec)), expectedQuali, true);
if (native) {
  assert.deepEqual(JSON.parse(wasm.to_spec(JSON.stringify(quali.spec))), native.case2.normalized_spec);
  assert.deepEqual(second, inspect(native.case2.full, expectedQuali, false));
}
console.log(JSON.stringify({ status: 'PASS', node: process.version, version: wasm.version(), glue,
  wasm: wasmPath, wasm_sha256: sha(readFileSync(wasmPath)), fixture_sha256: fixtureHashes,
  cases: ['review-four-sample', 'quali-explicit-witness-and-original-refusal'],
  native: native ? native.native : null,
  comparison: native ? 'literal-and-native-wire-projection' : 'WASM-literals-only',
  source_snapshot_equality: 'Root provenance prerequisite; not inferred from version or equal outputs' }));
console.log('private IO WASM ingestion assertions OK');
