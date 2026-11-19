// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { decodeAssembledFull } from '../assembled-wire.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const native = JSON.parse(readFileSync(join(here, 'fixtures/assembled-full-native.json'), 'utf8'));
assert.equal(native.schema, 'nirs4all.io.assembled-wire-native-sample.v1');

// Expected logical values are literals (from the CSV text and IEEE-754 tables),
// never produced by the decoder. NaN marks a masked cell.
function assertLogical(actual, expected, label) {
  assert.ok(actual instanceof Float64Array, `${label} is a Float64Array`);
  assert.equal(actual.length, expected.length, label);
  expected.forEach((value, index) => {
    assert.ok(Object.is(actual[index], value), `${label}[${index}]: ${actual[index]} !== ${value}`);
  });
}

const F64 = { dtype: 'float64', origin: 'default', source_dtype: null, requested_dtype: null, consumer: null };
const LABEL = { ...F64, dtype: 'string-label' };
const packet = (dtype, nRows, nCols, hex, validity, precision = { ...F64, dtype }) =>
  ({ dtype, n_rows: nRows, n_cols: nCols, data_bytes_hex: hex, validity, precision });

function syntheticFull() {
  return {
    assembled_schema_version: 3,
    name: 'synthetic',
    blocks: {
      train: {
        n_samples: 2,
        source_ids: ['s'],
        // float64 [5e-324 (subnormal), 1.0] / [masked, -2.5]
        x: [packet('float64', 2, 2,
          '0100000000000000000000000000f03f000000000000000000000000000004c0',
          [true, true, false, true])],
        feature_axes: [[{ name: 'wavelength', unit: 'nm',
          coordinates: { kind: 'numeric', count: 2, data_bytes_hex: '0000000000408f400000000000309140' } }]],
        y: { columns: [
          { name: 'grade', kind: 'classification_integer',
            block: packet('float64', 2, 1, '00000000000000400000000000000000', [true, false]) },
          // labels "b" (valid) and "" (masked)
          { name: 'site', kind: 'classification_string',
            block: { dtype: 'string-label', n_rows: 2, n_cols: 1,
              data_bytes_hex: '0100000000000000620000000000000000', validity: [true, false], precision: LABEL } },
        ] },
        y_headers: ['grade', 'site'],
        y_categorical: { site: { categories: ['a', 'b'] } },
        metadata: { n_rows: 2, columns: [{ name: 'id', values: ['r1', 'r2'] }] },
        weights: null,
        weights_header: null,
        // float32 [0.5, -0.0] / [masked, 3.0]
        processings: [[{ name: 'snv', matrix: packet('float32', 2, 2, '0000003f000000800000000000004040',
          [true, true, false, true], { ...F64, dtype: 'float32', origin: 'requested', requested_dtype: 'float32' }) }]],
      },
    },
    identity: { provenance: { source_ids: ['s'], sample_id: 'id', observation_id: null } },
    folds: [[[0], [1]]],
  };
}

test('native default sample decodes to literal logical values and keeps packets', () => {
  const input = native.full_default;
  const before = JSON.stringify(input);
  const decoded = decodeAssembledFull(input);
  assert.equal(JSON.stringify(input), before, 'input is not mutated');
  const block = decoded.blocks.train;
  assertLogical(block.x[0].data, [0.1, 0.25, 0.2, NaN, -0, 0.35], 'x');
  assertLogical(block.y.data, [1.5, 0, 3, 2.5, 1, 7, NaN, 0, 3], 'y row-major');
  assertLogical(block.weights, [1, NaN, 0.5], 'weights');
  const { data, ...xPacket } = block.x[0];
  assert.deepEqual(xPacket, input.blocks.train.x[0], 'x packet fields retained');
  assert.deepEqual(block.y.columns, input.blocks.train.y.columns, 'typed target columns retained');
  for (const field of ['metadata', 'feature_axes', 'feature_headers', 'feature_names', 'y_categorical',
    'y_headers', 'representation_ids', 'header_units', 'signal_types', 'weights_header', 'processings']) {
    assert.deepEqual(block[field], input.blocks.train[field], field);
  }
  const { blocks, ...rest } = decoded;
  const { blocks: _, ...inputRest } = input;
  assert.deepEqual(rest, inputRest, 'identity/folds/audits/warnings retained');
});

test('native float32 storage decodes the stored binary32 values exactly', () => {
  const decoded = decodeAssembledFull(native.full_float32);
  assertLogical(decoded.blocks.train.x[0].data,
    [0.10000000149011612, 0.25, 0.20000000298023224, NaN, -0, 0.3499999940395355], 'x float32');
  assert.equal(decoded.blocks.train.x[0].precision.origin, 'requested');
});

test('synthetic packets: subnormal, signed zero, integer classes, labels, processings', () => {
  const decoded = decodeAssembledFull(syntheticFull());
  const block = decoded.blocks.train;
  assertLogical(block.x[0].data, [5e-324, 1, NaN, -2.5], 'x');
  assertLogical(block.processings[0][0].matrix.data, [0.5, -0, NaN, 3], 'processing');
  assertLogical(block.y.data, [2, 1, NaN, NaN], 'y codes');
  assert.equal(block.weights, null);
  assert.deepEqual(decoded.folds, [[[0], [1]]]);
});

const refusals = [
  ['schema 2 wire', (f) => { f.assembled_schema_version = 2; }, /unsupported assembled_schema_version 2/],
  ['missing schema', (f) => { delete f.assembled_schema_version; }, /unsupported assembled_schema_version/],
  ['already decoded', (f) => { f.blocks.train.x[0].data = new Float64Array(4); }, /exactly the fields/],
  ['extra packet field', (f) => { f.blocks.train.x[0].extra = 1; }, /exactly the fields/],
  ['unsafe n_rows', (f) => { f.blocks.train.x[0].n_rows = 2 ** 53; }, /unsigned safe integer/],
  ['fractional n_cols', (f) => { f.blocks.train.x[0].n_cols = 1.5; }, /unsigned safe integer/],
  ['unknown dtype', (f) => { f.blocks.train.x[0].dtype = 'int64'; }, /unsupported numeric dtype/],
  ['precision mismatch', (f) => { f.blocks.train.x[0].precision.dtype = 'float32'; }, /disagrees with precision/],
  ['precision field', (f) => { delete f.blocks.train.x[0].precision.consumer; }, /precision requires exactly/],
  ['uppercase hex', (f) => { f.blocks.train.x[0].data_bytes_hex = f.blocks.train.x[0].data_bytes_hex.replace('f03f', 'F03F'); }, /lowercase hex/],
  ['short hex', (f) => { f.blocks.train.x[0].data_bytes_hex = '00'; }, /byte count mismatch/],
  ['nonboolean mask', (f) => { f.blocks.train.x[0].validity[0] = 1; }, /one boolean per cell/],
  ['masked negative zero', (f) => {
    f.blocks.train.x[0].data_bytes_hex = '0100000000000000000000000000f03f000000000000008000000000000004c0';
  }, /positive zero/],
  ['observed NaN', (f) => {
    f.blocks.train.x[0].data_bytes_hex = '000000000000f87f000000000000f03f000000000000000000000000000004c0';
  }, /must be finite/],
  ['row mismatch', (f) => { f.blocks.train.n_samples = 3; }, /rows differ/],
  ['unknown target kind', (f) => { f.blocks.train.y.columns[0].kind = 'ordinal'; }, /unsupported target kind/],
  ['header disagreement', (f) => { f.blocks.train.y_headers = ['site', 'grade']; }, /disagree with y_headers/],
  ['label outside vocabulary', (f) => { f.blocks.train.y_categorical.site.categories = ['a', 'c']; }, /absent from its category/],
  ['duplicate vocabulary', (f) => { f.blocks.train.y_categorical.site.categories = ['b', 'b']; }, /unique string category/],
  ['missing vocabulary', (f) => { f.blocks.train.y_categorical = {}; }, /unique string category/],
  ['truncated label', (f) => { f.blocks.train.y.columns[1].block.data_bytes_hex = '010000000000000062000000'; }, /truncated label/],
  ['trailing label bytes', (f) => { f.blocks.train.y.columns[1].block.data_bytes_hex += '00'; }, /trailing label/],
  ['masked nonempty label', (f) => {
    f.blocks.train.y.columns[1].block.data_bytes_hex = '010000000000000062010000000000000061';
  }, /masked label must be empty/],
  ['two-column weights', (f) => {
    f.blocks.train.weights = packet('float64', 1, 2, '000000000000f03f000000000000f03f', [true, true]);
  }, /must have one column/],
  ['missing y field', (f) => { delete f.blocks.train.y; }, /requires y/],
];
for (const [name, mutate, message] of refusals) {
  test(`refuses ${name}`, () => {
    const full = syntheticFull();
    mutate(full);
    assert.throws(() => decodeAssembledFull(full), (error) => error instanceof TypeError && message.test(error.message));
  });
}

test('refuses non-object input', () => {
  for (const value of [null, [], 'x', new Float64Array(1)]) {
    assert.throws(() => decodeAssembledFull(value), TypeError);
  }
});

// Optional cross-check against a built WASM `assembleDataset` on the same CSV and
// spec. Uses bindings/wasm/pkg (nodejs target) or IO_WASM_GLUE (nodejs or web target).
const glue = process.env.IO_WASM_GLUE
  ? resolve(process.env.IO_WASM_GLUE) : join(here, '..', 'pkg', 'nirs4all_io_wasm.js');
test('decoded native sample equals WASM assembleDataset', { skip: !existsSync(glue) && `no WASM glue at ${glue}` }, async () => {
  let wasm;
  try {
    wasm = createRequire(import.meta.url)(glue);
  } catch {
    wasm = await import(pathToFileURL(glue).href);
  }
  if (typeof wasm.initSync === 'function') {
    // Web target: instantiate synchronously from the sibling module bytes.
    wasm.initSync({ module: readFileSync(join(dirname(glue), 'nirs4all_io_wasm_bg.wasm')) });
  }
  const files = native.files.map((file) => ({ name: file.name, bytes: new TextEncoder().encode(file.text) }));
  const assembled = wasm.assembleDataset(files, [], JSON.stringify(native.spec));
  assert.deepStrictEqual(decodeAssembledFull(native.full_default), assembled);
});
