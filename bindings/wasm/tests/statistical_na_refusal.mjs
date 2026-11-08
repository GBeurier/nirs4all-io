// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
// Run after rebuilding IO: node statistical_na_refusal.mjs [ABS_CANDIDATE_NODE_GLUE].
// Root can supply its private candidate explicitly; no artifact is rebuilt here.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wasm = require(process.argv[2] ?? '../pkg-node/nirs4all_io_wasm.js');
const A = 'Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;10.0;3.0\n';
const B = 'Sample_ID;set;1100;target\nS_missing;train;;1.0\nS_finite;train;2.0;2.0\nS_test;test;30.0;3.0\n';
const files = text => [{ name: 'literal.csv', bytes: new TextEncoder().encode(text) }];
const spec = na => ({
  schema_version: 1,
  name: 'statistical-na-mean-heldout-probe',
  task_type: 'regression',
  sample_index: { by: 'id', key: 'Sample_ID' },
  params: { delimiter: ';', decimal: '.', has_header: true, na },
  sources: [{
    id: 'x', role: 'mixed', input: 'literal.csv', key: 'Sample_ID',
    params: { header_unit: 'nm' },
    columns: { features: ['1100'], targets: ['target'], metadata: ['set'] },
  }],
  partitions: {
    by: 'column', column: 'set', train_values: ['train'],
    test_values: ['test'], unknown_policy: 'error',
  },
});
const refusal = /na\.fill\.method: .* is refused: .*fitted preprocessing/;
for (const method of ['mean', 'median', ' Median ', 'MEAN']) {
  for (const policy of ['replace', 'ignore', 'remove_sample', 'remove_feature']) {
    for (const per_column of [true, false]) {
      for (const sourceOverride of [false, true]) {
        const na = { policy, fill: { method, per_column } };
        const value = spec(sourceOverride ? { policy: 'ignore' } : na);
        if (sourceOverride) value.sources[0].params.na = na;
        const text = JSON.stringify(value);
        assert.throws(() => wasm.validate(text), refusal);
        assert.throws(() => wasm.assembleDataset(files(A), [], text), refusal);
      }
    }
  }
}

const ignored = JSON.stringify(spec({ policy: 'ignore' }));
wasm.validate(ignored);
const a = wasm.assembleDataset(files(A), [], ignored);
const b = wasm.assembleDataset(files(B), [], ignored);
assert.deepEqual(a.blocks.train, b.blocks.train);
const precision = {
  dtype: 'float64', origin: 'default', source_dtype: null,
  requested_dtype: null, consumer: null,
};
const ids = block => block.metadata.columns.find(column => column.name === 'Sample_ID').values;
for (const full of [a, b]) {
  assert.equal(full.assembled_schema_version, 3);
  assert.deepEqual(full.partition_order, ['train', 'test']);
  assert.deepEqual(full.identity.provenance.source_ids, ['x']);
  assert.equal(full.identity.provenance.sample_id, 'Sample_ID');
  assert.deepEqual(ids(full.blocks.train), ['S_missing', 'S_finite']);
  assert.deepEqual(ids(full.blocks.test), ['S_test']);
  const train = full.blocks.train.x[0];
  assert.equal(train.n_rows, 2);
  assert.equal(train.n_cols, 1);
  assert.equal(train.data_bytes_hex, '00000000000000000000000000000040');
  assert.deepEqual(train.validity, [false, true]);
  assert.deepEqual(train.precision, precision);
  assert.ok(train.data instanceof Float64Array);
  assert.ok(Number.isNaN(train.data[0]));
  assert.equal(train.data[1], 2);
  assert.deepEqual(full.blocks.train.y.columns[0].block.validity, [true, true]);
  assert.equal(full.blocks.train.y.columns[0].block.data_bytes_hex, '000000000000f03f0000000000000040');
}
assert.equal(a.blocks.test.x[0].data[0], 10);
assert.equal(b.blocks.test.x[0].data[0], 30);
console.log('Statistical NA methods refused; literal ignore controls retain train bytes, masks and IDs.');
