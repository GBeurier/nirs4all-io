// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
// Missing-value tokens match without regard to ASCII case, in every role.
// Build first: wasm-pack build bindings/wasm --target nodejs --out-dir pkg
// then: node bindings/wasm/tests/missing_token_case.mjs [ABS_NODE_GLUE]
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wasm = require(process.argv[2] ?? '../pkg/nirs4all_io_wasm.js');

const MISSING = [
  '', 'NA', 'na', 'Na', 'nA', 'N/A', 'n/a', 'N/a', 'NaN', 'nan', 'Nan', 'NAN', 'naN', 'nAn', 'null', 'NULL', 'Null',
  'None', 'none', 'NONE', '#N/A', '#n/a', '#NA', '#na', '<NA>', '<na>', '-NaN', '-nan', '-NAN',
];
const TEXT = [' ', '-', '?', '.', 'missing', ' NA'];
const ROW = 3;

const files = (role, token) => {
  const rows = ['id,x1,x2,x3,site,batch,y'];
  for (let row = 0; row < 8; row += 1) {
    const cells = { spectral: `${0.25 * row}`, metadata: `${(row % 3) + 1}`, target: `${10 + 1.5 * row}` };
    if (row === ROW) cells[role] = token;
    rows.push(`s${row},${row}.5,${cells.spectral},${0.5 * row},${'AB'[row % 2]},${cells.metadata},${cells.target}`);
  }
  return [{ name: 'data.csv', bytes: new TextEncoder().encode(`${rows.join('\n')}\n`) }];
};
const spec = JSON.stringify({
  task_type: 'regression',
  sources: [{ id: 't', input: 'data.csv', columns: { features: ['x1', 'x2', 'x3'], targets: 'y', metadata: ['id', 'site', 'batch'] } }],
  sample_index: { by: 'id', key: 'id' },
  params: { delimiter: ',', na: { policy: 'ignore' } },
});
const train = (role, token) => wasm.assembleDataset(files(role, token), [], spec).blocks.train;
const dtype = (role, token, name) => wasm
  .describeTables(files(role, token), { params: { 'data.csv': { delimiter: ',' } } })[0]
  .columns.find((column) => column.name === name).dtype;
const inferredRole = (role, token, name) => wasm
  .inferDataset(files(role, token), [], { params: { delimiter: ',', na: { policy: 'ignore' } } })
  .resolved_spec.sources[0].columns.find((entry) => entry.select.includes(name)).role;
const batch = (block) => block.metadata.columns.find((column) => column.name === 'batch').values;

for (const token of MISSING) {
  const label = JSON.stringify(token);
  const target = train('target', token).y.columns[0];
  assert.equal(target.kind, 'numeric', label);
  assert.deepEqual(target.block.validity, Array.from({ length: 8 }, (_, row) => row !== ROW), label);
  assert.equal(dtype('target', token, 'y'), 'numeric', label);
  assert.equal(inferredRole('target', token, 'y'), 'targets', label);

  const x = train('spectral', token).x[0];
  assert.deepEqual(x.validity, Array.from({ length: 24 }, (_, index) => index !== ROW * 3 + 1), label);
  assert.equal(dtype('spectral', token, 'x2'), 'numeric', label);

  assert.deepEqual(batch(train('metadata', token)), Array.from({ length: 8 }, (_, row) => (row === ROW ? null : (row % 3) + 1)), label);
  assert.equal(dtype('metadata', token, 'batch'), 'numeric', label);
}
for (const token of TEXT) {
  const label = JSON.stringify(token);
  assert.equal(train('target', token).y.columns[0].kind, 'classification_string', label);
  assert.equal(dtype('target', token, 'y'), 'string', label);
  assert.equal(inferredRole('target', token, 'y'), 'metadata', label);
  assert.equal(batch(train('metadata', token))[ROW], token, label);
}
console.log(`Missing-value tokens: ${MISSING.length} spellings missing in target, spectral and metadata columns; ${TEXT.length} stay text.`);
