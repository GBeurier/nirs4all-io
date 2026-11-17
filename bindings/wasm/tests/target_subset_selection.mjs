// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
// Selecting some columns of a file: the others are ignored and never drive the NA policy.
// Build first: wasm-pack build bindings/wasm --target nodejs --out-dir pkg
// then: node bindings/wasm/tests/target_subset_selection.mjs [ABS_NODE_GLUE]
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wasm = require(process.argv[2] ?? '../pkg/nirs4all_io_wasm.js');
const bytes = (text) => new TextEncoder().encode(text);

// Six samples; b misses row 0, the unselected a and c miss rows 3 and 1.
const files = [
  { name: 'X.csv', bytes: bytes(`1000;1100\n${[0, 1, 2, 3, 4, 5].map((i) => `${i};${i + 10}\n`).join('')}`) },
  { name: 'Y.csv', bytes: bytes('a;b;c\n1;;1\n2;2;\n3;3;3\n;4;4\n5;5;5\n6;6;6\n') },
  { name: 'M.csv', bytes: bytes('sid;note\ns0;\ns1;k\ns2;\ns3;k\ns4;\ns5;k\n') },
];
const load = (part, policy, filter) => {
  const spec = wasm.to_spec(JSON.stringify({
    [`${part}_x`]: 'X.csv', [`${part}_y`]: 'Y.csv', [`${part}_y_filter`]: filter,
    [`${part}_group`]: 'M.csv', [`${part}_group_filter`]: ['sid'],
    global_params: { delimiter: ';', has_header: true, na: { policy } },
  }));
  const block = wasm.assembleDataset(files, [], spec).blocks[part];
  return {
    x: Array.from({ length: block.x[0].n_rows }, (_, row) => block.x[0].data[row * block.x[0].n_cols]),
    targets: block.y.columns.map((column) => column.name),
    validity: block.y.columns[0].block.validity,
    sid: block.metadata.columns.find((column) => column.name === 'sid').values,
    metadata: block.metadata.columns.map((column) => column.name),
  };
};

for (const part of ['train', 'test']) {
  for (const filter of [[1], ['b']]) {
    for (const policy of ['ignore', 'auto']) {
      const kept = load(part, policy, filter);
      assert.deepEqual(kept.targets, ['b']);
      assert.deepEqual(kept.metadata, ['sid']);
      assert.deepEqual(kept.x, [0, 1, 2, 3, 4, 5]);
      assert.deepEqual(kept.validity, [false, true, true, true, true, true]);
    }
    const removed = load(part, 'remove_sample', filter);
    assert.deepEqual(removed.x, [1, 2, 3, 4, 5]);
    assert.deepEqual(removed.sid, ['s1', 's2', 's3', 's4', 's5']);
    assert.deepEqual(removed.validity, [true, true, true, true, true]);
  }
  const both = load(part, 'remove_sample', ['a', 'c']);
  assert.deepEqual(both.targets, ['a', 'c']);
  assert.deepEqual(both.x, [0, 2, 4, 5]);
  assert.deepEqual(both.sid, ['s0', 's2', 's4', 's5']);
  assert.throws(() => load(part, 'abort', [1]), /column 'b'/);
}
console.log('Column filters ignore unselected columns; their missing cells never remove a sample or abort.');
