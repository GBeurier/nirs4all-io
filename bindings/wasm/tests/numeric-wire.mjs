// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const { numericCoordinates } = await import(process.env.NIRS4ALL_IO_NUMERIC_WIRE ?? '../numeric-wire.mjs');

// The Rust unit test includes these exact bytes; neither side generates answers.
const corpus = JSON.parse(readFileSync(new URL('./fixtures/numeric-coordinates.json', import.meta.url), 'utf8'));
assert.equal(corpus.schema, 'nirs4all.io.numeric-coordinates-corpus.v1');
for (const entry of corpus.cases) {
  test(`numeric_coordinates_owner_corpus: ${entry.id}`, () => {
    if (Object.hasOwn(entry, 'refusal')) {
      assert.throws(() => numericCoordinates(entry.packet, entry.expectedCount), {
        name: 'TypeError', message: entry.refusal,
      });
      return;
    }
    const actual = numericCoordinates(entry.packet, entry.expectedCount);
    assert.ok(actual instanceof Float64Array);
    assert.equal(actual.length, entry.expectedCount);
    assert.equal(entry.values.length, entry.expectedCount);
    assert.equal(entry.bits.length, entry.expectedCount);
    const cell = new DataView(new ArrayBuffer(8));
    for (let index = 0; index < actual.length; index += 1) {
      assert.ok(Number.isFinite(actual[index]));
      assert.ok(Object.is(actual[index], entry.values[index]), `${entry.id}[${index}] literal value/sign`);
      // Encode the returned number to compare with literal IEEE-754 bits.
      // This never decodes the packet and is independent of the accessor.
      cell.setFloat64(0, actual[index], false);
      assert.equal(cell.getBigUint64(0, false).toString(16).padStart(16, '0'), entry.bits[index]);
    }
  });
}

test('requires an explicit column count and own packet fields', () => {
  const entry = corpus.cases[0];
  assert.throws(() => numericCoordinates(entry.packet), {
    name: 'TypeError', message: 'Invalid expected coordinate count',
  });
  assert.throws(() => numericCoordinates(Object.create(entry.packet), entry.expectedCount), TypeError);
  assert.throws(() => numericCoordinates({ ...entry.packet, [Symbol('extra')]: 0 }, entry.expectedCount), {
    name: 'TypeError', message: 'Coordinate dtype/count/bytes mismatch',
  });
});
