// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later

/**
 * Read an assembled-schema-3 numeric coordinate packet into a fresh array.
 * Pass the owner matrix n_cols as expectedCount. Mirrors the numeric branch of
 * materialize/projection.rs::axes_from_full; no ordering or unit conversion.
 * @param {unknown} packet
 * @param {number} expectedCount
 * @returns {Float64Array}
 */
export function numericCoordinates(packet, expectedCount) {
  if (!Number.isSafeInteger(expectedCount) || expectedCount < 0) {
    throw new TypeError("Invalid expected coordinate count");
  }
  if (packet === null || typeof packet !== "object" || Array.isArray(packet)
      || !Object.hasOwn(packet, "kind") || packet.kind !== "numeric") {
    throw new TypeError("Unsupported coordinate kind");
  }
  if (!Object.hasOwn(packet, "count") || !Number.isSafeInteger(packet.count)
      || packet.count < 0 || Object.is(packet.count, -0)) {
    throw new TypeError("Invalid coordinate count");
  }
  if (!Object.hasOwn(packet, "data_bytes_hex") || typeof packet.data_bytes_hex !== "string") {
    throw new TypeError("Exact coordinate bytes required");
  }
  const count = packet.count;
  const text = packet.data_bytes_hex;
  if (Reflect.ownKeys(packet).length !== 3 || count !== expectedCount
      || text.length !== count * 16 || /[^0-9a-f]/.test(text)) {
    throw new TypeError("Coordinate dtype/count/bytes mismatch");
  }
  const coordinates = new Float64Array(count);
  const cell = new DataView(new ArrayBuffer(8));
  for (let index = 0; index < count; index += 1) {
    for (let byte = 0; byte < 8; byte += 1) {
      const offset = index * 16 + byte * 2;
      cell.setUint8(byte, Number.parseInt(text.slice(offset, offset + 2), 16));
    }
    const value = cell.getFloat64(0, true);
    if (!Number.isFinite(value)) {
      throw new TypeError("Observed coordinate is nonfinite");
    }
    coordinates[index] = value;
  }
  return coordinates;
}
