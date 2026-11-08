// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later

/** Exact numeric-coordinate packet emitted by assembled-schema-3 axes_full. */
export interface NumericCoordinatesPacket {
  kind: "numeric";
  count: number;
  data_bytes_hex: string;
}

/**
 * Decode finite little-endian binary64 coordinates into a fresh Float64Array.
 * Preserves order and signed zero. Requires exactly the three packet fields,
 * an unsigned safe-integer count, canonical lowercase hex and count * 16 length.
 * expectedCount is required: pass the owner matrix n_cols to check alignment.
 * Throws TypeError on malformed packets or a count mismatch; no default cap.
 */
export function numericCoordinates(packet: unknown, expectedCount: number): Float64Array;
