// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later

/**
 * Binding serialization adapter for the native `assembled_full` JSON wire.
 *
 * Native hosts (pyo3 `assembled_full`, CLI, C ABI) emit assembled schema 3:
 * numeric matrices as exact little-endian packets with explicit validity and
 * precision records. The WASM `assembleDataset` export returns the same full
 * value with a logical `Float64Array` attached at `data` (x, processings and
 * the row-major target matrix) and weights as a `Float64Array`.
 * `decodeAssembledFull` produces that JS shape from the JSON wire.
 *
 * This is decoding only. It never assembles, encodes, imputes, filters or
 * reorders anything. All packets and non-numeric fields (metadata, identity,
 * axes, class vocabularies, folds, audits) are cloned unchanged. Masked cells
 * become NaN in the logical arrays, as in `assembleDataset`. Malformed or
 * unsupported input is refused with a TypeError.
 */

const SCHEMA_VERSION = 3;
const NUMERIC_FIELDS = ["dtype", "n_rows", "n_cols", "data_bytes_hex", "validity", "precision"];
const PRECISION_FIELDS = ["dtype", "origin", "source_dtype", "requested_dtype", "consumer"];
const WIDTH = { float32: 4, float64: 8 };

function fail(message) {
  throw new TypeError(`assembled wire: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    && !ArrayBuffer.isView(value);
}

function exactKeys(value, fields, label) {
  if (!isRecord(value)) fail(`${label} must be an object`);
  const keys = Object.keys(value);
  if (keys.length !== fields.length || fields.some((field) => !Object.hasOwn(value, field))) {
    fail(`${label} requires exactly the fields ${fields.join(", ")}`);
  }
}

function dimension(value, label) {
  if (!Number.isSafeInteger(value) || value < 0 || Object.is(value, -0)) {
    fail(`${label} must be an unsigned safe integer`);
  }
  return value;
}

function hexBytes(text, byteCount, label) {
  if (typeof text !== "string") fail(`${label} data_bytes_hex must be a string`);
  if (text.length !== byteCount * 2) fail(`${label} byte count mismatch`);
  const bytes = new Uint8Array(byteCount);
  for (let index = 0; index < byteCount; index += 1) {
    const high = hexDigit(text.charCodeAt(index * 2));
    const low = hexDigit(text.charCodeAt(index * 2 + 1));
    if (high < 0 || low < 0) fail(`${label} requires canonical lowercase hex`);
    bytes[index] = (high << 4) | low;
  }
  return bytes;
}

function hexDigit(code) {
  if (code >= 48 && code <= 57) return code - 48;
  if (code >= 97 && code <= 102) return code - 87;
  return -1;
}

function validityMask(value, count, label) {
  if (!Array.isArray(value) || value.length !== count
      || value.some((flag) => typeof flag !== "boolean")) {
    fail(`${label} validity must contain exactly one boolean per cell`);
  }
  return value;
}

function precisionRecord(value, dtype, label) {
  exactKeys(value, PRECISION_FIELDS, `${label} precision`);
  if (PRECISION_FIELDS.some((field) => value[field] !== null && typeof value[field] !== "string")) {
    fail(`${label} precision requires five scalar fields`);
  }
  if (value.dtype !== dtype) fail(`${label} dtype disagrees with precision record`);
}

/**
 * Decode one float32/float64 packet (Matrix::from_full_value wire) into its
 * logical values: observed cells exactly, masked cells NaN. Masked storage must
 * be canonical positive zero and observed cells finite, as the owner requires.
 */
function numericPacket(packet, label) {
  exactKeys(packet, NUMERIC_FIELDS, label);
  const nRows = dimension(packet.n_rows, `${label} n_rows`);
  const nCols = dimension(packet.n_cols, `${label} n_cols`);
  const count = nRows * nCols;
  if (!Number.isSafeInteger(count)) fail(`${label} shape overflow`);
  if (typeof packet.dtype !== "string" || !Object.hasOwn(WIDTH, packet.dtype)) {
    fail(`${label} unsupported numeric dtype ${JSON.stringify(packet.dtype)}`);
  }
  const width = WIDTH[packet.dtype];
  if (!Number.isSafeInteger(count * width * 2)) fail(`${label} byte count overflow`);
  const validity = validityMask(packet.validity, count, label);
  precisionRecord(packet.precision, packet.dtype, label);
  const view = new DataView(hexBytes(packet.data_bytes_hex, count * width, label).buffer);
  const data = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const value = width === 4
      ? view.getFloat32(index * 4, true)
      : view.getFloat64(index * 8, true);
    if (!validity[index]) {
      if (!Object.is(value, 0)) fail(`${label} masked placeholder must be positive zero`);
      data[index] = Number.NaN;
    } else {
      if (!Number.isFinite(value)) fail(`${label} observed values must be finite`);
      data[index] = value;
    }
  }
  return { nRows, nCols, data, validity };
}

/** Unframe the owner's U64-LE length-prefixed UTF-8 labels (string_target_full). */
function stringLabels(packet, label) {
  exactKeys(packet, NUMERIC_FIELDS, label);
  if (packet.dtype !== "string-label") fail(`${label} requires a string-label packet`);
  const nRows = dimension(packet.n_rows, `${label} n_rows`);
  if (packet.n_cols !== 1) fail(`${label} string-label packet must have one column`);
  const validity = validityMask(packet.validity, nRows, label);
  precisionRecord(packet.precision, "string-label", label);
  if (typeof packet.data_bytes_hex !== "string" || packet.data_bytes_hex.length % 2 !== 0) {
    fail(`${label} byte count mismatch`);
  }
  const bytes = hexBytes(packet.data_bytes_hex, packet.data_bytes_hex.length / 2, label);
  const view = new DataView(bytes.buffer);
  // These are framed label values; a leading U+FEFF is part of the category.
  const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  const labels = [];
  let offset = 0;
  for (let row = 0; row < nRows; row += 1) {
    if (offset + 8 > bytes.length) fail(`${label} truncated label length`);
    const high = view.getUint32(offset + 4, true);
    const length = view.getUint32(offset, true);
    offset += 8;
    if (high !== 0 || offset + length > bytes.length) fail(`${label} truncated label value`);
    let text;
    try {
      text = decoder.decode(bytes.subarray(offset, offset + length));
    } catch {
      fail(`${label} label is not valid UTF-8`);
    }
    offset += length;
    if (!validity[row] && text !== "") fail(`${label} masked label must be empty`);
    labels.push(text);
  }
  if (offset !== bytes.length) fail(`${label} trailing label bytes`);
  return { nRows, labels, validity };
}

/** Owner class vocabulary lookup; the code is the label's unique category index. */
function classCodes(column, categorical, label) {
  const entry = isRecord(categorical) && Object.hasOwn(categorical, column.name)
    ? categorical[column.name] : null;
  const categories = isRecord(entry) ? entry.categories : null;
  if (!Array.isArray(categories) || categories.some((item) => typeof item !== "string")
      || new Set(categories).size !== categories.length) {
    fail(`${label} requires a unique string category vocabulary in y_categorical`);
  }
  const { nRows, labels, validity } = stringLabels(column.block, label);
  const codes = new Float64Array(nRows);
  for (let row = 0; row < nRows; row += 1) {
    if (!validity[row]) {
      codes[row] = Number.NaN;
      continue;
    }
    const code = categories.indexOf(labels[row]);
    if (code < 0) fail(`${label} label is absent from its category vocabulary`);
    codes[row] = code;
  }
  return { nRows, data: codes };
}

function rowsMatch(nRows, nSamples, label) {
  if (nRows !== nSamples) fail(`${label} rows differ from block n_samples`);
}

function decodeTargets(block, nSamples, label) {
  if (block.y === null) return;
  exactKeys(block.y, ["columns"], `${label}.y`);
  const { columns } = block.y;
  if (!Array.isArray(columns) || columns.length === 0) fail(`${label}.y columns must be a non-empty array`);
  if (!Array.isArray(block.y_headers) || block.y_headers.length !== columns.length
      || columns.some((column, index) => !isRecord(column) || column.name !== block.y_headers[index])) {
    fail(`${label}.y columns disagree with y_headers`);
  }
  const width = columns.length;
  const data = new Float64Array(nSamples * width);
  columns.forEach((column, index) => {
    const columnLabel = `${label}.y[${JSON.stringify(column.name)}]`;
    exactKeys(column, ["name", "kind", "block"], columnLabel);
    let decoded;
    if (column.kind === "numeric" || column.kind === "classification_integer") {
      decoded = numericPacket(column.block, columnLabel);
      if (decoded.nCols !== 1) fail(`${columnLabel} must have one column`);
    } else if (column.kind === "classification_string") {
      decoded = classCodes(column, block.y_categorical, columnLabel);
    } else {
      fail(`${columnLabel} unsupported target kind ${JSON.stringify(column.kind)}`);
    }
    rowsMatch(decoded.nRows, nSamples, columnLabel);
    for (let row = 0; row < nSamples; row += 1) data[row * width + index] = decoded.data[row];
  });
  block.y.data = data;
}

function decodeBlock(block, label) {
  if (!isRecord(block)) fail(`${label} must be an object`);
  for (const field of ["x", "processings", "y", "weights"]) {
    if (!Object.hasOwn(block, field)) fail(`${label} requires ${field}`);
  }
  const nSamples = dimension(block.n_samples, `${label} n_samples`);
  if (!Array.isArray(block.x)) fail(`${label}.x must be an array`);
  block.x.forEach((packet, index) => {
    const decoded = numericPacket(packet, `${label}.x[${index}]`);
    rowsMatch(decoded.nRows, nSamples, `${label}.x[${index}]`);
    packet.data = decoded.data;
  });
  if (!Array.isArray(block.processings)) fail(`${label}.processings must be an array`);
  block.processings.forEach((source, sourceIndex) => {
    if (!Array.isArray(source)) fail(`${label}.processings[${sourceIndex}] must be an array`);
    source.forEach((processing, index) => {
      const processingLabel = `${label}.processings[${sourceIndex}][${index}]`;
      exactKeys(processing, ["name", "matrix"], processingLabel);
      const decoded = numericPacket(processing.matrix, `${processingLabel}.matrix`);
      rowsMatch(decoded.nRows, nSamples, processingLabel);
      processing.matrix.data = decoded.data;
    });
  });
  decodeTargets(block, nSamples, label);
  if (block.weights !== null) {
    const decoded = numericPacket(block.weights, `${label}.weights`);
    if (decoded.nCols !== 1) fail(`${label}.weights must have one column`);
    rowsMatch(decoded.nRows, nSamples, `${label}.weights`);
    block.weights = decoded.data;
  }
}

/**
 * Adapt a native `assembled_full` value (assembled schema 3) to the JS shape
 * returned by the WASM `assembleDataset`. The input is not mutated.
 * @param {unknown} value
 * @returns {unknown}
 */
export function decodeAssembledFull(value) {
  if (!isRecord(value)) fail("assembled_full value must be an object");
  if (value.assembled_schema_version !== SCHEMA_VERSION) {
    fail(`unsupported assembled_schema_version ${JSON.stringify(value.assembled_schema_version)}; `
      + `expected ${SCHEMA_VERSION}`);
  }
  let full;
  try {
    full = structuredClone(value);
  } catch {
    fail("assembled_full value must be plain JSON data");
  }
  if (!isRecord(full.blocks)) fail("blocks must be an object");
  for (const [part, block] of Object.entries(full.blocks)) {
    decodeBlock(block, `blocks[${JSON.stringify(part)}]`);
  }
  return full;
}
