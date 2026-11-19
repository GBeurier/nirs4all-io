// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later

/**
 * Adapt a native `assembled_full` JSON value (pyo3 `assembled_full`, CLI or
 * C ABI; `assembled_schema_version` 3) to the JS shape returned by the WASM
 * `assembleDataset` export (see `AssembledDataset` in `./types/nirs4all-io`).
 *
 * Each X and processing packet keeps its exact fields and gains a logical
 * `data` Float64Array; `y` keeps its typed `columns` and gains a row-major
 * `data` Float64Array (class codes index `y_categorical[name].categories`);
 * `weights` becomes a Float64Array. Masked cells are NaN. Every other field is
 * cloned unchanged; the input is not mutated.
 *
 * Throws TypeError on any other schema version, a malformed packet, a masked
 * placeholder other than positive zero, a nonfinite observed value, an unsafe
 * integer shape, a row-count mismatch or a label outside its vocabulary.
 * Decoding only: no assembly, encoding, imputation, filtering or reordering.
 */
export function decodeAssembledFull(value: unknown): unknown;
