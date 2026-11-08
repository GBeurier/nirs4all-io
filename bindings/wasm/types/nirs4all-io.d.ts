// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//
// Hand-written TypeScript types for the nirs4all-io wasm binding.
//
// wasm-bindgen generates `any` for every `JsValue` parameter and return value
// (see the generated `nirs4all_io_wasm.d.ts`). This file gives the real shapes
// of the inputs and results so editors and `tsc` can check call sites. Import it
// alongside the generated module, or re-export through `idiomatic.mjs`.

import type { NumericCoordinatesPacket } from "../numeric-wire";

/** A raw file: its name (used for source matching) and its bytes. */
export interface NamedFile {
  name: string;
  bytes: Uint8Array;
}

/** Inspect original tables without joining, applying role NA policies or assembling. */
export interface DescribeOptions {
  /** LoadingParams overrides by exact file name; inspection always keeps NA. */
  params?: Record<string, Record<string, unknown>>;
  /** Exact [file name, column name] pairs whose string values are requested. */
  distinct?: [string, string][];
}

/** Existing owner column profile with an uncalibrated role guess. */
export interface TableColumnDescription {
  name: string;
  dtype: "numeric" | "string" | "datetime" | "bool";
  numeric_kind: "Float" | "NonFloatNumeric" | "NonNumeric";
  nunique_with_na: number;
  is_unique: boolean;
  /** First nonmissing owner-rendered value; null when none exists. */
  preview: string | null;
  role: { role: string; score: number; evidence: string[] };
}

/** Verbatim existing input-NA report, before role policies or joins. */
export interface IgnoredInputNaReport {
  strategy: "ignore";
  na_detected: true;
  na_count: number;
  na_samples: number;
  na_features: number;
  removed_samples: [];
  removed_features: [];
  fill_method: null;
  na_preserved: true;
}

/** One result per input, in input order; refusal does not suppress other files. */
export interface TableDescription {
  /** True when an undeclared header cannot be recognized; choose has_header explicitly. */
  header_ambiguous: boolean;
  name: string;
  readable: boolean;
  refusal?: string;
  delimiter_unresolved?: { candidates: string[] };
  /** Effective reader parameters; na.policy is always "ignore" for inspection. */
  params: Record<string, unknown>;
  /** Null on refusal: no shape counts are fabricated. */
  n_rows: number | null;
  n_cols: number | null;
  columns: TableColumnDescription[];
  identity: { column: string; score: number; unique: boolean; evidence: string[] } | null;
  /** Requested columns only, first-appearance order; includes owner NA strings. */
  distinct: Record<string, string[]>;
  warnings: string[];
  /** Absent when the existing helper returns no report or inspection refuses. */
  ignored_input_na_report?: IgnoredInputNaReport;
}

/** A set of decoded `nirs4all-formats` records for one source. */
export interface RecordSet {
  source: string;
  format?: string;
  records: unknown[];
}

/** Options for the inference entry points. */
export interface InferOptions {
  /** Override the default `nirs4all-classic` convention list. */
  conventions?: string[];
  /** Global LoadingParams defaults, applied before the first parse. */
  params?: Record<string, unknown>;
  /** Exact input name -> LoadingParams; overrides global params. */
  file_params?: Record<string, Record<string, unknown>>;
}

/** A user lock fed back into `proposeDataset` to honour a confirmed decision. */
export interface ConfirmedLock {
  kind: string;
  target: string;
  value?: unknown;
  status?: string;
}

/** Options for the iterative `proposeDataset` builder. */
export interface ProposeOptions extends InferOptions {
  confirmed?: ConfirmedLock[];
}

/** A single scored inference decision (structure, signal type, task type, ...). */
export interface Decision<T = unknown> {
  value: T;
  score: number;
  ambiguous: boolean;
  evidence: string[];
  alternatives: unknown[];
}

/** One source of a `DatasetSpec`. */
export interface SourceSpec {
  id: string;
  role: string;
  input: string | string[];
  partition?: string;
  join?: unknown;
  params?: Record<string, unknown>;
  [key: string]: unknown;
}

/** A canonical dataset specification. */
export interface DatasetSpec {
  schema_version: number;
  name?: string;
  sources: SourceSpec[];
  params?: Record<string, unknown>;
  partitions?: unknown;
  folds?: unknown;
  [key: string]: unknown;
}

/** A scored dataset plan produced by the inference entry points. */
export interface DatasetPlan {
  resolved_spec: DatasetSpec;
  overall_score: number;
  recommendations: string[];
  warnings: string[];
  structure: Decision<string>;
  signal_type: Decision<string>;
  task_type: Decision<string>;
  columns: unknown[];
  input: Record<string, unknown>;
  [key: string]: unknown;
}

/** A still-open (or echoed confirmed) decision from `proposeDataset`. */
export interface Proposal {
  kind: string;
  target: string;
  value: unknown;
  alternatives: unknown[];
  evidence: string[];
  score: number;
  ambiguous: boolean;
  status?: string;
}

/** The iterative builder result. */
export interface ProposeResult {
  plan: DatasetPlan;
  proposals: Proposal[];
  spec: DatasetSpec;
  valid: boolean;
  validation_errors: string[];
}

/** Selected numeric storage precision carried by the owner wire. */
export interface AssembledNumericPrecision {
  readonly dtype: "float32" | "float64";
  readonly origin: "default" | "source-declared" | "requested" | "consumer";
  readonly source_dtype: string | null;
  readonly requested_dtype: "float32" | "float64" | null;
  readonly consumer: string | null;
  readonly [key: string]: unknown;
}

/** Exact selected bytes and mask; target column blocks have no logical data view. */
export interface AssembledNumericWire {
  readonly dtype: "float32" | "float64";
  readonly n_rows: number;
  readonly n_cols: number;
  readonly data_bytes_hex: string;
  readonly validity: readonly boolean[];
  readonly precision: AssembledNumericPrecision;
  readonly [key: string]: unknown;
}

/** X/processing matrix with its WASM logical row-major view; missing cells are NaN. */
export interface AssembledMatrix extends AssembledNumericWire {
  readonly data: Float64Array;
}

/** Exact string-label target bytes, not the numeric codes in y.data. */
export interface AssembledStringTargetBlock {
  readonly dtype: "string-label";
  readonly n_rows: number;
  readonly n_cols: 1;
  readonly data_bytes_hex: string;
  readonly validity: readonly boolean[];
  readonly precision: {
    readonly dtype: "string-label";
    readonly origin: "default";
    readonly source_dtype: string | null;
    readonly requested_dtype: null;
    readonly consumer: string | null;
    readonly [key: string]: unknown;
  };
  readonly [key: string]: unknown;
}

/** Column order is owner y_headers order; kind governs the exact block format. */
export type AssembledTargetColumn = {
  readonly name: string;
  readonly kind: "numeric" | "classification_integer";
  readonly block: AssembledNumericWire;
  readonly [key: string]: unknown;
} | {
  readonly name: string;
  readonly kind: "classification_string";
  readonly block: AssembledStringTargetBlock;
  readonly [key: string]: unknown;
};

/** Target wrapper: exact per-column blocks plus the original logical WASM matrix. */
export interface AssembledTargets {
  readonly columns: readonly AssembledTargetColumn[];
  /** Row-major in y_headers order; NaN for missing cells, codes for string labels. */
  readonly data: Float64Array;
  readonly [key: string]: unknown;
}

export type AssembledAxisCoordinates = Readonly<NumericCoordinatesPacket> | {
  readonly kind: "string";
  readonly values: readonly string[];
} | null;

/** Numeric coordinates stay encoded; read them with the owner numericCoordinates. */
export interface AssembledAxis {
  readonly name: string;
  readonly unit: string | null;
  readonly coordinates: AssembledAxisCoordinates;
  readonly [key: string]: unknown;
}

export interface AssembledMetadata {
  readonly n_rows: number;
  readonly columns: readonly {
    readonly name: string;
    readonly values: readonly (number | string | boolean | null)[];
    readonly [key: string]: unknown;
  }[];
  readonly [key: string]: unknown;
}

/** Scientific identity columns; null means no identity column was declared. */
export interface AssembledIdentityProvenance {
  readonly source_ids: readonly string[];
  readonly sample_id: string | null;
  readonly observation_id: string | null;
  readonly repetition_id: string | null;
  readonly group_id: string | null;
  readonly [key: string]: unknown;
}

export interface AssembledFoldProvenance {
  readonly train_observation_ids: readonly string[];
  readonly val_observation_ids: readonly string[];
  readonly [key: string]: unknown;
}

/** One schema-3 partition; per-source arrays align with x and source_ids. */
export interface AssembledBlock {
  readonly n_samples: number;
  readonly source_ids: readonly string[];
  readonly x: readonly AssembledMatrix[];
  readonly feature_headers: readonly (readonly string[])[];
  readonly feature_names: readonly (readonly string[] | null)[];
  readonly feature_axes: readonly (readonly AssembledAxis[])[];
  readonly representation_ids: readonly string[];
  readonly header_units: readonly string[];
  readonly signal_types: readonly (string | null)[];
  readonly y: AssembledTargets | null;
  readonly y_headers: readonly string[];
  readonly y_categorical: Readonly<Record<string, {
    readonly categories: readonly string[];
    readonly [key: string]: unknown;
  }>>;
  readonly metadata: AssembledMetadata | null;
  /** WASM replaces a present full weight packet with its logical typed array. */
  readonly weights: Float64Array | null;
  readonly weights_header: string | null;
  readonly processings: readonly (readonly {
    readonly name: string;
    readonly matrix: AssembledMatrix;
    readonly [key: string]: unknown;
  }[])[];
  readonly [key: string]: unknown;
}

/** The full assembled-schema-3 object returned by WASM assembleDataset. */
export interface AssembledDataset {
  readonly assembled_schema_version: 3;
  readonly name: string;
  readonly task_type: string;
  readonly signal_type: string;
  readonly n_sources: number;
  readonly partition_order: readonly string[];
  readonly blocks: Readonly<Record<string, AssembledBlock>>;
  /** Each pair contains train indices followed by validation indices. */
  readonly folds: readonly (readonly [readonly number[], readonly number[]])[];
  readonly fold_provenance: readonly AssembledFoldProvenance[];
  readonly identity: {
    readonly provenance: AssembledIdentityProvenance;
    readonly [key: string]: unknown;
  };
  readonly repetition: string | null;
  /** Retained owner aggregation configuration; no profile interpretation here. */
  readonly aggregate: Readonly<Record<string, unknown>> | null;
  readonly warnings: readonly string[];
  readonly audits: readonly unknown[];
  readonly [key: string]: unknown;
}

// --- typed views of the wasm-pack exports -------------------------------------

export function to_spec(specJson: string): string;
export function validate(specJson: string): void;
export function version(): string;
export function inferFiles(files: NamedFile[], options?: InferOptions): DatasetPlan;
/** Malformed arguments throw; per-file parsing errors appear in refusal. */
export function describeTables(files: NamedFile[], options?: DescribeOptions): TableDescription[];
export function inferRecords(recordSets: RecordSet[]): DatasetPlan;
export function inferDataset(
  files: NamedFile[],
  recordSets: RecordSet[],
  options?: InferOptions,
): DatasetPlan;
export function proposeDataset(
  files: NamedFile[],
  recordSets: RecordSet[],
  options?: ProposeOptions,
): ProposeResult;
export function assembleDataset(
  files: NamedFile[],
  recordSets: RecordSet[],
  specJson: string,
): AssembledDataset;
/** Canonical structural summary from the same native fs-free assembler. */
export function loadSummary(
  files: NamedFile[],
  recordSets: RecordSet[],
  specJson: string,
): string;
