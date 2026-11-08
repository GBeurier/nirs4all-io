/*
 * io.nirs4all.org: owner-backed Dataset Builder for delimited tables (vanilla ES module).
 *
 * The nirs4all-io WASM owner does all inspection, proposal, validation and materialization:
 *   describeTables -> proposeDataset(confirmed locks) -> column/declaration overlay on the owner
 *   spec -> validate/to_spec -> assembleDataset.
 * This host keeps the original file bytes and the complete owner results, maps the user's explicit
 * decisions onto documented DatasetSpec fields, and hands labels/counts to the shared nirs4all-ui
 * components. It has no table parser, no role or partition heuristic, no numerical method and no
 * missing-value logic: unsupported behaviours are refused visibly, never silently dropped.
 */
import { mountColumnMapping } from "./vendor/ui-columns/column-mapping.mjs";

// --------------------------------------------------------------------------- //
// Vocabulary (owner DatasetSpec / proposal vocabulary, mirrored from builder.js) //
// --------------------------------------------------------------------------- //

const WIZARD_ROLES = ["x", "y", "metadata", "id", "partition", "replicate", "group", "ignored"];
const WIZARD_OF_OWNER = { features: "x", targets: "y", metadata: "metadata", ignore: "ignored" };
const OWNER_OF_WIZARD = {
  x: "features", y: "targets", metadata: "metadata", id: "metadata",
  partition: "metadata", replicate: "metadata", group: "metadata", ignored: "ignore",
};
const OWNER_ROLE_ORDER = ["features", "targets", "metadata", "weights", "ignore"];
const KIND_ORDER = ["structure", "role", "partition", "pairing", "identity", "signal_type", "task_type", "folds"];
const KIND_LABEL = {
  structure: "Structure", role: "Role", partition: "Partition", pairing: "Pairing",
  identity: "Identity", signal_type: "Signal", task_type: "Task", folds: "Folds",
};
const OVERRIDE_VOCAB = {
  role: ["features", "targets", "metadata", "weights", "ignore"],
  partition: ["train", "test", "val", "predict"],
  signal_type: ["absorbance", "reflectance", "reflectance%", "transmittance", "transmittance%", "kubelka-munk"],
  task_type: ["regression", "binary", "multiclass"],
};
// Lock order sent to the owner: role locks first, identity after them, pairing last.
const LOCK_RANK = { role: 0, structure: 1, partition: 1, signal_type: 1, task_type: 1, folds: 1, identity: 2, pairing: 3 };
const NA_POLICIES = ["auto", "abort", "remove_sample", "remove_feature", "ignore", "replace"];
const UNKNOWN_POLICIES = ["train", "test", "drop", "error"];
const AGGREGATE_METHODS = ["mean", "median", "vote", "robust_mean"];
const DELIMITERS = [[";", "; (semicolon)"], [",", ", (comma)"], ["\t", "tab"], ["|", "| (pipe)"]];
const PARTITION_CATEGORIES = [["train", "Train"], ["test", "Test"], ["val", "Validation"], ["predict", "Predict"]];

// --------------------------------------------------------------------------- //
// State                                                                       //
// --------------------------------------------------------------------------- //

const $ = (selector) => document.querySelector(selector);

const state = {
  files: [], // { name, bytes } : the exact bytes the user supplied, sent unchanged to every owner call
  descriptions: new Map(), // file name -> complete owner TableDescription
  distinctKey: "", // the (file, column) pairs the last describeTables call was asked to enumerate
  parsing: new Map(), // file name -> { delimiter?, decimal_separator?, has_header? } explicit user choices
  proposalLocks: [], // locks confirmed through the proposal list
  usedLocks: [], // locks actually sent in the last owner cycle
  edits: new Map(), // file name -> { edited, roles, pinned, guessed, manual, selected, unresolved, mismatch }
  ui: { activeFile: null, filter: "all" },
  name: "",
  naDefault: { policy: "", fill: "" },
  naBySource: new Map(),
  partition: { values: { train: [], test: [], val: [], predict: [] }, unknown_policy: "" },
  replicate: { mode: "", method: "" },
  notice: "", // one-off message shown with the owner notes
  result: null, // the complete owner ProposeResult
  canonical: null, // owner-validated canonical DatasetSpec text (what is exported)
  issues: [], // { level: "error" | "warn", text } produced while mapping decisions to the spec
  specError: null,
  assembled: null, // the complete owner AssembledDataset
  assembleError: null,
  previewKeptMissing: null, // owner NA-abort message when the preview was materialized with missing values kept
  builderError: null,
  pending: 0, // owner cycles in flight
  inputBusy: 0, // file-set changes in flight
  epoch: 0, // moves on reset: a late read must not restore files
  revision: 0, // moves when an input change starts and when it is committed
};

let io = null;
let sentProposals = null; // the proposals array last projected into the list
let proposalsHandle = null;
let columnsHandle = null;
const previewMounts = new Map(); // partition -> { el, handle }

const runtimeReady = (async () => {
  try {
    const mod = await import("./pkg/io/nirs4all_io_wasm.js");
    await mod.default();
    for (const name of ["describeTables", "proposeDataset", "assembleDataset", "validate", "to_spec"]) {
      if (typeof mod[name] !== "function") throw new Error(`the vendored nirs4all-io WASM has no ${name}`);
    }
    io = mod;
    setRuntime(`io ${safe(() => mod.version())}`, "ok", "nirs4all-io WASM owner loaded");
    return true;
  } catch (err) {
    setRuntime("WASM unavailable", "err", String(err?.message || err));
    setStatus(`The nirs4all-io WASM runtime could not be loaded: ${err?.message || err}`);
    return false;
  }
})();

// --------------------------------------------------------------------------- //
// Small helpers                                                               //
// --------------------------------------------------------------------------- //

function safe(fn) {
  try { return fn(); } catch (_) { return "?"; }
}

function setRuntime(text, cls, title) {
  const chip = $("#runtime");
  chip.textContent = text;
  chip.className = `chip ${cls}`;
  chip.title = title || "";
}

function setStatus(text) {
  $("#status").textContent = text;
}

function frame() {
  return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

function h(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (["value", "checked", "disabled", "selected"].includes(key)) node[key] = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

function select(options, value, onchange, props = {}) {
  const node = h("select", { ...props, onchange: (event) => onchange(event.target.value) });
  for (const [optionValue, label] of options) {
    node.append(h("option", { value: optionValue, selected: optionValue === value }, label));
  }
  node.value = value;
  return node;
}

function field(label, control) {
  return h("label", { class: "field" }, h("span", {}, label), control);
}

function msg(kind, text) {
  return h("div", { class: `msg ${kind || ""}`.trim() }, text);
}

function extOf(name) {
  const base = String(name).split("/").pop();
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
}

function listOf(value) {
  if (Array.isArray(value)) return value.filter((item) => typeof item === "string");
  return typeof value === "string" && value ? [value] : [];
}

function singleInput(source) {
  const inputs = Array.isArray(source?.input) ? source.input : [source?.input];
  return inputs.length === 1 && typeof inputs[0] === "string" ? inputs[0] : null;
}

/** All string inputs of a source (single or listed). */
function inputsOf(source) {
  const inputs = Array.isArray(source?.input) ? source.input : [source?.input];
  return inputs.filter((item) => typeof item === "string");
}

/**
 * Readable files that no source of the owner's specification names as an input: they would be left out of the
 * export. Empty until an owner result exists. The owner alone decides which files become sources.
 */
function unassignedFiles() {
  if (!state.result) return [];
  const named = new Set((state.result.spec?.sources || []).flatMap(inputsOf));
  return readableFiles().filter((file) => !named.has(file.name));
}

/** Echoed (user-confirmed) owner pairing proposals; an open proposal is not a decision. */
function confirmedPairings(result) {
  return (result?.proposals || []).filter((proposal) => proposal.kind === "pairing" && proposal.status !== "proposed");
}

function isFeatureLike(source) {
  return ["features", "mixed"].includes(source?.role ?? "features");
}

function sameBytes(a, b) {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
}

function fmtInt(n) {
  return Number.isFinite(n) ? Math.round(n).toLocaleString("en-US") : "-";
}

function fmtBytes(n) {
  return n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;
}

function safeFileStem(name) {
  return String(name || "dataset").trim().replace(/[^a-z0-9._-]+/gi, "_").replace(/^_+|_+$/g, "") || "dataset";
}

const isBusy = () => state.pending > 0 || state.inputBusy > 0;
const ioFiles = () => state.files.map((file) => ({ name: file.name, bytes: file.bytes }));

/** Files the owner reports readable. Every selected file is retained; an unreadable one is shown with the owner's refusal and never reaches proposeDataset/assembleDataset. */
function readableFiles() {
  return state.files.filter((file) => state.descriptions.get(file.name)?.readable);
}

function wizardType(column) {
  if (column.dtype === "numeric") return column.numeric_kind === "NonFloatNumeric" ? "integer" : "float";
  if (column.dtype === "bool") return "boolean";
  if (column.dtype === "datetime") return "date";
  if (column.dtype === "string") return "text";
  return "unknown";
}

// --------------------------------------------------------------------------- //
// Owner calls                                                                 //
// --------------------------------------------------------------------------- //

function distinctRequests() {
  const pairs = [];
  for (const [file, edit] of state.edits) {
    if (!edit.edited) continue;
    for (const [column, role] of edit.roles) if (role === "partition") pairs.push([file, column]);
  }
  return pairs;
}

function describeAll() {
  const params = {};
  for (const [name, choice] of state.parsing) if (Object.keys(choice).length) params[name] = { ...choice };
  const distinct = distinctRequests();
  const described = io.describeTables(ioFiles(), { params, distinct });
  state.descriptions = new Map(described.map((description) => [description.name, description]));
  state.distinctKey = JSON.stringify(distinct);
}

/** Owner-proposed source for a file: the single-input source naming it, if any. */
function sourceFor(result, file) {
  return (result?.spec?.sources || []).find((source) => singleInput(source) === file) || null;
}

/** Initial wizard roles for one file, taken from the owner's literal specification (C1). */
function seedEdit(file, result, description) {
  const source = sourceFor(result, file);
  const names = description.columns.map((column) => column.name);
  const roles = new Map();
  const pinned = new Map();
  const guessed = new Set();
  const unresolved = [];
  const mismatch = [];
  const assign = (name, ownerRole) => {
    if (WIZARD_OF_OWNER[ownerRole]) roles.set(name, WIZARD_OF_OWNER[ownerRole]);
    else pinned.set(name, ownerRole); // for example weights: kept as the owner assigned them
  };
  const keys = listOf(source?.key);
  const entries = Array.isArray(source?.columns) ? source.columns : [];
  for (const entry of entries) {
    const select = entry.select;
    if (Array.isArray(select) && select.every((item) => typeof item === "string")) {
      for (const name of select) {
        if (names.includes(name)) assign(name, entry.role);
        else mismatch.push(name);
      }
    } else {
      unresolved.push(`${entry.role}: ${JSON.stringify(select)}`);
    }
  }
  if (!entries.length && source && ["features", "targets", "metadata", "ignore", "weights"].includes(source.role)) {
    for (const name of names) if (!keys.includes(name)) assign(name, source.role);
  }
  for (const name of keys) if (names.includes(name)) { roles.set(name, "id"); pinned.delete(name); }
  description.columns.forEach((column) => {
    if (roles.has(column.name) || pinned.has(column.name)) return;
    // Not decided by a literal selector: show the owner's column guess, labelled as a guess.
    // An identity is never guessed into a role: it comes from the owner's key or from your choice.
    roles.set(column.name, WIZARD_OF_OWNER[column.role?.role] || "metadata");
    guessed.add(column.name);
  });
  const seedIds = names.filter((name) => roles.get(name) === "id");
  return { edited: false, roles, pinned, guessed, manual: new Set(), selected: new Set(), unresolved, mismatch, seedIds };
}

/** The edit state of a file: re-seeded from the latest owner spec until the user changes a role. */
function editFor(file) {
  const description = state.descriptions.get(file);
  if (!description?.readable) return null;
  let edit = state.edits.get(file);
  if (!edit || !edit.edited) {
    const seeded = seedEdit(file, state.result, description);
    if (edit) seeded.selected = edit.selected;
    edit = seeded;
    state.edits.set(file, edit);
  }
  return edit;
}

function ownerRoleOf(edit, name) {
  if (edit.pinned.has(name)) return edit.pinned.get(name);
  return OWNER_OF_WIZARD[edit.roles.get(name) ?? "ignored"];
}

/** Owner role -> non-key column names, for one edited source. Key columns are never listed. */
function plannedGroups(source, edit, description) {
  const keys = listOf(source.key);
  const groups = new Map();
  for (const column of description.columns) {
    if (keys.includes(column.name)) continue;
    const role = ownerRoleOf(edit, column.name);
    if (!groups.has(role)) groups.set(role, []);
    groups.get(role).push(column.name);
  }
  return groups;
}

function plannedRole(groups) {
  if (groups.size === 0) return null;
  return groups.size === 1 ? [...groups.keys()][0] : "mixed";
}

/** id columns an edited source declares (wizard role id), in header order. */
function declaredIds(edit, description) {
  return description.columns.map((column) => column.name).filter((name) => edit.roles.get(name) === "id");
}

/** The ID declaration only counts as your decision when it differs from the owner's key at the first edit. */
function changedIds(edit, description) {
  const ids = declaredIds(edit, description);
  return JSON.stringify(ids) === JSON.stringify(edit.seedIds) ? [] : ids;
}

/** Locks derived from the column editor; they win over a proposal lock with the same kind and target. */
function editorLocks(result) {
  const locks = [];
  if (!result) return locks;
  const identity = [];
  for (const source of result.spec?.sources || []) {
    const file = singleInput(source);
    const edit = file && state.edits.get(file);
    const description = file && state.descriptions.get(file);
    if (!edit?.edited || !description?.readable) continue;
    const role = plannedRole(plannedGroups(source, edit, description));
    if (role) locks.push({ kind: "role", target: file, value: role, status: "confirmed" });
    const ids = changedIds(edit, description);
    if (!ids.length) continue;
    if (isFeatureLike(source)) {
      identity.push(ids.length === 1 ? ids[0] : { by: "id", key: ids });
    } else if (ids.length === 1) {
      const pairing = (result.proposals || []).find((proposal) => proposal.kind === "pairing"
        && String(proposal.target).split("|").some((part) => part === source.id || part === file));
      const [left, right] = pairing ? String(pairing.target).split("|") : [];
      const anchorId = left === source.id || left === file ? right : left;
      const anchor = pairing && (result.spec.sources || []).find((item) => item.id === anchorId);
      // A join anchor needs a key that is a real column of the anchor's own table: a key the identity lock
      // wrote onto a table that does not contain it (the owner tolerates it) is not an anchor.
      const anchorFile = anchor && singleInput(anchor);
      const anchorColumns = (anchorFile && state.descriptions.get(anchorFile)?.columns || []).map((column) => column.name);
      const anchorCandidate = anchor && listOf(anchor.key)[0];
      const anchorKey = anchorCandidate && anchorColumns.includes(anchorCandidate) ? anchorCandidate : null;
      if (pairing && anchorKey) {
        locks.push({
          kind: "pairing", target: pairing.target, status: "confirmed",
          value: { mode: "join_id", left_on: anchorKey, right_on: ids[0] },
        });
      } else if (pairedWithEveryFeatureTable(result, source)) {
        // No keyed anchor to join on (the feature tables carry no ID, tables stay paired as the owner proposed):
        // the user's explicit choice is the dataset identity, sent as the owner's identity lock. The owner then
        // requires that column in the assembled data; nothing is inferred and no pairing is changed.
        identity.push(ids[0]);
      }
      // Otherwise the decision is held back and reported by buildSpec: the identity would be attached to the
      // feature tables through an alignment the user has not confirmed.
    }
  }
  const distinct = new Set(identity.map((entry) => JSON.stringify(entry)));
  if (distinct.size === 1) locks.push({ kind: "identity", target: "dataset", value: identity[0], status: "confirmed" });
  return locks;
}

/**
 * A non-feature table may carry the dataset identity only when the owner pairing between it and every
 * feature table of its partition has been echoed as a confirmed decision (row order or id join).
 */
function pairedWithEveryFeatureTable(result, source) {
  const confirmed = confirmedPairings(result).map((proposal) => String(proposal.target).split("|"));
  const features = (result.spec?.sources || []).filter((other) => isFeatureLike(other)
    && (!source.partition || !other.partition || other.partition === source.partition));
  return features.length > 0 && features.every((feature) => confirmed.some((pair) => pair.length === 2
    && pair.includes(feature.id) && pair.includes(source.id)));
}

function composeLocks(result) {
  const byKey = new Map();
  for (const lock of [...state.proposalLocks, ...editorLocks(result)]) byKey.set(`${lock.kind}|${lock.target}`, lock);
  return [...byKey.values()].sort((a, b) => (LOCK_RANK[a.kind] ?? 1) - (LOCK_RANK[b.kind] ?? 1));
}

function naObject(current, label, issues) {
  if (!current.policy) return null;
  if (current.policy !== "replace") return { policy: current.policy };
  const fill = Number(current.fill);
  if (current.fill === "" || !Number.isFinite(fill)) {
    issues.push({ level: "error", text: `${label}: the replace policy needs a finite numeric fill value.` });
    return null;
  }
  return { policy: "replace", fill: { method: "value", fill_value: fill } };
}

/** Map the user's explicit decisions onto the owner specification. Pure transcription: the owner validates. */
function buildSpec(result) {
  const spec = structuredClone(result.spec);
  const issues = [];
  if (state.name.trim()) spec.name = state.name.trim();

  const declared = { partition: [], replicate: [], group: [] };
  const identityDecls = [];
  for (const source of spec.sources || []) {
    const file = singleInput(source);
    const edit = file && state.edits.get(file);
    const description = file && state.descriptions.get(file);
    if (!file) continue;
    if (description?.readable) {
      const ids = edit?.edited ? changedIds(edit, description) : [];
      if (edit?.edited) {
        if (isFeatureLike(source) && ids.length > 1) {
          source.key = ids; // composite key: written literally, together with sample_index below
          identityDecls.push(ids);
        } else if (isFeatureLike(source) && ids.length === 1) {
          identityDecls.push(ids);
          if (JSON.stringify(listOf(source.key)) !== JSON.stringify(ids)) {
            issues.push({ level: "error", text: `The owner did not apply the identity key '${ids[0]}' to ${file}. Check the owner notes.` });
          }
        } else if (!isFeatureLike(source) && ids.length === 1) {
          const join = source.join;
          const identityKey = spec.sample_index?.by === "id" ? listOf(spec.sample_index.key) : [];
          if (join && join.right_on === ids[0]) {
            // joined on the ID you chose (owner pairing lock)
          } else if (identityKey.length === 1 && identityKey[0] === ids[0]) {
            identityDecls.push(ids); // dataset identity declared from this table (owner identity lock)
          } else if (!pairedWithEveryFeatureTable(result, source)) {
            issues.push({ level: "error", text: `ID column '${ids[0]}' of ${file} was not applied: confirm the owner pairing between ${file} and every feature table first. Otherwise the identity would be attached to those tables through an alignment you have not confirmed.` });
          } else {
            issues.push({ level: "error", text: `ID column '${ids[0]}' of ${file} was not applied: neither an owner id pairing nor the owner identity lock carried it. Remove the ID role or check the owner notes.` });
          }
        } else if (!isFeatureLike(source) && ids.length > 1) {
          issues.push({ level: "error", text: `${file}: several ID columns on a joined source are not applied. Keep a single ID column.` });
        }
        const groups = plannedGroups(source, edit, description);
        if (groups.size === 1) {
          source.role = [...groups.keys()][0];
          delete source.columns;
        } else if (groups.size > 1) {
          const ordered = [...OWNER_ROLE_ORDER.filter((role) => groups.has(role)), ...[...groups.keys()].filter((role) => !OWNER_ROLE_ORDER.includes(role))];
          source.role = "mixed";
          source.columns = ordered.map((role) => ({ role, select: groups.get(role) }));
        }
        for (const [name, role] of edit.roles) {
          if (role in declared) declared[role].push({ file, column: name });
        }
      }
      // Parsing: explicit user choices, and the effective values the description used when undeclared (C5).
      const choice = state.parsing.get(file) || {};
      const params = { ...(source.params || {}) };
      for (const key of ["delimiter", "decimal_separator", "has_header"]) {
        if (choice[key] !== undefined) params[key] = choice[key];
        else if (params[key] === undefined && spec.params?.[key] === undefined && description.params?.[key] !== undefined) params[key] = description.params[key];
      }
      const na = naObject(state.naBySource.get(file) || { policy: "", fill: "" }, `${file} missing values`, issues);
      if (na) params.na = na;
      if (Object.keys(params).length) source.params = params;
    }
  }

  if (identityDecls.length) {
    const distinct = new Set(identityDecls.map((ids) => JSON.stringify(ids)));
    if (distinct.size > 1) issues.push({ level: "error", text: "Sources declare different ID columns; one identity key per dataset is supported." });
    else if (identityDecls[0].length > 1) spec.sample_index = { ...(spec.sample_index || {}), by: "id", key: identityDecls[0] };
  }

  const datasetNa = naObject(state.naDefault, "Dataset missing values", issues);
  if (datasetNa) spec.params = { ...(spec.params || {}), na: datasetNa };

  if (declared.partition.length > 1) {
    issues.push({ level: "error", text: "Only one partition column is supported; assign the Partition role to a single column." });
  } else if (declared.partition.length === 1) {
    const { column } = declared.partition[0];
    const partitions = { by: "column", column };
    const seen = new Map();
    for (const [category] of PARTITION_CATEGORIES) {
      const values = state.partition.values[category];
      if (values.length) partitions[`${category}_values`] = [...values];
      for (const value of values) {
        if (seen.has(value)) issues.push({ level: "error", text: `Partition value '${value}' is assigned to both ${seen.get(value)} and ${category}.` });
        seen.set(value, category);
      }
    }
    if (!seen.size) issues.push({ level: "error", text: `Choose the values of '${column}' that belong to each partition (nothing is preselected).` });
    else {
      if (state.partition.unknown_policy) partitions.unknown_policy = state.partition.unknown_policy;
      if (spec.partitions) issues.push({ level: "warn", text: "The column partition you declared replaces the partitions the owner proposed." });
      spec.partitions = partitions;
    }
  }

  if (declared.replicate.length > 1) {
    issues.push({ level: "error", text: "Only one replicate column is supported." });
  } else if (declared.replicate.length === 1) {
    const { column } = declared.replicate[0];
    if (state.replicate.mode === "repetition_id") {
      if (!spec.sample_index) issues.push({ level: "error", text: "The per-scan replicate index needs an ID column: assign the ID role to a column first." });
      else spec.sample_index = { ...spec.sample_index, repetition_id: column };
    } else if (state.replicate.mode === "repetition") {
      spec.repetition = column;
      if (state.replicate.method) {
        const next = { by: column, method: state.replicate.method };
        if (spec.aggregate && JSON.stringify(spec.aggregate) !== JSON.stringify(next)) {
          issues.push({ level: "warn", text: `The aggregation you chose replaces the owner's aggregate ${JSON.stringify(spec.aggregate)}.` });
        }
        spec.aggregate = next;
      }
    } else {
      issues.push({ level: "error", text: `Say how replicate column '${column}' is used (nothing is preselected).` });
    }
  }
  if (declared.group.length > 1) {
    issues.push({ level: "error", text: "Only one group column is supported." });
  } else if (declared.group.length === 1) {
    if (!spec.sample_index) issues.push({ level: "error", text: "The group column needs an ID column: assign the ID role to a column first." });
    else spec.sample_index = { ...spec.sample_index, group_id: declared.group[0].column };
  }
  return { spec, issues };
}

/** Mirrors the main page: preview-only copy whose aborting NA policies keep missing values. */
function keepMissingValues(spec) {
  const aborts = (na, inherited) => na?.policy === "abort" || (inherited && (!na?.policy || na.policy === "auto"));
  const keep = (params, inherited) => (aborts(params?.na, inherited) ? { ...params, na: { ...params?.na, policy: "ignore" } } : params);
  return {
    ...spec,
    params: keep(spec.params || {}, true),
    sources: (spec.sources || []).map((source) => ({
      ...source,
      params: keep(source.params, false),
      ...(source.variations ? { variations: source.variations.map((variation) => ({ ...variation, params: keep(variation.params, false) })) } : {}),
    })),
  };
}

function clearDerived() {
  Object.assign(state, {
    result: null, canonical: null, issues: [], specError: null, assembled: null, assembleError: null,
    previewKeptMissing: null, builderError: null, usedLocks: [],
  });
  sentProposals = null;
}

/** One owner cycle: propose with locks, overlay, validate, materialize. */
async function cycle(stale = () => false) {
  state.pending += 1;
  renderAll();
  try {
    await frame();
    if (stale()) return;
    state.builderError = null;
    const files = readableFiles().map((file) => ({ name: file.name, bytes: file.bytes }));
    if (!files.length) {
      clearDerived();
      return;
    }
    const names = new Set(files.map((file) => file.name));
    const file_params = Object.fromEntries([...state.parsing].filter(([name, choice]) => names.has(name) && Object.keys(choice).length));
    let locks = composeLocks(state.result);
    let result = null;
    for (let pass = 0; pass < 2; pass++) {
      result = io.proposeDataset(files, [], { confirmed: locks, file_params });
      const next = composeLocks(result);
      if (JSON.stringify(next) === JSON.stringify(locks)) break;
      locks = next;
    }
    if (stale()) return;
    state.result = result;
    state.usedLocks = locks;
    for (const file of state.files) editFor(file.name);
    finishCycle(files);
  } catch (err) {
    clearDerived();
    state.builderError = String(err?.message || err);
  } finally {
    state.pending -= 1;
    renderAll();
  }
}

/** Rebuild the specification and preview from the retained owner result. */
function finishCycle(files = readableFiles().map((file) => ({ name: file.name, bytes: file.bytes }))) {
  const result = state.result;
  state.canonical = null;
  state.specError = null;
  state.assembled = null;
  state.assembleError = null;
  state.previewKeptMissing = null;
  if (!result) return;
  const { spec, issues } = buildSpec(result);
  state.issues = issues;
  let canonical;
  try {
    const raw = JSON.stringify(spec);
    io.validate(raw);
    canonical = io.to_spec(raw);
    io.validate(canonical);
  } catch (err) {
    state.specError = String(err?.message || err);
    return;
  }
  state.canonical = canonical;
  try {
    state.assembled = io.assembleDataset(files, [], canonical);
  } catch (err) {
    const message = String(err?.message || err);
    state.assembleError = message;
    if (message.includes("na_policy is 'abort'")) {
      try {
        state.assembled = io.assembleDataset(files, [], JSON.stringify(keepMissingValues(JSON.parse(canonical))));
        state.previewKeptMissing = message;
        state.assembleError = null;
      } catch (retry) {
        state.assembleError = `${message}\nNA positions are not available: the owner could not materialize the dataset even with missing values kept (${String(retry?.message || retry)}).`;
      }
    }
  }
}

// --------------------------------------------------------------------------- //
// Serialized recomputation and input binding                                  //
// --------------------------------------------------------------------------- //

let queue = Promise.resolve();
function enqueue(task) {
  queue = queue.then(task, task);
  return queue;
}

async function runRecompute() {
  const revision = state.revision;
  const stale = () => revision !== state.revision;
  clearDerived();
  if (!state.files.length) {
    state.descriptions = new Map();
    renderAll();
    return;
  }
  setStatus("Inspecting tables with nirs4all-io…");
  await frame();
  try {
    describeAll();
  } catch (err) {
    state.builderError = `Inspection failed: ${err?.message || err}`;
    renderAll();
    return;
  }
  if (stale()) return;
  await cycle(stale);
}

function beginInputChange() {
  state.notice = "";
  state.revision += 1;
  state.inputBusy += 1;
  clearDerived();
  renderAll();
}

function endInputChange() {
  state.inputBusy -= 1;
  renderAll();
}

async function changeInputs(load, onError) {
  const epoch = state.epoch;
  beginInputChange();
  let failure = null;
  try {
    const loaded = await load();
    if (epoch !== state.epoch) return;
    let replaced = false;
    for (const incoming of loaded) {
      const before = state.files.find((file) => file.name === incoming.name);
      if (before && !sameBytes(before.bytes, incoming.bytes)) replaced = true;
      state.files = state.files.filter((file) => file.name !== incoming.name);
      // Every selected input is retained, whatever its name: describeTables decides readability.
      state.files.push(incoming);
      if (before) {
        state.edits.delete(incoming.name);
        state.parsing.delete(incoming.name);
      }
    }
    if (replaced) {
      // Decisions were validated against the old bytes: start over.
      state.proposalLocks = [];
      state.edits.clear();
    }
    state.revision += 1;
    await enqueue(runRecompute);
    if (replaced) {
      state.notice = "A file was replaced, so earlier decisions were cleared and must be made again.";
      renderAll();
    }
  } catch (err) {
    failure = err;
  } finally {
    endInputChange();
  }
  if (failure) {
    if (onError) onError(failure);
    if (state.files.length) await enqueue(runRecompute).catch(() => {});
  }
}

function browserFileName(file) {
  return String(file.webkitRelativePath || file.name || "file").replace(/\\/g, "/").replace(/^\/+/, "");
}

async function addFiles(fileList) {
  const list = [...fileList];
  if (!list.length) return;
  if (!(await runtimeReady)) return;
  await changeInputs(
    async () => Promise.all(list.map(async (file) => ({ name: browserFileName(file), bytes: new Uint8Array(await file.arrayBuffer()) }))),
    (err) => setStatus(`Files could not be loaded: ${err?.message || err}`),
  );
}

async function loadSamples() {
  if (!(await runtimeReady)) return;
  await changeInputs(async () => {
    const manifest = await (await fetch("./samples/manifest.json")).json();
    return Promise.all(manifest.files.map(async (entry) => {
      const response = await fetch(`./samples/${entry.name}`);
      if (!response.ok) throw new Error(`sample ${entry.name}: HTTP ${response.status}`);
      return { name: entry.name, bytes: new Uint8Array(await response.arrayBuffer()) };
    }));
  }, (err) => setStatus(`The sample dataset could not be loaded: ${err?.message || err}`));
}

async function removeFile(name) {
  if (isBusy()) return;
  const epoch = state.epoch;
  beginInputChange();
  try {
    if (epoch !== state.epoch) return;
    state.files = state.files.filter((file) => file.name !== name);
    state.edits.delete(name);
    state.parsing.delete(name);
    state.naBySource.delete(name);
    state.proposalLocks = []; // locks may name the removed file
    state.revision += 1;
    await enqueue(runRecompute);
  } finally {
    endInputChange();
  }
}

function reset() {
  state.epoch += 1;
  state.revision += 1;
  Object.assign(state, {
    files: [], descriptions: new Map(), distinctKey: "", parsing: new Map(), proposalLocks: [],
    edits: new Map(), ui: { activeFile: null, filter: "all" }, name: "", notice: "", naDefault: { policy: "", fill: "" },
    naBySource: new Map(), partition: { values: { train: [], test: [], val: [], predict: [] }, unknown_policy: "" },
    replicate: { mode: "", method: "" },
  });
  clearDerived();
  columnsHandle?.unmount();
  columnsHandle = null;
  for (const { el, handle } of previewMounts.values()) { handle.unmount(); el.remove(); }
  previewMounts.clear();
  setStatus("Waiting for files.");
  renderAll();
}

// --------------------------------------------------------------------------- //
// Decision actions                                                            //
// --------------------------------------------------------------------------- //

function upsertLock(lock) {
  const index = state.proposalLocks.findIndex((item) => item.kind === lock.kind && item.target === lock.target);
  if (index >= 0) state.proposalLocks[index] = lock;
  else state.proposalLocks.push(lock);
}

function currentProposal(id) {
  if (isBusy() || sentProposals !== state.result?.proposals) return null;
  return state.result.proposals.find((proposal) => proposal.id === id && proposal.status === "proposed") || null;
}

function onConfirm(id) {
  const proposal = currentProposal(id);
  if (!proposal) return;
  upsertLock({ kind: proposal.kind, target: proposal.target, value: proposal.value, status: "confirmed" });
  enqueue(() => cycle());
}

function onAlternative(id, choiceId) {
  const proposal = currentProposal(id);
  const index = Number(choiceId);
  const alt = proposal && Number.isInteger(index) && String(index) === choiceId ? (proposal.alternatives || [])[index] : null;
  if (!alt) return;
  upsertLock({ kind: proposal.kind, target: proposal.target, value: alt.value, status: "overridden" });
  enqueue(() => cycle());
}

function onOverride(id, value) {
  const proposal = currentProposal(id);
  if (!proposal || !(OVERRIDE_VOCAB[proposal.kind] || []).includes(value)) return;
  upsertLock({ kind: proposal.kind, target: proposal.target, value, status: "overridden" });
  enqueue(() => cycle());
}

function onUndo(kind, target) {
  if (isBusy()) return;
  state.proposalLocks = state.proposalLocks.filter((lock) => !(lock.kind === kind && lock.target === target));
  enqueue(() => cycle());
}

function assignRoles(file, names, role) {
  if (isBusy() || !WIZARD_ROLES.includes(role)) return;
  const edit = editFor(file);
  const description = state.descriptions.get(file);
  if (!edit || !description) return;
  if (unassignedFiles().some((item) => item.name === file)) return; // read-only: the owner proposed no source
  const source = sourceFor(state.result, file);
  const next = new Map(edit.roles);
  for (const name of names) if (next.has(name)) next.set(name, role);
  // The owner spec requires the source key: refuse to leave a keyed source without an ID column.
  if (source && listOf(source.key).length && ![...next.values()].includes("id")) {
    setStatus(`${file} has an owner-declared key; choose another ID column before changing the ID role.`);
    return;
  }
  const cleaned = { ...edit, edited: true, roles: next, manual: new Set([...edit.manual, ...names]) };
  state.edits.set(file, cleaned);
  enqueue(async () => {
    if (JSON.stringify(distinctRequests()) !== state.distinctKey) {
      try { describeAll(); } catch (err) { state.builderError = `Inspection failed: ${err?.message || err}`; }
    }
    await cycle();
  });
}

function rebuild() {
  if (isBusy() || !state.result) return;
  enqueue(async () => {
    state.pending += 1;
    renderAll();
    try {
      await frame();
      finishCycle();
    } finally {
      state.pending -= 1;
      renderAll();
    }
  });
}

function setParsing(file, key, value) {
  if (isBusy()) return;
  const choice = { ...(state.parsing.get(file) || {}) };
  if (value === undefined) delete choice[key];
  else choice[key] = value;
  if (Object.keys(choice).length) state.parsing.set(file, choice);
  else state.parsing.delete(file);
  state.edits.delete(file); // the column list may change with the dialect
  enqueue(async () => {
    state.pending += 1;
    renderAll();
    try {
      describeAll();
    } catch (err) {
      state.builderError = `Inspection failed: ${err?.message || err}`;
    } finally {
      state.pending -= 1;
    }
    await cycle();
  });
}

// --------------------------------------------------------------------------- //
// Rendering                                                                   //
// --------------------------------------------------------------------------- //

function orderByKind(proposals) {
  const rank = (kind) => (KIND_ORDER.includes(kind) ? KIND_ORDER.indexOf(kind) : KIND_ORDER.length);
  return [...proposals].sort((a, b) => rank(a.kind) - rank(b.kind));
}

function valueLabel(p) {
  const v = p.value;
  if (v && typeof v === "object") {
    if (v.mode === "join_id") return `join by id (${v.on || v.right_on || "shared key"})`;
    if (v.mode === "row_order") return "align by row order";
    if (v.mode === "concat_samples") return `stack ${(v.parts || []).length} files → ${v.whole || "y"}`;
    if (v.by) return v.key ? `by ${v.by} · ${Array.isArray(v.key) ? v.key.join(", ") : v.key}` : `by ${v.by}`;
    return JSON.stringify(v);
  }
  return String(v);
}

function projectProposal(p) {
  const display = {
    id: p.id,
    kind: p.kind,
    target: p.target,
    valueLabel: valueLabel(p),
    kindLabel: KIND_LABEL[p.kind] || p.kind,
    score: Number.isFinite(p.score) ? p.score : null,
    ambiguous: Boolean(p.ambiguous),
    alternatives: (p.alternatives || []).map((alt, index) => ({
      choiceId: String(index),
      label: valueLabel({ value: alt.value }),
      score: Number.isFinite(alt.score) ? alt.score : null,
    })),
  };
  if (Array.isArray(p.evidence) && p.evidence.length) display.evidence = p.evidence.map(String);
  const vocab = OVERRIDE_VOCAB[p.kind];
  if (vocab) display.overrideChoices = vocab.map((option) => ({ choiceId: option, label: option, value: option }));
  return display;
}

function sharedUI() {
  const ui = window.Nirs4allUIBrowser;
  if (!ui || typeof ui.mountOwnerProposals !== "function" || typeof ui.mountDatasetPreview !== "function") {
    throw new Error("The shared nirs4all-ui browser bundle is not loaded.");
  }
  return ui;
}

function renderProposals() {
  const open = (state.result?.proposals || []).filter((proposal) => proposal.status === "proposed");
  const echoes = (state.result?.proposals || []).filter((proposal) => proposal.status !== "proposed");
  const props = {
    proposals: orderByKind(open).map(projectProposal),
    busy: isBusy(),
    error: state.builderError,
    emptyMessage: state.builderError
      ? ""
      : state.inputBusy > 0
        ? "Input files are changing: decisions are proposed again once inspection has finished."
        : echoes.length
          ? "All decisions confirmed — review the materialized preview below."
          : "Add table files to see decisions to validate.",
    onConfirm,
    onAlternative,
    onOverride,
  };
  sentProposals = state.result?.proposals ?? null;
  if (proposalsHandle) proposalsHandle.update(props);
  else proposalsHandle = sharedUI().mountOwnerProposals($("#proposals-host"), props);

  const host = $("#echoes");
  if (!echoes.length) {
    host.replaceChildren();
  } else {
    const editorKeys = new Set(editorLocks(state.result).map((lock) => `${lock.kind}|${lock.target}`));
    host.replaceChildren(h("details", { class: "confirmed-list" },
      h("summary", {}, `Confirmed (${echoes.length})`),
      ...echoes.map((proposal) => {
        const fromEditor = editorKeys.has(`${proposal.kind}|${proposal.target}`);
        return h("div", { class: "msg" },
          `${KIND_LABEL[proposal.kind] || proposal.kind} · ${proposal.target}: ${valueLabel(proposal)} `,
          fromEditor
            ? h("em", {}, "(from the column mapping)")
            : h("button", { class: "small", type: "button", disabled: isBusy(), onclick: () => onUndo(proposal.kind, proposal.target) }, "Undo"));
      })));
  }

  // Owner notes: warnings verbatim, locks the owner did not echo, and an incomplete owner spec.
  const notes = [];
  if (state.notice) notes.push(msg("warn", state.notice));
  for (const warning of state.result?.plan?.warnings || []) notes.push(msg("warn", warning));
  if (state.result && !state.result.valid) {
    notes.push(msg("warn", `The owner's own proposed specification is incomplete: ${(state.result.validation_errors || []).join("; ") || "no detail"}`));
  }
  const echoedKeys = new Set(echoes.map((proposal) => `${proposal.kind}|${proposal.target}`));
  for (const lock of state.usedLocks) {
    if (!echoedKeys.has(`${lock.kind}|${lock.target}`)) {
      notes.push(msg("warn", `The owner did not echo the ${lock.kind} lock for '${lock.target}' (value ${JSON.stringify(lock.value)}).`));
    }
  }
  $("#owner-notes").replaceChildren(...notes);
}

function renderFiles() {
  const host = $("#files");
  if (!state.files.length) {
    host.replaceChildren(h("div", { class: "empty" }, "No files yet. Drop table files here, use “Add files”, or load the sample."));
    return;
  }
  const rows = [];
  for (const file of state.files) {
    const description = state.descriptions.get(file.name);
    let status;
    if (!description) status = h("span", { class: "chip" }, "inspecting…");
    else if (description.readable) {
      const unassigned = unassignedFiles().some((item) => item.name === file.name);
      status = unassigned
        ? h("div", {},
          h("span", { class: "chip warn" }, `${fmtInt(description.n_rows)} rows × ${fmtInt(description.n_cols)} columns · unassigned`),
          h("div", { class: "msg warn" }, "Readable as delimited text, but the owner proposed no source for this file. It is not editable and would be omitted from the specification; remove it to export."))
        : h("span", { class: "chip ok" }, `${fmtInt(description.n_rows)} rows × ${fmtInt(description.n_cols)} columns`);
    } else if (description.delimiter_unresolved) {
      const candidates = description.delimiter_unresolved.candidates || [];
      status = h("div", {},
        h("span", { class: "chip warn" }, "delimiter unresolved"),
        h("div", { class: "msg warn" }, `${description.refusal || "The owner could not decide the delimiter."} The file is not read until you choose one.`),
        select([["", "choose a delimiter…"], ...candidates.map((c) => [c, c === "\t" ? "tab" : c])], "", (value) => value && setParsing(file.name, "delimiter", value), { disabled: isBusy() }));
    } else {
      status = h("div", {}, h("span", { class: "chip err" }, "refused by the owner"), h("div", { class: "msg err" }, description.refusal || "unreadable"));
    }
    rows.push(h("tr", {},
      h("td", {}, file.name),
      h("td", {}, fmtBytes(file.bytes.byteLength)),
      h("td", {}, status),
      h("td", {}, h("button", { class: "small", type: "button", disabled: isBusy(), onclick: () => removeFile(file.name) }, "Remove"))));
  }
  host.replaceChildren(h("table", { class: "files" },
    h("thead", {}, h("tr", {}, h("th", {}, "File"), h("th", {}, "Size"), h("th", {}, "Owner inspection"), h("th", {}, ""))),
    h("tbody", {}, rows)));
}

function datasetSourceFor(file, withColumns) {
  const description = state.descriptions.get(file.name);
  const edit = withColumns ? editFor(file.name) : null;
  const warnings = description.warnings || [];
  const columns = edit
    ? description.columns.filter((column) => !edit.pinned.has(column.name)).map((column) => ({
      id: column.name,
      name: column.name,
      previewValue: column.preview,
      detectedType: wizardType(column),
      assignedRole: edit.roles.get(column.name) ?? "ignored",
      manual: edit.manual.has(column.name),
      selected: edit.selected.has(column.name),
    }))
    : [];
  return {
    id: file.name,
    name: file.name,
    kind: "file",
    fileType: extOf(file.name) || "table",
    signalType: "table",
    status: warnings.length ? "warning" : "parsed",
    rowCount: description.n_rows ?? undefined,
    columnCount: description.n_cols ?? undefined,
    sizeBytes: file.bytes.byteLength,
    parsing: {
      separator: description.params?.delimiter,
      decimal: description.params?.decimal_separator,
      headerMode: description.params?.has_header === false ? "none" : "horizontal",
    },
    usage: {},
    columns,
    notes: warnings,
  };
}

function renderColumns() {
  const host = $("#columns-host");
  const unassigned = new Set(unassignedFiles().map((file) => file.name));
  const readable = readableFiles().filter((file) => !unassigned.has(file.name));
  if (!readable.length) {
    columnsHandle?.unmount();
    columnsHandle = null;
    host.replaceChildren();
    $("#source-pick").replaceChildren();
    $("#column-notes").replaceChildren();
    $("#source-controls").replaceChildren();
    return;
  }
  if (!readable.some((file) => file.name === state.ui.activeFile)) state.ui.activeFile = readable[0].name;
  const active = readable.find((file) => file.name === state.ui.activeFile);
  // The shared source card ships its own source switch hidden (`display: none` in datasetBuilder.css),
  // so the host provides the visible control for choosing which table's columns to edit.
  $("#source-pick").replaceChildren(...(readable.length > 1
    ? [field("Table to edit", select(readable.map((file) => [file.name, file.name]), active.name, (name) => {
      state.ui.activeFile = name;
      renderColumns();
    }, { "aria-label": "Table to edit" }))]
    : []));
  const edit = editFor(active.name);
  const description = state.descriptions.get(active.name);
  const source = datasetSourceFor(active, true);
  const filterOk = (column) => state.ui.filter === "all" || (state.ui.filter === "unassigned" ? column.assignedRole === "ignored" : column.assignedRole === state.ui.filter);
  const visible = source.columns.filter(filterOk);
  const props = {
    source,
    sources: readable.map((file) => (file.name === active.name ? source : datasetSourceFor(file, false))),
    onChangeSource: (id) => { state.ui.activeFile = id; renderColumns(); },
    roles: WIZARD_ROLES,
    selectedCount: edit.selected.size,
    onToggleColumn: (id, selected) => {
      if (selected === false) edit.selected.delete(id); else edit.selected.add(id);
      renderColumns();
    },
    onToggleAll: (selected) => {
      for (const column of visible) { if (selected) edit.selected.add(column.id); else edit.selected.delete(column.id); }
      renderColumns();
    },
    onAssignColumnRole: (id, role) => assignRoles(active.name, [id], role),
    onAssignRole: (role) => {
      const names = [...edit.selected];
      if (!names.length) { setStatus("Select columns first, then pick a role."); return; }
      edit.selected.clear();
      assignRoles(active.name, names, role);
    },
    filter: state.ui.filter,
    onFilterChange: (filter) => { state.ui.filter = filter; renderColumns(); },
  };
  if (columnsHandle) columnsHandle.update(props);
  else columnsHandle = mountColumnMapping(host, props);

  const notes = [];
  if (edit.unresolved.length) {
    notes.push(msg("warn", `The owner specification uses selectors this page does not resolve (${edit.unresolved.join("; ")}); the columns they cover show the owner's column guess instead.`));
  }
  if (edit.mismatch.length) {
    notes.push(msg("warn", `The owner specification names columns the inspection did not find: ${edit.mismatch.join(", ")}.`));
  }
  const guessedNames = [...edit.guessed];
  if (guessedNames.length && !edit.edited) {
    notes.push(msg("", `${guessedNames.length} column role(s) are owner guesses, not part of the owner's specification: ${guessedNames.slice(0, 12).join(", ")}${guessedNames.length > 12 ? "…" : ""}. They become explicit decisions in the specification as soon as you change a role in this table.`));
  }
  if (edit.pinned.size) {
    notes.push(msg("", `Kept exactly as the owner assigned them (not editable here): ${[...edit.pinned].map(([name, role]) => `${name} (${role})`).join(", ")}.`));
  }
  if (!sourceFor(state.result, active.name) && state.result) {
    notes.push(msg("warn", `${active.name} is not a single-file source of the owner specification (it may be stacked into another source); edits to it are not applied.`));
  }
  if (unassigned.size) {
    notes.push(msg("warn", `Read-only, not shown above: ${[...unassigned].join(", ")} (readable, but the owner proposed no source for it).`));
  }
  notes.push(msg("", "Assigning the ID role to a feature table sends an identity lock; on a joined table it sends the owner's id pairing. “wavelength” badges in the table are decoration. Partition, replicate and group columns are kept as metadata."));
  $("#column-notes").replaceChildren(...notes);

  renderSourceControls(active.name, description);
}

function naControl(label, current, onChange, disabled) {
  const policy = select([["", "owner default"], ...NA_POLICIES.map((p) => [p, p])], current.policy, (value) => onChange({ ...current, policy: value }), { disabled });
  const fill = current.policy === "replace"
    ? field("Fill value (number)", h("input", { type: "text", value: current.fill, disabled, onchange: (event) => onChange({ ...current, fill: event.target.value.trim() }) }))
    : null;
  return [field(label, policy), fill];
}

function renderSourceControls(file, description) {
  const choice = state.parsing.get(file) || {};
  const effective = description.params || {};
  const hasHeader = typeof choice.has_header === "boolean" ? String(choice.has_header) : "";
  const na = state.naBySource.get(file) || { policy: "", fill: "" };
  const controls = h("div", {},
    h("div", { class: "row-controls" },
      field("Delimiter", select(DELIMITERS, choice.delimiter ?? effective.delimiter ?? ";", (value) => setParsing(file, "delimiter", value), { disabled: isBusy() })),
      field("Decimal mark", select([[".", "."], [",", ","]], choice.decimal_separator ?? effective.decimal_separator ?? ".", (value) => setParsing(file, "decimal_separator", value), { disabled: isBusy() })),
      field("Header row", select([["", "Auto (undeclared)"], ["true", "first row is a header"], ["false", "no header"]], hasHeader, (value) => setParsing(file, "has_header", value === "" ? undefined : value === "true"), { disabled: isBusy() })),
      ...naControl(`Missing values in ${file}`, na, (next) => { state.naBySource.set(file, next); rebuild(); }, isBusy())),
    description.ignored_input_na_report
      ? msg("warn", `Owner input report (counts before role policies and joins, no positions): ${description.ignored_input_na_report.na_count} missing cell(s) in ${description.ignored_input_na_report.na_samples} row(s) and ${description.ignored_input_na_report.na_features} column(s).`)
      : null,
    ...(description.warnings || []).map((warning) => msg("warn", warning)));
  $("#source-controls").replaceChildren(controls);
}

function renderDeclarations() {
  const host = $("#declarations");
  if (!state.files.length) {
    host.replaceChildren(h("div", { class: "empty" }, "Declarations appear once files are loaded."));
    return;
  }
  const busy = isBusy();
  const declared = { partition: [], replicate: [], group: [] };
  for (const [file, edit] of state.edits) {
    if (!edit.edited) continue;
    for (const [column, role] of edit.roles) if (role in declared) declared[role].push({ file, column });
  }
  const parts = [];
  parts.push(h("div", { class: "row-controls" },
    field("Dataset name", h("input", { type: "text", value: state.name, placeholder: "owner default", disabled: busy, onchange: (event) => { state.name = event.target.value; rebuild(); } })),
    ...naControl("Missing values (dataset default)", state.naDefault, (next) => { state.naDefault = next; rebuild(); }, busy)));
  parts.push(msg("", "Missing values are handled by the owner per role: with abort, target columns and every column that is not metadata are checked, including columns you mark Ignored; metadata columns are exempt. The owner refuses cross-sample statistical fills (mean, median) and admits forward/backward fill within a sample; this page offers only the policies listed above."));

  if (declared.partition.length === 1) {
    const { file, column } = declared.partition[0];
    const values = state.descriptions.get(file)?.distinct?.[column] || [];
    parts.push(h("div", {},
      h("strong", {}, `Partition column '${column}'`),
      msg("", "Choose which values belong to each partition. Nothing is preselected; values are the owner's text rendering of the column."),
      ...PARTITION_CATEGORIES.map(([category, label]) => field(label, h("div", { class: "values" }, values.length
        ? values.map((value) => h("label", {}, h("input", {
          type: "checkbox", disabled: busy, checked: state.partition.values[category].includes(value),
          onchange: (event) => {
            const set = new Set(state.partition.values[category]);
            if (event.target.checked) set.add(value); else set.delete(value);
            state.partition.values[category] = [...set];
            rebuild();
          },
        }), value))
        : [h("span", { class: "empty" }, "No values listed yet.")]))),
      field("Rows with other values", select([["", "owner default"], ...UNKNOWN_POLICIES.map((p) => [p, p])], state.partition.unknown_policy, (value) => { state.partition.unknown_policy = value; rebuild(); }, { disabled: busy }))));
  } else if (declared.partition.length > 1) {
    parts.push(msg("err", "Several columns have the Partition role; only one is supported."));
  }
  if (declared.replicate.length === 1) {
    const { column } = declared.replicate[0];
    parts.push(h("div", { class: "row-controls" },
      field(`Replicate column '${column}' is`, select([
        ["", "choose…"],
        ["repetition_id", "the per-scan index inside a sample (sample_index.repetition_id)"],
        ["repetition", "the sample id shared by repeated measurements (repetition)"],
      ], state.replicate.mode, (value) => { state.replicate.mode = value; rebuild(); }, { disabled: busy })),
      state.replicate.mode === "repetition"
        ? field("Aggregate repeated measurements", select([["", "none"], ...AGGREGATE_METHODS.map((m) => [m, m])], state.replicate.method, (value) => { state.replicate.method = value; rebuild(); }, { disabled: busy }))
        : null));
  } else if (declared.replicate.length > 1) {
    parts.push(msg("err", "Several columns have the Replicate role; only one is supported."));
  }
  if (declared.group.length === 1) {
    parts.push(msg("", `Group column '${declared.group[0].column}' is written as sample_index.group_id: declared for downstream leakage groups, not consumed by IO materialization.`));
  } else if (declared.group.length > 1) {
    parts.push(msg("err", "Several columns have the Group role; only one is supported."));
  }
  host.replaceChildren(...parts);
}

function missingSummary(assembled) {
  const lines = [];
  const report = (partition, label, isMissing, total) => {
    const rows = [];
    for (let row = 0; row < total; row++) if (isMissing(row)) rows.push(row + 1);
    if (rows.length) lines.push({ partition, label, rows });
  };
  for (const [partition, block] of Object.entries(assembled.blocks || {})) {
    (block.x || []).forEach((matrix, index) => {
      report(partition, `X ${block.source_ids?.[index] || index + 1}`, (row) => {
        for (let col = 0; col < matrix.n_cols; col++) if (Number.isNaN(matrix.data[row * matrix.n_cols + col])) return true;
        return false;
      }, matrix.n_rows);
    });
    const width = (block.y_headers || []).length;
    (block.y_headers || []).forEach((header, col) => {
      report(partition, `y '${header}'`, (row) => Number.isNaN(block.y.data[row * width + col]), block.n_samples);
    });
    for (const column of block.metadata?.columns || []) {
      report(partition, `metadata '${column.name}'`, (row) => column.values[row] == null, column.values.length);
    }
  }
  return lines;
}

function blockView(part, block, assembled) {
  const sources = (block.x || []).map((matrix, index) => ({ id: block.source_ids?.[index] ?? `source ${index + 1}`, cols: Number(matrix.n_cols) || 0 }));
  const xCols = sources.reduce((sum, item) => sum + item.cols, 0);
  const rows = Number(block.n_samples);
  const yCols = block.y ? Number(block.y.columns.length) : null;
  const metaCols = block.metadata?.columns?.length || 0;
  const task = typeof assembled.task_type === "string" && assembled.task_type ? assembled.task_type : null;
  const stats = [
    { id: "samples", label: "Rows", value: fmtInt(rows), detail: "observations, not unique samples", tone: "default" },
    { id: "features", label: "X columns", value: fmtInt(xCols), detail: sources.length ? sources.map((item) => `${item.id}: ${fmtInt(item.cols)}`).join(" · ") : null, tone: "default" },
  ];
  if (yCols !== null) stats.push({ id: "targets", label: "y columns", value: fmtInt(yCols), detail: (block.y_headers || []).join(", ") || null, tone: "default" });
  const badges = [{ id: "partition", label: part, tone: "muted" }];
  if (task) badges.push({ id: "task", label: task, tone: "default" });
  return {
    id: `partition-${part}`,
    title: part,
    description: [yCols === null ? "no y" : null, metaCols ? `${fmtInt(metaCols)} metadata column${metaCols === 1 ? "" : "s"}` : null].filter(Boolean).join(" · ") || null,
    taskKind: "unknown",
    taskLabel: task || "Task not declared",
    sampleCount: Number.isFinite(rows) ? rows : null,
    sampleCountLabel: Number.isFinite(rows) ? `${fmtInt(rows)} rows` : null,
    featureCount: xCols,
    featureCountLabel: `${fmtInt(xCols)} columns`,
    targetCount: yCols,
    targetCountLabel: yCols === null ? null : `${fmtInt(yCols)} columns`,
    classCount: null,
    classCountLabel: null,
    spectralRangeLabel: null,
    splitCounts: [],
    splitSummaryLabel: null,
    tags: [],
    badges,
    stats,
  };
}

/** Identity and alignment provenance as the owner specification and the confirmed pairings state them. */
function provenanceLines() {
  if (!state.canonical || !state.result) return [];
  const spec = JSON.parse(state.canonical);
  const key = spec.sample_index?.by === "id" ? spec.sample_index.key : null;
  if (typeof key !== "string") return [];
  const carrying = state.files.filter((file) => state.descriptions.get(file.name)?.columns?.some((column) => column.name === key)).map((file) => file.name);
  const lines = [`Identity '${key}': a column of ${carrying.length ? carrying.join(", ") : "no loaded table"}.`];
  const pairings = confirmedPairings(state.result);
  const withoutKey = (spec.sources || []).map(inputsOf).flat().filter((name) => !carrying.includes(name));
  if (withoutKey.length) {
    lines.push(`Tables without '${key}' (${withoutKey.join(", ")}) are aligned only through the confirmed pairings: ${pairings.length ? pairings.map((proposal) => `${proposal.target} (${proposal.value?.mode || valueLabel(proposal)})`).join("; ") : "none"}.`);
  }
  return lines;
}

function renderPreview() {
  const notes = [];
  for (const issue of state.issues) notes.push(msg(issue.level === "error" ? "err" : "warn", issue.text));
  if (state.specError) notes.push(msg("err", `The owner rejected the specification: ${state.specError}`));
  if (state.assembleError) notes.push(msg("warn", `Not materializable yet: ${state.assembleError}`));
  if (state.previewKeptMissing) {
    const chosen = state.canonical ? JSON.parse(state.canonical).params?.na?.policy || "owner default" : "owner default";
    notes.push(msg("warn", `Preview materialized with missing values kept (ignore). The exported specification keeps the policy you chose (${chosen}), which the owner refuses on this data: ${state.previewKeptMissing}`));
  }
  const assembled = state.assembled;
  if (assembled) {
    for (const warning of assembled.warnings || []) notes.push(msg("warn", String(warning)));
    for (const audit of assembled.audits || []) {
      if (audit?.na) notes.push(msg("warn", `Owner NA report for source ${audit.source}: ${audit.na.na_count} missing cell(s), ${audit.na.na_samples} row(s), ${audit.na.na_features} column(s) (counts only).`));
      else notes.push(msg("", `Owner audit: ${JSON.stringify(audit)}`));
    }
    if (state.previewKeptMissing) {
      for (const line of missingSummary(assembled)) {
        notes.push(msg("", `${line.partition} · ${line.label}: missing at row ${line.rows.slice(0, 10).join(", ")}${line.rows.length > 10 ? ` … (${line.rows.length} rows)` : ""} (position within the materialized partition)`));
      }
    }
  }
  for (const line of provenanceLines()) notes.push(msg("", line));
  $("#preview-notes").replaceChildren(...notes);

  const parts = assembled ? Object.keys(assembled.blocks || {}) : [];
  for (const [part, mount] of [...previewMounts]) {
    if (!parts.includes(part)) { mount.handle.unmount(); mount.el.remove(); previewMounts.delete(part); }
  }
  const ui = window.Nirs4allUIBrowser;
  for (const part of parts) {
    const view = blockView(part, assembled.blocks[part], assembled);
    const existing = previewMounts.get(part);
    if (existing) { existing.handle.update({ view }); continue; }
    const el = h("div", { dataset: { partition: part } });
    $("#preview-cards").append(el);
    previewMounts.set(part, { el, handle: ui.mountDatasetPreview(el, { view }) });
  }
}

/**
 * L3: every declared source `key` must be a column of each named input, per the current describeTables facts
 * (read with the parameters of the exported specification). Missing or stale facts are an explicit refusal;
 * the specification is never rewritten and nothing is scanned again.
 */
function keyFactViolations() {
  if (!state.canonical) return [];
  const spec = JSON.parse(state.canonical);
  const out = [];
  for (const source of spec.sources || []) {
    if (source.key === undefined || source.key === null) continue;
    const declared = Array.isArray(source.key) ? source.key : [source.key];
    if (!declared.length || declared.some((name) => typeof name !== "string")) {
      out.push(`Source '${source.id}' declares a key (${JSON.stringify(source.key)}) whose column names cannot be checked against the inspected tables: export refused.`);
      continue;
    }
    const inputs = inputsOf(source);
    if (!inputs.length) {
      out.push(`Source '${source.id}' declares key '${declared.join("', '")}' but names no table input whose columns are known: export refused.`);
      continue;
    }
    const effective = { ...(spec.params || {}), ...(source.params || {}) };
    for (const input of inputs) {
      const description = state.descriptions.get(input);
      if (!description?.readable) {
        out.push(`Source '${source.id}' declares key '${declared.join("', '")}' but the columns of '${input}' are unavailable: export refused.`);
        continue;
      }
      const stale = ["delimiter", "decimal_separator", "has_header"].filter((name) => effective[name] !== undefined
        && description.params?.[name] !== undefined && effective[name] !== description.params[name]);
      if (stale.length) {
        out.push(`Source '${source.id}' declares key '${declared.join("', '")}' but '${input}' was inspected with different parsing parameters (${stale.join(", ")}) than the specification uses: export refused.`);
        continue;
      }
      const names = (description.columns || []).map((column) => column.name);
      const missing = declared.filter((name) => !names.includes(name));
      if (missing.length) {
        out.push(`Source '${source.id}' declares key '${missing.join("', '")}' but '${input}' has no such column with the final parsing parameters.`);
      }
    }
  }
  return out;
}

function blockers() {
  const list = [];
  if (!state.files.length) list.push("Add at least one table file.");
  for (const file of state.files) {
    const description = state.descriptions.get(file.name);
    if (description && !description.readable) list.push(`${file.name} cannot be read by the owner: fix or remove it.`);
    if (description?.readable && description.header_ambiguous
      && typeof state.parsing.get(file.name)?.has_header !== "boolean") {
      list.push(`${file.name}: the header row is ambiguous. Choose first row is a header or no header before export.`);
    }
  }
  for (const file of unassignedFiles()) {
    list.push(`${file.name} is readable, but the owner proposed no source for it, so it would be omitted from the specification: remove it.`);
  }
  const openDecisions = (state.result?.proposals || []).filter((proposal) => proposal.status === "proposed"
    && (proposal.kind === "identity" || proposal.kind === "pairing"));
  if (openDecisions.length) {
    list.push(`Confirm every identity and pairing decision before export (${openDecisions.length} open: ${openDecisions.map((proposal) => `${proposal.kind} ${proposal.target}`).join(", ")}).`);
  }
  if (state.builderError) list.push(`Owner error: ${state.builderError}`);
  for (const issue of state.issues) if (issue.level === "error") list.push(issue.text);
  list.push(...keyFactViolations());
  if (state.specError) list.push("The owner rejected the specification.");
  if (!state.canonical && !state.specError && state.files.length && !state.builderError && !isBusy()) list.push("No owner specification yet.");
  if (state.assembleError) list.push("The owner could not materialize the dataset.");
  if (isBusy()) list.push("The owner is working.");
  return list;
}

function renderExport() {
  const list = blockers();
  $("#export").disabled = list.length > 0;
  $("#export-notes").replaceChildren(...list.map((text) => msg("warn", text)), ...(list.length ? [] : provenanceLines().map((line) => msg("", line))));
}

function renderAll() {
  renderFiles();
  renderProposals();
  renderColumns();
  renderDeclarations();
  renderPreview();
  renderExport();
  if (state.builderError) setStatus(`Inference failed: ${state.builderError}`);
  else if (isBusy()) setStatus("The owner is working…");
  else if (state.result) {
    const open = (state.result.proposals || []).filter((proposal) => proposal.status === "proposed").length;
    const structure = state.result.plan?.structure?.value || "unknown";
    setStatus(open ? `${open} decision${open > 1 ? "s" : ""} to validate. Structure: ${structure}.` : `All decisions confirmed. Structure: ${structure}.`);
  }
}

// --------------------------------------------------------------------------- //
// Export and events                                                           //
// --------------------------------------------------------------------------- //

function showExport() {
  if (blockers().length || !state.canonical) return;
  $("#export-json").textContent = JSON.stringify(JSON.parse(state.canonical), null, 2);
  $("#export-overlay").hidden = false;
}

function downloadExport() {
  if (blockers().length || !state.canonical) return;
  const name = JSON.parse(state.canonical).name;
  const url = URL.createObjectURL(new Blob([state.canonical], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `${safeFileStem(name)}.nirs4all.json` });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

$("#add-files").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", async (event) => {
  await addFiles(event.target.files);
  event.target.value = "";
});
$("#load-samples").addEventListener("click", loadSamples);
$("#reset").addEventListener("click", reset);
$("#export").addEventListener("click", showExport);
$("#export-close").addEventListener("click", () => { $("#export-overlay").hidden = true; });
$("#export-download").addEventListener("click", downloadExport);

["dragover", "dragenter"].forEach((type) => document.body.addEventListener(type, (event) => {
  event.preventDefault();
  document.body.classList.add("dragging");
}));
["dragleave", "drop"].forEach((type) => document.body.addEventListener(type, (event) => {
  event.preventDefault();
  if (type === "drop" && event.dataTransfer?.files?.length) addFiles(event.dataTransfer.files);
  document.body.classList.remove("dragging");
}));

// Test and debugging seam, read-only: the complete owner results and retained bytes.
window.NIRS4ALL_IO_BUILDER_STATE = state;

renderAll();
runtimeReady.then((ok) => {
  if (ok && location.hash.includes("demo")) loadSamples();
});
