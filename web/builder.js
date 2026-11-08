// Iterative dataset builder (mode === "builder").
//
// app.js stays the decode / inference / viz / spec-form engine and exposes a
// small surface on window.NIRS4ALL_APP. This module drives the propose cycle:
// add files -> io.proposeDataset(files, records, {confirmed}) -> render the open
// decisions as confirmable cards -> on accept/override, push a lock and re-infer
// -> materialize a preview with io.assembleDataset. The one-shot view is untouched.

const APP = () => window.NIRS4ALL_APP;
const KIND_ORDER = ["structure", "role", "partition", "pairing", "identity", "signal_type", "task_type", "folds"];
const KIND_LABEL = {
  structure: "Structure",
  role: "Role",
  partition: "Partition",
  pairing: "Pairing",
  identity: "Identity",
  signal_type: "Signal",
  task_type: "Task",
  folds: "Folds",
};
const OVERRIDE_VOCAB = {
  role: ["features", "targets", "metadata", "weights", "ignore"],
  partition: ["train", "test", "val", "predict"],
  signal_type: ["absorbance", "reflectance", "reflectance%", "transmittance", "transmittance%", "kubelka-munk"],
  task_type: ["regression", "binary", "multiclass"],
};

let lastFileKey = "";

// Shared nirs4all-ui islands (local IIFE `Nirs4allUIBrowser`). Each island lives in
// a dedicated, empty host div that is never written with innerHTML; handles are kept
// here and unmounted before their host is cleared.
let pending = 0; // owner cycles in flight; list actions are ignored while > 0
let sentProposals = null; // the s.proposals array last projected into the list
let proposalUi = null; // { host, mount, echoes, handle }
let previewUi = null; // { host, msg, cards, foot, mounts: Map<partition, { el, handle }> }

function state() {
  return APP().state;
}

function esc(s) {
  return APP().esc(s);
}

function sharedUI() {
  const ui = window.Nirs4allUIBrowser;
  if (!ui || typeof ui.mountOwnerProposals !== "function" || typeof ui.mountDatasetPreview !== "function") {
    throw new Error("The shared nirs4all-ui browser bundle is not loaded.");
  }
  return ui;
}

// --------------------------------------------------------------------------- //
// Propose cycle                                                               //
// --------------------------------------------------------------------------- //

// Public entry. Counts in-flight cycles so the proposal list shows (and the
// callbacks enforce) a busy state; overlapping calls are all still executed.
async function runProposeCycle() {
  pending += 1;
  if (state().mode === "builder") renderProposals();
  try {
    await proposeCycle();
  } finally {
    pending -= 1;
    if (state().mode === "builder") renderProposals();
  }
}

async function proposeCycle() {
  const app = APP();
  const s = state();
  s.builderError = null;
  const io = app.getIo();
  const { files, recordSets } = app.usableFilesForIo();

  if (!io?.proposeDataset || (!files.length && !recordSets.length)) {
    s.plan = null;
    s.proposals = [];
    s.confirmedEchoes = [];
    s.workingSpec = null;
    s.spec = null;
    s.assembled = null;
    app.setStatus(files.length || recordSets.length ? "Add a dataset to start the builder." : "Waiting for files.");
    app.renderAll();
    return;
  }

  let res;
  try {
    res = io.proposeDataset(files, recordSets, { confirmed: s.confirmed });
  } catch (err) {
    // Clear all derived state so a failed cycle never leaves a stale spec/preview.
    s.builderError = String(err?.message || err);
    s.plan = null;
    s.proposals = [];
    s.confirmedEchoes = [];
    s.workingSpec = null;
    s.spec = null;
    s.assembled = null;
    s.assembleError = null;
    s.validation = { ok: false, message: s.builderError };
    app.setStatus(`Inference failed: ${s.builderError}`);
    app.renderAll();
    return;
  }

  const proposals = Array.isArray(res.proposals) ? res.proposals : [];
  s.plan = res.plan || null;
  s.proposals = proposals.filter((p) => p.status === "proposed");
  s.confirmedEchoes = proposals.filter((p) => p.status !== "proposed");
  s.workingSpec = res.spec || app.defaultSpec();
  s.spec = s.workingSpec;
  s.validation = res.valid
    ? { ok: true, message: "spec is valid" }
    : { ok: false, message: (res.validation_errors || []).join("; ") || "spec is incomplete" };

  app.chooseDefaultSignal();
  app.chooseDefaultTarget();
  const view = app.datasetView();
  if (!s.userVizSelected && (view.targetValues.length || view.targetPreviewValues.length)) s.activeViz = "targets";

  pushStep();
  await materializePreview();
  app.setStatus(builderStatus());
  app.renderAll();
}

function pushStep() {
  const s = state();
  const names = s.files.map((f) => f.name);
  const key = names.join("|");
  const openCount = s.proposals.length;
  const confirmedCount = s.confirmedEchoes.length;
  const structure = s.plan?.structure?.value || "unknown";
  if (key === lastFileKey && s.steps.length) {
    const step = s.steps[s.steps.length - 1];
    step.openCount = openCount;
    step.confirmedCount = confirmedCount;
    step.structure = structure;
    return;
  }
  const prev = s.steps.length ? s.steps[s.steps.length - 1].names : [];
  const added = names.filter((n) => !prev.includes(n));
  s.steps.push({
    n: s.steps.length + 1,
    names,
    added: added.length ? added : names,
    openCount,
    confirmedCount,
    structure,
  });
  lastFileKey = key;
}

async function materializePreview() {
  const app = APP();
  const s = state();
  s.assembled = null;
  s.assembleError = null;
  const io = app.getIo();
  if (!io?.assembleDataset || !s.workingSpec) return;
  if (s.validation && !s.validation.ok) {
    s.assembleError = s.validation.message;
    return;
  }
  try {
    // proposeDataset discovers sources from the file list (by name), so it needs
    // every file; assembleDataset instead resolves each spec `input` against the
    // union of files + record sets and rejects a name present in both, so each
    // input is sent once (vendor records, or table bytes).
    const { files, recordSets } = app.assemblyInputsForIo();
    s.assembled = app.assemblePreview(files, recordSets, s.workingSpec);
  } catch (err) {
    s.assembleError = String(err?.message || err);
  }
}

function builderStatus() {
  const s = state();
  const open = s.proposals.length;
  const structure = s.plan?.structure?.value || "unknown";
  if (s.builderError) return `Inference failed: ${s.builderError}`;
  if (!open) return `All decisions confirmed. Structure: ${structure}.`;
  return `${open} decision${open > 1 ? "s" : ""} to validate. Structure: ${structure}.`;
}

// --------------------------------------------------------------------------- //
// Confirm / override                                                          //
// --------------------------------------------------------------------------- //

function upsertConfirmed(lock) {
  const s = state();
  const idx = s.confirmed.findIndex((l) => l.kind === lock.kind && l.target === lock.target);
  if (idx >= 0) s.confirmed[idx] = lock;
  else s.confirmed.push(lock);
}

function removeConfirmed(kind, target) {
  const s = state();
  s.confirmed = s.confirmed.filter((l) => !(l.kind === kind && l.target === target));
}

function acceptAll() {
  const s = state();
  // Only the proposals on screen, and never while inputs or the owner are mid-run.
  if (isBusy() || sentProposals !== s.proposals) return;
  for (const p of s.proposals) {
    upsertConfirmed({ kind: p.kind, target: p.target, value: p.value, status: "confirmed" });
  }
  runProposeCycle();
}

// --------------------------------------------------------------------------- //
// Rendering                                                                   //
// --------------------------------------------------------------------------- //

function render() {
  if (state().mode !== "builder") return;
  renderProposals();
  renderSteps();
  renderDatasetSoFar();
  renderPreview();
}

function scoreClass(score) {
  if (score >= 0.75) return "ok";
  if (score >= 0.45) return "warn";
  return "";
}

function valueLabel(p) {
  const v = p.value;
  if (v && typeof v === "object") {
    if (v.mode === "join_id") return `join by id (${v.on || v.right_on || "shared key"})`;
    if (v.mode === "row_order") return "align by row order";
    if (v.mode === "concat_samples") return `stack ${(v.parts || []).length} files → ${v.whole || "y"}`;
    if (v.by) return v.key ? `by ${v.by} · ${v.key}` : `by ${v.by}`;
    return JSON.stringify(v);
  }
  return String(v);
}

// Known kinds first (KIND_ORDER), unknown kinds appended in the owner's order. The
// sort is stable, so the owner's order is kept within a kind. Nothing is dropped.
function orderByKind(proposals) {
  const rank = (kind) => {
    const i = KIND_ORDER.indexOf(kind);
    return i < 0 ? KIND_ORDER.length : i;
  };
  return [...proposals].sort((a, b) => rank(a.kind) - rank(b.kind));
}

// Presentation-only projection of one owner proposal. Alternatives are identified by
// their index in the owner's list; override choices carry the host vocabulary string.
// The original owner values never travel through the UI: callbacks look them up again
// in the latest s.proposals (see currentProposal).
function projectProposal(p) {
  const display = {
    id: p.id,
    kind: p.kind,
    target: p.target,
    valueLabel: valueLabel(p),
    kindLabel: KIND_LABEL[p.kind] || p.kind,
    score: Number.isFinite(p.score) ? p.score : null,
    scoreClassName: `pill ${scoreClass(p.score)}`.trim(),
    ambiguous: Boolean(p.ambiguous),
    alternatives: (p.alternatives || []).map((alt, index) => ({
      choiceId: String(index),
      label: valueLabel({ value: alt.value }),
      score: Number.isFinite(alt.score) ? alt.score : null,
    })),
  };
  if (Array.isArray(p.evidence) && p.evidence.length) display.evidence = p.evidence.map(String);
  const vocab = OVERRIDE_VOCAB[p.kind];
  if (vocab) display.overrideChoices = vocab.map((opt) => ({ choiceId: opt, label: opt, value: opt }));
  return display;
}

// The owner proposal a list action refers to, or null when the action must be
// ignored: an owner cycle is running, or the list on screen no longer shows the
// latest s.proposals array (sources changed since it was projected).
function currentProposal(id) {
  const s = state();
  if (isBusy() || sentProposals !== s.proposals) return null;
  return s.proposals.find((x) => x.id === id) || null;
}

// Decision actions are refused while an owner cycle runs or while the bound input files
// are changing (state.inputBusy, set by app.js before any byte is read).
function isBusy() {
  return pending > 0 || state().inputBusy > 0;
}

function onConfirm(id) {
  const p = currentProposal(id);
  if (!p) return;
  upsertConfirmed({ kind: p.kind, target: p.target, value: p.value, status: "confirmed" });
  runProposeCycle();
}

function onAlternative(id, choiceId) {
  const p = currentProposal(id);
  const index = Number(choiceId);
  const alt = p && Number.isInteger(index) && String(index) === choiceId ? (p.alternatives || [])[index] : null;
  if (!alt) return;
  upsertConfirmed({ kind: p.kind, target: p.target, value: alt.value, status: "overridden" });
  runProposeCycle();
}

function onOverride(id, value) {
  const p = currentProposal(id);
  if (!p || !(OVERRIDE_VOCAB[p.kind] || []).includes(value)) return;
  upsertConfirmed({ kind: p.kind, target: p.target, value, status: "overridden" });
  runProposeCycle();
}

function onEchoClick(event) {
  const el = event.target.closest('[data-action="undo"]');
  if (!el || isBusy() || sentProposals !== state().proposals) return;
  removeConfirmed(el.dataset.kind, el.dataset.target);
  runProposeCycle();
}

function ensureProposalDom(host) {
  if (proposalUi && proposalUi.host === host) return proposalUi;
  const mount = document.createElement("div");
  mount.dataset.mount = "owner-proposals";
  const echoes = document.createElement("div");
  echoes.dataset.role = "confirmed-echoes";
  echoes.addEventListener("click", onEchoClick);
  host.replaceChildren(mount, echoes);
  proposalUi = { host, mount, echoes, handle: null };
  return proposalUi;
}

function renderProposals() {
  const host = document.querySelector("#proposalGroups");
  const scoreBadge = document.querySelector("#builderScore");
  const s = state();
  if (scoreBadge) {
    const overall = s.plan?.overall_score;
    scoreBadge.textContent = Number.isFinite(overall) ? overall.toFixed(2) : "-";
    scoreBadge.className = `pill ${scoreClass(overall || 0)}`;
  }
  if (!host) return;

  const ui = ensureProposalDom(host);
  const props = {
    proposals: orderByKind(s.proposals).map(projectProposal),
    busy: isBusy(),
    error: s.builderError || null,
    emptyMessage: s.builderError
      ? ""
      : s.inputBusy > 0
        ? "Input files are changing: decisions are proposed again once decoding has finished."
        : s.confirmedEchoes.length
        ? "All decisions confirmed — review the materialized preview below."
        : "Add files to see decisions to validate.",
    onConfirm,
    onAlternative,
    onOverride,
  };
  sentProposals = s.proposals;
  const acceptAllButton = document.querySelector("#acceptAll");
  if (acceptAllButton) acceptAllButton.disabled = props.busy;
  if (ui.handle) ui.handle.update(props);
  else ui.handle = sharedUI().mountOwnerProposals(ui.mount, props);

  // Confirmed echoes (with their undo links) stay host DOM, outside the island.
  if (!s.confirmedEchoes.length) {
    ui.echoes.replaceChildren();
    return;
  }
  ui.echoes.innerHTML = `<details class="confirmed-list"><summary>Confirmed (${s.confirmedEchoes.length})</summary>
      ${s.confirmedEchoes
        .map(
          (p) =>
            `<div class="confirmed-row"><span>${esc(KIND_LABEL[p.kind] || p.kind)} · ${esc(p.target)} → ${esc(
              valueLabel(p)
            )}</span><button class="link" type="button" data-action="undo" data-kind="${esc(p.kind)}" data-target="${esc(
              p.target
            )}">undo</button></div>`
        )
        .join("")}
    </details>`;
}

function renderSteps() {
  const host = document.querySelector("#stepList");
  if (!host) return;
  const s = state();
  if (!s.steps.length) {
    host.innerHTML = `<div class="empty">Each file you add becomes a step.</div>`;
    return;
  }
  host.innerHTML = s.steps
    .map((step) => {
      const done = step.openCount === 0;
      return `<div class="step-row ${done ? "done" : ""}">
        <span class="step">${step.n}</span>
        <div class="step-body">
          <div class="step-files">+ ${esc(step.added.join(", "))}</div>
          <div class="step-meta">${esc(step.structure)} · ${step.openCount} open · ${step.confirmedCount} confirmed</div>
        </div>
        <span class="pill ${done ? "ok" : "warn"}">${done ? "✓" : step.openCount}</span>
      </div>`;
    })
    .join("");
}

function renderDatasetSoFar() {
  const host = document.querySelector("#datasetSoFar");
  if (!host) return;
  const spec = state().workingSpec;
  const sources = spec?.sources || [];
  if (!sources.length) {
    host.innerHTML = `<div class="empty">No sources yet.</div>`;
    return;
  }
  const rows = sources
    .map((src) => {
      const role = src.role || "features";
      const part = src.partition ? `<span class="map-chip">${esc(src.partition)}</span>` : "";
      const join = src.join ? `<span class="map-chip join">⋈ ${esc(src.join.to || src.join.right || "")}</span>` : "";
      const input = Array.isArray(src.input) ? src.input.join(" + ") : src.input || src.id;
      return `<div class="sofar-row">
        <span class="role-tag role-${esc(role)}">${esc(role)}</span>
        <span class="sofar-name">${esc(input)}</span>${part}${join}
      </div>`;
    })
    .join("");
  const split = spec.partitions?.by ? `<div class="sofar-foot">split: ${esc(spec.partitions.by)}</div>` : "";
  const folds = spec.folds ? `<div class="sofar-foot">folds: yes</div>` : "";
  host.innerHTML = rows + split + folds;
}

// Card view for one assembled partition block, passed as `view` only (the card's
// `dataset` input would word counts as "samples"). `n_samples` is the owner's row
// count (observations); per-source `n_cols` come straight from the owner matrices.
// Nothing here decodes feature_axes or alters the owner result.
function blockView(part, block, assembled) {
  const num = (n) => APP().fmtInt(Number(n));
  const sources = (block.x || []).map((matrix, index) => ({
    id: block.source_ids?.[index] ?? `source ${index + 1}`,
    cols: Number(matrix.n_cols) || 0,
  }));
  const xCols = sources.reduce((sum, source) => sum + source.cols, 0);
  const rows = Number(block.n_samples);
  const yCols = block.y ? Number(block.y.columns.length) : null;
  const metaCols = block.metadata?.columns?.length || 0;
  const task = typeof assembled.task_type === "string" && assembled.task_type ? assembled.task_type : null;
  const stats = [
    { id: "samples", label: "Rows", value: num(rows), detail: "observations, not unique samples", tone: "default" },
    {
      id: "features",
      label: "X columns",
      value: num(xCols),
      detail: sources.length ? sources.map((source) => `${source.id}: ${num(source.cols)}`).join(" · ") : null,
      tone: "default",
    },
  ];
  if (yCols !== null) {
    stats.push({ id: "targets", label: "y columns", value: num(yCols), detail: (block.y_headers || []).join(", ") || null, tone: "default" });
  }
  const badges = [{ id: "partition", label: part, tone: "muted" }];
  if (task) badges.push({ id: "task", label: task, tone: "default" });
  return {
    id: `partition-${part}`,
    title: part,
    description: [yCols === null ? "no y" : null, metaCols ? `${num(metaCols)} metadata column${metaCols === 1 ? "" : "s"}` : null].filter(Boolean).join(" · ") || null,
    taskKind: "unknown",
    taskLabel: task || "Task not declared",
    sampleCount: Number.isFinite(rows) ? rows : null,
    sampleCountLabel: Number.isFinite(rows) ? `${num(rows)} rows` : null,
    featureCount: xCols,
    featureCountLabel: `${num(xCols)} columns`,
    targetCount: yCols,
    targetCountLabel: yCols === null ? null : `${num(yCols)} columns`,
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

function ensurePreviewDom(host) {
  if (previewUi && previewUi.host === host) return previewUi;
  const msg = document.createElement("div");
  msg.style.gridColumn = "1 / -1";
  const cards = document.createElement("div"); // holds one dedicated mount div per partition
  cards.style.display = "contents";
  const foot = document.createElement("div"); // host-rendered footers
  foot.style.display = "contents";
  host.replaceChildren(msg, cards, foot);
  previewUi = { host, msg, cards, foot, mounts: new Map() };
  return previewUi;
}

function releasePreviewCards(ui) {
  for (const { el, handle } of ui.mounts.values()) {
    handle.unmount();
    el.remove();
  }
  ui.mounts.clear();
}

function renderPreview() {
  const host = document.querySelector("#materializePreview");
  if (!host) return;
  const s = state();
  const ui = ensurePreviewDom(host);
  const message = (cls, text) => {
    releasePreviewCards(ui);
    ui.foot.replaceChildren();
    ui.msg.className = cls;
    ui.msg.textContent = text;
    ui.msg.hidden = false;
  };
  if (s.assembleError) return message("empty warn", `Not materializable yet: ${s.assembleError}`);
  const a = s.assembled;
  if (!a || !a.blocks) return message("empty", "Confirm the open decisions to preview the assembled dataset.");

  ui.msg.hidden = true;
  const parts = Object.keys(a.blocks);
  // Drop cards of partitions that disappeared (unmount before removing the host div).
  for (const [part, mount] of [...ui.mounts]) {
    if (parts.includes(part)) continue;
    mount.handle.unmount();
    mount.el.remove();
    ui.mounts.delete(part);
  }
  for (const part of parts) {
    const props = { view: blockView(part, a.blocks[part], a) };
    const mount = ui.mounts.get(part);
    if (mount) {
      mount.handle.update(props);
    } else {
      const el = document.createElement("div");
      el.dataset.partition = part;
      ui.cards.appendChild(el);
      ui.mounts.set(part, { el, handle: sharedUI().mountDatasetPreview(el, props) });
    }
    ui.cards.appendChild(ui.mounts.get(part).el); // keep owner partition order
  }

  const folds = (a.folds || []).length ? `<div class="preview-foot">${a.folds.length} fold(s)</div>` : "";
  const missing = APP().missingValueSummary(a).map((item) =>
    `<div class="preview-foot">${esc(`${item.partition} · ${item.label}: ${item.rows.length} of ${item.total} samples missing (${APP().missingRowsLabel(item.rows)})`)}</div>`).join("");
  const refused = a.naAbort ? `<div class="preview-foot">Missing values are kept for the preview; the current NA policy refuses them when loading.</div>` : "";
  ui.foot.innerHTML = folds + refused + missing;
}

// --------------------------------------------------------------------------- //
// Events                                                                      //
// --------------------------------------------------------------------------- //

// List actions (confirm / alternative / override) arrive as callbacks of the shared
// proposal list; the confirmed-echo undo link has its own listener on its own host
// div (see ensureProposalDom), so no delegated listener overlaps the island.
function wire() {
  document.querySelector("#acceptAll")?.addEventListener("click", acceptAll);
}

function resetBuilder() {
  lastFileKey = "";
  render();
}

function init() {
  window.NIRS4ALL_BUILDER = { runProposeCycle, render, reset: resetBuilder };
  wire();
  render();
  // If we loaded into builder mode with files already present (e.g. a future
  // default or a restored session), run the first propose cycle now.
  if (state().mode === "builder" && state().files.length) runProposeCycle();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
