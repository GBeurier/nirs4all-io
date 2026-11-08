// Served Playwright smoke for dataset-builder.html (owner-backed Dataset Builder).
//
// Serves web/ from an owned localhost server (closed at the end), blocks every non-localhost
// request (fonts, analytics) so the whole flow also runs offline, and drives the real page against
// the vendored nirs4all-io WASM (it needs `describeTables`) and the column-mapping host bundle
// (`node build-dataset-builder.mjs ...` must have produced vendor/ui-columns/).
//
// Usage: node web/tests/dataset-builder-smoke.mjs
// Env:   PLAYWRIGHT_MODULE (path or specifier), CHROME_BIN.
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const root = path.resolve(here, "..");
const chromeExecutable = process.env.CHROME_BIN || "/usr/bin/google-chrome";

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml", ".png": "image/png",
  ".ico": "image/x-icon", ".csv": "text/csv",
};

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    const target = path.resolve(root, `.${decodeURIComponent(url.pathname)}`);
    if (!target.startsWith(root + path.sep)) throw new Error("outside the web root");
    const body = await readFile(target);
    response.writeHead(200, { "content-type": MIME[path.extname(target)] || "application/octet-stream" });
    response.end(body);
  } catch (_) {
    response.writeHead(404);
    response.end("not found");
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const pageUrl = `http://127.0.0.1:${server.address().port}/dataset-builder.html`;

async function loadPlaywright() {
  const moduleName = process.env.PLAYWRIGHT_MODULE || "playwright";
  return import(moduleName.startsWith("/") ? pathToFileURL(moduleName).href : moduleName);
}

const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromeExecutable, headless: true });
const errors = [];
const external = [];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function newPage() {
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route((url) => !["127.0.0.1", "localhost"].includes(url.hostname), (route) => {
    external.push(route.request().url());
    return route.abort();
  });
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("#runtime")?.textContent.startsWith("io "), null, { timeout: 15000 });
  return page;
}

const stateOf = (page, fn, arg) => page.evaluate(fn, arg);
async function idle(page) {
  await page.waitForFunction(() => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    return s && s.pending === 0 && s.inputBusy === 0;
  }, null, { timeout: 30000 });
  await page.waitForTimeout(50);
  await page.waitForFunction(() => window.NIRS4ALL_IO_BUILDER_STATE.pending === 0, null, { timeout: 30000 });
}

async function loadSample(page) {
  await page.click("#load-samples");
  await idle(page);
}

function files(entries) {
  return entries.map(([name, text]) => ({ name, mimeType: "text/csv", buffer: Buffer.from(text) }));
}

try {
  await sampleFlow();
  await explicitHeaderlessFlow();
  await columnMappingFlow();
  await metadataIdentityFlow();
  await unconfirmedIdentityFlow();
  await keyColumnAbsentFlow();
  await naFlow();
  await refusalAndReplacementFlow();
  const real = errors.filter((text) => !/Failed to load resource|net::ERR_FAILED/.test(text));
  assert(!real.length, `console/page errors: ${real.join(" | ")}`);
  console.log(`dataset-builder smoke passed (${external.length} external request(s) blocked)`);
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}

async function sampleFlow() {
  const page = await newPage();
  await loadSample(page);
  const info = await stateOf(page, async () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const digest = async (bytes) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("");
    return {
      files: await Promise.all(s.files.map(async (f) => ({ name: f.name, sha: await digest(f.bytes) }))),
      columns: Object.fromEntries([...s.descriptions].map(([name, d]) => [name, d.columns.map((c) => c.name)])),
      readable: [...s.descriptions.values()].every((d) => d.readable),
      proposals: s.result?.proposals?.length ?? -1,
      canonical: s.canonical,
      error: s.builderError || s.specError,
    };
  });
  assert(info.files.length === 6, `expected six sample files, got ${info.files.length}`);
  for (const file of info.files) {
    const bytes = await readFile(path.join(root, "samples", file.name));
    assert(file.sha === createHash("sha256").update(bytes).digest("hex"), `${file.name}: retained bytes differ from the file`);
  }
  assert(info.readable && !info.error, `owner inspection/proposal failed: ${info.error}`);
  assert(info.columns["X_train.csv"].length === 12 && info.columns["X_train.csv"][0] === "1000", "X_train columns are not the owner's header order");
  assert(info.columns["metadata_train.csv"].join(",") === "sample_id,cultivar,replicate,site", "metadata_train columns differ");
  assert(info.proposals > 0, "no owner proposals");
  assert(info.canonical && JSON.parse(info.canonical).sources.length >= 1, "no canonical specification");
  assert(await page.locator("#proposals-host *").count() > 0, "the shared proposal list did not render");
  assert(await page.locator("#columns-host .dsb-mapping__table").count() === 1, "the column table did not render");

  // Confirm every open decision through the shared list, one at a time.
  for (let guard = 0; guard < 20; guard++) {
    const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
    if (!(await button.count())) break;
    await button.click();
    await idle(page);
  }
  const confirmed = await stateOf(page, () => window.NIRS4ALL_IO_BUILDER_STATE.proposalLocks.length);
  assert(confirmed > 0, "confirming decisions sent no locks");

  // Export equals the canonical owner text.
  await page.waitForFunction(() => !document.querySelector("#export").disabled, null, { timeout: 10000 }).catch(async () => {
    throw new Error(`export stayed disabled: ${await page.textContent("#export-notes")}`);
  });
  await page.click("#export");
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#export-download")]);
  const exported = await readFile(await download.path(), "utf8");
  const canonical = await stateOf(page, () => window.NIRS4ALL_IO_BUILDER_STATE.canonical);
  assert(exported === canonical, "downloaded specification differs from the owner's canonical text");
  assert(download.suggestedFilename().endsWith(".nirs4all.json"), "unexpected download name");
  assert(await page.locator("#preview-cards [data-partition]").count() >= 1, "no materialized partition card");
  await page.close();
}

async function columnMappingFlow() {
  const page = await newPage();
  // Every table carries sample_id, so a dataset identity can be declared. (The shipped sample aligns its
  // tables by row order and has no id column in X, so the owner infers no identity there.)
  await page.setInputFiles("#file-input", files([
    ["X_train.csv", "sample_id;1000;1010;1020\ns1;0.1;0.2;0.3\ns2;0.2;0.3;0.4\ns3;0.3;0.4;0.5\ns4;0.4;0.5;0.6\n"],
    ["y_train.csv", "sample_id;protein\ns1;1.0\ns2;2.0\ns3;3.0\ns4;4.0\n"],
    ["metadata_train.csv", "sample_id;cultivar;site\ns1;A;north\ns2;B;south\ns3;A;north\ns4;B;south\n"],
  ]));
  await idle(page);
  for (let guard = 0; guard < 20; guard++) {
    const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
    if (!(await button.count())) break;
    await button.click();
    await idle(page);
  }
  // The user's explicit ID decision, made in the column table of the feature table (never inferred for them):
  // an owner-inferred key alone is not a confirmed identity for the group declaration below.
  await page.selectOption('select[aria-label="Table to edit"]', "X_train.csv");
  await page.waitForSelector('select[aria-label="sample_id role"]');
  await page.selectOption('select[aria-label="sample_id role"]', "id");
  await idle(page);
  await page.selectOption('select[aria-label="Table to edit"]', "metadata_train.csv");
  await page.waitForSelector('select[aria-label="site role"]');
  await page.selectOption('select[aria-label="site role"]', "group");
  await idle(page);
  await page.selectOption('select[aria-label="cultivar role"]', "ignored");
  await idle(page);
  const after = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const spec = s.canonical ? JSON.parse(s.canonical) : null;
    const columns = Object.values(s.assembled?.blocks || {}).flatMap((b) => (b.metadata?.columns || []).map((c) => c.name));
    return { spec, columns, issues: s.issues.map((i) => i.text), error: s.specError,
      owner: { sample_index: s.result?.spec?.sample_index, sources: (s.result?.spec?.sources || []).map((x) => ({ id: x.id, input: x.input, role: x.role, key: x.key, join: x.join })),
        proposals: (s.result?.proposals || []).map((p) => `${p.kind}|${p.target}|${p.status}`), locks: s.usedLocks } };
  });
  assert(!after.error, `owner rejected the mapped specification: ${after.error}`);
  assert(after.spec?.sample_index?.by === "id" && after.spec.sample_index.key === "sample_id", `the confirmed ID decision is not in the owner specification: ${JSON.stringify(after.spec?.sample_index)} owner=${JSON.stringify(after.owner)}`);
  assert(after.owner.locks.some((l) => l.kind === "identity" && l.target === "dataset" && l.value === "sample_id"), `no confirmed identity lock was sent for sample_id: ${JSON.stringify(after.owner.locks)}`);
  assert(after.spec?.sample_index?.group_id === "site", `group_id was not written: ${JSON.stringify(after.spec?.sample_index)} ${after.issues.join(";")} owner=${JSON.stringify(after.owner)}`);
  assert(!after.columns.includes("cultivar"), "an ignored metadata column is still in the assembled metadata");
  await page.close();
}

// The delivered sample: tables are paired by row order and `sample_id` exists only in the metadata table.
// The owner infers no identity, so the user's explicit ID decision is the only source of one.
async function metadataIdentityFlow() {
  const page = await newPage();
  await loadSample(page);
  for (let guard = 0; guard < 20; guard++) {
    const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
    if (!(await button.count())) break;
    await button.click();
    await idle(page);
  }
  const snapshot = () => stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const spec = s.canonical ? JSON.parse(s.canonical) : null;
    const train = s.assembled?.blocks?.train;
    return {
      ownerSampleIndex: s.result?.spec?.sample_index ?? null, canonicalSampleIndex: spec?.sample_index ?? null,
      joins: (spec?.sources || []).map((x) => x.join ?? null),
      sources: (spec?.sources || []).map((x) => ({ id: x.id, role: x.role, key: x.key, join: x.join, columns: x.columns })),
      ownerSources: (s.result?.spec?.sources || []).map((x) => ({ id: x.id, role: x.role, key: x.key, join: x.join })),
      locks: s.usedLocks, identityLocks: s.usedLocks.filter((l) => l.kind === "identity"),
      specError: s.specError, assembleError: s.assembleError, issues: s.issues.map((i) => i.text),
      provenance: s.assembled?.identity?.provenance?.sample_id ?? null, rows: train?.n_samples ?? null,
      ids: train?.metadata?.columns?.find((c) => c.name === "sample_id")?.values ?? null,
      // R3: a source key must name a column that source's own table contains.
      keyed: (spec?.sources || []).filter((x) => x.key !== undefined).map((x) => ({ id: x.id, key: x.key,
        has: (s.descriptions.get(x.input)?.columns || []).some((c) => c.name === x.key) })),
      featureKeys: (spec?.sources || []).filter((x) => x.role === "features").map((x) => ({ id: x.id, key: x.key ?? null })),
    };
  });
  const before = await snapshot();
  assert(!before.ownerSampleIndex && !before.identityLocks.length, `an identity was inferred or sent before any user decision: ${JSON.stringify(before)}`);

  // The user's explicit decision, in the metadata table's column editor.
  await page.selectOption('select[aria-label="Table to edit"]', "metadata_train.csv");
  await page.waitForSelector('select[aria-label="sample_id role"]');
  await page.selectOption('select[aria-label="sample_id role"]', "id");
  await idle(page);
  const after = await snapshot();
  assert(!after.specError && !after.assembleError, `the owner refused the metadata-table identity: ${JSON.stringify({ ...after, before: before.sources })}`);
  assert(after.identityLocks.length === 1 && after.identityLocks[0].value === "sample_id", `no identity lock for sample_id was sent: ${JSON.stringify(after.identityLocks)} ${after.issues.join(";")}`);
  assert(after.canonicalSampleIndex?.by === "id" && after.canonicalSampleIndex.key === "sample_id", `identity is not in the owner specification: ${JSON.stringify(after.canonicalSampleIndex)} ${after.issues.join(";")}`);
  assert(after.provenance === "sample_id", "the owner assembly does not carry the declared identity");
  assert(after.rows === 6 && after.ids?.join(",") === "tr_001,tr_002,tr_003,tr_004,tr_005,tr_006", `identity values differ from the metadata file order: ${JSON.stringify(after.ids)}`);
  assert(JSON.stringify(after.joins) === JSON.stringify(before.joins), "declaring the identity changed the owner's table pairing");
  assert(after.featureKeys.length > 0 && after.featureKeys.every((x) => x.key === null), `a feature source carries a key its table lacks (R3): ${JSON.stringify(after.featureKeys)}`);
  assert(after.keyed.every((x) => x.has), `a source key names a column its own table does not contain (R3): ${JSON.stringify(after.keyed)}`);
  assert(!after.issues.some((text) => text.includes("was not applied")), `the ID decision was dropped: ${after.issues.join(";")}`);

  // With a confirmed identity, a group column can now be declared.
  await page.selectOption('select[aria-label="site role"]', "group");
  await idle(page);
  const grouped = await snapshot();
  assert(!grouped.specError && !grouped.assembleError, `group declaration refused: ${JSON.stringify(grouped)}`);
  assert(grouped.canonicalSampleIndex?.group_id === "site" && grouped.canonicalSampleIndex.key === "sample_id", `group_id was not written: ${JSON.stringify(grouped.canonicalSampleIndex)} ${grouped.issues.join(";")}`);
  await page.close();
}

// Re-inference with explicit parsing must retain confirmed locks and report the owner's refusal when
// sample_id disappears. The R3 owner may avoid a dangling key, so the obsolete L3 text is not required.
async function keyColumnAbsentFlow() {
  const page = await newPage();
  await page.setInputFiles("#file-input", files([
    ["X_train.csv", "sample_id;1000;1010;1020\ns1;0.1;0.2;0.3\ns2;0.2;0.3;0.4\ns3;0.3;0.4;0.5\ns4;0.4;0.5;0.6\n"],
    ["y_train.csv", "sample_id;protein\ns1;1.0\ns2;2.0\ns3;3.0\ns4;4.0\n"],
    ["metadata_train.csv", "sample_id;cultivar;site\ns1;A;north\ns2;B;south\ns3;A;north\ns4;B;south\n"],
  ]));
  await idle(page);
  for (let guard = 0; guard < 30; guard++) {
    const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
    if (!(await button.count())) break;
    await button.click();
    await idle(page);
  }
  const keyed = () => stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const spec = s.canonical ? JSON.parse(s.canonical) : null;
    return { keyed: (spec?.sources || []).filter((x) => x.key !== undefined).map((x) => ({ id: x.id, key: x.key })),
      blockers: document.querySelector("#export-notes").textContent, locks: s.proposalLocks };
  });
  const before = await keyed();
  assert(before.keyed.length > 0, `the fixture has no declared source key to test: ${JSON.stringify(before)}`);
  assert(!before.blockers.includes("has no such column with the final parsing parameters"), `the key guard fired before any edit: ${before.blockers}`);

  await page.selectOption('select[aria-label="Table to edit"]', "X_train.csv");
  await page.selectOption('label:has-text("Header row") select', "false");
  await idle(page);
  const after = await stateOf(page, async () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    let ownerBuilderRefusal = null;
    if (s.builderError) {
      // A host-side exception must not count as an owner refusal.
      const owner = await import("/pkg/io/nirs4all_io_wasm.js");
      const inputs = s.files.filter((file) => s.descriptions.get(file.name)?.readable).map(({ name, bytes }) => ({ name, bytes }));
      const names = new Set(inputs.map((file) => file.name));
      const file_params = Object.fromEntries([...s.parsing].filter(([name, choice]) => names.has(name) && Object.keys(choice).length));
      try { owner.proposeDataset(inputs, [], { confirmed: s.proposalLocks, file_params }); }
      catch (error) {
        const message = String(error?.message || error);
        if (message === s.builderError) ownerBuilderRefusal = message;
      }
    }
    const echoes = new Set((s.result?.proposals || [])
      .filter((p) => p.status === "confirmed" || p.status === "overridden").map((p) => `${p.kind}|${p.target}`));
    const unresolvedKeyLocks = s.usedLocks.filter((lock) => ["identity", "pairing"].includes(lock.kind)
      && JSON.stringify(lock.value).includes("sample_id") && !echoes.has(`${lock.kind}|${lock.target}`));
    return { blockers: document.querySelector("#export-notes").textContent, disabled: document.querySelector("#export").disabled,
      columns: s.descriptions.get("X_train.csv")?.columns.map((c) => c.name), locks: s.proposalLocks,
      ownerRefusals: [ownerBuilderRefusal, s.specError, s.assembleError].filter(Boolean),
      unresolvedKeyWarnings: unresolvedKeyLocks.length ? (s.result?.plan?.warnings || []).filter((warning) => warning.includes("sample_id")) : [],
      displayed: ["#owner-notes", "#preview-notes", "#export-notes"].map((selector) => document.querySelector(selector)?.textContent || "").join("\n") };
  });
  assert(!after.columns.includes("sample_id"), `the edit did not remove the key column from the inspected columns: ${after.columns}`);
  assert(JSON.stringify(after.locks) === JSON.stringify(before.locks), "changing parsing cleared or rewrote the confirmed locks");
  const ownerReasons = [...after.ownerRefusals, ...after.unresolvedKeyWarnings];
  const displayedReason = ownerReasons.find((reason) => after.displayed.includes(reason));
  assert(after.disabled && displayedReason,
    `export is not blocked with the actual owner refusal for the absent key column: ${JSON.stringify(after)}`);
  assert(await page.locator("#owner-notes, #preview-notes, #export-notes").filter({ hasText: displayedReason }).first().isVisible(),
    `the owner refusal is not visible: ${displayedReason}`);
  await page.close();
}

async function explicitHeaderlessFlow() {
  const page = await newPage();
  const name = "myspectra.csv";
  const text = Array.from({ length: 20 }, (_, row) =>
    `0.${String(465041 + row * 10000).padStart(7, "0")};${Array(699).fill("0.2").join(";")}\n`
  ).join("");
  await page.setInputFiles("#file-input", files([[name, text]]));
  await idle(page);
  const headerStatus = () => stateOf(page, (fileName) => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const description = s.descriptions.get(fileName);
    return { readable: description?.readable, ambiguous: description?.header_ambiguous,
      choice: s.parsing.get(fileName)?.has_header, effective: description?.params?.has_header,
      disabled: document.querySelector("#export").disabled,
      notes: document.querySelector("#export-notes").textContent };
  }, name);
  const auto = await headerStatus();
  assert(auto.readable && auto.ambiguous && auto.choice === undefined && auto.disabled
    && auto.notes.includes("header row is ambiguous"), `numeric Auto must block export: ${JSON.stringify(auto)}`);
  assert(await page.inputValue('label:has-text("Header row") select') === "", "an undeclared header choice was shown as explicit");
  await page.selectOption('label:has-text("Header row") select', "true");
  await idle(page);
  const declared = await headerStatus();
  assert(declared.choice === true && declared.effective === true && !declared.ambiguous
    && !declared.notes.includes("header row is ambiguous"), `explicit true was not applied: ${JSON.stringify(declared)}`);
  assert(await page.inputValue('label:has-text("Header row") select') === "true", "explicit true was lost on render");
  await page.selectOption('label:has-text("Header row") select', "");
  await idle(page);
  const reset = await headerStatus();
  assert(reset.choice === undefined && reset.ambiguous && reset.disabled
    && reset.notes.includes("header row is ambiguous"), `Auto retained an explicit choice: ${JSON.stringify(reset)}`);
  await page.selectOption('label:has-text("Header row") select', "false");
  await idle(page);
  const actual = await stateOf(page, async (fileName) => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const source = s.result?.spec?.sources?.find((item) => item.input === fileName);
    const description = s.descriptions.get(fileName);
    const block = s.assembled?.blocks?.train;
    const matrix = block?.x?.[block.source_ids.indexOf(source?.id)];
    const target = block?.y?.columns?.[0];
    const retained = s.files.find((file) => file.name === fileName);
    const digest = await crypto.subtle.digest("SHA-256", retained.bytes);
    return { header: source?.params?.has_header, canonicalHeader: s.canonical ? JSON.parse(s.canonical).sources[0].params?.has_header : null,
      rows: block?.n_samples, matrixRows: matrix?.n_rows, first: matrix?.data?.[0], last: matrix?.data?.[19 * matrix.n_cols],
      matrixCols: matrix?.n_cols,
      literalValues: matrix?.data?.length === 20 * 699 && matrix.data.every((value, index) => value === (index % 699
        ? 0.2 : Number(`0.${String(465041 + Math.floor(index / 699) * 10000).padStart(7, "0")}`))),
      valid: matrix?.validity?.length === 20 * 699 && matrix.validity.every((value) => value === true),
      targets: block?.y?.columns?.length === 1 && target.block?.dtype === "float64"
        && target.block.n_rows === 20 && target.block.n_cols === 1
        // Exact little-endian float64 bytes for twenty literal 0.2 targets.
        && target.block.data_bytes_hex === "9a9999999999c93f".repeat(20),
      targetsValid: target?.block?.validity?.length === 20 && target.block.validity.every((value) => value === true),
      columns: description?.columns.map((column) => column.name), featureHeaders: block?.feature_headers?.[0],
      sha: [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join(""),
      ownerNotes: document.querySelector("#owner-notes").textContent,
      exportDisabled: document.querySelector("#export").disabled,
      error: s.builderError || s.specError || s.assembleError };
  }, name);
  assert(!actual.error, `explicit no-header parsing was refused: ${JSON.stringify(actual)}`);
  assert(actual.header === false && actual.canonicalHeader === false, `the owner proposal did not receive the explicit header choice: ${JSON.stringify(actual)}`);
  assert(actual.rows === 20 && actual.matrixRows === 20 && actual.first === 0.0465041 && actual.last === 0.0655041,
    `explicit no-header parsing lost or changed an observation: ${JSON.stringify(actual)}`);
  assert(actual.matrixCols === 699 && actual.literalValues && actual.valid && actual.targets && actual.targetsValid,
    `explicit no-header values, shapes or masks differ: ${JSON.stringify(actual)}`);
  assert(!actual.exportDisabled, "explicit false did not release export");
  assert(await page.inputValue('label:has-text("Header row") select') === "false", "explicit false was lost on render");
  const positional = Array.from({ length: 700 }, (_, index) => String(index));
  assert(JSON.stringify(actual.columns) === JSON.stringify(positional), `inspection did not retain owner positional names: ${actual.columns}`);
  assert(JSON.stringify(actual.featureHeaders) === JSON.stringify(positional.slice(0, 699)), `proposal selectors differ from owner positional names: ${actual.featureHeaders}`);
  assert(actual.sha === createHash("sha256").update(Buffer.from(text)).digest("hex"), "the header choice changed the retained input bytes");
  assert(!actual.ownerNotes.includes("were computed without your parsing choices"), "the obsolete parsing note is still shown");
  await page.close();

  const namedPage = await newPage();
  await namedPage.setInputFiles("#file-input", files([
    ["named.csv", "sample_id;1000;1100;1200\ns1;0.1;0.2;0.3\ns2;0.4;0.5;0.6\n"],
  ]));
  await idle(namedPage);
  const named = await stateOf(namedPage, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const description = s.descriptions.get("named.csv");
    return { readable: description?.readable, ambiguous: description?.header_ambiguous,
      header: description?.params?.has_header, columns: description?.columns.map((column) => column.name),
      choice: s.parsing.get("named.csv")?.has_header,
      notes: document.querySelector("#export-notes").textContent };
  });
  assert(named.readable && !named.ambiguous && named.header === true && named.choice === undefined
    && named.columns.join(",") === "sample_id,1000,1100,1200" && !named.notes.includes("header row is ambiguous"),
    `ordinary named-header input received a false header block: ${JSON.stringify(named)}`);
  assert(await namedPage.inputValue('label:has-text("Header row") select') === "", "named-header Auto became an explicit choice");
  await namedPage.close();
}

// R2 counterexample: the same metadata-table ID decision, but the owner's pairings were never confirmed.
// The identity must not be attached through an unconfirmed positional alignment, and export stays blocked.
async function unconfirmedIdentityFlow() {
  const page = await newPage();
  await loadSample(page);
  await page.selectOption('select[aria-label="Table to edit"]', "metadata_train.csv");
  await page.waitForSelector('select[aria-label="sample_id role"]');
  await page.selectOption('select[aria-label="sample_id role"]', "id");
  await idle(page);
  const held = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const spec = s.canonical ? JSON.parse(s.canonical) : null;
    return {
      identityLocks: s.usedLocks.filter((l) => l.kind === "identity"), sampleIndex: spec?.sample_index ?? null,
      issues: s.issues.map((i) => `${i.level}:${i.text}`), disabled: document.querySelector("#export").disabled,
      blockers: document.querySelector("#export-notes").textContent,
    };
  });
  assert(!held.identityLocks.length && !held.sampleIndex, `an identity was attached through unconfirmed pairings: ${JSON.stringify(held)}`);
  assert(held.issues.some((text) => text.startsWith("error:") && text.includes("confirm the owner pairing")), `the held-back decision is not reported: ${JSON.stringify(held.issues)}`);
  assert(held.disabled && held.blockers.includes("Confirm every identity and pairing decision"), `export is not blocked by the open decisions: ${held.blockers}`);

  // The user now confirms every open decision through the shared list: the pending ID decision is applied (not lost)
  // and the visible provenance names the identity column and the confirmed alignment.
  for (let guard = 0; guard < 30; guard++) {
    const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
    if (!(await button.count())) break;
    await button.click();
    await idle(page);
  }
  const applied = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const spec = s.canonical ? JSON.parse(s.canonical) : null;
    return {
      identityLocks: s.usedLocks.filter((l) => l.kind === "identity"), sampleIndex: spec?.sample_index ?? null,
      issues: s.issues.map((i) => `${i.level}:${i.text}`), disabled: document.querySelector("#export").disabled,
      blockers: document.querySelector("#export-notes").textContent, preview: document.querySelector("#preview-notes").textContent,
      specError: s.specError, assembleError: s.assembleError,
    };
  });
  assert(applied.identityLocks.length === 1 && applied.sampleIndex?.by === "id" && applied.sampleIndex.key === "sample_id", `the confirmed ID decision was not applied after the pairings were confirmed: ${JSON.stringify(applied)}`);
  assert(applied.preview.includes("Identity 'sample_id'") && applied.preview.includes("confirmed pairings"), `identity/alignment provenance is not visible: ${applied.preview}`);
  assert(!applied.disabled, `export stayed blocked after every decision was confirmed: ${applied.blockers} ${JSON.stringify(applied.issues)}`);
  await page.close();
}

async function naFlow() {
  const page = await newPage();
  await page.setInputFiles("#file-input", files([
    ["X_train.csv", "1000;1010;1020\n1;2;\n4;5;6\n"],
    ["y_train.csv", "protein\n1.0\n2.0\n"],
  ]));
  await idle(page);
  const aborted = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    return { kept: s.previewKeptMissing, canonical: !!s.canonical, rows: Object.values(s.assembled?.blocks || {}).map((b) => b.n_samples) };
  });
  assert(aborted.canonical, "no specification for the NA fixture");
  assert(aborted.kept && aborted.kept.includes("na_policy is 'abort'"), "the owner abort refusal was not shown for a missing X cell");
  assert((await page.textContent("#preview-notes")).includes("missing values kept"), "the keep-missing preview banner is missing");
  assert(JSON.parse(await stateOf(page, () => window.NIRS4ALL_IO_BUILDER_STATE.canonical)).params?.na?.policy !== "ignore", "the preview override leaked into the specification");

  await page.selectOption('label:has-text("Missing values (dataset default)") select', "remove_sample");
  await idle(page);
  const removed = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    return { policy: JSON.parse(s.canonical).params?.na?.policy, rows: Object.values(s.assembled?.blocks || {}).map((b) => b.n_samples), kept: s.previewKeptMissing };
  });
  assert(removed.policy === "remove_sample" && !removed.kept, "remove_sample was not written or still aborts");
  assert(removed.rows.reduce((a, b) => a + b, 0) < aborted.rows.reduce((a, b) => a + b, 0), "remove_sample did not lower the owner's row count");

  // Replace with a chosen numeric value: the value reaches the owner specification (nested `fill`) and the assembly.
  await page.selectOption('label:has-text("Missing values (dataset default)") select', "replace");
  await idle(page);
  const fillInput = page.locator('label:has-text("Fill value (number)") input');
  await fillInput.fill("7.5");
  await fillInput.press("Tab");
  await idle(page);
  const replaced = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const block = Object.values(s.assembled?.blocks || {})[0];
    return { na: s.canonical ? JSON.parse(s.canonical).params?.na : null, kept: s.previewKeptMissing, error: s.specError || s.assembleError,
      cell: block?.x?.[0]?.data?.[2], valid: block?.x?.[0]?.validity?.[2], rows: block?.n_samples };
  });
  assert(!replaced.error && !replaced.kept, `replace was not accepted by the owner: ${JSON.stringify(replaced)}`);
  assert(replaced.na?.policy === "replace" && replaced.na?.fill?.method === "value" && replaced.na?.fill?.fill_value === 7.5, `the chosen fill value is not in the owner specification: ${JSON.stringify(replaced.na)}`);
  assert(replaced.rows === 2 && replaced.cell === 7.5 && replaced.valid === true, `the chosen fill value did not reach the owner assembly: ${JSON.stringify(replaced)}`);
  await page.close();

  // An Ignored non-metadata column is NOT exempt from abort.
  const second = await newPage();
  await second.setInputFiles("#file-input", files([
    ["X_train.csv", "1000;1010\n1;2\n4;5\n"],
    ["y_train.csv", "protein;note\n1.0;a\n2.0;\n"],
  ]));
  await idle(second);
  await second.selectOption('select[aria-label="Table to edit"]', "y_train.csv");
  await second.waitForSelector('select[aria-label="note role"]');
  await second.selectOption('select[aria-label="note role"]', "ignored");
  await idle(second);
  const ignored = await stateOf(second, () => window.NIRS4ALL_IO_BUILDER_STATE.previewKeptMissing);
  assert(ignored && ignored.includes("na_policy is 'abort'"), "an ignored column with a missing cell did not abort under the default policy");
  await second.close();
}

async function refusalAndReplacementFlow() {
  const page = await newPage();
  await loadSample(page);
  // Every selected input is retained: readability is the owner's decision, not the file name's.
  // A table with a wide data row is refused by the owner's reader (or flagged as an unresolved delimiter).
  await page.setInputFiles("#file-input", files([["bad.csv", "a;b\n1;2;3\n"]]));
  await idle(page);
  const unreadable = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const d = s.descriptions.get("bad.csv");
    return { retained: s.files.some((f) => f.name === "bad.csv"), readable: d?.readable, why: d?.refusal || (d?.delimiter_unresolved ? "delimiter unresolved" : ""), disabled: document.querySelector("#export").disabled };
  });
  assert(unreadable.retained && unreadable.readable === false && unreadable.why, `the owner did not refuse bad.csv explicitly: ${JSON.stringify(unreadable)}`);
  assert(unreadable.disabled, "export stayed enabled while an owner-refused input is present");
  assert((await page.textContent("#files")).includes("bad.csv"), "the refused file is not listed");
  await page.locator("#files tr", { hasText: "bad.csv" }).locator("button", { hasText: "Remove" }).click();
  await idle(page);
  assert(await stateOf(page, () => !window.NIRS4ALL_IO_BUILDER_STATE.files.some((f) => f.name === "bad.csv")), "the refused file was not removed");

  // Regression: a readable extensionless table is retained and described by the owner, not dropped by suffix.
  await page.setInputFiles("#file-input", files([["table_noext", "a;b\n1;2\n3;4\n"]]));
  await idle(page);
  const noext = await stateOf(page, () => {
    const s = window.NIRS4ALL_IO_BUILDER_STATE;
    const d = s.descriptions.get("table_noext");
    return { retained: s.files.some((f) => f.name === "table_noext"), readable: d?.readable, columns: d?.columns.map((c) => c.name), rows: d?.n_rows };
  });
  assert(noext.retained && noext.readable === true && noext.columns?.join(",") === "a,b" && noext.rows === 2, `an extensionless table was not retained and read by the owner: ${JSON.stringify(noext)}`);
  // R1: readable but with no owner source, the file must not vanish silently from the export. It is shown as
  // unassigned, kept out of the column editor and blocks export (no suffix rule is involved).
  const unassignedView = await stateOf(page, () => ({
    row: [...document.querySelectorAll("#files tr")].find((tr) => tr.textContent.includes("table_noext"))?.textContent || "",
    pickable: [...document.querySelectorAll('select[aria-label="Table to edit"] option')].map((o) => o.value),
    blockers: document.querySelector("#export-notes").textContent, disabled: document.querySelector("#export").disabled,
  }));
  assert(unassignedView.row.includes("unassigned") && unassignedView.row.includes("proposed no source"), `a readable file without an owner source is not shown as unassigned: ${unassignedView.row}`);
  assert(!unassignedView.pickable.includes("table_noext"), "the column editor offers a file the owner gave no source");
  assert(unassignedView.disabled && unassignedView.blockers.includes("table_noext is readable, but the owner proposed no source"), `export is not blocked for the unassigned file: ${unassignedView.blockers}`);
  await page.locator("#files tr", { hasText: "table_noext" }).locator("button", { hasText: "Remove" }).click();
  await idle(page);

  // Duplicate header names are the owner's mangled names.
  await page.setInputFiles("#file-input", files([["dup.csv", "a;a;b\n1;2;3\n"]]));
  await idle(page);
  const names = await stateOf(page, () => window.NIRS4ALL_IO_BUILDER_STATE.descriptions.get("dup.csv")?.columns.map((c) => c.name));
  assert(names?.join(",") === "a,a.1,b", `unexpected duplicate-header names: ${names}`);

  // Replacing a file with different bytes clears the decisions made against the old bytes.
  const button = page.locator("#proposals-host button", { hasText: /^Confirm$/ }).first();
  if (await button.count()) { await button.click(); await idle(page); }
  await page.setInputFiles("#file-input", files([["y_train.csv", "protein\n9.9\n8.8\n7.7\n6.6\n5.5\n4.4\n"]]));
  await idle(page);
  assert(await stateOf(page, () => window.NIRS4ALL_IO_BUILDER_STATE.proposalLocks.length) === 0, "replacing a file kept stale locks");

  // Reset empties the state and unmounts every island card.
  await page.click("#reset");
  await idle(page);
  const cleared = await stateOf(page, () => ({ files: window.NIRS4ALL_IO_BUILDER_STATE.files.length, cards: document.querySelectorAll("#preview-cards [data-partition]").length, columns: document.querySelector("#columns-host").children.length }));
  assert(cleared.files === 0 && cleared.cards === 0 && cleared.columns === 0, "reset left files or mounted islands behind");
  await page.close();
}
