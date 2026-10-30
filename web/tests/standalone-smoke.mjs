import { pathToFileURL } from "node:url";
import path from "node:path";
import { readFile } from "node:fs/promises";

const here = path.dirname(new URL(import.meta.url).pathname);
const root = path.resolve(here, "..");
const workspace = path.resolve(root, "..", "..");
const formatsRoot = process.env.N4IO_FORMATS_ROOT || path.join(workspace, "nirs4all-formats");
const formatsSamples = path.join(formatsRoot, "samples");
const standalone = process.env.N4IO_PAGE_URL || pathToFileURL(path.join(root, "dist", "nirs4all-io-standalone.html")).href;
const chromeExecutable = process.env.CHROME_BIN || "/usr/bin/google-chrome";

async function loadPlaywright() {
  const moduleName = process.env.PLAYWRIGHT_MODULE || "playwright";
  const specifier = moduleName.startsWith("/") ? pathToFileURL(moduleName).href : moduleName;
  return import(specifier);
}

const { chromium } = await loadPlaywright();
const browser = await chromium.launch({
  executablePath: chromeExecutable,
  headless: true,
  args: ["--allow-file-access-from-files"],
});

const errors = [];
function hook(page) {
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
}

async function newStandalonePage(viewport = { width: 1280, height: 900 }) {
  const page = await browser.newPage({ viewport });
  hook(page);
  await page.goto(standalone, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => document.querySelector("#runtime")?.textContent.includes("formats"),
    null,
    { timeout: 10000 },
  );
  return page;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

try {
  await sampleDatasetFlow();
  await scatterExplanationFlow();
  await auxiliaryTablesFlow();
  await targetTableAuditFlow();
  await tableValidationEditingFlow();
  await canonicalEditingFlow();
  await unreadableFileFlow();
  if (process.env.N4IO_PAGE_URL) await startupFlow();
  await dragAndDropFlow();
  await mixedRepeatedTableFlow();
  await zstdParquetFlow();
  await sidecarFlow();
  await folderUploadFlow();
  await nonSpectralRefusalFlow();
  await formatRefusalFlow();

  if (errors.length) {
    throw new Error(`browser errors:\n${errors.join("\n")}`);
  }
  console.log(`OK: IO page smoke passed (${process.env.N4IO_PAGE_URL || "standalone file://"})`);
} finally {
  await browser.close();
}

async function sampleDatasetFlow() {
  const page = await newStandalonePage({ width: 1440, height: 1100 });
  await page.click("#loadSample");
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("loaded"),
    null,
    { timeout: 15000 },
  );
  const result = await page.evaluate(() => {
    const canvas = document.querySelector("#vizCanvas");
    const ctx = canvas.getContext("2d");
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonWhite = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (!(data[i] > 248 && data[i + 1] > 248 && data[i + 2] > 248)) nonWhite++;
    }
    return {
      status: document.querySelector("#status")?.textContent,
      validation: document.querySelector("#validationBadge")?.textContent,
      readers: document.querySelectorAll(".reader-tags span").length,
      sourceRows: document.querySelectorAll(".source-row").length,
      fileErrors: document.querySelectorAll("#fileList .probe-error").length,
      target: document.querySelector("#targetSelect")?.value,
      activeTab: document.querySelector(".tab.active")?.textContent,
      legend: document.querySelector("#legend")?.textContent,
      canvasNonWhite: nonWhite,
    };
  });
  assert(result.status.includes("train/test"), "sample flow did not infer train/test");
  assert(result.validation === "valid", "sample DatasetSpec is not valid");
  assert(result.readers >= 40, "reader coverage is not visible");
  assert(result.sourceRows >= 6, "sample sources not rendered");
  assert(result.fileErrors === 0, "supported target/metadata tables are shown as decode errors");
  assert(result.target === "protein", "sample target not selected");
  assert(result.activeTab === "Y histogram", "Y histogram not active");
  assert(result.legend.includes("protein"), "Y histogram legend missing target");
  assert(result.canvasNonWhite > 1000, "dataviz canvas appears blank");
  await page.close();
}

async function scatterExplanationFlow() {
  const page = await newStandalonePage();
  await page.click('[data-viz="scatter"]');
  assert(await page.locator("#scatterInfo").isVisible(), "empty scatter explanation is hidden");
  assert((await page.textContent("#scatterSummary")).includes("No scatter points"), "empty scatter retains stale data");

  // Exercise the existing decoded-record seam with known values, including missing
  // channels, a single finite channel, and a record without the selected signal.
  await page.evaluate(() => {
    const app = window.NIRS4ALL_APP;
    app.state.decoded = [{ file: "preview.csv", ok: true, records: Array.from({ length: 24 }, (_, index) => ({
      signals: {
        signal: { values: index === 1 ? [7] : [1, NaN, null, Infinity, 3] },
        ...(index === 2 ? {} : { alternate: { values: index === 1 ? [10] : [9, 5, 1] } }),
      },
      targets: { protein: 12, fat: 2 },
      metadata: { partition: index < 12 ? "train" : "test" },
    })) }];
    app.state.activeSignal = "signal";
    app.state.activeTarget = "protein";
    app.renderAll();
  });
  await page.click("#scatterData summary");
  const firstCells = await page.locator("#scatterValues tr").first().locator("th, td").allTextContents();
  assert(firstCells.slice(3, 6).join(",") === "2,2,2", "scatter coordinates do not reflect finite signal values");
  const singleCells = await page.locator("#scatterValues tr").nth(1).locator("th, td").allTextContents();
  assert(singleCells.slice(3, 6).join(",") === "1,7,0", "single-channel endpoint change is incorrect");
  assert(await page.locator("#scatterValues tr").count() === 24, "scatter table does not include all displayed points");
  const axes = await page.textContent("#scatterAxes");
  assert(axes.includes("Signal units are unknown") && axes.includes("last finite value minus its first"), "scatter axes or units are unexplained");
  assert(await page.getAttribute("#vizCanvas", "aria-describedby") === "scatterAxes scatterSummary", "canvas is not connected to its text alternative");

  await page.selectOption("#signalSelect", "alternate");
  const alternateCells = await page.locator("#scatterValues tr").first().locator("th, td").allTextContents();
  assert(alternateCells.slice(2, 6).join(",") === "alternate,3,5,-8", "signal selection did not update plotted values");
  const fallbackCells = await page.locator("#scatterValues tr").nth(2).locator("th, td").allTextContents();
  assert(fallbackCells.slice(2, 6).join(",") === "signal,2,2,2", "scatter table mislabels the fallback signal");
  await page.selectOption("#targetSelect", "fat");
  assert(await page.getAttribute('[data-viz="scatter"]', "class") === "tab active", "target selection leaves the scatter view");
  const targetCells = await page.locator("#scatterValues tr").first().locator("th, td").allTextContents();
  assert(targetCells[7] === "fat: 2" && targetCells.slice(4, 6).join(",") === "5,-8", "target selection changes coordinates or leaves stale labels");
  await page.selectOption("#colorSelect", "index");
  assert((await page.textContent("#legend")).includes("Colors cycle through record order"), "index coloring has a misleading target legend");
  await page.locator("#sampleLimit").evaluate((input) => { input.value = "20"; input.dispatchEvent(new Event("input", { bubbles: true })); });
  assert(await page.locator("#scatterValues tr").count() === 20, "scatter values ignore the sample cap");
  assert((await page.textContent("#scatterSummary")).includes("Showing 20 of 24"), "scatter summary ignores the sample cap");

  await page.setViewportSize({ width: 320, height: 900 });
  const overflow = await page.locator("#scatterData").evaluate((element) => element.scrollWidth > element.clientWidth);
  assert(!overflow, "scatter text/table overflows its container on a narrow screen");
  await page.click('[data-viz="spectra"]');
  assert(await page.locator("#scatterInfo").isHidden() && await page.locator("#scatterData").isHidden(), "scatter explanation remains visible in another view");
  await page.close();
}

async function auxiliaryTablesFlow() {
  const page = await newStandalonePage();
  await page.setInputFiles("#fileInput", [
    { name: "X.csv", mimeType: "text/csv", buffer: Buffer.from("1000;1010;1020\n0.1;0.2;0.3\n0.4;0.5;0.6\n") },
    { name: "Y.csv", mimeType: "text/csv", buffer: Buffer.from("protein\n10\n12\n") },
    { name: "M.csv", mimeType: "text/csv", buffer: Buffer.from("batch;site;2024\nA;north;1\nB;south;2\n") },
  ]);
  await page.waitForFunction(() => window.NIRS4ALL_APP?.state.spec, null, { timeout: 15000 });
  const result = await page.evaluate(() => {
    const { state } = window.NIRS4ALL_APP;
    return { decoded: state.decoded, sources: state.spec.sources, warnings: state.plan.warnings };
  });
  assert(result.sources.some((source) => source.role === "metadata" && source.input === "M.csv"), "metadata table was not inferred");
  assert(!result.decoded.some((entry) => entry.file === "M.csv" || entry.file === "Y.csv"), "auxiliary table was sent to the spectral reader");
  assert(result.decoded.some((entry) => entry.file === "X.csv" && entry.ok), "spectral table was not decoded");
  assert(!result.warnings.some((warning) => warning.includes("M.csv") && warning.includes("excluded")), "metadata table was excluded from inference");
  await page.click('#modeToggle [data-mode="builder"]');
  await page.waitForFunction(() => window.NIRS4ALL_APP.state.assembled || window.NIRS4ALL_APP.state.assembleError, null, { timeout: 15000 });
  const preview = await page.evaluate(() => ({ assembled: window.NIRS4ALL_APP.state.assembled, error: window.NIRS4ALL_APP.state.assembleError }));
  assert(preview.assembled, `dataset with metadata did not assemble: ${preview.error}`);
  const blocks = Object.values(preview.assembled.blocks || {});
  assert(blocks.some((block) => ["batch", "site", "2024"].every((name) => block.metadata?.columns.some((column) => column.name === name))), "metadata columns were lost during assembly");
  await page.close();
}

async function targetTableAuditFlow() {
  for (const [content, expected] of [
    ["protein\n10\n12\n", "protein"],
    ["class\nred\nblue\n", "class"],
  ]) {
    const page = await newStandalonePage();
    await page.setInputFiles("#fileInput", [
      { name: "X.csv", mimeType: "text/csv", buffer: Buffer.from("1000;1010;1020\n0.1;0.2;0.3\n0.4;0.5;0.6\n") },
      { name: "Y.csv", mimeType: "text/csv", buffer: Buffer.from(content) },
    ]);
    await page.waitForFunction(() => window.NIRS4ALL_APP?.state.spec);
    const row = page.locator("#fileList .file-row").filter({ hasText: "Y.csv" });
    assert(!(await row.innerText()).toLowerCase().includes("unsupported"), "valid target shows an unsupported spectral audit");
    assert(await page.locator("#targetSelect").inputValue() === expected, "target header was not selected");
    const targetPreview = await page.evaluate(() => {
      const view = window.NIRS4ALL_APP.datasetView();
      return { values: view.targetPreviewValues.map((item) => item.value), aligned: view.rows.map((row) => row.target) };
    });
    const expectedValues = expected === "class" ? "red,blue" : "10,12";
    assert(targetPreview.values.join(",") === expectedValues && targetPreview.aligned.join(",") === expectedValues, "target preview differs from IO assembly or includes the header as a value");
    assert(!(await page.textContent("#datasetAlerts")).includes("assembly could not"), "valid target did not assemble");
    await page.click('#modeToggle [data-mode="builder"]');
    await page.waitForFunction(() => window.NIRS4ALL_APP.state.assembled || window.NIRS4ALL_APP.state.assembleError);
    const block = await page.evaluate(() => Object.values(window.NIRS4ALL_APP.state.assembled?.blocks || {})[0]);
    assert(block?.y_headers?.includes(expected) && block.y.n_rows === 2, "target was not assembled");
    if (expected === "class") assert(block.y_categorical.class.categories.join(",") === "red,blue", "categorical labels were lost");
    else assert(block.y.data.join(",") === "10,12", "numeric target values were lost");
    await page.close();
  }
  const page = await newStandalonePage();
  await page.setInputFiles("#fileInput", [
    { name: "X.csv", mimeType: "text/csv", buffer: Buffer.from("1000;1010;1020\n0.1;0.2;0.3\n0.4;0.5;0.6\n") },
    { name: "Y.csv", mimeType: "text/csv", buffer: Buffer.from('protein\n"unclosed\n12\n') },
  ]);
  await page.waitForFunction(() => window.NIRS4ALL_APP?.state.spec);
  const alert = await page.textContent("#datasetAlerts");
  assert(alert.includes("assembly could not validate") && alert.includes("train_y") && alert.includes("row-aligned"), "malformed target assembly error is not actionable");
  assert(await page.locator("#validationBadge").textContent() === "invalid", "failed table assembly retains a successful validation badge");
  await page.close();
}

async function tableValidationEditingFlow() {
  const page = await newStandalonePage();
  await page.setInputFiles("#fileInput", [
    { name: "X.csv", mimeType: "text/csv", buffer: Buffer.from("1000;1010;1020\n0.1;0.2;0.3\n0.4;0.5;0.6\n") },
    { name: "Y.csv", mimeType: "text/csv", buffer: Buffer.from("class\nred\nblue\n") },
  ]);
  await page.waitForFunction(() => window.NIRS4ALL_APP?.state.auxiliaryAssembly?.blocks);
  // The first form commit also makes inferred defaults explicit. Thereafter,
  // identity edits and display controls must reuse the materialized tables.
  await page.locator('[name="name"]').fill("Named dataset");
  await page.locator('[name="name"]').press("Tab");
  await page.evaluate(() => { window.__tableAssembly = window.NIRS4ALL_APP.state.auxiliaryAssembly; });
  await page.locator('[name="name"]').fill("Renamed dataset");
  await page.locator('[name="name"]').press("Tab");
  await page.selectOption("#colorSelect", "index");
  assert(await page.evaluate(() => window.__tableAssembly === window.NIRS4ALL_APP.state.auxiliaryAssembly), "cosmetic edits or display controls rerun table assembly");
  const sourceInput = page.locator('.source-row [name="source_input"]').nth(1);
  await sourceInput.fill("missing.csv");
  assert(await page.evaluate(() => window.__tableAssembly === window.NIRS4ALL_APP.state.auxiliaryAssembly), "table assembly runs before a text field is committed");
  assert(await page.locator("#targetSelect").isDisabled(), "uncommitted source edit retains stale target values");
  await sourceInput.press("Tab");
  assert(await page.locator("#validationBadge").textContent() === "invalid", "missing table input does not invalidate assembly");
  await sourceInput.fill("Y.csv");
  await sourceInput.press("Tab");
  assert(await page.locator("#validationBadge").textContent() === "valid", "correcting a source does not clear assembly failure");
  assert(!(await page.textContent("#datasetAlerts")).includes("assembly could not"), "correcting a source leaves a stale assembly alert");

  const editor = page.locator('textarea[name="json"]');
  const originalSpec = await page.evaluate(() => structuredClone(window.NIRS4ALL_APP.state.spec));
  const noHeader = structuredClone(originalSpec);
  noHeader.sources[1].params.has_header = false;
  await editor.fill(JSON.stringify(noHeader));
  await page.locator("#status").click();
  assert(await page.locator("#validationBadge").textContent() === "invalid", "loading parameter edit reused stale assembly success");
  await editor.fill(JSON.stringify(originalSpec));
  await page.locator("#status").click();
  assert(await page.locator("#validationBadge").textContent() === "valid", "loading parameter recovery retains stale assembly failure");
  const metadataOnly = structuredClone(originalSpec);
  metadataOnly.sources[1].role = "metadata";
  await editor.fill(JSON.stringify(metadataOnly));
  await page.locator("#status").click();
  assert(await page.locator("#targetSelect").isDisabled(), "role edit retains stale target values");

  await page.setInputFiles("#fileInput", { name: "Y.csv", mimeType: "text/csv", buffer: Buffer.from('species\n"red,ripe"\n"blue,green"\n') });
  await page.waitForFunction(() => window.NIRS4ALL_APP.datasetView().targetKeys.includes("species"));
  const replaced = await page.evaluate(() => window.NIRS4ALL_APP.datasetView().rows.map((row) => row.target));
  assert(replaced.join("|") === "red,ripe|blue,green", "same-name replacement or CSV quoting reuses stale or misparsed target values");
  await page.click("#reset");
  assert(await page.evaluate(() => window.NIRS4ALL_APP.state.auxiliaryAssembly === null), "reset retains materialized table data");
  assert(await page.locator("#targetSelect").isDisabled(), "reset retains target options");
  await page.close();
}

async function canonicalEditingFlow() {
  const page = await newStandalonePage();
  await page.click("#loadSample");
  await page.waitForFunction(() => window.NIRS4ALL_APP?.state.spec);
  const editor = page.locator("textarea[name='json']");
  const canonical = await page.evaluate(() => {
    const app = window.NIRS4ALL_APP;
    return app.getIo().to_spec(JSON.stringify(app.state.spec));
  });
  assert(await editor.inputValue() === canonical, "editor does not use IO canonical JSON");
  await editor.focus();
  await page.keyboard.press("End");
  await page.keyboard.type("  ");
  assert(await editor.evaluate((el) => document.activeElement === el), "JSON typing lost focus");

  for (const invalid of ["{", "null", "[]", '{"sources":{}}', '{"sources":[],"partitions":{"by":"random"}}']) {
    await editor.fill(invalid);
    assert(await page.locator("#validationBadge").textContent() === "invalid", "invalid JSON accepted");
    assert(await page.locator("#downloadSpec").isDisabled(), "invalid JSON can be downloaded");
    assert(await page.locator("#copySpec").isDisabled(), "invalid JSON can be copied");
  }
  await editor.fill(canonical);
  await page.locator("#status").click();

  // Editing a visible field must retain loading options authored in JSON.
  const edited = JSON.parse(canonical);
  edited.name = "Canonical café";
  edited.sources[0].params.na = { policy: "replace", fill: { method: "value", fill_value: 7 } };
  await editor.fill(JSON.stringify(edited));
  await page.locator("#status").click();
  await page.locator("input[name='name']").fill("Edited café");
  await page.locator("input[name='name']").press("Tab");
  const fill = await page.evaluate(() => window.NIRS4ALL_APP.state.spec.sources[0].params.na.fill);
  assert(fill.method === "value" && fill.fill_value === 7, "form edit discarded NA fill options");
  assert(await page.locator("#validationBadge").textContent() === "valid", "form edit invalidated spec");

  const expected = await page.evaluate(() => {
    const app = window.NIRS4ALL_APP;
    return app.getIo().to_spec(JSON.stringify(app.state.spec));
  });
  const downloaded = page.waitForEvent("download");
  await page.click("#downloadSpec");
  const download = await downloaded;
  assert(await readFile(await download.path(), "utf8") === expected, "export differs from canonical IO bytes");

  await page.evaluate(() => {
    window.__copiedSpec = null;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: async (text) => { window.__copiedSpec = text; } },
    });
  });
  await page.click("#copySpec");
  assert(await page.evaluate(() => window.__copiedSpec) === expected, "clipboard differs from canonical IO bytes");

  await editor.fill("null");
  await page.click("#resetSpec");
  assert(await page.locator("#validationBadge").textContent() === "valid", "reset did not recover invalid draft");
  assert(await page.locator("#downloadSpec").isEnabled(), "reset did not restore export");
  await page.close();
}

async function unreadableFileFlow() {
  const page = await newStandalonePage();
  await page.evaluate(() => {
    File.prototype.arrayBuffer = async () => { throw new Error("File access was denied"); };
  });
  await page.setInputFiles("#fileInput", {
    name: "unreadable.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("1000;1010\n1;2\n"),
  });
  await page.waitForFunction(() => document.querySelector("#status")?.textContent.includes("File access was denied"));
  assert(await page.locator("#downloadSpec").isDisabled(), "read failure left export enabled");
  await page.close();
}

async function startupFlow() {
  const page = await browser.newPage();
  hook(page);
  let releaseModule;
  const moduleGate = new Promise((resolve) => { releaseModule = resolve; });
  await page.route("**/pkg/io/nirs4all_io_wasm.js", async (route) => {
    await moduleGate;
    await route.continue();
  });
  try {
    await page.goto(standalone, { waitUntil: "domcontentloaded" });
    await page.click("#loadSample");
    assert(await page.evaluate(() => window.NIRS4ALL_APP.state.spec === null), "import ran before runtime startup");
    releaseModule();
    await page.waitForFunction(() => document.querySelector("#status")?.textContent.includes("loaded"));
    assert(await page.locator("#validationBadge").textContent() === "valid", "early import did not resume after startup");
  } finally {
    releaseModule();
    await page.close();
  }
}

async function dragAndDropFlow() {
  const page = await newStandalonePage({ width: 1100, height: 820 });
  await page.evaluate(async () => {
    const file = new File([
      "sample_id;1000;1010\ns1;0.1;0.2\ns2;0.2;0.3\n",
    ], "dragged.csv", { type: "text/csv" });
    const event = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", {
      value: {
        types: [],
        items: [],
        files: [file],
        dropEffect: "none",
      },
      configurable: true,
    });
    document.querySelector("#drop").dispatchEvent(event);
  });
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("loaded"),
    null,
    { timeout: 15000 },
  );
  const result = await page.evaluate(() => ({
    status: document.querySelector("#status")?.textContent,
    fileText: document.querySelector("#fileList")?.textContent,
    validation: document.querySelector("#validationBadge")?.textContent,
  }));
  assert(result.status.includes("single table"), "drop flow did not infer a single table");
  assert(result.fileText.includes("dragged.csv"), "drop flow did not add the dropped file");
  assert(result.validation === "valid", "drop flow DatasetSpec is not valid");
  await page.close();
}

async function mixedRepeatedTableFlow() {
  const page = await newStandalonePage({ width: 1280, height: 980 });
  const text = [
    "sample_id;replicate;protein;1000;1010;1020;batch",
    "s1;r1;10;0.1;0.2;0.3;A",
    "s1;r2;10;0.2;0.3;0.4;A",
    "s2;r1;12;0.3;0.4;0.5;B",
    "s2;r2;12;0.4;0.5;0.6;B",
    "s3;r1;13;0.5;0.6;0.7;C",
    "",
  ].join("\n");
  await page.setInputFiles("#fileInput", [
    {
      name: "mixed.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(text),
    },
  ]);
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("loaded"),
    null,
    { timeout: 15000 },
  );
  const result = await page.evaluate(() => {
    const canvas = document.querySelector("#vizCanvas");
    const ctx = canvas.getContext("2d");
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonWhite = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (!(data[i] > 248 && data[i + 1] > 248 && data[i + 2] > 248)) nonWhite++;
    }
    const spec = JSON.parse(document.querySelector('[name="json"]')?.value || "{}");
    const source = spec.sources?.[0] || {};
    const byRole = Object.fromEntries((source.columns || []).map((entry) => [entry.role, entry.select || []]));
    return {
      validation: document.querySelector("#validationBadge")?.textContent,
      target: document.querySelector("#targetSelect")?.value,
      activeTab: document.querySelector(".tab.active")?.textContent,
      legend: document.querySelector("#legend")?.textContent,
      canvasNonWhite: nonWhite,
      sampleKey: document.querySelector('[name="sample_index_key"]')?.value,
      observationId: document.querySelector('[name="sample_observation_id"]')?.value,
      repetitionId: document.querySelector('[name="sample_repetition_id"]')?.value,
      repetition: document.querySelector('[name="repetition"]')?.value,
      selectedInference: document.querySelector("#decisionList")?.textContent,
      spec,
      byRole,
    };
  });
  assert(result.validation === "valid", "mixed repeated table DatasetSpec is not valid");
  assert(result.sampleKey === "sample_id", "mixed table sample key not inferred");
  assert(result.observationId === "sample_id", "mixed table observation id not inferred");
  assert(result.repetitionId === "replicate", "mixed table repetition id not inferred");
  assert(result.repetition === "replicate", "mixed table top-level repetition not inferred");
  assert(result.target === "protein", "mixed table target not selected");
  assert(result.activeTab === "Y histogram", "mixed table did not show Y histogram");
  assert(result.legend.includes("protein"), "mixed table Y histogram legend missing protein");
  assert(result.canvasNonWhite > 1000, "mixed table dataviz canvas appears blank");
  assert(
    ["1000", "1010", "1020"].every((col) => result.byRole.features?.includes(col)),
    `mixed table features were not inferred from spectral columns: ${JSON.stringify(result.byRole.features)}`,
  );
  assert(result.byRole.targets?.includes("protein"), "mixed table target column protein not inferred as Y");
  assert(!result.byRole.features?.includes("protein"), "mixed table target protein was inferred as X");
  assert(result.byRole.metadata?.includes("batch"), "mixed table batch not inferred as metadata");
  assert(result.byRole.metadata?.includes("replicate"), "mixed table replicate not inferred as metadata");
  assert(result.selectedInference.includes("single table"), "mixed table structure evidence missing");
  await page.close();
}

async function zstdParquetFlow() {
  const page = await newStandalonePage({ width: 1280, height: 920 });
  await page.setInputFiles("#fileInput", path.join(formatsSamples, "parquet/synthetic_nirs.parquet"));
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("loaded"),
    null,
    { timeout: 15000 },
  );
  const result = await page.evaluate(() => {
    const canvas = document.querySelector("#vizCanvas");
    const ctx = canvas.getContext("2d");
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonWhite = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (!(data[i] > 248 && data[i + 1] > 248 && data[i + 2] > 248)) nonWhite++;
    }
    return {
      status: document.querySelector("#status")?.textContent,
      validation: document.querySelector("#validationBadge")?.textContent,
      fileText: document.querySelector("#fileList")?.textContent,
      target: document.querySelector("#targetSelect")?.value,
      canvasNonWhite: nonWhite,
      alerts: document.querySelector("#datasetAlerts")?.textContent || "",
    };
  });
  assert(result.status.includes("loaded"), "zstd Parquet status did not load");
  assert(result.validation === "valid", "zstd Parquet DatasetSpec is not valid");
  assert(result.fileText.includes("parquet-nirs-table"), "zstd Parquet format not shown");
  assert(result.target === "protein", "zstd Parquet target not selected");
  assert(!result.alerts.includes("could not be decoded"), "zstd Parquet still refused");
  assert(result.canvasNonWhite > 1000, "zstd Parquet dataviz canvas appears blank");
  await page.close();
}

async function sidecarFlow() {
  const page = await newStandalonePage({ width: 1000, height: 760 });
  await page.setInputFiles("#fileInput", [
    {
      name: "scene.lan",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("HEAD74 synthetic ERDAS LAN header"),
    },
  ]);
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("companion"),
    null,
    { timeout: 10000 },
  );
  const missing = await page.evaluate(() => document.querySelector("#datasetAlerts")?.textContent || "");
  assert(missing.includes("scene.spc"), "ERDAS LAN sidecar alert missing scene.spc");

  await page.setInputFiles("#fileInput", [
    {
      name: "scene.spc",
      mimeType: "application/octet-stream",
      buffer: Buffer.from("wavelengths"),
    },
  ]);
  await page.waitForFunction(
    () => !document.querySelector("#datasetAlerts")?.textContent.includes("scene.spc"),
    null,
    { timeout: 10000 },
  );
  await page.close();
}

async function folderUploadFlow() {
  const page = await newStandalonePage({ width: 1100, height: 820 });
  await page.evaluate(async () => {
    const files = [
      ["nirs/train/X_train.csv", "id;400;410\na;1;2\nb;2;3\nc;3;4\n"],
      ["nirs/train/y_train.csv", "id;protein\na;10\nb;11\nc;12\n"],
      ["nirs/train/metadata_train.csv", "id;batch\na;A\nb;A\nc;B\n"],
      ["nirs/test/X_test.csv", "id;400;410\nd;4;5\ne;5;6\n"],
      ["nirs/test/y_test.csv", "id;protein\nd;13\ne;14\n"],
      ["nirs/test/metadata_test.csv", "id;batch\nd;B\ne;C\n"],
    ].map(([relativePath, text]) => {
      const file = new File([text], relativePath.split("/").pop(), { type: "text/csv" });
      Object.defineProperty(file, "webkitRelativePath", { value: relativePath, configurable: true });
      return file;
    });
    const dt = new DataTransfer();
    files.forEach((file) => dt.items.add(file));
    const input = document.querySelector("#folderInput");
    Object.defineProperty(input, "files", { value: dt.files, configurable: true });
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("loaded"),
    null,
    { timeout: 15000 },
  );
  const result = await page.evaluate(() => ({
    status: document.querySelector("#status")?.textContent,
    firstFile: document.querySelector(".file-name")?.textContent,
    target: document.querySelector("#targetSelect")?.value,
    map: document.querySelector("#datasetMap")?.textContent,
  }));
  assert(result.status.includes("train/test"), "folder flow did not infer train/test");
  assert(result.firstFile.includes("nirs/train/X_train.csv"), "relative path not preserved");
  assert(result.target === "protein", "folder flow target not detected");
  assert(result.map.includes("joins train_x"), "folder flow joins not shown");
  await page.close();
}

async function nonSpectralRefusalFlow() {
  const page = await newStandalonePage({ width: 1000, height: 760 });
  const text = "SCAN ID,YEAR,DATE,DOY,DegDay,TIME,SITE,EXPERIMENT,BLOCK,TREATMENT,REP,NDVI (MODIS),EVI (MODIS),EVI2 (MODIS),PRI (550 Reference),PRI (570 Ref),WBI,Chl Index,LAI\n1,2007,2007-06-01,152,0,12:00,A,E,B,T,1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n";
  await page.setInputFiles("#fileInput", [
    {
      name: "arc_lter_unispec_dc_2007_2019_indices.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(text),
    },
  ]);
  await page.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("no wavelength-indexed spectra"),
    null,
    { timeout: 10000 },
  );
  const result = await page.evaluate(() => ({
    alert: document.querySelector("#datasetAlerts")?.textContent,
    probe: document.querySelector(".file-probe")?.textContent,
    specText: document.querySelector("#specForm")?.textContent,
    decision: document.querySelector("#decisionList")?.textContent,
  }));
  assert(result.alert.includes("excluded from X inference"), "non-spectral alert missing");
  assert(result.probe.includes("not wavelength-i"), "format refusal message missing");
  assert(!result.probe.includes("does not support in-memory reads"), "generic in-memory fallback leaked");
  assert(result.specText.includes("Dataset properties appear after inference"), "non-spectral file created a spec");
  assert(result.decision.includes("non-spectral"), "non-spectral decision missing");
  await page.close();
}

async function formatRefusalFlow() {
  const mzmlPage = await newStandalonePage({ width: 1000, height: 760 });
  await mzmlPage.setInputFiles("#fileInput", path.join(formatsSamples, "mzml/example.mzML"));
  await mzmlPage.waitForFunction(
    () => document.querySelector("#status")?.textContent.includes("no wavelength-indexed spectra"),
    null,
    { timeout: 10000 },
  );
  const mzml = await mzmlPage.evaluate(() => ({
    alert: document.querySelector("#datasetAlerts")?.textContent,
    probe: document.querySelector(".file-probe")?.textContent,
    specText: document.querySelector("#specForm")?.textContent,
    decision: document.querySelector("#decisionList")?.textContent,
  }));
  assert(mzml.alert.includes("excluded from X inference"), "mzML non-spectral alert missing");
  assert(mzml.probe.includes("mass-spectrometry data, not NIRS spectroscopy"), "mzML refusal message missing");
  assert(mzml.specText.includes("Dataset properties appear after inference"), "mzML created a spec");
  assert(mzml.decision.includes("non-spectral"), "mzML non-spectral decision missing");
  await mzmlPage.close();
}
