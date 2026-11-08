#!/usr/bin/env node
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, "dist");
const outFile = join(outDir, "nirs4all-io-standalone.html");

const readText = (path) => readFileSync(join(here, path), "utf8");
const readBase64 = (path) => readFileSync(join(here, path)).toString("base64");
const scriptSafe = (value) => value.replace(/<\/(script)/gi, "<\\/$1");

const html = readText("index.html");
const css = readText("styles.css");
const appJs = readText("app.js");
const builderJs = readText("builder.js");

// Shared nirs4all-ui browser entry: the exact bytes staged by the release build
// (one UI build, one copy per host). This script only embeds them.
const uiJs = readText("vendor/ui/nirs4all-ui-browser.iife.js");
const uiCss = readText("vendor/ui/browser.css");
// Licence/notice FILES among the vendor/ui descendants only (nested ones included,
// directories ignored): a file named like a notice/licence, or anything under LICENSES/.
function stagedNoticeFiles(dir = "") {
  const found = [];
  for (const entry of readdirSync(join(here, "vendor", "ui", dir), { withFileTypes: true })) {
    const rel = dir ? `${dir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) found.push(...stagedNoticeFiles(rel));
    else if (entry.isFile() && (/NOTICE|LICEN[CS]E|LICENSING/i.test(entry.name) || rel.startsWith("LICENSES/"))) found.push(rel);
  }
  return found;
}
const uiNoticeFiles = stagedNoticeFiles().sort();
if (!uiNoticeFiles.length) throw new Error("No notice/licence file is staged next to the shared UI bundle");
const uiNotices = uiNoticeFiles.map((name) => `=== ${name} ===\n${readText(`vendor/ui/${name}`)}`).join("\n\n");

const wasmBundles = {
  formats: {
    js: readText("pkg/formats/nirs4all_formats_wasm.js"),
    wasm: readBase64("pkg/formats/nirs4all_formats_wasm_bg.wasm"),
  },
  io: {
    js: readText("pkg/io/nirs4all_io_wasm.js"),
    wasm: readBase64("pkg/io/nirs4all_io_wasm_bg.wasm"),
  },
};
const brandAssets = {
  iconSvg: `data:image/svg+xml;base64,${readBase64("brand/icon.svg")}`,
  favicon: `data:image/x-icon;base64,${readBase64("brand/favicon.ico")}`,
  icon180: `data:image/png;base64,${readBase64("brand/icon-180.png")}`,
};

const manifest = JSON.parse(readText("samples/manifest.json"));
const sampleFiles = manifest.files.map((entry) => ({
  name: entry.name,
  bytes: readBase64(`samples/${entry.name}`),
}));

function replaceOne(source, pattern, replacement, label) {
  const next = source.replace(pattern, () => replacement);
  if (next === source) throw new Error(`Could not replace ${label}`);
  return next;
}

// Replace one literal marker exactly once; a missing or repeated marker is a build error.
function replaceMarker(source, marker, replacement, label) {
  const count = source.split(marker).length - 1;
  if (count !== 1) throw new Error(`Expected exactly one ${label} marker, found ${count}`);
  return replaceOne(source, marker, replacement, label);
}

// Inline CSS must not close its own <style>; scriptSafe only protects script blocks.
for (const [label, text] of [["styles.css", css], ["shared UI CSS", uiCss]]) {
  if (/<\/style/i.test(text)) throw new Error(`${label} contains </style and cannot be inlined`);
}
// The notices ride in an inert text/plain block, embedded verbatim: refuse a closer.
if (/<\/script/i.test(uiNotices)) throw new Error("UI notices contain </script and cannot be inlined");
// '<!--' before '<script' puts the HTML parser in the double-escaped script state,
// which the closing-tag escape below does not repair.
if (/<!--[\s\S]*<script/i.test(uiJs)) throw new Error("Shared UI bundle contains '<!--' before '<script'");

const standaloneBootstrap = `
const wasmSources = {
  formats: ${JSON.stringify(scriptSafe(wasmBundles.formats.js))},
  io: ${JSON.stringify(scriptSafe(wasmBundles.io.js))}
};
const wasmPayloads = {
  formats: ${JSON.stringify(wasmBundles.formats.wasm)},
  io: ${JSON.stringify(wasmBundles.io.wasm)}
};
const embeddedSamples = ${JSON.stringify(sampleFiles)};

const bytesCache = new Map();
const moduleUrlCache = new Map();

function base64ToBytes(base64) {
  const cached = bytesCache.get(base64);
  if (cached) return new Uint8Array(cached);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  bytesCache.set(base64, bytes);
  return new Uint8Array(bytes);
}

function moduleUrl(name) {
  if (!moduleUrlCache.has(name)) {
    moduleUrlCache.set(name, URL.createObjectURL(new Blob([wasmSources[name]], { type: "text/javascript" })));
  }
  return moduleUrlCache.get(name);
}

async function loadWasmModule(name) {
  const mod = await import(moduleUrl(name));
  const wasmBytes = base64ToBytes(wasmPayloads[name]);
  await mod.default({ module_or_path: new Response(wasmBytes, { headers: { "Content-Type": "application/wasm" } }) });
  return mod;
}

window.NIRS4ALL_STANDALONE = {
  async loadWasmModules() {
    return Promise.all([loadWasmModule("formats"), loadWasmModule("io")]);
  },
  sampleFiles() {
    return embeddedSamples.map((file) => ({ name: file.name, bytes: base64ToBytes(file.bytes) }));
  }
};
`;

let standalone = html;
standalone = replaceMarker(
  standalone,
  '<link rel="stylesheet" href="./vendor/ui/browser.css" />',
  `<style>\n/* nirs4all-ui browser entry */\n${uiCss}\n  </style>`,
  "shared UI stylesheet",
);
// Classic IIFE, before every module script (and before the page script that mounts it).
standalone = replaceMarker(
  standalone,
  '<script src="./vendor/ui/nirs4all-ui-browser.iife.js"></script>',
  `<script type="text/plain" id="nirs4all-ui-notices">\n${uiNotices}\n  </script>\n  <script>\n/* nirs4all-ui browser entry (see notices above) */\n${scriptSafe(uiJs)}\n  </script>`,
  "shared UI script",
);
standalone = replaceOne(
  standalone,
  /\s*<link rel="preconnect" href="https:\/\/fonts\.googleapis\.com" \/>\n\s*<link rel="preconnect" href="https:\/\/fonts\.gstatic\.com" crossorigin \/>\n\s*<link href="https:\/\/fonts\.googleapis\.com[^"]+" rel="stylesheet" \/>\n/,
  "\n",
  "Google font links"
);
standalone = replaceOne(
  standalone,
  /\s*<link rel="stylesheet" href="\.\/styles\.css" \/>\n/,
  `\n  <style>\n${css}\n  </style>\n`,
  "stylesheet link"
);
standalone = replaceOne(
  standalone,
  /\s*<script type="module" src="\.\/app\.js"><\/script>\n\s*<script type="module" src="\.\/builder\.js"><\/script>\n/,
  `\n  <script type="module">\n${scriptSafe(standaloneBootstrap)}\n  </script>\n  <script type="module">\n${scriptSafe(appJs)}\n  </script>\n  <script type="module">\n${scriptSafe(builderJs)}\n  </script>\n`,
  "app + builder scripts"
);
standalone = standalone
  .replaceAll('href="brand/icon.svg"', `href="${brandAssets.iconSvg}"`)
  .replaceAll('href="brand/favicon.ico"', `href="${brandAssets.favicon}"`)
  .replaceAll('href="brand/icon-180.png"', `href="${brandAssets.icon180}"`)
  .replaceAll('src="brand/icon.svg"', `src="${brandAssets.iconSvg}"`);
standalone = standalone.replace(
  /\s*<!-- GoatCounter privacy-friendly analytics \(no cookies\) -->\n\s*<script data-goatcounter="https:\/\/nirs4all\.goatcounter\.com\/count" data-goatcounter-settings='\{"path": "\/io"\}' async src="\/\/gc\.zgo\.at\/count\.js"><\/script>\n/,
  "\n",
);
standalone = standalone.replace(
  /<title>nirs4all-io · WASM dataset builder<\/title>/,
  "<title>nirs4all-io · standalone WASM dataset builder</title>"
);
// The verbatim notice texts may mention vendor/ui/ in prose; only a reference outside them is a load.
if (standalone.replace(uiNotices, "").includes("vendor/ui")) throw new Error("Standalone output still references the staged UI files");
standalone = `<!-- Generated by ${basename(fileURLToPath(import.meta.url))}. Do not edit this file directly. -->\n${standalone}`;

mkdirSync(outDir, { recursive: true });
writeFileSync(outFile, standalone);
console.log(`Standalone HTML written to ${outFile}`);
