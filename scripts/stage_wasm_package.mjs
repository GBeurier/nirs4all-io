#!/usr/bin/env node
// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Stage the authored JS/types/legal surface into a wasm-pack package.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasm = path.join(root, "bindings/wasm");
const legalTopLevel = ["LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md", "COPY_PROVENANCE.md"];

function regularFileNames(directory) {
  return fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
}

function verifyLegalMirror() {
  const rootLicenseDir = path.join(root, "LICENSES");
  const wasmLicenseDir = path.join(wasm, "LICENSES");
  const expectedLicenses = regularFileNames(rootLicenseDir);
  const actualLicenses = regularFileNames(wasmLicenseDir);
  if (JSON.stringify(actualLicenses) !== JSON.stringify(expectedLicenses)) {
    throw new Error(
      `bindings/wasm/LICENSES must mirror root LICENSES exactly; expected ${expectedLicenses.join(", ")}, got ${actualLicenses.join(", ")}`,
    );
  }

  for (const relative of [
    ...legalTopLevel,
    ...expectedLicenses.map((name) => path.join("LICENSES", name)),
  ]) {
    const canonical = fs.readFileSync(path.join(root, relative));
    const authored = fs.readFileSync(path.join(wasm, relative));
    if (!canonical.equals(authored)) {
      throw new Error(`bindings/wasm/${relative} differs from canonical root ${relative}`);
    }
  }

  const referencedLicenseFiles = new Set();
  for (const name of ["LICENSING.md", "THIRD_PARTY_NOTICES.md"]) {
    const text = fs.readFileSync(path.join(wasm, name), "utf8");
    for (const match of text.matchAll(/LICENSES\/([A-Za-z0-9._-]+)/g)) {
      referencedLicenseFiles.add(match[1]);
    }
  }
  for (const name of referencedLicenseFiles) {
    if (!actualLicenses.includes(name)) {
      throw new Error(`bindings/wasm legal documents reference missing LICENSES/${name}`);
    }
  }

  return expectedLicenses;
}

const expectedLicenses = verifyLegalMirror();
if (process.argv.includes("--check-legal")) {
  console.log(`verified WASM legal mirror (${expectedLicenses.length} license texts)`);
  process.exit(0);
}

const pkgDirArg = process.argv.slice(2).find((argument) => !argument.startsWith("--"));
const pkgDir = path.resolve(pkgDirArg ?? path.join(root, "bindings/wasm/pkg-node"));
const cargo = fs.readFileSync(path.join(root, "Cargo.toml"), "utf8");
const workspace = cargo.match(/\[workspace\.package\][\s\S]*?^version\s*=\s*"([^"]+)"/m);
if (!workspace) throw new Error("could not read [workspace.package] version");

const packagePath = path.join(pkgDir, "package.json");
const manifest = JSON.parse(fs.readFileSync(packagePath, "utf8"));
manifest.name = process.env.NPM_PKG_NAME ?? "@nirs4all/io-wasm";
manifest.version = workspace[1];
manifest.publishConfig = { access: "public", provenance: true };
manifest.repository = {
  type: "git",
  url: "https://github.com/GBeurier/nirs4all-io.git",
  directory: "bindings/wasm",
};
manifest.files = [...new Set([
  ...(manifest.files ?? []),
  "idiomatic.mjs",
  "idiomatic.d.ts",
  "types/nirs4all-io.d.ts",
  "LICENSE",
  "LICENSES",
  "LICENSING.md",
  "THIRD_PARTY_NOTICES.md",
  "COPY_PROVENANCE.md",
])];
manifest.exports = {
  ".": {
    types: "./nirs4all_io_wasm.d.ts",
    require: "./nirs4all_io_wasm.js",
    default: "./nirs4all_io_wasm.js",
  },
  "./idiomatic": {
    types: "./idiomatic.d.ts",
    import: "./idiomatic.mjs",
    default: "./idiomatic.mjs",
  },
  "./types": { types: "./types/nirs4all-io.d.ts" },
};

fs.mkdirSync(path.join(pkgDir, "types"), { recursive: true });
fs.rmSync(path.join(pkgDir, "LICENSES"), { recursive: true, force: true });
fs.mkdirSync(path.join(pkgDir, "LICENSES"), { recursive: true });
fs.copyFileSync(path.join(wasm, "idiomatic.d.ts"), path.join(pkgDir, "idiomatic.d.ts"));
fs.copyFileSync(
  path.join(wasm, "types/nirs4all-io.d.ts"),
  path.join(pkgDir, "types/nirs4all-io.d.ts"),
);
const wrapper = fs
  .readFileSync(path.join(wasm, "idiomatic.mjs"), "utf8")
  .replace('"./pkg/nirs4all_io_wasm.js"', '"./nirs4all_io_wasm.js"');
if (wrapper.includes('"./pkg/nirs4all_io_wasm.js"')) {
  throw new Error("failed to retarget the idiomatic wrapper to the staged WASM module");
}
fs.writeFileSync(path.join(pkgDir, "idiomatic.mjs"), wrapper);

for (const name of legalTopLevel) {
  fs.copyFileSync(path.join(wasm, name), path.join(pkgDir, name));
}
for (const name of expectedLicenses) {
  fs.copyFileSync(path.join(wasm, "LICENSES", name), path.join(pkgDir, "LICENSES", name));
}
fs.writeFileSync(packagePath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`staged ${manifest.name}@${manifest.version} in ${pkgDir}`);
