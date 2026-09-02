#!/usr/bin/env node
// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Stage the authored JS/types/legal surface into a wasm-pack package.

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasm = path.join(root, "bindings/wasm");
const legalTopLevel = ["LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md", "COPY_PROVENANCE.md"];
const legalMirrors = [
  "bindings/python",
  "bindings/r/inst",
  "bindings/wasm",
  "crates/nirs4all-io-core",
  "crates/nirs4all-io",
  "crates/nirs4all-io-capi",
  "crates/nirs4all-io-cli",
];
const wasmLicenseClosureChecksum = "6ebddd95f465a1cccc52c3cf4dd941357c9a24f8667d90aa30c5004d1c393770";
const lockedLicenseSources = [
  {
    packageName: "ryu",
    version: "1.0.23",
    crateChecksum: "9774ba4a74de5f7b1c1451ed6cd5285a32eddb5cccb8cc655a4e50009e06477f",
    licenseFile: "Apache-2.0.txt",
    licenseChecksum: "62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a",
  },
  {
    packageName: "unicode-ident",
    version: "1.0.24",
    crateChecksum: "e6e4313cd5fcd3dad5cafa179702e2b244f760991f45397d14d4ebf38247da75",
    licenseFile: "Unicode-3.0.txt",
    licenseChecksum: "f7db81051789b729fea528a63ec4c938fdcb93d9d61d97dc8cc2e9df6d47f2a1",
  },
];

function regularFileNames(directory) {
  return fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .sort();
}

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function verifyWasmLicenseClosure() {
  const result = spawnSync(
    "cargo",
    [
      "metadata",
      "--manifest-path",
      "bindings/wasm/Cargo.toml",
      "--locked",
      "--offline",
      "--format-version=1",
    ],
    { cwd: root, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  );
  if (result.status !== 0) {
    throw new Error(`could not audit the locked WASM license closure: ${result.stderr.trim()}`);
  }
  const metadata = JSON.parse(result.stdout);
  const lines = metadata.packages
    .map((pkg) => `${pkg.name}@${pkg.version}|${pkg.license ?? "NO-LICENSE"}`)
    .sort();
  const checksum = sha256(Buffer.from(`${lines.join("\n")}\n`));
  if (checksum !== wasmLicenseClosureChecksum) {
    throw new Error(
      `locked WASM license closure changed (${checksum}); audit every license expression before staging`,
    );
  }
}

function verifyLockedLicenseSources() {
  const lock = fs.readFileSync(path.join(wasm, "Cargo.lock"), "utf8");
  const packageBlocks = lock.split("[[package]]").slice(1);
  for (const source of lockedLicenseSources) {
    const blocks = packageBlocks.filter((block) => {
      const name = block.match(/^name = "([^"]+)"$/m)?.[1];
      return name === source.packageName;
    });
    if (blocks.length !== 1) {
      throw new Error(`expected exactly one ${source.packageName} package in bindings/wasm/Cargo.lock`);
    }
    const version = blocks[0].match(/^version = "([^"]+)"$/m)?.[1];
    const crateChecksum = blocks[0].match(/^checksum = "([0-9a-f]+)"$/m)?.[1];
    if (version !== source.version || crateChecksum !== source.crateChecksum) {
      throw new Error(
        `${source.packageName} lock identity changed; re-audit its selected license before staging`,
      );
    }
    const license = fs.readFileSync(path.join(root, "LICENSES", source.licenseFile));
    if (sha256(license) !== source.licenseChecksum) {
      throw new Error(
        `LICENSES/${source.licenseFile} is not the audited ${source.packageName} ${source.version} upstream text`,
      );
    }
  }
}

function verifyLegalMirror(surface, expectedLicenses) {
  const surfaceRoot = path.join(root, surface);
  const surfaceLicenseDir = path.join(surfaceRoot, "LICENSES");
  const actualLicenses = regularFileNames(surfaceLicenseDir);
  if (JSON.stringify(actualLicenses) !== JSON.stringify(expectedLicenses)) {
    throw new Error(
      `${surface}/LICENSES must mirror root LICENSES exactly; expected ${expectedLicenses.join(", ")}, got ${actualLicenses.join(", ")}`,
    );
  }

  for (const relative of [
    ...legalTopLevel,
    ...expectedLicenses.map((name) => path.join("LICENSES", name)),
  ]) {
    const canonical = fs.readFileSync(path.join(root, relative));
    const authored = fs.readFileSync(path.join(surfaceRoot, relative));
    if (!canonical.equals(authored)) {
      throw new Error(`${surface}/${relative} differs from canonical root ${relative}`);
    }
  }
}

function verifyReleaseLegalMirrors() {
  verifyWasmLicenseClosure();
  verifyLockedLicenseSources();
  const rootLicenseDir = path.join(root, "LICENSES");
  const expectedLicenses = regularFileNames(rootLicenseDir);
  for (const surface of legalMirrors) verifyLegalMirror(surface, expectedLicenses);
  const referencedLicenseFiles = new Set();
  for (const name of ["LICENSING.md", "THIRD_PARTY_NOTICES.md"]) {
    const text = fs.readFileSync(path.join(root, name), "utf8");
    for (const match of text.matchAll(/LICENSES\/([A-Za-z0-9._-]+)/g)) {
      referencedLicenseFiles.add(match[1]);
    }
  }
  for (const name of referencedLicenseFiles) {
    if (!expectedLicenses.includes(name)) {
      throw new Error(`canonical legal documents reference missing LICENSES/${name}`);
    }
  }

  return expectedLicenses;
}

const expectedLicenses = verifyReleaseLegalMirrors();
if (process.argv.includes("--check-legal")) {
  console.log(
    `verified ${legalMirrors.length} release legal mirrors (${expectedLicenses.length} license texts)`,
  );
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
