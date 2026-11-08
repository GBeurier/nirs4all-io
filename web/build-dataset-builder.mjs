#!/usr/bin/env node
/*
 * Build the column-mapping host bundle used by dataset-builder.html.
 *
 * Input: the accepted, unpublished nirs4all-ui `.1` package tarball. The tarball's SHA-256 is
 * verified first, then it is extracted into a fresh temporary directory; the dirty sibling
 * `nirs4all-ui/dist` is never read. esbuild bundles builder-app/main.tsx (three public
 * `dist/datasetBuilder` components behind the alias `n4ui-datasetbuilder`) into
 * vendor/ui-columns/column-mapping.mjs and copies the package stylesheet verbatim.
 *
 * React, React DOM and esbuild are resolved from a node_modules directory passed on the
 * command line (this repository has none); the versions used are recorded in BUILD.json.
 *
 * Usage:
 *   node web/build-dataset-builder.mjs --tgz <nirs4all-ui-....tgz> --node-modules <dir>
 * Optional: --sha256 <hex> overrides the expected tarball digest (default: the accepted `.1`).
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ACCEPTED_SHA256 = "2f9c7847886beefde05a661481b784b544dd36d1c003cea896c1a76659501b70";
const here = dirname(fileURLToPath(import.meta.url));

function option(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const tgz = option("--tgz");
const nodeModules = option("--node-modules");
const expected = option("--sha256") ?? ACCEPTED_SHA256;
if (!tgz || !nodeModules) {
  console.error("usage: node build-dataset-builder.mjs --tgz <package.tgz> --node-modules <dir> [--sha256 <hex>]");
  process.exit(2);
}

const digest = createHash("sha256").update(readFileSync(resolve(tgz))).digest("hex");
if (digest !== expected) throw new Error(`package digest ${digest} differs from the accepted ${expected}`);

const modulesDir = resolve(nodeModules);
const require = createRequire(join(modulesDir, "index.js"));
const esbuild = require("esbuild");
const versionOf = (name) => JSON.parse(readFileSync(join(modulesDir, name, "package.json"), "utf8")).version;

const scratch = mkdtempSync(join(tmpdir(), "n4io-ui-columns-"));
try {
  execFileSync("tar", ["-xzf", resolve(tgz), "-C", scratch], { stdio: "inherit" });
  const pkg = join(scratch, "package");
  const components = join(pkg, "dist", "datasetBuilder");
  for (const file of ["ColumnMappingTable.js", "RoleSelectionCards.js", "SourceSummaryCard.js"]) {
    if (!existsSync(join(components, file))) throw new Error(`accepted package lacks dist/datasetBuilder/${file}`);
  }

  const outDir = resolve(here, "vendor", "ui-columns");
  mkdirSync(outDir, { recursive: true });
  const outfile = join(outDir, "column-mapping.mjs");
  await esbuild.build({
    entryPoints: [resolve(here, "builder-app", "main.tsx")],
    bundle: true,
    format: "esm",
    outfile,
    jsx: "automatic",
    minify: true,
    target: ["es2020"],
    alias: { "n4ui-datasetbuilder": components },
    nodePaths: [modulesDir],
    define: { "process.env.NODE_ENV": '"production"' },
    legalComments: "none",
    logLevel: "info",
  });
  copyFileSync(join(pkg, "assets", "datasetBuilder.css"), join(outDir, "datasetBuilder.css"));

  const sha = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");
  writeFileSync(
    join(outDir, "BUILD.json"),
    `${JSON.stringify({
      package_tgz_sha256: digest,
      package_version: JSON.parse(readFileSync(join(pkg, "package.json"), "utf8")).version,
      bundle_sha256: sha(outfile),
      stylesheet_sha256: sha(join(outDir, "datasetBuilder.css")),
      react: versionOf("react"),
      react_dom: versionOf("react-dom"),
      scheduler: versionOf("scheduler"),
      esbuild: versionOf("esbuild"),
      components: ["SourceSummaryCard", "RoleSelectionCards", "ColumnMappingTable"],
      licences: "React, React DOM and scheduler (MIT) texts are staged under ../ui/LICENSES/THIRD-PARTY/",
    }, null, 2)}\n`,
  );
  console.log("Column-mapping bundle written to web/vendor/ui-columns/");
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
