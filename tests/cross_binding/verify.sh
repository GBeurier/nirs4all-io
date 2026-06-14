#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
#
# EPIC 12.4 — cross-binding behavioral parity. The same spec, normalized through
# every binding, must yield the IDENTICAL canonical DatasetSpec JSON. Each binding
# is a thin wrapper over one Rust core, so their outputs must agree byte-for-byte.
#
# Covered here (all run the C-ABI / core path and emit canonical JSON directly):
#   - the CLI            (`nirs4all-io to-spec --spec`)
#   - the WASM binding   (`to_spec(spec_json)` under node)
#   - the R binding      (`n4io_to_spec(spec_json)`)
# The pyo3 binding's byte-parity is proven separately (bindings/python/tests/
# test_parity.py). Each leg SKIPs if its toolchain is missing, so this is green on
# a partial machine and exhaustive in CI.
set -euo pipefail

io_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# A spec dict (no paths → no abspath divergence): every binding normalizes it
# into the same canonical DatasetSpec.
spec='{"name":"xbind","sources":[{"id":"s","role":"features","input":"a.csv"}]}'
printf '%s' "${spec}" > "${work}/spec.json"

outputs=()
labels=()

# --- CLI ---
if command -v cargo >/dev/null 2>&1; then
  echo ">> CLI to-spec"
  ( cd "${io_root}" && cargo build -q -p nirs4all-io-cli )
  cli="$(find "${io_root}/target" -maxdepth 2 -name nirs4all-io -type f | head -1)"
  "${cli}" to-spec --spec "${work}/spec.json" > "${work}/cli.out"
  outputs+=("${work}/cli.out"); labels+=("cli")
fi

# --- WASM (node) ---
if command -v wasm-pack >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
  echo ">> WASM to_spec"
  wasm-pack build "${io_root}/bindings/wasm" --target nodejs --out-dir pkg >/dev/null 2>&1
  node -e "const w=require('${io_root}/bindings/wasm/pkg/nirs4all_io_wasm.js'); process.stdout.write(w.to_spec(process.argv[1]));" "${spec}" > "${work}/wasm.out"
  outputs+=("${work}/wasm.out"); labels+=("wasm")
fi

# --- R ---
if command -v R >/dev/null 2>&1 && command -v Rscript >/dev/null 2>&1; then
  echo ">> R n4io_to_spec"
  ( cd "${io_root}" && cargo build -q -p nirs4all-io-capi --release )
  rlib="$(mktemp -d)"
  N4IO_R_LIB="${rlib}" Rscript -e 'lib <- Sys.getenv("N4IO_R_LIB"); repos <- Sys.getenv("N4IO_R_REPOS", "https://cloud.r-project.org"); if (!"jsonlite" %in% rownames(installed.packages(lib.loc = lib))) install.packages("jsonlite", lib = lib, repos = repos)'
  N4IO_INCLUDE="${io_root}/crates/nirs4all-io-capi/include" \
  N4IO_CAPI_DIR="${io_root}/target/release" \
  R_LIBS_USER="${rlib}:${R_LIBS_USER:-}" \
    R CMD INSTALL --no-multiarch --library="${rlib}" "${io_root}/bindings/r" >/dev/null 2>&1
  R_LIBS_USER="${rlib}:${R_LIBS_USER:-}" Rscript -e "library(nirs4allio); cat(n4io_to_spec(commandArgs(TRUE)[1]))" "${spec}" > "${work}/r.out"
  outputs+=("${work}/r.out"); labels+=("r")
fi

if [ "${#outputs[@]}" -lt 2 ]; then
  echo "SKIP: fewer than two binding toolchains available (${labels[*]:-none}); cross-binding parity not run."
  exit 0
fi

echo ">> comparing ${labels[*]}"
ref="${outputs[0]}"
for i in "${!outputs[@]}"; do
  if ! diff -u "${ref}" "${outputs[$i]}" >/dev/null; then
    echo "MISMATCH between ${labels[0]} and ${labels[$i]}:"
    diff -u "${ref}" "${outputs[$i]}" || true
    exit 1
  fi
done
echo "ALL ${#outputs[@]} bindings agree byte-for-byte: ${labels[*]}"
