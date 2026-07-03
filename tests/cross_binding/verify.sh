#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
#
# EPIC 12.4 — cross-binding behavioral parity. The same spec, normalized through
# every binding, must yield the IDENTICAL canonical DatasetSpec JSON. Each binding
# is a thin wrapper over one Rust core, so their outputs must agree byte-for-byte.
#
# Covered here (all normalize through the same Rust implementation and compare the
# canonical DatasetSpec JSON):
#   - the CLI            (`nirs4all-io to-spec --spec`)
#   - the Python binding (pyo3 `_native.to_spec(spec_dict)`)
#   - the WASM binding   (`to_spec(spec_json)` under node)
#   - the R binding      (`n4io_to_spec(spec_json)`)
# The pyo3 binding still has its own golden-parity suite
# (`bindings/python/tests/test_parity.py`); this harness includes it when a
# compatible local toolchain is present so CLI+pyo3 machines do not collapse to a
# false SKIP. Each leg SKIPs if its toolchain is missing, so this is green on a
# partial machine and exhaustive in CI.
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

find_python_binding_interpreter() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" - >/dev/null 2>&1 <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

# --- CLI ---
if command -v cargo >/dev/null 2>&1; then
  echo ">> CLI to-spec"
  ( cd "${io_root}" && cargo build -q -p nirs4all-io-cli )
  cli="$(find "${io_root}/target" -maxdepth 2 -name nirs4all-io -type f | head -1)"
  "${cli}" to-spec --spec "${work}/spec.json" > "${work}/cli.out"
  outputs+=("${work}/cli.out"); labels+=("cli")
fi

# --- Python (pyo3) ---
if command -v maturin >/dev/null 2>&1; then
  python_bin="$(find_python_binding_interpreter || true)"
  if [ -n "${python_bin}" ]; then
    echo ">> Python to_spec"
    python_wheels="${work}/python-wheels"
    mkdir -p "${python_wheels}"
    ( cd "${io_root}/bindings/python" && maturin build -q --interpreter "${python_bin}" -o "${python_wheels}" )
    python_wheel="$(find "${python_wheels}" -maxdepth 1 -name '*.whl' -type f | head -1)"
    "${python_bin}" - <<'PY' "${python_wheel}" "${spec}" > "${work}/python.out"
import importlib.util
import json
import pathlib
import sys
import tempfile
import zipfile

wheel = pathlib.Path(sys.argv[1])
spec_json = sys.argv[2]

with tempfile.TemporaryDirectory() as td:
    td_path = pathlib.Path(td)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(td_path)
    native = next((td_path / "nirs4all_io").glob("_native*.so"))
    module_spec = importlib.util.spec_from_file_location("nirs4all_io._native", native)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"unable to load python binding extension from {native}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    output = module.to_spec(json.loads(spec_json), None, None)
    sys.stdout.write(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
PY
    outputs+=("${work}/python.out"); labels+=("python")
  fi
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
  rlib="$(mktemp -d)"
  N4IO_R_LIB="${rlib}" Rscript -e 'lib <- Sys.getenv("N4IO_R_LIB"); repos <- Sys.getenv("N4IO_R_REPOS", "https://cloud.r-project.org"); if (!"jsonlite" %in% rownames(installed.packages(lib.loc = lib))) install.packages("jsonlite", lib = lib, repos = repos)'
  ( cd "${io_root}/bindings/r" && N4IO_R_VENDOR=1 ./configure )
  r_install_log="${work}/r-install.log"
  if ! R_LIBS_USER="${rlib}:${R_LIBS_USER:-}" R CMD INSTALL --no-multiarch --library="${rlib}" "${io_root}/bindings/r" >"${r_install_log}" 2>&1; then
    tail -n 120 "${r_install_log}" >&2
    exit 1
  fi
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
