#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
# Build the nirs4all-io-capi cdylib, install the R package against it, and run
# the smoke test (EPIC 11.2). Self-contained: computes paths from the repo root.
set -euo pipefail

io_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# On Windows the R package is linked by Rtools' mingw-w64 gcc (gnu ABI), so the
# Rust capi must be built with the matching x86_64-pc-windows-gnu toolchain — not
# the default MSVC host. An MSVC staticlib linked by mingw ld fails with
# MSVC-mangled / CRT-mismatched symbols (??_7type_info@@6B@, __imp_WSAGetLastError,
# __chkstk). src/Makevars.win supplies the gnu runtime/system libs.
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    is_windows=1
    rust_target="x86_64-pc-windows-gnu"
    rustup target add "${rust_target}" >/dev/null 2>&1 || true
    capi_dir="${io_root}/target/${rust_target}/release"
    ;;
  *)
    is_windows=0
    capi_dir="${io_root}/target/release"
    ;;
esac

if [ "${is_windows}" = 1 ]; then
  # Build ONLY the staticlib crate-type for the gnu target. Two reasons:
  #  - The R package links the staticlib (libnirs4all_io_capi.a) by exact path;
  #    it never needs the cdylib DLL (Windows has no rpath — a self-contained
  #    nirs4allio.dll is what we want).
  #  - `cargo build` would also link the *cdylib*, and rustc's windows-gnu spec
  #    emits `-lgcc_eh`, which Rtools45's mingw does NOT ship (its EH/unwind +
  #    __chkstk live inside libgcc.a, with no separate libgcc_eh.a) — so the
  #    cdylib link fails with "cannot find -lgcc_eh". A staticlib emit is `ar`
  #    archiving only: it invokes no linker, so it sidesteps that entirely.
  echo ">> building nirs4all-io-capi staticlib (release, ${rust_target})"
  ( cd "${io_root}" && cargo rustc -q -p nirs4all-io-capi --release \
      --target "${rust_target}" --crate-type staticlib )
else
  echo ">> building nirs4all-io-capi (release)"
  ( cd "${io_root}" && cargo build -q -p nirs4all-io-capi --release )
fi

export N4IO_INCLUDE="${io_root}/crates/nirs4all-io-capi/include"
export N4IO_CAPI_DIR="${capi_dir}"
export N4IO_CORPUS="${io_root}/tests/goldens/contract/corpus"

# Force a clean compile so a stale/foreign-arch object is never reused
# (.dll is the Windows shared-object).
rm -f "${io_root}/bindings/r/src/"*.o "${io_root}/bindings/r/src/"*.so "${io_root}/bindings/r/src/"*.dll

lib="${install_lib:-$(mktemp -d)}"
echo ">> R CMD INSTALL -> ${lib}"
R CMD INSTALL --no-multiarch --library="${lib}" "${io_root}/bindings/r"

echo ">> smoke"
R_LIBS_USER="${lib}" Rscript "${io_root}/bindings/r/tests/smoke.R"
