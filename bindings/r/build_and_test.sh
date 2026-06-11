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
    rust_target="x86_64-pc-windows-gnu"
    rustup target add "${rust_target}" >/dev/null 2>&1 || true
    target_flag="--target ${rust_target}"
    capi_dir="${io_root}/target/${rust_target}/release"
    # `rustup target add` installs the gnu std but NOT a gnu linker. Bind rustc's
    # windows-gnu linker to Rtools' mingw gcc so we don't depend on a bare `gcc`
    # happening to be first on PATH (and so it's the SAME compiler that links the
    # R package — one ABI/CRT end to end). Prefer the explicit rtools name, fall
    # back to whatever `gcc` resolves to.
    rtools_gcc="$(command -v x86_64-w64-mingw32.static.posix-gcc || command -v gcc || true)"
    if [ -n "${rtools_gcc}" ]; then
      export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="${rtools_gcc}"
    fi
    ;;
  *)
    target_flag=""
    capi_dir="${io_root}/target/release"
    ;;
esac

echo ">> building nirs4all-io-capi (release) ${target_flag}"
( cd "${io_root}" && cargo build -q -p nirs4all-io-capi --release ${target_flag} )

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
