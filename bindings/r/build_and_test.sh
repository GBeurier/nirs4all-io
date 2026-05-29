#!/usr/bin/env bash
# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
# Build the nirs4all-io-capi cdylib, install the R package against it, and run
# the smoke test (EPIC 11.2). Self-contained: computes paths from the repo root.
set -euo pipefail

io_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo ">> building nirs4all-io-capi (release)"
( cd "${io_root}" && cargo build -q -p nirs4all-io-capi --release )

export N4IO_INCLUDE="${io_root}/crates/nirs4all-io-capi/include"
export N4IO_CAPI_DIR="${io_root}/target/release"
export N4IO_CORPUS="${io_root}/tests/goldens/contract/corpus"

# Force a clean compile so a stale/foreign-arch object is never reused.
rm -f "${io_root}/bindings/r/src/"*.o "${io_root}/bindings/r/src/"*.so

lib="${install_lib:-$(mktemp -d)}"
echo ">> R CMD INSTALL -> ${lib}"
R CMD INSTALL --no-multiarch --library="${lib}" "${io_root}/bindings/r"

echo ">> smoke"
R_LIBS_USER="${lib}" Rscript "${io_root}/bindings/r/tests/smoke.R"
