<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# cran-comments.md — nirs4allio

Maintainer: Gregory Beurier (CIRAD) <gregory.beurier@cirad.fr>

> **TEMPLATE — not yet CRAN-ready.** This package is the R binding of
> `nirs4all-io`, a Rust-first dataset-assembly bridge for the nirs4all
> ecosystem. The R binding is a thin **C shim** (`src/n4io.c`) that links the
> **prebuilt** `nirs4all-io-capi` shared library (`libnirs4all_io_capi`) via the
> `N4IO_INCLUDE` / `N4IO_CAPI_DIR` environment variables (see `src/Makevars`).
>
> **Honest CRAN-self-containment note (follow-up tracked).** In its current
> form this package is **NOT CRAN-submittable**: CRAN's build farm has no
> prebuilt `libnirs4all_io_capi`, and the package does not vendor or compile the
> Rust core at install time. The tarball produced by `release-r.yml`
> (`nirs4allio_<version>.tar.gz`) is therefore an **R-universe / GitHub-Release
> asset only**. A CRAN-submittable variant requires **reworking the binding** to
> bundle and compile the Rust core offline at install time (extendr-static or a
> vendored cdylib build, mirroring the `nirs4all-formats` `./configure`
> vendor-mode tarball). That rework is a tracked follow-up; until it lands the R
> binding ships via **R-universe** and the **GitHub Release**, not CRAN.

## Package summary

`nirs4allio` exposes the stable `n4io_*` C ABI of `nirs4all-io` to R: normalize
arbitrary inputs into a canonical `DatasetSpec` (`n4io_to_spec`), infer a
`DatasetPlan` (`n4io_infer`), validate a `DatasetSpec` (`n4io_validate`), and
report the ABI version (`n4io_abi_version`). The JSON surface crosses the C ABI;
results are canonical JSON strings. License: `CeCILL-2.1 | AGPL-3`.

## Test environments (R-universe / Release path)

* local Ubuntu / WSL2, R release — `bindings/r/build_and_test.sh`: builds the
  cdylib, installs against it, smoke passes (`R binding smoke OK`).
* CI matrix (`.github/workflows/release-r.yml`): Ubuntu 22.04 (R release + devel),
  macOS 14 (R release, arm64), Windows Server 2022 (R release) — install + smoke.

## R CMD check status

Not yet run as `--as-cran` against a self-contained tarball, because the package
is **not CRAN-self-contained** in its current form (it links a prebuilt cdylib).
`R CMD build bindings/r` produces a loadable tarball for R-universe / the Release;
a `--as-cran`-clean tarball follows the binding rework described above.

## Paste-ready CRAN submission comment (for the future CRAN-ready variant)

> Once the binding is reworked to vendor + compile the Rust core offline at
> install time, submit only `nirs4allio_<version>.tar.gz` at
> <https://cran.r-project.org/submit.html> with the comment below.

```text
This is a new submission.

nirs4allio is a thin R binding for the Rust-first nirs4all-io dataset-assembly
bridge for the nirs4all NIRS / spectroscopy ecosystem. It exposes the stable
n4io_* C ABI to R: normalize arbitrary inputs into a canonical DatasetSpec,
infer a DatasetPlan, and validate a DatasetSpec; the JSON surface crosses the
C ABI and results are canonical JSON strings. License: CeCILL-2.1 | AGPL-3.

Self-contained source tarball: the package vendors the nirs4all-io Rust core and
its crates.io transitive dependencies and compiles them OFFLINE at install time
via src/Makevars(.win) (no network, no external monorepo crates/). The Cargo /
rustc toolchain is declared in SystemRequirements.

Test environments: local Ubuntu/WSL2 R release (offline standalone install ->
installs, loads, native path active); CI matrix (release-r.yml) on Ubuntu 22.04
(R release + devel), macOS 14 (R release, arm64), Windows Server 2022 (R
release); win-builder + R-hub v2 run manually before submission.

R CMD check --as-cran: 0 ERRORs. Any WARNING/NOTE comes from the bundled
third-party Rust sources or the local toolchain, not from the package's own R or
build logic. The package does no network access during install/examples/tests
and imports only base R.

Maintainer: Grégory Beurier (CIRAD), gregory.beurier@cirad.fr.
```

> **CRAN version note:** CRAN rejects SemVer pre-release suffixes
> (`0.1.0-alpha.0`). While the project is pre-`0.1.0` the R spelling is the
> development version `0.1.0.9000`, which is **R-universe / dev only and is NOT
> submitted to CRAN**. The first CRAN-eligible R version is the plain `0.1.0`
> cut by `scripts/bump_version.sh --bump 0.1.0` — and only after the
> self-containment rework above.
