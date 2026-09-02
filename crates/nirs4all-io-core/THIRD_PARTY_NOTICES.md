# Third-Party Notices — nirs4all-io

`nirs4all-io` is distributed under `CeCILL-2.1 OR AGPL-3.0-or-later` (plus an optional
commercial license; see [`LICENSING.md`](LICENSING.md)). It relies on the third-party
open-source components listed below.

The distribution model differs by surface and must not be described as uniformly
"unvendored":

- the pure-Python compatibility package installs its Python dependencies separately;
- Rust executables, native libraries, Python wheels, and WASM packages contain compiled
  Rust dependencies from the exact closure pinned by `Cargo.lock` (or the binding lock);
- the R source package deliberately vendors that Rust source closure for an offline CRAN
  build. Its `src/rust/vendor.tar.xz` retains every upstream crate's own license and notice
  files.

The release CycloneDX SBOM is the machine-readable component/version/license inventory for
the shipped Rust closure. `Cargo.lock`, `bindings/python/Cargo.lock`,
`bindings/wasm/Cargo.lock`, and `bindings/r/Cargo.lock.rust` are the authoritative version
locks. This notice summarizes direct runtime dependencies; it does not replace the
transitive inventory, upstream notices in the R vendor bundle, or the SBOM.

## Python runtime dependencies

| Component | License (SPDX) | Copyright | Upstream |
|---|---|---|---|
| `numpy` | BSD-3-Clause | Copyright (c) 2005-2025, NumPy Developers | https://github.com/numpy/numpy |
| `pandas` | BSD-3-Clause | Copyright (c) 2008-2025, pandas Development Team | https://github.com/pandas-dev/pandas |
| `pyyaml` | MIT | Copyright (c) 2006-2025 PyYAML contributors | https://github.com/yaml/pyyaml |
| `jsonschema` | MIT | Copyright (c) 2013 Julian Berman | https://github.com/python-jsonschema/jsonschema |

## Direct Rust runtime dependencies

| Component | License expression | Upstream |
|---|---|---|
| `dag-ml-data` | `CeCILL-2.1 OR AGPL-3.0-or-later` | https://github.com/GBeurier/dag-ml-data |
| `serde`, `serde_json` | `MIT OR Apache-2.0` | https://github.com/serde-rs |
| `thiserror`, `anyhow` | `MIT OR Apache-2.0` | https://github.com/dtolnay |
| `sha2` | `MIT OR Apache-2.0` | https://github.com/RustCrypto/hashes |
| `regex` | `MIT OR Apache-2.0` | https://github.com/rust-lang/regex |
| `toml` | `MIT OR Apache-2.0` | https://github.com/toml-rs/toml |
| `indexmap` | `Apache-2.0 OR MIT` | https://github.com/indexmap-rs/indexmap |
| `csv` | `Unlicense OR MIT` | https://github.com/BurntSushi/rust-csv |
| `glob` | `MIT OR Apache-2.0` | https://github.com/rust-lang/glob |
| `flate2` | `MIT OR Apache-2.0` | https://github.com/rust-lang/flate2-rs |
| `zip` | `MIT` | https://github.com/zip-rs/zip2 |
| Apache Arrow Rust (`arrow-*`, `parquet`) | `Apache-2.0` and, for `arrow-array`, `Apache-2.0 AND MIT` | https://github.com/apache/arrow-rs |
| `clap` | `MIT OR Apache-2.0` | https://github.com/clap-rs/clap |
| `wasm-bindgen`, `js-sys` | `MIT OR Apache-2.0` | https://github.com/wasm-bindgen/wasm-bindgen |

## Bundled license-family texts (`LICENSES/`)

- `LICENSES/BSD-3-Clause.txt` — BSD-3-Clause
- `LICENSES/MIT.txt` — MIT

Additional license families and copyright notices in the transitive Rust closure are
preserved in the upstream crate sources inside the self-contained R vendor archive and are
enumerated in the CycloneDX SBOM. Binary redistributors must ship this notice, the release
SBOM, and the applicable upstream license files together with the binary artifact.
