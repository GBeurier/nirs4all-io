# Changelog

## 0.3.0 - Unreleased

- Package and export the WASM numeric-coordinate accessor and its declaration.

- Refuse explicit NA `mean`/`median` methods at native spec admission under
  every policy; direct statistics to fitted preprocessing and retain constant,
  row-local, drop and ignore options. The identified Python oracle retains its
  historical behavior: the 16 shared statistical-fill vectors are historical
  oracle requests, not native production parity cases (their frozen expectations
  remain unchanged).

- Preserve selected numeric precision and validity in f64-backed matrices,
  weights and typed dataset payloads. `DatasetPackage::from_assembled` now
  returns `Result`; consumers must handle construction refusals explicitly.
- Require Formats 0.2.15 for native XLSX decoded cells, including the reduced
  R workspace dependency declaration.
- Use the already-declared assembled wire 3 and dataset-package wire 4;
  DatasetSpec, convention and canonical-JSON versions remain unchanged.
- Remove the Linux `N4IO_1` ELF version node. Binaries requiring that node are
  incompatible with this release; public `n4io_*` signatures, C ABI 0.3.0 and
  the fixed SONAME remain unchanged.
