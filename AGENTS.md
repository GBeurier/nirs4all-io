# IO contributor instructions

## Ownership and entrypoints

- IO owns RESOLVE -> INFER -> CONFIGURE -> MATERIALIZE: roles, sources/joins, axes/units/masks, identity, targets, declared deterministic partitions and signal/task inference. DatasetSpec is the validated serializable IR.
- [Formats](../nirs4all-formats/AGENTS.md) decodes vendor bytes; [DMD](../dag-ml-data/AGENTS.md) owns generic data/identity constraints; [DAG](../dag-ml/AGENTS.md) owns fold scheduling; [Methods](../nirs4all-methods/AGENTS.md) owns splitter algorithms.
- Sibling links are optional navigation.
- `nirs4all-io-core` is pure logic over neutral descriptions: no filesystem or Formats dependency.
- The facade owns file access/descriptions/vendor calls; the bridge crate owns DMD emit.
- Keep assembly target-agnostic before optional SpectroDataset adaptation.
- Read [API](docs/API.md), [dataset configurations](docs/DATASET_CONFIGURATIONS.md), [development](docs/development.md), [versioning](docs/VERSIONING.md), [binding matrix](COMPAT.md) and the owned [operations note](docs/AGENT_OPERATIONS.md).

## Load-bearing boundaries

- Preserve the Python MVP as an expressly identified dev/parity oracle.
- The private root oracle distribution is not the published pyo3 binding.
- Importing IO must not import the SDK; only the explicit SpectroDataset adapter may lazily do so at call time.
- Delegate vendor parsing lazily to Formats. Retain COPY_PROVENANCE/license records for copied tabular/NA logic and update them when more logic is copied; do not rederive a divergent loader.
- Partitions by column/index/index_file are deterministic. Reject random, percentage, stratified and shuffled splitting here; DAG schedules folds and Methods owns splitter algorithms.
- Keep the named Python/Rust IR canonical-JSON profiles and shared goldens byte-identical; this is not universal cross-language in-memory equality. Keep declared key/map ordering, float roundtrip/pyfmt and nonrecursive profile resolution. Refuse an ambiguous used file reference that matches distinct physical identities before reading; unassigned duplicate names retain warnings, repeated one-identity inputs keep operation-specific semantics.
- Update both owners of the oracle/port logic for intended changes, not goldens merely to pass.
- Keep convention TOML copies synchronized.
- Separate tagged public-content identity from IR canonical JSON. Rust `DatasetPackage::to_assembled` preserves canonical masked `+0`/false storage; the Python ndarray view and raw IR retain logical NaN/false. Observed signed-zero and subnormal bits remain intact in selected numeric storage and IR canonical JSON. Tagged `nirs4all.io.dataset-content.v1` retains its declared zero folding; join/group keys preserve numeric identity, including matching `0.0` and `-0.0`. F32 selection rounds or refuses as declared. Compare each representation against its own contract, with independent expected-value, named-profile and identity checks.
- Preserve multi-source/sample/group/repetition/origin alignment, units/masks and precision; inference scores are uncalibrated ranking/triage.
- Multimodal/ragged/provider paths preserve tensors, boundaries and finite/resumable identities; they never encode, impute, fit, construct CV folds or invoke ML frameworks. Those operations belong to runtime owners.
- Preserve the DMD bridge's supported contract and explicit errors/refusals; verify named current profiles rather than replacing a real path with a stub or claiming all routes qualified. Discover current decoder, binding, task and wire capabilities from owner manifests, API/version contracts, the binding matrix and executed evidence for the requested profile/operation/artifact/platform; dated cohorts, counts or acceptance IDs do not establish support.

## Gates and conventions

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace --no-default-features
```

- Python oracle: package venv, `ruff check .`, `mypy .`, `pytest`; preserve bindings exclusion from root mypy and `parity`/`formats` markers. Workspace Rust checks exclude bindings: run their maturin/WASM/R/MATLAB gates separately.
- New load-supported vocabulary is unshipped until a cookbook CATALOGUE fixture covers it; retain inference precision corpus, adversarial SpecError, stage/e2e and intended byte-parity checks. Detailed invocations remain in operations.
- Rust 2021/Python 3.11+, configured Ruff/mypy and CeCILL-2.1 OR AGPL-3.0-or-later rules remain. Version sync uses Cargo workspace version and owner script; schema/profile/canonical-JSON dimensions remain independently versioned.
- Python has no CLI: use the Rust binary. Distinguish oracle and pyo3 packages; YAML uses lists, not tuples. Capture actual raw exit status; RTK/filtered output or chained commands do not prove success.
- Keep clean-V1 removals within authorized migrated/parity-established callers; preserve the oracle. Browser-builder commands in operations are references, not permission for frontend changes. Never force-add private history.

## Working rules

- Read applicable local and deeper instructions before editing. Preserve dirty/concurrent work; make authorized owner fixes and propagate affected contracts/bindings/consumers. Do not reset another author's work.
- Use CodeGraph FIRST for implementation navigation only if this checkout's root already has `.codegraph/`; use its exact projectPath or run the CLI here. Never auto-index or silently substitute an ancestor index.
- Unindexed code and documentation use ordinary source navigation. Explicit source, tests and build tools are final authority. Owner links are navigation; this policy works without a parent workspace.
- All application/project content is English: UI text/subtitles/tooltips/errors, docs/comments/docstrings, test/fixture prose, reports/issues/PRs, commits and generated material. Preserve identifiers, quotations and historical/provenance meaning; converse with the maintainer in French.
- Do not commit credentials, private data, generated workspaces, caches or build outputs. Follow local licensing/fixture restrictions and report actual commands/exits/skips; a command reference grants no execution/publication permission.
