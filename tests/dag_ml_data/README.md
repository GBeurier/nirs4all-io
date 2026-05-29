<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# dag-ml-data emit — cross-CLI conformance (EPIC 10)

io owns the `AssembledDataset → CoordinatorDataPlanEnvelope` bridge (ADR-0001).
The emit lives in the **workspace-excluded** crate `crates/nirs4all-io-dagml`
(`to_dag_ml_data` + the `emit-dagml` binary). It is excluded because it
path-depends on the `dag-ml-data` sibling: an absent optional path dep would
break standalone `cargo build --workspace` resolution (verified), so the main
workspace and standalone CI stay free of any dag-ml-data dependency. The main
`nirs4all-io` CLI keeps an `emit-dag-ml-data` subcommand for discoverability that
points at this crate. io builds a `DatasetSchema` + `DataPlan` +
`SampleRelationTable` and calls `CoordinatorDataPlanEnvelope::from_parts` (which
fingerprints and self-validates). io does **not** emit dag-ml
`FoldSet`/`DataBinding` — those are dag-ml's domain.

## Two layers of verification

1. **In-process** (`crates/nirs4all-io-dagml/tests/emit.rs`,
   `cargo test --manifest-path crates/nirs4all-io-dagml/Cargo.toml`): builds the
   envelope for each contract-corpus case, then JSON-round-trips and
   re-`validate()`s it — exactly the checks `dag-ml-data-cli validate-envelope`
   runs, with no external binary. Needs the dag-ml-data sibling (ecosystem tree).

2. **Cross-CLI** (`verify_cross_cli.sh`): the full ecosystem acceptance (story
   10.4). The io-emitted envelope must pass **both**:
   - `dag-ml-data-cli validate-envelope <envelope.json>` — full envelope.
   - `dag-ml-cli validate-data-binding` — the lossy `ExternalDataPlanEnvelope`
     (schema/plan/relation fingerprints + coordinator relations) wrapped by a
     hand-authored `DataBinding` inside a minimal `CampaignSpec` (no folds, so the
     fold-safety check is a no-op; `require_relations=true` exercises the relation
     contract).

   Fingerprints are content-derived, so nothing brittle is pinned — the "golden"
   is the round trip *emit → both CLIs accept*. The script needs the sibling
   `dag-ml-data` and `dag-ml` repos (override locations with `NIRS4ALL_DAG_ML_DATA`
   / `NIRS4ALL_DAG_ML`); it SKIPs (exit 0) if either is absent, so standalone CI
   is unaffected. In the ecosystem CI all repos are present.

```bash
bash tests/dag_ml_data/verify_cross_cli.sh                 # default: train_test x_y_separate
bash tests/dag_ml_data/verify_cross_cli.sh train_test      # a specific case
```

`single_combined` is inference-only (no convention match), so the CLI emit path
(which loads via conventions) does not cover it; the in-process test exercises it
via `infer`.
