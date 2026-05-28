# Future work — out-of-scope of the current Python MVP

> This file consolidates items that are **intentionally deferred** because they
> depend on something outside `nirs4all-io`'s perimeter (a host extension, an
> external library decision, or a Phase 2 deliverable that has its own gate).
> Nothing here is a blocker for the lib's current users; it is a forward log so
> nothing slips through the cracks when the relevant external pieces move.

## Phase 2 — Rust core + `dag-ml-data` target

Status: **unblocked** (both gate blockers resolved 2026-05-28). The **actionable,
multi-agent plan** is in [`RUST_REWRITE_ROADMAP.md`](RUST_REWRITE_ROADMAP.md); gate status
in [`PHASE2_GATE.md`](PHASE2_GATE.md). The Python MVP's `AssembledDataset` IR is already
target-agnostic, so a `to_dag_ml_data(assembled)` adapter slots in beside
`to_spectrodataset(assembled)`.

Former blockers — both **resolved** by the `dag-ml-data` owners (2026-05-28):

1. ✅ **`AxisKind::Wavenumber`** added in `dag-ml-data-core/src/model.rs` (commit `5063fb0`).
2. ✅ **Connector-ownership ADR** (`ADR-0001`, Accepted) — `nirs4all-io` owns the
   `SpectroDataset → CoordinatorDataPlanEnvelope` bridge; `dag-ml-data` ROADMAP Phase 4 descoped.

What we will then do here (already designed, see Appendix H.2 of the redesign
doc):

- **`to_dag_ml_data(assembled)`** adapter — `DatasetSpec` → `DatasetSchema` +
  `DataPlan` + `SampleRelationTable`, assembled into a `CoordinatorDataPlanEnvelope`.
  **io does not emit `FoldSet` / `DataBinding`** (those stay in `dag-ml`). Consumes the
  `observation_id` / `group_id` fields from `sample_index` (already parsed and carried in
  the IR, see [`DATASET_CONFIGURATIONS.md §3`](DATASET_CONFIGURATIONS.md)).
- **Rust port of the Python core** (`describe` / spec parse / resolver /
  inference) so the same logic powers both Python and Rust callers; the
  Python lib becomes a thin facade over the Rust core (or stays Python --
  decision deferred to when the gate opens).
- **Cross-language goldens (story 6.4)** — fixtures of JSON produced by
  Python that must be byte-identical to ones produced by the Rust core.

## SpectroDataset extension on the nirs4all side (host-owned)

Three IR fields are carried but not materialized today because the target
SpectroDataset has no first-class slot for them. Extending the host's
`SpectroDataset` is **out of scope for `nirs4all-io`** -- the lib does not
modify nirs4all (see [`REPLUG.md`](REPLUG.md)).

| IR field | Today | Could be (host change) |
|---|---|---|
| `sample_index.observation_id` | parsed; stored as the row's natural index | `SpectroDataset.set_observation_ids()` for explicit per-row observation tags (independent of the sample-level group) |
| `sample_index.group_id` | parsed; can be smuggled in via `metadata` | `SpectroDataset.set_groups()` to drive leakage-aware CV without indirection through metadata |
| `role: weights` | surfaced as `__sample_weight__` metadata column | `SpectroDataset.set_sample_weights()` + automatic `fit(sample_weight=...)` plumbing in the pipeline |

For all three, the workaround today is to declare a regular metadata column
and read it explicitly in the pipeline.

## Inference calibration

`infer()` returns scores in `plan.scores`, currently **ordinal** (triage /
ranking). Brier/ECE calibration (story 3.6) requires a labelled
vendor/domain-split corpus that we do not have yet. The current uncalibrated
behaviour is explicit (Critique C5) and documented; nothing to do until a
corpus exists.

## Deferred polish

- **Stratified percentage split** at load time — intentionally rejected
  (`nirs4all-io` is a loader, not a splitter; see [`DATASET_CONFIGURATIONS.md §7`](DATASET_CONFIGURATIONS.md)).
  Stratification belongs in the pipeline's CV layer.
- **More vendor formats** beyond what `nirs4all-formats` ships -- driven by
  that library's roadmap, not this one.
