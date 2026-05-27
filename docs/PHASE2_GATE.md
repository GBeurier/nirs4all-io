# Phase 2 readiness gate (dag-ml-data target)

> Phase 2 (a Rust core emitting the `dag-ml-data` contract) is **gated** by the
> Appendix J readiness checklist of the redesign doc. This file records the
> **verified current status** of that gate (assessed against the live
> `dag-ml-data` and `dag-ml` repos). Phase 2 implementation does not start until
> the two blockers below are resolved by the dag-ml owners.

## Verdict: **BLOCKED** (closer than the doc's Appendix J states)

| # | Appendix J item | Status | Evidence |
|---|---|---|---|
| 1 | External construction path | 🟡 PARTIAL | No Python pkg; but all core structs are serde, a `dag-ml-data-cli` (`EnvelopePlan`/`Fingerprint*`/`ValidateEnvelope`) + C ABI exist. A **Rust-first** bridge (the planned form) has everything via the facade. Only an ergonomic Python builder is missing. |
| 2 | cm⁻¹ axis (`AxisKind::Wavenumber`) | 🔴 **BLOCKER** | `AxisKind` (`dag-ml-data-core/src/model.rs:10-26`) has `Wavelength`/`Frequency` but **no `Wavenumber`**; the interim `Feature`+`unit:"cm-1"` convention is available but **not ratified** in any doc. |
| 3 | Relation id mapping (`origin_id`↔`origin_sample_id`) | 🟢 GREEN | `coordinator_relations_from_sample_table` (`coordinator.rs:132-176`) resolves observation→sample; tested; dag-ml's `SampleRelation` is field-compatible; the shared `coordinator_data_plan_envelope.schema.json` is byte-identical across repos. |
| 4 | Fingerprints exposed | 🟢 GREEN | `schema_fingerprint`/`data_plan_fingerprint`/`sample_relation_fingerprint` all `pub` + facade-re-exported + CLI-reachable. |
| 5 | Array host path | 🟡 PARTIAL | `NumericFeatureMatrixF64` + typed C-ABI host path exist, but only an **in-memory test provider** ships (production arenas pending per dag-ml-data ROADMAP). Does not block a Rust-first schema/plan/relation emit. |
| 6 | `dag-ml validate-data-binding` reachable | 🟢 GREEN (caveat) | dag-ml CLI `ValidateDataBinding` consumes `ExternalDataPlanEnvelope` (drops the plan body). The bridge must emit dag-ml-data's `CoordinatorDataPlanEnvelope` and rely on the **shared JSON schema** compat (no shared Rust type). |
| 7 | Connector ownership | 🔴 **BLOCKER** | Three docs claim the SpectroDataset connector (dag-ml-data ROADMAP Phase 4; this redesign Appendix I; dag-ml design docs). **Not reconciled** — needs a co-design decision. |

## Minimal set to flip the gate GREEN (owner action, not ours)

1. **Add `AxisKind::Wavenumber`** to `dag-ml-data-core/src/model.rs` (a one-line enum
   addition; consuming code is permissive, serde/fingerprints unaffected) — *or*
   ratify the `Feature`+`unit:"cm-1"`+`coordinates` interim convention in a dag-ml-data doc.
2. **Record a connector-ownership ADR**: declare that **`nirs4all-io` owns the
   SpectroDataset → `CoordinatorDataPlanEnvelope` bridge** and descope dag-ml-data
   ROADMAP Phase 4 to "accept io-emitted artifacts".

Everything else needed to *start* Phase 2 (construct → fingerprint → emit the
coordinator envelope → validate via `dag-ml-data-cli ValidateEnvelope` **and**
`dag-ml ... validate-data-binding`) already exists. The right acceptance test
(story 4.4) is a cross-CLI golden of `coordinator_data_plan_envelope.json`.

## When unblocked — Phase 2 plan (already designed)

Appendix H.2 of the redesign doc fully specifies the mapping
(`DatasetSpec` → `DatasetSchema` + `SampleRelationTable` + the dag-ml campaign
`FoldSet`/`DataBinding`/`ExternalDataPlanEnvelope`). The Phase-1 `AssembledDataset`
IR is deliberately target-agnostic so a `to_dag_ml_data(assembled)` adapter slots
in beside `to_spectrodataset`.
