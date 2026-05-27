# Status

**Phase 1 (Python MVP) — COMPLETE & Codex-ACCEPTED** (2026-05-27).
178 tests pass; ruff + mypy clean; parity-verified against the real `nirs4all`.

## Per-epic

| Epic | Story | Module / artifact | State |
|---|---|---|---|
| 2.0/2.1 | DatasetSpec IR + selectors + validation | `spec/` (`dataset_spec`, `selectors`, `validate`) | ✅ |
| 2.1/5.1 | Versioned JSON Schema (wire contract) | `spec/json_schema.py` + `dataset_spec.schema.json` | ✅ |
| 2.2 | Alias normalizer (verbatim map) + legacy→spec + root-param shorthand | `spec/normalize.py` | ✅ |
| 2.3 | Declarative conventions (FolderParser parity) + 4 profiles | `conventions/` | ✅ |
| 2.4 | Cookbook coverage matrix (introspected, fails on uncovered vocab) | `tests/test_cookbook.py` | ✅ |
| 3.1 | Resolver → InputSet (identity/hash/sidecars/ordering) | `resolve/` | ✅ |
| 3.2/3.3/3.5 | `infer` + scored `DatasetPlan` + column-role inference + `describe` | `infer/` | ✅ |
| 3.4/3.6 | Labeled corpus + per-decision precision + abstention | `tests/test_inference_corpus.py` | ✅ (Brier/ECE deferred) |
| 4.0 | Relational join engine (cardinality/coverage/duplicate/audit) | `materialize/join.py` | ✅ |
| 4.1a/4.2 | Tabular loaders + NA policy + param precedence | `materialize/loaders.py` | ✅ |
| 4.1b | Assembler + lazy `SpectroDataset` adapter | `materialize/{assemble,spectrodataset}.py` | ✅ |
| 5.2 | Parity oracle vs real nirs4all (`pytest -m parity`) | `tests/test_parity.py` | ✅ |
| 5.3 | Re-plug guide | `docs/REPLUG.md` | ✅ |
| 5.4 | Copy-provenance manifest + dual license | `COPY_PROVENANCE.md`, `LICENSE` | ✅ |
| 6.1 | Hardening (adversarial inputs → clear errors) | `tests/test_hardening.py` | ✅ |
| 6.5 | Import-boundary (no nirs4all at import) | `tests/test_import_boundary.py` | ✅ |

## Deferred (documented, not blocking)

- **Brier/ECE calibration** (3.6) — needs a real vendor/domain-split corpus; current
  scores are explicitly *uncalibrated* (triage/ranking only, Critique C5).
- **Phase 2** (Rust core + `dag-ml-data`) — gated; see [`PHASE2_GATE.md`](PHASE2_GATE.md)
  (needs `AxisKind::Wavenumber` + a connector-ownership ADR from the dag-ml owners).
- **Cross-language goldens** (6.4) — Phase-2 gate.

## Codex review trail

Phase-0 rename · Phase-1 foundation (PASS) · Phase-1 build (2 blockers → fixed) ·
final round 1 (vendor-corpus blocker → fixed) · final round 2 (`filename_stem` stem
semantics → fixed, test-verified) · **final round 3 → VERDICT: ACCEPT**.
