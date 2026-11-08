<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# Versioning policy (story 7.5)

`nirs4all-io` carries several independently-versioned contracts. This file is the
single source of truth for the rules; the constants are asserted in CI on both
sides (`crates/nirs4all-io-core/tests/versions.rs`, `tests/test_versions.py`) so
the docs and the code cannot drift.

| Contract | Constant | Current | Bump rule |
|---|---|---|---|
| `DatasetSpec` schema | `DATASET_SPEC_SCHEMA_VERSION` (core) / `SCHEMA_VERSION` (py) | `1` | Bump (integer) on any **incompatible** change to the `DatasetSpec` wire shape (renamed/removed field, changed semantics). Additive optional fields do **not** bump it. The `schema_version` field is written into every emitted spec. |
| `DatasetPlan` shape | tracked with the spec schema | `1` | The plan is a recommendation, not a stored contract; its shape is versioned together with the spec schema (a plan always carries a `resolved_spec` whose `schema_version` is authoritative). |
| Convention profile | `CONVENTION_PROFILE_VERSION` (core) / per-profile `version` key | `1` | Bump a profile's `version` when its patterns change in a way that alters matches. The builtin floor is `CONVENTION_PROFILE_VERSION`. |
| Canonical JSON | `CANONICAL_JSON_VERSION` (core) | `1` | Bump only if the canonical-form rules in §2 of `bindings/SPEC.md` change (key order, indentation, float policy). A bump invalidates all goldens. |
| `AssembledDataset` summary/full wire | `ASSEMBLED_DATASET_VERSION` (core) / `assembled_schema_version` | `3` | Bump when the JSON-compatible assembly artifact changes. v3 full blocks retain exact selected bytes, masks, precision and direct source geometry; partition order and audits are required. Prior summaries do not reconstruct a v3 full record. |
| `DatasetPackage` summary/full wire | `DATASET_PACKAGE_VERSION` (core/py) / `schema_version` | `4` | v4 binds actual numeric storage/precision/masks with exact H4 payload framing; the full wire carries exact blocks, axes, partition order and audits. Identity, ordered manifest-root framing and metadata-only payload bytes are retained. |
| C ABI | `n4io_abi_version()` / `N4IO_ABI_VERSION` (capi) | `0.3.0` | Independent of crate semver (D-R6). MAJOR bumps on any breaking ABI change; MINOR on additive symbols; PATCH on fixes. Pre-1.0 every breaking change bumps MAJOR. Checked by `n4io_check_abi_compatibility`. |
| Targeted `dag-ml-data` schema | (EPIC 10) | tracked | io targets the `coordinator_data_plan_envelope.schema.json` shared by `dag-ml-data`/`dag-ml`. The exact targeted version is pinned in EPIC 10 and asserted against the live schema in the cross-CLI conformance gate (story 10.4). |

## Rules

1. **One bump, one reason.** Each contract bumps for its own reason; do not bump
   the spec schema because the ABI changed, or vice-versa.
2. **Goldens are append-only.** A canonical-JSON or schema bump re-blesses the
   golden corpus in the same PR that bumps the constant.
3. **Crate semver != ABI version.** The cargo workspace version
   (`0.3.0`) tracks the Rust crates; the C ABI tracks
   `N4IO_ABI_VERSION`. They move independently.

## Float64 owner batch: pre-V1 semantic corrections

The assembled full wire is version **3**, and the package wire is version **4**.
Both preserve exact selected storage bytes, mandatory validity, precision records,
source geometry, original partition order and the existing audit carrier. Cargo
package versions and dependencies are unchanged by this batch.

`nirs4all.io.dataset-content.v1` is introduced by this batch. It uses one corrected
normalizer shared with the owner factories, with unchanged tagged binary64 tree,
sorted-key order, zero folding, declared dtype, string-dtype relabel, U07 metadata
rule and SHA-256 framing. Earlier unlabelled pre-V1 fingerprints are not relabelled
as V1. The following semantic corrections affect their admission or bytes:

| Class | Input | Correct V1 behavior |
|---|---|---|
| A | Declared F32 integer cell whose conversion through F64 would double-round, including mixed lists | Convert the retained integer directly to F32; normalized bytes can change. |
| B | Declared F64 retained integer cell not exactly representable in F64 | Refuse before loss. Actual host-F64 inputs retain their own already-rounded bits. |
| C | Raw JSON integer token outside `[-2^63, 2^64-1]` | Refuse at text ingress before ordinary serde `Value` erases its integer origin. |
| D | Numeric int64/uint64 exact integers above `2^53-1`, such as `2^53` and `2^54` | Admit exact F64 values. The portable cap still applies to classification labels; the object-dtype cap is unchanged. |
| E | Declared F32 finite nonzero float, such as `5e-324`, rounding to zero | Refuse underflow instead of admitting and hashing zero. Overflow already refused. |
| F | A nonzero decimal float token, such as `1e-400` or `2e-324`, rounding to F64 zero | Refuse at text ingress; inspect mantissa digits only, so genuine zero tokens remain valid. |

A–F are the known affected classes, not an exhaustive claim about future findings.
Outside these classes, frozen whole-content proofs must retain their bytes. Pre-V1 Core
`dataset-content` and Studio fingerprints in these classes may differ or no longer
be admitted. There is no compatibility-normalizer path; propagation to Core and
Studio remains deferred. Whole-content identities remain distinct from package
roots, executed feature/target identities and the fixed X-only prediction profile.
Validate a declared source before any output request: declared F64 retained
inexact integers refuse under every request; declared F32 normalizes once by
correct direct conversion, and a requested F64 only widens valid source bits.
Integer/untyped sources retain their integer authority and can convert directly
to requested F32. Actual host-F64 inputs retain their already-rounded bits.

The whole-content link exists when the ORIGINAL source record is representable
in the request-free V1 binary64 tree. Valid integer-source execution under an
F32 request may have no whole link; its existing audit carrier records
`dataset_content_identity_unavailable` with profile
`nirs4all.io.dataset-content.v1` and reason `not_representable`. That status is
derived from actual owner validation and original integer representability,
preserved across full wires, and recomputed during receipt validation. It never
licenses a synthetic rounded record, an echoed hash, or mislabelled malformed,
missing or tampered source evidence.

Canonical JSON and scalar float text share the owner's shortest, closest,
ties-even finite-F64 kernel. Serde supplies the digits; the owner changes only
notation layout. Fixed notation uses scientific exponents -4 through 15;
otherwise lowercase `e`, explicit exponent sign and at least two exponent digits
apply. Signed zero and integral-float `.0` are preserved. Integer tokens stay
integral and strings are unchanged. This corrects finite-domain spelling defects;
it does not change binary payload hash framing. The text guard is the shared
`canonical_json::parse_json_value`; already-decoded `Value` or genuine host-F64
objects cannot recover original integer lexemes.

IO 0.3.0 introduces a breaking Linux linker compatibility change: the C ABI
build relies on rustc's cdylib export list instead of a
second, conflicting named GNU ld version script. Exact `n4io_*` exports, the
committed header, ABI version functions and fixed SONAME are retained. The obsolete
`N4IO_1` ELF version node is removed, so clients explicitly requiring that node are
not claimed compatible. Other language/vendor propagation and installed binding
qualification remain stage4 work.
