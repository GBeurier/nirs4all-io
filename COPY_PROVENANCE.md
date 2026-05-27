# Copy provenance manifest

> Story 5.4 / Critique C12. `nirs4all-io` is built by **copying and
> re-orchestrating** business logic from `nirs4all` (and, by design reference,
> `nirs4all-studio`). `nirs4all`'s default OSS license is **AGPL-3.0-or-later**
> with **CeCILL-2.1** as a variant; `nirs4all-io` is licensed
> **`CeCILL-2.1 OR AGPL-3.0-or-later`** so the copied logic stays
> license-compatible. Both projects are owned by **G. Beurier (GBeurier)**, who
> authorizes this reuse.

This file records every block of logic copied or adapted from `nirs4all` into
`nirs4all-io`, mapping the **source** (path in the `nirs4all` repo) to its
**destination** (module in this repo) and the **action** taken:

- **COPY-LOGIC** — the algorithm/data was reproduced (possibly re-architected).
- **GENERALIZE** — COPY-LOGIC + made declarative / format-agnostic / multi-target.
- **PORT** — translated across languages.
- **REFERENCE** — used only as a design reference (not copied).

| # | Source (`nirs4all/nirs4all/...`) | Destination (`src/nirs4all_io/...`) | Action | Notes |
|---|---|---|---|---|
| 2 | `data/parsers/normalizer.py` (alias map) | `spec/normalize.py` | COPY-LOGIC | partition×role synonym map reproduced verbatim |
| 4 | `data/parsers/folder_parser.py` (`FILE_PATTERNS`, word-boundary match) | `conventions/builtin/*.toml` + `conventions/engine.py` | GENERALIZE | patterns → declarative profiles; extension set widened beyond CSV |

_(rows are appended as each piece is copied)_
