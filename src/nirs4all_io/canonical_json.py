# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Canonical JSON (story 7.3) — the cross-language wire contract.

The single biggest correctness risk in the Rust rewrite is that Python ``json``
and Rust ``serde_json`` disagree byte-for-byte. We pin one canonical form here
and mirror it in ``nirs4all-io-core`` (``canonical_json.rs``); a parity gate on
both sides compares against the same blessed expected files.

Canonical form:

- **Encoding:** UTF-8, no BOM; non-ASCII verbatim (``ensure_ascii=False``), to
  match ``serde_json`` which does not ``\\uXXXX``-escape non-ASCII.
- **Key order:** lexicographic by Unicode code point (``sort_keys=True``);
  matches ``serde_json``'s default ``BTreeMap`` object map.
- **Indent:** two spaces, ``": "`` after keys, one element per line — identical
  layout to ``serde_json::to_string_pretty``.
- **Line endings:** ``\\n`` with a single trailing ``\\n``.
- **Floats:** every finite binary64 value uses shortest round-trip digits, closest to
  the exact value with ties choosing an even last digit. Scientific exponents -4
  through 15 use fixed notation; others use lowercase ``e``, an explicit sign and at
  least two exponent digits. Integral fixed floats keep ``.0`` and signed zero is
  preserved. Nonfinite values refuse (``allow_nan=False``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from .strict_json import check_integer

# The writer relies on CPython's shortest round-trip float repr (David Gay / short repr style).
assert sys.float_repr_style == "short", "canonical JSON needs the shortest round-trip float repr"


def _check_domain(value: Any) -> None:
    """The shared number domain: integers in [-2**63, 2**64 - 1] (``bool`` excluded); floats finite."""
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is int:
            check_integer(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to the canonical JSON form (see module docstring)."""
    _check_domain(value)
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    )
