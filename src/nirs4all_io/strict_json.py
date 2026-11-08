# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Strict JSON ingress and the shared cross-language number domain (Python oracle).

Both owners decode and write numbers over one domain, defined on decoded values:

- **Integers** lie in ``[-2**63, 2**64 - 1]`` inclusive (what ``serde_json`` keeps as
  ``i64``/``u64``). A JSON integer token (no ``.``, ``e`` or ``E``) outside the range is
  refused *before* any float conversion; the token ``-0`` decodes to ``-0.0`` exactly like
  ``serde_json``. An integer stays an integer until the selected storage conversion; it is
  never rounded here.
- **Float tokens** keep their own correctly rounded binary64 bits and must be finite;
  ``NaN`` / ``Infinity`` / ``-Infinity`` constants and exponent overflow are refused.
- Python **object** inputs (mappings, YAML documents) get the same integer domain
  (``bool`` is not an integer here) before any owner factory sees them.

Internal ``json.loads(json.dumps(...))`` copies are not ingress and keep using the standard
library directly. There is no custom parser: the standard decoder's own hooks are used.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

INTEGER_MIN = -(2**63)
INTEGER_MAX = 2**64 - 1


class NumberDomainError(ValueError):
    """A number lies outside the shared cross-language JSON number domain."""


def check_integer(value: int) -> int:
    """Refuse an integer outside ``[-2**63, 2**64 - 1]`` (booleans are not integers)."""
    if not INTEGER_MIN <= value <= INTEGER_MAX:
        raise NumberDomainError(f"integer {value} is outside the supported range [-2**63, 2**64 - 1]")
    return value


def _parse_int(token: str) -> int | float:
    if token == "-0":
        return -0.0
    return check_integer(int(token))


def decimal_float(token: str) -> float:
    """The ONE checked decimal-token primitive shared by JSON ingress and numeric text conversion.

    The token's correctly rounded binary64 value, refused when it overflows (non-finite) or when a nonzero
    mantissa digit parses to ``+-0`` (decimal underflow: the digits would be silently lost). Exponent digits never
    count as mantissa digits; genuine zeros (``0e5``, ``-0.000e-999``) keep their sign.
    """
    value = float(token)
    if not math.isfinite(value):
        raise NumberDomainError(f"number {token!r} overflows binary64")
    if value == 0.0 and any(digit in "123456789" for digit in token.lstrip("+-").lower().split("e")[0]):
        raise NumberDomainError(f"number {token!r} underflows to zero in binary64")
    return value


def _parse_float(token: str) -> float:
    return decimal_float(token)


def _parse_constant(token: str) -> Any:
    raise NumberDomainError(f"non-finite JSON constant {token!r} is not accepted")


def loads(text: str | bytes) -> Any:
    """Decode JSON text over the strict number domain (a ``json.JSONDecodeError`` stays a ``ValueError``)."""
    return json.loads(text, parse_int=_parse_int, parse_float=_parse_float, parse_constant=_parse_constant)


def check_numbers(value: Any) -> Any:
    """Refuse out-of-range Python ints anywhere in an object document; returns ``value`` unchanged."""
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is int:
            check_integer(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return value
