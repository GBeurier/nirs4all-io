# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""C6: a float sample/observation identity is its Python ``repr`` run through the ordinary identifier grammar.

No JS-safe cap: ``1e16`` / ``1e22`` refuse only because ``repr`` contains ``+``, which is not an identifier byte.
Non-finite and missing identities stay ``Missing projection identity``; an exact integer ID (C3) renders as its
canonical integer, never as a rounded float.
"""

from __future__ import annotations

import pytest

import nirs4all_io as nio
from nirs4all_io.materialize.numeric import NumericContractError

_SPEC = {
    "sample_index": {"by": "id", "key": "id"},
    "sources": [{"id": "t", "input": "d.csv", "key": "id", "columns": [{"role": "features", "select": ["x"]}, {"role": "targets", "select": ["y"]}]}],
    "params": {"delimiter": ",", "na": {"policy": "ignore"}},
}


def _ids(tmp_path, first: str, second: str = "7.5") -> list[str]:
    (tmp_path / "d.csv").write_text(f"id,x,y\n{first},1.5,1.0\n{second},2.5,2.0\n")
    package = nio.to_dataset_package(_SPEC, base_dir=tmp_path)
    return package.executed_projection(["t"]).identity.sample_ids


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("5.0", "5.0"),
        ("1e-07", "1e-07"),
        ("-0.0", "-0.0"),
        ("0.00001", "1e-05"),
        ("630989504857.15625", "630989504857.1562"),
        ("7.5", "7.5"),
    ],
)
def test_float_ids_are_their_python_spelling(tmp_path, token: str, expected: str) -> None:
    ids = _ids(tmp_path, token, "9.25")
    assert ids[0] == expected


@pytest.mark.parametrize("token", ["1e16", "1e22"])
def test_scientific_plus_spelling_fails_the_identifier_grammar_only(tmp_path, token: str) -> None:
    with pytest.raises(NumericContractError, match="Invalid primary identity"):
        _ids(tmp_path, token)


def test_missing_and_exact_integer_ids(tmp_path) -> None:
    with pytest.raises(NumericContractError, match="Missing projection identity"):
        _ids(tmp_path, "", "5")
    assert _ids(tmp_path, "9007199254740993", "1.5") == ["9007199254740993", "1.5"], "an exact integer ID is never a rounded float"
    assert _ids(tmp_path, "+0009007199254740993", "42") == ["9007199254740993", "42"]
