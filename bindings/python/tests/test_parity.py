# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Cross-language golden parity (EPIC 12.3): the pyo3 binding's to_spec / infer
output, canonicalized, is byte-identical to the committed contract goldens — the
same goldens the Rust facade and Python MVP reproduce, except for the explicitly
bounded V1-RUST-MIXED-SPECTRAL-HEADER case. This proves the binding is
on the byte-for-byte contract, not merely functional."""
import json
from pathlib import Path

import nirs4all_io as nio

CONTRACT = Path(__file__).resolve().parents[3] / "tests/goldens/contract"
CORPUS = CONTRACT / "corpus"


def canonical(obj) -> str:
    # The canonical-JSON wire contract (bindings/SPEC.md §2): 2-space indent,
    # sorted keys, ": " after keys, non-ASCII verbatim, finite-only, trailing \n.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def golden(name: str) -> str:
    return (CONTRACT / name).read_text()


def _normalize_contract_paths(payload: str, base: str | Path = CONTRACT) -> str:
    base_text = str(base)
    escaped_base = base_text.replace("\\", "\\\\")
    for prefix in (escaped_base, base_text):
        payload = payload.replace(f"{prefix}\\\\", "<CORPUS>/")
        payload = payload.replace(f"{prefix}/", "<CORPUS>/")
        payload = payload.replace(prefix, "<CORPUS>")
    return payload.replace("\\\\", "/")


def test_contract_path_normalization_handles_windows_json_escapes():
    base = r"D:\a\nirs4all-io\nirs4all-io\tests\goldens\contract"
    payload = (
        '{"ref": "'
        r"D:\\a\\nirs4all-io\\nirs4all-io\\tests\\goldens\\contract\\corpus\\single_combined"
        '"}'
    )
    assert _normalize_contract_paths(payload, base) == '{"ref": "<CORPUS>/corpus/single_combined"}'


def test_to_spec_dir_is_byte_identical_to_golden():
    # Directory inputs produce relative filenames (no abspath), so no path
    # normalization is needed — a direct byte comparison.
    for case, gold in [
        ("train_test", "ts_dir_train_test.to_spec.canonical"),
        ("x_y_separate", "ts_dir_x_y.to_spec.canonical"),
    ]:
        spec = nio.to_spec(str(CORPUS / case))
        assert canonical(spec) == golden(gold), f"to_spec parity drift for {case}"


def test_infer_dir_is_byte_identical_to_golden():
    for case, gold in [
        ("single_combined", "inf_single_combined.infer.canonical"),
        ("train_test", "inf_train_test.infer.canonical"),
        ("x_y_separate", "inf_x_y_separate.infer.canonical"),
    ]:
        plan = nio.infer(str(CORPUS / case))
        produced = _normalize_contract_paths(canonical(plan))
        expected = golden(gold)
        if case == "single_combined":
            frozen = json.loads(expected)
            frozen["params"]["data.csv"]["has_header"] = {"value": True, "score": 0.85}
            roles = frozen["columns"][0]["column_roles"]
            assert len(roles) == 13
            for role in roles[:12]:
                assert role["role"] == "features"
                role["score"] = 0.9
                role["evidence"] = ["monotonic nm/cm-1 wavelength column"]
            assert roles[12]["role"] == "targets"
            roles[12]["score"] = 0.65
            roles[12]["evidence"] = ["numeric non-wavelength column alongside a spectral block"]
            frozen["resolved_spec"]["sources"][0]["params"]["header_unit"] = "nm"
            expected = golden("inf_single_combined.infer.rust-v1.canonical")
            assert json.loads(expected) == frozen, "V1-RUST-MIXED-SPECTRAL-HEADER exceeded its reviewed delta"
        assert produced == expected, f"infer parity drift for {case}"


def test_explicit_header_options_precede_first_inference_read(tmp_path):
    from nirs4all_io._adapter import _decode_targets, _matrix

    path = tmp_path / "myspectra.csv"
    path.write_text("".join(
        f"0.{465041 + row * 10000:07};" + ";".join(["0.2"] * 699) + "\n"
        for row in range(20)
    ))
    auto = nio.infer(str(path))
    assert auto["params"][path.name]["header_ambiguous"] is True
    assert auto.resolved_spec["sources"][0]["params"]["has_header"] is True
    explicit = nio.infer(str(path), params={"has_header": True},
                         file_params={path.name: {"has_header": False}})
    assert explicit.resolved_spec["sources"][0]["params"]["has_header"] is False
    assert not any("header_ambiguous" in warning for warning in explicit.warnings)
    full = nio.assembled_full(explicit.resolved_spec)
    block = full["blocks"]["train"]
    assert block["n_samples"] == 20
    matrix = block["x"][0]
    x = _matrix(matrix)
    assert x.shape == (20, 699)
    assert x[0, 0] == 0.0465041
    assert x[19, 0] == 0.0655041
    assert all(matrix["validity"])
    y = _decode_targets(block["y"])
    assert y.shape == (20, 1)
    assert (y == 0.2).all()
    assert all(block["y"]["columns"][0]["block"]["validity"])
