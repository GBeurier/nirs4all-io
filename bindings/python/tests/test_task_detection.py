# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Real installed-candidate detector tests; source-only probes never run these."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import struct
import sys
import warnings
import zipfile
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
from pathlib import Path

import numpy as np
import pytest

import nirs4all_io as nio
from nirs4all_io import _native, detect_task_type

REFERENCE_PATH = Path(__file__).resolve().parents[3] / "src/nirs4all_io/infer/detectors.py"
REFERENCE_SHA256 = "89b77b1a5eca9c8eec70955884fa4f5b03f959e4ddba66712531ae739c569055"

_LITERAL_CASES = [
    ("empty", [], .05, ("regression", 0.)),
    ("all-nan", [float("nan"), float("nan")], .05, ("regression", 0.)),
    ("none", None, .05, ("regression", 0.)),
    ("binary-nan", [0., 1., float("nan")], .05, ("binary", .9)),
    ("negative-binary", [-3, -2], .05, ("binary", .9)),
    ("three", [0, 1, 2], .05, ("multiclass", .8)),
    ("100", list(range(100)), .05, ("multiclass", .8)),
    ("101", list(range(101)), .05, ("regression", .8)),
    ("constant", [7, 7], .05, ("regression", .8)),
    ("signed-zero", [-0., 0.], .05, ("regression", .8)),
    ("signed-zero-binary", [-0., 0., 1.], .05, ("binary", .9)),
    ("fraction-binary-equality", [.1, .2] * 20, .05, ("binary", .6)),
    ("fraction-binary-below", [.1, .2] * 19, .05, ("regression", .8)),
    ("five-of-100", [.1, .2, .3, .4, .5] * 20, .05, ("multiclass", .6)),
    ("six-of-100", [.1, .2, .3, .4, .5, .6] * 16 + [.1, .2, .3, .4], .05, ("regression", .8)),
    ("single-fraction", [.5] * 20, .05, ("multiclass", .6)),
    ("ordinary-regression", [-.5, .2, 2.3], .05, ("regression", .8)),
    ("infinity", [0, 1, float("inf")], .05, ("regression", .8)),
    ("negative-infinity", [-float("inf")], .05, ("regression", .8)),
    ("subnormal", [5e-324], .05, ("regression", .8)),
    ("threshold-zero", [.1, .2] * 2, 0., ("regression", .8)),
    ("threshold-negative", [.1, .2] * 2, -1., ("regression", .8)),
    ("threshold-one", [.1, .2] * 2, 1., ("binary", .6)),
    ("threshold-infinity", [.1, .2] * 2, float("inf"), ("binary", .6)),
    ("threshold-nan", [.1, .2] * 2, float("nan"), ("regression", .8)),
    ("threshold-int", [.1, .2] * 2, 2, ("binary", .6)),
    ("threshold-bool-true", [.1, .2] * 2, True, ("binary", .6)),
    ("threshold-bool-false", [.1, .2] * 2, False, ("regression", .8)),
    ("threshold-numpy-f64", [.1, .2] * 2, np.float64(.5), ("binary", .6)),
    ("threshold-adjacent-below", [.1, .2] * 2, np.nextafter(.5, 0.), ("regression", .8)),
    ("threshold-exact", [.1, .2] * 2, .5, ("binary", .6)),
    ("float32-threshold-f64", [.1, .2, .3, .4, .6, .7, .8, .1, .2, .3], np.float32(.7), ("regression", .8)),
]
# Literal-only exclusions: reference arithmetic can retain an np.float32 threshold,
# and an integer-data reference call can bypass invalid threshold extraction.
REFERENCE_EXCLUSIONS = {"float32-threshold-f64", "invalid-threshold-type"}


def _exact_result(actual, expected):
    assert type(actual) is tuple and len(actual) == 2
    assert type(actual[0]) is str and type(actual[1]) is float
    assert actual[0] == expected[0]
    assert struct.pack(">d", actual[1]) == struct.pack(">d", expected[1])


@pytest.fixture(scope="module", autouse=True)
def installed_candidate_origins():
    assert isinstance(_native.__spec__.loader, ExtensionFileLoader)
    prefix = Path(sys.prefix).resolve()
    package, native = Path(nio.__file__).resolve(), Path(_native.__file__).resolve()
    for origin in [package, native]:
        assert origin.is_relative_to(prefix) and "site-packages" in origin.parts, origin
    assert native.name in {f"_native{suffix}" for suffix in EXTENSION_SUFFIXES}
    distribution = importlib.metadata.distribution("nirs4all-io")
    direct_url = distribution.read_text("direct_url.json")
    assert direct_url is None or not json.loads(direct_url).get("dir_info", {}).get("editable", False)
    assert nio.__version__ == _native.__version__ == distribution.version
    assert "detect_task_type" in nio.__all__ and callable(_native.detect_task_type)
    print(json.dumps({"package": str(package), "native": str(native), "prefix": str(prefix),
                      "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(), "version": distribution.version}))


@pytest.fixture(scope="module")
def explicit_reference():
    assert REFERENCE_PATH.is_file(), f"Required explicit reference is absent: {REFERENCE_PATH}"
    assert hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("n4io_explicit_task_detection_reference", REFERENCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == REFERENCE_PATH.resolve()
    return module.detect_task_type


@pytest.mark.parametrize("route", ["public", "native"])
@pytest.mark.parametrize("case,values,threshold,expected", _LITERAL_CASES, ids=[case[0] for case in _LITERAL_CASES])
def test_literal_owner_results(route, case, values, threshold, expected):
    # The raw Vec<f64> entry receives the public normalization, including None -> NaN.
    actual = (detect_task_type(values, threshold) if route == "public" else
              _native.detect_task_type(np.asarray(values).ravel().astype(float).tolist(), threshold))
    _exact_result(actual, expected)


@pytest.mark.parametrize("case,values,threshold,expected", [row for row in _LITERAL_CASES if row[0] not in REFERENCE_EXCLUSIONS])
def test_shared_f64_reference_profile(explicit_reference, case, values, threshold, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Reference-only infinity arithmetic.
        reference = explicit_reference(values, threshold)
    _exact_result(reference, expected)
    _exact_result(detect_task_type(values, threshold), reference)


@pytest.mark.parametrize("threshold", ["x", None, object(), [.05]])
def test_invalid_threshold_type_is_refused_even_for_integer_data(threshold):
    # Literal-only: the explicit reference may not inspect threshold for integer y.
    with pytest.raises(TypeError):
        detect_task_type([0, 1], threshold)
    with pytest.raises(TypeError):
        _native.detect_task_type([0., 1.], threshold)


def test_threshold_default_positional_keyword_and_effective_nondefault():
    values = [.1, .2] * 20
    for actual in [detect_task_type(values), detect_task_type(values, .05), detect_task_type(values, threshold=.05)]:
        _exact_result(actual, ("binary", .6))
    _exact_result(detect_task_type([.1, .2] * 10), ("regression", .8))
    _exact_result(detect_task_type([.1, .2] * 10, .1), ("binary", .6))


@pytest.mark.parametrize("values", [np.array(3), [[0, 1], [0, 1]], np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
                                    np.array([[0, 1], [1, 0]], dtype=np.float32), np.array([False, True]),
                                    np.array([0, None, 1], dtype=object), np.array(["0", "1"]),
                                    np.asfortranarray([[0., 1.], [1., 0.]])])
def test_conversion_reference_and_no_input_mutation(explicit_reference, values):
    array = np.asarray(values)
    before = (array.shape, array.dtype, array.strides, array.tobytes(order="A"))
    _exact_result(detect_task_type(values), explicit_reference(values))
    after = np.asarray(values)
    assert (after.shape, after.dtype, after.strides, after.tobytes(order="A")) == before


@pytest.mark.parametrize("values", [["class-a", "class-b"], [[0, 1], [2]], [object()]])
def test_conversion_refusals_match_reference(explicit_reference, values):
    with pytest.raises((ValueError, TypeError)) as reference:
        explicit_reference(values)
    with pytest.raises(type(reference.value)):
        detect_task_type(values)


def test_complex_conversion_warning_by_class_name(explicit_reference):
    values = np.array([0 + 2j, 1 - 3j])
    before = values.tobytes()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        actual = detect_task_type(values)
    with warnings.catch_warnings(record=True) as reference_warnings:
        warnings.simplefilter("always")
        reference = explicit_reference(values)
    assert any(w.category.__name__ == "ComplexWarning" for w in observed)
    assert any(w.category.__name__ == "ComplexWarning" for w in reference_warnings)
    _exact_result(actual, reference)
    assert values.tobytes() == before


def test_infinity_has_no_reference_arithmetic_warning():
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        _exact_result(detect_task_type([float("inf")]), ("regression", .8))
    assert not any(w.category.__name__ == "RuntimeWarning" for w in observed)


@pytest.mark.skipif(
    "N4IO_TEST_WHEEL" not in os.environ,
    reason="Built-wheel artifact identity profile is not configured; set N4IO_TEST_WHEEL",
)
def test_built_wheel_installed_artifact_identity():
    # A configured packaging profile must identify the actual installed artifact.
    wheel = Path(os.environ["N4IO_TEST_WHEEL"])
    assert wheel.is_file(), f"Configured wheel does not exist: {wheel}"
    native_member = f"nirs4all_io/{Path(_native.__file__).name}"
    installed_record = importlib.metadata.distribution("nirs4all-io").read_text("RECORD")
    assert installed_record is not None
    installed_rows = list(csv.reader(io.StringIO(installed_record)))
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        native_members = [name for name in archive.namelist() if name.startswith("nirs4all_io/_native") and name.endswith((".so", ".pyd"))]
        assert native_members == [native_member]
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        assert len(records) == 1
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        for label, member, installed in [
            ("native", native_member, Path(_native.__file__)),
            ("wrapper", "nirs4all_io/__init__.py", Path(nio.__file__)),
        ]:
            data = installed.read_bytes()
            assert archive.read(member) == data, f"Installed {label} differs from the configured wheel"
            digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert [row[1:] for row in rows if row[0] == member] == [[digest, str(len(data))]]
            assert [row[1:] for row in installed_rows if row[0] == member] == [[digest, str(len(data))]]
            hashes[f"{label}_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps({"wheel": str(wheel.resolve()), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      **hashes, "installed_wheel_artifact_identity": True}))
