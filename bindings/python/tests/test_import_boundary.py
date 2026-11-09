# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Real installed-candidate subprocess boundary, before and after detection."""
import os
import subprocess
import sys


def test_import_does_not_load_nirs4all(tmp_path):
    code = r'''
import hashlib, importlib.metadata, json, sys
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
from pathlib import Path

def sdk_modules():
    return sorted(name for name in sys.modules if name == "nirs4all" or name.startswith("nirs4all."))

assert not sdk_modules(), sdk_modules()
import nirs4all_io as nio
from nirs4all_io import _native, detect_task_type
assert not sdk_modules(), sdk_modules()
assert isinstance(_native.__spec__.loader, ExtensionFileLoader)
for value in [nio.__file__, _native.__file__]:
    origin = Path(value).resolve()
    assert origin.is_relative_to(Path(sys.prefix).resolve()) and "site-packages" in origin.parts, origin
assert Path(_native.__file__).name in {"_native" + suffix for suffix in EXTENSION_SUFFIXES}
native_sha = hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()
assert native_sha != "2363375525b8a99daf68f1c9685603efc09b64e66ecec87a1f40a629157c4ffd"
assert nio.__version__ == _native.__version__ == importlib.metadata.version("nirs4all-io")
assert "detect_task_type" in nio.__all__
assert detect_task_type([0, 1]) == ("binary", 0.9)
assert _native.detect_task_type([0., 1.]) == ("binary", 0.9)
assert not sdk_modules(), sdk_modules()
print(json.dumps({"package": nio.__file__, "native": _native.__file__, "native_sha256": native_sha,
                  "isolated": sys.flags.isolated, "sdk_modules_before_and_after": []}))
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS"]:
        env[name] = "1"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path, env=env,
                            check=True, capture_output=True, text=True)
    print(result.stdout)
