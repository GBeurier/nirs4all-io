<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# MATLAB / Octave binding (EPIC 11.3)

C-ABI-first binding over `libnirs4all_io_capi` (the `n4io_*` JSON surface). A
single MEX (`n4io.c`) dispatches on a command string; the `+nirs4all_io` package
gives idiomatic wrappers.

```matlab
nirs4all_io.to_spec('"/data/run"')          % canonical DatasetSpec (JSON string)
nirs4all_io.infer('"/data/run"')            % DatasetPlan (JSON string)
nirs4all_io.validate(specJson)              % errors if invalid
nirs4all_io.abi_version()
```

Inputs cross as JSON values (a path is a quoted string, a file list is a JSON
array, a spec is a JSON object) — identical to the C ABI / other bindings. The
idiomatic JSON⇄struct layer (e.g. `jsondecode`) is the user's; v0 returns/accepts
canonical JSON strings.

## Build & test

```bash
bash bindings/matlab/build_and_test.sh      # builds the capi, mex, runs smoke (Octave)
```

`build.m` compiles the MEX against the prebuilt `nirs4all-io-capi` cdylib
(`N4IO_INCLUDE` / `N4IO_CAPI_DIR`); `smoke.m` exercises the surface on the
contract corpus. The same `build.m`/`smoke.m` run under MATLAB.

**Local testing note.** This binding is **CI-gated** (`octave-binding.yml`): MATLAB
and Octave are not assumed present on developer machines, so `build_and_test.sh`
skips cleanly when `octave` is absent. The MEX glue mirrors the verified R glue
(`bindings/r/src/n4io.c`) against the same frozen header.
