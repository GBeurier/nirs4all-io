<!-- SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later -->
# MATLAB / Octave binding (EPIC 11.3)

C-ABI-first binding over `libnirs4all_io_capi` (the `n4io_*` JSON surface). A
single MEX (`n4io.c`) dispatches on a command string; the `+nirs4all_io` package
gives idiomatic wrappers that take native MATLAB/Octave inputs and return decoded
structs.

```matlab
% Native inputs in, decoded structs out (jsonencode/jsondecode done internally).
spec = nirs4all_io.to_spec('/data/run');          % struct: canonical DatasetSpec
plan = nirs4all_io.infer('/data/run');            % struct: scored DatasetPlan
nirs4all_io.validate(spec);                       % true; errors if invalid
nirs4all_io.summary(plan);                        % readable scored-decision overview
plan.resolved_spec                                % the editable spec (a struct)
nirs4all_io.abi_version()

% A cellstr file list or a struct config are accepted natively too:
nirs4all_io.infer({'a.csv', 'b.csv'});
nirs4all_io.to_spec(struct('name', 'run', 'sources', {{...}}));
```

`input` accepts a native value — a char path, a cellstr of files, or a struct
config — which is `jsonencode`d into the C-ABI JSON form (a quoted string, a JSON
array, or a JSON object respectively) by `+nirs4all_io/+internal/encode_input.m`;
the canonical result is `jsondecode`d back into a struct. `validate` accepts a
struct or a JSON char. `nirs4all_io.summary` prints scored decisions for a plan
and name/sources for a spec.

The raw JSON surface stays reachable through the MEX dispatcher
(`n4io('to_spec', input_json, ...)`, `n4io('infer', ...)`,
`n4io('load_summary', input_json)`, `n4io('validate', spec_json)`,
`n4io('abi_version')`) — identical to the C ABI / other bindings.

`nirs4all_io.infer(input, options)` accepts empty options or a scalar struct with
optional `conventions` (cellstr), `params` (scalar struct), and `file_params`
(`containers.Map` with char keys and scalar parameter structs as values).
File choices override global parameters before the first read. Keys are exact
input refs: basenames for files/file lists, relative paths for directory inputs.
Unknown refs are refused.

```matlab
options = struct();
options.conventions = {'nirs4all-classic'};
options.params = struct('has_header', true);
options.file_params = containers.Map({'noheader.csv'}, {struct('has_header', false)});
plan = nirs4all_io.infer('/data/noheader.csv', options);
loaded = nirs4all_io.load_summary(plan.resolved_spec);
% Raw infer takes NULL (omitted/empty) or an InferOptions object, never an array:
n4io('infer', '"/data/noheader.csv"', ...
     '{"conventions":["nirs4all-classic"],"params":{"has_header":false}}');
```

`encode_infer_options` serializes each map key with `jsonencode` directly and
assembles the object from these JSON pairs; it never converts refs into struct
field names or relies on host-specific `jsonencode(containers.Map)` behavior.
This preserves dotted and slash-containing keys on the wire. Decoding a returned
plan into a struct may still normalize keys in its diagnostic maps, depending
on the host; use the raw JSON surface when those exact keys are needed.
MATLAB/Octave execution of this encoder remains a required smoke gate.

`to_spec` and `load_summary` still accept convention arrays via their existing
second arguments. An ambiguous unrecognized header defaults to true and emits
`header_ambiguous`; declare false for headerless data. Summary values are rounded
to four decimals rather than exposing original numeric precision.

`jsonencode`/`jsondecode` exist in MATLAB and in Octave ≥ 7 (the CI runs Octave
on `ubuntu-latest`, currently 8.x).

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
