% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
% Smoke test for the MATLAB/Octave binding. Assumes the `n4io` mex is on the path
% (run from bindings/matlab after build.m) and N4IO_CORPUS is set.
corpus = getenv('N4IO_CORPUS');
assert(~isempty(corpus), 'N4IO_CORPUS not set');

json_path = @(p) ['"' p '"'];

% to_spec on train_test -> canonical spec with schema_version.
spec = n4io('to_spec', json_path(fullfile(corpus, 'train_test')));
assert(~isempty(strfind(spec, '"schema_version"')), 'spec missing schema_version'); %#ok<STREMP>

% the produced spec validates.
n4io('validate', spec);

% infer returns a plan.
plan = n4io('infer', json_path(fullfile(corpus, 'single_combined')));
assert(~isempty(strfind(plan, '"resolved_spec"')), 'plan missing resolved_spec'); %#ok<STREMP>

% a bad spec is rejected.
rejected = false;
try
  n4io('validate', '{"partitions": {"by": "random"}}');
catch
  rejected = true;
end
assert(rejected, 'bad spec was not rejected');

assert(~isempty(n4io('abi_version')), 'empty abi_version');

disp('matlab/octave binding smoke OK');
