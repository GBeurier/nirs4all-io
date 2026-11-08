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

load_json = n4io('load_summary', json_path(fullfile(corpus, 'train_test')));
assert(~isempty(strfind(load_json, '"assembled_schema_version": 3')), 'summary missing schema version'); %#ok<STREMP>

% a bad spec is rejected.
rejected = false;
try
  n4io('validate', '{"partitions": {"by": "random"}}');
catch
  rejected = true;
end
assert(rejected, 'bad spec was not rejected');

assert(~isempty(n4io('abi_version')), 'empty abi_version');

% --- idiomatic layer (nirs4all_io.*) ------------------------------------------
% These accept native MATLAB/Octave inputs and return decoded structs. The
% +nirs4all_io package directory must be on the path (its PARENT must be added,
% e.g. addpath(fileparts(pwd)) when running from bindings/matlab, or build.m's
% dir). smoke.m is run with --path "$(pwd)" so the package is visible here.

% to_spec on a char path -> a struct with schema_version and sources.
sp = nirs4all_io.to_spec(fullfile(corpus, 'train_test'));
assert(isstruct(sp), 'to_spec did not return a struct');
assert(isfield(sp, 'schema_version') && sp.schema_version == 1, 'spec missing schema_version');
assert(isfield(sp, 'sources') && ~isempty(sp.sources), 'spec missing sources');

% the struct round-trips through validate.
assert(nirs4all_io.validate(sp) == true, 'valid spec rejected');

% infer on a char path -> a plan struct with resolved_spec + scored decisions.
pl = nirs4all_io.infer(fullfile(corpus, 'single_combined'));
assert(isstruct(pl) && isfield(pl, 'resolved_spec'), 'plan missing resolved_spec');
assert(isfield(pl, 'structure') && isfield(pl.structure, 'value'), 'plan missing scored decisions');

% summary renders both without error and returns char when asked.
s1 = nirs4all_io.summary(pl);
assert(ischar(s1) && ~isempty(strfind(s1, 'plan')), 'plan summary failed'); %#ok<STREMP>
s2 = nirs4all_io.summary(sp);
assert(ischar(s2) && ~isempty(strfind(s2, 'spec')), 'spec summary failed'); %#ok<STREMP>

% the plan's resolved_spec validates as a struct.
assert(nirs4all_io.validate(pl.resolved_spec) == true, 'resolved_spec rejected');

% InferOptions: singleton convention arrays and raw array refusal.
opts = struct();
opts.conventions = {'nirs4all-classic'};
encoded = nirs4all_io.internal.encode_infer_options(opts);
assert(strcmp(encoded, '{"conventions":["nirs4all-classic"]}'));
explicit_pl = nirs4all_io.infer(fullfile(corpus, 'single_combined'), opts);
assert(isequaln(explicit_pl.resolved_spec, pl.resolved_spec));
raw_plan = n4io('infer', json_path(fullfile(corpus, 'single_combined')), encoded);
assert(~isempty(strfind(raw_plan, '"resolved_spec"'))); %#ok<STREMP>
array_refused = false;
try
  n4io('infer', json_path(fullfile(corpus, 'single_combined')), '["nirs4all-classic"]');
catch err
  array_refused = ~isempty(strfind(err.message, 'invalid infer options')); %#ok<STREMP>
end
assert(array_refused, 'legacy infer array must be refused');
mixed = nirs4all_io.load_summary(explicit_pl.resolved_spec);
assert(mixed.blocks.train.n_samples == 6);
assert(isequal(mixed.blocks.train.x_shapes(:), [6; 12]));
assert(isequal(mixed.blocks.train.y(:), [12.5; 8.3; 15.1; 9.7; 11.2; 13.8]));
assert(strcmp(mixed.blocks.train.y_headers{1}, 'protein'));

% Preserve exact dotted/slash refs without converting map keys to struct fields.
opts.file_params = containers.Map({'dir/run.csv'}, {struct('has_header', false)});
encoded = nirs4all_io.internal.encode_infer_options(opts);
assert(~isempty(strfind(encoded, '"dir/run.csv":{"has_header":false}'))); %#ok<STREMP>
tmp = tempname;
mkdir(tmp);
cleanup_fixture = onCleanup(@() rmdir(tmp, 's'));
noheader = fullfile(tmp, 'noheader.csv');
fid = fopen(noheader, 'w');
assert(fid >= 0, 'cannot create headerless fixture');
fprintf(fid, '0.0465041;0.05;0.06\n0.07;0.08;0.09\n0.10;0.11;0.12\n');
fclose(fid);
opts.params = struct('has_header', true);
opts.file_params = containers.Map({'noheader.csv'}, {struct('has_header', false)});
hp = nirs4all_io.infer(noheader, opts);
assert(hp.resolved_spec.sources(1).params.has_header == false);
loaded = nirs4all_io.load_summary(hp.resolved_spec);
hb = loaded.blocks.train;
% Existing role inference selects two features and the last column as target.
assert(hb.n_samples == 3 && isequal(hb.x_shapes(:), [3; 2]));
assert(isequal(hb.y_shape(:), [3; 1]));
% The structural summary rounds values to four decimals.
assert(min(hb.x(:)) == 0.0465 && max(hb.x(:)) == 0.11);
assert(isequal(hb.y(:), [0.06; 0.09; 0.12]));
auto_header = nirs4all_io.infer(noheader);
assert(auto_header.resolved_spec.sources(1).params.has_header == true);
assert(any(cellfun(@(w) ~isempty(strfind(w, 'header_ambiguous')), auto_header.warnings))); %#ok<STREMP>
auto_loaded = nirs4all_io.load_summary(auto_header.resolved_spec);
assert(auto_loaded.blocks.train.n_samples == 2);
opts.file_params = containers.Map({'other.csv'}, {struct('has_header', false)});
unknown_refused = false;
try
  nirs4all_io.infer(noheader, opts);
catch err
  unknown_refused = ~isempty(strfind(err.message, 'unknown input')); %#ok<STREMP>
end
assert(unknown_refused, 'unknown file_params ref must be refused');

% a cellstr file list is accepted natively.
flist = {fullfile(corpus, 'x_y_separate', 'X.csv'), fullfile(corpus, 'x_y_separate', 'Y.csv')};
fpl = nirs4all_io.infer(flist);
assert(isstruct(fpl) && isfield(fpl, 'resolved_spec'), 'file-list infer failed');

lsum = nirs4all_io.load_summary(fullfile(corpus, 'train_test'));
assert(isstruct(lsum) && lsum.assembled_schema_version == 3, 'idiomatic load_summary failed');
% JSON null values decode as NaN in Octave; compare them as equal so this
% checks payload identity rather than IEEE NaN equality semantics.
assert(isequaln(lsum, nirs4all_io.load_summary(fullfile(corpus, 'train_test'), [], 'unlimited')));
limited = false;
try
  nirs4all_io.load_summary(fullfile(corpus, 'train_test'), [], struct('max_file_bytes', 1));
catch
  limited = true;
end
assert(limited, 'host read limits were ignored');

% validate rejects a bad spec struct.
bad_struct = struct('partitions', struct('by', 'random'));
rejected2 = false;
try
  nirs4all_io.validate(bad_struct);
catch
  rejected2 = true;
end
assert(rejected2, 'bad spec struct was not rejected');

disp('matlab/octave binding smoke OK');
