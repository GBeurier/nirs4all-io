% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function plan = infer(input, options)
  % INFER  Scored dataset plan from a native MATLAB/Octave input.
  %
  %   plan = nirs4all_io.infer(input)
  %   plan = nirs4all_io.infer(input, options)
  %
  % INPUT accepts a native value: a char path ('/data/run'), a cellstr of files
  % ({'a.csv','b.csv'}), or a struct config. It is jsonencoded internally and
  % the scored DatasetPlan is jsondecoded back into a struct (not a raw JSON
  % char). OPTIONS is empty or a scalar struct with optional conventions
  % (cellstr), params (scalar struct), and file_params (containers.Map keyed by
  % exact input refs, with scalar parameter structs as values).
  %
  % Use nirs4all_io.summary(plan) for a readable overview, and
  % plan.resolved_spec for the editable spec.
  %
  % The raw JSON surface stays reachable as n4io('infer', input_json, ...).
  if nargin < 2, options = []; end
  input_json = nirs4all_io.internal.encode_input(input);
  options_json = nirs4all_io.internal.encode_infer_options(options);
  plan = jsondecode(n4io('infer', input_json, options_json));
end
