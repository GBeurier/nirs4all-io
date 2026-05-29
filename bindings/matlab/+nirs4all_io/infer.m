% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function plan = infer(input_json, conventions_json)
  % Inspect a data input and return the scored DatasetPlan (JSON string).
  if nargin < 2, conventions_json = ''; end
  plan = n4io('infer', input_json, conventions_json);
end
