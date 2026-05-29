% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function spec = to_spec(input_json, conventions_json)
  % Normalize an input into a canonical DatasetSpec (JSON string).
  if nargin < 2, conventions_json = ''; end
  spec = n4io('to_spec', input_json, conventions_json);
end
