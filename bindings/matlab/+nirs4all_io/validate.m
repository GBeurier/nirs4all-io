% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function validate(spec_json)
  % Validate a DatasetSpec JSON string; errors if invalid.
  n4io('validate', spec_json);
end
