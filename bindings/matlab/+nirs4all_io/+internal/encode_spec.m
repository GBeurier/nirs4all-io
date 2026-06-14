% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function json = encode_spec(spec)
  % ENCODE_SPEC  Encode a DatasetSpec struct while preserving JSON arrays.
  %
  % Octave decodes one-element JSON object arrays as scalar structs. Without
  % restoring the known spec array fields, jsonencode turns sources/columns back
  % into objects and the Rust C ABI sees an empty DatasetSpec.
  json = jsonencode(preserve_arrays(spec, ''));
end

function out = preserve_arrays(value, field_name)
  if isstruct(value)
    if is_array_field(field_name) && should_encode_as_array(value, field_name)
      out = cell(1, numel(value));
      for i = 1:numel(value)
        out{i} = preserve_struct_fields(value(i));
      end
    else
      out = value;
      for i = 1:numel(value)
        out(i) = preserve_struct_fields(value(i));
      end
    end
  elseif iscell(value)
    out = value;
    for i = 1:numel(value)
      out{i} = preserve_arrays(value{i}, field_name);
    end
  else
    out = value;
  end
end

function out = preserve_struct_fields(s)
  out = s;
  names = fieldnames(s);
  for j = 1:numel(names)
    name = names{j};
    out.(name) = preserve_arrays(s.(name), name);
  end
end

function tf = is_array_field(name)
  tf = any(strcmp(name, {'sources', 'columns', 'variations', 'inline'}));
end

function tf = should_encode_as_array(value, field_name)
  if isempty(value)
    tf = true;
    return;
  end
  if strcmp(field_name, 'columns')
    names = fieldnames(value);
    tf = any(strcmp(names, 'role')) && any(strcmp(names, 'select'));
  else
    tf = true;
  end
end
