% SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
function json = encode_infer_options(options)
  % ENCODE_INFER_OPTIONS  Empty -> NULL; declared options -> InferOptions JSON.
  % Encode map keys individually: converting them to struct fields can change
  % dotted/slash input refs. Do not depend on host jsonencode(Map) behavior.
  if nargin < 1 || isempty(options)
    json = '';
    return;
  end
  if ~isstruct(options) || ~isscalar(options)
    error('n4io:arg', 'infer options must be a scalar struct or empty');
  end
  names = fieldnames(options);
  if any(~ismember(names, {'conventions', 'params', 'file_params'}))
    error('n4io:arg', 'unknown infer options field');
  end
  fields = {};
  if isfield(options, 'conventions')
    if ~iscellstr(options.conventions)
      error('n4io:arg', 'options.conventions must be a cellstr');
    end
    % Explicit brackets keep a singleton convention an array on both hosts.
    items = cellfun(@jsonencode, options.conventions, 'UniformOutput', false);
    fields{end + 1} = ['"conventions":[' strjoin(items(:)', ',') ']'];
  end
  if isfield(options, 'params')
    if ~isstruct(options.params) || ~isscalar(options.params)
      error('n4io:arg', 'options.params must be a scalar struct');
    end
    fields{end + 1} = ['"params":' jsonencode(options.params)];
  end
  if isfield(options, 'file_params')
    map = options.file_params;
    if ~isa(map, 'containers.Map') || ~strcmp(map.KeyType, 'char')
      error('n4io:arg', 'options.file_params must be a containers.Map with char keys');
    end
    refs = keys(map);
    entries = cell(1, numel(refs));
    for i = 1:numel(refs)
      value = map(refs{i});
      if ~isstruct(value) || ~isscalar(value)
        error('n4io:arg', 'file_params values must be scalar parameter structs');
      end
      entries{i} = [jsonencode(refs{i}) ':' jsonencode(value)];
    end
    fields{end + 1} = ['"file_params":{' strjoin(entries, ',') '}'];
  end
  if isempty(fields)
    json = '';
  else
    json = ['{' strjoin(fields, ',') '}'];
  end
end
