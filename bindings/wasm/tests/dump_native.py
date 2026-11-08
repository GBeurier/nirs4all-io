"""Private fixture dumper; run only by Root in the current installed IO wheel environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True, help='New private directory; must not exist')
    args = parser.parse_args()
    fixture_dir = Path(__file__).resolve().parent / 'fixtures'
    fixture_paths = [fixture_dir / 'case1.json', fixture_dir / 'quali-owner-case.json']
    case1, quali = [json.loads(p.read_bytes()) for p in fixture_paths]

    # No oracle, CLI, SDK adapter or checkout fallback: this is the installed native export.
    from nirs4all_io import _native

    dist = distribution('nirs4all-io')
    assert dist.version == '0.3.0', dist.version
    native_path = Path(_native.__file__).resolve()
    assert 'site-packages' in native_path.parts, native_path
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def materialize(label, named_files, spec):
        directory = out / label
        directory.mkdir()
        for named in named_files:
            # Names are literal private fixtures, not untrusted input or a CSV parser.
            assert Path(named['name']).name == named['name']
            with (directory / named['name']).open('xb') as stream:
                stream.write(named['text'].encode('utf-8'))
        with working_directory(directory):
            _native.validate(spec)
            return {'normalized_spec': _native.to_spec(spec), 'full': _native.assembled_full(spec)}

    first = materialize('case1', case1['files'], case1['spec'])
    refusal = json.loads(json.dumps(quali['spec']))
    refusal['sources'][2] = quali['original_three_file_refusal']['target_source_replacement']
    directory = out / 'case2-original'
    directory.mkdir()
    for named in quali['files']:
        with (directory / named['name']).open('xb') as stream:
            stream.write(named['text'].encode('utf-8'))
    with working_directory(directory):
        try:
            _native.assembled_full(refusal)
        except ValueError as error:
            refusal_message = str(error)
        else:
            raise AssertionError('Original three-file case must refuse, never broadcast positionally')
    assert refusal_message == quali['original_three_file_refusal']['message'], refusal_message
    second = materialize('case2-confirmed',
                         [*quali['files'], quali['in_memory_witness']], quali['spec'])
    second['original_refusal'] = refusal_message
    result = {
        'native': {'version': dist.version, 'module': str(native_path),
                   'sha256': sha256(native_path.read_bytes()).hexdigest(), 'python': sys.executable},
        'fixture_sha256': [sha256(p.read_bytes()).hexdigest() for p in fixture_paths],
        'case1': first, 'case2': second,
    }
    # Full-value numeric DTO carries canonical bytes/masks, not logical NaN JSON placeholders.
    with (out / 'native-full.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(out / 'native-full.json')


if __name__ == '__main__':
    main()
