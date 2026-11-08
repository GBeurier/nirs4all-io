# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Public API: ``to_spec`` and ``load`` (Appendix I).

``load`` is the end-to-end entry: *resolve -> configure -> materialize*. It
accepts a ``DatasetSpec``, a dict/JSON/YAML config (alias-normalized), a
directory or file list (matched against conventions), or in-memory arrays, and
materializes a ``SpectroDataset`` (default), the target-agnostic
``AssembledDataset``, or a ``DatasetPackage``. Objects from sibling catalog
packages may also expose a ``to_io_spec()`` adapter; the returned spec is then
materialized by this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import strict_json
from .conventions.engine import match_items
from .conventions.profiles import resolve_profiles
from .conventions.to_spec import assignments_to_spec_dict
from .infer.describe import proposed_decimal
from .infer.identity import key_sources_by_id, refuse_generated_identity
from .materialize.assemble import AssembledDataset, PartitionBlock, _assemble_raw, assemble
from .materialize.loaders import proposed_delimiter
from .materialize.numeric import CONSUMER_SPECTRODATASET, NumericContractError, NumericStorageOptions, effective_coercion, to_ir_f64
from .materialize.package import DatasetPackage, finalize_selected_storage
from .materialize.spectrodataset import to_spectrodataset
from .resolve import resolve
from .resolve.resolver import _absolutize_file_references, _file_reference_table
from .spec import DatasetSpec
from .spec.dataset_spec import LoadingParams
from .spec.enums import SpecError
from .spec.normalize import normalize_to_spec_dict

_CONFIG_SUFFIXES = (".json", ".yaml", ".yml")
_DEFAULT_CONVENTIONS = ["nirs4all-classic"]


def to_spec(
    inp: Any,
    *,
    conventions: list[str] | None = None,
    base_dir: str | Path | None = None,
    name: str | None = None,
) -> tuple[DatasetSpec, Path]:
    """Normalize any (non-in-memory) input into a (DatasetSpec, base_dir) pair."""
    from .infer.plan import DatasetPlan

    adapted = _adapt_to_io_spec(inp, base_dir=base_dir)
    if adapted is not None:
        spec, base = adapted
        if name:
            spec.name = name
        return spec, base
    if isinstance(inp, DatasetPlan):
        # a plan's resolved_spec carries absolute input paths (Appendix I)
        return inp.accept(), Path(base_dir or ".")
    if isinstance(inp, DatasetSpec):
        return inp, Path(base_dir or ".")
    if isinstance(inp, dict):
        spec = DatasetSpec.from_dict(normalize_to_spec_dict(strict_json.check_numbers(inp)))
        if name:
            spec.name = name
        return spec, Path(base_dir or ".")
    if isinstance(inp, (str, Path)):
        path = Path(inp)
        if path.suffix.lower() in _CONFIG_SUFFIXES and path.is_file():

            import yaml

            text = path.read_text(encoding="utf-8")
            raw = strict_json.check_numbers(yaml.safe_load(text)) if path.suffix.lower() != ".json" else strict_json.loads(text)
            spec = DatasetSpec.from_dict(normalize_to_spec_dict(raw))
            spec.name = name or spec.name or path.stem
            return spec, Path(base_dir or path.parent)
        if path.is_dir():
            return _spec_from_directory(path, conventions or _DEFAULT_CONVENTIONS, name or path.name), path
        if path.is_file():
            spec = DatasetSpec.from_dict({"name": name or path.stem, "sources": [{"id": "data", "role": "features", "input": path.name}]})
            if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
                spec.sources[0].params.delimiter = proposed_delimiter(path, spec.sources[0].params)
                if not spec.sources[0].params.decimal_separator:
                    spec.sources[0].params.decimal_separator = proposed_decimal(path, spec.sources[0].params)
            return spec, path.parent
        raise SpecError(f"path does not exist: {inp}")
    if isinstance(inp, (list, tuple)) and all(isinstance(x, (str, Path)) for x in inp):
        # a list/glob of files -> resolve + convention match, with ABSOLUTE inputs
        # (files may live in different directories / share basenames)
        iset = resolve(list(inp))
        result = match_items(list(dict.fromkeys(iset.names)), resolve_profiles(conventions or _DEFAULT_CONVENTIONS))
        spec_dict = assignments_to_spec_dict(result, name=name or "dataset")
        refs = _file_reference_table(iset, spec_dict)
        _propose_source_delimiters(spec_dict, refs)
        key_sources_by_id(spec_dict, refs)
        _absolutize_file_references(spec_dict, refs)
        return DatasetSpec.from_dict(spec_dict), Path(base_dir or ".")
    raise SpecError(f"cannot build a DatasetSpec from {type(inp).__name__}; pass arrays to load() directly")


def _adapt_to_io_spec(inp: Any, *, base_dir: str | Path | None) -> tuple[DatasetSpec, Path] | None:
    """Accept sibling reference-dataset objects without importing their package.

    ``nirs4all-datasets.NirsDataset`` implements ``to_io_spec()``. Keeping this
    as a duck-typed adapter avoids a runtime dependency from IO back to the
    catalog package while still letting ``nio.load(reference_dataset)`` use the
    normal DatasetSpec materialization path.
    """
    adapter = getattr(inp, "to_io_spec", None)
    if not callable(adapter):
        return None
    raw = adapter()
    adapted_base: str | Path | None = None
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise SpecError("to_io_spec() must return a spec dict/DatasetSpec or a (spec, base_dir) pair")
        raw, adapted_base = raw
    if isinstance(raw, DatasetSpec):
        spec = raw
    elif isinstance(raw, dict):
        spec = DatasetSpec.from_dict(normalize_to_spec_dict(raw))
    else:
        raise SpecError(f"to_io_spec() returned {type(raw).__name__}, expected dict or DatasetSpec")
    return spec, Path(base_dir or adapted_base or ".")


def _propose_source_delimiters(spec: dict, refs: dict) -> None:
    """Configure each generated source from its own logically named file refs."""
    for source in spec.get("sources", []):
        inputs = source["input"] if isinstance(source["input"], list) else [source["input"]]
        params = LoadingParams.from_dict(source.get("params", {}))
        if not params.delimiter:
            proposals: list[str | None] = []
            for ref in inputs:
                item = refs[ref]
                if item.extension not in {".csv", ".tsv", ".txt"}:
                    proposals.append(None)
                    continue
                proposals.append(proposed_delimiter(Path(item.identity), params))
            # A source has one params block. Never configure multiple different
            # dialects using the first file's evidence; assembly will diagnose it.
            if proposals and proposals[0] and all(value == proposals[0] for value in proposals):
                source.setdefault("params", {})["delimiter"] = proposals[0]
        if len(inputs) == 1 and refs[inputs[0]].extension in {".csv", ".tsv", ".txt"}:
            params = LoadingParams.from_dict(source.get("params", {}))
            if not params.decimal_separator:
                decimal = proposed_decimal(Path(refs[inputs[0]].identity), params)
                if decimal == ",":
                    source.setdefault("params", {})["decimal_separator"] = decimal


def _spec_from_directory(path: Path, conventions: list[str], name: str) -> DatasetSpec:
    iset = resolve(path)
    result = match_items(iset.names, resolve_profiles(conventions))
    spec_dict = assignments_to_spec_dict(result, name=name)
    refs = _file_reference_table(iset, spec_dict)
    _propose_source_delimiters(spec_dict, refs)
    key_sources_by_id(spec_dict, refs)
    if not spec_dict.get("sources"):
        raise SpecError(f"no dataset files recognized in {path} with conventions {conventions}; unmatched: {result.unmatched}")
    return DatasetSpec.from_dict(spec_dict)


# --------------------------------------------------------------------------- #
# In-memory inputs (arrays / (X, y[, split]) / dict-of-arrays)                #
# --------------------------------------------------------------------------- #
def _is_in_memory(inp: Any) -> bool:
    if isinstance(inp, np.ndarray):
        return True
    if isinstance(inp, tuple) and inp and isinstance(inp[0], np.ndarray):
        return True
    if isinstance(inp, dict) and any(isinstance(v, np.ndarray) for v in inp.values()):
        return True
    return False


def _exact_float64(values: Any, request: Any = None) -> np.ndarray:
    """Exact F64 IR of in-memory features under the features storage request."""
    try:
        return to_ir_f64(values, request)
    except NumericContractError as error:
        raise SpecError(str(error)) from error


def _target_array(y: Any, request: Any) -> np.ndarray:
    """The target array, with raw cells inspected BEFORE numpy infers a dtype.

    Integer and string targets keep their raw semantics (no blanket cast, no encoder). Only a raw LIST that mixes
    integer and float cells is converted cell by cell through the shared selector (direct integer -> F32 when
    requested, exact F64 otherwise); every Python integer must lie in the shared domain.
    """
    if isinstance(y, (list, tuple)):
        cells = np.asarray(y, dtype=object)
        integers = [c for c in cells.flat if isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_))]
        floats = any(isinstance(c, (float, np.floating)) for c in cells.flat)
        if integers and floats:
            return to_ir_f64(cells, request)
        for cell in integers:
            strict_json.check_integer(int(cell))
    return np.asarray(y)


def _assemble_in_memory(inp: Any, name: str | None, options: NumericStorageOptions | None = None) -> AssembledDataset:
    """In-memory assembly, then the shared finalizer for the REQUESTED roles (see ``finalize_selected_storage``)."""
    return finalize_selected_storage(_assemble_in_memory_raw(inp, name, options), options)


def _assemble_in_memory_raw(inp: Any, name: str | None, options: NumericStorageOptions | None = None) -> AssembledDataset:
    options = effective_coercion(options)
    x = y = meta = split = None
    if isinstance(inp, np.ndarray):
        x = inp
    elif isinstance(inp, tuple):
        x = inp[0]
        y = inp[1] if len(inp) > 1 else None
        split = inp[2] if len(inp) > 2 else None
    elif isinstance(inp, dict):
        x = inp.get("X") if "X" in inp else inp.get("x")
        y = inp.get("y") if "y" in inp else inp.get("Y")
        meta = inp.get("metadata") if "metadata" in inp else inp.get("meta")
    # The source dtype is captured from the real input BEFORE the exact F64 widening:
    # a typed ndarray declares its dtype, an untyped list declares nothing.
    x_declared = x.dtype.name if isinstance(x, np.ndarray) else None
    y_declared = y.dtype.name if isinstance(y, np.ndarray) else None
    x = _exact_float64(x, options.features)
    y_array = _target_array(y, options.targets) if y is not None else None
    headers = [str(i) for i in range(x.shape[1])]

    meta_df = None
    if meta is not None:
        import pandas as pd

        meta_df = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(np.asarray(meta))
        meta_df.columns = [str(c) for c in meta_df.columns]

    def _block(rows: np.ndarray) -> PartitionBlock:
        b = PartitionBlock(n_samples=int(rows.sum()) if rows.dtype == bool else len(rows))
        idx = rows
        b.source_ids = ["X"]
        b.X = [x[idx]]
        b.x_source_dtypes = [x_declared]
        b.feature_names = [None]  # an ndarray has genuinely unnamed columns; the "0", "1", ... headers are presentation only
        b.feature_axes = [[{"name": "feature", "unit": None, "coordinates": None}]]
        b.representation_ids = ["tabular_numeric"]
        b.feature_headers = [headers]
        b.header_units = ["index"]
        b.signal_types = [None]
        if y_array is not None:
            yarr = y_array
            yarr = yarr.reshape(-1, 1) if yarr.ndim == 1 else yarr
            b.y = yarr[idx]
            b.y_source_dtype = y_declared
            b.y_headers = [f"y{i}" for i in range(b.y.shape[1])]
        if meta_df is not None:
            b.metadata = meta_df.iloc[idx].reset_index(drop=True)
        return b

    assembled = AssembledDataset(name=name or "array_dataset", task_type="auto", signal_type="auto", n_sources=1)
    n = len(x)
    if split is None:
        partition = y is None and "predict" or "train"  # X-only -> predict, (X,y) -> train
        assembled.blocks[partition] = _block(np.ones(n, dtype=bool))
    else:
        split = np.asarray(split)
        for part in ("train", "test", "predict"):
            mask = split == part
            if mask.any():
                assembled.blocks[part] = _block(mask)
    return assembled


# --------------------------------------------------------------------------- #
# load                                                                         #
# --------------------------------------------------------------------------- #
def load(
    inp: Any,
    *,
    target: str = "spectrodataset",
    conventions: list[str] | None = None,
    base_dir: str | Path | None = None,
    name: str | None = None,
    spectro_dataset_cls: type | None = None,
    features: Any = None,
    targets: Any = None,
    weights: Any = None,
    consumer: str | None = None,
) -> Any:
    """Materialize ``inp`` into a dataset (default target: a nirs4all SpectroDataset).

    ``features`` / ``targets`` / ``weights`` are optional per-role numeric storage
    requests (``"float32"`` / ``"float64"``); ``consumer`` names a consumer boundary
    for packages. The SpectroDataset target applies its own F32 consumer policy and
    accepts the three role requests only (see ``project_spectrodataset``).
    """
    # ``request`` is the caller's REAL request (role storage + consumer); the raw assembly derives its own
    # effective integer coercion from it, so the precision records stay truthful. Only the ``assembled`` exit
    # finalizes the requested roles here; ``package`` selects once in ``from_assembled`` and ``spectrodataset``
    # keeps its own ``project_spectrodataset`` boundary (no package-level selection, no new refusal paths).
    request = NumericStorageOptions(features, targets, weights, consumer)
    finalize = target == "assembled"
    raw_request = NumericStorageOptions(features, targets, weights, CONSUMER_SPECTRODATASET) if target == "spectrodataset" else request
    if _is_in_memory(inp):
        assembled = _assemble_in_memory(inp, name, request) if finalize else _assemble_in_memory_raw(inp, name, raw_request)
    elif isinstance(inp, DatasetPackage):
        if target in ("dataset_package", "package") or finalize:
            if any(option is not None for option in (features, targets, weights, consumer)):
                raise SpecError("a built DatasetPackage already fixed its numeric storage; build a new package to request another precision")
            if not finalize:
                return inp
        assembled = inp.to_assembled()
    else:
        spec, base = to_spec(inp, conventions=conventions, base_dir=base_dir, name=name)
        spec.validate()
        from .infer.plan import DatasetPlan

        automatic = isinstance(inp, (list, tuple, DatasetPlan)) or isinstance(inp, (str, Path)) and Path(inp).suffix.lower() not in _CONFIG_SUFFIXES
        if automatic and spec.sample_index.by.value == "id":
            iset = resolve([str(base / value) for source in spec.sources for value in (source.input if isinstance(source.input, list) else [source.input])])
            refs = _file_reference_table(iset, spec.to_dict())
            refuse_generated_identity(spec, refs)
        assembled = assemble(spec, base, request) if finalize else _assemble_raw(spec, base, raw_request)

    if target == "assembled":
        return assembled
    if target in ("dataset_package", "package"):
        return DatasetPackage.from_assembled(assembled, features=features, targets=targets, weights=weights, consumer=consumer)
    if target == "spectrodataset":
        return to_spectrodataset(assembled, spectro_dataset_cls=spectro_dataset_cls, features=features, targets=targets, weights=weights)
    if target in ("dag-ml-data", "dag_ml_data"):
        raise NotImplementedError(
            "target 'dag-ml-data' is not exposed by the Python MVP load() surface; use the Rust bridge crate "
            "`crates/nirs4all-io-dagml` (`to_dag_ml_data` / `emit-dagml`)"
        )
    raise SpecError(f"unknown target {target!r}; expected 'spectrodataset' | 'assembled' | 'dataset_package' | 'package'")


def to_dataset_package(
    inp: Any,
    *,
    conventions: list[str] | None = None,
    base_dir: str | Path | None = None,
    name: str | None = None,
    features: Any = None,
    targets: Any = None,
    weights: Any = None,
    consumer: str | None = None,
) -> DatasetPackage:
    """Materialize ``inp`` into a public, target-agnostic :class:`DatasetPackage`."""
    if isinstance(inp, DatasetPackage):
        if any(option is not None for option in (features, targets, weights, consumer)):
            raise SpecError("a built DatasetPackage already fixed its numeric storage; build a new package to request another precision")
        return inp
    if isinstance(inp, AssembledDataset):
        return DatasetPackage.from_assembled(inp, features=features, targets=targets, weights=weights, consumer=consumer)
    package = load(inp, target="dataset_package", conventions=conventions, base_dir=base_dir, name=name, features=features, targets=targets, weights=weights, consumer=consumer)
    if not isinstance(package, DatasetPackage):  # defensive guard for type checkers and future targets
        raise SpecError(f"target 'dataset_package' returned {type(package).__name__}, expected DatasetPackage")
    return package


def describe_dataset_package(
    inp: Any,
    *,
    conventions: list[str] | None = None,
    base_dir: str | Path | None = None,
    name: str | None = None,
    canonical: bool = False,
    features: Any = None,
    targets: Any = None,
    weights: Any = None,
    consumer: str | None = None,
) -> dict[str, Any] | str:
    """Return the bytes-free ``DatasetPackage`` summary for ``inp``.

    Set ``canonical=True`` to receive canonical JSON; otherwise a JSON-ready dict
    is returned.
    """
    package = to_dataset_package(inp, conventions=conventions, base_dir=base_dir, name=name, features=features, targets=targets, weights=weights, consumer=consumer)
    return package.to_canonical_summary() if canonical else package.to_summary_dict()
