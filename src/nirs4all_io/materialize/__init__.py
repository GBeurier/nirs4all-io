# SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
"""Materialization: merge/join engine, loaders, and target builders (Epic 4)."""

from .join import (
    JoinAudit,
    JoinError,
    concat_features,
    concat_samples,
    join_tables,
    merge_by_key,
)
from .loaders import (
    LoadedTable,
    LoaderError,
    NAError,
    apply_na_policy,
    coerce_numeric,
    effective_params,
    encode_categorical,
    infer_dtypes,
    read_table,
)

__all__ = [
    # join engine
    "JoinAudit",
    "JoinError",
    "concat_samples",
    "concat_features",
    "join_tables",
    "merge_by_key",
    # loaders
    "LoadedTable",
    "LoaderError",
    "NAError",
    "read_table",
    "apply_na_policy",
    "effective_params",
    "infer_dtypes",
    "coerce_numeric",
    "encode_categorical",
]
