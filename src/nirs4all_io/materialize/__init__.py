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

__all__ = [
    "JoinAudit",
    "JoinError",
    "concat_samples",
    "concat_features",
    "join_tables",
    "merge_by_key",
]
