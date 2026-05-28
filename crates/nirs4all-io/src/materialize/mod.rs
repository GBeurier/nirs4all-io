// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Materialization: file IO that turns a spec into data (ports `nirs4all_io.materialize`).
//!
//! Tabular loaders today; the relational join, assembler, folds, and the
//! SpectroDataset adapter land with the load path.

pub mod folds;
pub mod frame;
pub mod join;
pub mod loaders;

pub use folds::{parse_fold_file, Fold};
pub use frame::{Cell, Column, Frame, Matrix};
pub use join::{concat_features, concat_samples, join_tables, merge_by_key, JoinAudit};
pub use loaders::{read_table, LoadedTable};
