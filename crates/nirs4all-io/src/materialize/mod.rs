// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Materialization: file IO that turns a spec into data (ports `nirs4all_io.materialize`).
//!
//! Tabular loaders today; the relational join, assembler, folds, and the
//! SpectroDataset adapter land with the load path.

pub mod loaders;

pub use loaders::{read_table, LoadedTable};
