// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! The canonical `DatasetSpec` IR and its closed vocabularies.
//!
//! Ports `nirs4all_io.spec` (enums, selectors, normalization, validation, the
//! IR, and its JSON schema). Pure: no file IO. The materializers consume only
//! the IR.

pub mod dataset_spec;
pub mod enums;
pub mod selectors;

pub use dataset_spec::DatasetSpec;
pub use enums::SpecError;
