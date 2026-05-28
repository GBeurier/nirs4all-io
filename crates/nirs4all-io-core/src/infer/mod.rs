// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Inference over neutral descriptions (ports the pure parts of `nirs4all_io.infer`).
//!
//! The pure scorers/detectors and the `DatasetPlan` live here; the facade
//! populates `FileDescription`/`TableProfile` from files and orchestrates the
//! `infer()` pipeline.

pub mod describe;
pub mod detectors;
pub mod plan;

pub use describe::{describe_text, FileDescription};
pub use detectors::{detect_signal_type, detect_task_type};
pub use plan::{DatasetPlan, Decision};
