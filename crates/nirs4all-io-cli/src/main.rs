// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Command-line front end for `nirs4all-io`.
//!
//! Subcommand handlers are wired in EPIC 9 (T9) once the facade lands; this
//! skeleton fixes the command surface (`infer` / `to-spec` / `validate` /
//! `load` / `emit-dag-ml-data`) so it is stable across the rewrite.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nirs4all-io", version, about = "Dataset-assembly bridge CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect an input and emit a scored DatasetPlan as JSON.
    Infer { input: String },
    /// Normalize an input into a DatasetSpec and emit it as JSON.
    ToSpec { input: String },
    /// Validate a DatasetSpec JSON file.
    Validate { input: String },
    /// Materialize an input into a dataset.
    Load { input: String },
    /// Emit a dag-ml-data CoordinatorDataPlanEnvelope (feature `dag-ml-data`).
    EmitDagMlData { input: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Infer { .. }
        | Command::ToSpec { .. }
        | Command::Validate { .. }
        | Command::Load { .. }
        | Command::EmitDagMlData { .. } => {
            anyhow::bail!("nirs4all-io-cli subcommands are implemented in EPIC 9 (T9)")
        }
    }
}
