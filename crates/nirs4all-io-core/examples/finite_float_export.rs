// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Qualification helper: stdin has one finite binary64 bit pattern per line.
//! Output is JSON-lines containing actual owner strings, without host rewriting.
use nirs4all_io_core::{canonical_json::canonical_json, materialize::Cell, pyfmt::py_float_repr};
use serde_json::json;
use std::io::{self, BufRead, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let bits = u64::from_str_radix(line.trim(), 16)?;
        let value = f64::from_bits(bits);
        if !value.is_finite() {
            return Err("finite binary64 required".into());
        }
        let result = json!({"bits_hex":format!("{bits:016x}"),
            "canonical":canonical_json(&json!(value))?,
            "repr":py_float_repr(value), "cell":Cell::Float(value).to_str_scalar()});
        serde_json::to_writer(&mut output, &result)?;
        output.write_all(b"\n")?;
    }
    Ok(())
}
