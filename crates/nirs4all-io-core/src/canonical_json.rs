// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Canonical JSON (story 7.3) — the cross-language wire contract.
//!
//! The single biggest correctness risk in the rewrite is that Python `json`
//! and Rust `serde_json` disagree byte-for-byte. We pin a canonical form and
//! verify it with a parity gate (`tests/canonical_parity.rs` here and
//! `tests/test_canonical_json.py` on the Python side, both comparing against the
//! same blessed expected files).
//!
//! ## Canonical form
//! - **Encoding:** UTF-8, no BOM. Non-ASCII is emitted verbatim (NOT
//!   `\uXXXX`-escaped); the Python side sets `ensure_ascii=False` to match.
//! - **Key order:** lexicographic by Unicode code point. The workspace enables
//!   serde_json's `preserve_order` so *input* map order mirrors Python dict
//!   insertion order (load-bearing for legacy-config normalization). Therefore
//!   `canonical_json` sorts object keys EXPLICITLY — it cannot rely on the map
//!   type. Rust `str` `Ord` is byte-wise UTF-8 = code-point order, matching
//!   Python `sort_keys=True`.
//! - **Indent:** pretty, two spaces, `": "` after keys, one element per line.
//!   `serde_json::to_string_pretty` and Python `json.dumps(indent=2)` produce
//!   identical layout (verified by the parity gate).
//! - **Line endings:** `\n`, with a single trailing `\n`.
//! - **Floats:** every finite binary64 value uses shortest round-trip digits,
//!   closest to the exact value with ties choosing an even last digit. Scientific
//!   exponents -4 through 15 use fixed notation; others use lowercase `e`, an
//!   explicit sign and at least two exponent digits. Integral fixed floats keep
//!   `.0`, and signed zero is preserved. Nonfinite values must refuse before
//!   conversion to `Value` (serde otherwise replaces them with null).

use serde::Serialize;
use serde_json::ser::{CompactFormatter, Formatter, PrettyFormatter};
use std::io::{self, Write};
use thiserror::Error;

/// Errors raised while producing canonical JSON.
#[derive(Debug, Error)]
pub enum CanonicalJsonError {
    /// The value could not be serialized to JSON.
    #[error("JSON serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Canonicalize an already-built [`serde_json::Value`].
///
/// Objects are re-emitted with sorted keys (guaranteed by `serde_json`'s
/// default `BTreeMap` map type), two-space pretty indentation, and a single
/// trailing newline.
pub fn canonical_json(value: &serde_json::Value) -> Result<String, CanonicalJsonError> {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(
        &mut out,
        CanonicalFormatter(PrettyFormatter::new()),
    );
    sort_keys(value).serialize(&mut serializer)?;
    out.push(b'\n');
    Ok(String::from_utf8(out).expect("JSON serializer emits UTF-8"))
}

/// Compact source-schema identity JSON with the same key and float rules.
pub(crate) fn canonical_json_compact(
    value: &serde_json::Value,
) -> Result<String, serde_json::Error> {
    let mut out = Vec::new();
    let mut serializer =
        serde_json::Serializer::with_formatter(&mut out, CanonicalFormatter(CompactFormatter));
    sort_keys(value).serialize(&mut serializer)?;
    Ok(String::from_utf8(out).expect("JSON serializer emits UTF-8"))
}

/// Shared finite-float spelling for JSON and Python-compatible scalar text.
/// Serde supplies shortest, closest, ties-even digits; only layout changes here.
pub(crate) fn finite_float_repr(value: f64) -> String {
    debug_assert!(value.is_finite());
    let mut raw = Vec::new();
    CompactFormatter
        .write_f64(&mut raw, value)
        .expect("Vec writes cannot fail");
    let raw = std::str::from_utf8(&raw).expect("float digits are ASCII");
    let (sign, unsigned) = raw.strip_prefix('-').map_or(("", raw), |v| ("-", v));
    let (mantissa, exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, 0), |(m, e)| {
            (m, e.parse::<i32>().expect("serde exponent is an integer"))
        });
    let decimal = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let digits = mantissa.replace('.', "");
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return format!("{sign}0.0");
    }
    let exponent = exponent + decimal - (digits.len() - significant.len()) as i32 - 1;
    let significant = significant.trim_end_matches('0');
    let mut out = sign.to_owned();
    if (-4..=15).contains(&exponent) {
        let point = exponent + 1;
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(significant);
        } else if (point as usize) < significant.len() {
            let (left, right) = significant.split_at(point as usize);
            out.push_str(left);
            out.push('.');
            out.push_str(right);
        } else {
            out.push_str(significant);
            out.extend(std::iter::repeat_n('0', point as usize - significant.len()));
            out.push_str(".0");
        }
    } else {
        out.push_str(&significant[..1]);
        if significant.len() > 1 {
            out.push('.');
            out.push_str(&significant[1..]);
        }
        out.push('e');
        out.push_str(&format!("{exponent:+03}"));
    }
    out
}

struct CanonicalFormatter<F>(F);

macro_rules! forward_layout {
    ($($name:ident),+ $(,)?) => {$ (
        fn $name<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
            self.0.$name(writer)
        }
    )+};
}
impl<F: Formatter> Formatter for CanonicalFormatter<F> {
    forward_layout!(
        begin_array,
        end_array,
        end_array_value,
        begin_object,
        end_object,
        end_object_key,
        begin_object_value,
        end_object_value
    );
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_array_value(writer, first)
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_object_key(writer, first)
    }
    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(finite_float_repr(value).as_bytes())
    }
    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }
}

/// Decode JSON text without silently converting an out-of-range integer token
/// to F64. Integer tokens must fit i64 or u64; float tokens retain serde's bits,
/// including negative zero. Already-decoded `Value` inputs cannot recover lost
/// original lexemes and are outside this text guard's authority.
pub fn parse_json_value(text: &str) -> serde_json::Result<serde_json::Value> {
    let value = serde_json::from_str(text)?;
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index += 1;
                while bytes[index] != b'"' {
                    index += if bytes[index] == b'\\' { 2 } else { 1 };
                }
                index += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = index;
                while index < bytes.len()
                    && matches!(bytes[index], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    index += 1;
                }
                let token = &text[start..index];
                if !token.contains(['.', 'e', 'E']) {
                    let (digits, limit) = token
                        .strip_prefix('-')
                        .map_or((token, "18446744073709551615"), |digits| {
                            (digits, "9223372036854775808")
                        });
                    if digits.len() > limit.len() || (digits.len() == limit.len() && digits > limit)
                    {
                        return Err(<serde_json::Error as serde::de::Error>::custom(
                            "JSON integer token outside i64/u64 range",
                        ));
                    }
                } else {
                    crate::materialize::numeric::checked_decimal_f64(token)
                        .map_err(<serde_json::Error as serde::de::Error>::custom)?;
                }
            }
            _ => index += 1,
        }
    }
    Ok(value)
}

/// Recursively rebuild `value` with object keys sorted lexicographically.
///
/// Required because the workspace enables `preserve_order`, so the default map
/// is insertion-ordered; we re-key it in sorted order before serializing.
fn sort_keys(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), sort_keys(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

/// Serialize any [`Serialize`] value to canonical JSON.
///
/// Routes through [`serde_json::Value`] so key ordering is normalized
/// regardless of struct field declaration order.
pub fn to_canonical_json<T: Serialize>(value: &T) -> Result<String, CanonicalJsonError> {
    let value = serde_json::to_value(value)?;
    canonical_json(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_sorted_unicode_escapes_and_float_spelling() {
        let value = json!({
            "😀": 2,
            "\u{e000}": 1,
            "z": [{"b": 900.0, "a": 900}],
            "a": ["é\"\\\n\u{0001}\u{2028}", -0.0, 1e-07, 1e+20]
        });
        assert_eq!(
            canonical_json_compact(&value).unwrap(),
            "{\"a\":[\"é\\\"\\\\\\n\\u0001\u{2028}\",-0.0,1e-07,1e+20],\"z\":[{\"a\":900,\"b\":900.0}],\"\u{e000}\":1,\"😀\":2}"
        );
    }

    #[test]
    fn generic_formatter_keeps_f32_widening_for_both_layouts() {
        for (value, expected) in [(1.2_f32, "1.2000000476837158"), (-0.0_f32, "-0.0")] {
            let mut compact = Vec::new();
            value
                .serialize(&mut serde_json::Serializer::with_formatter(
                    &mut compact,
                    CanonicalFormatter(CompactFormatter),
                ))
                .unwrap();
            let mut pretty = Vec::new();
            value
                .serialize(&mut serde_json::Serializer::with_formatter(
                    &mut pretty,
                    CanonicalFormatter(PrettyFormatter::new()),
                ))
                .unwrap();
            assert_eq!(compact, expected.as_bytes());
            assert_eq!(pretty, expected.as_bytes());
        }
    }

    #[test]
    fn keys_are_sorted_regardless_of_insertion_order() {
        let v = json!({"b": 1, "a": 2, "c": {"z": 1, "y": 2}});
        let out = canonical_json(&v).unwrap();
        assert_eq!(
            out,
            "{\n  \"a\": 2,\n  \"b\": 1,\n  \"c\": {\n    \"y\": 2,\n    \"z\": 1\n  }\n}\n"
        );
    }

    #[test]
    fn empty_containers_and_trailing_newline() {
        let v = json!({"arr": [], "obj": {}});
        let out = canonical_json(&v).unwrap();
        assert_eq!(out, "{\n  \"arr\": [],\n  \"obj\": {}\n}\n");
    }

    #[test]
    fn floats_keep_decimal_point_and_ints_stay_ints() {
        let v = json!({"i": 1, "f": 1.0, "s": 0.95});
        let out = canonical_json(&v).unwrap();
        assert_eq!(out, "{\n  \"f\": 1.0,\n  \"i\": 1,\n  \"s\": 0.95\n}\n");
    }

    #[test]
    fn unicode_is_emitted_verbatim() {
        let v = json!({"name": "café—é"});
        let out = canonical_json(&v).unwrap();
        assert_eq!(out, "{\n  \"name\": \"café—é\"\n}\n");
    }
}
