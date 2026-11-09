// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Owner precision selection and exact masked numeric transport.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericStorage {
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "float64")]
    Float64,
}
impl NumericStorage {
    pub fn dtype(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
    pub fn tag(self) -> u8 {
        match self {
            Self::Float32 => 1,
            Self::Float64 => 2,
        }
    }
    pub fn integer(self, value: i128) -> Result<f64, String> {
        match self {
            Self::Float64 => exact_integer_f64(value),
            Self::Float32 => {
                let rounded = value as f32;
                if !rounded.is_finite() {
                    return Err("Integer overflows float32 storage".into());
                }
                Ok(f64::from(rounded))
            }
        }
    }
    /// Convert retained numeric text only at a numeric role boundary.
    pub(crate) fn decimal(self, text: &str) -> Result<f64, String> {
        let text = text.trim();
        let body = text.strip_prefix(['+', '-']).unwrap_or(text);
        if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
            if !integer_in_domain(text) {
                return Err("Integer numeric text is outside the supported i64/u64 domain".into());
            }
            if body.bytes().all(|b| b == b'0') {
                return self.round(checked_decimal_f64(text)?);
            }
            return self.integer(
                text.parse::<i128>()
                    .expect("admitted i64/u64 integer fits i128"),
            );
        }
        self.round(checked_decimal_f64(text)?)
    }
    pub fn json_number(self, value: &Value, source_dtype: Option<&str>) -> Result<f64, String> {
        if source_dtype == Some("float64") {
            // A storage request cannot rescue an invalid declared source.
            return self.round(Self::Float64.json_number(value, None)?);
        }
        if source_dtype == Some("float32") {
            let normalized = Self::Float32.json_number(value, None)?;
            return self.round(normalized);
        }
        // A declaration is not a cast: retained integer inputs must reach the
        // selected storage directly before any possibly lossy F64 conversion.
        if let Some(integer) = value.as_i64() {
            return self.integer(i128::from(integer));
        }
        if let Some(integer) = value.as_u64() {
            return self.integer(i128::from(integer));
        }
        self.round(value.as_f64().ok_or("Numeric value required")?)
    }
    pub fn round(self, value: f64) -> Result<f64, String> {
        if !value.is_finite() {
            return Err("Observed numeric values must be finite".into());
        }
        if self == Self::Float64 {
            return Ok(value);
        }
        let rounded = value as f32;
        if !rounded.is_finite() {
            return Err("Value overflows float32 storage".into());
        }
        if value != 0.0 && rounded == 0.0 {
            return Err("Nonzero value rounds to zero in float32 storage".into());
        }
        Ok(f64::from(rounded))
    }
}

/// Retained integer text is limited to the JSON integer transport domain.
pub fn integer_in_domain(text: &str) -> bool {
    let text = text.trim();
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let digits = digits.trim_start_matches('0');
    let max = if text.starts_with('-') {
        "9223372036854775808"
    } else {
        "18446744073709551615"
    };
    digits.len() < max.len() || (digits.len() == max.len() && digits <= max)
}

/// The one decimal-text admission primitive: standard binary64 parsing, finite
/// admission and nonzero-mantissa underflow refusal. Strings are checked only
/// when an owner has identified a numeric role, never as general metadata.
pub(crate) fn checked_decimal_f64(text: &str) -> Result<f64, String> {
    let value = text
        .parse::<f64>()
        .map_err(|_| "Numeric decimal value required")?;
    if !value.is_finite() {
        return Err("Observed numeric values must be finite".into());
    }
    if value == 0.0
        && text
            .split(['e', 'E'])
            .next()
            .unwrap_or("")
            .bytes()
            .any(|b| matches!(b, b'1'..=b'9'))
    {
        return Err("Nonzero numeric text rounds to zero in float64".into());
    }
    Ok(value)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericStorageOptions {
    pub features: Option<NumericStorage>,
    pub targets: Option<NumericStorage>,
    pub weights: Option<NumericStorage>,
    pub consumer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrecisionOrigin {
    Default,
    SourceDeclared,
    Requested,
    Consumer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct PrecisionRecord {
    pub dtype: NumericStorage,
    pub origin: PrecisionOrigin,
    pub source_dtype: Option<String>,
    pub requested_dtype: Option<NumericStorage>,
    pub consumer: Option<String>,
}
impl Default for PrecisionRecord {
    fn default() -> Self {
        Self {
            dtype: NumericStorage::Float64,
            origin: PrecisionOrigin::Default,
            source_dtype: None,
            requested_dtype: None,
            consumer: None,
        }
    }
}
impl PrecisionRecord {
    pub fn select(
        source: Option<&str>,
        request: Option<NumericStorage>,
        consumer: Option<&str>,
        feature: bool,
    ) -> Result<Self, String> {
        if source.is_some_and(|s| {
            !matches!(
                s,
                "float32"
                    | "float64"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "int64"
                    | "uint8"
                    | "uint16"
                    | "uint32"
                    | "uint64"
            )
        }) {
            return Err("Unsupported numeric source dtype".into());
        }
        if consumer.is_some_and(|s| s != "SpectroDataset") {
            return Err("Unsupported numeric consumer".into());
        }
        if feature && consumer == Some("SpectroDataset") && request == Some(NumericStorage::Float64)
        {
            return Err(
                "SpectroDataset does not support an explicit float64 numerical guarantee".into(),
            );
        }
        let (dtype, origin) = if let Some(dtype) = request {
            (dtype, PrecisionOrigin::Requested)
        } else if feature && consumer == Some("SpectroDataset") {
            (NumericStorage::Float32, PrecisionOrigin::Consumer)
        } else if source == Some("float32") {
            (NumericStorage::Float32, PrecisionOrigin::SourceDeclared)
        } else if source == Some("float64") {
            (NumericStorage::Float64, PrecisionOrigin::SourceDeclared)
        } else {
            (NumericStorage::Float64, PrecisionOrigin::Default)
        };
        Ok(Self {
            dtype,
            origin,
            source_dtype: source.map(str::to_owned),
            requested_dtype: request,
            consumer: consumer.map(str::to_owned),
        })
    }
    pub fn validate(&self, feature: bool) -> Result<(), String> {
        let expected = Self::select(
            self.source_dtype.as_deref(),
            self.requested_dtype,
            self.consumer.as_deref(),
            feature,
        )?;
        if *self != expected {
            return Err("Precision record disagrees with owner selection".into());
        }
        Ok(())
    }
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = vec![
            self.dtype.tag(),
            match self.origin {
                PrecisionOrigin::Default => 0,
                PrecisionOrigin::SourceDeclared => 1,
                PrecisionOrigin::Requested => 2,
                PrecisionOrigin::Consumer => 3,
            },
        ];
        option_string(&mut out, self.source_dtype.as_deref());
        out.push(u8::from(self.requested_dtype.is_some()));
        if let Some(dtype) = self.requested_dtype {
            out.push(dtype.tag());
        }
        option_string(&mut out, self.consumer.as_deref());
        out
    }
}

pub(crate) fn u64_bytes(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_le_bytes());
}
pub(crate) fn bytes(out: &mut Vec<u8>, value: &[u8]) {
    u64_bytes(out, value.len());
    out.extend_from_slice(value);
}
pub(crate) fn string(out: &mut Vec<u8>, value: &str) {
    bytes(out, value.as_bytes());
}
pub(crate) fn option_string(out: &mut Vec<u8>, value: Option<&str>) {
    out.push(u8::from(value.is_some()));
    if let Some(value) = value {
        string(out, value);
    }
}
pub(crate) fn strings(out: &mut Vec<u8>, values: &[String]) {
    u64_bytes(out, values.len());
    for value in values {
        string(out, value);
    }
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode canonical wire hex only after the caller has checked its byte count.
pub(crate) fn decode_hex(value: &str, byte_count: usize) -> Result<Vec<u8>, String> {
    if byte_count.checked_mul(2) != Some(value.len()) {
        return Err("Numeric wire byte count mismatch".into());
    }
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    if !value.bytes().all(|byte| digit(byte).is_some()) {
        return Err("Numeric wire requires canonical lowercase hex".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_count)
        .map_err(|_| "Numeric wire allocation exceeds capacity")?;
    for pair in value.as_bytes().as_chunks::<2>().0 {
        bytes.push((digit(pair[0]).unwrap() << 4) | digit(pair[1]).unwrap());
    }
    Ok(bytes)
}

/// Decode a finite binary32 integer in the integer domain (no f64 comparison).
pub fn f32_integer(value: f32) -> Option<i128> {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 255) as i32;
    if exponent == 255 {
        return None;
    }
    if bits & 0x7fff_ffff == 0 {
        return Some(0);
    }
    let significand = if exponent == 0 {
        bits & 0x7f_ffff
    } else {
        (bits & 0x7f_ffff) | (1 << 23)
    } as i128;
    let shift = if exponent == 0 {
        -149
    } else {
        exponent - 127 - 23
    };
    let magnitude = if shift >= 0 {
        significand.checked_mul(1_i128.checked_shl(shift as u32)?)?
    } else {
        let right = (-shift) as u32;
        if right >= 128 || significand & ((1_i128 << right) - 1) != 0 {
            return None;
        }
        significand >> right
    };
    Some(if bits >> 31 == 0 {
        magnitude
    } else {
        -magnitude
    })
}
pub fn validate_label(label: i64, dtype: NumericStorage) -> Result<(), String> {
    if label.unsigned_abs() > 9_007_199_254_740_991 {
        return Err("Classification label exceeds portable integer cap".into());
    }
    if dtype == NumericStorage::Float32 && f32_integer(label as f32) != Some(i128::from(label)) {
        return Err("Classification label is not exact in float32 storage".into());
    }
    Ok(())
}

pub(crate) fn numeric_full(
    data: &[f64],
    rows: usize,
    cols: usize,
    validity: &[bool],
    precision: &PrecisionRecord,
) -> Value {
    let mut values = Vec::new();
    for (value, valid) in data.iter().zip(validity) {
        let value = if *valid { *value } else { 0.0 };
        match precision.dtype {
            NumericStorage::Float32 => values.extend_from_slice(&(value as f32).to_le_bytes()),
            NumericStorage::Float64 => values.extend_from_slice(&value.to_le_bytes()),
        }
    }
    json!({"dtype":precision.dtype,"n_rows":rows,"n_cols":cols,"data_bytes_hex":hex(&values),"validity":validity,"precision":precision})
}

impl TryFrom<Value> for PrecisionRecord {
    type Error = String;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let fields = [
            "dtype",
            "origin",
            "source_dtype",
            "requested_dtype",
            "consumer",
        ];
        let object = value
            .as_object()
            .ok_or("Precision record object required")?;
        if object.len() != fields.len() || fields.iter().any(|name| !object.contains_key(*name)) {
            return Err("Precision record requires exactly five fields".into());
        }
        let parse = |name: &str| {
            object
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Missing precision field {name}"))
        };
        let record = Self {
            dtype: serde_json::from_value(parse("dtype")?).map_err(|e| e.to_string())?,
            origin: serde_json::from_value(parse("origin")?).map_err(|e| e.to_string())?,
            source_dtype: serde_json::from_value(parse("source_dtype")?)
                .map_err(|e| e.to_string())?,
            requested_dtype: serde_json::from_value(parse("requested_dtype")?)
                .map_err(|e| e.to_string())?,
            consumer: serde_json::from_value(parse("consumer")?).map_err(|e| e.to_string())?,
        };
        if record.validate(true).is_err() && record.validate(false).is_err() {
            return Err("Precision record disagrees with owner selection".into());
        }
        Ok(record)
    }
}

/// Check integer representability before converting any original integer bits.
pub fn integer_exact_f64(value: i128) -> bool {
    let magnitude = value.unsigned_abs();
    let bits = 128 - magnitude.leading_zeros();
    bits <= 53 || magnitude.trailing_zeros() >= bits - 53
}
pub fn exact_integer_f64(value: i128) -> Result<f64, String> {
    if !integer_exact_f64(value) {
        return Err("Integer numeric value is not exactly representable in float64".into());
    }
    Ok(value as f64)
}
pub fn exact_json_number_f64(value: &Value) -> Result<f64, String> {
    if let Some(integer) = value.as_i64() {
        return exact_integer_f64(i128::from(integer));
    }
    if let Some(integer) = value.as_u64() {
        return exact_integer_f64(i128::from(integer));
    }
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or("Finite numeric value required".into())
}
