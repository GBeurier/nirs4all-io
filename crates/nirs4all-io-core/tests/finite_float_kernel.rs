//! Literal vectors are supplied by the independent oracle owner, never generated here.
use nirs4all_io_core::{
    canonical_json::{canonical_json, parse_json_value},
    materialize::Cell,
    pyfmt::py_float_repr,
};
use serde_json::{json, Value};

#[test]
fn shared_literal_vectors_exercise_both_writers_and_categorical_scalar_text() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/finite_float_vectors.json")).unwrap();
    assert_eq!(fixture["schema"], "nirs4all.io.finite-float-vectors.v1");
    for case in fixture["cases"].as_array().unwrap() {
        let bits = u64::from_str_radix(case["bits_hex"].as_str().unwrap(), 16).unwrap();
        let value = f64::from_bits(bits);
        let expected = case["repr"].as_str().unwrap();
        assert!(value.is_finite());
        assert_eq!(py_float_repr(value), expected, "{}", case["id"]);
        assert_eq!(
            Cell::Float(value).to_str_scalar(),
            expected,
            "{}",
            case["id"]
        );
        assert_eq!(
            canonical_json(&json!(value)).unwrap(),
            format!("{expected}\n"),
            "{}",
            case["id"]
        );
        assert_eq!(
            parse_json_value(expected)
                .unwrap()
                .as_f64()
                .unwrap()
                .to_bits(),
            bits
        );
    }
}

#[test]
fn nested_layout_strings_unicode_and_integer_tokens_keep_their_existing_bytes() {
    let value = json!({"é":[{"z":"1e-07", "a":1e-7},[],{}], "i":1, "f":1.0, "quote":"\"\\\n", "negative":-0.0});
    assert_eq!(canonical_json(&value).unwrap(), "{\n  \"f\": 1.0,\n  \"i\": 1,\n  \"negative\": -0.0,\n  \"quote\": \"\\\"\\\\\\n\",\n  \"é\": [\n    {\n      \"a\": 1e-07,\n      \"z\": \"1e-07\"\n    },\n    [],\n    {}\n  ]\n}\n");
    assert_eq!(
        canonical_json(&json!([i64::MIN, u64::MAX])).unwrap(),
        "[\n  -9223372036854775808,\n  18446744073709551615\n]\n"
    );
}

#[test]
fn validated_text_integer_guard_preserves_bits_and_refuses_out_of_range_tokens() {
    for token in [
        "-9223372036854775808",
        "18446744073709551615",
        "9223372036854775808",
        "0",
    ] {
        assert!(parse_json_value(token).unwrap().is_number());
    }
    for token in [
        "-9223372036854775809",
        "18446744073709551616",
        "18446744073709551617",
        "123456789012345678901234567890",
        "-123456789012345678901234567890",
        "1e400",
        "1e-400",
        "2e-324",
        "-2e-324",
        "NaN",
        "Infinity",
        "-Infinity",
    ] {
        assert!(parse_json_value(token).is_err(), "{token}");
    }
    assert_eq!(
        parse_json_value("-0").unwrap().as_f64().unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(parse_json_value("1E5").unwrap().as_f64(), Some(100000.0));
    assert!(parse_json_value("1E5").unwrap().is_f64());
    let escaped = r#"["18446744073709551617", "quote\"123456789012345678901234567890", "slash\\\"1", {"18446744073709551617": "é\\42"}]"#;
    assert_eq!(
        parse_json_value(escaped).unwrap(),
        serde_json::from_str::<Value>(escaped).unwrap()
    );
    assert!(parse_json_value(r#"{"x": [1,18446744073709551617]}"#).is_err());
    for token in ["0e5", "0.000e-999", "-0.000e-999"] {
        assert_eq!(parse_json_value(token).unwrap().as_f64().unwrap(), 0.0);
    }
    assert!(parse_json_value(r#""1e-400""#).unwrap().is_string());
    assert!(parse_json_value("01").is_err());
}
