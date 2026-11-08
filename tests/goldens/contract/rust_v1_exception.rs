// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Exact allowed deltas for V1-RUST-MIXED-SPECTRAL-HEADER, never a normalization.

use serde_json::{json, Value};

pub fn assert_exception(frozen: &Value, corrected: &Value, artifact: &str) {
    let mut expected = frozen.clone();
    match artifact {
        "infer" => {
            expected["params"]["data.csv"]["has_header"] = json!({"score": 0.85, "value": true});
            let roles = expected["columns"][0]["column_roles"]
                .as_array_mut()
                .unwrap();
            assert_eq!(roles.len(), 13);
            for role in &mut roles[..12] {
                assert_eq!(role["role"], "features");
                role["score"] = json!(0.9);
                role["evidence"] = json!(["monotonic nm/cm-1 wavelength column"]);
            }
            assert_eq!(roles[12]["role"], "targets");
            roles[12]["score"] = json!(0.65);
            roles[12]["evidence"] =
                json!(["numeric non-wavelength column alongside a spectral block"]);
            expected["resolved_spec"]["sources"][0]["params"]["header_unit"] = json!("nm");
        }
        "assembled" => expected["blocks"]["train"]["header_units"] = json!(["nm"]),
        _ => panic!("unknown V1-RUST-MIXED-SPECTRAL-HEADER artifact: {artifact}"),
    }
    assert_eq!(
        corrected, &expected,
        "V1-RUST-MIXED-SPECTRAL-HEADER exceeded its reviewed delta"
    );
}
