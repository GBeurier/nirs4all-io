// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Explicit loading choices admitted before inference reads any table.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::materialize::effective_params;
use crate::spec::dataset_spec::LoadingParams;
use crate::spec::SpecError;

fn empty_params() -> Value {
    json!({})
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferOptions {
    #[serde(default)]
    pub conventions: Vec<String>,
    #[serde(default = "empty_params")]
    pub params: Value,
    #[serde(default)]
    pub file_params: BTreeMap<String, Value>,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self {
            conventions: vec![],
            params: empty_params(),
            file_params: BTreeMap::new(),
        }
    }
}

impl InferOptions {
    /// Validate declarations and exact input names before the first table read.
    pub fn validate(&self, names: &[String]) -> Result<(), SpecError> {
        parse_params(&self.params)?;
        for (name, params) in &self.file_params {
            if !names.contains(name) {
                return Err(SpecError::new(format!(
                    "file_params names unknown input '{name}'"
                )));
            }
            parse_params(params)?;
        }
        Ok(())
    }

    pub fn effective_file_params(&self, name: &str) -> Result<Map<String, Value>, SpecError> {
        let global = parse_params(&self.params)?;
        let local = self
            .file_params
            .get(name)
            .map(parse_params)
            .transpose()?
            .unwrap_or_default();
        Ok(effective_params(&global, &local)
            .to_value()
            .as_object()
            .unwrap()
            .clone())
    }

    /// Install declarations before enrichment, refusing conflicting merged files.
    pub fn apply_to_spec(&self, spec: &mut Value) -> Result<(), SpecError> {
        spec["params"] = self.params.clone();
        let Some(sources) = spec["sources"].as_array_mut() else {
            return Ok(());
        };
        for source in sources {
            let global = LoadingParams::from_value(Some(&self.params))?;
            let local = LoadingParams::from_value(source.get("params"))?;
            source["params"] = effective_params(&global, &local).to_value();
            let inputs: Vec<&str> = match &source["input"] {
                Value::String(name) => vec![name],
                Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
                _ => vec![],
            };
            if !inputs
                .iter()
                .any(|name| self.file_params.contains_key(*name))
            {
                continue;
            }
            let mut common = None;
            for name in inputs {
                let params = self.effective_file_params(name)?;
                if common.as_ref().is_some_and(|old| old != &params) {
                    return Err(SpecError::new(format!(
                        "source '{}': conflicting file_params for multi-input source",
                        source["id"]
                    )));
                }
                common = Some(params);
            }
            if let Some(params) = common {
                let prior = LoadingParams::from_value(source.get("params"))?;
                let declared = LoadingParams::from_value(Some(&Value::Object(params)))?;
                source["params"] = effective_params(&prior, &declared).to_value();
            }
        }
        Ok(())
    }
}

fn parse_params(value: &Value) -> Result<LoadingParams, SpecError> {
    let object = value
        .as_object()
        .ok_or_else(|| SpecError::new("inference params must be an object"))?;
    for key in object.keys() {
        if ![
            "has_header",
            "header",
            "delimiter",
            "decimal_separator",
            "decimal",
            "encoding",
            "header_unit",
            "signal_type",
            "categorical",
            "na",
            "format",
        ]
        .contains(&key.as_str())
        {
            return Err(SpecError::new(format!(
                "unknown inference loading parameter '{key}'"
            )));
        }
    }
    for key in ["has_header", "header"] {
        if object
            .get(key)
            .is_some_and(|v| !v.is_null() && !v.is_boolean())
        {
            return Err(SpecError::new(format!(
                "inference params.{key} must be a boolean"
            )));
        }
    }
    for key in ["delimiter", "decimal_separator", "decimal", "encoding"] {
        if object
            .get(key)
            .is_some_and(|v| !v.is_null() && !v.is_string())
        {
            return Err(SpecError::new(format!(
                "inference params.{key} must be a string"
            )));
        }
    }
    for key in ["na", "format"] {
        if object
            .get(key)
            .is_some_and(|v| !v.is_null() && !v.is_object())
        {
            return Err(SpecError::new(format!(
                "inference params.{key} must be an object"
            )));
        }
    }
    LoadingParams::from_value(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_or_malformed_choices_refuse_before_reading() {
        for params in [
            json!(false),
            json!({"has_header": "false"}),
            json!({"delimiter": 3}),
        ] {
            let options = InferOptions {
                params,
                ..Default::default()
            };
            assert!(options.validate(&["data.csv".into()]).is_err());
        }
        let options = InferOptions {
            file_params: [("missing.csv".into(), json!({"has_header": false}))]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        assert!(options
            .validate(&["data.csv".into()])
            .unwrap_err()
            .message
            .contains("unknown input"));
    }

    #[test]
    fn merged_files_must_agree_on_effective_loading_parameters() {
        let options = InferOptions {
            params: json!({"has_header": true}),
            file_params: [("a.csv".into(), json!({"has_header": false}))]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let mut spec =
            json!({"sources": [{"id": "x", "role": "features", "input": ["a.csv", "b.csv"]}]});
        assert!(options
            .apply_to_spec(&mut spec)
            .unwrap_err()
            .message
            .contains("conflicting file_params"));
        let mut matching = options;
        matching
            .file_params
            .insert("b.csv".into(), json!({"header": false}));
        matching.apply_to_spec(&mut spec).unwrap();
        assert_eq!(spec["sources"][0]["params"]["has_header"], false);
    }
}
