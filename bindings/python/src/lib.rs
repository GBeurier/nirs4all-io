// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Native core of the pyo3 Python binding (EPIC 11.1). Compiled as
//! `nirs4all_io._native`; the user-facing `nirs4all_io` package
//! (`bindings/python/python/nirs4all_io`) wraps it and adds the lazy
//! SpectroDataset adapter.
//!
//! The JSON surface (`infer` / `to_spec` / `validate`) and the materialized
//! exports (`load_summary` = rounded structural summary, `assembled_full` =
//! full-precision arrays for the SpectroDataset adapter) return native Python
//! objects via `pythonize`. Nothing here imports `nirs4all`.

use nirs4all_io_facade::api::{
    load_assembled_with_limits, load_package_with_limits_and_storage, to_spec as facade_to_spec,
    Input, LoadLimits,
};
use nirs4all_io_facade::core::materialize::{NumericStorage, NumericStorageOptions};
use nirs4all_io_facade::core::spec::{validate_spec, DatasetSpec};
use nirs4all_io_facade::infer::{infer_path, infer_paths};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};
use pythonize::{depythonize, pythonize};
use serde_json::Value;

mod memory;

/// Convert a Python argument into a facade [`Input`]: `str` → path, a sequence of
/// `str` → file list, a `dict`/mapping → spec.
fn to_input(obj: &Bound<'_, PyAny>) -> PyResult<Input> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Input::Path(s));
    }
    if let Ok(paths) = obj.extract::<Vec<String>>() {
        return Ok(Input::Paths(paths));
    }
    let value: Value = depythonize(obj).map_err(|e| {
        PyTypeError::new_err(format!("input is not a path, file list, or spec dict: {e}"))
    })?;
    if value.is_object() {
        Ok(Input::Spec(value))
    } else {
        Err(PyTypeError::new_err(
            "input must be a path (str), file list (sequence of str), or spec (dict)",
        ))
    }
}

fn to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(pythonize(py, value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .unbind())
}

/// Detect task type through the existing IO owner, using its f64 contract.
#[pyfunction]
#[pyo3(signature = (y, threshold=0.05))]
fn detect_task_type(y: Vec<f64>, threshold: f64) -> (String, f64) {
    let (task_type, score) = nirs4all_io_facade::core::infer::detect_task_type(&y, threshold);
    (task_type.to_owned(), score)
}

/// Validate a declared float cell through the numeric owner before host casting.
#[pyfunction]
fn declared_float_cell(
    py: Python<'_>,
    cell: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let storage = match dtype {
        "float32" => NumericStorage::Float32,
        "float64" => NumericStorage::Float64,
        _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
    };
    let value = if cell.is_exact_instance_of::<PyInt>() {
        if let Ok(integer) = cell.extract::<i64>() {
            Value::from(integer)
        } else if let Ok(integer) = cell.extract::<u64>() {
            Value::from(integer)
        } else {
            // Never retry an out-of-domain Python integer through f64.
            return Err(PyValueError::new_err(
                "Integer numeric value is outside the supported i64/u64 domain",
            ));
        }
    } else if cell.is_exact_instance_of::<PyFloat>() {
        let value = cell.extract::<f64>()?;
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("Observed numeric values must be finite"))?
    } else {
        return Err(PyTypeError::new_err(
            "declared float cells must be exact builtin int or finite float values",
        ));
    };
    let normalized = storage
        .json_number(&value, Some(dtype))
        .map_err(PyValueError::new_err)?;
    if storage == NumericStorage::Float64 {
        // Validation must not turn an admitted original Python integer into a float.
        Ok(cell.clone().unbind())
    } else {
        Ok(PyFloat::new(py, normalized).into_any().unbind())
    }
}

/// Inspect a data input and return the scored DatasetPlan as a dict.
#[pyfunction]
#[pyo3(signature = (input, conventions=None, params=None, file_params=None))]
fn infer(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    conventions: Option<Vec<String>>,
    params: Option<&Bound<'_, PyAny>>,
    file_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let options = nirs4all_io_facade::core::infer::InferOptions {
        conventions: conventions.unwrap_or_default(),
        params: match params {
            Some(value) => depythonize(value)
                .map_err(|e| PyValueError::new_err(format!("invalid params: {e}")))?,
            None => serde_json::json!({}),
        },
        file_params: match file_params {
            Some(value) => depythonize(value)
                .map_err(|e| PyValueError::new_err(format!("invalid file_params: {e}")))?,
            None => Default::default(),
        },
    };
    let plan = match to_input(input)? {
        Input::Path(p) => infer_path(&p, &options),
        Input::Paths(ps) => infer_paths(&ps, &options),
        Input::Spec(_) => {
            return Err(PyValueError::new_err(
                "infer needs a data input (path or file list), not a spec",
            ))
        }
    }
    .map_err(|e| PyValueError::new_err(e.message))?;
    to_py(py, &plan.to_value())
}

/// Normalize an input into a canonical DatasetSpec dict.
#[pyfunction]
#[pyo3(signature = (input, conventions=None, name=None))]
fn to_spec(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    conventions: Option<Vec<String>>,
    name: Option<String>,
) -> PyResult<Py<PyAny>> {
    let (spec, _base) = facade_to_spec(&to_input(input)?, conventions.as_deref(), name.as_deref())
        .map_err(|e| PyValueError::new_err(e.message))?;
    to_py(py, &spec.to_value())
}

/// Validate a DatasetSpec (a dict or a JSON string); raises `ValueError` if invalid.
#[pyfunction]
fn validate(spec: &Bound<'_, PyAny>) -> PyResult<()> {
    let value: Value = if let Ok(s) = spec.extract::<String>() {
        nirs4all_io_facade::core::canonical_json::parse_json_value(&s)
            .map_err(|e| PyValueError::new_err(format!("invalid spec JSON: {e}")))?
    } else {
        depythonize(spec)
            .map_err(|e| PyTypeError::new_err(format!("spec must be a dict or JSON string: {e}")))?
    };
    let spec = DatasetSpec::from_value(&value).map_err(|e| PyValueError::new_err(e.message))?;
    validate_spec(&spec).map_err(|e| PyValueError::new_err(e.message))?;
    Ok(())
}

/// Materialize an input and return the rounded structural summary dict
/// (target="assembled").
#[pyfunction]
#[pyo3(signature = (input, conventions=None, name=None, *, limits=None))]
fn load_summary(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    conventions: Option<Vec<String>>,
    name: Option<String>,
    limits: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let assembled = load_assembled_with_limits(
        &to_input(input)?,
        conventions.as_deref(),
        name.as_deref(),
        parse_load_limits(limits)?,
    )
    .map_err(|e| PyValueError::new_err(e.message))?;
    to_py(py, &assembled.to_summary_value())
}

/// Materialize an input and return the full-precision arrays dict consumed by the
/// Python SpectroDataset adapter (X/y/processing matrices, metadata, weights, folds).
#[pyfunction]
#[pyo3(signature = (input, conventions=None, name=None, *, limits=None, storage=None))]
fn assembled_full(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    conventions: Option<Vec<String>>,
    name: Option<String>,
    limits: Option<&Bound<'_, PyAny>>,
    storage: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let options = parse_numeric_storage(storage)?;
    let package = load_package_with_limits_and_storage(
        &to_input(input)?,
        conventions.as_deref(),
        name.as_deref(),
        parse_load_limits(limits)?,
        &options,
    )
    .map_err(|e| PyValueError::new_err(e.message))?;
    let full: Value = package
        .to_assembled()
        .to_full_value()
        .map_err(PyValueError::new_err)?;
    to_py(py, &full)
}

fn parse_numeric_storage(storage: Option<&Bound<'_, PyAny>>) -> PyResult<NumericStorageOptions> {
    Ok(storage
        .map(|storage| {
            depythonize(storage)
                .map_err(|e| PyValueError::new_err(format!("invalid numeric storage: {e}")))
        })
        .transpose()?
        .unwrap_or_default())
}

fn parse_load_limits(limits: Option<&Bound<'_, PyAny>>) -> PyResult<LoadLimits> {
    limits
        .map(|limits| {
            let value: Value = depythonize(limits)
                .map_err(|e| PyTypeError::new_err(format!("invalid load limits: {e}")))?;
            LoadLimits::from_value(&value).map_err(|e| PyValueError::new_err(e.message))
        })
        .unwrap_or_else(|| Ok(LoadLimits::default()))
}

/// Assemble Methods-produced feature projections through native IO contracts.
#[pyfunction]
fn projected_matrix_dataset(
    py: Python<'_>,
    record: &Bound<'_, PyAny>,
    projections: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let record: Value =
        depythonize(record).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let projections: Vec<Value> =
        depythonize(projections).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (record, provenance) =
        nirs4all_io_facade::core::public_dataset::projected_matrix_dataset(&record, &projections)
            .map_err(PyValueError::new_err)?;
    to_py(
        py,
        &serde_json::json!({"record":record,"provenance":provenance}),
    )
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(detect_task_type, m)?)?;
    m.add_function(wrap_pyfunction!(declared_float_cell, m)?)?;
    m.add_function(wrap_pyfunction!(infer, m)?)?;
    m.add_function(wrap_pyfunction!(projected_matrix_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(to_spec, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(load_summary, m)?)?;
    m.add_function(wrap_pyfunction!(assembled_full, m)?)?;
    m.add_function(wrap_pyfunction!(memory::resolved_load_limits, m)?)?;
    m.add_function(wrap_pyfunction!(memory::assemble_frames, m)?)?;
    m.add_function(wrap_pyfunction!(memory::select_spectrodataset_features, m)?)?;
    Ok(())
}
