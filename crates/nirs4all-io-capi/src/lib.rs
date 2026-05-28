// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Stable C ABI for `nirs4all-io` (symbol prefix `n4io_`).
//!
//! v0 surface is JSON-string in / JSON-string out (`infer` / `to_spec` /
//! `validate`); no materialized arrays cross the ABI in v0 (D-R7). The full
//! surface, status/error model, and opaque handles land in EPIC 9. This file
//! currently exposes only the ABI-version probe and the string-free contract
//! so the header generates and the symbol governance can be wired early.

use std::ffi::{c_char, CString};

/// ABI version string. Independent of crate semver (D-R6); bump on ABI change.
pub const N4IO_ABI_VERSION: &str = "0.0.1";

/// Return the ABI version as a freshly allocated C string.
///
/// The caller owns the returned pointer and MUST release it with
/// [`n4io_string_free`]; never free it with the host allocator.
///
/// # Safety
/// The returned pointer is non-null and points to a NUL-terminated UTF-8
/// string allocated by this library.
#[no_mangle]
pub extern "C" fn n4io_abi_version() -> *mut c_char {
    CString::new(N4IO_ABI_VERSION)
        .expect("static version contains no nul")
        .into_raw()
}

/// Free a string previously returned by this library.
///
/// # Safety
/// `ptr` must be a pointer returned by an `n4io_*` function that documents
/// ownership transfer, or null. Passing any other pointer is undefined
/// behaviour. Each such pointer must be freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn n4io_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}
