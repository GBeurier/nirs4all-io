// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
fn main() {
    // The Windows executable default (1 MiB) cannot hold the bounded inference
    // call stack in debug builds. Match Linux's usual 8 MiB main-stack reserve;
    // Windows commits pages on demand. This changes neither reader budgets nor
    // the stack settings of library consumers or their threads.
    let target = std::env::var("TARGET").expect("Cargo supplies TARGET");
    if target.contains("windows-msvc") {
        println!("cargo:rustc-link-arg-bin=nirs4all-io=/STACK:8388608");
    } else if target.contains("windows-gnu") {
        println!("cargo:rustc-link-arg-bin=nirs4all-io=-Wl,--stack,8388608");
    }
}
