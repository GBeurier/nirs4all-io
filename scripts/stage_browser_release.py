#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Stage the wasm-pack ``web`` build as GitHub Release assets with its receipt.

nirs4all-web stages the browser payload from these exact Release asset names
and pins their sizes and SHA-256 digests; the receipt
(``nirs4all.owner-browser-release.v1``) records the source identity, the build
recipe and every asset. Generated files are flattened into
``nirs4all-io-browser-web-<path with "/" replaced by "__">``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "GBeurier/nirs4all-io"
ASSET_PREFIX = "nirs4all-io-browser-web"
RECEIPT_NAME = f"{ASSET_PREFIX}.provenance.json"
SCHEMA = "nirs4all.owner-browser-release.v1"
STEM = "nirs4all_io_wasm"
OPERATION = "Identity-aligned multimodal materialization; formats supplied by the browser host"
BUILD_COMMAND = "wasm-pack build bindings/wasm --target web --release -- --locked"
LOCK = "bindings/wasm/Cargo.lock"
# wasm-pack copies these crate files next to the generated module; they are not
# part of the browser payload (legal files ship through the npm package).
IGNORED = {".gitignore", "README.md"}
PRIMARY = (f"{STEM}.js", f"{STEM}.d.ts", f"{STEM}_bg.wasm", f"{STEM}_bg.wasm.d.ts", "package.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_files(pkg: Path) -> list[str]:
    """Return the generated payload in receipt order, refusing unknown files."""
    found: list[str] = []
    for path in sorted(pkg.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(pkg).as_posix()
        if relative in IGNORED or (path.parent == pkg and path.name.startswith("LICENSE")):
            continue
        if relative not in PRIMARY and not relative.startswith("snippets/"):
            raise SystemExit(f"unexpected wasm-pack output: {relative}")
        found.append(relative)
    missing = [name for name in PRIMARY if name not in found]
    if missing:
        raise SystemExit(f"wasm-pack output is missing {', '.join(missing)}")
    return [*PRIMARY, *sorted(name for name in found if name not in PRIMARY)]


def asset_name(relative: str) -> str:
    return f"{ASSET_PREFIX}-{relative.replace('/', '__')}"


def build_receipt(
    *,
    pkg: Path,
    version: str,
    tag: str,
    commit: str,
    tree: str,
    tool: str,
    rustc: str,
    lock_sha256: str,
    host: str,
    run_url: str,
) -> dict:
    package_version = json.loads((pkg / "package.json").read_text(encoding="utf-8"))["version"]
    if package_version != version:
        raise SystemExit(f"package.json version {package_version} != workspace version {version}")
    base_url = f"https://github.com/{OWNER}/releases/download/{tag}"
    files = {}
    for relative in payload_files(pkg):
        path = pkg / relative
        files[relative] = {
            "asset": asset_name(relative),
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "url": f"{base_url}/{asset_name(relative)}",
        }
    return {
        "schema": SCHEMA,
        "owner": OWNER,
        "version": version,
        "source_tag": tag,
        "source_commit": commit,
        "source_tree": tree,
        "operation": OPERATION,
        "build": {
            "tool": tool,
            "target": "web",
            "profile": "release",
            "cargo_locked": True,
            "lock_sha256": lock_sha256,
            "command": BUILD_COMMAND,
            "host": host,
        },
        "qualification": (
            "Built by the owner release-browser workflow after release validation (source CI, parity "
            f"oracle and cross-binding gates) passed on the tagged commit: {run_url}. Browser "
            "qualification of this payload is the nirs4all-web gate."
        ),
        "provenance": (
            f"Built in CI from the exact clean tagged source ({rustc}) with canonical Rust source-path "
            "remapping. Browser payload differs from the npm Node payload. No independent "
            "reproducibility claim."
        ),
        "files": files,
    }


def stage(pkg: Path, out: Path, receipt: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for relative, fact in receipt["files"].items():
        shutil.copyfile(pkg / relative, out / fact["asset"])
    (out / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def command_version(*command: str) -> str:
    return subprocess.check_output(list(command), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkg", type=Path, help="wasm-pack --target web output directory")
    parser.add_argument("out", type=Path, help="directory receiving the Release assets")
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wasm-pack", required=True, help="pinned wasm-pack version, e.g. 0.15.0")
    parser.add_argument("--run-url", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspection build: skip the tag-to-HEAD check (the assets are never attached)",
    )
    args = parser.parse_args()

    tool = command_version("wasm-pack", "--version")
    if tool != f"wasm-pack {args.wasm_pack}":
        raise SystemExit(f"wasm-pack version differs: {tool!r}")
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("the release checkout has tracked modifications")
    commit = git("rev-parse", "HEAD")
    if not args.dry_run and git("rev-parse", f"{args.tag}^{{commit}}") != commit:
        raise SystemExit(f"{args.tag} does not resolve to the checked-out commit {commit}")
    receipt = build_receipt(
        pkg=args.pkg,
        version=args.version,
        tag=args.tag,
        commit=commit,
        tree=git("rev-parse", "HEAD^{tree}"),
        tool=tool,
        rustc=command_version("rustc", "--version").split(" (")[0],
        lock_sha256=sha256(ROOT / LOCK),
        host=f"{platform.system()} {platform.machine()}",
        run_url=args.run_url,
    )
    stage(args.pkg, args.out, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
