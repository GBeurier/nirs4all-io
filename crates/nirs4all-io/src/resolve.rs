// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
//! Resolver: normalize a path/glob/file-list into a stable, ordered `InputSet`
//! (ports `resolve/resolver.py`).
//!
//! Every item carries a stable identity (absolute path), a content hash (so
//! plans/fingerprints are reproducible), an extension hint, and a sidecar group
//! (companion files like ENVI `.hdr`). Directory and glob expansions are
//! deterministically ordered (by lowercased absolute path). Foundational —
//! retrofitting identity later would break fingerprints.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use nirs4all_io_core::spec::SpecError;

use indexmap::IndexMap;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Known sidecar relationships: primary-extension → companion extensions.
fn sidecar_companions() -> &'static [(&'static str, &'static [&'static str])] {
    &[
        (".img", &[".hdr"]),
        (".sli", &[".hdr"]),
        (".dat", &[".hdr"]),
        (".bil", &[".hdr"]),
        (".bsq", &[".hdr"]),
        (".bip", &[".hdr"]),
    ]
}

const GLOB_CHARS: [char; 3] = ['*', '?', '['];

#[derive(Debug, Clone)]
pub struct InputItem {
    /// Display reference (basename or relative path).
    pub ref_: String,
    /// `"file"` | `"array"` | `"record_set"` | `"spectrodataset"`.
    pub kind: String,
    /// Stable id: absolute path | `array:<name>` | `object:<id>`.
    pub identity: String,
    pub content_hash: Option<String>,
    pub extension: Option<String>,
    pub sidecars: Vec<String>,
    pub hints: Map<String, Value>,
}

#[derive(Debug, Clone, Default)]
pub struct InputSet {
    pub items: Vec<InputItem>,
    pub origin: Map<String, Value>,
}

impl InputSet {
    /// Display refs in order (Python `InputSet.names`).
    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|i| i.ref_.clone()).collect()
    }
    /// File items only.
    pub fn files(&self) -> Vec<&InputItem> {
        self.items.iter().filter(|i| i.kind == "file").collect()
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compound-extension-aware suffix (lowercased): `.csv.gz`/`.csv.zip`/`.tar.*`
/// or the final `.suffix`.
fn compound_extension(name: &str) -> String {
    let low = name.to_lowercase();
    for compound in [".csv.gz", ".csv.zip", ".tar.gz", ".tar.bz2", ".tar.xz"] {
        if low.ends_with(compound) {
            return compound.to_string();
        }
    }
    match name.rfind('.') {
        Some(idx) => name[idx..].to_lowercase(),
        None => String::new(),
    }
}

fn is_glob(text: &str) -> bool {
    text.chars().any(|c| GLOB_CHARS.contains(&c))
}

fn canonical_string(path: &Path) -> String {
    let raw = std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();
    strip_windows_verbatim_prefix(&raw)
}

fn strip_windows_verbatim_prefix(text: &str) -> String {
    const VERBATIM: &str = "\\\\?\\";
    const VERBATIM_UNC: &str = "\\\\?\\UNC\\";

    if let Some(rest) = text.strip_prefix(VERBATIM_UNC) {
        return format!("\\\\{rest}");
    }
    if let Some(rest) = text.strip_prefix(VERBATIM) {
        let bytes = rest.as_bytes();
        if bytes.len() >= 3 && bytes[1] == b':' && matches!(bytes[2], b'\\' | b'/') {
            return rest.to_string();
        }
    }
    text.to_string()
}

/// Python `Path.stem`: filename minus the final suffix.
fn path_stem(name: &str) -> String {
    match name.rfind('.') {
        Some(idx) if idx > 0 => name[..idx].to_string(),
        _ => name.to_string(),
    }
}

/// Map each primary file (abspath) to its companion sidecar abspaths.
fn group_sidecars(paths: &[PathBuf]) -> BTreeMap<String, Vec<String>> {
    let mut by_stem: BTreeMap<(String, String), Vec<PathBuf>> = BTreeMap::new();
    for p in paths {
        let parent = p
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        by_stem.entry((parent, stem)).or_default().push(p.clone());
    }
    let mut sidecars: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for group in by_stem.values() {
        let exts: IndexMap<String, &PathBuf> = group
            .iter()
            .map(|p| {
                let ext = p
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
                    .unwrap_or_default();
                (ext, p)
            })
            .collect();
        for (ext, companions) in sidecar_companions() {
            let Some(primary) = exts.get(*ext) else {
                continue;
            };
            let attached: Vec<String> = companions
                .iter()
                .filter_map(|c| exts.get(*c).map(|p| canonical_string(p)))
                .collect();
            if !attached.is_empty() {
                sidecars.insert(canonical_string(primary), attached);
            }
        }
    }
    sidecars
}

fn file_item(
    path: &Path,
    sidecar_map: &BTreeMap<String, Vec<String>>,
    base: Option<&Path>,
) -> Result<InputItem, SpecError> {
    let abspath = canonical_string(path);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ref_ = match base {
        Some(base) => relpath(path, base)?,
        None => name.clone(),
    };
    let mut hints = Map::new();
    hints.insert("basename".into(), Value::from(name.clone()));
    hints.insert("stem".into(), Value::from(path_stem(&name)));
    Ok(InputItem {
        ref_,
        kind: "file".into(),
        identity: abspath.clone(),
        content_hash: sha256_file(path).ok(),
        extension: Some(compound_extension(&name)),
        sidecars: sidecar_map.get(&abspath).cloned().unwrap_or_default(),
        hints,
    })
}

pub(crate) fn logical_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|component| !matches!(component, Component::CurDir))
        .collect()
}

fn relpath(path: &Path, base: &Path) -> Result<String, SpecError> {
    let path = logical_path(path);
    let base = logical_path(base);
    path.strip_prefix(&base)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .map_err(|_| {
            SpecError::new(format!(
                "logical path {path:?} is outside scan base {base:?}"
            ))
        })
}

fn sort_by_lower_abspath(paths: &mut [PathBuf]) {
    paths.sort_by_key(|p| canonical_string(p).to_lowercase());
}

fn resolve_dir(path: &Path, recursive: bool) -> Result<InputSet, SpecError> {
    let pattern = if recursive { "**/*" } else { "*" };
    let glob_pat = format!(
        "{}/{}",
        glob::Pattern::escape(&path.to_string_lossy()),
        pattern
    );
    let mut files: Vec<PathBuf> = glob::glob(&glob_pat)
        .map_err(|error| SpecError::new(error.to_string()))?
        .map(|entry| entry.map_err(|error| SpecError::new(error.to_string())))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|path| path.is_file())
        .collect();
    files.sort();
    sort_by_lower_abspath(&mut files);
    let sidecar_map = group_sidecars(&files);
    let attached: HashSet<String> = sidecar_map.values().flatten().cloned().collect();
    let items: Vec<InputItem> = files
        .iter()
        .filter(|p| !attached.contains(&canonical_string(p)))
        .map(|p| file_item(p, &sidecar_map, Some(path)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut origin = Map::new();
    origin.insert("kind".into(), Value::from("directory"));
    origin.insert("ref".into(), Value::from(canonical_string(path)));
    origin.insert("recursive".into(), Value::from(recursive));
    Ok(InputSet { items, origin })
}

fn resolve_paths(paths: &[String], recursive: bool) -> Result<InputSet, SpecError> {
    let mut expanded: Vec<PathBuf> = Vec::new();
    for raw in paths {
        if Path::new(raw).is_file() {
            expanded.push(PathBuf::from(raw));
        } else if is_glob(raw) {
            if let Ok(globbed) = glob::glob(raw) {
                expanded.extend(globbed.filter_map(Result::ok));
            }
        } else {
            expanded.push(PathBuf::from(raw));
        }
    }
    let mut files: Vec<PathBuf> = expanded.into_iter().filter(|p| p.is_file()).collect();
    sort_by_lower_abspath(&mut files);
    let sidecar_map = group_sidecars(&files);
    let attached: HashSet<String> = sidecar_map.values().flatten().cloned().collect();
    let items: Vec<InputItem> = files
        .iter()
        .filter(|p| !attached.contains(&canonical_string(p)))
        .map(|p| file_item(p, &sidecar_map, None))
        .collect::<Result<Vec<_>, _>>()?;
    let missing: Vec<Value> = paths
        .iter()
        .filter(|raw| !is_glob(raw) && !Path::new(raw).exists())
        .map(|raw| Value::from(raw.clone()))
        .collect();
    let mut origin = Map::new();
    origin.insert("kind".into(), Value::from("files"));
    origin.insert(
        "ref".into(),
        Value::Array(paths.iter().map(|p| Value::from(p.clone())).collect()),
    );
    if !missing.is_empty() {
        origin.insert("missing".into(), Value::Array(missing));
    }
    let _ = recursive;
    Ok(InputSet { items, origin })
}

/// Resolve a single path (directory / glob / file) into an `InputSet`.
pub fn resolve_path(input: &str, recursive: bool) -> Result<InputSet, SpecError> {
    let path = Path::new(input);
    if path.is_dir() {
        resolve_dir(path, recursive)
    } else {
        resolve_paths(&[input.to_string()], recursive)
    }
}

/// Resolve a list of paths/globs into an `InputSet`.
pub fn resolve_list(paths: &[String], recursive: bool) -> Result<InputSet, SpecError> {
    resolve_paths(paths, recursive)
}

pub(crate) type FileReferences = HashMap<String, InputItem>;

pub(crate) fn used_file_references(
    iset: &InputSet,
    spec: &Value,
) -> Result<FileReferences, SpecError> {
    let mut used = Vec::new();
    for source in spec["sources"].as_array().into_iter().flatten() {
        match source.get("input") {
            Some(Value::String(value)) => used.push(value.as_str()),
            Some(Value::Array(values)) => used.extend(values.iter().filter_map(Value::as_str)),
            _ => {}
        }
    }
    if let Some(file) = spec["folds"]["file"].as_str() {
        used.push(file);
    }
    let mut table = HashMap::new();
    for name in used {
        let mut candidates: Vec<&InputItem> = iset
            .items
            .iter()
            .filter(|item| item.kind == "file" && item.ref_ == name)
            .collect();
        if candidates.is_empty() {
            candidates = iset
                .items
                .iter()
                .filter(|item| item.kind == "file" && item.identity == name)
                .collect();
        }
        let mut identities: Vec<&str> = candidates
            .iter()
            .map(|item| item.identity.as_str())
            .collect();
        identities.sort_unstable();
        identities.dedup();
        if identities.len() > 1 {
            return Err(SpecError::new(format!(
                "input reference {name:?} is ambiguous: {identities:?}"
            )));
        }
        let item = candidates.first().ok_or_else(|| {
            SpecError::new(format!(
                "generated input reference {name:?} is not a resolved file"
            ))
        })?;
        table.insert(name.to_string(), (**item).clone());
    }
    Ok(table)
}

pub(crate) fn absolutize_file_references(spec: &mut Value, table: &FileReferences) {
    if let Some(sources) = spec["sources"].as_array_mut() {
        for source in sources {
            if let Some(input) = source.get_mut("input") {
                match input {
                    Value::String(name) => *name = table[name.as_str()].identity.clone(),
                    Value::Array(items) => {
                        for item in items {
                            if let Some(name) = item.as_str() {
                                *item = Value::from(table[name].identity.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(folds) = spec.get_mut("folds") {
        if let Some(name) = folds.get("file").and_then(Value::as_str) {
            let item = &table[name];
            let physical = Path::new(&item.identity)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| format!(".{}", ext.to_lowercase()));
            let logical = item.extension.as_deref();
            let format = match logical {
                Some(".csv") => Some("csv"),
                Some(".json") => Some("json"),
                Some(".yaml" | ".yml") => Some("yaml"),
                Some(".txt") => Some("txt"),
                _ => None,
            };
            let auto = matches!(
                folds.get("format").and_then(Value::as_str),
                None | Some("") | Some("auto")
            );
            if let Some(format) = format.filter(|_| auto && physical.as_deref() != logical) {
                folds["format"] = Value::from(format);
            }
            folds["file"] = Value::from(item.identity.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &Path, name: &str, content: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn resolve_directory_sorted_basenames() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "Ycal.csv", "y\n1\n");
        write(tmp.path(), "Xcal.csv", "1000\n0.4\n");
        let iset = resolve_path(&tmp.path().to_string_lossy(), false).expect("resolve directory");
        assert_eq!(iset.names(), vec!["Xcal.csv", "Ycal.csv"]); // lower-abspath order
        assert_eq!(iset.origin["kind"], "directory");
        assert!(iset.items[0].content_hash.is_some());
        assert_eq!(iset.items[0].extension.as_deref(), Some(".csv"));
    }

    #[test]
    fn resolve_file_list_basenames_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "X.csv", "1000\n0.4\n");
        let present = tmp.path().join("X.csv").to_string_lossy().into_owned();
        let iset = resolve_list(&[present, "nope.csv".into()], false).expect("resolve list");
        assert_eq!(iset.names(), vec!["X.csv"]);
        assert_eq!(iset.origin["kind"], "files");
        assert_eq!(iset.origin["missing"], serde_json::json!(["nope.csv"]));
    }

    #[test]
    fn canonical_string_strips_windows_verbatim_prefixes() {
        assert_eq!(
            strip_windows_verbatim_prefix(r"\\?\C:\tmp\n4io_smoke\X.csv"),
            r"C:\tmp\n4io_smoke\X.csv"
        );
        assert_eq!(
            strip_windows_verbatim_prefix(r"\\?\UNC\server\share\X.csv"),
            r"\\server\share\X.csv"
        );
        assert_eq!(
            strip_windows_verbatim_prefix(r"\\?\not-a-drive-root"),
            r"\\?\not-a-drive-root"
        );
    }
}

#[cfg(test)]
mod path_reference_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn directory_alias_ties_are_logical_while_lists_keep_their_order() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let store = tmp.path().join("store");
        std::fs::write(&store, "x\n0.1\n").unwrap();
        let aliases = tmp.path().join("aliases");
        std::fs::create_dir(&aliases).unwrap();
        for name in ["z.csv", "a.csv", "m.csv"] {
            symlink(&store, aliases.join(name)).unwrap();
        }
        assert_eq!(
            resolve_path(aliases.to_str().unwrap(), false)
                .unwrap()
                .names(),
            vec!["a.csv", "m.csv", "z.csv"]
        );
        let paths = [
            aliases.join("z.csv").to_string_lossy().into_owned(),
            aliases.join("a.csv").to_string_lossy().into_owned(),
        ];
        assert_eq!(
            resolve_list(&paths, false).unwrap().names(),
            vec!["z.csv", "a.csv"]
        );
    }

    #[test]
    fn lexical_relative_comparison_handles_curdir_and_reports_mismatch() {
        assert_eq!(
            relpath(Path::new("data/a.csv"), Path::new("./data")).unwrap(),
            "a.csv"
        );
        assert!(relpath(Path::new("other/a.csv"), Path::new("data")).is_err());
    }
}
