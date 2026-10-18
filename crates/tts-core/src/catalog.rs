//! Installable voice packages and a native downloader for them.
//!
//! The catalog is read from the same pinned manifests the Python scripts use
//! (`assets/*.json`): every file has a fixed URL at a fixed upstream commit and
//! a SHA-256. [`install`] downloads into a hidden staging folder beside the
//! target, verifies every file, and renames the folder into place only when all
//! of it checks out, so a failed, cancelled or tampered download never leaves
//! a package that looks installed. No publisher code is executed.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

/// Marker written last into an installed package folder.
const MARKER: &str = "package.json";

/// One file of a package.
#[derive(Debug, Clone)]
pub struct PackageFile {
    /// Path inside the package folder.
    pub name: String,
    /// Where it is downloaded from; `None` when it is extracted from another file.
    pub url: Option<String>,
    pub sha256: String,
    pub bytes: Option<u64>,
    /// The archive (another file's `name`) this file is extracted from.
    pub derived_from: Option<String>,
    /// Needed to build the package but not kept in it.
    pub acquisition_only: bool,
}

/// An installable voice package.
#[derive(Debug, Clone)]
pub struct Package {
    /// Folder name and id, e.g. `kokoro-en-q8`.
    pub id: &'static str,
    pub title: &'static str,
    /// The `tts-speak --backend` that loads it.
    pub backend: &'static str,
    pub languages: &'static [&'static str],
    pub license: &'static str,
    /// Where the weights come from, for notices.
    pub source: String,
    pub files: Vec<PackageFile>,
}

impl Package {
    /// Bytes downloaded to install it (archives included).
    pub fn download_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter(|f| f.url.is_some())
            .filter_map(|f| f.bytes)
            .sum()
    }
}

fn files_from(records: &serde_json::Map<String, Value>) -> Result<Vec<PackageFile>, String> {
    records
        .iter()
        .map(|(name, record)| {
            Ok(PackageFile {
                name: name.clone(),
                url: record["url"].as_str().map(String::from),
                sha256: record["sha256"]
                    .as_str()
                    .ok_or_else(|| format!("{name}: missing sha256"))?
                    .to_string(),
                bytes: record["bytes"].as_u64(),
                derived_from: record["derived_from"].as_str().map(String::from),
                acquisition_only: record["acquisition_only"].as_bool() == Some(true),
            })
        })
        .collect()
}

/// Every package this build can install.
pub fn catalog() -> Vec<Package> {
    catalog_inner().expect("bundled package manifests are valid")
}

fn catalog_inner() -> Result<Vec<Package>, String> {
    let parse = |text: &str| serde_json::from_str::<Value>(text).map_err(|e| e.to_string());
    let english = parse(include_str!("../../../assets/english-assets.json"))?;
    let variants = parse(include_str!("../../../assets/kokoro-variants.json"))?;
    let compact = parse(include_str!("../../../assets/compact-english-assets.json"))?;

    let mut kokoro = english.as_object().ok_or("english assets")?.clone();
    let q8 = &variants["q8"];
    kokoro.insert(
        "model.onnx".into(),
        serde_json::json!({ "url": q8["url"], "sha256": q8["sha256"], "bytes": q8["bytes"] }),
    );
    for voice in crate::kokoro::voices() {
        let name = format!("{}.bin", voice.name);
        kokoro.entry(name).or_insert_with(|| {
            serde_json::json!({ "url": voice.url, "sha256": voice.sha256, "bytes": voice.bytes })
        });
    }
    let mut packages = vec![Package {
        id: "kokoro-en-q8",
        title: "Kokoro English (Q8) — 28 US and UK voices",
        backend: "kokoro",
        languages: &["en"],
        license: "Apache-2.0 (weights), BSD (CMU dictionary)",
        source: "huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX".into(),
        files: files_from(&kokoro)?,
    }];
    for (key, id, title, backend) in [
        (
            "kitten",
            "kitten-nano-en",
            "Kitten nano English — 8 voices",
            "kitten",
        ),
        (
            "inflect",
            "inflect-nano-en",
            "Inflect nano English — 1 voice",
            "inflect",
        ),
    ] {
        let pins = &compact[key];
        if pins["folder"].as_str() != Some(id) {
            return Err(format!("{key}: folder is not {id}"));
        }
        let mut files = files_from(pins["files"].as_object().ok_or("compact files")?)?;
        // The Python script copies the dictionary from an installed Kokoro
        // package; the native installer fetches the same pinned file.
        for file in &mut files {
            if file.url.is_none()
                && file.derived_from.is_none()
                && let Some(record) = english.get(&file.name)
            {
                file.url = record["url"].as_str().map(String::from);
            }
        }
        packages.push(Package {
            id,
            title,
            backend,
            languages: &["en"],
            license: "Apache-2.0 (weights), BSD (CMU dictionary)",
            source: format!(
                "huggingface.co/{}",
                pins["repository"].as_str().unwrap_or_default()
            ),
            files,
        });
    }
    for package in &packages {
        for file in &package.files {
            if file.url.is_none() && file.derived_from.is_none() {
                return Err(format!("{}: {} has no source", package.id, file.name));
            }
        }
    }
    Ok(packages)
}

/// The catalog entry for `id`.
pub fn package(id: &str) -> Result<Package, String> {
    catalog().into_iter().find(|p| p.id == id).ok_or_else(|| {
        let ids: Vec<&str> = catalog().iter().map(|p| p.id).collect();
        format!(
            "unknown voice package `{id}`; available: {}",
            ids.join(", ")
        )
    })
}

/// Ids of the packages installed under `voices_dir`.
pub fn installed(voices_dir: &Path) -> Vec<&'static str> {
    catalog()
        .into_iter()
        .filter(|p| voices_dir.join(p.id).join(MARKER).is_file())
        .map(|p| p.id)
        .collect()
}

/// Removes an installed package folder. Answers whether one was there.
pub fn remove(id: &str, voices_dir: &Path) -> Result<bool, String> {
    let package = package(id)?;
    let folder = voices_dir.join(package.id);
    if !folder.join(MARKER).is_file() {
        return Ok(false);
    }
    std::fs::remove_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    Ok(true)
}

/// Progress of an install: bytes received so far out of the total download.
pub type Progress<'a> = &'a mut dyn FnMut(u64, u64);

/// Downloads, verifies and installs `id` into `voices_dir/<id>`.
///
/// `mirror`, when given, is tried first as `<mirror>/<id>/<file>`; the pinned
/// upstream URL is the fallback. Every file must match its pinned SHA-256
/// whichever source served it. A package already installed is left alone.
pub fn install(
    id: &str,
    voices_dir: &Path,
    mirror: Option<&str>,
    progress: Progress<'_>,
    cancel: &AtomicBool,
) -> Result<PathBuf, String> {
    install_with(id, voices_dir, mirror, progress, cancel, &|url| {
        let response = ureq::get(url)
            .timeout(std::time::Duration::from_secs(300))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(code, _) => format!("{url} answered {code}"),
                other => format!("{url}: {other}"),
            })?;
        Ok(Box::new(response.into_reader()))
    })
}

type Fetch<'a> = &'a dyn Fn(&str) -> Result<Box<dyn Read + Send>, String>;

fn install_with(
    id: &str,
    voices_dir: &Path,
    mirror: Option<&str>,
    progress: Progress<'_>,
    cancel: &AtomicBool,
    fetch: Fetch<'_>,
) -> Result<PathBuf, String> {
    let package = package(id)?;
    let target = voices_dir.join(package.id);
    // An installed package is completed, not replaced: files this release
    // pins that an older install lacks (more Kokoro voices) are fetched and
    // verified on their own; what is already there is never downloaded again.
    let upgrade = target.join(MARKER).is_file();
    if !upgrade && target.exists() {
        return Err(format!(
            "{} exists but is not an installed package; move it aside first",
            target.display()
        ));
    }
    let wanted: Vec<&PackageFile> = package
        .files
        .iter()
        .filter(|f| f.url.is_some())
        .filter(|f| !upgrade || (!f.acquisition_only && !target.join(&f.name).is_file()))
        .collect();
    if upgrade && wanted.is_empty() {
        return Ok(target);
    }
    std::fs::create_dir_all(voices_dir).map_err(|e| format!("{}: {e}", voices_dir.display()))?;
    let stage = tempfile::Builder::new()
        .prefix(&format!(".{}.", package.id))
        .tempdir_in(voices_dir)
        .map_err(|e| e.to_string())?;
    let total: u64 = wanted.iter().filter_map(|f| f.bytes).sum();
    let mut done = 0u64;
    progress(0, total);
    for file in wanted.iter().copied() {
        let destination = stage.path().join(&file.name);
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let upstream = file.url.as_deref().unwrap_or_default();
        let mirrored =
            mirror.map(|m| format!("{}/{}/{}", m.trim_end_matches('/'), package.id, file.name));
        let mut last_error = String::new();
        let mut fetched = false;
        for url in mirrored.iter().map(String::as_str).chain([upstream]) {
            let before = done;
            match fetch_verified(url, &destination, file, cancel, fetch, &mut |n| {
                done += n;
                progress(done, total);
            }) {
                Ok(()) => {
                    fetched = true;
                    break;
                }
                Err(e) if cancel.load(Ordering::Relaxed) => return Err(e),
                Err(e) => {
                    done = before;
                    last_error = e;
                }
            }
        }
        if !fetched {
            return Err(format!("{}: {last_error}", file.name));
        }
    }
    if !upgrade {
        for file in package.files.iter().filter(|f| f.derived_from.is_some()) {
            let archive = stage
                .path()
                .join(file.derived_from.as_deref().unwrap_or_default());
            let bytes = extract_npy(&archive, &file.name)?;
            check(&bytes, file)?;
            let destination = stage.path().join(&file.name);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&destination, bytes).map_err(|e| e.to_string())?;
        }
        for file in package.files.iter().filter(|f| f.acquisition_only) {
            let _ = std::fs::remove_file(stage.path().join(&file.name));
        }
    }
    let record = serde_json::json!({
        "id": package.id,
        "backend": package.backend,
        "languages": package.languages,
        "license": package.license,
        "source": package.source,
        "files": package.files.iter().filter(|f| !f.acquisition_only).map(|f| serde_json::json!({
            "name": f.name, "url": f.url, "sha256": f.sha256, "derivedFrom": f.derived_from,
        })).collect::<Vec<_>>(),
    });
    let record = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    if upgrade {
        // Each verified file moves in; the marker is rewritten last, so an
        // interrupted upgrade leaves the old, working package plus extras.
        for file in &wanted {
            let destination = target.join(&file.name);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::rename(stage.path().join(&file.name), &destination)
                .map_err(|e| format!("{}: {e}", destination.display()))?;
        }
        let marker = stage.path().join(MARKER);
        std::fs::write(&marker, record).map_err(|e| e.to_string())?;
        std::fs::rename(&marker, target.join(MARKER)).map_err(|e| e.to_string())?;
        return Ok(target);
    }
    std::fs::write(stage.path().join(MARKER), record).map_err(|e| e.to_string())?;
    let staged = stage.keep();
    std::fs::rename(&staged, &target).map_err(|e| {
        let _ = std::fs::remove_dir_all(&staged);
        format!("{}: {e}", target.display())
    })?;
    Ok(target)
}

/// Installed packages whose folder lacks files this release pins — an
/// install of the same id completes them.
pub fn outdated(voices_dir: &Path) -> Vec<&'static str> {
    catalog()
        .into_iter()
        .filter(|p| voices_dir.join(p.id).join(MARKER).is_file())
        .filter(|p| {
            p.files.iter().any(|f| {
                f.url.is_some()
                    && !f.acquisition_only
                    && !voices_dir.join(p.id).join(&f.name).is_file()
            })
        })
        .map(|p| p.id)
        .collect()
}

fn fetch_verified(
    url: &str,
    destination: &Path,
    file: &PackageFile,
    cancel: &AtomicBool,
    fetch: Fetch<'_>,
    advanced: &mut dyn FnMut(u64),
) -> Result<(), String> {
    let mut reader = fetch(url)?;
    let mut out = std::fs::File::create(destination).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let n = reader
            .read(&mut buffer)
            .map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        size += n as u64;
        if file.bytes.is_some_and(|expected| size > expected) {
            return Err(format!("{url}: larger than the pinned size"));
        }
        hasher.update(&buffer[..n]);
        out.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
        advanced(n as u64);
    }
    let digest = hex(&hasher.finalize());
    if digest != file.sha256 || file.bytes.is_some_and(|expected| size != expected) {
        return Err(format!("{url}: checksum mismatch"));
    }
    out.sync_all().map_err(|e| e.to_string())
}

fn check(bytes: &[u8], file: &PackageFile) -> Result<(), String> {
    if hex(&Sha256::digest(bytes)) != file.sha256
        || file.bytes.is_some_and(|b| b != bytes.len() as u64)
    {
        return Err(format!("{}: checksum mismatch", file.name));
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The raw little-endian f32 payload of `styles/<stem>.f32` from a Kitten
/// `voices.npz`, checked for the (400, 256) float32 layout the loader expects.
fn extract_npy(archive: &Path, name: &str) -> Result<Vec<u8>, String> {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("bad style name")?;
    let file = std::fs::File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut entry = zip
        .by_name(&format!("{stem}.npy"))
        .map_err(|e| format!("{stem}.npy: {e}"))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err(format!("{stem}.npy: not a version 1 .npy file"));
    }
    let end = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(bytes.get(10..end).ok_or("short header")?)
        .map_err(|e| e.to_string())?;
    if header.trim() != "{'descr': '<f4', 'fortran_order': False, 'shape': (400, 256), }" {
        return Err(format!("{stem}.npy: unexpected style format"));
    }
    Ok(bytes.split_off(end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn catalog_has_sources_for_every_file() {
        let packages = catalog();
        assert_eq!(
            packages.iter().map(|p| p.id).collect::<Vec<_>>(),
            ["kokoro-en-q8", "kitten-nano-en", "inflect-nano-en"]
        );
        let kokoro = &packages[0];
        assert!(kokoro.files.iter().any(|f| f.name == "model.onnx"
            && f.url.as_deref().unwrap().ends_with("model_quantized.onnx")));
        assert!(kokoro.download_bytes() > 90_000_000);
        let voices = kokoro
            .files
            .iter()
            .filter(|f| f.name.ends_with(".bin"))
            .count();
        assert_eq!(voices, 28, "every pinned English Kokoro voice");
        assert!(package("missing").is_err());
    }

    /// A fake source for one tiny package-shaped install.
    fn fake(
        served: HashMap<String, Vec<u8>>,
    ) -> impl Fn(&str) -> Result<Box<dyn Read + Send>, String> {
        move |url: &str| {
            served
                .get(url)
                .cloned()
                .map(|b| Box::new(std::io::Cursor::new(b)) as Box<dyn Read + Send>)
                .ok_or_else(|| format!("{url} answered 404"))
        }
    }

    fn served_inflect(corrupt: Option<&str>) -> HashMap<String, Vec<u8>> {
        // Serve the real installed Inflect files when present; otherwise the
        // test only checks failure paths.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/inflect-nano-en");
        let package = package("inflect-nano-en").unwrap();
        let mut served = HashMap::new();
        for file in package.files {
            let Some(url) = file.url else { continue };
            let mut bytes = std::fs::read(root.join(&file.name)).unwrap_or_default();
            if corrupt == Some(file.name.as_str()) && !bytes.is_empty() {
                bytes[0] ^= 0xff;
            }
            served.insert(url, bytes);
        }
        served
    }

    #[test]
    fn a_mismatch_or_cancel_leaves_nothing_installed() {
        let dir = tempfile::tempdir().unwrap();
        let fetch = fake(served_inflect(Some("config.json")));
        let error = install_with(
            "inflect-nano-en",
            dir.path(),
            None,
            &mut |_, _| {},
            &AtomicBool::new(false),
            &fetch,
        )
        .unwrap_err();
        assert!(
            error.contains("mismatch") || error.contains("pinned size"),
            "{error}"
        );
        let cancel = AtomicBool::new(true);
        assert!(
            install_with(
                "inflect-nano-en",
                dir.path(),
                None,
                &mut |_, _| {},
                &cancel,
                &fetch
            )
            .is_err()
        );
        assert!(installed(dir.path()).is_empty());
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "no staging left"
        );
    }

    #[test]
    #[ignore = "requires installed Inflect nano"]
    fn installs_from_a_mirror_and_falls_back_to_upstream() {
        let dir = tempfile::tempdir().unwrap();
        let mut served = served_inflect(None);
        // The mirror has only the config; everything else falls back upstream.
        let config = package("inflect-nano-en")
            .unwrap()
            .files
            .into_iter()
            .find(|f| f.name == "config.json")
            .unwrap();
        let bytes = served[config.url.as_deref().unwrap()].clone();
        served.insert(
            "https://mirror.test/inflect-nano-en/config.json".into(),
            bytes,
        );
        let mut last = (0, 0);
        let path = install_with(
            "inflect-nano-en",
            dir.path(),
            Some("https://mirror.test/"),
            &mut |done, total| last = (done, total),
            &AtomicBool::new(false),
            &fake(served),
        )
        .unwrap();
        assert_eq!(last.0, last.1);
        assert_eq!(installed(dir.path()), ["inflect-nano-en"]);
        crate::compact_english::CompactEnglish::load(
            &path,
            crate::compact_english::CompactKind::Inflect,
        )
        .unwrap();
        // An install with a file missing completes it, fetching only that file.
        std::fs::remove_file(path.join("config.json")).unwrap();
        assert_eq!(outdated(dir.path()), ["inflect-nano-en"]);
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let inner = fake(served_inflect(None));
        let counting = |url: &str| {
            calls.fetch_add(1, Ordering::Relaxed);
            inner(url)
        };
        install_with(
            "inflect-nano-en",
            dir.path(),
            None,
            &mut |_, _| {},
            &AtomicBool::new(false),
            &counting,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(path.join("config.json").is_file());
        assert!(outdated(dir.path()).is_empty());
        assert!(remove("inflect-nano-en", dir.path()).unwrap());
        assert!(installed(dir.path()).is_empty());
    }
}
