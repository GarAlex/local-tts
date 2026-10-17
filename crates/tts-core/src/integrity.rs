//! SHA-256 corruption detection. Hash manifests are not signatures or legal clearance.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::Path, sync::atomic::AtomicBool};

pub fn hash_file(root: &Path, name: &str) -> Result<String, String> {
    hash_file_cancellable(root, name, &AtomicBool::new(false))
}
/// Hash with cancellation checked between 64 KiB reads.
pub fn hash_file_cancellable(
    root: &Path,
    name: &str,
    cancel: &AtomicBool,
) -> Result<String, String> {
    crate::paragraph::check_cancel(cancel)?;
    if !crate::dataset::safe_relative(name) {
        return Err("unsafe integrity path".into());
    }
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let path = root
        .join(name)
        .canonicalize()
        .map_err(|e| format!("{name}: {e}"))?;
    if !path.starts_with(&root) {
        return Err("integrity file escapes package".into());
    }
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > 2_000_000_000 {
        return Err("package file exceeds two gigabytes or is not a regular file".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        crate::paragraph::check_cancel(cancel)?;
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn verify(
    root: &Path,
    hashes: &BTreeMap<String, String>,
    required: &[&str],
) -> Result<(), String> {
    verify_cancellable(root, hashes, required, &AtomicBool::new(false))
}
/// Verify without continuing to hash a large package after cancellation.
pub fn verify_cancellable(
    root: &Path,
    hashes: &BTreeMap<String, String>,
    required: &[&str],
    cancel: &AtomicBool,
) -> Result<(), String> {
    crate::paragraph::check_cancel(cancel)?;
    for name in required {
        if !hashes.contains_key(*name) {
            return Err(format!("missing checksum for {name}"));
        }
    }
    if hashes.len() > 32 {
        return Err("too many checksum entries".into());
    }
    for (name, expected) in hashes {
        if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("invalid checksum for {name}"));
        }
        if hash_file_cancellable(root, name, cancel)? != expected.to_lowercase() {
            return Err(format!("checksum mismatch for {name}"));
        }
    }
    Ok(())
}
pub fn verify_vits(root: &Path, model: &str) -> Result<(), String> {
    let path = root.join("checksums.json");
    if fs::metadata(&path)
        .map_err(|_| "missing checksums.json; re-export or seal this trusted legacy package")?
        .len()
        > 65536
    {
        return Err("checksum manifest too large".into());
    }
    let hashes = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid checksums: {e}"))?;
    verify(root, &hashes, &[model, "tokens.json", "voice.json"])
}
/// Establish a baseline only for a package whose current contents you trust.
pub fn seal_vits(root: &Path) -> Result<(), String> {
    let manifest: crate::voice::VoiceManifest =
        serde_json::from_slice(&fs::read(root.join("voice.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    manifest.validate(root)?;
    if manifest.frontend != "grapheme-v1"
        && manifest.frontend != crate::frontend::VERSION
        && manifest.frontend != crate::training::IPA_FRONTEND
    {
        return Err("only VITS packages can be sealed".into());
    }
    let mut hashes = BTreeMap::new();
    for name in [
        &manifest.model,
        "tokens.json",
        "voice.json",
        "rights.json",
        "VITS-LICENSE.txt",
    ] {
        hashes.insert(name.to_string(), hash_file(root, name)?);
    }
    if manifest.frontend == crate::training::IPA_FRONTEND {
        for name in ["cmudict.dict", "CMU-LICENSE.txt"] {
            hashes.insert(name.into(), hash_file(root, name)?);
        }
    }
    if root.join("reference.json").exists() {
        hashes.insert("reference.json".into(), hash_file(root, "reference.json")?);
    }
    fs::write(
        root.join("checksums.json"),
        serde_json::to_vec_pretty(&hashes).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_precedes_file_access_and_preserves_normal_hashes() {
        let root = tempfile::tempdir().unwrap();
        let cancelled = AtomicBool::new(true);
        assert!(
            hash_file_cancellable(root.path(), "missing", &cancelled)
                .unwrap_err()
                .contains("cancelled")
        );
        assert!(
            verify_cancellable(root.path(), &BTreeMap::new(), &[], &cancelled)
                .unwrap_err()
                .contains("cancelled")
        );
        fs::write(root.path().join("model"), b"model").unwrap();
        assert_eq!(
            hash_file(root.path(), "model").unwrap(),
            hash_file_cancellable(root.path(), "model", &AtomicBool::new(false)).unwrap()
        );
    }
    #[test]
    fn verifies_and_rejects_changed_missing_and_unsafe_files() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("model"), b"model").unwrap();
        let mut hashes =
            BTreeMap::from([("model".into(), hash_file(root.path(), "model").unwrap())]);
        verify(root.path(), &hashes, &["model"]).unwrap();
        assert!(verify(root.path(), &hashes, &["tokens"]).is_err());
        fs::write(root.path().join("model"), b"changed").unwrap();
        assert!(
            verify(root.path(), &hashes, &["model"])
                .unwrap_err()
                .contains("mismatch")
        );
        hashes.insert("../outside".into(), "a".repeat(64));
        assert!(verify(root.path(), &hashes, &[]).is_err());
    }
}
