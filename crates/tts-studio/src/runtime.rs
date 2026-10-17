//! Resolve installed resources without depending on the build checkout.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
struct Manifest {
    version: String,
    files: BTreeMap<String, String>,
}

pub struct Runtime {
    pub root: PathBuf,
    pub cli: PathBuf,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked_path(root: &Path, name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if name.is_empty() || !path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err("unsafe runtime resource path".into());
    }
    let mut checked = root.to_path_buf();
    for component in path.components() {
        checked.push(component);
        if checked.is_symlink() {
            return Err("runtime resources must not contain symbolic links".into());
        }
    }
    Ok(checked)
}
fn verify(root: &Path, manifest: &Manifest) -> Result<(), String> {
    for (name, expected) in &manifest.files {
        let path = checked_path(root, name)?;
        if path
            .symlink_metadata()
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
            || digest(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
                != *expected
        {
            return Err(format!("runtime resource changed: {}", path.display()));
        }
    }
    Ok(())
}
fn install(source: &Path, home: &Path) -> Result<Runtime, String> {
    let bytes = std::fs::read(source.join("release-runtime.json")).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if manifest.version != env!("CARGO_PKG_VERSION") || manifest.files.is_empty() {
        return Err("runtime version mismatch or empty resource manifest".into());
    }
    verify(source, &manifest)?;
    std::fs::create_dir_all(home).map_err(|e| e.to_string())?;
    let root = home.join(format!(
        "runtime-{}-{}",
        manifest.version,
        &digest(&bytes)[..16]
    ));
    if !root.exists() {
        let stage = tempfile::tempdir_in(home).map_err(|e| e.to_string())?;
        for name in manifest.files.keys() {
            let target = checked_path(stage.path(), name)?;
            std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::copy(checked_path(source, name)?, target).map_err(|e| e.to_string())?;
        }
        // A concurrent launch may have completed installation first.
        if let Err(error) = std::fs::rename(stage.path(), &root)
            && !root.is_dir()
        {
            return Err(error.to_string());
        }
    }
    if root
        .symlink_metadata()
        .map_err(|e| e.to_string())?
        .file_type()
        .is_symlink()
    {
        return Err("runtime folder must not be a symbolic link".into());
    }
    verify(&root, &manifest)?;
    let cli = root.join(if cfg!(windows) {
        "bin/local-tts.exe"
    } else {
        "bin/local-tts"
    });
    if !cli.is_file() {
        return Err("packaged command-line tool is missing".into());
    }
    Ok(Runtime { root, cli })
}

pub fn locate() -> Result<Runtime, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let resources = exe
        .parent()
        .ok_or("executable has no parent")?
        .join("../Resources/runtime");
    if resources.join("release-runtime.json").is_file() {
        let home = match std::env::var_os("LOCAL_TTS_HOME") {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(std::env::var_os("HOME").ok_or("HOME is not set")?)
                .join("Library/Application Support/Local TTS"),
        };
        return install(&resources, &home);
    }
    #[cfg(debug_assertions)]
    {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let cli = root.join(if cfg!(windows) {
            "target/debug/local-tts.exe"
        } else {
            "target/debug/local-tts"
        });
        Ok(Runtime { root, cli })
    }
    #[cfg(not(debug_assertions))]
    Err("Release resources are missing. Launch the packaged Local TTS Studio application.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsafe_resource_paths_fail() {
        for path in ["../secret", "/tmp/secret", "", "a/../../secret"] {
            assert!(checked_path(Path::new("/tmp"), path).is_err());
        }
    }
    #[test]
    fn install_reuses_data_and_rejects_changed_code() {
        let source = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir(source.path().join("bin")).unwrap();
        let binary = if cfg!(windows) {
            "bin/local-tts.exe"
        } else {
            "bin/local-tts"
        };
        std::fs::write(source.path().join(binary), b"fixture").unwrap();
        std::fs::write(
            source.path().join("release-runtime.json"),
            serde_json::to_vec(&serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"), "files": {binary: digest(b"fixture")}
            }))
            .unwrap(),
        )
        .unwrap();
        let first = install(source.path(), home.path()).unwrap();
        std::fs::write(first.root.join("user-data"), b"keep").unwrap();
        assert_eq!(
            install(source.path(), home.path()).unwrap().root,
            first.root
        );
        assert_eq!(
            std::fs::read(first.root.join("user-data")).unwrap(),
            b"keep"
        );
        std::fs::write(first.cli, b"changed").unwrap();
        assert!(install(source.path(), home.path()).is_err());
    }
}
