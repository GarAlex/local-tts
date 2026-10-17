//! Portable reference-voice packages; model weights remain separate.
use crate::{Language, chatterbox::Chatterbox};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceProfile {
    pub schema_version: u32,
    pub name: String,
    /// Suggested synthesis language; reference identity can be used across languages.
    pub language: Language,
    pub exaggeration: f32,
    pub model_revision: String,
    pub reference_sha256: String,
}
fn revision() -> String {
    let pins: serde_json::Value =
        serde_json::from_str(include_str!("../../../assets/chatterbox-assets.json"))
            .expect("embedded pins");
    pins["revision"].as_str().expect("embedded revision").into()
}
impl VoiceProfile {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.model_revision != revision() {
            return Err("unsupported profile schema or Chatterbox model revision".into());
        }
        if self.name.trim().is_empty()
            || self.name.chars().count() > 80
            || self.name.chars().any(char::is_control)
        {
            return Err("profile name must contain 1–80 printable characters".into());
        }
        if self.language == Language::Uk {
            return Err("Chatterbox profiles do not support Ukrainian".into());
        }
        if !self.exaggeration.is_finite() || !(0.0..=1.0).contains(&self.exaggeration) {
            return Err("profile expressiveness must be between 0 and 1".into());
        }
        if self.reference_sha256.len() != 64
            || !self.reference_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid reference checksum".into());
        }
        Ok(())
    }
    /// Save a new directory. Existing packages are never overwritten.
    /// Reference recordings are copied, so profiles survive source-file relocation.
    pub fn create(
        folder: &Path,
        reference: &Path,
        name: &str,
        language: Language,
        exaggeration: f32,
    ) -> Result<Self, String> {
        let parent = folder
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut profile = Self {
            schema_version: 1,
            name: name.trim().into(),
            language,
            exaggeration,
            model_revision: revision(),
            reference_sha256: "0".repeat(64),
        };
        profile.validate()?;
        if folder.exists() {
            return Err("profile folder already exists; choose a new folder".into());
        }
        if std::fs::metadata(reference)
            .map_err(|e| e.to_string())?
            .len()
            > 8_000_000
        {
            return Err("reference file exceeds eight megabytes".into());
        }
        crate::chatterbox::read_reference(reference).map_err(|e| e.to_string())?;
        let stage = tempfile::tempdir_in(parent).map_err(|e| e.to_string())?;
        std::fs::copy(reference, stage.path().join("reference.wav")).map_err(|e| e.to_string())?;
        crate::chatterbox::read_reference(&stage.path().join("reference.wav"))
            .map_err(|e| e.to_string())?;
        profile.reference_sha256 = crate::integrity::hash_file(stage.path(), "reference.wav")?;
        std::fs::write(
            stage.path().join("profile.json"),
            serde_json::to_vec_pretty(&profile).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        // Exclusive publication prevents overwriting another profile created meanwhile.
        std::fs::create_dir(folder).map_err(|e| e.to_string())?;
        let result = (|| {
            for file in ["reference.wav", "profile.json"] {
                std::fs::rename(stage.path().join(file), folder.join(file))
                    .map_err(|e| e.to_string())?;
            }
            Ok::<_, String>(())
        })();
        if result.is_err() {
            for file in ["reference.wav", "profile.json"] {
                let _ = std::fs::remove_file(folder.join(file));
            }
            let _ = std::fs::remove_dir(folder);
        }
        result?;
        Ok(profile)
    }
    pub fn load(folder: &Path) -> Result<Self, String> {
        for name in ["profile.json", "reference.wav"] {
            let meta = folder
                .join(name)
                .symlink_metadata()
                .map_err(|e| e.to_string())?;
            let max = if name == "profile.json" {
                16384
            } else {
                8_000_000
            };
            if !meta.is_file() || meta.len() > max {
                return Err("profile asset is not a regular bounded file".into());
            }
        }
        let profile: Self = serde_json::from_slice(
            &std::fs::read(folder.join("profile.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        profile.validate()?;
        if crate::integrity::hash_file(folder, "reference.wav")? != profile.reference_sha256 {
            return Err("profile reference checksum mismatch".into());
        }
        crate::chatterbox::read_reference(&folder.join("reference.wav"))
            .map_err(|e| e.to_string())?;
        Ok(profile)
    }
}
/// Load the separately installed pinned model with this profile's reference and intensity.
pub fn load_engine(model_folder: &Path, profile_folder: &Path) -> Result<Chatterbox, String> {
    let profile = VoiceProfile::load(profile_folder)?;
    let mut engine =
        Chatterbox::load_with_reference(model_folder, Some(&profile_folder.join("reference.wav")))
            .map_err(|e| e.to_string())?;
    engine
        .set_exaggeration(profile.exaggeration)
        .map_err(|e| e.to_string())?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(path: &Path) {
        crate::Audio {
            sample_rate: 24000,
            samples: vec![0.1; 24000],
        }
        .write_wav(path)
        .unwrap();
    }
    #[test]
    fn portable_profile_survives_source_removal_and_detects_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.wav");
        reference(&source);
        let folder = dir.path().join("profile");
        VoiceProfile::create(&folder, &source, "My voice", Language::Ru, 0.7).unwrap();
        assert!(VoiceProfile::create(&folder, &source, "Replacement", Language::En, 0.5).is_err());
        std::fs::remove_file(&source).unwrap();
        let relocated = dir.path().join("moved");
        std::fs::rename(folder, &relocated).unwrap();
        let profile = VoiceProfile::load(&relocated).unwrap();
        assert_eq!(profile.name, "My voice");
        assert_eq!(profile.language, Language::Ru);
        assert_eq!(profile.exaggeration, 0.7);
        let path = relocated.join("reference.wav");
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(path, bytes).unwrap();
        assert!(
            VoiceProfile::load(&relocated)
                .unwrap_err()
                .contains("checksum")
        );
    }
    #[test]
    fn rejects_invalid_settings_revision_and_oversized_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.wav");
        reference(&source);
        let folder = dir.path().join("profile");
        for (name, language, intensity) in [
            ("", Language::En, 0.5),
            ("Voice", Language::Uk, 0.5),
            ("Voice", Language::En, f32::NAN),
        ] {
            assert!(VoiceProfile::create(&folder, &source, name, language, intensity).is_err());
            assert!(!folder.exists());
        }
        let mut profile =
            VoiceProfile::create(&folder, &source, "Voice", Language::En, 0.5).unwrap();
        profile.model_revision = "different-model".into();
        std::fs::write(
            folder.join("profile.json"),
            serde_json::to_vec(&profile).unwrap(),
        )
        .unwrap();
        assert!(VoiceProfile::load(&folder).is_err());
        std::fs::write(folder.join("profile.json"), vec![b' '; 16385]).unwrap();
        assert!(VoiceProfile::load(&folder).is_err());
    }
}
