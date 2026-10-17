use crate::{Language, dataset::safe_relative};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceManifest {
    pub schema_version: u32,
    pub name: String,
    pub language: Language,
    pub sample_rate: u32,
    pub model: String,
    /// Must identify the exact text frontend/token inventory used in training.
    pub frontend: String,
    /// Optional outer text contract; absence preserves legacy trained voices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<String>,
    pub speakers: Vec<String>,
    pub license: String,
    pub provenance: String,
}

impl VoiceManifest {
    pub fn validate(&self, root: &Path) -> Result<(), String> {
        crate::text::validate_version(self.normalization.as_deref())?;
        if self.normalization.is_some() && self.frontend == "grapheme-v1" {
            return Err("legacy grapheme frontend cannot use text-v1 normalization".into());
        }
        if self.schema_version != 1 {
            return Err("unsupported voice schema_version".into());
        }
        if self.name.trim().is_empty()
            || self.frontend.trim().is_empty()
            || self.license.trim().is_empty()
            || self.provenance.trim().is_empty()
        {
            return Err("name, frontend, license and provenance are required".into());
        }
        if !(8000..=96000).contains(&self.sample_rate) {
            return Err("invalid voice sample rate".into());
        }
        if self.speakers.is_empty() || self.speakers.iter().any(|s| s.trim().is_empty()) {
            return Err("at least one nonempty speaker is required".into());
        }
        let unique: std::collections::HashSet<_> = self.speakers.iter().collect();
        if unique.len() != self.speakers.len() {
            return Err("duplicate speaker".into());
        }
        if !safe_relative(&self.model) {
            return Err("model must be a relative path without traversal".into());
        }
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let model = root
            .join(&self.model)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if !model.starts_with(root) || !model.is_file() {
            return Err("model must be a file within the voice package".into());
        }
        Ok(())
    }
}
