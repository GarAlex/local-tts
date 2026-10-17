//! Native Qwen3 VoiceDesign inference. Models are acquired separately, never at load.
mod model;
mod sampler;
mod tokenizer;
use crate::{Audio, Language};
use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool};
#[derive(Debug)]
pub enum TtsError {
    Model(String),
    Synthesis(String),
    UnsupportedLanguage(String),
}
impl std::fmt::Display for TtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(s) | Self::Synthesis(s) => f.write_str(s),
            Self::UnsupportedLanguage(s) => {
                write!(f, "Qwen voice design does not support language {s}")
            }
        }
    }
}
impl std::error::Error for TtsError {}
pub struct QwenDesign {
    model: model::Model,
    tokenizer: tokenizer::Tokenizer,
}
impl QwenDesign {
    pub fn supports(language: Language) -> bool {
        tokenizer::language_id(language.code()).is_some()
    }
    pub fn load(root: &Path) -> Result<Self, TtsError> {
        Self::load_cancellable(root, &AtomicBool::new(false))
    }
    /// Cancellation is checked while hashing and between graph/table loads.
    pub fn load_cancellable(root: &Path, cancel: &AtomicBool) -> Result<Self, TtsError> {
        crate::paragraph::check_cancel(cancel).map_err(TtsError::Synthesis)?;
        let pins: serde_json::Value =
            serde_json::from_str(include_str!("../../../../assets/qwen-design-assets.json"))
                .map_err(|e| TtsError::Model(e.to_string()))?;
        let hashes: BTreeMap<String, String> = pins["files"]
            .as_object()
            .ok_or_else(|| TtsError::Model("invalid model pins".into()))?
            .iter()
            .map(|(name, r)| {
                Ok((
                    name.clone(),
                    r["sha256"]
                        .as_str()
                        .ok_or_else(|| TtsError::Model("missing pin".into()))?
                        .into(),
                ))
            })
            .collect::<Result<_, TtsError>>()?;
        let entries: Vec<_> = hashes.into_iter().collect();
        for batch in entries.chunks(16) {
            crate::integrity::verify_cancellable(
                root,
                &batch.iter().cloned().collect(),
                &[],
                cancel,
            )
            .map_err(TtsError::Model)?;
        }
        Ok(Self {
            tokenizer: tokenizer::Tokenizer::new(root)?,
            model: model::Model::load(root, cancel)?,
        })
    }
    /// Generate one short utterance from a voice description; no reference audio.
    /// English, Russian, DE/FR/ES/IT/PT supported. UK/PL/NL are rejected.
    pub fn synthesize_design(
        &self,
        text: &str,
        language: Language,
        description: &str,
        cancel: &AtomicBool,
    ) -> Result<Audio, TtsError> {
        crate::paragraph::check_cancel(cancel).map_err(TtsError::Synthesis)?;
        if !Self::supports(language) {
            return Err(TtsError::UnsupportedLanguage(language.code().into()));
        }
        if text.trim().is_empty()
            || description.trim().is_empty()
            || text.chars().count() > 300
            || description.chars().count() > 500
            || text.contains('\0')
            || description.contains('\0')
            || text.contains("<|")
            || description.contains("<|")
            || description.contains(['<', '>'])
        {
            return Err(TtsError::Synthesis("provide text (1–300 characters) and a voice description (1–500); reserved tokens are disallowed".into()));
        }
        let text = crate::text::prepare(text, language).map_err(TtsError::Synthesis)?;
        let tokens = self.tokenizer.encode(&text)?;
        let mut instruction = vec![tokenizer::IM_START];
        instruction.extend(self.tokenizer.encode("user")?);
        instruction.push(tokenizer::NEWLINE);
        instruction.extend(self.tokenizer.encode("<instruct>")?);
        instruction.extend(self.tokenizer.encode(description)?);
        instruction.extend(self.tokenizer.encode("</instruct>")?);
        instruction.extend([tokenizer::IM_END, tokenizer::NEWLINE]);
        if tokens.len() > 256 || instruction.len() > 256 {
            return Err(TtsError::Synthesis(
                "too many voice-design text tokens".into(),
            ));
        }
        self.model
            .synthesize(&tokens, language.code(), Some(&instruction), cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn advertised_language_support_is_explicit() {
        for lang in [
            Language::En,
            Language::Ru,
            Language::De,
            Language::Fr,
            Language::Es,
            Language::It,
            Language::Pt,
        ] {
            assert!(QwenDesign::supports(lang));
        }
        for lang in [Language::Uk, Language::Pl, Language::Nl] {
            assert!(!QwenDesign::supports(lang));
        }
    }
    #[test]
    fn missing_assets_fail_before_loading_graphs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            QwenDesign::load(dir.path()),
            Err(TtsError::Model(_))
        ));
    }
}
