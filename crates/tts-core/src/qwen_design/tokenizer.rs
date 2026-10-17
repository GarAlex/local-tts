// Adapted from WaveKat, Apache-2.0, Copyright 2026 WaveKat.
// Source revision ad3c7d0f927370ab23e1afad176c3cccff51467a.
// Modified here for the pinned local runtime, bounds and cancellation.
use std::path::Path;

use super::TtsError;

// Special text token IDs (Qwen3 BPE vocab)
pub const IM_START: u32 = 151644;
pub const IM_END: u32 = 151645;
pub const ASSISTANT: u32 = 77091;
pub const NEWLINE: u32 = 198;
pub const TTS_BOS: u32 = 151672;
pub const TTS_EOS: u32 = 151673;
pub const TTS_PAD: u32 = 151671;

/// Language → codec language ID mapping.
pub fn language_id(lang: &str) -> Option<i64> {
    match lang {
        "en" | "english" => Some(2050),
        "ko" | "korean" => Some(2064),
        "zh" | "chinese" => Some(2055),
        "ja" | "japanese" => Some(2058),
        "de" | "german" => Some(2053),
        "es" | "spanish" => Some(2054),
        "fr" | "french" => Some(2061),
        "ru" | "russian" => Some(2069),
        "it" | "italian" => Some(2070),
        "pt" | "portuguese" => Some(2071),
        _ => None,
    }
}

/// Loads the pinned publisher JSON tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Build tokenizer from `tokenizer/tokenizer.json` in `model_dir`.
    pub fn new(model_dir: &Path) -> Result<Self, TtsError> {
        let inner = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| TtsError::Model(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encode text into BPE token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::Synthesis(format!("tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}
