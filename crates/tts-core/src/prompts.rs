//! Original MIT starter prompts and resumable prompt-list matching.
use crate::{Language, dataset::Dataset};
use std::collections::HashSet;

pub struct PromptList {
    pub language: Language,
    pub lines: Vec<String>,
}
impl PromptList {
    pub fn starter(language: Language) -> Self {
        let text = match language {
            Language::En => include_str!("../../../assets/prompts/en.txt"),
            Language::Ru => include_str!("../../../assets/prompts/ru.txt"),
            Language::Uk => include_str!("../../../assets/prompts/uk.txt"),
            Language::De => include_str!("../../../assets/prompts/de.txt"),
            Language::Fr => include_str!("../../../assets/prompts/fr.txt"),
            Language::Es => include_str!("../../../assets/prompts/es.txt"),
            Language::It => include_str!("../../../assets/prompts/it.txt"),
            Language::Pt => include_str!("../../../assets/prompts/pt.txt"),
            Language::Pl => include_str!("../../../assets/prompts/pl.txt"),
            Language::Nl => include_str!("../../../assets/prompts/nl.txt"),
        };
        Self::parse(language, text).expect("valid bundled starter prompts")
    }
    /// UTF-8, one prompt per line. Empty lines ignored; duplicates rejected.
    pub fn parse(language: Language, text: &str) -> Result<Self, String> {
        if text.len() > 1_000_000 {
            return Err("prompt list exceeds one megabyte".into());
        }
        let mut lines = Vec::new();
        let mut seen = HashSet::new();
        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if line.chars().count() > 300 || line.chars().any(char::is_control) {
                return Err(
                    "each prompt must contain at most 300 characters and no control characters"
                        .into(),
                );
            }
            if !seen.insert(crate::frontend::normalize(line, language)?) {
                return Err("duplicate normalized prompt".into());
            }
            lines.push(line.into());
            if lines.len() > 10000 {
                return Err("prompt list exceeds 10000 lines".into());
            }
        }
        if lines.is_empty() {
            return Err("prompt list is empty".into());
        }
        Ok(Self { language, lines })
    }
    /// Matches all dataset splits by normalized transcript, scoped to speaker.
    pub fn remaining<'a>(&'a self, data: &Dataset, speaker: &str) -> Result<Vec<&'a str>, String> {
        if data.language != self.language {
            return Err("prompt and dataset languages differ".into());
        }
        if speaker.trim().is_empty() {
            return Err("enter a speaker".into());
        }
        let recorded: HashSet<_> = data
            .utterances
            .iter()
            .filter(|u| u.speaker == speaker.trim())
            .map(|u| crate::frontend::match_key(&u.text, self.language))
            .collect();
        Ok(self
            .lines
            .iter()
            .filter(|line| !recorded.contains(&crate::frontend::match_key(line, self.language)))
            .map(String::as_str)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Audio, collection::Collection, dataset::Split};
    #[test]
    fn resume_uses_saved_transcript_and_speaker() {
        let root = tempfile::tempdir().unwrap();
        let mut collection = Collection::create(root.path(), Language::Uk).unwrap();
        let prompts = PromptList::starter(Language::Uk);
        let audio = Audio {
            sample_rate: 22050,
            samples: vec![0.1; 22050],
        };
        collection
            .add_clip(
                &audio,
                &prompts.lines[0].to_uppercase(),
                "speaker",
                Split::Validation,
            )
            .unwrap();
        let reopened = Collection::open(collection.manifest()).unwrap();
        assert_eq!(
            prompts.remaining(&reopened.data, "speaker").unwrap().len(),
            11
        );
        assert_eq!(
            prompts.remaining(&reopened.data, "speaker").unwrap()[0],
            prompts.lines[1]
        );
        assert_eq!(
            prompts.remaining(&reopened.data, "other").unwrap().len(),
            12
        );
        assert!(
            PromptList::starter(Language::En)
                .remaining(&reopened.data, "speaker")
                .is_err()
        );
    }
    #[test]
    fn lists_are_bounded_unique_and_valid() {
        for language in Language::ALL {
            assert_eq!(PromptList::starter(language).lines.len(), 12);
        }
        assert_eq!(
            PromptList::parse(Language::En, "Hello.\r\n\r\nWorld.")
                .unwrap()
                .lines
                .len(),
            2
        );
        for bad in ["", "Hello\nHELLO", "a\0b", &"a".repeat(301)] {
            assert!(PromptList::parse(Language::En, bad).is_err());
        }
    }
}
