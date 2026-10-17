//! User-authored Cyrillic stress corrections; no bundled external language data.
use crate::Language;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub const FILE_NAME: &str = "pronunciations.json";
const MAX_BYTES: usize = 1_048_576;
const MAX_ENTRIES: usize = 10000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pronunciations {
    schema_version: u32,
    language: Language,
    entries: BTreeMap<String, String>,
}
impl Pronunciations {
    pub fn new(language: Language) -> Result<Self, String> {
        if !language.is_cyrillic() {
            return Err("stress corrections currently support Russian/Ukrainian".into());
        }
        Ok(Self {
            schema_version: 1,
            language,
            entries: BTreeMap::new(),
        })
    }
    pub fn language(&self) -> Language {
        self.language
    }
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }
    pub fn from_json(bytes: &[u8], language: Language) -> Result<Self, String> {
        if bytes.len() > MAX_BYTES {
            return Err("pronunciation dictionary exceeds one MiB".into());
        }
        let raw: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if raw.schema_version != 1 || raw.language != language || !language.is_cyrillic() {
            return Err("pronunciation dictionary schema/language mismatch".into());
        }
        let mut result = Self::new(language)?;
        for (key, value) in raw.entries {
            let canonical = canonical_word(&value, language)?;
            let base = canonical.replace('\u{301}', "");
            if key != base {
                return Err(format!(
                    "correction for {key:?} must preserve its letters and canonical spelling"
                ));
            }
            result.insert(&canonical)?;
        }
        Ok(result)
    }
    /// Insert an accented spelling (acute or + before vowel), preserving letters.
    pub fn insert(&mut self, stressed_word: &str) -> Result<String, String> {
        let value = canonical_word(stressed_word, self.language)?;
        let key = value.replace('\u{301}', "");
        if !self.entries.contains_key(&key) && self.entries.len() >= MAX_ENTRIES {
            return Err("pronunciation dictionary exceeds 10000 words".into());
        }
        self.entries.insert(key.clone(), value);
        Ok(key)
    }
    pub fn remove(&mut self, word: &str) -> Result<(), String> {
        let key =
            crate::frontend::normalize(&apostrophes(word), self.language)?.replace('\u{301}', "");
        self.entries.remove(&key);
        Ok(())
    }
    /// Apply saved stress to whole words; explicit input stress takes precedence.
    /// Does not infer unknown words or disambiguate a word from its context.
    pub fn apply(&self, text: &str) -> Result<String, String> {
        let text = crate::frontend::normalize(&apostrophes(text), self.language)?;
        let mut out = String::new();
        let mut word = String::new();
        let flush = |word: &mut String, out: &mut String| {
            if !word.contains('\u{301}')
                && let Some(value) = self.entries.get(word)
            {
                out.push_str(value);
            } else {
                out.push_str(word);
            }
            word.clear();
        };
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            // Accumulate all letters/digits to avoid matching inside a mixed-language word.
            let internal_apostrophe = c == '\''
                && word
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_alphabetic() || c == '\u{301}')
                && chars.peek().is_some_and(|c| c.is_alphabetic());
            if c.is_alphanumeric() || internal_apostrophe || c == '\u{301}' {
                word.push(c);
            } else {
                flush(&mut word, &mut out);
                out.push(c);
            }
        }
        flush(&mut word, &mut out);
        if out.chars().count() > 10000 {
            return Err("corrected text exceeds 10000 characters".into());
        }
        Ok(out)
    }
}
fn apostrophes(text: &str) -> String {
    text.replace(['’', 'ʼ'], "'")
}
fn canonical_word(text: &str, language: Language) -> Result<String, String> {
    let value = crate::frontend::normalize(&apostrophes(text), language)?;
    let letters = match language {
        Language::Ru => "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        Language::Uk => "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
        _ => return Err("unsupported correction language".into()),
    };
    if value.chars().count() > 96 || value.chars().filter(|c| *c == '\u{301}').count() != 1 {
        return Err("enter one word with exactly one stressed vowel (e.g. прив+іт)".into());
    }
    let base = value.replace('\u{301}', "");
    let chars: Vec<_> = base.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if letters.contains(*c) {
            continue;
        }
        if language == Language::Uk
            && *c == '\''
            && i > 0
            && i + 1 < chars.len()
            && letters.contains(chars[i - 1])
            && letters.contains(chars[i + 1])
        {
            continue;
        }
        return Err("correction must contain only letters of the selected language and internal apostrophes".into());
    }
    Ok(value)
}
pub(crate) fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
        Ok(meta) if !meta.file_type().is_file() => {
            return Err("pronunciation file must be a regular file, not a symlink".into());
        }
        Ok(_) => {}
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BYTES {
        return Err("pronunciation dictionary exceeds one MiB".into());
    }
    Ok(Some(bytes))
}
pub(crate) fn fingerprint(bytes: &Option<Vec<u8>>) -> Option<Vec<u8>> {
    bytes.as_ref().map(|bytes| Sha256::digest(bytes).to_vec())
}
/// Editable file snapshot. Saves are atomic and reject observed external changes.
/// The optimistic check is not a lock against simultaneous external writers.
pub struct PronunciationFile {
    path: PathBuf,
    saved_digest: Option<Vec<u8>>,
    data: Pronunciations,
}
impl PronunciationFile {
    /// A missing file is an empty dictionary; malformed or wrong-language files fail.
    pub fn open(path: &Path, language: Language) -> Result<Self, String> {
        let bytes = read_optional(path)?;
        let data = match &bytes {
            Some(bytes) => Pronunciations::from_json(bytes, language)?,
            None => Pronunciations::new(language)?,
        };
        Ok(Self {
            path: path.into(),
            saved_digest: fingerprint(&bytes),
            data,
        })
    }
    pub fn data(&self) -> &Pronunciations {
        &self.data
    }
    /// Persist a complete validated replacement; failure retains the loaded data.
    pub fn save(&mut self, data: Pronunciations) -> Result<(), String> {
        if data.language() != self.data.language() {
            return Err("correction language cannot change".into());
        }
        let bytes = serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?;
        Pronunciations::from_json(&bytes, self.data.language())?;
        if fingerprint(&read_optional(&self.path)?) != self.saved_digest {
            return Err("pronunciations changed outside Studio; reload before saving".into());
        }
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        temp.write_all(&bytes).map_err(|e| e.to_string())?;
        temp.as_file().sync_all().map_err(|e| e.to_string())?;
        if fingerprint(&read_optional(&self.path)?) != self.saved_digest {
            return Err("pronunciations changed while saving; reload before saving".into());
        }
        if self.saved_digest.is_none() {
            temp.persist_noclobber(&self.path)
                .map_err(|e| e.to_string())?;
        } else {
            temp.persist(&self.path).map_err(|e| e.to_string())?;
        }
        self.saved_digest = Some(Sha256::digest(bytes).to_vec());
        self.data = data;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn word_boundaries_explicit_stress_and_language_validation() {
        let mut ru = Pronunciations::new(Language::Ru).unwrap();
        ru.insert("З+АМОК").unwrap();
        assert_eq!(
            ru.apply("Замок, зам+ок; замоковый замокx!").unwrap(),
            "за́мок, замо́к; замоковый замокx!"
        );
        let mut uk = Pronunciations::new(Language::Uk).unwrap();
        uk.insert("Прив+іт").unwrap();
        uk.insert("п’+ять").unwrap();
        assert_eq!(uk.apply("Привіт! пʼять.").unwrap(), "приві́т! п'я́ть.");
        for form in [
            "привіт",
            "при+віт",
            "прив+іт св+іте",
            "п'ят+ы",
            "hello",
            "+ї́",
            "'п'я́ть",
            "п''я́ть",
            "2+а",
        ] {
            assert!(uk.insert(form).is_err(), "{form}");
        }
        let mut tiny = Pronunciations::new(Language::Ru).unwrap();
        tiny.insert("+а").unwrap();
        assert!(tiny.apply(&"а ".repeat(5000)).is_err());
        assert!(Pronunciations::new(Language::En).is_err());
        assert_eq!(uk.apply("'привіт' 2привіт").unwrap(), "'приві́т' 2привіт");
        uk.remove("ПʼЯТЬ").unwrap();
        assert_eq!(uk.apply("п'ять").unwrap(), "п'ять");
    }
    #[test]
    fn files_are_optional_atomic_and_protect_external_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        let mut file = PronunciationFile::open(&path, Language::Ru).unwrap();
        assert!(!path.exists());
        let mut data = file.data().clone();
        data.insert("вал+ерия").unwrap();
        file.save(data).unwrap();
        assert_eq!(
            PronunciationFile::open(&path, Language::Ru)
                .unwrap()
                .data()
                .apply("Валерия")
                .unwrap(),
            "вале́рия"
        );
        assert!(PronunciationFile::open(&path, Language::Uk).is_err());
        let mut pending = file.data().clone();
        pending.insert("з+амок").unwrap();
        fs::write(&path, b"external edits").unwrap();
        assert!(file.save(pending).unwrap_err().contains("outside"));
        assert_eq!(fs::read(&path).unwrap(), b"external edits");
        assert_eq!(file.data().entries().count(), 1);
        let bad = br#"{"schema_version":1,"language":"ru","entries":{"name":"different"}}"#;
        assert!(Pronunciations::from_json(bad, Language::Ru).is_err());
        assert!(Pronunciations::from_json(&vec![b' '; MAX_BYTES + 1], Language::Ru).is_err());
    }
    #[test]
    #[cfg(unix)]
    fn symlink_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("other"), b"{}").unwrap();
        std::os::unix::fs::symlink(dir.path().join("other"), dir.path().join(FILE_NAME)).unwrap();
        assert!(PronunciationFile::open(&dir.path().join(FILE_NAME), Language::Uk).is_err());
    }
}
