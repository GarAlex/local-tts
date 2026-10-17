//! Explicit, user-authored European spelling hints selected by adjacent words.
//! This module does not expand numbers or infer general grammatical context.
use crate::Language;
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
use unicode_normalization::char::is_combining_mark;

const MAX_BYTES: usize = 65536;
const MAX_RULES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub word: String,
    pub spoken: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRules {
    schema_version: u32,
    language: Language,
    rules: Vec<Rule>,
}
#[derive(Debug, Serialize)]
pub struct Change {
    pub word: String,
    pub spoken: String,
}
#[derive(Debug, Serialize)]
pub struct Prepared {
    pub text: String,
    pub changes: Vec<Change>,
}

impl ContextRules {
    pub fn new(language: Language) -> Result<Self, String> {
        if !language.is_european_extension() {
            return Err("context spelling rules require de/fr/es/it/pt/pl/nl".into());
        }
        Ok(Self {
            schema_version: 1,
            language,
            rules: Vec::new(),
        })
    }
    pub fn language(&self) -> Language {
        self.language
    }
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    pub fn from_json(bytes: &[u8], language: Language) -> Result<Self, String> {
        if bytes.len() > MAX_BYTES {
            return Err("context rules exceed 64 KiB".into());
        }
        let raw: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if raw.schema_version != 1 || raw.language != language || raw.rules.len() > MAX_RULES {
            return Err("context rules schema/language/count mismatch".into());
        }
        let mut result = Self::new(language)?;
        for rule in raw.rules {
            result.insert(rule)?;
        }
        Ok(result)
    }
    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_BYTES {
            return Err("context rules exceed 64 KiB".into());
        }
        Ok(bytes)
    }
    /// Insert a unique context. Conflicting duplicates fail without changing it.
    pub fn insert(&mut self, mut rule: Rule) -> Result<(), String> {
        rule.word = word(&rule.word, self.language)?;
        rule.spoken = word(&rule.spoken, self.language)?;
        rule.before = rule.before.map(|v| word(&v, self.language)).transpose()?;
        rule.after = rule.after.map(|v| word(&v, self.language)).transpose()?;
        if rule.before.is_none() && rule.after.is_none() {
            return Err("provide at least one adjacent context word".into());
        }
        if rule.word == rule.spoken {
            return Err("the spoken spelling must differ".into());
        }
        if let Some(existing) = self
            .rules
            .iter()
            .find(|r| r.word == rule.word && r.before == rule.before && r.after == rule.after)
        {
            return if *existing == rule {
                Ok(())
            } else {
                Err("a different spelling already uses this context; remove it first".into())
            };
        }
        if self.rules.len() >= MAX_RULES {
            return Err("context rules exceed 256 entries".into());
        }
        self.rules.push(rule);
        if let Err(error) = self.to_json() {
            self.rules.pop();
            return Err(error);
        }
        Ok(())
    }
    pub fn remove(&mut self, index: usize) -> Result<(), String> {
        if index >= self.rules.len() {
            return Err("context rule index is out of range".into());
        }
        self.rules.remove(index);
        Ok(())
    }
    /// Normalize first, then make one simultaneous pass over original word spans.
    /// Two cues outrank one; equally specific conflicting matches return an error.
    pub fn prepare(&self, input: &str) -> Result<Prepared, String> {
        let text = crate::text::prepare(input, self.language)?;
        let spans = spans(&text);
        let mut output = String::new();
        let mut changes = Vec::new();
        let mut end = 0;
        for (i, &(start, stop)) in spans.iter().enumerate() {
            output.push_str(&text[end..start]);
            let current = &text[start..stop];
            let before = i.checked_sub(1).and_then(|j| {
                let (a, b) = spans[j];
                text[b..start]
                    .chars()
                    .all(char::is_whitespace)
                    .then_some(&text[a..b])
            });
            let after = spans.get(i + 1).and_then(|&(a, b)| {
                text[stop..a]
                    .chars()
                    .all(char::is_whitespace)
                    .then_some(&text[a..b])
            });
            let mut best = 0;
            let mut selected: Option<&str> = None;
            let mut ambiguous = false;
            for rule in &self.rules {
                if rule.word != current
                    || rule.before.as_deref().is_some_and(|v| Some(v) != before)
                    || rule.after.as_deref().is_some_and(|v| Some(v) != after)
                {
                    continue;
                }
                let score = usize::from(rule.before.is_some()) + usize::from(rule.after.is_some());
                if score > best {
                    best = score;
                    selected = Some(&rule.spoken);
                    ambiguous = false;
                } else if score == best && selected != Some(&rule.spoken) {
                    ambiguous = true;
                }
            }
            if ambiguous {
                return Err(format!(
                    "conflicting context rules for {current:?}; add both adjacent cues or remove the conflict"
                ));
            }
            if let Some(spoken) = selected {
                output.push_str(spoken);
                changes.push(Change {
                    word: current.into(),
                    spoken: spoken.into(),
                });
            } else {
                output.push_str(current);
            }
            end = stop;
        }
        output.push_str(&text[end..]);
        if output.chars().count() > 10000 {
            return Err("contextual text exceeds 10000 characters".into());
        }
        Ok(Prepared {
            text: output,
            changes,
        })
    }
}
fn word(input: &str, language: Language) -> Result<String, String> {
    let value = crate::frontend::normalize(input, language)?;
    let chars: Vec<_> = value.chars().collect();
    if chars.is_empty() || chars.len() > 96 || !chars[0].is_alphabetic() {
        return Err("enter one word of at most 96 characters".into());
    }
    for (i, c) in chars.iter().enumerate() {
        if c.is_alphabetic() || is_combining_mark(*c) {
            continue;
        }
        if matches!(*c, '\'' | '’' | '-')
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_alphabetic()
            && chars[i + 1].is_alphabetic()
        {
            continue;
        }
        return Err(
            "context words and spellings accept letters and internal apostrophes/hyphens only"
                .into(),
        );
    }
    Ok(value)
}
fn spans(text: &str) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut start = None;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let internal = matches!(c, '\'' | '’' | '-')
            && start.is_some()
            && chars.peek().is_some_and(|(_, next)| next.is_alphabetic());
        if c.is_alphanumeric() || c == '_' || is_combining_mark(c) || internal {
            start.get_or_insert(i);
        } else if let Some(begin) = start.take() {
            result.push((begin, i));
        }
    }
    if let Some(begin) = start {
        result.push((begin, text.len()));
    }
    result
}

/// Editable local snapshot with atomic writes and external-change detection.
pub struct ContextFile {
    path: PathBuf,
    saved_digest: Option<Vec<u8>>,
    data: ContextRules,
}
impl ContextFile {
    pub fn open(path: &Path, language: Language) -> Result<Self, String> {
        let bytes = crate::pronunciations::read_optional(path)?;
        let data = match &bytes {
            Some(b) => ContextRules::from_json(b, language)?,
            None => ContextRules::new(language)?,
        };
        Ok(Self {
            path: path.into(),
            saved_digest: crate::pronunciations::fingerprint(&bytes),
            data,
        })
    }
    pub fn data(&self) -> &ContextRules {
        &self.data
    }
    pub fn save(&mut self, data: ContextRules) -> Result<(), String> {
        if data.language != self.data.language {
            return Err("context language cannot change".into());
        }
        let bytes = data.to_json()?;
        ContextRules::from_json(&bytes, self.data.language)?;
        let check = || -> Result<(), String> {
            if crate::pronunciations::fingerprint(&crate::pronunciations::read_optional(
                &self.path,
            )?) != self.saved_digest
            {
                return Err("context rules changed outside Studio; reload before saving".into());
            }
            Ok(())
        };
        check()?;
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        temp.write_all(&bytes).map_err(|e| e.to_string())?;
        temp.as_file().sync_all().map_err(|e| e.to_string())?;
        check()?;
        if self.saved_digest.is_none() {
            temp.persist_noclobber(&self.path)
                .map_err(|e| e.to_string())?;
        } else {
            temp.persist(&self.path).map_err(|e| e.to_string())?;
        }
        self.saved_digest = crate::pronunciations::fingerprint(&Some(bytes));
        self.data = data;
        Ok(())
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}
