//! Training preparation uses native tokenization shared with future VITS inference.
use crate::dataset::{self, Split};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::Path};

#[derive(Serialize, Deserialize)]
pub struct TrainingConfig {
    pub schema_version: u32,
    pub language: crate::Language,
    #[serde(default = "legacy_frontend")]
    pub frontend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<DictionaryConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditioning_speaker: Option<usize>,
    pub sample_rate: u32,
    pub symbols: Vec<char>,
    pub speaker: String,
    pub utterances: Vec<TrainingUtterance>,
    pub rights: dataset::Rights,
}
pub const IPA_FRONTEND: &str = "cmu-ipa-v1";
#[derive(Clone, Serialize, Deserialize)]
pub struct DictionaryConfig {
    pub path: String,
    pub sha256: String,
}
pub fn load_dictionary(config: &DictionaryConfig) -> Result<crate::english::English, String> {
    let hash = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&config.path).map_err(|e| e.to_string())?)
    );
    if hash != config.sha256 {
        return Err("frontend dictionary checksum mismatch".into());
    }
    crate::english::English::load(Path::new(&config.path))
}
/// Upstream VITS aliases apostrophe twice; its dictionary keeps the last index.
pub fn validate_symbols(symbols: &[char], frontend: &str) -> Result<(), String> {
    if frontend == crate::css10::VERSION && symbols != crate::css10::symbols() {
        return Err("CSS10 inventory differs from pinned indices".into());
    }
    if frontend == crate::neon::VERSION
        && symbols != crate::neon::symbols(crate::Language::It)?
        && symbols != crate::neon::symbols(crate::Language::Fr)?
    {
        return Err("Neon inventory differs from pinned indices".into());
    }
    if frontend == crate::piper_training::VERSION && symbols != crate::piper_training::symbols() {
        return Err("Piper training inventory differs from pinned indices".into());
    }
    if frontend == crate::vosk::TRAINING_VERSION && symbols != crate::vosk::training_symbols() {
        return Err("Russian training inventory differs from pinned indices".into());
    }
    let mut seen = BTreeSet::new();
    if symbols.first() != Some(&'\0') || symbols.len() < 2 {
        return Err("blank token must be at index zero".into());
    }
    let mut aliases = 0;
    for &c in symbols {
        if !seen.insert(c) {
            if frontend == IPA_FRONTEND && c == '\'' && aliases == 0 {
                aliases += 1;
            } else {
                return Err("invalid duplicate symbol".into());
            }
        }
    }
    Ok(())
}
fn legacy_frontend() -> String {
    "grapheme-v1".into()
}
#[derive(Serialize, Deserialize)]
pub struct TrainingUtterance {
    pub audio: String,
    pub sha256: String,
    pub tokens: Vec<i64>,
    pub split: Split,
}

pub fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn prepare(manifest: &Path, output: &Path) -> Result<(), String> {
    prepare_impl(manifest, output, None, true)
}

/// Preserve a base model's complete token inventory when adapting to a new voice.
pub fn prepare_from_base(
    manifest: &Path,
    output: &Path,
    base: &TrainingConfig,
) -> Result<(), String> {
    prepare_impl(manifest, output, Some(base), false)
}

/// Start a new target with text-v1 while retaining the base token inventory.
/// Use prepare_from_base for resume/evaluation, which preserves its exact contract.
pub fn prepare_from_base_current(
    manifest: &Path,
    output: &Path,
    base: &TrainingConfig,
) -> Result<(), String> {
    prepare_impl(manifest, output, Some(base), true)
}

fn prepare_impl(
    manifest: &Path,
    output: &Path,
    base: Option<&TrainingConfig>,
    current: bool,
) -> Result<(), String> {
    let data = dataset::read(manifest).map_err(|e| e.to_string())?;
    let root = manifest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let report = dataset::validate(&data, root);
    if !report.errors.is_empty() {
        return Err(report.errors.join("\n"));
    }
    if data.sample_rate != 22050 {
        return Err(
            "VITS training currently requires 22050 Hz mono WAV; resample recordings first".into(),
        );
    }
    let speakers: BTreeSet<_> = data.utterances.iter().map(|u| u.speaker.clone()).collect();
    if speakers.len() != 1 {
        return Err("initial VITS trainer supports exactly one speaker per voice".into());
    }
    if !data.utterances.iter().any(|u| u.split == Split::Train)
        || !data.utterances.iter().any(|u| u.split == Split::Validation)
    {
        return Err("training requires both train and validation recordings".into());
    }
    let frontend = base.map_or(crate::frontend::VERSION, |b| b.frontend.as_str());
    crate::text::validate_version(base.and_then(|b| b.normalization.as_deref()))?;
    let normalization = if current {
        Some(crate::text::VERSION.to_string())
    } else {
        base.and_then(|b| b.normalization.clone())
    };
    if frontend == "grapheme-v1" && normalization.is_some() {
        return Err("legacy grapheme frontend cannot use text-v1 normalization".into());
    }
    if !matches!(
        frontend,
        "grapheme-v1"
            | "grapheme-v2"
            | IPA_FRONTEND
            | crate::piper_training::VERSION
            | crate::vosk::TRAINING_VERSION
            | crate::css10::VERSION
            | crate::neon::VERSION
    ) {
        return Err("unsupported base frontend".into());
    }
    if frontend == crate::neon::VERSION {
        let expected = crate::neon::symbols(data.language)?;
        if base.is_none_or(|b| b.conditioning_speaker != Some(0) || b.symbols != expected) {
            return Err("invalid Neon inventory/conditioning speaker".into());
        }
    }
    if frontend == crate::css10::VERSION
        && (!matches!(data.language, crate::Language::De | crate::Language::Es)
            || base.and_then(|b| b.conditioning_speaker) != Some(0))
    {
        return Err("invalid CSS10 language/conditioning speaker".into());
    }
    if frontend == crate::piper_training::VERSION
        && (data.language != crate::Language::Uk
            || base
                .and_then(|b| b.conditioning_speaker)
                .is_none_or(|sid| sid >= 3))
    {
        return Err("invalid Ukrainian conditioning speaker".into());
    }
    let russian = if frontend == crate::vosk::TRAINING_VERSION {
        if data.language != crate::Language::Ru
            || base
                .and_then(|b| b.conditioning_speaker)
                .is_none_or(|sid| sid >= 5)
        {
            return Err("invalid Russian conditioning speaker/language".into());
        }
        let dictionary = base
            .and_then(|b| b.dictionary.as_ref())
            .ok_or("missing Russian dictionary")?;
        let actual = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&dictionary.path).map_err(|e| e.to_string())?)
        );
        if actual != dictionary.sha256 {
            return Err("frontend dictionary checksum mismatch".into());
        }
        Some(crate::vosk::Frontend::load(Path::new(&dictionary.path)).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let dictionary = if frontend == IPA_FRONTEND {
        if data.language != crate::Language::En {
            return Err("CMU IPA requires English".into());
        }
        Some(load_dictionary(
            base.and_then(|b| b.dictionary.as_ref())
                .ok_or("missing base dictionary")?,
        )?)
    } else {
        None
    };
    let normalized: Vec<_> = data
        .utterances
        .iter()
        .map(|u| {
            let text = if normalization.is_none() {
                u.text.clone()
            } else {
                crate::text::prepare(&u.text, data.language)
                    .map_err(|e| format!("{}: {e}", u.id))?
            };
            if let Some(russian) = &russian {
                russian
                    .normalize_training(&text)
                    .map_err(|e| format!("{}: {e}", u.id))
            } else if let Some(dictionary) = &dictionary {
                (if normalization.is_some() {
                    dictionary.phonemize_ipa_current(&text)
                } else {
                    dictionary.phonemize_ipa(&text)
                })
                .map_err(|e| format!("{}: {e}", u.id))
            } else if frontend == crate::piper_training::VERSION {
                crate::piper_training::normalize(&text, data.language)
                    .map_err(|e| format!("{}: {e}", u.id))
            } else if frontend == crate::neon::VERSION {
                crate::neon::normalize(&text, data.language).map_err(|e| format!("{}: {e}", u.id))
            } else if frontend == crate::css10::VERSION {
                crate::css10::normalize(&text, data.language).map_err(|e| format!("{}: {e}", u.id))
            } else if frontend == "grapheme-v1" {
                Ok(normalize(&text))
            } else {
                crate::frontend::normalize(&text, data.language)
                    .map_err(|e| format!("{}: {e}", u.id))
            }
        })
        .collect::<Result<_, _>>()?;
    let inventory: BTreeSet<char> = normalized.iter().flat_map(|s| s.chars()).collect();
    if inventory.contains(&'\0') {
        return Err("NUL is reserved for blank tokens".into());
    }
    let symbols = if let Some(base) = base {
        if base.schema_version != 1
            || base.language != data.language
            || base.sample_rate != data.sample_rate
        {
            return Err("base language/frontend/sample rate differs from target dataset".into());
        }
        validate_symbols(&base.symbols, frontend)?;
        if inventory.iter().any(|c| !base.symbols.contains(c)) {
            return Err(
                "target transcript contains symbols absent from base token inventory".into(),
            );
        }
        base.symbols.clone()
    } else {
        let mut symbols = vec!['\0'];
        symbols.extend(inventory);
        symbols
    };
    let mut utterances = Vec::new();
    let mut hashes = BTreeSet::new();
    for (u, text) in data.utterances.iter().zip(normalized) {
        let mut tokens = vec![0];
        for c in text.chars() {
            tokens.push(symbols.iter().rposition(|s| *s == c).unwrap() as i64);
            tokens.push(0);
        }
        if frontend == crate::piper_training::VERSION {
            tokens = crate::piper_training::encode(&text)?;
        }
        if frontend == crate::vosk::TRAINING_VERSION {
            tokens = crate::vosk::encode_training(&text)?;
        }
        if frontend == crate::neon::VERSION {
            tokens = crate::neon::encode(&text, data.language)?;
        } else if frontend == crate::css10::VERSION {
            tokens = crate::css10::encode(&text)?;
        }
        let path = root
            .join(&u.audio)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let reader = hound::WavReader::open(&path).map_err(|e| e.to_string())?;
        if reader.duration() < 8192 || reader.duration() > 22050 * 20 {
            return Err(format!(
                "{}: training requires clips between 8192 samples and twenty seconds",
                u.id
            ));
        }
        let hash = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&path).map_err(|e| e.to_string())?)
        );
        if !hashes.insert(hash.clone()) {
            return Err(format!(
                "{}: identical audio content at different paths",
                u.id
            ));
        }
        utterances.push(TrainingUtterance {
            audio: path.to_string_lossy().into_owned(),
            sha256: hash,
            tokens,
            split: u.split,
        });
    }
    let config = TrainingConfig {
        schema_version: 1,
        language: data.language,
        frontend: frontend.into(),
        normalization,
        dictionary: base.and_then(|b| b.dictionary.clone()),
        conditioning_speaker: base.and_then(|b| b.conditioning_speaker),
        sample_rate: data.sample_rate,
        symbols,
        speaker: speakers.into_iter().next().unwrap(),
        utterances,
        rights: data.rights,
    };
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    fn adaptation_preserves_base_indices_and_legacy_normalization() {
        let folder = tempfile::tempdir().unwrap();
        for (name, frequency) in [("train", 220.0), ("validation", 330.0)] {
            let mut wav = hound::WavWriter::create(
                folder.path().join(format!("{name}.wav")),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 22050,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for i in 0..22050 {
                wav.write_sample(
                    (8000.0 * (i as f64 * frequency * std::f64::consts::TAU / 22050.0).sin())
                        as i16,
                )
                .unwrap();
            }
            wav.finalize().unwrap();
        }
        let rights = serde_json::json!({"audio":"synthetic", "transcripts":"test", "speaker_consent":"no human"});
        let dataset = serde_json::json!({"schema_version":1,"language":"en","sample_rate":22050,"rights":rights,
            "utterances":[{"id":"train","audio":"train.wav","text":"e\u{301}","speaker":"target","split":"train"},
            {"id":"validation","audio":"validation.wav","text":"b","speaker":"target","split":"validation"}]});
        let manifest = folder.path().join("dataset.json");
        std::fs::write(&manifest, serde_json::to_vec(&dataset).unwrap()).unwrap();
        let mut base: super::TrainingConfig = serde_json::from_value(serde_json::json!({"schema_version":1,
            "language":"en","sample_rate":22050,"symbols":["\0","b","a","e","\u{301}"],"speaker":"base","utterances":[],"rights":rights})).unwrap();
        let output = folder.path().join("prepared.json");
        super::prepare_from_base(&manifest, &output, &base).unwrap();
        let prepared: super::TrainingConfig =
            serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(prepared.frontend, "grapheme-v1");
        assert_eq!(prepared.symbols, base.symbols);
        assert_eq!(prepared.utterances[0].tokens, [0, 3, 0, 4, 0]);
        assert_eq!(prepared.speaker, "target");
        assert!(prepared.normalization.is_none());
        base.frontend = crate::frontend::VERSION.into();
        assert!(
            super::prepare_from_base(&manifest, &output, &base)
                .unwrap_err()
                .contains("absent")
        );
        base.symbols.push('é');
        super::prepare_from_base(&manifest, &output, &base).unwrap();
        let prepared: super::TrainingConfig =
            serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(prepared.utterances[0].tokens, [0, 5, 0]);
        assert!(prepared.normalization.is_none());
        let legacy_bytes = std::fs::read(&output).unwrap();
        super::prepare_from_base(&manifest, &output, &prepared).unwrap();
        assert_eq!(legacy_bytes, std::fs::read(&output).unwrap());
        super::prepare_from_base_current(&manifest, &output, &base).unwrap();
        let current: super::TrainingConfig =
            serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(current.normalization.as_deref(), Some(crate::text::VERSION));
        let current_bytes = std::fs::read(&output).unwrap();
        super::prepare_from_base(&manifest, &output, &current).unwrap();
        assert_eq!(current_bytes, std::fs::read(&output).unwrap());
        base.language = crate::Language::Ru;
        assert!(
            super::prepare_from_base(&manifest, &output, &base)
                .unwrap_err()
                .contains("language")
        );
    }
    #[test]
    fn upstream_alias_and_dictionary_integrity_are_explicit() {
        let symbols = ['\0', '\'', 'a', '\''];
        super::validate_symbols(&symbols, super::IPA_FRONTEND).unwrap();
        assert!(super::validate_symbols(&symbols, "grapheme-v2").is_err());
        assert!(super::validate_symbols(&['\0', 'a', 'a'], super::IPA_FRONTEND).is_err());
        assert!(super::validate_symbols(&['\0', '\'', '\'', '\''], super::IPA_FRONTEND).is_err());
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("cmudict.dict");
        std::fs::write(&path, "hello HH AH0 L OW1\n").unwrap();
        use sha2::Digest;
        let config = super::DictionaryConfig {
            path: path.display().to_string(),
            sha256: format!("{:x}", sha2::Sha256::digest(std::fs::read(&path).unwrap())),
        };
        assert_eq!(
            super::load_dictionary(&config)
                .unwrap()
                .phonemize_ipa("Hello.")
                .unwrap(),
            "həlˈoʊ."
        );
        std::fs::write(&path, "hello HH EH1 L OW0\n").unwrap();
        assert!(
            super::load_dictionary(&config)
                .err()
                .unwrap()
                .contains("checksum mismatch")
        );
    }
    #[test]
    fn preserves_cyrillic() {
        assert_eq!(super::normalize("  Їжак  И Ё "), "їжак и ё");
    }
}

/// Keep a byte-identical evaluated checkpoint; do not replace existing files.
/// The SHA-256 binds this copy to the evaluation, even if latest.pt has advanced.
pub fn snapshot_checkpoint(
    source: &Path,
    destination: &Path,
    expected: &str,
) -> Result<(), String> {
    use std::io::{Read, Write};
    if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid checkpoint checksum".into());
    }
    let mut input = std::fs::File::open(source).map_err(|e| e.to_string())?;
    let metadata = input.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 2_000_000_000 {
        return Err("invalid or oversized checkpoint".into());
    }
    if destination.exists() {
        return Err("saved checkpoint already exists".into());
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temporary = parent.join(format!(".checkpoint-{}-{stamp}.part", std::process::id()));
    let mut created = false;
    let result = (|| {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        created = true;
        let mut total = 0_u64;
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            total += n as u64;
            if total > 2_000_000_000 {
                return Err("checkpoint exceeds two gigabytes".into());
            }
            hash.update(&buffer[..n]);
            output.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
        }
        if format!("{:x}", hash.finalize()) != expected.to_lowercase() {
            return Err(
                "checkpoint changed since evaluation; compare again before keeping it".into(),
            );
        }
        output.sync_all().map_err(|e| e.to_string())?;
        drop(output);
        // Exclusive install; no overwrite even if a destination appeared during the copy.
        std::fs::hard_link(&temporary, destination).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if created {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    #[test]
    fn saved_checkpoint_is_independent_and_bound_to_evaluated_bytes() {
        let folder = tempfile::tempdir().unwrap();
        let source = folder.path().join("latest.pt");
        std::fs::write(&source, b"evaluated checkpoint").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"evaluated checkpoint"));
        let target = folder.path().join("saved/checkpoint.pt");
        snapshot_checkpoint(&source, &target, &hash).unwrap();
        std::fs::write(&source, b"next checkpoint").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"evaluated checkpoint");
        let changed = folder.path().join("saved/changed.pt");
        assert!(
            snapshot_checkpoint(&source, &changed, &hash)
                .unwrap_err()
                .contains("changed since evaluation")
        );
        assert!(!changed.exists());
        assert!(snapshot_checkpoint(&source, &target, &hash).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"evaluated checkpoint");
        assert_eq!(
            std::fs::read_dir(target.parent().unwrap()).unwrap().count(),
            1
        );
    }
}
