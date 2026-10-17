//! Native Vosk 0.7 Russian frontend and ONNX inference, without Python at runtime.
//! G2P rules adapted from alphacep/vosk-tts (Apache-2.0); see third-party notices.
use crate::{Audio, Language, SynthesisRequest, Synthesizer, kokoro::Error};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

pub const SPEAKERS: [&str; 5] = ["F01", "F02", "F03", "M01", "M02"];
#[derive(Deserialize)]
struct Config {
    phoneme_id_map: BTreeMap<String, Vec<i64>>,
    inference: Inference,
}
#[derive(Deserialize)]
struct Inference {
    noise_level: f32,
    duration_noise_level: f32,
}
pub struct Vosk {
    frontend: Frontend,
    session: Session,
    config: Config,
}
/// Model-independent Russian phoneme frontend shared by preparation and inference.
pub struct Frontend {
    pronunciations: crate::pronunciations::Pronunciations,
    pronunciation_path: PathBuf,
    config: Config,
    dictionary: HashMap<String, (f32, Vec<i64>)>,
}
impl Frontend {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let root = path.parent().ok_or("missing dictionary directory")?;
            let actual = crate::integrity::hash_file(
                root,
                path.file_name()
                    .ok_or("missing dictionary name")?
                    .to_str()
                    .ok_or("invalid dictionary name")?,
            )
            .map_err(std::io::Error::other)?;
            let assets: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/russian-assets.json"))?;
            if actual
                != assets["files"]["dictionary"]["sha256"]
                    .as_str()
                    .ok_or("missing dictionary pin")?
            {
                return Err("Russian dictionary checksum mismatch".into());
            }
            let config: Config =
                serde_json::from_str(include_str!("../../../assets/vosk-ru-config.json"))?;
            let mut dictionary = HashMap::<String, (f32, Vec<i64>)>::new();
            for line in BufReader::new(std::fs::File::open(path)?).lines() {
                let line = line?;
                let mut items = line.split_whitespace();
                let word = items.next().ok_or("missing dictionary word")?;
                if word == "!SIL" || word == "[unk]" {
                    continue;
                }
                let probability: f32 = items
                    .next()
                    .ok_or("missing pronunciation probability")?
                    .parse()?;
                if !probability.is_finite() || probability <= 0.0 {
                    continue;
                }
                if dictionary
                    .get(word)
                    .is_some_and(|(old, _)| *old >= probability)
                {
                    continue;
                }
                let mut ids = Vec::new();
                for phone in items {
                    ids.extend(
                        config
                            .phoneme_id_map
                            .get(phone)
                            .ok_or("invalid dictionary phoneme")?,
                    );
                }
                if ids.is_empty() {
                    return Err("empty dictionary pronunciation".into());
                }
                dictionary.insert(word.into(), (probability, ids));
            }

            let pronunciation_path = root.join(crate::pronunciations::FILE_NAME);
            let pronunciations = crate::pronunciations::Pronunciations::new(Language::Ru)
                .map_err(std::io::Error::other)?;
            Ok(Self {
                pronunciations,
                pronunciation_path,
                config,
                dictionary,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    /// Reload user corrections without reloading the neural model. Failed reloads retain old corrections.
    pub fn reload_pronunciations(&mut self) -> Result<(), Error> {
        let file =
            crate::pronunciations::PronunciationFile::open(&self.pronunciation_path, Language::Ru)
                .map_err(Error)?;
        self.set_pronunciations(file.data().clone())
    }
    /// Install validated in-memory corrections; does not change the saved file.
    pub fn set_pronunciations(
        &mut self,
        data: crate::pronunciations::Pronunciations,
    ) -> Result<(), Error> {
        if data.language() != Language::Ru {
            return Err(Error("pronunciation language mismatch".into()));
        }
        self.pronunciations = data;
        Ok(())
    }
    /// Return expanded text with saved corrections; explicit stress takes precedence.
    pub fn prepare_text(&self, text: &str) -> Result<String, Error> {
        let expanded = crate::text::expand(text, Language::Ru).map_err(Error)?;
        let corrected = self.pronunciations.apply(&expanded).map_err(Error)?;
        crate::context::stress(&corrected, Language::Ru).map_err(Error)
    }
    pub fn tokenize(&self, text: &str) -> Result<Vec<i64>, Error> {
        let ids = self.tokenize_unbounded(text)?;
        if ids.len() > 1024 {
            return Err(Error("Russian chunk exceeds 1024 tokens".into()));
        }
        Ok(ids)
    }
    fn tokenize_unbounded(&self, text: &str) -> Result<Vec<i64>, Error> {
        let text = self.prepare_text(text)?;
        self.tokenize_prepared(&text)
    }
    fn tokenize_prepared(&self, text: &str) -> Result<Vec<i64>, Error> {
        let map = &self.config.phoneme_id_map;
        let mut phones = map["^"].clone();
        let mut word = String::new();
        let flush = |word: &mut String, phones: &mut Vec<i64>| -> Result<(), Error> {
            if word.is_empty() {
                return Ok(());
            }
            if !word.contains('\u{301}') {
                if let Some((_, ids)) = self.dictionary.get(word) {
                    phones.extend(ids);
                    word.clear();
                    return Ok(());
                }
                return Err(Error(format!(
                    "unknown Russian word {word:?}; add explicit stress (+ before the vowel)"
                )));
            }
            for phone in stressed_g2p(word).map_err(Error)? {
                phones.extend(
                    map.get(&phone)
                        .ok_or_else(|| Error(format!("unsupported Russian phoneme {phone}")))?,
                );
            }
            word.clear();
            Ok(())
        };
        for c in text.chars() {
            if ('а'..='я').contains(&c) || c == 'ё' || c == '\u{301}' {
                word.push(c);
            } else {
                flush(&mut word, &mut phones)?;
                if !" ,.!?;:\"()-'".contains(c) {
                    return Err(Error(format!(
                        "unsupported Russian character {c:?}; spell out dates and abbreviations"
                    )));
                }
                // The 0.7 model has no quote token; omit quotation marks only.
                if c != '"' {
                    phones.extend(
                        map.get(&c.to_string())
                            .ok_or_else(|| Error("unsupported punctuation".into()))?,
                    );
                }
            }
        }
        flush(&mut word, &mut phones)?;
        phones.extend(&map["$"]);
        if phones.len() <= 2 {
            return Err(Error("empty Russian input".into()));
        }
        let mut ids = Vec::with_capacity(phones.len() * 2 - 1);
        for phone in phones {
            if !ids.is_empty() {
                ids.push(0);
            }
            ids.push(phone);
        }
        Ok(ids)
    }

    /// Encode source phone IDs as internal scalar symbols for the training contract.
    pub fn normalize_training(&self, text: &str) -> Result<String, Error> {
        // v1 packages retain their original spelling/number contract. New training
        // configurations explicitly apply text-v1 before this token frontend.
        let expanded = crate::cyrillic_numbers::expand(text, Language::Ru).map_err(Error)?;
        let legacy =
            crate::frontend::normalize(&expanded.replace('—', "-"), Language::Ru).map_err(Error)?;
        let corrected = self.pronunciations.apply(&legacy).map_err(Error)?;
        let tokens = self.tokenize_prepared(&corrected)?;
        Ok(tokens[2..tokens.len() - 2]
            .iter()
            .step_by(2)
            .map(|id| training_symbols()[*id as usize])
            .collect())
    }
}
pub const TRAINING_VERSION: &str = "vosk-ru-phones-v1";
pub fn training_symbols() -> Vec<char> {
    (0..62)
        .map(|i| match i {
            0 => '\0',
            3 => ' ',
            _ => char::from_u32(0xe000 + i).unwrap(),
        })
        .collect()
}
pub fn encode_training(text: &str) -> Result<Vec<i64>, String> {
    let symbols = training_symbols();
    if text.trim().is_empty() {
        return Err("empty Russian input".into());
    }
    let mut tokens = vec![1, 0];
    for c in text.chars() {
        let id = symbols
            .iter()
            .position(|s| *s == c)
            .filter(|id| *id >= 3)
            .ok_or("invalid internal Russian phoneme")?;
        tokens.extend([id as i64, 0]);
    }
    tokens.push(2);
    if tokens.len() > 1024 {
        return Err("Russian chunk exceeds 1024 tokens".into());
    }
    Ok(tokens)
}
impl Vosk {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let assets: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/russian-assets.json"))?;
            let hashes = ["model.onnx", "config.json", "dictionary"]
                .into_iter()
                .map(|name| {
                    Ok((
                        name.into(),
                        assets["files"][name]["sha256"]
                            .as_str()
                            .ok_or("missing pinned checksum")?
                            .to_string(),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
            crate::integrity::verify(root, &hashes, &["model.onnx", "config.json", "dictionary"])
                .map_err(std::io::Error::other)?;
            let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
            let mut frontend = Frontend::load(&root.join("dictionary"))?;
            frontend.reload_pronunciations()?;
            Ok(Self {
                frontend,
                session: Session::builder()?
                    .with_intra_threads(2)?
                    .commit_from_file(root.join("model.onnx"))?,
                config,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    pub fn reload_pronunciations(&mut self) -> Result<(), Error> {
        self.frontend.reload_pronunciations()
    }
    pub fn set_pronunciations(
        &mut self,
        data: crate::pronunciations::Pronunciations,
    ) -> Result<(), Error> {
        self.frontend.set_pronunciations(data)
    }
    pub fn prepare_text(&self, text: &str) -> Result<String, Error> {
        self.frontend.prepare_text(text)
    }
    pub fn tokenize(&self, text: &str) -> Result<Vec<i64>, Error> {
        self.frontend.tokenize(text)
    }
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &AtomicBool,
    ) -> Result<Audio, Error> {
        if request.language != Language::Ru {
            return Err(Error("this Vosk voice supports Russian only".into()));
        }
        let sid = SPEAKERS
            .iter()
            .position(|s| *s == request.speaker)
            .ok_or_else(|| Error("unknown Russian speaker; use F01/F02/F03/M01/M02".into()))?
            as i64;
        if !request.duration_scale.is_finite() || !(0.5..=2.0).contains(&request.duration_scale) {
            return Err(Error("duration scale must be 0.5–2.0".into()));
        }
        let text = self.prepare_text(request.text)?;
        let parts = crate::paragraph::chunks(&text, 180).map_err(Error)?;
        let chunks = parts
            .iter()
            .map(|part| self.tokenize(part))
            .collect::<Result<Vec<_>, _>>()?;
        let mut audio = None;
        for ids in chunks {
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
            let mut run = || -> Result<Audio, Box<dyn std::error::Error>> {
                let outputs = self.session.run(ort::inputs![
                    "input" => Tensor::from_array(Array2::from_shape_vec((1, ids.len()), ids.clone())?)?,
                    "input_lengths" => Tensor::from_array(([1usize], vec![ids.len() as i64]))?,
                    "scales" => Tensor::from_array(([3usize], vec![self.config.inference.noise_level, request.duration_scale, self.config.inference.duration_noise_level]))?,
                    "sid" => Tensor::from_array(([1usize], vec![sid]))?
                ])?;
                let (_, samples) = outputs[0].try_extract_tensor::<f32>()?;
                Ok(Audio {
                    sample_rate: 22050,
                    samples: samples.to_vec(),
                })
            };
            crate::paragraph::append(&mut audio, run().map_err(|e| Error(e.to_string()))?)
                .map_err(Error)?;
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
        }
        audio.ok_or_else(|| Error("empty input".into()))
    }
}
impl Synthesizer for Vosk {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Error> {
        self.synthesize_cancellable(request, &AtomicBool::new(false))
    }
}

/// Current number/date/unit normalization with limited contextual stress.
pub fn prepare_text(text: &str) -> Result<String, String> {
    crate::text::prepare(text, Language::Ru)
}

/// Adapted palatalization/iotation rules from Vosk's g2p.py; explicit acute required.
fn stressed_g2p(word: &str) -> Result<Vec<String>, String> {
    let mut letters = vec![('#', false)];
    for c in word.chars() {
        if c == '\u{301}' {
            letters.last_mut().ok_or("missing stress vowel")?.1 = true;
        } else {
            letters.push((c, false));
        }
    }
    letters.push(('#', false));
    let mut result = Vec::new();
    for i in 1..letters.len() - 1 {
        let (c, stress) = letters[i];
        let next = letters[i + 1].0;
        let previous = letters[i - 1].0;
        let base = match c {
            'б' => "b",
            'в' => "v",
            'г' => "g",
            'д' => "d",
            'з' => "z",
            'к' => "k",
            'л' => "l",
            'м' => "m",
            'н' => "n",
            'п' => "p",
            'р' => "r",
            'с' => "s",
            'т' => "t",
            'ф' => "f",
            'х' => "h",
            'ж' => "zh",
            'ц' => "c",
            'ч' => "ch",
            'ш' => "sh",
            'щ' => "sch",
            'й' => "j",
            'а' | 'я' => "a",
            'у' | 'ю' => "u",
            'о' | 'ё' => "o",
            'э' | 'е' => "e",
            'и' => "i",
            'ы' => "y",
            'ь' | 'ъ' => continue,
            _ => return Err(format!("unsupported Russian letter {c}")),
        };
        if "аеёиоуыэюя".contains(c) {
            if "#ъьаяоёуюэеиы-".contains(previous) && "яюеё".contains(c) {
                result.push("j".into());
            }
            result.push(format!("{base}{}", u8::from(stress)));
        } else if "бвгдзклмнпрстфх".contains(c) && "яёюиье".contains(next) {
            result.push(format!("{base}j"));
        } else {
            result.push(base.into());
        }
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_stress_matches_upstream_rules() {
        assert_eq!(
            stressed_g2p("абстра́кция").unwrap().join(" "),
            "a0 b s t r a1 k c i0 j a0"
        );
        assert_eq!(stressed_g2p("ёж").unwrap().join(" "), "j o0 zh");
        assert_eq!(stressed_g2p("съе́л").unwrap().join(" "), "s j e1 l");
        assert_eq!(stressed_g2p("пя́ть").unwrap().join(" "), "pj a1 tj");
        assert!(stressed_g2p("ї").is_err());
    }
}
