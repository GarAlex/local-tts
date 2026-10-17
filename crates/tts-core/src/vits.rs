use crate::{Audio, SynthesisRequest, Synthesizer, kokoro::Error, training, voice::VoiceManifest};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::{collections::HashMap, path::Path};

pub struct Vits {
    session: Session,
    pub manifest: VoiceManifest,
    symbols: HashMap<char, i64>,
    n_symbols: usize,
    dictionary: Option<crate::english::English>,
    russian: Option<Box<crate::vosk::Frontend>>,
}
impl Vits {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let manifest: VoiceManifest =
                serde_json::from_slice(&std::fs::read(root.join("voice.json"))?)?;
            manifest.validate(root).map_err(std::io::Error::other)?;
            crate::integrity::verify_vits(root, &manifest.model).map_err(std::io::Error::other)?;
            if manifest.frontend != "grapheme-v1"
                && manifest.frontend != crate::frontend::VERSION
                && manifest.frontend != training::IPA_FRONTEND
                && manifest.frontend != crate::piper_training::VERSION
                && manifest.frontend != crate::vosk::TRAINING_VERSION
                && manifest.frontend != crate::css10::VERSION
                && manifest.frontend != crate::neon::VERSION
            {
                return Err("unsupported frontend".into());
            }
            if manifest.frontend == crate::neon::VERSION
                && !crate::neon::supports(manifest.language)
            {
                return Err("unsupported Neon language".into());
            }
            if manifest.frontend == crate::css10::VERSION
                && !matches!(manifest.language, crate::Language::De | crate::Language::Es)
            {
                return Err("CSS10 frontend requires German or Spanish".into());
            }
            if manifest.frontend == crate::piper_training::VERSION
                && manifest.language != crate::Language::Uk
            {
                return Err("Piper text training frontend requires Ukrainian".into());
            }
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("tokens.json"))?)?;
            if config["schema_version"].as_u64() != Some(1) {
                return Err("unsupported token schema".into());
            }
            let inventory: Vec<char> = config["symbols"]
                .as_array()
                .ok_or("missing symbols")?
                .iter()
                .map(|value| {
                    let mut chars = value.as_str().ok_or("invalid symbol")?.chars();
                    let c = chars.next().ok_or("empty symbol")?;
                    if chars.next().is_some() {
                        return Err("invalid symbol");
                    }
                    Ok(c)
                })
                .collect::<Result<_, _>>()?;
            if manifest.frontend == crate::neon::VERSION
                && inventory != crate::neon::symbols(manifest.language)?
            {
                return Err("Neon inventory differs from language contract".into());
            }
            training::validate_symbols(&inventory, &manifest.frontend)
                .map_err(std::io::Error::other)?;
            let symbols = inventory
                .iter()
                .enumerate()
                .map(|(id, c)| (*c, id as i64))
                .collect();
            let dictionary = if manifest.frontend == training::IPA_FRONTEND {
                if manifest.language != crate::Language::En {
                    return Err("CMU IPA requires English".into());
                }
                let hashes = serde_json::from_slice(&std::fs::read(root.join("checksums.json"))?)?;
                crate::integrity::verify(root, &hashes, &["cmudict.dict"])
                    .map_err(std::io::Error::other)?;
                Some(
                    crate::english::English::load(&root.join("cmudict.dict"))
                        .map_err(std::io::Error::other)?,
                )
            } else {
                None
            };
            let russian = if manifest.frontend == crate::vosk::TRAINING_VERSION {
                if manifest.language != crate::Language::Ru {
                    return Err("Russian phoneme frontend requires Russian".into());
                }
                let hashes = serde_json::from_slice(&std::fs::read(root.join("checksums.json"))?)?;
                crate::integrity::verify(root, &hashes, &["dictionary"])
                    .map_err(std::io::Error::other)?;
                Some(Box::new(crate::vosk::Frontend::load(
                    &root.join("dictionary"),
                )?))
            } else {
                None
            };
            let session = Session::builder()?
                .with_intra_threads(2)?
                .commit_from_file(root.join(&manifest.model))?;
            Ok(Self {
                session,
                manifest,
                symbols,
                n_symbols: inventory.len(),
                dictionary,
                russian,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    /// Adds CMU-format user pronunciations to an English CMU IPA voice.
    pub fn add_lexicon(&mut self, path: &Path) -> Result<usize, Error> {
        self.dictionary
            .as_mut()
            .ok_or_else(|| Error("lexicon requires an English CMU IPA voice".into()))?
            .add_lexicon(path)
            .map_err(Error)
    }
    pub fn synthesize_tokens(
        &mut self,
        tokens: &[i64],
        duration_scale: f32,
    ) -> Result<Audio, Error> {
        if tokens.is_empty()
            || tokens.len() > 1024
            || tokens
                .iter()
                .any(|id| *id < 0 || *id >= self.n_symbols as i64)
        {
            return Err(Error("invalid tokens or text too long".into()));
        }
        if !duration_scale.is_finite() || !(0.5..=2.0).contains(&duration_scale) {
            return Err(Error("duration scale must be 0.5–2.0".into()));
        }
        let mut run = || -> Result<Audio, Box<dyn std::error::Error>> {
            let outputs = self.session.run(ort::inputs![
                "tokens" => Tensor::from_array(Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?)?,
                "lengths" => Tensor::from_array(([1usize], vec![tokens.len() as i64]))?,
                "duration_scale" => Tensor::from_array(([1usize], vec![duration_scale]))?
            ])?;
            let (_, samples) = outputs[0].try_extract_tensor::<f32>()?;
            if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
                return Err("invalid model output".into());
            }
            Ok(Audio {
                sample_rate: self.manifest.sample_rate,
                samples: samples.to_vec(),
            })
        };
        run().map_err(|e| Error(e.to_string()))
    }
    pub fn verify(&mut self, root: &Path) -> Result<f32, Error> {
        #[derive(serde::Deserialize)]
        struct Reference {
            tokens: Vec<i64>,
            duration_scale: f32,
            samples: Vec<f32>,
        }
        let references: Vec<Reference> = serde_json::from_slice(
            &std::fs::read(root.join("reference.json")).map_err(|e| Error(e.to_string()))?,
        )
        .map_err(|e| Error(e.to_string()))?;
        let mut max_error = 0.0_f32;
        if references.is_empty() {
            return Err(Error("empty reference fixtures".into()));
        }
        for reference in references {
            let actual = self.synthesize_tokens(&reference.tokens, reference.duration_scale)?;
            if actual.samples.len() != reference.samples.len() || reference.samples.is_empty() {
                return Err(Error("native/reference audio lengths differ".into()));
            }
            for (a, b) in actual.samples.iter().zip(reference.samples) {
                if !b.is_finite() {
                    return Err(Error("invalid reference audio".into()));
                }
                max_error = max_error.max((a - b).abs());
            }
        }
        if max_error > 0.001 {
            return Err(Error(format!(
                "numerical parity failed: maximum absolute error {max_error}"
            )));
        }
        Ok(max_error)
    }
}
impl Synthesizer for Vits {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Error> {
        self.synthesize_cancellable(request, &std::sync::atomic::AtomicBool::new(false))
    }
}
impl Vits {
    /// Observe cancellation before and after each model invocation. Partial audio is discarded.
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Audio, Error> {
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        if request.language != self.manifest.language
            || !self.manifest.speakers.iter().any(|s| s == request.speaker)
        {
            return Err(Error("language/speaker does not match voice".into()));
        }
        let input = if self.manifest.normalization.is_some() {
            crate::text::prepare(request.text, request.language).map_err(Error)?
        } else {
            request.text.to_string()
        };
        let text = if let Some(russian) = &self.russian {
            russian.normalize_training(&input)?
        } else if let Some(dictionary) = &self.dictionary {
            if self.manifest.normalization.is_some() {
                dictionary.phonemize_ipa_current(&input).map_err(Error)?
            } else {
                dictionary.phonemize_ipa(&input).map_err(Error)?
            }
        } else if self.manifest.frontend == crate::piper_training::VERSION {
            crate::piper_training::normalize(&input, request.language).map_err(Error)?
        } else if self.manifest.frontend == crate::neon::VERSION {
            crate::neon::normalize(&input, request.language).map_err(Error)?
        } else if self.manifest.frontend == crate::css10::VERSION {
            crate::css10::normalize(&input, request.language).map_err(Error)?
        } else if self.manifest.frontend == crate::frontend::VERSION {
            crate::frontend::normalize(&input, request.language).map_err(Error)?
        } else {
            training::normalize(&input)
        };
        if text.is_empty() {
            return Err(Error("enter text".into()));
        }
        let chunks = crate::paragraph::chunks(
            &text,
            if matches!(
                self.manifest.frontend.as_str(),
                crate::piper_training::VERSION | crate::vosk::TRAINING_VERSION
            ) {
                240
            } else {
                511
            },
        )
        .map_err(Error)?;
        for c in text.chars() {
            if !self.symbols.contains_key(&c) {
                return Err(Error(format!(
                    "character '{c}' absent from training inventory"
                )));
            }
        }
        let mut output = None;
        for chunk in chunks {
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
            let mut tokens = vec![0];
            for c in chunk.chars() {
                tokens.push(self.symbols[&c]);
                tokens.push(0);
            }
            if self.manifest.frontend == crate::piper_training::VERSION {
                tokens = crate::piper_training::encode(&chunk).map_err(Error)?;
            }
            if self.manifest.frontend == crate::vosk::TRAINING_VERSION {
                tokens = crate::vosk::encode_training(&chunk).map_err(Error)?;
            }
            if self.manifest.frontend == crate::neon::VERSION {
                tokens = crate::neon::encode(&chunk, request.language).map_err(Error)?;
            } else if self.manifest.frontend == crate::css10::VERSION {
                tokens = crate::css10::encode(&chunk).map_err(Error)?;
            }
            crate::paragraph::append(
                &mut output,
                self.synthesize_tokens(&tokens, request.duration_scale)?,
            )
            .map_err(Error)?;
        }
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        output.ok_or_else(|| Error("enter text".into()))
    }
}
