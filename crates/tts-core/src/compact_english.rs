//! Experimental compact English ONNX backends with native BSD CMU frontend.
//! Vocabulary data follows publisher inventories; notices accompany model assets.
use crate::{Audio, Language, SynthesisRequest, Synthesizer, english::English, kokoro::Error};
use ndarray::{Array2, ArrayD, arr0};
use ort::{session::Session, value::Tensor};
use rand::{Rng, SeedableRng};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::atomic::AtomicBool,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactKind {
    Inflect,
    Kitten,
}
impl CompactKind {
    fn key(self) -> &'static str {
        match self {
            Self::Inflect => "inflect",
            Self::Kitten => "kitten",
        }
    }
}
/// Speaker names a Kitten package provides.
pub const KITTEN_VOICES: [&str; 8] = [
    "expr-voice-2-f",
    "expr-voice-2-m",
    "expr-voice-3-f",
    "expr-voice-3-m",
    "expr-voice-4-f",
    "expr-voice-4-m",
    "expr-voice-5-f",
    "expr-voice-5-m",
];

pub struct CompactEnglish {
    kind: CompactKind,
    first: Session,
    decode: Option<Session>,
    dictionary: English,
    symbols: HashMap<char, i64>,
    styles: HashMap<String, Vec<f32>>,
}
impl CompactEnglish {
    /// Adds CMU-format user pronunciations; see [`English::add_lexicon`].
    pub fn add_lexicon(&mut self, path: &Path) -> Result<usize, Error> {
        self.dictionary.add_lexicon(path).map_err(Error)
    }
    pub fn load(root: &Path, kind: CompactKind) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let pins: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/compact-english-assets.json"))?;
            let hashes: BTreeMap<String, String> = pins[kind.key()]["files"]
                .as_object()
                .ok_or("invalid pins")?
                .iter()
                .filter(|(_, v)| v["acquisition_only"] != true)
                .map(|(n, v)| {
                    Ok((
                        n.clone(),
                        v["sha256"].as_str().ok_or("invalid hash")?.into(),
                    ))
                })
                .collect::<Result<_, &str>>()?;
            crate::integrity::verify(root, &hashes, &[]).map_err(std::io::Error::other)?;
            let vocab: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/compact-english-symbols.json"))?;
            let symbols = vocab[kind.key()]
                .as_str()
                .ok_or("invalid symbols")?
                .chars()
                .enumerate()
                .map(|(i, c)| (c, i as i64))
                .collect();
            let dictionary =
                English::load(&root.join("cmudict.dict")).map_err(std::io::Error::other)?;
            let session = |name: &str| {
                Session::builder()?
                    .with_intra_threads(2)?
                    .commit_from_file(root.join(name))
            };
            let mut styles = HashMap::new();
            if kind == CompactKind::Kitten {
                for name in KITTEN_VOICES {
                    let b = std::fs::read(root.join(format!("styles/{name}.f32")))?;
                    if b.len() != 400 * 256 * 4 {
                        return Err("invalid style shape".into());
                    }
                    let values: Vec<f32> = b
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect();
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err("invalid style values".into());
                    }
                    styles.insert(name.into(), values);
                }
            }
            Ok(Self {
                kind,
                first: session(if kind == CompactKind::Inflect {
                    "onnx/duration.onnx"
                } else {
                    "kitten_tts_nano_v0_8.onnx"
                })?,
                decode: if kind == CompactKind::Inflect {
                    Some(session("onnx/decode.onnx")?)
                } else {
                    None
                },
                dictionary,
                symbols,
                styles,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &AtomicBool,
    ) -> Result<Audio, Error> {
        let run = || -> Result<Audio, Box<dyn std::error::Error>> {
            if request.language != Language::En {
                return Err("compact models support English only".into());
            }
            if !request.duration_scale.is_finite() || !(0.5..=2.0).contains(&request.duration_scale)
            {
                return Err("duration scale must be 0.5–2".into());
            }
            let speaker = match request.speaker {
                "default" | "Bruno" => "expr-voice-3-m",
                "Bella" => "expr-voice-2-f",
                "Jasper" => "expr-voice-2-m",
                "Luna" => "expr-voice-3-f",
                "Rosie" => "expr-voice-4-f",
                "Hugo" => "expr-voice-4-m",
                "Kiki" => "expr-voice-5-f",
                "Leo" => "expr-voice-5-m",
                v => v,
            };
            if self.kind == CompactKind::Inflect && !matches!(request.speaker, "default" | "") {
                return Err("Inflect has one default male voice".into());
            }
            if self.kind == CompactKind::Kitten && !self.styles.contains_key(speaker) {
                return Err("unknown Kitten voice".into());
            }
            let mut joined = None;
            // Resolve numeric forms and contextual readings over the entire passage.
            // Splitting raw text can detach units or abbreviations from their numbers.
            let prepared = self
                .dictionary
                .phonemize_ipa_current(request.text)
                .map_err(std::io::Error::other)?;
            for chunk in crate::paragraph::chunks(&prepared, 240).map_err(std::io::Error::other)? {
                crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
                let phones = chunk.clone();
                let phones = if self.kind == CompactKind::Kitten {
                    kitten_spacing(&phones)
                } else {
                    phones
                };
                let ids: Vec<i64> = phones
                    .chars()
                    .map(|c| {
                        self.symbols
                            .get(&c)
                            .copied()
                            .ok_or_else(|| format!("unsupported phoneme {c}"))
                    })
                    .collect::<Result<_, _>>()
                    .map_err(std::io::Error::other)?;
                if ids.is_empty() || ids.len() > 1024 {
                    return Err("invalid phoneme length".into());
                }
                let samples = if self.kind == CompactKind::Inflect {
                    let mut tokens = vec![0; ids.len() * 2 + 1];
                    for (i, v) in ids.iter().enumerate() {
                        tokens[i * 2 + 1] = *v;
                    }
                    let len = tokens.len();
                    let out=self.first.run(ort::inputs!["tokens"=>Tensor::from_array(Array2::from_shape_vec((1,len),tokens)?)?,"lengths"=>Tensor::from_array(([1],vec![len as i64]))?,"length_scale"=>Tensor::from_array(arr0(request.duration_scale))?])?;
                    let m = out["m_p_exp"].try_extract_array::<f32>()?.to_owned();
                    let logs = out["logs_p_exp"].try_extract_array::<f32>()?.to_owned();
                    let mask = out["y_mask"].try_extract_array::<f32>()?.to_owned();
                    if m.ndim() != 3 || m.shape()[2] > 4096 || m.iter().any(|v| !v.is_finite()) {
                        return Err("invalid or excessive duration output".into());
                    }
                    crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
                    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
                    let noise = ArrayD::from_shape_vec(
                        m.raw_dim(),
                        (0..m.len())
                            .map(|_| {
                                let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
                                (-2.0 * u.ln()).sqrt()
                                    * (std::f32::consts::TAU * rng.random::<f32>()).cos()
                            })
                            .collect(),
                    )?;
                    let out=self.decode.as_mut().unwrap().run(ort::inputs!["m_p_exp"=>Tensor::from_array(m)?,"logs_p_exp"=>Tensor::from_array(logs)?,"y_mask"=>Tensor::from_array(mask)?,"zp_noise"=>Tensor::from_array(noise)?,"noise_scale"=>Tensor::from_array(arr0(0.667f32))?])?;
                    out["waveform"].try_extract_tensor::<f32>()?.1.to_vec()
                } else {
                    let mut tokens = vec![0];
                    tokens.extend(ids);
                    tokens.extend([10, 0]);
                    let len = tokens.len();
                    let row = chunk.chars().count().min(399);
                    let style = self.styles[speaker][row * 256..(row + 1) * 256].to_vec();
                    let prior = if speaker == "expr-voice-4-m" {
                        0.9
                    } else {
                        0.8
                    };
                    let out=self.first.run(ort::inputs!["input_ids"=>Tensor::from_array(Array2::from_shape_vec((1,len),tokens)?)?,"style"=>Tensor::from_array(Array2::from_shape_vec((1,256),style)?)?,"speed"=>Tensor::from_array(([1],vec![prior/request.duration_scale]))?])?;
                    let mut samples = out[0].try_extract_tensor::<f32>()?.1.to_vec();
                    samples.truncate(samples.len().saturating_sub(5000));
                    samples
                };
                crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
                if samples.len() > 24000 * 60 {
                    return Err("excessive compact audio output".into());
                }
                crate::paragraph::append(
                    &mut joined,
                    Audio {
                        sample_rate: 24000,
                        samples,
                    },
                )
                .map_err(std::io::Error::other)?;
            }
            joined.ok_or_else(|| "no audio".into())
        };
        let mut run = run;
        run().map_err(|e| Error(e.to_string()))
    }
}
fn kitten_spacing(phones: &str) -> String {
    let mut words = Vec::new();
    let mut word = String::new();
    for c in phones.chars() {
        if c.is_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            if !c.is_whitespace() {
                words.push(c.to_string());
            }
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words.join(" ")
}
impl Synthesizer for CompactEnglish {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Error> {
        self.synthesize_cancellable(request, &AtomicBool::new(false))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stress_spacing_follows_kitten_contract() {
        assert_eq!(kitten_spacing("həlˈoʊ, wˈɜːld!"), "həlˈoʊ , wˈɜːld !");
    }
}
