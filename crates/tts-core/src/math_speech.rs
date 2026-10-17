//! English arithmetic parsing, a PCM word bank, and neural synthesis backends.
use crate::{Audio, Language, SynthesisRequest, Synthesizer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

pub fn vocabulary() -> Vec<String> {
    let mut words = std::collections::BTreeSet::new();
    for n in (0..20).chain((20..100).step_by(10)) {
        words.insert(crate::english_numbers::cardinal(n));
    }
    for word in "hundred thousand million plus minus times divided by equals negative positive point open close parenthesis".split_whitespace() {
        words.insert(word.into());
    }
    words.into_iter().collect()
}

/// Read literal expressions; never evaluate or silently repair an equation.
/// Integers 0..999999999; decimals retain every fractional digit.
pub fn words(expression: &str) -> Result<Vec<String>, String> {
    words_for_language(expression, Language::En)
}

/// Literal arithmetic in English, Russian or Ukrainian, including lexical stress.
pub fn words_for_language(expression: &str, language: Language) -> Result<Vec<String>, String> {
    if !matches!(language, Language::En | Language::Ru | Language::Uk) {
        return Err("arithmetic language must be en, ru or uk".into());
    }
    if expression.is_empty() || expression.len() > 4096 {
        return Err("enter 1–4096 bytes of arithmetic".into());
    }
    let chars: Vec<char> = expression.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut depth, mut operand, mut equality) = (0, 0usize, true, false);
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            if !operand {
                return Err("expected an operator".into());
            }
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let digits: String = chars[start..i].iter().collect();
            let n: u32 = digits.parse().map_err(|_| "integer is too large")?;
            if n > 999_999_999 {
                return Err("integer exceeds 999999999".into());
            }
            out.extend(cardinal(n, language).split_whitespace().map(str::to_owned));
            if chars.get(i) == Some(&'.') {
                i += 1;
                let start = i;
                out.push(term("point", language).into());
                while i < chars.len() && chars[i].is_ascii_digit() {
                    out.push(cardinal(chars[i].to_digit(10).unwrap(), language));
                    i += 1;
                }
                if i == start {
                    return Err("decimal needs fractional digits".into());
                }
            }
            operand = false;
            continue;
        }
        match c {
            '(' if operand && depth < 16 => {
                depth += 1;
                out.extend(
                    term("open parenthesis", language)
                        .split_whitespace()
                        .map(str::to_owned),
                );
            }
            ')' if !operand && depth > 0 => {
                depth -= 1;
                out.extend(
                    term("close parenthesis", language)
                        .split_whitespace()
                        .map(str::to_owned),
                );
            }
            '+' | '-' | '−' if operand => {
                out.extend(
                    term(if c == '+' { "positive" } else { "negative" }, language)
                        .split_whitespace()
                        .map(str::to_owned),
                );
                i += 1;
                while chars.get(i).is_some_and(|c| c.is_whitespace()) {
                    i += 1;
                }
                if !chars
                    .get(i)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '(')
                {
                    return Err("sign needs a number or parenthesis".into());
                }
                continue;
            }
            '+' | '-' | '−' | '*' | '×' | '/' | '÷' if !operand => {
                out.extend(
                    term(
                        match c {
                            '+' => "plus",
                            '-' | '−' => "minus",
                            '*' | '×' => "times",
                            _ => "divided by",
                        },
                        language,
                    )
                    .split_whitespace()
                    .map(str::to_owned),
                );
                operand = true;
            }
            '=' if !operand && depth == 0 && !equality => {
                equality = true;
                operand = true;
                out.push(term("equals", language).into());
            }
            _ => return Err(format!("unexpected arithmetic character '{c}'")),
        }
        i += 1;
    }
    if operand || depth != 0 {
        return Err("incomplete expression or unclosed parenthesis".into());
    }
    Ok(out)
}

fn cardinal(n: u32, language: Language) -> String {
    if language == Language::En {
        crate::english_numbers::cardinal(n)
    } else {
        crate::cyrillic_numbers::read_cardinal(n, false, language)
    }
}
fn term(word: &str, language: Language) -> &str {
    match (language, word) {
        (Language::Ru, "plus" | "positive") => "плюс",
        (Language::Ru, "minus" | "negative") => "ми́нус",
        (Language::Ru, "times") => "умно́жить на",
        (Language::Ru, "divided by") => "раздели́ть на",
        (Language::Ru, "equals") => "равно́",
        (Language::Ru, "point") => "то́чка",
        (Language::Ru, "open parenthesis") => "открыва́ется ско́бка",
        (Language::Ru, "close parenthesis") => "закрыва́ется ско́бка",
        (Language::Uk, "plus" | "positive") => "плюс",
        (Language::Uk, "minus" | "negative") => "мі́нус",
        (Language::Uk, "times") => "помно́жити на",
        (Language::Uk, "divided by") => "поділи́ти на",
        (Language::Uk, "equals") => "дорі́внює",
        (Language::Uk, "point") => "кра́пка",
        (Language::Uk, "open parenthesis") => "відкрива́ється ду́жка",
        (Language::Uk, "close parenthesis") => "закрива́ється ду́жка",
        _ => word,
    }
}

/// Complete bounded numeric vocabulary, including gender and magnitude forms.
pub fn vocabulary_for_language(language: Language) -> Result<Vec<String>, String> {
    if language == Language::En {
        return Ok(vocabulary());
    }
    words_for_language("0", language)?;
    let mut words = std::collections::BTreeSet::new();
    for n in (0..1000).chain([1000, 2000, 5000, 1000000, 2000000, 5000000]) {
        words.extend(cardinal(n, language).split_whitespace().map(str::to_owned));
    }
    for key in [
        "plus",
        "minus",
        "times",
        "divided by",
        "equals",
        "point",
        "open parenthesis",
        "close parenthesis",
    ] {
        words.extend(term(key, language).split_whitespace().map(str::to_owned));
    }
    // Feminine thousands must retain their own acoustic entries.
    for n in [1000, 2000] {
        words.extend(cardinal(n, language).split_whitespace().map(str::to_owned));
    }
    Ok(words.into_iter().collect())
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    sample_rate: u32,
    sha256: String,
    provenance: String,
    // Offsets and lengths are in mono signed little-endian PCM16 samples.
    words: BTreeMap<String, [usize; 2]>,
}
pub struct MathVoice {
    manifest: Manifest,
    samples: Vec<f32>,
}
impl MathVoice {
    /// Write data generated from any licensed Synthesizer. Preserve teacher notices separately.
    pub fn build<S: Synthesizer>(
        root: &Path,
        teacher: &mut S,
        speaker: &str,
        provenance: &str,
    ) -> Result<usize, String>
    where
        S::Error: std::fmt::Display,
    {
        if root.exists() {
            return Err("output folder already exists".into());
        }
        let mut pcm = Vec::new();
        let mut entries = BTreeMap::new();
        for word in vocabulary() {
            let audio = teacher
                .synthesize(&SynthesisRequest {
                    text: &word,
                    language: Language::En,
                    speaker,
                    duration_scale: 1.0,
                })
                .map_err(|e| e.to_string())?;
            if audio.sample_rate == 0 || audio.samples.iter().any(|x| !x.is_finite()) {
                return Err("invalid teacher audio".into());
            }
            let peak = audio
                .samples
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max);
            let threshold = (peak * 0.015).max(0.001);
            let first = audio
                .samples
                .iter()
                .position(|v| v.abs() > threshold)
                .ok_or("silent teacher word")?;
            let last = audio
                .samples
                .iter()
                .rposition(|v| v.abs() > threshold)
                .unwrap();
            let margin = audio.sample_rate as usize / 100;
            let source = &audio.samples
                [first.saturating_sub(margin)..(last + margin + 1).min(audio.samples.len())];
            // Windowed-sinc downsampling avoids aliasing at the fixed 16 kHz bank rate.
            let count = source.len() * 16000 / audio.sample_rate as usize;
            let offset = pcm.len() / 2;
            let cutoff = (16000.0 / audio.sample_rate as f64).min(1.0) * 0.9;
            for j in 0..count {
                let position = j as f64 * audio.sample_rate as f64 / 16000.0;
                let center = position.floor() as isize;
                let (mut value, mut weight) = (0.0, 0.0);
                for k in -24..=24 {
                    let index = center + k;
                    if index < 0 || index >= source.len() as isize {
                        continue;
                    }
                    let d = position - index as f64;
                    let x = std::f64::consts::PI * d * cutoff;
                    let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                    let w = sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / 25.0).cos());
                    value += source[index as usize] as f64 * w;
                    weight += w;
                }
                let fade = (j.min(count.saturating_sub(j + 1)) as f64 / 80.0).min(1.0);
                let value =
                    (value / weight.max(1e-10) * fade * 0.85 / peak as f64).clamp(-1.0, 1.0);
                pcm.extend_from_slice(&((value * i16::MAX as f64) as i16).to_le_bytes());
            }
            if count == 0 || count > 16000 * 5 {
                return Err("invalid word duration".into());
            }
            entries.insert(word, [offset, count]);
        }
        let manifest = Manifest {
            version: 1,
            sample_rate: 16000,
            sha256: format!("{:x}", Sha256::digest(&pcm)),
            provenance: provenance.into(),
            words: entries,
        };
        let json = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
        let total = json.len() + pcm.len();
        std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
        std::fs::write(root.join("words.pcm"), pcm).map_err(|e| e.to_string())?;
        std::fs::write(root.join("math-voice.json"), json).map_err(|e| e.to_string())?;
        Ok(total)
    }
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join("math-voice.json");
        if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 65536 {
            return Err("manifest too large".into());
        }
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let path = root.join("words.pcm");
        if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 4_000_000 {
            return Err("word bank too large".into());
        }
        let pcm = std::fs::read(path).map_err(|e| e.to_string())?;
        if manifest.version != 1
            || manifest.sample_rate != 16000
            || pcm.len() % 2 != 0
            || format!("{:x}", Sha256::digest(&pcm)) != manifest.sha256
        {
            return Err("invalid or modified math voice".into());
        }
        let samples: Vec<f32> = pcm
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        for word in vocabulary() {
            let [offset, length] = *manifest.words.get(&word).ok_or("missing required word")?;
            if length == 0
                || length > 80000
                || offset
                    .checked_add(length)
                    .is_none_or(|end| end > samples.len())
            {
                return Err("invalid word range".into());
            }
        }
        Ok(Self { manifest, samples })
    }
    pub fn expression(&self, text: &str) -> Result<Audio, String> {
        let mut samples = Vec::new();
        for word in words(text)? {
            let [offset, length] = self.manifest.words[&word];
            let pause = if ["plus", "minus", "times", "divided", "equals"].contains(&word.as_str())
            {
                960
            } else {
                240
            };
            if !samples.is_empty() {
                samples.resize(samples.len() + pause, 0.0);
            }
            samples.extend_from_slice(&self.samples[offset..offset + length]);
            if samples.len() > 16000 * 120 {
                return Err("expression audio exceeds two minutes".into());
            }
        }
        Ok(Audio {
            sample_rate: 16000,
            samples,
        })
    }
}
impl Synthesizer for MathVoice {
    type Error = crate::kokoro::Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Self::Error> {
        if request.language != Language::En
            || !matches!(request.speaker, "" | "default")
            || request.duration_scale != 1.0
        {
            return Err(crate::kokoro::Error(
                "math voice supports English, default speaker and duration 1 only".into(),
            ));
        }
        self.expression(request.text).map_err(crate::kokoro::Error)
    }
}

/// Neural arithmetic adapter for English VITS or numeric-only latent packages.
/// It shares the exact parser with the fragment baseline.
enum MathBackend {
    Sequence(Box<crate::math_sequence::MathSequence>),
    Vits(Box<crate::vits::Vits>),
    Latent(Box<crate::math_latent::MathLatent>),
}
pub struct MathStudent(MathBackend);
impl MathStudent {
    pub fn load(root: &Path) -> Result<Self, crate::kokoro::Error> {
        if root.join("sequence.json").exists() {
            return Ok(Self(MathBackend::Sequence(Box::new(
                crate::math_sequence::MathSequence::load(root)?,
            ))));
        }
        if root.join("arithmetic-neural.json").exists() {
            return Ok(Self(MathBackend::Latent(Box::new(
                crate::math_latent::MathLatent::load(root)?,
            ))));
        }
        let voice = crate::vits::Vits::load(root)?;
        if voice.manifest.language != Language::En
            || !matches!(
                voice.manifest.frontend.as_str(),
                "grapheme-v2" | "cmu-ipa-v1"
            )
        {
            return Err(crate::kokoro::Error(
                "math student requires English grapheme-v2 or CMU IPA export".into(),
            ));
        }
        Ok(Self(MathBackend::Vits(Box::new(voice))))
    }
}
impl MathStudent {
    pub fn language(&self) -> Language {
        match &self.0 {
            MathBackend::Sequence(v) => v.language,
            MathBackend::Latent(v) => v.language,
            MathBackend::Vits(v) => v.manifest.language,
        }
    }
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Audio, crate::kokoro::Error> {
        crate::paragraph::check_cancel(cancel).map_err(crate::kokoro::Error)?;
        if request.language != self.language() || !matches!(request.speaker, "" | "default") {
            return Err(crate::kokoro::Error(
                "arithmetic voice language must match request; use default speaker".into(),
            ));
        }
        let words =
            words_for_language(request.text, request.language).map_err(crate::kokoro::Error)?;
        let voice = match &mut self.0 {
            MathBackend::Latent(voice) => {
                return voice.synthesize_words(&words, request.duration_scale, cancel);
            }
            MathBackend::Sequence(voice) => {
                return voice.synthesize_words(&words, request.duration_scale, cancel);
            }
            MathBackend::Vits(voice) => voice,
        };
        let text = words.join(" ");
        voice.synthesize_cancellable(
            &SynthesisRequest {
                text: &text,
                speaker: &voice.manifest.speakers[0].clone(),
                ..*request
            },
            cancel,
        )
    }
}

impl Synthesizer for MathStudent {
    type Error = crate::kokoro::Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Self::Error> {
        self.synthesize_cancellable(request, &std::sync::atomic::AtomicBool::new(false))
    }
}

/// Reproducible teacher curriculum with disjoint spoken forms and balanced structures.
#[derive(Serialize)]
pub struct CorpusLine {
    pub expression: String,
    pub spoken: String,
    pub split: crate::dataset::Split,
}
/// Short utterances for establishing speech before long arithmetic chains.
pub fn short_corpus(count: usize) -> Result<Vec<CorpusLine>, String> {
    use rand::{Rng, SeedableRng};
    if !(128..=384).contains(&count) {
        return Err("short curriculum requires 128–384 lines".into());
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d6174687633);
    let mut seen = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for i in 0..count {
        let mut accepted = None;
        for _ in 0..10000 {
            let a = rng.random_range(0..100u32);
            let b = rng.random_range(0..20u32);
            let expression = match i % 4 {
                0 => a.to_string(),
                1 => format!("{} + {b}", a % 20),
                2 => format!("{} - {b}", a % 20),
                _ => format!("{} × {b}", a % 20),
            };
            let spoken = words(&expression)?.join(" ");
            if seen.insert(spoken.clone()) {
                accepted = Some((expression, spoken));
                break;
            }
        }
        let (expression, spoken) = accepted.ok_or("short curriculum exhausted")?;
        let split = match (i / 4) % 8 {
            6 => crate::dataset::Split::Validation,
            7 => crate::dataset::Split::Test,
            _ => crate::dataset::Split::Train,
        };
        result.push(CorpusLine {
            expression,
            spoken,
            split,
        });
    }
    Ok(result)
}
pub fn corpus(count: usize) -> Result<Vec<CorpusLine>, String> {
    use rand::{Rng, SeedableRng};
    if !(128..=10000).contains(&count) {
        return Err("curriculum requires 128–10000 lines".into());
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d6174687632);
    let mut seen = std::collections::BTreeSet::new();
    let mut result = Vec::new();
    for i in 0..count {
        let mut accepted = None;
        for _ in 0..1000 {
            let a = rng.random_range(0..1000u32);
            let b = rng.random_range(1..1000u32);
            let expression = match i % 16 {
                0 => format!("{a} + {b} = {}", a + b),
                1 => format!("{a} - {b}"),
                2 => format!("{} × {}", a % 30, b % 30),
                3 => format!("{a} ÷ {b}"),
                4 => format!("-{a} + {b}"),
                5 => format!("{}.{} + {}.{}", a % 10, a % 100, b % 10, b % 100),
                6 => format!("({a} + {b}) × {}", b % 9 + 1),
                7 => format!("{} + {b}", a * 1000),
                8 => format!("+{a} - {b}"),
                9 => format!("{a} × 0 = 0"),
                10 => format!("0.00{} × {b}", a % 10),
                11 => format!("{} - {b}", a * 1_000_000 + b),
                12 => format!("-{a} ÷ -{b}"),
                13 => format!("{a} + ({b} - {})", a % 100),
                14 => format!("({a} + {b}) ÷ ({} + {})", a % 20 + 1, b % 20 + 1),
                _ => format!("{}.{} = {}.{}", a, a % 100, a, a % 100),
            };
            let spoken = words(&expression)?.join(" ");
            if seen.insert(spoken.clone()) {
                accepted = Some((expression, spoken));
                break;
            }
        }
        let (expression, spoken) = accepted.ok_or("unable to generate unique curriculum")?;
        // Each block of 16 covers all structures. Reserve complete blocks so
        // validation/test cover every operator family, unlike the v1 split.
        let split = match (i / 16) % 8 {
            6 => crate::dataset::Split::Validation,
            7 => crate::dataset::Split::Test,
            _ => crate::dataset::Split::Train,
        };
        result.push(CorpusLine {
            expression,
            spoken,
            split,
        });
    }
    Ok(result)
}
