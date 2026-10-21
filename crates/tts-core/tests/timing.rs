//! Word timing from the models' own duration predictors (0.7): the timed
//! audio is the untimed audio, the durations account for every sample, and
//! the words cover what was said in order. Needs installed packages, so
//! ignored in CI like the other model tests: run with
//! `cargo test -p local-tts-core --test timing -- --ignored`.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer, TimedSynthesizer,
    compact_english::{CompactEnglish, CompactKind},
    english::English,
    kokoro::{Kokoro, Variant},
};
use std::path::PathBuf;

fn voices() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices")
}

const CORPUS: [&str; 6] = [
    "Meet Euclid. Geometry, proved step by step.",
    "Save $49.99 on Jan. 5 — the CEO said it twice.",
    "The camera never moved; the viewport did.",
    "Narration is written here, not recorded.",
    "One, two, three. Four! Five? Six.",
    "Built for 1,250 worksheets at 3:19 p.m. on promoshot.app.",
];

fn request<'a>(text: &'a str, speaker: &'a str) -> SynthesisRequest<'a> {
    SynthesisRequest {
        text,
        language: Language::En,
        speaker,
        duration_scale: 1.0,
    }
}

fn kokoro(variant: Variant) -> Kokoro {
    let mut engine = Kokoro::load(&voices().join(variant.folder())).unwrap();
    engine.set_fallback(true).unwrap();
    engine
}

fn kitten() -> CompactEnglish {
    let mut engine =
        CompactEnglish::load(&voices().join("kitten-nano-en"), CompactKind::Kitten).unwrap();
    engine.set_fallback(true).unwrap();
    engine
}

/// Words are in order, inside the audio, each a positive span, one per
/// spoken word of the text.
fn sane(words: &[local_tts_core::SpokenWord], seconds: f64, text: &str) {
    assert!(!words.is_empty(), "{text}");
    for w in words {
        assert!(w.start < w.end, "{text}: {w:?}");
        assert!(w.end <= seconds + 1e-9, "{text}: {w:?} past {seconds}");
    }
    for pair in words.windows(2) {
        assert!(pair[0].end <= pair[1].start + 1e-9, "{text}: {pair:?}");
    }
}

#[test]
#[ignore = "requires installed Kokoro F32, Q8 and Q4"]
fn timed_audio_equals_untimed_audio() {
    for variant in Variant::ALL {
        let mut timed = kokoro(variant);
        let mut plain = Kokoro::load_unpatched(&voices().join(variant.folder())).unwrap();
        plain.set_fallback(true).unwrap();
        for text in CORPUS {
            let a = timed.synthesize_timed(&request(text, "af_heart")).unwrap();
            let b = timed.synthesize(&request(text, "af_heart")).unwrap();
            // The 0.6 session, built from the file with no extra output.
            let c = plain.synthesize(&request(text, "af_heart")).unwrap();
            let bits = |s: &[f32]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&a.audio.samples),
                bits(&b.samples),
                "{variant:?} {text}"
            );
            assert_eq!(
                bits(&a.audio.samples),
                bits(&c.samples),
                "{variant:?} unpatched {text}"
            );
            sane(&a.words, a.audio.samples.len() as f64 / 24000.0, text);
        }
    }
}

/// Kitten's graph draws noise, so two plain calls already differ sample
/// by sample (as in 0.6): what timing can promise there is the same length
/// — the durations decide it — and the same words every time.
#[test]
#[ignore = "requires installed Kitten nano"]
fn timed_audio_equals_untimed_audio_for_kitten() {
    let mut engine = kitten();
    for text in CORPUS {
        for speaker in ["Luna", "Bruno"] {
            let a = engine.synthesize_timed(&request(text, speaker)).unwrap();
            let b = engine.synthesize(&request(text, speaker)).unwrap();
            let c = engine.synthesize(&request(text, speaker)).unwrap();
            assert_eq!(a.audio.samples.len(), b.samples.len(), "{speaker} {text}");
            assert_eq!(b.samples.len(), c.samples.len());
            let again = engine.synthesize_timed(&request(text, speaker)).unwrap();
            assert_eq!(a.words, again.words, "{speaker} {text}");
            sane(&a.words, a.audio.samples.len() as f64 / 24000.0, text);
        }
    }
}

#[test]
#[ignore = "requires installed Inflect nano"]
fn inflect_says_timing_is_not_supported() {
    let mut engine =
        CompactEnglish::load(&voices().join("inflect-nano-en"), CompactKind::Inflect).unwrap();
    let err = engine
        .synthesize_timed(&request("Hello there.", "default"))
        .err()
        .unwrap();
    assert!(err.to_string().contains("timing not supported for Inflect"));
}

/// The unit is pinned, not assumed: Kokoro's frames are 600 samples at
/// 24 kHz on every export, within one frame.
#[test]
#[ignore = "requires installed Kokoro F32, Q8 and Q4"]
fn durations_sum_to_samples() {
    for variant in Variant::ALL {
        let mut engine = kokoro(variant);
        for text in CORPUS {
            let phones = engine.phonemize(text).unwrap();
            let (audio, durations) = engine
                .synthesize_phonemes_timed_as("af_heart", &phones, 1.0)
                .unwrap();
            assert_eq!(
                durations.len(),
                phones.chars().count() + 2,
                "one per token, pads too"
            );
            let frames: i64 = durations.iter().sum();
            let expected = frames as f64 * local_tts_core::kokoro::SAMPLES_PER_FRAME;
            assert!(
                (audio.samples.len() as f64 - expected).abs() <= 600.0,
                "{variant:?} {text}: {} samples, {frames} frames",
                audio.samples.len()
            );
        }
        let timed = engine
            .synthesize_timed(&request(CORPUS[0], "af_heart"))
            .unwrap();
        assert!(
            (timed.frame_seconds - 0.025).abs() < 1e-3,
            "{}",
            timed.frame_seconds
        );
    }
}

/// Kitten's `duration` output is frames per token too; its unit is
/// measured the same way and the 5000-sample tail trim respected.
#[test]
#[ignore = "requires installed Kitten nano"]
fn kitten_durations_account_for_the_samples() {
    let mut engine = kitten();
    let mut units = Vec::new();
    for text in CORPUS {
        let timed = engine.synthesize_timed(&request(text, "Luna")).unwrap();
        units.push(timed.frame_seconds * 24000.0);
    }
    let first = units[0];
    for unit in &units {
        assert!((unit - first).abs() / first < 0.02, "{units:?}");
    }
    eprintln!("kitten samples per frame: {units:?}");
}

#[test]
#[ignore = "requires installed Kokoro Q8 (its dictionary)"]
fn phoneme_strings_unchanged_by_spans() {
    let mut english = English::load(&voices().join("kokoro-en-q8/cmudict.dict")).unwrap();
    english.set_fallback(true).unwrap();
    let mut lines: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    lines.extend(
        local_tts_core::math_speech::corpus(128)
            .unwrap()
            .into_iter()
            .take(60)
            .map(|l| l.spoken),
    );
    for line in &lines {
        let Ok(plain) = english.phonemize_american(line) else {
            continue;
        };
        assert_eq!(
            english.phonemize_american_spans(line).unwrap().0,
            plain,
            "{line}"
        );
        assert_eq!(
            english.phonemize_ipa_current_spans(line).unwrap().0,
            english.phonemize_ipa_current(line).unwrap(),
            "{line}"
        );
    }
}

/// Every spoken word has a span; spans are in order, never overlap, and
/// hold only that word's phones (no spaces or punctuation of the text).
#[test]
#[ignore = "requires installed Kokoro Q8 (its dictionary)"]
fn word_spans_cover_every_spoken_word() {
    let mut english = English::load(&voices().join("kokoro-en-q8/cmudict.dict")).unwrap();
    english.set_fallback(true).unwrap();
    for text in CORPUS {
        for (phones, spans) in [
            english.phonemize_american_spans(text).unwrap(),
            english.phonemize_ipa_current_spans(text).unwrap(),
        ] {
            let spoken = local_tts_core::english::narration_text(text).unwrap();
            let expected = spoken
                .split(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
                .filter(|w| !w.is_empty())
                .count();
            assert_eq!(spans.len(), expected, "{text}: {spans:?}");
            for pair in spans.windows(2) {
                assert!(pair[0].range.end <= pair[1].range.start, "{text}: {pair:?}");
            }
            for span in &spans {
                let piece = &phones[span.range.clone()];
                assert!(!piece.is_empty());
                // Only a word's own phones: a spelled acronym's letters may
                // stand apart inside it, but no span begins or ends on a gap.
                assert_eq!(piece.trim(), piece, "{text}: {piece:?}");
                assert!(
                    !piece.contains(['.', ',', '!', '?', ';', ':']),
                    "{text}: {piece}"
                );
            }
        }
    }
}

/// "$49.99" is said as words, and each is timed in order.
#[test]
#[ignore = "requires installed Kokoro Q8"]
fn normalized_words_are_reported_as_spoken() {
    let mut engine = kokoro(Variant::Q8);
    let timed = engine
        .synthesize_timed(&request("Save $49.99 today.", "af_heart"))
        .unwrap();
    let texts: Vec<&str> = timed.words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "save", "forty", "nine", "dollars", "ninety", "nine", "cents", "today"
        ]
    );
}

#[test]
#[ignore = "requires installed Kokoro Q8 and Kitten nano"]
fn timed_is_deterministic() {
    let mut engine = kokoro(Variant::Q8);
    let mut small = kitten();
    for text in CORPUS {
        let a = serde_json::to_string(
            &engine
                .synthesize_timed(&request(text, "af_heart"))
                .unwrap()
                .words,
        )
        .unwrap();
        let b = serde_json::to_string(
            &engine
                .synthesize_timed(&request(text, "af_heart"))
                .unwrap()
                .words,
        )
        .unwrap();
        assert_eq!(a, b);
        let a = serde_json::to_string(
            &small
                .synthesize_timed(&request(text, "Luna"))
                .unwrap()
                .words,
        )
        .unwrap();
        let b = serde_json::to_string(
            &small
                .synthesize_timed(&request(text, "Luna"))
                .unwrap()
                .words,
        )
        .unwrap();
        assert_eq!(a, b);
    }
}

/// The patch is in memory: loading leaves the file's pinned digest as it was.
#[test]
#[ignore = "requires installed Kokoro Q8"]
fn onnx_expose_is_idempotent_and_keeps_the_pinned_digest() {
    let root = voices().join("kokoro-en-q8");
    let before = local_tts_core::integrity::hash_file(&root, "model.onnx").unwrap();
    let _ = kokoro(Variant::Q8);
    assert_eq!(
        local_tts_core::integrity::hash_file(&root, "model.onnx").unwrap(),
        before
    );
    let bytes = std::fs::read(root.join("model.onnx")).unwrap();
    use local_tts_core::onnx_output::{ElemType, expose, outputs};
    let once = expose(
        &bytes,
        "/encoder/Clip_output_0",
        "durations",
        ElemType::Float,
    )
    .unwrap();
    assert_eq!(outputs(&once).unwrap(), ["waveform", "durations"]);
    let twice = expose(
        &once,
        "/encoder/Clip_output_0",
        "durations",
        ElemType::Float,
    )
    .unwrap();
    assert_eq!(once, twice);
}
