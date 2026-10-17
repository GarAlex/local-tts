use local_tts_core::{
    Audio, Language, SynthesisRequest, Synthesizer,
    math_speech::{MathVoice, words},
};
#[test]
fn arithmetic_readings_and_invalid_syntax() {
    assert_eq!(
        words("12 + 225 = 237").unwrap().join(" "),
        "twelve plus two hundred twenty five equals two hundred thirty seven"
    );
    assert_eq!(
        words("-12.05 × (2 - 1) ÷ +3").unwrap().join(" "),
        "negative twelve point zero five times open parenthesis two minus one close parenthesis divided by positive three"
    );
    // Literal speech preserves a mathematically false equation.
    assert!(words("1+1=3").is_ok());
    for invalid in [
        "",
        "1+",
        "2 3",
        "2(3)",
        "1..2",
        "1.",
        "(1+2",
        "1=2=3",
        "1e3",
        "--2",
        "1000000000",
        "hello",
        "1^2",
    ] {
        assert!(words(invalid).is_err(), "{invalid}");
    }
}
struct Tone;
impl Synthesizer for Tone {
    type Error = std::io::Error;
    fn synthesize(&mut self, _: &SynthesisRequest<'_>) -> Result<Audio, Self::Error> {
        Ok(Audio {
            sample_rate: 24000,
            samples: (0..4800).map(|i| (i as f32 * 0.1).sin() * 0.5).collect(),
        })
    }
}
#[test]
fn portable_bank_integrity_and_request_validation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("bank");
    let size = MathVoice::build(&root, &mut Tone, "default", "test fixture").unwrap();
    assert!(size < 1_000_000);
    let mut bank = MathVoice::load(&root).unwrap();
    let mut request = SynthesisRequest {
        text: "12 + 225 = 237",
        language: Language::En,
        speaker: "default",
        duration_scale: 1.0,
    };
    let audio = bank.synthesize(&request).unwrap();
    assert_eq!(audio.sample_rate, 16000);
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    assert!(audio.samples.iter().any(|v| v.abs() > 0.01));
    request.language = Language::Ru;
    assert!(bank.synthesize(&request).is_err());
    assert!(MathVoice::build(&root, &mut Tone, "default", "test").is_err());
    let path = root.join("words.pcm");
    let mut pcm = std::fs::read(&path).unwrap();
    pcm[0] ^= 1;
    std::fs::write(path, pcm).unwrap();
    assert!(MathVoice::load(&root).is_err());
}

#[test]
#[ignore = "requires the experimental step-100 arithmetic student export"]
fn real_student_uses_shared_parser_and_validates_requests() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../checkpoints/math-student-en-v1/voice-step100");
    let mut voice = local_tts_core::math_speech::MathStudent::load(&root).unwrap();
    let mut request = SynthesisRequest {
        text: "807 + 694 = 1501",
        language: Language::En,
        speaker: "default",
        duration_scale: 1.0,
    };
    assert!(
        voice
            .synthesize_cancellable(&request, &std::sync::atomic::AtomicBool::new(true))
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    let audio = voice.synthesize(&request).unwrap();
    assert_eq!(audio.sample_rate, 22050);
    assert!(!audio.samples.is_empty());
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    request.text = "1+";
    assert!(voice.synthesize(&request).is_err());
    request.text = "1+2";
    request.language = Language::Ru;
    assert!(voice.synthesize(&request).is_err());
}

#[test]
fn curriculum_is_unique_reproducible_and_covers_each_split() {
    use std::collections::BTreeSet;
    let lines = local_tts_core::math_speech::corpus(256).unwrap();
    assert_eq!(
        serde_json::to_string(&lines).unwrap(),
        serde_json::to_string(&local_tts_core::math_speech::corpus(256).unwrap()).unwrap()
    );
    assert_eq!(
        lines
            .iter()
            .map(|l| &l.spoken)
            .collect::<BTreeSet<_>>()
            .len(),
        256
    );
    for split in [
        local_tts_core::dataset::Split::Train,
        local_tts_core::dataset::Split::Validation,
        local_tts_core::dataset::Split::Test,
    ] {
        let group: Vec<_> = lines
            .iter()
            .filter(|l| std::mem::discriminant(&l.split) == std::mem::discriminant(&split))
            .collect();
        for word in [
            "plus",
            "minus",
            "times",
            "divided",
            "equals",
            "negative",
            "positive",
            "point",
            "parenthesis",
            "million",
        ] {
            assert!(
                group
                    .iter()
                    .any(|l| l.spoken.split_whitespace().any(|w| w == word)),
                "{split:?} missing {word}"
            );
        }
    }
    assert!(local_tts_core::math_speech::corpus(127).is_err());
}
#[test]
fn short_curriculum_is_unique_reproducible_and_reserves_each_structure() {
    use local_tts_core::{dataset::Split, math_speech::short_corpus};
    use std::collections::BTreeSet;
    let lines = short_corpus(384).unwrap();
    let again = short_corpus(384).unwrap();
    assert_eq!(
        lines.iter().map(|l| &l.expression).collect::<Vec<_>>(),
        again.iter().map(|l| &l.expression).collect::<Vec<_>>()
    );
    assert_eq!(
        lines
            .iter()
            .map(|l| &l.spoken)
            .collect::<BTreeSet<_>>()
            .len(),
        384
    );
    assert!(
        lines
            .iter()
            .all(|l| l.spoken.split_whitespace().count() <= 5)
    );
    for split in [Split::Train, Split::Validation, Split::Test] {
        let group: Vec<_> = lines
            .iter()
            .filter(|l| std::mem::discriminant(&l.split) == std::mem::discriminant(&split))
            .collect();
        for operator in ["plus", "minus", "times"] {
            assert!(group.iter().any(|l| l.spoken.contains(operator)));
        }
        assert!(group.iter().any(|l| l.expression.parse::<u32>().is_ok()));
    }
    assert!(short_corpus(127).is_err());
    assert!(short_corpus(385).is_err());
}
