use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    compact_english::{CompactEnglish, CompactKind},
};
use std::{path::Path, sync::atomic::AtomicBool};
#[test]
#[ignore = "requires pinned Inflect and Kitten English assets"]
fn real_compact_models_voices_validation_and_cancellation() {
    for (kind, folder) in [
        (CompactKind::Inflect, "inflect-nano-en"),
        (CompactKind::Kitten, "kitten-nano-en"),
    ] {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../voices")
            .join(folder);
        let mut model = CompactEnglish::load(&root, kind).unwrap();
        let mut request = SynthesisRequest {
            text: "Hello world. Welcome to the studio.",
            language: Language::En,
            speaker: "default",
            duration_scale: 1.0,
        };
        assert!(
            model
                .synthesize_cancellable(&request, &AtomicBool::new(true))
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
        request.language = Language::Uk;
        assert!(model.synthesize(&request).is_err());
        request.language = Language::En;
        request.speaker = "missing";
        assert!(model.synthesize(&request).is_err());
        request.speaker = "default";
        let voices = if kind == CompactKind::Kitten {
            vec![
                "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
            ]
        } else {
            vec!["default"]
        };
        for voice in voices {
            request.speaker = voice;
            let audio = model.synthesize(&request).unwrap();
            assert_eq!(audio.sample_rate, 24000);
            assert!(audio.samples.len() > 2400);
            assert!(audio.samples.iter().all(|v| v.is_finite()));
            assert!(audio.samples.iter().any(|v| v.abs() > 0.001));
        }
        let dictionary =
            local_tts_core::english::English::load(&root.join("cmudict.dict")).unwrap();
        let long = format!("{}2.5 kg is the weight.", "Hello. ".repeat(33));
        for text in [
            "On 2026-10-03 at 12:30, Dr. Smith paid $5.20 for 2.5 kg. It was 5% off on the 21st day.",
            "I have 123 items and 1/2 of a kilogram.",
            &long,
        ] {
            let spoken = local_tts_core::text::expand(text, Language::En).unwrap();
            // Compare the deterministic frontend, not stochastic graph output.
            assert_eq!(
                dictionary.phonemize_ipa_current(text).unwrap(),
                dictionary.phonemize_ipa_current(&spoken).unwrap()
            );
            let request = SynthesisRequest {
                text,
                language: Language::En,
                speaker: "default",
                duration_scale: 1.0,
            };
            let audio = model.synthesize(&request).unwrap();
            assert!(audio.samples.len() > 2400);
            assert!(audio.samples.iter().all(|v| v.is_finite()));
            assert!(audio.samples.iter().any(|v| v.abs() > 0.001));
        }
        request.text = "xyzzyunknownword";
        assert!(model.synthesize(&request).is_err());
    }
}
