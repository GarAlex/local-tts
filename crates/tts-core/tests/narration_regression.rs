//! Narration lines reported from real use, run through one Kokoro engine
//! repeatedly the way PromoShot's server does.
use local_tts_core::{Language, SynthesisRequest, Synthesizer, kokoro::Kokoro};
use std::path::Path;

#[test]
#[ignore = "requires installed Kokoro Q8"]
fn reported_lines_synthesize_every_time_and_identically() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/kokoro-en-q8");
    let mut engine = Kokoro::load(&root).unwrap();
    engine.set_fallback(true).unwrap();
    // Reported failing about 1 call in 15 with "model produced invalid audio".
    let text = "The nor'easter hit at 3:19 a.m. Details at promoshot.app/demo.";
    let mut lengths = std::collections::BTreeSet::new();
    for _ in 0..20 {
        let audio = engine
            .synthesize(&SynthesisRequest {
                text,
                language: Language::En,
                speaker: "af_heart",
                duration_scale: 1.0,
            })
            .unwrap();
        assert!(audio.samples.iter().all(|s| s.is_finite()));
        lengths.insert(audio.samples.len());
    }
    assert_eq!(lengths.len(), 1, "same text, same audio: {lengths:?}");
}
