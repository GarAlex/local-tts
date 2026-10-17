use local_tts_core::{Language, math_latent::MathLatent, math_speech::words_for_language};
use std::{path::Path, sync::atomic::AtomicBool};
#[test]
#[ignore = "requires experimental per-entry RU/UK packages"]
fn per_entry_joins_match_independent_decodes_and_retain_repeated_digits() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("checkpoints");
    for (language, folder) in [
        (Language::Ru, "math-latent-ru-per-entry-q4-v3"),
        (Language::Uk, "math-latent-uk-per-entry-q8-v3"),
    ] {
        let path = root.join(folder);
        let mut voice = MathLatent::load(&path).unwrap();
        assert_eq!(voice.verify(&path).unwrap(), 0.0);
        let cancel = AtomicBool::new(false);
        let words = words_for_language("12 + 12", language).unwrap();
        for duration in [0.7, 1.0, 1.4] {
            let mut expected = Vec::new();
            for word in &words {
                let mut part = voice
                    .synthesize_words(std::slice::from_ref(word), duration, &cancel)
                    .unwrap()
                    .samples;
                let count = part.len();
                let fade = 66.min(count / 2);
                for i in 0..fade {
                    let gain = i as f32 / fade as f32;
                    part[i] *= gain;
                    part[count - 1 - i] *= gain;
                }
                expected.extend(part);
            }
            let actual = voice.synthesize_words(&words, duration, &cancel).unwrap();
            assert_eq!(actual.samples, expected);
        }
        let words = words_for_language("0.009", language).unwrap();
        let total: usize = words
            .iter()
            .map(|word| {
                voice
                    .synthesize_words(std::slice::from_ref(word), 1.0, &cancel)
                    .unwrap()
                    .samples
                    .len()
            })
            .sum();
        let actual = voice.synthesize_words(&words, 1.0, &cancel).unwrap();
        assert_eq!(
            actual.samples.len(),
            total + (45.0f32 * 22.05).round() as usize
        );
        assert!(actual.samples.iter().all(|s| s.is_finite()));
        assert!(
            voice
                .synthesize_words(&words, 1.0, &AtomicBool::new(true))
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires schema-3 double-zero RU/UK packages"]
fn double_zero_cache_keeps_fractional_digit_counts_and_native_geometry() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("checkpoints");
    for (language, folder) in [
        (Language::Ru, "math-latent-ru-double-zero-q4-v4"),
        (Language::Uk, "math-latent-uk-double-zero-q8-v4"),
    ] {
        let path = root.join(folder);
        let mut voice = MathLatent::load(&path).unwrap();
        assert_eq!(voice.verify(&path).unwrap(), 0.0);
        let bytes: u64 = std::fs::read_dir(&path)
            .unwrap()
            .map(|p| p.unwrap().metadata().unwrap().len())
            .sum();
        assert!(bytes < 10_000_000);
        let cancel = AtomicBool::new(false);
        let zero = words_for_language("0", language).unwrap()[0].clone();
        let point = words_for_language("0.0", language).unwrap()[1].clone();
        let pair = format!("{zero} {zero}");
        let zero_frames = voice
            .synthesize_words(std::slice::from_ref(&zero), 1.0, &cancel)
            .unwrap()
            .samples
            .len();
        let point_frames = voice
            .synthesize_words(&[point], 1.0, &cancel)
            .unwrap()
            .samples
            .len();
        let pair_frames = voice
            .synthesize_words(&[pair], 1.0, &cancel)
            .unwrap()
            .samples
            .len();
        for (literal, pairs) in [("0.00", 1), ("0.0000", 2)] {
            let words = words_for_language(literal, language).unwrap();
            assert_eq!(words.len(), 2 + pairs * 2);
            let audio = voice.synthesize_words(&words, 1.0, &cancel).unwrap();
            assert_eq!(
                audio.samples.len(),
                zero_frames + point_frames + pair_frames * pairs
            );
            assert!(audio.samples.iter().all(|s| s.is_finite()));
        }
    }
}
