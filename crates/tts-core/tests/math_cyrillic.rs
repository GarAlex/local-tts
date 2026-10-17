use local_tts_core::{
    Language,
    math_speech::{vocabulary_for_language, words_for_language},
};
#[test]
fn literal_cyrillic_arithmetic_preserves_numbers_and_signs() {
    assert_eq!(
        words_for_language("12 + 225 = 237", Language::Ru)
            .unwrap()
            .join(" "),
        "двена́дцать плюс две́сти два́дцать пять равно́ две́сти три́дцать семь"
    );
    assert_eq!(
        words_for_language("12 + 225 = 237", Language::Uk)
            .unwrap()
            .join(" "),
        "двана́дцять плюс дві́сті два́дцять п'ять дорі́внює дві́сті три́дцять сім"
    );
    for language in [Language::Ru, Language::Uk] {
        let vocabulary = vocabulary_for_language(language).unwrap();
        for expression in [
            "-12 + 225 = 237",
            "(2 * 3) / 4",
            "1001001 + 2002002 = 999999999",
            "0.009 + 670.70 = 670.70",
        ] {
            let words = words_for_language(expression, language).unwrap();
            assert!(
                words.iter().all(|w| vocabulary.contains(w)),
                "{language:?}: {words:?}"
            );
        }
        let fractional = words_for_language("0.009", language).unwrap();
        assert_eq!(fractional.len(), 5);
        assert_eq!(fractional[2], fractional[3]);
        for invalid in [
            "1..2",
            "1 +",
            "= 2",
            "1 = 2 = 3",
            "1 2",
            "1000000000",
            "1e3",
        ] {
            assert!(words_for_language(invalid, language).is_err());
        }
    }
    assert!(words_for_language("12", Language::Fr).is_err());
}

#[test]
#[ignore = "requires exported RU/UK numeric packages"]
fn cyrillic_neural_packages_run_through_public_api() {
    use local_tts_core::{
        SynthesisRequest, Synthesizer, math_latent::MathLatent, math_speech::MathStudent,
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (language, folder) in [
        (Language::Ru, "math-latent-ru-q4-v1"),
        (Language::Uk, "math-latent-uk-q8-v1"),
    ] {
        let path = root.join("checkpoints").join(folder);
        let mut latent = MathLatent::load(&path).unwrap();
        assert_eq!(latent.verify(&path).unwrap(), 0.0);
        let bytes: u64 = std::fs::read_dir(&path)
            .unwrap()
            .map(|f| f.unwrap().metadata().unwrap().len())
            .sum();
        assert!(bytes < 10_000_000, "{folder}: {bytes}");
        let mut voice = MathStudent::load(&path).unwrap();
        assert_eq!(voice.language(), language);
        for text in ["12 + 225 = 237", "(2 * 3) / 4", "0.009 + 1001001"] {
            let audio = voice
                .synthesize(&SynthesisRequest {
                    text,
                    language,
                    speaker: "default",
                    duration_scale: 1.0,
                })
                .unwrap();
            assert_eq!(audio.sample_rate, 22050);
            assert!(!audio.samples.is_empty());
            assert!(audio.samples.iter().all(|s| s.is_finite()));
            assert!(audio.samples.iter().any(|s| s.abs() > 0.01));
        }
        assert!(
            voice
                .synthesize(&SynthesisRequest {
                    text: "12",
                    language: Language::En,
                    speaker: "default",
                    duration_scale: 1.0
                })
                .is_err()
        );
    }
}
