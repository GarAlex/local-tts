//! Optional real-model regression; run after scripts/download_cyrillic.py all.
use local_tts_core::{
    Audio, Language, SynthesisRequest, Synthesizer,
    piper::Piper,
    vosk::{SPEAKERS, Vosk},
};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};
fn check(audio: &Audio) {
    assert_eq!(audio.sample_rate, 22050);
    assert!(audio.samples.len() > 2205 && audio.samples.len() < 22050 * 60);
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    let rms = (audio
        .samples
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        / audio.samples.len() as f64)
        .sqrt();
    assert!(rms > 0.001, "silent model output: {rms}");
}
#[test]
#[ignore = "requires downloaded RU/UK model packages"]
fn real_russian_and_ukrainian_voices() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("target/pretrained-smoke");
    std::fs::create_dir_all(&out).unwrap();
    let mut metrics = Vec::new();
    let start = Instant::now();
    let mut ru = Vosk::load(&root.join("voices/vosk-ru")).unwrap();
    metrics.push(
        serde_json::json!({"backend":"vosk-ru", "load_seconds":start.elapsed().as_secs_f64()}),
    );
    assert_eq!(
        ru.tokenize("з+амок").unwrap(),
        ru.tokenize("за́мок").unwrap()
    );
    assert_ne!(
        ru.tokenize("з+амок").unwrap(),
        ru.tokenize("зам+ок").unwrap()
    );
    assert!(ru.tokenize("ї").is_err());
    assert_eq!(
        ru.tokenize("123").unwrap(),
        ru.tokenize("сто два́дцать три").unwrap()
    );
    let mut ru_waves = Vec::new();
    for speaker in SPEAKERS {
        let request = SynthesisRequest {
            text: "Привет, мир! Это русский голос.",
            language: Language::Ru,
            speaker,
            duration_scale: 1.0,
        };
        let start = Instant::now();
        let audio = ru.synthesize(&request).unwrap();
        check(&audio);
        audio
            .write_wav(&out.join(format!("ru-{speaker}.wav")))
            .unwrap();
        metrics.push(serde_json::json!({"backend":"vosk-ru", "speaker":speaker,"seconds":audio.samples.len() as f64/22050.0,"inference_seconds":start.elapsed().as_secs_f64()}));
        ru_waves.push(audio.samples);
        assert!(
            ru.synthesize_cancellable(&request, &AtomicBool::new(true))
                .is_err()
        );
    }
    assert!(ru_waves.windows(2).all(|pair| pair[0] != pair[1]));
    for (language, speaker, scale) in [
        (Language::Uk, "F03", 1.0),
        (Language::Ru, "unknown", 1.0),
        (Language::Ru, "F03", f32::NAN),
    ] {
        assert!(
            ru.synthesize(&SynthesisRequest {
                text: "Привет",
                language,
                speaker,
                duration_scale: scale
            })
            .is_err()
        );
    }
    let number_text = "21 3,05 1001.";
    let russian_numbers = ru
        .synthesize(&SynthesisRequest {
            text: number_text,
            language: Language::Ru,
            speaker: "F03",
            duration_scale: 1.0,
        })
        .unwrap();
    check(&russian_numbers);
    russian_numbers
        .write_wav(&out.join("ru-numbers.wav"))
        .unwrap();
    for text in ["12:30", "2026-10-02", "2кг"] {
        assert!(ru.tokenize(text).is_err());
    }
    let original = ru.tokenize("Валерия").unwrap();
    let explicit = ru.tokenize("вал+ерия").unwrap();
    let override_ids = ru.tokenize("валер+ия").unwrap();
    let mut corrections =
        local_tts_core::pronunciations::Pronunciations::new(Language::Ru).unwrap();
    corrections.insert("вал+ерия").unwrap();
    ru.set_pronunciations(corrections).unwrap();
    assert_eq!(ru.tokenize("Валерия").unwrap(), explicit);
    assert_eq!(ru.tokenize("валер+ия").unwrap(), override_ids);
    assert!(
        ru.set_pronunciations(
            local_tts_core::pronunciations::Pronunciations::new(Language::Uk).unwrap()
        )
        .is_err()
    );
    let audio = ru
        .synthesize(&SynthesisRequest {
            text: "Меня зовут Валерия.",
            language: Language::Ru,
            speaker: "F03",
            duration_scale: 1.0,
        })
        .unwrap();
    check(&audio);
    audio.write_wav(&out.join("ru-saved-stress.wav")).unwrap();
    ru.reload_pronunciations().unwrap();
    assert_eq!(ru.tokenize("Валерия").unwrap(), original);
    drop(ru);
    let start = Instant::now();
    let mut uk = Piper::load(&root.join("voices/piper-uk")).unwrap();
    metrics.push(
        serde_json::json!({"backend":"piper-uk", "load_seconds":start.elapsed().as_secs_f64()}),
    );
    assert_eq!(
        uk.tokenize("ґ+анок їй").unwrap(),
        uk.tokenize("ґа́нок і\u{308}и\u{306}").unwrap()
    );
    let speakers: Vec<_> = uk.speakers().map(str::to_owned).collect();
    assert_eq!(speakers, ["lada", "mykyta", "tetiana"]);
    for speaker in speakers {
        let request = SynthesisRequest {
            text: "Прив+іт, св+іте! Це укра+їнська м+ова.",
            language: Language::Uk,
            speaker: &speaker,
            duration_scale: 1.0,
        };
        let start = Instant::now();
        let audio = uk.synthesize(&request).unwrap();
        check(&audio);
        audio
            .write_wav(&out.join(format!("uk-{speaker}.wav")))
            .unwrap();
        metrics.push(serde_json::json!({"backend":"piper-uk", "speaker":speaker,"seconds":audio.samples.len() as f64/22050.0,"inference_seconds":start.elapsed().as_secs_f64()}));
        assert!(
            uk.synthesize_cancellable(&request, &AtomicBool::new(true))
                .is_err()
        );
    }
    for (language, speaker, scale) in [
        (Language::Ru, "lada", 1.0),
        (Language::Uk, "unknown", 1.0),
        (Language::Uk, "lada", 0.1),
    ] {
        assert!(
            uk.synthesize(&SynthesisRequest {
                text: "Привіт",
                language,
                speaker,
                duration_scale: scale
            })
            .is_err()
        );
    }
    let ukrainian_numbers = uk
        .synthesize(&SynthesisRequest {
            text: number_text,
            language: Language::Uk,
            speaker: "lada",
            duration_scale: 1.0,
        })
        .unwrap();
    check(&ukrainian_numbers);
    ukrainian_numbers
        .write_wav(&out.join("uk-numbers.wav"))
        .unwrap();
    for text in ["12:30", "2026-10-02", "2кг"] {
        assert!(uk.tokenize(text).is_err());
    }
    let original = uk.tokenize("Привіт").unwrap();
    let explicit = uk.tokenize("прив+іт").unwrap();
    let override_ids = uk.tokenize("пр+ивіт").unwrap();
    let mut corrections =
        local_tts_core::pronunciations::Pronunciations::new(Language::Uk).unwrap();
    corrections.insert("прив+іт").unwrap();
    uk.set_pronunciations(corrections).unwrap();
    assert_eq!(uk.tokenize("Привіт").unwrap(), explicit);
    assert_eq!(uk.tokenize("пр+ивіт").unwrap(), override_ids);
    assert!(
        uk.set_pronunciations(
            local_tts_core::pronunciations::Pronunciations::new(Language::Ru).unwrap()
        )
        .is_err()
    );
    let audio = uk
        .synthesize(&SynthesisRequest {
            text: "Привіт!",
            language: Language::Uk,
            speaker: "lada",
            duration_scale: 1.0,
        })
        .unwrap();
    check(&audio);
    audio.write_wav(&out.join("uk-saved-stress.wav")).unwrap();
    uk.reload_pronunciations().unwrap();
    assert_eq!(uk.tokenize("Привіт").unwrap(), original);
    let long = "Прив+іт, св+іте! ".repeat(30);
    check(
        &uk.synthesize(&SynthesisRequest {
            text: &long,
            language: Language::Uk,
            speaker: "lada",
            duration_scale: 1.0,
        })
        .unwrap(),
    );
    std::fs::write(
        out.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
}
