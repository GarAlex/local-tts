use local_tts_core::{Language, frontend, prompts::PromptList, text};

#[test]
fn stable_language_ids_and_accented_text() {
    for (index, language) in Language::ALL.into_iter().enumerate() {
        assert_eq!(language.abi_id(), index as u32);
        assert_eq!(Language::from_abi(index as u32).unwrap(), language);
        assert_eq!(language.code().parse::<Language>().unwrap(), language);
        assert_eq!(
            serde_json::from_str::<Language>(&format!("\"{}\"", language.code())).unwrap(),
            language
        );
        assert_eq!(PromptList::starter(language).lines.len(), 12);
    }
    assert!(Language::from_abi(10).is_err());
    assert!("unknown".parse::<Language>().is_err());
    for (language, source, expected) in [
        (
            Language::Fr,
            "  ÉCOLE  cœur, Noe\u{308}l. ",
            "école cœur, noël.",
        ),
        (Language::De, "GRÜẞE", "grüße"),
        (Language::Es, "NIÑO, CORAZÓN", "niño, corazón"),
        (Language::It, "CITTÀ E CAFFÈ", "città e caffè"),
        (Language::Pt, "AÇÃO E PÃO", "ação e pão"),
        (Language::Pl, "ZAŻÓŁĆ GĘŚLĄ JAŹŃ", "zażółć gęślą jaźń"),
        (Language::Nl, "ÉÉN, IDEEËN", "één, ideeën"),
    ] {
        assert_eq!(frontend::normalize(source, language).unwrap(), expected);
        assert_eq!(text::prepare(source, language).unwrap(), expected);
        assert!(local_tts_core::pronunciations::Pronunciations::new(language).is_err());
    }
}

#[test]
fn all_added_starter_prompts_prepare_without_losing_accents() {
    for language in Language::ALL
        .into_iter()
        .filter(|l| l.is_european_extension())
    {
        for prompt in PromptList::starter(language).lines {
            assert_eq!(
                text::prepare(&prompt, language).unwrap(),
                frontend::normalize(&prompt, language).unwrap()
            );
        }
        assert!(text::prepare("", language).is_err());
        assert!(text::prepare("\0", language).is_err());
        assert!(text::prepare(&"é".repeat(10001), language).is_err());
    }
}

#[test]
#[ignore = "requires pinned Chatterbox assets (1.56 GB)"]
fn french_native_inference_and_cancellation() {
    use local_tts_core::{SynthesisRequest, Synthesizer, chatterbox::Chatterbox};
    use std::{path::Path, sync::atomic::AtomicBool};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/chatterbox-multilingual");
    let mut model = Chatterbox::load(&root).unwrap();
    let mut request = SynthesisRequest {
        text: "Bonjour, le soleil entre par la fenêtre.",
        language: Language::Fr,
        speaker: "default",
        duration_scale: 1.0,
    };
    let error = model
        .synthesize_cancellable(&request, &AtomicBool::new(true))
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    request.language = Language::Uk;
    assert!(
        model
            .synthesize(&request)
            .unwrap_err()
            .to_string()
            .contains("Ukrainian")
    );
    request.language = Language::Fr;
    request.duration_scale = 0.8;
    assert!(model.synthesize(&request).is_err());
    request.duration_scale = 1.0;
    let cancellation = std::sync::Arc::new(AtomicBool::new(false));
    let trigger = cancellation.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(100));
        trigger.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let error = model
        .synthesize_cancellable(&request, &cancellation)
        .unwrap_err();
    thread.join().unwrap();
    assert!(error.to_string().contains("cancelled"));
    let mut metrics = Vec::new();
    for language in [
        Language::Fr,
        Language::De,
        Language::Es,
        Language::It,
        Language::Pt,
        Language::Pl,
        Language::Nl,
    ] {
        let text = PromptList::starter(language).lines[0].clone();
        let start = std::time::Instant::now();
        let audio = model
            .synthesize(&SynthesisRequest {
                text: &text,
                language,
                speaker: "default",
                duration_scale: 1.0,
            })
            .unwrap();
        assert_eq!(audio.sample_rate, 24000);
        assert!(audio.samples.len() > 24000);
        assert!(audio.samples.iter().all(|s| s.is_finite()));
        assert!(audio.samples.iter().any(|s| s.abs() > 0.001));
        let output = root
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("target")
            .join(format!("{}-chatterbox.wav", language.code()));
        audio.write_wav(&output).unwrap();
        metrics.push(serde_json::json!({"language":language.code(),"seconds":audio.samples.len() as f64/24000.0,"generation_seconds":start.elapsed().as_secs_f64(),"file":output,"text":text}));
        println!("{}: native synthesis passed", language.code());
    }
    std::fs::write(
        root.parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("target/european-native-metrics.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
}

#[test]
fn neon_starter_prompts_use_the_pinned_language_inventory() {
    for language in [
        Language::Fr,
        Language::It,
        Language::Pt,
        Language::Pl,
        Language::Nl,
    ] {
        for prompt in PromptList::starter(language).lines {
            let normalized = local_tts_core::neon::normalize(&prompt, language).unwrap();
            let tokens = local_tts_core::neon::encode(&normalized, language).unwrap();
            assert!(tokens.iter().all(|id| *id >= 3
                && (*id as usize) < local_tts_core::neon::symbols(language).unwrap().len()));
        }
    }
}

#[test]
#[ignore = "requires pinned Chatterbox assets (1.56 GB)"]
fn custom_speech_reference_english_and_russian() {
    use local_tts_core::{SynthesisRequest, Synthesizer, chatterbox::Chatterbox};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../voices/chatterbox-multilingual");
    // An explicit publisher speech reference exercises conditioning, not a new
    // speaker identity or consent/similarity evaluation.
    let mut model =
        Chatterbox::load_with_reference(&root, Some(&root.join("default_voice.wav"))).unwrap();
    assert!(model.set_exaggeration(f32::NAN).is_err());
    assert!(model.set_exaggeration(1.1).is_err());
    model.set_exaggeration(0.7).unwrap();
    for (language, text) in [(Language::En, "Hello."), (Language::Ru, "Привет.")] {
        let audio = model
            .synthesize(&SynthesisRequest {
                text,
                language,
                speaker: "default",
                duration_scale: 1.0,
            })
            .unwrap();
        assert_eq!(audio.sample_rate, 24000);
        assert!(audio.samples.len() >= 2400);
        assert!(audio.samples.iter().all(|s| s.is_finite()));
        assert!(audio.samples.iter().any(|s| s.abs() > 0.001));
        audio
            .write_wav(&root.join(format!("../../target/clone-{}.wav", language.code())))
            .unwrap();
    }
}
