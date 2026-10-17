//! Opt-in original Russian dictionary/token contract checks, without model conversion assumptions.
use local_tts_core::{Language, SynthesisRequest, Synthesizer, vits::Vits, vosk};
use std::path::PathBuf;

#[test]
fn russian_internal_inventory_and_boundaries() {
    let symbols = vosk::training_symbols();
    assert_eq!(symbols.len(), 62);
    assert_eq!(symbols[0], '\0');
    assert_eq!(symbols[3], ' ');
    let text: String = [43, 46, 31, 16, 23, 51]
        .into_iter()
        .map(|id| symbols[id])
        .collect();
    assert_eq!(
        vosk::encode_training(&text).unwrap(),
        vec![1, 0, 43, 0, 46, 0, 31, 0, 16, 0, 23, 0, 51, 0, 2]
    );
    for bad in ["", " ", "привет", "\0", "\u{e001}", "\u{e002}", "\u{e03e}"] {
        assert!(vosk::encode_training(bad).is_err());
    }
    assert!(vosk::encode_training(&text.repeat(100)).is_err());
    assert!(local_tts_core::training::validate_symbols(&symbols, vosk::TRAINING_VERSION).is_ok());
    let mut wrong = symbols;
    wrong.swap(15, 16);
    assert!(local_tts_core::training::validate_symbols(&wrong, vosk::TRAINING_VERSION).is_err());
}

#[test]
#[ignore = "requires Russian dictionary, original training checkpoint and exported base"]
fn imported_base_shares_original_russian_frontend() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let front = vosk::Frontend::load(&root.join("voices/vosk-ru/dictionary")).unwrap();
    let mut own = Vits::load(&root.join("target/public-ru-base-check/base-voice")).unwrap();
    for text in [
        "Прив+ет, м+ир!",
        "Привет мир.",
        "Это р+усский яз+ык. У меня 12 кни́г.",
        "Ёж и ёлка.",
    ] {
        assert_eq!(
            front.tokenize(text).unwrap(),
            vosk::encode_training(&front.normalize_training(text).unwrap()).unwrap()
        );
        let audio = own
            .synthesize(&SynthesisRequest {
                text,
                language: Language::Ru,
                speaker: "F01",
                duration_scale: 1.0,
            })
            .unwrap();
        assert_eq!(audio.sample_rate, 22050);
        assert!(!audio.samples.is_empty() && audio.samples.iter().all(|s| s.is_finite()));
    }
    assert_eq!(
        front.tokenize("Прив+ет!").unwrap(),
        front.tokenize("Приве́т!").unwrap()
    );
    for text in ["abcdefgh", "абракадабрашка", "^", "\u{e00f}"] {
        assert!(front.tokenize(text).is_err());
    }
    // Long paragraphs chunk after shared phone preparation rather than hit the short-input token limit.
    let long = "Прив+ет, м+ир! ".repeat(40);
    assert!(front.tokenize(&long).is_err());
    assert!(
        own.synthesize(&SynthesisRequest {
            text: &long,
            language: Language::Ru,
            speaker: "F01",
            duration_scale: 0.8
        })
        .is_ok()
    );
}

#[test]
#[ignore = "requires publisher-reference.json emitted by tests/russian_base_smoke.py"]
fn source_generator_matches_publisher_onnx_with_its_unnormalized_inverse() {
    use ndarray::Array2;
    use ort::{session::Session, value::Tensor};
    #[derive(serde::Deserialize)]
    struct Reference {
        tokens: Vec<i64>,
        duration_scale: f32,
        samples: Vec<f32>,
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let references: Vec<Reference> = serde_json::from_slice(
        &std::fs::read(root.join("target/public-ru-base-check/publisher-reference.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(references.len(), 6);
    let mut original = Session::builder()
        .unwrap()
        .with_intra_threads(2)
        .unwrap()
        .commit_from_file(root.join("voices/vosk-ru/model.onnx"))
        .unwrap();
    for reference in references {
        let outputs=original.run(ort::inputs![
            "input" => Tensor::from_array(Array2::from_shape_vec((1,reference.tokens.len()),reference.tokens.clone()).unwrap()).unwrap(),
            "input_lengths" => Tensor::from_array(([1usize],vec![reference.tokens.len() as i64])).unwrap(),
            "scales" => Tensor::from_array(([3usize],vec![0.0_f32,reference.duration_scale,0.0])).unwrap(),
            "sid" => Tensor::from_array(([1usize],vec![0_i64])).unwrap()
        ]).unwrap();
        let (_, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(actual.len(), reference.samples.len());
        let error = actual
            .iter()
            .zip(reference.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "Original Vosk unnormalized conversion parity: {} tokens scale {}: {error}",
            reference.tokens.len(),
            reference.duration_scale
        );
        assert!(
            error < 0.001,
            "source/publisher conversion differs: {error}"
        );
    }
}
