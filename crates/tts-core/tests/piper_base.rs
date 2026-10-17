//! Opt-in parity against the original Ukrainian Piper ONNX package.
use local_tts_core::{Language, SynthesisRequest, Synthesizer, piper::Piper, vits::Vits};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::path::PathBuf;
#[test]
#[ignore = "requires installed public Ukrainian training base exported by tests/ukrainian_base_smoke.py"]
fn imported_base_matches_original_piper_tokens_and_waveforms() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let voice = std::env::var_os("TTS_UK_BASE_VOICE")
        .map(PathBuf::from)
        .unwrap_or(root.join("target/public-uk-base-check/base-voice"));
    let piper = Piper::load(&root.join("voices/piper-uk")).unwrap();
    let mut original = Session::builder()
        .unwrap()
        .with_intra_threads(2)
        .unwrap()
        .commit_from_file(root.join("voices/piper-uk/model.onnx"))
        .unwrap();
    let mut imported = Vits::load(&voice).unwrap();
    assert_eq!(
        imported.manifest.frontend,
        local_tts_core::piper_training::VERSION
    );
    for text in ["Прив+іт, сві́те!", "Ґ Ї й є і — 12.", "Це украї́нська мо́ва."]
    {
        let ids = piper.tokenize(text).unwrap();
        let normalized = local_tts_core::piper_training::normalize(text, Language::Uk).unwrap();
        assert_eq!(
            ids,
            local_tts_core::piper_training::encode(&normalized).unwrap()
        );
        for scale in [0.8_f32, 1.0, 1.3] {
            let outputs=original.run(ort::inputs![
                "input" => Tensor::from_array(Array2::from_shape_vec((1,ids.len()),ids.clone()).unwrap()).unwrap(),
                "input_lengths" => Tensor::from_array(([1usize],vec![ids.len() as i64])).unwrap(),
                "scales" => Tensor::from_array(([3usize],vec![0.0_f32,scale,0.0])).unwrap(),
                "sid" => Tensor::from_array(([1usize],vec![0_i64])).unwrap()
            ]).unwrap();
            let (_, expected) = outputs[0].try_extract_tensor::<f32>().unwrap();
            let actual = imported
                .synthesize(&SynthesisRequest {
                    text,
                    language: Language::Uk,
                    speaker: "lada",
                    duration_scale: scale,
                })
                .unwrap();
            assert_eq!(
                actual.samples.len(),
                expected.len(),
                "length: {text} scale {scale}"
            );
            let max = actual
                .samples
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!("Original Piper parity: {text} scale {scale}: {max}");
            assert!(max < 0.001, "waveform parity: {max}");
        }
    }
}
