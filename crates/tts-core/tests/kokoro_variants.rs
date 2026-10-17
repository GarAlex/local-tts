//! Actual pinned FP32/Q8/Q4 CPU inference; artifacts support listening comparison.
use local_tts_core::{
    Audio, Language, SynthesisRequest, Synthesizer,
    kokoro::{Kokoro, Variant},
};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

fn stats(audio: &Audio) -> serde_json::Value {
    assert_eq!(audio.sample_rate, 24000);
    assert!((2400..24000 * 60).contains(&audio.samples.len()));
    assert!(audio.samples.iter().all(|v| v.is_finite()));
    let rms = (audio
        .samples
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        / audio.samples.len() as f64)
        .sqrt();
    assert!(rms > 0.001, "silent output: {rms}");
    let peak = audio
        .samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(peak < 4.0, "unstable output: {peak}");
    serde_json::json!({"audio_seconds":audio.samples.len() as f64 / 24000.0, "rms":rms,"peak":peak,
        "samples_above_one":audio.samples.iter().filter(|v| v.abs() > 1.0).count()})
}
#[test]
#[ignore = "requires scripts/download_english.py all"]
fn actual_kokoro_variants() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("target/kokoro-comparison");
    std::fs::create_dir_all(&out).unwrap();
    let mut results = Vec::new();
    let texts = [
        "Hello world. This is a local speech test.",
        "I have 21 apples and 3.05 dollars.",
        "The quick brown fox jumps over the lazy dog.",
    ];
    let mut reference_phonemes = None;
    for variant in Variant::ALL {
        let folder = root.join("voices").join(variant.folder());
        let start = Instant::now();
        let mut engine = Kokoro::load(&folder).unwrap();
        let load_seconds = start.elapsed().as_secs_f64();
        assert_eq!(engine.variant(), variant);
        let phonemes = engine.phonemize(texts[0]).unwrap();
        if let Some(reference) = &reference_phonemes {
            assert_eq!(&phonemes, reference);
        } else {
            reference_phonemes = Some(phonemes);
        }
        let request = |text, duration_scale| SynthesisRequest {
            text,
            language: Language::En,
            speaker: "af_heart",
            duration_scale,
        };
        // Warm-up is excluded from timings. Each subsequent model runs the same cases.
        stats(&engine.synthesize(&request(texts[0], 1.0)).unwrap());
        let mut cases = Vec::new();
        for (index, text) in texts.iter().enumerate() {
            let mut times = Vec::new();
            for repeat in 0..3 {
                let start = Instant::now();
                let audio = engine.synthesize(&request(text, 1.0)).unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                let mut values = stats(&audio);
                values["inference_seconds"] = elapsed.into();
                values["rtf"] = (elapsed / (audio.samples.len() as f64 / 24000.0)).into();
                times.push(values);
                if repeat == 0 {
                    audio
                        .write_wav(&out.join(format!("{}-{index}.wav", variant.id())))
                        .unwrap();
                }
            }
            cases.push(serde_json::json!({"text":text,"runs":times}));
        }
        let fast = engine.synthesize(&request(texts[0], 0.8)).unwrap();
        let slow = engine.synthesize(&request(texts[0], 1.3)).unwrap();
        stats(&fast);
        stats(&slow);
        assert!(slow.samples.len() > fast.samples.len());
        let paragraph = texts[0].repeat(15);
        let audio = engine.synthesize(&request(&paragraph, 1.0)).unwrap();
        let paragraph_stats = stats(&audio);
        audio
            .write_wav(&out.join(format!("{}-paragraph.wav", variant.id())))
            .unwrap();
        assert!(
            engine
                .synthesize_cancellable(&request(texts[0], 1.0), &AtomicBool::new(true))
                .is_err()
        );
        for (language, speaker, duration_scale) in [
            (Language::Ru, "af_heart", 1.0),
            (Language::En, "unknown", 1.0),
            (Language::En, "af_heart", f32::NAN),
        ] {
            assert!(
                engine
                    .synthesize(&SynthesisRequest {
                        text: texts[0],
                        language,
                        speaker,
                        duration_scale
                    })
                    .is_err()
            );
        }
        assert!(engine.synthesize(&request("zzzzunknownword", 1.0)).is_err());
        let values = serde_json::json!({"variant":variant.id(),"model_bytes":std::fs::metadata(folder.join("model.onnx")).unwrap().len(),
            "load_seconds":load_seconds,"cases":cases,"paragraph":paragraph_stats,"duration_controls_passed":true,"cancellation_and_invalid_inputs_passed":true});
        println!("{values}");
        results.push(values);
    }
    std::fs::write(out.join("metrics.json"),serde_json::to_vec_pretty(&serde_json::json!({"platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "threads":2,"build":"debug","warmup":1,"variants":results,"quality_note":"Finite, non-silent PCM and controls tested. Listening quality is not validated; quantized outputs need not match FP32."})).unwrap()).unwrap();
}
