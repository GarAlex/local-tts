use local_tts_core::{
    Language, SynthesisRequest, Synthesizer, math_latent::MathLatent, math_speech::MathStudent,
};
use std::path::PathBuf;

fn root() -> PathBuf {
    std::env::var_os("TTS_MATH_LATENT_TEST_VOICE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../checkpoints/math-latent-context-en-v1")
        })
}

#[test]
#[ignore = "requires locally exported arithmetic latent voice"]
fn native_parity_and_numeric_contract() {
    let path = root();
    let mut decoder = MathLatent::load(&path).unwrap();
    assert!(decoder.verify(&path).unwrap() < 1e-4);
    let mut voice = MathStudent::load(&path).unwrap();
    let request = SynthesisRequest {
        text: "12 + 225 = 237",
        language: Language::En,
        speaker: "default",
        duration_scale: 1.0,
    };
    let audio = voice.synthesize(&request).unwrap();
    assert_eq!(audio.sample_rate, 22050);
    assert!(audio.samples.len() > 22050);
    assert!(audio.samples.iter().all(|x| x.is_finite()));
    let rms =
        (audio.samples.iter().map(|v| v * v).sum::<f32>() / audio.samples.len() as f32).sqrt();
    assert!(rms > 0.01);
    assert!(
        voice
            .synthesize(&SynthesisRequest {
                text: "hello",
                ..request
            })
            .is_err()
    );
    assert!(
        voice
            .synthesize(&SynthesisRequest {
                duration_scale: f32::NAN,
                ..request
            })
            .is_err()
    );
    let cancelled = std::sync::atomic::AtomicBool::new(true);
    assert!(voice.synthesize_cancellable(&request, &cancelled).is_err());
    let bytes: u64 = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(
        bytes < 10_000_000,
        "package {bytes} exceeds requested 10 MB"
    );
}

#[test]
#[ignore = "requires locally exported arithmetic latent voice"]
fn reject_modified_conditioning() {
    let original = root();
    let temp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(&original).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), temp.path().join(entry.file_name())).unwrap();
    }
    let path = temp.path().join("conditioning.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(path, bytes).unwrap();
    assert!(
        MathLatent::load(temp.path())
            .err()
            .unwrap()
            .to_string()
            .contains("checksum mismatch")
    );
}

#[test]
#[ignore = "requires locally exported arithmetic latent voice"]
fn reject_invalid_resealed_conditioning_geometry() {
    use sha2::{Digest, Sha256};
    let original = root();
    let temp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(&original).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), temp.path().join(entry.file_name())).unwrap();
    }
    let base: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("arithmetic-neural.json")).unwrap())
            .unwrap();
    for invalid in ["offset", "frames", "scale"] {
        let mut manifest = base.clone();
        match invalid {
            "offset" => manifest["entries"]["one"]["offset"] = serde_json::json!(usize::MAX),
            "frames" => manifest["entries"]["one"]["frames"] = serde_json::json!(0),
            _ => {
                if manifest["schema_version"] == 2 {
                    manifest["entries"]["one"]["scales_offset"] = serde_json::json!(3);
                } else {
                    manifest["entries"]["one"]["scales"][0] = serde_json::json!(0.0);
                }
            }
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(temp.path().join("arithmetic-neural.json"), &bytes).unwrap();
        let mut hashes: serde_json::Value =
            serde_json::from_slice(&std::fs::read(original.join("checksums.json")).unwrap())
                .unwrap();
        hashes["arithmetic-neural.json"] =
            serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
        std::fs::write(
            temp.path().join("checksums.json"),
            serde_json::to_vec(&hashes).unwrap(),
        )
        .unwrap();
        assert!(
            MathLatent::load(temp.path()).is_err(),
            "accepted invalid {invalid}"
        );
    }
}

#[test]
#[ignore = "requires a schema-2 compound voice selected with TTS_MATH_LATENT_TEST_VOICE"]
fn compound_numbers_keep_fractional_digits_literal() {
    let path = root();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("arithmetic-neural.json")).unwrap())
            .unwrap();
    if manifest["schema_version"] != 2 {
        return;
    }
    let entries = &manifest["entries"];
    let frames = |word: &str| entries[word]["frames"].as_u64().unwrap() as usize;
    let mut voice = MathStudent::load(&path).unwrap();
    let request = SynthesisRequest {
        text: "74",
        language: Language::En,
        speaker: "default",
        duration_scale: 1.0,
    };
    assert_eq!(
        voice.synthesize(&request).unwrap().samples.len(),
        frames("seventy four") * 256
    );
    assert_eq!(
        voice
            .synthesize(&SynthesisRequest {
                text: "74 + 74",
                ..request
            })
            .unwrap()
            .samples
            .len(),
        (2 * frames("seventy four") + frames("plus")) * 256
    );
    assert_eq!(
        voice
            .synthesize(&SynthesisRequest {
                text: "0.74",
                ..request
            })
            .unwrap()
            .samples
            .len(),
        ["zero", "point", "seven", "four"]
            .iter()
            .map(|w| frames(w) * 256)
            .sum::<usize>()
    );
    let gap =
        (manifest["repeat_digit_pause_ms"].as_u64().unwrap_or(0) as f32 * 22.05).round() as usize;
    assert_eq!(
        voice
            .synthesize(&SynthesisRequest {
                text: "0.000",
                ..request
            })
            .unwrap()
            .samples
            .len(),
        (4 * frames("zero") + frames("point")) * 256 + 2 * gap
    );
    if entries.get("four hundred").is_some() {
        assert_eq!(
            voice
                .synthesize(&SynthesisRequest {
                    text: "418",
                    ..request
                })
                .unwrap()
                .samples
                .len(),
            (frames("four hundred") + frames("eighteen")) * 256
        );
    }
}

#[test]
#[ignore = "requires schema-2 compound voice selected with TTS_MATH_LATENT_TEST_VOICE"]
fn reject_nonfinite_resealed_binary_scales() {
    use sha2::{Digest, Sha256};
    let original = root();
    if !original.join("conditioning-scales.bin").exists() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(&original).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), temp.path().join(entry.file_name())).unwrap();
    }
    let path = temp.path().join("conditioning-scales.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(path, &bytes).unwrap();
    let mut hashes: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("checksums.json")).unwrap()).unwrap();
    hashes["conditioning-scales.bin"] = serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
    std::fs::write(
        temp.path().join("checksums.json"),
        serde_json::to_vec(&hashes).unwrap(),
    )
    .unwrap();
    assert!(
        MathLatent::load(temp.path())
            .err()
            .unwrap()
            .to_string()
            .contains("invalid conditioning entry")
    );
}
