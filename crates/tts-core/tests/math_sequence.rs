use local_tts_core::{
    Language, SynthesisRequest, Synthesizer, math_sequence::MathSequence, math_speech::MathStudent,
};
use std::path::PathBuf;
fn root() -> PathBuf {
    std::env::var_os("TTS_MATH_SEQUENCE_TEST_VOICE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../checkpoints/math-sequence-ru-native-v6")
        })
}
#[test]
#[ignore = "requires local continuous sequence export"]
fn parity_numeric_contract_and_cancellation() {
    let path = root();
    let bytes: u64 = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    if let Some(archive) = std::env::var_os("TTS_MATH_SEQUENCE_TEST_ARCHIVE") {
        // Repacking validates archive CRC and contained hashes; the user budget
        // is download size, so extracted candidates may exceed 10 MB.
        assert!(std::fs::metadata(archive).unwrap().len() < 10_000_000);
        assert!(bytes < 16_000_000);
    } else {
        assert!(bytes < 10_000_000);
    }
    assert!(MathSequence::load(&path).unwrap().verify(&path).unwrap() < 1e-5);
    let mut model = MathStudent::load(&path).unwrap();
    let base = SynthesisRequest {
        text: "12 + 225 = 237",
        language: model.language(),
        speaker: "default",
        duration_scale: 1.0,
    };
    for (text, scale) in [
        ("12", 1.0),
        ("2 + 3", 0.7),
        ("14 * 7", 1.0),
        (base.text, 1.0),
        ("0.009 * 164", 1.2),
    ] {
        let audio = model
            .synthesize(&SynthesisRequest {
                text,
                duration_scale: scale,
                ..base
            })
            .unwrap();
        assert_eq!(audio.sample_rate, 22050);
        assert!(audio.samples.len() > 2205);
        assert!(audio.samples.iter().all(|v| v.is_finite()));
    }
    for invalid in [
        SynthesisRequest {
            text: "hello",
            ..base
        },
        SynthesisRequest {
            language: Language::En,
            ..base
        },
        SynthesisRequest {
            language: if base.language == Language::Ru {
                Language::Uk
            } else {
                Language::Ru
            },
            ..base
        },
        SynthesisRequest {
            duration_scale: f32::NAN,
            ..base
        },
    ] {
        assert!(model.synthesize(&invalid).is_err());
    }
    assert!(
        model
            .synthesize_cancellable(&base, &std::sync::atomic::AtomicBool::new(true))
            .is_err()
    );
}
#[test]
#[ignore = "requires local continuous sequence export"]
fn reject_modified_phonemes() {
    let temp = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(root()).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), temp.path().join(e.file_name())).unwrap();
    }
    let words = temp.path().join("words.json");
    let mut bytes = std::fs::read(&words).unwrap();
    bytes[0] ^= 1;
    std::fs::write(words, bytes).unwrap();
    assert!(
        MathSequence::load(temp.path())
            .err()
            .unwrap()
            .to_string()
            .contains("checksum mismatch")
    );
}

#[test]
#[ignore = "requires local continuous sequence export"]
fn reject_resealed_invalid_lexicon() {
    use sha2::{Digest, Sha256};
    let temp = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(root()).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), temp.path().join(e.file_name())).unwrap();
    }
    let path = temp.path().join("words.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    for invalid in ["range", "boundary", "missing"] {
        let mut words = original.clone();
        let entries = words["words"].as_object_mut().unwrap();
        let key = entries.keys().next().unwrap().clone();
        match invalid {
            "range" => entries.get_mut(&key).unwrap()[2] = serde_json::json!(9999),
            "boundary" => entries.get_mut(&key).unwrap()[0] = serde_json::json!(2),
            _ => {
                entries.remove(&key);
            }
        }
        let bytes = serde_json::to_vec(&words).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let hashes_path = temp.path().join("checksums.json");
        let mut hashes: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&hashes_path).unwrap()).unwrap();
        hashes["words.json"] = serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
        std::fs::write(hashes_path, serde_json::to_vec(&hashes).unwrap()).unwrap();
        assert!(MathSequence::load(temp.path()).is_err(), "{invalid}");
    }
}
