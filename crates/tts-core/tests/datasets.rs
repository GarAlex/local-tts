use local_tts_core::{
    Language,
    dataset::{Dataset, Rights, Split, Utterance, validate},
};
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("local-tts-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn wav(&self, name: &str, rate: u32, silent: bool) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(self.0.join(name), spec).unwrap();
        for i in 0..rate {
            let sample = if silent {
                0
            } else {
                ((i as f32 * 0.1).sin() * 8000.0) as i16
            };
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn dataset() -> Dataset {
    Dataset {
        schema_version: 1,
        language: Language::En,
        sample_rate: 22050,
        rights: Rights {
            audio: "owned".into(),
            transcripts: "original".into(),
            speaker_consent: "agreement".into(),
        },
        utterances: vec![Utterance {
            id: "one".into(),
            audio: "one.wav".into(),
            text: "Hello world.".into(),
            speaker: "one".into(),
            split: Split::Train,
        }],
    }
}

#[test]
fn valid_audio_reports_duration() {
    let fixture = Fixture::new();
    fixture.wav("one.wav", 22050, false);
    let report = validate(&dataset(), &fixture.0);
    assert_eq!(report.speakers["one"].valid_clips, 1);
    assert_eq!(report.splits["train"].duration_seconds, 1.0);
    assert_eq!(report.splits["train"].shortest_seconds, Some(1.0));
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.utterances, 1);
    assert!((report.duration_seconds - 1.0).abs() < 0.0001);
}

#[test]
fn rejects_silence_rate_mismatch_and_alias_leakage() {
    let fixture = Fixture::new();
    fixture.wav("one.wav", 16000, true);
    let mut data = dataset();
    data.utterances.push(Utterance {
        id: "two".into(),
        audio: "one.wav".into(),
        text: "Different text.".into(),
        speaker: "one".into(),
        split: Split::Test,
    });
    let report = validate(&data, &fixture.0);
    for expected in ["sample rate", "silent", "duplicate audio"] {
        assert!(
            report.errors.iter().any(|e| e.contains(expected)),
            "missing {expected}: {:?}",
            report.errors
        );
    }
}

#[test]
fn rejects_missing_and_corrupt_audio() {
    let fixture = Fixture::new();
    assert!(
        validate(&dataset(), &fixture.0)
            .errors
            .iter()
            .any(|e| e.contains("unavailable"))
    );
    std::fs::write(fixture.0.join("one.wav"), b"not a wav").unwrap();
    assert!(
        validate(&dataset(), &fixture.0)
            .errors
            .iter()
            .any(|e| e.contains("invalid WAV"))
    );
}

#[cfg(unix)]
#[test]
fn rejects_symlink_escape() {
    let fixture = Fixture::new();
    let outside = Fixture::new();
    outside.wav("external.wav", 22050, false);
    std::os::unix::fs::symlink(outside.0.join("external.wav"), fixture.0.join("one.wav")).unwrap();
    assert!(
        validate(&dataset(), &fixture.0)
            .errors
            .iter()
            .any(|e| e.contains("outside dataset root"))
    );
}

#[test]
fn statistics_count_declared_entries_but_exclude_bad_audio() {
    let fixture = Fixture::new();
    fixture.wav("one.wav", 22050, false);
    fixture.wav("two.wav", 22050, false);
    fixture.wav("silent.wav", 22050, true);
    let mut data = dataset();
    for (id, audio, speaker, split) in [
        ("two", "two.wav", "second", Split::Validation),
        ("three", "silent.wav", "second", Split::Test),
        ("four", "missing.wav", "second", Split::Train),
        ("five", "one.wav", "one", Split::Train),
    ] {
        data.utterances.push(Utterance {
            id: id.into(),
            audio: audio.into(),
            text: id.into(),
            speaker: speaker.into(),
            split,
        });
    }
    let report = validate(&data, &fixture.0);
    assert_eq!(report.speakers["second"].utterances, 3);
    assert_eq!(report.speakers["second"].valid_clips, 1);
    assert_eq!(report.speakers["second"].duration_seconds, 1.0);
    assert_eq!(report.speakers["one"].valid_clips, 1);
    assert_eq!(report.splits["train"].utterances, 3);
    assert_eq!(report.splits["train"].duration_seconds, 1.0);
    assert_eq!(report.splits["test"].shortest_seconds, None);
    assert!(
        report
            .statistics_text()
            .contains("Speaker second: 1 / 3 valid clips")
    );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["splits"]["validation"]["valid_clips"], 1);
}

#[test]
fn statistics_keep_declared_counts_when_root_is_missing() {
    let root = Fixture::new();
    let report = validate(&dataset(), &root.0.join("missing"));
    assert_eq!(report.speakers["one"].utterances, 1);
    assert_eq!(report.splits["train"].utterances, 1);
    assert_eq!(report.speakers["one"].valid_clips, 0);
    assert!(!report.errors.is_empty());
}
