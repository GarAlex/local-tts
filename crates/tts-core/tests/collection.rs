use local_tts_core::{Audio, Language, clips, collection::Collection, dataset::Split};

fn tone(rate: u32, frequency: f32) -> Audio {
    Audio {
        sample_rate: rate,
        samples: (0..rate)
            .map(|i| 0.3 * (std::f32::consts::TAU * frequency * i as f32 / rate as f32).sin())
            .collect(),
    }
}

#[test]
fn resampling_preserves_duration_and_rejects_aliases() {
    let low = clips::resample(&tone(48000, 1000.0), 22050).unwrap();
    let high = clips::resample(&tone(48000, 16000.0), 22050).unwrap();
    let rms = |a: &Audio| {
        (a.samples[200..a.samples.len() - 200]
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            / (a.samples.len() - 400) as f64)
            .sqrt()
    };
    assert_eq!(low.samples.len(), 22050);
    assert!((rms(&low) - 0.3 / 2_f64.sqrt()).abs() < 0.002);
    assert!(rms(&high) < 0.002);
    let dc = clips::resample(
        &Audio {
            sample_rate: 48000,
            samples: vec![0.25; 48000],
        },
        22050,
    )
    .unwrap();
    assert!(dc.samples.iter().all(|x| (*x - 0.25).abs() < 1e-6));
    assert!(
        clips::resample(
            &Audio {
                sample_rate: 192000,
                samples: vec![0.0]
            },
            8000
        )
        .is_err()
    );
    assert!(
        clips::resample(
            &Audio {
                sample_rate: 48000,
                samples: vec![f32::NAN; 48000]
            },
            22050
        )
        .is_err()
    );
    assert_eq!(clips::trim(&low, 0.25, 0.75).unwrap().samples.len(), 11025);
    assert!(clips::trim(&low, 0.75, 0.25).is_err());
}

#[test]
fn stereo_import_downmixes_without_modifying_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stereo.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    for _ in 0..22050 {
        writer.write_sample(16384_i16).unwrap();
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
    let before = std::fs::read(&path).unwrap();
    let audio = clips::import_wav(&path).unwrap();
    assert_eq!(audio.samples.len(), 22050);
    assert!(audio.samples.iter().all(|x| *x == 0.25));
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn collection_roundtrip_splits_and_external_edit_protection() {
    let dir = tempfile::tempdir().unwrap();
    let mut collection = Collection::create(dir.path(), Language::Uk).unwrap();
    assert!(Collection::create(dir.path(), Language::En).is_err());
    let audio = tone(22050, 440.0);
    assert!(
        collection
            .add_clip(&audio, "", "speaker", Split::Train)
            .is_err()
    );
    assert!(
        collection
            .add_clip(
                &Audio {
                    sample_rate: 22050,
                    samples: vec![0.0; 22050]
                },
                "silent",
                "speaker",
                Split::Train
            )
            .is_err()
    );
    for text in [
        "Привіт світе",
        " Другий рядок ",
        "Третій рядок",
        "ПРИВІТ СВІТЕ",
    ] {
        collection
            .add_clip(&audio, text, "speaker", Split::Train)
            .unwrap();
    }
    collection.assign_splits(1234).unwrap();
    let splits: Vec<_> = collection.data.utterances.iter().map(|u| u.split).collect();
    assert_eq!(splits[0], splits[3]);
    assert!(
        splits.contains(&Split::Train)
            && splits.contains(&Split::Validation)
            && splits.contains(&Split::Test)
    );
    collection.assign_splits(1234).unwrap();
    assert_eq!(
        splits,
        collection
            .data
            .utterances
            .iter()
            .map(|u| u.split)
            .collect::<Vec<_>>()
    );
    collection.save().unwrap();
    let reopened = Collection::open(collection.manifest()).unwrap();
    assert_eq!(reopened.data.utterances.len(), 4);
    assert_eq!(reopened.data.language, Language::Uk);
    std::fs::write(collection.manifest(), "external edit").unwrap();
    assert!(
        collection
            .add_clip(&audio, "Another", "speaker", Split::Train)
            .unwrap_err()
            .contains("outside Studio")
    );
    assert_eq!(collection.data.utterances.len(), 4);
    assert_eq!(
        std::fs::read_dir(dir.path().join("audio")).unwrap().count(),
        4
    );
    assert_eq!(
        std::fs::read_to_string(collection.manifest()).unwrap(),
        "external edit"
    );
}
