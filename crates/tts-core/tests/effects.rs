use local_tts_core::{
    Audio,
    effects::{VoiceEffects, process},
};

fn tone(rate: u32, frequency: f64, amplitude: f32) -> Audio {
    Audio {
        sample_rate: rate,
        samples: (0..rate)
            .map(|i| {
                (2.0 * std::f64::consts::PI * frequency * f64::from(i) / f64::from(rate)).sin()
                    as f32
                    * amplitude
            })
            .collect(),
    }
}
fn rms(audio: &Audio) -> f64 {
    let samples = &audio.samples[audio.samples.len() / 2..];
    (samples.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
}
fn gain(frequency: f64, settings: VoiceEffects) -> f64 {
    let source = tone(24000, frequency, 0.02);
    let (output, _) = process(&source, settings).unwrap();
    20.0 * (rms(&output) / rms(&source)).log10()
}
#[test]
fn neutral_is_exact_and_source_is_preserved() {
    let source = tone(22050, 317.0, 0.3);
    let original = source.samples.clone();
    let (output, stats) = process(&source, VoiceEffects::default()).unwrap();
    assert_eq!(source.samples, original);
    assert_eq!(output.samples, original);
    assert_eq!(output.sample_rate, source.sample_rate);
    assert_eq!(stats.peak_gain, 1.0);
    assert_eq!(stats.max_compression_db, 0.0);
}
#[test]
fn shelves_change_measured_spectrum_in_both_directions() {
    for db in [-12.0, 6.0, 12.0] {
        let bass = VoiceEffects {
            bass_db: db,
            ..Default::default()
        };
        assert!((gain(40.0, bass) - f64::from(db)).abs() < 0.15);
        assert!(gain(4000.0, bass).abs() < 0.03);
        assert!((gain(180.0, bass) - f64::from(db) / 2.0).abs() < 0.03);
        let treble = VoiceEffects {
            brightness_db: db,
            ..Default::default()
        };
        assert!((gain(8500.0, treble) - f64::from(db)).abs() < 0.1);
        assert!(gain(100.0, treble).abs() < 0.03);
        assert!((gain(3000.0, treble) - f64::from(db) / 2.0).abs() < 0.03);
    }
}
#[test]
fn compression_attenuates_loud_passages_and_releases() {
    let settings = VoiceEffects {
        compression: 1.0,
        ..Default::default()
    };
    let quiet = tone(24000, 1000.0, 0.02);
    assert_eq!(process(&quiet, settings).unwrap().0.samples, quiet.samples);
    let mut source = Audio {
        sample_rate: 24000,
        samples: vec![0.5; 24000],
    };
    source.samples.extend(vec![0.02; 24000]);
    let (output, stats) = process(&source, settings).unwrap();
    assert_eq!(
        output.samples[0], 0.5,
        "attack preserves the first transient"
    );
    assert!(
        output.samples[2000] < 0.2,
        "compression settles after attack"
    );
    let expected = 0.5 * 10.0_f64.powf(-0.75 * (20.0 * 0.5_f64.log10() + 18.0) / 20.0);
    assert!((f64::from(output.samples[23000]) - expected).abs() < 1e-5);
    assert!(output.samples[24000] < 0.01);
    assert!((output.samples[47999] - 0.02).abs() < 1e-5);
    assert!((stats.max_compression_db - 8.98455).abs() < 0.01);
    assert_eq!(stats.peak_gain, 1.0);
}
#[test]
fn peaks_are_scaled_without_flattening_or_upward_normalization() {
    let source = Audio {
        sample_rate: 24000,
        samples: vec![2.0, -1.0, 0.25],
    };
    let settings = VoiceEffects {
        peak_ceiling: 0.5,
        ..Default::default()
    };
    let (output, stats) = process(&source, settings).unwrap();
    assert_eq!(output.samples, vec![0.5, -0.25, 0.0625]);
    assert_eq!(stats.peak_gain, 0.25);
    assert_eq!(stats.input_peak, 2.0);
    assert_eq!(stats.output_peak, 0.5);
    let quiet = tone(24000, 123.0, 0.01);
    assert_eq!(process(&quiet, settings).unwrap().0.samples, quiet.samples);
}
#[test]
fn extreme_settings_are_finite_bounded_and_keep_duration() {
    for rate in [8000, 16000, 22050, 24000, 44100, 48000, 96000] {
        for bass_db in [-12.0, 12.0] {
            for brightness_db in [-12.0, 12.0] {
                for compression in [0.0, 1.0] {
                    let settings = VoiceEffects {
                        bass_db,
                        brightness_db,
                        compression,
                        ..Default::default()
                    };
                    for samples in [
                        vec![0.0; 100],
                        vec![0.7],
                        vec![f32::MAX, -f32::MAX],
                        tone(rate, 100.0, 0.8).samples,
                    ] {
                        let source = Audio {
                            sample_rate: rate,
                            samples,
                        };
                        let (output, stats) = process(&source, settings).unwrap();
                        assert_eq!(output.samples.len(), source.samples.len());
                        assert_eq!(output.sample_rate, rate);
                        assert!(
                            output
                                .samples
                                .iter()
                                .all(|x| x.is_finite() && x.abs() <= settings.peak_ceiling)
                        );
                        assert!(stats.peak_gain > 0.0 && stats.peak_gain <= 1.0);
                        assert!(stats.max_compression_db.is_finite());
                    }
                }
            }
        }
    }
}
#[test]
fn malformed_inputs_fail_before_processing() {
    let mut source = tone(24000, 100.0, 0.1);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 12.1, -12.1] {
        assert!(
            process(
                &source,
                VoiceEffects {
                    pitch_semitones: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            process(
                &source,
                VoiceEffects {
                    formant_semitones: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            process(
                &source,
                VoiceEffects {
                    bass_db: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            process(
                &source,
                VoiceEffects {
                    brightness_db: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(
        process(
            &source,
            VoiceEffects {
                formant_semitones: 6.1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        process(
            &source,
            VoiceEffects {
                formant_semitones: -6.1,
                ..Default::default()
            }
        )
        .is_err()
    );
    for value in [f32::NAN, -0.1, 1.1] {
        assert!(
            process(
                &source,
                VoiceEffects {
                    compression: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            process(
                &source,
                VoiceEffects {
                    peak_ceiling: value,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    for rate in [0, 7999, 96001] {
        source.sample_rate = rate;
        assert!(process(&source, VoiceEffects::default()).is_err());
    }
    source.sample_rate = 8000;
    source.samples = vec![0.0; 8000 * 600 + 1];
    assert!(process(&source, VoiceEffects::default()).is_err());
    source.samples.clear();
    assert!(process(&source, VoiceEffects::default()).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        source.samples = vec![value];
        assert!(process(&source, VoiceEffects::default()).is_err());
    }
}

fn crossings(audio: &Audio, start: usize, end: usize) -> f64 {
    let count = audio.samples[start..end]
        .windows(2)
        .filter(|x| x[0] < 0.0 && x[1] >= 0.0)
        .count();
    count as f64 * f64::from(audio.sample_rate) / (end - start) as f64
}
#[test]
fn pitch_changes_measured_frequency_without_speed_or_source_changes() {
    for rate in [8000, 22050, 24000, 48000, 96000] {
        for semitones in [-12.0, -4.0, 4.0, 12.0] {
            let input = tone(rate, 440.0, 0.2);
            let original = input.samples.clone();
            let settings = VoiceEffects {
                pitch_semitones: semitones,
                ..Default::default()
            };
            let (output, _) = process(&input, settings).unwrap();
            let expected = 440.0 * 2.0_f64.powf(f64::from(semitones) / 12.0);
            let measured = crossings(&output, rate as usize / 4, rate as usize * 3 / 4);
            assert!(
                (measured - expected).abs() < 4.0,
                "{rate} Hz, {semitones} semitones: {measured} != {expected}"
            );
            assert!(rms(&output) > 0.015 && rms(&output) < 0.35);
            assert_eq!(output.sample_rate, rate);
            assert_eq!(output.samples.len(), input.samples.len());
            assert_eq!(input.samples, original);
        }
    }
}
#[test]
fn shifted_silence_short_clips_and_extremes_remain_bounded() {
    for rate in [8000, 22050, 24000, 96000] {
        for pitch_semitones in [-12.0, 0.0, 12.0] {
            for formant_semitones in [-6.0, 0.0, 6.0] {
                let settings = VoiceEffects {
                    pitch_semitones,
                    formant_semitones,
                    bass_db: 12.0,
                    brightness_db: -12.0,
                    compression: 1.0,
                    ..Default::default()
                };
                for samples in [vec![0.0; 100], vec![0.7], vec![f32::MAX, -f32::MAX]] {
                    let input = Audio {
                        sample_rate: rate,
                        samples,
                    };
                    let (output, _) = process(&input, settings).unwrap();
                    assert_eq!(output.samples.len(), input.samples.len());
                    assert!(
                        output
                            .samples
                            .iter()
                            .all(|x| x.is_finite() && x.abs() <= 0.98)
                    );
                    if input.samples.iter().all(|&x| x == 0.0) {
                        assert!(output.samples.iter().all(|&x| x == 0.0));
                    }
                }
            }
        }
    }
}
#[test]
fn spectral_latency_is_removed_and_both_clip_boundaries_are_drained() {
    let rate = 24000;
    let mut input = tone(rate, 440.0, 0.2);
    for x in &mut input.samples[4800..19200] {
        *x = 0.0;
    }
    let (output, _) = process(
        &input,
        VoiceEffects {
            pitch_semitones: 4.0,
            ..Default::default()
        },
    )
    .unwrap();
    let energy = |from: usize, to: usize| {
        output.samples[from..to].iter().map(|x| x * x).sum::<f32>() / (to - from) as f32
    };
    assert!(energy(200, 3000) > 0.001, "lost starting audio");
    assert!(energy(21000, 23800) > 0.001, "lost final audio");
    assert!(
        energy(9000, 15000) < 1e-7,
        "shifted boundaries into middle silence"
    );
    let (again, _) = process(
        &input,
        VoiceEffects {
            pitch_semitones: 4.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        output.samples, again.samples,
        "processing must be reproducible"
    );
}

fn vowel() -> Audio {
    let rate = 24000;
    Audio {
        sample_rate: rate,
        samples: (0..rate)
            .map(|i| {
                (1..=55)
                    .map(|harmonic| {
                        let frequency = f64::from(harmonic) * 120.0;
                        let weight = 0.002
                            + 0.025 * (-0.5 * ((frequency - 600.0) / 110.0).powi(2)).exp()
                            + 0.018 * (-0.5 * ((frequency - 1800.0) / 160.0).powi(2)).exp();
                        weight
                            * (2.0 * std::f64::consts::PI * frequency * f64::from(i)
                                / f64::from(rate))
                            .cos()
                    })
                    .sum::<f64>() as f32
            })
            .collect(),
    }
}
fn envelope_center(audio: &Audio, fundamental: f64, low: f64, high: f64) -> f64 {
    let mut energy = 0.0;
    let mut weighted = 0.0;
    for harmonic in 1..=80 {
        let frequency = fundamental * f64::from(harmonic);
        if frequency < low || frequency > high {
            continue;
        }
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &x) in audio.samples.iter().enumerate().skip(6000).take(12000) {
            let phase =
                2.0 * std::f64::consts::PI * frequency * i as f64 / f64::from(audio.sample_rate);
            re += f64::from(x) * phase.cos();
            im += f64::from(x) * phase.sin();
        }
        let power = re * re + im * im;
        energy += power;
        weighted += frequency * power;
    }
    weighted / energy
}
#[test]
fn formants_shift_envelope_independently_of_pitch() {
    let input = vowel();
    let baseline = envelope_center(&input, 120.0, 250.0, 1250.0);
    let (up, _) = process(
        &input,
        VoiceEffects {
            formant_semitones: 6.0,
            ..Default::default()
        },
    )
    .unwrap();
    let (down, _) = process(
        &input,
        VoiceEffects {
            formant_semitones: -6.0,
            ..Default::default()
        },
    )
    .unwrap();
    let raised = envelope_center(&up, 120.0, 250.0, 1250.0);
    // The second resonance also moves down (~1273 Hz); exclude it when
    // measuring the first resonance rather than blend the two formants.
    let lowered = envelope_center(&down, 120.0, 250.0, 800.0);
    assert!(
        raised > baseline * 1.20,
        "formant did not rise: {baseline} → {raised}"
    );
    assert!(
        lowered < baseline * 0.85,
        "formant did not fall: {baseline} → {lowered}"
    );
    let (pitched, _) = process(
        &input,
        VoiceEffects {
            pitch_semitones: 4.0,
            ..Default::default()
        },
    )
    .unwrap();
    let compensated = envelope_center(&pitched, 120.0 * 2.0_f64.powf(4.0 / 12.0), 250.0, 1250.0);
    assert!(
        (compensated - baseline).abs() < 120.0,
        "pitch moved formant too far: {baseline} → {compensated}"
    );
    let fundamental_strength = |clip: &Audio| {
        let period = 200;
        let error = (6000..12000)
            .map(|i| (clip.samples[i] - clip.samples[i + period]).powi(2))
            .sum::<f32>();
        let power = clip.samples[6000..12000].iter().map(|x| x * x).sum::<f32>();
        error / power
    };
    assert!(
        fundamental_strength(&up) < 0.15,
        "formant shift changed periodicity"
    );
    assert!(
        fundamental_strength(&down) < 0.15,
        "formant shift changed periodicity"
    );
}
