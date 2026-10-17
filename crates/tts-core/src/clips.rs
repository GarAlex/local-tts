//! Native WAV import, downmixing, trimming and band-limited resampling.
use crate::Audio;
use std::path::Path;

pub fn import_wav(path: &Path) -> Result<Audio, String> {
    import_wav_at_rate(path, 22050)
}

/// Downmix a WAV and convert directly to the requested native sample rate.
pub fn import_wav_at_rate(path: &Path, sample_rate: u32) -> Result<Audio, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.channels > 2 || !(8000..=192000).contains(&spec.sample_rate) {
        return Err("unsupported WAV channels/sample rate".into());
    }
    if reader.duration() == 0
        || reader.duration() > spec.sample_rate * 300
        || reader.len() > 30_000_000
    {
        return Err("import a WAV between zero and five minutes".into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
        }
    }
    .map_err(|e| e.to_string())?;
    if samples.len() % usize::from(spec.channels) != 0 {
        return Err("incomplete WAV frame".into());
    }
    let mono = samples
        .chunks_exact(usize::from(spec.channels))
        .map(|frame| {
            frame.iter().map(|s| f64::from(*s)).sum::<f64>() as f32 / f32::from(spec.channels)
        })
        .collect();
    resample(
        &Audio {
            sample_rate: spec.sample_rate,
            samples: mono,
        },
        sample_rate,
    )
}

/// Windowed-sinc polyphase conversion. No automatic loudness processing.
pub fn resample(audio: &Audio, sample_rate: u32) -> Result<Audio, String> {
    if !(8000..=192000).contains(&audio.sample_rate)
        || !(8000..=192000).contains(&sample_rate)
        || audio.samples.is_empty()
        || audio
            .samples
            .iter()
            .any(|s| !s.is_finite() || s.abs() > 1.0)
        || audio.samples.len() as u64 > u64::from(audio.sample_rate) * 300
    {
        return Err(
            "invalid audio, unsupported sample rate or clip longer than five minutes".into(),
        );
    }
    if sample_rate == audio.sample_rate {
        return Ok(audio.clone());
    }
    const RADIUS: isize = 64;
    const PHASES: usize = 1024;
    let cutoff = (f64::from(sample_rate) / f64::from(audio.sample_rate)).min(1.0) * 0.95;
    let kernels: Vec<Vec<f64>> = (0..PHASES)
        .map(|phase| {
            let fraction = phase as f64 / PHASES as f64;
            (-RADIUS + 1..=RADIUS)
                .map(|offset| {
                    let distance = offset as f64 - fraction;
                    let angle = std::f64::consts::PI * cutoff * distance;
                    let sinc = if angle.abs() < 1e-12 {
                        1.0
                    } else {
                        angle.sin() / angle
                    };
                    let window =
                        0.5 * (1.0 + (std::f64::consts::PI * distance / RADIUS as f64).cos());
                    cutoff * sinc * window
                })
                .collect()
        })
        .collect();
    let count = (audio.samples.len() as f64 * f64::from(sample_rate) / f64::from(audio.sample_rate))
        .round() as usize;
    if count == 0 {
        return Err("clip is too short to resample".into());
    }
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let position = i as f64 * f64::from(audio.sample_rate) / f64::from(sample_rate);
        let center = position.floor() as isize;
        let phase = ((position.fract() * PHASES as f64).floor() as usize).min(PHASES - 1);
        let (mut sum, mut weight_sum) = (0.0, 0.0);
        for (j, weight) in kernels[phase].iter().enumerate() {
            let index = center + j as isize - RADIUS + 1;
            if index >= 0
                && let Some(sample) = audio.samples.get(index as usize)
            {
                sum += f64::from(*sample) * weight;
                weight_sum += weight;
            }
        }
        // Overshoot can occur around discontinuities; PCM output must remain in range.
        samples.push((sum / weight_sum).clamp(-1.0, 1.0) as f32);
    }
    Ok(Audio {
        sample_rate,
        samples,
    })
}

pub fn trim(audio: &Audio, start: f64, end: f64) -> Result<Audio, String> {
    let duration = audio.samples.len() as f64 / f64::from(audio.sample_rate);
    if audio.sample_rate == 0
        || !start.is_finite()
        || !end.is_finite()
        || start < 0.0
        || end > duration + 1e-9
        || end <= start
    {
        return Err("choose a valid start/end range".into());
    }
    let start = (start * f64::from(audio.sample_rate)).round() as usize;
    let end = ((end * f64::from(audio.sample_rate)).round() as usize).min(audio.samples.len());
    if start >= end {
        return Err("trimmed audio is empty".into());
    }
    Ok(Audio {
        sample_rate: audio.sample_rate,
        samples: audio.samples[start..end].to_vec(),
    })
}
