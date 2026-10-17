//! Offline mono voice processing. Preserves sample rate, duration and source PCM.
//! Shelves use the RBJ equations: https://www.w3.org/TR/audio-eq-cookbook/.
use crate::Audio;

/// No makeup gain: compression and peak protection only attenuate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceEffects {
    /// Low shelf at 180 Hz, in [-12, 12] dB.
    pub bass_db: f32,
    /// High shelf at 3000 Hz, in [-12, 12] dB.
    pub brightness_db: f32,
    /// [0, 1]: disabled through 4:1 compression, -18 dBFS threshold, 6 dB knee.
    pub compression: f32,
    /// Whole-clip peak ceiling in [0.1, 1]. Default 0.98.
    pub peak_ceiling: f32,
    /// Duration-preserving transposition in [-12, 12] semitones; zero bypasses.
    pub pitch_semitones: f32,
    /// Independent spectral envelope shift in [-6, 6] semitones.
    pub formant_semitones: f32,
}
impl Default for VoiceEffects {
    fn default() -> Self {
        Self {
            bass_db: 0.0,
            brightness_db: 0.0,
            compression: 0.0,
            peak_ceiling: 0.98,
            pitch_semitones: 0.0,
            formant_semitones: 0.0,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub input_peak: f64,
    pub output_peak: f64,
    /// Global gain applied after EQ/compression, at most one.
    pub peak_gain: f64,
    pub max_compression_db: f64,
}

impl VoiceEffects {
    pub fn validate(self) -> Result<(), String> {
        for (name, value, range) in [
            ("bass", self.bass_db, -12.0..=12.0),
            ("brightness", self.brightness_db, -12.0..=12.0),
            ("compression", self.compression, 0.0..=1.0),
            ("peak ceiling", self.peak_ceiling, 0.1..=1.0),
            ("pitch", self.pitch_semitones, -12.0..=12.0),
            ("formants", self.formant_semitones, -6.0..=6.0),
        ] {
            if !value.is_finite() || !range.contains(&value) {
                return Err(format!(
                    "invalid {name}: expected finite value in {range:?}"
                ));
            }
        }
        Ok(())
    }
}

/// Process at most ten minutes of finite PCM at 8–96 kHz. Returns a separately
/// owned clip. This is whole-clip processing, not a streaming limiter: global
/// attenuation protects peaks without clipping or changing relative dynamics.
/// Neutral settings preserve samples exactly when they are below the ceiling.
pub fn process(audio: &Audio, settings: VoiceEffects) -> Result<(Audio, Stats), String> {
    settings.validate()?;
    if !(8000..=96000).contains(&audio.sample_rate) {
        return Err("effects require a sample rate from 8000 to 96000 Hz".into());
    }
    if audio.samples.is_empty() || audio.samples.len() > audio.sample_rate as usize * 600 {
        return Err("effects require nonempty audio of at most ten minutes".into());
    }
    if audio.samples.iter().any(|x| !x.is_finite()) {
        return Err("effects require finite PCM".into());
    }
    let rate = f64::from(audio.sample_rate);
    let mut bass = Shelf::new(rate, 180.0, settings.bass_db, false);
    let mut brightness = Shelf::new(rate, 3000.0, settings.brightness_db, true);
    let mut compressor = Compressor::new(rate, settings.compression);
    let mut stats = Stats {
        input_peak: 0.0,
        output_peak: 0.0,
        peak_gain: 1.0,
        max_compression_db: 0.0,
    };
    // f64 intermediates allow even large finite f32 input to be protected safely.
    let mut values = audio
        .samples
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    stats.input_peak = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if settings.pitch_semitones != 0.0 || settings.formant_semitones != 0.0 {
        // Normalize large finite PCM before spectral energy/envelope estimation.
        let scale = stats.input_peak.max(1.0);
        for x in &mut values {
            *x /= scale;
        }
        let mut shifted = vec![0.0; values.len()];
        let status = unsafe {
            local_tts_voice_shift(
                values.as_ptr(),
                shifted.as_mut_ptr(),
                values.len(),
                audio.sample_rate,
                f64::from(settings.pitch_semitones),
                f64::from(settings.formant_semitones),
            )
        };
        if status != 0 {
            return Err(format!(
                "native pitch/formant processing failed (status {status})"
            ));
        }
        values = shifted;
        for x in &mut values {
            *x *= scale;
        }
    }
    let mut peak = 0.0_f64;
    for sample in &mut values {
        let (value, reduction) = compressor.tick(brightness.tick(bass.tick(*sample)));
        if !value.is_finite() {
            return Err("nonfinite effects output".into());
        }
        stats.max_compression_db = stats.max_compression_db.max(reduction);
        peak = peak.max(value.abs());
        *sample = value;
    }
    let ceiling = f64::from(settings.peak_ceiling);
    if peak > ceiling {
        stats.peak_gain = ceiling / peak;
    }
    let samples = values
        .into_iter()
        .map(|x| {
            let sample = (x * stats.peak_gain) as f32;
            sample.clamp(-settings.peak_ceiling, settings.peak_ceiling)
        })
        .collect::<Vec<_>>();
    stats.output_peak = samples
        .iter()
        .map(|x| f64::from(x.abs()))
        .fold(0.0, f64::max);
    Ok((
        Audio {
            sample_rate: audio.sample_rate,
            samples,
        },
        stats,
    ))
}

unsafe extern "C" {
    fn local_tts_voice_shift(
        input: *const f64,
        output: *mut f64,
        count: usize,
        rate: u32,
        pitch: f64,
        formant: f64,
    ) -> i32;
}

struct Shelf {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
    bypass: bool,
}
impl Shelf {
    fn new(rate: f64, frequency: f64, gain: f32, high: bool) -> Self {
        let a = 10.0_f64.powf(f64::from(gain) / 40.0);
        let w = 2.0 * std::f64::consts::PI * frequency / rate;
        let c = w.cos();
        let beta = a.sqrt() * w.sin() * 2.0_f64.sqrt();
        let p = a + 1.0;
        let m = a - 1.0;
        let (b, den) = if high {
            (
                [
                    a * (p + m * c + beta),
                    -2.0 * a * (m + p * c),
                    a * (p + m * c - beta),
                ],
                [p - m * c + beta, 2.0 * (m - p * c), p - m * c - beta],
            )
        } else {
            (
                [
                    a * (p - m * c + beta),
                    2.0 * a * (m - p * c),
                    a * (p - m * c - beta),
                ],
                [p + m * c + beta, -2.0 * (m + p * c), p + m * c - beta],
            )
        };
        Self {
            b: b.map(|v| v / den[0]),
            a: [den[1] / den[0], den[2] / den[0]],
            z: [0.0; 2],
            bypass: gain == 0.0,
        }
    }
    fn tick(&mut self, x: f64) -> f64 {
        if self.bypass {
            return x;
        }
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}
struct Compressor {
    attack: f64,
    release: f64,
    envelope: f64,
    slope: f64,
}
impl Compressor {
    fn new(rate: f64, amount: f32) -> Self {
        Self {
            attack: (-1.0 / (rate * 0.005)).exp(),
            release: (-1.0 / (rate * 0.120)).exp(),
            envelope: 0.0,
            slope: 1.0 / (1.0 + 3.0 * f64::from(amount)) - 1.0,
        }
    }
    fn tick(&mut self, x: f64) -> (f64, f64) {
        if self.slope == 0.0 {
            return (x, 0.0);
        }
        let level = x.abs();
        let coefficient = if level > self.envelope {
            self.attack
        } else {
            self.release
        };
        self.envelope = coefficient * self.envelope + (1.0 - coefficient) * level;
        let excess = 20.0 * self.envelope.max(f64::MIN_POSITIVE).log10() + 18.0;
        let gain_db = if excess <= -3.0 {
            0.0
        } else if excess >= 3.0 {
            self.slope * excess
        } else {
            self.slope * (excess + 3.0).powi(2) / 12.0
        };
        (x * 10.0_f64.powf(gain_db / 20.0), -gain_db)
    }
}
