use cpal::{
    FromSample, Sample, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use local_tts_core::Audio;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Buffer {
    samples: Vec<f32>,
    error: Option<String>,
    complete: bool,
}

pub struct Recorder {
    stream: cpal::Stream,
    buffer: Arc<Mutex<Buffer>>,
    rate: u32,
    pub device: String,
}
impl Recorder {
    pub fn start() -> Result<Self, String> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or("no default microphone")?;
        let supported = device.default_input_config().map_err(|e| e.to_string())?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        if config.channels == 0
            || config.channels > 32
            || !(8000..=192000).contains(&config.sample_rate.0)
        {
            return Err("unsupported microphone configuration".into());
        }
        let buffer = Arc::new(Mutex::new(Buffer {
            samples: Vec::with_capacity(config.sample_rate.0 as usize * 20),
            ..Default::default()
        }));
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, buffer.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, buffer.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, buffer.clone()),
            cpal::SampleFormat::I32 => build::<i32>(&device, &config, buffer.clone()),
            _ => return Err(format!("unsupported microphone format {format}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            buffer,
            rate: config.sample_rate.0,
            device: device
                .name()
                .unwrap_or_else(|_| "Default microphone".into()),
        })
    }
    pub fn progress(&self) -> (f64, bool) {
        let buffer = self.buffer.lock().unwrap();
        (
            buffer.samples.len() as f64 / f64::from(self.rate),
            buffer.complete || buffer.error.is_some(),
        )
    }
    pub fn finish(self) -> Result<Audio, String> {
        drop(self.stream);
        let mut buffer = self.buffer.lock().map_err(|e| e.to_string())?;
        if let Some(error) = buffer.error.take() {
            return Err(error);
        }
        if buffer.samples.is_empty() {
            return Err(
                "no microphone samples received; check microphone permission/device".into(),
            );
        }
        Ok(Audio {
            sample_rate: self.rate,
            samples: std::mem::take(&mut buffer.samples),
        })
    }
}
fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    buffer: Arc<Mutex<Buffer>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    let limit = config.sample_rate.0 as usize * 20;
    let errors = buffer.clone();
    device
        .build_input_stream(
            config,
            move |samples: &[T], _| {
                if let Ok(mut buffer) = buffer.lock() {
                    if buffer.complete {
                        return;
                    }
                    for frame in samples.chunks_exact(channels) {
                        let value = frame.iter().map(|s| f32::from_sample(*s)).sum::<f32>()
                            / channels as f32;
                        if !value.is_finite() {
                            buffer.error = Some("invalid microphone sample".into());
                            buffer.complete = true;
                            break;
                        }
                        buffer.samples.push(value.clamp(-1.0, 1.0));
                        if buffer.samples.len() >= limit {
                            buffer.complete = true;
                            break;
                        }
                    }
                }
            },
            move |error| {
                if let Ok(mut buffer) = errors.lock() {
                    buffer.error = Some(error.to_string());
                }
            },
            None,
        )
        .map_err(|e| e.to_string())
}
