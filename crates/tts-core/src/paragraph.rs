//! Bounded paragraph synthesis helpers. Chunking is not model streaming.
use crate::Audio;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cancellation is observed at chunk boundaries, never inside a model invocation.
pub fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err("synthesis cancelled".into())
    } else {
        Ok(())
    }
}

/// Split at sentence punctuation when possible, otherwise at whitespace.
/// A word exceeding the model limit fails instead of being truncated.
pub fn chunks(text: &str, limit: usize) -> Result<Vec<String>, String> {
    if limit == 0 || text.trim().is_empty() || text.chars().count() > 10000 {
        return Err("enter 1–10000 characters within the model's chunk limit".into());
    }
    let mut remaining = text.trim();
    let mut result = Vec::new();
    while !remaining.is_empty() {
        if remaining.chars().count() <= limit {
            result.push(remaining.into());
            break;
        }
        let mut sentence = None;
        let mut space = None;
        let mut iter = remaining.char_indices().peekable();
        for _ in 0..limit {
            let Some((offset, c)) = iter.next() else {
                break;
            };
            if c.is_whitespace() {
                space = Some(offset);
            }
            if ".!?;:…".contains(c) && iter.peek().is_none_or(|(_, next)| next.is_whitespace()) {
                sentence = Some(offset + c.len_utf8());
            }
        }
        if let Some((offset, c)) = iter.peek()
            && c.is_whitespace()
        {
            space = Some(*offset);
        }
        let split = sentence
            .or(space)
            .filter(|n| *n > 0)
            .ok_or("a word exceeds the model chunk limit")?;
        result.push(remaining[..split].trim().into());
        remaining = remaining[split..].trim_start();
    }
    Ok(result)
}

/// Join with 120 ms silence; reject incompatible or excessive model output.
pub fn append(output: &mut Option<Audio>, audio: Audio) -> Result<(), String> {
    if audio.sample_rate == 0
        || audio.samples.is_empty()
        || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err("invalid chunk audio".into());
    }
    let total = output.as_ref().map_or(0, |a| {
        a.samples.len() + audio.sample_rate as usize * 120 / 1000
    });
    if total + audio.samples.len() > audio.sample_rate as usize * 600 {
        return Err("paragraph audio exceeds ten minutes".into());
    }
    match output {
        None => *output = Some(audio),
        Some(output) => {
            if output.sample_rate != audio.sample_rate {
                return Err("chunk sample rates differ".into());
            }
            output.samples.resize(total, 0.0);
            output.samples.extend(audio.samples);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_and_sentence_boundaries_preserve_text() {
        let text = "Привіт світе! Це українська мова. Ще один рядок.";
        let parts = chunks(text, 24).unwrap();
        assert_eq!(parts[0], "Привіт світе!");
        assert!(parts.iter().all(|p| p.chars().count() <= 24));
        assert_eq!(parts.join(" "), text);
        assert_eq!(
            chunks("hello world again", 11).unwrap(),
            ["hello world", "again"]
        );
        assert!(chunks("toolongword", 4).is_err());
        assert!(chunks(" ", 4).is_err());
        assert!(chunks(&"a".repeat(10001), 510).is_err());
    }
    #[test]
    fn joining_adds_silence_and_checks_audio() {
        let mut output = None;
        for _ in 0..2 {
            append(
                &mut output,
                Audio {
                    sample_rate: 1000,
                    samples: vec![0.25; 100],
                },
            )
            .unwrap();
        }
        let audio = output.unwrap();
        assert_eq!(audio.samples.len(), 320);
        assert!(audio.samples[100..220].iter().all(|s| *s == 0.0));
        let mut output = Some(audio);
        assert!(
            append(
                &mut output,
                Audio {
                    sample_rate: 2000,
                    samples: vec![0.1; 10]
                }
            )
            .is_err()
        );
    }
}
