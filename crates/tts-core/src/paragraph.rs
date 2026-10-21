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
    Ok(chunks_with_offsets(text, limit)?
        .into_iter()
        .map(|(chunk, _)| chunk)
        .collect())
}

/// [`chunks`], each with the byte offset in `text` where it starts (after
/// its trim), so a position in the text can be found in its chunk.
pub fn chunks_with_offsets(text: &str, limit: usize) -> Result<Vec<(String, usize)>, String> {
    if limit == 0 || text.trim().is_empty() || text.chars().count() > 10000 {
        return Err("enter 1–10000 characters within the model's chunk limit".into());
    }
    let mut remaining = text.trim();
    let mut base = text.len() - text.trim_start().len();
    let mut result = Vec::new();
    while !remaining.is_empty() {
        if remaining.chars().count() <= limit {
            result.push((remaining.into(), base));
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
        let piece = &remaining[..split];
        let lead = piece.len() - piece.trim_start().len();
        result.push((piece.trim().into(), base + lead));
        let rest = &remaining[split..];
        let skipped = rest.len() - rest.trim_start().len();
        base += split + skipped;
        remaining = rest.trim_start();
    }
    Ok(result)
}

/// The silence [`append`] puts between chunks, in samples at `sample_rate`.
pub fn gap_samples(sample_rate: u32) -> usize {
    sample_rate as usize * 120 / 1000
}

/// Join with 120 ms silence; reject incompatible or excessive model output.
pub fn append(output: &mut Option<Audio>, audio: Audio) -> Result<(), String> {
    if audio.sample_rate == 0
        || audio.samples.is_empty()
        || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err("invalid chunk audio".into());
    }
    let total = output
        .as_ref()
        .map_or(0, |a| a.samples.len() + gap_samples(audio.sample_rate));
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
    fn chunk_offsets_join_like_append() {
        for (text, limit) in [
            ("  Привіт світе! Це українська мова. Ще один рядок.  ", 24),
            ("hello world again", 11),
            ("one.  two three   four five six", 9),
            ("single", 100),
        ] {
            let parts = chunks_with_offsets(text, limit).unwrap();
            assert_eq!(
                parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>(),
                chunks(text, limit).unwrap()
            );
            // Each offset is where its chunk stands in the text.
            for (chunk, offset) in &parts {
                assert_eq!(&text[*offset..*offset + chunk.len()], chunk, "{text:?}");
            }
            assert!(parts.windows(2).all(|w| w[0].1 + w[0].0.len() <= w[1].1));
        }
        assert_eq!(gap_samples(24000), 2880);
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
