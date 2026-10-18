//! English grapheme-to-phoneme guesses for words the CMU dictionary lacks.
//!
//! A small attention seq2seq trained only on the CMU Pronouncing Dictionary
//! (BSD) by `training/g2p/train_g2p.py`: a bidirectional GRU over letters and
//! a GRU decoder emitting ARPAbet phones with stress, decoded greedily. It runs
//! here in plain Rust — no ONNX Runtime — and reproduces the trainer's own
//! outputs (see the parity test). The weights are bundled, about 6 MB.
use std::collections::HashMap;

const MODEL: &[u8] = include_bytes!("../../../assets/g2p-en-v1.bin");

struct Gru {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    input: usize,
    hidden: usize,
}

impl Gru {
    /// One step, PyTorch's gate order (r, z, n).
    fn step(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let n = self.hidden;
        let gi = affine(&self.w_ih, &self.b_ih, x, 3 * n, self.input);
        let gh = affine(&self.w_hh, &self.b_hh, h, 3 * n, n);
        (0..n)
            .map(|i| {
                let r = sigmoid(gi[i] + gh[i]);
                let z = sigmoid(gi[n + i] + gh[n + i]);
                let c = (gi[2 * n + i] + r * gh[2 * n + i]).tanh();
                (1.0 - z) * c + z * h[i]
            })
            .collect()
    }
}

fn affine(w: &[f32], b: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            b[r] + row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct G2p {
    letters: Vec<char>,
    phones: Vec<String>,
    emb: usize,
    dec: usize,
    max_phones: usize,
    letter_emb: Vec<f32>,
    phone_emb: Vec<f32>,
    forward: Gru,
    backward: Gru,
    bridge_w: Vec<f32>,
    bridge_b: Vec<f32>,
    decoder: Gru,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

impl G2p {
    /// The bundled model.
    pub fn bundled() -> Result<Self, String> {
        Self::from_bytes(MODEL)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let bad = || "invalid G2P model".to_string();
        let header_len =
            u32::from_le_bytes(bytes.get(..4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(bytes.get(4..4 + header_len).ok_or_else(bad)?)
                .map_err(|e| e.to_string())?;
        if header["format"] != "local-tts-g2p-v1" {
            return Err(bad());
        }
        let mut offset = 4 + header_len;
        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
        for entry in header["tensors"].as_array().ok_or_else(bad)? {
            let name = entry[0].as_str().ok_or_else(bad)?;
            let count: usize = entry[1]
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .product();
            let raw = bytes.get(offset..offset + 4 * count).ok_or_else(bad)?;
            offset += 4 * count;
            tensors.insert(
                name.to_string(),
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            );
        }
        if offset != bytes.len() {
            return Err(bad());
        }
        let dims = &header["dims"];
        let dim = |k: &str| dims[k].as_u64().map(|v| v as usize).ok_or_else(bad);
        let (emb, enc, dec) = (dim("emb")?, dim("enc")?, dim("dec")?);
        let mut take = |name: &str| {
            tensors
                .remove(name)
                .ok_or_else(|| format!("G2P model lacks {name}"))
        };
        let gru = |take: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>,
                   suffix: &str,
                   prefix: &str,
                   input: usize,
                   hidden: usize|
         -> Result<Gru, String> {
            Ok(Gru {
                w_ih: take(&format!("{prefix}weight_ih{suffix}"))?,
                w_hh: take(&format!("{prefix}weight_hh{suffix}"))?,
                b_ih: take(&format!("{prefix}bias_ih{suffix}"))?,
                b_hh: take(&format!("{prefix}bias_hh{suffix}"))?,
                input,
                hidden,
            })
        };
        Ok(Self {
            letters: header["letters"]
                .as_str()
                .ok_or_else(bad)?
                .chars()
                .collect(),
            phones: header["phones"]
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|p| p.as_str().unwrap_or_default().to_string())
                .collect(),
            emb,
            dec,
            max_phones: dim("max_phones")?,
            letter_emb: take("letters.weight")?,
            phone_emb: take("phones.weight")?,
            forward: gru(&mut take, "_l0", "encoder.", emb, enc)?,
            backward: gru(&mut take, "_l0_reverse", "encoder.", emb, enc)?,
            bridge_w: take("bridge.weight")?,
            bridge_b: take("bridge.bias")?,
            decoder: gru(&mut take, "", "decoder.", emb + dec, dec)?,
            out_w: take("out.weight")?,
            out_b: take("out.bias")?,
        })
    }

    /// ARPAbet phones with stress digits for a lowercase word of letters and
    /// apostrophes, or None for a word the model cannot read.
    pub fn guess(&self, word: &str) -> Option<Vec<String>> {
        let ids: Vec<usize> = word
            .chars()
            .map(|c| self.letters.iter().position(|l| *l == c).map(|i| i + 1))
            .collect::<Option<_>>()?;
        if ids.is_empty() || ids.len() > 30 {
            return None;
        }
        let embed = |table: &[f32], id: usize| table[id * self.emb..(id + 1) * self.emb].to_vec();
        let enc = self.forward.hidden;
        let mut fwd = Vec::with_capacity(ids.len());
        let mut h = vec![0.0; enc];
        for &id in &ids {
            h = self.forward.step(&embed(&self.letter_emb, id), &h);
            fwd.push(h.clone());
        }
        let mut bwd = vec![Vec::new(); ids.len()];
        let mut h = vec![0.0; enc];
        for (t, &id) in ids.iter().enumerate().rev() {
            h = self.backward.step(&embed(&self.letter_emb, id), &h);
            bwd[t] = h.clone();
        }
        let states: Vec<Vec<f32>> = fwd
            .into_iter()
            .zip(bwd)
            .map(|(f, b)| [f, b].concat())
            .collect();
        let mean: Vec<f32> = (0..self.dec)
            .map(|i| states.iter().map(|s| s[i]).sum::<f32>() / states.len() as f32)
            .collect();
        let mut hidden: Vec<f32> =
            affine(&self.bridge_w, &self.bridge_b, &mean, self.dec, self.dec)
                .into_iter()
                .map(f32::tanh)
                .collect();
        let mut context = vec![0.0; self.dec];
        let mut prev = 1usize;
        let mut out = Vec::new();
        for _ in 0..self.max_phones {
            let input = [embed(&self.phone_emb, prev), context.clone()].concat();
            hidden = self.decoder.step(&input, &hidden);
            let scores: Vec<f32> = states
                .iter()
                .map(|s| s.iter().zip(&hidden).map(|(a, b)| a * b).sum())
                .collect();
            let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let total: f32 = weights.iter().sum();
            context = (0..self.dec)
                .map(|i| {
                    states
                        .iter()
                        .zip(&weights)
                        .map(|(s, w)| s[i] * w)
                        .sum::<f32>()
                        / total
                })
                .collect();
            let logits = affine(
                &self.out_w,
                &self.out_b,
                &[hidden.clone(), context.clone()].concat(),
                self.phones.len() + 2,
                2 * self.dec,
            );
            prev = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(1);
            if prev == 1 {
                break;
            }
            if prev >= 2 {
                out.push(self.phones[prev - 2].clone());
            }
        }
        (!out.is_empty()).then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust decoder reproduces the trainer's greedy outputs.
    #[test]
    fn matches_the_trainers_reference_outputs() {
        let model = G2p::bundled().unwrap();
        let reference: HashMap<String, String> =
            serde_json::from_str(include_str!("../../../assets/g2p-en-v1.reference.json")).unwrap();
        let mismatched: Vec<_> = reference
            .iter()
            .filter(|(word, phones)| {
                model.guess(word).map(|p| p.join(" ")).as_deref() != Some(phones.as_str())
            })
            .map(|(word, _)| word.clone())
            .collect();
        assert!(
            mismatched.is_empty(),
            "differs from the trainer: {mismatched:?}"
        );
    }
}
