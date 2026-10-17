//! Prepare arithmetic word conditioning with the native pronunciation frontend.
use local_tts_core::{english::English, math_speech::vocabulary};
use std::{collections::HashMap, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(args.len() == 3
        || (args.len() == 4 && matches!(args[3].as_str(), "--compounds" | "--hundreds")))
    {
        return Err(
            "usage: math_latent_tokens <source-voice> <new-output.json> [--compounds|--hundreds]"
                .into(),
        );
    }
    let root = Path::new(&args[1]);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("tokens.json"))?)?;
    let inventory: HashMap<_, _> = config["symbols"]
        .as_array()
        .ok_or("missing symbols")?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Ok((
                v.as_str()
                    .ok_or("invalid symbol")?
                    .chars()
                    .next()
                    .ok_or("empty symbol")?,
                i as i64,
            ))
        })
        .collect::<Result<_, &str>>()?;
    let frontend = English::load(&root.join("cmudict.dict")).map_err(std::io::Error::other)?;
    let mut records = Vec::new();
    let mut words = vocabulary();
    if args.len() == 4 {
        for n in (21..100).filter(|n| n % 10 != 0) {
            words.push(
                local_tts_core::math_speech::words(&n.to_string())
                    .map_err(std::io::Error::other)?
                    .join(" "),
            );
        }
        if args[3] == "--hundreds" {
            for n in 1..10 {
                words.push(
                    local_tts_core::math_speech::words(&(n * 100).to_string())
                        .map_err(std::io::Error::other)?
                        .join(" "),
                );
            }
        }
    }
    for word in words {
        let phones = frontend
            .phonemize_ipa(&word)
            .map_err(std::io::Error::other)?;
        let mut tokens = vec![0];
        for c in phones.chars() {
            tokens.push(*inventory.get(&c).ok_or("missing phoneme")?);
            tokens.push(0);
        }
        records.push(serde_json::json!({"word":word,"tokens":tokens}));
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer(&mut file, &records)?;
    Ok(())
}
