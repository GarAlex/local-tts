//! Print what narration mode will say: `cargo run --example narration_text -- "Save $49.99 today"`.
fn main() {
    for text in std::env::args().skip(1) {
        match local_tts_core::english::narration_text(&text) {
            Ok(spoken) => println!("{spoken}"),
            Err(e) => eprintln!("{e}"),
        }
    }
}
