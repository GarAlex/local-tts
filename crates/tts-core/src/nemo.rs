//! English text normalization by NVIDIA NeMo's grammars, run natively.
//!
//! NeMo-text-processing (Apache-2.0) writes its English normalizer as
//! weighted finite-state grammars: money, percentages, dates, times, years
//! and decades, measures, ordinals, fractions, telephone numbers, ranges,
//! serials, abbreviations, @handles and URLs. `training/nemo/build_grammars.sh`
//! compiles them with pynini at the pinned release and exports the three
//! transducers bundled here (assets/nemo-en-tn, about 1.1 MB gzipped). This
//! module reproduces NeMo's Python pipeline around them — tag, parse the
//! tokens, try field orders until the verbalizer accepts, post-process — on
//! `rustfst`, without Python. `tests/nemo_parity.rs` holds it to Python NeMo's
//! own outputs on NeMo's English test cases.
use rustfst::Trs;
use rustfst::algorithms::compose::compose;
use rustfst::algorithms::{top_sort, tr_sort};
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::{CoreFst, ExpandedFst, SerializableFst};
use rustfst::prelude::{ILabelCompare, TropicalWeight};
use rustfst::utils::acceptor;
use rustfst::{Label, Semiring, StateId};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::sync::OnceLock;

type Fst = VectorFst<TropicalWeight>;

const CLASSIFY: &[u8] = include_bytes!("../../../assets/nemo-en-tn/classify.fst.gz");
const VERBALIZE: &[u8] = include_bytes!("../../../assets/nemo-en-tn/verbalize.fst.gz");
const POST: &[u8] = include_bytes!("../../../assets/nemo-en-tn/post.fst.gz");
const MANIFEST: &str = include_str!("../../../assets/nemo-en-tn/manifest.json");

/// NeMo's own bound on field orderings tried per group of tokens.
const MAX_PERMUTATIONS: usize = 729;

struct Grammars {
    classify: Fst,
    verbalize: Fst,
    post: Fst,
}

fn grammars() -> Result<&'static Grammars, String> {
    static GRAMMARS: OnceLock<Result<Grammars, String>> = OnceLock::new();
    GRAMMARS
        .get_or_init(|| {
            let manifest: serde_json::Value =
                serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
            let load = |name: &str, gz: &[u8]| -> Result<Fst, String> {
                let mut bytes = Vec::new();
                flate2::read::GzDecoder::new(gz)
                    .read_to_end(&mut bytes)
                    .map_err(|e| format!("NeMo {name} grammar: {e}"))?;
                let digest: String = Sha256::digest(&bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                if manifest["files"][name]["sha256"].as_str() != Some(digest.as_str()) {
                    return Err(format!(
                        "NeMo {name} grammar does not match its pinned checksum"
                    ));
                }
                let mut fst = Fst::load(&bytes).map_err(|e| format!("NeMo {name} grammar: {e}"))?;
                tr_sort(&mut fst, ILabelCompare {});
                Ok(fst)
            };
            Ok(Grammars {
                classify: load("classify", CLASSIFY)?,
                verbalize: load("verbalize", VERBALIZE)?,
                post: load("post", POST)?,
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The NeMo release the bundled grammars were compiled from.
pub fn version() -> String {
    serde_json::from_str::<serde_json::Value>(MANIFEST)
        .ok()
        .and_then(|m| m["nemo_text_processing"].as_str().map(String::from))
        .unwrap_or_default()
}

/// Spoken-form English: `Save $49.99 on Jan. 5` → `Save forty nine dollars
/// ninety nine cents on january fifth`. Text NeMo cannot tag comes back
/// unchanged, as in NeMo. Long text is normalized a sentence at a time.
pub fn normalize(text: &str) -> Result<String, String> {
    let grammars = grammars()?;
    let mut out = Vec::new();
    for sentence in sentences(text) {
        out.push(normalize_sentence(grammars, sentence)?);
    }
    Ok(out.join(" "))
}

/// Sentences: split after . ! ? or a line break when what follows begins a
/// new sentence, so each composition stays small. Abbreviations inside a
/// sentence (Dr. Smith, 5 p.m. today) keep it whole: the next word is lowercase
/// or a number would have to follow.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (k, &(i, c)) in chars.iter().enumerate() {
        let end = i + c.len_utf8();
        let breaks = match c {
            '\n' => true,
            '.' | '!' | '?' => {
                chars.get(k + 1).is_some_and(|(_, n)| n.is_whitespace())
                    && chars[k + 1..]
                        .iter()
                        .find(|(_, n)| !n.is_whitespace())
                        .is_some_and(|(_, n)| n.is_uppercase() || *n == '"' || *n == '“')
                    && !abbreviation_before(&text[start..end])
            }
            _ => false,
        };
        if breaks {
            if !text[start..end].trim().is_empty() {
                out.push(text[start..end].trim());
            }
            start = end;
        }
    }
    if !text[start..].trim().is_empty() {
        out.push(text[start..].trim());
    }
    out
}

/// Does `piece` end with a title abbreviation whose dot does not end the
/// sentence (Dr. Smith, Mr. Jones)?
fn abbreviation_before(piece: &str) -> bool {
    let last = piece.split_whitespace().last().unwrap_or_default();
    matches!(
        last,
        "Dr." | "Mr." | "Mrs." | "Ms." | "Prof." | "St." | "Sr." | "Jr." | "Mt." | "vs." | "No."
    )
}

fn normalize_sentence(g: &Grammars, text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(String::new());
    }
    let Some(tagged) = best(text, &g.classify)? else {
        return Ok(text.into());
    };
    let tokens = Parser::new(&tagged).list()?;
    let mut output = String::new();
    for group in groups(tokens) {
        // The cartesian product of each token's field orders, walked lazily,
        // until the verbalizer accepts one — NeMo's generate_permutations.
        let options: Vec<Vec<String>> = group
            .iter()
            .map(|t| permute(std::slice::from_ref(t)))
            .collect();
        let mut index = vec![0usize; options.len()];
        let found = 'search: loop {
            let candidate: String = options
                .iter()
                .zip(&index)
                .map(|(o, &i)| o[i].as_str())
                .collect();
            if let Some(spoken) = best(&candidate, &g.verbalize)? {
                break Some(spoken);
            }
            for d in (0..index.len()).rev() {
                index[d] += 1;
                if index[d] < options[d].len() {
                    continue 'search;
                }
                index[d] = 0;
            }
            break None;
        };
        let Some(spoken) = found else {
            return Ok(text.into());
        };
        output.push(' ');
        output.push_str(&spoken);
    }
    let output = output
        .split(' ')
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(best(&output, &g.post)?.unwrap_or(output))
}

/// The lowest-cost output for `text` through `fst`; None when it accepts nothing.
fn best(text: &str, fst: &Fst) -> Result<Option<String>, String> {
    let labels: Vec<Label> = text.bytes().map(Label::from).collect();
    let input: Fst = acceptor(&labels, TropicalWeight::one());
    let lattice: Fst =
        compose::<TropicalWeight, Fst, Fst, Fst, _, _>(input, fst).map_err(|e| e.to_string())?;
    match best_path(lattice)? {
        Some(labels) => String::from_utf8(labels.into_iter().map(|l| l as u8).collect())
            .map(Some)
            .map_err(|e| e.to_string()),
        None => Ok(None),
    }
}

/// The exact lowest-cost path's output labels. NeMo marks preferred readings
/// with small negative weights (1.1 − 0.0001) that a Dijkstra-style search can
/// get wrong; a sentence's lattice is acyclic, so one pass in topological
/// order is exact. An exact tie goes to the path writing fewer symbols, which
/// is how NeMo's Python resolves the ties its post-processing has.
fn best_path(mut lattice: Fst) -> Result<Option<Vec<Label>>, String> {
    if lattice.num_states() == 0 {
        return Ok(None);
    }
    top_sort(&mut lattice).map_err(|e| e.to_string())?;
    let Some(start) = lattice.start() else {
        return Ok(None);
    };
    let n = lattice.num_states();
    let better = |a: (f32, usize), b: (f32, usize)| {
        a.0 < b.0 - 1e-6 || ((a.0 - b.0).abs() <= 1e-6 && a.1 < b.1)
    };
    let mut dist = vec![(f32::INFINITY, usize::MAX); n];
    let mut back: Vec<Option<(StateId, Label)>> = vec![None; n];
    dist[start as usize] = (0.0, 0);
    let mut best: Option<((f32, usize), StateId)> = None;
    for s in 0..n as StateId {
        let here = dist[s as usize];
        if !here.0.is_finite() {
            continue;
        }
        if let Some(w) = lattice.final_weight(s).map_err(|e| e.to_string())? {
            let total = (here.0 + *w.value(), here.1);
            if best.is_none_or(|(b, _)| better(total, b)) {
                best = Some((total, s));
            }
        }
        for tr in lattice.get_trs(s).map_err(|e| e.to_string())?.trs() {
            let next = tr.nextstate as usize;
            let candidate = (
                here.0 + *tr.weight.value(),
                here.1 + usize::from(tr.olabel != 0),
            );
            if !dist[next].0.is_finite() || better(candidate, dist[next]) {
                dist[next] = candidate;
                back[next] = Some((s, tr.olabel));
            }
        }
    }
    let Some((_, mut s)) = best else {
        return Ok(None);
    };
    let mut out = Vec::new();
    while let Some((prev, olabel)) = back[s as usize] {
        if olabel != 0 {
            out.push(olabel);
        }
        s = prev;
    }
    out.reverse();
    Ok(Some(out))
}

#[derive(Clone, Debug)]
enum Value {
    Str(String),
    Map(Vec<(String, Value)>),
    True,
}

/// NeMo's TokenParser: `tokens { name: "a" } tokens { money { … } }`.
struct Parser {
    chars: Vec<char>,
    i: usize,
}

impl Parser {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            i: 0,
        }
    }
    fn ch(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }
    fn ws(&mut self) -> bool {
        while self.ch() == Some(' ') {
            self.i += 1;
        }
        self.ch().is_some()
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.ch() != Some(c) {
            return Err(format!("NeMo tokens: expected {c:?} at {}", self.i));
        }
        self.i += 1;
        Ok(())
    }
    fn list(&mut self) -> Result<Vec<(String, Value)>, String> {
        let mut out = Vec::new();
        while self.ws() {
            let Some(token) = self.token()? else { break };
            out.push(token);
        }
        Ok(out)
    }
    fn token(&mut self) -> Result<Option<(String, Value)>, String> {
        let start = self.i;
        while self
            .ch()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            self.i += 1;
        }
        if self.i == start {
            return Ok(None);
        }
        let key: String = self.chars[start..self.i].iter().collect();
        self.ws();
        if key == "preserve_order" {
            self.expect(':')?;
            self.ws();
            for c in "true".chars() {
                self.expect(c)?;
            }
            return Ok(Some((key, Value::True)));
        }
        match self.ch() {
            Some(':') => {
                self.i += 1;
                self.ws();
                self.expect('"')?;
                let start = self.i;
                // A value ends at a quote followed by a space, as in NeMo.
                while !(self.ch() == Some('"') && self.chars.get(self.i + 1) == Some(&' ')) {
                    if self.ch().is_none() {
                        return Err("NeMo tokens: unterminated value".into());
                    }
                    self.i += 1;
                }
                let value = self.chars[start..self.i].iter().collect();
                self.expect('"')?;
                Ok(Some((key, Value::Str(value))))
            }
            Some('{') => {
                self.i += 1;
                let inner = self.list()?;
                self.expect('}')?;
                Ok(Some((key, Value::Map(inner))))
            }
            _ => Err(format!("NeMo tokens: unexpected input at {}", self.i)),
        }
    }
}

/// Groups of tokens whose orderings to try stay within NeMo's bound.
fn groups(tokens: Vec<(String, Value)>) -> Vec<Vec<(String, Value)>> {
    let mut groups = vec![Vec::new()];
    let mut current = 1usize;
    for token in tokens {
        let n = estimate(std::slice::from_ref(&token));
        if n.saturating_mul(current) > MAX_PERMUTATIONS && !groups.last().is_none_or(Vec::is_empty)
        {
            groups.push(Vec::new());
            current = 1;
        }
        current = current.saturating_mul(n);
        groups.last_mut().expect("one group").push(token);
    }
    groups
}

fn estimate(d: &[(String, Value)]) -> usize {
    let inner: usize = d
        .iter()
        .map(|(_, v)| match v {
            Value::Map(m) => estimate(m),
            _ => 1,
        })
        .product();
    (1..=d.len()).product::<usize>().saturating_mul(inner)
}

fn orders<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let first = rest.remove(i);
        for mut order in orders(&rest) {
            order.insert(0, first.clone());
            out.push(order);
        }
    }
    out
}

/// NeMo's _permute: every serialization order of a (nested) token.
fn permute(d: &[(String, Value)]) -> Vec<String> {
    let all = if d.iter().any(|(k, _)| k == "preserve_order") {
        vec![d.to_vec()]
    } else {
        orders(d)
    };
    let mut out = Vec::new();
    for order in all {
        let mut partial = vec![String::new()];
        for (k, v) in order {
            let parts: Vec<String> = match &v {
                Value::Str(s) => vec![format!("{k}: \"{s}\" ")],
                Value::True => vec![format!("{k}: true ")],
                Value::Map(inner) => permute(inner)
                    .into_iter()
                    .map(|r| format!(" {k} {{ {r} }} "))
                    .collect(),
            };
            partial = partial
                .iter()
                .flat_map(|a| parts.iter().map(move |b| format!("{a}{b}")))
                .collect();
        }
        out.extend(partial);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_at_ends_not_at_abbreviations() {
        assert_eq!(
            sentences("Dr. Smith arrived. It was 5 p.m. today! Then\nnext"),
            ["Dr. Smith arrived.", "It was 5 p.m. today!", "Then", "next"]
        );
    }

    #[test]
    fn narration_forms_read_as_words() {
        assert_eq!(
            normalize("Save $49.99 today, that's 30% off!").unwrap(),
            "Save forty nine dollars ninety nine cents today, that's thirty percent off!"
        );
        assert_eq!(
            normalize("In the 1990s.").unwrap(),
            "In the nineteen nineties."
        );
    }

    /// Parity with Python NeMo on NeMo's own English test cases: every
    /// output identical, except two punctuation pile-ups where tied paths
    /// still break differently (listed, so a new difference fails).
    #[test]
    fn matches_python_nemo_on_its_test_cases() {
        let g = grammars().unwrap();
        let cases: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../../../assets/nemo-en-tn/reference.json"))
                .unwrap();
        let known = [
            "abbreviation. U.S.A..",
            "Hi it's 5pm,4A.M.?-34. Hi,no,yes,34! 12,again,4 and NO?17 and $.01,here & there -0.004kg",
        ];
        let mut differ = Vec::new();
        let mut expected = 0;
        for case in &cases {
            let written = case["written"].as_str().unwrap();
            let got = normalize_sentence(g, written).unwrap();
            expected += usize::from(got == case["expected"].as_str().unwrap());
            if got != case["python"].as_str().unwrap() && !known.contains(&written) {
                differ.push(format!(
                    "{written:?}\n  python: {}\n  native: {got}",
                    case["python"]
                ));
            }
        }
        assert!(
            differ.is_empty(),
            "{} differ from Python NeMo:\n{}",
            differ.len(),
            differ.join("\n")
        );
        assert!(
            cases.len() >= 494 && expected >= 488,
            "{expected} of {} match NeMo's expected",
            cases.len()
        );
    }
}
