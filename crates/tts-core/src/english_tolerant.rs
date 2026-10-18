//! The pre-pass narration mode runs before NeMo's normalizer
//! ([`crate::nemo`]): only what NeMo's English grammars leave alone.
//!
//! - Accented Latin letters lose their marks: Kīlauea → Kilauea, ß → ss
//!   (NeMo passes them through, and the dictionary spells without them).
//! - Multipliers: 2.5x and 10× → 2.5 times, 10 times (NeMo reads "x").
//! - Web addresses lose `https://` and `www.`: narration says "promoshot
//!   dot app slash docs", where NeMo would read the scheme out in full.
//! - Forms NeMo reads awkwardly, rewritten into ones it reads well: a score
//!   next to a game word (won 112-108 → won 112 to 108), units per unit
//!   (km/h → km per hour), a unit used as an adjective (a 5 kg camera → a 5
//!   kilogram camera), and US/European time zones after a time (9 a.m. PT →
//!   9 a.m. Pacific Time).
//! - Roman numerals after a regnal name read as ordinals (Louis XIV → Louis
//!   the 14th) and after other capitalized words as numbers (Rocky II).
//! - Roman numerals after a word that introduces a number: Chapter IV,
//!   World War II, Super Bowl LVIII → digits, which NeMo reads (its cased
//!   English grammars leave Roman numerals as written).
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

pub(crate) fn prepare(text: &str) -> String {
    let folded = numbered_roman(&word_rules(&spoken_urls(&fold(text))));
    let chars: Vec<char> = folded.chars().collect();
    let mut out = String::with_capacity(folded.len());
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || !chars[i - 1].is_alphanumeric();
        if boundary && let Some(used) = multiplier(&chars[i..]) {
            out.extend(&chars[i..i + used - 1]);
            out.push_str(" times");
            i += used;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `https://www.promoshot.app/docs/` → `promoshot.app/docs`.
fn spoken_urls(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let lower = word.to_ascii_lowercase();
            let mut rest = word;
            for scheme in ["https://", "http://"] {
                if lower.starts_with(scheme) {
                    rest = &word[scheme.len()..];
                }
            }
            if rest.to_ascii_lowercase().starts_with("www.") && rest.len() > 4 {
                rest = &rest[4..];
            }
            if rest.len() == word.len() {
                return word.to_string();
            }
            // A trailing slash says nothing aloud; keep sentence punctuation.
            let core = rest.trim_end_matches(|c: char| ",.;:!?)".contains(c));
            format!("{}{}", core.trim_end_matches('/'), &rest[core.len()..])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Words after which a Roman numeral is a number.
const NUMBERING: &[&str] = &[
    "chapter", "part", "volume", "vol.", "book", "act", "scene", "episode", "season", "phase",
    "level", "stage", "section", "article", "appendix", "round", "war", "bowl", "title", "canto",
    "psalm", "movement", "op.", "no.",
];

/// `Chapter IV` → `Chapter 4`: a valid Roman numeral (1–3999, standard form)
/// right after a numbering word.
fn numbered_roman(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let core = word.trim_end_matches(|c: char| ",.;:!?)".contains(c));
        let tail = &word[core.len()..];
        let previous = if i > 0 { words[i - 1] } else { "" };
        let after_numbering = NUMBERING.contains(&previous.to_lowercase().trim_end_matches(','));
        let name = previous.len() >= 2
            && previous
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
            && previous.chars().skip(1).all(|c| c.is_ascii_lowercase());
        let regnal = name && REGNAL.contains(&previous);
        match roman(core) {
            Some(n) if after_numbering => out.push(format!("{n}{tail}")),
            // Louis XIV, Elizabeth I: the 14th, the 1st.
            Some(n) if regnal && n <= 39 => out.push(format!("the {n}{}{tail}", ordinal_suffix(n))),
            // Rocky II, Final Fantasy VII — but not DC (600), XL (40) or a lone V/X/I.
            Some(n) if name && (2..=39).contains(&n) && core.len() >= 2 => {
                out.push(format!("{n}{tail}"))
            }
            _ => out.push(word.to_string()),
        }
    }
    out.join(" ")
}

/// Names monarchs and popes are numbered by: a Roman numeral after one is an
/// ordinal (Louis XIV → Louis the fourteenth).
const REGNAL: &[&str] = &[
    "Louis",
    "Henry",
    "Elizabeth",
    "Charles",
    "George",
    "Edward",
    "William",
    "Richard",
    "James",
    "Mary",
    "Anne",
    "Victoria",
    "John",
    "Stephen",
    "Matilda",
    "Benedict",
    "Pius",
    "Leo",
    "Gregory",
    "Clement",
    "Innocent",
    "Paul",
    "Urban",
    "Boniface",
    "Sixtus",
    "Peter",
    "Catherine",
    "Frederick",
    "Philip",
    "Ferdinand",
    "Alexander",
    "Nicholas",
    "Ivan",
    "Napoleon",
    "Henri",
    "Carlos",
    "Juan",
    "Alfonso",
    "Isabella",
    "Wilhelm",
    "Friedrich",
    "Ludwig",
    "Gustav",
    "Christian",
    "Olaf",
    "Harald",
    "Constantine",
    "Justinian",
    "Ramesses",
    "Thutmose",
    "Amenhotep",
    "Ptolemy",
    "Antiochus",
    "Mehmed",
    "Suleiman",
    "Selim",
    "Murad",
    "Rama",
    "Haakon",
    "Margrethe",
    "Juliana",
    "Felipe",
    "Manuel",
    "Pedro",
    "Joseph",
    "Leopold",
    "Otto",
    "Conrad",
    "Rudolf",
    "Sigismund",
    "Casimir",
    "Stanislaus",
    "Vladimir",
    "Boris",
    "Michael",
];

fn ordinal_suffix(n: u32) -> &'static str {
    match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    }
}

/// Words that make an N-N a score: won 112-108, a 3-2 win.
const SCORE_BEFORE: &[&str] = &[
    "won", "win", "wins", "beat", "beats", "lost", "lose", "loses", "defeated", "edged", "tied",
    "drew", "leads", "led", "lead", "trailed", "trails", "trail", "scored", "final", "finished",
    "routed", "crushed", "downed", "topped", "blanked", "shut",
];
const SCORE_AFTER: &[&str] = &[
    "win", "victory", "loss", "defeat", "lead", "deficit", "score", "final", "decision", "rout",
    "thriller", "triumph", "upset", "draw", "tie",
];

/// Units per unit as narration says them.
const PER_UNITS: &[(&str, &str)] = &[
    ("km/h", "km per hour"),
    ("kph", "km per hour"),
    ("mi/h", "miles per hour"),
    ("m/s", "meters per second"),
    ("km/s", "kilometers per second"),
    ("ft/s", "feet per second"),
    ("MB/s", "megabytes per second"),
    ("GB/s", "gigabytes per second"),
    ("Mb/s", "megabits per second"),
    ("Gb/s", "gigabits per second"),
    ("Mbps", "megabits per second"),
    ("Gbps", "gigabits per second"),
];

/// Unit abbreviations and their singular words, for "a 5 kg camera".
const UNIT_WORDS: &[(&str, &str)] = &[
    ("kg", "kilogram"),
    ("g", "gram"),
    ("mg", "milligram"),
    ("km", "kilometer"),
    ("m", "meter"),
    ("cm", "centimeter"),
    ("mm", "millimeter"),
    ("mi", "mile"),
    ("ft", "foot"),
    ("lb", "pound"),
    ("lbs", "pound"),
    ("oz", "ounce"),
    ("l", "liter"),
    ("L", "liter"),
    ("ml", "milliliter"),
    ("mL", "milliliter"),
    ("min", "minute"),
    ("hr", "hour"),
    ("sec", "second"),
    ("MP", "megapixel"),
    ("GB", "gigabyte"),
    ("TB", "terabyte"),
    ("MB", "megabyte"),
    ("W", "watt"),
    ("kW", "kilowatt"),
    ("V", "volt"),
];

/// Time zones after a time, as narration says them.
const TIME_ZONES: &[(&str, &str)] = &[
    ("PT", "Pacific Time"),
    ("PST", "Pacific Standard Time"),
    ("PDT", "Pacific Daylight Time"),
    ("ET", "Eastern Time"),
    ("EST", "Eastern Standard Time"),
    ("EDT", "Eastern Daylight Time"),
    ("CT", "Central Time"),
    ("CST", "Central Standard Time"),
    ("CDT", "Central Daylight Time"),
    ("MT", "Mountain Time"),
    ("MST", "Mountain Standard Time"),
    ("MDT", "Mountain Daylight Time"),
    ("BST", "British Summer Time"),
    ("CET", "Central European Time"),
    ("CEST", "Central European Summer Time"),
];

/// The word-level rewrites: scores, units per unit, units as adjectives,
/// time zones after a time.
fn word_rules(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let bare = |w: &str| -> String {
        w.trim_matches(|c: char| ",.;:!?()\"'".contains(c))
            .to_string()
    };
    let lower = |w: &str| bare(w).to_lowercase();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let core = word.trim_end_matches(|c: char| ",.;:!?)".contains(c));
        let tail = &word[core.len()..];
        let prev = |k: usize| {
            i.checked_sub(k)
                .map(|j| lower(words[j]))
                .unwrap_or_default()
        };
        let next = words.get(i + 1).map(|w| lower(w)).unwrap_or_default();
        // won 112-108, a 3–2 win
        if let Some((a, b)) = core.split_once(['-', '–']) {
            let digits =
                |x: &str| (1..=3).contains(&x.len()) && x.chars().all(|c| c.is_ascii_digit());
            let game = (1..=3).any(|k| SCORE_BEFORE.contains(&prev(k).as_str()))
                || SCORE_AFTER.contains(&next.as_str());
            if digits(a) && digits(b) && game {
                out.push(format!("{a} to {b}{tail}"));
                continue;
            }
        }
        // 120 km/h, 120km/h
        if let Some((unit, spoken)) = PER_UNITS.iter().find(|(u, _)| {
            core == *u
                || core.strip_suffix(u).is_some_and(|n| {
                    !n.is_empty() && n.chars().all(|c| c.is_ascii_digit() || c == '.')
                })
        }) {
            let number = &core[..core.len() - unit.len()];
            let joiner = if number.is_empty() { "" } else { " " };
            out.push(format!("{number}{joiner}{spoken}{tail}"));
            continue;
        }
        // a 5 kg camera: after "a"/"an" a measure is an adjective, singular.
        let article = matches!(prev(2).as_str(), "a" | "an");
        let numeric = |w: &str| !w.is_empty() && w.chars().all(|c| c.is_ascii_digit() || c == '.');
        if article
            && tail.is_empty()
            && numeric(&bare(words[i.saturating_sub(1)]))
            && next.chars().next().is_some_and(|c| c.is_ascii_lowercase())
            && !matches!(
                next.as_str(),
                "of" | "per" | "in" | "and" | "or" | "to" | "a" | "the"
            )
            && let Some((_, singular)) = UNIT_WORDS.iter().find(|(u, _)| *u == core)
        {
            out.push(singular.to_string());
            continue;
        }
        // 9 a.m. PT, 15:00 CET
        let after_time = i > 0 && {
            let p = bare(words[i - 1]).to_lowercase();
            matches!(p.as_str(), "am" | "pm" | "a.m" | "p.m")
                || p.ends_with("am")
                    && p.trim_end_matches("am")
                        .chars()
                        .all(|c| c.is_ascii_digit() || c == ':')
                    && p.len() > 2
                || p.ends_with("pm")
                    && p.trim_end_matches("pm")
                        .chars()
                        .all(|c| c.is_ascii_digit() || c == ':')
                    && p.len() > 2
                || (p.len() >= 4
                    && p.contains(':')
                    && p.chars().all(|c| c.is_ascii_digit() || c == ':'))
        };
        if after_time && let Some((_, spoken)) = TIME_ZONES.iter().find(|(z, _)| *z == core) {
            out.push(format!("{spoken}{tail}"));
            continue;
        }
        out.push(word.to_string());
    }
    out.join(" ")
}

/// The value of a standard-form Roman numeral in capitals, 1–3999.
fn roman(word: &str) -> Option<u32> {
    if word.is_empty() || !word.chars().all(|c| "IVXLCDM".contains(c)) {
        return None;
    }
    let value = |c: char| match c {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        _ => 1000,
    };
    let digits: Vec<u32> = word.chars().map(value).collect();
    let mut total: i64 = 0;
    for (i, &d) in digits.iter().enumerate() {
        if digits.get(i + 1).is_some_and(|&next| next > d) {
            total -= i64::from(d);
        } else {
            total += i64::from(d);
        }
    }
    let total = u32::try_from(total)
        .ok()
        .filter(|t| (1..=3999).contains(t))?;
    // Only the canonical spelling counts: IIII or VX are not numerals here.
    (to_roman(total) == word).then_some(total)
}

fn to_roman(mut n: u32) -> String {
    let table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (value, letters) in table {
        while n >= value {
            out.push_str(letters);
            n -= value;
        }
    }
    out
}

/// Accented Latin letters to their base letters; ligatures and a few
/// letters with no decomposition spelled out. Other scripts pass through.
fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ß' => out.push_str("ss"),
            'æ' => out.push_str("ae"),
            'Æ' => out.push_str("Ae"),
            'œ' => out.push_str("oe"),
            'Œ' => out.push_str("Oe"),
            'ø' => out.push('o'),
            'Ø' => out.push('O'),
            'ł' => out.push('l'),
            'Ł' => out.push('L'),
            'đ' | 'ð' => out.push('d'),
            'Đ' | 'Ð' => out.push('D'),
            'þ' => out.push_str("th"),
            'Þ' => out.push_str("Th"),
            'ı' => out.push('i'),
            c if c.is_ascii() => out.push(c),
            c => out.extend(c.to_string().nfd().filter(|d| !is_combining_mark(*d))),
        }
    }
    out
}

fn digits(chars: &[char]) -> usize {
    chars.iter().take_while(|c| c.is_ascii_digit()).count()
}

fn ends_word(chars: &[char], at: usize) -> bool {
    chars.get(at).is_none_or(|c| !c.is_alphanumeric())
}

/// `2.5x`, `10x`, `3×` → the number, then "times" (the caller adds it).
/// Answers the length including the x.
fn multiplier(chars: &[char]) -> Option<usize> {
    let mut at = digits(chars);
    if at == 0 {
        return None;
    }
    if chars.get(at) == Some(&'.') && digits(&chars[at + 1..]) > 0 {
        at += 1 + digits(&chars[at + 1..]);
    }
    if !matches!(chars.get(at), Some('x' | 'X' | '×')) || !ends_word(chars, at + 1) {
        return None;
    }
    Some(at + 1)
}

#[cfg(test)]
mod tests {
    use super::prepare;

    #[test]
    fn accents_and_multipliers_become_readable() {
        assert_eq!(
            prepare("Kīlauea, Zürich, Straße"),
            "Kilauea, Zurich, Strasse"
        );
        assert_eq!(
            prepare("It is 2.5x faster, even 10× faster"),
            "It is 2.5 times faster, even 10 times faster"
        );
        assert_eq!(
            prepare("Chapter IV, World War II and Super Bowl LVIII."),
            "Chapter 4, World War 2 and Super Bowl 58."
        );
        assert_eq!(
            prepare("Visit https://www.promoshot.app/docs/, or http://example.com."),
            "Visit promoshot.app/docs, or example.com."
        );
        assert_eq!(
            prepare("The Lakers won 112-108, a 3–2 win."),
            "The Lakers won 112 to 108, a 3 to 2 win."
        );
        assert_eq!(
            prepare("It does 120 km/h, or 30 m/s."),
            "It does 120 km per hour, or 30 meters per second."
        );
        assert_eq!(
            prepare("a 5 kg camera, weighs 5 kg, a 2 kg of rice"),
            "a 5 kilogram camera, weighs 5 kg, a 2 kg of rice"
        );
        assert_eq!(
            prepare("at 9 a.m. PT or 5pm EST, 15:00 CET"),
            "at 9 a.m. Pacific Time or 5pm Eastern Standard Time, 15:00 Central European Time"
        );
        assert_eq!(
            prepare("Louis XIV, Elizabeth I and Rocky II"),
            "Louis the 14th, Elizabeth the 1st and Rocky 2"
        );
        for keep in [
            "Washington DC",
            "Size XL",
            "Malcolm X",
            "the 2023-2024 season",
            "call 555-1234",
            "the PT cruiser",
        ] {
            assert_eq!(prepare(keep), keep);
        }
        for keep in [
            "0x1f",
            "3:19 a.m.",
            "1990s",
            "max 5",
            "I like it",
            "the IV drip",
            "Act IIII",
        ] {
            assert_eq!(prepare(keep), keep);
        }
    }
}
