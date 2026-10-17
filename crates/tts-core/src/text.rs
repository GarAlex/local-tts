//! Bounded, original EN/RU/UK text normalization. No external language data.
//! Ambiguous or unsupported numeric syntax returns an error instead of losing digits.
use crate::{
    Language,
    text_units::{self, Unit},
};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

pub const VERSION: &str = "text-v1";
const LIMIT: usize = 10000;

pub fn validate_version(version: Option<&str>) -> Result<(), String> {
    if version.is_some_and(|v| v != VERSION) {
        return Err("unsupported text normalization version".into());
    }
    Ok(())
}

/// Expand text and apply limited contextual stress. Explicit annotations win.
/// Engines with saved corrections apply them between `expand` and contextual stress.
pub fn prepare(text: &str, language: Language) -> Result<String, String> {
    crate::context::stress(&expand(text, language)?, language)
}

/// Expand dates, quantities and whitelisted abbreviations; canonicalize Unicode/stress.
/// EN numeric dates use ISO only; RU/UK additionally accept DD.MM.YYYY.
/// Measurements use nominative quantity agreement, not arbitrary sentence inflection.
pub fn expand(text: &str, language: Language) -> Result<String, String> {
    if language.is_european_extension() {
        if text.trim().is_empty()
            || text
                .chars()
                .any(|c| c.is_numeric() || (c.is_control() && !c.is_whitespace()))
        {
            return Err(
                "For this language, use nonempty text with numbers written as spoken words".into(),
            );
        }
        return crate::frontend::normalize(text, language);
    }

    if text.chars().count() > LIMIT {
        return Err("text exceeds 10000 characters".into());
    }
    if text.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return Err("control characters are not supported in speech text".into());
    }
    let text: String = text.nfc().collect();
    let text = text
        .replace(['’', 'ʼ', '‘'], "'")
        .replace(['“', '”', '«', '»'], "\"");
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut size = 0;
    let mut i = 0;
    while i < chars.len() {
        if boundary_before(&chars, i)
            && let Some((replacement, end)) = abbreviation(&chars, i, language)
        {
            emit(&mut out, &mut size, replacement)?;
            if chars[end - 1] == '.' && chars[end..].iter().all(|c| c.is_whitespace()) {
                emit(&mut out, &mut size, ".")?;
            }
            i = end;
            continue;
        }
        let numeric_prefix =
            "+-−$€£₽₴№#".contains(chars[i]) && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit());
        if "+-−$€£₽₴№#".contains(chars[i]) && chars.get(i + 1).is_some_and(|c| c.is_whitespace())
        {
            let mut next = i + 1;
            while chars.get(next).is_some_and(|c| c.is_whitespace()) {
                next += 1;
            }
            if chars.get(next).is_some_and(|c| c.is_ascii_digit()) {
                return Err("a numeric prefix must touch its digits; spell out list markers or spaced signs".into());
            }
        }
        if chars[i].is_ascii_digit() || numeric_prefix {
            if !boundary_before(&chars, i) {
                return Err("embedded digits need an explicit spoken spelling".into());
            }
            let (spoken, end) = number(&chars, i, language)?;
            emit(&mut out, &mut size, &spoken)?;
            i = end;
        } else {
            let c = if chars[i] == '—' { '-' } else { chars[i] };
            emit(&mut out, &mut size, &c.to_string())?;
            i += 1;
        }
    }
    let normalized = crate::frontend::normalize(&out, language)?;
    if normalized.chars().count() > LIMIT {
        return Err("normalized text exceeds 10000 characters".into());
    }
    Ok(normalized)
}
fn emit(out: &mut String, size: &mut usize, text: &str) -> Result<(), String> {
    *size += text.chars().count();
    if *size > LIMIT {
        return Err("expanded text exceeds 10000 characters".into());
    }
    out.push_str(text);
    Ok(())
}
fn attached(c: char) -> bool {
    c.is_alphanumeric() || is_combining_mark(c) || "'’ʼ".contains(c)
}
fn boundary_before(chars: &[char], i: usize) -> bool {
    i == 0 || !(attached(chars[i - 1]) || "/:-+.,$€£₽₴−–".contains(chars[i - 1]))
}
fn digits(chars: &[char], start: usize) -> (String, usize) {
    let mut end = start;
    while end < chars.len() && chars[end].is_ascii_digit() {
        end += 1;
    }
    (chars[start..end].iter().collect(), end)
}
fn value(s: &str) -> Result<u32, String> {
    if s.is_empty() || s.len() > 9 {
        return Err("numbers must contain one to nine digits".into());
    }
    s.parse().map_err(|_| "invalid number".into())
}
fn cardinal(n: u32, feminine: bool, language: Language) -> String {
    if language == Language::En {
        crate::english_numbers::cardinal(n)
    } else {
        crate::cyrillic_numbers::read_cardinal(n, feminine, language)
    }
}
fn raw_number(
    integer: &str,
    fraction: &str,
    separator: char,
    language: Language,
) -> Result<String, String> {
    let text = format!(
        "{integer}{}{fraction}",
        if fraction.is_empty() {
            String::new()
        } else {
            separator.to_string()
        }
    );
    if language == Language::En {
        crate::english_numbers::expand(&text)
    } else {
        crate::cyrillic_numbers::expand(&text, language)
    }
}
fn quantity(
    integer: u32,
    fraction: &str,
    unit: Unit,
    language: Language,
) -> Result<String, String> {
    let spoken = if fraction.is_empty() {
        cardinal(integer, unit.feminine, language)
    } else if language == Language::En {
        raw_number(&integer.to_string(), fraction, '.', language)?
    } else {
        let whole = match (language, crate::cyrillic_numbers::form(integer)) {
            (Language::Ru, 0) => "це́лая",
            (Language::Ru, _) => "це́лых",
            (Language::Uk, 0) => "ці́ла",
            _ => "ці́лих",
        };
        format!(
            "{} {whole} {}",
            cardinal(integer, true, language),
            decimal_fraction(fraction, language)?
        )
    };
    Ok(format!(
        "{spoken} {}",
        unit.label(integer, !fraction.is_empty(), language)
    ))
}
fn decimal_fraction(fraction: &str, language: Language) -> Result<String, String> {
    let n = value(fraction)?;
    let singular = crate::cyrillic_numbers::form(n) == 0;
    let labels = match language {
        Language::Ru => [
            ["деся́тая", "деся́тых"],
            ["со́тая", "со́тых"],
            ["ты́сячная", "ты́сячных"],
            ["десятиты́сячная", "десятиты́сячных"],
            ["стоты́сячная", "стоты́сячных"],
            ["миллио́нная", "миллио́нных"],
            ["десятимиллио́нная", "десятимиллио́нных"],
            ["стомиллио́нная", "стомиллио́нных"],
            ["миллиа́рдная", "миллиа́рдных"],
        ],
        _ => [
            ["деся́та", "деся́тих"],
            ["со́та", "со́тих"],
            ["ти́сячна", "ти́сячних"],
            ["десятити́сячна", "десятити́сячних"],
            ["стоти́сячна", "стоти́сячних"],
            ["мільйо́нна", "мільйо́нних"],
            ["десятимільйо́нна", "десятимільйо́нних"],
            ["стомільйо́нна", "стомільйо́нних"],
            ["мілья́рдна", "мілья́рдних"],
        ],
    };
    Ok(format!(
        "{} {}",
        cardinal(n, true, language),
        labels[fraction.len() - 1][usize::from(!singular)]
    ))
}
fn number(chars: &[char], start: usize, language: Language) -> Result<(String, usize), String> {
    let mut i = start;
    let mut prefix = String::new();
    let mut currency = None;
    match chars[i] {
        '+' | '-' | '−' => {
            prefix.push_str(match (language, chars[i] == '+') {
                (Language::En, true) => "plus ",
                (Language::En, false) => "minus ",
                (Language::Ru, true) => "плюс ",
                (Language::Ru, false) => "ми́нус ",
                (Language::Uk, true) => "плюс ",
                _ => "мі́нус ",
            });
            i += 1;
        }
        '$' | '€' | '£' | '₽' | '₴' => {
            currency = Some(currency_key(chars[i]));
            i += 1;
        }
        '№' | '#' => {
            prefix.push_str(if language == Language::En {
                "number "
            } else {
                "но́мер "
            });
            i += 1;
        }
        _ => {}
    }
    let (mut integer, end) = digits(chars, i);
    i = end;
    if let Some((spoken, end)) = date(chars, &integer, i, language)? {
        if !prefix.is_empty() || currency.is_some() {
            return Err("dates cannot have a numeric prefix".into());
        }
        return Ok((spoken, end));
    }
    if chars.get(i) == Some(&':') {
        if !prefix.is_empty() || currency.is_some() || integer.len() > 2 {
            return Err("invalid clock time".into());
        }
        let (minute, end) = digits(chars, i + 1);
        let h = value(&integer)?;
        let m = value(&minute)?;
        if minute.len() != 2 || h > 23 || m > 59 {
            return Err("use a valid 24-hour HH:MM time".into());
        }
        check_end(chars, end)?;
        let suffix: String = chars[end..].iter().take(6).collect();
        let suffix = suffix.trim_start().to_ascii_lowercase();
        if ["am", "pm", "a.m.", "p.m."].iter().any(|s| {
            suffix
                .strip_prefix(s)
                .is_some_and(|after| after.chars().next().is_none_or(|c| !attached(c)))
        }) {
            return Err("clock times use 24-hour HH:MM; spell out AM/PM readings".into());
        }
        return Ok((
            format!(
                "{} {}",
                quantity(h, "", text_units::lookup("h", language).unwrap(), language)?,
                quantity(
                    m,
                    "",
                    text_units::lookup("min", language).unwrap(),
                    language
                )?
            ),
            end,
        ));
    }
    // English comma grouping; Cyrillic dot/comma always mean a decimal.
    let grouping = if language == Language::En {
        ','
    } else {
        '\u{202f}'
    };
    if chars
        .get(i)
        .is_some_and(|c| *c == grouping || (language != Language::En && *c == '\u{a0}'))
        && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
    {
        if integer.len() > 3 || integer.starts_with('0') {
            return Err("invalid thousands grouping".into());
        }
        let actual = chars[i];
        while chars.get(i) == Some(&actual) && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
        {
            let (part, end) = digits(chars, i + 1);
            if part.len() != 3 {
                return Err("thousands groups must contain exactly three digits".into());
            }
            integer.push_str(&part);
            i = end;
        }
    }
    let n = value(&integer)?;
    let mut fraction = String::new();
    let mut separator = '.';
    if chars
        .get(i)
        .is_some_and(|c| *c == '.' || (language != Language::En && *c == ','))
        && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
    {
        separator = chars[i];
        (fraction, i) = digits(chars, i + 1);
        value(&fraction)?;
    }
    if fraction.is_empty() && currency.is_none() {
        if language == Language::En {
            let ending: String = chars.iter().skip(i).take(2).collect();
            if ["st", "nd", "rd", "th"].contains(&ending.as_str()) {
                let expected = if (11..=13).contains(&(n % 100)) {
                    "th"
                } else {
                    match n % 10 {
                        1 => "st",
                        2 => "nd",
                        3 => "rd",
                        _ => "th",
                    }
                };
                if ending != expected || (integer.len() > 1 && integer.starts_with('0')) {
                    return Err("invalid English ordinal suffix".into());
                }
                check_end(chars, i + 2)?;
                return Ok((format!("{prefix}{}", ordinal_en(n)), i + 2));
            }
        }
        if chars.get(i) == Some(&'/') {
            let (denominator, end) = digits(chars, i + 1);
            let d = value(&denominator)?;
            if !(2..=20).contains(&d) {
                return Err("fractions support denominators 2–20; spell out other forms".into());
            }
            let mut unit_start = end;
            while chars.get(unit_start).is_some_and(|c| c.is_whitespace()) {
                unit_start += 1;
            }
            let (key, unit_end) = unit_at(chars, unit_start, language);
            let mut spoken = fraction_words(n, d, language)?;
            let end = if let Some(key) = key {
                if text_units::minor(&key, language).is_some() {
                    return Err("spell out fractional currency amounts".into());
                }
                let unit = text_units::lookup(&key, language).unwrap();
                if language == Language::En {
                    spoken.push_str(&format!(" of a {}", unit.forms[0]));
                } else {
                    spoken.push_str(&format!(" {}", unit.forms[3]));
                }
                unit_end
            } else {
                end
            };
            check_end(chars, end)?;
            return Ok((format!("{prefix}{spoken}"), end));
        }
        if language != Language::En && chars.get(i) == Some(&'-') {
            let mut end = i + 1;
            while chars.get(end).is_some_and(|c| c.is_alphabetic()) {
                end += 1;
            }
            let suffix: String = chars[i + 1..end].iter().collect();
            let ending = match suffix.as_str() {
                "й" => Some(Ending::Masculine),
                "го" => Some(Ending::Genitive),
                _ => None,
            };
            if let Some(ending) = ending {
                check_end(chars, end)?;
                return Ok((
                    format!("{prefix}{}", ordinal_cyr(n, language, ending)?),
                    end,
                ));
            }
        }
    }
    if chars.get(i) == Some(&'–') && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
        if !prefix.is_empty() || currency.is_some() {
            return Err("spell out signed or currency ranges".into());
        }
        let (upper, mut end) = digits(chars, i + 1);
        let upper_n = value(&upper)?;
        let mut upper_fraction = String::new();
        if chars.get(end) == Some(&'.') && chars.get(end + 1).is_some_and(|c| c.is_ascii_digit()) {
            (upper_fraction, end) = digits(chars, end + 1);
            value(&upper_fraction)?;
        }
        if language != Language::En && (!fraction.is_empty() || !upper_fraction.is_empty()) {
            return Err("spell out fractional Cyrillic ranges".into());
        }
        let mut unit_start = end;
        while chars.get(unit_start).is_some_and(|c| c.is_whitespace()) {
            unit_start += 1;
        }
        let (key, unit_end) = unit_at(chars, unit_start, language);
        let unit = key
            .as_deref()
            .and_then(|key| text_units::lookup(key, language));
        if key
            .as_deref()
            .is_some_and(|key| text_units::minor(key, language).is_some())
        {
            return Err("spell out currency ranges".into());
        }
        if key.is_some() {
            end = unit_end;
        }
        check_end(chars, end)?;
        let feminine = unit.is_some_and(|unit| unit.feminine);
        let mut spoken = if language == Language::En {
            format!(
                "from {} to {}",
                raw_number(&integer, &fraction, separator, language)?,
                raw_number(&upper, &upper_fraction, '.', language)?
            )
        } else {
            format!(
                "{} {} до {}",
                if language == Language::Ru {
                    "от"
                } else {
                    "від"
                },
                crate::cyrillic_numbers::read_genitive(n, feminine, language),
                crate::cyrillic_numbers::read_genitive(upper_n, feminine, language)
            )
        };
        if let Some(unit) = unit {
            let label = if language == Language::En {
                unit.label(upper_n, !upper_fraction.is_empty(), language)
            } else {
                unit.forms[if crate::cyrillic_numbers::form(upper_n) == 0 {
                    3
                } else {
                    2
                }]
            };
            spoken.push(' ');
            spoken.push_str(label);
        }
        return Ok((spoken, end));
    }
    // ASCII-space grouping conflicts with lists; request an unambiguous spelling.
    if fraction.is_empty() && chars.get(i) == Some(&' ') {
        let mut next = i;
        while chars.get(next) == Some(&' ') {
            next += 1;
        }
        if digits(chars, next).0.len() == 3 {
            return Err("space-grouped numbers are ambiguous; use EN commas or RU/UK nonbreaking spaces, or remove grouping".into());
        }
    }
    let mut unit_start = i;
    while chars.get(unit_start).is_some_and(|c| c.is_whitespace()) {
        unit_start += 1;
    }
    let (unit_key, unit_end) = unit_at(chars, unit_start, language);
    let unit_key = unit_key.as_deref();
    if unit_key.is_none() && unit_start > i {
        let (_, end) = digits(chars, unit_start);
        if end > unit_start && chars.get(end) == Some(&'/') {
            return Err("spell out mixed-number fractions".into());
        }
    }
    if unit_key.is_none()
        && unit_start > i
        && chars.get(unit_start).is_some_and(|c| "/:+-−–".contains(*c))
    {
        return Err(
            "spell out spaced arithmetic/fractions, or use an explicit supported numeric form"
                .into(),
        );
    }
    if currency.is_some() && unit_key.is_some() {
        return Err("use only one currency marker".into());
    }
    let key = currency.or(unit_key);
    let spoken = if let Some(key) = key {
        let unit =
            text_units::lookup(key, language).ok_or("unsupported currency in this language")?;
        if text_units::minor(key, language).is_some() {
            money(n, &fraction, key, unit, language)?
        } else {
            quantity(n, &fraction, unit, language)?
        }
    } else {
        raw_number(&integer, &fraction, separator, language)?
    };
    if unit_key.is_some() {
        i = unit_end;
    }
    check_end(chars, i)?;
    Ok((format!("{prefix}{spoken}"), i))
}
fn currency_key(c: char) -> &'static str {
    match c {
        '$' => "USD",
        '€' => "EUR",
        '£' => "GBP",
        '₽' => "RUB",
        _ => "UAH",
    }
}
fn unit_at(chars: &[char], start: usize, language: Language) -> (Option<String>, usize) {
    let mut end = start;
    while chars
        .get(end)
        .is_some_and(|c| c.is_alphabetic() || "/°%$€£₽₴".contains(*c))
    {
        end += 1;
    }
    let key: String = chars[start..end].iter().collect();
    if text_units::lookup(&key, language).is_some() {
        let canonical = if key.chars().count() == 1 && "$€£₽₴".contains(&key) {
            currency_key(key.chars().next().unwrap()).to_string()
        } else if ["руб", "грн"].contains(&key.as_str()) {
            if key == "руб" { "RUB" } else { "UAH" }.to_string()
        } else {
            key
        };
        (Some(canonical), end)
    } else {
        (None, start)
    }
}
fn check_end(chars: &[char], end: usize) -> Result<(), String> {
    if chars
        .get(end)
        .is_some_and(|c| attached(*c) || "/:-+%$€£₽₴−–".contains(*c))
        || (chars.get(end).is_some_and(|c| ".,".contains(*c))
            && chars.get(end + 1).is_some_and(|c| c.is_ascii_digit()))
    {
        return Err("unsupported or ambiguous numeric form; spell out its intended reading".into());
    }
    Ok(())
}
fn money(
    n: u32,
    fraction: &str,
    key: &str,
    unit: Unit,
    language: Language,
) -> Result<String, String> {
    if fraction.len() > 2 {
        return Err("currency amounts support at most two fractional digits".into());
    }
    let mut out = quantity(n, "", unit, language)?;
    if !fraction.is_empty() {
        let cents = value(fraction)? * if fraction.len() == 1 { 10 } else { 1 };
        if cents > 0 {
            let minor =
                text_units::minor(key, language).ok_or("unsupported minor currency unit")?;
            out.push_str(match language {
                Language::En => " and ",
                Language::Ru => " и ",
                Language::Uk => " і ",
                _ => return Err("unsupported currency language".into()),
            });
            out.push_str(&quantity(cents, "", minor, language)?);
        }
    }
    Ok(out)
}
fn fraction_words(n: u32, d: u32, language: Language) -> Result<String, String> {
    if language == Language::En {
        let denominator = match (d, n == 1) {
            (2, true) => "half".into(),
            (2, false) => "halves".into(),
            (4, true) => "quarter".into(),
            (4, false) => "quarters".into(),
            (_, true) => ordinal_en(d),
            _ => format!("{}s", ordinal_en(d)),
        };
        Ok(format!("{} {denominator}", cardinal(n, false, language)))
    } else {
        let adjective = ordinal_cyr(
            d,
            language,
            if crate::cyrillic_numbers::form(n) == 0 {
                Ending::Feminine
            } else {
                Ending::PluralGenitive
            },
        )?;
        Ok(format!("{} {adjective}", cardinal(n, true, language)))
    }
}
fn ordinal_en(n: u32) -> String {
    let words = cardinal(n, false, Language::En);
    let (prefix, last) = words.rsplit_once(' ').unwrap_or(("", words.as_str()));
    let ordinal = match last {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        word if word.ends_with('y') => format!("{}ieth", &word[..word.len() - 1]),
        word => format!("{word}th"),
    };
    if prefix.is_empty() {
        ordinal
    } else {
        format!("{prefix} {ordinal}")
    }
}

#[derive(Clone, Copy)]
enum Ending {
    Masculine,
    Genitive,
    Feminine,
    PluralGenitive,
}
fn ordinal_cyr(n: u32, language: Language, ending: Ending) -> Result<String, String> {
    if !(1..=99).contains(&n) {
        return Err("Cyrillic ordinal is outside the supported range".into());
    }
    let ru = language == Language::Ru;
    let stems = if ru {
        [
            "",
            "пе́рв",
            "втор",
            "тре́ть",
            "четвёрт",
            "пя́т",
            "шест",
            "седьм",
            "восьм",
            "девя́т",
            "деся́т",
            "оди́ннадцат",
            "двена́дцат",
            "трина́дцат",
            "четы́рнадцат",
            "пятна́дцат",
            "шестна́дцат",
            "семна́дцат",
            "восемна́дцат",
            "девятна́дцат",
        ]
    } else {
        [
            "",
            "пе́рш",
            "дру́г",
            "тре́т",
            "четве́рт",
            "п'я́т",
            "шо́ст",
            "сьо́м",
            "во́сьм",
            "де́в'ят",
            "деся́т",
            "одина́дцят",
            "двана́дцят",
            "трина́дцят",
            "чотирна́дцят",
            "п'ятна́дцят",
            "шістна́дцят",
            "сімна́дцят",
            "вісімна́дцят",
            "дев'ятна́дцят",
        ]
    };
    let tens = if ru {
        [
            "",
            "",
            "два́дцат",
            "три́дцат",
            "сороков",
            "пятидеся́т",
            "шестидеся́т",
            "семидеся́т",
            "восьмидеся́т",
            "девяно́ст",
        ]
    } else {
        [
            "",
            "",
            "двадця́т",
            "тридця́т",
            "сороко́в",
            "п'ятдеся́т",
            "шістдеся́т",
            "сімдеся́т",
            "вісімдеся́т",
            "дев'яно́ст",
        ]
    };
    if n >= 20 && !n.is_multiple_of(10) {
        return Ok(format!(
            "{} {}",
            cardinal(n / 10 * 10, false, language),
            ordinal_cyr(n % 10, language, ending)?
        ));
    }
    let mut stem = if n < 20 {
        stems[n as usize]
    } else {
        tens[(n / 10) as usize]
    }
    .to_string();
    // These Russian ordinals have ending stress; Ukrainian fortieth retains stem stress.
    let ending_stress = ru && [2, 6, 7, 8, 40].contains(&n);
    let suffix = match (ru, n == 3, ending) {
        (true, true, Ending::Masculine) => "ий",
        (false, true, Ending::Masculine) => "ій",
        (true, false, Ending::Masculine) => {
            if ending_stress {
                "ой"
            } else {
                "ый"
            }
        }
        (false, false, Ending::Masculine) => "ий",
        (true, true, Ending::Genitive) => "его",
        (false, true, Ending::Genitive) => "ього",
        (true, true, Ending::Feminine) => "я",
        (false, true, Ending::Feminine) => "я",
        (_, true, Ending::PluralGenitive) => {
            if ru {
                "их"
            } else {
                "іх"
            }
        }
        (_, false, Ending::Genitive) => "ого",
        (_, false, Ending::Feminine) => {
            if ru {
                "ая"
            } else {
                "а"
            }
        }
        (_, false, Ending::PluralGenitive) => {
            if ru {
                "ых"
            } else {
                "их"
            }
        }
    };
    if ending_stress {
        let mut cs = suffix.chars();
        stem.push(cs.next().unwrap());
        stem.push('\u{301}');
        stem.extend(cs);
    } else {
        stem.push_str(suffix);
    }
    Ok(stem)
}
fn date(
    chars: &[char],
    first: &str,
    end: usize,
    language: Language,
) -> Result<Option<(String, usize)>, String> {
    let iso = first.len() == 4 && chars.get(end) == Some(&'-');
    let local = language != Language::En && first.len() == 2 && chars.get(end) == Some(&'.');
    if !iso && !local {
        return Ok(None);
    }
    let sep = if iso { '-' } else { '.' };
    let (second, middle) = digits(chars, end + 1);
    if second.len() != 2 || chars.get(middle) != Some(&sep) {
        if iso {
            return Err("use a valid ISO YYYY-MM-DD date".into());
        }
        return Ok(None);
    }
    let (third, finish) = digits(chars, middle + 1);
    if third.len() != if iso { 2 } else { 4 } {
        return Err("use YYYY-MM-DD or Cyrillic DD.MM.YYYY dates".into());
    }
    check_end(chars, finish)?;
    let a = value(first)?;
    let month = value(&second)?;
    let b = value(&third)?;
    let (year, day) = if iso { (a, b) } else { (b, a) };
    if !(1900..=2099).contains(&year) || !(1..=12).contains(&month) {
        return Err("dates support years 1900–2099 and valid calendar months/days".into());
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day == 0 || day > days[(month - 1) as usize] {
        return Err("invalid calendar date".into());
    }
    let months = match language {
        Language::En => [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        Language::Ru => [
            "января́",
            "февраля́",
            "ма́рта",
            "апре́ля",
            "ма́я",
            "ию́ня",
            "ию́ля",
            "а́вгуста",
            "сентября́",
            "октября́",
            "ноября́",
            "декабря́",
        ],
        Language::Uk => [
            "сі́чня",
            "лю́того",
            "бе́резня",
            "кві́тня",
            "тра́вня",
            "че́рвня",
            "ли́пня",
            "се́рпня",
            "ве́ресня",
            "жо́втня",
            "листопа́да",
            "гру́дня",
        ],
        _ => return Err("unsupported date language".into()),
    };
    let spoken = if language == Language::En {
        format!(
            "{} {}, {}",
            months[(month - 1) as usize],
            ordinal_en(day),
            cardinal(year, false, language)
        )
    } else {
        // A date is read as a genitive calendar phrase; no arbitrary case prediction.
        let day = ordinal_cyr(day, language, Ending::Genitive)?;
        let year_words = if year == 2000 {
            if language == Language::Ru {
                "двухты́сячного"
            } else {
                "двоти́сячного"
            }
            .to_string()
        } else if year == 1900 {
            if language == Language::Ru {
                "ты́сяча девятисо́того"
            } else {
                "ти́сяча дев'ятисо́того"
            }
            .to_string()
        } else {
            format!(
                "{} {}",
                cardinal(year / 100 * 100, false, language),
                ordinal_cyr(year % 100, language, Ending::Genitive)?
            )
        };
        format!(
            "{day} {} {year_words} {}",
            months[(month - 1) as usize],
            if language == Language::Ru {
                "го́да"
            } else {
                "ро́ку"
            }
        )
    };
    Ok(Some((spoken, finish)))
}
fn abbreviation(chars: &[char], start: usize, language: Language) -> Option<(&'static str, usize)> {
    let pairs: &[(&str, &str)] = match language {
        Language::En => &[
            ("Dr.", "doctor"),
            ("Mr.", "mister"),
            ("Mrs.", "missus"),
            ("Ms.", "miz"),
            ("Prof.", "professor"),
            ("e.g.", "for example"),
            ("i.e.", "that is"),
            ("etc.", "et cetera"),
            ("vs.", "versus"),
            ("No.", "number"),
            ("CPU", "central processing unit"),
            ("GPU", "graphics processing unit"),
            ("API", "application programming interface"),
        ],
        Language::Ru => &[
            ("т.е.", "то есть"),
            ("т. е.", "то есть"),
            ("т.д.", "так да́лее"),
            ("т. д.", "так да́лее"),
            ("т.п.", "то́му подо́бное"),
            ("т. п.", "то́му подо́бное"),
            ("напр.", "наприме́р"),
            ("проф.", "профе́ссор"),
            ("CPU", "центра́льный проце́ссор"),
            ("GPU", "графи́ческий проце́ссор"),
            ("API", "програ́ммный интерфе́йс"),
        ],
        Language::Uk => &[
            ("тобто", "тобто"),
            ("т.д.", "так да́лі"),
            ("т. д.", "так да́лі"),
            ("т.п.", "то́му поді́бне"),
            ("т. п.", "то́му поді́бне"),
            ("напр.", "наприкла́д"),
            ("проф.", "профе́сор"),
            ("CPU", "центра́льний проце́сор"),
            ("GPU", "графі́чний проце́сор"),
            ("API", "програ́мний інтерфе́йс"),
        ],
        _ => &[],
    };
    for (key, replacement) in pairs {
        let length = key.chars().count();
        let end = start + length;
        if *key == "No." {
            let next = chars.iter().skip(end).find(|c| !c.is_whitespace());
            if !next.is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
        }
        if chars.get(start..end).is_some_and(|s| {
            if language == Language::En {
                s.iter().copied().eq(key.chars())
            } else {
                s.iter()
                    .flat_map(|c| c.to_lowercase())
                    .eq(key.chars().flat_map(char::to_lowercase))
            }
        }) && chars.get(end).is_none_or(|c| !attached(*c))
        {
            return Some((replacement, end));
        }
    }
    None
}
