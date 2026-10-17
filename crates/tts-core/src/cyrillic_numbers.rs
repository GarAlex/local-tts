//! Original, bounded RU/UK cardinal readings with lexical stress.
//! Nominative masculine units; feminine thousands. No contextual case/gender guessing.
use crate::Language;
use unicode_normalization::char::is_combining_mark;

struct Words {
    small: [&'static str; 20],
    tens: [&'static str; 10],
    hundreds: [&'static str; 10],
    feminine: [&'static str; 2],
    thousand: [&'static str; 3],
    million: [&'static str; 3],
    separators: [&'static str; 2],
}
const RU: Words = Words {
    small: [
        "ноль",
        "оди́н",
        "два",
        "три",
        "четы́ре",
        "пять",
        "шесть",
        "семь",
        "во́семь",
        "де́вять",
        "де́сять",
        "оди́ннадцать",
        "двена́дцать",
        "трина́дцать",
        "четы́рнадцать",
        "пятна́дцать",
        "шестна́дцать",
        "семна́дцать",
        "восемна́дцать",
        "девятна́дцать",
    ],
    tens: [
        "",
        "",
        "два́дцать",
        "три́дцать",
        "со́рок",
        "пятьдеся́т",
        "шестьдеся́т",
        "се́мьдесят",
        "во́семьдесят",
        "девяно́сто",
    ],
    hundreds: [
        "",
        "сто",
        "две́сти",
        "три́ста",
        "четы́реста",
        "пятьсо́т",
        "шестьсо́т",
        "семьсо́т",
        "восемьсо́т",
        "девятьсо́т",
    ],
    feminine: ["одна́", "две"],
    thousand: ["ты́сяча", "ты́сячи", "ты́сяч"],
    million: ["миллио́н", "миллио́на", "миллио́нов"],
    separators: ["то́чка", "запята́я"],
};
const UK: Words = Words {
    small: [
        "нуль",
        "оди́н",
        "два",
        "три",
        "чоти́ри",
        "п'ять",
        "шість",
        "сім",
        "ві́сім",
        "де́в'ять",
        "де́сять",
        "одина́дцять",
        "двана́дцять",
        "трина́дцять",
        "чотирна́дцять",
        "п'ятна́дцять",
        "шістна́дцять",
        "сімна́дцять",
        "вісімна́дцять",
        "дев'ятна́дцять",
    ],
    tens: [
        "",
        "",
        "два́дцять",
        "три́дцять",
        "со́рок",
        "п'ятдеся́т",
        "шістдеся́т",
        "сімдеся́т",
        "вісімдеся́т",
        "дев'яно́сто",
    ],
    hundreds: [
        "",
        "сто",
        "дві́сті",
        "три́ста",
        "чоти́риста",
        "п'ятсо́т",
        "шістсо́т",
        "сімсо́т",
        "вісімсо́т",
        "дев'ятсо́т",
    ],
    feminine: ["одна́", "дві"],
    thousand: ["ти́сяча", "ти́сячі", "ти́сяч"],
    million: ["мільйо́н", "мільйо́ни", "мільйо́нів"],
    separators: ["кра́пка", "ко́ма"],
};
pub(crate) fn form(n: u32) -> usize {
    if (11..=14).contains(&(n % 100)) {
        2
    } else {
        match n % 10 {
            1 => 0,
            2..=4 => 1,
            _ => 2,
        }
    }
}
fn cardinal(n: u32, feminine: bool, w: &Words) -> String {
    if n < 20 {
        return if feminine && (1..=2).contains(&n) {
            w.feminine[(n - 1) as usize]
        } else {
            w.small[n as usize]
        }
        .into();
    }
    let (label, remainder) = if n < 100 {
        (w.tens[(n / 10) as usize].to_string(), n % 10)
    } else if n < 1000 {
        (w.hundreds[(n / 100) as usize].to_string(), n % 100)
    } else {
        let scale = if n < 1_000_000 { 1000 } else { 1_000_000 };
        let labels = if scale == 1000 { w.thousand } else { w.million };
        (
            format!(
                "{} {}",
                cardinal(n / scale, scale == 1000, w),
                labels[form(n / scale)]
            ),
            n % scale,
        )
    };
    if remainder == 0 {
        label
    } else {
        format!("{label} {}", cardinal(remainder, feminine, w))
    }
}
fn attached(c: char) -> bool {
    c.is_alphanumeric() || is_combining_mark(c) || "'’ʼ".contains(c)
}
pub(crate) fn read_cardinal(n: u32, feminine: bool, language: Language) -> String {
    cardinal(
        n,
        feminine,
        if language == Language::Ru { &RU } else { &UK },
    )
}
/// Expand unsigned integers below one billion. Leading zeroes and decimal
/// fractional digits are spoken individually. Dot and comma are decimal separators;
/// grouping is not supported. Dates, ordinals, signs, units and currencies must
/// be spelled out. No changes are made to already written words/stress overrides.
pub fn expand(text: &str, language: Language) -> Result<String, String> {
    let w = match language {
        Language::Ru => &RU,
        Language::Uk => &UK,
        _ => {
            return Err("Cyrillic number expansion requires Russian or Ukrainian".into());
        }
    };
    if text.chars().count() > 10000 {
        return Err("text exceeds 10000 characters".into());
    }
    let chars: Vec<_> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let start = i;
        if i > 0 && (attached(chars[i - 1]) || "/:-+.,$€£₴₽−–—".contains(chars[i - 1]))
        {
            return Err(
                "spell out dates, times, signed numbers, currencies and embedded digits".into(),
            );
        }
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        let integer: String = chars[start..i].iter().collect();
        if integer.len() > 9 {
            return Err("numbers must be below one billion; spell out larger values".into());
        }
        let mut spoken = if integer.len() > 1 && integer.starts_with('0') {
            integer
                .bytes()
                .map(|b| w.small[(b - b'0') as usize])
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            cardinal(integer.parse().map_err(|_| "invalid number")?, false, w)
        };
        let mut decimal = false;
        if i + 1 < chars.len() && ".,".contains(chars[i]) && chars[i + 1].is_ascii_digit() {
            spoken.push(' ');
            spoken.push_str(w.separators[usize::from(chars[i] == ',')]);
            decimal = true;
            i += 1;
            let fraction_start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                spoken.push(' ');
                spoken.push_str(w.small[(chars[i] as u8 - b'0') as usize]);
                i += 1;
            }
            if i - fraction_start > 9 {
                return Err("decimal fractions may contain at most nine digits".into());
            }
        }
        if i < chars.len()
            && (attached(chars[i])
                || "/:-+%$€£₴₽−–—".contains(chars[i])
                || (i + 1 < chars.len()
                    && ".,".contains(chars[i])
                    && chars[i + 1].is_ascii_digit()))
        {
            return Err("spell out ordinals, ranges, dates, times, units and percentages".into());
        }
        // A three-digit group after whitespace may be a thousands separator.
        if !decimal && i < chars.len() && chars[i].is_whitespace() {
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let next = j;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j - next == 3 {
                return Err(
                    "space-grouped numbers are ambiguous; remove grouping or spell them out".into(),
                );
            }
        }
        out.push_str(&spoken);
        if out.chars().count() > 10000 {
            return Err("expanded text exceeds 10000 characters".into());
        }
    }
    if out.chars().count() > 10000 {
        return Err("expanded text exceeds 10000 characters".into());
    }
    Ok(out)
}
/// Genitive counts for explicitly delimited ranges; not sentence-wide case inference.
pub(crate) fn read_genitive(n: u32, feminine: bool, language: Language) -> String {
    let ru = language == Language::Ru;
    let small = if ru {
        [
            "нуля́",
            "одного́",
            "двух",
            "трёх",
            "четырёх",
            "пяти́",
            "шести́",
            "семи́",
            "восьми́",
            "девяти́",
            "десяти́",
            "одиннадцати́",
            "двенадцати́",
            "тринадцати́",
            "четырнадцати́",
            "пятнадцати́",
            "шестнадцати́",
            "семнадцати́",
            "восемнадцати́",
            "девятнадцати́",
        ]
    } else {
        [
            "нуля́",
            "одного́",
            "двох",
            "трьох",
            "чотирьо́х",
            "п'яти́",
            "шести́",
            "семи́",
            "восьми́",
            "дев'яти́",
            "десяти́",
            "одинадцяти́",
            "дванадцяти́",
            "тринадцяти́",
            "чотирнадцяти́",
            "п'ятнадцяти́",
            "шістнадцяти́",
            "сімнадцяти́",
            "вісімнадцяти́",
            "дев'ятнадцяти́",
        ]
    };
    if n < 20 {
        return if n == 1 && feminine {
            if ru { "одно́й" } else { "одніє́ї" }
        } else {
            small[n as usize]
        }
        .into();
    }
    let tens = if ru {
        [
            "",
            "",
            "двадцати́",
            "тридцати́",
            "сорока́",
            "пятидесяти́",
            "шестидесяти́",
            "семидесяти́",
            "восьмидесяти́",
            "девяно́ста",
        ]
    } else {
        [
            "",
            "",
            "двадцяти́",
            "тридцяти́",
            "сорока́",
            "п'ятдесяти́",
            "шістдесяти́",
            "сімдесяти́",
            "вісімдесяти́",
            "дев'яно́ста",
        ]
    };
    let hundreds = if ru {
        [
            "",
            "ста",
            "двухсо́т",
            "трёхсо́т",
            "четырёхсо́т",
            "пятисо́т",
            "шестисо́т",
            "семисо́т",
            "восьмисо́т",
            "девятисо́т",
        ]
    } else {
        [
            "",
            "ста",
            "двохсо́т",
            "трьохсо́т",
            "чотирьохсо́т",
            "п'ятисо́т",
            "шестисо́т",
            "семисо́т",
            "восьмисо́т",
            "дев'ятисо́т",
        ]
    };
    let (prefix, remainder) = if n < 100 {
        (tens[(n / 10) as usize].to_string(), n % 10)
    } else if n < 1000 {
        (hundreds[(n / 100) as usize].to_string(), n % 100)
    } else {
        let scale = if n < 1_000_000 { 1000 } else { 1_000_000 };
        let one = form(n / scale) == 0;
        let noun = match (ru, scale == 1000, one) {
            (true, true, true) => "ты́сячи",
            (true, true, false) => "ты́сяч",
            (false, true, true) => "ти́сячі",
            (false, true, false) => "ти́сяч",
            (true, false, true) => "миллио́на",
            (true, false, false) => "миллио́нов",
            (false, false, true) => "мільйо́на",
            (false, false, false) => "мільйо́нів",
        };
        (
            format!(
                "{} {noun}",
                read_genitive(n / scale, scale == 1000, language)
            ),
            n % scale,
        )
    };
    if remainder == 0 {
        prefix
    } else {
        format!("{prefix} {}", read_genitive(remainder, feminine, language))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plain(s: &str) -> String {
        s.replace('\u{301}', "")
    }
    #[test]
    fn language_specific_cardinals_and_scale_agreement() {
        assert_eq!(
            plain(&expand("0 21 42 1001 22000 111000 21000000", Language::Ru).unwrap()),
            "ноль двадцать один сорок два одна тысяча один двадцать две тысячи сто одиннадцать тысяч двадцать один миллион"
        );
        assert_eq!(
            plain(&expand("0 21 42 1001 22000 111000 21000000", Language::Uk).unwrap()),
            "нуль двадцять один сорок два одна тисяча один двадцять дві тисячі сто одинадцять тисяч двадцять один мільйон"
        );
        assert_eq!(
            plain(&expand("999999999", Language::Uk).unwrap()),
            "дев'ятсот дев'яносто дев'ять мільйонів дев'ятсот дев'яносто дев'ять тисяч дев'ятсот дев'яносто дев'ять"
        );
        for (n, expected) in [
            (1, 0),
            (2, 1),
            (5, 2),
            (11, 2),
            (12, 2),
            (14, 2),
            (21, 0),
            (22, 1),
            (101, 0),
            (112, 2),
        ] {
            assert_eq!(form(n), expected);
        }
    }
    #[test]
    fn fractional_digits_leading_zeroes_and_existing_stress() {
        assert_eq!(
            expand("3,05 007! Ґ+анок.", Language::Uk).unwrap(),
            "три ко́ма нуль п'ять нуль нуль сім! Ґ+анок."
        );
        assert_eq!(
            expand("3.05 007!", Language::Ru).unwrap(),
            "три то́чка ноль пять ноль ноль семь!"
        );
        for lang in [Language::Ru, Language::Uk] {
            for n in 0..=10000 {
                let output = expand(&n.to_string(), lang).unwrap();
                assert!(!output.chars().any(|c| c.is_ascii_digit()));
                crate::frontend::normalize(&output, lang).unwrap();
            }
        }
    }
    #[test]
    fn rejects_ambiguous_and_unbounded_forms() {
        for text in [
            "1 000",
            "12:30",
            "2026-10-02",
            "02.10.2026",
            "$5",
            "5₴",
            "5%",
            "21-й",
            "v2",
            "2кг",
            "2/3",
            "-5",
            "+5",
            "−5",
            "1000000000",
            "1.2.3",
            "а́2",
            "2́",
            "3.1234567890",
            "1'000",
            "1\u{a0}000",
        ] {
            assert!(expand(text, Language::Uk).is_err(), "{text}");
        }
        assert!(expand("1", Language::En).is_err());
        assert!(expand(&"9 ".repeat(4999), Language::Ru).is_err());
        assert!(expand(&"x".repeat(10001), Language::Ru).is_err());
    }
}
