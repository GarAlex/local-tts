//! Conservative English cardinal/decimal expansion. No date or currency guessing.
const SMALL: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
pub(crate) fn cardinal(n: u32) -> String {
    if n < 20 {
        return SMALL[n as usize].into();
    }
    if n < 100 {
        return if n.is_multiple_of(10) {
            TENS[(n / 10) as usize].into()
        } else {
            format!("{} {}", TENS[(n / 10) as usize], SMALL[(n % 10) as usize])
        };
    }
    for (scale, label) in [(1_000_000, "million"), (1000, "thousand"), (100, "hundred")] {
        if n >= scale {
            let mut out = format!("{} {label}", cardinal(n / scale));
            if !n.is_multiple_of(scale) {
                out.push(' ');
                out.push_str(&cardinal(n % scale));
            }
            return out;
        }
    }
    unreachable!()
}
pub fn expand(text: &str) -> Result<String, String> {
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
        if start > 0
            && (chars[start - 1].is_alphanumeric() || "/:-+.,$€£".contains(chars[start - 1]))
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
                .map(|c| SMALL[(c - b'0') as usize])
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            cardinal(integer.parse().map_err(|_| "invalid number")?)
        };
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            spoken.push_str(" point");
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                spoken.push(' ');
                spoken.push_str(SMALL[(chars[i] as u8 - b'0') as usize]);
                i += 1;
            }
        }
        if i < chars.len()
            && (chars[i].is_alphanumeric()
                || "/:-+%".contains(chars[i])
                || (i + 1 < chars.len()
                    && ",.".contains(chars[i])
                    && chars[i + 1].is_ascii_digit()))
        {
            return Err(
                "spell out ordinals, ranges, dates, times, percentages and grouped numbers".into(),
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cardinals_decimals_and_leading_zeroes() {
        assert_eq!(
            expand("I have 21 books and 3.05 coins.").unwrap(),
            "I have twenty one books and three point zero five coins."
        );
        assert_eq!(
            expand("0 007 100 1001 999999999").unwrap(),
            "zero zero zero seven one hundred one thousand one nine hundred ninety nine million nine hundred ninety nine thousand nine hundred ninety nine"
        );
        assert_eq!(
            expand("Take 42, please.").unwrap(),
            "Take forty two, please."
        );
    }
    #[test]
    fn ambiguous_formats_fail_explicitly() {
        for text in [
            "1,000",
            "12:30",
            "2026-10-02",
            "$5",
            "5%",
            "21st",
            "v2",
            "2/3",
            "-5",
            "+5",
            "1000000000",
            "1.2.3",
        ] {
            assert!(expand(text).is_err(), "{text}");
        }
    }
}
