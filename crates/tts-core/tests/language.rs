use local_tts_core::{Language, text};
fn plain(text: &str, language: Language) -> String {
    text::prepare(text, language)
        .unwrap()
        .replace('\u{301}', "")
}

#[test]
fn english_dates_times_prices_units_and_ordinals() {
    for (input, expected) in [
        (
            "Dr. Smith paid $5.20 on 2026-10-03.",
            "doctor smith paid five dollars and twenty cents on october third, two thousand twenty six.",
        ),
        (
            "1 kg, 2.5kg, −3°C, +5%, 60 km/h.",
            "one kilogram, two point five kilograms, minus three degrees celsius, plus five percent, sixty kilometers per hour.",
        ),
        (
            "21st 12th 100th 0th",
            "twenty first twelfth one hundredth zeroth",
        ),
        (
            "1,234,567.05, please.",
            "one million two hundred thirty four thousand five hundred sixty seven point zero five, please.",
        ),
        (
            "12:30; 00:05; 23:00.",
            "twelve hours thirty minutes; zero hours five minutes; twenty three hours zero minutes.",
        ),
        (
            "1/2 2/3 3/4; 1/2 kg.",
            "one half two thirds three quarters; one half of a kilogram.",
        ),
        (
            "£1.01, €2.50, 3 USD; No. 42.",
            "one pound and one penny, two euros and fifty cents, three dollars; number forty two.",
        ),
        (
            "e.g. CPU vs. GPU",
            "for example central processing unit versus graphics processing unit",
        ),
        ("No. No. 5. etc.", "no. number five. et cetera."),
    ] {
        assert_eq!(plain(input, Language::En), expected, "{input}");
    }
}

#[test]
fn explicit_ranges_have_genitive_quantity_agreement() {
    assert_eq!(
        plain("0.5–2.5 kg; 5–10%", Language::En),
        "from zero point five to two point five kilograms; from five to ten percent"
    );
    assert_eq!(
        plain("5–21 кг; 1–2 мин; 1000–2000 м", Language::Ru),
        "от пяти до двадцати одного килограмма; от одной до двух минут; от одной тысячи до двух тысяч метров"
    );
    assert_eq!(
        plain("5–21 кг; 1–2 хв; 1000–2000 м", Language::Uk),
        "від п'яти до двадцяти одного кілограма; від однієї до двох хвилин; від однієї тисячі до двох тисяч метрів"
    );
    for language in [Language::En, Language::Ru, Language::Uk] {
        for input in ["5–10–15", "$5–10", "5–10 USD"] {
            assert!(text::prepare(input, language).is_err());
        }
    }
    assert!(text::prepare("0,5–2,5 кг", Language::Ru).is_err());
}

#[test]
fn russian_and_ukrainian_quantity_agreement_is_separate() {
    assert_eq!(
        plain(
            "1 кг 2кг 5 кг; 21 мин 22 мин 12 мин; 2,5 кг; 1,01 л.",
            Language::Ru
        ),
        "один килограмм два килограмма пять килограммов; двадцать одна минута двадцать две минуты двенадцать минут; две целых пять десятых килограмма; одна целая одна сотая литра."
    );
    assert_eq!(
        plain(
            "1 кг 2кг 5 кг; 21 хв 22 хв 12 хв; 2,5 кг; 1,01 л.",
            Language::Uk
        ),
        "один кілограм два кілограми п'ять кілограмів; двадцять одна хвилина двадцять дві хвилини дванадцять хвилин; дві цілих п'ять десятих кілограма; одна ціла одна сота літра."
    );
    assert_eq!(
        plain("21,02 ₴; 11%; −5°C; 1/2 кг; 21-й.", Language::Uk),
        "двадцять одна гривня і дві копійки; одинадцять відсотків; мінус п'ять градусів цельсія; одна друга кілограма; двадцять перший."
    );
    assert_eq!(
        plain("21,02 ₽; 11%; −5°C; 1/2 кг; 21-й.", Language::Ru),
        "двадцать один рубль и две копейки; одиннадцать процентов; минус пять градусов цельсия; одна вторая килограмма; двадцать первый."
    );
    assert_eq!(
        plain("1\u{a0}000 кг; 1\u{202f}234,5 м.", Language::Uk),
        "одна тисяча кілограмів; одна тисяча двісті тридцять чотири цілих п'ять десятих метра."
    );
}

#[test]
fn ordinal_stress_is_language_specific_and_survives_date_inflection() {
    assert_eq!(
        text::prepare("20-й 30-й 40-й 6-го 7-го 8-го", Language::Uk).unwrap(),
        "двадця́тий тридця́тий сороко́вий шо́стого сьо́мого во́сьмого"
    );
    assert_eq!(
        text::prepare("20-й 30-й 40-й 6-го 7-го 8-го", Language::Ru).unwrap(),
        "два́дцатый три́дцатый сороково́й шесто́го седьмо́го восьмо́го"
    );
    assert_eq!(
        text::prepare("2020-10-20", Language::Uk).unwrap(),
        "двадця́того жо́втня дві ти́сячі двадця́того ро́ку"
    );
    assert_eq!(
        text::prepare("2040-10-30", Language::Uk).unwrap(),
        "тридця́того жо́втня дві ти́сячі сороко́вого ро́ку"
    );
}

#[test]
fn calendar_validation_and_language_specific_readings() {
    assert_eq!(
        plain("03.10.2026", Language::Ru),
        "третьего октября две тысячи двадцать шестого года"
    );
    assert_eq!(
        plain("2026-10-03", Language::Uk),
        "третього жовтня дві тисячі двадцять шостого року"
    );
    assert_eq!(
        plain("2000-02-29", Language::Ru),
        "двадцать девятого февраля двухтысячного года"
    );
    assert_eq!(
        plain("1900-01-01", Language::Uk),
        "першого січня тисяча дев'ятисотого року"
    );
    assert_eq!(
        plain("12:30", Language::Ru),
        "двенадцать часов тридцать минут"
    );
    assert_eq!(
        plain("21:02", Language::Uk),
        "двадцять одна година дві хвилини"
    );
    for language in [Language::En, Language::Ru, Language::Uk] {
        for input in [
            "2026-02-29",
            "1900-02-29",
            "2026-04-31",
            "2026-00-01",
            "2026-12-00",
            "2100-01-01",
            "2026-1-01",
            "24:00",
            "12:60",
            "12:3",
            "12:30 pm",
        ] {
            assert!(
                text::prepare(input, language).is_err(),
                "{language:?}: {input}"
            );
        }
    }
}

#[test]
fn ambiguity_malformed_annotations_and_growth_are_bounded() {
    for language in [Language::En, Language::Ru, Language::Uk] {
        for input in [
            "v2",
            "2/0",
            "1/25",
            "1.2.3",
            "5-10",
            "1 000",
            "1000000000",
            "1.1234567890",
            "2\u{301}",
            "03/10/2026",
            "5kgX",
            "2026-10-03abc",
            "12:30:45",
            "hello\0world",
            "- 5",
            "$ 5",
            "1 / 2",
            "1 1/2",
            "5 - 10",
        ] {
            assert!(
                text::prepare(input, language).is_err(),
                "{language:?}: {input}"
            );
        }
        assert!(text::prepare(&"9 ".repeat(4999), language).is_err());
        assert!(text::prepare(&"x".repeat(10001), language).is_err());
    }
    for input in [
        "1,00",
        "1,000,00",
        "01,000",
        "$2.001",
        "$2 EUR",
        "1th",
        "11st",
        "02st",
        "03.10.2026",
    ] {
        assert!(text::prepare(input, Language::En).is_err(), "{input}");
    }
    assert!(text::prepare("з+ам+ок", Language::Ru).is_err());
    assert!(text::prepare("+ы", Language::Uk).is_err());
    assert!(text::validate_version(Some("text-v999")).is_err());
    assert!(text::validate_version(None).is_ok());
}

#[test]
fn prepared_output_is_idempotent_and_saved_stress_wins() {
    for (language, input) in [
        (Language::En, "Dr. Smith paid $5.20 on 2026-10-03."),
        (Language::Ru, "Дверной замок. Встреча 03.10.2026; 2,5 кг."),
        (Language::Uk, "Дверний замок. Зустріч 03.10.2026; 2,5 кг."),
    ] {
        let prepared = text::prepare(input, language).unwrap();
        assert_eq!(text::prepare(&prepared, language).unwrap(), prepared);
        if language != Language::En {
            let mut saved = local_tts_core::pronunciations::Pronunciations::new(language).unwrap();
            saved.insert("з+амок").unwrap();
            let expanded = text::expand(input, language).unwrap();
            let result =
                local_tts_core::context::stress(&saved.apply(&expanded).unwrap(), language)
                    .unwrap();
            assert!(result.contains("за́мок"));
            assert!(!result.contains("замо́к"));
        }
    }
}

#[test]
fn legacy_token_contracts_retain_original_numeric_rules() {
    assert!(local_tts_core::english_numbers::expand("$5").is_err());
    assert!(local_tts_core::piper_training::normalize("2 кг", Language::Uk).is_ok());
    assert!(local_tts_core::piper_training::normalize("2кг", Language::Uk).is_err());
    assert!(local_tts_core::piper_training::normalize("2026-10-03", Language::Uk).is_err());
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("cmudict.dict");
    std::fs::write(
        &path,
        "five F AY1 V\ndollars D AA1 L ER0 Z\ni AY1\nhave HH AE1 V\nwill W IH1 L\nread R IY1 D\n",
    )
    .unwrap();
    {
        let frontend = local_tts_core::english::English::load(&path).unwrap();
        assert!(frontend.phonemize_ipa("$5").is_err());
        assert!(frontend.phonemize_ipa_current("$5").is_ok());
        assert_ne!(
            frontend.phonemize("I have read.").unwrap(),
            frontend.phonemize("I will read.").unwrap()
        );
        assert!(frontend.phonemize("Ms.").is_ok());
        assert!(frontend.phonemize("0th").is_ok());
        std::fs::write(&path, "two T UW1\neleventh IH0 L EH1 V AH0 N TH\n").unwrap();
        let fractions = local_tts_core::english::English::load(&path).unwrap();
        assert_eq!(fractions.phonemize("2/11").unwrap(), "tˈu ɪlˈɛvənθs");
    }
}

#[test]
#[ignore = "requires installed Kokoro FP32/Q8/Q4, Vosk Russian and Piper Ukrainian"]
fn new_forms_tokenize_and_synthesize_with_real_models() {
    use local_tts_core::{Audio, SynthesisRequest, Synthesizer};
    use local_tts_core::{
        kokoro::{Kokoro, Variant},
        piper::Piper,
        vosk::Vosk,
    };
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("target/language-smoke");
    std::fs::create_dir_all(&out).unwrap();
    fn save(audio: Audio, path: &std::path::Path) {
        assert!(audio.samples.len() > audio.sample_rate as usize / 10);
        assert!(audio.samples.len() < audio.sample_rate as usize * 120);
        assert!(audio.samples.iter().all(|v| v.is_finite()));
        let rms = (audio
            .samples
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            / audio.samples.len() as f64)
            .sqrt();
        assert!(rms > 0.001);
        audio.write_wav(path).unwrap();
    }
    let en =
        "Dr. Smith paid $5.20 on 2026-10-03. I have read the report. The lead pipe is 2 m long.";
    for variant in Variant::ALL {
        let mut voice = Kokoro::load(&root.join("voices").join(variant.folder())).unwrap();
        for d in 2..=20 {
            voice.phonemize(&format!("2/{d}")).unwrap();
        }
        for input in [
            "Ms. Smith",
            "0th",
            "21st",
            "12th",
            "1000000th",
            "1 mm",
            "2 cm",
            "3 ml",
            "4 l",
            "5 min",
            "6 s",
            "7 °F",
            "8 °C",
            "9%",
            "10 m/s",
            "11 km/h",
            "0.5–2.5 kg",
            "1/2 kg",
            "CPU API e.g. GPU",
        ] {
            voice
                .phonemize(input)
                .unwrap_or_else(|e| panic!("{input}: {e}"));
        }
        let phonemes = voice.phonemize(en).unwrap();
        assert!(phonemes.contains("lˈɛd"));
        save(
            voice
                .synthesize(&SynthesisRequest {
                    text: en,
                    language: Language::En,
                    speaker: "af_heart",
                    duration_scale: 1.0,
                })
                .unwrap(),
            &out.join(format!("en-{}.wav", variant.id())),
        );
    }
    let mut ru = Vosk::load(&root.join("voices/vosk-ru")).unwrap();
    for input in [
        "21-й",
        "22-го",
        "1–2 мин",
        "1000–2000 м",
        "5–21 кг",
        "1 mm",
        "2 cm",
        "3 ml",
        "4 l",
        "5 min",
        "6 s",
        "7 °F",
        "8 °C",
        "9%",
        "10 m/s",
        "11 km/h",
        "1/2 kg",
        "CPU API т.е. GPU",
    ] {
        ru.tokenize(input)
            .unwrap_or_else(|e| panic!("{input}: {e}"));
    }
    let ru_text = "Встреча 03.10.2026. Вес 2,5 кг. Дверной замок.";
    assert_eq!(
        ru.tokenize("Дверной замок").unwrap(),
        ru.tokenize("Дверной зам+ок").unwrap()
    );
    let mut saved = local_tts_core::pronunciations::Pronunciations::new(Language::Ru).unwrap();
    saved.insert("з+амок").unwrap();
    ru.set_pronunciations(saved).unwrap();
    assert!(ru.prepare_text("Дверной замок").unwrap().contains("за́мок"));
    assert!(ru.prepare_text("Дверной зам+ок").unwrap().contains("замо́к"));
    ru.set_pronunciations(
        local_tts_core::pronunciations::Pronunciations::new(Language::Ru).unwrap(),
    )
    .unwrap();
    save(
        ru.synthesize(&SynthesisRequest {
            text: ru_text,
            language: Language::Ru,
            speaker: "F03",
            duration_scale: 1.0,
        })
        .unwrap(),
        &out.join("ru.wav"),
    );
    let mut uk = Piper::load(&root.join("voices/piper-uk")).unwrap();
    for input in [
        "20-й 30-й 40-й",
        "2020-10-20; 2040-10-30",
        "21-й",
        "22-го",
        "1–2 хв",
        "1000–2000 м",
        "5–21 кг",
        "1 mm",
        "2 cm",
        "3 ml",
        "4 l",
        "5 min",
        "6 s",
        "7 °F",
        "8 °C",
        "9%",
        "10 m/s",
        "11 km/h",
        "1/2 kg",
        "CPU API напр. GPU",
    ] {
        uk.tokenize(input)
            .unwrap_or_else(|e| panic!("{input}: {e}"));
    }
    assert_eq!(
        uk.tokenize("Дверний замок").unwrap(),
        uk.tokenize("Дверний зам+ок").unwrap()
    );
    let uk_text = "Зустріч 03.10.2026. Вага 2,5 кг. Дверний замок.";
    save(
        uk.synthesize(&SynthesisRequest {
            text: uk_text,
            language: Language::Uk,
            speaker: "lada",
            duration_scale: 1.0,
        })
        .unwrap(),
        &out.join("uk.wav"),
    );
}
