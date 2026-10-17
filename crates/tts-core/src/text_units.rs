//! Original measurement vocabulary; nominative quantities, not a case inflector.
use crate::Language;

#[derive(Clone, Copy)]
pub(crate) struct Unit {
    pub forms: [&'static str; 4], // one, few, many, fractional genitive singular
    pub feminine: bool,
}
impl Unit {
    pub fn label(self, n: u32, fraction: bool, language: Language) -> &'static str {
        if fraction {
            self.forms[if language == Language::En { 2 } else { 3 }]
        } else if language == Language::En {
            self.forms[if n == 1 { 0 } else { 2 }]
        } else {
            self.forms[crate::cyrillic_numbers::form(n)]
        }
    }
}

pub(crate) fn lookup(key: &str, language: Language) -> Option<Unit> {
    let key = match key {
        "км" => "km",
        "м" => "m",
        "см" => "cm",
        "мм" => "mm",
        "кг" => "kg",
        "г" => "g",
        "л" | "L" => "l",
        "мл" | "mL" => "ml",
        "ч" => "h",
        "год" if language == Language::Uk => "h",
        "мин" | "хв" => "min",
        "с" => "s",
        "км/ч" | "км/год" => "km/h",
        "м/с" => "m/s",
        "$" => "USD",
        "€" => "EUR",
        "£" => "GBP",
        "₽" | "руб" => "RUB",
        "₴" | "грн" => "UAH",
        key => key,
    };
    let (forms, feminine) = match (language, key) {
        (Language::En, "km") => (
            ["kilometer", "kilometers", "kilometers", "kilometer"],
            false,
        ),
        (Language::En, "m") => (["meter", "meters", "meters", "meter"], false),
        (Language::En, "cm") => (
            ["centimeter", "centimeters", "centimeters", "centimeter"],
            false,
        ),
        (Language::En, "mm") => (
            ["millimeter", "millimeters", "millimeters", "millimeter"],
            false,
        ),
        (Language::En, "kg") => (["kilogram", "kilograms", "kilograms", "kilogram"], false),
        (Language::En, "g") => (["gram", "grams", "grams", "gram"], false),
        (Language::En, "l") => (["liter", "liters", "liters", "liter"], false),
        (Language::En, "ml") => (
            ["milliliter", "milliliters", "milliliters", "milliliter"],
            false,
        ),
        (Language::En, "h") => (["hour", "hours", "hours", "hour"], false),
        (Language::En, "min") => (["minute", "minutes", "minutes", "minute"], false),
        (Language::En, "s") => (["second", "seconds", "seconds", "second"], false),
        (Language::En, "km/h") => (
            [
                "kilometer per hour",
                "kilometers per hour",
                "kilometers per hour",
                "kilometer per hour",
            ],
            false,
        ),
        (Language::En, "m/s") => (
            [
                "meter per second",
                "meters per second",
                "meters per second",
                "meter per second",
            ],
            false,
        ),
        (Language::En, "°C") => (
            [
                "degree celsius",
                "degrees celsius",
                "degrees celsius",
                "degree celsius",
            ],
            false,
        ),
        (Language::En, "°F") => (
            [
                "degree fahrenheit",
                "degrees fahrenheit",
                "degrees fahrenheit",
                "degree fahrenheit",
            ],
            false,
        ),
        (Language::En, "%") => (["percent"; 4], false),
        (Language::En, "USD") => (["dollar", "dollars", "dollars", "dollar"], false),
        (Language::En, "EUR") => (["euro", "euros", "euros", "euro"], false),
        (Language::En, "GBP") => (["pound", "pounds", "pounds", "pound"], false),
        (Language::Ru, "km") => (["киломе́тр", "киломе́тра", "киломе́тров", "киломе́тра"], false),
        (Language::Ru, "m") => (["метр", "ме́тра", "ме́тров", "ме́тра"], false),
        (Language::Ru, "cm") => (
            ["сантиме́тр", "сантиме́тра", "сантиме́тров", "сантиме́тра"],
            false,
        ),
        (Language::Ru, "mm") => (
            ["миллиме́тр", "миллиме́тра", "миллиме́тров", "миллиме́тра"],
            false,
        ),
        (Language::Ru, "kg") => (
            ["килогра́мм", "килогра́мма", "килогра́ммов", "килогра́мма"],
            false,
        ),
        (Language::Ru, "g") => (["грамм", "гра́мма", "гра́ммов", "гра́мма"], false),
        (Language::Ru, "l") => (["литр", "ли́тра", "ли́тров", "ли́тра"], false),
        (Language::Ru, "ml") => (
            ["миллили́тр", "миллили́тра", "миллили́тров", "миллили́тра"],
            false,
        ),
        (Language::Ru, "h") => (["час", "часа́", "часо́в", "ча́са"], false),
        (Language::Ru, "min") => (["мину́та", "мину́ты", "мину́т", "мину́ты"], true),
        (Language::Ru, "s") => (["секу́нда", "секу́нды", "секу́нд", "секу́нды"], true),
        (Language::Ru, "km/h") => (
            [
                "киломе́тр в час",
                "киломе́тра в час",
                "киломе́тров в час",
                "киломе́тра в час",
            ],
            false,
        ),
        (Language::Ru, "m/s") => (
            [
                "метр в секу́нду",
                "ме́тра в секу́нду",
                "ме́тров в секу́нду",
                "ме́тра в секу́нду",
            ],
            false,
        ),
        (Language::Ru, "°C") => (
            [
                "гра́дус це́льсия",
                "гра́дуса це́льсия",
                "гра́дусов це́льсия",
                "гра́дуса це́льсия",
            ],
            false,
        ),
        (Language::Ru, "°F") => (
            [
                "гра́дус фаренге́йта",
                "гра́дуса фаренге́йта",
                "гра́дусов фаренге́йта",
                "гра́дуса фаренге́йта",
            ],
            false,
        ),
        (Language::Ru, "%") => (["проце́нт", "проце́нта", "проце́нтов", "проце́нта"], false),
        (Language::Ru, "USD") => (["до́ллар", "до́ллара", "до́лларов", "до́ллара"], false),
        (Language::Ru, "EUR") => (["е́вро"; 4], false),
        (Language::Ru, "GBP") => (["фунт", "фу́нта", "фу́нтов", "фу́нта"], false),
        (Language::Ru, "RUB") => (["рубль", "рубля́", "рубле́й", "рубля́"], false),
        (Language::Ru, "UAH") => (["гри́вна", "гри́вны", "гри́вен", "гри́вны"], true),
        (Language::Uk, "km") => (["кіломе́тр", "кіломе́три", "кіломе́трів", "кіломе́тра"], false),
        (Language::Uk, "m") => (["метр", "ме́три", "ме́трів", "ме́тра"], false),
        (Language::Uk, "cm") => (
            ["сантиме́тр", "сантиме́три", "сантиме́трів", "сантиме́тра"],
            false,
        ),
        (Language::Uk, "mm") => (["міліме́тр", "міліме́три", "міліме́трів", "міліме́тра"], false),
        (Language::Uk, "kg") => (["кілогра́м", "кілогра́ми", "кілогра́мів", "кілогра́ма"], false),
        (Language::Uk, "g") => (["грам", "гра́ми", "гра́мів", "гра́ма"], false),
        (Language::Uk, "l") => (["літр", "лі́три", "лі́трів", "лі́тра"], false),
        (Language::Uk, "ml") => (["мілілі́тр", "мілілі́три", "мілілі́трів", "мілілі́тра"], false),
        (Language::Uk, "h") => (["годи́на", "годи́ни", "годи́н", "годи́ни"], true),
        (Language::Uk, "min") => (["хвили́на", "хвили́ни", "хвили́н", "хвили́ни"], true),
        (Language::Uk, "s") => (["секу́нда", "секу́нди", "секу́нд", "секу́нди"], true),
        (Language::Uk, "km/h") => (
            [
                "кіломе́тр за годи́ну",
                "кіломе́три за годи́ну",
                "кіломе́трів за годи́ну",
                "кіломе́тра за годи́ну",
            ],
            false,
        ),
        (Language::Uk, "m/s") => (
            [
                "метр за секу́нду",
                "ме́три за секу́нду",
                "ме́трів за секу́нду",
                "ме́тра за секу́нду",
            ],
            false,
        ),
        (Language::Uk, "°C") => (
            [
                "гра́дус це́льсія",
                "гра́дуси це́льсія",
                "гра́дусів це́льсія",
                "гра́дуса це́льсія",
            ],
            false,
        ),
        (Language::Uk, "°F") => (
            [
                "гра́дус фаренге́йта",
                "гра́дуси фаренге́йта",
                "гра́дусів фаренге́йта",
                "гра́дуса фаренге́йта",
            ],
            false,
        ),
        (Language::Uk, "%") => (["відсо́ток", "відсо́тки", "відсо́тків", "відсо́тка"], false),
        (Language::Uk, "USD") => (["до́лар", "до́лари", "до́ларів", "до́лара"], false),
        (Language::Uk, "EUR") => (["є́вро"; 4], false),
        (Language::Uk, "GBP") => (["фунт", "фу́нти", "фу́нтів", "фу́нта"], false),
        (Language::Uk, "RUB") => (["рубль", "рублі́", "рублі́в", "рубля́"], false),
        (Language::Uk, "UAH") => (["гри́вня", "гри́вні", "гри́вень", "гри́вні"], true),
        _ => return None,
    };
    Some(Unit { forms, feminine })
}

pub(crate) fn minor(key: &str, language: Language) -> Option<Unit> {
    let forms = match (language, key) {
        (Language::En, "USD" | "EUR") => ["cent", "cents", "cents", "cent"],
        (Language::En, "GBP") => ["penny", "pence", "pence", "penny"],
        (Language::Ru, "USD" | "EUR") => ["цент", "це́нта", "це́нтов", "це́нта"],
        (Language::Ru, "GBP") => ["пе́нни"; 4],
        (Language::Ru, "RUB" | "UAH") => ["копе́йка", "копе́йки", "копе́ек", "копе́йки"],
        (Language::Uk, "USD" | "EUR") => ["цент", "це́нти", "це́нтів", "це́нта"],
        (Language::Uk, "GBP") => ["пе́нні"; 4],
        (Language::Uk, "RUB" | "UAH") => ["копі́йка", "копі́йки", "копі́йок", "копі́йки"],
        _ => return None,
    };
    Some(Unit {
        forms,
        feminine: matches!(key, "RUB" | "UAH"),
    })
}
