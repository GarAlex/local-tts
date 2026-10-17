//! Stable language catalogue shared by library, CLI and Studio.
use crate::Language;
impl Language {
    pub const ALL: [Self; 10] = [
        Self::En,
        Self::Ru,
        Self::Uk,
        Self::De,
        Self::Fr,
        Self::Es,
        Self::It,
        Self::Pt,
        Self::Pl,
        Self::Nl,
    ];
    pub const fn code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ru => "ru",
            Self::Uk => "uk",
            Self::De => "de",
            Self::Fr => "fr",
            Self::Es => "es",
            Self::It => "it",
            Self::Pt => "pt",
            Self::Pl => "pl",
            Self::Nl => "nl",
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::En => "English",
            Self::Ru => "Russian",
            Self::Uk => "Ukrainian",
            Self::De => "German",
            Self::Fr => "French",
            Self::Es => "Spanish",
            Self::It => "Italian",
            Self::Pt => "Portuguese",
            Self::Pl => "Polish",
            Self::Nl => "Dutch",
        }
    }
    pub const fn abi_id(self) -> u32 {
        match self {
            Self::En => 0,
            Self::Ru => 1,
            Self::Uk => 2,
            Self::De => 3,
            Self::Fr => 4,
            Self::Es => 5,
            Self::It => 6,
            Self::Pt => 7,
            Self::Pl => 8,
            Self::Nl => 9,
        }
    }
    pub fn from_abi(id: u32) -> Result<Self, String> {
        Self::ALL
            .get(id as usize)
            .copied()
            .ok_or_else(|| "unknown language".into())
    }
    pub const fn is_cyrillic(self) -> bool {
        matches!(self, Self::Ru | Self::Uk)
    }
    pub const fn is_european_extension(self) -> bool {
        !matches!(self, Self::En | Self::Ru | Self::Uk)
    }
}
impl std::str::FromStr for Language {
    type Err = String;
    fn from_str(code: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|l| l.code() == code)
            .ok_or_else(|| {
                format!("unsupported language {code}; use en, ru, uk, de, fr, es, it, pt, pl or nl")
            })
    }
}
