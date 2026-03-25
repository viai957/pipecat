use serde::{Deserialize, Serialize};

/// Language identifiers for speech-to-text and text-to-speech.
///
/// Uses BCP-47 language tags. The enum covers the most common languages
/// used in voice AI; additional languages can be represented with `Other`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    #[serde(rename = "en")]
    English,
    #[serde(rename = "en-US")]
    EnglishUs,
    #[serde(rename = "en-GB")]
    EnglishGb,
    #[serde(rename = "en-AU")]
    EnglishAu,
    #[serde(rename = "en-IN")]
    EnglishIn,
    #[serde(rename = "es")]
    Spanish,
    #[serde(rename = "es-ES")]
    SpanishEs,
    #[serde(rename = "es-MX")]
    SpanishMx,
    #[serde(rename = "fr")]
    French,
    #[serde(rename = "fr-FR")]
    FrenchFr,
    #[serde(rename = "fr-CA")]
    FrenchCa,
    #[serde(rename = "de")]
    German,
    #[serde(rename = "de-DE")]
    GermanDe,
    #[serde(rename = "it")]
    Italian,
    #[serde(rename = "it-IT")]
    ItalianIt,
    #[serde(rename = "pt")]
    Portuguese,
    #[serde(rename = "pt-BR")]
    PortugueseBr,
    #[serde(rename = "pt-PT")]
    PortuguesePt,
    #[serde(rename = "zh")]
    Chinese,
    #[serde(rename = "zh-CN")]
    ChineseCn,
    #[serde(rename = "zh-TW")]
    ChineseTw,
    #[serde(rename = "ja")]
    Japanese,
    #[serde(rename = "ja-JP")]
    JapaneseJp,
    #[serde(rename = "ko")]
    Korean,
    #[serde(rename = "ko-KR")]
    KoreanKr,
    #[serde(rename = "hi")]
    Hindi,
    #[serde(rename = "hi-IN")]
    HindiIn,
    #[serde(rename = "ar")]
    Arabic,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "nl")]
    Dutch,
    #[serde(rename = "pl")]
    Polish,
    #[serde(rename = "sv")]
    Swedish,
    #[serde(rename = "tr")]
    Turkish,
    #[serde(rename = "vi")]
    Vietnamese,
    #[serde(rename = "th")]
    Thai,
    #[serde(rename = "uk")]
    Ukrainian,
    #[serde(rename = "cs")]
    Czech,
    #[serde(rename = "da")]
    Danish,
    #[serde(rename = "fi")]
    Finnish,
    #[serde(rename = "el")]
    Greek,
    #[serde(rename = "he")]
    Hebrew,
    #[serde(rename = "hu")]
    Hungarian,
    #[serde(rename = "id")]
    Indonesian,
    #[serde(rename = "ms")]
    Malay,
    #[serde(rename = "no")]
    Norwegian,
    #[serde(rename = "ro")]
    Romanian,
    #[serde(rename = "sk")]
    Slovak,
    #[serde(rename = "bg")]
    Bulgarian,
    #[serde(rename = "hr")]
    Croatian,
    #[serde(rename = "ca")]
    Catalan,
    #[serde(rename = "ta")]
    Tamil,
    #[serde(rename = "te")]
    Telugu,
    /// Any BCP-47 language tag not covered by the enum.
    Other(String),
}

impl Language {
    /// Return the BCP-47 tag as a string slice.
    pub fn as_bcp47(&self) -> &str {
        match self {
            Self::English => "en",
            Self::EnglishUs => "en-US",
            Self::EnglishGb => "en-GB",
            Self::EnglishAu => "en-AU",
            Self::EnglishIn => "en-IN",
            Self::Spanish => "es",
            Self::SpanishEs => "es-ES",
            Self::SpanishMx => "es-MX",
            Self::French => "fr",
            Self::FrenchFr => "fr-FR",
            Self::FrenchCa => "fr-CA",
            Self::German => "de",
            Self::GermanDe => "de-DE",
            Self::Italian => "it",
            Self::ItalianIt => "it-IT",
            Self::Portuguese => "pt",
            Self::PortugueseBr => "pt-BR",
            Self::PortuguesePt => "pt-PT",
            Self::Chinese => "zh",
            Self::ChineseCn => "zh-CN",
            Self::ChineseTw => "zh-TW",
            Self::Japanese => "ja",
            Self::JapaneseJp => "ja-JP",
            Self::Korean => "ko",
            Self::KoreanKr => "ko-KR",
            Self::Hindi => "hi",
            Self::HindiIn => "hi-IN",
            Self::Arabic => "ar",
            Self::Russian => "ru",
            Self::Dutch => "nl",
            Self::Polish => "pl",
            Self::Swedish => "sv",
            Self::Turkish => "tr",
            Self::Vietnamese => "vi",
            Self::Thai => "th",
            Self::Ukrainian => "uk",
            Self::Czech => "cs",
            Self::Danish => "da",
            Self::Finnish => "fi",
            Self::Greek => "el",
            Self::Hebrew => "he",
            Self::Hungarian => "hu",
            Self::Indonesian => "id",
            Self::Malay => "ms",
            Self::Norwegian => "no",
            Self::Romanian => "ro",
            Self::Slovak => "sk",
            Self::Bulgarian => "bg",
            Self::Croatian => "hr",
            Self::Catalan => "ca",
            Self::Tamil => "ta",
            Self::Telugu => "te",
            Self::Other(tag) => tag.as_str(),
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_bcp47())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcp47_tags() {
        assert_eq!(Language::EnglishUs.as_bcp47(), "en-US");
        assert_eq!(Language::Japanese.as_bcp47(), "ja");
        assert_eq!(Language::Other("sw-KE".into()).as_bcp47(), "sw-KE");
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", Language::French), "fr");
        assert_eq!(format!("{}", Language::ChineseCn), "zh-CN");
    }

    #[test]
    fn serde_roundtrip() {
        let lang = Language::EnglishUs;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, "\"en-US\"");
        let parsed: Language = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Language::EnglishUs);
    }

    #[test]
    fn serde_other_variant() {
        let lang = Language::Other("sw-KE".into());
        let json = serde_json::to_string(&lang).unwrap();
        let parsed: Language = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, lang);
    }

    #[test]
    fn equality() {
        assert_eq!(Language::English, Language::English);
        assert_ne!(Language::English, Language::EnglishUs);
    }
}
