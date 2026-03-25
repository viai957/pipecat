use serde::{Deserialize, Serialize};

/// DTMF (Dual-Tone Multi-Frequency) keypad digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DtmfKey {
    #[serde(rename = "0")]
    Zero,
    #[serde(rename = "1")]
    One,
    #[serde(rename = "2")]
    Two,
    #[serde(rename = "3")]
    Three,
    #[serde(rename = "4")]
    Four,
    #[serde(rename = "5")]
    Five,
    #[serde(rename = "6")]
    Six,
    #[serde(rename = "7")]
    Seven,
    #[serde(rename = "8")]
    Eight,
    #[serde(rename = "9")]
    Nine,
    #[serde(rename = "*")]
    Star,
    #[serde(rename = "#")]
    Pound,
    #[serde(rename = "A")]
    A,
    #[serde(rename = "B")]
    B,
    #[serde(rename = "C")]
    C,
    #[serde(rename = "D")]
    D,
}

impl DtmfKey {
    /// Convert a character to a DTMF key, if valid.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '0' => Some(Self::Zero),
            '1' => Some(Self::One),
            '2' => Some(Self::Two),
            '3' => Some(Self::Three),
            '4' => Some(Self::Four),
            '5' => Some(Self::Five),
            '6' => Some(Self::Six),
            '7' => Some(Self::Seven),
            '8' => Some(Self::Eight),
            '9' => Some(Self::Nine),
            '*' => Some(Self::Star),
            '#' => Some(Self::Pound),
            'A' | 'a' => Some(Self::A),
            'B' | 'b' => Some(Self::B),
            'C' | 'c' => Some(Self::C),
            'D' | 'd' => Some(Self::D),
            _ => None,
        }
    }

    /// Convert the key to its character representation.
    pub fn as_char(self) -> char {
        match self {
            Self::Zero => '0',
            Self::One => '1',
            Self::Two => '2',
            Self::Three => '3',
            Self::Four => '4',
            Self::Five => '5',
            Self::Six => '6',
            Self::Seven => '7',
            Self::Eight => '8',
            Self::Nine => '9',
            Self::Star => '*',
            Self::Pound => '#',
            Self::A => 'A',
            Self::B => 'B',
            Self::C => 'C',
            Self::D => 'D',
        }
    }
}

impl std::fmt::Display for DtmfKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// Parse a string of DTMF characters into a vec of keys.
///
/// Invalid characters are silently skipped.
pub fn parse_dtmf_keys(s: &str) -> Vec<DtmfKey> {
    s.chars().filter_map(DtmfKey::from_char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_digits() {
        for c in "0123456789".chars() {
            assert!(DtmfKey::from_char(c).is_some(), "digit {c} should parse");
        }
    }

    #[test]
    fn special_keys() {
        assert_eq!(DtmfKey::from_char('*'), Some(DtmfKey::Star));
        assert_eq!(DtmfKey::from_char('#'), Some(DtmfKey::Pound));
    }

    #[test]
    fn letter_keys_case_insensitive() {
        assert_eq!(DtmfKey::from_char('A'), Some(DtmfKey::A));
        assert_eq!(DtmfKey::from_char('a'), Some(DtmfKey::A));
        assert_eq!(DtmfKey::from_char('D'), Some(DtmfKey::D));
        assert_eq!(DtmfKey::from_char('d'), Some(DtmfKey::D));
    }

    #[test]
    fn invalid_char() {
        assert_eq!(DtmfKey::from_char('X'), None);
        assert_eq!(DtmfKey::from_char(' '), None);
    }

    #[test]
    fn roundtrip_char() {
        let keys = [
            DtmfKey::Zero, DtmfKey::One, DtmfKey::Star, DtmfKey::Pound,
            DtmfKey::A, DtmfKey::D,
        ];
        for key in keys {
            assert_eq!(DtmfKey::from_char(key.as_char()), Some(key));
        }
    }

    #[test]
    fn parse_string() {
        let keys = parse_dtmf_keys("12*#AX9");
        assert_eq!(keys.len(), 6); // X is skipped
        assert_eq!(keys[0], DtmfKey::One);
        assert_eq!(keys[4], DtmfKey::A);
        assert_eq!(keys[5], DtmfKey::Nine);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", DtmfKey::Star), "*");
        assert_eq!(format!("{}", DtmfKey::Five), "5");
    }

    #[test]
    fn serde_roundtrip() {
        let key = DtmfKey::Pound;
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, "\"#\"");
        let parsed: DtmfKey = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, DtmfKey::Pound);
    }
}
