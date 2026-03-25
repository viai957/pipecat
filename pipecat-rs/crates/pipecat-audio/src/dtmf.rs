//! DTMF keypad entry types.

use std::fmt;

/// Represents a standard DTMF keypad entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeypadEntry {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Star,
    Pound,
}

impl fmt::Display for KeypadEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = match self {
            KeypadEntry::Zero => '0',
            KeypadEntry::One => '1',
            KeypadEntry::Two => '2',
            KeypadEntry::Three => '3',
            KeypadEntry::Four => '4',
            KeypadEntry::Five => '5',
            KeypadEntry::Six => '6',
            KeypadEntry::Seven => '7',
            KeypadEntry::Eight => '8',
            KeypadEntry::Nine => '9',
            KeypadEntry::Star => '*',
            KeypadEntry::Pound => '#',
        };
        write!(f, "{c}")
    }
}

impl KeypadEntry {
    /// Try to parse a character into a keypad entry.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '0' => Some(KeypadEntry::Zero),
            '1' => Some(KeypadEntry::One),
            '2' => Some(KeypadEntry::Two),
            '3' => Some(KeypadEntry::Three),
            '4' => Some(KeypadEntry::Four),
            '5' => Some(KeypadEntry::Five),
            '6' => Some(KeypadEntry::Six),
            '7' => Some(KeypadEntry::Seven),
            '8' => Some(KeypadEntry::Eight),
            '9' => Some(KeypadEntry::Nine),
            '*' => Some(KeypadEntry::Star),
            '#' => Some(KeypadEntry::Pound),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_digits() {
        assert_eq!(KeypadEntry::Zero.to_string(), "0");
        assert_eq!(KeypadEntry::One.to_string(), "1");
        assert_eq!(KeypadEntry::Nine.to_string(), "9");
    }

    #[test]
    fn display_symbols() {
        assert_eq!(KeypadEntry::Star.to_string(), "*");
        assert_eq!(KeypadEntry::Pound.to_string(), "#");
    }

    #[test]
    fn from_char_roundtrip() {
        for c in "0123456789*#".chars() {
            let entry = KeypadEntry::from_char(c).unwrap();
            assert_eq!(entry.to_string(), c.to_string());
        }
    }

    #[test]
    fn from_char_invalid() {
        assert!(KeypadEntry::from_char('A').is_none());
        assert!(KeypadEntry::from_char(' ').is_none());
    }

    #[test]
    fn equality() {
        assert_eq!(KeypadEntry::Five, KeypadEntry::Five);
        assert_ne!(KeypadEntry::Star, KeypadEntry::Pound);
    }

    #[test]
    fn clone_and_copy() {
        let a = KeypadEntry::Seven;
        let b = a;
        assert_eq!(a, b);
    }
}
