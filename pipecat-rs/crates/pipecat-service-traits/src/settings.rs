//! The [`Setting<T>`] system for delta-style service configuration updates.
//!
//! In Python Pipecat, `NOT_GIVEN` is a sentinel value that distinguishes
//! "the caller did not provide this field" from "the caller explicitly set it
//! to `None`". The Rust equivalent is the [`Setting`] enum, which carries
//! either [`Setting::NotGiven`] (skip during apply) or [`Setting::Value(T)`]
//! (apply the new value).
//!
//! This module also provides [`ServiceSettings`] (base settings all services
//! share), [`ServiceSettingsDelta`] (a delta that can be applied at runtime),
//! and domain-specific settings structs for LLM, TTS, and STT services.

use std::collections::HashMap;

/// Represents a field that may or may not be provided in a delta update.
///
/// Replaces Python's `NOT_GIVEN` sentinel.
///
/// # Examples
///
/// ```
/// use pipecat_service_traits::Setting;
///
/// let given: Setting<f32> = Setting::Value(0.7);
/// assert!(given.is_given());
/// assert_eq!(given.value(), Some(&0.7));
///
/// let not_given: Setting<f32> = Setting::NotGiven;
/// assert!(!not_given.is_given());
/// assert_eq!(not_given.value(), None);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Setting<T> {
    /// Field was not included in the delta -- skip during apply.
    #[default]
    NotGiven,
    /// Field was explicitly set to this value.
    Value(T),
}

impl<T> Setting<T> {
    /// Returns `true` if this setting carries an explicit value.
    pub fn is_given(&self) -> bool {
        matches!(self, Setting::Value(_))
    }

    /// Returns a reference to the inner value, or `None` if [`NotGiven`](Setting::NotGiven).
    pub fn value(&self) -> Option<&T> {
        match self {
            Setting::Value(v) => Some(v),
            Setting::NotGiven => None,
        }
    }

    /// Consumes `self` and returns the inner value, or `None` if [`NotGiven`](Setting::NotGiven).
    pub fn into_value(self) -> Option<T> {
        match self {
            Setting::Value(v) => Some(v),
            Setting::NotGiven => None,
        }
    }

    /// Maps a `Setting<T>` to `Setting<U>` by applying a function to the
    /// contained value (if any).
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Setting<U> {
        match self {
            Setting::Value(v) => Setting::Value(f(v)),
            Setting::NotGiven => Setting::NotGiven,
        }
    }

    /// Returns the contained value or a default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Setting::Value(v) => v,
            Setting::NotGiven => default,
        }
    }
}

impl<T> From<T> for Setting<T> {
    fn from(value: T) -> Self {
        Setting::Value(value)
    }
}

// ---------------------------------------------------------------------------
// ServiceSettings -- base settings shared by all services
// ---------------------------------------------------------------------------

/// Base service settings shared by all AI services.
///
/// Every service has at least an optional `model` name. The `extra` map
/// provides an escape hatch for provider-specific parameters that don't have
/// first-class fields.
#[derive(Debug, Clone)]
pub struct ServiceSettings {
    /// The model identifier (e.g. `"gpt-4o"`, `"nova-2"`, `"eleven_turbo_v2"`).
    pub model: Option<String>,
    /// Arbitrary provider-specific key-value pairs.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ServiceSettings {
    /// Create empty settings with no model and no extras.
    pub fn new() -> Self {
        Self {
            model: None,
            extra: HashMap::new(),
        }
    }

    /// Create settings with a model name and no extras.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            extra: HashMap::new(),
        }
    }
}

impl Default for ServiceSettings {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ServiceSettingsDelta
// ---------------------------------------------------------------------------

/// A delta for updating [`ServiceSettings`] at runtime.
///
/// Fields set to [`Setting::NotGiven`] are left unchanged; fields set to
/// [`Setting::Value`] are applied. The [`apply_to`](ServiceSettingsDelta::apply_to)
/// method returns a map of field names to their *old* values, which is useful
/// for undo or change-notification logic.
#[derive(Debug, Clone, Default)]
pub struct ServiceSettingsDelta {
    /// New model name, if any.
    pub model: Setting<String>,
    /// Extra key-value overrides to merge into the store.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ServiceSettingsDelta {
    /// Apply this delta to `store`, returning a map of changed field names
    /// to their **previous** values (serialised as JSON).
    pub fn apply_to(&self, store: &mut ServiceSettings) -> HashMap<String, serde_json::Value> {
        let mut changed = HashMap::new();

        if let Setting::Value(ref new_model) = self.model {
            if store.model.as_deref() != Some(new_model.as_str()) {
                let old = store.model.take();
                changed.insert(
                    "model".to_string(),
                    serde_json::to_value(&old).unwrap_or_default(),
                );
                store.model = Some(new_model.clone());
            }
        }

        for (key, new_val) in &self.extra {
            let old = store.extra.get(key).cloned();
            if old.as_ref() != Some(new_val) {
                changed.insert(key.clone(), old.unwrap_or(serde_json::Value::Null));
                store.extra.insert(key.clone(), new_val.clone());
            }
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// Domain-specific settings
// ---------------------------------------------------------------------------

/// LLM-specific settings.
///
/// Wraps the base [`ServiceSettings`] and adds fields common to most LLM
/// providers (temperature, max tokens, etc.).
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// Common service settings (model, extras).
    pub base: ServiceSettings,
    /// Sampling temperature (0.0 = deterministic, higher = more creative).
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Frequency penalty (reduce repetition of frequent tokens).
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (reduce repetition of any used token).
    pub presence_penalty: Option<f32>,
    /// Random seed for reproducible completions.
    pub seed: Option<u64>,
}

impl LlmSettings {
    /// Create default LLM settings with no model and all parameters unset.
    pub fn new() -> Self {
        Self {
            base: ServiceSettings::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }

    /// Create LLM settings with a model name and default parameters.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            base: ServiceSettings::with_model(model),
            ..Self::new()
        }
    }
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// TTS-specific settings.
///
/// Wraps the base [`ServiceSettings`] and adds voice and language fields.
#[derive(Debug, Clone)]
pub struct TtsSettings {
    /// Common service settings (model, extras).
    pub base: ServiceSettings,
    /// Voice identifier (e.g. `"alloy"`, `"Joanna"`).
    pub voice: Option<String>,
    /// BCP-47 language code for synthesis.
    pub language: Option<String>,
}

impl TtsSettings {
    /// Create default TTS settings with no model, voice, or language.
    pub fn new() -> Self {
        Self {
            base: ServiceSettings::new(),
            voice: None,
            language: None,
        }
    }

    /// Create TTS settings with a model name and default parameters.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            base: ServiceSettings::with_model(model),
            ..Self::new()
        }
    }
}

impl Default for TtsSettings {
    fn default() -> Self {
        Self::new()
    }
}

/// STT-specific settings.
///
/// Wraps the base [`ServiceSettings`] and adds a language field.
#[derive(Debug, Clone)]
pub struct SttSettings {
    /// Common service settings (model, extras).
    pub base: ServiceSettings,
    /// BCP-47 language code for recognition.
    pub language: Option<String>,
}

impl SttSettings {
    /// Create default STT settings with no model or language.
    pub fn new() -> Self {
        Self {
            base: ServiceSettings::new(),
            language: None,
        }
    }

    /// Create STT settings with a model name and default language.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            base: ServiceSettings::with_model(model),
            ..Self::new()
        }
    }
}

impl Default for SttSettings {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Setting<T> ---------------------------------------------------------

    #[test]
    fn setting_not_given_is_default() {
        let s: Setting<String> = Setting::default();
        assert!(!s.is_given());
        assert_eq!(s.value(), None);
    }

    #[test]
    fn setting_value_is_given() {
        let s = Setting::Value(42);
        assert!(s.is_given());
        assert_eq!(s.value(), Some(&42));
    }

    #[test]
    fn setting_into_value() {
        let s = Setting::Value("hello".to_string());
        assert_eq!(s.into_value(), Some("hello".to_string()));

        let s: Setting<String> = Setting::NotGiven;
        assert_eq!(s.into_value(), None);
    }

    #[test]
    fn setting_map() {
        let s = Setting::Value(3);
        let doubled = s.map(|v| v * 2);
        assert_eq!(doubled.value(), Some(&6));

        let s: Setting<i32> = Setting::NotGiven;
        let mapped = s.map(|v| v * 2);
        assert!(!mapped.is_given());
    }

    #[test]
    fn setting_unwrap_or() {
        assert_eq!(Setting::Value(10).unwrap_or(99), 10);
        assert_eq!(Setting::<i32>::NotGiven.unwrap_or(99), 99);
    }

    #[test]
    fn setting_from_value() {
        let s: Setting<u32> = 42u32.into();
        assert_eq!(s, Setting::Value(42));
    }

    #[test]
    fn setting_clone() {
        let s = Setting::Value("data".to_string());
        let c = s.clone();
        assert_eq!(s, c);
    }

    #[test]
    fn setting_equality() {
        assert_eq!(Setting::Value(1), Setting::Value(1));
        assert_ne!(Setting::Value(1), Setting::Value(2));
        assert_ne!(Setting::Value(1), Setting::<i32>::NotGiven);
        assert_eq!(Setting::<i32>::NotGiven, Setting::<i32>::NotGiven);
    }

    // -- ServiceSettings ----------------------------------------------------

    #[test]
    fn service_settings_new_empty() {
        let s = ServiceSettings::new();
        assert!(s.model.is_none());
        assert!(s.extra.is_empty());
    }

    #[test]
    fn service_settings_with_model() {
        let s = ServiceSettings::with_model("gpt-4o");
        assert_eq!(s.model.as_deref(), Some("gpt-4o"));
        assert!(s.extra.is_empty());
    }

    #[test]
    fn service_settings_default() {
        let s = ServiceSettings::default();
        assert!(s.model.is_none());
    }

    // -- ServiceSettingsDelta -----------------------------------------------

    #[test]
    fn delta_apply_model_change() {
        let mut store = ServiceSettings::with_model("old-model");
        let delta = ServiceSettingsDelta {
            model: Setting::Value("new-model".to_string()),
            extra: HashMap::new(),
        };
        let changed = delta.apply_to(&mut store);
        assert_eq!(store.model.as_deref(), Some("new-model"));
        assert!(changed.contains_key("model"));
        // Old value should be "old-model" serialized
        assert_eq!(
            changed["model"],
            serde_json::Value::String("old-model".to_string())
        );
    }

    #[test]
    fn delta_apply_model_no_change_when_same() {
        let mut store = ServiceSettings::with_model("same");
        let delta = ServiceSettingsDelta {
            model: Setting::Value("same".to_string()),
            extra: HashMap::new(),
        };
        let changed = delta.apply_to(&mut store);
        assert!(changed.is_empty());
        assert_eq!(store.model.as_deref(), Some("same"));
    }

    #[test]
    fn delta_apply_not_given_skips_model() {
        let mut store = ServiceSettings::with_model("keep-me");
        let delta = ServiceSettingsDelta::default();
        let changed = delta.apply_to(&mut store);
        assert!(changed.is_empty());
        assert_eq!(store.model.as_deref(), Some("keep-me"));
    }

    #[test]
    fn delta_apply_extra_fields() {
        let mut store = ServiceSettings::new();
        store
            .extra
            .insert("key1".into(), serde_json::json!("old_val"));

        let mut extra = HashMap::new();
        extra.insert("key1".into(), serde_json::json!("new_val"));
        extra.insert("key2".into(), serde_json::json!(42));

        let delta = ServiceSettingsDelta {
            model: Setting::NotGiven,
            extra,
        };
        let changed = delta.apply_to(&mut store);

        assert_eq!(store.extra["key1"], serde_json::json!("new_val"));
        assert_eq!(store.extra["key2"], serde_json::json!(42));
        assert_eq!(changed["key1"], serde_json::json!("old_val"));
        assert_eq!(changed["key2"], serde_json::Value::Null); // was absent
    }

    #[test]
    fn delta_apply_extra_no_change_when_same() {
        let mut store = ServiceSettings::new();
        store
            .extra
            .insert("k".into(), serde_json::json!("same"));

        let mut extra = HashMap::new();
        extra.insert("k".into(), serde_json::json!("same"));
        let delta = ServiceSettingsDelta {
            model: Setting::NotGiven,
            extra,
        };
        let changed = delta.apply_to(&mut store);
        assert!(changed.is_empty());
    }

    #[test]
    fn delta_apply_model_from_none() {
        let mut store = ServiceSettings::new();
        let delta = ServiceSettingsDelta {
            model: Setting::Value("first-model".to_string()),
            extra: HashMap::new(),
        };
        let changed = delta.apply_to(&mut store);
        assert_eq!(store.model.as_deref(), Some("first-model"));
        assert!(changed.contains_key("model"));
        assert_eq!(changed["model"], serde_json::Value::Null); // was None
    }

    // -- LlmSettings --------------------------------------------------------

    #[test]
    fn llm_settings_default() {
        let s = LlmSettings::default();
        assert!(s.base.model.is_none());
        assert!(s.temperature.is_none());
        assert!(s.max_tokens.is_none());
        assert!(s.top_p.is_none());
        assert!(s.top_k.is_none());
        assert!(s.frequency_penalty.is_none());
        assert!(s.presence_penalty.is_none());
        assert!(s.seed.is_none());
    }

    #[test]
    fn llm_settings_with_model() {
        let s = LlmSettings::with_model("claude-3-opus");
        assert_eq!(s.base.model.as_deref(), Some("claude-3-opus"));
    }

    #[test]
    fn llm_settings_custom_params() {
        let mut s = LlmSettings::with_model("gpt-4o");
        s.temperature = Some(0.7);
        s.max_tokens = Some(4096);
        s.top_p = Some(0.9);
        assert_eq!(s.temperature, Some(0.7));
        assert_eq!(s.max_tokens, Some(4096));
        assert_eq!(s.top_p, Some(0.9));
    }

    // -- TtsSettings --------------------------------------------------------

    #[test]
    fn tts_settings_default() {
        let s = TtsSettings::default();
        assert!(s.base.model.is_none());
        assert!(s.voice.is_none());
        assert!(s.language.is_none());
    }

    #[test]
    fn tts_settings_with_model() {
        let s = TtsSettings::with_model("eleven_turbo_v2");
        assert_eq!(s.base.model.as_deref(), Some("eleven_turbo_v2"));
    }

    #[test]
    fn tts_settings_custom_voice() {
        let mut s = TtsSettings::with_model("tts-1");
        s.voice = Some("alloy".to_string());
        s.language = Some("en-US".to_string());
        assert_eq!(s.voice.as_deref(), Some("alloy"));
        assert_eq!(s.language.as_deref(), Some("en-US"));
    }

    // -- SttSettings --------------------------------------------------------

    #[test]
    fn stt_settings_default() {
        let s = SttSettings::default();
        assert!(s.base.model.is_none());
        assert!(s.language.is_none());
    }

    #[test]
    fn stt_settings_with_model() {
        let s = SttSettings::with_model("nova-2");
        assert_eq!(s.base.model.as_deref(), Some("nova-2"));
    }

    #[test]
    fn stt_settings_custom_language() {
        let mut s = SttSettings::with_model("whisper-1");
        s.language = Some("fr-FR".to_string());
        assert_eq!(s.language.as_deref(), Some("fr-FR"));
    }
}
