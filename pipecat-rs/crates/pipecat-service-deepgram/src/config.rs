//! Configuration for the Deepgram STT service.

/// Configuration for connecting to the Deepgram streaming STT API.
///
/// Sensible defaults match the Python Pipecat Deepgram integration.
///
/// # Example
///
/// ```
/// use pipecat_service_deepgram::DeepgramConfig;
///
/// let config = DeepgramConfig {
///     api_key: "your-api-key".into(),
///     model: "nova-3-general".into(),
///     ..Default::default()
/// };
/// assert!(config.websocket_url().contains("nova-3-general"));
/// ```
#[derive(Debug, Clone)]
pub struct DeepgramConfig {
    /// Deepgram API key used for authentication.
    pub api_key: String,

    /// Base WebSocket URL for the Deepgram streaming endpoint.
    pub base_url: String,

    /// Deepgram model name (e.g. "nova-3-general", "nova-2").
    pub model: String,

    /// BCP-47 language code for transcription.
    pub language: String,

    /// Audio encoding format (e.g. "linear16").
    pub encoding: String,

    /// Audio sample rate in Hz.
    pub sample_rate: u32,

    /// Number of audio channels.
    pub channels: u16,

    /// Whether to return interim (non-final) transcription results.
    pub interim_results: bool,

    /// Whether to enable automatic punctuation.
    pub punctuate: bool,

    /// Whether to enable smart formatting (numerals, dates, etc.).
    pub smart_format: bool,

    /// Whether to filter profanity from transcription results.
    pub profanity_filter: bool,

    /// Whether to enable VAD events (deprecated in newer Deepgram API versions).
    pub vad_events: bool,
}

impl Default for DeepgramConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "wss://api.deepgram.com/v1/listen".into(),
            model: "nova-3-general".into(),
            language: "en".into(),
            encoding: "linear16".into(),
            sample_rate: 16000,
            channels: 1,
            interim_results: true,
            punctuate: true,
            smart_format: false,
            profanity_filter: true,
            vad_events: false,
        }
    }
}

impl DeepgramConfig {
    /// Build the full WebSocket URL with query parameters.
    ///
    /// The API key is NOT included in the URL; it is sent as an
    /// `Authorization` header during the WebSocket handshake.
    pub fn websocket_url(&self) -> String {
        format!(
            "{}?model={}&language={}&encoding={}&sample_rate={}&channels={}&interim_results={}&punctuate={}&smart_format={}&profanity_filter={}",
            self.base_url,
            self.model,
            self.language,
            self.encoding,
            self.sample_rate,
            self.channels,
            self.interim_results,
            self.punctuate,
            self.smart_format,
            self.profanity_filter,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = DeepgramConfig::default();
        assert_eq!(cfg.base_url, "wss://api.deepgram.com/v1/listen");
        assert_eq!(cfg.model, "nova-3-general");
        assert_eq!(cfg.language, "en");
        assert_eq!(cfg.encoding, "linear16");
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.channels, 1);
        assert!(cfg.interim_results);
        assert!(cfg.punctuate);
        assert!(!cfg.smart_format);
        assert!(cfg.profanity_filter);
        assert!(!cfg.vad_events);
        assert!(cfg.api_key.is_empty());
    }

    #[test]
    fn websocket_url_contains_all_params() {
        let cfg = DeepgramConfig {
            api_key: "secret".into(),
            ..Default::default()
        };
        let url = cfg.websocket_url();

        assert!(url.starts_with("wss://api.deepgram.com/v1/listen?"));
        assert!(url.contains("model=nova-3-general"));
        assert!(url.contains("language=en"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        assert!(url.contains("channels=1"));
        assert!(url.contains("interim_results=true"));
        assert!(url.contains("punctuate=true"));
        assert!(url.contains("smart_format=false"));
        assert!(url.contains("profanity_filter=true"));
        // API key should NOT be in the URL
        assert!(!url.contains("secret"));
    }

    #[test]
    fn websocket_url_custom_params() {
        let cfg = DeepgramConfig {
            api_key: "key".into(),
            model: "nova-2".into(),
            language: "es".into(),
            sample_rate: 8000,
            channels: 2,
            interim_results: false,
            smart_format: true,
            ..Default::default()
        };
        let url = cfg.websocket_url();

        assert!(url.contains("model=nova-2"));
        assert!(url.contains("language=es"));
        assert!(url.contains("sample_rate=8000"));
        assert!(url.contains("channels=2"));
        assert!(url.contains("interim_results=false"));
        assert!(url.contains("smart_format=true"));
    }

    #[test]
    fn config_is_clone() {
        let cfg = DeepgramConfig {
            api_key: "key".into(),
            ..Default::default()
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.api_key, "key");
        assert_eq!(cloned.model, cfg.model);
    }

    #[test]
    fn config_is_debug() {
        let cfg = DeepgramConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("DeepgramConfig"));
        assert!(debug.contains("nova-3-general"));
    }
}
