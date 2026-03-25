use serde::{Deserialize, Serialize};

/// Time-to-first-byte metrics for a service call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtfbMetricsData {
    /// Name of the processor that generated this metric.
    pub processor: String,
    /// Time to first byte in seconds.
    pub value: f64,
}

/// Processing-time metrics for a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetricsData {
    /// Name of the processor that generated this metric.
    pub processor: String,
    /// Processing duration in seconds.
    pub value: f64,
}

/// Token usage reported by an LLM service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Cache-related token counts (provider-specific).
    pub cache_read_input_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
}

/// LLM usage metrics tying token usage to a specific processor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmUsageMetricsData {
    pub processor: String,
    pub model: String,
    pub usage: LlmTokenUsage,
}

/// Envelope for all metric kinds that can be attached to a `Metrics` frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MetricsData {
    #[serde(rename = "ttfb")]
    Ttfb(TtfbMetricsData),

    #[serde(rename = "processing")]
    Processing(ProcessingMetricsData),

    #[serde(rename = "llm_usage")]
    LlmUsage(LlmUsageMetricsData),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttfb_serde_roundtrip() {
        let data = MetricsData::Ttfb(TtfbMetricsData {
            processor: "stt".into(),
            value: 0.123,
        });
        let json = serde_json::to_string(&data).unwrap();
        let parsed: MetricsData = serde_json::from_str(&json).unwrap();
        match parsed {
            MetricsData::Ttfb(t) => {
                assert_eq!(t.processor, "stt");
                assert!((t.value - 0.123).abs() < f64::EPSILON);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn llm_usage_serde_roundtrip() {
        let data = MetricsData::LlmUsage(LlmUsageMetricsData {
            processor: "openai".into(),
            model: "gpt-4".into(),
            usage: LlmTokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: Some(20),
                cache_creation_input_tokens: None,
            },
        });
        let json = serde_json::to_string(&data).unwrap();
        let parsed: MetricsData = serde_json::from_str(&json).unwrap();
        match parsed {
            MetricsData::LlmUsage(u) => {
                assert_eq!(u.usage.total_tokens, 150);
                assert_eq!(u.usage.cache_read_input_tokens, Some(20));
                assert_eq!(u.usage.cache_creation_input_tokens, None);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn processing_metrics_serde() {
        let data = MetricsData::Processing(ProcessingMetricsData {
            processor: "tts".into(),
            value: 0.456,
        });
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"type\":\"processing\""));
        let parsed: MetricsData = serde_json::from_str(&json).unwrap();
        match parsed {
            MetricsData::Processing(p) => assert_eq!(p.processor, "tts"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn default_token_usage() {
        let usage = LlmTokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert!(usage.cache_read_input_tokens.is_none());
    }
}
