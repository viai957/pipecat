//! Metrics data types for Pipecat pipelines.
//!
//! This module defines plain data structs for various types of metrics
//! collected throughout the pipeline, including timing, token usage, and
//! processing statistics.

use serde::{Deserialize, Serialize};

/// Base metrics data — all metrics carry a processor name and optional model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Time To First Byte (TTFB) metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TTFBMetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// TTFB measurement in seconds.
    pub value: f64,
}

/// General processing time metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProcessingMetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Processing time measurement in seconds.
    pub value: f64,
}

/// Token usage statistics for LLM operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LLMTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl LLMTokenUsage {
    /// Create a new `LLMTokenUsage` with prompt and completion token counts.
    ///
    /// The `total_tokens` field is computed as `prompt + completion`.
    /// Optional cache and reasoning fields default to `None`.
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            reasoning_tokens: None,
        }
    }
}

impl Default for LLMTokenUsage {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// LLM token usage metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMUsageMetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub value: LLMTokenUsage,
}

/// Text-to-Speech usage metrics (character count).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TTSUsageMetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Number of characters processed by TTS.
    pub value: u32,
}

/// Turn detection metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnMetricsData {
    pub processor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether the turn is predicted to be complete.
    pub is_complete: bool,
    /// Confidence probability of the turn completion prediction.
    pub probability: f64,
    /// End-to-end processing time in milliseconds, measured from VAD
    /// speech-to-silence transition to turn completion.
    pub e2e_processing_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_data_default() {
        let m = MetricsData::default();
        assert_eq!(m.processor, "");
        assert_eq!(m.model, None);
    }

    #[test]
    fn llm_token_usage_new() {
        let usage = LLMTokenUsage::new(100, 50);
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.reasoning_tokens, None);
    }

    #[test]
    fn llm_token_usage_default() {
        let usage = LLMTokenUsage::default();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn serde_round_trip_metrics_data() {
        let m = MetricsData {
            processor: "llm-processor".to_string(),
            model: Some("gpt-4".to_string()),
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: MetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_round_trip_ttfb() {
        let m = TTFBMetricsData {
            processor: "tts".to_string(),
            model: None,
            value: 0.123,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: TTFBMetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_round_trip_processing() {
        let m = ProcessingMetricsData {
            processor: "stt".to_string(),
            model: Some("whisper".to_string()),
            value: 0.456,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: ProcessingMetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_round_trip_llm_usage() {
        let m = LLMUsageMetricsData {
            processor: "llm".to_string(),
            model: Some("claude-3".to_string()),
            value: LLMTokenUsage {
                prompt_tokens: 200,
                completion_tokens: 100,
                total_tokens: 300,
                cache_read_input_tokens: Some(50),
                cache_creation_input_tokens: None,
                reasoning_tokens: Some(20),
            },
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: LLMUsageMetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_round_trip_tts_usage() {
        let m = TTSUsageMetricsData {
            processor: "tts".to_string(),
            model: None,
            value: 1500,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: TTSUsageMetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_round_trip_turn_metrics() {
        let m = TurnMetricsData {
            processor: "turn-detector".to_string(),
            model: Some("smart-turn-v1".to_string()),
            is_complete: true,
            probability: 0.95,
            e2e_processing_time_ms: 42.5,
        };
        let json = serde_json::to_string(&m).unwrap();
        let deserialized: TurnMetricsData = serde_json::from_str(&json).unwrap();
        assert_eq!(m, deserialized);
    }

    #[test]
    fn serde_skip_none_fields() {
        let m = MetricsData {
            processor: "test".to_string(),
            model: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("model"), "None model field should be skipped");
    }
}
