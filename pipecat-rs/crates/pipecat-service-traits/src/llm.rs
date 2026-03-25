//! The [`LLMService`] trait for Large Language Model integrations.
//!
//! LLM services process conversation context and produce text, tool-call, and
//! lifecycle frames. They also support registering function-call handlers that
//! are invoked when the model requests a tool call.

use async_trait::async_trait;

use pipecat_core::Frame;

use crate::ai_service::AIService;

/// A boxed, sendable function-call handler.
///
/// Receives the tool-call arguments as a JSON value and returns a JSON result.
pub type FunctionCallHandler =
    Box<dyn FnMut(serde_json::Value) -> serde_json::Value + Send>;

/// Trait for Large Language Model services.
///
/// LLM services process a conversation context (represented as a JSON value
/// containing messages, system prompt, tool definitions, etc.) and produce
/// a stream of frames:
///
/// # Frame sequence
///
/// A typical `process_llm` call returns frames in this order:
///
/// 1. [`Frame::LlmResponseStart`] -- signals the start of generation
/// 2. One or more [`Frame::TextLlm`] -- streamed text tokens
/// 3. Zero or more [`Frame::LlmToolCall`] -- tool/function call requests
/// 4. [`Frame::LlmResponseEnd`] -- signals the end of generation
///
/// # Function registration
///
/// Use [`register_function`](LLMService::register_function) to register
/// handlers for tool calls. When the model produces an
/// [`Frame::LlmToolCall`], the pipeline can invoke the registered handler
/// and feed the result back via [`Frame::LlmToolResult`].
#[async_trait]
pub trait LLMService: AIService {
    /// Process an LLM context and return resulting frames.
    ///
    /// The `context` parameter is a JSON value typically containing:
    /// - `messages`: array of conversation messages
    /// - `tools`: optional tool/function definitions
    /// - `tool_choice`: optional tool selection preference
    ///
    /// Returns a `Vec` of frames produced by the model.
    async fn process_llm(&mut self, context: &serde_json::Value) -> Vec<Frame>;

    /// Register a function-call handler for the given tool name.
    ///
    /// When the model produces a tool call matching `name`, the pipeline can
    /// invoke the handler with the call's arguments and return the result.
    fn register_function(&mut self, name: &str, handler: FunctionCallHandler);

    /// Unregister a previously registered function-call handler.
    ///
    /// Returns `true` if a handler was found and removed.
    fn unregister_function(&mut self, name: &str) -> bool {
        let _ = name;
        false
    }

    /// Check whether a function handler is registered for the given name.
    fn has_function(&self, name: &str) -> bool {
        let _ = name;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use pipecat_core::{Frame, FrameDirection, FrameHeader, Result, TextData};
    use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct FakeLlm {
        /// Wrapped in a Mutex so that `FakeLlm` is `Sync` (required by
        /// `FrameProcessor: Send + Sync`). `FnMut` closures are `Send`
        /// but not `Sync`, so bare `HashMap<_, FunctionCallHandler>` would
        /// break the bound.
        functions: Mutex<HashMap<String, FunctionCallHandler>>,
    }

    impl FakeLlm {
        fn new() -> Self {
            Self {
                functions: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait]
    impl FrameProcessor for FakeLlm {
        async fn process_frame(
            &mut self,
            frame: Frame,
            direction: FrameDirection,
            ctx: &ProcessorContext,
        ) -> Result<()> {
            ctx.push_frame(frame, direction).await
        }
        fn name(&self) -> &str {
            "FakeLlm"
        }
    }

    #[async_trait]
    impl AIService for FakeLlm {
        fn model_name(&self) -> Option<&str> {
            Some("fake-llm")
        }

        fn can_generate_metrics(&self) -> bool {
            true
        }
    }

    #[async_trait]
    impl LLMService for FakeLlm {
        async fn process_llm(&mut self, context: &serde_json::Value) -> Vec<Frame> {
            let messages = context
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            if messages == 0 {
                return vec![];
            }

            vec![
                Frame::LlmResponseStart(FrameHeader::new()),
                Frame::TextLlm {
                    header: FrameHeader::new(),
                    data: TextData {
                        text: "I am a helpful assistant.".to_string(),
                    },
                },
                Frame::LlmResponseEnd(FrameHeader::new()),
            ]
        }

        fn register_function(&mut self, name: &str, handler: FunctionCallHandler) {
            self.functions
                .lock()
                .unwrap()
                .insert(name.to_string(), handler);
        }

        fn unregister_function(&mut self, name: &str) -> bool {
            self.functions.lock().unwrap().remove(name).is_some()
        }

        fn has_function(&self, name: &str) -> bool {
            self.functions.lock().unwrap().contains_key(name)
        }
    }

    #[tokio::test]
    async fn llm_produces_response_sequence() {
        let mut llm = FakeLlm::new();
        let ctx = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        });
        let frames = llm.process_llm(&ctx).await;
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].name(), "LlmResponseStart");
        assert_eq!(frames[1].name(), "TextLlm");
        assert_eq!(frames[2].name(), "LlmResponseEnd");
    }

    #[tokio::test]
    async fn llm_empty_context_produces_nothing() {
        let mut llm = FakeLlm::new();
        let ctx = serde_json::json!({ "messages": [] });
        let frames = llm.process_llm(&ctx).await;
        assert!(frames.is_empty());
    }

    #[test]
    fn llm_register_function() {
        let mut llm = FakeLlm::new();
        assert!(!llm.has_function("get_weather"));

        llm.register_function(
            "get_weather",
            Box::new(|args| {
                serde_json::json!({
                    "temp": 72,
                    "location": args.get("location").cloned().unwrap_or(serde_json::Value::Null)
                })
            }),
        );
        assert!(llm.has_function("get_weather"));
    }

    #[test]
    fn llm_unregister_function() {
        let mut llm = FakeLlm::new();
        llm.register_function(
            "tool1",
            Box::new(|_| serde_json::json!(null)),
        );
        assert!(llm.has_function("tool1"));
        assert!(llm.unregister_function("tool1"));
        assert!(!llm.has_function("tool1"));
        assert!(!llm.unregister_function("tool1")); // already removed
    }

    #[test]
    fn llm_invoke_registered_handler() {
        let mut llm = FakeLlm::new();
        llm.register_function(
            "add",
            Box::new(|args| {
                let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                serde_json::json!(a + b)
            }),
        );

        let mut fns = llm.functions.lock().unwrap();
        let handler = fns.get_mut("add").unwrap();
        let result = handler(serde_json::json!({"a": 3, "b": 4}));
        assert_eq!(result, serde_json::json!(7));
    }

    #[test]
    fn llm_model_name_and_metrics() {
        let llm = FakeLlm::new();
        assert_eq!(llm.model_name(), Some("fake-llm"));
        assert!(llm.can_generate_metrics());
    }
}
