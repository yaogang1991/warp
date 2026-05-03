//! Anthropic-compatible API client implementation with streaming support.

use super::{
    AnthropicMessageRequest, AnthropicStreamChunk, ChatMessage, LocalAIConfig, LocalAIError,
    ProviderClient, ProviderType,
};
use futures::{future, Stream, StreamExt};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use std::pin::Pin;

/// Anthropic-compatible client.
pub struct AnthropicClient {
    http_client: Client,
}

impl AnthropicClient {
    pub fn new() -> Self {
        Self {
            http_client: Client::new(),
        }
    }

    /// Send a streaming request to Anthropic-compatible endpoint.
    async fn send_streaming_request_internal(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, LocalAIError>> + Send>>,
        LocalAIError,
    > {
        let endpoint = config.chat_endpoint();
        let model = config.default_model();
        let api_key = config.api_key.clone();
        let headers = self.build_headers(&api_key);
        let timeout = config.timeout();

        let request = AnthropicMessageRequest {
            model: model.to_string(),
            messages,
            max_tokens: 4096,
            stream: true,
        };

        // Build the request with SSE support
        let mut request_builder = self.http_client.post(&endpoint);

        for (name, value) in headers.iter() {
            request_builder = request_builder.header(name, value);
        }

        request_builder = request_builder.timeout(timeout).json(&request);

        // Create SSE stream
        let stream = request_builder
            .eventsource()
            .map_err(|e| LocalAIError::StreamError(format!("Failed to create stream: {}", e)))?;

        Ok(Box::pin(stream.filter_map(move |event| {
            match event {
                Ok(Event::Message(message)) => {
                    // Anthropic sends SSE events with JSON data
                    match serde_json::from_str::<AnthropicStreamChunk>(&message.data) {
                        Ok(chunk) => {
                            match chunk.r#type.as_str() {
                                "content_block_delta" => {
                                    // Streaming content delta
                                    if let Some(delta) = chunk.delta {
                                        if let Some(text) = delta.text {
                                            return future::ready(Some(Ok(text)));
                                        }
                                    }
                                }
                                "message_stop" => {
                                    // Stream finished
                                }
                                _ => {
                                    // Other event types (ping, error, etc.)
                                    if chunk.r#type == "error" {
                                        return future::ready(Some(Err(
                                            LocalAIError::ApiError(
                                                message.data.clone(),
                                            ),
                                        )));
                                    }
                                }
                            }
                            future::ready(None)
                        }
                        Err(e) => {
                            // Some events might be metadata or control events
                            log::debug!("Failed to parse Anthropic stream chunk: {}", e);
                            future::ready(None)
                        }
                    }
                }
                Ok(Event::Open) => future::ready(None),
                Err(e) => future::ready(Some(Err(LocalAIError::StreamError(
                    format!("SSE error: {}", e),
                )))),
            }
        })))
    }
}

#[async_trait]
impl ProviderClient for AnthropicClient {
    async fn send_chat_request_streaming(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LocalAIError>> + Send>>, LocalAIError> {
        self.send_streaming_request_internal(config, messages).await
    }

    fn build_headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        if let Ok(value) = HeaderValue::from_str(api_key) {
            headers.insert("x-api-key", value);
        } else {
            log::warn!("Local AI: API key contains invalid header characters, request may fail");
        }
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );
        headers
    }
}

impl Default for AnthropicClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_endpoint_construction() {
        let config = LocalAIConfig {
            base_url: "https://api.example.com".to_string(),
            api_key: "test-key".to_string(),
            provider_type: ProviderType::Anthropic,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        assert_eq!(
            config.chat_endpoint(),
            "https://api.example.com/v1/messages"
        );
    }

    #[test]
    fn test_chat_endpoint_with_v1() {
        let config = LocalAIConfig {
            base_url: "https://api.example.com/v1".to_string(),
            api_key: "test-key".to_string(),
            provider_type: ProviderType::Anthropic,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        assert_eq!(
            config.chat_endpoint(),
            "https://api.example.com/v1/messages"
        );
    }

    #[test]
    fn test_anthropic_headers() {
        let client = AnthropicClient::new();
        let headers = client.build_headers("sk-ant-test-key");

        assert_eq!(
            headers.get(CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(
            headers.get("x-api-key").unwrap().to_str().unwrap(),
            "sk-ant-test-key"
        );
        assert_eq!(
            headers.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }
}
