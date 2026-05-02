//! Anthropic-compatible API client implementation with streaming support.

use super::{
    AnthropicMessageRequest, AnthropicStreamChunk, ChatMessage, LocalAIConfig, LocalAIError,
    ProviderClient,
};
use futures::{future, StreamExt};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::Deserialize;
use std::pin::Pin;
use std::time::Duration;

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

    async fn send_request_internal(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LocalAIError> {
        let endpoint = config.chat_endpoint();
        let model = config.default_model();

        // Anthropic expects a specific message format with role "user"
        // For simplicity, we'll convert all messages to user messages
        let anthropic_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                serde_json::json!({
                    "role": "user",
                    "content": msg.content
                })
            })
            .collect();

        let request = AnthropicMessageRequest {
            model: model.to_string(),
            messages: anthropic_messages,
            max_tokens: 4096,
            stream: false,
        };

        let response = self
            .http_client
            .post(&endpoint)
            .headers(self.build_headers(&config.api_key))
            .timeout(config.timeout())
            .json(&request)
            .send()
            .await
            .map_err(LocalAIError::RequestFailed)?;

        let status = response.status();
        let body = response.bytes().await.map_err(LocalAIError::RequestFailed)?;

        if !status.is_success() {
            let error_msg = String::from_utf8_lossy(&body);
            return Err(LocalAIError::ApiError(format!(
                "Status {}: {}",
                status, error_msg
            )));
        }

        #[derive(Deserialize)]
        struct MessageResponse {
            content: Vec<ContentBlock>,
        }

        #[derive(Deserialize)]
        struct ContentBlock {
            r#type: String,
            text: Option<String>,
        }

        let message_response: MessageResponse =
            serde_json::from_slice(&body).map_err(|e| {
                LocalAIError::InvalidResponse(format!("Failed to parse response: {}", e))
            })?;

        // Extract text from content blocks
        let text_parts: Vec<String> = message_response
            .content
            .iter()
            .filter_map(|block| {
                if block.r#type == "text" {
                    block.text.as_ref()
                } else {
                    None
                }
            })
            .cloned()
            .collect();

        if text_parts.is_empty() {
            return Err(LocalAIError::InvalidResponse(
                "No text content in response".to_string(),
            ));
        }

        Ok(text_parts.join(""))
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

        // Anthropic expects a specific message format
        let anthropic_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                serde_json::json!({
                    "role": "user",
                    "content": msg.content
                })
            })
            .collect();

        let request = AnthropicMessageRequest {
            model: model.to_string(),
            messages: anthropic_messages,
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
            .await
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
    async fn send_chat_request(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LocalAIError> {
        self.send_request_internal(config, messages).await
    }

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
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(api_key).expect("Invalid API key"),
        );
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
            provider_type: super::ProviderType::Anthropic,
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
            provider_type: super::ProviderType::Anthropic,
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
