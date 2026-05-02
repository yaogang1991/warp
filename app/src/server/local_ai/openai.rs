//! OpenAI-compatible API client implementation with streaming support.

use super::{
    ChatMessage, LocalAIConfig, LocalAIError, OpenAIChatRequest, OpenAIStreamChunk, ProviderClient, ProviderType,
};
use futures::{future, Stream, StreamExt};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::Deserialize;
use std::pin::Pin;

/// OpenAI-compatible client.
pub struct OpenAIClient {
    http_client: Client,
}

impl OpenAIClient {
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

        let request = OpenAIChatRequest {
            model: model.to_string(),
            messages,
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
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: Message,
        }

        #[derive(Deserialize)]
        struct Message {
            content: String,
        }

        let chat_response: ChatResponse = serde_json::from_slice(&body).map_err(|e| {
            LocalAIError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        chat_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or_else(|| LocalAIError::InvalidResponse("No choices in response".to_string()))
    }

    /// Send a streaming request to OpenAI-compatible endpoint.
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

        let request = OpenAIChatRequest {
            model: model.to_string(),
            messages,
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
                    if message.data == "[DONE]" {
                        // Stream finished
                    future::ready(None)
                    } else {
                        // Parse JSON chunk
                        match serde_json::from_str::<OpenAIStreamChunk>(&message.data) {
                            Ok(chunk) => {
                                // Extract content from delta
                                if let Some(content) = chunk
                                    .choices
                                    .first()
                                    .and_then(|c| c.delta.content.as_ref())
                                {
                                    future::ready(Some(Ok(content.clone())))
                                } else {
                                    future::ready(None)
                                }
                            }
                            Err(e) => {
                                // Log but continue - some events might be metadata
                                log::debug!("Failed to parse stream chunk: {}", e);
                                future::ready(None)
                            }
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
impl ProviderClient for OpenAIClient {
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .expect("Invalid auth header"),
        );
        headers
    }
}

impl Default for OpenAIClient {
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
            provider_type: ProviderType::OpenAI,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        assert_eq!(
            config.chat_endpoint(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_chat_endpoint_with_trailing_slash() {
        let config = LocalAIConfig {
            base_url: "https://api.example.com/".to_string(),
            api_key: "test-key".to_string(),
            provider_type: ProviderType::OpenAI,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        assert_eq!(
            config.chat_endpoint(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_chat_endpoint_with_v1() {
        let config = LocalAIConfig {
            base_url: "https://api.example.com/v1".to_string(),
            api_key: "test-key".to_string(),
            provider_type: ProviderType::OpenAI,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        assert_eq!(
            config.chat_endpoint(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_openai_headers() {
        let client = OpenAIClient::new();
        let headers = client.build_headers("sk-test-key");

        assert_eq!(
            headers.get(CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(
            headers.get(AUTHORIZATION).unwrap().to_str().unwrap(),
            "Bearer sk-test-key"
        );
    }
}
