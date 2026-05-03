//! Local AI client module for direct API calls to custom endpoints.
//!
//! This module provides support for calling AI services directly without going through
//! Warp's GraphQL API. It supports OpenAI-compatible and Anthropic-compatible endpoints
//! with streaming response support.

pub mod openai;
pub mod anthropic;

use crate::server::server_api::AIApiError;
use anyhow::anyhow;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;

/// Supported provider types for local AI endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    OpenAI,
    Anthropic,
}

impl ProviderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderType::OpenAI => "openai",
            ProviderType::Anthropic => "anthropic",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(ProviderType::OpenAI),
            "anthropic" => Some(ProviderType::Anthropic),
            _ => None,
        }
    }
}

/// Configuration for local AI client.
#[derive(Debug, Clone)]
pub struct LocalAIConfig {
    pub base_url: String,
    pub api_key: String,
    pub provider_type: ProviderType,
    pub model: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum number of retries for transient errors
    pub max_retries: usize,
}

impl LocalAIConfig {
    pub fn new(base_url: String, api_key: String, provider_type: ProviderType) -> Self {
        Self {
            base_url,
            api_key,
            provider_type,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Get the full endpoint URL for chat completions.
    pub fn chat_endpoint(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        match self.provider_type {
            ProviderType::OpenAI => {
                if base.ends_with("/chat/completions") {
                    base.to_string()
                } else if base.ends_with("/v1") {
                    format!("{}/chat/completions", base)
                } else {
                    format!("{}/v1/chat/completions", base)
                }
            }
            ProviderType::Anthropic => {
                if base.ends_with("/messages") {
                    base.to_string()
                } else if base.ends_with("/v1") {
                    format!("{}/messages", base)
                } else {
                    format!("{}/v1/messages", base)
                }
            }
        }
    }

    /// Get the default model name for this provider.
    pub fn default_model(&self) -> &str {
        self.model.as_deref().unwrap_or(match self.provider_type {
            ProviderType::OpenAI => "gpt-4o",
            ProviderType::Anthropic => "claude-3-5-sonnet-20241022",
        })
    }

    /// Get the request timeout duration.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }
}

/// Common request/response structures for both providers.

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

impl ChatMessage {
    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

/// OpenAI-compatible request structure.
#[derive(Debug, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
}

/// Anthropic-compatible request structure.
#[derive(Debug, Serialize)]
pub struct AnthropicMessageRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub stream: bool,
}

/// Streaming chunk types for different providers.

/// OpenAI streaming chunk
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<u64>,
    pub model: Option<String>,
    pub choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: usize,
    pub delta: OpenAIStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

/// Anthropic streaming chunk
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamChunk {
    pub r#type: String,
    pub index: Option<usize>,
    pub delta: Option<AnthropicStreamDelta>,
    pub message: Option<AnthropicStreamMessage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamDelta {
    pub r#type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamMessage {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicStreamContent>,
    pub model: String,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamContent {
    pub r#type: String,
    pub text: Option<String>,
}

/// Error type for local AI operations.
#[derive(Debug, thiserror::Error)]
pub enum LocalAIError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("API returned error: {0}")]
    ApiError(String),

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),
}

impl From<LocalAIError> for AIApiError {
    fn from(err: LocalAIError) -> Self {
        AIApiError::Other(anyhow!("{err}"))
    }
}

/// Trait for provider-specific implementations.
#[async_trait]
pub trait ProviderClient: Send + Sync {
    /// Send a non-streaming chat request (for backwards compatibility).
    async fn send_chat_request(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<String, LocalAIError> {
        let mut full_response = String::new();
        let mut stream = self.send_chat_request_streaming(config, messages).await?;
        while let Some(chunk) = stream.next().await {
            full_response.push_str(&chunk?);
        }
        Ok(full_response)
    }

    /// Send a streaming chat request, returning a stream of text chunks.
    async fn send_chat_request_streaming(
        &self,
        config: &LocalAIConfig,
        messages: Vec<ChatMessage>,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String, LocalAIError>> + Send>>, LocalAIError>;

    fn build_headers(&self, api_key: &str) -> HeaderMap;

    /// Check if an error is retryable (transient).
    fn is_retryable_error(&self, status: u16, body: &str) -> bool {
        matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
            || body.contains("timeout")
            || body.contains("rate limit")
            || body.contains("temporary")
    }
}

/// Factory function to get the appropriate provider client.
pub fn get_provider_client(provider_type: ProviderType) -> std::sync::Arc<dyn ProviderClient> {
    match provider_type {
        ProviderType::OpenAI => std::sync::Arc::new(openai::OpenAIClient::new()),
        ProviderType::Anthropic => std::sync::Arc::new(anthropic::AnthropicClient::new()),
    }
}

/// Send a chat request with retry logic for transient errors.
pub async fn send_chat_with_retry(
    client: &dyn ProviderClient,
    config: &LocalAIConfig,
    messages: Vec<ChatMessage>,
) -> Result<String, LocalAIError> {
    let max_attempts = config.max_retries + 1;

    for attempt in 0..max_attempts {
        match client.send_chat_request(config, messages.clone()).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                let is_retryable = match &e {
                    LocalAIError::RequestFailed(re) => {
                        if re.is_timeout() {
                            true
                        } else if let Some(status) = re.status() {
                            client.is_retryable_error(status.as_u16(), "request failed")
                        } else {
                            false
                        }
                    }
                    LocalAIError::ApiError(msg) => {
                        msg.contains("429") || msg.contains("rate limit") || msg.contains("timeout")
                    }
                    LocalAIError::StreamError(msg) => {
                        // Streaming connection errors (HTTP 429/500/502/503, timeouts)
                        msg.contains("429")
                            || msg.contains("500")
                            || msg.contains("502")
                            || msg.contains("503")
                            || msg.contains("timeout")
                            || msg.contains("rate limit")
                    }
                    LocalAIError::Timeout(_) => true,
                    _ => false,
                };

                if !is_retryable || attempt >= max_attempts - 1 {
                    return Err(e);
                }

                let backoff_ms = 1000 * (1 << attempt.min(5));
                log::debug!(
                    "Local AI request failed (attempt {}/{}), retrying after {}ms: {:?}",
                    attempt + 1,
                    max_attempts,
                    backoff_ms,
                    e
                );
                warpui::r#async::Timer::after(Duration::from_millis(backoff_ms)).await;
            }
        }
    }

    unreachable!()
}
