//! Local AI client that implements AIClient trait for direct API calls.
//!
//! This client bypasses Warp's GraphQL API and calls AI services directly
//! using OpenAI-compatible or Anthropic-compatible protocols with streaming support.

use crate::ai::ai_assistant::execution_context::WarpAiExecutionContext;
use crate::ai::ai_assistant::requests::{GenerateDialogueResult, RequestLimitInfo};
use crate::ai::ai_assistant::utils::TranscriptPart;
use crate::ai::llms::{ModelsByFeature, LLMInfo, LLMSpec, LLMUsageMetadata, AvailableLLMs};
use crate::ai::request_usage_model::RequestUsageInfo;
use crate::server::local_ai::{LocalAIConfig, ProviderType, ChatMessage, get_provider_client, LocalAIError};
use crate::server::AIApiError;
use crate::{ai::AIGeneratedCommand, drive::workflows::ai_assist::GeneratedCommandMetadata};
use futures::{Future, StreamExt};
use std::time::{Duration, Instant};
use anyhow::anyhow;
use async_trait::async_trait;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use warpui::AppContext;
use warpui::r#async::Timer;

/// Local AI client that implements AIClient trait.
pub struct LocalAIClient {
    config: LocalAIConfig,
    provider_client: Arc<dyn crate::server::local_ai::ProviderClient>,
}

impl LocalAIClient {
    /// Create a new LocalAIClient from configuration.
    pub fn new(config: LocalAIConfig) -> Self {
        let provider_client = get_provider_client(config.provider_type);
        Self {
            config,
            provider_client,
        }
    }

    /// Create LocalAIClient from AppContext (reads from secure storage).
    pub fn from_context(ctx: &AppContext) -> Option<Self> {
        use ai::api_keys::ApiKeyManager;

        let api_keys = ApiKeyManager::as_ref(ctx).keys();

        let base_url = api_keys.base_url.as_ref()?;
        let api_key = api_keys.openai.as_ref()
            .or(api_keys.anthropic.as_ref())
            .or(api_keys.open_router.as_ref())
            .or(api_keys.zai.as_ref())?;

        // All supported providers use the OpenAI-compatible protocol
        let provider_type = ProviderType::OpenAI;

        // Respect configured provider type if available
        let provider_type = api_keys.local_provider_type
            .map(|pt| match pt {
                ai::api_keys::LocalAIProviderType::OpenAI => ProviderType::OpenAI,
                ai::api_keys::LocalAIProviderType::Anthropic => ProviderType::Anthropic,
                ai::api_keys::LocalAIProviderType::Zai => ProviderType::OpenAI,
            })
            .unwrap_or(provider_type);

        let config = LocalAIConfig {
            base_url: base_url.clone(),
            api_key: api_key.clone(),
            provider_type,
            model: None,
            timeout_secs: 120,
            max_retries: 3,
        };

        Some(Self::new(config))
    }

    /// Check if local AI is configured in the given context.
    pub fn is_configured(ctx: &AppContext) -> bool {
        use ai::api_keys::ApiKeyManager;
        ApiKeyManager::as_ref(ctx).has_local_ai_config()
    }

    /// Convert transcript parts to chat messages.
    fn transcript_to_messages(
        transcript: &[TranscriptPart],
        prompt: &str,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // Add conversation history from transcript
        for part in transcript {
            // Add user message
            if !part.prompt.is_empty() {
                messages.push(ChatMessage::user(part.prompt.clone()));
            }
            // Add assistant response (as context for next message)
            if !part.answer.is_empty() {
                // Note: Most APIs expect user/assistant alternation
                // For simplicity, we'll include the assistant's answer in the next user prompt
                // or we could use the messages array format if the API supports it
            }
        }

        // Add the current prompt
        if !prompt.is_empty() {
            messages.push(ChatMessage::user(prompt.to_string()));
        }

        messages
    }

    /// Send request with retry logic for transient errors.
    async fn send_with_retry<F, T>(
        &self,
        mut request_fn: F,
    ) -> Result<T, LocalAIError>
    where
        F: FnMut() -> Pin<Box<dyn futures::Future<Output = Result<T, LocalAIError>> + Send>>,
    {
        let mut last_error = None;
        let max_attempts = self.config.max_retries + 1;

        for attempt in 0..max_attempts {
            match request_fn().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e.clone());

                    // Check if error is retryable
                    let is_retryable = match &e {
                        LocalAIError::RequestFailed(re) => {
                            if re.is_timeout() {
                                true
                            } else if let Some(status) = re.status() {
                                self.provider_client.is_retryable_error(
                                    status.as_u16(),
                                    &"request failed"
                                )
                            } else {
                                false
                            }
                        }
                        LocalAIError::ApiError(msg) => {
                            // Try to extract status code from error message
                            if msg.contains("429") || msg.contains("rate limit") {
                                true
                            } else if msg.contains("timeout") {
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    };

                    if !is_retryable || attempt >= max_attempts - 1 {
                        break;
                    }

                    // Exponential backoff: 1s, 2s, 4s...
                    let backoff_ms = 1000 * (1 << attempt.min(5));
                    log::debug!(
                        "Local AI request failed (attempt {}/{}), retrying after {}ms: {:?}",
                        attempt + 1,
                        max_attempts,
                        backoff_ms,
                        e
                    );
                    Timer::after(Duration::from_millis(backoff_ms)).await;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            LocalAIError::ApiError("Max retries exceeded with no error recorded".to_string())
        }))
    }
}

#[async_trait]
impl crate::server::server_api::ai::AIClient for LocalAIClient {
    async fn generate_commands_from_natural_language(
        &self,
        prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> Result<Vec<AIGeneratedCommand>, crate::ai::GenerateCommandsFromNaturalLanguageError> {
        // For simplicity, return a single command with the AI response
        let messages = vec![ChatMessage::user(prompt)];

        let response = self.send_with_retry(|| {
            let client = self.provider_client.clone();
            let config = self.config.clone();
            let messages = messages.clone();
            Box::pin(async move {
                client.send_chat_request(&config, messages).await
            })
        }).await
        .map_err(|e| crate::ai::GenerateCommandsFromNaturalLanguageError::Other(e.into()))?;

        // Parse the response into commands
        // This is a simplified implementation
        Ok(vec![AIGeneratedCommand {
            command: response,
            original_text: None,
        }])
    }

    async fn generate_dialogue_answer(
        &self,
        transcript: Vec<TranscriptPart>,
        prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> anyhow::Result<GenerateDialogueResult> {
        let messages = Self::transcript_to_messages(&transcript, &prompt);
        let start_time = Instant::now();

        // Use streaming with retry logic
        let full_response = self.send_with_retry(|| {
            let client = self.provider_client.clone();
            let config = self.config.clone();
            let messages = messages.clone();
            Box::pin(async move {
                let mut stream = client.send_chat_request_streaming(&config, messages).await?;
                let mut full_response = String::new();

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            full_response.push_str(&chunk);
                        }
                        Err(e) => {
                            // Log chunk error but continue - some chunks might be malformed
                            log::warn!("Error processing stream chunk: {:?}", e);
                        }
                    }
                }

                if full_response.is_empty() {
                    return Err(LocalAIError::InvalidResponse(
                        "Empty response from stream".to_string()
                    ));
                }

                Ok(full_response)
            })
        }).await
        .map_err(|e| anyhow!("Local AI request failed: {:?}", e))?;

        log::debug!(
            "Local AI response generated in {:?}",
            start_time.elapsed()
        );

        Ok(GenerateDialogueResult {
            answer: full_response,
            request_limit_info: RequestLimitInfo {
                is_unlimited: true,
                request_limit: 1000,
                requests_used_since_last_refresh: 0,
                next_refresh_time: None,
            },
            transcript_summarized: false,
            truncated: false,
        })
    }

    async fn generate_metadata_for_command(
        &self,
        command: String,
    ) -> Result<GeneratedCommandMetadata, crate::drive::workflows::ai_assist::GeneratedCommandMetadataError> {
        let prompt = format!("Generate metadata for command: {}", command);
        let messages = vec![ChatMessage::user(prompt)];

        let response = self.send_with_retry(|| {
            let client = self.provider_client.clone();
            let config = self.config.clone();
            let messages = messages.clone();
            Box::pin(async move {
                client.send_chat_request(&config, messages).await
            })
        }).await
        .map_err(|e| crate::drive::workflows::ai_assist::GeneratedCommandMetadataError::Other(e.into()))?;

        // Parse the response into metadata
        // This is a simplified implementation - actual parsing would need more sophistication
        Ok(GeneratedCommandMetadata {
            title: response.chars().take(50).collect(),
            description: None,
            tags: Vec::new(),
        })
    }

    async fn get_request_limit_info(&self) -> Result<RequestUsageInfo, anyhow::Error> {
        // Local AI has no quota limits
        Ok(RequestUsageInfo {
            is_unlimited: true,
            requests_used: 0,
            request_limit: None,
            next_refresh_time: None,
            credits_remaining: None,
        })
    }

    async fn get_feature_model_choices(&self) -> Result<ModelsByFeature, anyhow::Error> {
        // Return a simple model list for local AI
        let model_name = match self.config.provider_type {
            ProviderType::OpenAI => "local-openai",
            ProviderType::Anthropic => "local-anthropic",
        };

        let llm_info = LLMInfo {
            display_name: "Local AI Model".to_string(),
            base_model_name: model_name.to_string(),
            id: model_name.into(),
            reasoning_level: None,
            usage_metadata: LLMUsageMetadata {
                request_multiplier: 1,
                credit_multiplier: None,
            },
            description: Some("Custom endpoint via local AI configuration".to_string()),
            disable_reason: None,
            vision_supported: false,
            spec: None,
            provider: crate::ai::llms::LLMProvider::Unknown,
            host_configs: HashMap::new(),
            discount_percentage: None,
            context_window: crate::ai::llms::LLMContextWindow::default(),
        };

        let available = AvailableLLMs::new(
            model_name.into(),
            vec![llm_info],
            None,
        )?;

        Ok(ModelsByFeature {
            agent_mode: available.clone(),
            coding: available,
            cli_agent: None,
            computer_use: None,
        })
    }

    async fn get_free_available_models(
        &self,
        _referrer: Option<String>,
    ) -> Result<ModelsByFeature, anyhow::Error> {
        self.get_feature_model_choices().await
    }

    // The following methods are not supported for local AI and return errors
    // They require server-side functionality

    async fn update_merkle_tree(
        &self,
        _embedding_config: crate::ai::index::full_source_code_embedding::EmbeddingConfig,
        _nodes: Vec<crate::ai::index::full_source_code_embedding::store_client::IntermediateNode>,
    ) -> anyhow::Result<HashMap<crate::ai::index::full_source_code_embedding::NodeHash, bool>> {
        Err(anyhow!("update_merkle_tree is not supported for local AI"))
    }

    async fn generate_code_embeddings(
        &self,
        _embedding_config: crate::ai::index::full_source_code_embedding::EmbeddingConfig,
        _fragments: Vec<crate::ai::index::full_source_code_embedding::Fragment>,
        _root_hash: crate::ai::index::full_source_code_embedding::NodeHash,
        _repo_metadata: crate::ai::index::full_source_code_embedding::RepoMetadata,
    ) -> anyhow::Result<HashMap<crate::ai::index::full_source_code_embedding::ContentHash, bool>> {
        Err(anyhow!("generate_code_embeddings is not supported for local AI"))
    }

    async fn provide_negative_feedback_response_for_ai_conversation(
        &self,
        _conversation_id: String,
        _request_ids: Vec<String>,
    ) -> anyhow::Result<i32, anyhow::Error> {
        Err(anyhow!("provide_negative_feedback_response_for_ai_conversation is not supported for local AI"))
    }

    async fn create_agent_task(
        &self,
        _prompt: String,
        _environment_uid: Option<String>,
        _parent_run_id: Option<String>,
        _config: Option<crate::ai::ambient_agents::AgentConfigSnapshot>,
    ) -> anyhow::Result<crate::ai::ambient_agents::AmbientAgentTaskId, anyhow::Error> {
        // For basic agent tasks, we could implement this locally
        // For now, return an error
        Err(anyhow!("create_agent_task requires server-side support"))
    }

    async fn update_agent_task(
        &self,
        _task_id: crate::ai::ambient_agents::AmbientAgentTaskId,
        _task_state: Option<crate::server::server_api::ai::AgentTaskState>,
        _session_id: Option<session_sharing_protocol::common::SessionId>,
        _conversation_id: Option<String>,
        _status_message: Option<crate::server::server_api::ai::TaskStatusUpdate>,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("update_agent_task is not supported for local AI"))
    }

    async fn spawn_agent(
        &self,
        _request: crate::server::server_api::ai::SpawnAgentRequest,
    ) -> anyhow::Result<crate::server::server_api::ai::SpawnAgentResponse, anyhow::Error> {
        Err(anyhow!("spawn_agent is not supported for local AI"))
    }

    async fn list_ambient_agent_tasks(
        &self,
        _limit: i32,
        _filter: crate::server::server_api::ai::TaskListFilter,
    ) -> anyhow::Result<Vec<crate::ai::ambient_agents::AmbientAgentTask>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn list_agent_runs_raw(
        &self,
        _limit: i32,
        _filter: crate::server::server_api::ai::TaskListFilter,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!([]))
    }

    async fn get_ambient_agent_task(
        &self,
        _task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> anyhow::Result<crate::ai::ambient_agents::AmbientAgentTask, anyhow::Error> {
        Err(anyhow!("get_ambient_agent_task is not supported for local AI"))
    }

    async fn get_agent_run_raw(
        &self,
        _task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Err(anyhow!("get_agent_run_raw is not supported for local AI"))
    }

    async fn submit_run_followup(
        &self,
        _run_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
        _request: crate::server::server_api::ai::RunFollowupRequest,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("submit_run_followup is not supported for local AI"))
    }

    async fn get_scheduled_agent_history(
        &self,
        _schedule_id: &str,
    ) -> anyhow::Result<crate::server::server_api::ai::ScheduledAgentHistory, anyhow::Error> {
        Err(anyhow!("get_scheduled_agent_history is not supported for local AI"))
    }

    async fn get_ai_conversation(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<(warp_multi_agent_api::ConversationData, crate::ai::agent::conversation::ServerAIConversationMetadata), anyhow::Error> {
        Err(anyhow!("get_ai_conversation is not supported for local AI"))
    }

    async fn list_ai_conversation_metadata(
        &self,
        _conversation_ids: Option<Vec<String>>,
    ) -> anyhow::Result<Vec<crate::ai::agent::conversation::ServerAIConversationMetadata>> {
        Ok(Vec::new())
    }

    async fn get_ai_conversation_format(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<crate::ai::agent::conversation::AIAgentConversationFormat, anyhow::Error> {
        Err(anyhow!("get_ai_conversation_format is not supported for local AI"))
    }

    async fn get_block_snapshot(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<crate::terminal::model::block::SerializedBlock, anyhow::Error> {
        Err(anyhow!("get_block_snapshot is not supported for local AI"))
    }

    async fn delete_ai_conversation(
        &self,
        _server_conversation_token: String,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("delete_ai_conversation is not supported for local AI"))
    }

    async fn list_agents(
        &self,
        _repo: Option<String>,
    ) -> anyhow::Result<Vec<crate::server::server_api::ai::AgentListItem>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn cancel_ambient_agent_task(
        &self,
        _task_id: &crate::ai::ambient_agents::AmbientAgentTaskId,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("cancel_ambient_agent_task is not supported for local AI"))
    }

    async fn get_task_attachments(
        &self,
        _task_id: String,
    ) -> anyhow::Result<Vec<crate::ai::ambient_agents::TaskAttachment>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn create_file_artifact_upload_target(
        &self,
        _filename: String,
        _file_size: u64,
    ) -> anyhow::Result<crate::ai::artifacts::FileArtifactUploadTarget, anyhow::Error> {
        Err(anyhow!("create_file_artifact_upload_target is not supported for local AI"))
    }

    async fn confirm_file_artifact_upload(
        &self,
        _upload_id: String,
        _parts: Vec<String>,
    ) -> anyhow::Result<crate::ai::artifacts::ConfirmedFileArtifactUpload, anyhow::Error> {
        Err(anyhow!("confirm_file_artifact_upload is not supported for local AI"))
    }

    async fn get_codebase_context_config(
        &self,
    ) -> anyhow::Result<crate::ai::index::full_source_code_embedding::CodebaseContextConfig, anyhow::Error> {
        Err(anyhow!("get_codebase_context_config is not supported for local AI"))
    }

    async fn get_relevant_fragments(
        &self,
        _query: String,
        _codebase_context_config: crate::ai::index::full_source_code_embedding::CodebaseContextConfig,
    ) -> anyhow::Result<Vec<crate::ai::index::full_source_code_embedding::Fragment>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn get_platform_status(&self) -> anyhow::Result<crate::server::server_api::ai::PlatformStatus> {
        Ok(crate::server::server_api::ai::PlatformStatus {
            platform_error_code: None,
            platform_status_message: None,
        })
    }

    async fn get_conversation_usage(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<crate::persistence::model::ConversationUsageMetadata, anyhow::Error> {
        Err(anyhow!("get_conversation_usage is not supported for local AI"))
    }
}
