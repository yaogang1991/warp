//! Local AI client that implements AIClient trait for direct API calls.
//!
//! This client bypasses Warp's GraphQL API and calls AI services directly
//! using OpenAI-compatible or Anthropic-compatible protocols with streaming support.
//!
//! Methods that require server-side infrastructure (agent tasks, file uploads, etc.)
//! return errors since local AI does not have access to the Warp server.

use crate::ai::generate_code_review_content::api::{
    GenerateCodeReviewContentRequest, GenerateCodeReviewContentResponse,
};
use crate::ai::llms::{AvailableLLMs, LLMContextWindow, LLMInfo, LLMProvider, LLMUsageMetadata, ModelsByFeature};
use crate::ai::request_usage_model::{RequestLimitInfo, RequestUsageInfo};
use crate::ai::ambient_agents::AmbientAgentTaskId;
use crate::ai_assistant::{
    execution_context::WarpAiExecutionContext,
    requests::GenerateDialogueResult,
    utils::TranscriptPart,
    GenerateCommandsFromNaturalLanguageError,
};
use crate::drive::workflows::ai_assist::GeneratedCommandMetadataError;
use crate::server::local_ai::{get_provider_client, ChatMessage, LocalAIConfig, ProviderType};
use crate::server::server_api::ai::{
    AgentListItem, ArtifactDownloadResponse, AttachmentFileInfo, CreateFileArtifactUploadRequest,
    CreateFileArtifactUploadResponse, DownloadAttachmentsResponse, FileArtifactRecord,
    ListAgentMessagesRequest, AgentMessageHeader, PrepareAttachmentUploadsResponse,
    ReadAgentMessageResponse, ReportAgentEventRequest, ReportAgentEventResponse,
    RunFollowupRequest, SendAgentMessageRequest, SendAgentMessageResponse, SpawnAgentRequest,
    SpawnAgentResponse, TaskListFilter,
};
use crate::server::server_api::AIApiError;
use anyhow::anyhow;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use warpui::AppContext;
use warpui::SingletonEntity;

use ai::index::full_source_code_embedding::{
    self, store_client::IntermediateNode, EmbeddingConfig, NodeHash, ContentHash, RepoMetadata,
};

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

        for part in transcript {
            let user_text = &part.user.raw;
            if !user_text.is_empty() {
                messages.push(ChatMessage::user(user_text.clone()));
            }
            let assistant_text = &part.assistant.formatted_message.raw;
            if !assistant_text.is_empty() {
                messages.push(ChatMessage::assistant(assistant_text.clone()));
            }
        }

        if !prompt.is_empty() {
            messages.push(ChatMessage::user(prompt.to_string()));
        }

        messages
    }
}

#[async_trait]
impl crate::server::server_api::ai::AIClient for LocalAIClient {
    async fn generate_commands_from_natural_language(
        &self,
        _prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> Result<Vec<crate::ai_assistant::AIGeneratedCommand>, GenerateCommandsFromNaturalLanguageError> {
        // Local AI does not support command generation through this path.
        // Commands are generated via the dialogue flow instead.
        Err(GenerateCommandsFromNaturalLanguageError::Other)
    }

    async fn generate_dialogue_answer(
        &self,
        transcript: Vec<TranscriptPart>,
        prompt: String,
        _ai_execution_context: Option<WarpAiExecutionContext>,
    ) -> anyhow::Result<GenerateDialogueResult> {
        let messages = Self::transcript_to_messages(&transcript, &prompt);
        let start_time = Instant::now();

        let full_response = crate::server::local_ai::send_chat_with_retry(
            &*self.provider_client,
            &self.config,
            messages,
        ).await
        .map_err(|e| AIApiError::from(e))?;

        log::debug!(
            "Local AI response generated in {:?}",
            start_time.elapsed()
        );

        Ok(GenerateDialogueResult::Success {
            answer: full_response,
            request_limit_info: RequestLimitInfo::default(),
            transcript_summarized: false,
            truncated: false,
        })
    }

    async fn generate_metadata_for_command(
        &self,
        _command: String,
    ) -> Result<crate::drive::workflows::ai_assist::GeneratedCommandMetadata, GeneratedCommandMetadataError> {
        // Not supported for local AI
        Err(GeneratedCommandMetadataError::Other)
    }

    async fn get_request_limit_info(&self) -> Result<RequestUsageInfo, anyhow::Error> {
        Ok(RequestUsageInfo {
            request_limit_info: RequestLimitInfo::default(),
            bonus_grants: vec![],
        })
    }

    async fn get_feature_model_choices(&self) -> Result<ModelsByFeature, anyhow::Error> {
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
            provider: LLMProvider::Unknown,
            host_configs: HashMap::new(),
            discount_percentage: None,
            context_window: LLMContextWindow::default(),
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

    // --- Methods not supported for local AI (require server-side infrastructure) ---

    async fn update_merkle_tree(
        &self,
        _embedding_config: EmbeddingConfig,
        _nodes: Vec<IntermediateNode>,
    ) -> anyhow::Result<HashMap<NodeHash, bool>> {
        Err(anyhow!("update_merkle_tree is not supported for local AI"))
    }

    async fn generate_code_embeddings(
        &self,
        _embedding_config: EmbeddingConfig,
        _fragments: Vec<full_source_code_embedding::Fragment>,
        _root_hash: NodeHash,
        _repo_metadata: RepoMetadata,
    ) -> anyhow::Result<HashMap<ContentHash, bool>> {
        Err(anyhow!("generate_code_embeddings is not supported for local AI"))
    }

    async fn provide_negative_feedback_response_for_ai_conversation(
        &self,
        _conversation_id: String,
        _request_ids: Vec<String>,
    ) -> anyhow::Result<i32, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn create_agent_task(
        &self,
        _prompt: String,
        _environment_uid: Option<String>,
        _parent_run_id: Option<String>,
        _config: Option<crate::ai::ambient_agents::AgentConfigSnapshot>,
    ) -> anyhow::Result<AmbientAgentTaskId, anyhow::Error> {
        Err(anyhow!("create_agent_task requires server-side support"))
    }

    async fn update_agent_task(
        &self,
        _task_id: AmbientAgentTaskId,
        _task_state: Option<warp_graphql::ai::AgentTaskState>,
        _session_id: Option<session_sharing_protocol::common::SessionId>,
        _conversation_id: Option<String>,
        _status_message: Option<crate::server::server_api::ai::TaskStatusUpdate>,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn spawn_agent(
        &self,
        _request: SpawnAgentRequest,
    ) -> anyhow::Result<SpawnAgentResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn list_ambient_agent_tasks(
        &self,
        _limit: i32,
        _filter: TaskListFilter,
    ) -> anyhow::Result<Vec<crate::ai::ambient_agents::AmbientAgentTask>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn list_agent_runs_raw(
        &self,
        _limit: i32,
        _filter: TaskListFilter,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!([]))
    }

    async fn get_ambient_agent_task(
        &self,
        _task_id: &AmbientAgentTaskId,
    ) -> anyhow::Result<crate::ai::ambient_agents::AmbientAgentTask, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_agent_run_raw(
        &self,
        _task_id: &AmbientAgentTaskId,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn submit_run_followup(
        &self,
        _run_id: &AmbientAgentTaskId,
        _request: RunFollowupRequest,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_scheduled_agent_history(
        &self,
        _schedule_id: &str,
    ) -> anyhow::Result<warp_graphql::queries::get_scheduled_agent_history::ScheduledAgentHistory, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_ai_conversation(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<(warp_multi_agent_api::ConversationData, crate::ai::agent::conversation::ServerAIConversationMetadata), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
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
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_block_snapshot(
        &self,
        _server_conversation_token: crate::ai::agent::api::ServerConversationToken,
    ) -> anyhow::Result<crate::terminal::model::block::SerializedBlock, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn delete_ai_conversation(
        &self,
        _server_conversation_token: String,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn list_agents(
        &self,
        _repo: Option<String>,
    ) -> anyhow::Result<Vec<AgentListItem>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn cancel_ambient_agent_task(
        &self,
        _task_id: &AmbientAgentTaskId,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_task_attachments(
        &self,
        _task_id: String,
    ) -> anyhow::Result<Vec<crate::server::server_api::ai::TaskAttachment>, anyhow::Error> {
        Ok(Vec::new())
    }

    async fn create_file_artifact_upload_target(
        &self,
        _request: CreateFileArtifactUploadRequest,
    ) -> anyhow::Result<CreateFileArtifactUploadResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn confirm_file_artifact_upload(
        &self,
        _artifact_uid: String,
        _checksum: String,
    ) -> anyhow::Result<FileArtifactRecord, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_artifact_download(
        &self,
        _artifact_uid: &str,
    ) -> anyhow::Result<ArtifactDownloadResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn prepare_attachments_for_upload(
        &self,
        _task_id: &AmbientAgentTaskId,
        _files: &[AttachmentFileInfo],
    ) -> anyhow::Result<PrepareAttachmentUploadsResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn download_task_attachments(
        &self,
        _task_id: &AmbientAgentTaskId,
        _attachment_ids: &[String],
    ) -> anyhow::Result<DownloadAttachmentsResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_handoff_snapshot_attachments(
        &self,
        _task_id: &AmbientAgentTaskId,
    ) -> anyhow::Result<Vec<crate::server::server_api::ai::TaskAttachment>, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn send_agent_message(
        &self,
        _request: SendAgentMessageRequest,
    ) -> anyhow::Result<SendAgentMessageResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn list_agent_messages(
        &self,
        _run_id: &str,
        _request: ListAgentMessagesRequest,
    ) -> anyhow::Result<Vec<AgentMessageHeader>, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn update_event_sequence_on_server(
        &self,
        _run_id: &str,
        _sequence: i64,
    ) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn report_agent_event(
        &self,
        _run_id: &str,
        _request: ReportAgentEventRequest,
    ) -> anyhow::Result<ReportAgentEventResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn mark_message_delivered(&self, _message_id: &str) -> anyhow::Result<(), anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn read_agent_message(
        &self,
        _message_id: &str,
    ) -> anyhow::Result<ReadAgentMessageResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_public_conversation(
        &self,
        _conversation_id: &str,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn get_run_conversation(
        &self,
        _run_id: &str,
    ) -> anyhow::Result<serde_json::Value, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }

    async fn generate_code_review_content(
        &self,
        _request: GenerateCodeReviewContentRequest,
    ) -> Result<GenerateCodeReviewContentResponse, anyhow::Error> {
        Err(anyhow!("not supported for local AI"))
    }
}
