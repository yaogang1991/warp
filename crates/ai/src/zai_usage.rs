//! Z.ai (Zhipu AI / GLM) usage query client.
//!
//! Queries the GLM Coding Plan monitoring API to retrieve quota usage
//! percentages (5-hour token window, monthly MCP usage) and 24-hour
//! model usage statistics.

use serde::Deserialize;

const ZAI_API_BASE: &str = "https://open.bigmodel.cn/api/coding/paas/v4";
const QUOTA_LIMIT_PATH: &str = "/api/monitor/usage/quota/limit";
const MODEL_USAGE_PATH: &str = "/api/monitor/usage/model-usage";

// ── Response types ──────────────────────────────────────────────

/// Response from the Z.ai quota limit endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ZaiQuotaResponse {
    pub success: bool,
    pub data: Option<ZaiQuotaData>,
}

/// Quota usage percentages returned by Z.ai.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZaiQuotaData {
    /// Token usage percentage over the 5-hour rolling window (0–100).
    pub token_usage5_hour: f64,
    /// MCP usage percentage over the 1-month rolling window (0–100).
    pub mcp_usage1_month: Option<f64>,
}

/// Response from the Z.ai model usage endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ZaiModelUsageResponse {
    pub success: bool,
    pub data: Option<ZaiModelUsageData>,
}

/// 24-hour model usage statistics.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZaiModelUsageData {
    /// Total tokens used in the 24-hour window.
    pub total_tokens: u64,
    /// Total API calls in the 24-hour window.
    pub total_calls: u64,
    /// Per-model breakdown (optional).
    pub models: Option<Vec<ZaiModelEntry>>,
}

/// A single model's usage entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZaiModelEntry {
    pub model_name: String,
    pub tokens: u64,
    pub calls: u64,
}

// ── Aggregated usage info (consumed by the UI layer) ───────────

/// Aggregated Z.ai usage information for display in Warp.
#[derive(Debug, Clone, Default)]
pub struct ZaiUsageInfo {
    /// Whether the API call succeeded.
    pub available: bool,
    /// 5-hour token usage percentage (0–100).
    pub token_usage_5h_percent: Option<f64>,
    /// Monthly MCP usage percentage (0–100).
    pub mcp_usage_monthly_percent: Option<f64>,
    /// Total tokens used in the last 24 hours.
    pub tokens_24h: Option<u64>,
    /// Total API calls in the last 24 hours.
    pub calls_24h: Option<u64>,
}

// ── Client ──────────────────────────────────────────────────────

/// Lightweight HTTP client for querying Z.ai usage.
pub struct ZaiUsageClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl ZaiUsageClient {
    /// Create a new client with the given Z.ai API key.
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, ZAI_API_BASE.to_string())
    }

    /// Create a client with a custom base URL (useful for testing).
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("reqwest client build should not fail");
        Self {
            api_key,
            base_url,
            client,
        }
    }

    /// Fetch all usage information in one call.
    pub async fn fetch_usage(&self) -> ZaiUsageInfo {
        let (quota, model_usage) =
            tokio::join!(self.fetch_quota(), self.fetch_model_usage());

        let quota_data = quota.ok().and_then(|r| r.data);
        let model_data = model_usage.ok().and_then(|r| r.data);

        let (token_5h, mcp_monthly) = match &quota_data {
            Some(d) => (Some(d.token_usage5_hour), d.mcp_usage1_month),
            None => (None, None),
        };

        let (tokens_24h, calls_24h) = match &model_data {
            Some(d) => (Some(d.total_tokens), Some(d.total_calls)),
            None => (None, None),
        };

        ZaiUsageInfo {
            available: quota_data.is_some() || model_data.is_some(),
            token_usage_5h_percent: token_5h,
            mcp_usage_monthly_percent: mcp_monthly,
            tokens_24h,
            calls_24h,
        }
    }

    /// Query the 5-hour / monthly quota endpoint.
    pub async fn fetch_quota(&self) -> Result<ZaiQuotaResponse, reqwest::Error> {
        let url = format!("{}{}", self.base_url, QUOTA_LIMIT_PATH);
        self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept-Language", "en-US,en")
            .send()
            .await?
            .json()
            .await
    }

    /// Query the 24-hour model usage endpoint.
    pub async fn fetch_model_usage(&self) -> Result<ZaiModelUsageResponse, reqwest::Error> {
        let url = format!("{}{}", self.base_url, MODEL_USAGE_PATH);
        self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept-Language", "en-US,en")
            .send()
            .await?
            .json()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_quota_response() {
        let json = r#"{"success":true,"data":{"tokenUsage5Hour":45.2,"mcpUsage1Month":12.3}}"#;
        let resp: ZaiQuotaResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        let data = resp.data.unwrap();
        assert!((data.token_usage5_hour - 45.2).abs() < 0.01);
        assert!((data.mcp_usage1_month.unwrap() - 12.3).abs() < 0.01);
    }

    #[test]
    fn deserialize_model_usage_response() {
        let json = r#"{"success":true,"data":{"totalTokens":12500000,"totalCalls":1234,"models":[{"modelName":"glm-4-flash","tokens":8000000,"calls":800}]}}"#;
        let resp: ZaiModelUsageResponse = serde_json::from_str(json).unwrap();
        assert!(resp.success);
        let data = resp.data.unwrap();
        assert_eq!(data.total_tokens, 12_500_000);
        assert_eq!(data.total_calls, 1234);
        let models = data.models.unwrap();
        assert_eq!(models[0].model_name, "glm-4-flash");
    }
}
