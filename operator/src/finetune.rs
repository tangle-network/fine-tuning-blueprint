//! Fine-tuning backend client.
//!
//! Two modes:
//! - **Local**: calls a training server (axolotl, unsloth, or torchtune HTTP API)
//! - **API**: forwards to Modal/Lambda/RunPod training endpoint
//!
//! Configurable via `FINETUNE_ENDPOINT` env var or operator config.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};

/// Hyperparameters for a fine-tuning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparams {
    pub n_epochs: u32,
    pub batch_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
}

/// Status of a fine-tuning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuneStatus {
    pub status: String,
    pub progress_pct: Option<f32>,
    pub error: Option<String>,
}

/// Request body sent to the training backend.
#[derive(Debug, Serialize)]
struct TrainingRequest {
    base_model: String,
    dataset_url: String,
    method: String,
    hyperparams: Hyperparams,
}

/// Response from the training backend on job submission.
#[derive(Debug, Deserialize)]
struct TrainingSubmitResponse {
    job_id: String,
}

/// Response from the training backend on status check.
#[derive(Debug, Deserialize)]
struct TrainingStatusResponse {
    status: String,
    #[serde(default)]
    progress_pct: Option<f32>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    adapter_url: Option<String>,
}

/// Client for communicating with the fine-tuning training backend.
#[derive(Clone)]
pub struct FinetuneClient {
    endpoint: String,
    client: reqwest::Client,
}

impl FinetuneClient {
    /// Create a new client from operator config.
    pub fn new(config: &crate::config::OperatorConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            endpoint: config.finetune.endpoint.clone(),
            client,
        })
    }

    /// Create a client from a raw endpoint URL (for testing).
    pub fn from_endpoint(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Submit a fine-tuning job to the training backend.
    /// Returns the job ID assigned by the backend.
    pub async fn submit_job(
        &self,
        base_model: &str,
        dataset_url: &str,
        method: &str,
        hyperparams: Hyperparams,
    ) -> anyhow::Result<String> {
        let req = TrainingRequest {
            base_model: base_model.to_string(),
            dataset_url: dataset_url.to_string(),
            method: method.to_string(),
            hyperparams,
        };

        let resp = self
            .client
            .post(format!("{}/v1/fine_tuning/jobs", self.endpoint))
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("training backend returned {status}: {body}");
        }

        let result: TrainingSubmitResponse = resp.json().await?;
        Ok(result.job_id)
    }

    /// Get the status of a fine-tuning job.
    pub async fn get_status(&self, job_id: &str) -> anyhow::Result<FinetuneStatus> {
        let resp = self
            .client
            .get(format!("{}/v1/fine_tuning/jobs/{}", self.endpoint, job_id))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("training backend returned {status}: {body}");
        }

        let result: TrainingStatusResponse = resp.json().await?;
        Ok(FinetuneStatus {
            status: result.status,
            progress_pct: result.progress_pct,
            error: result.error,
        })
    }

    /// Get the adapter download URL for a completed job.
    pub async fn get_result(&self, job_id: &str) -> anyhow::Result<String> {
        let resp = self
            .client
            .get(format!("{}/v1/fine_tuning/jobs/{}", self.endpoint, job_id))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("training backend returned {status}: {body}");
        }

        let result: TrainingStatusResponse = resp.json().await?;
        result
            .adapter_url
            .ok_or_else(|| anyhow::anyhow!("job {job_id} has no adapter URL (not completed?)"))
    }

    /// Check if the training backend is reachable.
    pub async fn health_check(&self) -> anyhow::Result<()> {
        let resp = self
            .client
            .get(format!("{}/health", self.endpoint))
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!(
                "training backend health check failed: {}",
                resp.status()
            );
        }
        Ok(())
    }

    /// Cancel a running fine-tuning job.
    pub async fn cancel_job(&self, job_id: &str) -> anyhow::Result<()> {
        let resp = self
            .client
            .post(format!(
                "{}/v1/fine_tuning/jobs/{}/cancel",
                self.endpoint, job_id
            ))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("cancel failed: {status}: {body}");
        }
        Ok(())
    }

    /// List all fine-tuning jobs from the backend.
    pub async fn list_jobs(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let resp = self
            .client
            .get(format!("{}/v1/fine_tuning/jobs", self.endpoint))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("list jobs failed: {status}: {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let jobs = body["data"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        Ok(jobs)
    }
}
