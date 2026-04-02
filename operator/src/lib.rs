pub mod config;
pub mod finetune;
pub mod health;
pub mod qos;
pub mod server;

use blueprint_std::sync::{Arc, OnceLock};

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::finetune::FinetuneClient;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call for submitting a fine-tuning job.
    struct FinetuneRequest {
        string baseModel;
        string datasetUrl;
        string method;
        uint32 epochs;
        uint32 batchSize;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct FinetuneResult {
        string jobId;
        string status;
        string adapterUrl;
    }
}

// --- Job IDs ---

pub const FINETUNE_JOB: u8 = 0;
pub const FINETUNE_STATUS_JOB: u8 = 1;

// --- Shared state for on-chain job handlers ---

static FINETUNE_ENDPOINT: OnceLock<FinetuneEndpoint> = OnceLock::new();

struct FinetuneEndpoint {
    client: FinetuneClient,
}

#[allow(clippy::result_large_err)]
fn register_finetune_endpoint(config: &OperatorConfig) -> Result<(), RunnerError> {
    let client = FinetuneClient::new(config).map_err(|e| {
        RunnerError::Other(format!("failed to build finetune client: {e}").into())
    })?;
    let endpoint = FinetuneEndpoint { client };
    let _ = FINETUNE_ENDPOINT.set(endpoint);
    Ok(())
}

/// Initialize the finetune endpoint for testing.
pub fn init_for_testing(base_url: &str) {
    let client = FinetuneClient::from_endpoint(base_url);
    let _ = FINETUNE_ENDPOINT.set(FinetuneEndpoint { client });
}

// --- Router ---

pub fn router() -> Router {
    Router::new()
        .route(
            FINETUNE_JOB,
            run_finetune
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            FINETUNE_STATUS_JOB,
            check_finetune_status
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
}

// --- Job handlers ---

/// Handle a fine-tuning job submission on-chain.
#[debug_job]
pub async fn run_finetune(
    TangleArg(request): TangleArg<FinetuneRequest>,
) -> Result<TangleResult<FinetuneResult>, RunnerError> {
    let endpoint = FINETUNE_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other(
            "finetune endpoint not registered — FinetuneServer not started".into(),
        )
    })?;

    let job_id = endpoint
        .client
        .submit_job(
            &request.baseModel,
            &request.datasetUrl,
            &request.method,
            finetune::Hyperparams {
                n_epochs: request.epochs,
                batch_size: request.batchSize,
                learning_rate_multiplier: None,
            },
        )
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "finetune job submission failed");
            RunnerError::Other(format!("finetune job submission failed: {e}").into())
        })?;

    Ok(TangleResult(FinetuneResult {
        jobId: job_id,
        status: "queued".to_string(),
        adapterUrl: String::new(),
    }))
}

/// Handle a fine-tuning status check on-chain.
/// Input: FinetuneResult with jobId set, output: updated FinetuneResult.
#[debug_job]
pub async fn check_finetune_status(
    TangleArg(request): TangleArg<FinetuneResult>,
) -> Result<TangleResult<FinetuneResult>, RunnerError> {
    let endpoint = FINETUNE_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other(
            "finetune endpoint not registered — FinetuneServer not started".into(),
        )
    })?;

    let status = endpoint
        .client
        .get_status(&request.jobId)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, job_id = %request.jobId, "finetune status check failed");
            RunnerError::Other(format!("finetune status check failed: {e}").into())
        })?;

    let adapter_url = if status.status == "completed" {
        endpoint
            .client
            .get_result(&request.jobId)
            .await
            .unwrap_or_default()
    } else {
        String::new()
    };

    Ok(TangleResult(FinetuneResult {
        jobId: request.jobId,
        status: status.status,
        adapterUrl: adapter_url,
    }))
}

// --- Background service: HTTP server ---

/// Runs the OpenAI Fine-tuning API compatible HTTP server as a
/// [`BackgroundService`]. Starts before the BlueprintRunner begins
/// polling for on-chain jobs.
#[derive(Clone)]
pub struct FinetuneServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for FinetuneServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // Register the finetune endpoint for on-chain job handlers
            if let Err(e) = register_finetune_endpoint(&config) {
                tracing::error!(error = %e, "failed to register finetune endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // Verify finetune backend is reachable
            let client = &FINETUNE_ENDPOINT.get().unwrap().client;
            match client.health_check().await {
                Ok(()) => tracing::info!("finetune backend is reachable"),
                Err(e) => {
                    tracing::warn!(error = %e, "finetune backend health check failed — will retry on requests");
                }
            }

            // Create shutdown channel for graceful shutdown
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // Start the HTTP server
            let state = server::AppState {
                config: config.clone(),
                finetune: Arc::new(FINETUNE_ENDPOINT.get().unwrap().client.clone()),
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started — background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // Shutdown listener
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("received shutdown signal");
                    let _ = shutdown_tx.send(true);
                }
            }
        });

        Ok(rx)
    }
}
