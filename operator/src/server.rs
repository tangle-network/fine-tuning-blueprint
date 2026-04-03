use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::finetune::FinetuneClient;

// --- x402 constants ---

const X402_PAYMENT_REQUIRED: &str = "X-Payment-Required";
const X402_PAYMENT_SIGNATURE: &str = "X-Payment-Signature";

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub finetune: Arc<FinetuneClient>,
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/fine_tuning/jobs", post(create_job))
        .route("/v1/fine_tuning/jobs", get(list_jobs))
        .route("/v1/fine_tuning/jobs/:id", get(get_job))
        .route("/v1/fine_tuning/jobs/:id/cancel", post(cancel_job))
        .route("/health", get(health_check))
        .layer(DefaultBodyLimit::max(
            state.config.server.max_request_body_bytes,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.stream_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal, draining connections");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request / Response types (OpenAI Fine-tuning API compatible) ---

#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    /// Base model to fine-tune (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    pub model: String,
    /// URL to the training dataset (JSONL format)
    pub training_file: String,
    /// Fine-tuning method: "lora", "qlora", or "full"
    #[serde(default = "default_method")]
    pub method: String,
    /// Training hyperparameters
    #[serde(default)]
    pub hyperparameters: HyperparametersInput,

    /// SpendAuth payload for billing (can also be in X-Payment-Signature header)
    pub spend_auth: Option<SpendAuthPayload>,
}

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub amount: u64,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct HyperparametersInput {
    #[serde(default = "default_epochs")]
    pub n_epochs: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default)]
    pub learning_rate_multiplier: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub status: String,
    pub method: String,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<i64>,
    pub hyperparameters: HyperparametersOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_files: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trained_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
}

#[derive(Debug, Serialize)]
pub struct HyperparametersOutput {
    pub n_epochs: u32,
    pub batch_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct JobError {
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ListJobsResponse {
    pub object: &'static str,
    pub data: Vec<JobResponse>,
}

fn default_method() -> String {
    "lora".to_string()
}

fn default_epochs() -> u32 {
    3
}

fn default_batch_size() -> u32 {
    4
}

// --- Handlers ---

/// POST /v1/fine_tuning/jobs — create a fine-tuning job
async fn create_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    // Validate SpendAuth if billing is required
    if state.config.billing.required {
        let has_spend_auth = req.spend_auth.is_some()
            || headers.get(X402_PAYMENT_SIGNATURE).is_some();
        if !has_spend_auth {
            return (
                StatusCode::PAYMENT_REQUIRED,
                [(X402_PAYMENT_REQUIRED, "true")],
                Json(serde_json::json!({
                    "error": {
                        "message": "Payment required. Include spend_auth or X-Payment-Signature header.",
                        "type": "payment_required"
                    }
                })),
            )
                .into_response();
        }
    }

    // Validate method
    let valid_methods = &state.config.finetune.supported_methods;
    if !valid_methods.contains(&req.method) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("unsupported method '{}'. Supported: {:?}", req.method, valid_methods),
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Validate model
    let valid_models = &state.config.finetune.supported_models;
    if !valid_models.contains(&req.model) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("unsupported model '{}'. Supported: {:?}", req.model, valid_models),
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Validate epochs
    if req.hyperparameters.n_epochs > state.config.finetune.max_epochs {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("n_epochs {} exceeds max {}", req.hyperparameters.n_epochs, state.config.finetune.max_epochs),
                    "type": "invalid_request_error"
                }
            })),
        )
            .into_response();
    }

    // Submit to backend
    let hyperparams = crate::finetune::Hyperparams {
        n_epochs: req.hyperparameters.n_epochs,
        batch_size: req.hyperparameters.batch_size,
        learning_rate_multiplier: req.hyperparameters.learning_rate_multiplier,
    };

    match state
        .finetune
        .submit_job(&req.model, &req.training_file, &req.method, hyperparams)
        .await
    {
        Ok(job_id) => {
            let now = chrono::Utc::now().timestamp();
            let resp = JobResponse {
                id: job_id,
                object: "fine_tuning.job",
                model: req.model,
                status: "queued".to_string(),
                method: req.method,
                created_at: now,
                finished_at: None,
                hyperparameters: HyperparametersOutput {
                    n_epochs: req.hyperparameters.n_epochs,
                    batch_size: req.hyperparameters.batch_size,
                    learning_rate_multiplier: req.hyperparameters.learning_rate_multiplier,
                },
                result_files: None,
                trained_tokens: None,
                error: None,
            };
            (StatusCode::OK, Json(serde_json::to_value(&resp).unwrap())).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to submit fine-tuning job");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("failed to submit job: {e}"),
                        "type": "server_error"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// GET /v1/fine_tuning/jobs — list all jobs
async fn list_jobs(State(state): State<AppState>) -> impl IntoResponse {
    match state.finetune.list_jobs().await {
        Ok(jobs) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "object": "list",
                "data": jobs
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("failed to list jobs: {e}"),
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
    }
}

/// GET /v1/fine_tuning/jobs/:id — get job status
async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.finetune.get_status(&job_id).await {
        Ok(status) => {
            let adapter_url = if status.status == "completed" {
                state.finetune.get_result(&job_id).await.ok()
            } else {
                None
            };

            let resp = serde_json::json!({
                "id": job_id,
                "object": "fine_tuning.job",
                "status": status.status,
                "progress_pct": status.progress_pct,
                "result_files": adapter_url.map(|u| vec![u]),
                "error": status.error.map(|msg| { serde_json::json!({"message": msg}) }),
            });
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("job not found: {e}"),
                    "type": "not_found"
                }
            })),
        )
            .into_response(),
    }
}

/// POST /v1/fine_tuning/jobs/:id/cancel — cancel a job
async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.finetune.cancel_job(&job_id).await {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": job_id,
                "object": "fine_tuning.job",
                "status": "cancelled"
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("failed to cancel job: {e}"),
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
    }
}

/// GET /health — health check
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    let backend_ok = state.finetune.health_check().await.is_ok();
    let status = if backend_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "status": if backend_ok { "ok" } else { "degraded" },
            "backend_reachable": backend_ok,
        })),
    )
}
