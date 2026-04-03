use std::sync::Arc;

use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use fine_tuning::config::{
    BillingConfig, FinetuneConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig,
};
use fine_tuning::finetune::FinetuneClient;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn test_config(backend_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        finetune: FinetuneConfig {
            endpoint: format!("http://127.0.0.1:{backend_port}"),
            supported_methods: vec!["lora".into(), "qlora".into(), "full".into()],
            supported_models: vec![
                "meta-llama/Llama-3.1-8B-Instruct".into(),
                "mistralai/Mistral-7B-v0.3".into(),
            ],
            max_epochs: 100,
            max_dataset_size_bytes: 1024 * 1024 * 1024,
            output_dir: std::path::PathBuf::from("/tmp/test-adapters"),
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            max_request_body_bytes: 16 * 1024 * 1024,
            stream_timeout_secs: 300,
        },
        billing: BillingConfig {
            required: false,
            price_per_epoch_per_billion_params: 100000,
            max_spend_per_request: 10000000,
            min_credit_balance: 10000,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            gpu_model: None,
            monitor_interval_secs: 30,
        },
        qos: None,
    }
}

async fn start_test_server(
    backend_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(backend_port);
    config.server.port = server_port;
    let config = Arc::new(config);

    let finetune = Arc::new(FinetuneClient::from_endpoint(&format!(
        "http://127.0.0.1:{backend_port}"
    )));

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let state = fine_tuning::server::AppState {
        config,
        finetune,
    };

    let handle = fine_tuning::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_healthy() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["backend_reachable"], true);
}

#[tokio::test]
async fn test_health_check_unhealthy() {
    let mock = MockServer::start().await;
    // no health mock -> 404 -> unhealthy

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "degraded");
}

#[tokio::test]
async fn test_create_job_success() {
    let mock = MockServer::start().await;

    // The FinetuneClient POSTs to /v1/fine_tuning/jobs on the BACKEND
    Mock::given(method("POST"))
        .and(path("/v1/fine_tuning/jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "job_id": "ft-abc123"
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/fine_tuning/jobs"))
        .json(&serde_json::json!({
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "training_file": "https://example.com/dataset.jsonl",
            "method": "lora",
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 4
            }
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "ft-abc123");
    assert_eq!(body["object"], "fine_tuning.job");
    assert_eq!(body["status"], "queued");
    assert_eq!(body["method"], "lora");
    assert_eq!(body["model"], "meta-llama/Llama-3.1-8B-Instruct");
}

#[tokio::test]
async fn test_create_job_unsupported_method() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/fine_tuning/jobs"))
        .json(&serde_json::json!({
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "training_file": "https://example.com/dataset.jsonl",
            "method": "nonexistent",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("unsupported method"));
}

#[tokio::test]
async fn test_create_job_unsupported_model() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/fine_tuning/jobs"))
        .json(&serde_json::json!({
            "model": "not-a-real-model",
            "training_file": "https://example.com/dataset.jsonl",
            "method": "lora",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("unsupported model"));
}

#[tokio::test]
async fn test_get_job_status() {
    let mock = MockServer::start().await;

    // The FinetuneClient GETs /v1/fine_tuning/jobs/:id on the BACKEND
    Mock::given(method("GET"))
        .and(path("/v1/fine_tuning/jobs/ft-abc123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "running",
            "progress_pct": 45.0,
            "error": null,
            "adapter_url": null
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!(
        "http://127.0.0.1:{port}/v1/fine_tuning/jobs/ft-abc123"
    ))
    .await
    .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "ft-abc123");
    assert_eq!(body["status"], "running");
    assert_eq!(body["progress_pct"], 45.0);
}

#[tokio::test]
async fn test_get_job_not_found() {
    let mock = MockServer::start().await;

    // Backend returns 404 for unknown job
    Mock::given(method("GET"))
        .and(path("/v1/fine_tuning/jobs/nonexistent"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": "not found"
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!(
        "http://127.0.0.1:{port}/v1/fine_tuning/jobs/nonexistent"
    ))
    .await
    .unwrap();

    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_list_jobs() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/fine_tuning/jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"id": "ft-1", "status": "completed"},
                {"id": "ft-2", "status": "running"}
            ]
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/v1/fine_tuning/jobs"))
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_cancel_job() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/fine_tuning/jobs/ft-abc123/cancel"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{port}/v1/fine_tuning/jobs/ft-abc123/cancel"
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "cancelled");
}
