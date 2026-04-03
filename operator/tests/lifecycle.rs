//! Full lifecycle test -- fine-tune submission through real handler + wiremock backend.

use anyhow::{Result, ensure};
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::{method, path}};
use fine_tuning::FinetuneRequest;

#[tokio::test]
async fn test_submit_finetune_direct_with_wiremock() -> Result<()> {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/fine_tuning/jobs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "job_id": "ft-job-xyz789"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    fine_tuning::init_for_testing(&mock_server.uri());

    let request = FinetuneRequest {
        baseModel: "meta-llama/Llama-3.1-8B".into(),
        datasetUrl: "https://example.com/dataset.jsonl".into(),
        method: "lora".into(),
        epochs: 3,
        batchSize: 4,
    };

    let result = fine_tuning::submit_finetune_direct(&request).await;

    match result {
        Ok(job_id) => {
            ensure!(
                job_id == "ft-job-xyz789",
                "expected job_id 'ft-job-xyz789', got '{job_id}'"
            );
        }
        Err(e) => panic!("Fine-tune submission failed: {e}"),
    }

    mock_server.verify().await;

    Ok(())
}
