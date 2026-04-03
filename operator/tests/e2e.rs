use fine_tuning::finetune::{FinetuneStatus, Hyperparams};

#[test]
fn hyperparams_serialization_roundtrip() {
    let params = Hyperparams {
        n_epochs: 3,
        batch_size: 8,
        learning_rate_multiplier: Some(1.5),
    };

    let json = serde_json::to_string(&params).unwrap();
    let deserialized: Hyperparams = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.n_epochs, 3);
    assert_eq!(deserialized.batch_size, 8);
    assert!(
        (deserialized.learning_rate_multiplier.unwrap() - 1.5).abs() < f64::EPSILON
    );
}

#[test]
fn hyperparams_without_learning_rate() {
    let params = Hyperparams {
        n_epochs: 5,
        batch_size: 16,
        learning_rate_multiplier: None,
    };

    let json = serde_json::to_string(&params).unwrap();
    // learning_rate_multiplier should be skipped when None
    assert!(!json.contains("learning_rate_multiplier"));

    let deserialized: Hyperparams = serde_json::from_str(&json).unwrap();
    assert!(deserialized.learning_rate_multiplier.is_none());
}

#[test]
fn hyperparams_from_json() {
    let json = r#"{"n_epochs": 10, "batch_size": 4}"#;
    let params: Hyperparams = serde_json::from_str(json).unwrap();

    assert_eq!(params.n_epochs, 10);
    assert_eq!(params.batch_size, 4);
    assert!(params.learning_rate_multiplier.is_none());
}

#[test]
fn finetune_status_serialization() {
    let status = FinetuneStatus {
        status: "running".to_string(),
        progress_pct: Some(45.5),
        error: None,
    };

    let json = serde_json::to_string(&status).unwrap();
    let deserialized: FinetuneStatus = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.status, "running");
    assert!((deserialized.progress_pct.unwrap() - 45.5).abs() < f32::EPSILON);
    assert!(deserialized.error.is_none());
}

#[test]
fn finetune_status_with_error() {
    let status = FinetuneStatus {
        status: "failed".to_string(),
        progress_pct: Some(23.0),
        error: Some("CUDA out of memory".to_string()),
    };

    let json = serde_json::to_string(&status).unwrap();
    let deserialized: FinetuneStatus = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.status, "failed");
    assert_eq!(
        deserialized.error.as_deref(),
        Some("CUDA out of memory")
    );
}

#[test]
fn finetune_status_completed() {
    let status = FinetuneStatus {
        status: "succeeded".to_string(),
        progress_pct: Some(100.0),
        error: None,
    };

    let json = serde_json::to_string(&status).unwrap();
    assert!(json.contains("succeeded"));
    assert!(json.contains("100"));
}

#[test]
fn finetune_client_from_endpoint() {
    // Verify the client constructor works without a full config
    let client = fine_tuning::finetune::FinetuneClient::from_endpoint("http://localhost:8080");
    // Client should be constructed (no panic). We can't test HTTP calls
    // without a server, but the construction path is validated.
    let _ = client;
}
