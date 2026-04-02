use blueprint_std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use fine_tuning::config::OperatorConfig;
use fine_tuning::health;
use fine_tuning::FinetuneServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for FinetuneBSM.onRegister.
/// Format: abi.encode(string[] supportedMethods, string[] supportedModels, uint32 gpuCount, uint32 totalVramMib, string endpoint)
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.finetune.supported_methods.clone(),
        config.finetune.supported_models.clone(),
        gpu_count,
        total_vram,
        endpoint,
    )
        .abi_encode()
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    // Load operator config
    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    // Load blueprint environment
    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit registration inputs and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            methods = ?config.finetune.supported_methods,
            models = ?config.finetune.supported_models,
            "Registration payload saved"
        );
        return Ok(());
    }

    // Check GPU availability (non-fatal)
    match health::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed — running in CPU mode");
        }
    }

    // Get Tangle client
    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    // Get service ID
    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    // Producer + Consumer
    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        match fine_tuning::qos::start_heartbeat(config.clone()).await {
            Ok(_handle) => {
                let interval = config.qos.as_ref().unwrap().heartbeat_interval_secs;
                tracing::info!(interval_secs = interval, "QoS heartbeat started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS heartbeat failed to start (disabled)");
            }
        }
    } else {
        tracing::info!("QoS heartbeat disabled");
    }

    // Background service: HTTP server
    let finetune_server = FinetuneServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(fine_tuning::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(finetune_server)
        .run()
        .await?;

    Ok(())
}
