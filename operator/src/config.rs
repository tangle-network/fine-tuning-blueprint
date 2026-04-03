use serde::{Deserialize, Serialize};
use blueprint_sdk::std::fmt;
use blueprint_sdk::std::path::PathBuf;

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration
    pub tangle: TangleConfig,

    /// Fine-tuning backend configuration
    pub finetune: FinetuneConfig,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Billing configuration
    pub billing: BillingConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional -- disabled by default)
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("finetune", &self.finetune)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    pub rpc_url: String,

    /// Chain ID
    pub chain_id: u64,

    /// Operator's private key (hex, without 0x prefix)
    pub operator_key: String,

    /// Tangle core contract address
    pub tangle_core: String,

    /// ShieldedCredits contract address
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuneConfig {
    /// Training backend endpoint URL (local server or remote API)
    pub endpoint: String,

    /// Supported fine-tuning methods (e.g. ["lora", "qlora", "full"])
    #[serde(default = "default_supported_methods")]
    pub supported_methods: Vec<String>,

    /// Supported base models for fine-tuning
    #[serde(default)]
    pub supported_models: Vec<String>,

    /// Maximum number of training epochs allowed
    #[serde(default = "default_max_epochs")]
    pub max_epochs: u32,

    /// Maximum dataset size in bytes (to prevent abuse)
    #[serde(default = "default_max_dataset_size_bytes")]
    pub max_dataset_size_bytes: u64,

    /// Directory where trained adapters/checkpoints are stored
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// External host to bind
    #[serde(default = "default_host")]
    pub host: String,

    /// External port to bind
    #[serde(default = "default_port")]
    pub port: u16,

    /// Maximum request body size in bytes (default 16 MiB for dataset uploads)
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Per-request timeout in seconds (default 300)
    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing is required for HTTP requests.
    #[serde(default = "default_billing_required")]
    pub required: bool,

    /// Price per epoch per billion parameters in tsUSD base units.
    /// Training is priced per compute: epochs * model_size_tier * this rate.
    pub price_per_epoch_per_billion_params: u64,

    /// Maximum amount a single SpendAuth can authorize (anti-abuse)
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account to serve a request
    pub min_credit_balance: u64,

    /// Minimum charge amount per request (gas cost protection).
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls.
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks.
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Path to persist used nonces across restarts (replay protection).
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<PathBuf>,

    /// ERC-20 token address for x402 payment (e.g. tsUSD).
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB.
    /// LoRA: 24576 (24 GB), full fine-tuning: 81920 (80 GB).
    pub min_vram_mib: u32,

    /// GPU model name (e.g. "NVIDIA A100", "RTX 4090")
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
}

fn default_supported_methods() -> Vec<String> {
    vec!["lora".to_string(), "qlora".to_string()]
}

fn default_max_epochs() -> u32 {
    100
}

fn default_max_dataset_size_bytes() -> u64 {
    1024 * 1024 * 1024 // 1 GiB
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("data/adapters")
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_request_body_bytes() -> usize {
    16 * 1024 * 1024 // 16 MiB
}

fn default_stream_timeout_secs() -> u64 {
    300
}

fn default_billing_required() -> bool {
    true
}

fn default_monitor_interval() -> u64 {
    30
}

fn default_claim_max_retries() -> u32 {
    3
}

fn default_clock_skew_tolerance() -> u64 {
    30
}

fn default_nonce_store_path() -> Option<PathBuf> {
    Some(PathBuf::from("data/nonces.json"))
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: FT_OP_ (e.g. FT_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("FT_OP")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "finetune": {
                "endpoint": "http://localhost:9000",
                "supported_models": ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-v0.3"],
                "supported_methods": ["lora", "qlora", "full"]
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "billing": {
                "price_per_epoch_per_billion_params": 100000,
                "max_spend_per_request": 10000000,
                "min_credit_balance": 10000
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 24576
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.finetune.endpoint, "http://localhost:9000");
        assert_eq!(cfg.finetune.supported_models.len(), 2);
        assert_eq!(cfg.finetune.supported_methods.len(), 3);
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.billing.price_per_epoch_per_billion_params, 100000);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.finetune.max_epochs, 100);
        assert_eq!(cfg.finetune.max_dataset_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.finetune.output_dir, PathBuf::from("data/adapters"));
        assert_eq!(cfg.server.max_request_body_bytes, 16 * 1024 * 1024);
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_serialize() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: OperatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.tangle.chain_id, cfg2.tangle.chain_id);
        assert_eq!(cfg.finetune.endpoint, cfg2.finetune.endpoint);
        assert_eq!(cfg.server.port, cfg2.server.port);
    }
}
