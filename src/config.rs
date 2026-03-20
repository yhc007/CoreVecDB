use serde::Deserialize;
use config::{Config, ConfigError, File, Environment};

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub index: IndexConfig,
    pub quantization: QuantizationConfig,
    pub payload: PayloadConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub grpc_port: u16,
    pub http_port: u16,
    pub data_dir: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexConfig {
    pub dim: usize,
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct QuantizationConfig {
    /// Enable scalar quantization (float32 -> uint8)
    pub enabled: bool,
    /// Keep original vectors for reranking
    pub keep_originals: bool,
    /// Oversample factor for reranking (e.g., 3 means fetch 3x candidates)
    pub rerank_oversample: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PayloadConfig {
    /// Enable payload indexing for fast filtering
    pub index_enabled: bool,
    /// Fields to index (e.g., ["category", "status", "type"])
    pub indexed_fields: Vec<String>,
}

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let builder = Config::builder()
            // Default settings
            .set_default("server.grpc_port", 50051)?
            .set_default("server.http_port", 3000)?
            .set_default("server.data_dir", "data")?
            .set_default("index.dim", 128)?
            .set_default("index.max_elements", 10000)?
            .set_default("index.m", 24)?
            .set_default("index.ef_construction", 400)?
            .set_default("quantization.enabled", false)?
            .set_default("quantization.keep_originals", true)?
            .set_default("quantization.rerank_oversample", 3)?
            .set_default("payload.index_enabled", true)?
            .set_default::<&str, Vec<String>>("payload.indexed_fields", vec![])?
            // Load from config file (optional)
            .add_source(File::with_name("config").required(false))
            // Load from environment variables (e.g. APP_SERVER__GRPC_PORT=50052)
            .add_source(Environment::with_prefix("APP").separator("__"));

        builder.build()?.try_deserialize()
    }
}
