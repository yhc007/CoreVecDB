use serde::Deserialize;
use config::{Config, ConfigError, File, Environment};

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub index: IndexConfig,
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
            // Load from config file (optional)
            .add_source(File::with_name("config").required(false))
            // Load from environment variables (e.g. APP_SERVER__GRPC_PORT=50052)
            .add_source(Environment::with_prefix("APP").separator("__"));

        builder.build()?.try_deserialize()
    }
}
