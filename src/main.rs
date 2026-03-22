use anyhow::{Context, Result};
use std::sync::Arc;
use std::path::Path;
use tonic::transport::Server;
use tracing::{error, info, warn};
use vectordb::collection::CollectionManager;
use vectordb::config::AppConfig;
use vectordb::api::VectorServiceImpl;
use vectordb::proto::vectordb::vector_service_server::VectorServiceServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Basic tracing setup
    tracing_subscriber::fmt::init();

    // Initialize metrics
    vectordb::metrics::init_metrics();
    info!("Metrics initialized");

    // Load configuration
    let config = AppConfig::load().context("Failed to load configuration")?;
    info!("Configuration loaded: {:?}", config);

    let data_dir = Path::new(&config.server.data_dir);

    // Create collection manager
    let manager = Arc::new(
        CollectionManager::new(data_dir).context("Failed to create collection manager")?
    );
    info!("Collection manager initialized");
    info!("  - Collections: {:?}", manager.names());

    // gRPC Server
    let grpc_service = VectorServiceImpl::new(manager.clone());
    let grpc_addr = format!("0.0.0.0:{}", config.server.grpc_port)
        .parse()
        .context("Invalid gRPC address")?;
    info!("VectorDB gRPC listening on {}", grpc_addr);

    let grpc_handle = tokio::spawn(async move {
        if let Err(e) = Server::builder()
            .add_service(VectorServiceServer::new(grpc_service))
            .serve(grpc_addr)
            .await
        {
            error!("gRPC server error: {:?}", e);
        }
    });

    // HTTP Server with multi-collection support
    let http_router = vectordb::api::http::collection_router(manager.clone()).await;
    let http_addr = format!("0.0.0.0:{}", config.server.http_port);
    info!("VectorDB HTTP listening on {}", http_addr);

    let http_addr_clone = http_addr.clone();
    let http_handle = tokio::spawn(async move {
        match tokio::net::TcpListener::bind(&http_addr_clone).await {
            Ok(listener) => {
                if let Err(e) = axum::serve(listener, http_router).await {
                    error!("HTTP server error: {:?}", e);
                }
            }
            Err(e) => {
                error!("Failed to bind HTTP listener on {}: {:?}", http_addr_clone, e);
            }
        }
    });

    // Wait for shutdown signal
    shutdown_signal().await;

    info!("Shutting down...");

    // Flush all collections
    match manager.flush_all() {
        Ok(_) => info!("All collections flushed successfully."),
        Err(e) => error!("Failed to flush collections: {:?}", e),
    }

    // Save all indexes
    if let Err(e) = manager.save_all_indexes() {
        error!("Failed to save indexes: {:?}", e);
    } else {
        info!("All indexes saved successfully.");
    }

    // Abort servers gracefully
    grpc_handle.abort();
    http_handle.abort();

    info!("Shutdown complete.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        match tokio::signal::ctrl_c().await {
            Ok(_) => info!("Received Ctrl+C signal"),
            Err(e) => warn!("Failed to listen for Ctrl+C: {:?}", e),
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
                info!("Received terminate signal");
            }
            Err(e) => {
                warn!("Failed to install SIGTERM handler: {:?}", e);
                // Fall back to pending future so ctrl_c can still work
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
