use anyhow::Result;
use std::sync::Arc;
use std::path::Path;
use tonic::transport::Server;
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
    println!("Metrics initialized");

    // Load configuration
    let config = AppConfig::load()?;
    println!("Configuration loaded: {:?}", config);

    let data_dir = Path::new(&config.server.data_dir);

    // Create collection manager
    let manager = Arc::new(CollectionManager::new(data_dir)?);
    println!("Collection manager initialized");
    println!("  - Collections: {:?}", manager.names());

    // gRPC Server
    let grpc_service = VectorServiceImpl::new(manager.clone());
    let grpc_addr = format!("0.0.0.0:{}", config.server.grpc_port).parse()?;
    println!("VectorDB gRPC listening on {}", grpc_addr);

    let grpc_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(VectorServiceServer::new(grpc_service))
            .serve(grpc_addr)
            .await
            .unwrap();
    });

    // HTTP Server with multi-collection support
    let http_router = vectordb::api::http::collection_router(manager.clone()).await;
    let http_addr = format!("0.0.0.0:{}", config.server.http_port);
    println!("VectorDB HTTP listening on {}", http_addr);

    let http_handle = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
        axum::serve(listener, http_router).await.unwrap();
    });

    // Wait for shutdown signal
    shutdown_signal().await;

    println!("Shutting down...");

    // Flush all collections
    if let Err(e) = manager.flush_all() {
        eprintln!("Failed to flush collections: {:?}", e);
    } else {
        println!("All collections flushed successfully.");
    }

    // Save all indexes
    if let Err(e) = manager.save_all_indexes() {
        eprintln!("Failed to save indexes: {:?}", e);
    }

    // Abort servers
    grpc_handle.abort();
    http_handle.abort();

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
