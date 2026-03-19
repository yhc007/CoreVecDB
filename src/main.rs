use anyhow::Result;
use std::sync::Arc;
use tonic::transport::Server;
use vectordb::api::VectorServiceImpl;
use vectordb::index::{HnswIndexer, DistanceMetric};
use vectordb::storage::{MemmapVectorStore, SledMetadataStore, VectorStore};
use vectordb::proto::vectordb::vector_service_server::VectorServiceServer;
use std::path::Path;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    // Basic tracing setup
    tracing_subscriber::fmt::init();

    // Load configuration
    use vectordb::config::AppConfig;
    let config = AppConfig::load()?;
    println!("Configuration loaded: {:?}", config);

    let data_dir = &config.server.data_dir;
    fs::create_dir_all(data_dir)?;

    let vector_path = format!("{}/vectors.bin", data_dir);
    let index_path = format!("{}/index.hnsw", data_dir);
    let meta_path = format!("{}/meta.sled", data_dir);

    // 1. Storage
    let dim = config.index.dim; 
    let vector_store = Arc::new(MemmapVectorStore::new(&vector_path, dim)?);
    let metadata_store = Arc::new(SledMetadataStore::new(&meta_path)?);

    // 2. Index
    // specific to hnsw-rs: it tries to load if exists, or create new.
    // My wrapper `load` and `new` are separate.
    let graph_path = format!("{}.hnsw.graph", index_path);
    let indexer = if Path::new(&graph_path).exists() {
        println!("Loading index from {}", index_path);
        Arc::new(HnswIndexer::load(Path::new(&index_path), dim, DistanceMetric::Euclidean)?)
    } else {
        println!("Creating new index");
        Arc::new(HnswIndexer::new(dim, config.index.max_elements, config.index.m, config.index.ef_construction))
    };

    // 3. Service
    let service = VectorServiceImpl::new(
        vector_store.clone(),
        metadata_store.clone(),
        indexer.clone(),
    );

    // 4. Server
    let addr = format!("0.0.0.0:{}", config.server.grpc_port).parse()?;
    println!("VectorDB gRPC listening on {}", addr);

    let http_service = service.clone();

    // Spawn HTTP Server
    let http_router = vectordb::api::http::router(Arc::new(http_service)).await;
    let http_addr = format!("0.0.0.0:{}", config.server.http_port);
    println!("VectorDB HTTP listening on {}", http_addr);
    
    let http_handle = tokio::spawn(async move {
        // Need to parse socket addr for Axum serve, assuming 0.0.0.0
        let listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
        axum::serve(listener, http_router).await.unwrap();
    });

    Server::builder()
        .add_service(VectorServiceServer::new(service))
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;
        
    println!("Shutting down...");

    // Flush buffered writes
    if let Err(e) = vector_store.flush() {
        eprintln!("Failed to flush vector store: {:?}", e);
    } else {
        println!("Vector store flushed successfully.");
    }

    // Save index
    println!("Saving index to {}", index_path);
    if let Err(e) = indexer.save(Path::new(&index_path)) {
        eprintln!("Failed to save index: {:?}", e);
    } else {
        println!("Index saved successfully.");
    }
    
    // Abort HTTP if gRPC finishes (HTTP server runs in background)
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
