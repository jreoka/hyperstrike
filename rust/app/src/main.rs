use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::serve::ListenerExt;
use match_engine::SimPool;
use server_core::AppState;
use server_core::config::Config;

fn resolve_base_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let in_cargo_target = exe_dir
            .parent()
            .and_then(|p| p.file_name())
            .is_some_and(|n| n == "target");
        if !in_cargo_target {
            return exe_dir.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() {
    let base_dir = resolve_base_dir();
    let config = Config::from_env(&base_dir);
    let port = config.port;
    let sim_workers = config.sim_workers;
    let region = config.region.clone();
    let data_dir_display = config.data_dir.display().to_string();

    let backend = Arc::new(SimPool::new(sim_workers));
    let state = AppState::new(config, backend);
    let bg_tasks = state.spawn_background_tasks();

    let app = server_core::router(Arc::clone(&state));
    let listener = match tokio::net::TcpListener::bind(SocketAddr::from(([0u16; 8], port))).await {
        Ok(l) => l,
        Err(_) => tokio::net::TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port)))
            .await
            .expect("binding the port"),
    };
    let listener = listener.tap_io(|tcp| {
        let _ = tcp.set_nodelay(true);
    });

    eprintln!(
        "[HYPERSTRIKE-RS] Listening on :{port} (region={region}, sim_workers={sim_workers}, data={data_dir_display})"
    );

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .expect("serving");

    bg_tasks.shutdown().await;
    state.flush_db_sync();
}
