use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use axum::serve::ListenerExt;
use match_engine::SimPool;
use server_core::AppState;
use server_core::config::Config;

fn try_reload_savant_policy(path: &Path, last_mtime: &mut Option<SystemTime>) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    let Ok(mtime) = meta.modified() else {
        return;
    };
    if *last_mtime == Some(mtime) {
        return;
    }
    if let Ok(bytes) = std::fs::read(path)
        && match_engine::reload_savant_policy_bytes(&bytes)
    {
        let initial = last_mtime.is_none();
        *last_mtime = Some(mtime);
        eprintln!(
            "[SAVANT] {} live policy from {} ({} bytes)",
            if initial { "Loaded" } else { "Hot-reloaded" },
            path.display(),
            bytes.len()
        );
    }
}

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
    let savant_policy_file = config.savant_policy_file.clone();

    let mut last_policy_mtime = None;
    try_reload_savant_policy(&savant_policy_file, &mut last_policy_mtime);
    let policy_watch_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        interval.tick().await;
        loop {
            interval.tick().await;
            try_reload_savant_policy(&savant_policy_file, &mut last_policy_mtime);
        }
    });

    let model = match_engine::server_match::ac_model();
    server_core::anticheat::set_model_info(server_core::anticheat::ModelInfo {
        version: model.version.clone(),
        trained: model.is_trained(),
        human_ref: model.human_ref.clone(),
        metrics: model.metrics.clone(),
    });

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

    policy_watch_task.abort();
    bg_tasks.shutdown().await;
    state.flush_db_sync();
}
