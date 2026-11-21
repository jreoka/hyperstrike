use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use match_engine::SimPool;
use serde_json::{Value, json};
use server_core::AppState;
use server_core::auth::issue_client_cookie;
use server_core::config::Config;
use sim::{NET_MAGIC, NET_VERSION};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

#[tokio::test]
async fn e2e_websocket_auth_and_live_solo_match_snapshots() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "hyperstrike_e2e_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut config = Config::from_env(&repo_root);
    config.data_dir = tmp_dir.clone();
    config.db_file = tmp_dir.join("db.json");
    config.client_secret = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into();
    config.allowed_origin.clear();
    config.allowed_host.clear();

    // Seed a user + session directly in the DB so we can exercise WS auth + solo match.
    let backend = Arc::new(SimPool::new(2));
    let state = AppState::new(config.clone(), backend);
    let token = {
        let mut db = state.db.lock().unwrap();
        let salt = "testsalt12345678".to_string();
        let user = server_core::db::User {
            id: "u_e2e_1".into(),
            username: "RustPilot".into(),
            pass_hash: "00".repeat(64),
            salt,
            created_at: 1_700_000_000_000,
            level: 1,
            xp: 0,
            total_xp: 0,
            rp: 100,
            peak_rp: 100,
            stats: server_core::db::UserStats::default(),
            car: server_core::db::sanitize_car(None),
            anti_cheat_flags: 0,
            last_report_at: 0,
            ac_ban_until: None,
            ac_log: Vec::new(),
        };
        db.data.users.insert("rustpilot".into(), user);
        db.create_session("rustpilot")
    };

    let app = server_core::router(Arc::clone(&state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_task = tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });

    // Build signed hs_client cookie matching server_core::auth
    let set_cookie = issue_client_cookie(&config.client_secret, "");
    let cookie_pair = set_cookie.split(';').next().unwrap().to_string();

    let mut req = format!("ws://{addr}/").into_client_request().unwrap();
    req.headers_mut()
        .insert("Cookie", cookie_pair.parse().unwrap());
    req.headers_mut()
        .insert("Origin", format!("http://{addr}").parse().unwrap());

    let (mut ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("WS connect");

    // 1. Authenticate
    ws.send(Message::Text(
        json!({ "type": "auth", "token": token }).to_string().into(),
    ))
    .await
    .unwrap();

    let mut authed = false;
    while let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_secs(2), ws.next()).await {
        if let Message::Text(txt) = msg {
            let v: Value = serde_json::from_str(&txt).unwrap();
            if v["type"] == "auth_ok" {
                assert_eq!(v["user"]["username"], "RustPilot");
                authed = true;
                break;
            }
        }
    }
    assert!(authed, "expected auth_ok");

    // 2. Start a live solo match
    ws.send(Message::Text(
        json!({
            "type": "solo_start",
            "settings": {
                "teamSize": 1,
                "mode": "mayhem",
                "arena": 0,
                "layout": "default",
                "duration": 120,
                "botFill": true,
                "botDiff": 1
            }
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    let mut saw_started = false;
    let mut saw_kickoff = false;
    let mut saw_binary_snap = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline && !(saw_started && saw_kickoff && saw_binary_snap)
    {
        match tokio::time::timeout(Duration::from_millis(500), ws.next()).await {
            Ok(Some(Ok(Message::Text(txt)))) => {
                let v: Value = serde_json::from_str(&txt).unwrap();
                match v["type"].as_str() {
                    Some("solo_match_start") => {
                        saw_started = true;
                        let mid = v["mid"].as_u64().expect("mid");
                        ws.send(Message::Text(
                            json!({ "type": "match_loaded", "m": mid })
                                .to_string()
                                .into(),
                        ))
                        .await
                        .unwrap();
                        ws.send(Message::Text(
                            json!({ "type": "in", "m": mid, "s": 1, "c": [1.0, 0.0, 0.0, 0.0, 0.0, 2] })
                                .to_string()
                                .into(),
                        ))
                        .await
                        .unwrap();
                    }
                    Some("match_kickoff") => {
                        saw_kickoff = true;
                    }
                    _ => {}
                }
            }
            Ok(Some(Ok(Message::Binary(bytes)))) => {
                assert!(bytes.len() >= 32);
                assert_eq!(bytes[0], NET_MAGIC);
                assert_eq!(bytes[1], NET_VERSION);
                saw_binary_snap = true;
            }
            _ => break,
        }
    }

    assert!(saw_started, "expected solo_match_start");
    assert!(saw_kickoff, "expected match_kickoff");
    assert!(saw_binary_snap, "expected binary snapshot frame");

    // 3. Clean leave
    ws.send(Message::Text(
        json!({ "type": "solo_leave" }).to_string().into(),
    ))
    .await
    .unwrap();
    let _ = ws.close(None).await;
    server_task.abort();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
