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
            "cfg": {
                "teamSize": 1,
                "mode": "mayhem",
                "arena": 0,
                "layout": "default",
                "duration": 120,
                "difficulty": 1
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
                        assert_eq!(v["settings"]["mode"], "mayhem");
                        assert_eq!(v["settings"]["teamSize"], 1);
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

#[tokio::test]
async fn e2e_cross_room_kick_blocked_and_reauth_cleans_up() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "hyperstrike_e2e_sec_{}",
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

    let backend = Arc::new(SimPool::new(1));
    let state = AppState::new(config.clone(), backend);
    let (tok_a, tok_b, tok_c) = {
        let mut db = state.db.lock().unwrap();
        let make_user = |id: &str, name: &str| server_core::db::User {
            id: id.into(),
            username: name.into(),
            pass_hash: "00".repeat(64),
            salt: "testsalt12345678".into(),
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
        db.data.users.insert("alice".into(), make_user("u_a", "Alice"));
        db.data.users.insert("bob".into(), make_user("u_b", "Bob"));
        db.data.users.insert("charlie".into(), make_user("u_c", "Charlie"));
        (
            db.create_session("alice"),
            db.create_session("bob"),
            db.create_session("charlie"),
        )
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

    let set_cookie = issue_client_cookie(&config.client_secret, "");
    let cookie_pair = set_cookie.split(';').next().unwrap().to_string();
    let connect = || {
        let mut req = format!("ws://{addr}/").into_client_request().unwrap();
        req.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
        req.headers_mut()
            .insert("Origin", format!("http://{addr}").parse().unwrap());
        tokio_tungstenite::connect_async(req)
    };

    let (mut ws_a, _) = connect().await.expect("WS A");
    let (mut ws_b, _) = connect().await.expect("WS B");

    ws_a.send(Message::Text(json!({ "type": "auth", "token": tok_a }).to_string().into()))
        .await
        .unwrap();
    ws_b.send(Message::Text(json!({ "type": "auth", "token": tok_b }).to_string().into()))
        .await
        .unwrap();

    // Drain auth_ok
    let _ = tokio::time::timeout(Duration::from_secs(1), ws_a.next()).await;
    let _ = tokio::time::timeout(Duration::from_secs(1), ws_b.next()).await;

    // Both create their own squad and lobby
    ws_a.send(Message::Text(json!({ "type": "squad_create" }).to_string().into())).await.unwrap();
    ws_a.send(Message::Text(json!({ "type": "lobby_create" }).to_string().into())).await.unwrap();
    ws_b.send(Message::Text(json!({ "type": "squad_create" }).to_string().into())).await.unwrap();
    ws_b.send(Message::Text(json!({ "type": "lobby_create" }).to_string().into())).await.unwrap();

    // Drain squad_update + lobby_joined for both
    for _ in 0..2 {
        let _ = tokio::time::timeout(Duration::from_secs(1), ws_a.next()).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), ws_b.next()).await;
    }

    // Bob attempts to kick Alice from her lobby and squad
    ws_b.send(Message::Text(json!({ "type": "lobby_kick", "userId": "u_a" }).to_string().into()))
        .await
        .unwrap();
    ws_b.send(Message::Text(json!({ "type": "squad_kick", "userId": "u_a" }).to_string().into()))
        .await
        .unwrap();

    // Send a ping on ws_b to ensure prior messages were processed
    ws_b.send(Message::Text(json!({ "type": "ping", "t": 1.0 }).to_string().into()))
        .await
        .unwrap();
    let _ = tokio::time::timeout(Duration::from_secs(1), ws_b.next()).await;

    {
        let rt = state.rt.lock().unwrap();
        let a_cid = *rt.clients_by_user_id.get("u_a").expect("Alice online");
        let a_conn = rt.conns.get(&a_cid).expect("Alice conn");
        assert!(a_conn.lobby_code.is_some(), "Alice should still be in her lobby");
        assert!(a_conn.squad_code.is_some(), "Alice should still be in her squad");
    }

    // Bob re-authenticates as Charlie on the same socket
    ws_b.send(Message::Text(json!({ "type": "auth", "token": tok_c }).to_string().into()))
        .await
        .unwrap();
    let msg = tokio::time::timeout(Duration::from_secs(1), ws_b.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    if let Message::Text(txt) = msg {
        let v: Value = serde_json::from_str(&txt).unwrap();
        assert_eq!(v["type"], "auth_ok");
        assert_eq!(v["user"]["username"], "Charlie");
    } else {
        panic!("expected auth_ok text message");
    }

    {
        let rt = state.rt.lock().unwrap();
        assert!(!rt.clients_by_user_id.contains_key("u_b"), "Bob should be removed on re-auth");
        let c_cid = *rt.clients_by_user_id.get("u_c").expect("Charlie online");
        let c_conn = rt.conns.get(&c_cid).expect("Charlie conn");
        assert!(c_conn.lobby_code.is_none(), "Old lobby should be left on re-auth");
        assert!(c_conn.squad_code.is_none(), "Old squad should be left on re-auth");
        assert_eq!(rt.lobbies.len(), 1, "Only Alice's lobby should remain");
        assert_eq!(rt.squads.len(), 1, "Only Alice's squad should remain");
    }

    let _ = ws_a.close(None).await;
    let _ = ws_b.close(None).await;
    server_task.abort();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[tokio::test]
async fn e2e_multiplayer_quit_and_afk_kick_replace_players_with_bots() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "hyperstrike_e2e_afk_{}",
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
    config.afk_warn_secs = 0.25;
    config.afk_kick_secs = 0.6;

    let backend = Arc::new(SimPool::new(2));
    let state = AppState::new(config.clone(), backend);
    let (tok_a, tok_b, tok_c) = {
        let mut db = state.db.lock().unwrap();
        let make_user = |id: &str, name: &str| server_core::db::User {
            id: id.into(),
            username: name.into(),
            pass_hash: "00".repeat(64),
            salt: "testsalt12345678".into(),
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
        db.data.users.insert("driver1".into(), make_user("u_1", "Driver1"));
        db.data.users.insert("quitter".into(), make_user("u_2", "Quitter"));
        db.data.users.insert("sleeper".into(), make_user("u_3", "Sleeper"));
        (
            db.create_session("driver1"),
            db.create_session("quitter"),
            db.create_session("sleeper"),
        )
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

    let set_cookie = issue_client_cookie(&config.client_secret, "");
    let cookie_pair = set_cookie.split(';').next().unwrap().to_string();
    let connect = || {
        let mut req = format!("ws://{addr}/").into_client_request().unwrap();
        req.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
        req.headers_mut()
            .insert("Origin", format!("http://{addr}").parse().unwrap());
        tokio_tungstenite::connect_async(req)
    };

    let (mut ws_a, _) = connect().await.expect("WS A");
    let (mut ws_b, _) = connect().await.expect("WS B");
    let (mut ws_c, _) = connect().await.expect("WS C");

    for (ws, tok) in [(&mut ws_a, &tok_a), (&mut ws_b, &tok_b), (&mut ws_c, &tok_c)] {
        ws.send(Message::Text(json!({ "type": "auth", "token": tok }).to_string().into()))
            .await
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
    }

    // Driver1 creates a 2v2 lobby, Quitter & Sleeper join
    ws_a.send(Message::Text(
        json!({ "type": "lobby_create", "settings": { "teamSize": 2, "botFill": false, "arena": 0 } })
            .to_string()
            .into(),
    ))
    .await
    .unwrap();

    let mut lobby_code = String::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline && lobby_code.is_empty() {
        if let Ok(Some(Ok(Message::Text(txt)))) =
            tokio::time::timeout(Duration::from_millis(200), ws_a.next()).await
        {
            let v: Value = serde_json::from_str(&txt).unwrap();
            if v["type"] == "lobby_joined" {
                lobby_code = v["lobby"]["code"].as_str().unwrap().to_string();
            }
        }
    }
    assert!(!lobby_code.is_empty(), "expected lobby_joined with code");

    for ws in [&mut ws_b, &mut ws_c] {
        ws.send(Message::Text(
            json!({ "type": "lobby_join", "code": lobby_code }).to_string().into(),
        ))
        .await
        .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
    }

    // Start multiplayer match
    ws_a.send(Message::Text(json!({ "type": "lobby_start" }).to_string().into()))
        .await
        .unwrap();

    let mut mid: Option<u64> = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline && mid.is_none() {
        if let Ok(Some(Ok(Message::Text(txt)))) =
            tokio::time::timeout(Duration::from_millis(200), ws_a.next()).await
        {
            let v: Value = serde_json::from_str(&txt).unwrap();
            if v["type"] == "match_start" {
                mid = v["mid"].as_u64();
            }
        }
    }
    let mid = mid.expect("expected match_start with mid");

    // All three clients report loaded
    for ws in [&mut ws_a, &mut ws_b, &mut ws_c] {
        ws.send(Message::Text(
            json!({ "type": "match_loaded", "m": mid }).to_string().into(),
        ))
        .await
        .unwrap();
    }

    // Quitter quits mid-match -> should be replaced by bot immediately
    ws_b.send(Message::Text(json!({ "type": "lobby_leave" }).to_string().into()))
        .await
        .unwrap();

    // Driver1 keeps sending varying inputs while countdown (3s) finishes + Sleeper goes AFK (~0.6s)
    let mut seq = 1_i64;
    let mut saw_quitter_bot = false;
    let mut saw_sleeper_bot = false;
    let mut sleeper_saw_warn = false;
    let mut sleeper_saw_kick = false;
    let mut drive_tick = tokio::time::interval(Duration::from_millis(40));

    let end_deadline = tokio::time::Instant::now() + Duration::from_secs(6);
    while tokio::time::Instant::now() < end_deadline
        && !(saw_quitter_bot && saw_sleeper_bot && sleeper_saw_warn && sleeper_saw_kick)
    {
        tokio::select! {
            _ = drive_tick.tick() => {
                let thr = if seq % 2 == 0 { 1.0 } else { 0.3 };
                let _ = ws_a.send(Message::Text(
                    json!({ "type": "in", "m": mid, "s": seq, "c": [thr, 0.0, 0.0, 0.0, 0.0, 0] })
                        .to_string()
                        .into(),
                )).await;
                let _ = ws_c.send(Message::Text(
                    json!({ "type": "in", "m": mid, "s": seq, "c": [0.0, 0.0, 0.0, 0.0, 0.0, 0] })
                        .to_string()
                        .into(),
                )).await;
                seq += 1;
            }
            maybe_a = ws_a.next() => {
                if let Some(Ok(Message::Text(txt))) = maybe_a {
                    let v: Value = serde_json::from_str(&txt).unwrap();
                    if v["type"] == "player_replaced_by_bot" {
                        if v["userId"] == "u_2" && v["reason"] == "left" {
                            saw_quitter_bot = true;
                        }
                        if v["userId"] == "u_3" && v["reason"] == "afk" {
                            saw_sleeper_bot = true;
                        }
                    }
                }
            }
            maybe_c = ws_c.next() => {
                if let Some(Ok(Message::Text(txt))) = maybe_c {
                    let v: Value = serde_json::from_str(&txt).unwrap();
                    if v["type"] == "afk_warn" {
                        sleeper_saw_warn = true;
                    }
                    if v["type"] == "afk_kick" {
                        sleeper_saw_kick = true;
                    }
                }
            }
        }
    }

    assert!(saw_quitter_bot, "expected quitter to be replaced by a bot");
    assert!(sleeper_saw_warn, "expected AFK player to receive afk_warn");
    assert!(sleeper_saw_kick, "expected AFK player to receive afk_kick");
    assert!(saw_sleeper_bot, "expected AFK-kicked player to be replaced by a bot");

    {
        let rt = state.rt.lock().unwrap();
        let lb = rt.lobbies.get(&lobby_code).expect("lobby still active for Driver1");
        assert!(lb.players.contains_key("u_1"), "active player remains in lobby");
        assert!(!lb.players.contains_key("u_2"), "quitter removed from lobby");
        assert!(!lb.players.contains_key("u_3"), "AFK player removed from lobby");
    }

    let _ = ws_a.close(None).await;
    let _ = ws_b.close(None).await;
    let _ = ws_c.close(None).await;
    server_task.abort();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
