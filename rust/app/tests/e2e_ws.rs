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

#[tokio::test]
async fn e2e_multi_region_cluster_sync_and_cross_region_handoff() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "hyperstrike_cluster_e2e_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let secret = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".to_string();

    // Pre-bind both listeners so each node knows the other's URL
    let listener_ord = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_ord = listener_ord.local_addr().unwrap();
    let listener_fra = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_fra = listener_fra.local_addr().unwrap();

    let mut cfg_ord = Config::from_env(&repo_root);
    cfg_ord.data_dir = tmp_dir.join("ord");
    cfg_ord.db_file = cfg_ord.data_dir.join("db.json");
    cfg_ord.region = "ord".into();
    cfg_ord.primary_region = "ord".into();
    cfg_ord.cluster_regions = vec!["ord".into(), "fra".into()];
    cfg_ord.peer_urls = std::collections::HashMap::from([("fra".into(), format!("http://{addr_fra}"))]);
    cfg_ord.client_secret = secret.clone();
    cfg_ord.allowed_origin.clear();
    cfg_ord.allowed_host.clear();

    let mut cfg_fra = Config::from_env(&repo_root);
    cfg_fra.data_dir = tmp_dir.join("fra");
    cfg_fra.db_file = cfg_fra.data_dir.join("db.json");
    cfg_fra.region = "fra".into();
    cfg_fra.primary_region = "ord".into();
    cfg_fra.cluster_regions = vec!["ord".into(), "fra".into()];
    cfg_fra.peer_urls = std::collections::HashMap::from([("ord".into(), format!("http://{addr_ord}"))]);
    cfg_fra.client_secret = secret.clone();
    cfg_fra.allowed_origin.clear();
    cfg_fra.allowed_host.clear();

    let state_ord = AppState::new(cfg_ord.clone(), Arc::new(SimPool::new(1)));
    let state_fra = AppState::new(cfg_fra.clone(), Arc::new(SimPool::new(1)));

    // Seed two users ONLY in ord's authoritative DB
    let (tok_a, tok_b) = {
        let mut db = state_ord.db.lock().unwrap();
        for (id, uname) in [("u_ord", "USPlayer"), ("u_fra", "EUPlayer")] {
            let key = uname.to_ascii_lowercase();
            db.data.users.insert(
                key,
                server_core::db::User {
                    id: id.into(),
                    username: uname.into(),
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
                },
            );
        }
        (db.create_session("usplayer"), db.create_session("euplayer"))
    };

    let bg_ord = state_ord.spawn_background_tasks();
    let bg_fra = state_fra.spawn_background_tasks();

    let app_ord = server_core::router(Arc::clone(&state_ord));
    let app_fra = server_core::router(Arc::clone(&state_fra));
    let task_ord = tokio::spawn(async move {
        let _ = axum::serve(listener_ord, app_ord.into_make_service_with_connect_info::<SocketAddr>()).await;
    });
    let task_fra = tokio::spawn(async move {
        let _ = axum::serve(listener_fra, app_fra.into_make_service_with_connect_info::<SocketAddr>()).await;
    });

    // Give background tasks a moment to start
    tokio::time::sleep(Duration::from_millis(150)).await;

    // 1. Non-primary /api/me must return fly-replay: region=ord
    let http_client = tokio::net::TcpStream::connect(addr_fra).await.unwrap();
    drop(http_client);
    let set_cookie = issue_client_cookie(&secret, "");
    let cookie_pair = set_cookie.split(';').next().unwrap().to_string();

    // Test WS upgrade with ?region=ord on fra -> must reject upgrade with HTTP 200 + fly-replay: region=ord
    let mut replay_ws_req = format!("ws://{addr_fra}/?region=ord").into_client_request().unwrap();
    replay_ws_req.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
    replay_ws_req.headers_mut().insert("Origin", format!("http://{addr_fra}").parse().unwrap());
    let replay_err = tokio_tungstenite::connect_async(replay_ws_req).await.unwrap_err();
    match replay_err {
        tokio_tungstenite::tungstenite::Error::Http(resp) => {
            assert_eq!(
                resp.headers().get("fly-replay").and_then(|v| v.to_str().ok()),
                Some("region=ord"),
                "expected fly-replay: region=ord on cross-region WS upgrade"
            );
        }
        other => panic!("expected Http fly-replay response, got {other:?}"),
    }

    // 2. Connect User A to ord, User B to fra
    let mut req_a = format!("ws://{addr_ord}/").into_client_request().unwrap();
    req_a.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
    req_a.headers_mut().insert("Origin", format!("http://{addr_ord}").parse().unwrap());
    let (mut ws_a, _) = tokio_tungstenite::connect_async(req_a).await.unwrap();

    let mut req_b = format!("ws://{addr_fra}/").into_client_request().unwrap();
    req_b.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
    req_b.headers_mut().insert("Origin", format!("http://{addr_fra}").parse().unwrap());
    let (mut ws_b, _) = tokio_tungstenite::connect_async(req_b).await.unwrap();

    ws_a.send(Message::Text(json!({ "type": "auth", "token": tok_a }).to_string().into())).await.unwrap();
    ws_b.send(Message::Text(json!({ "type": "auth", "token": tok_b }).to_string().into())).await.unwrap();

    let mut a_ok = false;
    let mut b_ok = false;
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_a.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "auth_ok" {
            assert_eq!(v["region"], "ord");
            a_ok = true;
            break;
        }
    }
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_b.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "auth_ok" {
            assert_eq!(v["region"], "fra");
            assert_eq!(v["user"]["username"], "EUPlayer");
            b_ok = true;
            break;
        }
    }
    assert!(a_ok && b_ok, "both users should authenticate across primary & replica");

    // 3. User A creates a public lobby on ord
    ws_a.send(Message::Text(json!({ "type": "lobby_create", "settings": { "name": "CROSS REGION", "isPublic": true } }).to_string().into())).await.unwrap();
    let mut lobby_code = String::new();
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_a.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "lobby_joined" {
            lobby_code = v["lobby"]["code"].as_str().unwrap().to_string();
            break;
        }
    }
    assert!(!lobby_code.is_empty());

    // Wait briefly for instant notify_changed sync
    tokio::time::sleep(Duration::from_millis(150)).await;

    // 4. User B on fra lists lobbies -> must see ord's lobby with region="ord" & global online=2
    ws_b.send(Message::Text(json!({ "type": "lobby_list" }).to_string().into())).await.unwrap();
    let mut saw_remote_lobby = false;
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_b.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "lobby_list" {
            assert_eq!(v["online"], 2, "expected global online count = 2");
            let lobbies = v["lobbies"].as_array().unwrap();
            if lobbies.iter().any(|l| l["code"] == lobby_code && l["region"] == "ord") {
                saw_remote_lobby = true;
                break;
            }
        }
    }
    assert!(saw_remote_lobby, "EU player on fra should see US player's lobby on ord");

    // 5. User B on fra attempts to join ord's lobby -> must get region_redirect to ord
    ws_b.send(Message::Text(json!({ "type": "lobby_join", "code": lobby_code }).to_string().into())).await.unwrap();
    let mut saw_redirect = false;
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_b.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "region_redirect" {
            assert_eq!(v["region"], "ord");
            assert_eq!(v["action"], "lobby_join");
            assert_eq!(v["code"], lobby_code);
            saw_redirect = true;
            break;
        }
    }
    assert!(saw_redirect, "EU player joining US lobby should receive region_redirect");

    // 6. A token revoked on the primary must be rejected on the replica, even though the replica's
    //    mirror still holds a copy of the session (it cached it on a previous successful auth).
    let tok_c = {
        let mut db = state_ord.db.lock().unwrap();
        db.create_session("euplayer")
    };
    let connect_fra = || {
        let mut req = format!("ws://{addr_fra}/").into_client_request().unwrap();
        req.headers_mut().insert("Cookie", cookie_pair.parse().unwrap());
        req.headers_mut().insert("Origin", format!("http://{addr_fra}").parse().unwrap());
        tokio_tungstenite::connect_async(req)
    };
    let (mut ws_c, _) = connect_fra().await.unwrap();
    ws_c.send(Message::Text(json!({ "type": "auth", "token": tok_c }).to_string().into())).await.unwrap();
    let mut c_ok = false;
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_c.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "auth_ok" {
            c_ok = true;
            break;
        }
    }
    assert!(c_ok, "fresh token should auth on replica via primary");
    assert!(state_fra.db.lock().unwrap().data.sessions.contains_key(&tok_c), "replica should mirror the session");
    let _ = ws_c.close(None).await;
    state_ord.db.lock().unwrap().data.sessions.shift_remove(&tok_c);

    let (mut ws_d, _) = connect_fra().await.unwrap();
    ws_d.send(Message::Text(json!({ "type": "auth", "token": tok_c }).to_string().into())).await.unwrap();
    let mut d_result = String::new();
    while let Ok(Some(Ok(Message::Text(txt)))) = tokio::time::timeout(Duration::from_secs(2), ws_d.next()).await {
        let v: Value = serde_json::from_str(&txt).unwrap();
        if v["type"] == "auth_ok" || v["type"] == "auth_error" {
            d_result = v["type"].as_str().unwrap().to_string();
            break;
        }
    }
    assert_eq!(d_result, "auth_error", "revoked token must not auth via stale replica mirror");
    assert!(!state_fra.db.lock().unwrap().data.sessions.contains_key(&tok_c), "replica should drop the revoked session");
    let _ = ws_d.close(None).await;

    // 7. Result submission is idempotent: the same result_id applied twice counts once.
    let solo_req = server_core::cluster::SoloResultRequest {
        result_id: "test-result-1".into(),
        user_id: "u_fra".into(),
        username_lower: "euplayer".into(),
        duration: 120,
        difficulty: 1,
        tournament: false,
        player_team: 0,
        sim_score: [3, 1],
        player_stats: std::collections::HashMap::new(),
    };
    let before = state_ord.db.lock().unwrap().data.users["euplayer"].stats.matches;
    let r1 = server_core::cluster::apply_solo_result_via_primary(&state_fra, &solo_req).await;
    let r2 = server_core::cluster::apply_solo_result_via_primary(&state_fra, &solo_req).await;
    assert!(r1.is_some() && r2.is_some());
    assert_eq!(r1.unwrap().reward, r2.unwrap().reward, "retry should return the cached reward");
    let after = state_ord.db.lock().unwrap().data.users["euplayer"].stats.matches;
    assert_eq!(after, before + 1, "duplicate result_id must not double-apply");

    // 8. /internal/* must reject anything that came through the public Fly edge.
    {
        let mut cfg = cfg_ord.clone();
        cfg.trust_proxy = true;
        let st = AppState::new(cfg, Arc::new(server_core::protocol::NoopMatchBackend));
        let mut h = axum::http::HeaderMap::new();
        h.insert(server_core::cluster::CLUSTER_SECRET_HEADER, secret.parse().unwrap());
        assert!(server_core::cluster::verify_cluster_secret(&st, &h));
        h.insert("fly-client-ip", "203.0.113.9".parse().unwrap());
        assert!(!server_core::cluster::verify_cluster_secret(&st, &h));
    }

    let _ = ws_a.close(None).await;
    let _ = ws_b.close(None).await;
    bg_ord.shutdown().await;
    bg_fra.shutdown().await;
    task_ord.abort();
    task_fra.abort();
    let _ = std::fs::remove_dir_all(&tmp_dir);
}
