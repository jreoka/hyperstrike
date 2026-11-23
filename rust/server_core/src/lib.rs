pub mod admin;
pub mod auth;
pub mod config;
pub mod db;
pub mod http;
pub mod protocol;
pub mod ws;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::HeaderValue;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{any, get, post};
use config::Config;
use db::{Database, RateLimiter, atomic_write_async};
use protocol::MatchBackend;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use ws::RuntimeState;

#[derive(Clone)]
pub struct CachedIndexHtml {
    pub bytes: Bytes,
    pub etag: HeaderValue,
}

pub struct AppState {
    pub config: Config,
    pub started_at: Instant,
    pub cached_index_html: RwLock<Option<CachedIndexHtml>>,
    pub db: Mutex<Database>,
    pub rate_limiter: Mutex<RateLimiter>,
    pub rt: Mutex<RuntimeState>,
    pub backend: Arc<dyn MatchBackend>,
    pub metrics: admin::MetricsState,
}

impl AppState {
    pub fn new(config: Config, backend: Arc<dyn MatchBackend>) -> Arc<Self> {
        let database = Database::load(config.db_file.clone());
        let state = Arc::new(Self {
            config,
            started_at: Instant::now(),
            cached_index_html: RwLock::new(None),
            db: Mutex::new(database),
            rate_limiter: Mutex::new(RateLimiter::default()),
            rt: Mutex::new(RuntimeState::default()),
            backend,
            metrics: admin::MetricsState::default(),
        });
        state.metrics.record_sample(&state);
        state
    }

    pub fn flush_db_sync(&self) {
        let mut db = self.db.lock().unwrap();
        db.flush_sync();
    }

    pub fn spawn_background_tasks(self: &Arc<Self>) -> BackgroundTasks {
        let (shutdown_tx, mut shutdown_rx_db) = tokio::sync::watch::channel(false);
        let mut shutdown_rx_rl = shutdown_tx.subscribe();
        let mut shutdown_rx_metrics = shutdown_tx.subscribe();

        let state_db = Arc::clone(self);
        let db_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            interval.tick().await;
            loop {
                tokio::select! {
                    _ = shutdown_rx_db.changed() => {
                        if *shutdown_rx_db.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        let payload = {
                            let mut db = state_db.db.lock().unwrap();
                            db.take_flush_payload()
                        };
                        if let Some((path, json)) = payload
                            && let Err(err) = atomic_write_async(&path, &json).await {
                                eprintln!("[DB] Async flush error: {err}");
                                let mut db = state_db.db.lock().unwrap();
                                db.mark_dirty();
                            }
                    }
                }
            }
        });

        let state_rl = Arc::clone(self);
        let rl_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.tick().await;
            loop {
                tokio::select! {
                    _ = shutdown_rx_rl.changed() => {
                        if *shutdown_rx_rl.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        let mut rl = state_rl.rate_limiter.lock().unwrap();
                        rl.cleanup();
                    }
                }
            }
        });

        let state_metrics = Arc::clone(self);
        let metrics_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.tick().await;
            loop {
                tokio::select! {
                    _ = shutdown_rx_metrics.changed() => {
                        if *shutdown_rx_metrics.borrow() {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        state_metrics.metrics.record_sample(&state_metrics);
                    }
                }
            }
        });

        BackgroundTasks {
            shutdown_tx,
            db_task: Some(db_task),
            rl_task: Some(rl_task),
            metrics_task: Some(metrics_task),
        }
    }
}

pub struct BackgroundTasks {
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    db_task: Option<tokio::task::JoinHandle<()>>,
    rl_task: Option<tokio::task::JoinHandle<()>>,
    metrics_task: Option<tokio::task::JoinHandle<()>>,
}

impl BackgroundTasks {
    pub async fn shutdown(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(t) = self.db_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.rl_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.metrics_task.take() {
            let _ = t.await;
        }
    }
}

async fn count_http_requests(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    state
        .metrics
        .http_requests_total
        .fetch_add(1, Ordering::Relaxed);
    next.run(req).await
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(http::health))
        .route("/", get(http::root_get).options(http::options_handler))
        .route(
            "/index.html",
            get(http::index_html_get).options(http::options_handler),
        )
        .route("/admin", get(admin::serve_admin_page))
        .route("/admin/", get(admin::serve_admin_page))
        .route("/manifest.webmanifest", get(http::manifest_get))
        .route("/sw.js", get(http::sw_get))
        .route("/icon.svg", get(http::icon_svg_get))
        .route("/icon-192.png", get(http::icon_192_get))
        .route("/icon-512.png", get(http::icon_512_get))
        .route("/icon-maskable-512.png", get(http::icon_maskable_512_get))
        .route("/apple-touch-icon.png", get(http::apple_touch_icon_get))
        .route(
            "/api/register",
            post(http::register).options(http::options_handler),
        )
        .route(
            "/api/login",
            post(http::login).options(http::options_handler),
        )
        .route(
            "/api/logout",
            post(http::logout).options(http::options_handler),
        )
        .route(
            "/api/password",
            post(http::change_password).options(http::options_handler),
        )
        .route("/api/me", get(http::me).options(http::options_handler))
        .route(
            "/api/sessions",
            get(http::list_sessions).options(http::options_handler),
        )
        .route(
            "/api/sessions/revoke",
            post(http::revoke_session).options(http::options_handler),
        )
        .route(
            "/api/sessions/revoke-others",
            post(http::revoke_other_sessions).options(http::options_handler),
        )
        .route(
            "/api/car",
            post(http::save_car).options(http::options_handler),
        )
        .route(
            "/api/leaderboard",
            get(http::leaderboard).options(http::options_handler),
        )
        .route(
            "/api/admin/overview",
            get(admin::overview).options(http::options_handler),
        )
        .route(
            "/api/admin/users",
            get(admin::list_users).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/update",
            post(admin::update_user).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/ban",
            post(admin::ban_user).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/kick",
            post(admin::kick_user).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/revoke-sessions",
            post(admin::revoke_user_sessions).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/reset-password",
            post(admin::reset_user_password).options(http::options_handler),
        )
        .route(
            "/api/admin/users/{username}/delete",
            post(admin::delete_user).options(http::options_handler),
        )
        .route(
            "/api/admin/sessions",
            get(admin::list_all_sessions).options(http::options_handler),
        )
        .route(
            "/api/admin/sessions/revoke",
            post(admin::revoke_any_session).options(http::options_handler),
        )
        .route(
            "/api/admin/live",
            get(admin::live_state).options(http::options_handler),
        )
        .route(
            "/api/admin/lobbies/{code}/close",
            post(admin::close_lobby).options(http::options_handler),
        )
        .route(
            "/api/admin/broadcast",
            post(admin::broadcast).options(http::options_handler),
        )
        .route("/api/{*rest}", any(http::api_not_found))
        .fallback(http::fallback_handler)
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            count_http_requests,
        ))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode, header};
    use http_body_util::BodyExt;
    use protocol::NoopMatchBackend;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn test_state(tag: &str) -> Arc<AppState> {
        let dir = std::env::temp_dir().join(format!(
            "hyperstrike_server_core_test_{tag}_{}",
            auth::random_hex(6)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let index_file = dir.join("index.html");
        std::fs::write(&index_file, "<html><head></head><body>OK</body></html>").unwrap();
        let cfg = Config {
            port: 0,
            data_dir: dir.clone(),
            db_file: dir.join("db.json"),
            index_file,
            region: "test".into(),
            allowed_origin: String::new(),
            allowed_host: String::new(),
            client_secret: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            sim_workers: 1,
            trust_proxy: false,
            afk_warn_secs: 20.0,
            afk_kick_secs: 30.0,
            admin_users: vec!["jreoka".into()],
        };
        AppState::new(cfg, Arc::new(NoopMatchBackend))
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_and_full_auth_http_flow() {
        let state = test_state("auth_flow");
        let app = router(Arc::clone(&state));

        // 1. /api/health works without hs_client cookie
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let h = body_json(resp).await;
        assert_eq!(h["ok"], true);
        assert_eq!(h["region"], "test");
        assert!(h.get("uptime").is_some());
        assert_eq!(h["online"], 0);
        assert_eq!(h["lobbies"], 0);
        assert_eq!(h["workers"], 0);
        assert_eq!(h["activeMatches"], 0);

        // 2. /api/leaderboard without cookie -> 403
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/leaderboard")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        // 3. GET / issues hs_client cookie + security headers
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("host", "localhost:8080")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::X_FRAME_OPTIONS).unwrap(),
            "SAMEORIGIN"
        );
        let etag = resp
            .headers()
            .get(header::ETAG)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(etag.starts_with('"') && etag.ends_with('"'));
        let set_cookie = resp
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let cookie_pair = set_cookie.split(';').next().unwrap().to_string();

        // 3b. Conditional GET / with matching If-None-Match -> 304 Not Modified
        let resp304 = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("host", "localhost:8080")
                    .header(header::IF_NONE_MATCH, &etag)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp304.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(resp304.headers().get(header::ETAG).unwrap(), etag.as_str());
        assert!(resp304.headers().get(header::SET_COOKIE).is_some());

        // 4. Malformed JSON on /api/register -> 400 Invalid JSON
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/register")
                    .header("cookie", &cookie_pair)
                    .body(Body::from("{not valid json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(resp).await["error"], "Invalid JSON");

        // 5. Register user -> 201
        let reg_body = json!({
            "username": "AceDriver",
            "password": "password123",
            "confirmPassword": "password123"
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/register")
                    .header("cookie", &cookie_pair)
                    .body(Body::from(reg_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let reg_json = body_json(resp).await;
        let token = reg_json["token"].as_str().unwrap().to_string();
        assert_eq!(reg_json["user"]["username"], "AceDriver");
        assert_eq!(reg_json["user"]["rp"], 100);

        // 6. GET /api/me with bearer token -> 200
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/me")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let me_json = body_json(resp).await;
        assert_eq!(me_json["user"]["username"], "AceDriver");
        assert_eq!(me_json["server"]["region"], "test");

        // 7. POST /api/car -> 200 sanitized
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/car")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(json!({"car":{"body":99,"primary":3}}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let car_json = body_json(resp).await;
        assert_eq!(car_json["car"]["body"], 4);
        assert_eq!(car_json["car"]["primary"], 3);

        // 8. POST /api/login -> 200
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/login")
                    .header("cookie", &cookie_pair)
                    .body(Body::from(
                        json!({"username":"acedriver","password":"password123"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 9. POST /api/password -> 200 and invalidates other sessions
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/password")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        json!({
                            "currentPassword": "password123",
                            "newPassword": "newpassword456",
                            "confirmPassword": "newpassword456"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 10. GET /api/leaderboard -> 200
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/leaderboard?sort=rp")
                    .header("cookie", &cookie_pair)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let lb_json = body_json(resp).await;
        assert_eq!(lb_json["leaderboard"].as_array().unwrap().len(), 1);
        assert_eq!(lb_json["sort"], "rp");

        // 11. POST /api/logout -> 200, subsequent /api/me -> 401
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/logout")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/me")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Clean up temp dir
        let _ = std::fs::remove_dir_all(&state.config.data_dir);
    }

    #[tokio::test]
    async fn pwa_assets_served_with_headers_and_etag() {
        let state = test_state("pwa_assets");
        let app = router(Arc::clone(&state));

        // Manifest
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/manifest.webmanifest")
                    .header("host", "localhost:8080")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/manifest+json; charset=utf-8"
        );
        let manifest_etag = resp
            .headers()
            .get(header::ETAG)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let manifest = body_json(resp).await;
        assert_eq!(manifest["short_name"], "HYPERSTRIKE");
        assert_eq!(manifest["display"], "standalone");
        assert!(manifest["icons"].as_array().unwrap().len() >= 3);

        // Conditional GET on manifest -> 304
        let resp304 = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/manifest.webmanifest")
                    .header("host", "localhost:8080")
                    .header(header::IF_NONE_MATCH, &manifest_etag)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp304.status(), StatusCode::NOT_MODIFIED);

        // Service Worker
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/sw.js")
                    .header("host", "localhost:8080")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(resp.headers().get("service-worker-allowed").unwrap(), "/");

        // Icons
        for (uri, ct) in [
            ("/icon.svg", "image/svg+xml"),
            ("/icon-192.png", "image/png"),
            ("/icon-512.png", "image/png"),
            ("/icon-maskable-512.png", "image/png"),
            ("/apple-touch-icon.png", "image/png"),
        ] {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(uri)
                        .header("host", "localhost:8080")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "failed for {uri}");
            assert_eq!(
                resp.headers().get(header::CONTENT_TYPE).unwrap(),
                ct,
                "wrong content-type for {uri}"
            );
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            assert!(!bytes.is_empty(), "empty body for {uri}");
        }

        let _ = std::fs::remove_dir_all(&state.config.data_dir);
    }

    #[test]
    fn loads_existing_workspace_db_json() {
        let path = PathBuf::from(r"C:\Users\cross\Downloads\hyperstrike\data\db.json");
        if path.exists() {
            let db = Database::load(path);
            assert!(db.data.users.contains_key("jreoka"));
        }
    }

    #[tokio::test]
    async fn user_sessions_and_admin_panel_flow() {
        let state = test_state("admin_sessions");
        let app = router(Arc::clone(&state));

        // 1. GET /admin returns HTML + hs_client cookie
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/admin")
                    .header("host", "localhost:8080")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let set_cookie = resp
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let cookie_pair = set_cookie.split(';').next().unwrap().to_string();
        let html = String::from_utf8(
            resp.into_body().collect().await.unwrap().to_bytes().to_vec(),
        )
        .unwrap();
        assert!(html.contains("Admin Console"));

        // 2. Register admin user ("jreoka") and regular user ("player1")
        let reg_admin = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/register")
                    .header("cookie", &cookie_pair)
                    .header(
                        "user-agent",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0.0.0 Safari/537.36",
                    )
                    .body(Body::from(
                        json!({
                            "username": "jreoka",
                            "password": "password123",
                            "confirmPassword": "password123"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reg_admin.status(), StatusCode::CREATED);
        let admin_json = body_json(reg_admin).await;
        assert_eq!(admin_json["isAdmin"], true);
        let admin_tok1 = admin_json["token"].as_str().unwrap().to_string();

        // Login admin from a second device (iPhone)
        let login_admin2 = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/login")
                    .header("cookie", &cookie_pair)
                    .header(
                        "user-agent",
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
                    )
                    .body(Body::from(
                        json!({ "username": "JREOKA", "password": "password123" }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(login_admin2.status(), StatusCode::OK);
        let admin_tok2 = body_json(login_admin2).await["token"]
            .as_str()
            .unwrap()
            .to_string();

        // Register regular user
        let reg_user = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/register")
                    .header("cookie", &cookie_pair)
                    .body(Body::from(
                        json!({
                            "username": "player1",
                            "password": "password123",
                            "confirmPassword": "password123"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reg_user.status(), StatusCode::CREATED);
        let user_json = body_json(reg_user).await;
        assert_eq!(user_json["isAdmin"], false);
        let user_tok = user_json["token"].as_str().unwrap().to_string();

        // 3. Regular user cannot access /api/admin/overview -> 403
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/admin/overview")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {user_tok}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        // 4. User session listing shows 2 sessions with device metadata
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let sess_json = body_json(resp).await;
        let sessions = sess_json["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0]["isCurrent"], true);
        assert_eq!(sessions[0]["os"], "Windows");
        assert_eq!(sessions[1]["os"], "iOS");
        assert_eq!(sessions[1]["deviceType"], "Mobile");
        let second_sid = sessions[1]["id"].as_str().unwrap().to_string();

        // 5. Regular user cannot revoke admin's session
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/sessions/revoke")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {user_tok}"))
                    .body(Body::from(json!({ "sessionId": second_sid }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // 6. Admin revokes their own second session -> second token invalidated
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/sessions/revoke")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::from(json!({ "sessionId": second_sid }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/me")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok2}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // 7. Admin accesses /api/admin/overview & /api/admin/users
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/admin/overview")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ov = body_json(resp).await;
        assert_eq!(ov["counts"]["users"], 2);
        assert!(!ov["samples"].as_array().unwrap().is_empty());
        assert_eq!(ov["signupsByDay"].as_array().unwrap().len(), 14);

        // 8. Admin updates & bans player1, then deletes player1
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/admin/users/player1/update")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::from(json!({ "rp": 1250, "level": 12 }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/admin/users/player1/ban")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::from(
                        json!({ "durationMinutes": 60, "reason": "Test ban" }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Admin cannot delete themselves
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/admin/users/jreoka/delete")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Admin deletes player1
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/admin/users/player1/delete")
                    .header("cookie", &cookie_pair)
                    .header("authorization", format!("Bearer {admin_tok1}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let _ = std::fs::remove_dir_all(&state.config.data_dir);
    }
}
