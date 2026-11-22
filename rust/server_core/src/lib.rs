pub mod auth;
pub mod config;
pub mod db;
pub mod http;
pub mod protocol;
pub mod ws;

use axum::Router;
use axum::body::Bytes;
use axum::http::HeaderValue;
use axum::routing::{any, get, post};
use config::Config;
use db::{Database, RateLimiter, atomic_write_async};
use protocol::MatchBackend;
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
}

impl AppState {
    pub fn new(config: Config, backend: Arc<dyn MatchBackend>) -> Arc<Self> {
        let database = Database::load(config.db_file.clone());
        Arc::new(Self {
            config,
            started_at: Instant::now(),
            cached_index_html: RwLock::new(None),
            db: Mutex::new(database),
            rate_limiter: Mutex::new(RateLimiter::default()),
            rt: Mutex::new(RuntimeState::default()),
            backend,
        })
    }

    pub fn flush_db_sync(&self) {
        let mut db = self.db.lock().unwrap();
        db.flush_sync();
    }

    pub fn spawn_background_tasks(self: &Arc<Self>) -> BackgroundTasks {
        let (shutdown_tx, mut shutdown_rx_db) = tokio::sync::watch::channel(false);
        let mut shutdown_rx_rl = shutdown_tx.subscribe();

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

        BackgroundTasks {
            shutdown_tx,
            db_task: Some(db_task),
            rl_task: Some(rl_task),
        }
    }
}

pub struct BackgroundTasks {
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    db_task: Option<tokio::task::JoinHandle<()>>,
    rl_task: Option<tokio::task::JoinHandle<()>>,
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
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(http::health))
        .route("/", get(http::root_get).options(http::options_handler))
        .route(
            "/index.html",
            get(http::index_html_get).options(http::options_handler),
        )
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
            "/api/car",
            post(http::save_car).options(http::options_handler),
        )
        .route(
            "/api/leaderboard",
            get(http::leaderboard).options(http::options_handler),
        )
        .route("/api/{*rest}", any(http::api_not_found))
        .fallback(http::fallback_handler)
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
}
