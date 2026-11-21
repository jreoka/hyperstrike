pub mod auth;
pub mod config;
pub mod db;
pub mod http;
pub mod protocol;
pub mod ws;

use axum::Router;
use axum::body::Bytes;
use axum::routing::{any, get, post};
use config::Config;
use db::{Database, RateLimiter, atomic_write_async};
use protocol::MatchBackend;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use ws::RuntimeState;

pub struct AppState {
    pub config: Config,
    pub started_at: Instant,
    pub cached_index_html: RwLock<Option<Bytes>>,
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

    pub fn spawn_background_tasks(self: &Arc<Self>) {
        let state_db = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            interval.tick().await;
            loop {
                interval.tick().await;
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
        });

        let state_rl = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.tick().await;
            loop {
                interval.tick().await;
                let mut rl = state_rl.rate_limiter.lock().unwrap();
                rl.cleanup();
            }
        });
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
        let set_cookie = resp
            .headers()
            .get(header::SET_COOKIE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let cookie_pair = set_cookie.split(';').next().unwrap().to_string();

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

    #[test]
    fn loads_existing_workspace_db_json() {
        let path = PathBuf::from(r"C:\Users\cross\Downloads\hyperstrike\data\db.json");
        if path.exists() {
            let db = Database::load(path);
            assert!(db.data.users.contains_key("jreoka"));
        }
    }
}
