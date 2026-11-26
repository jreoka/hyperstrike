use crate::{AppState, CachedIndexHtml};
use crate::auth::{
    get_bearer_token, has_valid_client_cookie, hash_password, issue_client_cookie, now_ms,
    random_hex, session_public_id, verify_password,
};
use crate::config::header_str;
use crate::db::{CarCustomization, User, UserStats, compute_rank, public_user, sanitize_car};
use axum::body::Bytes;
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{Extensions, HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;

const MAX_JSON_BODY_BYTES: usize = 32_768;

pub fn json_response(state: &AppState, status: StatusCode, data: &Value) -> Response {
    let body = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());
    let mut resp = (status, body).into_response();
    let headers = resp.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    if let Some(origin) = state.config.cors_origin_header()
        && let Ok(val) = HeaderValue::from_str(origin) {
            headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, val);
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                HeaderValue::from_static("true"),
            );
        }
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Authorization"),
    );
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

pub fn forbidden_origin_response() -> Response {
    let mut resp = (StatusCode::FORBIDDEN, r#"{"error":"Forbidden"}"#).into_response();
    let headers = resp.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

pub(crate) fn check_origin(state: &AppState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    if !state.config.is_request_allowed(headers, false) {
        return Err(Box::new(forbidden_origin_response()));
    }
    Ok(())
}

pub(crate) fn check_api_access(state: &AppState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    check_origin(state, headers)?;
    if !has_valid_client_cookie(&state.config.client_secret, headers) {
        return Err(Box::new(json_response(
            state,
            StatusCode::FORBIDDEN,
            &json!({ "error": "Client not recognized. Reload the game from the official site." }),
        )));
    }
    Ok(())
}

pub(crate) fn parse_json_body(state: &AppState, body: &Bytes) -> Result<Value, Box<Response>> {
    if body.len() > MAX_JSON_BODY_BYTES {
        return Err(Box::new(json_response(
            state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Payload too large" }),
        )));
    }
    if body.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice::<Value>(body).map_err(|_| {
        Box::new(json_response(
            state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Invalid JSON" }),
        ))
    })
}

pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    let sim = state.backend.stats();
    let (online, lobbies) = {
        let rt = state.rt.lock().unwrap();
        (rt.clients_by_user_id.len(), rt.lobbies.len())
    };
    let uptime = state.started_at.elapsed().as_secs();
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "ok": true,
            "region": state.config.region,
            "uptime": uptime,
            "online": online,
            "lobbies": lobbies,
            "workers": sim.ready_workers,
            "activeMatches": sim.active_matches
        }),
    )
}

pub async fn options_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_origin(&state, &headers) {
        return *resp;
    }
    let mut resp = StatusCode::NO_CONTENT.into_response();
    let h = resp.headers_mut();
    if let Some(origin) = state.config.cors_origin_header()
        && let Ok(val) = HeaderValue::from_str(origin) {
            h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, val);
            h.insert(
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                HeaderValue::from_static("true"),
            );
        }
    h.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Authorization"),
    );
    h.insert(header::VARY, HeaderValue::from_static("Origin"));
    h.insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static("86400"),
    );
    resp
}

#[derive(Debug, Deserialize)]
pub struct RootQuery {
    pub region: Option<String>,
}

pub async fn root_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
    Query(query): Query<RootQuery>,
    ws: Result<WebSocketUpgrade, axum::extract::ws::rejection::WebSocketUpgradeRejection>,
) -> Response {
    if let Ok(upgrade) = ws {
        if let Some(target) = query.region.as_deref() {
            let target_lower = target.trim().to_ascii_lowercase();
            if !target_lower.is_empty()
                && target_lower != state.config.region
                && state.config.has_region(&target_lower)
            {
                return crate::cluster::fly_replay_region(&target_lower);
            }
        }
        let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
        return crate::ws::handle_ws_upgrade(state, headers, remote, upgrade);
    }
    serve_index(&state, &headers)
}

pub async fn index_html_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_index(&state, &headers)
}

const EMBEDDED_INDEX_HTML: &str = include_str!("../../../index.html");
const EMBEDDED_MANIFEST: &[u8] = include_bytes!("../assets/manifest.webmanifest");
const EMBEDDED_SW_JS: &[u8] = include_bytes!("../assets/sw.js");
const EMBEDDED_ICON_SVG: &[u8] = include_bytes!("../assets/icon.svg");
const EMBEDDED_ICON_192: &[u8] = include_bytes!("../assets/icon-192.png");
const EMBEDDED_ICON_512: &[u8] = include_bytes!("../assets/icon-512.png");
const EMBEDDED_ICON_MASKABLE_512: &[u8] = include_bytes!("../assets/icon-maskable-512.png");
const EMBEDDED_APPLE_TOUCH_ICON: &[u8] = include_bytes!("../assets/apple-touch-icon.png");
const EMBEDDED_SAVANT_POLICY: &[u8] = include_bytes!("../../sim/assets/savant_policy.bin");

fn serve_static_asset(
    state: &AppState,
    headers: &HeaderMap,
    bytes: &'static [u8],
    content_type: &'static str,
    cache_control: &'static str,
    extra_header: Option<(header::HeaderName, &'static str)>,
) -> Response {
    if let Err(resp) = check_origin(state, headers) {
        return *resp;
    }
    let digest = Sha256::digest(bytes);
    let tag = format!("\"{}\"", hex::encode(&digest[..16]));
    let etag = HeaderValue::from_str(&tag).unwrap_or_else(|_| HeaderValue::from_static("\"0\""));

    let mut resp = if etag_matches(headers, &etag) {
        StatusCode::NOT_MODIFIED.into_response()
    } else {
        let mut r = (StatusCode::OK, Bytes::from_static(bytes)).into_response();
        r.headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
        r
    };
    let h = resp.headers_mut();
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache_control));
    h.insert(header::ETAG, etag);
    if let Some((name, value)) = extra_header {
        h.insert(name, HeaderValue::from_static(value));
    }
    resp
}

pub async fn manifest_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_MANIFEST,
        "application/manifest+json; charset=utf-8",
        "no-cache",
        None,
    )
}

pub async fn savant_policy_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_SAVANT_POLICY,
        "application/octet-stream",
        "no-cache",
        None,
    )
}

pub async fn sw_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_SW_JS,
        "application/javascript; charset=utf-8",
        "no-cache",
        Some((
            header::HeaderName::from_static("service-worker-allowed"),
            "/",
        )),
    )
}

pub async fn icon_svg_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_ICON_SVG,
        "image/svg+xml",
        "public, max-age=86400",
        None,
    )
}

pub async fn icon_192_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_ICON_192,
        "image/png",
        "public, max-age=86400",
        None,
    )
}

pub async fn icon_512_get(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_ICON_512,
        "image/png",
        "public, max-age=86400",
        None,
    )
}

pub async fn icon_maskable_512_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_ICON_MASKABLE_512,
        "image/png",
        "public, max-age=86400",
        None,
    )
}

pub async fn apple_touch_icon_get(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    serve_static_asset(
        &state,
        &headers,
        EMBEDDED_APPLE_TOUCH_ICON,
        "image/png",
        "public, max-age=86400",
        None,
    )
}

fn etag_matches(headers: &HeaderMap, etag: &HeaderValue) -> bool {
    let Ok(expected) = etag.to_str() else {
        return false;
    };
    let Some(raw) = headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    raw.split(',').any(|part| {
        let t = part.trim();
        t == "*" || t == expected || t.strip_prefix("W/") == Some(expected)
    })
}

fn serve_index(state: &AppState, headers: &HeaderMap) -> Response {
    if let Err(resp) = check_origin(state, headers) {
        return *resp;
    }
    let cached_index = {
        let cached = state.cached_index_html.read().unwrap();
        cached.clone()
    };
    let entry = match cached_index {
        Some(c) => c,
        None => {
            let mut html = std::fs::read_to_string(&state.config.index_file)
                .unwrap_or_else(|_| EMBEDDED_INDEX_HTML.to_string());
            if !state.config.allowed_origin.is_empty() {
                let origin_json =
                    serde_json::to_string(&state.config.allowed_origin).unwrap_or_default();
                let inject =
                    format!("<script>window.__HYPERSTRIKE_URL__={origin_json};</script></head>");
                html = html.replacen("</head>", &inject, 1);
            }
            let bytes = Bytes::from(html.into_bytes());
            let digest = Sha256::digest(&bytes);
            let tag = format!("\"{}\"", hex::encode(&digest[..16]));
            let etag = HeaderValue::from_str(&tag).unwrap_or_else(|_| HeaderValue::from_static("\"0\""));
            let c = CachedIndexHtml { bytes, etag };
            let mut cache = state.cached_index_html.write().unwrap();
            *cache = Some(c.clone());
            c
        }
    };

    let expected = state.config.expected_origin(headers, false);
    let cookie = issue_client_cookie(&state.config.client_secret, &expected);

    let mut resp = if etag_matches(headers, &entry.etag) {
        StatusCode::NOT_MODIFIED.into_response()
    } else {
        let mut r = (StatusCode::OK, entry.bytes).into_response();
        r.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        );
        r
    };
    let h = resp.headers_mut();
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(header::ETAG, entry.etag);
    if let Ok(val) = HeaderValue::from_str(&cookie) {
        h.insert(header::SET_COOKIE, val);
    }
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("frame-ancestors 'self'"),
    );
    h.insert(
        header::X_FRAME_OPTIONS,
        HeaderValue::from_static("SAMEORIGIN"),
    );
    resp
}

fn captcha_error_response(state: &AppState, err: crate::captcha::CaptchaError) -> Response {
    let status = match err {
        crate::captcha::CaptchaError::RateLimited(_) => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::FORBIDDEN,
    };
    let mut resp = json_response(
        state,
        status,
        &json!({ "error": err.message(), "captchaRequired": true }),
    );
    if let crate::captcha::CaptchaError::RateLimited(wait_secs) = err
        && let Ok(val) = HeaderValue::from_str(&wait_secs.max(1).to_string())
    {
        resp.headers_mut().insert(header::RETRY_AFTER, val);
    }
    resp
}

pub async fn captcha(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    if !crate::captcha::is_enabled() {
        return json_response(&state, StatusCode::OK, &json!({ "enabled": false }));
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let action = parsed
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("login");
    let signals: crate::captcha::BrowserSignals = parsed
        .get("signals")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();
    let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = state.config.get_client_ip(&headers, remote);
    match state
        .captcha
        .issue_challenge(&headers, &ip, action, &signals)
    {
        Ok(ch) => json_response(
            &state,
            StatusCode::OK,
            &serde_json::to_value(ch).unwrap_or_else(|_| json!({})),
        ),
        Err(err) => captcha_error_response(&state, err),
    }
}

pub async fn register(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = state.config.get_client_ip(&headers, remote);
    {
        let mut rl = state.rate_limiter.lock().unwrap();
        if !rl.check(&format!("auth:{ip}"), 20, 60_000) {
            return json_response(
                &state,
                StatusCode::TOO_MANY_REQUESTS,
                &json!({ "error": "Too many requests. Please wait a minute." }),
            );
        }
    }
    if let Some(wait_secs) = state.captcha.registration_throttled_secs(&ip) {
        return captcha_error_response(&state, crate::captcha::CaptchaError::RateLimited(wait_secs));
    }

    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let captcha_sol: Option<crate::captcha::CaptchaSolution> = parsed
        .get("captcha")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok());
    let hp_token = parsed
        .get("hpToken")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if let Err(err) = state.captcha.verify_solution(
        &headers,
        &ip,
        "register",
        captcha_sol.as_ref(),
        hp_token,
    ) {
        return captcha_error_response(&state, err);
    }

    let username = parsed
        .get("username")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .unwrap_or("");
    let password = parsed
        .get("password")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let confirm_password = parsed
        .get("confirmPassword")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let ulen = username.chars().count();
    if !(3..=18).contains(&ulen) {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Username must be 3 to 18 characters." }),
        );
    }
    if !username
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Username can only use letters, numbers, _ and -." }),
        );
    }
    let plen = password.chars().count();
    if !(6..=128).contains(&plen) {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Password must be at least 6 characters." }),
        );
    }
    if password != confirm_password {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Passwords do not match." }),
        );
    }

    let key = username.to_ascii_lowercase();
    {
        let db = state.db.lock().unwrap();
        if db.data.users.contains_key(&key) {
            return json_response(
                &state,
                StatusCode::CONFLICT,
                &json!({ "error": "That username is already taken." }),
            );
        }
    }

    let salt = random_hex(16);
    let pass_hash = match hash_password(password.to_string(), salt.clone()).await {
        Ok(h) => h,
        Err(e) => {
            return json_response(
                &state,
                StatusCode::BAD_REQUEST,
                &json!({ "error": e }),
            );
        }
    };

    let mut db = state.db.lock().unwrap();
    if db.data.users.contains_key(&key) {
        return json_response(
            &state,
            StatusCode::CONFLICT,
            &json!({ "error": "That username is already taken." }),
        );
    }

    let user = User {
        id: random_hex(8),
        username: username.to_string(),
        pass_hash,
        salt,
        created_at: now_ms(),
        level: 1,
        xp: 0,
        total_xp: 0,
        rp: 100,
        peak_rp: 100,
        stats: UserStats::default(),
        car: CarCustomization::default(),
        anti_cheat_flags: 0,
        last_report_at: 0,
        ac_ban_until: None,
        ac_log: Vec::new(),
    };
    let pub_u = public_user(&user);
    let is_admin = state.config.is_admin_username(&user.username);
    let ua = header_str(&headers, "user-agent");
    db.data.users.insert(key.clone(), user);
    let token = db.create_session_with_meta(&key, &ip, ua);
    db.mark_dirty();
    drop(db);
    state.captcha.record_ip_registration(&ip);
    state
        .metrics
        .registrations_total
        .fetch_add(1, Ordering::Relaxed);

    json_response(
        &state,
        StatusCode::CREATED,
        &json!({ "token": token, "user": pub_u, "isAdmin": is_admin }),
    )
}

pub async fn login(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = state.config.get_client_ip(&headers, remote);
    {
        let mut rl = state.rate_limiter.lock().unwrap();
        if !rl.check(&format!("auth:{ip}"), 25, 60_000) {
            return json_response(
                &state,
                StatusCode::TOO_MANY_REQUESTS,
                &json!({ "error": "Too many login attempts. Please wait a minute." }),
            );
        }
    }

    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let captcha_sol: Option<crate::captcha::CaptchaSolution> = parsed
        .get("captcha")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok());
    let hp_token = parsed
        .get("hpToken")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if let Err(err) = state.captcha.verify_solution(
        &headers,
        &ip,
        "login",
        captcha_sol.as_ref(),
        hp_token,
    ) {
        return captcha_error_response(&state, err);
    }

    let username = parsed
        .get("username")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .unwrap_or("");
    let password = parsed
        .get("password")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let key = username.to_ascii_lowercase();

    let creds = {
        let db = state.db.lock().unwrap();
        db.data
            .users
            .get(&key)
            .map(|u| (u.salt.clone(), u.pass_hash.clone()))
    };

    let Some((salt, expected_hash)) = creds else {
        state.captcha.record_ip_failure(&ip);
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Invalid username or password." }),
        );
    };

    if !verify_password(password.to_string(), salt, expected_hash).await {
        state.captcha.record_ip_failure(&ip);
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Invalid username or password." }),
        );
    }

    let mut db = state.db.lock().unwrap();
    let Some(pub_u) = db.data.users.get(&key).map(public_user) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Invalid username or password." }),
        );
    };
    let is_admin = state.config.is_admin_username(&pub_u.username);
    let ua = header_str(&headers, "user-agent");
    let token = db.create_session_with_meta(&key, &ip, ua);
    drop(db);
    state.metrics.logins_total.fetch_add(1, Ordering::Relaxed);
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "token": token, "user": pub_u, "isAdmin": is_admin }),
    )
}

pub async fn logout(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    if let Some(token) = get_bearer_token(&headers) {
        let removed = {
            let mut db = state.db.lock().unwrap();
            let existed = db.data.sessions.shift_remove(&token).is_some();
            if existed {
                db.mark_dirty();
            }
            existed
        };
        if removed {
            crate::ws::disconnect_session_tokens(&state, std::slice::from_ref(&token), "Signed out.");
            crate::cluster::fanout_kick_tokens(Arc::clone(&state), vec![token], "Signed out.".into());
        }
    }
    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn change_password(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };

    let (user_id, key, salt, pass_hash) = {
        let mut db = state.db.lock().unwrap();
        let Some(u) = db.get_user_by_token(&token) else {
            return json_response(
                &state,
                StatusCode::UNAUTHORIZED,
                &json!({ "error": "Not authenticated" }),
            );
        };
        (
            u.id.clone(),
            u.username.to_ascii_lowercase(),
            u.salt.clone(),
            u.pass_hash.clone(),
        )
    };

    {
        let mut rl = state.rate_limiter.lock().unwrap();
        if !rl.check(&format!("pass:{user_id}"), 10, 60_000) {
            return json_response(
                &state,
                StatusCode::TOO_MANY_REQUESTS,
                &json!({ "error": "Too many password change attempts. Please wait a minute." }),
            );
        }
    }

    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let current_password = parsed
        .get("currentPassword")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let new_password = parsed
        .get("newPassword")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let confirm_password = parsed
        .get("confirmPassword")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if current_password.is_empty()
        || !verify_password(current_password.to_string(), salt, pass_hash).await
    {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Current password is incorrect." }),
        );
    }

    let nlen = new_password.chars().count();
    if !(6..=128).contains(&nlen) {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "New password must be at least 6 characters." }),
        );
    }
    if new_password != confirm_password {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "New passwords do not match." }),
        );
    }
    if new_password == current_password {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "New password must be different from your current password." }),
        );
    }

    let new_salt = random_hex(16);
    let new_hash = match hash_password(new_password.to_string(), new_salt.clone()).await {
        Ok(h) => h,
        Err(e) => {
            return json_response(
                &state,
                StatusCode::BAD_REQUEST,
                &json!({ "error": e }),
            );
        }
    };

    let removed = {
        let mut db = state.db.lock().unwrap();
        if let Some(u) = db.data.users.get_mut(&key) {
            u.salt = new_salt;
            u.pass_hash = new_hash;
        }
        let rem = db.remove_user_sessions(&key, Some(&token));
        db.mark_dirty();
        rem
    };
    if !removed.is_empty() {
        crate::ws::disconnect_session_tokens(
            &state,
            &removed,
            "Password changed on another device.",
        );
        crate::cluster::fanout_kick_tokens(
            Arc::clone(&state),
            removed,
            "Password changed on another device.".into(),
        );
    }

    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn me(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = state.config.get_client_ip(&headers, remote);
    let ua = header_str(&headers, "user-agent");
    let pub_u = {
        let mut db = state.db.lock().unwrap();
        let u = db.get_user_by_token(&token).map(public_user);
        if u.is_some() {
            db.touch_session(&token, &ip, ua);
        }
        u
    };
    let Some(user) = pub_u else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let is_admin = state.config.is_admin_username(&user.username);
    let online = crate::cluster::global_online_count(&state);
    let lobbies = crate::cluster::global_lobby_count(&state);
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "user": user,
            "isAdmin": is_admin,
            "server": {
                "region": state.config.region,
                "online": online,
                "lobbies": lobbies
            }
        }),
    )
}

pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    extensions: Extensions,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let remote = extensions.get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let ip = state.config.get_client_ip(&headers, remote);
    let ua = header_str(&headers, "user-agent");

    let key = {
        let mut db = state.db.lock().unwrap();
        let k = db.get_username_by_token(&token);
        if k.is_some() {
            db.touch_session(&token, &ip, ua);
        }
        k
    };
    let Some(key) = key else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };

    let now = now_ms();
    let active_ws: HashSet<String> = crate::cluster::global_active_session_tokens(&state);

    let mut sessions: Vec<(bool, i64, Value)> = {
        let db = state.db.lock().unwrap();
        db.data
            .sessions
            .iter()
            .filter(|(_, s)| s.username_lower == key && s.expires_at >= now)
            .map(|(tok, s)| {
                let is_current = tok == &token;
                let last_seen = s.last_seen_at.unwrap_or(s.created_at);
                let sid = session_public_id(tok, s.id.as_deref());
                let val = json!({
                    "id": sid,
                    "isCurrent": is_current,
                    "isOnline": active_ws.contains(tok),
                    "deviceType": s.device_type.as_deref().unwrap_or("Unknown"),
                    "os": s.os.as_deref().unwrap_or("Unknown"),
                    "browser": s.browser.as_deref().unwrap_or("Unknown"),
                    "ip": s.ip.as_deref().unwrap_or("unknown"),
                    "lastIp": s.last_ip.as_deref().or(s.ip.as_deref()).unwrap_or("unknown"),
                    "userAgent": s.user_agent.as_deref().unwrap_or(""),
                    "createdAt": s.created_at,
                    "lastSeenAt": last_seen,
                    "expiresAt": s.expires_at
                });
                (is_current, last_seen, val)
            })
            .collect()
    };

    sessions.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let list: Vec<Value> = sessions.into_iter().map(|x| x.2).collect();
    json_response(&state, StatusCode::OK, &json!({ "sessions": list }))
}

pub async fn revoke_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let key = {
        let mut db = state.db.lock().unwrap();
        db.get_username_by_token(&token)
    };
    let Some(key) = key else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };

    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let session_id = parsed
        .get("sessionId")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .unwrap_or("");
    if session_id.is_empty() {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Missing sessionId" }),
        );
    }

    let removed_token = {
        let mut db = state.db.lock().unwrap();
        let Some(tok) = db.find_token_by_session_id(session_id, Some(&key)) else {
            return json_response(
                &state,
                StatusCode::NOT_FOUND,
                &json!({ "error": "Session not found" }),
            );
        };
        db.data.sessions.shift_remove(&tok);
        db.mark_dirty();
        tok
    };

    let revoked_current = removed_token == token;
    crate::ws::disconnect_session_tokens(&state, std::slice::from_ref(&removed_token), "Session revoked.");
    crate::cluster::fanout_kick_tokens(Arc::clone(&state), vec![removed_token], "Session revoked.".into());

    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "revokedCurrent": revoked_current }),
    )
}

pub async fn revoke_other_sessions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let key = {
        let mut db = state.db.lock().unwrap();
        db.get_username_by_token(&token)
    };
    let Some(key) = key else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };

    let removed = {
        let mut db = state.db.lock().unwrap();
        db.remove_user_sessions(&key, Some(&token))
    };
    let count = removed.len();
    if !removed.is_empty() {
        crate::ws::disconnect_session_tokens(
            &state,
            &removed,
            "Signed out from another device.",
        );
        crate::cluster::fanout_kick_tokens(
            Arc::clone(&state),
            removed,
            "Signed out from another device.".into(),
        );
    }
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "revokedCount": count }),
    )
}

pub async fn save_car(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let Some(token) = get_bearer_token(&headers) else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };
    let key = {
        let mut db = state.db.lock().unwrap();
        db.get_username_by_token(&token)
    };
    let Some(key) = key else {
        return json_response(
            &state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        );
    };

    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let car = sanitize_car(parsed.get("car"));
    let updated_user = {
        let mut db = state.db.lock().unwrap();
        if let Some(u) = db.data.users.get_mut(&key) {
            u.car = car.clone();
            let cloned = u.clone();
            db.mark_dirty();
            Some(cloned)
        } else {
            None
        }
    };
    if let Some(u) = updated_user {
        crate::cluster::fanout_user_sync(Arc::clone(&state), key, Some(u));
    }
    json_response(&state, StatusCode::OK, &json!({ "car": car }))
}

#[derive(Debug, Deserialize)]
pub struct LeaderboardQuery {
    pub sort: Option<String>,
}

pub async fn leaderboard(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<LeaderboardQuery>,
) -> Response {
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    let sort = if query.sort.as_deref() == Some("level") {
        "level"
    } else {
        "rp"
    };

    let mut entries: Vec<(u64, u64, i64, Value)> = {
        let db = state.db.lock().unwrap();
        db.data
            .users
            .values()
            .map(|u| {
                let r = compute_rank(u.rp);
                let level = u.level.max(1);
                let entry = json!({
                    "username": u.username,
                    "level": level,
                    "totalXp": u.total_xp,
                    "rp": u.rp,
                    "rank": r.name,
                    "rankColor": r.color,
                    "wins": u.stats.wins,
                    "goals": u.stats.goals,
                    "matches": u.stats.matches
                });
                (level, u.total_xp, u.rp, entry)
            })
            .collect()
    };

    if sort == "level" {
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    } else {
        entries.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.cmp(&a.0)));
    }
    entries.truncate(50);

    let list: Vec<Value> = entries.into_iter().map(|e| e.3).collect();
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "leaderboard": list, "sort": sort }),
    )
}

pub async fn fallback_handler(
    State(state): State<Arc<AppState>>,
    method: axum::http::Method,
    headers: HeaderMap,
) -> Response {
    if method == axum::http::Method::OPTIONS {
        return options_handler(State(state), headers).await;
    }
    if let Err(resp) = check_origin(&state, &headers) {
        return *resp;
    }
    json_response(
        &state,
        StatusCode::NOT_FOUND,
        &json!({ "error": "Not found" }),
    )
}

pub async fn api_not_found(
    State(state): State<Arc<AppState>>,
    method: axum::http::Method,
    headers: HeaderMap,
) -> Response {
    if method == axum::http::Method::OPTIONS {
        return options_handler(State(state), headers).await;
    }
    if let Err(resp) = check_api_access(&state, &headers) {
        return *resp;
    }
    json_response(
        &state,
        StatusCode::NOT_FOUND,
        &json!({ "error": "Not found" }),
    )
}
