use crate::AppState;
use crate::auth::now_ms;
use crate::db::{DbData, Session, User, public_user};
use crate::http::json_response;
use crate::protocol::{LobbyStatus, MatchParticipant, MatchStatsEntry};
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use subtle::ConstantTimeEq;
use tokio::sync::Notify;

pub const CLUSTER_SECRET_HEADER: &str = "x-cluster-secret";
const PEER_STALE_MS: i64 = 15_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicLobbySummary {
    pub code: String,
    pub name: String,
    pub host: String,
    pub players: usize,
    pub max_players: i32,
    pub status: String,
    pub ranked: bool,
    pub mode: String,
    pub team_size: i32,
    pub arena: i32,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeerSnapshot {
    pub region: String,
    pub updated_at: i64,
    pub online_count: usize,
    pub active_matches: usize,
    pub ws_conns: usize,
    pub public_lobbies: Vec<PublicLobbySummary>,
    pub lobby_codes: Vec<String>,
    pub squad_codes: Vec<String>,
    pub active_session_tokens: Vec<String>,
    pub online_user_ids: Vec<String>,
}

pub struct ClusterState {
    pub http: reqwest::Client,
    pub peers: RwLock<HashMap<String, PeerSnapshot>>,
    pub change_notify: Notify,
}

impl Default for ClusterState {
    fn default() -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(4))
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self {
            http,
            peers: RwLock::new(HashMap::new()),
            change_notify: Notify::new(),
        }
    }
}

impl ClusterState {
    pub fn notify_changed(&self) {
        self.change_notify.notify_one();
    }

    pub fn upsert_peer(&self, mut snap: PeerSnapshot) {
        snap.updated_at = now_ms();
        let mut peers = self.peers.write().unwrap();
        peers.insert(snap.region.clone(), snap);
    }

    pub fn fresh_peers(&self) -> Vec<PeerSnapshot> {
        let now = now_ms();
        let peers = self.peers.read().unwrap();
        peers
            .values()
            .filter(|p| now - p.updated_at <= PEER_STALE_MS)
            .cloned()
            .collect()
    }
}

pub fn fly_replay_region(region: &str) -> Response {
    let mut resp = StatusCode::OK.into_response();
    if let Ok(val) = HeaderValue::from_str(&format!("region={region}")) {
        resp.headers_mut().insert("fly-replay", val);
    }
    resp
}

pub fn verify_cluster_secret(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(given) = headers
        .get(CLUSTER_SECRET_HEADER)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let expected = state.config.client_secret.as_bytes();
    let actual = given.as_bytes();
    !expected.is_empty() && expected.len() == actual.len() && bool::from(expected.ct_eq(actual))
}

pub fn build_local_snapshot(state: &AppState) -> PeerSnapshot {
    let sim = state.backend.stats();
    let rt = state.rt.lock().unwrap();
    let public_lobbies: Vec<PublicLobbySummary> = rt
        .lobbies
        .values()
        .filter(|lb| lb.settings.is_public)
        .map(|lb| {
            let host = lb
                .players
                .get(&lb.host_id)
                .map(|p| p.username.clone())
                .unwrap_or_else(|| "HOST".to_string());
            PublicLobbySummary {
                code: lb.code.clone(),
                name: lb.settings.name.clone(),
                host,
                players: lb.players.len(),
                max_players: lb.settings.team_size * 2,
                status: lb.status.as_str().to_string(),
                ranked: lb.settings.ranked,
                mode: lb.settings.mode.clone(),
                team_size: lb.settings.team_size,
                arena: lb.settings.arena,
                region: state.config.region.clone(),
            }
        })
        .collect();
    let lobby_codes: Vec<String> = rt.lobbies.keys().cloned().collect();
    let squad_codes: Vec<String> = rt.squads.keys().cloned().collect();
    let active_session_tokens: Vec<String> = rt
        .conns
        .values()
        .filter_map(|c| c.session_token.clone())
        .collect();
    let online_user_ids: Vec<String> = rt.clients_by_user_id.keys().cloned().collect();
    PeerSnapshot {
        region: state.config.region.clone(),
        updated_at: now_ms(),
        online_count: online_user_ids.len(),
        active_matches: sim.active_matches,
        ws_conns: rt.conns.len(),
        public_lobbies,
        lobby_codes,
        squad_codes,
        active_session_tokens,
        online_user_ids,
    }
}

pub fn global_online_count(state: &AppState) -> usize {
    let mut ids: HashSet<String> = {
        let rt = state.rt.lock().unwrap();
        rt.clients_by_user_id.keys().cloned().collect()
    };
    for p in state.cluster.fresh_peers() {
        for uid in p.online_user_ids {
            ids.insert(uid);
        }
    }
    ids.len()
}

pub fn global_lobby_count(state: &AppState) -> usize {
    let local = {
        let rt = state.rt.lock().unwrap();
        rt.lobbies.len()
    };
    let remote: usize = state
        .cluster
        .fresh_peers()
        .into_iter()
        .map(|p| p.lobby_codes.len())
        .sum();
    local + remote
}

pub fn global_squad_count(state: &AppState) -> usize {
    let local = {
        let rt = state.rt.lock().unwrap();
        rt.squads.len()
    };
    let remote: usize = state
        .cluster
        .fresh_peers()
        .into_iter()
        .map(|p| p.squad_codes.len())
        .sum();
    local + remote
}

pub fn global_ws_conn_count(state: &AppState) -> usize {
    let local = {
        let rt = state.rt.lock().unwrap();
        rt.conns.len()
    };
    let remote: usize = state
        .cluster
        .fresh_peers()
        .into_iter()
        .map(|p| p.ws_conns)
        .sum();
    local + remote
}

pub fn global_active_matches(state: &AppState) -> usize {
    let local = state.backend.stats().active_matches;
    let remote: usize = state
        .cluster
        .fresh_peers()
        .into_iter()
        .map(|p| p.active_matches)
        .sum();
    local + remote
}

pub fn global_active_session_tokens(state: &AppState) -> HashSet<String> {
    let mut tokens = crate::ws::active_session_tokens(state);
    for p in state.cluster.fresh_peers() {
        for tok in p.active_session_tokens {
            tokens.insert(tok);
        }
    }
    tokens
}

pub fn global_public_lobbies(state: &AppState) -> Vec<Value> {
    let mut list: Vec<Value> = {
        let rt = state.rt.lock().unwrap();
        rt.lobbies
            .values()
            .filter(|lb| lb.settings.is_public)
            .map(|lb| {
                let host = lb
                    .players
                    .get(&lb.host_id)
                    .map(|p| p.username.as_str())
                    .unwrap_or("HOST");
                json!({
                    "code": lb.code,
                    "name": lb.settings.name,
                    "host": host,
                    "players": lb.players.len(),
                    "maxPlayers": lb.settings.team_size * 2,
                    "status": lb.status.as_str(),
                    "ranked": lb.settings.ranked,
                    "mode": lb.settings.mode,
                    "teamSize": lb.settings.team_size,
                    "arena": lb.settings.arena,
                    "region": state.config.region
                })
            })
            .collect()
    };
    for p in state.cluster.fresh_peers() {
        for lb in p.public_lobbies {
            if let Ok(v) = serde_json::to_value(lb) {
                list.push(v);
            }
        }
    }
    list
}

pub async fn find_remote_code_region(state: &AppState, kind: &str, code: &str) -> Option<String> {
    if !state.config.is_clustered() {
        return None;
    }
    // 1. Check fresh peer cache first (0ms)
    for p in state.cluster.fresh_peers() {
        let found = if kind == "squad" {
            p.squad_codes.iter().any(|c| c == code)
        } else {
            p.lobby_codes.iter().any(|c| c == code)
        };
        if found {
            return Some(p.region);
        }
    }
    // 2. Live fallback query in parallel (handles codes created <2s ago)
    let mut handles = Vec::new();
    for (region, base_url) in &state.config.peer_urls {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let url = format!("{base_url}/internal/cluster/find?kind={kind}&code={code}");
        let region = region.clone();
        handles.push(tokio::spawn(async move {
            let resp = http
                .get(&url)
                .header(CLUSTER_SECRET_HEADER, secret)
                .timeout(Duration::from_millis(1200))
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let body: Value = resp.json().await.ok()?;
            if body.get("found").and_then(|v| v.as_bool()) == Some(true) {
                Some(region)
            } else {
                None
            }
        }));
    }
    for h in handles {
        if let Ok(Some(region)) = h.await {
            return Some(region);
        }
    }
    None
}

// ============================================================
// OUTBOUND SYNC & FAN-OUT HELPERS
// ============================================================

pub async fn sync_peers_once(state: &AppState) {
    if !state.config.is_clustered() {
        return;
    }
    let local_snap = build_local_snapshot(state);
    let mut handles = Vec::new();
    for (region, base_url) in &state.config.peer_urls {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let url = format!("{base_url}/internal/cluster/state");
        let payload = local_snap.clone();
        let region = region.clone();
        handles.push(tokio::spawn(async move {
            let resp = http
                .post(&url)
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&payload)
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let peer_snap: PeerSnapshot = resp.json().await.ok()?;
            Some((region, peer_snap))
        }));
    }
    for h in handles {
        if let Ok(Some((_region, snap))) = h.await {
            state.cluster.upsert_peer(snap);
        }
    }
}

pub async fn pull_db_snapshot_from_primary(state: &AppState) -> bool {
    let Some(primary_url) = state.config.primary_url() else {
        return false;
    };
    let url = format!("{primary_url}/internal/db/snapshot");
    let resp = match state
        .cluster
        .http
        .get(&url)
        .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            eprintln!("[CLUSTER] DB snapshot pull failed: HTTP {}", r.status());
            return false;
        }
        Err(err) => {
            eprintln!("[CLUSTER] DB snapshot pull error: {err}");
            return false;
        }
    };
    match resp.json::<DbData>().await {
        Ok(data) => {
            let mut db = state.db.lock().unwrap();
            db.data = data;
            true
        }
        Err(err) => {
            eprintln!("[CLUSTER] DB snapshot parse error: {err}");
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrimaryAuthResp {
    pub ok: bool,
    pub user: Option<User>,
    pub session: Option<Session>,
}

pub async fn auth_token_via_primary(
    state: &AppState,
    token: &str,
    ip: &str,
    user_agent: &str,
) -> Option<(User, Session)> {
    let primary_url = state.config.primary_url()?;
    let url = format!("{primary_url}/internal/db/auth");
    let resp = state
        .cluster
        .http
        .post(&url)
        .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
        .json(&json!({
            "token": token,
            "ip": ip,
            "userAgent": user_agent
        }))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let parsed: PrimaryAuthResp = resp.json().await.ok()?;
    if !parsed.ok {
        return None;
    }
    Some((parsed.user?, parsed.session?))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoloResultRequest {
    pub user_id: String,
    pub username_lower: String,
    pub duration: i32,
    pub difficulty: i32,
    pub tournament: bool,
    pub player_team: i32,
    pub sim_score: [i32; 2],
    pub player_stats: HashMap<String, MatchStatsEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoloResultResponse {
    pub reward: Value,
    pub user: Option<User>,
}

pub async fn apply_solo_result_via_primary(
    state: &AppState,
    req: &SoloResultRequest,
) -> Option<SoloResultResponse> {
    let primary_url = state.config.primary_url()?;
    let url = format!("{primary_url}/internal/db/apply-solo-result");
    let resp = state
        .cluster
        .http
        .post(&url)
        .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
        .json(req)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<SoloResultResponse>().await.ok()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MpResultRequest {
    pub started_at: i64,
    pub participants: Vec<MatchParticipant>,
    pub is_ranked_setting: bool,
    pub players_list: Vec<(String, String, i32)>,
    pub sim_score: [i32; 2],
    pub player_stats: HashMap<String, MatchStatsEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MpResultResponse {
    pub rewards: serde_json::Map<String, Value>,
    pub ranked_void: Option<String>,
    pub updated_users: HashMap<String, User>,
}

pub async fn apply_mp_result_via_primary(
    state: &AppState,
    req: &MpResultRequest,
) -> Option<MpResultResponse> {
    let primary_url = state.config.primary_url()?;
    let url = format!("{primary_url}/internal/db/apply-mp-result");
    let resp = state
        .cluster
        .http
        .post(&url)
        .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
        .json(req)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<MpResultResponse>().await.ok()
}

pub fn notify_primary_ac_penalize(state: Arc<AppState>, username_lower: String, reason: String) {
    let Some(primary_url) = state.config.primary_url().map(|s| s.to_string()) else {
        return;
    };
    tokio::spawn(async move {
        let url = format!("{primary_url}/internal/db/ac-penalize");
        if let Ok(resp) = state
            .cluster
            .http
            .post(&url)
            .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
            .json(&json!({ "usernameLower": username_lower, "reason": reason }))
            .send()
            .await
            && resp.status().is_success()
            && let Ok(v) = resp.json::<Value>().await
            && let Some(u_val) = v.get("user")
            && let Ok(user) = serde_json::from_value::<User>(u_val.clone())
        {
            let mut db = state.db.lock().unwrap();
            db.data.users.insert(username_lower, user);
        }
    });
}

pub fn fanout_kick_user(state: Arc<AppState>, user_id: String, reason: String) {
    if !state.config.is_clustered() {
        return;
    }
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let uid = user_id.clone();
        let reason = reason.clone();
        tokio::spawn(async move {
            let _ = http
                .post(format!("{base_url}/internal/cluster/kick-user"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "userId": uid, "reason": reason }))
                .send()
                .await;
        });
    }
}

pub async fn fanout_kick_user_await(state: &AppState, user_id: &str, reason: &str) -> bool {
    if !state.config.is_clustered() {
        return false;
    }
    let mut handles = Vec::new();
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let uid = user_id.to_string();
        let reason = reason.to_string();
        handles.push(tokio::spawn(async move {
            let resp = http
                .post(format!("{base_url}/internal/cluster/kick-user"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "userId": uid, "reason": reason }))
                .send()
                .await
                .ok()?;
            let v: Value = resp.json().await.ok()?;
            v.get("kicked").and_then(|b| b.as_bool())
        }));
    }
    let mut any_kicked = false;
    for h in handles {
        if let Ok(Some(true)) = h.await {
            any_kicked = true;
        }
    }
    any_kicked
}

pub fn fanout_kick_tokens(state: Arc<AppState>, tokens: Vec<String>, reason: String) {
    if !state.config.is_clustered() || tokens.is_empty() {
        return;
    }
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let toks = tokens.clone();
        let reason = reason.clone();
        tokio::spawn(async move {
            let _ = http
                .post(format!("{base_url}/internal/cluster/kick-tokens"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "tokens": toks, "reason": reason }))
                .send()
                .await;
        });
    }
}

pub async fn fanout_close_lobby(state: &AppState, code: &str) -> bool {
    if !state.config.is_clustered() {
        return false;
    }
    let mut handles = Vec::new();
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let code = code.to_string();
        handles.push(tokio::spawn(async move {
            let resp = http
                .post(format!("{base_url}/internal/cluster/close-lobby"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "code": code }))
                .send()
                .await
                .ok()?;
            let v: Value = resp.json().await.ok()?;
            v.get("closed").and_then(|b| b.as_bool())
        }));
    }
    let mut any = false;
    for h in handles {
        if let Ok(Some(true)) = h.await {
            any = true;
        }
    }
    any
}

pub async fn fanout_broadcast(state: &AppState, text: &str) -> usize {
    if !state.config.is_clustered() {
        return 0;
    }
    let mut handles = Vec::new();
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let text = text.to_string();
        handles.push(tokio::spawn(async move {
            let resp = http
                .post(format!("{base_url}/internal/cluster/broadcast"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "text": text }))
                .send()
                .await
                .ok()?;
            let v: Value = resp.json().await.ok()?;
            v.get("sent").and_then(|n| n.as_u64()).map(|n| n as usize)
        }));
    }
    let mut total = 0usize;
    for h in handles {
        if let Ok(Some(n)) = h.await {
            total += n;
        }
    }
    total
}

pub fn fanout_user_sync(state: Arc<AppState>, username_lower: String, user: Option<User>) {
    if !state.config.is_clustered() || !state.config.is_primary() {
        return;
    }
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        let ulower = username_lower.clone();
        let u = user.clone();
        tokio::spawn(async move {
            let _ = http
                .post(format!("{base_url}/internal/db/user-sync"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .json(&json!({ "usernameLower": ulower, "user": u }))
                .send()
                .await;
        });
    }
}

pub async fn fetch_all_peer_live(state: &AppState) -> Vec<Value> {
    if !state.config.is_clustered() {
        return Vec::new();
    }
    let mut handles = Vec::new();
    for base_url in state.config.peer_urls.values().cloned() {
        let http = state.cluster.http.clone();
        let secret = state.config.client_secret.clone();
        handles.push(tokio::spawn(async move {
            let resp = http
                .get(format!("{base_url}/internal/cluster/live"))
                .header(CLUSTER_SECRET_HEADER, secret)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            resp.json::<Value>().await.ok()
        }));
    }
    let mut out = Vec::new();
    for h in handles {
        if let Ok(Some(v)) = h.await {
            out.push(v);
        }
    }
    out
}

// ============================================================
// INTERNAL HTTP HANDLERS (/internal/*)
// ============================================================

fn unauthorized(state: &AppState) -> Response {
    json_response(
        state,
        StatusCode::UNAUTHORIZED,
        &json!({ "error": "Unauthorized" }),
    )
}

pub async fn internal_cluster_state(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    if let Ok(snap) = serde_json::from_slice::<PeerSnapshot>(&body)
        && snap.region != state.config.region
    {
        state.cluster.upsert_peer(snap);
    }
    let local = build_local_snapshot(&state);
    let val = serde_json::to_value(local).unwrap_or_else(|_| json!({}));
    json_response(&state, StatusCode::OK, &val)
}

#[derive(Debug, Deserialize)]
pub struct FindQuery {
    pub kind: Option<String>,
    pub code: Option<String>,
}

pub async fn internal_cluster_find(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<FindQuery>,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let code = query.code.unwrap_or_default().trim().to_ascii_uppercase();
    let kind = query.kind.unwrap_or_default();
    let found = {
        let rt = state.rt.lock().unwrap();
        if kind == "squad" {
            rt.squads.contains_key(&code)
        } else {
            rt.lobbies.contains_key(&code)
        }
    };
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "found": found, "region": state.config.region }),
    )
}

pub async fn internal_cluster_live(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let region = &state.config.region;
    let (lobbies, squads, connections) = {
        let rt = state.rt.lock().unwrap();
        let lobbies: Vec<Value> = rt
            .lobbies
            .values()
            .map(|lb| {
                let players: Vec<Value> = lb
                    .players
                    .values()
                    .map(|p| {
                        json!({
                            "userId": p.user_id,
                            "username": p.username,
                            "team": p.team,
                            "ready": p.ready
                        })
                    })
                    .collect();
                json!({
                    "code": lb.code,
                    "name": lb.settings.name,
                    "region": region,
                    "status": lb.status.as_str(),
                    "mode": lb.settings.mode,
                    "ranked": lb.settings.ranked,
                    "isPublic": lb.settings.is_public,
                    "teamSize": lb.settings.team_size,
                    "playerCount": lb.players.len(),
                    "players": players,
                    "matchStartedAt": lb.active_match.as_ref().map(|m| m.started_at)
                })
            })
            .collect();
        let squads: Vec<Value> = rt
            .squads
            .values()
            .map(|sq| {
                let members: Vec<&str> = sq.members.values().map(|m| m.username.as_str()).collect();
                json!({
                    "code": sq.code,
                    "region": region,
                    "memberCount": sq.members.len(),
                    "members": members
                })
            })
            .collect();
        let connections: Vec<Value> = rt
            .conns
            .values()
            .map(|c| {
                let activity = if c.solo_sim.is_some() {
                    "solo_match"
                } else if let Some(code) = &c.lobby_code {
                    if rt
                        .lobbies
                        .get(code)
                        .is_some_and(|l| l.status == LobbyStatus::InMatch)
                    {
                        "mp_match"
                    } else {
                        "lobby"
                    }
                } else if c.user_id.is_some() {
                    "menu"
                } else {
                    "unauthenticated"
                };
                json!({
                    "connId": c.conn_id,
                    "region": region,
                    "userId": c.user_id,
                    "username": c.username,
                    "ip": c.ip,
                    "connectedAt": c.connected_at,
                    "activity": activity,
                    "lobbyCode": c.lobby_code,
                    "squadCode": c.squad_code
                })
            })
            .collect();
        (lobbies, squads, connections)
    };
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "region": region,
            "lobbies": lobbies,
            "squads": squads,
            "connections": connections
        }),
    )
}

pub async fn internal_kick_user(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let uid = parsed.get("userId").and_then(|v| v.as_str()).unwrap_or("");
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Disconnected.");
    let kicked = if !uid.is_empty() {
        crate::ws::disconnect_user(&state, uid, reason)
    } else {
        false
    };
    json_response(&state, StatusCode::OK, &json!({ "kicked": kicked }))
}

pub async fn internal_kick_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let tokens: Vec<String> = parsed
        .get("tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Session revoked.");
    let count = crate::ws::disconnect_session_tokens(&state, &tokens, reason);
    json_response(&state, StatusCode::OK, &json!({ "count": count }))
}

pub async fn internal_close_lobby(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let code = parsed
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_uppercase();
    let closed = if !code.is_empty() {
        crate::ws::force_close_lobby(&state, &code)
    } else {
        false
    };
    json_response(&state, StatusCode::OK, &json!({ "closed": closed }))
}

pub async fn internal_broadcast(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let text = parsed.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let sent = if !text.is_empty() {
        crate::ws::broadcast_toast(&state, text)
    } else {
        0
    };
    json_response(&state, StatusCode::OK, &json!({ "sent": sent }))
}

pub async fn internal_db_snapshot(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let data = {
        let db = state.db.lock().unwrap();
        db.data.clone()
    };
    let val = serde_json::to_value(data).unwrap_or_else(|_| json!({}));
    json_response(&state, StatusCode::OK, &val)
}

pub async fn internal_db_auth(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let token = parsed.get("token").and_then(|v| v.as_str()).unwrap_or("");
    let ip = parsed.get("ip").and_then(|v| v.as_str()).unwrap_or("");
    let ua = parsed
        .get("userAgent")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mut db = state.db.lock().unwrap();
    let Some(key) = db.get_username_by_token(token) else {
        return json_response(&state, StatusCode::OK, &json!({ "ok": false }));
    };
    db.touch_session(token, ip, ua);
    let user = db.data.users.get(&key).cloned();
    let session = db.data.sessions.get(token).cloned();
    let ok = user.is_some() && session.is_some();
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": ok, "user": user, "session": session }),
    )
}

pub async fn internal_db_user_sync(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let ulower = parsed
        .get("usernameLower")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !ulower.is_empty() {
        let mut db = state.db.lock().unwrap();
        match parsed.get("user") {
            Some(v) if !v.is_null() => {
                if let Ok(user) = serde_json::from_value::<User>(v.clone()) {
                    db.data.users.insert(ulower, user);
                }
            }
            _ => {
                db.data.users.shift_remove(&ulower);
            }
        }
    }
    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn internal_db_apply_solo_result(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let Ok(req) = serde_json::from_slice::<SoloResultRequest>(&body) else {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Invalid payload" }),
        );
    };
    let (reward, user_after) = {
        let mut db = state.db.lock().unwrap();
        let reward = crate::ws::apply_solo_rewards_to_db(
            &mut db,
            &req.user_id,
            &req.username_lower,
            req.duration,
            req.difficulty,
            req.tournament,
            req.player_team,
            req.sim_score,
            &req.player_stats,
        );
        let u = db.data.users.get(&req.username_lower).cloned();
        (reward, u)
    };
    if let Some(ref u) = user_after {
        fanout_user_sync(Arc::clone(&state), req.username_lower.clone(), Some(u.clone()));
    }
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "reward": reward, "user": user_after }),
    )
}

pub async fn internal_db_apply_mp_result(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let Ok(req) = serde_json::from_slice::<MpResultRequest>(&body) else {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Invalid payload" }),
        );
    };
    let (rewards, ranked_void, updated_users) = {
        let mut db = state.db.lock().unwrap();
        let (rewards, ranked_void, touched) = crate::ws::apply_mp_rewards_to_db(
            &mut db,
            req.started_at,
            &req.participants,
            req.is_ranked_setting,
            &req.players_list,
            req.sim_score,
            &req.player_stats,
        );
        let mut users_map = HashMap::new();
        for key in touched {
            if let Some(u) = db.data.users.get(&key) {
                users_map.insert(key, u.clone());
            }
        }
        (rewards, ranked_void, users_map)
    };
    for (k, u) in &updated_users {
        fanout_user_sync(Arc::clone(&state), k.clone(), Some(u.clone()));
    }
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "rewards": rewards,
            "rankedVoid": ranked_void,
            "updatedUsers": updated_users
        }),
    )
}

pub async fn internal_db_ac_penalize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return unauthorized(&state);
    }
    let parsed: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let ulower = parsed
        .get("usernameLower")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Anti-cheat violation");
    let updated = {
        let mut db = state.db.lock().unwrap();
        if let Some(user) = db.data.users.get_mut(&ulower) {
            crate::protocol::ac_penalize(user, reason);
            let cloned = user.clone();
            db.mark_dirty();
            Some(cloned)
        } else {
            None
        }
    };
    if let Some(ref u) = updated {
        fanout_user_sync(Arc::clone(&state), ulower, Some(u.clone()));
    }
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "ok": updated.is_some(),
            "user": updated,
            "publicUser": updated.as_ref().map(public_user)
        }),
    )
}
