use crate::AppState;
use crate::auth::{
    get_bearer_token, hash_password, issue_client_cookie, now_ms, random_hex, session_public_id,
};
use crate::db::compute_rank;
use crate::http::{check_api_access, check_origin, json_response, parse_json_body};
use crate::protocol::{ac_ban_remaining_ms, ac_log};
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const MAX_METRICS_SAMPLES: usize = 240; // 1 hour at 15s intervals
const EMBEDDED_ADMIN_HTML: &str = include_str!("../assets/admin.html");

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsSample {
    pub t: i64,
    pub online: usize,
    pub lobbies: usize,
    pub active_matches: usize,
    pub ws_conns: usize,
    pub http_req_delta: u64,
    pub mem_rss_kb: Option<u64>,
}

pub struct MetricsState {
    pub http_requests_total: AtomicU64,
    pub ws_connections_total: AtomicU64,
    pub logins_total: AtomicU64,
    pub registrations_total: AtomicU64,
    last_http_snapshot: AtomicU64,
    pub samples: Mutex<VecDeque<MetricsSample>>,
}

impl Default for MetricsState {
    fn default() -> Self {
        Self {
            http_requests_total: AtomicU64::new(0),
            ws_connections_total: AtomicU64::new(0),
            logins_total: AtomicU64::new(0),
            registrations_total: AtomicU64::new(0),
            last_http_snapshot: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::with_capacity(MAX_METRICS_SAMPLES)),
        }
    }
}

impl MetricsState {
    pub fn record_sample(&self, state: &AppState) {
        let sim = state.backend.stats();
        let (online, lobbies, ws_conns) = {
            let rt = state.rt.lock().unwrap();
            (rt.clients_by_user_id.len(), rt.lobbies.len(), rt.conns.len())
        };
        let total_http = self.http_requests_total.load(Ordering::Relaxed);
        let prev_http = self.last_http_snapshot.swap(total_http, Ordering::Relaxed);
        let http_req_delta = total_http.saturating_sub(prev_http);
        let sample = MetricsSample {
            t: now_ms(),
            online,
            lobbies,
            active_matches: sim.active_matches,
            ws_conns,
            http_req_delta,
            mem_rss_kb: read_process_rss_kb(),
        };
        let mut guard = self.samples.lock().unwrap();
        if guard.len() >= MAX_METRICS_SAMPLES {
            guard.pop_front();
        }
        guard.push_back(sample);
    }

    pub fn snapshot_samples(&self) -> Vec<MetricsSample> {
        self.samples.lock().unwrap().iter().cloned().collect()
    }
}

pub fn read_process_rss_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.split_whitespace().next()?;
                return kb_str.parse::<u64>().ok();
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

pub async fn serve_admin_page(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if let Err(resp) = check_origin(&state, &headers) {
        return *resp;
    }
    let expected = state.config.expected_origin(&headers, false);
    let cookie = issue_client_cookie(&state.config.client_secret, &expected);

    let mut resp = (StatusCode::OK, EMBEDDED_ADMIN_HTML).into_response();
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
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

pub(crate) fn require_admin(state: &AppState, headers: &HeaderMap) -> Result<(String, String), Box<Response>> {
    check_api_access(state, headers)?;
    let Some(token) = get_bearer_token(headers) else {
        return Err(Box::new(json_response(
            state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        )));
    };
    let username = {
        let mut db = state.db.lock().unwrap();
        db.get_user_by_token(&token).map(|u| u.username.clone())
    };
    let Some(username) = username else {
        return Err(Box::new(json_response(
            state,
            StatusCode::UNAUTHORIZED,
            &json!({ "error": "Not authenticated" }),
        )));
    };
    if !state.config.is_admin_username(&username) {
        return Err(Box::new(json_response(
            state,
            StatusCode::FORBIDDEN,
            &json!({ "error": "Admin privileges required" }),
        )));
    }
    Ok((token, username.to_ascii_lowercase()))
}

pub async fn overview(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }

    let now = now_ms();
    let sim = state.backend.stats();
    let uptime = state.started_at.elapsed().as_secs();
    let online = crate::cluster::global_online_count(&state);
    let lobbies = crate::cluster::global_lobby_count(&state);
    let squads = crate::cluster::global_squad_count(&state);
    let ws_conns = crate::cluster::global_ws_conn_count(&state);
    let active_matches_global = crate::cluster::global_active_matches(&state);

    let (
        total_users,
        active_sessions,
        banned_users,
        flagged_users,
        totals,
        rank_dist,
        signups_by_day,
        device_dist,
        os_dist,
        browser_dist,
        recent_ac,
        top_players,
    ) = {
        let db = state.db.lock().unwrap();
        let total_users = db.data.users.len();
        let mut banned_users = 0usize;
        let mut flagged_users = 0usize;
        let mut total_matches = 0u64;
        let mut total_mp_matches = 0u64;
        let mut total_goals = 0u64;
        let mut total_assists = 0u64;
        let mut total_saves = 0u64;
        let mut total_shots = 0u64;
        let mut total_demos = 0u64;

        let tier_order = [
            "BRONZE",
            "SILVER",
            "GOLD",
            "PLATINUM",
            "DIAMOND",
            "CHAMPION",
            "GRAND CHAMPION",
            "HYPERSTRIKE LEGEND",
        ];
        let mut rank_counts: HashMap<&str, usize> = HashMap::new();

        // 14-day signup buckets (UTC day index)
        let day_ms: i64 = 86_400_000;
        let today_day = now.div_euclid(day_ms);
        let mut signup_days: BTreeMap<i64, usize> = BTreeMap::new();
        for offset in (0..14).rev() {
            signup_days.insert(today_day - offset, 0);
        }

        let mut recent_ac: Vec<Value> = Vec::new();
        let mut players_for_top: Vec<(i64, u64, Value)> = Vec::new();

        for u in db.data.users.values() {
            if ac_ban_remaining_ms(Some(u)) > 0 {
                banned_users += 1;
            }
            if u.anti_cheat_flags > 0 {
                flagged_users += 1;
            }
            total_matches = total_matches.saturating_add(u.stats.matches);
            total_mp_matches = total_mp_matches.saturating_add(u.stats.mp_matches);
            total_goals = total_goals.saturating_add(u.stats.goals);
            total_assists = total_assists.saturating_add(u.stats.assists);
            total_saves = total_saves.saturating_add(u.stats.saves);
            total_shots = total_shots.saturating_add(u.stats.shots);
            total_demos = total_demos.saturating_add(u.stats.demos);

            let r = compute_rank(u.rp);
            for &t in &tier_order {
                if r.tier == t {
                    *rank_counts.entry(t).or_insert(0) += 1;
                    break;
                }
            }

            let u_day = u.created_at.div_euclid(day_ms);
            if let Some(slot) = signup_days.get_mut(&u_day) {
                *slot += 1;
            }

            for entry in &u.ac_log {
                recent_ac.push(json!({
                    "username": u.username,
                    "t": entry.t,
                    "reason": entry.reason,
                    "flags": u.anti_cheat_flags,
                    "banned": ac_ban_remaining_ms(Some(u)) > 0
                }));
            }

            players_for_top.push((
                u.rp,
                u.level,
                json!({
                    "username": u.username,
                    "level": u.level.max(1),
                    "rp": u.rp,
                    "rank": r.name,
                    "rankColor": r.color,
                    "matches": u.stats.matches,
                    "wins": u.stats.wins,
                    "goals": u.stats.goals
                }),
            ));
        }

        recent_ac.sort_by(|a, b| {
            b["t"]
                .as_i64()
                .unwrap_or(0)
                .cmp(&a["t"].as_i64().unwrap_or(0))
        });
        recent_ac.truncate(40);

        players_for_top.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        players_for_top.truncate(8);
        let top_players: Vec<Value> = players_for_top.into_iter().map(|x| x.2).collect();

        let rank_dist: Vec<Value> = tier_order
            .iter()
            .map(|&tier| {
                let sample_rp = match tier {
                    "BRONZE" => 0,
                    "SILVER" => 300,
                    "GOLD" => 600,
                    "PLATINUM" => 900,
                    "DIAMOND" => 1200,
                    "CHAMPION" => 1500,
                    "GRAND CHAMPION" => 1800,
                    _ => 2100,
                };
                let color = compute_rank(sample_rp).color;
                json!({
                    "tier": if tier == "HYPERSTRIKE LEGEND" { "LEGEND" } else { tier },
                    "count": rank_counts.get(tier).copied().unwrap_or(0),
                    "color": color
                })
            })
            .collect();

        let signups_by_day: Vec<Value> = signup_days
            .into_iter()
            .map(|(day, count)| {
                json!({
                    "dayTs": day * day_ms,
                    "count": count
                })
            })
            .collect();

        let mut active_sessions = 0usize;
        let mut device_counts: HashMap<String, usize> = HashMap::new();
        let mut os_counts: HashMap<String, usize> = HashMap::new();
        let mut browser_counts: HashMap<String, usize> = HashMap::new();
        for sess in db.data.sessions.values() {
            if sess.expires_at < now {
                continue;
            }
            active_sessions += 1;
            let dev = sess
                .device_type
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());
            let os = sess.os.clone().unwrap_or_else(|| "Unknown".to_string());
            let br = sess
                .browser
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());
            *device_counts.entry(dev).or_insert(0) += 1;
            *os_counts.entry(os).or_insert(0) += 1;
            *browser_counts.entry(br).or_insert(0) += 1;
        }

        let to_sorted_vec = |map: HashMap<String, usize>| -> Vec<Value> {
            let mut v: Vec<(String, usize)> = map.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            v.into_iter()
                .map(|(name, count)| json!({ "name": name, "count": count }))
                .collect()
        };

        (
            total_users,
            active_sessions,
            banned_users,
            flagged_users,
            json!({
                "matches": total_matches,
                "mpMatches": total_mp_matches,
                "goals": total_goals,
                "assists": total_assists,
                "saves": total_saves,
                "shots": total_shots,
                "demos": total_demos
            }),
            rank_dist,
            signups_by_day,
            to_sorted_vec(device_counts),
            to_sorted_vec(os_counts),
            to_sorted_vec(browser_counts),
            recent_ac,
            top_players,
        )
    };

    let samples = state.metrics.snapshot_samples();
    let mem_rss_kb = read_process_rss_kb();
    let cluster = build_cluster_overview(&state, now);

    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "server": {
                "region": state.config.region,
                "uptime": uptime,
                "online": online,
                "wsConns": ws_conns,
                "lobbies": lobbies,
                "squads": squads,
                "workers": sim.ready_workers,
                "activeMatches": active_matches_global,
                "memRssKb": mem_rss_kb,
                "adminUsers": state.config.admin_users,
                "now": now
            },
            "counters": {
                "httpRequests": state.metrics.http_requests_total.load(Ordering::Relaxed),
                "wsConnections": state.metrics.ws_connections_total.load(Ordering::Relaxed),
                "logins": state.metrics.logins_total.load(Ordering::Relaxed),
                "registrations": state.metrics.registrations_total.load(Ordering::Relaxed)
            },
            "counts": {
                "users": total_users,
                "sessions": active_sessions,
                "banned": banned_users,
                "flagged": flagged_users
            },
            "totals": totals,
            "samples": samples,
            "rankDistribution": rank_dist,
            "signupsByDay": signups_by_day,
            "devices": device_dist,
            "os": os_dist,
            "browsers": browser_dist,
            "recentAc": recent_ac,
            "topPlayers": top_players,
            "cluster": cluster
        }),
    )
}

/// Per-region view of the deployment as seen from this node: its own live numbers plus the
/// latest snapshot each peer pushed over the 2s state sync, and link health to each peer.
fn build_cluster_overview(state: &AppState, now: i64) -> Value {
    use crate::cluster::PeerSnapshot;
    const STALE_MS: i64 = 15_000;

    let cfg = &state.config;
    let local = crate::cluster::build_local_snapshot(state);
    let peers: HashMap<String, PeerSnapshot> = state
        .cluster
        .all_peers()
        .into_iter()
        .map(|p| (p.region.clone(), p))
        .collect();
    let links = state.cluster.links_snapshot();

    // Configured regions first (in config order), then anything else we've heard from.
    let mut order: Vec<String> = Vec::new();
    let mut push = |r: &str| {
        if !r.is_empty() && !order.iter().any(|x| x == r) {
            order.push(r.to_string());
        }
    };
    if !cfg.primary_region.is_empty() {
        push(&cfg.primary_region);
    }
    for r in &cfg.cluster_regions {
        push(r);
    }
    let mut extra: Vec<&String> = cfg.peer_urls.keys().chain(peers.keys()).collect();
    extra.sort();
    for r in extra {
        push(r);
    }
    push(&cfg.region);

    let regions: Vec<Value> = order
        .iter()
        .map(|region| {
            let is_self = *region == cfg.region;
            let snap = if is_self { Some(&local) } else { peers.get(region) };
            let link = links.get(region);
            let status = if is_self {
                "online"
            } else {
                match snap {
                    Some(p) if now - p.updated_at <= STALE_MS => "online",
                    Some(_) => "stale",
                    None if link.is_some() => "offline",
                    None => "unknown",
                }
            };
            let node = snap.and_then(|p| p.node.as_ref());
            json!({
                "region": region,
                "isSelf": is_self,
                "isPrimary": *region == cfg.primary_region || (cfg.primary_region.is_empty() && is_self),
                "status": status,
                "lastSeenAt": snap.map(|p| p.updated_at),
                "online": snap.map(|p| p.online_count),
                "wsConns": snap.map(|p| p.ws_conns),
                "activeMatches": snap.map(|p| p.active_matches),
                "lobbies": snap.map(|p| p.lobby_codes.len()),
                "publicLobbies": snap.map(|p| p.public_lobbies.len()),
                "squads": snap.map(|p| p.squad_codes.len()),
                "node": node,
                "link": link,
            })
        })
        .collect();

    json!({
        "clustered": cfg.is_clustered(),
        "appName": cfg.fly_app_name,
        "selfRegion": cfg.region,
        "primaryRegion": cfg.primary_region,
        "syncIntervalMs": 2000,
        "staleMs": STALE_MS,
        "regions": regions,
    })
}

#[derive(Debug, Deserialize)]
pub struct UsersQuery {
    pub q: Option<String>,
    pub sort: Option<String>,
    pub filter: Option<String>,
}

pub async fn list_users(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<UsersQuery>,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }

    let now = now_ms();
    let q = query
        .q
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let filter = query.filter.as_deref().unwrap_or("all");
    let sort = query.sort.as_deref().unwrap_or("createdAt");

    let online_map: HashMap<String, String> = {
        let rt = state.rt.lock().unwrap();
        rt.clients_by_user_id
            .iter()
            .filter_map(|(uid, cid)| {
                rt.conns
                    .get(cid)
                    .map(|c| (uid.clone(), c.ip.clone()))
            })
            .collect()
    };

    let mut list: Vec<(i64, i64, u64, u64, u32, String, Value)> = {
        let db = state.db.lock().unwrap();
        let mut session_counts: HashMap<&str, usize> = HashMap::new();
        for sess in db.data.sessions.values() {
            if sess.expires_at >= now {
                *session_counts
                    .entry(sess.username_lower.as_str())
                    .or_insert(0) += 1;
            }
        }

        db.data
            .users
            .values()
            .filter_map(|u| {
                let ulower = u.username.to_ascii_lowercase();
                if !q.is_empty() && !ulower.contains(&q) && !u.id.to_ascii_lowercase().contains(&q)
                {
                    return None;
                }
                let is_online = online_map.contains_key(&u.id);
                let ban_rem = ac_ban_remaining_ms(Some(u));
                let is_banned = ban_rem > 0;
                match filter {
                    "online" if !is_online => return None,
                    "banned" if !is_banned => return None,
                    "flagged" if u.anti_cheat_flags == 0 => return None,
                    _ => {}
                }
                let r = compute_rank(u.rp);
                let sess_count = session_counts.get(ulower.as_str()).copied().unwrap_or(0);
                let val = json!({
                    "id": u.id,
                    "username": u.username,
                    "createdAt": u.created_at,
                    "level": u.level.max(1),
                    "xp": u.xp,
                    "totalXp": u.total_xp,
                    "rp": u.rp,
                    "peakRp": u.peak_rp,
                    "rank": r.name,
                    "rankColor": r.color,
                    "stats": u.stats,
                    "antiCheatFlags": u.anti_cheat_flags,
                    "acBanUntil": u.ac_ban_until,
                    "banRemainingMs": ban_rem,
                    "acLog": u.ac_log,
                    "sessionCount": sess_count,
                    "isOnline": is_online,
                    "currentIp": online_map.get(&u.id),
                    "isAdmin": state.config.is_admin_username(&u.username)
                });
                Some((
                    u.created_at,
                    u.rp,
                    u.level.max(1),
                    u.stats.matches,
                    u.anti_cheat_flags,
                    ulower,
                    val,
                ))
            })
            .collect()
    };

    match sort {
        "rp" => list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2))),
        "level" => list.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1))),
        "matches" => list.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| b.1.cmp(&a.1))),
        "flags" => list.sort_by(|a, b| b.4.cmp(&a.4).then_with(|| b.0.cmp(&a.0))),
        "username" => list.sort_by(|a, b| a.5.cmp(&b.5)),
        _ => list.sort_by_key(|a| std::cmp::Reverse(a.0)),
    }

    let users: Vec<Value> = list.into_iter().map(|x| x.6).collect();
    json_response(&state, StatusCode::OK, &json!({ "users": users }))
}

pub async fn update_user(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
    body: Bytes,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let key = username.trim().to_ascii_lowercase();
    let mut db = state.db.lock().unwrap();
    let Some(u) = db.data.users.get_mut(&key) else {
        return json_response(
            &state,
            StatusCode::NOT_FOUND,
            &json!({ "error": "User not found" }),
        );
    };

    if let Some(rp) = parsed.get("rp").and_then(|v| v.as_i64()) {
        let clamped = rp.clamp(0, 100_000);
        u.rp = clamped;
        if clamped > u.peak_rp {
            u.peak_rp = clamped;
        }
    }
    if let Some(level) = parsed.get("level").and_then(|v| v.as_u64()) {
        u.level = level.clamp(1, 100_000);
    }
    if let Some(xp) = parsed.get("xp").and_then(|v| v.as_u64()) {
        u.xp = xp.min(100_000_000);
    }
    if let Some(flags) = parsed.get("antiCheatFlags").and_then(|v| v.as_u64()) {
        u.anti_cheat_flags = (flags as u32).min(999);
    }
    if parsed.get("clearAcLog").and_then(|v| v.as_bool()) == Some(true) {
        u.ac_log.clear();
    }
    if parsed.get("resetStats").and_then(|v| v.as_bool()) == Some(true) {
        u.stats = Default::default();
    }
    let updated_user = u.clone();
    db.mark_dirty();
    drop(db);
    crate::cluster::fanout_user_sync(Arc::clone(&state), key, Some(updated_user));

    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn ban_user(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
    body: Bytes,
) -> Response {
    let (_, admin_lower) = match require_admin(&state, &headers) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let key = username.trim().to_ascii_lowercase();
    let minutes = parsed
        .get("durationMinutes")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("Admin action");

    if minutes > 0 && key == admin_lower {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "You cannot ban yourself." }),
        );
    }

    let (target_uid, updated_user) = {
        let mut db = state.db.lock().unwrap();
        let Some(u) = db.data.users.get_mut(&key) else {
            return json_response(
                &state,
                StatusCode::NOT_FOUND,
                &json!({ "error": "User not found" }),
            );
        };
        if minutes <= 0 {
            u.ac_ban_until = None;
            ac_log(u, &format!("ADMIN UNBAN: {reason}"));
        } else {
            let ms = minutes.clamp(1, 525_600 * 10).saturating_mul(60_000);
            u.ac_ban_until = Some(now_ms().saturating_add(ms));
            ac_log(u, &format!("ADMIN BAN {minutes}m: {reason}"));
        }
        let uid = u.id.clone();
        let cloned = u.clone();
        db.mark_dirty();
        (uid, cloned)
    };
    crate::cluster::fanout_user_sync(Arc::clone(&state), key, Some(updated_user));

    if minutes > 0 {
        let ban_msg = format!("Banned by admin ({minutes}m): {reason}");
        crate::ws::disconnect_user(&state, &target_uid, &ban_msg);
        crate::cluster::fanout_kick_user(Arc::clone(&state), target_uid, ban_msg);
    }

    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "banned": minutes > 0 }),
    )
}

pub async fn kick_user(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
    body: Bytes,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("Kicked by admin.");

    let key = username.trim().to_ascii_lowercase();
    let uid = {
        let db = state.db.lock().unwrap();
        db.data.users.get(&key).map(|u| u.id.clone())
    };
    let Some(uid) = uid else {
        return json_response(
            &state,
            StatusCode::NOT_FOUND,
            &json!({ "error": "User not found" }),
        );
    };

    let kicked_local = crate::ws::disconnect_user(&state, &uid, reason);
    let kicked_remote = crate::cluster::fanout_kick_user_await(&state, &uid, reason).await;
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "kicked": kicked_local || kicked_remote }),
    )
}

pub async fn revoke_user_sessions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let key = username.trim().to_ascii_lowercase();
    let removed = {
        let mut db = state.db.lock().unwrap();
        if !db.data.users.contains_key(&key) {
            return json_response(
                &state,
                StatusCode::NOT_FOUND,
                &json!({ "error": "User not found" }),
            );
        }
        db.remove_user_sessions(&key, None)
    };
    let count = removed.len();
    if !removed.is_empty() {
        crate::ws::disconnect_session_tokens(&state, &removed, "Session revoked by admin.");
        crate::cluster::fanout_kick_tokens(
            Arc::clone(&state),
            removed,
            "Session revoked by admin.".into(),
        );
    }
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "revokedCount": count }),
    )
}

pub async fn reset_user_password(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
    body: Bytes,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let new_password = parsed
        .get("newPassword")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let nlen = new_password.chars().count();
    if !(6..=128).contains(&nlen) {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Password must be 6 to 128 characters." }),
        );
    }

    let key = username.trim().to_ascii_lowercase();
    {
        let db = state.db.lock().unwrap();
        if !db.data.users.contains_key(&key) {
            return json_response(
                &state,
                StatusCode::NOT_FOUND,
                &json!({ "error": "User not found" }),
            );
        }
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
        let rem = db.remove_user_sessions(&key, None);
        db.mark_dirty();
        rem
    };
    if !removed.is_empty() {
        crate::ws::disconnect_session_tokens(&state, &removed, "Password reset by admin.");
        crate::cluster::fanout_kick_tokens(
            Arc::clone(&state),
            removed,
            "Password reset by admin.".into(),
        );
    }

    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn delete_user(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(username): Path<String>,
) -> Response {
    let (_, admin_lower) = match require_admin(&state, &headers) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let key = username.trim().to_ascii_lowercase();
    if key == admin_lower || state.config.is_admin_username(&key) {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Cannot delete an admin account." }),
        );
    }

    let (uid, removed_tokens) = {
        let mut db = state.db.lock().unwrap();
        let Some(u) = db.data.users.shift_remove(&key) else {
            return json_response(
                &state,
                StatusCode::NOT_FOUND,
                &json!({ "error": "User not found" }),
            );
        };
        let rem = db.remove_user_sessions(&key, None);
        db.mark_dirty();
        (u.id, rem)
    };

    crate::cluster::fanout_user_sync(Arc::clone(&state), key, None);
    crate::ws::disconnect_user(&state, &uid, "Account deleted by admin.");
    crate::cluster::fanout_kick_user(Arc::clone(&state), uid, "Account deleted by admin.".into());
    if !removed_tokens.is_empty() {
        crate::ws::disconnect_session_tokens(&state, &removed_tokens, "Account deleted by admin.");
        crate::cluster::fanout_kick_tokens(
            Arc::clone(&state),
            removed_tokens,
            "Account deleted by admin.".into(),
        );
    }

    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn list_all_sessions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    let (current_token, _) = match require_admin(&state, &headers) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let now = now_ms();
    let active_ws_tokens: HashSet<String> = crate::cluster::global_active_session_tokens(&state);

    let mut sessions: Vec<(i64, Value)> = {
        let db = state.db.lock().unwrap();
        db.data
            .sessions
            .iter()
            .filter(|(_, s)| s.expires_at >= now)
            .map(|(tok, s)| {
                let username = db
                    .data
                    .users
                    .get(&s.username_lower)
                    .map(|u| u.username.clone())
                    .unwrap_or_else(|| s.username_lower.clone());
                let last_seen = s.last_seen_at.unwrap_or(s.created_at);
                let sid = session_public_id(tok, s.id.as_deref());
                let val = json!({
                    "id": sid,
                    "username": username,
                    "isCurrent": tok == &current_token,
                    "isOnline": active_ws_tokens.contains(tok),
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
                (last_seen, val)
            })
            .collect()
    };

    sessions.sort_by_key(|a| std::cmp::Reverse(a.0));
    let out: Vec<Value> = sessions.into_iter().map(|x| x.1).collect();
    json_response(&state, StatusCode::OK, &json!({ "sessions": out }))
}

pub async fn revoke_any_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let (current_token, _) = match require_admin(&state, &headers) {
        Ok(v) => v,
        Err(resp) => return *resp,
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
        let Some(tok) = db.find_token_by_session_id(session_id, None) else {
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

    let revoked_current = removed_token == current_token;
    crate::ws::disconnect_session_tokens(
        &state,
        std::slice::from_ref(&removed_token),
        "Session revoked by admin.",
    );
    crate::cluster::fanout_kick_tokens(
        Arc::clone(&state),
        vec![removed_token],
        "Session revoked by admin.".into(),
    );

    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "revokedCurrent": revoked_current }),
    )
}

pub async fn live_state(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }

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
                    "memberCount": sq.members.len(),
                    "members": members
                })
            })
            .collect();

        let mut connections: Vec<Value> = rt
            .conns
            .values()
            .map(|c| {
                let activity = if c.solo_sim.is_some() {
                    "solo_match"
                } else if let Some(code) = &c.lobby_code {
                    if rt
                        .lobbies
                        .get(code)
                        .is_some_and(|l| l.status == crate::protocol::LobbyStatus::InMatch)
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
        connections.sort_by(|a, b| {
            b["connectedAt"]
                .as_i64()
                .unwrap_or(0)
                .cmp(&a["connectedAt"].as_i64().unwrap_or(0))
        });

        (lobbies, squads, connections)
    };

    let mut lobbies = lobbies;
    let mut squads = squads;
    let mut connections = connections;
    let local_region = state.config.region.clone();
    for lb in &mut lobbies {
        if let Some(obj) = lb.as_object_mut() {
            obj.insert("region".into(), json!(local_region));
        }
    }
    for sq in &mut squads {
        if let Some(obj) = sq.as_object_mut() {
            obj.insert("region".into(), json!(local_region));
        }
    }
    for conn in &mut connections {
        if let Some(obj) = conn.as_object_mut() {
            obj.insert("region".into(), json!(local_region));
        }
    }
    for peer_live in crate::cluster::fetch_all_peer_live(&state).await {
        if let Some(arr) = peer_live.get("lobbies").and_then(|v| v.as_array()) {
            lobbies.extend(arr.iter().cloned());
        }
        if let Some(arr) = peer_live.get("squads").and_then(|v| v.as_array()) {
            squads.extend(arr.iter().cloned());
        }
        if let Some(arr) = peer_live.get("connections").and_then(|v| v.as_array()) {
            connections.extend(arr.iter().cloned());
        }
    }
    connections.sort_by(|a, b| {
        b["connectedAt"]
            .as_i64()
            .unwrap_or(0)
            .cmp(&a["connectedAt"].as_i64().unwrap_or(0))
    });

    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "lobbies": lobbies,
            "squads": squads,
            "connections": connections
        }),
    )
}

pub async fn close_lobby(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(code): Path<String>,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let clean = code.trim().to_ascii_uppercase();
    let closed_local = crate::ws::force_close_lobby(&state, &clean);
    let closed_remote = if !closed_local {
        crate::cluster::fanout_close_lobby(&state, &clean).await
    } else {
        false
    };
    if !closed_local && !closed_remote {
        return json_response(
            &state,
            StatusCode::NOT_FOUND,
            &json!({ "error": "Lobby not found" }),
        );
    }
    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

pub async fn broadcast(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(resp) = require_admin(&state, &headers) {
        return *resp;
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let text = parsed
        .get("message")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .unwrap_or("");
    if text.is_empty() || text.chars().count() > 200 {
        return json_response(
            &state,
            StatusCode::BAD_REQUEST,
            &json!({ "error": "Message must be 1 to 200 characters." }),
        );
    }
    let sent_local = crate::ws::broadcast_toast(&state, text);
    let sent_remote = crate::cluster::fanout_broadcast(&state, text).await;
    json_response(
        &state,
        StatusCode::OK,
        &json!({ "ok": true, "sent": sent_local + sent_remote }),
    )
}
