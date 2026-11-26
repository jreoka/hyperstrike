use crate::AppState;
use crate::auth::{has_valid_client_cookie, issue_client_cookie, now_ms, random_code};
use crate::db::{Database, add_xp, compute_rank, public_user};
use crate::protocol::{
    AC_INPUT_MAX_PER_SEC, AC_INPUT_STRIKE_AT, AC_KICK_STRIKES, AC_WARN_STRIKES, ARENA_COUNT,
    ActiveMatchMeta, Lobby, LobbyPlayer, LobbySettings, LobbyStatus, MatchCreateOptions,
    MatchEventSink, MatchHandle, MatchParticipant, MatchPlayerSpec, MatchStatsEntry,
    RankedEligibility, SNAP_BACKLOG_MAX, STANDARD_ARENA_COUNT, SnapshotPayload, SoloMatchMeta,
    Squad, SquadMember, ac_ban_message, ac_ban_remaining_ms, ac_penalize, decay_strikes,
    default_lobby_settings, is_valid_layout, is_valid_mode, json_strict_int, make_ac_rec,
    packet_rate_ok, ranked_eligibility_from_participants, resolve_arena, sanitize_lobby_settings,
};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use indexmap::IndexMap;
use rand::Rng;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;

pub enum WsOut {
    Text(String),
    Binary(Vec<u8>),
    Ping,
    /// Ping carrying a send timestamp; the browser echoes it in a Pong (RTT measurement)
    PingData(Vec<u8>),
    Close,
}

/// Interval of timestamped WebSocket pings used to measure network RTT for anti-cheat.
const RTT_PROBE_SECS: u64 = 3;

#[derive(Clone)]
pub struct ClientHandle {
    pub conn_id: u64,
    pub ip: String,
    pub user_agent: String,
    pub connected_at: i64,
    pub session_token: Option<String>,
    pub tx: mpsc::UnboundedSender<WsOut>,
    pub backlog_bytes: Arc<AtomicUsize>,
    pub user_id: Option<String>,
    pub username: Option<String>,
    pub username_lower: Option<String>,
    pub squad_code: Option<String>,
    pub lobby_code: Option<String>,
    pub solo_sim: Option<Arc<dyn MatchHandle>>,
    pub solo_match: Option<SoloMatchMeta>,
    pub solo_ac: Option<crate::protocol::AcRecord>,
    pub solo_gen: u64,
    /// Smoothed WebSocket round trip in ms (f64 bits; 0 = unknown)
    pub rtt_bits: Arc<std::sync::atomic::AtomicU64>,
}

impl ClientHandle {
    pub fn rtt_ms(&self) -> Option<f64> {
        let v = f64::from_bits(self.rtt_bits.load(Ordering::Relaxed));
        (v > 0.0).then_some(v)
    }
}

impl ClientHandle {
    pub fn send_json(&self, obj: &Value) {
        if let Ok(text) = serde_json::to_string(obj) {
            let len = text.len();
            self.backlog_bytes.fetch_add(len, Ordering::Relaxed);
            if self.tx.send(WsOut::Text(text)).is_err() {
                self.backlog_bytes.fetch_sub(len, Ordering::Relaxed);
            }
        }
    }

    pub fn send_snapshot_if_unblocked(&self, payload: SnapshotPayload) {
        if self.backlog_bytes.load(Ordering::Relaxed) >= SNAP_BACKLOG_MAX {
            return;
        }
        let len = payload.byte_len();
        self.backlog_bytes.fetch_add(len, Ordering::Relaxed);
        let msg = match payload {
            SnapshotPayload::Binary(b) => WsOut::Binary(b),
            SnapshotPayload::Text(s) => WsOut::Text(s),
        };
        if self.tx.send(msg).is_err() {
            self.backlog_bytes.fetch_sub(len, Ordering::Relaxed);
        }
    }

    pub fn close(&self) {
        let _ = self.tx.send(WsOut::Close);
    }
}

pub struct RuntimeState {
    pub next_conn_id: u64,
    pub conns: HashMap<u64, ClientHandle>,
    pub clients_by_user_id: HashMap<String, u64>,
    pub squads: IndexMap<String, Squad>,
    pub lobbies: IndexMap<String, Lobby>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            next_conn_id: 1,
            conns: HashMap::new(),
            clients_by_user_id: HashMap::new(),
            squads: IndexMap::new(),
            lobbies: IndexMap::new(),
        }
    }
}

pub fn disconnect_user(state: &AppState, user_id: &str, reason: &str) -> bool {
    disconnect_user_except(state, user_id, reason, None)
}

/// Like `disconnect_user`, but leaves the connection alone if it authenticated with `except_token`.
pub fn disconnect_user_except(
    state: &AppState,
    user_id: &str,
    reason: &str,
    except_token: Option<&str>,
) -> bool {
    let mut rt = state.rt.lock().unwrap();
    let db = state.db.lock().unwrap();
    let Some(&cid) = rt.clients_by_user_id.get(user_id) else {
        return false;
    };
    if let Some(except) = except_token.filter(|t| !t.is_empty())
        && rt
            .conns
            .get(&cid)
            .and_then(|c| c.session_token.as_deref())
            == Some(except)
    {
        return false;
    }
    send_to_conn(&rt, cid, &json!({ "type": "kicked", "reason": reason }));
    stop_solo_match(&mut rt, cid);
    leave_lobby(&mut rt, &db, cid, "kicked");
    leave_squad(&mut rt, &db, cid);
    rt.clients_by_user_id.remove(user_id);
    if let Some(c) = rt.conns.get(&cid) {
        c.close();
    }
    true
}

pub fn disconnect_session_tokens(state: &AppState, tokens: &[String], reason: &str) -> usize {
    if tokens.is_empty() {
        return 0;
    }
    let token_set: HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
    let mut rt = state.rt.lock().unwrap();
    let db = state.db.lock().unwrap();
    let targets: Vec<(u64, Option<String>)> = rt
        .conns
        .values()
        .filter(|c| {
            c.session_token
                .as_deref()
                .is_some_and(|t| token_set.contains(t))
        })
        .map(|c| (c.conn_id, c.user_id.clone()))
        .collect();
    let count = targets.len();
    for (cid, uid) in targets {
        send_to_conn(&rt, cid, &json!({ "type": "kicked", "reason": reason }));
        stop_solo_match(&mut rt, cid);
        leave_lobby(&mut rt, &db, cid, "kicked");
        leave_squad(&mut rt, &db, cid);
        if let Some(u) = uid
            && rt.clients_by_user_id.get(&u).copied() == Some(cid)
        {
            rt.clients_by_user_id.remove(&u);
        }
        if let Some(c) = rt.conns.get(&cid) {
            c.close();
        }
    }
    count
}

pub fn active_session_tokens(state: &AppState) -> HashSet<String> {
    let rt = state.rt.lock().unwrap();
    rt.conns
        .values()
        .filter_map(|c| c.session_token.clone())
        .collect()
}

pub fn force_close_lobby(state: &AppState, code: &str) -> bool {
    let mut rt = state.rt.lock().unwrap();
    let Some(mut lb) = rt.lobbies.shift_remove(code) else {
        return false;
    };
    if let Some(sim) = lb.sim.take() {
        sim.stop();
    }
    let player_ids: Vec<String> = lb.players.keys().cloned().collect();
    for uid in player_ids {
        if let Some(&cid) = rt.clients_by_user_id.get(&uid) {
            if let Some(c) = rt.conns.get_mut(&cid)
                && c.lobby_code.as_deref() == Some(code)
            {
                c.lobby_code = None;
            }
            send_to_conn(
                &rt,
                cid,
                &json!({ "type": "lobby_left", "reason": "Lobby closed by admin." }),
            );
            send_to_conn(
                &rt,
                cid,
                &json!({ "type": "toast", "text": "LOBBY CLOSED BY ADMIN" }),
            );
        }
    }
    true
}

pub fn broadcast_toast(state: &AppState, text: &str) -> usize {
    let rt = state.rt.lock().unwrap();
    let msg = json!({ "type": "toast", "text": text });
    let mut count = 0;
    for c in rt.conns.values() {
        if c.user_id.is_some() {
            c.send_json(&msg);
            count += 1;
        }
    }
    count
}

pub fn handle_ws_upgrade(
    state: Arc<AppState>,
    headers: HeaderMap,
    remote_addr: Option<SocketAddr>,
    ws: WebSocketUpgrade,
) -> Response {
    if !state.config.is_request_allowed(&headers, false)
        || !has_valid_client_cookie(&state.config.client_secret, &headers)
    {
        return (StatusCode::FORBIDDEN, "Forbidden").into_response();
    }

    let ip = state.config.get_client_ip(&headers, remote_addr);
    let user_agent = crate::config::header_str(&headers, "user-agent").to_string();
    let expected = state.config.expected_origin(&headers, false);
    let cookie = issue_client_cookie(&state.config.client_secret, &expected);

    let mut resp = ws
        .max_message_size(16_384)
        .on_upgrade(move |socket| run_socket(state, socket, ip, user_agent));

    if let Ok(val) = HeaderValue::from_str(&cookie) {
        resp.headers_mut().insert(header::SET_COOKIE, val);
    }
    resp
}

async fn run_socket(state: Arc<AppState>, socket: WebSocket, ip: String, user_agent: String) {
    state
        .metrics
        .ws_connections_total
        .fetch_add(1, Ordering::Relaxed);
    let (mut ws_sink, mut ws_stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<WsOut>();
    let backlog_bytes = Arc::new(AtomicUsize::new(0));
    let writer_backlog = Arc::clone(&backlog_bytes);
    let rtt_bits = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let conn_id = {
        let mut rt = state.rt.lock().unwrap();
        let id = rt.next_conn_id;
        rt.next_conn_id += 1;
        rt.conns.insert(
            id,
            ClientHandle {
                conn_id: id,
                ip,
                user_agent,
                connected_at: now_ms(),
                session_token: None,
                tx: tx.clone(),
                backlog_bytes,
                user_id: None,
                username: None,
                username_lower: None,
                squad_code: None,
                lobby_code: None,
                solo_sim: None,
                solo_match: None,
                solo_ac: None,
                solo_gen: 0,
                rtt_bits: Arc::clone(&rtt_bits),
            },
        );
        id
    };

    let writer = tokio::spawn(async move {
        while let Some(out) = rx.recv().await {
            let res = match out {
                WsOut::Text(s) => {
                    let len = s.len();
                    let r = ws_sink.send(Message::Text(s.into())).await;
                    writer_backlog.fetch_sub(len, Ordering::Relaxed);
                    r
                }
                WsOut::Binary(b) => {
                    let len = b.len();
                    let r = ws_sink.send(Message::Binary(b.into())).await;
                    writer_backlog.fetch_sub(len, Ordering::Relaxed);
                    r
                }
                WsOut::Ping => ws_sink.send(Message::Ping(Vec::new().into())).await,
                WsOut::PingData(d) => ws_sink.send(Message::Ping(d.into())).await,
                WsOut::Close => {
                    let _ = ws_sink.send(Message::Close(None)).await;
                    break;
                }
            };
            if res.is_err() {
                break;
            }
        }
    });

    let mut is_alive = true;
    let mut msg_count: u32 = 0;
    let mut msg_window_start = now_ms();
    let mut chat_times: Vec<i64> = Vec::new();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(25));
    heartbeat.tick().await; // skip immediate tick
    let mut rtt_probe = tokio::time::interval(Duration::from_secs(RTT_PROBE_SECS));
    rtt_probe.tick().await;

    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                // Stop stale solo matches (> 2 hours)
                {
                    let mut rt = state.rt.lock().unwrap();
                    let stale = rt
                        .conns
                        .get(&conn_id)
                        .and_then(|c| c.solo_sim.as_ref())
                        .map(|s| now_ms() - s.created_at() > 2 * 60 * 60 * 1000)
                        .unwrap_or(false);
                    if stale {
                        stop_solo_match(&mut rt, conn_id);
                    }
                }
                if !is_alive {
                    let _ = tx.send(WsOut::Close);
                    break;
                }
                is_alive = false;
                if tx.send(WsOut::Ping).is_err() {
                    break;
                }
            }
            _ = rtt_probe.tick() => {
                let _ = tx.send(WsOut::PingData(now_ms().to_le_bytes().to_vec()));
            }
            next_msg = ws_stream.next() => {
                match next_msg {
                    Some(Ok(Message::Pong(payload))) => {
                        is_alive = true;
                        if payload.len() == 8 {
                            let mut b = [0u8; 8];
                            b.copy_from_slice(&payload);
                            let rtt = (now_ms() - i64::from_le_bytes(b)) as f64;
                            if (0.0..5000.0).contains(&rtt) {
                                let old = f64::from_bits(rtt_bits.load(Ordering::Relaxed));
                                let v = if old > 0.0 { old * 0.8 + rtt * 0.2 } else { rtt.max(1.0) };
                                rtt_bits.store(v.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                    Some(Ok(Message::Ping(_))) => {
                        is_alive = true;
                    }
                    Some(Ok(Message::Text(text))) => {
                        let now = now_ms();
                        if now - msg_window_start > 1000 {
                            msg_window_start = now;
                            msg_count = 0;
                        }
                        msg_count += 1;
                        if msg_count > 200 {
                            continue;
                        }
                        if let Ok(msg) = serde_json::from_str::<Value>(&text) {
                            handle_ws_message(&state, conn_id, msg, now, &mut chat_times).await;
                        }
                    }
                    Some(Ok(Message::Binary(bytes))) => {
                        let now = now_ms();
                        if now - msg_window_start > 1000 {
                            msg_window_start = now;
                            msg_count = 0;
                        }
                        msg_count += 1;
                        if msg_count > 200 {
                            continue;
                        }
                        if let Ok(msg) = serde_json::from_slice::<Value>(&bytes) {
                            handle_ws_message(&state, conn_id, msg, now, &mut chat_times).await;
                        }
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => {
                        break;
                    }
                }
            }
        }
    }

    // Disconnect cleanup
    {
        let mut rt = state.rt.lock().unwrap();
        let db = state.db.lock().unwrap();
        if let Some(uid) = rt.conns.get(&conn_id).and_then(|c| c.user_id.clone())
            && rt.clients_by_user_id.get(&uid).copied() == Some(conn_id) {
                rt.clients_by_user_id.remove(&uid);
            }
        stop_solo_match(&mut rt, conn_id);
        leave_lobby(&mut rt, &db, conn_id, "disconnected");
        leave_squad(&mut rt, &db, conn_id);
        rt.conns.remove(&conn_id);
    }
    state.cluster.notify_changed();

    let _ = tx.send(WsOut::Close);
    let _ = writer.await;
}

fn send_to_conn(rt: &RuntimeState, conn_id: u64, obj: &Value) {
    if let Some(c) = rt.conns.get(&conn_id) {
        c.send_json(obj);
    }
}

fn send_to_user(rt: &RuntimeState, user_id: &str, obj: &Value) {
    if let Some(&cid) = rt.clients_by_user_id.get(user_id) {
        send_to_conn(rt, cid, obj);
    }
}

fn squad_snapshot(rt: &RuntimeState, db: &Database, sq: &Squad) -> Value {
    let members: Vec<Value> = sq
        .members
        .values()
        .map(|m| {
            let u = db.data.users.get(&m.username_lower);
            let rank = compute_rank(u.map(|x| x.rp).unwrap_or(0));
            json!({
                "id": m.user_id,
                "username": m.username,
                "level": u.map(|x| x.level.max(1)).unwrap_or(1),
                "rank": rank.short,
                "rankColor": rank.color,
                "isLeader": m.user_id == sq.leader_id,
                "online": rt.clients_by_user_id.contains_key(&m.user_id)
            })
        })
        .collect();
    json!({
        "code": sq.code,
        "leaderId": sq.leader_id,
        "members": members
    })
}

fn broadcast_squad(rt: &RuntimeState, sq: &Squad, msg: &Value) {
    for m in sq.members.values() {
        send_to_user(rt, &m.user_id, msg);
    }
}

fn lobby_snapshot(db: &Database, lb: &Lobby) -> Value {
    let players: Vec<Value> = lb
        .players
        .values()
        .map(|p| {
            let u = db.data.users.get(&p.username_lower);
            let rank = compute_rank(u.map(|x| x.rp).unwrap_or(0));
            let car = u.map(|x| x.car.clone()).unwrap_or_default();
            json!({
                "id": p.user_id,
                "username": p.username,
                "team": p.team,
                "ready": p.ready,
                "isHost": p.user_id == lb.host_id,
                "level": u.map(|x| x.level.max(1)).unwrap_or(1),
                "rp": u.map(|x| x.rp).unwrap_or(0),
                "rank": rank.short,
                "rankColor": rank.color,
                "squadCode": p.squad_code,
                "car": car
            })
        })
        .collect();
    json!({
        "code": lb.code,
        "hostId": lb.host_id,
        "status": lb.status.as_str(),
        "settings": lb.settings,
        "players": players
    })
}

fn broadcast_lobby(rt: &RuntimeState, lb: &Lobby, msg: &Value, exclude_user_id: Option<&str>) {
    for p in lb.players.values() {
        if exclude_user_id == Some(p.user_id.as_str()) {
            continue;
        }
        send_to_user(rt, &p.user_id, msg);
    }
}

fn stop_solo_match(rt: &mut RuntimeState, conn_id: u64) {
    if let Some(c) = rt.conns.get_mut(&conn_id) {
        c.solo_gen = c.solo_gen.wrapping_add(1);
        if let Some(sim) = c.solo_sim.take() {
            sim.stop();
        }
        c.solo_match = None;
        c.solo_ac = None;
    }
}

fn leave_squad(rt: &mut RuntimeState, db: &Database, conn_id: u64) {
    let (squad_code, user_id) = match rt.conns.get_mut(&conn_id) {
        Some(c) => (c.squad_code.take(), c.user_id.clone()),
        None => return,
    };
    let (Some(code), Some(uid)) = (squad_code, user_id) else {
        return;
    };
    let Some(sq) = rt.squads.get_mut(&code) else {
        return;
    };
    sq.members.shift_remove(&uid);
    if sq.members.is_empty() {
        rt.squads.shift_remove(&code);
        return;
    }
    if sq.leader_id == uid
        && let Some(next) = sq.members.values().next() {
            sq.leader_id = next.user_id.clone();
        }
    if let Some(sq_ref) = rt.squads.get(&code) {
        let msg = json!({ "type": "squad_update", "squad": squad_snapshot(rt, db, sq_ref) });
        broadcast_squad(rt, sq_ref, &msg);
    }
}

fn leave_lobby(rt: &mut RuntimeState, db: &Database, conn_id: u64, reason: &str) {
    let (lobby_code, user_id, username) = match rt.conns.get_mut(&conn_id) {
        Some(c) => (
            c.lobby_code.take(),
            c.user_id.clone(),
            c.username.clone().unwrap_or_default(),
        ),
        None => return,
    };
    let (Some(code), Some(uid)) = (lobby_code, user_id) else {
        return;
    };

    let mut migrated_msg: Option<Value> = None;
    let mut left_match_msg: Option<Value> = None;
    let mut removed_lobby = false;

    if let Some(lb) = rt.lobbies.get_mut(&code) {
        let was_host = lb.host_id == uid;
        let removed_player = lb.players.shift_remove(&uid);
        let was_driver = removed_player
            .as_ref()
            .is_some_and(|p| p.team == 0 || p.team == 1);
        lb.ac.remove(&uid);
        if let Some(sim) = lb.sim.as_ref() {
            sim.remove_player(&uid, reason);
        }

        if lb.players.is_empty() {
            if let Some(sim) = lb.sim.take() {
                sim.stop();
            }
            removed_lobby = true;
        } else {
            if was_host
                && let Some(next) = lb.players.values().next() {
                    lb.host_id = next.user_id.clone();
                    if lb.status == LobbyStatus::InMatch {
                        migrated_msg = Some(json!({
                            "type": "host_migrated",
                            "hostId": lb.host_id,
                            "hostName": next.username
                        }));
                    }
                }
            if lb.status == LobbyStatus::InMatch {
                left_match_msg = Some(json!({
                    "type": "player_left_match",
                    "userId": uid,
                    "username": username,
                    "reason": reason,
                    "replacedByBot": was_driver
                }));
            }
        }
    }

    if removed_lobby {
        rt.lobbies.shift_remove(&code);
        return;
    }

    if let Some(lb) = rt.lobbies.get(&code) {
        if let Some(msg) = migrated_msg {
            broadcast_lobby(rt, lb, &msg, None);
        }
        if let Some(msg) = left_match_msg {
            broadcast_lobby(rt, lb, &msg, None);
        }
        let upd = json!({ "type": "lobby_update", "lobby": lobby_snapshot(db, lb) });
        broadcast_lobby(rt, lb, &upd, None);
    }
}

fn join_lobby_internal(
    rt: &mut RuntimeState,
    db: &Database,
    conn_id: u64,
    lobby_code: &str,
    preferred_team: Option<i32>,
) -> bool {
    let (cur_lobby, user_id, username, username_lower, squad_code) = match rt.conns.get(&conn_id) {
        Some(c) => (
            c.lobby_code.clone(),
            c.user_id.clone(),
            c.username.clone(),
            c.username_lower.clone(),
            c.squad_code.clone(),
        ),
        None => return false,
    };
    let (Some(uid), Some(uname), Some(ulower)) = (user_id, username, username_lower) else {
        return false;
    };
    if cur_lobby.as_deref() == Some(lobby_code) {
        return true;
    }
    let user_rec = db.data.users.get(&ulower);
    if ac_ban_remaining_ms(user_rec) > 0 {
        send_to_conn(
            rt,
            conn_id,
            &json!({ "type": "error", "text": ac_ban_message(user_rec) }),
        );
        return false;
    }
    if cur_lobby.is_some() {
        leave_lobby(rt, db, conn_id, "left");
    }
    let Some(lb) = rt.lobbies.get_mut(lobby_code) else {
        return false;
    };
    let team = lb.pick_team(preferred_team);
    let is_host = lb.host_id == uid;
    lb.players.insert(
        uid.clone(),
        LobbyPlayer {
            user_id: uid.clone(),
            username: uname,
            username_lower: ulower,
            team,
            ready: is_host,
            squad_code,
        },
    );
    if let Some(c) = rt.conns.get_mut(&conn_id) {
        c.lobby_code = Some(lobby_code.to_string());
    }
    if let Some(lb) = rt.lobbies.get(lobby_code) {
        let snap = lobby_snapshot(db, lb);
        send_to_conn(rt, conn_id, &json!({ "type": "lobby_joined", "lobby": snap }));
        broadcast_lobby(
            rt,
            lb,
            &json!({ "type": "lobby_update", "lobby": snap }),
            Some(&uid),
        );
    }
    true
}

fn pull_squad_into_lobby(
    rt: &mut RuntimeState,
    db: &Database,
    leader_conn_id: u64,
    lobby_code: &str,
    preferred_team: Option<i32>,
) {
    let (squad_code, leader_uid) = match rt.conns.get(&leader_conn_id) {
        Some(c) => (c.squad_code.clone(), c.user_id.clone()),
        None => return,
    };
    let (Some(sq_code), Some(l_uid)) = (squad_code, leader_uid) else {
        return;
    };
    let member_cids: Vec<u64> = match rt.squads.get(&sq_code) {
        Some(sq) if sq.leader_id == l_uid => sq
            .members
            .values()
            .filter(|m| m.user_id != l_uid)
            .filter_map(|m| rt.clients_by_user_id.get(&m.user_id).copied())
            .collect(),
        _ => return,
    };
    for mcid in member_cids {
        join_lobby_internal(rt, db, mcid, lobby_code, preferred_team);
        send_to_conn(
            rt,
            mcid,
            &json!({
                "type": "toast",
                "text": format!("SQUAD LEADER PULLED YOU INTO LOBBY {lobby_code}")
            }),
        );
    }
}

fn add_lobby_strike(
    state: &Arc<AppState>,
    rt: &mut RuntimeState,
    db: &mut Database,
    lobby_code: &str,
    conn_id: u64,
    reason: &str,
    severity: u32,
) -> bool {
    let (uid, uname, ulower) = match rt.conns.get(&conn_id) {
        Some(c) => match (&c.user_id, &c.username, &c.username_lower) {
            (Some(id), Some(n), Some(l)) => (id.clone(), n.clone(), l.clone()),
            _ => return false,
        },
        None => return false,
    };

    let now = now_ms();
    let (strikes, prev_strikes) = {
        let Some(lb) = rt.lobbies.get_mut(lobby_code) else {
            return false;
        };
        let rec = lb.ac.entry(uid).or_insert_with(|| make_ac_rec(now));
        decay_strikes(rec, now);
        let prev = rec.strikes;
        rec.strikes = rec.strikes.saturating_add(severity);
        (rec.strikes, prev)
    };

    if strikes >= AC_KICK_STRIKES {
        if let Some(user) = db.data.users.get_mut(&ulower) {
            ac_penalize(user, reason);
            let cloned = user.clone();
            db.mark_dirty();
            if state.config.is_primary() {
                crate::cluster::fanout_user_sync(Arc::clone(state), ulower.clone(), Some(cloned));
            }
        }
        if !state.config.is_primary() {
            crate::cluster::notify_primary_ac_penalize(
                Arc::clone(state),
                ulower.clone(),
                reason.to_string(),
            );
        }
        send_to_conn(
            rt,
            conn_id,
            &json!({ "type": "anticheat_kick", "reason": format!("Anti-Cheat Kick: {reason}") }),
        );
        leave_lobby(rt, db, conn_id, "anticheat");
        if let Some(lb) = rt.lobbies.get(lobby_code) {
            broadcast_lobby(
                rt,
                lb,
                &json!({
                    "type": "lobby_chat",
                    "system": true,
                    "text": format!("[ANTI-CHEAT] {uname} was removed ({reason}).")
                }),
                None,
            );
        }
        return false;
    }

    if strikes >= AC_WARN_STRIKES && prev_strikes < AC_WARN_STRIKES {
        send_to_conn(
            rt,
            conn_id,
            &json!({
                "type": "anticheat_warn",
                "reason": format!("Suspicious activity detected ({reason}). Continued violations will kick you.")
            }),
        );
    }
    true
}

fn add_solo_strike(
    state: &Arc<AppState>,
    rt: &mut RuntimeState,
    db: &mut Database,
    conn_id: u64,
    reason: &str,
    severity: u32,
) -> bool {
    let now = now_ms();
    let (ulower, strikes, prev_strikes) = match rt.conns.get_mut(&conn_id) {
        Some(c) => {
            let rec = c.solo_ac.get_or_insert_with(|| make_ac_rec(now));
            decay_strikes(rec, now);
            let prev = rec.strikes;
            rec.strikes = rec.strikes.saturating_add(severity);
            (c.username_lower.clone(), rec.strikes, prev)
        }
        None => return false,
    };

    if strikes >= AC_KICK_STRIKES {
        if let Some(ul) = ulower {
            if let Some(user) = db.data.users.get_mut(&ul) {
                ac_penalize(user, reason);
                let cloned = user.clone();
                db.mark_dirty();
                if state.config.is_primary() {
                    crate::cluster::fanout_user_sync(Arc::clone(state), ul.clone(), Some(cloned));
                }
            }
            if !state.config.is_primary() {
                crate::cluster::notify_primary_ac_penalize(
                    Arc::clone(state),
                    ul,
                    reason.to_string(),
                );
            }
        }
        stop_solo_match(rt, conn_id);
        send_to_conn(
            rt,
            conn_id,
            &json!({ "type": "anticheat_kick", "reason": format!("Anti-Cheat Kick: {reason}") }),
        );
        return false;
    }

    if strikes >= AC_WARN_STRIKES && prev_strikes < AC_WARN_STRIKES {
        send_to_conn(
            rt,
            conn_id,
            &json!({
                "type": "anticheat_warn",
                "reason": format!("Suspicious activity detected ({reason}). Continued violations will kick you.")
            }),
        );
    }
    true
}

pub fn apply_solo_rewards_to_db(
    db: &mut Database,
    uid: &str,
    ulower: &str,
    duration: i32,
    difficulty: i32,
    tournament: bool,
    player_team: i32,
    sim_score: [i32; 2],
    player_stats: &HashMap<String, MatchStatsEntry>,
) -> Value {
    let blue_score = sim_score[0].clamp(0, 99);
    let orange_score = sim_score[1].clamp(0, 99);
    let winner_team = if blue_score > orange_score {
        0
    } else if orange_score > blue_score {
        1
    } else {
        -1
    };

    if duration <= 0 || (player_team != 0 && player_team != 1) {
        return Value::Null;
    }
    let Some(user) = db.data.users.get_mut(ulower) else {
        return Value::Null;
    };
    let st = player_stats.get(uid).cloned().unwrap_or_default();
    let goals = st.goals.min(20);
    let assists = st.assists.min(20);
    let saves = st.saves.min(25);
    let shots = st.shots.min(35);
    let demos = st.demos.min(20);
    let won = winner_team == player_team;
    let lost = winner_team >= 0 && winner_team != player_team;

    user.stats.matches += 1;
    if won {
        user.stats.wins += 1;
    } else if lost {
        user.stats.losses += 1;
    }
    user.stats.goals += goals;
    user.stats.assists += assists;
    user.stats.saves += saves;
    user.stats.shots += shots;
    user.stats.demos += demos;

    let diff_mult =
        1.0 + f64::from(difficulty) * 0.15 + if tournament { 0.25 } else { 0.0 };
    let time_factor = (f64::from(duration) / 60.0).min(1.0);
    let raw_xp = (75.0
        + if won { 65.0 } else { 20.0 }
        + (goals * 30 + assists * 20 + saves * 25 + shots * 6 + demos * 10) as f64)
        * diff_mult
        * time_factor;
    let xp_gain = (raw_xp.round() as u64).min(750);
    let xp_res = add_xp(user, xp_gain);

    let mut rp_delta: i64 = 0;
    if won && user.rp < 600 {
        rp_delta = 8 + i64::from(difficulty) * 3;
        user.rp = (user.rp + rp_delta).min(600);
        user.peak_rp = user.peak_rp.max(user.rp);
    }
    let pub_u = public_user(user);
    db.mark_dirty();
    json!({
        "xpGained": xp_res.xp_gained,
        "oldLevel": xp_res.old_level,
        "newLevel": xp_res.new_level,
        "leveledUp": xp_res.leveled_up,
        "rpDelta": rp_delta,
        "user": pub_u
    })
}

fn finalize_solo_match(
    state: &Arc<AppState>,
    conn_id: u64,
    solo_gen: u64,
    sim_score: [i32; 2],
    player_stats: HashMap<String, MatchStatsEntry>,
) {
    let (sm, uid, ulower) = {
        let mut rt = state.rt.lock().unwrap();
        match rt.conns.get_mut(&conn_id) {
            Some(c) => {
                if c.solo_gen != solo_gen {
                    return;
                }
                let Some(sm) = c.solo_match.as_mut() else {
                    return;
                };
                if sm.finished {
                    return;
                }
                sm.finished = true;
                let sm_clone = sm.clone();
                if let Some(sim) = c.solo_sim.take() {
                    sim.stop();
                }
                c.solo_match = None;
                c.solo_ac = None;
                (sm_clone, c.user_id.clone(), c.username_lower.clone())
            }
            None => return,
        }
    };

    let blue_score = sim_score[0].clamp(0, 99);
    let orange_score = sim_score[1].clamp(0, 99);
    let winner_team = if blue_score > orange_score {
        0
    } else if orange_score > blue_score {
        1
    } else {
        -1
    };

    if sm.player_team != 0 && sm.player_team != 1 {
        let rt = state.rt.lock().unwrap();
        send_to_conn(
            &rt,
            conn_id,
            &json!({
                "type": "solo_match_finished",
                "score": [blue_score, orange_score],
                "winnerTeam": winner_team,
                "reward": Value::Null
            }),
        );
        return;
    }

    if state.config.is_primary() {
        let (reward, updated_user) = {
            let mut db = state.db.lock().unwrap();
            if let (Some(uid), Some(ulower)) = (&uid, &ulower) {
                let r = apply_solo_rewards_to_db(
                    &mut db,
                    uid,
                    ulower,
                    sm.duration,
                    sm.difficulty,
                    sm.tournament,
                    sm.player_team,
                    sim_score,
                    &player_stats,
                );
                let u = db.data.users.get(ulower).cloned();
                (r, u.map(|u| (ulower.clone(), u)))
            } else {
                (Value::Null, None)
            }
        };
        if let Some((ul, u)) = updated_user {
            crate::cluster::fanout_user_sync(Arc::clone(state), ul, Some(u));
        }
        let rt = state.rt.lock().unwrap();
        send_to_conn(
            &rt,
            conn_id,
            &json!({
                "type": "solo_match_finished",
                "score": [blue_score, orange_score],
                "winnerTeam": winner_team,
                "reward": reward
            }),
        );
    } else {
        let state_clone = Arc::clone(state);
        tokio::spawn(async move {
            let reward = if let (Some(uid), Some(ulower)) = (uid, ulower) {
                let req = crate::cluster::SoloResultRequest {
                    result_id: crate::auth::random_hex(16),
                    user_id: uid.clone(),
                    username_lower: ulower.clone(),
                    duration: sm.duration,
                    difficulty: sm.difficulty,
                    tournament: sm.tournament,
                    player_team: sm.player_team,
                    sim_score,
                    player_stats: player_stats.clone(),
                };
                if let Some(res) =
                    crate::cluster::apply_solo_result_via_primary(&state_clone, &req).await
                {
                    if let Some(u) = res.user {
                        let mut db = state_clone.db.lock().unwrap();
                        db.data.users.insert(ulower, u);
                    }
                    res.reward
                } else {
                    let mut db = state_clone.db.lock().unwrap();
                    apply_solo_rewards_to_db(
                        &mut db,
                        &uid,
                        &ulower,
                        sm.duration,
                        sm.difficulty,
                        sm.tournament,
                        sm.player_team,
                        sim_score,
                        &player_stats,
                    )
                }
            } else {
                Value::Null
            };
            let rt = state_clone.rt.lock().unwrap();
            send_to_conn(
                &rt,
                conn_id,
                &json!({
                    "type": "solo_match_finished",
                    "score": [blue_score, orange_score],
                    "winnerTeam": winner_team,
                    "reward": reward
                }),
            );
        });
    }
}

pub fn apply_mp_rewards_to_db(
    db: &mut Database,
    started_at: i64,
    participants: &[MatchParticipant],
    is_ranked_setting: bool,
    players_list: &[(String, String, i32)],
    sim_score: [i32; 2],
    player_stats: &HashMap<String, MatchStatsEntry>,
) -> (serde_json::Map<String, Value>, Option<String>, Vec<String>) {
    let duration_sec = ((now_ms() - started_at) as f64 / 1000.0).max(1.0);
    let blue_score = sim_score[0].clamp(0, 99);
    let orange_score = sim_score[1].clamp(0, 99);
    let winner_team = if blue_score > orange_score {
        0
    } else if orange_score > blue_score {
        1
    } else {
        -1
    };

    let mut is_ranked = is_ranked_setting && duration_sec >= 25.0;
    let mut ranked_void: Option<String> = None;
    if is_ranked
        && let RankedEligibility::Ineligible(reason) =
            ranked_eligibility_from_participants(participants)
    {
        is_ranked = false;
        ranked_void = Some(reason.to_string());
    }

    let mut rewards = serde_json::Map::new();
    let mut stayed_ids: HashSet<String> = HashSet::new();
    let mut touched_users: Vec<String> = Vec::new();

    for (uid, ulower, team) in players_list {
        stayed_ids.insert(uid.clone());
        let Some(user) = db.data.users.get_mut(ulower) else {
            continue;
        };
        let st = player_stats.get(uid).cloned().unwrap_or_default();
        let goals = st.goals.min(20);
        let assists = st.assists.min(20);
        let saves = st.saves.min(25);
        let shots = st.shots.min(35);
        let demos = st.demos.min(20);
        let won = winner_team == *team;
        let lost = winner_team >= 0 && winner_team != *team;

        user.stats.matches += 1;
        user.stats.mp_matches += 1;
        if won {
            user.stats.wins += 1;
            user.stats.mp_wins += 1;
        } else if lost {
            user.stats.losses += 1;
        }
        user.stats.goals += goals;
        user.stats.assists += assists;
        user.stats.saves += saves;
        user.stats.shots += shots;
        user.stats.demos += demos;

        let time_factor = (duration_sec / 45.0).min(1.0);
        let raw_xp = (120.0
            + if won { 100.0 } else { 35.0 }
            + (goals * 45 + assists * 30 + saves * 35 + shots * 10 + demos * 15) as f64)
            * if is_ranked { 1.25 } else { 1.0 }
            * time_factor;
        let xp_gain = (raw_xp.round() as u64).min(1200);
        let xp_res = add_xp(user, xp_gain);

        let mut rp_delta: i64 = 0;
        if is_ranked {
            let perf_bonus = (goals * 3 + saves * 2 + assists * 2).min(12) as i64;
            if won {
                rp_delta = 28 + perf_bonus;
            } else if lost {
                rp_delta = -(22 - ((perf_bonus as f64) * 0.5).floor() as i64).max(10);
            } else {
                rp_delta = 5;
            }
            user.rp = (user.rp + rp_delta).max(0);
            user.peak_rp = user.peak_rp.max(user.rp);
        }

        let pub_u = public_user(user);
        touched_users.push(ulower.clone());
        rewards.insert(
            uid.clone(),
            json!({
                "xpGained": xp_res.xp_gained,
                "oldLevel": xp_res.old_level,
                "newLevel": xp_res.new_level,
                "leveledUp": xp_res.leveled_up,
                "rpDelta": rp_delta,
                "user": pub_u
            }),
        );
    }

    if is_ranked {
        for part in participants {
            if stayed_ids.contains(&part.user_id) {
                continue;
            }
            if let Some(user) = db.data.users.get_mut(&part.username_lower) {
                user.stats.matches += 1;
                user.stats.mp_matches += 1;
                user.stats.losses += 1;
                user.rp = (user.rp - 22).max(0);
                touched_users.push(part.username_lower.clone());
            }
        }
    }
    db.mark_dirty();
    (rewards, ranked_void, touched_users)
}

fn finalize_multiplayer_match(
    state: &Arc<AppState>,
    lobby_code: &str,
    match_gen: u64,
    sim_score: [i32; 2],
    player_stats: HashMap<String, MatchStatsEntry>,
) {
    let (started_at, participants, is_ranked_setting, players_list) = {
        let mut rt = state.rt.lock().unwrap();
        let Some(lb) = rt.lobbies.get_mut(lobby_code) else {
            return;
        };
        let Some(m) = lb.active_match.as_mut() else {
            return;
        };
        if m.match_gen != match_gen || m.finished {
            return;
        }
        m.finished = true;
        lb.status = LobbyStatus::Waiting;
        if let Some(sim) = lb.sim.take() {
            sim.stop();
        }
        let host_id = lb.host_id.clone();
        for p in lb.players.values_mut() {
            p.ready = p.user_id == host_id;
        }
        let plist: Vec<(String, String, i32)> = lb
            .players
            .values()
            .filter(|p| p.team == 0 || p.team == 1)
            .map(|p| (p.user_id.clone(), p.username_lower.clone(), p.team))
            .collect();
        (m.started_at, m.participants.clone(), lb.settings.ranked, plist)
    };
    state.cluster.notify_changed();

    let blue_score = sim_score[0].clamp(0, 99);
    let orange_score = sim_score[1].clamp(0, 99);
    let winner_team = if blue_score > orange_score {
        0
    } else if orange_score > blue_score {
        1
    } else {
        -1
    };

    if state.config.is_primary() {
        let (rewards, ranked_void, synced_users) = {
            let mut db = state.db.lock().unwrap();
            let (rewards, ranked_void, touched) = apply_mp_rewards_to_db(
                &mut db,
                started_at,
                &participants,
                is_ranked_setting,
                &players_list,
                sim_score,
                &player_stats,
            );
            let synced: Vec<(String, crate::db::User)> = touched
                .into_iter()
                .filter_map(|k| db.data.users.get(&k).cloned().map(|u| (k, u)))
                .collect();
            (rewards, ranked_void, synced)
        };
        for (k, u) in synced_users {
            crate::cluster::fanout_user_sync(Arc::clone(state), k, Some(u));
        }
        let rt = state.rt.lock().unwrap();
        let db = state.db.lock().unwrap();
        if let Some(lb) = rt.lobbies.get(lobby_code) {
            let msg = json!({
                "type": "match_finished",
                "score": [blue_score, orange_score],
                "winnerTeam": winner_team,
                "rewards": rewards,
                "rankedVoid": ranked_void,
                "lobby": lobby_snapshot(&db, lb)
            });
            broadcast_lobby(&rt, lb, &msg, None);
        }
    } else {
        let state_clone = Arc::clone(state);
        let code_owned = lobby_code.to_string();
        tokio::spawn(async move {
            let req = crate::cluster::MpResultRequest {
                result_id: crate::auth::random_hex(16),
                started_at,
                participants: participants.clone(),
                is_ranked_setting,
                players_list: players_list.clone(),
                sim_score,
                player_stats: player_stats.clone(),
            };
            let (rewards, ranked_void) = if let Some(res) =
                crate::cluster::apply_mp_result_via_primary(&state_clone, &req).await
            {
                let mut db = state_clone.db.lock().unwrap();
                for (k, u) in res.updated_users {
                    db.data.users.insert(k, u);
                }
                (res.rewards, res.ranked_void)
            } else {
                let mut db = state_clone.db.lock().unwrap();
                let (r, rv, _) = apply_mp_rewards_to_db(
                    &mut db,
                    started_at,
                    &participants,
                    is_ranked_setting,
                    &players_list,
                    sim_score,
                    &player_stats,
                );
                (r, rv)
            };
            let rt = state_clone.rt.lock().unwrap();
            let db = state_clone.db.lock().unwrap();
            if let Some(lb) = rt.lobbies.get(&code_owned) {
                let msg = json!({
                    "type": "match_finished",
                    "score": [blue_score, orange_score],
                    "winnerTeam": winner_team,
                    "rewards": rewards,
                    "rankedVoid": ranked_void,
                    "lobby": lobby_snapshot(&db, lb)
                });
                broadcast_lobby(&rt, lb, &msg, None);
            }
        });
    }
}

struct LobbyEventSink {
    state: Arc<AppState>,
    lobby_code: String,
    match_gen: u64,
}

impl LobbyEventSink {
    fn is_current(&self, lb: &Lobby) -> bool {
        lb.status == LobbyStatus::InMatch
            && lb
                .active_match
                .as_ref()
                .is_some_and(|m| m.match_gen == self.match_gen && !m.finished)
    }
}

impl MatchEventSink for LobbyEventSink {
    fn rtt_ms(&self, user_id: &str) -> Option<f64> {
        let rt = self.state.rt.lock().unwrap();
        rt.clients_by_user_id
            .get(user_id)
            .and_then(|cid| rt.conns.get(cid))
            .and_then(|c| c.rtt_ms())
    }

    fn on_ac_report(&self, report: crate::protocol::AcReport) {
        crate::anticheat::submit(Arc::clone(&self.state), report);
    }

    fn broadcast(&self, msg: Value) {
        let rt = self.state.rt.lock().unwrap();
        if let Some(lb) = rt.lobbies.get(&self.lobby_code)
            && self.is_current(lb)
        {
            broadcast_lobby(&rt, lb, &msg, None);
        }
    }

    fn send_user(&self, user_id: &str, msg: Value) {
        let rt = self.state.rt.lock().unwrap();
        let Some(lb) = rt.lobbies.get(&self.lobby_code) else {
            return;
        };
        if !self.is_current(lb) {
            return;
        }
        if let Some(&cid) = rt.clients_by_user_id.get(user_id)
            && let Some(c) = rt.conns.get(&cid)
            && c.lobby_code.as_deref() == Some(&self.lobby_code)
        {
            c.send_json(&msg);
        }
    }

    fn send_raw(&self, user_id: &str, payload: SnapshotPayload) {
        let rt = self.state.rt.lock().unwrap();
        let Some(lb) = rt.lobbies.get(&self.lobby_code) else {
            return;
        };
        if !self.is_current(lb) {
            return;
        }
        if let Some(&cid) = rt.clients_by_user_id.get(user_id)
            && let Some(c) = rt.conns.get(&cid)
            && c.lobby_code.as_deref() == Some(&self.lobby_code)
        {
            c.send_snapshot_if_unblocked(payload);
        }
    }

    fn send_spectators(&self, payload: SnapshotPayload) {
        let rt = self.state.rt.lock().unwrap();
        let Some(lb) = rt.lobbies.get(&self.lobby_code) else {
            return;
        };
        if !self.is_current(lb) {
            return;
        }
        for p in lb.players.values() {
            if p.team == 0 || p.team == 1 {
                continue;
            }
            if let Some(&cid) = rt.clients_by_user_id.get(&p.user_id)
                && let Some(c) = rt.conns.get(&cid)
            {
                c.send_snapshot_if_unblocked(payload.clone());
            }
        }
    }

    fn on_goal(&self, mut info: Value) {
        if let Some(obj) = info.as_object_mut() {
            obj.insert("type".to_string(), json!("match_goal_ack"));
        } else {
            info = json!({ "type": "match_goal_ack" });
        }
        self.broadcast(info);
    }

    fn on_end(&self, score: [i32; 2], stats: HashMap<String, MatchStatsEntry>) {
        finalize_multiplayer_match(&self.state, &self.lobby_code, self.match_gen, score, stats);
    }

    fn on_afk_kick(&self, user_id: &str) {
        let mut rt = self.state.rt.lock().unwrap();
        let db = self.state.db.lock().unwrap();
        let (conn_id, username) = {
            let Some(lb) = rt.lobbies.get(&self.lobby_code) else {
                return;
            };
            if !self.is_current(lb) {
                return;
            }
            let Some(p) = lb.players.get(user_id) else {
                return;
            };
            let uname = p.username.clone();
            let cid = rt.clients_by_user_id.get(user_id).copied();
            (cid, uname)
        };
        if let Some(cid) = conn_id
            && rt
                .conns
                .get(&cid)
                .is_some_and(|c| c.lobby_code.as_deref() == Some(self.lobby_code.as_str()))
        {
            send_to_conn(
                &rt,
                cid,
                &json!({
                    "type": "afk_kick",
                    "reason": "Removed from match for inactivity (AFK)."
                }),
            );
            leave_lobby(&mut rt, &db, cid, "afk");
            if let Some(lb) = rt.lobbies.get(&self.lobby_code) {
                broadcast_lobby(
                    &rt,
                    lb,
                    &json!({
                        "type": "lobby_chat",
                        "system": true,
                        "text": format!("[AFK] {username} was kicked for inactivity and replaced by a bot.")
                    }),
                    None,
                );
            }
        }
    }

    fn on_error(&self, message: String) {
        eprintln!("[SIM] Match aborted: {message}");
        let mut rt = self.state.rt.lock().unwrap();
        let db = self.state.db.lock().unwrap();
        let should_broadcast = {
            let Some(lb) = rt.lobbies.get_mut(&self.lobby_code) else {
                return;
            };
            let Some(m) = lb.active_match.as_mut() else {
                return;
            };
            if m.match_gen != self.match_gen || m.finished {
                return;
            }
            m.finished = true;
            lb.status = LobbyStatus::Waiting;
            lb.sim = None;
            let host_id = lb.host_id.clone();
            for p in lb.players.values_mut() {
                p.ready = p.user_id == host_id;
            }
            true
        };
        if should_broadcast
            && let Some(lb) = rt.lobbies.get(&self.lobby_code) {
                broadcast_lobby(
                    &rt,
                    lb,
                    &json!({ "type": "error", "text": "Match ended unexpectedly due to a simulation error." }),
                    None,
                );
                broadcast_lobby(
                    &rt,
                    lb,
                    &json!({ "type": "lobby_update", "lobby": lobby_snapshot(&db, lb) }),
                    None,
                );
            }
    }
}

struct SoloEventSink {
    state: Arc<AppState>,
    conn_id: u64,
    user_id: String,
    solo_gen: u64,
    spectating: bool,
}

impl SoloEventSink {
    fn is_current(&self, rt: &RuntimeState) -> bool {
        rt.conns
            .get(&self.conn_id)
            .is_some_and(|c| c.solo_gen == self.solo_gen)
    }
}

impl MatchEventSink for SoloEventSink {
    fn rtt_ms(&self, _user_id: &str) -> Option<f64> {
        let rt = self.state.rt.lock().unwrap();
        rt.conns.get(&self.conn_id).and_then(|c| c.rtt_ms())
    }

    fn on_ac_report(&self, mut report: crate::protocol::AcReport) {
        report.kind = "solo".into();
        crate::anticheat::submit(Arc::clone(&self.state), report);
    }

    fn broadcast(&self, msg: Value) {
        let rt = self.state.rt.lock().unwrap();
        if self.is_current(&rt) {
            send_to_conn(&rt, self.conn_id, &msg);
        }
    }

    fn send_user(&self, user_id: &str, msg: Value) {
        if user_id != self.user_id {
            return;
        }
        self.broadcast(msg);
    }

    fn send_raw(&self, user_id: &str, payload: SnapshotPayload) {
        if user_id != self.user_id {
            return;
        }
        let rt = self.state.rt.lock().unwrap();
        if self.is_current(&rt)
            && let Some(c) = rt.conns.get(&self.conn_id)
        {
            c.send_snapshot_if_unblocked(payload);
        }
    }

    fn send_spectators(&self, payload: SnapshotPayload) {
        if !self.spectating {
            return;
        }
        let rt = self.state.rt.lock().unwrap();
        if self.is_current(&rt)
            && let Some(c) = rt.conns.get(&self.conn_id)
        {
            c.send_snapshot_if_unblocked(payload);
        }
    }

    fn on_goal(&self, mut info: Value) {
        if let Some(obj) = info.as_object_mut() {
            obj.insert("type".to_string(), json!("match_goal_ack"));
        } else {
            info = json!({ "type": "match_goal_ack" });
        }
        self.broadcast(info);
    }

    fn on_end(&self, score: [i32; 2], stats: HashMap<String, MatchStatsEntry>) {
        finalize_solo_match(&self.state, self.conn_id, self.solo_gen, score, stats);
    }

    fn on_afk_kick(&self, _user_id: &str) {}

    fn on_error(&self, message: String) {
        eprintln!("[SIM] Solo match aborted: {message}");
        let mut rt = self.state.rt.lock().unwrap();
        if !self.is_current(&rt) {
            return;
        }
        stop_solo_match(&mut rt, self.conn_id);
        send_to_conn(
            &rt,
            self.conn_id,
            &json!({ "type": "error", "text": "Match ended unexpectedly due to a simulation error." }),
        );
    }
}

async fn handle_ws_message(
    state: &Arc<AppState>,
    conn_id: u64,
    msg: Value,
    now: i64,
    chat_times: &mut Vec<i64>,
) {
    let Some(msg_type) = msg.get("type").and_then(|v| v.as_str()) else {
        return;
    };

    // Latency probe
    if msg_type == "ping" {
        if let Some(t) = msg.get("t").and_then(|v| v.as_f64()).filter(|t| t.is_finite()) {
            let rt = state.rt.lock().unwrap();
            send_to_conn(&rt, conn_id, &json!({ "type": "pong", "t": t }));
        }
        return;
    }

    // Auth
    if msg_type == "auth" {
        let token = msg.get("token").and_then(|v| v.as_str()).unwrap_or("");
        let (conn_ip, conn_ua) = {
            let rt = state.rt.lock().unwrap();
            rt.conns
                .get(&conn_id)
                .map(|c| (c.ip.clone(), c.user_agent.clone()))
                .unwrap_or_default()
        };
        let user_info = if state.config.is_primary() {
            let mut db = state.db.lock().unwrap();
            let info = db.get_user_by_token(token).map(|u| {
                (
                    u.id.clone(),
                    u.username.clone(),
                    u.username.to_ascii_lowercase(),
                    public_user(u),
                )
            });
            if info.is_some() {
                db.touch_session(token, &conn_ip, &conn_ua);
            }
            info
        } else {
            match crate::cluster::auth_token_via_primary(state, token, &conn_ip, &conn_ua).await {
                crate::cluster::PrimaryAuth::Ok(user, session) => {
                    let ulower = user.username.to_ascii_lowercase();
                    let info = (
                        user.id.clone(),
                        user.username.clone(),
                        ulower.clone(),
                        public_user(&user),
                    );
                    let mut db = state.db.lock().unwrap();
                    db.data.users.insert(ulower, *user);
                    db.data.sessions.insert(token.to_string(), *session);
                    Some(info)
                }
                crate::cluster::PrimaryAuth::Rejected => {
                    // Primary is authoritative: drop any stale mirror copy of this session so a
                    // revoked/expired token can't be accepted later via the fallback below.
                    let mut db = state.db.lock().unwrap();
                    db.data.sessions.shift_remove(token);
                    None
                }
                crate::cluster::PrimaryAuth::Unavailable => {
                    // Primary unreachable: fall back to the local mirror.
                    let mut db = state.db.lock().unwrap();
                    let info = db.get_user_by_token(token).map(|u| {
                        (
                            u.id.clone(),
                            u.username.clone(),
                            u.username.to_ascii_lowercase(),
                            public_user(u),
                        )
                    });
                    if info.is_none() {
                        // Not in the mirror either (e.g. brand-new account). Don't send
                        // auth_error: the client treats it as a revoked session and logs out.
                        // Close instead so it reconnects and retries in a few seconds.
                        drop(db);
                        let rt = state.rt.lock().unwrap();
                        send_to_conn(
                            &rt,
                            conn_id,
                            &json!({ "type": "toast", "text": "SERVER BUSY, RETRYING..." }),
                        );
                        if let Some(c) = rt.conns.get(&conn_id) {
                            c.close();
                        }
                        return;
                    }
                    info
                }
            }
        };
        let Some((uid, uname, ulower, pub_u)) = user_info else {
            let rt = state.rt.lock().unwrap();
            send_to_conn(
                &rt,
                conn_id,
                &json!({ "type": "auth_error", "error": "Invalid session" }),
            );
            return;
        };

        let mut rt = state.rt.lock().unwrap();
        let db = state.db.lock().unwrap();
        // If this socket was already authenticated as a different user, clean up the prior identity.
        if let Some(old_uid) = rt.conns.get(&conn_id).and_then(|c| c.user_id.clone())
            && old_uid != uid
        {
            stop_solo_match(&mut rt, conn_id);
            leave_lobby(&mut rt, &db, conn_id, "left");
            leave_squad(&mut rt, &db, conn_id);
            if rt.clients_by_user_id.get(&old_uid).copied() == Some(conn_id) {
                rt.clients_by_user_id.remove(&old_uid);
            }
        }
        if let Some(&prev_cid) = rt.clients_by_user_id.get(&uid)
            && prev_cid != conn_id {
                send_to_conn(
                    &rt,
                    prev_cid,
                    &json!({ "type": "kicked", "reason": "Signed in from another tab/device." }),
                );
                stop_solo_match(&mut rt, prev_cid);
                leave_lobby(&mut rt, &db, prev_cid, "left");
                leave_squad(&mut rt, &db, prev_cid);
                if let Some(prev) = rt.conns.get(&prev_cid) {
                    prev.close();
                }
            }
        if let Some(c) = rt.conns.get_mut(&conn_id) {
            c.user_id = Some(uid.clone());
            c.username = Some(uname);
            c.username_lower = Some(ulower);
            c.session_token = Some(token.to_string());
        }
        rt.clients_by_user_id.insert(uid.clone(), conn_id);
        drop(db);
        drop(rt);
        crate::cluster::fanout_kick_user_except(
            Arc::clone(state),
            uid,
            "Signed in from another tab/device.".into(),
            Some(token.to_string()),
        );
        state.cluster.notify_changed();
        let online = crate::cluster::global_online_count(state);
        let rt = state.rt.lock().unwrap();
        send_to_conn(
            &rt,
            conn_id,
            &json!({
                "type": "auth_ok",
                "user": pub_u,
                "region": state.config.region,
                "online": online
            }),
        );
        return;
    }

    // Require auth for all remaining messages
    let authed_info = {
        let rt = state.rt.lock().unwrap();
        rt.conns.get(&conn_id).and_then(|c| {
            Some((
                c.user_id.clone()?,
                c.username.clone()?,
                c.username_lower.clone()?,
            ))
        })
    };
    let Some((user_id, username, username_lower)) = authed_info else {
        let rt = state.rt.lock().unwrap();
        send_to_conn(
            &rt,
            conn_id,
            &json!({ "type": "auth_error", "error": "Please sign in first." }),
        );
        return;
    };

    match msg_type {
        // ---- SQUADS ----
        "squad_create" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            if rt.conns.get(&conn_id).and_then(|c| c.squad_code.as_ref()).is_some() {
                leave_squad(&mut rt, &db, conn_id);
            }
            let mut code = random_code(4, "SQ");
            while rt.squads.contains_key(&code) {
                code = random_code(4, "SQ");
            }
            let mut members = IndexMap::new();
            members.insert(
                user_id.clone(),
                SquadMember {
                    user_id: user_id.clone(),
                    username,
                    username_lower,
                },
            );
            let sq = Squad {
                code: code.clone(),
                leader_id: user_id,
                members,
            };
            let snap = squad_snapshot(&rt, &db, &sq);
            rt.squads.insert(code.clone(), sq);
            if let Some(c) = rt.conns.get_mut(&conn_id) {
                c.squad_code = Some(code);
            }
            send_to_conn(&rt, conn_id, &json!({ "type": "squad_update", "squad": snap }));
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "squad_join" => {
            let code = msg
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_uppercase();
            let local_exists = {
                let rt = state.rt.lock().unwrap();
                rt.squads.contains_key(&code)
            };
            if !local_exists
                && let Some(target_region) =
                    crate::cluster::find_remote_code_region(state, "squad", &code).await
            {
                let rt = state.rt.lock().unwrap();
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({
                        "type": "region_redirect",
                        "region": target_region,
                        "action": "squad_join",
                        "code": code
                    }),
                );
                return;
            }
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(sq_len) = rt.squads.get(&code).map(|s| s.members.len()) else {
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({ "type": "error", "text": format!("Squad \"{code}\" not found.") }),
                );
                return;
            };
            if sq_len >= 5 {
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({ "type": "error", "text": "Squad is full (max 5)." }),
                );
                return;
            }
            let prev_sq = rt.conns.get(&conn_id).and_then(|c| c.squad_code.clone());
            if prev_sq.as_deref().is_some_and(|s| s != code) {
                leave_squad(&mut rt, &db, conn_id);
            }
            if let Some(sq) = rt.squads.get_mut(&code) {
                sq.members.insert(
                    user_id.clone(),
                    SquadMember {
                        user_id: user_id.clone(),
                        username,
                        username_lower,
                    },
                );
            }
            if let Some(c) = rt.conns.get_mut(&conn_id) {
                c.squad_code = Some(code.clone());
            }
            let leader_id = if let Some(sq) = rt.squads.get(&code) {
                let upd = json!({ "type": "squad_update", "squad": squad_snapshot(&rt, &db, sq) });
                broadcast_squad(&rt, sq, &upd);
                sq.leader_id.clone()
            } else {
                return;
            };
            if let Some(&leader_cid) = rt.clients_by_user_id.get(&leader_id)
                && let Some(lcode) = rt.conns.get(&leader_cid).and_then(|c| c.lobby_code.clone()) {
                    let pull_team = rt.lobbies.get(&lcode).and_then(|lb| {
                        if lb.status == LobbyStatus::Waiting {
                            Some(lb.players.get(&leader_id).map(|p| p.team))
                        } else {
                            None
                        }
                    });
                    if let Some(pref) = pull_team {
                        join_lobby_internal(&mut rt, &db, conn_id, &lcode, pref);
                    }
                }
        }

        "squad_leave" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            leave_squad(&mut rt, &db, conn_id);
            send_to_conn(
                &rt,
                conn_id,
                &json!({ "type": "squad_update", "squad": Value::Null }),
            );
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "squad_kick" => {
            let target_id = msg.get("userId").and_then(|v| v.as_str()).unwrap_or("");
            if target_id.is_empty() || target_id == user_id {
                return;
            }
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(sq_code) = rt.conns.get(&conn_id).and_then(|c| c.squad_code.clone()) else {
                return;
            };
            let allowed = rt
                .squads
                .get(&sq_code)
                .map(|s| s.leader_id == user_id && s.members.contains_key(target_id))
                .unwrap_or(false);
            if !allowed {
                return;
            }
            if let Some(&target_cid) = rt.clients_by_user_id.get(target_id)
                && rt
                    .conns
                    .get(&target_cid)
                    .is_some_and(|c| c.squad_code.as_deref() == Some(sq_code.as_str()))
            {
                leave_squad(&mut rt, &db, target_cid);
                send_to_conn(
                    &rt,
                    target_cid,
                    &json!({ "type": "squad_update", "squad": Value::Null }),
                );
                send_to_conn(
                    &rt,
                    target_cid,
                    &json!({ "type": "toast", "text": "YOU WERE REMOVED FROM THE SQUAD" }),
                );
            } else if let Some(sq) = rt.squads.get_mut(&sq_code) {
                sq.members.shift_remove(target_id);
                if let Some(sq_ref) = rt.squads.get(&sq_code) {
                    let upd =
                        json!({ "type": "squad_update", "squad": squad_snapshot(&rt, &db, sq_ref) });
                    broadcast_squad(&rt, sq_ref, &upd);
                }
            }
        }

        // ---- LOBBIES ----
        "lobby_list" => {
            let list = crate::cluster::global_public_lobbies(state);
            let online = crate::cluster::global_online_count(state);
            let rt = state.rt.lock().unwrap();
            send_to_conn(
                &rt,
                conn_id,
                &json!({
                    "type": "lobby_list",
                    "lobbies": list,
                    "online": online
                }),
            );
        }

        "lobby_create" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let me = db.data.users.get(&username_lower);
            if ac_ban_remaining_ms(me) > 0 {
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({ "type": "error", "text": ac_ban_message(me) }),
                );
                return;
            }
            stop_solo_match(&mut rt, conn_id);
            if rt.conns.get(&conn_id).and_then(|c| c.lobby_code.as_ref()).is_some() {
                leave_lobby(&mut rt, &db, conn_id, "left");
            }
            let mut code = random_code(6, "");
            while rt.lobbies.contains_key(&code) {
                code = random_code(6, "");
            }
            let settings = sanitize_lobby_settings(
                msg.get("settings"),
                &default_lobby_settings(&username),
            );
            rt.lobbies.insert(
                code.clone(),
                Lobby {
                    code: code.clone(),
                    host_id: user_id,
                    status: LobbyStatus::Waiting,
                    settings,
                    players: IndexMap::new(),
                    ac: HashMap::new(),
                    active_match: None,
                    sim: None,
                    next_match_gen: 1,
                },
            );
            join_lobby_internal(&mut rt, &db, conn_id, &code, Some(0));
            pull_squad_into_lobby(&mut rt, &db, conn_id, &code, Some(0));
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "lobby_join" => {
            let code = msg
                .get("code")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_uppercase();
            let local_exists = {
                let rt = state.rt.lock().unwrap();
                rt.lobbies.contains_key(&code)
            };
            if !local_exists
                && let Some(target_region) =
                    crate::cluster::find_remote_code_region(state, "lobby", &code).await
            {
                let rt = state.rt.lock().unwrap();
                let redirect = json!({
                    "type": "region_redirect",
                    "region": target_region,
                    "action": "lobby_join",
                    "code": code
                });
                send_to_conn(&rt, conn_id, &redirect);
                // If this player is a squad leader, also redirect their local squadmates!
                if let Some(sq_code) = rt.conns.get(&conn_id).and_then(|c| c.squad_code.as_deref())
                    && let Some(sq) = rt.squads.get(sq_code)
                    && sq.leader_id == user_id
                {
                    for m in sq.members.values() {
                        if m.user_id != user_id
                            && let Some(&mcid) = rt.clients_by_user_id.get(&m.user_id)
                        {
                            send_to_conn(&rt, mcid, &redirect);
                        }
                    }
                }
                return;
            }
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(status) = rt.lobbies.get(&code).map(|lb| lb.status) else {
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({ "type": "error", "text": format!("Lobby \"{code}\" not found.") }),
                );
                return;
            };
            if status == LobbyStatus::InMatch {
                send_to_conn(
                    &rt,
                    conn_id,
                    &json!({ "type": "error", "text": "Match is already in progress." }),
                );
                return;
            }
            stop_solo_match(&mut rt, conn_id);
            join_lobby_internal(&mut rt, &db, conn_id, &code, None);
            let my_team = rt
                .lobbies
                .get(&code)
                .and_then(|lb| lb.players.get(&user_id))
                .map(|p| p.team);
            pull_squad_into_lobby(&mut rt, &db, conn_id, &code, my_team);
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "lobby_leave" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            leave_lobby(&mut rt, &db, conn_id, "left");
            send_to_conn(&rt, conn_id, &json!({ "type": "lobby_left" }));
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "lobby_settings" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                return;
            };
            if let Some(lb) = rt.lobbies.get_mut(&code) {
                if lb.host_id != user_id || lb.status != LobbyStatus::Waiting {
                    return;
                }
                lb.settings = sanitize_lobby_settings(msg.get("settings"), &lb.settings);
                let mut b = 0;
                let mut o = 0;
                let max = lb.settings.team_size;
                for p in lb.players.values_mut() {
                    if p.team == 0 {
                        b += 1;
                        if b > max {
                            p.team = -1;
                        }
                    } else if p.team == 1 {
                        o += 1;
                        if o > max {
                            p.team = -1;
                        }
                    }
                }
            }
            if let Some(lb) = rt.lobbies.get(&code) {
                let upd = json!({ "type": "lobby_update", "lobby": lobby_snapshot(&db, lb) });
                broadcast_lobby(&rt, lb, &upd, None);
            }
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "lobby_team" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                return;
            };
            let want_res: Result<i32, &'static str> = {
                let Some(lb) = rt.lobbies.get(&code) else {
                    return;
                };
                if lb.status != LobbyStatus::Waiting || !lb.players.contains_key(&user_id) {
                    return;
                }
                if msg.get("team").and_then(|v| v.as_str()) == Some("random") {
                    let mut blue = 0;
                    let mut orange = 0;
                    for pl in lb.players.values() {
                        if pl.user_id == user_id {
                            continue;
                        }
                        if pl.team == 0 {
                            blue += 1;
                        } else if pl.team == 1 {
                            orange += 1;
                        }
                    }
                    let mut avail = Vec::new();
                    if blue < lb.settings.team_size {
                        avail.push(0);
                    }
                    if orange < lb.settings.team_size {
                        avail.push(1);
                    }
                    if avail.is_empty() {
                        Err("Both teams are full.")
                    } else {
                        let idx = rand::rngs::OsRng.gen_range(0..avail.len());
                        Ok(avail[idx])
                    }
                } else {
                    let Some(want) = json_strict_int(msg.get("team")).map(|n| n as i32) else {
                        return;
                    };
                    if !matches!(want, -1..=1) {
                        return;
                    }
                    if want == 0 || want == 1 {
                        let count = lb
                            .players
                            .values()
                            .filter(|pl| pl.team == want && pl.user_id != user_id)
                            .count() as i32;
                        if count >= lb.settings.team_size {
                            Err("That team is full.")
                        } else {
                            Ok(want)
                        }
                    } else {
                        Ok(want)
                    }
                }
            };
            match want_res {
                Err(err_text) => {
                    send_to_conn(&rt, conn_id, &json!({ "type": "error", "text": err_text }));
                }
                Ok(want) => {
                    if let Some(lb) = rt.lobbies.get_mut(&code)
                        && let Some(p) = lb.players.get_mut(&user_id) {
                            p.team = want;
                        }
                    if let Some(lb) = rt.lobbies.get(&code) {
                        let upd =
                            json!({ "type": "lobby_update", "lobby": lobby_snapshot(&db, lb) });
                        broadcast_lobby(&rt, lb, &upd, None);
                    }
                }
            }
        }

        "lobby_ready" => {
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                return;
            };
            if let Some(lb) = rt.lobbies.get_mut(&code) {
                if lb.status != LobbyStatus::Waiting {
                    return;
                }
                if let Some(p) = lb.players.get_mut(&user_id) {
                    p.ready = !p.ready;
                } else {
                    return;
                }
            }
            if let Some(lb) = rt.lobbies.get(&code) {
                let upd = json!({ "type": "lobby_update", "lobby": lobby_snapshot(&db, lb) });
                broadcast_lobby(&rt, lb, &upd, None);
            }
            drop(db);
            drop(rt);
            state.cluster.notify_changed();
        }

        "lobby_kick" => {
            let target_id = msg.get("userId").and_then(|v| v.as_str()).unwrap_or("");
            if target_id.is_empty() || target_id == user_id {
                return;
            }
            let mut rt = state.rt.lock().unwrap();
            let db = state.db.lock().unwrap();
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                return;
            };
            let allowed = rt
                .lobbies
                .get(&code)
                .map(|lb| {
                    lb.host_id == user_id
                        && lb.status == LobbyStatus::Waiting
                        && lb.players.contains_key(target_id)
                })
                .unwrap_or(false);
            if !allowed {
                return;
            }
            if let Some(&target_cid) = rt.clients_by_user_id.get(target_id)
                && rt
                    .conns
                    .get(&target_cid)
                    .is_some_and(|c| c.lobby_code.as_deref() == Some(code.as_str()))
            {
                leave_lobby(&mut rt, &db, target_cid, "kicked");
                send_to_conn(
                    &rt,
                    target_cid,
                    &json!({ "type": "lobby_left", "kicked": true }),
                );
                send_to_conn(
                    &rt,
                    target_cid,
                    &json!({ "type": "toast", "text": "YOU WERE KICKED FROM THE LOBBY" }),
                );
            }
        }

        "lobby_chat" => {
            let rt = state.rt.lock().unwrap();
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.as_deref()) else {
                return;
            };
            let Some(lb) = rt.lobbies.get(code) else {
                return;
            };
            chat_times.retain(|&t| now - t < 10_000);
            if chat_times.len() >= 6 {
                return;
            }
            chat_times.push(now);
            let text: String = msg
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .chars()
                .take(160)
                .collect();
            if text.is_empty() {
                return;
            }
            broadcast_lobby(
                &rt,
                lb,
                &json!({ "type": "lobby_chat", "from": username, "text": text }),
                None,
            );
        }

        "lobby_start" => {
            let start_prep = {
                let mut rt = state.rt.lock().unwrap();
                let db = state.db.lock().unwrap();
                let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                    return;
                };
                let active_players: Vec<(String, String, String, i32)> = {
                    let Some(lb) = rt.lobbies.get(&code) else {
                        return;
                    };
                    if lb.host_id != user_id || lb.status != LobbyStatus::Waiting {
                        return;
                    }
                    lb.players
                        .values()
                        .filter(|p| p.team == 0 || p.team == 1)
                        .map(|p| {
                            (
                                p.user_id.clone(),
                                p.username.clone(),
                                p.username_lower.clone(),
                                p.team,
                            )
                        })
                        .collect()
                };
                if active_players.is_empty() {
                    send_to_conn(
                        &rt,
                        conn_id,
                        &json!({ "type": "error", "text": "At least one player must be on Blue or Orange." }),
                    );
                    return;
                }
                let participants: Vec<MatchParticipant> = active_players
                    .iter()
                    .map(|(uid, _, ulower, team)| {
                        let ip = rt
                            .clients_by_user_id
                            .get(uid)
                            .and_then(|cid| rt.conns.get(cid))
                            .map(|c| c.ip.clone())
                            .unwrap_or_else(|| format!("unknown:{uid}"));
                        MatchParticipant {
                            user_id: uid.clone(),
                            username_lower: ulower.clone(),
                            team: *team,
                            ip,
                        }
                    })
                    .collect();

                let match_players: Vec<MatchPlayerSpec> = active_players
                    .iter()
                    .map(|(uid, uname, ulower, team)| {
                        let car = db
                            .data
                            .users
                            .get(ulower)
                            .map(|u| u.car.clone())
                            .unwrap_or_default();
                        MatchPlayerSpec {
                            user_id: uid.clone(),
                            username: uname.clone(),
                            team: *team,
                            car,
                        }
                    })
                    .collect();

                let (match_gen, settings) = {
                    let lb = rt.lobbies.get_mut(&code).unwrap();
                    lb.status = LobbyStatus::InMatch;
                    lb.ac.clear();
                    let mgen = lb.next_match_gen;
                    lb.next_match_gen += 1;
                    lb.active_match = Some(ActiveMatchMeta {
                        match_gen: mgen,
                        started_at: now,
                        finished: false,
                        participants,
                    });
                    let mut s = lb.settings.clone();
                    s.arena = resolve_arena(s.arena);
                    (mgen, s)
                };

                (code, match_gen, settings, match_players)
            };
            state.cluster.notify_changed();

            let (code, match_gen, settings, match_players) = start_prep;
            let state_clone = Arc::clone(state);
            let sink = Arc::new(LobbyEventSink {
                state: Arc::clone(state),
                lobby_code: code.clone(),
                match_gen,
            });
            let fut = state.backend.create_match(MatchCreateOptions {
                settings: settings.clone(),
                teammate_diff: None,
                afk_enabled: true,
                afk_warn_s: Some(state.config.afk_warn_secs),
                afk_kick_s: Some(state.config.afk_kick_secs),
                players: match_players.clone(),
                callbacks: sink,
            });
            tokio::spawn(async move {
                match fut.await {
                    Ok(sim) => {
                        let mut rt = state_clone.rt.lock().unwrap();
                        let db = state_clone.db.lock().unwrap();
                        let still_valid = rt
                            .lobbies
                            .get(&code)
                            .map(|lb| {
                                lb.status == LobbyStatus::InMatch
                                    && lb
                                        .active_match
                                        .as_ref()
                                        .map(|m| m.match_gen == match_gen && !m.finished)
                                        .unwrap_or(false)
                            })
                            .unwrap_or(false);
                        if !still_valid {
                            sim.stop();
                            return;
                        }
                        let (present_ids, snap, host_id) = {
                            let lb = rt.lobbies.get_mut(&code).unwrap();
                            lb.sim = Some(Arc::clone(&sim));
                            let mut present = Vec::new();
                            for mp in &match_players {
                                if lb.players.contains_key(&mp.user_id) {
                                    present.push(mp.user_id.clone());
                                } else {
                                    sim.remove_player(&mp.user_id, "left");
                                }
                            }
                            let snap = lobby_snapshot(&db, lb);
                            (present, snap, lb.host_id.clone())
                        };
                        if let Some(lb) = rt.lobbies.get(&code) {
                            let players_json = snap.get("players").cloned().unwrap_or(json!([]));
                            broadcast_lobby(
                                &rt,
                                lb,
                                &json!({
                                    "type": "match_start",
                                    "lobby": snap,
                                    "hostId": host_id,
                                    "settings": settings,
                                    "players": players_json,
                                    "cars": sim.car_info(),
                                    "mid": sim.id(),
                                    "serverSim": true
                                }),
                                None,
                            );
                        }
                        sim.start_when_loaded(present_ids);
                    }
                    Err(err) => {
                        eprintln!("[SIM] Failed to start match: {err}");
                        let mut rt = state_clone.rt.lock().unwrap();
                        let should_err = if let Some(lb) = rt.lobbies.get_mut(&code) {
                            let matches = lb
                                .active_match
                                .as_ref()
                                .map(|m| m.match_gen == match_gen && !m.finished)
                                .unwrap_or(false);
                            if matches {
                                lb.status = LobbyStatus::Waiting;
                                lb.active_match = None;
                                lb.sim = None;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if should_err
                            && let Some(lb) = rt.lobbies.get(&code) {
                                broadcast_lobby(
                                    &rt,
                                    lb,
                                    &json!({ "type": "error", "text": "Server could not start the match." }),
                                    None,
                                );
                            }
                    }
                }
            });
        }

        // ---- SOLO / AI / TOURNAMENT ----
        "solo_start" => {
            let solo_prep = {
                let mut rt = state.rt.lock().unwrap();
                let db = state.db.lock().unwrap();
                let me = db.data.users.get(&username_lower);
                if ac_ban_remaining_ms(me) > 0 {
                    send_to_conn(
                        &rt,
                        conn_id,
                        &json!({ "type": "error", "text": ac_ban_message(me) }),
                    );
                    return;
                }
                {
                    let mut rl = state.rate_limiter.lock().unwrap();
                    if !rl.check(&format!("sstart:{user_id}"), 20, 60_000) {
                        send_to_conn(
                            &rt,
                            conn_id,
                            &json!({ "type": "error", "text": "Too many matches started. Please wait a moment." }),
                        );
                        return;
                    }
                }
                if let Some(lcode) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) {
                    if rt
                        .lobbies
                        .get(&lcode)
                        .map(|lb| lb.status == LobbyStatus::InMatch)
                        .unwrap_or(false)
                    {
                        send_to_conn(
                            &rt,
                            conn_id,
                            &json!({ "type": "error", "text": "Finish or leave your multiplayer match first." }),
                        );
                        return;
                    }
                    leave_lobby(&mut rt, &db, conn_id, "left");
                }
                stop_solo_match(&mut rt, conn_id);
                let solo_gen = rt.conns.get(&conn_id).map(|c| c.solo_gen).unwrap_or(0);
                let car = me.map(|u| u.car.clone()).unwrap_or_default();
                (solo_gen, car)
            };

            let (solo_gen, car) = solo_prep;
            let raw = msg.get("cfg").and_then(|v| v.as_object());
            let tournament = raw
                .and_then(|o| o.get("tournament"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let team_size = raw
                .and_then(|o| json_strict_int(o.get("teamSize")))
                .filter(|&n| (1..=5).contains(&n))
                .map(|n| n as i32)
                .unwrap_or(2);
            let difficulty = raw
                .and_then(|o| json_strict_int(o.get("difficulty")))
                .filter(|&n| (0..=5).contains(&n))
                .map(|n| n as i32)
                .unwrap_or(1);
            let duration = if tournament {
                180
            } else {
                raw.and_then(|o| json_strict_int(o.get("duration")))
                    .filter(|n| matches!(n, 0 | 120 | 180 | 300))
                    .map(|n| n as i32)
                    .unwrap_or(300)
            };
            let arena = raw
                .and_then(|o| json_strict_int(o.get("arena")))
                .filter(|&n| (0..i64::from(ARENA_COUNT)).contains(&n))
                .map(|n| n as i32)
                .unwrap_or_else(|| rand::rngs::OsRng.gen_range(0..STANDARD_ARENA_COUNT));
            let layout = raw
                .and_then(|o| o.get("layout"))
                .and_then(|v| v.as_str())
                .filter(|s| is_valid_layout(s))
                .unwrap_or("default")
                .to_string();
            let mode = raw
                .and_then(|o| o.get("mode"))
                .and_then(|v| v.as_str())
                .filter(|s| is_valid_mode(s))
                .unwrap_or("standard")
                .to_string();
            let interval = raw
                .and_then(|o| json_strict_int(o.get("interval")))
                .filter(|n| matches!(n, 6 | 10 | 15))
                .map(|n| n as i32)
                .unwrap_or(10);
            let player_team = if tournament {
                0
            } else {
                match raw.and_then(|o| json_strict_int(o.get("side"))) {
                    Some(0) => 0,
                    Some(1) => 1,
                    Some(-1) => -1,
                    _ => rand::rngs::OsRng.gen_range(0..=1),
                }
            };
            let team_names: [String; 2] = raw
                .and_then(|o| o.get("teamNames"))
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 2)
                .map(|a| {
                    let t0: String = a[0]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("BLUE")
                        .chars()
                        .take(24)
                        .collect::<String>()
                        .to_uppercase();
                    let t1: String = a[1]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .unwrap_or("ORANGE")
                        .chars()
                        .take(24)
                        .collect::<String>()
                        .to_uppercase();
                    [t0, t1]
                })
                .unwrap_or_else(|| ["BLUE".to_string(), "ORANGE".to_string()]);

            let settings = LobbySettings {
                name: String::new(),
                is_public: false,
                ranked: false,
                team_size,
                mode,
                interval,
                arena,
                layout,
                duration,
                bot_fill: true,
                bot_diff: difficulty,
                boost_mode: "normal".to_string(),
                mutator: "normal".to_string(),
            };
            let teammate_diff = if tournament {
                (difficulty + 1).min(4)
            } else {
                difficulty
            };

            let state_clone = Arc::clone(state);
            let sink = Arc::new(SoloEventSink {
                state: Arc::clone(state),
                conn_id,
                user_id: user_id.clone(),
                solo_gen,
                spectating: player_team < 0,
            });
            let fut = state.backend.create_match(MatchCreateOptions {
                settings: settings.clone(),
                teammate_diff: Some(teammate_diff),
                afk_enabled: false,
                afk_warn_s: None,
                afk_kick_s: None,
                players: vec![MatchPlayerSpec {
                    user_id: user_id.clone(),
                    username,
                    team: player_team,
                    car,
                }],
                callbacks: sink,
            });
            tokio::spawn(async move {
                match fut.await {
                    Ok(sim) => {
                        let mut rt = state_clone.rt.lock().unwrap();
                        let valid = rt
                            .conns
                            .get(&conn_id)
                            .map(|c| c.solo_gen == solo_gen)
                            .unwrap_or(false);
                        if !valid {
                            sim.stop();
                            return;
                        }
                        if let Some(c) = rt.conns.get_mut(&conn_id) {
                            c.solo_sim = Some(Arc::clone(&sim));
                            c.solo_ac = Some(make_ac_rec(now_ms()));
                            c.solo_match = Some(SoloMatchMeta {
                                started_at: now_ms(),
                                duration,
                                difficulty,
                                tournament,
                                player_team,
                                finished: false,
                            });
                        }
                        let settings_json = json!({
                            "teamSize": settings.team_size,
                            "mode": settings.mode,
                            "interval": settings.interval,
                            "arena": settings.arena,
                            "layout": settings.layout,
                            "duration": settings.duration,
                            "botFill": settings.bot_fill,
                            "botDiff": settings.bot_diff,
                            "boostMode": settings.boost_mode,
                            "mutator": settings.mutator
                        });
                        send_to_conn(
                            &rt,
                            conn_id,
                            &json!({
                                "type": "solo_match_start",
                                "settings": settings_json,
                                "cars": sim.car_info(),
                                "teamNames": team_names,
                                "tournament": tournament,
                                "solo": true,
                                "mid": sim.id(),
                                "serverSim": true
                            }),
                        );
                        sim.start_when_loaded(vec![user_id]);
                    }
                    Err(err) => {
                        eprintln!("[SIM] Failed to start solo match: {err}");
                        let rt = state_clone.rt.lock().unwrap();
                        if rt
                            .conns
                            .get(&conn_id)
                            .map(|c| c.solo_gen == solo_gen)
                            .unwrap_or(false)
                        {
                            send_to_conn(
                                &rt,
                                conn_id,
                                &json!({ "type": "error", "text": "Server could not start the match." }),
                            );
                        }
                    }
                }
            });
        }

        "solo_leave" => {
            let mut rt = state.rt.lock().unwrap();
            stop_solo_match(&mut rt, conn_id);
        }

        "solo_pause" => {
            let paused = msg.get("paused").and_then(|v| v.as_bool()).unwrap_or(false);
            let rt = state.rt.lock().unwrap();
            if let Some(sim) = rt.conns.get(&conn_id).and_then(|c| c.solo_sim.as_ref()) {
                sim.set_paused(paused);
            }
        }

        "match_loaded" => {
            let mid = json_strict_int(msg.get("m")).and_then(|v| u64::try_from(v).ok());
            let rt = state.rt.lock().unwrap();
            if let Some(c) = rt.conns.get(&conn_id) {
                if let Some(sim) = c.solo_sim.as_ref()
                    && Some(sim.id()) == mid {
                        sim.mark_loaded(&user_id);
                        return;
                    }
                if let Some(code) = c.lobby_code.as_deref()
                    && let Some(sim) = rt.lobbies.get(code).and_then(|lb| lb.sim.as_ref())
                        && Some(sim.id()) == mid {
                            sim.mark_loaded(&user_id);
                        }
            }
        }

        "replay_done" => {
            let mid = json_strict_int(msg.get("m")).and_then(|v| u64::try_from(v).ok());
            let rt = state.rt.lock().unwrap();
            if let Some(c) = rt.conns.get(&conn_id) {
                if let Some(sim) = c.solo_sim.as_ref()
                    && Some(sim.id()) == mid {
                        sim.replay_done(&user_id);
                        return;
                    }
                if let Some(code) = c.lobby_code.as_deref()
                    && let Some(sim) = rt.lobbies.get(code).and_then(|lb| lb.sim.as_ref())
                        && Some(sim.id()) == mid {
                            sim.replay_done(&user_id);
                        }
            }
        }

        "in" => {
            let in_mid = json_strict_int(msg.get("m")).and_then(|v| u64::try_from(v).ok());
            let seq = json_strict_int(msg.get("s")).unwrap_or(0);
            let packed = msg
                .get("c")
                .and_then(|v| v.as_array())
                .map(|a| a.as_slice())
                .unwrap_or(&[]);

            let mut rt = state.rt.lock().unwrap();
            let mut db = state.db.lock().unwrap();

            // Solo match input
            let solo_sim = rt.conns.get(&conn_id).and_then(|c| c.solo_sim.clone());
            if let Some(sim) = solo_sim {
                if in_mid != Some(sim.id()) {
                    return;
                }
                let (rate_ok, pkt_count) = {
                    let c = rt.conns.get_mut(&conn_id).unwrap();
                    let rec = c.solo_ac.get_or_insert_with(|| make_ac_rec(now));
                    let ok = packet_rate_ok(&mut rec.in_pkt, now, AC_INPUT_MAX_PER_SEC);
                    (ok, rec.in_pkt.pkt_count)
                };
                if !rate_ok {
                    if pkt_count == AC_INPUT_STRIKE_AT {
                        add_solo_strike(state, &mut rt, &mut db, conn_id, "Input packet flood", 2);
                    }
                    return;
                }
                if !sim.push_input(&user_id, seq, packed) {
                    add_solo_strike(state, &mut rt, &mut db, conn_id, "Malformed input packet", 1);
                }
                return;
            }

            // Multiplayer lobby input
            let Some(code) = rt.conns.get(&conn_id).and_then(|c| c.lobby_code.clone()) else {
                return;
            };
            let lb_sim = {
                let Some(lb) = rt.lobbies.get(&code) else {
                    return;
                };
                if lb.status != LobbyStatus::InMatch {
                    return;
                }
                let Some(sim) = lb.sim.clone() else {
                    return;
                };
                if in_mid != Some(sim.id()) {
                    return;
                }
                sim
            };
            let (rate_ok, pkt_count) = {
                let lb = rt.lobbies.get_mut(&code).unwrap();
                let rec = lb.ac.entry(user_id.clone()).or_insert_with(|| make_ac_rec(now));
                let ok = packet_rate_ok(&mut rec.in_pkt, now, AC_INPUT_MAX_PER_SEC);
                (ok, rec.in_pkt.pkt_count)
            };
            if !rate_ok {
                if pkt_count == AC_INPUT_STRIKE_AT {
                    add_lobby_strike(state, &mut rt, &mut db, &code, conn_id, "Input packet flood", 2);
                }
                return;
            }
            if !lb_sim.push_input(&user_id, seq, packed) {
                add_lobby_strike(
                    state,
                    &mut rt,
                    &mut db,
                    &code,
                    conn_id,
                    "Malformed input packet",
                    1,
                );
            }
        }

        _ => {}
    }
}
