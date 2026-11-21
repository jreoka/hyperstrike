use crate::auth::now_ms;
use crate::db::{AcLogEntry, CarCustomization, User};
use indexmap::IndexMap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub const STANDARD_ARENA_COUNT: i32 = 12;
pub const ARENA_COUNT: i32 = 15;

pub const AC_KICK_STRIKES: u32 = 12;
pub const AC_WARN_STRIKES: u32 = 6;
pub const AC_DECAY_MS: i64 = 15_000;
pub const AC_BASE_BAN_MS: i64 = 10 * 60 * 1000;
pub const AC_MAX_BAN_MS: i64 = 7 * 24 * 60 * 60 * 1000;
pub const AC_INPUT_MAX_PER_SEC: u32 = 105;
pub const AC_INPUT_STRIKE_AT: u32 = 135;
pub const SNAP_BACKLOG_MAX: usize = 48 * 1024;

const VALID_LAYOUTS: &[&str] = &[
    "default", "standard", "octagon", "hexagon", "round", "capsule",
];
const VALID_MODES: &[&str] = &["standard", "mayhem"];
const VALID_BOOST: &[&str] = &["normal", "unlimited", "recharge"];
const VALID_MUTATORS: &[&str] = &["normal", "lowgrav", "fastball"];

pub fn is_valid_layout(s: &str) -> bool {
    VALID_LAYOUTS.contains(&s)
}

pub fn is_valid_mode(s: &str) -> bool {
    VALID_MODES.contains(&s)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LobbySettings {
    pub name: String,
    pub is_public: bool,
    pub ranked: bool,
    pub team_size: i32,
    pub mode: String,
    pub interval: i32,
    pub arena: i32,
    pub layout: String,
    pub duration: i32,
    pub bot_fill: bool,
    pub bot_diff: i32,
    pub boost_mode: String,
    pub mutator: String,
}

pub fn default_lobby_settings(host_name: &str) -> LobbySettings {
    LobbySettings {
        name: format!("{host_name}'S LOBBY"),
        is_public: true,
        ranked: false,
        team_size: 2,
        mode: "standard".to_string(),
        interval: 10,
        arena: 0,
        layout: "default".to_string(),
        duration: 300,
        bot_fill: true,
        bot_diff: 1,
        boost_mode: "normal".to_string(),
        mutator: "normal".to_string(),
    }
}

pub fn sanitize_lobby_settings(
    raw: Option<&serde_json::Value>,
    current: &LobbySettings,
) -> LobbySettings {
    let mut s = current.clone();
    let Some(obj) = raw.and_then(|v| v.as_object()) else {
        return s;
    };
    if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
        let trimmed: String = name.trim().chars().take(28).collect();
        let upper = trimmed.to_uppercase();
        if !upper.is_empty() {
            s.name = upper;
        }
    }
    if let Some(b) = obj.get("isPublic").and_then(|v| v.as_bool()) {
        s.is_public = b;
    }
    if let Some(b) = obj.get("ranked").and_then(|v| v.as_bool()) {
        s.ranked = b;
    }
    if let Some(n) = json_strict_int(obj.get("teamSize"))
        && (1..=4).contains(&n) {
            s.team_size = n as i32;
        }
    if let Some(m) = obj.get("mode").and_then(|v| v.as_str())
        && VALID_MODES.contains(&m) {
            s.mode = m.to_string();
        }
    if let Some(n) = json_strict_int(obj.get("interval"))
        && matches!(n, 6 | 10 | 15) {
            s.interval = n as i32;
        }
    if let Some(n) = json_strict_int(obj.get("arena"))
        && (-1..i64::from(ARENA_COUNT)).contains(&n) {
            s.arena = n as i32;
        }
    if let Some(l) = obj.get("layout").and_then(|v| v.as_str())
        && VALID_LAYOUTS.contains(&l) {
            s.layout = l.to_string();
        }
    if let Some(n) = json_strict_int(obj.get("duration"))
        && matches!(n, 0 | 120 | 180 | 300) {
            s.duration = n as i32;
        }
    if let Some(b) = obj.get("botFill").and_then(|v| v.as_bool()) {
        s.bot_fill = b;
    }
    if let Some(n) = json_strict_int(obj.get("botDiff"))
        && (0..=3).contains(&n) {
            s.bot_diff = n as i32;
        }
    if let Some(bm) = obj.get("boostMode").and_then(|v| v.as_str())
        && VALID_BOOST.contains(&bm) {
            s.boost_mode = bm.to_string();
        }
    if let Some(mutator) = obj.get("mutator").and_then(|v| v.as_str())
        && VALID_MUTATORS.contains(&mutator) {
            s.mutator = mutator.to_string();
        }
    // Ranked enforces fair rules
    if s.ranked {
        s.boost_mode = "normal".to_string();
        s.mutator = "normal".to_string();
        if s.duration == 0 {
            s.duration = 300;
        }
    }
    s
}

pub fn json_strict_int(v: Option<&serde_json::Value>) -> Option<i64> {
    let n = v?.as_number()?;
    if let Some(i) = n.as_i64() {
        return Some(i);
    }
    if let Some(f) = n.as_f64()
        && f.is_finite() && f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
            return Some(f as i64);
        }
    None
}

pub fn resolve_arena(arena: i32) -> i32 {
    if arena == -1 {
        rand::rngs::OsRng.gen_range(0..STANDARD_ARENA_COUNT)
    } else {
        arena
    }
}

pub fn is_valid_packed_controls(arr: &[serde_json::Value]) -> bool {
    if arr.len() != 6 {
        return false;
    }
    for item in &arr[..5] {
        let Some(v) = item.as_f64() else {
            return false;
        };
        if !v.is_finite() || !(-1.001..=1.001).contains(&v) {
            return false;
        }
    }
    match json_strict_int(arr.get(5)) {
        Some(bits) => (0..=31).contains(&bits),
        None => false,
    }
}

// ============================================================
// SQUADS & LOBBIES
// ============================================================

#[derive(Debug, Clone)]
pub struct SquadMember {
    pub user_id: String,
    pub username: String,
    pub username_lower: String,
}

#[derive(Debug, Clone)]
pub struct Squad {
    pub code: String,
    pub leader_id: String,
    pub members: IndexMap<String, SquadMember>,
}

#[derive(Debug, Clone)]
pub struct LobbyPlayer {
    pub user_id: String,
    pub username: String,
    pub username_lower: String,
    pub team: i32,
    pub ready: bool,
    pub squad_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MatchParticipant {
    pub user_id: String,
    pub username_lower: String,
    pub team: i32,
    pub ip: String,
}

#[derive(Debug, Clone)]
pub struct ActiveMatchMeta {
    pub match_gen: u64,
    pub started_at: i64,
    pub finished: bool,
    pub participants: Vec<MatchParticipant>,
}

#[derive(Debug, Clone)]
pub struct SoloMatchMeta {
    pub started_at: i64,
    pub duration: i32,
    pub difficulty: i32,
    pub tournament: bool,
    pub player_team: i32,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobbyStatus {
    Waiting,
    InMatch,
}

impl LobbyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            LobbyStatus::Waiting => "waiting",
            LobbyStatus::InMatch => "in_match",
        }
    }
}

pub struct Lobby {
    pub code: String,
    pub host_id: String,
    pub status: LobbyStatus,
    pub settings: LobbySettings,
    pub players: IndexMap<String, LobbyPlayer>,
    pub ac: HashMap<String, AcRecord>,
    pub active_match: Option<ActiveMatchMeta>,
    pub sim: Option<Arc<dyn MatchHandle>>,
    pub next_match_gen: u64,
}

impl Lobby {
    pub fn pick_team(&self, preferred_team: Option<i32>) -> i32 {
        let max_per_team = self.settings.team_size;
        let mut blue = 0;
        let mut orange = 0;
        for p in self.players.values() {
            if p.team == 0 {
                blue += 1;
            } else if p.team == 1 {
                orange += 1;
            }
        }
        if preferred_team == Some(0) && blue < max_per_team {
            return 0;
        }
        if preferred_team == Some(1) && orange < max_per_team {
            return 1;
        }
        if blue <= orange && blue < max_per_team {
            return 0;
        }
        if orange < max_per_team {
            return 1;
        }
        if blue < max_per_team {
            return 0;
        }
        -1
    }
}

// ============================================================
// ANTI-CHEAT TRACKER
// ============================================================

#[derive(Debug, Clone)]
pub struct AcPacketWindow {
    pub pkt_window: i64,
    pub pkt_count: u32,
}

#[derive(Debug, Clone)]
pub struct AcRecord {
    pub strikes: u32,
    pub last_decay: i64,
    pub in_pkt: AcPacketWindow,
}

pub fn make_ac_rec(now: i64) -> AcRecord {
    AcRecord {
        strikes: 0,
        last_decay: now,
        in_pkt: AcPacketWindow {
            pkt_window: now,
            pkt_count: 0,
        },
    }
}

pub fn decay_strikes(rec: &mut AcRecord, now: i64) {
    if rec.strikes > 0 && now - rec.last_decay >= AC_DECAY_MS {
        let steps = ((now - rec.last_decay) / AC_DECAY_MS) as u32;
        rec.strikes = rec.strikes.saturating_sub(steps);
        rec.last_decay += i64::from(steps) * AC_DECAY_MS;
    } else if rec.strikes == 0 {
        rec.last_decay = now;
    }
}

pub fn packet_rate_ok(win: &mut AcPacketWindow, now: i64, max_per_sec: u32) -> bool {
    if now - win.pkt_window >= 1000 {
        win.pkt_window = now;
        win.pkt_count = 0;
    }
    win.pkt_count = win.pkt_count.saturating_add(1);
    win.pkt_count <= max_per_sec
}

pub fn ac_ban_remaining_ms(user: Option<&User>) -> i64 {
    let Some(until) = user.and_then(|u| u.ac_ban_until) else {
        return 0;
    };
    (until - now_ms()).max(0)
}

pub fn ac_ban_message(user: Option<&User>) -> String {
    let rem = ac_ban_remaining_ms(user);
    let mins = (rem as f64 / 60_000.0).ceil() as i64;
    if mins >= 60 {
        let hrs = (mins as f64 / 60.0).ceil() as i64;
        format!("Anti-cheat matchmaking ban: {hrs}h remaining.")
    } else {
        format!("Anti-cheat matchmaking ban: {mins}m remaining.")
    }
}

pub fn ac_log(user: &mut User, reason: &str) {
    eprintln!("[ANTI-CHEAT] {}: {reason}", user.username);
    let trimmed: String = reason.chars().take(120).collect();
    user.ac_log.push(AcLogEntry {
        t: now_ms(),
        reason: trimmed,
    });
    if user.ac_log.len() > 25 {
        let excess = user.ac_log.len() - 25;
        user.ac_log.drain(0..excess);
    }
}

pub fn ac_penalize(user: &mut User, reason: &str) {
    user.anti_cheat_flags = user.anti_cheat_flags.saturating_add(1);
    let shift = user.anti_cheat_flags.saturating_sub(1).min(30);
    let ms = AC_MAX_BAN_MS.min(AC_BASE_BAN_MS.saturating_mul(1i64 << shift));
    user.ac_ban_until = Some(now_ms() + ms);
    let mins = ((ms as f64) / 60_000.0).round() as i64;
    ac_log(user, &format!("KICK+BAN {mins}m: {reason}"));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankedEligibility {
    Ok,
    Ineligible(&'static str),
}

pub fn ranked_eligibility_from_participants(participants: &[MatchParticipant]) -> RankedEligibility {
    let mut blue_ips: Vec<&str> = Vec::new();
    let mut orange_ips: Vec<&str> = Vec::new();
    for p in participants {
        if p.team == 0 {
            blue_ips.push(p.ip.as_str());
        } else if p.team == 1 {
            orange_ips.push(p.ip.as_str());
        }
    }
    if blue_ips.is_empty() || orange_ips.is_empty() {
        return RankedEligibility::Ineligible("Ranked needs human players on both teams");
    }
    for ip in &blue_ips {
        if orange_ips.contains(ip) {
            return RankedEligibility::Ineligible("Opposing players share a network");
        }
    }
    RankedEligibility::Ok
}

// ============================================================
// MATCH BACKEND TRAIT (PLUGGABLE SIMULATION POOL)
// ============================================================

#[derive(Debug, Clone)]
pub enum SnapshotPayload {
    Binary(Vec<u8>),
    Text(String),
}

impl SnapshotPayload {
    pub fn byte_len(&self) -> usize {
        match self {
            SnapshotPayload::Binary(b) => b.len(),
            SnapshotPayload::Text(s) => s.len(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchStatsEntry {
    pub goals: u64,
    pub assists: u64,
    pub saves: u64,
    pub shots: u64,
    pub demos: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchPlayerSpec {
    pub user_id: String,
    pub username: String,
    pub team: i32,
    pub car: CarCustomization,
}

pub trait MatchEventSink: Send + Sync {
    fn broadcast(&self, msg: serde_json::Value);
    fn send_raw(&self, user_id: &str, payload: SnapshotPayload);
    fn send_spectators(&self, payload: SnapshotPayload);
    fn on_goal(&self, info: serde_json::Value);
    fn on_end(&self, score: [i32; 2], stats: HashMap<String, MatchStatsEntry>);
    fn on_error(&self, message: String);
}

pub struct MatchCreateOptions {
    pub settings: LobbySettings,
    pub teammate_diff: Option<i32>,
    pub players: Vec<MatchPlayerSpec>,
    pub callbacks: Arc<dyn MatchEventSink>,
}

pub trait MatchHandle: Send + Sync {
    fn id(&self) -> u64;
    fn created_at(&self) -> i64;
    fn car_info(&self) -> serde_json::Value;
    fn start_when_loaded(&self, user_ids: Vec<String>);
    fn mark_loaded(&self, user_id: &str);
    fn replay_done(&self, user_id: &str);
    fn push_input(&self, user_id: &str, seq: i64, packed: &[serde_json::Value]) -> bool;
    fn remove_player(&self, user_id: &str);
    fn set_paused(&self, paused: bool);
    fn stop(&self);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackendStats {
    pub ready_workers: usize,
    pub active_matches: usize,
}

pub type CreateMatchFuture =
    Pin<Box<dyn Future<Output = Result<Arc<dyn MatchHandle>, String>> + Send>>;

pub trait MatchBackend: Send + Sync {
    fn stats(&self) -> BackendStats;
    fn create_match(&self, opts: MatchCreateOptions) -> CreateMatchFuture;
}

#[derive(Default)]
pub struct NoopMatchBackend;

impl MatchBackend for NoopMatchBackend {
    fn stats(&self) -> BackendStats {
        BackendStats::default()
    }

    fn create_match(&self, _opts: MatchCreateOptions) -> CreateMatchFuture {
        Box::pin(async { Err("Simulation backend not initialized".to_string()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_settings_enforces_ranked_rules() {
        let base = default_lobby_settings("Alice");
        let raw = serde_json::json!({
            "name": "  my cool lobby  ",
            "ranked": true,
            "boostMode": "unlimited",
            "mutator": "lowgrav",
            "duration": 0,
            "teamSize": 3
        });
        let s = sanitize_lobby_settings(Some(&raw), &base);
        assert_eq!(s.name, "MY COOL LOBBY");
        assert!(s.ranked);
        assert_eq!(s.boost_mode, "normal");
        assert_eq!(s.mutator, "normal");
        assert_eq!(s.duration, 300);
        assert_eq!(s.team_size, 3);
    }

    #[test]
    fn ranked_eligibility_detects_shared_ip() {
        let same_ip = vec![
            MatchParticipant {
                user_id: "1".into(),
                username_lower: "a".into(),
                team: 0,
                ip: "1.2.3.4".into(),
            },
            MatchParticipant {
                user_id: "2".into(),
                username_lower: "b".into(),
                team: 1,
                ip: "1.2.3.4".into(),
            },
        ];
        assert_eq!(
            ranked_eligibility_from_participants(&same_ip),
            RankedEligibility::Ineligible("Opposing players share a network")
        );

        let ok = vec![
            MatchParticipant {
                user_id: "1".into(),
                username_lower: "a".into(),
                team: 0,
                ip: "1.2.3.4".into(),
            },
            MatchParticipant {
                user_id: "2".into(),
                username_lower: "b".into(),
                team: 1,
                ip: "5.6.7.8".into(),
            },
        ];
        assert_eq!(
            ranked_eligibility_from_participants(&ok),
            RankedEligibility::Ok
        );
    }
}
