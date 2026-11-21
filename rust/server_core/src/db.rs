use crate::auth::{SESSION_TTL_MS, now_ms, random_hex};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================
// PROGRESSION: INFINITE LEVELING & RANKS
// ============================================================

pub fn xp_for_level(level: u64) -> u64 {
    let l = level.max(1) as f64;
    (250.0 + (l - 1.0) * 140.0 + 20.0 * (l - 1.0).powf(1.35)).floor() as u64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct XpGainResult {
    pub xp_gained: u64,
    pub old_level: u64,
    pub new_level: u64,
    pub leveled_up: bool,
}

pub fn add_xp(user: &mut User, amount: u64) -> XpGainResult {
    let old_level = user.level.max(1);
    user.xp = user.xp.saturating_add(amount);
    user.total_xp = user.total_xp.saturating_add(amount);
    user.level = old_level;
    let mut safety = 0u32;
    while user.xp >= xp_for_level(user.level) && safety < 500 {
        user.xp -= xp_for_level(user.level);
        user.level += 1;
        safety += 1;
    }
    XpGainResult {
        xp_gained: amount,
        old_level,
        new_level: user.level,
        leveled_up: user.level > old_level,
    }
}

struct RankTierDef {
    name: &'static str,
    min: i64,
    color: &'static str,
}

const RANK_TIERS: &[RankTierDef] = &[
    RankTierDef { name: "BRONZE", min: 0, color: "#cd7f32" },
    RankTierDef { name: "SILVER", min: 300, color: "#c4d0e0" },
    RankTierDef { name: "GOLD", min: 600, color: "#ffd23f" },
    RankTierDef { name: "PLATINUM", min: 900, color: "#4ff2e8" },
    RankTierDef { name: "DIAMOND", min: 1200, color: "#4d9fff" },
    RankTierDef { name: "CHAMPION", min: 1500, color: "#c06bff" },
    RankTierDef { name: "GRAND CHAMPION", min: 1800, color: "#ff4f7a" },
];

const ROMAN: &[&str] = &["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

fn to_roman(n: i64) -> String {
    if (1..=10).contains(&n) {
        ROMAN[n as usize].to_string()
    } else {
        n.to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankInfo {
    pub tier: String,
    pub division: String,
    pub name: String,
    pub short: String,
    pub color: String,
    pub rp: i64,
    pub tier_min: i64,
    pub next_rp: i64,
    pub progress: f64,
}

pub fn compute_rank(rp_raw: i64) -> RankInfo {
    let rp = rp_raw.max(0);
    if rp >= 2100 {
        let over = rp - 2100;
        let legend_tier = (over / 150) + 1;
        let base = 2100 + (legend_tier - 1) * 150;
        let next_rp = base + 150;
        let div = to_roman(legend_tier);
        return RankInfo {
            tier: "HYPERSTRIKE LEGEND".to_string(),
            division: div.clone(),
            name: format!("LEGEND {div}"),
            short: format!("LEG {div}"),
            color: "#ffdf6b".to_string(),
            rp,
            tier_min: base,
            next_rp,
            progress: ((rp - base) as f64 / 150.0).min(1.0),
        };
    }

    let mut tier = &RANK_TIERS[0];
    for t in RANK_TIERS.iter().rev() {
        if rp >= t.min {
            tier = t;
            break;
        }
    }
    let offset = rp - tier.min;
    let div_idx = (offset / 100).min(2) as usize;
    let div_base = tier.min + (div_idx as i64) * 100;
    let div_roman = ["I", "II", "III"][div_idx];
    RankInfo {
        tier: tier.name.to_string(),
        division: div_roman.to_string(),
        name: format!("{} {div_roman}", tier.name),
        short: format!("{} {div_roman}", &tier.name[..3]),
        color: tier.color.to_string(),
        rp,
        tier_min: div_base,
        next_rp: div_base + 100,
        progress: ((rp - div_base) as f64 / 100.0).min(1.0),
    }
}

// ============================================================
// USER, STATS, COSMETICS, PERSISTENCE
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CarCustomization {
    pub body: i32,
    pub primary: i32,
    pub secondary: i32,
    pub finish: i32,
    pub decal: i32,
    pub wheels: i32,
    pub trail: i32,
    pub explosion: i32,
}

impl Default for CarCustomization {
    fn default() -> Self {
        Self {
            body: 0,
            primary: -1,
            secondary: 0,
            finish: 0,
            decal: 0,
            wheels: 0,
            trail: 0,
            explosion: 0,
        }
    }
}

fn js_number(val: Option<&serde_json::Value>) -> Option<f64> {
    match val? {
        serde_json::Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        serde_json::Value::Null => Some(0.0),
        serde_json::Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                Some(0.0)
            } else {
                t.parse::<f64>().ok().filter(|f| f.is_finite())
            }
        }
        _ => None,
    }
}

fn clamp_car_field(val: Option<&serde_json::Value>, min: i32, max: i32, fallback: i32) -> i32 {
    match js_number(val) {
        Some(n) => (n.floor() as i32).clamp(min, max),
        None => fallback,
    }
}

pub fn sanitize_car(raw: Option<&serde_json::Value>) -> CarCustomization {
    let Some(obj) = raw.and_then(|v| v.as_object()) else {
        return CarCustomization::default();
    };
    CarCustomization {
        body: clamp_car_field(obj.get("body"), 0, 4, 0),
        primary: clamp_car_field(obj.get("primary"), -1, 11, -1),
        secondary: clamp_car_field(obj.get("secondary"), 0, 11, 0),
        finish: clamp_car_field(obj.get("finish"), 0, 4, 0),
        decal: clamp_car_field(obj.get("decal"), 0, 6, 0),
        wheels: clamp_car_field(obj.get("wheels"), 0, 5, 0),
        trail: clamp_car_field(obj.get("trail"), 0, 6, 0),
        explosion: clamp_car_field(obj.get("explosion"), 0, 4, 0),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", default)]
pub struct UserStats {
    pub matches: u64,
    pub wins: u64,
    pub losses: u64,
    pub goals: u64,
    pub assists: u64,
    pub saves: u64,
    pub shots: u64,
    pub demos: u64,
    pub mp_matches: u64,
    pub mp_wins: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcLogEntry {
    pub t: i64,
    pub reason: String,
}

fn default_level() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    pub id: String,
    pub username: String,
    pub pass_hash: String,
    pub salt: String,
    pub created_at: i64,
    #[serde(default = "default_level")]
    pub level: u64,
    #[serde(default)]
    pub xp: u64,
    #[serde(default)]
    pub total_xp: u64,
    #[serde(default)]
    pub rp: i64,
    #[serde(default)]
    pub peak_rp: i64,
    #[serde(default)]
    pub stats: UserStats,
    #[serde(default)]
    pub car: CarCustomization,
    #[serde(default)]
    pub anti_cheat_flags: u32,
    #[serde(default)]
    pub last_report_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ac_ban_until: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ac_log: Vec<AcLogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicUser {
    pub id: String,
    pub username: String,
    pub level: u64,
    pub xp: u64,
    pub xp_next: u64,
    pub xp_progress: f64,
    pub total_xp: u64,
    pub rp: i64,
    pub peak_rp: i64,
    pub rank: RankInfo,
    pub stats: UserStats,
    pub car: CarCustomization,
}

pub fn public_user(user: &User) -> PublicUser {
    let level = user.level.max(1);
    let xp = user.xp;
    let xp_next = xp_for_level(level);
    let rp = user.rp;
    let peak_rp = if user.peak_rp == 0 { rp } else { user.peak_rp };
    PublicUser {
        id: user.id.clone(),
        username: user.username.clone(),
        level,
        xp,
        xp_next,
        xp_progress: (xp as f64 / xp_next as f64).min(1.0),
        total_xp: user.total_xp,
        rp,
        peak_rp,
        rank: compute_rank(rp),
        stats: user.stats.clone(),
        car: user.car.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub username_lower: String,
    pub created_at: i64,
    pub expires_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DbData {
    #[serde(default)]
    pub users: IndexMap<String, User>,
    #[serde(default)]
    pub sessions: IndexMap<String, Session>,
}

pub struct Database {
    pub db_file: PathBuf,
    pub data: DbData,
    dirty: bool,
}

impl Database {
    pub fn load(db_file: PathBuf) -> Self {
        if let Some(parent) = db_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut data = DbData::default();
        if db_file.exists() {
            match std::fs::read_to_string(&db_file)
                .and_then(|raw| serde_json::from_str::<DbData>(&raw).map_err(std::io::Error::other))
            {
                Ok(parsed) => {
                    data = parsed;
                    let now = now_ms();
                    let users = &data.users;
                    data.sessions
                        .retain(|_, sess| sess.expires_at >= now && users.contains_key(&sess.username_lower));
                }
                Err(err) => {
                    eprintln!("[DB] Failed to load db.json: {err}");
                }
            }
        }
        Self {
            db_file,
            data,
            dirty: false,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn take_flush_payload(&mut self) -> Option<(PathBuf, String)> {
        if !self.dirty {
            return None;
        }
        match serde_json::to_string(&self.data) {
            Ok(payload) => {
                self.dirty = false;
                Some((self.db_file.clone(), payload))
            }
            Err(err) => {
                eprintln!("[DB] Serialize error: {err}");
                None
            }
        }
    }

    pub fn flush_sync(&mut self) {
        let Some((path, payload)) = self.take_flush_payload() else {
            return;
        };
        if let Err(err) = atomic_write_sync(&path, &payload) {
            eprintln!("[DB] Flush error: {err}");
            self.dirty = true;
        }
    }

    pub fn create_session(&mut self, username_lower: &str) -> String {
        let token = random_hex(32);
        let now = now_ms();
        self.data.sessions.insert(
            token.clone(),
            Session {
                username_lower: username_lower.to_string(),
                created_at: now,
                expires_at: now + SESSION_TTL_MS,
            },
        );
        self.dirty = true;
        token
    }

    pub fn get_username_by_token(&mut self, token: &str) -> Option<String> {
        let sess = self.data.sessions.get(token)?;
        if sess.expires_at < now_ms() {
            self.data.sessions.shift_remove(token);
            self.dirty = true;
            return None;
        }
        let key = sess.username_lower.clone();
        if self.data.users.contains_key(&key) {
            Some(key)
        } else {
            None
        }
    }

    pub fn get_user_by_token(&mut self, token: &str) -> Option<&User> {
        let key = self.get_username_by_token(token)?;
        self.data.users.get(&key)
    }

    pub fn get_user_by_token_mut(&mut self, token: &str) -> Option<&mut User> {
        let key = self.get_username_by_token(token)?;
        self.data.users.get_mut(&key)
    }
}

pub fn atomic_write_sync(path: &std::path::Path, payload: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    std::fs::write(&tmp_path, payload)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub async fn atomic_write_async(path: &std::path::Path, payload: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    tokio::fs::write(&tmp_path, payload).await?;
    tokio::fs::rename(&tmp_path, path).await?;
    Ok(())
}

// ============================================================
// RATE LIMITER (HTTP + WS)
// ============================================================

#[derive(Default)]
pub struct RateLimiter {
    buckets: HashMap<String, (u32, i64)>,
}

impl RateLimiter {
    pub fn check(&mut self, key: &str, max_hits: u32, window_ms: i64) -> bool {
        let now = now_ms();
        let entry = self
            .buckets
            .entry(key.to_string())
            .or_insert((0, now + window_ms));
        if now > entry.1 {
            entry.0 = 0;
            entry.1 = now + window_ms;
        }
        entry.0 = entry.0.saturating_add(1);
        entry.0 <= max_hits
    }

    pub fn cleanup(&mut self) {
        let now = now_ms();
        self.buckets.retain(|_, (_, reset_at)| now <= *reset_at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_db_json_format() {
        let raw = r#"{"users":{"jreoka":{"id":"ec6dfe8a27528dd0","username":"jreoka","passHash":"5d962224dfa4f1898cd19fcd7616c32c0a0431e2824ea1b74ba14b24d8e08a4ee701a61c4ab790f5d1f518e731b0f860b163b7526685bdbf4364215166523373","salt":"c0b619b0a7404ac1ac9427b856382df1","createdAt":1795229058144,"level":1,"xp":0,"totalXp":0,"rp":100,"peakRp":100,"stats":{"matches":0,"wins":0,"losses":0,"goals":0,"assists":0,"saves":0,"shots":0,"demos":0,"mpMatches":0,"mpWins":0},"car":{"body":0,"primary":-1,"secondary":0,"finish":0,"decal":0,"wheels":0,"trail":0,"explosion":0},"antiCheatFlags":0,"lastReportAt":0}},"sessions":{"5dff616aa3e2abcaf8fdc381babbf1d8bcc989f3e044c8d406e7361789c7f510":{"usernameLower":"jreoka","createdAt":1795229058144,"expiresAt":1797821058144}}}"#;
        let parsed: DbData = serde_json::from_str(raw).unwrap();
        let u = parsed.users.get("jreoka").unwrap();
        assert_eq!(u.id, "ec6dfe8a27528dd0");
        assert_eq!(u.rp, 100);
        assert_eq!(u.car.primary, -1);
        let roundtrip = serde_json::to_string(&parsed).unwrap();
        assert_eq!(roundtrip, raw);
    }

    #[test]
    fn rank_and_xp_progression_match_js() {
        assert_eq!(xp_for_level(1), 250);
        assert_eq!(xp_for_level(2), 410);
        assert_eq!(xp_for_level(5), 939);

        let r0 = compute_rank(100);
        assert_eq!(r0.tier, "BRONZE");
        assert_eq!(r0.name, "BRONZE II");
        assert_eq!(r0.short, "BRO II");

        let r_leg = compute_rank(2250);
        assert_eq!(r_leg.tier, "HYPERSTRIKE LEGEND");
        assert_eq!(r_leg.name, "LEGEND II");
        assert_eq!(r_leg.short, "LEG II");
        assert_eq!(r_leg.tier_min, 2250);
        assert_eq!(r_leg.next_rp, 2400);
    }

    #[test]
    fn sanitize_car_clamps_values() {
        let v = serde_json::json!({
            "body": 99,
            "primary": -5,
            "secondary": "7.9",
            "finish": "bad",
            "decal": 3,
            "wheels": -1,
            "trail": 6,
            "explosion": 4
        });
        let c = sanitize_car(Some(&v));
        assert_eq!(c.body, 4);
        assert_eq!(c.primary, -1);
        assert_eq!(c.secondary, 7);
        assert_eq!(c.finish, 0);
        assert_eq!(c.decal, 3);
        assert_eq!(c.wheels, 0);
        assert_eq!(c.trail, 6);
        assert_eq!(c.explosion, 4);
    }
}
