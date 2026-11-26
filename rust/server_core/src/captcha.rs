//! Self-hosted Proof-of-Work Smart Captcha & Browser Integrity Check ("HyperShield").
//!
//! Zero third-party services or extra dependencies: combines an adaptive
//! SHA-256 hashcash puzzle, a server-bound JS/browser execution probe, HTTP
//! header & browser telemetry risk scoring, single-use challenge redemption,
//! and a hidden form honeypot. Verification costs one SHA-256 hash (< 1 µs)
//! before `scrypt` password hashing ever runs.

use crate::auth::{now_ms, random_hex};
use axum::http::{HeaderMap, header};
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Lifetime of an issued captcha challenge (5 minutes).
pub const CHALLENGE_TTL_MS: i64 = 5 * 60 * 1000;
/// Sliding window for per-IP challenge burst & failure tracking (10 minutes).
pub const IP_WINDOW_MS: i64 = 10 * 60 * 1000;
/// Maximum challenges a single IP may request within `IP_WINDOW_MS`.
pub const MAX_CHALLENGES_PER_IP: u32 = 60;
/// Maximum account registrations allowed per IP within `IP_WINDOW_MS`.
pub const MAX_REGISTRATIONS_PER_IP: u32 = 10;
/// Default leading-zero-bit difficulty for low-risk browsers (~131k hashes avg, ~150 ms in JS).
pub const DEFAULT_BASE_DIFFICULTY: u8 = 17;
/// Hard ceiling on in-memory pending challenges.
const MAX_PENDING_CHALLENGES: usize = 50_000;

/// Suspicious non-browser or headless User-Agent markers (lowercase).
const BOT_UA_MARKERS: &[&str] = &[
    "curl/",
    "wget/",
    "python-requests",
    "python-urllib",
    "httpx",
    "aiohttp",
    "go-http-client",
    "scrapy",
    "headlesschrome",
    "headless",
    "phantomjs",
    "playwright",
    "puppeteer",
    "selenium",
    "webdriver",
    "node-fetch",
    "undici",
    "axios/",
    "libwww-perl",
    "okhttp/",
    "java/",
    "postmanruntime",
    "insomnia/",
    "httpie/",
    "bot/",
    "spider",
    "crawler",
    "mechanize",
    "zgrab",
];

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BrowserSignals {
    #[serde(default)]
    pub webdriver: bool,
    #[serde(default)]
    pub automation: bool,
    #[serde(default)]
    pub headless_ua: bool,
    #[serde(default)]
    pub languages: u16,
    #[serde(default)]
    pub screen_w: u32,
    #[serde(default)]
    pub screen_h: u32,
    #[serde(default)]
    pub inner_w: u32,
    #[serde(default)]
    pub inner_h: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub cores: u16,
    #[serde(default)]
    pub canvas_ok: bool,
    #[serde(default)]
    pub layout_ok: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pub worker_ok: bool,
    #[serde(default)]
    pub signals_present: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BrowserProbe {
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CaptchaChallenge {
    pub enabled: bool,
    pub id: String,
    pub salt: String,
    pub action: String,
    pub difficulty: u8,
    pub probe: BrowserProbe,
    pub interactive: bool,
    pub risk: u8,
    pub expires_at: i64,
    pub mac: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CaptchaSolution {
    pub id: String,
    pub nonce: u64,
    pub mac: String,
    #[serde(default)]
    pub probe: u64,
    #[serde(default)]
    pub interacted: bool,
    #[serde(default)]
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
struct PendingChallenge {
    salt: String,
    action: String,
    ip: String,
    ua_hash: String,
    difficulty: u8,
    probe_answer: u64,
    interactive: bool,
    expires_at: i64,
    mac: String,
}

#[derive(Debug, Default, Clone)]
struct IpActivity {
    window_start: i64,
    challenges: u32,
    failures: u32,
    registrations: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptchaError {
    RateLimited(i64),
    Missing,
    Invalid(&'static str),
}

impl CaptchaError {
    pub fn message(&self) -> String {
        match self {
            CaptchaError::RateLimited(wait_secs) => {
                let mins = (*wait_secs + 59) / 60;
                format!(
                    "Too many verification requests. Try again in {} minute{}.",
                    mins.max(1),
                    if mins == 1 { "" } else { "s" }
                )
            }
            CaptchaError::Missing => {
                "Please complete the human verification check and try again.".to_string()
            }
            CaptchaError::Invalid(reason) => (*reason).to_string(),
        }
    }
}

pub struct CaptchaState {
    hmac_key: [u8; 32],
    pending: Mutex<HashMap<String, PendingChallenge>>,
    activity: Mutex<HashMap<String, IpActivity>>,
}

impl Default for CaptchaState {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptchaState {
    pub fn new() -> Self {
        let mut hmac_key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut hmac_key);
        Self {
            hmac_key,
            pending: Mutex::new(HashMap::new()),
            activity: Mutex::new(HashMap::new()),
        }
    }

    pub fn cleanup(&self) {
        let now = now_ms();
        {
            let mut pending = self.pending.lock().unwrap();
            pending.retain(|_, v| v.expires_at > now);
        }
        {
            let mut activity = self.activity.lock().unwrap();
            activity.retain(|_, v| now - v.window_start < IP_WINDOW_MS);
        }
    }

    fn compute_mac(
        &self,
        id: &str,
        salt: &str,
        action: &str,
        difficulty: u8,
        probe: &BrowserProbe,
        interactive: bool,
        expires_at: i64,
    ) -> String {
        let mut mac =
            HmacSha256::new_from_slice(&self.hmac_key).expect("HMAC accepts 32-byte key");
        let payload = format!(
            "{id}|{salt}|{action}|{difficulty}|{}|{}|{}|{}|{expires_at}",
            probe.a,
            probe.b,
            probe.c,
            if interactive { 1 } else { 0 }
        );
        mac.update(payload.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Records a failed verification, honeypot trip, or failed login for this IP
    /// so subsequent challenges escalate in PoW difficulty.
    pub fn record_ip_failure(&self, ip: &str) {
        let ip = ip.trim();
        if ip.is_empty() {
            return;
        }
        let now = now_ms();
        let mut map = self.activity.lock().unwrap();
        let entry = touch_ip_activity(&mut map, ip, now);
        entry.failures = entry.failures.saturating_add(1);
    }

    /// Returns seconds to wait if this IP has exceeded the registration budget.
    pub fn registration_throttled_secs(&self, ip: &str) -> Option<i64> {
        let ip = ip.trim();
        if ip.is_empty() {
            return None;
        }
        let now = now_ms();
        let map = self.activity.lock().unwrap();
        let entry = map.get(ip)?;
        let remaining_ms = entry.window_start + IP_WINDOW_MS - now;
        (remaining_ms > 0 && entry.registrations >= MAX_REGISTRATIONS_PER_IP)
            .then_some((remaining_ms + 999) / 1000)
    }

    /// Records a successful account registration against this IP's sliding window.
    pub fn record_ip_registration(&self, ip: &str) {
        let ip = ip.trim();
        if ip.is_empty() {
            return;
        }
        let now = now_ms();
        let mut map = self.activity.lock().unwrap();
        let entry = touch_ip_activity(&mut map, ip, now);
        entry.registrations = entry.registrations.saturating_add(1);
    }

    /// Issues a fresh single-use Proof-of-Work challenge tailored to the caller's
    /// HTTP headers, IP history, and reported browser integrity signals.
    pub fn issue_challenge(
        &self,
        headers: &HeaderMap,
        ip: &str,
        action_raw: &str,
        signals: &BrowserSignals,
    ) -> Result<CaptchaChallenge, CaptchaError> {
        let action = normalize_action(action_raw);
        let now = now_ms();
        let ip_key = ip.trim();

        let (ip_challenges, ip_failures) = if !ip_key.is_empty() {
            let mut map = self.activity.lock().unwrap();
            let entry = touch_ip_activity(&mut map, ip_key, now);
            if entry.challenges >= MAX_CHALLENGES_PER_IP {
                let remaining_secs = ((entry.window_start + IP_WINDOW_MS - now) + 999) / 1000;
                return Err(CaptchaError::RateLimited(remaining_secs.max(1)));
            }
            entry.challenges = entry.challenges.saturating_add(1);
            (entry.challenges, entry.failures)
        } else {
            (0, 0)
        };

        let risk = evaluate_risk(headers, action, signals, ip_challenges, ip_failures);
        #[cfg(not(test))]
        let base = base_difficulty();
        #[cfg(test)]
        let base = 8u8;
        let (difficulty, interactive) = difficulty_for_risk(risk, base);

        let id = random_hex(16);
        let salt = random_hex(16);
        let mut rng = rand::rngs::OsRng;
        let probe = BrowserProbe {
            a: (rng.next_u32() % 8_000) + 1_000,
            b: (rng.next_u32() % 8_000) + 1_000,
            c: rng.next_u32() & 0x0fff_ffff,
        };
        let probe_answer = expected_probe_answer(&probe);
        let expires_at = now + CHALLENGE_TTL_MS;
        let mac = self.compute_mac(&id, &salt, action, difficulty, &probe, interactive, expires_at);

        let record = PendingChallenge {
            salt: salt.clone(),
            action: action.to_string(),
            ip: ip_key.to_string(),
            ua_hash: ua_hash(headers),
            difficulty,
            probe_answer,
            interactive,
            expires_at,
            mac: mac.clone(),
        };

        {
            let mut pending = self.pending.lock().unwrap();
            if pending.len() >= MAX_PENDING_CHALLENGES {
                pending.retain(|_, v| v.expires_at > now);
                if pending.len() >= MAX_PENDING_CHALLENGES {
                    let drop_count = pending.len() / 4;
                    let keys: Vec<String> = pending.keys().take(drop_count).cloned().collect();
                    for k in keys {
                        pending.remove(&k);
                    }
                }
            }
            pending.insert(id.clone(), record);
        }

        Ok(CaptchaChallenge {
            enabled: true,
            id,
            salt,
            action: action.to_string(),
            difficulty,
            probe,
            interactive,
            risk,
            expires_at,
            mac,
        })
    }

    /// Validates and atomically consumes a single-use captcha solution.
    pub fn verify_solution(
        &self,
        headers: &HeaderMap,
        ip: &str,
        expected_action: &str,
        solution: Option<&CaptchaSolution>,
        honeypot: &str,
    ) -> Result<(), CaptchaError> {
        if !is_enabled() {
            return Ok(());
        }

        let ip_key = ip.trim();
        if !honeypot.trim().is_empty() {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Verification failed. Please reload and try again.",
            ));
        }

        let Some(sol) = solution else {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Missing);
        };

        let id = sol.id.trim();
        if id.is_empty() {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Missing);
        }

        let pending = {
            let mut map = self.pending.lock().unwrap();
            map.remove(id)
        };

        let Some(record) = pending else {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Verification expired or already used. Please try again.",
            ));
        };

        let now = now_ms();
        if now > record.expires_at {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Verification expired. Please verify again.",
            ));
        }

        if !ct_str_eq(&record.mac, sol.mac.trim()) {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Invalid verification signature. Please try again.",
            ));
        }

        if record.action != normalize_action(expected_action) {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Verification did not match this action. Please try again.",
            ));
        }

        if record.ip != ip_key {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Network changed during verification. Please verify again.",
            ));
        }

        if !ct_str_eq(&record.ua_hash, &ua_hash(headers)) {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Browser session mismatch. Please verify again.",
            ));
        }

        if record.interactive && !sol.interacted {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Please click the verification checkbox to continue.",
            ));
        }

        if sol.probe != record.probe_answer {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Browser check failed. Please verify again.",
            ));
        }

        if !pow_matches(
            id,
            &record.salt,
            record.probe_answer,
            sol.nonce,
            record.difficulty,
        ) {
            self.record_ip_failure(ip_key);
            return Err(CaptchaError::Invalid(
                "Proof-of-work verification failed. Please try again.",
            ));
        }

        Ok(())
    }
}

/// Whether the smart PoW captcha is enabled (default: true).
/// Set `HYPERSTRIKE_CAPTCHA=off` (or `0`/`false`) to disable.
pub fn is_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("HYPERSTRIKE_CAPTCHA")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off" | "no" | "disabled"
        )
    })
}

#[cfg_attr(test, allow(dead_code))]
fn base_difficulty() -> u8 {
    static BASE: OnceLock<u8> = OnceLock::new();
    *BASE.get_or_init(|| {
        std::env::var("HYPERSTRIKE_CAPTCHA_DIFFICULTY")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
            .map(|d| d.clamp(8, 22))
            .unwrap_or(DEFAULT_BASE_DIFFICULTY)
    })
}

pub fn normalize_action(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "register" | "signup" => "register",
        _ => "login",
    }
}

pub fn expected_probe_answer(probe: &BrowserProbe) -> u64 {
    let prod = (probe.a as u64).wrapping_mul(probe.b as u64);
    (prod ^ (probe.c as u64)) & 0xffff_ffff
}

pub fn pow_matches(
    id: &str,
    salt: &str,
    probe_answer: u64,
    nonce: u64,
    difficulty: u8,
) -> bool {
    let input = format!("{id}:{salt}:{probe_answer}:{nonce}");
    let digest = Sha256::digest(input.as_bytes());
    has_leading_zero_bits(&digest, difficulty)
}

pub fn has_leading_zero_bits(digest: &[u8], bits: u8) -> bool {
    let bits = (bits as usize).min(digest.len() * 8);
    let full_bytes = bits / 8;
    let rem_bits = bits % 8;
    if digest[..full_bytes].iter().any(|&b| b != 0) {
        return false;
    }
    if rem_bits > 0 {
        let shift = 8 - rem_bits;
        if (digest[full_bytes] >> shift) != 0 {
            return false;
        }
    }
    true
}

fn ct_str_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    ab.len() == bb.len() && bool::from(ab.ct_eq(bb))
}

fn ua_hash(headers: &HeaderMap) -> String {
    let ua = headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    hex::encode(Sha256::digest(ua.as_bytes()))
}

fn looks_like_bot_ua(ua: &str) -> bool {
    let trimmed = ua.trim();
    if trimmed.is_empty() || trimmed.len() < 12 {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    BOT_UA_MARKERS.iter().any(|m| lower.contains(m))
}

fn evaluate_risk(
    headers: &HeaderMap,
    action: &str,
    signals: &BrowserSignals,
    ip_challenges: u32,
    ip_failures: u32,
) -> u8 {
    let mut risk: u16 = 0;

    let ua = headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if looks_like_bot_ua(ua) {
        risk += 35;
    }

    if headers
        .get(header::ACCEPT_LANGUAGE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
    {
        risk += 15;
    }

    if headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
    {
        risk += 10;
    }

    if let Some(site) = headers
        .get("sec-fetch-site")
        .and_then(|v| v.to_str().ok())
    {
        if site.trim().eq_ignore_ascii_case("cross-site") {
            risk += 20;
        }
    } else if (ua.contains("Chrome/") || ua.contains("Firefox/"))
        && headers.get("sec-fetch-mode").is_none()
    {
        risk += 5;
    }

    if action == "register" {
        risk += 10;
    }

    if ip_challenges >= 20 {
        risk += 30;
    } else if ip_challenges >= 10 {
        risk += 20;
    } else if ip_challenges >= 5 {
        risk += 10;
    }

    if ip_failures >= 5 {
        risk += 30;
    } else if ip_failures >= 2 {
        risk += 15;
    }

    if !signals.signals_present {
        risk += 25;
    } else {
        if signals.webdriver {
            risk += 35;
        }
        if signals.automation {
            risk += 40;
        }
        if signals.headless_ua {
            risk += 35;
        }
        if signals.languages == 0 {
            risk += 15;
        }
        if signals.screen_w == 0
            || signals.screen_h == 0
            || signals.inner_w == 0
            || signals.inner_h == 0
        {
            risk += 20;
        }
        if !signals.canvas_ok {
            risk += 12;
        }
        if !signals.layout_ok {
            risk += 12;
        }
    }

    risk.min(100) as u8
}

fn difficulty_for_risk(risk: u8, base: u8) -> (u8, bool) {
    let (bump, interactive) = match risk {
        0..=24 => (0, false),
        25..=49 => (2, false),
        50..=74 => (3, true),
        _ => (4, true),
    };
    (base.saturating_add(bump).clamp(8, 23), interactive)
}

fn touch_ip_activity<'a>(
    map: &'a mut HashMap<String, IpActivity>,
    ip: &str,
    now: i64,
) -> &'a mut IpActivity {
    if map.len() > MAX_PENDING_CHALLENGES {
        map.retain(|_, v| now - v.window_start < IP_WINDOW_MS);
    }
    let entry = map.entry(ip.to_string()).or_insert_with(|| IpActivity {
        window_start: now,
        challenges: 0,
        failures: 0,
        registrations: 0,
    });
    if now - entry.window_start >= IP_WINDOW_MS {
        *entry = IpActivity {
            window_start: now,
            challenges: 0,
            failures: 0,
            registrations: 0,
        };
    }
    entry
}

/// Test helper solving an issued challenge with the exact same SHA-256 PoW formula as the client.
pub fn solve_challenge_sync(challenge: &CaptchaChallenge) -> CaptchaSolution {
    let probe = expected_probe_answer(&challenge.probe);
    let mut nonce = 0u64;
    while !pow_matches(
        &challenge.id,
        &challenge.salt,
        probe,
        nonce,
        challenge.difficulty,
    ) {
        nonce += 1;
    }
    CaptchaSolution {
        id: challenge.id.clone(),
        nonce,
        mac: challenge.mac.clone(),
        probe,
        interacted: true,
        elapsed_ms: 42,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn browser_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::USER_AGENT,
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
                .parse()
                .unwrap(),
        );
        h.insert(header::ACCEPT, "application/json".parse().unwrap());
        h.insert(header::ACCEPT_LANGUAGE, "en-US,en;q=0.9".parse().unwrap());
        h.insert("sec-fetch-site", "same-origin".parse().unwrap());
        h.insert("sec-fetch-mode", "cors".parse().unwrap());
        h
    }

    fn clean_signals() -> BrowserSignals {
        BrowserSignals {
            languages: 2,
            screen_w: 1920,
            screen_h: 1080,
            inner_w: 1440,
            inner_h: 900,
            cores: 8,
            canvas_ok: true,
            layout_ok: true,
            worker_ok: true,
            signals_present: true,
            ..Default::default()
        }
    }

    #[test]
    fn leading_zero_bits_boundaries() {
        let mut d = [0xffu8; 32];
        assert!(has_leading_zero_bits(&d, 0));
        assert!(!has_leading_zero_bits(&d, 1));
        d[0] = 0x00;
        d[1] = 0x01;
        assert!(has_leading_zero_bits(&d, 15));
        assert!(!has_leading_zero_bits(&d, 16));
        d[1] = 0x00;
        d[2] = 0x7f;
        assert!(has_leading_zero_bits(&d, 17));
        assert!(!has_leading_zero_bits(&d, 18));
    }

    #[test]
    fn issue_verify_once_and_reject_replay() {
        let st = CaptchaState::new();
        let headers = browser_headers();
        let ch = st
            .issue_challenge(&headers, "198.51.100.5", "login", &clean_signals())
            .unwrap();
        assert!(!ch.interactive);
        let sol = solve_challenge_sync(&ch);
        assert!(
            st.verify_solution(&headers, "198.51.100.5", "login", Some(&sol), "")
                .is_ok()
        );
        assert!(matches!(
            st.verify_solution(&headers, "198.51.100.5", "login", Some(&sol), ""),
            Err(CaptchaError::Invalid(_))
        ));
    }

    #[test]
    fn suspicious_client_escalates_and_requires_click() {
        let st = CaptchaState::new();
        let mut bot_h = HeaderMap::new();
        bot_h.insert(header::USER_AGENT, "python-requests/2.32".parse().unwrap());
        let mut bad = clean_signals();
        bad.webdriver = true;
        bad.automation = true;

        let clean = st
            .issue_challenge(&browser_headers(), "198.51.100.6", "login", &clean_signals())
            .unwrap();
        let risky = st
            .issue_challenge(&bot_h, "198.51.100.7", "register", &bad)
            .unwrap();
        assert!(risky.risk >= 75);
        assert!(risky.difficulty > clean.difficulty);
        assert!(risky.interactive);

        let mut sol = solve_challenge_sync(&risky);
        sol.interacted = false;
        assert!(
            st.verify_solution(&bot_h, "198.51.100.7", "register", Some(&sol), "")
                .is_err()
        );
    }

    #[test]
    fn rejects_honeypot_wrong_action_ip_ua_and_probe() {
        let st = CaptchaState::new();
        let h = browser_headers();

        let ch = st
            .issue_challenge(&h, "198.51.100.8", "register", &clean_signals())
            .unwrap();
        let sol = solve_challenge_sync(&ch);
        assert!(
            st.verify_solution(&h, "198.51.100.8", "register", Some(&sol), "bot")
                .is_err()
        );

        let ch = st
            .issue_challenge(&h, "198.51.100.8", "login", &clean_signals())
            .unwrap();
        let sol = solve_challenge_sync(&ch);
        assert!(
            st.verify_solution(&h, "198.51.100.8", "register", Some(&sol), "")
                .is_err()
        );

        let ch = st
            .issue_challenge(&h, "198.51.100.8", "login", &clean_signals())
            .unwrap();
        let sol = solve_challenge_sync(&ch);
        assert!(
            st.verify_solution(&h, "198.51.100.99", "login", Some(&sol), "")
                .is_err()
        );

        let ch = st
            .issue_challenge(&h, "198.51.100.8", "login", &clean_signals())
            .unwrap();
        let mut sol = solve_challenge_sync(&ch);
        sol.probe ^= 1;
        assert!(
            st.verify_solution(&h, "198.51.100.8", "login", Some(&sol), "")
                .is_err()
        );
    }
}
