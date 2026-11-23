//! Behavioral anti-cheat (server side).
//!
//! * Match engines produce one [`AcReport`] per human per match (inputs scored by a tiny MLP plus
//!   hard client-invariant checks and shadow-bot similarity; see `sim::anticheat`).
//! * Reports are forwarded to the primary region, aggregated per player and appended to an NDJSON
//!   log used for retraining.
//! * Periodically a **ban wave** runs on the primary: players whose evidence clears strict
//!   thresholds are summarized (pseudonymous ids, numbers only — no usernames or chat, so there is
//!   nothing a player can inject into the prompt) and judged by an LLM through any
//!   OpenAI-compatible chat-completions endpoint. "ban"
//!   verdicts above the confidence floor get the existing escalating matchmaking ban.
//! * Admins review waves and undo false positives ("clear"), which also becomes a training label.
//!
//! Configuration (env):
//!   AC_LLM_BASE_URL          OpenAI-compatible API base, e.g. https://openrouter.ai/api/v1
//!   AC_LLM_API_KEY           required for waves to judge anyone (otherwise waves only list suspects)
//!   AC_LLM_MODEL             model id sent to the endpoint
//!   AC_ENFORCE               "0"/"false" = judge but never ban (default: enforce)
//!   AC_WAVE_HOURS            hours between waves (default 24)
//!   AC_LLM_MIN_CONFIDENCE    minimum verdict confidence to ban (default 0.8)
//!   AC_MAX_BANS_PER_WAVE     hard cap per wave (default 20)

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::AppState;
use crate::auth::{now_ms, random_hex};
use crate::cluster::{CLUSTER_SECRET_HEADER, verify_cluster_secret};
use crate::http::{json_response, parse_json_body};
use crate::protocol::{AcReport, ac_log, ac_penalize};

pub const DEFAULT_LLM_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_LLM_MODEL: &str = "tencent/hy4-flash";

const DAY_MS: i64 = 86_400_000;
const WINDOW_MS: i64 = 14 * DAY_MS;
const MAX_REPORTS_PER_USER: usize = 30;
const MAX_WAVES: usize = 40;
const LOG_ROTATE_BYTES: u64 = 24 * 1024 * 1024;
const MIN_LIVE_SECS: f32 = 90.0;
const MIN_MATCHES: usize = 5;
/// Client-invariant violation slots in one match that count as "hard" evidence
const HARD_VIOL_SLOTS: f32 = 30.0;
const LLM_BATCH: usize = 8;
/// More candidates than this (or 5% of recently active players, whichever is larger) means
/// something systemic changed (client update, model bug): abort instead of mass-banning.
const ANOMALY_MIN: usize = 30;

// ---------------------------------------------------------------------------------------------
// Config + model info
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AcConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub llm_model: String,
    pub enforce: bool,
    pub wave_hours: f64,
    pub min_confidence: f64,
    pub max_bans: usize,
}

impl AcConfig {
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        Self {
            api_key: env("AC_LLM_API_KEY"),
            base_url: env("AC_LLM_BASE_URL")
                .unwrap_or_else(|| DEFAULT_LLM_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            llm_model: env("AC_LLM_MODEL").unwrap_or_else(|| DEFAULT_LLM_MODEL.to_string()),
            enforce: !matches!(env("AC_ENFORCE").as_deref(), Some("0" | "false" | "no" | "off")),
            wave_hours: env("AC_WAVE_HOURS").and_then(|v| v.parse().ok()).filter(|v: &f64| *v >= 0.25).unwrap_or(24.0),
            min_confidence: env("AC_LLM_MIN_CONFIDENCE")
                .and_then(|v| v.parse().ok())
                .map(|v: f64| v.clamp(0.5, 0.99))
                .unwrap_or(0.8),
            max_bans: env("AC_MAX_BANS_PER_WAVE").and_then(|v| v.parse().ok()).unwrap_or(20),
        }
    }
}

/// Info about the loaded detection model (set by the app at startup).
#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelInfo {
    pub version: String,
    pub trained: bool,
    pub human_ref: BTreeMap<String, [f32; 3]>,
    pub metrics: BTreeMap<String, f32>,
}

static MODEL_INFO: OnceLock<ModelInfo> = OnceLock::new();

pub fn set_model_info(info: ModelInfo) {
    let _ = MODEL_INFO.set(info);
}

fn model_info() -> ModelInfo {
    MODEL_INFO.get().cloned().unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// Stored state
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReportLite {
    pub t: i64,
    pub kind: String,
    pub mode: String,
    pub team_size: i32,
    pub live_secs: f32,
    pub score: f32,
    pub model: String,
    pub viol: f32,
    pub sh_corr: f32,
    pub sh_lead: f32,
    pub sh_lag_net: f32,
    pub react_med: f32,
    pub react_xfast: f32,
    pub jump_cv: f32,
    pub jump_mode: f32,
    pub jump_presses: f32,
    pub kick_std: f32,
    pub rtt: f32,
    pub rtt_known: f32,
    pub starve: f32,
    pub touches_pm: f32,
    pub air_frac: f32,
    pub steer_mid: f32,
}

impl ReportLite {
    fn from_report(r: &AcReport) -> Self {
        let e = |k: &str| r.evidence.get(k).copied().filter(|v| v.is_finite()).unwrap_or(0.0);
        Self {
            t: r.t,
            kind: r.kind.clone(),
            mode: r.mode.clone(),
            team_size: r.team_size,
            live_secs: r.live_secs,
            score: r.score,
            model: r.model_version.clone(),
            viol: e("viol_total"),
            sh_corr: e("shadow_steer_corr"),
            sh_lead: e("shadow_lead"),
            sh_lag_net: e("shadow_lag_minus_net_ms"),
            react_med: e("react_med_ms"),
            react_xfast: e("react_excess_fast"),
            jump_cv: e("jump_hold_cv"),
            jump_mode: e("jump_hold_mode_share"),
            jump_presses: e("jump_presses"),
            kick_std: e("kick_delay_std_ms"),
            rtt: e("rtt_ms"),
            rtt_known: e("rtt_known"),
            starve: e("starve_rate"),
            touches_pm: e("touches_per_min"),
            air_frac: e("air_touch_frac"),
            steer_mid: e("steer_mid_frac"),
        }
    }

    fn shadow_hit(&self) -> bool {
        self.sh_corr >= 0.85 && self.sh_lead >= 0.08 && self.sh_lag_net < 140.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub t: i64,
    pub wave_id: String,
    pub verdict: String,
    pub confidence: f32,
    pub summary: String,
    pub reasoning: String,
    pub applied: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AcUserRec {
    pub reports: VecDeque<ReportLite>,
    pub cleared_at: Option<i64>,
    #[serde(default)]
    pub wave_bans: Vec<i64>,
    pub last_verdict: Option<Verdict>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WaveEntry {
    pub user_id: String,
    pub username: String,
    pub pid: String,
    pub verdict: String,
    pub confidence: f32,
    pub summary: String,
    pub reasoning: String,
    pub applied: bool,
    pub stats: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WaveRecord {
    pub id: String,
    pub t: i64,
    pub trigger: String,
    pub status: String,
    pub note: String,
    pub llm_model: String,
    pub enforce: bool,
    pub candidates: usize,
    pub bans: usize,
    pub entries: Vec<WaveEntry>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcInner {
    users: HashMap<String, AcUserRec>,
    waves: VecDeque<WaveRecord>,
    last_wave_at: i64,
    last_attempt_at: i64,
    #[serde(skip)]
    dirty: bool,
}

pub struct AcStore {
    pub cfg: AcConfig,
    state_file: PathBuf,
    log_file: PathBuf,
    inner: Mutex<AcInner>,
    log_lock: Mutex<()>,
    wave_running: AtomicBool,
    llm: reqwest::Client,
}

impl AcStore {
    pub fn new(data_dir: &std::path::Path, is_primary: bool) -> Self {
        let state_file = data_dir.join("ac_state.json");
        let log_file = data_dir.join("ac_reports.ndjson");
        let mut inner = AcInner::default();
        if is_primary
            && let Ok(s) = std::fs::read_to_string(&state_file)
        {
            match serde_json::from_str::<AcInner>(&s) {
                Ok(v) => inner = v,
                Err(e) => eprintln!("[ANTI-CHEAT] could not parse {}: {e}", state_file.display()),
            }
        }
        if inner.last_wave_at == 0 {
            // First boot: let a full interval of data accumulate before the first wave.
            inner.last_wave_at = now_ms();
            inner.dirty = true;
        }
        let llm = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(180))
            .build()
            .expect("reqwest client");
        Self {
            cfg: AcConfig::from_env(),
            state_file,
            log_file,
            inner: Mutex::new(inner),
            log_lock: Mutex::new(()),
            wave_running: AtomicBool::new(false),
            llm,
        }
    }

    fn ingest_local(&self, report: &AcReport) {
        {
            let mut g = self.inner.lock().unwrap();
            let rec = g.users.entry(report.user_id.clone()).or_default();
            rec.reports.push_back(ReportLite::from_report(report));
            while rec.reports.len() > MAX_REPORTS_PER_USER {
                rec.reports.pop_front();
            }
            g.dirty = true;
        }
        self.append_log(report);
    }

    fn append_log(&self, report: &AcReport) {
        let Ok(line) = serde_json::to_string(report) else {
            return;
        };
        let _g = self.log_lock.lock().unwrap();
        if std::fs::metadata(&self.log_file).map(|m| m.len()).unwrap_or(0) > LOG_ROTATE_BYTES {
            let _ = std::fs::rename(&self.log_file, rotated(&self.log_file));
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.log_file) {
            let _ = writeln!(f, "{line}");
        }
    }

    fn flush_payload(&self) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        if !g.dirty {
            return None;
        }
        // prune players with nothing recent and no history worth keeping
        let cutoff = now_ms() - 30 * DAY_MS;
        g.users.retain(|_, r| {
            r.reports.back().is_some_and(|x| x.t > cutoff) || !r.wave_bans.is_empty() || r.cleared_at.is_some()
        });
        g.dirty = false;
        serde_json::to_string(&*g).ok()
    }

    pub fn flush_sync(&self) {
        if let Some(p) = self.flush_payload()
            && let Err(e) = crate::db::atomic_write_sync(&self.state_file, &p)
        {
            eprintln!("[ANTI-CHEAT] state flush failed: {e}");
            self.inner.lock().unwrap().dirty = true;
        }
    }
}

fn rotated(p: &std::path::Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".1");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------------------------
// Report intake
// ---------------------------------------------------------------------------------------------

/// Called by the match event sinks (any region). Stores on the primary, forwards otherwise.
pub fn submit(state: Arc<AppState>, mut report: AcReport) {
    report.region = state.config.region.clone();
    if state.config.is_primary() {
        state.ac.ingest_local(&report);
        return;
    }
    let Some(primary) = state.config.primary_url().map(|s| s.to_string()) else {
        state.ac.ingest_local(&report);
        return;
    };
    tokio::spawn(async move {
        let url = format!("{primary}/internal/ac/report");
        for attempt in 0..3 {
            let ok = state
                .cluster
                .http
                .post(&url)
                .header(CLUSTER_SECRET_HEADER, &state.config.client_secret)
                .json(&report)
                .send()
                .await
                .is_ok_and(|r| r.status().is_success());
            if ok {
                return;
            }
            tokio::time::sleep(Duration::from_millis(500 * (attempt + 1))).await;
        }
        eprintln!("[ANTI-CHEAT] could not forward report for {} to primary", report.user_id);
    });
}

pub async fn internal_report(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if !verify_cluster_secret(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }
    match serde_json::from_slice::<AcReport>(&body) {
        Ok(r) if r.features.len() <= 256 && r.evidence.len() <= 256 && r.user_id.len() <= 128 => {
            state.ac.ingest_local(&r);
            (StatusCode::OK, "ok").into_response()
        }
        _ => (StatusCode::BAD_REQUEST, "bad report").into_response(),
    }
}

// ---------------------------------------------------------------------------------------------
// Aggregation + candidate selection
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Agg {
    pub n: usize,
    pub mean_score: f32,
    pub median_score: f32,
    pub frac_high: f32,
    pub hard_matches: usize,
    pub max_viol: f32,
    pub shadow_hits: usize,
    pub med_shadow_corr: f32,
    pub med_shadow_lead: f32,
    pub med_shadow_lag_net_ms: f32,
    pub med_react_ms: f32,
    pub med_react_excess_fast: f32,
    pub med_jump_hold_cv: f32,
    pub med_jump_mode_share: f32,
    pub med_kick_std_ms: f32,
    pub med_rtt_ms: f32,
    pub rtt_unknown_matches: usize,
    pub med_starve_rate: f32,
    pub med_touches_per_min: f32,
    pub med_air_touch_frac: f32,
    pub med_steer_mid_frac: f32,
    pub kinds: BTreeMap<String, usize>,
    pub scores: Vec<f32>,
    pub last_t: i64,
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn r2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

pub fn aggregate(rec: &AcUserRec, now: i64) -> Agg {
    let since = (now - WINDOW_MS).max(rec.cleared_at.unwrap_or(0));
    let rs: Vec<&ReportLite> = rec
        .reports
        .iter()
        .filter(|r| r.t > since && r.live_secs >= MIN_LIVE_SECS)
        .collect();
    let rs: Vec<&ReportLite> = rs.iter().rev().take(20).rev().copied().collect();
    let n = rs.len();
    let mut a = Agg {
        n,
        ..Default::default()
    };
    if n == 0 {
        return a;
    }
    let col = |f: &dyn Fn(&ReportLite) -> f32| -> Vec<f32> { rs.iter().map(|r| f(r)).collect() };
    let scores = col(&|r| r.score);
    a.mean_score = scores.iter().sum::<f32>() / n as f32;
    a.frac_high = scores.iter().filter(|&&s| s >= 0.8).count() as f32 / n as f32;
    a.median_score = median(scores.clone());
    a.scores = scores.iter().map(|&s| r2(s)).collect();
    a.hard_matches = rs.iter().filter(|r| r.viol >= HARD_VIOL_SLOTS).count();
    a.max_viol = rs.iter().map(|r| r.viol).fold(0.0, f32::max);
    a.shadow_hits = rs.iter().filter(|r| r.shadow_hit()).count();
    a.med_shadow_corr = r2(median(col(&|r| r.sh_corr)));
    a.med_shadow_lead = r2(median(col(&|r| r.sh_lead)));
    a.med_shadow_lag_net_ms = median(col(&|r| r.sh_lag_net)).round();
    a.med_react_ms = median(col(&|r| r.react_med)).round();
    a.med_react_excess_fast = r2(median(col(&|r| r.react_xfast)));
    a.med_jump_hold_cv = r2(median(col(&|r| r.jump_cv)));
    a.med_jump_mode_share = r2(median(col(&|r| r.jump_mode)));
    a.med_kick_std_ms = median(col(&|r| r.kick_std)).round();
    a.med_rtt_ms = median(col(&|r| r.rtt)).round();
    a.rtt_unknown_matches = rs.iter().filter(|r| r.rtt_known < 0.5).count();
    a.med_starve_rate = (median(col(&|r| r.starve)) * 1000.0).round() / 1000.0;
    a.med_touches_per_min = r2(median(col(&|r| r.touches_pm)));
    a.med_air_touch_frac = r2(median(col(&|r| r.air_frac)));
    a.med_steer_mid_frac = r2(median(col(&|r| r.steer_mid)));
    for r in &rs {
        *a.kinds.entry(r.kind.clone()).or_default() += 1;
    }
    a.last_t = rs.last().map(|r| r.t).unwrap_or(0);
    a
}

/// Strict entry criteria for LLM review. Several independent paths, each conservative.
pub fn is_candidate(a: &Agg) -> bool {
    let model = a.n >= MIN_MATCHES && a.mean_score >= 0.85 && a.frac_high >= 0.7;
    let hard = a.hard_matches >= 2;
    let shadow = a.n >= MIN_MATCHES && a.shadow_hits >= 3 && a.mean_score >= 0.5;
    model || hard || shadow
}

fn suspicion_rank(a: &Agg) -> f32 {
    a.mean_score + 0.15 * a.hard_matches as f32 + 0.1 * a.shadow_hits as f32
}

struct Candidate {
    user_id: String,
    username: String,
    agg: Agg,
    account: Value,
}

fn collect_candidates(state: &AppState, now: i64) -> (Vec<Candidate>, usize) {
    let aggs: Vec<(String, Agg, Option<Verdict>)> = {
        let g = state.ac.inner.lock().unwrap();
        g.users
            .iter()
            .map(|(uid, rec)| (uid.clone(), aggregate(rec, now), rec.last_verdict.clone()))
            .collect()
    };
    let active = aggs.iter().filter(|(_, a, _)| a.n > 0).count();
    let db = state.db.lock().unwrap();
    let by_id: HashMap<&str, &crate::db::User> = db.data.users.values().map(|u| (u.id.as_str(), u)).collect();
    let mut out = Vec::new();
    for (uid, agg, last) in aggs {
        if !is_candidate(&agg) {
            continue;
        }
        let Some(u) = by_id.get(uid.as_str()) else {
            continue;
        };
        if state.config.is_admin_username(&u.username) {
            continue;
        }
        if u.ac_ban_until.is_some_and(|t| t > now) {
            continue;
        }
        // Already judged "not a ban" and nothing new since: don't re-ask every wave.
        if let Some(v) = &last
            && v.verdict != "ban"
            && v.verdict != "error"
            && v.t >= agg.last_t
        {
            continue;
        }
        let s = &u.stats;
        out.push(Candidate {
            user_id: uid,
            username: u.username.clone(),
            account: json!({
                "accountAgeDays": ((now - u.created_at) as f64 / DAY_MS as f64).round(),
                "level": u.level,
                "rankPoints": u.rp,
                "totalMatches": s.matches,
                "multiplayerMatches": s.mp_matches,
                "multiplayerWinRate": if s.mp_matches > 0 { ((s.mp_wins as f64 / s.mp_matches as f64) * 100.0).round() / 100.0 } else { 0.0 },
                "priorAntiCheatFlags": u.anti_cheat_flags,
            }),
            agg,
        });
    }
    out.sort_by(|a, b| suspicion_rank(&b.agg).partial_cmp(&suspicion_rank(&a.agg)).unwrap_or(std::cmp::Ordering::Equal));
    (out, active)
}

// ---------------------------------------------------------------------------------------------
// LLM judgment
// ---------------------------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are the final reviewer in Hyperstrike's automated anti-cheat system.
Hyperstrike is a browser car-soccer game. Physics are fully server-authoritative, so the only cheats that exist are automated inputs: a bot playing for the player, an aim-assist that takes over near the ball, or a trigger-bot that times jumps/boost. For each pseudonymous player you get evidence aggregated over their recent matches (each match >= 90 s of live play). Decide: "ban", "no_ban", or "insufficient".

Signal guide:
- hardMatches / maxViolationSlots: the official client ALWAYS sends yaw == steer and roll in {-1, 0, 1} (or roll == steer while air-roll-free is held). A genuine client cannot violate this; violations mean a modified client or input injector. Strong evidence when present in several matches.
- modelScore: a small neural network's per-match probability of automated play, trained on simulated humans vs. bots. Meaningful only when consistently high across many matches; isolated high matches are normal noise.
- shadowBot: the player's steering is correlated with what the game's own champion AI would do in their seat, at the best-fitting time lag. lead > ~0.05 means they track the AI's intent better at network-only lag than at human reaction lag, which humans cannot do. lagMinusNetworkMs < ~120 together with corr >= 0.85 is bot-like. shadowHits counts matches meeting all of these. Moderate correlation (0.4-0.8) is normal for humans who chase the ball.
- reaction: reactMs = response latency to unpredictable ball touches by others after subtracting measured network delay (it is noisy; humans typically >= 120-150 ms). reactExcessFast = how much more often they respond within 120 ms right after such events than at random moments; humans ~0.
- jump timing: bots repeat identical hold durations (jumpHoldModeShare high, jumpHoldCv low), but only with enough presses.
- humanReference gives [p5, p50, p95] of each measure for simulated legitimate players.
Benign explanations to rule out: bad network (high starveRate, rttUnknownMatches > 0 makes timing unreliable), keyboard vs gamepad vs touch differences, highly skilled players (many touches or aerials are NOT evidence of cheating), small samples.

Policy: ban only when at least two independent signal families agree (e.g. shadow-bot + reaction timing, or client-invariant violations + anything else) across multiple matches. Wrongly banning a real player is far worse than missing a cheater. When evidence is mixed, answer no_ban or insufficient. Confidence is your probability that the player is using automation.

Respond with JSON only, no prose:
{"verdicts":[{"id":"P1","verdict":"ban|no_ban|insufficient","confidence":0.0,"summary":"one short sentence","reasoning":"<= 80 words citing the specific numbers"}]}"#;

fn dossier(pid: &str, c: &Candidate) -> Value {
    let a = &c.agg;
    json!({
        "id": pid,
        "account": c.account,
        "analyzedMatches": a.n,
        "matchKinds": a.kinds,
        "modelScore": { "mean": r2(a.mean_score), "median": r2(a.median_score), "fracAtLeast0.8": r2(a.frac_high), "perMatch": a.scores },
        "hardEvidence": { "hardMatches": a.hard_matches, "maxViolationSlots": a.max_viol },
        "shadowBot": { "shadowHits": a.shadow_hits, "medianCorr": a.med_shadow_corr, "medianLead": a.med_shadow_lead, "medianLagMinusNetworkMs": a.med_shadow_lag_net_ms },
        "reaction": { "medianReactMs": a.med_react_ms, "medianReactExcessFast": a.med_react_excess_fast },
        "jumpTiming": { "medianJumpHoldCv": a.med_jump_hold_cv, "medianJumpHoldModeShare": a.med_jump_mode_share },
        "kickoff": { "medianKickDelayStdMs": a.med_kick_std_ms },
        "network": { "medianRttMs": a.med_rtt_ms, "rttUnknownMatches": a.rtt_unknown_matches, "medianStarveRate": a.med_starve_rate },
        "inputStyle": { "medianSteerAnalogFrac": a.med_steer_mid_frac },
        "outcomes": { "medianTouchesPerMin": a.med_touches_per_min, "medianAirTouchFrac": a.med_air_touch_frac },
    })
}

fn human_ref_subset() -> Value {
    let info = model_info();
    let keys = [
        ("shadow_steer_corr", "shadowCorr"),
        ("shadow_lead", "shadowLead"),
        ("shadow_lag_minus_net_ms", "shadowLagMinusNetworkMs"),
        ("react_med_ms", "reactMs"),
        ("react_excess_fast", "reactExcessFast"),
        ("jump_hold_cv", "jumpHoldCv"),
        ("jump_hold_mode_share", "jumpHoldModeShare"),
        ("kick_delay_std_ms", "kickDelayStdMs"),
        ("steer_mid_frac", "steerAnalogFrac"),
    ];
    let mut m = serde_json::Map::new();
    for (k, name) in keys {
        if let Some(v) = info.human_ref.get(k) {
            m.insert(name.into(), json!([r2(v[0]), r2(v[1]), r2(v[2])]));
        }
    }
    Value::Object(m)
}

#[derive(Debug, Clone, Deserialize)]
struct LlmVerdict {
    id: String,
    verdict: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    reasoning: String,
}

fn extract_json(s: &str) -> Option<Value> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    serde_json::from_str(&s[start..=end]).ok()
}

async fn judge_batch(state: &AppState, players: Vec<Value>) -> Result<Vec<LlmVerdict>, String> {
    let key = state.ac.cfg.api_key.clone().ok_or("no AC_LLM_API_KEY")?;
    let user = json!({
        "humanReference": human_ref_subset(),
        "detectionModel": model_info().version,
        "players": players,
    });
    let mut body = json!({
        "model": state.ac.cfg.llm_model,
        "temperature": 0,
        "max_tokens": 4000,
        "response_format": { "type": "json_object" },
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user.to_string() },
        ],
    });
    let url = format!("{}/chat/completions", state.ac.cfg.base_url);
    let mut attempt = 0;
    let v: Value = loop {
        let resp = state
            .ac
            .llm
            .post(&url)
            .bearer_auth(&key)
            .header("X-Title", "Hyperstrike Anti-Cheat")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;
        let status = resp.status();
        let v: Value = resp.json().await.map_err(|e| format!("bad response ({status}): {e}"))?;
        if status.is_success() {
            break v;
        }
        // Not every OpenAI-compatible server supports JSON mode: retry once without it.
        if attempt == 0 && status.is_client_error() && body.get("response_format").is_some() {
            if let Some(o) = body.as_object_mut() {
                o.remove("response_format");
            }
            attempt += 1;
            continue;
        }
        let msg = v.pointer("/error/message").and_then(|m| m.as_str()).unwrap_or("unknown error");
        return Err(format!("LLM endpoint {status}: {msg}"));
    };
    let content = v
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .ok_or("no content in LLM response")?;
    let parsed = extract_json(content).ok_or("LLM did not return JSON")?;
    let list = parsed.get("verdicts").cloned().unwrap_or(Value::Null);
    serde_json::from_value::<Vec<LlmVerdict>>(list).map_err(|e| format!("bad verdict list: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Ban waves
// ---------------------------------------------------------------------------------------------

fn apply_ban(state: &Arc<AppState>, user_id: &str, reason: &str) -> bool {
    let updated = {
        let mut db = state.db.lock().unwrap();
        let Some((lower, u)) = db.data.users.iter_mut().find(|(_, u)| u.id == user_id) else {
            return false;
        };
        ac_penalize(u, reason);
        let out = (lower.clone(), u.clone());
        db.mark_dirty();
        out
    };
    let (lower, user) = updated;
    crate::cluster::fanout_user_sync(Arc::clone(state), lower, Some(user));
    let msg = "Anti-cheat: automated play detected. Matchmaking ban applied.";
    crate::ws::disconnect_user(state, user_id, msg);
    crate::cluster::fanout_kick_user(Arc::clone(state), user_id.to_string(), msg.to_string());
    true
}

fn store_wave(state: &AppState, w: &WaveRecord, completed: bool) {
    let mut g = state.ac.inner.lock().unwrap();
    g.waves.push_front(w.clone());
    while g.waves.len() > MAX_WAVES {
        g.waves.pop_back();
    }
    g.last_attempt_at = w.t;
    if completed {
        g.last_wave_at = w.t;
    }
    g.dirty = true;
}

pub async fn run_wave(state: Arc<AppState>, trigger: &str, dry_run: bool) -> WaveRecord {
    let now = now_ms();
    let cfg = state.ac.cfg.clone();
    let mut wave = WaveRecord {
        id: format!("w{}", &random_hex(4)),
        t: now,
        trigger: trigger.to_string(),
        status: String::new(),
        note: String::new(),
        llm_model: cfg.llm_model.clone(),
        enforce: cfg.enforce && !dry_run,
        ..Default::default()
    };
    if state.ac.wave_running.swap(true, Ordering::SeqCst) {
        wave.status = "busy".into();
        wave.note = "Another wave is already running.".into();
        return wave;
    }
    struct Release<'a>(&'a AtomicBool);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _release = Release(&state.ac.wave_running);

    let (cands, active) = collect_candidates(&state, now);
    wave.candidates = cands.len();
    let limit = ANOMALY_MIN.max(active / 20);
    let mk_entry = |c: &Candidate, pid: &str| WaveEntry {
        user_id: c.user_id.clone(),
        username: c.username.clone(),
        pid: pid.to_string(),
        stats: serde_json::to_value(&c.agg).unwrap_or(Value::Null),
        ..Default::default()
    };

    if cands.is_empty() {
        wave.status = "no_candidates".into();
        wave.note = format!("{active} recently active players; nobody met the review thresholds.");
        store_wave(&state, &wave, !dry_run);
        return wave;
    }
    if cands.len() > limit {
        wave.status = "aborted_anomaly".into();
        wave.note = format!(
            "{} candidates out of {active} active players exceeds the safety limit ({limit}). Something systemic \
             (client update, model regression) is likely; no one was judged or banned. Review the model before re-running.",
            cands.len()
        );
        wave.entries = cands.iter().enumerate().take(50).map(|(i, c)| mk_entry(c, &format!("P{}", i + 1))).collect();
        eprintln!("[ANTI-CHEAT] wave aborted: {}", wave.note);
        store_wave(&state, &wave, !dry_run);
        return wave;
    }
    if cfg.api_key.is_none() {
        wave.status = "skipped_no_api_key".into();
        wave.note = "AC_LLM_API_KEY is not set, so candidates were listed but not judged or banned.".into();
        wave.entries = cands.iter().enumerate().map(|(i, c)| mk_entry(c, &format!("P{}", i + 1))).collect();
        store_wave(&state, &wave, !dry_run);
        return wave;
    }

    let mut entries: Vec<WaveEntry> = Vec::new();
    let mut errors = 0usize;
    for (bi, chunk) in cands.chunks(LLM_BATCH).enumerate() {
        let pids: Vec<String> = (0..chunk.len()).map(|i| format!("P{}", bi * LLM_BATCH + i + 1)).collect();
        let players: Vec<Value> = chunk.iter().zip(&pids).map(|(c, pid)| dossier(pid, c)).collect();
        let result = judge_batch(&state, players).await;
        for (c, pid) in chunk.iter().zip(&pids) {
            let mut e = mk_entry(c, pid);
            match &result {
                Ok(vs) => {
                    if let Some(v) = vs.iter().find(|v| v.id == *pid) {
                        e.verdict = match v.verdict.as_str() {
                            "ban" | "no_ban" | "insufficient" => v.verdict.clone(),
                            _ => "insufficient".into(),
                        };
                        e.confidence = v.confidence.clamp(0.0, 1.0);
                        e.summary = v.summary.chars().take(240).collect();
                        e.reasoning = v.reasoning.chars().take(900).collect();
                    } else {
                        e.verdict = "error".into();
                        e.reasoning = "LLM returned no verdict for this player.".into();
                        errors += 1;
                    }
                }
                Err(err) => {
                    e.verdict = "error".into();
                    e.reasoning = err.clone();
                    errors += 1;
                }
            }
            entries.push(e);
        }
    }

    let mut bans = 0usize;
    for e in &mut entries {
        let eligible = e.verdict == "ban" && f64::from(e.confidence) >= cfg.min_confidence;
        if eligible && wave.enforce && bans < cfg.max_bans {
            let reason = format!(
                "Ban wave {}: automated play (LLM confidence {:.2}). {}",
                wave.id,
                e.confidence,
                e.summary
            );
            if apply_ban(&state, &e.user_id, &reason) {
                e.applied = true;
                bans += 1;
                eprintln!("[ANTI-CHEAT] wave {} banned {} ({:.2})", wave.id, e.username, e.confidence);
            }
        }
    }
    wave.bans = bans;

    if !dry_run {
        let mut g = state.ac.inner.lock().unwrap();
        for e in &entries {
            if let Some(rec) = g.users.get_mut(&e.user_id) {
                rec.last_verdict = Some(Verdict {
                    t: now,
                    wave_id: wave.id.clone(),
                    verdict: e.verdict.clone(),
                    confidence: e.confidence,
                    summary: e.summary.clone(),
                    reasoning: e.reasoning.clone(),
                    applied: e.applied,
                });
                if e.applied {
                    rec.wave_bans.push(now);
                }
            }
        }
        g.dirty = true;
    }
    let all_failed = errors == entries.len();
    wave.status = if dry_run {
        "dry_run".into()
    } else if all_failed {
        "failed".into()
    } else if errors > 0 {
        "partial".into()
    } else {
        "ok".into()
    };
    if !wave.enforce && !dry_run {
        wave.note = "AC_ENFORCE is off: verdicts recorded, nobody banned.".into();
    }
    wave.entries = entries;
    store_wave(&state, &wave, !dry_run && !all_failed);
    wave
}

/// Background loop on the primary: persist state, run scheduled waves.
pub fn spawn_tasks(state: Arc<AppState>, mut shutdown: tokio::sync::watch::Receiver<bool>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        tick.tick().await;
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break;
                    }
                }
                _ = tick.tick() => {
                    if !state.config.is_primary() {
                        continue;
                    }
                    if let Some(p) = state.ac.flush_payload() {
                        let path = state.ac.state_file.clone();
                        let st = Arc::clone(&state);
                        let _ = tokio::task::spawn_blocking(move || {
                            if let Err(e) = crate::db::atomic_write_sync(&path, &p) {
                                eprintln!("[ANTI-CHEAT] state flush failed: {e}");
                                st.ac.inner.lock().unwrap().dirty = true;
                            }
                        }).await;
                    }
                    let (last, attempt) = {
                        let g = state.ac.inner.lock().unwrap();
                        (g.last_wave_at, g.last_attempt_at)
                    };
                    let now = now_ms();
                    let interval = (state.ac.cfg.wave_hours * 3_600_000.0) as i64;
                    if now - last >= interval && now - attempt >= 3_600_000 {
                        let w = run_wave(Arc::clone(&state), "scheduled", false).await;
                        eprintln!("[ANTI-CHEAT] scheduled wave {}: {} ({} candidates, {} bans)", w.id, w.status, w.candidates, w.bans);
                    }
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Admin endpoints
// ---------------------------------------------------------------------------------------------

fn admin_guard(state: &AppState, headers: &HeaderMap) -> Result<String, Box<Response>> {
    crate::admin::require_admin(state, headers).map(|(_, lower)| lower)
}

pub async fn admin_behavior(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(r) = admin_guard(&state, &headers) {
        return *r;
    }
    let now = now_ms();
    let (rows, waves, last_wave, last_attempt, tracked) = {
        let g = state.ac.inner.lock().unwrap();
        let rows: Vec<(String, Agg, AcUserRec)> = g
            .users
            .iter()
            .map(|(uid, r)| (uid.clone(), aggregate(r, now), r.clone()))
            .collect();
        (rows, g.waves.iter().take(20).cloned().collect::<Vec<_>>(), g.last_wave_at, g.last_attempt_at, g.users.len())
    };
    let db = state.db.lock().unwrap();
    let by_id: HashMap<&str, &crate::db::User> = db.data.users.values().map(|u| (u.id.as_str(), u)).collect();
    let mut suspects: Vec<(f32, Value)> = rows
        .iter()
        .filter(|(_, a, r)| a.n > 0 && (a.mean_score >= 0.5 || a.hard_matches > 0 || a.shadow_hits > 0 || r.last_verdict.is_some() || !r.wave_bans.is_empty()))
        .filter_map(|(uid, a, r)| {
            let u = by_id.get(uid.as_str())?;
            Some((
                suspicion_rank(a),
                json!({
                    "userId": uid,
                    "username": u.username,
                    "candidate": is_candidate(a),
                    "agg": a,
                    "banUntil": u.ac_ban_until.filter(|t| *t > now),
                    "clearedAt": r.cleared_at,
                    "waveBans": r.wave_bans.len(),
                    "lastVerdict": r.last_verdict,
                }),
            ))
        })
        .collect();
    suspects.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let reports_recent = rows.iter().map(|(_, a, _)| a.n).sum::<usize>();
    let cfg = &state.ac.cfg;
    let info = model_info();
    let interval = (cfg.wave_hours * 3_600_000.0) as i64;
    json_response(
        &state,
        StatusCode::OK,
        &json!({
            "config": {
                "enforce": cfg.enforce,
                "apiKeySet": cfg.api_key.is_some(),
                "llmModel": cfg.llm_model,
                "llmBaseUrl": cfg.base_url,
                "waveHours": cfg.wave_hours,
                "minConfidence": cfg.min_confidence,
                "maxBansPerWave": cfg.max_bans,
                "detectionModel": info.version,
                "detectionModelTrained": info.trained,
                "detectionMetrics": info.metrics,
            },
            "lastWaveAt": last_wave,
            "lastAttemptAt": last_attempt,
            "nextWaveAt": (last_wave + interval).max(last_attempt + 3_600_000),
            "waveRunning": state.ac.wave_running.load(Ordering::SeqCst),
            "trackedPlayers": tracked,
            "recentReports": reports_recent,
            "suspects": suspects.into_iter().take(100).map(|(_, v)| v).collect::<Vec<_>>(),
            "waves": waves,
        }),
    )
}

pub async fn admin_user_reports(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(user_id): Path<String>,
) -> Response {
    if let Err(r) = admin_guard(&state, &headers) {
        return *r;
    }
    let rec = state.ac.inner.lock().unwrap().users.get(&user_id).cloned();
    match rec {
        Some(r) => json_response(&state, StatusCode::OK, &json!({ "reports": r.reports, "rec": { "clearedAt": r.cleared_at, "waveBans": r.wave_bans, "lastVerdict": r.last_verdict } })),
        None => json_response(&state, StatusCode::NOT_FOUND, &json!({ "error": "No reports for this player" })),
    }
}

pub async fn admin_run_wave(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = admin_guard(&state, &headers) {
        return *r;
    }
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let dry = parsed.get("dryRun").and_then(|v| v.as_bool()).unwrap_or(true);
    let w = run_wave(Arc::clone(&state), if dry { "admin_dry_run" } else { "admin" }, dry).await;
    json_response(&state, StatusCode::OK, &json!({ "wave": w }))
}

/// Undo a behavioral ban as a false positive: lifts the ban, refunds the escalation step, gives
/// the player a clean slate for future waves and records the label for retraining.
pub async fn admin_clear(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let admin = match admin_guard(&state, &headers) {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let parsed = match parse_json_body(&state, &body) {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(user_id) = parsed.get("userId").and_then(|v| v.as_str()).map(|s| s.to_string()) else {
        return json_response(&state, StatusCode::BAD_REQUEST, &json!({ "error": "userId required" }));
    };
    let now = now_ms();
    let updated = {
        let mut db = state.db.lock().unwrap();
        let found = db.data.users.iter_mut().find(|(_, u)| u.id == user_id).map(|(lower, u)| {
            u.ac_ban_until = None;
            u.anti_cheat_flags = u.anti_cheat_flags.saturating_sub(1);
            ac_log(u, &format!("CLEARED by {admin}: behavioral ban was a false positive"));
            (lower.clone(), u.clone())
        });
        if found.is_some() {
            db.mark_dirty();
        }
        found
    };
    let Some((lower, user)) = updated else {
        return json_response(&state, StatusCode::NOT_FOUND, &json!({ "error": "User not found" }));
    };
    crate::cluster::fanout_user_sync(Arc::clone(&state), lower, Some(user));
    {
        let mut g = state.ac.inner.lock().unwrap();
        let rec = g.users.entry(user_id.clone()).or_default();
        rec.cleared_at = Some(now);
        if let Some(v) = rec.last_verdict.as_mut() {
            v.summary = format!("[cleared by admin] {}", v.summary);
        }
        g.dirty = true;
    }
    json_response(&state, StatusCode::OK, &json!({ "ok": true }))
}

/// NDJSON of every stored report with a training label:
/// 1 = banned by a wave afterwards and never cleared, 0 = cleared by an admin afterwards,
/// null = no verdict (treated as "probably human" with low weight by train.py).
pub async fn admin_export(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(r) = admin_guard(&state, &headers) {
        return *r;
    }
    let labels: HashMap<String, (Vec<i64>, Option<i64>)> = {
        let g = state.ac.inner.lock().unwrap();
        g.users.iter().map(|(k, r)| (k.clone(), (r.wave_bans.clone(), r.cleared_at))).collect()
    };
    let files = vec![rotated(&state.ac.log_file), state.ac.log_file.clone()];
    let labels = Arc::new(labels);
    type Lines = tokio::io::Lines<tokio::io::BufReader<tokio::fs::File>>;
    let stream = futures_util::stream::unfold(
        (files, 0usize, None::<Lines>, labels),
        |(files, mut idx, mut cur, labels)| async move {
            use tokio::io::AsyncBufReadExt;
            loop {
                if cur.is_none() {
                    if idx >= files.len() {
                        return None;
                    }
                    let f = tokio::fs::File::open(&files[idx]).await;
                    idx += 1;
                    if let Ok(f) = f {
                        cur = Some(tokio::io::BufReader::new(f).lines());
                    }
                    continue;
                }
                let mut chunk = String::new();
                let mut n = 0;
                let lines = cur.as_mut().unwrap();
                while n < 200 {
                    match lines.next_line().await {
                        Ok(Some(line)) => {
                            if let Ok(mut v) = serde_json::from_str::<Value>(&line) {
                                let uid = v.get("userId").and_then(|x| x.as_str()).unwrap_or("").to_string();
                                let t = v.get("t").and_then(|x| x.as_i64()).unwrap_or(0);
                                let label = labels.get(&uid).and_then(|(bans, cleared)| {
                                    if cleared.is_some_and(|c| c > t) {
                                        Some(0)
                                    } else if bans.iter().any(|&b| b > t && cleared.is_none_or(|c| c < b)) {
                                        Some(1)
                                    } else {
                                        None
                                    }
                                });
                                if let Some(o) = v.as_object_mut() {
                                    o.insert("label".into(), json!(label));
                                }
                                chunk.push_str(&v.to_string());
                                chunk.push('\n');
                            }
                            n += 1;
                        }
                        _ => {
                            cur = None;
                            break;
                        }
                    }
                }
                if !chunk.is_empty() {
                    return Some((Ok::<Bytes, std::io::Error>(Bytes::from(chunk)), (files, idx, cur, labels)));
                }
            }
        },
    );
    let mut resp = Response::new(Body::from_stream(stream));
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/x-ndjson"));
    h.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"hyperstrike_ac_export.ndjson\""),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(score: f32, viol: f32) -> ReportLite {
        ReportLite {
            t: now_ms(),
            live_secs: 200.0,
            score,
            viol,
            ..Default::default()
        }
    }

    #[test]
    fn single_high_match_is_not_a_candidate() {
        let mut r = AcUserRec::default();
        r.reports.push_back(rep(0.99, 0.0));
        for _ in 0..6 {
            r.reports.push_back(rep(0.2, 0.0));
        }
        assert!(!is_candidate(&aggregate(&r, now_ms())));
    }

    #[test]
    fn consistent_high_scores_are_candidates() {
        let mut r = AcUserRec::default();
        for _ in 0..8 {
            r.reports.push_back(rep(0.95, 0.0));
        }
        assert!(is_candidate(&aggregate(&r, now_ms())));
    }

    #[test]
    fn repeated_invariant_violations_are_candidates() {
        let mut r = AcUserRec::default();
        r.reports.push_back(rep(0.1, 500.0));
        r.reports.push_back(rep(0.1, 900.0));
        assert!(is_candidate(&aggregate(&r, now_ms())));
    }

    #[test]
    fn clearing_resets_evidence() {
        let mut r = AcUserRec::default();
        for _ in 0..8 {
            r.reports.push_back(rep(0.95, 0.0));
        }
        r.cleared_at = Some(now_ms() + 1);
        assert_eq!(aggregate(&r, now_ms() + 2).n, 0);
    }

    #[test]
    fn extracts_json_from_fenced_reply() {
        let v = extract_json("```json\n{\"verdicts\":[]}\n```").unwrap();
        assert!(v.get("verdicts").is_some());
    }
}
