//! Behavioral anti-cheat: per-player feature extraction, a "shadow bot" that records what the
//! built-in AI would have done in the player's seat, hard client-invariant checks, and a tiny MLP
//! scorer.
//!
//! The SAME tracker code runs on the live server (match_engine) and in the offline synthetic data
//! generator (examples/ac_datagen.rs), so training and inference always see identical features.
//!
//! Everything here is cheap: O(1) work per player per 60 Hz input slot plus one shadow-bot update.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{BotAi, CarBallHit, World, net_pack_controls};

/// One server input slot (2 physics ticks at 120 Hz).
pub const SLOT_MS: f64 = 1000.0 / 60.0;
/// Shadow-bot lag search window: 0..LAGS slots (0..~800 ms).
pub const LAGS: usize = 48;
const MAX_SAMPLES: usize = 800;
/// A stimulus with no response within this many slots (~750 ms) counts as "no fast response".
const STIM_TIMEOUT: u64 = 45;
/// Corrected reaction latency below this is physiologically implausible for a human
/// responding to an unpredictable event.
pub const FAST_REACT_MS: f64 = 120.0;

/// Names of the features fed to the model, in order. Changing this list requires retraining.
pub const FEATURE_NAMES: &[&str] = &[
    "steer_zero_frac",
    "steer_full_frac",
    "steer_mid_frac",
    "steer_distinct_ratio",
    "steer_mid_absdelta",
    "steer_micro_frac",
    "steer_d2_rms",
    "steer_mid_repeat_frac",
    "steer_changes_per_s",
    "steer_flips_per_s",
    "thr_zero_frac",
    "thr_full_frac",
    "thr_mid_frac",
    "thr_distinct_ratio",
    "thr_mid_absdelta",
    "thr_micro_frac",
    "thr_d2_rms",
    "thr_mid_repeat_frac",
    "thr_changes_per_s",
    "thr_flips_per_s",
    "pitch_zero_frac",
    "pitch_full_frac",
    "pitch_mid_frac",
    "pitch_distinct_ratio",
    "pitch_mid_absdelta",
    "pitch_micro_frac",
    "pitch_d2_rms",
    "pitch_mid_repeat_frac",
    "pitch_changes_per_s",
    "pitch_flips_per_s",
    "pitch_eq_thr_frac",
    "jump_per_min",
    "jump_hold_mean_ms",
    "jump_hold_cv",
    "jump_hold_mode_share",
    "jump_hold_1slot_frac",
    "jump_gap_cv",
    "boost_per_min",
    "boost_hold_log_ms",
    "boost_hold_cv",
    "boost_hold_mode_share",
    "boost_duty",
    "react_med_ms",
    "react_p10_ms",
    "react_fast_frac",
    "react_excess_fast",
    "ctrl_fast_frac",
    "kick_preheld_frac",
    "kick_delay_std_ms",
    "rtt_ms",
    "depth_avg",
    "starve_rate",
    "roll_nonzero_frac",
    "handbrake_frac",
    "shadow_lead",
];

// ---------------------------------------------------------------------------------------------
// Per-channel analog statistics
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Chan {
    n: u32,
    zero: u32,
    full: u32,
    mid: u32,
    /// Bitset over the 2001 possible round3 values in [-1, 1]
    seen: [u64; 32],
    distinct: u32,
    d_n: u32,
    d_abs: f64,
    micro: u32,
    changes: u32,
    d2_sq: f64,
    d2_n: u32,
    flips: u32,
    mid_repeat: u32,
    prev: f64,
    prev_d: f64,
    has_prev: bool,
    has_prev_d: bool,
    last_sign: i8,
}

impl Default for Chan {
    fn default() -> Self {
        Self {
            n: 0,
            zero: 0,
            full: 0,
            mid: 0,
            seen: [0; 32],
            distinct: 0,
            d_n: 0,
            d_abs: 0.0,
            micro: 0,
            changes: 0,
            d2_sq: 0.0,
            d2_n: 0,
            flips: 0,
            mid_repeat: 0,
            prev: 0.0,
            prev_d: 0.0,
            has_prev: false,
            has_prev_d: false,
            last_sign: 0,
        }
    }
}

#[inline]
fn is_mid(a: f64) -> bool {
    a > 1e-9 && a < 0.999
}

impl Chan {
    fn observe(&mut self, v: f64) {
        self.n += 1;
        let a = v.abs();
        if a <= 1e-9 {
            self.zero += 1;
        } else if a >= 0.999 {
            self.full += 1;
        } else {
            self.mid += 1;
            let b = (((v + 1.0) * 1000.0).round() as i64).clamp(0, 2000) as usize;
            let (w, bit) = (b / 64, b % 64);
            if self.seen[w] & (1 << bit) == 0 {
                self.seen[w] |= 1 << bit;
                self.distinct += 1;
            }
        }
        if self.has_prev {
            let d = v - self.prev;
            if d.abs() > 1e-9 {
                self.changes += 1;
                if d.abs() < 0.02 {
                    self.micro += 1;
                }
            }
            if is_mid(a) && is_mid(self.prev.abs()) {
                self.d_n += 1;
                self.d_abs += d.abs();
                if d.abs() <= 1e-9 {
                    self.mid_repeat += 1;
                }
            }
            if self.has_prev_d {
                self.d2_sq += (d - self.prev_d) * (d - self.prev_d);
                self.d2_n += 1;
            }
            self.prev_d = d;
            self.has_prev_d = true;
        }
        let s: i8 = if v > 0.05 {
            1
        } else if v < -0.05 {
            -1
        } else {
            0
        };
        if s != 0 {
            if self.last_sign != 0 && s != self.last_sign {
                self.flips += 1;
            }
            self.last_sign = s;
        }
        self.prev = v;
        self.has_prev = true;
    }

    /// A gap in live play: don't difference across it.
    fn break_run(&mut self) {
        self.has_prev = false;
        self.has_prev_d = false;
    }

    fn features(&self, secs: f64, out: &mut Vec<f32>) {
        let n = f64::from(self.n.max(1));
        let mid = f64::from(self.mid);
        out.push((f64::from(self.zero) / n) as f32);
        out.push((f64::from(self.full) / n) as f32);
        out.push((mid / n) as f32);
        out.push((f64::from(self.distinct) / mid.clamp(1.0, 2001.0)) as f32);
        out.push(if self.d_n > 0 {
            (self.d_abs / f64::from(self.d_n)) as f32
        } else {
            0.0
        });
        out.push((f64::from(self.micro) / f64::from(self.changes.max(1))) as f32);
        out.push(if self.d2_n > 0 {
            (self.d2_sq / f64::from(self.d2_n)).sqrt() as f32
        } else {
            0.0
        });
        out.push((f64::from(self.mid_repeat) / f64::from(self.d_n.max(1))) as f32);
        out.push((f64::from(self.changes) / secs) as f32);
        out.push((f64::from(self.flips) / secs) as f32);
    }
}

// ---------------------------------------------------------------------------------------------
// Button timing statistics
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Btn {
    held: bool,
    since: u64,
    last_press: Option<u64>,
    presses: u32,
    held_slots: u32,
    holds: Vec<u16>,
    gaps: Vec<u16>,
    press_slots: Vec<u32>,
}

impl Btn {
    fn observe(&mut self, down: bool, slot: u64) {
        if down && !self.held {
            self.presses += 1;
            if let Some(lp) = self.last_press {
                let g = slot.saturating_sub(lp);
                if g <= 30 && self.gaps.len() < MAX_SAMPLES {
                    self.gaps.push(g as u16);
                }
            }
            if self.press_slots.len() < MAX_SAMPLES * 2 {
                self.press_slots.push(slot as u32);
            }
            self.last_press = Some(slot);
            self.since = slot;
            self.held = true;
        } else if !down && self.held {
            self.end_hold(slot);
        }
        if down {
            self.held_slots += 1;
        }
    }

    fn end_hold(&mut self, slot: u64) {
        if self.held {
            let d = slot.saturating_sub(self.since).clamp(1, 60_000);
            if self.holds.len() < MAX_SAMPLES {
                self.holds.push(d as u16);
            }
            self.held = false;
        }
    }
}

fn mean_cv(v: &[u16]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let n = v.len() as f64;
    let m = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = v.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / n;
    (m, if m > 0.0 { var.sqrt() / m } else { 0.0 })
}

fn mode_share(v: &[u16]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<u16, u32> = BTreeMap::new();
    for &x in v {
        *counts.entry(x).or_default() += 1;
    }
    f64::from(*counts.values().max().unwrap_or(&0)) / v.len() as f64
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * q).round() as usize;
    v[idx.min(v.len() - 1)]
}

// ---------------------------------------------------------------------------------------------
// Shadow-bot lagged similarity
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct Corr {
    n: f64,
    sx: f64,
    sy: f64,
    sxy: f64,
    sxx: f64,
    syy: f64,
    agree: f64,
}

impl Corr {
    #[inline]
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxy += x * y;
        self.sxx += x * x;
        self.syy += y * y;
        if (x - y).abs() < 0.1 {
            self.agree += 1.0;
        }
    }

    fn r(&self) -> f64 {
        if self.n < 240.0 {
            return 0.0;
        }
        let num = self.n * self.sxy - self.sx * self.sy;
        let den = ((self.n * self.sxx - self.sx * self.sx) * (self.n * self.syy - self.sy * self.sy))
            .max(0.0)
            .sqrt();
        if den < 1e-9 { 0.0 } else { num / den }
    }
}

// ---------------------------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Stim {
    start: u64,
    steer: f64,
    thr: f64,
    jump: bool,
    boost: bool,
}

impl Stim {
    fn responded(&self, a: &[f64; 6]) -> bool {
        let bits = a[5] as i32;
        (a[1] - self.steer).abs() > 0.4
            || (a[0] - self.thr).abs() > 0.9
            || ((bits & 1) != 0 && !self.jump)
            || (((bits & 2) != 0) != self.boost)
    }
}

/// Context the tracker can't observe itself (supplied when the report is built).
#[derive(Clone, Copy, Debug, Default)]
pub struct AcContext {
    /// Measured network round trip in ms (WebSocket ping/pong on the server).
    pub rtt_ms: f64,
}

/// Per-player, per-match result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcSummary {
    /// Model input vector (FEATURE_NAMES order).
    pub features: Vec<f32>,
    /// Named evidence for humans/LLMs: all model features plus hard checks, shadow-bot
    /// similarity and outcome stats.
    pub evidence: BTreeMap<String, f32>,
    pub live_secs: f32,
}

pub struct AcTracker {
    pub car_idx: usize,
    slot: u64,
    live_slots: u32,
    last: [f64; 6],
    has_last: bool,
    ch: [Chan; 3],
    pitch_eq_thr: u32,
    roll_nonzero: u32,
    handbrake: u32,
    jump: Btn,
    boost: Btn,
    // Hard client invariants (the official client can never violate these)
    viol_yaw: u32,
    viol_roll: u32,
    viol_arf: u32,
    // Reaction to unpredictable events vs. random control moments
    stim: Option<Stim>,
    ctrl: Option<Stim>,
    next_ctrl: u64,
    lcg: u64,
    react_slots: Vec<u16>,
    ctrl_slots: Vec<u16>,
    // Kickoffs
    kick_n: u32,
    kick_preheld: u32,
    kick_wait: Option<u64>,
    kick_delays: Vec<u16>,
    // Network
    depth_sum: f64,
    starved: u32,
    // Shadow bot
    ring: [[f64; 2]; LAGS],
    ring_pos: usize,
    ring_n: usize,
    corr_steer: [Corr; LAGS],
    corr_thr: [Corr; LAGS],
    shadow_jump_slots: Vec<u32>,
    shadow_prev_jump: bool,
    shadow_slots: u32,
    // Outcomes
    touches: u32,
    air_touches: u32,
}

impl AcTracker {
    pub fn new(car_idx: usize, seed: u64) -> Self {
        let lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1;
        Self {
            car_idx,
            slot: 0,
            live_slots: 0,
            last: [0.0; 6],
            has_last: false,
            ch: [Chan::default(), Chan::default(), Chan::default()],
            pitch_eq_thr: 0,
            roll_nonzero: 0,
            handbrake: 0,
            jump: Btn::default(),
            boost: Btn::default(),
            viol_yaw: 0,
            viol_roll: 0,
            viol_arf: 0,
            stim: None,
            ctrl: None,
            next_ctrl: 90,
            lcg,
            react_slots: Vec::new(),
            ctrl_slots: Vec::new(),
            kick_n: 0,
            kick_preheld: 0,
            kick_wait: None,
            kick_delays: Vec::new(),
            depth_sum: 0.0,
            starved: 0,
            ring: [[0.0; 2]; LAGS],
            ring_pos: 0,
            ring_n: 0,
            corr_steer: [Corr::default(); LAGS],
            corr_thr: [Corr::default(); LAGS],
            shadow_jump_slots: Vec::new(),
            shadow_prev_jump: false,
            shadow_slots: 0,
            touches: 0,
            air_touches: 0,
        }
    }

    fn rand_range(&mut self, lo: u64, hi: u64) -> u64 {
        self.lcg = self
            .lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (self.lcg >> 33) % (hi - lo).max(1)
    }

    pub fn live_secs(&self) -> f64 {
        f64::from(self.live_slots) * SLOT_MS / 1000.0
    }

    /// Hard client-invariant violations observed so far (slots).
    pub fn violations(&self) -> u32 {
        self.viol_yaw + self.viol_roll + self.viol_arf
    }

    /// Call once per input slot. `live` = live play (not countdown/goal/replay) and the car is not
    /// demolished. `applied` is exactly what the server applied for this player this slot.
    /// `shadow` is the shadow bot's packed controls for this slot (None if not computed).
    pub fn observe_slot(
        &mut self,
        live: bool,
        applied: &[f64; 6],
        shadow: Option<&[f64; 6]>,
        depth: usize,
        starved: bool,
    ) {
        self.slot += 1;
        let slot = self.slot;
        if !live {
            for c in &mut self.ch {
                c.break_run();
            }
            self.ring_n = 0;
            self.stim = None;
            self.ctrl = None;
            self.has_last = false;
            return;
        }
        self.live_slots += 1;
        let a = applied;
        let bits = a[5] as i32;

        // --- hard invariants of the official client ---
        // InputManager.controls(): out.yaw = out.steer, always.
        if (a[3] - a[1]).abs() > 1e-9 {
            self.viol_yaw += 1;
        }
        // roll comes only from digital keys/buttons (-1/0/1), except air-roll-free where roll = steer.
        let arf = (bits & 8) != 0;
        if arf {
            if (a[4] - a[1]).abs() > 1e-9 {
                self.viol_roll += 1;
            }
            // air-roll-free is only ever set while slide/handbrake is held
            if (bits & 4) == 0 {
                self.viol_arf += 1;
            }
        } else if a[4].abs() > 1e-9 && (a[4].abs() - 1.0).abs() > 1e-9 {
            self.viol_roll += 1;
        }

        // --- analog channels ---
        self.ch[0].observe(a[1]);
        self.ch[1].observe(a[0]);
        self.ch[2].observe(a[2]);
        if (a[2] - a[0]).abs() < 1e-9 || (a[2] + a[0]).abs() < 1e-9 {
            self.pitch_eq_thr += 1;
        }
        if !arf && a[4].abs() > 1e-9 {
            self.roll_nonzero += 1;
        }
        if (bits & 4) != 0 {
            self.handbrake += 1;
        }
        self.jump.observe((bits & 1) != 0, slot);
        self.boost.observe((bits & 2) != 0, slot);

        // --- reactions ---
        if let Some(s) = self.stim {
            if s.responded(a) {
                Self::push_lat(&mut self.react_slots, slot - s.start);
                self.stim = None;
            } else if slot - s.start > STIM_TIMEOUT {
                Self::push_lat(&mut self.react_slots, STIM_TIMEOUT + 1);
                self.stim = None;
            }
        }
        if let Some(s) = self.ctrl {
            if s.responded(a) {
                Self::push_lat(&mut self.ctrl_slots, slot - s.start);
                self.ctrl = None;
            } else if slot - s.start > STIM_TIMEOUT {
                Self::push_lat(&mut self.ctrl_slots, STIM_TIMEOUT + 1);
                self.ctrl = None;
            }
        }
        if self.ctrl.is_none() && slot >= self.next_ctrl {
            self.ctrl = Some(self.snapshot_stim(slot));
            self.next_ctrl = slot + self.rand_range(70, 200);
        }

        // --- kickoff ---
        if let Some(s) = self.kick_wait {
            if a[0] > 0.5 || (bits & 2) != 0 {
                if self.kick_delays.len() < MAX_SAMPLES {
                    self.kick_delays.push((slot - s).min(600) as u16);
                }
                self.kick_wait = None;
            } else if slot - s > 120 {
                self.kick_wait = None;
            }
        }

        self.depth_sum += depth as f64;
        if starved {
            self.starved += 1;
        }

        // --- shadow bot ---
        if let Some(s) = shadow {
            self.shadow_slots += 1;
            self.ring_pos = (self.ring_pos + 1) % LAGS;
            self.ring[self.ring_pos] = [s[1], s[0]];
            self.ring_n = (self.ring_n + 1).min(LAGS);
            for l in 0..self.ring_n {
                let idx = (self.ring_pos + LAGS - l) % LAGS;
                let [ss, st] = self.ring[idx];
                self.corr_steer[l].add(a[1], ss);
                self.corr_thr[l].add(a[0], st);
            }
            let sj = ((s[5] as i32) & 1) != 0;
            if sj && !self.shadow_prev_jump && self.shadow_jump_slots.len() < MAX_SAMPLES * 2 {
                self.shadow_jump_slots.push(slot as u32);
            }
            self.shadow_prev_jump = sj;
        } else {
            self.ring_n = 0;
        }

        self.last = *a;
        self.has_last = true;
    }

    fn push_lat(v: &mut Vec<u16>, slots: u64) {
        if v.len() < MAX_SAMPLES {
            v.push(slots.min(600) as u16);
        }
    }

    fn snapshot_stim(&self, slot: u64) -> Stim {
        let bits = self.last[5] as i32;
        Stim {
            start: slot,
            steer: self.last[1],
            thr: self.last[0],
            jump: (bits & 1) != 0,
            boost: (bits & 2) != 0,
        }
    }

    /// A ball touch happened this tick. Touches by other cars that sharply change the ball's
    /// motion are unpredictable stimuli; our own touches count toward outcome stats.
    pub fn on_hit(&mut self, w: &World, hit: &CarBallHit) {
        if hit.car_idx == self.car_idx {
            self.touches += 1;
            if w.cars.get(self.car_idx).is_some_and(|c| !c.on_ground) {
                self.air_touches += 1;
            }
            return;
        }
        if !self.has_last || self.stim.is_some() {
            return;
        }
        let Some(me) = w.cars.get(self.car_idx) else {
            return;
        };
        if me.demolished || me.pos.distance(w.ball.pos) > 55.0 {
            return;
        }
        if (hit.ball_speed - hit.prev_speed).abs() < 4.0 && hit.ball_speed < 10.0 {
            return;
        }
        self.stim = Some(self.snapshot_stim(self.slot));
    }

    /// The countdown just ended (kickoff "GO").
    pub fn on_kickoff_go(&mut self) {
        self.kick_n += 1;
        let bits = self.last[5] as i32;
        if self.has_last && (self.last[0] > 0.5 || (bits & 2) != 0) {
            self.kick_preheld += 1;
        } else {
            self.kick_wait = Some(self.slot);
        }
    }

    pub fn finish(&self, ctx: AcContext) -> AcSummary {
        let secs = self.live_secs().max(1.0);
        let mins = secs / 60.0;
        let mut f: Vec<f32> = Vec::with_capacity(FEATURE_NAMES.len());
        for c in &self.ch {
            c.features(secs, &mut f);
        }
        let live = f64::from(self.live_slots.max(1));
        f.push((f64::from(self.pitch_eq_thr) / live) as f32);

        let mut jump = self.jump.clone();
        jump.end_hold(self.slot + 1);
        let mut boost = self.boost.clone();
        boost.end_hold(self.slot + 1);

        let (jm, jcv) = mean_cv(&jump.holds);
        let (_, jgcv) = mean_cv(&jump.gaps);
        f.push((f64::from(jump.presses) / mins) as f32);
        f.push((jm * SLOT_MS) as f32);
        f.push(jcv as f32);
        f.push(mode_share(&jump.holds) as f32);
        f.push(if jump.holds.is_empty() {
            0.0
        } else {
            (jump.holds.iter().filter(|&&h| h <= 1).count() as f64 / jump.holds.len() as f64) as f32
        });
        f.push(jgcv as f32);

        let (bm, bcv) = mean_cv(&boost.holds);
        f.push((f64::from(boost.presses) / mins) as f32);
        f.push(((bm * SLOT_MS).max(1.0)).ln() as f32);
        f.push(bcv as f32);
        f.push(mode_share(&boost.holds) as f32);
        f.push((f64::from(boost.held_slots) / live) as f32);

        let depth_avg = self.depth_sum / live;
        let corr_ms = ctx.rtt_ms + depth_avg * SLOT_MS;
        let to_ms = |v: &[u16]| -> Vec<f64> {
            v.iter()
                .map(|&s| f64::from(s) * SLOT_MS - corr_ms)
                .collect()
        };
        let mut r = to_ms(&self.react_slots);
        let mut c = to_ms(&self.ctrl_slots);
        let fast = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().filter(|&&x| x < FAST_REACT_MS).count() as f64 / v.len() as f64
            }
        };
        let r_fast = fast(&r);
        let c_fast = fast(&c);
        let r_med = if r.is_empty() { 400.0 } else { quantile(&mut r, 0.5) };
        let r_p10 = if r.is_empty() { 300.0 } else { quantile(&mut r, 0.1) };
        let _ = quantile(&mut c, 0.5);
        f.push(r_med as f32);
        f.push(r_p10 as f32);
        f.push(r_fast as f32);
        f.push((r_fast - c_fast) as f32);
        f.push(c_fast as f32);

        let kick_n = f64::from(self.kick_n.max(1));
        f.push((f64::from(self.kick_preheld) / kick_n) as f32);
        let kd_std = if self.kick_delays.len() >= 3 {
            let (m, cv) = mean_cv(&self.kick_delays);
            m * cv * SLOT_MS
        } else {
            100.0
        };
        f.push(kd_std as f32);

        f.push(ctx.rtt_ms as f32);
        f.push(depth_avg as f32);
        f.push((f64::from(self.starved) / live) as f32);
        f.push((f64::from(self.roll_nonzero) / live) as f32);
        f.push((f64::from(self.handbrake) / live) as f32);
        // Shadow lead: how much better the player tracks the bot's intent at network-only lag
        // than at a human reaction lag. A human can't beat their own reaction time; a bot can.
        let lag_slot = |ms: f64| ((ms / SLOT_MS).round().max(0.0) as usize).min(LAGS - 1);
        let short_hi = lag_slot(corr_ms + 100.0);
        let long_lo = lag_slot(corr_ms + 150.0);
        let long_hi = lag_slot(corr_ms + 450.0);
        let max_r = |lo: usize, hi: usize| -> f64 {
            (lo..=hi).map(|l| self.corr_steer[l].r()).fold(0.0_f64, f64::max)
        };
        let short_r = max_r(0, short_hi);
        let long_r = if long_lo < LAGS - 1 { max_r(long_lo, long_hi) } else { self.corr_steer[LAGS - 1].r().max(0.0) };
        f.push((short_r - long_r) as f32);
        debug_assert_eq!(f.len(), FEATURE_NAMES.len());

        let mut ev: BTreeMap<String, f32> = BTreeMap::new();
        for (k, v) in FEATURE_NAMES.iter().zip(f.iter()) {
            ev.insert((*k).to_string(), *v);
        }
        ev.insert("live_secs".into(), secs as f32);
        ev.insert("react_n".into(), r.len() as f32);
        ev.insert("ctrl_n".into(), c.len() as f32);
        ev.insert("kick_n".into(), self.kick_n as f32);
        ev.insert("viol_yaw_slots".into(), self.viol_yaw as f32);
        ev.insert("viol_roll_slots".into(), self.viol_roll as f32);
        ev.insert("viol_arf_slots".into(), self.viol_arf as f32);
        ev.insert("viol_total".into(), self.violations() as f32);
        ev.insert("touches_per_min".into(), (f64::from(self.touches) / mins) as f32);
        ev.insert(
            "air_touch_frac".into(),
            (f64::from(self.air_touches) / f64::from(self.touches.max(1))) as f32,
        );

        // Shadow bot: best lag by steer correlation
        let mut best_l = 0usize;
        let mut best_r = f64::NEG_INFINITY;
        for l in 0..LAGS {
            let rr = self.corr_steer[l].r();
            if rr > best_r {
                best_r = rr;
                best_l = l;
            }
        }
        let cs = &self.corr_steer[best_l];
        let shadow_cov = f64::from(self.shadow_slots) / live;
        ev.insert("shadow_coverage".into(), shadow_cov as f32);
        ev.insert("shadow_steer_corr".into(), best_r.max(0.0) as f32);
        ev.insert("shadow_thr_corr".into(), self.corr_thr[best_l].r().max(0.0) as f32);
        ev.insert("shadow_lag_ms".into(), (best_l as f64 * SLOT_MS) as f32);
        ev.insert(
            "shadow_lag_minus_net_ms".into(),
            (best_l as f64 * SLOT_MS - corr_ms) as f32,
        );
        ev.insert(
            "shadow_steer_agree".into(),
            if cs.n > 0.0 { (cs.agree / cs.n) as f32 } else { 0.0 },
        );
        // Jump presses that coincide (±3 slots, at the best lag) with a shadow-bot jump press
        let mut hits = 0usize;
        let presses = &jump.press_slots;
        if !self.shadow_jump_slots.is_empty() {
            for &p in presses {
                let target = i64::from(p) - best_l as i64;
                let ok = self
                    .shadow_jump_slots
                    .binary_search_by(|&q| {
                        let q = i64::from(q);
                        if q < target - 3 {
                            std::cmp::Ordering::Less
                        } else if q > target + 3 {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .is_ok();
                if ok {
                    hits += 1;
                }
            }
        }
        ev.insert(
            "shadow_jump_agree".into(),
            if presses.len() >= 8 {
                (hits as f64 / presses.len() as f64) as f32
            } else {
                0.0
            },
        );
        ev.insert("jump_presses".into(), presses.len() as f32);

        AcSummary {
            features: f,
            evidence: ev,
            live_secs: secs as f32,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Shadow bot
// ---------------------------------------------------------------------------------------------

/// Runs the champion AI "in the player's seat" without affecting the match: it reads the live
/// world, and everything it writes (the car's controls, kickoff assignments) is restored.
pub struct ShadowBot {
    bot: BotAi,
}

impl ShadowBot {
    pub fn new<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            bot: BotAi::new_with_rng(4, 0, rng),
        }
    }

    pub fn reset(&mut self) {
        self.bot.reset();
    }

    pub fn predict(&mut self, car_idx: usize, dt: f64, w: &mut World) -> [f64; 6] {
        let saved_ctl = w.cars[car_idx].controls;
        let saved_kickers = w.brain.kickers;
        self.bot.update(car_idx, dt, w);
        let out = net_pack_controls(&w.cars[car_idx].controls);
        w.cars[car_idx].controls = saved_ctl;
        w.brain.kickers = saved_kickers;
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcLayer {
    /// Row-major [out][in]
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcModel {
    pub version: String,
    pub features: Vec<String>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub layers: Vec<AcLayer>,
    /// Reference ranges for synthetic humans: feature -> [p5, p50, p95]
    #[serde(default)]
    pub human_ref: BTreeMap<String, [f32; 3]>,
    /// Validation metrics recorded at training time
    #[serde(default)]
    pub metrics: BTreeMap<String, f32>,
}

const EMBEDDED_MODEL: &str = include_str!("../assets/ac_model.json");

impl AcModel {
    pub fn embedded() -> Self {
        Self::from_json(EMBEDDED_MODEL).unwrap_or_else(|e| {
            eprintln!("[ANTI-CHEAT] embedded model invalid ({e}); scoring disabled");
            Self::untrained()
        })
    }

    pub fn untrained() -> Self {
        Self {
            version: "untrained".into(),
            features: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            mean: vec![],
            std: vec![],
            layers: vec![],
            human_ref: BTreeMap::new(),
            metrics: BTreeMap::new(),
        }
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let m: AcModel = serde_json::from_str(s).map_err(|e| e.to_string())?;
        if m.layers.is_empty() {
            return Ok(m);
        }
        if m.features.len() != FEATURE_NAMES.len()
            || m.features.iter().zip(FEATURE_NAMES).any(|(a, b)| a != b)
        {
            return Err("feature list mismatch (retrain the model)".into());
        }
        let mut width = m.features.len();
        if m.mean.len() != width || m.std.len() != width {
            return Err("normalization size mismatch".into());
        }
        for l in &m.layers {
            if l.w.len() != l.b.len() || l.w.iter().any(|row| row.len() != width) {
                return Err("layer shape mismatch".into());
            }
            width = l.b.len();
        }
        if width != 1 {
            return Err("model must have one output".into());
        }
        Ok(m)
    }

    pub fn is_trained(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Probability in [0, 1] that the feature vector came from automated play. 0 if untrained.
    pub fn score(&self, features: &[f32]) -> f32 {
        if !self.is_trained() || features.len() != self.mean.len() {
            return 0.0;
        }
        let mut x: Vec<f32> = features
            .iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(v, (m, s))| {
                let v = if v.is_finite() { *v } else { *m };
                ((v - m) / s.max(1e-6)).clamp(-8.0, 8.0)
            })
            .collect();
        let last = self.layers.len() - 1;
        for (li, l) in self.layers.iter().enumerate() {
            let mut y = l.b.clone();
            for (o, row) in l.w.iter().enumerate() {
                let mut acc = y[o];
                for (wi, xi) in row.iter().zip(x.iter()) {
                    acc += wi * xi;
                }
                y[o] = if li < last { acc.max(0.0) } else { acc };
            }
            x = y;
        }
        1.0 / (1.0 + (-x[0]).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(thr: f64, steer: f64, pitch: f64, yaw: f64, roll: f64, bits: i32) -> [f64; 6] {
        [thr, steer, pitch, yaw, roll, f64::from(bits)]
    }

    #[test]
    fn official_client_inputs_never_violate_invariants() {
        let mut t = AcTracker::new(0, 1);
        for i in 0..600 {
            let s = ((i as f64) * 0.05).sin();
            let s = (s * 1000.0).round() / 1000.0;
            // keyboard + pad mixes the client can produce
            t.observe_slot(true, &pack(1.0, s, -0.3, s, 0.0, 2), None, 1, false);
            t.observe_slot(true, &pack(1.0, s, 0.0, s, -1.0, 1), None, 1, false);
            t.observe_slot(true, &pack(0.0, s, 0.0, s, s, 4 | 8), None, 1, false);
        }
        assert_eq!(t.violations(), 0);
    }

    #[test]
    fn raw_bot_controls_violate_invariants() {
        let mut t = AcTracker::new(0, 1);
        t.observe_slot(true, &pack(1.0, 0.3, 0.2, -0.5, 0.0, 0), None, 1, false);
        t.observe_slot(true, &pack(1.0, 0.3, 0.2, 0.3, 0.42, 0), None, 1, false);
        t.observe_slot(true, &pack(1.0, 0.3, 0.2, 0.3, 0.3, 8), None, 1, false);
        assert_eq!(t.violations(), 3);
    }

    #[test]
    fn feature_vector_matches_names() {
        let mut t = AcTracker::new(0, 7);
        for i in 0..2000 {
            let v = if i % 40 < 20 { 1.0 } else { 0.0 };
            t.observe_slot(true, &pack(v, -v, v, -v, 0.0, (i % 7 == 0) as i32), None, 1, false);
        }
        let s = t.finish(AcContext { rtt_ms: 50.0 });
        assert_eq!(s.features.len(), FEATURE_NAMES.len());
        assert!(s.features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embedded_model_loads_and_matches_training() {
        let m = AcModel::embedded();
        assert!(m.is_trained(), "embedded model should be trained");
        // Cross-check against the synthetic data if it's present (dev machines only)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/anticheat/data/synthetic.jsonl");
        let Ok(data) = std::fs::read_to_string(path) else { return };
        let (mut hum, mut bot, mut nh, mut nb) = (0.0, 0.0, 0.0, 0.0);
        for line in data.lines().take(3000) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let f: Vec<f32> = serde_json::from_value(v["features"].clone()).unwrap();
            let p = f64::from(m.score(&f));
            assert!((0.0..=1.0).contains(&p));
            if v["label"] == 0 { hum += p; nh += 1.0 } else { bot += p; nb += 1.0 }
        }
        eprintln!("mean score human {:.3} cheat {:.3}", hum / nh, bot / nb);
        assert!(hum / nh < 0.2 && bot / nb > 0.7);
    }

    #[test]
    fn untrained_model_scores_zero() {
        let m = AcModel::untrained();
        assert_eq!(m.score(&vec![0.0; FEATURE_NAMES.len()]), 0.0);
    }
}
