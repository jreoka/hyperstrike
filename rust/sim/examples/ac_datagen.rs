//! Synthetic training data for the behavioral anti-cheat model.
//!
//! Runs headless matches full of bots. A few "subject" cars per match are driven by a planner
//! (the built-in AI) whose intent is passed through either a HUMAN motor model (reaction delay,
//! keyboard / gamepad / touch device physics, button hold habits, lapses) or one of several CHEAT
//! models (raw bot, humanized bot, situational aim-assist, trigger-bot). Every subject's applied
//! inputs go through the exact same `AcTracker` + `ShadowBot` the live server uses, with simulated
//! network latency, jitter-buffer depth and starved slots.
//!
//! Usage: cargo run --release --example ac_datagen -- <out.jsonl> <matches> [threads] [seed]
//! Output: one JSON object per subject-match: {label, kind, device, rtt, features, evidence}

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Normal};
use sim::*;

const SLOT_DT: f64 = 2.0 / 120.0;
const HIST: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Human,
    RawBot,
    Humanized,
    Assist,
    Trigger,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Human => "human",
            Kind::RawBot => "raw_bot",
            Kind::Humanized => "humanized_bot",
            Kind::Assist => "assist",
            Kind::Trigger => "triggerbot",
        }
    }
    fn label(self) -> u8 {
        u8::from(self != Kind::Human)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Device {
    Keyboard,
    Pad,
    Touch,
}

impl Device {
    fn name(self) -> &'static str {
        match self {
            Device::Keyboard => "keyboard",
            Device::Pad => "pad",
            Device::Touch => "touch",
        }
    }
}

/// Persistent per-subject traits.
struct Persona {
    device: Device,
    rt_med_ms: f64,
    rt_sigma: f64,
    /// Low-latency delay (cheat engagement), ms
    cheat_rt_ms: f64,
    rtt_ms: f64,
    depth: usize,
    starve_p: f64,
    // keyboard
    pwm_period: u32,
    min_key_hold: u32,
    invert_y: bool,
    // pad / touch
    stick_tau_ms: f64,
    stick_noise: f64,
    deadzone: f64,
    trigger_full_p: f64,
    rest_pitch_sigma: f64,
    // buttons
    jump_hold_med: f64,
    boost_release_lag: u32,
    kick_prehold_p: f64,
    lapse_every_s: f64,
    // cheats
    preserve_invariants: bool,
    assist_bypass_device: bool,
    /// Weight of the subject's own wandering intent vs. the planner (humans are not bots)
    intent_noise: f64,
    miss_jump_p: f64,
}

impl Persona {
    fn sample(kind: Kind, rng: &mut StdRng) -> Self {
        let device = match rng.r#gen::<f64>() {
            x if x < 0.52 => Device::Keyboard,
            x if x < 0.88 => Device::Pad,
            _ => Device::Touch,
        };
        let rtt_ms = if rng.r#gen::<f64>() < 0.15 {
            rng.gen_range(150.0..320.0)
        } else {
            rng.gen_range(12.0..150.0)
        };
        let starve_p = if rng.r#gen::<f64>() < 0.25 {
            rng.gen_range(0.005..0.06)
        } else {
            rng.gen_range(0.0..0.005)
        };
        Self {
            device,
            rt_med_ms: rng.gen_range(165.0..330.0),
            rt_sigma: rng.gen_range(0.12..0.35),
            cheat_rt_ms: match kind {
                Kind::RawBot => rng.gen_range(0.0..25.0),
                _ => rng.gen_range(0.0..95.0),
            },
            rtt_ms,
            depth: rng.gen_range(1..=3),
            starve_p,
            pwm_period: rng.gen_range(7..18),
            min_key_hold: rng.gen_range(3..7),
            invert_y: rng.r#gen::<f64>() < 0.12,
            stick_tau_ms: rng.gen_range(25.0..110.0),
            stick_noise: rng.gen_range(0.01..0.08),
            deadzone: rng.gen_range(0.05..0.25),
            trigger_full_p: rng.gen_range(0.5..0.98),
            rest_pitch_sigma: rng.gen_range(0.02..0.35),
            jump_hold_med: rng.gen_range(3.0..9.0),
            boost_release_lag: rng.gen_range(1..9),
            kick_prehold_p: rng.gen_range(0.05..0.95),
            lapse_every_s: rng.gen_range(15.0..90.0),
            preserve_invariants: rng.r#gen::<f64>() < 0.5,
            assist_bypass_device: rng.r#gen::<f64>() < 0.3,
            intent_noise: match kind {
                Kind::Human | Kind::Assist | Kind::Trigger => rng.gen_range(0.0..0.85),
                Kind::Humanized => rng.gen_range(0.0..0.3),
                Kind::RawBot => 0.0,
            },
            miss_jump_p: rng.gen_range(0.0..0.4),
        }
    }
}

fn round3(v: f64) -> f64 {
    let r = (v.clamp(-1.0, 1.0) * 1000.0).round() / 1000.0;
    if r == 0.0 { 0.0 } else { r }
}

/// Runs a bot's policy for a car without letting it touch the match.
fn bot_controls(bot: &mut BotAi, ci: usize, dt: f64, w: &mut World) -> Controls {
    let saved = w.cars[ci].controls;
    let kick = w.brain.kickers;
    bot.update(ci, dt, w);
    let out = w.cars[ci].controls;
    w.cars[ci].controls = saved;
    w.brain.kickers = kick;
    out
}

/// Device + motor state that turns (delayed) intent into raw client controls.
#[derive(Default)]
struct Motor {
    // keyboard steering
    key_steer: i8,
    key_changed_at: u64,
    pwm_phase: u32,
    key_thr: i8,
    thr_changed_at: u64,
    // stick
    sx: f64,
    sy: f64,
    nx: f64,
    ny: f64,
    rest_y: f64,
    trig: f64,
    // buttons
    jump_down: bool,
    jump_since: u64,
    jump_hold_target: u64,
    jump_pending: Option<u64>,
    prev_want_jump: bool,
    boost_down: bool,
    boost_off_for: u32,
    boost_on_for: u32,
    hb_down: bool,
    item_down: bool,
    // lapses
    lapse_until: u64,
    lapse_ctl: Option<Controls>,
    // own intent (OU wander)
    ou_steer: f64,
    ou_thr: f64,
    drop_jump: bool,
}

struct Subject {
    ci: usize,
    kind: Kind,
    p: Persona,
    planner: BotAi,
    shadow: ShadowBot,
    tracker: AcTracker,
    hist: VecDeque<Controls>,
    rt_z: f64,
    m: Motor,
    last_applied: [f64; 6],
    preheld_this_kick: bool,
    engaged_slots: u32,
}

impl Subject {
    fn delayed(&self, delay_slots: usize) -> Controls {
        let d = delay_slots.min(self.hist.len().saturating_sub(1));
        self.hist[self.hist.len() - 1 - d]
    }

    fn net_slots(&self) -> f64 {
        (self.p.rtt_ms + (1.0 + self.p.depth as f64) * anticheat::SLOT_MS) / anticheat::SLOT_MS
    }

    fn engaged(&self, w: &World) -> bool {
        let car = &w.cars[self.ci];
        let d = car.pos.distance(w.ball.pos);
        d < 20.0 || (!car.on_ground && d < 28.0)
    }

    /// Produce this slot's applied input.
    fn produce(&mut self, slot: u64, w: &World, countdown_left: Option<u64>, rng: &mut StdRng) -> [f64; 6] {
        let net = self.net_slots();
        // OU process for the human reaction delay
        self.rt_z += -0.02 * self.rt_z + 0.2 * self.p.rt_sigma * Normal::new(0.0, 1.0).unwrap().sample(rng);
        let human_rt = (self.p.rt_med_ms * self.rt_z.exp()).clamp(110.0, 650.0);
        let fast_rt = self.p.cheat_rt_ms;

        let engaged = self.engaged(w);
        let (rt_ms, use_device, bot_buttons) = match self.kind {
            Kind::Human => (human_rt, true, false),
            Kind::RawBot => (fast_rt, false, true),
            Kind::Humanized => (fast_rt + 15.0 * rng.r#gen::<f64>(), true, false),
            Kind::Assist => {
                if engaged {
                    self.engaged_slots += 1;
                    (fast_rt, !self.p.assist_bypass_device, self.p.assist_bypass_device)
                } else {
                    (human_rt, true, false)
                }
            }
            Kind::Trigger => (human_rt, true, false),
        };
        let delay = (net + rt_ms / anticheat::SLOT_MS).round() as usize;
        let mut want = self.delayed(delay);
        if self.kind == Kind::Trigger {
            let fast = self.delayed((net + fast_rt / anticheat::SLOT_MS).round() as usize);
            want.jump = fast.jump;
            want.boost = fast.boost;
        }

        // Humans are not the planner: blend in their own wandering intent, miss some jumps.
        let human_part = match self.kind {
            Kind::Human | Kind::Trigger => true,
            Kind::Assist => !engaged,
            Kind::Humanized => true,
            Kind::RawBot => false,
        };
        if human_part && self.p.intent_noise > 0.0 {
            let n01 = Normal::new(0.0, 1.0).unwrap();
            self.m.ou_steer += -0.04 * self.m.ou_steer + 0.18 * n01.sample(rng);
            self.m.ou_thr += -0.03 * (self.m.ou_thr - 0.6) + 0.12 * n01.sample(rng);
            let k = self.p.intent_noise;
            want.steer = ((1.0 - k) * want.steer + k * self.m.ou_steer.clamp(-1.0, 1.0)).clamp(-1.0, 1.0);
            want.throttle = ((1.0 - k * 0.6) * want.throttle + k * 0.6 * self.m.ou_thr.clamp(-1.0, 1.0)).clamp(-1.0, 1.0);
            if want.jump && !self.m.prev_want_jump {
                self.m.drop_jump = rng.r#gen::<f64>() < self.p.miss_jump_p * k;
            }
            if self.m.drop_jump {
                if want.jump {
                    want.jump = false;
                } else {
                    self.m.drop_jump = false;
                }
            }
            if rng.r#gen::<f64>() < k * 0.004 {
                want.jump = true;
            }
        }

        // Kickoff pre-hold (humans often hold throttle through "3, 2, 1")
        if let Some(left) = countdown_left
            && left < 40
            && self.preheld_this_kick
            && self.kind != Kind::RawBot
        {
            want.throttle = 1.0;
            want.boost = left < 10 && rng.r#gen::<f64>() < 0.5;
        }

        // Lapses: brief stretches of stale/neutral input
        if self.kind != Kind::RawBot {
            if slot < self.m.lapse_until {
                if let Some(c) = self.m.lapse_ctl {
                    want = c;
                }
            } else if rng.r#gen::<f64>() < anticheat::SLOT_MS / 1000.0 / self.p.lapse_every_s {
                self.m.lapse_until = slot + rng.gen_range(15..80);
                self.m.lapse_ctl = Some(if rng.r#gen::<f64>() < 0.5 { want } else { Controls::default() });
            }
        }

        let car = &w.cars[self.ci];
        let airborne = !car.on_ground;
        let mut out = Controls::default();

        if !use_device {
            out = want;
            if self.p.preserve_invariants || self.kind != Kind::RawBot {
                out.yaw = out.steer;
                out.roll = if out.roll > 0.5 {
                    1.0
                } else if out.roll < -0.5 {
                    -1.0
                } else {
                    0.0
                };
                out.air_roll_free = false;
            }
        } else {
            match self.p.device {
                Device::Keyboard => self.keyboard(slot, &want, airborne, &mut out, rng),
                Device::Pad => self.pad(&want, airborne, &mut out, rng),
                Device::Touch => self.touch(&want, &mut out, rng),
            }
            // roll: digital Q/E or bumpers
            out.roll = if airborne && want.roll > 0.5 {
                1.0
            } else if airborne && want.roll < -0.5 {
                -1.0
            } else {
                0.0
            };
            out.handbrake = self.hb(&want, rng);
            if out.handbrake && out.roll.abs() < 0.01 {
                out.roll = round3(out.steer);
                out.air_roll_free = true;
            }
            out.yaw = out.steer;
        }

        if bot_buttons || !use_device {
            out.jump = want.jump;
            out.boost = want.boost;
            out.item = want.item;
        } else {
            out.jump = self.jump_btn(slot, want.jump, rng);
            out.boost = self.boost_btn(want.boost, rng);
            out.item = if want.item && !self.m.item_down {
                rng.r#gen::<f64>() < 0.5
            } else {
                want.item && self.m.item_down
            };
            self.m.item_down = out.item;
        }

        let mut packed = net_pack_controls(&out);
        if use_device {
            // the official client always sends yaw == steer after rounding
            packed[3] = packed[1];
            if (packed[5] as i32) & 8 != 0 {
                packed[4] = packed[1];
            }
        }
        packed
    }

    fn keyboard(&mut self, slot: u64, want: &Controls, airborne: bool, out: &mut Controls, rng: &mut StdRng) {
        let s = want.steer;
        let a = s.abs();
        let desired: i8 = if a >= 0.82 {
            s.signum() as i8
        } else if a >= 0.16 {
            // feathering: pulse the key with duty ~ |s|
            self.m.pwm_phase = (self.m.pwm_phase + 1) % self.p.pwm_period;
            let on = (f64::from(self.m.pwm_phase) / f64::from(self.p.pwm_period)) < a;
            if on { s.signum() as i8 } else { 0 }
        } else {
            0
        };
        let hold_ok = slot.saturating_sub(self.m.key_changed_at) >= u64::from(self.p.min_key_hold)
            + u64::from(rng.gen_range(0..3u32));
        if desired != self.m.key_steer && hold_ok {
            self.m.key_steer = desired;
            self.m.key_changed_at = slot;
        }
        out.steer = f64::from(self.m.key_steer);

        let src = if airborne { want.pitch } else { want.throttle };
        let want_thr: i8 = if src > 0.25 {
            1
        } else if src < -0.35 {
            -1
        } else {
            0
        };
        if want_thr != self.m.key_thr
            && slot.saturating_sub(self.m.thr_changed_at) >= u64::from(self.p.min_key_hold)
        {
            self.m.key_thr = want_thr;
            self.m.thr_changed_at = slot;
        }
        out.throttle = f64::from(self.m.key_thr);
        out.pitch = if self.p.invert_y { -out.throttle } else { out.throttle };
    }

    fn stick_step(&self, cur: f64, target: f64, rng: &mut StdRng) -> f64 {
        let alpha = 1.0 - (-anticheat::SLOT_MS / self.p.stick_tau_ms).exp();
        let mut v = cur + (target - cur) * alpha;
        v += self.p.stick_noise * Normal::new(0.0, 1.0).unwrap().sample(rng) * 0.5;
        v.clamp(-1.0, 1.0)
    }

    fn deadzone(&self, v: f64) -> f64 {
        let dz = self.p.deadzone;
        if v.abs() < dz { 0.0 } else { v.signum() * (v.abs() - dz) / (1.0 - dz) }
    }

    fn pad(&mut self, want: &Controls, airborne: bool, out: &mut Controls, rng: &mut StdRng) {
        // humans slam the stick to the gate when they want full lock
        let tx = if want.steer.abs() > 0.85 { want.steer.signum() * 1.05 } else { want.steer };
        self.m.sx = self.stick_step(self.m.sx, tx, rng);
        if self.m.sx.abs() > 0.97 && tx.abs() > 1.0 {
            self.m.sx = self.m.sx.signum();
        }
        self.m.rest_y += -0.01 * self.m.rest_y
            + self.p.rest_pitch_sigma * 0.08 * Normal::new(0.0, 1.0).unwrap().sample(rng);
        let ty = if airborne { -want.pitch } else { self.m.rest_y.clamp(-0.6, 0.6) };
        self.m.sy = self.stick_step(self.m.sy, ty, rng);
        out.steer = round3(self.deadzone(self.m.sx));
        let ly = self.deadzone(self.m.sy);
        out.pitch = round3(if ly.abs() > 0.0 { -ly } else { 0.0 });
        // triggers
        let tt = if want.throttle > 0.3 {
            if rng.r#gen::<f64>() < self.p.trigger_full_p { 1.0 } else { want.throttle.clamp(0.3, 1.0) }
        } else if want.throttle < -0.3 {
            -1.0
        } else {
            0.0
        };
        self.m.trig += (tt - self.m.trig) * 0.6;
        let tr = if self.m.trig.abs() > 0.05 { self.m.trig + 0.01 * Normal::new(0.0, 1.0).unwrap().sample(rng) } else { 0.0 };
        out.throttle = round3(if (tr - 1.0).abs() < 0.03 { 1.0 } else { tr });
    }

    fn touch(&mut self, want: &Controls, out: &mut Controls, rng: &mut StdRng) {
        let mut tx = want.steer;
        let mut ty = want.throttle;
        let len = (tx * tx + ty * ty).sqrt();
        if len > 1.0 {
            tx /= len;
            ty /= len;
        }
        let alpha = 1.0 - (-anticheat::SLOT_MS / (self.p.stick_tau_ms * 1.4)).exp();
        let noise = self.p.stick_noise * 1.6;
        self.m.nx += (tx - self.m.nx) * alpha + noise * Normal::new(0.0, 1.0).unwrap().sample(rng) * 0.5;
        self.m.ny += (ty - self.m.ny) * alpha + noise * Normal::new(0.0, 1.0).unwrap().sample(rng) * 0.5;
        let l = (self.m.nx * self.m.nx + self.m.ny * self.m.ny).sqrt();
        let sc = if l > 1.0 { 1.0 / l } else { 1.0 };
        let (nx, ny) = (self.m.nx * sc, self.m.ny * sc);
        out.steer = round3(if nx.abs() < 0.1 { 0.0 } else { nx });
        out.throttle = round3(if ny.abs() < 0.1 { 0.0 } else { ny });
        out.pitch = out.throttle;
    }

    fn jump_btn(&mut self, slot: u64, want: bool, rng: &mut StdRng) -> bool {
        let rising = want && !self.m.prev_want_jump;
        self.m.prev_want_jump = want;
        if rising && self.m.jump_pending.is_none() {
            self.m.jump_pending = Some(slot + rng.gen_range(0..3u64));
        }
        if self.m.jump_down {
            let held = slot - self.m.jump_since;
            if held >= self.m.jump_hold_target && (!want || held > 40) {
                self.m.jump_down = false;
            } else if rising && held >= 2 {
                // a second press (flip) while still holding: release now, re-press next slot
                self.m.jump_down = false;
                self.m.jump_pending = Some(slot + 1);
            }
        } else if let Some(at) = self.m.jump_pending
            && slot >= at
        {
            self.m.jump_down = true;
            self.m.jump_since = slot;
            self.m.jump_pending = None;
            let ln = LogNormal::new(self.p.jump_hold_med.ln(), 0.35).unwrap();
            self.m.jump_hold_target = (ln.sample(rng).round() as u64).clamp(2, 30);
        }
        self.m.jump_down
    }

    fn boost_btn(&mut self, want: bool, rng: &mut StdRng) -> bool {
        if self.m.boost_down {
            self.m.boost_on_for += 1;
            if want {
                self.m.boost_off_for = 0;
            } else {
                self.m.boost_off_for += 1;
                if self.m.boost_off_for > self.p.boost_release_lag + rng.gen_range(0..3u32)
                    && self.m.boost_on_for >= 3
                {
                    self.m.boost_down = false;
                    self.m.boost_on_for = 0;
                }
            }
        } else if want && rng.r#gen::<f64>() < 0.7 {
            self.m.boost_down = true;
            self.m.boost_off_for = 0;
            self.m.boost_on_for = 0;
        }
        self.m.boost_down
    }

    fn hb(&mut self, want: &Controls, rng: &mut StdRng) -> bool {
        if want.handbrake != self.m.hb_down && rng.r#gen::<f64>() < 0.4 {
            self.m.hb_down = want.handbrake;
        }
        self.m.hb_down
    }
}

fn pick_kind(rng: &mut StdRng) -> Kind {
    match rng.r#gen::<f64>() {
        x if x < 0.5 => Kind::Human,
        x if x < 0.62 => Kind::RawBot,
        x if x < 0.76 => Kind::Humanized,
        x if x < 0.91 => Kind::Assist,
        _ => Kind::Trigger,
    }
}

fn kickoff(w: &mut World, n: usize) {
    let r = w.phys.ball_r;
    w.ball.reset(0.0, r, 0.0);
    w.ball.frozen = false;
    let sets = kickoff_sets_for(n);
    let order: Vec<usize> = pick(sets, &mut w.rng).to_vec();
    for team in [0u8, 1u8] {
        let idx: Vec<usize> = (0..w.cars.len()).filter(|&i| w.cars[i].team == team).collect();
        for (i, ci) in idx.into_iter().enumerate() {
            let spot = KICKOFF_SPOTS[order[i % order.len()]];
            let (x, z) = w.shape.kickoff_pos(spot, team);
            let yaw = yaw_toward(x, z, 0.0, 0.0);
            let car = &mut w.cars[ci];
            car.place(x, 0.35, z, yaw);
            car.boost = 33.0;
            car.demolished = false;
            if let Some(bot) = car.bot.as_mut() {
                bot.reset();
            }
        }
    }
    w.reset_pads();
    w.kickoff = true;
    w.frozen = true;
    w.brain.choose_kickers(&w.cars, &w.ball);
}

fn play_match(seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n: usize = match rng.r#gen::<f64>() {
        x if x < 0.4 => 1,
        x if x < 0.75 => 2,
        _ => 3,
    };
    let arena = rng.gen_range(0..12usize).min(ARENAS.len() - 1);
    let mut w = World::with_seed(ARENAS[arena], &ArenaOpts::default(), seed ^ 0x9e37);
    w.allow_demos = true;
    if rng.r#gen::<f64>() < 0.15 {
        w.powers = Some(PowerSystem::new(10.0));
    }
    for team in [0u8, 1u8] {
        for k in 0..n {
            let ci = w.add_car(Car::new(w.cars.len(), team, format!("C{team}{k}"), rng.gen_range(0..HITBOXES.len())));
            let d = rng.gen_range(1..=4usize);
            w.cars[ci].bot = Some(BotAi::new_with_rng(d, rng.gen_range(0..PERSONALITIES.len()), &mut rng));
        }
    }
    let n_subjects = rng.gen_range(1..=(2 * n).min(3));
    let mut cand: Vec<usize> = (0..w.cars.len()).collect();
    let mut subjects: Vec<Subject> = Vec::new();
    for _ in 0..n_subjects {
        let ci = cand.remove(rng.gen_range(0..cand.len()));
        w.cars[ci].bot = None;
        let kind = pick_kind(&mut rng);
        let p = Persona::sample(kind, &mut rng);
        // humans of every skill; cheats use the stronger planners
        let diff = if kind == Kind::Human { rng.gen_range(0..=4usize) } else { rng.gen_range(3..=4usize) };
        subjects.push(Subject {
            ci,
            kind,
            planner: BotAi::new_with_rng(diff, rng.gen_range(0..PERSONALITIES.len()), &mut rng),
            shadow: ShadowBot::new(&mut rng),
            tracker: AcTracker::new(ci, seed.wrapping_add(ci as u64)),
            hist: VecDeque::from(vec![Controls::default(); 4]),
            rt_z: 0.0,
            m: Motor::default(),
            last_applied: [0.0; 6],
            preheld_this_kick: false,
            engaged_slots: 0,
            p,
        });
    }

    let live_target_s: f64 = rng.gen_range(100.0..300.0);
    let dt = w.phys.dt;
    let mut slot: u64 = 0;
    let mut live_slots: u64 = 0;
    let mut countdown: u64 = 180;
    let mut pause: u64 = 0;
    kickoff(&mut w, n);
    for s in &mut subjects {
        s.preheld_this_kick = rng.r#gen::<f64>() < s.p.kick_prehold_p;
    }
    let mut play_t = 0.0;

    while (live_slots as f64) * anticheat::SLOT_MS / 1000.0 < live_target_s {
        slot += 1;
        let live = countdown == 0 && pause == 0;
        let cd_left = if countdown > 0 { Some(countdown) } else { None };

        // planner intents (always computed, as a human would keep thinking)
        for s in &mut subjects {
            let c = bot_controls(&mut s.planner, s.ci, SLOT_DT, &mut w);
            s.hist.push_back(c);
            if s.hist.len() > HIST {
                s.hist.pop_front();
            }
        }
        for s in &mut subjects {
            let mut applied = s.produce(slot, &w, cd_left, &mut rng);
            let starved = rng.r#gen::<f64>() < s.p.starve_p;
            if starved {
                applied = s.last_applied;
            }
            s.last_applied = applied;
            let car_live = live && !w.cars[s.ci].demolished;
            let shadow = if car_live { Some(s.shadow.predict(s.ci, SLOT_DT, &mut w)) } else { None };
            s.tracker.observe_slot(car_live, &applied, shadow.as_ref(), s.p.depth, starved);
            net_unpack_controls(&applied, &mut w.cars[s.ci].controls);
        }

        for _ in 0..2 {
            w.step(dt);
            if !live {
                continue;
            }
            let evs = std::mem::take(&mut w.events);
            for e in &evs {
                if let WorldEvent::Hit(h) = e {
                    for s in &mut subjects {
                        s.tracker.on_hit(&w, h);
                    }
                }
            }
            if w.kickoff && (w.ball.last_touch.is_some() || play_t > 6.0) {
                w.kickoff = false;
            }
        }
        if live {
            live_slots += 1;
            play_t += SLOT_DT;
            if w.shape.in_goal(w.ball.pos, w.phys.ball_r) >= 0 {
                pause = 90;
            }
        } else if countdown > 0 {
            countdown -= 1;
            if countdown == 0 {
                w.frozen = false;
                play_t = 0.0;
                for s in &mut subjects {
                    s.tracker.on_kickoff_go();
                }
            }
        } else if pause > 0 {
            pause -= 1;
            if pause == 0 {
                kickoff(&mut w, n);
                for s in &mut subjects {
                    s.shadow.reset();
                    s.preheld_this_kick = rng.r#gen::<f64>() < s.p.kick_prehold_p;
                }
                countdown = 180;
            }
        }
    }

    subjects
        .iter()
        .map(|s| {
            let measured_rtt = (s.p.rtt_ms * (1.0 + 0.08 * Normal::new(0.0, 1.0).unwrap().sample(&mut rng))).max(1.0);
            let sum = s.tracker.finish(AcContext { rtt_ms: measured_rtt });
            serde_json::json!({
                "label": s.kind.label(),
                "kind": s.kind.name(),
                "device": s.p.device.name(),
                "rtt": measured_rtt,
                "engaged_frac": f64::from(s.engaged_slots) / (sum.live_secs as f64 * 60.0).max(1.0),
                "features": sum.features,
                "evidence": sum.evidence,
            })
            .to_string()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).cloned().unwrap_or_else(|| "ac_data.jsonl".into());
    let matches: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let threads: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

    let file = Arc::new(Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(&out_path).expect("create output"),
    )));
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let t0 = std::time::Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (file, next, done) = (Arc::clone(&file), Arc::clone(&next), Arc::clone(&done));
            std::thread::spawn(move || {
                loop {
                    let m = next.fetch_add(1, Ordering::Relaxed);
                    if m >= matches {
                        break;
                    }
                    let lines = play_match(seed.wrapping_mul(1_000_003).wrapping_add(m as u64));
                    {
                        let mut f = file.lock().unwrap();
                        for l in lines {
                            let _ = writeln!(f, "{l}");
                        }
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 50 == 0 || d == matches {
                        eprintln!("[datagen] {d}/{matches} matches ({:.0}s)", t0.elapsed().as_secs_f64());
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    file.lock().unwrap().flush().unwrap();
}
