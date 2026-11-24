use std::collections::HashMap;
use std::sync::OnceLock;

use glam::DVec3;
use rand::Rng;

use crate::arena::GoalMode;
use crate::math_ext::{
    Controls, PhysConfig, THROTTLE_CURVE, ThreeVec3Ext, clamp, curve, lerp, math_sign, sign,
};
use crate::physics::{Ball, Car, PredSlice};
use crate::policy::{
    OBS_DIM, POLICY_TICK_SKIP, decode_action, embedded_savant_policy, extract_obs,
    policy_supports_world,
};
use crate::powers::PowerTarget;
use crate::strike::{OppSnap, StrikeCtx, StrikePlan, simulate_strike, strike_controls};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct Difficulty {
    pub name: &'static str,
    pub react: f64,
    pub aim: f64,
    pub speed: f64,
    pub boost: f64,
    pub aerial: f64,
    pub aerial_h: f64,
    pub flip: f64,
    pub jump_shot: f64,
    pub rot: f64,
    pub kick: f64,
    pub err: f64,
    /// >= 0.5: recover from landing on the roof; >= 0.75: also skip wasteful ground speed-flips.
    pub fix: f64,
    /// Simulated shot planner replan interval in seconds (0 = planner off).
    pub plan: f64,
}

pub const DIFFICULTY: [Difficulty; 6] = [
    Difficulty {
        name: "ROOKIE",
        react: 0.42,
        aim: 0.4,
        speed: 0.76,
        boost: 0.4,
        aerial: 0.0,
        aerial_h: 0.0,
        flip: 0.15,
        jump_shot: 0.35,
        rot: 0.45,
        kick: 0.5,
        fix: 0.0,
        plan: 0.0,
        err: 0.9,
    },
    Difficulty {
        name: "PRO",
        react: 0.3,
        aim: 0.23,
        speed: 0.88,
        boost: 0.8,
        aerial: 0.4,
        aerial_h: 7.5,
        flip: 0.6,
        jump_shot: 0.75,
        rot: 0.75,
        kick: 0.8,
        fix: 0.0,
        plan: 0.0,
        err: 0.5,
    },
    Difficulty {
        name: "ALL-STAR",
        react: 0.11,
        aim: 0.07,
        speed: 0.97,
        boost: 1.0,
        aerial: 0.8,
        aerial_h: 13.0,
        flip: 0.9,
        jump_shot: 0.95,
        rot: 0.9,
        kick: 0.95,
        fix: 0.0,
        plan: 0.0,
        err: 0.25,
    },
    Difficulty {
        name: "CHAMP",
        react: 0.035,
        aim: 0.015,
        speed: 1.0,
        boost: 1.0,
        aerial: 1.0,
        aerial_h: 18.0,
        flip: 1.0,
        jump_shot: 1.0,
        rot: 1.0,
        kick: 1.0,
        fix: 0.5,
        plan: 0.0,
        err: 0.05,
    },
    Difficulty {
        name: "LEGEND",
        react: 0.02,
        aim: 0.005,
        speed: 1.0,
        boost: 1.0,
        aerial: 1.0,
        aerial_h: 22.0,
        flip: 1.0,
        jump_shot: 1.0,
        rot: 1.0,
        kick: 1.0,
        fix: 1.0,
        plan: 0.15,
        err: 0.02,
    },
    Difficulty {
        name: "SAVANT",
        react: 0.016,
        aim: 0.0,
        speed: 1.0,
        boost: 1.0,
        aerial: 1.0,
        aerial_h: 25.0,
        flip: 1.0,
        jump_shot: 1.0,
        rot: 1.0,
        kick: 1.0,
        fix: 1.0,
        plan: 0.06,
        err: 0.0,
    },
];

#[derive(Clone, Copy, Debug)]
pub struct Personality {
    pub name: &'static str,
    pub aggro: f64,
    pub defend: f64,
    pub air: f64,
    pub pass: f64,
}

pub const PERSONALITIES: [Personality; 5] = [
    Personality {
        name: "BALANCED",
        aggro: 0.5,
        defend: 0.5,
        air: 0.5,
        pass: 0.4,
    },
    Personality {
        name: "AGGRESSOR",
        aggro: 0.9,
        defend: 0.25,
        air: 0.5,
        pass: 0.15,
    },
    Personality {
        name: "GUARDIAN",
        aggro: 0.3,
        defend: 0.9,
        air: 0.3,
        pass: 0.4,
    },
    Personality {
        name: "PLAYMAKER",
        aggro: 0.55,
        defend: 0.5,
        air: 0.4,
        pass: 0.9,
    },
    Personality {
        name: "AERIALIST",
        aggro: 0.6,
        defend: 0.4,
        air: 1.0,
        pass: 0.3,
    },
];

pub const BOT_NAMES: [&str; 24] = [
    "BOLT", "NOVA", "VEX", "ROOK", "ECHO", "BLAZE", "KITE", "FROST", "RUMBLE", "JUNO", "ACE",
    "VIPER", "COMET", "DRIFT", "TITAN", "PIXEL", "SABLE", "ONYX", "ZEPHYR", "RIVET", "MAVERICK",
    "SPARK", "HALO", "GLITCH",
];

struct AccelTable {
    dt: f64,
    n: usize,
    tables: [Vec<Vec<f32>>; 2],
}

fn accel_table() -> &'static AccelTable {
    static TABLE: OnceLock<AccelTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let phys = PhysConfig::default();
        let dt = 0.05;
        let n = 120;
        let mut tables = [Vec::with_capacity(24), Vec::with_capacity(24)];
        for (b, tab_b) in tables.iter_mut().enumerate() {
            for v0 in 0..=23 {
                let mut arr = vec![0.0_f32; n];
                let mut v = v0 as f64;
                let mut d = 0.0;
                for item in arr.iter_mut() {
                    let a = curve(&THROTTLE_CURVE, v) + if b != 0 { phys.boost_acc } else { 0.0 };
                    v = (v + a * dt).min(phys.car_max);
                    d += v * dt;
                    *item = d as f32;
                }
                tab_b.push(arr);
            }
        }
        AccelTable { dt, n, tables }
    })
}

pub fn time_to_cover(dist: f64, v0: f64, boost: bool) -> f64 {
    if dist <= 0.0 {
        return 0.0;
    }
    let at = accel_table();
    let idx = clamp(v0.round(), 0.0, 23.0) as usize;
    let tab = &at.tables[usize::from(boost)][idx];
    let last = tab[at.n - 1] as f64;
    if dist > last {
        return at.n as f64 * at.dt + (dist - last) / 23.0;
    }
    let mut lo = 0;
    let mut hi = at.n - 1;
    while hi - lo > 1 {
        let m = (lo + hi) >> 1;
        if (tab[m] as f64) < dist {
            lo = m;
        } else {
            hi = m;
        }
    }
    (hi + 1) as f64 * at.dt
}

fn jump_profile() -> &'static [Vec<f64>; 2] {
    static PROF: OnceLock<[Vec<f64>; 2]> = OnceLock::new();
    PROF.get_or_init(|| {
        let phys = PhysConfig::default();
        let dt = 1.0 / 120.0;
        let mut out = [Vec::with_capacity(240), Vec::with_capacity(240)];
        for (dbl, arr) in out.iter_mut().enumerate() {
            let mut v = phys.jump_imp;
            let mut h = 0.0;
            for i in 0..240 {
                let t = i as f64 * dt;
                let mut a = -phys.g;
                if t < 0.2 {
                    a += phys.jump_hold;
                }
                if dbl != 0 && (t - 0.24).abs() < dt * 0.5 {
                    v += phys.jump_imp;
                }
                v += a * dt;
                h += v * dt;
                arr.push(h);
            }
        }
        out
    })
}

pub fn jump_time_for(h: f64, dbl: bool) -> f64 {
    let p = &jump_profile()[usize::from(dbl)];
    for i in 0..p.len() {
        if p[i] >= h {
            return i as f64 / 120.0;
        }
        if i > 30 && p[i] < p[i - 1] {
            break;
        }
    }
    -1.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BotMode {
    Idle,
    Kickoff,
    Attack,
    AerialPrep,
    Save,
    Shadow,
    Support,
    Boost,
    Defend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Attack,
    Support,
    Defend,
}

#[derive(Clone, Copy, Debug)]
pub struct SupportSlot {
    pub i: usize,
    pub n: usize,
}

#[derive(Clone, Copy, Debug)]
struct Intercept {
    slice: PredSlice,
    t: f64,
    aerial: bool,
    late: bool,
}

#[derive(Clone, Copy, Debug)]
struct SeqStep {
    t: f64,
    jump: bool,
    pitch: Option<f64>,
    steer: f64,
    roll: f64,
    boost: bool,
    end: bool,
}

#[derive(Clone, Debug)]
struct ActionSeq {
    t: f64,
    aim_ball: bool,
    boost_flip: bool,
    speedflip: bool,
    steps: Vec<SeqStep>,
}

#[derive(Clone, Copy, Debug)]
struct AerialState {
    p: DVec3,
    t_hit: f64,
    t: f64,
    dbl: bool,
    dodged: bool,
    aim_dir: DVec3,
}

#[derive(Clone, Debug)]
pub struct BotAi {
    pub diff_index: usize,
    pub d: Difficulty,
    pub p: Personality,
    pub mode: BotMode,
    pub decide_t: f64,
    pub decided_at: f64,
    pub now: f64,
    pub target: DVec3,
    seq: Option<ActionSeq>,
    aer: Option<AerialState>,
    intercept: Option<Intercept>,
    pub intercept_pos: DVec3,
    pub intercept_t: f64,
    pub shot_dir: DVec3,
    pub aim_err: f64,
    pub last_jump_t: f64,
    pub kick_done: bool,
    pub kick_boost_flip: bool,
    pub pad_target: Option<usize>,
    pub stuck_t: f64,
    pub unstuck: f64,
    pub unstuck_steer: f64,
    pub item_hold: f64,
    pub strike: Option<StrikePlan>,
    pub strike_next: f64,
    /// When true, SAVANT uses the handcrafted MPC planner even if a neural policy is embedded.
    pub force_mpc: bool,
    pub policy_ctl: Controls,
    pub policy_rem: u8,
}

impl BotAi {
    pub fn new(diff: usize, pers: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::new_with_rng(diff, pers, &mut rng)
    }

    pub fn new_with_rng<R: Rng + ?Sized>(diff: usize, pers: usize, rng: &mut R) -> Self {
        let di = diff.min(DIFFICULTY.len() - 1);
        let pi = pers % PERSONALITIES.len();
        Self {
            diff_index: di,
            d: DIFFICULTY[di],
            p: PERSONALITIES[pi],
            mode: BotMode::Idle,
            decide_t: rng.r#gen::<f64>() * 0.2,
            decided_at: 0.0,
            now: 0.0,
            target: DVec3::ZERO,
            seq: None,
            aer: None,
            intercept: None,
            intercept_pos: DVec3::ZERO,
            intercept_t: 0.0,
            shot_dir: DVec3::Z,
            aim_err: 0.0,
            last_jump_t: -10.0,
            kick_done: false,
            kick_boost_flip: false,
            pad_target: None,
            stuck_t: 0.0,
            unstuck: 0.0,
            unstuck_steer: 1.0,
            item_hold: 0.0,
            strike: None,
            strike_next: 0.0,
            force_mpc: false,
            policy_ctl: Controls::default(),
            policy_rem: 0,
        }
    }

    #[inline]
    pub fn is_champ(&self) -> bool {
        self.diff_index >= 4
    }

    #[inline]
    pub fn is_savant(&self) -> bool {
        self.diff_index >= 5
    }

    #[inline]
    fn attack_dir(car: &Car) -> f64 {
        if car.team == 0 { 1.0 } else { -1.0 }
    }

    pub fn reset(&mut self) {
        self.seq = None;
        self.aer = None;
        self.mode = BotMode::Idle;
        self.kick_done = false;
        self.kick_boost_flip = false;
        self.pad_target = None;
        self.stuck_t = 0.0;
        self.unstuck = 0.0;
        self.strike = None;
        self.strike_next = 0.0;
        self.policy_ctl = Controls::default();
        self.policy_rem = 0;
    }

    pub fn estimate(&self, car: &Car, target: DVec3, use_boost: bool) -> f64 {
        let to = target - car.pos;
        let dist = (to.length() - 1.6).max(0.0);
        let b = car.basis;
        let ang = b.f.three_angle_to(to.three_normalize());
        let vf = car.vel.dot(b.f).max(0.0);
        let turn = if self.is_champ() {
            ang * (0.22 + vf * 0.009)
        } else {
            ang * (0.3 + vf * 0.012)
        };
        turn + time_to_cover(
            dist,
            vf * if ang > 1.2 { 0.3 } else { 1.0 },
            use_boost && car.boost > 8.0 && self.d.boost > 0.5,
        ) / self.d.speed
            + if car.on_ground { 0.0 } else { 0.4 }
    }

    fn find_intercept(&self, car: &Car, w: &World) -> Intercept {
        let pred = &w.pred;
        let shape = &w.shape;
        let champ = self.is_champ();
        let savant = self.is_savant();
        let max_jump = if savant {
            3.7
        } else if self.d.jump_shot > 0.5 {
            3.4
        } else {
            2.0
        };
        let step = if champ { 1 } else { 3 };
        let min_aer_boost = if savant {
            14.0
        } else if champ {
            18.0
        } else {
            25.0
        };
        let max_aer_t = if savant {
            4.2
        } else if champ {
            3.8
        } else {
            3.5
        };
        let min_aer_t = if savant {
            0.26
        } else if champ {
            0.35
        } else {
            0.6
        };
        let wall_max_y = if savant { 16.0 } else { 14.0 };
        let mut i = if champ { 1 } else { 2 };
        while i < pred.valid {
            let s = pred.slices[i];
            let near_surface = shape.dist(s.pos.x, s.pos.y, s.pos.z) < w.phys.ball_r + 1.4;
            if s.pos.y > max_jump && !(near_surface && s.pos.y < wall_max_y) {
                if self.d.aerial > 0.0
                    && s.pos.y < self.d.aerial_h
                    && car.boost > min_aer_boost
                    && s.t < max_aer_t
                    && s.t > min_aer_t
                    && self.aerial_feasible(car, s.pos, s.t, &w.phys)
                {
                    return Intercept {
                        slice: s,
                        t: s.t,
                        aerial: true,
                        late: false,
                    };
                }
                i += step;
                continue;
            }
            let est = self.estimate(car, s.pos, true);
            if est <= s.t {
                return Intercept {
                    slice: s,
                    t: s.t,
                    aerial: false,
                    late: false,
                };
            }
            i += step;
        }
        let last = pred.slices[pred.valid.saturating_sub(1)];
        Intercept {
            slice: last,
            t: last.t,
            aerial: false,
            late: true,
        }
    }

    fn aerial_feasible(&self, car: &Car, p: DVec3, t: f64, phys: &PhysConfig) -> bool {
        let mut d = p - car.pos - car.vel * t;
        d.y += 0.5 * phys.g * t * t;
        if self.is_savant() {
            // Fast aerial (jump + hold + double jump) provides strong initial upward velocity
            let jump_lift = if p.y > 5.5 { 7.4 } else { 4.8 };
            d.y -= jump_lift * t * 0.74;
        } else {
            d.y -= 4.2 * t * 0.8;
        }
        let a_req = 2.0 * d.length() / (t * t);
        let fuel = a_req / phys.boost_acc * t * phys.boost_use;
        if self.is_savant() {
            a_req < phys.boost_acc * 0.96 && fuel < car.boost - 1.0
        } else if self.is_champ() {
            a_req < phys.boost_acc * 0.92 && fuel < car.boost - 2.0
        } else {
            a_req < phys.boost_acc * 0.85 && fuel < car.boost - 5.0
        }
    }

    pub fn update(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        self.update_core(car_idx, dt, w);
        if w.powers.is_some() && !w.cars[car_idx].demolished && !w.kickoff {
            self.use_items(car_idx, dt, w);
        }
    }

    fn use_items(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let ad = Self::attack_dir(&w.cars[car_idx]);
        let savant = self.is_savant();
        if w.ball.stuck_to == Some(car_idx) {
            if w.cars[car_idx].on_ground && self.seq.is_none() && self.aer.is_none() {
                let s = &w.shape;
                let car_pos = w.cars[car_idx].pos;
                let goal = if s.goal_mode == GoalMode::Standalone && (car_pos.z * ad) > s.goal_z - 0.5 {
                    let sx = if car_pos.x >= 0.0 { 1.0 } else { -1.0 };
                    DVec3::new(sx * (s.gw + s.gt + 5.0), 0.0, ad * (s.goal_z - 6.0))
                } else if let Some(sg) = s.split_goal {
                    let mut best_sx = if car_pos.x >= 0.0 { 1.0 } else { -1.0 };
                    if savant {
                        let score_side = |sx: f64| -> f64 {
                            let gx = sx * sg.gx;
                            let gz = ad * sg.gz;
                            let d = (car_pos.x - gx).hypot(car_pos.z - gz);
                            let def_pen = w.cars.iter().any(|o| {
                                o.team != w.cars[car_idx].team
                                    && !o.demolished
                                    && (o.pos.x - gx).hypot(o.pos.z - gz) < 13.0
                            });
                            d + if def_pen { 22.0 } else { 0.0 }
                        };
                        best_sx = if score_side(-1.0) < score_side(1.0) { -1.0 } else { 1.0 };
                    }
                    DVec3::new(
                        best_sx * (sg.gx + sg.gnx * 2.5),
                        0.0,
                        ad * (sg.gz + sg.gnz * 2.5),
                    )
                } else {
                    let gx = clamp(car_pos.x * 0.2, -s.gw + 2.0, s.gw - 2.0);
                    DVec3::new(gx, 0.0, ad * (s.goal_line_z() + 3.0))
                };
                self.drive_to(car_idx, goal, 23.0, w, false);
                let s = &w.shape;
                let car = &w.cars[car_idx];
                let to = goal - car.pos;
                let d = to.length();
                let facing = to.dot(car.basis.f) / d.max(0.01);
                let unobstructed = s.goal_mode != GoalMode::Standalone
                    || !Self::standalone_seg_hits_box(s, car.pos, goal, ad, 0.8);
                let mut near_opp = 1e9_f64;
                for o in &w.cars {
                    if o.team != car.team && !o.demolished {
                        near_opp = near_opp.min(o.pos.distance(car.pos));
                    }
                }
                if unobstructed && facing > 0.9 {
                    if savant {
                        // Launch into open net at close range or right before an enemy tackle
                        if d < 13.0 || (near_opp < 6.5 && d < 26.0) {
                            w.cars[car_idx].controls.item = true;
                        }
                    } else if d < 16.0 + self.d.aim * 10.0 && w.rng.r#gen::<f64>() < dt * 3.0 {
                        w.cars[car_idx].controls.item = true;
                    }
                }
            }
            return;
        }

        let Some(id) = w.cars[car_idx].item else {
            self.item_hold = 0.0;
            return;
        };
        self.item_hold += dt;
        let min_hold = if savant {
            0.08
        } else {
            0.5 + self.d.react * 4.0
        };
        if self.item_hold < min_hold {
            return;
        }
        let tgt = match w.powers.as_ref() {
            Some(p) => p.target_for(car_idx, id, &w.cars, &w.ball),
            None => None,
        };
        let Some(tgt) = tgt else {
            if self.item_hold > 20.0 {
                self.item_hold = 0.0;
            }
            return;
        };

        let attack_tgt = self.shot_target(car_idx, w, w.ball.pos);
        let car = &w.cars[car_idx];
        let ball = &w.ball;
        let s = &w.shape;
        let tb = ball.pos - car.pos;
        let db = tb.length();
        let fdot = tb.dot(car.basis.f) / db.max(0.01);
        let glz = s.goal_line_z();
        let ball_to_goal_x = attack_tgt.x - ball.pos.x;
        let ball_to_goal_z = attack_tgt.z - ball.pos.z;
        let bgl = ball_to_goal_x.hypot(ball_to_goal_z);
        let bgl = if bgl == 0.0 { 1.0 } else { bgl };
        let hl = tb.x.hypot(tb.z);
        let hl = if hl == 0.0 { 1.0 } else { hl };
        let align = (tb.x * ball_to_goal_x + tb.z * ball_to_goal_z) / (hl * bgl);
        let threat = w.pred.goal_team == car.team as i32 && w.pred.goal_time < 3.5;
        let scoring_ours = w.pred.goal_team == (1 - car.team) as i32 && w.pred.goal_time < 3.5;
        let own_goal_dist = if let Some(sg) = s.split_goal {
            let oz = -ad * sg.gz;
            (ball.pos.x + sg.gx)
                .hypot(ball.pos.z - oz)
                .min((ball.pos.x - sg.gx).hypot(ball.pos.z - oz))
        } else {
            (ball.pos.z + ad * glz).abs()
        };
        let enemy_has_spikes = ball.stuck_to.is_some_and(|ci| w.cars[ci].team != car.team);
        let punch_blocked = s.goal_mode == GoalMode::Standalone
            && (ball.pos.z * ad) > s.goal_z - 0.5
            && ball.pos.x.abs() < s.gw + s.gt + 2.0;

        let mut near_opp = 1e9_f64;
        for o in &w.cars {
            if o.team != car.team && !o.demolished {
                near_opp = near_opp.min(o.pos.distance(car.pos));
            }
        }

        let tgt_pos = match tgt {
            PowerTarget::Ball => Some(ball.pos),
            PowerTarget::Enemy(idx) => Some(w.cars[idx].pos),
            PowerTarget::SelfOnly => None,
        };

        let mut go = match id {
            "tornado" => db < 16.0 || near_opp < 11.0,
            "freeze" => {
                !scoring_ours
                    && ((threat && own_goal_dist < 35.0)
                        || enemy_has_spikes
                        || (savant
                            && (ball.vel.z * ad) < -18.0
                            && (ball.pos.z * ad) < 5.0
                            && db > 10.0))
            }
            "spikes" => db < if savant { 11.0 } else { 13.0 } && fdot > 0.6,
            "magnet" => db < 20.0 && align > 0.2,
            "grapple" => {
                let unsafe_own_pull = (ball.pos.z * ad) < -glz * 0.55
                    && (car.pos.z - ball.pos.z) * ad < -2.0;
                !unsafe_own_pull
                    && db > if savant { 13.0 } else { 15.0 }
                    && fdot > 0.7
                    && matches!(
                        self.mode,
                        BotMode::Attack | BotMode::Save | BotMode::AerialPrep
                    )
            }
            "tether" => {
                (threat && db > 7.5)
                    || (db > 10.0
                        && (car.pos.z - ball.pos.z) * ad > 5.5
                        && (!savant || (ball.pos.z * ad) > glz * 0.15))
            }
            "punch" => {
                !scoring_ours
                    && ((!punch_blocked && align > if savant { 0.78 } else { 0.72 })
                        || enemy_has_spikes
                        || (threat && (ball.pos.z - car.pos.z) * ad > 0.0))
            }
            "boot" | "scramble" => {
                let enemy_is_goalie = tgt_pos.is_some_and(|p| {
                    (ball.pos.z * ad) > 0.0
                        && if let Some(sg) = s.split_goal {
                            (p.x.abs() - sg.gx).abs() < s.gw + 6.0 && (p.z - ad * sg.gz).abs() < 16.0
                        } else {
                            p.x.abs() < s.gw + 6.0 && (p.z - ad * glz).abs() < 16.0
                        }
                });
                enemy_has_spikes
                    || enemy_is_goalie
                    || tgt_pos
                        .map(|p| p.distance(ball.pos) < 12.0 || p.distance(car.pos) < 10.0)
                        .unwrap_or(false)
            }
            "swap" => {
                !threat
                    && (enemy_has_spikes
                        || tgt_pos
                            .map(|p| p.distance(ball.pos) < db - 14.0)
                            .unwrap_or(false))
            }
            "shock" => near_opp < 9.0 || (db < 9.0 && align > 0.5),
            "overdrive" => db < 22.0 || near_opp < 14.0,
            "antigrav" => {
                !scoring_ours
                    && ((threat && ball.pos.y > 2.5)
                        || (savant && (ball.pos.z * ad) < -6.0 && ball.pos.y > 3.0)
                        || self.item_hold > 9.0)
            }
            "nitro" => db > 18.0 || self.mode == BotMode::Save,
            _ => false,
        };
        if !scoring_ours && self.item_hold > 13.0 + self.d.err * 6.0 {
            go = true;
        }
        if go {
            w.cars[car_idx].controls.item = true;
        }
    }

    fn update_core(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        self.now = w.time;
        {
            let c = &mut w.cars[car_idx].controls;
            c.throttle = 0.0;
            c.steer = 0.0;
            c.pitch = 0.0;
            c.yaw = 0.0;
            c.roll = 0.0;
            c.jump = false;
            c.boost = false;
            c.handbrake = false;
            c.air_roll_free = false;
            c.item = false;
        }
        if w.cars[car_idx].demolished {
            return;
        }
        self.decide_t -= dt;
        if self.decide_t <= 0.0 {
            self.decide_t = self.d.react * (0.7 + w.rng.r#gen::<f64>() * 0.6);
            self.decide(car_idx, w);
        }
        // Hybrid Neural-MPC SAVANT:
        // In 1v1/2v2 close-quarters scrambles, dribbles, and 50/50 challenges where the physics
        // planner has no high-value direct shot/clear (`score < 35`), hand off to our 54M-step
        // trained RL MLP (`savant_policy.bin`) for relentless pressure & emergent ball control!
        if self.is_savant()
            && !self.force_mpc
            && !w.kickoff
            && w.cars.len() <= 4
            && self.mode == BotMode::Attack
            && self.aer.is_none()
            && self.seq.is_none()
            && policy_supports_world(w)
            && let Some(pol) = embedded_savant_policy()
        {
            let car = &w.cars[car_idx];
            let ad = Self::attack_dir(car);
            let glz = w.shape.goal_line_z();
            let safe_from_own_net = (w.ball.pos.z * ad) > -glz * 0.35;
            let close_scramble = car.pos.distance(w.ball.pos) < 11.0;
            if safe_from_own_net && close_scramble {
                if self.d.plan > 0.0 && w.time >= self.strike_next {
                    self.strike_next = w.time + self.d.plan;
                    self.plan_strike(car_idx, w, false);
                }
                if self.strike.is_none_or(|pl| pl.score < 35.0) {
                    if self.policy_rem == 0 {
                        let mut obs = [0.0_f32; OBS_DIM];
                        extract_obs(w, car_idx, &self.policy_ctl, &mut obs);
                        let act = pol.act_greedy(&obs);
                        self.policy_ctl = decode_action(&act);
                        self.policy_rem = POLICY_TICK_SKIP as u8;
                    }
                    self.policy_rem = self.policy_rem.saturating_sub(1);
                    w.cars[car_idx].controls = self.policy_ctl;
                    return;
                }
            } else {
                self.policy_rem = 0;
            }
        }
        if self.decide_t <= 0.0 {
            self.decide_t = self.d.react * (0.7 + w.rng.r#gen::<f64>() * 0.6);
            self.decide(car_idx, w);
        }
        if self.d.fix >= 0.5 {
            let car = &mut w.cars[car_idx];
            if car.contacts == 0
                && car.body_contact
                && car.basis.u.y < 0.5
                && car.pos.y < 1.4
                && car.vel.y < 2.0
                && self.aer.is_none()
            {
                self.seq = None;
                car.controls.jump = !car.prev_jump;
                car.controls.throttle = 1.0;
                return;
            }
        }
        if let Some(mut pl) = self.strike
            && let Some(t0) = pl.seq_t0
        {
            if w.time - t0 < 0.7 {
                let ctl = strike_controls(&mut pl, &w.cars[car_idx], w.ball.pos, w.time, &w.phys);
                self.strike = Some(pl);
                w.cars[car_idx].controls = ctl;
                return;
            }
            self.strike = None;
        }
        if self.aer.is_some() {
            self.fly_aerial(car_idx, dt, w);
            return;
        }
        if self.seq.is_some() {
            self.run_seq(car_idx, dt, w);
            return;
        }
        if !w.cars[car_idx].on_ground && w.cars[car_idx].contacts == 0 {
            self.recover(car_idx, dt, w);
            return;
        }
        if w.cars[car_idx].body_contact && w.cars[car_idx].contacts < 2 {
            let j = w.rng.r#gen::<f64>() < 0.35;
            w.cars[car_idx].controls.jump = j;
            w.cars[car_idx].controls.throttle = 1.0;
            return;
        }
        if self.unstuck > 0.0 {
            self.unstuck -= dt;
            w.cars[car_idx].controls.throttle = -1.0;
            w.cars[car_idx].controls.steer = self.unstuck_steer;
            return;
        }
        self.act(car_idx, dt, w);
        let car = &w.cars[car_idx];
        if car.controls.throttle.abs() > 0.5 && car.vel.length() < 1.2 && car.on_ground {
            self.stuck_t += dt;
            if self.stuck_t > 0.6 {
                self.stuck_t = 0.0;
                self.unstuck = 0.85;
                let s = math_sign(car.controls.steer);
                self.unstuck_steer = -(if s != 0.0 { s } else { 1.0 });
            }
        } else {
            self.stuck_t = 0.0;
        }
    }

    fn decide(&mut self, car_idx: usize, w: &mut World) {
        if w.kickoff {
            self.mode = BotMode::Kickoff;
            return;
        }
        self.decided_at = w.time;
        let it = self.find_intercept(&w.cars[car_idx], w);
        self.intercept = Some(it);
        self.intercept_pos = it.slice.pos;
        self.intercept_t = it.t;
        self.aim_err = (w.rng.r#gen::<f64>() - 0.5) * 2.0 * self.d.aim;

        let car = &w.cars[car_idx];
        let role = w.brain.role_of(car_idx);
        let threat = w.pred.goal_team == car.team as i32 && w.pred.goal_time < 5.0;
        let ad = Self::attack_dir(car);
        let own_z = -ad * w.shape.goal_line_z();
        if threat
            && (role == Role::Defend || role == Role::Attack || self.closest_to_goal(car_idx, w))
        {
            self.mode = BotMode::Save;
        } else if role == Role::Attack {
            self.mode = if it.aerial {
                BotMode::AerialPrep
            } else {
                BotMode::Attack
            };
        } else if role == Role::Support {
            self.mode = BotMode::Support;
        } else {
            self.mode = BotMode::Defend;
        }

        if self.mode == BotMode::Attack {
            let champ = self.is_champ();
            let savant = self.is_savant();
            let ahead = (w.ball.pos.z - car.pos.z) * ad < if champ { -2.5 } else { -4.0 };
            let my_d = car.pos.distance(w.ball.pos);
            let margin = if champ { 4.0 } else { 6.0 };
            let opp_close = w.cars.iter().any(|o| {
                o.team != car.team && !o.demolished && o.pos.distance(w.ball.pos) < my_d - margin
            });
            if ahead
                && opp_close
                && self.d.rot > 0.6
                && (w.ball.pos.z - own_z).abs() < w.shape.l * 1.2
            {
                self.mode = BotMode::Shadow;
            } else if savant {
                // Predictive anti-overcommit: if an opponent clearly beats us to the ball and is
                // positioned to shoot toward our net, shadow tightly instead of diving out of position
                let mut best_opp_eta = 1e9_f64;
                let mut opp_shooting_line = false;
                for o in &w.cars {
                    if o.team != car.team && !o.demolished {
                        let o_eta = self.estimate(o, it.slice.pos, true);
                        if o_eta < best_opp_eta {
                            best_opp_eta = o_eta;
                            opp_shooting_line = (o.pos.z - it.slice.pos.z) * ad > -2.0;
                        }
                    }
                }
                let in_own_half = (w.ball.pos.z * ad) < w.shape.l * 0.25;
                if in_own_half
                    && opp_shooting_line
                    && best_opp_eta + 0.28 < it.t
                    && my_d > 11.0
                    && (car.pos.z - own_z).abs() > 14.0
                {
                    self.mode = BotMode::Shadow;
                }
            }
        }
        let cb = self.is_champ();
        let savant = self.is_savant();
        let sup_boost_thresh = if cb { 42.0 } else { 30.0 };
        let def_boost_thresh = if savant {
            if (w.ball.pos.z * ad) < 8.0 { 0.0 } else { 24.0 }
        } else if cb {
            22.0
        } else {
            12.0
        };
        let sup_pad_d = if cb { 28.0 } else { 40.0 };
        if self.d.boost > 0.3
            && ((self.mode == BotMode::Support && car.boost < sup_boost_thresh)
                || (self.mode == BotMode::Defend && car.boost < def_boost_thresh))
            && let Some(pad) = self.best_pad(
                car_idx,
                w,
                if self.mode == BotMode::Defend {
                    18.0
                } else {
                    sup_pad_d
                },
                None,
            )
        {
            self.pad_target = Some(pad);
            self.mode = BotMode::Boost;
        }
        if self.mode == BotMode::AerialPrep
            && !self.is_champ()
            && self.d.aerial * (0.6 + self.p.air * 0.6) < w.rng.r#gen::<f64>()
        {
            self.mode = BotMode::Attack;
        }
    }

    /// True if the 2D XZ segment `a -> b` intersects the solid standalone goal box at `end_sign` (`-1` or `+1`).
    fn standalone_seg_hits_box(
        s: &crate::arena::ArenaShape,
        a: DVec3,
        b: DVec3,
        end_sign: f64,
        pad: f64,
    ) -> bool {
        let min_x = -(s.gw + s.gt + pad);
        let max_x = s.gw + s.gt + pad;
        let min_z = s.goal_z - pad * 0.35;
        let max_z = s.goal_z + s.gd + s.gt + pad;
        let ax = a.x;
        let az = a.z * end_sign;
        let bx = b.x;
        let bz = b.z * end_sign;
        let dx = bx - ax;
        let dz = bz - az;
        let mut t0 = 0.0_f64;
        let mut t1 = 1.0_f64;
        for &(p0, d, lo, hi) in &[(ax, dx, min_x, max_x), (az, dz, min_z, max_z)] {
            if d.abs() < 1e-6 {
                if p0 < lo || p0 > hi {
                    return false;
                }
            } else {
                let mut u0 = (lo - p0) / d;
                let mut u1 = (hi - p0) / d;
                if u0 > u1 {
                    std::mem::swap(&mut u0, &mut u1);
                }
                t0 = t0.max(u0);
                t1 = t1.min(u1);
                if t0 > t1 {
                    return false;
                }
            }
        }
        true
    }

    fn own_goal_guard_pt(
        shape: &crate::arena::ArenaShape,
        ball_pos: DVec3,
        ad: f64,
        depth: f64,
        back_post: bool,
    ) -> DVec3 {
        if let Some(sg) = shape.split_goal {
            let sx = if ball_pos.x >= 0.0 { 1.0 } else { -1.0 };
            let cx = sx * (sg.gx - sg.gnx * depth);
            let cz = -ad * (sg.gz - sg.gnz * depth);
            let bp = if back_post { -sx * 1.8 } else { 0.0 };
            return DVec3::new(cx + bp, 0.0, cz);
        }
        let gz = -ad * shape.goal_line_z();
        let x = if back_post {
            if ball_pos.x >= 0.0 {
                -(shape.gw - 2.0)
            } else {
                shape.gw - 2.0
            }
        } else {
            clamp(ball_pos.x * 0.3, -shape.gw + 1.5, shape.gw - 1.5)
        };
        DVec3::new(x, 0.0, gz + ad * depth)
    }

    fn closest_to_goal(&self, car_idx: usize, w: &World) -> bool {
        let car = &w.cars[car_idx];
        let ad = Self::attack_dir(car);
        let gp = Self::own_goal_guard_pt(&w.shape, w.ball.pos, ad, 0.0, false);
        let mut best = None;
        let mut bd = 1e9_f64;
        for (i, o) in w.cars.iter().enumerate() {
            if o.team == car.team && !o.demolished {
                let d = (o.pos.x - gp.x).hypot(o.pos.z - gp.z);
                if d < bd {
                    bd = d;
                    best = Some(i);
                }
            }
        }
        best == Some(car_idx)
    }

    fn best_pad(
        &self,
        car_idx: usize,
        w: &World,
        max_d: f64,
        filter: Option<&dyn Fn(DVec3) -> bool>,
    ) -> Option<usize> {
        let car = &w.cars[car_idx];
        let ad = Self::attack_dir(car);
        let mut best = None;
        let mut bs = 1e9_f64;
        for (i, p) in w.pads.iter().enumerate() {
            if !p.active {
                continue;
            }
            if let Some(f) = filter
                && !f(p.pos)
            {
                continue;
            }
            let d = car.pos.distance(p.pos);
            let behind = (p.pos.z - w.ball.pos.z) * ad < 5.0;
            let in_goal = p.pos.z.abs() > w.shape.l - 2.0;
            let sc = d * if p.big { 0.55 } else { 1.2 } + if behind { 0.0 } else { 25.0 };
            if sc < bs && d < max_d && !in_goal {
                bs = sc;
                best = Some(i);
            }
        }
        best
    }

    fn act(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let ad = Self::attack_dir(&w.cars[car_idx]);
        match self.mode {
            BotMode::Kickoff => self.kickoff(car_idx, dt, w),
            BotMode::Attack | BotMode::AerialPrep => self.attack(car_idx, dt, w),
            BotMode::Save => self.save(car_idx, dt, w),
            BotMode::Shadow => {
                let glz = w.shape.goal_line_z();
                let gz = -ad * glz;
                let t = if let Some(sg) = w.shape.split_goal {
                    let sx = if w.ball.pos.x >= 0.0 { 1.0 } else { -1.0 };
                    let gx = sx * (sg.gx - sg.gnx * 2.5);
                    let ogz = -ad * (sg.gz - sg.gnz * 2.5);
                    w.shape.push_inside(
                        DVec3::new(
                            lerp(w.ball.pos.x, gx, if self.is_savant() { 0.44 } else { 0.52 }),
                            0.0,
                            lerp(w.ball.pos.z, ogz, if self.is_savant() { 0.48 } else { 0.55 }),
                        ),
                        4.0,
                    )
                } else if w.shape.goal_mode == GoalMode::Standalone
                    && (w.ball.pos.z * ad) < -glz + 0.5
                {
                    // Ball is behind our own net: guard the near post out front to intercept the wrap-around
                    let sx = if w.ball.pos.x >= 0.0 { 1.0 } else { -1.0 };
                    DVec3::new(sx * (w.shape.gw + 2.5), 0.0, gz + ad * 4.5)
                } else if self.is_savant() {
                    let bp_x = if w.ball.pos.x >= 0.0 {
                        -(w.shape.gw - 2.4)
                    } else {
                        w.shape.gw - 2.4
                    };
                    w.shape.push_inside(
                        DVec3::new(
                            lerp(w.ball.pos.x, bp_x, 0.36),
                            0.0,
                            lerp(w.ball.pos.z, gz + ad * 2.0, 0.48),
                        ),
                        4.0,
                    )
                } else {
                    w.shape.push_inside(
                        DVec3::new(w.ball.pos.x * 0.6, 0.0, lerp(w.ball.pos.z, gz, 0.55)),
                        5.0,
                    )
                };
                let spd = clamp(w.cars[car_idx].pos.distance(t) * 1.4, 5.0, 23.0);
                self.drive_to(car_idx, t, spd, w, !self.is_savant());
            }
            BotMode::Support => {
                let sl = w.brain.slot_of(car_idx);
                let n = sl.map(|s| s.n).unwrap_or(1);
                let glz = w.shape.goal_line_z();
                let champ = self.is_champ();
                let savant = self.is_savant();
                // SAVANT 2v2/3v3: opportunistic goalie demo when a teammate is attacking deep
                if savant
                    && w.allow_demos
                    && (w.ball.pos.z * ad) > glz * 0.32
                    && (w.cars[car_idx].boost > 28.0 || w.cars[car_idx].speed() > 16.5)
                {
                    let my_pos = w.cars[car_idx].pos;
                    let my_team = w.cars[car_idx].team;
                    let mut goalie: Option<DVec3> = None;
                    let mut best_d = 26.0;
                    for o in &w.cars {
                        let in_opp_crease = if let Some(sg) = w.shape.split_goal {
                            (o.pos.z - ad * sg.gz).abs() < 15.0
                                && (o.pos.x.abs() - sg.gx).abs() < w.shape.gw + 5.0
                        } else {
                            (o.pos.z - ad * glz).abs() < 15.0
                                && o.pos.x.abs() < w.shape.gw + 4.5
                        };
                        if o.team != my_team && !o.demolished && in_opp_crease {
                            let d = my_pos.distance(o.pos);
                            if d < best_d {
                                best_d = d;
                                goalie = Some(o.pos + o.vel * (d / 24.0).min(0.35));
                            }
                        }
                    }
                    if let Some(gp) = goalie {
                        let gp_in = w.shape.push_inside(DVec3::new(gp.x, 0.0, gp.z), 2.0);
                        self.drive_to(car_idx, gp_in, 23.0, w, false);
                        return;
                    }
                }
                let (x, z) = if n <= 1 {
                    let back = if champ {
                        11.0 + self.p.defend * 5.0
                    } else {
                        14.0 + self.p.defend * 8.0
                    };
                    let z = clamp(w.ball.pos.z - ad * back, -glz + 6.0, glz - 6.0);
                    let x = w.ball.pos.x * 0.45
                        + if w.cars[car_idx].pos.x > 0.0 {
                            6.0
                        } else {
                            -6.0
                        };
                    (x, z)
                } else {
                    let sl = sl.unwrap();
                    let spread = (sl.i as f64 / (n - 1) as f64 - 0.5) * 2.0;
                    let mid = spread.abs() < 0.34;
                    let z = clamp(
                        w.ball.pos.z
                            - ad * if mid {
                                9.0 + self.p.defend * 5.0
                            } else {
                                15.0 + self.p.defend * 8.0 + spread.abs() * 4.0
                            },
                        -glz + 6.0,
                        glz - 6.0,
                    );
                    let x = w.ball.pos.x * if mid { 0.6 } else { 0.3 }
                        + spread * w.shape.half_width_at(z) * 0.58;
                    (x, z)
                };
                let t = w.shape.push_inside(DVec3::new(x, 0.0, z), 5.0);
                let d = w.cars[car_idx].pos.distance(t);
                if d < 4.0 {
                    self.face_ball(car_idx, w);
                    return;
                }
                self.drive_to(car_idx, t, clamp(d * 0.9, 3.0, 23.0), w, true);
            }
            BotMode::Boost => {
                let pad_pos = self
                    .pad_target
                    .and_then(|pi| w.pads.get(pi))
                    .filter(|p| p.active)
                    .map(|p| p.pos);
                if pad_pos.is_none() || w.cars[car_idx].boost > 90.0 {
                    self.mode = BotMode::Support;
                    self.decide_t = 0.0;
                    return;
                }
                self.drive_to(car_idx, pad_pos.unwrap(), 23.0, w, true);
            }
            BotMode::Idle | BotMode::Defend => {
                let depth = if self.is_savant() {
                    if (w.ball.pos.z * ad) < -w.shape.goal_line_z() * 0.25 {
                        1.8
                    } else {
                        3.2
                    }
                } else if self.is_champ() {
                    2.6
                } else {
                    3.5
                };
                let t = Self::own_goal_guard_pt(
                    &w.shape,
                    w.ball.pos,
                    ad,
                    depth,
                    self.is_champ(),
                );
                let d = w.cars[car_idx].pos.distance(t);
                if d < if self.is_savant() { 2.2 } else { 3.0 } {
                    self.face_ball(car_idx, w);
                    return;
                }
                self.drive_to(car_idx, t, clamp(d * 1.4, 3.0, 23.0), w, !self.is_savant());
            }
        }
    }

    fn face_ball(&self, car_idx: usize, w: &mut World) {
        let ball_pos = w.ball.pos;
        let car = &mut w.cars[car_idx];
        let to = ball_pos - car.pos;
        let al = car.basis.to_local(to);
        let ang = al.z.atan2(al.x);
        let vf = car.vel.dot(car.basis.f);
        if ang.abs() > 0.5 {
            car.controls.throttle = if vf > 2.0 { -0.5 } else { 0.6 };
            car.controls.steer = clamp(ang * 2.0, -1.0, 1.0)
                * if car.controls.throttle < 0.0 {
                    -1.0
                } else {
                    1.0
                };
        } else {
            car.controls.throttle = clamp(-vf * 0.4, -1.0, 1.0);
        }
    }

    fn shot_target(&self, car_idx: usize, w: &World, pos: DVec3) -> DVec3 {
        let shape = &w.shape;
        let car = &w.cars[car_idx];
        let ad = Self::attack_dir(car);
        let glz = shape.goal_line_z();
        let opp_z = ad * (glz + 2.0);
        let own_half = (pos.z * ad) < -glz * 0.35;
        if own_half {
            if shape.goal_mode == GoalMode::Standalone && (pos.z * ad) < -(shape.goal_z + 0.5) {
                let sx = if pos.x >= 0.0 { 1.0 } else { -1.0 };
                return DVec3::new(sx * shape.w * 0.72, 0.0, -ad * (shape.goal_z - 12.0));
            }
            if self.is_champ()
                && (pos.z * ad) > -glz * 0.65
                && shape.goal_mode == GoalMode::Standard
            {
                let opp_in_net = w.cars.iter().any(|o| {
                    o.team != car.team
                        && !o.demolished
                        && (o.pos.z - ad * glz).abs() < 30.0
                        && o.pos.x.abs() < shape.gw + 8.0
                });
                if !opp_in_net {
                    return DVec3::new(0.0, 0.0, opp_z);
                }
            }
            return DVec3::new(
                if pos.x >= 0.0 {
                    shape.w * 0.9
                } else {
                    -shape.w * 0.9
                },
                0.0,
                pos.z + ad * 35.0,
            );
        }
        if shape.goal_mode == GoalMode::Standalone && (pos.z * ad) > shape.goal_z - 0.5 {
            let sx = if pos.x >= 0.0 { 1.0 } else { -1.0 };
            if pos.x.abs() < shape.gw + 3.5 {
                // Behind the solid back wall: push laterally around the back corner first
                return DVec3::new(
                    sx * (shape.gw + shape.gt + 7.5),
                    0.0,
                    ad * (shape.goal_z + shape.gd + 3.5),
                );
            }
            // Wide of the post: wrap/center the ball out in front of the open goal mouth
            return DVec3::new(0.0, 0.0, ad * (shape.goal_z - 8.5));
        }
        if let Some(sg) = shape.split_goal {
            let mut best_sx = if pos.x >= 0.0 { 1.0 } else { -1.0 };
            if self.is_champ() {
                let side_cost = |sx: f64| -> f64 {
                    let gx = sx * sg.gx;
                    let gz = ad * sg.gz;
                    let d = (pos.x - gx).hypot(pos.z - gz);
                    let mut def_pen = 0.0_f64;
                    for o in &w.cars {
                        if o.team != car.team && !o.demolished {
                            let od = (o.pos.x - gx).hypot(o.pos.z - gz);
                            if od < 16.0 {
                                def_pen = def_pen.max((16.0 - od) * 1.6);
                            }
                        }
                    }
                    d + def_pen
                };
                best_sx = if side_cost(-1.0) < side_cost(1.0) {
                    -1.0
                } else {
                    1.0
                };
            }
            let base_x = best_sx * (sg.gx + sg.gnx * 2.5);
            let base_z = ad * (sg.gz + sg.gnz * 2.5);
            if self.is_champ() {
                // Shift along the angled goal face away from the closest defender
                let mut near_def: Option<DVec3> = None;
                let mut nd = 18.0_f64;
                for o in &w.cars {
                    if o.team != car.team && !o.demolished {
                        let d = (o.pos.x - best_sx * sg.gx).hypot(o.pos.z - ad * sg.gz);
                        if d < nd {
                            nd = d;
                            near_def = Some(o.pos);
                        }
                    }
                }
                if let Some(dp) = near_def {
                    let tx = best_sx * sg.gtx;
                    let tz = ad * sg.gtz;
                    let def_u = (dp.x - best_sx * sg.gx) * tx + (dp.z - ad * sg.gz) * tz;
                    let shift = if def_u >= 0.0 {
                        -(shape.gw - 1.8)
                    } else {
                        shape.gw - 1.8
                    };
                    return DVec3::new(base_x + tx * shift, 0.0, base_z + tz * shift);
                }
            }
            return DVec3::new(base_x, 0.0, base_z);
        }
        if self.p.pass > 0.6 && pos.x.abs() > shape.w * 0.55 && (pos.z * ad) > glz * 0.45 {
            return DVec3::new(0.0, 0.0, ad * (glz - 12.0));
        }
        if self.is_champ() {
            let mut closest_def: Option<(f64, f64)> = None;
            let goal_line_z = ad * glz;
            for o in &w.cars {
                if o.team != car.team && !o.demolished {
                    let dz = (o.pos.z - goal_line_z).abs();
                    if dz < 26.0 && o.pos.x.abs() < shape.gw + 6.0 {
                        let d = o.pos.x.hypot(dz);
                        if closest_def.is_none_or(|(bd, _)| d < bd) {
                            closest_def = Some((d, o.pos.x));
                        }
                    }
                }
            }
            let gx = if let Some((_, def_x)) = closest_def {
                if def_x >= 0.0 {
                    -(shape.gw - 1.65)
                } else {
                    shape.gw - 1.65
                }
            } else {
                clamp(
                    pos.x * 0.25 + if pos.x >= 0.0 { -2.2 } else { 2.2 },
                    -shape.gw + 1.8,
                    shape.gw - 1.8,
                )
            };
            return DVec3::new(gx, 0.0, opp_z);
        }
        let gx = clamp(pos.x * 0.35, -shape.gw + 2.2, shape.gw - 2.2);
        DVec3::new(gx, 0.0, opp_z)
    }

    /// CHAMP: plan the touch by simulating candidate strikes with the real physics, then execute
    /// the best one. Returns false when no worthwhile plan exists (caller falls back to heuristics).
    fn run_strike(&mut self, car_idx: usize, w: &mut World, save: bool) -> bool {
        let now = w.time;
        if let Some(pl) = self.strike {
            let done = now > pl.t_hit + 0.35 || pl.seq_t0.is_some_and(|t0| now - t0 > 0.7);
            if done {
                self.strike = None;
            }
        }
        let mid_seq = self.strike.is_some_and(|p| p.seq_t0.is_some());
        if !mid_seq && now >= self.strike_next {
            self.strike_next = now + self.d.plan;
            self.plan_strike(car_idx, w, save);
        }
        let Some(mut pl) = self.strike else {
            return false;
        };
        let ctl = strike_controls(&mut pl, &w.cars[car_idx], w.ball.pos, now, &w.phys);
        if pl.seq_t0.is_some() && self.strike.is_some_and(|p| p.seq_t0.is_none()) {
            self.last_jump_t = now;
        }
        self.strike = Some(pl);
        w.cars[car_idx].controls = ctl;
        true
    }

    fn plan_strike(&mut self, car_idx: usize, w: &World, save: bool) {
        let now = w.time;
        let car = &w.cars[car_idx];
        if !car.on_ground || car.contacts < 3 {
            self.strike = None;
            return;
        }
        let Some(it) = self.intercept else {
            self.strike = None;
            return;
        };
        if it.aerial {
            self.strike = None;
            return;
        }
        // On Throwback Core, don't commit to a blind straight-line strike if a goal box sits between car and ball
        if w.shape.goal_mode == GoalMode::Standalone
            && (Self::standalone_seg_hits_box(&w.shape, car.pos, it.slice.pos, 1.0, 0.6)
                || Self::standalone_seg_hits_box(&w.shape, car.pos, it.slice.pos, -1.0, 0.6))
        {
            let gz0 = w.shape.goal_z;
            let car_front_opp = car.pos.z * Self::attack_dir(car) <= gz0 + 1.0;
            let ball_in_opp = it.slice.pos.z * Self::attack_dir(car) > gz0 - 0.5
                && it.slice.pos.z * Self::attack_dir(car) < gz0 + w.shape.gd
                && it.slice.pos.x.abs() < w.shape.gw;
            if !(car_front_opp && ball_in_opp) {
                self.strike = None;
                return;
            }
        }
        let ad = Self::attack_dir(car);
        let savant = self.is_savant();
        let opps: Vec<OppSnap> = if savant {
            w.cars
                .iter()
                .filter(|o| o.team != car.team && !o.demolished)
                .map(|o| OppSnap {
                    pos: o.pos,
                    vel: o.vel,
                    boost: o.boost,
                })
                .collect()
        } else {
            Vec::new()
        };
        let ctx = StrikeCtx {
            shape: &w.shape,
            phys: &w.phys,
            ad,
            tick: w.tick as i64,
            savant,
            save,
            opps: &opps,
        };
        let mut scratch = car.clone();
        scratch.bot = None;
        let mut sball = w.ball.clone();
        let mut best: Option<StrikePlan> = None;
        let mut consider = |pl: StrikePlan, best: &mut Option<StrikePlan>| {
            let sc = simulate_strike(
                &pl,
                car,
                car_idx,
                &w.ball,
                now,
                &ctx,
                &mut scratch,
                &mut sball,
            );
            if best.is_none_or(|b| sc > b.score) {
                let mut p2 = pl;
                p2.score = sc;
                *best = Some(p2);
            }
        };
        if let Some(cur) = self.strike {
            // incumbent gets a small bonus for stability
            let mut c2 = cur;
            c2.seq_t0 = None;
            consider(c2, &mut best);
            if let Some(b) = best.as_mut() {
                b.score += if savant { 4.5 } else { 6.0 };
            }
        }
        let t_rel0 = it.t - (now - self.decided_at);
        let max_t = 2.2;
        let glz = w.shape.goal_line_z();
        if savant {
            let gw = w.shape.gw;
            let offs: &[f64] = &[-0.05, 0.0, 0.14, 0.32];
            for &off in offs {
                let tr = t_rel0 + off;
                if tr < 0.04 || tr > max_t {
                    continue;
                }
                let sl = w.pred.at(tr);
                let pp = sl.pos;
                if pp.y > 5.0 {
                    continue;
                }
                let mut sds: Vec<DVec3> = Vec::with_capacity(4);
                let mut push_sd = |d: DVec3| {
                    let n = DVec3::new(d.x, 0.0, d.z).three_normalize();
                    if n.length_squared() > 0.5 && !sds.iter().any(|e| e.dot(n) > 0.985) {
                        sds.push(n);
                    }
                };
                let tgt = self.shot_target(car_idx, w, pp);
                push_sd(tgt - pp);
                if !save {
                    let opp_z = ad * (glz + 2.0);
                    if let Some(sg) = w.shape.split_goal {
                        // Test BOTH angled split goals so SAVANT can snipe whichever net is open
                        push_sd(DVec3::new(
                            -(sg.gx + sg.gnx * 2.5) - pp.x,
                            0.0,
                            ad * (sg.gz + sg.gnz * 2.5) - pp.z,
                        ));
                        push_sd(DVec3::new(
                            (sg.gx + sg.gnx * 2.5) - pp.x,
                            0.0,
                            ad * (sg.gz + sg.gnz * 2.5) - pp.z,
                        ));
                    } else if w.shape.goal_mode == GoalMode::Standalone
                        && (pp.z * ad) > w.shape.goal_z - 0.5
                    {
                        // Ball is behind opponent's net: test left/right wrap-outs instead of back wall
                        let wrap_x = gw + w.shape.gt + 7.0;
                        let wrap_z = ad * (w.shape.goal_z - 5.0);
                        push_sd(DVec3::new(wrap_x - pp.x, 0.0, wrap_z - pp.z));
                        push_sd(DVec3::new(-wrap_x - pp.x, 0.0, wrap_z - pp.z));
                    } else {
                        push_sd(DVec3::new(-(gw - 1.35) - pp.x, 0.0, opp_z - pp.z));
                        push_sd(DVec3::new((gw - 1.35) - pp.x, 0.0, opp_z - pp.z));
                    }
                } else {
                    // Save: test left/right side clears + downfield boom + emergency block
                    push_sd(DVec3::new(w.shape.w * 0.85 - pp.x, 0.0, ad * 35.0));
                    push_sd(DVec3::new(-w.shape.w * 0.85 - pp.x, 0.0, ad * 35.0));
                    push_sd(DVec3::new(
                        if pp.x >= 0.0 { 1.0 } else { -1.0 },
                        0.0,
                        ad * 0.6,
                    ));
                }
                if pp.y < 1.9 {
                    let configs: &[(u8, f64, f64, f64)] = if save {
                        &[
                            (3, 3.2, 1.4, 0.17),
                            (1, 1.8, 1.5, 0.21),
                            (0, 2.2, 1.25, 0.2),
                        ]
                    } else {
                        &[
                            (3, 4.4, 1.25, 0.18),
                            (1, 5.6, 1.35, 0.21),
                            (0, 3.0, 1.05, 0.2),
                        ]
                    };
                    for sd in &sds {
                        for &(dg, k, pace, lead) in configs {
                            let pl = StrikePlan {
                                p: pp,
                                t_hit: now + tr,
                                sd: *sd,
                                k,
                                pace,
                                dodge: dg,
                                dodge_lead: lead,
                                dodge_bias: 0.0,
                                seq_t0: None,
                                score: 0.0,
                            };
                            consider(pl, &mut best);
                        }
                    }
                } else {
                    let kp: &[(f64, f64)] = &[(3.2, 1.1), (5.2, 1.32)];
                    for sd in &sds {
                        for &(k, pace) in kp {
                            let pl = StrikePlan {
                                p: pp,
                                t_hit: now + tr,
                                sd: *sd,
                                k,
                                pace,
                                dodge: 2,
                                dodge_lead: 0.2,
                                dodge_bias: 0.0,
                                seq_t0: None,
                                score: 0.0,
                            };
                            consider(pl, &mut best);
                        }
                    }
                }
            }
            // Stage 2: Fine refinement around the best coarse plan to nail top-corner unsaveable shots
            if let Some(seed) = best
                && seed.score < 640.0
            {
                let base_tr = (seed.t_hit - now).clamp(0.04, max_t);
                // Refine timing & angle
                for &dt_ref in &[-0.045, 0.045] {
                    let tr = base_tr + dt_ref;
                    if tr >= 0.04 && tr <= max_t {
                        let pp = w.pred.at(tr).pos;
                        if pp.y <= 5.0 {
                            let mut pl = seed;
                            pl.p = pp;
                            pl.t_hit = now + tr;
                            pl.seq_t0 = None;
                            consider(pl, &mut best);
                        }
                    }
                }
                for &ang_ref in &[-0.09_f64, 0.09_f64] {
                    let (ca, sa) = (ang_ref.cos(), ang_ref.sin());
                    let sd2 = DVec3::new(
                        seed.sd.x * ca - seed.sd.z * sa,
                        0.0,
                        seed.sd.x * sa + seed.sd.z * ca,
                    )
                    .three_normalize();
                    let mut pl = seed;
                    pl.sd = sd2;
                    pl.seq_t0 = None;
                    consider(pl, &mut best);
                }
                if seed.dodge != 2 {
                    for &bias in &[-0.45_f64, 0.45_f64] {
                        let mut pl = seed;
                        pl.dodge_bias = bias;
                        pl.seq_t0 = None;
                        consider(pl, &mut best);
                    }
                    for &(dg, lead, pace) in &[(3_u8, 0.14_f64, 1.42_f64), (1, 0.26, 1.15)] {
                        let mut pl = seed;
                        pl.dodge = dg;
                        pl.dodge_lead = lead;
                        pl.pace = pace;
                        pl.seq_t0 = None;
                        consider(pl, &mut best);
                    }
                }
            }
            let thresh = if save { -120.0 } else { -25.0 };
            self.strike = best.filter(|b| b.score > thresh);
            return;
        }
        let offs: &[f64] = &[0.0, 0.18, 0.4];
        for &off in offs {
            let tr = t_rel0 + off;
            if tr < 0.05 || tr > max_t {
                continue;
            }
            let sl = w.pred.at(tr);
            let pp = sl.pos;
            if pp.y > 4.4 {
                continue;
            }
            let tgt = self.shot_target(car_idx, w, pp);
            let (ca, sa) = (self.aim_err.cos(), self.aim_err.sin());
            let rot = |v: DVec3| DVec3::new(v.x * ca - v.z * sa, 0.0, v.x * sa + v.z * ca);
            let mut sds: Vec<DVec3> = vec![rot(
                DVec3::new(tgt.x - pp.x, 0.0, tgt.z - pp.z).three_normalize()
            )];
            if !save {
                let c = if let Some(sg) = w.shape.split_goal {
                    let other_sx = if pp.x >= 0.0 { -1.0 } else { 1.0 };
                    rot(DVec3::new(
                        other_sx * (sg.gx + sg.gnx * 2.5) - pp.x,
                        0.0,
                        ad * (sg.gz + sg.gnz * 2.5) - pp.z,
                    )
                    .three_normalize())
                } else if w.shape.goal_mode == GoalMode::Standalone
                    && (pp.z * ad) > w.shape.goal_z - 0.5
                {
                    let sx = if pp.x >= 0.0 { 1.0 } else { -1.0 };
                    rot(DVec3::new(
                        sx * (w.shape.gw + w.shape.gt + 7.0) - pp.x,
                        0.0,
                        ad * (w.shape.goal_z - 5.0) - pp.z,
                    )
                    .three_normalize())
                } else {
                    rot(DVec3::new(-pp.x, 0.0, ad * (glz + 2.0) - pp.z).three_normalize())
                };
                if c.dot(sds[0]) < 0.995 {
                    sds.push(c);
                }
            }
            let dodges: &[u8] = if pp.y < 1.9 { &[0, 1] } else { &[2] };
            for sd in &sds {
                for &dg in dodges {
                    let kp: &[(f64, f64)] = &[(4.0, 1.0), (5.5, 1.3)];
                    for &(k, pace) in kp {
                        let pl = StrikePlan {
                            p: pp,
                            t_hit: now + tr,
                            sd: *sd,
                            k,
                            pace,
                            dodge: dg,
                            dodge_lead: 0.2,
                            dodge_bias: 0.0,
                            seq_t0: None,
                            score: 0.0,
                        };
                        consider(pl, &mut best);
                    }
                }
            }
        }
        let thresh = -20.0;
        self.strike = best.filter(|b| b.score > thresh);
    }

    fn attack(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
        if self.d.plan > 0.0 && self.mode == BotMode::Attack && self.run_strike(car_idx, w, false) {
            return;
        }
        self.strike = None;
        let Some(it) = self.intercept else { return };
        let champ = self.is_champ();
        let now = w.time;
        let t_left = (self.intercept_t - (now - self.decided_at)).max(0.02);
        let p = self.intercept_pos;
        let tgt = self.shot_target(car_idx, w, p);
        let mut sd = DVec3::new(tgt.x - p.x, 0.0, tgt.z - p.z).three_normalize();
        let (ca, sa) = (self.aim_err.cos(), self.aim_err.sin());
        sd = DVec3::new(sd.x * ca - sd.z * sa, 0.0, sd.x * sa + sd.z * ca);
        self.shot_dir = sd;

        {
            let car = &w.cars[car_idx];
            let min_aer_boost = if champ { 18.0 } else { 25.0 };
            let max_aer_ang = if champ { 0.48 } else { 0.35 };
            if it.aerial
                && self.mode == BotMode::AerialPrep
                && car.on_ground
                && car.boost > min_aer_boost
            {
                let mut to_p = p - car.pos;
                to_p.y = 0.0;
                let ang = car.basis.f.three_angle_to(to_p.three_normalize()).abs();
                if ang < max_aer_ang && self.aerial_feasible(car, p, it.t, &w.phys) {
                    self.start_aerial(w.time, p, now + it.t, sd);
                    return;
                }
            }
        }

        let car_pos = w.cars[car_idx].pos;
        let to_ball = p - car_pos;
        let dist = to_ball.x.hypot(to_ball.z);
        let side = DVec3::new(car_pos.x - p.x, 0.0, car_pos.z - p.z);
        let along = side.dot(sd);
        let align_shot = -side.dot(sd) / dist.max(0.01);
        let mut target = if dist < 7.0 && align_shot > 0.65 {
            let mut t = p + sd * 1.2;
            if p.y < 2.5 {
                t.y = 0.0;
            }
            t
        } else if along > 0.5 && dist < 20.0 {
            let lat = DVec3::new(-sd.z, 0.0, sd.x);
            let s = if side.dot(lat) >= 0.0 { 1.0 } else { -1.0 };
            p + lat * (s * 5.2) - sd * 4.5
        } else {
            let off = clamp(dist * 0.38, 0.6, 7.0) * if p.y > 2.5 { 0.3 } else { 1.0 };
            let mut t = p - sd * (off + 0.9);
            if p.y < 2.5 {
                t.y = 0.0;
            }
            t
        };
        if p.y < 2.5 && dist >= 7.0 && !(along > 0.5 && dist < 20.0) {
            target.y = 0.0;
        }
        let eta = self.estimate(&w.cars[car_idx], p, true);
        let mut desired = 23.0;
        if eta < t_left - 0.35 {
            desired = clamp(dist / t_left.max(0.05) * 1.2, 5.0, 23.0);
        }
        if champ && align_shot > 0.72 && dist < 16.0 {
            desired = 23.0;
        } else if align_shot > 0.8 && dist < 12.0 {
            desired = desired.max(15.0 + self.p.aggro * 8.0);
        }
        self.drive_to(car_idx, target, desired, w, false);

        let car = &w.cars[car_idx];
        let to_b = w.ball.pos - car.pos;
        let bdist = to_b.length();
        let al = car.basis.to_local(to_b);
        let facing = al.x / bdist.max(0.01);
        let jump_cd = if champ { 0.75 } else { 1.0 };
        if car.on_ground && now - self.last_jump_t > jump_cd && align_shot > 0.4 {
            let h = p.y - 0.4;
            let jump_y_min = if champ { 1.9 } else { 2.0 };
            let jump_y_max = if champ { 5.0 } else { 4.6 };
            if p.y > jump_y_min
                && p.y < jump_y_max
                && (champ || self.d.jump_shot > w.rng.r#gen::<f64>() * 0.4 + 0.3)
            {
                let dbl = h > 2.3;
                let tj = jump_time_for(h - 0.3, dbl);
                let tol = if champ { 0.08 } else { 0.06 };
                if tj > 0.0 && (t_left - tj).abs() < tol && facing > 0.75 && dist < 23.0 * tj + 4.0
                {
                    self.start_jump(0.2, dbl);
                }
            } else if p.y < 1.8
                && bdist
                    < if champ {
                        3.7 + car.speed() * 0.11
                    } else {
                        3.4 + car.speed() * 0.1
                    }
                && bdist > if champ { 1.2 } else { 1.4 }
                && facing > if champ { 0.85 } else { 0.88 }
                && (champ || w.rng.r#gen::<f64>() < self.d.flip * 0.6)
            {
                let lat = clamp(al.z / bdist * 2.5, -1.0, 1.0);
                let lat_scale = if champ { 0.75 } else { 0.6 };
                self.start_flip(1.0, lat * lat_scale, false);
            }
        }
    }

    fn save(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
        if self.d.plan > 0.0 && self.run_strike(car_idx, w, true) {
            return;
        }
        self.strike = None;
        let champ = self.is_champ();
        let ad = Self::attack_dir(&w.cars[car_idx]);
        let gz = -ad * w.shape.goal_line_z();
        let it = self.intercept;
        let p_opt = it.filter(|x| !x.late).map(|x| x.slice.pos);
        let Some(p) = p_opt else {
            let t = Self::own_goal_guard_pt(&w.shape, w.ball.pos, ad, 2.0, false);
            self.drive_to(car_idx, t, 23.0, w, false);
            return;
        };
        let it = it.unwrap();
        let clear_dir = if let Some(sg) = w.shape.split_goal {
            let sx = if p.x >= 0.0 { 1.0 } else { -1.0 };
            DVec3::new(-sx * sg.gnx, 0.0, ad * (sg.gnz + 0.7)).three_normalize()
        } else {
            DVec3::new(if p.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, ad * 1.4).three_normalize()
        };
        let car_pos = w.cars[car_idx].pos;
        let to_p = p - car_pos;
        let dist = to_p.x.hypot(to_p.z);
        let side = DVec3::new(car_pos.x - p.x, 0.0, car_pos.z - p.z);
        let mut target = if side.dot(clear_dir) > 0.5 && dist < 12.0 {
            let mut t = p - clear_dir * 1.5;
            t.z = gz + ad * 0.5 + (p.z - gz) * 0.3;
            t
        } else {
            p - clear_dir * clamp(dist * 0.25, 0.5, 4.0)
        };
        if p.y < 2.5 {
            target.y = 0.0;
        }
        self.drive_to(car_idx, target, 23.0, w, false);

        let car = &w.cars[car_idx];
        let savant = self.is_savant();
        let min_aer_boost = if savant {
            10.0
        } else if champ {
            14.0
        } else {
            20.0
        };
        if it.aerial && car.on_ground && car.boost > min_aer_boost && self.d.aerial > 0.3 {
            self.start_aerial(w.time, p, w.time + it.t, clear_dir);
            return;
        }
        let to_b = w.ball.pos - car.pos;
        let bdist = to_b.length();
        let jump_cd = if savant {
            0.55
        } else if champ {
            0.75
        } else {
            1.0
        };
        if car.on_ground && w.time - self.last_jump_t > jump_cd {
            if p.y > 2.0 && p.y < if savant { 5.2 } else { 4.8 } && self.d.jump_shot > 0.3 {
                let tj = jump_time_for(p.y - 0.7, p.y > 2.7);
                if tj > 0.0
                    && (self.intercept_t - (w.time - self.decided_at) - tj).abs() < 0.09
                    && bdist < 11.0
                {
                    self.start_jump(0.2, p.y > 2.7);
                }
            } else if bdist < if savant {
                4.2
            } else if champ {
                3.6
            } else {
                3.2
            } && p.y < if savant { 2.1 } else { 1.8 }
                && self.d.flip > 0.3
            {
                let lat = if savant {
                    let al = car.basis.to_local(to_b);
                    clamp(al.z / bdist.max(0.1) * 2.2, -1.0, 1.0)
                } else {
                    0.0
                };
                self.start_flip(1.0, lat, false);
            }
        }
    }

    fn kickoff(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
        let champ = self.is_champ();
        let team = w.cars[car_idx].team;
        let ad = Self::attack_dir(&w.cars[car_idx]);
        let ball_pos = w.ball.pos;
        let mut kicker = w.brain.kickers[team as usize] == Some(car_idx);
        if kicker {
            let my_d = w.cars[car_idx].pos.distance(ball_pos);
            let mut new_kicker = None;
            for (i, o) in w.cars.iter().enumerate() {
                if i == car_idx || o.team != team || o.bot.is_some() || o.demolished {
                    continue;
                }
                let od = o.pos.distance(ball_pos);
                let to_b = (ball_pos - o.pos).three_normalize();
                if od < my_d + 2.0 && o.vel.dot(to_b) > 4.0 {
                    new_kicker = Some(i);
                    break;
                }
            }
            if let Some(nk) = new_kicker {
                w.brain.kickers[team as usize] = Some(nk);
                kicker = false;
            }
        }
        if kicker {
            let savant = self.is_savant();
            let car_pos = w.cars[car_idx].pos;
            let d = car_pos.distance(ball_pos);
            let mut t = ball_pos;
            if savant {
                // Aim slightly goal-side of ball center to win dead-center 50/50s
                t.x += if car_pos.x > 0.5 {
                    0.18
                } else if car_pos.x < -0.5 {
                    -0.18
                } else {
                    0.0
                };
                t.z -= ad * 0.55;
            } else {
                t.x += if car_pos.x > 0.5 {
                    0.25
                } else if car_pos.x < -0.5 {
                    -0.25
                } else {
                    0.0
                };
                t.z -= ad * 0.4;
            }
            self.drive_to(car_idx, t, 23.0, w, false);
            let (speed, facing, on_ground, local_inward) = {
                let car = &mut w.cars[car_idx];
                car.controls.boost = car.boost > 0.0 && self.d.kick > 0.4;
                car.controls.throttle = 1.0;
                let to_b = (ball_pos - car.pos).three_normalize();
                // Car-local right (+r) dot vector toward center x=0, works identically for team 0 and team 1!
                let inward = car.basis.r.x * (-car.pos.x);
                (car.speed(), car.basis.f.dot(to_b), car.on_ground, inward)
            };
            // SAVANT: Speedflip kickoff (flip-cancel diagonal boost-flip to reach supersonic first!)
            if savant
                && !self.kick_boost_flip
                && on_ground
                && d > 19.5
                && speed > 9.0
                && facing > 0.94
            {
                self.kick_boost_flip = true;
                let side = if local_inward >= 0.0 { 1.0 } else { -1.0 };
                let diag = if car_pos.x.abs() > 1.0 { 0.39 } else { 0.26 };
                self.start_speedflip(side, diag, true);
                return;
            }
            if !self.kick_done
                && on_ground
                && d < if savant {
                    4.4 + speed * 0.13
                } else if champ {
                    4.3 + speed * 0.14
                } else {
                    4.1 + speed * 0.14
                }
                && facing > if champ { 0.90 } else { 0.94 }
                && self.d.flip > 0.3
            {
                self.kick_done = true;
                let side_bias = if savant {
                    if car_pos.x.abs() > 0.5 {
                        if local_inward >= 0.0 { 0.24 } else { -0.24 }
                    } else {
                        0.0
                    }
                } else if champ {
                    if car_pos.x > 0.5 {
                        -0.25
                    } else if car_pos.x < -0.5 {
                        0.25
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                self.start_flip(1.0, side_bias, false);
            }
        } else {
            let team_count = w.cars.iter().filter(|o| o.team == team).count();
            let gz = -ad * w.shape.goal_line_z();
            if team_count > 2 && self.closest_to_goal(car_idx, w) {
                let guard = if let Some(sg) = w.shape.split_goal {
                    DVec3::new(0.0, 0.0, -ad * (sg.gz - 12.0))
                } else {
                    DVec3::new(0.0, 0.0, gz + ad * 2.0)
                };
                self.drive_to(car_idx, guard, 18.0, w, true);
                return;
            }
            let lane_pos = w.brain.kickers[team as usize]
                .filter(|&k| k != car_idx && !w.cars[k].demolished)
                .map(|k| w.cars[k].pos);
            let lane_d = |p: DVec3, lp: DVec3| -> f64 {
                let ax = lp.x;
                let az = lp.z;
                let bx = ball_pos.x - ax;
                let bz = ball_pos.z - az;
                let l2 = bx * bx + bz * bz;
                let l2 = if l2 == 0.0 { 1.0 } else { l2 };
                let t = clamp(((p.x - ax) * bx + (p.z - az) * bz) / l2, 0.0, 1.0);
                (p.x - ax - bx * t).hypot(p.z - az - bz * t)
            };
            if champ && team_count == 2 {
                let mut cheat = DVec3::new(0.0, 0.0, gz * 0.32);
                if let Some(lp) = lane_pos
                    && lane_d(cheat, lp) < 7.0
                {
                    let cx = w.cars[car_idx].pos.x;
                    let s = math_sign(cx - lp.x);
                    cheat.x = (if s != 0.0 { s } else { 1.0 }) * 9.0;
                }
                self.drive_to(car_idx, cheat, 20.0, w, false);
                return;
            }
            let pad = if let Some(lp) = lane_pos {
                let pred = |pos: DVec3| lane_d(pos, lp) > 7.0;
                self.best_pad(car_idx, w, 45.0, Some(&pred))
            } else {
                self.best_pad(car_idx, w, 45.0, None)
            };
            if let Some(pi) = pad
                && (w.pads[pi].big || lane_pos.is_some())
            {
                let ppos = w.pads[pi].pos;
                self.drive_to(car_idx, ppos, 23.0, w, false);
                return;
            }
            let mut t = DVec3::new(0.0, 0.0, gz * 0.5);
            if let Some(lp) = lane_pos
                && lane_d(t, lp) < 8.0
            {
                let cx = w.cars[car_idx].pos.x;
                let s = math_sign(cx - lp.x);
                t.x = (if s != 0.0 {
                    s
                } else if cx >= 0.0 {
                    1.0
                } else {
                    -1.0
                }) * 12.0;
            }
            self.drive_to(car_idx, t, 18.0, w, false);
        }
    }

    fn drive_to(
        &mut self,
        car_idx: usize,
        target: DVec3,
        desired_speed: f64,
        w: &mut World,
        cautious: bool,
    ) {
        let s = &w.shape;
        let car_pos = w.cars[car_idx].pos;
        let mut nav_target = target;
        if nav_target.y < 1.5 && nav_target.z.abs() < s.l - 0.5 {
            nav_target = s.push_inside(nav_target, 1.2);
        }
        let mut rerouted = false;
        if s.goal_mode == GoalMode::Standalone {
            let gz0 = s.goal_z;
            let gz1 = s.goal_z + s.gd + s.gt;
            let bx = s.gw + s.gt + 1.3;
            for end_sign in [-1.0_f64, 1.0_f64] {
                let cz = car_pos.z * end_sign;
                let tz = nav_target.z * end_sign;
                let car_in_net =
                    cz > gz0 - 0.5 && cz < gz1 + 0.4 && car_pos.x.abs() < s.gw + 0.4;
                let tgt_in_net =
                    tz > gz0 - 0.5 && tz < gz1 - 0.4 && nav_target.x.abs() < s.gw - 0.3;
                if car_in_net && !tgt_in_net {
                    // Must exit through the open front mouth, never through side/back walls
                    nav_target = DVec3::new(
                        clamp(car_pos.x, -s.gw + 1.8, s.gw - 1.8),
                        0.0,
                        end_sign * (gz0 - 4.0),
                    );
                    rerouted = true;
                    break;
                }
                if !car_in_net
                    && Self::standalone_seg_hits_box(s, car_pos, nav_target, end_sign, 1.2)
                {
                    let entering_mouth = cz <= gz0 + 1.0 && tgt_in_net;
                    if !entering_mouth {
                        let side_off = s.gw + s.gt + 4.4;
                        let cost_pos =
                            (car_pos.x - side_off).abs() + (nav_target.x - side_off).abs();
                        let cost_neg =
                            (car_pos.x + side_off).abs() + (nav_target.x + side_off).abs();
                        let side_x = if cost_pos <= cost_neg {
                            side_off
                        } else {
                            -side_off
                        };
                        let wp_z = if cz >= gz1 - 0.5 && car_pos.x.abs() < bx + 1.0 {
                            end_sign * (gz1 + 2.8)
                        } else if cz <= gz0 + 0.5 && car_pos.x.abs() < bx + 1.0 {
                            end_sign * (gz0 - 2.8)
                        } else if tz > gz1 {
                            end_sign * (gz1 + 3.0)
                        } else {
                            end_sign * (gz0 - 3.5)
                        };
                        nav_target = DVec3::new(side_x, 0.0, wp_z);
                        rerouted = true;
                        break;
                    }
                }
            }
        } else if s.fig8.is_some() && car_pos.z * nav_target.z < 0.0 {
            let dz = nav_target.z - car_pos.z;
            if dz.abs() > 1e-4 {
                let t0 = -car_pos.z / dz;
                let mid_x = car_pos.x + (nav_target.x - car_pos.x) * t0;
                let waist_w = (s.half_width_at(0.0) - 3.2).max(4.0);
                if mid_x.abs() > waist_w {
                    nav_target = DVec3::new(clamp(mid_x, -waist_w, waist_w), 0.0, 0.0);
                    rerouted = true;
                }
            }
        } else if s.goal_mode == GoalMode::Standard {
            let in_goal = car_pos.z.abs() > s.l - 0.5 && car_pos.x.abs() < s.gw;
            let tgt_in_goal = nav_target.z.abs() > s.l - 0.5 && nav_target.x.abs() < s.gw;
            if in_goal && nav_target.x.abs() > s.gw - 1.2 {
                nav_target = DVec3::new(
                    clamp(nav_target.x, -s.gw + 1.5, s.gw - 1.5),
                    0.0,
                    sign(car_pos.z) * (s.l - 2.0),
                );
                rerouted = true;
            } else if !in_goal
                && tgt_in_goal
                && car_pos.x.abs() > s.gw - 1.0
                && car_pos.z.abs() > s.l - 6.0
            {
                nav_target = DVec3::new(
                    clamp(car_pos.x, -s.gw + 1.5, s.gw - 1.5),
                    0.0,
                    sign(nav_target.z) * (s.l - 2.0),
                );
                rerouted = true;
            }
        }

        let champ = self.is_champ();
        let savant = self.is_savant();
        // SAVANT: opportunistic boost-pad pathing when driving across the field
        if savant
            && !rerouted
            && w.cars[car_idx].boost < 78.0
            && self.mode != BotMode::Save
            && self.mode != BotMode::Kickoff
            && nav_target.y < 2.2
        {
            let seg = DVec3::new(nav_target.x - car_pos.x, 0.0, nav_target.z - car_pos.z);
            let seg_len = seg.length();
            if seg_len > 14.0 {
                let seg_dir = seg / seg_len;
                let mut best_pad_pos: Option<DVec3> = None;
                let mut best_cost = 1e9_f64;
                for p in &w.pads {
                    if !p.active {
                        continue;
                    }
                    let rel = DVec3::new(p.pos.x - car_pos.x, 0.0, p.pos.z - car_pos.z);
                    let proj = rel.dot(seg_dir);
                    if proj < 3.5 || proj > seg_len - 5.5 {
                        continue;
                    }
                    let lat = (rel - seg_dir * proj).length();
                    let max_lat = if p.big && w.cars[car_idx].boost < 52.0 {
                        5.2
                    } else {
                        2.9
                    };
                    if lat < max_lat {
                        let cost = lat * if p.big { 0.45 } else { 1.0 } + proj * 0.04;
                        if cost < best_cost {
                            best_cost = cost;
                            best_pad_pos = Some(p.pos);
                        }
                    }
                }
                if let Some(pp) = best_pad_pos {
                    nav_target = pp;
                }
            }
        }
        let (ang, dist, vf, on_ground, boost, team) = {
            let car = &mut w.cars[car_idx];
            let b = car.basis;
            let to = nav_target - car.pos;
            let al = b.to_local(to);
            let dist = al.x.hypot(al.z);
            let ang = al.z.atan2(al.x);
            let yaw_rate = car.ang_vel.dot(b.u);
            car.controls.steer = clamp(ang * 3.2 + yaw_rate * 0.25, -1.0, 1.0);
            let vf = car.vel.dot(b.f);
            if ang.abs() > 2.3 && dist < 7.0 && vf.abs() < 6.0 {
                car.controls.throttle = -1.0;
                car.controls.steer = if ang > 0.0 { -1.0 } else { 1.0 };
                return;
            }
            let hb_ang = if savant {
                1.05
            } else if champ {
                1.15
            } else {
                1.6
            };
            let hb_vf = if champ { 7.0 } else { 8.0 };
            car.controls.handbrake = ang.abs() > hb_ang && vf > hb_vf && car.on_ground;
            let want = desired_speed * self.d.speed;
            if want > vf + 0.6 {
                car.controls.throttle = 1.0;
                let max_boost_ang = if savant {
                    0.46
                } else if champ {
                    0.42
                } else {
                    0.32
                };
                let min_gn_y = if savant { -0.25 } else { 0.6 };
                let boost_ok = self.d.boost > 0.2
                    && car.boost > 0.0
                    && ang.abs() < max_boost_ang
                    && car.on_ground
                    && car.ground_normal.y > min_gn_y
                    && (want > 14.5 || want > vf + 7.0)
                    && vf < 22.6;
                car.controls.boost = boost_ok && (!cautious || car.boost > if savant { 22.0 } else { 30.0 });
            } else if want < vf - 2.5 {
                car.controls.throttle = if vf > 0.0 { -1.0 } else { 0.0 };
            } else {
                car.controls.throttle = clamp(want / 14.0, 0.0, 1.0);
            }
            (ang, dist, vf, car.on_ground, car.boost, car.team)
        };

        if savant
            && (!cautious || matches!(self.mode, BotMode::Shadow | BotMode::Defend))
            && on_ground
            && w.cars[car_idx].ground_normal.y > 0.85
            && self.mode != BotMode::Kickoff
            && ang.abs() < 0.11
            && dist > 23.0
            && vf > 9.5
            && vf < 19.2
            && (boost < 32.0 || dist > 34.0)
            && w.time - self.last_jump_t > 1.25
        {
            let side = if ang >= 0.0 { 1.0 } else { -1.0 };
            self.start_speedflip(side, 0.32, boost > 14.0 && dist > 28.0);
        } else if !cautious
            && self.d.fix < 0.75
            && on_ground
            && ang.abs() < if champ { 0.09 } else { 0.06 }
            && dist > if champ { 22.0 } else { 30.0 }
            && vf > if champ { 8.5 } else { 9.0 }
            && vf < if champ { 18.0 } else { 16.0 }
            && boost < if champ { 25.0 } else { 15.0 }
            && self.d.flip > 0.5
            && w.time - self.last_jump_t > if champ { 1.4 } else { 2.0 }
        {
            self.start_flip(1.0, 0.0, false);
        }
        if self.d.err < 0.3 && on_ground && w.time - self.last_jump_t > 1.0 {
            let demo_d = if champ { 9.0 } else { 7.0 };
            let demo_v = if champ { 16.0 } else { 18.0 };
            let dodge_demo = w.cars.iter().any(|o| {
                if o.team != team && o.supersonic && !o.demolished {
                    let rel = car_pos - o.pos;
                    let dd = rel.length();
                    dd < demo_d && dd > 0.0 && o.vel.dot(rel) / dd > demo_v
                } else {
                    false
                }
            });
            if dodge_demo {
                self.start_jump(0.2, false);
            }
        }
    }

    fn recover(&self, car_idx: usize, _dt: f64, w: &mut World) {
        let champ = self.is_champ();
        let savant = self.is_savant();
        let car = &w.cars[car_idx];
        let look = car.pos + car.vel * 0.25;
        let dd = w.shape.dist(look.x, look.y, look.z);
        let up = if dd < 5.0 {
            w.shape.normal(look.x, look.y, look.z)
        } else {
            DVec3::Y
        };
        let mut fwd = car.vel - up * car.vel.dot(up);
        if fwd.length_squared() < 1.0 {
            fwd = car.basis.f - up * car.basis.f.dot(up);
        }
        fwd = fwd.three_normalize();
        self.orient(
            car_idx,
            fwd,
            Some(up),
            if savant {
                1.85
            } else if champ {
                1.5
            } else {
                1.0
            },
            w,
        );
        let car = &mut w.cars[car_idx];
        car.controls.throttle = 1.0;
        if champ && dd < 2.0 && car.basis.f.dot(fwd) < 0.85 {
            car.controls.handbrake = true;
        }
        // SAVANT: wavedash landing burst right before wheels touch the floor
        if savant
            && car.has_flip
            && !car.jumping
            && !car.prev_jump
            && up.y > 0.9
            && car.vel.y < -0.7
            && car.pos.y > 0.40
            && car.pos.y < 1.08
            && car.basis.u.y > 0.84
            && car.basis.f.dot(fwd) > 0.90
        {
            let hspd = car.vel.x.hypot(car.vel.z);
            if hspd > 5.0 && hspd < 19.5 {
                car.controls.jump = true;
                car.controls.pitch = 1.0;
                car.controls.steer = 0.0;
                car.controls.yaw = 0.0;
            }
        }
    }

    fn orient(&self, car_idx: usize, fwd: DVec3, up: Option<DVec3>, gain: f64, w: &mut World) {
        let car = &mut w.cars[car_idx];
        let b = car.basis;
        let tf = b.to_local(fwd);
        let wl = b.to_local(car.ang_vel);
        let pitch_err = tf.y.atan2(tf.x);
        let yaw_err = tf.z.atan2(tf.x);
        let kp = 3.6 * gain;
        let kd = 0.62;
        car.controls.pitch = clamp(-(kp * pitch_err - kd * wl.z), -1.0, 1.0);
        car.controls.yaw = clamp(kp * yaw_err + kd * wl.y, -1.0, 1.0);
        car.controls.steer = car.controls.yaw;
        if let Some(up_v) = up {
            let u = up_v - b.f * up_v.dot(b.f);
            let ru = u.dot(b.u);
            let rr = u.dot(b.r);
            let roll_err = rr.atan2(ru);
            car.controls.roll = clamp(2.8 * roll_err - 0.35 * wl.x, -1.0, 1.0);
        }
    }

    fn start_jump(&mut self, hold: f64, dbl: bool) {
        self.last_jump_t = self.now;
        let mut steps = vec![
            SeqStep {
                t: 0.0,
                jump: true,
                pitch: None,
                steer: 0.0,
                roll: 0.0,
                boost: false,
                end: false,
            },
            SeqStep {
                t: hold,
                jump: false,
                pitch: None,
                steer: 0.0,
                roll: 0.0,
                boost: false,
                end: false,
            },
        ];
        if dbl {
            steps.push(SeqStep {
                t: hold + 0.04,
                jump: true,
                pitch: None,
                steer: 0.0,
                roll: 0.0,
                boost: false,
                end: false,
            });
            steps.push(SeqStep {
                t: hold + 0.1,
                jump: false,
                pitch: None,
                steer: 0.0,
                roll: 0.0,
                boost: false,
                end: false,
            });
        }
        steps.push(SeqStep {
            t: hold + 0.9,
            jump: false,
            pitch: None,
            steer: 0.0,
            roll: 0.0,
            boost: false,
            end: true,
        });
        self.seq = Some(ActionSeq {
            t: 0.0,
            aim_ball: true,
            boost_flip: false,
            speedflip: false,
            steps,
        });
    }

    fn start_flip(&mut self, f: f64, s: f64, boost_flip: bool) {
        self.last_jump_t = self.now;
        self.seq = Some(ActionSeq {
            t: 0.0,
            aim_ball: false,
            boost_flip,
            speedflip: false,
            steps: vec![
                SeqStep {
                    t: 0.0,
                    jump: true,
                    pitch: None,
                    steer: 0.0,
                    roll: 0.0,
                    boost: false,
                    end: false,
                },
                SeqStep {
                    t: 0.07,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    roll: 0.0,
                    boost: false,
                    end: false,
                },
                SeqStep {
                    t: 0.1,
                    jump: true,
                    pitch: Some(f),
                    steer: s,
                    roll: 0.0,
                    boost: false,
                    end: false,
                },
                SeqStep {
                    t: 0.16,
                    jump: false,
                    pitch: Some(f),
                    steer: s,
                    roll: 0.0,
                    boost: false,
                    end: false,
                },
                SeqStep {
                    t: 0.7,
                    jump: false,
                    pitch: Some(0.0),
                    steer: 0.0,
                    roll: 0.0,
                    boost: false,
                    end: false,
                },
                SeqStep {
                    t: 1.0,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    roll: 0.0,
                    boost: false,
                    end: true,
                },
            ],
        });
    }

    /// SAVANT: True flip-cancel speedflip (diagonal dodge + pitch cancel at flip_t > 0.1 + boost through).
    fn start_speedflip(&mut self, side: f64, diag: f64, use_boost: bool) {
        self.last_jump_t = self.now;
        let f_in = (1.0 - diag * diag).max(0.5).sqrt();
        self.seq = Some(ActionSeq {
            t: 0.0,
            aim_ball: false,
            boost_flip: use_boost,
            speedflip: true,
            steps: vec![
                SeqStep {
                    t: 0.0,
                    jump: true,
                    pitch: Some(0.0),
                    steer: -side * (diag * 0.65),
                    roll: 0.0,
                    boost: use_boost,
                    end: false,
                },
                SeqStep {
                    t: 0.04,
                    jump: false,
                    pitch: Some(0.0),
                    steer: 0.0,
                    roll: 0.0,
                    boost: use_boost,
                    end: false,
                },
                SeqStep {
                    t: 0.065,
                    jump: true,
                    pitch: Some(f_in),
                    steer: side * diag,
                    roll: 0.0,
                    boost: use_boost,
                    end: false,
                },
                SeqStep {
                    t: 0.10,
                    jump: false,
                    pitch: Some(f_in),
                    steer: side * diag,
                    roll: 0.0,
                    boost: use_boost,
                    end: false,
                },
                // flip_t crosses 0.10s at seq.t ~ 0.17s -> pitch = -1.0 triggers physics flip-cancel!
                SeqStep {
                    t: 0.175,
                    jump: false,
                    pitch: Some(-1.0),
                    steer: 0.0,
                    roll: -side * 0.85,
                    boost: use_boost,
                    end: false,
                },
                SeqStep {
                    t: 0.38,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    roll: 0.0,
                    boost: use_boost,
                    end: false,
                },
                SeqStep {
                    t: 0.72,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    roll: 0.0,
                    boost: false,
                    end: true,
                },
            ],
        });
    }

    fn run_seq(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let (cur_opt, aim_ball, boost_flip, speedflip, seq_t) = {
            let Some(s) = self.seq.as_mut() else { return };
            s.t += dt;
            let mut cur = None;
            for &st in &s.steps {
                if st.t <= s.t {
                    cur = Some(st);
                }
            }
            (cur, s.aim_ball, s.boost_flip, s.speedflip, s.t)
        };
        let Some(cur) = cur_opt else { return };
        {
            let car = &mut w.cars[car_idx];
            car.controls.jump = cur.jump;
            car.controls.throttle = 1.0;
            if speedflip {
                let fwd_ok = car.basis.f.dot(car.vel.three_normalize()) > 0.75 || seq_t < 0.1;
                car.controls.boost = cur.boost && car.boost > 0.0 && fwd_ok && car.speed() < 22.8;
            } else if boost_flip && seq_t < 0.15 && car.boost > 6.0 {
                car.controls.boost = true;
            }
        }
        if let Some(pitch) = cur.pitch {
            let c = &mut w.cars[car_idx].controls;
            c.pitch = pitch;
            c.steer = cur.steer;
            c.yaw = cur.steer;
            c.roll = cur.roll;
        } else if speedflip && !w.cars[car_idx].on_ground {
            self.recover(car_idx, dt, w);
        } else if aim_ball && !w.cars[car_idx].on_ground {
            let fwd = (w.ball.pos - w.cars[car_idx].pos).three_normalize();
            self.orient(car_idx, fwd, None, 0.6, w);
        }
        let min_land = if speedflip { 0.22 } else { 0.3 };
        if cur.end || (seq_t > min_land && w.cars[car_idx].on_ground && !cur.jump) {
            self.seq = None;
        }
    }

    fn start_aerial(&mut self, now: f64, p: DVec3, t_hit: f64, aim_dir: DVec3) {
        self.last_jump_t = now;
        let dbl_h = if self.is_savant() { 5.8 } else { 7.0 };
        self.aer = Some(AerialState {
            p,
            t_hit,
            t: 0.0,
            dbl: p.y > dbl_h,
            dodged: false,
            aim_dir,
        });
    }

    fn fly_aerial(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let champ = self.is_champ();
        let savant = self.is_savant();
        let mut a = match self.aer {
            Some(a) => a,
            None => return,
        };
        a.t += dt;
        let t_rem = a.t_hit - w.time;
        if t_rem < -0.15 || a.t > 4.5 || (a.t > 0.38 && w.cars[car_idx].on_ground) {
            self.aer = None;
            return;
        }
        // Abort unreachable aerial early when out of boost so SAVANT lands & recovers immediately
        if savant
            && a.t > 0.35
            && w.cars[car_idx].boost < 1.5
            && a.p.y - w.cars[car_idx].pos.y > 3.5
            && w.cars[car_idx].vel.y < 1.0
        {
            self.aer = None;
            return;
        }
        let sl = w.pred.at(t_rem.max(0.02));
        a.p = a.p.lerp(
            sl.pos,
            if savant {
                0.85
            } else if champ {
                0.55
            } else {
                0.3
            },
        );
        if savant {
            // Fast aerial: frame-0 full pitch + immediate boost, then clean neutral double-jump
            if a.t < 0.17 {
                let car = &mut w.cars[car_idx];
                car.controls.jump = true;
                car.controls.pitch = -1.0;
                car.controls.boost = car.boost > 0.0;
                self.aer = Some(a);
                return;
            }
            if a.dbl && a.t < 0.20 {
                let car = &mut w.cars[car_idx];
                car.controls.jump = false;
                car.controls.pitch = 0.0;
                car.controls.steer = 0.0;
                car.controls.yaw = 0.0;
                car.controls.boost = car.boost > 0.0;
                self.aer = Some(a);
                return;
            }
            if a.dbl && a.t < 0.25 {
                let car = &mut w.cars[car_idx];
                car.controls.jump = true;
                // Strictly neutral stick so physics executes vertical double-jump (never backflip)
                car.controls.pitch = 0.0;
                car.controls.steer = 0.0;
                car.controls.yaw = 0.0;
                car.controls.boost = car.boost > 0.0;
                self.aer = Some(a);
                return;
            }
        } else {
            if a.t < 0.2 {
                let car = &mut w.cars[car_idx];
                car.controls.jump = true;
                car.controls.pitch = -0.6;
                car.controls.boost = a.t > if champ { 0.04 } else { 0.08 } && car.boost > 0.0;
                self.aer = Some(a);
                return;
            }
            if a.dbl && a.t < 0.26 {
                w.cars[car_idx].controls.jump = false;
            } else if a.dbl && a.t < 0.32 {
                w.cars[car_idx].controls.jump = true;
            }
        }
        let tc = t_rem.max(0.05);
        let car_pos = w.cars[car_idx].pos;
        let car_vel = w.cars[car_idx].vel;
        let aim_pt = if savant && t_rem > 0.06 {
            a.p - a.aim_dir * 0.78
        } else if champ && t_rem > 0.08 {
            a.p - a.aim_dir * 0.65
        } else {
            a.p
        };
        let mut d = aim_pt - car_pos - car_vel * tc;
        d.y += 0.5 * w.phys.g * tc * tc;
        let a_req = d * (2.0 / (tc * tc));
        let mag = a_req.length();
        let mut dir = if mag > 0.5 {
            a_req.three_normalize()
        } else {
            (aim_pt - car_pos).three_normalize()
        };
        if t_rem < 0.22 {
            dir = (aim_pt - car_pos).three_normalize();
        }
        self.orient(
            car_idx,
            dir,
            None,
            if savant {
                2.0
            } else if champ {
                1.6
            } else {
                1.2
            },
            w,
        );
        let ball_pos = w.ball.pos;
        let car = &mut w.cars[car_idx];
        let align = car.basis.f.dot(dir);
        let min_align = if savant {
            0.64
        } else if champ {
            0.72
        } else {
            0.78
        };
        car.controls.boost = car.boost > 0.0 && align > min_align && mag > 1.8;
        if t_rem < if champ { 0.24 } else { 0.2 }
            && car.pos.distance(ball_pos) < if savant {
                4.2
            } else if champ {
                4.0
            } else {
                3.5
            }
            && car.has_flip
            && self.d.flip > 0.6
        {
            let target_dir = if savant {
                (ball_pos + a.aim_dir * 0.35) - car.pos
            } else {
                ball_pos - car.pos
            };
            let lp = car.basis.to_local(target_dir);
            car.controls.jump = true;
            car.controls.pitch = if clamp(lp.x, -1.0, 1.0) > 0.0 {
                1.0
            } else {
                0.0
            };
            let st = clamp(lp.z * 2.0, -1.0, 1.0);
            car.controls.yaw = st;
            car.controls.steer = st;
            a.dodged = true;
        }
        self.aer = Some(a);
    }
}

#[derive(Clone, Debug, Default)]
pub struct TeamBrain {
    pub roles: HashMap<usize, Role>,
    pub slots: HashMap<usize, SupportSlot>,
    pub t: f64,
    pub kickers: [Option<usize>; 2],
}

impl TeamBrain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, dt: f64, cars: &[Car], ball: &Ball, arena_l: f64) {
        self.t -= dt;
        if self.t > 0.0 {
            return;
        }
        let has_savant = cars
            .iter()
            .any(|c| c.bot.as_ref().is_some_and(|b| b.is_savant()));
        let has_champ = cars
            .iter()
            .any(|c| c.bot.as_ref().is_some_and(|b| b.is_champ()));
        self.t = if has_savant {
            0.06
        } else if has_champ {
            0.15
        } else {
            0.3
        };
        for team in 0..=1_u8 {
            let mem: Vec<usize> = cars
                .iter()
                .enumerate()
                .filter(|(_, c)| c.team == team && !c.demolished)
                .map(|(i, _)| i)
                .collect();
            if mem.is_empty() {
                continue;
            }
            let ad = if team == 0 { 1.0 } else { -1.0 };
            let gz = -ad * arena_l;
            let mut scored: Vec<(usize, f64)> = mem
                .iter()
                .map(|&idx| {
                    let c = &cars[idx];
                    let est = match &c.bot {
                        Some(bot) => bot.estimate(c, ball.pos, true),
                        None => c.pos.distance(ball.pos) / 14.0,
                    };
                    let ahead = if (ball.pos.z - c.pos.z) * ad < -3.0 {
                        2.2
                    } else {
                        0.0
                    };
                    let low = if c.boost < 15.0 { 0.4 } else { 0.0 };
                    let ag = match &c.bot {
                        Some(bot) => (bot.p.aggro - 0.5) * 0.8,
                        None => -0.8,
                    };
                    (idx, est + ahead + low - ag)
                })
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            self.roles.insert(scored[0].0, Role::Attack);
            let mut rest: Vec<usize> = scored[1..].iter().map(|&(i, _)| i).collect();
            rest.sort_by(|&a, &b| {
                let da = cars[a].pos.x.hypot(cars[a].pos.z - gz);
                let db = cars[b].pos.x.hypot(cars[b].pos.z - gz);
                da.total_cmp(&db)
            });
            if rest.len() == 1 {
                let c0 = &cars[rest[0]];
                let is_def = c0.bot.as_ref().is_some_and(|b| b.p.defend > 0.7)
                    || (mem.len() == 2 && (ball.pos.z * ad) < 0.0);
                self.roles
                    .insert(rest[0], if is_def { Role::Defend } else { Role::Support });
            } else if rest.len() >= 2 {
                self.roles.insert(rest[0], Role::Defend);
                let mut sup: Vec<usize> = rest[1..].to_vec();
                sup.sort_by(|&a, &b| cars[a].pos.x.total_cmp(&cars[b].pos.x));
                let n = sup.len();
                for (i, idx) in sup.into_iter().enumerate() {
                    self.roles.insert(idx, Role::Support);
                    self.slots.insert(idx, SupportSlot { i, n });
                }
            }
        }
    }

    pub fn role_of(&self, car_idx: usize) -> Role {
        self.roles.get(&car_idx).copied().unwrap_or(Role::Attack)
    }

    pub fn slot_of(&self, car_idx: usize) -> Option<SupportSlot> {
        if self.roles.get(&car_idx) == Some(&Role::Support) {
            self.slots.get(&car_idx).copied()
        } else {
            None
        }
    }

    pub fn choose_kickers(&mut self, cars: &[Car], ball: &Ball) {
        for team in 0..=1_u8 {
            let mut best = None;
            let mut bd = 1e9_f64;
            for (i, c) in cars.iter().enumerate() {
                if c.team != team {
                    continue;
                }
                let d = c.pos.distance(ball.pos) - if c.bot.is_some() { 0.0 } else { 1.0 }
                    + if c.pos.x * (if team == 0 { 1.0 } else { -1.0 }) < -0.5 {
                        -0.01
                    } else {
                        0.0
                    };
                if d < bd {
                    bd = d;
                    best = Some(i);
                }
            }
            self.kickers[team as usize] = best;
        }
    }
}
