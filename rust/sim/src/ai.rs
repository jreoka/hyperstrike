use std::collections::HashMap;
use std::sync::OnceLock;

use glam::DVec3;
use rand::Rng;

use crate::arena::GoalMode;
use crate::math_ext::{
    PhysConfig, THROTTLE_CURVE, ThreeVec3Ext, clamp, curve, lerp, math_sign, sign,
};
use crate::physics::{Ball, Car, PredSlice};
use crate::powers::PowerTarget;
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
}

pub const DIFFICULTY: [Difficulty; 4] = [
    Difficulty {
        name: "ROOKIE",
        react: 0.34,
        aim: 0.32,
        speed: 0.82,
        boost: 0.4,
        aerial: 0.0,
        aerial_h: 0.0,
        flip: 0.15,
        jump_shot: 0.35,
        rot: 0.45,
        kick: 0.5,
        err: 0.9,
    },
    Difficulty {
        name: "PRO",
        react: 0.2,
        aim: 0.15,
        speed: 0.94,
        boost: 0.8,
        aerial: 0.4,
        aerial_h: 7.5,
        flip: 0.6,
        jump_shot: 0.75,
        rot: 0.75,
        kick: 0.8,
        err: 0.5,
    },
    Difficulty {
        name: "ALL-STAR",
        react: 0.11,
        aim: 0.07,
        speed: 1.0,
        boost: 1.0,
        aerial: 0.8,
        aerial_h: 13.0,
        flip: 0.9,
        jump_shot: 0.95,
        rot: 0.9,
        kick: 0.95,
        err: 0.25,
    },
    Difficulty {
        name: "LEGEND",
        react: 0.05,
        aim: 0.025,
        speed: 1.0,
        boost: 1.0,
        aerial: 1.0,
        aerial_h: 18.0,
        flip: 1.0,
        jump_shot: 1.0,
        rot: 1.0,
        kick: 1.0,
        err: 0.1,
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
    end: bool,
}

#[derive(Clone, Debug)]
struct ActionSeq {
    t: f64,
    aim_ball: bool,
    steps: Vec<SeqStep>,
}

#[derive(Clone, Copy, Debug)]
struct AerialState {
    p: DVec3,
    t_hit: f64,
    t: f64,
    dbl: bool,
    dodged: bool,
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
    pub pad_target: Option<usize>,
    pub stuck_t: f64,
    pub unstuck: f64,
    pub unstuck_steer: f64,
    pub item_hold: f64,
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
            pad_target: None,
            stuck_t: 0.0,
            unstuck: 0.0,
            unstuck_steer: 1.0,
            item_hold: 0.0,
        }
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
        self.pad_target = None;
        self.stuck_t = 0.0;
        self.unstuck = 0.0;
    }

    pub fn estimate(&self, car: &Car, target: DVec3, use_boost: bool) -> f64 {
        let to = target - car.pos;
        let dist = (to.length() - 1.6).max(0.0);
        let b = car.basis;
        let ang = b.f.three_angle_to(to.three_normalize());
        let vf = car.vel.dot(b.f).max(0.0);
        let turn = ang * (0.3 + vf * 0.012);
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
        let max_jump = if self.d.jump_shot > 0.5 { 3.4 } else { 2.0 };
        let mut i = 2;
        while i < pred.valid {
            let s = pred.slices[i];
            let near_surface = shape.dist(s.pos.x, s.pos.y, s.pos.z) < w.phys.ball_r + 1.4;
            if s.pos.y > max_jump && !(near_surface && s.pos.y < 14.0) {
                if self.d.aerial > 0.0
                    && s.pos.y < self.d.aerial_h
                    && car.boost > 25.0
                    && s.t < 3.5
                    && s.t > 0.6
                    && self.aerial_feasible(car, s.pos, s.t, &w.phys)
                {
                    return Intercept {
                        slice: s,
                        t: s.t,
                        aerial: true,
                        late: false,
                    };
                }
                i += 3;
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
            i += 3;
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
        d.y -= 4.2 * t * 0.8;
        let a_req = 2.0 * d.length() / (t * t);
        let fuel = a_req / phys.boost_acc * t * phys.boost_use;
        a_req < phys.boost_acc * 0.85 && fuel < car.boost - 5.0
    }

    pub fn update(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        self.update_core(car_idx, dt, w);
        if w.powers.is_some() && !w.cars[car_idx].demolished && !w.kickoff {
            self.use_items(car_idx, dt, w);
        }
    }

    fn use_items(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let ad = Self::attack_dir(&w.cars[car_idx]);
        if w.ball.stuck_to == Some(car_idx) {
            if w.cars[car_idx].on_ground && self.seq.is_none() && self.aer.is_none() {
                let s = &w.shape;
                let car_pos = w.cars[car_idx].pos;
                let goal = if let Some(sg) = s.split_goal {
                    let sx = if car_pos.x >= 0.0 { 1.0 } else { -1.0 };
                    DVec3::new(sx * (sg.gx + sg.gnx * 2.5), 0.0, ad * (sg.gz + sg.gnz * 2.5))
                } else {
                    let gx = clamp(car_pos.x * 0.2, -s.gw + 2.0, s.gw - 2.0);
                    DVec3::new(gx, 0.0, ad * (s.goal_line_z() + 3.0))
                };
                self.drive_to(car_idx, goal, 23.0, w, false);
                let car = &w.cars[car_idx];
                let to = goal - car.pos;
                let d = to.length();
                let facing = to.dot(car.basis.f) / d.max(0.01);
                if d < 16.0 + self.d.aim * 10.0 && facing > 0.9 && w.rng.r#gen::<f64>() < dt * 3.0 {
                    w.cars[car_idx].controls.item = true;
                }
            }
            return;
        }

        let Some(id) = w.cars[car_idx].item else {
            self.item_hold = 0.0;
            return;
        };
        self.item_hold += dt;
        if self.item_hold < 0.5 + self.d.react * 4.0 {
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

        let car = &w.cars[car_idx];
        let ball = &w.ball;
        let s = &w.shape;
        let tb = ball.pos - car.pos;
        let db = tb.length();
        let fdot = tb.dot(car.basis.f) / db.max(0.01);
        let glz = s.goal_line_z();
        let ball_to_goal_x = -ball.pos.x;
        let ball_to_goal_z = ad * glz - ball.pos.z;
        let bgl = ball_to_goal_x.hypot(ball_to_goal_z);
        let bgl = if bgl == 0.0 { 1.0 } else { bgl };
        let hl = tb.x.hypot(tb.z);
        let hl = if hl == 0.0 { 1.0 } else { hl };
        let align = (tb.x * ball_to_goal_x + tb.z * ball_to_goal_z) / (hl * bgl);
        let threat = w.pred.goal_team == car.team as i32 && w.pred.goal_time < 3.5;
        let own_goal_dist = (ball.pos.z + ad * glz).abs();

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
                (threat && own_goal_dist < 35.0)
                    || (ball.pos.y > 5.0
                        && w.pred.goal_team == (1 - car.team) as i32
                        && w.rng.r#gen::<f64>() < 0.02)
            }
            "spikes" => db < 13.0 && fdot > 0.6,
            "magnet" => db < 20.0 && align > 0.2,
            "grapple" => {
                db > 15.0
                    && fdot > 0.7
                    && matches!(
                        self.mode,
                        BotMode::Attack | BotMode::Save | BotMode::AerialPrep
                    )
            }
            "tether" => (threat && db > 8.0) || (db > 12.0 && (car.pos.z - ball.pos.z) * ad > 6.0),
            "punch" => align > 0.72 || (threat && (ball.pos.z - car.pos.z) * ad > 0.0),
            "boot" | "scramble" => tgt_pos
                .map(|p| p.distance(ball.pos) < 12.0 || p.distance(car.pos) < 10.0)
                .unwrap_or(false),
            "swap" => tgt_pos
                .map(|p| p.distance(ball.pos) < db - 14.0)
                .unwrap_or(false),
            "shock" => near_opp < 9.0 || (db < 9.0 && align > 0.5),
            "overdrive" => db < 22.0 || near_opp < 14.0,
            "antigrav" => {
                (ball.pos.y > 4.0 && w.pred.goal_team == (1 - car.team) as i32)
                    || (threat && ball.pos.y > 3.0)
                    || self.item_hold > 8.0
            }
            "nitro" => db > 18.0 || self.mode == BotMode::Save,
            _ => false,
        };
        if self.item_hold > 13.0 + self.d.err * 6.0 {
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
        if threat && (role == Role::Defend || role == Role::Attack || self.closest_to_goal(car_idx, w))
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
            let ahead = (w.ball.pos.z - car.pos.z) * ad < -4.0;
            let my_d = car.pos.distance(w.ball.pos);
            let opp_close = w.cars.iter().any(|o| {
                o.team != car.team && !o.demolished && o.pos.distance(w.ball.pos) < my_d - 6.0
            });
            if ahead && opp_close && self.d.rot > 0.6 && (w.ball.pos.z - own_z).abs() < w.shape.l * 1.2
            {
                self.mode = BotMode::Shadow;
            }
        }
        if self.d.boost > 0.3
            && ((self.mode == BotMode::Support && car.boost < 30.0)
                || (self.mode == BotMode::Defend && car.boost < 12.0))
            && let Some(pad) = self.best_pad(
                car_idx,
                w,
                if self.mode == BotMode::Defend {
                    18.0
                } else {
                    40.0
                },
                None,
            )
        {
            self.pad_target = Some(pad);
            self.mode = BotMode::Boost;
        }
        if self.mode == BotMode::AerialPrep
            && self.d.aerial * (0.6 + self.p.air * 0.6) < w.rng.r#gen::<f64>()
        {
            self.mode = BotMode::Attack;
        }
    }

    fn closest_to_goal(&self, car_idx: usize, w: &World) -> bool {
        let car = &w.cars[car_idx];
        let gz = -Self::attack_dir(car) * w.shape.goal_line_z();
        let mut best = None;
        let mut bd = 1e9_f64;
        for (i, o) in w.cars.iter().enumerate() {
            if o.team == car.team && !o.demolished {
                let d = o.pos.x.hypot(o.pos.z - gz);
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
                let gz = -ad * w.shape.goal_line_z();
                let t = w.shape.push_inside(
                    DVec3::new(w.ball.pos.x * 0.6, 0.0, lerp(w.ball.pos.z, gz, 0.55)),
                    5.0,
                );
                let spd = clamp(w.cars[car_idx].pos.distance(t) * 1.2, 4.0, 23.0);
                self.drive_to(car_idx, t, spd, w, true);
            }
            BotMode::Support => {
                let sl = w.brain.slot_of(car_idx);
                let n = sl.map(|s| s.n).unwrap_or(1);
                let glz = w.shape.goal_line_z();
                let (x, z) = if n <= 1 {
                    let z = clamp(
                        w.ball.pos.z - ad * (14.0 + self.p.defend * 8.0),
                        -glz + 6.0,
                        glz - 6.0,
                    );
                    let x = w.ball.pos.x * 0.45 + if w.cars[car_idx].pos.x > 0.0 { 6.0 } else { -6.0 };
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
                let gz = -ad * w.shape.goal_line_z();
                let t = DVec3::new(
                    clamp(w.ball.pos.x * 0.3, -w.shape.gw + 1.5, w.shape.gw - 1.5),
                    0.0,
                    gz + ad * 3.2,
                );
                let d = w.cars[car_idx].pos.distance(t);
                if d < 3.0 {
                    self.face_ball(car_idx, w);
                    return;
                }
                self.drive_to(car_idx, t, clamp(d * 1.2, 3.0, 23.0), w, true);
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
            car.controls.steer =
                clamp(ang * 2.0, -1.0, 1.0) * if car.controls.throttle < 0.0 { -1.0 } else { 1.0 };
        } else {
            car.controls.throttle = clamp(-vf * 0.4, -1.0, 1.0);
        }
    }

    fn shot_target(&self, car_idx: usize, w: &World, pos: DVec3) -> DVec3 {
        let shape = &w.shape;
        let ad = Self::attack_dir(&w.cars[car_idx]);
        let glz = shape.goal_line_z();
        let own_half = (pos.z * ad) < -glz * 0.35;
        if own_half {
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
        if shape.goal_mode == GoalMode::Standalone && (pos.z * ad) > shape.goal_z + 1.0 {
            return DVec3::new(0.0, 0.0, ad * (shape.goal_z - 9.0));
        }
        if let Some(sg) = shape.split_goal {
            let sx = if pos.x >= 0.0 { 1.0 } else { -1.0 };
            return DVec3::new(
                sx * (sg.gx + sg.gnx * 2.5),
                0.0,
                ad * (sg.gz + sg.gnz * 2.5),
            );
        }
        let opp_z = ad * (glz + 2.0);
        if self.p.pass > 0.6 && pos.x.abs() > shape.w * 0.55 && (pos.z * ad) > glz * 0.45 {
            return DVec3::new(0.0, 0.0, ad * (glz - 12.0));
        }
        let gx = clamp(pos.x * 0.35, -shape.gw + 2.2, shape.gw - 2.2);
        DVec3::new(gx, 0.0, opp_z)
    }

    fn attack(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
        let Some(it) = self.intercept else { return };
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
            if it.aerial && self.mode == BotMode::AerialPrep && car.on_ground && car.boost > 25.0 {
                let mut to_p = p - car.pos;
                to_p.y = 0.0;
                let ang = car.basis.f.three_angle_to(to_p.three_normalize()).abs();
                if ang < 0.35 && self.aerial_feasible(car, p, it.t, &w.phys) {
                    self.start_aerial(w.time, p, now + it.t);
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
        if align_shot > 0.8 && dist < 12.0 {
            desired = desired.max(15.0 + self.p.aggro * 8.0);
        }
        self.drive_to(car_idx, target, desired, w, false);

        let car = &w.cars[car_idx];
        let to_b = w.ball.pos - car.pos;
        let bdist = to_b.length();
        let al = car.basis.to_local(to_b);
        let facing = al.x / bdist.max(0.01);
        if car.on_ground && now - self.last_jump_t > 1.0 && align_shot > 0.4 {
            let h = p.y - 0.4;
            if p.y > 2.0 && p.y < 4.6 && self.d.jump_shot > w.rng.r#gen::<f64>() * 0.4 + 0.3 {
                let dbl = h > 2.3;
                let tj = jump_time_for(h - 0.3, dbl);
                if tj > 0.0 && (t_left - tj).abs() < 0.06 && facing > 0.75 && dist < 23.0 * tj + 4.0
                {
                    self.start_jump(0.2, dbl);
                }
            } else if p.y < 1.8
                && bdist < 3.4 + car.speed() * 0.1
                && bdist > 1.4
                && facing > 0.88
                && w.rng.r#gen::<f64>() < self.d.flip * 0.6
            {
                let lat = clamp(al.z / bdist * 2.5, -1.0, 1.0);
                self.start_flip(1.0, lat * 0.6);
            }
        }
    }

    fn save(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
        let ad = Self::attack_dir(&w.cars[car_idx]);
        let gz = -ad * w.shape.goal_line_z();
        let it = self.intercept;
        let p_opt = it.filter(|x| !x.late).map(|x| x.slice.pos);
        let Some(p) = p_opt else {
            let t = DVec3::new(
                clamp(w.ball.pos.x * 0.5, -w.shape.gw + 1.0, w.shape.gw - 1.0),
                0.0,
                gz + ad * 1.5,
            );
            self.drive_to(car_idx, t, 23.0, w, false);
            return;
        };
        let it = it.unwrap();
        let clear_dir = DVec3::new(if p.x >= 0.0 { 1.0 } else { -1.0 }, 0.0, ad * 1.4)
            .three_normalize();
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
        if it.aerial && car.on_ground && car.boost > 20.0 && self.d.aerial > 0.3 {
            self.start_aerial(w.time, p, w.time + it.t);
            return;
        }
        let bdist = car.pos.distance(w.ball.pos);
        if car.on_ground && w.time - self.last_jump_t > 1.0 {
            if p.y > 2.0 && p.y < 4.6 && self.d.jump_shot > 0.3 {
                let tj = jump_time_for(p.y - 0.7, p.y > 2.7);
                if tj > 0.0
                    && (self.intercept_t - (w.time - self.decided_at) - tj).abs() < 0.07
                    && bdist < 10.0
                {
                    self.start_jump(0.2, p.y > 2.7);
                }
            } else if bdist < 3.2 && p.y < 1.8 && self.d.flip > 0.3 {
                self.start_flip(1.0, 0.0);
            }
        }
    }

    fn kickoff(&mut self, car_idx: usize, _dt: f64, w: &mut World) {
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
            let car_pos = w.cars[car_idx].pos;
            let d = car_pos.distance(ball_pos);
            let mut t = ball_pos;
            t.x += if car_pos.x > 0.5 {
                0.25
            } else if car_pos.x < -0.5 {
                -0.25
            } else {
                0.0
            };
            t.z -= ad * 0.4;
            self.drive_to(car_idx, t, 23.0, w, false);
            let car = &mut w.cars[car_idx];
            car.controls.boost = car.boost > 0.0 && self.d.kick > 0.4;
            car.controls.throttle = 1.0;
            let to_b = (ball_pos - car.pos).three_normalize();
            let facing = car.basis.f.dot(to_b);
            if !self.kick_done
                && d < 4.1 + car.speed() * 0.14
                && facing > 0.94
                && self.d.flip > 0.3
            {
                self.kick_done = true;
                self.start_flip(1.0, 0.0);
            }
        } else {
            let team_count = w.cars.iter().filter(|o| o.team == team).count();
            let gz = -ad * w.shape.goal_line_z();
            if team_count > 2 && self.closest_to_goal(car_idx, w) {
                self.drive_to(car_idx, DVec3::new(0.0, 0.0, gz + ad * 2.0), 18.0, w, true);
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
        if s.goal_mode == GoalMode::Standalone {
            let gz0 = s.goal_z;
            let gz1 = s.goal_z + s.gd + s.gt;
            let gx_lim = s.gw + s.gt + 1.5;
            let car_behind = car_pos.z.abs() > gz1 && car_pos.x.abs() < gx_lim;
            let tgt_behind = nav_target.z.abs() > gz1 && nav_target.x.abs() < gx_lim;
            let same_end = car_pos.z * nav_target.z > 0.0;
            if same_end
                && (car_behind != tgt_behind)
                && car_pos.x.abs() < gx_lim
                && nav_target.x.abs() < gx_lim
            {
                let side_x = (if car_pos.x >= 0.0 { 1.0 } else { -1.0 }) * (s.gw + s.gt + 4.5);
                nav_target = DVec3::new(side_x, 0.0, sign(car_pos.z) * (gz0 + s.gd * 0.5));
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
            car.controls.handbrake = ang.abs() > 1.6 && vf > 8.0 && car.on_ground;
            let want = desired_speed * self.d.speed;
            if want > vf + 0.6 {
                car.controls.throttle = 1.0;
                let boost_ok = self.d.boost > 0.2
                    && car.boost > 0.0
                    && ang.abs() < 0.32
                    && car.on_ground
                    && car.ground_normal.y > 0.6
                    && (want > 14.5 || want > vf + 7.0)
                    && vf < 22.6;
                car.controls.boost = boost_ok && (!cautious || car.boost > 30.0);
            } else if want < vf - 2.5 {
                car.controls.throttle = if vf > 0.0 { -1.0 } else { 0.0 };
            } else {
                car.controls.throttle = clamp(want / 14.0, 0.0, 1.0);
            }
            (ang, dist, vf, car.on_ground, car.boost, car.team)
        };

        if !cautious
            && on_ground
            && ang.abs() < 0.06
            && dist > 30.0
            && vf > 9.0
            && vf < 16.0
            && boost < 15.0
            && self.d.flip > 0.5
            && w.time - self.last_jump_t > 2.0
        {
            self.start_flip(1.0, 0.0);
        }
        if self.d.err < 0.3 && on_ground && w.time - self.last_jump_t > 1.0 {
            let dodge_demo = w.cars.iter().any(|o| {
                if o.team != team && o.supersonic && !o.demolished {
                    let rel = car_pos - o.pos;
                    let dd = rel.length();
                    dd < 7.0 && dd > 0.0 && o.vel.dot(rel) / dd > 18.0
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
        self.orient(car_idx, fwd, Some(up), 1.0, w);
        w.cars[car_idx].controls.throttle = 1.0;
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
                end: false,
            },
            SeqStep {
                t: hold,
                jump: false,
                pitch: None,
                steer: 0.0,
                end: false,
            },
        ];
        if dbl {
            steps.push(SeqStep {
                t: hold + 0.04,
                jump: true,
                pitch: None,
                steer: 0.0,
                end: false,
            });
            steps.push(SeqStep {
                t: hold + 0.1,
                jump: false,
                pitch: None,
                steer: 0.0,
                end: false,
            });
        }
        steps.push(SeqStep {
            t: hold + 0.9,
            jump: false,
            pitch: None,
            steer: 0.0,
            end: true,
        });
        self.seq = Some(ActionSeq {
            t: 0.0,
            aim_ball: true,
            steps,
        });
    }

    fn start_flip(&mut self, f: f64, s: f64) {
        self.last_jump_t = self.now;
        self.seq = Some(ActionSeq {
            t: 0.0,
            aim_ball: false,
            steps: vec![
                SeqStep {
                    t: 0.0,
                    jump: true,
                    pitch: None,
                    steer: 0.0,
                    end: false,
                },
                SeqStep {
                    t: 0.07,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    end: false,
                },
                SeqStep {
                    t: 0.1,
                    jump: true,
                    pitch: Some(f),
                    steer: s,
                    end: false,
                },
                SeqStep {
                    t: 0.16,
                    jump: false,
                    pitch: Some(f),
                    steer: s,
                    end: false,
                },
                SeqStep {
                    t: 0.7,
                    jump: false,
                    pitch: Some(0.0),
                    steer: 0.0,
                    end: false,
                },
                SeqStep {
                    t: 1.0,
                    jump: false,
                    pitch: None,
                    steer: 0.0,
                    end: true,
                },
            ],
        });
    }

    fn run_seq(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let (cur_opt, aim_ball, seq_t) = {
            let Some(s) = self.seq.as_mut() else { return };
            s.t += dt;
            let mut cur = None;
            for &st in &s.steps {
                if st.t <= s.t {
                    cur = Some(st);
                }
            }
            (cur, s.aim_ball, s.t)
        };
        let Some(cur) = cur_opt else { return };
        {
            let c = &mut w.cars[car_idx].controls;
            c.jump = cur.jump;
            c.throttle = 1.0;
        }
        if let Some(pitch) = cur.pitch {
            let c = &mut w.cars[car_idx].controls;
            c.pitch = pitch;
            c.steer = cur.steer;
            c.yaw = cur.steer;
        } else if aim_ball && !w.cars[car_idx].on_ground {
            let fwd = (w.ball.pos - w.cars[car_idx].pos).three_normalize();
            self.orient(car_idx, fwd, None, 0.6, w);
        }
        if cur.end || (seq_t > 0.3 && w.cars[car_idx].on_ground && !cur.jump) {
            self.seq = None;
        }
    }

    fn start_aerial(&mut self, now: f64, p: DVec3, t_hit: f64) {
        self.last_jump_t = now;
        self.aer = Some(AerialState {
            p,
            t_hit,
            t: 0.0,
            dbl: p.y > 7.0,
            dodged: false,
        });
    }

    fn fly_aerial(&mut self, car_idx: usize, dt: f64, w: &mut World) {
        let mut a = match self.aer {
            Some(a) => a,
            None => return,
        };
        a.t += dt;
        let t_rem = a.t_hit - w.time;
        if t_rem < -0.15 || a.t > 4.5 || (a.t > 0.4 && w.cars[car_idx].on_ground) {
            self.aer = None;
            return;
        }
        let sl = w.pred.at(t_rem.max(0.02));
        a.p = a.p.lerp(sl.pos, 0.3);
        if a.t < 0.2 {
            let car = &mut w.cars[car_idx];
            car.controls.jump = true;
            car.controls.pitch = -0.6;
            car.controls.boost = a.t > 0.08 && car.boost > 0.0;
            self.aer = Some(a);
            return;
        }
        if a.dbl && a.t < 0.26 {
            w.cars[car_idx].controls.jump = false;
        } else if a.dbl && a.t < 0.32 {
            w.cars[car_idx].controls.jump = true;
        }
        let tc = t_rem.max(0.05);
        let car_pos = w.cars[car_idx].pos;
        let car_vel = w.cars[car_idx].vel;
        let mut d = a.p - car_pos - car_vel * tc;
        d.y += 0.5 * w.phys.g * tc * tc;
        let a_req = d * (2.0 / (tc * tc));
        let mag = a_req.length();
        let mut dir = if mag > 0.5 {
            a_req.three_normalize()
        } else {
            (a.p - car_pos).three_normalize()
        };
        if t_rem < 0.25 {
            dir = (a.p - car_pos).three_normalize();
        }
        self.orient(car_idx, dir, None, 1.2, w);
        let ball_pos = w.ball.pos;
        let car = &mut w.cars[car_idx];
        let align = car.basis.f.dot(dir);
        car.controls.boost = car.boost > 0.0 && align > 0.78 && mag > 2.2;
        if t_rem < 0.2
            && car.pos.distance(ball_pos) < 3.5
            && car.has_flip
            && self.d.flip > 0.6
        {
            let lp = car.basis.to_local(ball_pos - car.pos);
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
        self.t = 0.3;
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
                let d = c.pos.distance(ball.pos)
                    - if c.bot.is_some() { 0.0 } else { 1.0 }
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
