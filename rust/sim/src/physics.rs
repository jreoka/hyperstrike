use std::collections::HashMap;

use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::ai::BotAi;
use crate::arena::ArenaShape;
use crate::math_ext::{
    BUMP_CURVE, BUMP_UP_CURVE, Basis, Controls, EXTRA_IMPULSE_CURVE, HITBOXES, PhysConfig,
    STEER_CURVE, THROTTLE_CURVE, ThreeQuatExt, ThreeVec3Ext, clamp, curve, damp, lerp,
    make_controls, math_sign,
};

#[derive(Clone, Copy, Debug)]
pub struct TouchRecord {
    pub car_idx: usize,
    pub time: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BounceEvent {
    pub strength: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
}

#[derive(Clone, Debug)]
pub struct Ball {
    pub r: f64,
    pub pos: DVec3,
    pub vel: DVec3,
    pub ang_vel: DVec3,
    pub quat: DQuat,
    pub prev_pos: DVec3,
    pub prev_quat: DQuat,
    pub vis_err_pos: DVec3,
    pub vis_err_quat: DQuat,
    pub last_touch: Option<usize>,
    pub touch_history: Vec<TouchRecord>,
    pub last_hit_time: f64,
    pub last_hit_strength: f64,
    pub on_ground: bool,
    pub frozen: bool,
    pub stuck_to: Option<usize>,
    pub stuck_local: DVec3,
    pub ice_t: f64,
    pub g_scale: f64,
    pub anti_t: f64,
}

impl Ball {
    pub fn new(phys: &PhysConfig) -> Self {
        let r = phys.ball_r;
        Self {
            r,
            pos: DVec3::new(0.0, r, 0.0),
            vel: DVec3::ZERO,
            ang_vel: DVec3::ZERO,
            quat: DQuat::IDENTITY,
            prev_pos: DVec3::ZERO,
            prev_quat: DQuat::IDENTITY,
            vis_err_pos: DVec3::ZERO,
            vis_err_quat: DQuat::IDENTITY,
            last_touch: None,
            touch_history: Vec::new(),
            last_hit_time: -10.0,
            last_hit_strength: 0.0,
            on_ground: true,
            frozen: false,
            stuck_to: None,
            stuck_local: DVec3::ZERO,
            ice_t: 0.0,
            g_scale: 1.0,
            anti_t: 0.0,
        }
    }

    pub fn reset(&mut self, x: f64, y: f64, z: f64) {
        self.pos = DVec3::new(x, y, z);
        self.vel = DVec3::ZERO;
        self.ang_vel = DVec3::ZERO;
        self.prev_pos = self.pos;
        self.vis_err_pos = DVec3::ZERO;
        self.vis_err_quat = DQuat::IDENTITY;
        self.last_touch = None;
        self.touch_history.clear();
        self.stuck_to = None;
        self.ice_t = 0.0;
        self.g_scale = 1.0;
        self.anti_t = 0.0;
    }

    pub fn save_prev(&mut self) {
        self.prev_pos = self.pos;
        self.prev_quat = self.quat;
    }

    pub fn step(&mut self, dt: f64, arena: &ArenaShape, phys: &PhysConfig) -> Option<BounceEvent> {
        if self.frozen {
            return None;
        }
        let mut st = BallState {
            pos: self.pos,
            vel: self.vel,
            ang_vel: self.ang_vel,
            on_ground: self.on_ground,
            g_scale: self.g_scale,
        };
        let hit = ball_integrate(&mut st, dt, arena, phys, true);
        self.pos = st.pos;
        self.vel = st.vel;
        self.ang_vel = st.ang_vel;
        self.on_ground = st.on_ground;
        self.quat = self.quat.integrate(self.ang_vel, dt);
        hit
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BallState {
    pub pos: DVec3,
    pub vel: DVec3,
    pub ang_vel: DVec3,
    pub on_ground: bool,
    pub g_scale: f64,
}

pub fn ball_integrate(
    b: &mut BallState,
    dt: f64,
    arena: &ArenaShape,
    phys: &PhysConfig,
    want_event: bool,
) -> Option<BounceEvent> {
    let r = phys.ball_r;
    b.vel.y -= phys.g * b.g_scale * dt;
    // Subtle Magnus effect (spin-induced curve in flight)
    if b.ang_vel.length_squared() > 1e-4 {
        let bt = b.ang_vel.cross(b.vel);
        b.vel += bt * (phys.ball_magnus * dt);
    }
    let drag = 1.0 - phys.ball_drag * dt;
    b.vel *= drag;
    let sp = b.vel.length();
    if sp > phys.ball_max {
        b.vel *= phys.ball_max / sp;
    }
    b.pos += b.vel * dt;
    let mut ev = None;
    b.on_ground = false;
    for _ in 0..3 {
        let d = arena.dist(b.pos.x, b.pos.y, b.pos.z);
        if d >= r {
            break;
        }
        let n = arena.normal(b.pos.x, b.pos.y, b.pos.z);
        b.pos += n * (r - d);
        if n.y > 0.7 {
            b.on_ground = true;
        }
        let vn = b.vel.dot(n);
        if vn < 0.0 {
            let br = n * (-r);
            let bc = b.ang_vel.cross(br) + b.vel;
            let e = if vn < -1.4 {
                phys.ball_rest
            } else if vn < -0.35 {
                phys.ball_rest * ((-vn - 0.35) / 1.05)
            } else {
                0.0
            };
            let jn = -(1.0 + e) * vn;
            b.vel += n * jn;
            let cvn = bc.dot(n);
            let bt_vec = bc - n * cvn;
            let vt = bt_vec.length();
            if vt > 1e-4 {
                let bt_dir = bt_vec / vt;
                let need = vt * (2.0 / 7.0);
                let jt = need.min(phys.ball_fric * jn + 0.02);
                b.vel -= bt_dir * jt;
                let ang_imp = br.cross(bt_dir) * (-jt / (0.4 * r * r));
                b.ang_vel += ang_imp;
            }
            if want_event && -vn > 2.0 {
                ev = Some(BounceEvent {
                    strength: -vn,
                    x: b.pos.x - n.x * r,
                    y: b.pos.y - n.y * r,
                    z: b.pos.z - n.z * r,
                    nx: n.x,
                    ny: n.y,
                    nz: n.z,
                });
            }
        }
    }
    let w = b.ang_vel.length();
    if w > phys.ball_max_ang {
        b.ang_vel *= phys.ball_max_ang / w;
    }
    if b.on_ground {
        b.ang_vel *= 1.0 - 0.03 * dt;
    }
    ev
}

#[derive(Clone, Copy, Debug)]
pub struct PredSlice {
    pub t: f64,
    pub pos: DVec3,
    pub vel: DVec3,
    pub goal: i32,
}

#[derive(Clone, Debug)]
pub struct BallPredictor {
    pub dt: f64,
    pub steps: usize,
    pub slices: Vec<PredSlice>,
    pub goal_team: i32,
    pub goal_time: f64,
    pub valid: usize,
}

impl Default for BallPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BallPredictor {
    pub fn new() -> Self {
        let steps = 300;
        Self {
            dt: 1.0 / 60.0,
            steps,
            slices: vec![
                PredSlice {
                    t: 0.0,
                    pos: DVec3::ZERO,
                    vel: DVec3::ZERO,
                    goal: -1,
                };
                steps
            ],
            goal_team: -1,
            goal_time: 99.0,
            valid: 0,
        }
    }

    pub fn update(&mut self, ball: &Ball, arena: &ArenaShape, phys: &PhysConfig) {
        let mut s = BallState {
            pos: ball.pos,
            vel: if ball.ice_t > 0.0 { DVec3::ZERO } else { ball.vel },
            ang_vel: ball.ang_vel,
            on_ground: false,
            g_scale: 1.0,
        };
        self.goal_team = -1;
        self.goal_time = 99.0;
        self.valid = self.steps;
        let hold = if ball.ice_t > 0.0 {
            ball.ice_t
        } else if ball.stuck_to.is_some() {
            0.5
        } else {
            0.0
        };
        let mut anti = ball.anti_t;
        for i in 0..self.steps {
            s.g_scale = if anti > 0.0 { 0.2 } else { 1.0 };
            anti -= self.dt;
            let t = (i + 1) as f64 * self.dt;
            if t < hold {
                s.pos += s.vel * self.dt;
                self.slices[i] = PredSlice {
                    t,
                    pos: s.pos,
                    vel: s.vel,
                    goal: -1,
                };
                continue;
            }
            ball_integrate(&mut s, self.dt, arena, phys, false);
            let g = arena.in_goal(s.pos, phys.ball_r);
            self.slices[i] = PredSlice {
                t,
                pos: s.pos,
                vel: s.vel,
                goal: g,
            };
            if g >= 0 && self.goal_team < 0 {
                self.goal_team = g;
                self.goal_time = t;
                self.valid = i + 1;
                break;
            }
        }
    }

    #[inline]
    pub fn at(&self, t: f64) -> PredSlice {
        if self.valid == 0 {
            return self.slices[0];
        }
        let raw = (t / self.dt).round() as isize - 1;
        let idx = raw.clamp(0, (self.valid - 1) as isize) as usize;
        self.slices[idx]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Wheel {
    pub local: DVec3,
    pub front: bool,
    pub r: f64,
    pub contact: bool,
    pub len: f64,
    pub prev_len: f64,
    pub normal: DVec3,
    pub spin: f64,
    pub steer: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CarStats {
    pub score: i32,
    pub goals: i32,
    pub assists: i32,
    pub saves: i32,
    pub shots: i32,
    pub demos: i32,
    pub touches: i32,
    pub powers: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarEventKind {
    Jump,
    DoubleJump,
    Flip,
    Land,
    Sonic,
    Crash,
}

impl CarEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CarEventKind::Jump => "jump",
            CarEventKind::DoubleJump => "djump",
            CarEventKind::Flip => "flip",
            CarEventKind::Land => "land",
            CarEventKind::Sonic => "sonic",
            CarEventKind::Crash => "crash",
        }
    }
}

const WHEEL_LAYOUT: [(f64, f64, bool); 4] = [
    (0.50, -0.37, true),
    (0.50, 0.37, true),
    (-0.47, -0.38, false),
    (-0.47, 0.38, false),
];

#[derive(Clone, Debug)]
pub struct Car {
    pub id: usize,
    pub team: u8,
    pub name: String,
    pub design: usize,
    pub half: DVec3,
    pub inertia: DVec3,
    pub pos: DVec3,
    pub vel: DVec3,
    pub quat: DQuat,
    pub ang_vel: DVec3,
    pub prev_pos: DVec3,
    pub prev_quat: DQuat,
    pub vis_err_pos: DVec3,
    pub vis_err_quat: DQuat,
    pub basis: Basis,
    pub controls: Controls,
    pub prev_jump: bool,
    pub boost: f64,
    pub unlimited_boost: bool,
    pub unlimited_boost_always: bool,
    pub wheels: [Wheel; 4],
    pub contacts: u8,
    pub on_ground: bool,
    pub ground_normal: DVec3,
    pub jumping: bool,
    pub jump_hold_t: f64,
    pub jumped: bool,
    pub has_flip: bool,
    pub time_since_jump: f64,
    pub double_jumped: bool,
    pub flipping: bool,
    pub flip_t: f64,
    pub flip_axis: DVec3,
    pub ground_lock: f64,
    pub coyote_t: f64,
    pub air_time: f64,
    pub body_contact_t: f64,
    pub body_contact: bool,
    pub is_boosting: bool,
    pub boost_t: f64,
    pub supersonic: bool,
    pub was_supersonic: bool,
    pub sliding: f64,
    pub traction: f64,
    pub slip_speed: f64,
    pub handbrake: bool,
    pub demolished: bool,
    pub respawn_t: f64,
    pub blast_t: f64,
    pub last_ball_touch_tick: i64,
    pub bump_cooldown: HashMap<usize, f64>,
    pub stats: CarStats,
    pub bot: Option<BotAi>,
    pub is_player: bool,
    pub is_remote: bool,
    pub events: Vec<CarEventKind>,
    pub sample_pts: Vec<DVec3>,

    // Power-up fields (Mayhem)
    pub pw_init: bool,
    pub item: Option<&'static str>,
    pub item_next: f64,
    pub prev_item: bool,
    pub last_item: Option<&'static str>,
    pub spike_t: f64,
    pub mag_t: f64,
    pub ov_t: f64,
    pub nitro_t: f64,
    pub scr_t: f64,
    pub torn_t: f64,
    pub speed_mul: f64,
    pub hook_t: f64,
}

impl Car {
    pub fn new(id: usize, team: u8, name: impl Into<String>, design: usize) -> Self {
        let phys = PhysConfig::default();
        let hb = HITBOXES.get(design).copied().unwrap_or(HITBOXES[0]);
        let half = DVec3::new(hb.hx, hb.hy, hb.hz);
        let l = hb.hx * 2.0;
        let hh = hb.hy * 2.0;
        let wd = hb.hz * 2.0;
        let inertia = DVec3::new(
            (hh * hh + wd * wd) / 12.0,
            (l * l + wd * wd) / 12.0,
            (l * l + hh * hh) / 12.0,
        );
        let wheels = std::array::from_fn(|i| {
            let (wx, wz, front) = WHEEL_LAYOUT[i];
            Wheel {
                local: DVec3::new(wx * hb.hx / 0.59, -0.04, wz * hb.hz / 0.42),
                front,
                r: 0.17,
                contact: false,
                len: phys.susp_rest,
                prev_len: phys.susp_rest,
                normal: DVec3::Y,
                spin: 0.0,
                steer: 0.0,
            }
        });
        let mut sample_pts = Vec::with_capacity(26);
        for sx in [-1.0, 0.0, 1.0] {
            for sy in [-1.0, 0.0, 1.0] {
                for sz in [-1.0, 0.0, 1.0] {
                    if sx == 0.0 && sy == 0.0 && sz == 0.0 {
                        continue;
                    }
                    sample_pts.push(DVec3::new(sx * half.x, sy * half.y, sz * half.z));
                }
            }
        }
        Self {
            id,
            team,
            name: name.into(),
            design,
            half,
            inertia,
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
            quat: DQuat::IDENTITY,
            ang_vel: DVec3::ZERO,
            prev_pos: DVec3::ZERO,
            prev_quat: DQuat::IDENTITY,
            vis_err_pos: DVec3::ZERO,
            vis_err_quat: DQuat::IDENTITY,
            basis: Basis::IDENTITY,
            controls: make_controls(),
            prev_jump: false,
            boost: 33.0,
            unlimited_boost: false,
            unlimited_boost_always: false,
            wheels,
            contacts: 0,
            on_ground: false,
            ground_normal: DVec3::Y,
            jumping: false,
            jump_hold_t: 0.0,
            jumped: false,
            has_flip: false,
            time_since_jump: 0.0,
            double_jumped: false,
            flipping: false,
            flip_t: 0.0,
            flip_axis: DVec3::ZERO,
            ground_lock: 0.0,
            coyote_t: 0.0,
            air_time: 0.0,
            body_contact_t: 0.0,
            body_contact: false,
            is_boosting: false,
            boost_t: 0.0,
            supersonic: false,
            was_supersonic: false,
            sliding: 0.0,
            traction: 0.0,
            slip_speed: 0.0,
            handbrake: false,
            demolished: false,
            respawn_t: 0.0,
            blast_t: 0.0,
            last_ball_touch_tick: -10,
            bump_cooldown: HashMap::new(),
            stats: CarStats::default(),
            bot: None,
            is_player: false,
            is_remote: false,
            events: Vec::new(),
            sample_pts,
            pw_init: false,
            item: None,
            item_next: 0.0,
            prev_item: false,
            last_item: None,
            spike_t: 0.0,
            mag_t: 0.0,
            ov_t: 0.0,
            nitro_t: 0.0,
            scr_t: 0.0,
            torn_t: 0.0,
            speed_mul: 0.0,
            hook_t: 0.0,
        }
    }

    #[inline]
    pub fn speed(&self) -> f64 {
        self.vel.length()
    }

    pub fn save_prev(&mut self) {
        self.prev_pos = self.pos;
        self.prev_quat = self.quat;
    }

    pub fn place(&mut self, x: f64, y: f64, z: f64, yaw: f64) {
        let phys = PhysConfig::default();
        self.pos = DVec3::new(x, y, z);
        self.vel = DVec3::ZERO;
        self.ang_vel = DVec3::ZERO;
        self.quat = DQuat::from_rotation_y(yaw);
        self.save_prev();
        self.vis_err_pos = DVec3::ZERO;
        self.vis_err_quat = DQuat::IDENTITY;
        self.jumping = false;
        self.jumped = false;
        self.flipping = false;
        self.double_jumped = false;
        self.has_flip = false;
        self.air_time = 0.0;
        self.coyote_t = 0.0;
        self.supersonic = false;
        self.was_supersonic = false;
        self.is_boosting = false;
        self.boost_t = 0.0;
        if self.bot.is_some() {
            self.controls = make_controls();
        }
        self.basis = Basis::from_quat(self.quat);
        self.contacts = 4;
        self.on_ground = true;
        self.ground_normal = DVec3::Y;
        self.ground_lock = 0.0;
        self.blast_t = 0.0;
        self.sliding = 0.0;
        self.flip_t = 0.0;
        self.jump_hold_t = 0.0;
        self.time_since_jump = 0.0;
        self.body_contact_t = 0.0;
        self.body_contact = false;
        for w in &mut self.wheels {
            w.len = phys.susp_rest - 0.044;
            w.prev_len = w.len;
            w.contact = true;
        }
    }

    #[inline]
    pub fn apply_ang_impulse(&mut self, l_vec: DVec3, mass: f64) {
        let mut cl = self.basis.to_local(l_vec);
        cl.x /= self.inertia.x * mass;
        cl.y /= self.inertia.y * mass;
        cl.z /= self.inertia.z * mass;
        self.ang_vel += self.basis.to_world(cl);
    }

    #[inline]
    pub fn inv_inertia_term(&self, r: DVec3, n: DVec3, mass: f64) -> f64 {
        let mut cl = self.basis.to_local(r.cross(n));
        cl.x /= self.inertia.x * mass;
        cl.y /= self.inertia.y * mass;
        cl.z /= self.inertia.z * mass;
        let w = self.basis.to_world(cl);
        n.dot(w.cross(r))
    }

    pub fn step(&mut self, dt: f64, arena: &ArenaShape, phys: &PhysConfig) {
        self.events.clear();
        if self.demolished {
            return;
        }
        let c = self.controls;
        self.basis = Basis::from_quat(self.quat);
        let b = self.basis;
        let f = b.f;
        let u = b.u;
        let mut acc = DVec3::new(0.0, -phys.g, 0.0);
        let mut ang_acc = DVec3::ZERO;
        let jump_pressed = c.jump && !self.prev_jump;
        self.prev_jump = c.jump;
        if self.ground_lock > 0.0 {
            self.ground_lock -= dt;
        }

        let mut contacts: u8 = 0;
        let mut gn = DVec3::ZERO;
        let down = -u;
        for w in &mut self.wheels {
            let wp = b.to_world(w.local) + self.pos;
            w.prev_len = w.len;
            let max_t = phys.susp_max + w.r + if w.contact { phys.susp_grace } else { 0.0 };
            let t = if self.ground_lock > 0.0 {
                -1.0
            } else {
                arena.raycast(wp, down, max_t)
            };
            if t >= 0.0 {
                w.contact = true;
                contacts += 1;
                let len = (t - w.r).max(0.0).min(phys.susp_max);
                w.len = len;
                let hit_p = wp + down * t;
                w.normal = arena.normal(hit_p.x, hit_p.y, hit_p.z);
                gn += w.normal;
                let comp = phys.susp_rest - len;
                let rate = clamp((w.prev_len - len) / dt, -8.0, 8.0);
                let force = clamp(phys.susp_k * comp + phys.susp_c * rate, 0.0, 22.0);
                acc += u * force;
                let rel = wp - self.pos;
                let torque = rel.cross(u) * force;
                let cl = b.to_local(torque);
                ang_acc.x += cl.x / self.inertia.x;
                ang_acc.y += cl.y / self.inertia.y;
                ang_acc.z += cl.z / self.inertia.z;
            } else {
                w.contact = false;
                w.len = (w.len + dt * 1.5).min(phys.susp_max);
            }
        }

        let was_ground = self.on_ground;
        self.contacts = contacts;
        if contacts > 0 {
            self.ground_normal = gn.three_normalize();
        } else {
            self.ground_normal = DVec3::Y;
        }
        self.on_ground = contacts >= 3;
        if contacts >= 2 && !self.jumping && !self.jumped {
            self.coyote_t = 0.08;
        } else {
            self.coyote_t = (self.coyote_t - dt).max(0.0);
        }
        let n = self.ground_normal;

        let throttle_in = if c.boost && (self.boost > 0.0 || self.unlimited_boost) {
            1.0
        } else {
            c.throttle
        };
        self.handbrake = c.handbrake;
        let cf = contacts as f64 / 4.0;
        self.traction = if contacts == 0 {
            0.0
        } else {
            let base = if self.traction != 0.0 {
                self.traction
            } else {
                cf
            };
            damp(base, cf, 30.0, dt)
        };

        if contacts > 0 {
            let traction = self.traction;
            let stick = phys.sticky * (1.0 + (1.0 - n.y.abs()));
            let slow_on_wall =
                n.y.abs() < 0.6 && self.vel.length_squared() < 9.0 && throttle_in.abs() < 0.1;
            let hooked = self.hook_t > 0.0;
            if !self.jumping && !slow_on_wall && !hooked {
                acc += n * (-stick);
            }
            let ft = (f - n * f.dot(n)).three_normalize();
            let rt = ft.cross(n);
            let vf = self.vel.dot(ft);
            let vr = self.vel.dot(rt);
            let a = if throttle_in.abs() > 0.01 {
                if vf * throttle_in < 0.0 && vf.abs() > 0.15 {
                    if hooked {
                        0.0
                    } else {
                        math_sign(throttle_in) * phys.brake
                    }
                } else {
                    throttle_in * curve(&THROTTLE_CURVE, vf.abs())
                }
            } else if !hooked {
                -math_sign(vf) * phys.coast.min(vf.abs() / dt)
            } else {
                0.0
            };
            acc += ft * (a * traction);
            let sliding = c.handbrake;
            self.sliding = damp(
                self.sliding,
                if sliding { 1.0 } else { 0.0 },
                if sliding { 14.0 } else { 3.5 },
                dt,
            );
            let max_lat = if hooked {
                3.0
            } else {
                lerp(62.0, 6.5, self.sliding)
            };
            let lim = max_lat * dt * traction;
            let dvl = clamp(-vr, -lim, lim);
            self.vel += rt * dvl;
            self.slip_speed = vr.abs();
            let planar_sp = vf.hypot(vr);
            let drift_dir = if vf.abs() > 0.5 {
                math_sign(vf)
            } else if throttle_in.abs() > 0.1 {
                math_sign(throttle_in)
            } else {
                1.0
            };
            let eff_vf = if self.sliding > 0.1 {
                lerp(vf, drift_dir * planar_sp, self.sliding * 0.75)
            } else {
                vf
            };
            let v_steer = if eff_vf.abs() < 2.0 && throttle_in.abs() > 0.1 {
                math_sign(throttle_in) * eff_vf.abs().max(2.0)
            } else {
                eff_vf
            };
            let target_yaw = -c.steer
                * v_steer
                * curve(&STEER_CURVE, v_steer.abs())
                * (1.0 + 0.55 * self.sliding);
            let cur_yaw = self.ang_vel.dot(n);
            self.ang_vel += n * ((target_yaw - cur_yaw) * (dt * 22.0 * traction).min(1.0));

            if contacts >= 2 && !self.jumping {
                let align_scale = if contacts >= 3 { 1.0 } else { 0.45 };
                let perp = self.ang_vel - n * self.ang_vel.dot(n);
                self.ang_vel -= perp * (dt * 12.0 * align_scale).min(1.0);
                self.ang_vel += u.cross(n) * (dt * 70.0 * align_scale);
                if self.flipping && self.flip_axis.x.abs() > 0.5 {
                    let rl = self.ang_vel.dot(f);
                    self.ang_vel -= f * (rl * 0.6);
                }
                if contacts >= 3 {
                    let vn = self.vel.dot(n);
                    if vn < 0.0 {
                        let spd = self.vel.length();
                        self.vel -= n * vn;
                        if -vn < spd * 0.45 {
                            let l = self.vel.length();
                            if l > 1e-3 {
                                self.vel *= spd.min(l + (spd - l) * 0.85) / l;
                            }
                        }
                    }
                }
            }
            if !self.jumping {
                self.jumped = false;
                self.double_jumped = false;
                self.has_flip = true;
                self.flipping = false;
            }
        } else {
            self.sliding = damp(
                self.sliding,
                if c.handbrake { 0.85 } else { 0.0 },
                if c.handbrake { 10.0 } else { 4.0 },
                dt,
            );
            self.slip_speed = 0.0;
        }

        self.time_since_jump += dt;
        if jump_pressed
            && (contacts >= 2 || self.coyote_t > 0.0)
            && !self.jumping
            && self.ground_lock <= 0.0
        {
            self.vel += u * phys.jump_imp;
            self.jumping = true;
            self.jump_hold_t = 0.0;
            self.jumped = true;
            self.has_flip = true;
            self.double_jumped = false;
            self.time_since_jump = 0.0;
            self.ground_lock = 0.12;
            self.coyote_t = 0.0;
            self.events.push(CarEventKind::Jump);
        } else if jump_pressed && contacts < 2 && self.body_contact && self.contacts == 0 {
            self.vel.y += 3.5;
            let mut axis = u.cross(DVec3::Y);
            if axis.length_squared() < 0.01 {
                axis = f;
            }
            self.ang_vel += axis.three_normalize() * 7.5;
            self.events.push(CarEventKind::Jump);
        } else if jump_pressed
            && contacts == 0
            && !self.jumping
            && self.has_flip
            && (self.time_since_jump < phys.dodge_window || !self.jumped)
        {
            let fin = c.pitch;
            let sin = if c.air_roll_free { 0.0 } else { c.steer };
            let mag = fin.hypot(sin);
            if mag > 0.4 {
                self.dodge(fin / mag, sin / mag, phys);
            } else {
                self.vel += u * phys.jump_imp;
                self.double_jumped = true;
                self.events.push(CarEventKind::DoubleJump);
            }
            self.has_flip = false;
        }
        if self.jumping {
            if c.jump && self.jump_hold_t < phys.jump_hold_t {
                acc += u * phys.jump_hold;
                self.jump_hold_t += dt;
            } else {
                self.jumping = false;
            }
        }

        if contacts == 0 {
            self.air_time += dt;
            let mut wl = b.to_local(self.ang_vel);
            let mut pin = c.pitch;
            let yin = if c.air_roll_free { 0.0 } else { c.yaw };
            let mut rin = c.roll;
            if self.flipping {
                self.flip_t += dt;
                if self.flip_t < phys.flip_time {
                    let cancel = self.flip_t > 0.1
                        && self.flip_axis.z * (-pin) < -0.3
                        && self.flip_axis.z.abs() > 0.3;
                    if !cancel {
                        wl.x = self.flip_axis.x * phys.flip_rate;
                        wl.z = self.flip_axis.z * phys.flip_rate;
                        self.ang_vel = b.to_world(wl);
                        pin = 0.0;
                        rin = 0.0;
                    } else {
                        wl.z *= 0.2;
                        self.ang_vel = b.to_world(wl);
                        self.flipping = false;
                    }
                    if self.flip_t > 0.15 && self.vel.y < 0.0 {
                        self.vel.y *= 0.35_f64.powf(dt * 4.0);
                    }
                } else if self.flip_t > 1.05 {
                    self.flipping = false;
                } else {
                    pin *= 0.3;
                    rin *= 0.3;
                }
            }
            let cl = b.to_local(self.ang_vel);
            let damp_flag = if self.flipping { 0.0 } else { 1.0 };
            ang_acc.x += phys.t_roll * rin - phys.d_roll * cl.x * damp_flag;
            ang_acc.z +=
                -phys.t_pitch * pin - damp_flag * (1.0 - pin.abs()) * phys.d_pitch * cl.z;
            ang_acc.y += -phys.t_yaw * yin - (1.0 - yin.abs()) * phys.d_yaw * cl.y;
            acc += f * (phys.air_throttle * c.throttle);
        } else {
            if !was_ground && self.on_ground && self.air_time > 0.25 {
                self.events.push(CarEventKind::Land);
            }
            self.air_time = 0.0;
        }

        self.is_boosting = false;
        if c.boost && (self.boost > 0.0 || self.unlimited_boost) {
            acc += f * (phys.boost_acc * if self.nitro_t > 0.0 { 1.45 } else { 1.0 });
            self.is_boosting = true;
            self.boost_t += dt;
            if !self.unlimited_boost {
                self.boost = (self.boost - phys.boost_use * dt).max(0.0);
            }
        } else {
            self.boost_t = 0.0;
        }

        self.vel += acc * dt;
        if self.blast_t > 0.0 {
            self.blast_t -= dt;
        }
        let sp = self.vel.length();
        let smul = if self.speed_mul != 0.0 {
            self.speed_mul
        } else {
            1.0
        };
        let blast_mul = if self.blast_t > 0.0 {
            1.0 + 0.6 * self.blast_t.min(1.0)
        } else {
            1.0
        };
        let cmax = phys.car_max * smul * blast_mul;
        if sp > cmax {
            self.vel *= cmax / sp;
        }
        self.was_supersonic = self.supersonic;
        self.supersonic =
            sp >= phys.supersonic - 0.01 || (self.supersonic && sp > phys.supersonic - 1.0);
        if self.supersonic && !self.was_supersonic {
            self.events.push(CarEventKind::Sonic);
        }
        let world_ang_acc = b.to_world(ang_acc);
        self.ang_vel += world_ang_acc * dt;
        let w_len = self.ang_vel.length();
        if w_len > phys.car_max_ang {
            self.ang_vel *= phys.car_max_ang / w_len;
        }
        self.pos += self.vel * dt;
        self.quat = self.quat.integrate(self.ang_vel, dt);

        self.collide_arena(arena, phys);

        let vf = self.vel.dot(f);
        for wh in &mut self.wheels {
            wh.spin += (vf / wh.r) * dt;
            if wh.front {
                wh.steer = damp(
                    wh.steer,
                    -c.steer * lerp(0.55, 0.18, clamp(vf.abs() / 23.0, 0.0, 1.0)),
                    14.0,
                    dt,
                );
            }
        }
    }

    pub fn dodge(&mut self, fin: f64, sin: f64, phys: &PhysConfig) {
        let b = self.basis;
        let mut fh = DVec3::new(b.f.x, 0.0, b.f.z);
        if fh.length_squared() < 0.04 {
            fh = DVec3::new(-b.u.x, 0.0, -b.u.z);
        }
        fh = fh.three_normalize();
        let rh = DVec3::new(-fh.z, 0.0, fh.x);
        let vfh = self.vel.dot(fh);
        let sp_ratio = vfh.abs() / phys.car_max;
        let f_imp = if fin >= 0.0 {
            5.0 * fin
        } else {
            5.33 * fin * (1.0 + 1.5 * sp_ratio)
        };
        let s_imp = 5.0 * sin * (1.0 + 0.9 * sp_ratio);
        let sideness = sin.abs() / (sin.abs() + fin.abs() + 1e-4);
        self.vel += fh * f_imp + rh * s_imp;
        self.vel.y = if self.vel.y > 0.0 {
            self.vel.y * (0.25 + 0.15 * sideness)
        } else {
            0.0
        };
        self.flipping = true;
        self.flip_t = 0.0;
        self.flip_axis = DVec3::new(sin, 0.0, -fin).three_normalize();
        let cl = DVec3::new(
            self.flip_axis.x * phys.flip_rate,
            0.0,
            self.flip_axis.z * phys.flip_rate,
        );
        self.ang_vel = b.to_world(cl);
        self.events.push(CarEventKind::Flip);
    }

    pub fn collide_arena(&mut self, arena: &ArenaShape, phys: &PhysConfig) {
        self.basis = Basis::from_quat(self.quat);
        let mut touched = false;
        for _ in 0..3 {
            let mut worst = 0.0;
            let mut wp: Option<DVec3> = None;
            for &lp in &self.sample_pts {
                let p = self.basis.to_world(lp) + self.pos;
                let d = arena.dist(p.x, p.y, p.z);
                if d < worst {
                    worst = d;
                    wp = Some(lp);
                }
            }
            let Some(lp) = wp else { break };
            touched = true;
            let cr = self.basis.to_world(lp);
            let world_p = cr + self.pos;
            let nrm = arena.normal(world_p.x, world_p.y, world_p.z);
            self.pos += nrm * (-worst);
            let vp = self.ang_vel.cross(cr) + self.vel;
            let vn = vp.dot(nrm);
            if vn < 0.0 {
                let m = phys.car_mass;
                let k = 1.0 / m + self.inv_inertia_term(cr, nrm, m);
                let e = if vn < -3.5 { 0.18 } else { 0.0 };
                let j = -(1.0 + e) * vn / k;
                self.vel += nrm * (j / m);
                self.apply_ang_impulse(cr.cross(nrm) * j, m);
                let vp2 = self.ang_vel.cross(cr) + self.vel;
                let vnn = vp2.dot(nrm);
                let tan = vp2 - nrm * vnn;
                let vt = tan.length();
                if vt > 1e-3 {
                    let t_dir = tan / vt;
                    let kt = 1.0 / m + self.inv_inertia_term(cr, t_dir, m);
                    let jt = (vt / kt).min(0.22 * j);
                    self.vel -= t_dir * (jt / m);
                    self.apply_ang_impulse(cr.cross(t_dir) * (-jt), m);
                }
                if -vn > 4.0 {
                    self.events.push(CarEventKind::Crash);
                }
            }
            self.basis = Basis::from_quat(self.quat);
        }
        self.body_contact = touched;
        if touched && self.contacts == 0 {
            self.body_contact_t += phys.dt;
            if self.body_contact_t > 0.35 {
                let mut axis = self.basis.u.cross(DVec3::Y);
                if axis.length_squared() < 0.001 {
                    axis = self.basis.f;
                }
                self.ang_vel += axis.three_normalize() * (phys.dt * 18.0);
            }
        } else {
            self.body_contact_t = 0.0;
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CarBallHit {
    pub car_idx: usize,
    pub strength: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub ball_speed: f64,
    pub prev_speed: f64,
    pub shot: bool,
    pub save: bool,
    pub epic: bool,
}

pub fn collide_car_ball(
    car: &mut Car,
    car_idx: usize,
    ball: &mut Ball,
    tick: i64,
    arena: Option<&ArenaShape>,
    phys: &PhysConfig,
) -> Option<CarBallHit> {
    if car.demolished {
        return None;
    }
    car.basis = Basis::from_quat(car.quat);
    let b = car.basis;
    let h = car.half;
    let kl = b.to_local(ball.pos - car.pos);
    let cx = clamp(kl.x, -h.x, h.x);
    let cy = clamp(kl.y, -h.y, h.y);
    let cz = clamp(kl.z, -h.z, h.z);
    let mut dx = kl.x - cx;
    let mut dy = kl.y - cy;
    let mut dz = kl.z - cz;
    let mut dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let r = ball.r;
    if dist >= r {
        return None;
    }
    if dist < 1e-5 {
        dx = 0.0;
        dy = 1.0;
        dz = 0.0;
        dist = 1e-5;
    }
    let kn = b.to_world(DVec3::new(dx / dist, dy / dist, dz / dist));
    let kcp = b.to_world(DVec3::new(cx, cy, cz)) + car.pos;
    let pen = r - dist;
    let mc = phys.car_mass;
    let mb = phys.ball_mass;
    let ball_v0 = ball.vel;
    let ball_speed0 = ball_v0.length();
    ball.pos += kn * (pen * mc / (mc + mb));
    car.pos -= kn * (pen * mb / (mc + mb));
    if let Some(a) = arena {
        let bd = a.dist(ball.pos.x, ball.pos.y, ball.pos.z);
        if bd < r {
            let bn = a.normal(ball.pos.x, ball.pos.y, ball.pos.z);
            ball.pos += bn * (r - bd);
        }
    }
    let rc = kcp - car.pos;
    let rb = kn * (-r);
    let vc = car.ang_vel.cross(rc) + car.vel;
    let pre_rel = ball.vel - vc;
    let rel_speed = pre_rel.length().min(46.0);
    let vb = ball.ang_vel.cross(rb) + ball.vel;
    let vrel = vb - vc;
    let vn = vrel.dot(kn);
    let ang_recoil = if car.contacts == 0 { 0.25 } else { 0.4 };
    if vn < 0.0 {
        let k = 1.0 / mb + 1.0 / mc + car.inv_inertia_term(rc, kn, mc);
        let j = -vn / k;
        ball.vel += kn * (j / mb);
        car.vel -= kn * (j / mc);
        car.apply_ang_impulse(rc.cross(kn) * (-j * ang_recoil), mc);
        let vt_vec = vrel - kn * vn;
        let vtl = vt_vec.length();
        if vtl > 1e-3 {
            let vt = vt_vec / vtl;
            let kt = 1.0 / mb
                + 1.0 / mc
                + car.inv_inertia_term(rc, vt, mc)
                + (r * r) / (0.4 * mb * r * r);
            let jt = (vtl / kt).min(2.0 * j);
            ball.vel -= vt * (jt / mb);
            car.vel += vt * (jt / mc);
            ball.ang_vel += rb.cross(vt) * (-jt / (0.4 * mb * r * r));
            car.apply_ang_impulse(rc.cross(vt) * (jt * ang_recoil), mc);
        }
    }
    // Mid-air flip reset
    if car.contacts == 0
        && !car.has_flip
        && kl.y < -h.y * 0.45
        && b.u.dot(kn) < -0.72
        && kl.x.abs() < h.x * 0.8
        && kl.z.abs() < h.z * 0.8
    {
        car.has_flip = true;
        car.jumped = false;
        car.time_since_jump = 0.0;
    }
    let first = tick - car.last_ball_touch_tick > 1;
    car.last_ball_touch_tick = tick;
    if first && rel_speed > 0.01 {
        let on_roof = b.u.dot(kn) > 0.72 && !car.flipping;
        let dribble_scale = if on_roof {
            clamp((rel_speed - 1.2) / 2.6, 0.0, 1.0)
        } else {
            1.0
        };
        if dribble_scale > 0.01 {
            let mut hit = ball.pos - car.pos;
            hit.y *= 0.35;
            hit = hit.three_normalize();
            let fd = hit.dot(b.f);
            hit = (hit - b.f * (fd * 0.35)).three_normalize();
            let flip_boost = if car.flipping { 1.12 } else { 1.0 };
            let extra =
                rel_speed * curve(&EXTRA_IMPULSE_CURVE, rel_speed) * dribble_scale * flip_boost;
            ball.vel += hit * extra;
        }
    }
    let sp = ball.vel.length();
    if sp > phys.ball_max {
        ball.vel *= phys.ball_max / sp;
    }
    if !first {
        return None;
    }
    let dv = (ball.vel - ball_v0).length();
    Some(CarBallHit {
        car_idx,
        strength: dv,
        x: kcp.x,
        y: kcp.y,
        z: kcp.z,
        ball_speed: ball.vel.length(),
        prev_speed: ball_speed0,
        shot: false,
        save: false,
        epic: false,
    })
}

pub fn obb_overlap(a: &mut Car, b: &mut Car) -> Option<(f64, DVec3)> {
    a.basis = Basis::from_quat(a.quat);
    b.basis = Basis::from_quat(b.quat);
    let ca = [a.basis.f, a.basis.u, a.basis.r];
    let cb = [b.basis.f, b.basis.u, b.basis.r];
    let ha = [a.half.x, a.half.y, a.half.z];
    let hb = [b.half.x, b.half.y, b.half.z];
    let t_vec = b.pos - a.pos;

    let mut axes = [DVec3::ZERO; 15];
    let mut n = 0;
    for v in ca {
        axes[n] = v;
        n += 1;
    }
    for v in cb {
        axes[n] = v;
        n += 1;
    }
    for va in ca {
        for vb in cb {
            let cr = va.cross(vb);
            if cr.length_squared() > 1e-6 {
                axes[n] = cr.three_normalize();
                n += 1;
            }
        }
    }

    let mut min_o = f64::INFINITY;
    let mut best = f64::INFINITY;
    let mut out_n = DVec3::ZERO;
    for (k, &axis) in axes.iter().take(n).enumerate() {
        let ra = ha[0] * ca[0].dot(axis).abs()
            + ha[1] * ca[1].dot(axis).abs()
            + ha[2] * ca[2].dot(axis).abs();
        let rb = hb[0] * cb[0].dot(axis).abs()
            + hb[1] * cb[1].dot(axis).abs()
            + hb[2] * cb[2].dot(axis).abs();
        let d = t_vec.dot(axis);
        let o = ra + rb - d.abs();
        if o <= 0.0 {
            return None;
        }
        let score = if k < 6 { o } else { o * 1.15 + 0.02 };
        if score < best {
            best = score;
            min_o = o;
            out_n = if d < 0.0 { -axis } else { axis };
        }
    }
    Some((min_o, out_n))
}

#[derive(Clone, Copy, Debug)]
pub enum CarCollisionResult {
    Touch {
        a_idx: usize,
        b_idx: usize,
        strength: f64,
    },
    Bump {
        attacker_idx: usize,
        victim_idx: usize,
        strength: f64,
    },
    Demo {
        attacker_idx: usize,
        victim_idx: usize,
        overdrive: bool,
    },
}

pub fn collide_cars(
    a: &mut Car,
    a_idx: usize,
    b: &mut Car,
    b_idx: usize,
    now: f64,
    allow_demos: bool,
    phys: &PhysConfig,
) -> Option<CarCollisionResult> {
    if a.demolished || b.demolished {
        return None;
    }
    if a.pos.distance_squared(b.pos) > 4.0 {
        return None;
    }
    let (pen, n) = obb_overlap(a, b)?;
    let corr = (pen - 0.004).max(0.0) * 0.8 * 0.5;
    a.pos -= n * corr;
    b.pos += n * corr;

    let ra_n = a.half.x * a.basis.f.dot(n).abs()
        + a.half.y * a.basis.u.dot(n).abs()
        + a.half.z * a.basis.r.dot(n).abs();
    let mid = (a.pos + b.pos) * 0.5;
    let ccp = mid + n * ((a.pos.dot(n) + ra_n - (pen - 2.0 * corr) * 0.5) - mid.dot(n));
    let ra = ccp - a.pos;
    let rb = ccp - b.pos;
    let va = a.ang_vel.cross(ra) + a.vel;
    let vb = b.ang_vel.cross(rb) + b.vel;
    let vrel = vb - va;
    let vn = vrel.dot(n);
    let m = phys.car_mass;
    if vn < 0.0 {
        let k = 2.0 / m + a.inv_inertia_term(ra, n, m) + b.inv_inertia_term(rb, n, m);
        let j = -(1.0 + clamp(0.18 + (-vn) * 0.02, 0.18, 0.38)) * vn / k;
        a.vel -= n * (j / m);
        b.vel += n * (j / m);
        a.apply_ang_impulse(ra.cross(n) * (-j), m);
        b.apply_ang_impulse(rb.cross(n) * j, m);

        let vt_vec = vrel - n * vn;
        let vtl = vt_vec.length();
        if vtl > 1e-3 {
            let vt = vt_vec / vtl;
            let kt = 2.0 / m + a.inv_inertia_term(ra, vt, m) + b.inv_inertia_term(rb, vt, m);
            let jt = (vtl / kt).min(0.35 * j);
            a.vel += vt * (jt / m);
            b.vel -= vt * (jt / m);
            a.apply_ang_impulse(ra.cross(vt) * jt, m);
            b.apply_ang_impulse(rb.cross(vt) * (-jt), m);
        }
        let wa = a.ang_vel.length();
        if wa > phys.car_max_ang {
            a.ang_vel *= phys.car_max_ang / wa;
        }
        let wb = b.ang_vel.length();
        if wb > phys.car_max_ang {
            b.ang_vel *= phys.car_max_ang / wb;
        }
    }

    if a.ov_t > 0.0 && a.team != b.team && allow_demos {
        return Some(CarCollisionResult::Demo {
            attacker_idx: a_idx,
            victim_idx: b_idx,
            overdrive: true,
        });
    }
    if b.ov_t > 0.0 && b.team != a.team && allow_demos {
        return Some(CarCollisionResult::Demo {
            attacker_idx: b_idx,
            victim_idx: a_idx,
            overdrive: true,
        });
    }

    let mut result = CarCollisionResult::Touch {
        a_idx,
        b_idx,
        strength: -vn,
    };

    for pass in 0..2 {
        let (att, att_idx, vic, vic_idx, dir) = if pass == 0 {
            (&mut *a, a_idx, &mut *b, b_idx, 1.0)
        } else {
            (&mut *b, b_idx, &mut *a, a_idx, -1.0)
        };
        let nd = n * dir;
        let fwd_hit = att.basis.f.dot(nd) > 0.55;
        let side_hit = !fwd_hit && att.basis.r.dot(nd).abs() > 0.6;
        let closing = att.vel.dot(nd) - vic.vel.dot(nd);
        if (!fwd_hit && !side_hit) || closing < if fwd_hit { 4.0 } else { 5.0 } {
            continue;
        }
        let last = att.bump_cooldown.get(&vic.id).copied().unwrap_or(-10.0);
        if now - last < 0.3 {
            continue;
        }
        att.bump_cooldown.insert(vic.id, now);
        if fwd_hit && att.supersonic && att.team != vic.team && allow_demos {
            result = CarCollisionResult::Demo {
                attacker_idx: att_idx,
                victim_idx: vic_idx,
                overdrive: false,
            };
            break;
        }
        let rs = closing.min(46.0);
        let mut bdir = DVec3::new(nd.x, 0.0, nd.z);
        if bdir.length_squared() < 1e-4 {
            bdir = att.basis.f;
        }
        bdir = bdir.three_normalize();
        let av = DVec3::new(att.vel.x, 0.0, att.vel.z);
        if av.length_squared() > 1.0 {
            bdir = bdir
                .lerp(av.three_normalize(), if fwd_hit { 0.4 } else { 0.15 })
                .three_normalize();
        }
        let scale = if fwd_hit { 0.85 } else { 0.4 };
        let up_v = curve(&BUMP_UP_CURVE, rs)
            * if vic.on_ground { 1.0 } else { 0.4 }
            * if fwd_hit { 1.0 } else { 0.3 };
        vic.vel += bdir * (curve(&BUMP_CURVE, rs) * scale) + DVec3::new(0.0, up_v, 0.0);
        let mut off = vic.pos - att.pos;
        off -= bdir * off.dot(bdir);
        vic.ang_vel +=
            DVec3::Y * clamp(off.dot(DVec3::Y.cross(bdir)) * rs * 0.25, -2.5, 2.5);
        if vic.on_ground {
            vic.ground_lock = vic.ground_lock.max(0.06);
        }
        result = CarCollisionResult::Bump {
            attacker_idx: att_idx,
            victim_idx: vic_idx,
            strength: rs,
        };
        break;
    }
    Some(result)
}
