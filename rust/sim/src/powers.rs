use glam::DVec3;
use rand::Rng;

use crate::math_ext::{Basis, ThreeQuatExt, ThreeVec3Ext, clamp, pick, rand_range};
use crate::physics::{Ball, Car, CarBallHit};
use crate::world::{World, WorldEvent};

pub const SWAP_CHARGE: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerTargetKind {
    None,
    Ball,
    Enemy,
}

#[derive(Clone, Copy, Debug)]
pub struct PowerDef {
    pub id: &'static str,
    pub name: &'static str,
    pub target: PowerTargetKind,
    pub range: f64,
    pub cone: Option<f64>,
}

pub const POWERS: [PowerDef; 14] = [
    PowerDef {
        id: "tornado",
        name: "TORNADO",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "freeze",
        name: "FREEZE",
        target: PowerTargetKind::Ball,
        range: 80.0,
        cone: None,
    },
    PowerDef {
        id: "spikes",
        name: "SPIKES",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "magnet",
        name: "MAGNET",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "grapple",
        name: "GRAPPLE",
        target: PowerTargetKind::Ball,
        range: 55.0,
        cone: None,
    },
    PowerDef {
        id: "tether",
        name: "TETHER",
        target: PowerTargetKind::Ball,
        range: 48.0,
        cone: None,
    },
    PowerDef {
        id: "punch",
        name: "PUNCH",
        target: PowerTargetKind::Ball,
        range: 30.0,
        cone: None,
    },
    PowerDef {
        id: "boot",
        name: "BOOT",
        target: PowerTargetKind::Enemy,
        range: 42.0,
        cone: None,
    },
    PowerDef {
        id: "shock",
        name: "SHOCKWAVE",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "overdrive",
        name: "OVERDRIVE",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "scramble",
        name: "SCRAMBLE",
        target: PowerTargetKind::Enemy,
        range: 48.0,
        cone: None,
    },
    PowerDef {
        id: "swap",
        name: "SWAP",
        target: PowerTargetKind::Enemy,
        range: 60.0,
        cone: None,
    },
    PowerDef {
        id: "antigrav",
        name: "ZERO-G",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
    PowerDef {
        id: "nitro",
        name: "NITRO",
        target: PowerTargetKind::None,
        range: 0.0,
        cone: None,
    },
];

pub fn power_by_id(id: &str) -> Option<&'static PowerDef> {
    POWERS.iter().find(|p| p.id == id)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerTarget {
    SelfOnly,
    Ball,
    Enemy(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Tornado {
    pub pos: DVec3,
    pub t: f64,
    pub life: f64,
    pub owner: usize,
    pub id: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShotTarget {
    Ball,
    Car(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShotPhase {
    Fly,
    Pull,
}

impl ShotPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            ShotPhase::Fly => "fly",
            ShotPhase::Pull => "pull",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PowerShot {
    pub kind: &'static str,
    pub from: usize,
    pub to: ShotTarget,
    pub pos: DVec3,
    pub t: f64,
    pub phase: ShotPhase,
    pub pt: f64,
}

#[derive(Clone, Debug)]
pub enum PowerEvent {
    Get {
        car_idx: usize,
        id: &'static str,
    },
    Fail {
        car_idx: usize,
        id: &'static str,
    },
    Use {
        car_idx: usize,
        id: &'static str,
        target: Option<ShotTarget>,
    },
    Freeze {
        car_idx: usize,
    },
    Release {
        car_idx: usize,
        launch: bool,
        pos: DVec3,
    },
    Shatter {
        car_idx: usize,
        pos: DVec3,
    },
    Thaw {
        pos: DVec3,
    },
    TornadoEnd {
        pos: DVec3,
    },
    OvHit {
        car_idx: usize,
        pos: DVec3,
    },
    Attach {
        car_idx: usize,
        pos: DVec3,
    },
    Shock {
        car_idx: usize,
        pos: DVec3,
    },
    Swap {
        car_idx: usize,
        target_idx: usize,
        from: DVec3,
        to: DVec3,
    },
    Scrambled {
        car_idx: usize,
        by_idx: usize,
    },
    PunchHit {
        car_idx: usize,
        pos: DVec3,
        dir: DVec3,
    },
    BootHit {
        car_idx: usize,
        target_idx: usize,
        pos: DVec3,
    },
    Hook {
        car_idx: usize,
        hook_kind: &'static str,
        pos: DVec3,
    },
}

#[derive(Clone, Debug)]
pub struct PowerSystem {
    pub interval: f64,
    pub pool: Vec<&'static str>,
    pub tornados: Vec<Tornado>,
    pub shots: Vec<PowerShot>,
    pub enabled: bool,
}

impl Default for PowerSystem {
    fn default() -> Self {
        Self::new(10.0)
    }
}

impl PowerSystem {
    pub fn new(interval: f64) -> Self {
        Self {
            interval: if interval > 0.0 { interval } else { 10.0 },
            pool: POWERS.iter().map(|p| p.id).collect(),
            tornados: Vec::new(),
            shots: Vec::new(),
            enabled: true,
        }
    }

    pub fn ensure<R: Rng + ?Sized>(c: &mut Car, rng: &mut R) {
        if c.pw_init {
            return;
        }
        c.pw_init = true;
        c.item = None;
        c.item_next = rand_range(rng, 1.5, 3.5);
        c.prev_item = false;
        c.spike_t = 0.0;
        c.mag_t = 0.0;
        c.ov_t = 0.0;
        c.nitro_t = 0.0;
        c.scr_t = 0.0;
        c.torn_t = 0.0;
        c.speed_mul = 1.0;
        c.hook_t = 0.0;
    }

    pub fn clear_effects(&mut self, w: &mut World) {
        w.ball.stuck_to = None;
        w.ball.ice_t = 0.0;
        w.ball.g_scale = 1.0;
        w.ball.anti_t = 0.0;
        self.tornados.clear();
        self.shots.clear();
        for c in &mut w.cars {
            Self::ensure(c, &mut w.rng);
            c.spike_t = 0.0;
            c.mag_t = 0.0;
            c.ov_t = 0.0;
            c.nitro_t = 0.0;
            c.scr_t = 0.0;
            c.torn_t = 0.0;
            c.hook_t = 0.0;
            c.speed_mul = 1.0;
        }
    }

    pub fn reset(&mut self, w: &mut World) {
        self.clear_effects(w);
        for c in &mut w.cars {
            c.item = None;
            c.item_next = rand_range(&mut w.rng, 1.2, 3.2);
            c.prev_item = false;
        }
    }

    pub fn pick_item<R: Rng + ?Sized>(&self, c: &mut Car, rng: &mut R) -> &'static str {
        let mut id = *pick(&self.pool, rng);
        if c.last_item == Some(id) && rng.r#gen::<f64>() < 0.7 {
            id = *pick(&self.pool, rng);
        }
        c.last_item = Some(id);
        id
    }

    pub fn target_for(&self, car_idx: usize, id: &str, cars: &[Car], ball: &Ball) -> Option<PowerTarget> {
        let p = power_by_id(id)?;
        let c = &cars[car_idx];
        match p.target {
            PowerTargetKind::Ball => {
                if ball.frozen {
                    return None;
                }
                let to = ball.pos - c.pos;
                let d = to.length();
                if d > p.range || ball.stuck_to == Some(car_idx) {
                    return None;
                }
                if let Some(cone) = p.cone
                    && to.dot(c.basis.f) / d.max(1e-3) < cone
                {
                    return None;
                }
                Some(PowerTarget::Ball)
            }
            PowerTargetKind::Enemy => {
                let mut best = None;
                let mut bs = 1e9_f64;
                for (i, o) in cars.iter().enumerate() {
                    if o.team == c.team || o.demolished {
                        continue;
                    }
                    let to = o.pos - c.pos;
                    let d = to.length();
                    if d > p.range {
                        continue;
                    }
                    let sc = d * (2.2 - to.dot(c.basis.f) / d.max(1e-3));
                    if sc < bs {
                        bs = sc;
                        best = Some(i);
                    }
                }
                best.map(PowerTarget::Enemy)
            }
            PowerTargetKind::None => Some(PowerTarget::SelfOnly),
        }
    }

    fn shot_start(c: &Car) -> DVec3 {
        c.pos + c.basis.f * 0.9 + c.basis.u * 0.25
    }

    pub fn use_power(&mut self, car_idx: usize, id: &'static str, w: &mut World) -> bool {
        let Some(tgt) = self.target_for(car_idx, id, &w.cars, &w.ball) else {
            return false;
        };
        let target_ev = match tgt {
            PowerTarget::SelfOnly => None,
            PowerTarget::Ball => Some(ShotTarget::Ball),
            PowerTarget::Enemy(i) => Some(ShotTarget::Car(i)),
        };

        match id {
            "tornado" => {
                self.tornados.retain(|t| t.owner != car_idx);
                let c = &mut w.cars[car_idx];
                let pos = DVec3::new(c.pos.x, c.pos.y + c.half.y, c.pos.z);
                c.torn_t = 6.5;
                self.tornados.push(Tornado {
                    pos,
                    t: 0.0,
                    life: 6.5,
                    owner: car_idx,
                    id: w.rng.r#gen::<f64>(),
                });
            }
            "freeze" => {
                if w.ball.stuck_to.is_some() {
                    self.release(false, w);
                }
                w.ball.ice_t = 3.2;
                w.ball.vel = DVec3::ZERO;
                w.events.push(WorldEvent::Power(PowerEvent::Freeze { car_idx }));
            }
            "spikes" => {
                w.cars[car_idx].spike_t = 7.5;
            }
            "magnet" => {
                w.cars[car_idx].mag_t = 6.0;
            }
            "grapple" | "tether" | "punch" | "boot" => {
                let to = match tgt {
                    PowerTarget::Ball => ShotTarget::Ball,
                    PowerTarget::Enemy(i) => ShotTarget::Car(i),
                    PowerTarget::SelfOnly => return false,
                };
                let start = Self::shot_start(&w.cars[car_idx]);
                self.shots.push(PowerShot {
                    kind: id,
                    from: car_idx,
                    to,
                    pos: start,
                    t: 0.0,
                    phase: ShotPhase::Fly,
                    pt: 0.0,
                });
            }
            "shock" => {
                self.shockwave(car_idx, w);
            }
            "overdrive" => {
                w.cars[car_idx].ov_t = 8.0;
            }
            "scramble" => {
                if let PowerTarget::Enemy(ti) = tgt {
                    w.cars[ti].scr_t = 4.5;
                    w.cars[ti].spike_t = 0.0;
                    if w.ball.stuck_to == Some(ti) {
                        self.release(false, w);
                    }
                    w.events.push(WorldEvent::Power(PowerEvent::Scrambled {
                        car_idx: ti,
                        by_idx: car_idx,
                    }));
                }
            }
            "swap" => {
                if let PowerTarget::Enemy(ti) = tgt {
                    self.shots
                        .retain(|s| !(s.kind == "swap" && s.from == car_idx));
                    let tpos = w.cars[ti].pos;
                    self.shots.push(PowerShot {
                        kind: "swap",
                        from: car_idx,
                        to: ShotTarget::Car(ti),
                        pos: tpos,
                        t: 0.0,
                        phase: ShotPhase::Pull,
                        pt: 0.0,
                    });
                }
            }
            "antigrav" => {
                w.ball.anti_t = 6.0;
            }
            "nitro" => {
                w.cars[car_idx].nitro_t = 5.0;
                w.cars[car_idx].boost = 100.0;
            }
            _ => return false,
        }

        w.cars[car_idx].stats.powers += 1;
        w.events.push(WorldEvent::Power(PowerEvent::Use {
            car_idx,
            id,
            target: target_ev,
        }));
        true
    }

    pub fn release(&mut self, launch: bool, w: &mut World) {
        let Some(c_idx) = w.ball.stuck_to else {
            return;
        };
        w.ball.stuck_to = None;
        if launch {
            let c = &w.cars[c_idx];
            w.ball.vel = c.vel + c.basis.f * 17.0 + c.basis.u * 6.5;
            let sp = w.ball.vel.length();
            if sp > w.phys.ball_max {
                w.ball.vel *= w.phys.ball_max / sp;
            }
            w.register_touch(c_idx);
        }
        w.cars[c_idx].spike_t = if launch {
            0.0
        } else {
            w.cars[c_idx].spike_t.min(0.6)
        };
        w.events.push(WorldEvent::Power(PowerEvent::Release {
            car_idx: c_idx,
            launch,
            pos: w.ball.pos,
        }));
    }

    pub fn shatter(&mut self, by_idx: usize, w: &mut World) {
        if w.ball.ice_t <= 0.0 {
            return;
        }
        w.ball.ice_t = 0.0;
        w.events.push(WorldEvent::Power(PowerEvent::Shatter {
            car_idx: by_idx,
            pos: w.ball.pos,
        }));
    }

    pub fn on_demolish(&mut self, car_idx: usize, w: &mut World) {
        if w.ball.stuck_to == Some(car_idx) {
            self.release(false, w);
        }
        let c = &mut w.cars[car_idx];
        c.spike_t = 0.0;
        c.mag_t = 0.0;
        c.ov_t = 0.0;
        c.nitro_t = 0.0;
        c.scr_t = 0.0;
        c.torn_t = 0.0;
        c.hook_t = 0.0;
        c.speed_mul = 1.0;

        let mut i = self.tornados.len();
        while i > 0 {
            i -= 1;
            if self.tornados[i].owner == car_idx {
                let t = self.tornados.remove(i);
                w.events
                    .push(WorldEvent::Power(PowerEvent::TornadoEnd { pos: t.pos }));
            }
        }
    }

    pub fn on_ball_contact(&mut self, car_idx: usize, hit: &CarBallHit, iced: bool, w: &mut World) {
        if iced {
            if hit.strength > 6.5 || w.cars[car_idx].ov_t > 0.0 {
                self.shatter(car_idx, w);
            } else {
                return;
            }
        }
        if let Some(st) = w.ball.stuck_to
            && st != car_idx
        {
            self.release(false, w);
        }
        if w.cars[car_idx].ov_t > 0.0 {
            let d = (w.ball.pos - w.cars[car_idx].pos).three_normalize();
            w.ball.vel += d * (9.0 + hit.strength * 0.35);
            let sp = w.ball.vel.length();
            if sp > w.phys.ball_max {
                w.ball.vel *= w.phys.ball_max / sp;
            }
            w.events.push(WorldEvent::Power(PowerEvent::OvHit {
                car_idx,
                pos: w.ball.pos,
            }));
        }
        if w.cars[car_idx].spike_t > 0.0 && w.ball.stuck_to.is_none() && w.ball.ice_t <= 0.0 {
            w.ball.stuck_to = Some(car_idx);
            let c = &mut w.cars[car_idx];
            c.basis = Basis::from_quat(c.quat);
            let mut local = c.basis.to_local(w.ball.pos - c.pos);
            let min_len = c.half.x + 0.75;
            if local.length() < min_len {
                local = local.with_length(min_len);
            }
            w.ball.stuck_local = local;
            w.events.push(WorldEvent::Power(PowerEvent::Attach {
                car_idx,
                pos: w.ball.pos,
            }));
        }
    }

    pub fn carry_ball(&mut self, dt: f64, w: &mut World) {
        let Some(c_idx) = w.ball.stuck_to else {
            return;
        };
        if w.cars[c_idx].demolished || w.cars[c_idx].spike_t <= 0.0 {
            self.release(false, w);
            w.ball.step(dt, &w.shape, &w.phys);
            return;
        }
        w.cars[c_idx].basis = Basis::from_quat(w.cars[c_idx].quat);
        let c = &w.cars[c_idx];
        let off = c.basis.to_world(w.ball.stuck_local);
        w.ball.pos = c.pos + off;
        w.ball.vel = c.vel + c.ang_vel.cross(off);
        w.ball.ang_vel = c.ang_vel;
        w.ball.quat = w.ball.quat.integrate(w.ball.ang_vel, dt);
        w.ball.on_ground = false;
        let d = w.shape.dist(w.ball.pos.x, w.ball.pos.y, w.ball.pos.z);
        if d < w.ball.r {
            let n = w.shape.normal(w.ball.pos.x, w.ball.pos.y, w.ball.pos.z);
            w.ball.pos += n * (w.ball.r - d);
            if n.y > 0.7 {
                w.ball.on_ground = true;
            }
            if w.ball.r - d > 0.35 {
                self.release(false, w);
            }
        }
    }

    fn shockwave(&mut self, car_idx: usize, w: &mut World) {
        let r = 16.0;
        let c_pos = w.cars[car_idx].pos;
        let c_team = w.cars[car_idx].team;
        let mut release_ball = false;
        for (i, o) in w.cars.iter_mut().enumerate() {
            if o.team == c_team || o.demolished {
                continue;
            }
            let mut to = o.pos - c_pos;
            let d = to.length();
            if d > r || d < 1e-3 {
                continue;
            }
            let k = 1.0 - d / r;
            to /= d;
            to.y = to.y.max(0.0) + 0.5;
            to = to.three_normalize();
            o.vel += to * (9.0 + 19.0 * k);
            o.ang_vel += DVec3::new(
                rand_range(&mut w.rng, -1.0, 1.0),
                rand_range(&mut w.rng, -1.0, 1.0),
                rand_range(&mut w.rng, -1.0, 1.0),
            ) * (5.0 * k);
            if w.ball.stuck_to == Some(i) {
                release_ball = true;
            }
        }
        if release_ball {
            self.release(false, w);
        }
        let mut to = w.ball.pos - c_pos;
        let d = to.length();
        if d < r && !w.ball.frozen && w.ball.stuck_to != Some(car_idx) {
            if w.ball.ice_t > 0.0 {
                self.shatter(car_idx, w);
            }
            if w.ball.stuck_to.is_some() {
                self.release(false, w);
            }
            let k = 1.0 - d / r;
            to /= d.max(1e-3);
            to.y = to.y.max(0.0) + 0.35;
            to = to.three_normalize();
            w.ball.vel += to * (10.0 + 26.0 * k);
            w.register_touch(car_idx);
        }
        w.events.push(WorldEvent::Power(PowerEvent::Shock {
            car_idx,
            pos: c_pos,
        }));
    }

    fn swap_cars(&mut self, a_idx: usize, o_idx: usize, w: &mut World) {
        if w.ball.stuck_to == Some(a_idx) || w.ball.stuck_to == Some(o_idx) {
            self.release(false, w);
        }
        let from = w.cars[a_idx].pos;
        let to = w.cars[o_idx].pos;
        let (a_pos, a_vel, a_ang, a_q) = (
            w.cars[a_idx].pos,
            w.cars[a_idx].vel,
            w.cars[a_idx].ang_vel,
            w.cars[a_idx].quat,
        );
        let (o_pos, o_vel, o_ang, o_q) = (
            w.cars[o_idx].pos,
            w.cars[o_idx].vel,
            w.cars[o_idx].ang_vel,
            w.cars[o_idx].quat,
        );
        {
            let a = &mut w.cars[a_idx];
            a.pos = o_pos;
            a.vel = o_vel;
            a.ang_vel = o_ang;
            a.quat = o_q;
            a.save_prev();
            a.basis = Basis::from_quat(a.quat);
        }
        {
            let o = &mut w.cars[o_idx];
            o.pos = a_pos;
            o.vel = a_vel;
            o.ang_vel = a_ang;
            o.quat = a_q;
            o.save_prev();
            o.basis = Basis::from_quat(o.quat);
        }
        w.events.push(WorldEvent::Power(PowerEvent::Swap {
            car_idx: a_idx,
            target_idx: o_idx,
            from,
            to,
        }));
    }

    fn affect_body(
        pos: DVec3,
        vel: &mut DVec3,
        ang_vel: Option<(&mut DVec3, usize)>,
        t: &Tornado,
        owner_vel: DVec3,
        g: f64,
        dt: f64,
    ) -> f64 {
        let rx = pos.x - t.pos.x;
        let rz = pos.z - t.pos.z;
        let d = rx.hypot(rz);
        let r = 8.5;
        if d > r || pos.y < t.pos.y - 2.5 || pos.y > t.pos.y + 18.0 {
            return 0.0;
        }
        let k = (1.0 - d / r).sqrt() * (t.t * 3.0).min(1.0) * ((t.life - t.t) * 2.0).min(1.0);
        let id = 1.0 / d.max(0.3);
        vel.x += (-rz * id * 24.0 - rx * id * 14.0) * k * dt;
        vel.z += (rx * id * 24.0 - rz * id * 14.0) * k * dt;
        vel.x += (owner_vel.x - vel.x) * 2.2 * k * dt;
        vel.z += (owner_vel.z - vel.z) * 2.2 * k * dt;
        vel.y += (g + 9.5) * k * dt;
        if vel.y > 13.5 {
            vel.y = 13.5;
        }
        if let Some((av, car_id)) = ang_vel {
            av.y += 6.0 * k * dt;
            av.x += (t.t * 7.0 + car_id as f64).sin() * 4.0 * k * dt;
        }
        k
    }

    pub fn pre_cars(&mut self, dt: f64, w: &mut World) {
        if w.ball.ice_t > 0.0 {
            w.ball.ice_t -= dt;
            if w.ball.ice_t <= 0.0 {
                w.ball.ice_t = 0.0;
                w.events
                    .push(WorldEvent::Power(PowerEvent::Thaw { pos: w.ball.pos }));
            }
        }
        if w.ball.anti_t > 0.0 {
            w.ball.anti_t -= dt;
        }
        w.ball.g_scale = if w.ball.anti_t > 0.0 { 0.2 } else { 1.0 };

        let n_cars = w.cars.len();
        for ci in 0..n_cars {
            Self::ensure(&mut w.cars[ci], &mut w.rng);
            if w.cars[ci].demolished {
                w.cars[ci].prev_item = w.cars[ci].controls.item;
                continue;
            }
            {
                let c = &mut w.cars[ci];
                if c.hook_t > 0.0 {
                    c.hook_t = (c.hook_t - dt).max(0.0);
                }
                if c.spike_t > 0.0 {
                    c.spike_t -= dt;
                }
                if c.mag_t > 0.0 {
                    c.mag_t -= dt;
                }
                if c.ov_t > 0.0 {
                    c.ov_t -= dt;
                }
                if c.nitro_t > 0.0 {
                    c.nitro_t -= dt;
                }
                if c.scr_t > 0.0 {
                    c.scr_t -= dt;
                }
            }
            if w.cars[ci].item.is_none() && self.enabled && !w.kickoff_freeze {
                w.cars[ci].item_next -= dt;
                if w.cars[ci].item_next <= 0.0 {
                    let id = self.pick_item(&mut w.cars[ci], &mut w.rng);
                    w.cars[ci].item = Some(id);
                    w.events
                        .push(WorldEvent::Power(PowerEvent::Get { car_idx: ci, id }));
                }
            }
            let pressed = w.cars[ci].controls.item && !w.cars[ci].prev_item;
            w.cars[ci].prev_item = w.cars[ci].controls.item;
            if pressed {
                if w.ball.stuck_to == Some(ci) {
                    self.release(true, w);
                } else if let Some(id) = w.cars[ci].item {
                    if self.use_power(ci, id, w) {
                        w.cars[ci].item = None;
                        w.cars[ci].item_next = self.interval * rand_range(&mut w.rng, 0.85, 1.15);
                    } else {
                        w.events
                            .push(WorldEvent::Power(PowerEvent::Fail { car_idx: ci, id }));
                    }
                }
            }
            if w.cars[ci].scr_t > 0.0 {
                let jump_flip = w.rng.r#gen::<f64>() < dt * 0.9;
                let time = w.time;
                let c = &mut w.cars[ci];
                let k = &mut c.controls;
                k.steer = -k.steer;
                k.yaw = -k.yaw;
                k.roll = -k.roll;
                k.boost = false;
                k.throttle = clamp(
                    k.throttle * 0.5 + (time * 2.3 + c.id as f64).sin() * 0.6,
                    -1.0,
                    1.0,
                );
                if jump_flip {
                    k.jump = !k.jump;
                }
            }
            {
                let c = &mut w.cars[ci];
                c.speed_mul = if c.nitro_t > 0.0 { 1.22 } else { 1.0 };
                if c.nitro_t > 0.0 {
                    c.boost = 100.0;
                }
            }
            if w.cars[ci].mag_t > 0.0
                && w.ball.stuck_to.is_none()
                && w.ball.ice_t <= 0.0
                && !w.ball.frozen
            {
                let c = &w.cars[ci];
                let tgt = c.pos + c.basis.f * 1.5 + c.basis.u * 0.5;
                let to = tgt - w.ball.pos;
                let d = to.length();
                if d < 34.0 && d > 0.6 {
                    let k = 1.0 - d / 38.0;
                    w.ball.vel += to * (17.0 * k / d * dt);
                    if d < 6.0 {
                        w.ball.vel = w.ball.vel.lerp(c.vel, (dt * 2.2).min(1.0));
                    }
                }
            }
        }

        // Tornados
        let mut i = self.tornados.len();
        while i > 0 {
            i -= 1;
            self.tornados[i].t += dt;
            let t_copy = self.tornados[i];
            let owner_ok = t_copy.owner < w.cars.len() && !w.cars[t_copy.owner].demolished;
            if t_copy.t > t_copy.life || !owner_ok {
                if t_copy.owner < w.cars.len() {
                    w.cars[t_copy.owner].torn_t = 0.0;
                }
                self.tornados.remove(i);
                w.events.push(WorldEvent::Power(PowerEvent::TornadoEnd {
                    pos: t_copy.pos,
                }));
                continue;
            }
            let owner = &w.cars[t_copy.owner];
            let owner_vel = owner.vel;
            self.tornados[i].pos = DVec3::new(owner.pos.x, owner.pos.y + owner.half.y, owner.pos.z);
            let t_cur = self.tornados[i];
            w.cars[t_cur.owner].torn_t = (t_cur.life - t_cur.t).max(0.0);
            let g = w.phys.g;
            for (ci, c) in w.cars.iter_mut().enumerate() {
                if ci != t_cur.owner && !c.demolished {
                    let id = c.id;
                    Self::affect_body(
                        c.pos,
                        &mut c.vel,
                        Some((&mut c.ang_vel, id)),
                        &t_cur,
                        owner_vel,
                        g,
                        dt,
                    );
                }
            }
            if !w.ball.frozen && w.ball.ice_t <= 0.0 {
                let k = Self::affect_body(
                    w.ball.pos,
                    &mut w.ball.vel,
                    None,
                    &t_cur,
                    owner_vel,
                    g,
                    dt,
                );
                if k > 0.35 {
                    w.register_touch(t_cur.owner);
                }
                if k > 0.45
                    && let Some(st) = w.ball.stuck_to
                    && st != t_cur.owner
                {
                    self.release(false, w);
                }
            }
        }

        // Shots
        let mut i = self.shots.len();
        while i > 0 {
            i -= 1;
            self.shots[i].t += dt;
            let s = self.shots[i];
            let from_demolished = w.cars[s.from].demolished;
            let tgt_invalid = match s.to {
                ShotTarget::Ball => w.ball.frozen,
                ShotTarget::Car(ti) => w.cars[ti].demolished,
            };
            if from_demolished || tgt_invalid || s.t > 3.0 {
                self.shots.remove(i);
                continue;
            }
            if s.kind == "swap" {
                self.shots[i].pt += dt;
                if let ShotTarget::Car(ti) = s.to {
                    self.shots[i].pos = w.cars[ti].pos;
                    if self.shots[i].pt >= SWAP_CHARGE {
                        self.shots.remove(i);
                        self.swap_cars(s.from, ti, w);
                    }
                }
                continue;
            }
            let tgt_pos = match s.to {
                ShotTarget::Ball => w.ball.pos,
                ShotTarget::Car(ti) => w.cars[ti].pos,
            };
            if s.phase == ShotPhase::Fly {
                let sp = match s.kind {
                    "punch" => 115.0,
                    "boot" => 95.0,
                    _ => 80.0,
                };
                let to = tgt_pos - s.pos;
                let d = to.length();
                let hit_r = match s.to {
                    ShotTarget::Ball => w.ball.r,
                    ShotTarget::Car(_) => 0.9,
                };
                if d <= sp * dt + hit_r {
                    self.shots[i].pos = tgt_pos;
                    self.arrive(i, w);
                } else {
                    self.shots[i].pos += to * (sp * dt / d);
                }
            } else {
                self.shots[i].pt += dt;
                self.shots[i].pos = tgt_pos;
                let pt = self.shots[i].pt;
                if s.kind == "grapple" {
                    let to = w.ball.pos - w.cars[s.from].pos;
                    let d = to.length();
                    if d < 2.4 || pt > 1.5 || w.ball.stuck_to.is_some() {
                        self.shots.remove(i);
                        let c = &mut w.cars[s.from];
                        c.speed_mul = if c.nitro_t > 0.0 { 1.22 } else { 1.0 };
                        c.hook_t = 0.0;
                        continue;
                    }
                    let g = w.phys.g;
                    let c = &mut w.cars[s.from];
                    c.speed_mul = 1.35;
                    c.hook_t = 0.1;
                    c.vel += to * (52.0 * dt / d);
                    c.vel.y += g * 0.8 * dt;
                } else {
                    let c = &w.cars[s.from];
                    let tp = c.pos + c.basis.f * 2.2;
                    let to = tp - w.ball.pos;
                    let d = to.length();
                    if d < 3.2 || pt > 1.7 || w.ball.stuck_to.is_some() || w.ball.ice_t > 0.0 {
                        self.shots.remove(i);
                        continue;
                    }
                    w.ball.vel += to * (46.0 * dt / d);
                    w.ball.vel *= 1.0 - 1.1 * dt;
                    w.ball.vel.y += w.phys.g * 0.5 * dt;
                }
            }
        }
    }

    fn arrive(&mut self, idx: usize, w: &mut World) {
        let s = self.shots[idx];
        let c_idx = s.from;
        if s.kind == "punch" {
            self.shots.remove(idx);
            if w.ball.ice_t > 0.0 {
                self.shatter(c_idx, w);
            }
            if w.ball.stuck_to.is_some() {
                self.release(false, w);
            }
            let mut dir = w.ball.pos - w.cars[c_idx].pos;
            dir.y = dir.y.max(0.0);
            dir = dir.three_normalize();
            dir.y += 0.28;
            dir = dir.three_normalize();
            w.ball.vel = w.ball.vel * 0.2 + dir * 34.0;
            let sp = w.ball.vel.length();
            if sp > w.phys.ball_max {
                w.ball.vel *= w.phys.ball_max / sp;
            }
            w.register_touch(c_idx);
            w.events.push(WorldEvent::Power(PowerEvent::PunchHit {
                car_idx: c_idx,
                pos: w.ball.pos,
                dir,
            }));
        } else if s.kind == "boot" {
            self.shots.remove(idx);
            if let ShotTarget::Car(v_idx) = s.to {
                let mut dir = w.cars[v_idx].pos - w.cars[c_idx].pos;
                dir.y = 0.0;
                if dir.length_squared() < 1e-4 {
                    dir = w.cars[c_idx].basis.f;
                }
                dir = dir.three_normalize();
                let ang_imp = DVec3::new(
                    rand_range(&mut w.rng, -1.0, 1.0),
                    rand_range(&mut w.rng, -1.0, 1.0),
                    rand_range(&mut w.rng, -1.0, 1.0),
                ) * 6.0;
                let v = &mut w.cars[v_idx];
                v.vel += dir * 17.0;
                v.vel.y += 13.0;
                v.ang_vel += ang_imp;
                let v_pos = v.pos;
                if w.ball.stuck_to == Some(v_idx) {
                    self.release(false, w);
                }
                w.events.push(WorldEvent::Power(PowerEvent::BootHit {
                    car_idx: c_idx,
                    target_idx: v_idx,
                    pos: v_pos,
                }));
            }
        } else {
            if w.ball.stuck_to.is_some() {
                self.release(false, w);
            }
            if w.ball.ice_t > 0.0 && s.kind == "tether" {
                self.shatter(c_idx, w);
            }
            self.shots[idx].phase = ShotPhase::Pull;
            self.shots[idx].pt = 0.0;
            w.events.push(WorldEvent::Power(PowerEvent::Hook {
                car_idx: c_idx,
                hook_kind: s.kind,
                pos: w.ball.pos,
            }));
        }
    }

    pub fn post_step(&mut self, _dt: f64) {}
}
