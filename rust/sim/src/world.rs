use glam::DVec3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::ai::TeamBrain;
use crate::arena::{ArenaDef, ArenaOpts, ArenaShape, GoalMode, Pad, make_boost_pads};
use crate::math_ext::{PI, PhysConfig, ThreeQuatExt};
use crate::physics::{
    Ball, BallPredictor, BounceEvent, Car, CarBallHit, CarCollisionResult, CarEventKind,
    TouchRecord, collide_car_ball, collide_cars,
};
use crate::powers::{PowerEvent, PowerSystem};

#[derive(Clone, Debug)]
pub enum WorldEvent {
    Bounce(BounceEvent),
    Car {
        kind: CarEventKind,
        car_idx: usize,
    },
    Hit(CarBallHit),
    Bump {
        attacker_idx: usize,
        victim_idx: usize,
        strength: f64,
    },
    Demo {
        victim_idx: usize,
        attacker_idx: usize,
        pos: DVec3,
        overdrive: bool,
    },
    Pickup {
        car_idx: usize,
        pad_idx: usize,
    },
    Respawn {
        car_idx: usize,
    },
    Power(PowerEvent),
}

#[derive(Clone, Copy, Debug)]
pub struct Threat {
    pub team: i32,
    pub time: f64,
    /// World time the prediction was made (to age `time` when it is read later)
    pub at: f64,
}

pub struct World {
    pub def: ArenaDef,
    pub shape: ArenaShape,
    pub phys: PhysConfig,
    pub ball: Ball,
    pub cars: Vec<Car>,
    pub pads: Vec<Pad>,
    pub pred: BallPredictor,
    pub brain: TeamBrain,
    pub tick: u64,
    pub time: f64,
    pub kickoff: bool,
    pub kickoff_freeze: bool,
    pub frozen: bool,
    pub allow_demos: bool,
    pub unlimited_boost: bool,
    pub boost_recharge: bool,
    pub is_online_client: bool,
    pub events: Vec<WorldEvent>,
    pub last_threat: Threat,
    pub powers: Option<PowerSystem>,
    pub rng: StdRng,
}

impl World {
    pub fn new(def: ArenaDef, opts: &ArenaOpts) -> Self {
        Self::with_phys_and_rng(def, opts, PhysConfig::default(), StdRng::from_entropy())
    }

    pub fn with_seed(def: ArenaDef, opts: &ArenaOpts, seed: u64) -> Self {
        Self::with_phys_and_rng(def, opts, PhysConfig::default(), StdRng::seed_from_u64(seed))
    }

    pub fn with_phys_and_rng(
        def: ArenaDef,
        opts: &ArenaOpts,
        phys: PhysConfig,
        rng: StdRng,
    ) -> Self {
        let shape = ArenaShape::new(&def, opts);
        let pads = make_boost_pads(&shape);
        let ball = Ball::new(&phys);
        Self {
            def,
            shape,
            phys,
            ball,
            cars: Vec::new(),
            pads,
            pred: BallPredictor::new(),
            brain: TeamBrain::new(),
            tick: 0,
            time: 0.0,
            kickoff: false,
            kickoff_freeze: false,
            frozen: false,
            allow_demos: true,
            unlimited_boost: false,
            boost_recharge: false,
            is_online_client: false,
            events: Vec::new(),
            last_threat: Threat {
                team: -1,
                time: 99.0,
                at: 0.0,
            },
            powers: None,
            rng,
        }
    }

    #[inline]
    pub fn arena(&self) -> &ArenaShape {
        &self.shape
    }

    pub fn add_car(&mut self, car: Car) -> usize {
        let idx = self.cars.len();
        self.cars.push(car);
        idx
    }

    pub fn step(&mut self, dt: f64) {
        self.events.clear();
        self.ball.save_prev();
        for c in &mut self.cars {
            c.save_prev();
        }

        if !self.is_online_client {
            // Refresh first so last_threat is the newest pre-touch prediction (at most 4 ticks old)
            if self.tick.is_multiple_of(4) {
                self.pred.update(&self.ball, &self.shape, &self.phys);
                self.last_threat = Threat {
                    team: self.pred.goal_team,
                    time: self.pred.goal_time,
                    at: self.time,
                };
            }
            self.brain
                .update(dt, &self.cars, &self.ball, self.shape.goal_line_z());
            for i in 0..self.cars.len() {
                if let Some(mut bot) = self.cars[i].bot.take() {
                    bot.update(i, dt, self);
                    self.cars[i].bot = Some(bot);
                }
            }
        }

        if !self.frozen
            && let Some(mut powers) = self.powers.take()
        {
            powers.pre_cars(dt, self);
            self.powers = Some(powers);
        }

        if !self.frozen {
            for i in 0..self.cars.len() {
                if self.cars[i].is_remote {
                    continue;
                }
                if self.cars[i].demolished {
                    self.cars[i].respawn_t -= dt;
                    if self.cars[i].respawn_t <= 0.0 {
                        self.respawn(i);
                    }
                    continue;
                }
                {
                    let c = &mut self.cars[i];
                    c.unlimited_boost = self.unlimited_boost || c.unlimited_boost_always;
                    if self.boost_recharge && !c.is_boosting && c.boost < 100.0 {
                        c.boost = (c.boost + 14.0 * dt).min(100.0);
                    }
                    c.step(dt, &self.shape, &self.phys);
                }
                let evs: Vec<CarEventKind> = self.cars[i].events.clone();
                for kind in evs {
                    self.events.push(WorldEvent::Car { kind, car_idx: i });
                }
            }

            if self.powers.is_some() && self.ball.stuck_to.is_some() {
                if let Some(mut powers) = self.powers.take() {
                    powers.carry_ball(dt, self);
                    self.powers = Some(powers);
                }
            } else if self.ball.ice_t > 0.0 {
                self.ball.vel = DVec3::ZERO;
                self.ball.ang_vel *= 0.97;
                self.ball.quat = self.ball.quat.integrate(self.ball.ang_vel, dt);
            } else if let Some(b_ev) = self.ball.step(dt, &self.shape, &self.phys) {
                self.events.push(WorldEvent::Bounce(b_ev));
            }

            for i in 0..self.cars.len() {
                if self.ball.stuck_to == Some(i) {
                    continue;
                }
                if self.is_online_client && self.cars[i].is_remote {
                    continue;
                }
                let iced = self.ball.ice_t > 0.0;
                let ice_pos = self.ball.pos;
                let hit = collide_car_ball(
                    &mut self.cars[i],
                    i,
                    &mut self.ball,
                    self.tick as i64,
                    Some(&self.shape),
                    &self.phys,
                );
                if let Some(ref h) = hit
                    && let Some(mut powers) = self.powers.take()
                {
                    powers.on_ball_contact(i, h, iced, self);
                    self.powers = Some(powers);
                }
                if iced && self.ball.ice_t > 0.0 {
                    self.ball.pos = ice_pos;
                    self.ball.vel = DVec3::ZERO;
                }
                if let Some(h) = hit {
                    self.on_touch(h);
                }
            }

            if !self.is_online_client {
                let n = self.cars.len();
                for i in 0..n {
                    for j in (i + 1)..n {
                        if self.cars[i].is_remote && self.cars[j].is_remote {
                            continue;
                        }
                        let (left, right) = self.cars.split_at_mut(j);
                        let res = collide_cars(
                            &mut left[i],
                            i,
                            &mut right[0],
                            j,
                            self.time,
                            self.allow_demos,
                            &self.phys,
                        );
                        match res {
                            Some(CarCollisionResult::Demo {
                                attacker_idx,
                                victim_idx,
                                overdrive,
                            }) => {
                                self.demolish(victim_idx, attacker_idx, overdrive);
                            }
                            Some(CarCollisionResult::DoubleDemo {
                                a_idx,
                                b_idx,
                                overdrive,
                            }) => {
                                self.demolish(a_idx, b_idx, overdrive);
                                self.demolish(b_idx, a_idx, overdrive);
                            }
                            Some(CarCollisionResult::Bump {
                                attacker_idx,
                                victim_idx,
                                strength,
                            }) => {
                                self.events.push(WorldEvent::Bump {
                                    attacker_idx,
                                    victim_idx,
                                    strength,
                                });
                                if self.ball.stuck_to == Some(victim_idx)
                                    && self.cars[attacker_idx].team != self.cars[victim_idx].team
                                    && let Some(mut powers) = self.powers.take()
                                {
                                    powers.release(false, self);
                                    self.powers = Some(powers);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            for ci in 0..self.cars.len() {
                if self.cars[ci].demolished {
                    continue;
                }
                for pi in 0..self.pads.len() {
                    if !self.pads[pi].active || self.cars[ci].boost >= 100.0 {
                        continue;
                    }
                    let dx = self.cars[ci].pos.x - self.pads[pi].pos.x;
                    let dz = self.cars[ci].pos.z - self.pads[pi].pos.z;
                    let r = self.pads[pi].radius;
                    if dx * dx + dz * dz < r * r && self.cars[ci].pos.y < 2.2 {
                        self.cars[ci].boost = (self.cars[ci].boost + self.pads[pi].amount).min(100.0);
                        self.pads[pi].active = false;
                        self.pads[pi].timer = self.pads[pi].respawn;
                        self.events.push(WorldEvent::Pickup {
                            car_idx: ci,
                            pad_idx: pi,
                        });
                    }
                }
            }
        }

        for p in &mut self.pads {
            if !p.active {
                p.timer -= dt;
                if p.timer <= 0.0 {
                    p.active = true;
                }
            }
        }

        if !self.frozen
            && let Some(mut powers) = self.powers.take()
        {
            powers.post_step(dt);
            self.powers = Some(powers);
        }

        self.tick += 1;
        self.time += dt;
    }

    pub fn on_touch(&mut self, mut hit: CarBallHit) {
        let ci = hit.car_idx;
        self.ball.last_touch = Some(ci);
        self.ball.last_hit_time = self.time;
        self.ball.last_hit_strength = hit.strength;
        self.ball.touch_history.push(TouchRecord {
            car_idx: ci,
            time: self.time,
        });
        if self.ball.touch_history.len() > 12 {
            self.ball.touch_history.remove(0);
        }
        self.cars[ci].stats.touches += 1;
        if !self.is_online_client {
            let before = self.last_threat;
            let before_t = before.time - (self.time - before.at);
            self.pred.update(&self.ball, &self.shape, &self.phys);
            let after = Threat {
                team: self.pred.goal_team,
                time: self.pred.goal_time,
                at: self.time,
            };
            let team = self.cars[ci].team as i32;
            let opp = 1 - team;
            // A shot is a touch that puts the ball on target, not one that keeps it there
            hit.shot = after.team == opp && before.team != opp;
            hit.save = before.team == team && after.team != team && before_t < 4.0;
            hit.epic = hit.save && before_t < 0.7;
            self.last_threat = after;
        }
        self.events.push(WorldEvent::Hit(hit));
    }

    pub fn demolish(&mut self, victim_idx: usize, attacker_idx: usize, overdrive: bool) {
        self.cars[victim_idx].demolished = true;
        self.cars[victim_idx].respawn_t = 3.0;
        self.cars[attacker_idx].stats.demos += 1;
        if let Some(mut powers) = self.powers.take() {
            powers.on_demolish(victim_idx, self);
            self.powers = Some(powers);
        }
        let pos = self.cars[victim_idx].pos;
        self.events.push(WorldEvent::Demo {
            victim_idx,
            attacker_idx,
            pos,
            overdrive,
        });
    }

    /// Discrete power-up strikes on the ball (magnet launch, blast, punch): a full touch
    /// so shots/saves are credited. Online clients only record the touch.
    pub fn power_touch(&mut self, car_idx: usize) {
        if self.is_online_client {
            self.register_touch(car_idx);
            return;
        }
        let b = &self.ball;
        let hit = CarBallHit {
            car_idx,
            strength: 0.0,
            x: b.pos.x,
            y: b.pos.y,
            z: b.pos.z,
            ball_speed: b.vel.length(),
            prev_speed: 0.0,
            shot: false,
            save: false,
            epic: false,
        };
        self.on_touch(hit);
    }

    pub fn register_touch(&mut self, car_idx: usize) {
        self.ball.last_touch = Some(car_idx);
        self.ball.last_hit_time = self.time;
        self.ball.touch_history.push(TouchRecord {
            car_idx,
            time: self.time,
        });
        if self.ball.touch_history.len() > 12 {
            self.ball.touch_history.remove(0);
        }
    }

    pub fn respawn(&mut self, car_idx: usize) {
        let ad = if self.cars[car_idx].team == 0 {
            1.0
        } else {
            -1.0
        };
        let rz = if self.shape.goal_mode == GoalMode::Standalone {
            self.shape.goal_z - 6.0
        } else {
            self.shape.l - 8.0
        };
        let side = if self.rng.r#gen::<f64>() < 0.5 {
            -1.0
        } else {
            1.0
        };
        let (vx, vz) = self
            .shape
            .push_inside_xz(side * self.shape.w * 0.45, -ad * rz, 5.0);
        let yaw = if ad > 0.0 { -PI / 2.0 } else { PI / 2.0 };
        let c = &mut self.cars[car_idx];
        c.place(vx, 0.35, vz, yaw);
        c.demolished = false;
        c.boost = 33.0;
        self.events.push(WorldEvent::Respawn { car_idx });
    }

    pub fn reset_pads(&mut self) {
        for p in &mut self.pads {
            p.active = true;
            p.timer = 0.0;
        }
    }
}
