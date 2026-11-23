//! Physics-simulated shot planner used by CHAMP bots.
//!
//! Candidate strike plans are rolled forward with the real car/ball physics (car + ball only)
//! and scored by where the ball ends up; the best plan is then executed live with the exact same
//! closed-loop controller, so what was simulated is what happens (barring other cars).

use glam::DVec3;

use crate::arena::ArenaShape;
use crate::math_ext::{Controls, PhysConfig, ThreeVec3Ext, clamp};
use crate::physics::{Ball, BallState, Car, ball_integrate, collide_car_ball};

#[derive(Clone, Copy, Debug)]
pub struct StrikePlan {
    /// Ball contact point and absolute world time it was planned for.
    pub p: DVec3,
    pub t_hit: f64,
    /// Horizontal unit direction the ball should leave along.
    pub sd: DVec3,
    /// Max distance of the approach point behind the ball.
    pub k: f64,
    /// Arrival pace multiplier (>1 arrives early / hits harder).
    pub pace: f64,
    /// 0 = no dodge, 1 = front dodge into the ball, 2 = jump (for raised balls).
    pub dodge: u8,
    /// Time the dodge/jump sequence started (absolute), if started.
    pub seq_t0: Option<f64>,
    pub score: f64,
}

const DODGE_LEAD: f64 = 0.2;

/// Closed-loop controller for a strike plan. Mutates `plan.seq_t0` when a dodge/jump starts.
pub fn strike_controls(
    plan: &mut StrikePlan,
    car: &Car,
    ball_pos: DVec3,
    now: f64,
    phys: &PhysConfig,
) -> Controls {
    let mut c = Controls::default();
    let b = car.basis;
    let rem = plan.t_hit - now;

    if let Some(t0) = plan.seq_t0 {
        let st = now - t0;
        c.throttle = 1.0;
        let to_b = ball_pos - car.pos;
        let al = b.to_local(to_b);
        if plan.dodge == 1 {
            // jump, release, dodge toward the ball
            if st < 0.05 {
                c.jump = true;
            } else if st < 0.08 {
                c.jump = false;
            } else if st < 0.14 {
                c.jump = true;
                let mag = al.x.hypot(al.z).max(1e-3);
                let fwd = al.x / mag;
                let side = al.z / mag;
                c.pitch = clamp(fwd * 1.6, -1.0, 1.0);
                c.steer = clamp(side * 1.6, -1.0, 1.0);
                c.yaw = c.steer;
            }
        } else {
            // raised ball: (double) jump straight up while aligned
            let dbl = plan.p.y > 3.0;
            c.jump = st < 0.2 || (dbl && st > 0.24 && st < 0.3);
            c.boost = car.boost > 0.0 && b.f.y < 0.5;
        }
        return c;
    }

    let to_p = plan.p - car.pos;
    let d = to_p.x.hypot(to_p.z);
    let off = clamp(d * 0.42, 0.0, plan.k);
    let mut aim = plan.p - plan.sd * off;
    aim.y = car.pos.y;
    let to = aim - car.pos;
    let al = b.to_local(to);
    let ang = al.z.atan2(al.x);
    let yaw_rate = car.ang_vel.dot(b.u);
    c.steer = clamp(ang * 3.4 + yaw_rate * 0.25, -1.0, 1.0);
    let vf = car.vel.dot(b.f);
    if car.on_ground {
        c.handbrake = ang.abs() > 1.25 && vf > 7.0;
    }
    let want = if rem > 0.02 {
        clamp((d - 1.2).max(0.0) / rem * plan.pace, 0.0, 23.0)
    } else {
        23.0
    };
    if want > vf + 0.5 {
        c.throttle = 1.0;
        c.boost = car.boost > 0.0
            && ang.abs() < 0.4
            && car.on_ground
            && (want > vf + 4.0 || want > 15.0)
            && vf < 22.7;
    } else if want < vf - 2.0 {
        c.throttle = if vf > 0.0 { -1.0 } else { 0.0 };
    } else {
        c.throttle = clamp(want / 14.0, 0.05, 1.0);
    }

    if car.on_ground && car.contacts >= 3 {
        let facing = b.f.dot((ball_pos - car.pos).three_normalize());
        match plan.dodge {
            1 => {
                let bd = car.pos.distance(ball_pos);
                if rem < DODGE_LEAD && rem > -0.1 && bd < 4.6 + car.speed() * 0.06 && facing > 0.6 {
                    plan.seq_t0 = Some(now);
                }
            }
            2 => {
                let h = plan.p.y - 0.35;
                let dbl = plan.p.y > 3.0;
                let tj = crate::ai::jump_time_for(h - 0.3, dbl);
                if tj > 0.0 && rem < tj + 0.01 && facing > 0.6 {
                    plan.seq_t0 = Some(now);
                }
            }
            _ => {}
        }
    }
    let _ = phys;
    c
}

pub struct StrikeCtx<'a> {
    pub shape: &'a ArenaShape,
    pub phys: &'a PhysConfig,
    /// +1 if this team attacks +z.
    pub ad: f64,
    pub tick: i64,
}

/// Simulates the plan with the real physics; returns (score, contact happened).
pub fn simulate_strike(
    plan: &StrikePlan,
    car: &Car,
    car_idx: usize,
    ball: &Ball,
    now: f64,
    ctx: &StrikeCtx,
    scratch: &mut Car,
    sball: &mut Ball,
) -> f64 {
    let phys = ctx.phys;
    let dt = phys.dt;
    scratch.clone_from(car);
    scratch.bot = None;
    sball.clone_from(ball);
    let mut p = *plan;
    let horizon = (plan.t_hit - now + 0.4).clamp(0.2, 2.4);
    let steps = (horizon / dt) as usize;
    let mut t = now;
    for i in 0..steps {
        scratch.controls = strike_controls(&mut p, scratch, sball.pos, t, phys);
        scratch.step(dt, ctx.shape, phys);
        sball.step(dt, ctx.shape, phys);
        let hit = collide_car_ball(
            scratch,
            car_idx,
            sball,
            ctx.tick + i as i64 + 1,
            Some(ctx.shape),
            phys,
        );
        t += dt;
        if hit.is_some() {
            return score_ball(sball, t - now, ctx) - (t - now) * 4.0;
        }
        if ctx.shape.in_goal(sball.pos, phys.ball_r) >= 0 {
            break;
        }
    }
    // never touched it: judge by where the ball goes on its own, heavily penalised
    score_ball(sball, horizon, ctx) - 60.0
}

/// Scores the ball's future (no more touches) from this team's point of view.
pub fn score_ball(ball: &Ball, t0: f64, ctx: &StrikeCtx) -> f64 {
    let phys = ctx.phys;
    let glz = ctx.shape.goal_line_z();
    let mut s = BallState {
        pos: ball.pos,
        vel: ball.vel,
        ang_vel: ball.ang_vel,
        on_ground: false,
        g_scale: 1.0,
    };
    let v0 = ball.vel;
    let dt = 1.0 / 60.0;
    let opp_team = if ctx.ad > 0.0 { 1 } else { 0 };
    let mut t = 0.0;
    let mut min_own = 1e9_f64;
    while t < 3.0 {
        ball_integrate(&mut s, dt, ctx.shape, phys, false);
        t += dt;
        let g = ctx.shape.in_goal(s.pos, phys.ball_r);
        if g >= 0 {
            return if g == opp_team {
                400.0 - 60.0 * (t + t0)
            } else {
                -400.0 + 40.0 * t
            };
        }
        min_own = min_own.min(s.pos.z * ctx.ad + glz);
    }
    // where the ball heads and where it rests after 3s
    let goal = DVec3::new(0.0, 0.0, ctx.ad * glz);
    let to_goal = (goal - ball.pos).three_normalize();
    let sp = v0.length();
    let q = if sp > 0.1 { v0.dot(to_goal) / sp } else { 0.0 };
    let z_end = s.pos.z * ctx.ad / glz; // -1 own goal .. +1 their goal
    let danger = if min_own < 18.0 {
        (18.0 - min_own) * 2.0
    } else {
        0.0
    };
    q * sp.min(45.0) * 1.2
        + z_end * 40.0
        + (s.pos.x.abs() < 12.0) as i32 as f64 * z_end.max(0.0) * 15.0
        - danger
}
