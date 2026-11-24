//! Physics-simulated shot planner used by LEGEND and SAVANT bots.
//!
//! Candidate strike plans are rolled forward with the real car/ball physics (car + ball only)
//! and scored by where the ball ends up; the best plan is then executed live with the exact same
//! closed-loop controller, so what was simulated is what happens (barring other cars).

use glam::DVec3;

use crate::arena::ArenaShape;
use crate::math_ext::{Controls, PhysConfig, ThreeVec3Ext, clamp};
use crate::physics::{Ball, BallState, Car, ball_integrate, collide_car_ball};

#[derive(Clone, Copy, Debug)]
pub struct OppSnap {
    pub pos: DVec3,
    pub vel: DVec3,
    pub boost: f64,
}

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
    /// 0 = no dodge, 1 = front dodge into the ball, 2 = jump (for raised balls),
    /// 3 = aimed powershot flip with instant pitch-cancel recovery (SAVANT).
    pub dodge: u8,
    /// Lead time before `t_hit` to initiate dodge (default 0.20).
    pub dodge_lead: f64,
    /// Lateral approach / flip bias (-1..1) for corner-shot placement (default 0.0).
    pub dodge_bias: f64,
    /// Time the dodge/jump sequence started (absolute), if started.
    pub seq_t0: Option<f64>,
    pub score: f64,
}

pub const DODGE_LEAD: f64 = 0.2;

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
                c.steer = clamp(side * 1.6 + plan.dodge_bias * 0.5, -1.0, 1.0);
                c.yaw = c.steer;
            }
        } else if plan.dodge == 3 {
            // SAVANT sniper flip: boost into short hop, aimed powershot dodge, then flip-cancel for instant landing
            if st < 0.045 {
                c.jump = true;
                c.boost = car.boost > 0.0 && al.x > 0.0;
            } else if st < 0.072 {
                c.jump = false;
                c.boost = car.boost > 0.0 && al.x > 0.0;
            } else if st < 0.125 {
                c.jump = true;
                let aim_w = ball_pos + plan.sd * 0.35 - car.pos;
                let al2 = b.to_local(aim_w);
                let mag = al2.x.hypot(al2.z).max(1e-3);
                let fwd = al2.x / mag;
                let side = al2.z / mag;
                c.pitch = clamp(fwd * 1.65, -1.0, 1.0);
                c.steer = clamp(side * 1.65 + plan.dodge_bias * 0.65, -1.0, 1.0);
                c.yaw = c.steer;
                c.boost = car.boost > 0.0 && fwd > 0.5;
            } else if st > 0.19 && st < 0.44 {
                // Cancel forward flip pitch so car lands flat on wheels immediately after impact
                c.pitch = -1.0;
                let ru = b.u.y;
                let rr = b.r.y;
                c.roll = clamp(-rr.atan2(ru) * 2.5, -1.0, 1.0);
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
    let lat = DVec3::new(-plan.sd.z, 0.0, plan.sd.x);
    let bias_off = if plan.dodge_bias != 0.0 {
        lat * (plan.dodge_bias * clamp(d * 0.09, 0.12, 0.75))
    } else {
        DVec3::ZERO
    };
    let mut aim = plan.p - plan.sd * off + bias_off;
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
        let lead = if plan.dodge_lead > 0.0 {
            plan.dodge_lead
        } else {
            DODGE_LEAD
        };
        match plan.dodge {
            1 | 3 => {
                let bd = car.pos.distance(ball_pos);
                let reach = if plan.dodge == 3 {
                    5.1 + car.speed() * 0.08
                } else {
                    4.6 + car.speed() * 0.06
                };
                if rem < lead && rem > -0.1 && bd < reach && facing > 0.58 {
                    plan.seq_t0 = Some(now);
                }
            }
            2 => {
                let h = plan.p.y - 0.35;
                let dbl = plan.p.y > 3.0;
                let tj = crate::ai::jump_time_for(h - 0.3, dbl);
                if tj > 0.0 && rem < tj + 0.015 && facing > 0.58 {
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
    pub savant: bool,
    pub save: bool,
    pub opps: &'a [OppSnap],
}

/// Simulates the plan with the real physics; returns the trajectory score.
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
    let horizon = (plan.t_hit - now + 0.38).clamp(0.18, 2.3);
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
            let elapsed = t - now;
            if ctx.savant {
                let mut sc = score_ball_savant(sball, scratch.pos, elapsed, ctx);
                // Prefer earlier contact so opponents don't beat us to the ball
                sc -= elapsed * 8.0;
                return sc;
            }
            return score_ball(sball, elapsed, ctx) - elapsed * 4.0;
        }
        if ctx.shape.in_goal(sball.pos, phys.ball_r) >= 0 {
            break;
        }
    }
    // never touched it: judge by where the ball goes on its own, heavily penalised
    if ctx.savant {
        score_ball_savant(sball, scratch.pos, horizon, ctx) - 90.0
    } else {
        score_ball(sball, horizon, ctx) - 60.0
    }
}

/// Estimates fastest time an opponent can intercept `pt` from their current snapshot.
#[inline]
fn opp_eta_to(opp: &OppSnap, pt: DVec3) -> f64 {
    let dx = pt.x - opp.pos.x;
    let dz = pt.z - opp.pos.z;
    let dh = dx.hypot(dz);
    let dir_x = if dh > 1e-3 { dx / dh } else { 0.0 };
    let dir_z = if dh > 1e-3 { dz / dh } else { 0.0 };
    let v_toward = (opp.vel.x * dir_x + opp.vel.z * dir_z).max(0.0);
    let hspd = opp.vel.x.hypot(opp.vel.z);
    // Turning penalty if opponent is moving away from the intercept point
    let turn_pen = if hspd > 3.0 && v_toward < hspd * 0.5 {
        0.35
    } else if hspd > 2.0 && v_toward < hspd * 0.85 {
        0.15
    } else {
        0.05
    };
    let travel = crate::ai::time_to_cover((dh - 1.5).max(0.0), v_toward, opp.boost > 8.0);
    let air_pen = if pt.y > 4.2 {
        if opp.boost < 14.0 {
            1.6
        } else {
            (pt.y - 2.0) * 0.09
        }
    } else if pt.y > 2.1 {
        (pt.y - 1.8) * 0.07
    } else {
        0.0
    };
    turn_pen + travel + air_pen
}

/// Opponent-aware post-hit ball evaluator for SAVANT.
pub fn score_ball_savant(ball: &Ball, car_pos: DVec3, t0: f64, ctx: &StrikeCtx) -> f64 {
    let phys = ctx.phys;
    let glz = ctx.shape.goal_line_z();
    let gw = ctx.shape.gw;
    let gh = ctx.shape.gh;
    let mut s = BallState {
        pos: ball.pos,
        vel: ball.vel,
        ang_vel: ball.ang_vel,
        on_ground: false,
        g_scale: 1.0,
    };
    let v0 = ball.vel;
    let sp0 = v0.length();
    let dt = 1.0 / 60.0;
    let opp_team = if ctx.ad > 0.0 { 1 } else { 0 };
    let mut t = 0.0;
    let mut min_own = 1e9_f64;
    let mut mid_pos = ball.pos;
    let mut got_mid = false;
    let mut goal_mouth_best = -1e9_f64;

    while t < 3.0 {
        ball_integrate(&mut s, dt, ctx.shape, phys, false);
        t += dt;
        if !got_mid && t >= 0.65 {
            mid_pos = s.pos;
            got_mid = true;
        }
        let g = ctx.shape.in_goal(s.pos, phys.ball_r);
        if g >= 0 {
            if g == opp_team {
                let total_t = t + t0;
                // Check whether any opponent can intercept this shot before it crosses the line
                let mut min_margin = 2.0_f64;
                let mut nearest_def_x: Option<f64> = None;
                let mut nearest_def_d = 1e9_f64;
                let goal_pt = DVec3::new(s.pos.x, s.pos.y, ctx.ad * glz);
                for opp in ctx.opps {
                    let eta = opp_eta_to(opp, goal_pt);
                    min_margin = min_margin.min(eta - total_t);
                    let dd = (opp.pos.z - ctx.ad * glz).abs();
                    if dd < 24.0 && dd < nearest_def_d {
                        nearest_def_d = dd;
                        nearest_def_x = Some(opp.pos.x);
                    }
                }
                // Reward corner placement away from defender and upper-90 height
                let corner_bonus = if let Some(dx) = nearest_def_x {
                    ((s.pos.x - dx).abs() / (gw * 2.0)).min(1.0) * 45.0
                } else {
                    (s.pos.x.abs() / gw).min(1.0) * 20.0
                };
                let height_bonus = if s.pos.y > 1.7 && s.pos.y < gh - 0.7 {
                    ((s.pos.y - 1.5) / (gh - 2.0)).clamp(0.0, 1.0) * 38.0
                } else {
                    0.0
                };
                let power_bonus = sp0.min(55.0) * 2.2;
                let base = if min_margin > 0.22 {
                    // Unsaveable banger!
                    680.0 - 42.0 * total_t
                } else if min_margin > -0.25 {
                    // Tight / difficult save
                    510.0 + min_margin * 280.0 - 48.0 * total_t
                } else {
                    // Saveable shot — still good, but strongly prefer unsaveable placement
                    340.0 + (min_margin * 70.0).max(-110.0) - 40.0 * total_t
                };
                return base + power_bonus + corner_bonus + height_bonus;
            } else {
                // Own goal: catastrophic penalty
                return -850.0 + 40.0 * t;
            }
        }
        let own_d = s.pos.z * ctx.ad + glz;
        min_own = min_own.min(own_d);
        // Track dangerous centering / backboard threat near opponent goal
        let opp_z_prog = s.pos.z * ctx.ad / glz;
        if opp_z_prog > 0.65 && s.pos.x.abs() < gw + 7.0 && s.pos.y < 9.5 {
            goal_mouth_best = goal_mouth_best.max(opp_z_prog * 22.0);
        }
    }

    let goal = DVec3::new(0.0, 0.0, ctx.ad * glz);
    let to_goal = (goal - ball.pos).three_normalize();
    let q = if sp0 > 0.1 {
        v0.dot(to_goal) / sp0
    } else {
        0.0
    };
    let z_end = s.pos.z * ctx.ad / glz; // -1 own goal .. +1 their goal
    let danger = if min_own < 20.0 {
        (20.0 - min_own) * if ctx.save { 5.5 } else { 2.8 }
    } else {
        0.0
    };

    // Penalty for leaving the ball centered in front of our own net
    let own_center_pen = if s.pos.z * ctx.ad < -glz * 0.48 && s.pos.x.abs() < gw + 5.0 {
        if ctx.save { 75.0 } else { 40.0 }
    } else {
        0.0
    };

    // Turnover check: penalize soft touches straight into an opponent who beats us to the ball
    let mut turnover_pen = 0.0_f64;
    if !ctx.opps.is_empty() {
        let my_follow = car_pos.distance(mid_pos) / 18.0;
        for opp in ctx.opps {
            let oe = opp_eta_to(opp, mid_pos) - t0;
            if oe < 0.72 && oe + 0.15 < my_follow {
                // Worse if opponent is on the attacking side of our net
                let opp_facing_ours = (mid_pos.z - opp.pos.z) * ctx.ad < 4.0;
                if opp_facing_ours {
                    turnover_pen = turnover_pen.max(if ctx.save { 55.0 } else { 42.0 });
                } else {
                    turnover_pen = turnover_pen.max(20.0);
                }
            }
        }
    }

    let clear_bonus = if ctx.save {
        let side_clear = (s.pos.x.abs() / ctx.shape.w).min(1.0) * 18.0;
        let out_vel = (v0.z * ctx.ad).max(0.0).min(45.0) * 1.4;
        side_clear + out_vel
    } else {
        0.0
    };

    let mouth_bonus = if goal_mouth_best > 0.0 {
        goal_mouth_best
    } else {
        0.0
    };

    q * sp0.min(52.0) * 1.35
        + z_end * 46.0
        + (s.pos.x.abs() < 12.0) as i32 as f64 * z_end.max(0.0) * 16.0
        + mouth_bonus
        + clear_bonus
        - danger
        - own_center_pen
        - turnover_pen
}

/// Scores the ball's future (no more touches) from this team's point of view (LEGEND).
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
