use std::slice;
use std::sync::Arc;
use std::thread;

use glam::DVec3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::*;

const STANDARD_ARENAS: usize = 18;

fn pick_training_arena(rng: &mut StdRng) -> usize {
    if rng.r#gen::<f64>() < 0.55 {
        0
    } else {
        rng.gen_range(0..STANDARD_ARENAS)
    }
}

fn setup_kickoff(w: &mut World, team_size: usize) {
    let r = w.phys.ball_r;
    w.ball.reset(0.0, r, 0.0);
    w.ball.frozen = false;
    let sets = kickoff_sets_for(team_size);
    let order: Vec<usize> = pick(sets, &mut w.rng).to_vec();
    for team in [0u8, 1u8] {
        let idx: Vec<usize> = (0..w.cars.len())
            .filter(|&i| w.cars[i].team == team)
            .collect();
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
                bot.decide_t = 0.0;
            }
        }
    }
    w.reset_pads();
    w.kickoff = true;
    w.brain.choose_kickers(&w.cars, &w.ball);
    w.pred.update(&w.ball, &w.shape, &w.phys);
    w.last_threat = Threat {
        team: w.pred.goal_team,
        time: w.pred.goal_time,
        at: w.time,
    };
}

/// Resets `w` into a rich mix of kickoffs, aerials, defensive saves, and open-play states.
fn setup_training_episode(w: &mut World, team_size: usize, kickoff_prob: f64) {
    let roll: f64 = w.rng.r#gen();
    if roll < kickoff_prob {
        setup_kickoff(w, team_size);
        return;
    }

    w.reset_pads();
    w.kickoff = false;
    let shape = w.shape.clone();
    let glz = shape.goal_line_z();
    let r = w.phys.ball_r;

    if roll < kickoff_prob + 0.22 {
        // High aerial / bounce setup
        let bx = rand_range(&mut w.rng, -shape.w * 0.55, shape.w * 0.55);
        let bz = rand_range(&mut w.rng, -glz * 0.65, glz * 0.65);
        let by = rand_range(&mut w.rng, 4.0, 12.5);
        let p = shape.push_inside(DVec3::new(bx, by, bz), 3.0);
        w.ball.reset(p.x, by, p.z);
        w.ball.vel = DVec3::new(
            rand_range(&mut w.rng, -14.0, 14.0),
            rand_range(&mut w.rng, 2.0, 11.0),
            rand_range(&mut w.rng, -14.0, 14.0),
        );
    } else if roll < kickoff_prob + 0.44 {
        // Defensive save / shot-on-goal setup (randomly attacks team 0 or team 1)
        let target_team: f64 = if w.rng.r#gen::<bool>() { 1.0 } else { -1.0 };
        let bx = rand_range(&mut w.rng, -shape.w * 0.45, shape.w * 0.45);
        let bz = target_team * rand_range(&mut w.rng, glz * 0.15, glz * 0.55);
        let by = rand_range(&mut w.rng, r, 5.5);
        let p = shape.push_inside(DVec3::new(bx, by, bz), 3.5);
        w.ball.reset(p.x, by, p.z);
        let gx = rand_range(&mut w.rng, -shape.gw * 0.85, shape.gw * 0.85);
        let gy = rand_range(&mut w.rng, 1.0, shape.gh * 0.85);
        let gz = target_team * (glz + 1.0);
        let dir = (DVec3::new(gx, gy, gz) - w.ball.pos).three_normalize();
        let spd = rand_range(&mut w.rng, 18.0, 36.0);
        w.ball.vel = dir * spd;
    } else {
        // Random open play / corner / wall roll
        let bx = rand_range(&mut w.rng, -shape.w * 0.82, shape.w * 0.82);
        let bz = rand_range(&mut w.rng, -glz * 0.82, glz * 0.82);
        let by = rand_range(&mut w.rng, r, 6.0);
        let p = shape.push_inside(DVec3::new(bx, by, bz), 2.5);
        w.ball.reset(p.x, by, p.z);
        w.ball.vel = DVec3::new(
            rand_range(&mut w.rng, -22.0, 22.0),
            rand_range(&mut w.rng, -2.0, 10.0),
            rand_range(&mut w.rng, -22.0, 22.0),
        );
    }

    let bpos = w.ball.pos;
    for ci in 0..w.cars.len() {
        let team = w.cars[ci].team;
        let ad = if team == 0 { 1.0 } else { -1.0 };
        // Place car mostly goal-side of ball with randomized offset & momentum
        let cz = clamp(
            bpos.z - ad * rand_range(&mut w.rng, 4.0, 28.0) + rand_range(&mut w.rng, -8.0, 8.0),
            -glz + 4.0,
            glz - 4.0,
        );
        let cx = rand_range(&mut w.rng, -shape.w * 0.72, shape.w * 0.72);
        let cp = shape.push_inside(DVec3::new(cx, 0.35, cz), 3.5);
        let yaw = yaw_toward(cp.x, cp.z, bpos.x, bpos.z) + rand_range(&mut w.rng, -0.55, 0.55);
        let spd = rand_range(&mut w.rng, 2.0, 18.0);
        let boost = rand_range(&mut w.rng, 15.0, 100.0);
        let car = &mut w.cars[ci];
        car.place(cp.x, 0.35, cp.z, yaw);
        car.vel = car.basis.f * spd;
        car.boost = boost;
        car.demolished = false;
        if let Some(bot) = car.bot.as_mut() {
            bot.reset();
            bot.decide_t = 0.0;
        }
    }
    w.pred.update(&w.ball, &w.shape, &w.phys);
    w.last_threat = Threat {
        team: w.pred.goal_team,
        time: w.pred.goal_time,
        at: w.time,
    };
}

// ---------------------------------------------------------------------------------------------
// 1. Multi-Threaded Vectorized PPO Self-Play + League Environment Pool (Pure From-Scratch RL)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum OppKind {
    SelfPlay,
    Legend,
    MpcSavant,
    Champ,
    Snapshot,
}

struct SingleEnv {
    world: World,
    team_size: usize,
    opp_kind: OppKind,
    /// Which team is controlled by the external Python learner when not SelfPlay (0 or 1).
    rl_team: u8,
    /// Controlled car indices in this env (all cars for SelfPlay, rl_team cars for VsBot).
    rl_cars: Vec<usize>,
    /// Opponent snapshot car indices (when `opp_kind == Snapshot`).
    snap_cars: Vec<usize>,
    snap_idx: usize,
    prev_ctls: Vec<Controls>,
    ep_steps: usize,
    since_kick: f64,
    prev_ball_z_norm: f64,
    prev_boost: Vec<f64>,
    rng: StdRng,
}

impl SingleEnv {
    fn new(team_size: usize, opp_kind: OppKind, rl_team: u8, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let arena_idx = pick_training_arena(&mut rng);
        let mut world = World::with_seed(ARENAS[arena_idx], &ArenaOpts::default(), rng.r#gen());
        let mut rl_cars = Vec::new();
        let mut snap_cars = Vec::new();
        let mut id = 0usize;

        for team in [0u8, 1u8] {
            for k in 0..team_size {
                let body = rng.gen_range(0..HITBOXES.len());
                let ci = world.add_car(Car::new(id, team, format!("C{id}"), body));
                let is_rl = opp_kind == OppKind::SelfPlay || team == rl_team;
                if is_rl {
                    rl_cars.push(ci);
                } else {
                    match opp_kind {
                        OppKind::Legend => {
                            world.cars[ci].bot = Some(BotAi::new_with_rng(4, k % PERSONALITIES.len(), &mut world.rng));
                        }
                        OppKind::MpcSavant => {
                            let mut bot = BotAi::new_with_rng(5, k % PERSONALITIES.len(), &mut world.rng);
                            bot.force_mpc = true;
                            world.cars[ci].bot = Some(bot);
                        }
                        OppKind::Champ => {
                            world.cars[ci].bot = Some(BotAi::new_with_rng(3, k % PERSONALITIES.len(), &mut world.rng));
                        }
                        OppKind::Snapshot => {
                            snap_cars.push(ci);
                        }
                        OppKind::SelfPlay => unreachable!(),
                    }
                }
                id += 1;
            }
        }

        let n_cars = world.cars.len();
        let mut env = Self {
            world,
            team_size,
            opp_kind,
            rl_team,
            rl_cars,
            snap_cars,
            snap_idx: 0,
            prev_ctls: vec![Controls::default(); n_cars],
            ep_steps: 0,
            since_kick: 0.0,
            prev_ball_z_norm: 0.0,
            prev_boost: vec![33.0; n_cars],
            rng,
        };
        env.reset(0);
        env
    }

    fn reset(&mut self, n_snapshots: usize) {
        let arena_idx = pick_training_arena(&mut self.rng);
        let new_def = ARENAS[arena_idx];
        if self.world.def.id != new_def.id {
            self.world.def = new_def;
            self.world.shape = ArenaShape::new(&new_def, &ArenaOpts::default());
            self.world.pads = make_boost_pads(&self.world.shape);
        }
        setup_training_episode(&mut self.world, self.team_size, 0.35);
        for c in &mut self.prev_ctls {
            *c = Controls::default();
        }
        self.ep_steps = 0;
        self.since_kick = 0.0;
        let glz = self.world.shape.goal_line_z().max(1.0);
        self.prev_ball_z_norm = self.world.ball.pos.z / glz;
        for (i, c) in self.world.cars.iter().enumerate() {
            self.prev_boost[i] = c.boost;
        }
        if self.opp_kind == OppKind::Snapshot && n_snapshots > 0 {
            self.snap_idx = self.rng.gen_range(0..n_snapshots);
            for &ci in &self.snap_cars {
                self.world.cars[ci].bot = None;
            }
        } else if self.opp_kind == OppKind::Snapshot {
            // Fallback to MPC-SAVANT until first snapshot is loaded
            for (k, &ci) in self.snap_cars.iter().enumerate() {
                if self.world.cars[ci].bot.is_none() {
                    let mut bot = BotAi::new_with_rng(5, k % PERSONALITIES.len(), &mut self.world.rng);
                    bot.force_mpc = true;
                    self.world.cars[ci].bot = Some(bot);
                }
            }
        }
    }

    fn write_obs(&self, obs_slice: &mut [f32]) {
        for (slot, &ci) in self.rl_cars.iter().enumerate() {
            let mut buf = [0.0_f32; OBS_DIM];
            extract_obs(&self.world, ci, &self.prev_ctls[ci], &mut buf);
            obs_slice[slot * OBS_DIM..(slot + 1) * OBS_DIM].copy_from_slice(&buf);
        }
    }

    fn step(
        &mut self,
        actions: &[u8],
        shaping_scale: f32,
        snapshots: &[Arc<SavantPolicy>],
        obs_out: &mut [f32],
        rew_out: &mut [f32],
        done_out: &mut [u8],
        goals_scored: &mut u32,
        goals_conceded: &mut u32,
        touches_out: &mut u32,
    ) {
        let n_cars = self.world.cars.len();
        for (slot, &ci) in self.rl_cars.iter().enumerate() {
            let mut a = [0u8; ACT_HEADS];
            a.copy_from_slice(&actions[slot * ACT_HEADS..(slot + 1) * ACT_HEADS]);
            let ctl = decode_action(&a);
            self.prev_ctls[ci] = ctl;
        }

        // Snapshot opponents run natively in Rust at 30 Hz
        if self.opp_kind == OppKind::Snapshot && !snapshots.is_empty() {
            let pol = &snapshots[self.snap_idx % snapshots.len()];
            for &ci in &self.snap_cars {
                if !self.world.cars[ci].demolished {
                    let mut ob = [0.0_f32; OBS_DIM];
                    extract_obs(&self.world, ci, &self.prev_ctls[ci], &mut ob);
                    let act = pol.act_greedy(&ob);
                    self.prev_ctls[ci] = decode_action(&act);
                }
            }
        }

        let dt = self.world.phys.dt;
        let glz = self.world.shape.goal_line_z().max(1.0);
        let mut car_event_rew = [0.0_f64; 8];
        let mut goal_team: i32 = -1;

        for _ in 0..POLICY_TICK_SKIP {
            for &ci in &self.rl_cars {
                self.world.cars[ci].controls = self.prev_ctls[ci];
            }
            if self.opp_kind == OppKind::Snapshot && !snapshots.is_empty() {
                for &ci in &self.snap_cars {
                    self.world.cars[ci].controls = self.prev_ctls[ci];
                }
            }
            self.world.step(dt);
            self.since_kick += dt;
            if self.world.kickoff && (self.world.ball.last_touch.is_some() || self.since_kick > 5.0) {
                self.world.kickoff = false;
            }
            for ev in &self.world.events {
                match *ev {
                    WorldEvent::Hit(h) => {
                        let ci = h.car_idx;
                        let ad = if self.world.cars[ci].team == 0 { 1.0 } else { -1.0 };
                        let opp_g = DVec3::new(0.0, 1.5, ad * glz);
                        let to_goal = (opp_g - self.world.ball.pos).three_normalize();
                        let goal_dir_q = self.world.ball.vel.dot(to_goal) / 35.0;
                        let touch_r = 0.14
                            + 0.48 * (h.strength / 30.0).min(1.3)
                            + 0.35 * (h.y / 10.0).min(1.0)
                            + 0.45 * goal_dir_q.clamp(-1.0, 1.5);
                        car_event_rew[ci] += touch_r;
                        if h.shot {
                            car_event_rew[ci] += 1.5;
                        }
                        if h.save {
                            car_event_rew[ci] += if h.epic { 3.0 } else { 2.0 };
                        }
                        if self.rl_cars.contains(&ci) {
                            *touches_out += 1;
                        }
                    }
                    WorldEvent::Demo {
                        attacker_idx,
                        victim_idx,
                        ..
                    } => {
                        car_event_rew[attacker_idx] += 1.0;
                        car_event_rew[victim_idx] -= 1.0;
                    }
                    _ => {}
                }
            }
            let g = self.world.shape.in_goal(self.world.ball.pos, self.world.phys.ball_r);
            if g >= 0 {
                goal_team = g;
                break;
            }
        }

        self.ep_steps += 1;
        let new_ball_z_norm = self.world.ball.pos.z / glz;
        let dz_ball = new_ball_z_norm - self.prev_ball_z_norm;
        self.prev_ball_z_norm = new_ball_z_norm;

        // Compute per-car dense shaping
        let mut dense = [0.0_f64; 8];
        let bpos = self.world.ball.pos;
        let bvel = self.world.ball.vel;
        for ci in 0..n_cars {
            let car = &self.world.cars[ci];
            let ad = if car.team == 0 { 1.0 } else { -1.0 };
            let opp_g = DVec3::new(0.0, 1.5, ad * glz);
            let to_goal = (opp_g - bpos).three_normalize();
            let vel_to_goal = bvel.dot(to_goal) / 40.0;

            let to_ball = (bpos - car.pos).three_normalize();
            let vel_to_ball = car.vel.dot(to_ball) / 23.0;
            let face_ball = car.basis.f.dot(to_ball);

            let dboost = (car.boost - self.prev_boost[ci]).max(0.0) / 100.0;
            self.prev_boost[ci] = car.boost;

            // Potential ball progress + ball velocity toward goal + boost pickup + annealed ball-approach
            let sh = shaping_scale as f64;
            let mut r_dense = 0.28 * (dz_ball * ad)
                + 0.022 * vel_to_goal
                + 0.10 * dboost
                + 0.045 * sh * vel_to_ball
                + 0.008 * sh * face_ball;

            // 2v2+ anti-double-commit spacing cue: discourage two teammates piling onto the exact same spot
            if self.team_size >= 2 {
                let my_bd = car.pos.distance(bpos);
                for oj in 0..n_cars {
                    if oj != ci && self.world.cars[oj].team == car.team && !self.world.cars[oj].demolished {
                        let sep = car.pos.distance(self.world.cars[oj].pos);
                        let o_bd = self.world.cars[oj].pos.distance(bpos);
                        if sep < 7.5 && my_bd < 16.0 && o_bd < 16.0 && my_bd > o_bd {
                            r_dense -= 0.02;
                        }
                    }
                }
            }
            dense[ci] = r_dense;
        }

        let mut team_dense = [0.0_f64; 2];
        let mut team_cnt = [0.0_f64; 2];
        for ci in 0..n_cars {
            let t = self.world.cars[ci].team as usize;
            team_dense[t] += dense[ci] + car_event_rew[ci];
            team_cnt[t] += 1.0;
        }
        for t in 0..2 {
            if team_cnt[t] > 0.0 {
                team_dense[t] /= team_cnt[t];
            }
        }

        let done = goal_team >= 0 || self.ep_steps >= 450;
        if goal_team >= 0 && self.opp_kind != OppKind::SelfPlay {
            if goal_team == self.rl_team as i32 {
                *goals_conceded += 1;
            } else {
                *goals_scored += 1;
            }
        }

        for (slot, &ci) in self.rl_cars.iter().enumerate() {
            let t = self.world.cars[ci].team as usize;
            let opp = 1 - t;
            // Team-blended + zero-sum reward (heavy goal dominance so winning > touch farming)
            let mut r = 0.65 * (dense[ci] + car_event_rew[ci]) + 0.35 * team_dense[t] - 0.35 * team_dense[opp];
            if goal_team >= 0 {
                if goal_team == t as i32 {
                    r -= 20.0;
                } else {
                    r += 20.0;
                }
            }
            rew_out[slot] = r as f32;
            done_out[slot] = u8::from(done);
        }

        if done {
            self.reset(snapshots.len());
        }
        self.write_obs(obs_out);
    }
}

pub struct EnvPool {
    envs: Vec<SingleEnv>,
    agent_offsets: Vec<usize>,
    total_agents: usize,
    snapshots: Vec<Arc<SavantPolicy>>,
    n_threads: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_pool_new(num_envs: usize, seed: u64, pool_mode: u32) -> *mut EnvPool {
    let mut envs = Vec::with_capacity(num_envs);
    let mut agent_offsets = Vec::with_capacity(num_envs);
    let mut total_agents = 0usize;

    // Interleaved evenly so every worker thread gets an identical compute load!
    // pool_mode == 0 (Genesis):     85% 1v1 / 15% 2v2 | 75% SelfPlay, 20% Snapshot, 5% Champ
    // pool_mode == 1 (Rising):      70% 1v1 / 30% 2v2 | 55% SelfPlay, 25% Snapshot, 8% Champ, 10% Legend, 2% MpcSavant
    // pool_mode == 2 (Competitive): 55% 1v1 / 45% 2v2 | 50% SelfPlay, 25% Snapshot, 15% Legend, 10% MpcSavant
    // pool_mode == 3 (Apex):        50% 1v1 / 50% 2v2 | 55% SelfPlay, 30% Snapshot, 5% Legend, 10% MpcSavant
    for i in 0..num_envs {
        let team_size = match pool_mode {
            0 => if i % 20 < 3 { 2 } else { 1 },
            1 => if i % 10 < 3 { 2 } else { 1 },
            2 => if i % 20 < 9 { 2 } else { 1 },
            _ => if i % 2 == 0 { 2 } else { 1 },
        };
        let m = i % 100;
        let opp_kind = match pool_mode {
            0 => {
                if m < 75 {
                    OppKind::SelfPlay
                } else if m < 95 {
                    OppKind::Snapshot
                } else {
                    OppKind::Champ
                }
            }
            1 => {
                if m < 55 {
                    OppKind::SelfPlay
                } else if m < 80 {
                    OppKind::Snapshot
                } else if m < 88 {
                    OppKind::Champ
                } else if m < 98 {
                    OppKind::Legend
                } else {
                    OppKind::MpcSavant
                }
            }
            2 => {
                if m < 50 {
                    OppKind::SelfPlay
                } else if m < 75 {
                    OppKind::Snapshot
                } else if m < 90 {
                    OppKind::Legend
                } else {
                    OppKind::MpcSavant
                }
            }
            _ => {
                if m < 55 {
                    OppKind::SelfPlay
                } else if m < 85 {
                    OppKind::Snapshot
                } else if m < 90 {
                    OppKind::Legend
                } else {
                    OppKind::MpcSavant
                }
            }
        };
        let rl_team = (i % 2) as u8;
        let env = SingleEnv::new(
            team_size,
            opp_kind,
            rl_team,
            seed.wrapping_add((i as u64 + 1) * 0x517CC1B727220A95),
        );
        agent_offsets.push(total_agents);
        total_agents += env.rl_cars.len();
        envs.push(env);
    }

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(24)
        .min(num_envs.max(1));

    Box::into_raw(Box::new(EnvPool {
        envs,
        agent_offsets,
        total_agents,
        snapshots: Vec::new(),
        n_threads,
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pool_num_agents(pool: *const EnvPool) -> usize {
    if pool.is_null() {
        return 0;
    }
    unsafe { (*pool).total_agents }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pool_reset(pool: *mut EnvPool, obs_out: *mut f32) {
    if pool.is_null() || obs_out.is_null() {
        return;
    }
    let p = unsafe { &mut *pool };
    let n_snaps = p.snapshots.len();
    let obs_all = unsafe { slice::from_raw_parts_mut(obs_out, p.total_agents * OBS_DIM) };
    for (ei, env) in p.envs.iter_mut().enumerate() {
        env.reset(n_snaps);
        let off = p.agent_offsets[ei];
        let n_a = env.rl_cars.len();
        env.write_obs(&mut obs_all[off * OBS_DIM..(off + n_a) * OBS_DIM]);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pool_add_snapshot(
    pool: *mut EnvPool,
    bytes_ptr: *const u8,
    bytes_len: usize,
    max_keep: usize,
) -> i32 {
    if pool.is_null() || bytes_ptr.is_null() || bytes_len == 0 {
        return 0;
    }
    let p = unsafe { &mut *pool };
    let bytes = unsafe { slice::from_raw_parts(bytes_ptr, bytes_len) };
    let Some(pol) = SavantPolicy::from_bytes(bytes) else {
        return 0;
    };
    p.snapshots.push(Arc::new(pol));
    let cap = max_keep.max(1);
    if p.snapshots.len() > cap {
        p.snapshots.remove(0);
    }
    p.snapshots.len() as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pool_step(
    pool: *mut EnvPool,
    actions_in: *const u8,
    shaping_scale: f32,
    obs_out: *mut f32,
    rew_out: *mut f32,
    done_out: *mut u8,
    stats_out: *mut u32, // [goals_scored, goals_conceded, touches]
) {
    if pool.is_null() || actions_in.is_null() || obs_out.is_null() || rew_out.is_null() || done_out.is_null() {
        return;
    }
    let p = unsafe { &mut *pool };
    let total = p.total_agents;
    let acts_all = unsafe { slice::from_raw_parts(actions_in, total * ACT_HEADS) };
    let obs_all = unsafe { slice::from_raw_parts_mut(obs_out, total * OBS_DIM) };
    let rew_all = unsafe { slice::from_raw_parts_mut(rew_out, total) };
    let done_all = unsafe { slice::from_raw_parts_mut(done_out, total) };

    let n_envs = p.envs.len();
    let n_threads = p.n_threads.min(n_envs.max(1));
    let chunk_envs = (n_envs + n_threads - 1) / n_threads;
    let snaps = &p.snapshots;

    let mut tot_gs = 0u32;
    let mut tot_gc = 0u32;
    let mut tot_tc = 0u32;

    thread::scope(|s| {
        let mut env_rest = p.envs.as_mut_slice();
        let mut off_idx = 0usize;
        let mut obs_rest = obs_all;
        let mut rew_rest = rew_all;
        let mut done_rest = done_all;
        let mut handles = Vec::with_capacity(n_threads);

        while !env_rest.is_empty() {
            let take = chunk_envs.min(env_rest.len());
            let (e_chunk, e_next) = env_rest.split_at_mut(take);
            env_rest = e_next;

            let a_start = p.agent_offsets[off_idx];
            let a_end = if off_idx + take < n_envs {
                p.agent_offsets[off_idx + take]
            } else {
                total
            };
            let a_cnt = a_end - a_start;
            off_idx += take;

            let a_slice = &acts_all[a_start * ACT_HEADS..a_end * ACT_HEADS];
            let (o_chunk, o_next) = obs_rest.split_at_mut(a_cnt * OBS_DIM);
            obs_rest = o_next;
            let (r_chunk, r_next) = rew_rest.split_at_mut(a_cnt);
            rew_rest = r_next;
            let (d_chunk, d_next) = done_rest.split_at_mut(a_cnt);
            done_rest = d_next;

            handles.push(s.spawn(move || {
                let mut gs = 0u32;
                let mut gc = 0u32;
                let mut tc = 0u32;
                let mut local_off = 0usize;
                for env in e_chunk.iter_mut() {
                    let na = env.rl_cars.len();
                    env.step(
                        &a_slice[local_off * ACT_HEADS..(local_off + na) * ACT_HEADS],
                        shaping_scale,
                        snaps,
                        &mut o_chunk[local_off * OBS_DIM..(local_off + na) * OBS_DIM],
                        &mut r_chunk[local_off..local_off + na],
                        &mut d_chunk[local_off..local_off + na],
                        &mut gs,
                        &mut gc,
                        &mut tc,
                    );
                    local_off += na;
                }
                (gs, gc, tc)
            }));
        }

        for h in handles {
            let (gs, gc, tc) = h.join().unwrap();
            tot_gs += gs;
            tot_gc += gc;
            tot_tc += tc;
        }
    });

    if !stats_out.is_null() {
        let st = unsafe { slice::from_raw_parts_mut(stats_out, 3) };
        st[0] = tot_gs;
        st[1] = tot_gc;
        st[2] = tot_tc;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_pool_free(pool: *mut EnvPool) {
    if !pool.is_null() {
        drop(unsafe { Box::from_raw(pool) });
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Fast Multi-Threaded Full-Match Evaluator (MLP vs LEGEND / MPC-SAVANT)
// ---------------------------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_eval_policy(
    bytes_ptr: *const u8,
    bytes_len: usize,
    opp_diff: usize,
    team_size: usize,
    num_matches: usize,
    secs: f64,
    arena_idx: usize,
    out_metrics: *mut i32, // [goals_a, goals_b, wins_a, wins_b, shots_a, shots_b, owngoals_a]
) -> i32 {
    if bytes_ptr.is_null() || bytes_len == 0 || out_metrics.is_null() {
        return 0;
    }
    let bytes = unsafe { slice::from_raw_parts(bytes_ptr, bytes_len) };
    let Some(pol) = SavantPolicy::from_bytes(bytes) else {
        return 0;
    };
    let pol = Arc::new(pol);
    let arena = ARENAS[arena_idx.min(ARENAS.len() - 1)];

    let mut agg = [0i32; 7];
    thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_matches);
        for m in 0..num_matches {
            let pol = Arc::clone(&pol);
            handles.push(s.spawn(move || {
                let swap = m % 2 == 1;
                let a_team: u8 = if swap { 1 } else { 0 };
                let mut w = World::with_seed(arena, &ArenaOpts::default(), 5000 + m as u64);
                let mut id = 0usize;
                let mut a_cars = Vec::new();
                for team in [0u8, 1u8] {
                    for k in 0..team_size {
                        let ci = w.add_car(Car::new(id, team, format!("E{id}"), 0));
                        if team == a_team {
                            a_cars.push(ci);
                        } else {
                            let mut bot = BotAi::new_with_rng(opp_diff, k % PERSONALITIES.len(), &mut w.rng);
                            bot.force_mpc = true;
                            w.cars[ci].bot = Some(bot);
                        }
                        id += 1;
                    }
                }
                setup_kickoff(&mut w, team_size);
                let mut prev_ctls = vec![Controls::default(); w.cars.len()];
                let mut rem = vec![0u8; w.cars.len()];
                let dt = w.phys.dt;
                let mut t = 0.0;
                let mut since = 0.0;
                let (mut ga, mut gb, mut sa, mut sb, mut oga) = (0i32, 0i32, 0i32, 0i32, 0i32);

                while t < secs {
                    for &ci in &a_cars {
                        if !w.cars[ci].demolished {
                            if rem[ci] == 0 {
                                let mut ob = [0.0_f32; OBS_DIM];
                                extract_obs(&w, ci, &prev_ctls[ci], &mut ob);
                                let act = pol.act_greedy(&ob);
                                prev_ctls[ci] = decode_action(&act);
                                rem[ci] = POLICY_TICK_SKIP as u8;
                            }
                            rem[ci] = rem[ci].saturating_sub(1);
                            w.cars[ci].controls = prev_ctls[ci];
                        }
                    }
                    w.step(dt);
                    for ev in &w.events {
                        if let WorldEvent::Hit(h) = ev
                            && h.shot
                        {
                            if w.cars[h.car_idx].team == a_team {
                                sa += 1;
                            } else {
                                sb += 1;
                            }
                        }
                    }
                    t += dt;
                    since += dt;
                    if w.kickoff && (w.ball.last_touch.is_some() || since > 6.0) {
                        w.kickoff = false;
                    }
                    let gt = w.shape.in_goal(w.ball.pos, w.phys.ball_r);
                    if gt >= 0 {
                        let scorer = 1 - gt as u8;
                        if scorer == a_team {
                            ga += 1;
                        } else {
                            gb += 1;
                            if let Some(lt) = w.ball.last_touch
                                && w.cars[lt].team == a_team
                            {
                                oga += 1;
                            }
                        }
                        setup_kickoff(&mut w, team_size);
                        prev_ctls.fill(Controls::default());
                        rem.fill(0);
                        since = 0.0;
                    }
                }
                let wa = i32::from(ga > gb);
                let wb = i32::from(gb > ga);
                [ga, gb, wa, wb, sa, sb, oga]
            }));
        }
        for h in handles {
            let r = h.join().unwrap();
            for i in 0..7 {
                agg[i] += r[i];
            }
        }
    });

    let out = unsafe { slice::from_raw_parts_mut(out_metrics, 7) };
    out.copy_from_slice(&agg);
    1
}
