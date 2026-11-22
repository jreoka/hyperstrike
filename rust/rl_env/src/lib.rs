use std::slice;
use std::thread;

use glam::DVec3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sim::{
    ARENAS, ArenaOpts, BotAi, Car, Controls, HITBOXES, KICKOFF_SPOTS, PERSONALITIES, World,
    WorldEvent,
};

pub const MAX_CARS: usize = 10;
pub const OTHER_CARS: usize = MAX_CARS - 1;
pub const NUM_PADS_OBS: usize = 6;

pub const SELF_DIM: usize = 24;
pub const BALL_DIM: usize = 16;
pub const OTHER_DIM: usize = 18;
pub const PAD_DIM: usize = 4;
pub const ACTION_DIM: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EnvConfig {
    pub num_envs: u32,
    pub min_team_size: u32,
    pub max_team_size: u32,
    pub ticks_per_step: u32,
    pub max_episode_steps: u32,
    pub bot_opponent_prob: f32,
    pub bot_diff: u32,
    pub num_threads: u32,
    pub seed: u64,
    // Reward weights
    pub w_goal: f32,
    pub w_touch: f32,
    pub w_shot: f32,
    pub w_save: f32,
    pub w_demo: f32,
    pub w_ball_to_goal: f32,
    pub w_player_to_ball: f32,
    pub w_boost_pickup: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub goals_blue: u32,
    pub goals_orange: u32,
    pub episodes_done: u32,
    pub touches: u32,
    pub bot_games_won: u32,
    pub bot_games_lost: u32,
}

struct SingleEnv {
    world: World,
    team_size: usize,
    step_count: u32,
    bot_opponent: bool,
    rng: StdRng,
    // Expert CHAMP bots used on demand for BC warmstart
    experts: Vec<BotAi>,
}

impl SingleEnv {
    fn new(cfg: &EnvConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let world = World::with_seed(ARENAS[0], &ArenaOpts::default(), rng.r#gen());
        let mut env = Self {
            world,
            team_size: 1,
            step_count: 0,
            bot_opponent: false,
            rng,
            experts: Vec::new(),
        };
        env.reset(cfg);
        env
    }

    fn reset(&mut self, cfg: &EnvConfig) {
        let min_ts = (cfg.min_team_size as usize).clamp(1, MAX_CARS / 2);
        let max_ts = (cfg.max_team_size as usize).clamp(min_ts, MAX_CARS / 2);
        let team_size = self.rng.gen_range(min_ts..=max_ts);
        let std_count = ARENAS.iter().take_while(|a| !a.weird).count().max(1);
        let arena_idx = self.rng.gen_range(0..std_count);
        let scale = if team_size >= 4 { 1.12 } else { 1.0 };
        let opts = ArenaOpts {
            layout: "default".to_string(),
            scale,
        };
        let mut world = World::with_seed(ARENAS[arena_idx], &opts, self.rng.r#gen());
        let bot_opponent = self.rng.r#gen::<f32>() < cfg.bot_opponent_prob;
        let bot_diff = (cfg.bot_diff as usize).min(4);

        let mut experts = Vec::with_capacity(team_size * 2);
        let mut car_id = 0;
        for team in 0..=1_u8 {
            for slot in 0..team_size {
                let body = self.rng.gen_range(0..HITBOXES.len());
                let mut car = Car::new(car_id, team, "RL", body);
                if bot_opponent && team == 1 {
                    let pers = slot % PERSONALITIES.len();
                    car.bot = Some(BotAi::new_with_rng(bot_diff, pers, &mut self.rng));
                }
                world.add_car(car);
                experts.push(BotAi::new_with_rng(
                    4,
                    slot % PERSONALITIES.len(),
                    &mut self.rng,
                ));
                car_id += 1;
            }
        }

        // Place cars on kickoff spots
        Self::place_kickoff(&mut world, team_size, &mut self.rng);
        self.world = world;
        self.team_size = team_size;
        self.step_count = 0;
        self.bot_opponent = bot_opponent;
        self.experts = experts;
    }

    fn place_kickoff(world: &mut World, team_size: usize, rng: &mut StdRng) {
        world.ball.reset(0.0, world.phys.ball_r + 0.05, 0.0);
        world.reset_pads();
        world.kickoff = true;
        world.kickoff_freeze = false;
        world.frozen = false;

        // Pick a random kickoff pattern offset so 1v1..5v5 see all kickoff formations
        let pat = rng.gen_range(0..KICKOFF_SPOTS.len());
        let mut idx_per_team = [0_usize, 0_usize];
        for i in 0..world.cars.len() {
            let team = world.cars[i].team;
            let k = idx_per_team[team as usize];
            idx_per_team[team as usize] += 1;
            let spot = KICKOFF_SPOTS[(pat + k) % KICKOFF_SPOTS.len()];
            let (x, z) = world.shape.kickoff_pos(spot, team);
            let yaw = (-z).atan2(-x);
            world.cars[i].place(x, 0.35, z, yaw);
            world.cars[i].boost = 33.0;
            world.cars[i].demolished = false;
            if let Some(ref mut bot) = world.cars[i].bot {
                bot.reset();
            }
        }
        world.brain.choose_kickers(&world.cars, &world.ball);
    }

    fn write_obs(
        &self,
        max_steps: u32,
        self_out: &mut [f32],
        ball_out: &mut [f32],
        other_out: &mut [f32],
        other_mask_out: &mut [f32],
        pads_out: &mut [f32],
        active_mask_out: &mut [f32],
        team_ids_out: &mut [i32],
    ) {
        self_out.fill(0.0);
        ball_out.fill(0.0);
        other_out.fill(0.0);
        other_mask_out.fill(0.0);
        pads_out.fill(0.0);
        active_mask_out.fill(0.0);
        team_ids_out.fill(0);

        let w = &self.world;
        let inv_w = 1.0 / w.shape.w.max(1.0);
        let inv_l = 1.0 / w.shape.goal_line_z().max(1.0);
        let inv_h = 1.0 / 20.0;
        let ts_norm = self.team_size as f32 / 5.0;
        let is_kickoff = if w.kickoff { 1.0_f32 } else { 0.0_f32 };
        let time_frac = 1.0 - (self.step_count as f32 / max_steps.max(1) as f32).min(1.0);

        let n_cars = w.cars.len().min(MAX_CARS);
        for ci in 0..n_cars {
            let car = &w.cars[ci];
            team_ids_out[ci] = car.team as i32;
            let is_neural = car.bot.is_none();
            active_mask_out[ci] = if is_neural { 1.0 } else { 0.0 };

            // Mirror team 1 so every car sees itself attacking +Z
            let s = if car.team == 0 { 1.0 } else { -1.0 };
            let c_pos = DVec3::new(car.pos.x * s, car.pos.y, car.pos.z * s);
            let c_vel = DVec3::new(car.vel.x * s, car.vel.y, car.vel.z * s);
            let c_fwd = DVec3::new(car.basis.f.x * s, car.basis.f.y, car.basis.f.z * s);
            let c_up = DVec3::new(car.basis.u.x * s, car.basis.u.y, car.basis.u.z * s);
            let c_ang = DVec3::new(car.ang_vel.x * s, car.ang_vel.y, car.ang_vel.z * s);

            // Self obs (24)
            let so = &mut self_out[ci * SELF_DIM..(ci + 1) * SELF_DIM];
            so[0] = (c_pos.x * inv_w) as f32;
            so[1] = (c_pos.y * inv_h) as f32;
            so[2] = (c_pos.z * inv_l) as f32;
            so[3] = (c_vel.x / 23.0) as f32;
            so[4] = (c_vel.y / 23.0) as f32;
            so[5] = (c_vel.z / 23.0) as f32;
            so[6] = c_fwd.x as f32;
            so[7] = c_fwd.y as f32;
            so[8] = c_fwd.z as f32;
            so[9] = c_up.x as f32;
            so[10] = c_up.y as f32;
            so[11] = c_up.z as f32;
            so[12] = (c_ang.x / 5.5) as f32;
            so[13] = (c_ang.y / 5.5) as f32;
            so[14] = (c_ang.z / 5.5) as f32;
            so[15] = (car.boost / 100.0) as f32;
            so[16] = if car.on_ground { 1.0 } else { 0.0 };
            so[17] = if car.has_flip { 1.0 } else { 0.0 };
            so[18] = if car.flipping { 1.0 } else { 0.0 };
            so[19] = if car.supersonic { 1.0 } else { 0.0 };
            so[20] = if car.demolished { 1.0 } else { 0.0 };
            so[21] = ts_norm;
            so[22] = is_kickoff;
            so[23] = time_frac;

            // Ball obs (16)
            let b_pos = DVec3::new(w.ball.pos.x * s, w.ball.pos.y, w.ball.pos.z * s);
            let b_vel = DVec3::new(w.ball.vel.x * s, w.ball.vel.y, w.ball.vel.z * s);
            let b_ang = DVec3::new(w.ball.ang_vel.x * s, w.ball.ang_vel.y, w.ball.ang_vel.z * s);
            let rel_b_pos = b_pos - c_pos;
            let rel_b_vel = b_vel - c_vel;
            let bo = &mut ball_out[ci * BALL_DIM..(ci + 1) * BALL_DIM];
            bo[0] = (b_pos.x * inv_w) as f32;
            bo[1] = (b_pos.y * inv_h) as f32;
            bo[2] = (b_pos.z * inv_l) as f32;
            bo[3] = (b_vel.x / 40.0) as f32;
            bo[4] = (b_vel.y / 40.0) as f32;
            bo[5] = (b_vel.z / 40.0) as f32;
            bo[6] = (b_ang.x / 6.0) as f32;
            bo[7] = (b_ang.y / 6.0) as f32;
            bo[8] = (b_ang.z / 6.0) as f32;
            bo[9] = (rel_b_pos.x / 40.0) as f32;
            bo[10] = (rel_b_pos.y / 20.0) as f32;
            bo[11] = (rel_b_pos.z / 40.0) as f32;
            bo[12] = (rel_b_vel.x / 40.0) as f32;
            bo[13] = (rel_b_vel.y / 40.0) as f32;
            bo[14] = (rel_b_vel.z / 40.0) as f32;
            bo[15] = (rel_b_pos.length() / 60.0) as f32;

            // Other cars obs (9 x 18)
            let mut slot = 0;
            for oi in 0..n_cars {
                if oi == ci {
                    continue;
                }
                let o = &w.cars[oi];
                let mask_idx = ci * OTHER_CARS + slot;
                if !o.demolished {
                    other_mask_out[mask_idx] = 1.0;
                    let o_pos = DVec3::new(o.pos.x * s, o.pos.y, o.pos.z * s);
                    let o_vel = DVec3::new(o.vel.x * s, o.vel.y, o.vel.z * s);
                    let o_fwd = DVec3::new(o.basis.f.x * s, o.basis.f.y, o.basis.f.z * s);
                    let rel_p = o_pos - c_pos;
                    let rel_v = o_vel - c_vel;
                    let base = (ci * OTHER_CARS + slot) * OTHER_DIM;
                    let oo = &mut other_out[base..base + OTHER_DIM];
                    oo[0] = (rel_p.x / 40.0) as f32;
                    oo[1] = (rel_p.y / 20.0) as f32;
                    oo[2] = (rel_p.z / 40.0) as f32;
                    oo[3] = (rel_v.x / 40.0) as f32;
                    oo[4] = (rel_v.y / 40.0) as f32;
                    oo[5] = (rel_v.z / 40.0) as f32;
                    oo[6] = (o_pos.x * inv_w) as f32;
                    oo[7] = (o_pos.y * inv_h) as f32;
                    oo[8] = (o_pos.z * inv_l) as f32;
                    oo[9] = (o_vel.x / 23.0) as f32;
                    oo[10] = (o_vel.y / 23.0) as f32;
                    oo[11] = (o_vel.z / 23.0) as f32;
                    oo[12] = o_fwd.x as f32;
                    oo[13] = o_fwd.y as f32;
                    oo[14] = o_fwd.z as f32;
                    oo[15] = (o.boost / 100.0) as f32;
                    oo[16] = if o.team == car.team { 1.0 } else { -1.0 };
                    oo[17] = if o.on_ground { 1.0 } else { 0.0 };
                }
                slot += 1;
            }

            // 6 nearest boost pads (6 x 4)
            let mut pad_dists: [(f64, usize); 40] = [(1e9, 0); 40];
            let np = w.pads.len().min(40);
            for pi in 0..np {
                let p = &w.pads[pi];
                let dx = p.pos.x * s - c_pos.x;
                let dz = p.pos.z * s - c_pos.z;
                pad_dists[pi] = (dx * dx + dz * dz, pi);
            }
            let slice_p = &mut pad_dists[..np];
            slice_p.sort_by(|a, b| a.0.total_cmp(&b.0));
            for k in 0..NUM_PADS_OBS.min(np) {
                let p = &w.pads[slice_p[k].1];
                let base = (ci * NUM_PADS_OBS + k) * PAD_DIM;
                let po = &mut pads_out[base..base + PAD_DIM];
                po[0] = ((p.pos.x * s - c_pos.x) / 40.0) as f32;
                po[1] = ((p.pos.z * s - c_pos.z) / 40.0) as f32;
                po[2] = if p.big { 1.0 } else { 0.0 };
                po[3] = if p.active { 1.0 } else { 0.0 };
            }
        }
    }

    fn step_env(
        &mut self,
        cfg: &EnvConfig,
        actions: &[f32],
        rewards_out: &mut [f32],
        done_out: &mut u8,
        stats: &mut StepStats,
    ) {
        rewards_out.fill(0.0);
        *done_out = 0;

        let n_cars = self.world.cars.len().min(MAX_CARS);
        let ticks = cfg.ticks_per_step.max(1);
        let dt = self.world.phys.dt;
        let step_dt = dt * ticks as f64;
        let glz = self.world.shape.goal_line_z();

        let mut prev_boost = [0.0_f64; MAX_CARS];
        for ci in 0..n_cars {
            prev_boost[ci] = self.world.cars[ci].boost;
        }

        // Apply neural actions to non-bot cars
        for ci in 0..n_cars {
            if self.world.cars[ci].bot.is_none() {
                let a = &actions[ci * ACTION_DIM..(ci + 1) * ACTION_DIM];
                self.world.cars[ci].controls = Controls {
                    throttle: (a[0] as f64).clamp(-1.0, 1.0),
                    steer: (a[1] as f64).clamp(-1.0, 1.0),
                    pitch: (a[2] as f64).clamp(-1.0, 1.0),
                    yaw: (a[3] as f64).clamp(-1.0, 1.0),
                    roll: (a[4] as f64).clamp(-1.0, 1.0),
                    jump: a[5] > 0.5,
                    boost: a[6] > 0.5,
                    handbrake: a[7] > 0.5,
                    air_roll_free: false,
                    item: false,
                };
            }
        }

        let mut goal_team: i32 = -1;
        for _ in 0..ticks {
            self.world.step(dt);
            if self.world.kickoff && self.world.ball.pos.x.hypot(self.world.ball.pos.z) > 0.3 {
                self.world.kickoff = false;
            }
            for ev in &self.world.events {
                match ev {
                    WorldEvent::Hit(h) => {
                        let ci = h.car_idx;
                        if ci < n_cars {
                            stats.touches += 1;
                            let str_bonus = (h.strength / 30.0).min(1.0) as f32;
                            let air_bonus = (h.y / 10.0).clamp(0.0, 1.0) as f32;
                            rewards_out[ci] += cfg.w_touch * (1.0 + str_bonus + 0.8 * air_bonus);
                            if h.shot {
                                rewards_out[ci] += cfg.w_shot;
                            }
                            if h.save {
                                rewards_out[ci] += cfg.w_save * if h.epic { 1.5 } else { 1.0 };
                            }
                        }
                    }
                    WorldEvent::Demo {
                        victim_idx,
                        attacker_idx,
                        ..
                    } => {
                        if *attacker_idx < n_cars {
                            rewards_out[*attacker_idx] += cfg.w_demo;
                        }
                        if *victim_idx < n_cars {
                            rewards_out[*victim_idx] -= cfg.w_demo * 0.6;
                        }
                    }
                    _ => {}
                }
            }
            let scored_on = self.world.shape.in_goal(self.world.ball.pos, self.world.phys.ball_r);
            if scored_on == 0 || scored_on == 1 {
                goal_team = 1 - scored_on; // team that SCORED
                break;
            }
        }

        // Dense shaping rewards
        let bpos = self.world.ball.pos;
        let bvel = self.world.ball.vel;
        for ci in 0..n_cars {
            let car = &self.world.cars[ci];
            let ad = if car.team == 0 { 1.0 } else { -1.0 };

            // Ball velocity toward opponent's goal
            let to_goal = DVec3::new(-bpos.x, 0.0, ad * glz - bpos.z);
            let gl = to_goal.length().max(1.0);
            let v_ball_goal = bvel.dot(to_goal / gl) / 25.0;
            rewards_out[ci] += (cfg.w_ball_to_goal as f64 * step_dt * v_ball_goal) as f32;

            // Player velocity toward ball
            if !car.demolished {
                let to_ball = bpos - car.pos;
                let bl = to_ball.length().max(1.0);
                let v_car_ball = car.vel.dot(to_ball / bl) / 23.0;
                rewards_out[ci] += (cfg.w_player_to_ball as f64 * step_dt * v_car_ball) as f32;
            }

            // Boost pickup reward
            let db = (car.boost - prev_boost[ci]).max(0.0);
            if db > 1.0 {
                rewards_out[ci] += cfg.w_boost_pickup * (db / 100.0) as f32;
            }
        }

        self.step_count += 1;
        let timeout = self.step_count >= cfg.max_episode_steps;
        if goal_team >= 0 {
            if goal_team == 0 {
                stats.goals_blue += 1;
                if self.bot_opponent {
                    stats.bot_games_won += 1;
                }
            } else {
                stats.goals_orange += 1;
                if self.bot_opponent {
                    stats.bot_games_lost += 1;
                }
            }
            for ci in 0..n_cars {
                let scored = self.world.cars[ci].team as i32 == goal_team;
                rewards_out[ci] += if scored { cfg.w_goal } else { -cfg.w_goal };
            }
            *done_out = 1;
            stats.episodes_done += 1;
            self.reset(cfg);
        } else if timeout {
            *done_out = 1;
            stats.episodes_done += 1;
            self.reset(cfg);
        }
    }

    fn compute_expert_actions(&mut self, dt_step: f64, actions_out: &mut [f32]) {
        actions_out.fill(0.0);
        let n_cars = self.world.cars.len().min(MAX_CARS);
        self.world
            .pred
            .update(&self.world.ball, &self.world.shape, &self.world.phys);
        self.world.brain.update(
            dt_step,
            &self.world.cars,
            &self.world.ball,
            self.world.shape.goal_line_z(),
        );
        for ci in 0..n_cars {
            let saved = self.world.cars[ci].controls;
            if let Some(exp) = self.experts.get_mut(ci) {
                exp.update(ci, dt_step, &mut self.world);
            }
            let c = self.world.cars[ci].controls;
            let a = &mut actions_out[ci * ACTION_DIM..(ci + 1) * ACTION_DIM];
            a[0] = c.throttle as f32;
            a[1] = c.steer as f32;
            a[2] = c.pitch as f32;
            a[3] = c.yaw as f32;
            a[4] = c.roll as f32;
            a[5] = if c.jump { 1.0 } else { 0.0 };
            a[6] = if c.boost { 1.0 } else { 0.0 };
            a[7] = if c.handbrake { 1.0 } else { 0.0 };
            self.world.cars[ci].controls = saved;
        }
    }
}

pub struct VecEnv {
    cfg: EnvConfig,
    envs: Vec<SingleEnv>,
}

#[unsafe(no_mangle)]
pub extern "C" fn rl_env_create(cfg: EnvConfig) -> *mut VecEnv {
    let n = (cfg.num_envs as usize).max(1);
    let mut envs = Vec::with_capacity(n);
    for i in 0..n {
        let env_seed = cfg.seed.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        envs.push(SingleEnv::new(&cfg, env_seed));
    }
    Box::into_raw(Box::new(VecEnv { cfg, envs }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_destroy(ptr: *mut VecEnv) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_set_bot_curriculum(
    ptr: *mut VecEnv,
    bot_opponent_prob: f32,
    bot_diff: u32,
) {
    if ptr.is_null() {
        return;
    }
    let ve = unsafe { &mut *ptr };
    ve.cfg.bot_opponent_prob = bot_opponent_prob.clamp(0.0, 1.0);
    ve.cfg.bot_diff = bot_diff.min(4);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_reset_all(
    ptr: *mut VecEnv,
    self_out: *mut f32,
    ball_out: *mut f32,
    other_out: *mut f32,
    other_mask_out: *mut f32,
    pads_out: *mut f32,
    active_mask_out: *mut f32,
    team_ids_out: *mut i32,
) {
    let ve = unsafe { &mut *ptr };
    let n = ve.envs.len();
    let cfg = ve.cfg;

    let self_s = unsafe { slice::from_raw_parts_mut(self_out, n * MAX_CARS * SELF_DIM) };
    let ball_s = unsafe { slice::from_raw_parts_mut(ball_out, n * MAX_CARS * BALL_DIM) };
    let other_s =
        unsafe { slice::from_raw_parts_mut(other_out, n * MAX_CARS * OTHER_CARS * OTHER_DIM) };
    let omask_s = unsafe { slice::from_raw_parts_mut(other_mask_out, n * MAX_CARS * OTHER_CARS) };
    let pads_s =
        unsafe { slice::from_raw_parts_mut(pads_out, n * MAX_CARS * NUM_PADS_OBS * PAD_DIM) };
    let amask_s = unsafe { slice::from_raw_parts_mut(active_mask_out, n * MAX_CARS) };
    let tids_s = unsafe { slice::from_raw_parts_mut(team_ids_out, n * MAX_CARS) };

    for i in 0..n {
        ve.envs[i].reset(&cfg);
        ve.envs[i].write_obs(
            cfg.max_episode_steps,
            &mut self_s[i * MAX_CARS * SELF_DIM..(i + 1) * MAX_CARS * SELF_DIM],
            &mut ball_s[i * MAX_CARS * BALL_DIM..(i + 1) * MAX_CARS * BALL_DIM],
            &mut other_s[i * MAX_CARS * OTHER_CARS * OTHER_DIM
                ..(i + 1) * MAX_CARS * OTHER_CARS * OTHER_DIM],
            &mut omask_s[i * MAX_CARS * OTHER_CARS..(i + 1) * MAX_CARS * OTHER_CARS],
            &mut pads_s[i * MAX_CARS * NUM_PADS_OBS * PAD_DIM
                ..(i + 1) * MAX_CARS * NUM_PADS_OBS * PAD_DIM],
            &mut amask_s[i * MAX_CARS..(i + 1) * MAX_CARS],
            &mut tids_s[i * MAX_CARS..(i + 1) * MAX_CARS],
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_step(
    ptr: *mut VecEnv,
    actions_in: *const f32,
    self_out: *mut f32,
    ball_out: *mut f32,
    other_out: *mut f32,
    other_mask_out: *mut f32,
    pads_out: *mut f32,
    active_mask_out: *mut f32,
    team_ids_out: *mut i32,
    rewards_out: *mut f32,
    dones_out: *mut u8,
    stats_out: *mut StepStats,
) {
    let ve = unsafe { &mut *ptr };
    let n = ve.envs.len();
    let cfg = ve.cfg;

    let act_s = unsafe { slice::from_raw_parts(actions_in, n * MAX_CARS * ACTION_DIM) };
    let self_s = unsafe { slice::from_raw_parts_mut(self_out, n * MAX_CARS * SELF_DIM) };
    let ball_s = unsafe { slice::from_raw_parts_mut(ball_out, n * MAX_CARS * BALL_DIM) };
    let other_s =
        unsafe { slice::from_raw_parts_mut(other_out, n * MAX_CARS * OTHER_CARS * OTHER_DIM) };
    let omask_s = unsafe { slice::from_raw_parts_mut(other_mask_out, n * MAX_CARS * OTHER_CARS) };
    let pads_s =
        unsafe { slice::from_raw_parts_mut(pads_out, n * MAX_CARS * NUM_PADS_OBS * PAD_DIM) };
    let amask_s = unsafe { slice::from_raw_parts_mut(active_mask_out, n * MAX_CARS) };
    let tids_s = unsafe { slice::from_raw_parts_mut(team_ids_out, n * MAX_CARS) };
    let rew_s = unsafe { slice::from_raw_parts_mut(rewards_out, n * MAX_CARS) };
    let done_s = unsafe { slice::from_raw_parts_mut(dones_out, n) };

    let threads = (cfg.num_threads as usize).clamp(1, 32).min(n);
    let chunk = n.div_ceil(threads);
    let mut partial_stats = vec![StepStats::default(); threads];

    thread::scope(|sc| {
        let mut env_chunks = ve.envs.chunks_mut(chunk);
        let mut act_chunks = act_s.chunks(chunk * MAX_CARS * ACTION_DIM);
        let mut self_chunks = self_s.chunks_mut(chunk * MAX_CARS * SELF_DIM);
        let mut ball_chunks = ball_s.chunks_mut(chunk * MAX_CARS * BALL_DIM);
        let mut other_chunks = other_s.chunks_mut(chunk * MAX_CARS * OTHER_CARS * OTHER_DIM);
        let mut omask_chunks = omask_s.chunks_mut(chunk * MAX_CARS * OTHER_CARS);
        let mut pads_chunks = pads_s.chunks_mut(chunk * MAX_CARS * NUM_PADS_OBS * PAD_DIM);
        let mut amask_chunks = amask_s.chunks_mut(chunk * MAX_CARS);
        let mut tids_chunks = tids_s.chunks_mut(chunk * MAX_CARS);
        let mut rew_chunks = rew_s.chunks_mut(chunk * MAX_CARS);
        let mut done_chunks = done_s.chunks_mut(chunk);

        for ps in &mut partial_stats {
            let Some(ec) = env_chunks.next() else { break };
            let ac = act_chunks.next().unwrap();
            let sc_self = self_chunks.next().unwrap();
            let sc_ball = ball_chunks.next().unwrap();
            let sc_other = other_chunks.next().unwrap();
            let sc_omask = omask_chunks.next().unwrap();
            let sc_pads = pads_chunks.next().unwrap();
            let sc_amask = amask_chunks.next().unwrap();
            let sc_tids = tids_chunks.next().unwrap();
            let sc_rew = rew_chunks.next().unwrap();
            let sc_done = done_chunks.next().unwrap();

            sc.spawn(move || {
                for i in 0..ec.len() {
                    ec[i].step_env(
                        &cfg,
                        &ac[i * MAX_CARS * ACTION_DIM..(i + 1) * MAX_CARS * ACTION_DIM],
                        &mut sc_rew[i * MAX_CARS..(i + 1) * MAX_CARS],
                        &mut sc_done[i],
                        ps,
                    );
                    ec[i].write_obs(
                        cfg.max_episode_steps,
                        &mut sc_self[i * MAX_CARS * SELF_DIM..(i + 1) * MAX_CARS * SELF_DIM],
                        &mut sc_ball[i * MAX_CARS * BALL_DIM..(i + 1) * MAX_CARS * BALL_DIM],
                        &mut sc_other[i * MAX_CARS * OTHER_CARS * OTHER_DIM
                            ..(i + 1) * MAX_CARS * OTHER_CARS * OTHER_DIM],
                        &mut sc_omask[i * MAX_CARS * OTHER_CARS..(i + 1) * MAX_CARS * OTHER_CARS],
                        &mut sc_pads[i * MAX_CARS * NUM_PADS_OBS * PAD_DIM
                            ..(i + 1) * MAX_CARS * NUM_PADS_OBS * PAD_DIM],
                        &mut sc_amask[i * MAX_CARS..(i + 1) * MAX_CARS],
                        &mut sc_tids[i * MAX_CARS..(i + 1) * MAX_CARS],
                    );
                }
            });
        }
    });

    if !stats_out.is_null() {
        let mut total = StepStats::default();
        for ps in partial_stats {
            total.goals_blue += ps.goals_blue;
            total.goals_orange += ps.goals_orange;
            total.episodes_done += ps.episodes_done;
            total.touches += ps.touches;
            total.bot_games_won += ps.bot_games_won;
            total.bot_games_lost += ps.bot_games_lost;
        }
        unsafe {
            *stats_out = total;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_expert_actions(ptr: *mut VecEnv, actions_out: *mut f32) {
    let ve = unsafe { &mut *ptr };
    let n = ve.envs.len();
    let cfg = ve.cfg;
    let dt_step = (1.0 / 120.0) * cfg.ticks_per_step.max(1) as f64;
    let act_s = unsafe { slice::from_raw_parts_mut(actions_out, n * MAX_CARS * ACTION_DIM) };

    let threads = (cfg.num_threads as usize).clamp(1, 32).min(n);
    let chunk = n.div_ceil(threads);
    thread::scope(|sc| {
        let mut env_chunks = ve.envs.chunks_mut(chunk);
        let mut act_chunks = act_s.chunks_mut(chunk * MAX_CARS * ACTION_DIM);
        for _ in 0..threads {
            let Some(ec) = env_chunks.next() else { break };
            let ac = act_chunks.next().unwrap();
            sc.spawn(move || {
                for i in 0..ec.len() {
                    ec[i].compute_expert_actions(
                        dt_step,
                        &mut ac[i * MAX_CARS * ACTION_DIM..(i + 1) * MAX_CARS * ACTION_DIM],
                    );
                }
            });
        }
    });
}
