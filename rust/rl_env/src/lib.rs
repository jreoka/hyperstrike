use std::slice;
use std::thread;

use glam::DVec3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sim::neural::{
    self, ACTION_DIM, BALL_DIM, MAX_CARS, NUM_PADS_OBS, OTHER_CARS, OTHER_DIM, PAD_DIM, SELF_DIM,
};
use sim::{
    ARENAS, ArenaOpts, BotAi, Car, Controls, HITBOXES, KICKOFF_SPOTS, PERSONALITIES, PowerEvent,
    PowerSystem, World, WorldEvent,
};

/// `dones_out` codes: episode ended by a goal (true terminal).
pub const DONE_TERMINAL: u8 = 1;
/// `dones_out` codes: episode cut off by the step limit (bootstrap the value).
pub const DONE_TRUNCATED: u8 = 2;

/// Minimum gap between ball contacts for a touch to be rewarded (unless a different car touched
/// in between). Stops pinning / dribbling from farming a reward every 120Hz tick.
const TOUCH_REWARD_GAP_S: f64 = 0.5;

/// Mayhem power-up intervals offered in the game UI (seconds).
const MAYHEM_INTERVALS: [f64; 3] = [6.0, 10.0, 15.0];

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
    // Mayhem
    pub mayhem_prob: f32,
    pub w_power_fail: f32,
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
    pub power_uses: u32,
    pub mayhem_goals: u32,
}

impl StepStats {
    fn add(&mut self, o: &StepStats) {
        self.goals_blue += o.goals_blue;
        self.goals_orange += o.goals_orange;
        self.episodes_done += o.episodes_done;
        self.touches += o.touches;
        self.bot_games_won += o.bot_games_won;
        self.bot_games_lost += o.bot_games_lost;
        self.power_uses += o.power_uses;
        self.mayhem_goals += o.mayhem_goals;
    }
}

/// Mutable output slices for one environment.
struct ObsOut<'a> {
    self_o: &'a mut [f32],
    ball_o: &'a mut [f32],
    other_o: &'a mut [f32],
    omask_o: &'a mut [f32],
    pads_o: &'a mut [f32],
    amask_o: &'a mut [f32],
    tids_o: &'a mut [i32],
}

struct SingleEnv {
    world: World,
    team_size: usize,
    step_count: u32,
    bot_opponent: bool,
    rng: StdRng,
    // Expert CHAMP bots used on demand for BC warmstart
    experts: Vec<BotAi>,
    // Touch-reward gating
    last_contact_t: [f64; MAX_CARS],
    last_rewarded_toucher: Option<usize>,
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
            last_contact_t: [-1e9; MAX_CARS],
            last_rewarded_toucher: None,
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
        if self.rng.r#gen::<f32>() < cfg.mayhem_prob {
            let interval = MAYHEM_INTERVALS[self.rng.gen_range(0..MAYHEM_INTERVALS.len())];
            world.powers = Some(PowerSystem::new(interval));
        }
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
        Self::place_kickoff(&mut world, &mut self.rng);
        self.world = world;
        self.team_size = team_size;
        self.step_count = 0;
        self.bot_opponent = bot_opponent;
        self.experts = experts;
        self.last_contact_t = [-1e9; MAX_CARS];
        self.last_rewarded_toucher = None;
    }

    fn place_kickoff(world: &mut World, rng: &mut StdRng) {
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
        if let Some(mut powers) = world.powers.take() {
            powers.reset(world);
            world.powers = Some(powers);
        }
        world.brain.choose_kickers(&world.cars, &world.ball);
    }

    /// Writes observations via `sim::neural::extract_obs` — the exact function used at inference.
    fn write_obs(&self, out: ObsOut<'_>) {
        out.self_o.fill(0.0);
        out.ball_o.fill(0.0);
        out.other_o.fill(0.0);
        out.omask_o.fill(0.0);
        out.pads_o.fill(0.0);
        out.amask_o.fill(0.0);
        out.tids_o.fill(0);

        let w = &self.world;
        let n_cars = w.cars.len().min(MAX_CARS);
        for ci in 0..n_cars {
            let car = &w.cars[ci];
            out.tids_o[ci] = car.team as i32;
            out.amask_o[ci] = if car.bot.is_none() { 1.0 } else { 0.0 };

            let obs = neural::extract_obs(ci, w);
            out.self_o[ci * SELF_DIM..(ci + 1) * SELF_DIM].copy_from_slice(&obs.self_obs);
            out.ball_o[ci * BALL_DIM..(ci + 1) * BALL_DIM].copy_from_slice(&obs.ball_obs);
            for k in 0..OTHER_CARS {
                let base = (ci * OTHER_CARS + k) * OTHER_DIM;
                out.other_o[base..base + OTHER_DIM].copy_from_slice(&obs.other_obs[k]);
                out.omask_o[ci * OTHER_CARS + k] = if obs.other_mask[k] { 1.0 } else { 0.0 };
            }
            for k in 0..NUM_PADS_OBS {
                let base = (ci * NUM_PADS_OBS + k) * PAD_DIM;
                out.pads_o[base..base + PAD_DIM].copy_from_slice(&obs.pads_obs[k]);
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
        let mayhem = self.world.powers.is_some();

        let mut prev_boost = [0.0_f64; MAX_CARS];
        for ci in 0..n_cars {
            prev_boost[ci] = self.world.cars[ci].boost;
        }

        // Neural controls for non-bot cars. Re-applied every tick, exactly like live inference
        // (which returns its cached controls each tick) — some powers (SCRAMBLE) mutate
        // `controls` in place per tick.
        let mut neural_ctrl: [Option<Controls>; MAX_CARS] = [None; MAX_CARS];
        for ci in 0..n_cars {
            if self.world.cars[ci].bot.is_none() {
                neural_ctrl[ci] = Some(neural::controls_from_actions(
                    &actions[ci * ACTION_DIM..(ci + 1) * ACTION_DIM],
                ));
            }
        }

        let mut goal_team: i32 = -1;
        for _ in 0..ticks {
            for ci in 0..n_cars {
                if let Some(c) = neural_ctrl[ci] {
                    self.world.cars[ci].controls = c;
                }
            }
            self.world.step(dt);
            if self.world.kickoff && self.world.ball.pos.x.hypot(self.world.ball.pos.z) > 0.3 {
                self.world.kickoff = false;
            }
            let now = self.world.time;
            for ev in &self.world.events {
                match ev {
                    WorldEvent::Hit(h) => {
                        let ci = h.car_idx;
                        if ci < n_cars {
                            let gap = now - self.last_contact_t[ci];
                            self.last_contact_t[ci] = now;
                            let fresh = gap > TOUCH_REWARD_GAP_S
                                || self.last_rewarded_toucher != Some(ci);
                            if fresh {
                                self.last_rewarded_toucher = Some(ci);
                                stats.touches += 1;
                                let str_bonus = (h.strength / 30.0).min(1.0) as f32;
                                let air_bonus = (h.y / 10.0).clamp(0.0, 1.0) as f32;
                                rewards_out[ci] +=
                                    cfg.w_touch * (1.0 + str_bonus + 0.8 * air_bonus);
                            }
                            // Shot/save flags are already edge-like (judged on ball trajectory).
                            if h.shot && fresh {
                                rewards_out[ci] += cfg.w_shot;
                            }
                            if h.save && fresh {
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
                    WorldEvent::Power(PowerEvent::Use { car_idx, .. }) => {
                        if *car_idx < n_cars {
                            stats.power_uses += 1;
                        }
                    }
                    WorldEvent::Power(PowerEvent::Fail { car_idx, .. }) => {
                        // Tiny cost for mashing the item button with no valid target.
                        if *car_idx < n_cars {
                            rewards_out[*car_idx] -= cfg.w_power_fail;
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

            // Ball velocity toward opponent's goal (zero-sum between teams)
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

            // Boost pickup reward (ignore NITRO's free refill)
            let db = (car.boost - prev_boost[ci]).max(0.0);
            if db > 1.0 && car.nitro_t <= 0.0 {
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
            if mayhem {
                stats.mayhem_goals += 1;
            }
            for ci in 0..n_cars {
                let scored = self.world.cars[ci].team as i32 == goal_team;
                rewards_out[ci] += if scored { cfg.w_goal } else { -cfg.w_goal };
            }
            *done_out = DONE_TERMINAL;
            stats.episodes_done += 1;
            self.reset(cfg);
        } else if timeout {
            *done_out = DONE_TRUNCATED;
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
            neural::actions_from_controls(&c, &mut actions_out[ci * ACTION_DIM..(ci + 1) * ACTION_DIM]);
            self.world.cars[ci].controls = saved;
        }
    }
}

pub struct VecEnv {
    cfg: EnvConfig,
    envs: Vec<SingleEnv>,
}

/// Raw output pointers for a whole vector of envs.
#[derive(Clone, Copy)]
struct OutPtrs {
    self_out: *mut f32,
    ball_out: *mut f32,
    other_out: *mut f32,
    omask_out: *mut f32,
    pads_out: *mut f32,
    amask_out: *mut f32,
    tids_out: *mut i32,
}

struct OutSlices<'a> {
    self_s: &'a mut [f32],
    ball_s: &'a mut [f32],
    other_s: &'a mut [f32],
    omask_s: &'a mut [f32],
    pads_s: &'a mut [f32],
    amask_s: &'a mut [f32],
    tids_s: &'a mut [i32],
}

impl OutPtrs {
    unsafe fn slices<'a>(self, n: usize) -> OutSlices<'a> {
        unsafe {
            OutSlices {
                self_s: slice::from_raw_parts_mut(self.self_out, n * MAX_CARS * SELF_DIM),
                ball_s: slice::from_raw_parts_mut(self.ball_out, n * MAX_CARS * BALL_DIM),
                other_s: slice::from_raw_parts_mut(
                    self.other_out,
                    n * MAX_CARS * OTHER_CARS * OTHER_DIM,
                ),
                omask_s: slice::from_raw_parts_mut(self.omask_out, n * MAX_CARS * OTHER_CARS),
                pads_s: slice::from_raw_parts_mut(
                    self.pads_out,
                    n * MAX_CARS * NUM_PADS_OBS * PAD_DIM,
                ),
                amask_s: slice::from_raw_parts_mut(self.amask_out, n * MAX_CARS),
                tids_s: slice::from_raw_parts_mut(self.tids_out, n * MAX_CARS),
            }
        }
    }
}

impl<'a> OutSlices<'a> {
    /// Output slices for env `i` (relative to this chunk).
    fn env(&mut self, i: usize) -> ObsOut<'_> {
        ObsOut {
            self_o: &mut self.self_s[i * MAX_CARS * SELF_DIM..(i + 1) * MAX_CARS * SELF_DIM],
            ball_o: &mut self.ball_s[i * MAX_CARS * BALL_DIM..(i + 1) * MAX_CARS * BALL_DIM],
            other_o: &mut self.other_s[i * MAX_CARS * OTHER_CARS * OTHER_DIM
                ..(i + 1) * MAX_CARS * OTHER_CARS * OTHER_DIM],
            omask_o: &mut self.omask_s[i * MAX_CARS * OTHER_CARS..(i + 1) * MAX_CARS * OTHER_CARS],
            pads_o: &mut self.pads_s[i * MAX_CARS * NUM_PADS_OBS * PAD_DIM
                ..(i + 1) * MAX_CARS * NUM_PADS_OBS * PAD_DIM],
            amask_o: &mut self.amask_s[i * MAX_CARS..(i + 1) * MAX_CARS],
            tids_o: &mut self.tids_s[i * MAX_CARS..(i + 1) * MAX_CARS],
        }
    }

    /// Split into per-thread chunks of `chunk` envs.
    fn split(self, chunk: usize) -> Vec<OutSlices<'a>> {
        let mut out = Vec::new();
        let mut a = self.self_s.chunks_mut(chunk * MAX_CARS * SELF_DIM);
        let mut b = self.ball_s.chunks_mut(chunk * MAX_CARS * BALL_DIM);
        let mut c = self.other_s.chunks_mut(chunk * MAX_CARS * OTHER_CARS * OTHER_DIM);
        let mut d = self.omask_s.chunks_mut(chunk * MAX_CARS * OTHER_CARS);
        let mut e = self.pads_s.chunks_mut(chunk * MAX_CARS * NUM_PADS_OBS * PAD_DIM);
        let mut f = self.amask_s.chunks_mut(chunk * MAX_CARS);
        let mut g = self.tids_s.chunks_mut(chunk * MAX_CARS);
        while let Some(self_s) = a.next() {
            out.push(OutSlices {
                self_s,
                ball_s: b.next().unwrap(),
                other_s: c.next().unwrap(),
                omask_s: d.next().unwrap(),
                pads_s: e.next().unwrap(),
                amask_s: f.next().unwrap(),
                tids_s: g.next().unwrap(),
            });
        }
        out
    }
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

/// Layout check so Python can refuse to run against a stale DLL.
/// Writes [SELF_DIM, BALL_DIM, OTHER_DIM, PAD_DIM, ACTION_DIM, MAX_CARS, NUM_PADS_OBS, sizeof(EnvConfig)].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rl_env_layout(out: *mut u32) {
    let v = [
        SELF_DIM,
        BALL_DIM,
        OTHER_DIM,
        PAD_DIM,
        ACTION_DIM,
        MAX_CARS,
        NUM_PADS_OBS,
        std::mem::size_of::<EnvConfig>(),
    ];
    let o = unsafe { slice::from_raw_parts_mut(out, v.len()) };
    for (d, s) in o.iter_mut().zip(v) {
        *d = s as u32;
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
pub unsafe extern "C" fn rl_env_set_mayhem_prob(ptr: *mut VecEnv, mayhem_prob: f32) {
    if ptr.is_null() {
        return;
    }
    let ve = unsafe { &mut *ptr };
    ve.cfg.mayhem_prob = mayhem_prob.clamp(0.0, 1.0);
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
    let ptrs = OutPtrs {
        self_out,
        ball_out,
        other_out,
        omask_out: other_mask_out,
        pads_out,
        amask_out: active_mask_out,
        tids_out: team_ids_out,
    };
    let mut out = unsafe { ptrs.slices(n) };
    for i in 0..n {
        ve.envs[i].reset(&cfg);
        ve.envs[i].write_obs(out.env(i));
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
    let rew_s = unsafe { slice::from_raw_parts_mut(rewards_out, n * MAX_CARS) };
    let done_s = unsafe { slice::from_raw_parts_mut(dones_out, n) };
    let ptrs = OutPtrs {
        self_out,
        ball_out,
        other_out,
        omask_out: other_mask_out,
        pads_out,
        amask_out: active_mask_out,
        tids_out: team_ids_out,
    };
    let out = unsafe { ptrs.slices(n) };

    let threads = if n <= 16 {
        1
    } else {
        (cfg.num_threads as usize).clamp(1, 32).min(n / 4).max(1)
    };
    let chunk = n.div_ceil(threads);

    let run_chunk = |envs: &mut [SingleEnv],
                     acts: &[f32],
                     rews: &mut [f32],
                     dones: &mut [u8],
                     mut obs: OutSlices<'_>|
     -> StepStats {
        let mut ps = StepStats::default();
        for i in 0..envs.len() {
            envs[i].step_env(
                &cfg,
                &acts[i * MAX_CARS * ACTION_DIM..(i + 1) * MAX_CARS * ACTION_DIM],
                &mut rews[i * MAX_CARS..(i + 1) * MAX_CARS],
                &mut dones[i],
                &mut ps,
            );
            envs[i].write_obs(obs.env(i));
        }
        ps
    };

    let mut total = StepStats::default();
    if threads == 1 {
        total = run_chunk(&mut ve.envs, act_s, rew_s, done_s, out);
    } else {
        let obs_chunks = out.split(chunk);
        let results: Vec<StepStats> = thread::scope(|sc| {
            let handles: Vec<_> = ve
                .envs
                .chunks_mut(chunk)
                .zip(act_s.chunks(chunk * MAX_CARS * ACTION_DIM))
                .zip(rew_s.chunks_mut(chunk * MAX_CARS))
                .zip(done_s.chunks_mut(chunk))
                .zip(obs_chunks)
                .map(|((((ec, ac), rc), dc), oc)| sc.spawn(move || run_chunk(ec, ac, rc, dc, oc)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for ps in &results {
            total.add(ps);
        }
    }

    if !stats_out.is_null() {
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
        for (ec, ac) in ve
            .envs
            .chunks_mut(chunk)
            .zip(act_s.chunks_mut(chunk * MAX_CARS * ACTION_DIM))
        {
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
