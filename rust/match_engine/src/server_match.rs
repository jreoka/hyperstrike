use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use indexmap::IndexMap;
use rand::Rng;
use serde_json::{Value, json};
use server_core::auth::now_ms;
use server_core::protocol::{
    LobbySettings, MatchCreateOptions, MatchEventSink, MatchStatsEntry, SnapshotPayload,
};
use sim::{
    ARENAS, ArenaOpts, BOT_NAMES, BotAi, Car, Controls, HITBOXES, KICKOFF_SPOTS, PERSONALITIES,
    PhysConfig, PowerEvent, PowerSystem, ShotTarget, SnapshotInfo, ThreeVec3Ext, World, WorldEvent,
    kickoff_sets_for, net_encode_snapshot, net_pack_controls, net_patch_snapshot, pick, yaw_toward,
};

pub const TICKS_PER_INPUT: usize = 2;
pub const MAX_AHEAD_SLOTS: usize = 10;
pub const SKIP_TO_SLOTS: usize = 3;
pub const MAX_FUTURE_SEQ: i64 = 240;
pub const IDLE_CLEAR_SLOTS: u32 = 30;
pub const COUNTDOWN_S: f64 = 3.0;
pub const GOAL_PAUSE_S: f64 = 3.4;
pub const REPLAY_MAX_S: f64 = 12.0;

#[inline]
fn r3(v: f64) -> f64 {
    if v.is_finite() {
        let r = (v * 1000.0).round() / 1000.0;
        if r == 0.0 { 0.0 } else { r }
    } else {
        0.0
    }
}

#[inline]
fn r4(v: f64) -> f64 {
    if v.is_finite() {
        let r = (v * 10000.0).round() / 10000.0;
        if r == 0.0 { 0.0 } else { r }
    } else {
        0.0
    }
}

#[inline]
fn clamp1(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub fn unpack_controls(arr: &[f64; 6], into: &mut Controls) {
    into.throttle = clamp1(arr[0]);
    into.steer = clamp1(arr[1]);
    into.pitch = clamp1(arr[2]);
    into.yaw = clamp1(arr[3]);
    into.roll = clamp1(arr[4]);
    let bits = arr[5] as i32;
    into.jump = (bits & 1) != 0;
    into.boost = (bits & 2) != 0;
    into.handbrake = (bits & 4) != 0;
    into.air_roll_free = (bits & 8) != 0;
    into.item = (bits & 16) != 0;
}

pub fn parse_packed_controls(arr: &[Value]) -> Option<[f64; 6]> {
    if arr.len() != 6 {
        return None;
    }
    let mut out = [0.0; 6];
    for (i, item) in arr[..5].iter().enumerate() {
        let v = item.as_f64()?;
        if !v.is_finite() || !(-1.001..=1.001).contains(&v) {
            return None;
        }
        out[i] = v;
    }
    let bits = arr[5].as_i64().or_else(|| {
        let f = arr[5].as_f64()?;
        if f.is_finite() && f.fract() == 0.0 {
            Some(f as i64)
        } else {
            None
        }
    })?;
    if !(0..=31).contains(&bits) {
        return None;
    }
    out[5] = bits as f64;
    Some(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchState {
    Idle,
    Countdown,
    Play,
    Goal,
    Ended,
}

impl MatchState {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchState::Idle => "idle",
            MatchState::Countdown => "countdown",
            MatchState::Play => "play",
            MatchState::Goal => "goal",
            MatchState::Ended => "ended",
        }
    }
}

struct HumanState {
    car_idx: usize,
    buf: HashMap<i64, [f64; 6]>,
    next_seq: i64,
    applied: [f64; 6],
    ack: i64,
    depth: usize,
    starves: u32,
    idle: u32,
    edge: i32,
}

impl HumanState {
    fn new(car_idx: usize) -> Self {
        Self {
            car_idx,
            buf: HashMap::new(),
            next_seq: 0,
            applied: [0.0; 6],
            ack: 0,
            depth: 0,
            starves: 0,
            idle: 0,
            edge: 0,
        }
    }
}

pub struct ServerMatch {
    pub settings: LobbySettings,
    pub world: World,
    pub car_info: Vec<Value>,
    pub created_at: i64,
    pub finished: bool,
    pub paused: bool,
    pub running: bool,
    callbacks: Arc<dyn MatchEventSink>,
    humans: IndexMap<String, HumanState>,
    car_user_ids: Vec<Option<String>>,
    car_explosions: Vec<i32>,
    car_kick: Vec<[f64; 4]>,
    replay_done: HashSet<String>,
    score: [i32; 2],
    time_left: f64,
    unlimited: bool,
    overtime: bool,
    ot_time: f64,
    timer_running: bool,
    match_over: bool,
    state: MatchState,
    state_t: f64,
    time_scale: f64,
    sim_acc: f64,
    slot: u64,
    events: Vec<Value>,
    last_shot_car: Option<usize>,
    last_shot_t: f64,
    stats_dirty: bool,
    last_instant: Instant,
    acc: f64,
    slot_sec: f64,
}

impl ServerMatch {
    pub fn new(opts: MatchCreateOptions) -> Self {
        let s = opts.settings;
        let mut phys = PhysConfig::default();
        if s.mutator == "lowgrav" {
            phys.g = 3.9;
        }
        if s.mutator == "fastball" {
            phys.ball_max = 78.0;
        }
        let arena_idx = usize::try_from(s.arena).unwrap_or(0);
        let def = ARENAS.get(arena_idx).copied().unwrap_or(ARENAS[0]);
        let arena_opts = ArenaOpts {
            layout: if s.layout.is_empty() {
                "default".to_string()
            } else {
                s.layout.clone()
            },
            scale: if s.team_size >= 4 { 1.12 } else { 1.0 },
        };
        let mut world = World::with_phys_and_rng(def, &arena_opts, phys, {
            use rand::SeedableRng;
            rand::rngs::StdRng::from_entropy()
        });
        if s.boost_mode == "unlimited" {
            world.unlimited_boost = true;
        }
        if s.boost_mode == "recharge" {
            world.boost_recharge = true;
        }
        if s.mode == "mayhem" {
            let interval = if s.interval > 0 {
                f64::from(s.interval)
            } else {
                10.0
            };
            world.powers = Some(PowerSystem::new(interval));
        }

        let mut humans = IndexMap::new();
        let mut car_info = Vec::new();
        let mut car_user_ids = Vec::new();
        let mut car_explosions = Vec::new();
        let mut car_kick = Vec::new();
        let mut bot_idx: usize = 0;

        for team in [0_u8, 1_u8] {
            let team_i32 = i32::from(team);
            let team_players: Vec<_> = opts
                .players
                .iter()
                .filter(|p| p.team == team_i32)
                .collect();
            for p in &team_players {
                let body = usize::try_from(p.car.body).unwrap_or(0) % HITBOXES.len();
                let name = p.username.to_uppercase();
                let idx = world.add_car(Car::new(world.cars.len(), team, &name, body));
                humans.insert(p.user_id.clone(), HumanState::new(idx));
                car_user_ids.push(Some(p.user_id.clone()));
                car_explosions.push(p.car.explosion);
                car_kick.push([0.0; 4]);
                car_info.push(json!({
                    "userId": p.user_id,
                    "botId": Value::Null,
                    "team": team,
                    "name": name,
                    "car": p.car,
                }));
            }
            if s.bot_fill {
                let target_size = usize::try_from(s.team_size.max(0)).unwrap_or(0);
                for i in team_players.len()..target_size {
                    let name = BOT_NAMES[bot_idx % BOT_NAMES.len()];
                    bot_idx += 1;
                    let body = world.rng.gen_range(0..HITBOXES.len());
                    let idx = world.add_car(Car::new(world.cars.len(), team, name, body));
                    let bot_id = format!("bot_{team}_{i}");
                    let diff = if !team_players.is_empty() {
                        opts.teammate_diff.unwrap_or(s.bot_diff)
                    } else {
                        s.bot_diff
                    };
                    let diff_idx = usize::try_from(diff.max(0)).unwrap_or(1);
                    let pers_idx = bot_idx % PERSONALITIES.len();
                    let bot = BotAi::new_with_rng(diff_idx, pers_idx, &mut world.rng);
                    world.cars[idx].bot = Some(bot);
                    let design = world.cars[idx].design;
                    car_user_ids.push(None);
                    car_explosions.push(0);
                    car_kick.push([0.0; 4]);
                    car_info.push(json!({
                        "userId": Value::Null,
                        "botId": bot_id,
                        "team": team,
                        "name": name,
                        "car": Value::Null,
                        "body": design,
                    }));
                }
            }
        }

        let slot_sec = world.phys.dt * (TICKS_PER_INPUT as f64);
        let unlimited = s.duration == 0;
        let time_left = if s.duration > 0 {
            f64::from(s.duration)
        } else {
            300.0
        };

        Self {
            settings: s,
            world,
            car_info,
            created_at: now_ms(),
            finished: false,
            paused: false,
            running: false,
            callbacks: opts.callbacks,
            humans,
            car_user_ids,
            car_explosions,
            car_kick,
            replay_done: HashSet::new(),
            score: [0, 0],
            time_left,
            unlimited,
            overtime: false,
            ot_time: 0.0,
            timer_running: false,
            match_over: false,
            state: MatchState::Idle,
            state_t: 0.0,
            time_scale: 1.0,
            sim_acc: 0.0,
            slot: 0,
            events: Vec::new(),
            last_shot_car: None,
            last_shot_t: 0.0,
            stats_dirty: false,
            last_instant: Instant::now(),
            acc: 0.0,
            slot_sec,
        }
    }

    pub fn start(&mut self) {
        self.world.allow_demos = true;
        self.begin_kickoff();
        self.running = true;
        self.last_instant = Instant::now();
        self.acc = 0.0;
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn stop(&mut self) {
        self.finished = true;
        self.running = false;
    }

    pub fn step_clock(&mut self, now: Instant) {
        if !self.running || self.finished {
            return;
        }
        if self.paused {
            self.last_instant = now;
            self.acc = 0.0;
            return;
        }
        let elapsed = now.saturating_duration_since(self.last_instant).as_secs_f64();
        self.acc += elapsed;
        self.last_instant = now;
        let mut n = 0;
        let mut should_snap = false;
        while self.acc >= self.slot_sec && n < 8 && !self.finished {
            if self.run_slot() {
                should_snap = true;
            }
            self.acc -= self.slot_sec;
            n += 1;
        }
        if n >= 8 {
            self.acc = 0.0;
        }
        if should_snap && !self.finished {
            self.send_snapshots();
        }
    }

    pub fn ms_until_next_slot(&self, now: Instant) -> f64 {
        if !self.running || self.finished {
            return f64::INFINITY;
        }
        if self.paused {
            return 50.0;
        }
        let elapsed = now.saturating_duration_since(self.last_instant).as_secs_f64();
        ((self.slot_sec - self.acc - elapsed) * 1000.0).max(0.0)
    }

    pub fn push_input(&mut self, user_id: &str, seq: i64, packed: [f64; 6]) -> bool {
        let Some(h) = self.humans.get_mut(user_id) else {
            return false;
        };
        if seq <= 0 {
            return false;
        }
        if !self.running {
            return true;
        }
        if h.next_seq == 0 {
            h.next_seq = seq;
        }
        if seq < h.next_seq {
            h.edge |= (packed[5] as i32) & 17;
            return true;
        }
        if seq > h.next_seq + MAX_FUTURE_SEQ {
            return true;
        }
        h.buf.insert(seq, packed);
        true
    }

    fn consume_input(h: &mut HumanState, in_play: bool) {
        if h.next_seq == 0 {
            return;
        }
        let mut depth = h.buf.len();
        if depth > MAX_AHEAD_SLOTS {
            while depth > SKIP_TO_SLOTS {
                if let Some(old) = h.buf.remove(&h.next_seq) {
                    h.edge |= (old[5] as i32) & 17;
                    depth -= 1;
                }
                h.next_seq += 1;
            }
        }
        h.depth = depth;
        if let Some(inp) = h.buf.remove(&h.next_seq) {
            h.applied = inp;
            h.idle = 0;
        } else {
            if in_play {
                h.starves = h.starves.saturating_add(1);
            }
            h.idle = h.idle.saturating_add(1);
            if h.idle > IDLE_CLEAR_SLOTS {
                h.applied = [0.0; 6];
            }
        }
        if h.edge != 0 {
            let bits = (h.applied[5] as i32) | h.edge;
            h.applied[5] = bits as f64;
            h.edge = 0;
        }
        h.ack = h.next_seq;
        h.next_seq += 1;
    }

    pub fn mark_replay_done(&mut self, user_id: &str) {
        if self.state == MatchState::Goal && self.humans.contains_key(user_id) {
            self.replay_done.insert(user_id.to_string());
        }
    }

    pub fn remove_player(&mut self, user_id: &str) {
        let Some(h) = self.humans.shift_remove(user_id) else {
            return;
        };
        let diff_idx = usize::try_from(self.settings.bot_diff.max(0)).unwrap_or(1);
        let bot = BotAi::new_with_rng(diff_idx, 0, &mut self.world.rng);
        if let Some(car) = self.world.cars.get_mut(h.car_idx) {
            car.bot = Some(bot);
        }
    }

    pub fn run_slot(&mut self) -> bool {
        let in_play = self.state == MatchState::Play;
        for h in self.humans.values_mut() {
            Self::consume_input(h, in_play);
        }
        let dt = self.world.phys.dt;
        for _ in 0..TICKS_PER_INPUT {
            for h in self.humans.values() {
                if let Some(car) = self.world.cars.get_mut(h.car_idx) {
                    unpack_controls(&h.applied, &mut car.controls);
                }
            }
            self.tick(dt);
        }
        self.slot += 1;
        !self.finished
    }

    fn tick(&mut self, dt: f64) {
        self.pre_update(dt);
        if self.finished || self.state == MatchState::Ended {
            return;
        }
        self.sim_acc += self.time_scale;
        while self.sim_acc >= 1.0 {
            self.sim_acc -= 1.0;
            self.world.step(dt);
            self.collect_events();
            if self.state == MatchState::Play {
                let gt = self.world.shape.in_goal(self.world.ball.pos, self.world.phys.ball_r);
                if gt >= 0 {
                    self.on_goal(gt as usize);
                }
            }
        }
    }

    fn set_state(&mut self, s: MatchState) {
        self.state = s;
        self.state_t = 0.0;
    }

    fn pre_update(&mut self, dt: f64) {
        self.state_t += dt;
        match self.state {
            MatchState::Countdown => {
                if COUNTDOWN_S - self.state_t <= 0.0 {
                    self.world.frozen = false;
                    self.set_state(MatchState::Play);
                }
            }
            MatchState::Play => {
                if self.world.kickoff && (self.world.ball.last_touch.is_some() || self.state_t > 6.0)
                {
                    self.world.kickoff = false;
                }
                if !self.timer_running
                    && (self.world.ball.last_touch.is_some() || self.state_t > 4.0)
                {
                    self.timer_running = true;
                }
                if self.timer_running && !self.unlimited {
                    if self.overtime {
                        self.ot_time += dt;
                    } else {
                        self.time_left = (self.time_left - dt).max(0.0);
                        if self.time_left <= 0.0
                            && (self.world.ball.on_ground
                                || self.world.ball.pos.y < self.world.phys.ball_r + 0.15)
                        {
                            self.time_up();
                        }
                    }
                }
            }
            MatchState::Goal => {
                if self.state_t < 1.3 {
                    let t = ((self.state_t - 0.5) / 0.8).clamp(0.0, 1.0);
                    self.time_scale = 0.22 + (1.0 - 0.22) * (t * t * (3.0 - 2.0 * t));
                } else {
                    self.time_scale = 1.0;
                }
                if self.state_t > GOAL_PAUSE_S {
                    self.time_scale = 1.0;
                    if self.match_over {
                        self.finish();
                        return;
                    }
                    self.world.frozen = true;
                    let all_done = self
                        .humans
                        .keys()
                        .all(|uid| self.replay_done.contains(uid));
                    if all_done || self.state_t > GOAL_PAUSE_S + REPLAY_MAX_S {
                        self.begin_kickoff();
                    }
                }
            }
            MatchState::Idle | MatchState::Ended => {}
        }
    }

    fn time_up(&mut self) {
        if self.score[0] == self.score[1] {
            self.overtime = true;
            self.ot_time = 0.0;
            self.callbacks.broadcast(json!({ "type": "match_overtime" }));
            self.begin_kickoff();
        } else {
            self.match_over = true;
            self.finish();
        }
    }

    fn kickoff_positions(&mut self) {
        let team_size = usize::try_from(self.settings.team_size.clamp(1, 5)).unwrap_or(1);
        let sets = kickoff_sets_for(team_size);
        let mut order: Vec<usize> = pick(sets, &mut self.world.rng).to_vec();
        for i in (1..order.len()).rev() {
            let j = self.world.rng.gen_range(0..=i);
            order.swap(i, j);
        }
        for team in [0_u8, 1_u8] {
            let indices: Vec<usize> = self
                .world
                .cars
                .iter()
                .enumerate()
                .filter(|(_, c)| c.team == team)
                .map(|(i, _)| i)
                .collect();
            for (i, car_idx) in indices.into_iter().enumerate() {
                let spot_idx = order[i % order.len()];
                let spot = KICKOFF_SPOTS
                    .get(spot_idx)
                    .copied()
                    .unwrap_or(KICKOFF_SPOTS[4]);
                let (mut x, z) = self.world.shape.kickoff_pos(spot, team);
                if i >= order.len() {
                    x += ((i - order.len() + 1) as f64) * 4.0;
                }
                let yaw = yaw_toward(x, z, 0.0, 0.0);
                let car = &mut self.world.cars[car_idx];
                car.place(x, 0.35, z, yaw);
                car.boost = 33.0;
                car.demolished = false;
                if let Some(bot) = car.bot.as_mut() {
                    bot.reset();
                }
                self.car_kick[car_idx] = [r3(x), 0.35, r3(z), r4(yaw)];
            }
        }
    }

    fn begin_kickoff(&mut self) {
        let ball_r = self.world.phys.ball_r;
        self.world.ball.reset(0.0, ball_r, 0.0);
        self.world.ball.frozen = false;
        self.kickoff_positions();
        self.world.reset_pads();
        self.world.frozen = true;
        self.world.kickoff = true;
        self.world
            .brain
            .choose_kickers(&self.world.cars, &self.world.ball);
        if let Some(mut powers) = self.world.powers.take() {
            powers.reset(&mut self.world);
            self.world.powers = Some(powers);
        }
        self.set_state(MatchState::Countdown);
        self.timer_running = false;
        self.time_scale = 1.0;
        self.sim_acc = 0.0;
        self.callbacks.broadcast(json!({
            "type": "match_kickoff",
            "cars": self.car_kick,
            "overtime": self.overtime,
        }));
    }

    fn collect_events(&mut self) {
        let events = std::mem::take(&mut self.world.events);
        for ev in events {
            match ev {
                WorldEvent::Hit(hit) => {
                    if self.state == MatchState::Play {
                        let car = &mut self.world.cars[hit.car_idx];
                        if hit.save {
                            car.stats.saves += 1;
                            car.stats.score += if hit.epic { 75 } else { 50 };
                        }
                        if hit.shot
                            && (self.last_shot_car != Some(hit.car_idx)
                                || self.world.time - self.last_shot_t > 2.0)
                        {
                            car.stats.shots += 1;
                            car.stats.score += 20;
                            self.last_shot_car = Some(hit.car_idx);
                            self.last_shot_t = self.world.time;
                        }
                        car.stats.score += 2;
                    }
                }
                WorldEvent::Demo {
                    victim_idx,
                    attacker_idx,
                    pos,
                    overdrive,
                } => {
                    self.world.cars[attacker_idx].stats.score += 25;
                    self.events.push(json!({
                        "type": "demo",
                        "victim": { "$c": victim_idx },
                        "attacker": { "$c": attacker_idx },
                        "pos": { "$v": [r3(pos.x), r3(pos.y), r3(pos.z)] },
                        "overdrive": overdrive,
                    }));
                }
                WorldEvent::Respawn { car_idx } => {
                    self.events.push(json!({
                        "type": "respawn",
                        "car": { "$c": car_idx },
                    }));
                }
                WorldEvent::Power(pe) => {
                    self.events.push(serialize_power_event(&pe));
                }
                WorldEvent::Bounce(_)
                | WorldEvent::Car { .. }
                | WorldEvent::Bump { .. }
                | WorldEvent::Pickup { .. } => {}
            }
        }
    }

    fn on_goal(&mut self, goal_team: usize) {
        let scoring_team = 1 - goal_team;
        self.score[scoring_team] += 1;
        let hist = &self.world.ball.touch_history;
        let mut scorer: Option<usize> = None;
        let mut assist: Option<usize> = None;
        for i in (0..hist.len()).rev() {
            let h = hist[i];
            if self.world.cars[h.car_idx].team == scoring_team as u8
                && self.world.time - h.time < 8.0
            {
                scorer = Some(h.car_idx);
                for j in (0..i).rev() {
                    let a = hist[j];
                    if a.car_idx != h.car_idx
                        && self.world.cars[a.car_idx].team == scoring_team as u8
                        && h.time - a.time < 5.0
                    {
                        assist = Some(a.car_idx);
                        break;
                    }
                    if self.world.cars[a.car_idx].team != scoring_team as u8 {
                        break;
                    }
                }
                break;
            }
        }
        let own = self
            .world
            .ball
            .last_touch
            .is_some_and(|idx| self.world.cars[idx].team == goal_team as u8);
        if let Some(sc) = scorer {
            self.world.cars[sc].stats.goals += 1;
            self.world.cars[sc].stats.score += 100;
        }
        if let Some(asst) = assist {
            self.world.cars[asst].stats.assists += 1;
            self.world.cars[asst].stats.score += 50;
        }
        let speed = (self.world.ball.vel.length() * 3.6).round() as i64;
        let pos = self.world.ball.pos;
        for c in &mut self.world.cars {
            let d = c.pos.distance(pos);
            if d >= 26.0 || c.demolished {
                continue;
            }
            let k = 1.0 - d / 26.0;
            let kk = k.sqrt();
            let mut dir = c.pos - pos;
            dir.y = dir.y.max(0.0) + d * 0.35;
            if dir.length_squared() < 1e-4 {
                dir = DVec3::Y;
            }
            dir = dir.three_normalize();
            c.vel += dir * (12.0 + 28.0 * kk);
            c.vel.y += 3.0 + 8.0 * kk;
            let rx: f64 = self.world.rng.r#gen::<f64>() - 0.5;
            let ry: f64 = self.world.rng.r#gen::<f64>() - 0.5;
            let rz: f64 = self.world.rng.r#gen::<f64>() - 0.5;
            c.ang_vel += DVec3::new(rx, ry, rz) * (9.0 * kk);
            c.blast_t = 1.4;
            c.ground_lock = c.ground_lock.max(0.3);
        }
        if let Some(mut powers) = self.world.powers.take() {
            powers.clear_effects(&mut self.world);
            self.world.powers = Some(powers);
        }
        self.world.ball.frozen = true;
        self.world.ball.pos = DVec3::new(0.0, -50.0, 0.0);
        self.world.ball.vel = DVec3::ZERO;
        self.set_state(MatchState::Goal);
        self.world.kickoff = false;
        self.replay_done.clear();
        if self.overtime {
            self.match_over = true;
        }
        self.stats_dirty = true;

        let scorer_i = scorer.map(|i| i as i64).unwrap_or(-1);
        let assist_i = assist.map(|i| i as i64).unwrap_or(-1);
        let scorer_name = scorer
            .map(|i| self.world.cars[i].name.as_str())
            .unwrap_or("");
        let assist_name = assist
            .map(|i| self.world.cars[i].name.as_str())
            .unwrap_or("");
        let scorer_id = scorer.and_then(|i| self.car_user_ids[i].clone());
        let explosion = scorer.map(|i| self.car_explosions[i]).unwrap_or(0);

        self.callbacks.on_goal(json!({
            "scorer": scorer_i,
            "assist": assist_i,
            "goalTeam": goal_team,
            "scoringTeam": scoring_team,
            "score": self.score,
            "scorerName": scorer_name,
            "assistName": assist_name,
            "scorerId": scorer_id,
            "own": own && scorer.is_none(),
            "speed": speed,
            "pos": [r3(pos.x), r3(pos.y), r3(pos.z)],
            "explosion": explosion,
        }));
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.set_state(MatchState::Ended);
        self.world.frozen = true;
        self.send_snapshots();
        self.stop();
        let mut stats = HashMap::new();
        for (i, c) in self.world.cars.iter().enumerate() {
            if let Some(uid) = &self.car_user_ids[i] {
                stats.insert(
                    uid.clone(),
                    MatchStatsEntry {
                        goals: c.stats.goals as u64,
                        assists: c.stats.assists as u64,
                        saves: c.stats.saves as u64,
                        shots: c.stats.shots as u64,
                        demos: c.stats.demos as u64,
                    },
                );
            }
        }
        self.callbacks.on_end(self.score, stats);
    }

    fn powers_snap(&self) -> Option<Value> {
        let p = self.world.powers.as_ref()?;
        let b = &self.world.ball;
        let cars: Vec<Value> = self
            .world
            .cars
            .iter()
            .map(|c| {
                json!([
                    r3(c.spike_t),
                    r3(c.mag_t),
                    r3(c.ov_t),
                    r3(c.nitro_t),
                    r3(c.scr_t),
                    r3(c.torn_t),
                    r3(c.speed_mul),
                    r3(c.hook_t),
                ])
            })
            .collect();
        let stuck = b.stuck_to.map(|i| i as i64).unwrap_or(-1);
        let tornados: Vec<Value> = p
            .tornados
            .iter()
            .map(|t| json!([r3(t.pos.x), r3(t.pos.y), r3(t.pos.z), r3(t.t), r3(t.life), t.owner, t.id]))
            .collect();
        let shots: Vec<Value> = p
            .shots
            .iter()
            .map(|s| {
                let to = match s.to {
                    ShotTarget::Ball => -1_i64,
                    ShotTarget::Car(ci) => ci as i64,
                };
                json!({
                    "k": s.kind,
                    "f": s.from,
                    "to": to,
                    "p": [r3(s.pos.x), r3(s.pos.y), r3(s.pos.z)],
                    "ph": s.phase.as_str(),
                    "t": r3(s.t),
                    "pt": r3(s.pt),
                })
            })
            .collect();
        Some(json!({
            "c": cars,
            "b": [r3(b.ice_t), r3(b.anti_t), r3(b.g_scale), stuck],
            "t": tornados,
            "s": shots,
        }))
    }

    fn countdown_ticks(&self) -> Option<u16> {
        if self.state != MatchState::Countdown {
            return None;
        }
        let dt = self.world.phys.dt;
        let mut st = self.state_t;
        let mut n: u16 = 0;
        loop {
            st += dt;
            if COUNTDOWN_S - st <= 0.0 || n > 4000 {
                break;
            }
            n += 1;
        }
        Some(n)
    }

    pub fn send_snapshots(&mut self) {
        let ctls: Vec<[f64; 6]> = self
            .world
            .cars
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if let Some(uid) = &self.car_user_ids[i]
                    && let Some(h) = self.humans.get(uid)
                {
                    return h.applied;
                }
                net_pack_controls(&c.controls)
            })
            .collect();

        let mut meta = serde_json::Map::new();
        if let Some(pw) = self.powers_snap() {
            meta.insert("pw".to_string(), pw);
            let it: Vec<Value> = self
                .world
                .cars
                .iter()
                .map(|c| match c.item {
                    Some(id) => Value::String(id.to_string()),
                    None => Value::Null,
                })
                .collect();
            let inx: Vec<f64> = self
                .world
                .cars
                .iter()
                .map(|c| (c.item_next * 10.0).round() / 10.0)
                .collect();
            meta.insert("it".to_string(), Value::Array(it));
            meta.insert("inx".to_string(), json!(inx));
        }
        if !self.events.is_empty() {
            let drained = std::mem::take(&mut self.events);
            meta.insert("ev".to_string(), Value::Array(drained));
        }
        if self.slot.is_multiple_of(30) || self.state == MatchState::Ended || self.stats_dirty {
            self.stats_dirty = false;
            let cs: Vec<Value> = self
                .world
                .cars
                .iter()
                .map(|c| {
                    json!([
                        c.stats.score,
                        c.stats.goals,
                        c.stats.assists,
                        c.stats.saves,
                        c.stats.shots,
                        c.stats.demos,
                    ])
                })
                .collect();
            meta.insert("cs".to_string(), Value::Array(cs));
        }

        let meta_str = if meta.is_empty() {
            None
        } else {
            serde_json::to_string(&Value::Object(meta)).ok()
        };

        let info = SnapshotInfo {
            state: self.state.as_str(),
            overtime: self.overtime,
            cd: self.countdown_ticks(),
            time_left: self.time_left,
            ot_time: self.ot_time,
            score: [
                self.score[0].clamp(0, 255) as u8,
                self.score[1].clamp(0, 255) as u8,
            ],
            ctls: &ctls,
            meta: meta_str.as_deref(),
        };
        let shared = net_encode_snapshot(&self.world, &info);

        for (uid, h) in &self.humans {
            let mut copy = shared.clone();
            let ack = u32::try_from(h.ack.max(0)).unwrap_or(0);
            let depth = h.depth as f32;
            net_patch_snapshot(&mut copy, ack, depth, h.starves);
            self.callbacks
                .send_raw(uid, SnapshotPayload::Binary(copy));
        }
        self.callbacks
            .send_spectators(SnapshotPayload::Binary(shared));
    }
}

fn serialize_power_event(pe: &PowerEvent) -> Value {
    let vec_val = |v: DVec3| json!({ "$v": [r3(v.x), r3(v.y), r3(v.z)] });
    match *pe {
        PowerEvent::Get { car_idx, id } => json!({
            "type": "power", "kind": "get", "car": { "$c": car_idx }, "id": id,
        }),
        PowerEvent::Fail { car_idx, id } => json!({
            "type": "power", "kind": "fail", "car": { "$c": car_idx }, "id": id,
        }),
        PowerEvent::Use {
            car_idx,
            id,
            target,
        } => {
            let mut m = serde_json::Map::new();
            m.insert("type".into(), json!("power"));
            m.insert("kind".into(), json!("use"));
            m.insert("car".into(), json!({ "$c": car_idx }));
            m.insert("id".into(), json!(id));
            if let Some(t) = target {
                match t {
                    ShotTarget::Ball => {
                        m.insert("target".into(), json!({ "$b": 1 }));
                    }
                    ShotTarget::Car(ci) => {
                        m.insert("target".into(), json!({ "$c": ci }));
                    }
                }
            }
            Value::Object(m)
        }
        PowerEvent::Freeze { car_idx } => json!({
            "type": "power", "kind": "freeze", "car": { "$c": car_idx },
        }),
        PowerEvent::Release {
            car_idx,
            launch,
            pos,
        } => json!({
            "type": "power", "kind": "release", "car": { "$c": car_idx }, "launch": launch, "pos": vec_val(pos),
        }),
        PowerEvent::Shatter { car_idx, pos } => json!({
            "type": "power", "kind": "shatter", "car": { "$c": car_idx }, "pos": vec_val(pos),
        }),
        PowerEvent::Thaw { pos } => json!({
            "type": "power", "kind": "thaw", "pos": vec_val(pos),
        }),
        PowerEvent::TornadoEnd { pos } => json!({
            "type": "power", "kind": "tornadoEnd", "pos": vec_val(pos),
        }),
        PowerEvent::OvHit { car_idx, pos } => json!({
            "type": "power", "kind": "ovHit", "car": { "$c": car_idx }, "pos": vec_val(pos),
        }),
        PowerEvent::Attach { car_idx, pos } => json!({
            "type": "power", "kind": "attach", "car": { "$c": car_idx }, "pos": vec_val(pos),
        }),
        PowerEvent::Shock { car_idx, pos } => json!({
            "type": "power", "kind": "shock", "car": { "$c": car_idx }, "pos": vec_val(pos),
        }),
        PowerEvent::Swap {
            car_idx,
            target_idx,
            from,
            to,
        } => json!({
            "type": "power", "kind": "swap", "car": { "$c": car_idx }, "target": { "$c": target_idx },
            "from": vec_val(from), "to": vec_val(to),
        }),
        PowerEvent::Scrambled { car_idx, by_idx } => json!({
            "type": "power", "kind": "scrambled", "car": { "$c": car_idx }, "by": { "$c": by_idx },
        }),
        PowerEvent::PunchHit { car_idx, pos, dir } => json!({
            "type": "power", "kind": "punchHit", "car": { "$c": car_idx }, "pos": vec_val(pos), "dir": vec_val(dir),
        }),
        PowerEvent::BootHit {
            car_idx,
            target_idx,
            pos,
        } => json!({
            "type": "power", "kind": "bootHit", "car": { "$c": car_idx }, "target": { "$c": target_idx }, "pos": vec_val(pos),
        }),
        PowerEvent::Hook {
            car_idx,
            hook_kind,
            pos,
        } => json!({
            "type": "power", "kind": "hook", "car": { "$c": car_idx }, "hookKind": hook_kind, "pos": vec_val(pos),
        }),
    }
}
