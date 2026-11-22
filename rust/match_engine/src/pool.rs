use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use server_core::protocol::{
    BackendStats, CreateMatchFuture, MatchBackend, MatchCreateOptions, MatchHandle,
};
use tokio::sync::mpsc;

use crate::server_match::{ServerMatch, parse_packed_controls};

pub fn detect_worker_count() -> usize {
    if let Ok(raw) = std::env::var("SIM_WORKERS")
        && let Ok(n) = raw.trim().parse::<usize>()
        && n >= 1
    {
        return n.min(64);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

enum MatchCmd {
    Start,
    Input {
        user_id: String,
        seq: i64,
        packed: [f64; 6],
    },
    ReplayDone(String),
    RemovePlayer {
        user_id: String,
        reason: String,
    },
    SetPaused(bool),
    Stop,
}

struct LoadState {
    started: bool,
    pending: Option<HashSet<String>>,
    humans: HashMap<String, i64>,
}

pub struct TokioMatchHandle {
    id: u64,
    created_at: i64,
    car_info: Value,
    stopped: AtomicBool,
    tx: mpsc::UnboundedSender<MatchCmd>,
    sim_rt: tokio::runtime::Handle,
    load: Mutex<LoadState>,
    active_matches: Arc<AtomicUsize>,
    released: AtomicBool,
}

impl TokioMatchHandle {
    fn trigger_start(&self) {
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        let mut st = self.load.lock().unwrap_or_else(|e| e.into_inner());
        if st.started {
            return;
        }
        st.started = true;
        st.pending = None;
        let _ = self.tx.send(MatchCmd::Start);
    }

    fn release_once(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.active_matches.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for TokioMatchHandle {
    fn drop(&mut self) {
        self.release_once();
    }
}

impl MatchHandle for TokioMatchHandle {
    fn id(&self) -> u64 {
        self.id
    }

    fn created_at(&self) -> i64 {
        self.created_at
    }

    fn car_info(&self) -> Value {
        self.car_info.clone()
    }

    fn start_when_loaded(&self, user_ids: Vec<String>) {
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        let should_start_now = {
            let mut st = self.load.lock().unwrap_or_else(|e| e.into_inner());
            if st.started {
                return;
            }
            let pending: HashSet<String> = user_ids
                .into_iter()
                .filter(|u| st.humans.contains_key(u))
                .collect();
            if pending.is_empty() {
                true
            } else {
                st.pending = Some(pending);
                false
            }
        };
        if should_start_now {
            self.trigger_start();
            return;
        }
        let tx = self.tx.clone();
        // Hold kickoff countdown at most 5s while clients finish building the arena.
        self.sim_rt.spawn(async move {
            tokio::time::sleep(Duration::from_millis(5000)).await;
            let _ = tx.send(MatchCmd::Start);
        });
    }

    fn mark_loaded(&self, user_id: &str) {
        let should_start = {
            let mut st = self.load.lock().unwrap_or_else(|e| e.into_inner());
            if st.started {
                return;
            }
            let Some(pending) = st.pending.as_mut() else {
                return;
            };
            pending.remove(user_id);
            pending.is_empty()
        };
        if should_start {
            self.trigger_start();
        }
    }

    fn replay_done(&self, user_id: &str) {
        if self.stopped.load(Ordering::Relaxed) {
            return;
        }
        let is_human = {
            let st = self.load.lock().unwrap_or_else(|e| e.into_inner());
            st.humans.contains_key(user_id)
        };
        if is_human {
            let _ = self.tx.send(MatchCmd::ReplayDone(user_id.to_string()));
        }
    }

    fn push_input(&self, user_id: &str, seq: i64, packed: &[Value]) -> bool {
        let Some(parsed) = parse_packed_controls(packed) else {
            return false;
        };
        if seq <= 0 {
            return false;
        }
        if self.stopped.load(Ordering::Relaxed) {
            return true;
        }
        {
            let mut st = self.load.lock().unwrap_or_else(|e| e.into_inner());
            let Some(last) = st.humans.get_mut(user_id) else {
                return true;
            };
            if seq <= *last {
                return true;
            }
            *last = seq;
        }
        let _ = self.tx.send(MatchCmd::Input {
            user_id: user_id.to_string(),
            seq,
            packed: parsed,
        });
        true
    }

    fn remove_player(&self, user_id: &str, reason: &str) {
        {
            let mut st = self.load.lock().unwrap_or_else(|e| e.into_inner());
            st.humans.remove(user_id);
        }
        self.mark_loaded(user_id);
        if !self.stopped.load(Ordering::Relaxed) {
            let _ = self.tx.send(MatchCmd::RemovePlayer {
                user_id: user_id.to_string(),
                reason: reason.to_string(),
            });
        }
    }

    fn set_paused(&self, paused: bool) {
        if !self.stopped.load(Ordering::Relaxed) {
            let _ = self.tx.send(MatchCmd::SetPaused(paused));
        }
    }

    fn stop(&self) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        self.release_once();
        let _ = self.tx.send(MatchCmd::Stop);
    }
}

pub struct SimPool {
    worker_count: usize,
    sim_rt: Option<tokio::runtime::Runtime>,
    sim_handle: tokio::runtime::Handle,
    active_matches: Arc<AtomicUsize>,
    next_id: AtomicU64,
}

impl SimPool {
    pub fn new(worker_count: usize) -> Self {
        let worker_count = worker_count.max(1);
        let sim_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_count)
            .thread_name("sim-worker")
            .enable_time()
            .build()
            .expect("failed to build simulation worker runtime");
        let sim_handle = sim_rt.handle().clone();
        Self {
            worker_count,
            sim_rt: Some(sim_rt),
            sim_handle,
            active_matches: Arc::new(AtomicUsize::new(0)),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn from_env() -> Self {
        Self::new(detect_worker_count())
    }
}

impl Drop for SimPool {
    fn drop(&mut self) {
        if let Some(rt) = self.sim_rt.take() {
            rt.shutdown_background();
        }
    }
}

impl Default for SimPool {
    fn default() -> Self {
        Self::from_env()
    }
}

impl MatchBackend for SimPool {
    fn stats(&self) -> BackendStats {
        BackendStats {
            ready_workers: self.worker_count,
            active_matches: self.active_matches.load(Ordering::Acquire),
        }
    }

    fn create_match(&self, opts: MatchCreateOptions) -> CreateMatchFuture {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let active_matches = self.active_matches.clone();
        let sim_rt = self.sim_handle.clone();
        Box::pin(async move {
            let humans: HashMap<String, i64> = opts
                .players
                .iter()
                .map(|p| (p.user_id.clone(), 0))
                .collect();
            let sim = ServerMatch::new(opts);
            let created_at = sim.created_at;
            let car_info = Value::Array(sim.car_info.clone());

            let (tx, rx) = mpsc::unbounded_channel();
            active_matches.fetch_add(1, Ordering::AcqRel);
            let handle = Arc::new(TokioMatchHandle {
                id,
                created_at,
                car_info,
                stopped: AtomicBool::new(false),
                tx,
                sim_rt: sim_rt.clone(),
                load: Mutex::new(LoadState {
                    started: false,
                    pending: None,
                    humans,
                }),
                active_matches,
                released: AtomicBool::new(false),
            });

            let handle_for_task = handle.clone();
            sim_rt.spawn(run_match_loop(sim, rx, handle_for_task));

            Ok(handle as Arc<dyn MatchHandle>)
        })
    }
}

async fn run_match_loop(
    mut sim: ServerMatch,
    mut rx: mpsc::UnboundedReceiver<MatchCmd>,
    handle: Arc<TokioMatchHandle>,
) {
    while !sim.finished {
        // Drain all immediately queued commands before stepping.
        while let Ok(cmd) = rx.try_recv() {
            if !apply_cmd(&mut sim, cmd) {
                handle.release_once();
                return;
            }
        }
        if sim.finished {
            break;
        }
        if sim.running {
            let now = Instant::now();
            sim.step_clock(now);
            if sim.finished {
                break;
            }
            let wait_ms = sim.ms_until_next_slot(Instant::now());
            if wait_ms.is_finite() {
                let sleep_dur = if wait_ms > 2.0 {
                    Duration::from_secs_f64((wait_ms - 1.0) / 1000.0)
                } else {
                    Duration::from_micros(500)
                };
                tokio::select! {
                    maybe_cmd = rx.recv() => {
                        match maybe_cmd {
                            Some(cmd) => {
                                if !apply_cmd(&mut sim, cmd) {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    () = tokio::time::sleep(sleep_dur) => {}
                }
            } else {
                match rx.recv().await {
                    Some(cmd) => {
                        if !apply_cmd(&mut sim, cmd) {
                            break;
                        }
                    }
                    None => break,
                }
            }
        } else {
            match rx.recv().await {
                Some(cmd) => {
                    if !apply_cmd(&mut sim, cmd) {
                        break;
                    }
                }
                None => break,
            }
        }
    }
    handle.stopped.store(true, Ordering::Release);
    handle.release_once();
}

fn apply_cmd(sim: &mut ServerMatch, cmd: MatchCmd) -> bool {
    match cmd {
        MatchCmd::Start => {
            if !sim.running && !sim.finished {
                sim.start();
            }
            true
        }
        MatchCmd::Input {
            user_id,
            seq,
            packed,
        } => {
            sim.push_input(&user_id, seq, packed);
            true
        }
        MatchCmd::ReplayDone(uid) => {
            sim.mark_replay_done(&uid);
            true
        }
        MatchCmd::RemovePlayer { user_id, reason } => {
            sim.remove_player_with_reason(&user_id, &reason);
            true
        }
        MatchCmd::SetPaused(p) => {
            sim.set_paused(p);
            true
        }
        MatchCmd::Stop => {
            sim.stop();
            false
        }
    }
}
