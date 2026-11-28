//! Compact quantized causal RoPE Transformer policy for `SAVANT` bots (trained via PPO self-play).
//!
//! Shared by both offline RL training (`rust/rl_env`) and live inference (`BotAi`),
//! guaranteeing identical observation features, temporal KV-cached RoPE attention, and action
//! decoding with zero train/serve skew.

use std::sync::{Arc, OnceLock, RwLock};

use glam::DVec3;

use crate::arena::GoalMode;
use crate::math_ext::{Controls, ThreeVec3Ext, clamp};
use crate::powers::{POWERS, PowerTarget, ShotTarget};
use crate::world::World;

pub const OBS_DIM: usize = 146;
pub const ACT_HEADS: usize = 9;
/// Per-head action branch sizes: `throttle, steer, pitch, yaw, roll` are fine-grained
/// 21-level axes (uniform 0.1 steps across [-1, 1], analog-thumbstick resolution);
/// `jump, boost, handbrake, item` are binary.
pub const ACT_BRANCHES: [usize; ACT_HEADS] = [21, 21, 21, 21, 21, 2, 2, 2, 2];
/// Sum of `ACT_BRANCHES` (derived so the three can never drift apart).
pub const TOTAL_LOGITS: usize = {
    let mut total = 0usize;
    let mut i = 0usize;
    while i < ACT_HEADS {
        total += ACT_BRANCHES[i];
        i += 1;
    }
    total
};
/// Decision period in 120 Hz physics ticks (4 ticks = 30 Hz = 33.3 ms per action).
pub const POLICY_TICK_SKIP: u64 = 4;

pub const POLICY_MAGIC: &[u8; 4] = b"HSP2";

const MAX_D_MODEL: usize = 512;
const MAX_N_LAYERS: usize = 16;
const MAX_FFN_DIM: usize = 2048;
const MAX_CTX_LEN: usize = 128;

/// Uniform `n`-level bin spanning `[-1, 1]` (level 0 = -1, level n-1 = +1).
#[inline]
fn axis_val(idx: usize, n: usize) -> f64 {
    let n = n.max(2);
    let i = idx.min(n - 1) as f64;
    -1.0 + 2.0 * i / (n - 1) as f64
}

/// Nearest bin for a continuous control value in `[-1, 1]`.
#[inline]
fn quantize_axis(v: f64, n: usize) -> u8 {
    let n = n.max(2);
    let i = ((clamp(v, -1.0, 1.0) + 1.0) * 0.5 * (n - 1) as f64).round() as i32;
    i.clamp(0, n as i32 - 1) as u8
}

/// Canonicalize a world-space vector into team-attacking `+Z` frame (180° Y-rotation for team 1).
/// Proper rotation (`det = +1`), so `car.basis.to_local(...)` is invariant without sign flips.
#[inline]
fn canon_vec(v: DVec3, flip: f64) -> DVec3 {
    DVec3::new(v.x * flip, v.y, v.z * flip)
}

/// Extracts the 146-float team-symmetric observation vector for `car_idx`
/// (covers standard arenas, experimental Labs arenas, and Mayhem power-ups).
pub fn extract_obs(w: &World, car_idx: usize, prev_ctl: &Controls, out: &mut [f32; OBS_DIM]) {
    out.fill(0.0);
    let car = &w.cars[car_idx];
    let shape = &w.shape;
    let flip = if car.team == 0 { 1.0 } else { -1.0 };
    let inv_w = 1.0 / shape.w.max(1.0);
    let inv_h = 1.0 / shape.h.max(1.0);
    let glz = shape.goal_line_z().max(1.0);
    let inv_l = 1.0 / glz;

    let mut k = 0;
    let mut push = |v: f64| {
        let vf = if v.is_finite() { v as f32 } else { 0.0 };
        out[k] = vf.clamp(-5.0, 5.0);
        k += 1;
    };

    // 1. Self car (23 floats)
    let cp = canon_vec(car.pos, flip);
    let cv = canon_vec(car.vel, flip);
    let cf = canon_vec(car.basis.f, flip);
    let cu = canon_vec(car.basis.u, flip);
    let cw = canon_vec(car.ang_vel, flip);
    push(cp.x * inv_w);
    push(cp.y * inv_h);
    push(cp.z * inv_l);
    push(cv.x / 23.0);
    push(cv.y / 23.0);
    push(cv.z / 23.0);
    push(cf.x);
    push(cf.y);
    push(cf.z);
    push(cu.x);
    push(cu.y);
    push(cu.z);
    push(cw.x / 5.5);
    push(cw.y / 5.5);
    push(cw.z / 5.5);
    push(car.boost / 100.0);
    push(if car.on_ground { 1.0 } else { 0.0 });
    push(if car.has_flip { 1.0 } else { 0.0 });
    push(if car.jumping { 1.0 } else { 0.0 });
    push(if car.flipping { 1.0 } else { 0.0 });
    push(if car.supersonic { 1.0 } else { 0.0 });
    push(if car.prev_jump { 1.0 } else { 0.0 });
    push((car.time_since_jump / 1.5).min(1.0));

    // 2. Ball + car-local relative + predicted trajectory (22 floats)
    let bp = canon_vec(w.ball.pos, flip);
    let bv = canon_vec(w.ball.vel, flip);
    let bw = canon_vec(w.ball.ang_vel, flip);
    push(bp.x * inv_w);
    push(bp.y * inv_h);
    push(bp.z * inv_l);
    push(bv.x / 40.0);
    push(bv.y / 40.0);
    push(bv.z / 40.0);
    push(bw.x / 6.0);
    push(bw.y / 6.0);
    push(bw.z / 6.0);

    let rel_bp = car.basis.to_local(w.ball.pos - car.pos);
    let rel_bv = car.basis.to_local(w.ball.vel - car.vel);
    push(rel_bp.x / 40.0);
    push(rel_bp.y / 40.0);
    push(rel_bp.z / 40.0);
    push(rel_bv.x / 40.0);
    push(rel_bv.y / 40.0);
    push(rel_bv.z / 40.0);

    let p05 = canon_vec(w.pred.at(0.5).pos, flip);
    let p12 = canon_vec(w.pred.at(1.2).pos, flip);
    push(p05.x * inv_w);
    push(p05.y * inv_h);
    push(p05.z * inv_l);
    push(p12.x * inv_w);
    push(p12.y * inv_h);
    push(p12.z * inv_l);
    let goal_sig = if w.pred.goal_team >= 0 {
        let urgency = (-w.pred.goal_time / 2.0).exp();
        if w.pred.goal_team == car.team as i32 {
            -urgency
        } else {
            urgency
        }
    } else {
        0.0
    };
    push(goal_sig);

    // 3. Opponents sorted by distance to ball (15 + 9 = 24 floats)
    let mut opp_list: [(usize, f64); 8] = [(0, 1e9); 8];
    let mut n_opps = 0usize;
    let mut mate_best: Option<(usize, f64)> = None;
    let my_ball_d = car.pos.distance(w.ball.pos);

    for (i, o) in w.cars.iter().enumerate() {
        if i == car_idx || o.demolished {
            continue;
        }
        let d_ball = o.pos.distance(w.ball.pos);
        if o.team != car.team {
            if n_opps < opp_list.len() {
                opp_list[n_opps] = (i, d_ball);
                n_opps += 1;
            }
        } else if mate_best.is_none_or(|(_, bd)| d_ball < bd) {
            mate_best = Some((i, d_ball));
        }
    }
    opp_list[..n_opps].sort_by(|a, b| a.1.total_cmp(&b.1));

    if n_opps >= 1 {
        let o = &w.cars[opp_list[0].0];
        let op = canon_vec(o.pos, flip);
        let ov = canon_vec(o.vel, flip);
        let of = canon_vec(o.basis.f, flip);
        let rel_o = car.basis.to_local(o.pos - car.pos);
        push(op.x * inv_w);
        push(op.y * inv_h);
        push(op.z * inv_l);
        push(ov.x / 23.0);
        push(ov.y / 23.0);
        push(ov.z / 23.0);
        push(of.x);
        push(of.y);
        push(of.z);
        push(rel_o.x / 40.0);
        push(rel_o.y / 40.0);
        push(rel_o.z / 40.0);
        push(o.boost / 100.0);
        push(if o.on_ground { 1.0 } else { 0.0 });
        push(1.0);
    } else {
        for _ in 0..15 {
            push(0.0);
        }
    }

    if n_opps >= 2 {
        let o = &w.cars[opp_list[1].0];
        let op = canon_vec(o.pos, flip);
        let ov = canon_vec(o.vel, flip);
        push(op.x * inv_w);
        push(op.y * inv_h);
        push(op.z * inv_l);
        push(ov.x / 23.0);
        push(ov.y / 23.0);
        push(ov.z / 23.0);
        push((op.x - bp.x) / 40.0);
        push((op.z - bp.z) / 40.0);
        push(1.0);
    } else {
        for _ in 0..9 {
            push(0.0);
        }
    }

    // 4. Nearest Teammate (12 floats)
    if let Some((mi, m_ball_d)) = mate_best {
        let m = &w.cars[mi];
        let mp = canon_vec(m.pos, flip);
        let mv = canon_vec(m.vel, flip);
        push(mp.x * inv_w);
        push(mp.y * inv_h);
        push(mp.z * inv_l);
        push(mv.x / 23.0);
        push(mv.y / 23.0);
        push(mv.z / 23.0);
        push((mp.x - bp.x) / 40.0);
        push((mp.y - bp.y) / 20.0);
        push((mp.z - bp.z) / 40.0);
        push(m.boost / 100.0);
        push(if m_ball_d < my_ball_d { 1.0 } else { 0.0 });
        push(1.0);
    } else {
        for _ in 0..12 {
            push(0.0);
        }
    }

    // 5. Goals, arena & nearest active boost pads (14 floats)
    let (own_g, opp_g, alt_opp_g) = if let Some(sg) = shape.split_goal {
        let sx = if w.ball.pos.x >= 0.0 { 1.0 } else { -1.0 };
        (
            DVec3::new(sx * sg.gx, 2.0, -flip * sg.gz),
            DVec3::new(sx * sg.gx, 2.0, flip * sg.gz),
            Some(DVec3::new(-sx * sg.gx, 2.0, flip * sg.gz)),
        )
    } else {
        (
            DVec3::new(0.0, 2.0, -flip * glz),
            DVec3::new(0.0, 2.0, flip * glz),
            None,
        )
    };
    let loc_own = car.basis.to_local((own_g - car.pos).three_normalize());
    let loc_opp = car.basis.to_local((opp_g - car.pos).three_normalize());
    push(loc_own.x);
    push(loc_own.y);
    push(loc_own.z);
    push(loc_opp.x);
    push(loc_opp.y);
    push(loc_opp.z);
    push(shape.gw / 10.0);
    push(if w.kickoff { 1.0 } else { 0.0 });
    push(shape.dist(car.pos.x, car.pos.y, car.pos.z) / 10.0);

    let mut best_big: Option<(DVec3, f64)> = None;
    let mut best_any: Option<(DVec3, f64)> = None;
    for p in &w.pads {
        if !p.active {
            continue;
        }
        let d2 = car.pos.distance_squared(p.pos);
        if best_any.is_none_or(|(_, bd)| d2 < bd) {
            best_any = Some((p.pos, d2));
        }
        if p.big && best_big.is_none_or(|(_, bd)| d2 < bd) {
            best_big = Some((p.pos, d2));
        }
    }
    if let Some((pp, _)) = best_big {
        let lp = car.basis.to_local(pp - car.pos);
        push(lp.x / 40.0);
        push(lp.z / 40.0);
        push(1.0);
    } else {
        push(0.0);
        push(0.0);
        push(0.0);
    }
    if let Some((pp, _)) = best_any {
        let lp = car.basis.to_local(pp - car.pos);
        push(lp.x / 40.0);
        push(lp.z / 40.0);
    } else {
        push(0.0);
        push(0.0);
    }

    // 6. Direct ball/shot geometric cues (5 floats — no prev-action causal confusion!)
    let to_ball_w = w.ball.pos - car.pos;
    let bdist = to_ball_w.length();
    let to_ball_n = if bdist > 1e-4 {
        to_ball_w / bdist
    } else {
        car.basis.f
    };
    let ball_to_opp_g = (opp_g - w.ball.pos).three_normalize();
    push(car.speed() / 23.0);
    push(bdist / 80.0);
    push(car.basis.f.dot(to_ball_n));
    push(car.basis.r.dot(to_ball_n));
    push(to_ball_n.dot(ball_to_opp_g));

    // 7. Experimental / Labs arena geometry (10 floats)
    push(if shape.fig8.is_some() { 1.0 } else { 0.0 });
    push(if shape.goal_mode == GoalMode::Standalone { 1.0 } else { 0.0 });
    push(if shape.goal_mode == GoalMode::Split { 1.0 } else { 0.0 });
    push(glz / shape.l.max(1.0));
    push(shape.w / 45.0);
    push(shape.l / 55.0);
    if let Some(alt_g) = alt_opp_g {
        let loc_alt = car.basis.to_local((alt_g - car.pos).three_normalize());
        push(loc_alt.x);
        push(loc_alt.z);
    } else {
        push(0.0);
        push(0.0);
    }
    push(shape.wall_dist_2d(car.pos.x, car.pos.z) / 20.0);
    push(shape.wall_dist_2d(w.ball.pos.x, w.ball.pos.z) / 20.0);

    // 8. Mayhem mode & power-up state (36 floats; all 0.0 in standard matches)
    if let Some(pw) = w.powers.as_ref() {
        push(1.0);
        for pdef in &POWERS {
            push(if car.item == Some(pdef.id) { 1.0 } else { 0.0 });
        }
        let tgt = car
            .item
            .and_then(|id| pw.target_for(car_idx, id, &w.cars, &w.ball));
        let can_fire = w.ball.stuck_to == Some(car_idx) || tgt.is_some();
        push(if can_fire { 1.0 } else { 0.0 });
        push(if car.item.is_none() {
            clamp(car.item_next / 10.0, 0.0, 1.0)
        } else {
            0.0
        });
        push(if car.prev_item { 1.0 } else { 0.0 });
        let tgt_dist = match tgt {
            Some(PowerTarget::Ball) => (w.ball.pos.distance(car.pos) / 80.0).min(1.0),
            Some(PowerTarget::Enemy(ei)) => (w.cars[ei].pos.distance(car.pos) / 60.0).min(1.0),
            _ => 0.0,
        };
        push(tgt_dist);

        // Self active power timers (7)
        push(clamp(car.spike_t / 7.5, 0.0, 1.0));
        push(clamp(car.mag_t / 6.0, 0.0, 1.0));
        push(clamp(car.ov_t / 8.0, 0.0, 1.0));
        push(clamp(car.nitro_t / 5.0, 0.0, 1.0));
        push(clamp(car.scr_t / 4.5, 0.0, 1.0));
        push(clamp(car.torn_t / 6.5, 0.0, 1.0));
        push(clamp(car.hook_t / 0.2, 0.0, 1.0));

        // Ball power state (5)
        let stuck_self = w.ball.stuck_to == Some(car_idx);
        let stuck_mate = w
            .ball
            .stuck_to
            .is_some_and(|si| si != car_idx && w.cars[si].team == car.team);
        let stuck_opp = w
            .ball
            .stuck_to
            .is_some_and(|si| w.cars[si].team != car.team);
        push(if stuck_self { 1.0 } else { 0.0 });
        push(if stuck_mate { 1.0 } else { 0.0 });
        push(if stuck_opp { 1.0 } else { 0.0 });
        push(clamp(w.ball.ice_t / 3.2, 0.0, 1.0));
        push(clamp(w.ball.anti_t / 6.0, 0.0, 1.0));

        // Nearest other tornado (3)
        let mut best_torn: Option<(DVec3, f64)> = None;
        for t in &pw.tornados {
            if t.owner == car_idx {
                continue;
            }
            let d2 = car.pos.distance_squared(t.pos);
            if best_torn.is_none_or(|(_, bd)| d2 < bd) {
                best_torn = Some((t.pos, d2));
            }
        }
        if let Some((tp, _)) = best_torn {
            let lt = car.basis.to_local(tp - car.pos);
            push(lt.x / 40.0);
            push(lt.z / 40.0);
            push(1.0);
        } else {
            push(0.0);
            push(0.0);
            push(0.0);
        }

        // Opponent active danger + swap charge signal (2)
        let opp_danger = w.cars.iter().any(|o| {
            o.team != car.team
                && !o.demolished
                && (o.ov_t > 0.0 || o.spike_t > 0.0 || o.torn_t > 0.0)
        });
        push(if opp_danger { 1.0 } else { 0.0 });
        let mut swap_sig = 0.0;
        for s in &pw.shots {
            if s.kind == "swap" {
                if s.from == car_idx {
                    swap_sig = 1.0;
                    break;
                } else if s.to == ShotTarget::Car(car_idx) {
                    swap_sig = -1.0;
                }
            }
        }
        push(swap_sig);
    } else {
        for _ in 0..36 {
            push(0.0);
        }
    }

    let _ = prev_ctl;
    debug_assert_eq!(k, OBS_DIM);
}

#[inline]
pub fn decode_action(act: &[u8; ACT_HEADS]) -> Controls {
    Controls {
        throttle: axis_val(act[0] as usize, ACT_BRANCHES[0]),
        steer: axis_val(act[1] as usize, ACT_BRANCHES[1]),
        pitch: axis_val(act[2] as usize, ACT_BRANCHES[2]),
        yaw: axis_val(act[3] as usize, ACT_BRANCHES[3]),
        roll: axis_val(act[4] as usize, ACT_BRANCHES[4]),
        jump: act[5] != 0,
        boost: act[6] != 0,
        handbrake: act[7] != 0,
        air_roll_free: false,
        item: act[8] != 0,
    }
}

#[inline]
pub fn encode_action(c: &Controls) -> [u8; ACT_HEADS] {
    [
        quantize_axis(c.throttle, ACT_BRANCHES[0]),
        quantize_axis(c.steer, ACT_BRANCHES[1]),
        quantize_axis(c.pitch, ACT_BRANCHES[2]),
        quantize_axis(c.yaw, ACT_BRANCHES[3]),
        quantize_axis(c.roll, ACT_BRANCHES[4]),
        u8::from(c.jump),
        u8::from(c.boost),
        u8::from(c.handbrake),
        u8::from(c.item),
    ]
}

// ---------------------------------------------------------------------------------------------
// Quantized int8 Causal RoPE Transformer (`HSP2` binary format)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct QLinear {
    in_dim: usize,
    out_dim: usize,
    /// Per-output-row dequantization scale (`w_fp32 = w_i8 * scale[row]`).
    scales: Vec<f32>,
    biases: Vec<f32>,
    weights: Vec<i8>,
}

#[derive(Clone, Debug)]
struct RmsNorm {
    weight: Vec<f32>,
}

#[derive(Clone, Debug)]
struct TransformerBlock {
    norm1: RmsNorm,
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    norm2: RmsNorm,
    w_gate: QLinear,
    w_up: QLinear,
    w_down: QLinear,
}

/// Per-agent sliding-window KV cache for causal RoPE Transformer inference.
///
/// Default-constructed caches allocate zero memory until the first forward pass.
#[derive(Clone, Debug, Default)]
pub struct SavantPolicyCache {
    /// Flattened K cache: `[n_layers][ctx_len][d_model]`
    k: Vec<f32>,
    /// Flattened V cache: `[n_layers][ctx_len][d_model]`
    v: Vec<f32>,
    n_layers: usize,
    ctx_len: usize,
    d_model: usize,
    /// Monotonic step counter within the current episode/sequence (RoPE position).
    pos: usize,
    /// Number of valid cached frames (`0..=ctx_len`).
    len: usize,
}

impl SavantPolicyCache {
    #[inline]
    pub fn reset(&mut self) {
        self.pos = 0;
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ensure_for(&mut self, pol: &SavantPolicy) {
        if self.n_layers != pol.n_layers
            || self.ctx_len != pol.ctx_len
            || self.d_model != pol.d_model
        {
            let total = pol.n_layers * pol.ctx_len * pol.d_model;
            self.k.resize(total, 0.0);
            self.v.resize(total, 0.0);
            self.n_layers = pol.n_layers;
            self.ctx_len = pol.ctx_len;
            self.d_model = pol.d_model;
            self.pos = 0;
            self.len = 0;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SavantPolicy {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    ctx_len: usize,
    rope_theta: f32,
    inv_freq: Vec<f64>,
    in_proj: QLinear,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    head: QLinear,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return None;
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    fn read_u16(&mut self) -> Option<u16> {
        let b = self.read_bytes(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_f32(&mut self) -> Option<f32> {
        let b = self.read_bytes(4)?;
        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32_vec(&mut self, n: usize) -> Option<Vec<f32>> {
        let b = self.read_bytes(n * 4)?;
        let mut out = Vec::with_capacity(n);
        for chunk in b.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Some(out)
    }

    fn read_i8_vec(&mut self, n: usize) -> Option<Vec<i8>> {
        let b = self.read_bytes(n)?;
        Some(b.iter().map(|&x| x as i8).collect())
    }

    fn read_linear(&mut self, in_dim: usize, out_dim: usize) -> Option<QLinear> {
        let scales = self.read_f32_vec(out_dim)?;
        let biases = self.read_f32_vec(out_dim)?;
        let weights = self.read_i8_vec(out_dim * in_dim)?;
        Some(QLinear {
            in_dim,
            out_dim,
            scales,
            biases,
            weights,
        })
    }

    fn read_rms_norm(&mut self, dim: usize) -> Option<RmsNorm> {
        let weight = self.read_f32_vec(dim)?;
        Some(RmsNorm { weight })
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn linear_forward(l: &QLinear, x: &[f32], y: &mut [f32]) {
    let in_d = l.in_dim;
    for row in 0..l.out_dim {
        let w = &l.weights[row * in_d..(row + 1) * in_d];
        let mut acc = 0.0_f32;
        for i in 0..in_d {
            acc += (w[i] as f32) * x[i];
        }
        y[row] = acc * l.scales[row] + l.biases[row];
    }
}

fn rms_norm(norm: &RmsNorm, x: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mut ms = 0.0_f32;
    for &v in x {
        ms += v * v;
    }
    let inv_rms = 1.0 / (ms / n + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * norm.weight[i];
    }
}

fn apply_rope_inplace(vec: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for k in 0..half {
            let x1 = vec[base + k];
            let x2 = vec[base + half + k];
            let c = cos[k];
            let s = sin[k];
            vec[base + k] = x1 * c - x2 * s;
            vec[base + half + k] = x1 * s + x2 * c;
        }
    }
}

impl SavantPolicy {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        let mut r = Reader::new(bytes);
        if r.read_bytes(4)? != POLICY_MAGIC {
            return None;
        }
        let obs_d = r.read_u16()? as usize;
        let d_model = r.read_u16()? as usize;
        let n_layers = r.read_u16()? as usize;
        let n_heads = r.read_u16()? as usize;
        let ffn_dim = r.read_u16()? as usize;
        let ctx_len = r.read_u16()? as usize;
        let out_d = r.read_u16()? as usize;
        let _flags = r.read_u16()?;
        let rope_theta = r.read_f32()?;

        if obs_d != OBS_DIM
            || out_d != TOTAL_LOGITS
            || d_model == 0
            || d_model > MAX_D_MODEL
            || n_layers == 0
            || n_layers > MAX_N_LAYERS
            || n_heads == 0
            || d_model % n_heads != 0
            || (d_model / n_heads) % 2 != 0
            || ffn_dim == 0
            || ffn_dim > MAX_FFN_DIM
            || ctx_len == 0
            || ctx_len > MAX_CTX_LEN
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return None;
        }

        let head_dim = d_model / n_heads;
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for k in 0..half {
            let exp = -(2.0 * k as f64) / (head_dim as f64);
            inv_freq.push((rope_theta as f64).powf(exp));
        }

        let in_proj = r.read_linear(obs_d, d_model)?;
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let norm1 = r.read_rms_norm(d_model)?;
            let wq = r.read_linear(d_model, d_model)?;
            let wk = r.read_linear(d_model, d_model)?;
            let wv = r.read_linear(d_model, d_model)?;
            let wo = r.read_linear(d_model, d_model)?;
            let norm2 = r.read_rms_norm(d_model)?;
            let w_gate = r.read_linear(d_model, ffn_dim)?;
            let w_up = r.read_linear(d_model, ffn_dim)?;
            let w_down = r.read_linear(ffn_dim, d_model)?;
            blocks.push(TransformerBlock {
                norm1,
                wq,
                wk,
                wv,
                wo,
                norm2,
                w_gate,
                w_up,
                w_down,
            });
        }
        let final_norm = r.read_rms_norm(d_model)?;
        let head = r.read_linear(d_model, out_d)?;

        Some(Self {
            d_model,
            n_layers,
            n_heads,
            head_dim,
            ffn_dim,
            ctx_len,
            rope_theta,
            inv_freq,
            in_proj,
            blocks,
            final_norm,
            head,
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    /// Runs one causal decision step through the RoPE Transformer, updating `cache` in place.
    pub fn forward_step(
        &self,
        obs: &[f32; OBS_DIM],
        cache: &mut SavantPolicyCache,
    ) -> [f32; TOTAL_LOGITS] {
        cache.ensure_for(self);

        let d = self.d_model;
        let nh = self.n_heads;
        let hd = self.head_dim;
        let half = hd / 2;
        let ffn = self.ffn_dim;
        let ctx = self.ctx_len;
        let pos = cache.pos;
        let slot = pos % ctx;
        let valid_len = (pos + 1).min(ctx);
        let scale = 1.0_f32 / (hd as f32).sqrt();

        // Precompute RoPE cos/sin for `pos` using f64 trig for exact relative invariance at any step.
        let mut cos = [0.0_f32; MAX_D_MODEL / 2];
        let mut sin = [0.0_f32; MAX_D_MODEL / 2];
        let pos_f = pos as f64;
        for k in 0..half {
            let (s, c) = (pos_f * self.inv_freq[k]).sin_cos();
            cos[k] = c as f32;
            sin[k] = s as f32;
        }

        let mut x = [0.0_f32; MAX_D_MODEL];
        let mut h = [0.0_f32; MAX_D_MODEL];
        let mut q = [0.0_f32; MAX_D_MODEL];
        let mut k_cur = [0.0_f32; MAX_D_MODEL];
        let mut v_cur = [0.0_f32; MAX_D_MODEL];
        let mut attn_out = [0.0_f32; MAX_D_MODEL];
        let mut proj = [0.0_f32; MAX_D_MODEL];
        let mut gate = [0.0_f32; MAX_FFN_DIM];
        let mut up = [0.0_f32; MAX_FFN_DIM];
        let mut scores = [0.0_f32; MAX_CTX_LEN];

        linear_forward(&self.in_proj, obs, &mut x[..d]);

        let layer_stride = ctx * d;
        for (li, blk) in self.blocks.iter().enumerate() {
            // 1. Pre-RMSNorm + RoPE Multi-Head Causal Sliding-Window Attention
            rms_norm(&blk.norm1, &x[..d], &mut h[..d]);
            linear_forward(&blk.wq, &h[..d], &mut q[..d]);
            linear_forward(&blk.wk, &h[..d], &mut k_cur[..d]);
            linear_forward(&blk.wv, &h[..d], &mut v_cur[..d]);

            apply_rope_inplace(&mut q[..d], nh, hd, &cos[..half], &sin[..half]);
            apply_rope_inplace(&mut k_cur[..d], nh, hd, &cos[..half], &sin[..half]);

            let layer_off = li * layer_stride;
            let dst_off = layer_off + slot * d;
            cache.k[dst_off..dst_off + d].copy_from_slice(&k_cur[..d]);
            cache.v[dst_off..dst_off + d].copy_from_slice(&v_cur[..d]);

            let k_layer = &cache.k[layer_off..layer_off + layer_stride];
            let v_layer = &cache.v[layer_off..layer_off + layer_stride];

            for head in 0..nh {
                let hb = head * hd;
                let qh = &q[hb..hb + hd];

                let mut max_s = f32::NEG_INFINITY;
                for s_idx in 0..valid_len {
                    let kh = &k_layer[s_idx * d + hb..s_idx * d + hb + hd];
                    let mut dot = 0.0_f32;
                    for j in 0..hd {
                        dot += qh[j] * kh[j];
                    }
                    let sc = dot * scale;
                    scores[s_idx] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }

                let mut sum_exp = 0.0_f32;
                for s_idx in 0..valid_len {
                    let e = (scores[s_idx] - max_s).exp();
                    scores[s_idx] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp.max(1e-12);

                let out_h = &mut attn_out[hb..hb + hd];
                out_h.fill(0.0);
                for s_idx in 0..valid_len {
                    let w = scores[s_idx] * inv_sum;
                    let vh = &v_layer[s_idx * d + hb..s_idx * d + hb + hd];
                    for j in 0..hd {
                        out_h[j] += w * vh[j];
                    }
                }
            }

            linear_forward(&blk.wo, &attn_out[..d], &mut proj[..d]);
            for i in 0..d {
                x[i] += proj[i];
            }

            // 2. Pre-RMSNorm + SwiGLU FFN
            rms_norm(&blk.norm2, &x[..d], &mut h[..d]);
            linear_forward(&blk.w_gate, &h[..d], &mut gate[..ffn]);
            linear_forward(&blk.w_up, &h[..d], &mut up[..ffn]);
            for i in 0..ffn {
                gate[i] = silu(gate[i]) * up[i];
            }
            linear_forward(&blk.w_down, &gate[..ffn], &mut proj[..d]);
            for i in 0..d {
                x[i] += proj[i];
            }
        }

        cache.pos = pos + 1;
        cache.len = valid_len;

        rms_norm(&self.final_norm, &x[..d], &mut h[..d]);
        let mut logits = [0.0_f32; TOTAL_LOGITS];
        linear_forward(&self.head, &h[..d], &mut logits);
        logits
    }

    /// Stateless single-frame forward pass (convenience wrapper with a fresh 1-step cache).
    pub fn forward(&self, obs: &[f32; OBS_DIM]) -> [f32; TOTAL_LOGITS] {
        let mut cache = SavantPolicyCache::default();
        self.forward_step(obs, &mut cache)
    }

    /// Runs one causal decision step with KV cache and returns greedy argmax action per branch.
    pub fn act_greedy(&self, obs: &[f32; OBS_DIM], cache: &mut SavantPolicyCache) -> [u8; ACT_HEADS] {
        let logits = self.forward_step(obs, cache);
        let mut act = [0u8; ACT_HEADS];
        let mut off = 0usize;
        for (h, &n) in ACT_BRANCHES.iter().enumerate() {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for i in 0..n {
                let v = logits[off + i];
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            act[h] = best_i as u8;
            off += n;
        }
        act
    }
}

/// Embedded trained SAVANT neural policy bytes (empty until trained).
pub const EMBEDDED_POLICY_BYTES: &[u8] = include_bytes!("../assets/savant_policy.bin");

pub fn embedded_savant_policy() -> Option<&'static SavantPolicy> {
    static POL: OnceLock<Option<SavantPolicy>> = OnceLock::new();
    POL.get_or_init(|| SavantPolicy::from_bytes(EMBEDDED_POLICY_BYTES))
        .as_ref()
}

fn live_policy_slot() -> &'static RwLock<Option<Arc<SavantPolicy>>> {
    static LIVE_POL: OnceLock<RwLock<Option<Arc<SavantPolicy>>>> = OnceLock::new();
    LIVE_POL.get_or_init(|| {
        RwLock::new(SavantPolicy::from_bytes(EMBEDDED_POLICY_BYTES).map(Arc::new))
    })
}

/// Returns the currently active SAVANT neural policy (hot-reloaded from disk if available,
/// otherwise falling back to `EMBEDDED_POLICY_BYTES`).
pub fn active_savant_policy() -> Option<Arc<SavantPolicy>> {
    live_policy_slot()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().map(Arc::clone))
}

/// Parses `bytes` as an `HSP2` policy and atomically swaps it in as the active SAVANT policy.
/// Returns `true` if the bytes were valid and installed, `false` otherwise.
pub fn reload_savant_policy_bytes(bytes: &[u8]) -> bool {
    let Some(pol) = SavantPolicy::from_bytes(bytes) else {
        return false;
    };
    if let Ok(mut guard) = live_policy_slot().write() {
        *guard = Some(Arc::new(pol));
        true
    } else {
        false
    }
}

/// Returns true if the arena shape and match state are supported by the trained neural policy.
/// Trained across all standard arenas, experimental Labs arenas (Figure-8, Standalone, Split-Goal),
/// and Mayhem power-up matches — only paused while the ball is frozen for countdown/goal replay.
#[inline]
pub fn policy_supports_world(w: &World) -> bool {
    !w.ball.frozen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_hsp2(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        ffn_dim: usize,
        ctx_len: usize,
        rope_theta: f32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(POLICY_MAGIC);
        for v in [
            OBS_DIM as u16,
            d_model as u16,
            n_layers as u16,
            n_heads as u16,
            ffn_dim as u16,
            ctx_len as u16,
            TOTAL_LOGITS as u16,
            0u16,
        ] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&rope_theta.to_le_bytes());

        let mut seed = 0x1234_5678_9abc_def0_u64;
        fn next_f(seed: &mut u64) -> f32 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((*seed >> 40) as i32) - (1 << 23);
            (bits as f32) / ((1 << 23) as f32)
        }
        fn push_lin(buf: &mut Vec<u8>, seed: &mut u64, in_d: usize, out_d: usize) {
            let scale = 0.05 / (in_d as f32).sqrt();
            for _ in 0..out_d {
                buf.extend_from_slice(&scale.to_le_bytes());
            }
            for _ in 0..out_d {
                buf.extend_from_slice(&(next_f(seed) * 0.05).to_le_bytes());
            }
            for _ in 0..(out_d * in_d) {
                let q = (next_f(seed) * 127.0).round().clamp(-127.0, 127.0) as i8;
                buf.push(q as u8);
            }
        }
        fn push_norm(buf: &mut Vec<u8>, seed: &mut u64, d: usize) {
            for _ in 0..d {
                buf.extend_from_slice(&(1.0 + next_f(seed) * 0.1).to_le_bytes());
            }
        }

        push_lin(&mut buf, &mut seed, OBS_DIM, d_model);
        for _ in 0..n_layers {
            push_norm(&mut buf, &mut seed, d_model);
            push_lin(&mut buf, &mut seed, d_model, d_model);
            push_lin(&mut buf, &mut seed, d_model, d_model);
            push_lin(&mut buf, &mut seed, d_model, d_model);
            push_lin(&mut buf, &mut seed, d_model, d_model);
            push_norm(&mut buf, &mut seed, d_model);
            push_lin(&mut buf, &mut seed, d_model, ffn_dim);
            push_lin(&mut buf, &mut seed, d_model, ffn_dim);
            push_lin(&mut buf, &mut seed, ffn_dim, d_model);
        }
        push_norm(&mut buf, &mut seed, d_model);
        push_lin(&mut buf, &mut seed, d_model, TOTAL_LOGITS);
        buf
    }

    #[test]
    fn hsp2_rope_transformer_parse_step_and_relative_invariance() {
        let bytes = make_synthetic_hsp2(32, 2, 4, 64, 8, 100.0);
        let pol = SavantPolicy::from_bytes(&bytes).expect("parse valid HSP2");
        assert_eq!(pol.d_model(), 32);
        assert_eq!(pol.n_layers(), 2);
        assert_eq!(pol.n_heads(), 4);
        assert_eq!(pol.ctx_len(), 8);

        // Build a deterministic 24-frame sequence (> ctx_len to exercise sliding window)
        let mut seq = Vec::new();
        for t in 0..24usize {
            let mut obs = [0.0_f32; OBS_DIM];
            for i in 0..OBS_DIM {
                obs[i] = ((t * 17 + i * 31) as f32 * 0.03).sin();
            }
            seq.push(obs);
        }

        // Run streaming from pos=0
        let mut cache_a = SavantPolicyCache::default();
        let mut logits_a = Vec::new();
        for obs in &seq {
            let out = pol.forward_step(obs, &mut cache_a);
            assert!(out.iter().all(|v| v.is_finite()));
            logits_a.push(out);
        }
        assert_eq!(cache_a.len(), 8);

        // Run the EXACT same sequence after advancing pos by 1000 warm-up frames and resetting via
        // shifted RoPE pos (testing pure relative-position invariance of RoPE + sliding window).
        // After L * (ctx_len - 1) = 2 * 7 = 14 steps of identical observations, outputs must match!
        let mut cache_b = SavantPolicyCache::default();
        let mut dummy = [0.0_f32; OBS_DIM];
        dummy[0] = 1.0;
        for _ in 0..50 {
            pol.forward_step(&dummy, &mut cache_b);
        }
        let mut logits_b = Vec::new();
        for obs in &seq {
            logits_b.push(pol.forward_step(obs, &mut cache_b));
        }

        // For t >= L * (ctx_len - 1) = 14, all dependencies in both caches are strictly inside `seq`,
        // so outputs must match to floating-point precision!
        for t in 14..24 {
            for j in 0..TOTAL_LOGITS {
                let diff = (logits_a[t][j] - logits_b[t][j]).abs();
                assert!(
                    diff < 1e-4,
                    "RoPE sliding-window mismatch at t={t} j={j}: a={} b={} diff={diff}",
                    logits_a[t][j],
                    logits_b[t][j]
                );
            }
        }

        // Cache reset restores exact step-0 behavior
        cache_a.reset();
        assert!(cache_a.is_empty());
        let after_reset = pol.forward_step(&seq[0], &mut cache_a);
        for j in 0..TOTAL_LOGITS {
            assert!((after_reset[j] - logits_a[0][j]).abs() < 1e-6);
        }
    }

    #[test]
    fn hsp2_rejects_invalid_magic() {
        let mut bad = make_synthetic_hsp2(32, 2, 4, 64, 8, 100.0);
        bad[3] = b'X';
        assert!(SavantPolicy::from_bytes(&bad).is_none());
    }

    #[test]
    fn obs_and_action_handle_experimental_arenas_and_mayhem() {
        use crate::arena::{ARENAS, ArenaOpts};
        use crate::physics::Car;
        use crate::powers::PowerSystem;

        for arena_idx in [0usize, 18, 19, 20] {
            let mut w = World::with_seed(ARENAS[arena_idx], &ArenaOpts::default(), 42 + arena_idx as u64);
            w.powers = Some(PowerSystem::new(10.0));
            let c0 = w.add_car(Car::new(0, 0, "A", 0));
            let c1 = w.add_car(Car::new(1, 1, "B", 0));
            w.cars[c0].place(4.0, 0.35, -10.0, 0.0);
            w.cars[c1].place(-4.0, 0.35, 10.0, std::f64::consts::PI);
            w.cars[c0].item = Some("punch");
            w.cars[c1].ov_t = 3.0;
            w.ball.reset(2.0, 2.0, 0.0);
            w.ball.ice_t = 1.5;
            w.ball.anti_t = 2.0;

            let mut obs = [0.0_f32; OBS_DIM];
            extract_obs(&w, c0, &Controls::default(), &mut obs);
            assert!(obs.iter().all(|v| v.is_finite()));
            // Mayhem active flag
            assert_eq!(obs[110], 1.0);
            // Experimental arena flags
            assert_eq!(obs[100], if arena_idx == 18 { 1.0 } else { 0.0 });
            assert_eq!(obs[101], if arena_idx == 19 { 1.0 } else { 0.0 });
            assert_eq!(obs[102], if arena_idx == 20 { 1.0 } else { 0.0 });
            assert!(policy_supports_world(&w));
        }

        let ctl = Controls {
            throttle: 1.0,
            steer: -0.5,
            pitch: 1.0,
            yaw: -1.0,
            roll: 0.0,
            jump: true,
            boost: true,
            handbrake: false,
            air_roll_free: false,
            item: true,
        };
        let enc = encode_action(&ctl);
        let dec = decode_action(&enc);
        assert!(dec.item);
        assert!(dec.jump);
        assert!(dec.boost);
        assert!(!dec.handbrake);

        // Fine-grained action space: 21-level (0.1-step) axes round-trip exactly.
        assert_eq!(ACT_BRANCHES, [21, 21, 21, 21, 21, 2, 2, 2, 2]);
        assert_eq!(TOTAL_LOGITS, 113);
        for v in [-1.0, -0.7, -0.3, -0.1, 0.0, 0.1, 0.3, 0.7, 1.0] {
            let enc = encode_action(&Controls { throttle: v, ..ctl });
            let dec = decode_action(&enc);
            assert!((dec.throttle - v).abs() < 1e-9, "throttle {v} -> {dec:?}");
            assert!((dec.steer - ctl.steer).abs() < 1e-9);
        }
        // Max steering granularity is 0.1.
        let lo = decode_action(&[0, 10, 10, 10, 10, 0, 0, 0, 0]);
        let hi = decode_action(&[0, 11, 10, 10, 10, 0, 0, 0, 0]);
        assert!((hi.steer - lo.steer - 0.1).abs() < 1e-9);
    }
}
