//! Compact quantized MLP policy for `SAVANT` bots (trained via BC + PPO self-play).
//!
//! Shared by both offline RL training (`rust/rl_env`) and live inference (`BotAi`),
//! guaranteeing identical observation features and action decoding with zero train/serve skew.

use std::sync::OnceLock;

use glam::DVec3;

use crate::arena::GoalMode;
use crate::math_ext::{Controls, ThreeVec3Ext, clamp};
use crate::world::World;

pub const OBS_DIM: usize = 100;
pub const ACT_HEADS: usize = 8;
pub const ACT_BRANCHES: [usize; ACT_HEADS] = [3, 5, 3, 3, 3, 2, 2, 2];
pub const TOTAL_LOGITS: usize = 23;
/// Decision period in 120 Hz physics ticks (4 ticks = 30 Hz = 33.3 ms per action).
pub const POLICY_TICK_SKIP: u64 = 4;

pub const POLICY_MAGIC: &[u8; 4] = b"HSP1";

const THROTTLE_VALS: [f64; 3] = [-1.0, 0.0, 1.0];
const STEER_VALS: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
const AXIS3_VALS: [f64; 3] = [-1.0, 0.0, 1.0];

/// Canonicalize a world-space vector into team-attacking `+Z` frame (180° Y-rotation for team 1).
/// Proper rotation (`det = +1`), so `car.basis.to_local(...)` is invariant without sign flips.
#[inline]
fn canon_vec(v: DVec3, flip: f64) -> DVec3 {
    DVec3::new(v.x * flip, v.y, v.z * flip)
}

/// Extracts the 100-float team-symmetric observation vector for `car_idx`.
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
    let own_g = DVec3::new(0.0, 2.0, -flip * glz);
    let opp_g = DVec3::new(0.0, 2.0, flip * glz);
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

    let _ = prev_ctl;
    debug_assert_eq!(k, OBS_DIM);
}

#[inline]
pub fn decode_action(act: &[u8; ACT_HEADS]) -> Controls {
    let steer = STEER_VALS[(act[1] as usize).min(4)];
    let yaw = AXIS3_VALS[(act[3] as usize).min(2)];
    Controls {
        throttle: THROTTLE_VALS[(act[0] as usize).min(2)],
        steer,
        pitch: AXIS3_VALS[(act[2] as usize).min(2)],
        yaw,
        roll: AXIS3_VALS[(act[4] as usize).min(2)],
        jump: act[5] != 0,
        boost: act[6] != 0,
        handbrake: act[7] != 0,
        air_roll_free: false,
        item: false,
    }
}

#[inline]
fn quantize3(v: f64) -> u8 {
    if v < -0.33 {
        0
    } else if v > 0.33 {
        2
    } else {
        1
    }
}

#[inline]
fn quantize5(v: f64) -> u8 {
    let idx = ((clamp(v, -1.0, 1.0) + 1.0) * 2.0).round() as i32;
    idx.clamp(0, 4) as u8
}

#[inline]
pub fn encode_action(c: &Controls) -> [u8; ACT_HEADS] {
    [
        quantize3(c.throttle),
        quantize5(c.steer),
        quantize3(c.pitch),
        quantize3(c.yaw),
        quantize3(c.roll),
        u8::from(c.jump),
        u8::from(c.boost),
        u8::from(c.handbrake),
    ]
}

// ---------------------------------------------------------------------------------------------
// Quantized int8 MLP (`HSP1` binary format)
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
struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SavantPolicy {
    l1: QLinear,
    n1: LayerNorm,
    l2: QLinear,
    n2: LayerNorm,
    l3: QLinear,
    n3: LayerNorm,
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

    fn read_ln(&mut self, dim: usize) -> Option<LayerNorm> {
        let gamma = self.read_f32_vec(dim)?;
        let beta = self.read_f32_vec(dim)?;
        Some(LayerNorm { gamma, beta })
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

fn ln_silu_inplace(ln: &LayerNorm, x: &mut [f32]) {
    let n = x.len() as f32;
    let mut mean = 0.0_f32;
    for &v in x.iter() {
        mean += v;
    }
    mean /= n;
    let mut var = 0.0_f32;
    for &v in x.iter() {
        let d = v - mean;
        var += d * d;
    }
    let inv_std = 1.0 / (var / n + 1e-5).sqrt();
    for i in 0..x.len() {
        let norm = (x[i] - mean) * inv_std * ln.gamma[i] + ln.beta[i];
        x[i] = silu(norm);
    }
}

impl SavantPolicy {
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let mut r = Reader::new(bytes);
        if r.read_bytes(4)? != POLICY_MAGIC {
            return None;
        }
        let obs_d = r.read_u16()? as usize;
        let h1 = r.read_u16()? as usize;
        let h2 = r.read_u16()? as usize;
        let h3 = r.read_u16()? as usize;
        let out_d = r.read_u16()? as usize;
        let _flags = r.read_u16()?;
        if obs_d != OBS_DIM || out_d != TOTAL_LOGITS || h1 == 0 || h2 == 0 || h3 == 0 || h1 > 512 || h2 > 512 || h3 > 512 {
            return None;
        }
        let l1 = r.read_linear(obs_d, h1)?;
        let n1 = r.read_ln(h1)?;
        let l2 = r.read_linear(h1, h2)?;
        let n2 = r.read_ln(h2)?;
        let l3 = r.read_linear(h2, h3)?;
        let n3 = r.read_ln(h3)?;
        let head = r.read_linear(h3, out_d)?;
        Some(Self {
            l1,
            n1,
            l2,
            n2,
            l3,
            n3,
            head,
        })
    }

    pub fn forward(&self, obs: &[f32; OBS_DIM]) -> [f32; TOTAL_LOGITS] {
        let mut b1 = [0.0_f32; 512];
        let mut b2 = [0.0_f32; 512];
        let mut b3 = [0.0_f32; 512];
        let h1 = self.l1.out_dim;
        let h2 = self.l2.out_dim;
        let h3 = self.l3.out_dim;
        linear_forward(&self.l1, obs, &mut b1[..h1]);
        ln_silu_inplace(&self.n1, &mut b1[..h1]);
        linear_forward(&self.l2, &b1[..h1], &mut b2[..h2]);
        ln_silu_inplace(&self.n2, &mut b2[..h2]);
        linear_forward(&self.l3, &b2[..h2], &mut b3[..h3]);
        ln_silu_inplace(&self.n3, &mut b3[..h3]);
        let mut logits = [0.0_f32; TOTAL_LOGITS];
        linear_forward(&self.head, &b3[..h3], &mut logits);
        logits
    }

    pub fn act_greedy(&self, obs: &[f32; OBS_DIM]) -> [u8; ACT_HEADS] {
        let logits = self.forward(obs);
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

/// Returns true if the arena shape is supported by the trained neural policy.
#[inline]
pub fn policy_supports_world(w: &World) -> bool {
    w.shape.goal_mode == GoalMode::Standard
        && w.shape.fig8.is_none()
        && w.ball.stuck_to.is_none()
}
