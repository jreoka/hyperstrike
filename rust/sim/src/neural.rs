use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use glam::DVec3;
use serde::Deserialize;

use crate::math_ext::Controls;
use crate::powers::POWERS;
use crate::world::World;

// Observation / action layout shared by live inference AND the RL trainer (`rust/rl_env`).
// Keep `rl/env.py` in sync with these constants.
pub const NEURAL_FORMAT: &str = "hyperstrike_rope_transformer_v1";
pub const MAX_CARS: usize = 10;
pub const OTHER_CARS: usize = MAX_CARS - 1;
pub const NUM_PADS_OBS: usize = 6;
pub const NUM_POWERS: usize = POWERS.len(); // 14
pub const SELF_DIM: usize = 24 + 2 + NUM_POWERS + 8; // 48
pub const BALL_DIM: usize = 19;
pub const OTHER_DIM: usize = 22;
pub const PAD_DIM: usize = 4;
pub const ACTION_DIM: usize = 9;
const TOTAL_TOKENS: usize = 1 + 1 + OTHER_CARS + NUM_PADS_OBS; // 17
const TICKS_PER_ACTION: u8 = 8;

pub const ACTION_BIN_VALUES: [&[f64]; ACTION_DIM] = [
    &[-1.0, 0.0, 1.0],              // throttle
    &[-1.0, -0.5, 0.0, 0.5, 1.0],   // steer
    &[-1.0, 0.0, 1.0],              // pitch
    &[-1.0, 0.0, 1.0],              // yaw
    &[-1.0, 0.0, 1.0],              // roll
    &[0.0, 1.0],                    // jump
    &[0.0, 1.0],                    // boost
    &[0.0, 1.0],                    // handbrake
    &[0.0, 1.0],                    // item (Mayhem power-up, rising edge triggers)
];

/// Convert raw float actions (as produced by the trainer / this policy) into `Controls`.
pub fn controls_from_actions(a: &[f32]) -> Controls {
    Controls {
        throttle: (a[0] as f64).clamp(-1.0, 1.0),
        steer: (a[1] as f64).clamp(-1.0, 1.0),
        pitch: (a[2] as f64).clamp(-1.0, 1.0),
        yaw: (a[3] as f64).clamp(-1.0, 1.0),
        roll: (a[4] as f64).clamp(-1.0, 1.0),
        jump: a[5] > 0.5,
        boost: a[6] > 0.5,
        handbrake: a[7] > 0.5,
        air_roll_free: false,
        item: a[8] > 0.5,
    }
}

/// Inverse of `controls_from_actions` (used to record expert actions for BC).
pub fn actions_from_controls(c: &Controls, out: &mut [f32]) {
    out[0] = c.throttle as f32;
    out[1] = c.steer as f32;
    out[2] = c.pitch as f32;
    out[3] = c.yaw as f32;
    out[4] = c.roll as f32;
    out[5] = if c.jump { 1.0 } else { 0.0 };
    out[6] = if c.boost { 1.0 } else { 0.0 };
    out[7] = if c.handbrake { 1.0 } else { 0.0 };
    out[8] = if c.item { 1.0 } else { 0.0 };
}

#[derive(Deserialize)]
struct TensorDesc {
    name: String,
    shape: Vec<usize>,
    byte_offset: usize,
    byte_len: usize,
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    d_model: usize,
    n_heads: usize,
    entity_layers: usize,
    temporal_layers: usize,
    seq_len: usize,
    #[serde(default = "default_rope_base")]
    rope_base: f64,
    action_n_bins: Vec<usize>,
    tensors: Vec<TensorDesc>,
}

fn default_rope_base() -> f64 {
    10000.0
}

struct Linear {
    weight: Vec<f32>, // [out_dim, in_dim]
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    #[inline]
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        matvec(&self.weight, self.bias.as_deref(), self.out_dim, self.in_dim, x, out);
    }
}

#[inline]
fn matvec(w: &[f32], bias: Option<&[f32]>, out_dim: usize, in_dim: usize, x: &[f32], out: &mut [f32]) {
    let chunks = in_dim / 4;
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut s0 = bias.map_or(0.0_f32, |b| b[o]);
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        for c in 0..chunks {
            let i = c * 4;
            s0 += row[i] * x[i];
            s1 += row[i + 1] * x[i + 1];
            s2 += row[i + 2] * x[i + 2];
            s3 += row[i + 3] * x[i + 3];
        }
        let mut sum = (s0 + s1) + (s2 + s3);
        for i in (chunks * 4)..in_dim {
            sum += row[i] * x[i];
        }
        out[o] = sum;
    }
}

#[inline]
fn gelu_tanh(z: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * z * (1.0 + (SQRT_2_OVER_PI * (z + 0.044715 * z * z * z)).tanh())
}

struct RmsNorm {
    weight: Vec<f32>,
}

impl RmsNorm {
    #[inline]
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let mut sq = 0.0_f32;
        for &v in x {
            sq += v * v;
        }
        let inv = 1.0 / (sq / n as f32 + 1e-6).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv * self.weight[i];
        }
    }
}

/// Pre-norm transformer block mirroring `RoPETransformerBlock` in `rl/model.py`:
/// `x += W_o·MHA(RMSNorm(x))`, `x += W_2·GELU(W_1·RMSNorm(x))`.
struct TfBlock {
    attn_norm: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    mlp_norm: RmsNorm,
    fc1: Linear,
    fc2: Linear,
}

pub struct NeuralPolicy {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    max_mlp_dim: usize,
    self_proj: Linear,
    ball_proj: Linear,
    other_proj: Linear,
    pad_proj: Linear,
    type_emb: Vec<f32>, // [4, d_model]
    entity_blocks: Vec<TfBlock>,
    entity_norm: RmsNorm,
    temporal_blocks: Vec<TfBlock>,
    final_norm: RmsNorm,
    action_heads: Vec<Linear>,
    rope_cos: Vec<f32>, // [seq_len, head_dim / 2]
    rope_sin: Vec<f32>, // [seq_len, head_dim / 2]
}

impl NeuralPolicy {
    pub fn load(bin_path: &Path, json_path: &Path) -> Result<Self, String> {
        let meta_str = fs::read_to_string(json_path)
            .map_err(|e| format!("failed to read {}: {e}", json_path.display()))?;
        let manifest: Manifest = serde_json::from_str(&meta_str)
            .map_err(|e| format!("invalid manifest {}: {e}", json_path.display()))?;
        if manifest.format != NEURAL_FORMAT {
            return Err(format!("unsupported neural format: {}", manifest.format));
        }
        if manifest.action_n_bins.len() != ACTION_DIM {
            return Err(format!("expected {ACTION_DIM} action heads"));
        }
        if manifest.n_heads == 0 || !manifest.d_model.is_multiple_of(manifest.n_heads) {
            return Err("d_model must be divisible by n_heads".into());
        }
        if manifest.seq_len == 0 {
            return Err("seq_len must be > 0".into());
        }
        let raw = fs::read(bin_path)
            .map_err(|e| format!("failed to read {}: {e}", bin_path.display()))?;

        let get_tensor = |name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
            let desc = manifest
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            let end = desc.byte_offset + desc.byte_len;
            if end > raw.len() || !desc.byte_len.is_multiple_of(4) {
                return Err(format!("corrupt byte range for tensor {name}"));
            }
            let slice = &raw[desc.byte_offset..end];
            let mut vals = Vec::with_capacity(desc.byte_len / 4);
            for chunk in slice.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok((desc.shape.clone(), vals))
        };

        let load_linear = |prefix: &str, has_bias: bool| -> Result<Linear, String> {
            let (shape, weight) = get_tensor(&format!("{prefix}.weight"))?;
            if shape.len() != 2 {
                return Err(format!("{prefix}.weight must be 2D"));
            }
            let bias = if has_bias {
                Some(get_tensor(&format!("{prefix}.bias"))?.1)
            } else {
                None
            };
            Ok(Linear {
                weight,
                bias,
                out_dim: shape[0],
                in_dim: shape[1],
            })
        };

        let load_norm = |prefix: &str| -> Result<RmsNorm, String> {
            Ok(RmsNorm {
                weight: get_tensor(&format!("{prefix}.weight"))?.1,
            })
        };

        let load_block = |prefix: &str| -> Result<TfBlock, String> {
            Ok(TfBlock {
                attn_norm: load_norm(&format!("{prefix}.attn_norm"))?,
                wq: load_linear(&format!("{prefix}.wq"), false)?,
                wk: load_linear(&format!("{prefix}.wk"), false)?,
                wv: load_linear(&format!("{prefix}.wv"), false)?,
                wo: load_linear(&format!("{prefix}.wo"), false)?,
                mlp_norm: load_norm(&format!("{prefix}.mlp_norm"))?,
                fc1: load_linear(&format!("{prefix}.fc1"), false)?,
                fc2: load_linear(&format!("{prefix}.fc2"), false)?,
            })
        };

        let d_model = manifest.d_model;
        let n_heads = manifest.n_heads;
        let head_dim = d_model / n_heads;
        let seq_len = manifest.seq_len;

        let entity_blocks = (0..manifest.entity_layers)
            .map(|i| load_block(&format!("entity_blocks.{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        let temporal_blocks = (0..manifest.temporal_layers)
            .map(|i| load_block(&format!("temporal_blocks.{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        let max_mlp_dim = entity_blocks
            .iter()
            .chain(temporal_blocks.iter())
            .map(|b| b.fc1.out_dim)
            .max()
            .unwrap_or(d_model);

        let mut action_heads = Vec::with_capacity(ACTION_DIM);
        for i in 0..ACTION_DIM {
            action_heads.push(load_linear(&format!("action_heads.{i}"), true)?);
        }

        // RoPE tables matching `build_rope_cache` (f32 inv_freq, f32 angles) in rl/model.py.
        let half = head_dim / 2;
        let mut rope_cos = vec![0.0_f32; seq_len * half];
        let mut rope_sin = vec![0.0_f32; seq_len * half];
        for j in 0..half {
            let inv_freq =
                (1.0_f64 / manifest.rope_base.powf((2 * j) as f64 / head_dim as f64)) as f32;
            for p in 0..seq_len {
                let angle = p as f32 * inv_freq;
                rope_cos[p * half + j] = angle.cos();
                rope_sin[p * half + j] = angle.sin();
            }
        }

        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            seq_len,
            max_mlp_dim,
            self_proj: load_linear("self_proj", true)?,
            ball_proj: load_linear("ball_proj", true)?,
            other_proj: load_linear("other_proj", true)?,
            pad_proj: load_linear("pad_proj", true)?,
            type_emb: get_tensor("type_emb.weight")?.1,
            entity_blocks,
            entity_norm: load_norm("entity_norm")?,
            temporal_blocks,
            final_norm: load_norm("final_norm")?,
            action_heads,
            rope_cos,
            rope_sin,
        })
    }

    /// Split-half RoPE matching `x1, x2 = x.chunk(2, dim=-1); cat([x1*cos - x2*sin, x1*sin + x2*cos])`,
    /// applied independently to every head of a `[d_model]` vector.
    #[inline]
    fn apply_rope_pos(&self, vec: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        let cos = &self.rope_cos[pos * half..(pos + 1) * half];
        let sin = &self.rope_sin[pos * half..(pos + 1) * half];
        for h in 0..self.n_heads {
            let base = h * self.head_dim;
            for j in 0..half {
                let x1 = vec[base + j];
                let x2 = vec[base + half + j];
                vec[base + j] = x1 * cos[j] - x2 * sin[j];
                vec[base + half + j] = x1 * sin[j] + x2 * cos[j];
            }
        }
    }

    /// Runs one transformer block in place over `x = [n, d]`.
    /// `only_row = Some(r)` updates just row `r` (used on the last layer, where only one
    /// output token is read); keys/values are still computed from every row.
    fn block_forward(
        &self,
        blk: &TfBlock,
        x: &mut [f32],
        n: usize,
        causal: bool,
        rope: bool,
        only_row: Option<usize>,
        s: &mut NeuralScratch,
    ) {
        let d = self.d_model;
        let nh = self.n_heads;
        let hd = self.head_dim;
        let mlp_d = blk.fc1.out_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        for t in 0..n {
            let row = t * d..(t + 1) * d;
            blk.attn_norm.forward(&x[row.clone()], &mut s.g[row.clone()]);
            blk.wk.forward(&s.g[row.clone()], &mut s.k[row.clone()]);
            blk.wv.forward(&s.g[row.clone()], &mut s.v[row.clone()]);
            if rope {
                self.apply_rope_pos(&mut s.k[row], t);
            }
        }

        let rows = match only_row {
            Some(r) => r..r + 1,
            None => 0..n,
        };
        for r in rows {
            blk.wq.forward(&s.g[r * d..(r + 1) * d], &mut s.q);
            if rope {
                self.apply_rope_pos(&mut s.q, r);
            }
            let n_keys = if causal { r + 1 } else { n };

            s.attn_o.fill(0.0);
            for h in 0..nh {
                let ho = h * hd;
                let mut max_sc = f32::NEG_INFINITY;
                for t in 0..n_keys {
                    let k_off = t * d + ho;
                    let mut dot = 0.0_f32;
                    for j in 0..hd {
                        dot += s.q[ho + j] * s.k[k_off + j];
                    }
                    let sc = dot * scale;
                    s.scores[t] = sc;
                    if sc > max_sc {
                        max_sc = sc;
                    }
                }
                let mut denom = 0.0_f32;
                for sc in s.scores.iter_mut().take(n_keys) {
                    let e = (*sc - max_sc).exp();
                    *sc = e;
                    denom += e;
                }
                let inv_d = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                for t in 0..n_keys {
                    let w = s.scores[t] * inv_d;
                    let v_off = t * d + ho;
                    for j in 0..hd {
                        s.attn_o[ho + j] += w * s.v[v_off + j];
                    }
                }
            }

            let xr = &mut x[r * d..(r + 1) * d];
            blk.wo.forward(&s.attn_o, &mut s.tmp);
            for i in 0..d {
                xr[i] += s.tmp[i];
            }

            blk.mlp_norm.forward(xr, &mut s.norm_buf);
            blk.fc1.forward(&s.norm_buf, &mut s.mlp_h[..mlp_d]);
            for v in s.mlp_h[..mlp_d].iter_mut() {
                *v = gelu_tanh(*v);
            }
            blk.fc2.forward(&s.mlp_h[..mlp_d], &mut s.tmp);
            for i in 0..d {
                xr[i] += s.tmp[i];
            }
        }
    }

    fn encode_frame_into(
        &self,
        self_obs: &[f32; SELF_DIM],
        ball_obs: &[f32; BALL_DIM],
        other_obs: &[[f32; OTHER_DIM]; OTHER_CARS],
        other_mask: &[bool; OTHER_CARS],
        pads_obs: &[[f32; PAD_DIM]; NUM_PADS_OBS],
        scratch: &mut NeuralScratch,
        out_emb: &mut [f32],
    ) {
        let d = self.d_model;
        let mut tokens = std::mem::take(&mut scratch.tokens);

        // Compact only active tokens (masked keys never influence the [SELF] token).
        let mut n_tok = 0_usize;
        let mut push = |proj: &Linear, inp: &[f32], ty: usize, tokens: &mut [f32]| {
            let off = n_tok * d;
            proj.forward(inp, &mut tokens[off..off + d]);
            for i in 0..d {
                tokens[off + i] += self.type_emb[ty * d + i];
            }
            n_tok += 1;
        };
        push(&self.self_proj, self_obs, 0, &mut tokens);
        push(&self.ball_proj, ball_obs, 1, &mut tokens);
        for k in 0..OTHER_CARS {
            if other_mask[k] {
                push(&self.other_proj, &other_obs[k], 2, &mut tokens);
            }
        }
        for pad in pads_obs.iter() {
            push(&self.pad_proj, pad, 3, &mut tokens);
        }

        let n_layers = self.entity_blocks.len();
        for (li, blk) in self.entity_blocks.iter().enumerate() {
            let only = if li + 1 == n_layers { Some(0) } else { None };
            self.block_forward(blk, &mut tokens[..n_tok * d], n_tok, false, false, only, scratch);
        }

        self.entity_norm.forward(&tokens[0..d], out_emb);
        scratch.tokens = tokens;
    }

    /// Runs the causal RoPE temporal encoder over the full `[seq_len, d]` window (positions
    /// 0..T-1, exactly like training) and greedily decodes the action heads at the last position.
    fn forward_temporal(&self, window: &[f32], scratch: &mut NeuralScratch) -> Controls {
        let d = self.d_model;
        let t_len = self.seq_len;
        let last = t_len - 1;

        let mut x = std::mem::take(&mut scratch.seq);
        x.copy_from_slice(window);
        let n_layers = self.temporal_blocks.len();
        for (li, blk) in self.temporal_blocks.iter().enumerate() {
            let only = if li + 1 == n_layers { Some(last) } else { None };
            self.block_forward(blk, &mut x, t_len, true, true, only, scratch);
        }
        self.final_norm
            .forward(&x[last * d..(last + 1) * d], &mut scratch.norm_buf);
        scratch.seq = x;

        let mut vals = [0.0_f64; ACTION_DIM];
        let mut logits = [0.0_f32; 8];
        for (k, head) in self.action_heads.iter().enumerate() {
            let nb = head.out_dim;
            head.forward(&scratch.norm_buf, &mut logits[..nb]);
            let mut best_idx = 0;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits[..nb].iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_idx = i;
                }
            }
            vals[k] = ACTION_BIN_VALUES[k][best_idx.min(ACTION_BIN_VALUES[k].len() - 1)];
        }

        Controls {
            throttle: vals[0],
            steer: vals[1],
            pitch: vals[2],
            yaw: vals[3],
            roll: vals[4],
            jump: vals[5] > 0.5,
            boost: vals[6] > 0.5,
            handbrake: vals[7] > 0.5,
            air_roll_free: false,
            item: vals[8] > 0.5,
        }
    }
}

#[derive(Clone, Debug)]
struct NeuralScratch {
    tokens: Vec<f32>,   // [TOTAL_TOKENS, d]
    seq: Vec<f32>,      // [seq_len, d]
    g: Vec<f32>,        // [max_rows, d]
    k: Vec<f32>,        // [max_rows, d]
    v: Vec<f32>,        // [max_rows, d]
    q: Vec<f32>,        // [d]
    attn_o: Vec<f32>,   // [d]
    tmp: Vec<f32>,      // [d]
    norm_buf: Vec<f32>, // [d]
    mlp_h: Vec<f32>,    // [max_mlp_dim]
    scores: Vec<f32>,   // [max_rows]
    emb: Vec<f32>,      // [d]
}

impl NeuralScratch {
    fn new(policy: &NeuralPolicy) -> Self {
        let d = policy.d_model;
        let max_rows = TOTAL_TOKENS.max(policy.seq_len);
        Self {
            tokens: vec![0.0; TOTAL_TOKENS * d],
            seq: vec![0.0; policy.seq_len * d],
            g: vec![0.0; max_rows * d],
            k: vec![0.0; max_rows * d],
            v: vec![0.0; max_rows * d],
            q: vec![0.0; d],
            attn_o: vec![0.0; d],
            tmp: vec![0.0; d],
            norm_buf: vec![0.0; d],
            mlp_h: vec![0.0; policy.max_mlp_dim],
            scores: vec![0.0; max_rows],
            emb: vec![0.0; d],
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuralBotState {
    /// Sliding window of the last `seq_len` frame embeddings, oldest first (mirrors the
    /// trainer's `HistoryBuffer`: on a fresh episode every slot is filled with the first frame).
    history: Vec<f32>,
    fresh: bool,
    tick_accum: u8,
    stagger_applied: bool,
    cached_controls: Controls,
    scratch: NeuralScratch,
}

impl NeuralBotState {
    pub fn new(policy: &NeuralPolicy) -> Self {
        Self {
            history: vec![0.0_f32; policy.seq_len * policy.d_model],
            fresh: true,
            tick_accum: TICKS_PER_ACTION,
            stagger_applied: false,
            cached_controls: Controls::default(),
            scratch: NeuralScratch::new(policy),
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.fresh = true;
        self.tick_accum = TICKS_PER_ACTION;
        self.stagger_applied = false;
        self.cached_controls = Controls::default();
    }

    pub fn step(&mut self, policy: &NeuralPolicy, car_idx: usize, w: &World) -> Controls {
        if !self.stagger_applied {
            self.tick_accum = TICKS_PER_ACTION - (car_idx as u8 % TICKS_PER_ACTION);
            self.stagger_applied = true;
        }
        if self.tick_accum < TICKS_PER_ACTION {
            self.tick_accum += 1;
            return self.cached_controls;
        }
        self.tick_accum = 1;

        let obs = extract_obs(car_idx, w);
        let mut emb = std::mem::take(&mut self.scratch.emb);
        policy.encode_frame_into(
            &obs.self_obs,
            &obs.ball_obs,
            &obs.other_obs,
            &obs.other_mask,
            &obs.pads_obs,
            &mut self.scratch,
            &mut emb,
        );

        let d = policy.d_model;
        if self.fresh {
            for slot in self.history.chunks_exact_mut(d) {
                slot.copy_from_slice(&emb);
            }
            self.fresh = false;
        } else {
            self.history.copy_within(d.., 0);
            let n = self.history.len();
            self.history[n - d..].copy_from_slice(&emb);
        }
        self.scratch.emb = emb;

        self.cached_controls = policy.forward_temporal(&self.history, &mut self.scratch);
        self.cached_controls
    }
}

/// Per-car observation in the trainer's exact layout (team 1 is mirrored so every car attacks +Z).
#[derive(Clone, Debug)]
pub struct CarObs {
    pub self_obs: [f32; SELF_DIM],
    pub ball_obs: [f32; BALL_DIM],
    pub other_obs: [[f32; OTHER_DIM]; OTHER_CARS],
    pub other_mask: [bool; OTHER_CARS],
    pub pads_obs: [[f32; PAD_DIM]; NUM_PADS_OBS],
}

#[inline]
fn b2f(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Build the observation for `car_idx`. This single function is used by both live inference and
/// `rust/rl_env`, so training and deployment can never drift apart.
///
/// Self layout: [0..24) car state (as v2), [24] mayhem on, [25] holding item,
/// [26..40) item one-hot (POWERS order), [40] item usable now (has valid target),
/// [41] time until next item, [42..47) spikes/magnet/overdrive/nitro/scrambled timers,
/// [47] ball stuck to me.
/// Ball extras: [16] iced, [17] zero-g, [18] stuck to (+1 teammate/me, -1 enemy, 0 free).
/// Other extras: [18] holds item, [19] spikes, [20] scrambled, [21] overdrive|nitro.
pub fn extract_obs(car_idx: usize, w: &World) -> CarObs {
    let mut self_obs = [0.0_f32; SELF_DIM];
    let mut ball_obs = [0.0_f32; BALL_DIM];
    let mut other_obs = [[0.0_f32; OTHER_DIM]; OTHER_CARS];
    let mut other_mask = [false; OTHER_CARS];
    let mut pads_obs = [[0.0_f32; PAD_DIM]; NUM_PADS_OBS];

    let car = &w.cars[car_idx];
    let inv_w = 1.0 / w.shape.w.max(1.0);
    let inv_l = 1.0 / w.shape.goal_line_z().max(1.0);
    let inv_h = 1.0 / 20.0;
    let team_size = w.cars.iter().filter(|c| c.team == car.team).count().max(1);
    let s = if car.team == 0 { 1.0 } else { -1.0 };
    let mayhem = w.powers.is_some();

    let c_pos = DVec3::new(car.pos.x * s, car.pos.y, car.pos.z * s);
    let c_vel = DVec3::new(car.vel.x * s, car.vel.y, car.vel.z * s);
    let c_fwd = DVec3::new(car.basis.f.x * s, car.basis.f.y, car.basis.f.z * s);
    let c_up = DVec3::new(car.basis.u.x * s, car.basis.u.y, car.basis.u.z * s);
    let c_ang = DVec3::new(car.ang_vel.x * s, car.ang_vel.y, car.ang_vel.z * s);

    self_obs[0] = (c_pos.x * inv_w) as f32;
    self_obs[1] = (c_pos.y * inv_h) as f32;
    self_obs[2] = (c_pos.z * inv_l) as f32;
    self_obs[3] = (c_vel.x / 23.0) as f32;
    self_obs[4] = (c_vel.y / 23.0) as f32;
    self_obs[5] = (c_vel.z / 23.0) as f32;
    self_obs[6] = c_fwd.x as f32;
    self_obs[7] = c_fwd.y as f32;
    self_obs[8] = c_fwd.z as f32;
    self_obs[9] = c_up.x as f32;
    self_obs[10] = c_up.y as f32;
    self_obs[11] = c_up.z as f32;
    self_obs[12] = (c_ang.x / 5.5) as f32;
    self_obs[13] = (c_ang.y / 5.5) as f32;
    self_obs[14] = (c_ang.z / 5.5) as f32;
    self_obs[15] = (car.boost / 100.0) as f32;
    self_obs[16] = b2f(car.on_ground);
    self_obs[17] = b2f(car.has_flip);
    self_obs[18] = b2f(car.flipping);
    self_obs[19] = b2f(car.supersonic);
    self_obs[20] = b2f(car.demolished);
    self_obs[21] = (team_size as f32 / 5.0).min(1.0);
    self_obs[22] = b2f(w.kickoff);
    // Live matches have no fixed episode horizon, so this is a constant in training too.
    self_obs[23] = 0.5;

    if mayhem {
        self_obs[24] = 1.0;
        if let Some(id) = car.item {
            self_obs[25] = 1.0;
            if let Some(pi) = POWERS.iter().position(|p| p.id == id) {
                self_obs[26 + pi] = 1.0;
            }
            let usable = w
                .powers
                .as_ref()
                .and_then(|ps| ps.target_for(car_idx, id, &w.cars, &w.ball))
                .is_some();
            self_obs[40] = b2f(usable);
        } else {
            self_obs[41] = (car.item_next.max(0.0) / 15.0).min(1.0) as f32;
        }
        self_obs[42] = (car.spike_t.max(0.0) / 8.0) as f32;
        self_obs[43] = (car.mag_t.max(0.0) / 8.0) as f32;
        self_obs[44] = (car.ov_t.max(0.0) / 8.0) as f32;
        self_obs[45] = (car.nitro_t.max(0.0) / 8.0) as f32;
        self_obs[46] = (car.scr_t.max(0.0) / 8.0) as f32;
        self_obs[47] = b2f(w.ball.stuck_to == Some(car_idx));
    }

    let b_pos = DVec3::new(w.ball.pos.x * s, w.ball.pos.y, w.ball.pos.z * s);
    let b_vel = DVec3::new(w.ball.vel.x * s, w.ball.vel.y, w.ball.vel.z * s);
    let b_ang = DVec3::new(w.ball.ang_vel.x * s, w.ball.ang_vel.y, w.ball.ang_vel.z * s);
    let rel_b_pos = b_pos - c_pos;
    let rel_b_vel = b_vel - c_vel;

    ball_obs[0] = (b_pos.x * inv_w) as f32;
    ball_obs[1] = (b_pos.y * inv_h) as f32;
    ball_obs[2] = (b_pos.z * inv_l) as f32;
    ball_obs[3] = (b_vel.x / 40.0) as f32;
    ball_obs[4] = (b_vel.y / 40.0) as f32;
    ball_obs[5] = (b_vel.z / 40.0) as f32;
    ball_obs[6] = (b_ang.x / 6.0) as f32;
    ball_obs[7] = (b_ang.y / 6.0) as f32;
    ball_obs[8] = (b_ang.z / 6.0) as f32;
    ball_obs[9] = (rel_b_pos.x / 40.0) as f32;
    ball_obs[10] = (rel_b_pos.y / 20.0) as f32;
    ball_obs[11] = (rel_b_pos.z / 40.0) as f32;
    ball_obs[12] = (rel_b_vel.x / 40.0) as f32;
    ball_obs[13] = (rel_b_vel.y / 40.0) as f32;
    ball_obs[14] = (rel_b_vel.z / 40.0) as f32;
    ball_obs[15] = (rel_b_pos.length() / 60.0) as f32;
    if mayhem {
        ball_obs[16] = (w.ball.ice_t.max(0.0) / 3.2).min(1.0) as f32;
        ball_obs[17] = (w.ball.anti_t.max(0.0) / 6.0).min(1.0) as f32;
        ball_obs[18] = match w.ball.stuck_to {
            Some(i) if i < w.cars.len() => {
                if w.cars[i].team == car.team { 1.0 } else { -1.0 }
            }
            _ => 0.0,
        };
    }

    let mut slot = 0;
    for (oi, o) in w.cars.iter().enumerate() {
        if oi == car_idx || slot >= OTHER_CARS {
            continue;
        }
        if !o.demolished {
            other_mask[slot] = true;
            let o_pos = DVec3::new(o.pos.x * s, o.pos.y, o.pos.z * s);
            let o_vel = DVec3::new(o.vel.x * s, o.vel.y, o.vel.z * s);
            let o_fwd = DVec3::new(o.basis.f.x * s, o.basis.f.y, o.basis.f.z * s);
            let rel_p = o_pos - c_pos;
            let rel_v = o_vel - c_vel;
            let oo = &mut other_obs[slot];
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
            oo[17] = b2f(o.on_ground);
            if mayhem {
                oo[18] = b2f(o.item.is_some());
                oo[19] = b2f(o.spike_t > 0.0);
                oo[20] = b2f(o.scr_t > 0.0);
                oo[21] = b2f(o.ov_t > 0.0 || o.nitro_t > 0.0);
            }
        }
        slot += 1;
    }

    let mut pad_dists: [(f64, usize); 40] = [(1e9, 0); 40];
    let np = w.pads.len().min(40);
    for (pi, p) in w.pads.iter().take(np).enumerate() {
        let dx = p.pos.x * s - c_pos.x;
        let dz = p.pos.z * s - c_pos.z;
        pad_dists[pi] = (dx * dx + dz * dz, pi);
    }
    let sp = &mut pad_dists[..np];
    sp.sort_by(|a, b| a.0.total_cmp(&b.0));
    for k in 0..NUM_PADS_OBS.min(np) {
        let p = &w.pads[sp[k].1];
        pads_obs[k][0] = ((p.pos.x * s - c_pos.x) / 40.0) as f32;
        pads_obs[k][1] = ((p.pos.z * s - c_pos.z) / 40.0) as f32;
        pads_obs[k][2] = b2f(p.big);
        pads_obs[k][3] = b2f(p.active);
    }

    CarObs {
        self_obs,
        ball_obs,
        other_obs,
        other_mask,
        pads_obs,
    }
}

fn candidate_checkpoint_pairs() -> Vec<(PathBuf, PathBuf)> {
    let mut pairs = Vec::new();
    if let Ok(custom) = env::var("HYPERSTRIKE_NEURAL_POLICY") {
        let p = PathBuf::from(custom);
        if p.is_dir() {
            pairs.push((p.join("neural_policy_best.bin"), p.join("neural_policy_best.json")));
            pairs.push((p.join("neural_policy.bin"), p.join("neural_policy.json")));
        } else {
            let json = p.with_extension("json");
            pairs.push((p, json));
        }
    }
    for base in [
        PathBuf::from("rl/checkpoints"),
        PathBuf::from("../../rl/checkpoints"),
    ] {
        pairs.push((
            base.join("neural_policy_best.bin"),
            base.join("neural_policy_best.json"),
        ));
        pairs.push((
            base.join("neural_policy.bin"),
            base.join("neural_policy.json"),
        ));
    }
    pairs
}

static GLOBAL_POLICY: OnceLock<Option<Arc<NeuralPolicy>>> = OnceLock::new();

pub fn global_policy() -> Option<Arc<NeuralPolicy>> {
    GLOBAL_POLICY
        .get_or_init(|| {
            for (bin_p, json_p) in candidate_checkpoint_pairs() {
                if bin_p.is_file() && json_p.is_file() {
                    match NeuralPolicy::load(&bin_p, &json_p) {
                        Ok(policy) => return Some(Arc::new(policy)),
                        Err(_) => continue,
                    }
                }
            }
            None
        })
        .clone()
}

pub fn is_available() -> bool {
    global_policy().is_some()
}
