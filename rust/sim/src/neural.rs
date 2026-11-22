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
pub const NEURAL_FORMAT: &str = "hyperstrike_energy_rope_v3";
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
    mlp_ratio: usize,
    entity_steps: usize,
    temporal_steps: usize,
    seq_len: usize,
    #[serde(default)]
    symmetric_k: bool,
    action_n_bins: Vec<usize>,
    tensors: Vec<TensorDesc>,
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

/// Computes `out = W^T @ x` where `W` has shape `[rows, cols]`, `x` has len `rows`, `out` has len `cols`.
#[inline]
fn matvec_t(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    out[..cols].fill(0.0);
    let chunks = cols / 4;
    for r in 0..rows {
        let xr = x[r];
        if xr == 0.0 {
            continue;
        }
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..chunks {
            let i = c * 4;
            out[i] += xr * row[i];
            out[i + 1] += xr * row[i + 1];
            out[i + 2] += xr * row[i + 2];
            out[i + 3] += xr * row[i + 3];
        }
        for i in (chunks * 4)..cols {
            out[i] += xr * row[i];
        }
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

struct EnergyBlock {
    norm: RmsNorm,
    w_q: Vec<f32>,  // [dim, dim]
    w_k: Vec<f32>,  // [dim, dim]
    w_m: Vec<f32>,  // [mlp_dim, dim]
    mlp_dim: usize,
    beta: Vec<f32>, // [num_heads] = exp(log_beta)
    dt: f32,
    steps: usize,
    symmetric_k: bool,
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
    entity_block: EnergyBlock,
    entity_norm: RmsNorm,
    temporal_block: EnergyBlock,
    final_norm: RmsNorm,
    action_heads: Vec<Linear>,
    inv_freq: Vec<f64>, // [head_dim / 2]
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

        let load_energy = |prefix: &str, steps: usize, symmetric_k: bool| -> Result<EnergyBlock, String> {
            let norm = load_norm(&format!("{prefix}.norm"))?;
            let (_, w_q) = get_tensor(&format!("{prefix}.w_q"))?;
            let (_, w_k) = get_tensor(&format!("{prefix}.w_k"))?;
            let (m_shape, w_m) = get_tensor(&format!("{prefix}.w_m"))?;
            let (_, log_beta) = get_tensor(&format!("{prefix}.log_beta"))?;
            let (_, dt_v) = get_tensor(&format!("{prefix}.dt"))?;
            let beta = log_beta.into_iter().map(|v| v.exp()).collect();
            Ok(EnergyBlock {
                norm,
                w_q,
                w_k,
                w_m,
                mlp_dim: m_shape[0],
                beta,
                dt: dt_v[0],
                steps,
                symmetric_k,
            })
        };

        let d_model = manifest.d_model;
        let n_heads = manifest.n_heads;
        let head_dim = d_model / n_heads;
        let seq_len = manifest.seq_len;

        let entity_block = load_energy("entity_block", manifest.entity_steps, manifest.symmetric_k)?;
        let temporal_block = load_energy("temporal_block", manifest.temporal_steps, false)?;
        let max_mlp_dim = entity_block
            .mlp_dim
            .max(temporal_block.mlp_dim)
            .max(d_model * manifest.mlp_ratio);

        let mut action_heads = Vec::with_capacity(ACTION_DIM);
        for i in 0..ACTION_DIM {
            action_heads.push(load_linear(&format!("action_heads.{i}"), true)?);
        }

        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for j in 0..half {
            inv_freq.push(1.0_f64 / 10000.0_f64.powf((2 * j) as f64 / head_dim as f64));
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
            entity_block,
            entity_norm: load_norm("entity_norm")?,
            temporal_block,
            final_norm: load_norm("final_norm")?,
            action_heads,
            inv_freq,
        })
    }

    /// Split-half RoPE matching `x1, x2 = x.chunk(2, dim=-1); cat([x1*cos - x2*sin, x1*sin + x2*cos])`.
    /// Pass `invert = true` for `apply_rope(x, cos, -sin)`.
    #[inline]
    fn apply_rope_pos(&self, vec: &mut [f32], pos: usize, invert: bool) {
        let half = self.head_dim / 2;
        let p = pos as f64;
        let sign = if invert { -1.0_f32 } else { 1.0_f32 };
        for h in 0..self.n_heads {
            let base = h * self.head_dim;
            for j in 0..half {
                let angle = p * self.inv_freq[j];
                let c = angle.cos() as f32;
                let s = (angle.sin() as f32) * sign;
                let x1 = vec[base + j];
                let x2 = vec[base + half + j];
                vec[base + j] = x1 * c - x2 * s;
                vec[base + half + j] = x1 * s + x2 * c;
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
        let nh = self.n_heads;
        let hd = self.head_dim;

        // Compact only active tokens
        let mut n_tok = 0_usize;

        self.self_proj.forward(self_obs, &mut scratch.tokens[0..d]);
        for i in 0..d {
            scratch.tokens[i] += self.type_emb[i];
        }
        n_tok += 1;

        let off = n_tok * d;
        self.ball_proj
            .forward(ball_obs, &mut scratch.tokens[off..off + d]);
        for i in 0..d {
            scratch.tokens[off + i] += self.type_emb[d + i];
        }
        n_tok += 1;

        for k in 0..OTHER_CARS {
            if !other_mask[k] {
                continue;
            }
            let off = n_tok * d;
            self.other_proj
                .forward(&other_obs[k], &mut scratch.tokens[off..off + d]);
            for i in 0..d {
                scratch.tokens[off + i] += self.type_emb[2 * d + i];
            }
            n_tok += 1;
        }

        for k in 0..NUM_PADS_OBS {
            let off = n_tok * d;
            self.pad_proj
                .forward(&pads_obs[k], &mut scratch.tokens[off..off + d]);
            for i in 0..d {
                scratch.tokens[off + i] += self.type_emb[3 * d + i];
            }
            n_tok += 1;
        }

        let blk = &self.entity_block;
        let mlp_d = blk.mlp_dim;
        for step in 0..blk.steps {
            let is_last = step + 1 == blk.steps;
            let q_tokens = if is_last && !blk.symmetric_k { 1 } else { n_tok };

            for s in 0..n_tok {
                blk.norm
                    .forward(&scratch.tokens[s * d..(s + 1) * d], &mut scratch.g_tok[s * d..(s + 1) * d]);
                let gs = &scratch.g_tok[s * d..(s + 1) * d];
                if s < q_tokens || blk.symmetric_k {
                    matvec(&blk.w_q, None, d, d, gs, &mut scratch.q[s * d..(s + 1) * d]);
                }
                matvec(&blk.w_k, None, d, d, gs, &mut scratch.k[s * d..(s + 1) * d]);
            }

            let upd_tokens = if is_last { 1 } else { n_tok };
            for s in 0..upd_tokens {
                // Attention gradient wrt Q: softmax((q * beta) @ k^T) @ k
                scratch.grad_buf[..d].fill(0.0);
                for h in 0..nh {
                    let beta = blk.beta[h];
                    let q_off = s * d + h * hd;
                    let mut max_sc = f32::NEG_INFINITY;
                    for t in 0..n_tok {
                        let k_off = t * d + h * hd;
                        let mut dot = 0.0_f32;
                        for j in 0..hd {
                            dot += scratch.q[q_off + j] * scratch.k[k_off + j];
                        }
                        let sc = dot * beta;
                        scratch.scores[t] = sc;
                        if sc > max_sc {
                            max_sc = sc;
                        }
                    }
                    let mut denom = 0.0_f32;
                    for sc in scratch.scores.iter_mut().take(n_tok) {
                        let e = (*sc - max_sc).exp();
                        *sc = e;
                        denom += e;
                    }
                    let inv_d = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    let out_off = h * hd;
                    for t in 0..n_tok {
                        let w = scratch.scores[t] * inv_d;
                        let k_off = t * d + h * hd;
                        for j in 0..hd {
                            scratch.grad_buf[out_off + j] += w * scratch.k[k_off + j];
                        }
                    }
                }
                matvec_t(&blk.w_q, d, d, &scratch.grad_buf[..d], &mut scratch.d_attn);

                if blk.symmetric_k {
                    scratch.grad_buf[..d].fill(0.0);
                    for h in 0..nh {
                        let beta = blk.beta[h];
                        let k_off = s * d + h * hd;
                        let mut max_sc = f32::NEG_INFINITY;
                        for t in 0..n_tok {
                            let q_off = t * d + h * hd;
                            let mut dot = 0.0_f32;
                            for j in 0..hd {
                                dot += scratch.k[k_off + j] * scratch.q[q_off + j];
                            }
                            let sc = dot * beta;
                            scratch.scores[t] = sc;
                            if sc > max_sc {
                                max_sc = sc;
                            }
                        }
                        let mut denom = 0.0_f32;
                        for sc in scratch.scores.iter_mut().take(n_tok) {
                            let e = (*sc - max_sc).exp();
                            *sc = e;
                            denom += e;
                        }
                        let inv_d = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                        let out_off = h * hd;
                        for t in 0..n_tok {
                            let w = scratch.scores[t] * inv_d;
                            let q_off = t * d + h * hd;
                            for j in 0..hd {
                                scratch.grad_buf[out_off + j] += w * scratch.q[q_off + j];
                            }
                        }
                    }
                    matvec_t(&blk.w_k, d, d, &scratch.grad_buf[..d], &mut scratch.d_attn_k);
                    for i in 0..d {
                        scratch.d_attn[i] += scratch.d_attn_k[i];
                    }
                }

                // FFN energy gradient: W_m^T @ gelu(W_m @ g)
                let gs = &scratch.g_tok[s * d..(s + 1) * d];
                matvec(&blk.w_m, None, mlp_d, d, gs, &mut scratch.mlp_h[..mlp_d]);
                for v in scratch.mlp_h[..mlp_d].iter_mut() {
                    *v = gelu_tanh(*v);
                }
                matvec_t(&blk.w_m, mlp_d, d, &scratch.mlp_h[..mlp_d], &mut scratch.d_ffn);

                let row = &mut scratch.tokens[s * d..(s + 1) * d];
                for i in 0..d {
                    row[i] += blk.dt * (scratch.d_attn[i] + scratch.d_ffn[i]);
                }
            }
        }

        self.entity_norm.forward(&scratch.tokens[0..d], out_emb);
    }

    /// Exact per-energy-step RoPE K-cache for causal temporal UnifiedEnergyBlock:
    /// projects only the single new token at each energy step (O(steps) instead of O(steps * T)).
    fn step_temporal_kv(
        &self,
        frame_emb: &[f32],
        step_pos: usize,
        k_cache: &mut [f32], // [temporal_steps * seq_len * d]
        scratch: &mut NeuralScratch,
    ) -> Controls {
        let d = self.d_model;
        let nh = self.n_heads;
        let hd = self.head_dim;
        let t_max = self.seq_len;
        let slot = step_pos % t_max;
        let valid_len = (step_pos + 1).min(t_max);
        let blk = &self.temporal_block;
        let mlp_d = blk.mlp_dim;

        scratch.cur_x.copy_from_slice(frame_emb);

        for st in 0..blk.steps {
            let layer_off = st * t_max * d;
            blk.norm.forward(&scratch.cur_x, &mut scratch.norm_buf);

            matvec(&blk.w_q, None, d, d, &scratch.norm_buf, &mut scratch.q[..d]);
            self.apply_rope_pos(&mut scratch.q[..d], step_pos, false);

            let k_slot = layer_off + slot * d;
            matvec(&blk.w_k, None, d, d, &scratch.norm_buf, &mut k_cache[k_slot..k_slot + d]);
            self.apply_rope_pos(&mut k_cache[k_slot..k_slot + d], step_pos, false);

            scratch.grad_buf[..d].fill(0.0);
            for h in 0..nh {
                let beta = blk.beta[h];
                let q_off = h * hd;
                let mut max_sc = f32::NEG_INFINITY;
                for idx in 0..valid_len {
                    let k_off = layer_off + idx * d + h * hd;
                    let mut dot = 0.0_f32;
                    for j in 0..hd {
                        dot += scratch.q[q_off + j] * k_cache[k_off + j];
                    }
                    let sc = dot * beta;
                    scratch.scores[idx] = sc;
                    if sc > max_sc {
                        max_sc = sc;
                    }
                }
                let mut denom = 0.0_f32;
                for sc in scratch.scores.iter_mut().take(valid_len) {
                    let e = (*sc - max_sc).exp();
                    *sc = e;
                    denom += e;
                }
                let inv_d = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let out_off = h * hd;
                for idx in 0..valid_len {
                    let w = scratch.scores[idx] * inv_d;
                    let k_off = layer_off + idx * d + h * hd;
                    for j in 0..hd {
                        scratch.grad_buf[out_off + j] += w * k_cache[k_off + j];
                    }
                }
            }

            // Inverse RoPE on attention gradient before projecting back with W_q^T
            self.apply_rope_pos(&mut scratch.grad_buf[..d], step_pos, true);
            matvec_t(&blk.w_q, d, d, &scratch.grad_buf[..d], &mut scratch.d_attn);

            matvec(&blk.w_m, None, mlp_d, d, &scratch.norm_buf, &mut scratch.mlp_h[..mlp_d]);
            for v in scratch.mlp_h[..mlp_d].iter_mut() {
                *v = gelu_tanh(*v);
            }
            matvec_t(&blk.w_m, mlp_d, d, &scratch.mlp_h[..mlp_d], &mut scratch.d_ffn);

            for i in 0..d {
                scratch.cur_x[i] += blk.dt * (scratch.d_attn[i] + scratch.d_ffn[i]);
            }
        }

        self.final_norm
            .forward(&scratch.cur_x, &mut scratch.norm_buf);

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
    tokens: Vec<f32>,
    g_tok: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    scores: Vec<f32>,
    norm_buf: Vec<f32>,
    grad_buf: Vec<f32>,
    d_attn: Vec<f32>,
    d_attn_k: Vec<f32>,
    mlp_h: Vec<f32>,
    d_ffn: Vec<f32>,
    cur_x: Vec<f32>,
    emb: Vec<f32>,
}

impl NeuralScratch {
    fn new(policy: &NeuralPolicy) -> Self {
        let d = policy.d_model;
        let max_s = TOTAL_TOKENS.max(policy.seq_len);
        Self {
            tokens: vec![0.0; TOTAL_TOKENS * d],
            g_tok: vec![0.0; TOTAL_TOKENS * d],
            q: vec![0.0; TOTAL_TOKENS * d],
            k: vec![0.0; TOTAL_TOKENS * d],
            scores: vec![0.0; max_s],
            norm_buf: vec![0.0; d],
            grad_buf: vec![0.0; d],
            d_attn: vec![0.0; d],
            d_attn_k: vec![0.0; d],
            mlp_h: vec![0.0; policy.max_mlp_dim],
            d_ffn: vec![0.0; d],
            cur_x: vec![0.0; d],
            emb: vec![0.0; d],
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuralBotState {
    k_cache: Vec<f32>,
    step_pos: usize,
    tick_accum: u8,
    stagger_applied: bool,
    cached_controls: Controls,
    scratch: NeuralScratch,
}

impl NeuralBotState {
    pub fn new(policy: &NeuralPolicy) -> Self {
        let cache_len = policy.temporal_block.steps * policy.seq_len * policy.d_model;
        Self {
            k_cache: vec![0.0_f32; cache_len],
            step_pos: 0,
            tick_accum: TICKS_PER_ACTION,
            stagger_applied: false,
            cached_controls: Controls::default(),
            scratch: NeuralScratch::new(policy),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.step_pos = 0;
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
        self.cached_controls =
            policy.step_temporal_kv(&emb, self.step_pos, &mut self.k_cache, &mut self.scratch);
        self.scratch.emb = emb;
        self.step_pos += 1;
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
