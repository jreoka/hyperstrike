use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use glam::DVec3;
use serde::Deserialize;

use crate::math_ext::Controls;
use crate::world::World;

const MAX_CARS: usize = 10;
const OTHER_CARS: usize = MAX_CARS - 1;
const NUM_PADS_OBS: usize = 6;
const SELF_DIM: usize = 24;
const BALL_DIM: usize = 16;
const OTHER_DIM: usize = 18;
const PAD_DIM: usize = 4;
const TOTAL_TOKENS: usize = 1 + 1 + OTHER_CARS + NUM_PADS_OBS; // 17
const TICKS_PER_ACTION: u8 = 8;

const ACTION_BIN_VALUES: [&[f64]; 8] = [
    &[-1.0, 0.0, 1.0],              // throttle
    &[-1.0, -0.5, 0.0, 0.5, 1.0],   // steer
    &[-1.0, 0.0, 1.0],              // pitch
    &[-1.0, 0.0, 1.0],              // yaw
    &[-1.0, 0.0, 1.0],              // roll
    &[0.0, 1.0],                    // jump
    &[0.0, 1.0],                    // boost
    &[0.0, 1.0],                    // handbrake
];

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
    n_entity_layers: usize,
    n_temporal_layers: usize,
    seq_len: usize,
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
        let in_d = self.in_dim;
        let bias = self.bias.as_deref();
        for o in 0..self.out_dim {
            let row = &self.weight[o * in_d..(o + 1) * in_d];
            let mut s0 = bias.map_or(0.0_f32, |b| b[o]);
            let mut s1 = 0.0_f32;
            let mut s2 = 0.0_f32;
            let mut s3 = 0.0_f32;
            let chunks = in_d / 4;
            for c in 0..chunks {
                let i = c * 4;
                s0 += row[i] * x[i];
                s1 += row[i + 1] * x[i + 1];
                s2 += row[i + 2] * x[i + 2];
                s3 += row[i + 3] * x[i + 3];
            }
            let mut sum = (s0 + s1) + (s2 + s3);
            for i in (chunks * 4)..in_d {
                sum += row[i] * x[i];
            }
            out[o] = sum;
        }
    }

    #[inline]
    fn forward_slice(&self, x: &[f32], out: &mut [f32], row_start: usize, row_count: usize) {
        let in_d = self.in_dim;
        let bias = self.bias.as_deref();
        for k in 0..row_count {
            let o = row_start + k;
            let row = &self.weight[o * in_d..(o + 1) * in_d];
            let mut s0 = bias.map_or(0.0_f32, |b| b[o]);
            let mut s1 = 0.0_f32;
            let mut s2 = 0.0_f32;
            let mut s3 = 0.0_f32;
            let chunks = in_d / 4;
            for c in 0..chunks {
                let i = c * 4;
                s0 += row[i] * x[i];
                s1 += row[i + 1] * x[i + 1];
                s2 += row[i + 2] * x[i + 2];
                s3 += row[i + 3] * x[i + 3];
            }
            let mut sum = (s0 + s1) + (s2 + s3);
            for i in (chunks * 4)..in_d {
                sum += row[i] * x[i];
            }
            out[k] = sum;
        }
    }
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

struct SwiGlu {
    w12: Linear,
    w3: Linear,
}

impl SwiGlu {
    #[inline]
    fn forward(&self, x: &[f32], out: &mut [f32], buf_h2: &mut [f32], buf_h: &mut [f32]) {
        let h = self.w3.in_dim;
        self.w12.forward(x, &mut buf_h2[..2 * h]);
        for i in 0..h {
            let x1 = buf_h2[i];
            let x2 = buf_h2[h + i];
            let silu = x1 / (1.0 + (-x1).exp());
            buf_h[i] = silu * x2;
        }
        self.w3.forward(&buf_h[..h], out);
    }
}

struct TransformerBlock {
    norm1: RmsNorm,
    qkv: Linear,
    proj: Linear,
    norm2: RmsNorm,
    mlp: SwiGlu,
}

pub struct NeuralPolicy {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    max_mlp_h: usize,
    self_proj: Linear,
    ball_proj: Linear,
    other_proj: Linear,
    pad_proj: Linear,
    type_emb: Vec<f32>, // [4, d_model]
    entity_blocks: Vec<TransformerBlock>,
    entity_norm: RmsNorm,
    temporal_blocks: Vec<TransformerBlock>,
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
        if manifest.format != "hyperstrike_rope_v1" {
            return Err(format!("unsupported neural format: {}", manifest.format));
        }
        if manifest.action_n_bins.len() != 8 {
            return Err("expected 8 action heads".to_string());
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

        let load_block = |prefix: &str| -> Result<TransformerBlock, String> {
            Ok(TransformerBlock {
                norm1: load_norm(&format!("{prefix}.norm1"))?,
                qkv: load_linear(&format!("{prefix}.qkv"), false)?,
                proj: load_linear(&format!("{prefix}.proj"), false)?,
                norm2: load_norm(&format!("{prefix}.norm2"))?,
                mlp: SwiGlu {
                    w12: load_linear(&format!("{prefix}.mlp.w12"), false)?,
                    w3: load_linear(&format!("{prefix}.mlp.w3"), false)?,
                },
            })
        };

        let d_model = manifest.d_model;
        let n_heads = manifest.n_heads;
        let head_dim = d_model / n_heads;
        let seq_len = manifest.seq_len;

        let mut max_mlp_h = d_model;
        let mut entity_blocks = Vec::with_capacity(manifest.n_entity_layers);
        for i in 0..manifest.n_entity_layers {
            let blk = load_block(&format!("entity_blocks.{i}"))?;
            max_mlp_h = max_mlp_h.max(blk.mlp.w3.in_dim);
            entity_blocks.push(blk);
        }
        let mut temporal_blocks = Vec::with_capacity(manifest.n_temporal_layers);
        for i in 0..manifest.n_temporal_layers {
            let blk = load_block(&format!("temporal_blocks.{i}"))?;
            max_mlp_h = max_mlp_h.max(blk.mlp.w3.in_dim);
            temporal_blocks.push(blk);
        }
        let mut action_heads = Vec::with_capacity(8);
        for i in 0..8 {
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
            max_mlp_h,
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
            inv_freq,
        })
    }

    #[inline]
    fn apply_rope_pos(&self, vec: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        let p = pos as f64;
        for h in 0..self.n_heads {
            let base = h * self.head_dim;
            for j in 0..half {
                let angle = p * self.inv_freq[j];
                let c = angle.cos() as f32;
                let s = angle.sin() as f32;
                let x1 = vec[base + 2 * j];
                let x2 = vec[base + 2 * j + 1];
                vec[base + 2 * j] = x1 * c - x2 * s;
                vec[base + 2 * j + 1] = x1 * s + x2 * c;
            }
        }
    }

    /// Fast entity encoder: compacts active tokens (skips empty car slots) and on the final
    /// entity layer computes Q/proj/MLP only for token 0 ([SELF]), cutting FLOPs by ~6-10x.
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
        let scale = 1.0 / (hd as f32).sqrt();

        // Compact only active tokens into scratch.tokens[0..n_tok * d]
        let mut n_tok = 0_usize;

        // Token 0: SELF
        self.self_proj.forward(self_obs, &mut scratch.tokens[0..d]);
        for i in 0..d {
            scratch.tokens[i] += self.type_emb[i];
        }
        n_tok += 1;

        // Token 1: BALL
        let off = n_tok * d;
        self.ball_proj
            .forward(ball_obs, &mut scratch.tokens[off..off + d]);
        for i in 0..d {
            scratch.tokens[off + i] += self.type_emb[d + i];
        }
        n_tok += 1;

        // Active OTHER_CARS only
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

        // 6 nearest PADS
        for k in 0..NUM_PADS_OBS {
            let off = n_tok * d;
            self.pad_proj
                .forward(&pads_obs[k], &mut scratch.tokens[off..off + d]);
            for i in 0..d {
                scratch.tokens[off + i] += self.type_emb[3 * d + i];
            }
            n_tok += 1;
        }

        let n_layers = self.entity_blocks.len();
        for (li, blk) in self.entity_blocks.iter().enumerate() {
            let is_last = li + 1 == n_layers;
            let q_tokens = if is_last { 1 } else { n_tok };

            for s in 0..n_tok {
                blk.norm1
                    .forward(&scratch.tokens[s * d..(s + 1) * d], &mut scratch.norm_buf);
                if s < q_tokens {
                    blk.qkv.forward(&scratch.norm_buf, &mut scratch.qkv_buf);
                    scratch.q[s * d..(s + 1) * d].copy_from_slice(&scratch.qkv_buf[0..d]);
                    scratch.k[s * d..(s + 1) * d].copy_from_slice(&scratch.qkv_buf[d..2 * d]);
                    scratch.v[s * d..(s + 1) * d].copy_from_slice(&scratch.qkv_buf[2 * d..3 * d]);
                } else {
                    // Only need K and V for non-query tokens on the final layer
                    blk.qkv
                        .forward_slice(&scratch.norm_buf, &mut scratch.k[s * d..(s + 1) * d], d, d);
                    blk.qkv.forward_slice(
                        &scratch.norm_buf,
                        &mut scratch.v[s * d..(s + 1) * d],
                        2 * d,
                        d,
                    );
                }
            }

            scratch.attn_out[..q_tokens * d].fill(0.0);
            for s in 0..q_tokens {
                for h in 0..nh {
                    let q_off = s * d + h * hd;
                    let mut max_score = f32::NEG_INFINITY;
                    for t in 0..n_tok {
                        let k_off = t * d + h * hd;
                        let mut dot = 0.0_f32;
                        for j in 0..hd {
                            dot += scratch.q[q_off + j] * scratch.k[k_off + j];
                        }
                        let sc = dot * scale;
                        scratch.scores[t] = sc;
                        if sc > max_score {
                            max_score = sc;
                        }
                    }
                    let mut denom = 0.0_f32;
                    for sc in scratch.scores.iter_mut().take(n_tok) {
                        let e = (*sc - max_score).exp();
                        *sc = e;
                        denom += e;
                    }
                    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    let out_off = s * d + h * hd;
                    for t in 0..n_tok {
                        let w = scratch.scores[t] * inv_denom;
                        let v_off = t * d + h * hd;
                        for j in 0..hd {
                            scratch.attn_out[out_off + j] += w * scratch.v[v_off + j];
                        }
                    }
                }
            }

            for s in 0..q_tokens {
                blk.proj.forward(
                    &scratch.attn_out[s * d..(s + 1) * d],
                    &mut scratch.proj_buf,
                );
                let row = &mut scratch.tokens[s * d..(s + 1) * d];
                for i in 0..d {
                    row[i] += scratch.proj_buf[i];
                }
                blk.norm2.forward(row, &mut scratch.norm_buf);
                blk.mlp.forward(
                    &scratch.norm_buf,
                    &mut scratch.mlp_buf,
                    &mut scratch.buf_h2,
                    &mut scratch.buf_h,
                );
                for i in 0..d {
                    row[i] += scratch.mlp_buf[i];
                }
            }
        }

        self.entity_norm.forward(&scratch.tokens[0..d], out_emb);
    }

    /// Exact sliding-window RoPE KV-cached temporal step: projects only the single new token
    /// per layer (O(1) projections instead of O(seq_len)).
    fn step_temporal_kv(
        &self,
        frame_emb: &[f32],
        step_pos: usize,
        k_cache: &mut [f32], // [n_layers * seq_len * d]
        v_cache: &mut [f32], // [n_layers * seq_len * d]
        scratch: &mut NeuralScratch,
    ) -> Controls {
        let d = self.d_model;
        let nh = self.n_heads;
        let hd = self.head_dim;
        let t_max = self.seq_len;
        let scale = 1.0 / (hd as f32).sqrt();
        let slot = step_pos % t_max;
        let valid_len = (step_pos + 1).min(t_max);

        scratch.cur_x.copy_from_slice(frame_emb);

        for (li, blk) in self.temporal_blocks.iter().enumerate() {
            let layer_off = li * t_max * d;
            blk.norm1.forward(&scratch.cur_x, &mut scratch.norm_buf);
            blk.qkv.forward(&scratch.norm_buf, &mut scratch.qkv_buf);

            scratch.q[..d].copy_from_slice(&scratch.qkv_buf[0..d]);
            self.apply_rope_pos(&mut scratch.q[..d], step_pos);

            let k_slot = layer_off + slot * d;
            k_cache[k_slot..k_slot + d].copy_from_slice(&scratch.qkv_buf[d..2 * d]);
            self.apply_rope_pos(&mut k_cache[k_slot..k_slot + d], step_pos);
            v_cache[k_slot..k_slot + d].copy_from_slice(&scratch.qkv_buf[2 * d..3 * d]);

            scratch.attn_out[..d].fill(0.0);
            for h in 0..nh {
                let q_off = h * hd;
                let mut max_score = f32::NEG_INFINITY;
                for idx in 0..valid_len {
                    let k_off = layer_off + idx * d + h * hd;
                    let mut dot = 0.0_f32;
                    for j in 0..hd {
                        dot += scratch.q[q_off + j] * k_cache[k_off + j];
                    }
                    let sc = dot * scale;
                    scratch.scores[idx] = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }
                let mut denom = 0.0_f32;
                for sc in scratch.scores.iter_mut().take(valid_len) {
                    let e = (*sc - max_score).exp();
                    *sc = e;
                    denom += e;
                }
                let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                let out_off = h * hd;
                for idx in 0..valid_len {
                    let w = scratch.scores[idx] * inv_denom;
                    let v_off = layer_off + idx * d + h * hd;
                    for j in 0..hd {
                        scratch.attn_out[out_off + j] += w * v_cache[v_off + j];
                    }
                }
            }

            blk.proj
                .forward(&scratch.attn_out[..d], &mut scratch.proj_buf);
            for i in 0..d {
                scratch.cur_x[i] += scratch.proj_buf[i];
            }
            blk.norm2.forward(&scratch.cur_x, &mut scratch.norm_buf);
            blk.mlp.forward(
                &scratch.norm_buf,
                &mut scratch.mlp_buf,
                &mut scratch.buf_h2,
                &mut scratch.buf_h,
            );
            for i in 0..d {
                scratch.cur_x[i] += scratch.mlp_buf[i];
            }
        }

        self.final_norm
            .forward(&scratch.cur_x, &mut scratch.norm_buf);

        let mut vals = [0.0_f64; 8];
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
            item: false,
        }
    }
}

#[derive(Clone, Debug)]
struct NeuralScratch {
    tokens: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    norm_buf: Vec<f32>,
    qkv_buf: Vec<f32>,
    proj_buf: Vec<f32>,
    mlp_buf: Vec<f32>,
    buf_h2: Vec<f32>,
    buf_h: Vec<f32>,
    cur_x: Vec<f32>,
    emb: Vec<f32>,
}

impl NeuralScratch {
    fn new(policy: &NeuralPolicy) -> Self {
        let d = policy.d_model;
        let max_s = TOTAL_TOKENS.max(policy.seq_len);
        Self {
            tokens: vec![0.0; TOTAL_TOKENS * d],
            q: vec![0.0; TOTAL_TOKENS * d],
            k: vec![0.0; TOTAL_TOKENS * d],
            v: vec![0.0; TOTAL_TOKENS * d],
            attn_out: vec![0.0; TOTAL_TOKENS * d],
            scores: vec![0.0; max_s],
            norm_buf: vec![0.0; d],
            qkv_buf: vec![0.0; 3 * d],
            proj_buf: vec![0.0; d],
            mlp_buf: vec![0.0; d],
            buf_h2: vec![0.0; 2 * policy.max_mlp_h],
            buf_h: vec![0.0; policy.max_mlp_h],
            cur_x: vec![0.0; d],
            emb: vec![0.0; d],
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuralBotState {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    step_pos: usize,
    tick_accum: u8,
    stagger_applied: bool,
    cached_controls: Controls,
    scratch: NeuralScratch,
}

impl NeuralBotState {
    pub fn new(policy: &NeuralPolicy) -> Self {
        let cache_len = policy.temporal_blocks.len() * policy.seq_len * policy.d_model;
        Self {
            k_cache: vec![0.0_f32; cache_len],
            v_cache: vec![0.0_f32; cache_len],
            step_pos: 0,
            tick_accum: TICKS_PER_ACTION,
            stagger_applied: false,
            cached_controls: Controls::default(),
            scratch: NeuralScratch::new(policy),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.step_pos = 0;
        self.tick_accum = TICKS_PER_ACTION;
        self.stagger_applied = false;
        self.cached_controls = Controls::default();
    }

    pub fn step(&mut self, policy: &NeuralPolicy, car_idx: usize, w: &World) -> Controls {
        if !self.stagger_applied {
            // Spread multi-bot inference evenly across the 8 physics ticks to avoid CPU spikes
            self.tick_accum = TICKS_PER_ACTION - (car_idx as u8 % TICKS_PER_ACTION);
            self.stagger_applied = true;
        }
        if self.tick_accum < TICKS_PER_ACTION {
            self.tick_accum += 1;
            return self.cached_controls;
        }
        self.tick_accum = 1;

        let (self_obs, ball_obs, other_obs, other_mask, pads_obs) = extract_obs(car_idx, w);
        let mut emb = std::mem::take(&mut self.scratch.emb);
        policy.encode_frame_into(
            &self_obs,
            &ball_obs,
            &other_obs,
            &other_mask,
            &pads_obs,
            &mut self.scratch,
            &mut emb,
        );
        self.cached_controls = policy.step_temporal_kv(
            &emb,
            self.step_pos,
            &mut self.k_cache,
            &mut self.v_cache,
            &mut self.scratch,
        );
        self.scratch.emb = emb;
        self.step_pos += 1;
        self.cached_controls
    }
}

fn extract_obs(
    car_idx: usize,
    w: &World,
) -> (
    [f32; SELF_DIM],
    [f32; BALL_DIM],
    [[f32; OTHER_DIM]; OTHER_CARS],
    [bool; OTHER_CARS],
    [[f32; PAD_DIM]; NUM_PADS_OBS],
) {
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
    self_obs[16] = if car.on_ground { 1.0 } else { 0.0 };
    self_obs[17] = if car.has_flip { 1.0 } else { 0.0 };
    self_obs[18] = if car.flipping { 1.0 } else { 0.0 };
    self_obs[19] = if car.supersonic { 1.0 } else { 0.0 };
    self_obs[20] = if car.demolished { 1.0 } else { 0.0 };
    self_obs[21] = (team_size as f32 / 5.0).min(1.0);
    self_obs[22] = if w.kickoff { 1.0 } else { 0.0 };
    self_obs[23] = 0.5;

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
            oo[17] = if o.on_ground { 1.0 } else { 0.0 };
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
        pads_obs[k][2] = if p.big { 1.0 } else { 0.0 };
        pads_obs[k][3] = if p.active { 1.0 } else { 0.0 };
    }

    (self_obs, ball_obs, other_obs, other_mask, pads_obs)
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
