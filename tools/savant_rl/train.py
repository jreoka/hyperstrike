#!/usr/bin/env python3
"""Zero-flag, autonomous from-scratch RL trainer for SAVANT (Causal RoPE Transformer).

Just run:
    python tools/savant_rl/train.py

Everything is handled automatically:
- Causal RoPE Transformer (Pre-RMSNorm + RoPE sliding-window attention + SwiGLU FFN) for both Actor and Critic.
- Exact zero-skew streaming KV-cache rollouts + prefix-padded sliding-window chunked PPO updates.
- Pure from-scratch raw-button RL (no imitation / BC warmstart): 0.1-granular
  analog-resolution axes (21 levels each) + binary buttons.
- Auto-resumes from `checkpoints/latest.pt` (or `best.pt`) if present.
- Auto-detects & auto-advances curriculum stages (Genesis -> Rising -> Competitive -> Apex).
- Trains forever until you press Ctrl+C (safely saves `latest.pt` on Ctrl+C).
- Evaluates & checkpoints every 50 updates.
- Every new best checkpoint is written to `best.pt` **and** exported to `rust/sim/assets/savant_policy.bin`
  (no opponent / score gate while training — the best model seen is always the live one).
- Automatically cleans up any old/stale checkpoint files, keeping only `latest.pt` and `best.pt`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import math
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RL_ENV_DIR = ROOT / "rust" / "rl_env"
DLL_PATH = RL_ENV_DIR / "target" / "release" / ("rl_env.dll" if os.name == "nt" else "librl_env.so")
OUT_MODEL = ROOT / "rust" / "sim" / "assets" / "savant_policy.bin"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
LATEST_PT = CKPT_DIR / "latest.pt"
BEST_PT = CKPT_DIR / "best.pt"

ARCH_TAG = "rope_transformer_v3"
POLICY_MAGIC = b"HSP2"

OBS_DIM = 146
# Fine-grained action space: throttle/steer/pitch/yaw/roll are 21-level axes
# (uniform 0.1 steps across [-1, 1], i.e. analog-thumbstick resolution),
# then jump/boost/handbrake/item are binary.
ACT_BRANCHES = (21, 21, 21, 21, 21, 2, 2, 2, 2)
ACT_HEADS = len(ACT_BRANCHES)
TOTAL_LOGITS = sum(ACT_BRANCHES)
EVAL_ARENA_EXPERIMENTAL_CYCLE = 99

# Causal RoPE Transformer hyperparameters
ACTOR_D_MODEL = 128
ACTOR_LAYERS = 2
ACTOR_HEADS = 4
ACTOR_FFN_DIM = 256

CRITIC_D_MODEL = 192
CRITIC_LAYERS = 2
CRITIC_HEADS = 6
CRITIC_FFN_DIM = 384

CTX_LEN = 16          # Sliding window of 16 frames (0.53s/layer; 1.03s effective across 2 layers at 30 Hz)
ROPE_THETA = 100.0    # Tuned for 1..16 frame relative temporal resolution
PREFIX_LEN = max(ACTOR_LAYERS, CRITIC_LAYERS) * (CTX_LEN - 1)  # 30: exact receptive field for zero train/serve skew

# Training hyperparameters (zero flags needed)
NUM_ENVS = 256
ROLLOUT_STEPS = 128
CHUNK_LEN = 64        # Divides ROLLOUT_STEPS cleanly; 2 chunks per agent per rollout
MINIBATCH_SIZE = 4096
MINIBATCH_CHUNKS = max(1, MINIBATCH_SIZE // CHUNK_LEN)
PPO_EPOCHS = 4
EVAL_EVERY = 50
GAMMA = 0.994
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
MAX_SNAPSHOTS = 24

# Stability & anti-stall controls (100% HSP2-compatible)
WARMUP_UPDATES = 25
TARGET_KL = 0.015        # Target total KL across 9 heads (~0.0017/head)
MAX_KL = 0.045           # Actor early-stop ceiling (critic always finishes all epochs)
CATASTROPHIC_KL = 0.20   # Abort even minibatch 0 if KL is truly explosive
LOG_RATIO_CLAMP = 10.0   # Prevents exp(log_ratio) overflow -> Inf/NaN
WEIGHT_DECAY = 1e-4      # Keeps attention Q/K & FFN weights bounded over long runs
LOGIT_Z_LOSS = 1e-4      # Prevents output logit explosion & dead softmax gradients
VALUE_HUBER_BETA = 10.0  # Exact MSE for normal errors, linear cap on outlier return spikes

STAGES = {
    0: {"name": "GENESIS",     "actor_lr": 2.5e-4, "critic_lr": 6.0e-4, "shaping": 1.00, "ent_coef": 0.0040, "ent_floor": 0.12},
    1: {"name": "RISING",      "actor_lr": 1.8e-4, "critic_lr": 4.5e-4, "shaping": 0.45, "ent_coef": 0.0020, "ent_floor": 0.08},
    2: {"name": "COMPETITIVE", "actor_lr": 1.0e-4, "critic_lr": 3.0e-4, "shaping": 0.12, "ent_coef": 0.0008, "ent_floor": 0.04},
    3: {"name": "APEX",        "actor_lr": 5.0e-5, "critic_lr": 2.0e-4, "shaping": 0.00, "ent_coef": 0.0003, "ent_floor": 0.025},
}


def disable_windows_quick_edit():
    """Prevents accidental clicks in Windows PowerShell/CMD from pausing stdout & freezing training."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        h = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(h, new_mode)
    except Exception:
        pass


def build_and_load_dll() -> ctypes.CDLL:
    print("[build] compiling rust/rl_env (--release)...", flush=True)
    subprocess.run(["cargo", "build", "--release"], cwd=RL_ENV_DIR, check=True)
    lib = ctypes.CDLL(str(DLL_PATH))

    lib.rl_pool_new.argtypes = [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32]
    lib.rl_pool_new.restype = ctypes.c_void_p

    lib.rl_pool_num_agents.argtypes = [ctypes.c_void_p]
    lib.rl_pool_num_agents.restype = ctypes.c_size_t

    lib.rl_pool_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.rl_pool_reset.restype = None

    lib.rl_pool_add_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.rl_pool_add_snapshot.restype = ctypes.c_int

    lib.rl_pool_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.rl_pool_step.restype = None

    lib.rl_pool_free.argtypes = [ctypes.c_void_p]
    lib.rl_pool_free.restype = None

    lib.rl_eval_policy.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.rl_eval_policy.restype = ctypes.c_int
    return lib


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def make_sliding_episode_mask(ep_ids: torch.Tensor, ctx_len: int) -> torch.Tensor:
    """Boolean SDPA mask [B, 1, S, S]: True iff causal, within `ctx_len`, and same episode."""
    s = ep_ids.shape[1]
    idx = torch.arange(s, device=ep_ids.device)
    delta = idx[:, None] - idx[None, :]
    win = (delta >= 0) & (delta < ctx_len)
    same_ep = ep_ids[:, :, None] == ep_ids[:, None, :]
    return (win[None, :, :] & same_ep)[:, None, :, :]


class KVCache:
    """Batched sliding-window KV cache for O(1)-projection causal RoPE inference."""

    def __init__(
        self,
        batch_size: int,
        n_layers: int,
        ctx_len: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.k = torch.zeros((batch_size, n_layers, ctx_len, n_heads, head_dim), dtype=torch.float32, device=device)
        self.v = torch.zeros((batch_size, n_layers, ctx_len, n_heads, head_dim), dtype=torch.float32, device=device)
        self.pos = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        self.length = torch.zeros((batch_size,), dtype=torch.int64, device=device)
        self.batch_idx = torch.arange(batch_size, device=device)
        self.slot_arange = torch.arange(ctx_len, device=device)
        self.ctx_len = ctx_len

    def reset_mask(self, done_mask: torch.Tensor):
        self.pos.masked_fill_(done_mask, 0)
        self.length.masked_fill_(done_mask, 0)

    def clone(self) -> KVCache:
        c = KVCache.__new__(KVCache)
        c.k = self.k.clone()
        c.v = self.v.clone()
        c.pos = self.pos.clone()
        c.length = self.length.clone()
        c.batch_idx = self.batch_idx
        c.slot_arange = self.slot_arange
        c.ctx_len = self.ctx_len
        return c


class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, n_layers: int):
        super().__init__()
        assert d_model % n_heads == 0 and (d_model // n_heads) % 2 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.norm2 = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, ffn_dim)
        self.w_up = nn.Linear(d_model, ffn_dim)
        self.w_down = nn.Linear(ffn_dim, d_model)

        # Residual-branch scaling for stable deep RL optimization from step 0
        res_scale = 1.0 / math.sqrt(2.0 * n_layers)
        nn.init.orthogonal_(self.wo.weight, gain=res_scale)
        nn.init.zeros_(self.wo.bias)
        nn.init.orthogonal_(self.w_down.weight, gain=res_scale)
        nn.init.zeros_(self.w_down.bias)

    def forward_seq(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        return_kv: bool = False,
    ):
        b, s, d = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(b, s, d)
        x = x + self.wo(attn)

        h2 = self.norm2(x)
        ffn = self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
        out = x + ffn
        if return_kv:
            return out, k.transpose(1, 2), v.transpose(1, 2)
        return out

    def forward_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache,
        layer_idx: int,
        slot: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        b, d = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(b, self.n_heads, self.head_dim)
        k = self.wk(h).view(b, self.n_heads, self.head_dim)
        v = self.wv(h).view(b, self.n_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        cache.k[cache.batch_idx, layer_idx, slot] = k
        cache.v[cache.batch_idx, layer_idx, slot] = v

        k_all = cache.k[:, layer_idx]  # [B, CTX, H, D_h]
        v_all = cache.v[:, layer_idx]
        scores = torch.einsum("bhd,bshd->bhs", q, k_all) * self.scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn = torch.einsum("bhs,bshd->bhd", weights, v_all).reshape(b, d)
        x = x + self.wo(attn)

        h2 = self.norm2(x)
        ffn = self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
        return x + ffn


class RoPEBackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        ctx_len: int,
        rope_theta: float,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ffn_dim = ffn_dim
        self.ctx_len = ctx_len
        self.rope_theta = rope_theta

        half = self.head_dim // 2
        inv_freq = torch.tensor(
            [rope_theta ** (-(2.0 * k) / self.head_dim) for k in range(half)],
            dtype=torch.float64,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.in_proj = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList(
            [RoPETransformerBlock(d_model, n_heads, ffn_dim, n_layers) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

    def make_cache(self, batch_size: int, device: torch.device) -> KVCache:
        return KVCache(batch_size, self.n_layers, self.ctx_len, self.n_heads, self.head_dim, device)

    def forward_seq(self, obs_seq: torch.Tensor, ep_ids: torch.Tensor) -> torch.Tensor:
        """obs_seq: [B, S, OBS_DIM], ep_ids: [B, S] -> hidden [B, S, D_MODEL]."""
        s = obs_seq.shape[1]
        x = self.in_proj(obs_seq)
        positions = torch.arange(s, device=obs_seq.device, dtype=torch.float64)
        angles = positions[:, None] * self.inv_freq[None, :]
        cos = torch.cos(angles).to(x.dtype)[None, None, :, :]
        sin = torch.sin(angles).to(x.dtype)[None, None, :, :]
        attn_mask = make_sliding_episode_mask(ep_ids, self.ctx_len)
        for blk in self.blocks:
            x = blk.forward_seq(x, cos, sin, attn_mask)
        return self.final_norm(x)

    def forward_step(self, obs: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """obs: [B, OBS_DIM] -> hidden [B, D_MODEL], updating `cache` in place."""
        x = self.in_proj(obs)
        angles = cache.pos.to(torch.float64)[:, None] * self.inv_freq[None, :]
        cos = torch.cos(angles).to(x.dtype)[:, None, :]
        sin = torch.sin(angles).to(x.dtype)[:, None, :]
        slot = cache.pos % self.ctx_len
        new_len = (cache.pos + 1).clamp(max=self.ctx_len)
        valid = cache.slot_arange[None, :] < new_len[:, None]
        for li, blk in enumerate(self.blocks):
            x = blk.forward_step(x, cos, sin, cache, li, slot, valid)
        cache.pos = cache.pos + 1
        cache.length = new_len
        return self.final_norm(x)


    @torch.no_grad()
    def reprime_cache(
        self,
        hist_obs: torch.Tensor,
        hist_ep: torch.Tensor,
        cur_ep: torch.Tensor,
        cache: KVCache,
    ):
        """Recomputes K/V cache from `hist_obs` with current weights so rollout t=0 has exact zero skew."""
        s_len = hist_obs.shape[0]
        device = hist_obs.device
        # Trailing frames in `hist_ep` belonging to `cur_ep` (strictly contiguous at the tail)
        active = hist_ep == cur_ep[None, :]  # [S, A]
        ep_len = active.sum(dim=0)           # [A], in 0..S
        m = ep_len.clamp(max=self.ctx_len - 1)  # [A], number of past frames needed at next step

        obs_b = hist_obs.transpose(0, 1).contiguous()  # [A, S, OBS_DIM]
        ep_b = hist_ep.transpose(0, 1).contiguous()    # [A, S]

        # Shift per-agent RoPE positions so the last `m[a]` frames get exact positions 0..m[a]-1
        s_idx = torch.arange(s_len, device=device)[None, :]  # [1, S]
        positions = (s_idx - (s_len - m)[:, None]).to(torch.float64)  # [A, S]
        angles = positions[:, :, None] * self.inv_freq[None, None, :]
        x = self.in_proj(obs_b)
        cos = torch.cos(angles).to(x.dtype)[:, None, :, :]
        sin = torch.sin(angles).to(x.dtype)[:, None, :, :]
        attn_mask = make_sliding_episode_mask(ep_b, self.ctx_len)

        keep_len = self.ctx_len - 1
        j = torch.arange(keep_len, device=device)[None, :]      # [1, CTX-1]
        src_idx = ((s_len - m[:, None]) + j).clamp(0, s_len - 1)  # [A, CTX-1]
        batch_gather = cache.batch_idx[:, None]                 # [A, 1]

        for li, blk in enumerate(self.blocks):
            x, k_seq, v_seq = blk.forward_seq(x, cos, sin, attn_mask, return_kv=True)
            cache.k[:, li, :keep_len] = k_seq[batch_gather, src_idx]
            cache.v[:, li, :keep_len] = v_seq[batch_gather, src_idx]

        cache.pos.copy_(m)
        cache.length.copy_(m)


class ActorNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        d_model: int = ACTOR_D_MODEL,
        n_layers: int = ACTOR_LAYERS,
        n_heads: int = ACTOR_HEADS,
        ffn_dim: int = ACTOR_FFN_DIM,
        ctx_len: int = CTX_LEN,
        rope_theta: float = ROPE_THETA,
    ):
        super().__init__()
        self.backbone = RoPEBackbone(obs_dim, d_model, n_layers, n_heads, ffn_dim, ctx_len, rope_theta)
        self.head = nn.Linear(d_model, TOTAL_LOGITS)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def make_cache(self, batch_size: int, device: torch.device) -> KVCache:
        return self.backbone.make_cache(batch_size, device)

    @torch.no_grad()
    def reprime_cache(self, hist_obs: torch.Tensor, hist_ep: torch.Tensor, cur_ep: torch.Tensor, cache: KVCache):
        self.backbone.reprime_cache(hist_obs, hist_ep, cur_ep, cache)

    def forward_seq(self, obs_seq: torch.Tensor, ep_ids: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
        h = self.backbone.forward_seq(obs_seq, ep_ids)
        if prefix_len > 0:
            h = h[:, prefix_len:, :]
        return self.head(h)

    def forward_step(self, obs: torch.Tensor, cache: KVCache) -> torch.Tensor:
        return self.head(self.backbone.forward_step(obs, cache))

    def split_logits(self, logits: torch.Tensor):
        return torch.split(logits, ACT_BRANCHES, dim=-1)

    @torch.no_grad()
    def sample_action_step(self, obs: torch.Tensor, cache: KVCache):
        logits = torch.nan_to_num(self.forward_step(obs, cache), nan=0.0, posinf=20.0, neginf=-20.0)
        parts = self.split_logits(logits)
        acts = []
        logp_total = torch.zeros(obs.shape[0], device=obs.device)
        for p in parts:
            dist = torch.distributions.Categorical(logits=p)
            a = dist.sample()
            acts.append(a)
            logp_total = logp_total + dist.log_prob(a)
        return torch.stack(acts, dim=-1), logp_total

    def eval_actions_seq(
        self,
        obs_seq: torch.Tensor,
        ep_ids: torch.Tensor,
        actions: torch.Tensor,
        prefix_len: int,
    ):
        logits = self.forward_seq(obs_seq, ep_ids, prefix_len=prefix_len)  # [B, C, TOTAL_LOGITS]
        parts = self.split_logits(logits)
        logp = torch.zeros(actions.shape[:2], device=obs_seq.device)
        ent = torch.zeros(actions.shape[:2], device=obs_seq.device)
        head_ents = []
        for i, p in enumerate(parts):
            dist = torch.distributions.Categorical(logits=p)
            logp = logp + dist.log_prob(actions[..., i])
            h_ent = dist.entropy()
            ent = ent + h_ent
            head_ents.append(h_ent.mean())
        return logp, ent, torch.stack(head_ents), logits


class CriticNet(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        d_model: int = CRITIC_D_MODEL,
        n_layers: int = CRITIC_LAYERS,
        n_heads: int = CRITIC_HEADS,
        ffn_dim: int = CRITIC_FFN_DIM,
        ctx_len: int = CTX_LEN,
        rope_theta: float = ROPE_THETA,
    ):
        super().__init__()
        self.backbone = RoPEBackbone(obs_dim, d_model, n_layers, n_heads, ffn_dim, ctx_len, rope_theta)
        self.head = nn.Linear(d_model, 1)

    def make_cache(self, batch_size: int, device: torch.device) -> KVCache:
        return self.backbone.make_cache(batch_size, device)

    @torch.no_grad()
    def reprime_cache(self, hist_obs: torch.Tensor, hist_ep: torch.Tensor, cur_ep: torch.Tensor, cache: KVCache):
        self.backbone.reprime_cache(hist_obs, hist_ep, cur_ep, cache)

    def forward_step(self, obs: torch.Tensor, cache: KVCache) -> torch.Tensor:
        return self.head(self.backbone.forward_step(obs, cache)).squeeze(-1)

    def forward_seq(self, obs_seq: torch.Tensor, ep_ids: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
        h = self.backbone.forward_seq(obs_seq, ep_ids)
        if prefix_len > 0:
            h = h[:, prefix_len:, :]
        return self.head(h).squeeze(-1)


def make_adamw(model: nn.Module, lr: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        eps=1e-5,
    )


def models_finite(*models: nn.Module) -> bool:
    for m in models:
        for p in m.parameters():
            if not torch.isfinite(p).all():
                return False
    return True


def clone_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def export_hsp2(actor: ActorNet) -> bytes:
    """Exports ActorNet (RoPE Transformer) to compact per-row int8 quantized HSP2 binary format."""
    bb = actor.backbone
    buf = bytearray()
    buf += POLICY_MAGIC
    buf += struct.pack(
        "<HHHHHHHHf",
        bb.obs_dim,
        bb.d_model,
        bb.n_layers,
        bb.n_heads,
        bb.ffn_dim,
        bb.ctx_len,
        TOTAL_LOGITS,
        0,
        float(bb.rope_theta),
    )

    def write_linear(layer: nn.Linear):
        w = layer.weight.detach().cpu().float().numpy()
        b = layer.bias.detach().cpu().float().numpy()
        max_abs = np.max(np.abs(w), axis=1)
        scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int8)
        buf.extend(scales.tobytes())
        buf.extend(b.astype(np.float32).tobytes())
        buf.extend(q.tobytes())

    def write_rms(norm: RMSNorm):
        buf.extend(norm.weight.detach().cpu().float().numpy().astype(np.float32).tobytes())

    write_linear(bb.in_proj)
    for blk in bb.blocks:
        write_rms(blk.norm1)
        write_linear(blk.wq)
        write_linear(blk.wk)
        write_linear(blk.wv)
        write_linear(blk.wo)
        write_rms(blk.norm2)
        write_linear(blk.w_gate)
        write_linear(blk.w_up)
        write_linear(blk.w_down)
    write_rms(bb.final_norm)
    write_linear(actor.head)
    return bytes(buf)


def eval_against(
    lib: ctypes.CDLL,
    model_bytes: bytes,
    opp_diff: int,
    team_size: int = 1,
    matches: int = 6,
    secs: float = 180.0,
    arena: int = 0,
    mayhem: int = 0,
):
    arr = np.frombuffer(model_bytes, dtype=np.uint8)
    metrics = np.zeros(7, dtype=np.int32)
    ok = lib.rl_eval_policy(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(model_bytes),
        opp_diff,
        team_size,
        matches,
        secs,
        arena,
        mayhem,
        metrics.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert ok == 1, "rl_eval_policy failed to parse HSP2 bytes"
    return {
        "goals_a": int(metrics[0]),
        "goals_b": int(metrics[1]),
        "wins_a": int(metrics[2]),
        "wins_b": int(metrics[3]),
        "shots_a": int(metrics[4]),
        "shots_b": int(metrics[5]),
        "owngoals_a": int(metrics[6]),
    }


def evaluate_full(lib: ctypes.CDLL, model_bytes: bytes):
    specs = {
        "chp1": dict(opp_diff=3, team_size=1, matches=4, secs=180.0, arena=0, mayhem=0),
        "leg1": dict(opp_diff=4, team_size=1, matches=6, secs=180.0, arena=0, mayhem=0),
        "mpc1": dict(opp_diff=5, team_size=1, matches=6, secs=180.0, arena=0, mayhem=0),
        "leg2": dict(opp_diff=4, team_size=2, matches=4, secs=180.0, arena=0, mayhem=0),
        "mpc2": dict(opp_diff=5, team_size=2, matches=4, secs=180.0, arena=0, mayhem=0),
        "exp1": dict(opp_diff=5, team_size=1, matches=6, secs=180.0, arena=EVAL_ARENA_EXPERIMENTAL_CYCLE, mayhem=0),
        "may1": dict(opp_diff=5, team_size=1, matches=4, secs=180.0, arena=0, mayhem=1),
        "may2": dict(opp_diff=4, team_size=2, matches=4, secs=180.0, arena=0, mayhem=1),
    }
    # ctypes.CDLL releases the GIL, so running all 8 suites concurrently saturates CPU cores (~3-4x faster)
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futs = {k: ex.submit(eval_against, lib, model_bytes, **kw) for k, kw in specs.items()}
        res = {k: f.result() for k, f in futs.items()}

    e_chp1, e_leg1, e_mpc1 = res["chp1"], res["leg1"], res["mpc1"]
    e_leg2, e_mpc2, e_exp1 = res["leg2"], res["mpc2"], res["exp1"]
    e_may1, e_may2 = res["may1"], res["may2"]

    gd_chp1 = e_chp1["goals_a"] - e_chp1["goals_b"]
    gd_leg1 = e_leg1["goals_a"] - e_leg1["goals_b"]
    gd_mpc1 = e_mpc1["goals_a"] - e_mpc1["goals_b"]
    gd_leg2 = e_leg2["goals_a"] - e_leg2["goals_b"]
    gd_mpc2 = e_mpc2["goals_a"] - e_mpc2["goals_b"]
    gd_exp1 = e_exp1["goals_a"] - e_exp1["goals_b"]
    gd_may1 = e_may1["goals_a"] - e_may1["goals_b"]
    gd_may2 = e_may2["goals_a"] - e_may2["goals_b"]
    score = (
        gd_mpc1 * 2.5
        + gd_mpc2 * 2.5
        + gd_exp1 * 1.5
        + gd_may1 * 1.5
        + gd_may2 * 1.0
        + gd_leg1 * 1.0
        + gd_leg2 * 1.0
        + gd_chp1 * 0.5
        - (e_mpc1["owngoals_a"] + e_mpc2["owngoals_a"] + e_exp1["owngoals_a"] + e_may1["owngoals_a"]) * 3.0
    )
    res["score"] = float(score)
    return res


def detect_stage(ev: dict, current_stage: int) -> int:
    """Monotonically determines the right curriculum stage from live eval results."""
    stage = current_stage
    if stage < 1 and (ev["chp1"]["goals_a"] >= ev["chp1"]["goals_b"] or (ev["chp1"]["goals_a"] >= 8 and ev["score"] > -430.0)):
        stage = 1
    if stage < 2 and ev["leg1"]["goals_a"] > ev["leg1"]["goals_b"] and ev["leg1"]["wins_a"] >= ev["leg1"]["wins_b"]:
        stage = 2
    if stage < 3 and ev["mpc1"]["goals_a"] > ev["mpc1"]["goals_b"] and ev["leg2"]["goals_a"] >= ev["leg2"]["goals_b"]:
        stage = 3
    return stage


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_write_bytes(data: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def cleanup_old_checkpoints():
    """Deletes any stale/extra files in checkpoints/, keeping only latest.pt and best.pt."""
    if not CKPT_DIR.exists():
        return
    keep = {LATEST_PT.name, BEST_PT.name}
    for p in CKPT_DIR.iterdir():
        if p.is_file() and p.name not in keep:
            try:
                p.unlink()
            except OSError:
                pass


def add_snapshot(lib: ctypes.CDLL, pool, blob: bytes):
    arr = np.frombuffer(blob, dtype=np.uint8)
    lib.rl_pool_add_snapshot(pool, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(blob), MAX_SNAPSHOTS)


def make_pool(lib: ctypes.CDLL, stage: int, seed: int, snapshots: list[bytes]):
    pool = lib.rl_pool_new(NUM_ENVS, seed, stage)
    num_agents = lib.rl_pool_num_agents(pool)
    for blob in snapshots[-MAX_SNAPSHOTS:]:
        add_snapshot(lib, pool, blob)
    obs_np = np.zeros((num_agents, OBS_DIM), dtype=np.float32)
    act_np = np.zeros((num_agents, ACT_HEADS), dtype=np.uint8)
    rew_np = np.zeros((num_agents,), dtype=np.float32)
    done_np = np.zeros((num_agents,), dtype=np.uint8)
    stats_np = np.zeros((3,), dtype=np.uint32)
    lib.rl_pool_reset(pool, obs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return pool, num_agents, obs_np, act_np, rew_np, done_np, stats_np


def alloc_rollout_state(num_agents: int, actor: ActorNet, critic: CriticNet, device: torch.device):
    return {
        "obs_buf": torch.zeros((ROLLOUT_STEPS, num_agents, OBS_DIM), dtype=torch.float32, device=device),
        "act_buf": torch.zeros((ROLLOUT_STEPS, num_agents, ACT_HEADS), dtype=torch.int64, device=device),
        "logp_buf": torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device),
        "val_buf": torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device),
        "rew_buf": torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device),
        "done_buf": torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device),
        "ep_buf": torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.int64, device=device),
        "hist_obs": torch.zeros((PREFIX_LEN, num_agents, OBS_DIM), dtype=torch.float32, device=device),
        "hist_ep": torch.full((PREFIX_LEN, num_agents), -1, dtype=torch.int64, device=device),
        "cur_ep": torch.zeros((num_agents,), dtype=torch.int64, device=device),
        "actor_cache": actor.make_cache(num_agents, device),
        "critic_cache": critic.make_cache(num_agents, device),
    }


def format_eval(upd: int, stage: int, total_steps: int, ev: dict) -> str:
    return (
        f"[EVAL upd {upd:05d} | {total_steps/1e6:.2f}M | S{stage}:{STAGES[stage]['name']}] "
        f"1v1 CHAMP: {ev['chp1']['goals_a']}-{ev['chp1']['goals_b']} | "
        f"1v1 LEGEND: {ev['leg1']['goals_a']}-{ev['leg1']['goals_b']} (W {ev['leg1']['wins_a']}-{ev['leg1']['wins_b']}) | "
        f"1v1 MPC: {ev['mpc1']['goals_a']}-{ev['mpc1']['goals_b']} (W {ev['mpc1']['wins_a']}-{ev['mpc1']['wins_b']}) | "
        f"2v2 LEGEND: {ev['leg2']['goals_a']}-{ev['leg2']['goals_b']} | "
        f"2v2 MPC: {ev['mpc2']['goals_a']}-{ev['mpc2']['goals_b']} | "
        f"1v1 LABS: {ev['exp1']['goals_a']}-{ev['exp1']['goals_b']} | "
        f"1v1 MAYHEM: {ev['may1']['goals_a']}-{ev['may1']['goals_b']} | "
        f"2v2 MAYHEM: {ev['may2']['goals_a']}-{ev['may2']['goals_b']} | "
        f"score={ev['score']:.1f}"
    )


def is_compatible_ckpt(ck: dict) -> bool:
    if ck.get("arch") != ARCH_TAG:
        return False
    cfg = ck.get("config", {})
    return (
        cfg.get("obs_dim") == OBS_DIM
        and cfg.get("total_logits") == TOTAL_LOGITS
        and cfg.get("actor_d_model") == ACTOR_D_MODEL
        and cfg.get("actor_layers") == ACTOR_LAYERS
        and cfg.get("actor_heads") == ACTOR_HEADS
        and cfg.get("actor_ffn_dim") == ACTOR_FFN_DIM
        and cfg.get("critic_d_model") == CRITIC_D_MODEL
        and cfg.get("critic_layers") == CRITIC_LAYERS
        and cfg.get("critic_heads") == CRITIC_HEADS
        and cfg.get("critic_ffn_dim") == CRITIC_FFN_DIM
    )


def is_compatible_hsp2(blob: bytes) -> bool:
    if len(blob) < 24 or blob[:4] != POLICY_MAGIC:
        return False
    obs_d, _d, _l, _h, _ffn, _ctx, out_d, _flags, _theta = struct.unpack("<HHHHHHHHf", blob[4:24])
    return obs_d == OBS_DIM and out_d == TOTAL_LOGITS


def main():
    disable_windows_quick_edit()
    seed = int(time.time()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[init] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) | "
        f"RoPE Transformer (d={ACTOR_D_MODEL}, L={ACTOR_LAYERS}, H={ACTOR_HEADS}, ctx={CTX_LEN}, theta={ROPE_THETA}) | "
        f"Ctrl+C anytime to save & stop",
        flush=True,
    )

    lib = build_and_load_dll()
    cleanup_old_checkpoints()

    actor = ActorNet().to(device)
    critic = CriticNet().to(device)
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"[init] params: actor={actor_params:,} critic={critic_params:,}", flush=True)

    actor_opt = make_adamw(actor, STAGES[0]["actor_lr"])
    critic_opt = make_adamw(critic, STAGES[0]["critic_lr"])

    stage = 0
    upd = 0
    total_env_steps = 0
    best_score = -1e9
    actor_lr = STAGES[0]["actor_lr"]
    best_bytes: bytes | None = None
    snapshots: list[bytes] = []

    # Seed the self-play league with the currently deployed live model (no protection gate —
    # the best model is exported to OUT_MODEL on every new best below)
    if OUT_MODEL.exists() and OUT_MODEL.stat().st_size > 24:
        live_bytes = OUT_MODEL.read_bytes()
        if is_compatible_hsp2(live_bytes):
            best_bytes = live_bytes
            snapshots.append(live_bytes)
            print(f"[init] live {OUT_MODEL.name} added to the league snapshot pool", flush=True)
        else:
            print(
                f"[init] ignoring incompatible legacy {OUT_MODEL.name} (expected obs={OBS_DIM}, logits={TOTAL_LOGITS})",
                flush=True,
            )

    # Resume the best-score tracker so `best.pt` keeps meaning "best ever seen"
    if BEST_PT.exists():
        bck = torch.load(BEST_PT, map_location="cpu", weights_only=True)
        if is_compatible_ckpt(bck):
            best_score = float(bck.get("best_score", bck.get("score", -1e9)))
        else:
            print(f"[init] ignoring incompatible legacy {BEST_PT.name}", flush=True)

    # Auto-resume from latest.pt (or best.pt) if compatible
    resume_path = None
    for cand_path in (LATEST_PT, BEST_PT):
        if cand_path.exists():
            cand_ck = torch.load(cand_path, map_location="cpu", weights_only=True)
            if is_compatible_ckpt(cand_ck):
                resume_path = cand_path
                break
            print(f"[init] skipping incompatible legacy {cand_path.name}", flush=True)

    if resume_path is not None:
        ck = torch.load(resume_path, map_location=device, weights_only=True)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        if not models_finite(actor, critic):
            print(f"[init] WARNING: non-finite weights in {resume_path.name} — starting fresh!", flush=True)
            actor = ActorNet().to(device)
            critic = CriticNet().to(device)
            actor_opt = make_adamw(actor, STAGES[0]["actor_lr"])
            critic_opt = make_adamw(critic, STAGES[0]["critic_lr"])
            snapshots.append(export_hsp2(actor))
        else:
            if "actor_opt" in ck:
                try:
                    actor_opt.load_state_dict(ck["actor_opt"])
                    critic_opt.load_state_dict(ck["critic_opt"])
                except Exception:
                    pass
            upd = int(ck.get("upd", 0))
            total_env_steps = int(ck.get("total_env_steps", 0))
            best_score = max(best_score, float(ck.get("best_score", ck.get("score", -1e9))))
            saved_stage = int(ck.get("stage", 0))

            cur_bytes = export_hsp2(actor)
            snapshots.append(cur_bytes)
            print(f"[init] benchmarking resumed checkpoint {resume_path.name}...", flush=True)
            init_ev = evaluate_full(lib, cur_bytes)
            stage = detect_stage(init_ev, saved_stage)
            stage_lr = STAGES[stage]["actor_lr"]
            actor_lr = float(ck.get("actor_lr", stage_lr))
            actor_lr = max(stage_lr * 0.15, min(stage_lr * 1.5, actor_lr))
            print(
                f"[init] auto-resumed from {resume_path.name} "
                f"(upd={upd}, steps={total_env_steps/1e6:.2f}M, best={best_score:.1f}, lr={actor_lr:.2e})",
                flush=True,
            )
            print(format_eval(upd, stage, total_env_steps, init_ev), flush=True)
    else:
        snapshots.append(export_hsp2(actor))
        print("[init] no compatible checkpoints found — starting completely from scratch (Stage 0: GENESIS)", flush=True)

    # Also seed league with best.pt if it is compatible
    if BEST_PT.exists():
        bck = torch.load(BEST_PT, map_location="cpu", weights_only=True)
        if is_compatible_ckpt(bck):
            tmp_actor = ActorNet()
            tmp_actor.load_state_dict(bck["actor"])
            if models_finite(tmp_actor):
                best_bytes = export_hsp2(tmp_actor)
                snapshots.append(best_bytes)
                # Make sure the live game asset always holds the best model, from startup onward
                atomic_write_bytes(best_bytes, OUT_MODEL)
                print(
                    f"[init] exported best.pt (score={best_score:.1f}) -> {OUT_MODEL}",
                    flush=True,
                )

    def make_checkpoint_payload(cur_score: float) -> dict:
        return {
            "arch": ARCH_TAG,
            "config": {
                "obs_dim": OBS_DIM,
                "act_branches": list(ACT_BRANCHES),
                "total_logits": TOTAL_LOGITS,
                "actor_d_model": ACTOR_D_MODEL,
                "actor_layers": ACTOR_LAYERS,
                "actor_heads": ACTOR_HEADS,
                "actor_ffn_dim": ACTOR_FFN_DIM,
                "critic_d_model": CRITIC_D_MODEL,
                "critic_layers": CRITIC_LAYERS,
                "critic_heads": CRITIC_HEADS,
                "critic_ffn_dim": CRITIC_FFN_DIM,
                "ctx_len": CTX_LEN,
                "rope_theta": ROPE_THETA,
            },
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "upd": upd,
            "total_env_steps": total_env_steps,
            "stage": stage,
            "actor_lr": actor_lr,
            "score": cur_score,
            "best_score": best_score,
        }

    pool, num_agents, obs_np, act_np, rew_np, done_np, stats_np = make_pool(lib, stage, seed, snapshots)
    st = alloc_rollout_state(num_agents, actor, critic, device)
    healthy_backup = {"actor": clone_state_cpu(actor), "critic": clone_state_cpu(critic)}

    print(
        f"[PPO] running forever in Stage {stage} ({STAGES[stage]['name']}) | "
        f"{NUM_ENVS} envs, {num_agents} controlled agents, eval/save every {EVAL_EVERY} updates",
        flush=True,
    )

    t_start = time.time()
    last_eval_score = best_score
    n_chunks_per_agent = ROLLOUT_STEPS // CHUNK_LEN

    try:
        while True:
            upd += 1
            cfg = STAGES[stage]
            stage_lr = cfg["actor_lr"]
            if upd <= WARMUP_UPDATES:
                warmup_frac = 0.10 + 0.90 * (upd / WARMUP_UPDATES)
                eff_actor_lr = stage_lr * warmup_frac
                eff_critic_lr = cfg["critic_lr"] * warmup_frac
            else:
                eff_actor_lr = actor_lr
                eff_critic_lr = cfg["critic_lr"]

            for pg in actor_opt.param_groups:
                pg["lr"] = eff_actor_lr
            for pg in critic_opt.param_groups:
                pg["lr"] = eff_critic_lr
            shaping = cfg["shaping"]
            ent_coef = cfg["ent_coef"]
            ent_floor = cfg["ent_floor"]

            obs_buf = st["obs_buf"]
            act_buf = st["act_buf"]
            logp_buf = st["logp_buf"]
            val_buf = st["val_buf"]
            rew_buf = st["rew_buf"]
            done_buf = st["done_buf"]
            ep_buf = st["ep_buf"]
            actor_cache = st["actor_cache"]
            critic_cache = st["critic_cache"]
            cur_ep = st["cur_ep"]

            t_roll = time.time()
            ep_gs = ep_gc = ep_tc = 0
            actor.eval()
            critic.eval()
            for t in range(ROLLOUT_STEPS):
                obs_t = torch.nan_to_num(torch.from_numpy(obs_np).to(device), nan=0.0, posinf=5.0, neginf=-5.0)
                obs_buf[t] = obs_t
                ep_buf[t] = cur_ep
                with torch.no_grad():
                    a_t, lp_t = actor.sample_action_step(obs_t, actor_cache)
                    v_t = torch.nan_to_num(critic.forward_step(obs_t, critic_cache), nan=0.0, posinf=50.0, neginf=-50.0)
                act_buf[t] = a_t
                logp_buf[t] = lp_t
                val_buf[t] = v_t

                act_np[:] = a_t.cpu().numpy().astype(np.uint8)
                lib.rl_pool_step(
                    pool,
                    act_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    ctypes.c_float(shaping),
                    obs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    rew_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    done_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    stats_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                )
                rew_buf[t] = torch.nan_to_num(torch.from_numpy(rew_np).to(device), nan=0.0, posinf=50.0, neginf=-50.0)
                done_t = torch.from_numpy(done_np.astype(np.float32)).to(device)
                done_buf[t] = done_t
                done_mask = done_t > 0.5
                if done_mask.any():
                    actor_cache.reset_mask(done_mask)
                    critic_cache.reset_mask(done_mask)
                    cur_ep = cur_ep + done_mask.to(torch.int64)

                ep_gs += int(stats_np[0])
                ep_gc += int(stats_np[1])
                ep_tc += int(stats_np[2])

            st["cur_ep"] = cur_ep
            total_env_steps += ROLLOUT_STEPS * num_agents
            dt_roll = time.time() - t_roll

            # GAE (peek next_val on a cloned cache so rollout state stays clean)
            with torch.no_grad():
                next_obs = torch.nan_to_num(torch.from_numpy(obs_np).to(device), nan=0.0, posinf=5.0, neginf=-5.0)
                next_val = torch.nan_to_num(
                    critic.forward_step(next_obs, critic_cache.clone()), nan=0.0, posinf=50.0, neginf=-50.0
                )
                adv_buf = torch.zeros_like(rew_buf)
                last_gae = torch.zeros(num_agents, device=device)
                for t in reversed(range(ROLLOUT_STEPS)):
                    next_nonterm = 1.0 - done_buf[t]
                    next_v = next_val if t == ROLLOUT_STEPS - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + GAMMA * next_v * next_nonterm - val_buf[t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterm * last_gae
                    adv_buf[t] = last_gae
                ret_buf = adv_buf + val_buf

            # Build exact prefix-padded sliding-window sequence chunks for zero-skew PPO
            full_obs = torch.cat([st["hist_obs"], obs_buf], dim=0)  # [PREFIX + T, A, OBS_DIM]
            full_ep = torch.cat([st["hist_ep"], ep_buf], dim=0)     # [PREFIX + T, A]
            st["hist_obs"] = full_obs[-PREFIX_LEN:].clone()
            st["hist_ep"] = full_ep[-PREFIX_LEN:].clone()

            seq_len = PREFIX_LEN + CHUNK_LEN
            chunk_obs_list = []
            chunk_ep_list = []
            chunk_act_list = []
            chunk_logp_list = []
            chunk_adv_list = []
            chunk_ret_list = []
            for c_idx in range(n_chunks_per_agent):
                t0 = c_idx * CHUNK_LEN
                t1 = t0 + CHUNK_LEN
                # [S, A, ...] -> [A, S, ...]
                chunk_obs_list.append(full_obs[t0 : t0 + seq_len].transpose(0, 1))
                chunk_ep_list.append(full_ep[t0 : t0 + seq_len].transpose(0, 1))
                chunk_act_list.append(act_buf[t0:t1].transpose(0, 1))
                chunk_logp_list.append(logp_buf[t0:t1].transpose(0, 1))
                chunk_adv_list.append(adv_buf[t0:t1].transpose(0, 1))
                chunk_ret_list.append(ret_buf[t0:t1].transpose(0, 1))

            b_obs = torch.cat(chunk_obs_list, dim=0).contiguous()    # [N_chunks, PREFIX+C, OBS_DIM]
            b_ep = torch.cat(chunk_ep_list, dim=0).contiguous()      # [N_chunks, PREFIX+C]
            b_act = torch.cat(chunk_act_list, dim=0).contiguous()    # [N_chunks, C, ACT_HEADS]
            b_logp = torch.cat(chunk_logp_list, dim=0).contiguous()  # [N_chunks, C]
            b_adv = torch.cat(chunk_adv_list, dim=0).contiguous()    # [N_chunks, C]
            b_ret = torch.cat(chunk_ret_list, dim=0).contiguous()    # [N_chunks, C]
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            n_chunks = b_obs.shape[0]
            actor.train()
            critic.train()
            v_losses, ent_vals, min_head_ents, kls = [], [], [], []
            actor_kl_break = False
            actor_steps = 0

            for ep_i in range(PPO_EPOCHS):
                perm = torch.randperm(n_chunks, device=device)
                for i in range(0, n_chunks, MINIBATCH_CHUNKS):
                    mb = perm[i : i + MINIBATCH_CHUNKS]

                    # Critic ALWAYS trains all epochs (decoupled from actor KL early-stopping)
                    v_pred = critic.forward_seq(b_obs[mb], b_ep[mb], prefix_len=PREFIX_LEN)
                    v_loss = F.huber_loss(v_pred, b_ret[mb], delta=VALUE_HUBER_BETA)
                    if torch.isfinite(v_loss):
                        critic_opt.zero_grad(set_to_none=True)
                        v_loss.backward()
                        gn_c = nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                        if torch.isfinite(gn_c):
                            critic_opt.step()
                            v_losses.append(v_loss.item())

                    if actor_kl_break:
                        continue

                    new_logp, ent, head_ents, logits = actor.eval_actions_seq(
                        b_obs[mb], b_ep[mb], b_act[mb], prefix_len=PREFIX_LEN
                    )
                    log_ratio = (new_logp - b_logp[mb]).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    if not math.isfinite(approx_kl):
                        actor_kl_break = True
                        continue
                    kls.append(approx_kl)

                    is_first_mb = ep_i == 0 and i == 0
                    if approx_kl > (CATASTROPHIC_KL if is_first_mb else MAX_KL):
                        actor_kl_break = True
                        continue

                    pg1 = -b_adv[mb] * ratio
                    pg2 = -b_adv[mb] * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                    pg_loss = torch.max(pg1, pg2).mean()
                    ent_mean = ent.mean()
                    ent_floor_bonus = F.relu(ent_floor - head_ents).sum()
                    z_loss = logits.pow(2).mean()
                    loss = pg_loss - ent_coef * ent_mean - 0.05 * ent_floor_bonus + LOGIT_Z_LOSS * z_loss

                    if torch.isfinite(loss):
                        actor_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        gn_a = nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                        if torch.isfinite(gn_a):
                            actor_opt.step()
                            actor_steps += 1
                            ent_vals.append(ent_mean.item())
                            min_head_ents.append(head_ents.min().item())

            # Self-healing NaN/Inf guard: roll back to last healthy weights instead of bricking training
            if not models_finite(actor, critic):
                print(
                    f"[RECOVER] Non-finite weights detected at upd {upd} — restoring last healthy backup, "
                    f"resetting optimizers & halving actor_lr!",
                    flush=True,
                )
                actor.load_state_dict({k: v.to(device) for k, v in healthy_backup["actor"].items()})
                critic.load_state_dict({k: v.to(device) for k, v in healthy_backup["critic"].items()})
                actor_lr = max(stage_lr * 0.15, actor_lr * 0.5)
                actor_opt = make_adamw(actor, actor_lr)
                critic_opt = make_adamw(critic, cfg["critic_lr"])
            else:
                if upd % 10 == 0:
                    healthy_backup = {"actor": clone_state_cpu(actor), "critic": clone_state_cpu(critic)}

                # Adaptive KL learning-rate control (prevents both KL-break stalls and sluggish updates)
                if upd > WARMUP_UPDATES and kls:
                    mean_kl = float(np.mean(kls))
                    if actor_kl_break or mean_kl > 1.5 * TARGET_KL:
                        actor_lr = max(stage_lr * 0.15, actor_lr * 0.90)
                    elif mean_kl < 0.5 * TARGET_KL and actor_steps > 0:
                        actor_lr = min(stage_lr * 1.50, actor_lr * 1.06)

            # Re-prime KV caches with updated weights so step 0 of next rollout has exact ZERO skew!
            actor.eval()
            critic.eval()
            actor.reprime_cache(st["hist_obs"], st["hist_ep"], st["cur_ep"], st["actor_cache"])
            critic.reprime_cache(st["hist_obs"], st["hist_ep"], st["cur_ep"], st["critic_cache"])

            sps = (ROLLOUT_STEPS * num_agents) / max(1e-3, dt_roll)
            if upd == 1 or upd % 10 == 0:
                mean_r = rew_buf.mean().item() * 30.0
                print(
                    f"[PPO S{stage}] upd {upd:05d} steps={total_env_steps/1e6:.2f}M "
                    f"sps={sps:,.0f} rew/s={mean_r:+.3f} vsBotG={ep_gs}-{ep_gc} touches={ep_tc} "
                    f"vloss={np.mean(v_losses) if v_losses else 0:.3f} "
                    f"ent={np.mean(ent_vals) if ent_vals else 0:.2f} "
                    f"minH={np.mean(min_head_ents) if min_head_ents else 0:.2f} "
                    f"kl={np.mean(kls) if kls else 0:.4f} lr={eff_actor_lr:.1e} "
                    f"elapsed={(time.time()-t_start)/60:.1f}m",
                    flush=True,
                )

            if upd % EVAL_EVERY == 0:
                cand_bytes = export_hsp2(actor)
                snapshots.append(cand_bytes)
                if len(snapshots) > MAX_SNAPSHOTS:
                    snapshots.pop(0)
                if best_bytes is not None and best_bytes not in snapshots:
                    snapshots[0] = best_bytes
                add_snapshot(lib, pool, cand_bytes)

                print(f"[eval] benchmarking upd {upd:05d} across 8 suites...", flush=True)
                ev = evaluate_full(lib, cand_bytes)
                score = ev["score"]
                last_eval_score = score
                print(format_eval(upd, stage, total_env_steps, ev), flush=True)

                # No quality gate: every new best goes to best.pt AND the live game asset
                if score > best_score:
                    best_score = score
                    best_bytes = cand_bytes
                    atomic_torch_save(make_checkpoint_payload(score), BEST_PT)
                    atomic_write_bytes(cand_bytes, OUT_MODEL)
                    print(f"[SAVE] NEW BEST checkpoint -> {BEST_PT.name} (score={best_score:.1f})", flush=True)
                    print(f"[DEPLOY] NEW LIVE SAVANT MODEL -> {OUT_MODEL} ({len(cand_bytes):,} bytes, score={best_score:.1f})", flush=True)

                # Save latest.pt and clean up any old/stale files
                atomic_torch_save(make_checkpoint_payload(score), LATEST_PT)
                cleanup_old_checkpoints()

                # Auto-advance curriculum stage if ready
                next_stage = detect_stage(ev, stage)
                if next_stage > stage:
                    print(f"[STAGE] AUTO-PROMOTING Stage {stage} ({STAGES[stage]['name']}) -> Stage {next_stage} ({STAGES[next_stage]['name']})!", flush=True)
                    stage = next_stage
                    actor_lr = STAGES[stage]["actor_lr"] * 0.6
                    atomic_torch_save(make_checkpoint_payload(score), LATEST_PT)
                    lib.rl_pool_free(pool)
                    pool, num_agents, obs_np, act_np, rew_np, done_np, stats_np = make_pool(
                        lib, stage, (seed + upd * 997) & 0x7FFFFFFF, snapshots
                    )
                    st = alloc_rollout_state(num_agents, actor, critic, device)
                    print(f"[PPO] rebuilt pool for Stage {stage} ({STAGES[stage]['name']}): {NUM_ENVS} envs, {num_agents} controlled agents", flush=True)
    except KeyboardInterrupt:
        print(f"\n[STOP] Ctrl+C received at update {upd} ({total_env_steps/1e6:.2f}M steps) — saving {LATEST_PT.name}...", flush=True)
        atomic_torch_save(make_checkpoint_payload(last_eval_score), LATEST_PT)
        cleanup_old_checkpoints()
        print(f"[STOP] Saved cleanly. Re-run `python tools\\savant_rl\\train.py` anytime to auto-resume.", flush=True)
    finally:
        lib.rl_pool_free(pool)


if __name__ == "__main__":
    main()
