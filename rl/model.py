import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import (
    ACTION_N_BINS,
    BALL_DIM,
    NUM_PADS_OBS,
    OTHER_CARS,
    OTHER_DIM,
    PAD_DIM,
    SELF_DIM,
)


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPETransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block:
        x = x + W_o · MHA(RMSNorm(x))      (optional RoPE on q/k, optional causal mask)
        x = x + W_2 · GELU(W_1 · RMSNorm(x))
    No biases. Mirrored exactly by `TfBlock` in `rust/sim/src/neural.rs`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        is_causal: bool = False,
        use_rope: bool = True,
        max_len: int = 64,
        n_layers_total: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "d_model must be divisible by n_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.attn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = nn.RMSNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio, bias=False)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim, bias=False)

        for lin in (self.wq, self.wk, self.wv, self.fc1):
            nn.init.normal_(lin.weight, std=0.02)
        # GPT-2 style residual-projection scaling for stable deep residual streams
        res_std = 0.02 / math.sqrt(2.0 * max(1, n_layers_total))
        for lin in (self.wo, self.fc2):
            nn.init.normal_(lin.weight, std=res_std)

        if self.use_rope:
            cos, sin = build_rope_cache(max_len, self.head_dim, torch.device("cpu"), torch.float32)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim

        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(h).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(h).view(B, T, H, hd).transpose(1, 2)
        if self.use_rope:
            if T <= self._rope_cos.shape[2]:
                cos = self._rope_cos[:, :, :T, :].to(dtype=q.dtype)
                sin = self._rope_sin[:, :, :T, :].to(dtype=q.dtype)
            else:
                cos, sin = build_rope_cache(T, hd, x.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal and attn_mask is None,
        )
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.mlp_norm(x)), approximate="tanh"))
        return x


class RoPEHyperstrikePolicy(nn.Module):
    """
    RoPE Transformer Policy for Hyperstrike (1v1 through 5v5).

    1. Entity encoder: `entity_layers` bidirectional pre-norm transformer blocks (no positional
       encoding -> permutation-invariant) over [SELF], [BALL], up to 9 other cars (masked for
       1v1..5v5), and 6 nearest boost pads. The refined [SELF] token is the frame embedding.
    2. Temporal encoder: `temporal_layers` causal pre-norm RoPE transformer blocks over the
       sliding history [e_{t-T+1}, ..., e_t]; the last position feeds the heads.
    3. Raw Control Heads: 9 factored (incl. Mayhem item button) categorical heads + PPO value head.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        entity_layers: int = 2,
        temporal_layers: int = 2,
        seq_len: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.entity_layers = entity_layers
        self.temporal_layers = temporal_layers
        self.seq_len = seq_len

        self.self_proj = nn.Linear(SELF_DIM, d_model)
        self.ball_proj = nn.Linear(BALL_DIM, d_model)
        self.other_proj = nn.Linear(OTHER_DIM, d_model)
        self.pad_proj = nn.Linear(PAD_DIM, d_model)
        self.type_emb = nn.Embedding(4, d_model)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        self.entity_blocks = nn.ModuleList(
            [
                RoPETransformerBlock(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=False,
                    use_rope=False,
                    n_layers_total=entity_layers,
                )
                for _ in range(entity_layers)
            ]
        )
        self.entity_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.temporal_blocks = nn.ModuleList(
            [
                RoPETransformerBlock(
                    dim=d_model,
                    num_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    use_rope=True,
                    max_len=max(64, seq_len),
                    n_layers_total=temporal_layers,
                )
                for _ in range(temporal_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.action_head = nn.Linear(d_model, sum(ACTION_N_BINS))
        self.value_head = nn.Linear(d_model, 1)

        self.n_act_heads = len(ACTION_N_BINS)
        self.max_bins = max(ACTION_N_BINS)
        # Precompute padded [n_heads, max_bins] scatter mask for vectorized single-op sampling & log_prob
        max_bins = max(ACTION_N_BINS)
        valid_mask = torch.zeros((len(ACTION_N_BINS), max_bins), dtype=torch.bool)
        for k, nb in enumerate(ACTION_N_BINS):
            valid_mask[k, :nb] = True
        self.register_buffer(
            "_act_flat_idx",
            valid_mask.view(-1).nonzero(as_tuple=False).squeeze(-1),
            persistent=False,
        )

        self._init_heads()

    def _init_heads(self) -> None:
        for proj in (self.self_proj, self.ball_proj, self.other_proj, self.pad_proj):
            nn.init.orthogonal_(proj.weight, gain=math.sqrt(2))
            nn.init.zeros_(proj.bias)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def encode_frame(
        self,
        self_obs: torch.Tensor,
        ball_obs: torch.Tensor,
        other_obs: torch.Tensor,
        other_mask: torch.Tensor,
        pads_obs: torch.Tensor,
    ) -> torch.Tensor:
        B = self_obs.shape[0]
        device = self_obs.device

        t_self = self.self_proj(self_obs).unsqueeze(1) + self.type_emb.weight[0]
        t_ball = self.ball_proj(ball_obs).unsqueeze(1) + self.type_emb.weight[1]
        t_other = self.other_proj(other_obs) + self.type_emb.weight[2]
        t_pads = self.pad_proj(pads_obs) + self.type_emb.weight[3]

        tokens = torch.cat([t_self, t_ball, t_other, t_pads], dim=1)  # [B, 17, D]

        ones_2 = torch.ones((B, 2), device=device, dtype=torch.bool)
        ones_p = torch.ones((B, NUM_PADS_OBS), device=device, dtype=torch.bool)
        valid = torch.cat([ones_2, other_mask > 0.5, ones_p], dim=1)  # [B, 17]
        attn_mask = valid[:, None, None, :]

        for blk in self.entity_blocks:
            tokens = blk(tokens, attn_mask=attn_mask)
        return self.entity_norm(tokens[:, 0])

    def forward_temporal(self, frame_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = frame_seq
        for blk in self.temporal_blocks:
            x = blk(x)
        h = self.final_norm(x[:, -1])
        raw_logits = self.action_head(h)  # [B, sum(bins)]
        # Scatter into padded [B, heads, max_bins] with -1e9 for unused bins so sampling/eval is 1 vectorized op
        B = h.shape[0]
        H, MB = self.n_act_heads, self.max_bins
        padded = torch.full((B, H * MB), -1e9, device=h.device, dtype=raw_logits.dtype)
        padded[:, self._act_flat_idx] = raw_logits
        logits = padded.view(B, H, MB)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def forward_sequence(
        self,
        self_seq: torch.Tensor,
        ball_seq: torch.Tensor,
        other_seq: torch.Tensor,
        other_mask_seq: torch.Tensor,
        pads_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self_seq.shape[:2]
        e = self.encode_frame(
            self_seq.reshape(B * T, SELF_DIM),
            ball_seq.reshape(B * T, BALL_DIM),
            other_seq.reshape(B * T, OTHER_CARS, OTHER_DIM),
            other_mask_seq.reshape(B * T, OTHER_CARS),
            pads_seq.reshape(B * T, NUM_PADS_OBS, PAD_DIM),
        ).view(B, T, self.d_model)
        return self.forward_temporal(e)

    @staticmethod
    def sample_actions(
        logits: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # logits: [B, heads, max_bins] (padded with -1e9 for invalid bins)
        lg = logits.float()
        lp_all = F.log_softmax(lg, dim=-1)
        probs = lp_all.exp()
        if deterministic:
            acts = lg.argmax(dim=-1)
        else:
            # Vectorized Gumbel-Max sampling across all 8 heads in ONE kernel (no multinomial!)
            u = torch.rand_like(lg).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            acts = (lg + gumbel).argmax(dim=-1)
        logp = lp_all.gather(-1, acts.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        ent = -(probs * lp_all).sum(dim=-1).sum(dim=-1)
        return acts, logp, ent

    @staticmethod
    def eval_actions(
        logits: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lg = logits.float()
        lp_all = F.log_softmax(lg, dim=-1)
        probs = lp_all.exp()
        logp = lp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        ent = -(probs * lp_all).sum(dim=-1).sum(dim=-1)
        return logp, ent

    def export_weights(self, out_dir: Path, stem: str = "neural_policy") -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        bin_path = out_dir / f"{stem}.bin"
        meta_path = out_dir / f"{stem}.json"

        manifest = {
            "format": "hyperstrike_rope_transformer_v1",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
            "entity_layers": self.entity_layers,
            "temporal_layers": self.temporal_layers,
            "seq_len": self.seq_len,
            "rope_base": 10000.0,
            "action_n_bins": ACTION_N_BINS,
            "tensors": [],
        }

        # Export fused action_head sliced back into action_heads.0..7 so Rust loader stays 100% compatible!
        export_items = []
        for name, param in self.state_dict().items():
            if name == "action_head.weight":
                off = 0
                for k, nb in enumerate(ACTION_N_BINS):
                    export_items.append((f"action_heads.{k}.weight", param[off : off + nb]))
                    off += nb
            elif name == "action_head.bias":
                off = 0
                for k, nb in enumerate(ACTION_N_BINS):
                    export_items.append((f"action_heads.{k}.bias", param[off : off + nb]))
                    off += nb
            else:
                export_items.append((name, param))

        offset = 0
        with open(bin_path, "wb") as f:
            for name, param in export_items:
                arr = np.ascontiguousarray(param.detach().cpu().float().numpy())
                raw = arr.tobytes()
                f.write(raw)
                manifest["tensors"].append(
                    {
                        "name": name,
                        "shape": list(arr.shape),
                        "dtype": "f32",
                        "byte_offset": offset,
                        "byte_len": len(raw),
                    }
                )
                offset += len(raw)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return bin_path, meta_path
