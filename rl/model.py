import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class RotaryEmbedding(nn.Module):
    """Standard RoPE (Rotary Position Embeddings) for the temporal Transformer."""

    def __init__(self, head_dim: int, max_seq_len: int = 128, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, head_dim / 2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        T = x.shape[2]
        cos = self.cos_cached[:T].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin_cached[:T].to(dtype=x.dtype)[None, None, :, :]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack((out1, out2), dim=-1).flatten(-2)


class EntityBlock(nn.Module):
    """Permutation-invariant Transformer block over game entities (self, ball, cars, pads)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalRoPEBlock(nn.Module):
    """Causal Temporal Transformer block with Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.67):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = rope.apply(q)
        k = rope.apply(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class RoPEHyperstrikePolicy(nn.Module):
    """
    Entity-Token + Temporal RoPE Transformer Policy for Hyperstrike (1v1 through 5v5).

    1. Encodes each frame's entities (self, ball, up to 9 other cars masked for 1v1..5v5,
       and 6 nearest boost pads) via a 2-layer Entity Transformer into a frame embedding e_t.
    2. Runs a causal RoPE Transformer over the sliding history [e_{t-T+1}, ..., e_t] so the
       agent reads trajectories, ball spin/bounces, opponent fakes, and rotation timing.
    3. Outputs 8 factored raw-control categorical heads (throttle, steer, pitch, yaw, roll,
       jump, boost, handbrake) plus a scalar state-value head for PPO.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_entity_layers: int = 1,
        n_temporal_layers: int = 2,
        seq_len: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_entity_layers = n_entity_layers
        self.n_temporal_layers = n_temporal_layers
        self.seq_len = seq_len

        self.self_proj = nn.Linear(SELF_DIM, d_model)
        self.ball_proj = nn.Linear(BALL_DIM, d_model)
        self.other_proj = nn.Linear(OTHER_DIM, d_model)
        self.pad_proj = nn.Linear(PAD_DIM, d_model)

        # Learned token type embeddings: 0=self, 1=ball, 2=other_car, 3=pad
        self.type_emb = nn.Embedding(4, d_model)

        self.entity_blocks = nn.ModuleList(
            [EntityBlock(d_model, n_heads) for _ in range(n_entity_layers)]
        )
        self.entity_norm = RMSNorm(d_model)

        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max(64, seq_len * 2))
        self.temporal_blocks = nn.ModuleList(
            [TemporalRoPEBlock(d_model, n_heads) for _ in range(n_temporal_layers)]
        )
        self.final_norm = RMSNorm(d_model)

        # Factored raw control heads: [throttle(3), steer(5), pitch(3), yaw(3), roll(3), jump(2), boost(2), handbrake(2)]
        self.action_heads = nn.ModuleList([nn.Linear(d_model, n) for n in ACTION_N_BINS])
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in self.action_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
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
        """
        Encode a single frame per agent into a [B, d_model] embedding.
        Inputs:
          self_obs:   [B, 24]
          ball_obs:   [B, 16]
          other_obs:  [B, 9, 18]
          other_mask: [B, 9] (1.0 = active car, 0.0 = empty/demolished)
          pads_obs:   [B, 6, 4]
        """
        B = self_obs.shape[0]
        device = self_obs.device

        t_self = self.self_proj(self_obs).unsqueeze(1) + self.type_emb.weight[0]
        t_ball = self.ball_proj(ball_obs).unsqueeze(1) + self.type_emb.weight[1]
        t_other = self.other_proj(other_obs) + self.type_emb.weight[2]
        t_pads = self.pad_proj(pads_obs) + self.type_emb.weight[3]

        tokens = torch.cat([t_self, t_ball, t_other, t_pads], dim=1)  # [B, 17, D]

        # Build key mask: self(1), ball(1), other(9), pads(6)
        ones_2 = torch.ones((B, 2), device=device, dtype=torch.bool)
        ones_p = torch.ones((B, NUM_PADS_OBS), device=device, dtype=torch.bool)
        valid = torch.cat([ones_2, other_mask > 0.5, ones_p], dim=1)  # [B, 17]
        # SDPA boolean mask: True = attend, False = mask out -> shape [B, 1, 1, 17]
        attn_mask = valid[:, None, None, :]

        for blk in self.entity_blocks:
            tokens = blk(tokens, attn_mask=attn_mask)

        return self.entity_norm(tokens[:, 0])  # pooled [SELF] token: [B, D]

    def forward_temporal(self, frame_seq: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Run the causal RoPE Transformer over a sequence of frame embeddings.
        frame_seq: [B, T, d_model]
        Returns: (list of 8 logit tensors [B, n_bins_k], value [B]) for the LAST timestep.
        """
        x = frame_seq
        for blk in self.temporal_blocks:
            x = blk(x, self.rope)
        h = self.final_norm(x[:, -1])
        logits = [head(h) for head in self.action_heads]
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def forward_sequence(
        self,
        self_seq: torch.Tensor,
        ball_seq: torch.Tensor,
        other_seq: torch.Tensor,
        other_mask_seq: torch.Tensor,
        pads_seq: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Full end-to-end forward pass from raw history tensors [B, T, ...] -> (logits, value)."""
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
        logits: List[torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (actions [B, 8], total_log_prob [B], total_entropy [B])."""
        acts = []
        logp = 0.0
        ent = 0.0
        for lg in logits:
            dist = torch.distributions.Categorical(logits=lg.float())
            a = lg.argmax(dim=-1) if deterministic else dist.sample()
            acts.append(a)
            logp = logp + dist.log_prob(a)
            ent = ent + dist.entropy()
        return torch.stack(acts, dim=-1), logp, ent

    @staticmethod
    def eval_actions(
        logits: List[torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (total_log_prob [B], total_entropy [B]) for given [B, 8] discrete actions."""
        logp = 0.0
        ent = 0.0
        for k, lg in enumerate(logits):
            dist = torch.distributions.Categorical(logits=lg.float())
            logp = logp + dist.log_prob(actions[:, k])
            ent = ent + dist.entropy()
        return logp, ent

    def export_weights(self, out_dir: Path, stem: str = "neural_policy") -> Tuple[Path, Path]:
        """
        Export portable float32 binary weights + JSON manifest for loading into Rust/JS
        as a future NEURAL bot difficulty.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        bin_path = out_dir / f"{stem}.bin"
        meta_path = out_dir / f"{stem}.json"

        manifest = {
            "format": "hyperstrike_rope_v1",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_entity_layers": self.n_entity_layers,
            "n_temporal_layers": self.n_temporal_layers,
            "seq_len": self.seq_len,
            "action_n_bins": ACTION_N_BINS,
            "tensors": [],
        }

        offset = 0
        with open(bin_path, "wb") as f:
            for name, param in self.state_dict().items():
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
