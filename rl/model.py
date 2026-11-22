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


class UnifiedEnergyBlock(nn.Module):
    """
    Unified Energy Transformer block (Hopfield attention + GELU FFN gradient flow)
    with tied weights (w_q/w_q^T, w_k/w_k^T, w_m/w_m^T), learned inverse temperature
    log_beta, learned step size dt, and optional RoPE + inverse-RoPE on attention gradients.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 3,
        steps: int = 3,
        dt: float = 0.25,
        is_causal: bool = False,
        use_rope: bool = True,
        symmetric_k: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_ratio = mlp_ratio
        self.steps = steps
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.symmetric_k = symmetric_k and not is_causal
        self.norm = nn.RMSNorm(dim, eps=1e-6)

        self.w_q = nn.Parameter(torch.empty(dim, dim))
        self.w_k = nn.Parameter(torch.empty(dim, dim))
        self.w_m = nn.Parameter(torch.empty(dim * mlp_ratio, dim))
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_m)

        self.log_beta = nn.Parameter(
            torch.full((1, num_heads, 1, 1), -0.5 * math.log(float(self.head_dim)))
        )
        self.dt = nn.Parameter(torch.tensor(float(dt)))

        if self.use_rope:
            cos, sin = build_rope_cache(64, self.head_dim, torch.device("cpu"), torch.float32)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        if self.use_rope:
            if T <= self._rope_cos.shape[2]:
                cos = self._rope_cos[:, :, :T, :].to(dtype=x.dtype)
                sin = self._rope_sin[:, :, :T, :].to(dtype=x.dtype)
            else:
                cos, sin = build_rope_cache(T, self.head_dim, x.device, x.dtype)
        else:
            cos = sin = None
        beta = self.log_beta.exp().to(x.dtype)

        for _ in range(self.steps):
            g = self.norm(x)

            q = F.linear(g, self.w_q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(g, self.w_k).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            if self.use_rope:
                q_rot = apply_rope(q, cos, sin)
                k_rot = apply_rope(k, cos, sin)
            else:
                q_rot = q
                k_rot = k

            grad_q_rot = F.scaled_dot_product_attention(
                q_rot * beta,
                k_rot,
                k_rot,
                attn_mask=attn_mask,
                is_causal=self.is_causal,
                scale=1.0,
            )
            if self.use_rope:
                grad_q = apply_rope(grad_q_rot, cos, -sin).transpose(1, 2).reshape(B, T, D)
            else:
                grad_q = grad_q_rot.transpose(1, 2).reshape(B, T, D)
            d_attn = F.linear(grad_q, self.w_q.t())

            if self.symmetric_k:
                grad_k_rot = F.scaled_dot_product_attention(
                    k_rot * beta,
                    q_rot,
                    q_rot,
                    attn_mask=attn_mask,
                    is_causal=False,
                    scale=1.0,
                )
                if self.use_rope:
                    grad_k = apply_rope(grad_k_rot, cos, -sin).transpose(1, 2).reshape(B, T, D)
                else:
                    grad_k = grad_k_rot.transpose(1, 2).reshape(B, T, D)
                d_attn = d_attn + F.linear(grad_k, self.w_k.t())

            h = F.gelu(F.linear(g, self.w_m), approximate="tanh")
            d_ffn = F.linear(h, self.w_m.t())

            x = x + self.dt * (d_attn + d_ffn)

        return x


class RoPEHyperstrikePolicy(nn.Module):
    """
    Unified Energy Transformer Policy for Hyperstrike (1v1 through 5v5).

    1. Entity UnifiedEnergyBlock: Permutation-invariant Hopfield energy refinement over
       [SELF], [BALL], up to 9 other cars (masked for 1v1..5v5), and 6 nearest boost pads.
    2. Temporal UnifiedEnergyBlock: Causal RoPE energy refinement over the sliding history
       [e_{t-T+1}, ..., e_t] (supports exact per-step KV-caching in Rust!).
    3. Raw Control Heads: 9 factored (incl. Mayhem item button) raw-control categorical heads + PPO value head.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        mlp_ratio: int = 3,
        entity_steps: int = 2,
        temporal_steps: int = 3,
        seq_len: int = 12,
        dt: float = 0.25,
        symmetric_k: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.entity_steps = entity_steps
        self.temporal_steps = temporal_steps
        self.seq_len = seq_len
        self.symmetric_k = symmetric_k

        self.self_proj = nn.Linear(SELF_DIM, d_model)
        self.ball_proj = nn.Linear(BALL_DIM, d_model)
        self.other_proj = nn.Linear(OTHER_DIM, d_model)
        self.pad_proj = nn.Linear(PAD_DIM, d_model)
        self.type_emb = nn.Embedding(4, d_model)

        self.entity_block = UnifiedEnergyBlock(
            dim=d_model,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            steps=entity_steps,
            dt=dt,
            is_causal=False,
            use_rope=False,
            symmetric_k=symmetric_k,
        )
        self.entity_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.temporal_block = UnifiedEnergyBlock(
            dim=d_model,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            steps=temporal_steps,
            dt=dt,
            is_causal=True,
            use_rope=True,
            symmetric_k=False,
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

        tokens = self.entity_block(tokens, attn_mask=attn_mask)
        return self.entity_norm(tokens[:, 0])

    def forward_temporal(self, frame_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.temporal_block(frame_seq)
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
            "format": "hyperstrike_energy_rope_v3",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
            "entity_steps": self.entity_steps,
            "temporal_steps": self.temporal_steps,
            "seq_len": self.seq_len,
            "symmetric_k": self.symmetric_k,
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
