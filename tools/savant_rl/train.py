#!/usr/bin/env python3
"""Train the SAVANT neural MLP policy (BC warmstart + Multi-Threaded PPO Self-Play + League).

Runs Rust physics across CPU threads (`rust/rl_env`) + batched PyTorch PPO on CUDA GPU (RTX 4090).
Exports compact int8-quantized `HSP1` weights directly to `rust/sim/assets/savant_policy.bin`.
"""

from __future__ import annotations

import argparse
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

OBS_DIM = 100
ACT_BRANCHES = (3, 5, 3, 3, 3, 2, 2, 2)
ACT_HEADS = len(ACT_BRANCHES)
TOTAL_LOGITS = sum(ACT_BRANCHES)
H1, H2, H3 = 256, 256, 128


def build_and_load_dll() -> ctypes.CDLL:
    print("[build] compiling rust/rl_env (--release)...", flush=True)
    subprocess.run(["cargo", "build", "--release"], cwd=RL_ENV_DIR, check=True)
    lib = ctypes.CDLL(str(DLL_PATH))

    lib.rl_collect_bc.argtypes = [
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.rl_collect_bc.restype = ctypes.c_size_t

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
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.rl_eval_policy.restype = ctypes.c_int
    return lib


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, h1: int = H1, h2: int = H2, h3: int = H3):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, h1)
        self.n1 = nn.LayerNorm(h1)
        self.l2 = nn.Linear(h1, h2)
        self.n2 = nn.LayerNorm(h2)
        self.l3 = nn.Linear(h2, h3)
        self.n3 = nn.LayerNorm(h3)
        self.head = nn.Linear(h3, TOTAL_LOGITS)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.n1(self.l1(x)))
        x = F.silu(self.n2(self.l2(x)))
        x = F.silu(self.n3(self.l3(x)))
        return self.head(x)

    def split_logits(self, logits: torch.Tensor):
        return torch.split(logits, ACT_BRANCHES, dim=-1)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor):
        logits = self.forward(obs)
        parts = self.split_logits(logits)
        acts = []
        logp_total = torch.zeros(obs.shape[0], device=obs.device)
        for p in parts:
            dist = torch.distributions.Categorical(logits=p)
            a = dist.sample()
            acts.append(a)
            logp_total = logp_total + dist.log_prob(a)
        return torch.stack(acts, dim=-1), logp_total

    def eval_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.forward(obs)
        parts = self.split_logits(logits)
        logp = torch.zeros(obs.shape[0], device=obs.device)
        ent = torch.zeros(obs.shape[0], device=obs.device)
        for i, p in enumerate(parts):
            dist = torch.distributions.Categorical(logits=p)
            logp = logp + dist.log_prob(actions[:, i])
            ent = ent + dist.entropy()
        return logp, ent


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, h1: int = 384, h2: int = 384, h3: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.SiLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def export_hsp1(actor: ActorNet) -> bytes:
    """Exports ActorNet to compact per-row int8 quantized HSP1 binary format."""
    buf = bytearray()
    buf += b"HSP1"
    buf += struct.pack("<HHHHHH", OBS_DIM, H1, H2, H3, TOTAL_LOGITS, 0)

    def write_linear(layer: nn.Linear):
        w = layer.weight.detach().cpu().float().numpy()  # [out, in]
        b = layer.bias.detach().cpu().float().numpy()    # [out]
        max_abs = np.max(np.abs(w), axis=1)
        scales = np.where(max_abs > 1e-8, max_abs / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int8)
        buf.extend(scales.tobytes())
        buf.extend(b.astype(np.float32).tobytes())
        buf.extend(q.tobytes())

    def write_ln(ln: nn.LayerNorm):
        buf.extend(ln.weight.detach().cpu().float().numpy().astype(np.float32).tobytes())
        buf.extend(ln.bias.detach().cpu().float().numpy().astype(np.float32).tobytes())

    write_linear(actor.l1)
    write_ln(actor.n1)
    write_linear(actor.l2)
    write_ln(actor.n2)
    write_linear(actor.l3)
    write_ln(actor.n3)
    write_linear(actor.head)
    return bytes(buf)


def eval_against(lib: ctypes.CDLL, model_bytes: bytes, opp_diff: int, team_size: int = 1, matches: int = 6, secs: float = 180.0, arena: int = 0):
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
        metrics.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    assert ok == 1, "rl_eval_policy failed to parse HSP1 bytes"
    return {
        "goals_a": int(metrics[0]),
        "goals_b": int(metrics[1]),
        "wins_a": int(metrics[2]),
        "wins_b": int(metrics[3]),
        "shots_a": int(metrics[4]),
        "shots_b": int(metrics[5]),
        "owngoals_a": int(metrics[6]),
    }


def run_bc_warmstart(
    lib: ctypes.CDLL,
    actor: ActorNet,
    device: torch.device,
    samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
):
    t0 = time.time()
    print(f"[BC] collecting {samples:,} transitions from MPC-SAVANT across CPU cores...", flush=True)
    obs = np.empty((samples, OBS_DIM), dtype=np.float32)
    acts = np.empty((samples, ACT_HEADS), dtype=np.uint8)
    got = lib.rl_collect_bc(
        samples,
        seed,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        acts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    print(f"[BC] collected {got:,} samples in {time.time() - t0:.1f}s ({got / max(1e-3, time.time() - t0):,.0f} samples/s)", flush=True)

    obs_t = torch.from_numpy(obs).to(device)
    acts_t = torch.from_numpy(acts.astype(np.int64)).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Extra weight on jump/boost/steer so flips & aim are crisp!
    head_weights = [1.0, 1.4, 1.2, 1.0, 0.8, 2.0, 1.3, 1.0]

    actor.train()
    n = obs_t.shape[0]
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        tot_loss = 0.0
        tot_acc = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = actor(obs_t[idx])
            parts = actor.split_logits(logits)
            loss = 0.0
            acc = 0.0
            for h, p in enumerate(parts):
                loss = loss + head_weights[h] * F.cross_entropy(p, acts_t[idx, h], label_smoothing=0.01)
                acc = acc + (p.argmax(dim=-1) == acts_t[idx, h]).float().mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_acc += acc / ACT_HEADS
            batches += 1
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[BC] epoch {ep:02d}/{epochs} loss={tot_loss / batches:.4f} avg_head_acc={100.0 * tot_acc / batches:.2f}%", flush=True)


def run_ppo(
    lib: ctypes.CDLL,
    actor: ActorNet,
    critic: CriticNet,
    device: torch.device,
    num_envs: int,
    rollout_steps: int,
    total_updates: int,
    minibatch_size: int,
    ppo_epochs: int,
    eval_every: int,
    seed: int,
    critic_warmup: int = 5,
    stage3: bool = False,
    init_best_score: float = -1e9,
):
    pool = lib.rl_pool_new(num_envs, seed, 1 if stage3 else 0)
    num_agents = lib.rl_pool_num_agents(pool)
    print(f"[PPO] created pool (mode={'stage3' if stage3 else 'base'}): {num_envs} envs, {num_agents} controlled agents, batch={num_agents * rollout_steps:,} steps/update", flush=True)

    obs_np = np.zeros((num_agents, OBS_DIM), dtype=np.float32)
    act_np = np.zeros((num_agents, ACT_HEADS), dtype=np.uint8)
    rew_np = np.zeros((num_agents,), dtype=np.float32)
    done_np = np.zeros((num_agents,), dtype=np.uint8)
    stats_np = np.zeros((3,), dtype=np.uint32)

    lib.rl_pool_reset(pool, obs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    # Seed initial snapshot with current actor so Snapshot opponents are strong from update 1
    bc_bytes = export_hsp1(actor)
    bc_arr = np.frombuffer(bc_bytes, dtype=np.uint8)
    lib.rl_pool_add_snapshot(pool, bc_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(bc_bytes), 16)

    base_actor_lr = 8e-5 if stage3 else 1.2e-4
    base_critic_lr = 4e-4 if stage3 else 6e-4
    actor_opt = torch.optim.Adam(actor.parameters(), lr=base_actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=base_critic_lr, eps=1e-5)

    obs_buf = torch.zeros((rollout_steps, num_agents, OBS_DIM), dtype=torch.float32, device=device)
    act_buf = torch.zeros((rollout_steps, num_agents, ACT_HEADS), dtype=torch.int64, device=device)
    logp_buf = torch.zeros((rollout_steps, num_agents), dtype=torch.float32, device=device)
    val_buf = torch.zeros((rollout_steps, num_agents), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((rollout_steps, num_agents), dtype=torch.float32, device=device)
    done_buf = torch.zeros((rollout_steps, num_agents), dtype=torch.float32, device=device)

    gamma = 0.992
    gae_lambda = 0.95
    clip_eps = 0.2
    best_score = init_best_score
    total_env_steps = 0
    last_upd = 0
    t_start = time.time()

    try:
        for upd in range(1, total_updates + 1):
            frac = 1.0 - (upd - 1) / max(1, total_updates)
            shaping = (0.25 * frac) if stage3 else max(0.10, frac)
            ent_coef = (0.00025 * frac) if stage3 else (0.0003 + 0.0012 * frac)
            # Linear LR decay
            actor_lr = base_actor_lr * (0.12 + 0.88 * frac) if upd > critic_warmup else 0.0
            for pg in actor_opt.param_groups:
                pg["lr"] = actor_lr
            for pg in critic_opt.param_groups:
                pg["lr"] = base_critic_lr * (0.12 + 0.88 * frac)

            t_roll = time.time()
            ep_gs = ep_gc = ep_tc = 0
            actor.eval()
            critic.eval()
            for t in range(rollout_steps):
                obs_t = torch.from_numpy(obs_np).to(device, non_blocking=True)
                obs_buf[t] = obs_t
                with torch.no_grad():
                    a_t, lp_t = actor.sample_action(obs_t)
                    v_t = critic(obs_t)
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
                rew_buf[t] = torch.from_numpy(rew_np).to(device, non_blocking=True)
                done_buf[t] = torch.from_numpy(done_np.astype(np.float32)).to(device, non_blocking=True)
                ep_gs += int(stats_np[0])
                ep_gc += int(stats_np[1])
                ep_tc += int(stats_np[2])

            total_env_steps += rollout_steps * num_agents
            dt_roll = time.time() - t_roll

            # GAE computation
            with torch.no_grad():
                next_val = critic(torch.from_numpy(obs_np).to(device))
                adv_buf = torch.zeros_like(rew_buf)
                last_gae = torch.zeros(num_agents, device=device)
                for t in reversed(range(rollout_steps)):
                    next_nonterm = 1.0 - done_buf[t]
                    next_v = next_val if t == rollout_steps - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + gamma * next_v * next_nonterm - val_buf[t]
                    last_gae = delta + gamma * gae_lambda * next_nonterm * last_gae
                    adv_buf[t] = last_gae
                ret_buf = adv_buf + val_buf

            b_obs = obs_buf.reshape(-1, OBS_DIM)
            b_act = act_buf.reshape(-1, ACT_HEADS)
            b_logp = logp_buf.reshape(-1)
            b_adv = adv_buf.reshape(-1)
            b_ret = ret_buf.reshape(-1)
            b_val = val_buf.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            n_batch = b_obs.shape[0]
            actor.train()
            critic.train()
            pg_losses, v_losses, ent_vals, kls = [], [], [], []
            for _ in range(ppo_epochs):
                perm = torch.randperm(n_batch, device=device)
                kl_break = False
                for i in range(0, n_batch, minibatch_size):
                    mb = perm[i : i + minibatch_size]
                    # Critic update
                    v_pred = critic(b_obs[mb])
                    v_loss = 0.5 * F.mse_loss(v_pred, b_ret[mb])
                    critic_opt.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_opt.step()
                    v_losses.append(v_loss.item())

                    if upd <= critic_warmup:
                        continue

                    new_logp, ent = actor.eval_actions(b_obs[mb], b_act[mb])
                    log_ratio = new_logp - b_logp[mb]
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    kls.append(approx_kl)
                    if approx_kl > 0.02:
                        kl_break = True
                        break

                    pg1 = -b_adv[mb] * ratio
                    pg2 = -b_adv[mb] * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                    pg_loss = torch.max(pg1, pg2).mean()
                    ent_mean = ent.mean()
                    loss = pg_loss - ent_coef * ent_mean

                    actor_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_opt.step()
                    pg_losses.append(pg_loss.item())
                    ent_vals.append(ent_mean.item())
                if kl_break:
                    break

            last_upd = upd
            sps = (rollout_steps * num_agents) / max(1e-3, dt_roll)
            if upd == 1 or upd % 5 == 0 or upd == critic_warmup:
                mean_r = rew_buf.mean().item() * 30.0  # per-second reward
                print(
                    f"[PPO] upd {upd:03d}/{total_updates} steps={total_env_steps/1e6:.2f}M "
                    f"sps={sps:,.0f} rew/s={mean_r:+.3f} vsBotG={ep_gs}-{ep_gc} touches={ep_tc} "
                    f"vloss={np.mean(v_losses):.3f} ent={np.mean(ent_vals) if ent_vals else 0:.2f} "
                    f"kl={np.mean(kls) if kls else 0:.4f} elapsed={(time.time()-t_start)/60:.1f}m",
                    flush=True,
                )

            # Periodically add snapshot to league pool & run native full-match evaluation
            if upd % eval_every == 0 or upd == total_updates:
                cand_bytes = export_hsp1(actor)
                cand_arr = np.frombuffer(cand_bytes, dtype=np.uint8)
                lib.rl_pool_add_snapshot(
                    pool,
                    cand_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    len(cand_bytes),
                    16,
                )
                e_chp1 = eval_against(lib, cand_bytes, opp_diff=3, team_size=1, matches=4, secs=180.0, arena=0)
                e_leg1 = eval_against(lib, cand_bytes, opp_diff=4, team_size=1, matches=6, secs=180.0, arena=0)
                e_mpc1 = eval_against(lib, cand_bytes, opp_diff=5, team_size=1, matches=6, secs=180.0, arena=0)
                e_mpc2 = eval_against(lib, cand_bytes, opp_diff=5, team_size=2, matches=4, secs=180.0, arena=0)
                e_leg2 = eval_against(lib, cand_bytes, opp_diff=4, team_size=2, matches=4, secs=180.0, arena=0)

                # Composite strength score prioritizing beating BOTH MPC-SAVANT and LEGEND across 1v1 & 2v2
                gd_chp1 = e_chp1["goals_a"] - e_chp1["goals_b"]
                gd_leg1 = e_leg1["goals_a"] - e_leg1["goals_b"]
                gd_mpc1 = e_mpc1["goals_a"] - e_mpc1["goals_b"]
                gd_mpc2 = e_mpc2["goals_a"] - e_mpc2["goals_b"]
                gd_leg2 = e_leg2["goals_a"] - e_leg2["goals_b"]
                score = gd_mpc1 * 2.5 + gd_mpc2 * 2.5 + gd_leg1 * 1.0 + gd_leg2 * 1.0 + gd_chp1 * 0.5 - (e_mpc1["owngoals_a"] + e_mpc2["owngoals_a"]) * 3.0

                print(
                    f"[EVAL upd {upd:03d}] "
                    f"1v1 CHAMP: {e_chp1['goals_a']}-{e_chp1['goals_b']} | "
                    f"1v1 LEGEND: {e_leg1['goals_a']}-{e_leg1['goals_b']} (W {e_leg1['wins_a']}-{e_leg1['wins_b']}) | "
                    f"1v1 MPC: {e_mpc1['goals_a']}-{e_mpc1['goals_b']} (W {e_mpc1['wins_a']}-{e_mpc1['wins_b']}) | "
                    f"2v2 LEGEND: {e_leg2['goals_a']}-{e_leg2['goals_b']} | "
                    f"2v2 MPC: {e_mpc2['goals_a']}-{e_mpc2['goals_b']} | "
                    f"score={score:.1f}",
                    flush=True,
                )
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                if score > best_score:
                    best_score = score
                    (CKPT_DIR / "best_policy.bin").write_bytes(cand_bytes)
                    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "upd": upd, "score": score, "best_score": best_score}, CKPT_DIR / "best.pt")
                    if e_leg1["goals_a"] > e_leg1["goals_b"]:
                        OUT_MODEL.write_bytes(cand_bytes)
                        print(f"[SAVE] DEPLOYED NEW BEST SAVANT MLP -> {OUT_MODEL} ({len(cand_bytes):,} bytes, score={best_score:.1f})", flush=True)
                    else:
                        print(f"[SAVE] NEW BEST CKPT -> {CKPT_DIR / 'best_policy.bin'} (score={best_score:.1f}, waiting to beat LEGEND before deploying)", flush=True)
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "upd": upd, "score": score, "best_score": best_score}, CKPT_DIR / "latest.pt")
    except KeyboardInterrupt:
        print(f"\n[INT] Ctrl+C received after update {last_upd} — saving {CKPT_DIR / 'latest.pt'}...", flush=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": actor.state_dict(), "critic": critic.state_dict(), "upd": last_upd, "score": best_score, "best_score": best_score},
            CKPT_DIR / "latest.pt",
        )
        print(f"[INT] Saved latest.pt safely. Best deployed model remains intact at {OUT_MODEL} (best score={best_score:.1f}).", flush=True)
    finally:
        lib.rl_pool_free(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-samples", type=int, default=800_000)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--bc-batch", type=int, default=4096)
    ap.add_argument("--envs", type=int, default=192)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--stage3", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)

    lib = build_and_load_dll()
    actor = ActorNet().to(device)
    critic = CriticNet().to(device)

    init_best = -1e9
    best_ckpt_path = CKPT_DIR / "best.pt"
    if best_ckpt_path.exists():
        bck = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
        init_best = float(bck.get("best_score", bck.get("score", -1e9)))
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        init_best = max(init_best, float(ck.get("best_score", ck.get("score", -1e9))))
        print(f"[init] resumed from {args.resume} (best score threshold={init_best:.1f})", flush=True)
    else:
        run_bc_warmstart(lib, actor, device, args.bc_samples, args.bc_epochs, args.bc_batch, args.seed)
        bc_bytes = export_hsp1(actor)
        e_leg = eval_against(lib, bc_bytes, opp_diff=4, team_size=1, matches=6, secs=180.0, arena=0)
        e_mpc = eval_against(lib, bc_bytes, opp_diff=5, team_size=1, matches=6, secs=180.0, arena=0)
        e_leg2 = eval_against(lib, bc_bytes, opp_diff=4, team_size=2, matches=4, secs=180.0, arena=0)
        print(
            f"[BC EVAL] 1v1 vs LEGEND: {e_leg['goals_a']}-{e_leg['goals_b']} (W {e_leg['wins_a']}-{e_leg['wins_b']}) | "
            f"1v1 vs MPC: {e_mpc['goals_a']}-{e_mpc['goals_b']} (W {e_mpc['wins_a']}-{e_mpc['wins_b']}) | "
            f"2v2 vs LEGEND: {e_leg2['goals_a']}-{e_leg2['goals_b']}",
            flush=True,
        )
        # Save BC baseline immediately if it already beats LEGEND
        if e_leg["goals_a"] > e_leg["goals_b"]:
            OUT_MODEL.write_bytes(bc_bytes)
            print(f"[SAVE] BC baseline saved -> {OUT_MODEL} ({len(bc_bytes):,} bytes)", flush=True)

    run_ppo(
        lib,
        actor,
        critic,
        device,
        num_envs=args.envs,
        rollout_steps=args.rollout_steps,
        total_updates=args.updates,
        minibatch_size=args.minibatch,
        ppo_epochs=args.ppo_epochs,
        eval_every=args.eval_every,
        seed=args.seed + (5000 if args.stage3 else 1000),
        critic_warmup=3 if args.stage3 else 5,
        stage3=args.stage3,
        init_best_score=init_best,
    )


if __name__ == "__main__":
    main()
