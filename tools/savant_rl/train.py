#!/usr/bin/env python3
"""Zero-flag, autonomous from-scratch RL trainer for SAVANT.

Just run:
    python tools/savant_rl/train.py

Everything is handled automatically:
- Pure from-scratch raw-button RL (no imitation / BC warmstart).
- Auto-resumes from `checkpoints/latest.pt` (or `best.pt`) if present.
- Auto-detects & auto-advances curriculum stages (Genesis -> Rising -> Competitive -> Apex).
- Trains forever until you press Ctrl+C (safely saves `latest.pt` on Ctrl+C).
- Evaluates & checkpoints every 50 updates.
- Only ever overwrites `best.pt` and `rust/sim/assets/savant_policy.bin` with a strictly better model.
- Automatically cleans up any old/stale checkpoint files, keeping only `latest.pt` and `best.pt`.
"""

from __future__ import annotations

import ctypes
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

OBS_DIM = 100
ACT_BRANCHES = (3, 5, 3, 3, 3, 2, 2, 2)
ACT_HEADS = len(ACT_BRANCHES)
TOTAL_LOGITS = sum(ACT_BRANCHES)
H1, H2, H3 = 256, 256, 128

# Training hyperparameters (zero flags needed)
NUM_ENVS = 256
ROLLOUT_STEPS = 128
MINIBATCH_SIZE = 4096
PPO_EPOCHS = 4
EVAL_EVERY = 50
GAMMA = 0.994
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
MAX_SNAPSHOTS = 24

STAGES = {
    0: {"name": "GENESIS",     "actor_lr": 3.0e-4, "critic_lr": 8.0e-4, "shaping": 1.00, "ent_coef": 0.0040},
    1: {"name": "RISING",      "actor_lr": 2.0e-4, "critic_lr": 6.0e-4, "shaping": 0.45, "ent_coef": 0.0020},
    2: {"name": "COMPETITIVE", "actor_lr": 1.0e-4, "critic_lr": 4.0e-4, "shaping": 0.12, "ent_coef": 0.0008},
    3: {"name": "APEX",        "actor_lr": 5.0e-5, "critic_lr": 2.5e-4, "shaping": 0.00, "ent_coef": 0.0003},
}


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
        w = layer.weight.detach().cpu().float().numpy()
        b = layer.bias.detach().cpu().float().numpy()
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


def evaluate_full(lib: ctypes.CDLL, model_bytes: bytes):
    e_chp1 = eval_against(lib, model_bytes, opp_diff=3, team_size=1, matches=4, secs=180.0, arena=0)
    e_leg1 = eval_against(lib, model_bytes, opp_diff=4, team_size=1, matches=6, secs=180.0, arena=0)
    e_mpc1 = eval_against(lib, model_bytes, opp_diff=5, team_size=1, matches=6, secs=180.0, arena=0)
    e_leg2 = eval_against(lib, model_bytes, opp_diff=4, team_size=2, matches=4, secs=180.0, arena=0)
    e_mpc2 = eval_against(lib, model_bytes, opp_diff=5, team_size=2, matches=4, secs=180.0, arena=0)

    gd_chp1 = e_chp1["goals_a"] - e_chp1["goals_b"]
    gd_leg1 = e_leg1["goals_a"] - e_leg1["goals_b"]
    gd_mpc1 = e_mpc1["goals_a"] - e_mpc1["goals_b"]
    gd_leg2 = e_leg2["goals_a"] - e_leg2["goals_b"]
    gd_mpc2 = e_mpc2["goals_a"] - e_mpc2["goals_b"]
    score = (
        gd_mpc1 * 2.5
        + gd_mpc2 * 2.5
        + gd_leg1 * 1.0
        + gd_leg2 * 1.0
        + gd_chp1 * 0.5
        - (e_mpc1["owngoals_a"] + e_mpc2["owngoals_a"]) * 3.0
    )
    return {
        "chp1": e_chp1,
        "leg1": e_leg1,
        "mpc1": e_mpc1,
        "leg2": e_leg2,
        "mpc2": e_mpc2,
        "score": float(score),
    }


def detect_stage(ev: dict, current_stage: int) -> int:
    """Monotonically determines the right curriculum stage from live eval results."""
    stage = current_stage
    if stage < 1 and (ev["chp1"]["goals_a"] >= ev["chp1"]["goals_b"] or (ev["chp1"]["goals_a"] >= 8 and ev["score"] > -280.0)):
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


def format_eval(upd: int, stage: int, total_steps: int, ev: dict) -> str:
    return (
        f"[EVAL upd {upd:05d} | {total_steps/1e6:.2f}M | S{stage}:{STAGES[stage]['name']}] "
        f"1v1 CHAMP: {ev['chp1']['goals_a']}-{ev['chp1']['goals_b']} | "
        f"1v1 LEGEND: {ev['leg1']['goals_a']}-{ev['leg1']['goals_b']} (W {ev['leg1']['wins_a']}-{ev['leg1']['wins_b']}) | "
        f"1v1 MPC: {ev['mpc1']['goals_a']}-{ev['mpc1']['goals_b']} (W {ev['mpc1']['wins_a']}-{ev['mpc1']['wins_b']}) | "
        f"2v2 LEGEND: {ev['leg2']['goals_a']}-{ev['leg2']['goals_b']} | "
        f"2v2 MPC: {ev['mpc2']['goals_a']}-{ev['mpc2']['goals_b']} | "
        f"score={ev['score']:.1f}"
    )


def main():
    seed = int(time.time()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) | pure from-scratch RL | Ctrl+C anytime to save & stop", flush=True)

    lib = build_and_load_dll()
    cleanup_old_checkpoints()

    actor = ActorNet().to(device)
    critic = CriticNet().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=STAGES[0]["actor_lr"], eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=STAGES[0]["critic_lr"], eps=1e-5)

    stage = 0
    upd = 0
    total_env_steps = 0
    best_score = -1e9
    deployed_best_score = -1e9
    snapshots: list[bytes] = []

    # Protect any already-deployed live model from ever being overwritten by a weaker policy
    if OUT_MODEL.exists() and OUT_MODEL.stat().st_size > 16:
        try:
            live_bytes = OUT_MODEL.read_bytes()
            live_ev = evaluate_full(lib, live_bytes)
            deployed_best_score = live_ev["score"]
            snapshots.append(live_bytes)
            print(f"[init] live deployed {OUT_MODEL.name} benchmarked at score={deployed_best_score:.1f} (protected — will only overwrite if beaten)", flush=True)
        except Exception as exc:
            print(f"[init] could not benchmark existing {OUT_MODEL.name}: {exc}", flush=True)

    # Check best.pt threshold so best.pt can never be downgraded either
    if BEST_PT.exists():
        bck = torch.load(BEST_PT, map_location="cpu", weights_only=True)
        best_score = float(bck.get("best_score", bck.get("score", -1e9)))
        deployed_best_score = max(deployed_best_score, best_score)

    # Auto-resume from latest.pt (or best.pt)
    resume_path = LATEST_PT if LATEST_PT.exists() else (BEST_PT if BEST_PT.exists() else None)
    if resume_path is not None:
        ck = torch.load(resume_path, map_location=device, weights_only=True)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
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

        cur_bytes = export_hsp1(actor)
        snapshots.append(cur_bytes)
        init_ev = evaluate_full(lib, cur_bytes)
        stage = detect_stage(init_ev, saved_stage)
        print(f"[init] auto-resumed from {resume_path.name} (upd={upd}, steps={total_env_steps/1e6:.2f}M, best={best_score:.1f})", flush=True)
        print(format_eval(upd, stage, total_env_steps, init_ev), flush=True)
    else:
        snapshots.append(export_hsp1(actor))
        print("[init] no checkpoints found — starting completely from scratch (Stage 0: GENESIS)", flush=True)

    # Also seed league with best.pt if it differs from latest.pt
    if BEST_PT.exists():
        try:
            tmp_actor = ActorNet()
            bck = torch.load(BEST_PT, map_location="cpu", weights_only=True)
            tmp_actor.load_state_dict(bck["actor"])
            snapshots.append(export_hsp1(tmp_actor))
        except Exception:
            pass

    def make_checkpoint_payload(cur_score: float) -> dict:
        return {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "upd": upd,
            "total_env_steps": total_env_steps,
            "stage": stage,
            "score": cur_score,
            "best_score": best_score,
        }

    pool, num_agents, obs_np, act_np, rew_np, done_np, stats_np = make_pool(lib, stage, seed, snapshots)
    obs_buf = torch.zeros((ROLLOUT_STEPS, num_agents, OBS_DIM), dtype=torch.float32, device=device)
    act_buf = torch.zeros((ROLLOUT_STEPS, num_agents, ACT_HEADS), dtype=torch.int64, device=device)
    logp_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
    val_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
    done_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)

    print(
        f"[PPO] running forever in Stage {stage} ({STAGES[stage]['name']}) | "
        f"{NUM_ENVS} envs, {num_agents} controlled agents, eval/save every {EVAL_EVERY} updates",
        flush=True,
    )

    t_start = time.time()
    last_eval_score = best_score

    try:
        while True:
            upd += 1
            cfg = STAGES[stage]
            for pg in actor_opt.param_groups:
                pg["lr"] = cfg["actor_lr"]
            for pg in critic_opt.param_groups:
                pg["lr"] = cfg["critic_lr"]
            shaping = cfg["shaping"]
            ent_coef = cfg["ent_coef"]

            t_roll = time.time()
            ep_gs = ep_gc = ep_tc = 0
            actor.eval()
            critic.eval()
            for t in range(ROLLOUT_STEPS):
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

            total_env_steps += ROLLOUT_STEPS * num_agents
            dt_roll = time.time() - t_roll

            # GAE
            with torch.no_grad():
                next_val = critic(torch.from_numpy(obs_np).to(device))
                adv_buf = torch.zeros_like(rew_buf)
                last_gae = torch.zeros(num_agents, device=device)
                for t in reversed(range(ROLLOUT_STEPS)):
                    next_nonterm = 1.0 - done_buf[t]
                    next_v = next_val if t == ROLLOUT_STEPS - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + GAMMA * next_v * next_nonterm - val_buf[t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterm * last_gae
                    adv_buf[t] = last_gae
                ret_buf = adv_buf + val_buf

            b_obs = obs_buf.reshape(-1, OBS_DIM)
            b_act = act_buf.reshape(-1, ACT_HEADS)
            b_logp = logp_buf.reshape(-1)
            b_adv = adv_buf.reshape(-1)
            b_ret = ret_buf.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            n_batch = b_obs.shape[0]
            actor.train()
            critic.train()
            v_losses, ent_vals, kls = [], [], []
            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(n_batch, device=device)
                kl_break = False
                for i in range(0, n_batch, MINIBATCH_SIZE):
                    mb = perm[i : i + MINIBATCH_SIZE]

                    v_pred = critic(b_obs[mb])
                    v_loss = 0.5 * F.mse_loss(v_pred, b_ret[mb])
                    critic_opt.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_opt.step()
                    v_losses.append(v_loss.item())

                    new_logp, ent = actor.eval_actions(b_obs[mb], b_act[mb])
                    log_ratio = new_logp - b_logp[mb]
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    kls.append(approx_kl)
                    if approx_kl > 0.025:
                        kl_break = True
                        break

                    pg1 = -b_adv[mb] * ratio
                    pg2 = -b_adv[mb] * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                    pg_loss = torch.max(pg1, pg2).mean()
                    ent_mean = ent.mean()
                    loss = pg_loss - ent_coef * ent_mean

                    actor_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_opt.step()
                    ent_vals.append(ent_mean.item())
                if kl_break:
                    break

            sps = (ROLLOUT_STEPS * num_agents) / max(1e-3, dt_roll)
            if upd == 1 or upd % 10 == 0:
                mean_r = rew_buf.mean().item() * 30.0
                print(
                    f"[PPO S{stage}] upd {upd:05d} steps={total_env_steps/1e6:.2f}M "
                    f"sps={sps:,.0f} rew/s={mean_r:+.3f} vsBotG={ep_gs}-{ep_gc} touches={ep_tc} "
                    f"vloss={np.mean(v_losses):.3f} ent={np.mean(ent_vals) if ent_vals else 0:.2f} "
                    f"kl={np.mean(kls) if kls else 0:.4f} elapsed={(time.time()-t_start)/60:.1f}m",
                    flush=True,
                )

            if upd % EVAL_EVERY == 0:
                cand_bytes = export_hsp1(actor)
                snapshots.append(cand_bytes)
                if len(snapshots) > MAX_SNAPSHOTS:
                    snapshots.pop(0)
                add_snapshot(lib, pool, cand_bytes)

                ev = evaluate_full(lib, cand_bytes)
                score = ev["score"]
                last_eval_score = score
                print(format_eval(upd, stage, total_env_steps, ev), flush=True)

                # Only ever overwrite best.pt with a strictly better checkpoint
                if score > best_score:
                    best_score = score
                    atomic_torch_save(make_checkpoint_payload(score), BEST_PT)
                    print(f"[SAVE] NEW BEST checkpoint -> {BEST_PT.name} (score={best_score:.1f})", flush=True)

                # Only ever overwrite the live game model if strictly better AND beats LEGEND 1v1
                if score > deployed_best_score and ev["leg1"]["goals_a"] > ev["leg1"]["goals_b"]:
                    deployed_best_score = score
                    atomic_write_bytes(cand_bytes, OUT_MODEL)
                    print(f"[DEPLOY] NEW LIVE SAVANT MODEL -> {OUT_MODEL} ({len(cand_bytes):,} bytes, score={deployed_best_score:.1f})", flush=True)

                # Save latest.pt and clean up any old/stale files
                atomic_torch_save(make_checkpoint_payload(score), LATEST_PT)
                cleanup_old_checkpoints()

                # Auto-advance curriculum stage if ready
                next_stage = detect_stage(ev, stage)
                if next_stage > stage:
                    print(f"[STAGE] AUTO-PROMOTING Stage {stage} ({STAGES[stage]['name']}) -> Stage {next_stage} ({STAGES[next_stage]['name']})!", flush=True)
                    stage = next_stage
                    atomic_torch_save(make_checkpoint_payload(score), LATEST_PT)
                    lib.rl_pool_free(pool)
                    pool, num_agents, obs_np, act_np, rew_np, done_np, stats_np = make_pool(
                        lib, stage, (seed + upd * 997) & 0x7FFFFFFF, snapshots
                    )
                    obs_buf = torch.zeros((ROLLOUT_STEPS, num_agents, OBS_DIM), dtype=torch.float32, device=device)
                    act_buf = torch.zeros((ROLLOUT_STEPS, num_agents, ACT_HEADS), dtype=torch.int64, device=device)
                    logp_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
                    val_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
                    rew_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
                    done_buf = torch.zeros((ROLLOUT_STEPS, num_agents), dtype=torch.float32, device=device)
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
