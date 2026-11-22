import argparse
import copy
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prodigyopt import Prodigy
from tqdm import tqdm

from env import (
    ACTION_DIM,
    BALL_DIM,
    MAX_CARS,
    NUM_PADS_OBS,
    OTHER_CARS,
    OTHER_DIM,
    PAD_DIM,
    SELF_DIM,
    HyperstrikeVecEnv,
    controls_to_discrete,
    discrete_to_controls,
)
from model import RoPEHyperstrikePolicy

CKPT_PATTERN = re.compile(r"^ckpt_upd_(\d+)\.pt$")


class HistoryBuffer:
    """Maintains a sliding window of the last T frame embeddings per (env, car)."""

    def __init__(self, num_envs: int, max_cars: int, seq_len: int, d_model: int, device: torch.device):
        self.num_envs = num_envs
        self.max_cars = max_cars
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device
        self.buf = torch.zeros((num_envs, max_cars, seq_len, d_model), device=device, dtype=torch.float32)

    def reset_envs(self, dones: np.ndarray) -> None:
        done_idx = np.nonzero(dones)[0]
        if len(done_idx) > 0:
            self.buf[done_idx] = 0.0

    def reset_all(self) -> None:
        self.buf.zero_()

    def push(self, frame_emb: torch.Tensor) -> torch.Tensor:
        is_fresh = self.buf[:, :, -1, :].abs().sum(dim=-1) == 0.0
        self.buf = torch.roll(self.buf, shifts=-1, dims=2)
        self.buf[:, :, -1, :] = frame_emb
        if is_fresh.any():
            expanded = frame_emb.unsqueeze(2).expand(-1, -1, self.seq_len, -1)
            self.buf = torch.where(is_fresh[:, :, None, None], expanded, self.buf)
        return self.buf


def obs_to_tensors(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "self": torch.from_numpy(obs["self"]).to(device, non_blocking=True),
        "ball": torch.from_numpy(obs["ball"]).to(device, non_blocking=True),
        "other": torch.from_numpy(obs["other"]).to(device, non_blocking=True),
        "other_mask": torch.from_numpy(obs["other_mask"]).to(device, non_blocking=True),
        "pads": torch.from_numpy(obs["pads"]).to(device, non_blocking=True),
        "active_mask": torch.from_numpy(obs["active_mask"]).to(device, non_blocking=True),
    }


def atomic_torch_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    latest = ckpt_dir / "latest.pt"
    if latest.is_file():
        return latest
    numbered = []
    if ckpt_dir.is_dir():
        for p in ckpt_dir.iterdir():
            m = CKPT_PATTERN.match(p.name)
            if m:
                numbered.append((int(m.group(1)), p))
    if numbered:
        numbered.sort(key=lambda x: x[0])
        return numbered[-1][1]
    return None


def prune_old_checkpoints(ckpt_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    numbered = []
    for p in ckpt_dir.iterdir():
        m = CKPT_PATTERN.match(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda x: x[0])
    for _, old_path in numbered[:-keep]:
        try:
            old_path.unlink()
        except OSError:
            pass


def prodigy_effective_lr(optimizer: Prodigy) -> float:
    pg = optimizer.param_groups[0]
    return float(pg.get("d", 1.0) * pg.get("lr", 1.0))


def run_bc_warmstart(
    policy: RoPEHyperstrikePolicy,
    env: HyperstrikeVecEnv,
    optimizer: Prodigy,
    device: torch.device,
    bc_updates: int,
    rollout_steps: int,
    use_bf16: bool,
) -> None:
    """Optional Behavior Cloning warmstart from built-in CHAMP bots across 1v1..5v5."""
    if bc_updates <= 0:
        return
    env.set_bot_curriculum(0.0, 4)
    obs = env.reset()
    N, C, T = env.num_envs, MAX_CARS, policy.seq_len
    hist = HistoryBuffer(N, C, T, policy.d_model, device)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    pbar = tqdm(range(1, bc_updates + 1), desc="BC Warmstart", dynamic_ncols=True)
    for _ in pbar:
        seq_embs_list = []
        target_acts_list = []

        for _ in range(rollout_steps):
            expert_raw = env.get_expert_actions()
            expert_disc = controls_to_discrete(expert_raw)
            t_obs = obs_to_tensors(obs, device)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                e = policy.encode_frame(
                    t_obs["self"].reshape(N * C, SELF_DIM),
                    t_obs["ball"].reshape(N * C, BALL_DIM),
                    t_obs["other"].reshape(N * C, OTHER_CARS, OTHER_DIM),
                    t_obs["other_mask"].reshape(N * C, OTHER_CARS),
                    t_obs["pads"].reshape(N * C, NUM_PADS_OBS, PAD_DIM),
                ).view(N, C, policy.d_model)

            win = hist.push(e.float())
            active = obs["active_mask"] > 0.5
            if active.any():
                seq_embs_list.append(win[active].detach().cpu())
                target_acts_list.append(torch.from_numpy(expert_disc[active]))

            obs, _, dones, _ = env.step(expert_raw)
            hist.reset_envs(dones)

        X = torch.cat(seq_embs_list, dim=0)
        Y = torch.cat(target_acts_list, dim=0)
        perm = torch.randperm(X.shape[0])
        bs = 1024
        losses = []
        for start in range(0, X.shape[0], bs):
            idx = perm[start : start + bs]
            xb = X[idx].to(device, non_blocking=True)
            yb = Y[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                logits, _ = policy.forward_temporal(xb)
                loss = sum(F.cross_entropy(lg, yb[:, k]) for k, lg in enumerate(logits)) / ACTION_DIM
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        pbar.set_postfix(loss=f"{np.mean(losses):.4f}", d_lr=f"{prodigy_effective_lr(optimizer):.2e}")
    pbar.close()


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA GPU not available. Pass --cpu to force CPU training.")
    device = torch.device("cpu" if args.cpu else "cuda")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(
        f"[Hyperstrike RL] Device: {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | "
        f"bf16={use_bf16} | envs={args.num_envs} | "
        f"team_sizes={args.min_team_size}v{args.min_team_size}..{args.max_team_size}v{args.max_team_size} | "
        f"opt=Prodigy"
    )

    env = HyperstrikeVecEnv(
        num_envs=args.num_envs,
        min_team_size=args.min_team_size,
        max_team_size=args.max_team_size,
        ticks_per_step=args.ticks_per_step,
        max_episode_steps=args.max_episode_steps,
        bot_opponent_prob=args.bot_prob,
        bot_diff=args.bot_diff,
        num_threads=args.threads,
        seed=args.seed,
    )

    policy = RoPEHyperstrikePolicy(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_entity_layers=args.entity_layers,
        n_temporal_layers=args.temporal_layers,
        seq_len=args.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[Model] RoPEHyperstrikePolicy: {n_params / 1e6:.2f}M params (d={args.d_model}, h={args.n_heads}, T={args.seq_len})")

    optimizer = Prodigy(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_coef=args.d_coef,
        safeguard_warmup=True,
        use_bias_correction=True,
        decouple=True,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_update = 1
    total_env_steps = 0
    total_episodes = 0
    best_winrate = -1.0
    resumed = False

    # Auto-resume unless --fresh is explicitly passed
    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = Path(args.resume)
    elif not args.fresh:
        resume_path = find_latest_checkpoint(ckpt_dir)

    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 0) + 1
        total_env_steps = ckpt.get("total_env_steps", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        best_winrate = ckpt.get("best_winrate", -1.0)
        resumed = True
        print(
            f"[Auto-Resume] Loaded {resume_path} at update {start_update - 1} "
            f"({total_episodes:,} episodes, best_vsBot={best_winrate * 100:.1f}%)"
            if best_winrate >= 0
            else f"[Auto-Resume] Loaded {resume_path} at update {start_update - 1} ({total_episodes:,} episodes)"
        )

    if start_update > args.total_updates:
        print(f"[Done] Already reached update {start_update - 1} >= total_updates ({args.total_updates}).")
        env.close()
        return

    # Optional Behavior Cloning warmstart (only when starting fresh)
    if args.bc_updates > 0 and not resumed:
        run_bc_warmstart(policy, env, optimizer, device, args.bc_updates, rollout_steps=16, use_bf16=use_bf16)

    # Restore PPO bot sparring curriculum
    env.set_bot_curriculum(args.bot_prob, args.bot_diff)
    obs = env.reset()

    N, C, T = args.num_envs, MAX_CARS, args.seq_len
    R = args.rollout_steps
    hist = HistoryBuffer(N, C, T, args.d_model, device)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    opponent_pool: deque[RoPEHyperstrikePolicy] = deque(maxlen=6)

    buf_self = torch.zeros((R, N, C, SELF_DIM), dtype=torch.float32)
    buf_ball = torch.zeros((R, N, C, BALL_DIM), dtype=torch.float32)
    buf_other = torch.zeros((R, N, C, OTHER_CARS, OTHER_DIM), dtype=torch.float32)
    buf_omask = torch.zeros((R, N, C, OTHER_CARS), dtype=torch.float32)
    buf_pads = torch.zeros((R, N, C, NUM_PADS_OBS, PAD_DIM), dtype=torch.float32)
    buf_hist = torch.zeros((R, N, C, T, args.d_model), dtype=torch.float16)
    buf_actions = torch.zeros((R, N, C, ACTION_DIM), dtype=torch.int64)
    buf_logp = torch.zeros((R, N, C), dtype=torch.float32)
    buf_values = torch.zeros((R, N, C), dtype=torch.float32)
    buf_rewards = torch.zeros((R, N, C), dtype=torch.float32)
    buf_dones = torch.zeros((R, N), dtype=torch.float32)
    buf_active = torch.zeros((R, N, C), dtype=torch.float32)

    def save_checkpoint(upd: int, is_best: bool = False) -> None:
        ckpt_data = {
            "update": upd,
            "total_env_steps": total_env_steps,
            "total_episodes": total_episodes,
            "best_winrate": best_winrate,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        periodic_path = ckpt_dir / f"ckpt_upd_{upd:07d}.pt"
        atomic_torch_save(ckpt_data, periodic_path)
        atomic_torch_save(ckpt_data, ckpt_dir / "latest.pt")
        prune_old_checkpoints(ckpt_dir, args.keep_checkpoints)
        policy.export_weights(ckpt_dir, stem="neural_policy")
        if is_best:
            atomic_torch_save(ckpt_data, ckpt_dir / "best.pt")
            policy.export_weights(ckpt_dir, stem="neural_policy_best")

    last_completed_update = start_update - 1
    pbar = tqdm(
        range(start_update, args.total_updates + 1),
        initial=start_update - 1,
        total=args.total_updates,
        desc="PPO",
        dynamic_ncols=True,
    )

    try:
        for update in pbar:
            frac = 1.0 - (update - 1) / max(1, args.total_updates)
            ent_coef = args.ent_coef_end + (args.ent_coef_start - args.ent_coef_end) * frac

            ep_goals_b = 0
            ep_goals_o = 0
            ep_done_count = 0
            ep_touches = 0
            ep_bot_w = 0
            ep_bot_l = 0

            past_opp = (
                opponent_pool[np.random.randint(len(opponent_pool))]
                if (len(opponent_pool) > 0 and np.random.rand() < 0.35)
                else None
            )
            opp_hist = HistoryBuffer(N, C, T, args.d_model, device) if past_opp is not None else None
            if opp_hist is not None:
                opp_hist.buf.copy_(hist.buf)

            policy.eval()
            t_roll = time.perf_counter()
            with torch.no_grad():
                for step in range(R):
                    buf_self[step].copy_(torch.from_numpy(obs["self"]))
                    buf_ball[step].copy_(torch.from_numpy(obs["ball"]))
                    buf_other[step].copy_(torch.from_numpy(obs["other"]))
                    buf_omask[step].copy_(torch.from_numpy(obs["other_mask"]))
                    buf_pads[step].copy_(torch.from_numpy(obs["pads"]))
                    buf_active[step].copy_(torch.from_numpy(obs["active_mask"]))

                    t_obs = obs_to_tensors(obs, device)
                    with torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                        e = policy.encode_frame(
                            t_obs["self"].reshape(N * C, SELF_DIM),
                            t_obs["ball"].reshape(N * C, BALL_DIM),
                            t_obs["other"].reshape(N * C, OTHER_CARS, OTHER_DIM),
                            t_obs["other_mask"].reshape(N * C, OTHER_CARS),
                            t_obs["pads"].reshape(N * C, NUM_PADS_OBS, PAD_DIM),
                        ).view(N, C, args.d_model)
                        win = hist.push(e.float())
                        logits, vals = policy.forward_temporal(win.reshape(N * C, T, args.d_model))

                    acts, logp, _ = policy.sample_actions(logits)
                    acts_nc = acts.view(N, C, ACTION_DIM)
                    buf_hist[step].copy_(win.to(torch.float16).cpu())
                    buf_actions[step].copy_(acts_nc.cpu())
                    buf_logp[step].copy_(logp.view(N, C).cpu())
                    buf_values[step].copy_(vals.view(N, C).float().cpu())

                    exec_acts = acts_nc.cpu().numpy()

                    if past_opp is not None and opp_hist is not None:
                        with torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                            oe = past_opp.encode_frame(
                                t_obs["self"].reshape(N * C, SELF_DIM),
                                t_obs["ball"].reshape(N * C, BALL_DIM),
                                t_obs["other"].reshape(N * C, OTHER_CARS, OTHER_DIM),
                                t_obs["other_mask"].reshape(N * C, OTHER_CARS),
                                t_obs["pads"].reshape(N * C, NUM_PADS_OBS, PAD_DIM),
                            ).view(N, C, args.d_model)
                            owin = opp_hist.push(oe.float())
                            ologits, _ = past_opp.forward_temporal(owin.reshape(N * C, T, args.d_model))
                        oacts, _, _ = past_opp.sample_actions(ologits)
                        oacts_np = oacts.view(N, C, ACTION_DIM).cpu().numpy()
                        team1_mask = (obs["team_ids"] == 1) & (obs["active_mask"] > 0.5)
                        exec_acts[team1_mask] = oacts_np[team1_mask]
                        buf_active[step][torch.from_numpy(team1_mask)] = 0.0

                    raw_ctrl = discrete_to_controls(exec_acts)
                    obs, rews, dones, stats = env.step(raw_ctrl)
                    hist.reset_envs(dones)
                    if opp_hist is not None:
                        opp_hist.reset_envs(dones)

                    buf_rewards[step].copy_(torch.from_numpy(rews))
                    buf_dones[step].copy_(torch.from_numpy(dones.astype(np.float32)))

                    ep_goals_b += stats.goals_blue
                    ep_goals_o += stats.goals_orange
                    ep_done_count += stats.episodes_done
                    ep_touches += stats.touches
                    ep_bot_w += stats.bot_games_won
                    ep_bot_l += stats.bot_games_lost

                t_obs = obs_to_tensors(obs, device)
                with torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                    e_last = policy.encode_frame(
                        t_obs["self"].reshape(N * C, SELF_DIM),
                        t_obs["ball"].reshape(N * C, BALL_DIM),
                        t_obs["other"].reshape(N * C, OTHER_CARS, OTHER_DIM),
                        t_obs["other_mask"].reshape(N * C, OTHER_CARS),
                        t_obs["pads"].reshape(N * C, NUM_PADS_OBS, PAD_DIM),
                    ).view(N, C, args.d_model)
                    win_last = torch.roll(hist.buf, shifts=-1, dims=2)
                    win_last[:, :, -1, :] = e_last.float()
                    _, next_vals = policy.forward_temporal(win_last.reshape(N * C, T, args.d_model))
                next_vals = next_vals.view(N, C).float().cpu()

            roll_sec = time.perf_counter() - t_roll
            total_env_steps += R * N
            total_episodes += ep_done_count

            # GAE advantage computation
            advantages = torch.zeros_like(buf_rewards)
            last_gae = torch.zeros((N, C), dtype=torch.float32)
            for t in reversed(range(R)):
                next_nonterminal = (1.0 - buf_dones[t]).unsqueeze(-1)
                next_v = next_vals if t == R - 1 else buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * next_v * next_nonterminal - buf_values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_values

            mask = buf_active > 0.5
            b_self = buf_self[mask]
            b_ball = buf_ball[mask]
            b_other = buf_other[mask]
            b_omask = buf_omask[mask]
            b_pads = buf_pads[mask]
            b_hist = buf_hist[mask]
            b_act = buf_actions[mask]
            b_old_logp = buf_logp[mask]
            b_old_val = buf_values[mask]
            b_adv = advantages[mask]
            b_ret = returns[mask]

            M = b_act.shape[0]
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            policy.train()
            t_opt = time.perf_counter()
            pg_losses, v_losses, ent_list = [], [], []
            for _ in range(args.ppo_epochs):
                perm = torch.randperm(M)
                for start in range(0, M, args.minibatch_size):
                    idx = perm[start : start + args.minibatch_size]
                    if len(idx) < 32:
                        continue
                    mb_self = b_self[idx].to(device, non_blocking=True)
                    mb_ball = b_ball[idx].to(device, non_blocking=True)
                    mb_other = b_other[idx].to(device, non_blocking=True)
                    mb_omask = b_omask[idx].to(device, non_blocking=True)
                    mb_pads = b_pads[idx].to(device, non_blocking=True)
                    mb_hist = b_hist[idx].to(device, non_blocking=True, dtype=torch.float32)
                    mb_act = b_act[idx].to(device, non_blocking=True)
                    mb_old_logp = b_old_logp[idx].to(device, non_blocking=True)
                    mb_old_val = b_old_val[idx].to(device, non_blocking=True)
                    mb_adv = b_adv[idx].to(device, non_blocking=True)
                    mb_ret = b_ret[idx].to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_bf16, dtype=amp_dtype):
                        e_cur = policy.encode_frame(mb_self, mb_ball, mb_other, mb_omask, mb_pads)
                        seq_in = torch.cat([mb_hist[:, :-1, :], e_cur.unsqueeze(1).float()], dim=1)
                        logits, new_val = policy.forward_temporal(seq_in)

                    new_logp, entropy = policy.eval_actions(logits, mb_act)
                    log_ratio = new_logp - mb_old_logp
                    ratio = log_ratio.exp()

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_clipped = mb_old_val + torch.clamp(new_val.float() - mb_old_val, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((new_val.float() - mb_ret) ** 2, (v_clipped - mb_ret) ** 2).mean()
                    ent_loss = entropy.mean()
                    loss = pg_loss + args.vf_coef * v_loss - ent_coef * ent_loss

                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    ent_list.append(ent_loss.item())

            opt_sec = time.perf_counter() - t_opt
            agent_sps = M / max(1e-6, roll_sec + opt_sec)
            bot_total = ep_bot_w + ep_bot_l
            bot_wr = (ep_bot_w / bot_total) if bot_total > 0 else float("nan")
            last_completed_update = update

            pbar.set_postfix(
                eps=f"{total_episodes:,}",
                asps=f"{agent_sps:.0f}",
                g=ep_goals_b + ep_goals_o,
                t=ep_touches,
                vsBot=f"{bot_wr * 100:.0f}%" if bot_total > 0 else "n/a",
                d=f"{prodigy_effective_lr(optimizer):.1e}",
                vf=f"{np.mean(v_losses):.2f}",
                ent=f"{np.mean(ent_list):.2f}",
            )

            if update % args.pool_interval == 0:
                snap = copy.deepcopy(policy).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                opponent_pool.append(snap)

            if update % args.save_interval == 0 or update == args.total_updates:
                is_best = bot_total >= 10 and bot_wr >= best_winrate
                if is_best:
                    best_winrate = bot_wr
                save_checkpoint(update, is_best=is_best)
                if is_best:
                    tqdm.write(
                        f"[Checkpoint] upd {update}: new best vsBot={best_winrate * 100:.1f}% "
                        f"-> saved best.pt & neural_policy_best.bin"
                    )
    except KeyboardInterrupt:
        pbar.close()
        if last_completed_update >= start_update:
            print(f"\n[Interrupt] Saving checkpoint at update {last_completed_update} before exiting...")
            save_checkpoint(last_completed_update, is_best=False)
        env.close()
        return

    pbar.close()
    env.close()
    print("[Done] Training complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a RoPE Transformer to play Hyperstrike (1v1..5v5) via Self-Play PPO + Prodigy.")
    p.add_argument("--num-envs", type=int, default=128, help="Parallel Rust environments")
    p.add_argument("--min-team-size", type=int, default=1, help="Minimum team size (1..5)")
    p.add_argument("--max-team-size", type=int, default=5, help="Maximum team size (1..5)")
    p.add_argument("--ticks-per-step", type=int, default=8, help="120Hz physics ticks per action (8 = 15Hz)")
    p.add_argument("--max-episode-steps", type=int, default=900, help="Max steps per episode (900 @ 15Hz = 60s)")
    p.add_argument("--rollout-steps", type=int, default=64, help="Steps per env per PPO update")
    p.add_argument("--total-updates", type=int, default=50000, help="Total PPO updates")
    p.add_argument("--bc-updates", type=int, default=60, help="CHAMP-bot Behavior Cloning warmstart updates (set 0 to skip)")
    p.add_argument("--bot-prob", type=float, default=0.25, help="Fraction of envs sparring against built-in bots")
    p.add_argument("--bot-diff", type=int, default=4, help="Sparring bot difficulty (0=Rookie..4=Champ)")
    p.add_argument("--d-model", type=int, default=64, help="Transformer hidden dimension (64 = fast Fly.io CPU)")
    p.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    p.add_argument("--entity-layers", type=int, default=1, help="Entity Transformer layers")
    p.add_argument("--temporal-layers", type=int, default=2, help="Causal Temporal RoPE Transformer layers")
    p.add_argument("--seq-len", type=int, default=12, help="Temporal context window in frames")
    p.add_argument("--lr", type=float, default=1.0, help="Prodigy base LR multiplier (keep 1.0)")
    p.add_argument("--d-coef", type=float, default=1.0, help="Prodigy d_coef scaling factor")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Prodigy decoupled weight decay")
    p.add_argument("--gamma", type=float, default=0.992, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip ratio")
    p.add_argument("--vf-coef", type=float, default=0.5, help="Value loss weight")
    p.add_argument("--ent-coef-start", type=float, default=0.015, help="Initial entropy bonus")
    p.add_argument("--ent-coef-end", type=float, default=0.002, help="Final entropy bonus")
    p.add_argument("--ppo-epochs", type=int, default=3, help="PPO epochs per update")
    p.add_argument("--minibatch-size", type=int, default=2048, help="Minibatch size on GPU")
    p.add_argument("--threads", type=int, default=None, help="Rust worker threads (default: auto)")
    p.add_argument("--pool-interval", type=int, default=100, help="Updates between adding snapshot to self-play pool")
    p.add_argument("--save-interval", type=int, default=100, help="Updates between periodic checkpoint saves")
    p.add_argument("--keep-checkpoints", type=int, default=5, help="Number of periodic ckpt_upd_*.pt files to keep")
    p.add_argument("--checkpoint-dir", type=str, default="rl/checkpoints", help="Checkpoint output directory")
    p.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume from")
    p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints in checkpoint-dir and start fresh")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
