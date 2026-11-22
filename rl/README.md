# Hyperstrike RoPE Transformer RL Trainer

GPU-accelerated Self-Play + Bot-Sparring PPO trainer backed by the **exact Rust `sim` physics engine** (`rust/rl_env`) across **1v1 through 5v5**, using **AdamW** with `tqdm` progress bars, automatic resuming, and periodic checkpoint saving/pruning.

## Architecture
- **Simulator (`rust/rl_env`)**: Multi-threaded Rust `cdylib` stepping `N` parallel `sim::World` instances at 120Hz (default 8 ticks/action = 15Hz). Randomly samples 1v1..5v5 (with extended pitch for 4v4/5v5) and mirrors Orange team so the policy is team-symmetric.
- **Modes**: `--mayhem-prob` (default 0.4) of episodes run Mayhem with a random 6/10/15s power-up interval. Observations include held item (one-hot), whether it has a valid target, power-up timers, ball iced / zero-g / stuck state; a 9th action head drives the item button. Observations are built by `sim::neural::extract_obs`, the same function the live game uses, so training and inference can't drift.
- **Rewards**: goal ±10, ball-velocity-to-goal shaping, touches (rewarded only after a 0.5s contact gap so pinning/dribbling can't farm), shots/saves/demos, boost pickup, small penalty for failed item use. Rewards are normalized by the running return std; timeouts are bootstrapped (not treated as losses).
- **Model (`rl/model.py` — `UnifiedEnergyBlock` / `RoPEHyperstrikePolicy`)**:
  - **Entity `UnifiedEnergyBlock` (`d=64, h=4, mlp=3x, steps=2`)**: Permutation-invariant Hopfield energy refinement with tied weights (`w_q/w_q^T`, `w_k`, `w_m/w_m^T`) over `[SELF]`, `[BALL]`, up to 9 other cars (masked for 1v1..5v5), and 6 nearest boost pads.
  - **Temporal `UnifiedEnergyBlock` (`d=64, h=4, mlp=3x, steps=3, T=12`, causal RoPE + inverse-RoPE gradient flow)**: ~47K params (~190 KB `neural_policy.bin`) with exact per-step RoPE K-caching in Rust for ultra-fast Fly.io CPU inference.
  - **Raw Control Heads**: 9 factored raw-control heads (`throttle`, `steer`, `pitch`, `yaw`, `roll`, `jump`, `boost`, `handbrake`, `item`) directly driving `sim::Controls`. Export format `hyperstrike_energy_rope_v3` (v2 checkpoints are incompatible; start over with `--fresh`).
- **Optimizer**: `torch.optim.AdamW` (`lr=3e-3` cosine-decayed for BC warmstart; `lr=3e-4 -> 5e-5` with KL early-stopping for PPO).
- **Checkpoints**: Periodically saves `ckpt_upd_XXXXXXX.pt` + `latest.pt` + portable `neural_policy.bin`/`neural_policy.json`, prunes old periodic checkpoints beyond `--keep-checkpoints`, and **auto-resumes** from `latest.pt` on restart (pass `--fresh` to start over).

## Quick Start

```powershell
# 1. Start (or auto-resume) full 1v1..5v5 training with default settings:
#    --num-envs 128 --min-team-size 1 --max-team-size 5 --bc-updates 60 --total-updates 50000 --bot-prob 0.25 --bot-diff 4
python rl/train.py

# 2. Quick smoke test
python rl/train.py --num-envs 64 --rollout-steps 32 --bc-updates 2 --total-updates 5 --save-interval 2 --keep-checkpoints 2

# 3. Pass --fresh to ignore existing checkpoints and start from scratch.
```
