# Hyperstrike RoPE Transformer RL Trainer

GPU-accelerated Self-Play + Bot-Sparring PPO trainer backed by the **exact Rust `sim` physics engine** (`rust/rl_env`) across **1v1 through 5v5**, using **Prodigy** as the optimizer with `tqdm` progress bars, automatic resuming, and periodic checkpoint saving/pruning.

## Architecture
- **Simulator (`rust/rl_env`)**: Multi-threaded Rust `cdylib` stepping `N` parallel `sim::World` instances at 120Hz (default 8 ticks/action = 15Hz). Randomly samples 1v1..5v5 (with extended pitch for 4v4/5v5) and mirrors Orange team so the policy is team-symmetric.
- **Model (`rl/model.py` — `RoPEHyperstrikePolicy`)**:
  - **Entity Encoder (1-layer masked Transformer, `d=64`)**: Attends over `[SELF]`, `[BALL]`, up to 9 other cars (masked for 1v1..5v5), and 6 nearest boost pads.
  - **Temporal RoPE Transformer (2-layer causal Transformer with Rotary Position Embeddings, RMSNorm, SwiGLU, FlashAttention/SDPA)**: Processes a sliding window of `T=12` frames (~0.8s), ~150K params (~600 KB) so it runs in microseconds on small Fly.io CPUs with KV-caching.
  - **Raw Control Heads**: 8 factored raw-control heads (`throttle`, `steer`, `pitch`, `yaw`, `roll`, `jump`, `boost`, `handbrake`) directly driving `sim::Controls`.
- **Optimizer**: `prodigyopt.Prodigy` (`lr=1.0`, `safeguard_warmup=True`, `decouple=True`).
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
