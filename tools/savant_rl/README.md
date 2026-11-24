# SAVANT Neural Policy Training (`savant_rl`)

This directory contains the reinforcement-learning training pipeline for **SAVANT**, Hyperstrike's top bot difficulty.  
It runs multi-threaded Rust physics environments (`rust/rl_env`) across your CPU cores while running batched PyTorch PPO on your CUDA GPU, then exports a compact ~137 KB quantized `int8` (`HSP1`) MLP directly to `rust/sim/assets/savant_policy.bin`.

No speed hacks or physics cheats are used — SAVANT obeys the exact same car physics, boost, and hitbox rules as human players.

---

## Quick Start: Train Longer (Resume from Current Best)

Run from the repository root (`hyperstrike/`):

```powershell
# ~1.5–2 hours (~115M more env steps on Ryzen 9 5950X + RTX 4090)
python tools\savant_rl\train.py --resume tools\savant_rl\checkpoints\best.pt --stage3 --envs 256 --updates 1500 --eval-every 25 --seed 101
```

### Run Length Guide (`--envs 256`, `--stage3`)
Each `100` updates is ~7.7M env steps (~31M physics ticks) and takes ~11 minutes on a 5950X + RTX 4090:

| Target Duration | `--updates` | Approx. Env Steps |
|---|---:|---:|
| ~40 min | `350` | ~27M |
| ~1.5–2 hours | `1500` | ~115M |
| ~3 hours | `2500` | ~193M |
| Overnight (~6–7 hours) | `5000` | ~386M |

> **Tip:** Bump `--seed` (`101`, `102`, `200`, …) on each new resumed session so the environment pool doesn't replay the exact same episode sequence.

---

## Resuming After Stopping

- **Resume from the all-time strongest checkpoint** *(recommended if the last eval dipped)*:
  ```powershell
  python tools\savant_rl\train.py --resume tools\savant_rl\checkpoints\best.pt --stage3 --envs 256 --updates 2000 --eval-every 25 --seed 202
  ```
- **Resume from the exact point where you stopped / pressed `Ctrl+C`**:
  ```powershell
  python tools\savant_rl\train.py --resume tools\savant_rl\checkpoints\latest.pt --stage3 --envs 256 --updates 2000 --eval-every 25 --seed 203
  ```

---

## Training Stages Explained

Training uses a 3-stage curriculum:

1. **Stage 1 — Behavioral Cloning (BC Warmstart)** *(automatic when `--resume` is omitted)*  
   Collects transitions from the handcrafted MPC-SAVANT planner and trains the MLP via supervised learning so it starts with basic driving, boosting, and flipping rather than random noise.
2. **Stage 2 — Base PPO** *(default when `--stage3` is omitted)*  
   - **80% 1v1 / 20% 2v2**
   - Opponent mix: 65% self-play, 25% past snapshots, 6% Champ, 3% Legend, 1% MPC-SAVANT.
   - Higher learning rate (`1.2e-4`), higher exploration entropy, and stronger ball-approach shaping reward to build core mechanics quickly.
3. **Stage 3 — Competitive Fine-Tuning (`--stage3`)** *(always use this when resuming)*  
   - **55% 1v1 / 45% 2v2** (heavier 2v2 with teammate anti-double-commit spacing cues).
   - Harder opponent mix: 50% self-play, 25% past snapshots, 15% Legend, 10% MPC-SAVANT (no Champ).
   - Pure goal focus: ball-chasing shaping reward anneals from `0.25 → 0.0` so the bot optimizes for scoring and preventing goals rather than touch farming.
   - Lower learning rate (`8e-5`) and tighter entropy to polish mechanics without destabilizing learned skills.

### Starting Completely From Scratch (Stages 1 + 2)
Only do this if you want to discard the existing checkpoints and train a brand-new policy from zero:

```powershell
python tools\savant_rl\train.py --bc-samples 400000 --bc-epochs 12 --envs 256 --updates 400 --eval-every 25 --seed 42
```

---

## Automatic Checkpointing, Deployment & `Ctrl+C`

Every `--eval-every` updates, `train.py` runs full-match native Rust evaluations against `CHAMP`, `LEGEND`, and `MPC-SAVANT` across 1v1 and 2v2:

- **`tools/savant_rl/checkpoints/latest.pt`** — Saved on every evaluation and immediately on `Ctrl+C`.
- **`tools/savant_rl/checkpoints/best.pt`** & **`best_policy.bin`** — Updated only when the composite evaluation score beats the previous all-time best.
- **`rust/sim/assets/savant_policy.bin`** *(live game model)* — **Automatically updated** whenever a new all-time best also beats `LEGEND` in 1v1. Look for:
  ```text
  [SAVE] DEPLOYED NEW BEST SAVANT MLP -> ...\rust\sim\assets\savant_policy.bin
  ```

### Pressing `Ctrl+C`
- Press **`Ctrl+C` once** and wait ~1 second.
- `train.py` catches `KeyboardInterrupt`, saves your current weights to `tools/savant_rl/checkpoints/latest.pt`, and cleanly frees the Rust worker pool.
- Your best deployed `rust/sim/assets/savant_policy.bin` and `best.pt` are **never overwritten or downgraded** when interrupting or resuming from `latest.pt`.

---

## Applying a Newly Trained Model to the Game

`rust/sim` and `rust/server_core` embed `rust/sim/assets/savant_policy.bin` at compile time via `include_bytes!`. After training deploys a new best model, rebuild the Rust workspace and restart your server:

```powershell
cargo build --release --manifest-path rust\Cargo.toml
```

Once the server restarts and you refresh the browser, both server-side bots and the browser client's `/savant_policy.bin` endpoint will use the newly trained weights.

---

## Useful CLI Flags

| Flag | Default | Description |
|---|---:|---|
| `--resume <path>` | `""` | Resume from a `.pt` checkpoint (`best.pt` or `latest.pt`) |
| `--stage3` | off | Enable Stage 3 competitive fine-tuning curriculum |
| `--envs <N>` | `192` | Parallel Rust match environments (`256` works well on 16-core CPUs) |
| `--updates <N>` | `300` | Number of PPO updates to run |
| `--eval-every <N>` | `20` | Run full-match evals & save checkpoints every `N` updates |
| `--rollout-steps <N>` | `128` | Policy steps per rollout (30 Hz decision rate; `128` ≈ 4.3s) |
| `--minibatch <N>` | `4096` | PPO minibatch size on GPU |
| `--ppo-epochs <N>` | `4` | PPO gradient epochs per update |
| `--seed <N>` | `42` | RNG seed for Python, PyTorch, and Rust envs |
