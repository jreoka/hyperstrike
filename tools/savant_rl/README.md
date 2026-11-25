# SAVANT Neural Policy Training (`savant_rl`)

Zero-flag, pure from-scratch reinforcement learning for **SAVANT**.  
The **Causal RoPE Transformer** (Pre-RMSNorm + Rotary Position Embedding sliding-window attention + SwiGLU FFN, exported to int8 `HSP2` format with O(1)-projection KV-cached inference in Rust and JS) gets raw controller access with **fine-grained analog-resolution axes** — `throttle`, `steer`, `pitch`, `yaw`, `roll` each quantized to **21 levels (uniform 0.1 steps across [-1, 1])** — plus binary `jump`, `boost`, `handbrake`, `item`, with zero imitation/BC warmstart and figures out how to play across **all standard arenas, experimental Labs arenas (`HYPERLOOP 8`, `THROWBACK CORE`, `TWIN VORTEX`), and Mayhem power-up mode** through multi-threaded Rust self-play + league RL on your CPU/GPU.

---

## How to Run

From the repository root (`hyperstrike/`):

```powershell
python tools\savant_rl\train.py
```

That's it — **no flags needed.**

### What It Handles Automatically
- **Runs forever** until you press **`Ctrl+C`**.
- **Auto-resumes** from `tools/savant_rl/checkpoints/latest.pt` (or `best.pt`) whenever you run it again.
- **Pure from-scratch start:** Delete `tools/savant_rl/checkpoints/` if you ever want to start a brand-new random neural network from zero.
- **Auto-stages:** Automatically detects strength from evaluations and promotes through 4 stages without restarting:
  - **Stage 0 (`GENESIS`)** — 85% 1v1 / 15% 2v2 · Self-play + Snapshots + 5% Champ · Learns driving, flipping, hitting, scoring.
  - **Stage 1 (`RISING`)** — 70% 1v1 / 30% 2v2 · Adds 10% Legend + 2% MPC-SAVANT once it matches Champ.
  - **Stage 2 (`COMPETITIVE`)** — 55% 1v1 / 45% 2v2 · 15% Legend + 10% MPC-SAVANT once it beats Legend 1v1.
  - **Stage 3 (`APEX`)** — 50% 1v1 / 50% 2v2 · Pure goal reward (`shaping = 0`), deep self-play & historical league once it beats MPC-SAVANT 1v1 and matches Legend 2v2.
- **Evaluates & saves every 50 updates** (~5 minutes on Ryzen 9 5950X + RTX 4090).
- **Safe checkpointing & cleanup:**
  - Keeps **only** `latest.pt` and `best.pt` in `tools/savant_rl/checkpoints/` (automatically deletes stale/old checkpoint files and uses atomic writes so `Ctrl+C` never corrupts a file).
  - **No quality gate while training:** every new best checkpoint is written to `best.pt` **and** exported straight to the live game model (`rust/sim/assets/savant_policy.bin`) — no "must beat LEGEND 1v1" or score-protection gate. On startup, an existing `best.pt` is also exported to `rust/sim/assets/savant_policy.bin` so the asset always holds the best model.
- **Graceful `Ctrl+C`:** Press `Ctrl+C` once and wait ~1 second — it saves `latest.pt` safely and exits cleanly.

---

## Loading a New Best Model Into the Game

Whenever training prints `[DEPLOY] NEW LIVE SAVANT MODEL -> ...\rust\sim\assets\savant_policy.bin`, rebuild and restart the Rust server so the embedded weights update:

```powershell
cargo build --release --manifest-path rust\Cargo.toml
```
