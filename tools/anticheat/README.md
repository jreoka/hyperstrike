# Behavioral anti-cheat

Detects automated play (bots, aim-assist, trigger-bots) from the inputs the server already
receives. Physics are server-authoritative, so input automation is the only cheat left.

## How it works

1. **Per match, per human** (`rust/sim/src/anticheat.rs`, runs inside the match on the server):
   - **Hard client invariants:** the official client always sends `yaw == steer` and digital roll.
     Violations mean a modified client or injector.
   - **Shadow bot:** the champion AI runs "in the player's seat" (read-only). The server measures
     how closely the player tracks its intent and at what lag. A human can't beat their own
     reaction time; a bot can.
   - **Motor/timing features** (55 total): analog signal shape, button-hold regularity, reaction
     latency to unpredictable touches (after subtracting the measured WebSocket RTT), kickoff
     timing, network jitter.
   - **Tiny MLP** (`rust/sim/assets/ac_model.json`, ~6k params, microseconds per score).
2. **Reports** go to the primary region: `/data/ac_state.json` holds the aggregates and
   `/data/ac_reports.ndjson` holds the full log used for retraining.
3. **Ban waves** (primary, every `AC_WAVE_HOURS`):
   - Players who pass strict thresholds become candidates: consistently high model score, repeated
     invariant violations, or repeated shadow-bot matches.
   - Candidates are summarized as pseudonymous numbers only (no names or chat, so there's no
     prompt-injection surface) and sent to an OpenAI-compatible LLM.
   - A `ban` verdict with confidence ≥ `AC_LLM_MIN_CONFIDENCE` applies the existing escalating
     matchmaking ban.
   - Safety: a wave aborts if candidates exceed max(30, 5% of active players), since that points
     to a client update or model bug rather than real cheating. Bans are capped per wave.
4. **Your only job:** Admin → Anti-Cheat → **Unban (false positive)**. That lifts the ban, refunds
   the escalation step, resets their evidence, and labels their matches as legit for retraining.

## Configuration

`fly.toml` already sets `AC_LLM_BASE_URL` (OpenRouter) and `AC_LLM_MODEL` (`tencent/hy4-flash`).
Set the key as a secret:

```
fly secrets set AC_LLM_API_KEY=sk-or-...
```

Optional settings:

| Variable | Default | Effect |
|---|---|---|
| `AC_ENFORCE` | `1` | `0` = judge and record verdicts, never ban |
| `AC_WAVE_HOURS` | `24` | Hours between waves |
| `AC_LLM_MIN_CONFIDENCE` | `0.8` | Minimum LLM confidence required to ban |
| `AC_MAX_BANS_PER_WAVE` | `20` | Hard cap on bans per wave |
| `AC_MODEL_PATH` | unset | Load a model JSON from disk instead of the embedded one |

The first wave runs one full interval after the first deploy, so data accumulates first. You can
trigger a dry run any time from the admin panel.

## Retraining (on the 4090 machine)

```
# synthetic data only (regenerates ~6000 simulated matches, ~35 min on 32 threads)
python tools/anticheat/train.py --gen 6000

# better: add real data. Admin → Anti-Cheat → "Export training data", then:
python tools/anticheat/train.py --real ~/Downloads/hyperstrike_ac_export.ndjson

fly deploy
```

How real exports are weighted:

| Rows | Treated as | Weight |
|---|---|---|
| Unlabeled real players | human | 0.5 |
| Admin-cleared false positives | human | 3 |
| Wave-banned (never cleared) | cheat | 2 |

Retrain every few weeks, or after you clear a false positive.
