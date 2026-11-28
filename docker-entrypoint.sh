#!/bin/sh
set -eu

DATA_DIR="${DATA_DIR:-/data}"
RL_DIR="${DATA_DIR}/savant_rl"
CKPT_DIR="${RL_DIR}/checkpoints"
POLICY_OUT="${SAVANT_POLICY_PATH:-${RL_DIR}/savant_policy.bin}"

mkdir -p "${CKPT_DIR}"

# Seed persistent volume from image checkpoints on first boot (never overwrite existing progress)
if [ ! -f "${CKPT_DIR}/latest.pt" ] && [ -f "/app/seed/checkpoints/latest.pt" ]; then
    echo "[entrypoint] Seeding ${CKPT_DIR}/latest.pt from image"
    cp /app/seed/checkpoints/latest.pt "${CKPT_DIR}/latest.pt"
fi
if [ ! -f "${CKPT_DIR}/best.pt" ] && [ -f "/app/seed/checkpoints/best.pt" ]; then
    echo "[entrypoint] Seeding ${CKPT_DIR}/best.pt from image"
    cp /app/seed/checkpoints/best.pt "${CKPT_DIR}/best.pt"
fi
if [ ! -f "${POLICY_OUT}" ] && [ -f "/app/seed/savant_policy.bin" ]; then
    echo "[entrypoint] Seeding ${POLICY_OUT} from image"
    cp /app/seed/savant_policy.bin "${POLICY_OUT}"
fi

NPROC="$(nproc 2>/dev/null || echo 1)"
SERVER_PID=""
TRAIN_WRAPPER_PID=""
STOPPING=0

shutdown() {
    if [ "${STOPPING}" -eq 1 ]; then
        return
    fi
    STOPPING=1
    echo "[entrypoint] Shutdown signal received — saving trainer checkpoint and flushing server..."
    if [ -n "${TRAIN_WRAPPER_PID}" ]; then
        kill -TERM "${TRAIN_WRAPPER_PID}" 2>/dev/null || true
    fi
    if [ -n "${SERVER_PID}" ]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${TRAIN_WRAPPER_PID}" ]; then
        wait "${TRAIN_WRAPPER_PID}" 2>/dev/null || true
    fi
    if [ -n "${SERVER_PID}" ]; then
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit 0
}

trap shutdown TERM INT

# Start live Rust game server pinned to CPU 0 when >= 2 CPUs are available
if [ "${NPROC}" -ge 2 ] && command -v taskset >/dev/null 2>&1; then
    echo "[entrypoint] Starting game server pinned to CPU 0 (nproc=${NPROC}, SIM_WORKERS=${SIM_WORKERS:-auto})"
    taskset -c 0 /app/server &
else
    echo "[entrypoint] Starting game server (nproc=${NPROC}, SIM_WORKERS=${SIM_WORKERS:-auto})"
    /app/server &
fi
SERVER_PID=$!

# Start background SAVANT RL trainer pinned to CPU 1 (nice 19, single-threaded, pauses on live players)
if [ "${SAVANT_BG_TRAIN:-0}" = "1" ]; then
    (
        TRAIN_CHILD=""
        stop_trainer() {
            if [ -n "${TRAIN_CHILD}" ]; then
                kill -TERM "${TRAIN_CHILD}" 2>/dev/null || true
                wait "${TRAIN_CHILD}" 2>/dev/null || true
            fi
            exit 0
        }
        trap stop_trainer TERM INT

        # Give the game server a moment to bind /api/health
        sleep 3

        export SAVANT_RL_LIB="/app/librl_env.so"
        export SAVANT_CKPT_DIR="${CKPT_DIR}"
        export SAVANT_OUT_MODEL="${POLICY_OUT}"
        export SAVANT_HEALTH_URL="http://127.0.0.1:${PORT:-8080}/api/health"
        export SAVANT_TORCH_THREADS="1"
        export SAVANT_EVAL_WORKERS="1"
        export RL_ENV_THREADS="1"
        export OMP_NUM_THREADS="1"
        export MKL_NUM_THREADS="1"
        export OPENBLAS_NUM_THREADS="1"
        export NUMEXPR_NUM_THREADS="1"

        while true; do
            if [ "${NPROC}" -ge 2 ] && command -v taskset >/dev/null 2>&1; then
                echo "[entrypoint] Starting SAVANT RL trainer pinned to CPU 1 (nice 19)"
                taskset -c 1 nice -n 19 /opt/venv/bin/python -u /app/tools/savant_rl/train.py &
            else
                echo "[entrypoint] Starting SAVANT RL trainer (nice 19)"
                nice -n 19 /opt/venv/bin/python -u /app/tools/savant_rl/train.py &
            fi
            TRAIN_CHILD=$!
            set +e
            wait "${TRAIN_CHILD}"
            CODE=$?
            set -e
            TRAIN_CHILD=""
            echo "[entrypoint] SAVANT RL trainer exited with code ${CODE}; restarting in 15s..."
            sleep 15 &
            wait $! || true
        done
    ) &
    TRAIN_WRAPPER_PID=$!
fi

set +e
wait "${SERVER_PID}"
SERVER_EXIT=$?
set -e

echo "[entrypoint] Game server exited with code ${SERVER_EXIT}; stopping trainer..."
if [ -n "${TRAIN_WRAPPER_PID}" ]; then
    kill -TERM "${TRAIN_WRAPPER_PID}" 2>/dev/null || true
    wait "${TRAIN_WRAPPER_PID}" 2>/dev/null || true
fi
exit "${SERVER_EXIT}"
