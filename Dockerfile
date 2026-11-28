# Multi-stage Rust + CPU PyTorch build for Hyperstrike (Fly.io shared-cpu-2x / 1024MB)
FROM rust:1-bookworm AS builder

WORKDIR /build

COPY index.html ./index.html
COPY rust/sim ./rust/sim
COPY rust/server_core ./rust/server_core
COPY rust/match_engine ./rust/match_engine
COPY rust/app ./rust/app
COPY rust/rl_env ./rust/rl_env

RUN cargo build --release --manifest-path rust/app/Cargo.toml \
    && strip rust/app/target/release/server \
    && cargo build --release --manifest-path rust/rl_env/Cargo.toml \
    && strip rust/rl_env/target/release/librl_env.so

FROM debian:bookworm-slim

WORKDIR /app

ENV PORT=8080
ENV DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv util-linux ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir numpy \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && useradd -r -u 1000 -m hyperstrike \
    && mkdir -p /data /app/tools/savant_rl /app/seed/checkpoints \
    && chown -R hyperstrike:hyperstrike /app /data

COPY --from=builder /build/rust/app/target/release/server /app/server
COPY --from=builder /build/rust/rl_env/target/release/librl_env.so /app/librl_env.so
COPY tools/savant_rl/train.py /app/tools/savant_rl/train.py
COPY tools/savant_rl/checkpoints/ /app/seed/checkpoints/
COPY rust/sim/assets/savant_policy.bin /app/seed/savant_policy.bin
COPY docker-entrypoint.sh /app/entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh \
    && chown -R hyperstrike:hyperstrike /app

USER hyperstrike

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
