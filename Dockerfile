# Multi-stage pure Rust build for Hyperstrike (Fly.io 256MB RAM compatible)
FROM rust:1-bookworm AS builder

WORKDIR /build

COPY index.html ./index.html
COPY rust/sim ./rust/sim
COPY rust/server_core ./rust/server_core
COPY rust/match_engine ./rust/match_engine
COPY rust/app ./rust/app

RUN cargo build --release --manifest-path rust/app/Cargo.toml \
    && strip rust/app/target/release/server

COPY rl ./rl
RUN mkdir -p /build/neural_out \
    && if [ -f /build/rl/checkpoints/neural_policy.bin ] && [ -f /build/rl/checkpoints/neural_policy.json ]; then \
         cp /build/rl/checkpoints/neural_policy.bin /build/rl/checkpoints/neural_policy.json /build/neural_out/; \
       elif [ -f /build/rl/checkpoints/neural_policy_best.bin ] && [ -f /build/rl/checkpoints/neural_policy_best.json ]; then \
         cp /build/rl/checkpoints/neural_policy_best.bin /build/neural_out/neural_policy.bin \
         && cp /build/rl/checkpoints/neural_policy_best.json /build/neural_out/neural_policy.json; \
       fi

FROM debian:bookworm-slim

WORKDIR /app

ENV PORT=8080
ENV DATA_DIR=/data
ENV HYPERSTRIKE_NEURAL_POLICY=/app/rl/checkpoints

RUN useradd -r -u 1000 -m hyperstrike \
    && mkdir -p /data /app/rl/checkpoints \
    && chown -R hyperstrike:hyperstrike /app /data

COPY --from=builder /build/rust/app/target/release/server /app/server
COPY --from=builder --chown=hyperstrike:hyperstrike /build/neural_out/ /app/rl/checkpoints/

USER hyperstrike

EXPOSE 8080

CMD ["/app/server"]
