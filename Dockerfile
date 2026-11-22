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

FROM debian:bookworm-slim

WORKDIR /app

ENV PORT=8080
ENV DATA_DIR=/data

RUN useradd -r -u 1000 -m hyperstrike \
    && mkdir -p /data \
    && chown -R hyperstrike:hyperstrike /app /data

COPY --from=builder /build/rust/app/target/release/server /app/server

USER hyperstrike

EXPOSE 8080

CMD ["/app/server"]
