FROM rust:bookworm AS builder
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl

COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+crt-static" \
    cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/shopping-sessions /shopping-sessions
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]
