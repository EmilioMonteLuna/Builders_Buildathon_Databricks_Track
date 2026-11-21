FROM rust:1.85-bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

COPY src ./src
RUN RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=builder /app/target/release/shopping-sessions /usr/local/bin/shopping-sessions

USER 1000:1000
CMD ["/usr/local/bin/shopping-sessions"]