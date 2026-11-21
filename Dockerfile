FROM rust:1-alpine AS builder
RUN apk add --no-cache build-base
WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+crt-static" cargo build --release

FROM scratch
COPY --from=builder /app/target/release/shopping-sessions /shopping-sessions
USER 1000:1000
ENTRYPOINT ["/shopping-sessions"]