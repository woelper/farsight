# predict — local word + sentence prediction for Linux

Local, private typing assistant. Suggests the next word and the rest of the
sentence while the user types in any app. Everything runs locally; no network.

See `plan.md` for the full milestone plan.

## Layout

Cargo workspace (`Cargo.toml` at root), Rust stable, edition 2021:

- `crates/predict-core` — types: `Context`, `Candidate`, `StyleSpec`; `Predictor` trait; ranking
- `crates/predict-ngram` — fast tier: trie + n-gram, personal counts
- `crates/predict-llm` — slow tier: `Backend` trait, llama.cpp implementation
- `crates/predict-store` — SQLite store for personal data
- `crates/predict-proto` — IPC messages, versioned, serialized with postcard
- `crates/predict-eval` — offline replay harness and metrics
- `crates/predictd` — daemon binary (per-user, Unix socket)
- `crates/predict-cli` — terminal client for manual testing
- `crates/frontend-ibus` — IBus engine over D-Bus (zbus)

Docs:

- `docs/ARCHITECTURE.md` — current architecture
- `docs/adr/` — architecture decision records
- `CHANGELOG.md` — per-milestone changes

## Build / test / lint

```sh
./scripts/ci.sh
# or individually:
cargo build
cargo test
cargo clippy -- -D warnings
# run the eval on the sample corpus:
cargo run -p predict-eval --example eval_sample
```

## Try it

```sh
./target/debug/predictd &          # start the per-user daemon
./target/debug/predict-cli         # type; Tab accepts, Enter commits, Esc quits
```

## Status

M2 done — daemon (`predictd`) + terminal client (`predict-cli`) work over
versioned IPC: typing shows live suggestions with no perceptible lag.
M1 fast tier numbers hold (top-1 0.897, savings 0.722, p99 0.181 ms).
LLM tier, store, style, IBus still placeholders.
