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
# sentence-tier eval (needs a GGUF model):
cargo run -p predict-eval --example eval_sentence -- <model.gguf> [threshold]
```

## Try it

```sh
./scripts/start.sh                # build, start daemon, launch CLI (needs a terminal)
./scripts/start.sh --daemon-only  # daemon only; test with ./target/debug/predict-cli
./scripts/start.sh --stop         # stop the daemon
# manual equivalent:
./target/debug/predictd &          # start the per-user daemon
./target/debug/predict-cli         # type; Tab accepts word, pause for grey
                                   # sentence, Ctrl+Right accepts it, Esc quits
```

To enable the slow tier, place a GGUF model (e.g. Qwen2.5-1.5B base Q4_K_M)
at a known path and write `~/.config/predict/predictd.toml`:

```toml
[llm]
enabled = true
model_path = "/path/to/qwen2.5-1.5b-q4_k_m.gguf"
max_tokens = 32
confidence_threshold = -1.0
```

## Status

M3 done — slow LLM tier works end to end: pause for a grey sentence
suggestion, Ctrl+Right accepts; eval shows +49 keystrokes over the word
tier (savings 0.755 vs 0.722, TTFT p50 ~160 ms).
Store, style, IBus still placeholders.
