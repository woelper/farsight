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
./scripts/setup-model.sh          # one-time: download LLM (~941 MB) + write config
./scripts/start.sh                # build, start daemon, launch CLI (needs a terminal)
./scripts/start.sh --daemon-only  # daemon only; test with ./target/debug/predict-cli
./scripts/start.sh --stop         # stop the daemon
# manual equivalent:
./target/debug/predictd &          # start the per-user daemon
./target/debug/predict-cli         # type; Tab accepts word, grey sentence
                                   # follows as you type, Ctrl+Right accepts
                                   # it, Enter commits, Ctrl+P pauses learning,
                                   # Ctrl+F twice forgets all, Esc quits
```

Without `setup-model.sh` there are no sentence suggestions — `start.sh`
says so on startup. The generated config (`~/.config/predict/predictd.toml`):

```toml
[llm]
enabled = true
model_path = "/path/to/qwen2.5-1.5b-q4_k_m.gguf"
max_tokens = 32
confidence_threshold = -1.5

[personal]
enabled = true   # learning on by default; Ctrl+P pauses at runtime
lambda = 0.7     # p = lambda * p_base + (1 - lambda) * p_personal
db_path = ""     # empty = XDG data default (~/.local/share/predict/predict.db)
```

## Status

M6 done — prediction works in every IBus app: `frontend-ibus` activates
through the daemon, ghost continuations render inline, `Tab` accepts,
password/terminal fields stay silent, typing never stalls. Install:
`docs/INSTALL-linux.md`. M5 done — style-aware prediction (default/du/sie
builtins, per-app + sticky resolution, word filtering, LLM logit-bias and
stop control, style-cycled CLI, zero violations on the style eval).
Novel generalization numbers: `docs/NOVEL-EVAL.md`.
