# Changelog

All notable changes to `predict` are documented here, per milestone.

## [0.3.0] — M2 daemon + test client

- `predict-proto`: versioned envelopes (`PROTOCOL_VERSION = 1`), `postcard`
  bodies with `u32`-LE length prefix (256 KiB cap), `serde` derives;
  `ClientMsg::{ContextUpdate, Suggest, Cancel}`, `DaemonMsg::Suggestion`;
  user-scoped `socket_path()`.
- `predictd`: Unix-socket daemon, thread per connection, `newest_seen`
  generation filter, embedded sample-corpus model.
- `predict-cli`: `crossterm` terminal UI — live suggestions, Tab accepts,
  Enter commits, Esc quits, 300 ms daemon timeout.
- Verified live: typing shows suggestions with no perceptible lag;
  loopback e2e roundtrip ~1 ms (test budget 500 ms).
- Docs: `docs/adr/0003-m2-daemon-cli.md`, ARCHITECTURE status + results.

## [0.2.0] — M1 fast tier + eval harness

- `predict-core`: `Context`, `Candidate`, `ResolvedStyle`/`StyleSpec`
  (id placeholders), sync `Predictor::complete_word`, `rank_candidates`.
  `continue_text` deferred to M3.
- `predict-ngram`: lowercase unicode-alphanumeric tokenizer, `BTreeMap`
  prefix index, unigram/bigram/trigram counts with tier-weighted backoff,
  context-boosted completion (top 5), empty on sensitive/empty.
- `predict-eval`: replay harness (per-query top-1/top-3, optimal-accept
  keystroke savings, nearest-rank p50/p99) + `eval_sample` example.
- `corpora/sample_en_de.txt`: 353-token EN+DE sample corpus.
- Measured: top-1 0.897, top-3 0.997, savings 0.722, p50 0.087 ms /
  p99 0.181 ms (budget 5 ms).
- Docs: `docs/adr/0002-m1-fast-tier-eval.md`, ARCHITECTURE status + results.

## [0.1.0] — M0 skeleton

- Cargo workspace with nine crates: `predict-core`, `predict-ngram`,
  `predict-llm`, `predict-store`, `predict-proto`, `predict-eval`,
  `predictd`, `predict-cli`, `frontend-ibus` (empty placeholders).
- Conventions: edition 2021, `thiserror` in libraries, `anyhow` in binaries,
  `scripts/ci.sh` running `cargo build` + `cargo test` +
  `cargo clippy -- -D warnings`.
- Docs: `README.md`, `docs/ARCHITECTURE.md`, `docs/adr/0001-architecture.md`.
