# Changelog

All notable changes to `predict` are documented here, per milestone.

## [0.7.0] — M6 Linux desktop input

- `frontend-ibus`: IBus engine over D-Bus (`zbus` blocking API) that
  forwards typing context to `predictd` and renders suggestions as preedit
  ghost text with a lookup-table fallback. 10 ms predictd budget per
  keystroke; keys always pass through (only Tab-accept consumes).
  See `docs/adr/0009-m6-ibus.md`, install: `docs/INSTALL-linux.md`.
- Engine self-registers (`RegisterComponent`, payload in
  `frontend-ibus::component` with signature unit tests); the daemon
  instantiates through our factory on `SetGlobalEngine`. Finding: the
  daemon only activates catalog-known component names (unknown names stay
  inert), and this build scans only `/usr/share/ibus/component` — hence
  system-level install.
- Exact IBus wire tags (`IBusText`, attrs, lookup table) after the daemon
  failed closed on tagless structs; password/terminal silence via
  `Properties.Set(ContentType)`; settled text learned on focus-out.
- Tests: engine-direct e2e against a real daemon (`ibus_engine`) plus full
  client path on a private daemon (`ibus_mediated`: activation, ghost,
  commit, password silence, slow-daemon safety). GTK/Qt app check stays
  manual (documented steps).

## [0.6.0] — M5 style-aware prediction

- `predict-core::style`: `StyleSpec`/`ResolvedStyle`, `default`/`du`/`sie`
  builtins, `detect_address` (capitalized Sie), word-ban lists, violation
  checking. Resolution: explicit > sticky > per-app > global.
- Daemon: `StyleRegistry`, per-request style resolution, word-tier
  filtering, LLM logit-bias + post-check, `StopMode::Clause`.
- CLI: style cycling (Ctrl+S), style display.
- Eval: `eval_style` run clean — 0 violations (du accept 0.000, sie 0.200
  on the style corpus). See `docs/adr/0008-m5-style.md`.
- Honest generalization numbers on unseen novels (Pride and Prejudice,
  Werther): word top-1 ~0.2, personal-tier temporal deltas +397 EN / +812
  DE keystrokes. See `docs/NOVEL-EVAL.md`.

## [0.5.0] — M4 personal memory (plus unreleased CLI + base overhauls)

- `predict-cli` overhaul: bordered TUI (title, grey ghost sentence,
  highlighted top word, status line with word RTT, dimmed history/footer,
  width fitting), sentence prediction on every keystroke by default
  (`--no-sentence` opts out, `--help` documents keys), pure `draw_frame`
  covered by tests. See `docs/adr/0005-cli-overhaul.md`.
- Base-quality overhaul: real EN+DE training corpora (Gutenberg books, see
  `corpora/SOURCES.md`), one model per language with function-word
  detection (no more mixed-language suggestions; interim until M5),
  pre-sorted n-gram indexes (p99 back inside 5 ms on real vocab),
  `scripts/setup-model.sh` one-command LLM setup, `start.sh` warns when the
  slow tier is off, confidence gate re-tuned to −1.5.
  See `docs/adr/0006-base-quality.md`.
- `predict-store` (new, rusqlite bundled): settled documents (text, style
  id, language, timestamp), per-language uni/bi/tri counts, FTS5/BM25 index
  with sync triggers; `clear_all` for forget-all.
- `predict-ngram`: `PersonalCounts`/`PersonalBundle`, `BlendedPredictor`
  (`λ·p_base + (1−λ)·p_personal`, contextual + OOV personal only),
  `LangBlended` routing by current sentence (shared by daemon and eval);
  shared `tokenize_text` / `sentence_fragment` moved to `predict-core`.
- `predict-proto` v3: `CommitText{text, style_id, sensitive}`, `ForgetAll`,
  `SetLearning{enabled}`, `LearningState{enabled, documents}`.
- `predict-llm`: `build_grounded_prompt` for retrieval grounding.
- `predictd`: SQLite store (`[personal]`: db path, λ default 0.7, on by
  default), per-request bundle reloads, FTS5 retrieval refreshed at sentence
  boundaries only, acked commits/forget/pause, sensitive silence for both
  tiers, audited lock order.
- `predict-cli`: Enter commits settled text, Ctrl+P pauses/resumes,
  Ctrl+F twice forgets, status shows `learn on (N)` / `learn paused`.
- `predict-eval`: unified simulation with retrieval hook, temporal split
  (`temporal_split` + `evaluate_temporal`), `eval_temporal` example.
- Measured: +6 keystrokes on held-out text (256 vs 262); forget-all leaves
  0 documents / 0 counts / 0 FTS rows.
- Docs: `docs/adr/0007-m4-personal-memory.md`, ARCHITECTURE status + results.

## [0.4.0] — M3 slow tier (local LLM)

- `predict-llm`: sync `Backend` trait (`complete_sentence` + `CancelToken`),
  `LlamaBackend` on a worker thread (llama-cpp-2 0.1.156): greedy decoding,
  strip-and-constrain token healing, mean-logprob gate (default −1.0),
  consecutive-position KV reuse, `LlmConfig` from `[llm]` TOML,
  `StubBackend` for tests. Model: Qwen2.5-1.5B base Q4_K_M (gitignored).
- `predict-proto`: v2 (`PROTOCOL_VERSION = 2`) with `SuggestSentence` /
  `Sentence{generation, text, confidence}`.
- `predictd`: sentence worker threads with shared cancel state (stale work
  stays silent), write mutex, config `~/.config/predict/predictd.toml`,
  graceful degradation to word-only.
- `predict-cli`: 200 ms pause gating (`poll`), grey inline sentence,
  Ctrl+Right accepts, deadline reads that discard foreign frames.
- `predict-eval`: unified word+sentence simulation (`evaluate_combined`),
  acceptance proxy, wrong-suggestion rate, TTFT percentiles.
- Measured: 24/51 sentences accepted, wrong rate 0.40, TTFT p50 ~160 ms,
  combined savings 0.755 vs M1 0.722 (**+49 keystrokes**).
- Docs: `docs/adr/0004-m3-slow-tier.md`, ARCHITECTURE status + results.

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
