# ARCHITECTURE

Status: M3 done. Slow LLM tier works end to end (daemon + CLI + eval); store/style/IBus are still placeholders.

## Overview

`predict` is a local typing assistant for Linux. A per-user daemon (`predictd`)
owns prediction; frontends (starting with IBus) send typing context and render
suggestions. A terminal client (`predict-cli`) exists for manual testing.

```text
+----------------+     Unix socket      +------------------+
| frontend-ibus  | <------------------> |     predictd     |
| predict-cli    |   predict-proto      |  core+ngram+llm  |
+----------------+   (postcard)         |  +store          |
                                       +------------------+
```

No network code in any crate (non-negotiable).

## Crates

- **predict-core**: shared types (`Context`, `Candidate`, `StyleSpec` /
  `ResolvedStyle` as id-only placeholders), sync `Predictor` trait
  (`complete_word`; `continue_text` deferred to M3), `rank_candidates`. No I/O.
- **predict-ngram**: fast tier (M1 done). Lowercase unicode-alphanumeric
  tokens; `BTreeMap` prefix index (range scan, not a separate trie);
  unigram/bigram/trigram counts with tier-weighted backoff (tri 1e6 / bi 1e3 /
  uni); mid-word completion with bigram context boost; empty vec for
  sensitive contexts and empty models.
- **predict-llm**: slow tier (M3 done). Sync `Backend` trait
  (`complete_sentence` + `CancelToken`; `Ok(None)` = nothing worth showing),
  `LlamaBackend` on a dedicated worker thread (llama.cpp context is `!Send`):
  greedy decoding, strip-and-constrain token healing (space-prefixed pieces
  included), mean-logprob confidence gate (default −1.0), consecutive-position
  KV reuse (repeats re-decode 1 token, deterministic). `LlmConfig` from
  `[llm]` TOML; `StubBackend` for tests. Model: Qwen2.5-1.5B base Q4_K_M
  (local GGUF, gitignored).
- **predict-store**: SQLite personal memory (from M4). Stores settled text only
  (after pause / field leave), never keystrokes, never when `sensitive`.
  Personal n-gram counts, FTS5 retrieval (top 3, sentence boundaries only).
- **predict-proto**: versioned IPC (M2 done, v2 in M3). `PROTOCOL_VERSION = 2`
  envelope on every frame, `postcard` body with `u32`-LE length prefix
  (256 KiB cap): `ContextUpdate`, `SuggestRequest { generation }`,
  `Suggestion`, `Cancel { generation }`, plus `SuggestSentence` / `Sentence`
  (slow tier; silence = stale/cancelled/gated/disabled). User-scoped socket
  path (`$XDG_RUNTIME_DIR/...`, else `predictd-$USER.sock`).
  Plus `ForgetAll` / `PauseLearning` (M4).
  Stays free of `predict-core` so frontends stay light.
- **predict-eval**: offline replay harness (M1 done, sentence tier M3 done).
  Word tier: single-space reconstruction of typing, one `complete_word`
  query per prefix length (`k = 0` = word boundary); per-query top-1/top-3,
  optimal-accept keystroke savings (`k + 1` vs word length), nearest-rank
  latency p50/p99. Sentence tier: one slow trigger per sentence (after 3
  words), acceptance proxy (common word-prefix), wrong-suggestion rate,
  TTFT percentiles, honest delta vs word-only. Sample corpus
  `corpora/sample_en_de.txt` (353 tokens, EN + DE).
  Later: temporal split (M4), du/Sie violation rate (M5).
- **predictd**: per-user daemon (M2 done, slow path M3 done). Unix-socket
  listener, one `std::thread` per connection, `newest_seen` generation
  filter (stale work gets no reply). Sentence requests run on worker threads
  against the optional LLM backend (config `~/.config/predict/predictd.toml`);
  replies only while the generation is current. Serves the embedded
  sample-corpus n-gram model until M4.
- **predict-cli**: terminal test client (`crossterm`; M2 done, sentence UI
  M3 done, overhauled post-M3 — see ADR 0005). Bordered TUI: grey ghost
  sentence, highlighted top word, status line with word RTT, dimmed history.
  Live word suggestions per keystroke, Tab accepts the top word, Enter
  commits, Esc quits. Sentence prediction runs on every keystroke by default
  (`--no-sentence` opts out), shown grey inline, Ctrl+Right accepts. Reads
  use deadlines and discard foreign frames — a slow daemon never stalls
  typing. Later: style display.
- **frontend-ibus**: IBus engine in Rust over D-Bus (`zbus`) (M6). Surrounding
  text → `ContextUpdate`; preedit / lookup-table rendering; sensitive content
  types → no suggestions, no learning; 10 ms daemon timeout, never blocks typing.

## Key flows (planned)

- **Word suggestion (M1–M2, working)**: keystroke → CLI sends `ContextUpdate` +
  `SuggestRequest{generation}` → daemon `complete_word` (fast tier) →
  `Suggestion` → CLI renders. Stale generations get no reply on either side.
- **Sentence suggestion (M3, working)**: typing pause (200 ms) → CLI sends
  `SuggestSentence{generation}` (same generation: it refines, not
  supersedes) → daemon worker runs the LLM with the shared cancel token +
  KV reuse + token healing → replies `Sentence` only while the generation
  is current and confidence ≥ gate → CLI renders grey inline, Ctrl+Right
  accepts. Cancel/newer generation aborts silently.
- **Learning (M4)**: settled text commit → `predict-store` (tagged by style
  from M5) → personal counts + FTS index. Blend at decode:
  `p = λ·p_llm + (1−λ)·p_personal`.
- **Style (M5)**: TOML `StyleSpec` (language, du/sie/none, word/phrase/sentence).
  Resolution: explicit > inferred-sticky > per-app > global. Decoding bans
  (du↔Sie families), length via stop criteria.
- **IBus (M6)**: engine requests surrounding text, maps password/sensitive
  content types to `Context.sensitive = true`.

## M3 results

`eval_sentence` (Qwen2.5-1.5B base Q4_K_M, gate −1.0, 51 sentences): 42
shown, 24 accepted (rate 0.471), wrong rate 0.400 (16/40), 70 words /
332 chars at 24 accept keys, TTFT p50 ~160 ms / p99 ~2 s. Combined savings
0.755 vs M1 0.722: **+49 keystrokes — the slow tier pays off.** Same
optimism caveat as M1 (train text = test text). Live run (real daemon +
CLI under a pty): typing `the quick brown`, pausing, Ctrl+Right, Enter
produced `committed: the quick brown fox jumps over the lazy dog`.

## M2 results

Live run (real `predictd` + `predict-cli` under a pty): typing `the qui`
renders `suggestions [default]: quick` per keystroke with no perceptible
lag; Tab → `the quick `; Enter commits; Esc quits (exit 0). Daemon loopback
e2e test budgets roundtrip < 500 ms (actual ~1 ms).

## M1 results

`cargo run -p predict-eval --example eval_sample` (train + replay on
`corpora/sample_en_de.txt`): 353 words / 1462 queries, top-1 0.897, top-3
0.997, keystroke savings 0.722, latency p50 0.087 ms / p99 0.181 ms
(budget 5 ms). Scores are optimistic — same text trains and tests;
M4's temporal split is the honest benchmark.

## Non-negotiables

- No network code in any crate.
- No learning from sensitive fields.
- Frontend never blocks typing (10 ms daemon timeout in IBus).
- Every milestone documented (ADR + this file + CHANGELOG in the same commit).

See `docs/adr/0001-architecture.md` for the M0 decisions,
`docs/adr/0002-m1-fast-tier-eval.md` for the M1 design,
`docs/adr/0003-m2-daemon-cli.md` for the M2 design, and
`docs/adr/0004-m3-slow-tier.md` for the M3 design.
