# ADR 0004: M3 slow tier (local LLM)

Status: accepted

## Context

M3 needs sentence continuation from a small local LLM: pause-gated,
cancelable, KV-cached, token-healed, confidence-gated, with a grey CLI
suggestion on a separate accept key — and eval must show it adds keystroke
savings over M1. Choices: bindings crate, model, sync-vs-async API shape,
healing mechanics, daemon threading, protocol evolution, and eval design.

## Decision

1. **`llama-cpp-2` 0.1.156, verified.** 2026-09-02, ~1.3M downloads,
   maintained by MarcusDunn; builds clean (cmake + g++, ~1 min on 20 cores).
   API note: `LlamaContext`/`LlamaBatch`/`LlamaSampler` are all `!Send` —
   every LLM object lives on one dedicated worker thread, so the sync
   `Backend` trait (below) was the natural shape, not an async stream.
2. **Sync `Backend` trait + `CancelToken`, not async.** The plan sketched
   `continue_text` as an async stream; with `std::thread` workers plus an
   `Arc<AtomicBool>` cancel flag we get the required semantics (pause-gated
   start, cancel on keystroke) with no runtime. `complete_sentence` returns
   `Ok(None)` for "nothing worth showing" and `Err(Cancelled)` for aborts;
   every worker reply resolves exactly once so callers never hang.
3. **Model: Qwen2.5-1.5B base Q4_K_M (941 MB).** The plan allows 0.5–1.5B
   base Q4. The 0.5B Q4_0 was functionally fine (1+1=2, conf −0.28) but its
   open-text distribution sat in a digit attractor ("the quick " →
   "20000000" at logit ~16, both quants) — too weak for continuations.
   1.5B completes stories ("Once upon a time…" → "10-year-old boy named
   Tom."), handles German ("Die Hauptstadt…" → " Paris ."), and drives the
   eval delta below. Path from `~/.config/predict/predictd.toml` (`[llm]`
   section, parsed by `LlmConfig::from_toml_str` via the new `toml` dep —
   M5 styles will reuse the pattern); any config/load failure disables the
   tier with a warning while the word tier keeps serving.
4. **Token healing by strip-and-constrain.** The fragment is stripped from
   the prompt (decoding restarts at a word boundary) and the first token is
   constrained to the top-200 logits whose piece starts with the fragment —
   including space-prefixed BPE pieces (`" brown"`, `"▁brown"`), which the
   first implementation missed, rejecting 51/51 eval triggers. This unifies
   mid-word and word-boundary prompts: at a boundary the model re-emits the
   known word ("the quick " + "brown" → strip → " fox jumps…"). No match in
   top-200 → `None` (correct: the model doesn't want to continue the word).
5. **Consecutive-positions cache protocol.** This llama.cpp rejects
   re-decoding a cached position (`Y = X + 1` required), so logits can never
   be "refreshed" in place. Every request rewinds to
   `min(common_prefix, P - 1)` and decodes from there — repeats re-decode 1
   token (verified deterministic across calls), appends decode only the
   suffix, divergent prompts recompute from the fork. `BOS`-first is
   enforced unconditionally so tokenization is stable across requests.
6. **Confidence = mean token logprob, gate default −1.0.** Greedy decoding
   (deterministic, reproducible evals). Stop at EOG / newline / sentence
   punctuation / 32 tokens. Threshold sweep on the sample corpus: −1.5 and
   −1.0 accept the same 24 sentences (+49 keystrokes), −1.0 shows 2 fewer
   wrong ones (16 vs 18) — hence the default.
7. **Protocol v2, silence as a signal.** `SuggestSentence{generation}` /
   `Sentence{generation, text, confidence}`; version bumped to 2. The daemon
   spawns one worker per sentence request sharing an `Arc<Mutex<…>>`
   connection state plus a write mutex (frames never interleave); the worker
   replies only while its generation is still current, else silence
   (gated, healed-away, cancelled, stale, or tier disabled — the CLI treats
   all uniformly via timeouts). Cancel/newer-generation aborts via the
   shared token; verified deterministically with a blocking test backend.
8. **CLI: `poll(200 ms)` pause gating, grey inline suggestion, Ctrl+Right
   accept.** Sentence requests reuse the word request's generation (they
   refine, not supersede). Reads use deadlines and discard foreign frames,
   so a late slow reply can never desynchronize the word path.
9. **Eval: unified simulation with an acceptance proxy.** Once per sentence
   (after 3 words) the slow tier may consume upcoming words at one accept
   key; consumed words leave the word-tier counters, so the delta vs the
   word-only run on the same corpus is honest. Reports acceptance rate,
   wrong-suggestion rate, TTFT percentiles.

## Consequences

- Measured (`eval_sentence`, 1.5B Q4_K_M, threshold −1.0, 51 sentences):
  42 shown, 24 accepted (rate 0.471), wrong rate 0.400 (16/40), 70 words /
  332 chars at 24 accept keys, TTFT p50 ~160 ms / p99 ~2 s (grows with
  history length as the 1024-char window slides). Combined savings 0.755
  vs M1 0.722: **+49 keystrokes — the M3 done-criterion.**
- Live run (real daemon + CLI under a pty): typing `the quick brown`,
  pausing, Ctrl+Right, Enter produced
  `committed: the quick brown fox jumps over the lazy dog`.
- Tests: 19 (llm: trait/stub/config/healing/UTF-8 + opt-in real-model
  smoke) + 6 new (eval: proxy math, gate, caps) + 7 new (daemon: reply,
  gate/cancel silence, config fallbacks) + 6 new (CLI: read-until framing,
  sentence accept). `cargo test` stays fast — the model only loads with
  `PREDICT_MODEL_PATH` set or in examples.
- Scores remain optimistic (train text = test text; M4's temporal split is
  the honest benchmark) and the wrong rate (0.40) is high — both recorded
  as open, not hidden.
- `models/*.gguf` is gitignored (941 MB stays local); `toml`/`serde`
  added to `predict-llm`; `predict-eval` and `predictd` depend on
  `predict-llm`; CLI dropped nothing, added no deps.
- Open: recurrent-state models (no rewind), smarter prompt windowing for
  TTFT, wrong-rate reduction (reranking, better gate features), streaming
  tokens to the UI, GPU offload.
