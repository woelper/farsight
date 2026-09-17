# ADR 0007: M4 personal memory

Status: accepted

## Context

M4 needs the assistant to follow the user's own writing: settled text in
SQLite, personal n-gram counts blended at decode (`p = λ·p_llm +
(1−λ)·p_personal`), FTS5 retrieval grounding the slow tier, forget-all and
pause-learning commands, and a temporal-split eval proving gains. It also
absorbed the base-quality overhaul (ADR 0006) mid-flight, which changed
several assumptions underneath.

## Decision

1. **`rusqlite` 0.40.2 with bundled SQLite (verified 2026-08, ~107M
   downloads).** FTS5 + `unicode61` confirmed by test, not by docs.
   Schema: `documents` (text, style_id for M5, lang, timestamp),
   `counts(lang, n, ctx, word)` (per-language uni/bi/tri), and an FTS5
   table with content-sync triggers. Unknown-language commits store the
   document but count to both sides — marker-less personal vocabulary
   ("grilled courgette") would otherwise never be learned.
2. **Settled text only, via explicit `CommitText`.** The daemon never
   stores `ContextUpdate` keystrokes by construction; sensitive commits are
   dropped at the protocol handler (unit + e2e tested), as are sensitive
   sentence requests (an M3 gap this milestone closed). Commits are acked
   with `LearningState` so frontends show live store state.
3. **Blend at the word tier; retrieval grounds the slow tier.** Full
   logit-level blending of the LLM was judged disproportionate: the word
   tier blends exact count-probabilities, and the slow tier is personalized
   by top-3 BM25 snippets in its prompt (plus shared personal vocabulary
   through the same distributions). `LangBlended` (per-language bases +
   bundle, borrowed, no per-query clones) is shared by daemon and eval.
4. **Personal distribution is contextual + OOV only — no blind unigram
   backoff.** First implementation blended raw personal unigram mass and
   lost 24 keystrokes on held-out text: frequent train words outranked good
   base picks everywhere. Personal now contributes seen trigram/bigram
   conditionals plus unigram mass solely for words the base never saw (the
   only way user-invented words surface). With no contextual or OOV
   evidence the blend provably preserves base ranking.
5. **Routing follows the current sentence, not cumulative history.**
   Detecting on full field text let earlier sentences outvote the live one
   on mixed text (measured −36 with the stub tier). `LangBlended` routes on
   the sentence fragment; the eval replays history with terminators (the
   model tokenizes both forms identically) so harness and daemon agree.
6. **Protocol v3**: `CommitText{text, style_id, sensitive}`,
   `ForgetAll`, `SetLearning{enabled}`, `LearningState{enabled,
   documents}`. Retrieval refreshes at sentence boundaries only
   (terminator-count change or non-extension). Pause is a daemon-global
   atomic; every control command is acked. Lock order is connection →
   store → counts, never reversed (audited; commits/reloads never nest
   against the grain).
7. **Shared tokenizer in `predict-core`.** Third use of the same rule
   (fast tier, eval, store) forced centralization; `sentence_fragment`
   joined it for routing/retrieval.
8. **λ = 0.7 default.** The stub-temporal sweep was flat across
   0.5/0.7/0.9 (word side neutral once V2 removed the noise); 0.7 keeps
   strong contextual boosts with base dominance elsewhere.

## Consequences

- Measured (`eval_temporal`, 1.5B, 85 held-out words): baseline 262
  keystrokes (0.262) vs personal 256 (0.279): **+6 — the M4
  done-criterion.** Small because the sample corpus barely repeats across
  the split (8 shared bigrams); word-side stub runs read exactly +0, so
  the +6 is retrieval grounding + blend interaction on the sentence tier.
- Live run: commit → `learn on (1)` with `courgette` topping `grilled `;
  Ctrl+P → `learn paused` (commits ignored, proven by doc count);
  Ctrl+F ×2 → 0 documents / 0 counts / 0 FTS rows on disk.
- Tests: 9 (store, in-memory: counts sides, BM25 quoting, reload
  round-trip, forget completeness), 7 (daemon: boost/forget/pause/
  sensitive×2/retrieval-grounding-via-echo/config), 6 (eval temporal +
  hook), 4 (CLI control replies + status). `cargo test` needs no model.
- Personal data lives in `~/.local/share/predict/predict.db` (XDG-aware),
  learning on by default, `models/*.gguf` still ignored.
- Open: per-app/style personalization (M5 tags the stored style ids
  already), wrong-rate work, OOV mass diluting at scale (count-gated
  unigram is the documented fallback), GPU offload.
