# ADR 0002: M1 fast tier and eval harness

Status: accepted

## Context

M1 must deliver word completion from a prefix and next-word prediction from a
base n-gram model (EN + DE), plus an offline replay harness reporting
keystroke savings, top-1/top-3 hit rates, and latency p50/p99 — with
`complete_word` p99 < 5 ms. The plan's core-types sketch includes an async
`continue_text` (slow tier, M3); pulling it into the trait now would force
every M1 test and the eval harness to deal with streams and cancel tokens.

## Decision

1. **Core stays sync-only in M1.** `predict-core` ships `Context`, `Candidate`,
   `ResolvedStyle` (id only), `StyleSpec` (id + language placeholders),
   `rank_candidates` (score desc, text tiebreak), and
   `Predictor { fn complete_word }`. `continue_text` is deferred to M3, when
   the `Backend` trait and cancel semantics land. `complete_word` returns an
   empty vec for "no suggestion", including all `sensitive` contexts
   (non-negotiable: no suggestions, and later no learning, for sensitive
   fields).

2. **Tokenizer: lowercase unicode-alphanumeric split.** `is_alphanumeric`
   boundaries, `to_lowercase` per char. One rule shared by the n-gram tier
   and the eval harness (duplicated 15-line function, no shared dep — eval
   stays generic over `Predictor`). Consequence: case and punctuation carry
   no signal; German `ß`/`ü` survive correctly (`"Grüße"` → `"grüße"`).

3. **Prefix index is a `BTreeMap` range scan, not a trie.** The plan says
   "trie + n-gram"; a `BTreeMap<String, u64>` with `range(prefix..)` +
   `take_while(starts_with)` is a sorted-string trie in one std type, zero
   deps, and measured p99 0.18 ms on the sample corpus — 27x inside budget.
   A dedicated trie becomes justified only with evidence the scan is too
   slow (vocab ~148 words here; revisit past ~100k).

4. **N-gram: unigram + bigram + trigram with tier-weighted backoff.**
   `predict_next` scores trigram `1e6 + count`, bigram `1e3 + count`,
   unigram `count`, so longer context always outranks raw frequency (a naive
   count ranking let unigram `"the"` beat bigram `"quick"` after `"the "` —
   caught by a test). Mid-word completion over-fetches 20 prefix matches and
   re-ranks with a `+2 × bigram_count` context boost. `MAX_CANDIDATES = 5`;
   hit rates use top-1/top-3 within those 5.

5. **Eval simulation (single-space reconstruction, accept-cost-1).**
   The corpus is re-tokenized; each query's `before` is prior words joined
   with single spaces plus the current prefix (`k = 0` is the word boundary,
   testing next-word prediction). Hit rates are per query. Keystroke savings
   assume optimal accept: shortest prefix with top-1 == target costs `k + 1`
   (prefix + accept key), else full length. Latency percentiles are
   nearest-rank over per-query `Instant` measurements. `Display` prints the
   report; `evaluate_file` loads a corpus path.

6. **Sample corpus replays its own training text.**
   `corpora/sample_en_de.txt` (353 tokens, 148 words, EN + DE, deliberately
   repetitive) is used for both training and replay. Scores are therefore
   optimistic by construction (top-1 0.897, savings 0.722) — they prove the
   harness and the tier work, not generalization. M4's temporal split
   (train first 75%, test last 25%) is the honest benchmark.

## Consequences

- Measured on the sample corpus (`cargo run -p predict-eval --example
  eval_sample`): 353 words / 1462 queries, top-1 0.897, top-3 0.997,
  keystroke savings 0.722, latency p50 0.087 ms / p99 0.181 ms. M1 done
  criteria (eval runs, p99 < 5 ms) met with large margin.
- Crate deps added: none beyond M0 (`thiserror` in libs). `predict-eval`
  takes `predict-ngram` as a dev-dependency only (example + end-to-end
  test); the harness itself depends only on `predict-core`.
- Tests: 4 (core) + 12 (ngram, incl. a 1200-query p99 budget test) + 7
  (eval, incl. train-and-replay end-to-end asserting p99 < 5 ms and
  savings/top-1 > 0). No `unwrap()` outside `#[cfg(test)]`.
- Open: honest generalization numbers (M4 temporal split), case/punctuation
  fidelity, vocab scaling past thousands of words.
