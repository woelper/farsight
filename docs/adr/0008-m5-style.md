# ADR 0008: M5 prediction style

Status: accepted

## Context

M5 needs du/Sie modes, length policy, style resolution, per-style personal
data, and eval proof (0 violations, acceptance per style). Constraints from
the plan: TOML specs, explicit > inferred-sticky > per-app > global order,
simple du/Sie detection rules, decode-time bans (ambiguous `ihr`/`sie`
never banned), stop-criteria lengths, style-tagged storage with filtered
personal counts and retrieval.

## Decision

1. **Style types in `predict-core::style`, re-exported.** `StyleSpec`
   (id/language/address/length), `ResolvedStyle` (id + effective
   address/length), built-ins `default`/`du`/`sie` (the CLI cycles exactly
   these). Ban lists follow the plan verbatim: du-family case-insensitive,
   Sie-family case-sensitive only.
2. **No protocol change.** The `style_id` already flowing in
   `ContextUpdate` doubles as the explicit signal; `""` and `"default"`
   both mean unspecified (the frontend's resting state falls through to
   inference/defaults — otherwise the CLI's placeholder would permanently
   shadow inference). Replies echo the *resolved* id, which the CLI already
   displays.
3. **Sticky inference per connection, address-only.** First confident
   `detect_address` on field text wins and holds; it overrides just the
   address field of the per-app/global base style.
4. **Two-layer bans, both required for the 0 target.** Word tier: candidate
   filter in the daemon. Slow tier: single-token banned ids as `-inf`
   logit biases (bare + space-prefixed piece forms) *plus* a
   post-generation reject for multi-token sneaks. Either layer alone leaks
   (pieces merge unpredictably; filters alone abandon the plan's "in
   decoding").
5. **Length via stop mode; `word` disables the slow tier.** `Phrase`
   stops at `,;:` as well as sentence end (new `StopMode::Clause`);
   `Word` gets sentence-request silence from the daemon.
6. **Strict per-style personal data.** Counts and retrieval filter by the
   resolved id (cold start per style accepted); commits tag the requested
   id when known else `default`. The store gained a `style_id` column
   freely — M4 never shipped, so no migration exists.
7. **Corpus capitalization fix.** The sample corpus was all-lowercase, so
   capitalized-Sie detection found nothing; proper German capitals were
   restored (tokenizer-neutral, M1/M4 numbers untouched). Detection still
   needs standard capitalization — documented, not worked around.
8. **Marker paradigms completed.** `deine/meine/seine/keine/unser/ihre`
   inflections joined the language markers (exact-token counting needs
   them); `euch`/`uns` count for du-*detection* only, never bans.

## Consequences

- Measured (`eval_style`, 1.5B, 6 du + 5 sie sentences): **0 violations**,
  acceptance du 0.000 / sie 0.200, word top-1 ~0.73 both. Live CLI:
  Ctrl+S cycles `default → du → sie` with the resolved id in status.
- Tests: 4 (core lists/rules) + 7 (daemon: parse/resolve/sticky/filter/
  length-silence) + 3 (eval: word filter/violations/length-skip) + CLI
  cycle test; real-model ban-guarantee test is opt-in via
  `PREDICT_MODEL_PATH`.
- Open: explicit-default under a custom global is inexpressible (falls
  through); all-lowercase typing defeats Sie detection; per-style cold
  starts; GPU offload.
