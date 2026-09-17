# ADR 0006: Base-quality overhaul (real corpora, language split, setup)

Status: accepted

## Context

Testing the CLI showed two product-level problems: (1) no sentence
prediction at all without a hand-built model config the tooling never
mentioned, and (2) word prediction trained on a 353-token EN+DE toy corpus —
mostly empty, and prefixes like `dan…` completed into the wrong language.
A new p99 test on real-sized data also exposed a latency regression
(boundary queries hit 5.9 ms). Separately, M4 groundwork (shared tokenizer,
personal-count blending machinery) had landed half-finished in the tree.

## Decision

1. **Real base corpora, split by language.** `corpora/base_en.txt` (Alice +
   Holmes, ~133k tokens) and `corpora/base_de.txt` (Nietzsche's Zarathustra,
   ~85k tokens), public-domain Gutenberg texts, headers stripped, sources in
   `corpora/SOURCES.md`. The daemon trains one `NgramModel` per language
   (0.6 s at startup) and `sample_en_de.txt` stays a fast eval-regression
   corpus only.
2. **Minimal language detection for model selection (interim until M5).**
   `detect_language` counts unambiguous function words (cross-language
   words like `was`/`her`/`will`/`in` are in NEITHER list on purpose).
   English/German context queries only that model; `Unknown` (first
   keystroke, ties) shows the more confident model only — top pick by
   unigram probability, English-first tie-break — never a mix. An earlier
   reciprocal-rank merge was replaced: it still displayed both languages,
   which was the complaint. M5 (explicit/style/app defaults) supersedes
   this heuristic.
3. **Pre-sorted n-gram indexes.** `predict_next` scanned whole tables per
   query (2–5 ms on real vocab); per-tier sorted vectors built once at
   train time restored p99 (boundary queries now ~0.1 ms). Personal counts
   keep scans (small data; indexed if it ever matters).
4. **One-command model setup + visible status.** `scripts/setup-model.sh`
   downloads Qwen2.5-1.5B Q4_K_M to XDG data and writes `predictd.toml`
   (backs up existing configs); `scripts/start.sh` prints the daemon's LLM
   line and warns with the setup hint when the tier is off.
5. **Gate re-tuned to −1.5.** A perfect continuation (`Sherlock Hol` →
   `…mes was a famous detective…`, conf −1.464) was silenced at −1.0; −1.5
   keeps the M3 eval delta identical (+49) while showing strictly more on
   real text. Mid-word healing drags the mean down (the forced first token
   is often low-probability) — a better gate feature is future work.
6. **What was deliberately NOT changed.** A temperature/penalty sampling
   trial produced odd corners (`buy some ` → test-question text) with no
   clear win over greedy, so greedy stays (deterministic, measured).
   Digit-heavy completions on some open prompts are small-base-model
   priors (font specimens, trivia populations), not pipeline bugs — the
   pipeline is proven by arithmetic, stories, and healed proper nouns.
   Full logit-level personal blending stays an M4 item; the word-tier blend
   machinery already in the tree is staged groundwork, unused by the daemon
   so far.

## Consequences

- Typing English offers English words (`Sherlock Hol` → `hold holmes
  holder…`, no German); German likewise (`Also sprach Zar` →
  `zarathustra…`). First-keystroke ties resolve to one language, never a mix.
- Fresh `./scripts/setup-model.sh && ./scripts/start.sh` gives sentences;
  live pty run showed the grey `Holmes was a famous detective…` accepted
  via Ctrl+Right.
- Tests: language markers, per-language separation, one-language Unknown,
  real-corpus p99, blend reordering (unchanged behavior when personal is
  empty). `cargo test` stays fast (unit corpora are inline; 700 KB/500 KB
  files load only in the real-corpus p99 test, ~1 s).
- `models/*.gguf` was already ignored; the 0.5B experiment files were
  removed (1.5B is the model). Gutenberg books (~1.2 MB text) ship in git.
- Open: M4 personal memory + temporal eval (the staged blend plugs in
  there), M5 language/style defaults replacing the heuristic, gate
  features, GPU offload.
