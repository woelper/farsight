# Project: predict (local word + sentence prediction for Linux)

## Goal
A local, private typing assistant for Linux. It suggests the next word and the rest
of the sentence while the user types in any app. Suggestions follow the user's own
writing and an active style (formal/casual, German du/Sie, short/long).
Everything runs locally. No network access anywhere.

## Scope for this plan
In: Rust workspace, prediction engine, personal memory, style modes, daemon,
a terminal test client, an IBus engine as the first real frontend.
Out (do not build, but do not block): macOS, Windows, Fcitx5, fine-tuning,
text rewriting, settings GUI.

## Engineering rules
- Rust stable, 2021 edition or newer. Cargo workspace.
- Simple over clever. No abstraction without a second use case, except the
  frontend/daemon boundary and the LLM backend trait (both listed below).
- Every milestone ends green: `cargo build`, `cargo test`, `cargo clippy -- -D warnings`.
- Document every decision in `docs/adr/NNNN-title.md` (context, decision, consequences).
- Keep `docs/ARCHITECTURE.md` and `CHANGELOG.md` current in the same commit as the change.
- No `unwrap()` outside tests. Errors via `thiserror` in libraries, `anyhow` in binaries.
- Work one milestone at a time. Stop after each and summarize: what was built,
  what was decided, what is open. Do not start the next milestone unasked.
- If a crate choice is uncertain, verify it builds and is maintained before using it,
  and record the choice in an ADR.

## Workspace layout
crates/
  predict-core    # types: Context, Candidate, StyleSpec; Predictor trait; ranking
  predict-ngram   # fast tier: trie + n-gram, personal counts
  predict-llm     # slow tier: Backend trait, llama.cpp implementation
  predict-store   # SQLite store for personal data
  predict-proto   # IPC messages, versioned, serialized with postcard
  predict-eval    # offline replay harness and metrics
  predictd        # daemon binary
  predict-cli     # terminal client for manual testing
  frontend-ibus   # IBus engine over D-Bus (zbus)

## Core types (starting point, adjust if needed, record changes in an ADR)
pub struct Context {
    pub app_id: String,
    pub before: String,        // text before cursor
    pub after: String,         // text after cursor, may be empty
    pub sensitive: bool,       // password or similar field
    pub style: ResolvedStyle,
}

pub trait Predictor: Send + Sync {
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate>;       // sync, p99 < 5 ms
    fn continue_text(&self, ctx: &Context, cancel: CancelToken)     // async, cancelable
        -> impl Stream<Item = Candidate>;
}

## Milestones

### M0: Skeleton
- Workspace, empty crates, CI script running build/test/clippy.
- README, docs/ARCHITECTURE.md, docs/adr/0001-architecture.md, CHANGELOG.md.
Done when: workspace builds clean and docs describe the layout above.

### M1: Fast tier + eval harness
- predict-ngram: word completion from a prefix, next-word prediction from a
  base n-gram model built from a plain text corpus (English + German).
- predict-eval: replays a text file as simulated typing and reports
  keystroke savings rate, top-1 / top-3 hit rate, latency p50/p99.
Done when: eval runs on a sample corpus and complete_word p99 < 5 ms.

### M2: Daemon + test client
- predictd: per-user daemon on a Unix socket. Messages in predict-proto:
  ContextUpdate, SuggestRequest { generation }, Suggestion, Cancel { generation }.
  A newer generation cancels older work.
- predict-cli: terminal UI; type text, see word suggestions live,
  Tab accepts a word.
Done when: typing in predict-cli shows suggestions with no visible lag.

### M3: Slow tier (small local LLM)
- predict-llm: Backend trait plus a llama.cpp-based implementation.
  Model: small base (not chat) GGUF, 0.5 to 1.5B params, Q4. Path from config.
- Sentence continuation starts after a typing pause (default 200 ms),
  is cancelled on the next keystroke, reuses KV cache for a shared prefix.
- Token healing: if the user is mid-word, the output must continue that word.
- Confidence gate: only show a sentence suggestion above a configurable threshold.
- predict-cli: grey sentence suggestion, a separate key accepts the whole sentence.
- Extend eval: sentence acceptance proxy, wrong-suggestion rate, time to first token.
Done when: eval shows the LLM tier adds keystroke savings over M1 on the sample corpus.

### M4: Personal memory
- predict-store: SQLite. Stores settled text (text after the user pauses or leaves
  the field), never keystrokes. Never stores anything when Context.sensitive is true.
- Personal n-gram counts updated on each settled commit.
- Blend at decode time: p = lambda * p_llm + (1 - lambda) * p_personal.
  lambda configurable.
- Retrieval: SQLite FTS5 (BM25) over stored text, top 3 snippets go into the prompt,
  refreshed at sentence boundaries only.
- "Forget all" and "pause learning" commands in predict-proto and predict-cli.
- Extend eval: temporal split (learn from first 75% of a corpus, test on last 25%).
Done when: temporal-split eval beats M3, and forget-all removes all personal data.

### M5: Prediction style
- StyleSpec loaded from a TOML file: id, language, address_form (du/sie/none),
  length (word/phrase/sentence).
- Style resolution order: explicit mode set by the user > style inferred from
  text already in the field (sticky once set) > per-app default > global default.
- du/Sie detection by simple rules on preceding text.
- Sie mode: forbid du-family tokens (du, dich, dir, dein*) in decoding.
  du mode: the mirror. Ambiguous words (ihr, sie) are not banned.
- Length policy via stop criteria (clause end vs sentence end).
- Tag every stored text with the active style. Personal n-grams and retrieval
  filter by the active style.
- Suggestions carry the active style id so the UI can show it.
- Extend eval: du/Sie violation rate (target 0), acceptance per style.
Done when: du/Sie violation rate is 0 on the eval set and predict-cli shows
the active style.

### M6: IBus frontend
- frontend-ibus: IBus engine process in Rust over D-Bus (zbus).
- Requests surrounding text capability, sends ContextUpdate to predictd.
- Shows suggestions as preedit text, falls back to the lookup table.
- Treats password/sensitive content types as sensitive (no suggestions, no learning).
- Hard rule: if predictd does not answer within 10 ms, pass the key through
  and show nothing. The engine must never slow typing down.
- Install instructions in docs/INSTALL-linux.md.
Done when: suggestions work in a GTK app (e.g. gedit) and a Qt app, and typing
in a password field produces no suggestions and no stored data.

## Non-negotiables
- No network code in any crate.
- No learning from sensitive fields.
- Frontend never blocks typing.
- Every milestone documented (ADR + ARCHITECTURE + CHANGELOG).
