# ADR 0005: CLI overhaul (styled TUI, sentences by default)

Status: accepted

## Context

The M2/M3 `predict-cli` was a bare alternate-screen dump, and sentence
prediction only ran after a 200 ms pause — easy to miss, and the first
impression of the whole project. The ask: make it look decent and predict
sentences by default.

## Decision

1. **Polish in `crossterm`, no TUI framework.** `ratatui` would add a heavy
   dependency for a test client. Instead: a bordered input box, cyan title,
   bold buffer, dark-grey ghost sentence, reverse-video top word pick,
   dimmed status/history/footer, terminal-width fitting with `…`
   truncation, `\r\n` line endings (raw mode has no ONLCR — the old
   `writeln!` output staircased on strict terminals). Layout math runs on
   plain text and styles wrap fitted segments, so ANSI codes can never be
   cut or miscounted; `draw_frame` is a pure function with content/width
   tests.
2. **Sentence request on every refresh (default on, `--no-sentence` opts
   out).** The pause-only trigger meant a sentence appeared only after an
   idle window; now each keystroke sends `Suggest` + `SuggestSentence` on
   the same generation (refines, not supersedes) and cancels the previous
   one. In-flight slow work dies at the next keystroke via the existing
   generation-cancel protocol, so while typing fast nothing completes and
   the moment the user pauses the latest request finishes — the pause
   behavior emerges instead of being special-cased. The idle branch stays
   as a collect-only backstop. Cost: one worker spawn + cancelled fast-fail
   per keystroke; the LLM backend serializes those behind its pre-check.
3. **Status line shows what matters for a test client:** style id,
   generation, last word round-trip ms (proves "no visible lag"), and
   sentence state (`ready` / `…` pending / `off` / `none`).

## Consequences

- Live run (pty, 100 cols): bordered box, `> the quick brown` + grey
  `fox jumps…`, highlighted top word, `word 0.3ms` status; Ctrl+Right
  committed the full sentence. `--help` documents keys and flags.
- Tests: 17 in the CLI (arg parsing, pure `draw_frame` content/width/
  truncation/empty/pending cases, framing read-until cases, accept fns).
- `render` params bundled into `Frame`/`SentenceUi` for the
  `too_many_arguments` lint; no new dependencies.
- Open: no animation (render is event-driven), no mouse, 24-col minimum
  width degrades gracefully.
