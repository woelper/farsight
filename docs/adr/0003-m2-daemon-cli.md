# ADR 0003: M2 daemon protocol and test client

Status: accepted

## Context

M2 needs a per-user daemon on a Unix socket speaking the plan's messages
(`ContextUpdate`, `SuggestRequest { generation }`, `Suggestion`,
`Cancel { generation }`, newer generation cancels older work) plus a terminal
UI where typing shows live word suggestions and Tab accepts. Choices: IPC
serialization, framing/versioning, daemon concurrency model, terminal-input
crate, and what model the daemon serves before the personal store (M4).

## Decision

1. **`postcard` 1.1.3 + `serde` 1.x for IPC.** Verified: postcard 1.1.3
   (2025-07-24, ~62M downloads, maintained by James Munns), `cargo search`
   + full build green. Compact binary, no schema compiler, `no_std`
   friendly — fits versioned structs. One catch found at build time:
   `to_stdvec` needs the non-default `use-std` feature, now enabled
   explicitly. Frames are `u32`-LE length prefix + postcard body, capped at
   256 KiB (`FrameTooLarge` rejected before allocating).
2. **Version envelope on every frame.** Each payload is
   `{ version: u16, msg }`; readers reject mismatches with
   `VersionMismatch` (tested). `PROTOCOL_VERSION = 1`; any breaking change
   bumps it. No handshake roundtrip — every frame is self-describing.
3. **Socket path is user-scoped.** `$XDG_RUNTIME_DIR/predict/predictd.sock`,
   fallback `temp_dir()/predictd-$USER.sock` so two users never share a
   socket (tested for both shapes).
4. **Generation-cancel rule, enforced on both sides.** Per connection the
   daemon keeps `newest_seen`; `Suggest`/`Cancel` with an older generation
   are ignored (stale `Suggest` gets no reply). The CLI bumps the generation
   per keystroke and ignores replies for old generations; daemon reads time
   out after 300 ms and render with no suggestions instead of stalling, so a
   slow daemon never blocks typing. `Cancel` is recorded already in M2 (it
   only pays off with the M3 slow tier).
5. **`std::thread` per connection, no async runtime.** The work is
   synchronous and sub-millisecond (model p99 0.18 ms from M1); Tokio would
   be abstraction without a second use case. Revisited if M3's streaming
   LLM needs it.
6. **`crossterm` 0.29.0 for the terminal UI.** Verified: 0.29.0 (2025-04-05,
   ~191M downloads, maintained), builds clean. Raw mode + alternate screen
   behind a drop guard (restored on error paths too), `Tab` accepts the top
   word (fragment replaced + space appended), `Enter` commits the line,
   `Esc`/`Ctrl-C` quits. CLI speaks only `predict-proto` types, so its
   `predict-core` dependency was removed.
7. **Daemon serves the embedded sample corpus.** `NgramModel` trained at
   startup from `corpora/sample_en_de.txt` via `include_str!`. Placeholder
   until M4 (store) + M3 (LLM); the `suggest_for` seam keeps the swap local.

## Consequences

- Measured live (real `predictd` binary + `predict-cli` under a pty):
  typing `the qui` renders `suggestions [default]: quick` per keystroke with
  no perceptible lag; Tab → `the quick ` with next-word suggestions; Enter
  commits; Esc quits cleanly (exit 0). Daemon loopback e2e test asserts
  roundtrip < 500 ms (actual ~1 ms).
- Tests: 8 (proto: roundtrips, multi-frame stream, version rejection,
  oversize/truncation, socket path) + 4 (daemon: reply shape, sensitive,
  generation filter, socket e2e incl. stale-gets-no-reply) + 4 (CLI: accept
  mid-word/boundary/unicode, context fields). No `unwrap()` outside tests.
- Test-harness notes (pty, not product bugs): bytes piped into `script`
  before the child enables raw mode pass through the canonical line
  discipline (`\r`→`\n` via ICRNL); delay input ~2 s after spawning. Real
  Enter sends CR, which maps to `KeyCode::Enter`; a literal LF parses as
  Ctrl+J per crossterm's control mapping.
- Open: multi-client contention (fine — threads + clone-per-connection),
  single-instance lock (currently unlink-then-bind), slow-tier streaming
  (M3), personal model (M4).
