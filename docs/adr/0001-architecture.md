# ADR 0001: Workspace architecture and M0 skeleton

Status: accepted

## Context

We need a starting point for `predict` (see `plan.md`): a local-only Linux
typing assistant with a fast n-gram tier, a slow LLM tier, personal memory,
style modes, a per-user daemon, a terminal test client, and an IBus frontend.
M0 must produce a green workspace (build/test/clippy) plus README,
ARCHITECTURE, and CHANGELOG, without implementing behavior.

Choices to lock in M0:

1. Language/edition and workspace shape.
2. Crate boundaries and dependency direction.
3. IPC serialization and daemon transport.
4. Error-handling and lint conventions.
5. What M0 deliberately leaves empty.

## Decision

1. **Rust stable, edition 2021, Cargo workspace** at the repo root with
   `resolver = "2"`. Nine members under `crates/`:
   `predict-core`, `predict-ngram`, `predict-llm`, `predict-store`,
   `predict-proto`, `predict-eval`, `predictd`, `predict-cli`,
   `frontend-ibus`. Binaries (`predictd`, `predict-cli`, `frontend-ibus`)
   depend toward libraries; libraries depend toward `predict-core` only
   (plus external crates as needed). No cycles.

2. **Crate roles** (details in `docs/ARCHITECTURE.md`):
   - `predict-core`: pure types + `Predictor` trait + ranking. No I/O.
   - `predict-ngram`: fast tier (M1). Depends on `predict-core`.
   - `predict-llm`: slow tier behind a `Backend` trait (M3). Depends on
     `predict-core`.
   - `predict-store`: SQLite personal memory (M4). Depends on `predict-core`.
   - `predict-proto`: versioned IPC structs, `postcard` serialization (M2).
     Standalone (no internal deps) so frontends stay light.
   - `predict-eval`: offline replay harness (M1). Depends on `predict-core`.
   - `predictd` / `predict-cli` / `frontend-ibus`: binaries depending on
     `predict-core` + `predict-proto`.

3. **IPC**: per-user daemon on a Unix socket; messages
   `ContextUpdate`, `SuggestRequest { generation }`, `Suggestion`,
   `Cancel { generation }` with generation-based cancellation — as specified
   in the plan. Serialization with `postcard` (compact, no-std friendly, no
   schema compiler, fits versioned structs). Final adoption verified in M2;
   if `postcard` proves awkward, revisit in a new ADR.

4. **Conventions**:
   - No `unwrap()` outside tests (enforced by review + clippy attention).
   - Errors via `thiserror` in libraries, `anyhow` in binaries. M0 wires
     `thiserror = "2"` into all six libraries and `anyhow = "1"` into all
     three binaries so the convention compiles from day one.
   - Every milestone ends green:
     `cargo build`, `cargo test`, `cargo clippy -- -D warnings`,
     wrapped by `scripts/ci.sh`.
   - Docs (`docs/ARCHITECTURE.md`, `CHANGELOG.md`, `docs/adr/NNNN-*.md`)
     updated in the same commit as the change.

5. **M0 scope**: all crates are empty placeholders (`lib.rs` doc comment /
   `main() -> anyhow::Result<()>` returning `Ok(())`). Core types, protocol,
   and tiers land in M1+. Crate choices beyond `thiserror`/`anyhow`
   (SQLite driver, `zbus`, llama.cpp bindings, `postcard`) are verified for
   build + maintenance in their own milestone ADRs before use.

## Consequences

- `cargo build`, `cargo test`, `cargo clippy -- -D warnings` (via
  `scripts/ci.sh`) are green on a fresh checkout with only
  `thiserror`/`anyhow` as external deps.
- Dependency direction is fixed: nothing depends on binaries; `predict-proto`
  stays dependency-light for frontends.
- Future milestones each add one documented seam:
  M1 (ngram + eval), M2 (proto + daemon/CLI), M3 (`Backend` trait + impl),
  M4 (store), M5 (style), M6 (IBus/`zbus`).
- Risk: `postcard` / `zbus` / SQLite / llama.cpp binding choices are not yet
  build-verified; each gets a verification ADR in its milestone per the plan's
  engineering rules.
