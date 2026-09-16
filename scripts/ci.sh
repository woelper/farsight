#!/usr/bin/env bash
set -euo pipefail
cargo build
cargo test
cargo clippy -- -D warnings
