//! frontend-ibus library: IBus wire types and the engine state machine.
//!
//! The D-Bus service glue lives in the binary (`main.rs`); everything
//! protocol-testable is here.

pub mod component;
pub mod engine;
pub mod ibus_types;
