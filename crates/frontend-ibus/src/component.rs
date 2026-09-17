//! Runtime component registration payload (`RegisterComponent`).
//!
//! IBus engines must announce themselves: the daemon only activates engines
//! it knows, and on this stack (ibus 1.5.29) the usable catalog comes from
//! the registry — a live `RegisterComponent` from the running engine is what
//! links our factory to the `predict` engine name. The wire layout below was
//! captured from libibus and cross-checked against the daemon's own registry
//! cache signatures (`(sa{sv}ssssssssavav)` / `(sa{sv}ssssssssussssssss)`).
//!
//! Two traps, both verified against a real daemon:
//! * the component body must be a struct — zbus already frames a `Value`
//!   argument as a variant, so an extra `Value::Value` wrap sends
//!   variant-in-variant and the daemon rejects it;
//! * empty arrays are `av` (array of variant), not `as` or `a{sv}`.

use zbus::zvariant::{Array, Dict, Signature, Structure, StructureBuilder, Value};

/// Well-known component/engine names.
pub const COMPONENT_NAME: &str = "org.freedesktop.IBus.Predict";
/// Engine name clients select (`SetGlobalEngine("predict")`).
pub const ENGINE_NAME: &str = "predict";

fn empty_dict() -> Value<'static> {
    let key: Signature = "s".try_into().expect("s signature");
    let val: Signature = "v".try_into().expect("v signature");
    Value::Dict(Dict::new(&key, &val))
}

fn empty_av() -> Value<'static> {
    let elem: Signature = "v".try_into().expect("v signature");
    Value::Array(Array::new(&elem))
}

fn text(s: &'static str) -> Value<'static> {
    Value::Str(s.into())
}

/// One `IBusEngineDesc` struct (NOT variant-wrapped; callers wrap).
pub fn engine_desc(engine_name: &'static str) -> Structure<'static> {
    StructureBuilder::new()
        .append_field(text("IBusEngineDesc"))
        .append_field(empty_dict())
        .append_field(text(engine_name))
        .append_field(text("Predict"))
        .append_field(text("Local word and sentence prediction"))
        .append_field(text("en"))
        .append_field(text("MIT"))
        .append_field(text("predict contributors"))
        .append_field(text("input-keyboard"))
        .append_field(text("default"))
        .append_field(Value::U32(99))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .append_field(text(""))
        .build()
        .expect("engine desc struct")
}

/// Full `RegisterComponent` argument: the component struct as a `Value`.
/// zbus frames it as the method's `v` parameter on send.
///
/// `exec` is the daemon-spawn command line (binary + `--ibus`); it must
/// exist and accept being spawned by the daemon on engine activation.
/// `component_name` is overrideable for tests: the daemon only activates
/// engines whose component name it knows from its catalog, so hermetic
/// tests shadow a catalog entry (see `tests/ibus_mediated.rs`). Production
/// always uses [`COMPONENT_NAME`].
pub fn component_value(component_name: &'static str, exec: String) -> Value<'static> {
    let engines_elem: Signature = "v".try_into().expect("v signature");
    let mut engines = Array::new(&engines_elem);
    engines
        .append(Value::Value(Box::new(Value::Structure(engine_desc(
            ENGINE_NAME,
        )))))
        .expect("append engine");
    let component = StructureBuilder::new()
        .append_field(text("IBusComponent"))
        .append_field(empty_dict())
        .append_field(text(component_name))
        .append_field(text("Local word and sentence prediction (predictd)"))
        .append_field(Value::Str(exec.into()))
        .append_field(text("0.7.0"))
        .append_field(text("predict contributors"))
        .append_field(text("MIT"))
        .append_field(text("https://github.com/woelper/farsight"))
        .append_field(text("predict"))
        .append_field(empty_av())
        .append_field(Value::Array(engines))
        .build()
        .expect("component struct");
    Value::Structure(component)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_signatures_match_daemon_registry() {
        // Signatures lifted from the daemon's own registry cache file.
        assert_eq!(
            engine_desc(ENGINE_NAME).signature().to_string(),
            "(sa{sv}ssssssssussssssss)"
        );
        let body = component_value(
            COMPONENT_NAME,
            "/usr/local/bin/frontend-ibus --ibus".to_string(),
        );
        let Value::Structure(body) = &body else {
            panic!("component body must be a bare struct, not a variant");
        };
        assert_eq!(body.signature().to_string(), "(sa{sv}ssssssssavav)");
    }
}
