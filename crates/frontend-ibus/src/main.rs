//! frontend-ibus: IBus prediction engine for predictd.
//!
//! A per-user IBus engine (D-Bus, `zbus` blocking API) that forwards typing
//! context to predictd and renders suggestions as preedit ghost text with a
//! lookup-table fallback. The engine never slows typing down: predictd gets
//! 10 ms per keystroke, then the key passes through suggestion-free.
//!
//! Run under ibus-daemon (see `docs/INSTALL-linux.md`), or directly with
//! `IBUS_ADDRESS` set for testing:
//! `IBUS_ADDRESS=unix:path=/tmp/ibus.sock ./target/debug/frontend-ibus --ibus`.

use anyhow::{Context as _, Result};
use frontend_ibus::engine::{Display, EngineState, KeyAction, PREDICT_BUDGET};
use frontend_ibus::ibus_types::{IBusLookupTable, IBusText};
use predict_proto::{
    ClientMsg, ContextUpdate, SuggestRequest, read_sentence_reply, read_word_reply, socket_path,
    write_client_msg,
};
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
use std::time::Duration;
use zbus::blocking::{Connection, connection::Builder};
use zbus::interface;
use zbus::zvariant::OwnedValue;

/// Well-known bus name of this component.
const BUS_NAME: &str = "org.freedesktop.IBus.Predict";
/// Factory object path (mirrors the stock engines' layout).
const FACTORY_PATH: &str = "/org/freedesktop/IBus/Factory";
/// Engine object path prefix; instances append `/<n>`.
const ENGINE_PREFIX: &str = "/org/freedesktop/IBus/engine/predict";
/// IBus engine interface name.
const ENGINE_IFACE: &str = "org.freedesktop.IBus.Engine";
/// IBus bus interface (for `Hello`).
const BUS_IFACE: &str = "org.freedesktop.IBus";
const BUS_PATH: &str = "/org/freedesktop/IBus";
/// predictd exchange budget per keystroke (hard rule: never slow typing).
const DAEMON_BUDGET: Duration = PREDICT_BUDGET;

fn main() -> Result<()> {
    let mut ibus_flag = false;
    let mut socket_override: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ibus" => ibus_flag = true,
            "--predictd-socket" => {
                socket_override = args.next();
            }
            "--help" | "-h" => {
                println!("frontend-ibus [--ibus] [--predictd-socket PATH]");
                println!("  --ibus   run as an IBus engine (ibus-daemon spawns with this flag)");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }
    if !ibus_flag {
        eprintln!("frontend-ibus: missing --ibus (see docs/INSTALL-linux.md)");
    }
    if let Some(path) = socket_override {
        // SAFETY: process-local test hook only, before any threads spawn.
        unsafe { std::env::set_var("PREDICTD_SOCKET", path) };
    }

    let address = std::env::var("IBUS_ADDRESS").context(
        "IBUS_ADDRESS is not set (run under ibus-daemon or export it for testing)",
    )?;
    let conn = Builder::address(address.as_str())
        .context("parse IBUS_ADDRESS")?
        .build()
        .context("connect to IBus")?;
    conn.request_name(BUS_NAME).context("claim bus name")?;
    let factory = Factory {
        conn: conn.clone(),
        next_id: Mutex::new(1u64),
    };
    conn.object_server()
        .at(FACTORY_PATH, factory)
        .context("serve factory")?;
    // Register the connection like the stock engines do.
    let _ = conn.call_method(Some(BUS_NAME), BUS_PATH, Some(BUS_IFACE), "Hello", &());
    // Announce our component: without this the daemon never routes engine
    // activation (SetGlobalEngine/CreateEngine) to our factory.
    // PREDICT_COMPONENT_NAME exists so hermetic tests can shadow a catalog
    // entry (daemons only activate known component names); production
    // leaves it unset.
    let component_name: &'static str = match std::env::var("PREDICT_COMPONENT_NAME") {
        Ok(name) if !name.is_empty() => Box::leak(name.into_boxed_str()),
        _ => frontend_ibus::component::COMPONENT_NAME,
    };
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "frontend-ibus".to_string());
    let component = frontend_ibus::component::component_value(component_name, format!("{exe} --ibus"));
    if let Err(e) = conn.call_method(
        Some("org.freedesktop.IBus"),
        BUS_PATH,
        Some(BUS_IFACE),
        "RegisterComponent",
        &(component,),
    ) {
        eprintln!("frontend-ibus: RegisterComponent failed (daemon-mediated use will not work): {e:#}");
    }
    eprintln!("frontend-ibus: serving {BUS_NAME}");
    // Block until the bus goes away.
    conn.closed();
    Ok(())
}

/// predictd socket path with test override.
fn predictd_path() -> String {
    if let Ok(path) = std::env::var("PREDICTD_SOCKET") {
        if !path.is_empty() {
            return path;
        }
    }
    socket_path().to_string_lossy().into_owned()
}

/// Component factory: the daemon calls `CreateEngine` to instantiate us.
struct Factory {
    conn: Connection,
    next_id: Mutex<u64>,
}

#[interface(name = "org.freedesktop.IBus.Factory")]
impl Factory {
    /// Create one engine instance; returns its object path.
    fn create_engine(&mut self, _name: &str) -> zbus::zvariant::OwnedObjectPath {
        let mut next = self.next_id.lock().unwrap_or_else(|e| e.into_inner());
        let id = *next;
        *next += 1;
        drop(next);
        let path = format!("{ENGINE_PREFIX}/{id}");
        let engine = EngineObject::new(self.conn.clone(), path.clone());
        if let Err(e) = self.conn.object_server().at(path.clone(), engine) {
            eprintln!("frontend-ibus: serve engine failed: {e:#}");
        }
        path.try_into().expect("engine path must be a valid object path")
    }
}

/// Per-engine D-Bus object: thin methods over shared state.
struct EngineObject {
    conn: Connection,
    path: String,
    state: Mutex<EngineInner>,
}

struct EngineInner {
    engine: EngineState,
    focused_context: Option<String>,
}

impl EngineObject {
    fn new(conn: Connection, path: String) -> Self {
        Self {
            conn,
            path,
            state: Mutex::new(EngineInner {
                engine: EngineState::new(),
                focused_context: None,
            }),
        }
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut EngineInner) -> T) -> T {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    /// Emit one engine signal (best effort: a failed emit must not break
    /// key handling).
    fn emit(&self, signal: &str, body: &(impl serde::Serialize + zbus::zvariant::DynamicType)) {
        if let Err(e) = self
            .conn
            .emit_signal(None::<&str>, self.path.clone(), ENGINE_IFACE, signal, body)
        {
            eprintln!("frontend-ibus: emit {signal} failed: {e:#}");
        }
    }

    fn hide_display(&self) {
        let hidden = IBusText::plain("").into_variant();
        self.emit("UpdatePreeditText", &(hidden, 0u32, false, 0u32));
        let empty = IBusLookupTable::from_words(&[]).into_variant();
        self.emit("UpdateLookupTable", &(empty, false));
    }

    fn show_display(&self, display: &Display) {
        match &display.preedit {
            Some(text) => {
                let preedit = IBusText::underlined(text).into_variant();
                let cursor = text.chars().count() as u32;
                self.emit("UpdatePreeditText", &(preedit, cursor, true, 0u32));
            }
            None => {
                let hidden = IBusText::plain("").into_variant();
                self.emit("UpdatePreeditText", &(hidden, 0u32, false, 0u32));
            }
        }
        if display.lookup.is_empty() {
            let empty = IBusLookupTable::from_words(&[]).into_variant();
            self.emit("UpdateLookupTable", &(empty, false));
        } else {
            let table = IBusLookupTable::from_words(&display.lookup).into_variant();
            self.emit("UpdateLookupTable", &(table, true));
        }
    }

    /// One predictd round-trip within the hard budget. Returns
    /// `(words, sentence)`; slow daemons yield empty hands, never stalls.
    /// Lock hygiene: pure socket I/O, no engine locks held.
    fn ask_predictd(before: &str, after: &str, generation: u64) -> (Vec<String>, Option<String>) {
        let empty = (Vec::new(), None);
        let mut stream = match UnixStream::connect(predictd_path()) {
            Ok(stream) => stream,
            Err(_) => return empty,
        };
        let context = ContextUpdate {
            app_id: "frontend-ibus".to_string(),
            before: before.to_string(),
            after: after.to_string(),
            sensitive: false,
            style_id: "default".to_string(),
        };
        if write_client_msg(&mut stream, &ClientMsg::ContextUpdate(context)).is_err() {
            return empty;
        }
        if write_client_msg(
            &mut stream,
            &ClientMsg::Suggest(SuggestRequest { generation }),
        )
        .is_err()
        {
            return empty;
        }
        if write_client_msg(
            &mut stream,
            &ClientMsg::SuggestSentence(SuggestRequest { generation }),
        )
        .is_err()
        {
            return empty;
        }
        let start = std::time::Instant::now();
        let words = read_word_reply(&mut stream, generation, DAEMON_BUDGET)
            .ok()
            .flatten()
            .map(|s| s.candidates.into_iter().map(|c| c.text).collect())
            .unwrap_or_default();
        let elapsed = start.elapsed();
        let sentence = if elapsed < DAEMON_BUDGET {
            read_sentence_reply(&mut stream, generation, DAEMON_BUDGET - elapsed)
                .ok()
                .flatten()
                .map(|s| s.text)
        } else {
            None
        };
        let _ = write_client_msg(
            &mut stream,
            &ClientMsg::Cancel(predict_proto::CancelMsg { generation }),
        );
        (words, sentence)
    }

    /// Fire-and-forget settled text for learning.
    fn learn_commit(text: &str) {
        let Ok(mut stream) = UnixStream::connect(predictd_path()) else {
            return;
        };
        let commit = ClientMsg::CommitText(predict_proto::CommitText {
            text: text.to_string(),
            style_id: "default".to_string(),
            sensitive: false,
        });
        let _ = write_client_msg(&mut stream, &commit);
    }
}

/// Decode an IBusText variant from `SetSurroundingText`.
fn decode_text(value: OwnedValue) -> Option<IBusText> {
    value.try_into().ok()
}

#[interface(name = "org.freedesktop.IBus.Engine")]
impl EngineObject {
    /// Handle one key event. Returns true when consumed.
    fn process_key_event(&mut self, keyval: u32, _keycode: u32, state: u32) -> bool {
        let now = std::time::Instant::now();
        let action = self.with_state(|inner| inner.engine.handle_key(keyval, state, now));
        match action {
            KeyAction::Pass => false,
            KeyAction::ClearAndPass => {
                self.hide_display();
                false
            }
            KeyAction::Query {
                generation,
                before,
                after,
            } => {
                let start = std::time::Instant::now();
                let (words, sentence) = Self::ask_predictd(&before, &after, generation);
                let display = if start.elapsed() >= DAEMON_BUDGET {
                    self.with_state(|inner| {
                        inner.engine.apply_words(generation, Vec::new());
                        inner.engine.apply_sentence(generation, None);
                        inner.engine.display()
                    })
                } else {
                    self.with_state(|inner| {
                        inner.engine.apply_words(generation, words);
                        inner.engine.apply_sentence(generation, sentence);
                        inner.engine.display()
                    })
                };
                self.show_display(&display);
                false
            }
            KeyAction::Commit { delete_before, text } => {
                if delete_before > 0 {
                    self.delete_surrounding(delete_before);
                }
                let commit = IBusText::plain(&text).into_variant();
                self.emit("CommitText", &(commit,));
                self.hide_display();
                true
            }
            KeyAction::SettleAndPass { settled } => {
                if !settled.trim().is_empty() {
                    Self::learn_commit(&settled);
                }
                self.hide_display();
                false
            }
        }
    }

    /// Delete `nchars` before the cursor on the focused input context.
    fn delete_surrounding(&self, nchars: usize) {
        let path = self.with_state(|inner| inner.focused_context.clone());
        let Some(path) = path else { return };
        let offset = -(nchars as i32);
        let _ = self.conn.call_method(
            None::<&str>,
            path.as_str(),
            Some("org.freedesktop.IBus.InputContext"),
            "DeleteSurroundingText",
            &(offset, nchars as u32),
        );
    }

    fn focus_in(&mut self) {
        self.with_state(|inner| {
            inner.engine.reset();
        });
        self.hide_display();
    }

    fn focus_out(&mut self) {
        let settled = self.with_state(|inner| inner.engine.focus_out());
        if let Some(text) = settled {
            Self::learn_commit(&text);
        }
        self.hide_display();
    }

    fn reset(&mut self) {
        self.with_state(|inner| inner.engine.reset());
        self.hide_display();
    }

    fn enable(&mut self) {
        // Ask the daemon to keep surrounding text flowing our way.
        self.emit("RequireSurroundingText", &());
    }

    fn disable(&mut self) {
        self.with_state(|inner| inner.engine.reset());
        self.hide_display();
    }

    fn set_capabilities(&mut self, _caps: u32) {}

    fn set_cursor_location(&mut self, _x: i32, _y: i32, _w: i32, _h: i32) {}

    fn property_activate(&mut self, _name: &str, _state: u32) {}

    fn property_show(&mut self, _name: &str) {}

    fn property_hide(&mut self, _name: &str) {}

    fn candidate_clicked(&mut self, _index: u32, _button: u32, _state: u32) {}

    fn page_up(&mut self) {}

    fn page_down(&mut self) {}

    fn cursor_up(&mut self) {}

    fn cursor_down(&mut self) {}

    fn focus_in_id(&mut self, object_path: &str, _client: &str) {
        let path = object_path.to_string();
        self.with_state(|inner| {
            inner.engine.reset();
            inner.focused_context = Some(path);
        });
        self.hide_display();
    }

    fn focus_out_id(&mut self, _object_path: &str) {
        self.focus_out();
    }

    fn set_surrounding_text(&mut self, text: OwnedValue, cursor_pos: u32, anchor_pos: u32) {
        let full = match decode_text(text) {
            Some(t) => t.text,
            None => return,
        };
        self.with_state(|inner| {
            inner
                .engine
                .set_surrounding(&full, cursor_pos as usize, anchor_pos as usize);
        });
    }

    /// ContentType property setter (purpose, hints); password/terminal
    /// fields go sensitive. A getter exists only because the macro wants
    /// the pair — the daemon never reads it back.
    #[zbus(property)]
    fn set_content_type(&mut self, value: (u32, u32)) {
        let (purpose, hints) = value;
        self.with_state(|inner| inner.engine.set_content_type(purpose, hints));
    }

    #[zbus(property)]
    fn content_type(&self) -> (u32, u32) {
        (0, 0)
    }
}
