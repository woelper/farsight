//! Full install path: private ibus-daemon + runtime component registration +
//! daemon-spawned engine + client-path key events through an InputContext.
//!
//! Runs only when `ibus-daemon` is on PATH (skips otherwise, so plain
//! `cargo test` stays green anywhere). Everything is hermetic: private
//! socket, private XDG dirs, scripted predictd; the daemon is killed at
//! the end. This validates what the engine-direct test cannot:
//!   * our `RegisterComponent` payload registers (`GetEnginesByNames`),
//!   * the daemon spawns our binary via the component `Exec` line,
//!   * `SetGlobalEngine` routes client keys to our engine,
//!   * engine signals reach the client (daemon forwards them to the IC),
//!   * client `SetContentType(password)` silences us end to end.

use frontend_ibus::ibus_types::IBusText;
use predict_proto::{
    ClientMsg, DaemonMsg, LearningState, ProtoCandidate, Suggestion, read_client_msg,
    write_daemon_msg,
};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use zbus::blocking::{Connection, MessageIterator, connection::Builder};
use zbus::proxy;

// ---------------------------------------------------------------------------
// D-Bus helpers
// ---------------------------------------------------------------------------

/// Test client for an input context owned by the daemon.
#[proxy(
    interface = "org.freedesktop.IBus.InputContext",
    default_service = "org.freedesktop.IBus"
)]
trait InputContext {
    fn focus_in(&self) -> zbus::Result<()>;
    fn focus_out(&self) -> zbus::Result<()>;
    fn process_key_event(&self, keyval: u32, keycode: u32, state: u32) -> zbus::Result<bool>;
    fn set_capabilities(&self, caps: u32) -> zbus::Result<()>;
    fn set_surrounding_text(
        &self,
        text: zbus::zvariant::OwnedValue,
        cursor_pos: u32,
        anchor_pos: u32,
    ) -> zbus::Result<()>;
}

/// Client content-type the way toolkits send it: `Properties.Set` on the
/// input context (there is no `SetContentType` method on this daemon
/// build — verified against libibus traffic).
fn ic_set_content_type(conn: &Connection, ic_path: &str, purpose: u32, hints: u32) {
    use zbus::zvariant::{StructureBuilder, Value};
    let content = StructureBuilder::new()
        .append_field(Value::U32(purpose))
        .append_field(Value::U32(hints))
        .build()
        .expect("content type struct");
    conn.call_method(
        Some("org.freedesktop.IBus"),
        ic_path,
        Some("org.freedesktop.DBus.Properties"),
        "Set",
        &(
            "org.freedesktop.IBus.InputContext",
            "ContentType",
            Value::Structure(content),
        ),
    )
    .expect("Properties.Set ContentType");
}

/// One captured client-visible signal on the input context.
#[derive(Debug)]
struct Signal {
    member: String,
    text: Option<String>,
    visible: Option<bool>,
}

/// Background pump for IC-path signals (this is what a toolkit sees).
///
/// NOTE: never touch `message.header()` here. The daemon signs its signals
/// with its well-known name, and zbus's `header()` unconditionally parses
/// the sender as a *unique* name and panics (`Invalid field
/// reconstruction`). Member dispatch is by body shape instead — arities
/// are strict, so shapes cannot alias.
fn pump_ic_signals(conn: &Connection, ic_path: &str) -> mpsc::Receiver<Signal> {
    let (tx, rx) = mpsc::channel();
    let rule = format!(
        "type='signal',interface='org.freedesktop.IBus.InputContext',path='{ic_path}'"
    );
    conn.call_method(
        Some("org.freedesktop.DBus"),
        "/org/freedesktop/DBus",
        Some("org.freedesktop.DBus"),
        "AddMatch",
        &rule.as_str(),
    )
    .expect("AddMatch");
    let mut iter = MessageIterator::from(conn);
    std::thread::spawn(move || {
        for message in iter.by_ref().flatten() {
            if message.message_type() != zbus::message::Type::Signal {
                continue;
            }
            let signal = decode_ic_signal(&message);
            if let Some(signal) = signal {
                let _ = tx.send(signal);
            }
        }
    });
    rx
}

/// Identify an IC signal by body shape (see `pump_ic_signals`).
fn decode_ic_signal(message: &zbus::message::Message) -> Option<Signal> {
    use frontend_ibus::ibus_types::IBusText;
    let body = message.body();
    if let Ok((variant, _cursor, visible, _mode)) = body
        .deserialize::<(zbus::zvariant::OwnedValue, u32, bool, u32)>()
    {
        return Some(Signal {
            member: "UpdatePreeditText".to_string(),
            text: variant
                .try_into()
                .map(|text: IBusText| Some(text.text))
                .unwrap_or(None),
            visible: Some(visible),
        });
    }
    if let Ok((variant, _cursor, visible)) = body
        .deserialize::<(zbus::zvariant::OwnedValue, u32, bool)>()
    {
        return Some(Signal {
            member: "UpdatePreeditText".to_string(),
            text: variant
                .try_into()
                .map(|text: IBusText| Some(text.text))
                .unwrap_or(None),
            visible: Some(visible),
        });
    }
    if let Ok((variant,)) = body.deserialize::<(zbus::zvariant::OwnedValue,)>() {
        return Some(Signal {
            member: "CommitText".to_string(),
            text: variant
                .try_into()
                .map(|text: IBusText| Some(text.text))
                .unwrap_or(None),
            visible: None,
        });
    }
    None
}

/// Drain available signals for up to `wait`.
fn collect_signals(rx: &mpsc::Receiver<Signal>, wait: Duration) -> Vec<Signal> {
    let deadline = Instant::now() + wait;
    let mut out = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining.min(Duration::from_millis(50))) {
            Ok(signal) => out.push(signal),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Fake predictd (same scripted protocol as the engine-direct test)
// ---------------------------------------------------------------------------

struct FakePredictd {
    log: Arc<Mutex<Vec<String>>>,
    slow: Arc<std::sync::atomic::AtomicBool>,
}

impl FakePredictd {
    fn spawn(sock: &std::path::Path) -> Self {
        let _ = std::fs::remove_file(sock);
        let listener = UnixListener::bind(sock).expect("bind fake predictd");
        let fake = Self {
            log: Arc::new(Mutex::new(Vec::new())),
            slow: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let log = Arc::clone(&fake.log);
        let slow = Arc::clone(&fake.slow);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let log = Arc::clone(&log);
                let slow = Arc::clone(&slow);
                std::thread::spawn(move || Self::serve(stream, &log, &slow));
            }
        });
        fake
    }

    fn serve(
        mut stream: UnixStream,
        log: &Arc<Mutex<Vec<String>>>,
        slow: &Arc<std::sync::atomic::AtomicBool>,
    ) {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        while let Ok(msg) = read_client_msg(&mut stream) {
            if !Self::handle_one(&mut stream, msg, log, slow) {
                break;
            }
        }
    }

    fn handle_one(
        stream: &mut UnixStream,
        msg: ClientMsg,
        log: &Arc<Mutex<Vec<String>>>,
        slow: &Arc<std::sync::atomic::AtomicBool>,
    ) -> bool {
        let note = |tag: String| log.lock().unwrap().push(tag);
        match msg {
            ClientMsg::ContextUpdate(ctx) => {
                note(format!("ctx:{}", ctx.before));
            }
            ClientMsg::Suggest(req) => {
                note(format!("suggest:{}", req.generation));
                Self::maybe_slow(slow);
                let reply = DaemonMsg::Suggestion(Suggestion {
                    generation: req.generation,
                    candidates: vec![ProtoCandidate {
                        text: "world".to_string(),
                        score: 2.0,
                    }],
                    style_id: "default".to_string(),
                });
                if write_daemon_msg(stream, &reply).is_err() {
                    return false;
                }
            }
            ClientMsg::SuggestSentence(req) => {
                note(format!("sentence-req:{}", req.generation));
                Self::maybe_slow(slow);
                let reply = DaemonMsg::Sentence(predict_proto::SentenceSuggestion {
                    generation: req.generation,
                    text: " wide.".to_string(),
                    confidence: -0.3,
                    style_id: "default".to_string(),
                });
                if write_daemon_msg(stream, &reply).is_err() {
                    return false;
                }
            }
            ClientMsg::Cancel(req) => {
                note(format!("cancel:{}", req.generation));
            }
            ClientMsg::CommitText(commit) => {
                note(format!("commit:{}", commit.text));
            }
            ClientMsg::SetLearning(_) | ClientMsg::ForgetAll => {
                let reply = DaemonMsg::LearningState(LearningState {
                    enabled: true,
                    documents: 0,
                });
                if write_daemon_msg(stream, &reply).is_err() {
                    return false;
                }
            }
        }
        true
    }

    fn maybe_slow(slow: &Arc<std::sync::atomic::AtomicBool>) {
        if slow.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn queries(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// Component XML + daemon spawning
// ---------------------------------------------------------------------------

fn have_ibus_daemon() -> bool {
    std::process::Command::new("ibus-daemon")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Catalog shadowing
// ---------------------------------------------------------------------------

/// Component name from the system catalog for the engine to shadow.
///
/// Verified against ibus 1.5.29: the daemon only activates engines whose
/// component name it knows from its catalog (XML/cache) — unknown names
/// register fine but stay inert ("Cannot find engine"). Shadowing merges:
/// the catalog's own engines keep working, ours becomes activatable, all
/// on a private daemon so nothing else can observe it.
fn catalog_shadow_name() -> Option<String> {
    let dir = std::path::Path::new("/usr/share/ibus/component");
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "xml").unwrap_or(false) {
            let text = std::fs::read_to_string(&path).ok()?;
            // First <name> in a component file is the component name.
            for line in text.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("<name>") {
                    if let Some(name) = rest.strip_suffix("</name>") {
                        names.push(name.to_string());
                    }
                    break;
                }
            }
        }
    }
    ["org.freedesktop.IBus.Simple", "org.freedesktop.IBus.Table"]
        .iter()
        .map(|s| s.to_string())
        .find(|n| names.contains(n))
        .or(names.into_iter().next())
}

/// Wait until `cond` holds (poll every 100 ms) or panic after `timeout`.
fn wait_for(timeout: Duration, what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("timed out waiting for {what}");
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

const KEY_H: u32 = 0x68;
const KEY_R: u32 = 0x72;
const KEY_TAB: u32 = 0xff09;

#[test]
fn daemon_mediated_client_path() {
    if !have_ibus_daemon() {
        eprintln!("skipped: ibus-daemon not on PATH");
        return;
    }
    let dir =
        std::env::temp_dir().join(format!("predict-ibus-mediated-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // Private XDG dirs: the daemon must not touch the real session state
    // (it would otherwise refuse as "session already has an ibus-daemon").
    let data = dir.join("data");
    let config = dir.join("config");
    let cache = dir.join("cache");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&config).unwrap();
    std::fs::create_dir_all(&cache).unwrap();

    let predictd_sock = dir.join("predictd.sock");
    let fake = FakePredictd::spawn(&predictd_sock);

    let engine_bin = env!("CARGO_BIN_EXE_frontend-ibus");

    let bus_sock = dir.join("ibus.sock");
    let address = format!("unix:path={}", bus_sock.display());
    struct Guard(std::process::Child);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = self.0.kill();
        }
    }
    let daemon = std::process::Command::new("ibus-daemon")
        .arg("--panel=disable")
        .arg(format!("--address={address}"))
        .env("XDG_DATA_HOME", &data)
        .env("XDG_CONFIG_HOME", &config)
        .env("XDG_CACHE_HOME", &cache)
        // Inherited by the engine below (and by anything the daemon spawns).
        .env("IBUS_ADDRESS", &address)
        .env("PREDICTD_SOCKET", &predictd_sock)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(
            std::fs::File::create(dir.join("daemon-stderr.log")).unwrap(),
        ))
        .spawn()
        .expect("spawn ibus-daemon");
    let _daemon_guard = Guard(daemon);

    // The daemon needs a moment to create its socket.
    let conn = {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match Builder::address(address.as_str()).expect("addr").build() {
                Ok(c) => break c,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => panic!("connect to test daemon: {e}"),
            }
        }
    };

    // 1. Our engine self-registers under a shadowed catalog name (see
    // `catalog_shadow_name`): the daemon only activates known components.
    let shadow = catalog_shadow_name().expect("no usable component in system catalog");
    let engine = std::process::Command::new(engine_bin)
        .env("IBUS_ADDRESS", &address)
        .env("PREDICTD_SOCKET", &predictd_sock)
        .env("PREDICT_COMPONENT_NAME", &shadow)
        .arg("--ibus")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::from(
            std::fs::File::create(dir.join("engine-stderr.log")).unwrap(),
        ))
        .spawn()
        .expect("spawn engine");
    let _engine_guard = Guard(engine);
    wait_for(Duration::from_secs(20), "engine bus name", || {
        let names: Vec<String> = conn
            .call_method(
                Some("org.freedesktop.DBus"),
                "/org/freedesktop/DBus",
                Some("org.freedesktop.DBus"),
                "ListNames",
                &(),
            )
            .expect("ListNames")
            .body()
            .deserialize()
            .expect("names");
        names.iter().any(|n| n == "org.freedesktop.IBus.Predict")
    });

    // 2. Activating `predict` instantiates it through our factory. Retry:
    // the engine claims its bus name just before registering, so the first
    // attempts can race the registration.
    {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let reply = conn.call_method(
                Some("org.freedesktop.IBus"),
                "/org/freedesktop/IBus",
                Some("org.freedesktop.IBus"),
                "SetGlobalEngine",
                &("predict",),
            );
            match reply {
                Ok(msg) => {
                    msg.body().deserialize::<()>().expect("SetGlobalEngine reply");
                    break;
                }
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(200));
                }
                Err(e) => panic!("SetGlobalEngine: {e}"),
            }
        }
    }

    // 3. Client path: create a context, focus it, type through the daemon.
    let ic_path: zbus::zvariant::OwnedObjectPath = conn
        .call_method(
            Some("org.freedesktop.IBus"),
            "/org/freedesktop/IBus",
            Some("org.freedesktop.IBus"),
            "CreateInputContext",
            &("test-client",),
        )
        .expect("CreateInputContext")
        .body()
        .deserialize()
        .expect("IC path");
    let ic_path = ic_path.to_string();
    let ic = InputContextProxyBlocking::new(&conn, ic_path.as_str()).expect("IC proxy");
    let signals = pump_ic_signals(&conn, &ic_path);

    // Capabilities before focus: the daemon rejects FocusIn otherwise
    // ("input context does not support focus").
    ic.set_capabilities(63).expect("SetCapabilities");
    ic.focus_in().expect("FocusIn");
    ic.set_surrounding_text(IBusText::plain("hello wo").into_variant(), 8, 8)
        .expect("surrounding");
    let _ = collect_signals(&signals, Duration::from_millis(300));

    // Type 'r' through the daemon: passes through, ghost arrives at client.
    let start = Instant::now();
    let handled = ic.process_key_event(KEY_R, 0, 0).expect("ProcessKeyEvent");
    let key_rtt = start.elapsed();
    assert!(!handled, "printable key must pass through");
    assert!(
        key_rtt < Duration::from_secs(2),
        "mediated key too slow: {key_rtt:?}"
    );
    assert!(
        fake.queries().iter().any(|q| q == "ctx:hello wor"),
        "context not forwarded: {:?}",
        fake.queries()
    );
    let events = collect_signals(&signals, Duration::from_secs(5));
    assert!(
        events.iter().any(|s| s.text.as_deref() == Some(" wide.")
            && s.visible == Some(true)),
        "no client-visible ghost preedit: {events:?}"
    );

    // Tab through the daemon commits the sentence at the client.
    let handled = ic.process_key_event(KEY_TAB, 0, 0).expect("Tab");
    assert!(handled, "Tab with suggestion must be consumed");
    let events = collect_signals(&signals, Duration::from_secs(5));
    assert!(
        events
            .iter()
            .any(|s| s.member == "CommitText" && s.text.as_deref() == Some(" wide.")),
        "no client-visible commit: {events:?}"
    );

    // Password purpose end to end: client declares it, engine goes silent.
    ic_set_content_type(&conn, &ic_path, 8, 0);
    std::thread::sleep(Duration::from_millis(300));
    let before = fake.queries().len();
    let handled = ic.process_key_event(KEY_H, 0, 0).expect("key");
    assert!(!handled, "password keys must pass through");
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(
        fake.queries().len(),
        before,
        "predictd queried from password field: {:?}",
        fake.queries()
    );
    let events = collect_signals(&signals, Duration::from_millis(300));
    assert!(
        !events.iter().any(|s| s.visible == Some(true)
            && matches!(s.text.as_deref(), Some("world") | Some(" wide."))),
        "suggestion shown in password field: {events:?}"
    );

    // Slow daemon still never stalls the client path.
    ic_set_content_type(&conn, &ic_path, 0, 0);
    fake.slow.store(true, std::sync::atomic::Ordering::SeqCst);
    let start = Instant::now();
    let handled = ic.process_key_event(KEY_H, 0, 0).expect("key");
    let slow_rtt = start.elapsed();
    assert!(!handled);
    assert!(
        slow_rtt < Duration::from_millis(500),
        "slow daemon stalled typing: {slow_rtt:?}"
    );
    fake.slow.store(false, std::sync::atomic::Ordering::SeqCst);

    // Focus out settles the buffer for learning.
    ic.focus_out().expect("FocusOut");
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        fake.queries().iter().any(|q| q.starts_with("commit:")),
        "no settled commit on focus out: {:?}",
        fake.queries()
    );

    std::fs::remove_dir_all(&dir).unwrap_or(());
}
