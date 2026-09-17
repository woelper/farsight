//! End-to-end: real ibus-daemon + engine binary + scripted predictd.
//!
//! Requires `IBUS_TEST_ADDRESS` (a private ibus-daemon socket); skips
//! otherwise so plain `cargo test` stays green anywhere:
//! ```sh
//! ibus-daemon --daemonize --panel=disable --address=unix:path=/tmp/ibus-test.sock
//! IBUS_TEST_ADDRESS=unix:path=/tmp/ibus-test-sock cargo test -p frontend-ibus --test ibus_engine
//! ```
//!
//! The fake predictd speaks the real protocol (framing, generations) with
//! scripted replies and records every inbound message for assertions.

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

/// Test client for the engine under test.
#[proxy(
    interface = "org.freedesktop.IBus.Engine",
    default_service = "org.freedesktop.IBus.Predict"
)]
trait Engine {
    fn process_key_event(&self, keyval: u32, keycode: u32, state: u32) -> zbus::Result<bool>;
    fn focus_in(&self) -> zbus::Result<()>;
    fn focus_out(&self) -> zbus::Result<()>;
    fn reset(&self) -> zbus::Result<()>;
    fn enable(&self) -> zbus::Result<()>;
    fn set_surrounding_text(
        &self,
        text: zbus::zvariant::OwnedValue,
        cursor_pos: u32,
        anchor_pos: u32,
    ) -> zbus::Result<()>;
    fn focus_in_id(&self, object_path: &str, client: &str) -> zbus::Result<()>;

    #[zbus(property)]
    fn set_content_type(&self, value: (u32, u32)) -> zbus::Result<()>;
}

/// Test client for our factory.
#[proxy(
    interface = "org.freedesktop.IBus.Factory",
    default_service = "org.freedesktop.IBus.Predict",
    default_path = "/org/freedesktop/IBus/Factory"
)]
trait Factory {
    fn create_engine(&self, name: &str) -> zbus::Result<zbus::zvariant::OwnedObjectPath>;
}

/// One captured engine signal, decoded with exact wire types (this also
/// validates our IBusText/Table encodings against real D-Bus traffic).
#[derive(Debug)]
struct Signal {
    member: String,
    /// Preedit/commit text for text-carrying signals.
    text: Option<String>,
    /// Visibility flag for preedit/lookup signals.
    visible: Option<bool>,
}

/// Background pump: subscribes to the engine path and forwards decoded
/// signals.
fn pump_signals(conn: &Connection, engine_path: &str) -> mpsc::Receiver<Signal> {
    use frontend_ibus::ibus_types::{IBusLookupTable, IBusText};
    let (tx, rx) = mpsc::channel();
    let rule = format!(
        "type='signal',interface='org.freedesktop.IBus.Engine',path='{engine_path}'"
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
            let member = message
                .header()
                .member()
                .map(|member| member.to_string())
                .unwrap_or_default();
            let signal = match member.as_str() {
                "UpdatePreeditText" => message
                    .body()
                    .deserialize::<(zbus::zvariant::OwnedValue, u32, bool, u32)>()
                    .ok()
                    .map(|(variant, _cursor, visible, _mode)| {
                        let text: Option<String> = variant
                            .try_into()
                            .map(|text: IBusText| Some(text.text))
                            .unwrap_or(None);
                        Signal {
                            member: member.clone(),
                            text,
                            visible: Some(visible),
                        }
                    }),
                "CommitText" => message
                    .body()
                    .deserialize::<(zbus::zvariant::OwnedValue,)>()
                    .ok()
                    .map(|(variant,)| {
                        let text: Option<String> = variant
                            .try_into()
                            .map(|text: IBusText| Some(text.text))
                            .unwrap_or(None);
                        Signal {
                            member: member.clone(),
                            text,
                            visible: None,
                        }
                    }),
                "UpdateLookupTable" => message
                    .body()
                    .deserialize::<(zbus::zvariant::OwnedValue, bool)>()
                    .ok()
                    .map(|(variant, visible)| {
                        let count = variant
                            .try_into()
                            .map(|table: IBusLookupTable| table.candidates.len())
                            .unwrap_or(usize::MAX);
                        Signal {
                            member: member.clone(),
                            text: Some(format!("{count} candidates")),
                            visible: Some(visible),
                        }
                    }),
                _ => continue,
            };
            if let Some(signal) = signal {
                let _ = tx.send(signal);
            }
        }
    });
    rx
}

/// Drain available signals for up to `wait`, returning what arrived.
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
// Fake predictd
// ---------------------------------------------------------------------------

/// Scripted predictd speaking the real protocol over a Unix socket.
struct FakePredictd {
    log: Arc<Mutex<Vec<String>>>,
    slow: Arc<std::sync::atomic::AtomicBool>,
}

impl FakePredictd {
    fn spawn(sock: &std::path::Path) -> (Self, std::thread::JoinHandle<()>) {
        let _ = std::fs::remove_file(sock);
        let listener = UnixListener::bind(sock).expect("bind fake predictd");
        let fake = Self {
            log: Arc::new(Mutex::new(Vec::new())),
            slow: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        let log = Arc::clone(&fake.log);
        let slow = Arc::clone(&fake.slow);
        let handle = std::thread::spawn(move || {
            // The engine reconnects per keystroke: serve every connection
            // on its own thread until the test tears the socket down.
            for stream in listener.incoming().flatten() {
                let log = Arc::clone(&log);
                let slow = Arc::clone(&slow);
                std::thread::spawn(move || Self::serve(stream, &log, &slow));
            }
        });
        (fake, handle)
    }

    /// Serve one engine connection to EOF.
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

    /// Handle one message; false means the connection is done.
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

    fn maybe_slow(slow: &std::sync::atomic::AtomicBool) {
        if slow.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn queries(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

const KEY_H: u32 = 0x68;
const KEY_TAB: u32 = 0xff09;

fn engine_text_variant(text: &str) -> zbus::zvariant::OwnedValue {
    IBusText::plain(text).into_variant()
}

#[test]
fn engine_end_to_end_against_real_daemon() {
    let address = std::env::var("IBUS_TEST_ADDRESS").unwrap_or_default();
    if address.is_empty() {
        eprintln!("skipped: set IBUS_TEST_ADDRESS to a test ibus-daemon socket");
        return;
    }
    let dir = std::env::temp_dir().join(format!("predict-ibus-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let sock = dir.join("predictd.sock");

    let (fake, _fake_handle) = FakePredictd::spawn(&sock);

    let mut engine_cmd = std::process::Command::new(env!("CARGO_BIN_EXE_frontend-ibus"));
    engine_cmd
        .env("IBUS_ADDRESS", &address)
        .env("PREDICTD_SOCKET", &sock)
        .arg("--ibus")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let engine_child = engine_cmd.spawn().expect("spawn engine");

    struct Guard(std::process::Child);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = self.0.kill();
        }
    }
    let _guard = Guard(engine_child);

    // Wait for our bus name to appear.
    let conn = Builder::address(address.as_str())
        .expect("parse address")
        .build()
        .expect("test bus connection");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
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
        if names
            .iter()
            .any(|name| name == "org.freedesktop.IBus.Predict")
        {
            break;
        }
        if Instant::now() > deadline {
            panic!("engine never claimed its bus name");
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Instantiate one engine through our own factory.
    let factory = FactoryProxyBlocking::new(&conn).expect("factory proxy");
    let engine_path = factory
        .create_engine("predict")
        .expect("CreateEngine");
    let engine_path = engine_path.to_string();
    assert!(
        engine_path.starts_with("/org/freedesktop/IBus/engine/predict/"),
        "unexpected path: {engine_path}"
    );
    let engine = EngineProxyBlocking::new(&conn, engine_path.as_str()).expect("engine proxy");
    let signals = pump_signals(&conn, &engine_path);

    // Focus + surrounding: the engine learns client context for queries.
    engine.focus_in_id("/fake/ic/1", "fake-client").expect("FocusInId");
    engine
        .set_surrounding_text(engine_text_variant("hello wo"), 8, 8)
        .expect("surrounding");
    let _ = collect_signals(&signals, Duration::from_millis(200));

    // Type 'r': passes through, ghost preedit + no lookup table (1 word).
    let start = Instant::now();
    let handled = engine
        .process_key_event(0x72, 0, 0)
        .expect("ProcessKeyEvent");
    let key_rtt = start.elapsed();
    assert!(!handled, "printable key must pass through");
    assert!(
        key_rtt < Duration::from_millis(500),
        "key handling too slow: {key_rtt:?}"
    );
    let queries = fake.queries();
    assert!(
        queries.iter().any(|q| q == "ctx:hello wor"),
        "context not forwarded: {queries:?}"
    );
    let events = collect_signals(&signals, Duration::from_secs(2));
    let preedit = events.iter().find(|s| {
        s.member == "UpdatePreeditText" && s.text.as_deref() == Some(" wide.")
    });
    assert!(
        preedit.is_some(),
        "no ghost preedit for sentence: {events:?}"
    );

    // Tab consumes the key and commits the sentence verbatim.
    let handled = engine.process_key_event(KEY_TAB, 0, 0).expect("Tab");
    assert!(handled, "Tab with suggestion must be consumed");
    let events = collect_signals(&signals, Duration::from_secs(2));
    let commit = events.iter().find(|s| {
        s.member == "CommitText" && s.text.as_deref() == Some(" wide.")
    });
    assert!(commit.is_some(), "no sentence commit: {events:?}");

    // Password field: silence and no predictd traffic.
    engine
        .set_content_type((8, 0))
        .expect("set password content type");
    let before = fake.queries().len();
    let handled = engine.process_key_event(KEY_H, 0, 0).expect("key");
    assert!(!handled, "password keys must pass through");
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        fake.queries().len(),
        before,
        "predictd queried from password field"
    );
    let events = collect_signals(&signals, Duration::from_millis(300));
    assert!(
        !events.iter().any(|s| s.member == "UpdatePreeditText"
            && s.visible == Some(true)
            && matches!(s.text.as_deref(), Some("world") | Some(" wide."))),
        "suggestion shown in password field: {events:?}"
    );

    // Slow daemon: the key still sails through fast.
    engine
        .set_content_type((0, 0))
        .expect("clear content type");
    fake.slow.store(true, std::sync::atomic::Ordering::SeqCst);
    let start = Instant::now();
    let handled = engine.process_key_event(KEY_H, 0, 0).expect("key");
    let slow_rtt = start.elapsed();
    assert!(!handled);
    assert!(
        slow_rtt < Duration::from_millis(400),
        "slow daemon stalled typing: {slow_rtt:?}"
    );
    fake.slow.store(false, std::sync::atomic::Ordering::SeqCst);

    // Focus out settles the buffer for learning.
    engine.focus_out().expect("FocusOut");
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        fake.queries().iter().any(|q| q.starts_with("commit:")),
        "no settled commit on focus out: {:?}",
        fake.queries()
    );

    std::fs::remove_dir_all(&dir).unwrap_or(());
}
