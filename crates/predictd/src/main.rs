//! predictd: per-user prediction daemon.
//!
//! Listens on a Unix socket ([`predict_proto::socket_path`]). Word requests
//! (`Suggest`) are answered synchronously from the fast tier. Sentence
//! requests (`SuggestSentence`, M3) run on a worker thread against the
//! optional LLM backend: a newer generation cancels older work, and the
//! worker answers only while its generation is still current — otherwise it
//! stays silent. One `std::thread` per connection plus one per slow request;
//! no async runtime (see ADR 0003/0004).

use anyhow::{Context as _, Result};
use predict_core::{Context, Predictor as _, ResolvedStyle};
use predict_llm::{Backend, CancelToken, LlmConfig, SentenceRequest, llama::LlamaBackend};
use predict_ngram::NgramModel;
use predict_proto::{
    CancelMsg, ClientMsg, ContextUpdate, DaemonMsg, ProtoCandidate, SentenceSuggestion,
    SuggestRequest, Suggestion, read_client_msg, socket_path, write_daemon_msg,
};
use std::io::ErrorKind;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Base n-gram training text until the personal store lands in M4.
const BASE_CORPUS: &str = include_str!("../../../corpora/sample_en_de.txt");

fn main() -> Result<()> {
    let model =
        NgramModel::from_text(BASE_CORPUS).map_err(|e| anyhow::anyhow!("train base model: {e}"))?;
    eprintln!(
        "predictd: {} tokens, {} words",
        model.total_tokens(),
        model.vocab_size()
    );
    let slow = load_llm_from(&config_path());

    let path = socket_path();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create socket dir {}", parent.display()))?;
        }
    }
    // Remove a stale socket left by a previous run; a live daemon would have
    // an exclusive bind, so unlink-then-bind is the simple M2 choice.
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("remove stale socket {}", path.display()))?;
    }
    let listener =
        UnixListener::bind(&path).with_context(|| format!("bind {}", path.display()))?;
    eprintln!("predictd: listening on {}", path.display());

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let model = model.clone();
                let slow = slow.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &model, slow) {
                        eprintln!("predictd: connection error: {e:#}");
                    }
                });
            }
            Err(e) => eprintln!("predictd: accept error: {e:#}"),
        }
    }
    Ok(())
}

/// Slow-tier configuration shared across connections.
#[derive(Clone)]
struct SlowCfg {
    backend: Arc<dyn Backend>,
    max_tokens: usize,
    threshold: f32,
}

/// Path of the daemon config file.
fn config_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("predict/predictd.toml");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".config/predict/predictd.toml")
}

/// Parse the LLM config from a file and load the backend. Missing files,
/// disabled sections, and anything that fails to load resolve to `None`
/// (with a warning): the daemon always keeps serving the word tier.
fn load_llm_from(path: &std::path::Path) -> Option<SlowCfg> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            eprintln!("predictd: no {} (llm disabled)", path.display());
            return None;
        }
        Err(e) => {
            eprintln!(
                "predictd: cannot read {}: {e:#} (llm disabled)",
                path.display()
            );
            return None;
        }
    };
    let cfg = match LlmConfig::from_toml_str(&text) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!(
                "predictd: bad config {}: {e} (llm disabled)",
                path.display()
            );
            return None;
        }
    };
    if !cfg.enabled {
        eprintln!("predictd: llm disabled by config");
        return None;
    }
    maybe_load_llm(&cfg)
}

/// Load the backend for an enabled config; `None` with a warning on failure.
fn maybe_load_llm(cfg: &LlmConfig) -> Option<SlowCfg> {
    match LlamaBackend::load(cfg) {
        Ok(backend) => {
            eprintln!("predictd: llm enabled ({})", cfg.model_path);
            Some(SlowCfg {
                backend: Arc::new(backend),
                max_tokens: cfg.max_tokens,
                threshold: cfg.confidence_threshold,
            })
        }
        Err(e) => {
            eprintln!("predictd: llm disabled: {e}");
            None
        }
    }
}

/// Slow-tier work in flight for one connection.
#[derive(Debug)]
struct SlowJob {
    generation: u64,
    cancel: CancelToken,
}

/// Per-connection state, shared with slow-tier worker threads.
#[derive(Debug, Default)]
struct SharedConn {
    /// Newest generation seen; anything older is stale.
    newest_seen: u64,
    ctx: Option<ContextUpdate>,
    slow: Option<SlowJob>,
}

impl SharedConn {
    /// Record a generation; false when stale. A fresh generation supersedes
    /// older slow work (a same-generation duplicate does not).
    fn observe(&mut self, generation: u64) -> bool {
        if generation < self.newest_seen {
            return false;
        }
        if let Some(job) = &self.slow {
            if job.generation < generation {
                job.cancel.cancel();
            }
        }
        self.newest_seen = generation;
        true
    }

    /// Record a cancellation: slow jobs at or below it are aborted, and the
    /// generation is marked superseded.
    fn cancel_through(&mut self, generation: u64) {
        if generation > self.newest_seen {
            self.newest_seen = generation;
        }
        if let Some(job) = &self.slow {
            if job.generation <= generation {
                job.cancel.cancel();
            }
        }
    }
}

/// Work decided while holding the connection lock; I/O happens after.
enum Action {
    ReplyWord(Suggestion),
    SpawnSlow {
        generation: u64,
        cancel: CancelToken,
        before: String,
        style_id: String,
        cfg: SlowCfg,
    },
    None,
}

/// Build the daemon reply for one generation (fast tier).
fn suggest_for(model: &NgramModel, ctx: &ContextUpdate, generation: u64) -> Suggestion {
    let core_ctx = Context::new(
        ctx.app_id.clone(),
        ctx.before.clone(),
        ctx.after.clone(),
        ctx.sensitive,
        ResolvedStyle::new(ctx.style_id.clone()),
    );
    let candidates = model
        .complete_word(&core_ctx)
        .into_iter()
        .map(|c| ProtoCandidate {
            text: c.text,
            score: c.score,
        })
        .collect();
    Suggestion {
        generation,
        candidates,
        style_id: ctx.style_id.clone(),
    }
}

fn default_ctx() -> ContextUpdate {
    ContextUpdate {
        app_id: String::new(),
        before: String::new(),
        after: String::new(),
        sensitive: false,
        style_id: "default".to_string(),
    }
}

/// Send one message, serializing writers across connection + worker threads
/// so frames never interleave.
fn send_msg(writer: &Arc<Mutex<UnixStream>>, msg: &DaemonMsg) -> Result<()> {
    let mut stream = writer
        .lock()
        .map_err(|_| anyhow::anyhow!("writer lock poisoned"))?;
    write_daemon_msg(&mut *stream, msg).context("write daemon message")
}

/// Serve one client until it disconnects or the stream breaks.
fn handle_connection(
    stream: UnixStream,
    model: &NgramModel,
    slow: Option<SlowCfg>,
) -> Result<()> {
    let writer = Arc::new(Mutex::new(
        stream.try_clone().context("clone stream")?,
    ));
    let mut reader = stream;
    let shared = Arc::new(Mutex::new(SharedConn::default()));
    loop {
        let msg = match read_client_msg(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                if let predict_proto::ProtoError::Io(io) = &e {
                    if io.kind() == ErrorKind::UnexpectedEof {
                        return Ok(());
                    }
                }
                return Err(e).context("read client message");
            }
        };
        let action = {
            let mut guard = shared
                .lock()
                .map_err(|_| anyhow::anyhow!("connection lock poisoned"))?;
            match msg {
                ClientMsg::ContextUpdate(ctx) => {
                    guard.ctx = Some(ctx);
                    Action::None
                }
                ClientMsg::Suggest(SuggestRequest { generation }) => {
                    if !guard.observe(generation) {
                        Action::None
                    } else {
                        let ctx = guard.ctx.clone().unwrap_or_else(default_ctx);
                        Action::ReplyWord(suggest_for(model, &ctx, generation))
                    }
                }
                ClientMsg::SuggestSentence(SuggestRequest { generation }) => {
                    if !guard.observe(generation) {
                        Action::None
                    } else {
                        let duplicate = guard
                            .slow
                            .as_ref()
                            .is_some_and(|job| job.generation == generation);
                        match (duplicate, slow.clone()) {
                            (false, Some(cfg)) => {
                                let cancel = CancelToken::new();
                                let ctx = guard.ctx.clone().unwrap_or_else(default_ctx);
                                guard.slow = Some(SlowJob {
                                    generation,
                                    cancel: cancel.clone(),
                                });
                                Action::SpawnSlow {
                                    generation,
                                    cancel,
                                    before: ctx.before,
                                    style_id: ctx.style_id,
                                    cfg,
                                }
                            }
                            // Duplicate request, or slow tier disabled:
                            // silence, like a confidence gate.
                            _ => Action::None,
                        }
                    }
                }
                ClientMsg::Cancel(CancelMsg { generation }) => {
                    guard.cancel_through(generation);
                    Action::None
                }
            }
        };
        match action {
            Action::None => {}
            Action::ReplyWord(reply) => {
                send_msg(&writer, &DaemonMsg::Suggestion(reply))?;
            }
            Action::SpawnSlow {
                generation,
                cancel,
                before,
                style_id,
                cfg,
            } => {
                let writer = Arc::clone(&writer);
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let req = SentenceRequest {
                        before,
                        max_tokens: cfg.max_tokens,
                        confidence_threshold: cfg.threshold,
                    };
                    let result = cfg.backend.complete_sentence(&req, &cancel);
                    let fresh = {
                        let mut guard = match shared.lock() {
                            Ok(guard) => guard,
                            Err(_) => return,
                        };
                        if guard
                            .slow
                            .as_ref()
                            .is_some_and(|job| job.generation == generation)
                        {
                            guard.slow = None;
                        }
                        !cancel.is_cancelled() && guard.newest_seen == generation
                    };
                    if !fresh {
                        return;
                    }
                    if let Ok(Some(out)) = result {
                        let reply = SentenceSuggestion {
                            generation,
                            text: out.text,
                            confidence: out.confidence,
                            style_id,
                        };
                        let _ = send_msg(&writer, &DaemonMsg::Sentence(reply));
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use predict_llm::{SentenceOutput, StubBackend};
    use predict_proto::{read_daemon_msg, write_client_msg};
    use std::time::{Duration, Instant};

    fn test_model() -> NgramModel {
        NgramModel::from_text(BASE_CORPUS).unwrap()
    }

    fn ctx(before: &str) -> ContextUpdate {
        ContextUpdate {
            app_id: "test".to_string(),
            before: before.to_string(),
            after: String::new(),
            sensitive: false,
            style_id: "default".to_string(),
        }
    }

    fn stub_slow() -> SlowCfg {
        SlowCfg {
            backend: Arc::new(StubBackend::fixed("brown fox", -0.2)),
            max_tokens: 32,
            threshold: -1.5,
        }
    }

    #[test]
    fn suggest_for_answers_with_matching_generation() {
        let model = test_model();
        let reply = suggest_for(&model, &ctx("hello wo"), 9);
        assert_eq!(reply.generation, 9);
        assert_eq!(reply.style_id, "default");
        assert!(!reply.candidates.is_empty());
        assert_eq!(reply.candidates[0].text, "world");
    }

    #[test]
    fn suggest_for_respects_sensitive() {
        let model = test_model();
        let sensitive = ContextUpdate {
            sensitive: true,
            ..ctx("hello wo")
        };
        let reply = suggest_for(&model, &sensitive, 1);
        assert!(reply.candidates.is_empty());
    }

    #[test]
    fn stale_generations_are_ignored() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        assert!(state.observe(5)); // duplicate of newest still counts
        assert!(!state.observe(4)); // older is stale
        assert!(state.observe(6));
        assert!(!state.observe(5)); // now stale
    }

    #[test]
    fn fresh_generation_cancels_older_slow_job() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        let cancel = CancelToken::new();
        state.slow = Some(SlowJob {
            generation: 5,
            cancel: cancel.clone(),
        });
        assert!(state.observe(6));
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn cancel_through_aborts_current_slow_job() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        let cancel = CancelToken::new();
        state.slow = Some(SlowJob {
            generation: 5,
            cancel: cancel.clone(),
        });
        state.cancel_through(5);
        assert!(cancel.is_cancelled());
        // Late duplicates stay stale.
        assert!(!state.observe(4));
    }

    #[test]
    fn missing_config_disables_slow_tier() {
        let dir = std::env::temp_dir().join(format!("predict-test-{}", std::process::id()));
        let missing = dir.join("no-such.toml");
        assert!(load_llm_from(&missing).is_none());
    }

    #[test]
    fn bad_config_disables_slow_tier() {
        let dir = std::env::temp_dir().join(format!("predict-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.toml");
        std::fs::write(&path, "[llm\nbroken").unwrap();
        assert!(load_llm_from(&path).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    /// Full loopback: temp socket -> daemon thread -> framed reply.
    #[test]
    fn end_to_end_suggest_over_unix_socket() {
        let sock = std::env::temp_dir().join(format!("predictd-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let model = test_model();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &model, None).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("hello wo"))).unwrap();
        let start = Instant::now();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
            other => panic!("expected Suggestion, got {other:?}"),
        };
        let elapsed = start.elapsed();
        assert_eq!(reply.generation, 1);
        assert_eq!(reply.candidates[0].text, "world");
        assert!(
            elapsed < Duration::from_millis(500),
            "roundtrip too slow: {elapsed:?}"
        );

        // A stale generation gets no reply: the daemon skips it silently.
        write_client_msg(&mut client, &ClientMsg::Cancel(CancelMsg { generation: 2 })).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let stale = read_daemon_msg(&mut client);
        assert!(stale.is_err(), "stale generation got a reply: {stale:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Slow path over the socket with a stub backend.
    #[test]
    fn end_to_end_sentence_over_unix_socket() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-slow-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let model = test_model();
        let slow = stub_slow();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &model, Some(slow)).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Sentence(s) => s,
            other => panic!("expected Sentence, got {other:?}"),
        };
        assert_eq!(reply.generation, 1);
        assert_eq!(reply.text, "brown fox");
        assert_eq!(reply.confidence, -0.2);
        assert_eq!(reply.style_id, "default");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// A gated-out sentence stays silent (no reply, client times out).
    #[test]
    fn gated_sentence_gets_no_reply() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-gate-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let model = test_model();
        let slow = SlowCfg {
            threshold: 0.0, // stub confidence -0.2 never passes
            ..stub_slow()
        };

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &model, Some(slow)).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_err(), "gated sentence got a reply: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Slow backend that blocks in cancel-aware slices (deterministic abort).
    struct BlockingBackend;

    impl Backend for BlockingBackend {
        fn complete_sentence(
            &self,
            _req: &SentenceRequest,
            cancel: &CancelToken,
        ) -> Result<Option<SentenceOutput>, predict_llm::LlmError> {
            for _ in 0..10 {
                if cancel.is_cancelled() {
                    return Err(predict_llm::LlmError::Cancelled);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(Some(SentenceOutput {
                text: "slow result".to_string(),
                confidence: -0.1,
                time_to_first_token: Duration::from_millis(50),
                tokens_generated: 2,
            }))
        }

        fn name(&self) -> &str {
            "blocking-test"
        }
    }

    /// Cancelling in-flight slow work produces silence, not a stale reply.
    #[test]
    fn cancelled_sentence_gets_no_reply() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-cancel-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let model = test_model();
        let slow = SlowCfg {
            backend: Arc::new(BlockingBackend),
            max_tokens: 32,
            threshold: -1.5,
        };

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, &model, Some(slow)).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        // Abort before the 50 ms backend can finish (it polls every 5 ms).
        write_client_msg(&mut client, &ClientMsg::Cancel(CancelMsg { generation: 1 })).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_err(), "cancelled sentence got a reply: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }
}
