//! predictd: per-user prediction daemon.
//!
//! Listens on a Unix socket ([`predict_proto::socket_path`]), answers
//! `Suggest` requests with fast-tier completions. A newer generation cancels
//! older work: both sides track the newest generation seen and ignore stale
//! ones. One `std::thread` per connection — the work is synchronous and
//! sub-millisecond, so no async runtime (see ADR 0003).

use anyhow::{Context as _, Result};
use predict_core::{Context, Predictor as _, ResolvedStyle};
use predict_ngram::NgramModel;
use predict_proto::{
    CancelMsg, ClientMsg, ContextUpdate, DaemonMsg, ProtoCandidate, SuggestRequest, Suggestion,
    read_client_msg, socket_path, write_daemon_msg,
};
use std::io::ErrorKind;
use std::os::unix::net::{UnixListener, UnixStream};

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
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &model) {
                        eprintln!("predictd: connection error: {e:#}");
                    }
                });
            }
            Err(e) => eprintln!("predictd: accept error: {e:#}"),
        }
    }
    Ok(())
}

/// Per-connection state: newest generation seen; anything older is stale.
#[derive(Debug, Default)]
struct ConnState {
    newest_seen: u64,
    ctx: Option<ContextUpdate>,
}

impl ConnState {
    /// Record a generation; returns false when it is stale (an older or
    /// already-cancelled generation).
    fn observe(&mut self, generation: u64) -> bool {
        if generation < self.newest_seen {
            return false;
        }
        self.newest_seen = generation;
        true
    }
}

/// Build the daemon reply for one generation.
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

/// Serve one client until it disconnects or the stream breaks.
fn handle_connection(stream: UnixStream, model: &NgramModel) -> Result<()> {
    let mut stream = stream;
    let mut state = ConnState::default();
    loop {
        match read_client_msg(&mut stream) {
            Ok(ClientMsg::ContextUpdate(ctx)) => {
                state.ctx = Some(ctx);
            }
            Ok(ClientMsg::Suggest(SuggestRequest { generation })) => {
                if !state.observe(generation) {
                    continue;
                }
                let fallback = ContextUpdate {
                    app_id: String::new(),
                    before: String::new(),
                    after: String::new(),
                    sensitive: false,
                    style_id: "default".to_string(),
                };
                let ctx = state.ctx.as_ref().unwrap_or(&fallback);
                let reply = suggest_for(model, ctx, generation);
                write_daemon_msg(&mut stream, &DaemonMsg::Suggestion(reply))
                    .context("write suggestion")?;
            }
            Ok(ClientMsg::Cancel(CancelMsg { generation })) => {
                // No reply; just mark the generation superseded so a late
                // duplicate arriving afterwards is ignored too.
                state.observe(generation);
            }
            Err(e) => {
                if let predict_proto::ProtoError::Io(io) = &e {
                    if io.kind() == ErrorKind::UnexpectedEof {
                        return Ok(());
                    }
                }
                return Err(e).context("read client message");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let mut state = ConnState::default();
        assert!(state.observe(5));
        assert!(state.observe(5)); // duplicate of newest still counts
        assert!(!state.observe(4)); // older is stale
        assert!(state.observe(6));
        assert!(!state.observe(5)); // now stale
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
            handle_connection(stream, &model).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::ContextUpdate(ctx("hello wo")),
        )
        .unwrap();
        let start = Instant::now();
        write_client_msg(&mut client, &ClientMsg::Suggest(SuggestRequest {
            generation: 1,
        }))
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
        };
        let elapsed = start.elapsed();
        assert_eq!(reply.generation, 1);
        assert_eq!(reply.candidates[0].text, "world");
        assert!(
            elapsed < Duration::from_millis(500),
            "roundtrip too slow: {elapsed:?}"
        );

        // A stale generation gets no reply: the daemon skips it silently.
        write_client_msg(&mut client, &ClientMsg::Cancel(CancelMsg { generation: 2 }))
            .unwrap();
        write_client_msg(&mut client, &ClientMsg::Suggest(SuggestRequest {
            generation: 1,
        }))
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
}
