//! IPC messages between predictd, frontends, and test clients.
//!
//! Versioned envelopes serialized with `postcard`, sent as
//! `u32`-length-prefixed frames over a Unix socket. See ADR 0003 (v1) and
//! ADR 0004 (v2: sentence messages for the slow tier).

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;
use thiserror::Error;

/// Protocol version. Any breaking change bumps this; readers reject frames
/// whose version differs.
pub const PROTOCOL_VERSION: u16 = 3;

/// Maximum frame payload in bytes (postcard body, excluding length prefix).
pub const MAX_FRAME_BYTES: usize = 256 * 1024;

/// Filesystem path of the per-user daemon socket.
///
/// Prefers `$XDG_RUNTIME_DIR/predict/predictd.sock`, falls back to a
/// user-scoped file in the temp dir so two users never share a socket.
pub fn socket_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("predict/predictd.sock");
        }
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    std::env::temp_dir().join(format!("predictd-{user}.sock"))
}

/// Errors from framing and (de)serialization.
#[derive(Debug, Error)]
pub enum ProtoError {
    /// Underlying transport error (including peer disconnect).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Postcard encode/decode failure.
    #[error("codec error: {0}")]
    Codec(#[from] postcard::Error),
    /// Frame payload exceeds [`MAX_FRAME_BYTES`].
    #[error("frame too large: {len} bytes")]
    FrameTooLarge {
        /// Advertised payload length.
        len: u32,
    },
    /// Frame carries an unexpected protocol version.
    #[error("protocol version mismatch: got {got}, expected {expected}")]
    VersionMismatch {
        /// Version this build speaks.
        expected: u16,
        /// Version on the wire.
        got: u16,
    },
}

/// Typing context sent by a frontend (mirrors `predict_core::Context`;
/// kept as plain data here so frontends stay light — see ADR 0001).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextUpdate {
    /// Frontend-provided app id (e.g. `"predict-cli"`).
    pub app_id: String,
    /// Text before the cursor.
    pub before: String,
    /// Text after the cursor (may be empty).
    pub after: String,
    /// Sensitive field: no suggestions, no learning.
    pub sensitive: bool,
    /// Active style id.
    pub style_id: String,
}

/// Request suggestions for the latest `ContextUpdate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuggestRequest {
    /// Monotonic per-keystroke counter; a newer generation cancels older work.
    pub generation: u64,
}

/// Cancel in-flight work for a generation (used by the M3 slow tier;
/// recorded by the daemon already in M2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelMsg {
    /// Generation being cancelled.
    pub generation: u64,
}

/// Settled text from the frontend (committed after a pause or field leave —
/// never keystrokes). The daemon stores it unless learning is paused or the
/// field is sensitive, and answers with [`LearningState`] so frontends can
/// show live store state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitText {
    /// The settled text.
    pub text: String,
    /// Active style id (recorded for M5).
    pub style_id: String,
    /// Sensitive field: must not be stored.
    pub sensitive: bool,
}

/// Pause or resume learning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetLearning {
    /// False pauses learning (commits ignored); true resumes it.
    pub enabled: bool,
}

/// Client (frontend) to daemon messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    /// Latest typing context.
    ContextUpdate(ContextUpdate),
    /// Ask for word suggestions (fast tier, answered synchronously).
    Suggest(SuggestRequest),
    /// Ask for a sentence continuation (slow tier, answered asynchronously;
    /// silence means stale, cancelled, or gated).
    SuggestSentence(SuggestRequest),
    /// Cancel a generation.
    Cancel(CancelMsg),
    /// Store settled text (personal memory).
    CommitText(CommitText),
    /// Remove all personal data (answered with [`LearningState`]).
    ForgetAll,
    /// Pause/resume learning (answered with [`LearningState`]).
    SetLearning(SetLearning),
}

/// One suggestion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtoCandidate {
    /// Suggested word.
    pub text: String,
    /// Score, higher is better.
    pub score: f32,
}

/// Suggestions for one generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    /// Generation this answers.
    pub generation: u64,
    /// Up to 5 candidates, best first.
    pub candidates: Vec<ProtoCandidate>,
    /// Style id the suggestions were made under.
    pub style_id: String,
}

/// Sentence continuation for one generation (slow tier).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SentenceSuggestion {
    /// Generation this answers.
    pub generation: u64,
    /// Continuation AFTER the context text (may be empty when the daemon
    /// has nothing worth showing, e.g. LLM disabled).
    pub text: String,
    /// Mean token logprob of the shown text (higher is better, ≤ 0).
    pub confidence: f32,
    /// Style id the suggestion was made under.
    pub style_id: String,
}

/// Learning state after [`ClientMsg::ForgetAll`]/[`ClientMsg::SetLearning`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearningState {
    /// Learning currently enabled (false while paused).
    pub enabled: bool,
    /// Stored documents (0 right after forget-all).
    pub documents: u64,
}

/// Daemon to client messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DaemonMsg {
    /// Word suggestions for a generation.
    Suggestion(Suggestion),
    /// Sentence continuation for a generation.
    Sentence(SentenceSuggestion),
    /// Learning state after a control command.
    LearningState(LearningState),
}

/// Version envelope around every frame payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Envelope<M> {
    version: u16,
    msg: M,
}

fn write_frame(writer: &mut impl Write, bytes: &[u8]) -> Result<(), ProtoError> {
    if bytes.len() > MAX_FRAME_BYTES {
        let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        return Err(ProtoError::FrameTooLarge { len });
    }
    let len =
        u32::try_from(bytes.len()).map_err(|_| ProtoError::FrameTooLarge { len: u32::MAX })?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<Vec<u8>, ProtoError> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf);
    if len as usize > MAX_FRAME_BYTES {
        return Err(ProtoError::FrameTooLarge { len });
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_envelope<M: Serialize>(writer: &mut impl Write, msg: &M) -> Result<(), ProtoError> {
    let env = Envelope {
        version: PROTOCOL_VERSION,
        msg,
    };
    let bytes = postcard::to_stdvec(&env)?;
    write_frame(writer, &bytes)
}

fn read_envelope<M>(reader: &mut impl Read) -> Result<M, ProtoError>
where
    M: serde::de::DeserializeOwned,
{
    let bytes = read_frame(reader)?;
    let env: Envelope<M> = postcard::from_bytes(&bytes)?;
    if env.version != PROTOCOL_VERSION {
        return Err(ProtoError::VersionMismatch {
            expected: PROTOCOL_VERSION,
            got: env.version,
        });
    }
    Ok(env.msg)
}

/// Send one client-to-daemon message.
pub fn write_client_msg(writer: &mut impl Write, msg: &ClientMsg) -> Result<(), ProtoError> {
    write_envelope(writer, msg)
}

/// Receive one client-to-daemon message.
pub fn read_client_msg(reader: &mut impl Read) -> Result<ClientMsg, ProtoError> {
    read_envelope(reader)
}

/// Send one daemon-to-client message.
pub fn write_daemon_msg(writer: &mut impl Write, msg: &DaemonMsg) -> Result<(), ProtoError> {
    write_envelope(writer, msg)
}

/// Receive one daemon-to-client message.
pub fn read_daemon_msg(reader: &mut impl Read) -> Result<DaemonMsg, ProtoError> {
    read_envelope(reader)
}

/// True for read timeouts (no data yet), as opposed to fatal errors.
/// Note: on Linux an expired socket timeout surfaces as `WouldBlock`.
pub fn is_timeout(err: &ProtoError) -> bool {
    matches!(
        err,
        ProtoError::Io(e)
            if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock
    )
}

/// Read until the word reply for `generation` arrives or `timeout` passes.
/// Stray messages (stale generations, late sentences) are discarded;
/// `None` means the daemon didn't answer in time.
pub fn read_word_reply(
    stream: &mut std::os::unix::net::UnixStream,
    generation: u64,
    timeout: std::time::Duration,
) -> Result<Option<Suggestion>, ProtoError> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        stream
            .set_read_timeout(Some(remaining))
            .map_err(ProtoError::Io)?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::Suggestion(s)) if s.generation == generation => {
                return Ok(Some(s));
            }
            Ok(_) => {} // stale or sentence: discard, keep waiting
            Err(e) if is_timeout(&e) => {
                if std::time::Instant::now() >= deadline {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// Read until the sentence reply for `generation` arrives or `timeout`
/// passes. Returns `None` on timeout (still computing, gated, disabled,
/// or filtered).
pub fn read_sentence_reply(
    stream: &mut std::os::unix::net::UnixStream,
    generation: u64,
    timeout: std::time::Duration,
) -> Result<Option<SentenceSuggestion>, ProtoError> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        stream
            .set_read_timeout(Some(remaining))
            .map_err(ProtoError::Io)?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::Sentence(s)) if s.generation == generation && !s.text.is_empty() => {
                return Ok(Some(s));
            }
            Ok(_) => {} // stale or word reply: discard, keep waiting
            Err(e) if is_timeout(&e) => {
                if std::time::Instant::now() >= deadline {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_ctx() -> ContextUpdate {
        ContextUpdate {
            app_id: "predict-cli".to_string(),
            before: "hello wo".to_string(),
            after: String::new(),
            sensitive: false,
            style_id: "default".to_string(),
        }
    }

    fn roundtrip_client(msg: &ClientMsg) {
        let mut buf = Cursor::new(Vec::new());
        write_client_msg(&mut buf, msg).unwrap();
        buf.set_position(0);
        assert_eq!(read_client_msg(&mut buf).unwrap(), *msg);
    }

    #[test]
    fn context_update_roundtrips() {
        roundtrip_client(&ClientMsg::ContextUpdate(sample_ctx()));
    }

    #[test]
    fn suggest_and_cancel_roundtrip() {
        roundtrip_client(&ClientMsg::Suggest(SuggestRequest { generation: 41 }));
        roundtrip_client(&ClientMsg::SuggestSentence(SuggestRequest { generation: 42 }));
        roundtrip_client(&ClientMsg::Cancel(CancelMsg { generation: 41 }));
    }

    #[test]
    fn suggestion_roundtrips() {
        let msg = DaemonMsg::Suggestion(Suggestion {
            generation: 7,
            candidates: vec![
                ProtoCandidate {
                    text: "world".to_string(),
                    score: 3.0,
                },
                ProtoCandidate {
                    text: "word".to_string(),
                    score: 1.0,
                },
            ],
            style_id: "default".to_string(),
        });
        let mut buf = Cursor::new(Vec::new());
        write_daemon_msg(&mut buf, &msg).unwrap();
        buf.set_position(0);
        assert_eq!(read_daemon_msg(&mut buf).unwrap(), msg);
    }

    #[test]
    fn sentence_suggestion_roundtrips() {
        let msg = DaemonMsg::Sentence(SentenceSuggestion {
            generation: 3,
            text: "brown fox.".to_string(),
            confidence: -0.4,
            style_id: "default".to_string(),
        });
        let mut buf = Cursor::new(Vec::new());
        write_daemon_msg(&mut buf, &msg).unwrap();
        buf.set_position(0);
        assert_eq!(read_daemon_msg(&mut buf).unwrap(), msg);
    }

    #[test]
    fn control_messages_roundtrip() {
        roundtrip_client(&ClientMsg::CommitText(CommitText {
            text: "settled line".to_string(),
            style_id: "default".to_string(),
            sensitive: false,
        }));
        roundtrip_client(&ClientMsg::ForgetAll);
        roundtrip_client(&ClientMsg::SetLearning(SetLearning { enabled: false }));
        let msg = DaemonMsg::LearningState(LearningState {
            enabled: true,
            documents: 12,
        });
        let mut buf = Cursor::new(Vec::new());
        write_daemon_msg(&mut buf, &msg).unwrap();
        buf.set_position(0);
        assert_eq!(read_daemon_msg(&mut buf).unwrap(), msg);
    }

    #[test]
    fn multiple_frames_share_one_stream() {
        let mut buf = Cursor::new(Vec::new());
        let first = ClientMsg::ContextUpdate(sample_ctx());
        let second = ClientMsg::Suggest(SuggestRequest { generation: 2 });
        write_client_msg(&mut buf, &first).unwrap();
        write_client_msg(&mut buf, &second).unwrap();
        buf.set_position(0);
        assert_eq!(read_client_msg(&mut buf).unwrap(), first);
        assert_eq!(read_client_msg(&mut buf).unwrap(), second);
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let env = Envelope {
            version: PROTOCOL_VERSION + 1,
            msg: ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        };
        let bytes = postcard::to_stdvec(&env).unwrap();
        let mut buf = Cursor::new(Vec::new());
        write_frame(&mut buf, &bytes).unwrap();
        buf.set_position(0);
        let err = read_client_msg(&mut buf).unwrap_err();
        assert!(
            matches!(
                err,
                ProtoError::VersionMismatch { expected, got }
                if expected == PROTOCOL_VERSION && got == PROTOCOL_VERSION + 1
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn oversized_frame_is_rejected_before_allocating() {
        let mut buf = Cursor::new(Vec::new());
        buf.write_all(&(MAX_FRAME_BYTES as u32 + 1).to_le_bytes())
            .unwrap();
        buf.set_position(0);
        let err = read_client_msg(&mut buf).unwrap_err();
        assert!(
            matches!(err, ProtoError::FrameTooLarge { .. }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn truncated_stream_is_an_io_error() {
        let mut buf = Cursor::new(vec![0x02, 0x00]);
        let err = read_client_msg(&mut buf).unwrap_err();
        assert!(
            matches!(err, ProtoError::Io(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn socket_path_is_user_scoped() {
        let path = socket_path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        // XDG path ends in predictd.sock; temp-dir fallback embeds $USER.
        assert!(
            name == "predictd.sock" || name.starts_with("predictd-"),
            "unexpected socket name: {name}"
        );
    }

    fn word_msg(gen: u64, word: &str) -> DaemonMsg {
        DaemonMsg::Suggestion(Suggestion {
            generation: gen,
            candidates: vec![ProtoCandidate {
                text: word.to_string(),
                score: 1.0,
            }],
            style_id: "default".to_string(),
        })
    }

    fn sentence_msg(gen: u64, text: &str) -> DaemonMsg {
        DaemonMsg::Sentence(SentenceSuggestion {
            generation: gen,
            text: text.to_string(),
            confidence: -0.5,
            style_id: "default".to_string(),
        })
    }

    #[test]
    fn word_reader_skips_stray_sentence() {
        use std::os::unix::net::UnixStream;
        let (mut client, mut server) = UnixStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(7, "late")).unwrap();
        write_daemon_msg(&mut server, &word_msg(9, "world")).unwrap();
        let reply = read_word_reply(&mut client, 9, std::time::Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(reply.generation, 9);
        assert_eq!(reply.candidates[0].text, "world");
    }

    #[test]
    fn word_reader_times_out_to_none() {
        use std::os::unix::net::UnixStream;
        let (mut client, _server) = UnixStream::pair().unwrap();
        let reply = read_word_reply(&mut client, 3, std::time::Duration::from_millis(50)).unwrap();
        assert!(reply.is_none());
    }

    #[test]
    fn sentence_reader_skips_stray_suggestion() {
        use std::os::unix::net::UnixStream;
        let (mut client, mut server) = UnixStream::pair().unwrap();
        write_daemon_msg(&mut server, &word_msg(9, "world")).unwrap();
        write_daemon_msg(&mut server, &sentence_msg(9, " wide.")).unwrap();
        let reply =
            read_sentence_reply(&mut client, 9, std::time::Duration::from_secs(5)).unwrap().unwrap();
        assert_eq!(reply.text, " wide.");
    }

    #[test]
    fn sentence_reader_times_out_to_none() {
        use std::os::unix::net::UnixStream;
        let (mut client, _server) = UnixStream::pair().unwrap();
        let reply =
            read_sentence_reply(&mut client, 3, std::time::Duration::from_millis(50)).unwrap();
        assert!(reply.is_none());
    }

    #[test]
    fn timeout_predicate_covers_socket_timeouts() {
        use std::os::unix::net::UnixStream;
        let (mut client, _server) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_millis(10)))
            .unwrap();
        let err = read_daemon_msg(&mut client).unwrap_err();
        assert!(is_timeout(&err), "expected timeout, got {err:?}");
        assert!(!is_timeout(&ProtoError::FrameTooLarge { len: 1 }));
    }
}
