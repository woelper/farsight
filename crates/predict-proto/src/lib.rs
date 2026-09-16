//! IPC messages between predictd, frontends, and test clients.
//!
//! Versioned envelopes serialized with `postcard`, sent as
//! `u32`-length-prefixed frames over a Unix socket. See ADR 0003.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::PathBuf;
use thiserror::Error;

/// Protocol version. Any breaking change bumps this; readers reject frames
/// whose version differs.
pub const PROTOCOL_VERSION: u16 = 1;

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

/// Client (frontend) to daemon messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    /// Latest typing context.
    ContextUpdate(ContextUpdate),
    /// Ask for suggestions.
    Suggest(SuggestRequest),
    /// Cancel a generation.
    Cancel(CancelMsg),
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

/// Daemon to client messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DaemonMsg {
    /// Suggestions for a generation.
    Suggestion(Suggestion),
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
                ProtoError::VersionMismatch { expected: 1, got: 2 }
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
}
