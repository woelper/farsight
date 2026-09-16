//! predict-cli: terminal test client for the prediction daemon.
//!
//! Type text, see word suggestions live, Tab accepts the top word.
//! Every keystroke bumps the generation and re-requests; replies for older
//! generations are ignored. A slow daemon never blocks typing: reads time
//! out after [`READ_TIMEOUT`] and render with no suggestions.

use anyhow::{Context as _, Result};
use crossterm::cursor::MoveTo;
use crossterm::event::{Event, KeyCode, KeyModifiers, read};
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
    enable_raw_mode,
};
use crossterm::{execute};
use predict_proto::{
    ClientMsg, ContextUpdate, DaemonMsg, ProtoCandidate, ProtoError, SuggestRequest,
    read_daemon_msg, socket_path, write_client_msg,
};
use std::io::{Stdout, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// How long to wait for the daemon before rendering without suggestions.
const READ_TIMEOUT: Duration = Duration::from_millis(300);

fn main() -> Result<()> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).with_context(|| {
        format!(
            "connect to predictd at {} — is predictd running?",
            path.display()
        )
    })?;
    stream
        .set_read_timeout(Some(READ_TIMEOUT))
        .context("set read timeout")?;

    let _terminal = TerminalGuard::enter()?;
    let mut stdout = std::io::stdout();
    let mut buffer = String::new();
    let mut committed: Vec<String> = Vec::new();
    let mut generation: u64 = 0;
    let mut state = refresh(&mut stream, &buffer, &mut generation)?;
    render(&mut stdout, &buffer, &state, &committed)?;

    loop {
        if let Event::Key(key) = read()? {
            match key.code {
                KeyCode::Esc => break,
                KeyCode::Char('c')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    break;
                }
                KeyCode::Char(c) => {
                    buffer.push(c);
                    state = refresh(&mut stream, &buffer, &mut generation)?;
                }
                KeyCode::Backspace => {
                    buffer.pop();
                    state = refresh(&mut stream, &buffer, &mut generation)?;
                }
                KeyCode::Tab => {
                    if let Some(top) = state.candidates.first() {
                        buffer = accept_completion(&buffer, &top.text);
                        state = refresh(&mut stream, &buffer, &mut generation)?;
                    }
                }
                KeyCode::Enter => {
                    committed.push(std::mem::take(&mut buffer));
                    state = refresh(&mut stream, &buffer, &mut generation)?;
                }
                _ => {}
            }
        }
        render(&mut stdout, &buffer, &state, &committed)?;
    }
    Ok(())
}

/// Raw mode + alternate screen, restored on drop (including error paths).
struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        execute!(std::io::stdout(), EnterAlternateScreen).context("enter alternate screen")?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
        let _ = disable_raw_mode();
    }
}

/// Current suggestion view.
struct SuggestionState {
    generation: u64,
    candidates: Vec<ProtoCandidate>,
    style_id: String,
}

/// Bump the generation, send context + request, read the reply.
///
/// A daemon timeout yields empty suggestions instead of an error, so typing
/// never stalls. Other transport errors propagate (e.g. daemon died).
fn refresh(
    stream: &mut UnixStream,
    buffer: &str,
    generation: &mut u64,
) -> Result<SuggestionState> {
    *generation = generation.wrapping_add(1);
    let gen = *generation;
    write_client_msg(stream, &ClientMsg::ContextUpdate(build_context_update(buffer)))
        .context("send context")?;
    write_client_msg(stream, &ClientMsg::Suggest(SuggestRequest {
        generation: gen,
    }))
    .context("send suggest request")?;

    match read_daemon_msg(stream) {
        Ok(DaemonMsg::Suggestion(s)) => {
            if s.generation != gen {
                return Ok(SuggestionState {
                    generation: gen,
                    candidates: Vec::new(),
                    style_id: "default".to_string(),
                });
            }
            Ok(SuggestionState {
                generation: gen,
                candidates: s.candidates,
                style_id: s.style_id,
            })
        }
        Err(ProtoError::Io(e))
            if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock =>
        {
            Ok(SuggestionState {
                generation: gen,
                candidates: Vec::new(),
                style_id: "default".to_string(),
            })
        }
        Err(e) => Err(e).context("read suggestion"),
    }
}

/// Build the proto context for the current input line.
fn build_context_update(buffer: &str) -> ContextUpdate {
    ContextUpdate {
        app_id: "predict-cli".to_string(),
        before: buffer.to_string(),
        after: String::new(),
        sensitive: false,
        style_id: "default".to_string(),
    }
}

/// Accept a word suggestion: replace the in-progress fragment (if any) with
/// the candidate and append a space so typing continues with the next word.
fn accept_completion(buffer: &str, candidate: &str) -> String {
    let prefix_len: usize = buffer
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric())
        .count();
    let char_count: usize = buffer.chars().count();
    let stem: String = buffer
        .chars()
        .take(char_count.saturating_sub(prefix_len))
        .collect();
    format!("{stem}{candidate} ")
}

fn render(
    stdout: &mut Stdout,
    buffer: &str,
    state: &SuggestionState,
    committed: &[String],
) -> Result<()> {
    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    writeln!(stdout, "predict-cli — Tab: accept word · Enter: commit · Esc: quit")?;
    writeln!(stdout, "> {buffer}")?;
    if state.candidates.is_empty() {
        writeln!(stdout, "(no suggestions, gen {})", state.generation)?;
    } else {
        let list: Vec<&str> = state
            .candidates
            .iter()
            .map(|c| c.text.as_str())
            .collect();
        writeln!(
            stdout,
            "suggestions [{}]: {}",
            state.style_id,
            list.join("  ")
        )?;
    }
    for line in committed {
        writeln!(stdout, "committed: {line}")?;
    }
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_mid_word_replaces_fragment() {
        assert_eq!(accept_completion("hello wo", "world"), "hello world ");
    }

    #[test]
    fn accept_at_boundary_appends() {
        assert_eq!(accept_completion("hello ", "world"), "hello world ");
        assert_eq!(accept_completion("", "hello"), "hello ");
    }

    #[test]
    fn accept_handles_unicode_fragment() {
        assert_eq!(accept_completion("grü", "grüße"), "grüße ");
    }

    #[test]
    fn context_carries_buffer_and_app_id() {
        let ctx = build_context_update("hello wo");
        assert_eq!(ctx.app_id, "predict-cli");
        assert_eq!(ctx.before, "hello wo");
        assert!(ctx.after.is_empty());
        assert!(!ctx.sensitive);
        assert_eq!(ctx.style_id, "default");
    }
}
