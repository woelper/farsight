//! predict-cli: terminal test client for the prediction daemon.
//!
//! Type text, see word suggestions live, Tab accepts the top word. After a
//! 200 ms pause the daemon's slow tier is asked for a sentence continuation,
//! shown grey after the cursor; Ctrl+Right accepts the whole sentence.
//! Every keystroke bumps the generation and cancels superseded work; replies
//! for older generations are ignored. A slow daemon never blocks typing:
//! reads time out and render proceeds without suggestions.

use anyhow::{Context as _, Result};
use crossterm::cursor::MoveTo;
use crossterm::event::{Event, KeyCode, KeyModifiers, poll, read};
use crossterm::style::{Color, ResetColor, SetForegroundColor};
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
    enable_raw_mode,
};
use crossterm::execute;
use predict_proto::{
    CancelMsg, ClientMsg, ContextUpdate, DaemonMsg, ProtoCandidate, ProtoError, SuggestRequest,
    read_daemon_msg, socket_path, write_client_msg,
};
use std::io::{Stdout, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

/// How long to wait for the daemon before rendering without suggestions.
const WORD_TIMEOUT: Duration = Duration::from_millis(300);
/// Idle time before asking for a sentence continuation.
const PAUSE: Duration = Duration::from_millis(200);
/// How long an idle cycle waits for a slow-tier reply before polling keys.
const SENTENCE_TRY: Duration = Duration::from_millis(50);

fn main() -> Result<()> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).with_context(|| {
        format!(
            "connect to predictd at {} — is predictd running?",
            path.display()
        )
    })?;

    let _terminal = TerminalGuard::enter()?;
    let mut stdout = std::io::stdout();
    let mut buffer = String::new();
    let mut committed: Vec<String> = Vec::new();
    let mut generation: u64 = 0;
    let mut sentence_req: Option<u64> = None;
    let mut sentence: Option<String> = None;
    let mut state = on_buffer_changed(
        &mut stream,
        &buffer,
        &mut generation,
        &mut sentence_req,
        &mut sentence,
    )?;
    render(&mut stdout, &buffer, &state, &sentence, &committed)?;

    loop {
        if poll(PAUSE)? {
            let mut mutated = false;
            if let Event::Key(key) = read()? {
                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        break;
                    }
                    KeyCode::Char(c) => {
                        buffer.push(c);
                        mutated = true;
                    }
                    KeyCode::Backspace => {
                        buffer.pop();
                        mutated = true;
                    }
                    KeyCode::Tab => {
                        if let Some(top) = state.candidates.first() {
                            buffer = accept_completion(&buffer, &top.text);
                            mutated = true;
                        }
                    }
                    KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        if let Some(text) = sentence.take() {
                            buffer = accept_sentence(&buffer, &text);
                            mutated = true;
                        }
                    }
                    KeyCode::Enter => {
                        committed.push(std::mem::take(&mut buffer));
                        mutated = true;
                    }
                    _ => {}
                }
            }
            if mutated {
                state = on_buffer_changed(
                    &mut stream,
                    &buffer,
                    &mut generation,
                    &mut sentence_req,
                    &mut sentence,
                )?;
            }
        } else if !buffer.is_empty() && sentence.is_none() {
            // Typing pause: ask for (or collect) a sentence continuation.
            if sentence_req != Some(generation) {
                request_sentence(&mut stream, &buffer, generation)?;
                sentence_req = Some(generation);
            }
            if let Some(text) = read_sentence_reply(&mut stream, generation, SENTENCE_TRY)? {
                sentence = Some(text);
            }
        }
        render(&mut stdout, &buffer, &state, &sentence, &committed)?;
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

/// Current word-suggestion view.
struct SuggestionState {
    generation: u64,
    candidates: Vec<ProtoCandidate>,
    style_id: String,
}

/// True for read timeouts (no data yet), as opposed to fatal errors.
fn timed_out(err: &ProtoError) -> bool {
    matches!(
        err,
        ProtoError::Io(e)
            if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock
    )
}

/// Read until the word reply for `generation` arrives or `timeout` passes.
/// Stray messages (stale generations, late sentences) are discarded.
fn read_words_reply(
    stream: &mut UnixStream,
    generation: u64,
    timeout: Duration,
) -> Result<SuggestionState> {
    let empty = || SuggestionState {
        generation,
        candidates: Vec::new(),
        style_id: "default".to_string(),
    };
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        stream
            .set_read_timeout(Some(remaining))
            .context("set read timeout")?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::Suggestion(s)) if s.generation == generation => {
                return Ok(SuggestionState {
                    generation,
                    candidates: s.candidates,
                    style_id: s.style_id,
                });
            }
            Ok(_) => {} // stale or sentence: discard, keep waiting
            Err(e) if timed_out(&e) => {
                if Instant::now() >= deadline {
                    return Ok(empty());
                }
            }
            Err(e) => return Err(e).context("read suggestion"),
        }
    }
}

/// Read until the sentence reply for `generation` arrives or `timeout`
/// passes. Returns `None` on timeout (still computing, gated, or disabled).
fn read_sentence_reply(
    stream: &mut UnixStream,
    generation: u64,
    timeout: Duration,
) -> Result<Option<String>> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        stream
            .set_read_timeout(Some(remaining))
            .context("set read timeout")?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::Sentence(s)) if s.generation == generation && !s.text.is_empty() => {
                return Ok(Some(s.text));
            }
            Ok(_) => {} // stale or word reply: discard, keep waiting
            Err(e) if timed_out(&e) => {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e).context("read sentence"),
        }
    }
}

/// The buffer changed: cancel superseded slow work, bump the generation,
/// clear the sentence, and refresh word suggestions.
fn on_buffer_changed(
    stream: &mut UnixStream,
    buffer: &str,
    generation: &mut u64,
    sentence_req: &mut Option<u64>,
    sentence: &mut Option<String>,
) -> Result<SuggestionState> {
    if let Some(old) = sentence_req.take() {
        write_client_msg(stream, &ClientMsg::Cancel(CancelMsg { generation: old }))
            .context("send cancel")?;
    }
    *sentence = None;
    *generation = generation.wrapping_add(1);
    let gen = *generation;
    write_client_msg(stream, &ClientMsg::ContextUpdate(build_context_update(buffer)))
        .context("send context")?;
    write_client_msg(
        stream,
        &ClientMsg::Suggest(SuggestRequest { generation: gen }),
    )
    .context("send suggest request")?;
    read_words_reply(stream, gen, WORD_TIMEOUT)
}

/// Ask for a sentence continuation for the current generation (same
/// generation as the word request: it refines, not supersedes).
fn request_sentence(stream: &mut UnixStream, buffer: &str, generation: u64) -> Result<()> {
    write_client_msg(stream, &ClientMsg::ContextUpdate(build_context_update(buffer)))
        .context("send context")?;
    write_client_msg(
        stream,
        &ClientMsg::SuggestSentence(SuggestRequest { generation }),
    )
    .context("send sentence request")?;
    Ok(())
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

/// Accept a sentence suggestion: append it verbatim (it already continues
/// the cursor, including any needed space).
fn accept_sentence(buffer: &str, sentence: &str) -> String {
    format!("{buffer}{sentence}")
}

fn render(
    stdout: &mut Stdout,
    buffer: &str,
    state: &SuggestionState,
    sentence: &Option<String>,
    committed: &[String],
) -> Result<()> {
    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    writeln!(
        stdout,
        "predict-cli — Tab: accept word · Ctrl+→: accept sentence · Enter: commit · Esc: quit"
    )?;
    write!(stdout, "> {buffer}")?;
    if let Some(text) = sentence {
        execute!(stdout, SetForegroundColor(Color::DarkGrey))?;
        write!(stdout, "{text}")?;
        execute!(stdout, ResetColor)?;
        writeln!(stdout)?;
    } else {
        writeln!(stdout)?;
    }
    if state.candidates.is_empty() {
        writeln!(stdout, "(no word suggestions, gen {})", state.generation)?;
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
    use predict_proto::{SentenceSuggestion, Suggestion, write_daemon_msg};
    use std::os::unix::net::UnixStream as PairStream;

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
    fn accept_sentence_appends_verbatim() {
        assert_eq!(accept_sentence("the quick brown", " fox."), "the quick brown fox.");
        assert_eq!(accept_sentence("the qui", "ck."), "the quick.");
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

    fn suggestion_msg(gen: u64) -> DaemonMsg {
        DaemonMsg::Suggestion(Suggestion {
            generation: gen,
            candidates: vec![predict_proto::ProtoCandidate {
                text: "world".to_string(),
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
    fn words_reply_skips_stray_sentence() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(7, "late")).unwrap();
        write_daemon_msg(&mut server, &suggestion_msg(9)).unwrap();
        let state = read_words_reply(&mut client, 9, Duration::from_secs(5)).unwrap();
        assert_eq!(state.generation, 9);
        assert_eq!(state.candidates[0].text, "world");
    }

    #[test]
    fn words_reply_times_out_empty() {
        let (mut client, _server) = PairStream::pair().unwrap();
        let state = read_words_reply(&mut client, 3, Duration::from_millis(50)).unwrap();
        assert!(state.candidates.is_empty());
    }

    #[test]
    fn sentence_reply_skips_stray_suggestion() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &suggestion_msg(9)).unwrap();
        write_daemon_msg(&mut server, &sentence_msg(9, " fox.")).unwrap();
        let text = read_sentence_reply(&mut client, 9, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(text, " fox.");
    }

    #[test]
    fn sentence_reply_times_out_to_none() {
        let (mut client, _server) = PairStream::pair().unwrap();
        let text = read_sentence_reply(&mut client, 3, Duration::from_millis(50)).unwrap();
        assert!(text.is_none());
    }

    #[test]
    fn stale_replies_are_ignored() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(8, "old")).unwrap();
        write_daemon_msg(&mut server, &suggestion_msg(8)).unwrap();
        // Wanting gen 9: both stale frames are discarded, then timeout.
        let state = read_words_reply(&mut client, 9, Duration::from_millis(50)).unwrap();
        assert!(state.candidates.is_empty());
        let text = read_sentence_reply(&mut client, 9, Duration::from_millis(50)).unwrap();
        assert!(text.is_none());
    }
}
