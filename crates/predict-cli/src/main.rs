//! predict-cli: terminal test client for the prediction daemon.
//!
//! Type text, see word suggestions live, Tab accepts the top word. Sentence
//! prediction is on by default: every refresh also asks the slow tier, and
//! the latest completion renders grey after the cursor (Ctrl+Right accepts
//! the whole sentence). Every keystroke bumps the generation and cancels
//! superseded work; replies for older generations are ignored. A slow daemon
//! never blocks typing: reads time out and render proceeds without
//! suggestions.

use anyhow::{Context as _, Result};
use crossterm::cursor::MoveTo;
use crossterm::event::{Event, KeyCode, KeyModifiers, poll, read};
use crossterm::style::Stylize;
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
    enable_raw_mode, size as term_size,
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
/// Idle time before polling the slow tier for an arrived reply.
const PAUSE: Duration = Duration::from_millis(200);
/// How long an idle cycle waits for a slow-tier reply before polling keys.
const SENTENCE_TRY: Duration = Duration::from_millis(50);

/// CLI options (all default-on; flags opt out).
#[derive(Debug, PartialEq)]
struct Args {
    sentence_enabled: bool,
}

fn parse_args(args: &[String]) -> Args {
    Args {
        sentence_enabled: !args.iter().any(|a| a == "--no-sentence"),
    }
}

fn usage() -> &'static str {
    "predict-cli [--no-sentence]\n\
     \n\
     Type to get live word suggestions from predictd.\n\
     \n\
     Keys:\n  \
     Tab          accept the top word\n  \
     Ctrl+Right   accept the whole sentence (grey text)\n  \
     Enter        commit the line\n  \
     Esc / Ctrl+C quit\n\
     \n\
     Sentence prediction runs on every keystroke by default and is\n\
     cancelled by the next one; --no-sentence turns it off."
}

fn main() -> Result<()> {
    let raw_args: Vec<String> = std::env::args().collect();
    if raw_args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", usage());
        return Ok(());
    }
    let args = parse_args(&raw_args);

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
    let (mut state, mut word_rtt_ms) = on_buffer_changed(
        &mut stream,
        &buffer,
        &mut generation,
        &mut sentence_req,
        &mut sentence,
        args.sentence_enabled,
    )?;
    render(
        &mut stdout,
        &snapshot(
            &buffer,
            &state,
            &SentenceUi {
                text: &sentence,
                requested: sentence_req,
                generation,
                enabled: args.sentence_enabled,
            },
            word_rtt_ms,
            &committed,
        ),
    )?;

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
                let (fresh, rtt) = on_buffer_changed(
                    &mut stream,
                    &buffer,
                    &mut generation,
                    &mut sentence_req,
                    &mut sentence,
                    args.sentence_enabled,
                )?;
                state = fresh;
                word_rtt_ms = rtt;
            }
        } else if args.sentence_enabled && !buffer.is_empty() && sentence.is_none() {
            // Typing pause: collect a sentence reply (requested on the last
            // refresh; re-request defensively if the generation moved on).
            if sentence_req != Some(generation) {
                request_sentence(&mut stream, &buffer, generation)?;
                sentence_req = Some(generation);
            }
            if let Some(text) = read_sentence_reply(&mut stream, generation, SENTENCE_TRY)? {
                sentence = Some(text);
            }
        }
        render(
            &mut stdout,
            &snapshot(
                &buffer,
                &state,
                &SentenceUi {
                    text: &sentence,
                    requested: sentence_req,
                    generation,
                    enabled: args.sentence_enabled,
                },
                word_rtt_ms,
                &committed,
            ),
        )?;
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
/// clear the sentence, refresh word suggestions — and, by default, ask the
/// slow tier too (same generation: it refines, not supersedes).
///
/// Returns the word view plus the word round-trip time in milliseconds.
fn on_buffer_changed(
    stream: &mut UnixStream,
    buffer: &str,
    generation: &mut u64,
    sentence_req: &mut Option<u64>,
    sentence: &mut Option<String>,
    sentence_enabled: bool,
) -> Result<(SuggestionState, f64)> {
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
    let start = Instant::now();
    let state = read_words_reply(stream, gen, WORD_TIMEOUT)?;
    let rtt_ms = start.elapsed().as_secs_f64() * 1000.0;
    if sentence_enabled && !buffer.is_empty() {
        request_sentence(stream, buffer, gen)?;
        *sentence_req = Some(gen);
    }
    Ok((state, rtt_ms))
}

/// Ask for a sentence continuation for `generation`.
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

/// Everything the frame renderer needs.
struct Frame<'a> {
    width: usize,
    max_committed: usize,
    buffer: &'a str,
    sentence: &'a Option<String>,
    sentence_pending: bool,
    sentence_enabled: bool,
    candidates: &'a [ProtoCandidate],
    style_id: &'a str,
    generation: u64,
    word_rtt_ms: f64,
    committed: &'a [String],
}

/// Keep the last `max` chars (cursor context); prefix `…` when truncated.
fn fit_tail(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    format!("…{}", text.chars().skip(count - max + 1).collect::<String>())
}

/// Keep the first `max` chars; suffix `…` when truncated.
fn fit_head(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    format!(
        "{}…",
        text.chars().take(max.saturating_sub(1)).collect::<String>()
    )
}

/// Render one full screen as a string (lines joined with `\r\n` for raw
/// mode). Pure function: all layout math happens on plain text, styles are
/// applied afterwards, so tests can assert on content and line widths.
fn draw_frame(view: &Frame) -> String {
    let width = view.width.max(24);
    let inner = width.saturating_sub(2);
    let mut lines: Vec<String> = Vec::new();

    // Header (full width; styled after padding so widths stay exact —
    // "predict-cli" is unique in the line, so targeted styling is safe).
    let title = "predict-cli";
    let top_plain = format!(
        "╭─ {title} {}╮",
        "─".repeat(inner.saturating_sub(title.chars().count() + 4))
    );
    lines.push(top_plain.replacen(title, &format!("{}", title.bold().cyan()), 1));
    // Input box: `> buffer` + grey sentence, fitted to the box. Width math
    // runs on plain text; styles wrap fitted segments afterwards.
    let prompt = "> ";
    let buf_room = inner.saturating_sub(prompt.chars().count() + 2);
    let (buf_shown, sent_shown) = match view.sentence {
        Some(text) if !text.is_empty() => {
            let sent_len = text.chars().count().min(buf_room / 2);
            let buf_len = buf_room.saturating_sub(sent_len);
            (fit_tail(view.buffer, buf_len), fit_head(text, sent_len))
        }
        _ => (fit_tail(view.buffer, buf_room), String::new()),
    };
    let pad = inner.saturating_sub(
        prompt.chars().count() + buf_shown.chars().count() + sent_shown.chars().count(),
    );
    lines.push(format!(
        "│{}{}{}{}│",
        prompt.bold().cyan(),
        buf_shown.bold().white(),
        sent_shown.dark_grey(),
        " ".repeat(pad),
    ));
    lines.push(format!("╰{}╯", "─".repeat(inner)).dim().to_string());

    // Word candidates: fit plain text first (styling afterwards, so ANSI
    // codes can never be cut or miscounted), top pick highlighted.
    if view.candidates.is_empty() {
        lines.push("(no word suggestions)".dim().to_string());
    } else {
        let prefix = "words: ";
        let avail = width.saturating_sub(prefix.chars().count());
        let joined = view
            .candidates
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("  ");
        let fitted = fit_head(&joined, avail);
        let first = view.candidates[0].text.as_str();
        // Style the top pick only when fully present (byte-exact prefix).
        let row = if fitted.starts_with(first) && !first.is_empty() {
            let rest = &fitted[first.len()..];
            format!("{prefix}{}{rest}", first.black().on_white())
        } else {
            format!("{prefix}{fitted}")
        };
        lines.push(row);
    }

    // Status line.
    let sentence_status = if !view.sentence_enabled {
        "sentence off"
    } else if view.sentence.is_some() {
        "sentence ready"
    } else if view.sentence_pending {
        "sentence …"
    } else {
        "no sentence"
    };
    lines.push(
        format!(
            "style {} · gen {} · word {:.1}ms · {sentence_status}",
            view.style_id, view.generation, view.word_rtt_ms,
        )
        .dim()
        .to_string(),
    );

    // Committed history (dimmed, newest last, capped to fit).
    let shown: Vec<&String> = view
        .committed
        .iter()
        .rev()
        .take(view.max_committed)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    for line in shown {
        lines.push(fit_head(&format!("committed: {line}"), width).dim().to_string());
    }

    // Footer.
    lines.push(
        "Tab word · Ctrl+→ sentence · Enter commit · Esc quit"
            .dim()
            .to_string(),
    );

    lines.join("\r\n")
}

fn render(stdout: &mut Stdout, frame: &Frame) -> Result<()> {
    execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
    write!(stdout, "{}", draw_frame(frame))?;
    stdout.flush()?;
    Ok(())
}

/// Sentence UI state (grouped so helpers stay under the argument limit).
struct SentenceUi<'a> {
    text: &'a Option<String>,
    requested: Option<u64>,
    generation: u64,
    enabled: bool,
}

/// Snapshot the display state into a [`Frame`] (terminal-sized).
fn snapshot<'a>(
    buffer: &'a str,
    state: &'a SuggestionState,
    sent: &SentenceUi<'a>,
    word_rtt_ms: f64,
    committed: &'a [String],
) -> Frame<'a> {
    let (cols, rows) = term_size().unwrap_or((80, 24));
    Frame {
        width: cols as usize,
        max_committed: (rows as usize)
            .saturating_sub(8)
            .max(1)
            .min(committed.len().max(1)),
        buffer,
        sentence: sent.text,
        sentence_pending: sent.requested == Some(sent.generation) && sent.text.is_none(),
        sentence_enabled: sent.enabled,
        candidates: &state.candidates,
        style_id: &state.style_id,
        generation: state.generation,
        word_rtt_ms,
        committed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use predict_proto::{SentenceSuggestion, Suggestion, write_daemon_msg};
    use std::os::unix::net::UnixStream as PairStream;

    #[test]
    fn args_default_to_sentence_on() {
        assert_eq!(
            parse_args(&["predict-cli".to_string()]),
            Args {
                sentence_enabled: true
            }
        );
    }

    #[test]
    fn no_sentence_flag_opts_out() {
        assert_eq!(
            parse_args(&["predict-cli".to_string(), "--no-sentence".to_string()]),
            Args {
                sentence_enabled: false
            }
        );
    }

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
        assert_eq!(
            accept_sentence("the quick brown", " fox."),
            "the quick brown fox."
        );
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

    fn frame_for(buffer: &str, sentence: Option<&str>) -> String {
        let owned = sentence.map(str::to_string);
        let candidates = vec![
            ProtoCandidate {
                text: "world".to_string(),
                score: 2.0,
            },
            ProtoCandidate {
                text: "word".to_string(),
                score: 1.0,
            },
        ];
        let view = Frame {
            width: 60,
            max_committed: 5,
            buffer,
            sentence: &owned,
            sentence_pending: false,
            sentence_enabled: true,
            candidates: &candidates,
            style_id: "default",
            generation: 7,
            word_rtt_ms: 0.4,
            committed: &[],
        };
        draw_frame(&view)
    }

    /// Strip SGR escape sequences for width assertions.
    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c2 in chars.by_ref() {
                    if c2 == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn frame_shows_buffer_sentence_and_candidates() {
        let frame = frame_for("hello wo", Some("rld."));
        let plain = strip_ansi(&frame);
        assert!(plain.contains("hello wo"), "buffer missing:\n{plain}");
        assert!(plain.contains("rld."), "sentence missing:\n{plain}");
        assert!(plain.contains("world"), "candidate missing:\n{plain}");
        assert!(plain.contains("gen 7"), "generation missing:\n{plain}");
        assert!(plain.contains("predict-cli"), "header missing:\n{plain}");
    }

    #[test]
    fn frame_lines_fit_width() {
        let frame = frame_for("hello wo", Some("rld and the rest of the sentence"));
        for line in strip_ansi(&frame).split("\r\n") {
            assert!(
                line.chars().count() <= 60,
                "line too wide ({}): {line:?}",
                line.chars().count()
            );
        }
    }

    #[test]
    fn frame_truncates_long_buffer_with_ellipsis() {
        let long = "a".repeat(200);
        let frame = frame_for(&long, None);
        let plain = strip_ansi(&frame);
        assert!(!plain.contains(&long), "buffer not truncated");
        assert!(plain.contains('…'), "no ellipsis marker");
    }

    #[test]
    fn frame_empty_state() {
        let owned: Option<String> = None;
        let candidates: Vec<ProtoCandidate> = Vec::new();
        let view = Frame {
            width: 60,
            max_committed: 5,
            buffer: "",
            sentence: &owned,
            sentence_pending: false,
            sentence_enabled: false,
            candidates: &candidates,
            style_id: "default",
            generation: 1,
            word_rtt_ms: 0.0,
            committed: &[],
        };
        let plain = strip_ansi(&draw_frame(&view));
        assert!(plain.contains("no word suggestions"), "empty hint missing:\n{plain}");
        assert!(plain.contains("sentence off"), "disabled hint missing:\n{plain}");
    }

    #[test]
    fn frame_pending_sentence_shows_ellipsis() {
        let owned: Option<String> = None;
        let candidates: Vec<ProtoCandidate> = Vec::new();
        let view = Frame {
            width: 60,
            max_committed: 5,
            buffer: "hello",
            sentence: &owned,
            sentence_pending: true,
            sentence_enabled: true,
            candidates: &candidates,
            style_id: "default",
            generation: 2,
            word_rtt_ms: 0.3,
            committed: &["first line".to_string(), "second line".to_string()],
        };
        let plain = strip_ansi(&draw_frame(&view));
        assert!(plain.contains('…'), "pending marker missing:\n{plain}");
        assert!(plain.contains("committed: second line"), "history missing:\n{plain}");
    }
}
