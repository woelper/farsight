//! predict-cli: terminal test client for the prediction daemon.
//!
//! Type text, see word suggestions live, Tab accepts the top word. Sentence
//! prediction is on by default: every refresh also asks the slow tier, and
//! the latest completion renders grey after the cursor (Shift+Tab accepts
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
    CancelMsg, ClientMsg, CommitText, ContextUpdate, DaemonMsg, LearningState, ProtoCandidate,
    SentenceSuggestion, SetLearning, SuggestRequest, read_daemon_msg, socket_path,
    write_client_msg,
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
/// How long control commands (pause/forget) wait for their reply.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);

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
     Shift+Tab    accept the whole sentence (grey text)\n  \
     Enter        commit the line (settled text for learning)\n  \
     Ctrl+S       cycle prediction style (default/du/sie)\n  \
     Ctrl+P       pause/resume learning\n  \
     Ctrl+F       forget all personal data (press twice to confirm)\n  \
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
    // Assumed until the first control reply corrects it.
    let mut learn_on = true;
    let mut learn_docs: u64 = 0;
    let mut forget_armed = false;
    let mut current_style = "default";
    let (mut state, mut word_rtt_ms) = on_buffer_changed(
        &mut stream,
        &buffer,
        current_style,
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
            &LearnUi {
                on: learn_on,
                docs: learn_docs,
                forget_armed,
            },
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
                    KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        forget_armed = false;
                        current_style = cycle_style(current_style);
                        mutated = true;
                    }
                    KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        forget_armed = false;
                        if let Some(state) =
                            toggle_learning(&mut stream, learn_on, generation, &mut sentence)?
                        {
                            learn_on = state.enabled;
                            learn_docs = state.documents;
                        }
                    }
                    KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        forget_armed = !forget_armed;
                        if !forget_armed {
                            if let Some(state) = forget_all(
                                &mut stream,
                                generation,
                                &mut sentence,
                            )? {
                                learn_docs = state.documents;
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        forget_armed = false;
                        buffer.push(c);
                        mutated = true;
                    }
                    KeyCode::Backspace => {
                        forget_armed = false;
                        buffer.pop();
                        mutated = true;
                    }
                    KeyCode::Tab => {
                        forget_armed = false;
                        if is_sentence_accept(KeyCode::Tab, key.modifiers) {
                            // Shift+Tab on terminals that report it as
                            // Tab+SHIFT instead of BackTab.
                            if let Some(text) = sentence.take() {
                                buffer = accept_sentence(&buffer, &text);
                                mutated = true;
                            }
                        } else if let Some(top) = state.candidates.first() {
                            buffer = accept_completion(&buffer, &top.text);
                            mutated = true;
                        }
                    }
                    // Most terminals report Shift+Tab as BackTab (ESC[Z).
                    KeyCode::BackTab => {
                        forget_armed = false;
                        if let Some(text) = sentence.take() {
                            buffer = accept_sentence(&buffer, &text);
                            mutated = true;
                        }
                    }
                    KeyCode::Enter => {
                        forget_armed = false;
                        let done = std::mem::take(&mut buffer);
                        send_commit(&mut stream, &done)?;
                        // Commit ack carries live store state (timeout keeps
                        // the old display; transport errors exit below).
                        if let Some(state) = read_learning_reply(
                            &mut stream,
                            generation,
                            &mut sentence,
                            CONTROL_TIMEOUT,
                        )? {
                            learn_on = state.enabled;
                            learn_docs = state.documents;
                        }
                        committed.push(done);
                        mutated = true;
                    }
                    _ => {
                        forget_armed = false;
                    }
                }
            }
            if mutated {
                let (fresh, rtt) = on_buffer_changed(
                    &mut stream,
                    &buffer,
                    current_style,
                    &mut generation,
                    &mut sentence_req,
                    &mut sentence,
                    args.sentence_enabled,
                )?;
                state = fresh;
                word_rtt_ms = rtt;
            }
        } else if args.sentence_enabled && !buffer.is_empty() {
            // Typing pause: collect sentence messages for this generation
            // (requested on the last refresh; re-request defensively if the
            // generation moved on). Streaming workers send a message per
            // grown prefix; applying head to tail keeps the ghost live.
            if sentence_req != Some(generation) {
                request_sentence(&mut stream, &buffer, current_style, generation)?;
                sentence_req = Some(generation);
            }
            let updates = drain_sentence(&mut stream, generation, SENTENCE_TRY)?;
            apply_sentence(&mut sentence, &updates);
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
                &LearnUi {
                    on: learn_on,
                    docs: learn_docs,
                    forget_armed,
                },
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
    match predict_proto::read_word_reply(stream, generation, timeout) {
        Ok(Some(s)) => Ok(SuggestionState {
            generation,
            candidates: s.candidates,
            style_id: s.style_id,
        }),
        Ok(None) => Ok(empty()),
        Err(e) => Err(anyhow::anyhow!(e).context("read suggestion")),
    }
}

/// Drain every sentence message for `generation` that arrives within
/// `timeout`, in order. Streaming workers send one per grown prefix plus a
/// terminal message; callers apply them head to tail (empty text retracts).
fn drain_sentence(
    stream: &mut UnixStream,
    generation: u64,
    timeout: Duration,
) -> Result<Vec<SentenceSuggestion>> {
    let deadline = Instant::now() + timeout;
    let mut out = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        stream
            .set_read_timeout(Some(remaining))
            .context("set read timeout")?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::Sentence(s)) if s.generation == generation => out.push(s),
            Ok(_) => {} // stale or word reply: discard, keep waiting
            Err(e) if predict_proto::is_timeout(&e) => {
                if Instant::now() >= deadline {
                    break;
                }
            }
            Err(e) => return Err(anyhow::anyhow!(e).context("read sentence")),
        }
    }
    Ok(out)
}

/// Apply drained sentence messages to the ghost: latest text wins, an empty
/// message retracts a shown partial (gated or failed tail).
fn apply_sentence(sentence: &mut Option<String>, updates: &[SentenceSuggestion]) {
    for update in updates {
        if update.text.is_empty() {
            *sentence = None;
        } else {
            *sentence = Some(update.text.clone());
        }
    }
}

/// Cycle to the next built-in style id (the daemon always knows these).
fn cycle_style(current: &str) -> &'static str {
    let styles = predict_core::BUILTIN_STYLES;
    let pos = styles.iter().position(|s| *s == current).unwrap_or(0);
    styles[(pos + 1) % styles.len()]
}

/// The buffer changed: cancel superseded slow work, bump the generation,
/// clear the sentence, refresh word suggestions — and, by default, ask the
/// slow tier too (same generation: it refines, not supersedes).
///
/// Returns the word view plus the word round-trip time in milliseconds.
fn on_buffer_changed(
    stream: &mut UnixStream,
    buffer: &str,
    style_id: &str,
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
    write_client_msg(
        stream,
        &ClientMsg::ContextUpdate(build_context_update(buffer, style_id)),
    )
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
        request_sentence(stream, buffer, style_id, gen)?;
        *sentence_req = Some(gen);
    }
    Ok((state, rtt_ms))
}

/// Ask for a sentence continuation for `generation`.
fn request_sentence(
    stream: &mut UnixStream,
    buffer: &str,
    style_id: &str,
    generation: u64,
) -> Result<()> {
    write_client_msg(
        stream,
        &ClientMsg::ContextUpdate(build_context_update(buffer, style_id)),
    )
    .context("send context")?;
    write_client_msg(
        stream,
        &ClientMsg::SuggestSentence(SuggestRequest { generation }),
    )
    .context("send sentence request")?;
    Ok(())
}

/// Build a settled-text commit for a committed line.
fn build_commit_msg(text: &str) -> ClientMsg {
    ClientMsg::CommitText(CommitText {
        text: text.to_string(),
        style_id: "default".to_string(),
        sensitive: false,
    })
}

/// Send settled text for learning. Write errors propagate (a dead daemon
/// surfaces on the next refresh at the latest).
fn send_commit(stream: &mut UnixStream, text: &str) -> Result<()> {
    write_client_msg(stream, &build_commit_msg(text)).context("send commit")
}

/// Read until the learning-state reply arrives or `timeout` passes. A fresh
/// sentence arriving meanwhile is stashed instead of dropped.
fn read_learning_reply(
    stream: &mut UnixStream,
    current_gen: u64,
    sentence: &mut Option<String>,
    timeout: Duration,
) -> Result<Option<LearningState>> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        stream
            .set_read_timeout(Some(remaining))
            .context("set read timeout")?;
        match read_daemon_msg(stream) {
            Ok(DaemonMsg::LearningState(state)) => return Ok(Some(state)),
            Ok(DaemonMsg::Sentence(s)) if s.generation == current_gen => {
                apply_sentence(sentence, std::slice::from_ref(&s));
            }
            Ok(_) => {} // stale: discard, keep waiting
            Err(e) if predict_proto::is_timeout(&e) => {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e).context("read learning state"),
        }
    }
}

/// Toggle pause-learning; returns the new state (`None` on timeout, keeping
/// the old display).
fn toggle_learning(
    stream: &mut UnixStream,
    learn_on: bool,
    generation: u64,
    sentence: &mut Option<String>,
) -> Result<Option<LearningState>> {
    write_client_msg(
        stream,
        &ClientMsg::SetLearning(SetLearning {
            enabled: !learn_on,
        }),
    )
    .context("send pause toggle")?;
    read_learning_reply(stream, generation, sentence, CONTROL_TIMEOUT)
}

/// Forget-all (confirmed by the caller via double-press); returns the new
/// state (`None` on timeout).
fn forget_all(
    stream: &mut UnixStream,
    generation: u64,
    sentence: &mut Option<String>,
) -> Result<Option<LearningState>> {
    write_client_msg(stream, &ClientMsg::ForgetAll).context("send forget-all")?;
    read_learning_reply(stream, generation, sentence, CONTROL_TIMEOUT)
}

/// Build the proto context for the current input line and style.
fn build_context_update(buffer: &str, style_id: &str) -> ContextUpdate {
    ContextUpdate {
        app_id: "predict-cli".to_string(),
        before: buffer.to_string(),
        after: String::new(),
        sensitive: false,
        style_id: style_id.to_string(),
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

/// Sentence-accept key: Shift+Tab, reported either as BackTab (most
/// terminals, ESC[Z) or as Tab with the Shift modifier (modifyOtherKeys).
/// Plain Tab stays word-accept.
fn is_sentence_accept(code: KeyCode, modifiers: KeyModifiers) -> bool {
    matches!(code, KeyCode::BackTab)
        || (matches!(code, KeyCode::Tab) && modifiers.contains(KeyModifiers::SHIFT))
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
    learn_on: bool,
    learn_docs: u64,
    forget_armed: bool,
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

    // Status line (compact segments so it fits narrow terminals).
    let sentence_status = if !view.sentence_enabled {
        "sent off"
    } else if view.sentence.is_some() {
        "sent ready"
    } else if view.sentence_pending {
        "sent …"
    } else {
        "no sent"
    };
    let learn_status = if view.learn_on {
        format!("learn on ({})", view.learn_docs)
    } else {
        "learn paused".to_string()
    };
    lines.push(
        fit_head(
            &format!(
                "{} · gen {} · {:.1}ms · {sentence_status} · {learn_status}",
                view.style_id, view.generation, view.word_rtt_ms,
            ),
            width,
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

    // Footer (confirm prompt while a forget-all is armed).
    let footer = if view.forget_armed {
        "FORGET ALL personal data? Ctrl+F again to confirm, any other key aborts"
    } else {
        "Tab word · Ctrl+→ sent · Enter commit · Ctrl+S style · Ctrl+P learn · Ctrl+F forget · Esc"
    };
    lines.push(fit_head(footer, width).dim().to_string());

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

/// Learning UI state (grouped so helpers stay under the argument limit).
struct LearnUi {
    on: bool,
    docs: u64,
    forget_armed: bool,
}

/// Snapshot the display state into a [`Frame`] (terminal-sized).
fn snapshot<'a>(
    buffer: &'a str,
    state: &'a SuggestionState,
    sent: &SentenceUi<'a>,
    word_rtt_ms: f64,
    learn: &LearnUi,
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
        learn_on: learn.on,
        learn_docs: learn.docs,
        forget_armed: learn.forget_armed,
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
        assert_eq!(accept_sentence("", "Hello."), "Hello.");
    }

    #[test]
    fn sentence_accept_key_covers_both_shift_tab_encodings() {
        use crossterm::event::KeyModifiers;
        assert!(is_sentence_accept(KeyCode::BackTab, KeyModifiers::empty()));
        assert!(is_sentence_accept(KeyCode::BackTab, KeyModifiers::SHIFT));
        assert!(is_sentence_accept(KeyCode::Tab, KeyModifiers::SHIFT));
        assert!(!is_sentence_accept(KeyCode::Tab, KeyModifiers::empty()));
        assert!(!is_sentence_accept(KeyCode::Tab, KeyModifiers::CONTROL));
        assert!(!is_sentence_accept(KeyCode::Enter, KeyModifiers::SHIFT));
    }

    #[test]
    fn context_carries_buffer_app_and_style() {
        let ctx = build_context_update("hello wo", "sie");
        assert_eq!(ctx.app_id, "predict-cli");
        assert_eq!(ctx.before, "hello wo");
        assert!(ctx.after.is_empty());
        assert!(!ctx.sensitive);
        assert_eq!(ctx.style_id, "sie");
    }

    #[test]
    fn style_cycles_through_builtins() {
        assert_eq!(cycle_style("default"), "du");
        assert_eq!(cycle_style("du"), "sie");
        assert_eq!(cycle_style("sie"), "default");
        assert_eq!(cycle_style("unknown"), "du");
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
        let updates = drain_sentence(&mut client, 9, Duration::from_secs(5)).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].text, " fox.");
    }

    #[test]
    fn sentence_reply_times_out_to_none() {
        let (mut client, _server) = PairStream::pair().unwrap();
        let updates = drain_sentence(&mut client, 3, Duration::from_millis(50)).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn stale_replies_are_ignored() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(8, "old")).unwrap();
        write_daemon_msg(&mut server, &suggestion_msg(8)).unwrap();
        // Wanting gen 9: both stale frames are discarded, then timeout.
        let state = read_words_reply(&mut client, 9, Duration::from_millis(50)).unwrap();
        assert!(state.candidates.is_empty());
        let updates = drain_sentence(&mut client, 9, Duration::from_millis(50)).unwrap();
        assert!(updates.is_empty());
    }

    #[test]
    fn drain_keeps_every_same_generation_update_in_order() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(9, " f")).unwrap();
        write_daemon_msg(&mut server, &sentence_msg(9, " fox")).unwrap();
        write_daemon_msg(&mut server, &sentence_msg(9, " fox.")).unwrap();
        let updates = drain_sentence(&mut client, 9, Duration::from_secs(5)).unwrap();
        let texts: Vec<&str> = updates.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, vec![" f", " fox", " fox."]);
    }

    #[test]
    fn apply_sentence_overwrites_and_retracts() {
        let mut sentence = None;
        apply_sentence(&mut sentence, &[]);
        assert_eq!(sentence, None);
        let partial = SentenceSuggestion {
            generation: 9,
            text: " fox".to_string(),
            confidence: -0.5,
            style_id: "default".to_string(),
        };
        let full = SentenceSuggestion {
            generation: 9,
            text: " fox.".to_string(),
            confidence: -0.5,
            style_id: "default".to_string(),
        };
        let retract = SentenceSuggestion {
            generation: 9,
            text: String::new(),
            confidence: f32::NEG_INFINITY,
            style_id: "default".to_string(),
        };
        apply_sentence(&mut sentence, &[partial]);
        assert_eq!(sentence.as_deref(), Some(" fox"));
        apply_sentence(&mut sentence, &[full]);
        assert_eq!(sentence.as_deref(), Some(" fox."));
        apply_sentence(&mut sentence, &[retract]);
        assert_eq!(sentence, None);
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
            learn_on: true,
            learn_docs: 3,
            forget_armed: false,
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
            learn_on: false,
            learn_docs: 0,
            forget_armed: false,
            committed: &[],
        };
        let plain = strip_ansi(&draw_frame(&view));
        assert!(plain.contains("no word suggestions"), "empty hint missing:\n{plain}");
        assert!(plain.contains("sent off"), "disabled hint missing:\n{plain}");
        assert!(plain.contains("learn paused"), "learn state missing:\n{plain}");
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
            learn_on: true,
            learn_docs: 12,
            forget_armed: true,
            committed: &["first line".to_string(), "second line".to_string()],
        };
        let plain = strip_ansi(&draw_frame(&view));
        assert!(plain.contains('…'), "pending marker missing:\n{plain}");
        assert!(plain.contains("committed: second line"), "history missing:\n{plain}");
        assert!(plain.contains("learn on (12)"), "learn state missing:\n{plain}");
        assert!(plain.contains("FORGET ALL"), "confirm prompt missing:\n{plain}");
    }

    #[test]
    fn commit_msg_marks_settled_text() {
        match build_commit_msg("hello world") {
            ClientMsg::CommitText(commit) => {
                assert_eq!(commit.text, "hello world");
                assert_eq!(commit.style_id, "default");
                assert!(!commit.sensitive);
            }
            other => panic!("expected CommitText, got {other:?}"),
        }
    }

    fn learning_msg(enabled: bool, documents: u64) -> DaemonMsg {
        DaemonMsg::LearningState(predict_proto::LearningState {
            enabled,
            documents,
        })
    }

    #[test]
    fn learning_reply_returns_state() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &learning_msg(false, 7)).unwrap();
        let mut sentence = None;
        let state = read_learning_reply(&mut client, 3, &mut sentence, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert!(!state.enabled);
        assert_eq!(state.documents, 7);
        assert!(sentence.is_none());
    }

    #[test]
    fn learning_reply_stashes_fresh_sentence() {
        let (mut client, mut server) = PairStream::pair().unwrap();
        write_daemon_msg(&mut server, &sentence_msg(4, " fox.")).unwrap();
        write_daemon_msg(&mut server, &learning_msg(true, 2)).unwrap();
        let mut sentence = None;
        let state = read_learning_reply(&mut client, 4, &mut sentence, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert!(state.enabled);
        assert_eq!(sentence.as_deref(), Some(" fox."));
    }

    #[test]
    fn learning_reply_times_out_to_none() {
        let (mut client, _server) = PairStream::pair().unwrap();
        let mut sentence = None;
        let state =
            read_learning_reply(&mut client, 3, &mut sentence, Duration::from_millis(50)).unwrap();
        assert!(state.is_none());
    }

    /// Full sentence flow over a socket pair against a stub daemon thread:
    /// type, get words, collect the grey sentence, accept it, commit it.
    /// Mirrors the main loop's exact call sequence (no TTY needed).
    #[test]
    fn full_sentence_flow_over_socket_pair() {
        use std::sync::{Arc, Mutex};
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let daemon_log = Arc::clone(&log);

        let (mut client, mut server) = PairStream::pair().unwrap();
        let daemon = std::thread::spawn(move || {
            let mut docs = 0u64;
            loop {
                let note = |tag: String| daemon_log.lock().unwrap().push(tag);
                match predict_proto::read_client_msg(&mut server) {
                    Ok(ClientMsg::ContextUpdate(_)) => note("ctx".to_string()),
                    Ok(ClientMsg::Suggest(req)) => {
                        note(format!("suggest:{}", req.generation));
                        let reply = DaemonMsg::Suggestion(Suggestion {
                            generation: req.generation,
                            candidates: vec![ProtoCandidate {
                                text: "world".to_string(),
                                score: 2.0,
                            }],
                            style_id: "default".to_string(),
                        });
                        write_daemon_msg(&mut server, &reply).unwrap();
                    }
                    Ok(ClientMsg::SuggestSentence(req)) => {
                        note(format!("sentence-req:{}", req.generation));
                        let reply = DaemonMsg::Sentence(SentenceSuggestion {
                            generation: req.generation,
                            text: " wide.".to_string(),
                            confidence: -0.3,
                            style_id: "default".to_string(),
                        });
                        write_daemon_msg(&mut server, &reply).unwrap();
                    }
                    Ok(ClientMsg::Cancel(req)) => {
                        note(format!("cancel:{}", req.generation));
                    }
                    Ok(ClientMsg::CommitText(_)) => {
                        note("commit".to_string());
                        docs += 1;
                        let reply = DaemonMsg::LearningState(predict_proto::LearningState {
                            enabled: true,
                            documents: docs,
                        });
                        write_daemon_msg(&mut server, &reply).unwrap();
                    }
                    Ok(ClientMsg::SetLearning(_)) | Ok(ClientMsg::ForgetAll) => {
                        let reply = DaemonMsg::LearningState(predict_proto::LearningState {
                            enabled: true,
                            documents: docs,
                        });
                        write_daemon_msg(&mut server, &reply).unwrap();
                    }
                    Err(_) => break,
                }
            }
        });

        // Type "hello wo": words arrive, sentence requested for gen 1.
        let mut generation = 0u64;
        let mut sentence_req = None;
        let mut sentence = None;
        let (state, _) = on_buffer_changed(
            &mut client,
            "hello wo",
            "default",
            &mut generation,
            &mut sentence_req,
            &mut sentence,
            true,
        )
        .unwrap();
        assert_eq!(generation, 1);
        assert_eq!(state.candidates[0].text, "world");
        assert_eq!(sentence_req, Some(1));

        // Pause collects the grey sentence (drained head to tail).
        let updates = drain_sentence(&mut client, generation, Duration::from_secs(5)).unwrap();
        apply_sentence(&mut sentence, &updates);
        assert_eq!(sentence.as_deref(), Some(" wide."));

        // Shift+Tab appends it verbatim.
        let buffer = accept_sentence("hello wo", sentence.as_deref().unwrap());
        assert_eq!(buffer, "hello wo wide.");

        // Typing more cancels gen 1 before requesting gen 2.
        let (state2, _) = on_buffer_changed(
            &mut client,
            &buffer,
            "default",
            &mut generation,
            &mut sentence_req,
            &mut sentence,
            true,
        )
        .unwrap();
        assert_eq!(generation, 2);
        assert!(state2.candidates.iter().any(|c| c.text == "world"));

        // Enter commits settled text; the ack carries the doc count.
        send_commit(&mut client, &buffer).unwrap();
        let mut no_sentence = None;
        let learned = read_learning_reply(&mut client, generation, &mut no_sentence, Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(learned.documents, 1);

        drop(client);
        daemon.join().unwrap();
        let log = log.lock().unwrap();
        let sequence: Vec<&str> = log.iter().map(String::as_str).collect();
        // Cancel{1} precedes the gen-2 requests (same order as main loop).
        let cancel_pos = sequence.iter().position(|s| *s == "cancel:1").unwrap();
        let suggest_pos = sequence.iter().position(|s| *s == "suggest:2").unwrap();
        assert!(cancel_pos < suggest_pos, "order wrong: {sequence:?}");
        assert!(sequence.contains(&"sentence-req:1"));
        assert!(sequence.contains(&"commit"));
    }
}
