//! IBus engine state machine (pure logic, no D-Bus): key handling, display
//! decisions, and predictd query construction. The zbus glue in `main.rs`
//! translates [`KeyAction`] into signals.

use std::time::{Duration, Instant};

/// GDK-style keyvals we interpret.
pub const KEY_BACKSPACE: u32 = 0xff08;
pub const KEY_TAB: u32 = 0xff09;
pub const KEY_RETURN: u32 = 0xff0d;
pub const KEY_ESCAPE: u32 = 0xff1b;
pub const KEY_LEFT: u32 = 0xff51;
pub const KEY_UP: u32 = 0xff52;
pub const KEY_RIGHT: u32 = 0xff53;
pub const KEY_DOWN: u32 = 0xff54;
/// IBus modifier bits (control/alt suppress text input; shift does not).
pub const MOD_CONTROL: u32 = 1 << 2;
pub const MOD_ALT: u32 = 1 << 3;
/// IBus input purposes that count as sensitive (verified against IBus 1.5).
pub const PURPOSE_PASSWORD: u32 = 8;
pub const PURPOSE_TERMINAL: u32 = 10;
/// Hard rule: slower than this, the key passes through with no display.
pub const PREDICT_BUDGET: Duration = Duration::from_millis(10);

/// Map a keyval (+ modifiers) to a typed char, if it inserts text.
/// GDK printable keyvals match Unicode scalar values directly (Latin-1 and
/// `0x01000000 | codepoint` for the rest); control/alt combinations never do.
pub fn keyval_to_char(keyval: u32, state: u32) -> Option<char> {
    if state & (MOD_CONTROL | MOD_ALT) != 0 {
        return None;
    }
    // GDK special keys live here (Tab 0xff09 doubles as U+FF09 on paper —
    // the real character always arrives in the 0x01000000 form instead).
    if (0xff00..=0xffff).contains(&keyval) {
        return None;
    }
    let codepoint = if keyval & 0xff00_0000 == 0x0100_0000 {
        keyval & 0x00ff_ffff
    } else {
        keyval
    };
    let c = char::from_u32(codepoint)?;
    if c.is_control() || (c.is_whitespace() && c != ' ') {
        return None;
    }
    Some(c)
}

/// Trailing in-progress fragment length in chars (for surrounding deletes).
pub fn fragment_len(text: &str) -> usize {
    text.chars()
        .rev()
        .take_while(|c| c.is_alphanumeric())
        .count()
}

/// What the engine decided for one key event.
#[derive(Debug, PartialEq)]
pub enum KeyAction {
    /// Pass the key through, display unchanged.
    Pass,
    /// Pass the key through after clearing the display.
    ClearAndPass,
    /// Pass the key through, then query predictd with this context.
    /// Carries the fresh generation: replies must echo it back.
    Query {
        /// Generation for this keystroke (bumped on every state change).
        generation: u64,
        /// Text before the cursor for the query.
        before: String,
        /// Text after the cursor for the query.
        after: String,
    },
    /// Consume the key: delete `delete_before` chars, then commit `text`.
    Commit {
        /// Chars to DeleteSurroundingText first (word-accept fragment).
        delete_before: usize,
        /// Text to CommitText-signal.
        text: String,
    },
    /// Pass the key, clear display, and offer settled text for learning.
    SettleAndPass {
        /// Settled buffer for a predictd `CommitText` (skipped when empty).
        settled: String,
    },
}

/// What to show: ghost preedit and/or lookup table.
#[derive(Debug, PartialEq, Default)]
pub struct Display {
    /// Grey ghost continuation after the cursor (sentence preferred).
    pub preedit: Option<String>,
    /// Word candidates for the lookup table.
    pub lookup: Vec<String>,
}

/// Prediction engine state: input mirror, surrounding text, sensitivity,
/// and the latest suggestions.
pub struct EngineState {
    buffer: String,
    surrounding_before: String,
    surrounding_after: String,
    has_surrounding: bool,
    sensitive: bool,
    words: Vec<String>,
    sentence: Option<String>,
    generation: u64,
}

impl EngineState {
    /// Fresh, unfocused state.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            surrounding_before: String::new(),
            surrounding_after: String::new(),
            has_surrounding: false,
            sensitive: false,
            words: Vec::new(),
            sentence: None,
            generation: 0,
        }
    }

    /// Current generation (bumped per handled keystroke).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Surrounding-text update from the daemon (`SetSurroundingText`).
    /// Offsets that don't land on char boundaries fall back to treating
    /// everything as before-cursor text.
    pub fn set_surrounding(&mut self, text: &str, cursor_pos: usize, _anchor_pos: usize) {
        let split = text
            .char_indices()
            .nth(cursor_pos)
            .map(|(byte, _)| byte)
            .unwrap_or(text.len());
        if text.is_char_boundary(split) {
            self.surrounding_before = text[..split].to_string();
            self.surrounding_after = text[split..].to_string();
        } else {
            self.surrounding_before = text.to_string();
            self.surrounding_after.clear();
        }
        self.has_surrounding = true;
    }

    /// Content-type update (`ContentType` property): password/terminal
    /// fields are sensitive — no queries, no learning, no display.
    pub fn set_content_type(&mut self, purpose: u32, _hints: u32) {
        self.sensitive = purpose == PURPOSE_PASSWORD || purpose == PURPOSE_TERMINAL;
        if self.sensitive {
            self.words.clear();
            self.sentence = None;
        }
    }

    /// Whether this is a sensitive field.
    #[cfg(test)]
    fn is_sensitive(&self) -> bool {
        self.sensitive
    }

    /// Focus out: offer the settled buffer for learning, then reset.
    /// With surrounding text (the common case) the buffer is empty because
    /// keys pass through, so the best-known field text settles instead
    /// (capped; clients report one update behind keystrokes).
    pub fn focus_out(&mut self) -> Option<String> {
        self.words.clear();
        self.sentence = None;
        self.has_surrounding = false;
        self.sensitive = false;
        let buffered = std::mem::take(&mut self.buffer);
        if !buffered.trim().is_empty() {
            return Some(buffered);
        }
        const MAX_SETTLED: usize = 2000;
        let tail: String = self
            .surrounding_before
            .chars()
            .rev()
            .take(MAX_SETTLED)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        self.surrounding_before.clear();
        self.surrounding_after.clear();
        if tail.trim().is_empty() {
            None
        } else {
            Some(tail)
        }
    }

    /// Reset composition state without learning.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.words.clear();
        self.sentence = None;
    }

    /// Context for a predictd query: surrounding text when the client
    /// reports it, otherwise the engine-side buffer.
    fn context_before(&self) -> String {
        if self.has_surrounding {
            format!("{}{}", self.surrounding_before, self.buffer)
        } else {
            self.buffer.clone()
        }
    }

    /// Bump the generation; every state-changing key does this so late
    /// replies can never resurrect (see `apply_*`).
    fn bump_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    /// Handle one key event. Returns what to do; the caller performs I/O
    /// (predictd queries, signals) and feeds replies back via
    /// [`EngineState::apply_words`] / [`EngineState::apply_sentence`].
    pub fn handle_key(&mut self, keyval: u32, state: u32, _now: Instant) -> KeyAction {
        if self.sensitive {
            return KeyAction::ClearAndPass;
        }
        match keyval {
            KEY_ESCAPE => {
                self.reset();
                self.bump_generation();
                KeyAction::ClearAndPass
            }
            KEY_TAB => {
                if let Some(sentence) = self.sentence.clone() {
                    // Backend already stripped the fragment: commit verbatim.
                    self.words.clear();
                    self.sentence = None;
                    self.buffer.clear();
                    self.bump_generation();
                    KeyAction::Commit {
                        delete_before: 0,
                        text: sentence,
                    }
                } else if let Some(top) = self.words.first().cloned() {
                    // Full-word candidate: delete the typed fragment first.
                    let base = self.context_before();
                    let frag = fragment_len(&base);
                    self.words.clear();
                    self.buffer.clear();
                    self.bump_generation();
                    KeyAction::Commit {
                        delete_before: frag,
                        text: top,
                    }
                } else {
                    KeyAction::Pass
                }
            }
            KEY_RETURN => {
                let settled = std::mem::take(&mut self.buffer);
                self.words.clear();
                self.sentence = None;
                self.bump_generation();
                KeyAction::SettleAndPass { settled }
            }
            KEY_BACKSPACE => {
                if !self.has_surrounding {
                    self.buffer.pop();
                }
                let generation = self.bump_generation();
                KeyAction::Query {
                    generation,
                    before: self.context_before(),
                    after: self.surrounding_after.clone(),
                }
            }
            KEY_LEFT | KEY_RIGHT | KEY_UP | KEY_DOWN => {
                // Cursor moved elsewhere: local context is stale.
                self.reset();
                self.bump_generation();
                KeyAction::ClearAndPass
            }
            _ => match keyval_to_char(keyval, state) {
                Some(c) => {
                    if !self.has_surrounding {
                        self.buffer.push(c);
                    }
                    let generation = self.bump_generation();
                    // Query with the key applied (clients report
                    // surrounding text one update behind keystrokes).
                    let mut before = self.context_before();
                    if self.has_surrounding {
                        before.push(c);
                    }
                    KeyAction::Query {
                        generation,
                        before,
                        after: self.surrounding_after.clone(),
                    }
                }
                None => KeyAction::Pass,
            },
        }
    }

    /// Word reply arrived; ignored when stale (a newer keystroke has
    /// already bumped past `generation`).
    pub fn apply_words(&mut self, generation: u64, words: Vec<String>) {
        if generation == self.generation {
            self.words = words;
        }
    }

    /// Sentence reply arrived (same staleness rule).
    pub fn apply_sentence(&mut self, generation: u64, text: Option<String>) {
        if generation == self.generation {
            self.sentence = text;
        }
    }

    /// Current display: sentence ghost preferred, else top word; lookup
    /// table whenever several word candidates exist.
    pub fn display(&self) -> Display {
        if self.sensitive {
            return Display::default();
        }
        Display {
            preedit: self
                .sentence
                .clone()
                .or_else(|| self.words.first().cloned()),
            lookup: if self.words.len() >= 2 {
                self.words.clone()
            } else {
                Vec::new()
            },
        }
    }
}

impl Default for EngineState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> Instant {
        Instant::now()
    }

    #[test]
    fn keyval_mapping_covers_text_and_controls() {
        assert_eq!(keyval_to_char(0x61, 0), Some('a'));
        assert_eq!(keyval_to_char(0x20, 0), Some(' '));
        assert_eq!(keyval_to_char(0xfc, 0), Some('ü'));
        assert_eq!(keyval_to_char(0x010000e4, 0), Some('ä'));
        assert_eq!(keyval_to_char(0x61, MOD_CONTROL), None);
        assert_eq!(keyval_to_char(0x61, MOD_ALT), None);
        assert_eq!(keyval_to_char(0x61, 1 << 0), Some('a'));
        assert_eq!(keyval_to_char(KEY_TAB, 0), None);
        assert_eq!(keyval_to_char(KEY_BACKSPACE, 0), None);
        assert_eq!(keyval_to_char(0x00, 0), None);
    }

    #[test]
    fn typing_queries_with_growing_buffer() {
        let mut engine = EngineState::new();
        let action = engine.handle_key(0x68, 0, now()); // 'h'
        assert!(matches!(action, KeyAction::Query { before, .. } if before == "h"));
        let action = engine.handle_key(0x69, 0, now()); // 'i'
        assert!(matches!(action, KeyAction::Query { before, .. } if before == "hi"));
        assert_eq!(engine.generation(), 2);
    }

    #[test]
    fn surrounding_text_replaces_buffer() {
        let mut engine = EngineState::new();
        engine.set_surrounding("hello wo", 8, 8);
        // Buffer stays empty; the key applies on top for the query.
        let action = engine.handle_key(0x72, 0, now()); // 'r'
        assert!(matches!(action, KeyAction::Query { before, .. } if before == "hello wor"));
    }

    #[test]
    fn surrounding_split_handles_offsets() {
        let mut engine = EngineState::new();
        engine.set_surrounding("hello world", 5, 5);
        let action = engine.handle_key(0x58, 0, now()); // 'X'
        match action {
            KeyAction::Query {
                generation: _,
                before,
                after,
            } => {
                assert_eq!(before, "helloX");
                assert_eq!(after, " world");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn tab_accepts_sentence_verbatim() {
        let mut engine = EngineState::new();
        engine.apply_sentence(0, Some(" fox jumps.".to_string()));
        let action = engine.handle_key(KEY_TAB, 0, now());
        assert_eq!(
            action,
            KeyAction::Commit {
                delete_before: 0,
                text: " fox jumps.".to_string()
            }
        );
        assert_eq!(engine.display(), Display::default());
    }

    #[test]
    fn tab_accepts_top_word_with_fragment_delete() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now());
        let _ = engine.handle_key(0x65, 0, now());
        let _ = engine.handle_key(0x6c, 0, now()); // "hel"
        engine.apply_words(3, vec!["hello".to_string(), "help".to_string()]);
        let action = engine.handle_key(KEY_TAB, 0, now());
        assert_eq!(
            action,
            KeyAction::Commit {
                delete_before: 3,
                text: "hello".to_string()
            }
        );
    }

    #[test]
    fn tab_without_suggestion_passes() {
        let mut engine = EngineState::new();
        assert_eq!(engine.handle_key(KEY_TAB, 0, now()), KeyAction::Pass);
    }

    #[test]
    fn backspace_shrinks_buffer_and_requeries() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now());
        let _ = engine.handle_key(0x69, 0, now());
        let action = engine.handle_key(KEY_BACKSPACE, 0, now());
        assert!(matches!(action, KeyAction::Query { before, .. } if before == "h"));
    }

    #[test]
    fn arrows_clear_stale_context() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now());
        engine.apply_words(1, vec!["hi".to_string()]);
        assert_eq!(engine.handle_key(KEY_LEFT, 0, now()), KeyAction::ClearAndPass);
        assert_eq!(engine.display(), Display::default());
    }

    #[test]
    fn enter_settles_buffer_for_learning() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now());
        let _ = engine.handle_key(0x69, 0, now());
        assert_eq!(
            engine.handle_key(KEY_RETURN, 0, now()),
            KeyAction::SettleAndPass {
                settled: "hi".to_string()
            }
        );
        assert_eq!(engine.display(), Display::default());
    }

    #[test]
    fn sensitive_suppresses_everything() {
        let mut engine = EngineState::new();
        engine.set_content_type(PURPOSE_PASSWORD, 0);
        assert!(engine.is_sensitive());
        assert_eq!(engine.handle_key(0x68, 0, now()), KeyAction::ClearAndPass);
        engine.apply_words(0, vec!["hi".to_string()]);
        engine.apply_sentence(0, Some(" there.".to_string()));
        assert_eq!(engine.display(), Display::default());
        // Focus out clears sensitivity for the next field.
        assert_eq!(engine.focus_out(), None);
        assert!(!engine.is_sensitive());
    }

    #[test]
    fn terminal_counts_as_sensitive() {
        let mut engine = EngineState::new();
        engine.set_content_type(PURPOSE_TERMINAL, 0);
        assert!(engine.is_sensitive());
    }

    #[test]
    fn focus_out_settles_and_resets() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now());
        let _ = engine.handle_key(0x69, 0, now());
        assert_eq!(engine.focus_out(), Some("hi".to_string()));
        assert_eq!(engine.focus_out(), None);
    }

    #[test]
    fn focus_out_settles_surrounding_when_buffer_empty() {
        let mut engine = EngineState::new();
        engine.set_surrounding("hello world", 11, 11);
        assert_eq!(engine.focus_out(), Some("hello world".to_string()));
    }

    #[test]
    fn display_prefers_sentence_shows_table_for_words() {
        let mut engine = EngineState::new();
        engine.apply_words(0, vec!["one".to_string(), "two".to_string()]);
        let shown = engine.display();
        assert_eq!(shown.preedit, Some("one".to_string()));
        assert_eq!(shown.lookup, vec!["one".to_string(), "two".to_string()]);
        engine.apply_sentence(0, Some(" green.".to_string()));
        let shown = engine.display();
        assert_eq!(shown.preedit, Some(" green.".to_string()));
    }

    #[test]
    fn stale_replies_never_resurrect() {
        let mut engine = EngineState::new();
        let _ = engine.handle_key(0x68, 0, now()); // gen 1
        let _ = engine.handle_key(0x69, 0, now()); // gen 2
        // Late gen-1 reply: dropped, display stays empty.
        engine.apply_words(1, vec!["stale".to_string()]);
        engine.apply_sentence(1, Some(" stale.".to_string()));
        assert_eq!(engine.display(), Display::default());
        // Current generation still applies.
        engine.apply_words(2, vec!["hi".to_string()]);
        assert_eq!(engine.display().preedit, Some("hi".to_string()));
    }

    #[test]
    fn fragment_len_counts_unicode() {
        assert_eq!(fragment_len("hello wo"), 2);
        assert_eq!(fragment_len("hello "), 0);
        assert_eq!(fragment_len("grü"), 3);
    }
}
