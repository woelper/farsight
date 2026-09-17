//! Core types for predict: context, candidates, style, predictor trait.
//!
//! M1: sync word completion only. Sentence continuation (`continue_text`)
//! is deferred to M3 (see ADR 0002).

/// Prediction styles: specs, address-form detection, decoding bans.
pub mod style;

pub use style::{
    AddressForm, LanguageSpec, LengthMode, ResolvedStyle, StyleSpec, banned_words, builtin_spec,
    detect_address, violates, BUILTIN_STYLES, DU_BANNED, SIE_BANNED,
};

/// Typing context sent by a frontend to the predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    /// Frontend-provided app id (e.g. `"gedit"`, `"eval"`).
    pub app_id: String,
    /// Text before the cursor.
    pub before: String,
    /// Text after the cursor (may be empty).
    pub after: String,
    /// Password / sensitive field: no suggestions, no learning.
    pub sensitive: bool,
    /// Active style for this field.
    pub style: ResolvedStyle,
}

impl Context {
    /// Build a context from its parts.
    pub fn new(
        app_id: impl Into<String>,
        before: impl Into<String>,
        after: impl Into<String>,
        sensitive: bool,
        style: ResolvedStyle,
    ) -> Self {
        Self {
            app_id: app_id.into(),
            before: before.into(),
            after: after.into(),
            sensitive,
            style,
        }
    }
}

/// One suggestion.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Suggested text (a single word in M1).
    pub text: String,
    /// Score, higher is better. Meaning is backend-specific.
    pub score: f32,
}

impl Candidate {
    /// Build a candidate from text and score.
    pub fn new(text: impl Into<String>, score: f32) -> Self {
        Self {
            text: text.into(),
            score,
        }
    }
}

/// Sort candidates in place: highest score first, ties broken by text.
pub fn rank_candidates(candidates: &mut [Candidate]) {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.text.cmp(&b.text))
    });
}

/// Lowercase unicode-alphanumeric word splitter shared by every tier.
///
/// Splits on anything that is not `char::is_alphanumeric` and lowercases the
/// rest: `"Hallo, Welt!"` -> `["hallo", "welt"]`, `"Grüße"` -> `["grüße"]`.
/// Centralized in M4 (third use: fast tier, eval harness, personal store).
pub fn tokenize_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                current.push(lc);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Current sentence fragment: text after the last `.`/`!`/`?`/newline,
/// trimmed, capped to a ~200-char tail. Used for retrieval queries and
/// prompt grounding budgets.
pub fn sentence_fragment(before: &str) -> String {
    let mut start = 0;
    for (i, c) in before.char_indices() {
        if matches!(c, '.' | '!' | '?' | '\n') {
            start = i + c.len_utf8();
        }
    }
    let frag = before[start..].trim();
    const MAX: usize = 200;
    if frag.chars().count() <= MAX {
        return frag.to_string();
    }
    let skip = frag.chars().count() - MAX;
    frag.chars().skip(skip).collect()
}

/// Word predictor. M1 covers the sync fast tier only.
///
/// Contract: `complete_word` must have p99 < 5 ms.
/// An empty vec means "no suggestion" (e.g. sensitive fields).
pub trait Predictor: Send + Sync {
    /// Complete the word at the cursor or predict the next word.
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_style_default_is_default() {
        let style = ResolvedStyle::default_style();
        assert_eq!(style.style_id, "default");
        assert_eq!(style.address_form, AddressForm::None);
        assert_eq!(style.length, LengthMode::Sentence);
        assert_eq!(ResolvedStyle::new("du").style_id, "du");
    }

    #[test]
    fn context_new_stores_fields() {
        let ctx = Context::new("eval", "hel", "", false, ResolvedStyle::default_style());
        assert_eq!(ctx.app_id, "eval");
        assert_eq!(ctx.before, "hel");
        assert!(ctx.after.is_empty());
        assert!(!ctx.sensitive);
        assert_eq!(ctx.style.style_id, "default");
    }

    #[test]
    fn rank_candidates_sorts_by_score_then_text() {
        let mut cs = vec![
            Candidate::new("b", 1.0),
            Candidate::new("a", 1.0),
            Candidate::new("c", 2.0),
        ];
        rank_candidates(&mut cs);
        assert_eq!(cs[0].text, "c");
        assert_eq!(cs[1].text, "a");
        assert_eq!(cs[2].text, "b");
    }

    #[test]
    fn rank_candidates_handles_nan_scores_without_panicking() {
        let mut cs = vec![Candidate::new("a", f32::NAN), Candidate::new("b", 1.0)];
        rank_candidates(&mut cs);
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn tokenize_text_handles_english_and_german() {
        assert_eq!(tokenize_text("Hallo, Welt!"), vec!["hallo", "welt"]);
        assert_eq!(tokenize_text("Grüße dich"), vec!["grüße", "dich"]);
        assert!(tokenize_text("   ... ").is_empty());
    }

    #[test]
    fn sentence_fragment_takes_tail() {
        assert_eq!(sentence_fragment("Hello. How are you"), "How are you");
        assert_eq!(sentence_fragment("no terminator"), "no terminator");
        assert_eq!(sentence_fragment("  spaced.  out  "), "out");
        let long = format!("x. {}", "w".repeat(300));
        assert!(sentence_fragment(&long).chars().count() <= 200);
    }
}
