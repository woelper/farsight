//! Core types for predict: context, candidates, style, predictor trait.
//!
//! M1: sync word completion only. Sentence continuation (`continue_text`)
//! is deferred to M3 (see ADR 0002).

/// Style resolved for a given field / cursor position.
///
/// M1 placeholder: only the style id. Expanded in M5 (language,
/// address form du/sie, length policy).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedStyle {
    /// Active style id (e.g. `"default"`).
    pub style_id: String,
}

impl ResolvedStyle {
    /// The default style used when nothing else is configured.
    pub fn default_style() -> Self {
        Self {
            style_id: "default".to_string(),
        }
    }

    /// Build a resolved style from an id.
    pub fn new(id: impl Into<String>) -> Self {
        Self { style_id: id.into() }
    }
}

/// Static style specification (loaded from TOML in M5).
///
/// M1 placeholder: id + language only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleSpec {
    /// Style id (e.g. `"default"`, `"formal-de-sie"`).
    pub id: String,
    /// BCP-47-ish language tag (e.g. `"en"`, `"de"`).
    pub language: String,
}

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
        assert_eq!(ResolvedStyle::default_style().style_id, "default");
        assert_eq!(
            ResolvedStyle::default(),
            ResolvedStyle {
                style_id: String::new()
            }
        );
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
}
