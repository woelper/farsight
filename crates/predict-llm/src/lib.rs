//! Slow tier: LLM backend trait and llama.cpp implementation.
//!
//! The [`Backend`] trait is a synchronous, cooperatively-cancelable API:
//! implementations run on a worker thread (see `llama.rs`, where the
//! `!Send` llama.cpp context forces single-thread ownership). Callers pass
//! a [`CancelToken`] set on the next keystroke.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use thiserror::Error;

pub mod llama;

/// Cooperative cancellation for in-flight generation.
///
/// Cloned handles share one flag; the backend checks it between tokens.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// New, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation (idempotent).
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// True once [`cancel`](Self::cancel) has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// One sentence-continuation request.
#[derive(Debug, Clone)]
pub struct SentenceRequest {
    /// Text before the cursor; the model continues it.
    pub before: String,
    /// Generation cap (tokens).
    pub max_tokens: usize,
    /// Minimum mean token logprob for the output to be returned.
    pub confidence_threshold: f32,
}

/// A suggestion produced by the slow tier.
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceOutput {
    /// Continuation AFTER `before` (mid-word fragment already stripped).
    pub text: String,
    /// Mean token logprob over generated tokens (higher is better, ≤ 0).
    pub confidence: f32,
    /// Time from request start to the first sampled token.
    pub time_to_first_token: Duration,
    /// Tokens generated (excluding prompt).
    pub tokens_generated: usize,
}

/// Errors from the slow tier.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LlmError {
    /// Model file missing at the configured path.
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    /// Model failed to load.
    #[error("failed to load model: {0}")]
    LoadFailed(String),
    /// Failure during inference.
    #[error("inference failed: {0}")]
    Inference(String),
    /// Generation was cancelled before completing.
    #[error("cancelled")]
    Cancelled,
}

/// Slow-tier backend: complete the text before the cursor.
///
/// Synchronous with cooperative cancellation; implementations are expected
/// to run on a dedicated worker thread. Returning `Ok(None)` means "no
/// suggestion worth showing" (empty prompt, healing refusal, EOG first,
/// confidence below threshold).
pub trait Backend: Send + Sync {
    /// Generate a continuation, or `None` when nothing passes the gate.
    fn complete_sentence(
        &self,
        req: &SentenceRequest,
        cancel: &CancelToken,
    ) -> Result<Option<SentenceOutput>, LlmError>;

    /// Backend name for logs and eval reports.
    fn name(&self) -> &str;
}

/// LLM configuration (TOML `[llm]` section of `predictd.toml`).
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Master switch; false means "word tier only".
    pub enabled: bool,
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Generation cap per request.
    pub max_tokens: usize,
    /// Minimum mean token logprob to surface a suggestion.
    pub confidence_threshold: f32,
    /// Context window (tokens).
    pub n_ctx: u32,
    /// Inference threads (0 = auto).
    pub n_threads: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: String::new(),
            max_tokens: 32,
            confidence_threshold: -1.0,
            n_ctx: 2048,
            n_threads: 0,
        }
    }
}

impl LlmConfig {
    /// Parse the `[llm]` section of a TOML config file.
    ///
    /// Missing file content or a missing `[llm]` section yields a disabled
    /// config rather than an error; malformed TOML is an error.
    pub fn from_toml_str(text: &str) -> Result<Self, LlmError> {
        #[derive(serde::Deserialize, Default)]
        struct File {
            #[serde(default)]
            llm: Section,
        }
        #[derive(serde::Deserialize, Default)]
        struct Section {
            #[serde(default)]
            enabled: bool,
            #[serde(default)]
            model_path: String,
            max_tokens: Option<usize>,
            confidence_threshold: Option<f32>,
            n_ctx: Option<u32>,
            n_threads: Option<u32>,
        }
        let file: File = toml::from_str(text).map_err(|e| LlmError::Inference(format!("bad llm config: {e}")))?;
        let base = Self::default();
        Ok(Self {
            enabled: file.llm.enabled,
            model_path: file.llm.model_path,
            max_tokens: file.llm.max_tokens.unwrap_or(base.max_tokens),
            confidence_threshold: file
                .llm
                .confidence_threshold
                .unwrap_or(base.confidence_threshold),
            n_ctx: file.llm.n_ctx.unwrap_or(base.n_ctx),
            n_threads: file.llm.n_threads.unwrap_or(base.n_threads),
        })
    }

    /// Resolve 0 (auto) thread counts.
    pub fn resolved_threads(&self) -> u32 {
        if self.n_threads > 0 {
            return self.n_threads;
        }
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4)
    }
}

/// Trailing in-progress word fragment of `before` (may be empty).
///
/// Same rule as the fast tier: maximal trailing alphanumeric run.
pub fn trailing_fragment(before: &str) -> String {
    before
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric())
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

/// Split `before` into (prompt without fragment, fragment).
pub fn strip_fragment(before: &str) -> (String, String) {
    let fragment = trailing_fragment(before);
    let cut: usize = before.chars().count().saturating_sub(fragment.chars().count());
    let prompt: String = before.chars().take(cut).collect();
    (prompt, fragment)
}

/// Token-healing check: a freshly sampled piece continues the in-progress
/// word exactly when its bytes start with the fragment.
///
/// BPE pieces often carry the preceding space (`" brown"` GPT-2 style or
/// `"▁brown"` SentencePiece style) when the prompt ends at a word boundary;
/// one such leading blank is stripped before comparing.
pub fn piece_continues_word(piece: &[u8], fragment: &str) -> bool {
    if fragment.is_empty() {
        return true;
    }
    let frag = fragment.as_bytes();
    if piece.starts_with(frag) {
        return true;
    }
    strip_leading_blank(piece).starts_with(frag)
}

/// Remove one leading blank from a token piece: ASCII space (GPT-2 style
/// `" brown"`) or U+2581 (SentencePiece style `"▁brown"`).
fn strip_leading_blank(piece: &[u8]) -> &[u8] {
    if piece.first() == Some(&b' ') {
        return &piece[1..];
    }
    let ulower = "▁".as_bytes();
    if piece.starts_with(ulower) {
        return &piece[ulower.len()..];
    }
    piece
}

/// Deterministic stub backend for tests and harness development.
///
/// Returns a fixed continuation (fragment-stripped by the test setup) when
/// the prompt is non-blank and the fixed confidence passes the request
/// threshold; honors cancellation.
pub struct StubBackend {
    text: String,
    confidence: f32,
}

impl StubBackend {
    /// Stub that always suggests `text` at `confidence`.
    pub fn fixed(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            text: text.into(),
            confidence,
        }
    }

    /// Stub that never suggests anything.
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            confidence: f32::NEG_INFINITY,
        }
    }
}

impl Backend for StubBackend {
    fn complete_sentence(
        &self,
        req: &SentenceRequest,
        cancel: &CancelToken,
    ) -> Result<Option<SentenceOutput>, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        if req.before.trim().is_empty() || self.text.is_empty() {
            return Ok(None);
        }
        if self.confidence < req.confidence_threshold {
            return Ok(None);
        }
        Ok(Some(SentenceOutput {
            text: self.text.clone(),
            confidence: self.confidence,
            time_to_first_token: Duration::from_millis(1),
            tokens_generated: self.text.split_whitespace().count().max(1),
        }))
    }

    fn name(&self) -> &str {
        "stub"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_token_starts_clear_and_latches() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        assert!(token.clone().is_cancelled());
    }

    #[test]
    fn trailing_fragment_takes_alnum_tail() {
        assert_eq!(trailing_fragment("hello wo"), "wo");
        assert_eq!(trailing_fragment("hello "), "");
        assert_eq!(trailing_fragment(""), "");
        assert_eq!(trailing_fragment("grü"), "grü");
    }

    #[test]
    fn strip_fragment_splits_prompt() {
        assert_eq!(strip_fragment("the qui"), ("the ".to_string(), "qui".to_string()));
        assert_eq!(strip_fragment("the "), ("the ".to_string(), String::new()));
    }

    #[test]
    fn piece_continues_word_cases() {
        assert!(piece_continues_word(b"quick", "qui"));
        assert!(!piece_continues_word(b"ck", "qui"));
        assert!(piece_continues_word(b"anything", ""));
        // Leading blanks (word-boundary prompts): accepted, stripped later.
        assert!(piece_continues_word(b" brown", "brown"));
        assert!(piece_continues_word("▁brown".as_bytes(), "brown"));
        assert!(!piece_continues_word(b" fox", "brown"));
    }

    #[test]
    fn strip_leading_blank_cases() {
        assert_eq!(strip_leading_blank(b" brown"), b"brown");
        assert_eq!(strip_leading_blank("▁brown".as_bytes()), b"brown");
        assert_eq!(strip_leading_blank(b"brown"), b"brown");
        assert_eq!(strip_leading_blank(b""), b"");
    }

    fn req(before: &str, threshold: f32) -> SentenceRequest {
        SentenceRequest {
            before: before.to_string(),
            max_tokens: 32,
            confidence_threshold: threshold,
        }
    }

    #[test]
    fn stub_returns_fixed_output_above_threshold() {
        let backend = StubBackend::fixed("brown fox", -0.2);
        let out = backend
            .complete_sentence(&req("the quick ", -1.5), &CancelToken::new())
            .unwrap()
            .unwrap();
        assert_eq!(out.text, "brown fox");
    }

    #[test]
    fn stub_gate_rejects_low_confidence() {
        let backend = StubBackend::fixed("brown fox", -3.0);
        let out = backend
            .complete_sentence(&req("the quick ", -1.5), &CancelToken::new())
            .unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn stub_returns_none_for_blank_prompt_or_empty_output() {
        let backend = StubBackend::fixed("x", -0.1);
        assert!(
            backend
                .complete_sentence(&req("   ", -99.0), &CancelToken::new())
                .unwrap()
                .is_none()
        );
        let empty = StubBackend::empty();
        assert!(
            empty
                .complete_sentence(&req("hello", -99.0), &CancelToken::new())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stub_honors_cancel() {
        let backend = StubBackend::fixed("x", -0.1);
        let token = CancelToken::new();
        token.cancel();
        let err = backend
            .complete_sentence(&req("hello", -99.0), &token)
            .unwrap_err();
        assert!(matches!(err, LlmError::Cancelled));
    }

    #[test]
    fn config_defaults_to_disabled() {
        let cfg = LlmConfig::from_toml_str("").unwrap();
        assert!(!cfg.enabled);
        let cfg = LlmConfig::from_toml_str("[other]\nx = 1\n").unwrap();
        assert!(!cfg.enabled);
    }

    #[test]
    fn config_parses_llm_section() {
        let cfg = LlmConfig::from_toml_str(
            "[llm]\nenabled = true\nmodel_path = \"/m.gguf\"\nmax_tokens = 16\nconfidence_threshold = -2.0\n",
        )
        .unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.model_path, "/m.gguf");
        assert_eq!(cfg.max_tokens, 16);
        assert_eq!(cfg.confidence_threshold, -2.0);
        assert_eq!(cfg.n_ctx, 2048);
    }

    #[test]
    fn config_rejects_bad_toml() {
        assert!(LlmConfig::from_toml_str("[llm\nbroken").is_err());
    }

    #[test]
    fn threads_resolve_auto() {
        assert!(LlmConfig::default().resolved_threads() >= 1);
        let cfg = LlmConfig {
            n_threads: 6,
            ..Default::default()
        };
        assert_eq!(cfg.resolved_threads(), 6);
    }
}
