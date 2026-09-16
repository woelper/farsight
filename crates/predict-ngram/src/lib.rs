//! Fast tier: prefix completion + n-gram next-word prediction.
//!
//! Model: unigram / bigram / trigram counts over lowercased
//! unicode-alphanumeric tokens, prefix index as a `BTreeMap` range scan.
//! See ADR 0002 for the trade-offs.

use predict_core::{Candidate, Context, Predictor, rank_candidates};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use thiserror::Error;

/// How many candidates `complete_word` returns at most.
pub const MAX_CANDIDATES: usize = 5;

/// Errors from the n-gram tier.
#[derive(Debug, Error)]
pub enum NgramError {
    /// Corpus file could not be read.
    #[error("cannot read corpus file {path}: {source}")]
    Io {
        /// File that failed to load.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Corpus contained no tokens.
    #[error("corpus is empty")]
    EmptyCorpus,
}

/// Lowercase unicode-alphanumeric tokenizer.
///
/// Splits on anything that is not `char::is_alphanumeric`, lowercases the
/// rest. `"Hallo, Welt!"` -> `["hallo", "welt"]`; `"Grüße"` -> `["grüße"]`.
pub fn tokenize(text: &str) -> Vec<String> {
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

/// Trailing word fragment of `before` (lowercased).
///
/// `"hello wo"` -> `"wo"`; `"hello "` -> `""`.
pub fn current_prefix(before: &str) -> String {
    let tail: String = before
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric())
        .collect();
    tail.chars().rev().flat_map(|c| c.to_lowercase()).collect()
}

/// Last `n` whole words before the cursor, excluding the in-progress prefix.
///
/// `"the quick bro"` with n=2 -> `["the", "quick"]`.
/// `"the quick "` with n=2 -> `["the", "quick"]`.
pub fn previous_words(before: &str, n: usize) -> Vec<String> {
    let prefix_len: usize = before.chars().rev().take_while(|c| c.is_alphanumeric()).count();
    let char_count: usize = before.chars().count();
    let stem: String = before.chars().take(char_count.saturating_sub(prefix_len)).collect();
    let mut words = tokenize(&stem);
    if words.len() > n {
        words.drain(..words.len() - n);
    }
    words
}

/// Base n-gram model: unigram + bigram + trigram counts.
#[derive(Debug, Default, Clone)]
pub struct NgramModel {
    unigrams: HashMap<String, u64>,
    bigrams: HashMap<(String, String), u64>,
    trigrams: HashMap<(String, String, String), u64>,
    vocab: BTreeMap<String, u64>,
    total_tokens: u64,
}

impl NgramModel {
    /// Empty model (predicts nothing).
    pub fn new() -> Self {
        Self::default()
    }

    /// Train from running text.
    pub fn train(&mut self, text: &str) {
        let tokens = tokenize(text);
        self.total_tokens += tokens.len() as u64;
        for w in &tokens {
            *self.unigrams.entry(w.clone()).or_insert(0) += 1;
            *self.vocab.entry(w.clone()).or_insert(0) += 1;
        }
        for pair in tokens.windows(2) {
            *self
                .bigrams
                .entry((pair[0].clone(), pair[1].clone()))
                .or_insert(0) += 1;
        }
        for triple in tokens.windows(3) {
            *self
                .trigrams
                .entry((triple[0].clone(), triple[1].clone(), triple[2].clone()))
                .or_insert(0) += 1;
        }
    }

    /// Train from a string, failing when it holds no tokens.
    pub fn from_text(text: &str) -> Result<Self, NgramError> {
        let mut model = Self::new();
        model.train(text);
        if model.total_tokens == 0 {
            return Err(NgramError::EmptyCorpus);
        }
        Ok(model)
    }

    /// Train from a plain-text corpus file.
    pub fn from_file(path: &Path) -> Result<Self, NgramError> {
        let text = std::fs::read_to_string(path).map_err(|source| NgramError::Io {
            path: path.to_string_lossy().into_owned(),
            source,
        })?;
        Self::from_text(&text)
    }

    /// Number of distinct words.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of training tokens.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Top words starting with `prefix` (unigram frequency order).
    pub fn complete(&self, prefix: &str, limit: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() || limit == 0 {
            return Vec::new();
        }
        let mut hits: Vec<(String, u64)> = self
            .vocab
            .range(prefix.to_string()..)
            .take_while(|(word, _)| word.starts_with(prefix))
            .map(|(word, count)| (word.clone(), *count))
            .collect();
        hits.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(limit);
        hits
    }

    /// Top likely next words after `prev` (up to 2 words of context).
    ///
    /// Backoff with tier-weighted scores so a longer context always beats a
    /// shorter one: trigram `1e6 + count`, bigram `1e3 + count`, unigram
    /// `count`. Within a tier, higher frequency wins, ties alphabetical.
    pub fn predict_next(&self, prev: &[String], limit: usize) -> Vec<(String, f32)> {
        if limit == 0 {
            return Vec::new();
        }
        let mut out: Vec<(String, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        if prev.len() >= 2 {
            let (a, b) = (&prev[prev.len() - 2], &prev[prev.len() - 1]);
            let mut tri: Vec<(String, u64)> = self
                .trigrams
                .iter()
                .filter(|((x, y, _), _)| x == a && y == b)
                .map(|((_, _, z), c)| (z.clone(), *c))
                .collect();
            tri.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
            for (w, c) in tri {
                if out.len() >= limit {
                    break;
                }
                if seen.insert(w.clone()) {
                    out.push((w, 1_000_000.0 + c as f32));
                }
            }
        }
        if !prev.is_empty() && out.len() < limit {
            let b = &prev[prev.len() - 1];
            let mut bi: Vec<(String, u64)> = self
                .bigrams
                .iter()
                .filter(|((x, _), _)| x == b)
                .map(|((_, z), c)| (z.clone(), *c))
                .collect();
            bi.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
            for (w, c) in bi {
                if out.len() >= limit {
                    break;
                }
                if seen.insert(w.clone()) {
                    out.push((w, 1_000.0 + c as f32));
                }
            }
        }
        if out.len() < limit {
            let mut uni: Vec<(String, u64)> = self
                .unigrams
                .iter()
                .map(|(w, c)| (w.clone(), *c))
                .collect();
            uni.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
            for (w, c) in uni {
                if out.len() >= limit {
                    break;
                }
                if seen.insert(w.clone()) {
                    out.push((w, c as f32));
                }
            }
        }
        out
    }

    fn score_completion(&self, word: &str, base_count: u64, prev: &[String]) -> f32 {
        let mut score = base_count as f32;
        if let Some(last) = prev.last() {
            if let Some(bigram_count) = self
                .bigrams
                .get(&(last.clone(), word.to_string()))
            {
                // Context boost so "quick" after "the" outranks a globally
                // more frequent but contextually wrong word.
                score += 2.0 * (*bigram_count as f32);
            }
        }
        score
    }
}

impl Predictor for NgramModel {
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate> {
        if ctx.sensitive {
            return Vec::new();
        }
        let prefix = current_prefix(&ctx.before);
        let prev = previous_words(&ctx.before, 2);
        let mut candidates: Vec<Candidate> = if prefix.is_empty() {
            self.predict_next(&prev, MAX_CANDIDATES)
                .into_iter()
                .map(|(text, score)| Candidate::new(text, score))
                .collect()
        } else {
            // Over-fetch, then re-rank with context boost.
            self.complete(&prefix, MAX_CANDIDATES * 4)
                .into_iter()
                .map(|(text, count)| {
                    let score = self.score_completion(&text, count, &prev);
                    Candidate::new(text, score)
                })
                .collect()
        };
        rank_candidates(&mut candidates);
        candidates.truncate(MAX_CANDIDATES);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "the quick brown fox jumps over the lazy dog \
         the quick brown cat sleeps all day \
         ich bin ein Berliner und du bist ein Hamburger";

    #[test]
    fn tokenize_handles_english_and_german() {
        assert_eq!(tokenize("Hallo, Welt!"), vec!["hallo", "welt"]);
        assert_eq!(tokenize("Grüße dich"), vec!["grüße", "dich"]);
        assert!(tokenize("   ... ").is_empty());
    }

    #[test]
    fn current_prefix_takes_trailing_fragment() {
        assert_eq!(current_prefix("hello wo"), "wo");
        assert_eq!(current_prefix("hello "), "");
        assert_eq!(current_prefix(""), "");
        assert_eq!(current_prefix("Grü"), "grü");
    }

    #[test]
    fn previous_words_skips_in_progress_prefix() {
        assert_eq!(
            previous_words("the quick bro", 2),
            vec!["the".to_string(), "quick".to_string()]
        );
        assert_eq!(
            previous_words("the quick ", 2),
            vec!["the".to_string(), "quick".to_string()]
        );
        assert!(previous_words("hel", 2).is_empty());
    }

    #[test]
    fn from_text_rejects_empty_corpus() {
        let err = NgramModel::from_text("  ...  ").unwrap_err();
        assert!(matches!(err, NgramError::EmptyCorpus));
    }

    #[test]
    fn complete_returns_frequency_order() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let hits = model.complete("th", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, "the");
    }

    #[test]
    fn predict_next_uses_bigram_context() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let next = model.predict_next(&["the".to_string()], 3);
        assert!(!next.is_empty());
        // "quick" follows "the" twice in SAMPLE.
        assert_eq!(next[0].0, "quick");
    }

    #[test]
    fn predict_next_backs_off_to_unigram() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let next = model.predict_next(&["zzz-unknown".to_string()], 2);
        assert_eq!(next[0].0, "the");
    }

    fn ctx(before: &str) -> Context {
        Context::new(
            "test",
            before,
            "",
            false,
            predict_core::ResolvedStyle::default_style(),
        )
    }

    #[test]
    fn complete_word_mid_word_suggests_completion() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let out = model.complete_word(&ctx("the qui"));
        assert!(!out.is_empty());
        assert_eq!(out[0].text, "quick");
        assert!(out.len() <= MAX_CANDIDATES);
    }

    #[test]
    fn complete_word_at_boundary_predicts_next() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let out = model.complete_word(&ctx("the "));
        assert!(!out.is_empty());
        assert_eq!(out[0].text, "quick");
    }

    #[test]
    fn complete_word_respects_sensitive_flag() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let sensitive = Context::new(
            "test",
            "the qui",
            "",
            true,
            predict_core::ResolvedStyle::default_style(),
        );
        assert!(model.complete_word(&sensitive).is_empty());
    }

    #[test]
    fn empty_model_predicts_nothing() {
        let model = NgramModel::new();
        assert!(model.complete_word(&ctx("hel")).is_empty());
        assert!(model.complete_word(&ctx("the ")).is_empty());
    }

    #[test]
    fn complete_word_p99_under_5ms_on_sample() {
        let model = NgramModel::from_text(SAMPLE).unwrap();
        let contexts = ["the qui", "hello wo", "ich b", "the ", "brown f", "du "];
        let mut latencies: Vec<u128> = Vec::new();
        for _ in 0..200 {
            for before in &contexts {
                let start = std::time::Instant::now();
                let _ = model.complete_word(&ctx(before));
                latencies.push(start.elapsed().as_micros());
            }
        }
        latencies.sort_unstable();
        let p99 = latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)];
        assert!(
            p99 < 5_000,
            "p99 {p99}µs exceeds 5ms budget"
        );
    }
}
