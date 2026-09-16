//! Offline replay harness: type a corpus character by character,
//! query a [`Predictor`](predict_core::Predictor) at every prefix,
//! and report keystroke savings, hit rates, and latency.
//!
//! Simulation model (see ADR 0002):
//! - The corpus is lowercased and split into words; context is rebuilt with
//!   single spaces, so `before` at each query is
//!   `prev words + current prefix`.
//! - At every prefix length `k` (0 = word boundary) one `complete_word`
//!   query is issued. Top-1 / top-3 hit rates are per query.
//! - Keystroke savings assume optimal use: per word, the shortest prefix
//!   where top-1 equals the target word costs `k + 1` keystrokes
//!   (prefix + one accept key) instead of the full word length.

use predict_core::{Context, Predictor, ResolvedStyle};
use std::fmt;
use std::path::Path;
use std::time::Instant;
use thiserror::Error;

/// Errors from the eval harness.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Corpus file could not be read.
    #[error("cannot read corpus file {path}: {source}")]
    Io {
        /// File that failed to load.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Corpus contained no words.
    #[error("corpus is empty")]
    EmptyCorpus,
}

/// Aggregate result of one eval run.
#[derive(Debug, Default, Clone)]
pub struct EvalMetrics {
    /// Words simulated.
    pub total_words: usize,
    /// `complete_word` queries issued.
    pub queries: usize,
    /// Queries where the top candidate equaled the target word.
    pub top1_hits: usize,
    /// Queries where any of the top 3 equaled the target word.
    pub top3_hits: usize,
    /// Sum of word lengths (chars).
    pub keystrokes_total: usize,
    /// Keystrokes with optimal accept (`k + 1` per predicted word).
    pub keystrokes_with_prediction: usize,
    /// Per-query latency in microseconds.
    pub latencies_us: Vec<u64>,
}

impl EvalMetrics {
    /// Fraction of queries with the target as top-1.
    pub fn top1_rate(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.top1_hits as f64 / self.queries as f64
        }
    }

    /// Fraction of queries with the target in the top 3.
    pub fn top3_rate(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.top3_hits as f64 / self.queries as f64
        }
    }

    /// Fraction of keystrokes saved under optimal accept.
    pub fn keystroke_savings_rate(&self) -> f64 {
        if self.keystrokes_total == 0 {
            0.0
        } else {
            1.0 - (self.keystrokes_with_prediction as f64 / self.keystrokes_total as f64)
        }
    }

    /// Nearest-rank percentile of latency in milliseconds.
    fn percentile_ms(&self, pct: f64) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies_us.clone();
        sorted.sort_unstable();
        let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(sorted.len() - 1);
        sorted[idx] as f64 / 1000.0
    }

    /// Median `complete_word` latency in milliseconds.
    pub fn latency_p50_ms(&self) -> f64 {
        self.percentile_ms(50.0)
    }

    /// p99 `complete_word` latency in milliseconds.
    pub fn latency_p99_ms(&self) -> f64 {
        self.percentile_ms(99.0)
    }
}

impl fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "words:              {}", self.total_words)?;
        writeln!(f, "queries:            {}", self.queries)?;
        writeln!(
            f,
            "top-1 hit rate:     {:.3} ({}/{})",
            self.top1_rate(),
            self.top1_hits,
            self.queries
        )?;
        writeln!(
            f,
            "top-3 hit rate:     {:.3} ({}/{})",
            self.top3_rate(),
            self.top3_hits,
            self.queries
        )?;
        writeln!(
            f,
            "keystroke savings:  {:.3} ({} -> {} keystrokes)",
            self.keystroke_savings_rate(),
            self.keystrokes_total,
            self.keystrokes_with_prediction
        )?;
        writeln!(f, "latency p50:        {:.3} ms", self.latency_p50_ms())?;
        write!(f, "latency p99:        {:.3} ms", self.latency_p99_ms())
    }
}

/// Lowercase alphanumeric word splitter (same rule as the n-gram tier).
fn split_words(corpus: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for c in corpus.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                current.push(lc);
            }
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Rebuild `before` text for a query: previous words plus the first `k`
/// chars of the current word.
fn build_before(prev: &[String], prefix: &str) -> String {
    let mut before = prev.join(" ");
    if !prev.is_empty() {
        before.push(' ');
    }
    before.push_str(prefix);
    before
}

/// Replay `corpus` as simulated typing against `predictor`.
pub fn evaluate(predictor: &impl Predictor, corpus: &str) -> Result<EvalMetrics, EvalError> {
    let words = split_words(corpus);
    if words.is_empty() {
        return Err(EvalError::EmptyCorpus);
    }

    let mut metrics = EvalMetrics {
        total_words: words.len(),
        ..Default::default()
    };

    for (idx, target) in words.iter().enumerate() {
        let prev: Vec<String> = words[..idx].to_vec();
        let target_chars: Vec<char> = target.chars().collect();
        let mut best_cost: Option<usize> = None;

        // Prefix lengths 0 (word boundary) .. len-1 (last missing char).
        for k in 0..target_chars.len() {
            let prefix: String = target_chars[..k].iter().collect();
            let before = build_before(&prev, &prefix);
            let ctx = Context::new("eval", before, "", false, ResolvedStyle::default_style());

            let start = Instant::now();
            let candidates = predictor.complete_word(&ctx);
            metrics.latencies_us.push(start.elapsed().as_micros() as u64);
            metrics.queries += 1;

            if candidates.first().is_some_and(|c| &c.text == target) {
                metrics.top1_hits += 1;
                if best_cost.is_none() {
                    best_cost = Some(k);
                }
            }
            if candidates.iter().take(3).any(|c| &c.text == target) {
                metrics.top3_hits += 1;
            }
        }

        let len = target_chars.len();
        metrics.keystrokes_total += len;
        let cost = match best_cost {
            Some(k) => (k + 1).min(len),
            None => len,
        };
        metrics.keystrokes_with_prediction += cost;
    }

    Ok(metrics)
}

/// Replay a corpus file as simulated typing.
pub fn evaluate_file(
    predictor: &impl Predictor,
    path: &Path,
) -> Result<EvalMetrics, EvalError> {
    let text = std::fs::read_to_string(path).map_err(|source| EvalError::Io {
        path: path.to_string_lossy().into_owned(),
        source,
    })?;
    evaluate(predictor, &text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use predict_core::Candidate;

    /// Stub predictor returning a fixed list per query (ignores context).
    struct FixedPredictor {
        candidates: Vec<Candidate>,
    }

    impl Predictor for FixedPredictor {
        fn complete_word(&self, _ctx: &Context) -> Vec<Candidate> {
            self.candidates.clone()
        }
    }

    #[test]
    fn empty_corpus_is_an_error() {
        let p = FixedPredictor { candidates: vec![] };
        let err = evaluate(&p, "  ...  ").unwrap_err();
        assert!(matches!(err, EvalError::EmptyCorpus));
    }

    #[test]
    fn hit_rates_count_per_query() {
        // Corpus "ab cd": queries = 2 + 2 = 4. Stub always says ["ab"].
        let p = FixedPredictor {
            candidates: vec![Candidate::new("ab", 1.0)],
        };
        let m = evaluate(&p, "ab cd").unwrap();
        assert_eq!(m.total_words, 2);
        assert_eq!(m.queries, 4);
        // "ab" matches at both its prefixes ("", "a"); "cd" never.
        assert_eq!(m.top1_hits, 2);
        assert_eq!(m.top3_hits, 2);
    }

    #[test]
    fn keystroke_savings_assumes_optimal_accept() {
        // "ab" predicted from k=0 costs 1 instead of 2; "cd" never costs 2.
        let p = FixedPredictor {
            candidates: vec![Candidate::new("ab", 1.0)],
        };
        let m = evaluate(&p, "ab cd").unwrap();
        assert_eq!(m.keystrokes_total, 4);
        assert_eq!(m.keystrokes_with_prediction, 1 + 2);
        assert!((m.keystroke_savings_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn top3_counts_when_target_is_not_first() {
        let p = FixedPredictor {
            candidates: vec![
                Candidate::new("xx", 3.0),
                Candidate::new("yy", 2.0),
                Candidate::new("ab", 1.0),
            ],
        };
        let m = evaluate(&p, "ab").unwrap();
        assert_eq!(m.top1_hits, 0);
        assert_eq!(m.top3_hits, 2); // both prefixes of "ab"
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let m = EvalMetrics {
            latencies_us: vec![1000, 2000, 3000, 4000],
            ..Default::default()
        };
        assert!((m.latency_p50_ms() - 2.0).abs() < 1e-9);
        assert!((m.latency_p99_ms() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn empty_metrics_report_zero_rates() {
        let m = EvalMetrics::default();
        assert_eq!(m.top1_rate(), 0.0);
        assert_eq!(m.top3_rate(), 0.0);
        assert_eq!(m.keystroke_savings_rate(), 0.0);
        assert_eq!(m.latency_p50_ms(), 0.0);
        assert_eq!(m.latency_p99_ms(), 0.0);
    }

    /// End-to-end: train on the sample corpus, replay it, stay in budget.
    #[test]
    fn end_to_end_on_sample_corpus_meets_latency_budget() {
        let corpus = include_str!("../../../corpora/sample_en_de.txt");
        let model =
            predict_ngram::NgramModel::from_text(corpus).expect("sample corpus trains");
        let metrics = evaluate(&model, corpus).expect("eval runs");
        assert!(metrics.queries > 100, "queries = {}", metrics.queries);
        assert!(
            metrics.latency_p99_ms() < 5.0,
            "p99 = {:.3} ms",
            metrics.latency_p99_ms()
        );
        // Sanity: a model replaying its own training text must save
        // keystrokes and hit sometimes.
        assert!(metrics.keystroke_savings_rate() > 0.0);
        assert!(metrics.top1_rate() > 0.0);
    }
}
