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
use predict_llm::{Backend, CancelToken, SentenceRequest};
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
    /// Slow-tier backend failed.
    #[error("llm error: {0}")]
    Llm(String),
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
        percentile_ms(&self.latencies_us, pct)
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

/// Nearest-rank percentile of microsecond latencies, in milliseconds.
fn percentile_ms(latencies_us: &[u64], pct: f64) -> f64 {
    if latencies_us.is_empty() {
        return 0.0;
    }
    let mut sorted = latencies_us.to_vec();
    sorted.sort_unstable();
    let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx] as f64 / 1000.0
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

/// Outcome of simulating one word keystroke by keystroke.
struct WordSimOutcome {
    queries: usize,
    top1_hits: usize,
    top3_hits: usize,
    /// Shortest prefix length with a top-1 hit, if any.
    best_cost: Option<usize>,
    len: usize,
}

/// Simulate typing one word: one `complete_word` query per prefix length
/// (`k = 0` is the word boundary). Shared by [`evaluate`] and
/// [`evaluate_combined`] so both agree on word-tier semantics.
fn simulate_word(
    predictor: &dyn Predictor,
    prev: &[String],
    target: &str,
    latencies_us: &mut Vec<u64>,
) -> WordSimOutcome {
    let target_chars: Vec<char> = target.chars().collect();
    let mut outcome = WordSimOutcome {
        queries: 0,
        top1_hits: 0,
        top3_hits: 0,
        best_cost: None,
        len: target_chars.len(),
    };
    for k in 0..target_chars.len() {
        let prefix: String = target_chars[..k].iter().collect();
        let before = build_before(prev, &prefix);
        let ctx = Context::new("eval", before, "", false, ResolvedStyle::default_style());

        let start = Instant::now();
        let candidates = predictor.complete_word(&ctx);
        latencies_us.push(start.elapsed().as_micros() as u64);
        outcome.queries += 1;

        if candidates.first().is_some_and(|c| c.text.as_str() == target) {
            outcome.top1_hits += 1;
            if outcome.best_cost.is_none() {
                outcome.best_cost = Some(k);
            }
        }
        if candidates.iter().take(3).any(|c| c.text.as_str() == target) {
            outcome.top3_hits += 1;
        }
    }
    outcome
}

/// Optimal-accept keystroke cost for one simulated word: shortest hitting
/// prefix plus one accept key, or the full length when never predicted.
fn word_cost(outcome: &WordSimOutcome) -> usize {
    match outcome.best_cost {
        Some(k) => (k + 1).min(outcome.len),
        None => outcome.len,
    }
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
    let dyn_predictor: &dyn Predictor = predictor;

    for (idx, target) in words.iter().enumerate() {
        let prev: Vec<String> = words[..idx].to_vec();
        let outcome = simulate_word(dyn_predictor, &prev, target, &mut metrics.latencies_us);
        metrics.queries += outcome.queries;
        metrics.top1_hits += outcome.top1_hits;
        metrics.top3_hits += outcome.top3_hits;
        metrics.keystrokes_total += outcome.len;
        metrics.keystrokes_with_prediction += word_cost(&outcome);
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

/// Options for the sentence-tier eval.
#[derive(Debug, Clone)]
pub struct SentenceEvalOpts {
    /// Words typed before the slow tier is triggered (default 3).
    pub trigger_words: usize,
    /// Generation cap per trigger.
    pub max_tokens: usize,
    /// Minimum mean token logprob to show a suggestion.
    pub confidence_threshold: f32,
    /// Cap on slow-tier calls (cost control); None means no cap.
    pub max_triggers: Option<usize>,
}

impl Default for SentenceEvalOpts {
    fn default() -> Self {
        Self {
            trigger_words: 3,
            max_tokens: 32,
            confidence_threshold: -1.0,
            max_triggers: None,
        }
    }
}

/// Sentence-tier results: acceptance proxy, wrong-suggestion rate, TTFT.
#[derive(Debug, Default, Clone)]
pub struct SentenceTierReport {
    /// Non-empty sentences processed.
    pub sentences: usize,
    /// Slow-tier calls issued.
    pub triggers: usize,
    /// Calls returning a suggestion (passed the gate).
    pub shown: usize,
    /// Shown suggestions matching ≥ 1 upcoming word.
    pub accepted_sentences: usize,
    /// Upcoming words matched across all triggers.
    pub accepted_words: usize,
    /// Matched chars (gross keystrokes saved by the tier).
    pub accepted_chars: usize,
    /// Accept-key cost (1 per accepted sentence).
    pub accept_keys: usize,
    /// Shown suggestions matching nothing.
    pub wrong: usize,
    /// Time to first token per call, microseconds.
    pub ttft_us: Vec<u64>,
}

impl SentenceTierReport {
    /// Fraction of triggers with ≥ 1 accepted word.
    pub fn acceptance_rate(&self) -> f64 {
        if self.triggers == 0 {
            0.0
        } else {
            self.accepted_sentences as f64 / self.triggers as f64
        }
    }

    /// Fraction of shown suggestions matching nothing.
    pub fn wrong_rate(&self) -> f64 {
        if self.shown == 0 {
            0.0
        } else {
            self.wrong as f64 / self.shown as f64
        }
    }

    /// Median time to first token in milliseconds.
    pub fn ttft_p50_ms(&self) -> f64 {
        percentile_ms(&self.ttft_us, 50.0)
    }

    /// p99 time to first token in milliseconds.
    pub fn ttft_p99_ms(&self) -> f64 {
        percentile_ms(&self.ttft_us, 99.0)
    }
}

impl fmt::Display for SentenceTierReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "sentences:          {}", self.sentences)?;
        writeln!(f, "triggers:           {}", self.triggers)?;
        writeln!(
            f,
            "shown:              {} ({:.3} of triggers)",
            self.shown,
            if self.triggers == 0 {
                0.0
            } else {
                self.shown as f64 / self.triggers as f64
            }
        )?;
        writeln!(
            f,
            "acceptance rate:    {:.3} ({}/{} triggers)",
            self.acceptance_rate(),
            self.accepted_sentences,
            self.triggers
        )?;
        writeln!(
            f,
            "wrong-sugg. rate:   {:.3} ({}/{} shown)",
            self.wrong_rate(),
            self.wrong,
            self.shown
        )?;
        writeln!(
            f,
            "accepted:           {} words / {} chars ({} accept keys)",
            self.accepted_words, self.accepted_chars, self.accept_keys
        )?;
        writeln!(f, "TTFT p50:           {:.1} ms", self.ttft_p50_ms())?;
        write!(f, "TTFT p99:           {:.1} ms", self.ttft_p99_ms())
    }
}

/// Word tier + sentence tier on one corpus.
///
/// Each sentence is word-simulated, except at most one slow-tier trigger
/// after [`SentenceEvalOpts::trigger_words`] words: when the continuation
/// matches upcoming words (acceptance proxy), those words are consumed at
/// one accept key instead of being word-simulated. Consumed words are
/// excluded from the word-tier counters, so [`CombinedReport::saved_vs_baseline`]
/// against [`evaluate`] on the same corpus is honest (positive = the LLM
/// tier saved keystrokes the word tier would have spent).
#[derive(Debug, Default, Clone)]
pub struct CombinedReport {
    /// Word-tier counters over non-consumed words only.
    pub word: EvalMetrics,
    /// Sentence-tier counters.
    pub sentence: SentenceTierReport,
    /// All corpus word chars, including sentence-consumed ones.
    pub total_chars: usize,
}

impl CombinedReport {
    /// Keystrokes with both tiers active.
    pub fn keystrokes_with_prediction(&self) -> usize {
        self.word.keystrokes_with_prediction + self.sentence.accept_keys
    }

    /// Combined keystroke savings rate.
    pub fn savings_rate(&self) -> f64 {
        if self.total_chars == 0 {
            0.0
        } else {
            1.0 - (self.keystrokes_with_prediction() as f64 / self.total_chars as f64)
        }
    }

    /// Extra keystrokes saved vs the word-only `baseline` on the same
    /// corpus. Positive means the LLM tier helped.
    pub fn saved_vs_baseline(&self, baseline: &EvalMetrics) -> i64 {
        baseline.keystrokes_with_prediction as i64 - self.keystrokes_with_prediction() as i64
    }
}

impl fmt::Display for CombinedReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- word tier (non-consumed words) ---")?;
        writeln!(
            f,
            "words sim'd:        {} ({} queries)",
            self.word.total_words, self.word.queries
        )?;
        writeln!(f, "--- sentence tier ---")?;
        writeln!(f, "{}", self.sentence)?;
        writeln!(f, "--- combined ---")?;
        writeln!(
            f,
            "keystrokes:         {} -> {}",
            self.total_chars,
            self.keystrokes_with_prediction()
        )?;
        write!(f, "combined savings:   {:.3}", self.savings_rate())
    }
}

/// Split a corpus into sentences on `.`, `!`, `?`, and newlines.
fn split_sentences(corpus: &str) -> Vec<String> {
    corpus
        .split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Longest common word-prefix length of two lowercased word sequences.
fn common_word_prefix_len(generated: &[String], actual: &[String]) -> usize {
    generated
        .iter()
        .zip(actual.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

/// Replay `corpus` with the word tier, triggering the slow tier once per
/// sentence after [`SentenceEvalOpts::trigger_words`] words.
pub fn evaluate_combined(
    word: &dyn Predictor,
    llm: &dyn Backend,
    corpus: &str,
    opts: &SentenceEvalOpts,
) -> Result<CombinedReport, EvalError> {
    let sentences = split_sentences(corpus);
    if sentences.iter().all(|s| split_words(s).is_empty()) {
        return Err(EvalError::EmptyCorpus);
    }
    let mut report = CombinedReport::default();
    // Global word history across sentences: mirrors evaluate()'s full-text
    // history so word-tier semantics (and the baseline comparison) agree.
    // Consumed words stay in history — the text exists either way.
    let mut history: Vec<String> = Vec::new();

    for sentence in &sentences {
        let words = split_words(sentence);
        if words.is_empty() {
            continue;
        }
        report.sentence.sentences += 1;
        report.total_chars += words.iter().map(|w| w.chars().count()).sum::<usize>();

        let mut idx = 0;
        while idx < words.len() {
            let under_cap = opts.max_triggers.is_none_or(|m| report.sentence.triggers < m);
            if idx == opts.trigger_words && under_cap {
                let before = history
                    .iter()
                    .chain(words[..idx].iter())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ");
                let req = SentenceRequest {
                    before,
                    max_tokens: opts.max_tokens,
                    confidence_threshold: opts.confidence_threshold,
                };
                let output = llm
                    .complete_sentence(&req, &CancelToken::new())
                    .map_err(|e| EvalError::Llm(e.to_string()))?;
                report.sentence.triggers += 1;
                if let Some(out) = output {
                    report.sentence.shown += 1;
                    report
                        .sentence
                        .ttft_us
                        .push(out.time_to_first_token.as_micros() as u64);
                    let generated = split_words(&out.text);
                    let matched = common_word_prefix_len(&generated, &words[idx..]);
                    if matched >= 1 {
                        let accepted = words[idx..idx + matched].join(" ");
                        report.sentence.accepted_sentences += 1;
                        report.sentence.accepted_words += matched;
                        report.sentence.accepted_chars += accepted.chars().count();
                        report.sentence.accept_keys += 1;
                        idx += matched;
                        continue;
                    }
                    report.sentence.wrong += 1;
                }
            }
            let mut prev = history.clone();
            prev.extend_from_slice(&words[..idx]);
            let outcome = simulate_word(word, &prev, &words[idx], &mut report.word.latencies_us);
            report.word.total_words += 1;
            report.word.queries += outcome.queries;
            report.word.top1_hits += outcome.top1_hits;
            report.word.top3_hits += outcome.top3_hits;
            report.word.keystrokes_total += outcome.len;
            report.word.keystrokes_with_prediction += word_cost(&outcome);
            idx += 1;
        }
        history.extend(words);
    }

    Ok(report)
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

    fn sentence_opts() -> SentenceEvalOpts {
        SentenceEvalOpts {
            trigger_words: 3,
            max_tokens: 32,
            confidence_threshold: -1.5,
            max_triggers: None,
        }
    }

    #[test]
    fn sentences_split_on_terminators_and_newlines() {
        let out = split_sentences("Hello world. How are you?\nFine!");
        assert_eq!(out, vec!["Hello world", "How are you", "Fine"]);
        assert!(split_sentences("... !!!").is_empty());
    }

    #[test]
    fn combined_rejects_empty_corpus() {
        let word = FixedPredictor { candidates: vec![] };
        let llm = predict_llm::StubBackend::empty();
        let err = evaluate_combined(&word, &llm, "  ...  ", &sentence_opts()).unwrap_err();
        assert!(matches!(err, EvalError::EmptyCorpus));
    }

    #[test]
    fn combined_matching_stub_beats_word_only() {
        // Word tier predicts nothing; stub continues "dd ee" at every
        // trigger, so each sentence consumes 2 words at 1 accept key.
        let word = FixedPredictor { candidates: vec![] };
        let llm = predict_llm::StubBackend::fixed("dd ee", -0.1);
        let corpus = "aa bb cc dd ee. aa bb cc dd ee.";
        let baseline = evaluate(&word, corpus).unwrap();
        let combined = evaluate_combined(&word, &llm, corpus, &sentence_opts()).unwrap();
        assert_eq!(combined.sentence.sentences, 2);
        assert_eq!(combined.sentence.triggers, 2);
        assert_eq!(combined.sentence.shown, 2);
        assert_eq!(combined.sentence.accepted_sentences, 2);
        assert_eq!(combined.sentence.accepted_words, 4);
        assert_eq!(combined.sentence.wrong, 0);
        // Baseline: 10 words x 2 chars = 20 keystrokes, nothing saved.
        assert_eq!(baseline.keystrokes_with_prediction, 20);
        // Combined: 6 sim'd words (12) + 2 accept keys = 14 -> saves 6.
        assert_eq!(combined.keystrokes_with_prediction(), 14);
        assert_eq!(combined.saved_vs_baseline(&baseline), 6);
        assert!((combined.sentence.acceptance_rate() - 1.0).abs() < 1e-9);
        assert_eq!(combined.sentence.wrong_rate(), 0.0);
    }

    #[test]
    fn combined_empty_stub_matches_baseline_exactly() {
        let word = FixedPredictor {
            candidates: vec![Candidate::new("aa", 1.0)],
        };
        let llm = predict_llm::StubBackend::empty();
        let corpus = "aa bb cc dd ee. aa bb cc dd ee.";
        let baseline = evaluate(&word, corpus).unwrap();
        let combined = evaluate_combined(&word, &llm, corpus, &sentence_opts()).unwrap();
        assert_eq!(combined.sentence.triggers, 2);
        assert_eq!(combined.sentence.shown, 0);
        assert_eq!(
            combined.keystrokes_with_prediction(),
            baseline.keystrokes_with_prediction
        );
        assert_eq!(combined.saved_vs_baseline(&baseline), 0);
    }

    #[test]
    fn combined_counts_wrong_suggestions() {
        let word = FixedPredictor { candidates: vec![] };
        let llm = predict_llm::StubBackend::fixed("zzz qqq", -0.1);
        let corpus = "aa bb cc dd ee.";
        let baseline = evaluate(&word, corpus).unwrap();
        let combined = evaluate_combined(&word, &llm, corpus, &sentence_opts()).unwrap();
        assert_eq!(combined.sentence.shown, 1);
        assert_eq!(combined.sentence.accepted_sentences, 0);
        assert_eq!(combined.sentence.wrong, 1);
        assert!((combined.sentence.wrong_rate() - 1.0).abs() < 1e-9);
        assert_eq!(combined.saved_vs_baseline(&baseline), 0);
    }

    #[test]
    fn combined_respects_max_triggers() {
        let word = FixedPredictor { candidates: vec![] };
        let llm = predict_llm::StubBackend::fixed("dd ee", -0.1);
        let corpus = "aa bb cc dd ee. aa bb cc dd ee. aa bb cc dd ee.";
        let mut opts = sentence_opts();
        opts.max_triggers = Some(1);
        let combined = evaluate_combined(&word, &llm, corpus, &opts).unwrap();
        assert_eq!(combined.sentence.triggers, 1);
        assert_eq!(combined.sentence.accepted_sentences, 1);
        // Word tier still simulates every non-consumed word.
        assert_eq!(combined.word.total_words, 15 - 2);
    }
}
