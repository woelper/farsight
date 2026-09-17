//! Fast tier: prefix completion + n-gram next-word prediction.
//!
//! Model: unigram / bigram / trigram counts over lowercased
//! unicode-alphanumeric tokens, prefix index as a `BTreeMap` range scan.
//! See ADR 0002 for the trade-offs.

use predict_core::{
    Candidate, Context, Predictor, rank_candidates, sentence_fragment, tokenize_text,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use thiserror::Error;

/// How many candidates `complete_word` returns at most.
pub const MAX_CANDIDATES: usize = 5;

/// Per-tier over-fetch for blending (distributions are truncated here).
const BLEND_FETCH: usize = 20;

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

/// Lowercase unicode-alphanumeric tokenizer (shared rule, see
/// [`predict_core::tokenize_text`]).
pub fn tokenize(text: &str) -> Vec<String> {
    tokenize_text(text)
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

/// Text language for base-model selection.
///
/// Interim heuristic until M5 (explicit/style/app defaults supersede it):
/// counts unambiguous function words. Ambiguous-across-languages words
/// (`was`, `her`, `will`, `in`, …) are in NEITHER list on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// English markers win.
    En,
    /// German markers win.
    De,
    /// Tie or no markers (e.g. empty context, single word).
    Unknown,
}

/// Common English function words that are rare in German text.
const EN_MARKERS: &[&str] = &[
    "the", "and", "of", "to", "you", "that", "it", "for", "with", "are", "have", "this", "they",
    "from", "his", "him", "our", "your", "them", "would", "there", "their", "what", "when",
    "which", "were", "been", "does", "into", "a",
];

/// Common German function words that are rare in English text (full
/// paradigms: detection counts exact tokens, so inflections are listed).
const DE_MARKERS: &[&str] = &[
    "der", "die", "das", "und", "ist", "nicht", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "dich", "sich", "uns", "euch", "mein", "dein", "sein", "den", "dem", "ein", "eine",
    "mit", "auf", "auch", "denn", "meine", "meinem", "meinen", "meiner", "meines", "deine",
    "deinem", "deinen", "deiner", "deines", "seine", "seinem", "seinen", "seiner", "seines",
    "kein", "keine", "keinem", "keinen", "keiner", "keines", "unsere", "unserem", "unseren",
    "unserer", "unseres", "unser", "ihre", "ihrem", "ihren", "ihrer", "ihres",
];

/// Detect English vs German from function-word counts (`Unknown` on ties,
/// including empty input).
pub fn detect_language(text: &str) -> Language {
    let (mut en, mut de) = (0u32, 0u32);
    for word in tokenize(text) {
        if EN_MARKERS.contains(&word.as_str()) {
            en += 1;
        }
        if DE_MARKERS.contains(&word.as_str()) {
            de += 1;
        }
    }
    match en.cmp(&de) {
        std::cmp::Ordering::Greater => Language::En,
        std::cmp::Ordering::Less => Language::De,
        std::cmp::Ordering::Equal => Language::Unknown,
    }
}

/// Base n-gram model: unigram + bigram counts plus pre-sorted indexes.
///
/// The raw trigram map is folded into [`Self::trigrams_by_pair`] at train
/// time: boundary queries (`predict_next`) must never scan whole tables
/// (that broke the p99 budget on the real corpus — see the `real_corpus`
/// test). All per-tier lists are sorted by count desc, text asc.
#[derive(Debug, Default, Clone)]
pub struct NgramModel {
    unigrams: HashMap<String, u64>,
    bigrams: HashMap<(String, String), u64>,
    vocab: BTreeMap<String, u64>,
    total_tokens: u64,
    trigrams_by_pair: HashMap<(String, String), Vec<(String, u64)>>,
    bigrams_by_first: HashMap<String, Vec<(String, u64)>>,
    unigrams_sorted: Vec<(String, u64)>,
}

/// Add token counts into raw maps (shared by [`NgramModel::train`] and
/// [`PersonalCounts::add_text`]).
fn absorb_counts(
    tokens: &[String],
    unigrams: &mut HashMap<String, u64>,
    bigrams: &mut HashMap<(String, String), u64>,
    trigrams: &mut HashMap<(String, String, String), u64>,
    vocab: &mut BTreeMap<String, u64>,
) {
    for w in tokens {
        *unigrams.entry(w.clone()).or_insert(0) += 1;
        *vocab.entry(w.clone()).or_insert(0) += 1;
    }
    for pair in tokens.windows(2) {
        *bigrams
            .entry((pair[0].clone(), pair[1].clone()))
            .or_insert(0) += 1;
    }
    for triple in tokens.windows(3) {
        *trigrams
            .entry((triple[0].clone(), triple[1].clone(), triple[2].clone()))
            .or_insert(0) += 1;
    }
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
        let mut trigrams = HashMap::new();
        absorb_counts(
            &tokens,
            &mut self.unigrams,
            &mut self.bigrams,
            &mut trigrams,
            &mut self.vocab,
        );
        self.rebuild_index(trigrams);
    }

    /// Fold raw counts into the sorted per-tier indexes.
    fn rebuild_index(&mut self, trigrams: HashMap<(String, String, String), u64>) {
        let mut tri: HashMap<(String, String), Vec<(String, u64)>> = HashMap::new();
        for ((a, b, c), count) in trigrams {
            tri.entry((a, b)).or_default().push((c, count));
        }
        for list in tri.values_mut() {
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
        self.trigrams_by_pair = tri;

        let mut bi: HashMap<String, Vec<(String, u64)>> = HashMap::new();
        for ((a, b), count) in &self.bigrams {
            bi.entry(a.clone()).or_default().push((b.clone(), *count));
        }
        for list in bi.values_mut() {
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
        self.bigrams_by_first = bi;

        let mut uni: Vec<(String, u64)> =
            self.unigrams.iter().map(|(w, c)| (w.clone(), *c)).collect();
        uni.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.unigrams_sorted = uni;
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

    /// Raw unigram count for one word (0 when unseen).
    pub fn unigram_count(&self, word: &str) -> u64 {
        self.unigrams.get(word).copied().unwrap_or(0)
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
    /// Served from pre-sorted indexes (no table scans).
    pub fn predict_next(&self, prev: &[String], limit: usize) -> Vec<(String, f32)> {
        if limit == 0 {
            return Vec::new();
        }
        let mut out: Vec<(String, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        if prev.len() >= 2 {
            let key = (prev[prev.len() - 2].clone(), prev[prev.len() - 1].clone());
            if let Some(tri) = self.trigrams_by_pair.get(&key) {
                for (w, c) in tri.iter().take(limit) {
                    if out.len() >= limit {
                        break;
                    }
                    if seen.insert(w.clone()) {
                        out.push((w.clone(), 1_000_000.0 + *c as f32));
                    }
                }
            }
        }
        if !prev.is_empty() && out.len() < limit {
            if let Some(bi) = self.bigrams_by_first.get(&prev[prev.len() - 1]) {
                // Full scan with dedup: earlier tiers may have taken words.
                for (w, c) in bi.iter() {
                    if out.len() >= limit {
                        break;
                    }
                    if seen.insert(w.clone()) {
                        out.push((w.clone(), 1_000.0 + *c as f32));
                    }
                }
            }
        }
        if out.len() < limit {
            for (w, c) in self.unigrams_sorted.iter() {
                if out.len() >= limit {
                    break;
                }
                if seen.insert(w.clone()) {
                    out.push((w.clone(), *c as f32));
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

    /// Word completion with personal blending (see [`BlendedPredictor`]).
    fn complete_word_with(
        &self,
        personal: Option<(&PersonalCounts, f32)>,
        ctx: &Context,
    ) -> Vec<Candidate> {
        if ctx.sensitive {
            return Vec::new();
        }
        let prefix = current_prefix(&ctx.before);
        let prev = previous_words(&ctx.before, 2);
        let blending = matches!(personal, Some((counts, lambda)) if !counts.is_empty() && lambda < 1.0);
        let mut candidates: Vec<Candidate> = if prefix.is_empty() {
            match personal {
                Some((counts, lambda)) if blending => self
                    .predict_next_blended(&prev, counts, lambda, MAX_CANDIDATES)
                    .into_iter()
                    .map(|(text, score)| Candidate::new(text, score))
                    .collect(),
                _ => self
                    .predict_next(&prev, MAX_CANDIDATES)
                    .into_iter()
                    .map(|(text, score)| Candidate::new(text, score))
                    .collect(),
            }
        } else {
            match personal {
                Some((counts, lambda)) if blending => self
                    .complete_blended(&prefix, &prev, counts, lambda, MAX_CANDIDATES)
                    .into_iter()
                    .map(|(text, score)| Candidate::new(text, score))
                    .collect(),
                _ => {
                    // Over-fetch, then re-rank with context boost.
                    self.complete(&prefix, MAX_CANDIDATES * 4)
                        .into_iter()
                        .map(|(text, count)| {
                            let score = self.score_completion(&text, count, &prev);
                            Candidate::new(text, score)
                        })
                        .collect()
                }
            }
        };
        rank_candidates(&mut candidates);
        candidates.truncate(MAX_CANDIDATES);
        candidates
    }
}

/// Personal n-gram counts learned from settled commits (see `predict-store`
/// for persistence). Same shapes as [`NgramModel`]; produced by the store,
/// consumed here so the two crates never form a cycle.
#[derive(Debug, Default, Clone)]
pub struct PersonalCounts {
    /// Word -> count (ordered for prefix scans).
    pub vocab: BTreeMap<String, u64>,
    unigrams: HashMap<String, u64>,
    bigrams: HashMap<(String, String), u64>,
    trigrams: HashMap<(String, String, String), u64>,
    total_tokens: u64,
}

impl PersonalCounts {
    /// Empty counts (blending falls back to the base model).
    pub fn new() -> Self {
        Self::default()
    }

    /// True when nothing was ever learned.
    pub fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    /// Total learned tokens.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Learn running text (one settled commit).
    pub fn add_text(&mut self, text: &str) {
        let tokens = tokenize(text);
        for window in tokens.windows(1) {
            self.add_ngram(1, "", &window[0], 1);
        }
        for pair in tokens.windows(2) {
            self.add_ngram(2, &pair[0], &pair[1], 1);
        }
        for triple in tokens.windows(3) {
            self.add_ngram(3, &format!("{} {}", triple[0], triple[1]), &triple[2], 1);
        }
    }

    /// Add one n-gram count directly (row-based loading, e.g. from SQLite).
    ///
    /// `n` is 1 (unigram, `ctx` ignored), 2 (bigram, `ctx` = previous word),
    /// or 3 (trigram, `ctx` = two previous words space-joined). Other orders
    /// are ignored.
    pub fn add_ngram(&mut self, n: u32, ctx: &str, word: &str, count: u64) {
        if word.is_empty() || count == 0 {
            return;
        }
        match n {
            1 => {
                *self.unigrams.entry(word.to_string()).or_insert(0) += count;
                *self.vocab.entry(word.to_string()).or_insert(0) += count;
                self.total_tokens += count;
            }
            2 => {
                *self
                    .bigrams
                    .entry((ctx.to_string(), word.to_string()))
                    .or_insert(0) += count;
            }
            3 => {
                let mut parts = ctx.splitn(2, ' ');
                let (a, b) = (
                    parts.next().unwrap_or(""),
                    parts.next().unwrap_or(""),
                );
                if a.is_empty() || b.is_empty() {
                    return;
                }
                *self
                    .trigrams
                    .entry((a.to_string(), b.to_string(), word.to_string()))
                    .or_insert(0) += count;
            }
            _ => {}
        }
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

    /// Top likely next words after `prev` (frequency order per tier).
    pub fn predict_next(&self, prev: &[String], limit: usize) -> Vec<(String, u64)> {
        if limit == 0 {
            return Vec::new();
        }
        let mut out: Vec<(String, u64)> = Vec::new();
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
                    out.push((w, c));
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
                    out.push((w, c));
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
                    out.push((w, c));
                }
            }
        }
        out
    }
}

/// One tier's word probability with Katz-style backoff: trigram conditional
/// when seen, else bigram conditional when seen, else unigram. No magic
/// weights — unseen contexts fall back cleanly.
fn tier_word_prob(
    unigrams: &HashMap<String, u64>,
    bigrams: &HashMap<(String, String), u64>,
    trigrams_by_pair: &HashMap<(String, String), Vec<(String, u64)>>,
    total_tokens: u64,
    prev: &[String],
    word: &str,
) -> f64 {
    if total_tokens == 0 {
        return 0.0;
    }
    let unigram = *unigrams.get(word).unwrap_or(&0) as f64 / total_tokens as f64;
    if prev.is_empty() {
        return unigram;
    }
    let last = prev[prev.len() - 1].clone();
    if prev.len() >= 2 {
        let first = prev[prev.len() - 2].clone();
        let triple = trigrams_by_pair
            .get(&(first.clone(), last.clone()))
            .and_then(|list| list.iter().find(|(w, _)| w == word));
        if let (Some((_, triple)), Some(&context)) =
            (triple, bigrams.get(&(first, last.clone())))
        {
            if *triple > 0 && context > 0 {
                return *triple as f64 / context as f64;
            }
        }
    }
    if let Some(&pair) = bigrams.get(&(last.clone(), word.to_string())) {
        if pair > 0 {
            if let Some(&context) = unigrams.get(&last) {
                if context > 0 {
                    return pair as f64 / context as f64;
                }
            }
        }
    }
    unigram
}

/// Personal tier's word probability: seen trigram/bigram conditionals only,
/// plus unigram mass for words the base model never saw (OOV vocabulary).
/// Deliberately NO blind unigram backoff: raw personal frequency would add
/// noise to every query on held-out text, while contextual matches and
/// genuinely new words are pure signal.
fn personal_word_prob(
    personal: &PersonalCounts,
    personal_tri: &HashMap<(String, String), Vec<(String, u64)>>,
    base_unigrams: &HashMap<String, u64>,
    prev: &[String],
    word: &str,
) -> f64 {
    if personal.total_tokens() == 0 {
        return 0.0;
    }
    if !prev.is_empty() {
        let last = &prev[prev.len() - 1];
        if prev.len() >= 2 {
            let first = &prev[prev.len() - 2];
            if let Some(list) = personal_tri.get(&(first.clone(), last.clone())) {
                if let Some((_, triple)) = list.iter().find(|(w, _)| w == word) {
                    if *triple > 0 {
                        if let Some(&context) =
                            personal.bigrams.get(&(first.clone(), last.clone()))
                        {
                            if context > 0 {
                                return *triple as f64 / context as f64;
                            }
                        }
                    }
                }
            }
        }
        if let Some(&pair) = personal
            .bigrams
            .get(&(last.clone(), word.to_string()))
        {
            if pair > 0 {
                if let Some(&context) = personal.unigrams.get(last) {
                    if context > 0 {
                        return pair as f64 / context as f64;
                    }
                }
            }
        }
    }
    // OOV: the base model has no opinion; personal frequency is the only
    // signal (and the only way user-invented words surface at all).
    if !base_unigrams.contains_key(word) {
        if let Some(&count) = personal.unigrams.get(word) {
            if count > 0 {
                return count as f64 / personal.total_tokens() as f64;
            }
        }
    }
    0.0
}

impl NgramModel {
    /// Blend one tier distribution pair:
    /// `p = lambda * p_base + (1 - lambda) * p_personal`
    /// over the union of both top-`fetch` sets (an approximation over
    /// truncated distributions — documented in ADR 0006).
    fn blend_scores(
        &self,
        personal: &PersonalCounts,
        lambda: f64,
        prev: &[String],
        base_words: Vec<String>,
        personal_words: Vec<String>,
    ) -> Vec<(String, f32)> {
        let mut words = base_words;
        for word in personal_words {
            if !words.iter().any(|w| w == &word) {
                words.push(word);
            }
        }
        // Personal trigram pairs, indexed on the fly (personal data is
        // small; the base tier uses its prebuilt index).
        let mut pers_tri: HashMap<(String, String), Vec<(String, u64)>> = HashMap::new();
        for ((a, b, c), count) in &personal.trigrams {
            pers_tri
                .entry((a.clone(), b.clone()))
                .or_default()
                .push((c.clone(), *count));
        }
        let mut scored: Vec<(String, f32)> = words
            .into_iter()
            .map(|word| {
                let base = tier_word_prob(
                    &self.unigrams,
                    &self.bigrams,
                    &self.trigrams_by_pair,
                    self.total_tokens,
                    prev,
                    &word,
                );
                let pers =
                    personal_word_prob(personal, &pers_tri, &self.unigrams, prev, &word);
                (word, (lambda * base + (1.0 - lambda) * pers) as f32)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored
    }

    /// Next-word prediction blended with personal counts.
    pub fn predict_next_blended(
        &self,
        prev: &[String],
        personal: &PersonalCounts,
        lambda: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        if limit == 0 {
            return Vec::new();
        }
        let base_words: Vec<String> = self
            .predict_next(prev, BLEND_FETCH)
            .into_iter()
            .map(|(word, _)| word)
            .collect();
        let personal_words: Vec<String> = personal
            .predict_next(prev, BLEND_FETCH)
            .into_iter()
            .map(|(word, _)| word)
            .collect();
        let mut scored = self.blend_scores(
            personal,
            f64::from(lambda),
            prev,
            base_words,
            personal_words,
        );
        scored.truncate(limit);
        scored
    }

    /// Prefix completion blended with personal counts.
    pub fn complete_blended(
        &self,
        prefix: &str,
        prev: &[String],
        personal: &PersonalCounts,
        lambda: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        if limit == 0 || prefix.is_empty() {
            return Vec::new();
        }
        let base_words: Vec<String> = self
            .complete(prefix, BLEND_FETCH)
            .into_iter()
            .map(|(word, _)| word)
            .collect();
        let personal_words: Vec<String> = personal
            .complete(prefix, BLEND_FETCH)
            .into_iter()
            .map(|(word, _)| word)
            .collect();
        let mut scored = self.blend_scores(
            personal,
            f64::from(lambda),
            prev,
            base_words,
            personal_words,
        );
        scored.truncate(limit);
        scored
    }
}

/// Word predictor blending a base model with personal counts:
///
/// `p = lambda * p_base + (1 - lambda) * p_personal`
///
/// Borrows both sides (no clones per query). An empty personal store — or
/// `lambda` clamped to 1 — behaves exactly like the base model.
pub struct BlendedPredictor<'a> {
    base: &'a NgramModel,
    personal: &'a PersonalCounts,
    lambda: f32,
}

impl<'a> BlendedPredictor<'a> {
    /// Build a blended predictor; `lambda` is clamped into `[0, 1]`
    /// (NaN counts as 1, i.e. base only).
    pub fn new(base: &'a NgramModel, personal: &'a PersonalCounts, lambda: f32) -> Self {
        let lambda = if lambda.is_nan() {
            1.0
        } else {
            lambda.clamp(0.0, 1.0)
        };
        Self {
            base,
            personal,
            lambda,
        }
    }
}

impl Predictor for BlendedPredictor<'_> {
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate> {
        self.base
            .complete_word_with(Some((self.personal, self.lambda)), ctx)
    }
}

/// One [`PersonalCounts`] per language side.
#[derive(Debug, Default, Clone)]
pub struct PersonalBundle {
    /// English-side counts.
    pub en: PersonalCounts,
    /// German-side counts.
    pub de: PersonalCounts,
}

impl PersonalBundle {
    /// Empty bundle (blending falls back to the base models).
    pub fn new() -> Self {
        Self::default()
    }
}

/// Word predictor routing by detected language, blending each side with
/// its personal counts. Borrows everything (no clones per query); shared
/// by the daemon and the eval harness.
pub struct LangBlended<'a> {
    en_base: &'a NgramModel,
    de_base: &'a NgramModel,
    personal: &'a PersonalBundle,
    lambda: f32,
}

impl<'a> LangBlended<'a> {
    /// Build a language-routing blended predictor (`lambda` clamped into
    /// `[0, 1]` per side, as in [`BlendedPredictor`]).
    pub fn new(
        en_base: &'a NgramModel,
        de_base: &'a NgramModel,
        personal: &'a PersonalBundle,
        lambda: f32,
    ) -> Self {
        Self {
            en_base,
            de_base,
            personal,
            lambda,
        }
    }

    /// Pick the (base, personal) pair for this context: the detected
    /// language, or — when unknown — the model whose top pick is relatively
    /// more frequent (unigram probability normalizes the corpora's sizes).
    /// Ties (and two empty lists) go English-first, deterministically, and
    /// the display is always single-language, never mixed.
    ///
    /// Detection runs on the current sentence fragment, not the whole
    /// field: language switches mid-text route per sentence instead of
    /// drowning in cumulative history.
    fn pick(&self, ctx: &Context) -> (&'a NgramModel, &'a PersonalCounts) {
        match detect_language(&sentence_fragment(&ctx.before)) {
            Language::En => (self.en_base, &self.personal.en),
            Language::De => (self.de_base, &self.personal.de),
            Language::Unknown => {
                let en_top = self.en_base.complete_word(ctx);
                let de_top = self.de_base.complete_word(ctx);
                let confidence = |model: &NgramModel, picks: &[Candidate]| -> f64 {
                    picks
                        .first()
                        .map(|c| {
                            model.unigram_count(&c.text) as f64
                                / model.total_tokens().max(1) as f64
                        })
                        .unwrap_or(0.0)
                };
                if confidence(self.en_base, &en_top) >= confidence(self.de_base, &de_top) {
                    (self.en_base, &self.personal.en)
                } else {
                    (self.de_base, &self.personal.de)
                }
            }
        }
    }
}

impl Predictor for LangBlended<'_> {
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate> {
        let (base, personal) = self.pick(ctx);
        BlendedPredictor::new(base, personal, self.lambda).complete_word(ctx)
    }
}

impl Predictor for NgramModel {
    fn complete_word(&self, ctx: &Context) -> Vec<Candidate> {
        self.complete_word_with(None, ctx)
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
    fn language_detection_picks_markers() {
        use Language::{De, En, Unknown};
        assert_eq!(detect_language(""), Unknown);
        assert_eq!(detect_language("hi"), Unknown);
        assert_eq!(detect_language("the quick brown fox"), En);
        assert_eq!(detect_language("The weather is nice"), En);
        assert_eq!(detect_language("der schnelle braune Fuchs"), De);
        assert_eq!(detect_language("Ich bin heute im Büro"), De);
        // Ambiguous words decide nothing; ties stay unknown.
        assert_eq!(detect_language("the der"), Unknown);
        assert_eq!(detect_language("it was und"), Unknown);
    }

    fn lang_models() -> (NgramModel, NgramModel) {
        let en = NgramModel::from_text("the quick brown fox jumps hello world").unwrap();
        let de = NgramModel::from_text("der schnelle braune fuchs springt hallo welt").unwrap();
        (en, de)
    }

    fn lang_ctx(before: &str) -> Context {
        Context::new(
            "test",
            before,
            "",
            false,
            predict_core::ResolvedStyle::default_style(),
        )
    }

    #[test]
    fn routing_follows_the_current_sentence() {
        let (en, de) = lang_models();
        let bundle = PersonalBundle::default();
        let predictor = LangBlended::new(&en, &de, &bundle, 0.5);
        // Cumulative history is English, but the live sentence is German.
        let out = predictor.complete_word(&lang_ctx("the weather is nice. der bra"));
        assert_eq!(out[0].text, "braune");
        // And mirrored.
        let out = predictor.complete_word(&lang_ctx("das Wetter ist schön. the qui"));
        assert_eq!(out[0].text, "quick");
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

    #[test]
    fn complete_word_p99_under_5ms_on_real_corpus() {
        // The production vocabulary is ~60x the sample's; the latency
        // contract must hold there too.
        let en = NgramModel::from_text(include_str!("../../../corpora/base_en.txt")).unwrap();
        let de = NgramModel::from_text(include_str!("../../../corpora/base_de.txt")).unwrap();
        assert!(en.vocab_size() > 1000);
        assert!(de.vocab_size() > 1000);
        let cases: &[(&NgramModel, &str)] = &[
            (&en, "the "),
            (&en, "sher"),
            (&en, "Alice was beginning to get very "),
            (&en, "q"),
            (&de, "der "),
            (&de, "Zarath"),
            (&de, "Also sprach "),
            (&de, "x"),
        ];
        let mut latencies: Vec<u128> = Vec::new();
        for _ in 0..100 {
            for (model, before) in cases {
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

    fn personal_of(text: &str) -> PersonalCounts {
        let mut personal = PersonalCounts::new();
        personal.add_text(text);
        personal
    }

    #[test]
    fn personal_counts_learn_text() {
        let personal = personal_of("hello world hello");
        assert!(!personal.is_empty());
        assert_eq!(personal.total_tokens(), 3);
        assert_eq!(personal.complete("he", 5)[0].0, "hello");
        assert!(PersonalCounts::new().is_empty());
    }

    #[test]
    fn blend_prefers_personal_context() {
        // Base ties left/right after "go"; personal always turns right, so
        // the blended top follows the user's habit, not the alphabet.
        let base = NgramModel::from_text("go left go right").unwrap();
        let personal = personal_of("go right go right");
        assert_eq!(base.complete_word(&ctx("go "))[0].text, "left");
        let blended = BlendedPredictor::new(&base, &personal, 0.5);
        assert_eq!(blended.complete_word(&ctx("go "))[0].text, "right");
    }

    #[test]
    fn blend_surfaces_oov_personal_words() {
        // "courgette" is unknown to the base model; personal frequency is
        // the only signal and must surface it on prefix.
        let base = NgramModel::from_text("the quick brown").unwrap();
        let personal = personal_of("courgette quiche");
        let blended = BlendedPredictor::new(&base, &personal, 0.5);
        let out = blended.complete_word(&ctx("cour"));
        assert!(!out.is_empty());
        assert_eq!(out[0].text, "courgette");
    }

    #[test]
    fn blend_uses_bigram_context() {
        // "courgette" is globally rare but always follows "grilled" for the user.
        let base = NgramModel::from_text("the the the the the soup soup soup").unwrap();
        let personal = personal_of("grilled courgette grilled courgette");
        let blended = BlendedPredictor::new(&base, &personal, 0.5);
        let out = blended.complete_word(&ctx("grilled "));
        assert!(!out.is_empty());
        assert_eq!(out[0].text, "courgette");
    }

    #[test]
    fn empty_personal_matches_base_exactly() {
        let base = NgramModel::from_text(SAMPLE).unwrap();
        let personal = PersonalCounts::new();
        let blended = BlendedPredictor::new(&base, &personal, 0.3);
        for before in ["the qui", "hello wo", "the ", "brown f", "xyz"] {
            assert_eq!(blended.complete_word(&ctx(before)), base.complete_word(&ctx(before)));
        }
    }

    #[test]
    fn lambda_clamps_into_unit_range() {
        let base = NgramModel::from_text(SAMPLE).unwrap();
        let personal = personal_of("zzz qqq");
        // lambda 1 (and NaN) behave base-only; 0 behaves personal-only.
        let full = BlendedPredictor::new(&base, &personal, 1.0);
        assert_eq!(full.complete_word(&ctx("the ")), base.complete_word(&ctx("the ")));
        let nan = BlendedPredictor::new(&base, &personal, f32::NAN);
        assert_eq!(nan.complete_word(&ctx("the ")), base.complete_word(&ctx("the ")));
        let zero = BlendedPredictor::new(&base, &personal, 0.0);
        assert_eq!(zero.complete_word(&ctx("zzz "))[0].text, "qqq");
    }

    #[test]
    fn blended_respects_sensitive() {
        let base = NgramModel::from_text(SAMPLE).unwrap();
        let personal = personal_of("hello world");
        let blended = BlendedPredictor::new(&base, &personal, 0.5);
        let sensitive = Context::new(
            "test",
            "hel",
            "",
            true,
            predict_core::ResolvedStyle::default_style(),
        );
        assert!(blended.complete_word(&sensitive).is_empty());
    }
}
