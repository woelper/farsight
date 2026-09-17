//! Temporal-split eval: learn the first 75% of a corpus into a
//! store, test word+sentence tiers on the last 25%.
//!
//! The M3-equivalent baseline uses one mixed base model (train text only);
//! the M4 system uses per-language bases plus personal counts and
//! retrieval grounding — mirroring the daemon.
//!
//! Run with:
//! `cargo run -p predict-eval --example eval_temporal -- <model.gguf|none> [threshold] [lambda] [corpus.txt] [max-test-sentences]`
//! or set `PREDICT_MODEL_PATH` instead of passing the path. `none` uses a
//! stub backend (word tiers only — seconds, no model needed).

use predict_eval::{SentenceEvalOpts, evaluate_temporal, split_sentences};
use predict_llm::{Backend, LlmConfig, llama::LlamaBackend};
use predict_ngram::{LangBlended, Language, NgramModel, PersonalBundle, detect_language};
use predict_store::Store;
use std::error::Error;

const CORPUS: &str = include_str!("../../../corpora/sample_en_de.txt");

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("PREDICT_MODEL_PATH").ok())
        .ok_or("usage: eval_temporal <model.gguf|none> [threshold] [lambda] [corpus.txt] [max-test-sentences]")?;
    let max_test: usize = args
        .get(5)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad max-test-sentences: {e}"))?
        .unwrap_or(usize::MAX);
    let threshold: f32 = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad threshold: {e}"))?
        .unwrap_or(-1.5);
    let lambda: f32 = args
        .get(3)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad lambda: {e}"))?
        .unwrap_or(0.7);
    let corpus: String = match args.get(4) {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("read corpus: {e}"))?,
        None => CORPUS.to_string(),
    };
    let corpus: &str = &corpus;

    // Temporal split by whole sentences: first ~75% of words is the past
    // (learn), the rest is the future. Sentence granularity preserves
    // terminators (trigger points) and matches production commits.
    let all_sentences = split_sentences(corpus);
    let total_words: usize = all_sentences
        .iter()
        .map(|s| predict_core::tokenize_text(s).len())
        .sum();
    let target = total_words * 3 / 4;
    let mut train_sentences: Vec<String> = Vec::new();
    let mut test_sentences: Vec<String> = Vec::new();
    let mut seen = 0usize;
    for sentence in all_sentences {
        if seen < target {
            seen += predict_core::tokenize_text(&sentence).len();
            train_sentences.push(sentence);
        } else {
            test_sentences.push(sentence);
        }
    }
    if test_sentences.is_empty() {
        return Err("no held-out sentences".into());
    }
    test_sentences.truncate(max_test);
    let train_text = train_sentences.join(" ");
    let test_text = test_sentences.join(". ");
    println!(
        "train: {} sentences ({} words), test: {} sentences",
        train_sentences.len(),
        seen,
        test_sentences.len()
    );

    // M3-equivalent word tier: one mixed base model on train text only.
    let base_mixed =
        NgramModel::from_text(&train_text).map_err(|e| format!("train base: {e}"))?;

    // M4 word tier: per-language bases (like the daemon) plus personal
    // counts learned from settled train sentences.
    let mut en_corpus = String::new();
    let mut de_corpus = String::new();
    let store = Store::open_in_memory().map_err(|e| format!("store: {e}"))?;
    let mut commits = 0;
    for sentence in &train_sentences {
        if store
            .commit(sentence, "default")
            .map_err(|e| format!("commit: {e}"))?
        {
            commits += 1;
        }
        match detect_language(sentence) {
            Language::En => {
                en_corpus.push_str(sentence);
                en_corpus.push(' ');
            }
            Language::De => {
                de_corpus.push_str(sentence);
                de_corpus.push(' ');
            }
            Language::Unknown => {
                en_corpus.push_str(sentence);
                en_corpus.push(' ');
                de_corpus.push_str(sentence);
                de_corpus.push(' ');
            }
        }
    }
    let en_base =
        NgramModel::from_text(&en_corpus).map_err(|e| format!("train en base: {e}"))?;
    let de_base =
        NgramModel::from_text(&de_corpus).map_err(|e| format!("train de base: {e}"))?;
    let bundle = PersonalBundle {
        en: store
            .personal_counts(Language::En, None)
            .map_err(|e| format!("en counts: {e}"))?,
        de: store
            .personal_counts(Language::De, None)
            .map_err(|e| format!("de counts: {e}"))?,
    };
    println!("settled commits: {commits}");
    let blended = LangBlended::new(&en_base, &de_base, &bundle, lambda);

    let config = LlmConfig {
        enabled: true,
        model_path: model_path.clone(),
        ..Default::default()
    };
    let real_llm = if model_path == "none" {
        None
    } else {
        Some(LlamaBackend::load(&config).map_err(|e| format!("load llm: {e}"))?)
    };
    if let Some(llm) = &real_llm {
        println!("backend: {}", llm.name());
    } else {
        println!("backend: stub (word tiers only, no slow tier)");
    }
    let stub = predict_llm::StubBackend::empty();
    let llm: &dyn predict_llm::Backend = match &real_llm {
        Some(llm) => llm,
        None => &stub,
    };

    let opts = SentenceEvalOpts {
        confidence_threshold: threshold,
        ..Default::default()
    };
    let retrieval = |before: &str| {
        store
            .search(&predict_core::sentence_fragment(before), 3, None)
            .unwrap_or_default()
    };
    let report = evaluate_temporal(&base_mixed, &blended, llm, &test_text, &opts, Some(&retrieval), None)
        .map_err(|e| format!("temporal: {e}"))?;
    println!("\n=== temporal (lambda {lambda}, threshold {threshold}) ===\n{report}");
    Ok(())
}
