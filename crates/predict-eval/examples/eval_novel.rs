//! Novel accuracy: word top-1/top-3/savings plus sentence acceptance on
//! unseen long-form text, using production base models and no personal
//! data (honest generalization numbers).
//!
//! Run with:
//! `cargo run -p predict-eval --example eval_novel -- <model.gguf> <corpus.txt> [max-sentences] [threshold]`
//! or set `PREDICT_MODEL_PATH` instead of passing the model.

use predict_eval::{SentenceEvalOpts, evaluate_combined, split_sentences};
use predict_llm::{Backend, LlmConfig, llama::LlamaBackend};
use predict_ngram::{LangBlended, NgramModel, PersonalBundle};
use std::error::Error;

const BASE_EN: &str = include_str!("../../../corpora/base_en.txt");
const BASE_DE: &str = include_str!("../../../corpora/base_de.txt");

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("PREDICT_MODEL_PATH").ok())
        .ok_or("usage: eval_novel <model.gguf> <corpus.txt> [max-sentences] [threshold]")?;
    let corpus_path = args.get(2).cloned().ok_or("missing <corpus.txt>")?;
    let max_sentences: usize = args
        .get(3)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad max-sentences: {e}"))?
        .unwrap_or(100);
    let threshold: f32 = args
        .get(4)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad threshold: {e}"))?
        .unwrap_or(-1.5);

    let corpus = std::fs::read_to_string(&corpus_path).map_err(|e| format!("read corpus: {e}"))?;
    let sentences = split_sentences(&corpus);
    let sample: Vec<&str> = sentences
        .iter()
        .filter(|s| predict_core::tokenize_text(s).len() >= 4)
        .take(max_sentences)
        .map(String::as_str)
        .collect();
    if sample.is_empty() {
        return Err("no usable sentences".into());
    }
    println!(
        "{}: {} sampled sentences ({} total)",
        corpus_path,
        sample.len(),
        sentences.len()
    );

    let en_base = NgramModel::from_text(BASE_EN).map_err(|e| format!("en base: {e}"))?;
    let de_base = NgramModel::from_text(BASE_DE).map_err(|e| format!("de base: {e}"))?;
    let bundle = PersonalBundle::default();
    let blended = LangBlended::new(&en_base, &de_base, &bundle, 0.7);

    let config = LlmConfig {
        enabled: true,
        model_path,
        ..Default::default()
    };
    let llm = LlamaBackend::load(&config).map_err(|e| format!("load llm: {e}"))?;
    println!("backend: {}", llm.name());

    let opts = SentenceEvalOpts {
        confidence_threshold: threshold,
        ..Default::default()
    };
    let text = sample.join(". ");
    let report = evaluate_combined(&blended, &llm, &text, &opts, None, None)
        .map_err(|e| format!("eval: {e}"))?;
    println!("\n=== novel accuracy ===\n{report}");
    println!(
        "word top-1: {:.3}, top-3: {:.3}",
        report.word.top1_rate(),
        report.word.top3_rate()
    );
    Ok(())
}
