//! Sentence-tier eval: word-tier baseline vs word+LLM combined.
//!
//! Run with:
//! `cargo run -p predict-eval --example eval_sentence -- <model.gguf> [threshold]`
//! or set `PREDICT_MODEL_PATH` instead of passing the path.

use predict_eval::{SentenceEvalOpts, evaluate, evaluate_combined};
use predict_llm::{Backend, LlmConfig, llama::LlamaBackend};
use predict_ngram::NgramModel;
use std::error::Error;

const CORPUS: &str = include_str!("../../../corpora/sample_en_de.txt");

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("PREDICT_MODEL_PATH").ok())
        .ok_or("usage: eval_sentence <model.gguf> [confidence_threshold]")?;
    let threshold: f32 = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad threshold: {e}"))?
        .unwrap_or(-1.5);

    let word_model = NgramModel::from_text(CORPUS).map_err(|e| format!("train: {e}"))?;
    let baseline = evaluate(&word_model, CORPUS).map_err(|e| format!("baseline: {e}"))?;
    println!("=== word tier (M1 baseline) ===\n{baseline}\n");

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
    let combined = evaluate_combined(&word_model, &llm, CORPUS, &opts, None, None)
        .map_err(|e| format!("combined: {e}"))?;
    println!("\n=== combined (M3, threshold {threshold}) ===\n{combined}");
    println!(
        "\nLLM tier delta: {:+} keystrokes vs word-only",
        combined.saved_vs_baseline(&baseline)
    );
    Ok(())
}
