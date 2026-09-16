//! Manual eval run: train the n-gram tier on the sample corpus and replay it.
//!
//! Run with: `cargo run -p predict-eval --example eval_sample`

use predict_eval::evaluate;
use predict_ngram::NgramModel;
use std::error::Error;

const CORPUS: &str = include_str!("../../../corpora/sample_en_de.txt");

fn main() -> Result<(), Box<dyn Error>> {
    let model = NgramModel::from_text(CORPUS).map_err(|e| format!("train: {e}"))?;
    println!(
        "trained: {} tokens, {} distinct words",
        model.total_tokens(),
        model.vocab_size()
    );
    let metrics = evaluate(&model, CORPUS).map_err(|e| format!("eval: {e}"))?;
    println!("{metrics}");
    if metrics.latency_p99_ms() >= 5.0 {
        return Err(format!("p99 {:.3} ms exceeds 5 ms budget", metrics.latency_p99_ms()).into());
    }
    Ok(())
}
