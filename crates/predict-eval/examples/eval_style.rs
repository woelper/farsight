//! Style eval (M5): du/Sie violation rate (must be 0) plus acceptance per
//! style, measured with the real backend on German sample sentences split
//! by detected address form.
//!
//! Run with:
//! `cargo run -p predict-eval --example eval_style -- <model.gguf> [threshold]`
//! or set `PREDICT_MODEL_PATH`. Exits non-zero on any violation.

use predict_core::{AddressForm, LengthMode, ResolvedStyle, detect_address};
use predict_eval::{SentenceEvalOpts, evaluate_combined, split_sentences};
use predict_llm::{Backend, LlmConfig, llama::LlamaBackend};
use predict_ngram::{LangBlended, Language, NgramModel, PersonalBundle, detect_language};
use std::error::Error;

const CORPUS: &str = include_str!("../../../corpora/sample_en_de.txt");

fn style_for(form: AddressForm) -> ResolvedStyle {
    let id = match form {
        AddressForm::Du => "du",
        AddressForm::Sie => "sie",
        AddressForm::None => "default",
    };
    ResolvedStyle {
        style_id: id.to_string(),
        address_form: form,
        length: LengthMode::Sentence,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("PREDICT_MODEL_PATH").ok())
        .ok_or("usage: eval_style <model.gguf> [threshold]")?;
    let threshold: f32 = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("bad threshold: {e}"))?
        .unwrap_or(-1.5);

    // German sample sentences grouped by detected address form.
    let mut du_text = Vec::new();
    let mut sie_text = Vec::new();
    for sentence in split_sentences(CORPUS) {
        if detect_language(&sentence) != predict_ngram::Language::De {
            continue;
        }
        match detect_address(&sentence) {
            Some(AddressForm::Du) => du_text.push(sentence),
            Some(AddressForm::Sie) => sie_text.push(sentence),
            _ => {}
        }
    }
    println!("du sentences: {}, sie sentences: {}", du_text.len(), sie_text.len());
    if du_text.is_empty() || sie_text.is_empty() {
        return Err("need both du and sie sentences".into());
    }

    // Production-shaped word tier: per-language bases over the corpus.
    let mut en_corpus = String::new();
    let mut de_corpus = String::new();
    for sentence in split_sentences(CORPUS) {
        match detect_language(&sentence) {
            Language::En => {
                en_corpus.push_str(&sentence);
                en_corpus.push(' ');
            }
            Language::De => {
                de_corpus.push_str(&sentence);
                de_corpus.push(' ');
            }
            _ => {}
        }
    }
    let en_base = NgramModel::from_text(&en_corpus).map_err(|e| format!("en base: {e}"))?;
    let de_base = NgramModel::from_text(&de_corpus).map_err(|e| format!("de base: {e}"))?;
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
    let mut total_violations = 0;
    for (label, form, sents) in [
        ("du", AddressForm::Du, du_text.join(". ")),
        ("sie", AddressForm::Sie, sie_text.join(". ")),
    ] {
        let style = style_for(form);
        let report = evaluate_combined(&blended, &llm, &sents, &opts, None, Some(&style))
            .map_err(|e| format!("{label} eval: {e}"))?;
        println!(
            "\n=== {label} ({} sentences) ===\n{}",
            report.sentence.sentences, report.sentence
        );
        println!(
            "{label} word top-1: {:.3}, sentence acceptance: {:.3}",
            report.word.top1_rate(),
            report.sentence.acceptance_rate()
        );
        total_violations += report.sentence.violations;
    }
    println!("\ntotal violations: {total_violations}");
    if total_violations > 0 {
        return Err(format!("{total_violations} du/Sie violations").into());
    }
    Ok(())
}
