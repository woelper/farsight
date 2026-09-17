//! llama.cpp implementation of [`Backend`](super::Backend).
//!
//! All `!Send` llama.cpp state (context, sampler, batch) lives on one
//! dedicated worker thread that owns the KV cache across requests; the
//! public [`LlamaBackend`] handle is `Send + Sync` and blocks on a reply
//! channel. Every job resolves its reply exactly once, so callers never
//! hang — cancellation included.

use super::{
    Backend, CancelToken, LlmConfig, LlmError, SentenceOutput, SentenceRequest, StopMode,
    piece_continues_word, strip_fragment,
};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaCppBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::logit_bias::LlamaLogitBias;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// llama.cpp sequence id used for all single-stream inference.
const SEQ_ID: i32 = 0;
/// Prompt tail kept in characters (char-boundary safe truncation).
const PROMPT_MAX_CHARS: usize = 1024;
/// Prompt tail kept in tokens after tokenization.
const PROMPT_MAX_TOKENS: usize = 384;
/// Distribution scan width for token healing.
const HEALING_TOP_K: usize = 200;
/// Scratch buffer for one token piece (bytes).
const PIECE_BUF: usize = 64;

/// Work item for the inference thread.
struct Job {
    before: String,
    max_tokens: usize,
    threshold: f32,
    stop: StopMode,
    address: predict_core::AddressForm,
    cancel: CancelToken,
    reply: mpsc::Sender<Result<Option<SentenceOutput>, LlmError>>,
}

/// llama.cpp backend. Cheap to share (`Send + Sync`); all inference runs
/// on the internal worker thread, so concurrent callers are serialized and
/// share one KV cache.
pub struct LlamaBackend {
    tx: mpsc::Sender<Job>,
}

impl LlamaBackend {
    /// Load the GGUF model and start the worker thread.
    pub fn load(config: &LlmConfig) -> Result<Self, LlmError> {
        if !Path::new(&config.model_path).exists() {
            return Err(LlmError::ModelNotFound(config.model_path.clone()));
        }
        let (tx, rx) = mpsc::channel::<Job>();
        let cfg = config.clone();
        std::thread::spawn(move || worker_main(cfg, rx));
        Ok(Self { tx })
    }
}

impl Backend for LlamaBackend {
    fn complete_sentence(
        &self,
        req: &SentenceRequest,
        cancel: &CancelToken,
    ) -> Result<Option<SentenceOutput>, LlmError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let job = Job {
            before: req.before.clone(),
            max_tokens: req.max_tokens,
            threshold: req.confidence_threshold,
            stop: req.stop,
            address: req.address,
            cancel: cancel.clone(),
            reply: reply_tx,
        };
        self.tx
            .send(job)
            .map_err(|_| LlmError::Inference("llm worker is gone".to_string()))?;
        reply_rx
            .recv()
            .map_err(|_| LlmError::Inference("llm worker dropped the reply".to_string()))?
    }

    fn name(&self) -> &str {
        "llama.cpp"
    }
}

/// Worker entry: model, backend, and context all live as thread locals
/// (the context borrows the model, so they can never sit in one struct).
/// Serves jobs until all handles are dropped; a failed open poisons every
/// queued job with the error.
fn worker_main(config: LlmConfig, rx: mpsc::Receiver<Job>) {
    let backend = match LlamaCppBackend::init() {
        Ok(backend) => backend,
        Err(e) => {
            drain_with_error(rx, LlmError::LoadFailed(e.to_string()));
            return;
        }
    };
    let model = match LlamaModel::load_from_file(
        &backend,
        Path::new(&config.model_path),
        &LlamaModelParams::default(),
    ) {
        Ok(model) => model,
        Err(e) => {
            drain_with_error(rx, LlmError::LoadFailed(e.to_string()));
            return;
        }
    };
    let params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(config.n_ctx))
        .with_n_threads(config.resolved_threads() as i32);
    let mut ctx = match model.new_context(&backend, params) {
        Ok(ctx) => ctx,
        Err(e) => {
            drain_with_error(rx, LlmError::LoadFailed(e.to_string()));
            return;
        }
    };
    let mut cached: Vec<LlamaToken> = Vec::new();
    for job in rx {
        let result = process_job(&model, &mut ctx, &mut cached, &config, &job);
        let _ = job.reply.send(result);
    }
}

/// Reply `err` to every queued job, then return (open-failure path).
fn drain_with_error(rx: mpsc::Receiver<Job>, err: LlmError) {
    for job in rx {
        let _ = job.reply.send(Err(err.clone()));
    }
}

/// Shared-prefix length of two token sequences.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Keep the last `max` characters on a char boundary.
fn truncate_to_last_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    text.chars().skip(count - max).collect()
}

/// Decode `buf[emitted..]` up to the last complete UTF-8 char, advancing
/// `emitted`. Invalid bytes are skipped (replaced by nothing); incomplete
/// suffixes wait for the next token.
fn emit_complete(buf: &[u8], emitted: &mut usize) -> String {
    let rest = match buf.get(*emitted..) {
        Some(rest) => rest,
        None => return String::new(),
    };
    match std::str::from_utf8(rest) {
        Ok(s) => {
            *emitted = buf.len();
            s.to_string()
        }
        Err(e) => {
            let valid = e.valid_up_to();
            let skip = e.error_len().unwrap_or(0);
            let out = std::str::from_utf8(&rest[..valid]).unwrap_or("").to_string();
            *emitted += valid + skip;
            out
        }
    }
}

/// Sentence-terminal punctuation (stop AFTER including it).
fn is_sentence_end(text: &str) -> bool {
    text.ends_with(['.', '!', '?', '…'])
}

/// Clause-terminal punctuation for [`StopMode::Clause`] (stop AFTER
/// including it; sentence ends count as clause ends too).
fn is_clause_end(text: &str) -> bool {
    text.ends_with([',', ';', ':', '.', '!', '?', '…'])
}

/// Single-token ids spelling banned words (each in bare and space-prefixed
/// piece form). Multi-token spellings are left to the post-generation
/// check — logit bias can only ban whole tokens.
fn banned_token_ids(model: &LlamaModel, address: predict_core::AddressForm) -> Vec<LlamaLogitBias> {
    let (words, _) = predict_core::banned_words(address);
    let mut ids = Vec::new();
    let mut seen: Vec<LlamaToken> = Vec::new();
    for word in words {
        for form in [(*word).to_string(), format!(" {word}")] {
            if let Ok(tokens) = model.str_to_token(&form, AddBos::Never) {
                if let [single] = tokens.as_slice() {
                    if !seen.contains(single) {
                        seen.push(*single);
                        ids.push(LlamaLogitBias::new(*single, f32::NEG_INFINITY));
                    }
                }
            }
        }
    }
    ids
}

#[allow(clippy::too_many_lines)]
fn process_job(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    cached: &mut Vec<LlamaToken>,
    config: &LlmConfig,
    job: &Job,
) -> Result<Option<SentenceOutput>, LlmError> {
    let start = Instant::now();
    if job.cancel.is_cancelled() {
        return Err(LlmError::Cancelled);
    }
    let before = truncate_to_last_chars(job.before.trim_end_matches('\n'), PROMPT_MAX_CHARS);
    if before.trim().is_empty() {
        return Ok(None);
    }
    // Token healing setup: drop the in-progress fragment from the prompt so
    // decoding starts at a word boundary, then constrain the first token.
    let (prompt_text, fragment) = strip_fragment(&before);
    let mut prompt_tokens = model
        .str_to_token(&prompt_text, AddBos::Always)
        .map_err(|e| LlmError::Inference(e.to_string()))?;
    if prompt_tokens.len() > PROMPT_MAX_TOKENS {
        prompt_tokens = prompt_tokens[prompt_tokens.len() - PROMPT_MAX_TOKENS..].to_vec();
    }
    let bos = model.token_bos();
    // BOS-first, unconditionally: the same prompt must tokenize identically
    // on every request or the KV cache cannot be reused.
    if prompt_tokens.first() != Some(&bos) {
        prompt_tokens.insert(0, bos);
        if prompt_tokens.len() > PROMPT_MAX_TOKENS {
            let keep = prompt_tokens[prompt_tokens.len() - PROMPT_MAX_TOKENS + 1..].to_vec();
            prompt_tokens = std::iter::once(bos).chain(keep).collect();
        }
    }
    // Room for generation inside the context window.
    let room = config.n_ctx as usize;
    let want = job.max_tokens.max(1);
    if prompt_tokens.len() + want + 1 > room {
        let keep = room.saturating_sub(want + 1).max(1);
        prompt_tokens = prompt_tokens[prompt_tokens.len().saturating_sub(keep)..].to_vec();
    }

    // KV-cache alignment with an always-fresh prompt-end distribution.
    //
    // This llama.cpp version requires consecutive positions (each batch must
    // start at last_cached + 1), so an already-cached position can never be
    // re-decoded. `rewind_to = min(common, P - 1)` unifies every case: the
    // last prompt token is always decoded as a fresh position, ending with
    // fresh logits for sampling. Repeats re-decode 1 token, appends decode
    // only the suffix, divergent prompts recompute from the fork.
    let common = common_prefix_len(cached, &prompt_tokens);
    let rewind_to = common.min(prompt_tokens.len().saturating_sub(1));
    if rewind_to < cached.len() {
        ctx.kv_cache_seq_rm(SEQ_ID, Some(rewind_to as u32), None)
            .map_err(|e| LlmError::Inference(e.to_string()))?;
        cached.truncate(rewind_to);
    }
    let fresh = &prompt_tokens[rewind_to..];
    {
        // NOTE: `fresh` is never empty (`rewind_to <= P - 1`, `P >= 1` via BOS).
        let mut batch = LlamaBatch::new(fresh.len(), 1);
        for (i, token) in fresh.iter().enumerate() {
            let pos = (rewind_to + i) as i32;
            let last = i + 1 == fresh.len();
            batch
                .add(*token, pos, &[SEQ_ID], last)
                .map_err(|e| LlmError::Inference(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Inference(format!("prompt decode: {e}")))?;
        cached.extend_from_slice(fresh);
    }
    if job.cancel.is_cancelled() {
        return Err(LlmError::Cancelled);
    }

    // Greedy decoding: deterministic and reproducible (evals compare
    // like-for-like). Temperature/penalty variants were trialled and showed
    // no clear win on open prose (see ADR 0006); revisit with data.
    // Address-form bans ride along as -inf logit biases.
    let mut sampler = match banned_token_ids(model, job.address) {
        biases if biases.is_empty() => LlamaSampler::greedy(),
        biases => LlamaSampler::chain_simple([
            LlamaSampler::logit_bias(model.n_vocab(), &biases),
            LlamaSampler::greedy(),
        ]),
    };
    let mut out_bytes: Vec<u8> = Vec::new();
    let mut emitted = 0usize;
    let mut text = String::new();
    let mut logprob_sum = 0.0f64;
    let mut generated = 0usize;
    let mut first_token = true;
    let mut ttft = Duration::ZERO;

    for _ in 0..want {
        if job.cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        let array = ctx.token_data_array();
        let (token, logprob) = if first_token && !fragment.is_empty() {
            match heal_first_token(model, &array, &fragment) {
                Some(found) => found,
                None => return Ok(None),
            }
        } else {
            greedy_choice(&array)
        };
        if first_token {
            ttft = start.elapsed();
            first_token = false;
        }
        if model.is_eog_token(token) {
            break;
        }
        sampler.accept(token);
        logprob_sum += f64::from(logprob);
        generated += 1;

        let piece = model
            .token_to_piece_bytes(token, PIECE_BUF, false, None)
            .map_err(|e| LlmError::Inference(e.to_string()))?;
        out_bytes.extend_from_slice(&piece);
        text.push_str(&emit_complete(&out_bytes, &mut emitted));
        if text.contains('\n') {
            text = text.split('\n').next().unwrap_or("").to_string();
            break;
        }
        if is_sentence_end(&text) {
            break;
        }
        if job.stop == StopMode::Clause && is_clause_end(&text) {
            break;
        }

        // Feed the token back for the next step.
        let pos = cached.len() as i32;
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(token, pos, &[SEQ_ID], true)
            .map_err(|e| LlmError::Inference(e.to_string()))?;
        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Inference(format!("gen decode: {e}")))?;
        cached.push(token);
    }

    if generated == 0 {
        return Ok(None);
    }
    // Strip the healed fragment: the first piece starts with it (after
    // one optional leading blank), so the suggestion continues the cursor.
    let mut completion = text;
    if !fragment.is_empty() {
        let no_blank = completion.strip_prefix(' ').unwrap_or(&completion);
        if let Some(stripped) = no_blank.strip_prefix(fragment.as_str()) {
            completion = stripped.to_string();
        } else {
            return Ok(None);
        }
    }
    if completion.trim().is_empty() {
        return Ok(None);
    }
    // Backstop for multi-token banned forms the logit bias cannot cover:
    // never show an address violation, at the cost of silence.
    if predict_core::violates(&completion, job.address) {
        return Ok(None);
    }
    let confidence = (logprob_sum / generated as f64) as f32;
    if confidence < job.threshold {
        return Ok(None);
    }
    Ok(Some(SentenceOutput {
        text: completion,
        confidence,
        time_to_first_token: ttft,
        tokens_generated: generated,
    }))
}

/// Greedy choice with its logprob: argmax logit, log-softmax normalized.
/// Greedy choice with its logprob: argmax logit, log-softmax normalized.
fn greedy_choice(array: &llama_cpp_2::token::data_array::LlamaTokenDataArray) -> (LlamaToken, f32) {
    let mut best_id = LlamaToken(0);
    let mut best_logit = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;
    for datum in array.data.iter() {
        let logit = datum.logit();
        if logit > max_logit {
            max_logit = logit;
        }
        if logit > best_logit {
            best_logit = logit;
            best_id = datum.id();
        }
    }
    (best_id, best_logit - log_sum_exp(array, max_logit))
}

fn log_sum_exp(
    array: &llama_cpp_2::token::data_array::LlamaTokenDataArray,
    max_logit: f32,
) -> f32 {
    if !max_logit.is_finite() {
        return max_logit;
    }
    let sum: f64 = array
        .data
        .iter()
        .map(|datum| f64::from((datum.logit() - max_logit).exp()))
        .sum();
    max_logit + sum.ln() as f32
}

/// Token healing: among the top-K logits, pick the best token whose piece
/// starts with the in-progress fragment. Returns the token and its logprob.
fn heal_first_token(
    model: &LlamaModel,
    array: &llama_cpp_2::token::data_array::LlamaTokenDataArray,
    fragment: &str,
) -> Option<(LlamaToken, f32)> {
    let mut by_logit: Vec<(LlamaToken, f32)> =
        array.data.iter().map(|datum| (datum.id(), datum.logit())).collect();
    by_logit.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max_logit = by_logit.first().map_or(f32::NEG_INFINITY, |(_, logit)| *logit);
    let lse = log_sum_exp(array, max_logit);
    for (id, logit) in by_logit.into_iter().take(HEALING_TOP_K) {
        let piece = model.token_to_piece_bytes(id, PIECE_BUF, false, None).ok()?;
        if piece_continues_word(&piece, fragment) {
            return Some((id, logit - lse));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::{Backend, CancelToken, LlmConfig, SentenceRequest, StopMode};
    use super::{common_prefix_len, emit_complete, is_clause_end, is_sentence_end, truncate_to_last_chars};
    use llama_cpp_2::token::LlamaToken;

    #[test]
    fn common_prefix_counts_matching_head() {
        let a = vec![LlamaToken(1), LlamaToken(2), LlamaToken(3)];
        let b = vec![LlamaToken(1), LlamaToken(2), LlamaToken(9)];
        assert_eq!(common_prefix_len(&a, &b), 2);
        assert_eq!(common_prefix_len(&a, &a), 3);
        assert_eq!(common_prefix_len(&a, &[]), 0);
    }

    #[test]
    fn truncate_keeps_tail_on_char_boundary() {
        assert_eq!(truncate_to_last_chars("hello", 10), "hello");
        assert_eq!(truncate_to_last_chars("hello", 3), "llo");
        assert_eq!(truncate_to_last_chars("grüße", 2), "ße");
    }

    #[test]
    fn emit_complete_waits_for_split_multibyte_chars() {
        // "é" = 0xC3 0xA9 split across tokens.
        let mut emitted = 0;
        assert_eq!(emit_complete(b"a\xc3", &mut emitted), "a");
        assert_eq!(emitted, 1);
        assert_eq!(emit_complete(b"a\xc3\xa9", &mut emitted), "é");
        assert_eq!(emitted, 3);
    }

    #[test]
    fn emit_complete_skips_invalid_bytes() {
        let mut emitted = 0;
        // Invalid byte 0xFF is skipped; "b" arrives on the next call.
        assert_eq!(emit_complete(b"a\xffb", &mut emitted), "a");
        assert_eq!(emit_complete(b"a\xffb", &mut emitted), "b");
    }

    #[test]
    fn sentence_end_detects_terminal_punctuation() {
        assert!(is_sentence_end("and then."));
        assert!(is_sentence_end("really?"));
        assert!(!is_sentence_end("half,"));
        assert!(!is_sentence_end(""));
    }

    #[test]
    fn clause_end_detects_phrase_boundaries() {
        assert!(is_clause_end("salt,"));
        assert!(is_clause_end("this;"));
        assert!(is_clause_end("note:"));
        assert!(is_clause_end("done."));
        assert!(!is_clause_end("plain words"));
    }

    /// Opt-in style-guarantee test against a real GGUF (set
    /// PREDICT_MODEL_PATH). Passes vacuously without a model; with one it
    /// asserts the M5 done-criterion shape: whatever comes back under a Sie
    /// ban contains no du-family token (rejection shows as `None`).
    #[test]
    fn real_model_never_violates_ban() {
        let path = std::env::var("PREDICT_MODEL_PATH").unwrap_or_default();
        if path.is_empty() {
            return;
        }
        let config = LlmConfig {
            enabled: true,
            model_path: path,
            max_tokens: 16,
            confidence_threshold: -99.0,
            ..Default::default()
        };
        let backend = super::LlamaBackend::load(&config).unwrap();
        for before in [
            "Kannst du mir sagen, ob ",
            "Wenn du morgen Zeit hast, ",
            "Bitte prüfe, ob du ",
        ] {
            let req = SentenceRequest {
                before: before.to_string(),
                max_tokens: 16,
                confidence_threshold: -99.0,
                stop: StopMode::Sentence,
                address: predict_core::AddressForm::Sie,
            };
            let out = backend.complete_sentence(&req, &CancelToken::new()).unwrap();
            assert!(
                out.is_none_or(|o| !predict_core::violates(&o.text, predict_core::AddressForm::Sie)),
                "ban violated for {before:?}"
            );
        }
    }

    /// Opt-in smoke test against a real GGUF (set PREDICT_MODEL_PATH).
    /// Skipped silently otherwise so plain `cargo test` stays fast.
    #[test]
    fn real_model_continues_text() {
        let path = std::env::var("PREDICT_MODEL_PATH").unwrap_or_default();
        if path.is_empty() {
            return;
        }
        let config = LlmConfig {
            enabled: true,
            model_path: path,
            max_tokens: 16,
            confidence_threshold: -99.0,
            ..Default::default()
        };
        let backend = super::LlamaBackend::load(&config).unwrap();
        let req = SentenceRequest {
            before: "the quick brown fox jumps over the lazy ".to_string(),
            max_tokens: 16,
            confidence_threshold: -99.0,
        stop: StopMode::Sentence,
        address: predict_core::AddressForm::None,
        };
        let out = backend
            .complete_sentence(&req, &CancelToken::new())
            .unwrap()
            .expect("model should continue the phrase");
        assert!(!out.text.trim().is_empty());
        assert!(out.confidence.is_finite());
        assert!(out.tokens_generated >= 1);
    }
}
