//! predictd: per-user prediction daemon.
//!
//! Listens on a Unix socket ([`predict_proto::socket_path`]). Word requests
//! (`Suggest`) are answered synchronously from the fast tier. Sentence
//! requests (`SuggestSentence`, M3) run on a worker thread against the
//! optional LLM backend: a newer generation cancels older work, and the
//! worker answers only while its generation is still current — otherwise it
//! stays silent. One `std::thread` per connection plus one per slow request;
//! no async runtime (see ADR 0003/0004).

use anyhow::{Context as _, Result};
use predict_core::{AddressForm, Context, LanguageSpec, LengthMode, Predictor as _, ResolvedStyle, StyleSpec};
use predict_llm::{
    Backend, CancelToken, LlmConfig, SentenceRequest, StopMode, build_grounded_prompt,
    llama::LlamaBackend,
};
use predict_ngram::{LangBlended, NgramModel, PersonalBundle};
use predict_proto::{
    CancelMsg, ClientMsg, CommitText, ContextUpdate, DaemonMsg, LearningState, ProtoCandidate,
    SentenceSuggestion, SetLearning, SuggestRequest, Suggestion, read_client_msg, socket_path,
    write_daemon_msg,
};
use predict_store::Store;
use std::io::ErrorKind;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, Ordering},
};

/// Base training texts: public-domain books, one per language (see
/// `corpora/SOURCES.md`). Split models so English typing never offers
/// German words and vice versa (interim until M5 style modes).
const BASE_EN: &str = include_str!("../../../corpora/base_en.txt");
const BASE_DE: &str = include_str!("../../../corpora/base_de.txt");

/// Per-language base models, shared across connections.
struct BaseModels {
    en: NgramModel,
    de: NgramModel,
}

impl BaseModels {
    fn train(en_text: &str, de_text: &str) -> anyhow::Result<Self> {
        let start = std::time::Instant::now();
        let en =
            NgramModel::from_text(en_text).map_err(|e| anyhow::anyhow!("train en model: {e}"))?;
        let de =
            NgramModel::from_text(de_text).map_err(|e| anyhow::anyhow!("train de model: {e}"))?;
        eprintln!(
            "predictd: base models trained in {:.1}s (en: {} tokens/{} words, de: {} tokens/{} words)",
            start.elapsed().as_secs_f64(),
            en.total_tokens(),
            en.vocab_size(),
            de.total_tokens(),
            de.vocab_size(),
        );
        Ok(Self { en, de })
    }
}

fn main() -> Result<()> {
    let models = Arc::new(BaseModels::train(BASE_EN, BASE_DE)?);
    let slow = load_llm_from(&config_path());
    let personal = load_personal_from(&config_path());
    let styles = Arc::new(load_styles_from(&config_path()));

    let path = socket_path();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create socket dir {}", parent.display()))?;
        }
    }
    // Remove a stale socket left by a previous run; a live daemon would have
    // an exclusive bind, so unlink-then-bind is the simple M2 choice.
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("remove stale socket {}", path.display()))?;
    }
    let listener =
        UnixListener::bind(&path).with_context(|| format!("bind {}", path.display()))?;
    eprintln!("predictd: listening on {}", path.display());

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let models = Arc::clone(&models);
                let slow = slow.clone();
                let personal = personal.clone();
                let styles = Arc::clone(&styles);
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, models, slow, personal, styles) {
                        eprintln!("predictd: connection error: {e:#}");
                    }
                });
            }
            Err(e) => eprintln!("predictd: accept error: {e:#}"),
        }
    }
    Ok(())
}

/// Slow-tier configuration shared across connections.
#[derive(Clone)]
struct SlowCfg {
    backend: Arc<dyn Backend>,
    max_tokens: usize,
    threshold: f32,
}

/// Style registry: built-in specs plus `[styles.*]` config, per-app
/// defaults, and a global default.
///
/// Resolution order (plan M5): explicit style id from the frontend >
/// sticky inferred address > per-app default > global default. Inference
/// only ever yields an address form, so it overrides just that field.
struct StyleRegistry {
    specs: std::collections::HashMap<String, StyleSpec>,
    per_app: std::collections::HashMap<String, String>,
    global_default: String,
}

impl StyleRegistry {
    fn new() -> Self {
        let mut specs = std::collections::HashMap::new();
        for id in predict_core::BUILTIN_STYLES {
            specs.insert(id.to_string(), predict_core::builtin_spec(id));
        }
        Self {
            specs,
            per_app: std::collections::HashMap::new(),
            global_default: "default".to_string(),
        }
    }

    /// Load `[style]` / `[styles.*]` / `[apps.*]` sections (bad entries are
    /// skipped with a warning, never fatal).
    fn load_toml(&mut self, text: &str) {
        let file: StyleFile = match toml::from_str(text) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("predictd: bad [style] config: {e} (styles fall back to built-ins)");
                return;
            }
        };
        if let Some(default) = file.style.default {
            if self.specs.contains_key(&default) || !default.is_empty() {
                self.global_default = default;
            }
        }
        for (id, section) in file.styles {
            match parse_spec(&id, &section) {
                Ok(spec) => {
                    self.specs.insert(id, spec);
                }
                Err(e) => eprintln!("predictd: skipping style: {e}"),
            }
        }
        for (app, section) in file.apps {
            if let Some(style) = section.style {
                self.per_app.insert(app, style);
            }
        }
    }

    /// True for registered style ids (built-in or configured).
    fn knows(&self, id: &str) -> bool {
        self.specs.contains_key(id)
    }

    /// Resolve the active style for one request.
    fn resolve(
        &self,
        explicit_id: &str,
        sticky: Option<AddressForm>,
        app_id: &str,
    ) -> ResolvedStyle {
        // "" and "default" both mean unspecified (the frontend's resting
        // state): fall through to inference and defaults. Any other known
        // id is an explicit user choice and wins outright.
        let explicit = match explicit_id {
            "" | "default" => None,
            id => self.specs.get(id),
        };
        if let Some(spec) = explicit {
            return ResolvedStyle::from_spec(spec);
        }
        let base_id = self
            .per_app
            .get(app_id)
            .cloned()
            .unwrap_or_else(|| self.global_default.clone());
        let mut resolved = self
            .specs
            .get(&base_id)
            .map(ResolvedStyle::from_spec)
            .unwrap_or_else(ResolvedStyle::default_style);
        if let Some(address) = sticky {
            if address != AddressForm::None {
                resolved.address_form = address;
            }
        }
        resolved
    }
}

/// Whole-file style config (`[style]` + `[styles.*]` + `[apps.*]`).
#[derive(serde::Deserialize, Default)]
struct StyleFile {
    #[serde(default)]
    style: StyleTop,
    #[serde(default)]
    styles: std::collections::HashMap<String, StyleSection>,
    #[serde(default)]
    apps: std::collections::HashMap<String, StyleApp>,
}

#[derive(serde::Deserialize, Default)]
struct StyleTop {
    default: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct StyleSection {
    language: Option<String>,
    address_form: Option<String>,
    length: Option<String>,
}

#[derive(serde::Deserialize, Default)]
struct StyleApp {
    style: Option<String>,
}

/// Parse one `[styles.<id>]` section (all fields optional, spec defaults).
fn parse_spec(id: &str, section: &StyleSection) -> Result<StyleSpec, String> {
    Ok(StyleSpec {
        id: id.to_string(),
        language: section
            .language
            .as_deref()
            .map(parse_language)
            .transpose()?
            .unwrap_or(LanguageSpec::Auto),
        address_form: section
            .address_form
            .as_deref()
            .map(parse_address)
            .transpose()?
            .unwrap_or(AddressForm::None),
        length: section
            .length
            .as_deref()
            .map(parse_length)
            .transpose()?
            .unwrap_or(LengthMode::Sentence),
    })
}

fn parse_language(s: &str) -> Result<LanguageSpec, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(LanguageSpec::Auto),
        "en" | "english" => Ok(LanguageSpec::En),
        "de" | "german" | "deutsch" => Ok(LanguageSpec::De),
        other => Err(format!("unknown language '{other}'")),
    }
}

fn parse_address(s: &str) -> Result<AddressForm, String> {
    match s.to_lowercase().as_str() {
        "du" => Ok(AddressForm::Du),
        "sie" => Ok(AddressForm::Sie),
        "none" => Ok(AddressForm::None),
        other => Err(format!("unknown address_form '{other}'")),
    }
}

fn parse_length(s: &str) -> Result<LengthMode, String> {
    match s.to_lowercase().as_str() {
        "word" => Ok(LengthMode::Word),
        "phrase" => Ok(LengthMode::Phrase),
        "sentence" => Ok(LengthMode::Sentence),
        other => Err(format!("unknown length '{other}'")),
    }
}

/// Load the style registry from the on-disk config (missing file means
/// built-ins only; bad sections are skipped with warnings inside).
fn load_styles_from(path: &std::path::Path) -> StyleRegistry {
    let mut registry = StyleRegistry::new();
    match std::fs::read_to_string(path) {
        Ok(text) => registry.load_toml(&text),
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => eprintln!("predictd: cannot read {}: {e:#}", path.display()),
    }
    registry
}

/// Path of the daemon config file.
fn config_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("predict/predictd.toml");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".config/predict/predictd.toml")
}

/// Parse the LLM config from a file and load the backend. Missing files,
/// disabled sections, and anything that fails to load resolve to `None`
/// (with a warning): the daemon always keeps serving the word tier.
fn load_llm_from(path: &std::path::Path) -> Option<SlowCfg> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            eprintln!("predictd: no {} (llm disabled)", path.display());
            return None;
        }
        Err(e) => {
            eprintln!(
                "predictd: cannot read {}: {e:#} (llm disabled)",
                path.display()
            );
            return None;
        }
    };
    let cfg = match LlmConfig::from_toml_str(&text) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!(
                "predictd: bad config {}: {e} (llm disabled)",
                path.display()
            );
            return None;
        }
    };
    if !cfg.enabled {
        eprintln!("predictd: llm disabled by config");
        return None;
    }
    maybe_load_llm(&cfg)
}

/// Load the backend for an enabled config; `None` with a warning on failure.
fn maybe_load_llm(cfg: &LlmConfig) -> Option<SlowCfg> {
    match LlamaBackend::load(cfg) {
        Ok(backend) => {
            eprintln!("predictd: llm enabled ({})", cfg.model_path);
            Some(SlowCfg {
                backend: Arc::new(backend),
                max_tokens: cfg.max_tokens,
                threshold: cfg.confidence_threshold,
            })
        }
        Err(e) => {
            eprintln!("predictd: llm disabled: {e}");
            None
        }
    }
}

/// Personal-memory configuration (`[personal]` section of `predictd.toml`).
#[derive(Debug, Clone)]
struct PersonalCfg {
    enabled: bool,
    lambda: f32,
    db_path: PathBuf,
}

impl PersonalCfg {
    fn from_toml_str(text: &str, default_db: PathBuf) -> Result<Self, String> {
        #[derive(serde::Deserialize, Default)]
        struct File {
            #[serde(default)]
            personal: Section,
        }
        #[derive(serde::Deserialize, Default)]
        struct Section {
            enabled: Option<bool>,
            lambda: Option<f32>,
            db_path: Option<String>,
        }
        let file: File =
            toml::from_str(text).map_err(|e| format!("bad [personal] config: {e}"))?;
        Ok(Self {
            enabled: file.personal.enabled.unwrap_or(true),
            lambda: file.personal.lambda.unwrap_or(0.7),
            db_path: file
                .personal
                .db_path
                .map(PathBuf::from)
                .unwrap_or(default_db),
        })
    }
}

/// Personal memory shared across connections: the store (behind a mutex —
/// `rusqlite::Connection` is `!Sync`), per-style count bundles, the blend
/// weight, and the pause-learning switch.
#[derive(Clone)]
struct Personal {
    store: Arc<Mutex<Store>>,
    bundles: Arc<RwLock<std::collections::HashMap<String, PersonalBundle>>>,
    lambda: f32,
    paused: Arc<AtomicBool>,
}

impl Personal {
    fn open(cfg: &PersonalCfg) -> Result<Self> {
        let store = Store::open(&cfg.db_path).with_context(|| {
            format!("open personal store {}", cfg.db_path.display())
        })?;
        let store = Arc::new(Mutex::new(store));
        let bundles = Arc::new(RwLock::new(reload_bundles(&store)?));
        Ok(Self {
            store,
            bundles,
            lambda: if cfg.lambda.is_nan() {
                0.7
            } else {
                cfg.lambda.clamp(0.0, 1.0)
            },
            paused: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Bundle for one style id (empty when the style never learned —
    /// blending then behaves base-only).
    fn bundle_for(&self, style_id: &str) -> PersonalBundle {
        self.bundles
            .read()
            .map(|bundles| bundles.get(style_id).cloned().unwrap_or_default())
            .unwrap_or_default()
    }
}

/// Reload all per-style bundles from SQLite (after commits and forget-all).
fn reload_bundles(
    store: &Arc<Mutex<Store>>,
) -> Result<std::collections::HashMap<String, PersonalBundle>> {
    let store = store
        .lock()
        .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
    store.personal_bundles().context("load bundles")
}

/// Open personal memory from the on-disk config (warns and disables on any
/// problem; the word tier always works, and learning is on by default).
fn load_personal_from(path: &std::path::Path) -> Option<Personal> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => String::new(),
        Err(e) => {
            eprintln!(
                "predictd: cannot read {}: {e:#} (personal memory disabled)",
                path.display()
            );
            return None;
        }
    };
    let cfg = match PersonalCfg::from_toml_str(&text, predict_store::default_db_path()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("predictd: {e} (personal memory disabled)",);
            return None;
        }
    };
    if !cfg.enabled {
        eprintln!("predictd: personal memory disabled by config");
        return None;
    }
    match Personal::open(&cfg) {
        Ok(personal) => {
            eprintln!(
                "predictd: personal memory enabled ({})",
                cfg.db_path.display()
            );
            Some(personal)
        }
        Err(e) => {
            eprintln!("predictd: personal memory disabled: {e:#}");
            None
        }
    }
}

/// Slow-tier work in flight for one connection.
#[derive(Debug)]
struct SlowJob {
    generation: u64,
    cancel: CancelToken,
}

/// Per-connection state, shared with slow-tier worker threads.
#[derive(Debug, Default)]
struct SharedConn {
    /// Newest generation seen; anything older is stale.
    newest_seen: u64,
    ctx: Option<ContextUpdate>,
    slow: Option<SlowJob>,
    retrieval: RetrievalCache,
    /// Inferred address form, sticky once set (plan M5).
    inferred: Option<AddressForm>,
}

impl SharedConn {
    /// Record a generation; false when stale. A fresh generation supersedes
    /// older slow work (a same-generation duplicate does not).
    fn observe(&mut self, generation: u64) -> bool {
        if generation < self.newest_seen {
            return false;
        }
        if let Some(job) = &self.slow {
            if job.generation < generation {
                job.cancel.cancel();
            }
        }
        self.newest_seen = generation;
        true
    }

    /// Record a cancellation: slow jobs at or below it are aborted, and the
    /// generation is marked superseded.
    fn cancel_through(&mut self, generation: u64) {
        if generation > self.newest_seen {
            self.newest_seen = generation;
        }
        if let Some(job) = &self.slow {
            if job.generation <= generation {
                job.cancel.cancel();
            }
        }
    }
}

/// Resolve the active style for one request: explicit id > sticky
/// inferred address > per-app default > global default. The sticky
/// inference is set once from confident detections and then kept.
fn resolve_for(
    registry: &StyleRegistry,
    guard: &mut SharedConn,
    ctx: &ContextUpdate,
) -> ResolvedStyle {
    if guard.inferred.is_none() {
        guard.inferred = predict_core::detect_address(&ctx.before);
    }
    registry.resolve(&ctx.style_id, guard.inferred, &ctx.app_id)
}

/// Work decided while holding the connection lock; I/O happens after.
enum Action {
    ReplyWord(Suggestion),
    SpawnSlow {
        generation: u64,
        cancel: CancelToken,
        before: String,
        style: ResolvedStyle,
        snippets: Vec<String>,
        stop: StopMode,
        cfg: SlowCfg,
    },
    /// Settled text from the frontend (stored unless paused).
    Commit { text: String, style_id: String },
    /// Forget-all request.
    Forget,
    /// Pause-learning toggle.
    SetLearning(bool),
    None,
}

/// Build the daemon reply for one generation (fast tier, blended with the
/// personal bundle; an empty bundle behaves exactly like the base models).
/// Address bans apply at decode selection, and the reply carries the
/// resolved style id so the UI can show it.
fn suggest_for(
    models: &BaseModels,
    personal: &PersonalBundle,
    lambda: f32,
    style: &ResolvedStyle,
    ctx: &ContextUpdate,
    generation: u64,
) -> Suggestion {
    let core_ctx = Context::new(
        ctx.app_id.clone(),
        ctx.before.clone(),
        ctx.after.clone(),
        ctx.sensitive,
        ResolvedStyle::new(ctx.style_id.clone()),
    );
    let blended = LangBlended::new(&models.en, &models.de, personal, lambda);
    let (banned, case_sensitive) = predict_core::banned_words(style.address_form);
    let candidates = blended
        .complete_word(&core_ctx)
        .into_iter()
        .filter(|c| !banned_word_in(&c.text, banned, case_sensitive))
        .map(|c| ProtoCandidate {
            text: c.text,
            score: c.score,
        })
        .collect();
    Suggestion {
        generation,
        candidates,
        style_id: style.style_id.clone(),
    }
}

/// Single-word ban check (du-family forms are listed explicitly, so exact
/// matching suffices; the empty list never matches).
fn banned_word_in(word: &str, banned: &[&str], case_sensitive: bool) -> bool {
    if banned.is_empty() {
        return false;
    }
    if case_sensitive {
        banned.contains(&word)
    } else {
        banned.contains(&word.to_lowercase().as_str())
    }
}

fn default_ctx() -> ContextUpdate {
    ContextUpdate {
        app_id: String::new(),
        before: String::new(),
        after: String::new(),
        sensitive: false,
        style_id: "default".to_string(),
    }
}

/// Send one message, serializing writers across connection + worker threads
/// so frames never interleave.
fn send_msg(writer: &Arc<Mutex<UnixStream>>, msg: &DaemonMsg) -> Result<()> {
    let mut stream = writer
        .lock()
        .map_err(|_| anyhow::anyhow!("writer lock poisoned"))?;
    write_daemon_msg(&mut *stream, msg).context("write daemon message")
}

/// Retrieval state per connection: cached snippets plus the context they
/// were fetched for. Refreshed only when the context stops extending the
/// previous one or crosses a sentence boundary (see M4 plan).
#[derive(Debug, Default)]
struct RetrievalCache {
    last_before: Option<String>,
    snippets: Vec<String>,
}

/// Count sentence terminators (boundary-crossing detector for retrieval).
fn term_count(text: &str) -> usize {
    text.chars()
        .filter(|c| matches!(c, '.' | '!' | '?' | '\n'))
        .count()
}

/// Fetch up to 3 grounding snippets for `before`'s current sentence
/// fragment (empty without a personal store or for a blank fragment).
fn retrieve(personal: &Option<Personal>, before: &str, style: Option<&str>) -> Vec<String> {
    let Some(personal) = personal else {
        return Vec::new();
    };
    let query = predict_core::sentence_fragment(before);
    if query.trim().is_empty() {
        return Vec::new();
    }
    let Ok(store) = personal.store.lock() else {
        return Vec::new();
    };
    store.search(&query, 3, style).unwrap_or_default()
}

/// Current learning state for replies (disabled store reports off/empty).
fn learning_state(personal: &Option<Personal>) -> LearningState {
    match personal {
        Some(personal) => {
            let documents = personal
                .store
                .lock()
                .map(|store| store.doc_count().unwrap_or(0))
                .unwrap_or(0);
            LearningState {
                enabled: !personal.paused.load(Ordering::SeqCst),
                documents,
            }
        }
        None => LearningState {
            enabled: false,
            documents: 0,
        },
    }
}

/// Serve one client until it disconnects or the stream breaks.
///
/// Lock hygiene (deadlock audit): the connection mutex is only ever held
/// while briefly taking the store mutex (retrieval) or cloning the count
/// bundle; the store mutex is only ever held while taking the counts lock
/// (commit/forget reload). No path reverses these orders, and worker
/// threads take the connection mutex alone.
fn handle_connection(
    stream: UnixStream,
    models: Arc<BaseModels>,
    slow: Option<SlowCfg>,
    personal: Option<Personal>,
    styles: Arc<StyleRegistry>,
) -> Result<()> {
    let writer = Arc::new(Mutex::new(
        stream.try_clone().context("clone stream")?,
    ));
    let mut reader = stream;
    let shared = Arc::new(Mutex::new(SharedConn::default()));
    loop {
        let msg = match read_client_msg(&mut reader) {
            Ok(msg) => msg,
            Err(e) => {
                if let predict_proto::ProtoError::Io(io) = &e {
                    if io.kind() == ErrorKind::UnexpectedEof {
                        return Ok(());
                    }
                }
                return Err(e).context("read client message");
            }
        };
        let action = {
            let mut guard = shared
                .lock()
                .map_err(|_| anyhow::anyhow!("connection lock poisoned"))?;
            match msg {
                ClientMsg::ContextUpdate(ctx) => {
                    guard.ctx = Some(ctx);
                    Action::None
                }
                ClientMsg::Suggest(SuggestRequest { generation }) => {
                    if !guard.observe(generation) {
                        Action::None
                    } else {
                        let ctx = guard.ctx.clone().unwrap_or_else(default_ctx);
                        let style = resolve_for(&styles, &mut guard, &ctx);
                        // Lock order: connection -> counts (never reversed).
                        let (bundle, lambda) = match &personal {
                            Some(personal) => (
                                personal.bundle_for(&style.style_id),
                                personal.lambda,
                            ),
                            None => (PersonalBundle::default(), 1.0),
                        };
                        Action::ReplyWord(suggest_for(
                            &models, &bundle, lambda, &style, &ctx, generation,
                        ))
                    }
                }
                ClientMsg::SuggestSentence(SuggestRequest { generation }) => {
                    let ctx = guard.ctx.clone().unwrap_or_else(default_ctx);
                    if ctx.sensitive {
                        // Never suggest (or retrieve) for sensitive fields.
                        Action::None
                    } else if !guard.observe(generation) {
                        Action::None
                    } else {
                        let style = resolve_for(&styles, &mut guard, &ctx);
                        if style.length == LengthMode::Word {
                            // Word-only style: no sentence tier.
                            Action::None
                        } else {
                            let duplicate = guard
                                .slow
                                .as_ref()
                                .is_some_and(|job| job.generation == generation);
                            match (duplicate, slow.clone()) {
                                (false, Some(cfg)) => {
                                    let cancel = CancelToken::new();
                                    guard.slow = Some(SlowJob {
                                        generation,
                                        cancel: cancel.clone(),
                                    });
                                    // Retrieval refresh: sentence boundaries
                                    // only (lock order: connection -> store),
                                    // filtered to the active style.
                                    let mut snippets =
                                        guard.retrieval.snippets.clone();
                                    let refresh = match &guard.retrieval.last_before {
                                        None => true,
                                        Some(prev) => {
                                            !ctx.before.starts_with(prev.as_str())
                                                || term_count(&ctx.before) != term_count(prev)
                                        }
                                    };
                                    if refresh {
                                        snippets = retrieve(
                                            &personal,
                                            &ctx.before,
                                            Some(style.style_id.as_str()),
                                        );
                                        guard.retrieval.last_before =
                                            Some(ctx.before.clone());
                                        guard.retrieval.snippets = snippets.clone();
                                    }
                                    let stop = match style.length {
                                        LengthMode::Phrase => StopMode::Clause,
                                        _ => StopMode::Sentence,
                                    };
                                    Action::SpawnSlow {
                                        generation,
                                        cancel,
                                        before: ctx.before,
                                        style,
                                        snippets,
                                        stop,
                                        cfg,
                                    }
                                }
                                // Duplicate request, or slow tier disabled:
                                // silence, like a confidence gate.
                                _ => Action::None,
                            }
                        }
                    }
                }
                ClientMsg::CommitText(CommitText {
                    text,
                    style_id,
                    sensitive,
                }) => {
                    if sensitive {
                        // Never learn from sensitive fields.
                        Action::None
                    } else {
                        // Tag with the active style: the requested id when
                        // known, else the default.
                        let tag = if styles.knows(&style_id) {
                            style_id
                        } else {
                            "default".to_string()
                        };
                        Action::Commit {
                            text,
                            style_id: tag,
                        }
                    }
                }
                ClientMsg::ForgetAll => Action::Forget,
                ClientMsg::SetLearning(SetLearning { enabled }) => {
                    Action::SetLearning(enabled)
                }
                ClientMsg::Cancel(CancelMsg { generation }) => {
                    guard.cancel_through(generation);
                    Action::None
                }
            }
        };
        match action {
            Action::None => {}
            Action::ReplyWord(reply) => {
                send_msg(&writer, &DaemonMsg::Suggestion(reply))?;
            }
            Action::Commit { text, style_id } => {
                // Personal ops never break the word tier: failures are
                // logged, the connection stays up. Always ack so frontends
                // can show live store state.
                if let Some(personal) = &personal {
                    if !personal.paused.load(Ordering::SeqCst) {
                        let stored = personal
                            .store
                            .lock()
                            .map(|store| store.commit(&text, &style_id))
                            .map_err(|_| anyhow::anyhow!("store lock poisoned"));
                        match stored {
                            Ok(Ok(true)) => match reload_bundles(&personal.store) {
                                Ok(bundles) => {
                                    if let Ok(mut cached) = personal.bundles.write() {
                                        *cached = bundles;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("predictd: counts reload failed: {e:#}")
                                }
                            },
                            Ok(Ok(false)) => {}
                            Ok(Err(e)) => eprintln!("predictd: commit failed: {e:#}"),
                            Err(e) => eprintln!("predictd: {e:#}"),
                        }
                    }
                }
                let _ = send_msg(
                    &writer,
                    &DaemonMsg::LearningState(learning_state(&personal)),
                );
            }
            Action::Forget => {
                if let Some(personal) = &personal {
                    let result = (|| -> Result<()> {
                        let store = personal
                            .store
                            .lock()
                            .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
                        store.clear_all().context("forget all")?;
                        let bundles =
                            store.personal_bundles().context("reload bundles")?;
                        drop(store);
                        personal
                            .bundles
                            .write()
                            .map(|mut cached| *cached = bundles)
                            .map_err(|_| anyhow::anyhow!("counts lock poisoned"))?;
                        Ok(())
                    })();
                    if let Err(e) = result {
                        eprintln!("predictd: forget-all failed: {e:#}");
                    }
                }
                let _ = send_msg(
                    &writer,
                    &DaemonMsg::LearningState(learning_state(&personal)),
                );
            }
            Action::SetLearning(enabled) => {
                if let Some(personal) = &personal {
                    personal.paused.store(!enabled, Ordering::SeqCst);
                }
                let _ = send_msg(
                    &writer,
                    &DaemonMsg::LearningState(learning_state(&personal)),
                );
            }
            Action::SpawnSlow {
                generation,
                cancel,
                before,
                style,
                snippets,
                stop,
                cfg,
            } => {
                let writer = Arc::clone(&writer);
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let req = SentenceRequest {
                        before: build_grounded_prompt(&before, &snippets),
                        max_tokens: cfg.max_tokens,
                        confidence_threshold: cfg.threshold,
                        stop,
                        address: style.address_form,
                    };
                    // Stream partials live (same message shape, same
                    // generation): clients paint at first-token time and
                    // overwrite as the text grows. Stale generations are
                    // filtered client-side; a cancelled worker just stops.
                    let part_writer = Arc::clone(&writer);
                    let part_cancel = cancel.clone();
                    let part_style = style.style_id.clone();
                    let on_partial =
                        Box::new(move |partial: predict_llm::PartialSentence| {
                            if part_cancel.is_cancelled() {
                                return;
                            }
                            let reply = SentenceSuggestion {
                                generation,
                                text: partial.text,
                                confidence: partial.confidence,
                                style_id: part_style.clone(),
                            };
                            let _ = send_msg(&part_writer, &DaemonMsg::Sentence(reply));
                        });
                    let result =
                        cfg.backend
                            .complete_sentence_streaming(&req, &cancel, on_partial);
                    let fresh = {
                        let mut guard = match shared.lock() {
                            Ok(guard) => guard,
                            Err(_) => return,
                        };
                        if guard
                            .slow
                            .as_ref()
                            .is_some_and(|job| job.generation == generation)
                        {
                            guard.slow = None;
                        }
                        !cancel.is_cancelled() && guard.newest_seen == generation
                    };
                    if !fresh {
                        return;
                    }
                    match result {
                        Ok(Some(out)) => {
                            let reply = SentenceSuggestion {
                                generation,
                                text: out.text,
                                confidence: out.confidence,
                                style_id: style.style_id.clone(),
                            };
                            let _ = send_msg(&writer, &DaemonMsg::Sentence(reply));
                        }
                        // Gated or failed after partials were shown: retract
                        // so no stale ghost lingers. Cancelled means a newer
                        // generation owns the display — stay silent.
                        Ok(None) => {
                            let retract = SentenceSuggestion {
                                generation,
                                text: String::new(),
                                confidence: f32::NEG_INFINITY,
                                style_id: style.style_id.clone(),
                            };
                            let _ = send_msg(&writer, &DaemonMsg::Sentence(retract));
                        }
                        Err(predict_llm::LlmError::Cancelled) => {}
                        Err(_) => {
                            let retract = SentenceSuggestion {
                                generation,
                                text: String::new(),
                                confidence: f32::NEG_INFINITY,
                                style_id: style.style_id.clone(),
                            };
                            let _ = send_msg(&writer, &DaemonMsg::Sentence(retract));
                        }
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use predict_llm::{SentenceOutput, StubBackend};
    use predict_proto::{read_daemon_msg, write_client_msg};
    use std::time::{Duration, Instant};

    fn test_models() -> Arc<BaseModels> {
        Arc::new(BaseModels {
            en: NgramModel::from_text("the quick brown fox jumps hello world").unwrap(),
            de: NgramModel::from_text("der schnelle braune fuchs springt hallo welt").unwrap(),
        })
    }

    fn ctx(before: &str) -> ContextUpdate {
        ContextUpdate {
            app_id: "test".to_string(),
            before: before.to_string(),
            after: String::new(),
            sensitive: false,
            style_id: "default".to_string(),
        }
    }

    fn stub_slow() -> SlowCfg {
        SlowCfg {
            backend: Arc::new(StubBackend::fixed("brown fox", -0.2)),
            max_tokens: 32,
            threshold: -1.5,
        }
    }

    fn test_styles() -> Arc<StyleRegistry> {
        let mut registry = StyleRegistry::new();
        registry.load_toml(
            "[styles.wordy]\nlanguage = \"en\"\naddress_form = \"none\"\nlength = \"word\"\n\
             [styles.du-mail]\nlanguage = \"de\"\naddress_form = \"du\"\nlength = \"sentence\"\n\
             [apps.mail]\nstyle = \"du-mail\"\n",
        );
        Arc::new(registry)
    }

    fn plain_style() -> ResolvedStyle {
        ResolvedStyle::default_style()
    }

    #[test]
    fn suggest_for_answers_with_matching_generation() {
        let models = test_models();
        let reply = suggest_for(
            &models,
            &PersonalBundle::default(),
            0.7,
            &plain_style(),
            &ctx("the qui"),
            9,
        );
        assert_eq!(reply.generation, 9);
        assert_eq!(reply.style_id, "default");
        assert!(!reply.candidates.is_empty());
        assert_eq!(reply.candidates[0].text, "quick");
    }

    #[test]
    fn english_context_never_offers_german() {
        let models = test_models();
        let reply = suggest_for(
            &models,
            &PersonalBundle::default(),
            0.7,
            &plain_style(),
            &ctx("the qui"),
            1,
        );
        assert_eq!(reply.candidates[0].text, "quick");
        assert!(
            !reply.candidates.iter().any(|c| c.text == "braune"),
            "German leaked into English: {:?}",
            reply.candidates
        );
    }

    #[test]
    fn german_context_never_offers_english() {
        let models = test_models();
        let reply = suggest_for(
            &models,
            &PersonalBundle::default(),
            0.7,
            &plain_style(),
            &ctx("der bra"),
            1,
        );
        assert_eq!(reply.candidates[0].text, "braune");
        assert!(
            !reply.candidates.iter().any(|c| c.text == "brown"),
            "English leaked into German: {:?}",
            reply.candidates
        );
    }

    #[test]
    fn unknown_language_shows_one_language_only() {
        let models = test_models();
        // Empty context: no markers. English wins the confidence tie-break
        // here; what matters is that German never leaks in alongside.
        let reply = suggest_for(&models, &PersonalBundle::default(), 0.7, &plain_style(), &ctx(""), 1);
        assert_eq!(reply.candidates[0].text, "brown");
        for c in &reply.candidates {
            assert!(
                !["braune", "der", "fuchs", "hallo", "schnelle", "springt", "welt"]
                    .contains(&c.text.as_str()),
                "German leaked into unknown-language reply: {c:?}"
            );
        }
        // Prefix only the German model can complete: German wins outright.
        let reply = suggest_for(
            &models,
            &PersonalBundle::default(),
            0.7,
            &plain_style(),
            &ctx("hal"),
            1,
        );
        assert_eq!(reply.candidates[0].text, "hallo");
    }

    #[test]
    fn suggest_for_respects_sensitive() {
        let models = test_models();
        let sensitive = ContextUpdate {
            sensitive: true,
            ..ctx("the qui")
        };
        let reply = suggest_for(&models, &PersonalBundle::default(), 0.7, &plain_style(), &sensitive, 1);
        assert!(reply.candidates.is_empty());
    }

    #[test]
    fn style_config_parses_sections() {
        let mut registry = StyleRegistry::new();
        registry.load_toml(
            "[style]\ndefault = \"formal-de\"\n\
             [styles.formal-de]\nlanguage = \"de\"\naddress_form = \"sie\"\nlength = \"sentence\"\n\
             [styles.broken]\naddress_form = \"royal\"\n\
             [apps.gedit]\nstyle = \"formal-de\"\n",
        );
        assert_eq!(registry.global_default, "formal-de");
        let spec = registry.specs.get("formal-de").unwrap();
        assert_eq!(spec.address_form, AddressForm::Sie);
        assert_eq!(spec.language, LanguageSpec::De);
        assert!(registry.specs.get("broken").is_none());
        assert_eq!(registry.per_app.get("gedit").unwrap(), "formal-de");
        // Malformed TOML leaves built-ins intact.
        let mut plain = StyleRegistry::new();
        plain.load_toml("[styles\nbroken");
        assert_eq!(plain.global_default, "default");
    }

    #[test]
    fn resolution_order_is_explicit_sticky_app_global() {
        let styles = test_styles();
        let app = "test";
        // Explicit known id wins over everything.
        let resolved = styles.resolve("sie", Some(AddressForm::Du), app);
        assert_eq!(resolved.style_id, "sie");
        assert_eq!(resolved.address_form, AddressForm::Sie);
        // "default" is unspecified: sticky inferred address overrides base.
        let resolved = styles.resolve("default", Some(AddressForm::Du), app);
        assert_eq!(resolved.style_id, "default");
        assert_eq!(resolved.address_form, AddressForm::Du);
        let resolved = styles.resolve("", Some(AddressForm::Du), app);
        assert_eq!(resolved.style_id, "default");
        assert_eq!(resolved.address_form, AddressForm::Du);
        // No sticky: per-app default, else global.
        let resolved = styles.resolve("", None, "mail");
        assert_eq!(resolved.style_id, "du-mail");
        let resolved = styles.resolve("", None, app);
        assert_eq!(resolved.style_id, "default");
        assert_eq!(resolved.address_form, AddressForm::None);
    }

    #[test]
    fn sticky_inference_sets_once_and_holds() {
        let styles = test_styles();
        let mut guard = SharedConn::default();
        // First confident detection sticks...
        let style = resolve_for(&styles, &mut guard, &ctx("kannst du helfen"));
        assert_eq!(style.address_form, AddressForm::Du);
        // ...even when later text suggests otherwise.
        let style = resolve_for(&styles, &mut guard, &ctx("können Sie helfen"));
        assert_eq!(style.address_form, AddressForm::Du);
        // Ambiguous text leaves it unset.
        let mut fresh = SharedConn::default();
        let style = resolve_for(&styles, &mut fresh, &ctx("hello world"));
        assert_eq!(style.address_form, AddressForm::None);
    }

    fn sie_style() -> ResolvedStyle {
        ResolvedStyle::from_spec(&predict_core::builtin_spec("sie"))
    }

    fn du_style() -> ResolvedStyle {
        ResolvedStyle::from_spec(&predict_core::builtin_spec("du"))
    }

    #[test]
    fn sie_mode_filters_du_words_and_echoes_style() {
        let models = test_models();
        // "du " routes German; the personal carrier lives on that side.
        let mut personal = PersonalBundle::default();
        personal.de.add_text("du dich");
        let reply = suggest_for(&models, &personal, 0.0, &sie_style(), &ctx("du "), 1);
        assert_eq!(reply.style_id, "sie");
        assert!(
            !reply.candidates.iter().any(|c| c.text == "du" || c.text == "dich"),
            "banned words shown: {:?}",
            reply.candidates
        );
    }

    #[test]
    fn du_mode_filters_sie_words() {
        let models = test_models();
        let mut personal = PersonalBundle::default();
        personal.de.add_text("Sie Ihnen");
        let reply = suggest_for(&models, &personal, 0.0, &du_style(), &ctx("der Sie "), 1);
        assert_eq!(reply.style_id, "du");
        assert!(
            !reply.candidates.iter().any(|c| c.text == "Sie" || c.text == "Ihnen"),
            "banned words shown: {:?}",
            reply.candidates
        );
    }

    #[test]
    fn word_length_style_gets_no_sentence() {
        let (listener, sock) = personal_sock("wordlen");
        let models = test_models();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(stub_slow()), None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        // "wordy" is a word-length style in test_styles().
        let mut wordy = ctx("the quick ");
        wordy.style_id = "wordy".to_string();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(wordy)).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_err(), "word-length style got a sentence: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    #[test]
    fn stale_generations_are_ignored() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        assert!(state.observe(5)); // duplicate of newest still counts
        assert!(!state.observe(4)); // older is stale
        assert!(state.observe(6));
        assert!(!state.observe(5)); // now stale
    }

    #[test]
    fn fresh_generation_cancels_older_slow_job() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        let cancel = CancelToken::new();
        state.slow = Some(SlowJob {
            generation: 5,
            cancel: cancel.clone(),
        });
        assert!(state.observe(6));
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn cancel_through_aborts_current_slow_job() {
        let mut state = SharedConn::default();
        assert!(state.observe(5));
        let cancel = CancelToken::new();
        state.slow = Some(SlowJob {
            generation: 5,
            cancel: cancel.clone(),
        });
        state.cancel_through(5);
        assert!(cancel.is_cancelled());
        // Late duplicates stay stale.
        assert!(!state.observe(4));
    }

    #[test]
    fn missing_config_disables_slow_tier() {
        let dir = std::env::temp_dir().join(format!("predict-test-{}", std::process::id()));
        let missing = dir.join("no-such.toml");
        assert!(load_llm_from(&missing).is_none());
    }

    #[test]
    fn bad_config_disables_slow_tier() {
        let dir = std::env::temp_dir().join(format!("predict-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.toml");
        std::fs::write(&path, "[llm\nbroken").unwrap();
        assert!(load_llm_from(&path).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    /// Full loopback: temp socket -> daemon thread -> framed reply.
    #[test]
    fn end_to_end_suggest_over_unix_socket() {
        let sock = std::env::temp_dir().join(format!("predictd-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let models = test_models();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, None, None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("hello wo"))).unwrap();
        let start = Instant::now();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
            other => panic!("expected Suggestion, got {other:?}"),
        };
        let elapsed = start.elapsed();
        assert_eq!(reply.generation, 1);
        assert_eq!(reply.candidates[0].text, "world");
        assert!(
            elapsed < Duration::from_millis(500),
            "roundtrip too slow: {elapsed:?}"
        );

        // A stale generation gets no reply: the daemon skips it silently.
        write_client_msg(&mut client, &ClientMsg::Cancel(CancelMsg { generation: 2 })).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let stale = read_daemon_msg(&mut client);
        assert!(stale.is_err(), "stale generation got a reply: {stale:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Slow path over the socket with a stub backend.
    #[test]
    fn end_to_end_sentence_over_unix_socket() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-slow-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let models = test_models();
        let slow = stub_slow();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(slow), None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Sentence(s) => s,
            other => panic!("expected Sentence, got {other:?}"),
        };
        assert_eq!(reply.generation, 1);
        assert_eq!(reply.text, "brown fox");
        assert_eq!(reply.confidence, -0.2);
        assert_eq!(reply.style_id, "default");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// A gated-out sentence retracts (empty text) rather than staying
    /// silent, so clients clear any shown partials.
    #[test]
    fn gated_sentence_retracts() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-gate-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let models = test_models();
        let slow = SlowCfg {
            threshold: 0.0, // stub confidence -0.2 never passes
            ..stub_slow()
        };

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(slow), None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Sentence(s) => s,
            other => panic!("expected retract Sentence, got {other:?}"),
        };
        assert_eq!(reply.generation, 1);
        assert!(reply.text.is_empty(), "retract must clear: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Slow backend that blocks in cancel-aware slices (deterministic abort).
    struct BlockingBackend;

    impl Backend for BlockingBackend {
        fn complete_sentence(
            &self,
            _req: &SentenceRequest,
            cancel: &CancelToken,
        ) -> Result<Option<SentenceOutput>, predict_llm::LlmError> {
            for _ in 0..10 {
                if cancel.is_cancelled() {
                    return Err(predict_llm::LlmError::Cancelled);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(Some(SentenceOutput {
                text: "slow result".to_string(),
                confidence: -0.1,
                time_to_first_token: Duration::from_millis(50),
                tokens_generated: 2,
            }))
        }

        fn name(&self) -> &str {
            "blocking-test"
        }
    }

    /// Cancelling in-flight slow work produces silence, not a stale reply.
    #[test]
    fn cancelled_sentence_gets_no_reply() {
        let sock = std::env::temp_dir()
            .join(format!("predictd-cancel-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        let models = test_models();
        let slow = SlowCfg {
            backend: Arc::new(BlockingBackend),
            max_tokens: 32,
            threshold: -1.5,
        };

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(slow), None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the quick "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        // Abort before the 50 ms backend can finish (it polls every 5 ms).
        write_client_msg(&mut client, &ClientMsg::Cancel(CancelMsg { generation: 1 })).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_err(), "cancelled sentence got a reply: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    fn test_personal() -> (Personal, Arc<Mutex<Store>>) {
        let store = Store::open_in_memory().unwrap();
        let store = Arc::new(Mutex::new(store));
        let personal = Personal {
            store: Arc::clone(&store),
            bundles: Arc::new(RwLock::new(std::collections::HashMap::new())),
            lambda: 0.5,
            paused: Arc::new(AtomicBool::new(false)),
        };
        (personal, store)
    }

    fn personal_sock(name: &str) -> (UnixListener, std::path::PathBuf) {
        let sock = std::env::temp_dir().join(format!("predictd-{name}-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();
        (listener, sock)
    }

    fn read_learning(
        client: &mut UnixStream,
    ) -> predict_proto::LearningState {
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        match read_daemon_msg(client).unwrap() {
            DaemonMsg::LearningState(state) => state,
            other => panic!("expected LearningState, got {other:?}"),
        }
    }

    /// Send settled text and return the commit ack.
    fn commit_text(
        client: &mut UnixStream,
        text: &str,
        sensitive: bool,
    ) -> predict_proto::LearningState {
        write_client_msg(
            client,
            &ClientMsg::CommitText(CommitText {
                text: text.to_string(),
                style_id: "default".to_string(),
                sensitive,
            }),
        )
        .unwrap();
        read_learning(client)
    }

    /// Settled commits teach the word tier through the full daemon path.
    #[test]
    fn personal_commit_boosts_suggestion() {
        let (listener, sock) = personal_sock("boost");
        let models = test_models();
        let (personal, _store) = test_personal();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, None, Some(personal), test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        // "grilled" is unknown to the base models; learn it as settled text.
        // The commit ack doubles as ordering proof (daemon loop is sequential).
        let ack = commit_text(&mut client, "grilled courgette", false);
        assert_eq!(ack.documents, 1);
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("grilled "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
            other => panic!("expected Suggestion, got {other:?}"),
        };
        assert_eq!(reply.candidates[0].text, "courgette");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Sensitive commits are never stored.
    #[test]
    fn sensitive_commit_is_ignored() {
        let (listener, sock) = personal_sock("sensitive");
        let models = test_models();
        let (personal, store) = test_personal();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, None, Some(personal), test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        // Sensitive commits get no ack (silence by design): send without
        // waiting; the Suggest round-trip proves ordering.
        write_client_msg(
            &mut client,
            &ClientMsg::CommitText(CommitText {
                text: "s3cret password hunter2".to_string(),
                style_id: "default".to_string(),
                sensitive: true,
            }),
        )
        .unwrap();
        // A normal suggest still answers after the ignored commit.
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_ok(), "word tier broke after sensitive commit");
        assert_eq!(
            store.lock().unwrap().doc_count().unwrap(),
            0,
            "sensitive text was stored"
        );

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Forget-all wipes documents and counts; the daemon reports it.
    #[test]
    fn forget_all_clears_and_reports() {
        let (listener, sock) = personal_sock("forget");
        let models = test_models();
        let (personal, _store) = test_personal();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, None, Some(personal), test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        for text in ["grilled courgette", "another line here"] {
            let ack = commit_text(&mut client, text, false);
            assert!(ack.documents >= 1);
        }
        // Personal boost visible before forgetting.
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("grilled "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let before = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
            other => panic!("expected Suggestion, got {other:?}"),
        };
        assert_eq!(before.candidates[0].text, "courgette");

        write_client_msg(&mut client, &ClientMsg::ForgetAll).unwrap();
        let state = read_learning(&mut client);
        assert_eq!(state.documents, 0);

        // Boost gone after forgetting.
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("grilled "))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::Suggest(SuggestRequest { generation: 2 }),
        )
        .unwrap();
        let after = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Suggestion(s) => s,
            other => panic!("expected Suggestion, got {other:?}"),
        };
        assert_ne!(after.candidates[0].text, "courgette");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Pause-learning ignores commits until resumed.
    #[test]
    fn pause_learning_toggle() {
        let (listener, sock) = personal_sock("pause");
        let models = test_models();
        let (personal, store) = test_personal();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, None, Some(personal), test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SetLearning(SetLearning { enabled: false }),
        )
        .unwrap();
        let paused = read_learning(&mut client);
        assert!(!paused.enabled);

        // The commit ack proves the daemon processed it (sequential loop).
        let ack = commit_text(&mut client, "paused line here", false);
        assert_eq!(ack.documents, 0);
        assert_eq!(store.lock().unwrap().doc_count().unwrap(), 0);

        write_client_msg(
            &mut client,
            &ClientMsg::SetLearning(SetLearning { enabled: true }),
        )
        .unwrap();
        let resumed = read_learning(&mut client);
        assert!(resumed.enabled);
        let ack = commit_text(&mut client, "resumed line here", false);
        assert_eq!(ack.documents, 1);
        assert_eq!(store.lock().unwrap().doc_count().unwrap(), 1);

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Sensitive fields get no sentence suggestions either (M3 gap).
    #[test]
    fn sensitive_sentence_gets_no_reply() {
        let (listener, sock) = personal_sock("sensitive-sent");
        let models = test_models();

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(stub_slow()), None, test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        let mut sensitive = ctx("hello wo");
        sensitive.sensitive = true;
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(sensitive)).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let reply = read_daemon_msg(&mut client);
        assert!(reply.is_err(), "sensitive sentence got a reply: {reply:?}");

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    /// Backend that echoes the first words of the prompt it received, so
    /// tests can observe retrieval grounding without a model.
    struct EchoBackend;

    impl Backend for EchoBackend {
        fn complete_sentence(
            &self,
            req: &SentenceRequest,
            cancel: &CancelToken,
        ) -> Result<Option<SentenceOutput>, predict_llm::LlmError> {
            if cancel.is_cancelled() {
                return Err(predict_llm::LlmError::Cancelled);
            }
            let words: Vec<&str> = req.before.split_whitespace().take(8).collect();
            if words.is_empty() {
                return Ok(None);
            }
            Ok(Some(SentenceOutput {
                text: format!("echo:{}", words.join(" ")),
                confidence: -0.1,
                time_to_first_token: Duration::from_millis(1),
                tokens_generated: words.len(),
            }))
        }

        fn name(&self) -> &str {
            "echo-test"
        }
    }

    /// Retrieved snippets reach the slow-tier prompt (echo proves it).
    #[test]
    fn retrieval_grounds_sentence_prompt() {
        let (listener, sock) = personal_sock("retrieval");
        let models = test_models();
        let (personal, _store) = test_personal();
        let slow = SlowCfg {
            backend: Arc::new(EchoBackend),
            max_tokens: 32,
            threshold: -1.5,
        };

        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(stream, models, Some(slow), Some(personal), test_styles()).unwrap();
        });

        let mut client = UnixStream::connect(&sock).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        // Consume the commit ack so the sentence read below is aligned.
        let ack = commit_text(&mut client, "the vessel hovers above eels", false);
        assert_eq!(ack.documents, 1);
        write_client_msg(&mut client, &ClientMsg::ContextUpdate(ctx("the vessel hovers"))).unwrap();
        write_client_msg(
            &mut client,
            &ClientMsg::SuggestSentence(SuggestRequest { generation: 1 }),
        )
        .unwrap();
        let reply = match read_daemon_msg(&mut client).unwrap() {
            DaemonMsg::Sentence(s) => s,
            other => panic!("expected Sentence, got {other:?}"),
        };
        // "Related notes" header only exists when snippets grounded the prompt.
        assert!(
            reply.text.contains("Related"),
            "no grounding in reply: {:?}",
            reply.text
        );

        drop(client);
        server.join().unwrap();
        std::fs::remove_file(&sock).unwrap();
    }

    #[test]
    fn personal_config_parses_with_defaults() {
        let cfg = PersonalCfg::from_toml_str("", predict_store::default_db_path()).unwrap();
        assert!(cfg.enabled);
        assert!((cfg.lambda - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.db_path, predict_store::default_db_path());
        let cfg = PersonalCfg::from_toml_str(
            "[personal]\nenabled = false\nlambda = 0.3\n",
            predict_store::default_db_path(),
        )
        .unwrap();
        assert!(!cfg.enabled);
        assert!((cfg.lambda - 0.3).abs() < f32::EPSILON);
        assert!(PersonalCfg::from_toml_str("[personal\nbroken", predict_store::default_db_path()).is_err());
    }
}
