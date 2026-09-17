//! Personal memory: SQLite store for settled text and personal counts.
//!
//! Stores settled text only (committed by the frontend after a pause or
//! field leave) — never keystrokes, never anything sensitive (the daemon
//! filters those before calling here). Feeds two consumers: personal
//! n-gram counts for blending, and FTS5 (BM25) retrieval for grounding.

use predict_core::tokenize_text;
use predict_ngram::{Language, PersonalBundle, PersonalCounts, detect_language};
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors from the personal store.
#[derive(Debug, Error)]
pub enum StoreError {
    /// SQLite failure.
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
    /// Filesystem failure (creating the database directory).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Personal SQLite store: settled documents, per-language n-gram counts,
/// and an FTS5 index over document text.
pub struct Store {
    conn: Connection,
}

impl Store {
    fn init_schema(conn: &Connection) -> Result<(), StoreError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents(
                 id INTEGER PRIMARY KEY,
                 text TEXT NOT NULL,
                 style_id TEXT NOT NULL DEFAULT 'default',
                 lang TEXT NOT NULL DEFAULT '',
                 created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS counts(
                 lang TEXT NOT NULL,
                 style_id TEXT NOT NULL DEFAULT 'default',
                 n INTEGER NOT NULL,
                 ctx TEXT NOT NULL DEFAULT '',
                 word TEXT NOT NULL,
                 count INTEGER NOT NULL,
                 PRIMARY KEY (lang, style_id, n, ctx, word)
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                 USING fts5(text, style_id, content='documents', content_rowid='id',
                            tokenize='unicode61');
             CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                 INSERT INTO documents_fts(rowid, text, style_id)
                     VALUES (new.id, new.text, new.style_id);
             END;
             CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                 INSERT INTO documents_fts(documents_fts, rowid, text, style_id)
                     VALUES ('delete', old.id, old.text, old.style_id);
             END;
             CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                 INSERT INTO documents_fts(documents_fts, rowid, text, style_id)
                     VALUES ('delete', old.id, old.text, old.style_id);
                 INSERT INTO documents_fts(rowid, text, style_id)
                     VALUES (new.id, new.text, new.style_id);
             END;",
        )?;
        Ok(())
    }

    /// Open (creating parents as needed) a file-backed store.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Open an in-memory store (tests, eval harness).
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Store one settled commit. Returns false (storing nothing) for blank
    /// text. Detects the language for count attribution and tags the
    /// active `style_id` (M5 filters counts and retrieval by it).
    pub fn commit(&self, text: &str, style_id: &str) -> Result<bool, StoreError> {
        if text.trim().is_empty() {
            return Ok(false);
        }
        let lang = match detect_language(text) {
            Language::En => "en",
            Language::De => "de",
            Language::Unknown => "",
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO documents(text, style_id, lang, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![text, style_id, lang, now],
        )?;
        let tokens = tokenize_text(text);
        for side in count_langs(lang) {
            bump_windows(&tx, side, style_id, &tokens)?;
        }
        tx.commit()?;
        Ok(true)
    }

    /// Number of stored documents.
    pub fn doc_count(&self) -> Result<u64, StoreError> {
        let count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        Ok(count.max(0) as u64)
    }

    /// Personal counts for one language side, optionally filtered to one
    /// style id (M5; `None` counts everything).
    pub fn personal_counts(
        &self,
        lang: Language,
        style: Option<&str>,
    ) -> Result<PersonalCounts, StoreError> {
        let tag = lang_tag(lang);
        let mut counts = PersonalCounts::new();
        // Scoped loops (not match-collect): the row iterator borrows the
        // statement, so each must drain before its statement drops.
        if let Some(style_id) = style {
            let mut stmt = self.conn.prepare(
                "SELECT n, ctx, word, count FROM counts WHERE lang = ?1 AND style_id = ?2",
            )?;
            for row in stmt.query_map(params![tag, style_id], decode_count_row)? {
                let (n, ctx, word, count) = row?;
                counts.add_ngram(n as u32, &ctx, &word, count.max(0) as u64);
            }
        } else {
            let mut stmt = self
                .conn
                .prepare("SELECT n, ctx, word, count FROM counts WHERE lang = ?1")?;
            for row in stmt.query_map([tag], decode_count_row)? {
                let (n, ctx, word, count) = row?;
                counts.add_ngram(n as u32, &ctx, &word, count.max(0) as u64);
            }
        }
        Ok(counts)
    }

    /// Top-`limit` stored texts matching `query` (FTS5 BM25 order),
    /// optionally filtered to one style id (M5; `None` searches everything).
    ///
    /// The query runs as a quoted phrase (user text is never FTS syntax),
    /// truncated snippets of ~240 chars.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        style: Option<&str>,
    ) -> Result<Vec<String>, StoreError> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let phrase = format!("\"{}\"", query.replace('"', "\"\""));
        let mut out = Vec::new();
        if let Some(style_id) = style {
            let mut stmt = self.conn.prepare(
                "SELECT text FROM documents_fts WHERE documents_fts MATCH ?1
                 AND style_id = ?2 ORDER BY rank LIMIT ?3",
            )?;
            for row in stmt.query_map(params![phrase, style_id, limit as i64], |row| {
                row.get::<_, String>(0)
            })? {
                out.push(snippet(&row?));
            }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT text FROM documents_fts WHERE documents_fts MATCH ?1
                 ORDER BY rank LIMIT ?2",
            )?;
            for row in stmt.query_map(params![phrase, limit as i64], |row| {
                row.get::<_, String>(0)
            })? {
                out.push(snippet(&row?));
            }
        }
        Ok(out)
    }

    /// All personal bundles keyed by style id (one SQLite pass). The
    /// daemon caches these and rebuilds on commits/forget-all, so no
    /// per-keystroke SQL is needed.
    pub fn personal_bundles(&self) -> Result<HashMap<String, PersonalBundle>, StoreError> {
        let mut stmt = self
            .conn
            .prepare("SELECT lang, style_id, n, ctx, word, count FROM counts")?;
        let mut bundles: HashMap<String, PersonalBundle> = HashMap::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?;
        for row in rows {
            let (lang, style_id, n, ctx, word, count) = row?;
            let bundle = bundles.entry(style_id).or_default();
            let counts = match lang.as_str() {
                "de" => &mut bundle.de,
                _ => &mut bundle.en,
            };
            counts.add_ngram(n as u32, &ctx, &word, count.max(0) as u64);
        }
        Ok(bundles)
    }

    /// Remove every document, count, and index entry (forget-all).
    pub fn clear_all(&self) -> Result<(), StoreError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM documents", [])?;
        tx.execute("DELETE FROM counts", [])?;
        tx.commit()?;
        Ok(())
    }
}

/// Count attribution sides for a commit: the detected language, or both
/// sides when detection is ambiguous. Marker-less personal vocabulary
/// ("grilled courgette", project names) would otherwise never be learned;
/// the small cross-side pollution this causes is bounded by the blend
/// weight (measured in the temporal eval).
fn count_langs(lang: &str) -> &'static [&'static str] {
    match lang {
        "en" => &["en"],
        "de" => &["de"],
        _ => &["en", "de"],
    }
}

/// Language tag used in the `counts`/`documents` tables.
fn lang_tag(lang: Language) -> &'static str {
    match lang {
        Language::En => "en",
        Language::De => "de",
        Language::Unknown => "",
    }
}

/// Decode one `counts` row for [`Store::personal_counts`].
fn decode_count_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, String, String, i64)> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
    ))
}

/// Truncate a retrieved document to snippet size on a char boundary.
fn snippet(text: &str) -> String {
    const MAX: usize = 240;
    if text.chars().count() <= MAX {
        return text.to_string();
    }
    format!(
        "{}…",
        text.chars().take(MAX.saturating_sub(1)).collect::<String>()
    )
}

/// Add uni/bi/trigram counts for one token sequence to the open transaction.
fn bump_windows(
    tx: &rusqlite::Transaction<'_>,
    lang: &str,
    style_id: &str,
    tokens: &[String],
) -> Result<(), StoreError> {
    {
        let mut upsert = tx.prepare_cached(
            "INSERT INTO counts(lang, style_id, n, ctx, word, count) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(lang, style_id, n, ctx, word) DO UPDATE SET count = count + excluded.count",
        )?;
        for window in tokens.windows(1) {
            upsert.execute(params![lang, style_id, 1, "", window[0], 1])?;
        }
        for pair in tokens.windows(2) {
            upsert.execute(params![lang, style_id, 2, pair[0], pair[1], 1])?;
        }
        for triple in tokens.windows(3) {
            let ctx = format!("{} {}", triple[0], triple[1]);
            upsert.execute(params![lang, style_id, 3, ctx, triple[2], 1])?;
        }
    }
    Ok(())
}

/// Default on-disk location for the personal database.
pub fn default_db_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("predict/predict.db");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".local/share/predict/predict.db")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_stores_doc_and_counts() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.commit("the quick brown fox", "default").unwrap());
        assert_eq!(store.doc_count().unwrap(), 1);
        let en = store.personal_counts(Language::En, None).unwrap();
        assert_eq!(en.total_tokens(), 4);
        assert!(!en.is_empty());
        // German side untouched by English text.
        assert!(store.personal_counts(Language::De, None).unwrap().is_empty());
    }

    #[test]
    fn blank_commits_store_nothing() {
        let store = Store::open_in_memory().unwrap();
        assert!(!store.commit("   \n  ", "default").unwrap());
        assert_eq!(store.doc_count().unwrap(), 0);
    }

    #[test]
    fn german_commit_lands_on_german_side() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.commit("der schnelle braune Fuchs", "default").unwrap());
        assert!(store.personal_counts(Language::En, None).unwrap().is_empty());
        let de = store.personal_counts(Language::De, None).unwrap();
        assert_eq!(de.total_tokens(), 4);
    }

    #[test]
    fn search_finds_settled_text_by_bm25() {
        let store = Store::open_in_memory().unwrap();
        store.commit("the quick brown fox jumps", "default").unwrap();
        store.commit("vielen Dank für die Blumen", "default").unwrap();
        let hits = store.search("quick brown", 3, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].contains("quick brown"));
        // Unknown-side phrase with quotes is safely quoted, not syntax.
        assert!(store.search("quick \"brown", 3, None).unwrap().len() <= 1);
        assert!(store.search("", 3, None).unwrap().is_empty());
        assert!(store.search("quick", 0, None).unwrap().is_empty());
    }

    #[test]
    fn counts_reload_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        store.commit("grilled courgette grilled courgette", "default").unwrap();
        let en = store.personal_counts(Language::En, None).unwrap();
        // Same content as the equivalent in-memory build.
        let mut direct = PersonalCounts::new();
        direct.add_text("grilled courgette grilled courgette");
        assert_eq!(en.total_tokens(), direct.total_tokens());
        assert_eq!(en.complete("cour", 5), direct.complete("cour", 5));
        // Ambiguous fragments land on both sides.
        let de = store.personal_counts(Language::De, None).unwrap();
        assert_eq!(de.total_tokens(), direct.total_tokens());
    }

    #[test]
    fn style_id_is_recorded() {
        let store = Store::open_in_memory().unwrap();
        store.commit("hello world", "formal").unwrap();
        let style: String = store
            .conn
            .query_row("SELECT style_id FROM documents", [], |row| row.get(0))
            .unwrap();
        assert_eq!(style, "formal");
    }

    #[test]
    fn style_filter_scopes_counts_and_search() {
        let store = Store::open_in_memory().unwrap();
        store.commit("the quick brown fox", "casual").unwrap();
        store.commit("the lazy dog sleeps", "formal").unwrap();
        // Unfiltered sees everything.
        assert_eq!(
            store
                .personal_counts(Language::En, None)
                .unwrap()
                .total_tokens(),
            8
        );
        assert_eq!(store.search("the", 5, None).unwrap().len(), 2);
        // Filtered to one style sees only its half.
        let casual = store
            .personal_counts(Language::En, Some("casual"))
            .unwrap();
        assert_eq!(casual.total_tokens(), 4);
        assert!(casual.complete("laz", 5).is_empty());
        assert_eq!(casual.complete("bro", 5)[0].0, "brown");
        assert_eq!(store.search("the", 5, Some("casual")).unwrap().len(), 1);
        assert_eq!(store.search("the", 5, Some("formal")).unwrap().len(), 1);
        assert!(store.search("the", 5, Some("nope")).unwrap().is_empty());
    }

    #[test]
    fn bundles_group_by_style() {
        let store = Store::open_in_memory().unwrap();
        store.commit("the quick brown fox", "casual").unwrap();
        store.commit("der schnelle Fuchs", "formal").unwrap();
        let bundles = store.personal_bundles().unwrap();
        assert_eq!(bundles.len(), 2);
        assert_eq!(bundles["casual"].en.total_tokens(), 4);
        assert!(bundles["casual"].de.is_empty());
        assert_eq!(bundles["formal"].de.total_tokens(), 3);
        assert!(bundles.get("nope").is_none());
    }

    #[test]
    fn forget_all_removes_everything() {
        let store = Store::open_in_memory().unwrap();
        store.commit("the quick brown fox", "default").unwrap();
        store.commit("der schnelle Fuchs", "default").unwrap();
        assert_eq!(store.doc_count().unwrap(), 2);
        store.clear_all().unwrap();
        assert_eq!(store.doc_count().unwrap(), 0);
        assert!(store.personal_counts(Language::En, None).unwrap().is_empty());
        assert!(store.personal_counts(Language::De, None).unwrap().is_empty());
        assert!(store.search("quick", 3, None).unwrap().is_empty());
        assert!(store.search("Fuchs", 3, None).unwrap().is_empty());
    }

    #[test]
    fn file_store_creates_parents() {
        let dir = std::env::temp_dir().join(format!("predict-store-test-{}", std::process::id()));
        let path = dir.join("sub").join("test.db");
        let _ = std::fs::remove_dir_all(&dir);
        let store = Store::open(&path).unwrap();
        assert!(store.commit("hello world", "default").unwrap());
        assert_eq!(store.doc_count().unwrap(), 1);
        drop(store);
        let reopened = Store::open(&path).unwrap();
        assert_eq!(reopened.doc_count().unwrap(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snippet_truncates_long_docs() {
        assert_eq!(snippet("short"), "short");
        let long = "w".repeat(300);
        let cut = snippet(&long);
        assert!(cut.ends_with('…'));
        assert!(cut.chars().count() <= 240);
    }
}
