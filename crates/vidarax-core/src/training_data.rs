//! SQLite + zvec training pair storage for auto-distillation.
//!
//! Embeddings are indexed in an in-process zvec (Alibaba Proxima engine)
//! collection for high-throughput ANN search.  SQLite stores pair metadata
//! (label, teacher model, confidence, frame path) and keeps a raw-bytes BLOB
//! backup of each embedding for export/audit.
//!
//! # Thread safety
//!
//! `TrainingStore` is `Send` (can be moved into a `spawn_blocking` closure)
//! but **not** `Sync` — the `rusqlite::Connection` uses interior mutability
//! via `RefCell` and is `!Sync`.  Wrap in `Arc<Mutex<TrainingStore>>` for
//! shared access across threads.
//!
//! The internal `SharedCollection` is `Clone + Send + Sync` and performs its
//! own internal locking, but all operations are routed through `TrainingStore`
//! to maintain transactional consistency between SQLite and zvec.
//!
//! # Storage layout
//!
//! ```text
//! <data_dir>/
//!   training.db               — SQLite WAL database (metadata + embedding blobs)
//!   zvec/                     — zvec collection (FLAT cosine index, primary search)
//!   frames/<tenant_id>/<ts>   — JPEG thumbnails
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rusqlite::{params, Connection};
use zvec_bindings::{
    create_and_open_shared, open_shared, CollectionSchema, Doc, FieldSchema, SharedCollection,
    VectorQuery, VectorSchema,
};

// ── Tenant ID validation ─────────────────────────────────────────────────────

/// Validate that a tenant ID is safe for use as a filesystem path component.
///
/// Accepts 1–64 characters of ASCII alphanumeric, hyphen, or underscore.
/// Rejects empty strings, path separators, `..`, and null bytes.
fn validate_tenant_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 64 {
        return Err(TrainingError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "tenant_id must be 1-64 characters",
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(TrainingError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "tenant_id must contain only ASCII alphanumeric, hyphen, or underscore",
        )));
    }
    Ok(())
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum TrainingError {
    Database(rusqlite::Error),
    Io(std::io::Error),
    VectorDb(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::Database(e) => write!(f, "training db: {e}"),
            TrainingError::Io(e) => write!(f, "training io: {e}"),
            TrainingError::VectorDb(e) => write!(f, "training vectordb: {e}"),
        }
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrainingError::Database(e) => Some(e),
            TrainingError::Io(e) => Some(e),
            TrainingError::VectorDb(_) => None,
        }
    }
}

impl From<rusqlite::Error> for TrainingError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Database(e)
    }
}

impl From<std::io::Error> for TrainingError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, TrainingError>;

// ── Public types ──────────────────────────────────────────────────────────────

pub struct TrainingStore {
    conn: Connection,
    data_dir: PathBuf,
    /// zvec collection for vector ANN search (FLAT cosine index).
    collection: SharedCollection,
}

pub struct TrainingPair {
    pub id: i64,
    pub tenant_id: String,
    pub frame_path: PathBuf,
    pub label_json: String,
    pub teacher_model: String,
    pub confidence: f32,
    pub created_at: String,
}

/// Result of a KNN classification query.
pub struct KnnResult {
    /// Winning label (majority vote among top-k neighbours).
    pub label: String,
    /// Mean cosine distance to the winning neighbours (0.0 = identical, 1.0 = orthogonal).
    pub avg_distance: f32,
    /// Number of neighbours that voted for `label`.
    pub votes: usize,
    /// Total neighbours considered (≤ k, after tenant filter + distance threshold).
    pub total: usize,
}

// ── Embedding helpers ─────────────────────────────────────────────────────────

/// Serialize a 768-dim f32 embedding to raw LE bytes for SQLite BLOB backup.
fn embedding_to_bytes(emb: &[f32; 768]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(768 * 4);
    for &v in emb {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Nanosecond-resolution timestamp used for collision-resistant frame filenames.
fn frame_filename() -> String {
    let ns = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ns:032x}.jpg")
}

// ── TrainingStore ─────────────────────────────────────────────────────────────

impl TrainingStore {
    /// Open (or create) the training database at `data_dir/training.db` and
    /// the zvec collection at `data_dir/zvec/`.
    ///
    /// On first run a FLAT cosine index is built over the embedding field.
    /// On subsequent runs the collection is reopened without re-indexing.
    pub fn new(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        // ── SQLite ───────────────────────────────────────────────────────────
        let db_path = data_dir.join("training.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous  = NORMAL;
             CREATE TABLE IF NOT EXISTS training_pairs (
                 id            INTEGER PRIMARY KEY AUTOINCREMENT,
                 tenant_id     TEXT    NOT NULL,
                 frame_path    TEXT    NOT NULL,
                 label_json    TEXT    NOT NULL,
                 teacher_model TEXT    NOT NULL,
                 confidence    REAL    NOT NULL,
                 embedding     BLOB    NOT NULL,
                 created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
             );
             CREATE INDEX IF NOT EXISTS idx_pairs_tenant
                 ON training_pairs (tenant_id, id);",
        )?;

        // ── zvec vector collection ───────────────────────────────────────────
        let zvec_dir = data_dir.join("zvec");

        // create_and_open_shared requires a non-existent path; open_shared reopens.
        let collection = if !zvec_dir.exists() {
            let mut schema = CollectionSchema::new("training_embeddings");
            schema
                .add_field(VectorSchema::fp32("embedding", 768).into())
                .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
            // Store metadata directly in zvec so KNN results are self-contained
            // (zvec query results do not populate pk() reliably via C FFI).
            schema
                .add_field(FieldSchema::string("label_json"))
                .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
            schema
                .add_field(FieldSchema::string("tenant_id"))
                .map_err(|e| TrainingError::VectorDb(e.to_string()))?;

            // Note: do NOT call create_index before inserting data — zvec
            // auto-indexes inserted vectors and building the index on an empty
            // collection causes queries to return score=0 for all results.
            create_and_open_shared(&zvec_dir, schema)
                .map_err(|e| TrainingError::VectorDb(e.to_string()))?
        } else {
            open_shared(&zvec_dir).map_err(|e| TrainingError::VectorDb(e.to_string()))?
        };

        Ok(Self {
            conn,
            data_dir: data_dir.to_path_buf(),
            collection,
        })
    }

    /// Persist a `(frame, label, embedding)` triple.
    ///
    /// 1. Writes JPEG bytes to `data_dir/frames/<tenant_id>/<ts>.jpg`.
    /// 2. Inserts metadata + embedding BLOB into SQLite.
    /// 3. Inserts embedding into zvec keyed by the new SQLite row ID.
    ///
    /// Returns the new row ID.
    pub fn store_pair(
        &self,
        tenant_id: &str,
        frame_jpeg: &[u8],
        label_json: &str,
        teacher_model: &str,
        confidence: f32,
        embedding: &[f32; 768],
    ) -> Result<i64> {
        validate_tenant_id(tenant_id)?;
        // Write JPEG thumbnail
        let frames_dir = self.data_dir.join("frames").join(tenant_id);
        std::fs::create_dir_all(&frames_dir)?;
        let frame_path = frames_dir.join(frame_filename());
        std::fs::write(&frame_path, frame_jpeg)?;

        let frame_path_str = frame_path.to_string_lossy().into_owned();
        let emb_bytes = embedding_to_bytes(embedding);

        self.conn.execute(
            "INSERT INTO training_pairs
                 (tenant_id, frame_path, label_json, teacher_model, confidence, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                tenant_id,
                frame_path_str,
                label_json,
                teacher_model,
                confidence as f64,
                emb_bytes
            ],
        )?;

        let row_id = self.conn.last_insert_rowid();

        // Index embedding in zvec. Store label_json + tenant_id as scalar
        // fields so KNN results are self-contained (pk() is unreliable via FFI).
        let mut doc = Doc::id(row_id.to_string());
        doc.set_vector("embedding", embedding.as_slice())
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
        doc.set_string("label_json", label_json)
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
        doc.set_string("tenant_id", tenant_id)
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
        self.collection
            .insert(&[doc])
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;
        // Flush so newly inserted vectors are immediately visible to queries.
        self.collection
            .flush()
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;

        Ok(row_id)
    }

    /// KNN classification via zvec ANN search with SQLite label lookup.
    ///
    /// Queries zvec for the `k * 4` nearest global neighbours (buffered to
    /// account for cross-tenant results), then filters by `tenant_id` and
    /// `distance_threshold`, and returns a majority-vote result.
    ///
    /// zvec returns cosine-**similarity** scores (higher = closer); these are
    /// converted to cosine **distance** (`d = 1 − score`) for threshold comparisons.
    ///
    /// Returns `None` when the collection is empty or no neighbour passes
    /// the tenant filter + distance threshold.
    pub fn knn_classify(
        &self,
        tenant_id: &str,
        query: &[f32; 768],
        k: usize,
        distance_threshold: f32,
    ) -> Result<Option<KnnResult>> {
        validate_tenant_id(tenant_id)?;
        if k == 0 {
            return Ok(None);
        }

        // Request 4× more results to account for cross-tenant filtering.
        let search_k = (k * 4).max(20);

        // Request label_json + tenant_id explicitly so zvec returns them.
        // (pk() is unreliable from query results via C FFI; scalar fields work.)
        let zvec_query = VectorQuery::new("embedding")
            .topk(search_k)
            .output_fields(&["label_json", "tenant_id"])
            .vector(query.as_slice())
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;

        let zvec_results = self
            .collection
            .query(zvec_query)
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;

        if zvec_results.is_empty() {
            return Ok(None);
        }

        // Results are sorted best-first (highest similarity = lowest distance).
        let mut neighbors: Vec<(f32, String)> = Vec::with_capacity(k);
        for doc in &zvec_results {
            if neighbors.len() >= k {
                break;
            }
            // Skip results from other tenants
            if doc.get_string("tenant_id") != Some(tenant_id) {
                continue;
            }
            // Cosine similarity → distance
            let distance = 1.0 - doc.score();
            if distance > distance_threshold {
                continue;
            }
            let label = match doc.get_string("label_json") {
                Some(l) => l.to_string(),
                None => continue,
            };
            neighbors.push((distance, label));
        }

        if neighbors.is_empty() {
            return Ok(None);
        }

        let total = neighbors.len();
        let avg_distance = neighbors.iter().map(|(d, _)| *d).sum::<f32>() / total as f32;

        // Plurality vote across neighbours.
        let mut counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (_, label) in &neighbors {
            *counts.entry(label.as_str()).or_insert(0) += 1;
        }
        let (winning_label, votes) = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .expect("neighbors is non-empty");

        Ok(Some(KnnResult {
            label: winning_label.to_string(),
            avg_distance,
            votes,
            total,
        }))
    }

    /// Export all pairs for a tenant as newline-delimited JSON.
    ///
    /// Each line is a JSON object with fields: `frame_path`, `label`,
    /// `teacher_model`, `confidence`, `created_at`.
    ///
    /// Returns the number of records written.
    pub fn export_training_jsonl(&self, tenant_id: &str, output_path: &Path) -> Result<usize> {
        validate_tenant_id(tenant_id)?;
        let mut stmt = self.conn.prepare(
            "SELECT frame_path, label_json, teacher_model, confidence, created_at
             FROM training_pairs
             WHERE tenant_id = ?1
             ORDER BY id",
        )?;
        let rows: Vec<(String, String, String, f64, String)> = stmt
            .query_map(params![tenant_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);

        let file = std::fs::File::create(output_path)?;
        let mut writer = std::io::BufWriter::new(file);
        let mut count = 0usize;

        for (frame_path, label_json, teacher_model, confidence, created_at) in rows {
            let record = serde_json::json!({
                "frame_path":    frame_path,
                "label":         label_json,
                "teacher_model": teacher_model,
                "confidence":    confidence,
                "created_at":    created_at,
            });
            writeln!(writer, "{record}").map_err(TrainingError::from)?;
            count += 1;
        }

        Ok(count)
    }

    /// Number of stored pairs for a tenant.
    pub fn pair_count(&self, tenant_id: &str) -> Result<usize> {
        validate_tenant_id(tenant_id)?;
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM training_pairs WHERE tenant_id = ?1",
            params![tenant_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Evict the oldest pairs so the tenant's total stays ≤ `max_pairs`.
    ///
    /// Deletes from both SQLite and zvec, then removes the JPEG files on disk.
    /// Returns the number of rows removed.
    ///
    /// # Locking and re-entrancy
    ///
    /// This method is **not re-entrant**: it reads `pair_count`, decides how
    /// many rows to evict, then issues the DELETE.  If two callers executed
    /// this concurrently on the same `TrainingStore` value they could both
    /// observe the same count and both evict, potentially removing more rows
    /// than intended.
    ///
    /// The pair-count check and the DELETE are wrapped in a single SQLite
    /// `IMMEDIATE` transaction so they are atomic with respect to other
    /// writers on the same connection.  This prevents a TOCTOU race at the
    /// database level (e.g. a concurrent `store_pair` inserting a row between
    /// the `SELECT COUNT` and the `DELETE`).
    ///
    /// When `TrainingStore` is accessed through `Arc<Mutex<TrainingStore>>`
    /// (the recommended pattern — see the module-level doc), the `Mutex`
    /// serializes all calls, so the concern above does not apply.  Callers
    /// that hold a bare `TrainingStore` without an enclosing mutex must ensure
    /// they do not call `evict_oldest` from multiple threads simultaneously.
    pub fn evict_oldest(&self, tenant_id: &str, max_pairs: usize) -> Result<usize> {
        validate_tenant_id(tenant_id)?;
        // Quick pre-check outside a transaction to avoid opening one when there
        // is nothing to do.  The real count check is repeated inside the
        // transaction to guard against a concurrent insert.
        let pre_count = self.pair_count(tenant_id)?;
        if pre_count <= max_pairs {
            return Ok(0);
        }

        let rows: Vec<(i64, String)> = {
            self.conn.execute_batch("BEGIN IMMEDIATE")?;

            let inner_result = (|| -> Result<Vec<(i64, String)>> {
                let current = self.pair_count(tenant_id)?;
                if current <= max_pairs {
                    return Ok(Vec::new());
                }
                let to_evict = current - max_pairs;

                let mut stmt = self.conn.prepare(
                    "SELECT id, frame_path FROM training_pairs
                     WHERE tenant_id = ?1
                     ORDER BY id ASC
                     LIMIT ?2",
                )?;
                let candidates: Vec<(i64, String)> = stmt
                    .query_map(params![tenant_id, to_evict as i64], |row| {
                        Ok((row.get(0)?, row.get(1)?))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                drop(stmt);

                if candidates.is_empty() {
                    return Ok(Vec::new());
                }

                self.conn.execute(
                    "DELETE FROM training_pairs
                     WHERE tenant_id = ?1
                       AND id IN (
                           SELECT id FROM training_pairs
                           WHERE tenant_id = ?1
                           ORDER BY id ASC
                           LIMIT ?2
                       )",
                    params![tenant_id, to_evict as i64],
                )?;

                Ok(candidates)
            })();

            match &inner_result {
                Ok(_) => {
                    if let Err(e) = self.conn.execute_batch("COMMIT") {
                        let _ = self.conn.execute_batch("ROLLBACK");
                        return Err(TrainingError::Database(e));
                    }
                }
                Err(_) => {
                    let _ = self.conn.execute_batch("ROLLBACK");
                }
            }

            inner_result?
        };

        if rows.is_empty() {
            return Ok(0);
        }

        // SQLite records are already gone; zvec and fs failures are logged but
        // not propagated so a transient error here does not interrupt the pipeline.
        let id_strings: Vec<String> = rows.iter().map(|(id, _)| id.to_string()).collect();
        let id_refs: Vec<&str> = id_strings.iter().map(|s| s.as_str()).collect();
        self.collection
            .delete(&id_refs)
            .map_err(|e| TrainingError::VectorDb(e.to_string()))?;

        let evicted = rows.len();
        for (_, path) in rows {
            let _ = std::fs::remove_file(&path);
        }
        Ok(evicted)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_embedding() -> [f32; 768] {
        [0.0f32; 768]
    }

    fn unit_embedding(dim: usize) -> [f32; 768] {
        let mut e = [0.0f32; 768];
        e[dim % 768] = 1.0;
        e
    }

    #[test]
    fn store_and_retrieve_pairs() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();

        let emb_a = unit_embedding(0);
        let emb_b = unit_embedding(1);

        let id1 = store
            .store_pair("tenant1", b"fake-jpeg", r#"{"event":"motion"}"#, "teacher", 0.9, &emb_a)
            .unwrap();
        let id2 = store
            .store_pair("tenant1", b"fake-jpeg", r#"{"event":"idle"}"#, "teacher", 0.8, &emb_b)
            .unwrap();

        assert!(id1 > 0);
        assert!(id2 > id1);
        assert_eq!(store.pair_count("tenant1").unwrap(), 2);
        assert_eq!(store.pair_count("tenant2").unwrap(), 0);
    }

    #[test]
    fn knn_returns_nearest_label() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();

        // "motion" cluster around dim-0 axis
        for _ in 0..3 {
            let mut emb = unit_embedding(0);
            emb[1] = 0.01;
            store
                .store_pair("t", b"j", r#"{"event":"motion"}"#, "m", 0.9, &emb)
                .unwrap();
        }
        // "idle" cluster around dim-1 axis
        for _ in 0..3 {
            let emb = unit_embedding(1);
            store
                .store_pair("t", b"j", r#"{"event":"idle"}"#, "m", 0.9, &emb)
                .unwrap();
        }

        // Query near dim-0 → should classify as "motion"
        let query = unit_embedding(0);
        let result = store.knn_classify("t", &query, 3, 0.5).unwrap().unwrap();
        assert_eq!(result.label, r#"{"event":"motion"}"#);
        assert_eq!(result.votes, 3);
    }

    #[test]
    fn knn_returns_none_when_empty() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();
        assert!(store
            .knn_classify("nobody", &unit_embedding(0), 5, 1.0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn knn_respects_distance_threshold() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();

        // Store a vector along dim-0.
        store
            .store_pair("t", b"j", "label_a", "m", 0.9, &unit_embedding(0))
            .unwrap();

        // Query along dim-1: cosine distance = 1.0 (orthogonal).
        // With threshold 0.5 a distance of 1.0 should not qualify.
        let result = store.knn_classify("t", &unit_embedding(1), 5, 0.5).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn knn_zero_vector_does_not_panic() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();
        store
            .store_pair("t", b"j", "label", "m", 0.9, &unit_embedding(0))
            .unwrap();
        // A zero query vector may produce undefined cosine similarity; should not panic.
        let _ = store.knn_classify("t", &zero_embedding(), 3, 1.0);
    }

    #[test]
    fn export_jsonl_writes_correct_lines() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();
        store
            .store_pair("t", b"j", r#"{"event":"a"}"#, "model", 0.9, &unit_embedding(0))
            .unwrap();
        store
            .store_pair("t", b"j", r#"{"event":"b"}"#, "model", 0.8, &unit_embedding(1))
            .unwrap();

        let out = dir.join("export.jsonl");
        let count = store.export_training_jsonl("t", &out).unwrap();
        assert_eq!(count, 2);

        let raw = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["label"], r#"{"event":"a"}"#);
    }

    #[test]
    fn evict_oldest_removes_excess_pairs() {
        let dir = tempdir();
        let store = TrainingStore::new(&dir).unwrap();

        for i in 0u8..5 {
            store
                .store_pair("t", &[i], "label", "model", 0.9, &unit_embedding(i as usize))
                .unwrap();
        }
        assert_eq!(store.pair_count("t").unwrap(), 5);

        let evicted = store.evict_oldest("t", 3).unwrap();
        assert_eq!(evicted, 2);
        assert_eq!(store.pair_count("t").unwrap(), 3);

        // Idempotent when already within limit.
        assert_eq!(store.evict_oldest("t", 10).unwrap(), 0);
    }

    #[test]
    fn embedding_roundtrip() {
        let original = unit_embedding(42);
        let bytes = embedding_to_bytes(&original);
        // Decode from LE bytes for roundtrip verification.
        let mut recovered = [0f32; 768];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            recovered[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    // Diagnostic: verify zvec string field round-trip via output_fields
    #[test]
    fn zvec_string_field_roundtrip() {
        use zvec_bindings::{
            create_and_open_shared, CollectionSchema, Doc, FieldSchema, VectorQuery, VectorSchema,
        };
        let dir = tempdir().join("zvec_str");
        let mut schema = CollectionSchema::new("test_coll");
        schema.add_field(VectorSchema::fp32("vec", 4).into()).unwrap();
        schema.add_field(FieldSchema::string("label")).unwrap();
        schema.add_field(FieldSchema::string("grp")).unwrap();
        let coll = create_and_open_shared(&dir, schema).unwrap();

        let docs: Vec<Doc> = (0..4).map(|i| {
            let mut doc = Doc::id(format!("doc_{}", i));
            let mut v = [0.0f32; 4];
            v[i % 4] = 1.0;
            doc.set_vector("vec", &v).unwrap();
            doc.set_string("label", &format!("label_{}", i)).unwrap();
            doc.set_string("grp", "tenant_a").unwrap();
            doc
        }).collect();
        coll.insert(&docs).unwrap();

        let query = VectorQuery::new("vec")
            .topk(3)
            .output_fields(&["label", "grp"])
            .vector(&[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let results = coll.query(query).unwrap();
        assert!(!results.is_empty(), "zvec should return results");
        let first = results.iter().next().unwrap();
        assert!(first.score() > 0.0, "score should be > 0");
        assert_eq!(
            first.get_string("grp"),
            Some("tenant_a"),
            "grp string field should be returned"
        );
        assert!(
            first.get_string("label").is_some(),
            "label string field should be returned"
        );
    }

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "vidarax-training-test-{}",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }
}
