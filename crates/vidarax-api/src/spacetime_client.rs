//! HTTP client for the SpacetimeDB module deployed at `http://<host>:3000`.
//!
//! Provides both async (tokio/reqwest) and sync (reqwest::blocking) methods
//! that mirror the two reducers and two public tables defined in `spacetime-module/src/lib.rs`.
//!
//! # Endpoint mapping
//!
//! | Operation         | HTTP                                              |
//! |-------------------|---------------------------------------------------|
//! | emit_event        | `POST /v1/database/{db}/call/emit_event`          |
//! | store_keyframe    | `POST /v1/database/{db}/call/store_keyframe`      |
//! | query agent_event | `POST /v1/database/{db}/sql` (plain-text SQL)     |
//! | query keyframe_store | `POST /v1/database/{db}/sql` (plain-text SQL)  |
//!
//! # Identity persistence
//!
//! SpacetimeDB issues a JWT in the `spacetime-identity-token` response header.
//! The client captures this token on the first call and sends it as
//! `Authorization: Bearer <token>` on subsequent calls so all reducer invocations
//! share the same persistent identity.

use std::sync::Arc;

use arc_swap::ArcSwapOption;
use serde_json::Value;

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SpacetimeError {
    Http(String),
    Parse(String),
    BadResponse(u16, String),
}

impl std::fmt::Display for SpacetimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "spacetimedb http: {msg}"),
            Self::Parse(msg) => write!(f, "spacetimedb parse: {msg}"),
            Self::BadResponse(code, body) => {
                write!(f, "spacetimedb bad response {code}: {body}")
            }
        }
    }
}

impl std::error::Error for SpacetimeError {}

// ─── Request types ────────────────────────────────────────────────────────────

/// Arguments for the `emit_event` reducer.
#[derive(Debug, Clone)]
pub struct EmitEventRequest {
    pub run_id: String,
    pub session_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    /// One of: "scene_cut" | "loop_detected" | "vlm" | "goal_reached" | "artifact"
    pub event_type: String,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
    pub description: String,
}

/// Arguments for the `submit_feedback` reducer.
#[derive(Debug, Clone)]
pub struct SubmitFeedbackRequest {
    pub run_id: String,
    pub session_id: String,
    pub rating: u32,
    pub category: String,
    pub feedback: String,
}

/// Arguments for the `store_keyframe` reducer.
#[derive(Debug, Clone)]
pub struct StoreKeyframeRequest {
    pub run_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub event_type: String,
    pub description: String,
    /// Base64-encoded JPEG bytes.
    pub jpeg_data: Vec<u8>,
}

// ─── Response types ───────────────────────────────────────────────────────────

/// Row from the `agent_event` table.
#[derive(Debug, Clone)]
pub struct AgentEvent {
    pub id: u64,
    /// Hex-encoded SpacetimeDB Identity.
    pub agent_id: String,
    pub run_id: String,
    pub session_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub event_type: String,
    pub confidence: f32,
    pub description: String,
    /// Wall-clock time in microseconds since the Unix epoch.
    pub timestamp_micros: i64,
}

/// Row from the `feedback` table.
#[derive(Debug, Clone)]
pub struct FeedbackRow {
    pub id: u64,
    pub agent_id: String,
    pub run_id: String,
    pub session_id: String,
    pub rating: u64,
    pub category: String,
    pub feedback: String,
    pub timestamp_micros: i64,
}

/// Row from the `keyframe_store` table.
#[derive(Debug, Clone)]
pub struct StoredKeyframe {
    pub id: u64,
    pub agent_id: String,
    pub run_id: String,
    pub frame_index: u64,
    pub pts_ms: u64,
    pub event_type: String,
    pub description: String,
    pub jpeg_data: Vec<u8>,
    pub timestamp_micros: i64,
}

// ─── Client ───────────────────────────────────────────────────────────────────

struct Inner {
    base_url: String,
    database: String,
    async_client: reqwest::Client,
    /// JWT issued by SpacetimeDB — persisted across calls for identity continuity.
    token: ArcSwapOption<Arc<str>>,
}

/// Cheap-to-clone (Arc-backed) HTTP client for the Vidarax SpacetimeDB module.
#[derive(Clone)]
pub struct SpacetimeClient {
    inner: Arc<Inner>,
}

impl SpacetimeClient {
    /// Create a new client.
    ///
    /// # Example
    /// ```no_run
    /// use vidarax_api::spacetime_client::SpacetimeClient;
    /// let client = SpacetimeClient::new("http://127.0.0.1:3000", "vidarax");
    /// ```
    pub fn new(base_url: impl Into<String>, database: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                base_url: base_url.into(),
                database: database.into(),
                async_client: reqwest::Client::new(),
                token: ArcSwapOption::from(None::<Arc<Arc<str>>>),
            }),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn reducer_url(&self, name: &str) -> String {
        format!(
            "{}/v1/database/{}/call/{}",
            self.inner.base_url, self.inner.database, name
        )
    }

    fn sql_url(&self) -> String {
        format!(
            "{}/v1/database/{}/sql",
            self.inner.base_url, self.inner.database
        )
    }

    fn read_token(&self) -> Option<Arc<str>> {
        self.inner
            .token
            .load_full()
            .map(|token| Arc::clone(&*token))
    }

    fn store_token_from_headers(&self, headers: &reqwest::header::HeaderMap) {
        if let Some(val) = headers.get("spacetime-identity-token") {
            if let Ok(tok) = val.to_str() {
                self.inner.token.store(Some(Arc::new(Arc::from(tok))));
            }
        }
    }

    fn store_token_from_headers_blocking(&self, headers: &reqwest::header::HeaderMap) {
        self.store_token_from_headers(headers);
    }

    fn auth_header(&self) -> Option<String> {
        self.read_token().map(|t| format!("Bearer {t}"))
    }

    fn blocking_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::new()
    }

    // ── Async methods ────────────────────────────────────────────────────────

    /// Call the `emit_event` reducer (async).
    pub async fn emit_event_async(&self, req: &EmitEventRequest) -> Result<(), SpacetimeError> {
        let body = serde_json::json!([
            req.run_id,
            req.session_id,
            req.frame_index,
            req.pts_ms,
            req.event_type,
            req.confidence,
            req.description,
        ]);
        let mut rb = self
            .inner
            .async_client
            .post(self.reducer_url("emit_event"))
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| SpacetimeError::Http(e.to_string()))?;
        self.store_token_from_headers(resp.headers());
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        Ok(())
    }

    /// Call the `store_keyframe` reducer (async).
    pub async fn store_keyframe_async(
        &self,
        req: &StoreKeyframeRequest,
    ) -> Result<(), SpacetimeError> {
        let body = serde_json::json!([
            req.run_id,
            req.frame_index,
            req.pts_ms,
            req.event_type,
            req.description,
            req.jpeg_data,
        ]);
        let mut rb = self
            .inner
            .async_client
            .post(self.reducer_url("store_keyframe"))
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| SpacetimeError::Http(e.to_string()))?;
        self.store_token_from_headers(resp.headers());
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        Ok(())
    }

    /// Query the `agent_event` table (async).
    ///
    /// If `run_id` is `Some`, only rows for that run are returned.
    pub async fn query_agent_events_async(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<AgentEvent>, SpacetimeError> {
        let sql = build_select("agent_event", run_id);
        let rows = self.sql_async(&sql).await?;
        rows.into_iter().map(parse_agent_event).collect()
    }

    /// Query the `keyframe_store` table (async).
    ///
    /// If `run_id` is `Some`, only rows for that run are returned.
    pub async fn query_keyframes_async(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<StoredKeyframe>, SpacetimeError> {
        let sql = build_select("keyframe_store", run_id);
        let rows = self.sql_async(&sql).await?;
        rows.into_iter().map(parse_keyframe).collect()
    }

    /// Call the `submit_feedback` reducer (async).
    pub async fn submit_feedback_async(
        &self,
        req: &SubmitFeedbackRequest,
    ) -> Result<(), SpacetimeError> {
        let body = serde_json::json!([
            req.run_id,
            req.session_id,
            req.rating,
            req.category,
            req.feedback,
        ]);
        let mut rb = self
            .inner
            .async_client
            .post(self.reducer_url("submit_feedback"))
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| SpacetimeError::Http(e.to_string()))?;
        self.store_token_from_headers(resp.headers());
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        Ok(())
    }

    /// Query the `feedback` table (async).
    ///
    /// If `run_id` is `Some`, only rows for that run are returned.
    pub async fn query_feedback_async(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<FeedbackRow>, SpacetimeError> {
        let sql = build_select("feedback", run_id);
        let rows = self.sql_async(&sql).await?;
        rows.into_iter().map(parse_feedback).collect()
    }

    async fn sql_async(&self, sql: &str) -> Result<Vec<Value>, SpacetimeError> {
        let mut rb = self
            .inner
            .async_client
            .post(self.sql_url())
            .header("Content-Type", "text/plain")
            .body(sql.to_string());
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| SpacetimeError::Http(e.to_string()))?;
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        let json: Value = resp
            .json()
            .await
            .map_err(|e| SpacetimeError::Parse(e.to_string()))?;
        extract_rows(json)
    }

    // ── Sync methods ─────────────────────────────────────────────────────────
    //
    // These use `reqwest::blocking::Client` which manages its own internal
    // tokio runtime. Do NOT call these from within an async context; use the
    // `_async` variants there.

    /// Call the `emit_event` reducer (sync).
    pub fn emit_event(&self, req: &EmitEventRequest) -> Result<(), SpacetimeError> {
        let body = serde_json::json!([
            req.run_id,
            req.session_id,
            req.frame_index,
            req.pts_ms,
            req.event_type,
            req.confidence,
            req.description,
        ]);
        let client = Self::blocking_client();
        let mut rb = client
            .post(self.reducer_url("emit_event"))
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb.send().map_err(|e| SpacetimeError::Http(e.to_string()))?;
        self.store_token_from_headers_blocking(resp.headers());
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        Ok(())
    }

    /// Call the `store_keyframe` reducer (sync).
    pub fn store_keyframe(&self, req: &StoreKeyframeRequest) -> Result<(), SpacetimeError> {
        let body = serde_json::json!([
            req.run_id,
            req.frame_index,
            req.pts_ms,
            req.event_type,
            req.description,
            req.jpeg_data,
        ]);
        let client = Self::blocking_client();
        let mut rb = client
            .post(self.reducer_url("store_keyframe"))
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb.send().map_err(|e| SpacetimeError::Http(e.to_string()))?;
        self.store_token_from_headers_blocking(resp.headers());
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        Ok(())
    }

    /// Query the `agent_event` table (sync).
    ///
    /// If `run_id` is `Some`, only rows for that run are returned.
    pub fn query_agent_events(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<AgentEvent>, SpacetimeError> {
        let sql = build_select("agent_event", run_id);
        let rows = self.sql_sync(&sql)?;
        rows.into_iter().map(parse_agent_event).collect()
    }

    /// Query the `keyframe_store` table (sync).
    ///
    /// If `run_id` is `Some`, only rows for that run are returned.
    pub fn query_keyframes(
        &self,
        run_id: Option<&str>,
    ) -> Result<Vec<StoredKeyframe>, SpacetimeError> {
        let sql = build_select("keyframe_store", run_id);
        let rows = self.sql_sync(&sql)?;
        rows.into_iter().map(parse_keyframe).collect()
    }

    fn sql_sync(&self, sql: &str) -> Result<Vec<Value>, SpacetimeError> {
        let client = Self::blocking_client();
        let mut rb = client
            .post(self.sql_url())
            .header("Content-Type", "text/plain")
            .body(sql.to_string());
        if let Some(auth) = self.auth_header() {
            rb = rb.header("Authorization", auth);
        }
        let resp = rb.send().map_err(|e| SpacetimeError::Http(e.to_string()))?;
        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().unwrap_or_default();
            return Err(SpacetimeError::BadResponse(status, text));
        }
        let json: Value = resp
            .json()
            .map_err(|e| SpacetimeError::Parse(e.to_string()))?;
        extract_rows(json)
    }
}

// ─── SQL helpers ──────────────────────────────────────────────────────────────

/// Build a SELECT statement with an optional `run_id` filter.
///
/// Both `table` and `run_id` are validated to contain only safe characters
/// (`[a-zA-Z0-9_-]`) to prevent SQL injection (C-2).
fn build_select(table: &str, run_id: Option<&str>) -> String {
    fn is_safe_identifier(s: &str) -> bool {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    }

    // Table names are compile-time constants, but validate defensively.
    assert!(is_safe_identifier(table), "invalid table name: {table}");

    match run_id {
        Some(id) => {
            assert!(is_safe_identifier(id), "invalid run_id format");
            format!("SELECT * FROM {table} WHERE run_id = '{id}'")
        }
        None => format!("SELECT * FROM {table}"),
    }
}

// ─── Row extraction ───────────────────────────────────────────────────────────

/// Pull the `rows` array from a SpacetimeDB SQL response.
///
/// Response shape:
/// ```json
/// [{"schema": {...}, "rows": [[val, ...], ...], "stats": {...}}]
/// ```
fn extract_rows(json: Value) -> Result<Vec<Value>, SpacetimeError> {
    let sets = json
        .as_array()
        .ok_or_else(|| SpacetimeError::Parse("expected JSON array of result sets".into()))?;
    if sets.is_empty() {
        return Ok(vec![]);
    }
    let rows = sets[0]
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| SpacetimeError::Parse("missing 'rows' field in result set".into()))?
        .clone();
    Ok(rows)
}

// ─── Column accessors ─────────────────────────────────────────────────────────

fn col_str(row: &Value, idx: usize) -> Result<String, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected string")))
}

/// SpacetimeDB encodes `Vec<u8>` as a JSON array of numbers `[255, 216, ...]`.
fn col_bytes(row: &Value, idx: usize) -> Result<Vec<u8>, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_u64)
                .map(|v| v as u8)
                .collect()
        })
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected byte array")))
}

fn col_u64(row: &Value, idx: usize) -> Result<u64, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_u64)
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected u64")))
}

fn col_f64(row: &Value, idx: usize) -> Result<f64, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_f64)
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected f64")))
}

/// SpacetimeDB encodes `Identity` as a 1-element JSON array `["0xhex..."]`.
fn col_identity(row: &Value, idx: usize) -> Result<String, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected Identity array")))
}

/// SpacetimeDB encodes `Timestamp` as a 1-element JSON array `[i64_micros]`.
fn col_timestamp(row: &Value, idx: usize) -> Result<i64, SpacetimeError> {
    row.get(idx)
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_i64)
        .ok_or_else(|| SpacetimeError::Parse(format!("col {idx}: expected Timestamp array")))
}

// ─── Row parsers ──────────────────────────────────────────────────────────────

/// Column order from `SELECT * FROM agent_event`:
/// id | agent_id | run_id | session_id | frame_index | pts_ms | event_type | confidence | description | timestamp
fn parse_agent_event(row: Value) -> Result<AgentEvent, SpacetimeError> {
    Ok(AgentEvent {
        id: col_u64(&row, 0)?,
        agent_id: col_identity(&row, 1)?,
        run_id: col_str(&row, 2)?,
        session_id: col_str(&row, 3)?,
        frame_index: col_u64(&row, 4)?,
        pts_ms: col_u64(&row, 5)?,
        event_type: col_str(&row, 6)?,
        confidence: col_f64(&row, 7)? as f32,
        description: col_str(&row, 8)?,
        timestamp_micros: col_timestamp(&row, 9)?,
    })
}

// ─── EventSink impl ───────────────────────────────────────────────────────────

/// Bridges `SpacetimeClient` into the `vidarax_core` worker pool interface.
///
/// Worker threads hold `Arc<dyn EventSink>` and call these sync methods from
/// non-async contexts.
impl vidarax_core::webrtc::workers::EventSink for SpacetimeClient {
    fn emit_event_sync(
        &self,
        run_id: &str,
        session_id: &str,
        frame_index: u64,
        pts_ms: u64,
        event_type: &str,
        confidence: f32,
        description: &str,
    ) -> Result<(), String> {
        self.emit_event(&EmitEventRequest {
            run_id: run_id.to_string(),
            session_id: session_id.to_string(),
            frame_index,
            pts_ms,
            event_type: event_type.to_string(),
            confidence,
            description: description.to_string(),
        })
        .map_err(|e| e.to_string())
    }

    fn store_keyframe_sync(
        &self,
        run_id: &str,
        frame_index: u64,
        pts_ms: u64,
        event_type: &str,
        description: &str,
        jpeg_data: &[u8],
    ) -> Result<(), String> {
        self.store_keyframe(&StoreKeyframeRequest {
            run_id: run_id.to_string(),
            frame_index,
            pts_ms,
            event_type: event_type.to_string(),
            description: description.to_string(),
            jpeg_data: jpeg_data.to_vec(),
        })
        .map_err(|e| e.to_string())
    }
}

/// Column order from `SELECT * FROM keyframe_store`:
/// id | agent_id | run_id | frame_index | pts_ms | event_type | description | jpeg_b_64 | timestamp
fn parse_keyframe(row: Value) -> Result<StoredKeyframe, SpacetimeError> {
    Ok(StoredKeyframe {
        id: col_u64(&row, 0)?,
        agent_id: col_identity(&row, 1)?,
        run_id: col_str(&row, 2)?,
        frame_index: col_u64(&row, 3)?,
        pts_ms: col_u64(&row, 4)?,
        event_type: col_str(&row, 5)?,
        description: col_str(&row, 6)?,
        jpeg_data: col_bytes(&row, 7)?,
        timestamp_micros: col_timestamp(&row, 8)?,
    })
}

/// Column order from `SELECT * FROM feedback`:
/// id | agent_id | run_id | session_id | rating | category | feedback | timestamp
fn parse_feedback(row: Value) -> Result<FeedbackRow, SpacetimeError> {
    Ok(FeedbackRow {
        id: col_u64(&row, 0)?,
        agent_id: col_identity(&row, 1)?,
        run_id: col_str(&row, 2)?,
        session_id: col_str(&row, 3)?,
        rating: col_u64(&row, 4)?,
        category: col_str(&row, 5)?,
        feedback: col_str(&row, 6)?,
        timestamp_micros: col_timestamp(&row, 7)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    #[test]
    fn token_reader_observes_latest_stored_header_value() {
        let client = SpacetimeClient::new("http://localhost:3000", "vidarax");
        let mut headers = HeaderMap::new();
        headers.insert(
            "spacetime-identity-token",
            HeaderValue::from_static("tok-a"),
        );
        client.store_token_from_headers(&headers);
        assert_eq!(client.auth_header().as_deref(), Some("Bearer tok-a"));

        headers.insert(
            "spacetime-identity-token",
            HeaderValue::from_static("tok-b"),
        );
        client.store_token_from_headers(&headers);
        assert_eq!(client.auth_header().as_deref(), Some("Bearer tok-b"));
    }
}
