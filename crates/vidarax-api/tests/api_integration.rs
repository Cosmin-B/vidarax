//! Integration tests for feedback, WHIP prompt-update, models catalogue,
//! and related API surfaces.
//!
//! Uses `tower::ServiceExt::oneshot` to drive the router directly without
//! binding a real TCP socket, so tests start fast and run in parallel.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tower::ServiceExt;
use vidarax_api::{AppState, app_router};

// ─── Helpers ─────────────────────────────────────────────────────────────────

static WAL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Return a unique temp-file path for a WAL so parallel tests don't collide.
fn tmp_wal(tag: &str) -> PathBuf {
    let n = WAL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vidarax-integ-{tag}-{pid}-{n}.wal"))
}

/// Consume a response body and parse it as JSON.
async fn collect_json(body: Body) -> Value {
    let bytes = body
        .collect()
        .await
        .expect("body collect failed")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("body is not valid JSON")
}

/// Build a POST request with a JSON body.
fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Build a PATCH request with a JSON body.
fn patch_json(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("PATCH")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Build a GET request.
fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::from(String::new()))
        .unwrap()
}

// ─── Constants ───────────────────────────────────────────────────────────────

/// A syntactically valid run-id ("run-" + 16 hex chars = 20 chars total).
const TEST_RUN_ID: &str = "run-0000000000000001";

// ─── Feedback validation ─────────────────────────────────────────────────────

#[tokio::test]
async fn feedback_rating_above_10_returns_unprocessable_entity() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("fb-rating")));

    let res = router
        .oneshot(post_json(
            &format!("/v1/runs/{TEST_RUN_ID}/feedback"),
            json!({ "rating": 11, "category": "accuracy" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = collect_json(res.into_body()).await;
    assert_eq!(body["error"]["code"].as_str().unwrap_or(""), "validation_error");
    // The field-level detail should mention "rating".
    let details = body["error"]["details"].as_array().unwrap();
    assert!(
        details.iter().any(|d| d["field"].as_str() == Some("rating")),
        "expected a 'rating' field error, got: {details:?}"
    );
}

#[tokio::test]
async fn feedback_empty_category_returns_unprocessable_entity() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("fb-category")));

    let res = router
        .oneshot(post_json(
            &format!("/v1/runs/{TEST_RUN_ID}/feedback"),
            json!({ "rating": 5, "category": "" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = collect_json(res.into_body()).await;
    assert_eq!(body["error"]["code"].as_str().unwrap_or(""), "validation_error");
    let details = body["error"]["details"].as_array().unwrap();
    assert!(
        details.iter().any(|d| d["field"].as_str() == Some("category")),
        "expected a 'category' field error, got: {details:?}"
    );
}

#[tokio::test]
async fn feedback_without_spacetimedb_returns_internal_error() {
    // AppState::with_wal_for_tests has no SpacetimeDB client attached.
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("fb-stdb")));

    let res = router
        .oneshot(post_json(
            &format!("/v1/runs/{TEST_RUN_ID}/feedback"),
            json!({ "rating": 7, "category": "quality" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = collect_json(res.into_body()).await;
    assert_eq!(body["error"]["code"].as_str().unwrap_or(""), "internal_error");
}

#[tokio::test]
async fn feedback_malformed_run_id_returns_unprocessable_entity() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("fb-runid")));

    let res = router
        .oneshot(post_json(
            "/v1/runs/not-a-valid-run-id/feedback",
            json!({ "rating": 5, "category": "quality" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ─── WHIP prompt update ───────────────────────────────────────────────────────

#[tokio::test]
async fn whip_update_prompt_unknown_session_returns_not_found() {
    // No session has been inserted into the store; handler returns 404.
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("whip-prompt")));

    let res = router
        .oneshot(patch_json(
            "/v1/stream/whip/sess-aaaa0000bbbb1111/prompt",
            json!({ "prompt": "describe what you see" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn whip_update_prompt_requires_json_body() {
    // Sending a non-JSON body to the prompt endpoint should yield a client error.
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("whip-prompt-json")));

    let req = Request::builder()
        .method("PATCH")
        .uri("/v1/stream/whip/sess-aaaa0000bbbb1111/prompt")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from("not json"))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    // axum rejects non-JSON content before hitting the handler
    assert!(
        res.status().is_client_error(),
        "expected a 4xx for non-JSON body, got {}",
        res.status()
    );
}

// ─── Models catalogue ────────────────────────────────────────────────────────

#[tokio::test]
async fn models_returns_200_with_non_empty_catalog() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("models-catalog")));

    let res = router.oneshot(get("/v1/models")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let body = collect_json(res.into_body()).await;
    let models = body["models"].as_array().expect("'models' should be an array");
    assert!(!models.is_empty(), "expected at least one model in catalog");
}

#[tokio::test]
async fn models_availability_is_unavailable_without_inference_endpoints() {
    // AppState::with_wal_for_tests configures no inference provider URLs, so
    // every model should report availability = "unavailable".
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("models-avail")));

    let res = router.oneshot(get("/v1/models")).await.unwrap();
    let body = collect_json(res.into_body()).await;
    let models = body["models"].as_array().unwrap();
    for model in models {
        assert_eq!(
            model["availability"].as_str().unwrap_or(""),
            "unavailable",
            "expected 'unavailable' when no inference endpoints are configured; model={model}"
        );
    }
}

#[tokio::test]
async fn models_catalog_items_have_required_fields() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("models-fields")));

    let res = router.oneshot(get("/v1/models")).await.unwrap();
    let body = collect_json(res.into_body()).await;
    let models = body["models"].as_array().unwrap();
    // Validate all items, not just the first.
    for item in models {
        assert!(item["id"].is_string(), "model must have string 'id'");
        assert!(item["tier"].is_string(), "model must have string 'tier'");
        assert!(
            item["availability"].is_string(),
            "model must have string 'availability'"
        );
        assert!(
            item["providers_available"].is_array(),
            "model must have array 'providers_available'"
        );
        assert!(
            item["fallback_candidates"].is_array(),
            "model must have array 'fallback_candidates'"
        );
    }
}

#[tokio::test]
async fn models_response_contains_request_id() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("models-reqid")));

    let res = router.oneshot(get("/v1/models")).await.unwrap();
    let body = collect_json(res.into_body()).await;
    assert!(
        body["request_id"].is_string(),
        "response should include 'request_id'"
    );
}

// ─── Semantic search ──────────────────────────────────────────────────────────

#[tokio::test]
async fn search_returns_200_with_empty_hits_when_wal_is_empty() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("search-empty")));

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "person walking" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = collect_json(res.into_body()).await;
    assert!(body["request_id"].is_string());
    assert_eq!(body["scanned"].as_u64().unwrap_or(1), 0);
    assert_eq!(body["total_hits"].as_u64().unwrap_or(1), 0);
    assert!(body["hits"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn search_rejects_empty_query() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("search-empty-q")));

    let res = router
        .oneshot(post_json("/v1/search", json!({ "query": "" })))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_rejects_zero_limit() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("search-zero-limit")));

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "test", "limit": 0 }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_rejects_limit_over_500() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("search-big-limit")));

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "test", "limit": 501 }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_with_run_id_rejects_invalid_run_id() {
    let router = app_router(AppState::with_wal_for_tests(tmp_wal("search-bad-run")));

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "person", "run_id": "not-a-valid-run-id" }),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_finds_matching_descriptions_in_wal() {
    // Seed the WAL with a timeline event that contains a description field.
    use vidarax_core::timeline::{TimelineEvent, append_event};

    let wal = tmp_wal("search-match");
    let event = TimelineEvent {
        seq: 1,
        run_id: "run-aabbccddeeff0011".to_string(),
        stream_id: "stream-0".to_string(),
        pts_ms: 1000,
        kind: "semantic_chunk_inferred".to_string(),
        payload: serde_json::json!({
            "description": "a person walking through a doorway",
            "event_type": "scene_cut",
            "chunk_index": 0
        })
        .to_string(),
    };
    append_event(&wal, &event).unwrap();

    let state = AppState::with_wal_for_tests(wal);
    // Register the run so it is known.
    state
        .append_run_event(
            "run-aabbccddeeff0011",
            "run_created",
            serde_json::json!({ "principal_key": "public" }),
        )
        .unwrap();

    let router = app_router(state);

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "person walking" }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = collect_json(res.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "expected exactly one hit: {body}");
    assert_eq!(
        hits[0]["description"].as_str().unwrap(),
        "a person walking through a doorway"
    );
    assert_eq!(hits[0]["kind"].as_str().unwrap(), "semantic_chunk_inferred");
}

#[tokio::test]
async fn search_is_case_insensitive() {
    use vidarax_core::timeline::{TimelineEvent, append_event};

    let wal = tmp_wal("search-case");
    let event = TimelineEvent {
        seq: 1,
        run_id: "run-aabbccddeeff0022".to_string(),
        stream_id: "stream-0".to_string(),
        pts_ms: 500,
        kind: "semantic_chunk_inferred".to_string(),
        payload: serde_json::json!({ "description": "FORKLIFT moving pallets" }).to_string(),
    };
    append_event(&wal, &event).unwrap();

    let router = app_router(AppState::with_wal_for_tests(wal));

    let res = router
        .oneshot(post_json("/v1/search", json!({ "query": "forklift" })))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = collect_json(res.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "case-insensitive match should return a hit");
}

#[tokio::test]
async fn search_respects_limit() {
    use vidarax_core::timeline::{TimelineEvent, append_event};

    let wal = tmp_wal("search-limit");
    for i in 0u64..10 {
        let event = TimelineEvent {
            seq: i + 1,
            run_id: "run-aabbccddeeff0033".to_string(),
            stream_id: "stream-0".to_string(),
            pts_ms: i * 100,
            kind: "semantic_chunk_inferred".to_string(),
            payload: serde_json::json!({ "description": format!("car on road frame {i}") })
                .to_string(),
        };
        append_event(&wal, &event).unwrap();
    }

    let router = app_router(AppState::with_wal_for_tests(wal));

    let res = router
        .oneshot(post_json(
            "/v1/search",
            json!({ "query": "car on road", "limit": 3 }),
        ))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = collect_json(res.into_body()).await;
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3, "limit should cap results at 3");
}

#[tokio::test]
async fn search_no_match_returns_empty_hits() {
    use vidarax_core::timeline::{TimelineEvent, append_event};

    let wal = tmp_wal("search-no-match");
    let event = TimelineEvent {
        seq: 1,
        run_id: "run-aabbccddeeff0044".to_string(),
        stream_id: "stream-0".to_string(),
        pts_ms: 0,
        kind: "semantic_chunk_inferred".to_string(),
        payload: serde_json::json!({ "description": "empty parking lot at night" }).to_string(),
    };
    append_event(&wal, &event).unwrap();

    let router = app_router(AppState::with_wal_for_tests(wal));

    let res = router
        .oneshot(post_json("/v1/search", json!({ "query": "bicycle" })))
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = collect_json(res.into_body()).await;
    assert_eq!(
        body["total_hits"].as_u64().unwrap_or(1),
        0,
        "no match should return zero hits"
    );
}
