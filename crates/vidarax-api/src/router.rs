use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, patch, post};
use axum::Router;
use tower_http::compression::CompressionLayer;

use crate::handlers::{
    analyze_run, create_run, delete_run, get_events, get_markers, get_run, get_state, health,
    infer, infer_batch, ingest_run, keepalive_run, list_feedback, list_models, list_runs, metrics,
    query, reason_realtime_run, search, stop_run, submit_feedback, upload_file,
};
use crate::security::enforce_security;
use crate::state::AppState;
use crate::whip::{whip_ice, whip_offer, whip_terminate, whip_update_prompt};

pub fn app_router(state: AppState) -> Router {
    let middleware_state = state.clone();
    Router::new()
        // Existing run/analysis routes
        .route("/v1/runs", get(list_runs).post(create_run))
        .route("/v1/runs/{run_id}", get(get_run).delete(delete_run))
        .route(
            "/v1/upload",
            post(upload_file).layer(DefaultBodyLimit::max(200 * 1024 * 1024)),
        )
        .route("/v1/runs/{run_id}/ingest", post(ingest_run))
        .route("/v1/runs/{run_id}/analyze", post(analyze_run))
        .route("/v1/runs/{run_id}/reason", post(reason_realtime_run))
        .route("/v1/runs/{run_id}/stop", post(stop_run))
        .route("/v1/runs/{run_id}/keepalive", post(keepalive_run))
        .route("/v1/runs/{run_id}/events", get(get_events))
        .route("/v1/runs/{run_id}/markers", get(get_markers))
        .route("/v1/runs/{run_id}/state", get(get_state))
        .route("/v1/runs/{run_id}/feedback", post(submit_feedback))
        .route("/v1/feedback", get(list_feedback))
        .route("/v1/query", post(query))
        .route("/v1/search", post(search))
        .route("/v1/infer", post(infer))
        .route("/v1/infer/batch", post(infer_batch))
        .route("/v1/models", get(list_models))
        .route("/v1/health", get(health))
        .route("/v1/metrics", get(metrics))
        // WHIP WebRTC ingestion (RFC 9725)
        .route("/v1/stream/whip", post(whip_offer))
        .route(
            "/v1/stream/whip/{sess_id}",
            patch(whip_ice).delete(whip_terminate),
        )
        .route(
            "/v1/stream/whip/{sess_id}/prompt",
            patch(whip_update_prompt),
        )
        .with_state(state)
        .layer(CompressionLayer::new())
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            middleware_state,
            enforce_security,
        ))
}
