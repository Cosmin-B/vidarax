use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, patch, post};
use axum::Router;

use crate::handlers::{
    analyze_run, create_run, get_events, get_markers, get_state, health, infer, infer_batch,
    ingest_run, keepalive_run, list_models, metrics, query, reason_realtime_run, stop_run,
};
use crate::security::enforce_security;
use crate::state::AppState;
use crate::whip::{whip_ice, whip_offer, whip_terminate};

pub fn app_router(state: AppState) -> Router {
    let middleware_state = state.clone();
    Router::new()
        // Existing run/analysis routes
        .route("/v1/runs", post(create_run))
        .route("/v1/runs/{run_id}/ingest", post(ingest_run))
        .route("/v1/runs/{run_id}/analyze", post(analyze_run))
        .route("/v1/runs/{run_id}/reason", post(reason_realtime_run))
        .route("/v1/runs/{run_id}/stop", post(stop_run))
        .route("/v1/runs/{run_id}/keepalive", post(keepalive_run))
        .route("/v1/runs/{run_id}/events", get(get_events))
        .route("/v1/runs/{run_id}/markers", get(get_markers))
        .route("/v1/runs/{run_id}/state", get(get_state))
        .route("/v1/query", post(query))
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
        .with_state(state)
        .layer(DefaultBodyLimit::max(4 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            middleware_state,
            enforce_security,
        ))
}
