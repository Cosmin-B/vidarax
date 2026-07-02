use axum::extract::{Multipart, Path, Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};
use std::cmp::min;
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path as FsPath, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::task::JoinSet;
use vidarax_contracts::models::{
    fallback_candidates, REQUIRED_MEDIUM_MODELS, REQUIRED_SMALL_MODELS,
};
use vidarax_core::gate::{FrameSignal, GateConfig};
use vidarax_core::ingest::{
    compute_semantic_frame_indices, probe_source_fps, DecodedJpegFrame, InputSource,
    Mp4DecodeConfig,
};
use vidarax_core::pipeline::{TwoPassConfig, TwoPassPipeline};
use vidarax_core::provider::{InferenceProvider, InferenceRequest, ProviderError, ProviderKind};
use vidarax_core::timeline::TimelineEvent;

use crate::auth::{header_value, strong_hash_hex, HEADER_TENANT_ID};
use crate::config::UPLOAD_DIR_NAME;
use crate::ids::validate_run_id;
use crate::models::{
    AnalyzeFrameMetadata, AnalyzeFramesRequest, AnalyzeFramesResponse, AnalyzeMarker,
    CreateRunRequest, CreateRunResponse, FieldError, InferBatchItemError, InferBatchItemResult,
    InferBatchRequest, InferBatchResponse, InferRequest, InferResponse, ModelCatalogItem,
    ModelCatalogResponse, RealtimeReasonRequest, RealtimeReasonResponse, SamplingPolicy, SearchHit,
    SearchRequest, SearchResponse,
};
use crate::response::{
    conflict_error, internal_error, not_found_error, ok, validation_error, ApiResponse,
};
use crate::semantic::{build_marker_lifecycle, MarkerConfig};
use crate::semantic_infer::{
    adaptive_sample_fps, compose_frame_metadata, estimate_sample_fps,
    load_decoded_signals_from_events, percentile_ms, prepare_realtime_chunks,
    run_semantic_dispatch, semantic_marker_to_api_marker, ChunkPrep, ChunkSemanticResult,
};
use crate::state::AppState;

const UPLOAD_MEDIA_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
use crate::validation::{normalize_mode, normalize_model};

#[derive(Debug, serde::Deserialize)]
pub struct MarkerQueryParams {
    pub status: Option<String>,
    pub event_type: Option<String>,
    pub from_frame: Option<u64>,
    pub to_frame: Option<u64>,
}

/// Query parameters accepted by `GET /v1/runs/{run_id}/events`.
#[derive(Debug, Default, serde::Deserialize)]
pub struct EventsQueryParams {
    /// When set, only events whose payload contains `"index_name": "<value>"`
    /// are returned.  Supports multiple analysis passes on the same run.
    pub index: Option<String>,
}

#[tracing::instrument(name = "api.create_run", skip_all)]
pub async fn create_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<CreateRunRequest>,
) -> impl IntoResponse {
    let mode = match normalize_mode(payload.mode) {
        Ok(mode) => mode,
        Err(message) => {
            return validation_error(
                &state,
                "invalid create-run payload",
                vec![field_error("mode", message)],
            );
        }
    };

    let model = match normalize_model(payload.model) {
        Ok(model) => model,
        Err(message) => {
            return validation_error(
                &state,
                "invalid create-run payload",
                vec![field_error("model", message)],
            );
        }
    };

    let tenant_id = header_value(&headers, HEADER_TENANT_ID).map(ToString::to_string);
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let _slot = match state.try_reserve_stream_slot(&principal, now_epoch_ms()) {
        Some(slot) => slot,
        None => {
            return conflict_error(
                &state,
                "active stream limit exceeded",
                vec![field_error(
                    "run_id",
                    format!(
                        "principal exceeded active stream limit: {}/{}",
                        state.active_stream_limit(),
                        state.active_stream_limit()
                    ),
                )],
            )
        }
    };

    let run_id = state.next_run_id();
    let request_id = state.next_request_id();
    let payload = json!({
        "request_id": request_id,
        "mode": mode,
        "model": model,
        "principal_key": principal,
        "tenant_id": tenant_id
    });
    if let Err(err) = state
        .append_run_event_async(&run_id, "run_created", payload)
        .await
    {
        return internal_error(&state, format!("failed to append run_created event: {err}"));
    }

    ok(json!(CreateRunResponse {
        run_id,
        request_id,
        status: "pending",
        mode,
        model,
    }))
}

#[tracing::instrument(name = "api.ingest_run", skip_all, fields(run_id))]
pub async fn ingest_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid ingest request") {
        return error;
    }

    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    if run_snapshot.state.is_terminal() {
        return conflict_error(
            &state,
            "cannot ingest into terminal run",
            vec![field_error(
                "run_id",
                format!("run is in terminal state: {:?}", run_snapshot.state),
            )],
        );
    }

    let request_id = state.next_request_id();
    let source_uri = payload
        .get("source_uri")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);

    if let Some(source_uri) = source_uri {
        let sampling_policy = match SamplingPolicy::parse(
            payload
                .get("sampling_policy")
                .and_then(|value| value.as_str()),
        ) {
            Ok(policy) => policy,
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid ingest request",
                    vec![field_error("sampling_policy", message.to_string())],
                );
            }
        };
        let fixed_fps = payload.get("fixed_fps").and_then(|value| value.as_f64());
        let sample_fps = payload
            .get("sample_fps")
            .and_then(|value| value.as_f64())
            .or(fixed_fps);
        if sampling_policy == SamplingPolicy::Fixed {
            let Some(sample_fps) = sample_fps else {
                return validation_error(
                    &state,
                    "invalid ingest request",
                    vec![field_error(
                        "fixed_fps",
                        "fixed_fps (or sample_fps) is required when sampling_policy=fixed"
                            .to_string(),
                    )],
                );
            };
            if !(0.2..=120.0).contains(&sample_fps) {
                return validation_error(
                    &state,
                    "invalid ingest request",
                    vec![field_error(
                        "fixed_fps",
                        "fixed_fps must be in [0.2, 120.0]".to_string(),
                    )],
                );
            }
        }
        let max_frames = payload
            .get("max_frames")
            .and_then(|value| value.as_u64())
            .unwrap_or(512);
        if !(1..=500_000).contains(&max_frames) {
            return validation_error(
                &state,
                "invalid ingest request",
                vec![field_error(
                    "max_frames",
                    "max_frames must be in [1, 500000]".to_string(),
                )],
            );
        }
        let stream_id = payload
            .get("stream_id")
            .and_then(|value| value.as_str())
            .unwrap_or("stream-0")
            .to_string();
        let allowed_roots = ingest_file_roots_with_upload_root(&state);
        let decode_source = match InputSource::parse_and_validate(&source_uri, &allowed_roots) {
            Ok(source) => source,
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid ingest request",
                    vec![field_error("source_uri", message)],
                );
            }
        };
        if let Err(error) = enforce_file_source_visibility(
            &state,
            &headers,
            &source_uri,
            &decode_source,
            "invalid ingest request",
        ) {
            return error;
        }
        let requested_sample_fps = sample_fps.unwrap_or(2.0);
        let decode_pipeline = state.decode_pipeline();
        let decoded = match tokio::task::spawn_blocking(move || {
            let source_fps = probe_source_fps(&decode_source);
            let effective_sample_fps = match sampling_policy {
                SamplingPolicy::SourceFpsAdaptive => source_fps
                    .map(adaptive_sample_fps)
                    .unwrap_or(requested_sample_fps as f32),
                SamplingPolicy::Fixed => requested_sample_fps as f32,
            };
            let decode_config = Mp4DecodeConfig {
                sample_fps: effective_sample_fps,
                max_frames: max_frames as usize,
            };
            decode_pipeline
                .decode_signals(&decode_source, decode_config)
                .map(|decoded| (decoded, source_fps, effective_sample_fps))
        })
        .await
        {
            Ok(Ok(decoded)) => decoded,
            Ok(Err(err)) => {
                return validation_error(
                    &state,
                    "invalid ingest request",
                    vec![field_error("source_uri", err)],
                );
            }
            Err(err) => {
                return internal_error(&state, format!("ingest decode worker join failure: {err}"));
            }
        };
        let (decoded, source_fps, effective_sample_fps) = decoded;

        if let Err(err) = state
            .append_run_event_async(
                &run_id,
                "ingest_received",
                json!({
                    "request_id": request_id,
                    "ingest": payload,
                    "decoded_frames": decoded.frame_signals.len(),
                    "source_uri": decoded.source_uri,
                    "sampling_policy": sampling_policy.as_str(),
                    "sample_fps": effective_sample_fps
                }),
            )
            .await
        {
            return internal_error(&state, format!("failed to append ingest event: {err}"));
        }

        let signals = decoded
            .frame_signals
            .iter()
            .map(|signal| {
                json!({
                    "frame_index": signal.frame_index,
                    "pts_ms": signal.pts_ms,
                    "perceptual_hash": signal.perceptual_hash,
                    "luma_mean": signal.luma_mean,
                    "flicker_score": signal.flicker_score,
                    "ghosting_score": signal.ghosting_score,
                    "noise_variance_score": signal.noise_variance_score
                })
            })
            .collect::<Vec<_>>();
        if let Err(err) = state
            .append_run_event_async(
                &run_id,
                "frames_decoded",
                json!({
                    "request_id": request_id,
                    "source_uri": decoded.source_uri,
                    "stream_id": stream_id,
                    "sampling_policy": sampling_policy.as_str(),
                    "source_fps": source_fps,
                    "sample_fps": effective_sample_fps,
                    "decoded_frames": signals.len(),
                    "width": decoded.width,
                    "height": decoded.height,
                    "pixel_format": decoded.pixel_format,
                    "signals": signals
                }),
            )
            .await
        {
            return internal_error(
                &state,
                format!("failed to append frames_decoded event: {err}"),
            );
        }

        return ok(json!({
            "request_id": request_id,
            "run_id": run_id,
            "status": "processing",
            "decoded_frames": decoded.frame_signals.len(),
            "source_uri": source_uri,
            "sampling_policy": sampling_policy.as_str(),
            "source_fps": source_fps,
            "sample_fps": effective_sample_fps
        }));
    }

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "ingest_received",
            json!({
                "request_id": request_id,
                "ingest": payload
            }),
        )
        .await
    {
        return internal_error(&state, format!("failed to append ingest event: {err}"));
    }

    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
        "status": "processing"
    }))
}

#[tracing::instrument(name = "api.stop_run", skip_all, fields(run_id))]
pub async fn stop_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid stop request") {
        return error;
    }

    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    if run_snapshot.state.is_terminal() {
        return conflict_error(
            &state,
            "run already terminal",
            vec![field_error(
                "run_id",
                format!("run is in terminal state: {:?}", run_snapshot.state),
            )],
        );
    }

    let request_id = state.next_request_id();
    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "stop_requested",
            json!({ "request_id": request_id }),
        )
        .await
    {
        return internal_error(&state, format!("failed to append stop event: {err}"));
    }

    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
        "status": "cancelled"
    }))
}

pub async fn keepalive_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid keepalive request") {
        return error;
    }

    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    if run_snapshot.state.is_terminal() {
        return conflict_error(
            &state,
            "cannot keepalive a terminal run",
            vec![field_error(
                "run_id",
                format!("run is in terminal state: {:?}", run_snapshot.state),
            )],
        );
    }

    let request_id = state.next_request_id();
    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "keepalive_refreshed",
            json!({ "request_id": request_id }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append keepalive_refreshed event: {err}"),
        );
    }

    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
        "state": "processing"
    }))
}

pub async fn get_events(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Query(query): Query<EventsQueryParams>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid events request") {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }

    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error,
    };

    // Filter by index_name when ?index=<name> is supplied.
    // The index is stored as `"index_name": "<name>"` inside the event payload
    // JSON.  Events without an `index_name` field are only returned when no
    // filter is specified (i.e. when `query.index` is `None`).
    let events = events
        .into_iter()
        .filter(|event| {
            match &query.index {
                None => true,
                Some(wanted) => {
                    // Parse the payload to check for a matching index_name.
                    serde_json::from_str::<serde_json::Value>(&event.payload)
                        .ok()
                        .and_then(|v| {
                            v.get("index_name")
                                .and_then(|v| v.as_str())
                                .map(|s| s == wanted.as_str())
                        })
                        .unwrap_or(false)
                }
            }
        })
        .map(|event| {
            json!({
                "seq": event.seq,
                "pts_ms": event.pts_ms,
                "kind": event.kind,
                "payload": parse_payload(&event.payload)
            })
        })
        .collect::<Vec<_>>();

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "events": events
    }))
}

pub async fn get_state(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid state request") {
        return error;
    }

    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    let state_value = run_snapshot.state;

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "state": state_value.as_lowercase_str()
    }))
}

pub async fn query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let run_id = payload.get("run_id").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(error) = validate_run_id_or_error(&state, run_id, "invalid query payload") {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, run_id) {
        return error;
    }

    let kind_filter = payload.get("kind").and_then(|v| v.as_str());
    let from_seq = payload
        .get("from_seq")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let events = match load_existing_events(&state, run_id).await {
        Ok(events) => events,
        Err(error) => return error,
    };

    let matches = events
        .into_iter()
        .filter(|event| event.seq >= from_seq)
        .filter(|event| kind_filter.map(|kind| event.kind == kind).unwrap_or(true))
        .map(|event| {
            json!({
                "seq": event.seq,
                "pts_ms": event.pts_ms,
                "kind": event.kind,
                "payload": parse_payload(&event.payload)
            })
        })
        .collect::<Vec<_>>();

    ok(json!({
        "request_id": state.next_request_id(),
        "query": payload,
        "matches": matches
    }))
}

#[tracing::instrument(name = "api.infer", skip_all)]
pub async fn infer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<InferRequest>,
) -> impl IntoResponse {
    if state.provider().is_none() {
        return internal_error(
            &state,
            "inference providers are not configured; set VIDARAX_VLLM_BASE_URL and VIDARAX_SGLANG_BASE_URL",
        );
    }
    let prepared =
        match validate_infer_request(&state, &headers, payload, "invalid infer payload").await {
            Ok(prepared) => prepared,
            Err(error) => return error,
        };
    match execute_infer_request(state.clone(), prepared).await {
        Ok(response) => ok(json!(response)),
        Err(error) => infer_execution_error_to_response(&state, error),
    }
}

#[tracing::instrument(name = "api.infer_batch", skip_all)]
pub async fn infer_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<InferBatchRequest>,
) -> impl IntoResponse {
    if state.provider().is_none() {
        return internal_error(
            &state,
            "inference providers are not configured; set VIDARAX_VLLM_BASE_URL and VIDARAX_SGLANG_BASE_URL",
        );
    }
    let InferBatchRequest {
        requests,
        max_parallel,
    } = payload;
    let total = requests.len();
    if requests.is_empty() || total > 256 {
        return validation_error(
            &state,
            "invalid infer-batch payload",
            vec![field_error(
                "requests",
                "requests length must be in [1, 256]".to_string(),
            )],
        );
    }

    let max_parallel = max_parallel.unwrap_or(8);
    if !(1..=64).contains(&max_parallel) {
        return validation_error(
            &state,
            "invalid infer-batch payload",
            vec![field_error(
                "max_parallel",
                "max_parallel must be in [1, 64]".to_string(),
            )],
        );
    }

    let mut prepared = Vec::with_capacity(total);
    for request in requests {
        match validate_infer_request(&state, &headers, request, "invalid infer-batch payload").await
        {
            Ok(item) => prepared.push(item),
            Err(error) => return error,
        }
    }

    // Keep in-flight provider calls bounded to avoid unbounded memory growth on large batches.
    let mut join_set = JoinSet::new();
    let chunk_size = min(max_parallel, prepared.len());
    let mut pending = prepared.into_iter().enumerate();
    let mut results = Vec::with_capacity(total);
    let mut processed = 0usize;
    let mut succeeded = 0usize;
    let mut failed = 0usize;
    let mut ordered = std::iter::repeat_with(|| None)
        .take(total)
        .collect::<Vec<Option<InferBatchItemResult>>>();

    for _ in 0..chunk_size {
        if let Some((index, item)) = pending.next() {
            let state_for_task = state.clone();
            join_set
                .spawn(async move { (index, execute_infer_request(state_for_task, item).await) });
        }
    }

    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok((index, result)) => {
                processed += 1;
                match result {
                    Ok(item) => {
                        succeeded += 1;
                        ordered[index] = Some(InferBatchItemResult {
                            index,
                            ok: true,
                            result: Some(item),
                            error: None,
                        });
                    }
                    Err(error) => {
                        failed += 1;
                        ordered[index] = Some(InferBatchItemResult {
                            index,
                            ok: false,
                            result: None,
                            error: Some(InferBatchItemError {
                                code: error.code,
                                message: error.message,
                            }),
                        });
                    }
                }
                if let Some((next_index, next_item)) = pending.next() {
                    let state_for_task = state.clone();
                    join_set.spawn(async move {
                        (
                            next_index,
                            execute_infer_request(state_for_task, next_item).await,
                        )
                    });
                }
            }
            Err(err) => {
                return internal_error(&state, format!("inference worker join failure: {err}"));
            }
        }
    }

    results.extend(ordered.into_iter().flatten());
    results.sort_by_key(|entry| entry.index);
    ok(json!(InferBatchResponse {
        request_id: state.next_request_id(),
        processed,
        succeeded,
        failed,
        results,
    }))
}

#[tracing::instrument(name = "api.analyze_run", skip_all, fields(run_id))]
pub async fn analyze_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<AnalyzeFramesRequest>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid analyze request") {
        return error;
    }
    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    if run_snapshot.state.is_terminal() {
        return conflict_error(
            &state,
            "cannot analyze terminal run",
            vec![field_error(
                "run_id",
                format!("run is in terminal state: {:?}", run_snapshot.state),
            )],
        );
    }

    let mode = match normalize_mode(payload.mode) {
        Ok(mode) => mode,
        Err(message) => {
            return validation_error(
                &state,
                "invalid analyze payload",
                vec![field_error("mode", message)],
            );
        }
    };
    let model = match normalize_model(Some(payload.model)) {
        Ok(Some(model)) => model,
        Ok(None) => unreachable!("model is required for analyze payload"),
        Err(message) => {
            return validation_error(
                &state,
                "invalid analyze payload",
                vec![field_error("model", message)],
            );
        }
    };

    let window_size = payload.window_size.unwrap_or(16);
    if !(2..=256).contains(&window_size) {
        return validation_error(
            &state,
            "invalid analyze payload",
            vec![field_error(
                "window_size",
                "window_size must be in [2, 256]".to_string(),
            )],
        );
    }
    let segment_ms = payload.segment_ms.unwrap_or(250);
    if !(50..=60_000).contains(&segment_ms) {
        return validation_error(
            &state,
            "invalid analyze payload",
            vec![field_error(
                "segment_ms",
                "segment_ms must be in [50, 60000]".to_string(),
            )],
        );
    }

    let (signals, sampling_policy, sample_fps) = if payload.frames.is_empty() {
        let events = match load_existing_events(&state, &run_id).await {
            Ok(events) => events,
            Err(error) => return error,
        };
        match load_decoded_signals_from_events(&events) {
            Ok(decoded) => (decoded.signals, decoded.sampling_policy, decoded.sample_fps),
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid analyze payload",
                    vec![field_error("frames", message)],
                );
            }
        }
    } else {
        if payload.frames.len() > 4096 {
            return validation_error(
                &state,
                "invalid analyze payload",
                vec![field_error(
                    "frames",
                    "frames length must be in [1, 4096]".to_string(),
                )],
            );
        }

        let mut signals = Vec::with_capacity(payload.frames.len());
        for frame in &payload.frames {
            if !(0.0..=1.0).contains(&frame.luma_mean)
                || !(0.0..=1.0).contains(&frame.flicker_score)
                || !(0.0..=1.0).contains(&frame.ghosting_score)
                || !(0.0..=1.0).contains(&frame.noise_variance_score)
            {
                return validation_error(
                    &state,
                    "invalid analyze payload",
                    vec![field_error(
                        "frames",
                        "frame scores/luma must be normalized to [0.0, 1.0]".to_string(),
                    )],
                );
            }

            signals.push(FrameSignal {
                frame_index: frame.frame_index,
                pts_ms: frame.pts_ms,
                perceptual_hash: frame.perceptual_hash,
                luma_mean: frame.luma_mean,
                flicker_score: frame.flicker_score,
                ghosting_score: frame.ghosting_score,
                noise_variance_score: frame.noise_variance_score,
            });
        }
        let sampling_policy = match SamplingPolicy::parse(payload.sampling_policy.as_deref()) {
            Ok(policy) => policy,
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid analyze payload",
                    vec![field_error("sampling_policy", message.to_string())],
                );
            }
        };
        let sample_fps = if sampling_policy == SamplingPolicy::Fixed {
            let Some(fixed) = payload.fixed_fps else {
                return validation_error(
                    &state,
                    "invalid analyze payload",
                    vec![field_error(
                        "fixed_fps",
                        "fixed_fps is required when sampling_policy=fixed".to_string(),
                    )],
                );
            };
            if !(0.2..=120.0).contains(&fixed) {
                return validation_error(
                    &state,
                    "invalid analyze payload",
                    vec![field_error(
                        "fixed_fps",
                        "fixed_fps must be in [0.2, 120.0]".to_string(),
                    )],
                );
            }
            fixed
        } else {
            estimate_sample_fps(&signals).unwrap_or(1.0)
        };
        (signals, sampling_policy, sample_fps)
    };

    let principal = state.security_policy().principal_key_from_headers(&headers);
    let label_map_key = label_map_key_from_principal(&principal);
    let stream_id = payload.stream_id.unwrap_or_else(|| "stream-0".to_string());
    let request_id = state.next_request_id();
    let trace_id = payload
        .trace_id
        .unwrap_or_else(|| format!("trace-{}", &request_id[4..]));
    let mut pipeline = TwoPassPipeline::new(
        TwoPassConfig {
            window_size,
            segment_ms,
            confidence_weights: Default::default(),
        },
        GateConfig::default(),
    );
    let analyzed = pipeline.analyze_batch(&signals);

    let mut marker_inputs = Vec::with_capacity(analyzed.len());
    let metadata = analyzed
        .iter()
        .copied()
        .map(|m| {
            let (metadata, marker_input) = compose_frame_metadata(
                &state,
                label_map_key,
                &run_id,
                &stream_id,
                mode,
                model,
                sampling_policy,
                sample_fps,
                segment_ms,
                &request_id,
                &trace_id,
                m,
                None,
                false,
                None,
            );
            marker_inputs.push(marker_input);
            metadata
        })
        .collect::<Vec<_>>();

    let markers = build_marker_lifecycle(
        &run_id,
        &stream_id,
        &marker_inputs,
        &MarkerConfig::default(),
    )
    .into_iter()
    .map(semantic_marker_to_api_marker)
    .collect::<Vec<_>>();

    for marker in &markers {
        if let Err(err) = state
            .append_run_event_async(&run_id, "marker_emitted", json!(marker))
            .await
        {
            return internal_error(
                &state,
                format!("failed to append marker_emitted event: {err}"),
            );
        }
    }

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "analysis_generated",
            json!({
                "request_id": request_id,
                "stream_id": stream_id,
                "frames": metadata.len(),
                "window_size": window_size,
                "segment_ms": segment_ms,
                "sampling_policy": sampling_policy.as_str(),
                "sample_fps": sample_fps,
                "mode": mode,
                "model": model,
                "markers": markers.len()
            }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append analysis_generated event: {err}"),
        );
    }

    ok(json!(AnalyzeFramesResponse {
        request_id,
        run_id,
        generated: metadata.len(),
        metadata,
        markers,
    }))
}

struct RealtimeReasonParams {
    mode: &'static str,
    model: &'static str,
    sampling_policy: SamplingPolicy,
    max_frames: u64,
    chunk_size: usize,
    window_size: usize,
    segment_ms: u64,
    semantic_inference: bool,
    semantic_frames_per_chunk: usize,
    semantic_timeout_ms: u64,
    semantic_prompt: String,
    tiered_config: vidarax_core::tiered_vlm::TieredVlmConfig,
    decode_source: InputSource,
    video_clip_mode: bool,
    video_clip_duration_s: f32,
    fixed_fps: f32,
}

fn validate_realtime_reason_params(
    state: &AppState,
    payload: &RealtimeReasonRequest,
) -> Result<RealtimeReasonParams, ApiResponse> {
    let mode = normalize_mode(payload.mode.clone()).map_err(|message| {
        validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error("mode", message)],
        )
    })?;
    let model = normalize_model(Some(payload.model.clone()))
        .map_err(|message| {
            validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error("model", message)],
            )
        })?
        .expect("model is required");
    let sampling_policy =
        SamplingPolicy::parse(payload.sampling_policy.as_deref()).map_err(|message| {
            validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error("sampling_policy", message.to_string())],
            )
        })?;
    let max_frames = payload.max_frames.unwrap_or(120_000);
    if !(1..=500_000).contains(&max_frames) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "max_frames",
                "max_frames must be in [1, 500000]".to_string(),
            )],
        ));
    }
    let chunk_size = payload.chunk_size.unwrap_or(25);
    if !(5..=500).contains(&chunk_size) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "chunk_size",
                "chunk_size must be in [5, 500]".to_string(),
            )],
        ));
    }
    let window_size = payload.window_size.unwrap_or(16);
    if !(2..=256).contains(&window_size) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "window_size",
                "window_size must be in [2, 256]".to_string(),
            )],
        ));
    }
    let segment_ms = payload.segment_ms.unwrap_or(250);
    if segment_ms == 0 {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "segment_ms",
                "segment_ms must be >= 1".to_string(),
            )],
        ));
    }
    let semantic_inference = payload.semantic_inference.unwrap_or(true);
    let semantic_frames_per_chunk = payload.semantic_frames_per_chunk.unwrap_or(2);
    if !(1..=4).contains(&semantic_frames_per_chunk) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_frames_per_chunk",
                "semantic_frames_per_chunk must be in [1, 4]".to_string(),
            )],
        ));
    }
    let semantic_timeout_ms = payload.semantic_timeout_ms.unwrap_or(1_500);
    if !(100..=120_000).contains(&semantic_timeout_ms) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_timeout_ms",
                "semantic_timeout_ms must be in [100, 120000]".to_string(),
            )],
        ));
    }
    let semantic_prompt = payload.semantic_prompt.clone().unwrap_or_else(|| {
        "You are classifying a short video chunk. Return strict JSON with keys: event_type, object_label, summary, description, confidence (0..1). event_type must be one of: scene_cut, artifact_suspected, keyframe_keep, context_observation.".to_string()
    });
    if semantic_prompt.trim().is_empty() || semantic_prompt.len() > 4_096 {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_prompt",
                "semantic_prompt must be non-empty and <= 4096 bytes".to_string(),
            )],
        ));
    }

    let tiered_config = {
        use vidarax_core::tiered_vlm::TieredVlmConfig;
        let first = payload.first_pass_model.as_deref().unwrap_or(model);
        let second = payload.second_pass_model.as_deref().unwrap_or(model);
        let threshold = payload.second_pass_threshold.unwrap_or(0.7);
        TieredVlmConfig {
            first_pass_model: Arc::from(first),
            second_pass_model: Arc::from(second),
            second_pass_threshold: threshold.clamp(0.0, 1.0),
            second_pass_max_tokens: 256,
        }
    };

    let allowed_roots = ingest_file_roots_with_upload_root(state);
    let decode_source = InputSource::parse_and_validate(&payload.source_uri, &allowed_roots)
        .map_err(|message| {
            validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error("source_uri", message)],
            )
        })?;

    let video_clip_mode = payload.video_clip_mode.unwrap_or(false);
    let video_clip_duration_s = payload.video_clip_duration_s.unwrap_or(0.5);
    if video_clip_mode
        && (!video_clip_duration_s.is_finite()
            || video_clip_duration_s <= 0.0
            || video_clip_duration_s > 60.0)
    {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "video_clip_duration_s",
                "video_clip_duration_s must be in (0, 60]".to_string(),
            )],
        ));
    }

    let fixed_fps = payload.fixed_fps.unwrap_or(1.0);
    if sampling_policy == SamplingPolicy::Fixed && !(0.2..=120.0).contains(&fixed_fps) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "fixed_fps",
                "fixed_fps must be in [0.2, 120.0]".to_string(),
            )],
        ));
    }

    Ok(RealtimeReasonParams {
        mode,
        model,
        sampling_policy,
        max_frames,
        chunk_size,
        window_size,
        segment_ms,
        semantic_inference,
        semantic_frames_per_chunk,
        semantic_timeout_ms,
        semantic_prompt,
        tiered_config,
        decode_source,
        video_clip_mode,
        video_clip_duration_s,
        fixed_fps,
    })
}

struct RealtimeAssemblyOutput {
    metadata: Vec<AnalyzeFrameMetadata>,
    markers: Vec<AnalyzeMarker>,
    lag_p95_ms: u64,
    lag_p99_ms: u64,
}

#[allow(clippy::too_many_arguments)]
async fn assemble_realtime_reason_response(
    state: &AppState,
    run_id: &str,
    stream_id: &str,
    mode: &str,
    model: &str,
    sampling_policy: SamplingPolicy,
    sample_fps: f32,
    source_fps: Option<f32>,
    semantic_segment_ms: u64,
    request_id: &str,
    trace_id: &str,
    tenant_id: Option<&str>,
    index_name: &Option<String>,
    marker_config: &MarkerConfig,
    chunk_preps: Vec<ChunkPrep>,
    mut semantic_results: Vec<Option<ChunkSemanticResult>>,
    task_end_times: Vec<Instant>,
) -> Result<RealtimeAssemblyOutput, ApiResponse> {
    let decoded_frames = chunk_preps.iter().map(|prep| prep.analyzed.len()).sum();
    let mut metadata = Vec::with_capacity(decoded_frames);
    let mut marker_inputs = Vec::with_capacity(decoded_frames);
    let mut chunk_lags = Vec::new();

    for (chunk_idx, prep) in chunk_preps.into_iter().enumerate() {
        let semantic_overlay = semantic_results[chunk_idx].take().unwrap_or_default();
        let finished = task_end_times[chunk_idx];

        if let Some(mut details) = semantic_overlay.event_payload(chunk_idx, request_id, stream_id)
        {
            if let Some(ref idx) = index_name {
                if let Some(obj) = details.as_object_mut() {
                    obj.insert("index_name".to_string(), serde_json::json!(idx));
                }
            }
            if let Err(err) = state
                .append_run_event_async(run_id, "semantic_chunk_inferred", details)
                .await
            {
                return Err(internal_error(
                    state,
                    format!("failed to append semantic_chunk_inferred event: {err}"),
                ));
            }
        }

        for frame in prep.analyzed {
            let (row, marker_input) = compose_frame_metadata(
                state,
                tenant_id,
                run_id,
                stream_id,
                mode,
                model,
                sampling_policy,
                sample_fps,
                semantic_segment_ms,
                request_id,
                trace_id,
                frame,
                semantic_overlay.overlay.as_ref(),
                semantic_overlay.used_fallback,
                semantic_overlay.finish_reason.clone(),
            );
            metadata.push(row);
            marker_inputs.push(marker_input);
        }

        let process_ms = finished.duration_since(prep.started).as_millis() as u64;
        let source_span_ms = prep.pts_end_ms.saturating_sub(prep.pts_start_ms);
        let lag_ms = process_ms.saturating_sub(source_span_ms);
        chunk_lags.push(lag_ms);

        if let Err(err) = state
            .append_run_event_async(
                run_id,
                "semantic_chunk_generated",
                json!({
                    "request_id": request_id,
                    "stream_id": stream_id,
                    "chunk_index": chunk_idx,
                    "chunk_frames": prep.chunk_len,
                    "process_ms": process_ms,
                    "source_span_ms": source_span_ms,
                    "lag_ms": lag_ms,
                    "index_name": index_name,
                }),
            )
            .await
        {
            return Err(internal_error(
                state,
                format!("failed to append semantic_chunk_generated event: {err}"),
            ));
        }
    }

    let markers = build_marker_lifecycle(run_id, stream_id, &marker_inputs, marker_config)
        .into_iter()
        .map(semantic_marker_to_api_marker)
        .collect::<Vec<_>>();
    for marker in &markers {
        let mut marker_payload = json!(marker);
        if let Some(ref idx) = index_name {
            if let Some(obj) = marker_payload.as_object_mut() {
                obj.insert("index_name".to_string(), serde_json::json!(idx));
            }
        }
        if let Err(err) = state
            .append_run_event_async(run_id, "marker_emitted", marker_payload)
            .await
        {
            return Err(internal_error(
                state,
                format!("failed to append marker_emitted event: {err}"),
            ));
        }
    }

    let lag_p95_ms = percentile_ms(&chunk_lags, 95);
    let lag_p99_ms = percentile_ms(&chunk_lags, 99);
    if let Err(err) = state
        .append_run_event_async(
            run_id,
            "analysis_generated",
            json!({
                "request_id": request_id,
                "stream_id": stream_id,
                "frames": metadata.len(),
                "markers": markers.len(),
                "sampling_policy": sampling_policy.as_str(),
                "source_fps": source_fps,
                "sample_fps": sample_fps,
                "lag_p95_ms": lag_p95_ms,
                "lag_p99_ms": lag_p99_ms,
                "mode": mode,
                "model": model,
                "index_name": index_name,
            }),
        )
        .await
    {
        return Err(internal_error(
            state,
            format!("failed to append analysis_generated event: {err}"),
        ));
    }

    if let Err(err) = state
        .append_run_event_async(
            run_id,
            "run_completed",
            json!({
                "request_id": request_id,
                "stream_id": stream_id,
                "frames": metadata.len(),
                "markers": markers.len(),
                "index_name": index_name,
            }),
        )
        .await
    {
        return Err(internal_error(
            state,
            format!("failed to append run_completed event: {err}"),
        ));
    }

    Ok(RealtimeAssemblyOutput {
        metadata,
        markers,
        lag_p95_ms,
        lag_p99_ms,
    })
}

#[tracing::instrument(name = "api.reason_realtime_run", skip_all, fields(run_id))]
pub async fn reason_realtime_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<RealtimeReasonRequest>,
) -> impl IntoResponse {
    if let Some(error) =
        validate_run_id_or_error(&state, &run_id, "invalid realtime reason request")
    {
        return error;
    }
    let run_snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(snapshot) => snapshot,
        Err(error) => return error,
    };
    if run_snapshot.state.is_terminal() {
        return conflict_error(
            &state,
            "cannot reason over terminal run",
            vec![field_error(
                "run_id",
                format!("run is in terminal state: {:?}", run_snapshot.state),
            )],
        );
    }

    let params = match validate_realtime_reason_params(&state, &payload) {
        Ok(params) => params,
        Err(error) => return error,
    };
    if let Err(error) = enforce_file_source_visibility(
        &state,
        &headers,
        &payload.source_uri,
        &params.decode_source,
        "invalid realtime reason request",
    ) {
        return error;
    }
    let mode = params.mode;
    let model = params.model;
    let sampling_policy = params.sampling_policy;
    let max_frames = params.max_frames;
    let chunk_size = params.chunk_size;
    let window_size = params.window_size;
    let segment_ms = params.segment_ms;
    let semantic_inference = params.semantic_inference;
    let semantic_frames_per_chunk = params.semantic_frames_per_chunk;
    let semantic_timeout_ms = params.semantic_timeout_ms;
    let semantic_prompt = params.semantic_prompt;
    let tiered_config = params.tiered_config;
    let video_clip_mode = params.video_clip_mode;
    let video_clip_duration_s = params.video_clip_duration_s;
    let fixed_fps = params.fixed_fps;
    let semantic_decode_enabled = semantic_inference && state.provider().is_some();
    let decode_source_ref = params.decode_source.clone();
    let decode_source = params.decode_source;
    let decode_pipeline = state.decode_pipeline();
    let (decoded, source_fps, sample_fps, decoded_jpegs) =
        match tokio::task::spawn_blocking(move || {
            let source_fps = probe_source_fps(&decode_source);
            let sample_fps = match sampling_policy {
                SamplingPolicy::SourceFpsAdaptive => {
                    source_fps.map(adaptive_sample_fps).unwrap_or(24.0)
                }
                SamplingPolicy::Fixed => fixed_fps,
            };
            let decode_config = Mp4DecodeConfig {
                sample_fps,
                max_frames: max_frames as usize,
            };
            // Pass 1: frame signals (cheap, no encoding)
            let decoded = decode_pipeline.decode_signals(&decode_source, decode_config)?;

            // In video_clip_mode, JPEG decoding is skipped here; clips are
            // extracted per chunk below.
            let decoded_jpegs = if semantic_decode_enabled && !video_clip_mode {
                let indices = compute_semantic_frame_indices(
                    decoded.frame_signals.len(),
                    chunk_size,
                    semantic_frames_per_chunk,
                );
                let jpegs = decode_pipeline.decode_jpegs(
                    &decode_source,
                    sample_fps,
                    &indices,
                    max_frames as usize,
                )?;
                let lookup: std::collections::HashMap<u64, DecodedJpegFrame> =
                    jpegs.into_iter().map(|f| (f.frame_index, f)).collect();
                Some(lookup)
            } else {
                None
            };
            Ok((decoded, source_fps, sample_fps, decoded_jpegs))
        })
        .await
        {
            Ok(Ok(decoded)) => decoded,
            Ok(Err(err)) => {
                return validation_error(
                    &state,
                    "invalid realtime reason request",
                    vec![field_error("source_uri", err)],
                );
            }
            Err(err) => {
                return internal_error(
                    &state,
                    format!("realtime reason decode worker join failure: {err}"),
                );
            }
        };

    let request_id = state.next_request_id();
    let trace_id = payload
        .trace_id
        .unwrap_or_else(|| format!("trace-{}", &request_id[4..]));
    let stream_id = payload.stream_id.unwrap_or_else(|| "stream-0".to_string());
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let label_map_key = label_map_key_from_principal(&principal);
    // Optional index tag — carried through all WAL events for this pass so
    // callers can filter with GET /v1/runs/{id}/events?index=<name>.
    let index_name: Option<String> = payload.index_name;

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "ingest_received",
            json!({
                "request_id": request_id,
                "source_uri": payload.source_uri.as_str(),
                "index_name": index_name,
            }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append ingest_received event: {err}"),
        );
    }

    let marker_config = MarkerConfig {
        correction_window_frames: payload.marker_correction_window_frames.unwrap_or(3),
        ..MarkerConfig::default()
    };
    let mut pipeline = TwoPassPipeline::new(
        TwoPassConfig {
            window_size,
            segment_ms,
            confidence_weights: Default::default(),
        },
        GateConfig::default(),
    );

    let providers = state.provider().cloned();
    let semantic_available = semantic_inference && providers.is_some();
    let semantic_segment_ms = segment_ms;
    if semantic_inference && !semantic_available {
        if let Err(err) = state
            .append_run_event_async(
                &run_id,
                "semantic_fallback_activated",
                json!({
                    "request_id": request_id,
                    "stream_id": stream_id,
                    "reason": "provider_not_configured"
                }),
            )
            .await
        {
            return internal_error(
                &state,
                format!("failed to append semantic_fallback_activated event: {err}"),
            );
        }
    }

    let chunk_preps = prepare_realtime_chunks(
        &decoded.frame_signals,
        chunk_size,
        decoded_jpegs.as_ref(),
        &mut pipeline,
        &decode_source_ref,
        video_clip_mode,
        semantic_decode_enabled,
        video_clip_duration_s,
    )
    .await;

    let visual_diff = payload.visual_diff.unwrap_or(false);
    let temporal_chain = visual_diff || payload.temporal_chain.unwrap_or(false);
    let guided_json_str: Option<String> = payload
        .output_schema
        .as_ref()
        .and_then(|s| serde_json::to_string(s).ok());
    let vlm_concurrency = payload.vlm_concurrency.unwrap_or(4).clamp(1, 64);
    let (semantic_results, task_end_times) = run_semantic_dispatch(
        &chunk_preps,
        providers,
        semantic_available,
        &semantic_prompt,
        semantic_timeout_ms,
        semantic_frames_per_chunk,
        tiered_config,
        guided_json_str,
        visual_diff,
        temporal_chain,
        vlm_concurrency,
    )
    .await;

    let assembled = match assemble_realtime_reason_response(
        &state,
        &run_id,
        &stream_id,
        mode,
        model,
        sampling_policy,
        sample_fps,
        source_fps,
        semantic_segment_ms,
        &request_id,
        &trace_id,
        label_map_key,
        &index_name,
        &marker_config,
        chunk_preps,
        semantic_results,
        task_end_times,
    )
    .await
    {
        Ok(assembled) => assembled,
        Err(error) => return error,
    };

    ok(json!(RealtimeReasonResponse {
        request_id,
        run_id,
        generated: assembled.metadata.len(),
        markers_emitted: assembled.markers.len(),
        decoded_frames: decoded.frame_signals.len(),
        sample_fps,
        lag_p95_ms: assembled.lag_p95_ms,
        lag_p99_ms: assembled.lag_p99_ms,
        metadata: assembled.metadata,
        markers: assembled.markers,
    }))
}

pub async fn get_markers(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Query(query): Query<MarkerQueryParams>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid markers request") {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }
    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error,
    };

    let mut markers = Vec::new();
    for event in events {
        if event.kind != "marker_emitted" {
            continue;
        }
        let Ok(marker) = serde_json::from_str::<AnalyzeMarker>(&event.payload) else {
            continue;
        };
        if query
            .status
            .as_deref()
            .map(|status| marker.status == status)
            .unwrap_or(true)
            && query
                .event_type
                .as_deref()
                .map(|event_type| marker.event_type == event_type)
                .unwrap_or(true)
            && query
                .from_frame
                .map(|from| marker.end_frame >= from)
                .unwrap_or(true)
            && query
                .to_frame
                .map(|to| marker.start_frame <= to)
                .unwrap_or(true)
        {
            markers.push(marker);
        }
    }
    markers.sort_by(|a, b| {
        a.start_frame
            .cmp(&b.start_frame)
            .then(a.end_frame.cmp(&b.end_frame))
            .then(a.marker_id.as_str().cmp(b.marker_id.as_str()))
    });

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "markers": markers
    }))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let request_id = state.next_request_id();
    let provider = state.provider().cloned();
    let is_saturated = state.inference_metrics().is_high_latency();
    let availability = match tokio::task::spawn_blocking(move || {
        runtime_model_availability(provider, is_saturated)
    })
    .await
    {
        Ok(availability) => availability,
        Err(err) => {
            return internal_error(&state, format!("model catalog worker join failure: {err}"));
        }
    };
    let providers_available = availability.providers;
    let status = availability.status;

    let mut models = Vec::with_capacity(REQUIRED_MEDIUM_MODELS.len() + REQUIRED_SMALL_MODELS.len());
    for model in REQUIRED_MEDIUM_MODELS {
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "medium".to_string(),
            availability: status.to_string(),
            providers_available: providers_available.clone(),
            fallback_candidates: fallback_candidates(model)
                .iter()
                .map(ToString::to_string)
                .collect(),
        });
    }
    for model in REQUIRED_SMALL_MODELS {
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "small".to_string(),
            availability: status.to_string(),
            providers_available: providers_available.clone(),
            fallback_candidates: fallback_candidates(model)
                .iter()
                .map(ToString::to_string)
                .collect(),
        });
    }

    ok(json!(ModelCatalogResponse { request_id, models }))
}

pub async fn health() -> impl IntoResponse {
    ok(json!({ "status": "ok" }))
}

/// `POST /v1/search`
///
/// Substring search over VLM descriptions stored in the WAL.
///
/// Scans all WAL events and returns those whose payload contains a
/// `description` field matching the query string (case-insensitive).  When
/// `run_id` is supplied only events belonging to that run are scanned.
///
/// Exact substring matching is O(n) in the number of stored events but is fast
/// enough for the typical WAL sizes encountered in development and staging.
/// A vector-embedding upgrade path is available by storing description
/// embeddings at write time and replacing this scan with a k-NN query.
#[tracing::instrument(name = "api.search", skip_all)]
pub async fn search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<SearchRequest>,
) -> impl IntoResponse {
    let query = payload.query.trim();
    if query.is_empty() {
        return validation_error(
            &state,
            "invalid search request",
            vec![field_error("query", "query must not be empty".to_string())],
        );
    }
    if query.len() > 1024 {
        return validation_error(
            &state,
            "invalid search request",
            vec![field_error(
                "query",
                "query must be <= 1024 bytes".to_string(),
            )],
        );
    }

    let limit = payload.limit.unwrap_or(50);
    if limit == 0 || limit > 500 {
        return validation_error(
            &state,
            "invalid search request",
            vec![field_error(
                "limit",
                "limit must be in [1, 500]".to_string(),
            )],
        );
    }

    let principal = state.security_policy().principal_key_from_headers(&headers);
    let events = if let Some(ref run_id) = payload.run_id {
        if let Some(error) = validate_run_id_or_error(&state, run_id, "invalid search request") {
            return error;
        }
        if let Err(error) = load_run_snapshot(&state, &headers, run_id) {
            return error;
        }
        match state.read_run_events_async(run_id).await {
            Ok(events) => {
                if events.iter().any(|event| event.kind == "run_deleted") {
                    return not_found_error(
                        &state,
                        "run_id was not found",
                        vec![field_error("run_id", run_id.to_string())],
                    );
                }
                events
            }
            Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
        }
    } else {
        match state.read_all_events_async().await {
            Ok(events) => events,
            Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
        }
    };

    let owned_run_ids = if payload.run_id.is_some() {
        None
    } else {
        Some(owned_run_ids_from_events(&events, &principal))
    };

    // Case-insensitive substring search over the `description` field in every
    // event payload.  The lowercase query is computed once.
    let query_lower = query.to_lowercase();

    let mut hits: Vec<SearchHit> = Vec::new();
    let mut scanned = 0usize;
    let mut total_hits = 0usize;

    for event in events {
        if let Some(owned_run_ids) = &owned_run_ids {
            if !owned_run_ids.contains(&event.run_id) {
                continue;
            }
        }
        scanned += 1;
        let payload_val = parse_payload(&event.payload);

        // Extract a description string from the event payload.  Different event
        // kinds store it under different keys:
        // - semantic_chunk_inferred: payload.description (from SemanticOverlay)
        // - vlm / vlm_tiered: payload.description
        // - analysis_generated: no per-frame description; skip
        //
        // We try the most common keys in priority order.
        let description = payload_val
            .get("description")
            .and_then(|v| v.as_str())
            .or_else(|| payload_val.get("summary").and_then(|v| v.as_str()))
            .map(str::to_string);

        let Some(description) = description else {
            continue;
        };

        if !description.to_lowercase().contains(&query_lower) {
            continue;
        }
        total_hits += 1;

        // Extract optional index_name for cross-index searches.
        let index_name = payload_val
            .get("index_name")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);

        if hits.len() < limit {
            hits.push(SearchHit {
                seq: event.seq,
                run_id: event.run_id,
                pts_ms: event.pts_ms,
                kind: event.kind,
                description,
                index_name,
            });
        }
    }

    ok(json!(SearchResponse {
        request_id: state.next_request_id(),
        scanned,
        total_hits,
        hits,
    }))
}

/// Query parameters accepted by `GET /v1/runs/{run_id}/interactions`.
#[derive(Debug, Default, serde::Deserialize)]
pub struct InteractionsQueryParams {
    /// When set, only events whose payload contains `"index_name": "<value>"`
    /// are included.  Mirrors the filter on GET /events.
    pub index: Option<String>,
}

pub async fn get_interactions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(run_id): Path<String>,
    Query(query): Query<InteractionsQueryParams>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid interactions request") {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }
    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error,
    };

    // Build chunk timing map from semantic_chunk_generated events.
    // Key: chunk_index  Value: (pts_start_ms, pts_end_ms)
    let mut chunk_timing: std::collections::HashMap<u64, (u64, u64)> =
        std::collections::HashMap::new();
    for event in events
        .iter()
        .filter(|e| e.kind == "semantic_chunk_generated")
    {
        let payload = parse_payload(&event.payload);
        if let Some(idx) = payload.get("chunk_index").and_then(|v| v.as_u64()) {
            let pts_start = payload
                .get("pts_start_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(event.pts_ms);
            let pts_end = payload
                .get("pts_end_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(event.pts_ms);
            chunk_timing.insert(idx, (pts_start, pts_end));
        }
    }

    // Filter semantic_chunk_inferred events, optionally by index_name.
    let inferred_events: Vec<_> = events
        .iter()
        .filter(|e| e.kind == "semantic_chunk_inferred")
        .filter(|e| match &query.index {
            None => true,
            Some(wanted) => serde_json::from_str::<Value>(&e.payload)
                .ok()
                .and_then(|v| {
                    v.get("index_name")
                        .and_then(|v| v.as_str())
                        .map(|s| s == wanted.as_str())
                })
                .unwrap_or(false),
        })
        .collect();

    let mut interactions: Vec<Value> = Vec::new();

    for event in &inferred_events {
        let payload = parse_payload(&event.payload);

        let chunk_index = payload.get("chunk_index").and_then(|v| v.as_u64());
        let (pts_start_ms, pts_end_ms) = chunk_index
            .and_then(|idx| chunk_timing.get(&idx).copied())
            .unwrap_or((event.pts_ms, event.pts_ms));

        // Extract raw_output from the payload — this is what the VLM returned
        // when an output_schema (guided JSON) was provided.
        let raw_output = payload.get("raw_output");

        match raw_output {
            // Guided-JSON mode: raw_output is an array — flatten all items.
            Some(Value::Array(items)) => {
                for item in items {
                    let mut enriched = item.clone();
                    if let Some(obj) = enriched.as_object_mut() {
                        obj.entry("chunk_index")
                            .or_insert_with(|| json!(chunk_index));
                        obj.entry("pts_start_ms")
                            .or_insert_with(|| json!(pts_start_ms));
                        obj.entry("pts_end_ms").or_insert_with(|| json!(pts_end_ms));
                    }
                    interactions.push(enriched);
                }
            }
            // Guided-JSON mode: raw_output is a single object.
            Some(Value::Object(_)) => {
                let mut enriched = raw_output.unwrap().clone();
                if let Some(obj) = enriched.as_object_mut() {
                    obj.entry("chunk_index")
                        .or_insert_with(|| json!(chunk_index));
                    obj.entry("pts_start_ms")
                        .or_insert_with(|| json!(pts_start_ms));
                    obj.entry("pts_end_ms").or_insert_with(|| json!(pts_end_ms));
                }
                interactions.push(enriched);
            }
            // Legacy / classification mode: synthesise an item from
            // object_label and event_type fields (backward compat).
            _ => {
                let object_label = payload
                    .get("object_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let event_type = payload
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if object_label.is_empty() && event_type.is_empty() {
                    continue;
                }
                interactions.push(json!({
                    "chunk_index": chunk_index,
                    "pts_start_ms": pts_start_ms,
                    "pts_end_ms": pts_end_ms,
                    "object_label": object_label,
                    "event_type": event_type,
                }));
            }
        }
    }

    let count = interactions.len();
    ok(json!({
        "run_id": run_id,
        "count": count,
        "interactions": interactions,
    }))
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let (runs, events) = state.metrics_snapshot();
    let mut metrics =
        format!("vidarax_runs_created_total {runs}\nvidarax_timeline_events_total {events}\n");
    metrics.push_str(&state.inference_metrics().render_prometheus());
    metrics.push_str(&state.pipeline_metrics().render_prometheus());
    (axum::http::StatusCode::OK, metrics)
}

#[derive(Clone)]
struct PreparedInferRequest {
    run_id: Option<String>,
    request: InferenceRequest,
    primary_provider: ProviderKind,
}

struct InferExecutionError {
    code: &'static str,
    message: String,
}

async fn validate_infer_request(
    state: &AppState,
    headers: &HeaderMap,
    payload: InferRequest,
    context: &'static str,
) -> Result<PreparedInferRequest, ApiResponse> {
    if let Some(run_id) = payload.run_id.as_deref() {
        if let Some(error) = validate_run_id_or_error(state, run_id, context) {
            return Err(error);
        }
        let run_snapshot = load_run_snapshot(state, headers, run_id)?;
        if run_snapshot.state.is_terminal() {
            return Err(conflict_error(
                state,
                "cannot run inference on terminal run",
                vec![field_error(
                    "run_id",
                    format!("run is in terminal state: {:?}", run_snapshot.state),
                )],
            ));
        }
    }

    let model = match normalize_model(Some(payload.model)) {
        Ok(Some(model)) => model,
        Ok(None) => unreachable!("model is required in infer request"),
        Err(message) => {
            return Err(validation_error(
                state,
                context,
                vec![field_error("model", message)],
            ));
        }
    };

    let prompt = payload.prompt.trim();
    if prompt.is_empty() {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "prompt",
                "prompt must not be empty".to_string(),
            )],
        ));
    }
    if prompt.len() > 32_768 {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "prompt",
                "prompt length must be <= 32768 bytes".to_string(),
            )],
        ));
    }

    let max_tokens = payload.max_tokens.unwrap_or(256);
    if max_tokens == 0 || max_tokens > 4096 {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "max_tokens",
                "max_tokens must be in [1, 4096]".to_string(),
            )],
        ));
    }

    let temperature = payload.temperature.unwrap_or(0.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "temperature",
                "temperature must be in [0.0, 2.0]".to_string(),
            )],
        ));
    }

    let timeout_ms = payload.timeout_ms.unwrap_or(20_000);
    if timeout_ms == 0 || timeout_ms > 120_000 {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "timeout_ms",
                "timeout_ms must be in [1, 120000]".to_string(),
            )],
        ));
    }

    let primary_provider = match parse_provider(payload.primary_provider.as_deref()) {
        Ok(provider) => provider,
        Err(message) => {
            return Err(validation_error(
                state,
                context,
                vec![field_error("primary_provider", message.to_string())],
            ));
        }
    };

    Ok(PreparedInferRequest {
        run_id: payload.run_id,
        request: InferenceRequest {
            model: Arc::from(model),
            prompt: Arc::from(prompt),
            input_images: Vec::new(),
            input_videos: Vec::new(),
            max_tokens,
            temperature,
            timeout_ms,
            allow_fallback: payload.allow_fallback.unwrap_or(true),
            guided_json: payload
                .output_schema
                .map(|schema| Arc::from(schema.to_string())),
        },
        primary_provider,
    })
}

async fn execute_infer_request(
    state: AppState,
    prepared: PreparedInferRequest,
) -> Result<InferResponse, InferExecutionError> {
    let provider = state.provider().cloned().ok_or(InferExecutionError {
        code: "internal_error",
        message: "inference providers are not configured".to_string(),
    })?;

    let request_id = state.next_request_id();
    let started = Instant::now();
    let primary_provider_for_metrics = prepared.primary_provider;
    let request_for_provider = prepared.request.clone();
    let result =
        match tokio::task::spawn_blocking(move || provider.infer(&request_for_provider)).await {
            Ok(result) => match result {
                Ok(result) => result,
                Err(err) => {
                    state.inference_metrics().record_error(
                        primary_provider_for_metrics,
                        started.elapsed().as_millis() as u64,
                    );
                    return Err(map_provider_execution_error(err));
                }
            },
            Err(err) => {
                state.inference_metrics().record_error(
                    primary_provider_for_metrics,
                    started.elapsed().as_millis() as u64,
                );
                return Err(InferExecutionError {
                    code: "internal_error",
                    message: format!("inference worker join failure: {err}"),
                });
            }
        };
    state.inference_metrics().record_success(
        result.provider,
        started.elapsed().as_millis() as u64,
        result.fallback_used,
    );

    if let Some(run_id) = prepared.run_id.as_deref() {
        let event_payload = json!({
            "request_id": request_id,
            "provider": provider_name(result.provider),
            "model": &*result.model,
            "fallback_used": result.fallback_used,
            "prompt_bytes": prepared.request.prompt.len(),
            "output_bytes": result.output_text.len()
        });
        if let Err(err) = state
            .append_run_event_async(run_id, "inference_completed", event_payload)
            .await
        {
            return Err(InferExecutionError {
                code: "internal_error",
                message: format!("failed to append inference event: {err}"),
            });
        }
    }

    Ok(InferResponse {
        request_id,
        run_id: prepared.run_id,
        provider: provider_name(result.provider).to_string(),
        model: result.model.to_string(),
        fallback_used: result.fallback_used,
        output_text: result.output_text,
        finish_reason: result.finish_reason,
        inference_latency_ms: result.inference_latency_ms,
    })
}

fn map_provider_execution_error(err: ProviderError) -> InferExecutionError {
    match err {
        ProviderError::UnsupportedModel(_) => InferExecutionError {
            code: "validation_error",
            message: "model is not in the supported model contract".to_string(),
        },
        ProviderError::HttpStatus(code) => InferExecutionError {
            code: "provider_http_status",
            message: format!("inference provider returned http status {code}"),
        },
        ProviderError::Transport(message) => InferExecutionError {
            code: "provider_transport",
            message: format!("inference provider transport error: {message}"),
        },
        ProviderError::InvalidResponse(message) => InferExecutionError {
            code: "provider_invalid_response",
            message: format!("inference provider invalid response: {message}"),
        },
    }
}

fn infer_execution_error_to_response(state: &AppState, err: InferExecutionError) -> ApiResponse {
    if err.code == "validation_error" {
        return validation_error(
            state,
            "invalid infer payload",
            vec![field_error("model", err.message)],
        );
    }
    internal_error(state, err.message)
}

#[cfg(test)]
mod tests {
    use super::{
        feedback_rows_to_json_for_owned_runs, owned_run_ids_from_events, run_command_with_timeout,
        validate_infer_request, InferRequest,
    };
    use crate::spacetime_client::FeedbackRow;
    use crate::state::AppState;
    use axum::http::HeaderMap;
    use serde_json::json;
    use std::collections::HashSet;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    static WAL_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_state(tag: &str) -> AppState {
        let n = WAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("vidarax-handlers-{tag}-{n}.wal"));
        AppState::with_wal_for_tests(path)
    }

    fn infer_request(schema: serde_json::Value) -> InferRequest {
        InferRequest {
            run_id: None,
            model: "Qwen/Qwen3-VL-2B-Instruct".to_string(),
            prompt: "return structured data".to_string(),
            max_tokens: None,
            temperature: None,
            timeout_ms: None,
            allow_fallback: None,
            primary_provider: Some("vllm".to_string()),
            output_schema: Some(schema),
        }
    }

    #[tokio::test]
    async fn infer_validation_maps_output_schema_to_guided_json() {
        let state = test_state("infer-schema");
        let prepared = validate_infer_request(
            &state,
            &HeaderMap::new(),
            infer_request(json!({
                "type":"object",
                "properties":{"count":{"type":"number"}},
                "required":["count"]
            })),
            "invalid infer payload",
        )
        .await
        .unwrap();

        let schema = prepared.request.guided_json.as_deref().unwrap();
        let value: serde_json::Value = serde_json::from_str(schema).unwrap();
        assert_eq!(
            value["properties"]["count"]["type"].as_str(),
            Some("number")
        );
    }

    #[tokio::test]
    async fn infer_batch_validation_maps_output_schema_to_guided_json() {
        let state = test_state("batch-schema");
        let prepared = validate_infer_request(
            &state,
            &HeaderMap::new(),
            infer_request(json!({
                "type":"object",
                "properties":{"ok":{"type":"boolean"}},
                "required":["ok"]
            })),
            "invalid infer-batch payload",
        )
        .await
        .unwrap();

        let schema = prepared.request.guided_json.as_deref().unwrap();
        let value: serde_json::Value = serde_json::from_str(schema).unwrap();
        assert_eq!(value["properties"]["ok"]["type"].as_str(), Some("boolean"));
    }

    #[test]
    fn feedback_list_filters_rows_to_owned_runs() {
        let rows = vec![
            FeedbackRow {
                id: 1,
                agent_id: "0xagent".to_string(),
                run_id: "run-00000000000000aa".to_string(),
                session_id: "sess-a".to_string(),
                rating: 8,
                category: "quality".to_string(),
                feedback: "owned".to_string(),
                timestamp_micros: 1,
            },
            FeedbackRow {
                id: 2,
                agent_id: "0xagent".to_string(),
                run_id: "run-00000000000000bb".to_string(),
                session_id: "sess-b".to_string(),
                rating: 2,
                category: "quality".to_string(),
                feedback: "other".to_string(),
                timestamp_micros: 2,
            },
        ];
        let owned = HashSet::from(["run-00000000000000aa".to_string()]);

        let filtered = feedback_rows_to_json_for_owned_runs(rows, &owned);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["run_id"].as_str(), Some("run-00000000000000aa"));
        assert_eq!(filtered[0]["feedback"].as_str(), Some("owned"));
    }

    #[test]
    fn visible_owned_run_set_excludes_deleted_runs() {
        let principal = "public";
        let live_run = "run-00000000000000aa";
        let deleted_run = "run-00000000000000bb";
        let state = test_state("deleted-visible-set");
        state
            .append_run_event(
                live_run,
                "run_created",
                json!({ "principal_key": principal }),
            )
            .unwrap();
        state
            .append_run_event(
                deleted_run,
                "run_created",
                json!({ "principal_key": principal }),
            )
            .unwrap();
        state
            .append_run_event(deleted_run, "run_deleted", json!({}))
            .unwrap();
        let events = state.read_all_events().unwrap();

        let visible = owned_run_ids_from_events(&events, principal);

        assert!(visible.contains(live_run));
        assert!(
            !visible.contains(deleted_run),
            "deleted runs must not remain visible to search or feedback listing"
        );
    }

    #[test]
    fn upload_probe_command_timeout_kills_slow_child() {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 5"]);
        let started = Instant::now();

        let err = run_command_with_timeout(
            &mut command,
            Duration::from_millis(50),
            "uploaded media inspection timed out",
        )
        .expect_err("slow probe command must time out");

        assert_eq!(err, "uploaded media inspection timed out");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "timeout path should not wait for the child sleep duration"
        );
    }
}

#[tracing::instrument(name = "api.submit_feedback", skip_all)]
pub async fn submit_feedback(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
    Json(payload): Json<crate::models::FeedbackRequest>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid feedback request") {
        return error;
    }

    if payload.rating > 10 {
        return validation_error(
            &state,
            "invalid feedback payload",
            vec![field_error(
                "rating",
                "rating must be between 0 and 10".to_string(),
            )],
        );
    }
    if payload.category.is_empty() {
        return validation_error(
            &state,
            "invalid feedback payload",
            vec![field_error(
                "category",
                "category must not be empty".to_string(),
            )],
        );
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }

    let Some(stdb) = state.spacetime_client() else {
        return internal_error(&state, "spacetimedb client not configured");
    };

    let req = crate::spacetime_client::SubmitFeedbackRequest {
        run_id: run_id.clone(),
        session_id: String::new(),
        rating: payload.rating,
        category: payload.category,
        feedback: payload.feedback.unwrap_or_default(),
    };
    if let Err(err) = stdb.submit_feedback_async(&req).await {
        return internal_error(&state, format!("spacetimedb submit_feedback failed: {err}"));
    }

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "status": "submitted"
    }))
}

#[tracing::instrument(name = "api.list_feedback", skip_all)]
pub async fn list_feedback(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let Some(stdb) = state.spacetime_client() else {
        return internal_error(&state, "spacetimedb client not configured");
    };
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let all_events = match state.read_all_events_async().await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
    };
    let owned_run_ids = owned_run_ids_from_events(&all_events, &principal);

    match stdb.query_feedback_async(None).await {
        Ok(rows) => {
            let items = feedback_rows_to_json_for_owned_runs(rows, &owned_run_ids);
            ok(json!({
                "request_id": state.next_request_id(),
                "feedback": items
            }))
        }
        Err(err) => internal_error(&state, format!("spacetimedb query feedback failed: {err}")),
    }
}

// ─── New resource endpoints ────────────────────────────────────────────────

#[tracing::instrument(name = "api.list_runs", skip_all)]
pub async fn list_runs(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let all_events = match state.read_all_events_async().await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
    };

    let mut by_run: std::collections::HashMap<String, Vec<TimelineEvent>> =
        std::collections::HashMap::new();
    for event in all_events {
        by_run.entry(event.run_id.clone()).or_default().push(event);
    }

    let now_ms = now_epoch_ms();
    let mut runs: Vec<Value> = by_run
        .into_iter()
        .filter_map(|(run_id, events)| {
            // Skip runs that have been deleted.
            if events.iter().any(|e| e.kind == "run_deleted") {
                return None;
            }
            let created_event = events.iter().find(|e| e.kind == "run_created")?;
            let created_payload = parse_payload(&created_event.payload);
            let event_principal = created_payload
                .get("principal_key")
                .and_then(|v| v.as_str())
                .unwrap_or("public");
            if event_principal != principal {
                return None;
            }
            let (mode, model, source_uri, created_at_ms, updated_at_ms) =
                extract_run_metadata(&events);
            let snapshot = state.run_runtime_snapshot(&run_id, now_ms)?;
            let status = snapshot.state.as_lowercase_str();
            Some(json!({
                "run_id": run_id,
                "status": status,
                "mode": mode,
                "model": model,
                "source_uri": source_uri,
                "created_at": ms_to_iso(created_at_ms),
                "updated_at": ms_to_iso(updated_at_ms),
            }))
        })
        .collect();

    // Stable ordering by creation time.
    runs.sort_by(|a, b| {
        let ca = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let cb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        ca.cmp(cb)
    });

    ok(json!(runs))
}

#[tracing::instrument(name = "api.get_run", skip_all, fields(run_id))]
pub async fn get_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid run request") {
        return error;
    }
    let snapshot = match load_run_snapshot(&state, &headers, &run_id) {
        Ok(s) => s,
        Err(error) => return error,
    };
    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error,
    };
    if events.iter().any(|e| e.kind == "run_deleted") {
        return not_found_error(
            &state,
            "run_id was not found",
            vec![field_error("run_id", run_id.to_string())],
        );
    }
    let (mode, model, source_uri, created_at_ms, updated_at_ms) = extract_run_metadata(&events);
    let status = snapshot.state.as_lowercase_str();
    ok(json!({
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "model": model,
        "source_uri": source_uri,
        "created_at": ms_to_iso(created_at_ms),
        "updated_at": ms_to_iso(updated_at_ms),
    }))
}

#[tracing::instrument(name = "api.delete_run", skip_all, fields(run_id))]
pub async fn delete_run(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid delete request") {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }
    let request_id = state.next_request_id();
    if let Err(err) = state
        .append_run_event_async(&run_id, "run_deleted", json!({ "request_id": request_id }))
        .await
    {
        return internal_error(&state, format!("failed to append run_deleted event: {err}"));
    }
    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
    }))
}

/// GET /v1/files/{filename}
///
/// Serve a file by bare filename from any directory listed in `VIDARAX_INGEST_FILE_ROOTS`.
/// Uploaded files in the dedicated upload root are private to the uploader
/// principal via a filename prefix. Other configured ingest roots are
/// operator-trusted shared roots and do not use upload ownership prefixes.
///
/// Security: only files whose canonical path starts with one of the allowed ingest roots
/// are served.  Path traversal (`../`) is rejected by the canonicalization check.
pub async fn serve_file(
    State(state): State<AppState>,
    Path(filename): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::Response;

    // Reject filenames with path separators or obvious traversal attempts.
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("invalid filename"))
            .unwrap();
    }
    if !allowed_served_file_extension(&filename) {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("unsupported file type"))
            .unwrap();
    }
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let owner_prefix = upload_owner_prefix_from_principal(&principal);

    // Search the dedicated upload root plus each operator-configured root.
    for root in file_serve_roots(&state) {
        let candidate = root.join(&filename);
        // Canonicalize to resolve any symlinks and check containment.
        let canonical = match candidate.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
        };
        // Security: ensure the resolved path is still inside the allowed root.
        let root_canonical = match root.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
        };
        if !canonical.starts_with(&root_canonical) {
            continue;
        }
        if uploaded_path_requires_owner_prefix(&state, &canonical) {
            let Some(resolved_filename) =
                upload_root_regular_file_name_for_visibility(&candidate, &canonical)
            else {
                continue;
            };
            if !filename_is_visible_to_principal(resolved_filename, &principal, &owner_prefix) {
                continue;
            }
        }
        // Read the file and stream it back.
        let data = match tokio::fs::read(&canonical).await {
            Ok(d) => d,
            Err(_) => continue,
        };
        let mime = if filename.ends_with(".mp4") {
            "video/mp4"
        } else if filename.ends_with(".webm") {
            "video/webm"
        } else if filename.ends_with(".mov") {
            "video/quicktime"
        } else if filename.ends_with(".avi") {
            "video/x-msvideo"
        } else {
            "application/octet-stream"
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime)
            .header(header::CONTENT_LENGTH, data.len())
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(data))
            .unwrap();
    }

    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Body::from("file not found"))
        .unwrap()
}

pub async fn upload_file(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let owner_prefix = upload_owner_prefix_from_principal(&principal);
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(err) => {
                return internal_error(&state, format!("multipart error: {err}"));
            }
        };
        if field.name() != Some("file") {
            continue;
        }
        let raw_name = field.file_name().unwrap_or("upload").to_string();
        // Sanitize: keep only safe characters.
        let safe_name: String = raw_name
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_'))
            .collect();
        let safe_name = if safe_name.is_empty() {
            "upload".to_string()
        } else {
            safe_name
        };
        if !allowed_served_file_extension(&safe_name) {
            return validation_error(
                &state,
                "invalid upload request",
                vec![field_error("file", "unsupported file type".to_string())],
            );
        }
        let safe_name = format!("{owner_prefix}{safe_name}");
        let Some(upload_root) = shared_upload_root() else {
            return internal_error(&state, "failed to prepare upload root".to_string());
        };
        let dest = upload_root.join(&safe_name);
        let data = match field.bytes().await {
            Ok(data) => data,
            Err(err) => {
                return internal_error(&state, format!("failed to read upload field: {err}"));
            }
        };
        if let Err(err) = tokio::fs::write(&dest, &data).await {
            return internal_error(&state, format!("failed to write upload: {err}"));
        }
        if let Err(message) = validate_uploaded_media_container(&dest, &data).await {
            let _ = tokio::fs::remove_file(&dest).await;
            return validation_error(
                &state,
                "invalid upload request",
                vec![field_error("file", message)],
            );
        }
        return ok(json!({ "file_path": dest.display().to_string() }));
    }
    validation_error(
        &state,
        "upload request missing file field",
        vec![field_error(
            "file",
            "no file field found in multipart form".to_string(),
        )],
    )
}

// ─── Run metadata helpers ──────────────────────────────────────────────────

fn extract_run_metadata(events: &[TimelineEvent]) -> (String, String, String, u64, u64) {
    let created = events.iter().find(|e| e.kind == "run_created");
    let created_payload = created
        .map(|e| parse_payload(&e.payload))
        .unwrap_or_default();
    let mode = created_payload
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = created_payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let source_uri = events
        .iter()
        .find(|e| e.kind == "ingest_received")
        .and_then(|e| {
            let p = parse_payload(&e.payload);
            p.get("source_uri")
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
        })
        .unwrap_or_default();
    let created_at_ms = created.map(|e| e.pts_ms).unwrap_or(0);
    let updated_at_ms = events
        .iter()
        .map(|e| e.pts_ms)
        .max()
        .unwrap_or(created_at_ms);
    (mode, model, source_uri, created_at_ms, updated_at_ms)
}

/// Convert a Unix epoch millisecond timestamp to an ISO 8601 string.
/// Uses Howard Hinnant's civil_from_days algorithm; no external dependencies.
fn ms_to_iso(ms: u64) -> String {
    let total_secs = ms / 1000;
    let millis = ms % 1000;
    let time_of_day = total_secs % 86400;
    let days = total_secs / 86400;

    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;

    // civil_from_days: https://howardhinnant.github.io/date_algorithms.html
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z")
}

fn validate_run_id_or_error(
    state: &AppState,
    run_id: &str,
    context: &'static str,
) -> Option<ApiResponse> {
    (!validate_run_id(run_id)).then(|| {
        validation_error(
            state,
            context,
            vec![field_error(
                "run_id",
                "run_id must match run-<16 or 32 hex chars>".to_string(),
            )],
        )
    })
}

fn owned_run_ids_from_events(events: &[TimelineEvent], principal: &str) -> HashSet<String> {
    let mut owned = HashSet::new();
    let mut deleted = HashSet::new();
    for event in events {
        match event.kind.as_str() {
            "run_created" => {
                let payload = parse_payload(&event.payload);
                let event_principal = payload
                    .get("principal_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("public");
                if event_principal == principal {
                    owned.insert(event.run_id.clone());
                }
            }
            "run_deleted" => {
                deleted.insert(event.run_id.clone());
            }
            _ => {}
        }
    }
    for run_id in deleted {
        owned.remove(&run_id);
    }
    owned
}

fn label_map_key_from_principal(principal: &str) -> Option<&str> {
    (principal != "public").then_some(principal)
}

fn feedback_rows_to_json_for_owned_runs(
    rows: Vec<crate::spacetime_client::FeedbackRow>,
    owned_run_ids: &HashSet<String>,
) -> Vec<Value> {
    rows.into_iter()
        .filter(|r| owned_run_ids.contains(&r.run_id))
        .map(|r| {
            json!({
                "id": r.id,
                "run_id": r.run_id,
                "session_id": r.session_id,
                "rating": r.rating,
                "category": r.category,
                "feedback": r.feedback,
                "timestamp_micros": r.timestamp_micros,
            })
        })
        .collect()
}

fn upload_owner_prefix_from_principal(principal: &str) -> String {
    if principal == "public" {
        // Public/open mode is a shared, development-only upload namespace. It
        // provides no tenant isolation; authenticated callers use a namespace
        // derived from the API-key principal. One API key = one tenant; for
        // sub-tenant isolation issue separate keys.
        return "public__".to_string();
    }
    principal
        .strip_prefix("api-key:")
        .filter(|_| !principal.is_empty())
        .map(|_| format!("{}__", strong_hash_hex(principal)))
        .unwrap_or_else(|| "public__".to_string())
}

fn filename_is_visible_to_principal(filename: &str, principal: &str, owner_prefix: &str) -> bool {
    filename.starts_with(owner_prefix) || (principal == "public" && !filename.contains("__"))
}

fn uploaded_path_requires_owner_prefix(_state: &AppState, canonical: &FsPath) -> bool {
    let Some(upload_root) = shared_upload_root() else {
        return false;
    };
    canonical.starts_with(&upload_root)
}

fn upload_root_regular_file_name_for_visibility<'a>(
    requested_path: &FsPath,
    canonical: &'a FsPath,
) -> Option<&'a str> {
    let file_type = std::fs::symlink_metadata(requested_path).ok()?.file_type();
    if !file_type.is_file() {
        return None;
    }
    canonical.file_name().and_then(|name| name.to_str())
}

fn allowed_served_file_extension(filename: &str) -> bool {
    let Some((_, ext)) = filename.rsplit_once('.') else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "webm" | "mov" | "avi"
    )
}

fn enforce_file_source_visibility(
    state: &AppState,
    headers: &HeaderMap,
    requested_source_uri: &str,
    source: &InputSource,
    context: &'static str,
) -> Result<(), ApiResponse> {
    let InputSource::FilePath(path) = source else {
        return Ok(());
    };
    let Ok(canonical) = FsPath::new(path).canonicalize() else {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "source_uri",
                "source_uri file path is invalid or does not exist".to_string(),
            )],
        ));
    };
    if !uploaded_path_requires_owner_prefix(state, &canonical) {
        // Non-upload ingest roots are admin-configured and trusted shared media
        // roots.
        return Ok(());
    }
    let requested_path = requested_file_path_for_visibility(requested_source_uri)
        .unwrap_or_else(|| PathBuf::from(path));
    let Some(filename) = upload_root_regular_file_name_for_visibility(&requested_path, &canonical)
    else {
        return Err(validation_error(
            state,
            context,
            vec![field_error(
                "source_uri",
                "source_uri file is not visible to the caller".to_string(),
            )],
        ));
    };
    let principal = state.security_policy().principal_key_from_headers(headers);
    let owner_prefix = upload_owner_prefix_from_principal(&principal);
    // Legacy unprefixed files in the upload temp root are not auto-claimed by
    // authenticated callers; open-mode `public` remains a shared dev namespace.
    if filename_is_visible_to_principal(filename, &principal, &owner_prefix) {
        return Ok(());
    }
    Err(validation_error(
        state,
        context,
        vec![field_error(
            "source_uri",
            "source_uri file is not visible to the caller".to_string(),
        )],
    ))
}

fn requested_file_path_for_visibility(source_uri: &str) -> Option<PathBuf> {
    let trimmed = source_uri.trim();
    if trimmed.contains("://") {
        let url = reqwest::Url::parse(trimmed).ok()?;
        if url.scheme() != "file" {
            return None;
        }
        return url.to_file_path().ok();
    }
    Some(PathBuf::from(trimmed))
}

fn shared_upload_root() -> Option<PathBuf> {
    let root = std::env::temp_dir().join(UPLOAD_DIR_NAME);
    std::fs::create_dir_all(&root).ok()?;
    root.canonicalize().ok()
}

fn ingest_file_roots_with_upload_root(state: &AppState) -> Vec<PathBuf> {
    let mut roots = Vec::with_capacity(state.ingest_file_roots().len() + 1);
    if let Some(upload_root) = shared_upload_root() {
        roots.push(upload_root);
    }
    roots.extend_from_slice(state.ingest_file_roots());
    roots
}

fn file_serve_roots(state: &AppState) -> Vec<PathBuf> {
    ingest_file_roots_with_upload_root(state)
}

async fn validate_uploaded_media_container(path: &FsPath, data: &[u8]) -> Result<(), String> {
    let trimmed = data
        .iter()
        .copied()
        .skip_while(|b| b.is_ascii_whitespace())
        .take(7)
        .collect::<Vec<_>>();
    if trimmed.eq_ignore_ascii_case(b"#EXTM3U") {
        return Err("uploaded file must be a media container, not a playlist manifest".to_string());
    }

    let path = path.to_path_buf();
    let probe = tokio::task::spawn_blocking(move || validate_uploaded_media_container_file(&path));
    match tokio::time::timeout(UPLOAD_MEDIA_PROBE_TIMEOUT + Duration::from_secs(1), probe).await {
        Ok(Ok(result)) => result,
        Ok(Err(_join_err)) => Err("failed to inspect uploaded media".to_string()),
        Err(_elapsed) => Err("uploaded media inspection timed out".to_string()),
    }
}

fn validate_uploaded_media_container_file(path: &FsPath) -> Result<(), String> {
    // Extension checks are not a security boundary. Probe the just-written file
    // with file-only protocols and reject playlist/manifest demuxers where raw
    // uploaded media is expected.
    let mut command = Command::new(vidarax_core::ingest::ffprobe_path());
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-show_entries",
            "format=format_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path);
    let output = run_command_with_timeout(
        &mut command,
        UPLOAD_MEDIA_PROBE_TIMEOUT,
        "uploaded media inspection timed out",
    )?;
    if !output.status.success() {
        return Err("uploaded file must be a valid media container".to_string());
    }
    let format_name = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_ascii_lowercase();
    if format_name.is_empty() {
        return Err("uploaded file must declare a media container format".to_string());
    }
    if format_name
        .split(',')
        .any(|name| matches!(name, "hls" | "concat"))
    {
        return Err("uploaded file must be a media container, not a playlist manifest".to_string());
    }
    Ok(())
}

#[derive(Debug)]
struct TimedCommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
}

fn run_command_with_timeout(
    command: &mut Command,
    timeout: Duration,
    timeout_message: &'static str,
) -> Result<TimedCommandOutput, String> {
    command.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = command
        .spawn()
        .map_err(|_| "failed to inspect uploaded media".to_string())?;
    let started = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                if let Some(mut pipe) = child.stdout.take() {
                    pipe.read_to_end(&mut stdout)
                        .map_err(|_| "failed to inspect uploaded media".to_string())?;
                }
                return Ok(TimedCommandOutput { status, stdout });
            }
            Ok(None) => {
                if started.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(timeout_message.to_string());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("failed to inspect uploaded media".to_string());
            }
        }
    }
}

async fn load_existing_events(
    state: &AppState,
    run_id: &str,
) -> Result<Vec<TimelineEvent>, ApiResponse> {
    state
        .read_run_events_async(run_id)
        .await
        .map_err(|err| internal_error(state, format!("failed to read run events: {err}")))
}

fn load_run_snapshot(
    state: &AppState,
    headers: &HeaderMap,
    run_id: &str,
) -> Result<crate::state::RunRuntimeSnapshot, ApiResponse> {
    let Some(snapshot) = state.run_runtime_snapshot(run_id, now_epoch_ms()) else {
        return Err(not_found_error(
            state,
            "run_id was not found",
            vec![field_error("run_id", run_id.to_string())],
        ));
    };
    let requested = state.security_policy().principal_key_from_headers(headers);
    // Principal ownership is introduced in this release; pre-ownership runs
    // without `principal_key` are public only. See docs/security.md.
    if snapshot.principal_key == requested {
        return Ok(snapshot);
    }
    Err(not_found_error(
        state,
        "run_id was not found",
        vec![field_error("run_id", run_id.to_string())],
    ))
}

fn parse_payload(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!({ "raw": raw }))
}

struct RuntimeAvailability {
    status: &'static str,
    providers: Vec<String>,
}

fn runtime_model_availability(
    provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    is_saturated: bool,
) -> RuntimeAvailability {
    let Some(provider) = provider else {
        return RuntimeAvailability {
            status: "unavailable",
            providers: Vec::new(),
        };
    };
    // With the abstract provider layer, we report the top-level kind as
    // available.  Health-checking individual backends is deferred to a future
    // backends health endpoint.
    let kind_name = provider.kind().name().to_string();
    let status = if is_saturated { "saturated" } else { "ready" };
    RuntimeAvailability {
        status,
        providers: vec![kind_name],
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn field_error(field: &'static str, message: String) -> FieldError {
    FieldError { field, message }
}

fn parse_provider(raw: Option<&str>) -> Result<ProviderKind, &'static str> {
    match raw.unwrap_or("vllm").to_ascii_lowercase().as_str() {
        "vllm" => Ok(ProviderKind::Vllm),
        "sglang" => Ok(ProviderKind::Sglang),
        "gemini" => Ok(ProviderKind::Gemini),
        _ => Err("primary_provider must be one of: vllm, sglang, gemini"),
    }
}

fn provider_name(kind: ProviderKind) -> &'static str {
    kind.name()
}
