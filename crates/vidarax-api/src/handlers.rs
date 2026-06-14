use axum::extract::{Multipart, Path, Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use std::cmp::min;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::task::JoinSet;
use vidarax_contracts::models::{
    fallback_candidates, REQUIRED_MEDIUM_MODELS, REQUIRED_SMALL_MODELS,
};
use vidarax_core::gate::{FrameSignal, GateConfig, GateEventType};
use vidarax_core::ingest::{
    compute_semantic_frame_indices, extract_video_clip, probe_source_fps, DecodedJpegFrame,
    InputSource, Mp4DecodeConfig,
};
use vidarax_core::pipeline::{FrameMetadata, TwoPassConfig, TwoPassPipeline};
use vidarax_core::provider::{
    InferenceImage, InferenceProvider, InferenceRequest, InferenceVideo, ProviderError, ProviderKind,
};
use vidarax_core::timeline::TimelineEvent;

use crate::ids::validate_run_id;
use crate::models::{
    AnalyzeAnnotations, AnalyzeEvent, AnalyzeFallback, AnalyzeFrameMetadata, AnalyzeFramesRequest,
    AnalyzeFramesResponse, AnalyzeMarker, AnalyzeObject, AnalyzeTrace, AnalyzeWindow,
    CreateRunRequest, CreateRunResponse, FieldError, InferBatchItemError, InferBatchItemResult,
    InferBatchRequest, InferBatchResponse, InferRequest, InferResponse, ModelCatalogItem,
    ModelCatalogResponse, RealtimeReasonRequest, RealtimeReasonResponse, SamplingPolicy,
    SearchHit, SearchRequest, SearchResponse,
};
use crate::response::{
    conflict_error, internal_error, not_found_error, ok, validation_error, ApiResponse,
};
use crate::semantic::{build_marker_lifecycle, MarkerConfig, MarkerInput, SemanticMarker};
use crate::state::AppState;
use crate::validation::{normalize_mode, normalize_model};

const HEADER_API_KEY: &str = "x-api-key";
const HEADER_TENANT_ID: &str = "x-tenant-id";

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
    let principal = principal_key_from_headers(&headers);
    let active = state.count_active_runs_for_principal(&principal, now_epoch_ms());
    if active >= state.active_stream_limit() {
        return conflict_error(
            &state,
            "active stream limit exceeded",
            vec![field_error(
                "run_id",
                format!(
                    "principal exceeded active stream limit: {}/{}",
                    active,
                    state.active_stream_limit()
                ),
            )],
        );
    }

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
        let decode_source =
            match InputSource::parse_and_validate(&source_uri, state.ingest_file_roots()) {
                Ok(source) => source,
                Err(message) => {
                    return validation_error(
                        &state,
                        "invalid ingest request",
                        vec![field_error("source_uri", message)],
                    );
                }
            };
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

    let tenant_id = header_value(&headers, HEADER_TENANT_ID);
    let stream_id = payload.stream_id.unwrap_or_else(|| "stream-0".to_string());
    let request_id = state.next_request_id();
    let trace_id = payload
        .trace_id
        .unwrap_or_else(|| format!("trace-{}", &request_id[4..]));
    let mut pipeline = TwoPassPipeline::new(
        TwoPassConfig {
            window_size,
            segment_ms,
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
                tenant_id,
                &run_id,
                &stream_id,
                &mode,
                &model,
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

    let mode = match normalize_mode(payload.mode) {
        Ok(mode) => mode,
        Err(message) => {
            return validation_error(
                &state,
                "invalid realtime reason request",
                vec![field_error("mode", message)],
            );
        }
    };
    let model = match normalize_model(Some(payload.model)) {
        Ok(Some(model)) => model,
        Ok(None) => unreachable!("model is required"),
        Err(message) => {
            return validation_error(
                &state,
                "invalid realtime reason request",
                vec![field_error("model", message)],
            );
        }
    };
    let sampling_policy = match SamplingPolicy::parse(payload.sampling_policy.as_deref()) {
        Ok(policy) => policy,
        Err(message) => {
            return validation_error(
                &state,
                "invalid realtime reason request",
                vec![field_error("sampling_policy", message.to_string())],
            );
        }
    };
    let max_frames = payload.max_frames.unwrap_or(120_000);
    if !(1..=500_000).contains(&max_frames) {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "max_frames",
                "max_frames must be in [1, 500000]".to_string(),
            )],
        );
    }
    let chunk_size = payload.chunk_size.unwrap_or(25);
    if !(5..=500).contains(&chunk_size) {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "chunk_size",
                "chunk_size must be in [5, 500]".to_string(),
            )],
        );
    }
    let semantic_inference = payload.semantic_inference.unwrap_or(true);
    let semantic_frames_per_chunk = payload.semantic_frames_per_chunk.unwrap_or(2);
    if !(1..=4).contains(&semantic_frames_per_chunk) {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_frames_per_chunk",
                "semantic_frames_per_chunk must be in [1, 4]".to_string(),
            )],
        );
    }
    let semantic_timeout_ms = payload.semantic_timeout_ms.unwrap_or(1_500);
    if !(100..=120_000).contains(&semantic_timeout_ms) {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_timeout_ms",
                "semantic_timeout_ms must be in [100, 120000]".to_string(),
            )],
        );
    }
    let semantic_prompt = payload.semantic_prompt.unwrap_or_else(|| {
        "You are classifying a short video chunk. Return strict JSON with keys: event_type, object_label, summary, description, confidence (0..1). event_type must be one of: scene_cut, artifact_suspected, keyframe_keep, context_observation.".to_string()
    });
    if semantic_prompt.trim().is_empty() || semantic_prompt.len() > 4_096 {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_prompt",
                "semantic_prompt must be non-empty and <= 4096 bytes".to_string(),
            )],
        );
    }


    let tiered_config = {
        use vidarax_core::tiered_vlm::TieredVlmConfig;
        let first = payload.first_pass_model.as_deref().unwrap_or(&model);
        let second = payload.second_pass_model.as_deref().unwrap_or(&model);
        let threshold = payload.second_pass_threshold.unwrap_or(0.7);
        TieredVlmConfig {
            first_pass_model: Arc::from(first),
            second_pass_model: Arc::from(second),
            second_pass_threshold: threshold.clamp(0.0, 1.0),
            second_pass_max_tokens: 256,
        }
    };

    let decode_source =
        match InputSource::parse_and_validate(&payload.source_uri, state.ingest_file_roots()) {
            Ok(source) => source,
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid realtime reason request",
                    vec![field_error("source_uri", message)],
                );
            }
        };

    let video_clip_mode = payload.video_clip_mode.unwrap_or(false);
    let video_clip_duration_s = payload.video_clip_duration_s.unwrap_or(0.5);
    if video_clip_mode {
        if !video_clip_duration_s.is_finite()
            || video_clip_duration_s <= 0.0
            || video_clip_duration_s > 60.0
        {
            return validation_error(
                &state,
                "invalid realtime reason request",
                vec![field_error(
                    "video_clip_duration_s",
                    "video_clip_duration_s must be in (0, 60]".to_string(),
                )],
            );
        }
    }

    let fixed_fps = payload.fixed_fps.unwrap_or(1.0);
    if sampling_policy == SamplingPolicy::Fixed && !(0.2..=120.0).contains(&fixed_fps) {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "fixed_fps",
                "fixed_fps must be in [0.2, 120.0]".to_string(),
            )],
        );
    }

    let semantic_decode_enabled = semantic_inference && state.provider().is_some();
    // Keep a clone of the source for video clip extraction in Phase 1 (decode_source
    // is moved into the spawn_blocking closure below).
    let decode_source_ref = decode_source.clone();
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

            // Pre-compute which frames go to VLM (pure math, zero I/O).
            // In video_clip_mode, JPEG decoding is skipped here; clips are
            // extracted per-chunk during Phase 1 below.
            let decoded_jpegs = if semantic_decode_enabled && !video_clip_mode {
                let indices = compute_semantic_frame_indices(
                    decoded.frame_signals.len(),
                    chunk_size,
                    semantic_frames_per_chunk,
                );
                // Pass 2: selective JPEG — only the ~4% of frames needed
                let jpegs = decode_pipeline.decode_jpegs(
                    &decode_source,
                    sample_fps,
                    &indices,
                    max_frames as usize,
                )?;
                // Build lookup by frame_index for O(1) access during chunking
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
    let tenant_id = header_value(&headers, HEADER_TENANT_ID);
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
            window_size: payload.window_size.unwrap_or(16),
            segment_ms: payload.segment_ms.unwrap_or(250),
        },
        GateConfig::default(),
    );

    let mut metadata = Vec::with_capacity(decoded.frame_signals.len());
    let mut marker_inputs = Vec::with_capacity(decoded.frame_signals.len());
    let mut chunk_lags = Vec::new();
    let providers = state.provider().cloned();
    let semantic_available = semantic_inference && providers.is_some();
    let semantic_segment_ms = payload.segment_ms.unwrap_or(250);
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

    // --- Phase 1: serial pipeline analysis + chunk preparation ---
    // TwoPassPipeline carries mutable state, so analysis must remain sequential.
    struct ChunkPrep {
        analyzed: Vec<FrameMetadata>,
        frame_offset: usize,
        chunk_jpegs: Vec<DecodedJpegFrame>,
        /// MP4 clip bytes for this chunk, present only when video_clip_mode is true.
        chunk_video_clip: Option<Vec<u8>>,
        pts_start_ms: u64,
        pts_end_ms: u64,
        chunk_len: usize,
        started: Instant,
    }
    let mut chunk_preps: Vec<ChunkPrep> = Vec::new();
    for (chunk_idx, chunk) in decoded.frame_signals.chunks(chunk_size).enumerate() {
        let started = Instant::now();
        let analyzed = pipeline.analyze_batch(chunk).to_vec();
        let frame_offset = chunk_idx * chunk_size;
        let chunk_jpegs: Vec<DecodedJpegFrame> = decoded_jpegs
            .as_ref()
            .map(|lookup| {
                let mut jpegs: Vec<DecodedJpegFrame> = (frame_offset..frame_offset + chunk.len())
                    .filter_map(|idx| lookup.get(&(idx as u64)).cloned())
                    .collect();
                jpegs.sort_by_key(|f| f.frame_index);
                jpegs
            })
            .unwrap_or_default();

        // In video_clip_mode, extract an MP4 segment covering this chunk's
        // time window.  Clip start is derived from actual frame PTS so the
        // VLM sees footage matching the decoded frames, not sequential offsets.
        let pts_start_ms_for_clip = chunk.first().map(|f| f.pts_ms).unwrap_or(0);
        let chunk_video_clip: Option<Vec<u8>> = if video_clip_mode && semantic_decode_enabled {
            let clip_start = pts_start_ms_for_clip as f32 / 1000.0;
            let source_for_clip = decode_source_ref.clone();
            let duration = video_clip_duration_s;
            match tokio::task::spawn_blocking(move || {
                extract_video_clip(&source_for_clip, clip_start, duration)
            })
            .await
            {
                Ok(Ok(bytes)) => Some(bytes),
                Ok(Err(err)) => {
                    tracing::warn!(
                        chunk_idx,
                        clip_start_ms = pts_start_ms_for_clip,
                        duration = video_clip_duration_s,
                        error = %err,
                        "video clip extraction failed for chunk; falling back to no-video"
                    );
                    None
                }
                Err(err) => {
                    tracing::warn!(chunk_idx, error = %err, "clip extraction task panicked");
                    None
                }
            }
        } else {
            None
        };

        chunk_preps.push(ChunkPrep {
            started,
            analyzed,
            frame_offset,
            chunk_jpegs,
            chunk_video_clip,
            pts_start_ms: chunk.first().map(|f| f.pts_ms).unwrap_or(0),
            pts_end_ms: chunk.last().map(|f| f.pts_ms).unwrap_or(0),
            chunk_len: chunk.len(),
        });
    }

    // --- Phase 2: VLM inference ---
    let num_chunks = chunk_preps.len();
    let mut semantic_results: Vec<Option<ChunkSemanticResult>> = (0..num_chunks).map(|_| None).collect();
    let mut task_end_times: Vec<Instant> = vec![Instant::now(); num_chunks];
    let visual_diff = payload.visual_diff.unwrap_or(false);
    let temporal_chain = visual_diff || payload.temporal_chain.unwrap_or(false);

    if semantic_available {
        let guided_json_str: Option<String> = payload
            .output_schema
            .as_ref()
            .and_then(|s| serde_json::to_string(s).ok());

        if temporal_chain {
            // Sequential mode: each chunk gets the previous chunk's VLM output
            // as context. When visual_diff is on, also sends the previous
            // chunk's representative frame as a second image.
            let mut last_description = String::new();
            let mut last_pts_ms: u64 = 0;
            let mut last_jpeg: Option<Vec<u8>> = None;

            for (chunk_idx, prep) in chunk_preps.iter().enumerate() {
                let prompt_with_context = if last_description.is_empty() {
                    semantic_prompt.clone()
                } else {
                    format!(
                        "{semantic_prompt}\n[previous_state ({last_pts_ms}ms): {}]",
                        &last_description[..last_description.len().min(200)]
                    )
                };

                let prev_jpeg_ref = if visual_diff { last_jpeg.as_deref() } else { None };

                let result = infer_chunk_semantics(
                    providers.clone(),
                    true,
                    &model,
                    &prompt_with_context,
                    semantic_timeout_ms,
                    semantic_frames_per_chunk,
                    &prep.chunk_jpegs,
                    prep.frame_offset as u64,
                    prep.pts_start_ms,
                    prep.pts_end_ms,
                    tiered_config.clone(),
                    guided_json_str.clone(),
                    prev_jpeg_ref,
                    prep.chunk_video_clip.clone(),
                )
                .await;

                // Track last JPEG for visual_diff.
                if visual_diff {
                    if let Some(frame) = select_semantic_images(&prep.chunk_jpegs, 1).first() {
                        last_jpeg = Some(frame.jpeg_bytes.clone());
                    }
                }

                if let Some(ref raw) = result.raw_output {
                    let s = raw.to_string();
                    if s.len() > 4 {
                        last_description.clear();
                        last_description.push_str(&s[..s.len().min(200)]);
                        last_pts_ms = prep.pts_end_ms;
                    }
                } else if let Some(ref overlay) = result.overlay {
                    last_description.clear();
                    last_description.push_str(&overlay.description[..overlay.description.len().min(200)]);
                    last_pts_ms = prep.pts_end_ms;
                }

                semantic_results[chunk_idx] = Some(result);
                task_end_times[chunk_idx] = Instant::now();
            }
        } else {
            // Parallel mode (default): fire all chunks concurrently.
            let vlm_concurrency = payload.vlm_concurrency.unwrap_or(4).clamp(1, 64);
            let sem = Arc::new(tokio::sync::Semaphore::new(vlm_concurrency));
            let mut join_set: JoinSet<(usize, ChunkSemanticResult, Instant)> = JoinSet::new();

            for (chunk_idx, prep) in chunk_preps.iter().enumerate() {
                let providers_c = providers.clone();
                let prompt_c = semantic_prompt.clone();
                let chunk_jpegs_c = prep.chunk_jpegs.clone();
                let chunk_video_clip_c = prep.chunk_video_clip.clone();
                let sem_c = Arc::clone(&sem);
                let frame_offset = prep.frame_offset as u64;
                let pts_start_ms = prep.pts_start_ms;
                let pts_end_ms = prep.pts_end_ms;
                let tiered_config_c = tiered_config.clone();
                let guided_json_c = guided_json_str.clone();
                join_set.spawn(async move {
                    let _permit = sem_c.acquire().await.unwrap();
                    let overlay = infer_chunk_semantics(
                        providers_c,
                        true,
                        model,
                        &prompt_c,
                        semantic_timeout_ms,
                        semantic_frames_per_chunk,
                        &chunk_jpegs_c,
                        frame_offset,
                        pts_start_ms,
                        pts_end_ms,
                        tiered_config_c,
                        guided_json_c,
                        None, // no visual_diff in parallel mode
                        chunk_video_clip_c,
                    )
                    .await;
                    (chunk_idx, overlay, Instant::now())
                });
            }

            while let Some(Ok((idx, result, finished))) = join_set.join_next().await {
                semantic_results[idx] = Some(result);
                task_end_times[idx] = finished;
            }
        }
    }

    // --- Phase 3: sequential post-processing (WAL events, metadata, lag tracking) ---
    for (chunk_idx, prep) in chunk_preps.into_iter().enumerate() {
        let semantic_overlay = semantic_results[chunk_idx]
            .take()
            .unwrap_or_default();
        let finished = task_end_times[chunk_idx];

        if let Some(mut details) =
            semantic_overlay.event_payload(chunk_idx, request_id.as_str(), stream_id.as_str())
        {
            // Attach the index tag so the event can be filtered later.
            if let Some(ref idx) = index_name {
                if let Some(obj) = details.as_object_mut() {
                    obj.insert("index_name".to_string(), serde_json::json!(idx));
                }
            }
            if let Err(err) = state
                .append_run_event_async(&run_id, "semantic_chunk_inferred", details)
                .await
            {
                return internal_error(
                    &state,
                    format!("failed to append semantic_chunk_inferred event: {err}"),
                );
            }
        }

        for frame in prep.analyzed {
            let (row, marker_input) = compose_frame_metadata(
                &state,
                tenant_id,
                &run_id,
                &stream_id,
                &mode,
                &model,
                sampling_policy,
                sample_fps,
                semantic_segment_ms,
                &request_id,
                &trace_id,
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
                &run_id,
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
            return internal_error(
                &state,
                format!("failed to append semantic_chunk_generated event: {err}"),
            );
        }
    }

    let markers = build_marker_lifecycle(&run_id, &stream_id, &marker_inputs, &marker_config)
        .into_iter()
        .map(semantic_marker_to_api_marker)
        .collect::<Vec<_>>();
    for marker in &markers {
        // Embed index_name so marker events can be filtered by index.
        let mut marker_payload = json!(marker);
        if let Some(ref idx) = index_name {
            if let Some(obj) = marker_payload.as_object_mut() {
                obj.insert("index_name".to_string(), serde_json::json!(idx));
            }
        }
        if let Err(err) = state
            .append_run_event_async(&run_id, "marker_emitted", marker_payload)
            .await
        {
            return internal_error(
                &state,
                format!("failed to append marker_emitted event: {err}"),
            );
        }
    }

    let lag_p95_ms = percentile_ms(&chunk_lags, 95);
    let lag_p99_ms = percentile_ms(&chunk_lags, 99);
    if let Err(err) = state
        .append_run_event_async(
            &run_id,
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
        return internal_error(
            &state,
            format!("failed to append analysis_generated event: {err}"),
        );
    }

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
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
        return internal_error(
            &state,
            format!("failed to append run_completed event: {err}"),
        );
    }

    ok(json!(RealtimeReasonResponse {
        request_id,
        run_id,
        generated: metadata.len(),
        markers_emitted: markers.len(),
        decoded_frames: decoded.frame_signals.len(),
        sample_fps,
        lag_p95_ms,
        lag_p99_ms,
        metadata,
        markers,
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
    let availability =
        match tokio::task::spawn_blocking(move || {
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
                .into_iter()
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
                .into_iter()
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
/// This is an MVP implementation: no embedding model is required.  Exact
/// substring matching is O(n) in the number of stored events but is fast
/// enough for the typical WAL sizes encountered in development and staging.
/// A vector-embedding upgrade path is available by storing description
/// embeddings at write time and replacing this scan with a k-NN query.
#[tracing::instrument(name = "api.search", skip_all)]
pub async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> impl IntoResponse {
    // Validate query string.
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

    // Load all events — either run-scoped or global.
    let events = if let Some(ref run_id) = payload.run_id {
        if let Some(error) = validate_run_id_or_error(&state, run_id, "invalid search request") {
            return error;
        }
        match state.read_run_events_async(run_id).await {
            Ok(events) => events,
            Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
        }
    } else {
        match state.read_all_events_async().await {
            Ok(events) => events,
            Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
        }
    };

    let scanned = events.len();

    // Case-insensitive substring search over the `description` field in every
    // event payload.  The lowercase query is computed once.
    let query_lower = query.to_lowercase();

    let mut hits: Vec<SearchHit> = Vec::new();

    for event in events {
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

        // Extract optional index_name for cross-index searches.
        let index_name = payload_val
            .get("index_name")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);

        hits.push(SearchHit {
            seq: event.seq,
            run_id: event.run_id,
            pts_ms: event.pts_ms,
            kind: event.kind,
            description,
            index_name,
        });

        if hits.len() >= limit {
            break;
        }
    }

    let total_hits = hits.len();

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
    for event in events.iter().filter(|e| e.kind == "semantic_chunk_generated") {
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
                        obj.entry("pts_end_ms")
                            .or_insert_with(|| json!(pts_end_ms));
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
                    obj.entry("pts_end_ms")
                        .or_insert_with(|| json!(pts_end_ms));
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
            guided_json: None,
        },
        primary_provider,
    })
}

async fn execute_infer_request(
    state: AppState,
    prepared: PreparedInferRequest,
) -> Result<InferResponse, InferExecutionError> {
    let provider = state
        .provider()
        .cloned()
        .ok_or(InferExecutionError {
            code: "internal_error",
            message: "inference providers are not configured".to_string(),
        })?;

    let request_id = state.next_request_id();
    let started = Instant::now();
    let primary_provider_for_metrics = prepared.primary_provider;
    let request_for_provider = prepared.request.clone();
    let result = match tokio::task::spawn_blocking(move || {
        provider.infer(&request_for_provider)
    })
    .await
    {
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

#[tracing::instrument(name = "api.submit_feedback", skip_all)]
pub async fn submit_feedback(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
    Json(payload): Json<crate::models::FeedbackRequest>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid feedback request") {
        return error;
    }

    if payload.rating > 10 {
        return validation_error(
            &state,
            "invalid feedback payload",
            vec![field_error("rating", "rating must be between 0 and 10".to_string())],
        );
    }
    if payload.category.is_empty() {
        return validation_error(
            &state,
            "invalid feedback payload",
            vec![field_error("category", "category must not be empty".to_string())],
        );
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
pub async fn list_feedback(State(state): State<AppState>) -> impl IntoResponse {
    let Some(stdb) = state.spacetime_client() else {
        return internal_error(&state, "spacetimedb client not configured");
    };

    match stdb.query_feedback_async(None).await {
        Ok(rows) => {
            let items: Vec<Value> = rows
                .into_iter()
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
                .collect();
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
pub async fn list_runs(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let principal = principal_key_from_headers(&headers);
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
        .append_run_event_async(
            &run_id,
            "run_deleted",
            json!({ "request_id": request_id }),
        )
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
/// This allows the browser to load videos that were uploaded to the server's temp directory
/// via the `/v1/upload` endpoint, which returns an absolute file path.
///
/// Security: only files whose canonical path starts with one of the allowed ingest roots
/// are served.  Path traversal (`../`) is rejected by the canonicalization check.
pub async fn serve_file(
    State(state): State<AppState>,
    Path(filename): Path<String>,
) -> impl IntoResponse {
    use axum::http::{header, StatusCode};
    use axum::response::Response;
    use axum::body::Body;

    // Reject filenames with path separators or obvious traversal attempts.
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("invalid filename"))
            .unwrap();
    }

    // Search each allowed root for a file with this name.
    for root in state.ingest_file_roots() {
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
    mut multipart: Multipart,
) -> impl IntoResponse {
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
        let dest = std::env::temp_dir().join(&safe_name);
        let data = match field.bytes().await {
            Ok(data) => data,
            Err(err) => {
                return internal_error(&state, format!("failed to read upload field: {err}"));
            }
        };
        if let Err(err) = tokio::fs::write(&dest, &data).await {
            return internal_error(&state, format!("failed to write upload: {err}"));
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

    format!(
        "{year:04}-{month:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z"
    )
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
    let requested = principal_key_from_headers(headers);
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

struct DecodedSignals {
    signals: Vec<FrameSignal>,
    sampling_policy: SamplingPolicy,
    sample_fps: f32,
}

#[derive(Debug, Clone)]
struct SemanticOverlay {
    event_type: String,
    object_label: String,
    summary: String,
    description: String,
    confidence: f32,
}

#[derive(Debug, Default)]
struct ChunkSemanticResult {
    overlay: Option<SemanticOverlay>,
    /// Raw VLM output text when using custom `output_schema` (passthrough mode).
    raw_output: Option<Value>,
    provider: Option<String>,
    provider_fallback_used: bool,
    used_fallback: bool,
    error: Option<String>,
    attempted: bool,
    finish_reason: Option<String>,
}

impl ChunkSemanticResult {
    fn event_payload(&self, chunk_idx: usize, request_id: &str, stream_id: &str) -> Option<Value> {
        self.attempted.then(|| {
            json!({
                "request_id": request_id,
                "stream_id": stream_id,
                "chunk_index": chunk_idx,
                "provider": self.provider,
                "provider_fallback_used": self.provider_fallback_used,
                "semantic_fallback_used": self.used_fallback,
                "semantic_error": self.error,
                "event_type": self.overlay.as_ref().map(|o| o.event_type.clone()),
                "object_label": self.overlay.as_ref().map(|o| o.object_label.clone()),
                "confidence": self.overlay.as_ref().map(|o| o.confidence),
                "raw_output": self.raw_output,
            })
        })
    }
}

fn load_decoded_signals_from_events(events: &[TimelineEvent]) -> Result<DecodedSignals, String> {
    // We consume the most recent decode event so repeated ingest calls can safely supersede stale
    // frame sets without changing query semantics for analyze.
    let Some(decoded_event) = events
        .iter()
        .rev()
        .find(|event| event.kind == "frames_decoded")
    else {
        return Err("frames must be provided when no decoded ingest frames exist".to_string());
    };

    let payload = serde_json::from_str::<Value>(&decoded_event.payload)
        .map_err(|_| "decoded ingest payload is invalid json".to_string())?;
    let sampling_policy = SamplingPolicy::parse(
        payload
            .get("sampling_policy")
            .and_then(|value| value.as_str()),
    )
    .map_err(ToString::to_string)?;
    let Some(signals) = payload.get("signals").and_then(|value| value.as_array()) else {
        return Err("decoded ingest payload is missing signals array".to_string());
    };
    if signals.is_empty() {
        return Err("decoded ingest payload contains no frame signals".to_string());
    }
    if signals.len() > 500_000 {
        return Err("decoded ingest frame signals exceed limit of 500000".to_string());
    }

    let mut out = Vec::with_capacity(signals.len());
    for signal in signals {
        let frame_index = signal
            .get("frame_index")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing frame_index".to_string())?;
        let pts_ms = signal
            .get("pts_ms")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing pts_ms".to_string())?;
        let perceptual_hash = signal
            .get("perceptual_hash")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| "decoded signal is missing perceptual_hash".to_string())?;
        let luma_mean = signal
            .get("luma_mean")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing luma_mean".to_string())?
            as f32;
        let flicker_score = signal
            .get("flicker_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing flicker_score".to_string())?
            as f32;
        let ghosting_score = signal
            .get("ghosting_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing ghosting_score".to_string())?
            as f32;
        let noise_variance_score = signal
            .get("noise_variance_score")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| "decoded signal is missing noise_variance_score".to_string())?
            as f32;
        if !(0.0..=1.0).contains(&luma_mean)
            || !(0.0..=1.0).contains(&flicker_score)
            || !(0.0..=1.0).contains(&ghosting_score)
            || !(0.0..=1.0).contains(&noise_variance_score)
        {
            return Err("decoded signal values must be normalized to [0.0, 1.0]".to_string());
        }

        out.push(FrameSignal {
            frame_index,
            pts_ms,
            perceptual_hash,
            luma_mean,
            flicker_score,
            ghosting_score,
            noise_variance_score,
        });
    }

    let sample_fps = payload
        .get("sample_fps")
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .or_else(|| estimate_sample_fps(&out))
        .unwrap_or(1.0);

    Ok(DecodedSignals {
        signals: out,
        sampling_policy,
        sample_fps,
    })
}

#[allow(clippy::too_many_arguments)]
async fn infer_chunk_semantics(
    providers: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    semantic_available: bool,
    _model: &str,
    semantic_prompt: &str,
    timeout_ms: u64,
    semantic_frames_per_chunk: usize,
    chunk_jpegs: &[DecodedJpegFrame],
    frame_start_index: u64,
    pts_start_ms: u64,
    pts_end_ms: u64,
    tiered_config: vidarax_core::tiered_vlm::TieredVlmConfig,
    guided_json: Option<String>,
    prev_jpeg: Option<&[u8]>,
    // When `Some`, send this MP4 clip via `input_videos` instead of JPEG frames.
    video_clip: Option<Vec<u8>>,
) -> ChunkSemanticResult {
    if !semantic_available {
        return ChunkSemanticResult::default();
    }

    let mut result = ChunkSemanticResult {
        attempted: true,
        used_fallback: false,
        ..ChunkSemanticResult::default()
    };
    let Some(provider) = providers else {
        result.used_fallback = true;
        result.error = Some("provider_not_configured".to_string());
        return result;
    };

    // In video_clip_mode the clip replaces JPEG frames entirely.  In JPEG mode
    // we require at least one selected frame.
    let use_video_clip = video_clip.is_some();
    let selected = if use_video_clip {
        Vec::new()
    } else {
        let sel = select_semantic_images(chunk_jpegs, semantic_frames_per_chunk);
        if sel.is_empty() {
            result.used_fallback = true;
            result.error = Some("chunk_has_no_jpeg_frames".to_string());
            return result;
        }
        sel
    };

    let prompt = format!(
        "{semantic_prompt}\nchunk_frame_start={frame_start_index}\nchunk_frame_end={}\nchunk_pts_start_ms={pts_start_ms}\nchunk_pts_end_ms={pts_end_ms}",
        frame_start_index.saturating_add(chunk_jpegs.len() as u64).saturating_sub(1)
    );

    // Build image and video lists for the request.
    // video_clip_mode: send the MP4 clip via input_videos, no images.
    // JPEG mode: send selected frames via input_images (+ optional prev_jpeg for
    //            visual_diff), no videos.
    let (images, videos) = if let Some(clip_bytes) = video_clip {
        let vids = vec![InferenceVideo {
            media_type: "video/mp4",
            data_base64: BASE64_STANDARD.encode(&clip_bytes),
        }];
        (Vec::new(), vids)
    } else {
        let mut imgs: Vec<InferenceImage> = Vec::with_capacity(selected.len() + 1);
        // When visual_diff is active, prepend the previous chunk's frame so the
        // VLM can see both "before" and "after" states.
        if let Some(prev) = prev_jpeg {
            imgs.push(InferenceImage {
                media_type: "image/jpeg",
                data_base64: BASE64_STANDARD.encode(prev),
            });
        }
        imgs.extend(selected.iter().map(|frame| InferenceImage {
            media_type: "image/jpeg",
            data_base64: BASE64_STANDARD.encode(&frame.jpeg_bytes),
        }));
        (imgs, Vec::new())
    };

    let prompt_arc: Arc<str> = Arc::from(prompt);
    let guided_json_arc: Option<Arc<str>> = guided_json.as_deref().map(Arc::from);
    let first_request = InferenceRequest {
        model: tiered_config.first_pass_model.clone(),
        prompt: Arc::clone(&prompt_arc),
        input_images: images.clone(),
        input_videos: videos.clone(),
        max_tokens: if guided_json.is_some() { 1024 } else { 160 },
        temperature: 0.0,
        timeout_ms,
        allow_fallback: true,
        guided_json: guided_json_arc.clone(),
    };

    let first_result = match tokio::task::spawn_blocking({
        let provider = Arc::clone(&provider);
        move || provider.infer(&first_request)
    })
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => {
            result.used_fallback = true;
            result.error = Some(match err {
                ProviderError::UnsupportedModel(_) => "unsupported_model".to_string(),
                ProviderError::HttpStatus(code) => format!("http_status_{code}"),
                ProviderError::Transport(_) => "transport_error".to_string(),
                ProviderError::InvalidResponse(_) => "invalid_response".to_string(),
            });
            return result;
        }
        Err(err) => {
            result.used_fallback = true;
            result.error = Some(format!("join_error:{err}"));
            return result;
        }
    };

    // If tiered routing is active and first-pass confidence is low, run second pass.
    let provider_result = if tiered_config.is_tiered() {
        let first_conf = parse_confidence_from_output(&first_result.output_text);
        if tiered_config.needs_second_pass(first_conf) {
            let second_request = InferenceRequest {
                model: tiered_config.second_pass_model.clone(),
                prompt: prompt_arc,
                input_images: images,
                input_videos: videos,
                max_tokens: tiered_config.second_pass_max_tokens,
                temperature: 0.0,
                timeout_ms,
                allow_fallback: true,
                guided_json: guided_json_arc,
            };
            match tokio::task::spawn_blocking(move || {
                provider.infer(&second_request)
            })
            .await
            {
                Ok(Ok(output)) => output,
                _ => first_result, // fallback to first pass on second-pass error
            }
        } else {
            first_result
        }
    } else {
        first_result
    };

    result.provider = Some(provider_name(provider_result.provider).to_string());
    result.provider_fallback_used = provider_result.fallback_used;
    result.finish_reason = provider_result.finish_reason.clone();

    if guided_json.is_some() {
        // Passthrough mode: store raw VLM output as JSON, skip overlay parsing.
        let parsed = serde_json::from_str::<Value>(&provider_result.output_text)
            .unwrap_or_else(|_| json!({"raw": provider_result.output_text}));
        result.raw_output = Some(parsed);
        result.overlay = None;
        result
    } else {
        match parse_semantic_overlay(&provider_result.output_text) {
            Some(overlay) => {
                result.overlay = Some(overlay);
                result.used_fallback = provider_result.fallback_used;
                result
            }
            None => {
                result.used_fallback = true;
                result.error = Some("semantic_parse_failed".to_string());
                result
            }
        }
    }
}

fn select_semantic_images<'a>(
    chunk_jpegs: &'a [DecodedJpegFrame],
    semantic_frames_per_chunk: usize,
) -> Vec<&'a DecodedJpegFrame> {
    if chunk_jpegs.is_empty() {
        return Vec::new();
    }
    if semantic_frames_per_chunk >= chunk_jpegs.len() {
        return chunk_jpegs.iter().collect();
    }

    let mut out = Vec::with_capacity(semantic_frames_per_chunk);
    if semantic_frames_per_chunk == 1 {
        out.push(&chunk_jpegs[chunk_jpegs.len() / 2]);
        return out;
    }

    let mut last_idx = usize::MAX;
    for i in 0..semantic_frames_per_chunk {
        let idx = i * (chunk_jpegs.len() - 1) / (semantic_frames_per_chunk - 1);
        if idx != last_idx {
            out.push(&chunk_jpegs[idx]);
            last_idx = idx;
        }
    }
    out
}

/// Extract a confidence float from VLM JSON output for tiered routing decisions.
///
/// Looks for `"confidence": 0.XX` in the JSON. Falls back to `0.5` (triggers
/// second pass at the default threshold of 0.7).
fn parse_confidence_from_output(text: &str) -> f32 {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(conf) = val.get("confidence").and_then(|v| v.as_f64()) {
            return conf as f32;
        }
    }
    0.5
}

fn parse_semantic_overlay(raw: &str) -> Option<SemanticOverlay> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let json_text = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        trimmed
    } else {
        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        if start >= end {
            return None;
        }
        &trimmed[start..=end]
    };
    let value: Value = serde_json::from_str(json_text).ok()?;

    let event_type = normalize_semantic_event(
        value
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("context_observation"),
    );
    let object_label = normalize_semantic_object(
        value
            .get("object_label")
            .and_then(|v| v.as_str())
            .unwrap_or("frame_context"),
    );
    let summary = value
        .get("summary")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("semantic summary unavailable")
        .to_string();
    let description = value
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("semantic description unavailable")
        .to_string();
    let confidence = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .map(|v| (v as f32).clamp(0.0, 1.0))
        .unwrap_or(0.5);

    Some(SemanticOverlay {
        event_type,
        object_label,
        summary,
        description,
        confidence,
    })
}

fn normalize_semantic_event(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "scene_cut" => "scene_cut".to_string(),
        "artifact_suspected" => "artifact_suspected".to_string(),
        "keyframe_keep" => "keyframe_keep".to_string(),
        "context_observation" => "context_observation".to_string(),
        _ => "context_observation".to_string(),
    }
}

fn normalize_semantic_object(raw: &str) -> String {
    let normalized = raw.trim().to_ascii_lowercase().replace(' ', "_");
    if normalized.is_empty() {
        "frame_context".to_string()
    } else {
        normalized
    }
}

#[allow(clippy::too_many_arguments)]
fn compose_frame_metadata(
    state: &AppState,
    tenant_id: Option<&str>,
    run_id: &str,
    stream_id: &str,
    mode: &str,
    model: &str,
    sampling_policy: SamplingPolicy,
    sample_fps: f32,
    segment_ms: u64,
    request_id: &str,
    trace_id: &str,
    m: FrameMetadata,
    semantic: Option<&SemanticOverlay>,
    semantic_fallback: bool,
    finish_reason: Option<String>,
) -> (AnalyzeFrameMetadata, MarkerInput) {
    let (det_event_type, det_description) = match (m.scene_cut, m.suspect_artifact, m.gate_event) {
        (true, _, _) => ("scene_cut", "Hard transition detected from pass-1 gate"),
        (_, true, _) => ("artifact_suspected", "Temporal artifact signal elevated"),
        (_, _, GateEventType::KeepKeyframe) => {
            ("keyframe_keep", "Keyframe retained by deterministic gate")
        }
        _ => (
            "context_observation",
            "No hard trigger; contextual metadata only",
        ),
    };
    let det_object_label = if m.gate_event == GateEventType::KeepKeyframe {
        "keyframe_candidate"
    } else {
        "frame_context"
    };
    let event_type = semantic
        .map(|s| s.event_type.as_str())
        .unwrap_or(det_event_type);
    let object_label = semantic
        .map(|s| s.object_label.as_str())
        .unwrap_or(det_object_label);
    let description = semantic
        .map(|s| s.description.as_str())
        .unwrap_or(det_description);
    let confidence = semantic
        .map(|s| s.confidence)
        .unwrap_or(m.confidence)
        .clamp(0.0, 1.0);
    let mapped_event = state.map_event_label(tenant_id, event_type);
    let mapped_object = state.map_object_label(tenant_id, object_label);
    let summary = semantic.map(|s| s.summary.clone()).unwrap_or_else(|| {
        format!(
            "novelty={:.3}, stability={:.3}, motion={:.3}",
            m.novelty_score, m.temporal_stability, m.motion_score
        )
    });
    (
        AnalyzeFrameMetadata {
            run_id: run_id.to_string(),
            stream_id: stream_id.to_string(),
            frame_index: m.frame_index,
            pts_ms: m.pts_ms,
            mode: mode.to_string(),
            model: model.to_string(),
            sampling_policy: sampling_policy.as_str().to_string(),
            sample_fps,
            window: AnalyzeWindow {
                start_ms: m.segment_start_ms,
                end_ms: m.segment_end_ms,
                segment_id: format!("seg-{:08x}", (m.segment_start_ms / segment_ms) as u32),
                source: "frame",
            },
            annotations: AnalyzeAnnotations {
                summary,
                objects: vec![AnalyzeObject {
                    label: mapped_object.label,
                    score: confidence,
                }],
                events: vec![AnalyzeEvent {
                    r#type: mapped_event.label.clone(),
                    score: confidence,
                    description: description.to_string(),
                }],
            },
            confidence,
            fallback: AnalyzeFallback {
                used: semantic_fallback
                    || mapped_event.used_fallback
                    || mapped_object.used_fallback,
            },
            trace: AnalyzeTrace {
                request_id: request_id.to_string(),
                trace_id: trace_id.to_string(),
                span_id: format!("span-{:016x}", m.frame_index),
            },
            ordering_key: format!("{}:{}:{}", run_id, m.pts_ms, m.frame_index),
            finish_reason,
        },
        MarkerInput {
            frame_index: m.frame_index,
            pts_ms: m.pts_ms,
            event_type: mapped_event.label,
            confidence,
        },
    )
}

fn semantic_marker_to_api_marker(marker: SemanticMarker) -> AnalyzeMarker {
    AnalyzeMarker {
        marker_id: marker.marker_id,
        run_id: marker.run_id,
        stream_id: marker.stream_id,
        event_type: marker.event_type,
        status: marker.status,
        start_frame: marker.start_frame,
        end_frame: marker.end_frame,
        start_pts_ms: marker.start_pts_ms,
        end_pts_ms: marker.end_pts_ms,
        confidence: marker.confidence,
        supersedes_marker_id: marker.supersedes_marker_id,
    }
}

fn percentile_ms(values: &[u64], percentile: u64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let p = percentile.clamp(0, 100) as f64 / 100.0;
    let idx = ((n as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn estimate_sample_fps(signals: &[FrameSignal]) -> Option<f32> {
    if signals.len() < 2 {
        return None;
    }
    let mut delta_sum = 0u64;
    let mut delta_count = 0u64;
    for window in signals.windows(2) {
        let delta = window[1].pts_ms.saturating_sub(window[0].pts_ms);
        if delta > 0 {
            delta_sum += delta;
            delta_count += 1;
        }
    }
    if delta_count == 0 {
        return None;
    }
    let avg_ms = delta_sum as f32 / delta_count as f32;
    Some((1000.0 / avg_ms).clamp(0.2, 120.0))
}

fn adaptive_sample_fps(source_fps: f32) -> f32 {
    source_fps.clamp(0.2, 120.0)
}

fn header_value<'a>(headers: &'a HeaderMap, key: &str) -> Option<&'a str> {
    headers.get(key).and_then(|value| value.to_str().ok())
}

fn principal_key_from_headers(headers: &HeaderMap) -> String {
    if let Some(tenant_id) = header_value(headers, HEADER_TENANT_ID) {
        return format!("tenant:{tenant_id}");
    }
    if let Some(api_key) = header_value(headers, HEADER_API_KEY) {
        return format!("api-key:{:016x}", stable_hash(api_key));
    }
    "public".to_string()
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for b in value.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
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
