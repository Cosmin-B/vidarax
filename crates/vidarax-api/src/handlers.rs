use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use std::cmp::min;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::task::JoinSet;
use vidarax_contracts::models::{
    fallback_candidates, REQUIRED_MEDIUM_MODELS, REQUIRED_SMALL_MODELS,
};
use vidarax_core::gate::{FrameSignal, GateConfig, GateEventType};
use vidarax_core::ingest::{
    compute_semantic_frame_indices, decode_mp4_to_frame_signals, decode_selective_jpeg_frames,
    probe_source_fps, DecodedJpegFrame, InputSource, Mp4DecodeConfig,
};
use vidarax_core::pipeline::{FrameMetadata, TwoPassConfig, TwoPassPipeline};
use vidarax_core::provider::{
    infer_with_endpoints, InferenceImage, InferenceRequest, ProviderError, ProviderKind,
};
use vidarax_core::timeline::TimelineEvent;

use crate::ids::validate_run_id;
use crate::models::{
    AnalyzeAnnotations, AnalyzeEvent, AnalyzeFallback, AnalyzeFrameMetadata, AnalyzeFramesRequest,
    AnalyzeFramesResponse, AnalyzeMarker, AnalyzeObject, AnalyzeTrace, AnalyzeWindow,
    CreateRunRequest, CreateRunResponse, FieldError, InferBatchItemError, InferBatchItemResult,
    InferBatchRequest, InferBatchResponse, InferRequest, InferResponse, ModelCatalogItem,
    ModelCatalogResponse, RealtimeReasonRequest, RealtimeReasonResponse, SamplingPolicy,
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
            decode_mp4_to_frame_signals(&decode_source, decode_config)
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

    let events = events
        .into_iter()
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
        "state": format!("{state_value:?}").to_ascii_lowercase()
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
    if state.inference_endpoints().is_none() {
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
    if state.inference_endpoints().is_none() {
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
    let model = match normalize_model(Some(payload.model.clone())) {
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
        .into_iter()
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
    let model = match normalize_model(Some(payload.model.clone())) {
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
        "You are classifying a short video chunk. Return strict JSON with keys: event_type, object_label, summary, description, confidence (0..1). event_type must be one of: scene_cut, artifact_suspected, keyframe_keep, context_observation."
            .to_string()
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
    let semantic_primary_provider = if semantic_inference {
        match parse_provider(payload.primary_provider.as_deref()) {
            Ok(provider) => provider,
            Err(message) => {
                return validation_error(
                    &state,
                    "invalid realtime reason request",
                    vec![field_error("primary_provider", message.to_string())],
                );
            }
        }
    } else {
        ProviderKind::Vllm
    };

    let tiered_config = {
        use vidarax_core::tiered_vlm::TieredVlmConfig;
        let first = payload.first_pass_model.as_deref().unwrap_or(&model);
        let second = payload.second_pass_model.as_deref().unwrap_or(&model);
        let threshold = payload.second_pass_threshold.unwrap_or(0.7);
        TieredVlmConfig {
            first_pass_model: first.to_string(),
            second_pass_model: second.to_string(),
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

    let semantic_decode_enabled = semantic_inference && state.inference_endpoints().is_some();
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
            let decoded = decode_mp4_to_frame_signals(&decode_source, decode_config)?;

            // Pre-compute which frames go to VLM (pure math, zero I/O)
            let decoded_jpegs = if semantic_decode_enabled {
                let indices = compute_semantic_frame_indices(
                    decoded.frame_signals.len(),
                    chunk_size,
                    semantic_frames_per_chunk,
                );
                // Pass 2: selective JPEG — only the ~4% of frames needed
                let jpegs = decode_selective_jpeg_frames(
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
    let providers = state.inference_endpoints().cloned();
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
        pts_start_ms: u64,
        pts_end_ms: u64,
        chunk_len: usize,
        started: Instant,
    }
    let mut chunk_preps: Vec<ChunkPrep> = Vec::new();
    for (chunk_idx, chunk) in decoded.frame_signals.chunks(chunk_size).enumerate() {
        let started = Instant::now();
        let analyzed = pipeline.analyze_batch(chunk);
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
        chunk_preps.push(ChunkPrep {
            started,
            analyzed,
            frame_offset,
            chunk_jpegs,
            pts_start_ms: chunk.first().map(|f| f.pts_ms).unwrap_or(0),
            pts_end_ms: chunk.last().map(|f| f.pts_ms).unwrap_or(0),
            chunk_len: chunk.len(),
        });
    }

    // --- Phase 2: parallel VLM inference (≤4 concurrent, saturates vLLM batching) ---
    let num_chunks = chunk_preps.len();
    let mut semantic_results: Vec<Option<ChunkSemanticResult>> = (0..num_chunks).map(|_| None).collect();
    let mut task_end_times: Vec<Instant> = vec![Instant::now(); num_chunks];

    if semantic_available {
        let sem = Arc::new(tokio::sync::Semaphore::new(4));
        let mut join_set: JoinSet<(usize, ChunkSemanticResult, Instant)> = JoinSet::new();

        for (chunk_idx, prep) in chunk_preps.iter().enumerate() {
            let providers_c = providers.clone();
            let model_c = model.clone();
            let prompt_c = semantic_prompt.clone();
            let chunk_jpegs_c = prep.chunk_jpegs.clone();
            let sem_c = Arc::clone(&sem);
            let frame_offset = prep.frame_offset as u64;
            let pts_start_ms = prep.pts_start_ms;
            let pts_end_ms = prep.pts_end_ms;
            let tiered_config_c = tiered_config.clone();
            join_set.spawn(async move {
                let _permit = sem_c.acquire().await.unwrap();
                let overlay = infer_chunk_semantics(
                    providers_c.as_ref(),
                    true,
                    semantic_primary_provider,
                    &model_c,
                    &prompt_c,
                    semantic_timeout_ms,
                    semantic_frames_per_chunk,
                    &chunk_jpegs_c,
                    frame_offset,
                    pts_start_ms,
                    pts_end_ms,
                    tiered_config_c,
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

    // --- Phase 3: sequential post-processing (WAL events, metadata, lag tracking) ---
    for (chunk_idx, prep) in chunk_preps.into_iter().enumerate() {
        let semantic_overlay = semantic_results[chunk_idx]
            .take()
            .unwrap_or_default();
        let finished = task_end_times[chunk_idx];

        if let Some(details) =
            semantic_overlay.event_payload(chunk_idx, request_id.as_str(), stream_id.as_str())
        {
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
                    "lag_ms": lag_ms
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
                "model": model
            }),
        )
        .await
    {
        return internal_error(
            &state,
            format!("failed to append analysis_generated event: {err}"),
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
    markers.sort_by_key(|m| (m.start_frame, m.end_frame, m.marker_id.clone()));

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "markers": markers
    }))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let request_id = state.next_request_id();
    let endpoints = state.inference_endpoints().cloned();
    let availability =
        match tokio::task::spawn_blocking(move || runtime_model_availability(endpoints)).await {
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

    let model = match normalize_model(Some(payload.model.clone())) {
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
            model,
            prompt: prompt.to_string(),
            input_images: Vec::new(),
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
    let endpoints = state
        .inference_endpoints()
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
        infer_with_endpoints(
            &endpoints,
            primary_provider_for_metrics,
            &request_for_provider,
        )
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
            "model": result.model,
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
        model: result.model,
        fallback_used: result.fallback_used,
        output_text: result.output_text,
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
    provider: Option<String>,
    provider_fallback_used: bool,
    used_fallback: bool,
    error: Option<String>,
    attempted: bool,
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
                "confidence": self.overlay.as_ref().map(|o| o.confidence)
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
    providers: Option<&vidarax_core::provider::ProviderEndpoints>,
    semantic_available: bool,
    primary_provider: ProviderKind,
    _model: &str,
    semantic_prompt: &str,
    timeout_ms: u64,
    semantic_frames_per_chunk: usize,
    chunk_jpegs: &[DecodedJpegFrame],
    frame_start_index: u64,
    pts_start_ms: u64,
    pts_end_ms: u64,
    tiered_config: vidarax_core::tiered_vlm::TieredVlmConfig,
) -> ChunkSemanticResult {
    if !semantic_available {
        return ChunkSemanticResult::default();
    }

    let mut result = ChunkSemanticResult {
        attempted: true,
        used_fallback: false,
        ..ChunkSemanticResult::default()
    };
    let Some(endpoints) = providers.cloned() else {
        result.used_fallback = true;
        result.error = Some("provider_not_configured".to_string());
        return result;
    };

    let selected = select_semantic_images(chunk_jpegs, semantic_frames_per_chunk);
    if selected.is_empty() {
        result.used_fallback = true;
        result.error = Some("chunk_has_no_jpeg_frames".to_string());
        return result;
    }

    let prompt = format!(
        "{semantic_prompt}\nchunk_frame_start={frame_start_index}\nchunk_frame_end={}\nchunk_pts_start_ms={pts_start_ms}\nchunk_pts_end_ms={pts_end_ms}",
        frame_start_index.saturating_add(chunk_jpegs.len() as u64).saturating_sub(1)
    );
    let images = selected
        .iter()
        .map(|frame| InferenceImage {
            media_type: "image/jpeg".to_string(),
            data_base64: BASE64_STANDARD.encode(&frame.jpeg_bytes),
        })
        .collect::<Vec<_>>();
    let first_request = InferenceRequest {
        model: tiered_config.first_pass_model.clone(),
        prompt: prompt.clone(),
        input_images: images.clone(),
        max_tokens: 160,
        temperature: 0.0,
        timeout_ms,
        allow_fallback: true,
        guided_json: None,
    };

    let first_result = match tokio::task::spawn_blocking({
        let endpoints = endpoints.clone();
        move || infer_with_endpoints(&endpoints, primary_provider, &first_request)
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
                prompt,
                input_images: images,
                max_tokens: tiered_config.second_pass_max_tokens,
                temperature: 0.0,
                timeout_ms,
                allow_fallback: true,
                guided_json: None,
            };
            match tokio::task::spawn_blocking(move || {
                infer_with_endpoints(&endpoints, primary_provider, &second_request)
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
    endpoints: Option<vidarax_core::provider::ProviderEndpoints>,
) -> RuntimeAvailability {
    let Some(endpoints) = endpoints else {
        return RuntimeAvailability {
            status: "unavailable",
            providers: Vec::new(),
        };
    };
    let vllm_up = provider_endpoint_is_up(&endpoints.vllm_base_url);
    let sglang_up = provider_endpoint_is_up(&endpoints.sglang_base_url);
    let mut providers = Vec::with_capacity(2);
    if vllm_up {
        providers.push("vllm".to_string());
    }
    if sglang_up {
        providers.push("sglang".to_string());
    }

    let status = if vllm_up && sglang_up {
        "ready"
    } else if vllm_up || sglang_up {
        "degraded"
    } else {
        "unavailable"
    };
    RuntimeAvailability { status, providers }
}

fn provider_endpoint_is_up(base_url: &str) -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(400))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    for suffix in ["/health", "/v1/models"] {
        let url = format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        );
        let response = match client.get(&url).send() {
            Ok(response) => response,
            Err(_) => continue,
        };
        if response.status().is_success() {
            return true;
        }
    }
    false
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
        _ => Err("primary_provider must be one of: vllm, sglang"),
    }
}

fn provider_name(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Vllm => "vllm",
        ProviderKind::Sglang => "sglang",
    }
}
