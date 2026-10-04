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
    fallback_candidates, EXPERIMENTAL_MODELS, GEMINI_MODELS, REQUIRED_MEDIUM_MODELS,
    REQUIRED_SMALL_MODELS,
};
use vidarax_core::audio_sidecar::AudioSidecarClient;
use vidarax_core::coordinates::{FrameCoordinates, IMAGE_COORDINATE_SCHEMA};
use vidarax_core::gate::{FrameSignal, GateEventType};
use vidarax_core::ingest::{
    prepare_source_for_reuse, probe_source_fps, DecodedJpegFrame, InputSource, Mp4DecodeConfig,
};
use vidarax_core::pipeline::{TwoPassConfig, TwoPassPipeline};
use vidarax_core::provider::{
    InferenceObserver, InferenceProvider, InferenceRequest, MediaTransport, ProviderError,
    ProviderKind,
};
use vidarax_core::timeline::TimelineEvent;

use crate::auth::{header_value, strong_hash_hex, HEADER_TENANT_ID};
use crate::config::UPLOAD_DIR_NAME;
use crate::ids::validate_run_id;
use crate::inference_metrics::{InferenceMetrics, PipelineInferenceObserver};
use crate::models::{
    AnalyzeFrameMetadata, AnalyzeFramesRequest, AnalyzeFramesResponse, AnalyzeMarker,
    CreateRunRequest, CreateRunResponse, FieldError, InferBatchItemError, InferBatchItemResult,
    InferBatchRequest, InferBatchResponse, InferRequest, InferResponse, IngestRequest,
    MediaAnalysisMode, MediaAnalysisResolution, ModelCatalogItem, ModelCatalogResponse,
    QueryRequest, RealtimeReasonRequest, RealtimeReasonResponse, SamplingPolicy, SearchHit,
    SearchRequest, SearchResponse, TokenMetrics,
};
use crate::response::{
    bad_request_error, conflict_error, internal_error, not_found_error, ok, service_unavailable,
    validation_error, ApiResponse,
};
use crate::semantic::{build_marker_lifecycle, MarkerConfig};
use crate::semantic_infer::{
    adaptive_sample_fps, compose_frame_metadata, estimate_sample_fps,
    load_decoded_signals_from_events, percentile_ms, prepare_realtime_chunks,
    run_semantic_dispatch, semantic_marker_to_api_marker, AudioTraceContext, ChunkPrep,
    ChunkSemanticResult, LocalAudioConfig, SemanticMediaConfig, SemanticMediaMode,
};
use crate::state::AppState;
use crate::wal_sink::persist_media_blob;

const UPLOAD_MEDIA_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const SPACETIME_MARKER_MIRROR_TIMEOUT: Duration = Duration::from_secs(2);
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
    Json(payload): Json<IngestRequest>,
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
    let source_uri = payload.source_uri.trim().to_string();
    if source_uri.is_empty() {
        return validation_error(
            &state,
            "invalid ingest request",
            vec![field_error(
                "source_uri",
                "source_uri must not be empty".to_string(),
            )],
        );
    }

    let sampling_policy = match SamplingPolicy::parse(payload.sampling_policy.as_deref()) {
        Ok(policy) => policy,
        Err(message) => {
            return validation_error(
                &state,
                "invalid ingest request",
                vec![field_error("sampling_policy", message.to_string())],
            );
        }
    };
    let fixed_fps = payload.fixed_fps;
    let sample_fps = payload.sample_fps.or(fixed_fps);
    if sampling_policy == SamplingPolicy::Fixed {
        let Some(sample_fps) = sample_fps else {
            return validation_error(
                &state,
                "invalid ingest request",
                vec![field_error(
                    "fixed_fps",
                    "fixed_fps (or sample_fps) is required when sampling_policy=fixed".to_string(),
                )],
            );
        };
        if !(vidarax_contracts::processing::REQUEST_FPS_MIN
            ..=vidarax_contracts::processing::REQUEST_FPS_MAX)
            .contains(&sample_fps)
        {
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
    let max_frames = payload.max_frames.unwrap_or(512);
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
        .stream_id
        .as_deref()
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
        // Fetch a remote source once so the probe and signal decode share the
        // local copy rather than downloading it twice.
        let prepared = prepare_source_for_reuse(&decode_source)?;
        let decode_source = prepared.source();
        let source_fps = probe_source_fps(decode_source);
        let effective_sample_fps = match sampling_policy {
            SamplingPolicy::SourceFpsAdaptive => source_fps
                .map(adaptive_sample_fps)
                .unwrap_or(requested_sample_fps),
            SamplingPolicy::Fixed => requested_sample_fps,
        };
        let decode_config = Mp4DecodeConfig {
            sample_fps: effective_sample_fps,
            max_frames: max_frames as usize,
            max_edge: None,
            // The signals-only ingest endpoint does not expose a crop; the
            // region-of-interest lever lives on the /reason analysis path.
            crop: None,
        };
        let decode_started = Instant::now();
        decode_pipeline
            .decode_signals(decode_source, decode_config)
            .map(|decoded| {
                (
                    decoded,
                    source_fps,
                    effective_sample_fps,
                    decode_started.elapsed().as_micros() as u64,
                )
            })
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
    let (decoded, source_fps, effective_sample_fps, decode_elapsed_us) = decoded;
    state
        .pipeline_metrics()
        .record_decoded_batch(decoded.frame_signals.len() as u64, decode_elapsed_us);

    if let Err(err) = state
        .append_run_event_async(
            &run_id,
            "ingest_received",
            json!({
                "request_id": request_id,
                "ingest": payload,
                "decoded_frames": decoded.frame_signals.len(),
                // Record what the caller asked to ingest. When the source is a
                // remote URL, decoded.source_uri holds the internal prefetch
                // temp path, which is deleted after decode and meaningless to
                // clients reading run metadata later.
                "source_uri": source_uri.as_str(),
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
                "source_uri": source_uri.as_str(),
                "stream_id": stream_id,
                "sampling_policy": sampling_policy.as_str(),
                "source_fps": source_fps,
                "sample_fps": effective_sample_fps,
                "decoded_frames": signals.len(),
                "width": decoded.width,
                "height": decoded.height,
                "pixel_format": decoded.pixel_format,
                "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
                "coordinates": decoded.coordinates,
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

    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
        "status": "processing",
        "decoded_frames": decoded.frame_signals.len(),
        "source_uri": source_uri,
        "sampling_policy": sampling_policy.as_str(),
        "source_fps": source_fps,
        "sample_fps": effective_sample_fps
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
    let transaction = tokio::spawn(transition_live_run(
        state.clone(),
        run_id.clone(),
        request_id.clone(),
        "stop_requested",
        true,
    ));
    match transaction.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return internal_error(&state, format!("failed to stop run: {err}"));
        }
        Err(err) => {
            return internal_error(&state, format!("stop transaction join failure: {err}"));
        }
    }

    ok(json!({
        "request_id": request_id,
        "run_id": run_id,
        "status": "cancelled"
    }))
}

async fn transition_live_run(
    state: AppState,
    run_id: String,
    request_id: String,
    event_kind: &'static str,
    preserve_history: bool,
) -> Result<(), String> {
    // The durable state transition and live-session close belong to one
    // detached transaction. Dropping the HTTP request cannot leave a stopped
    // or deleted run with its media pipeline still running.
    state
        .append_run_event_async(&run_id, event_kind, json!({ "request_id": request_id }))
        .await?;
    state.close_live_session_for_run(&run_id, preserve_history);
    Ok(())
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
    Json(payload): Json<QueryRequest>,
) -> impl IntoResponse {
    if let Some(error) = validate_run_id_or_error(&state, &payload.run_id, "invalid query payload")
    {
        return error;
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &payload.run_id) {
        return error;
    }

    let kind_filter = payload.kind.as_deref();
    let from_seq = payload.from_seq.unwrap_or(0);
    let events = match state.read_run_events_from(&payload.run_id, from_seq).await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read run events: {err}")),
    };

    let matches = events
        .into_iter()
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

    let (signals, sampling_policy, sample_fps, coordinates) = if payload.frames.is_empty() {
        let events = match load_existing_events(&state, &run_id).await {
            Ok(events) => events,
            Err(error) => return error,
        };
        match load_decoded_signals_from_events(&events) {
            Ok(decoded) => (
                decoded.signals,
                decoded.sampling_policy,
                decoded.sample_fps,
                decoded.coordinates,
            ),
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
            if !(vidarax_contracts::processing::REQUEST_FPS_MIN
                ..=vidarax_contracts::processing::REQUEST_FPS_MAX)
                .contains(&fixed)
            {
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
        (signals, sampling_policy, sample_fps, None)
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
        state.webrtc_config().gate_config.clone(),
    );
    let gate_started = Instant::now();
    let analyzed = pipeline.analyze_batch(&signals);
    let gate_elapsed_us = gate_started.elapsed().as_micros() as u64;
    let selected = analyzed
        .iter()
        .filter(|frame| frame.gate_event == GateEventType::KeepKeyframe)
        .count() as u64;
    state
        .pipeline_metrics()
        .record_gate_batch(analyzed.len() as u64, selected, gate_elapsed_us);

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
                coordinates,
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

    let mut analysis_event = json!({
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
    });
    if let Some(coordinates) = coordinates {
        analysis_event["coordinate_schema"] = json!(IMAGE_COORDINATE_SCHEMA);
        analysis_event["coordinates"] = json!(coordinates);
    }
    if let Err(err) = state
        .append_run_event_async(&run_id, "analysis_generated", analysis_event)
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
    semantic_frame_max_edge: Option<u32>,
    crop: Option<vidarax_core::crop::CropRegion>,
    semantic_timeout_ms: u64,
    semantic_prompt: String,
    tiered_config: vidarax_core::tiered_vlm::TieredVlmConfig,
    decode_source: InputSource,
    media: SemanticMediaConfig,
    local_audio: Option<LocalAudioConfig>,
    include_frame_metadata: bool,
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
    if !(1..=256).contains(&semantic_frames_per_chunk) {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_frames_per_chunk",
                "semantic_frames_per_chunk must be in [1, 256]".to_string(),
            )],
        ));
    }
    let context_frames = payload.semantic_context_frames.unwrap_or(0);
    if context_frames > 128
        || payload
            .source_end_ms
            .is_some_and(|end| end <= payload.source_start_ms.unwrap_or(0))
    {
        return Err(validation_error(
            state,
            "invalid review coverage",
            vec![field_error(
                "semantic_context_frames/source_end_ms",
                "context must be <=128 and source_end_ms must exceed source_start_ms".into(),
            )],
        ));
    }
    let semantic_frame_max_edge = payload.semantic_frame_max_edge;
    if let Some(edge) = semantic_frame_max_edge {
        if !(64..=4_096).contains(&edge) {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "semantic_frame_max_edge",
                    "semantic_frame_max_edge must be in [64, 4096]".to_string(),
                )],
            ));
        }
    }
    let crop = payload.crop;
    if let Some(region) = crop {
        if let Err(err) = region.validate() {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error("crop", err.to_string())],
            ));
        }
    }
    let native_media_requested = payload
        .media
        .as_ref()
        .is_some_and(|media| !matches!(media.mode, MediaAnalysisMode::Frames));
    let semantic_timeout_ms = payload
        .semantic_timeout_ms
        .unwrap_or(if native_media_requested {
            30_000
        } else {
            1_500
        });
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
    if payload.media.is_some()
        && (payload.video_clip_mode.is_some() || payload.video_clip_duration_s.is_some())
    {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "media",
                "media cannot be combined with video_clip_mode or video_clip_duration_s"
                    .to_string(),
            )],
        ));
    }
    let has_local_audio = payload.local_audio.is_some();
    let media = if let Some(options) = payload.media {
        let mode = match options.mode {
            MediaAnalysisMode::Frames => SemanticMediaMode::Frames,
            MediaAnalysisMode::Video => SemanticMediaMode::Video,
            MediaAnalysisMode::AudioVideo => SemanticMediaMode::AudioVideo,
        };
        if mode != SemanticMediaMode::Frames && payload.chunk_size.is_some() {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "chunk_size",
                    "chunk_size cannot be combined with native video media modes".to_string(),
                )],
            ));
        }
        let window_ms = options
            .window_ms
            .unwrap_or(if mode == SemanticMediaMode::Frames {
                500
            } else if has_local_audio {
                20_000
            } else {
                8_000
            });
        if !(100..=60_000).contains(&window_ms) {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "media.window_ms",
                    "media.window_ms must be in [100, 60000]".to_string(),
                )],
            ));
        }
        let overlap_ms = options.overlap_ms.unwrap_or(0);
        if overlap_ms >= window_ms
            || overlap_ms > window_ms / 2
            || (mode == SemanticMediaMode::Frames
                && (overlap_ms > 0 || options.video_fps.is_some()))
        {
            return Err(validation_error(state, "invalid media controls", vec![field_error("media.overlap_ms", "overlap must be <= half the window and native video controls require video/audio_video mode".into())]));
        }
        if options
            .video_fps
            .is_some_and(|fps| !fps.is_finite() || fps <= 0.0 || fps > 24.0)
        {
            return Err(validation_error(
                state,
                "invalid media controls",
                vec![field_error(
                    "media.video_fps",
                    "video_fps must be finite and in (0, 24]".into(),
                )],
            ));
        }
        SemanticMediaConfig {
            overlap_ms,
            video_fps: options.video_fps,
            context_frames,
            source_start_ms: payload.source_start_ms,
            source_end_ms: payload.source_end_ms,
            mode,
            window_ms,
            resolution: options
                .resolution
                .unwrap_or(MediaAnalysisResolution::Low)
                .into(),
            persist_evidence: options
                .persist_evidence
                .unwrap_or(mode == SemanticMediaMode::AudioVideo),
            timestamp_windows: true,
        }
    } else {
        let legacy_video = payload.video_clip_mode.unwrap_or(false);
        let duration_s = payload.video_clip_duration_s.unwrap_or(0.5);
        // Reject sub-millisecond windows before conversion: a positive float
        // can otherwise round to zero and become a zero scheduling stride.
        if legacy_video && (!duration_s.is_finite() || !(0.001..=60.0).contains(&duration_s)) {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "video_clip_duration_s",
                    "video_clip_duration_s must be finite and in [0.001, 60]".to_string(),
                )],
            ));
        }
        SemanticMediaConfig {
            overlap_ms: 0,
            video_fps: None,
            context_frames,
            source_start_ms: payload.source_start_ms,
            source_end_ms: payload.source_end_ms,
            mode: if legacy_video {
                SemanticMediaMode::Video
            } else {
                SemanticMediaMode::Frames
            },
            window_ms: (duration_s * 1_000.0).round() as u64,
            resolution: vidarax_core::provider::MediaResolution::Low,
            persist_evidence: false,
            timestamp_windows: false,
        }
    };
    if media.mode != SemanticMediaMode::Frames && context_frames > 0 {
        return Err(validation_error(
            state,
            "invalid media controls",
            vec![field_error(
                "semantic_context_frames",
                "use media.overlap_ms for native video context".into(),
            )],
        ));
    }
    if media.video_fps.is_some() && !semantic_inference {
        return Err(validation_error(
            state,
            "invalid media controls",
            vec![field_error(
                "media.video_fps",
                "explicit provider sampling requires semantic_inference=true".into(),
            )],
        ));
    }
    let local_audio = if let Some(options) = payload.local_audio {
        if media.mode != SemanticMediaMode::AudioVideo {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "local_audio",
                    "local_audio requires media.mode=audio_video".to_string(),
                )],
            ));
        }
        if !options.min_confidence.is_finite() || !(0.0..=1.0).contains(&options.min_confidence) {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "local_audio.min_confidence",
                    "min_confidence must be finite and in [0, 1]".to_string(),
                )],
            ));
        }
        if !(1..=64).contains(&options.max_events) {
            return Err(validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "local_audio.max_events",
                    "max_events must be in [1, 64]".to_string(),
                )],
            ));
        }
        let address = std::env::var("VIDARAX_AUDIO_SIDECAR_ADDR")
            .ok()
            .filter(|address| !address.trim().is_empty())
            .ok_or_else(|| {
                service_unavailable(
                    state,
                    "audio_sidecar_unavailable",
                    "local_audio requires VIDARAX_AUDIO_SIDECAR_ADDR",
                )
            })?;
        AudioSidecarClient::new(&address, semantic_timeout_ms).map_err(|error| {
            validation_error(
                state,
                "invalid realtime reason request",
                vec![field_error(
                    "local_audio",
                    format!("invalid audio sidecar configuration: {error}"),
                )],
            )
        })?;
        Some(LocalAudioConfig {
            sidecar_addr: Arc::from(address),
            profile: options.profile,
            speech_engine: options.speech_engine,
            min_confidence: options.min_confidence,
            max_events: options.max_events,
            voice_feedback: options.voice_feedback,
            trace: AudioTraceContext::default(),
            metrics: Arc::clone(state.pipeline_metrics_arc()),
        })
    } else {
        None
    };
    let semantic_prompt = payload.semantic_prompt.clone().unwrap_or_else(|| {
        if media.mode == SemanticMediaMode::AudioVideo {
            "Analyze the synchronized audio and video as one physical event window. Return strict JSON with a moments array. Include speech intent only when the sound supports it. Include non-speech sounds, effects, music, ambient or mechanical noise, and their relationship to visible actions.".to_string()
        } else {
            "You are classifying a short video chunk. Return strict JSON with keys: event_type, object_label, summary, description, confidence (0..1). event_type must be one of: scene_cut, artifact_suspected, keyframe_keep, context_observation.".to_string()
        }
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

    let fixed_fps = payload.fixed_fps.unwrap_or(1.0);
    if sampling_policy == SamplingPolicy::Fixed
        && !(vidarax_contracts::processing::REQUEST_FPS_MIN
            ..=vidarax_contracts::processing::REQUEST_FPS_MAX)
            .contains(&fixed_fps)
    {
        return Err(validation_error(
            state,
            "invalid realtime reason request",
            vec![field_error(
                "fixed_fps",
                "fixed_fps must be in [0.2, 120.0]".to_string(),
            )],
        ));
    }

    let include_frame_metadata = payload
        .include_frame_metadata
        .unwrap_or(media.mode == SemanticMediaMode::Frames);
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
        semantic_frame_max_edge,
        crop,
        semantic_timeout_ms,
        semantic_prompt,
        tiered_config,
        decode_source,
        media,
        local_audio,
        include_frame_metadata,
        fixed_fps,
    })
}

struct RealtimeAssemblyOutput {
    metadata: Vec<AnalyzeFrameMetadata>,
    markers: Vec<AnalyzeMarker>,
    lag_p95_ms: u64,
    lag_p99_ms: u64,
    tokens: TokenMetrics,
}

fn marker_to_emit_event_request(
    marker: &AnalyzeMarker,
) -> crate::spacetime_client::EmitEventRequest {
    crate::spacetime_client::EmitEventRequest {
        run_id: marker.run_id.clone(),
        session_id: marker.stream_id.clone(),
        frame_index: marker.start_frame,
        pts_ms: marker.start_pts_ms,
        event_type: marker.event_type.clone(),
        confidence: marker.confidence,
        description: format!(
            "{} ({}..{} frames, {}..{} ms)",
            marker.status.as_str(),
            marker.start_frame,
            marker.end_frame,
            marker.start_pts_ms,
            marker.end_pts_ms
        ),
    }
}

fn spawn_semantic_journal(
    state: AppState,
    run_id: String,
    request_id: String,
    stream_id: String,
    index_name: Option<String>,
    overlap_ms: u64,
    mut semantic_event_rx: tokio::sync::mpsc::Receiver<(usize, ChunkSemanticResult)>,
) -> tokio::task::JoinHandle<Result<(), String>> {
    // The journal owns its receiver through active window completion. An
    // append error closes it so a sender cannot wait for an exited consumer.
    tokio::spawn(async move {
        let mut seen_moments = Vec::new();
        let mut pending = std::collections::BTreeMap::new();
        let mut next_chunk = 0;
        while let Some((chunk_idx, result)) = semantic_event_rx.recv().await {
            pending.insert(chunk_idx, result);
            while let Some(mut result) = pending.remove(&next_chunk) {
                let chunk_idx = next_chunk;
                next_chunk += 1;
                if overlap_ms > 0 {
                    crate::semantic_infer::deduplicate_source_moments(
                        &mut result.moments,
                        &mut seen_moments,
                    );
                }
                append_semantic_chunk_event(
                    &state,
                    &run_id,
                    &request_id,
                    &stream_id,
                    &index_name,
                    chunk_idx,
                    &result,
                )
                .await?;
            }
        }
        // Cancellation can leave a chunk without a sender after a task
        // panic. Completed later chunks still require their WAL append.
        for (chunk_idx, mut result) in pending {
            if overlap_ms > 0 {
                crate::semantic_infer::deduplicate_source_moments(
                    &mut result.moments,
                    &mut seen_moments,
                );
            }
            append_semantic_chunk_event(
                &state,
                &run_id,
                &request_id,
                &stream_id,
                &index_name,
                chunk_idx,
                &result,
            )
            .await?;
        }
        Ok::<(), String>(())
    })
}

async fn append_semantic_chunk_event(
    state: &AppState,
    run_id: &str,
    request_id: &str,
    stream_id: &str,
    index_name: &Option<String>,
    chunk_idx: usize,
    result: &ChunkSemanticResult,
) -> Result<(), String> {
    let Some(mut details) = result.event_payload(chunk_idx, request_id, stream_id) else {
        return Ok(());
    };
    if result
        .error
        .as_ref()
        .is_some_and(|error| error.is_media_extraction())
    {
        state.pipeline_metrics().inc_media_clip_extraction_failure();
    }
    if result.local_audio_telemetry.wav_bytes > 0 {
        let audio_span = tracing::info_span!(
            "audio.chunk",
            run_id,
            request_id,
            stream_id,
            chunk_id = chunk_idx,
            wav_bytes = result.local_audio_telemetry.wav_bytes,
            requested_duration_ms = result.local_audio_telemetry.requested_duration_ms,
        );
        let _entered = audio_span.enter();
        state.pipeline_metrics().record_local_audio_extraction(
            result.local_audio_telemetry.wav_bytes,
            result.local_audio.as_ref().map_or(
                result.local_audio_telemetry.requested_duration_ms,
                |audio| audio.audio_duration_ms,
            ),
            result.local_audio_telemetry.wav_extraction_ms,
        );
        if let Some(audio) = &result.local_audio {
            state.pipeline_metrics().record_local_audio_analysis(
                audio,
                result
                    .local_audio_telemetry
                    .wav_extraction_ms
                    .saturating_add(result.local_audio_telemetry.round_trip_ms),
            );
        } else if let Some(reason) = result.local_audio_telemetry.failure_reason {
            state
                .pipeline_metrics()
                .record_local_audio_failure_reason(reason);
        }
        if result.local_audio_telemetry.tts_attempted {
            state.pipeline_metrics().inc_local_audio_tts_attempt();
            if let Some(feedback) = &result.feedback_audio {
                state.pipeline_metrics().record_local_audio_tts_success(
                    feedback.bytes.len() as u64,
                    feedback.processing_ms,
                    feedback.capacity,
                );
            } else if let Some(reason) = result.local_audio_telemetry.tts_failure_reason {
                state
                    .pipeline_metrics()
                    .record_local_audio_tts_failure_reason(reason);
            }
        }
    }
    let mut evidence = None;
    if let Some(media) = &result.media {
        state
            .pipeline_metrics()
            .record_media_clip_extracted(media.bytes.len() as u64, media.extraction_ms);
        if media.persist_evidence && !result.moments.is_empty() {
            let state_for_blob = state.clone();
            let bytes = Arc::clone(&media.bytes);
            let media_type = media.media_type;
            let blob_result = tokio::task::spawn_blocking(move || {
                persist_media_blob(&state_for_blob, bytes.as_ref(), media_type)
            })
            .await
            .map_err(|error| format!("media sidecar worker failed: {error}"))?;
            let blob = match blob_result {
                Ok(blob) => blob,
                Err(error) => {
                    state.pipeline_metrics().inc_media_blob_failure();
                    return Err(error);
                }
            };
            state.pipeline_metrics().record_media_blob(blob.created);
            evidence = Some(json!({
                "media_ref": blob.media_ref,
                "media_type": media.media_type,
                "media_bytes": blob.bytes,
                "media_sha256": blob.sha256,
                "created": blob.created,
            }));
        }
        if let Some(object) = details.as_object_mut() {
            object.insert("media_mode".to_string(), json!(media.mode.as_str()));
            object.insert(
                "media_resolution".to_string(),
                json!(media.resolution.as_str()),
            );
            object.insert("pts_start_ms".to_string(), json!(media.source_start_ms));
            object.insert("pts_end_ms".to_string(), json!(media.source_end_ms));
            object.insert(
                "timestamp_resolution_ms".to_string(),
                json!(if result.provider == Some("gemini") {
                    1_000
                } else {
                    1
                }),
            );
            object.insert(
                "provider_sampling_interval_ms".into(),
                json!(media.video_fps.map(|fps| (1000.0 / fps).ceil() as u64)),
            );
            object.insert(
                "provider_sampling_status".into(),
                json!(media.video_fps.map(|_| "requested_unverified")),
            );
            object.insert("clip_bytes".to_string(), json!(media.bytes.len()));
            object.insert("extraction_ms".to_string(), json!(media.extraction_ms));
            object.insert("audio_streams".to_string(), json!(media.audio_streams));
            object.insert("audio_channels".to_string(), json!(media.audio_channels));
            object.insert("audio_mixed".to_string(), json!(media.audio_mixed));
            if let Some(evidence) = &evidence {
                object.insert("evidence".to_string(), evidence.clone());
            }
        }
    }
    if let Some(feedback) = &result.feedback_audio {
        let state_for_blob = state.clone();
        let bytes = Arc::clone(&feedback.bytes);
        let media_type = feedback.media_type;
        let blob_result = tokio::task::spawn_blocking(move || {
            persist_media_blob(&state_for_blob, bytes.as_ref(), media_type)
        })
        .await
        .map_err(|error| format!("feedback audio sidecar worker failed: {error}"))?;
        let blob = match blob_result {
            Ok(blob) => blob,
            Err(error) => {
                state.pipeline_metrics().inc_media_blob_failure();
                return Err(error);
            }
        };
        state.pipeline_metrics().record_media_blob(blob.created);
        if let Some(object) = details.as_object_mut() {
            object.insert(
                "feedback_audio".to_string(),
                json!({
                    "media_ref": blob.media_ref,
                    "media_type": feedback.media_type,
                    "media_bytes": blob.bytes,
                    "media_sha256": blob.sha256,
                    "created": blob.created,
                    "model": feedback.model,
                    "sample_rate_hz": feedback.sample_rate_hz,
                    "processing_ms": feedback.processing_ms,
                }),
            );
        }
    }
    if let Some(index) = index_name {
        if let Some(object) = details.as_object_mut() {
            object.insert("index_name".to_string(), serde_json::json!(index));
        }
    }
    state
        .append_run_event_async(run_id, "semantic_chunk_inferred", details)
        .await?;

    for (moment_idx, moment) in result.moments.iter().enumerate() {
        state
            .append_run_event_async(
                run_id,
                "multimodal_moment",
                json!({
                    "moment_id": format!("{request_id}:{chunk_idx}:{moment_idx}"),
                    "request_id": request_id,
                    "stream_id": stream_id,
                    "chunk_index": chunk_idx,
                    "start_offset_ms": moment.start_offset_ms,
                    "end_offset_ms": moment.end_offset_ms,
                    "start_pts_ms": moment.start_pts_ms,
                    "end_pts_ms": moment.end_pts_ms,
                    "timestamp_resolution_ms": if result.provider == Some("gemini") { 1_000 } else { 1 },
                    "provider_sampling_interval_ms": result.media.as_ref().and_then(|media| media.video_fps).map(|fps| (1000.0 / fps).ceil() as u64),
                    "provider_sampling_status": result.media.as_ref().and_then(|media| media.video_fps).map(|_| "requested_unverified"),
                    "modalities": &moment.modalities,
                    "kind": moment.kind.as_str(),
                    "description": moment.description.as_str(),
                    "intent": moment.intent.as_deref(),
                    "audio_visual_relation": moment.audio_visual_relation.as_deref(),
                    "confidence": moment.confidence,
                    "provider": result.provider,
                    "index_name": index_name,
                    "evidence": evidence.as_ref(),
                }),
            )
            .await?;
    }
    state
        .pipeline_metrics()
        .add_multimodal_moments(result.moments.len() as u64);
    Ok(())
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
    coordinates: FrameCoordinates,
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
    let mut token_metrics = TokenMetrics::default();

    for (chunk_idx, prep) in chunk_preps.into_iter().enumerate() {
        let semantic_overlay = semantic_results[chunk_idx].take().unwrap_or_default();
        let finished = task_end_times[chunk_idx];

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
                Some(coordinates),
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

        // e2e token/latency accounting: fold this chunk's model spend into the
        // run total so the response reports the full cost of the analysis.
        if semantic_overlay.attempted {
            token_metrics.accumulate_chunk(
                semantic_overlay.usage,
                semantic_overlay.inference_latency_ms,
            );
        }

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
                    "prompt_tokens": semantic_overlay.usage.prompt_tokens,
                    "completion_tokens": semantic_overlay.usage.completion_tokens,
                    "thinking_tokens": semantic_overlay.usage.thinking_tokens,
                    "total_tokens": semantic_overlay.usage.total_tokens,
                    "inference_latency_ms": semantic_overlay.inference_latency_ms,
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
        if let Some(stdb) = state.spacetime_client() {
            let req = marker_to_emit_event_request(marker);
            match tokio::time::timeout(SPACETIME_MARKER_MIRROR_TIMEOUT, stdb.emit_event_async(&req))
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!(
                    run_id = %run_id,
                    marker_id = %marker.marker_id.as_str(),
                    error = %err,
                    "spacetimedb marker mirror failed; continuing (WAL is authoritative)"
                ),
                Err(_elapsed) => tracing::warn!(
                    run_id = %run_id,
                    marker_id = %marker.marker_id.as_str(),
                    timeout_ms = SPACETIME_MARKER_MIRROR_TIMEOUT.as_millis() as u64,
                    "spacetimedb marker mirror timed out; continuing (WAL is authoritative)"
                ),
            }
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
                "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
                "coordinates": coordinates,
                "lag_p95_ms": lag_p95_ms,
                "lag_p99_ms": lag_p99_ms,
                "mode": mode,
                "model": model,
                "index_name": index_name,
                "prompt_tokens": token_metrics.prompt_tokens,
                "completion_tokens": token_metrics.completion_tokens,
                "thinking_tokens": token_metrics.thinking_tokens,
                "total_tokens": token_metrics.total_tokens,
                "inference_latency_ms": token_metrics.inference_latency_ms,
                "chunks_analyzed": token_metrics.chunks_analyzed,
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
        tokens: token_metrics,
    })
}

fn review_plan_chunk_count(
    media: &SemanticMediaConfig,
    sample_count: usize,
    first_sample_pts: u64,
    chunk_size: usize,
    source_duration_ms: Option<u64>,
) -> Result<u64, String> {
    let count = if media.mode == SemanticMediaMode::Frames || !media.timestamp_windows {
        sample_count.div_ceil(chunk_size) as u64
    } else {
        let start = media.source_start_ms.unwrap_or(first_sample_pts);
        let end = media
            .source_end_ms
            .or(source_duration_ms)
            .ok_or("native review requires known source duration")?;
        1 + end
            .saturating_sub(start)
            .saturating_sub(media.window_ms)
            .div_ceil(media.window_ms - media.overlap_ms)
    };
    if count > 2048 {
        return Err(format!("review exceeds 2048 chunks/windows ({count} planned); narrow the interval or split the request"));
    }
    Ok(count)
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
    let semantic_frame_max_edge = params.semantic_frame_max_edge;
    let crop = params.crop;
    let semantic_timeout_ms = params.semantic_timeout_ms;
    let semantic_prompt = params.semantic_prompt;
    let tiered_config = params.tiered_config;
    let media = params.media;
    let mut local_audio = params.local_audio;
    let include_frame_metadata = params.include_frame_metadata;
    let fixed_fps = params.fixed_fps;
    let semantic_decode_enabled =
        (semantic_inference && state.provider().is_some()) || local_audio.is_some();
    if media.timestamp_windows
        && media.mode != SemanticMediaMode::Frames
        && !semantic_inference
        && local_audio.is_none()
    {
        return validation_error(
            &state,
            "invalid realtime reason request",
            vec![field_error(
                "semantic_inference",
                "native media analysis requires semantic_inference=true or local_audio".to_string(),
            )],
        );
    }
    if semantic_inference && media.timestamp_windows && media.mode != SemanticMediaMode::Frames {
        let Some(provider) = state.provider() else {
            return service_unavailable(
                &state,
                "inference_provider_unavailable",
                "native media analysis requires a configured binary media provider",
            );
        };
        if provider.media_transport_for_model(tiered_config.first_pass_model.as_ref())
            != MediaTransport::BinaryFile
        {
            return validation_error(
                &state,
                "invalid realtime reason request",
                vec![field_error(
                    "media.mode",
                    "native video modes require a provider with binary media transport".to_string(),
                )],
            );
        }
    }
    let budget_previous_image = payload.visual_diff.unwrap_or(false);
    let strict_coverage = semantic_frames_per_chunk > 4
        || media.context_frames > 0
        || media.overlap_ms > 0
        || media.video_fps.is_some()
        || media.source_start_ms.is_some()
        || media.source_end_ms.is_some();
    if semantic_inference {
        if let Some(provider) = state.provider() {
            for model in [
                &tiered_config.first_pass_model,
                &tiered_config.second_pass_model,
            ] {
                let invalid = if media.mode == SemanticMediaMode::Frames {
                    semantic_frames_per_chunk + usize::from(payload.visual_diff.unwrap_or(false))
                        > provider.max_input_images_for_model(model)
                } else {
                    (media.timestamp_windows
                        && provider.media_transport_for_model(model) != MediaTransport::BinaryFile)
                        || media.video_fps.is_some_and(|fps| {
                            provider
                                .max_video_fps_for_model(model)
                                .is_none_or(|max| fps > max)
                        })
                };
                if invalid {
                    return validation_error(&state, "unsupported review controls", vec![field_error("model", format!("{model} does not support the requested frame count/native FPS; configure provider capacity or select a compatible model"))]);
                }
            }
        } else if strict_coverage {
            return service_unavailable(
                &state,
                "inference_provider_unavailable",
                "explicit review coverage requires a configured provider",
            );
        }
    }
    let decode_source = params.decode_source;
    let decode_pipeline = state.decode_pipeline();
    let (
        prepared_source,
        decoded,
        source_fps,
        sample_fps,
        decoded_jpegs,
        media_info,
        decode_elapsed_us,
    ) = match tokio::task::spawn_blocking(move || {
        // Fetch a remote source once here. The probe, signal decode, JPEG
        // decode, and per-chunk clip extraction below all read the same local
        // copy instead of re-downloading it on every call.
        let prepared = prepare_source_for_reuse(&decode_source)?;
        let decode_source = prepared.source();
        let media_info = if media.mode == SemanticMediaMode::Frames {
            None
        } else {
            Some(prepared.media_info()?.clone())
        };
        if media.mode == SemanticMediaMode::AudioVideo {
            let info = media_info
                .as_ref()
                .ok_or_else(|| "audio-video media probe did not complete".to_string())?;
            if info.video_streams == 0 {
                return Err("audio-video analysis requires a video stream".to_string());
            }
            if info.audio_streams == 0 {
                return Err("audio-video analysis requires an audio stream".to_string());
            }
            if info.audio_streams > 8 {
                return Err("audio-video analysis supports at most 8 audio streams".to_string());
            }
        }
        let source_fps = probe_source_fps(decode_source);
        let sample_fps = match sampling_policy {
            SamplingPolicy::SourceFpsAdaptive => {
                source_fps.map(adaptive_sample_fps).unwrap_or(24.0)
            }
            SamplingPolicy::Fixed => fixed_fps,
        };
        let decode_config = Mp4DecodeConfig {
            sample_fps,
            max_frames: max_frames as usize + usize::from(strict_coverage),
            // Signals stay at source resolution; only VLM-bound JPEGs are
            // downscaled (below), so the gate engine keeps full detail.
            max_edge: None,
            // Crop applies to signals too: the gate should judge only the
            // region the VLM will ultimately see, so both agree on the ROI.
            crop,
        };
        // Pass 1: frame signals (cheap, no encoding)
        let decode_started = Instant::now();
        let mut decoded = decode_pipeline.decode_signals(decode_source, decode_config)?;
        if strict_coverage && decoded.frame_signals.len() > max_frames as usize {
            return Err(
                "requested coverage exceeds max_frames; increase the explicit decode budget"
                    .to_string(),
            );
        }
        let start = media.source_start_ms.unwrap_or(0);
        let duration = prepared.media_info().ok().and_then(|info| info.duration_ms);
        if media
            .source_end_ms
            .is_some_and(|end| duration.is_some_and(|duration| end > duration))
            || duration.is_some_and(|duration| start >= duration)
        {
            return Err("requested source interval is outside video duration".to_string());
        }
        decoded.frame_signals.retain(|frame| {
            frame.pts_ms >= start && media.source_end_ms.is_none_or(|end| frame.pts_ms < end)
        });
        if decoded.frame_signals.is_empty() {
            return Err(
                "requested source interval contains no decoded samples; increase fixed_fps".into(),
            );
        }
        review_plan_chunk_count(&media, decoded.frame_signals.len(), decoded.frame_signals[0].pts_ms, chunk_size, duration)?;
        let decode_elapsed_us = decode_started.elapsed().as_micros() as u64;

        // Native media modes extract encoded windows just in time inside
        // the bounded inference tasks.
        let decoded_jpegs = if semantic_decode_enabled && media.mode == SemanticMediaMode::Frames {
            if crate::semantic_infer::review_submission_count(&decoded.frame_signals, chunk_size, semantic_frames_per_chunk, media.context_frames, budget_previous_image) > 20_000 {
                return Err("review exceeds 20000 image submissions across tiers; narrow the interval or split the request".into());
            }
            let indices = crate::semantic_infer::review_frame_indices(
                &decoded.frame_signals,
                chunk_size,
                semantic_frames_per_chunk,
                media.context_frames,
            );
            if indices.len() > 10_000 {
                return Err(
                    "review JPEG count exceeds 10000; narrow the interval or split the request"
                        .into(),
                );
            }
            let jpegs = decode_pipeline.decode_jpegs(
                decode_source,
                sample_fps,
                &indices,
                max_frames as usize,
                semantic_frame_max_edge,
                crop,
            )?;
            if jpegs.len() != indices.len() {
                return Err("JPEG decoder did not produce every requested frame".into());
            }
            if jpegs
                .iter()
                .map(|frame| frame.jpeg_bytes.len())
                .sum::<usize>()
                > 256 * 1024 * 1024
            {
                return Err(
                    "review JPEG bytes exceed 256 MiB; downscale frames or split the request"
                        .into(),
                );
            }
            let timestamps: std::collections::HashMap<_, _> = decoded
                .frame_signals
                .iter()
                .map(|signal| (signal.frame_index, signal.pts_ms))
                .collect();
            let lookup: std::collections::HashMap<u64, DecodedJpegFrame> = jpegs
                .into_iter()
                .map(|mut frame| {
                    frame.pts_ms = timestamps
                        .get(&frame.frame_index)
                        .copied()
                        .unwrap_or(frame.pts_ms);
                    (frame.frame_index, frame)
                })
                .collect();
            Some(lookup)
        } else {
            None
        };
        Ok((
            prepared,
            decoded,
            source_fps,
            sample_fps,
            decoded_jpegs,
            media_info,
            decode_elapsed_us,
        ))
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
    state
        .pipeline_metrics()
        .record_decoded_batch(decoded.frame_signals.len() as u64, decode_elapsed_us);

    let request_id = state.next_request_id();
    let trace_id = payload
        .trace_id
        .unwrap_or_else(|| format!("trace-{}", &request_id[4..]));
    let stream_id = payload.stream_id.unwrap_or_else(|| "stream-0".to_string());
    if let Some(config) = local_audio.as_mut() {
        config.trace = AudioTraceContext {
            run_id: Arc::from(run_id.as_str()),
            request_id: Arc::from(request_id.as_str()),
            stream_id: Arc::from(stream_id.as_str()),
        };
    }
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
                "coordinate_schema": IMAGE_COORDINATE_SCHEMA,
                "coordinates": decoded.coordinates,
                "media_mode": media.mode.as_str(),
                "media_window_ms": media.window_ms,
                "media_overlap_ms": media.overlap_ms,
                "provider_video_fps": media.video_fps,
                "source_start_ms": media.source_start_ms,
                "source_end_ms": media.source_end_ms,
                "semantic_frames_per_chunk": semantic_frames_per_chunk,
                "semantic_context_frames": media.context_frames,
                "video_streams": media_info.as_ref().map(|info| info.video_streams),
                "audio_streams": media_info.as_ref().map(|info| info.audio_streams),
                "audio_channels": media_info.as_ref().map(|info| info.audio_channels),
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
        state.webrtc_config().gate_config.clone(),
    );

    let providers = if semantic_inference {
        state.admitted_provider(&principal)
    } else {
        None
    };
    let semantic_available = local_audio.is_some() || providers.is_some();
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

    // Share the prepared source across bounded clip tasks so every window reads
    // the same prefetched media.
    let prepared_source = Arc::new(prepared_source);
    let clip_decode_pipeline = state.decode_pipeline();
    let gate_started = Instant::now();
    let chunk_preps = prepare_realtime_chunks(
        &decoded.frame_signals,
        chunk_size,
        semantic_frames_per_chunk,
        decoded_jpegs.as_ref(),
        &mut pipeline,
        &clip_decode_pipeline,
        &prepared_source,
        media,
        semantic_decode_enabled,
        crop,
        local_audio.clone(),
    )
    .await;
    let gate_elapsed_us = gate_started.elapsed().as_micros() as u64;
    let gate_analyzed = chunk_preps
        .iter()
        .map(|chunk| chunk.analyzed.len() as u64)
        .sum::<u64>();
    let gate_selected = chunk_preps
        .iter()
        .flat_map(|chunk| chunk.analyzed.iter())
        .filter(|frame| frame.gate_event == GateEventType::KeepKeyframe)
        .count() as u64;
    state
        .pipeline_metrics()
        .record_gate_batch(gate_analyzed, gate_selected, gate_elapsed_us);

    let visual_diff = payload.visual_diff.unwrap_or(false);
    let temporal_chain = visual_diff || payload.temporal_chain.unwrap_or(false);
    let guided_json_str: Option<Arc<str>> = payload
        .output_schema
        .as_ref()
        .and_then(|s| serde_json::to_string(s).ok())
        .map(Arc::from);
    let vlm_concurrency = payload
        .vlm_concurrency
        .unwrap_or(if media.mode == SemanticMediaMode::AudioVideo {
            2
        } else {
            4
        })
        .clamp(1, 64);
    // Same InferenceMetrics instance /metrics reads from, so analyze's tiered
    // passes are attributed to their true provider the same way WHIP's are.
    let analyze_observer: Option<Arc<dyn InferenceObserver>> =
        Some(Arc::new(PipelineInferenceObserver::new(
            Arc::clone(state.inference_metrics_arc()),
            Arc::clone(state.pipeline_metrics_arc()),
        )));
    let (semantic_event_tx, semantic_event_rx) = tokio::sync::mpsc::channel(vlm_concurrency);
    let semantic_dispatch = run_semantic_dispatch(
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
        analyze_observer,
        Some(state.inference_dispatch()),
        Some(semantic_event_tx),
    );
    let semantic_journal = spawn_semantic_journal(
        state.clone(),
        run_id.clone(),
        request_id.clone(),
        stream_id.clone(),
        index_name.clone(),
        media.overlap_ms,
        semantic_event_rx,
    );
    let ((semantic_results, task_end_times), semantic_journal_result) =
        tokio::join!(semantic_dispatch, semantic_journal);
    let semantic_journal_result = semantic_journal_result
        .unwrap_or_else(|error| Err(format!("semantic journal join failure: {error}")));
    if let Err(err) = semantic_journal_result {
        return internal_error(
            &state,
            format!("failed to append semantic_chunk_inferred event: {err}"),
        );
    }

    let assembled = match assemble_realtime_reason_response(
        &state,
        &run_id,
        &stream_id,
        mode,
        model,
        sampling_policy,
        sample_fps,
        source_fps,
        decoded.coordinates,
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

    let generated = assembled.metadata.len();
    let response_metadata = if include_frame_metadata {
        assembled.metadata
    } else {
        Vec::new()
    };
    ok(json!(RealtimeReasonResponse {
        request_id,
        run_id,
        generated,
        markers_emitted: assembled.markers.len(),
        decoded_frames: decoded.frame_signals.len(),
        sample_fps,
        lag_p95_ms: assembled.lag_p95_ms,
        lag_p99_ms: assembled.lag_p99_ms,
        tokens: assembled.tokens,
        frame_metadata_included: include_frame_metadata,
        metadata: response_metadata,
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
    let provider_for_probe = provider.clone();
    let is_saturated = state.inference_metrics().is_high_latency();
    let availability = match tokio::task::spawn_blocking(move || {
        runtime_model_availability(provider_for_probe, is_saturated)
    })
    .await
    {
        Ok(availability) => availability,
        Err(err) => {
            return internal_error(&state, format!("model catalog worker join failure: {err}"));
        }
    };
    let mut models = Vec::with_capacity(
        REQUIRED_MEDIUM_MODELS.len()
            + REQUIRED_SMALL_MODELS.len()
            + EXPERIMENTAL_MODELS.len()
            + GEMINI_MODELS.len(),
    );
    for model in REQUIRED_MEDIUM_MODELS {
        let (status, providers_available) =
            model_availability(provider.as_ref(), &availability, model);
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "medium".to_string(),
            availability: status.to_string(),
            providers_available,
            fallback_candidates: fallback_candidates(model)
                .iter()
                .map(ToString::to_string)
                .collect(),
        });
    }
    for model in REQUIRED_SMALL_MODELS {
        let (status, providers_available) =
            model_availability(provider.as_ref(), &availability, model);
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "small".to_string(),
            availability: status.to_string(),
            providers_available,
            fallback_candidates: fallback_candidates(model)
                .iter()
                .map(ToString::to_string)
                .collect(),
        });
    }
    for model in EXPERIMENTAL_MODELS {
        let (status, providers_available) =
            model_availability(provider.as_ref(), &availability, model);
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "experimental".to_string(),
            availability: status.to_string(),
            providers_available,
            fallback_candidates: fallback_candidates(model)
                .iter()
                .map(ToString::to_string)
                .collect(),
        });
    }
    for model in GEMINI_MODELS {
        let (status, providers_available) =
            model_availability(provider.as_ref(), &availability, model);
        models.push(ModelCatalogItem {
            id: (*model).to_string(),
            tier: "cloud".to_string(),
            availability: status.to_string(),
            providers_available,
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
    metrics.push_str(&InferenceMetrics::render_admission_prometheus(
        state.inference_admission(),
    ));
    metrics.push_str(&state.pipeline_metrics().render_prometheus());
    metrics.push_str(&state.render_media_capacity_prometheus());
    metrics.push_str(&state.delivery_metrics().render_prometheus());
    (axum::http::StatusCode::OK, metrics)
}

#[derive(Clone)]
struct PreparedInferRequest {
    run_id: Option<String>,
    request: InferenceRequest,
    primary_provider: ProviderKind,
    principal: String,
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
            scheduling: vidarax_core::provider::InferenceScheduling::new(
                Arc::from("direct"),
                vidarax_core::admission::LatencyClass::Live,
                timeout_ms,
                timeout_ms.saturating_sub(1).min(1_000),
            ),
        },
        primary_provider,
        principal: state.security_policy().principal_key_from_headers(headers),
    })
}

async fn execute_infer_request(
    state: AppState,
    prepared: PreparedInferRequest,
) -> Result<InferResponse, InferExecutionError> {
    let provider = state
        .admitted_provider(&prepared.principal)
        .ok_or(InferExecutionError {
            code: "internal_error",
            message: "inference providers are not configured".to_string(),
        })?;

    let dispatch_permit =
        state
            .try_acquire_inference_dispatch()
            .map_err(|message| InferExecutionError {
                code: "provider_saturated",
                message,
            })?;
    // The admitted call owns completion and its WAL append through provider
    // exit. Dropping the HTTP waiter only releases the response handle.
    tokio::spawn(
        async move { complete_infer_request(state, prepared, provider, dispatch_permit).await },
    )
    .await
    .map_err(|err| InferExecutionError {
        code: "internal_error",
        message: format!("inference transaction join failure: {err}"),
    })?
}

async fn complete_infer_request(
    state: AppState,
    prepared: PreparedInferRequest,
    provider: Arc<dyn InferenceProvider + Send + Sync>,
    dispatch_permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<InferResponse, InferExecutionError> {
    let _dispatch_permit = dispatch_permit;
    let request_id = state.next_request_id();
    let started = Instant::now();
    state.pipeline_metrics().inc_vlm_inferences();
    let primary_provider_for_metrics = prepared.primary_provider;
    let request_for_provider = prepared.request.clone();
    let result =
        match tokio::task::spawn_blocking(move || provider.infer(&request_for_provider)).await {
            Ok(result) => match result {
                Ok(result) => result,
                Err(err) => {
                    state
                        .pipeline_metrics()
                        .vlm_latency_ms
                        .record(started.elapsed().as_millis() as u64);
                    state.inference_metrics().record_error(
                        primary_provider_for_metrics,
                        started.elapsed().as_millis() as u64,
                    );
                    return Err(map_provider_execution_error(err));
                }
            },
            Err(err) => {
                state
                    .pipeline_metrics()
                    .vlm_latency_ms
                    .record(started.elapsed().as_millis() as u64);
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
        result.usage,
    );
    state
        .pipeline_metrics()
        .vlm_latency_ms
        .record(started.elapsed().as_millis() as u64);

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
        tokens: result.usage,
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
        ProviderError::Saturated { .. } => InferExecutionError {
            code: "provider_saturated",
            message: "inference capacity is temporarily unavailable".to_string(),
        },
        ProviderError::DeadlineMissed => InferExecutionError {
            code: "deadline_missed",
            message: "inference could not start before its deadline".to_string(),
        },
        ProviderError::RequestBudget => InferExecutionError {
            code: "request_budget_exceeded",
            message: "inference request exceeds the configured process budget".to_string(),
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
    if matches!(err.code, "provider_saturated" | "deadline_missed") {
        return service_unavailable(state, err.code, err.message);
    }
    if err.code == "request_budget_exceeded" {
        return validation_error(
            state,
            err.message,
            vec![field_error(
                "request",
                "reduce media bytes or max_tokens".to_string(),
            )],
        );
    }
    internal_error(state, err.message)
}

#[cfg(test)]
mod tests {
    use super::{
        event_media_type_for_blob, event_references_keyframe_blob,
        feedback_events_to_json_for_owned_runs, feedback_payload_error,
        infer_execution_error_to_response, marker_to_emit_event_request, owned_run_ids_from_events,
        parse_provider, run_command_with_timeout, validate_infer_request, AnalyzeMarker,
        InferExecutionError, InferRequest, ProviderKind, MAX_FEEDBACK_CATEGORY_LEN,
        MAX_FEEDBACK_TEXT_LEN,
    };
    use crate::state::AppState;
    use axum::http::HeaderMap;
    use serde_json::json;
    use std::collections::HashSet;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};
    use vidarax_core::timeline::TimelineEvent;

    #[test]
    fn parse_provider_accepts_every_configured_backend() {
        // mlx was wired everywhere (ProviderKind, metrics, backend flavor) except
        // this string boundary, so an /v1/infer with primary_provider=mlx used to
        // 422 and an omitted field defaulted to vLLM, mis-attributing MLX errors.
        assert_eq!(parse_provider(Some("vllm")), Ok(ProviderKind::Vllm));
        assert_eq!(parse_provider(Some("sglang")), Ok(ProviderKind::Sglang));
        assert_eq!(parse_provider(Some("gemini")), Ok(ProviderKind::Gemini));
        assert_eq!(parse_provider(Some("mlx")), Ok(ProviderKind::Mlx));
        assert_eq!(parse_provider(Some("MLX")), Ok(ProviderKind::Mlx));
        assert_eq!(parse_provider(None), Ok(ProviderKind::Vllm));
        assert!(parse_provider(Some("bogus")).is_err());
    }

    #[test]
    fn feedback_payload_error_bounds_category_and_feedback() {
        // Within bounds.
        assert!(feedback_payload_error(7, "accuracy", Some("solid")).is_none());
        assert!(feedback_payload_error(0, "quality", None).is_none());
        // Inclusive upper bounds are accepted.
        assert!(feedback_payload_error(10, &"a".repeat(MAX_FEEDBACK_CATEGORY_LEN), None).is_none());
        assert!(
            feedback_payload_error(5, "accuracy", Some(&"a".repeat(MAX_FEEDBACK_TEXT_LEN)))
                .is_none()
        );
        // Pre-existing rules still hold.
        assert_eq!(
            feedback_payload_error(11, "accuracy", None).unwrap().0,
            "rating"
        );
        assert_eq!(feedback_payload_error(5, "", None).unwrap().0, "category");
        // New length caps reject over-long fields (would otherwise 500 at the reducer).
        assert_eq!(
            feedback_payload_error(5, &"a".repeat(MAX_FEEDBACK_CATEGORY_LEN + 1), None)
                .unwrap()
                .0,
            "category"
        );
        assert_eq!(
            feedback_payload_error(5, "accuracy", Some(&"a".repeat(MAX_FEEDBACK_TEXT_LEN + 1)))
                .unwrap()
                .0,
            "feedback"
        );
    }

    static WAL_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_state(tag: &str) -> AppState {
        let n = WAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("vidarax-handlers-{tag}-{n}.wal"));
        AppState::with_wal_for_tests(path)
    }

    fn plan_media(timestamp_windows: bool) -> super::SemanticMediaConfig {
        super::SemanticMediaConfig {
            mode: super::SemanticMediaMode::Video,
            window_ms: 1000,
            overlap_ms: 0,
            timestamp_windows,
            video_fps: None,
            context_frames: 0,
            source_start_ms: None,
            source_end_ms: None,
            resolution: vidarax_core::provider::MediaResolution::Low,
            persist_evidence: false,
        }
    }

    #[test]
    fn review_plan_legacy_and_frames_use_filtered_sample_chunks() {
        let mut media = plan_media(false);
        for mode in [
            super::SemanticMediaMode::Video,
            super::SemanticMediaMode::Frames,
        ] {
            media.mode = mode;
            assert_eq!(
                super::review_plan_chunk_count(&media, 2048 * 5, 500, 5, Some(300_000)).unwrap(),
                2048
            );
            assert!(
                super::review_plan_chunk_count(&media, 2048 * 5 + 1, 500, 5, Some(300_000))
                    .is_err()
            );
            let error =
                super::review_plan_chunk_count(&media, 300 * 60, 0, 5, Some(300_000)).unwrap_err();
            assert!(error.contains("3600 planned"), "{error}");
            assert_eq!(
                super::review_plan_chunk_count(&media, 10, 2500, 5, None).unwrap(),
                2
            );
        }
    }

    #[test]
    fn review_plan_native_overlap_counts_only_scheduled_windows() {
        let mut media = plan_media(true);
        media.overlap_ms = 500;
        assert_eq!(
            super::review_plan_chunk_count(&media, 60, 0, 5, Some(1000)).unwrap(),
            1
        );
        assert_eq!(
            super::review_plan_chunk_count(&media, 61, 0, 5, Some(1001)).unwrap(),
            2
        );
        assert_eq!(
            super::review_plan_chunk_count(&media, 90, 0, 5, Some(1500)).unwrap(),
            2
        );
        assert_eq!(
            super::review_plan_chunk_count(&media, 91, 0, 5, Some(1501)).unwrap(),
            3
        );
        assert_eq!(
            super::review_plan_chunk_count(&media, 1, 0, 5, Some(1000 + 2047 * 500)).unwrap(),
            2048
        );
        assert!(super::review_plan_chunk_count(&media, 1, 0, 5, Some(1001 + 2047 * 500)).is_err());
        media.source_start_ms = Some(250);
        media.source_end_ms = Some(2750);
        assert_eq!(
            super::review_plan_chunk_count(&media, 150, 267, 5, Some(10_000)).unwrap(),
            4
        );
        media.source_start_ms = None;
        media.source_end_ms = None;
        assert_eq!(
            super::review_plan_chunk_count(&media, 60, 500, 5, Some(1500)).unwrap(),
            1
        );
    }

    #[test]
    fn legacy_clip_duration_rejects_invalid_and_submillisecond_values_before_source_validation() {
        let state = test_state("legacy-duration-invalid");
        let mut payload: crate::models::RealtimeReasonRequest = serde_json::from_value(json!({
            "source_uri": "/nonexistent-unit-test-source.mp4",
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "video_clip_mode": true,
            "semantic_inference": false
        }))
        .unwrap();
        for duration in [
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            f32::from_bits(1),
            0.0001,
            0.00049,
            0.0005,
            0.000999,
            f32::from_bits(0.001_f32.to_bits() - 1),
            f32::NAN,
            f32::INFINITY,
            60.00001,
            f32::from_bits(60.0_f32.to_bits() + 1),
            f32::MAX,
        ] {
            payload.video_clip_duration_s = Some(duration);
            let (status, axum::Json(body)) =
                match super::validate_realtime_reason_params(&state, &payload) {
                    Ok(_) => panic!("invalid legacy duration accepted: {duration:?}"),
                    Err(error) => error,
                };
            assert_eq!(status, axum::http::StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(body["error"]["code"], "validation_error");
            assert_eq!(
                body["error"]["details"][0]["field"], "video_clip_duration_s",
                "duration={duration:?}, body={body}"
            );
        }
    }

    #[test]
    fn legacy_clip_duration_keeps_boundary_rounding_nonzero_and_default_unchanged() {
        let state = test_state("legacy-duration-valid");
        let sequence = WAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let source = std::env::temp_dir().join(format!(
            "vidarax-duration-unit-{}-{sequence}.mp4",
            std::process::id()
        ));
        // Validation only: no media decoder or inference is invoked.
        std::fs::write(&source, b"unit-test path fixture").unwrap();
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _fixture = Fixture(source.clone());
        let mut payload: crate::models::RealtimeReasonRequest = serde_json::from_value(json!({
            "source_uri": source.to_string_lossy(),
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "video_clip_mode": true,
            "semantic_inference": false
        }))
        .unwrap();
        for (duration, expected_ms) in [
            (0.001, 1),
            (0.00149, 1),
            (0.0015, 2),
            (0.00151, 2),
            (0.5, 500),
            (60.0, 60_000),
        ] {
            payload.video_clip_duration_s = Some(duration);
            let params = super::validate_realtime_reason_params(&state, &payload).unwrap_or_else(
                |(status, axum::Json(body))| {
                    panic!("duration={duration}, status={status}, body={body}")
                },
            );
            assert_eq!(params.media.window_ms, expected_ms);
            assert!(params.media.window_ms > params.media.overlap_ms);
        }
        payload.video_clip_duration_s = None;
        let params = super::validate_realtime_reason_params(&state, &payload)
            .unwrap_or_else(|(_, axum::Json(body))| panic!("{body}"));
        assert_eq!(params.media.window_ms, 500);
    }

    #[tokio::test]
    async fn semantic_journal_failure_returns_without_blocking_remaining_completions() {
        use axum::extract::{Path, State};
        use axum::response::IntoResponse;
        use axum::Json;
        use http_body_util::BodyExt;
        use std::sync::{Arc, Mutex};
        use vidarax_core::provider::{
            InferenceProvider, InferenceRequest, InferenceResult, ProviderError, TokenUsage,
        };

        struct FailJournalProvider {
            state: Mutex<Option<AppState>>,
            calls: AtomicU64,
        }

        impl InferenceProvider for FailJournalProvider {
            fn kind(&self) -> ProviderKind {
                ProviderKind::Vllm
            }

            fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                // Earlier ingestion events must commit before the journal fails.
                // Taking the state also breaks the test provider's ownership cycle.
                if let Some(state) = self.state.lock().unwrap().take() {
                    state.set_timeline_append_failure_for_tests(true);
                }
                Ok(InferenceResult {
                    provider: ProviderKind::Vllm,
                    model: Arc::clone(&request.model),
                    output_text: r#"{"event_type":"context_observation","object_label":"frame_context","summary":"ok","description":"chunk completed","confidence":0.95}"#.to_string(),
                    fallback_used: false,
                    finish_reason: Some("stop".to_string()),
                    inference_latency_ms: 1,
                    usage: TokenUsage::default(),
                })
            }
        }

        struct Cleanup {
            state: AppState,
            provider: Arc<FailJournalProvider>,
            directory: std::path::PathBuf,
        }

        impl Drop for Cleanup {
            fn drop(&mut self) {
                self.state.set_timeline_append_failure_for_tests(false);
                self.provider.state.lock().unwrap().take();
                let _ = std::fs::remove_dir_all(&self.directory);
            }
        }

        let n = WAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "vidarax-journal-failure-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let provider = Arc::new(FailJournalProvider {
            state: Mutex::new(None),
            calls: AtomicU64::new(0),
        });
        let state = AppState::with_wal_for_tests_and_endpoints(
            directory.join("timeline.wal"),
            Some(provider.clone()),
        );
        let _cleanup = Cleanup {
            state: state.clone(),
            provider: provider.clone(),
            directory: directory.clone(),
        };
        *provider.state.lock().unwrap() = Some(state.clone());

        let source = directory.join("source.mp4");
        let mut command = Command::new("ffmpeg");
        command
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=10:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-y",
            ])
            .arg(&source);
        assert!(run_command_with_timeout(
            &mut command,
            Duration::from_secs(5),
            "journal test fixture generation timed out",
        )
        .unwrap()
        .status
        .success());

        let run_id = state.next_run_id();
        state
            .append_run_event(&run_id, "run_created", json!({"principal_key":"public"}))
            .unwrap();
        let payload = serde_json::from_value(json!({
            "source_uri": source.to_string_lossy(),
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "sampling_policy": "fixed", "fixed_fps": 10, "max_frames": 30,
            "chunk_size": 5, "semantic_frames_per_chunk": 1, "vlm_concurrency": 1,
            "semantic_timeout_ms": 1000,
        }))
        .unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            super::reason_realtime_run(State(state), Path(run_id), HeaderMap::new(), Json(payload)),
        )
        .await
        .expect("journal failure must close its receiver so completion sends can finish")
        .into_response();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8_lossy(&body);
        assert_eq!(
            status,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "{body}"
        );
        assert!(body.contains("internal_error"), "{body}");
        assert!(provider.calls.load(Ordering::Relaxed) >= 3);
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

    #[test]
    fn marker_to_emit_event_request_maps_marker_fields() {
        let marker = AnalyzeMarker {
            marker_id: "marker-1".to_string(),
            run_id: "run-00000000000000aa".to_string(),
            stream_id: "stream-primary".to_string(),
            event_type: "goal_reached".to_string(),
            status: "active".to_string(),
            start_frame: 42,
            end_frame: 64,
            start_pts_ms: 1400,
            end_pts_ms: 2133,
            confidence: 0.875,
            supersedes_marker_id: Some("marker-0".to_string()),
        };

        let req = marker_to_emit_event_request(&marker);

        assert_eq!(req.run_id, "run-00000000000000aa");
        assert_eq!(req.session_id, "stream-primary");
        assert_eq!(req.frame_index, 42);
        assert_eq!(req.pts_ms, 1400);
        assert_eq!(req.event_type, "goal_reached");
        assert_eq!(req.confidence, 0.875);
        assert_eq!(req.description, "active (42..64 frames, 1400..2133 ms)");
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

    #[tokio::test]
    async fn semantic_journal_finishes_after_waiter_drop_and_chunk_gap() {
        let state = test_state("semantic-journal-cancel-gap");
        let run_id = "run-00000000000000bb";
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let journal = super::spawn_semantic_journal(
            state.clone(),
            run_id.to_string(),
            "request-test".to_string(),
            "stream-primary".to_string(),
            Some("review-test".to_string()),
            0,
            rx,
        );
        drop(journal);
        tx.send((
            1,
            super::ChunkSemanticResult {
                attempted: true,
                raw_output: Some(json!({"description":"completed"})),
                ..Default::default()
            },
        ))
        .await
        .unwrap();
        drop(tx);
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let events = state.read_run_events(run_id).unwrap();
                if let Some(event) = events
                    .iter()
                    .find(|event| event.kind == "semantic_chunk_inferred")
                {
                    let payload: serde_json::Value = serde_json::from_str(&event.payload).unwrap();
                    assert_eq!(payload["chunk_index"], 1);
                    assert_eq!(payload["index_name"], "review-test");
                    assert_eq!(
                        events
                            .iter()
                            .filter(|event| event.kind == "semantic_chunk_inferred")
                            .count(),
                        1
                    );
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed later chunk must append after journal waiter exits");
    }

    #[tokio::test]
    async fn cancelled_infer_waiter_keeps_completion_durable() {
        use std::sync::{mpsc, Arc, Mutex};
        use vidarax_core::provider::{
            InferenceProvider, InferenceRequest, InferenceResult, ProviderError, TokenUsage,
        };

        struct BlockingProvider {
            started: tokio::sync::Notify,
            release: Mutex<mpsc::Receiver<()>>,
        }

        impl InferenceProvider for BlockingProvider {
            fn kind(&self) -> ProviderKind {
                ProviderKind::Vllm
            }

            fn infer(&self, request: &InferenceRequest) -> Result<InferenceResult, ProviderError> {
                self.started.notify_one();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|error| ProviderError::Transport(error.to_string()))?;
                Ok(InferenceResult {
                    provider: ProviderKind::Vllm,
                    model: Arc::clone(&request.model),
                    output_text: "completed".to_string(),
                    fallback_used: false,
                    finish_reason: Some("stop".to_string()),
                    inference_latency_ms: 1,
                    usage: TokenUsage::default(),
                })
            }
        }

        let (release_tx, release_rx) = mpsc::channel();
        let provider = Arc::new(BlockingProvider {
            started: tokio::sync::Notify::new(),
            release: Mutex::new(release_rx),
        });
        let directory = std::env::temp_dir().join(format!(
            "vidarax-infer-cancel-{}-{}",
            std::process::id(),
            WAL_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let state = AppState::with_wal_for_tests_and_endpoints(
            directory.join("timeline.wal"),
            Some(provider.clone()),
        );
        let run_id = "run-00000000000000aa";
        state
            .append_run_event(run_id, "run_created", json!({"principal_key":"public"}))
            .unwrap();
        let mut payload = infer_request(json!({"type":"object"}));
        payload.run_id = Some(run_id.to_string());
        let prepared =
            validate_infer_request(&state, &HeaderMap::new(), payload, "invalid infer payload")
                .await
                .unwrap();
        let state_for_waiter = state.clone();
        let waiter = tokio::spawn(super::execute_infer_request(state_for_waiter, prepared));
        tokio::time::timeout(Duration::from_secs(1), provider.started.notified())
            .await
            .unwrap();
        let available = state.inference_dispatch().available_permits();
        waiter.abort();
        assert!(matches!(waiter.await, Err(error) if error.is_cancelled()));
        assert_eq!(state.inference_dispatch().available_permits(), available);
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let completions = state
                    .read_run_events(run_id)
                    .unwrap()
                    .into_iter()
                    .filter(|event| event.kind == "inference_completed")
                    .count();
                if completions == 1
                    && state.inference_dispatch().available_permits() == available + 1
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("accepted inference must append after its waiter exits");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn direct_infer_keeps_short_deadlines_dispatchable() {
        let state = test_state("short-infer-deadline");
        let mut payload = infer_request(json!({"type": "object"}));
        payload.timeout_ms = Some(1);
        let prepared =
            validate_infer_request(&state, &HeaderMap::new(), payload, "invalid infer payload")
                .await
                .unwrap();

        assert_eq!(prepared.request.scheduling.deadline_ms, 1);
        assert_eq!(prepared.request.scheduling.estimated_service_ms, 0);
    }

    #[test]
    fn inference_capacity_errors_have_non_500_statuses() {
        let state = test_state("infer-capacity-status");
        let deadline = infer_execution_error_to_response(
            &state,
            InferExecutionError {
                code: "deadline_missed",
                message: "deadline missed".to_string(),
            },
        );
        assert_eq!(deadline.0, axum::http::StatusCode::SERVICE_UNAVAILABLE);

        let budget = infer_execution_error_to_response(
            &state,
            InferExecutionError {
                code: "request_budget_exceeded",
                message: "budget exceeded".to_string(),
            },
        );
        assert_eq!(budget.0, axum::http::StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn feedback_list_filters_wal_events_to_owned_runs() {
        let events = vec![
            TimelineEvent {
                seq: 1,
                run_id: "run-00000000000000aa".to_string(),
                stream_id: "stream-0".to_string(),
                pts_ms: 10,
                kind: "operator_feedback_submitted".to_string(),
                payload: json!({
                    "session_id": "sess-a",
                    "rating": 8,
                    "category": "quality",
                    "feedback": "owned",
                })
                .to_string(),
            },
            TimelineEvent {
                seq: 2,
                run_id: "run-00000000000000bb".to_string(),
                stream_id: "stream-0".to_string(),
                pts_ms: 20,
                kind: "operator_feedback_submitted".to_string(),
                payload: json!({
                    "session_id": "sess-b",
                    "rating": 2,
                    "category": "quality",
                    "feedback": "other",
                })
                .to_string(),
            },
        ];
        let owned = HashSet::from(["run-00000000000000aa".to_string()]);

        let filtered = feedback_events_to_json_for_owned_runs(&events, &owned);

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
    fn keyframe_authorization_accepts_only_supported_event_references() {
        let sha = "a".repeat(64);
        assert!(event_references_keyframe_blob(
            "keyframe_stored",
            &json!({ "image_sha256": sha }),
            &"a".repeat(64),
        ));
        assert!(event_references_keyframe_blob(
            "restricted_zone_activity_entered",
            &json!({ "evidence": { "image_sha256": "B".repeat(64) } }),
            &"b".repeat(64),
        ));
        assert!(event_references_keyframe_blob(
            "loading_bay_entry",
            &json!({
                "trigger": { "program_id": "loading-bay" },
                "provenance": { "pipeline_generation": 7 },
                "evidence": { "image_sha256": "C".repeat(64) }
            }),
            &"c".repeat(64),
        ));
        assert!(!event_references_keyframe_blob(
            "untrusted_event",
            &json!({ "evidence": { "image_sha256": "c".repeat(64) } }),
            &"c".repeat(64),
        ));
    }

    #[test]
    fn media_authorization_accepts_only_semantic_evidence_references() {
        let sha = "d".repeat(64);
        for kind in ["semantic_chunk_inferred", "multimodal_moment"] {
            assert_eq!(
                event_media_type_for_blob(
                    kind,
                    &json!({ "evidence": {
                        "media_sha256": sha,
                        "media_type": "video/mp4"
                    } }),
                    &"D".repeat(64),
                )
                .as_deref(),
                Some("video/mp4")
            );
        }
        assert_eq!(
            event_media_type_for_blob(
                "semantic_chunk_inferred",
                &json!({ "feedback_audio": {
                    "media_sha256": "e".repeat(64),
                    "media_type": "audio/wav"
                } }),
                &"e".repeat(64),
            )
            .as_deref(),
            Some("audio/wav")
        );
        assert!(event_media_type_for_blob(
            "untrusted_event",
            &json!({ "evidence": {
                "media_sha256": "d".repeat(64),
                "media_type": "video/mp4"
            } }),
            &"d".repeat(64),
        )
        .is_none());
        assert!(event_media_type_for_blob(
            "multimodal_moment",
            &json!({ "media_sha256": "d".repeat(64) }),
            &"d".repeat(64),
        )
        .is_none());
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

/// Byte caps for feedback fields, mirroring the SpacetimeDB module's reducer
/// limits (spacetime-module). Enforcing them here turns oversized input into a
/// clear 400 instead of letting the reducer reject it and surface an opaque 500.
const MAX_FEEDBACK_CATEGORY_LEN: usize = 64;
const MAX_FEEDBACK_TEXT_LEN: usize = 64 * 1024;

/// First feedback field that fails validation, as (field, message), or None
/// when the payload is within bounds.
fn feedback_payload_error(
    rating: u32,
    category: &str,
    feedback: Option<&str>,
) -> Option<(&'static str, String)> {
    if rating > 10 {
        return Some(("rating", "rating must be between 0 and 10".to_string()));
    }
    if category.is_empty() {
        return Some(("category", "category must not be empty".to_string()));
    }
    if category.len() > MAX_FEEDBACK_CATEGORY_LEN {
        return Some((
            "category",
            format!("category must be at most {MAX_FEEDBACK_CATEGORY_LEN} bytes"),
        ));
    }
    if let Some(feedback) = feedback {
        if feedback.len() > MAX_FEEDBACK_TEXT_LEN {
            return Some((
                "feedback",
                format!("feedback must be at most {MAX_FEEDBACK_TEXT_LEN} bytes"),
            ));
        }
    }
    None
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

    if let Some((field, message)) = feedback_payload_error(
        payload.rating,
        &payload.category,
        payload.feedback.as_deref(),
    ) {
        return validation_error(
            &state,
            "invalid feedback payload",
            vec![field_error(field, message)],
        );
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error;
    }

    let req = crate::spacetime_client::SubmitFeedbackRequest {
        run_id: run_id.clone(),
        session_id: String::new(),
        rating: payload.rating,
        category: payload.category,
        feedback: payload.feedback.unwrap_or_default(),
    };
    // The local WAL is the source of truth. Persist before attempting the
    // optional mirror so feedback remains available in a standalone install
    // and a mirror outage can never turn a durable operator decision into a
    // failed request.
    let feedback_event = match state
        .append_run_event_async(
            &run_id,
            "operator_feedback_submitted",
            json!({
                "session_id": req.session_id,
                "rating": req.rating,
                "category": req.category,
                "feedback": req.feedback,
            }),
        )
        .await
    {
        Ok(event) => event,
        Err(err) => return internal_error(&state, format!("failed to append feedback: {err}")),
    };

    let mut mirrored_to_spacetimedb = false;
    if let Some(stdb) = state.spacetime_client() {
        match stdb.submit_feedback_async(&req).await {
            Ok(()) => mirrored_to_spacetimedb = true,
            Err(err) => tracing::warn!(
                run_id,
                feedback_seq = feedback_event.seq,
                %err,
                "SpacetimeDB feedback mirror failed after local WAL commit"
            ),
        }
    }

    ok(json!({
        "request_id": state.next_request_id(),
        "run_id": run_id,
        "feedback_id": feedback_event.seq,
        "status": "submitted",
        "storage": "local_wal",
        "mirrored_to_spacetimedb": mirrored_to_spacetimedb,
    }))
}

#[tracing::instrument(name = "api.list_feedback", skip_all)]
pub async fn list_feedback(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let principal = state.security_policy().principal_key_from_headers(&headers);
    let all_events = match state.read_all_events_async().await {
        Ok(events) => events,
        Err(err) => return internal_error(&state, format!("failed to read events: {err}")),
    };
    let owned_run_ids = owned_run_ids_from_events(&all_events, &principal);
    let items = feedback_events_to_json_for_owned_runs(&all_events, &owned_run_ids);
    ok(json!({
        "request_id": state.next_request_id(),
        "feedback": items,
        "storage": "local_wal",
    }))
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
    let transaction = tokio::spawn(transition_live_run(
        state.clone(),
        run_id.clone(),
        request_id.clone(),
        "run_deleted",
        false,
    ));
    match transaction.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return internal_error(&state, format!("failed to delete run: {err}"));
        }
        Err(err) => {
            return internal_error(&state, format!("delete transaction join failure: {err}"));
        }
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
        return bad_request_error(
            &state,
            "invalid filename",
            vec![field_error(
                "filename",
                "filename must not contain path separators or traversal sequences".to_string(),
            )],
        )
        .into_response();
    }
    if !allowed_served_file_extension(&filename) {
        return bad_request_error(
            &state,
            "unsupported file type",
            vec![field_error(
                "filename",
                "only mp4, webm, mov, or avi files are served".to_string(),
            )],
        )
        .into_response();
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

    not_found_error(
        &state,
        "file not found",
        vec![field_error("filename", filename)],
    )
    .into_response()
}

/// Return a raw JPEG referenced by a supported evidence event on an owned run.
pub async fn serve_keyframe(
    State(state): State<AppState>,
    Path((run_id, sha256)): Path<(String, String)>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::Response;

    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid keyframe request") {
        return error.into_response();
    }
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return validation_error(
            &state,
            "invalid keyframe request",
            vec![field_error(
                "sha256",
                "sha256 must be 64 hexadecimal characters".to_string(),
            )],
        )
        .into_response();
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }

    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error.into_response(),
    };
    if events.iter().any(|event| event.kind == "run_deleted") {
        return not_found_error(
            &state,
            "run_id was not found",
            vec![field_error("run_id", run_id)],
        )
        .into_response();
    }
    let referenced = events.iter().any(|event| {
        event_references_keyframe_blob(&event.kind, &parse_payload(&event.payload), &sha256)
    });
    if !referenced {
        return not_found_error(
            &state,
            "keyframe was not found",
            vec![field_error("sha256", sha256)],
        )
        .into_response();
    }

    let sha256 = sha256.to_ascii_lowercase();
    let path = state
        .keyframe_blob_root()
        .join(&sha256[..2])
        .join(format!("{sha256}.jpg"));
    let data = match tokio::fs::read(path).await {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return not_found_error(
                &state,
                "keyframe was not found",
                vec![field_error("sha256", sha256)],
            )
            .into_response();
        }
        Err(err) => {
            return internal_error(&state, format!("failed to read keyframe blob: {err}"))
                .into_response();
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CONTENT_LENGTH, data.len())
        .header(
            header::CACHE_CONTROL,
            "private, max-age=31536000, immutable",
        )
        .header(header::ETAG, format!("\"{sha256}\""))
        .body(Body::from(data))
        .unwrap()
}

/// Return raw binary media referenced by an owned semantic event.
pub async fn serve_media(
    State(state): State<AppState>,
    Path((run_id, sha256)): Path<(String, String)>,
    headers: HeaderMap,
) -> axum::response::Response {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::Response;

    if let Some(error) = validate_run_id_or_error(&state, &run_id, "invalid media request") {
        return error.into_response();
    }
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return validation_error(
            &state,
            "invalid media request",
            vec![field_error(
                "sha256",
                "sha256 must be 64 hexadecimal characters".to_string(),
            )],
        )
        .into_response();
    }
    if let Err(error) = load_run_snapshot(&state, &headers, &run_id) {
        return error.into_response();
    }
    let events = match load_existing_events(&state, &run_id).await {
        Ok(events) => events,
        Err(error) => return error.into_response(),
    };
    if events.iter().any(|event| event.kind == "run_deleted") {
        return not_found_error(
            &state,
            "run_id was not found",
            vec![field_error("run_id", run_id)],
        )
        .into_response();
    }
    let referenced_media_type = events.iter().find_map(|event| {
        event_media_type_for_blob(&event.kind, &parse_payload(&event.payload), &sha256)
    });
    let Some(media_type) = referenced_media_type else {
        return not_found_error(
            &state,
            "media was not found",
            vec![field_error("sha256", sha256)],
        )
        .into_response();
    };

    let sha256 = sha256.to_ascii_lowercase();
    let extension = match media_type.as_str() {
        "video/mp4" => "mp4",
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        _ => {
            return internal_error(&state, "event references an unsupported media type")
                .into_response()
        }
    };
    let path = state
        .media_blob_root()
        .join(&sha256[..2])
        .join(format!("{sha256}.{extension}"));
    let data = match tokio::fs::read(path).await {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return not_found_error(
                &state,
                "media was not found",
                vec![field_error("sha256", sha256)],
            )
            .into_response();
        }
        Err(error) => {
            return internal_error(&state, format!("failed to read media blob: {error}"))
                .into_response();
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, media_type)
        .header(header::CONTENT_LENGTH, data.len())
        .header(header::ACCEPT_RANGES, "bytes")
        .header(
            header::CACHE_CONTROL,
            "private, max-age=31536000, immutable",
        )
        .header(header::ETAG, format!("\"{sha256}\""))
        .body(Body::from(data))
        .unwrap()
}

fn event_references_keyframe_blob(kind: &str, payload: &Value, sha256: &str) -> bool {
    let referenced_sha = match kind {
        "keyframe_stored" => payload.get("image_sha256"),
        "restricted_zone_activity_entered" => payload
            .get("evidence")
            .and_then(|evidence| evidence.get("image_sha256")),
        _ if payload
            .get("trigger")
            .and_then(|trigger| trigger.get("program_id"))
            .and_then(Value::as_str)
            .is_some()
            && payload
                .get("provenance")
                .and_then(|provenance| provenance.get("pipeline_generation"))
                .and_then(Value::as_u64)
                .is_some() =>
        {
            payload
                .get("evidence")
                .and_then(|evidence| evidence.get("image_sha256"))
        }
        _ => None,
    };
    referenced_sha
        .and_then(Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case(sha256))
}

fn event_media_type_for_blob(kind: &str, payload: &Value, sha256: &str) -> Option<String> {
    if !matches!(kind, "semantic_chunk_inferred" | "multimodal_moment") {
        return None;
    }
    for field in ["evidence", "feedback_audio"] {
        let Some(media) = payload.get(field) else {
            continue;
        };
        if media
            .get("media_sha256")
            .and_then(Value::as_str)
            .is_some_and(|value| value.eq_ignore_ascii_case(sha256))
        {
            return media
                .get("media_type")
                .and_then(Value::as_str)
                .map(ToString::to_string);
        }
    }
    None
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

fn feedback_events_to_json_for_owned_runs(
    events: &[TimelineEvent],
    owned_run_ids: &HashSet<String>,
) -> Vec<Value> {
    events
        .iter()
        .filter(|event| {
            event.kind == "operator_feedback_submitted" && owned_run_ids.contains(&event.run_id)
        })
        .map(|event| {
            let payload = parse_payload(&event.payload);
            json!({
                "id": event.seq,
                "run_id": event.run_id,
                "session_id": payload.get("session_id").and_then(Value::as_str).unwrap_or(""),
                "rating": payload.get("rating").and_then(Value::as_u64).unwrap_or(0),
                "category": payload.get("category").and_then(Value::as_str).unwrap_or(""),
                "feedback": payload.get("feedback").and_then(Value::as_str).unwrap_or(""),
                "timestamp_micros": event.pts_ms.saturating_mul(1000),
                "storage": "local_wal",
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

pub(crate) fn load_run_snapshot(
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
    saturated: bool,
    providers: Vec<ProviderKind>,
}

fn runtime_model_availability(
    provider: Option<Arc<dyn InferenceProvider + Send + Sync>>,
    is_saturated: bool,
) -> RuntimeAvailability {
    let Some(provider) = provider else {
        return RuntimeAvailability {
            saturated: false,
            providers: Vec::new(),
        };
    };
    RuntimeAvailability {
        saturated: is_saturated,
        providers: provider.available_kinds(),
    }
}

fn model_availability(
    provider: Option<&Arc<dyn InferenceProvider + Send + Sync>>,
    runtime: &RuntimeAvailability,
    model: &str,
) -> (&'static str, Vec<String>) {
    let Some(provider) = provider else {
        return ("unavailable", Vec::new());
    };
    let providers = provider
        .configured_kinds_for_model(model)
        .into_iter()
        .filter(|kind| runtime.providers.contains(kind))
        .map(|kind| kind.name().to_string())
        .collect::<Vec<_>>();
    let status = if providers.is_empty() {
        "unavailable"
    } else if runtime.saturated {
        "saturated"
    } else {
        "ready"
    };
    (status, providers)
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
        "mlx" => Ok(ProviderKind::Mlx),
        _ => Err("primary_provider must be one of: vllm, sglang, gemini, mlx"),
    }
}

fn provider_name(kind: ProviderKind) -> &'static str {
    kind.name()
}
