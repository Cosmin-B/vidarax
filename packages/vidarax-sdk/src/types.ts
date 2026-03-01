/**
 * Core domain types for the Vidarax SDK.
 *
 * These interfaces map 1:1 to the Rust structs defined in
 * `crates/vidarax-api/src/models.rs` and the JSON shapes produced by the
 * handlers in `crates/vidarax-api/src/handlers.rs`.
 */

// ─── Shared primitives ────────────────────────────────────────────────────────

/** Milliseconds since the Unix epoch (u64 in Rust). */
export type EpochMs = number;

/** ISO-8601 UTC timestamp string. */
export type IsoTimestamp = string;

/** Sampling strategy for frame extraction. */
export type SamplingPolicy = "source_fps_adaptive" | "fixed";

/** Run lifecycle state (lowercased Debug repr from Rust). */
export type RunStatus = "pending" | "processing" | "completed" | "cancelled" | "failed";

/** Model tier. */
export type ModelTier = "small" | "medium";

// ─── Run ─────────────────────────────────────────────────────────────────────

/**
 * A Vidarax run resource, as returned by GET /v1/runs and GET /v1/runs/{id}.
 *
 * Corresponds to the JSON built in `list_runs` / `get_run` handlers.
 */
export interface Run {
  run_id: string;
  status: RunStatus;
  mode: string | null;
  model: string | null;
  source_uri: string | null;
  created_at: IsoTimestamp | null;
  updated_at: IsoTimestamp | null;
}

/** Request body for POST /v1/runs. */
export interface CreateRunRequest {
  mode?: string;
  model?: string;
}

/** Response body from POST /v1/runs. */
export interface CreateRunResponse {
  run_id: string;
  request_id: string;
  status: string;
  mode: string;
  model: string | null;
}

// ─── Events ──────────────────────────────────────────────────────────────────

/**
 * A raw timeline event stored against a run.
 *
 * Returned as elements of the `events` array in GET /v1/runs/{id}/events.
 */
export interface AgentEvent {
  seq: number;
  pts_ms: EpochMs;
  kind: string;
  payload: Record<string, unknown>;
}

/** Response envelope from GET /v1/runs/{id}/events. */
export interface EventsResponse {
  request_id: string;
  run_id: string;
  events: AgentEvent[];
}

// ─── Markers ─────────────────────────────────────────────────────────────────

/**
 * A temporal marker emitted during analysis.
 *
 * Maps to `AnalyzeMarker` in models.rs.
 */
export interface Marker {
  marker_id: string;
  run_id: string;
  stream_id: string;
  event_type: string;
  status: string;
  start_frame: number;
  end_frame: number;
  start_pts_ms: EpochMs;
  end_pts_ms: EpochMs;
  confidence: number;
  supersedes_marker_id: string | null;
}

/** Query parameters for GET /v1/runs/{id}/markers. */
export interface MarkerQuery {
  status?: string;
  event_type?: string;
  from_frame?: number;
  to_frame?: number;
}

/** Response envelope from GET /v1/runs/{id}/markers. */
export interface MarkersResponse {
  request_id: string;
  run_id: string;
  markers: Marker[];
}

// ─── Ingest ──────────────────────────────────────────────────────────────────

/** Request body for POST /v1/runs/{id}/ingest. */
export interface IngestRequest {
  source_uri: string;
  sampling_policy?: SamplingPolicy;
  /** Required when sampling_policy is "fixed". */
  fixed_fps?: number;
  /** Alias for fixed_fps. */
  sample_fps?: number;
  max_frames?: number;
  stream_id?: string;
}

/** Response body from POST /v1/runs/{id}/ingest (source_uri path). */
export interface IngestResponse {
  request_id: string;
  run_id: string;
  status: string;
  decoded_frames?: number;
  source_uri?: string;
  sampling_policy?: string;
  source_fps?: number | null;
  sample_fps?: number;
}

// ─── Frame signals ────────────────────────────────────────────────────────────

/**
 * Pre-computed frame signal for a single decoded video frame.
 *
 * Maps to `AnalyzeFrameInput` in models.rs.
 * All score values must be normalised to [0.0, 1.0].
 */
export interface FrameSignal {
  frame_index: number;
  pts_ms: EpochMs;
  perceptual_hash: number;
  /** Normalised [0.0, 1.0]. */
  luma_mean: number;
  /** Normalised [0.0, 1.0]. */
  flicker_score: number;
  /** Normalised [0.0, 1.0]. */
  ghosting_score: number;
  /** Normalised [0.0, 1.0]. */
  noise_variance_score: number;
}

// ─── Analysis ────────────────────────────────────────────────────────────────

/** Sub-window metadata within an analysis result. Maps to `AnalyzeWindow`. */
export interface AnalyzeWindow {
  start_ms: EpochMs;
  end_ms: EpochMs;
  segment_id: string;
  source: string;
}

/** A detected object within an annotated frame. Maps to `AnalyzeObject`. */
export interface AnalyzeObject {
  label: string;
  score: number;
}

/** A detected event within an annotated frame. Maps to `AnalyzeEvent`. */
export interface AnalyzeEvent {
  type: string;
  score: number;
  description: string;
}

/** Aggregated annotations for a single frame window. Maps to `AnalyzeAnnotations`. */
export interface AnalyzeAnnotations {
  summary: string;
  objects: AnalyzeObject[];
  events: AnalyzeEvent[];
}

/** Trace context attached to an analysis result. Maps to `AnalyzeTrace`. */
export interface AnalyzeTrace {
  request_id: string;
  trace_id: string;
  span_id: string;
}

/** Per-frame analysis metadata. Maps to `AnalyzeFrameMetadata`. */
export interface AnalyzeFrameMetadata {
  run_id: string;
  stream_id: string;
  frame_index: number;
  pts_ms: EpochMs;
  mode: string;
  model: string;
  sampling_policy: string;
  sample_fps: number;
  window: AnalyzeWindow;
  annotations: AnalyzeAnnotations;
  confidence: number;
  fallback: { used: boolean };
  trace: AnalyzeTrace;
  ordering_key: string;
  finish_reason: string | null;
}

/** Request body for POST /v1/runs/{id}/analyze. Maps to `AnalyzeFramesRequest`. */
export interface AnalyzeFramesRequest {
  model: string;
  mode?: string;
  stream_id?: string;
  sampling_policy?: SamplingPolicy;
  fixed_fps?: number;
  /** If omitted the server loads frames from prior ingest events. */
  frames?: FrameSignal[];
  window_size?: number;
  segment_ms?: number;
  trace_id?: string;
}

/** Response body from POST /v1/runs/{id}/analyze. Maps to `AnalyzeFramesResponse`. */
export interface AnalyzeFramesResponse {
  request_id: string;
  run_id: string;
  generated: number;
  metadata: AnalyzeFrameMetadata[];
  markers: Marker[];
}

// ─── Clip mode ────────────────────────────────────────────────────────────────

/** Multi-frame clip configuration for streaming / WHIP sessions. Maps to `ClipConfig`. */
export interface ClipConfig {
  /** Frames per second to sample (1–30, default 6). */
  target_fps?: number;
  /** Duration of each clip window in seconds (0.1–60, default 0.5). */
  clip_length_seconds?: number;
  /** Minimum delay between clip emissions in seconds (0–60, default 0.5). */
  delay_seconds?: number;
}

// ─── Realtime reason ─────────────────────────────────────────────────────────

/** Request body for POST /v1/runs/{id}/reason. Maps to `RealtimeReasonRequest`. */
export interface RealtimeReasonRequest {
  source_uri: string;
  model: string;
  mode?: string;
  stream_id?: string;
  sampling_policy?: SamplingPolicy;
  fixed_fps?: number;
  max_frames?: number;
  chunk_size?: number;
  window_size?: number;
  segment_ms?: number;
  marker_correction_window_frames?: number;
  semantic_inference?: boolean;
  semantic_frames_per_chunk?: number;
  semantic_timeout_ms?: number;
  primary_provider?: string;
  semantic_prompt?: string;
  first_pass_model?: string;
  second_pass_model?: string;
  second_pass_threshold?: number;
  trace_id?: string;
  output_schema?: Record<string, unknown>;
  clip_mode?: ClipConfig;
}

/** Response body from POST /v1/runs/{id}/reason. Maps to `RealtimeReasonResponse`. */
export interface RealtimeReasonResponse {
  request_id: string;
  run_id: string;
  generated: number;
  markers_emitted: number;
  decoded_frames: number;
  sample_fps: number;
  lag_p95_ms: number;
  lag_p99_ms: number;
  metadata: AnalyzeFrameMetadata[];
  markers: Marker[];
}

// ─── Inference ───────────────────────────────────────────────────────────────

/** Single inference request body for POST /v1/infer. Maps to `InferRequest`. */
export interface InferRequest {
  model: string;
  prompt: string;
  run_id?: string;
  max_tokens?: number;
  temperature?: number;
  timeout_ms?: number;
  allow_fallback?: boolean;
  primary_provider?: string;
  output_schema?: Record<string, unknown>;
}

/** Response body from POST /v1/infer. Maps to `InferResponse`. */
export interface InferResponse {
  request_id: string;
  run_id: string | null;
  provider: string;
  model: string;
  fallback_used: boolean;
  output_text: string;
  finish_reason: string | null;
  inference_latency_ms: number;
}

/** Request body for POST /v1/infer/batch. Maps to `InferBatchRequest`. */
export interface InferBatchRequest {
  requests: InferRequest[];
  max_parallel?: number;
}

/** Per-item result within a batch inference response. Maps to `InferBatchItemResult`. */
export interface InferBatchItemResult {
  index: number;
  ok: boolean;
  result: InferResponse | null;
  error: { code: string; message: string } | null;
}

/** Response body from POST /v1/infer/batch. Maps to `InferBatchResponse`. */
export interface InferBatchResponse {
  request_id: string;
  processed: number;
  succeeded: number;
  failed: number;
  results: InferBatchItemResult[];
}

// ─── Models ──────────────────────────────────────────────────────────────────

/** A single model entry from the catalog. Maps to `ModelCatalogItem`. */
export interface Model {
  id: string;
  tier: ModelTier;
  availability: string;
  providers_available: string[];
  fallback_candidates: string[];
}

/** Response body from GET /v1/models. Maps to `ModelCatalogResponse`. */
export interface ModelCatalogResponse {
  request_id: string;
  models: Model[];
}

// ─── Health ───────────────────────────────────────────────────────────────────

/** Response body from GET /v1/health. */
export interface HealthStatus {
  status: "ok" | string;
}

// ─── File upload ─────────────────────────────────────────────────────────────

/** Response body from POST /v1/upload. */
export interface UploadResponse {
  file_path: string;
}

// ─── Query ────────────────────────────────────────────────────────────────────

/** Request body for POST /v1/query. */
export interface QueryRequest {
  run_id: string;
  kind?: string;
  from_seq?: number;
}

/** Response body from POST /v1/query. */
export interface QueryResponse {
  request_id: string;
  query: QueryRequest;
  matches: AgentEvent[];
}

// ─── Feedback ────────────────────────────────────────────────────────────────

/** Request body for POST /v1/runs/{id}/feedback. Maps to `FeedbackRequest`. */
export interface FeedbackRequest {
  /** 0–10 rating scale. */
  rating: number;
  category: string;
  feedback?: string;
}

/** A stored feedback row from GET /v1/feedback. */
export interface FeedbackItem {
  id: number | string;
  run_id: string;
  session_id: string;
  rating: number;
  category: string;
  feedback: string;
  timestamp_micros: number;
}

/** Response body from GET /v1/feedback. */
export interface FeedbackListResponse {
  request_id: string;
  feedback: FeedbackItem[];
}

// ─── WHIP ─────────────────────────────────────────────────────────────────────

/** Configuration sent when attaching a WHIP stream session. */
export interface AttachStreamRequest {
  prompt?: string;
  max_output_tokens_per_second?: number;
  clip_mode?: ClipConfig;
}

/** An active WHIP session returned after a successful offer exchange. */
export interface WhipSession {
  sessionId: string;
  /** The SDP answer returned by the server. */
  answerSdp: string;
  /** Absolute URL for trickle-ICE PATCH and DELETE. */
  resourceUrl: string;
}

// ─── SDK-level convenience types ─────────────────────────────────────────────

/** Options accepted by `Vidarax.analyze()`. */
export interface AnalyzeOptions {
  prompt?: string;
  model?: string;
  mode?: string;
  samplingPolicy?: SamplingPolicy;
  fixedFps?: number;
  windowSize?: number;
  segmentMs?: number;
  maxFrames?: number;
}

/** Options accepted by `Vidarax.infer()`. */
export type InferOptions = Omit<InferRequest, "model" | "prompt">;

/** Options accepted by `Vidarax.inferBatch()`. */
export type InferBatchOptions = Pick<InferBatchRequest, "max_parallel">;

/** Search result placeholder for future search endpoint. */
export interface SearchResult {
  run_id: string;
  score: number;
  description: string;
  pts_ms: EpochMs;
}

/** SDK constructor options. */
export interface VidaraxOptions {
  /** Optional bearer / x-api-key value sent with every request. */
  apiKey?: string;
  /** Maximum number of automatic retries on transient failures (default: 3). */
  maxRetries?: number;
  /** Base delay in ms for exponential back-off (default: 200). */
  retryBaseDelayMs?: number;
  /** Request timeout in ms (default: 30 000). */
  timeoutMs?: number;
}

/** Progress callback signature used by `Vidarax.uploadFile()`. */
export type ProgressCallback = (loaded: number, total: number) => void;

/**
 * High-level result handle returned by `Vidarax.analyze()`.
 *
 * Provides convenience async-iterable access to events and markers stored
 * against the underlying run.
 */
export interface AnalysisResult {
  runId: string;
  analyzeResponse: AnalyzeFramesResponse;
  /** Stream all timeline events for this run. */
  events(): AsyncGenerator<AgentEvent>;
  /** Stream all markers for this run. */
  markers(): AsyncGenerator<Marker>;
}
