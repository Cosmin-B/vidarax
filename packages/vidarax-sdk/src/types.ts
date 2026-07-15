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

/** Fractional region of interest within a video frame. */
export interface CropRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

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
  semantic_frame_max_edge?: number;
  crop?: CropRegion;
  semantic_timeout_ms?: number;
  semantic_prompt?: string;
  first_pass_model?: string;
  second_pass_model?: string;
  second_pass_threshold?: number;
  trace_id?: string;
  output_schema?: Record<string, unknown>;
  index_name?: string;
  temporal_chain?: boolean;
  visual_diff?: boolean;
  video_clip_mode?: boolean;
  video_clip_duration_s?: number;
  vlm_concurrency?: number;
}

/** Token and model-time totals for a realtime reasoning run. */
export interface TokenMetrics {
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  total_tokens: number;
  inference_latency_ms: number;
  chunks_analyzed: number;
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
  tokens: TokenMetrics;
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

/** Raw response body from POST /v1/infer. Maps to `InferResponse`. */
export interface InferResponse {
  request_id: string;
  run_id: string | null;
  provider: string;
  model: string;
  fallback_used: boolean;
  output_text: string;
  finish_reason: string | null;
  inference_latency_ms: number;
  tokens: TokenUsage;
}

/** Token usage returned by inference endpoints. */
export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  thinking_tokens: number;
  total_tokens: number;
}

/**
 * High-level result handle returned by `Vidarax.infer()`.
 *
 * Mirrors the shape used by Overshoot-compatible clients, exposing a flat set
 * of fields that cover both success and failure paths without forcing callers
 * to dig into nested objects.
 *
 * @example
 * const result = await v.infer({
 *   model: 'Qwen/Qwen3-VL-2B-Instruct',
 *   prompt: 'Count the people in the frame',
 *   output_schema: { type: 'object', properties: { count: { type: 'number' } }, required: ['count'] }
 * })
 * if (result.ok) {
 *   const data = result.resultJson<{ count: number }>()
 *   console.log('People count:', data.count)
 * }
 */
export class InferResult {
  /** Model output — plain text, or a JSON string when `output_schema` was supplied. */
  readonly result: string;

  /** `true` when inference succeeded (HTTP 200). */
  readonly ok: boolean;

  /** Error message when `ok` is `false`, otherwise `null`. */
  readonly error: string | null;

  /** The reason the model stopped generating. */
  readonly finish_reason: "stop" | "length" | "content_filter" | null;

  /** Time from request dispatch to first token in milliseconds. */
  readonly inference_latency_ms: number;

  /** End-to-end round-trip time in milliseconds. */
  readonly total_latency_ms: number;

  /** Unique result identifier (matches `request_id` from the API). */
  readonly id: string;

  /** Model identifier that served this request. */
  readonly model_name: string;

  /** Inference backend that executed the request, e.g. `"vllm"` or `"sglang"`. */
  readonly provider: string;

  /** Alias for `provider`, kept for existing high-level SDK callers. */
  readonly model_backend: string;

  /** The prompt that was submitted to the model. */
  readonly prompt: string;

  /** Whether the provider chain fell back from the primary provider. */
  readonly fallback_used: boolean;

  /** Associated run ID when the inference was tied to a run, otherwise `null`. */
  readonly run_id: string | null;

  /** Provider-reported token usage. */
  readonly tokens: TokenUsage;

  constructor(fields: {
    result: string;
    ok: boolean;
    error: string | null;
    finish_reason: "stop" | "length" | "content_filter" | null;
    inference_latency_ms: number;
    total_latency_ms: number;
    id: string;
    model_name: string;
    provider: string;
    prompt: string;
    fallback_used: boolean;
    run_id: string | null;
    tokens: TokenUsage;
  }) {
    this.result = fields.result;
    this.ok = fields.ok;
    this.error = fields.error;
    this.finish_reason = fields.finish_reason;
    this.inference_latency_ms = fields.inference_latency_ms;
    this.total_latency_ms = fields.total_latency_ms;
    this.id = fields.id;
    this.model_name = fields.model_name;
    this.provider = fields.provider;
    this.model_backend = fields.provider;
    this.prompt = fields.prompt;
    this.fallback_used = fields.fallback_used;
    this.run_id = fields.run_id;
    this.tokens = fields.tokens;
  }

  /**
   * Parse `result` as JSON and cast it to `T`.
   *
   * Use this when the request included an `output_schema` and you expect a
   * structured JSON object in `result`.
   *
   * @throws `SyntaxError` when `result` is not valid JSON.
   *
   * @example
   * const data = result.resultJson<{ count: number; description: string }>()
   * console.log(data.count)
   */
  resultJson<T>(): T {
    return JSON.parse(this.result) as T;
  }
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

/** User-defined interaction object returned by a guided semantic pass. */
export type Interaction = Record<string, unknown>;

/** Response body from GET /v1/runs/{id}/interactions. */
export interface InteractionsResponse {
  run_id: string;
  count: number;
  interactions: Interaction[];
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
  crop?: CropRegion;
}

/** Request body for PATCH /v1/stream/whip/{session}/prompt. */
export interface WhipPromptUpdateRequest {
  prompt: string;
  output_schema?: Record<string, unknown> | null;
}

/** Response body from PATCH /v1/stream/whip/{session}/prompt. */
export interface WhipPromptUpdateResponse {
  session_id: string;
  prompt: string;
  output_schema: Record<string, unknown> | null;
}

/** An active WHIP session returned after a successful offer exchange. */
export interface WhipSession {
  sessionId: string;
  /** Server-side run ID created for this WHIP session, when returned by the API. */
  runId?: string;
  /** The SDP answer returned by the server. */
  answerSdp: string;
  /** Absolute URL for trickle-ICE PATCH and DELETE. */
  resourceUrl: string;
}

// ─── SDK-level convenience types ─────────────────────────────────────────────

/**
 * Options accepted by `Vidarax.analyze()`.
 *
 * There is no `prompt` field here on purpose: `analyze()` drives the
 * deterministic frame-signal pipeline (`AnalyzeFramesRequest` on the server),
 * which has no prompt input to accept. For prompt-driven semantic analysis,
 * use `Vidarax.reason()` and its `semantic_prompt` field instead.
 */
export interface AnalyzeOptions {
  model?: string;
  mode?: string;
  samplingPolicy?: SamplingPolicy;
  fixedFps?: number;
  windowSize?: number;
  segmentMs?: number;
  maxFrames?: number;
}

/**
 * Options accepted by `Vidarax.infer()`.
 *
 * `model` and `prompt` are required.  All other fields are optional and map
 * directly to the `InferRequest` body sent to the API.
 *
 * Supply `output_schema` with a JSON Schema object to request structured JSON
 * output from the model.  The schema is forwarded to the backend as
 * `guided_json`.  Parse the structured response with `InferResult.resultJson()`.
 */
export interface InferOptions {
  /** Model to use for inference. */
  model: string;
  /** Prompt text to send to the model. */
  prompt: string;
  /** Optional run context to associate with this inference. */
  run_id?: string;
  /** Maximum number of tokens to generate. */
  max_tokens?: number;
  /** Sampling temperature (0–2). */
  temperature?: number;
  /** Request timeout override in milliseconds. */
  timeout_ms?: number;
  /** Whether the server may fall back to a secondary provider. */
  allow_fallback?: boolean;
  /** Preferred primary provider for the request. */
  primary_provider?: string;
  /**
   * JSON Schema for structured output.
   *
   * When provided the model is guided to return a JSON object that conforms to
   * this schema.  Retrieve the parsed object via `InferResult.resultJson<T>()`.
   *
   * @example
   * output_schema: {
   *   type: 'object',
   *   properties: { count: { type: 'number' }, description: { type: 'string' } },
   *   required: ['count'],
   * }
   */
  output_schema?: Record<string, unknown>;
}

/** Options accepted by `Vidarax.inferBatch()`. */
export type InferBatchOptions = Pick<InferBatchRequest, "max_parallel">;

/** A single hit returned by `POST /v1/search`. */
export interface SearchHit {
  seq: number;
  run_id: string;
  pts_ms: EpochMs;
  kind: string;
  description: string;
  index_name: string | null;
}

/**
 * @deprecated Use `SearchHit` for individual hits. `Vidarax.search()` now returns `SearchResponse`.
 */
export type SearchResult = SearchHit;

/** Response body from `POST /v1/search`. */
export interface SearchResponse {
  request_id: string;
  scanned: number;
  total_hits: number;
  hits: SearchHit[];
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
 * Provides convenience async-iterable access to the current event and marker
 * snapshots stored against the underlying run.
 */
export interface AnalysisResult {
  runId: string;
  analyzeResponse: AnalyzeFramesResponse;
  /** Iterate over the current event snapshot for this run. */
  events(): AsyncGenerator<AgentEvent>;
  /** Iterate over the current marker snapshot for this run. */
  markers(): AsyncGenerator<Marker>;
}
