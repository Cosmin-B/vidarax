/**
 * Vidarax SDK — public entry point.
 *
 * ```typescript
 * import { Vidarax } from 'vidarax'
 *
 * const v = new Vidarax('http://localhost:8080')
 * const result = await v.analyze('video.mp4', { prompt: 'Describe what happens' })
 * for await (const event of result.events()) {
 *   console.log(event.kind, event.payload)
 * }
 * ```
 */

// ─── Main client ──────────────────────────────────────────────────────────────
export { Vidarax } from "./client.js";

// ─── Error classes ────────────────────────────────────────────────────────────
export {
  VidaraxError,
  HttpError,
  NetworkError,
  RetryExhaustedError,
  UploadError,
  ParseError,
  isVidaraxError,
  isHttpError,
  isNetworkError,
} from "./errors.js";

export type { ApiErrorBody, FieldError } from "./errors.js";

// ─── Types ────────────────────────────────────────────────────────────────────
export type {
  // Primitives
  EpochMs,
  IsoTimestamp,
  SamplingPolicy,
  RunStatus,
  ModelTier,
  ProgressCallback,

  // SDK-level options
  VidaraxOptions,
  AnalyzeOptions,
  InferOptions,
  InferBatchOptions,
  AnalysisResult,

  // Run
  Run,
  CreateRunRequest,
  CreateRunResponse,

  // Events
  AgentEvent,
  EventsResponse,

  // Markers
  Marker,
  MarkerQuery,
  MarkersResponse,

  // Ingest
  IngestRequest,
  IngestResponse,

  // Frame signals
  FrameSignal,

  // Analysis
  AnalyzeWindow,
  AnalyzeObject,
  AnalyzeEvent,
  AnalyzeAnnotations,
  AnalyzeTrace,
  AnalyzeFrameMetadata,
  AnalyzeFramesRequest,
  AnalyzeFramesResponse,

  // Clip mode
  ClipConfig,

  // Realtime reason
  RealtimeReasonRequest,
  RealtimeReasonResponse,

  // Inference
  InferRequest,
  InferResponse,
  InferBatchRequest,
  InferBatchResponse,
  InferBatchItemResult,

  // Models
  Model,
  ModelCatalogResponse,

  // Health
  HealthStatus,

  // Upload
  UploadResponse,

  // Query
  QueryRequest,
  QueryResponse,

  // Feedback
  FeedbackRequest,
  FeedbackItem,
  FeedbackListResponse,

  // WHIP
  AttachStreamRequest,
  WhipSession,

  // Search
  SearchResult,
} from "./types.js";
