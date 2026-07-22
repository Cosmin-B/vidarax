/**
 * Vidarax SDK — public entry point.
 *
 * ```typescript
 * import { Vidarax } from 'vidarax'
 *
 * const v = new Vidarax('http://localhost:8080', { apiKey: 'dev-key' })
 * const result = await v.analyze('/srv/vidarax-media/video.mp4', { mode: 'balanced' })
 * for (const event of await v.getEvents(result.runId)) {
 *   console.log(event.kind, event.payload)
 * }
 * ```
 */

// ─── Main client ──────────────────────────────────────────────────────────────
export { Vidarax } from "./client.js";

// ─── Result classes ───────────────────────────────────────────────────────────
export { InferResult } from "./types.js";

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
  CropRegion,
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
  FrameCoordinates,
  PixelExtent,
  PixelRect,
  AnalyzeFramesRequest,
  AnalyzeFramesResponse,

  // Clip mode
  ClipConfig,

  // Realtime reason
  RealtimeReasonRequest,
  RealtimeReasonResponse,
  TokenMetrics,

  // Inference
  InferRequest,
  InferResponse,
  InferBatchRequest,
  InferBatchResponse,
  InferBatchItemResult,
  TokenUsage,

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
  Interaction,
  InteractionsResponse,

  // Feedback
  FeedbackRequest,
  FeedbackSubmitResponse,
  FeedbackItem,
  FeedbackListResponse,

  // Policy revisions
  PolicyStatus,
  NormalizedRect,
  RestrictedZonePolicyParameters,
  PolicyParameters,
  CreatePolicyRequest,
  PolicyRevision,
  PolicyApplication,
  PolicyResponse,
  PolicyListResponse,
  ActivatePolicyRequest,
  RollbackPolicyRequest,
  ReplayPolicyRequest,
  PolicyReplayEvaluation,
  PolicyReplayResponse,

  // WHIP
  AttachStreamRequest,
  WhipSession,
  WhipPromptUpdateRequest,
  WhipPromptUpdateResponse,

  // Search
  SearchHit,
  SearchResult,
  SearchResponse,
} from "./types.js";
