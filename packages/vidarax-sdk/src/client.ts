/**
 * Vidarax SDK — main client class.
 *
 * ```typescript
 * import { Vidarax } from 'vidarax'
 * const v = new Vidarax('http://localhost:8080')
 * const run = await v.analyze('video.mp4', { prompt: 'Describe what happens' })
 * for await (const event of run.events()) {
 *   console.log(event.kind, event.payload)
 * }
 * ```
 */

import {
  HttpError,
  NetworkError,
  ParseError,
  RetryExhaustedError,
  UploadError,
  type ApiErrorBody,
} from "./errors.js";

import type {
  AgentEvent,
  AnalyzeFramesRequest,
  AnalyzeFramesResponse,
  AnalyzeOptions,
  AnalysisResult,
  AttachStreamRequest,
  CreateRunRequest,
  CreateRunResponse,
  EventsResponse,
  FeedbackItem,
  FeedbackListResponse,
  FeedbackRequest,
  HealthStatus,
  InferBatchOptions,
  InferBatchRequest,
  InferBatchResponse,
  InferOptions,
  InferRequest,
  InferResponse,
  IngestRequest,
  IngestResponse,
  Marker,
  MarkerQuery,
  MarkersResponse,
  Model,
  ModelCatalogResponse,
  ProgressCallback,
  QueryRequest,
  QueryResponse,
  RealtimeReasonRequest,
  RealtimeReasonResponse,
  Run,
  SearchResult,
  UploadResponse,
  VidaraxOptions,
  WhipSession,
} from "./types.js";

// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Sleep for `ms` milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Clamp a number between min and max. */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Returns true for HTTP status codes that should trigger an automatic retry.
 *
 * 429 (rate limited) and 5xx server errors are retryable; 4xx client errors
 * are not (except 429).
 */
function isRetryable(status: number): boolean {
  return status === 429 || status >= 500;
}

// ─── Vidarax class ────────────────────────────────────────────────────────────

/**
 * Primary SDK entry point.
 *
 * Instantiate once per application with the base URL of your Vidarax
 * deployment.  All methods are async and throw typed `VidaraxError` subclasses
 * on failure.
 */
export class Vidarax {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly maxRetries: number;
  private readonly retryBaseDelayMs: number;
  private readonly timeoutMs: number;

  /**
   * @param baseUrl  Base URL of the Vidarax API, e.g. `"http://10.0.0.1:8080"`.
   *                 A trailing slash is stripped automatically.
   * @param options  Optional SDK-wide configuration.
   */
  constructor(baseUrl: string, options: VidaraxOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.maxRetries = options.maxRetries ?? 3;
    this.retryBaseDelayMs = options.retryBaseDelayMs ?? 200;
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  // ─── Low-level request primitives ─────────────────────────────────────────

  /**
   * Build the common HTTP headers included in every request.
   */
  private headers(extra: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...extra,
    };
    if (this.apiKey !== undefined) {
      headers["x-api-key"] = this.apiKey;
    }
    return headers;
  }

  /**
   * Execute a single HTTP request with abort-based timeout, parse the JSON
   * response, and throw a typed error on non-2xx status.
   *
   * This is the innermost layer — no retry logic here.
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    extraHeaders: Record<string, string> = {},
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers: this.headers(extraHeaders),
        body: body !== undefined ? JSON.stringify(body) : null,
        signal: controller.signal,
      });
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.name === "AbortError"
            ? `Request to ${path} timed out after ${this.timeoutMs}ms`
            : `Network error on ${path}: ${err.message}`
          : `Network error on ${path}`;
      throw new NetworkError(message, err);
    } finally {
      clearTimeout(timerId);
    }

    if (!response.ok) {
      let apiError: ApiErrorBody | null = null;
      let text = "";
      try {
        text = await response.text();
        const parsed = JSON.parse(text) as { error?: ApiErrorBody };
        apiError = parsed.error ?? null;
      } catch {
        // not JSON; leave apiError null
      }
      const message =
        apiError?.message ??
        `HTTP ${response.status} ${response.statusText} on ${method} ${path}`;
      throw new HttpError(response.status, message, apiError);
    }

    // Handle 204 / empty bodies.
    const contentLength = response.headers.get("content-length");
    if (response.status === 204 || contentLength === "0") {
      return undefined as unknown as T;
    }

    let text = "";
    try {
      text = await response.text();
      if (text.trim() === "") {
        return undefined as unknown as T;
      }
      return JSON.parse(text) as T;
    } catch (err) {
      throw new ParseError(
        `Failed to parse response from ${method} ${path}: ${String(err)}`,
        text,
      );
    }
  }

  /**
   * Execute a request with automatic exponential back-off retry.
   *
   * Only network errors and retryable HTTP status codes trigger a retry.
   * Client errors (4xx except 429) are surfaced immediately.
   */
  private async requestWithRetry<T>(
    method: string,
    path: string,
    body?: unknown,
    extraHeaders: Record<string, string> = {},
  ): Promise<T> {
    let lastError!: NetworkError | HttpError;
    const maxAttempts = this.maxRetries + 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await this.request<T>(method, path, body, extraHeaders);
      } catch (err: unknown) {
        if (err instanceof HttpError && !isRetryable(err.status)) {
          // Hard client error — don't retry.
          throw err;
        }
        if (err instanceof HttpError || err instanceof NetworkError) {
          lastError = err;
        } else {
          // Unknown error type — rethrow immediately.
          throw err;
        }

        if (attempt < maxAttempts) {
          const jitter = Math.random() * this.retryBaseDelayMs;
          const delay = this.retryBaseDelayMs * 2 ** (attempt - 1) + jitter;
          await sleep(clamp(delay, 0, 30_000));
        }
      }
    }

    throw new RetryExhaustedError(maxAttempts, lastError);
  }

  // ─── Convenience wrappers ─────────────────────────────────────────────────

  private get<T>(path: string): Promise<T> {
    return this.requestWithRetry<T>("GET", path);
  }

  private post<T>(path: string, body?: unknown): Promise<T> {
    return this.requestWithRetry<T>("POST", path, body);
  }

  private delete<T>(path: string): Promise<T> {
    return this.requestWithRetry<T>("DELETE", path);
  }

  // ─── Runs ─────────────────────────────────────────────────────────────────

  /**
   * Create a new run resource.
   *
   * @example
   * const { run_id } = await v.createRun({ mode: 'analysis', model: 'llama3.2-vision' })
   */
  async createRun(options: CreateRunRequest = {}): Promise<CreateRunResponse> {
    return this.post<CreateRunResponse>("/v1/runs", options);
  }

  /**
   * List all runs owned by the current principal.
   *
   * Runs are ordered ascending by creation time.
   */
  async listRuns(): Promise<Run[]> {
    return this.get<Run[]>("/v1/runs");
  }

  /**
   * Retrieve a single run by ID.
   *
   * Throws `HttpError` with `status === 404` when the run does not exist.
   */
  async getRun(runId: string): Promise<Run> {
    return this.get<Run>(`/v1/runs/${encodeURIComponent(runId)}`);
  }

  /**
   * Permanently delete a run.
   *
   * The run is soft-deleted by appending a `run_deleted` timeline event and
   * will no longer appear in `listRuns()` results.
   */
  async deleteRun(runId: string): Promise<void> {
    await this.delete<void>(`/v1/runs/${encodeURIComponent(runId)}`);
  }

  // ─── Ingest ───────────────────────────────────────────────────────────────

  /**
   * Attach a source URI to an existing run and decode its frames server-side.
   *
   * The server probes the source, decodes the frames according to the sampling
   * policy, and stores the decoded frame signals as run events ready for a
   * subsequent `analyze()` call.
   */
  async ingestRun(runId: string, options: IngestRequest): Promise<IngestResponse> {
    return this.post<IngestResponse>(
      `/v1/runs/${encodeURIComponent(runId)}/ingest`,
      options,
    );
  }

  /**
   * Stop an active run.
   *
   * Transitions the run to the `cancelled` state.  Throws if the run is
   * already in a terminal state.
   */
  async stopRun(runId: string): Promise<void> {
    await this.post<void>(`/v1/runs/${encodeURIComponent(runId)}/stop`);
  }

  /**
   * Send a keepalive heartbeat for a streaming run.
   *
   * Call this periodically (e.g. every 30 s) to prevent the server from
   * marking a long-running ingest as stale.
   */
  async keepaliveRun(runId: string): Promise<void> {
    await this.post<void>(`/v1/runs/${encodeURIComponent(runId)}/keepalive`);
  }

  // ─── Events & markers ─────────────────────────────────────────────────────

  /**
   * Retrieve all timeline events for a run as a plain array.
   *
   * For a streaming/async-iterable version see `streamEvents()`.
   */
  async getEvents(runId: string): Promise<AgentEvent[]> {
    const res = await this.get<EventsResponse>(
      `/v1/runs/${encodeURIComponent(runId)}/events`,
    );
    return res.events;
  }

  /**
   * Retrieve all markers for a run with optional filtering.
   *
   * For a streaming version see `streamMarkers()`.
   */
  async getMarkers(runId: string, query: MarkerQuery = {}): Promise<Marker[]> {
    const params = new URLSearchParams();
    if (query.status !== undefined) params.set("status", query.status);
    if (query.event_type !== undefined) params.set("event_type", query.event_type);
    if (query.from_frame !== undefined) params.set("from_frame", String(query.from_frame));
    if (query.to_frame !== undefined) params.set("to_frame", String(query.to_frame));
    const qs = params.toString();
    const path = `/v1/runs/${encodeURIComponent(runId)}/markers${qs ? `?${qs}` : ""}`;
    const res = await this.get<MarkersResponse>(path);
    return res.markers;
  }

  /**
   * Get the current state of a run (without full event history).
   */
  async getRunState(runId: string): Promise<string> {
    const res = await this.get<{ request_id: string; run_id: string; state: string }>(
      `/v1/runs/${encodeURIComponent(runId)}/state`,
    );
    return res.state;
  }

  // ─── High-level analyze ───────────────────────────────────────────────────

  /**
   * Analyze a video source in a single ergonomic call.
   *
   * This is the primary SDK entry point for video analysis.  It orchestrates:
   * 1. `createRun()`
   * 2. `ingestRun()` — decodes the source server-side
   * 3. `analyzeRun()` — runs the two-pass analysis pipeline
   *
   * Returns an `AnalysisResult` handle with `.events()` and `.markers()`
   * async generators for iterating over results.
   *
   * @param source  A file path / URI understood by the Vidarax server
   *                (e.g. `"file:///data/video.mp4"` or an HTTP URL), or a
   *                `File` object that is uploaded first via `uploadFile()`.
   * @param options Analysis configuration.
   *
   * @example
   * const result = await v.analyze('video.mp4', { prompt: 'Describe what happens' })
   * for await (const event of result.events()) {
   *   console.log(event.kind, event.payload)
   * }
   */
  async analyze(
    source: string | File,
    options: AnalyzeOptions = {},
  ): Promise<AnalysisResult> {
    // 1. Resolve source URI — upload the File if necessary.
    let sourceUri: string;
    if (typeof source === "string") {
      sourceUri = source;
    } else {
      const uploaded = await this.uploadFile(source);
      sourceUri = uploaded.file_path;
    }

    // 2. Create run.
    const createOpts: CreateRunRequest = {};
    if (options.mode !== undefined) createOpts.mode = options.mode;
    if (options.model !== undefined) createOpts.model = options.model;
    const { run_id } = await this.createRun(createOpts);

    // 3. Ingest — decode source into frame signals stored on the run.
    const ingestOpts: IngestRequest = { source_uri: sourceUri };
    if (options.samplingPolicy !== undefined) ingestOpts.sampling_policy = options.samplingPolicy;
    if (options.fixedFps !== undefined) ingestOpts.fixed_fps = options.fixedFps;
    if (options.maxFrames !== undefined) ingestOpts.max_frames = options.maxFrames;
    await this.ingestRun(run_id, ingestOpts);

    // 4. Analyze — run the pipeline over the decoded frames.
    const analyzeOpts: AnalyzeFramesRequest = {
      model: options.model ?? "llama3.2-vision:11b",
    };
    if (options.mode !== undefined) analyzeOpts.mode = options.mode;
    if (options.windowSize !== undefined) analyzeOpts.window_size = options.windowSize;
    if (options.segmentMs !== undefined) analyzeOpts.segment_ms = options.segmentMs;
    const analyzeResponse = await this.analyzeRun(run_id, analyzeOpts);

    // Build and return the AnalysisResult handle.
    const self = this;
    return {
      runId: run_id,
      analyzeResponse,
      async *events(): AsyncGenerator<AgentEvent> {
        yield* self.streamEvents(run_id);
      },
      async *markers(): AsyncGenerator<Marker> {
        yield* self.streamMarkers(run_id);
      },
    };
  }

  /**
   * Execute the analysis pipeline for a run that has already been ingested.
   *
   * Lower-level than `analyze()` — use when you need fine-grained control
   * over frame signals or want to run analysis incrementally.
   */
  async analyzeRun(
    runId: string,
    options: AnalyzeFramesRequest,
  ): Promise<AnalyzeFramesResponse> {
    return this.post<AnalyzeFramesResponse>(
      `/v1/runs/${encodeURIComponent(runId)}/analyze`,
      options,
    );
  }

  /**
   * Run the realtime reason pipeline over a live or file-based source URI.
   *
   * This endpoint decodes the source, runs the two-pass pipeline with optional
   * semantic inference, and returns the full annotated result in one shot.
   */
  async reason(runId: string, options: RealtimeReasonRequest): Promise<RealtimeReasonResponse> {
    return this.post<RealtimeReasonResponse>(
      `/v1/runs/${encodeURIComponent(runId)}/reason`,
      options,
    );
  }

  // ─── Inference ────────────────────────────────────────────────────────────

  /**
   * Run a single prompt through the configured inference provider.
   *
   * @param prompt   The prompt text.
   * @param options  Optional inference parameters (model, temperature, etc.).
   */
  async infer(
    prompt: string,
    options: InferOptions & { model?: string } = {},
  ): Promise<InferResponse> {
    const body: InferRequest = {
      model: "llama3.2-vision:11b",
      ...options,
      prompt,
    };
    return this.post<InferResponse>("/v1/infer", body);
  }

  /**
   * Run multiple prompts through the inference provider in parallel.
   *
   * Results are returned in the same order as the input array regardless of
   * execution order.  Items that fail are represented with `ok: false` rather
   * than causing the whole batch to throw.
   *
   * @param requests  Array of individual inference request objects (each must
   *                  include `model` and `prompt`).
   * @param options   Batch-level options such as `max_parallel`.
   */
  async inferBatch(
    requests: InferRequest[],
    options: InferBatchOptions = {},
  ): Promise<InferBatchResponse> {
    const body: InferBatchRequest = {
      requests,
      ...options,
    };
    return this.post<InferBatchResponse>("/v1/infer/batch", body);
  }

  // ─── Streaming iterables ──────────────────────────────────────────────────

  /**
   * Yield all timeline events for a run.
   *
   * The current implementation fetches the full event list once and yields
   * them in sequence order.  A future version will support long-poll /
   * server-sent events for live streaming.
   *
   * @example
   * for await (const event of v.streamEvents(runId)) {
   *   console.log(event.kind)
   * }
   */
  async *streamEvents(runId: string): AsyncGenerator<AgentEvent> {
    const events = await this.getEvents(runId);
    for (const event of events) {
      yield event;
    }
  }

  /**
   * Yield all markers for a run.
   *
   * @example
   * for await (const marker of v.streamMarkers(runId)) {
   *   console.log(marker.event_type, marker.confidence)
   * }
   */
  async *streamMarkers(runId: string, query: MarkerQuery = {}): AsyncGenerator<Marker> {
    const markers = await this.getMarkers(runId, query);
    for (const marker of markers) {
      yield marker;
    }
  }

  // ─── Models ───────────────────────────────────────────────────────────────

  /**
   * Retrieve the full model catalog with availability and fallback information.
   */
  async listModels(): Promise<Model[]> {
    const res = await this.get<ModelCatalogResponse>("/v1/models");
    return res.models;
  }

  // ─── Health ───────────────────────────────────────────────────────────────

  /**
   * Check the health of the Vidarax server.
   *
   * Returns immediately with `{ status: "ok" }` when the server is reachable.
   * Throws a `NetworkError` when the server is unreachable.
   */
  async health(): Promise<HealthStatus> {
    return this.get<HealthStatus>("/v1/health");
  }

  // ─── File upload ──────────────────────────────────────────────────────────

  /**
   * Upload a file to the server and return the server-side file path.
   *
   * The returned `file_path` can be used as the `source_uri` in subsequent
   * `ingestRun()` or `analyze()` calls.
   *
   * @param file      The file to upload (web `File` object or Node.js `Blob`).
   * @param onProgress  Optional callback receiving `(loaded, total)` byte
   *                    counts.  Note: accurate `total` requires the `File` to
   *                    expose a non-zero `size`.
   *
   * @example
   * const file = new File([buffer], 'clip.mp4', { type: 'video/mp4' })
   * const { file_path } = await v.uploadFile(file, (loaded, total) => {
   *   console.log(`${Math.round(loaded / total * 100)}%`)
   * })
   */
  async uploadFile(file: File | Blob, onProgress?: ProgressCallback): Promise<UploadResponse> {
    const url = `${this.baseUrl}/v1/upload`;
    const formData = new FormData();
    formData.append("file", file);

    // XMLHttpRequest gives us upload progress; fetch does not.
    if (onProgress !== undefined && typeof XMLHttpRequest !== "undefined") {
      return new Promise<UploadResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url);
        if (this.apiKey !== undefined) {
          xhr.setRequestHeader("x-api-key", this.apiKey);
        }
        xhr.setRequestHeader("Accept", "application/json");

        xhr.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            onProgress(event.loaded, event.total);
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText) as UploadResponse);
            } catch {
              reject(new ParseError("Failed to parse upload response", xhr.responseText));
            }
          } else {
            let apiError: ApiErrorBody | null = null;
            try {
              const parsed = JSON.parse(xhr.responseText) as { error?: ApiErrorBody };
              apiError = parsed.error ?? null;
            } catch {
              // ignore
            }
            reject(
              new UploadError(
                apiError?.message ?? `Upload failed with HTTP ${xhr.status}`,
              ),
            );
          }
        });

        xhr.addEventListener("error", () => {
          reject(new NetworkError("Upload network error"));
        });
        xhr.addEventListener("abort", () => {
          reject(new NetworkError("Upload aborted"));
        });

        xhr.send(formData);
      });
    }

    // Fallback: use fetch without progress reporting.
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), this.timeoutMs);
    let response: Response;
    try {
      const fetchHeaders: Record<string, string> = { Accept: "application/json" };
      if (this.apiKey !== undefined) {
        fetchHeaders["x-api-key"] = this.apiKey;
      }
      response = await fetch(url, {
        method: "POST",
        headers: fetchHeaders,
        body: formData,
        signal: controller.signal,
      });
    } catch (err: unknown) {
      throw new NetworkError(
        err instanceof Error && err.name === "AbortError"
          ? `Upload timed out after ${this.timeoutMs}ms`
          : `Upload network error: ${String(err)}`,
        err,
      );
    } finally {
      clearTimeout(timerId);
    }

    if (!response.ok) {
      let apiError: ApiErrorBody | null = null;
      try {
        const parsed = (await response.json()) as { error?: ApiErrorBody };
        apiError = parsed.error ?? null;
      } catch {
        // ignore
      }
      throw new UploadError(apiError?.message ?? `Upload failed with HTTP ${response.status}`);
    }

    try {
      return (await response.json()) as UploadResponse;
    } catch (err) {
      throw new ParseError("Failed to parse upload response", String(err));
    }
  }

  // ─── Query ────────────────────────────────────────────────────────────────

  /**
   * Query timeline events for a run with optional kind and sequence filters.
   *
   * Useful for retrieving a specific event kind (e.g. `"analysis_generated"`)
   * or paginating from a known sequence number.
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.post<QueryResponse>("/v1/query", request);
  }

  // ─── Feedback ─────────────────────────────────────────────────────────────

  /**
   * Submit quality feedback for a completed run.
   *
   * Ratings are on a 0–10 scale.  Requires a SpacetimeDB connection on the
   * server side; throws `HttpError` with a 500 status if not configured.
   */
  async submitFeedback(runId: string, feedback: FeedbackRequest): Promise<void> {
    await this.post<void>(
      `/v1/runs/${encodeURIComponent(runId)}/feedback`,
      feedback,
    );
  }

  /**
   * Retrieve all stored feedback entries.
   *
   * Requires a SpacetimeDB connection on the server side.
   */
  async listFeedback(): Promise<FeedbackItem[]> {
    const res = await this.get<FeedbackListResponse>("/v1/feedback");
    return res.feedback;
  }

  // ─── Search (future) ──────────────────────────────────────────────────────

  /**
   * Semantic search over processed runs.
   *
   * This method is a forward-looking stub; the `/v1/search` endpoint is not
   * yet part of the server.  It will throw an `HttpError` (404) until the
   * endpoint is deployed.
   *
   * @param query  Natural-language search query.
   */
  async search(query: string): Promise<SearchResult[]> {
    return this.post<SearchResult[]>("/v1/search", { query });
  }

  // ─── WHIP WebRTC (browser-only) ───────────────────────────────────────────

  /**
   * Exchange a WebRTC SDP offer for an answer and begin a WHIP session.
   *
   * This is a browser-only helper.  Pass the SDP string produced by
   * `RTCPeerConnection.createOffer()` and receive back the server SDP answer
   * along with the session resource URL for trickle-ICE and teardown.
   *
   * The optional `attachConfig` is appended as a JSON body field alongside the
   * SDP — non-standard but supported by the Vidarax WHIP handler for setting
   * the initial prompt and clip mode.
   *
   * Per RFC 9725 the offer body is sent as `Content-Type: application/sdp`.
   *
   * @example
   * const pc = new RTCPeerConnection({ iceServers: [] })
   * const offer = await pc.createOffer()
   * await pc.setLocalDescription(offer)
   * const session = await v.whipOffer(offer.sdp!, { prompt: 'Watch for motion' })
   * await pc.setRemoteDescription({ type: 'answer', sdp: session.answerSdp })
   */
  async whipOffer(
    sdpOffer: string,
    attachConfig?: AttachStreamRequest,
  ): Promise<WhipSession> {
    const url = `${this.baseUrl}/v1/stream/whip`;
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), this.timeoutMs);

    const headers: Record<string, string> = {
      "Content-Type": "application/sdp",
      Accept: "application/sdp",
    };
    if (this.apiKey !== undefined) {
      headers["x-api-key"] = this.apiKey;
    }
    // Encode optional attach config as a custom header (non-standard extension).
    if (attachConfig !== undefined) {
      headers["x-attach-config"] = JSON.stringify(attachConfig);
    }

    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers,
        body: sdpOffer,
        signal: controller.signal,
      });
    } catch (err: unknown) {
      throw new NetworkError(
        err instanceof Error && err.name === "AbortError"
          ? `WHIP offer timed out after ${this.timeoutMs}ms`
          : `WHIP offer network error: ${String(err)}`,
        err,
      );
    } finally {
      clearTimeout(timerId);
    }

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new HttpError(
        response.status,
        `WHIP offer failed with HTTP ${response.status}: ${text}`,
      );
    }

    const answerSdp = await response.text();
    const location = response.headers.get("Location") ?? "";
    const resourceUrl = location.startsWith("http")
      ? location
      : `${this.baseUrl}${location}`;

    // Extract session ID from the Location path.
    const sessionId = location.split("/").pop() ?? "";

    return { sessionId, answerSdp, resourceUrl };
  }

  /**
   * Send a trickle-ICE candidate for an active WHIP session.
   *
   * @param sessionId  The session ID returned from `whipOffer()`.
   * @param candidate  The ICE candidate string (SDP fragment).
   */
  async whipIce(sessionId: string, candidate: string): Promise<void> {
    const url = `${this.baseUrl}/v1/stream/whip/${encodeURIComponent(sessionId)}`;
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/trickle-ice-sdpfrag",
      };
      if (this.apiKey !== undefined) {
        headers["x-api-key"] = this.apiKey;
      }
      const response = await fetch(url, {
        method: "PATCH",
        headers,
        body: candidate,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new HttpError(
          response.status,
          `WHIP ICE candidate failed with HTTP ${response.status}`,
        );
      }
    } catch (err: unknown) {
      if (err instanceof HttpError) throw err;
      throw new NetworkError(`WHIP ICE network error: ${String(err)}`, err);
    } finally {
      clearTimeout(timerId);
    }
  }

  /**
   * Update the VLM prompt for a live WHIP session without renegotiation.
   *
   * @param sessionId  The session ID returned from `whipOffer()`.
   * @param config     New prompt and optional clip configuration.
   */
  async whipUpdatePrompt(
    sessionId: string,
    config: AttachStreamRequest,
  ): Promise<void> {
    const url = `${this.baseUrl}/v1/stream/whip/${encodeURIComponent(sessionId)}/prompt`;
    const controller = new AbortController();
    const timerId = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        Accept: "application/json",
      };
      if (this.apiKey !== undefined) {
        headers["x-api-key"] = this.apiKey;
      }
      const response = await fetch(url, {
        method: "PATCH",
        headers,
        body: JSON.stringify(config),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new HttpError(
          response.status,
          `WHIP prompt update failed with HTTP ${response.status}`,
        );
      }
    } catch (err: unknown) {
      if (err instanceof HttpError) throw err;
      throw new NetworkError(`WHIP prompt update network error: ${String(err)}`, err);
    } finally {
      clearTimeout(timerId);
    }
  }

  /**
   * Terminate an active WHIP session and release server-side resources.
   *
   * @param sessionId  The session ID returned from `whipOffer()`.
   */
  async whipTerminate(sessionId: string): Promise<void> {
    await this.delete<void>(
      `/v1/stream/whip/${encodeURIComponent(sessionId)}`,
    );
  }

  // ─── Connection health monitoring ─────────────────────────────────────────

  /**
   * Probe the server repeatedly until it responds or the timeout elapses.
   *
   * Useful for startup health checks or waiting for the server to become
   * available after a deploy.
   *
   * @param pollIntervalMs  How often to poll in ms (default: 1 000).
   * @param totalTimeoutMs  Total wait budget in ms (default: 60 000).
   * @returns               `true` when the server responded `{ status: "ok" }`,
   *                        `false` when the total timeout was exceeded.
   */
  async waitUntilHealthy(
    pollIntervalMs = 1_000,
    totalTimeoutMs = 60_000,
  ): Promise<boolean> {
    const deadline = Date.now() + totalTimeoutMs;
    while (Date.now() < deadline) {
      try {
        const h = await this.health();
        if (h.status === "ok") return true;
      } catch {
        // Server not yet reachable — keep polling.
      }
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await sleep(Math.min(pollIntervalMs, remaining));
    }
    return false;
  }
}
