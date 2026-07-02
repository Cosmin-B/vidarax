/**
 * End-to-end tests for the Vidarax SDK.
 *
 * These tests run against a live Vidarax API server. Set the
 * VIDARAX_API_URL environment variable to override the default endpoint.
 *
 * The test video used for analysis tests must exist on the server at
 * /tmp/vidarax-e2e-test.mp4 and is referenced via file:/// URI.
 *
 * Note: the server enforces a maximum of 20 concurrent active runs per
 * principal. This file uses a single `beforeAll` / `afterAll` at the suite
 * root so that shared runs are created sequentially and cleaned up promptly.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Vidarax } from "../src/client.js";
import {
  HttpError,
  NetworkError,
  isHttpError,
  isNetworkError,
} from "../src/errors.js";
import type {
  Run,
  Model,
  CreateRunResponse,
  InferResult,
  InferBatchResponse,
  AgentEvent,
  Marker,
  IngestResponse,
  AnalyzeFramesResponse,
} from "../src/types.js";

declare const process: { env: Record<string, string | undefined> };

// ─── Configuration ────────────────────────────────────────────────────────────

const API_URL =
  process.env["VIDARAX_API_URL"] ?? "http://localhost:8080";

/** The small model available on the test server. */
const SMALL_MODEL = "Qwen/Qwen3-VL-2B-Instruct";

/** Path on the server filesystem used for analysis tests. */
const TEST_VIDEO_URI = "file:///tmp/vidarax-e2e-test.mp4";

/** Timeout for the full E2E suite setup (analysis pipeline). */
const SUITE_SETUP_TIMEOUT = 180_000;

/** Timeout for individual inference tests. */
const INFERENCE_TIMEOUT = 120_000;

/** Timeout for the analysis pipeline section. */
const ANALYSIS_TIMEOUT = 180_000;

// ─── Test client ──────────────────────────────────────────────────────────────

/**
 * Each test run uses a unique API key so it gets a fresh per-principal run
 * budget on the server (the server limits each principal to 20 total run
 * creations per session — soft-deleted runs still count against the limit).
 *
 * The key can be overridden via VIDARAX_API_KEY in the environment (useful
 * when pointing at a deployment that enforces real keys).
 */
const SESSION_API_KEY =
  process.env["VIDARAX_API_KEY"] ??
  `test-e2e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const client = new Vidarax(API_URL, {
  apiKey: SESSION_API_KEY,
  maxRetries: 1,
  retryBaseDelayMs: 100,
  timeoutMs: ANALYSIS_TIMEOUT,
});

// ─── Suite-level shared state ─────────────────────────────────────────────────
//
// All long-running setup (run creation, ingest, analyze) is done once here
// so that individual tests simply assert against the cached results.
//
// Runs are tracked for afterAll cleanup.

interface SuiteState {
  /** Run used for lifecycle tests. */
  lifecycleRun: CreateRunResponse;

  /** Inference result cached from beforeAll. */
  inferResult: InferResult;

  /** Batch inference result cached from beforeAll. */
  inferBatchResult: InferBatchResponse;

  /** Run that has been ingested + analyzed. */
  analysisRunId: string;
  analysisIngestResponse: IngestResponse;
  analysisAnalyzeResponse: AnalyzeFramesResponse;

  /** Run returned by the high-level analyze() helper. */
  highLevelAnalysisRunId: string;
  highLevelAnalysisResponse: AnalyzeFramesResponse;
}

// Filled in by beforeAll. Individual tests reference this.
let state: SuiteState;

/** IDs of any runs created inside test bodies (for afterAll cleanup). */
const testBodyRunIds: string[] = [];

/**
 * Small helper to track a run created inside a test body so it is
 * deleted in afterAll even if the test fails.
 */
function trackRun(runId: string): string {
  testBodyRunIds.push(runId);
  return runId;
}

// ─── Suite-level setup ────────────────────────────────────────────────────────

beforeAll(async () => {
  // 1. Create the lifecycle run used by section 4 tests.
  const lifecycleRun = await client.createRun({ mode: "balanced" });

  // 2. Start inference (does not need a run object).
  const inferResult = await client.infer({
    prompt: "Say hello",
    model: SMALL_MODEL,
  });

  // 3. Batch inference.
  const inferBatchResult = await client.inferBatch(
    [
      { model: SMALL_MODEL, prompt: "Reply with just: one" },
      { model: SMALL_MODEL, prompt: "Reply with just: two" },
    ],
    { max_parallel: 2 },
  );

  // 4. Create + ingest + analyze a run for the analysis-pipeline tests.
  const analysisRun = await client.createRun({
    mode: "balanced",
    model: SMALL_MODEL,
  });
  const analysisIngestResponse = await client.ingestRun(analysisRun.run_id, {
    source_uri: TEST_VIDEO_URI,
    max_frames: 5,
  });
  const analysisAnalyzeResponse = await client.analyzeRun(analysisRun.run_id, {
    model: SMALL_MODEL,
    mode: "balanced",
  });

  // 5. High-level analyze() for section 7.
  const hlResult = await client.analyze(TEST_VIDEO_URI, {
    model: SMALL_MODEL,
    mode: "balanced",
    maxFrames: 3,
  });

  state = {
    lifecycleRun,
    inferResult,
    inferBatchResult,
    analysisRunId: analysisRun.run_id,
    analysisIngestResponse,
    analysisAnalyzeResponse,
    highLevelAnalysisRunId: hlResult.runId,
    highLevelAnalysisResponse: hlResult.analyzeResponse,
  };
}, SUITE_SETUP_TIMEOUT);

afterAll(async () => {
  // Clean up all runs created by the suite setup and test bodies.
  const ids = [
    state?.lifecycleRun?.run_id,
    state?.analysisRunId,
    state?.highLevelAnalysisRunId,
    ...testBodyRunIds,
  ].filter((id): id is string => typeof id === "string");

  await Promise.allSettled(ids.map((id) => client.deleteRun(id)));
});

// ─── 1. Client initialisation ─────────────────────────────────────────────────

describe("Client initialisation", () => {
  it("constructs with a base URL and creates a Vidarax instance", () => {
    const c = new Vidarax(API_URL);
    expect(c).toBeInstanceOf(Vidarax);
  });

  it("strips a trailing slash from the base URL without breaking requests", () => {
    const c = new Vidarax(`${API_URL}/`);
    expect(c).toBeInstanceOf(Vidarax);
  });

  it("accepts an apiKey option", () => {
    const c = new Vidarax(API_URL, { apiKey: "test-key-e2e" });
    expect(c).toBeInstanceOf(Vidarax);
  });

  it("accepts custom retry and timeout settings", () => {
    const c = new Vidarax(API_URL, {
      maxRetries: 0,
      retryBaseDelayMs: 50,
      timeoutMs: 5_000,
    });
    expect(c).toBeInstanceOf(Vidarax);
  });
});

// ─── 2. Health ────────────────────────────────────────────────────────────────

describe("health()", () => {
  it("returns { status: 'ok' } from the live server", async () => {
    const result = await client.health();
    expect(result).toMatchObject({ status: "ok" });
  });

  it("returns an object with a string status property", async () => {
    const result = await client.health();
    expect(typeof result.status).toBe("string");
  });
});

// ─── 3. Models ────────────────────────────────────────────────────────────────

describe("listModels()", () => {
  let models: Model[];

  beforeAll(async () => {
    models = await client.listModels();
  });

  it("returns a non-empty array", () => {
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
  });

  it("each model has a non-empty id string", () => {
    for (const m of models) {
      expect(typeof m.id).toBe("string");
      expect(m.id.length).toBeGreaterThan(0);
    }
  });

  it("each model has a valid tier ('small' | 'medium')", () => {
    for (const m of models) {
      expect(["small", "medium"]).toContain(m.tier);
    }
  });

  it("each model has an availability string", () => {
    for (const m of models) {
      expect(typeof m.availability).toBe("string");
    }
  });

  it("each model has providers_available as an array", () => {
    for (const m of models) {
      expect(Array.isArray(m.providers_available)).toBe(true);
    }
  });

  it("each model has fallback_candidates as an array", () => {
    for (const m of models) {
      expect(Array.isArray(m.fallback_candidates)).toBe(true);
    }
  });

  it(`includes the expected small model '${SMALL_MODEL}'`, () => {
    expect(models.map((m) => m.id)).toContain(SMALL_MODEL);
  });
});

// ─── 4. Run lifecycle ─────────────────────────────────────────────────────────

describe("Run lifecycle", () => {
  it("createRun() returns a run_id matching the run-<32hex> pattern", () => {
    expect(state.lifecycleRun.run_id).toMatch(/^run-[0-9a-f]{32}$/);
  });

  it("createRun() response includes a request_id string", () => {
    expect(typeof state.lifecycleRun.request_id).toBe("string");
    expect(state.lifecycleRun.request_id.length).toBeGreaterThan(0);
  });

  it("createRun() response includes a status string", () => {
    expect(typeof state.lifecycleRun.status).toBe("string");
  });

  it("createRun({ mode: 'balanced' }) stores the mode", () => {
    expect(state.lifecycleRun.mode).toBe("balanced");
  });

  it("getRun() returns the run matching the created ID", async () => {
    const run: Run = await client.getRun(state.lifecycleRun.run_id);
    expect(run.run_id).toBe(state.lifecycleRun.run_id);
    expect(typeof run.status).toBe("string");
  });

  it("getRun() result has all required Run fields", async () => {
    const run: Run = await client.getRun(state.lifecycleRun.run_id);
    expect("run_id" in run).toBe(true);
    expect("status" in run).toBe(true);
    expect("mode" in run).toBe(true);
    expect("model" in run).toBe(true);
    expect("source_uri" in run).toBe(true);
    expect("created_at" in run).toBe(true);
    expect("updated_at" in run).toBe(true);
  });

  it("listRuns() returns an array containing the created run", async () => {
    const runs = await client.listRuns();
    expect(Array.isArray(runs)).toBe(true);
    const ids = runs.map((r) => r.run_id);
    expect(ids).toContain(state.lifecycleRun.run_id);
  });

  it("listRuns() returns Run objects with required fields", async () => {
    const runs = await client.listRuns();
    expect(runs.length).toBeGreaterThan(0);
    const first = runs[0]!;
    expect("run_id" in first).toBe(true);
    expect("status" in first).toBe(true);
    expect("mode" in first).toBe(true);
  });

  it("deleteRun() resolves without throwing", async () => {
    const tempRun = await client.createRun({ mode: "balanced" });
    // Delete immediately — don't track in testBodyRunIds.
    await expect(client.deleteRun(tempRun.run_id)).resolves.toBeUndefined();
  });

  it("getRun() throws HttpError(404) after a run is deleted", async () => {
    const tempRun = await client.createRun({ mode: "balanced" });
    await client.deleteRun(tempRun.run_id);

    const error = await client
      .getRun(tempRun.run_id)
      .catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    const httpErr = error as HttpError;
    expect(httpErr.status).toBe(404);
    expect(httpErr.isNotFound).toBe(true);
  });
});

// ─── 5. Events and markers (pre-analysis) ─────────────────────────────────────

describe("getEvents() on a fresh run", () => {
  it("returns an array (events are created even before analysis)", async () => {
    const run = await client.createRun({ mode: "balanced" });
    trackRun(run.run_id);
    const events: AgentEvent[] = await client.getEvents(run.run_id);
    expect(Array.isArray(events)).toBe(true);
    // A fresh run already has at least a run_created event.
    expect(events.some((e) => e.kind === "run_created")).toBe(true);
    await client.deleteRun(run.run_id).catch(() => undefined);
    // Remove from tracking since we deleted inline.
    const idx = testBodyRunIds.indexOf(run.run_id);
    if (idx !== -1) testBodyRunIds.splice(idx, 1);
  });
});

describe("getMarkers() on a fresh run", () => {
  it("returns an empty array for a run with no analysis", async () => {
    const run = await client.createRun({ mode: "balanced" });
    trackRun(run.run_id);
    const markers: Marker[] = await client.getMarkers(run.run_id);
    expect(Array.isArray(markers)).toBe(true);
    expect(markers.length).toBe(0);
    await client.deleteRun(run.run_id).catch(() => undefined);
    const idx = testBodyRunIds.indexOf(run.run_id);
    if (idx !== -1) testBodyRunIds.splice(idx, 1);
  });
});

// ─── 6. Analysis pipeline ─────────────────────────────────────────────────────
//
// The run was created + ingested + analyzed in the top-level beforeAll.

describe("Analysis pipeline (ingest + analyzeRun)", () => {
  it("ingestRun() decoded a positive number of frames", () => {
    const res = state.analysisIngestResponse;
    expect(typeof res.request_id).toBe("string");
    expect(res.run_id).toBe(state.analysisRunId);
    expect(typeof res.status).toBe("string");
    expect(typeof res.decoded_frames).toBe("number");
    expect(res.decoded_frames!).toBeGreaterThan(0);
    expect(res.decoded_frames!).toBeLessThanOrEqual(5);
    expect(res.source_uri).toBe(TEST_VIDEO_URI);
  });

  it("analyzeRun() response has generated count and markers array", () => {
    const res = state.analysisAnalyzeResponse;
    expect(typeof res.request_id).toBe("string");
    expect(res.run_id).toBe(state.analysisRunId);
    expect(typeof res.generated).toBe("number");
    expect(res.generated).toBeGreaterThan(0);
    expect(Array.isArray(res.markers)).toBe(true);
    expect(Array.isArray(res.metadata)).toBe(true);
  });

  it("getEvents() returns events after analysis", async () => {
    const events = await client.getEvents(state.analysisRunId);
    expect(events.length).toBeGreaterThan(0);
  });

  it("each event has seq, pts_ms, kind, and payload fields", async () => {
    const events = await client.getEvents(state.analysisRunId);
    for (const ev of events) {
      expect(typeof ev.seq).toBe("number");
      expect(typeof ev.pts_ms).toBe("number");
      expect(typeof ev.kind).toBe("string");
      expect(typeof ev.payload).toBe("object");
    }
  });

  it("events include a 'run_created' kind", async () => {
    const events = await client.getEvents(state.analysisRunId);
    expect(events.map((e) => e.kind)).toContain("run_created");
  });

  it("getMarkers() returns one or more markers after analysis", async () => {
    const markers = await client.getMarkers(state.analysisRunId);
    expect(markers.length).toBeGreaterThan(0);
  });

  it("each marker has required fields with correct types", async () => {
    const markers = await client.getMarkers(state.analysisRunId);
    for (const m of markers) {
      expect(typeof m.marker_id).toBe("string");
      expect(typeof m.run_id).toBe("string");
      expect(typeof m.stream_id).toBe("string");
      expect(typeof m.event_type).toBe("string");
      expect(typeof m.status).toBe("string");
      expect(typeof m.start_frame).toBe("number");
      expect(typeof m.end_frame).toBe("number");
      expect(typeof m.confidence).toBe("number");
      expect(m.run_id).toBe(state.analysisRunId);
    }
  });

  it("marker confidence values are in the [0, 1] range", async () => {
    const markers = await client.getMarkers(state.analysisRunId);
    for (const m of markers) {
      expect(m.confidence).toBeGreaterThanOrEqual(0);
      expect(m.confidence).toBeLessThanOrEqual(1);
    }
  });

  it("streamEvents() yields the same count as getEvents()", async () => {
    const fromGet = await client.getEvents(state.analysisRunId);
    const fromStream: AgentEvent[] = [];
    for await (const ev of client.streamEvents(state.analysisRunId)) {
      fromStream.push(ev);
    }
    expect(fromStream.length).toBe(fromGet.length);
  });

  it("streamMarkers() yields the same count as getMarkers()", async () => {
    const fromGet = await client.getMarkers(state.analysisRunId);
    const fromStream: Marker[] = [];
    for await (const m of client.streamMarkers(state.analysisRunId)) {
      fromStream.push(m);
    }
    expect(fromStream.length).toBe(fromGet.length);
  });

  it("getMarkers() with event_type filter returns only matching markers", async () => {
    const allMarkers = await client.getMarkers(state.analysisRunId);
    if (allMarkers.length === 0) return;

    const targetType = allMarkers[0]!.event_type;
    const filtered = await client.getMarkers(state.analysisRunId, {
      event_type: targetType,
    });

    for (const m of filtered) {
      expect(m.event_type).toBe(targetType);
    }
  });

  it("getRunState() returns a non-empty string after analysis", async () => {
    const state2 = await client.getRunState(state.analysisRunId);
    expect(typeof state2).toBe("string");
    expect(state2.length).toBeGreaterThan(0);
  });
});

// ─── 7. High-level analyze() ──────────────────────────────────────────────────

describe("analyze() high-level helper", () => {
  it("returns a runId matching the run-<hex> format", () => {
    expect(state.highLevelAnalysisRunId).toMatch(/^run-/);
  });

  it("analyzeResponse has request_id, generated, markers[], metadata[]", () => {
    const res = state.highLevelAnalysisResponse;
    expect(typeof res.request_id).toBe("string");
    expect(typeof res.generated).toBe("number");
    expect(Array.isArray(res.markers)).toBe(true);
    expect(Array.isArray(res.metadata)).toBe(true);
  });

  it("events() async generator yields at least one event", async () => {
    const events: AgentEvent[] = [];
    for await (const ev of client.streamEvents(state.highLevelAnalysisRunId)) {
      events.push(ev);
    }
    expect(events.length).toBeGreaterThan(0);
  });

  it("markers() async generator is iterable", async () => {
    const markers: Marker[] = [];
    for await (const m of client.streamMarkers(state.highLevelAnalysisRunId)) {
      markers.push(m);
    }
    // Markers may or may not exist — the generator must be iterable.
    expect(Array.isArray(markers)).toBe(true);
  });
});

// ─── 8. Inference ─────────────────────────────────────────────────────────────

describe("infer()", () => {
  it("returns an InferResult with a non-empty result", () => {
    expect(typeof state.inferResult.result).toBe("string");
    expect(state.inferResult.result.length).toBeGreaterThan(0);
  });

  it("returns an id string", () => {
    expect(typeof state.inferResult.id).toBe("string");
    expect(state.inferResult.id.length).toBeGreaterThan(0);
  });

  it("returns the requested served model", () => {
    expect(state.inferResult.model_name).toBe(SMALL_MODEL);
  });

  it("returns a non-empty provider string", () => {
    expect(typeof state.inferResult.provider).toBe("string");
    expect(state.inferResult.provider.length).toBeGreaterThan(0);
  });

  it("returns a non-negative numeric inference_latency_ms", () => {
    expect(typeof state.inferResult.inference_latency_ms).toBe("number");
    expect(state.inferResult.inference_latency_ms).toBeGreaterThanOrEqual(0);
  });

  it("returns fallback_used as a boolean", () => {
    expect(typeof state.inferResult.fallback_used).toBe("boolean");
  });

  it("run_id is null when no run context is provided", () => {
    expect(state.inferResult.run_id).toBeNull();
  });
});

describe("inferBatch()", () => {
  it("returns processed count equal to number of requests (2)", () => {
    expect(state.inferBatchResult.processed).toBe(2);
  });

  it("returns a numeric succeeded count", () => {
    expect(typeof state.inferBatchResult.succeeded).toBe("number");
  });

  it("returns a numeric failed count", () => {
    expect(typeof state.inferBatchResult.failed).toBe("number");
  });

  it("returns a results array of the same length as input", () => {
    expect(Array.isArray(state.inferBatchResult.results)).toBe(true);
    expect(state.inferBatchResult.results.length).toBe(2);
  });

  it("each result item has numeric index and boolean ok", () => {
    for (const item of state.inferBatchResult.results) {
      expect(typeof item.index).toBe("number");
      expect(typeof item.ok).toBe("boolean");
    }
  });

  it("successful items have a non-null result with output_text", () => {
    for (const item of state.inferBatchResult.results) {
      if (item.ok) {
        expect(item.result).not.toBeNull();
        expect(typeof item.result!.output_text).toBe("string");
      }
    }
  });

  it("all two items succeeded (failed=0, succeeded=2)", () => {
    expect(state.inferBatchResult.failed).toBe(0);
    expect(state.inferBatchResult.succeeded).toBe(2);
  });
});

// ─── 9. Query ────────────────────────────────────────────────────────────────

describe("query()", () => {
  it("returns a QueryResponse with request_id and matches array", async () => {
    const res = await client.query({ run_id: state.analysisRunId });
    expect(Array.isArray(res.matches)).toBe(true);
    expect(typeof res.request_id).toBe("string");
  });

  it("filtering by kind=run_created returns only that kind", async () => {
    const res = await client.query({
      run_id: state.analysisRunId,
      kind: "run_created",
    });
    for (const event of res.matches) {
      expect(event.kind).toBe("run_created");
    }
  });

  it("returns at least one run_created event for any analyzed run", async () => {
    const res = await client.query({
      run_id: state.analysisRunId,
      kind: "run_created",
    });
    expect(res.matches.length).toBeGreaterThanOrEqual(1);
  });
});

// ─── 10. Search ──────────────────────────────────────────────────────────────

describe("search()", () => {
  it("returns the live search response shape", async () => {
    const res = await client.search("test video");
    expect(Array.isArray(res.hits)).toBe(true);
    expect(typeof res.request_id).toBe("string");
    expect(typeof res.scanned).toBe("number");
    expect(typeof res.total_hits).toBe("number");
  });
});

// ─── 11. Error handling ──────────────────────────────────────────────────────

describe("Error handling", () => {
  it(
    "throws a VidaraxError for an unreachable host",
    async () => {
      const badClient = new Vidarax("http://127.0.0.1:19999", {
        maxRetries: 0,
        retryBaseDelayMs: 0,
        timeoutMs: 2_000,
      });

      const error = await badClient.health().catch((e: unknown) => e);

      expect(
        isNetworkError(error) ||
          (error instanceof Error &&
            (error.constructor.name === "RetryExhaustedError" ||
              error.constructor.name === "NetworkError")),
      ).toBe(true);
    },
    10_000,
  );

  it("getRun() throws HttpError(404) for a valid-format but non-existent run ID", async () => {
    const fakeId = "run-00000000000000000000000000000000";
    const error = await client.getRun(fakeId).catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    const httpErr = error as HttpError;
    expect(httpErr.status).toBe(404);
    expect(httpErr.isNotFound).toBe(true);
    expect(httpErr.isClientError).toBe(true);
  });

  it("getRun() throws HttpError(422) for an invalid run_id format", async () => {
    const error = await client
      .getRun("not-a-valid-run-id")
      .catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    const httpErr = error as HttpError;
    expect(httpErr.status).toBe(422);
    expect(httpErr.isValidationError).toBe(true);
  });

  it("infer() throws HttpError(422) for an unsupported model", async () => {
    const error = await client
      .infer({ prompt: "hello", model: "totally-invalid-model-xyz" })
      .catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    const httpErr = error as HttpError;
    expect(httpErr.status).toBe(422);
    expect(httpErr.isValidationError).toBe(true);
  });

  it("deleteRun() throws HttpError(404) for a non-existent run", async () => {
    const fakeId = "run-ffffffffffffffffffffffffffffffff";
    const error = await client.deleteRun(fakeId).catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    expect((error as HttpError).status).toBe(404);
  });

  it("HttpError.isClientError is true for 4xx, false for 5xx", () => {
    const err = new HttpError(404, "not found");
    expect(err.isClientError).toBe(true);
    expect(err.isServerError).toBe(false);
    expect(err.isNotFound).toBe(true);
    expect(err.isValidationError).toBe(false);
    expect(err.isConflict).toBe(false);
  });

  it("HttpError.isServerError is true for 5xx, false for 4xx", () => {
    const err = new HttpError(500, "internal server error");
    expect(err.isServerError).toBe(true);
    expect(err.isClientError).toBe(false);
  });

  it("HttpError carries the structured apiError body when provided", () => {
    const apiError = {
      code: "not_found",
      message: "run was not found",
      request_id: "req-abc",
    };
    const err = new HttpError(404, "not found", apiError);
    expect(err.apiError).toEqual(apiError);
    expect(err.code).toBe("not_found");
  });

  it("isHttpError() identifies HttpError instances and rejects others", () => {
    expect(isHttpError(new HttpError(404, "not found"))).toBe(true);
    expect(isHttpError(new NetworkError("connection refused"))).toBe(false);
    expect(isHttpError(new Error("plain error"))).toBe(false);
    expect(isHttpError(null)).toBe(false);
    expect(isHttpError("string")).toBe(false);
  });

  it("isNetworkError() identifies NetworkError instances and rejects others", () => {
    expect(isNetworkError(new NetworkError("connection refused"))).toBe(true);
    expect(isNetworkError(new HttpError(500, "server error"))).toBe(false);
    expect(isNetworkError(new Error("plain error"))).toBe(false);
  });
});

// ─── 12. waitUntilHealthy ────────────────────────────────────────────────────

describe("waitUntilHealthy()", () => {
  it("returns true when the server is healthy", async () => {
    const healthy = await client.waitUntilHealthy(200, 5_000);
    expect(healthy).toBe(true);
  });

  it(
    "returns false when the server is unreachable within the timeout",
    async () => {
      const badClient = new Vidarax("http://127.0.0.1:19999", {
        maxRetries: 0,
        retryBaseDelayMs: 0,
        timeoutMs: 500,
      });
      const healthy = await badClient.waitUntilHealthy(100, 600);
      expect(healthy).toBe(false);
    },
    5_000,
  );
});

// ─── 13. Run state ────────────────────────────────────────────────────────────

describe("getRunState()", () => {
  it("returns a non-empty state string for a valid run", async () => {
    const stateStr = await client.getRunState(state.analysisRunId);
    expect(typeof stateStr).toBe("string");
    expect(stateStr.length).toBeGreaterThan(0);
  });

  it("throws HttpError(404) for a non-existent run", async () => {
    const error = await client
      .getRunState("run-00000000000000000000000000000000")
      .catch((e: unknown) => e);

    expect(isHttpError(error)).toBe(true);
    expect((error as HttpError).status).toBe(404);
  });
});

// ─── 14. Run stop / keepalive ────────────────────────────────────────────────

describe("stopRun() and keepaliveRun()", () => {
  it("stopRun() transitions a pending run to a terminal state or throws an HttpError", async () => {
    const run = await client.createRun({ mode: "balanced" });
    trackRun(run.run_id);

    try {
      await client.stopRun(run.run_id);
      const stateStr = await client.getRunState(run.run_id);
      expect(["cancelled", "completed", "failed"]).toContain(stateStr);
    } catch (e: unknown) {
      // Some server implementations reject stop for already-terminal runs.
      expect(isHttpError(e)).toBe(true);
    }
  });

  it("keepaliveRun() does not throw for an active run (or throws HttpError if already terminal)", async () => {
    const run = await client.createRun({ mode: "balanced" });
    trackRun(run.run_id);

    try {
      await client.keepaliveRun(run.run_id);
    } catch (e: unknown) {
      expect(isHttpError(e)).toBe(true);
    }
  });
});
