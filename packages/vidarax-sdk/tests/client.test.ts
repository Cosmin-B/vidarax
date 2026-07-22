import { afterEach, describe, expect, it, vi } from "vitest";

import { Vidarax } from "../src/client.js";

describe("WHIP offer", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the server-created run ID header when present", async () => {
    const fetchMock = vi.fn(async () => new Response("v=0\r\n", {
      status: 201,
      headers: {
        Location: "/v1/stream/whip/sess-sdk0001",
        "x-vidarax-run-id": "run-sdk0001",
      },
    }));
    vi.stubGlobal(
      "fetch",
      fetchMock,
    );

    const client = new Vidarax("http://127.0.0.1:8080", { timeoutMs: 1000 });
    const attachConfig = { prompt: "watch motion 🚪", "token-cap": 42 };

    const session = await client.whipOffer("v=0\r\n", attachConfig);

    expect(session).toMatchObject({
      sessionId: "sess-sdk0001",
      runId: "run-sdk0001",
      answerSdp: "v=0\r\n",
      resourceUrl: "http://127.0.0.1:8080/v1/stream/whip/sess-sdk0001",
    });
    const [, init] = fetchMock.mock.calls[0];
    const headers = init?.headers as Record<string, string>;
    const encoded = headers["x-attach-config"];
    const base64 = encoded.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + (4 - base64.length % 4) % 4, "=");
    const bytes = Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
    expect(JSON.parse(new TextDecoder().decode(bytes))).toEqual(attachConfig);
  });
});

describe("policy lifecycle", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses the run-scoped revision, replay, and activation endpoints", async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/replay")) {
        return Response.json({
          request_id: "req-replay",
          run_id: "run-sdk0001",
          evaluation_id: 8,
          evaluation: {
            revision: 2,
            source: "local_wal",
            comparison: "persisted candidates",
            from_seq: 0,
            to_seq: null,
            candidate_events: 1,
            accepted_events: 1,
            rejected_events: 0,
            events_without_score: 0,
            threshold: 0.6,
            limitation: "persisted candidates only",
          },
        });
      }
      const body = JSON.parse(String(init?.body ?? "{}"));
      return Response.json({
        request_id: "req-policy",
        run_id: "run-sdk0001",
        policy: {
          revision: 2,
          parent_revision: null,
          status: body.stage ?? "draft",
          prompt: body.prompt ?? null,
          output_schema: null,
          parameters: body.parameters ?? {},
          created_at_ms: 1,
          updated_at_ms: 1,
          effective_generation: null,
          effective_on_current_generation: false,
          deferred_fields: [],
        },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new Vidarax("http://127.0.0.1:8080");
    const created = await client.createPolicy("run-sdk0001", { prompt: "watch the bay" });
    await client.activatePolicy("run-sdk0001", created.policy.revision, { stage: "shadow" });
    const replay = await client.replayPolicy("run-sdk0001", created.policy.revision);

    expect(created.policy.status).toBe("draft");
    expect(replay.evaluation.source).toBe("local_wal");
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([
      "http://127.0.0.1:8080/v1/runs/run-sdk0001/policies",
      "http://127.0.0.1:8080/v1/runs/run-sdk0001/policies/2/activate",
      "http://127.0.0.1:8080/v1/runs/run-sdk0001/policies/2/replay",
    ]);
  });
});

describe("trigger programs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses compile, validate, and evaluate endpoints with typed programs", async () => {
    const program = {
      isa_version: 1,
      program_id: "loading-bay-entry",
      version: 1,
      instructions: [
        { op: "load", signal: { kind: "motion_score" } },
        { op: "constant", value: 0.4 },
        { op: "greater_or_equal" },
        { op: "emit", event_type: "loading_bay_entry" },
        { op: "capture", kind: "keyframe", before_ms: 0, after_ms: 0 },
        { op: "halt" },
      ],
    } as const;
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const path = new URL(String(input)).pathname;
      if (path.endsWith("/compile")) {
        return Response.json({
          request_id: "req-compile",
          instruction_count: program.instructions.length,
          state_slots: 0,
          program,
        });
      }
      if (path.endsWith("/validate")) {
        return Response.json({
          request_id: "req-validate",
          valid: true,
          isa_version: 1,
          program_id: program.program_id,
          program_version: 1,
          instruction_count: program.instructions.length,
          state_slots: 0,
        });
      }
      return Response.json({
        request_id: "req-evaluate",
        program_id: program.program_id,
        program_version: 1,
        results: [{ pts_ms: 0, fired: false, missing_signal: false, actions: [] }],
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new Vidarax("http://127.0.0.1:8080");

    const compiled = await client.compileTrigger("trigger loading-bay-entry version 1");
    await client.validateTrigger(compiled.program);
    const evaluated = await client.evaluateTrigger(compiled.program, [
      { pts_ms: 0, motion_score: 0.1 },
    ]);

    expect(evaluated.program_id).toBe("loading-bay-entry");
    expect(fetchMock.mock.calls.map(([url]) => new URL(String(url)).pathname)).toEqual([
      "/v1/triggers/compile",
      "/v1/triggers/validate",
      "/v1/triggers/evaluate",
    ]);
  });
});

describe("durable event subscription", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses CloudEvents SSE and sends the durable cursor", async () => {
    const body = [
      "id: 6",
      "event: gate",
      'data: {"sequence":6,"pts_ms":100,"data":{"decision":"keep"}}',
      "",
      "id: 7",
      "event: keyframe_stored",
      'data: {"sequence":7,"pts_ms":110,"data":{"image_ref":"/v1/runs/run/keyframes/abc"}}',
      "",
      "",
    ].join("\n");
    const fetchMock = vi.fn(async () => new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const client = new Vidarax("https://vidarax.example", { apiKey: "test-key" });
    const events = [];
    for await (const event of client.subscribeEvents("run-0000000000000000", {
      after: 5,
      reconnect: false,
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      { seq: 6, pts_ms: 100, kind: "gate", payload: { decision: "keep" } },
      {
        seq: 7,
        pts_ms: 110,
        kind: "keyframe_stored",
        payload: { image_ref: "/v1/runs/run/keyframes/abc" },
      },
    ]);
    expect(fetchMock.mock.calls[0][0]).toContain("after=5");
    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers["Last-Event-ID"]).toBe("5");
    expect(headers["x-api-key"]).toBe("test-key");
  });
});
