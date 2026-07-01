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
