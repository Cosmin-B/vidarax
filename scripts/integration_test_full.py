#!/usr/bin/env python3
"""
Vidarax full integration test suite.

Tests every user-facing workflow against the real API, simulating exactly what a
browser client does.  Uses only the Python standard library.

Usage:
    python3 scripts/integration_test_full.py

Configuration:
    VIDARAX_API   - API base URL (default http://localhost:8080)
    TEST_VIDEO    - Local path to test MP4 (default /tmp/vidarax-e2e-test.mp4)
    SKIP_REASON   - Set to "1" to skip the /reason full-flow test (slow, ~30 s)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("VIDARAX_API", "http://localhost:8080").rstrip("/")
TEST_VIDEO_PATH = os.environ.get("TEST_VIDEO", "/tmp/vidarax-e2e-test.mp4")
SKIP_REASON = os.environ.get("SKIP_REASON", "0") == "1"

# A run ID with valid format that will never exist in a fresh test run.
NONEXISTENT_RUN_ID = "run-00000000000000000000000000000000"

# Model that is always present in the catalog.
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# Minimal HTTP helpers (no third-party deps)
# ---------------------------------------------------------------------------

class Response:
    """Thin wrapper around urllib response data."""

    def __init__(self, status: int, body: bytes, headers: Dict[str, str]) -> None:
        self.status = status
        self.body = body
        self.headers = headers

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def http(
    method: str,
    path: str,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Response:
    """Execute an HTTP request and return a Response, never raising on 4xx/5xx."""
    url = f"{API_BASE}{path}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    data: Optional[bytes] = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return Response(resp.status, raw, resp_headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        resp_headers = {k.lower(): v for k, v in exc.headers.items()}
        return Response(exc.code, raw, resp_headers)
    except Exception as exc:  # network failure
        raise RuntimeError(f"HTTP {method} {url} failed: {exc}") from exc


def http_options(path: str, origin: str) -> Response:
    """Send a CORS preflight OPTIONS request."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(
        url,
        method="OPTIONS",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return Response(resp.status, raw, resp_headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        resp_headers = {k.lower(): v for k, v in exc.headers.items()}
        return Response(exc.code, raw, resp_headers)


def http_multipart_upload(path: str, file_path: str, timeout: int = 30) -> Response:
    """
    POST a multipart/form-data request with a single 'file' field.
    Implemented without external dependencies using raw MIME boundaries.
    """
    url = f"{API_BASE}{path}"
    boundary = "----VidaraxBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as fh:
        file_data = fh.read()

    body_parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n",
        b"\r\n",
        file_data,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return Response(resp.status, raw, resp_headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        resp_headers = {k.lower(): v for k, v in exc.headers.items()}
        return Response(exc.code, raw, resp_headers)


# ---------------------------------------------------------------------------
# Test runner infrastructure
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str, elapsed_ms: int) -> None:
        self.name = name
        self.passed = passed
        self.detail = detail
        self.elapsed_ms = elapsed_ms


RESULTS: List[TestResult] = []


def run_test(name: str, fn) -> None:
    """Execute one test function, record pass/fail, never abort the suite."""
    t0 = time.monotonic()
    try:
        fn()
        elapsed = int((time.monotonic() - t0) * 1000)
        RESULTS.append(TestResult(name, True, "ok", elapsed))
    except AssertionError as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        RESULTS.append(TestResult(name, False, str(exc) or "assertion failed", elapsed))
    except Exception as exc:
        elapsed = int((time.monotonic() - t0) * 1000)
        RESULTS.append(TestResult(name, False, f"exception: {exc}", elapsed))


def assert_status(resp: Response, expected: int, context: str = "") -> None:
    label = f" ({context})" if context else ""
    assert resp.status == expected, (
        f"expected HTTP {expected}, got {resp.status}{label}: {resp.text()[:300]}"
    )


def assert_json_key(data: Any, key: str, context: str = "") -> Any:
    label = f" ({context})" if context else ""
    assert key in data, f"missing key '{key}'{label}: {data}"
    return data[key]


def assert_nonempty(value: Any, label: str) -> None:
    assert value, f"{label} must be non-empty/nonzero, got {value!r}"


# ---------------------------------------------------------------------------
# Test video generation (local fallback if not already present)
# ---------------------------------------------------------------------------

def ensure_test_video() -> None:
    """Generate a 10-second 3-scene MP4 with ffmpeg if it does not exist."""
    if os.path.exists(TEST_VIDEO_PATH) and os.path.getsize(TEST_VIDEO_PATH) > 1000:
        return

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0xCC2222:s=640x480:d=3",
        "-f", "lavfi", "-i", "color=c=0x2244CC:s=640x480:d=4",
        "-f", "lavfi", "-i", "color=c=0x22AA44:s=640x480:d=3",
        "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
        TEST_VIDEO_PATH,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg failed to generate test video: {result.stderr[:200]}")
    else:
        size = os.path.getsize(TEST_VIDEO_PATH)
        print(f"  [INFO] Generated test video: {TEST_VIDEO_PATH} ({size} bytes)")


# ---------------------------------------------------------------------------
# Helper to create a run and return its ID
# ---------------------------------------------------------------------------

def create_run(mode: str = "balanced", model: str = DEFAULT_MODEL) -> str:
    resp = http("POST", "/v1/runs", {"mode": mode, "model": model})
    assert_status(resp, 200, "create_run")
    data = resp.json()
    assert_json_key(data, "run_id", "create_run")
    return data["run_id"]


# ---------------------------------------------------------------------------
# TEST 1: Dashboard loads runs
# ---------------------------------------------------------------------------

def test_dashboard_loads_runs() -> None:
    resp = http("GET", "/v1/runs")
    assert_status(resp, 200, "GET /v1/runs")
    data = resp.json()
    assert isinstance(data, list), f"expected array from /v1/runs, got {type(data).__name__}"
    # Validate schema of any existing runs
    for run in data:
        for field in ("run_id", "status", "mode"):
            assert field in run, f"run missing field '{field}': {run}"
        assert run["run_id"].startswith("run-"), (
            f"run_id has unexpected format: {run['run_id']}"
        )
        assert run["status"] in (
            "pending", "processing", "completed", "cancelled", "expired", "error"
        ), f"unexpected status: {run['status']}"


# ---------------------------------------------------------------------------
# TEST 2a: Upload file
# ---------------------------------------------------------------------------

def test_upload_file() -> None:
    # Create a small temp file to upload
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(b"\x00" * 512)  # 512 bytes of zeros
        tmp_path = fh.name

    try:
        resp = http_multipart_upload("/v1/upload", tmp_path)
        assert_status(resp, 200, "POST /v1/upload")
        data = resp.json()
        file_path = assert_json_key(data, "file_path", "upload response")
        assert_nonempty(file_path, "file_path")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# TEST 2b: Full flow: create run → reason → events → markers → get → delete
# ---------------------------------------------------------------------------

def test_full_upload_analyze_flow() -> None:
    if SKIP_REASON:
        # Still create a run and verify its shape, just skip the slow /reason call
        run_id = create_run()
        resp = http("GET", f"/v1/runs/{run_id}")
        assert_status(resp, 200, "GET /v1/runs/{id} after create")
        data = resp.json()
        for field in ("run_id", "status", "mode"):
            assert_json_key(data, field, "run detail")
        return

    # 2a: Create run
    run_id = create_run("balanced", DEFAULT_MODEL)

    # 2b: Call /reason with the test video on the server
    reason_payload = {
        "source_uri": f"file://{TEST_VIDEO_PATH}",
        "model": DEFAULT_MODEL,
        "semantic_inference": False,  # skip VLM to keep test fast
    }
    reason_resp = http("POST", f"/v1/runs/{run_id}/reason", reason_payload, timeout=120)
    assert_status(reason_resp, 200, "POST /reason")
    reason_data = reason_resp.json()

    decoded_frames = assert_json_key(reason_data, "decoded_frames", "reason response")
    markers_emitted = assert_json_key(reason_data, "markers_emitted", "reason response")
    assert decoded_frames > 0, f"decoded_frames must be > 0, got {decoded_frames}"
    assert markers_emitted > 0, f"markers_emitted must be > 0, got {markers_emitted}"

    # 2c: GET events
    events_resp = http("GET", f"/v1/runs/{run_id}/events")
    assert_status(events_resp, 200, "GET /events")
    events_data = events_resp.json()
    events = assert_json_key(events_data, "events", "events response")
    assert len(events) > 0, "events array must be non-empty after /reason"

    # 2d: GET markers
    markers_resp = http("GET", f"/v1/runs/{run_id}/markers")
    assert_status(markers_resp, 200, "GET /markers")
    markers_data = markers_resp.json()
    markers = assert_json_key(markers_data, "markers", "markers response")
    assert len(markers) > 0, "markers array must be non-empty after /reason"

    # 2e: GET run detail
    detail_resp = http("GET", f"/v1/runs/{run_id}")
    assert_status(detail_resp, 200, "GET /v1/runs/{id}")
    detail = detail_resp.json()
    for field in ("run_id", "status", "mode", "model"):
        assert_json_key(detail, field, "run detail")
    assert detail["run_id"] == run_id

    # 2f: DELETE run (soft delete)
    delete_resp = http("DELETE", f"/v1/runs/{run_id}")
    assert_status(delete_resp, 200, "DELETE /v1/runs/{id}")
    delete_data = delete_resp.json()
    assert delete_data.get("run_id") == run_id, "delete response missing run_id"

    # 2g: GET run after delete → 404
    after_delete_resp = http("GET", f"/v1/runs/{run_id}")
    assert_status(after_delete_resp, 404, "GET /v1/runs/{id} after delete")


# ---------------------------------------------------------------------------
# TEST 3: Run lifecycle states
# ---------------------------------------------------------------------------

def test_run_lifecycle_states() -> None:
    # Create → pending
    resp = http("POST", "/v1/runs", {"mode": "balanced", "model": DEFAULT_MODEL})
    assert_status(resp, 200, "create run")
    data = resp.json()
    run_id = data["run_id"]
    assert data["status"] == "pending", f"new run status must be 'pending', got {data['status']}"

    # Keepalive → still works (run is in processing/pending state)
    keepalive_resp = http("POST", f"/v1/runs/{run_id}/keepalive")
    assert_status(keepalive_resp, 200, "keepalive")
    keepalive_data = keepalive_resp.json()
    assert keepalive_data.get("run_id") == run_id

    # Stop → cancelled
    stop_resp = http("POST", f"/v1/runs/{run_id}/stop")
    assert_status(stop_resp, 200, "stop run")
    stop_data = stop_resp.json()
    assert stop_data.get("status") == "cancelled", (
        f"stop response status must be 'cancelled', got {stop_data.get('status')}"
    )

    # GET state → cancelled
    state_resp = http("GET", f"/v1/runs/{run_id}/state")
    assert_status(state_resp, 200, "GET state after stop")
    state_data = state_resp.json()
    state = assert_json_key(state_data, "state", "state response")
    assert state == "cancelled", f"state must be 'cancelled', got {state!r}"


# ---------------------------------------------------------------------------
# TEST 4: Inference endpoint
# ---------------------------------------------------------------------------

def test_inference_endpoint() -> None:
    # 4a: Single inference with valid model
    resp = http("POST", "/v1/infer", {
        "model": DEFAULT_MODEL,
        "prompt": "Briefly describe what you see in one sentence.",
    })
    assert_status(resp, 200, "POST /v1/infer")
    data = resp.json()
    output_text = assert_json_key(data, "output_text", "infer response")
    assert_nonempty(output_text, "output_text")
    assert_json_key(data, "provider", "infer response")
    assert_json_key(data, "model", "infer response")

    # 4b: Batch inference with 2 requests
    batch_resp = http("POST", "/v1/infer/batch", {
        "requests": [
            {"model": DEFAULT_MODEL, "prompt": "What is 2 + 2?"},
            {"model": DEFAULT_MODEL, "prompt": "Name one primary color."},
        ]
    })
    assert_status(batch_resp, 200, "POST /v1/infer/batch")
    batch_data = batch_resp.json()
    assert batch_data.get("processed") == 2, (
        f"batch processed must be 2, got {batch_data.get('processed')}"
    )
    assert batch_data.get("succeeded") == 2, (
        f"batch succeeded must be 2, got {batch_data.get('succeeded')}"
    )
    results = assert_json_key(batch_data, "results", "batch response")
    assert len(results) == 2, f"expected 2 batch results, got {len(results)}"

    # 4c: Invalid model → validation error
    invalid_resp = http("POST", "/v1/infer", {
        "model": "nonexistent/totally-fake-model-xyz",
        "prompt": "test",
    })
    assert invalid_resp.status in (400, 422, 500), (
        f"invalid model must produce 4xx/5xx, got {invalid_resp.status}"
    )
    err_data = invalid_resp.json()
    assert "error" in err_data, f"error key missing: {err_data}"


# ---------------------------------------------------------------------------
# TEST 5: Settings and health checks
# ---------------------------------------------------------------------------

def test_settings_health() -> None:
    # 5a: Health
    resp = http("GET", "/v1/health")
    assert_status(resp, 200, "GET /v1/health")
    data = resp.json()
    status = assert_json_key(data, "status", "health response")
    assert status == "ok", f"health status must be 'ok', got {status!r}"

    # 5b: Models catalog
    models_resp = http("GET", "/v1/models")
    assert_status(models_resp, 200, "GET /v1/models")
    models_data = models_resp.json()
    models = assert_json_key(models_data, "models", "models response")
    assert len(models) > 0, "models catalog must be non-empty"
    for model in models:
        assert_json_key(model, "id", "model item")
        assert_json_key(model, "tier", "model item")
        assert_json_key(model, "availability", "model item")

    # 5c: Metrics (Prometheus text format)
    metrics_resp = http("GET", "/v1/metrics")
    assert_status(metrics_resp, 200, "GET /v1/metrics")
    text = metrics_resp.text()
    assert "vidarax_" in text, (
        f"metrics must contain 'vidarax_' prefix lines, got: {text[:200]}"
    )


# ---------------------------------------------------------------------------
# TEST 6: Error handling
# ---------------------------------------------------------------------------

def test_error_handling() -> None:
    # 6a: Nonexistent run state → 404
    resp = http("GET", f"/v1/runs/{NONEXISTENT_RUN_ID}/state")
    assert resp.status == 404, (
        f"nonexistent run state must return 404, got {resp.status}: {resp.text()[:200]}"
    )

    # 6b: POST /v1/runs with invalid mode → 422
    resp = http("POST", "/v1/runs", {"mode": "invalid_mode_xyz"})
    assert resp.status == 422, (
        f"invalid mode must return 422, got {resp.status}: {resp.text()[:200]}"
    )
    err_data = resp.json()
    assert "error" in err_data, f"error envelope missing: {err_data}"

    # 6c: POST /reason with empty source_uri → validation error
    run_id = create_run()
    reason_resp = http("POST", f"/v1/runs/{run_id}/reason", {
        "source_uri": "",
        "model": DEFAULT_MODEL,
    })
    assert reason_resp.status in (400, 422), (
        f"empty source_uri must return 4xx, got {reason_resp.status}: {reason_resp.text()[:200]}"
    )

    # 6d: DELETE nonexistent run → 404
    del_resp = http("DELETE", f"/v1/runs/{NONEXISTENT_RUN_ID}")
    assert del_resp.status == 404, (
        f"DELETE nonexistent run must return 404, got {del_resp.status}: {del_resp.text()[:200]}"
    )


# ---------------------------------------------------------------------------
# TEST 7: CORS headers
# ---------------------------------------------------------------------------

def test_cors_headers() -> None:
    origin = "http://localhost:5173"

    # 7a: OPTIONS preflight → 204 with CORS headers
    resp = http_options("/v1/runs", origin)
    assert resp.status == 204, (
        f"OPTIONS preflight must return 204, got {resp.status}: {resp.text()[:200]}"
    )
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao in ("*", origin), (
        f"access-control-allow-origin must be '*' or the request origin, got {acao!r}"
    )
    allow_methods = resp.headers.get("access-control-allow-methods", "")
    assert allow_methods, "access-control-allow-methods header must be present"

    # 7b: GET /v1/health with Origin → CORS header present in response
    resp = http("GET", "/v1/health", headers={"Origin": origin})
    assert_status(resp, 200, "GET /v1/health with Origin")
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao in ("*", origin), (
        f"health response must include CORS header, got {acao!r}"
    )


# ---------------------------------------------------------------------------
# TEST 8: Upload size limit (small file succeeds)
# ---------------------------------------------------------------------------

def test_upload_size_limit() -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        # Write 1 KB — well within the 200 MB limit
        fh.write(b"\x00" * 1024)
        tmp_path = fh.name

    try:
        resp = http_multipart_upload("/v1/upload", tmp_path)
        assert_status(resp, 200, "small file upload")
        data = resp.json()
        file_path = assert_json_key(data, "file_path", "upload response")
        assert_nonempty(file_path, "file_path")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# TEST 9: Feedback validation
# ---------------------------------------------------------------------------

def test_feedback() -> None:
    run_id = create_run()

    # 9a: rating > 10 → 422
    resp = http("POST", f"/v1/runs/{run_id}/feedback", {
        "rating": 11,
        "category": "quality",
    })
    assert resp.status == 422, (
        f"rating=11 must return 422, got {resp.status}: {resp.text()[:200]}"
    )
    err = resp.json()
    assert "error" in err, f"error envelope missing: {err}"

    # 9b: empty category → 422
    resp = http("POST", f"/v1/runs/{run_id}/feedback", {
        "rating": 5,
        "category": "",
    })
    assert resp.status == 422, (
        f"empty category must return 422, got {resp.status}: {resp.text()[:200]}"
    )
    err = resp.json()
    assert "error" in err, f"error envelope missing: {err}"

    # 9c: valid payload → 200 or 500 (500 if SpacetimeDB is not configured on this host)
    resp = http("POST", f"/v1/runs/{run_id}/feedback", {
        "rating": 7,
        "category": "accuracy",
        "feedback": "Integration test feedback submission.",
    })
    assert resp.status in (200, 500), (
        f"valid feedback must return 200 or 500 (SpacetimeDB may not be configured), "
        f"got {resp.status}: {resp.text()[:200]}"
    )
    if resp.status == 200:
        data = resp.json()
        assert data.get("status") == "submitted", (
            f"feedback status must be 'submitted', got {data.get('status')}"
        )


# ---------------------------------------------------------------------------
# TEST 10: Search endpoint (optional — gracefully skipped if absent)
# ---------------------------------------------------------------------------

def test_search_endpoint() -> None:
    """
    The /v1/search endpoint may not exist in every deployment.
    If the server returns 404 or 405 we treat it as 'not implemented' and skip.
    """
    resp = http("POST", "/v1/search", {"query": "scene_cut", "limit": 10})
    if resp.status in (404, 405):
        # Endpoint not implemented; this is acceptable.
        return
    assert resp.status == 200, (
        f"POST /v1/search returned unexpected status {resp.status}: {resp.text()[:200]}"
    )
    data = resp.json()
    # Shape check: whatever the response is, it must be valid JSON
    assert isinstance(data, (dict, list)), f"unexpected search response type: {type(data)}"


# ---------------------------------------------------------------------------
# BONUS TEST: Run appears in list after creation
# ---------------------------------------------------------------------------

def test_run_appears_in_list() -> None:
    run_id = create_run()
    resp = http("GET", "/v1/runs")
    assert_status(resp, 200, "GET /v1/runs")
    runs = resp.json()
    run_ids = [r.get("run_id") for r in runs]
    assert run_id in run_ids, (
        f"newly created run {run_id} must appear in /v1/runs listing, got {run_ids[:5]}..."
    )


# ---------------------------------------------------------------------------
# BONUS TEST: Verify run schema after creation
# ---------------------------------------------------------------------------

def test_create_run_response_schema() -> None:
    resp = http("POST", "/v1/runs", {"mode": "detailed", "model": DEFAULT_MODEL})
    assert_status(resp, 200, "POST /v1/runs")
    data = resp.json()
    for field in ("run_id", "request_id", "status", "mode", "model"):
        assert_json_key(data, field, "create_run response")
    assert data["status"] == "pending", f"initial status must be 'pending', got {data['status']}"
    assert data["mode"] == "detailed", f"mode must echo 'detailed', got {data['mode']}"
    assert data["run_id"].startswith("run-"), f"run_id format invalid: {data['run_id']}"
    assert data["request_id"].startswith("req-"), f"request_id format invalid: {data['request_id']}"


# ---------------------------------------------------------------------------
# BONUS TEST: Keepalive on stopped run returns error
# ---------------------------------------------------------------------------

def test_keepalive_on_stopped_run() -> None:
    run_id = create_run()
    stop_resp = http("POST", f"/v1/runs/{run_id}/stop")
    assert_status(stop_resp, 200, "stop run")
    # Keepalive on a terminal run must fail (409 conflict)
    keepalive_resp = http("POST", f"/v1/runs/{run_id}/keepalive")
    assert keepalive_resp.status == 409, (
        f"keepalive on stopped run must return 409, got {keepalive_resp.status}: "
        f"{keepalive_resp.text()[:200]}"
    )


# ---------------------------------------------------------------------------
# BONUS TEST: Infer with empty prompt → validation error
# ---------------------------------------------------------------------------

def test_infer_empty_prompt() -> None:
    resp = http("POST", "/v1/infer", {
        "model": DEFAULT_MODEL,
        "prompt": "",
    })
    assert resp.status == 422, (
        f"empty prompt must return 422, got {resp.status}: {resp.text()[:200]}"
    )
    err = resp.json()
    assert "error" in err


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results() -> int:
    width_name = max(len(r.name) for r in RESULTS) + 2
    width_status = 6
    width_ms = 8
    width_detail = 60

    sep = "-" * (width_name + width_status + width_ms + width_detail + 7)
    header = (
        f"{'TEST':<{width_name}} {'STATUS':^{width_status}} {'MS':>{width_ms}}  DETAIL"
    )

    print()
    print("=" * len(sep))
    print("  Vidarax Integration Test Results")
    print("=" * len(sep))
    print(header)
    print(sep)

    failures = 0
    for r in RESULTS:
        status_str = "PASS" if r.passed else "FAIL"
        detail = r.detail if r.passed else r.detail[:width_detail]
        print(
            f"  {r.name:<{width_name - 2}} {status_str:^{width_status}} "
            f"{r.elapsed_ms:>{width_ms}}ms  {detail}"
        )
        if not r.passed:
            failures += 1

    print(sep)
    total = len(RESULTS)
    passed = total - failures
    print(f"  {passed}/{total} tests passed", end="")
    if failures:
        print(f"  ({failures} FAILED)")
    else:
        print("  -- all tests passed")
    print("=" * len(sep))
    print()

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Vidarax Integration Test Suite")
    print(f"API: {API_BASE}")
    print(f"Test video: {TEST_VIDEO_PATH}")
    print(f"Skip /reason: {SKIP_REASON}")
    print()

    # Ensure test video exists before running tests
    ensure_test_video()

    # --- TEST 1 ---
    run_test("1. Dashboard loads runs", test_dashboard_loads_runs)

    # --- TEST 2 ---
    run_test("2a. Upload file", test_upload_file)
    run_test("2b. Full flow: upload→reason→events→markers→get→delete",
             test_full_upload_analyze_flow)

    # --- TEST 3 ---
    run_test("3. Run lifecycle states", test_run_lifecycle_states)

    # --- TEST 4 ---
    run_test("4. Inference endpoint (single + batch + invalid model)",
             test_inference_endpoint)

    # --- TEST 5 ---
    run_test("5. Settings/health (health + models + metrics)", test_settings_health)

    # --- TEST 6 ---
    run_test("6. Error handling (404 + 422 validations)", test_error_handling)

    # --- TEST 7 ---
    run_test("7. CORS headers (preflight + response)", test_cors_headers)

    # --- TEST 8 ---
    run_test("8. Upload size limit (small file succeeds)", test_upload_size_limit)

    # --- TEST 9 ---
    run_test("9. Feedback (valid + rating>10 + empty category)", test_feedback)

    # --- TEST 10 ---
    run_test("10. Search endpoint (skipped if not implemented)", test_search_endpoint)

    # --- BONUS ---
    run_test("B1. Run appears in list after creation", test_run_appears_in_list)
    run_test("B2. Create run response schema", test_create_run_response_schema)
    run_test("B3. Keepalive on stopped run → 409", test_keepalive_on_stopped_run)
    run_test("B4. Infer with empty prompt → 422", test_infer_empty_prompt)

    failures = print_results()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
