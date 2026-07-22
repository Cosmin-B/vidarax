import { test, expect } from '@playwright/test'

/**
 * Tracing page E2E tests.
 * Verifies the /tracing route renders real Prometheus-backed pipeline metrics.
 */

const METRICS = `
vidarax_pipeline_rtp_frames_received_total 12
vidarax_pipeline_frames_decoded_total 10
vidarax_pipeline_frames_dropped_total 2
vidarax_pipeline_gate_frames_analyzed_total 10
vidarax_pipeline_gate_keyframes_selected_total 3
vidarax_pipeline_keyframes_dropped_total 0
vidarax_pipeline_sink_keyframes_dropped_total 0
vidarax_pipeline_vlm_inferences_total 2
vidarax_pipeline_novelty_evaluated_total 2
vidarax_pipeline_novelty_reused_total 1
vidarax_pipeline_novelty_forced_refresh_total 0
vidarax_pipeline_novelty_embedding_unavailable_total 0
vidarax_pipeline_keyframe_blobs_written_total 2
vidarax_pipeline_keyframe_blobs_reused_total 1
vidarax_pipeline_keyframe_blob_failures_total 0
vidarax_pipeline_keyframe_blob_bytes_total 4096
vidarax_pipeline_restricted_zone_assertions_total 2
vidarax_pipeline_restricted_zone_evidence_failures_total 0
vidarax_pipeline_restricted_zone_queue_dropped_total 0
vidarax_pipeline_trigger_assertions_total 3
vidarax_pipeline_trigger_binary_write_failures_total 0
vidarax_pipeline_trigger_queue_dropped_total 1
vidarax_pipeline_trigger_missing_signal_total 0
vidarax_pipeline_trigger_local_outputs_total 2
vidarax_pipeline_trigger_local_output_failures_total 0
vidarax_infer_admission_active_tokens 128
vidarax_infer_admission_token_limit 4096
vidarax_infer_admission_active_bytes 4096
vidarax_infer_admission_byte_limit 134217728
vidarax_infer_admission_deadline_missed_total 0
vidarax_infer_admission_budget_rejections_total 0
vidarax_infer_admission_urgent_live_acquired_total 2
vidarax_infer_admission_live_acquired_total 1
vidarax_infer_admission_offline_acquired_total 1
vidarax_sse_subscribers_active 1
vidarax_sse_events_total 5
vidarax_sse_replayed_events_total 2
vidarax_sse_output_queue_stalls_total 0
vidarax_sse_commit_to_send_latency_ms_bucket{le="10"} 2
vidarax_sse_commit_to_send_latency_ms_bucket{le="25"} 4
vidarax_sse_commit_to_send_latency_ms_bucket{le="50"} 5
vidarax_sse_commit_to_send_latency_ms_bucket{le="100"} 5
vidarax_sse_commit_to_send_latency_ms_bucket{le="250"} 5
vidarax_sse_commit_to_send_latency_ms_bucket{le="500"} 5
vidarax_sse_commit_to_send_latency_ms_bucket{le="1000"} 5
vidarax_sse_commit_to_send_latency_ms_bucket{le="+Inf"} 5
vidarax_sse_commit_to_send_latency_ms_sum 72
vidarax_sse_commit_to_send_latency_ms_count 5
vidarax_webhooks_configured 1
vidarax_webhook_delivered_total 3
vidarax_webhook_retries_total 0
vidarax_webhook_dead_letters_total 0
vidarax_pipeline_sessions_created_total 1
vidarax_pipeline_sessions_removed_total 0
vidarax_pipeline_generations_active 1
vidarax_pipeline_generations_started_total 1
vidarax_pipeline_generation_shutdown_clean_total 0
vidarax_pipeline_generation_shutdown_faulted_total 0
vidarax_pipeline_generation_shutdown_forced_total 0
vidarax_pipeline_worker_faults_total_all 0
vidarax_pipeline_last_fault_timestamp_seconds 0
vidarax_infer_admission_waiting 0
vidarax_pipeline_decode_latency_us_bucket{le="10000"} 10
vidarax_pipeline_decode_latency_us_bucket{le="+Inf"} 10
vidarax_pipeline_decode_latency_us_sum 50000
vidarax_pipeline_decode_latency_us_count 10
vidarax_pipeline_gate_latency_us_bucket{le="10"} 10
vidarax_pipeline_gate_latency_us_bucket{le="+Inf"} 10
vidarax_pipeline_gate_latency_us_sum 20
vidarax_pipeline_gate_latency_us_count 10
vidarax_pipeline_vlm_latency_ms_bucket{le="1000"} 2
vidarax_pipeline_vlm_latency_ms_bucket{le="+Inf"} 2
vidarax_pipeline_vlm_latency_ms_sum 1200
vidarax_pipeline_vlm_latency_ms_count 2
`

test.describe('Tracing page', () => {
  test.beforeEach(async ({ page }) => {
    // Seed localStorage so the app uses the expected API endpoint
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })

    // Keep this deterministic: the dashboard must display only backend metrics.
    await page.route('http://localhost:8080/v1/metrics', async route => {
      await route.fulfill({ status: 200, contentType: 'text/plain', body: METRICS })
    })
  })

  // ── Route and header ─────────────────────────────────────────────────────────

  test('/tracing renders with correct page title', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page).toHaveTitle(/tracing.*vidarax/i)
  })

  test('"Pipeline Observability" heading is visible', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page.getByRole('heading', { name: /pipeline observability/i })).toBeVisible()
  })

  test('refresh button is visible and labelled', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page.getByRole('button', { name: /refresh metrics/i })).toBeVisible()
  })

  test('LIVE badge is shown when metrics are available', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page.getByText('LIVE', { exact: true })).toBeVisible({ timeout: 10_000 })
  })

  // ── Pipeline stages ──────────────────────────────────────────────────────────

  test('generation supervisor health is visible', async ({ page }) => {
    await page.goto('/tracing')

    const health = page.getByRole('region', { name: 'Pipeline generation health' })
    await expect(health).toBeVisible({ timeout: 10_000 })
    await expect(health.getByText('Healthy', { exact: true })).toBeVisible()
    await expect(health.getByText('Active', { exact: true })).toBeVisible()
    await expect(health.getByText('Worker faults', { exact: true })).toBeVisible()
    await expect(health.getByText('Forced stops', { exact: true })).toBeVisible()
  })

  test('Pipeline Flow section renders with 5 stage cards', async ({ page }) => {
    await page.goto('/tracing')

    const pipelineSection = page.getByRole('region', { name: 'Pipeline flow diagram' })
    await expect(pipelineSection).toBeVisible({ timeout: 10_000 })

    // Each of the 5 stages should appear in the pipeline diagram
    for (const stageName of ['WebRTC', 'Decode', 'Gate', 'VLM', 'Keyframe store']) {
      await expect(pipelineSection.getByText(stageName, { exact: true })).toBeVisible()
    }
  })

  test('Pipeline Flow section shows "Pipeline Flow" heading and LIVE badge', async ({ page }) => {
    await page.goto('/tracing')

    const pipelineSection = page.getByRole('region', { name: 'Pipeline flow diagram' })
    await expect(pipelineSection).toBeVisible({ timeout: 10_000 })
    await expect(pipelineSection.getByRole('heading', { name: /pipeline flow/i })).toBeVisible()
  })

  // ── Metric cards ─────────────────────────────────────────────────────────────

  test('Live Metrics section renders every operational metric card', async ({ page }) => {
    await page.goto('/tracing')

    const metricsSection = page.getByRole('region', { name: 'Live pipeline metrics' })
    await expect(metricsSection).toBeVisible({ timeout: 10_000 })

    for (const cardName of ['Decode', 'Gate Engine', 'VLM Inference', 'Keyframe sidecar', 'Delivery']) {
      await expect(metricsSection.getByRole('heading', { name: cardName })).toBeVisible()
    }
    await expect(metricsSection.getByText('Trigger queue drops', { exact: true })).toBeVisible()
    await expect(metricsSection.getByText('Token reservations', { exact: true })).toBeVisible()
    await expect(metricsSection.getByText('Local trigger outputs', { exact: true })).toBeVisible()
    await expect(metricsSection.getByText('Commit → SSE p95', { exact: true })).toBeVisible()
  })

  test('Live Metrics section header is visible', async ({ page }) => {
    await page.goto('/tracing')

    const metricsSection = page.getByRole('region', { name: 'Live pipeline metrics' })
    await expect(metricsSection).toBeVisible({ timeout: 10_000 })
    await expect(metricsSection.getByText(/pipeline metrics/i)).toBeVisible()
  })

  // ── Refresh ──────────────────────────────────────────────────────────────────

  test('clicking the refresh button does not navigate away', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page.getByRole('button', { name: /refresh metrics/i })).toBeVisible()

    await page.getByRole('button', { name: /refresh metrics/i }).click()
    await expect(page).toHaveURL('/tracing')
  })
})
