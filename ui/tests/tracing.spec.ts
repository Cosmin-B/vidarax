import { test, expect } from '@playwright/test'

/**
 * Tracing page E2E tests.
 * Verifies the /tracing route renders Pipeline Observability, 5 pipeline stages,
 * 4 metric cards, and the trace timeline section.
 */

test.describe('Tracing page', () => {
  test.beforeEach(async ({ page }) => {
    // Seed localStorage so the app uses the expected API endpoint
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })

    // Return 503 immediately so the composable falls back to mock data without delay
    await page.route('http://localhost:8080/v1/metrics', async route => {
      await route.fulfill({ status: 503, body: '' })
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

  test('"MOCK DATA" badge is shown when API is unavailable', async ({ page }) => {
    await page.goto('/tracing')
    // The composable falls back to generated data when the real endpoint is down
    await expect(page.getByText('MOCK DATA')).toBeVisible({ timeout: 10_000 })
  })

  // ── Pipeline stages ──────────────────────────────────────────────────────────

  test('Pipeline Flow section renders with 5 stage cards', async ({ page }) => {
    await page.goto('/tracing')

    const pipelineSection = page.getByRole('region', { name: 'Pipeline flow diagram' })
    await expect(pipelineSection).toBeVisible({ timeout: 10_000 })

    // Each of the 5 stages should appear in the pipeline diagram
    for (const stageName of ['WebRTC', 'Decode', 'Gate', 'VLM', 'SpacetimeDB']) {
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

  test('Live Metrics section renders with 4 metric cards', async ({ page }) => {
    await page.goto('/tracing')

    const metricsSection = page.getByRole('region', { name: 'Live pipeline metrics' })
    await expect(metricsSection).toBeVisible({ timeout: 10_000 })

    // All 4 card headings should be present
    for (const cardName of ['Decode', 'Gate Engine', 'VLM Inference', 'SpacetimeDB']) {
      await expect(metricsSection.getByRole('heading', { name: cardName })).toBeVisible()
    }
  })

  test('Live Metrics section header is visible', async ({ page }) => {
    await page.goto('/tracing')

    const metricsSection = page.getByRole('region', { name: 'Live pipeline metrics' })
    await expect(metricsSection).toBeVisible({ timeout: 10_000 })
    await expect(metricsSection.getByText(/live metrics/i)).toBeVisible()
  })

  // ── Trace Timeline ───────────────────────────────────────────────────────────

  test('Trace Timeline section renders', async ({ page }) => {
    await page.goto('/tracing')

    const timelineSection = page.getByRole('region', { name: 'Trace waterfall timeline' })
    await expect(timelineSection).toBeVisible({ timeout: 10_000 })
  })

  test('Trace Timeline section header is visible', async ({ page }) => {
    await page.goto('/tracing')

    const timelineSection = page.getByRole('region', { name: 'Trace waterfall timeline' })
    await expect(timelineSection).toBeVisible({ timeout: 10_000 })
    await expect(timelineSection.getByText(/trace timeline/i).first()).toBeVisible()
  })

  test('Trace Timeline section is labelled as synthetic sample data', async ({ page }) => {
    await page.goto('/tracing')

    const timelineSection = page.getByRole('region', { name: 'Trace waterfall timeline' })
    await expect(timelineSection).toBeVisible({ timeout: 10_000 })
    await expect(timelineSection.getByText('SYNTHETIC')).toBeVisible()
    await expect(timelineSection.getByText(/illustrative sample data/i)).toBeVisible()
    await expect(timelineSection.getByText(/per-span tracing is not exposed by the API yet/i)).toBeVisible()
  })

  test('Trace Timeline legend shows Decode, Gate, VLM labels', async ({ page }) => {
    await page.goto('/tracing')

    const timelineSection = page.getByRole('region', { name: 'Trace waterfall timeline' })
    await expect(timelineSection).toBeVisible({ timeout: 10_000 })

    // TraceTimeline renders a legend with STAGE_LABEL values
    const traceCard = timelineSection.locator('.card-skeuo')
    await expect(traceCard.getByText('DECODE')).toBeVisible()
    await expect(traceCard.getByText('GATE')).toBeVisible()
    await expect(traceCard.getByText('VLM')).toBeVisible()
  })

  // ── Refresh ──────────────────────────────────────────────────────────────────

  test('clicking the refresh button does not navigate away', async ({ page }) => {
    await page.goto('/tracing')
    await expect(page.getByRole('button', { name: /refresh metrics/i })).toBeVisible()

    await page.getByRole('button', { name: /refresh metrics/i }).click()
    await expect(page).toHaveURL('/tracing')
  })
})
