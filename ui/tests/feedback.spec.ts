import { test, expect } from '@playwright/test'

/**
 * Run detail page — feedback panel E2E tests.
 * Verifies the rating panel renders, buttons are clickable, and submit flow works.
 */

const RUN_ID = 'run-feedback'

const MOCK_RUN = {
  run_id: RUN_ID,
  status: 'completed',
  mode: 'batch',
  model: 'Qwen/Qwen3-VL-2B-Instruct',
  source_uri: 'file://test.mp4',
  created_at: '2024-01-15T10:00:00Z',
  updated_at: '2024-01-15T10:05:00Z',
}

test.describe('Run detail — feedback panel', () => {
  test.beforeEach(async ({ page }) => {
    let policies: Array<Record<string, unknown>> = []
    // Mock the run GET endpoint so the page loads without a real API server
    await page.route(`http://localhost:8080/v1/runs/${RUN_ID}`, async route => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(MOCK_RUN),
        })
      } else {
        await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' })
      }
    })

    // Mock the feedback submission endpoint
    await page.route(`http://localhost:8080/v1/runs/${RUN_ID}/feedback`, async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ request_id: 'req-001', run_id: RUN_ID, status: 'accepted' }),
      })
    })

    await page.route(`http://localhost:8080/v1/runs/${RUN_ID}/policies**`, async route => {
      const url = route.request().url()
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ request_id: 'req-policies', run_id: RUN_ID, policies }),
        })
        return
      }
      if (url.endsWith('/replay')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            request_id: 'req-replay',
            run_id: RUN_ID,
            evaluation_id: 9,
            evaluation: {
              candidate_events: 3,
              accepted_events: 2,
              rejected_events: 1,
              events_without_score: 0,
              threshold: 0.6,
              limitation: 'persisted candidates only',
            },
          }),
        })
        return
      }
      const body = route.request().postDataJSON() as Record<string, unknown>
      if (url.endsWith('/activate')) {
        const parts = url.split('/')
        const revision = Number(parts[parts.length - 2])
        policies = policies.map(policy => policy.revision === revision
          ? { ...policy, status: body.stage, updated_at_ms: 2 }
          : policy)
      } else {
        policies.push({
          revision: 2,
          parent_revision: null,
          status: 'draft',
          prompt: body.prompt,
          output_schema: null,
          parameters: {},
          created_at_ms: 1,
          updated_at_ms: 1,
          effective_generation: null,
          effective_on_current_generation: false,
          deferred_fields: [],
        })
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ request_id: 'req-policy', run_id: RUN_ID, policy: policies[0] }),
      })
    })

    // Seed localStorage with API endpoints
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })

    // Navigate to the run detail page and wait for run data to load
    await page.goto(`/runs/${RUN_ID}`)
    await expect(page.getByText(RUN_ID)).toBeVisible({ timeout: 10_000 })
  })

  // ── Page structure ───────────────────────────────────────────────────────────

  test('run ID and all four tabs render after load', async ({ page }) => {
    await expect(page.getByText(RUN_ID)).toBeVisible()
    await expect(page.getByRole('button', { name: /player/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /keyframes/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /policies/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /metadata/i })).toBeVisible()
  })

  test('creates, promotes, and replays a policy revision', async ({ page }) => {
    await page.getByRole('button', { name: /policies/i }).click()
    await expect(page.getByRole('heading', { name: /policy control loop/i })).toBeVisible()
    await page.getByLabel('New policy prompt').fill('Watch the loading bay')
    await page.getByRole('button', { name: /create revision/i }).click()
    await expect(page.getByText('Revision 2')).toBeVisible()
    await page.getByRole('button', { name: /promote to shadow/i }).click()
    await expect(page.getByText('shadow', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Replay', exact: true }).click()
    await expect(page.getByText(/2 accepted · 1 rejected/)).toBeVisible()
  })

  test('tab switching shows correct content', async ({ page }) => {
    // Default tab is Player.
    await expect(page.getByRole('heading', { name: 'Events', exact: true })).toBeVisible()

    // Switch to keyframes tab
    await page.getByRole('button', { name: /keyframes/i }).click()
    await expect(page.getByRole('heading', { name: /keyframe gallery/i })).toBeVisible()

    // Switch to metadata tab
    await page.getByRole('button', { name: /metadata/i }).click()
    await expect(page.getByRole('heading', { name: /raw metadata/i })).toBeVisible()

    // Switch back to Player
    await page.getByRole('button', { name: /player/i }).click()
    await expect(page.getByRole('heading', { name: 'Events', exact: true })).toBeVisible()
  })

  // ── Feedback panel ───────────────────────────────────────────────────────────

  test('"Rate this analysis" button renders for a completed run', async ({ page }) => {
    await expect(page.getByRole('button', { name: /rate this analysis/i })).toBeVisible()
  })

  test('clicking "Rate this analysis" opens the feedback panel', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()
    await expect(page.getByRole('heading', { name: /rate this analysis/i })).toBeVisible()
    // Toggle button text changes to "Cancel"
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeVisible()
  })

  test('rating buttons 0–10 are all rendered', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    for (let i = 0; i <= 10; i++) {
      await expect(
        page.getByRole('button', { name: String(i), exact: true })
      ).toBeVisible()
    }
  })

  test('clicking a rating button keeps the panel open', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()
    await page.getByRole('button', { name: '7', exact: true }).click()
    // Panel remains open after selecting a rating
    await expect(page.getByRole('heading', { name: /rate this analysis/i })).toBeVisible()
  })

  test('category dropdown has accuracy, latency, quality options', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    const select = page.getByRole('combobox')
    await expect(select).toBeVisible()
    await expect(select.locator('option[value="accuracy"]')).toHaveText('accuracy')
    await expect(select.locator('option[value="latency"]')).toHaveText('latency')
    await expect(select.locator('option[value="quality"]')).toHaveText('quality')
  })

  test('comments textarea is present and accepts input', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    const textarea = page.getByPlaceholder(/additional feedback/i)
    await expect(textarea).toBeVisible()
    await textarea.fill('Latency was excellent')
    await expect(textarea).toHaveValue('Latency was excellent')
  })

  test('"Submit feedback" button is visible but disabled before rating is selected', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    const submitBtn = page.getByRole('button', { name: /submit feedback/i })
    await expect(submitBtn).toBeVisible()
    // No rating has been selected yet (initial value = -1)
    await expect(submitBtn).toBeDisabled()
  })

  test('"Submit feedback" button is enabled after selecting a rating', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    // Submit is disabled before any rating
    const submitBtn = page.getByRole('button', { name: /submit feedback/i })
    await expect(submitBtn).toBeDisabled()

    // Select rating 7
    await page.getByRole('button', { name: '7', exact: true }).click()
    await expect(submitBtn).not.toBeDisabled()
  })

  test('Cancel button hides the feedback panel', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()
    await expect(page.getByRole('heading', { name: /rate this analysis/i })).toBeVisible()

    // The toggle button now reads "Cancel"
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()

    await expect(page.getByRole('heading', { name: /rate this analysis/i })).not.toBeVisible()
    // Toggle button reverts to its original label
    await expect(page.getByRole('button', { name: /rate this analysis/i })).toBeVisible()
  })

  test('submit feedback shows success message and removes rating button', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()
    // Must select a rating before submit becomes enabled
    await page.getByRole('button', { name: '8', exact: true }).click()
    await page.getByRole('button', { name: /submit feedback/i }).click()

    await expect(page.getByText(/feedback submitted/i)).toBeVisible({ timeout: 5_000 })
    // "Rate this analysis" button is removed after a successful submission
    await expect(page.getByRole('button', { name: /rate this analysis/i })).not.toBeVisible()
  })

  // ── Delete confirm flow ──────────────────────────────────────────────────────

  test('delete confirm flow shows Confirm? and can be cancelled', async ({ page }) => {
    // Click Delete to trigger the confirmation step
    await page.getByRole('button', { name: 'Delete', exact: true }).click()
    await expect(page.getByText('Confirm?')).toBeVisible()
    await expect(page.getByRole('button', { name: /yes, delete/i })).toBeVisible()

    // Cancel — confirmation disappears, original Delete button returns
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(page.getByText('Confirm?')).not.toBeVisible()
    await expect(page.getByRole('button', { name: 'Delete', exact: true })).toBeVisible()
  })
})
