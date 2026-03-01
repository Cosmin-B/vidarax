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
  model: 'qwen2.5-vl-7b',
  source_uri: 'file://test.mp4',
  created_at: '2024-01-15T10:00:00Z',
  updated_at: '2024-01-15T10:05:00Z',
}

test.describe('Run detail — feedback panel', () => {
  test.beforeEach(async ({ page }) => {
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

  test('run ID and all three tabs render after load', async ({ page }) => {
    await expect(page.getByText(RUN_ID)).toBeVisible()
    await expect(page.getByRole('button', { name: /timeline/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /keyframes/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /metadata/i })).toBeVisible()
  })

  test('tab switching shows correct content', async ({ page }) => {
    // Default tab is timeline — event timeline heading should be visible
    await expect(page.getByRole('heading', { name: /event timeline/i })).toBeVisible()

    // Switch to keyframes tab
    await page.getByRole('button', { name: /keyframes/i }).click()
    await expect(page.getByRole('heading', { name: /keyframe gallery/i })).toBeVisible()

    // Switch to metadata tab
    await page.getByRole('button', { name: /metadata/i }).click()
    await expect(page.getByRole('heading', { name: /raw metadata/i })).toBeVisible()

    // Switch back to timeline
    await page.getByRole('button', { name: /timeline/i }).click()
    await expect(page.getByRole('heading', { name: /event timeline/i })).toBeVisible()
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

  test('"Submit feedback" button is visible and enabled', async ({ page }) => {
    await page.getByRole('button', { name: /rate this analysis/i }).click()

    const submitBtn = page.getByRole('button', { name: /submit feedback/i })
    await expect(submitBtn).toBeVisible()
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
