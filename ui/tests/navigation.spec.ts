import { test, expect } from '@playwright/test'

/**
 * Navigation E2E tests
 * Verifies all 5 routes render correctly and sidebar navigation works.
 */

test.describe('App navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Pre-seed localStorage so the app doesn't redirect to settings
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })
  })

  // ── Route rendering ──────────────────────────────────────────────────────

  test('dashboard renders at /', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/Dashboard.*Vidarax/i)
    await expect(page.getByRole('heading', { name: /command center/i })).toBeVisible()
  })

  test('stream page renders at /stream', async ({ page }) => {
    await page.goto('/stream')
    await expect(page).toHaveTitle(/Stream.*Vidarax/i)
    await expect(page.getByText('Source', { exact: true })).toBeVisible()
  })

  test('upload page renders at /upload', async ({ page }) => {
    await page.goto('/upload')
    await expect(page).toHaveTitle(/Upload.*Vidarax/i)
    await expect(page.getByRole('heading', { name: /upload video/i })).toBeVisible()
    // Dropzone should be present
    await expect(page.getByRole('button', { name: /drop video file/i })).toBeVisible()
  })

  test('settings page renders at /settings', async ({ page }) => {
    await page.goto('/settings')
    await expect(page).toHaveTitle(/Settings.*Vidarax/i)
    await expect(page.getByRole('heading', { name: /settings/i, level: 2 })).toBeVisible()
  })

  test('unknown route redirects to dashboard', async ({ page }) => {
    await page.goto('/this-route-does-not-exist')
    await expect(page).toHaveURL('/')
    await expect(page.getByRole('heading', { name: /command center/i })).toBeVisible()
  })

  // ── Sidebar navigation ───────────────────────────────────────────────────

  test('sidebar links navigate to correct pages', async ({ page }) => {
    await page.goto('/')

    // Click Stream in sidebar
    await page.getByRole('link', { name: /stream/i }).first().click()
    await expect(page).toHaveURL('/stream')

    // Click Upload in sidebar
    await page.getByRole('link', { name: /upload/i }).first().click()
    await expect(page).toHaveURL('/upload')

    // Click Settings in sidebar
    await page.getByRole('link', { name: /settings/i }).first().click()
    await expect(page).toHaveURL('/settings')

    // Click Dashboard / logo to go back
    await page.getByRole('link', { name: /dashboard|vidarax/i }).first().click()
    await expect(page).toHaveURL('/')
  })

  // ── Browser back/forward ─────────────────────────────────────────────────

  test('browser back/forward works across routes', async ({ page }) => {
    await page.goto('/')
    await page.goto('/stream')
    await page.goto('/upload')

    // Back → stream
    await page.goBack()
    await expect(page).toHaveURL('/stream')

    // Back → dashboard
    await page.goBack()
    await expect(page).toHaveURL('/')

    // Forward → stream
    await page.goForward()
    await expect(page).toHaveURL('/stream')
  })

  // ── Page titles update ───────────────────────────────────────────────────

  test('page title updates on route change', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/dashboard/i)

    await page.goto('/stream')
    await expect(page).toHaveTitle(/stream/i)

    await page.goto('/settings')
    await expect(page).toHaveTitle(/settings/i)
  })

  // ── Run detail redirect ──────────────────────────────────────────────────

  test('run detail page for unknown run stays on page with error', async ({ page }) => {
    // With real API calls, the page will try to fetch and show an error — it should NOT redirect
    await page.goto('/runs/run_nonexistent_000')
    // After loading either shows error or loading — should not redirect to /
    // Give it time to resolve
    await page.waitForTimeout(2000)
    // Still on same URL (the page handles 404 with an error state now)
    await expect(page).toHaveURL('/runs/run_nonexistent_000')
  })

  // ── Upload page CTA ──────────────────────────────────────────────────────

  test('dashboard Upload Video button navigates to /upload', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('link', { name: /upload video/i }).click()
    await expect(page).toHaveURL('/upload')
  })

  test('dashboard New Stream button navigates to /stream', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('link', { name: /new stream/i }).click()
    await expect(page).toHaveURL('/stream')
  })
})
