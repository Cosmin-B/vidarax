import { test, expect } from '@playwright/test'

/**
 * Settings E2E tests
 * Verifies all settings inputs render, values persist after save + reload.
 */

const DEFAULT_ENDPOINT = 'http://localhost:8080'
const CUSTOM_ENDPOINT = 'http://vidarax.example.com:9090'
const CUSTOM_STDB = 'http://stdb.example.com:3001'

test.describe('Settings panel', () => {
  test.beforeEach(async ({ page }) => {
    // Start with clean localStorage
    await page.goto('/settings')
    await page.evaluate(() => localStorage.clear())
    await page.reload()
  })

  // ── Page structure ───────────────────────────────────────────────────────

  test('all accordion sections are visible', async ({ page }) => {
    await page.goto('/settings')
    await expect(page.getByRole('button', { name: /connection/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /models/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /stream/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /gate engine/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /advanced/i })).toBeVisible()
  })

  test('save button is visible', async ({ page }) => {
    await page.goto('/settings')
    await expect(page.getByTestId('save-button')).toBeVisible()
    await expect(page.getByTestId('save-button')).toContainText('Save')
  })

  test('reset defaults button is visible', async ({ page }) => {
    await page.goto('/settings')
    await expect(page.getByRole('button', { name: /reset defaults/i })).toBeVisible()
  })

  // ── Connection section ───────────────────────────────────────────────────

  test('connection section: inputs populate and save', async ({ page }) => {
    await page.goto('/settings')

    // Connection section should be open by default
    const apiInput = page.getByTestId('api-endpoint')
    await expect(apiInput).toBeVisible()

    // Clear and type a custom endpoint
    await apiInput.clear()
    await apiInput.fill(CUSTOM_ENDPOINT)

    const stdbInput = page.getByTestId('spacetime-endpoint')
    await stdbInput.clear()
    await stdbInput.fill(CUSTOM_STDB)

    // Save
    await page.getByTestId('save-button').click()
    await expect(page.getByTestId('save-button')).toContainText('Saved')

    // Reload and verify persistence
    await page.reload()
    await expect(page.getByTestId('api-endpoint')).toHaveValue(CUSTOM_ENDPOINT)
    await expect(page.getByTestId('spacetime-endpoint')).toHaveValue(CUSTOM_STDB)
  })

  test('connection section: test connection button is present', async ({ page }) => {
    await page.goto('/settings')
    await expect(page.getByTestId('test-connection')).toBeVisible()
    await expect(page.getByTestId('test-connection')).toContainText('Test Connection')
  })

  // ── Models section ───────────────────────────────────────────────────────

  test('models section: default model select is visible', async ({ page }) => {
    await page.goto('/settings')

    // Open models section
    await page.getByRole('button', { name: /models/i }).click()
    await expect(page.getByTestId('default-model')).toBeVisible()
    await expect(page.getByTestId('first-pass-model')).toBeVisible()
    await expect(page.getByTestId('second-pass-model')).toBeVisible()
  })

  test('models section: model selection persists', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /models/i }).click()

    const defaultModelSelect = page.getByTestId('default-model')
    await defaultModelSelect.selectOption('qwen2.5-vl-2b')
    await expect(defaultModelSelect).toHaveValue('qwen2.5-vl-2b')

    await page.getByTestId('save-button').click()
    await page.reload()

    // Reopen models section
    await page.getByRole('button', { name: /models/i }).click()
    await expect(page.getByTestId('default-model')).toHaveValue('qwen2.5-vl-2b')
  })

  // ── Stream section ───────────────────────────────────────────────────────

  test('stream section: FPS slider and value display', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()

    const slider = page.getByTestId('fps-slider')
    await expect(slider).toBeVisible()

    // Change value by setting the input value directly
    await slider.fill('15')
    await expect(page.getByTestId('fps-value')).toContainText('15')
  })

  test('stream section: FPS value persists after save and reload', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()

    await page.getByTestId('fps-slider').fill('10')
    await page.getByTestId('save-button').click()

    await page.reload()
    await page.getByRole('button', { name: /^stream$/i }).click()
    await expect(page.getByTestId('fps-value')).toContainText('10')
  })

  test('stream section: semantic inference toggle', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()

    const toggle = page.getByTestId('semantic-inference-toggle')
    await expect(toggle).toBeVisible()

    // Check initial aria-checked state
    const initialState = await toggle.getAttribute('aria-checked')

    // Click to toggle
    await toggle.click()
    const newState = await toggle.getAttribute('aria-checked')
    expect(newState).not.toBe(initialState)
  })

  // ── Gate Engine section ──────────────────────────────────────────────────

  test('gate engine section: sliders are present', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /gate engine/i }).click()

    // Hamming threshold
    await expect(page.getByRole('slider').nth(0)).toBeVisible()
    // Loop detection toggle
    await expect(page.getByRole('switch')).toBeVisible()
  })

  // ── Save confirmation ────────────────────────────────────────────────────

  test('save button shows "Saved!" then reverts', async ({ page }) => {
    await page.goto('/settings')
    const btn = page.getByTestId('save-button')
    await btn.click()
    await expect(btn).toContainText('Saved!')
    // Reverts after ~2s
    await expect(btn).toContainText('Save', { timeout: 4000 })
  })

  // ── Reset to defaults ────────────────────────────────────────────────────

  test('reset to defaults restores default endpoint', async ({ page }) => {
    await page.goto('/settings')

    // Change endpoint
    await page.getByTestId('api-endpoint').fill(CUSTOM_ENDPOINT)
    await page.getByTestId('save-button').click()

    // Reset
    await page.getByRole('button', { name: /reset defaults/i }).click()

    // Default endpoint should be restored in the form
    await expect(page.getByTestId('api-endpoint')).toHaveValue(DEFAULT_ENDPOINT)
  })

  // ── Persistence across multiple keys ─────────────────────────────────────

  test('all saved values appear in localStorage', async ({ page }) => {
    await page.goto('/settings')

    await page.getByTestId('api-endpoint').fill(CUSTOM_ENDPOINT)
    await page.getByRole('button', { name: /^stream$/i }).click()
    await page.getByTestId('fps-slider').fill('20')

    await page.getByTestId('save-button').click()

    const storedEndpoint = await page.evaluate(() =>
      localStorage.getItem('vidarax_endpoint')
    )
    const storedFps = await page.evaluate(() =>
      localStorage.getItem('vidarax_fps')
    )

    expect(storedEndpoint).toBe(CUSTOM_ENDPOINT)
    expect(storedFps).toBe('20')
  })
})
