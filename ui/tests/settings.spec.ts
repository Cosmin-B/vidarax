import { test, expect } from '@playwright/test'

/**
 * Settings E2E tests
 * Verifies all settings inputs render, values persist after save + reload.
 */

const DEFAULT_ENDPOINT = 'http://localhost:8080'
const CUSTOM_ENDPOINT = 'http://vidarax.example.com:9090'
const CUSTOM_STDB = 'http://stdb.example.com:3001'
const MODEL_2B = 'Qwen/Qwen3-VL-2B-Instruct'
const MODEL_FETCHED_ONLY = 'Qwen/Qwen3-VL-30B-Instruct'
const MODEL_CATALOG = {
  models: [
    { id: 'Qwen/Qwen3-VL-8B-Instruct', name: 'Qwen3-VL 8B', availability: 'available' },
    { id: 'Qwen/Qwen3-VL-4B-Instruct', name: 'Qwen3-VL 4B', availability: 'available' },
    { id: MODEL_2B, name: 'Qwen3-VL 2B', availability: 'available' },
    // This fetch-only entry is absent from fallback defaults so the mock must resolve.
    { id: MODEL_FETCHED_ONLY, name: 'Qwen3-VL 30B', availability: 'available' },
    { id: 'OpenGVLab/InternVL3_5-4B', name: 'InternVL3.5 4B', availability: 'available' },
    { id: 'LiquidAI/LFM2.5-VL-1.6B', name: 'LFM2.5-VL 1.6B', availability: 'available' },
  ],
}

test.describe('Settings panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/v1/models', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(MODEL_CATALOG),
      })
    })

    // Start with clean localStorage
    await page.goto('/settings')
    await page.evaluate(() => localStorage.clear())
    await page.reload()
  })

  // ── Page structure ───────────────────────────────────────────────────────

  test('all accordion sections are visible', async ({ page }) => {
    await page.goto('/settings')
    // Use exact names to avoid ambiguity with "Test Connection" button
    await expect(page.getByRole('button', { name: 'Connection', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Models', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Stream', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Per-frame filter', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Advanced', exact: true })).toBeVisible()
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
    await defaultModelSelect.selectOption(MODEL_FETCHED_ONLY)
    await expect(defaultModelSelect).toHaveValue(MODEL_FETCHED_ONLY)

    await page.getByTestId('save-button').click()
    await page.reload()

    // Reopen models section
    await page.getByRole('button', { name: /models/i }).click()
    await expect(page.getByTestId('default-model')).toHaveValue(MODEL_FETCHED_ONLY)
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

  // ── Per-frame filter section ─────────────────────────────────────────────

  test('per-frame filter section: sliders are present', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /per-frame filter/i }).click()

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

  // ── TURN server field ─────────────────────────────────────────────────────

  test('connection section: TURN server URL field is present', async ({ page }) => {
    await page.goto('/settings')
    // Connection section is open by default
    await expect(page.getByText(
      'Supports STUN or credential-less TURN URLs only; configure credentialed TURN relays server-side.',
      { exact: true },
    )).toBeVisible()
    await expect(page.getByTestId('turn-url')).toBeVisible()
  })

  test('TURN server URL field accepts input and persists after save', async ({ page }) => {
    const TURN_URL = 'turn:turn.example.com:3478?transport=tcp'
    await page.goto('/settings')

    await page.getByTestId('turn-url').fill(TURN_URL)
    await page.getByTestId('save-button').click()
    await expect(page.getByTestId('save-button')).toContainText('Saved')

    await page.reload()
    await expect(page.getByTestId('turn-url')).toHaveValue(TURN_URL)
  })

  // ── Token rate cap slider ─────────────────────────────────────────────────

  test('stream section: token rate cap is shown as server-managed', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()
    await expect(page.getByTestId('token-rate-cap')).toBeVisible()
    await expect(page.getByTestId('token-rate-cap')).toBeDisabled()
    await expect(page.getByText(/configured server-side/i)).toBeVisible()
  })

  test('server-managed controls do not persist inert localStorage values', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()
    await expect(page.getByTestId('token-rate-cap')).toBeDisabled()
    await expect(page.getByTestId('clip-mode-toggle')).toBeDisabled()

    await page.getByRole('button', { name: /per-frame filter/i }).click()
    await expect(page.getByTestId('scene-cut-hamming-threshold')).toBeDisabled()
    await expect(page.getByTestId('luma-shift-threshold')).toBeDisabled()
    await expect(page.getByTestId('loop-detection-toggle')).toBeDisabled()
    await expect(page.getByTestId('loop-repeat-trigger')).toBeDisabled()

    await page.getByRole('button', { name: /advanced/i }).click()
    await expect(page.getByTestId('gpu-decode')).toBeDisabled()

    await page.getByTestId('save-button').click()

    const inertValues = await page.evaluate(() => ({
      tokenRateCap: localStorage.getItem('vidarax_token_rate_cap'),
      clipMode: localStorage.getItem('vidarax_clip_mode'),
      sceneCutHammingThreshold: localStorage.getItem('vidarax_scene_cut_hamming_threshold'),
      lumaShiftThreshold: localStorage.getItem('vidarax_luma_shift_threshold'),
      loopDetection: localStorage.getItem('vidarax_loop_detection'),
      loopRepeatTrigger: localStorage.getItem('vidarax_loop_repeat_trigger'),
      gpuDecode: localStorage.getItem('vidarax_gpu_decode'),
    }))

    expect(inertValues).toEqual({
      tokenRateCap: null,
      clipMode: null,
      sceneCutHammingThreshold: null,
      lumaShiftThreshold: null,
      loopDetection: null,
      loopRepeatTrigger: null,
      gpuDecode: null,
    })
  })

  // ── Clip mode toggle ──────────────────────────────────────────────────────

  test('stream section: clip mode is shown as non-actionable', async ({ page }) => {
    await page.goto('/settings')
    await page.getByRole('button', { name: /^stream$/i }).click()
    await expect(page.getByTestId('clip-mode-toggle')).toBeVisible()
    await expect(page.getByTestId('clip-mode-toggle')).toBeDisabled()
    await expect(page.getByText(
      'This page does not send clip_mode; streaming clip mode is applied through the WHIP attach path, not from this page.',
      { exact: true },
    )).toBeVisible()
  })
})
