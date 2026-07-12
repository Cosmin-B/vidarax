import { test, expect } from '@playwright/test'

/**
 * Auth secret storage E2E tests.
 *
 * The API key and SpacetimeDB token are bearer secrets and must live in
 * sessionStorage (per-tab, cleared on close), never in localStorage (durable,
 * on disk). Endpoints stay in localStorage. These tests pin that split and the
 * one-time migration that moves a legacy localStorage secret across on upgrade.
 */

const API_KEY_STORAGE = 'vidarax_api_key'
const TOKEN_STORAGE = 'spacetime_token'
const ENDPOINT_STORAGE = 'vidarax_endpoint'

test.describe('Auth secret storage', () => {
  test('saving the API key writes it to sessionStorage, not localStorage', async ({ page }) => {
    await page.goto('/settings')

    const apiKeyInput = page.getByTestId('api-key')
    await expect(apiKeyInput).toBeVisible()
    await apiKeyInput.fill('secret-key-abc123')

    await page.getByTestId('save-button').click()
    await expect(page.getByTestId('save-button')).toContainText('Saved')

    const stored = await page.evaluate(
      (keys) => ({
        session: sessionStorage.getItem(keys.api),
        local: localStorage.getItem(keys.api),
      }),
      { api: API_KEY_STORAGE },
    )
    expect(stored.session).toBe('secret-key-abc123')
    expect(stored.local).toBeNull()
  })

  test('the API key survives a same-tab reload', async ({ page }) => {
    await page.goto('/settings')
    await page.getByTestId('api-key').fill('persist-across-reload')
    await page.getByTestId('save-button').click()
    await expect(page.getByTestId('save-button')).toContainText('Saved')

    await page.reload()

    const stored = await page.evaluate(
      (key) => sessionStorage.getItem(key),
      API_KEY_STORAGE,
    )
    expect(stored).toBe('persist-across-reload')
    await expect(page.getByTestId('api-key')).toHaveValue('persist-across-reload')
  })

  test('legacy localStorage secrets are migrated to sessionStorage and removed on load', async ({ page }) => {
    // Simulate an install from before the switch: both secrets sit in
    // localStorage and the endpoint alongside them.
    await page.goto('/')
    await page.evaluate(
      (keys) => {
        localStorage.setItem(keys.api, 'legacy-durable-key')
        localStorage.setItem(keys.token, 'legacy-durable-token')
        localStorage.setItem(keys.endpoint, 'http://localhost:8080')
        sessionStorage.clear()
      },
      { api: API_KEY_STORAGE, token: TOKEN_STORAGE, endpoint: ENDPOINT_STORAGE },
    )

    // Reload so the store re-initializes and runs the migration.
    await page.reload()

    // Both secrets should move to sessionStorage and be cleared from localStorage.
    await expect
      .poll(() => page.evaluate((key) => sessionStorage.getItem(key), API_KEY_STORAGE))
      .toBe('legacy-durable-key')
    const after = await page.evaluate(
      (keys) => ({
        sessionToken: sessionStorage.getItem(keys.token),
        localApi: localStorage.getItem(keys.api),
        localToken: localStorage.getItem(keys.token),
        localEndpoint: localStorage.getItem(keys.endpoint),
      }),
      { api: API_KEY_STORAGE, token: TOKEN_STORAGE, endpoint: ENDPOINT_STORAGE },
    )
    expect(after.sessionToken).toBe('legacy-durable-token')
    expect(after.localApi).toBeNull()
    expect(after.localToken).toBeNull()
    // The endpoint is not a secret and stays in localStorage.
    expect(after.localEndpoint).toBe('http://localhost:8080')
  })

  test('an active session wins over a stale localStorage copy', async ({ page }) => {
    // Both stores hold a value: the per-tab session must win and the durable
    // copy must be discarded, not resurrected over current-tab state.
    await page.goto('/')
    await page.evaluate(
      (keys) => {
        localStorage.setItem(keys.api, 'stale-durable')
        sessionStorage.setItem(keys.api, 'active-session')
      },
      { api: API_KEY_STORAGE },
    )

    await page.reload()

    const after = await page.evaluate(
      (key) => ({
        session: sessionStorage.getItem(key),
        local: localStorage.getItem(key),
      }),
      API_KEY_STORAGE,
    )
    expect(after.session).toBe('active-session')
    expect(after.local).toBeNull()
  })
})
