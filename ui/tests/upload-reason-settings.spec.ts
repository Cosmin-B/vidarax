import { test, expect } from '@playwright/test'

test.describe('Upload reason settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.evaluate(() => localStorage.clear())
  })

  test('fresh install sends displayed reason defaults in the upload reason POST body', async ({ page }) => {
    let reasonBody: Record<string, unknown> | null = null

    await page.route('http://localhost:8080/v1/upload', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ file_path: '/tmp/sample.mp4' }),
      })
    })
    await page.route('http://localhost:8080/v1/runs', async route => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-default-settings',
          status: 'created',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-default-settings/reason', async route => {
      reasonBody = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-default-settings',
          status: 'completed',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-default-settings/events', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ request_id: 'req-events', run_id: 'run-default-settings', events: [] }),
      })
    })

    await page.goto('/upload')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
    })
    await page.reload()

    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from('fake video'),
    })
    await page.getByRole('button', { name: /start analysis/i }).click()

    await expect.poll(() => reasonBody).not.toBeNull()
    expect(reasonBody).toMatchObject({
      first_pass_model: 'Qwen/Qwen3-VL-2B-Instruct',
      second_pass_model: 'Qwen/Qwen3-VL-4B-Instruct',
      semantic_frames_per_chunk: 4,
    })
  })

  test('saved reason settings are sent in the upload reason POST body', async ({ page }) => {
    let reasonBody: Record<string, unknown> | null = null

    await page.route('http://localhost:8080/v1/upload', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ file_path: '/tmp/sample.mp4' }),
      })
    })
    await page.route('http://localhost:8080/v1/runs', async route => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-settings',
          status: 'created',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-settings/reason', async route => {
      reasonBody = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-settings',
          status: 'completed',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-settings/events', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ request_id: 'req-events', run_id: 'run-settings', events: [] }),
      })
    })

    await page.goto('/upload')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_default_model', 'Qwen/Qwen3-VL-4B-Instruct')
      localStorage.setItem('vidarax_first_pass_model', 'Qwen/Qwen3-VL-2B-Instruct')
      localStorage.setItem('vidarax_second_pass_model', 'OpenGVLab/InternVL3_5-4B')
      localStorage.setItem('vidarax_semantic_frames_per_chunk', '3')
    })
    await page.reload()

    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from('fake video'),
    })
    await page.getByRole('button', { name: /start analysis/i }).click()

    await expect.poll(() => reasonBody).not.toBeNull()
    expect(reasonBody).toMatchObject({
      first_pass_model: 'Qwen/Qwen3-VL-2B-Instruct',
      second_pass_model: 'OpenGVLab/InternVL3_5-4B',
      semantic_frames_per_chunk: 3,
    })
    expect(reasonBody).not.toHaveProperty('clip_mode')
  })

  test('blank semantic frames per chunk is omitted from the upload reason POST body', async ({ page }) => {
    let reasonBody: Record<string, unknown> | null = null

    await page.route('http://localhost:8080/v1/upload', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ file_path: '/tmp/sample.mp4' }),
      })
    })
    await page.route('http://localhost:8080/v1/runs', async route => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-blank-settings',
          status: 'created',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-blank-settings/reason', async route => {
      reasonBody = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-blank-settings',
          status: 'completed',
          mode: 'batch',
          model: 'Qwen/Qwen3-VL-4B-Instruct',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-blank-settings/events', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ request_id: 'req-events', run_id: 'run-blank-settings', events: [] }),
      })
    })

    await page.goto('/upload')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_default_model', 'Qwen/Qwen3-VL-4B-Instruct')
      localStorage.setItem('vidarax_semantic_frames_per_chunk', '   ')
    })
    await page.reload()

    await page.locator('input[type="file"]').setInputFiles({
      name: 'sample.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from('fake video'),
    })
    await page.getByRole('button', { name: /start analysis/i }).click()

    await expect.poll(() => reasonBody).not.toBeNull()
    expect(reasonBody).not.toHaveProperty('semantic_frames_per_chunk')
  })
})
