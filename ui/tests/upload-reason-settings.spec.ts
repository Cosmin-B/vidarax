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

  test('audio and video mode sends native media options and renders timestamped moments', async ({ page }) => {
    let reasonBody: Record<string, unknown> | null = null
    const hash = 'a'.repeat(64)

    await page.route('http://localhost:8080/v1/models', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: [{ id: 'gemini-3.5-flash-lite', name: 'Gemini 3.5 Flash-Lite', tier: 'cloud' }],
        }),
      })
    })
    await page.route('http://localhost:8080/v1/upload', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ file_path: '/tmp/session.mp4' }),
      })
    })
    await page.route('http://localhost:8080/v1/runs', async route => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          run_id: 'run-audio-video',
          status: 'created',
          mode: 'balanced',
          model: 'gemini-3.5-flash-lite',
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-audio-video/reason', async route => {
      reasonBody = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          request_id: 'req-av',
          run_id: 'run-audio-video',
          generated: 1,
          markers_emitted: 0,
          decoded_frames: 80,
          sample_fps: 10,
          lag_p95_ms: 0,
          lag_p99_ms: 0,
          tokens: {},
          metadata: [],
          markers: [],
        }),
      })
    })
    await page.route('http://localhost:8080/v1/runs/run-audio-video/events', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          request_id: 'req-events',
          run_id: 'run-audio-video',
          events: [{
            seq: 4,
            pts_ms: 2_000,
            kind: 'multimodal_moment',
            payload: {
              moment_id: 'req-av:0:0',
              start_pts_ms: 2_000,
              end_pts_ms: 3_000,
              modalities: ['audio', 'video'],
              kind: 'interaction',
              description: 'The click sound coincides with the visible button press.',
              intent: 'The speaker is trying to run the build.',
              audio_visual_relation: 'Sound and press occur together.',
              evidence: { media_sha256: hash },
            },
          }],
        }),
      })
    })

    await page.goto('/upload')
    await page.evaluate(() => localStorage.setItem('vidarax_endpoint', 'http://localhost:8080'))
    await page.reload()
    await page.locator('input[type="file"]').setInputFiles({
      name: 'session.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from('fake media'),
    })
    await page.getByRole('radio', { name: 'Audio + video' }).click()
    await page.getByRole('switch', { name: 'Local audio events and selective speech' }).click()
    await page.getByLabel('Audio profile').selectOption('screen_recording')
    await page.getByLabel('Speech engine').selectOption('moonshine')
    await page.getByRole('switch', { name: 'Store spoken feedback as WAV evidence' }).click()
    await page.getByRole('button', { name: /start analysis/i }).click()

    await expect.poll(() => reasonBody).not.toBeNull()
    expect(reasonBody).toMatchObject({
      model: 'gemini-3.5-flash-lite',
      semantic_timeout_ms: 30_000,
      media: {
        mode: 'audio_video',
        window_ms: 8_000,
        resolution: 'low',
        persist_evidence: true,
      },
      local_audio: {
        profile: 'screen_recording',
        speech_engine: 'moonshine',
        min_confidence: 0.35,
        max_events: 32,
        voice_feedback: true,
      },
    })
    expect(reasonBody).not.toHaveProperty('chunk_size')
    await expect(page.getByTestId('multimodal-moment')).toContainText(
      'The click sound coincides with the visible button press.',
    )
    await expect(page.getByRole('button', { name: 'Download clip' })).toBeVisible()
  })
})
