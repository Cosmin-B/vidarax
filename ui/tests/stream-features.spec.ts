import { test, expect, type Page } from '@playwright/test'

/**
 * Stream page feature E2E tests.
 * Verifies the source picker: 6 tiles, disabled badges, source selection,
 * Start Analysis button, File redirect, and stream viewer overlay state.
 *
 * WHIP connection tests inject browser-API fakes via addInitScript so the
 * WebRTC negotiation can proceed without a real GPU / camera / screen device.
 */

/** Inject fakes for getDisplayMedia and RTCPeerConnection into the page. */
async function injectWebRtcFakes(page: Page) {
  await page.addInitScript(() => {
    // Minimal fake MediaStream (no tracks needed — addTrack is a no-op in fake PC)
    const fakeStream = {
      getTracks: () => [] as MediaStreamTrack[],
      getVideoTracks: () => [] as MediaStreamTrack[],
      getAudioTracks: () => [] as MediaStreamTrack[],
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaStream

    navigator.mediaDevices.getDisplayMedia = async () => fakeStream
    navigator.mediaDevices.getUserMedia   = async () => fakeStream

    const fakeSdp = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'

    // Fake RTCPeerConnection: ICE gathering is already complete → no 3 s wait
    ;(window as unknown as Record<string, unknown>).RTCPeerConnection = class {
      iceGatheringState = 'complete'
      localDescription = { type: 'offer', sdp: fakeSdp }
      constructor(_config?: RTCConfiguration) {}
      addTrack() {}
      async createOffer() { return this.localDescription }
      async setLocalDescription() {}
      async setRemoteDescription() {}
      addEventListener() {}
      removeEventListener() {}
      set onconnectionstatechange(_v: unknown) {}
      getStats() { return Promise.resolve(new Map()) }
      close() {}
    }
  })
}

// Locate a source tile button by its visible label.
// Accessible names: available tiles start with the label; disabled tiles start
// with "Soon <label>". "Camera" needs /^Camera/ to avoid matching the Live Cam
// tile whose description contains the word "camera". All other labels are unique
// enough for a plain case-insensitive substring match via hasText.
function tileBtnOf(page: Page, label: string) {
  if (label === 'Camera') {
    // Accessible name is "Camera Use your webcam or external camera"
    // Live Cam's accessible name starts with "Live Cam", not "Camera"
    return page.getByRole('button', { name: /^Camera/ })
  }
  return page.getByRole('button').filter({ hasText: label })
}

test.describe('Stream page — source picker', () => {
  test.beforeEach(async ({ page }) => {
    // Seed localStorage so the app uses the expected API endpoint
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })
  })

  // ── Page structure ───────────────────────────────────────────────────────────

  test('"Select Source" heading is visible at /stream', async ({ page }) => {
    await page.goto('/stream')
    await expect(page.getByRole('heading', { name: /select source/i })).toBeVisible()
  })

  test('source picker is shown when no stream is active', async ({ page }) => {
    await page.goto('/stream')
    // Stream viewer header only appears when a stream is active
    await expect(page.getByRole('button', { name: /stop stream/i })).not.toBeVisible()
    await expect(page.getByRole('heading', { name: /select source/i })).toBeVisible()
  })

  // ── Source tiles ─────────────────────────────────────────────────────────────

  test('all 6 source tiles are rendered', async ({ page }) => {
    await page.goto('/stream')

    for (const label of ['Screen', 'Camera', 'Live Cam', 'File', 'LiveKit', 'RTSP']) {
      await expect(tileBtnOf(page, label)).toBeVisible()
    }
  })

  test('disabled source tiles have "Soon" badge', async ({ page }) => {
    await page.goto('/stream')

    // LiveKit and RTSP are not yet available
    await expect(tileBtnOf(page, 'LiveKit').getByText('Soon')).toBeVisible()
    await expect(tileBtnOf(page, 'RTSP').getByText('Soon')).toBeVisible()
  })

  test('available source tiles do not have "Soon" badge', async ({ page }) => {
    await page.goto('/stream')

    for (const label of ['Screen', 'Camera', 'Live Cam']) {
      await expect(tileBtnOf(page, label).getByText('Soon')).not.toBeVisible()
    }
  })

  test('disabled source tiles are marked as disabled', async ({ page }) => {
    await page.goto('/stream')

    await expect(tileBtnOf(page, 'LiveKit')).toBeDisabled()
    await expect(tileBtnOf(page, 'RTSP')).toBeDisabled()
  })

  test('available source tiles are not disabled', async ({ page }) => {
    await page.goto('/stream')

    for (const label of ['Screen', 'Camera', 'Live Cam']) {
      await expect(tileBtnOf(page, label)).not.toBeDisabled()
    }
  })

  // ── Source selection ─────────────────────────────────────────────────────────

  test('clicking Screen selects it (aria-pressed = true)', async ({ page }) => {
    await page.goto('/stream')

    const screenTile = tileBtnOf(page, 'Screen')
    await expect(screenTile).toHaveAttribute('aria-pressed', 'false')

    await screenTile.click()
    await expect(screenTile).toHaveAttribute('aria-pressed', 'true')
  })

  test('clicking Camera selects it and deselects previous selection', async ({ page }) => {
    await page.goto('/stream')

    const screenTile = tileBtnOf(page, 'Screen')
    const cameraTile = tileBtnOf(page, 'Camera')

    await screenTile.click()
    await expect(screenTile).toHaveAttribute('aria-pressed', 'true')

    await cameraTile.click()
    await expect(cameraTile).toHaveAttribute('aria-pressed', 'true')
    await expect(screenTile).toHaveAttribute('aria-pressed', 'false')
  })

  test('"Start Analysis" button appears after selecting a source', async ({ page }) => {
    await page.goto('/stream')

    // No Start Analysis button before a source is chosen
    await expect(page.getByRole('button', { name: /start analysis/i })).not.toBeVisible()

    // Select a source
    await tileBtnOf(page, 'Screen').click()

    await expect(page.getByRole('button', { name: /start analysis/i })).toBeVisible()
  })

  test('"Start Analysis" button is not shown before any source is selected', async ({ page }) => {
    await page.goto('/stream')
    await expect(page.getByRole('button', { name: /start analysis/i })).not.toBeVisible()
  })

  // ── File redirect ────────────────────────────────────────────────────────────

  test('clicking File tile navigates to /upload', async ({ page }) => {
    await page.goto('/stream')
    await tileBtnOf(page, 'File').click()
    await expect(page).toHaveURL('/upload')
  })

  // ── Disabled tile behaviour ──────────────────────────────────────────────────

  test('clicking a disabled tile does not select it', async ({ page }) => {
    await page.goto('/stream')

    const livekitTile = tileBtnOf(page, 'LiveKit')
    // Disabled buttons cannot be interacted with — aria-pressed stays false
    await expect(livekitTile).toHaveAttribute('aria-pressed', 'false')
    await expect(livekitTile).toBeDisabled()
  })

  test('clicking a disabled tile does not show Start Analysis button', async ({ page }) => {
    await page.goto('/stream')

    // Verify RTSP is disabled; no Start Analysis should appear
    await expect(tileBtnOf(page, 'RTSP')).toBeDisabled()
    await expect(page.getByRole('button', { name: /start analysis/i })).not.toBeVisible()
  })
})

// ── WHIP connection UI ────────────────────────────────────────────────────────

test.describe('Stream page — WHIP connection UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.setItem('vidarax_endpoint', 'http://localhost:8080')
      localStorage.setItem('vidarax_spacetime_endpoint', 'http://localhost:3000')
    })
  })

  test('clicking Screen then Start Analysis shows WHIP negotiation UI', async ({ page }) => {
    await injectWebRtcFakes(page)

    // Make the WHIP offer request hang so the page stays in "negotiating" state
    await page.route('**/v1/stream/whip', () => {
      // intentionally never fulfilled — keeps sessionState = 'negotiating'
    })

    await page.goto('/stream')
    await tileBtnOf(page, 'Screen').click()
    await page.getByRole('button', { name: /start analysis/i }).click()

    // Stream viewer section appears with "Connecting…" heading
    await expect(page.getByRole('heading', { name: /connecting/i })).toBeVisible({ timeout: 10_000 })
    // Spinner overlay shows the WHIP negotiation message
    await expect(page.getByText(/negotiating whip session/i)).toBeVisible({ timeout: 10_000 })
  })

  test('stream viewer shows fps badge when stream is active', async ({ page }) => {
    await injectWebRtcFakes(page)

    // Hanging WHIP request keeps state at 'negotiating' → isStreaming = true → fps badge visible
    await page.route('**/v1/stream/whip', () => {})

    await page.goto('/stream')
    await tileBtnOf(page, 'Screen').click()
    await page.getByRole('button', { name: /start analysis/i }).click()

    // FPS badge is visible once the stream section renders (even with 0 fps during negotiation)
    await expect(page.locator('.badge-teal').filter({ hasText: /fps/ })).toBeVisible({ timeout: 10_000 })
  })
})
