import { test, expect, type Page } from '@playwright/test'
import { stopDurableWhipSession } from '../src/lib/whipLifecycle'

/**
 * Stream page feature E2E tests.
 * Verifies the source picker: 3 enabled tiles, no disabled or "Soon" tiles,
 * source selection, Start Stream button state, and stream viewer overlay state.
 *
 * WHIP connection tests inject browser-API fakes via addInitScript so the
 * WebRTC negotiation can proceed without a real GPU / camera / screen device.
 */

/** Inject fakes for getDisplayMedia and RTCPeerConnection into the page. */
async function injectWebRtcFakes(page: Page) {
  await page.addInitScript(() => {
    // Minimal fake MediaStream. No tracks are needed because the fake peer
    // connection accepts addTrack as a no-op.
    const fakeStream = {
      getTracks: () => [] as MediaStreamTrack[],
      getVideoTracks: () => [] as MediaStreamTrack[],
      getAudioTracks: () => [] as MediaStreamTrack[],
      addEventListener: () => {},
      removeEventListener: () => {},
    } as unknown as MediaStream

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        ...(navigator.mediaDevices ?? {}),
        getDisplayMedia: async () => fakeStream,
        getUserMedia: async () => fakeStream,
      },
    })

    const fakeSdp = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
    let framesEncoded = 0

    // ICE gathering is complete up front, so startStream does not wait for the
    // timeout fallback before posting the WHIP offer.
    ;(window as unknown as Record<string, unknown>).RTCPeerConnection = class {
      iceGatheringState = 'complete'
      connectionState: RTCPeerConnectionState = 'new'
      localDescription = { type: 'offer', sdp: fakeSdp } as RTCSessionDescriptionInit
      private connectionStateHandler: ((this: RTCPeerConnection, ev: Event) => unknown) | null = null

      constructor(_config?: RTCConfiguration) {}
      addTrack() {}
      async createOffer() { return this.localDescription }
      async setLocalDescription() {}
      async setRemoteDescription() {
        this.connectionState = 'connected'
        this.connectionStateHandler?.call(this as unknown as RTCPeerConnection, new Event('connectionstatechange'))
      }
      addEventListener() {}
      removeEventListener() {}
      set onconnectionstatechange(handler: ((this: RTCPeerConnection, ev: Event) => unknown) | null) {
        this.connectionStateHandler = handler
        if (handler && this.connectionState === 'connected') {
          setTimeout(() => {
            this.connectionStateHandler?.call(this as unknown as RTCPeerConnection, new Event('connectionstatechange'))
          }, 0)
        }
      }
      get onconnectionstatechange() {
        return this.connectionStateHandler
      }
      getStats() {
        framesEncoded += 30
        return Promise.resolve(new Map([
          ['outbound-video', {
            type: 'outbound-rtp',
            kind: 'video',
            bytesSent: framesEncoded * 1000,
            framesEncoded,
          }],
        ]))
      }
      close() {}
    }
  })
}

// The current picker renders Screen, Camera, and Live Cam as enabled buttons.
// Each tile's accessible name is the visible label.
function tileBtnOf(page: Page, label: string) {
  return page.getByRole('button', { name: label, exact: true })
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

  test('Source label is visible at /stream', async ({ page }) => {
    await page.goto('/stream')
    await expect(page.getByText('Source', { exact: true })).toBeVisible()
  })

  test('source picker is shown when no stream is active', async ({ page }) => {
    await page.goto('/stream')
    // Stream viewer header only appears when a stream is active
    await expect(page.getByRole('button', { name: /stop stream/i })).not.toBeVisible()
    await expect(page.getByText('Source', { exact: true })).toBeVisible()
  })

  // ── Source tiles ─────────────────────────────────────────────────────────────

  test('all 3 source tiles are rendered', async ({ page }) => {
    await page.goto('/stream')

    for (const label of ['Screen', 'Camera', 'Live Cam']) {
      await expect(tileBtnOf(page, label)).toBeVisible()
    }
  })

  test('source tiles are enabled', async ({ page }) => {
    await page.goto('/stream')

    for (const label of ['Screen', 'Camera', 'Live Cam']) {
      await expect(tileBtnOf(page, label)).toBeEnabled()
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

  test('"Start Stream" button enables after selecting a source', async ({ page }) => {
    await page.goto('/stream')

    const startButton = page.getByRole('button', { name: /start stream/i })
    await expect(startButton).toBeDisabled()

    await tileBtnOf(page, 'Screen').click()

    await expect(startButton).toBeEnabled()
  })

  test('"Start Stream" button is disabled before any source is selected', async ({ page }) => {
    await page.goto('/stream')
    await expect(page.getByRole('button', { name: /start stream/i })).toBeDisabled()
  })

  // ── Tile state ───────────────────────────────────────────────────────────────

  test('clicking Live Cam marks it selected', async ({ page }) => {
    await page.goto('/stream')

    const liveCamTile = tileBtnOf(page, 'Live Cam')
    await liveCamTile.click()
    await expect(liveCamTile).toHaveAttribute('aria-pressed', 'true')
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

  test('clicking Screen then Start Stream shows WHIP negotiation UI', async ({ page }) => {
    await injectWebRtcFakes(page)

    // Make the WHIP offer request hang so the page stays in "negotiating" state
    await page.route('**/v1/stream/whip', () => {
      // intentionally never fulfilled — keeps sessionState = 'negotiating'
    })

    await page.goto('/stream')
    await tileBtnOf(page, 'Screen').click()
    await page.getByRole('button', { name: /start stream/i }).click()

    await expect(page.getByText(/connecting/i)).toBeVisible({ timeout: 10_000 })
    // Spinner overlay shows the WHIP negotiation message
    await expect(page.getByText(/negotiating whip session/i)).toBeVisible({ timeout: 10_000 })
  })

  test('stream viewer shows fps badge when stream is active', async ({ page }) => {
    await injectWebRtcFakes(page)

    await page.route('**/v1/stream/whip', async route => {
      await route.fulfill({
        status: 201,
        headers: {
          'Content-Type': 'application/sdp',
          Location: '/v1/stream/whip/test-session-001',
        },
        body: 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n',
      })
    })

    await page.goto('/stream')
    await tileBtnOf(page, 'Screen').click()
    await page.getByRole('button', { name: /start stream/i }).click()

    await expect(page.locator('.sp-hud-badge').filter({ hasText: /fps/ })).toBeVisible({ timeout: 10_000 })
  })

  test('stops the durable run before terminating its WHIP transport', async () => {
    const lifecycle: string[] = []
    await stopDurableWhipSession(
      { runId: 'run-stream-history', sessionId: 'test-session-history' },
      async () => { lifecycle.push('stop-run') },
      async () => { lifecycle.push('terminate-whip') },
    )
    expect(lifecycle).toEqual(['stop-run', 'terminate-whip'])
  })
})
