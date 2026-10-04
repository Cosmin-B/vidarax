import { expect, test } from '@playwright/test'

test('timeline parse and callback failures release their reader', async ({ page }) => {
  await page.goto('/')
  const results = await page.evaluate(async () => {
    const { followRunEvents } = await import('/src/lib/api.ts')
    const originalFetch = window.fetch
    const results = []
    try {
      for (const invalidJson of [true, false]) {
        let cancelled = 0
        const body = new ReadableStream<Uint8Array>({
          start(controller) {
            const data = invalidJson ? '{' : '{"sequence":1,"pts_ms":0,"data":{}}'
            controller.enqueue(new TextEncoder().encode(`event: gate\ndata: ${data}\n\n`))
          },
          cancel() { cancelled++ },
        })
        window.fetch = async () => new Response(body)
        let failed = false
        try {
          await followRunEvents('run-1', 0, new AbortController().signal, () => {}, () => {
            throw new Error('consumer failed')
          })
        } catch { failed = true }
        results.push({ failed, cancelled, locked: body.locked })
      }
    } finally { window.fetch = originalFetch }
    return results
  })
  expect(results).toEqual([
    { failed: true, cancelled: 1, locked: false },
    { failed: true, cancelled: 1, locked: false },
  ])
})

test('replacing a subscription to the same run releases old evidence URLs', async ({ page }) => {
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const { api } = await import('/src/lib/api.ts')
    const { useEventStream } = await import('/src/composables/useEventStream.ts')
    const { useEventsStore } = await import('/src/stores/events.ts')
    const store = useEventsStore()
    store.clearAll()
    const originalFetch = window.fetch
    const originalKeyframe = api.runs.keyframe
    const originalCreate = URL.createObjectURL
    const originalRevoke = URL.revokeObjectURL
    const revoked: string[] = []
    let resolveBlob!: (blob: Blob) => void
    let evidenceSignal: AbortSignal | undefined
    let requested!: () => void
    const evidenceRequested = new Promise<void>(resolve => { requested = resolve })
    let requests = 0
    window.fetch = async (_url, init) => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          if (requests++ === 0) {
            const data = JSON.stringify({ sequence: 1, pts_ms: 0, data: { image_sha256: 'a'.repeat(64) } })
            controller.enqueue(new TextEncoder().encode(`event: keyframe_stored\ndata: ${data}\n\n`))
          }
          init?.signal?.addEventListener('abort', () => controller.close(), { once: true })
        },
      })
      return new Response(body)
    }
    api.runs.keyframe = (_runId, _sha, signal) => {
      evidenceSignal = signal
      requested()
      return new Promise(resolve => { resolveBlob = resolve })
    }
    URL.createObjectURL = () => 'blob:old-generation'
    URL.revokeObjectURL = url => { revoked.push(url) }
    const stream = useEventStream()
    try {
      await stream.connect('run-1')
      await evidenceRequested
      await stream.connect('run-1')
      resolveBlob(new Blob(['jpeg']))
      await new Promise(resolve => setTimeout(resolve, 0))
      return { total: store.keyframes.length, revoked, aborted: evidenceSignal?.aborted }
    } finally {
      stream.disconnect()
      window.fetch = originalFetch
      api.runs.keyframe = originalKeyframe
      URL.createObjectURL = originalCreate
      URL.revokeObjectURL = originalRevoke
    }
  })
  expect(result).toEqual({ total: 0, revoked: ['blob:old-generation'], aborted: true })
})

test('metrics polling owns one request and ignores stopped requests', async ({ page }) => {
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const { useMetrics } = await import('/src/composables/useMetrics.ts')
    const originalFetch = window.fetch
    const responses: ((response: Response) => void)[] = []
    const signals: AbortSignal[] = []
    window.fetch = (_url, init) => {
      signals.push(init!.signal!)
      return new Promise(resolve => { responses.push(resolve) })
    }
    const metrics = useMetrics(10)
    try {
      metrics.start()
      metrics.start()
      await new Promise(resolve => setTimeout(resolve, 25))
      const requestsWhilePending = responses.length
      metrics.stop()
      const aborted = signals[0]?.aborted
      metrics.start()
      responses[0]!(new Response('vidarax_pipeline_sessions_created_total 99'))
      await new Promise(resolve => setTimeout(resolve, 20))
      const staleIgnored = metrics.metrics.value === null
      responses[1]!(new Response('vidarax_pipeline_sessions_created_total 2'))
      while (!metrics.metrics.value && !metrics.error.value) {
        await new Promise(resolve => setTimeout(resolve, 0))
      }
      if (metrics.error.value) throw new Error(metrics.error.value)
      return { requestsWhilePending, aborted, staleIgnored, activeSessions: metrics.metrics.value?.activeSessions }
    } finally {
      metrics.stop()
      window.fetch = originalFetch
    }
  })
  expect(result).toEqual({ requestsWhilePending: 1, aborted: true, staleIgnored: true, activeSessions: 2 })
})

for (const scenario of ['late-media', 'late-offer', 'failed-answer', 'connected-stop'] as const) {
  test(`WHIP releases resources after ${scenario}`, async ({ page }) => {
    await page.goto('/')
    const result = await page.evaluate(async scenario => {
      const { api } = await import('/src/lib/api.ts')
      const { useWhip } = await import('/src/composables/useWhip.ts')
      const { useStreamStore } = await import('/src/stores/stream.ts')
      const originalMedia = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')
      const originalPeer = window.RTCPeerConnection
      const originalOffer = api.stream.whipOffer
      const originalStop = api.runs.stop
      const originalTerminate = api.stream.whipTerminate
      let tracksStopped = 0
      let peersClosed = 0
      const cleanup: string[] = []
      const media = { getTracks: () => [{ stop() { tracksStopped++ } }] } as unknown as MediaStream
      let resolveMedia!: (media: MediaStream) => void
      let resolveOffer!: (session: Awaited<ReturnType<typeof api.stream.whipOffer>>) => void
      let offerRequested!: () => void
      const requested = new Promise<void>(resolve => { offerRequested = resolve })
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: { getUserMedia: () => scenario === 'late-media'
          ? new Promise(resolve => { resolveMedia = resolve })
          : Promise.resolve(media) },
      })
      class Peer extends EventTarget {
        iceGatheringState = 'complete'
        localDescription: RTCSessionDescriptionInit | null = null
        onconnectionstatechange = null
        addTrack() {}
        async createOffer() { return { type: 'offer', sdp: 'v=0' } }
        async setLocalDescription(description: RTCSessionDescriptionInit) { this.localDescription = description }
        async setRemoteDescription() {
          if (scenario === 'failed-answer') throw new Error('invalid remote answer')
        }
        close() { peersClosed++; cleanup.push('peer') }
      }
      window.RTCPeerConnection = Peer as unknown as typeof RTCPeerConnection
      const response = { answer_sdp: 'v=0', session_id: 'session-1', location: '/v1/stream/whip/session-1', run_id: 'run-1' }
      api.stream.whipOffer = () => {
        offerRequested()
        return scenario === 'late-offer'
          ? new Promise(resolve => { resolveOffer = resolve })
          : Promise.resolve(response)
      }
      api.runs.stop = async run => { cleanup.push(`run:${run}`) }
      api.stream.whipTerminate = async session => { cleanup.push(`session:${session}`) }
      const whip = useWhip()
      try {
        const started = whip.startStream('camera')
        if (scenario === 'late-media') {
          await whip.stopStream()
          resolveMedia(media)
        } else if (scenario === 'late-offer') {
          await requested
          await whip.stopStream()
          resolveOffer(response)
        }
        await started
        if (scenario === 'connected-stop') await whip.stopStream()
        return {
          tracksStopped, peersClosed, cleanup,
          localStream: whip.localStream.value,
          session: useStreamStore().activeSession,
        }
      } finally {
        await whip.stopStream()
        if (originalMedia) Object.defineProperty(navigator, 'mediaDevices', originalMedia)
        else delete (navigator as unknown as Record<string, unknown>).mediaDevices
        window.RTCPeerConnection = originalPeer
        api.stream.whipOffer = originalOffer
        api.runs.stop = originalStop
        api.stream.whipTerminate = originalTerminate
      }
    }, scenario)
    expect(result.tracksStopped).toBe(1)
    expect(result.peersClosed).toBe(scenario === 'late-media' ? 0 : 1)
    expect(result.cleanup).toEqual(scenario === 'late-media' ? [] : ['run:run-1', 'peer', 'session:session-1'])
    expect(result.localStream).toBeNull()
    expect(result.session).toBeNull()
  })
}
