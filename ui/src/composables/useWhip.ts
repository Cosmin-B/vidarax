/**
 * useWhip — WHIP WebRTC streaming composable.
 *
 * Handles the full lifecycle:
 *   1. Acquire local media (camera / screen)
 *   2. Create RTCPeerConnection + offer SDP
 *   3. POST offer to /v1/stream/whip (WHIP protocol)
 *   4. Set remote answer, resolve ICE
 *   5. Poll getStats() for live metrics
 *   6. Terminate session on stop
 *
 * Usage:
 *   const { localStream, startStream, stopStream } = useWhip()
 */

import { ref, onUnmounted } from 'vue'
import { api } from '@/lib/api'
import { ICE_SERVERS } from '@/lib/config'
import { useStreamStore } from '@/stores/stream'
import type { StreamSourceType } from '@/stores/stream'

/** Wait for ICE gathering to complete, with a 3 s timeout fallback. */
function waitForIceComplete(pc: RTCPeerConnection): Promise<string> {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') {
      resolve(pc.localDescription!.sdp)
      return
    }
    const check = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check)
        resolve(pc.localDescription!.sdp)
      }
    }
    pc.addEventListener('icegatheringstatechange', check)
    setTimeout(() => {
      pc.removeEventListener('icegatheringstatechange', check)
      resolve(pc.localDescription?.sdp ?? '')
    }, 3000)
  })
}

export interface WhipStartOptions {
  /** Optional analysis prompt forwarded to the VLM via attachToRun. */
  prompt?: string
}

export function useWhip() {
  const streamStore = useStreamStore()

  const localStream = ref<MediaStream | null>(null)
  const pc = ref<RTCPeerConnection | null>(null)
  let metricsTimer: ReturnType<typeof setInterval> | null = null
  let prevBytesSent = 0
  let prevFrames = 0
  let prevTs = 0

  /** Acquire local media based on source type. */
  async function acquireMedia(sourceType: StreamSourceType): Promise<MediaStream> {
    if (sourceType === 'screen') {
      return navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: true,
      })
    }
    if (sourceType === 'camera') {
      return navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: true,
      })
    }
    throw new Error(`Source type "${sourceType}" is not supported by useWhip`)
  }

  async function startStream(sourceType: StreamSourceType, options: WhipStartOptions = {}): Promise<void> {
    streamStore.setState('negotiating')
    streamStore.setError(null)

    try {
      // ── 1. Acquire media ──────────────────────────────────────────────
      const stream = await acquireMedia(sourceType)
      localStream.value = stream
      streamStore.setMediaStream(stream)

      // ── 2. Peer connection ────────────────────────────────────────────
      const conn = new RTCPeerConnection({ iceServers: ICE_SERVERS })
      pc.value = conn

      stream.getTracks().forEach(track => conn.addTrack(track, stream))

      // ── 3. Create offer ───────────────────────────────────────────────
      const offer = await conn.createOffer()
      await conn.setLocalDescription(offer)

      // Wait for ICE candidates to trickle in (or hit timeout)
      const offerSdp = await waitForIceComplete(conn)
      if (!offerSdp) throw new Error('Failed to build local SDP')

      // ── 4. WHIP negotiation ───────────────────────────────────────────
      let whipResult: { answer_sdp: string; session_id: string; location: string }
      try {
        whipResult = await api.stream.whipOffer(offerSdp)
      } catch (whipErr) {
        const msg = whipErr instanceof Error ? whipErr.message : 'WHIP negotiation failed'
        throw new Error(`WHIP negotiation failed: ${msg}`)
      }
      const { answer_sdp, session_id, location } = whipResult

      await conn.setRemoteDescription({ type: 'answer', sdp: answer_sdp })

      // ── 5. Store session ──────────────────────────────────────────────
      const runId = options.prompt !== undefined
        ? await _attachToRun(session_id, options.prompt)
        : ''

      streamStore.setSession({
        sessionId: session_id,
        runId,
        locationUrl: location,
        createdAt: new Date().toISOString(),
      })
      streamStore.setState('connected')

      // ── 6. Metrics polling ────────────────────────────────────────────
      prevTs = Date.now()
      metricsTimer = setInterval(() => pollMetrics(conn), 1000)

      // Handle peer disconnect from the remote side
      conn.onconnectionstatechange = () => {
        if (conn.connectionState === 'failed' || conn.connectionState === 'disconnected') {
          streamStore.setError('WebRTC connection lost')
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Stream failed'
      streamStore.setError(msg)
      cleanup()
    }
  }

  /** Best-effort: create a run and attach the WHIP session to it. */
  async function _attachToRun(sessionId: string, prompt: string): Promise<string> {
    try {
      const run = await api.runs.create({ source_uri: `whip://${sessionId}` })
      await api.stream.attachToRun(run.run_id, {
        session_id: sessionId,
        prompt: prompt || undefined,
      })
      return run.run_id
    } catch {
      // non-fatal — stream still works without a run ID
      return ''
    }
  }

  async function stopStream(): Promise<void> {
    const session = streamStore.activeSession
    if (session?.sessionId) {
      api.stream.whipTerminate(session.sessionId).catch(() => { /* best-effort */ })
    }
    cleanup()
    streamStore.reset()
  }

  async function pollMetrics(conn: RTCPeerConnection): Promise<void> {
    try {
      const stats = await conn.getStats()
      const now = Date.now()
      const elapsed = (now - prevTs) / 1000

      let bytesSent = 0
      let framesEncoded = 0

      stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.kind === 'video') {
          bytesSent += (report as RTCOutboundRtpStreamStats).bytesSent ?? 0
          framesEncoded = (report as RTCOutboundRtpStreamStats).framesEncoded ?? 0
        }
      })

      const fps = elapsed > 0 ? Math.round((framesEncoded - prevFrames) / elapsed) : 0

      streamStore.updateMetrics({
        bytesTransferred: bytesSent,
        frameCount: framesEncoded,
        fps: Math.max(0, fps),
        latencyMs: 0, // round-trip time available via candidate-pair stats if needed
      })

      prevBytesSent = bytesSent
      prevFrames = framesEncoded
      prevTs = now
    } catch {
      // stats can fail transiently
    }
  }

  function cleanup(): void {
    if (metricsTimer !== null) {
      clearInterval(metricsTimer)
      metricsTimer = null
    }
    if (pc.value) {
      pc.value.onconnectionstatechange = null
      pc.value.close()
      pc.value = null
    }
    if (localStream.value) {
      localStream.value.getTracks().forEach(t => t.stop())
      localStream.value = null
    }
    prevBytesSent = 0
    prevFrames = 0
    prevTs = 0
  }

  onUnmounted(cleanup)

  return { localStream, startStream, stopStream }
}
