/**
 * useMetrics.ts
 * Polls GET /v1/metrics every 2s, parses Prometheus text format.
 * Falls back to mock data when the endpoint is unavailable.
 */

import { ref, onUnmounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PipelineStageMetrics {
  name: string
  throughputFps: number
  latencyMs: number
  totalFrames: number
  status: 'healthy' | 'slow' | 'error' | 'idle'
}

export interface MetricsData {
  stages: PipelineStageMetrics[]
  decodeFps: number
  decodeLatencyMs: number
  decodeFramesTotal: number
  gateFramesTotal: number
  gateKeyframesTotal: number
  gatePassRate: number
  sceneCuts: number
  vlmInferencesTotal: number
  vlmLatencyMs: number
  vlmQueueDepth: number
  spacetimeEventsTotal: number
  spacetimeEmitRate: number
  activeSessions: number
  lastUpdated: number
}

export interface TraceSpan {
  stage: 'decode' | 'gate' | 'vlm'
  startMs: number
  durationMs: number
  status: 'ok' | 'slow'
}

export interface Trace {
  id: string
  frameIndex: number
  timestamp: number
  totalMs: number
  spans: TraceSpan[]
  confidence: number
}

// ─── Prometheus text parser ───────────────────────────────────────────────────

function parsePrometheus(text: string): Map<string, number> {
  const result = new Map<string, number>()
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.trim()) continue
    // Strip labels: metric_name{...} value  →  metric_name value
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:\{[^}]*\})?\s+([\d.e+\-]+)/)
    if (match) {
      const value = parseFloat(match[2])
      if (!isNaN(value)) result.set(match[1], value)
    }
  }
  return result
}

// ─── Mock data generator ──────────────────────────────────────────────────────

function generateMockText(tick: number): string {
  const t = tick * 0.15
  const fps = (28 + Math.sin(t * 0.5) * 3).toFixed(1)
  const decodeLatency = (0.5 + Math.sin(t) * 0.2).toFixed(2)
  const gateLatency = (1.0 + Math.cos(t * 0.7) * 0.3).toFixed(2)
  const vlmLatency = (350 + Math.sin(t * 0.3) * 60).toFixed(0)
  const framesTotal = 1200 + tick * 28
  const keyframesTotal = Math.floor(framesTotal * 0.15)
  const vlmTotal = Math.floor(keyframesTotal * 0.85)
  const stdbTotal = Math.floor(vlmTotal * 1.2)
  const stdbRate = (parseFloat(fps) * 0.15).toFixed(1)
  const queueDepth = Math.max(0, Math.floor(2 + Math.sin(t * 2) * 2))

  return [
    `vidarax_pipeline_throughput_fps ${fps}`,
    `vidarax_decode_latency_ms ${decodeLatency}`,
    `vidarax_decode_frames_total ${framesTotal}`,
    `vidarax_gate_latency_ms ${gateLatency}`,
    `vidarax_gate_frames_total ${framesTotal}`,
    `vidarax_gate_keyframes_total ${keyframesTotal}`,
    `vidarax_gate_scene_cuts_total ${Math.floor(keyframesTotal * 0.3)}`,
    `vidarax_vlm_latency_ms ${vlmLatency}`,
    `vidarax_vlm_inferences_total ${vlmTotal}`,
    `vidarax_vlm_queue_depth ${queueDepth}`,
    `vidarax_spacetimedb_events_total ${stdbTotal}`,
    `vidarax_spacetimedb_emit_rate_per_s ${stdbRate}`,
    `vidarax_active_sessions 1`,
  ].join('\n')
}

function buildMetrics(map: Map<string, number>, tick: number): MetricsData {
  const get = (k: string) => map.get(k) ?? 0
  const t = tick * 0.15
  const fps = get('vidarax_pipeline_throughput_fps')
  const decodeLatency = get('vidarax_decode_latency_ms')
  const gateLatency = get('vidarax_gate_latency_ms')
  const vlmLatency = get('vidarax_vlm_latency_ms')
  const sessions = get('vidarax_active_sessions')
  const stdbRate = get('vidarax_spacetimedb_emit_rate_per_s')

  function stageStatus(
    latencyMs: number,
    slowThresh: number,
    errorThresh: number,
  ): PipelineStageMetrics['status'] {
    if (latencyMs === 0) return 'idle'
    if (latencyMs > errorThresh) return 'error'
    if (latencyMs > slowThresh) return 'slow'
    return 'healthy'
  }

  const stages: PipelineStageMetrics[] = [
    {
      name: 'WebRTC',
      throughputFps: fps,
      latencyMs: parseFloat((0.2 + Math.sin(t) * 0.05).toFixed(2)),
      totalFrames: get('vidarax_decode_frames_total'),
      status: sessions > 0 ? 'healthy' : 'idle',
    },
    {
      name: 'Decode',
      throughputFps: fps,
      latencyMs: decodeLatency,
      totalFrames: get('vidarax_decode_frames_total'),
      status: stageStatus(decodeLatency, 2, 5),
    },
    {
      name: 'Gate',
      throughputFps: fps,
      latencyMs: gateLatency,
      totalFrames: get('vidarax_gate_frames_total'),
      status: stageStatus(gateLatency, 3, 8),
    },
    {
      name: 'VLM',
      throughputFps: fps * 0.15,
      latencyMs: vlmLatency,
      totalFrames: get('vidarax_vlm_inferences_total'),
      status: stageStatus(vlmLatency, 500, 800),
    },
    {
      name: 'SpacetimeDB',
      throughputFps: stdbRate,
      latencyMs: parseFloat((5 + Math.sin(t * 0.2) * 2).toFixed(1)),
      totalFrames: get('vidarax_spacetimedb_events_total'),
      status: 'healthy',
    },
  ]

  const gateFrames = get('vidarax_gate_frames_total')
  const gateKeyframes = get('vidarax_gate_keyframes_total')

  return {
    stages,
    decodeFps: fps,
    decodeLatencyMs: decodeLatency,
    decodeFramesTotal: get('vidarax_decode_frames_total'),
    gateFramesTotal: gateFrames,
    gateKeyframesTotal: gateKeyframes,
    gatePassRate: gateFrames > 0 ? gateKeyframes / gateFrames : 0,
    sceneCuts: get('vidarax_gate_scene_cuts_total'),
    vlmInferencesTotal: get('vidarax_vlm_inferences_total'),
    vlmLatencyMs: vlmLatency,
    vlmQueueDepth: get('vidarax_vlm_queue_depth'),
    spacetimeEventsTotal: get('vidarax_spacetimedb_events_total'),
    spacetimeEmitRate: stdbRate,
    activeSessions: sessions,
    lastUpdated: Date.now(),
  }
}

// ─── Mock trace generator (exported for TracingPage) ─────────────────────────

let _traceFrameIdx = 5000
let _nextTraceId = 1

export function generateMockTrace(metrics: MetricsData): Trace {
  const id = `tr-${String(_nextTraceId++).padStart(4, '0')}`
  const frameIndex = _traceFrameIdx
  _traceFrameIdx += Math.floor(6 + Math.random() * 5)

  const baseDecodeMs = metrics.decodeLatencyMs || 0.5
  const baseGateMs = 1.0
  const baseVlmMs = metrics.vlmLatencyMs || 350

  const decodeMs = baseDecodeMs * (0.85 + Math.random() * 0.3)
  const gateMs = baseGateMs * (0.8 + Math.random() * 0.4)
  const vlmMs = baseVlmMs * (0.8 + Math.random() * 0.4)

  const spans: TraceSpan[] = [
    {
      stage: 'decode',
      startMs: 0,
      durationMs: parseFloat(decodeMs.toFixed(2)),
      status: decodeMs > 2 ? 'slow' : 'ok',
    },
    {
      stage: 'gate',
      startMs: parseFloat(decodeMs.toFixed(2)),
      durationMs: parseFloat(gateMs.toFixed(2)),
      status: gateMs > 3 ? 'slow' : 'ok',
    },
    {
      stage: 'vlm',
      startMs: parseFloat((decodeMs + gateMs).toFixed(2)),
      durationMs: parseFloat(vlmMs.toFixed(2)),
      status: vlmMs > 500 ? 'slow' : 'ok',
    },
  ]

  const totalMs = decodeMs + gateMs + vlmMs
  const confidence = 0.68 + Math.random() * 0.28

  return {
    id,
    frameIndex,
    timestamp: Date.now() - Math.floor(Math.random() * 30000),
    totalMs: parseFloat(totalMs.toFixed(1)),
    spans,
    confidence: parseFloat(confidence.toFixed(2)),
  }
}

// ─── Composable ───────────────────────────────────────────────────────────────

export function useMetrics(intervalMs = 2000) {
  const metrics = ref<MetricsData | null>(null)
  const error = ref<string | null>(null)
  const loading = ref(false)
  const usingMock = ref(false)

  let tick = 0
  let timer: ReturnType<typeof setInterval> | null = null

  async function fetchMetrics() {
    loading.value = true
    error.value = null
    try {
      const auth = useAuthStore()
      let text: string
      let usedMock = false
      try {
        const res = await fetch(`${auth.apiEndpoint}/v1/metrics`, {
          headers: { ...auth.defaultHeaders(), Accept: 'text/plain' },
          signal: AbortSignal.timeout(4000),
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        text = await res.text()
      } catch {
        text = generateMockText(tick)
        usedMock = true
      }
      usingMock.value = usedMock
      const map = parsePrometheus(text)
      metrics.value = buildMetrics(map, tick)
      tick++
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Metrics fetch failed'
    } finally {
      loading.value = false
    }
  }

  function start() {
    fetchMetrics()
    timer = setInterval(fetchMetrics, intervalMs)
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null }
  }

  onUnmounted(stop)

  return { metrics, error, loading, usingMock, start, stop, refetch: fetchMetrics }
}
