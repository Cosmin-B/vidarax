/**
 * useMetrics.ts
 * Polls GET /v1/metrics every 2s, parses Prometheus text format.
 * Falls back to mock data when the endpoint is unavailable.
 *
 * Histogram parsing: extracts cumulative bucket counts from lines of the form
 *   vidarax_pipeline_decode_latency_us_bucket{le="500"} 42
 * and computes p50/p95/p99 via linear interpolation between adjacent buckets.
 */

import { ref, onUnmounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HistogramPercentiles {
  p50: number
  p95: number
  p99: number
  /** Arithmetic mean derived from sum/count. */
  mean: number
  count: number
}

export interface PipelineStageMetrics {
  name: string
  throughputFps: number
  latencyMs: number
  totalFrames: number
  status: 'healthy' | 'slow' | 'error' | 'idle'
  /** Present when real histogram data is available from the backend. */
  percentiles?: HistogramPercentiles
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
  /** Real histogram percentiles from the backend, indexed by stage name. */
  histograms: Record<string, HistogramPercentiles>
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

// ─── Histogram parser ─────────────────────────────────────────────────────────

/**
 * Parsed representation of a single Prometheus histogram series.
 * `upperBounds` and `cumulativeCounts` are parallel arrays sorted ascending.
 * The `+Inf` bucket count is stored separately as `totalCount`.
 */
interface RawHistogram {
  upperBounds: number[]
  cumulativeCounts: number[]
  totalCount: number
  sum: number
}

/**
 * Parse all histogram series from a Prometheus text payload.
 *
 * Returns a map of base metric name → RawHistogram.  Only `_bucket`, `_sum`,
 * and `_count` lines are consumed; plain counter lines are ignored here.
 */
function parseHistograms(text: string): Map<string, RawHistogram> {
  const buckets = new Map<string, Map<string, number>>() // metricBase → (le → count)
  const sums = new Map<string, number>()
  const counts = new Map<string, number>()

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue

    // _bucket{le="..."} value
    const bucketMatch = line.match(
      /^([a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+([\d.e+\-]+)/,
    )
    if (bucketMatch) {
      const base = bucketMatch[1]!
      const le = bucketMatch[2]!
      const count = parseFloat(bucketMatch[3]!)
      if (!isNaN(count)) {
        if (!buckets.has(base)) buckets.set(base, new Map())
        buckets.get(base)!.set(le, count)
      }
      continue
    }

    // _sum value
    const sumMatch = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)_sum\s+([\d.e+\-]+)/)
    if (sumMatch) {
      const v = parseFloat(sumMatch[2]!)
      if (!isNaN(v)) sums.set(sumMatch[1]!, v)
      continue
    }

    // _count value
    const countMatch = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)_count\s+([\d.e+\-]+)/)
    if (countMatch) {
      const v = parseFloat(countMatch[2]!)
      if (!isNaN(v)) counts.set(countMatch[1]!, v)
      continue
    }
  }

  const result = new Map<string, RawHistogram>()
  for (const [base, leMap] of buckets) {
    const upperBounds: number[] = []
    const cumulativeCounts: number[] = []
    let totalCount = 0

    // Collect finite buckets in ascending order.
    const entries = [...leMap.entries()].filter(([le]) => le !== '+Inf')
    entries.sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))
    for (const [le, cnt] of entries) {
      upperBounds.push(parseFloat(le))
      cumulativeCounts.push(cnt)
    }

    // +Inf bucket equals the total observation count.
    const inf = leMap.get('+Inf')
    if (inf !== undefined) {
      totalCount = inf
    } else {
      totalCount = counts.get(base) ?? (cumulativeCounts[cumulativeCounts.length - 1] ?? 0)
    }

    result.set(base, {
      upperBounds,
      cumulativeCounts,
      totalCount,
      sum: sums.get(base) ?? 0,
    })
  }

  return result
}

/**
 * Estimate the value at quantile `q` (0–1) via linear interpolation between
 * adjacent histogram buckets.
 *
 * Algorithm matches the standard Prometheus `histogram_quantile` approach:
 * find the first bucket whose cumulative count exceeds `q * totalCount`, then
 * interpolate linearly within that bucket's range.
 */
function histogramQuantile(h: RawHistogram, q: number): number {
  if (h.totalCount === 0) return 0
  const target = q * h.totalCount

  let lowerBound = 0
  let lowerCount = 0

  for (let i = 0; i < h.upperBounds.length; i++) {
    const upper = h.upperBounds[i]!
    const upperCount = h.cumulativeCounts[i]!
    if (upperCount >= target) {
      // Linear interpolation within this bucket.
      const countInBucket = upperCount - lowerCount
      if (countInBucket === 0) return upper
      const fraction = (target - lowerCount) / countInBucket
      return lowerBound + fraction * (upper - lowerBound)
    }
    lowerBound = upper
    lowerCount = upperCount
  }

  // Quantile falls in the +Inf bucket — return the last finite upper bound as
  // a conservative estimate (we cannot interpolate to infinity).
  return h.upperBounds[h.upperBounds.length - 1] ?? 0
}

/** Derive p50/p95/p99/mean from a RawHistogram. */
function toPercentiles(h: RawHistogram): HistogramPercentiles {
  return {
    p50: histogramQuantile(h, 0.5),
    p95: histogramQuantile(h, 0.95),
    p99: histogramQuantile(h, 0.99),
    mean: h.totalCount > 0 ? h.sum / h.totalCount : 0,
    count: h.totalCount,
  }
}

// ─── Prometheus scalar parser ─────────────────────────────────────────────────

function parsePrometheus(text: string): Map<string, number> {
  const result = new Map<string, number>()
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.trim()) continue
    // Strip labels: metric_name{...} value  →  metric_name value
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?:\{[^}]*\})?\s+([\d.e+\-]+)/)
    if (match) {
      const value = parseFloat(match[2]!)
      if (!isNaN(value)) result.set(match[1]!, value)
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

// ─── Histogram → ms conversion helpers ───────────────────────────────────────

/**
 * Convert decode histogram percentiles from µs to ms, returning null when
 * no real histogram data is present.
 */
function decodePercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_decode_latency_us')
  if (!h || h.totalCount === 0) return null
  const p = toPercentiles(h)
  // Histogram is in microseconds; convert to milliseconds for the UI.
  return { p50: p.p50 / 1000, p95: p.p95 / 1000, p99: p.p99 / 1000, mean: p.mean / 1000, count: p.count }
}

function gatePercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_gate_latency_us')
  if (!h || h.totalCount === 0) return null
  const p = toPercentiles(h)
  return { p50: p.p50 / 1000, p95: p.p95 / 1000, p99: p.p99 / 1000, mean: p.mean / 1000, count: p.count }
}

function vlmPercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_vlm_latency_ms')
  if (!h || h.totalCount === 0) return null
  return toPercentiles(h)
}

function stdbPercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_stdb_emit_latency_ms')
  if (!h || h.totalCount === 0) return null
  return toPercentiles(h)
}

// ─── MetricsData builder ──────────────────────────────────────────────────────

function buildMetrics(
  map: Map<string, number>,
  histograms: Map<string, RawHistogram>,
  tick: number,
): MetricsData {
  const get = (k: string) => map.get(k) ?? 0
  const t = tick * 0.15
  const fps = get('vidarax_pipeline_throughput_fps')

  // Prefer real histogram mean for latency display; fall back to plain counter.
  const decodePct = decodePercentilesMs(histograms)
  const gatePct = gatePercentilesMs(histograms)
  const vlmPct = vlmPercentilesMs(histograms)
  const stdbPct = stdbPercentilesMs(histograms)

  const decodeLatency = decodePct?.mean ?? get('vidarax_decode_latency_ms')
  const gateLatency = gatePct?.mean ?? get('vidarax_gate_latency_ms')
  const vlmLatency = vlmPct?.mean ?? get('vidarax_vlm_latency_ms')
  const sessions = get('vidarax_active_sessions')
  const stdbRate = get('vidarax_spacetimedb_emit_rate_per_s')
  const stdbLatency = stdbPct?.mean ?? parseFloat((5 + Math.sin(t * 0.2) * 2).toFixed(1))

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
      percentiles: decodePct ?? undefined,
    },
    {
      name: 'Gate',
      throughputFps: fps,
      latencyMs: gateLatency,
      totalFrames: get('vidarax_gate_frames_total'),
      status: stageStatus(gateLatency, 3, 8),
      percentiles: gatePct ?? undefined,
    },
    {
      name: 'VLM',
      throughputFps: fps * 0.15,
      latencyMs: vlmLatency,
      totalFrames: get('vidarax_vlm_inferences_total'),
      status: stageStatus(vlmLatency, 500, 800),
      percentiles: vlmPct ?? undefined,
    },
    {
      name: 'SpacetimeDB',
      throughputFps: stdbRate,
      latencyMs: stdbLatency,
      totalFrames: get('vidarax_spacetimedb_events_total'),
      status: 'healthy',
      percentiles: stdbPct ?? undefined,
    },
  ]

  const gateFrames = get('vidarax_gate_frames_total')
  const gateKeyframes = get('vidarax_gate_keyframes_total')

  // Collect all available histogram percentiles for consumers (MetricsGrid, etc.)
  const histogramsOut: Record<string, HistogramPercentiles> = {}
  if (decodePct) histogramsOut['Decode'] = decodePct
  if (gatePct) histogramsOut['Gate'] = gatePct
  if (vlmPct) histogramsOut['VLM'] = vlmPct
  if (stdbPct) histogramsOut['SpacetimeDB'] = stdbPct

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
    histograms: histogramsOut,
  }
}

// ─── Mock trace generator (exported for TracingPage) ─────────────────────────

let _traceFrameIdx = 5000
let _nextTraceId = 1

export function generateMockTrace(metrics: MetricsData): Trace {
  const id = `tr-${String(_nextTraceId++).padStart(4, '0')}`
  const frameIndex = _traceFrameIdx
  _traceFrameIdx += Math.floor(6 + Math.random() * 5)

  // Use real p50 from histogram when available; fall back to current mean.
  const decodePct = metrics.histograms['Decode']
  const vlmPct = metrics.histograms['VLM']
  const gatePct = metrics.histograms['Gate']

  const baseDecodeMs = (decodePct?.p50 ?? metrics.decodeLatencyMs) || 0.5
  const baseGateMs = gatePct?.p50 ?? 1.0
  const baseVlmMs = (vlmPct?.p50 ?? metrics.vlmLatencyMs) || 350

  const decodeMs = baseDecodeMs * (0.85 + Math.random() * 0.3)
  const gateMs = baseGateMs * (0.8 + Math.random() * 0.4)
  const vlmMs = baseVlmMs * (0.8 + Math.random() * 0.4)

  // Determine "slow" thresholds from p95 when available.
  const decodeSlowThresh = decodePct?.p95 ?? 2
  const gateSlowThresh = gatePct?.p95 ?? 3
  const vlmSlowThresh = vlmPct?.p95 ?? 500

  const spans: TraceSpan[] = [
    {
      stage: 'decode',
      startMs: 0,
      durationMs: parseFloat(decodeMs.toFixed(2)),
      status: decodeMs > decodeSlowThresh ? 'slow' : 'ok',
    },
    {
      stage: 'gate',
      startMs: parseFloat(decodeMs.toFixed(2)),
      durationMs: parseFloat(gateMs.toFixed(2)),
      status: gateMs > gateSlowThresh ? 'slow' : 'ok',
    },
    {
      stage: 'vlm',
      startMs: parseFloat((decodeMs + gateMs).toFixed(2)),
      durationMs: parseFloat(vlmMs.toFixed(2)),
      status: vlmMs > vlmSlowThresh ? 'slow' : 'ok',
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
      const histograms = parseHistograms(text)
      metrics.value = buildMetrics(map, histograms, tick)
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
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  onUnmounted(stop)

  return { metrics, error, loading, usingMock, start, stop, refetch: fetchMetrics }
}
