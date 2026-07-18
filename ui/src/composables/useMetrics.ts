/**
 * useMetrics.ts
 * Polls GET /v1/metrics every 2s, parses Prometheus text format.
 * Failed requests remain explicit; this composable never fabricates telemetry.
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
  itemsTotal: number
  itemLabel: string
  latencyMs: number
  status: 'healthy' | 'slow' | 'error' | 'idle'
  /** Present when real histogram data is available from the backend. */
  percentiles?: HistogramPercentiles
}

export interface MetricsData {
  stages: PipelineStageMetrics[]
  decodeLatencyMs: number
  decodeFramesTotal: number
  droppedFramesTotal: number
  gateFramesTotal: number
  gateKeyframesTotal: number
  keyframesDroppedTotal: number
  gatePassRate: number
  loopDetectedTotal: number
  vlmInferencesTotal: number
  vlmLatencyMs: number
  noveltyEvaluatedTotal: number
  noveltyReusedTotal: number
  noveltyForcedRefreshTotal: number
  noveltyEmbeddingUnavailableTotal: number
  noveltyReuseRatio: number
  keyframeBlobsWrittenTotal: number
  keyframeBlobsReusedTotal: number
  keyframeBlobFailuresTotal: number
  keyframeBlobBytesTotal: number
  activeSessions: number
  activeGenerations: number
  generationsStartedTotal: number
  cleanShutdownsTotal: number
  faultedShutdownsTotal: number
  forcedShutdownsTotal: number
  workerFaultsTotal: number
  pipelineStatus: 'faulted' | 'saturated' | 'healthy' | 'idle'
  lastUpdated: number
  /** Real histogram percentiles from the backend, indexed by stage name. */
  histograms: Record<string, HistogramPercentiles>
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

function noveltyPercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_novelty_embedding_latency_ms')
  if (!h || h.totalCount === 0) return null
  return toPercentiles(h)
}

function keyframeBlobPercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_keyframe_blob_latency_ms')
  if (!h || h.totalCount === 0) return null
  return toPercentiles(h)
}

function eventMirrorPercentilesMs(
  histograms: Map<string, RawHistogram>,
): HistogramPercentiles | null {
  const h = histograms.get('vidarax_pipeline_stdb_emit_latency_ms')
  if (!h || h.totalCount === 0) return null
  return toPercentiles(h)
}

function buildMetrics(
  map: Map<string, number>,
  histograms: Map<string, RawHistogram>,
): MetricsData {
  const get = (k: string) => map.get(k) ?? 0

  // Latency is emitted as histograms by the backend. Missing histograms are
  // represented as idle, never estimated.
  const decodePct = decodePercentilesMs(histograms)
  const gatePct = gatePercentilesMs(histograms)
  const vlmPct = vlmPercentilesMs(histograms)
  const noveltyPct = noveltyPercentilesMs(histograms)
  const keyframeBlobPct = keyframeBlobPercentilesMs(histograms)
  const eventMirrorPct = eventMirrorPercentilesMs(histograms)

  const decodeLatency = decodePct?.mean ?? 0
  const gateLatency = gatePct?.mean ?? 0
  const vlmLatency = vlmPct?.mean ?? 0
  const sessions = Math.max(
    0,
    get('vidarax_pipeline_sessions_created_total')
      - get('vidarax_pipeline_sessions_removed_total'),
  )
  const activeGenerations = get('vidarax_pipeline_generations_active')
  const lastFaultSeconds = get('vidarax_pipeline_last_fault_timestamp_seconds')
  const faultIsRecent = lastFaultSeconds > 0
    && Math.floor(Date.now() / 1000) - lastFaultSeconds < 60
  const admissionWaiting = get('vidarax_infer_admission_waiting')
  const capacityRejectedSeconds = get('vidarax_media_capacity_last_rejection_timestamp_seconds')
  const capacityRejectedRecently = capacityRejectedSeconds > 0
    && Math.floor(Date.now() / 1000) - capacityRejectedSeconds < 60
  const memoryLimit = get('vidarax_media_capacity_memory_limit_bytes')
  const memoryReserved = get('vidarax_media_capacity_memory_reserved_bytes')
  const threadLimit = get('vidarax_media_capacity_worker_thread_limit')
  const threadsReserved = get('vidarax_media_capacity_worker_threads_reserved')
  const atCapacity = (memoryLimit > 0 && memoryReserved >= memoryLimit)
    || (threadLimit > 0 && threadsReserved >= threadLimit)
  const pipelineStatus: MetricsData['pipelineStatus'] = faultIsRecent
    ? 'faulted'
    : admissionWaiting > 0 || capacityRejectedRecently || atCapacity
      ? 'saturated'
      : activeGenerations > 0
        ? 'healthy'
        : 'idle'
  const rtpFrames = get('vidarax_pipeline_rtp_frames_received_total')
  const decodedFrames = get('vidarax_pipeline_frames_decoded_total')
  const droppedFrames = get('vidarax_pipeline_frames_dropped_total')
  const keyframes = get('vidarax_pipeline_keyframes_total')
  // New backends expose gate decisions independently from downstream VLM
  // queueing. Fall back to the legacy counters during rolling upgrades.
  const gateFrames = map.has('vidarax_pipeline_gate_frames_analyzed_total')
    ? get('vidarax_pipeline_gate_frames_analyzed_total')
    : decodedFrames
  const gateSelected = map.has('vidarax_pipeline_gate_keyframes_selected_total')
    ? get('vidarax_pipeline_gate_keyframes_selected_total')
    : keyframes
  const keyframesDropped = get('vidarax_pipeline_keyframes_dropped_total')
    + get('vidarax_pipeline_sink_keyframes_dropped_total')
  const vlmInferences = get('vidarax_pipeline_vlm_inferences_total')
  const noveltyEvaluated = get('vidarax_pipeline_novelty_evaluated_total')
  const noveltyReused = get('vidarax_pipeline_novelty_reused_total')
  const keyframeBlobsWritten = get('vidarax_pipeline_keyframe_blobs_written_total')
  const keyframeBlobFailures = get('vidarax_pipeline_keyframe_blob_failures_total')

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
      itemsTotal: rtpFrames,
      itemLabel: 'frames',
      latencyMs: 0,
      status: sessions > 0 ? 'healthy' : 'idle',
    },
    {
      name: 'Decode',
      itemsTotal: decodedFrames,
      itemLabel: 'frames',
      latencyMs: decodeLatency,
      status: stageStatus(decodeLatency, 2, 5),
      percentiles: decodePct ?? undefined,
    },
    {
      name: 'Gate',
      itemsTotal: gateSelected,
      itemLabel: 'selected',
      latencyMs: gateLatency,
      status: stageStatus(gateLatency, 3, 8),
      percentiles: gatePct ?? undefined,
    },
    {
      name: 'VLM',
      itemsTotal: vlmInferences,
      itemLabel: 'calls',
      latencyMs: vlmLatency,
      status: stageStatus(vlmLatency, 500, 800),
      percentiles: vlmPct ?? undefined,
    },
    {
      name: 'Keyframe store',
      itemsTotal: keyframeBlobsWritten,
      itemLabel: 'blobs',
      latencyMs: keyframeBlobPct?.mean ?? 0,
      status: keyframeBlobFailures > 0
        ? 'error'
        : stageStatus(keyframeBlobPct?.mean ?? 0, 20, 100),
      percentiles: keyframeBlobPct ?? undefined,
    },
  ]

  const histogramsOut: Record<string, HistogramPercentiles> = {}
  if (decodePct) histogramsOut['Decode'] = decodePct
  if (gatePct) histogramsOut['Gate'] = gatePct
  if (vlmPct) histogramsOut['VLM'] = vlmPct
  if (noveltyPct) histogramsOut['Novelty'] = noveltyPct
  if (keyframeBlobPct) histogramsOut['Keyframe store'] = keyframeBlobPct
  if (eventMirrorPct) histogramsOut['Event mirror'] = eventMirrorPct

  return {
    stages,
    decodeLatencyMs: decodeLatency,
    decodeFramesTotal: decodedFrames,
    droppedFramesTotal: droppedFrames,
    gateFramesTotal: gateFrames,
    gateKeyframesTotal: gateSelected,
    keyframesDroppedTotal: keyframesDropped,
    gatePassRate: gateFrames > 0 ? gateSelected / gateFrames : 0,
    loopDetectedTotal: get('vidarax_pipeline_loop_detected_total'),
    vlmInferencesTotal: vlmInferences,
    vlmLatencyMs: vlmLatency,
    noveltyEvaluatedTotal: noveltyEvaluated,
    noveltyReusedTotal: noveltyReused,
    noveltyForcedRefreshTotal: get('vidarax_pipeline_novelty_forced_refresh_total'),
    noveltyEmbeddingUnavailableTotal: get('vidarax_pipeline_novelty_embedding_unavailable_total'),
    noveltyReuseRatio: noveltyEvaluated > 0 ? noveltyReused / noveltyEvaluated : 0,
    keyframeBlobsWrittenTotal: keyframeBlobsWritten,
    keyframeBlobsReusedTotal: get('vidarax_pipeline_keyframe_blobs_reused_total'),
    keyframeBlobFailuresTotal: keyframeBlobFailures,
    keyframeBlobBytesTotal: get('vidarax_pipeline_keyframe_blob_bytes_total'),
    activeSessions: sessions,
    activeGenerations,
    generationsStartedTotal: get('vidarax_pipeline_generations_started_total'),
    cleanShutdownsTotal: get('vidarax_pipeline_generation_shutdown_clean_total'),
    faultedShutdownsTotal: get('vidarax_pipeline_generation_shutdown_faulted_total'),
    forcedShutdownsTotal: get('vidarax_pipeline_generation_shutdown_forced_total'),
    workerFaultsTotal: get('vidarax_pipeline_worker_faults_total_all'),
    pipelineStatus,
    lastUpdated: Date.now(),
    histograms: histogramsOut,
  }
}

// ─── Composable ───────────────────────────────────────────────────────────────

export function useMetrics(intervalMs = 2000) {
  const metrics = ref<MetricsData | null>(null)
  const error = ref<string | null>(null)
  const loading = ref(false)

  let timer: ReturnType<typeof setInterval> | null = null

  async function fetchMetrics() {
    loading.value = true
    error.value = null
    try {
      const auth = useAuthStore()
      const res = await fetch(`${auth.apiEndpoint}/v1/metrics`, {
        headers: { ...auth.defaultHeaders(), Accept: 'text/plain' },
        signal: AbortSignal.timeout(4000),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const text = await res.text()
      const map = parsePrometheus(text)
      const histograms = parseHistograms(text)
      metrics.value = buildMetrics(map, histograms)
    } catch (e) {
      const detail = e instanceof Error ? e.message : 'unknown error'
      error.value = `Metrics unavailable: ${detail}`
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

  return { metrics, error, loading, start, stop, refetch: fetchMetrics }
}
