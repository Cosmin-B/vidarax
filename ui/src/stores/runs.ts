import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type RunStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'expired'
export type RunMode = 'balanced' | 'detailed' | 'efficiency' | 'custom'

export interface Run {
  run_id: string
  status: RunStatus
  mode: RunMode
  model: string
  source_uri?: string
  created_at?: string
  updated_at?: string
  frame_count?: number
  event_count?: number
  duration_ms?: number
  error?: string
}

export interface RunSummary {
  run_id: string
  status: RunStatus
  mode: RunMode
  model: string
  source_uri?: string
  created_at?: string
  event_count?: number
}

export const useRunsStore = defineStore('runs', () => {
  const runs = ref<RunSummary[]>([])
  const activeRun = ref<Run | null>(null)
  const activeRunId = ref<string | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  const sortedRuns = computed(() =>
    [...runs.value].sort((a, b) => {
      const ta = a.created_at ? new Date(a.created_at).getTime() : 0
      const tb = b.created_at ? new Date(b.created_at).getTime() : 0
      return tb - ta
    })
  )

  const activeRunStatus = computed(() => activeRun.value?.status ?? null)

  function setRuns(list: RunSummary[]) {
    runs.value = list
  }

  function setActiveRun(run: Run | null) {
    activeRun.value = run
    activeRunId.value = run?.run_id ?? null
  }

  function upsertRun(summary: RunSummary) {
    const idx = runs.value.findIndex(r => r.run_id === summary.run_id)
    if (idx >= 0) {
      runs.value[idx] = summary
    } else {
      runs.value.unshift(summary)
    }
  }

  function removeRun(runId: string) {
    runs.value = runs.value.filter(r => r.run_id !== runId)
    if (activeRunId.value === runId) {
      activeRun.value = null
      activeRunId.value = null
    }
  }

  function updateRunStatus(runId: string, status: RunStatus) {
    const run = runs.value.find(r => r.run_id === runId)
    if (run) run.status = status
    if (activeRun.value?.run_id === runId) {
      activeRun.value.status = status
    }
  }

  function setLoading(loading: boolean) {
    isLoading.value = loading
  }

  function setError(msg: string | null) {
    error.value = msg
  }

  return {
    runs,
    activeRun,
    activeRunId,
    isLoading,
    error,
    sortedRuns,
    activeRunStatus,
    setRuns,
    setActiveRun,
    upsertRun,
    removeRun,
    updateRunStatus,
    setLoading,
    setError,
  }
})
