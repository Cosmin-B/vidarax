/**
 * Typed fetch wrapper for the Vidarax REST API.
 * Base URL and API key come from the auth store.
 */

import { useAuthStore } from '@/stores/auth'

/** Validate that an ID contains only safe characters (alphanumeric, dash, underscore). */
function validateId(id: string): string {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
    throw new Error(`Invalid ID format: ${id}`)
  }
  return id
}

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  overrideHeaders?: Record<string, string>,
): Promise<T> {
  const auth = useAuthStore()
  const url = `${auth.apiEndpoint}${path}`
  const headers: Record<string, string> = {
    ...auth.defaultHeaders(),
    ...overrideHeaders,
  }

  const init: RequestInit = { method, headers }
  if (body !== undefined) {
    init.body = JSON.stringify(body)
  }

  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, text)
  }

  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) {
    return res.json() as Promise<T>
  }
  return res.text() as unknown as Promise<T>
}

// ------- Type definitions -------

export interface CreateRunRequest {
  source_uri: string
  model?: string
  prompt?: string
  semantic_inference?: boolean
  fps?: number
  chunk_size?: number
}

export interface RunResponse {
  run_id: string
  status: string
  mode: string
  model: string
  source_uri: string
  created_at: string
  updated_at: string
}

export interface ReasonRequest {
  source_uri?: string
  model?: string
  prompt?: string
  semantic_inference?: boolean
  fps?: number
  chunk_size?: number
  semantic_frames_per_chunk?: number
}

export interface ModelInfo {
  id: string
  name: string
  provider: string
  context_length?: number
}

export interface ModelsResponse {
  models: ModelInfo[]
}

export interface HealthResponse {
  status: string
  version: string
  gpu: boolean
}

// ------- API methods -------

export const api = {
  health(): Promise<HealthResponse> {
    return request<HealthResponse>('GET', '/health')
  },

  models(): Promise<ModelsResponse> {
    return request<ModelsResponse>('GET', '/v1/models')
  },

  runs: {
    list(): Promise<RunResponse[]> {
      return request<RunResponse[]>('GET', '/v1/runs')
    },
    get(runId: string): Promise<RunResponse> {
      return request<RunResponse>('GET', `/v1/runs/${validateId(runId)}`)
    },
    create(data: CreateRunRequest): Promise<RunResponse> {
      return request<RunResponse>('POST', '/v1/runs', data)
    },
    reason(runId: string, data: ReasonRequest): Promise<RunResponse> {
      return request<RunResponse>('POST', `/v1/runs/${validateId(runId)}/reason`, data)
    },
    stop(runId: string): Promise<void> {
      return request<void>('POST', `/v1/runs/${validateId(runId)}/stop`)
    },
    delete(runId: string): Promise<void> {
      return request<void>('DELETE', `/v1/runs/${validateId(runId)}`)
    },
    keepalive(runId: string): Promise<void> {
      return request<void>('POST', `/v1/runs/${validateId(runId)}/keepalive`)
    },
  },

  stream: {
    whipOffer(offerSdp: string): Promise<{ answer_sdp: string; session_id: string; location: string }> {
      return request<{ answer_sdp: string; session_id: string; location: string }>(
        'POST',
        '/v1/stream/whip',
        offerSdp,
        { 'Content-Type': 'application/sdp', 'x-api-key': useAuthStore().apiKey },
      )
    },
    whipIce(sessionId: string, candidate: string): Promise<void> {
      return request<void>('PATCH', `/v1/stream/whip/${validateId(sessionId)}`, candidate, {
        'Content-Type': 'application/trickle-ice-sdpfrag',
      })
    },
    whipTerminate(sessionId: string): Promise<void> {
      return request<void>('DELETE', `/v1/stream/whip/${validateId(sessionId)}`)
    },
    attachToRun(runId: string, data: { session_id: string; model?: string; prompt?: string }): Promise<void> {
      return request<void>('POST', `/v1/runs/${validateId(runId)}/stream`, data)
    },
    uploadFile(file: File): Promise<{ file_path: string }> {
      const auth = useAuthStore()
      const url = `${auth.apiEndpoint}/v1/upload`
      const formData = new FormData()
      formData.append('file', file)
      return fetch(url, {
        method: 'POST',
        headers: { 'x-api-key': auth.apiKey },
        body: formData,
      }).then(res => {
        if (!res.ok) throw new ApiError(res.status, res.statusText)
        return res.json()
      })
    },
  },
}
