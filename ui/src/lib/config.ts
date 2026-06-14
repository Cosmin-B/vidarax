export const UI_DEFAULTS = {
  apiEndpoint: 'http://localhost:8080',
  apiKey: '',
  spacetimeEndpoint: 'http://localhost:3000',
  turnUrl: '',
  fps: 5,
  chunkSize: 30,
  tokenRateCap: 0,
  clipMode: false,
  semanticInference: true,
  semanticFramesPerChunk: 4,
  sceneCutHammingThreshold: 10,
  lumaShiftThreshold: 0.15,
  loopDetection: true,
  loopRepeatTrigger: 3,
  gpuDecode: 'auto',
  defaultModel: 'Qwen/Qwen3-VL-4B-Instruct',
  firstPassModel: 'Qwen/Qwen3-VL-2B-Instruct',
  secondPassModel: 'Qwen/Qwen3-VL-4B-Instruct',
} as const

export const STORAGE_KEYS = {
  apiKey: 'vidarax_api_key',
  spacetimeToken: 'spacetime_token',
  apiEndpoint: 'vidarax_endpoint',
  spacetimeEndpoint: 'vidarax_spacetime_endpoint',
  turnUrl: 'vidarax_turn_url',
  defaultModel: 'vidarax_default_model',
  firstPassModel: 'vidarax_first_pass_model',
  secondPassModel: 'vidarax_second_pass_model',
  fps: 'vidarax_fps',
  chunkSize: 'vidarax_chunk_size',
  tokenRateCap: 'vidarax_token_rate_cap',
  clipMode: 'vidarax_clip_mode',
  semanticInference: 'vidarax_semantic_inference',
  semanticFramesPerChunk: 'vidarax_semantic_frames_per_chunk',
  sceneCutHammingThreshold: 'vidarax_scene_cut_hamming_threshold',
  lumaShiftThreshold: 'vidarax_luma_shift_threshold',
  loopDetection: 'vidarax_loop_detection',
  loopRepeatTrigger: 'vidarax_loop_repeat_trigger',
  gpuDecode: 'vidarax_gpu_decode',
} as const

export const SPACETIME_MODULE_NAME = 'vidarax'
export const SPACETIME_PROTOCOLS = ['v1.json.spacetimedb'] as const

export const ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
]

export function ls(key: string, fallback: string): string {
  return localStorage.getItem(key) ?? fallback
}

export function lsNum(key: string, fallback: number): number {
  const v = localStorage.getItem(key)
  return v !== null ? Number(v) : fallback
}

export function lsBool(key: string, fallback: boolean): boolean {
  const v = localStorage.getItem(key)
  return v !== null ? v !== 'false' : fallback
}

export function buildSpacetimeSubscribeUrl(baseUrl: string): string {
  const url = new URL(baseUrl.replace(/\/$/, ''))
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = `/v1/database/${SPACETIME_MODULE_NAME}/subscribe`
  return url.toString()
}
