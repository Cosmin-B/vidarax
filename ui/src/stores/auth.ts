import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { STORAGE_KEYS, UI_DEFAULTS } from '@/lib/config'

// Earlier builds kept these bearer secrets in localStorage, where they persist
// on disk and stay readable long after a session ends. They now live in
// sessionStorage, which is per-tab and cleared when the tab closes. Carry a
// surviving localStorage value over once so an upgrading user keeps their
// session, then delete the durable copy so the secret no longer outlives the
// tab. Endpoints are not secret and stay in localStorage on purpose.
function loadSecret(key: string): string {
  const legacy = localStorage.getItem(key)
  if (legacy !== null) {
    try {
      if (sessionStorage.getItem(key) === null) {
        sessionStorage.setItem(key, legacy)
      }
      localStorage.removeItem(key)
    } catch {
      // sessionStorage can refuse writes (quota, private mode). Rather than let
      // a convenience migration abort store construction and blank the app, use
      // the legacy value for this session and leave it in place to retry later.
      return legacy
    }
  }
  return sessionStorage.getItem(key) ?? ''
}

// Remove a secret from both stores, tolerating either backend being
// unavailable so one failure does not skip the other.
function forgetSecret(key: string): void {
  try {
    sessionStorage.removeItem(key)
  } catch {
    // storage unavailable
  }
  try {
    localStorage.removeItem(key)
  } catch {
    // storage unavailable
  }
}

export const useAuthStore = defineStore('auth', () => {
  const apiKey = ref<string>(loadSecret(STORAGE_KEYS.apiKey))
  const spacetimeToken = ref<string>(loadSecret(STORAGE_KEYS.spacetimeToken))
  const apiEndpoint = ref<string>(
    localStorage.getItem(STORAGE_KEYS.apiEndpoint) ?? UI_DEFAULTS.apiEndpoint
  )
  const spacetimeEndpoint = ref<string>(
    localStorage.getItem(STORAGE_KEYS.spacetimeEndpoint) ?? UI_DEFAULTS.spacetimeEndpoint
  )

  const isConfigured = computed(() => apiEndpoint.value.length > 0)

  function setApiKey(key: string) {
    apiKey.value = key
    sessionStorage.setItem(STORAGE_KEYS.apiKey, key)
  }

  function setApiEndpoint(url: string) {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Only http and https protocols are allowed')
    }
    apiEndpoint.value = url
    localStorage.setItem(STORAGE_KEYS.apiEndpoint, url)
  }

  function setSpacetimeEndpoint(url: string) {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Only http and https protocols are allowed')
    }
    spacetimeEndpoint.value = url
    localStorage.setItem(STORAGE_KEYS.spacetimeEndpoint, url)
  }

  function setSpacetimeToken(token: string) {
    spacetimeToken.value = token
    sessionStorage.setItem(STORAGE_KEYS.spacetimeToken, token)
  }

  function clearAuth() {
    apiKey.value = ''
    spacetimeToken.value = ''
    // Clears sessionStorage and any legacy localStorage copy the migration missed.
    forgetSecret(STORAGE_KEYS.apiKey)
    forgetSecret(STORAGE_KEYS.spacetimeToken)
  }

  function defaultHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (apiKey.value) {
      headers['x-api-key'] = apiKey.value
    }
    return headers
  }

  return {
    apiKey,
    spacetimeToken,
    apiEndpoint,
    spacetimeEndpoint,
    isConfigured,
    setApiKey,
    setApiEndpoint,
    setSpacetimeEndpoint,
    setSpacetimeToken,
    clearAuth,
    defaultHeaders,
  }
})
