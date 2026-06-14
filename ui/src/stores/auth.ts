import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { STORAGE_KEYS, UI_DEFAULTS } from '@/lib/config'

export const useAuthStore = defineStore('auth', () => {
  const apiKey = ref<string>(localStorage.getItem(STORAGE_KEYS.apiKey) ?? '')
  const spacetimeToken = ref<string>(localStorage.getItem(STORAGE_KEYS.spacetimeToken) ?? '')
  const apiEndpoint = ref<string>(
    localStorage.getItem(STORAGE_KEYS.apiEndpoint) ?? UI_DEFAULTS.apiEndpoint
  )
  const spacetimeEndpoint = ref<string>(
    localStorage.getItem(STORAGE_KEYS.spacetimeEndpoint) ?? UI_DEFAULTS.spacetimeEndpoint
  )

  const isConfigured = computed(() => apiEndpoint.value.length > 0)

  function setApiKey(key: string) {
    apiKey.value = key
    localStorage.setItem(STORAGE_KEYS.apiKey, key)
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
    localStorage.setItem(STORAGE_KEYS.spacetimeToken, token)
  }

  function clearAuth() {
    apiKey.value = ''
    spacetimeToken.value = ''
    localStorage.removeItem(STORAGE_KEYS.apiKey)
    localStorage.removeItem(STORAGE_KEYS.spacetimeToken)
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
