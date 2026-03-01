import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const apiKey = ref<string>(localStorage.getItem('vidarax_api_key') ?? '')
  const spacetimeToken = ref<string>(localStorage.getItem('spacetime_token') ?? '')
  const apiEndpoint = ref<string>(
    localStorage.getItem('vidarax_endpoint') ?? 'http://localhost:8080'
  )
  const spacetimeEndpoint = ref<string>(
    localStorage.getItem('vidarax_spacetime_endpoint') ?? 'http://localhost:3000'
  )

  const isConfigured = computed(() => apiEndpoint.value.length > 0)

  function setApiKey(key: string) {
    apiKey.value = key
    localStorage.setItem('vidarax_api_key', key)
  }

  function setApiEndpoint(url: string) {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Only http and https protocols are allowed')
    }
    apiEndpoint.value = url
    localStorage.setItem('vidarax_endpoint', url)
  }

  function setSpacetimeEndpoint(url: string) {
    const parsed = new URL(url)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error('Only http and https protocols are allowed')
    }
    spacetimeEndpoint.value = url
    localStorage.setItem('vidarax_spacetime_endpoint', url)
  }

  function setSpacetimeToken(token: string) {
    spacetimeToken.value = token
    localStorage.setItem('spacetime_token', token)
  }

  function clearAuth() {
    apiKey.value = ''
    spacetimeToken.value = ''
    localStorage.removeItem('vidarax_api_key')
    localStorage.removeItem('spacetime_token')
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
