<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useEventsStore } from '@/stores/events'
import { useStreamStore } from '@/stores/stream'
import { Radio, Wifi, WifiOff, Server } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'

const route = useRoute()
const authStore = useAuthStore()
const eventsStore = useEventsStore()
const streamStore = useStreamStore()

const pageTitle = computed(() => {
  const meta = route.meta?.title as string | undefined
  return meta ?? 'Vidarax'
})

const endpointHost = computed(() => {
  try {
    return new URL(authStore.apiEndpoint).host
  } catch {
    return authStore.apiEndpoint
  }
})

const connectionState = computed(() => {
  if (eventsStore.isConnected) return 'connected'
  if (eventsStore.connectionError) return 'error'
  return 'disconnected'
})

const streamState = computed(() => streamStore.sessionState)
</script>

<template>
  <header
    class="h-14 shrink-0 flex items-center px-4 lg:px-6 gap-4"
    style="border-bottom: 1px solid #1e2633; background: rgba(15,17,23,0.8); backdrop-filter: blur(8px);"
  >
    <!-- Page title -->
    <h1 class="text-[#e2e8f0] font-semibold text-base flex-1 truncate">
      {{ pageTitle }}
    </h1>

    <!-- Stream status indicator -->
    <div
      v-if="streamState === 'connected'"
      class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg"
      style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.2);"
    >
      <AnimatedIcon
        :icon="Radio"
        :size="12"
        :stroke-width="2"
        animation="pulse"
        class="text-[#2dd4bf] icon-glow-teal"
      />
      <span class="text-[#2dd4bf] text-xs font-medium font-mono">
        LIVE &middot; {{ streamStore.fps.toFixed(0) }}fps
      </span>
    </div>

    <!-- Endpoint badge -->
    <div
      class="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg cursor-default"
      style="background: rgba(255,255,255,0.03); border: 1px solid #1e2633;"
    >
      <AnimatedIcon
        :icon="connectionState === 'disconnected' ? WifiOff : connectionState === 'error' ? WifiOff : Wifi"
        :size="12"
        :stroke-width="2"
        :animation="connectionState === 'connected' ? 'glow-teal' : undefined"
        :class="{
          'text-[#2dd4bf]': connectionState === 'connected',
          'text-[#ef4444]': connectionState === 'error',
          'text-[#475569]': connectionState === 'disconnected',
        }"
      />
      <AnimatedIcon
        :icon="Server"
        :size="12"
        :stroke-width="2"
        class="text-[#475569]"
      />
      <span class="text-[#64748b] text-xs font-mono truncate max-w-[140px]">
        {{ endpointHost }}
      </span>
    </div>
  </header>
</template>
