<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useStreamStore } from '@/stores/stream'
import { useEventsStore } from '@/stores/events'

const route = useRoute()
const streamStore = useStreamStore()
const eventsStore = useEventsStore()

interface NavItem {
  name: string
  path: string
  icon: string
  label: string
  badge?: string | null
}

const navItems: NavItem[] = [
  {
    name: 'dashboard',
    path: '/',
    icon: 'grid',
    label: 'Dashboard',
  },
  {
    name: 'stream',
    path: '/stream',
    icon: 'broadcast',
    label: 'Stream',
  },
  {
    name: 'upload',
    path: '/upload',
    icon: 'upload',
    label: 'Upload',
  },
  {
    name: 'settings',
    path: '/settings',
    icon: 'settings',
    label: 'Settings',
  },
]

function isActive(item: NavItem): boolean {
  if (item.path === '/') return route.path === '/'
  return route.path.startsWith(item.path)
}

const streamActive = computed(() => streamStore.isActive)
const spacetimeConnected = computed(() => eventsStore.isConnected)
</script>

<template>
  <!-- Desktop sidebar -->
  <aside
    class="hidden lg:flex flex-col w-16 xl:w-56 shrink-0 h-full"
    style="background: linear-gradient(180deg, #0f1117 0%, #0a0c12 100%);
           border-right: 1px solid #1e2633;"
  >
    <!-- Logo area -->
    <div class="flex items-center gap-3 px-4 py-5 border-b border-[#1e2633]">
      <div
        class="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
        style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
               box-shadow: 0 0 20px rgba(45,212,191,0.25);"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2 4L8 1L14 4V8C14 11.3 11.3 14.3 8 15C4.7 14.3 2 11.3 2 8V4Z"
                fill="rgba(255,255,255,0.9)" stroke="none"/>
          <circle cx="8" cy="8" r="2.5" fill="#0d9488"/>
        </svg>
      </div>
      <span class="hidden xl:block text-[#e2e8f0] font-semibold text-sm tracking-wide">
        Vidarax
      </span>
    </div>

    <!-- Nav items -->
    <nav class="flex flex-col gap-1 p-3 flex-1" role="navigation" aria-label="Main navigation">
      <RouterLink
        v-for="item in navItems"
        :key="item.name"
        :to="item.path"
        class="group flex items-center gap-3 px-2 py-2.5 rounded-[8px] transition-all duration-200 relative"
        :class="isActive(item)
          ? 'bg-[rgba(45,212,191,0.08)] text-[#2dd4bf]'
          : 'text-[#64748b] hover:text-[#94a3b8] hover:bg-[rgba(255,255,255,0.03)]'"
        :aria-current="isActive(item) ? 'page' : undefined"
      >
        <!-- Active indicator bar -->
        <div
          v-if="isActive(item)"
          class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r-full"
          style="background: #2dd4bf; box-shadow: 0 0 8px rgba(45,212,191,0.6);"
        />

        <!-- Icon -->
        <div class="w-5 h-5 shrink-0 flex items-center justify-center">
          <!-- Grid icon (Dashboard) -->
          <svg v-if="item.icon === 'grid'" width="18" height="18" viewBox="0 0 18 18" fill="none">
            <rect x="2" y="2" width="6" height="6" rx="1.5" fill="currentColor"/>
            <rect x="10" y="2" width="6" height="6" rx="1.5" fill="currentColor"/>
            <rect x="2" y="10" width="6" height="6" rx="1.5" fill="currentColor"/>
            <rect x="10" y="10" width="6" height="6" rx="1.5" fill="currentColor"/>
          </svg>
          <!-- Broadcast icon (Stream) -->
          <svg v-else-if="item.icon === 'broadcast'" width="18" height="18" viewBox="0 0 18 18" fill="none">
            <circle cx="9" cy="9" r="2.5" fill="currentColor"/>
            <path d="M5.5 5.5C4.3 6.7 4.3 11.3 5.5 12.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M12.5 5.5C13.7 6.7 13.7 11.3 12.5 12.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M3 3C1 5 1 13 3 15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
            <path d="M15 3C17 5 17 13 15 15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
          </svg>
          <!-- Upload icon -->
          <svg v-else-if="item.icon === 'upload'" width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M9 2L9 11M9 2L6 5M9 2L12 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M3 13V14C3 15.1 3.9 16 5 16H13C14.1 16 15 15.1 15 14V13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          <!-- Settings icon -->
          <svg v-else-if="item.icon === 'settings'" width="18" height="18" viewBox="0 0 18 18" fill="none">
            <circle cx="9" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5"/>
            <path d="M9 2V3.5M9 14.5V16M3.5 6.1L4.75 6.85M13.25 11.15L14.5 11.9M2 9H3.5M14.5 9H16M3.5 11.9L4.75 11.15M13.25 6.85L14.5 6.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>

        <!-- Label (only on xl) -->
        <span class="hidden xl:block text-sm font-medium">{{ item.label }}</span>

        <!-- Live stream indicator -->
        <span
          v-if="item.name === 'stream' && streamActive"
          class="hidden xl:flex ml-auto"
        >
          <span class="live-dot" style="width:6px;height:6px;" />
        </span>
      </RouterLink>
    </nav>

    <!-- Connection status footer -->
    <div class="p-3 border-t border-[#1e2633]">
      <div class="flex items-center gap-2 px-2 py-2">
        <div
          class="w-2 h-2 rounded-full shrink-0 transition-colors duration-200"
          :class="spacetimeConnected ? 'bg-[#2dd4bf]' : 'bg-[#475569]'"
          :style="spacetimeConnected ? 'box-shadow: 0 0 6px rgba(45,212,191,0.5)' : ''"
        />
        <span class="hidden xl:block text-xs text-[#475569] truncate">
          {{ spacetimeConnected ? 'Connected' : 'Disconnected' }}
        </span>
      </div>
    </div>
  </aside>

  <!-- Mobile bottom tab bar -->
  <nav
    class="lg:hidden fixed bottom-0 left-0 right-0 z-50 flex items-center justify-around px-2 py-1 safe-b"
    style="background: rgba(15,17,23,0.95);
           backdrop-filter: blur(12px);
           border-top: 1px solid #1e2633;"
    role="navigation"
    aria-label="Mobile navigation"
  >
    <RouterLink
      v-for="item in navItems"
      :key="item.name"
      :to="item.path"
      class="flex flex-col items-center gap-0.5 px-4 py-2 rounded-lg transition-all duration-200"
      :class="isActive(item) ? 'text-[#2dd4bf]' : 'text-[#475569]'"
      :aria-current="isActive(item) ? 'page' : undefined"
    >
      <div class="w-5 h-5 flex items-center justify-center">
        <svg v-if="item.icon === 'grid'" width="18" height="18" viewBox="0 0 18 18" fill="none">
          <rect x="2" y="2" width="6" height="6" rx="1.5" fill="currentColor"/>
          <rect x="10" y="2" width="6" height="6" rx="1.5" fill="currentColor"/>
          <rect x="2" y="10" width="6" height="6" rx="1.5" fill="currentColor"/>
          <rect x="10" y="10" width="6" height="6" rx="1.5" fill="currentColor"/>
        </svg>
        <svg v-else-if="item.icon === 'broadcast'" width="18" height="18" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="2.5" fill="currentColor"/>
          <path d="M5.5 5.5C4.3 6.7 4.3 11.3 5.5 12.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M12.5 5.5C13.7 6.7 13.7 11.3 12.5 12.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <svg v-else-if="item.icon === 'upload'" width="18" height="18" viewBox="0 0 18 18" fill="none">
          <path d="M9 2L9 11M9 2L6 5M9 2L12 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M3 13V14C3 15.1 3.9 16 5 16H13C14.1 16 15 15.1 15 14V13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <svg v-else-if="item.icon === 'settings'" width="18" height="18" viewBox="0 0 18 18" fill="none">
          <circle cx="9" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5"/>
          <path d="M9 2V3.5M9 14.5V16M3.5 6.1L4.75 6.85M13.25 11.15L14.5 11.9M2 9H3.5M14.5 9H16M3.5 11.9L4.75 11.15M13.25 6.85L14.5 6.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <span class="text-[10px] font-medium">{{ item.label }}</span>
      <!-- Active indicator -->
      <div
        v-if="isActive(item)"
        class="w-1 h-1 rounded-full"
        style="background: #2dd4bf;"
      />
    </RouterLink>
  </nav>
</template>
