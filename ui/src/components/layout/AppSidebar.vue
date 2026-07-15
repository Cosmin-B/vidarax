<script setup lang="ts">
import { computed, type Component } from 'vue'
import { useRoute } from 'vue-router'
import { useStreamStore } from '@/stores/stream'
import { useEventsStore } from '@/stores/events'
import { LayoutDashboard, Radio, Upload, Settings, Activity, Wifi, WifiOff } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'

const route = useRoute()
const streamStore = useStreamStore()
const eventsStore = useEventsStore()

interface NavItem {
  name: string
  path: string
  icon: Component
  label: string
  badge?: string | null
}

const navItems: NavItem[] = [
  {
    name: 'dashboard',
    path: '/',
    icon: LayoutDashboard,
    label: 'Dashboard',
  },
  {
    name: 'stream',
    path: '/stream',
    icon: Radio,
    label: 'Stream',
  },
  {
    name: 'upload',
    path: '/upload',
    icon: Upload,
    label: 'Upload',
  },
  {
    name: 'tracing',
    path: '/tracing',
    icon: Activity,
    label: 'Tracing',
  },
  {
    name: 'settings',
    path: '/settings',
    icon: Settings,
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
        class="group flex min-h-11 items-center gap-3 px-2 py-2.5 rounded-[8px] transition-all duration-200 relative"
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
          <AnimatedIcon
            :icon="item.icon"
            :size="18"
            :stroke-width="1.75"
            :animation="item.name === 'stream' && streamActive ? 'glow-teal' : undefined"
            :active="isActive(item)"
          />
        </div>

        <!-- Label (only on xl) -->
        <span class="hidden xl:block text-sm font-medium">{{ item.label }}</span>

        <!-- Live stream indicator -->
        <span
          v-if="item.name === 'stream' && streamActive"
          class="hidden xl:flex ml-auto"
        >
          <AnimatedIcon
            :icon="Radio"
            :size="10"
            :stroke-width="2"
            animation="pulse"
            class="text-[#2dd4bf] icon-glow-teal"
          />
        </span>
      </RouterLink>
    </nav>

    <!-- Connection status footer -->
    <div class="p-3 border-t border-[#1e2633]">
      <div class="flex items-center gap-2 px-2 py-2">
        <AnimatedIcon
          :icon="spacetimeConnected ? Wifi : WifiOff"
          :size="14"
          :stroke-width="1.75"
          :animation="spacetimeConnected ? 'glow-teal' : undefined"
          :class="spacetimeConnected ? 'text-[#2dd4bf]' : 'text-[#475569]'"
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
      class="flex min-h-11 flex-col items-center justify-center gap-0.5 px-4 py-2 rounded-lg transition-all duration-200"
      :class="isActive(item) ? 'text-[#2dd4bf]' : 'text-[#475569]'"
      :aria-current="isActive(item) ? 'page' : undefined"
    >
      <div class="w-5 h-5 flex items-center justify-center">
        <AnimatedIcon
          :icon="item.icon"
          :size="18"
          :stroke-width="1.75"
          :animation="item.name === 'stream' && streamActive ? 'glow-teal' : undefined"
          :active="isActive(item)"
        />
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
