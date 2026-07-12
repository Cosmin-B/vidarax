<script setup lang="ts">
import { onMounted } from 'vue'
import { RouterView } from 'vue-router'
import AppSidebar from '@/components/layout/AppSidebar.vue'
import AppHeader from '@/components/layout/AppHeader.vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

onMounted(() => {
  // Touch the store on mount so it initializes (secrets from sessionStorage,
  // endpoints from localStorage) and stays reactive.
  authStore.apiEndpoint
})
</script>

<template>
  <div
    class="flex h-screen overflow-hidden"
    style="background-color: #08090d;"
  >
    <!-- Sidebar -->
    <AppSidebar />

    <!-- Main area -->
    <div class="flex flex-col flex-1 min-w-0 overflow-hidden">
      <!-- Header -->
      <AppHeader />

      <!-- Page content -->
      <main
        class="flex-1 overflow-y-auto overflow-x-hidden"
        style="background-color: #08090d;"
      >
        <!-- Extra bottom padding on mobile so content clears tab bar -->
        <div class="min-h-full pb-20 lg:pb-0">
          <RouterView v-slot="{ Component, route }">
            <Transition
              name="page"
              mode="out-in"
              :duration="150"
            >
              <component :is="Component" :key="route.path" />
            </Transition>
          </RouterView>
        </div>
      </main>
    </div>
  </div>
</template>

<style>
/* Page transition */
.page-enter-active,
.page-leave-active {
  transition: opacity 150ms cubic-bezier(0.4, 0, 0.2, 1),
              transform 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

.page-enter-from {
  opacity: 0;
  transform: translateY(4px);
}

.page-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
