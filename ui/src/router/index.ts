import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'dashboard',
    component: () => import('@/pages/DashboardPage.vue'),
    meta: { title: 'Dashboard' },
  },
  {
    path: '/stream',
    name: 'stream',
    component: () => import('@/pages/StreamPage.vue'),
    meta: { title: 'Stream' },
  },
  {
    path: '/stream/:sessionId',
    name: 'stream-session',
    component: () => import('@/pages/StreamPage.vue'),
    meta: { title: 'Active Stream' },
  },
  {
    path: '/runs/:runId',
    name: 'run-detail',
    component: () => import('@/pages/RunDetailPage.vue'),
    meta: { title: 'Run Detail' },
  },
  {
    path: '/upload',
    name: 'upload',
    component: () => import('@/pages/UploadPage.vue'),
    meta: { title: 'Upload' },
  },
  {
    path: '/tracing',
    name: 'tracing',
    component: () => import('@/pages/TracingPage.vue'),
    meta: { title: 'Tracing' },
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/pages/SettingsPage.vue'),
    meta: { title: 'Settings' },
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/',
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior(_to, _from, savedPosition) {
    if (savedPosition) return savedPosition
    return { top: 0 }
  },
})

router.afterEach(to => {
  const title = to.meta?.title as string | undefined
  document.title = title ? `${title} — Vidarax` : 'Vidarax'
})

export default router
