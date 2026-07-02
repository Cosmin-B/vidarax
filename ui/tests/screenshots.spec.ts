/**
 * Screenshot capture spec — saves PNGs to /screenshots/ for every page and
 * interactive UI state (collapsible panels, modals, drawers, tab variants).
 *
 * Run with:
 *   npx playwright test screenshots.spec.ts --project=desktop
 */

import { test, expect } from '@playwright/test'
import path from 'path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const SCREENSHOTS_DIR = path.join(__dirname, '..', '..', 'screenshots')

function shot(name: string) {
  return path.join(SCREENSHOTS_DIR, `${name}.png`)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Wait for the page to settle (no network, no spinners). */
async function waitForIdle(page: import('@playwright/test').Page, ms = 800) {
  await page.waitForTimeout(ms)
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

test('page-dashboard', async ({ page }) => {
  await page.goto('/')
  await waitForIdle(page)
  await page.screenshot({ path: shot('page-dashboard'), fullPage: true })
})

// ─── Stream page (setup mode) ─────────────────────────────────────────────────

test('page-stream', async ({ page }) => {
  await page.goto('/stream')
  await waitForIdle(page)
  await page.screenshot({ path: shot('page-stream'), fullPage: true })
})

test('panel-stream-advanced-settings', async ({ page }) => {
  await page.goto('/stream')
  await waitForIdle(page)

  // Click the "Advanced Settings" toggle
  const toggle = page.locator('.sp-advanced-toggle')
  await toggle.click()
  await waitForIdle(page, 400)

  await page.screenshot({ path: shot('panel-stream-advanced-settings'), fullPage: true })
})

test('panel-stream-advanced-settings-json-output', async ({ page }) => {
  await page.goto('/stream')
  await waitForIdle(page)

  // Open Advanced Settings
  await page.locator('.sp-advanced-toggle').click()
  await waitForIdle(page, 300)

  // Switch to JSON output mode to reveal the JSON Schema editor
  const jsonBtn = page.locator('.sp-toggle-group[aria-label="Output format"] .sp-toggle-btn', {
    hasText: 'JSON',
  })
  await jsonBtn.click()
  await waitForIdle(page, 300)

  await page.screenshot({ path: shot('panel-stream-advanced-settings-json-output'), fullPage: true })
})

test('panel-stream-advanced-settings-frame-mode', async ({ page }) => {
  await page.goto('/stream')
  await waitForIdle(page)

  // Open Advanced Settings
  await page.locator('.sp-advanced-toggle').click()
  await waitForIdle(page, 300)

  // Switch processing mode to Frame
  const frameBtn = page.locator('.sp-toggle-group[aria-label="Processing mode"] .sp-toggle-btn', {
    hasText: 'Frame',
  })
  await frameBtn.click()
  await waitForIdle(page, 300)

  await page.screenshot({ path: shot('panel-stream-advanced-settings-frame-mode'), fullPage: true })
})

// ─── Stream page — Export SDK modal ──────────────────────────────────────────
// The Export button is only visible in streaming mode; we force-show the modal
// via page.evaluate so we don't need an actual WHIP session.

test('modal-stream-export-sdk', async ({ page }) => {
  await page.goto('/stream')
  await waitForIdle(page)

  // Forcibly set the showExportModal ref via Vue's __vue_app__ devtools hook
  // so we don't need a live stream to trigger it.
  await page.evaluate(() => {
    // Walk the component tree to find StreamPage and set showExportModal
    function findComponent(vnode: any, name: string): any {
      if (!vnode) return null
      if (vnode.type?.name === name || vnode.type?.__name === name) return vnode
      const children = vnode.component?.subTree
      if (children) {
        const found = findComponent(children, name)
        if (found) return found
      }
      if (Array.isArray(vnode.children)) {
        for (const child of vnode.children) {
          const found = findComponent(child, name)
          if (found) return found
        }
      }
      return null
    }

    const app = (document.querySelector('#app') as any).__vue_app__
    const root = app._instance?.subTree

    function walkTree(vnode: any): any {
      if (!vnode) return null
      const ctx = vnode.component?.exposed ?? vnode.component?.setupState
      if (ctx && 'showExportModal' in ctx) return vnode.component
      const sub = vnode.component?.subTree
      if (sub) {
        const r = walkTree(sub)
        if (r) return r
      }
      const kids = Array.isArray(vnode.children) ? vnode.children : []
      for (const k of kids) {
        const r = walkTree(k)
        if (r) return r
      }
      return null
    }

    const comp = walkTree(root)
    if (comp?.setupState?.showExportModal !== undefined) {
      comp.setupState.showExportModal.value = true
    }
  })

  await waitForIdle(page, 500)

  // Verify the modal appeared; if the Vue approach didn't work, skip gracefully
  const modal = page.locator('.sp-modal-backdrop')
  const visible = await modal.isVisible().catch(() => false)
  if (visible) {
    await page.screenshot({ path: shot('modal-stream-export-sdk'), fullPage: false })
  } else {
    // Fallback: screenshot the page as-is with a note
    await page.screenshot({ path: shot('modal-stream-export-sdk-unavailable'), fullPage: true })
  }
})

// ─── Upload page ──────────────────────────────────────────────────────────────

test('page-upload', async ({ page }) => {
  await page.goto('/upload')
  await waitForIdle(page)
  await page.screenshot({ path: shot('page-upload'), fullPage: true })
})

// ─── Tracing / Pipeline Observability ────────────────────────────────────────

test('page-tracing', async ({ page }) => {
  await page.goto('/tracing')
  // Tracing page has a 2s poll; wait for the first data render
  await waitForIdle(page, 2500)
  await page.screenshot({ path: shot('page-tracing'), fullPage: true })
})

// ─── Settings ─────────────────────────────────────────────────────────────────

test('page-settings', async ({ page }) => {
  await page.goto('/settings')
  await waitForIdle(page)
  await page.screenshot({ path: shot('page-settings'), fullPage: true })
})

// ─── Run detail page ─────────────────────────────────────────────────────────
// The run detail page needs a real run ID from the API; we navigate to a
// non-existent ID so we can capture the error/empty states that are always
// reachable, and then we try to grab a real run from the dashboard list.

test('page-run-detail-error-state', async ({ page }) => {
  await page.goto('/runs/nonexistent-run-id-for-screenshot')
  await waitForIdle(page, 1500)
  await page.screenshot({ path: shot('page-run-detail-error-state'), fullPage: true })
})

test('page-run-detail-with-tabs', async ({ page }) => {
  // First load dashboard to find an existing run ID
  await page.goto('/')
  await waitForIdle(page, 1200)

  // Try clicking the first run row if one exists
  const firstRun = page.locator('.divide-y button').first()
  const exists = await firstRun.isVisible().catch(() => false)

  if (exists) {
    await firstRun.click()
    await waitForIdle(page, 1500)
    await page.screenshot({ path: shot('page-run-detail-player-tab'), fullPage: true })

    // Click Keyframes tab
    const kfTab = page.locator('button', { hasText: 'Keyframes' })
    if (await kfTab.isVisible().catch(() => false)) {
      await kfTab.click()
      await waitForIdle(page, 600)
      await page.screenshot({ path: shot('page-run-detail-keyframes-tab'), fullPage: true })
    }

    // Click Metadata tab
    const metaTab = page.locator('button', { hasText: 'Metadata' })
    if (await metaTab.isVisible().catch(() => false)) {
      await metaTab.click()
      await waitForIdle(page, 400)
      await page.screenshot({ path: shot('page-run-detail-metadata-tab'), fullPage: true })
    }

    // Show feedback panel (Rate this analysis button)
    const feedbackBtn = page.locator('button', { hasText: 'Rate this analysis' })
    if (await feedbackBtn.isVisible().catch(() => false)) {
      await feedbackBtn.click()
      await waitForIdle(page, 400)
      await page.screenshot({ path: shot('panel-run-detail-feedback'), fullPage: true })
    }

    // Show confirm-delete inline confirmation
    const deleteBtn = page.locator('button', { hasText: 'Delete' }).first()
    if (await deleteBtn.isVisible().catch(() => false)) {
      await deleteBtn.click()
      await waitForIdle(page, 300)
      await page.screenshot({ path: shot('panel-run-detail-confirm-delete'), fullPage: true })
      // Cancel so we don't destroy the run
      const cancelBtn = page.locator('button', { hasText: 'Cancel' })
      if (await cancelBtn.isVisible().catch(() => false)) {
        await cancelBtn.click()
      }
    }
  } else {
    // No runs exist — capture the empty dashboard
    await page.screenshot({ path: shot('page-run-detail-no-runs'), fullPage: true })
  }
})

// ─── Navigation / Layout ──────────────────────────────────────────────────────

test('layout-sidebar', async ({ page }) => {
  // Just capture the full app shell from the dashboard
  await page.goto('/')
  await waitForIdle(page)
  await page.screenshot({ path: shot('layout-sidebar'), fullPage: false })
})
