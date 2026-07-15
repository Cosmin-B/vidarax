import { expect, test } from '@playwright/test'

test('keyframe replay is idempotent and keeps the per-run index consistent', async ({ page }) => {
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { useEventsStore, MAX_KEYFRAMES } = await import('/src/stores/events.ts')
    const store = useEventsStore()
    store.clearAll()

    const base = {
      id: 1,
      run_id: 'run-replay',
      frame_index: 10,
      pts_ms: 1000,
      event_type: 'scene_cut',
      description: 'first',
      image_sha256: 'aaa',
      image_url: 'blob:first',
      timestamp_ms: 100,
    }

    store.addKeyframe(base)
    store.addKeyframe({ ...base })
    const duplicateCount = store.keyframes.length
    const duplicateRunCount = store.keyframesForRun('run-replay').length

    store.addKeyframe({ ...base, description: 'updated', image_url: 'blob:updated' })
    const replaced = store.keyframesForRun('run-replay')[0]

    for (let id = 2; id <= MAX_KEYFRAMES + 1; id++) {
      store.addKeyframe({
        id,
        run_id: 'run-replay',
        frame_index: id,
        pts_ms: id * 100,
        event_type: 'scene_cut',
        description: `kf-${id}`,
        image_sha256: `sha-${id}`,
        image_url: `blob:${id}`,
        timestamp_ms: id * 1000,
      })
    }

    const runKeyframes = store.keyframesForRun('run-replay')
    return {
      duplicateCount,
      duplicateRunCount,
      replacedDescription: replaced?.description,
      replacedImageUrl: replaced?.image_url,
      total: store.keyframes.length,
      runTotal: runKeyframes.length,
      firstId: runKeyframes[0]?.id,
      lastId: runKeyframes[runKeyframes.length - 1]?.id,
      hasEvictedId: runKeyframes.some(kf => kf.id === 1),
    }
  })

  expect(result.duplicateCount).toBe(1)
  expect(result.duplicateRunCount).toBe(1)
  expect(result.replacedDescription).toBe('updated')
  expect(result.replacedImageUrl).toBe('blob:updated')
  expect(result.total).toBe(256)
  expect(result.runTotal).toBe(256)
  expect(result.firstId).toBe(2)
  expect(result.lastId).toBe(257)
  expect(result.hasEvictedId).toBe(false)
})

test('event replay does not scan the bounded event array for duplicates', async ({ page }) => {
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const { useEventsStore } = await import('/src/stores/events.ts')
    const store = useEventsStore()
    store.clearAll()

    let eventArrayFindCalls = 0
    const originalFind = store.events.find
    Object.defineProperty(store.events, 'find', {
      configurable: true,
      value: (...args: Parameters<typeof originalFind>) => {
        eventArrayFindCalls++
        return originalFind.apply(store.events, args)
      },
    })

    const event = {
      run_id: 'run-events',
      session_id: 'session-events',
      frame_index: 1,
      pts_ms: 100,
      event_type: 'scene_cut' as const,
      confidence: 0.9,
      description: 'first',
      timestamp_ms: 100,
    }

    store.addEvent(event)
    store.addEvent({ ...event })
    store.addEvent({ ...event, description: 'updated' })

    Object.defineProperty(store.events, 'find', {
      configurable: true,
      value: originalFind,
    })

    return {
      eventArrayFindCalls,
      total: store.events.length,
      runTotal: store.eventsForRun('run-events').length,
      description: store.eventsForRun('run-events')[0]?.description,
    }
  })

  expect(result.eventArrayFindCalls).toBe(0)
  expect(result.total).toBe(1)
  expect(result.runTotal).toBe(1)
  expect(result.description).toBe('updated')
})
