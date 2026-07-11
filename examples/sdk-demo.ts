/**
 * Vidarax SDK Demo — shows how easy it is to analyze video and stream events
 *
 * Run: VIDARAX_API_KEY=your-key npx tsx examples/sdk-demo.ts
 */

import { Vidarax } from '../packages/vidarax-sdk/src/index.js'

// No @types/node dependency here, so declare the ambient bits this script
// needs directly (same approach the SDK's own e2e test file uses).
declare const process: {
  env: Record<string, string | undefined>
  exit(code?: number): never
}

const API_URL = process.env['VIDARAX_API'] || 'http://localhost:8080'
const API_KEY = process.env['VIDARAX_API_KEY']

async function main() {
  // ── 1. Connect ─────────────────────────────────────────────────────
  // Built conditionally rather than `{ apiKey: API_KEY }` because the SDK's
  // exactOptionalPropertyTypes tsconfig rejects an explicit `undefined` for
  // an optional field — it must be omitted entirely when there's no key.
  const v = new Vidarax(API_URL, API_KEY !== undefined ? { apiKey: API_KEY } : {})
  console.log('🔌 Connected to', API_URL)

  // ── 2. Health check ────────────────────────────────────────────────
  const health = await v.health()
  console.log('💚 Health:', health.status)

  // ── 3. List available models ───────────────────────────────────────
  const models = await v.listModels()
  console.log(`🧠 ${models.length} models available:`)
  for (const m of models) {
    console.log(`   ${m.id} [${m.tier}] — ${m.availability}`)
  }

  // ── 4. Analyze a video with the realtime reason pipeline ───────────
  console.log('\n📹 Analyzing video...')
  const t0 = Date.now()

  const run = await v.createRun({ mode: 'balanced' })
  console.log(`   Run: ${run.run_id}`)

  // Use the test video on the server
  const result = await v.reason(run.run_id, {
    source_uri: 'file:///tmp/vidarax-e2e-test.mp4',
    model: 'Qwen/Qwen3-VL-2B-Instruct',
    mode: 'balanced',
    semantic_inference: true,
    semantic_prompt: 'Describe what is happening on screen. Identify UI elements and interactions.',
    chunk_size: 25,
    semantic_frames_per_chunk: 2,
  })

  const elapsed = Date.now() - t0
  console.log(`   ✅ Done in ${elapsed}ms`)
  console.log(`   Frames: ${result.decoded_frames}`)
  console.log(`   Markers: ${result.markers_emitted}`)
  console.log(`   Lag p95: ${result.lag_p95_ms}ms`)

  // ── 5. Stream events ───────────────────────────────────────────────
  console.log('\n📡 Events:')
  const events = await v.getEvents(run.run_id)
  for (const evt of events.slice(0, 10)) {
    console.log(`   ${evt.kind} [seq=${evt.seq}]`)
  }
  if (events.length > 10) {
    console.log(`   ... and ${events.length - 10} more`)
  }

  // ── 6. Get markers ─────────────────────────────────────────────────
  console.log('\n🎯 Markers:')
  const markers = await v.getMarkers(run.run_id)
  for (const m of markers.slice(0, 5)) {
    console.log(`   ${m.event_type} [${m.status}] frames ${m.start_frame}-${m.end_frame} conf=${(m.confidence * 100).toFixed(0)}%`)
  }

  // ── 7. Standalone inference ────────────────────────────────────────
  console.log('\n🤖 Quick inference:')
  const answer = await v.infer({
    model: 'Qwen/Qwen3-VL-2B-Instruct',
    prompt: 'What is 2+2? Just the number.',
    max_tokens: 8,
    temperature: 0.0,
    timeout_ms: 10000,
  })
  console.log(`   "${answer.result}"`)

  // ── 8. Search ──────────────────────────────────────────────────────
  console.log('\n🔍 Search:')
  try {
    const search = await v.search('scene', { limit: 3 })
    console.log(`   ${search.total_hits} hits (showing ${search.hits.length})`)
    for (const hit of search.hits) {
      console.log(`   [${hit.kind}] ${hit.description.slice(0, 60)}...`)
    }
  } catch {
    console.log('   (search not available on this server)')
  }

  // ── 9. Cleanup ─────────────────────────────────────────────────────
  await v.deleteRun(run.run_id)
  console.log(`\n🗑️  Run deleted`)
  console.log('\n✨ Demo complete!')
}

main().catch(err => {
  console.error('❌', err.message || err)
  process.exit(1)
})
