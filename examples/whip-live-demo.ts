/**
 * Vidarax WHIP Live Demo: the signalling flow for a live WebRTC session
 *
 * Run: VIDARAX_API_KEY=your-key npx tsx examples/whip-live-demo.ts
 *
 * Node cannot capture or encode media, so this script walks the signalling
 * calls only: offer/answer, trickle ICE, a live prompt update, and teardown.
 * The spots where a browser RTCPeerConnection or a GStreamer whipclientsink
 * would attach real media are marked MEDIA PLACEHOLDER. Until you paste a
 * real SDP offer below, the script prints the flow and exits, because the
 * server rejects an offer with no serveable video section.
 */

import { Vidarax, RetryExhaustedError, isHttpError } from '../packages/vidarax-sdk/src/index.js'

// No @types/node dependency here, so declare the ambient bits this script
// needs directly (same approach the SDK's own e2e test file uses).
declare const process: {
  env: Record<string, string | undefined>
  exit(code?: number): never
}

const API_URL = process.env['VIDARAX_API'] || 'http://localhost:8080'
const API_KEY = process.env['VIDARAX_API_KEY']

// ── MEDIA PLACEHOLDER: the SDP offer ─────────────────────────────────
// A real offer comes from whatever produces the media:
//
//   Browser:    const pc = new RTCPeerConnection({ iceServers: [] })
//               pc.addTrack(camera.getVideoTracks()[0], camera)
//               const offer = await pc.createOffer()
//               await pc.setLocalDescription(offer)
//               → use offer.sdp here
//
//   GStreamer:  gst-launch-1.0 videotestsrc ! x264enc tune=zerolatency \
//                 ! whipclientsink signaller::whip-endpoint=$API_URL/v1/stream/whip
//               (whipclientsink does the whole exchange itself, so this
//               script then only needs prompt-update and terminate calls)
const SDP_OFFER = 'PASTE_REAL_SDP_OFFER_HERE'

async function main() {
  // ── 1. Connect ─────────────────────────────────────────────────────
  const v = new Vidarax(API_URL, API_KEY !== undefined ? { apiKey: API_KEY } : {})
  console.log('🔌 Connected to', API_URL)

  if (SDP_OFFER === 'PASTE_REAL_SDP_OFFER_HERE') {
    console.log('\n⚠️  No SDP offer configured. Paste one into SDP_OFFER first.')
    console.log('   The flow below is what runs once you do:')
    console.log('   1. whipOffer(sdp, { prompt })  → sessionId, runId, answerSdp')
    console.log('   2. whipIce(sessionId, cand)    → forward each local ICE candidate')
    console.log('   3. whipUpdatePrompt(...)       → change the live prompt, no renegotiation')
    console.log('   4. whipTerminate(sessionId)    → close the session and its run')
    return
  }

  // ── 2. Offer / answer ──────────────────────────────────────────────
  // The attach config sets the initial prompt before workers start.
  // clip_mode and crop can also only be set here, not after start.
  const session = await v.whipOffer(SDP_OFFER, {
    prompt: 'Describe what is happening on screen.',
  })
  console.log('📡 Session:', session.sessionId)
  console.log('   Run:', session.runId ?? '(none)')

  // MEDIA PLACEHOLDER: apply the answer to the peer connection so media
  // starts flowing, e.g. in a browser:
  //   await pc.setRemoteDescription({ type: 'answer', sdp: session.answerSdp })

  try {
    // ── 3. Trickle ICE ───────────────────────────────────────────────
    // MEDIA PLACEHOLDER: forward each candidate the peer connection
    // gathers, e.g. pc.onicecandidate = e => e.candidate &&
    // v.whipIce(session.sessionId, e.candidate.candidate).
    // A candidate that fails to apply still returns 204, since the
    // connection may complete on other candidates.
    await v.whipIce(session.sessionId, 'candidate:1 1 udp 2130706431 192.0.2.1 50000 typ host')
    console.log('🧊 ICE candidate sent')

    // ── 4. Update the live prompt ────────────────────────────────────
    // The server forwards the update as a generation-tagged command and
    // answers 200 only after a VLM worker acknowledged it. Two failure
    // shapes matter here:
    //   409: the session's pipeline generation was closed or replaced,
    //        so the update was rejected. Stop sending updates.
    //   503: no worker acknowledgement within 2 seconds. The command
    //        was discarded server-side, so retrying is safe. The SDK
    //        retries 503 automatically, and only after those attempts
    //        are spent does it throw RetryExhaustedError wrapping the
    //        final HttpError.
    try {
      const updated = await v.whipUpdatePrompt(session.sessionId, {
        prompt: 'Watch for a person entering the frame.',
        output_schema: null,
      })
      console.log('✏️  Prompt now:', updated.prompt)
    } catch (err) {
      if (isHttpError(err) && err.isConflict) {
        console.log('⚠️  409: generation closed or replaced, update rejected')
      } else if (err instanceof RetryExhaustedError && isHttpError(err.lastError) && err.lastError.status === 503) {
        console.log('⚠️  503: worker acknowledgement timed out, command discarded. Retry later.')
      } else {
        throw err
      }
    }

    // MEDIA PLACEHOLDER: this is where the session would stream for a
    // while and VLM descriptions would land on the run's event timeline
    // (GET /v1/runs/{id}/events, or v.getEvents(session.runId)).
  } finally {
    // ── 5. Terminate ─────────────────────────────────────────────────
    await v.whipTerminate(session.sessionId)
    console.log('🗑️  Session terminated')
  }

  console.log('\n✨ Demo complete!')
}

main().catch(err => {
  console.error('❌', err.message || err)
  process.exit(1)
})
