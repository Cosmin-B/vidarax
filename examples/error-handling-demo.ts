/**
 * Vidarax Error Handling Demo: typed errors instead of string matching
 *
 * Run: VIDARAX_API_KEY=your-key npx tsx examples/error-handling-demo.ts
 *
 * Every failure the SDK throws is a VidaraxError subclass, so callers can
 * branch on instanceof (or the exported type guards) rather than parsing
 * message strings. This script triggers a real 404 and a real connection
 * failure and shows the shape of each.
 */

import {
  Vidarax,
  HttpError,
  NetworkError,
  RetryExhaustedError,
  isNetworkError,
} from '../packages/vidarax-sdk/src/index.js'

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
  const v = new Vidarax(API_URL, API_KEY !== undefined ? { apiKey: API_KEY } : {})
  console.log('🔌 Connected to', API_URL)

  // ── 2. HttpError: a 404 from a well-formed but missing run id ──────
  // The id must match run-<16 or 32 hex chars>. A malformed id would be
  // a 422 validation error instead of a 404, so use a valid shape here.
  console.log('\n🕳️  Fetching a run that does not exist...')
  try {
    await v.getRun('run-ffffffffffffffff')
    console.log('   (unexpected: the run exists)')
  } catch (err) {
    if (err instanceof HttpError && err.isNotFound) {
      // 4xx client errors are never retried, so the HttpError arrives
      // directly. err.code comes from the server's JSON error envelope
      // and err.apiError carries the full body, including field details.
      console.log('   ✅ HttpError caught')
      console.log(`   status=${err.status} code=${err.code}`)
      console.log(`   message="${err.apiError?.message}"`)
      // Sibling convenience getters: isClientError, isServerError,
      // isValidationError (422), isConflict (409).
    } else {
      throw err
    }
  }

  // ── 3. NetworkError: a client aimed at an unreachable port ─────────
  // Nothing listens on port 9, so fetch itself rejects and the SDK wraps
  // that in a NetworkError. Retried methods run through the retry loop
  // even with maxRetries: 0 (a single attempt), so the loop's final
  // throw is a RetryExhaustedError whose lastError is that NetworkError.
  // A raw NetworkError only surfaces from the single-shot paths that
  // bypass the retry loop: whipOffer, whipIce, and file upload.
  console.log('\n🔌 Calling a server that is not there...')
  const unreachable = new Vidarax('http://127.0.0.1:9', { maxRetries: 0, timeoutMs: 2000 })
  try {
    await unreachable.health()
    console.log('   (unexpected: something answered on port 9)')
  } catch (err) {
    if (err instanceof RetryExhaustedError) {
      console.log('   ✅ RetryExhaustedError caught')
      console.log(`   attempts=${err.attempts}`)
      if (isNetworkError(err.lastError)) {
        // The connection-level failure, with the underlying fetch error
        // preserved on .cause for logging.
        console.log(`   lastError is a NetworkError: "${err.lastError.message}"`)
      }
    } else if (err instanceof NetworkError) {
      // Reached only from the non-retried paths described above.
      console.log('   ✅ NetworkError caught:', err.message)
    } else {
      throw err
    }
  }

  // With maxRetries at its default of 3, the same unreachable call makes
  // 4 attempts with exponential back-off before the RetryExhaustedError
  // surfaces. Retryable HTTP failures (429 and 5xx) end the same way,
  // with an HttpError as lastError instead. HttpError above and this
  // block together cover every remote failure a caller needs to branch
  // on. The remaining exports, UploadError and ParseError, are thrown by
  // file upload and by malformed response bodies.

  console.log('\n✨ Demo complete!')
}

main().catch(err => {
  console.error('❌', err.message || err)
  process.exit(1)
})
