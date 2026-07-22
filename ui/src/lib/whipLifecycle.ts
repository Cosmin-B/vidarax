export interface DurableWhipSession {
  runId: string
  sessionId: string
}

/**
 * Preserve a run before releasing its WHIP transport. Each operation is
 * best-effort, but the ordering is strict: a failed run-stop attempt settles
 * before transport termination begins.
 */
export async function stopDurableWhipSession(
  session: DurableWhipSession,
  stopRun: (runId: string) => Promise<unknown>,
  terminateWhip: (sessionId: string) => Promise<unknown>,
): Promise<void> {
  if (session.runId) {
    try { await stopRun(session.runId) } catch { /* best-effort */ }
  }
  if (session.sessionId) {
    try { await terminateWhip(session.sessionId) } catch { /* best-effort */ }
  }
}
