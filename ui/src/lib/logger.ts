/** Production-safe logger. Silent when import.meta.env.PROD is true. */

const noop = (..._args: unknown[]) => {}

export const logger = {
  info: import.meta.env.PROD ? noop : console.info.bind(console),
  warn: import.meta.env.PROD ? noop : console.warn.bind(console),
}
