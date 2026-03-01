import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Run test suites (files) one at a time — prevents concurrent beforeAll
    // hooks from exceeding the server's 20 active run limit.
    sequence: {
      concurrent: false,
    },
    // Default timeouts (individual tests may override via the second argument)
    testTimeout: 30_000,
    hookTimeout: 30_000,
  },
});
