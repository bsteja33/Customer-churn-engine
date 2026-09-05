import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    // Dedicated port: 3000 is commonly occupied by other dev servers,
    // and reuseExistingServer would silently test the wrong app.
    baseURL: "http://localhost:3100",
    trace: "on-first-retry",
  },
  // 60s per test: the dev server compiles each route on first request,
  // and cold compiles under parallel workers can exceed the 30s default
  // (observed as navigation timeouts on the Firefox project).
  timeout: 60_000,
  // Cold compiles in dev mode can push hydration past the 5s default
  // before the first health poll lands; 15s covers it without masking
  // real failures.
  expect: { timeout: 15_000 },

  webServer: {
    command: "npm run dev -- -p 3100",
    url: "http://localhost:3100",
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
  ],
});
