import { defineConfig, devices } from "@playwright/test";

/**
 * Release-gate config: runs the production certification suite against the
 * live Vercel deployment. Never used by CI (`npx playwright test` uses
 * playwright.config.ts; this file is opt-in via --config).
 */
export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /prod-audit\.spec\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "https://customer-churn-engine.vercel.app",
    trace: "retain-on-failure",
    video: "off",
  },
  timeout: 120_000,
  expect: { timeout: 15_000 },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
