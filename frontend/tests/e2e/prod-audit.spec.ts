/**
 * Production certification suite.
 *
 * Runs against the LIVE deployment (baseURL in playwright.prod.config.ts).
 * NOT wired into CI: it depends on external uptime, shared rate limits,
 * and a live LLM provider key, so it is a release-gate tool, not a test.
 *
 * Rate-limit pacing: the backend allows 10 predicts/min and 5 script
 * generations/min. Each Analyze click costs one of each, and the analysis
 * page adds one extra baseline predict per fresh page load (5-min module
 * cache per tab). A 14 s gap keeps both buckets under their limits.
 */
import { test, expect, type Page } from "@playwright/test";

const PROD = "https://customer-churn-engine.vercel.app";

/** Global pacing gate shared across tests (workers=1 in the prod config). */
let lastAnalyzeAt = 0;
async function paceAnalyze(minGapMs = 14_000): Promise<void> {
  const wait = lastAnalyzeAt + minGapMs - Date.now();
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastAnalyzeAt = Date.now();
}

async function openParameters(page: Page): Promise<void> {
  await page.goto(`${PROD}/parameters`, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Input Engine" })).toBeVisible();
}

/** Load a preset and submit. Returns the gauge aria-label. */
async function analyzeWith(page: Page, presetLabel: string): Promise<string> {
  await openParameters(page);
  await applyPreset(page, presetLabel);
  await paceAnalyze();
  await page.getByRole("button", { name: "Analyze" }).click();
  await page.waitForURL("**/analysis", { timeout: 20_000 });
  const gauge = page.getByRole("img", { name: /Churn risk gauge reading/ });
  await expect(gauge).toBeVisible({ timeout: 15_000 });
  return (await gauge.getAttribute("aria-label")) ?? "";
}

/**
 * Click a preset and verify it actually landed. The SSR page silently
 * swallows clicks that arrive before React hydration finishes, so the
 * click is retried until the Age field reflects the preset (28 = high
 * risk, 54 = loyal). Keeps submit-dependent tests immune to slow
 * production loads.
 */
async function applyPreset(page: Page, presetLabel: string): Promise<void> {
  const presetBtn = page.getByRole("button", { name: presetLabel });
  const age = page.locator("#field-Age");
  await expect(async () => {
    await presetBtn.click();
    await expect(age).not.toHaveValue("", { timeout: 2_000 });
  }).toPass({ timeout: 20_000 });
}

function gaugePct(label: string): number {
  const m = label.match(/reading ([\d.]+) percent/);
  return m ? parseFloat(m[1]) : NaN;
}

async function setNumber(page: Page, key: string, value: string): Promise<void> {
  await page.fill(`#field-${key}`, value);
}

async function expectStillOnParameters(page: Page): Promise<void> {
  await page.waitForTimeout(1_500);
  expect(new URL(page.url()).pathname).toBe("/parameters");
}

test.describe("Production UI certification", () => {
  test.describe.configure({ mode: "serial" });

  test("01 landing page renders engine shell", async ({ page }) => {
    await page.goto(PROD, { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { level: 1 })).toContainText(
      "Enterprise Churn Engine"
    );
    await expect(page.getByText("Enter Engine")).toBeVisible();
  });

  test("02 landing navigates to input engine", async ({ page }) => {
    await page.goto(PROD, { waitUntil: "domcontentloaded" });
    await page.getByText("Enter Engine").click();
    await page.waitForURL("**/parameters");
    await expect(page.getByRole("heading", { name: "Input Engine" })).toBeVisible();
  });

  test("03 empty submission is blocked with field errors", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "Analyze" }).click();
    await expectStillOnParameters(page);
    const alerts = page.getByRole("alert");
    expect(await alerts.count()).toBeGreaterThan(0);
  });

  test("04 high-risk preset populates representative fields", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("2");
    await expect(page.locator("#field-tenure")).toHaveValue("2");
    await expect(page.locator("#field-Contract")).toHaveValue("Month-to-Month");
    await expect(page.locator("#field-InternetType")).toHaveValue("Fiber Optic");
  });

  test("05 high-risk profile predicts High risk", async ({ page }) => {
    const label = await analyzeWith(page, "High-Risk Profile");
    const pct = gaugePct(label);
    expect(pct).not.toBeNaN();
    expect(pct).toBeGreaterThan(50); // high-risk profile must read as risky
    await expect(page.getByText("Churn probability", { exact: true })).toBeVisible();
    await expect(page.getByText("Risk Classification")).toBeVisible();
  });

  test("06 SHAP panel and top-3 drivers render for a real prediction", async ({ page }) => {
    const label = await analyzeWith(page, "High-Risk Profile");
    expect(gaugePct(label)).not.toBeNaN();
    await expect(
      page.getByRole("heading", { name: "Feature Importance (SHAP)" })
    ).toBeVisible();
    await expect(page.getByText("Top 3 Drivers")).toBeVisible();
    const shapRows = page.locator(
      "section:has(h2:text('Feature Importance (SHAP)')) li"
    );
    expect(await shapRows.count()).toBeGreaterThan(0);
  });

  test("07 retention action plan renders with exactly one provenance badge", async ({ page }) => {
    const label = await analyzeWith(page, "High-Risk Profile");
    expect(gaugePct(label)).not.toBeNaN();
    await expect(page.getByText("Internal · CSM use only")).toBeVisible();
    const scriptBox = page.locator("pre");
    await expect(scriptBox).toBeVisible();
    const script = (await scriptBox.innerText()).trim();
    expect(script.length).toBeGreaterThan(0);
    // Exactly one provenance badge must be present (LLM or Default fallback).
    const hasLlm = await page.getByText("LLM", { exact: true }).count();
    const hasDefault = await page.getByText("Default", { exact: true }).count();
    expect(hasLlm + hasDefault).toBe(1);
    // Hallucination guard: no leaked placeholders / undefined / NaN.
    expect(script).not.toMatch(/undefined|NaN/);
  });

  test("08 loyal profile reads as low risk", async ({ page }) => {
    const loyal = gaugePct(await analyzeWith(page, "Loyal Profile"));
    expect(loyal).not.toBeNaN();
    expect(loyal).toBeLessThan(30);
  });

  test("09 boundary values (satisfaction 5, tenure 0, zero charges) accepted", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "Loyal Profile" }).click();
    await setNumber(page, "SatisfactionScore", "5");
    await setNumber(page, "tenure", "0");
    await setNumber(page, "MonthlyCharges", "0");
    await setNumber(page, "TotalCharges", "0");
    await paceAnalyze();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 20_000 });
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading/ })
    ).toBeVisible({ timeout: 15_000 });
  });

  test("10 SatisfactionScore 9 rejected client-side", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await setNumber(page, "SatisfactionScore", "9");
    await page.getByRole("button", { name: "Analyze" }).click();
    await expectStillOnParameters(page);
    await expect(page.locator("#field-SatisfactionScore-error")).toBeVisible();
  });

  test("11 negative tenure rejected client-side", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await setNumber(page, "tenure", "-5");
    await page.getByRole("button", { name: "Analyze" }).click();
    await expectStillOnParameters(page);
    await expect(page.locator("#field-tenure-error")).toBeVisible();
  });

  test("12 extreme CLTV does not crash or hang the UI", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await setNumber(page, "CLTV", "999999999999");
    await paceAnalyze();
    await page.getByRole("button", { name: "Analyze" }).click();
    // Either a result lands or a visible error appears; never a silent hang.
    const navigated = await page
      .waitForURL("**/analysis", { timeout: 20_000 })
      .then(() => true)
      .catch(() => false);
    if (navigated) {
      await expect(
        page.getByRole("img", { name: /Churn risk gauge reading/ })
      ).toBeVisible({ timeout: 15_000 });
    } else {
      await expect(page.locator("[role=alert]").first()).toBeVisible();
    }
  });

  test("13 identical inputs produce identical probability (determinism)", async ({ page }) => {
    const first = gaugePct(await analyzeWith(page, "High-Risk Profile"));
    const second = gaugePct(await analyzeWith(page, "High-Risk Profile"));
    expect(second).toBe(first);
  });

  test("14 analysis page never shows a stale previous result", async ({ page }) => {
    const loyalPct = gaugePct(await analyzeWith(page, "Loyal Profile"));
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await paceAnalyze();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 20_000 });
    const gauge = page.getByRole("img", { name: /Churn risk gauge reading/ });
    await expect(gauge).toBeVisible({ timeout: 15_000 });
    const highPct = gaugePct((await gauge.getAttribute("aria-label")) ?? "");
    expect(highPct).not.toBe(loyalPct);
    expect(highPct).toBeGreaterThan(50);
  });

  test("15 reset clears every field", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await expect(page.locator("#field-tenure")).toHaveValue("2");
    await page.getByRole("button", { name: "Reset" }).click();
    await expect(page.locator("#field-tenure")).toHaveValue("");
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("");
    await expect(page.locator("#field-Contract")).toHaveValue("");
  });

  test("16 applying the second preset fully overwrites the first", async ({ page }) => {
    await openParameters(page);
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Loyal Profile" }).click();
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("5");
    await expect(page.locator("#field-tenure")).toHaveValue("60");
    await expect(page.locator("#field-Contract")).toHaveValue("Two Year");
  });

  test("17 form values persist across analysis round-trip", async ({ page }) => {
    await analyzeWith(page, "High-Risk Profile");
    await page.getByText("← Back to Inputs").click();
    await page.waitForURL("**/parameters");
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("2");
    await expect(page.locator("#field-tenure")).toHaveValue("2");
  });

  test("18 direct /analysis visit without results redirects to /parameters", async ({ page }) => {
    await page.goto(`${PROD}/analysis`, { waitUntil: "domcontentloaded" });
    await page.waitForURL("**/parameters", { timeout: 15_000 });
    await expect(page.getByRole("heading", { name: "Input Engine" })).toBeVisible();
  });

  test("19 contract and offer options match the trained feature vocabulary", async ({ page }) => {
    await openParameters(page);
    const options = await page.locator("#field-Contract option").allInnerTexts();
    expect(options.filter((o) => o !== "-")).toEqual([
      "Month-to-Month",
      "One Year",
      "Two Year",
    ]);
    const offers = await page.locator("#field-Offer option").allInnerTexts();
    expect(offers.filter((o) => o !== "-")).toEqual([
      "None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E",
    ]);
  });

  test("20 keyboard Enter submits the form", async ({ page }) => {
    await openParameters(page);
    await applyPreset(page, "High-Risk Profile");
    await page.locator("#field-Age").click();
    await paceAnalyze();
    await page.keyboard.press("Enter");
    await page.waitForURL("**/analysis", { timeout: 20_000 });
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading/ })
    ).toBeVisible({ timeout: 15_000 });
  });

  test("21 mobile viewport completes the full flow", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const label = await analyzeWith(page, "High-Risk Profile");
    expect(gaugePct(label)).not.toBeNaN();
    await expect(page.getByText("Risk Classification")).toBeVisible();
  });

  test("22 rapid double-submit leaves a coherent single result", async ({ page }) => {
    // Prod latency stacks here: hydration retry window + 14s rate-limit
    // pacing + a baseline predict on analysis load. Give it room.
    test.setTimeout(240_000);
    await openParameters(page);
    await applyPreset(page, "High-Risk Profile");
    await paceAnalyze();
    const analyze = page.getByRole("button", { name: "Analyze" });
    // Two near-simultaneous clicks. The app navigates optimistically on
    // the first submit and unmounts the button, so the second click is
    // allowed to lose the race but it must fail FAST, not hang the
    // test waiting for an element that is gone (trace-verified: an
    // un-timed force click blocked for the full test timeout).
    await Promise.allSettled([
      analyze.click({ timeout: 5_000 }),
      analyze.click({ force: true, timeout: 5_000 }),
    ]);
    await page.waitForURL("**/analysis", { timeout: 30_000 });
    const gauge = page.getByRole("img", { name: /Churn risk gauge reading/ });
    // HF Space cold starts can make a single predict take >60s through
    // the Vercel rewrite; wait patiently rather than flaking.
    await expect(gauge).toBeVisible({ timeout: 120_000 });
    expect(await gauge.count()).toBe(1);
  });
});
