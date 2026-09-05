import { test, expect, type Page } from "@playwright/test";

/**
 * Extended end-to-end coverage for every page and the provider panel.
 *
 * All network traffic is intercepted at the Playwright route layer so the
 * suite exercises the real UI against the real API contract (routes,
 * payload shapes, tag parsing) without needing a live backend or a live
 * LLM provider. The mock bodies mirror the backend responses exactly:
 *
 * - /health          -> {status, model_loaded, model_path, latency_ms, fetched_at}
 * - /predict         -> PredictResponse (prediction, churn_probability, retention_risk, feature_importance)
 * - /generate_retention_script -> {script} with the "[Action Plan]"/"[Default Action Plan]" tag
 * - /llm/models      -> tier descriptors only; provider model ids are never exposed
 */

const MOCK_HEALTH_RESPONSE = {
  status: "healthy",
  model_loaded: true,
  model_path: "models/churn_model.pkl",
  latency_ms: 42,
  fetched_at: new Date().toISOString(),
};

const MOCK_PREDICT_RESPONSE = {
  prediction: 1,
  churn_probability: 0.8234,
  retention_risk: "High",
  feature_importance: [
    { feature: "Satisfaction_Score", value: 2, magnitude: 0.42, direction: "up" },
    { feature: "Tenure_in_Months", value: 12, magnitude: 0.18, direction: "down" },
    { feature: "Contract_Two_Year", value: 0, magnitude: 0.11, direction: "up" },
  ],
};

const MOCK_SCRIPT_RESPONSE = {
  script:
    "[Action Plan] Thank you for your loyalty. Let me apply a 10% discount to your next bill.",
};

const MOCK_MODELS_RESPONSE = {
  models: {
    standard:
      "Standard insights - medium-intelligence model, selected automatically",
    high_capacity:
      "Deep analysis - highest-intelligence model available, selected automatically",
  },
  default: "standard",
};

/** Install the happy-path mocks shared by most tests. */
async function installBaseMocks(page: Page) {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_HEALTH_RESPONSE),
    });
  });
  await page.route("**/predict", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_PREDICT_RESPONSE),
    });
  });
  await page.route("**/generate_retention_script", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_SCRIPT_RESPONSE),
    });
  });
  await page.route("**/llm/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_MODELS_RESPONSE),
    });
  });
}

/** Drive the canonical happy path: parameters -> preset -> analysis. */
async function runFlow(page: Page) {
  await page.goto("/parameters");
  await expect(page.locator("h1")).toContainText("Input Engine");
  await page.getByRole("button", { name: "High-Risk Profile" }).click();
  await page.getByRole("button", { name: "Analyze" }).click();
  await page.waitForURL("**/analysis", { timeout: 15000 });
  await expect(page.locator("h1")).toContainText("Results Terminal");
}

test.describe("System Status page", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("shows connection status, latency, and model path", async ({ page }) => {
    await page.goto("/status");
    await expect(page.locator("h1")).toContainText("System Status");
    await expect(page.getByText("Status: healthy")).toBeVisible();
    // latency_ms is measured client-side by the health store, so assert the
    // shape rather than a fixed value from the mock.
    await expect(page.getByText(/Latency: \d+ms/)).toBeVisible();

    await expect(page.getByText("models/churn_model.pkl")).toBeVisible();
  });

  test("shows Offline with the failure message when the backend errors", async ({ page }) => {
    await page.unroute("**/health");
    await page.route("**/health", async (route) => route.abort());
    await page.goto("/status");
    await expect(page.getByText("Offline", { exact: true })).toBeVisible();
  });

  test("Refresh button refetches health", async ({ page }) => {
    let healthCalls = 0;
    await page.unroute("**/health");
    await page.route("**/health", async (route) => {
      healthCalls += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_HEALTH_RESPONSE),
      });
    });
    await page.goto("/status");
    await expect(page.getByText("Status: healthy")).toBeVisible();
    const before = healthCalls;
    await page.getByLabel("Refresh health").click();
    await expect(page.getByText("Status: healthy")).toBeVisible();
    expect(healthCalls).toBeGreaterThan(before);
  });

  test("Back to Engine link returns home", async ({ page }) => {
    await page.goto("/status");
    await page.getByText("← Back to Engine").click();
    await page.waitForURL("**/");
    await expect(page.locator("h1")).toContainText("Enterprise Churn Engine");
  });
});

test.describe("Parameters (input engine) page", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("renders field groups and the sticky action bar", async ({ page }) => {
    await page.goto("/parameters");
    await expect(page.locator("h1")).toContainText("Input Engine");
    await expect(page.getByText("Contract", { exact: true })).toBeVisible();
    await expect(page.getByText("Charges & Usage")).toBeVisible();
    await expect(page.getByRole("button", { name: "Analyze" })).toBeVisible();
  });

  test("groups can be collapsed and re-expanded", async ({ page }) => {
    await page.goto("/parameters");
    const groupToggle = page.getByRole("button", { name: /Charges & Usage/ });
    await expect(groupToggle).toHaveAttribute("aria-expanded", "true");
    await groupToggle.click();
    await expect(groupToggle).toHaveAttribute("aria-expanded", "false");
    await groupToggle.click();
    await expect(groupToggle).toHaveAttribute("aria-expanded", "true");
  });

  test("empty submission is blocked by validation and never navigates", async ({ page }) => {
    await page.goto("/parameters");
    await page.getByRole("button", { name: "Analyze" }).click();
    await expect(page.locator('[aria-invalid="true"]').first()).toBeVisible();
    await expect(page).not.toHaveURL(/analysis/);
  });

  test("High-Risk preset populates the numeric fields", async ({ page }) => {
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("2");
    await expect(page.locator("#field-tenure")).toHaveValue("2");
    await expect(page.locator("#field-Contract")).toHaveValue("Month-to-Month");
  });

  test("Reset clears previously loaded preset values", async ({ page }) => {
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("2");
    await page.getByRole("button", { name: "Reset" }).click();
    await expect(page.locator("#field-SatisfactionScore")).toHaveValue("");
    await expect(page.locator("#field-Contract")).toHaveValue("");
  });

  test("edited field values are sent to the predict endpoint", async ({ page }) => {
    // Holder object: a plain `let` gets control-flow-narrowed to `null`
    // inside the route callback and trips `tsc` (TS2352).
    const captured: { payload: Record<string, unknown> | null } = {
      payload: null,
    };
    await page.route("**/predict", async (route) => {
      captured.payload = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_PREDICT_RESPONSE),
      });
    });
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.locator("#field-Contract").selectOption("Two Year");
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });
    expect(captured.payload).not.toBeNull();
    expect(captured.payload?.Contract).toBe("Two Year");
  });
});

test.describe("Command Center (home page)", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("renders hero, tagline, and health cards", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h1")).toContainText("Enterprise Churn Engine");
    await expect(page.getByText("Triage, explain, and act on churn risk.")).toBeVisible();
    await expect(page.getByText("API Engine")).toBeVisible();
    await expect(page.getByText("Predictive Model")).toBeVisible();
  });

  test("shows both quickstart presets", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("High-Risk Profile")).toBeVisible();
    await expect(page.getByText("Loyal Profile")).toBeVisible();
  });

  test("Enter Engine navigates to the parameters page", async ({ page }) => {
    await page.goto("/");
    await page.getByText("Enter Engine").click();
    await page.waitForURL("**/parameters");
    await expect(page.locator("h1")).toContainText("Input Engine");
  });

  test("System Status link navigates to the status page", async ({ page }) => {
    await page.goto("/");
    await page.getByText("System Status").click();
    await page.waitForURL("**/status");
    await expect(page.locator("h1")).toContainText("System Status");
  });

  test("health cards report Connected + Loaded when the backend is healthy", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Connected", { exact: true })).toBeVisible();
    await expect(page.getByText("Loaded", { exact: true })).toBeVisible();
  });

  test("health cards report Disconnected + Offline when the backend is unreachable", async ({ page }) => {
    await page.unroute("**/health");
    await page.route("**/health", async (route) => route.abort());
    await page.goto("/");
    await expect(page.getByText("Disconnected")).toBeVisible();
    await expect(page.getByText("Offline").first()).toBeVisible();
  });

  test("Predictive Model shows Offline when the model is not loaded", async ({ page }) => {
    await page.unroute("**/health");
    await page.route("**/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ...MOCK_HEALTH_RESPONSE, model_loaded: false }),
      });
    });
    await page.goto("/");
    await expect(page.getByText("Offline")).toBeVisible();
    await expect(page.getByText("Loaded", { exact: true })).toBeHidden();
  });
});

test.describe("Parameters (input engine) page - request lifecycle", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("shows Processing... while the prediction request is in flight", async ({ page }) => {
    await page.route("**/predict", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_PREDICT_RESPONSE),
      });
    });
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await expect(page.getByText("Processing...")).toBeVisible();
    await page.waitForURL("**/analysis", { timeout: 15000 });
  });

  test("surfaces the backend detail message on a 429 rate-limit response", async ({ page }) => {
    await page.route("**/predict", async (route) => {
      await route.fulfill({
        status: 429,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Rate limit exceeded. Try again shortly." }),
      });
    });
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await expect(
      page.getByText("Rate limit exceeded. Try again shortly.")
    ).toBeVisible({ timeout: 10000 });
    await expect(page).not.toHaveURL(/analysis/);
  });
});

test.describe("Analysis (results terminal) page", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("renders gauge, risk classification, and the LLM badge after a full flow", async ({ page }) => {
    await runFlow(page);
    await expect(page.getByText("High", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 82\.3 percent/ })
    ).toBeVisible();
    await expect(page.getByText("Risk Classification")).toBeVisible();
    await expect(page.getByText("LLM", { exact: true })).toBeVisible();
  });

  test("lists the top SHAP drivers", async ({ page }) => {
    await runFlow(page);
    await expect(page.getByText("Satisfaction_Score").first()).toBeVisible();
    await expect(page.getByText("Tenure_in_Months").first()).toBeVisible();
  });

  test("shows the Default badge when the backend fell back", async ({ page }) => {
    await page.route("**/generate_retention_script", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          script:
            "[Default Action Plan] - Audit the customer's actual usage against the active plan tier.",
        }),
      });
    });
    await runFlow(page);
    await expect(page.getByText("Default", { exact: true })).toBeVisible();
    await expect(page.getByText("LLM", { exact: true })).toBeHidden();
  });

  test("renders the degraded message when script generation fails entirely", async ({ page }) => {
    await page.route("**/generate_retention_script", async (route) =>
      route.abort()
    );
    await runFlow(page);
    await expect(page.getByText("Failed to generate script.")).toBeVisible();
    await expect(page.getByText("LLM", { exact: true })).toBeHidden();
  });

  test("offers a Copy plan action and a submitted-inputs audit line", async ({ page }) => {
    await runFlow(page);
    await expect(page.getByText("Copy plan")).toBeVisible();
    await expect(page.getByText(/field(s)? persisted from the input engine/)).toBeVisible();
  });
});



test.describe("Provider configuration panel", () => {
  test.beforeEach(async ({ page }) => { await installBaseMocks(page); });

  test("opens from the rail and renders tier-only model options", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    const dialog = page.getByRole("dialog", { name: "Provider configuration" });
    await expect(dialog).toBeVisible();
    const options = page.locator("#provider-model option");
    await expect(options).toHaveCount(2);
    await expect(options.filter({ hasText: "Standard insights" })).toHaveCount(1);
    await expect(options.filter({ hasText: "Deep analysis" })).toHaveCount(1);
    // Tier-only contract: no provider model ids may leak into the UI.
    const allOptionText = await options.allTextContents();
    expect(allOptionText.join(" ")).not.toMatch(/gpt|llama|groq|mixtral/i);
  });

  test("syncs the catalog from the backend and confirms it", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    await expect(page.getByText("Catalog synced from backend.")).toBeVisible();
  });

  test("shows an inline error when the catalog endpoint fails", async ({ page }) => {
    await page.route("**/llm/models", async (route) => route.abort());
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    // route.abort() surfaces as a network-level fetch failure; the exact
    // message differs per browser ("Failed to fetch" in Chromium,
    // "NetworkError when attempting to fetch resource." in Firefox).
    await expect(
      page.getByText(
        /Failed to fetch|NetworkError|Catalog unavailable|Model catalog failed/
      )
    ).toBeVisible();


  });

  test("saving a key activates it and updates the rail indicator", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    await page.locator("#provider-key").fill("gsk_test_key_123");
    await page.getByRole("button", { name: "Save" }).click();
    await expect(
      page.getByRole("dialog", { name: "Provider configuration" })
    ).toBeHidden();
    await expect(page.getByText("Key", { exact: true })).toBeVisible();
    // Re-opening shows the active state without the key leaking.
    await page.getByLabel("Open provider configuration").click();
    await expect(page.getByText("Key active for this session")).toBeVisible();
    await expect(page.locator("#provider-key")).toHaveValue("gsk_test_key_123");
  });

  test("Clear removes the key and returns the rail button to API state", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    await page.locator("#provider-key").fill("gsk_test_key_123");
    await page.getByRole("button", { name: "Save" }).click();
    await expect(page.getByText("Key", { exact: true })).toBeVisible();

    await page.getByLabel("Open provider configuration").click();
    await page.getByRole("button", { name: "Clear" }).click();
    await page.getByRole("button", { name: "Save" }).click();
    await expect(page.getByText("API", { exact: true })).toBeVisible();
  });

  test("backdrop click closes the dialog without saving", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    await expect(
      page.getByRole("dialog", { name: "Provider configuration" })
    ).toBeVisible();
    await page.mouse.click(10, 360);
    await expect(
      page.getByRole("dialog", { name: "Provider configuration" })
    ).toBeHidden();
  });

  test("switching the model tier persists for the session", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Open provider configuration").click();
    await page.locator("#provider-model").selectOption("high_capacity");
    await page.getByRole("button", { name: "Save" }).click();
    await page.getByLabel("Open provider configuration").click();
    await expect(page.locator("#provider-model")).toHaveValue("high_capacity");
  });
});
