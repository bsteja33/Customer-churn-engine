import { test, expect, type Page, type Route } from "@playwright/test";

/**
 * Input-sensitivity tests for the analysis pipeline.
 *
 * The mock /predict handler inspects the submitted payload and returns a
 * different canned response per contract type, so every assertion here
 * proves the UI renders the response for THE REQUEST THAT WAS SENT -
 * not a cached, hardcoded, or stale result from an earlier step.
 */

const HIGH_RESPONSE = {
  prediction: 1,
  churn_probability: 0.8234,
  retention_risk: "High",
  feature_importance: [
    { feature: "Satisfaction_Score", value: 2, magnitude: 0.42, direction: "up" },
    { feature: "Tenure_in_Months", value: 2, magnitude: 0.18, direction: "up" },
  ],
};

const LOW_RESPONSE = {
  prediction: 0,
  churn_probability: 0.1147,
  retention_risk: "Low",
  feature_importance: [
    { feature: "Tenure_in_Months", value: 60, magnitude: 0.31, direction: "down" },
    { feature: "Contract_Two_Year", value: 1, magnitude: 0.22, direction: "down" },
  ],
};

/** Route /predict by the contract actually present in the request body. */
async function installContractSensitivePredictMock(page: Page) {
  await page.route("**/predict", async (route: Route) => {
    const payload = route.request().postDataJSON() as { Contract?: string };
    const body =
      payload.Contract === "Month-to-Month" ? HIGH_RESPONSE : LOW_RESPONSE;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

async function installHealthMocks(page: Page) {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "healthy",
        model_loaded: true,
        model_path: "models/churn_model.pkl",
        latency_ms: 42,
        fetched_at: new Date().toISOString(),
      }),
    });
  });
  await page.route("**/generate_retention_script", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        script: "[Action Plan] Follow the standard retention playbook.",
      }),
    });
  });
  await page.route("**/llm/models", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: {
          standard: "Standard insights - medium-intelligence model",
          high_capacity: "Deep analysis - highest-intelligence model",
        },
        default: "standard",
      }),
    });
  });
}

test.describe("Analysis input sensitivity", () => {
  test.beforeEach(async ({ page }) => {
    await installHealthMocks(page);
    await installContractSensitivePredictMock(page);
  });

  test("gauge, tier, and SHAP panel track the contract that was submitted", async ({
    page,
  }) => {
    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });

    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 82\.3 percent/ })
    ).toBeVisible();
    await expect(page.getByText("High", { exact: true })).toBeVisible();
    await expect(page.getByText("Satisfaction_Score").first()).toBeVisible();
    await expect(page.getByText("Contract_Two_Year")).toBeHidden();

    // Flip the single most influential field and re-run. The UI must
    // re-render from the NEW response (11.5 / Low), not the old one.
    await page.getByRole("button", { name: "New Analysis" }).click();
    await page.waitForURL("**/parameters");
    await page.locator("#field-Contract").selectOption("Two Year");
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });

    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 11\.5 percent/ })
    ).toBeVisible();
    await expect(page.getByText("Low", { exact: true })).toBeVisible();
    await expect(page.getByText("Tenure_in_Months").first()).toBeVisible();
    await expect(page.getByText("Satisfaction_Score")).toBeHidden();
  });

  test("identical input is re-requested and the fresh response is rendered", async ({
    page,
  }) => {
    // Same payload both times, but the mock answers High then Low. If the
    // UI served a cached result on the second run, the gauge would still
    // read 82.3 percent.
    const responses = [HIGH_RESPONSE, LOW_RESPONSE];
    let calls = 0;
    await page.unroute("**/predict");
    await page.route("**/predict", async (route: Route) => {
      const body = responses[Math.min(calls, responses.length - 1)];
      calls += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    });

    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 82\.3 percent/ })
    ).toBeVisible();

    await page.getByRole("button", { name: "New Analysis" }).click();
    await page.waitForURL("**/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 11\.5 percent/ })
    ).toBeVisible();
    expect(calls).toBeGreaterThanOrEqual(2);
  });

  test("the action plan panel renders the latest script, not the first one", async ({
    page,
  }) => {
    const scripts = [
      "[Action Plan] Apply a 20 percent loyalty discount today.",
      "[Action Plan] Schedule a personal check-in call this week.",
    ];
    let scriptCalls = 0;
    await page.unroute("**/generate_retention_script");
    await page.route("**/generate_retention_script", async (route) => {
      const script = scripts[Math.min(scriptCalls, scripts.length - 1)];
      scriptCalls += 1;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ script }),
      });
    });

    await page.goto("/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });
    await expect(
      page.getByText("Apply a 20 percent loyalty discount today.")
    ).toBeVisible();

    await page.getByRole("button", { name: "New Analysis" }).click();
    await page.waitForURL("**/parameters");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();
    await page.waitForURL("**/analysis", { timeout: 15000 });
    await expect(
      page.getByText("Schedule a personal check-in call this week.")
    ).toBeVisible();
    await expect(
      page.getByText("Apply a 20 percent loyalty discount today.")
    ).toBeHidden();
  });
});
