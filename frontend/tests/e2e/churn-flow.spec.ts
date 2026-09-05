import { test, expect } from "@playwright/test";

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

// The backend prefixes successful LLM output with "[Action Plan]" and
// the analysis page derives the badge from that tag; the fallback path
// uses "[Default Action Plan]". The mock must match the real contract.
const MOCK_SCRIPT_RESPONSE = {
  script:
    "[Action Plan] Thank you for your loyalty. Let me apply a 10% discount to your next bill.",
};

const MOCK_HEALTH_RESPONSE = {
  status: "healthy",
  model_loaded: true,
  model_path: "models/churn_model.pkl",
};

test.describe("Enterprise Churn Prediction Flow", () => {
  test.beforeEach(async ({ page }) => {
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
  });

  test("navigates to parameters, submits form, and displays analysis results", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h1")).toContainText("Enterprise Churn Engine");

    await page.getByText("Enter Engine").click();
    await page.waitForURL("**/parameters");
    await expect(page.locator("h1")).toContainText("Input Engine");

    // The Zod schema coerces empty strings to 0 and SatisfactionScore
    // requires >= 1, so a partial fill can never pass validation. The real
    // user path is the built-in preset - use it to populate the form.
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();

    await page.waitForURL("**/analysis", { timeout: 15000 });
    await expect(page.locator("h1")).toContainText("Results Terminal");

    await expect(page.getByText("High", { exact: true })).toBeVisible();
    // Gauge exposes its reading via aria-label; "82.3" also appears in the
    // baseline card ("82.34") and the summary paragraph, so use the a11y name.
    await expect(
      page.getByRole("img", { name: /Churn risk gauge reading 82\.3 percent/ })
    ).toBeVisible();
    await expect(page.getByText("Churn probability", { exact: true })).toBeVisible();
    await expect(page.getByText("Risk Classification")).toBeVisible();
    await expect(page.getByText("LLM", { exact: true })).toBeVisible();
  });

  test("displays error on backend failure", async ({ page }) => {
    await page.route("**/predict", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Internal server error" }),
      });
    });
    await page.route("**/health", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_HEALTH_RESPONSE),
      });
    });

    await page.goto("/parameters");
    await expect(page.locator("h1")).toContainText("Input Engine");
    await page.getByRole("button", { name: "High-Risk Profile" }).click();
    await page.getByRole("button", { name: "Analyze" }).click();

    // The parameters page renders the backend's `detail` in a role=alert box;
    // assert on the text itself since Next's route announcer is also role=alert.
    await expect(page.getByText("Internal server error")).toBeVisible({
      timeout: 10000,
    });
  });
});
