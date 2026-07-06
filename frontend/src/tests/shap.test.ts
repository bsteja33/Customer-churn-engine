import { describe, it, expect } from "vitest";
import {
  topDrivers,
  formatDriversForLlm,
  deriveRiskSignals,
  buildRetentionRequest,
} from "../lib/shap";
import type { FeatureImportance, PredictResponse } from "../types/api";

const SAMPLE: FeatureImportance[] = [
  { feature: "SatisfactionScore", value: 1, magnitude: 0.42, direction: "up" },
  { feature: "Tenure", value: 2, magnitude: 0.18, direction: "up" },
  { feature: "Contract", value: "Month-to-Month", magnitude: 0.15, direction: "up" },
  { feature: "StreamingTV", value: 0, magnitude: 0.04, direction: "down" },
];

const PRED: PredictResponse = {
  prediction: 1,
  churn_probability: 0.62,
  retention_risk: "Medium",
  feature_importance: SAMPLE,
};

describe("topDrivers", () => {
  it("returns the top-k by |magnitude| in descending order", () => {
    const out = topDrivers(SAMPLE, 2);
    expect(out.map((f) => f.feature)).toEqual([
      "SatisfactionScore",
      "Tenure",
    ]);
  });

  it("returns [] on null/empty", () => {
    expect(topDrivers(null)).toEqual([]);
    expect(topDrivers([])).toEqual([]);
  });
});

describe("formatDriversForLlm", () => {
  it("renders direction, value, and magnitude in one line", () => {
    const s = formatDriversForLlm(SAMPLE);
    expect(s).toContain("SatisfactionScore=1 (+0.420)");
    expect(s).toContain("Contract=Month-to-Month (+0.150)");
    expect(s).toContain("StreamingTV=0 (-0.040)");
  });
});

describe("deriveRiskSignals", () => {
  it("emits a satisfaction precaution when the satisfaction driver fires up", () => {
    const out = deriveRiskSignals(PRED, SAMPLE, null);
    const titles = out.map((p) => p.title);
    expect(titles).toContain("Satisfaction-recovery outreach");
    expect(titles).toContain("Early-tenure retention play");
    expect(titles).toContain("Contract migration incentive");
  });

  it("adds an above-baseline alert when probability is high", () => {
    const out = deriveRiskSignals(
      { ...PRED, churn_probability: 0.55 },
      SAMPLE,
      null,
    );
    expect(out.some((p) => p.title === "Above-baseline alert")).toBe(true);
  });

  it("surfaces incomplete-record caveat when key inputs are blank", () => {
    const out = deriveRiskSignals(PRED, SAMPLE, {
      tenure: "",
      Contract: "Month-to-Month",
      SatisfactionScore: 1,
      MonthlyCharges: 80,
    });
    expect(out.some((p) => p.title === "Incomplete customer record")).toBe(
      true,
    );
  });

  it("returns [] when no prediction is available", () => {
    expect(deriveRiskSignals(null, SAMPLE, null)).toEqual([]);
  });
});

describe("buildRetentionRequest", () => {
  it("packages everything the LLM endpoint needs", () => {
    const req = buildRetentionRequest(PRED, SAMPLE, {
      tenure: "2",
      Contract: "Month-to-Month",
      SatisfactionScore: "1",
      MonthlyCharges: "85",
    });
    expect(req.risk_level).toBe("Medium");
    expect(req.probability_pct).toBe(62);
    expect(req.top_drivers.length).toBe(3);
    expect(req.risk_signals.length).toBeGreaterThan(0);
  });
});
