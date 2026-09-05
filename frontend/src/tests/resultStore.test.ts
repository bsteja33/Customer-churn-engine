import { describe, it, expect, beforeEach } from "vitest";
import { useResultStore } from "../store/useResultStore";
import type { PredictResponse, RetentionScriptResponse } from "../types/api";

describe("useResultStore", () => {
  beforeEach(() => {
    useResultStore.setState({
      prediction: null,
      retention: null,
      featureImportance: null,
    });
  });

  const mockPrediction: PredictResponse = {
    prediction: 1,
    churn_probability: 0.85,
    retention_risk: "High",
    feature_importance: [
      { feature: "tenure", value: 2, magnitude: 0.18, direction: "up" },
      { feature: "Contract: Month-to-Month", value: 1, magnitude: 0.34, direction: "up" },
    ],
  };

  const mockRetention: RetentionScriptResponse = {
    script: "[Action Plan] - Lock the 12-month contract.",
  };

  it("starts with null prediction, retention, and feature_importance", () => {
    const state = useResultStore.getState();
    expect(state.prediction).toBeNull();
    expect(state.retention).toBeNull();
    expect(state.featureImportance).toBeNull();
  });

  it("setResults stores prediction, retention, and caches feature_importance", () => {
    useResultStore.getState().setResults(mockPrediction, mockRetention);
    const state = useResultStore.getState();
    expect(state.prediction).toEqual(mockPrediction);
    expect(state.retention).toEqual(mockRetention);
    expect(state.featureImportance).toEqual(mockPrediction.feature_importance);
  });

  it("setResults sets featureImportance to null when API omits it", () => {
    const noShap: PredictResponse = {
      prediction: 0,
      churn_probability: 0.1,
      retention_risk: "Low",
      feature_importance: null,
    };
    useResultStore.getState().setResults(noShap, mockRetention);
    expect(useResultStore.getState().featureImportance).toBeNull();
  });

  it("setResults overwrites previous values", () => {
    useResultStore.getState().setResults(mockPrediction, mockRetention);
    const updated: PredictResponse = {
      prediction: 0,
      churn_probability: 0.25,
      retention_risk: "Low",
      feature_importance: null,
    };
    useResultStore.getState().setResults(updated, {
      script: "[Default Action Plan] Operational note.",
    });
    const state = useResultStore.getState();
    expect(state.prediction?.retention_risk).toBe("Low");
    expect(state.retention?.script).toContain("[Default Action Plan]");
    expect(state.featureImportance).toBeNull();
  });

  it("clear resets all three fields to null", () => {
    useResultStore.getState().setResults(mockPrediction, mockRetention);
    useResultStore.getState().clear();
    const state = useResultStore.getState();
    expect(state.prediction).toBeNull();
    expect(state.retention).toBeNull();
    expect(state.featureImportance).toBeNull();
  });

  it("clear is idempotent", () => {
    useResultStore.getState().clear();
    useResultStore.getState().clear();
    const state = useResultStore.getState();
    expect(state.prediction).toBeNull();
    expect(state.retention).toBeNull();
  });
});
