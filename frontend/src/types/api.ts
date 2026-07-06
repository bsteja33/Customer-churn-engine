export interface FeatureImportance {
  /** Human-readable feature label, e.g. "Contract: Month-to-Month" or "tenure". */
  feature: string;
  /** The value the customer record had for this feature. */
  value: string | number | null;
  /** Absolute SHAP contribution in log-odds space (>= 0). */
  magnitude: number;
  /** "up" pushes toward churn, "down" pushes toward stay. */
  direction: "up" | "down";
}

export interface PredictResponse {
  prediction: number;
  churn_probability: number;
  retention_risk: string;
  /**
   * Top SHAP feature attributions for this prediction, sorted by descending
   * magnitude. `null` if explainability is unavailable for this request.
   */
  feature_importance: FeatureImportance[] | null;
}

export interface RetentionScriptResponse {
  script: string;
}
