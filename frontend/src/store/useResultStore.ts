"use client";

import { create } from "zustand";
import type {
  PredictResponse,
  RetentionScriptResponse,
  FeatureImportance,
} from "../types/api";

/**
 * In-memory result store. Holds the most recent prediction (with its
 * feature_importance payload) and the most recent retention script.
 *
 * Not persisted: a stale prediction without its form values is
 * misleading, so we deliberately let a refresh clear results. The
 * form state, by contrast, lives in `useFormStore` and is persisted.
 */

interface ResultState {
  prediction: PredictResponse | null;
  retention: RetentionScriptResponse | null;
  /** Cached separately so consumers don't have to re-narrow each time. */
  featureImportance: FeatureImportance[] | null;

  /** Atomically write prediction + script after a successful /predict + /generate_retention_script. */
  setResults: (
    prediction: PredictResponse,
    retention: RetentionScriptResponse
  ) => void;
  /** Clear everything. */
  clear: () => void;
}

export const useResultStore = create<ResultState>((set) => ({
  prediction: null,
  retention: null,
  featureImportance: null,

  setResults: (prediction, retention) =>
    set(() => ({
      prediction,
      retention,
      featureImportance: prediction.feature_importance ?? null,
    })),

  clear: () =>
    set(() => ({
      prediction: null,
      retention: null,
      featureImportance: null,
    })),
}));
