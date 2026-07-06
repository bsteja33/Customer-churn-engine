import { apiFetch } from "./api";
import type { RetentionScriptResponse } from "../types/api";

export interface RetentionRequest {
  risk_level: string;
  reasons: string;
  top_drivers?: string[];
  risk_signals?: string[];
  probability_pct?: number;
}

export interface ModelCatalog {
  models: Record<string, string>;
  default: string;
}

export async function fetchRetentionScript(
  payload: RetentionRequest,
): Promise<RetentionScriptResponse> {
  const res = await apiFetch("/api/generate_retention_script", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Retention script failed: HTTP ${res.status}`);
  }
  return (await res.json()) as RetentionScriptResponse;
}

export async function fetchModelCatalog(): Promise<ModelCatalog> {
  const res = await apiFetch("/api/llm/models");
  if (!res.ok) {
    throw new Error(`Model catalog failed: HTTP ${res.status}`);
  }
  return (await res.json()) as ModelCatalog;
}
