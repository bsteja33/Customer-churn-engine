"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

export type ModelId = "standard" | "high_capacity";

interface ModelInfo {
  id: ModelId;
  label: string;
  apiName: string;
}

interface ProviderState {
  /** Raw key as typed by the user. Trimmed when used for headers. */
  key: string;
  /** Which model slot the user picked. The default is ``standard``. */
  model: ModelId;
  setKey: (key: string) => void;
  setModel: (model: ModelId) => void;
  clear: () => void;
}

/**
 * Synchronous derived selector — true when the user has typed
 * a non-whitespace key. Lives as a function (not stored state)
 * so the value can never desynchronize from ``key``.
 */
export function hasKey(s: ProviderState): boolean {
  return s.key.trim().length > 0;
}

const STORAGE_KEY = "churn-provider-config";
const STORAGE_VERSION = 1;
const DEFAULT_MODEL: ModelId = "standard";
const VALID_MODELS: ReadonlySet<ModelId> = new Set(["standard", "high_capacity"]);

/**
 * Reconcile legacy or corrupt persisted state with the current shape.
 *
 * Why this exists: Zustand's `persist` middleware refuses to apply a
 * payload whose `__version` does not match the current `version`,
 * unless a `migrate` function is provided. Without one, a browser that
 * previously held a `version: 0` (or no version at all) value throws
 * "State loaded from storage couldn't be migrated" on first rehydrate,
 * which is propagated as an unhandled rejection during SSR/hydration
 * and tears the React tree down.
 *
 * Contract: return a partial state to merge into the default state, or
 * `null` to discard the payload and start fresh. Anything that is not
 * a recognizable, well-typed record is treated as corrupt.
 */
function migrate(persisted: unknown, _fromVersion: number): Partial<ProviderState> | null {
  if (!persisted || typeof persisted !== "object") return null;
  const record = persisted as Record<string, unknown>;

  const rawKey = typeof record.key === "string" ? record.key : "";
  const rawModel = record.model;
  const model: ModelId = typeof rawModel === "string" && VALID_MODELS.has(rawModel as ModelId)
    ? (rawModel as ModelId)
    : DEFAULT_MODEL;

  // Old builds stored a derived `hasKey` field. Drop it; the selector
  // recomputes it on read so the field is never needed on disk.
  return { key: rawKey, model };
}

export const useProviderStore = create<ProviderState>()(
  persist(
    (set) => ({
      key: "",
      model: DEFAULT_MODEL,
      setKey: (key) => set({ key }),
      setModel: (model) => set({ model }),
      clear: () => set({ key: "", model: DEFAULT_MODEL }),
    }),
    {
      name: STORAGE_KEY,
      version: STORAGE_VERSION,
      storage: createJSONStorage(() => {
        // SSR guard. Rehydration happens on the client only.
        if (typeof window === "undefined") {
          return {
            getItem: () => null,
            setItem: () => undefined,
            removeItem: () => undefined,
          };
        }
        return window.localStorage;
      }),
      migrate,
      merge: (persistedState, currentState) => {
        // Defensive merge: even when the persisted version matches, we
        // re-validate the loaded shape. Stale `hasKey` fields from
        // older builds are dropped here because we only forward the
        // two fields we know about.
        const record =
          persistedState && typeof persistedState === "object"
            ? (persistedState as Record<string, unknown>)
            : {};
        const rawKey = typeof record.key === "string" ? record.key : "";
        const rawModel = record.model;
        const model: ModelId =
          typeof rawModel === "string" && VALID_MODELS.has(rawModel as ModelId)
            ? (rawModel as ModelId)
            : currentState.model;
        return { ...currentState, key: rawKey, model };
      },
      // Only the user-controlled fields are persisted. ``hasKey`` is
      // derived at read time, so it never needs to be on disk.
      partialize: (s) => ({ key: s.key, model: s.model }),
      // A corrupt payload must never crash rehydration. Returning the
      // default state silently is preferable to an unhandled rejection
      // that would tear down the React tree during hydration.
      onRehydrateStorage: () => (_state, error) => {
        if (error && process.env.NODE_ENV !== "production") {
          // eslint-disable-next-line no-console
          console.warn("[useProviderStore] rehydrate failed, using defaults", error);
        }
      },
    }
  )
);

export const MODEL_OPTIONS: ReadonlyArray<ModelInfo> = [
  { id: "standard", label: "Standard", apiName: "llm-default" },
  { id: "high_capacity", label: "High capacity", apiName: "llm-large" },
];

export { DEFAULT_MODEL };
