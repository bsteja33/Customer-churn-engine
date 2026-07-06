"use client";

import { create } from "zustand";
import { apiFetch } from "../lib/api";

/**
 * Module-level singleton: subscribeHealth() refcounts subscribers and starts
 * one shared setInterval. The in-flight guard prevents stacked requests on
 * a slow response.
 */

export interface HealthData {
  status: string;
  model_loaded: boolean;
  model_path: string;
  latency_ms: number;
  /** Wall-clock time the snapshot was taken. */
  fetched_at: number;
}

export type HealthPhase =
  | { phase: "loading" }
  | { phase: "ok"; data: HealthData }
  | { phase: "error"; message: string };

/** Coarse-grained status used by the rail chip. */
export type HealthSummary = "unknown" | "online" | "degraded" | "offline";

interface HealthState {
  health: HealthPhase;
  /** Derived from `health`; updated whenever a new phase is committed. */
  summary: HealthSummary;
  /** Force an immediate refresh (does not affect the refcount). */
  refresh: () => void;
}

const DEFAULT_INTERVAL_MS = 15_000;
// 60s covers Render's free-tier cold start (services sleep after
// 15 min of inactivity; the first request after sleep takes
// 30–60s). Polling on a 15s tick still recovers quickly once the
// service wakes, so a longer timeout does not stall the UI.
const REQUEST_TIMEOUT_MS = 60_000;

// --- Module-level singleton state ----------------------------------------

let subscribers = 0;
let intervalId: ReturnType<typeof setInterval> | null = null;
let inFlight = false;

function summarize(phase: HealthPhase): HealthSummary {
  if (phase.phase === "loading") return "unknown";
  if (phase.phase === "error") return "offline";
  return phase.data.model_loaded && phase.data.status === "healthy"
    ? "online"
    : "degraded";
}

async function pollOnce() {
  if (inFlight) return;
  inFlight = true;
  const start =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const res = await apiFetch("/api/health", {
      signal: controller.signal,
      // Skip the HTTP cache so a recovery is visible within one tick.
      cache: "no-store",
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const next: HealthPhase = {
        phase: "error",
        message: `HTTP ${res.status}`,
      };
      useHealthStore.setState({ health: next, summary: summarize(next) });
      return;
    }
    const body = (await res.json()) as {
      status?: string;
      model_loaded?: boolean;
      model_path?: string;
    };
    const latency =
      typeof performance !== "undefined"
        ? Math.round(performance.now() - start)
        : 0;
    const next: HealthPhase = {
      phase: "ok",
      data: {
        status: body.status ?? "unknown",
        model_loaded: body.model_loaded ?? false,
        model_path: body.model_path ?? "unknown",
        latency_ms: latency,
        fetched_at: Date.now(),
      },
    };
    useHealthStore.setState({ health: next, summary: summarize(next) });
  } catch (err) {
    // AbortError from our own timeout — keep the previous snapshot in
    // the store (don't clobber a healthy "ok" with a transient timeout)
    // and leave the summary as it was. Anything else is a real network
    // failure and the rail should reflect that.
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    const message =
      err instanceof Error ? err.message : "Unknown error";
    const next: HealthPhase = { phase: "error", message };
    useHealthStore.setState({ health: next, summary: summarize(next) });
  } finally {
    inFlight = false;
  }
}

function startPolling() {
  if (intervalId) return;
  // Fire one immediately, then on the interval.
  void pollOnce();
  intervalId = setInterval(pollOnce, DEFAULT_INTERVAL_MS);
}

function stopPolling() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

// The ``set`` parameter is required by zustand's signature but unused —
// all mutations go through ``useHealthStore.setState`` in pollOnce above.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const useHealthStore = create<HealthState>((_set) => ({
  health: { phase: "loading" },
  summary: "unknown",
  refresh: () => {
    void pollOnce();
  },
}));

/** Subscribe to the shared polling loop. Call inside a useEffect. */
export function subscribeHealth(): () => void {
  subscribers += 1;
  if (subscribers === 1) {
    startPolling();
  }
  return () => {
    subscribers = Math.max(0, subscribers - 1);
    if (subscribers === 0) {
      stopPolling();
    }
  };
}

/** Test-only reset. */
export function __resetHealthForTests() {
  subscribers = 0;
  if (intervalId) clearInterval(intervalId);
  intervalId = null;
  inFlight = false;
  useHealthStore.setState({ health: { phase: "loading" }, summary: "unknown" });
}
