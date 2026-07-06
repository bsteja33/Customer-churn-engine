"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { RefreshCw, Terminal, AlertTriangle, ShieldCheck } from "lucide-react";
import { useResultStore } from "../../store/useResultStore";
import { useFormStore } from "../../store/useFormStore";
import { loyalPreset } from "../../data/presets";
import { ChurnInputSchema, type ChurnInput } from "../../lib/schema";
import type { PredictResponse } from "../../types/api";
import { apiFetch } from "../../lib/api";
import { RiskGauge } from "../../components/RiskGauge";
import { ShapPanel } from "../../components/ShapPanel";
import { CopyButton } from "../../components/CopyButton";
import { deriveRiskSignals } from "../../lib/shap";
import { cn } from "../../lib/cn";

export const TAG_RE = /^\[(Action Plan|Default Action Plan)\]\s*/;

export const TAG_BADGES: Record<string, { label: string; className: string }> = {
  "Action Plan": {
    label: "LLM",
    className:
      "bg-green-900/80 text-green-300 px-2.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider",
  },
  "Default Action Plan": {
    label: "Default",
    className:
      "bg-amber-900/80 text-amber-300 px-2.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider",
  },
};

export function parseScriptTag(raw: string | undefined): {
  badge: { label: string; className: string } | null;
  cleanScript: string;
} {
  if (!raw) return { badge: null, cleanScript: "No action plan generated." };
  const match = raw.match(TAG_RE);
  if (match) {
    const tagName = match[1];
    const badge = TAG_BADGES[tagName] ?? null;
    const cleanScript = raw.slice(match[0].length);
    return { badge, cleanScript };
  }
  return { badge: null, cleanScript: raw };
}

/** Convert FormValues to the API contract: "" → null. Binary fields are
 *  already numeric (0 / 1) by the time they reach us because the form
 *  layer normalizes them at the input. */
function toApiPayload(values: Record<string, string | number | null | undefined>): ChurnInput {
  const out: Record<string, string | number | null | undefined> = {};
  for (const [k, v] of Object.entries(values)) {
    out[k] = v === "" || v === null || v === undefined ? null : v;
  }
  return out as ChurnInput;
}

interface BaselineState {
  status: "loading" | "ok" | "error";
  probability: number | null;
  error?: string;
}

const BASELINE_CACHE_MS = 5 * 60 * 1000; // 5 min
let baselineCache: { value: number; timestamp: number } | null = null;

async function fetchBaselineProbability(): Promise<number> {
  if (baselineCache && Date.now() - baselineCache.timestamp < BASELINE_CACHE_MS) {
    return baselineCache.value;
  }
  const payload = toApiPayload(loyalPreset.values);
  const validated = ChurnInputSchema.parse(payload);
  const res = await apiFetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
  });
  if (!res.ok) {
    throw new Error(`Baseline fetch failed: HTTP ${res.status}`);
  }
  const data = (await res.json()) as PredictResponse;
  baselineCache = { value: data.churn_probability, timestamp: Date.now() };
  return data.churn_probability;
}

export default function AnalysisPage() {
  const prediction = useResultStore((s) => s.prediction);
  const retention = useResultStore((s) => s.retention);
  const featureImportance = useResultStore((s) => s.featureImportance);
  const clear = useResultStore((s) => s.clear);
  const formValues = useFormStore((s) => s.values);
  const router = useRouter();

  const [baseline, setBaseline] = useState<BaselineState>({
    status: "loading",
    probability: null,
  });

  // Guard: if a user lands on /analysis without a prediction, bounce them.
  useEffect(() => {
    if (!prediction) {
      router.replace("/parameters");
    }
  }, [prediction, router]);

  // Fetch the loyal-preset baseline once. Failures degrade gracefully
  // (the delta card is hidden).
  useEffect(() => {
    let cancelled = false;
    fetchBaselineProbability()
      .then((p) => {
        if (cancelled) return;
        setBaseline({ status: "ok", probability: p });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setBaseline({
          status: "error",
          probability: null,
          error: err instanceof Error ? err.message : "Unknown error",
        });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const { badge, cleanScript } = useMemo(
    () => parseScriptTag(retention?.script),
    [retention?.script]
  );

  const riskSignals = useMemo(
    () => deriveRiskSignals(prediction, featureImportance, formValues),
    [prediction, featureImportance, formValues]
  );

  // Show top 3 features inline as a compact summary list. Computed even
  // if no prediction is present so the hook order stays stable.
  const top3 = useMemo(
    () =>
      [...(featureImportance ?? [])]
        .sort((a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude))
        .slice(0, 3),
    [featureImportance]
  );

  if (!prediction) return null;

  const isHighRisk = prediction.retention_risk === "High";
  const isMediumRisk = prediction.retention_risk === "Medium";
  const riskTone = isHighRisk
    ? "border-red/40 bg-red/5"
    : isMediumRisk
      ? "border-amber-500/30 bg-amber-500/5"
      : "border-green/30 bg-green/5";
  const riskTextTone = isHighRisk
    ? "text-red"
    : isMediumRisk
      ? "text-amber-300"
      : "text-green";

  const deltaPct = baseline.probability
    ? (prediction.churn_probability - baseline.probability) * 100
    : null;

  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 flex flex-col gap-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <Link
              href="/parameters"
              className="text-xs uppercase tracking-widest text-white/50 hover:text-white transition-colors"
            >
              ← Back to Inputs
            </Link>
            <h1 className="text-2xl font-bold tracking-widest uppercase">Results Terminal</h1>
          </div>
          <button
            type="button"
            onClick={() => {
              clear();
              router.push("/parameters");
            }}
            className="flex items-center gap-2 text-xs uppercase tracking-widest text-white/50 hover:text-white transition-colors"
          >
            <RefreshCw size={14} />
            New Analysis
          </button>
        </header>

        {/* Risk row: gauge + risk classification + delta + summary */}
        <section className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-8 lg:gap-12 items-start">
          <div className={cn("border p-6 sm:p-8 flex items-center justify-center", riskTone)}>
            <RiskGauge probability={prediction.churn_probability} size={260} label="Churn probability" />
          </div>

          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-widest text-white/50">Risk Classification</span>
              <span className={cn("text-5xl sm:text-6xl font-bold tracking-tighter", riskTextTone)}>
                {prediction.retention_risk}
              </span>
              <p className="text-sm text-white/60 max-w-md">
                {isHighRisk
                  ? "This customer shows multiple high-impact churn signals. Immediate intervention is recommended."
                  : isMediumRisk
                    ? "This customer is at moderate risk. Consider a proactive outreach with incentives."
                    : "This customer is unlikely to churn under current conditions. Maintain standard engagement."}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Delta vs loyal baseline */}
              <div className="border border-white/10 p-5 flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-widest text-white/40">
                  Δ vs Loyal Baseline
                </span>
                {baseline.status === "loading" && (
                  <span className="text-sm text-white/40 animate-pulse">Computing…</span>
                )}
                {baseline.status === "error" && (
                  <span className="text-xs text-white/40 font-mono flex items-center gap-2">
                    <AlertTriangle size={12} /> Baseline unavailable
                  </span>
                )}
                {baseline.status === "ok" && deltaPct !== null && (
                  <>
                    <span
                      className={cn(
                        "text-2xl font-bold tabular-nums tracking-tighter",
                        deltaPct > 0 ? "text-red" : deltaPct < 0 ? "text-green" : "text-white/60"
                      )}
                    >
                      {deltaPct > 0 ? "+" : ""}
                      {deltaPct.toFixed(2)} pts
                    </span>
                    <span className="text-[10px] text-white/40 font-mono">
                      Baseline: {(baseline.probability! * 100).toFixed(2)}% · Loyal preset
                    </span>
                  </>
                )}
              </div>

              {/* Top 3 drivers inline */}
              <div className="border border-white/10 p-5 flex flex-col gap-2">
                <span className="text-[10px] uppercase tracking-widest text-white/40">
                  Top 3 Drivers
                </span>
                {top3.length === 0 ? (
                  <span className="text-xs text-white/40 font-mono">No SHAP data</span>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {top3.map((f, i) => (
                      <li
                        key={`${f.feature}-${i}`}
                        className="flex items-center justify-between gap-2 text-[11px] font-mono"
                      >
                        <span className="text-white/70 truncate">{f.feature}</span>
                        <span
                          className={cn(
                            "tabular-nums",
                            f.direction === "up" ? "text-red" : "text-green"
                          )}
                        >
                          {f.direction === "up" ? "▲" : "▼"}{" "}
                          {f.magnitude.toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* SHAP feature importance */}
        <section className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs uppercase tracking-widest text-white/50">
              Feature Importance (SHAP)
            </h2>
            <span className="text-[10px] text-white/30 font-mono">
              {featureImportance?.length ?? 0} contributor
              {(featureImportance?.length ?? 0) === 1 ? "" : "s"}
            </span>
          </div>
          <ShapPanel features={featureImportance ?? []} />
        </section>

        {/* Practical precautions — derived from the actual SHAP drivers */}
        <section className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs uppercase tracking-widest text-white/50 flex items-center gap-2">
              <ShieldCheck size={12} />
              Practical Precautions
            </h2>
            <span className="text-[10px] text-white/30 font-mono">
              {riskSignals.length} action{riskSignals.length === 1 ? "" : "s"}
            </span>
          </div>
          {riskSignals.length === 0 ? (
            <div className="border border-white/10 p-5 text-sm text-white/50 font-mono">
              No precautions flagged at the current risk level. Maintain
              standard engagement.
            </div>
          ) : (
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {riskSignals.map((s) => (
                <li
                  key={s.id}
                  className="border border-white/10 p-4 flex flex-col gap-2 bg-[#0a0a0a]"
                >
                  <div className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 bg-amber-400" aria-hidden />
                    <span className="text-[11px] uppercase tracking-widest text-white/80 font-bold">
                      {s.title}
                    </span>
                  </div>
                  <p className="text-[11px] text-white/60 leading-relaxed font-mono">
                    {s.body}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Executive Retention Strategy & Action Plan — internal use only */}
        <section className="flex flex-col border border-white/10 min-h-[280px]">
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#0a0a0a]">
            <div className="flex items-center gap-3">
              <Terminal size={14} className="text-white/50" />
              <span className="text-xs uppercase tracking-widest text-white/50">
                Executive Retention Strategy &amp; Action Plan
              </span>
              <span className="text-[10px] uppercase tracking-widest text-white/30 font-mono">
                Internal · CSM use only
              </span>
              {badge && (
                <span className={badge.className}>{badge.label}</span>
              )}
            </div>
            <CopyButton value={cleanScript} label="Copy plan" />
          </div>
          <div className="flex-1 p-6 sm:p-8 bg-black overflow-y-auto flex flex-col gap-3">
            <pre className="font-mono text-sm leading-loose text-white/80 whitespace-pre-wrap break-words m-0">
              <span className="text-green-500 mr-4">~ %</span>
              {cleanScript}
            </pre>
          </div>
        </section>

        {/* Submission summary — what was actually sent (helps audits) */}
        <section className="border border-white/10 p-5 flex flex-col gap-2">
          <span className="text-[10px] uppercase tracking-widest text-white/40">
            Submitted Inputs
          </span>
          <p className="text-[11px] text-white/40 font-mono leading-relaxed">
            {Object.keys(formValues).length === 0
              ? "No form data retained in this session."
              : `${Object.keys(formValues).length} field${
                  Object.keys(formValues).length === 1 ? "" : "s"
                } persisted from the input engine.`}
          </p>
        </section>
      </div>
    </div>
  );
}
