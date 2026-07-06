"use client";

import type { FeatureImportance } from "../types/api";

interface ShapPanelProps {
  features: FeatureImportance[];
}

const DIRECTION_META = {
  up: {
    label: "Increases risk",
    color: "#ef4444",
    bg: "bg-red/10",
    text: "text-red",
    arrow: "▲",
  },
  down: {
    label: "Decreases risk",
    color: "#22c55e",
    bg: "bg-green/10",
    text: "text-green",
    arrow: "▼",
  },
} as const;

function formatValue(v: string | number | null): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(2);
  }
  return v;
}

/** Centered-bar SHAP panel sorted by |magnitude|. */
export function ShapPanel({ features }: ShapPanelProps) {
  if (features.length === 0) {
    return (
      <div className="border border-white/10 p-6 text-sm text-white/50 font-mono">
        No feature importance returned for this prediction.
      </div>
    );
  }

  const sorted = [...features].sort(
    (a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude)
  );
  // Normalize against the top contributor so the largest bar fills the lane.
  const maxAbs = Math.max(...sorted.map((f) => Math.abs(f.magnitude)), 0.0001);

  return (
    <div className="flex flex-col divide-y divide-white/5 border border-white/10">
      <div className="px-5 py-3 bg-[#0a0a0a] flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-white/50">
          Top Feature Drivers
        </span>
        <span className="text-[10px] uppercase tracking-widest text-white/30 font-mono">
          sorted by |magnitude|
        </span>
      </div>
      <ul role="list" className="flex flex-col">
        {sorted.map((f, i) => {
          const meta = DIRECTION_META[f.direction];
          const pct = (Math.abs(f.magnitude) / maxAbs) * 100;
          return (
            <li
              key={`${f.feature}-${i}`}
              className="grid grid-cols-[160px_1fr_120px] sm:grid-cols-[200px_1fr_140px] items-center gap-3 px-5 py-3 hover:bg-white/[0.02] transition-colors"
            >
              <div className="flex flex-col min-w-0">
                <span className="text-xs text-white/80 font-sans truncate" title={f.feature}>
                  {f.feature}
                </span>
                <span
                  className={`text-[10px] uppercase tracking-widest font-mono ${meta.text}`}
                >
                  {meta.arrow} {meta.label}
                </span>
              </div>

              <div className="relative h-2 bg-white/5 overflow-hidden">
                <div
                  className="absolute left-1/2 top-0 bottom-0 w-px bg-white/10"
                  aria-hidden
                />
                <div
                  className="absolute top-0 bottom-0"
                  style={{
                    backgroundColor: meta.color,
                    width: `${pct / 2}%`,
                    left: f.direction === "up" ? "50%" : `${50 - pct / 2}%`,
                    transition: "width 400ms ease-out, left 400ms ease-out",
                  }}
                  aria-hidden
                />
              </div>

              <div className="flex flex-col items-end gap-0.5 tabular-nums">
                <span className="text-[11px] font-mono text-white/70">
                  {formatValue(f.value)}
                </span>
                <span className="text-[10px] font-mono text-white/30">
                  {f.magnitude.toFixed(3)}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
