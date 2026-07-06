"use client";

import { useId, useMemo } from "react";

interface RiskGaugeProps {
  probability: number;
  size?: number;
  label?: string;
}

// Single source of truth for the gauge geometry.
//
// The ring is a 180° arc, the upper half of a circle of radius `r`
// centered at (cx, cy) = (size/2, size/2). The flat edge of the
// semicircle sits on the horizontal line y = cy. The SVG viewBox is
// a square of side `size`; the lower half is intentionally empty so
// the ring can be drawn with standard SVG arc commands without any
// per-band offset or per-element coordinate juggling.
//
// Every band, tick mark, label, and the needle are computed from the
// same (cx, cy, r) tuple. There is no per-element cy shift, no
// independent radius per band, and no viewBox clipping — all three
// colored bands are concentric arcs at radius `r`, drawn as
//   M p.x p.y A r r 0 0 1 q.x q.y
// where sweep flag 1 keeps the path on the upper semicircle.
//
// Angles are in math convention (0° = right, 90° = up, 180° = left)
// and converted to SVG space via
//   (x, y) = (cx + r cos θ, cy - r sin θ)
// which flips the y axis so the upper arc is produced.

const START_DEG = 180; // 0% — left
const END_DEG = 0; // 100% — right
const SWEEP = START_DEG - END_DEG; // 180°

const BAND_THRESHOLDS = [
  { upTo: 0.33, color: "#22c55e" },
  { upTo: 0.66, color: "#f59e0b" },
  { upTo: 1.01, color: "#ef4444" },
] as const;

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy - r * Math.sin(rad),
  };
}

export function fractionToAngle(fraction: number): number {
  const clamped = Math.max(0, Math.min(1, fraction));
  return START_DEG - clamped * SWEEP;
}

export function arcPath(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number
): string {
  const start = polar(cx, cy, r, startDeg);
  const end = polar(cx, cy, r, endDeg);
  const largeArc = Math.abs(startDeg - endDeg) > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

export function bandColor(probability: number): string {
  for (const b of BAND_THRESHOLDS) {
    if (probability <= b.upTo) return b.color;
  }
  return BAND_THRESHOLDS[BAND_THRESHOLDS.length - 1].color;
}

export function RiskGauge({
  probability,
  size = 240,
  label,
}: RiskGaugeProps) {
  const reactId = useId();

  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2;
  const strokeWidth = size * 0.07;
  const innerStroke = Math.max(3, strokeWidth * 0.45);
  const tickGap = size * 0.018;
  const tickLen = size * 0.028;
  const tickInner = r + strokeWidth / 2 + tickGap;
  const tickOuter = tickInner + tickLen;
  const labelRadius = tickOuter + size * 0.026;

  const clamped = Math.max(0, Math.min(1, probability));
  const pct = clamped * 100;
  const needleAngle = fractionToAngle(clamped);
  const accent = bandColor(clamped);

  const ticks = useMemo(
    () =>
      Array.from({ length: 11 }, (_, i) => {
        const t = i / 10;
        const ang = fractionToAngle(t);
        const inner = polar(cx, cy, tickInner, ang);
        const outer = polar(cx, cy, tickOuter, ang);
        const label = polar(cx, cy, labelRadius, ang);
        return {
          x1: inner.x,
          y1: inner.y,
          x2: outer.x,
          y2: outer.y,
          labelX: label.x,
          labelY: label.y,
          value: i * 10,
          isMajor: i % 5 === 0,
        };
      }),
    [cx, cy, tickInner, tickOuter, labelRadius]
  );

  const bands = useMemo(() => {
    const stops = [0, 0.33, 0.66, 1];
    return stops.slice(0, -1).map((s, i) => ({
      d: arcPath(cx, cy, r, fractionToAngle(s), fractionToAngle(stops[i + 1])),
      color: BAND_THRESHOLDS[i].color,
      key: `${reactId}-band-${i}`,
    }));
  }, [cx, cy, r, reactId]);

  const backgroundPath = arcPath(cx, cy, r, START_DEG, END_DEG);
  const needleLen = r - strokeWidth / 2 - size * 0.012;
  const needleTip = polar(cx, cy, needleLen, 90);
  const needleBaseBack = polar(cx, cy, size * 0.018, 90 + 90);
  const needleBaseFwd = polar(cx, cy, size * 0.018, 90 - 90);
  const needleRotation = 90 - needleAngle;

  return (
    <div
      className="inline-flex flex-col items-center"
      role="img"
      aria-label={`Churn risk gauge reading ${pct.toFixed(1)} percent`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="overflow-visible"
      >
        <defs>
          <filter
            id={`${reactId}-needle-glow`}
            x="-50%"
            y="-50%"
            width="200%"
            height="200%"
          >
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <path
          d={backgroundPath}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
          strokeLinecap="butt"
        />

        {bands.map((b) => (
          <path
            key={b.key}
            d={b.d}
            fill="none"
            stroke={b.color}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            opacity={0.95}
          />
        ))}

        <path
          d={backgroundPath}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={innerStroke}
        />

        {ticks.map((t, i) => (
          <g key={`tick-${i}`}>
            <line
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke={t.isMajor ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.2)"}
              strokeWidth={t.isMajor ? 1.4 : 1}
            />
            {t.isMajor && (
              <text
                x={t.labelX}
                y={t.labelY}
                fontSize={size * 0.045}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-white/40 font-mono tabular-nums"
              >
                {t.value}
              </text>
            )}
          </g>
        ))}

        <g
          style={{
            transform: `rotate(${needleRotation}deg)`,
            transformOrigin: `${cx}px ${cy}px`,
            transition: "transform 600ms cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        >
          {/* Halo behind the needle for high contrast on any band color */}
          <polygon
            points={`${needleBaseBack.x},${needleBaseBack.y} ${needleBaseFwd.x},${needleBaseFwd.y} ${needleTip.x},${needleTip.y}`}
            fill="rgba(0,0,0,0.55)"
            stroke="none"
          />
          {/* Bright pointer, layered above the bands */}
          <polygon
            points={`${needleBaseBack.x},${needleBaseBack.y} ${needleBaseFwd.x},${needleBaseFwd.y} ${needleTip.x},${needleTip.y}`}
            fill={accent}
            stroke="#ffffff"
            strokeWidth={1.25}
            strokeLinejoin="round"
            filter={`url(#${reactId}-needle-glow)`}
          />
        </g>

        <circle cx={cx} cy={cy} r={size * 0.024} fill="#000" stroke={accent} strokeWidth={2} />
        <circle cx={cx} cy={cy} r={size * 0.012} fill={accent} />
      </svg>

      <div className="-mt-3 flex flex-col items-center gap-1">
        <span
          className="text-5xl sm:text-6xl font-bold tabular-nums tracking-tighter"
          style={{ color: accent }}
        >
          {pct.toFixed(1)}
          <span className="text-2xl sm:text-3xl text-white/50 ml-1">%</span>
        </span>
        {label && (
          <span className="text-[10px] uppercase tracking-widest text-white/40 font-sans">
            {label}
          </span>
        )}
      </div>
    </div>
  );
}
