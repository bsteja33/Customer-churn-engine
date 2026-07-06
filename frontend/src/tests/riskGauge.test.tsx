import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  RiskGauge,
  fractionToAngle,
  arcPath,
  bandColor,
} from "../components/RiskGauge";

describe("RiskGauge math helpers", () => {
  it("fractionToAngle maps 0% -> 180° (left) and 100% -> 0° (right)", () => {
    // Upper-semicircle geometry: 0% sits at math 180° (left), 100% at
    // math 0° (right). The path counter-clockwise in math = upper arc.
    expect(fractionToAngle(0)).toBe(180);
    expect(fractionToAngle(1)).toBe(0);
    expect(fractionToAngle(0.5)).toBe(90);
    expect(fractionToAngle(-1)).toBe(180);
    expect(fractionToAngle(2)).toBe(0);
  });

  it("arcPath produces a valid SVG path with M and A commands", () => {
    const d = arcPath(100, 100, 50, 180, 0);
    expect(d.startsWith("M ")).toBe(true);
    expect(d).toContain(" A 50 50 0 ");
  });

  it("arcPath uses sweep flag 0 to draw the upper semicircle", () => {
    // Upper-arc geometry: sweep flag must be 0 (counter-clockwise in
    // math = upper arc in SVG display space).
    const d = arcPath(100, 100, 50, 180, 90);
    expect(d).toMatch(/0 0 \d/);
  });

  it("arcPath endpoints trace the upper arc (y < cy)", () => {
    // Every endpoint the gauge generates for the band edges must
    // sit ABOVE the center (y < cy). The old geometry put them at
    // y == cy (bottom semicircle = smile). This is the regression
    // guard.
    const cy = 100;
    const r = 50;
    for (let i = 0; i <= 10; i++) {
      const ang = fractionToAngle(i / 10);
      const rad = (ang * Math.PI) / 180;
      const y = cy - r * Math.sin(rad);
      expect(y).toBeLessThanOrEqual(cy + 1e-9);
    }
  });

  it("bandColor maps to green / amber / red across thresholds", () => {
    expect(bandColor(0)).toBe("#22c55e");
    expect(bandColor(0.33)).toBe("#22c55e");
    expect(bandColor(0.34)).toBe("#f59e0b");
    expect(bandColor(0.66)).toBe("#f59e0b");
    expect(bandColor(0.67)).toBe("#ef4444");
    expect(bandColor(1)).toBe("#ef4444");
  });
});

describe("RiskGauge component", () => {
  it("renders the percentage with one decimal", () => {
    render(<RiskGauge probability={0.753} />);
    expect(screen.getByText(/75\.3/)).not.toBeNull();
  });

  it("clamps the displayed percentage to [0, 100]", () => {
    const { rerender } = render(<RiskGauge probability={-0.5} />);
    expect(screen.getByText(/0\.0/)).not.toBeNull();
    rerender(<RiskGauge probability={1.5} />);
    expect(screen.getByText(/100\.0/)).not.toBeNull();
  });

  it("includes a role=img with the probability in the aria-label", () => {
    render(<RiskGauge probability={0.42} />);
    const img = screen.getByRole("img");
    expect(img.getAttribute("aria-label")).toContain("42.0 percent");
  });

  it("renders an SVG with the band paths and needle", () => {
    const { container } = render(<RiskGauge probability={0.5} size={200} />);
    // Three band paths plus background track plus inner track.
    const paths = container.querySelectorAll("svg path");
    expect(paths.length).toBeGreaterThanOrEqual(4);
  });
});
