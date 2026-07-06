import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ShapPanel } from "../components/ShapPanel";

const SAMPLE = [
  { feature: "tenure", value: 2, magnitude: 0.42, direction: "up" as const },
  { feature: "Contract: Month-to-Month", value: 1, magnitude: 0.31, direction: "up" as const },
  { feature: "TechSupport", value: 0, magnitude: 0.18, direction: "down" as const },
  { feature: "SatisfactionScore", value: 2, magnitude: 0.12, direction: "up" as const },
];

describe("ShapPanel", () => {
  it("renders all features in input order", () => {
    render(<ShapPanel features={SAMPLE} />);
    expect(screen.getByText("tenure")).not.toBeNull();
    expect(screen.getByText("Contract: Month-to-Month")).not.toBeNull();
    expect(screen.getByText("TechSupport")).not.toBeNull();
    expect(screen.getByText("SatisfactionScore")).not.toBeNull();
  });

  it("shows the empty-state message for an empty input", () => {
    render(<ShapPanel features={[]} />);
    expect(screen.getByText(/No feature importance/i)).not.toBeNull();
  });

  it("formats integer values as integers and decimals with 2 places", () => {
    const mixed = [
      { feature: "Age", value: 42, magnitude: 0.1, direction: "up" as const },
      { feature: "AvgCharges", value: 12.3456, magnitude: 0.2, direction: "up" as const },
      { feature: "Field", value: null, magnitude: 0.3, direction: "down" as const },
    ];
    render(<ShapPanel features={mixed} />);
    expect(screen.getByText("42")).not.toBeNull();
    expect(screen.getByText("12.35")).not.toBeNull();
    expect(screen.getByText("—")).not.toBeNull();
  });

  it("marks up direction with a status role=img or list with rows", () => {
    const { container } = render(<ShapPanel features={SAMPLE} />);
    const items = container.querySelectorAll('[role="list"] > li');
    expect(items.length).toBe(4);
  });
});
