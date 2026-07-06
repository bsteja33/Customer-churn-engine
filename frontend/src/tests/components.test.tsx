import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { parseScriptTag, TAG_BADGES } from "../app/analysis/page";
import { FormField } from "../components/ui/FormField";
import type { FieldDef } from "../components/ui/FormField";

describe("parseScriptTag", () => {
  it("returns null badge and default text for undefined input", () => {
    const result = parseScriptTag(undefined);
    expect(result.badge).toBeNull();
    expect(result.cleanScript).toBe("No action plan generated.");
  });

  it("returns null badge and default text for null input", () => {
    const result = parseScriptTag(null as unknown as undefined);
    expect(result.badge).toBeNull();
    expect(result.cleanScript).toBe("No action plan generated.");
  });

  it("detects [Action Plan] and returns green badge", () => {
    const result = parseScriptTag("[Action Plan] - Lock the 12-month contract.");
    expect(result.badge).not.toBeNull();
    expect(result.badge!.label).toBe("LLM");
    expect(result.badge!.className).toContain("bg-green-900/80");
    expect(result.cleanScript).toBe("- Lock the 12-month contract.");
  });

  it("detects [Default Action Plan] and returns amber badge", () => {
    const result = parseScriptTag("[Default Action Plan] Operational note.");
    expect(result.badge).not.toBeNull();
    expect(result.badge!.label).toBe("Default");
    expect(result.badge!.className).toContain("bg-amber-900/80");
    expect(result.cleanScript).toBe("Operational note.");
  });

  it("strips trailing whitespace after tag", () => {
    const result = parseScriptTag("[Action Plan]   - Lock the contract");
    expect(result.cleanScript).toBe("- Lock the contract");
  });

  it("returns null badge for raw text with no tag", () => {
    const result = parseScriptTag("Just some plain text.");
    expect(result.badge).toBeNull();
    expect(result.cleanScript).toBe("Just some plain text.");
  });

  it("returns null badge for empty string (falsy to default)", () => {
    const result = parseScriptTag("");
    expect(result.badge).toBeNull();
    expect(result.cleanScript).toBe("No action plan generated.");
  });

  it("preserves script text after the tag and space", () => {
    const plan = "[Action Plan] - Lock the 12-month contract.\n- Open a satisfaction-recovery ticket.";
    const result = parseScriptTag(plan);
    expect(result.badge!.label).toBe("LLM");
    expect(result.cleanScript).toBe(
      "- Lock the 12-month contract.\n- Open a satisfaction-recovery ticket."
    );
  });

  it("strips newlines after tag", () => {
    const plan = "[Default Action Plan]\n- Audit usage vs. plan tier.";
    const result = parseScriptTag(plan);
    expect(result.badge!.label).toBe("Default");
    expect(result.cleanScript).toBe("- Audit usage vs. plan tier.");
  });

  it("uses correct badge class for Action Plan", () => {
    const badge = TAG_BADGES["Action Plan"];
    expect(badge.label).toBe("LLM");
    expect(badge.className).toContain("green");
  });

  it("uses correct badge class for Default Action Plan", () => {
    const badge = TAG_BADGES["Default Action Plan"];
    expect(badge.label).toBe("Default");
    expect(badge.className).toContain("amber");
  });
});

describe("FormField rendering", () => {
  it("renders without crashing", () => {
    const field: FieldDef = { key: "test", label: "Test Field", type: "text" };
    const { container } = render(
      <FormField field={field} value="hello" onChange={() => {}} />
    );
    expect(container.querySelector("input")).not.toBeNull();
  });

  it("displays the field label", () => {
    const field: FieldDef = { key: "test", label: "Test Field", type: "text" };
    render(<FormField field={field} value="" onChange={() => {}} />);
    expect(screen.getByText("Test Field")).not.toBeNull();
  });

  it("renders a select element for select type", () => {
    const field: FieldDef = {
      key: "gender",
      label: "Gender",
      type: "select",
      options: ["Male", "Female"],
    };
    const { container } = render(
      <FormField field={field} value="" onChange={() => {}} />
    );
    expect(container.querySelector("select")).not.toBeNull();
  });

  it("renders an input element for number type", () => {
    const field: FieldDef = { key: "age", label: "Age", type: "number" };
    const { container } = render(
      <FormField field={field} value={null} onChange={() => {}} />
    );
    const input = container.querySelector('input[type="number"]');
    expect(input).not.toBeNull();
  });
});
