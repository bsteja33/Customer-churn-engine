import { describe, it, expect, beforeEach } from "vitest";
import { useFormStore } from "../store/useFormStore";
import type { FormValues } from "../store/useFormStore";

describe("useFormStore", () => {
  beforeEach(() => {
    useFormStore.setState({ values: {} });
    localStorage.removeItem("churn.form.v1");
  });

  it("starts with empty values", () => {
    expect(useFormStore.getState().values).toEqual({});
  });

  it("updateField writes a single key", () => {
    useFormStore.getState().updateField("Age", 42);
    expect(useFormStore.getState().values).toEqual({ Age: 42 });
  });

  it("updateField merges into existing values without dropping them", () => {
    useFormStore.getState().updateField("Age", 42);
    useFormStore.getState().updateField("tenure", 12);
    expect(useFormStore.getState().values).toEqual({ Age: 42, tenure: 12 });
  });

  it("updateField overwrites the same key", () => {
    useFormStore.getState().updateField("Age", 42);
    useFormStore.getState().updateField("Age", 43);
    expect(useFormStore.getState().values.Age).toBe(43);
  });

  it("setValues replaces the entire values record", () => {
    useFormStore.getState().updateField("Age", 42);
    const next: FormValues = { Contract: "Two Year", tenure: 60 };
    useFormStore.getState().setValues(next);
    expect(useFormStore.getState().values).toEqual(next);
  });

  it("reset clears values", () => {
    useFormStore.getState().updateField("Age", 42);
    useFormStore.getState().reset();
    expect(useFormStore.getState().values).toEqual({});
  });

  it("loadPreset copies the preset's values", () => {
    const preset = { values: { Contract: "Two Year", tenure: 60 } as FormValues };
    useFormStore.getState().loadPreset(preset);
    expect(useFormStore.getState().values).toEqual(preset.values);
  });

  it("loadPreset replaces (does not merge) existing values", () => {
    useFormStore.getState().updateField("Age", 42);
    useFormStore.getState().loadPreset({ values: { Contract: "Two Year" } });
    expect(useFormStore.getState().values).toEqual({ Contract: "Two Year" });
  });

  it("persists values to localStorage under churn.form.v1", async () => {
    useFormStore.getState().updateField("Age", 42);
    await new Promise((r) => setTimeout(r, 0));
    const raw = localStorage.getItem("churn.form.v1");
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed.state.values).toEqual({ Age: 42 });
    expect(parsed.version).toBe(1);
  });
});
