import { describe, it, expect, vi } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { FormField, type FieldDef } from "../components/ui/FormField";

describe("FormField binary Yes/No conversion", () => {
  const binaryField: FieldDef = {
    key: "SeniorCitizen",
    label: "Senior Citizen",
    type: "select",
    options: ["Yes", "No"],
  };

  it("renders the 'Yes' option selected when value is 1", () => {
    const { container } = render(
      <FormField field={binaryField} value={1} onChange={() => {}} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    expect(select.value).toBe("Yes");
  });

  it("renders the 'No' option selected when value is 0", () => {
    const { container } = render(
      <FormField field={binaryField} value={0} onChange={() => {}} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    expect(select.value).toBe("No");
  });

  it("renders the placeholder when value is null / empty", () => {
    const { container } = render(
      <FormField field={binaryField} value={null} onChange={() => {}} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    expect(select.value).toBe("");
  });

  it("emits 1 (not 'Yes') when the user picks 'Yes'", () => {
    const onChange = vi.fn();
    const { container } = render(
      <FormField field={binaryField} value="" onChange={onChange} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "Yes" } });
    expect(onChange).toHaveBeenCalledWith("SeniorCitizen", 1);
  });

  it("emits 0 (not 'No') when the user picks 'No'", () => {
    const onChange = vi.fn();
    const { container } = render(
      <FormField field={binaryField} value="" onChange={onChange} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "No" } });
    expect(onChange).toHaveBeenCalledWith("SeniorCitizen", 0);
  });

  it("emits '' when the user picks the placeholder option", () => {
    const onChange = vi.fn();
    const { container } = render(
      <FormField field={binaryField} value={1} onChange={onChange} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith("SeniorCitizen", "");
  });
});

describe("FormField non-binary select (no conversion)", () => {
  const genderField: FieldDef = {
    key: "Gender",
    label: "Gender",
    type: "select",
    options: ["Male", "Female"],
  };

  it("passes the chosen option through as a string", () => {
    const onChange = vi.fn();
    const { container } = render(
      <FormField field={genderField} value="" onChange={onChange} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "Female" } });
    expect(onChange).toHaveBeenCalledWith("Gender", "Female");
  });

  it("renders the chosen option as selected", () => {
    const { container } = render(
      <FormField field={genderField} value="Female" onChange={() => {}} />
    );
    const select = container.querySelector("select") as HTMLSelectElement;
    expect(select.value).toBe("Female");
  });
});
