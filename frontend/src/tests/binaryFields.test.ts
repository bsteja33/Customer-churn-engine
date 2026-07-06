import { describe, it, expect } from "vitest";
import {
  BINARY_FIELDS,
  toBinary,
  toYesNo,
  normalizeBinaryValues,
} from "../lib/binaryFields";

describe("toBinary", () => {
  it("converts 'Yes' to 1", () => {
    expect(toBinary("Yes")).toBe(1);
  });

  it("converts 'No' to 0", () => {
    expect(toBinary("No")).toBe(0);
  });

  it("is idempotent for the numeric form", () => {
    expect(toBinary(1)).toBe(1);
    expect(toBinary(0)).toBe(0);
  });

  it("accepts the string-of-digit form", () => {
    expect(toBinary("1")).toBe(1);
    expect(toBinary("0")).toBe(0);
  });

  it("returns null for empty / null / undefined / arbitrary input", () => {
    expect(toBinary("")).toBeNull();
    expect(toBinary(null)).toBeNull();
    expect(toBinary(undefined)).toBeNull();
    expect(toBinary("maybe")).toBeNull();
    expect(toBinary(2)).toBeNull();
    expect(toBinary(-1)).toBeNull();
  });
});

describe("toYesNo", () => {
  it("converts 1 / 'Yes' / '1' to 'Yes'", () => {
    expect(toYesNo(1)).toBe("Yes");
    expect(toYesNo("Yes")).toBe("Yes");
    expect(toYesNo("1")).toBe("Yes");
  });

  it("converts 0 / 'No' / '0' to 'No'", () => {
    expect(toYesNo(0)).toBe("No");
    expect(toYesNo("No")).toBe("No");
    expect(toYesNo("0")).toBe("No");
  });

  it("returns '' for empty / null / undefined / arbitrary input", () => {
    expect(toYesNo("")).toBe("");
    expect(toYesNo(null)).toBe("");
    expect(toYesNo(undefined)).toBe("");
    expect(toYesNo("maybe")).toBe("");
  });
});

describe("BINARY_FIELDS", () => {
  it("contains every expected Yes/No field", () => {
    const expected = [
      "SeniorCitizen", "Partner", "Dependents", "Married", "Under30",
      "ReferredAFriend", "PhoneService", "MultipleLines", "InternetService",
      "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
      "StreamingTV", "StreamingMovies", "StreamingMusic", "UnlimitedData",
      "PaperlessBilling",
    ];
    for (const k of expected) {
      expect(BINARY_FIELDS.has(k), `missing ${k}`).toBe(true);
    }
  });

  it("does not contain non-binary fields", () => {
    for (const k of ["Contract", "PaymentMethod", "InternetType", "Offer", "Gender"]) {
      expect(BINARY_FIELDS.has(k), `unexpected ${k}`).toBe(false);
    }
  });
});

describe("normalizeBinaryValues", () => {
  it("converts binary fields and leaves the rest alone", () => {
    const in_ = {
      SeniorCitizen: "Yes",
      Partner: "No",
      Under30: "Yes",
      Age: 42,
      Contract: "Month-to-Month",
    };
    const out = normalizeBinaryValues(in_);
    expect(out).toEqual({
      SeniorCitizen: 1,
      Partner: 0,
      Under30: 1,
      Age: 42,
      Contract: "Month-to-Month",
    });
  });

  it("passes through non-binary values unchanged", () => {
    const in_ = { Contract: "Two Year", PaymentMethod: "Credit Card" };
    expect(normalizeBinaryValues(in_)).toEqual(in_);
  });

  it("leaves null / empty binary values as null", () => {
    const out = normalizeBinaryValues({
      SeniorCitizen: "",
      Partner: null,
      Under30: undefined,
    });
    expect(out).toEqual({ SeniorCitizen: null, Partner: null, Under30: null });
  });
});
