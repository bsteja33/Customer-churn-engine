import type { FormValues } from "../store/useFormStore";

/**
 * A preset is a complete FormValues record plus a display name and
 * a one-line blurb shown on the home page and preset dropdown.
 *
 * The values here mirror the structure produced by `INITIAL_FORM` in
 * the parameters page: select fields use the human string ("Yes" /
 * "No" / "Month-to-Month"), numeric fields use raw numbers, and
 * intentionally-unset fields are omitted entirely. The form's
 * YES_NO_FIELDS handler and Zod schema normalize these to the API
 * contract at submit time.
 */

export interface Preset {
  id: string;
  label: string;
  blurb: string;
  values: FormValues;
}

const NUMERIC_FIELDS = new Set([
  "tenure",
  "Age",
  "NumberOfDependents",
  "NumberOfReferrals",
  "SatisfactionScore",
  "CLTV",
  "AvgMonthlyGBDownload",
  "AvgMonthlyLongDistanceCharges",
  "MonthlyCharges",
  "TotalCharges",
  "TotalRefunds",
  "TotalExtraDataCharges",
  "TotalLongDistanceCharges",
  "TotalRevenue",
]);

/** Internal helper: convert a record of raw values into FormValues. */
function toFormValues(
  raw: Record<string, string | number>
): FormValues {
  const out: FormValues = {};
  for (const [k, v] of Object.entries(raw)) {
    out[k] = NUMERIC_FIELDS.has(k) && typeof v !== "number" ? Number(v) : v;
  }
  return out;
}

export const highRiskPreset: Preset = {
  id: "highRisk",
  label: "High-Risk Profile",
  blurb:
    "Month-to-month, fiber optic, no support add-ons, low tenure, low satisfaction.",
  values: toFormValues({
    Gender: "Male",
    SeniorCitizen: "No",
    Partner: "No",
    Dependents: "No",
    Married: "No",
    Under30: "Yes",
    ReferredAFriend: "No",
    Age: 28,
    NumberOfDependents: 0,
    NumberOfReferrals: 0,
    SatisfactionScore: 2,
    tenure: 2,
    Contract: "Month-to-Month",
    Offer: "None",
    PaperlessBilling: "Yes",
    PaymentMethod: "Bank Withdrawal",
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Yes",
    InternetType: "Fiber Optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    StreamingMusic: "No",
    UnlimitedData: "Yes",
    AvgMonthlyLongDistanceCharges: 5,
    AvgMonthlyGBDownload: 80,
    MonthlyCharges: 95,
    TotalCharges: 190,
    TotalRefunds: 0,
    TotalExtraDataCharges: 10,
    TotalLongDistanceCharges: 20,
    TotalRevenue: 200,
  }),
};

export const loyalPreset: Preset = {
  id: "loyal",
  label: "Loyal Profile",
  blurb:
    "Two-year contract, all support add-ons, long tenure, high CLTV, high satisfaction.",
  values: toFormValues({
    Gender: "Female",
    SeniorCitizen: "No",
    Partner: "Yes",
    Dependents: "Yes",
    Married: "Yes",
    Under30: "No",
    ReferredAFriend: "Yes",
    Age: 54,
    NumberOfDependents: 2,
    NumberOfReferrals: 4,
    SatisfactionScore: 5,
    tenure: 60,
    Contract: "Two Year",
    Offer: "Offer E",
    PaperlessBilling: "Yes",
    PaymentMethod: "Credit Card",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "Yes",
    InternetType: "Fiber Optic",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    StreamingMusic: "Yes",
    UnlimitedData: "Yes",
    AvgMonthlyLongDistanceCharges: 25,
    AvgMonthlyGBDownload: 40,
    MonthlyCharges: 89,
    TotalCharges: 5340,
    TotalRefunds: 0,
    TotalExtraDataCharges: 0,
    TotalLongDistanceCharges: 1500,
    TotalRevenue: 5500,
  }),
};

export const PRESETS: ReadonlyArray<Preset> = [highRiskPreset, loyalPreset];
