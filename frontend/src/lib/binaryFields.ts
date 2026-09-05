/**
 * Central source of truth for the schema's binary Yes/No fields.
 * The form renders them as a Yes/No select and stores 0/1 so the Zod
 * schema validates without coercion.
 */

/** Field names that render as a Yes/No `<select>` and store 0/1 in RHF. */
export const BINARY_FIELDS: ReadonlySet<string> = new Set([
  "SeniorCitizen",
  "Partner",
  "Dependents",
  "Married",
  "Under30",
  "ReferredAFriend",
  "PhoneService",
  "MultipleLines",
  "InternetService",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
  "StreamingMusic",
  "UnlimitedData",
  "PaperlessBilling",
]);

/** `Yes` / `1` -> 1, `No` / `0` -> 0, anything else -> `null`. */
export function toBinary(value: unknown): 0 | 1 | null {
  if (value === "Yes" || value === 1 || value === "1") return 1;
  if (value === "No" || value === 0 || value === "0") return 0;
  return null;
}

/** `1` / `Yes` -> "Yes", `0` / `No` -> "No", unset -> "" (placeholder). */
export function toYesNo(value: unknown): "Yes" | "No" | "" {
  if (value === 1 || value === "1" || value === "Yes") return "Yes";
  if (value === 0 || value === "0" || value === "No") return "No";
  return "";
}

/**
 * Walk a `FormValues`-shaped record and replace any binary-field value
 * with its numeric form. Non-binary values pass through untouched.
 * Used when applying a preset so the form starts in a Zod-valid state
 * before the user has touched anything.
 */
export function normalizeBinaryValues(
  values: Record<string, unknown>
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(values)) {
    out[k] = BINARY_FIELDS.has(k) ? toBinary(v) : v;
  }
  return out;
}
