/**
 * Central source of truth for the schema's binary Yes/No fields.
 *
 * The Zod schema declares each of these as a 0-or-1 integer. The form
 * shows them as a "Yes"/"No" dropdown, so the conversion has to happen
 * somewhere in the data path. Doing it at the form-state level (inside
 * `FormField`) keeps RHF's state numeric and lets Zod validate without
 * the `coerce` ever needing to convert a string.
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

/** `Yes` / `1` → 1, `No` / `0` → 0, anything else → `null`. */
export function toBinary(value: unknown): 0 | 1 | null {
  if (value === "Yes" || value === 1 || value === "1") return 1;
  if (value === "No" || value === 0 || value === "0") return 0;
  return null;
}

/** `1` / `Yes` → "Yes", `0` / `No` → "No", unset → "" (placeholder). */
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
