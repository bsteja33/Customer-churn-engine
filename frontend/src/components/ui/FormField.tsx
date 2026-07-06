import React from "react";
import { BINARY_FIELDS, toBinary, toYesNo } from "../../lib/binaryFields";

export interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select";
  options?: string[];
}

interface FormFieldProps {
  field: FieldDef;
  /** RHF value (controlled). Strings for selects, strings for numbers (so the
   *  user can clear the field). Binary fields hold 0 | 1 (number) — see
   *  `toBinary` / `toYesNo` in `lib/binaryFields.ts`. */
  value: string | number | null | undefined;
  onChange: (key: string, value: string | number) => void;
  onBlur?: () => void;
  /** Optional Zod error to surface inline. */
  error?: string;
}

/**
 * Bound to react-hook-form via the parent <Controller>. The component is
 * intentionally a thin controlled wrapper so the same markup can be reused
 * outside RHF contexts in tests.
 *
 * For binary Yes/No fields (`BINARY_FIELDS`) the underlying `<select>`
 * still exposes "Yes" / "No" as its option values, but RHF state is
 * translated to 0 / 1 on change and back to "Yes" / "No" on render. That
 * way the Zod schema's `z.coerce.number()` always sees a real number
 * and the "expected number, received nan" failure mode goes away.
 */
export const FormField: React.FC<FormFieldProps> = ({
  field,
  value,
  onChange,
  onBlur,
  error,
}) => {
  const inputId = `field-${field.key}`;
  const errorId = error ? `${inputId}-error` : undefined;
  const hasError = Boolean(error);
  const isBinary = BINARY_FIELDS.has(field.key);

  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={inputId}
        className="text-xs uppercase tracking-wider text-white/50 font-sans"
      >
        {field.label}
      </label>
      {field.type === "select" ? (
        <select
          id={inputId}
          // For binary fields, RHF holds 0 | 1; map back to "Yes" / "No" so
          // the matching <option> renders selected. Empty / null becomes "".
          value={isBinary ? toYesNo(value) : ((value as string) ?? "")}
          onChange={(e) => {
            const raw = e.target.value;
            if (isBinary) {
              onChange(field.key, toBinary(raw) ?? "");
              return;
            }
            onChange(field.key, raw);
          }}
          onBlur={onBlur}
          aria-invalid={hasError || undefined}
          aria-describedby={errorId}
          className="w-full bg-transparent border-b text-white text-sm py-2.5 appearance-none cursor-pointer transition-colors outline-none rounded-none font-sans border-white/20 hover:border-white focus:border-white aria-[invalid=true]:border-red"
        >
          <option value="" className="bg-black text-white">—</option>
          {field.options?.map((opt) => (
            <option key={opt} value={opt} className="bg-black text-white">
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={inputId}
          type="number"
          value={value !== null && value !== undefined ? String(value) : ""}
          onChange={(e) =>
            onChange(
              field.key,
              e.target.value === "" ? "" : Number(e.target.value)
            )
          }
          onBlur={onBlur}
          aria-invalid={hasError || undefined}
          aria-describedby={errorId}
          placeholder="0"
          className="w-full bg-transparent border-b text-white text-sm py-2.5 tabular-nums transition-colors outline-none rounded-none font-sans border-white/20 hover:border-white focus:border-white aria-[invalid=true]:border-red"
        />
      )}
      {error && (
        <span
          id={errorId}
          role="alert"
          className="text-[10px] text-red uppercase tracking-wider font-sans"
        >
          {error}
        </span>
      )}
    </div>
  );
};
