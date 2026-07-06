"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Send,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useFormStore, type FormValues } from "../../store/useFormStore";
import { useResultStore } from "../../store/useResultStore";
import { FormField, FieldDef } from "../../components/ui/FormField";
import { ChurnInputSchema, type ChurnInput } from "../../lib/schema";
import { normalizeBinaryValues } from "../../lib/binaryFields";
import type { PredictResponse, RetentionScriptResponse } from "../../types/api";
import { apiFetch } from "../../lib/api";
import { fetchRetentionScript } from "../../lib/llm";
import { buildRetentionRequest } from "../../lib/shap";
import { PRESETS } from "../../data/presets";
import { cn } from "../../lib/cn";

/** Submission request timeout. Long enough for SHAP extraction on a
 *  cold model, short enough that a hung BE never locks the UI. */
const SUBMIT_TIMEOUT_MS = 15_000;

interface FieldGroup {
  id: string;
  title: string;
  description?: string;
  fields: FieldDef[];
}

const FIELDS: FieldDef[] = [
  { key: "Gender", label: "Gender", type: "select", options: ["Male", "Female"] },
  { key: "SeniorCitizen", label: "Senior Citizen", type: "select", options: ["Yes", "No"] },
  { key: "Partner", label: "Partner", type: "select", options: ["Yes", "No"] },
  { key: "Dependents", label: "Has Dependents", type: "select", options: ["Yes", "No"] },
  { key: "Married", label: "Married", type: "select", options: ["Yes", "No"] },
  { key: "Under30", label: "Under 30", type: "select", options: ["Yes", "No"] },
  { key: "ReferredAFriend", label: "Referred a Friend", type: "select", options: ["Yes", "No"] },
  { key: "Age", label: "Age", type: "number" },
  { key: "NumberOfDependents", label: "Dependents (Count)", type: "number" },
  { key: "NumberOfReferrals", label: "Referrals", type: "number" },
  { key: "SatisfactionScore", label: "Satisfaction Score (1-5)", type: "number" },
  { key: "CLTV", label: "CLTV", type: "number" },
  { key: "tenure", label: "Tenure (Months)", type: "number" },
  { key: "Contract", label: "Contract", type: "select", options: ["Month-to-Month", "One Year", "Two Year"] },
  { key: "Offer", label: "Offer", type: "select", options: ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"] },
  { key: "PaperlessBilling", label: "Paperless Billing", type: "select", options: ["Yes", "No"] },
  { key: "PaymentMethod", label: "Payment Method", type: "select", options: ["Bank Withdrawal", "Credit Card", "Mailed Check"] },
  { key: "PhoneService", label: "Phone Service", type: "select", options: ["Yes", "No"] },
  { key: "MultipleLines", label: "Multiple Lines", type: "select", options: ["Yes", "No"] },
  { key: "InternetService", label: "Internet Service", type: "select", options: ["Yes", "No"] },
  { key: "InternetType", label: "Internet Type", type: "select", options: ["DSL", "Fiber Optic", "Cable", "None"] },
  { key: "OnlineSecurity", label: "Online Security", type: "select", options: ["Yes", "No"] },
  { key: "OnlineBackup", label: "Online Backup", type: "select", options: ["Yes", "No"] },
  { key: "DeviceProtection", label: "Device Protection", type: "select", options: ["Yes", "No"] },
  { key: "TechSupport", label: "Premium Support", type: "select", options: ["Yes", "No"] },
  { key: "StreamingTV", label: "Streaming TV", type: "select", options: ["Yes", "No"] },
  { key: "StreamingMovies", label: "Streaming Movies", type: "select", options: ["Yes", "No"] },
  { key: "StreamingMusic", label: "Streaming Music", type: "select", options: ["Yes", "No"] },
  { key: "UnlimitedData", label: "Unlimited Data", type: "select", options: ["Yes", "No"] },
  { key: "AvgMonthlyLongDistanceCharges", label: "Avg LD Charges", type: "number" },
  { key: "AvgMonthlyGBDownload", label: "Avg GB Download", type: "number" },
  { key: "MonthlyCharges", label: "Monthly Charge", type: "number" },
  { key: "TotalCharges", label: "Total Charges", type: "number" },
  { key: "TotalRefunds", label: "Total Refunds", type: "number" },
  { key: "TotalExtraDataCharges", label: "Extra Data Charges", type: "number" },
  { key: "TotalLongDistanceCharges", label: "Total LD Charges", type: "number" },
  { key: "TotalRevenue", label: "Total Revenue", type: "number" },
];

const FIELD_MAP = new Map(FIELDS.map((f) => [f.key, f]));

const GROUPS: FieldGroup[] = [
  {
    id: "personal",
    title: "Personal & Account",
    description: "Demographics, tenure, and contract terms.",
    fields: ["Gender", "SeniorCitizen", "Partner", "Dependents", "Married", "Under30", "ReferredAFriend", "Age", "NumberOfDependents", "NumberOfReferrals", "SatisfactionScore", "tenure", "Contract", "Offer"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    id: "services",
    title: "Services",
    description: "Connectivity and protection add-ons.",
    fields: ["PhoneService", "MultipleLines", "InternetService", "InternetType", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    id: "streaming",
    title: "Streaming & Media",
    description: "Content subscriptions and billing preferences.",
    fields: ["StreamingTV", "StreamingMovies", "StreamingMusic", "UnlimitedData", "PaperlessBilling", "PaymentMethod"]
      .map((k) => FIELD_MAP.get(k)!),
  },
  {
    id: "charges",
    title: "Charges & Usage",
    description: "Monthly spend, lifetime value, and refunds.",
    fields: ["MonthlyCharges", "TotalCharges", "TotalRefunds", "TotalExtraDataCharges", "TotalLongDistanceCharges", "TotalRevenue", "AvgMonthlyLongDistanceCharges", "AvgMonthlyGBDownload", "CLTV"]
      .map((k) => FIELD_MAP.get(k)!),
  },
];

/** Default values aligned with the Zod schema (everything optional / nullable). */
const DEFAULT_VALUES = Object.fromEntries(
  FIELDS.map((f) => [f.key, f.type === "number" ? "" : ""])
) as FormValues;

export default function ParametersPage() {
  const router = useRouter();
  const persistedValues = useFormStore((s) => s.values);
  const loadPreset = useFormStore((s) => s.loadPreset);
  const resetFormStore = useFormStore((s) => s.reset);
  const setResults = useResultStore((s) => s.setResults);

  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [openGroups, setOpenGroups] = useState<Set<string>>(
    () => new Set(GROUPS.map((g) => g.id))
  );
  /** In-flight submission token. Using a ref (not state) means the
   *  unmount-cleanup effect never re-fires on render and the in-flight
   *  ``fetch`` is never aborted by a state-driven re-render. The
   *  number is the timeout id; bumping it cancels the prior timeout
   *  and lets us ignore responses from superseded submissions. */
  const inFlightRef = useRef<{ controller: AbortController; token: number } | null>(null);

  const initialValues = useMemo<FormValues>(
    () => ({ ...DEFAULT_VALUES, ...persistedValues }),
    // We only want to seed once from the store; subsequent edits to the
    // store shouldn't reset the user's in-progress form entries.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const {
    control,
    handleSubmit,
    reset,
    getValues,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({
    defaultValues: initialValues,
    resolver: zodResolver(ChurnInputSchema) as never,
    mode: "onBlur",
  });

  // Abort in-flight requests only on unmount. Storing the controller
  // in a ref (not state) means a parent re-render cannot re-fire this
  // cleanup and kill a request mid-flight.
  useEffect(() => {
    return () => {
      inFlightRef.current?.controller.abort();
      inFlightRef.current = null;
    };
  }, []);

  const toggleGroup = (id: string) => {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const allOpen = openGroups.size === GROUPS.length;
  const toggleAll = () => {
    setOpenGroups(allOpen ? new Set() : new Set(GROUPS.map((g) => g.id)));
  };

  const applyPreset = (presetId: string) => {
    const preset = PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    // Convert binary Yes/No strings to 0/1 BEFORE handing to RHF, so the
    // form starts in a Zod-valid state without waiting for the user to
    // touch each field.
    const normalized = normalizeBinaryValues(preset.values) as FormValues;
    // Update the form first so the inputs reflect the new state, then
    // mirror into the persisted store.
    reset({ ...DEFAULT_VALUES, ...normalized });
    loadPreset({ values: normalized });
  };

  const handleReset = () => {
    reset(DEFAULT_VALUES);
    resetFormStore();
  };

  /** Convert empty strings / null / undefined to null. Numeric and binary
   *  fields are already in their final form because `FormField` and
   *  `normalizeBinaryValues` normalize them at the form-state level. */
  const normalizeForApi = (raw: FormValues): ChurnInput => {
    const out: Record<string, string | number | null | undefined> = {};
    for (const [k, v] of Object.entries(raw)) {
      if (v === "" || v === null || v === undefined) {
        out[k] = null;
        continue;
      }
      out[k] = v;
    }
    return out as ChurnInput;
  };

  const onSubmit = handleSubmit(async (formValues) => {
    setLoading(true);
    setSubmitError(null);

    // Cancel a prior in-flight submission (button mashing) but never
    // the very first one or the one we are about to make.
    inFlightRef.current?.controller.abort();
    const controller = new AbortController();
    const timeoutId = window.setTimeout(
      () => controller.abort(),
      SUBMIT_TIMEOUT_MS,
    );
    inFlightRef.current = { controller, token: timeoutId };

    /** True if the caller is still the active submission. Prevents
     *  a slow response from clobbering a newer submission's state. */
    const isCurrent = () =>
      inFlightRef.current?.token === timeoutId;

    const clear = () => {
      window.clearTimeout(timeoutId);
      if (inFlightRef.current?.token === timeoutId) {
        inFlightRef.current = null;
      }
    };

    try {
      const payload = normalizeForApi(formValues);
      const validated = ChurnInputSchema.parse(payload);

      const predictRes = await apiFetch(`/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
        signal: controller.signal,
      });

      if (!isCurrent()) return;

      if (!predictRes.ok) {
        const errBody = await predictRes.json().catch(() => null);
        throw new Error(
          errBody?.detail || `Prediction failed with status ${predictRes.status}`
        );
      }

      const predData: PredictResponse = await predictRes.json();

      let scriptData: RetentionScriptResponse;
      try {
        scriptData = await fetchRetentionScript(
          buildRetentionRequest(predData, predData.feature_importance, getValues())
        );
      } catch {
        scriptData = { script: "Failed to generate script." };
      }

      if (!isCurrent()) return;

      setResults(predData, scriptData);
      clear();
      setLoading(false);
      router.push("/analysis");
    } catch (err) {
      if (!isCurrent()) return;
      clear();
      // AbortError from a user-driven re-submit is expected; do not
      // surface it as a red banner. The unmount-cleanup path is the
      // other legitimate source; both are silent.
      if (err instanceof DOMException && err.name === "AbortError") {
        setLoading(false);
        return;
      }
      setSubmitError(
        err instanceof Error ? err.message : "Analysis request failed"
      );
      setLoading(false);
    }
  });

  const onInvalid = (errs: typeof errors) => {
    // Open every group that has a visible error so the user immediately
    // sees what's wrong.
    const errKeys = Object.keys(errs);
    if (errKeys.length === 0) return;
    const groupsToOpen = new Set(openGroups);
    for (const key of errKeys) {
      for (const g of GROUPS) {
        if (g.fields.some((f) => f.key === key)) groupsToOpen.add(g.id);
      }
    }
    setOpenGroups(groupsToOpen);
  };

  const errorFor = (key: string): string | undefined => {
    const path = key as keyof FormValues;
    const err = errors[path];
    if (!err) return undefined;
    return err.message as string;
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans antialiased overflow-x-hidden">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 flex flex-col gap-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-xs uppercase tracking-widest text-white/50 hover:text-white transition-colors"
            >
              ← Back to Engine
            </Link>
            <h1 className="text-2xl font-bold tracking-widest uppercase">Input Engine</h1>
          </div>
          <span className="text-xs uppercase tracking-widest text-white/50">Telco Parameters</span>
        </header>

        {submitError && (
          <div
            role="alert"
            className="p-4 border border-red text-red text-sm font-mono tracking-wide"
          >
            {submitError}
          </div>
        )}

        <div className="flex flex-col gap-4 border border-white/10 p-4">
          <div className="flex flex-col md:flex-row md:items-center gap-3 md:gap-4">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-white/40" />
              <span className="text-xs uppercase tracking-widest text-white/40">
                Presets
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => applyPreset(preset.id)}
                  className="px-3 py-1.5 border border-white/10 text-[11px] uppercase tracking-widest text-white/60 hover:text-white hover:border-white/40 transition-colors"
                  title={preset.blurb}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={toggleAll}
              className="md:ml-auto px-3 py-1.5 text-[11px] uppercase tracking-widest text-white/40 hover:text-white transition-colors"
              aria-pressed={allOpen}
            >
              {allOpen ? "Collapse All" : "Expand All"}
            </button>
          </div>
        </div>

        <form
          onSubmit={onSubmit}
          onInvalid={() => onInvalid(errors)}
          noValidate
          className="flex flex-col gap-y-8"
        >
          {GROUPS.map((group) => {
            const isOpen = openGroups.has(group.id);
            const groupHasError = group.fields.some((f) => errorFor(f.key));
            return (
              <section
                key={group.id}
                className={cn(
                  "border p-6 transition-colors",
                  groupHasError
                    ? "border-red/40 bg-red/5"
                    : "border-white/10 bg-zinc-950"
                )}
              >
                <button
                  type="button"
                  onClick={() => toggleGroup(group.id)}
                  aria-expanded={isOpen}
                  aria-controls={`group-${group.id}`}
                  className="w-full flex items-center justify-between gap-4 pb-4 border-b border-white/10 text-left"
                >
                  <div className="flex flex-col gap-1">
                    <h2 className="text-sm uppercase tracking-widest text-white/70 font-sans">
                      {group.title}
                      {groupHasError && (
                        <span className="ml-2 text-[10px] text-red">· needs attention</span>
                      )}
                    </h2>
                    {group.description && (
                      <p className="text-[11px] text-white/40">{group.description}</p>
                    )}
                  </div>
                  {isOpen ? (
                    <ChevronUp size={16} className="text-white/40" />
                  ) : (
                    <ChevronDown size={16} className="text-white/40" />
                  )}
                </button>
                {isOpen && (
                  <div
                    id={`group-${group.id}`}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pt-6"
                  >
                    {group.fields.map((field) => (
                      <Controller
                        key={field.key}
                        name={field.key as keyof FormValues}
                        control={control}
                        render={({ field: rhf, fieldState }) => (
                          <FormField
                            field={field}
                            value={rhf.value}
                            onChange={(_k, v) => rhf.onChange(v)}
                            onBlur={rhf.onBlur}
                            error={fieldState.error?.message}
                          />
                        )}
                      />
                    ))}
                  </div>
                )}
              </section>
            );
          })}

          <div className="sticky bottom-0 left-0 right-0 z-30 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-4 bg-black/85 backdrop-blur border-t border-white/10 flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={handleReset}
              disabled={loading}
              className="flex items-center gap-2 px-3 py-1.5 text-[11px] uppercase tracking-widest text-white/40 hover:text-white transition-colors disabled:opacity-50"
            >
              <RotateCcw size={12} />
              Reset
            </button>
            <button
              type="submit"
              disabled={loading || isSubmitting}
              className="flex items-center gap-3 px-8 py-3 bg-white text-black text-sm font-bold tracking-widest uppercase hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-sans"
            >
              {loading ? "Processing..." : "Analyze"}
              {!loading && <Send size={16} />}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
