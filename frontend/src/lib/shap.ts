import type { FeatureImportance, PredictResponse } from "../types/api";
import type { FormValues } from "../store/useFormStore";

/** Top-k contributors by absolute magnitude, ties broken stably. */
export function topDrivers(
  features: FeatureImportance[] | null | undefined,
  k = 3,
): FeatureImportance[] {
  if (!features || features.length === 0) return [];
  return [...features]
    .sort((a, b) => Math.abs(b.magnitude) - Math.abs(a.magnitude))
    .slice(0, k);
}

/** Render the top drivers as a one-line string the LLM can quote
 *  without re-parsing. Positive magnitudes push toward churn. */
export function formatDriversForLlm(
  features: FeatureImportance[],
): string {
  return features
    .map((f) => {
      const sign = f.direction === "up" ? "+" : "-";
      const v = f.value === null || f.value === undefined ? "?" : String(f.value);
      return `${f.feature}=${v} (${sign}${f.magnitude.toFixed(3)})`;
    })
    .join("; ");
}

/** Practical precaution derived from a single up-direction SHAP
 *  driver. Each driver gets a tailored action; the unknown ones
 *  fall back to a generic prompt. */
interface Precaution {
  id: string;
  title: string;
  body: string;
}

const PRECAUTION_RULES: Array<{
  match: (f: string) => boolean;
  build: (f: FeatureImportance) => Precaution;
}> = [
  {
    match: (f) => /satisfaction/i.test(f),
    build: (f) => ({
      id: `satisfaction-${f.magnitude}`,
      title: "Satisfaction-recovery outreach",
      body:
        "Open a ticket within 24h. Senior CS rep calls the customer, acknowledges the recent friction, and offers a goodwill credit. Goal: lift Satisfaction Score from current to 4 within 30 days.",
    }),
  },
  {
    match: (f) => /tenure/i.test(f),
    build: (f) => ({
      id: `tenure-${f.magnitude}`,
      title: "Early-tenure retention play",
      body:
        "Enroll in a 90-day onboarding concierge. Pair with a curated walkthrough of premium features they have not yet tried; surface a 6-month price-lock to remove the renewal anxiety.",
    }),
  },
  {
    match: (f) => /contract/i.test(f),
    build: () => ({
      id: "contract-mtm",
      title: "Contract migration incentive",
      body:
        "Offer a 12-month contract at the current price with a one-time bill credit, no early-termination fee. Avoid a hard upsell; the goal is locking the term, not the tier.",
    }),
  },
  {
    match: (f) => /tech support|premium support/i.test(f),
    build: () => ({
      id: "tech-support",
      title: "Premium support trial",
      body:
        "Grant 60 days of Premium Support at no charge. Highlight the 24/7 channel and remote-diagnostics benefit. Track adoption so the renewal conversation has a concrete success metric.",
    }),
  },
  {
    match: (f) => /monthly charge|pricing|total charge/i.test(f),
    build: () => ({
      id: "pricing",
      title: "Pricing review",
      body:
        "Run an account audit against the customer's actual usage. Present two lower-tier options with a side-by-side savings estimate; do not require a plan change, just make the choice visible.",
    }),
  },
  {
    match: (f) => /payment/i.test(f),
    build: () => ({
      id: "payment",
      title: "Payment-method migration",
      body:
        "Switch the customer to bank withdrawal or credit card autopay. Mail-check customers churn at a higher rate; offer a one-time $10 credit for completing the migration.",
    }),
  },
  {
    match: (f) => /internet type/i.test(f),
    build: () => ({
      id: "internet-type",
      title: "Service-fit review",
      body:
        "Fiber customers on Month-to-Month contracts churn most. Schedule a network-quality check; if the line is healthy, lead with the speed-upgrade story, not a discount.",
    }),
  },
  {
    match: (f) => /cltv/i.test(f),
    build: () => ({
      id: "cltv",
      title: "Lifetime-value flag",
      body:
        "Escalate to the retention specialist queue. CLTV-driven churn is the highest-impact cohort; route to a senior agent with a 5% discretionary budget.",
    }),
  },
  {
    match: (f) => /streaming|unlimited data/i.test(f),
    build: () => ({
      id: "streaming",
      title: "Bundle rebalance",
      body:
        "The customer is paying for streaming add-ons they may not use. Surface a 30-day usage summary; offer a leaner bundle that keeps the core connectivity and drops the unused media add-ons.",
    }),
  },
];

/** Build the practical-precautions list from the actual top SHAP
 *  drivers. Each driver is matched against the rule table; unmatched
 *  up-direction drivers fall back to a generic precaution. */
export function deriveRiskSignals(
  prediction: PredictResponse | null,
  features: FeatureImportance[] | null | undefined,
  formValues: FormValues | null,
): Precaution[] {
  if (!prediction) return [];

  const drivers = topDrivers(features, 5).filter(
    (f) => f.direction === "up" && f.magnitude >= 0.05,
  );

  const seen = new Set<string>();
  const out: Precaution[] = [];

  for (const d of drivers) {
    for (const rule of PRECAUTION_RULES) {
      if (rule.match(d.feature) && !seen.has(d.feature)) {
        out.push(rule.build(d));
        seen.add(d.feature);
        break;
      }
    }
  }

  // Universal precaution: baseline comparison when the model is
  // predicting meaningfully above the loyal baseline.
  if (prediction.churn_probability >= 0.4) {
    out.push({
      id: "p0-baseline",
      title: "Above-baseline alert",
      body:
        `Predicted churn probability is ${(prediction.churn_probability * 100).toFixed(1)}%. ` +
        "Open a 30-day hold while the targeted plays run; re-score on day 21.",
    });
  }

  // Input-side cautions: if a key input is missing or zero, the
  // prediction is a floor estimate. Surface that.
  if (formValues) {
    const missing = REQUIRED_INPUTS.filter((k) => {
      const v = formValues[k];
      return v === "" || v === null || v === undefined;
    });
    if (missing.length > 0) {
      out.push({
        id: "p0-inputs",
        title: "Incomplete customer record",
        body:
          `${missing.length} key field(s) are blank: ${missing.join(", ")}. ` +
          "The current probability is a floor estimate; collect the missing data before quoting a number to stakeholders.",
      });
    }
  }

  return out;
}

const REQUIRED_INPUTS = [
  "tenure",
  "Contract",
  "SatisfactionScore",
  "MonthlyCharges",
] as const;

/** Convenience wrapper that builds the full LLM payload from the
 *  current prediction and form values. */
export function buildRetentionRequest(
  prediction: PredictResponse,
  features: FeatureImportance[] | null | undefined,
  formValues: FormValues | null,
): RetentionRequestPayload {
  const drivers = topDrivers(features, 3);
  return {
    risk_level: prediction.retention_risk,
    reasons: formatDriversForLlm(drivers),
    top_drivers: drivers.map((d) => `${d.feature} (${d.magnitude.toFixed(3)})`),
    risk_signals: deriveRiskSignals(prediction, features, formValues).map(
      (p) => p.title,
    ),
    probability_pct: Number((prediction.churn_probability * 100).toFixed(1)),
  };
}

export interface RetentionRequestPayload {
  risk_level: string;
  reasons: string;
  top_drivers: string[];
  risk_signals: string[];
  probability_pct: number;
}
