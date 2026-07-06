import { z } from "zod";

// Binary fields accept either a number (0/1) or a numeric string ("0"/"1");
// z.coerce.number() converts "1" -> 1 while still rejecting non-numeric input.
const binaryFlag = z.coerce.number().int().min(0).max(1).nullable().optional();

export const ChurnInputSchema = z.object({
  Gender: z.string().nullable().optional(),
  SeniorCitizen: binaryFlag,
  Partner: binaryFlag,
  Dependents: binaryFlag,
  tenure: z.coerce.number().int().nonnegative().nullable().optional(),
  PhoneService: binaryFlag,
  MultipleLines: binaryFlag,
  InternetService: binaryFlag,
  OnlineSecurity: binaryFlag,
  OnlineBackup: binaryFlag,
  DeviceProtection: binaryFlag,
  TechSupport: binaryFlag,
  StreamingTV: binaryFlag,
  StreamingMovies: binaryFlag,
  Contract: z.string().nullable().optional(),
  PaperlessBilling: binaryFlag,
  PaymentMethod: z.string().nullable().optional(),
  MonthlyCharges: z.coerce.number().nonnegative().nullable().optional(),
  TotalCharges: z.coerce.number().nonnegative().nullable().optional(),
  Married: binaryFlag,
  NumberOfDependents: z.coerce.number().int().nonnegative().nullable().optional(),
  NumberOfReferrals: z.coerce.number().int().nonnegative().nullable().optional(),
  SatisfactionScore: z.coerce.number().int().min(1).max(5).nullable().optional(),
  InternetType: z.string().nullable().optional(),
  Offer: z.string().nullable().optional(),
  Age: z.coerce.number().int().nonnegative().nullable().optional(),
  AvgMonthlyGBDownload: z.coerce.number().int().nonnegative().nullable().optional(),
  AvgMonthlyLongDistanceCharges: z.coerce.number().nonnegative().nullable().optional(),
  CLTV: z.coerce.number().int().nonnegative().nullable().optional(),
  Under30: binaryFlag,
  UnlimitedData: binaryFlag,
  StreamingMusic: binaryFlag,
  ReferredAFriend: binaryFlag,
  TotalRefunds: z.coerce.number().nonnegative().nullable().optional(),
  TotalExtraDataCharges: z.coerce.number().int().nonnegative().nullable().optional(),
  TotalLongDistanceCharges: z.coerce.number().nonnegative().nullable().optional(),
  TotalRevenue: z.coerce.number().nonnegative().nullable().optional(),
});

export type ChurnInput = z.infer<typeof ChurnInputSchema>;

export const ChurnResponseSchema = z.object({
  prediction: z.number().int(),
  churn_probability: z.number(),
  retention_risk: z.string(),
  feature_importance: z
    .array(
      z.object({
        feature: z.string(),
        value: z.union([z.string(), z.number(), z.null()]),
        magnitude: z.number(),
        direction: z.enum(["up", "down"]),
      })
    )
    .nullable()
    .optional(),
});

export type ChurnResponse = z.infer<typeof ChurnResponseSchema>;

export const RetentionScriptResponseSchema = z.object({
  script: z.string(),
});

export type RetentionScriptResponse = z.infer<typeof RetentionScriptResponseSchema>;
