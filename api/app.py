"""FastAPI application for churn prediction and LLM-synthesized retention plans.

Engine overview
---------------
The model is a LightGBM classifier persisted at ``models/churn_model.pkl``
that was fit on a 50K-row tabular dataset streamed from Hugging Face at
training time. At inference, the API receives a customer record, runs
it through ``engineer_features_inference`` to produce a 51-column
numeric frame, and feeds that frame to the booster. Two outputs are
returned to the caller:

* a churn probability in [0, 1] and a risk tier (Low / Medium / High);
* a top-k list of per-feature log-odds contributions extracted from the
  booster's native ``pred_contrib=True`` mode (``src/explain.py``).
  Mathematically, ``sum(per_feature) + bias == logit(predict_proba)``
  to within float precision, so the SHAP attributions sum exactly to
  the model's log-odds. The frontend renders these as a horizontal
  bar panel sorted by ``|magnitude|``.

The retention planning endpoint (``/generate_retention_script``) is a
thin orchestrator over an LLM provider: it loads the API key from
``LLM_PROVIDER_API_KEY`` (env var or local ``.env``), asks the model
to synthesize an internal Executive Retention Strategy & Action Plan
grounded in the Tree-SHAP drivers, and tags the result with an
``[Action Plan]`` prefix. The output is for customer success managers
and retention analysts — never read aloud to the customer. If the
key is missing or the call fails, a labelled default action plan is
returned and the API stays healthy — the endpoint is never allowed
to raise.
"""

import asyncio
import json
import logging
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, ConfigDict, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from src.feature_engineering import engineer_features_inference
from src.config import MODEL_CONFIG
from src.explain import explain_prediction, DEFAULT_TOP_K as _EXPLAIN_TOP_K

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Secrets resolution. ``load_dotenv(override=False)`` populates
# os.environ from a local ``.env`` file at the repo root when present,
# but never overrides values that are already set — so platform-injected
# env vars (Render / Railway / HF Spaces) always win over a developer's
# local file.
load_dotenv(ROOT / ".env", override=False)

LLM_PROVIDER_API_KEY = os.environ.get("LLM_PROVIDER_API_KEY", "").strip()
PORT = int(os.environ.get("PORT", "8000"))

# Comma-separated list of allowed CORS origins. Production deployments
# (Render, Railway, Fly, Vercel previews) override this with a
# single env var rather than editing source. The dev defaults are
# always included so local `make dev` keeps working.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Production Vercel deploys. The default tuple must accept the
    # live dashboard origin so a fresh Render/Railway boot serves
    # traffic from these URLs without requiring CORS_ORIGINS to be
    # set explicitly. Operators can still override CORS_ORIGINS to
    # add staging, custom domains, or to remove these defaults.
    "https://frontend-pi-sage-79.vercel.app",
)
CORS_ORIGINS: tuple[str, ...] = tuple(
    o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
) + _DEFAULT_CORS_ORIGINS


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for log aggregators (Datadog, ELK).
    Includes all extra context fields and exception tracebacks."""

    BASE_KEYS = frozenset({
        "timestamp", "level", "logger", "message", "module", "line",
    })

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in self.BASE_KEYS and not key.startswith("_"):
                try:
                    data[key] = value
                except TypeError:
                    data[key] = str(value)
        if record.exc_info and record.exc_info[1] is not None:
            data["exception"] = str(record.exc_info[1])
        return json.dumps(data)


logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.handlers.clear()
logger.addHandler(_handler)
logger.propagate = False

_model_path = ROOT / MODEL_CONFIG["save_path"]
_threshold = float(MODEL_CONFIG.get("threshold", 0.5))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model artifact is uploaded once to a persistent disk on the
    # host (see `render.yaml`'s `disk:` block or the Railway volume in
    # `DEPLOYMENT.md`). On a fresh container boot the file may be
    # missing while the upload is still propagating; we want the
    # container to start in a degraded state (`/health` returns
    # `model_loaded: false`) rather than crash-loop, so /health stays
    # answerable and the platform can route traffic.
    if not _model_path.exists():
        logger.error(
            "MODEL_ARTIFACT_MISSING",
            extra={
                "path": str(_model_path),
                "remediation": (
                    "Upload models/churn_model.pkl to the persistent disk "
                    "mounted at /app/models (see DEPLOYMENT.md)."
                ),
            },
        )
        app.state.model = None
        app.state.expected_features = None
        executor = ThreadPoolExecutor(max_workers=4)
        app.state.executor = executor
        yield
        executor.shutdown(wait=False)
        logger.info("MODEL_RELEASED")
        return
    artifact = joblib.load(_model_path)
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        logger.error(
            "MODEL_ARTIFACT_CORRUPT",
            extra={"path": str(_model_path)},
        )
        app.state.model = None
        app.state.expected_features = None
        executor = ThreadPoolExecutor(max_workers=4)
        app.state.executor = executor
        yield
        executor.shutdown(wait=False)
        logger.info("MODEL_RELEASED")
        return
    app.state.model = pipeline
    app.state.expected_features = getattr(
        pipeline, "feature_name_", None
    )

    executor = ThreadPoolExecutor(max_workers=4)
    app.state.executor = executor

    logger.info("MODEL_LOADED", extra={"path": str(_model_path)})
    if app.state.expected_features is not None:
        logger.info(
            "EXPECTED_FEATURES",
            extra={"count": len(app.state.expected_features)},
        )
    yield
    executor.shutdown(wait=False)
    logger.info("MODEL_RELEASED")


app = FastAPI(
    title="Enterprise Churn Engine",
    description="Production churn prediction API powered by Polars + LightGBM (ML) and "
    "an LLM provider for retention script synthesis. Evaluates customer "
    "churn risk and generates actionable agent scripts.",
    version="v1.0.0",
    # Explicit so the values are visible at the call site (defaults are
    # the same, but the FE rewrite relies on ``/openapi.json`` and
    # ``/docs`` being available here).
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter

# Allow CI/load-test environments to bypass the rate limiter
if os.environ.get("LIMITER_ENABLED", "true").strip().lower() == "false":
    limiter.enabled = False

app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000)
    logger.info(
        "REQUEST_COMPLETE",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": elapsed,
        },
    )
    return response


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    Gender: str | None = Field(None)
    SeniorCitizen: int | None = Field(None)
    Partner: int | None = Field(None)
    Dependents: int | None = Field(None)
    tenure: int | None = Field(None)
    PhoneService: int | None = Field(None)
    MultipleLines: int | None = Field(None)
    InternetService: int | None = Field(None)
    OnlineSecurity: int | None = Field(None)
    OnlineBackup: int | None = Field(None)
    DeviceProtection: int | None = Field(None)
    TechSupport: int | None = Field(None)
    StreamingTV: int | None = Field(None)
    StreamingMovies: int | None = Field(None)
    Contract: str | None = Field(None)
    PaperlessBilling: int | None = Field(None)
    PaymentMethod: str | None = Field(None)
    MonthlyCharges: float | None = Field(None)
    TotalCharges: float | None = Field(None)
    Married: int | None = Field(None)
    NumberOfDependents: int | None = Field(None)
    NumberOfReferrals: int | None = Field(None)
    SatisfactionScore: int | None = Field(None)
    InternetType: str | None = Field(None)
    Offer: str | None = Field(None)
    Age: int | None = Field(None)
    AvgMonthlyGBDownload: int | None = Field(None)
    AvgMonthlyLongDistanceCharges: float | None = Field(None)
    CLTV: int | None = Field(None)
    Under30: int | None = Field(None)
    UnlimitedData: int | None = Field(None)
    StreamingMusic: int | None = Field(None)
    ReferredAFriend: int | None = Field(None)
    TotalRefunds: float | None = Field(None)
    TotalExtraDataCharges: int | None = Field(None)
    TotalLongDistanceCharges: float | None = Field(None)
    TotalRevenue: float | None = Field(None)

    @field_validator("*")
    @classmethod
    def _reject_nan_infinity(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("NaN or Infinity is not allowed for float fields")
        return v


class FeatureImportance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str = Field(..., description="Human-readable feature label.")
    value: float | int | str | None = Field(
        None,
        description="The value the customer record had for this feature.",
    )
    magnitude: float = Field(
        ...,
        description="Absolute SHAP contribution in log-odds space.",
    )
    direction: Literal["up", "down"] = Field(
        ...,
        description='Either "up" (pushing toward churn) or "down" (away from churn).',
        json_schema_extra={"example": "up"},
    )


class ChurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: int = Field(..., description="1 = Churned, 0 = Stayed")
    churn_probability: float = Field(..., description="Model confidence score [0, 1]")
    retention_risk: str = Field(..., description="High / Medium / Low risk tier")
    feature_importance: list[FeatureImportance] | None = Field(
        None,
        description=(
            "Top SHAP feature attributions for this prediction. "
            "Null if explainability is unavailable for this request."
        ),
    )


class BatchChurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[ChurnResponse]
    total_records: int
    high_risk_count: int


class RetentionScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: str = Field(
        ...,
        description="Risk tier returned by the prediction endpoint.",
        json_schema_extra={"example": "High"},
    )
    reasons: str = Field(
        ...,
        description="Key churn reasons.",
        json_schema_extra={"example": "Customer cited billing confusion and lack of usage."},
    )
    top_drivers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Top Tree-SHAP drivers as ['Feature (magnitude)', ...]. "
            "When provided, the prompt weaves them in as a dedicated "
            "evidence block so the LLM grounds its script in the actual "
            "model output rather than a stock sentence."
        ),
    )
    risk_signals: Optional[List[str]] = Field(
        default=None,
        description=(
            "Practical-precaution titles already derived client-side. "
            "Used as extra context for the LLM; not a hard requirement."
        ),
    )
    probability_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Churn probability as a 0-100 percentage for the prompt header.",
    )


class RetentionScriptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    script: str = Field(
        ...,
        description="A 2-sentence retention script for the customer service agent.",
    )


_engineer_features = engineer_features_inference


def _classify(proba: float) -> dict:
    return {
        "prediction": int(proba >= _threshold),
        "churn_probability": round(proba, 4),
        "retention_risk": "High" if proba >= 0.70 else "Medium" if proba >= 0.40 else "Low",
    }


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Customer Churn Prediction API. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health_check(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(request.app.state, "model"),
        "model_path": str(_model_path),
    }


def _align_to_model(df: pd.DataFrame, expected: list[str] | None) -> pd.DataFrame:
    """Align an engineered DataFrame to the model's expected column order.

    Behavior:

    * Missing columns are back-filled with 0 and logged as a warning.
      This preserves the prior API contract — an empty / partial
      payload still returns HTTP 200 with a "no-signal" prediction
      (every blank field is treated as 0, the model's neutral default).
    * Extra columns (not in ``expected``) are dropped and logged.
    * If the model's expected feature list is unavailable (``None``)
      we return the frame as-is.
    * When the produced frame is *fully* empty (zero columns) — the
      engineering pipeline returned nothing — we raise
      :class:`ValueError` so the API responds 422 rather than calling
      ``predict_proba`` on an empty frame.
    """
    if expected is None:
        return df

    expected_set = set(expected)
    produced_set = set(df.columns)

    missing = [c for c in expected if c not in produced_set]
    extras = [c for c in produced_set if c not in expected_set]

    if not produced_set and missing:
        # Engineer produced nothing — refuse the prediction rather than
        # feeding the model a frame with only zero-filled columns
        # (which would be a meaningless prediction anyway).
        raise ValueError(
            "Engineered feature frame is empty; the model cannot predict."
        )

    if missing:
        logger.warning(
            "MISSING_FEATURES_AT_INFERENCE",
            extra={"missing": missing, "produced": sorted(produced_set)},
        )
        for col in missing:
            df[col] = 0

    if extras:
        logger.warning(
            "EXTRA_FEATURES_AT_INFERENCE",
            extra={"extras": sorted(extras)},
        )
        df = df.drop(columns=list(extras))

    return df[expected]


# Numeric 0/1 binary columns plus the two Senior Citizen one-hot
# columns plus the direct numerics. Listed here for visibility — the
# alignment helper back-fills all missing columns with 0 by default
# (matching the previous API behavior). Kept for the production log
# context: if the inference pipeline ever silently fails to produce one
# of these, ``MISSING_FEATURES_AT_INFERENCE`` will surface the issue.
STRUCTURAL_FEATURE_COLUMNS: frozenset[str] = frozenset({
    "Age", "CLTV", "Dependents", "Device_Protection_Plan",
    "Internet_Service", "Married", "Multiple_Lines",
    "Number_of_Dependents", "Number_of_Referrals", "Online_Backup",
    "Online_Security", "Paperless_Billing", "Partner", "Phone_Service",
    "Premium_Tech_Support", "Referred_a_Friend", "Streaming_Movies",
    "Streaming_Music", "Streaming_TV", "Under_30", "Unlimited_Data",
    "Tenure_in_Months", "Monthly_Charge", "Total_Charges",
    "Avg_Monthly_GB_Download", "Avg_Monthly_Long_Distance_Charges",
    "Total_Refunds", "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges", "Total_Revenue",
    "Satisfaction_Score", "Senior_Citizen_0", "Senior_Citizen_1",
})


def _process_prediction(pipeline, expected, record):
    df = _engineer_features(pd.DataFrame([record]))
    df = _align_to_model(df, expected)
    proba = float(pipeline.predict_proba(df)[0][1])
    result = _classify(proba)
    result["feature_importance"] = explain_prediction(pipeline, df, top_k=_EXPLAIN_TOP_K)
    return result


@app.post("/predict", response_model=ChurnResponse, tags=["Machine Learning"])
@limiter.limit("10/minute")
async def predict_endpoint(customer: CustomerFeatures, request: Request):
    try:
        pipeline = request.app.state.model
        expected = request.app.state.expected_features

        record = customer.model_dump(by_alias=False, exclude_none=False)
        record = {k: v for k, v in record.items() if v is not None}

        loop = asyncio.get_running_loop()
        result_dict = await loop.run_in_executor(
            request.app.state.executor, _process_prediction, pipeline, expected, record
        )
        return ChurnResponse(**result_dict)

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except TypeError as te:
        logger.exception("PREDICTION_TYPE_ERROR", extra={"error": str(te)})
        raise HTTPException(status_code=500, detail="Type evaluation failed during prediction.")
    except Exception as e:
        logger.exception("PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")


def _process_batch_prediction(pipeline, expected, records):
    df = _engineer_features(pd.DataFrame(records))
    df = _align_to_model(df, expected)

    probas = pipeline.predict_proba(df)[:, 1]
    results = []
    for i, proba in enumerate(probas):
        item = _classify(float(proba))
        row_df = df.iloc[[i]]
        item["feature_importance"] = explain_prediction(
            pipeline, row_df, top_k=_EXPLAIN_TOP_K
        )
        results.append(ChurnResponse(**item))
    high_risk = sum(1 for r in results if r.retention_risk == "High")
    return BatchChurnResponse(
        results=results,
        total_records=len(results),
        high_risk_count=high_risk,
    )


@app.post("/predict/batch", response_model=BatchChurnResponse, tags=["Machine Learning"])
async def predict_batch_endpoint(customers: list[CustomerFeatures], request: Request):
    if not customers:
        raise HTTPException(status_code=422, detail="Batch cannot be empty.")
    try:
        pipeline = request.app.state.model
        expected = request.app.state.expected_features

        records = [
            {k: v for k, v in c.model_dump(by_alias=False, exclude_none=False).items()
             if v is not None}
            for c in customers
        ]

        loop = asyncio.get_running_loop()
        batch_response = await loop.run_in_executor(
            request.app.state.executor, _process_batch_prediction, pipeline, expected, records
        )
        return batch_response

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("BATCH_PREDICTION_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error.")


_FALLBACK_SCRIPT = (
    "[Default Action Plan]\n"
    "- Audit the customer's actual usage against the active plan tier; "
    "if utilization is below 40% prepare a same-day downgrade offer with a "
    "fixed-price 12-month hold to remove renewal anxiety.\n"
    "- Open a satisfaction-recovery workflow: senior CSM outreach within 24h, "
    "acknowledge the recent friction, attach a one-time goodwill credit "
    "(5-10% of MRR), and define a 30-day satisfaction re-score target.\n"
    "- Lock in tenure: pair any discount with a 6- or 12-month contract at "
    "the current price; waive early-termination fees for the first 60 days "
    "to remove the customer's switching cost.\n"
    "- Migrate the customer off mailed check to autopay with a one-time $10 "
    "credit; route to a tech-support trial if internet-type churn is the "
    "primary driver."
)


# Catalog of model identifiers the provider can route to. Standard is
# the default. Recruiters and testers can switch via the in-app
# Provider Configuration Panel, which sends X-Provider-Model.
#
# Real model ids are tenant-specific (each provider publishes its own).
# ``LLM_STANDARD_MODEL`` and ``LLM_HIGH_CAPACITY_MODEL`` env vars let
# ops point the deployment at the actual ids in use; the defaults
# below are the publicly available Groq production models.
LLM_MODELS: dict[str, str] = {
    "standard": os.environ.get("LLM_STANDARD_MODEL", "llama-3.1-8b-instant"),
    "high_capacity": os.environ.get("LLM_HIGH_CAPACITY_MODEL", "llama-3.3-70b-versatile"),
}
LLM_DEFAULT_MODEL = LLM_MODELS["standard"]


def _resolve_provider_credentials(request: Request) -> tuple[str, str]:
    """Pick the API key + model the caller wants.

    Precedence: ``X-Provider-Key`` / ``X-Provider-Model`` request
    headers (set by the in-app Provider Configuration Panel) override
    the module-level env defaults. Missing headers fall back to the
    values loaded from the platform environment / ``.env`` file.
    """
    api_key = (request.headers.get("X-Provider-Key") or "").strip() or LLM_PROVIDER_API_KEY
    requested = (request.headers.get("X-Provider-Model") or "").strip()
    model = LLM_MODELS.get(requested, LLM_DEFAULT_MODEL)
    if requested and requested not in LLM_MODELS:
        logger.info(
            "LLM_MODEL_UNKNOWN: %s — falling back to %s",
            requested, LLM_DEFAULT_MODEL,
        )
    return api_key, model


def _get_llm_client(api_key: str) -> Groq:
    """Build the LLM provider client. Raises if the key is empty."""
    if not api_key:
        raise RuntimeError("LLM provider key is not configured")
    return Groq(api_key=api_key)


def _generate_script(
    prompt: str,
    api_key: str,
    model: str = LLM_DEFAULT_MODEL,
) -> str:
    """Generate a retention script via the configured LLM provider.

    Always returns a string — never raises. The fallback path is hit
    when the API key is missing OR the LLM call fails. Three log tags
    distinguish the cause:

    * ``LLM_PROVIDER_KEY_MISSING`` — set ``LLM_PROVIDER_API_KEY`` in
      your platform's environment (Render / Railway / HF Spaces), in a
      local ``.env`` file, or via the Provider Configuration Panel in
      the UI. See README for placement.
    * ``LLM_GENERATION_FAILED`` — the call was made but failed
      (network, quota, malformed prompt). Inspect the captured
      exception in the log.
    * ``LLM_MODEL_UNKNOWN`` — an unrecognised model id was passed via
      ``X-Provider-Model``; the call fell back to the default.
    """
    if not api_key:
        logger.warning(
            "LLM_PROVIDER_KEY_MISSING: set LLM_PROVIDER_API_KEY or use "
            "the Provider Configuration Panel; using fallback."
        )
        return _FALLBACK_SCRIPT

    try:
        client = _get_llm_client(api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=10.0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.error("LLM_GENERATION_FAILED", extra={"error": "empty_content", "model": model})
            return _FALLBACK_SCRIPT
        return "[Action Plan] " + content
    except Exception as exc:
        logger.error("LLM_GENERATION_FAILED", extra={"error": str(exc), "model": model})
        return _FALLBACK_SCRIPT


@app.post(
    "/generate_retention_script",
    response_model=RetentionScriptResponse,
    tags=["Generative AI"],
)
@limiter.limit("5/minute")
async def generate_retention_script(request_payload: RetentionScriptRequest, request: Request):
    """Synthesize a retention script using the configured LLM provider.

    The prompt is anchored on the customer's risk tier and the top
    Tree-SHAP driver reasons (extracted from the booster's per-feature
    log-odds contributions in ``src/explain.py``). The provider's key
    and model can be overridden per-request via the
    ``X-Provider-Key`` and ``X-Provider-Model`` headers (set by the
    in-app Provider Configuration Panel). Missing key or LLM failure
    routes to a labelled static fallback so this endpoint is never
    allowed to raise.
    """
    api_key, model = _resolve_provider_credentials(request)
    prompt = _build_retention_prompt(request_payload)
    loop = asyncio.get_running_loop()
    script = await loop.run_in_executor(
        request.app.state.executor, _generate_script, prompt, api_key, model
    )
    return RetentionScriptResponse(script=script)


def _build_retention_prompt(req: RetentionScriptRequest) -> str:
    """Compose the LLM prompt from the structured request payload.

    The output is an **internal** Executive Retention Strategy & Action
    Plan intended exclusively for customer success managers and
    retention analysts — never read aloud to the customer. The model
    is instructed to deliver 3-4 high-density, SHAP-grounded business
    precautions and tactical counter-measures.
    """
    header = (
        f"You are an internal customer-success strategist. "
        f"The account is flagged as a {req.risk_level} churn risk"
    )
    if req.probability_pct is not None:
        header += f" (predicted probability {req.probability_pct:.1f}%)"
    header += "."

    body = f"Operational notes from the form: {req.reasons}."

    evidence_parts: List[str] = []
    if req.top_drivers:
        bullets = "\n".join(f"  - {d}" for d in req.top_drivers)
        evidence_parts.append(
            "Top Tree-SHAP drivers (log-odds contribution, descending):\n"
            + bullets
        )
    if req.risk_signals:
        bullets = "\n".join(f"  - {s}" for s in req.risk_signals)
        evidence_parts.append(
            "Customer-specific signals flagged for action:\n" + bullets
        )

    evidence = ""
    if evidence_parts:
        evidence = "\n\n" + "\n\n".join(evidence_parts)

    instructions = (
        "\n\nProduce an internal 'Executive Retention Strategy & Action Plan' "
        "for the customer success team. Strict requirements:\n"
        "1. Output exactly 3 to 4 distinct, high-density bullet points. No more, no less.\n"
        "2. Each bullet must name a concrete action the organization must take "
        "behind the scenes or offer strategically to secure the account "
        "(e.g., a discount, a contract migration, a satisfaction-recovery "
        "outreach, a tech-support trial, a payment-method migration, a "
        "bundle rebalance, a CLTV escalation).\n"
        "3. Each bullet must be grounded in the strongest Tree-SHAP driver(s) "
        "above; reference the specific feature (SatisfactionScore, Tenure_in_Months, "
        "Contract_Month_to_Month, Monthly_Charge, CLTV, etc.) and the magnitude.\n"
        "4. No greetings, no customer-facing dialogue, no 'Hello [Name]', no "
        "scripts to read aloud. The output is for internal analyst use only.\n"
        "5. Each bullet is one to two sentences. Use imperative voice. No preamble, "
        "no closing summary, no markdown headings."
    )

    return f"{header} {body}{evidence}{instructions}"


@app.get("/llm/models", tags=["Generative AI"])
def list_llm_models():
    """Catalog of model identifiers the in-app Provider Configuration
    Panel can route to. Returned to the FE so the dropdown stays in
    sync with the backend."""
    return {"models": LLM_MODELS, "default": LLM_DEFAULT_MODEL}
