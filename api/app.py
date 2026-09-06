"""FastAPI application for churn prediction and LLM retention plans.

Inference: a customer record is engineered into the 51-column numeric
frame the LightGBM booster expects; the response carries a churn
probability, a risk tier, and top-k per-feature log-odds contributions
from ``pred_contrib=True`` (``src/explain.py``).

``/generate_retention_script`` sends the SHAP-grounded evidence to the
configured LLM provider and prefixes successful output with
``[Action Plan]``. A missing key or provider failure returns a labelled
``[Default Action Plan]`` instead of raising.
"""

import asyncio
import json
import logging
import math
import os
import pathlib
import re
import sys
import threading
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
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
)
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from src.feature_engineering import engineer_features_inference
from src.config import MODEL_CONFIG
from src.explain import explain_prediction, DEFAULT_TOP_K as _EXPLAIN_TOP_K


def _client_key(request: Request) -> str:
    """Rate-limit bucket key.

    Behind platform proxies (HF Spaces) the socket peer is the proxy
    edge, which may differ per request - IP-keying on it would hand
    every request a fresh bucket and silently disable the limiter (the
    exact production drift this keyfunc fixes). Prefer the first
    X-Forwarded-For hop; fall back to the peer address.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if (client and client.host) else "unknown"


limiter = Limiter(key_func=_client_key, default_limits=["30/minute"])

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Secrets resolution. ``load_dotenv(override=False)`` populates
# os.environ from a local ``.env`` file at the repo root when present,
# but never overrides values that are already set - so platform-injected
# env vars (HF Spaces, Vercel, any container host) always win over a
# developer's local file.
load_dotenv(ROOT / ".env", override=False)

LLM_PROVIDER_API_KEY = os.environ.get("LLM_PROVIDER_API_KEY", "").strip()
PORT = int(os.environ.get("PORT", "8000"))


def _resolve_git_commit() -> str:
    """Deployment version reported by ``/health`` for drift tracking.

    Lets anyone verify that the running backend matches the repository
    (repo ``master`` <-> deployed Space). Resolution order:

    1. ``GIT_COMMIT_SHA`` env var - set as a Hugging Face Space variable
       by ``deploy_huggingface.py`` (and injectable by any container host).
    2. ``COMMIT_SHA`` file next to the app - written by image builds that
       bake the version in at build time.
    3. ``"unknown"`` - local/dev runs where neither is present.
    """
    sha = os.environ.get("GIT_COMMIT_SHA", "").strip()
    if sha:
        return sha
    marker = ROOT / "COMMIT_SHA"
    if marker.is_file():
        stamped = marker.read_text(encoding="utf-8").strip()
        if stamped:
            return stamped
    return "unknown"


GIT_COMMIT_SHA = _resolve_git_commit()

# Dev origins are always included so local `make dev` keeps working;
# the production origin is accepted without requiring CORS_ORIGINS to
# be set. Operators can override CORS_ORIGINS entirely.
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://customer-churn-engine.vercel.app",
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
        return json.dumps(data, default=str)


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
    # On a fresh container boot the artifact may still be propagating;
    # boot degraded (/health reports model_loaded: false) rather than
    # crash-loop so the platform can route traffic.
    if not _model_path.exists():
        logger.error(
            "MODEL_ARTIFACT_MISSING",
            extra={
                "path": str(_model_path),
                "remediation": (
                    "Mount models/churn_model.pkl at /app/models inside "
                    "the container (or rebuild the image with the file "
                    "in COPY)."
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

# FastAPI's default RequestValidationError handler re-serializes each raw
# pydantic error including its ``input`` field. When the client submits
# NaN/Infinity (JSON does not support them, so they arrive as Python
# floats via a lenient JSON parse), jsonable_encoder raises and the
# endpoint returns 500 instead of 422. Strip ``input`` so every
# validation failure is reported as a clean 422.
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """429 with Retry-After.

    slowapi 0.1.9's stock handler omits Retry-After (its
    RateLimitExceeded.headers is None), so clients that back off on
    that header - our own frontend and load tests included - get no
    signal for when to retry. Reproduce slowapi's own header math
    from the window stats of the limit that was actually hit, with a
    full-window fallback if the storage is unreachable.
    """
    retry_after = None
    current_limit = getattr(getattr(request, "state", None), "view_rate_limit", None)
    if current_limit is not None:
        try:
            reset_at, _remaining = limiter.limiter.get_window_stats(*current_limit)
            retry_after = max(1, int(reset_at + 1 - time.time()))
        except Exception:  # storage failure must not turn 429 into 500
            retry_after = None
    if retry_after is None:
        item = getattr(getattr(exc, "limit", None), "limit", None)
        window = item.get_expiry() if item is not None else 60
        retry_after = int(window)
    return JSONResponse(
        status_code=429,
        content={"error": str(exc.detail)},
        headers={"Retry-After": str(retry_after)},
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    def _clean(err: dict) -> dict:
        stripped = {k: v for k, v in err.items() if k != "input"}
        # pydantic v2 keeps the live exception object in ``ctx["error"]``
        # for validator-raised errors; JSON cannot serialize a
        # ValueError instance, so stringify it.
        ctx = stripped.get("ctx")
        if isinstance(ctx, dict):
            stripped["ctx"] = {
                k: (str(v) if isinstance(v, Exception) else v)
                for k, v in ctx.items()
            }
        return stripped

    return JSONResponse(
        status_code=422, content={"detail": [_clean(err) for err in exc.errors()]}
    )


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
    # ``extra="forbid"`` rejects unknown fields; ``allow_inf_nan=False``
    # rejects NaN/Infinity floats, which JSON parsers may accept and
    # which would otherwise poison the feature frame downstream.
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    # Binary service flags (0/1), mirroring the frontend Zod schema.
    Gender: str | None = Field(None)
    SeniorCitizen: int | None = Field(None, ge=0, le=1)
    Partner: int | None = Field(None, ge=0, le=1)
    Dependents: int | None = Field(None, ge=0, le=1)
    tenure: int | None = Field(None, ge=0, le=720)  # months; 720 = 60 years
    PhoneService: int | None = Field(None, ge=0, le=1)
    MultipleLines: int | None = Field(None, ge=0, le=1)
    InternetService: int | None = Field(None, ge=0, le=1)
    OnlineSecurity: int | None = Field(None, ge=0, le=1)
    OnlineBackup: int | None = Field(None, ge=0, le=1)
    DeviceProtection: int | None = Field(None, ge=0, le=1)
    TechSupport: int | None = Field(None, ge=0, le=1)
    StreamingTV: int | None = Field(None, ge=0, le=1)
    StreamingMovies: int | None = Field(None, ge=0, le=1)
    Contract: str | None = Field(None)
    PaperlessBilling: int | None = Field(None, ge=0, le=1)
    PaymentMethod: str | None = Field(None)
    MonthlyCharges: float | None = Field(None, ge=0, le=1_000_000)
    TotalCharges: float | None = Field(None, ge=0, le=1_000_000)
    Married: int | None = Field(None, ge=0, le=1)
    NumberOfDependents: int | None = Field(None, ge=0, le=50)
    NumberOfReferrals: int | None = Field(None, ge=0, le=50)
    # Survey scale is 1-5 (IBM Telco churn dataset).
    SatisfactionScore: int | None = Field(None, ge=1, le=5)
    InternetType: str | None = Field(None)
    Offer: str | None = Field(None)
    Age: int | None = Field(None, ge=0, le=120)
    AvgMonthlyGBDownload: int | None = Field(None, ge=0, le=10_000)
    AvgMonthlyLongDistanceCharges: float | None = Field(None, ge=0, le=1_000_000)
    CLTV: int | None = Field(None, ge=0, le=1_000_000)
    Under30: int | None = Field(None, ge=0, le=1)
    UnlimitedData: int | None = Field(None, ge=0, le=1)
    StreamingMusic: int | None = Field(None, ge=0, le=1)
    ReferredAFriend: int | None = Field(None, ge=0, le=1)
    TotalRefunds: float | None = Field(None, ge=0, le=1_000_000)
    TotalExtraDataCharges: int | None = Field(None, ge=0, le=1_000_000)
    TotalLongDistanceCharges: float | None = Field(None, ge=0, le=1_000_000)
    TotalRevenue: float | None = Field(None, ge=0, le=1_000_000)

    @field_validator("*")
    @classmethod
    def _reject_nan_infinity(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("NaN or Infinity is not allowed for float fields")
        return v

    @model_validator(mode="after")
    def _reject_empty_payload(self):
        """Empty JSON objects predict from an all-zero feature frame and
        return a meaningless "no-signal" score; reject them loudly with
        422 instead. Partial payloads (at least one field set) remain
        valid, preserving the zero-fill contract in ``_align_to_model``.
        """
        if all(value is None for value in self.model_dump().values()):
            raise ValueError(
                "empty payload: provide at least one customer field "
                "(e.g. SatisfactionScore, tenure, MonthlyCharges)"
            )
        return self


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
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
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
    # ``app.state.model`` is always set (possibly None) by the lifespan
    # handler, so the check must be a truthiness test - ``hasattr``
    # would report a healthy model even in the degraded no-artifact
    # boot state. ``version`` is the deployed commit so drift between
    # this Space and the repository's master is externally observable.
    return {
        "status": "healthy" if request.app.state.model is not None else "degraded",
        "model_loaded": request.app.state.model is not None,
        "model_path": str(_model_path),
        "version": GIT_COMMIT_SHA,
    }


def _align_to_model(df: pd.DataFrame, expected: list[str] | None) -> pd.DataFrame:
    """Align an engineered DataFrame to the model's expected column order.

    Behavior:

    * Missing columns are back-filled with 0 and logged as a warning.
      This preserves the prior API contract - a partial payload still
      returns HTTP 200 with every blank field treated as 0 (the
      model's neutral default). Fully empty payloads are rejected at
      the schema layer (``CustomerFeatures._reject_empty_payload``)
      with a 422 before ever reaching this function.
    * Extra columns (not in ``expected``) are dropped and logged.
    * If the model's expected feature list is unavailable (``None``)
      we return the frame as-is.
    * When the produced frame is *fully* empty (zero columns) - the
      engineering pipeline returned nothing - we raise
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
        # Engineer produced nothing - refuse the prediction rather than
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
# columns plus the direct numerics. Listed here for visibility - the
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


# Model routing
# --------------
# The provider's model catalog changes over time - ids are added,
# decommissioned, and superseded. Hard-coding ids here would silently rot
# and pin the deployment to whatever existed at authoring time. The two
# serving tiers are therefore resolved dynamically from the provider's
# live model list:
#
#   * ``standard``      - a medium-intelligence chat model for routine
#                         insight generation.
#   * ``high_capacity`` - the highest-intelligence chat model available,
#                         used for complex strategy generation.
#
# ``LLM_STANDARD_MODEL`` / ``LLM_HIGH_CAPACITY_MODEL`` env vars may pin a
# tier to an explicit id for operators who need deterministic routing;
# when unset (the default) discovery is used. Discovery results are
# cached per API key for ``_LLM_CACHE_TTL_SECONDS``.
_LLM_CACHE_TTL_SECONDS = 3600

# Chat completion only. Speech, transcription, and safety-classifier
# endpoints are never valid targets for this feature.
_NON_CHAT_RE = re.compile(r"whisper|tts|guard|embed|moderation|playai", re.IGNORECASE)
# "8x7b"-style mixture-of-experts sizes and single dense sizes ("70b").
_MOE_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b", re.IGNORECASE)
_DENSE_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)


def _model_intelligence_score(model_id: str) -> float:
    """Heuristic capability score for a chat model id.

    Parameter count (log-scaled, so 70B beats 9B by a wide but not
    absurd margin) dominates; family tier hints nudge the ranking so a
    flagship build of a given size outranks a budget variant. Higher is
    more capable; non-chat models score ``-inf`` and are excluded.
    """
    mid = model_id.lower()
    if _NON_CHAT_RE.search(mid):
        return float("-inf")

    moe = _MOE_SIZE_RE.search(mid)
    if moe:
        params = float(moe.group(1)) * float(moe.group(2))
    else:
        dense = _DENSE_SIZE_RE.search(mid)
        params = float(dense.group(1)) if dense else 0.0

    score = math.log2(params + 1.0) if params > 0 else 0.0
    if "max" in mid:
        score += 1.0
    if re.search(r"mini", mid):
        score -= 1.5
    if "nano" in mid:
        score -= 2.0
    return score


def _select_routed_models(model_ids: List[str]) -> dict:
    """Pick the two serving tiers from a raw model-id list.

    ``high_capacity`` takes the top-ranked chat model. ``standard``
    takes the middle of the ranking so routine insights run on a
    medium-intelligence model instead of burning the flagship on every
    call. Ties break alphabetically for determinism.
    """
    ranked = sorted(
        (m for m in model_ids if _model_intelligence_score(m) != float("-inf")),
        key=lambda m: (-_model_intelligence_score(m), m),
    )
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {"standard": ranked[0], "high_capacity": ranked[0]}
    return {
        "standard": ranked[len(ranked) // 2],
        "high_capacity": ranked[0],
    }


_routing_lock = threading.Lock()
# api_key -> {"models": {tier: id}, "expires": monotonic deadline}
_routing_cache: dict = {}


def _pinned_tier_models() -> dict:
    """Operator-pinned tier ids from the environment, if any."""
    pinned = {}
    standard = os.environ.get("LLM_STANDARD_MODEL", "").strip()
    high = os.environ.get("LLM_HIGH_CAPACITY_MODEL", "").strip()
    if standard:
        pinned["standard"] = standard
    if high:
        pinned["high_capacity"] = high
    return pinned


def _discover_routed_models(api_key: str) -> dict:
    """Query the provider's live model list and route the two tiers."""
    client = _get_llm_client(api_key)
    listing = client.models.list()
    # ``models.list()`` returns a paginated response object whose model
    # entries live under ``.data``. Iterating the response directly would
    # yield pydantic ``(field_name, value)`` tuples, not models.
    items = getattr(listing, "data", None)
    if items is None:
        if (
            isinstance(listing, tuple)
            and listing
            and not getattr(listing[0], "id", None)
        ):
            # Defensive: some SDK builds return ``(data, cursor)`` tuples.
            items = listing[0]
        else:
            items = listing
    model_ids = [m.id for m in (items or []) if getattr(m, "id", None)]
    routed = _select_routed_models(model_ids)
    if not routed:
        raise RuntimeError("provider returned no eligible chat models")
    return routed


def get_routed_models(api_key: str) -> Optional[dict]:
    """Return ``{tier: model_id}`` for the key, from cache when fresh.

    Returns ``None`` when discovery fails and no previous result is
    cached - callers degrade gracefully rather than guessing an id.
    """
    now = time.monotonic()
    with _routing_lock:
        cached = _routing_cache.get(api_key)
        if cached and cached["expires"] > now:
            return cached["models"]
    try:
        routed = _discover_routed_models(api_key)
    except Exception as exc:
        logger.warning("LLM_MODEL_DISCOVERY_FAILED: %s", str(exc))
        with _routing_lock:
            stale = _routing_cache.get(api_key)
        return stale["models"] if stale else None
    with _routing_lock:
        _routing_cache[api_key] = {"models": routed, "expires": now + _LLM_CACHE_TTL_SECONDS}
    return routed


def _pick_model(api_key: str, requested: str) -> Optional[str]:
    """Resolve the concrete model id for a request.

    Precedence: an explicit operator pin beats discovery; a tier alias
    or a provider-validated raw id supplied by the caller beats the
    default. Unrecognised ids log ``LLM_MODEL_UNKNOWN`` and fall back
    to the standard tier. Returns ``None`` when no id can be determined
    (no key / discovery unavailable and nothing requested).
    """
    routed = get_routed_models(api_key) if api_key else None
    tiers = {**(routed or {}), **_pinned_tier_models()}
    if requested and requested not in ("standard", "high_capacity"):
        if routed and requested not in routed.values():
            # Not in the provider's live catalog - do not guess.
            logger.info("LLM_MODEL_UNKNOWN: %s - falling back to default tier", requested)
        else:
            # Valid id (or unverifiable because discovery failed):
            # honour the caller's explicit choice.
            return requested
    if not tiers:
        return None
    if requested in tiers:
        return tiers[requested]
    return tiers.get("standard")


def _resolve_provider_credentials(request: Request) -> tuple[str, Optional[str]]:
    """Pick the API key + model the caller wants.

    Precedence: ``X-Provider-Key`` / ``X-Provider-Model`` request
    headers (set by the in-app Provider Configuration Panel) override
    the module-level env defaults. Model ids are resolved dynamically -
    see the "Model routing" section above.
    """
    api_key = (request.headers.get("X-Provider-Key") or "").strip() or LLM_PROVIDER_API_KEY
    requested = (request.headers.get("X-Provider-Model") or "").strip()
    return api_key, _pick_model(api_key, requested)


def _get_llm_client(api_key: str) -> Groq:
    """Build the LLM provider client. Raises if the key is empty."""
    if not api_key:
        raise RuntimeError("LLM provider key is not configured")
    return Groq(api_key=api_key)


def _generate_script(
    prompt: str,
    api_key: str,
    model: Optional[str] = None,
) -> str:
    """Generate a retention script via the configured LLM provider.

    Always returns a string - never raises. The fallback path is hit
    when the API key is missing, no model could be routed, OR the LLM
    call fails. Log tags distinguish the cause:

    * ``LLM_PROVIDER_KEY_MISSING`` - set ``LLM_PROVIDER_API_KEY`` in
      your platform's environment (HF Spaces, Vercel, any container
      host), in a local ``.env`` file, or via the Provider
      Configuration Panel in the UI. See README for placement.
    * ``LLM_MODEL_UNAVAILABLE`` - the key was present but no model id
      could be resolved (model-list discovery failed and no tier is
      pinned via env vars).
    * ``LLM_GENERATION_FAILED`` - the call was made but failed
      (network, quota, malformed prompt). Inspect the captured
      exception in the log.
    * ``LLM_MODEL_UNKNOWN`` - an unrecognised model id was passed via
      ``X-Provider-Model``; the call fell back to the default tier.
    """
    if not api_key:
        logger.warning(
            "LLM_PROVIDER_KEY_MISSING: set LLM_PROVIDER_API_KEY or use "
            "the Provider Configuration Panel; using fallback."
        )
        return _FALLBACK_SCRIPT

    if not model:
        logger.error(
            "LLM_MODEL_UNAVAILABLE: no model routed for the request; using fallback."
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
    retention analysts - never read aloud to the customer. The model
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
    """Describe the model tiers the in-app Provider Configuration Panel
    can route to. Concrete model ids are resolved server-side from the
    provider's live catalog and are intentionally not exposed here -
    the UI speaks in tiers, never in provider-specific ids."""
    return {
        "models": {
            "standard": "Standard insights - medium-intelligence model, selected automatically",
            "high_capacity": "Deep analysis - highest-intelligence model available, selected automatically",
        },
        "default": "standard",
    }
