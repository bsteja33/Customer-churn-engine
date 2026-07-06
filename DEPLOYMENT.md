# Deployment guide

Live demo checklist for recruiters and reviewers. The two services are
decoupled — the FastAPI backend can be hosted on Render, Railway, Fly,
or any Docker host; the Next.js frontend can be hosted on Vercel,
Netlify, Cloudflare Pages, or the same Docker compose stack.

## Architecture

```
                    ┌──────────────────────────────┐
   Browser  ─────── │  Vercel / Netlify / CF Pages  │
                    │  Next.js 16 (standalone)     │
                    │  Domain: churn.example.com   │
                    └──────────────┬───────────────┘
                                   │ /api/*  → rewrite
                                   ▼
                    ┌──────────────────────────────┐
                    │  Render / Railway / Fly.io    │
                    │  FastAPI + uvicorn           │
                    │  Domain: api.example.com     │
                    │  Model on persistent disk    │
                    └──────────────┬───────────────┘
                                   │ HTTPS
                                   ▼
                    ┌──────────────────────────────┐
                    │  Groq LLM API                │
                    │  llama-3.1-8b-instant        │
                    └──────────────────────────────┘
```

## Backend (Render)

1. **Push to GitHub.**
2. **Connect at https://dashboard.render.com/blueprints.**
3. Render reads `render.yaml` and provisions the `churn-api` service
   from the repo root's `Dockerfile`.
4. In the Render dashboard, set:
   - `LLM_PROVIDER_API_KEY` — your provider key.
   - `CORS_ORIGINS` — your FE origin, e.g.
     `https://churn-engine.vercel.app`.
5. **Upload the model artifact** to the `churn-model` disk
   (`/app/models/churn_model.pkl`). Render's dashboard accepts a
   one-time upload; the disk survives redeploys.
6. Confirm health: `curl https://churn-api.onrender.com/health` →
   `{"status":"healthy","model_loaded":true,...}`.

### Backend (Railway)

1. **Push to GitHub.**
2. **Import at https://railway.app/new.** Railway reads `railway.toml`
   and uses the repo's `Dockerfile`.
3. In the Railway dashboard, set the same `LLM_PROVIDER_API_KEY` and
   `CORS_ORIGINS` env vars.
4. **Upload the model artifact** via the Railway shell:
   `scp models/churn_model.pkl railway:/app/models/`.
5. Confirm health at the public URL.

## Frontend (Vercel)

1. **Import the `frontend/` directory at https://vercel.com/new.**
2. Framework: Next.js (auto-detected).
3. **Set the environment variable**:
   - `BACKEND_INTERNAL_URL` — your API origin, e.g.
     `https://churn-api.onrender.com`.
4. The `vercel.json` at the repo root wires the `/api/*` rewrite to
   the backend. Vercel reads it automatically.
5. Confirm health: visit the deployment URL, the rail chip should
   pulse green within 15 seconds (the health-polling interval).

## Environment variables

### Backend (Render / Railway)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `LLM_PROVIDER_API_KEY` | For script generation | — | Provider key. Missing → labelled fallback. |
| `CORS_ORIGINS` | For non-localhost FE | — | Comma-separated, e.g. `https://churn-engine.vercel.app,https://staging.example.com` |
| `LLM_STANDARD_MODEL` | No | `llama-3.1-8b-instant` | Override the "standard" slot model id. |
| `LLM_HIGH_CAPACITY_MODEL` | No | `llama-3.3-70b-versatile` | Override the "high_capacity" slot model id. |
| `PORT` | No | `8000` | Auto-injected by Render/Railway. |
| `LIMITER_ENABLED` | No | `true` | Set `false` in CI / load tests. |
| `HF_TOKEN` | Only if training data is gated | — | Not needed for inference. |

### Frontend (Vercel)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `BACKEND_INTERNAL_URL` | For non-localhost API | `http://127.0.0.1:8000` | Set to the Render/Railway API origin. |
| `BACKEND_PORT` | No | `8000` | Ignored when `BACKEND_INTERNAL_URL` is set. |

## Local smoke test (after deploy)

```bash
# Health
curl https://churn-api.onrender.com/health

# Single prediction
curl -X POST https://churn-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Gender":"Male","SeniorCitizen":0,"Partner":0,"tenure":2,"PhoneService":1,"InternetService":1,"Contract":"Month-to-Month","PaperlessBilling":1,"PaymentMethod":"Bank Withdrawal","MonthlyCharges":95.0,"TotalCharges":190.0}'

# LLM catalog
curl https://churn-api.onrender.com/llm/models

# Retention script (with provider key)
curl -X POST https://churn-api.onrender.com/generate_retention_script \
  -H "Content-Type: application/json" \
  -H "X-Provider-Key: $LLM_PROVIDER_API_KEY" \
  -d '{"risk_level":"High","reasons":"SatisfactionScore=1, tenure=2mo.","top_drivers":["SatisfactionScore (0.42)"],"probability_pct":78.5}'
```

## Why not a single-host deploy?

The frontend is static-ish (no server-side data fetching), so a CDN
edge like Vercel/Netlify gives recruiters instant load times in any
region. The backend is stateful (model artifact on disk, ThreadPool
executor, lifespan-loaded LLM SDK), so a long-running container host
is the right shape. The two services communicate over HTTPS via the
`/api/*` rewrite, so CORS is the only configuration knob.
