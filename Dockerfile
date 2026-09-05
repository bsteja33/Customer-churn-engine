FROM python:3.11-slim AS builder

# System deps for the data-science stack. libgomp1 is required at
# runtime by LightGBM's OpenMP runtime; the slim image does not
# include it transitively on every release, so install it here so
# the compiled artifacts in /install can resolve libgomp.so.1.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
# --only-binary=:all: forces pip to fail rather than silently
# build from source. The python:3.11-slim image has no gcc, so
# any source build would fail with "Unknown compiler(s)"; we'd
# rather see the wheel-resolution error than the build error.
# `pip install --upgrade pip` first so the resolver picks a
# version that resolves the pinned versions on the active ABI.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runner

# Same libgomp1 install in the runtime image: the runner stage
# starts from a fresh python:3.11-slim and does not inherit
# packages from the builder's layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip in the runtime image too. No `pip install` runs
# here today, but a future operator adding one will benefit
# from a current pip without needing to remember.
RUN pip install --upgrade pip

ENV PYTHONUNBUFFERED=1

RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid appuser --no-create-home appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

USER appuser

CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
