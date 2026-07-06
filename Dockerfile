FROM python:3.13-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runner

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
