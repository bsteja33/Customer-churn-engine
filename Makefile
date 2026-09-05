# Makefile for Customer Churn Prediction project

PYTHON=python
PIP=pip
REQ=requirements.txt
PROJECT_ROOT=$(shell pwd)

.PHONY: train serve test docker-build dev dev-port

# Train the model using the default config
train:
	$(PYTHON) src/train.py

# Run the FastAPI service (port honors $PORT, default 8000).
serve:
	uvicorn api.app:app --host 0.0.0.0 --port $${PORT:-8000}

# Boot API and Next.js dev server together. The Next.js rewrite
# target reads $BACKEND_PORT, not $PORT, so the FE stays on its
# own 3000 port while the API stays on 8000 by default.
dev:
	@$(MAKE) -j2 serve dev-fe

dev-fe:
	cd frontend && BACKEND_PORT=$${BACKEND_PORT:-8000} npm run dev

# Boot both services on a non-default API port
# (e.g. `make dev-port BACKEND_PORT=8765`).
dev-port:
	@$(MAKE) dev BACKEND_PORT=$(BACKEND_PORT)

# Run the test suite
test:
	$(PYTHON) -m pytest -vv

# Lint backend sources (frontend: `cd frontend && npm run lint`)
lint:
	flake8 api/ src/

# Build the Docker image
docker-build:
	docker build -t churn-api:latest .

# Run Playwright E2E tests (auto-boots dev server)
playwright:
	cd frontend && npx playwright test --headless
