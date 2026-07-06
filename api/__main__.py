"""Entry point: ``python -m api`` boots the FastAPI service on the
port declared by the ``PORT`` environment variable (default 8000)."""
from __future__ import annotations

import uvicorn

from api.app import PORT, app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
