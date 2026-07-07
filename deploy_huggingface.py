"""One-shot Hugging Face Space deploy for the churn engine.

Reads HF_TOKEN, LLM_PROVIDER_API_KEY, and CORS_ORIGINS strictly from
os.environ. The .env.production file at the repo root is a tracked
TEMPLATE with empty placeholders; this script does not load it. The
real values live in the developer's shell, set via `! export`, and
are never printed to stdout.

Run from your terminal:

    ! export HF_TOKEN=hf_<new>
    ! export LLM_PROVIDER_API_KEY=<your-groq-key>
    ! export CORS_ORIGINS=https://frontend-pi-sage-79.vercel.app
    ! python deploy_huggingface.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parent
SPACE_NAME = "churn-engine"
SPACE_SDK = "docker"
APP_PORT = 8000

# Relative paths inside REPO_ROOT that the Space needs at runtime.
# `api/**` and `src/**` are recursive globs (pathlib PurePath match
# semantics): `**` crosses path separators, so every file under the
# directory is included. The single-file entries need no wildcard.
UPLOAD_PATHS = [
    "api/**",
    "src/**",
    "models/churn_model.pkl",
    "requirements.txt",
    "Dockerfile",
]

README_CONTENT = f"""---
title: Churn Engine
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: {APP_PORT}
pinned: false
license: mit
---

# Churn Engine API

FastAPI service for the customer churn prediction system. Public endpoints:
`/predict`, `/predict/batch`, `/generate_retention_script`, `/llm/models`,
`/health`, `/docs`, `/openapi.json`.

See the main repository at
https://github.com/bsteja33/Customer-churn-engine for the full architecture
and the Next.js dashboard that proxies to this service.
"""


def fail(msg: str, page_url: str, code: int = 1) -> int:
    print(f"  ! {msg}", file=sys.stderr)
    print(f"  inspect: {page_url}", file=sys.stderr)
    return code


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    llm_key = os.environ.get("LLM_PROVIDER_API_KEY", "").strip()
    cors_origins = (
        os.environ.get("CORS_ORIGINS", "").strip()
        or "https://frontend-pi-sage-79.vercel.app"
    )

    if not token:
        return fail(
            "HF_TOKEN missing. Run `! export HF_TOKEN=hf_<new>` first.",
            page_url="https://huggingface.co/spaces/<unknown>/" + SPACE_NAME,
        )

    if not llm_key:
        print(
            "  ! LLM_PROVIDER_API_KEY missing. /generate_retention_script "
            "will return the labelled static default plan. Continuing."
        )

    api = HfApi(token=token)
    me = whoami(token=token)
    namespace = (me.get("name") or "").strip()
    if not namespace:
        return fail(
            "whoami() returned no `name` field. Cannot resolve HF namespace.",
            page_url="https://huggingface.co/spaces",
        )
    space_repo_id = f"{namespace}/{SPACE_NAME}"
    space_subdomain = f"{namespace.lower()}-{SPACE_NAME}"
    health_url = f"https://{space_subdomain}.hf.space/health"
    space_page = f"https://huggingface.co/spaces/{space_repo_id}"
    print(f"Authenticated as: {namespace}")
    print(f"Target Space:     {space_repo_id}")

    print(f"Creating Space: {space_repo_id} (sdk={SPACE_SDK})")
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk=SPACE_SDK,
        exist_ok=True,
        private=False,
    )

    print("Uploading README.md (app_port=8000)")
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="docs: set app_port=8000 for FastAPI",
    )

    print("Resolving runtime file list:")
    resolved: list[Path] = []
    for rel in UPLOAD_PATHS:
        if rel.endswith("/**"):
            base = REPO_ROOT / rel[:-3]
            if not base.is_dir():
                return fail(
                    f"missing local directory: {base.name}",
                    page_url=space_page,
                )
            files = sorted(base.rglob("*"))
            files = [p for p in files if p.is_file()]
            print(f"  - {base.name}/ ({len(files)} files)")
            resolved.extend(files)
        else:
            src = REPO_ROOT / rel
            if not src.is_file():
                return fail(
                    f"missing local file: {rel}", page_url=space_page,
                )
            print(f"  - {rel}")
            resolved.append(src)
    print(f"Total files to upload: {len(resolved)}")
    print(f"Uploading {len(resolved)} files to {space_repo_id}...")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="feat: deploy FastAPI churn engine",
        allow_patterns=UPLOAD_PATHS,
        ignore_patterns=[
            ".git/*",
            ".github/*",
            "frontend/*",
            "tests/*",
            "load_tests/*",
            "scripts/*",
            "*.md",
            ".dockerignore",
            ".gitignore",
            ".env*",
            "deploy_huggingface.py",
            "LICENSE",
            "Makefile",
            "docker-compose.yml",
            "vercel.json",
        ],
    )

    print("Setting variable: CORS_ORIGINS")
    api.add_space_variable(
        repo_id=space_repo_id, key="CORS_ORIGINS", value=cors_origins,
    )
    if llm_key:
        print("Setting secret:   LLM_PROVIDER_API_KEY (value redacted)")
        api.add_space_secret(
            repo_id=space_repo_id, key="LLM_PROVIDER_API_KEY", value=llm_key,
        )
    print("Setting variable: LLM_STANDARD_MODEL")
    api.add_space_variable(
        repo_id=space_repo_id, key="LLM_STANDARD_MODEL",
        value="llama-3.1-8b-instant",
    )
    print("Setting variable: LLM_HIGH_CAPACITY_MODEL")
    api.add_space_variable(
        repo_id=space_repo_id, key="LLM_HIGH_CAPACITY_MODEL",
        value="llama-3.3-70b-versatile",
    )

    print("Waiting for Space build (max 10 min)...")
    deadline = time.time() + 600
    last = None
    while time.time() < deadline:
        info = api.get_space_runtime(repo_id=space_repo_id)
        stage = getattr(info, "stage", "UNKNOWN")
        if stage != last:
            print(f"  stage: {stage}")
            last = stage
        if stage == "RUNNING":
            break
        if stage in {"RUNTIME_ERROR", "BUILD_ERROR", "PAUSED"}:
            return fail(
                f"build failed at stage={stage}",
                page_url=space_page, code=2,
            )
        time.sleep(10)
    else:
        return fail("build timed out after 10 min", page_url=space_page, code=3)

    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(health_url, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8"))
            print(f"  /health: {body}")
            if body.get("model_loaded"):
                break
            return fail(
                "model_loaded is false; check Space logs",
                page_url=space_page, code=4,
            )
        except Exception as exc:
            if attempt == 1:
                print(
                    f"  health probe attempt 1 failed ({exc!r}); "
                    "retrying in 15s"
                )
                time.sleep(15)
            else:
                return fail(
                    f"health probe failed: {exc!r}",
                    page_url=space_page, code=5,
                )

    print()
    print(f"Backend URL: https://{space_subdomain}.hf.space")
    print(f"Space page:  {space_page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
