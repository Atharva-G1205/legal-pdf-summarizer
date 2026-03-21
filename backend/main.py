"""
Legal PDF Summarizer — FastAPI Backend
=======================================

Entry point for the API server.

Run:
    cd <project-root>
    PYTHONPATH=. uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api.summarize import router as summarize_router


# ─── Lifespan (model preloading can go here later) ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown hooks."""
    print("🚀 Legal PDF Summarizer API is starting …")
    yield
    print("👋 Shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal PDF Summarizer API",
    description="Upload legal PDFs and get multi-level summaries.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and common local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / next default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(summarize_router)


@app.get("/health")
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok"}
