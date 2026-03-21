"""
Summarize API Router
====================

Endpoints for the legal PDF summarization service.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse

# Ensure project root is on path so ``pipeline`` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.config import SUMMARY_CONFIGS, SummaryLevel, get_config_by_number
from pipeline.summarize import run_pipeline

from backend.api.schemas import SummaryLevelInfo, SummaryResponse
from backend.services.pdf_generator import SummaryPDFGenerator

router = APIRouter(prefix="/api", tags=["summarize"])

# Singleton PDF generator
_pdf_gen = SummaryPDFGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/summary-levels
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/summary-levels",
    response_model=List[SummaryLevelInfo],
    summary="List available summary levels",
)
async def list_summary_levels() -> List[SummaryLevelInfo]:
    """Return every configured summary level the user can choose from."""
    return [
        SummaryLevelInfo(
            value=level.value,
            name=cfg.name,
            emoji=cfg.emoji,
            description=cfg.description,
        )
        for level, cfg in SUMMARY_CONFIGS.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/summarize
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Upload a PDF and generate a summary",
)
async def summarize_pdf(
    file: UploadFile = File(..., description="The PDF file to summarize"),
    level: int = Form(1, description="Summary level (1-4)"),
    model_source: str = Form("finetuned", description="Model source: 'finetuned' or 'huggingface'"),
) -> SummaryResponse:
    """
    Accept a PDF upload and a summary-level choice, run the full
    extractive → abstractive pipeline, and return the summary.
    """

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate model_source
    if model_source not in ("finetuned", "huggingface"):
        raise HTTPException(status_code=400, detail="model_source must be 'finetuned' or 'huggingface'.")

    # Resolve config
    try:
        config = get_config_by_number(level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Save to temp file, run pipeline, clean up
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        summary = run_pipeline(tmp_path, config, model_source=model_source)

        if not summary:
            raise HTTPException(
                status_code=422,
                detail="Pipeline returned an empty summary. The PDF may not contain extractable text.",
            )

        return SummaryResponse(
            summary=summary,
            word_count=len(summary.split()),
            level_name=config.name,
            level_emoji=config.emoji,
            filename=file.filename or "unknown.pdf",
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/download-pdf
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/download-pdf",
    summary="Generate a downloadable PDF from summary text",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def download_summary_pdf(
    summary: str = Body(..., embed=True),
    level_name: str = Body("Summary", embed=True),
    filename: str = Body("document.pdf", embed=True),
) -> StreamingResponse:
    """Convert summary text into a formatted PDF and stream it back."""

    buf = _pdf_gen.generate(
        summary=summary,
        level_name=level_name,
        filename=filename,
    )

    safe_name = Path(filename).stem
    download_name = f"{safe_name}_{level_name.lower()}_summary.pdf"

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )
