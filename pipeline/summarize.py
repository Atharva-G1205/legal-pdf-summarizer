"""
Multi-Level Legal PDF Summarizer — Orchestrator
=================================================

End-to-end pipeline that:
  1. Loads a PDF and preprocesses it
  2. Flattens sentences from classified sections
  3. Ranks sentences with InLegalBERT (extractive)
  4. Generates an abstractive summary with Legal-LED / Legal-Pegasus

Usage:
    python pipeline/summarize.py                 # Interactive mode
    python pipeline/summarize.py path/to/case.pdf   # Direct mode
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List

# Allow running from project root: ``python pipeline/summarize.py``
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.config import (
    SummaryConfig,
    get_config_by_number,
    MENU_OPTIONS,
)
from pipeline.pdf_loader import PDFLoader
from pipeline.preprocessor import TextPreprocessor
from pipeline.retriever import rank_sentences
from pipeline.summarizer import LegalSummarizer, SummaryResult

logger = logging.getLogger(__name__)


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _step(icon: str, msg: str, detail: str = "") -> None:
    print(f"\n{icon}  {msg}")
    if detail:
        print(f"   {detail}")


def _ok(msg: str) -> None:
    print(f"   ✓ {msg}")


# ============================================================================
# USER INTERACTION
# ============================================================================

def select_summary_type() -> SummaryConfig:
    """Prompt the user to choose a summary level."""
    _header("Legal PDF Summarizer — Multi-Level Summary")
    print(MENU_OPTIONS)
    while True:
        try:
            choice = input("Enter choice (1/2/3/4): ").strip()
            return get_config_by_number(int(choice))
        except (ValueError, KeyError):
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def get_pdf_path() -> Path:
    """Prompt for and validate a PDF path."""
    path_str = input("Enter PDF path: ").strip().strip("\"'")
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")
    return path


# ============================================================================
# PIPELINE STEPS
# ============================================================================

def flatten_sentences(processed_doc: dict) -> List[Dict[str, Any]]:
    """
    Flatten per-section sentence lists, tagging each with its section name.
    """
    out: list[dict] = []
    for sec in ["headnote", "facts", "issues", "arguments", "reasoning", "order"]:
        section = processed_doc.get("sections", {}).get(sec)
        if section and "sentences" in section:
            for sent in section["sentences"]:
                out.append({**sent, "section": sec})
    return out


def run_pipeline(pdf_path: Path, config: SummaryConfig, model_source: str | None = None) -> str:
    """
    Execute the full summarization pipeline and return the summary string.

    Args:
        pdf_path:     Path to the PDF file.
        config:       SummaryConfig for the chosen level.
        model_source: Optional override — ``'finetuned'`` or ``'huggingface'``.
                      If None, uses value from config.
    """

    # 1. Load PDF -------------------------------------------------------
    _step("📂", "Loading PDF…")
    loader = PDFLoader()
    raw_doc = loader.load_pdf(pdf_path)

    # 2. Preprocess ------------------------------------------------------
    _step("📝", "Preprocessing document…")
    preprocessor = TextPreprocessor()
    processed = preprocessor.clean_document(raw_doc)

    total_sents = processed.get("summarization_input", {}).get("total_sentences", 0)
    sections = list(processed.get("sections", {}).keys())
    _ok(f"Sections: {sections}")
    _ok(f"Total sentences: {total_sents}")

    # 3. Flatten sentences -----------------------------------------------
    all_sentences = flatten_sentences(processed)
    _step("✂️", f"Flattened {len(all_sentences)} sentences from sections")

    if not all_sentences:
        print("⚠  No sentences found — cannot generate summary.")
        return ""

    # 4. Rank with InLegalBERT -------------------------------------------
    _step("🧠", f"Ranking top {config.top_n} sentences (InLegalBERT)…")
    # AUDIT FIX: keep >=40% sentences to improve coverage and recall in legal docs.
    top_sentences = rank_sentences(
        all_sentences,
        top_n=config.top_n,
        min_selection_ratio=0.4,
    )
    extraction_ratio = (len(top_sentences) / len(all_sentences)) if all_sentences else 0.0
    total_words = sum(len(s["text"].split()) for s in top_sentences)
    _ok(f"Selected {len(top_sentences)} sentences ({total_words:,} words)")
    _ok(f"Extraction ratio: {extraction_ratio:.2%}")

    # 5. Generate summary ------------------------------------------------
    if config.model == "none":
        # --- Extractive mode: return ranked sentences as-is ---
        _step("📌", "Extractive mode — returning top-ranked sentences")
        lines = [
            f"{i}. [{s.get('section', '?').upper()}] {s['text']}"
            for i, s in enumerate(top_sentences, 1)
        ]
        return "\n".join(lines)

    # Resolve model source
    effective_source = model_source or config.model_source

    # --- Abstractive mode ---
    _step("🤖", f"Loading {config.name} summarizer (model: {config.model}, source: {effective_source})…")
    summarizer = LegalSummarizer(model_key=config.model, model_source=effective_source)

    _step("📝", f"Generating {config.name} summary…")
    grounding = {
        "metadata_text": processed.get("metadata", {}).get("text", ""),
        "repeated_header": processed.get("metadata", {}).get("repeated_header", ""),
        "unique_citations": processed.get("citations", {}).get("unique_citations", []),
    }
    result: SummaryResult = summarizer.summarize_sentences(
        top_sentences,
        max_length=config.max_length,
        min_length=config.min_length,
        grounding=grounding,
    )
    # AUDIT FIX: log handoff size to catch under-coverage or silent truncation risk.
    logger.info(
        "AUDIT: stage1_to_stage2 words=%d extraction_ratio=%.4f model=%s",
        total_words,
        extraction_ratio,
        config.model,
    )

    return result.summary


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Interactive entry point."""
    try:
        # If a path was given as CLI arg, use it; otherwise prompt
        if len(sys.argv) > 1:
            pdf_path = Path(sys.argv[1]).expanduser().resolve()
            if not pdf_path.exists():
                print(f"❌ File not found: {pdf_path}")
                sys.exit(1)
        else:
            pdf_path = get_pdf_path()

        config = select_summary_type()

        summary = run_pipeline(pdf_path, config)

        # Display result
        _header(f"{config.emoji} {config.name} Summary")
        print()
        print(summary)
        print()
        print(f"({len(summary.split())} words)")

    except KeyboardInterrupt:
        print("\n\nCancelled.")
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        raise


if __name__ == "__main__":
    main()
