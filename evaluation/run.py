#!/usr/bin/env python3
"""
CLI runner for evaluation metrics
====================================

Accepts both PDF and plain-text files for source and summary.

Usage
-----
    # Evaluate from two PDFs:
    python -m evaluation.run --source judgment.pdf --summary summary.pdf

    # Mix PDF + text:
    python -m evaluation.run --source judgment.pdf --summary summary.txt

    # Select specific metrics:
    python -m evaluation.run --source s.pdf --summary g.pdf --metrics rouge bertscore

    # Save JSON output:
    python -m evaluation.run --source s.pdf --summary g.pdf --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.evaluate import evaluate_summary, format_report


def _read_input(path: str) -> str:
    """Read text from a file — supports both .pdf and plain text files.

    For PDFs, uses the project's ``pipeline.pdf_loader.PDFLoader``
    to extract text from all pages and concatenates them.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() == ".pdf":
        from pipeline.pdf_loader import PDFLoader

        loader = PDFLoader(ocr_enabled=True)
        result = loader.load_pdf(p)
        pages_text = [page["text"] for page in result.get("pages", []) if page.get("text")]
        if not pages_text:
            raise ValueError(f"Could not extract any text from PDF: {p}")
        return "\n".join(pages_text)

    # Treat everything else as plain text
    return p.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a generated summary against source text. "
                    "Accepts both PDF and plain-text files.",
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the source document file (.pdf or .txt).",
    )
    parser.add_argument(
        "--reference", required=False, default=None,
        help="Path to reference summary file (.pdf or .txt) for ROUGE/BERTScore.",
    )
    parser.add_argument(
        "--summary", required=True,
        help="Path to the generated summary file (.pdf or .txt).",
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["rouge", "bertscore", "entailment"],
        choices=["rouge", "bertscore", "entailment"],
        help="Which metrics to compute (default: all).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for model inference (cuda / cpu / auto).",
    )
    parser.add_argument(
        "--json", dest="json_out", default=None,
        help="Optional path to write JSON results.",
    )

    args = parser.parse_args()

    print("📄  Reading source file ...")
    source_text = _read_input(args.source)
    print(f"    → {len(source_text):,} characters extracted\n")

    print("📝  Reading summary file ...")
    summary_text = _read_input(args.summary)
    print(f"    → {len(summary_text):,} characters extracted\n")

    reference_text = None
    if args.reference:
        print("📚  Reading reference summary file ...")
        reference_text = _read_input(args.reference)
        print(f"    → {len(reference_text):,} characters extracted\n")

    results = evaluate_summary(
        source=source_text,
        summary=summary_text,
        reference=reference_text,  # AUDIT FIX: lexical metrics should use reference summary.
        metrics=args.metrics,
        device=args.device,
    )

    # Print human-readable report
    print(format_report(results))

    # Optionally dump JSON
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n💾  JSON results saved to {out_path}")


if __name__ == "__main__":
    main()
