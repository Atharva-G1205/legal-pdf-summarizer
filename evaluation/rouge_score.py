"""
ROUGE Score Evaluation
=======================

Computes ROUGE-1 and ROUGE-L between a generated summary and
reference / source text using the ``rouge-score`` library.
"""

from __future__ import annotations

from typing import Dict


def _lazy_import():
    """Lazy-import heavy dependencies only when needed."""
    global rouge_scorer
    from rouge_score import rouge_scorer  # noqa: F811


def compute_rouge(
    source: str,
    summary: str,
    *,
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """Return ROUGE-1 and ROUGE-L F-measure scores.

    Args:
        source:      Reference / source document text.
        summary:     Generated summary text.
        use_stemmer: Whether to apply Porter stemming before comparison.

    Returns:
        Dictionary with keys ``rouge1_f``, ``rouge1_p``, ``rouge1_r``,
        ``rougeL_f``, ``rougeL_p``, ``rougeL_r``.
    """
    _lazy_import()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=use_stemmer,
    )
    scores = scorer.score(source, summary)

    return {
        "rouge1_precision": round(scores["rouge1"].precision, 4),
        "rouge1_recall": round(scores["rouge1"].recall, 4),
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rougeL_precision": round(scores["rougeL"].precision, 4),
        "rougeL_recall": round(scores["rougeL"].recall, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }
