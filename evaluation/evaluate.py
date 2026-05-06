"""
Unified Summary Evaluator
===========================

Runs all evaluation metrics (ROUGE, BERTScore, Entailment) on a
generated summary against the source text and returns a combined report.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from evaluation.rouge_score import compute_rouge
from evaluation.bert_score import compute_bertscore
from evaluation.entailment_score import compute_entailment

logger = logging.getLogger(__name__)


def evaluate_summary(
    source: str,
    summary: str,
    reference: str | None = None,
    *,
    metrics: list[str] | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    """Run one or more evaluation metrics and return combined results.

    Args:
        source:   Source document text (used for faithfulness checks).
        summary:  Generated summary text.
        reference: Optional reference summary text. If absent, falls back
                  to source text for lexical metrics.
        metrics:  List of metric names to compute. Defaults to all three:
                  ``["rouge", "bertscore", "entailment"]``.
        device:   ``"cuda"`` / ``"cpu"`` / ``None`` (auto).

    Returns:
        Dictionary mapping each metric group to its sub-scores, e.g.::

          {
            "rouge": {"rouge1_f1": 0.42, ...},
            "bertscore": {"bertscore_f1": 0.87, ...},
            "entailment": {"entailment_ratio": 0.75, ...},
          }
    """
    if metrics is None:
        metrics = ["rouge", "bertscore", "entailment"]

    results: Dict[str, Any] = {}

    lexical_reference = reference if reference is not None else source
    if reference is None:
        logger.warning(
            "AUDIT FLAG: No reference summary provided; ROUGE/BERTScore are computed "
            "against source text, which can suppress recall."
        )

    if "rouge" in metrics:
        logger.info("Computing ROUGE scores ...")
        # AUDIT FIX: compute lexical metrics against reference summary.
        results["rouge"] = compute_rouge(lexical_reference, summary)

    if "bertscore" in metrics:
        logger.info("Computing BERTScore ...")
        # AUDIT FIX: compute lexical/semantic overlap against reference summary.
        results["bertscore"] = compute_bertscore(lexical_reference, summary, device=device)

    if "entailment" in metrics:
        logger.info("Computing Entailment / Faithfulness ...")
        results["entailment"] = compute_entailment(source, summary, device=device)

    return results


def format_report(results: Dict[str, Any]) -> str:
    """Return a human-readable text report from evaluation results."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  SUMMARY EVALUATION REPORT")
    lines.append("=" * 60)

    if "rouge" in results:
        r = results["rouge"]
        lines.append("")
        lines.append("📊  ROUGE Scores")
        lines.append(f"    ROUGE-1  Precision: {r['rouge1_precision']:.4f}")
        lines.append(f"    ROUGE-1  Recall:    {r['rouge1_recall']:.4f}")
        lines.append(f"    ROUGE-1  F1:        {r['rouge1_f1']:.4f}")
        lines.append(f"    ROUGE-L  Precision: {r['rougeL_precision']:.4f}")
        lines.append(f"    ROUGE-L  Recall:    {r['rougeL_recall']:.4f}")
        lines.append(f"    ROUGE-L  F1:        {r['rougeL_f1']:.4f}")

    if "bertscore" in results:
        b = results["bertscore"]
        lines.append("")
        lines.append("🧠  BERTScore")
        lines.append(f"    Precision: {b['bertscore_precision']:.4f}")
        lines.append(f"    Recall:    {b['bertscore_recall']:.4f}")
        lines.append(f"    F1:        {b['bertscore_f1']:.4f}")

    if "entailment" in results:
        e = results["entailment"]
        lines.append("")
        lines.append("✅  Entailment / Faithfulness")
        lines.append(f"    Entailment:    {e['entailment_ratio']:.4f}")
        lines.append(f"    Neutral:       {e['neutral_ratio']:.4f}")
        lines.append(f"    Contradiction: {e['contradiction_ratio']:.4f}")
        lines.append(f"    Sentences evaluated: {e['num_sentences_evaluated']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
