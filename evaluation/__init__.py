"""
Evaluation Metrics for Legal Document Summarization
=====================================================

Provides ROUGE, BERTScore, and Entailment / Faithfulness scoring
to compare generated summaries against source documents.
"""

from evaluation.rouge_score import compute_rouge
from evaluation.bert_score import compute_bertscore
from evaluation.entailment_score import compute_entailment
from evaluation.evaluate import evaluate_summary

__all__ = [
    "compute_rouge",
    "compute_bertscore",
    "compute_entailment",
    "evaluate_summary",
]
