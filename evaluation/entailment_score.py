"""
Entailment / Faithfulness Score
================================

Measures how *faithful* a generated summary is to the source document
by checking Natural Language Inference (NLI) entailment.

Strategy:
  1. Split the summary into individual sentences.
  2. For each sentence, classify ``(source, sentence)`` with an NLI model.
  3. Report the fraction of summary sentences entailed by the source
     (label == ``entailment``) as the overall faithfulness score.

Uses ``cross-encoder/nli-deberta-v3-base`` by default (fast + accurate).
"""

from __future__ import annotations

import re
from typing import Dict, List


# ── helpers ─────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter (handles common legal abbreviations)."""
    # Split on period/question/exclamation followed by whitespace + uppercase
    parts = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def _truncate(text: str, max_tokens: int = 400) -> str:
    """Keep only the first *max_tokens* white-space tokens."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


# ── main function ──────────────────────────────────────────────────────
def compute_entailment(
    source: str,
    summary: str,
    *,
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    device: str | None = None,
    max_source_tokens: int = 400,
) -> Dict[str, float]:
    """Return faithfulness / entailment metrics.

    Args:
        source:            Source document text.
        summary:           Generated summary text.
        model_name:        NLI cross-encoder from HuggingFace.
        device:            ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
        max_source_tokens: Truncate source to this many tokens per pair
                           (NLI models have limited input length).

    Returns:
        Dictionary with ``entailment_ratio``, ``contradiction_ratio``,
        ``neutral_ratio``, and ``num_sentences_evaluated``.
    """
    # Lazy imports
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    # Label mapping for DeBERTa NLI: {0: contradiction, 1: neutral, 2: entailment}
    LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}

    sentences = _split_sentences(summary)
    if not sentences:
        return {
            "entailment_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "neutral_ratio": 0.0,
            "num_sentences_evaluated": 0,
        }

    truncated_source = _truncate(source, max_source_tokens)
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}

    for sent in sentences:
        inputs = tokenizer(
            truncated_source,
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = int(logits.argmax(dim=-1).item())

        counts[LABEL_MAP[pred]] += 1

    total = len(sentences)
    return {
        "entailment_ratio": round(counts["entailment"] / total, 4),
        "contradiction_ratio": round(counts["contradiction"] / total, 4),
        "neutral_ratio": round(counts["neutral"] / total, 4),
        "num_sentences_evaluated": total,
    }
