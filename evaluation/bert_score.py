"""
BERTScore Evaluation
=====================

Computes BERTScore (Precision, Recall, F1) between a generated summary
and the source document using the ``bert-score`` library.

Uses ``roberta-large`` by default (good balance of quality and memory).
Automatically chunks long texts to avoid GPU OOM errors.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Maximum tokens per chunk — keeps GPU memory manageable
_MAX_CHUNK_WORDS = 300


def _chunk_text(text: str, max_words: int = _MAX_CHUNK_WORDS) -> List[str]:
    """Split text into word-level chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def compute_bertscore(
    source: str,
    summary: str,
    *,
    model_type: str = "roberta-large",
    device: str | None = None,
) -> Dict[str, float]:
    """Return BERTScore Precision, Recall, and F1.

    Long texts are automatically chunked. Each summary chunk is scored
    against the corresponding source chunk (or the last source chunk if
    the summary has fewer chunks). Final scores are averaged.

    Args:
        source:     Reference / source document text.
        summary:    Generated summary text.
        model_type: HuggingFace model used for embedding comparison.
                    Default ``roberta-large`` works well on ≤6 GB GPUs.
        device:     ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).

    Returns:
        Dictionary with keys ``bertscore_precision``, ``bertscore_recall``,
        ``bertscore_f1``.
    """
    from bert_score import score as bert_score_fn

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    source_chunks = _chunk_text(source)
    summary_chunks = _chunk_text(summary)

    # Pair each summary chunk with a source chunk
    # (cycle source chunks if summary is longer, or vice-versa)
    refs: List[str] = []
    cands: List[str] = []
    for i, s_chunk in enumerate(summary_chunks):
        ref_idx = min(i, len(source_chunks) - 1)
        refs.append(source_chunks[ref_idx])
        cands.append(s_chunk)

    try:
        P, R, F1 = bert_score_fn(
            cands=cands,
            refs=refs,
            model_type=model_type,
            device=device,
            verbose=False,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA OOM during BERTScore — falling back to CPU")
            import torch
            torch.cuda.empty_cache()
            P, R, F1 = bert_score_fn(
                cands=cands,
                refs=refs,
                model_type=model_type,
                device="cpu",
                verbose=False,
            )
        else:
            raise

    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }
