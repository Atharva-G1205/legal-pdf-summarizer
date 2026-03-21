"""
Legal Summarizer Module — Abstractive Summarization
=====================================================

Generates summaries using locally stored Legal-LED or Legal-Pegasus models.

Designed to receive **pre-ranked sentences** (from ``retriever.rank_sentences``)
rather than raw chunks.  This avoids the context-overflow and "word salad"
problems documented in the project reference notes.

Models are loaded from ``<project>/models/`` by key:
  - ``led``     → models/LED/IN_model/
  - ``pegasus`` → models/Pegasus/IN_pegasus_end/

Usage:
    from pipeline.summarizer import LegalSummarizer, summarize_document

    summarizer = LegalSummarizer(model_key="led")
    result = summarizer.summarize_text(text, max_length=350, min_length=150)

    # Or convenience function
    result = summarize_document(sentences, config)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy-loaded heavy imports
torch = None
AutoTokenizer = None
AutoModelForSeq2SeqLM = None


def _lazy_import():
    """Lazily import torch + transformers."""
    global torch, AutoTokenizer, AutoModelForSeq2SeqLM
    if torch is None:
        try:
            import torch as _torch
            from transformers import (
                AutoTokenizer as _AT,
                AutoModelForSeq2SeqLM as _AM,
            )
            torch = _torch
            AutoTokenizer = _AT
            AutoModelForSeq2SeqLM = _AM
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for summarization. "
                "Install with: pip install torch transformers"
            ) from exc


__all__ = [
    "LegalSummarizer",
    "SummaryResult",
    "summarize_document",
]

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_MODELS: Dict[str, Path] = {
    "led": _PROJECT_ROOT / "models" / "LED" / "IN_model",
    "pegasus": _PROJECT_ROOT / "models" / "Pegasus" / "IN_pegasus_end",
}

# HuggingFace model names (used for 'huggingface' source or tokenizer fallback)
HF_MODELS: Dict[str, str] = {
    "led": "allenai/led-base-16384",
    "pegasus": "nsi319/legal-pegasus",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SummaryResult:
    """Container for a generated summary."""

    summary: str
    model_used: str
    input_word_count: int
    output_word_count: int

    def __repr__(self) -> str:
        return (
            f"SummaryResult(model={self.model_used!r}, "
            f"in={self.input_word_count}w, out={self.output_word_count}w)"
        )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class LegalSummarizer:
    """
    Abstractive summarizer backed by local Legal-LED / Legal-Pegasus models.

    Accepts a flat string of pre-ranked sentences (typically 500–800 words)
    and produces a concise abstractive summary.
    """

    def __init__(
        self,
        model_key: str = "led",
        device: Optional[str] = None,
        model_source: str = "finetuned",
    ):
        """
        Args:
            model_key:     ``"led"`` or ``"pegasus"`` (maps to local model dirs).
            device:        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
            model_source:  ``"finetuned"`` loads from local `models/` dir,
                           ``"huggingface"`` downloads from HuggingFace Hub.
        """
        _lazy_import()

        if model_key not in LOCAL_MODELS:
            raise ValueError(
                f"Unknown model_key {model_key!r}. Choose from {list(LOCAL_MODELS)}"
            )

        self.model_key = model_key
        self.model_source = model_source
        local_path = LOCAL_MODELS[model_key]
        hf_name = HF_MODELS[model_key]

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_source == "huggingface":
            # --- Load everything from HuggingFace Hub ---
            logger.info(f"Loading {model_key} from HuggingFace: {hf_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
            self.model_name = hf_name
        else:
            # --- Load from local finetuned checkpoint ---
            # Tokenizer: try local first, fall back to HF
            tokenizer_path = local_path if (local_path / "tokenizer_config.json").exists() else None
            if tokenizer_path:
                logger.info(f"Loading tokenizer from local path: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                logger.info(
                    f"Local tokenizer missing at {local_path}, "
                    f"falling back to HuggingFace: {hf_name}"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(hf_name)

            # Model weights
            logger.info(f"Loading model weights from {local_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(local_path))
            self.model_name = str(local_path.name)

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"✓ {model_key} model loaded on {self.device} (source: {model_source})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_input_max(self) -> int:
        """Max input tokens the underlying model accepts."""
        return 16384 if self.model_key == "led" else 1024

    def _build_gen_kwargs(
        self,
        inputs: Any,
        max_length: int,
        min_length: int,
    ) -> Dict[str, Any]:
        """Build generation kwargs, with model-specific tweaks.

        Hyperparameters are tuned for *legal* text which is naturally
        repetitive (party names, section references, etc.).
        Aggressive repetition penalties cause entropy collapse →
        gibberish like "tchaffchaffchaffr".
        """
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_length,
            min_new_tokens=max(1, min(min_length, max_length // 2)),
            num_beams=4,
            length_penalty=1.0,           # ↑ from 0.8 — finish thoughts, don't cut off
            no_repeat_ngram_size=2,       # ↓ from 3 — legal text repeats legitimately
            repetition_penalty=1.1,       # ↓ from 1.2 — less aggressive for legal prose
            do_sample=False,
            early_stopping=True,
        )
        # LED-specific: set global_attention on the metadata prefix
        # (first 64 tokens) so the model grounds on case/court/acts.
        if self.model_key == "led":
            seq_len = inputs["input_ids"].shape[1]
            global_attn = torch.zeros_like(inputs["input_ids"])
            # Attend globally to first 64 tokens (prefix) + last 32 (verdict)
            global_attn[:, :min(64, seq_len)] = 1
            global_attn[:, max(0, seq_len - 32):] = 1
            gen_kwargs["global_attention_mask"] = global_attn
        return gen_kwargs

    def _generate_once(self, text: str, max_length: int, min_length: int) -> str:
        """Single-pass generation (no overflow handling)."""
        input_max = self._get_input_max()
        inputs = self.tokenizer(
            text,
            max_length=input_max,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        gen_kwargs = self._build_gen_kwargs(inputs, max_length, min_length)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs,
            )
        raw = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return _clean_decode_artifacts(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summarize_text(
        self,
        text: str,
        max_length: int = 350,
        min_length: int = 100,
        grounding: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate an abstractive summary of *text*.

        If the text exceeds the model's token budget, automatically
        switches to **hierarchical mode**: split → summarize each →
        merge → re-summarize.

        Args:
            text:       Input text (concatenated ranked sentences).
            max_length: Maximum output length in tokens.
            min_length: Minimum output length in tokens.
            grounding:  Optional metadata dict for grounding prefix.

        Returns:
            Summary string.
        """
        if not text or not text.strip():
            return ""

        # Prepend lightweight metadata prefix (not a verbose instruction)
        prefix = _build_metadata_prefix(grounding)
        full_text = prefix + text

        # Check whether the input fits the model's context window
        input_max = self._get_input_max()
        token_count = len(self.tokenizer.encode(full_text, add_special_tokens=False))

        if token_count <= input_max:
            # --- Single-pass: fits within context window ---
            summary = self._generate_once(full_text, max_length, min_length)
        else:
            # --- Hierarchical mode: split → summarize → merge → re-summarize ---
            logger.info(
                f"Input ({token_count} tokens) exceeds {self.model_key} limit "
                f"({input_max} tokens). Switching to hierarchical mode."
            )
            summary = self._hierarchical_summarize(
                text, prefix, max_length, min_length
            )

        if grounding:
            summary = validate_and_clean_summary(summary, grounding)
        return summary

    def _hierarchical_summarize(
        self,
        text: str,
        prefix: str,
        max_length: int,
        min_length: int,
    ) -> str:
        """
        Hierarchical summarization for texts that exceed the model's
        token budget.

        Strategy:
          1. Split text into groups of ~500 words (safe for both LED/Pegasus)
          2. Summarize each group independently
          3. Concatenate the group summaries
          4. Run a final summarization pass on the merged text
        """
        input_max = self._get_input_max()
        # Reserve tokens for the prefix
        prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        budget_per_group = input_max - prefix_tokens - 32  # 32-token safety margin

        # Split text into sentences, then pack into groups within budget
        sentences = re.split(r'(?<=[.!?])\s+', text)
        groups: List[str] = []
        current_group: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
            if current_tokens + sent_tokens > budget_per_group and current_group:
                groups.append(" ".join(current_group))
                current_group = [sent]
                current_tokens = sent_tokens
            else:
                current_group.append(sent)
                current_tokens += sent_tokens
        if current_group:
            groups.append(" ".join(current_group))

        logger.info(f"Hierarchical mode: splitting into {len(groups)} groups")

        # Summarize each group — shorter target, proportional to group count
        per_group_max = max(80, max_length // len(groups))
        per_group_min = max(30, min_length // len(groups))
        group_summaries: List[str] = []

        for i, group_text in enumerate(groups, 1):
            logger.info(f"  Summarizing group {i}/{len(groups)} "
                        f"({len(group_text.split())} words)")
            chunk_input = prefix + group_text
            s = self._generate_once(chunk_input, per_group_max, per_group_min)
            if s.strip():
                group_summaries.append(s.strip())

        if not group_summaries:
            logger.warning("All group summaries empty — falling back to truncated input")
            return self._generate_once(prefix + text, max_length, min_length)

        # Merge and do a final pass
        merged = " ".join(group_summaries)
        merged_tokens = len(self.tokenizer.encode(prefix + merged, add_special_tokens=False))

        if merged_tokens <= input_max:
            logger.info(f"Final merge pass ({len(merged.split())} words)")
            return self._generate_once(prefix + merged, max_length, min_length)
        else:
            # Merged text still too long — just return concatenated summaries
            logger.warning(
                f"Merged summaries still exceed limit ({merged_tokens} tokens). "
                f"Returning concatenated group summaries."
            )
            return merged

    def summarize_sentences(
        self,
        sentences: List[Dict[str, Any]],
        max_length: int = 350,
        min_length: int = 100,
        grounding: Optional[Dict[str, Any]] = None,
    ) -> SummaryResult:
        """
        Summarise a list of sentence dicts (as returned by ``rank_sentences``).

        Args:
            sentences: List of ``{"text": str, ...}`` dicts.
            max_length: Max summary tokens.
            min_length: Min summary tokens.

        Returns:
            :class:`SummaryResult`.
        """
        combined = " ".join(s["text"] for s in sentences)
        input_wc = len(combined.split())

        summary = self.summarize_text(combined, max_length, min_length, grounding=grounding)

        return SummaryResult(
            summary=summary,
            model_used=self.model_key,
            input_word_count=input_wc,
            output_word_count=len(summary.split()),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def summarize_document(
    sentences: List[Dict[str, Any]],
    config: Any,
    grounding: Optional[Dict[str, Any]] = None,
) -> SummaryResult:
    """
    High-level convenience: load model per *config* and summarise *sentences*.

    Args:
        sentences: Pre-ranked sentence dicts from ``retriever.rank_sentences``.
        config:    A ``SummaryConfig`` instance (from ``config.py``).

    Returns:
        :class:`SummaryResult`.
    """
    model_source = getattr(config, "model_source", "finetuned")
    summarizer = LegalSummarizer(
        model_key=config.model,
        model_source=model_source,
    )
    return summarizer.summarize_sentences(
        sentences,
        max_length=config.max_length,
        min_length=config.min_length,
        grounding=grounding,
    )


def _build_metadata_prefix(grounding: Optional[Dict[str, Any]]) -> str:
    """
    Build a lightweight metadata prefix for seq2seq models.

    These models (LED, Pegasus) are NOT instruction-tuned — they were
    trained on ``document → summary`` pairs.  A verbose "You are a legal
    expert…" prompt wastes tokens and can confuse the decoder.

    Instead, we inject only structured metadata tags that the model can
    use as anchoring context (case name, court, relevant Acts).
    """
    if not grounding:
        return ""

    parts: List[str] = []

    # Case title / header
    header = grounding.get("repeated_header", "") or ""
    if header.strip():
        parts.append(f"[CASE: {header.strip()[:120]}]")

    # Court name (from metadata text, first line often has it)
    meta_text = grounding.get("metadata_text", "") or ""
    if meta_text.strip():
        first_line = meta_text.strip().split("\n")[0].strip()[:100]
        if first_line:
            parts.append(f"[COURT: {first_line}]")

    # Citations / Acts
    citations: List[str] = []
    for k in ("allowed_citations", "citations", "unique_citations"):
        v = grounding.get(k)
        if isinstance(v, list):
            citations.extend([str(x).strip() for x in v if str(x).strip()])
    citations = list(dict.fromkeys(citations))[:10]  # de-dupe, cap at 10
    if citations:
        parts.append(f"[ACTS: {'; '.join(citations)}]")

    if not parts:
        return ""
    return " ".join(parts) + " "


def _clean_decode_artifacts(text: str) -> str:
    """
    Remove common decoding artifacts from model output:
    - Broken bracket sequences: [[[[1], [[[2B], etc.
    - Repeated nonsense fragments (entropy collapse)
    - Partial words / trailing gibberish
    """
    if not text:
        return text

    # 1. Strip broken bracket patterns: [[[[, ]]]], [[[1B], etc.
    text = re.sub(r'\[{2,}[^\]]*\]{0,}', '', text)
    text = re.sub(r'\]{2,}', '', text)
    text = re.sub(r'\[\d+[A-Za-z]*\]', '', text)  # [1], [2B], etc.

    # 2. Detect trailing gibberish — look for the last well-formed sentence
    #    A well-formed sentence ends with . ! or ?
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) > 1:
        # Check if the last "sentence" is gibberish (very low alpha ratio)
        last = sentences[-1]
        alpha_chars = sum(1 for c in last if c.isalpha())
        total_chars = len(last)
        if total_chars > 20 and (alpha_chars / max(total_chars, 1)) < 0.5:
            # Last chunk is likely gibberish — drop it
            sentences = sentences[:-1]
            text = ' '.join(sentences)

    # 3. Clean up extra whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text


def validate_and_clean_summary(summary: str, grounding: Dict[str, Any]) -> str:
    """
    Lightweight "anchor" step: drop obviously hallucinated sentences
    (foreign jurisdictions, finance regulators, impossible courts, etc.)
    and drop statute references that are not present in the extracted citations.
    """
    s = (summary or "").strip()
    if not s:
        return s

    allowed_text_parts: List[str] = []
    for k in ("metadata_text", "metadata", "repeated_header"):
        v = grounding.get(k)
        if isinstance(v, str):
            allowed_text_parts.append(v)
    for k in ("allowed_citations", "citations", "unique_citations"):
        v = grounding.get(k)
        if isinstance(v, list):
            allowed_text_parts.extend([str(x) for x in v])
    allowed_text = " ".join(allowed_text_parts)

    # Expand list of drift markers seen in this project
    suspicious_patterns = [
        r"\bUnited States\b",
        r"\bU\.S\.\b",
        r"\bSecurities Exchange Act\b",
        r"\bRule\s*10b-5\b",
        r"\bCalifornia\b",
        r"\bNew York\b",
        r"\bFederal Court\b",
        r"\bSEBI\b",
        r"\bSecurities and Exchange Board of India\b",
        r"\bBombay High Court\b",
        r"\bCalcutta High Court\b",
        r"\bHigh Court of West Bengal\b",
    ]
    suspicious_regexes = [re.compile(p, re.IGNORECASE) for p in suspicious_patterns]

    # Pre-clean: strip bracket artifacts from model output
    s = _clean_decode_artifacts(s)

    # Statute sanity check: if we have extracted "Section ..." strings, enforce them.
    allowed_sections = [
        str(x).lower() for x in grounding.get("allowed_sections", []) or []
    ]
    if not allowed_sections:
        cites = grounding.get("allowed_citations") or grounding.get("unique_citations") or []
        if isinstance(cites, list):
            allowed_sections = [c.lower() for c in cites if str(c).lower().startswith("section ")]

    section_regex = re.compile(r"\b[Ss]ection\s+\d+[A-Za-z]*(?:\s*\(\d+\))?")

    raw_sentences = re.split(r"(?<=[.!?])\s+", s)
    kept: List[str] = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Drop if it contains a known drift marker not present in metadata/citations
        drop = False
        for rgx in suspicious_regexes:
            m = rgx.search(sent)
            if m and m.group(0).lower() not in allowed_text.lower():
                drop = True
                break
        if drop:
            continue

        # Drop if it mentions a Section not in extracted citations (when available)
        if allowed_sections:
            for match in section_regex.findall(sent):
                norm = match.lower()
                if not any(norm in a for a in allowed_sections):
                    drop = True
                    break
        if drop:
            continue

        kept.append(sent)

    return " ".join(kept) if kept else s
