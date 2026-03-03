"""
Legal Summarizer Module - Steps 8-9 of Summarization Pipeline
===============================================================

Abstractive summarization using Legal-Pegasus (or fallback models).

Features:
- Legal-Pegasus fine-tuned for legal text
- Hierarchical summarization (chunk-level → document-level)
- GPU acceleration with batch processing
- Fallback to general Pegasus or Long-T5

Usage:
    from pipeline.summarizer import LegalSummarizer, summarize_document
    
    # Quick usage
    summary = summarize_document(selected_chunks)
    
    # Or with custom settings
    summarizer = LegalSummarizer()
    summary = summarizer.hierarchical_summary(selected_chunks)
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'LegalSummarizer',
    'summarize_document',
    'summarize_chunks',
]


@dataclass
class SummaryResult:
    """Container for summarization results."""
    summary: str
    chunk_summaries: Optional[List[str]] = None
    model_name: str = ""
    input_length: int = 0
    output_length: int = 0
    
    def __repr__(self):
        return f"SummaryResult(length={len(self.summary)} chars, model={self.model_name})"


class LegalSummarizer:
    """
    Abstractive summarization using Legal-Pegasus or fallback models.
    
    Implements hierarchical summarization:
    1. First pass: Summarize each selected chunk
    2. Second pass: Summarize the chunk summaries into final document summary
    """
    
    # Model preferences (in order of priority)
    LEGAL_PEGASUS = "nsi319/legal-pegasus"
    PEGASUS_LARGE = "google/pegasus-large"
    LONG_T5 = "google/long-t5-tglobal-base"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 100,
        batch_size: int = 4
    ):
        """
        Initialize summarizer with model.
        
        Args:
            model_name: HuggingFace model name. If None, tries Legal-Pegasus first.
            device: Device to use ('cuda', 'cpu'). If None, auto-detects.
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            batch_size: Batch size for processing multiple chunks
        """
        self.max_length = max_length
        self.min_length = min_length
        self.batch_size = batch_size
        
        # Lazy imports
        self._lazy_import()
        
        # Auto-detect device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Try models in order of preference
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
        if model_name:
            # User specified a model
            self._load_model(model_name)
        else:
            # Try in order of preference
            for model in [self.LEGAL_PEGASUS, self.PEGASUS_LARGE, self.LONG_T5]:
                logger.info(f"Attempting to load {model}...")
                if self._load_model(model):
                    break
            
            if self.model is None:
                raise RuntimeError(
                    "Failed to load any summarization model. "
                    "Please install transformers and try again."
                )
        
        logger.info(f"✓ Loaded {self.model_name} on {self.device}")
    
    def _lazy_import(self):
        """Lazy import heavy dependencies."""
        global AutoTokenizer, AutoModelForSeq2SeqLM, torch
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for summarization. "
                "Install with: pip install transformers torch"
            ) from e
    
    def _load_model(self, model_name: str) -> bool:
        """
        Try to load a model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_name = model_name
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            return False
    
    def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> str:
        """
        Summarize a single text.
        
        Args:
            text: Input text to summarize
            max_length: Override max summary length
            min_length: Override min summary length
            
        Returns:
            Summary string
        """
        if not text or not text.strip():
            return ""
        
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=1024,  # Input max length
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary with anti-repetition controls
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                num_beams=4,
                no_repeat_ngram_size=2,  # Prevent 3-gram repetition
                repetition_penalty=1.5,   # Penalize repeated tokens
                early_stopping=True
            )
        
        # Decode
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary.strip()
    
    def summarize_chunks(
        self,
        chunks: List[Any],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> List[str]:
        """
        Summarize multiple chunks (first pass).
        
        Args:
            chunks: List of Chunk objects
            max_length: Override max summary length per chunk
            min_length: Override min summary length per chunk
            
        Returns:
            List of chunk summaries
        """
        if not chunks:
            return []
        
        logger.info(f"Summarizing {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [
            chunk.text if hasattr(chunk, 'text') else str(chunk)
            for chunk in chunks
        ]
        
        # Batch processing
        summaries = []
        max_len = max_length or (self.max_length // 2)  # Shorter for chunk summaries
        min_len = min_length or (self.min_length // 2)
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            for text in batch:
                summary = self.summarize_text(text, max_len, min_len)
                summaries.append(summary)
        
        logger.info(f"✓ Generated {len(summaries)} chunk summaries")
        return summaries
    
    def hierarchical_summary(
        self,
        chunks: List[Any],
        final_max_length: Optional[int] = None,
        final_min_length: Optional[int] = None
    ) -> str:
        """
        Generate hierarchical summary (two-pass).
        
        Pass 1: Summarize each chunk
        Pass 2: Summarize the chunk summaries into final document summary
        
        Args:
            chunks: List of selected Chunk objects
            final_max_length: Max length for final summary
            final_min_length: Min length for final summary
            
        Returns:
            Final document summary
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            # Single chunk: just summarize it directly
            logger.info("Single chunk, direct summarization")
            return self.summarize_text(
                chunks[0].text if hasattr(chunks[0], 'text') else str(chunks[0]),
                final_max_length,
                final_min_length
            )
        
        logger.info(f"Starting hierarchical summarization of {len(chunks)} chunks...")
        
        # Pass 1: Chunk-level summaries
        chunk_summaries = self.summarize_chunks(chunks)
        
        # Pass 2: Combine and summarize chunk summaries
        combined = "\n\n".join(chunk_summaries)
        logger.info(f"Combined chunk summaries: {len(combined)} chars")
        
        final_max = final_max_length or self.max_length
        final_min = final_min_length or self.min_length
        
        logger.info("Generating final document summary...")
        final_summary = self.summarize_text(combined, final_max, final_min)
        
        logger.info(f"✓ Final summary: {len(final_summary)} chars")
        return final_summary
    
    def generate_structured_summary(
        self,
        chunks: List[Any],
        include_chunk_summaries: bool = False
    ) -> SummaryResult:
        """
        Generate structured summary with metadata.
        
        Args:
            chunks: List of Chunk objects 
            include_chunk_summaries: Whether to include intermediate summaries
            
        Returns:
            SummaryResult with summary and metadata
        """
        if not chunks:
            return SummaryResult(
                summary="",
                chunk_summaries=[],
                model_name=self.model_name,
                input_length=0,
                output_length=0
            )
        
        # Calculate input length
        input_text = "\n".join(
            chunk.text if hasattr(chunk, 'text') else str(chunk)
            for chunk in chunks
        )
        input_length = len(input_text)
        
        # Generate summaries
        if include_chunk_summaries:
            chunk_summaries = self.summarize_chunks(chunks)
            combined = "\n\n".join(chunk_summaries)
            final_summary = self.summarize_text(combined)
        else:
            chunk_summaries = None
            final_summary = self.hierarchical_summary(chunks)
        
        return SummaryResult(
            summary=final_summary,
            chunk_summaries=chunk_summaries,
            model_name=self.model_name,
            input_length=input_length,
            output_length=len(final_summary)
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_summarizer: Optional[LegalSummarizer] = None

def _get_summarizer() -> LegalSummarizer:
    """Get or create singleton summarizer instance."""
    global _summarizer
    if _summarizer is None:
        logger.info("Initializing singleton summarizer...")
        _summarizer = LegalSummarizer()
    return _summarizer


def summarize_document(
    chunks: List[Any],
    max_length: int = 512,
    min_length: int = 100
) -> str:
    """
    Summarize a document from selected chunks.
    
    Convenience function for quick summarization.
    
    Args:
        chunks: List of selected Chunk objects
        max_length: Maximum summary length
        min_length: Minimum summary length
        
    Returns:
        Summary string
    
    Example:
        >>> from pipeline import chunk_text, LegalEmbedder, select_chunks
        >>> from pipeline.summarizer import summarize_document
        >>> 
        >>> chunks = chunk_text(legal_text)
        >>> embedder = LegalEmbedder()
        >>> result = embedder.embed_chunks(chunks)
        >>> selected = select_chunks(chunks, result.embeddings, top_n=10)
        >>> summary = summarize_document(selected)
    """
    summarizer = _get_summarizer()
    return summarizer.hierarchical_summary(
        chunks,
        final_max_length=max_length,
        final_min_length=min_length
    )


def summarize_chunks(chunks: List[Any]) -> List[str]:
    """
    Get individual chunk summaries.
    
    Useful for debugging or when you want fine-grained control.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        List of summary strings
    """
    return _get_summarizer().summarize_chunks(chunks)
