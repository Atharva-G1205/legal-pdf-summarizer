"""
Document Chunker Module - Step 3 of Summarization Pipeline
===========================================================

Splits preprocessed legal documents into overlapping chunks suitable for
embedding generation with InLegalBERT.

Features:
- Configurable chunk size (300-500 words, default 400)
- Overlapping chunks for context continuity
- Respects paragraph/section boundaries when possible
- Handles edge cases (short docs, single paragraphs)

Usage:
    from pipeline.chunker import DocumentChunker, chunk_text, chunk_document
    
    # Quick function call
    chunks = chunk_text(long_legal_text)
    
    # Or with custom settings
    chunker = DocumentChunker(target_words=400, overlap_words=50)
    chunks = chunker.chunk_text(text)
    
    # From preprocessed document
    chunks = chunk_document(preprocessed_doc)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'DocumentChunker',
    'Chunk',
    'chunk_text',
    'chunk_document',
]


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    index: int
    word_count: int
    char_start: int
    char_end: int
    section: Optional[str] = None
    
    def __repr__(self):
        preview = self.text[:50] + '...' if len(self.text) > 50 else self.text
        return f"Chunk({self.index}, words={self.word_count}, section={self.section}, text='{preview}')"


class DocumentChunker:
    """
    Splits legal documents into overlapping chunks for embedding.
    
    The chunker respects paragraph boundaries when possible, ensuring
    chunks don't break mid-sentence. Overlap ensures context continuity
    between adjacent chunks.
    """
    
    def __init__(
        self,
        target_words: int = 400,
        min_words: int = 300,
        max_words: int = 500,
        overlap_words: int = 50
    ):
        """
        Initialize the chunker.
        
        Args:
            target_words: Target chunk size in words (default 400)
            min_words: Minimum chunk size (default 300)
            max_words: Maximum chunk size (default 500)
            overlap_words: Number of overlapping words between chunks (default 50)
        """
        self.target_words = target_words
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
        
        # Patterns for splitting
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        self.section_marker_pattern = re.compile(r'\[([A-Z]+)\]')
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = self.paragraph_pattern.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section marker from text if present."""
        match = self.section_marker_pattern.search(text)
        return match.group(1) if match else None
    
    def _merge_small_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Merge very small paragraphs with neighbors."""
        if not paragraphs:
            return []
        
        merged = []
        buffer = ""
        
        for para in paragraphs:
            if self._count_words(buffer + " " + para) <= self.max_words:
                buffer = (buffer + "\n\n" + para).strip() if buffer else para
            else:
                if buffer:
                    merged.append(buffer)
                buffer = para
        
        if buffer:
            merged.append(buffer)
        
        return merged
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        total_words = self._count_words(text)
        
        # If text is short enough, return as single chunk
        if total_words <= self.max_words:
            return [Chunk(
                text=text,
                index=0,
                word_count=total_words,
                char_start=0,
                char_end=len(text),
                section=self._extract_section(text)
            )]
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        paragraphs = self._merge_small_paragraphs(paragraphs)
        
        chunks = []
        current_chunk_paras = []
        current_word_count = 0
        char_position = 0
        
        for para in paragraphs:
            para_words = self._count_words(para)
            
            # If single paragraph exceeds max, split by sentences
            if para_words > self.max_words:
                # Flush current buffer first
                if current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        word_count=self._count_words(chunk_text),
                        char_start=char_position,
                        char_end=char_position + len(chunk_text),
                        section=self._extract_section(chunk_text)
                    ))
                    char_position += len(chunk_text) + 2
                    current_chunk_paras = []
                    current_word_count = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._chunk_by_sentences(para, char_position)
                for sc in sentence_chunks:
                    sc.index = len(chunks)
                    chunks.append(sc)
                    char_position = sc.char_end + 2
                continue
            
            # Check if adding this paragraph exceeds target
            if current_word_count + para_words > self.target_words:
                # If we have enough words, create a chunk
                if current_word_count >= self.min_words:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        word_count=current_word_count,
                        char_start=char_position,
                        char_end=char_position + len(chunk_text),
                        section=self._extract_section(chunk_text)
                    ))
                    
                    # Calculate overlap - keep last paragraph(s) for context
                    overlap_paras = []
                    overlap_words = 0
                    for p in reversed(current_chunk_paras):
                        p_words = self._count_words(p)
                        if overlap_words + p_words <= self.overlap_words:
                            overlap_paras.insert(0, p)
                            overlap_words += p_words
                        else:
                            break
                    
                    char_position += len(chunk_text) + 2
                    current_chunk_paras = overlap_paras
                    current_word_count = overlap_words
            
            current_chunk_paras.append(para)
            current_word_count += para_words
        
        # Don't forget the last chunk
        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras)
            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                word_count=self._count_words(chunk_text),
                char_start=char_position,
                char_end=char_position + len(chunk_text),
                section=self._extract_section(chunk_text)
            ))
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.index = i
        
        logger.info(f"Created {len(chunks)} chunks from {total_words} words")
        return chunks
    
    def _chunk_by_sentences(self, text: str, start_pos: int) -> List[Chunk]:
        """Chunk a large paragraph by sentences."""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_sentences = []
        current_word_count = 0
        char_position = start_pos
        
        for sentence in sentences:
            sent_words = self._count_words(sentence)
            
            if current_word_count + sent_words > self.target_words and current_word_count >= self.min_words:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=0,  # Will be re-indexed
                    word_count=current_word_count,
                    char_start=char_position,
                    char_end=char_position + len(chunk_text),
                    section=self._extract_section(chunk_text)
                ))
                
                # Keep last sentence for overlap
                overlap_sent = current_sentences[-1] if current_sentences else ""
                char_position += len(chunk_text) + 1
                current_sentences = [overlap_sent] if overlap_sent else []
                current_word_count = self._count_words(overlap_sent) if overlap_sent else 0
            
            current_sentences.append(sentence)
            current_word_count += sent_words
        
        # Last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=0,
                word_count=current_word_count,
                char_start=char_position,
                char_end=char_position + len(chunk_text),
                section=self._extract_section(chunk_text)
            ))
        
        return chunks
    
    def chunk_document(self, preprocessed_doc: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a preprocessed document.
        
        Uses the summarization_input text from preprocessor output.
        
        Args:
            preprocessed_doc: Output from TextPreprocessor.clean_document()
            
        Returns:
            List of Chunk objects
        """
        # Get summarization input text
        if 'summarization_input' in preprocessed_doc:
            text = preprocessed_doc['summarization_input'].get('text', '')
        elif 'text' in preprocessed_doc:
            text = preprocessed_doc['text']
        else:
            # Try to extract from sections
            sections = preprocessed_doc.get('sections', {})
            text = "\n\n".join(
                s.get('text', s) if isinstance(s, dict) else s 
                for s in sections.values()
            )
        
        if not text:
            logger.warning("No text found in document")
            return []
        
        chunks = self.chunk_text(text)
        
        # Add document metadata to chunks if available
        filename = preprocessed_doc.get('filename', 'unknown')
        for chunk in chunks:
            chunk.source_file = filename
        
        return chunks


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
_chunker: Optional[DocumentChunker] = None

def _get_chunker() -> DocumentChunker:
    """Get or create singleton chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = DocumentChunker()
    return _chunker


def chunk_text(
    text: str,
    target_words: int = 400,
    overlap_words: int = 50
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        target_words: Target chunk size in words
        overlap_words: Overlap between chunks
        
    Returns:
        List of Chunk objects
    
    Example:
        >>> from pipeline.chunker import chunk_text
        >>> chunks = chunk_text(long_legal_text)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.index}: {chunk.word_count} words")
    """
    chunker = DocumentChunker(
        target_words=target_words,
        overlap_words=overlap_words
    )
    return chunker.chunk_text(text)


def chunk_document(preprocessed_doc: Dict[str, Any]) -> List[Chunk]:
    """
    Chunk a preprocessed document.
    
    Args:
        preprocessed_doc: Output from TextPreprocessor.clean_document()
        
    Returns:
        List of Chunk objects
    
    Example:
        >>> from pipeline.chunker import chunk_document
        >>> from pipeline.preprocessor import preprocess_file
        >>> doc = preprocess_file('case.json')
        >>> chunks = chunk_document(doc)
    """
    return _get_chunker().chunk_document(preprocessed_doc)
