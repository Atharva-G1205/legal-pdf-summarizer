"""
Legal Retriever Module - Steps 6-7 of Summarization Pipeline
==============================================================

Retrieves relevant chunks using:
1. Global relevance (similarity to document centroid)
2. Section-based relevance (legal intent queries)

Features:
- Cosine similarity scoring
- Legal intent queries (Facts, Issues, Arguments, Reasoning, Decision)
- Duplicate removal and ranking
- Configurable top-k selection

Usage:
    from pipeline.retriever import LegalRetriever, select_chunks
    
    # Quick usage
    selected = select_chunks(chunks, embeddings, top_n=12)
    
    # Or with custom settings
    retriever = LegalRetriever(embedder)
    selected = retriever.select_chunks(chunks, embeddings, top_n=15)
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'LegalRetriever',
    'select_chunks',
    'LEGAL_INTENT_QUERIES',
]


# Legal intent queries for section-based retrieval
LEGAL_INTENT_QUERIES = {
    'facts': "What are the material facts and circumstances of this case?",
    'issues': "What are the legal issues and questions for consideration?",
    'arguments': "What arguments and submissions were made by the parties and their counsel?",
    'reasoning': "What is the court's legal analysis, reasoning, and interpretation?",
    'decision': "What is the final decision, order, and judgment of the court?"
}


@dataclass
class RetrievalResult:
    """Container for chunk retrieval results."""
    chunks: List[Any]
    scores: List[float]
    selection_method: str
    
    def __repr__(self):
        return f"RetrievalResult(chunks={len(self.chunks)}, avg_score={np.mean(self.scores):.3f})"


class LegalRetriever:
    """
    Retrieves relevant chunks for summarization using semantic similarity.
    
    Combines global relevance (document centroid) with section-based 
    retrieval (legal intent queries) to select the most important chunks.
    """
    
    def __init__(self, embedder=None):
        """
        Initialize retriever.
        
        Args:
            embedder: Optional LegalEmbedder instance for query embedding.
                     If None, will lazy-load when needed.
        """
        self.embedder = embedder
        self._intent_embeddings = None
    
    def _get_embedder(self):
        """Lazy-load embedder if not provided."""
        if self.embedder is None:
            from pipeline.embedder import LegalEmbedder
            logger.info("Lazy-loading LegalEmbedder for query embedding")
            self.embedder = LegalEmbedder()
        return self.embedder
    
    def _get_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """Get or compute embeddings for legal intent queries."""
        if self._intent_embeddings is None:
            embedder = self._get_embedder()
            logger.info("Embedding legal intent queries...")
            
            self._intent_embeddings = {}
            for intent, query in LEGAL_INTENT_QUERIES.items():
                embedding = embedder.embed_text(query)
                self._intent_embeddings[intent] = embedding
            
            logger.info(f"Embedded {len(self._intent_embeddings)} intent queries")
        
        return self._intent_embeddings
    
    def score_global_relevance(
        self,
        chunk_embeddings: np.ndarray,
        centroid: np.ndarray
    ) -> np.ndarray:
        """
        Score chunks by similarity to document centroid.
        
        Args:
            chunk_embeddings: Array of shape (n_chunks, 768)
            centroid: Document centroid of shape (768,)
            
        Returns:
            Similarity scores of shape (n_chunks,)
        """
        # Normalize embeddings
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        chunk_norms = chunk_embeddings / (
            np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-9
        )
        
        # Cosine similarity
        similarities = np.dot(chunk_norms, centroid_norm)
        return similarities
    
    def score_section_relevance(
        self,
        chunk_embeddings: np.ndarray,
        top_k: int = 3
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Score chunks by similarity to legal intent queries.
        
        Retrieves top-k chunks per intent, then computes composite score.
        
        Args:
            chunk_embeddings: Array of shape (n_chunks, 768)
            top_k: Number of top chunks to retrieve per intent
            
        Returns:
            Tuple of (composite_scores, intent_chunks_dict)
            - composite_scores: Average similarity across all intents
            - intent_chunks_dict: Dict mapping intent -> list of chunk indices
        """
        intent_embeddings = self._get_intent_embeddings()
        n_chunks = len(chunk_embeddings)
        
        # Normalize chunk embeddings once
        chunk_norms = chunk_embeddings / (
            np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-9
        )
        
        # Score against each intent
        intent_scores = np.zeros((n_chunks, len(intent_embeddings)))
        intent_top_chunks = {}
        
        for i, (intent, query_embedding) in enumerate(intent_embeddings.items()):
            # Normalize query
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            
            # Compute similarities
            similarities = np.dot(chunk_norms, query_norm)
            intent_scores[:, i] = similarities
            
            # Get top-k chunk indices for this intent
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            intent_top_chunks[intent] = top_indices.tolist()
        
        # Composite score: average across all intents
        composite = np.mean(intent_scores, axis=1)
        
        return composite, intent_top_chunks
    
    def select_chunks(
        self,
        chunks: List[Any],
        embeddings: np.ndarray,
        top_n: int = 12,
        global_weight: float = 0.4,
        section_weight: float = 0.6,
        diversity_threshold: float = 0.95
    ) -> List[Any]:
        """
        Select top-n chunks using hybrid scoring.
        
        Combines global relevance (centroid similarity) with section-based
        relevance (legal intent queries), removes duplicates, and enforces
        diversity to avoid redundant chunks.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Chunk embeddings array of shape (n_chunks, 768)
            top_n: Number of chunks to select (default: 12)
            global_weight: Weight for global relevance (default: 0.4)
            section_weight: Weight for section relevance (default: 0.6)
            diversity_threshold: Similarity threshold above which chunks
                                are considered duplicates (default: 0.95)
            
        Returns:
            List of selected Chunk objects
        """
        if len(chunks) == 0:
            return []
        
        if len(chunks) <= top_n:
            logger.info(f"Document has {len(chunks)} chunks, returning all")
            return chunks
        
        logger.info(f"Selecting {top_n} from {len(chunks)} chunks...")
        
        # Step 1: Compute document centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Step 2: Global relevance scoring
        global_scores = self.score_global_relevance(embeddings, centroid)
        
        # Step 3: Section-based relevance scoring
        section_scores, intent_chunks = self.score_section_relevance(
            embeddings, 
            top_k=max(3, top_n // 5)  # Scale top-k with top_n
        )
        
        # Step 4: Combine scores
        combined_scores = (
            global_weight * global_scores + 
            section_weight * section_scores
        )
        
        # Step 5: Collect candidate chunks
        # Get top chunks from combined scoring
        top_combined = np.argsort(combined_scores)[-top_n*2:][::-1]  # Get 2x for diversity filtering
        
        # Add top chunks from each intent
        intent_indices = set()
        for intent_list in intent_chunks.values():
            intent_indices.update(intent_list)
        
        # Merge candidates
        candidate_indices = set(top_combined.tolist()) | intent_indices
        candidate_indices = sorted(candidate_indices, 
                                  key=lambda i: combined_scores[i], 
                                  reverse=True)
        
        # Step 6: Diversity-based selection
        selected_indices = []
        selected_embeddings = []
        
        for idx in candidate_indices:
            if len(selected_indices) >= top_n:
                break
            
            # Check diversity with already selected chunks
            if selected_embeddings:
                # Compute similarity with selected chunks
                current_emb = embeddings[idx]
                selected_embs = np.array(selected_embeddings)
                
                # Normalize
                current_norm = current_emb / (np.linalg.norm(current_emb) + 1e-9)
                selected_norms = selected_embs / (
                    np.linalg.norm(selected_embs, axis=1, keepdims=True) + 1e-9
                )
                
                # Compute similarities
                sims = np.dot(selected_norms, current_norm)
                max_sim = np.max(sims)
                
                # Skip if too similar to any selected chunk
                if max_sim > diversity_threshold:
                    logger.debug(f"Skipping chunk {idx} (similarity={max_sim:.3f})")
                    continue
            
            selected_indices.append(idx)
            selected_embeddings.append(embeddings[idx])
        
        # Step 7: Sort by original document order for coherent reading
        selected_indices.sort()
        
        selected_chunks = [chunks[i] for i in selected_indices]
        selected_scores = [combined_scores[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_chunks)} chunks "
                   f"(avg score: {np.mean(selected_scores):.3f})")
        logger.info(f"Score range: [{np.min(selected_scores):.3f}, "
                   f"{np.max(selected_scores):.3f}]")
        
        return selected_chunks
    
    def get_selection_stats(
        self,
        chunks: List[Any],
        embeddings: np.ndarray,
        selected_chunks: List[Any]
    ) -> Dict[str, Any]:
        """
        Get statistics about the selection process.
        
        Returns coverage metrics, section distribution, etc.
        """
        selected_indices = set(
            chunks.index(c) for c in selected_chunks if c in chunks
        )
        
        # Section distribution
        section_counts = {}
        for idx in selected_indices:
            chunk = chunks[idx]
            section = getattr(chunk, 'section', 'unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        # Coverage (% of document)
        total_words = sum(getattr(c, 'word_count', 0) for c in chunks)
        selected_words = sum(getattr(c, 'word_count', 0) for c in selected_chunks)
        coverage = selected_words / total_words if total_words > 0 else 0
        
        return {
            'total_chunks': len(chunks),
            'selected_chunks': len(selected_chunks),
            'selection_ratio': len(selected_chunks) / len(chunks),
            'section_distribution': section_counts,
            'word_coverage': coverage,
            'total_words': total_words,
            'selected_words': selected_words,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_retriever: Optional[LegalRetriever] = None

def _get_retriever() -> LegalRetriever:
    """Get or create singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever()
    return _retriever


def select_chunks(
    chunks: List[Any],
    embeddings: np.ndarray,
    top_n: int = 12
) -> List[Any]:
    """
    Select top-n relevant chunks for summarization.
    
    Convenience function that uses default retriever settings.
    
    Args:
        chunks: List of Chunk objects
        embeddings: Chunk embeddings array
        top_n: Number of chunks to select
        
    Returns:
        List of selected chunks
    
    Example:
        >>> from pipeline import chunk_text, LegalEmbedder, select_chunks
        >>> chunks = chunk_text(legal_text)
        >>> embedder = LegalEmbedder()
        >>> result = embedder.embed_chunks(chunks)
        >>> selected = select_chunks(chunks, result.embeddings, top_n=10)
    """
    return _get_retriever().select_chunks(chunks, embeddings, top_n)
