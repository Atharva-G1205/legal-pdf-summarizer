"""
Legal Embedder Module
Generates semantic embeddings for text chunks using InLegalBERT,
a BERT model pretrained on Indian legal documents.

Features:
1. Loads law-ai/InLegalBERT from HuggingFace (or local cache)
2. GPU acceleration with automatic device detection
3. Batch processing for efficiency
4. Mean pooling for chunk-level embeddings

Usage:
    from pipeline.embedder import LegalEmbedder, embed_chunks
    
    # Quick function call
    embedder = LegalEmbedder()
    embeddings = embedder.embed_chunks(chunks)
    
    # Single text embedding
    embedding = embedder.embed_text("Legal text here")
    
    # Check device
    print(f"Using device: {embedder.device}")
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports for torch and transformers
torch = None
AutoTokenizer = None
AutoModel = None

def _lazy_import():
    """Lazily import heavy dependencies."""
    global torch, AutoTokenizer, AutoModel
    if torch is None:
        try:
            import torch as _torch
            from transformers import AutoTokenizer as _AutoTokenizer
            from transformers import AutoModel as _AutoModel
            torch = _torch
            AutoTokenizer = _AutoTokenizer
            AutoModel = _AutoModel
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for embedding. "
                "Install with: pip install torch transformers"
            ) from e


__all__ = [
    'LegalEmbedder',
    'embed_text',
    'embed_chunks',
    'EmbeddingResult',
]


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    embeddings: np.ndarray  # Shape: (n_chunks, 768)
    chunk_indices: List[int]
    model_name: str
    device: str
    
    def __repr__(self):
        return f"EmbeddingResult(shape={self.embeddings.shape}, device={self.device})"


class LegalEmbedder:
    """
    Generates embeddings using InLegalBERT.
    
    Uses mean pooling of token embeddings to create fixed-size
    representations for variable-length text chunks.
    """
    
    MODEL_NAME = "law-ai/InLegalBERT"
    EMBEDDING_DIM = 768
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name (default: law-ai/InLegalBERT)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum token length (default 512 for BERT)
            batch_size: Batch size for processing multiple chunks
        """
        _lazy_import()
        
        self.model_name = model_name or self.MODEL_NAME
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing LegalEmbedder with {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to token embeddings.
        
        Uses attention mask to ignore padding tokens.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of shape (768,)
        """
        if not text or not text.strip():
            return np.zeros(self.EMBEDDING_DIM)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(**encoded)
            embedding = self._mean_pooling(output, encoded['attention_mask'])
        
        return embedding.cpu().numpy().squeeze()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape (n_texts, 768)
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Filter empty texts, track indices
            valid_indices = []
            valid_texts = []
            for j, text in enumerate(batch_texts):
                if text and text.strip():
                    valid_indices.append(i + j)
                    valid_texts.append(text)
            
            if not valid_texts:
                # All empty - add zero embeddings
                all_embeddings.extend([np.zeros(self.EMBEDDING_DIM)] * len(batch_texts))
                continue
            
            # Tokenize batch
            encoded = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                output = self.model(**encoded)
                batch_embeddings = self._mean_pooling(output, encoded['attention_mask'])
            
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Reconstruct full batch (including zeros for empty texts)
            embed_idx = 0
            for j, text in enumerate(batch_texts):
                if text and text.strip():
                    all_embeddings.append(batch_embeddings[embed_idx])
                    embed_idx += 1
                else:
                    all_embeddings.append(np.zeros(self.EMBEDDING_DIM))
        
        return np.array(all_embeddings)
    
    def embed_chunks(self, chunks: List[Any]) -> EmbeddingResult:
        """
        Generate embeddings for a list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects (from DocumentChunker)
            
        Returns:
            EmbeddingResult with embeddings array and metadata
        """
        if not chunks:
            return EmbeddingResult(
                embeddings=np.array([]),
                chunk_indices=[],
                model_name=self.model_name,
                device=self.device
            )
        
        # Extract text from chunks
        texts = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
        indices = [chunk.index if hasattr(chunk, 'index') else i for i, chunk in enumerate(chunks)]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embed_texts(texts)
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return EmbeddingResult(
            embeddings=embeddings,
            chunk_indices=indices,
            model_name=self.model_name,
            device=self.device
        )
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and chunk embeddings.
        
        Args:
            query_embedding: Single embedding of shape (768,)
            chunk_embeddings: Array of shape (n_chunks, 768)
            
        Returns:
            Similarity scores of shape (n_chunks,)
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-9)
        
        # Cosine similarity
        similarities = np.dot(chunk_norms, query_norm)
        return similarities
    
    def compute_document_centroid(self, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the centroid (mean embedding) of all chunks.
        
        This represents the overall semantic theme of the document.
        
        Args:
            chunk_embeddings: Array of shape (n_chunks, 768)
            
        Returns:
            Centroid embedding of shape (768,)
        """
        if len(chunk_embeddings) == 0:
            return np.zeros(self.EMBEDDING_DIM)
        return np.mean(chunk_embeddings, axis=0)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
_embedder: Optional[LegalEmbedder] = None

def _get_embedder() -> LegalEmbedder:
    """Get or create singleton embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = LegalEmbedder()
    return _embedder


def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Input text string
        
    Returns:
        Numpy array of shape (768,)
    
    Example:
        >>> from pipeline.embedder import embed_text
        >>> embedding = embed_text("This is a legal judgment about...")
        >>> print(embedding.shape)  # (768,)
    """
    return _get_embedder().embed_text(text)


def embed_chunks(chunks: List[Any]) -> EmbeddingResult:
    """
    Generate embeddings for a list of chunks.
    
    Args:
        chunks: List of Chunk objects (from DocumentChunker)
        
    Returns:
        EmbeddingResult with embeddings array and metadata
    
    Example:
        >>> from pipeline.chunker import chunk_text
        >>> from pipeline.embedder import embed_chunks
        >>> chunks = chunk_text(long_legal_text)
        >>> result = embed_chunks(chunks)
        >>> print(result.embeddings.shape)  # (n_chunks, 768)
    """
    return _get_embedder().embed_chunks(chunks)
