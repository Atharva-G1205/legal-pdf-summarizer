"""
Pipeline Module - Legal PDF Processing Pipeline

Provides tools for extracting, preprocessing, chunking, and embedding legal PDF documents.

Usage:
    from pipeline import preprocess, chunk_document, embed_chunks
    from pipeline import PDFLoader, TextPreprocessor, DocumentChunker, LegalEmbedder
"""

# Import all public exports from each module
from pipeline.pdf_loader import *
from pipeline.preprocessor import *
from pipeline.chunker import *
from pipeline.embedder import *

# Alias for CLI entry point
from pipeline.preprocessor import main as preprocess_main

# Consolidated __all__ from all submodules
__all__ = [
    # PDF Loader
    'PDFLoader',
    # Preprocessor
    'TextPreprocessor', 'PageType',
    'preprocess', 'preprocess_file', 'preprocess_directory', 'process_path',
    'get_summarization_text', 'get_sections', 'get_clean_text',
    'get_metadata', 'get_citations', 'preprocess_main',
    # Chunker
    'DocumentChunker', 'Chunk', 'chunk_text', 'chunk_document',
    # Embedder
    'LegalEmbedder', 'EmbeddingResult', 'embed_text', 'embed_chunks',
]
