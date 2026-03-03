"""
Pipeline Module - Legal PDF Processing Pipeline

End-to-end pipeline for legal PDF summarization:
PDF → Extract → Preprocess → Chunk → Embed → Retrieve → Summarize

Usage:
    from pipeline import (
        PDFLoader, preprocess, chunk_text, 
        LegalEmbedder, select_chunks, summarize_document
    )
"""

# Import all public exports from each module
from pipeline.pdf_loader import *
from pipeline.preprocessor import *
from pipeline.chunker import *
from pipeline.embedder import *
from pipeline.retriever import *
from pipeline.summarizer import *

# Alias for CLI entry point
from pipeline.preprocessor import main as preprocess_main

# Consolidated __all__ from all submodules
__all__ = [
    # PDF Loader
    'PDFLoader',
    # Preprocessor
    'TextPreprocessor', 'PageType',
    'preprocess', 'preprocess_file', 'preprocess_directory',
    'get_summarization_text', 'get_sections', 'get_clean_text',
    'get_metadata', 'get_citations', 'preprocess_main',
    # Chunker
    'DocumentChunker', 'Chunk', 'chunk_text', 'chunk_document',
    # Embedder
    'LegalEmbedder', 'EmbeddingResult', 'embed_text', 'embed_chunks',
    # Retriever
    'LegalRetriever', 'select_chunks', 'LEGAL_INTENT_QUERIES',
    # Summarizer
    'LegalSummarizer', 'summarize_document', 'summarize_chunks',
]
