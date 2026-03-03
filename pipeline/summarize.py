"""
Multi-Level Legal PDF Summarizer
=================================

Generates executive, detailed, or technical summaries based on user choice.

Usage:
    python summarize.py                    # Interactive mode
    python summarize.py --help            # Show help

Configuration:
    Edit config.py to customize models, lengths, and chunk settings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from config import (
    SummaryLevel,
    SummaryConfig,
    get_config_by_number,
    MENU_OPTIONS,
)
from pipeline import (
    PDFLoader,
    preprocess,
    get_summarization_text,
    chunk_text,
    LegalEmbedder,
    select_chunks,
)
from pipeline.summarizer import LegalSummarizer

if TYPE_CHECKING:
    from pipeline.chunker import Chunk
    from pipeline.embedder import EmbeddingResult


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_step(icon: str, message: str, detail: str = '') -> None:
    """Print a pipeline step with optional detail."""
    print(f"\n{icon}  {message}")
    if detail:
        print(f"   {detail}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"   ✓ {message}")


# =============================================================================
# USER INTERACTION
# =============================================================================

def select_summary_type() -> SummaryConfig:
    """Prompt user to select summary type.
    
    Returns:
        Selected SummaryConfig
    """
    print_header("Legal PDF Summarizer - Multi-Level Summary")
    print(MENU_OPTIONS)
    
    while True:
        try:
            choice = input("Enter choice (1/2/3): ").strip()
            return get_config_by_number(int(choice))
        except (ValueError, KeyError):
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def get_pdf_path() -> Path:
    """Prompt user for PDF path and validate.
    
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If not a PDF file
    """
    path_str = input("Enter PDF path: ").strip().strip('"\'')
    path = Path(path_str).expanduser().resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Not a PDF file: {path}")
    
    return path


# =============================================================================
# PIPELINE STEPS
# =============================================================================

def load_document(pdf_path: Path) -> tuple[dict, str]:
    """Load and preprocess PDF document.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (processed_doc, text_content)
    """
    print_step("📂", "Loading PDF...")
    loader = PDFLoader()
    doc = loader.load_pdf(pdf_path)
    
    print_step("📝", "Preprocessing document...")
    processed = preprocess(doc)
    text = get_summarization_text(doc)
    print_success(f"Extracted {len(text):,} characters")
    
    return processed, text


def create_chunks(text: str, config: SummaryConfig) -> list:
    """Chunk text according to config settings.
    
    Args:
        text: Document text
        config: Summary configuration
        
    Returns:
        List of Chunk objects
    """
    print_step("✂️", "Chunking document...")
    chunks = chunk_text(
        text,
        target_words=config.chunk_words,
        overlap_words=config.chunk_overlap
    )
    print_success(f"Created {len(chunks)} chunks")
    return chunks


def embed_and_retrieve(
    chunks: list, 
    config: SummaryConfig
) -> list:
    """Generate embeddings and retrieve top chunks.
    
    Args:
        chunks: List of Chunk objects
        config: Summary configuration
        
    Returns:
        Selected chunks for summarization
    """
    print_step("🧠", "Generating embeddings (InLegalBERT)...")
    embedder = LegalEmbedder()
    result = embedder.embed_chunks(chunks)
    print_success(f"Embeddings: {result.embeddings.shape} on {result.device}")
    
    print_step("🎯", f"Selecting top {config.top_n} relevant chunks...")
    selected = select_chunks(chunks, result.embeddings, top_n=config.top_n)
    total_words = sum(c.word_count for c in selected)
    print_success(f"Selected {len(selected)} chunks ({total_words:,} words)")
    
    return selected


def generate_summary(
    chunks: list, 
    config: SummaryConfig
) -> str:
    """Generate summary using configured model.
    
    Args:
        chunks: Selected chunks
        config: Summary configuration
        
    Returns:
        Generated summary text
    """
    print_step("🤖", f"Loading {config.name} summarizer...")
    print(f"   Model: {config.model}")
    summarizer = LegalSummarizer(model_name=config.model)
    
    print_step("📝", f"Generating {config.name} summary...")
    summary = summarizer.hierarchical_summary(
        chunks,
        final_max_length=config.max_length,
        final_min_length=config.min_length
    )
    
    return summary


# =============================================================================
# OUTPUT & STATS
# =============================================================================

def display_summary(summary: str, config: SummaryConfig) -> None:
    """Display the generated summary."""
    print_header(f"{config.emoji} {config.name.upper()} SUMMARY")
    print()
    print(summary)
    print()
    print('=' * 70)


def display_stats(
    summary: str,
    config: SummaryConfig,
    selected_count: int,
    total_chunks: int,
    source_words: int,
    source_chars: int
) -> None:
    """Display summary statistics."""
    word_count = len(summary.split())
    compression = (len(summary) / source_chars) * 100 if source_chars else 0
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Type: {config.name}")
    print(f"   Model: {config.model}")
    print(f"   Length: {word_count} words ({len(summary)} characters)")
    print(f"   Source: {selected_count}/{total_chunks} chunks ({source_words:,} words)")
    print(f"   Compression: {source_chars:,} → {len(summary):,} chars ({compression:.1f}%)")
    print()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(pdf_path: Path, config: SummaryConfig) -> str:
    """Run the full summarization pipeline.
    
    Args:
        pdf_path: Path to PDF file
        config: Summary configuration
        
    Returns:
        Generated summary
    """
    print_header("Processing PDF")
    
    # Step 1: Load & preprocess
    processed, text = load_document(pdf_path)
    
    # Step 2: Chunk
    chunks = create_chunks(text, config)
    
    # Step 3: Embed & retrieve
    selected = embed_and_retrieve(chunks, config)
    
    # Step 4: Generate summary
    summary = generate_summary(selected, config)
    
    # Step 5: Display results
    display_summary(summary, config)
    
    # Step 6: Show stats
    source_words = sum(c.word_count for c in selected)
    display_stats(
        summary=summary,
        config=config,
        selected_count=len(selected),
        total_chunks=len(chunks),
        source_words=source_words,
        source_chars=len(text)
    )
    
    return summary


def main() -> None:
    """Main entry point for interactive summarization."""
    # Step 1: Select summary type
    config = select_summary_type()
    print(f"\n✓ Selected: {config.emoji} {config.name} Summary")
    print(f"  {config.description}")
    
    # Step 2: Get PDF path
    pdf_path = get_pdf_path()
    
    # Step 3: Run pipeline
    run_pipeline(pdf_path, config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise