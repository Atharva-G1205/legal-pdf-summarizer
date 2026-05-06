"""
Summarization Configuration Module
===================================

Centralized configuration for the legal PDF summarizer.
Easily modify summary levels, models, and parameters here.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict


class SummaryLevel(Enum):
    """Summary level types."""
    EXECUTIVE = 1
    DETAILED = 2
    TECHNICAL = 3
    EXTRACTIVE = 4


@dataclass(frozen=True)
class SummaryConfig:
    """Configuration for a specific summary level.
    
    Attributes:
        name: Human-readable name
        emoji: Display emoji
        model: Model key ("led" or "pegasus")
        backend: "local" or "online"
        top_n: Number of sentences to retrieve
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
        description: User-facing description
        chunk_words: Target words per chunk
        chunk_overlap: Overlap words between chunks
    """
    name: str
    emoji: str
    model: str
    top_n: int
    max_length: int
    min_length: int
    description: str
    backend: str = 'local'
    model_source: str = 'finetuned'   # 'finetuned' or 'huggingface'
    chunk_words: int = 400
    chunk_overlap: int = 50

    @property
    def target_words(self) -> int:
        """Approximate target word count for output."""
        # Rough estimate: ~0.75 words per token
        return int(self.max_length * 0.75)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Model keys correspond to LegalSummarizer.LOCAL_MODELS entries.
# These are locally stored fine-tuned models under <project>/models/.
MODELS = {
    'led': 'led',            # Legal-LED-base-16384  (lighter, 16K input)
    'pegasus': 'pegasus',    # Legal-Pegasus          (larger, higher quality)
}


# =============================================================================
# SUMMARY LEVEL PRESETS
# =============================================================================

SUMMARY_CONFIGS: Dict[SummaryLevel, SummaryConfig] = {
    SummaryLevel.EXECUTIVE: SummaryConfig(
        name='Executive',
        emoji='📋',
        model=MODELS['pegasus'],        # Pegasus: higher quality for concise summaries
        top_n=20,                       # ~700 words of ground truth
        max_length=350,
        min_length=150,
        description='Concise overview for decision-makers (~150 words)',
        chunk_words=350,
        chunk_overlap=40,
    ),

    SummaryLevel.DETAILED: SummaryConfig(
        name='Detailed',
        emoji='📄',
        model=MODELS['led'],            # LED: 16K context avoids hierarchical gibberish
        top_n=35,                       # richer ground truth for reasoning/order
        max_length=800,
        min_length=360,
        description='Comprehensive summary for lawyers (~400 words)',
        chunk_words=400,
        chunk_overlap=50,
    ),

    SummaryLevel.TECHNICAL: SummaryConfig(
        name='Technical',
        emoji='🔬',
        model=MODELS['led'],            # LED: 16K context avoids hierarchical gibberish
        top_n=45,                       # long-form analysis with ample context
        max_length=1020,
        min_length=500,
        description='In-depth analysis for researchers (~600 words)',
        chunk_words=500,
        chunk_overlap=75,
    ),

    SummaryLevel.EXTRACTIVE: SummaryConfig(
        name='Extractive',
        emoji='📌',
        model='none',                  # no abstractive model — ranked sentences only
        top_n=25,                       # top 25 sentences ≈ 400 words
        max_length=0,
        min_length=0,
        description='Top-ranked sentences, no generation (~400 words)',
    ),
}


def get_config(level: SummaryLevel) -> SummaryConfig:
    """Get configuration for a summary level.
    
    Args:
        level: SummaryLevel enum value
        
    Returns:
        Corresponding SummaryConfig
        
    Raises:
        KeyError: If level not found
    """
    return SUMMARY_CONFIGS[level]


def get_config_by_number(choice: int) -> SummaryConfig:
    """Get configuration by menu choice number (1-3).
    
    Args:
        choice: Integer 1, 2, or 3
        
    Returns:
        Corresponding SummaryConfig
        
    Raises:
        ValueError: If choice not in valid range
    """
    try:
        level = SummaryLevel(choice)
        return SUMMARY_CONFIGS[level]
    except ValueError:
        raise ValueError(f"Invalid choice: {choice}. Must be 1, 2, 3, or 4.")


# CLI display strings
MENU_OPTIONS = """
Choose summary type:

  1. 📋 EXECUTIVE   - Brief overview for decision-makers
                      (~150 words, key holdings only)

  2. 📄 DETAILED    - Comprehensive summary for lawyers
                      (~400 words, facts + reasoning + decision)

  3. 🔬 TECHNICAL   - In-depth analysis for researchers
                      (~600 words, full legal analysis)

  4. 📌 EXTRACTIVE  - Top-ranked sentences only, no generation
                      (~400 words, zero hallucination)
"""
