"""
Pydantic response models for the Legal PDF Summarizer API.
"""

from pydantic import BaseModel, Field
from typing import List


class SummaryLevelInfo(BaseModel):
    """Describes one available summary level."""
    value: int = Field(..., description="Numeric choice value (1-4)")
    name: str = Field(..., description="Human-readable name")
    emoji: str = Field(..., description="Display emoji")
    description: str = Field(..., description="What the user can expect")


class SummaryResponse(BaseModel):
    """Returned after successful summarization."""
    summary: str = Field(..., description="The generated summary text")
    word_count: int = Field(..., description="Word count of the summary")
    level_name: str = Field(..., description="Summary level used")
    level_emoji: str = Field(..., description="Emoji for the level")
    filename: str = Field(..., description="Original uploaded filename")


class ErrorResponse(BaseModel):
    """Returned on error."""
    detail: str
