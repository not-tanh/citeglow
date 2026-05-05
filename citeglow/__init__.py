"""Public API for CiteGlow."""

from __future__ import annotations

from .highlighter_common import DEFAULT_STOP_WORDS, TokenizerMode
from .highlighter_lcs_neighbor import (
    DEFAULT_SPAN_EXPANSION_REGEX,
    HighlightOptions,
    find_answer_highlights,
)

__all__ = [
    "DEFAULT_STOP_WORDS",
    "DEFAULT_SPAN_EXPANSION_REGEX",
    "HighlightOptions",
    "TokenizerMode",
    "find_answer_highlights",
]
