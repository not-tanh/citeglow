"""Shared helpers used by the citation highlighter modules.

``highlighter`` (LCS) and ``highlighter_lcs_neighbor`` (LCS + neighbor BoW
— the production highlighter wired into ``answer_builder``) both need
the same configurable tokenizer, the same gap/sentence-break aware run
merger, the same meaningful-run filter, and the same final
``MAX_HIGHLIGHT_RATIO`` safety valve. Putting them here keeps each
algorithm-specific module focused on its own logic.

Names here intentionally have no leading underscore — they are package-
private (not re-exported from ``__init__``) but are imported by sibling
modules in this package, which is a poor fit for the "module-private"
convention.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Set
from typing import Literal

# A small, curated stop-word set covering the languages this product
# serves (Vietnamese and English). Spans whose only content is one of
# these tokens highlight nothing meaningful.
DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        # English
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "had", "has", "have", "he", "her", "his", "i", "if",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
        "she", "so", "that", "the", "their", "them", "they", "this",
        "to", "was", "we", "were", "will", "with", "you", "your",
        # Vietnamese (common function words)
        "là", "và", "của", "trong", "một", "các", "có", "không", "để",
        "được", "với", "cho", "từ", "đã", "sẽ", "này", "đó", "thì",
        "nhưng", "hoặc", "khi", "nếu", "vì", "bởi", "đến", "tại",
        "cũng", "hay", "rằng", "mà", "ở", "nên",
    }
)

# Backward-compatible alias for users who imported the constant directly.
STOP_WORDS = DEFAULT_STOP_WORDS

# Tunables. Defaults chosen so we surface confident matches without
# fragmenting natural phrases. Shared across highlighters so direct
# A/B comparison is meaningful.
MIN_RUN_TOKENS = 2            # need at least 2 content tokens in a surviving span
MIN_SINGLE_TOKEN_CHARS = 5    # OR a single content token of at least this length
MERGE_GAP_TOKENS = 2          # bridge runs separated by at most this many tokens
MAX_HIGHLIGHT_RATIO = 0.5     # collapse to longest span when highlights dominate

TokenizerMode = Literal["unicode_word", "char"]

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# Punctuation we treat as a sentence break. Two consecutive newlines also
# count, to handle bullet/paragraph boundaries in pre-rendered chunks.
SENTENCE_BREAK_RE = re.compile(r"[.!?…]|\n\s*\n")


def tokenize(
    text: str,
    tokenizer: TokenizerMode = "unicode_word",
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return lowercased tokens and their original char spans for ``text``."""

    if tokenizer == "char":
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []
        for index, char in enumerate(text):
            if not char.isspace():
                tokens.append(char.lower())
                spans.append((index, index + 1))
        return tokens, spans
    if tokenizer != "unicode_word":
        raise ValueError("tokenizer must be 'unicode_word' or 'char'")

    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in WORD_RE.finditer(text):
        tokens.append(match.group(0).lower())
        spans.append(match.span())
    return tokens, spans


def merge_close_runs(
    runs: list[tuple[int, int]],
    chunk_text: str,
    chunk_spans: list[tuple[int, int]],
    *,
    gap_tokens: int = MERGE_GAP_TOKENS,
) -> list[tuple[int, int]]:
    """Fuse token-index runs separated by at most ``gap_tokens`` tokens.

    Two runs are NOT merged when the chunk text between them contains a
    sentence break (``.``, ``!``, ``?``, ``…``, or a blank line) — that
    keeps a single highlight from spanning two unrelated sentences. The
    default gap matches ``MERGE_GAP_TOKENS``; algorithms that anchor BoW
    matches near LCS spans pass a larger value so nearby BoW matches get
    absorbed into the LCS run.
    """

    sorted_runs = sorted(runs)
    merged: list[tuple[int, int]] = []
    for start, end in sorted_runs:
        if merged:
            previous_start, previous_end = merged[-1]
            current_gap = start - previous_end
            if current_gap <= gap_tokens:
                # Check the source characters between adjacent or gapped
                # token runs; punctuation can sit between adjacent tokens
                # even when their token-index gap is zero. Overlapping
                # runs have no reliable in-between text, so they merge.
                blocked_by_sentence = False
                if current_gap >= 0:
                    gap_text = chunk_text[
                        chunk_spans[previous_end - 1][1]
                        : chunk_spans[start][0]
                    ]
                    if SENTENCE_BREAK_RE.search(gap_text):
                        blocked_by_sentence = True
                if not blocked_by_sentence:
                    merged[-1] = (previous_start, max(previous_end, end))
                    continue
        merged.append((start, end))
    return merged


def normalize_stop_words(stop_words: Iterable[str]) -> frozenset[str]:
    """Return a lowercased immutable stop-word set."""

    return frozenset(word.lower() for word in stop_words)


def is_meaningful_run(
    tokens: list[str],
    *,
    stop_words: Set[str] = DEFAULT_STOP_WORDS,
    min_run_tokens: int = MIN_RUN_TOKENS,
    min_single_token_chars: int = MIN_SINGLE_TOKEN_CHARS,
) -> bool:
    """True when a run carries enough non-stop content to be worth showing."""

    content = [token for token in tokens if token not in stop_words]
    if len(content) >= min_run_tokens:
        return True
    return any(len(token) >= min_single_token_chars for token in content)


def enforce_max_highlight_ratio(
    spans: list[tuple[int, int]],
    chunk_length: int,
    *,
    max_highlight_ratio: float = MAX_HIGHLIGHT_RATIO,
) -> list[tuple[int, int]]:
    """Collapse to the single longest span when highlights would dominate the chunk."""

    if chunk_length <= 0 or not spans:
        return spans
    total_highlighted = sum(end - start for start, end in spans)
    if total_highlighted / chunk_length <= max_highlight_ratio:
        return spans
    longest = max(spans, key=lambda span: span[1] - span[0])
    return [longest]
