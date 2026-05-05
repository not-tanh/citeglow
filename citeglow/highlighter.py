"""LCS-style citation highlighter.

Locates spans in a chunk's text that overlap with the answer using
``difflib.SequenceMatcher``'s contiguous matching blocks. Pure
post-processing: no LLM calls, no extra retrieval, no change to the
answer prompt or the structured payload.

Algorithm:

1. Tokenize both strings into the configured token stream.
2. Run ``SequenceMatcher`` over the lowercased token streams and take its
   matching blocks (longest non-overlapping contiguous matches).
3. Merge adjacent matches separated by ≤ ``MERGE_GAP_TOKENS`` tokens — but
   never across sentence breaks.
4. Drop spans that are only stop words or a single short token.
5. Apply ``MAX_HIGHLIGHT_RATIO`` safety valve to avoid lighting up most
   of the chunk.

Shared tokenizer, stop-word set, merge logic, meaningful-run filter, and
safety valve live in ``highlighter_common``.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from collections.abc import Iterable

from .highlighter_common import (
    DEFAULT_STOP_WORDS,
    MAX_HIGHLIGHT_RATIO,
    MERGE_GAP_TOKENS,
    MIN_RUN_TOKENS,
    MIN_SINGLE_TOKEN_CHARS,
    TokenizerMode,
    enforce_max_highlight_ratio,
    is_meaningful_run,
    merge_close_runs,
    normalize_stop_words,
    tokenize,
)


def find_answer_highlights(
    answer: str,
    chunk_text: str,
    *,
    stop_words: Iterable[str] = DEFAULT_STOP_WORDS,
    merge_gap_tokens: int = MERGE_GAP_TOKENS,
    min_run_tokens: int = MIN_RUN_TOKENS,
    min_single_token_chars: int = MIN_SINGLE_TOKEN_CHARS,
    max_highlight_ratio: float = MAX_HIGHLIGHT_RATIO,
    tokenizer: TokenizerMode = "unicode_word",
) -> list[tuple[int, int]]:
    """Return char offsets in ``chunk_text`` whose tokens overlap with ``answer``.

    Each returned span is a half-open ``(start, end)`` pair into
    ``chunk_text``. Spans are sorted by start position and never overlap.
    Returns an empty list when no meaningful overlap is found.
    """

    if not answer or not chunk_text:
        return []

    answer_tokens, _ = tokenize(answer, tokenizer)
    chunk_tokens, chunk_spans = tokenize(chunk_text, tokenizer)
    if not answer_tokens or not chunk_tokens:
        return []

    matcher = SequenceMatcher(a=answer_tokens, b=chunk_tokens, autojunk=False)
    raw_runs = [
        (block.b, block.b + block.size)
        for block in matcher.get_matching_blocks()
        if block.size > 0
    ]
    if not raw_runs:
        return []

    normalized_stop_words = normalize_stop_words(stop_words)
    merged_runs = merge_close_runs(
        raw_runs,
        chunk_text,
        chunk_spans,
        gap_tokens=merge_gap_tokens,
    )
    surviving = [
        (start, end)
        for start, end in merged_runs
        if is_meaningful_run(
            chunk_tokens[start:end],
            stop_words=normalized_stop_words,
            min_run_tokens=min_run_tokens,
            min_single_token_chars=min_single_token_chars,
        )
    ]
    if not surviving:
        return []

    char_spans = [
        (chunk_spans[start][0], chunk_spans[end - 1][1])
        for start, end in surviving
    ]
    return enforce_max_highlight_ratio(
        char_spans,
        len(chunk_text),
        max_highlight_ratio=max_highlight_ratio,
    )
