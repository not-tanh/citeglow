"""LCS-anchored BoW citation highlighter.

LCS produces high-precision phrase matches but misses words the answer
reordered or paraphrased away from contiguous form. This highlighter
treats LCS spans as *anchors*: BoW matches in the chunk fire only when
they are within ``neighborhood_tokens`` tokens of an LCS span. Surviving
BoW matches absorb into the nearest LCS span via ``merge_close_runs``
called with the same neighborhood as its gap parameter — that's how a
nearby BoW match becomes part of one larger highlighted phrase rather
than its own isolated single-word span.

After the merge step a final filter drops spans whose word count is
below ``min_span_words``, sweeping up single-word debris (LCS's or
BoW's) that doesn't carry enough context to be useful as a citation
highlight.

The very last step can expand each surviving char span outward to a
regex-defined display unit. The default expansion unit is one rendered
line, but callers can disable expansion or provide their own regex for
paragraphs, tags, bullets, or other chunk structure. After expansion
any newly overlapping spans are merged, and the safety valve runs on
the final coverage.

When LCS finds nothing in the chunk we return an empty list rather than
firing BoW with no anchor — silence is more honest than guessing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Set
from dataclasses import dataclass
from typing import Optional, Pattern, Union

from .highlighter import find_answer_highlights as find_highlights_lcs
from .highlighter_common import (
    DEFAULT_STOP_WORDS,
    MAX_HIGHLIGHT_RATIO,
    MERGE_GAP_TOKENS,
    MIN_RUN_TOKENS,
    MIN_SINGLE_TOKEN_CHARS,
    TokenizerMode,
    enforce_max_highlight_ratio,
    merge_close_runs,
    normalize_stop_words,
    tokenize,
)

# How many tokens away from an LCS span a BoW match may sit and still
# count as evidence. Doubles as the merge gap so a qualifying BoW match
# is naturally absorbed into the LCS span it neighbors.
NEIGHBORHOOD_TOKENS = 6

# Final spans with fewer than this many word tokens are dropped. 2 keeps
# two-word phrases like "phone number" while pruning lone single-word
# highlights. Bump to 1 if your knowledge base has lots of single-word
# evidence (product names, error codes); raise to 3 for stricter output.
MIN_SPAN_WORDS = 2

# Vocabulary floor: ignore one-character tokens (mostly stop-word-like).
MIN_VOCAB_TOKEN_CHARS = 2

# By default, expand highlights to the containing rendered line. Users
# can replace this with a regex that matches their own document units:
# paragraphs, bullet items, XML-ish tags, markdown sections, etc.
DEFAULT_SPAN_EXPANSION_REGEX = r"[^\r\n]+"
SpanExpansionRegex = Union[str, Pattern[str]]


@dataclass(frozen=True)
class HighlightOptions:
    """Configuration for LCS-anchored BoW highlighting.

    Defaults match the original CiteGlow behavior. Pass an instance to
    ``find_answer_highlights(..., options=...)`` when you want reusable
    tuning, or pass keyword overrides for a single call.
    """

    neighborhood_tokens: int = NEIGHBORHOOD_TOKENS
    min_span_words: int = MIN_SPAN_WORDS
    min_vocab_token_chars: int = MIN_VOCAB_TOKEN_CHARS
    stop_words: Iterable[str] = DEFAULT_STOP_WORDS
    lcs_merge_gap_tokens: int = MERGE_GAP_TOKENS
    lcs_min_run_tokens: int = MIN_RUN_TOKENS
    lcs_min_single_token_chars: int = MIN_SINGLE_TOKEN_CHARS
    max_highlight_ratio: float = MAX_HIGHLIGHT_RATIO
    tokenizer: TokenizerMode = "unicode_word"
    expand_spans: bool = True
    span_expansion_regex: SpanExpansionRegex = DEFAULT_SPAN_EXPANSION_REGEX


def find_answer_highlights(
    answer: str,
    chunk_text: str,
    *,
    keep_longest_only: bool = True,
    options: Optional[HighlightOptions] = None,
    neighborhood_tokens: Optional[int] = None,
    min_span_words: Optional[int] = None,
    min_vocab_token_chars: Optional[int] = None,
    stop_words: Optional[Iterable[str]] = None,
    tokenizer: Optional[TokenizerMode] = None,
    expand_spans: Optional[bool] = None,
    span_expansion_regex: Optional[SpanExpansionRegex] = None,
) -> list[tuple[int, int]]:
    """Return LCS spans extended by BoW matches that lie within the LCS neighborhood.

    Same return contract as the other highlighters: half-open
    ``(start, end)`` char offsets into ``chunk_text``, sorted by start
    position and never overlapping. Returns an empty list when LCS finds
    nothing or when every merged span is below the word-count floor.

    When ``keep_longest_only`` is True, after the meaningful-run filter
    runs, only the single longest char span survives. The selection
    happens BEFORE span expansion so "longest" reflects how much
    matched evidence the span actually carries, not how long the line
    it happens to sit on is. Useful when each cited source should
    surface exactly one supporting passage rather than a list.

    Pass ``HighlightOptions`` for reusable tuning, or use the direct
    keyword overrides for common per-call changes:
    ``neighborhood_tokens``, ``min_span_words``,
    ``min_vocab_token_chars``, ``stop_words``, ``tokenizer``,
    ``expand_spans``, and ``span_expansion_regex``.
    """

    if not answer or not chunk_text:
        return []

    resolved = _resolve_options(
        options=options,
        neighborhood_tokens=neighborhood_tokens,
        min_span_words=min_span_words,
        min_vocab_token_chars=min_vocab_token_chars,
        stop_words=stop_words,
        tokenizer=tokenizer,
        expand_spans=expand_spans,
        span_expansion_regex=span_expansion_regex,
    )

    lcs_char_spans = find_highlights_lcs(
        answer,
        chunk_text,
        stop_words=resolved.stop_words,
        merge_gap_tokens=resolved.lcs_merge_gap_tokens,
        min_run_tokens=resolved.lcs_min_run_tokens,
        min_single_token_chars=resolved.lcs_min_single_token_chars,
        max_highlight_ratio=resolved.max_highlight_ratio,
        tokenizer=resolved.tokenizer,
    )
    if not lcs_char_spans:
        return []

    chunk_tokens, chunk_token_spans = tokenize(chunk_text, resolved.tokenizer)
    if not chunk_tokens:
        return []

    lcs_token_spans = _char_spans_to_token_spans(
        lcs_char_spans, chunk_token_spans
    )
    if not lcs_token_spans:
        # LCS produced spans that don't align to token boundaries (should
        # not happen given how LCS builds its char spans), so we have no
        # anchors to expand from.
        return []

    answer_vocab = _build_bow_vocabulary(
        answer,
        stop_words=resolved.stop_words,
        min_vocab_token_chars=resolved.min_vocab_token_chars,
        tokenizer=resolved.tokenizer,
    )
    raw_bow_runs = [
        (index, index + 1)
        for index, token in enumerate(chunk_tokens)
        if token in answer_vocab
    ]
    bow_runs = merge_close_runs(
        raw_bow_runs,
        chunk_text,
        chunk_token_spans,
        gap_tokens=resolved.neighborhood_tokens,
    )
    near_lcs_runs = [
        (start, end)
        for start, end in bow_runs
        if any(
            start < anchor_end + resolved.neighborhood_tokens
            and end > anchor_start - resolved.neighborhood_tokens
            for anchor_start, anchor_end in lcs_token_spans
        )
    ]

    combined_runs = list(lcs_token_spans) + near_lcs_runs
    merged_runs = merge_close_runs(
        combined_runs,
        chunk_text,
        chunk_token_spans,
        gap_tokens=resolved.neighborhood_tokens,
    )

    surviving = [
        (start, end)
        for start, end in merged_runs
        if (end - start) >= resolved.min_span_words
    ]
    if not surviving:
        return []

    char_spans = [
        (chunk_token_spans[start][0], chunk_token_spans[end - 1][1])
        for start, end in surviving
    ]
    if keep_longest_only and char_spans:
        # Pick the span that carries the most matched evidence (longest
        # token-aligned char range) BEFORE span expansion bloats every
        # short span out to a full line — otherwise "longest" would be
        # decided by surrounding line length instead of by match content.
        longest = max(char_spans, key=lambda span: span[1] - span[0])
        char_spans = [longest]
    if resolved.expand_spans:
        char_spans = _expand_to_regex_bounds(
            char_spans,
            chunk_text,
            resolved.span_expansion_regex,
        )
    return enforce_max_highlight_ratio(
        char_spans,
        len(chunk_text),
        max_highlight_ratio=resolved.max_highlight_ratio,
    )


def _resolve_options(
    *,
    options: Optional[HighlightOptions],
    neighborhood_tokens: Optional[int],
    min_span_words: Optional[int],
    min_vocab_token_chars: Optional[int],
    stop_words: Optional[Iterable[str]],
    tokenizer: Optional[TokenizerMode],
    expand_spans: Optional[bool],
    span_expansion_regex: Optional[SpanExpansionRegex],
) -> HighlightOptions:
    """Merge reusable options with per-call keyword overrides."""

    base = options or HighlightOptions()
    resolved = HighlightOptions(
        neighborhood_tokens=(
            base.neighborhood_tokens
            if neighborhood_tokens is None
            else neighborhood_tokens
        ),
        min_span_words=base.min_span_words if min_span_words is None else min_span_words,
        min_vocab_token_chars=(
            base.min_vocab_token_chars
            if min_vocab_token_chars is None
            else min_vocab_token_chars
        ),
        stop_words=(
            normalize_stop_words(base.stop_words)
            if stop_words is None
            else normalize_stop_words(stop_words)
        ),
        lcs_merge_gap_tokens=base.lcs_merge_gap_tokens,
        lcs_min_run_tokens=base.lcs_min_run_tokens,
        lcs_min_single_token_chars=base.lcs_min_single_token_chars,
        max_highlight_ratio=base.max_highlight_ratio,
        tokenizer=base.tokenizer if tokenizer is None else tokenizer,
        expand_spans=base.expand_spans if expand_spans is None else expand_spans,
        span_expansion_regex=(
            base.span_expansion_regex
            if span_expansion_regex is None
            else span_expansion_regex
        ),
    )
    _validate_options(resolved)
    return resolved


def _validate_options(options: HighlightOptions) -> None:
    """Fail early on values that make matching behavior undefined."""

    if options.neighborhood_tokens < 0:
        raise ValueError("neighborhood_tokens must be >= 0")
    if options.min_span_words < 1:
        raise ValueError("min_span_words must be >= 1")
    if options.min_vocab_token_chars < 1:
        raise ValueError("min_vocab_token_chars must be >= 1")
    if options.lcs_merge_gap_tokens < 0:
        raise ValueError("lcs_merge_gap_tokens must be >= 0")
    if options.lcs_min_run_tokens < 1:
        raise ValueError("lcs_min_run_tokens must be >= 1")
    if options.lcs_min_single_token_chars < 1:
        raise ValueError("lcs_min_single_token_chars must be >= 1")
    if options.max_highlight_ratio <= 0:
        raise ValueError("max_highlight_ratio must be > 0")
    tokenize("", options.tokenizer)
    _compile_span_expansion_regex(options.span_expansion_regex)


def _build_bow_vocabulary(
    answer: str,
    *,
    stop_words: Set[str],
    min_vocab_token_chars: int,
    tokenizer: TokenizerMode,
) -> frozenset[str]:
    """Return the answer's content-word vocabulary for BoW matching."""

    answer_tokens, _ = tokenize(answer, tokenizer)
    return frozenset(
        token
        for token in answer_tokens
        if token not in stop_words and len(token) >= min_vocab_token_chars
    )


def _char_spans_to_token_spans(
    char_spans: list[tuple[int, int]],
    token_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert char-span pairs back to token-index pairs.

    LCS builds its char spans by reading token boundaries, so each char
    span starts at some ``token_spans[i][0]`` and ends at
    ``token_spans[j-1][1]``. We recover those indices by linear scan;
    chunks are short enough that this is fine.
    """

    token_start_to_index = {ts: i for i, (ts, _) in enumerate(token_spans)}
    token_end_to_index_plus_one = {
        te: i + 1 for i, (_, te) in enumerate(token_spans)
    }
    result: list[tuple[int, int]] = []
    for char_start, char_end in char_spans:
        token_start = token_start_to_index.get(char_start)
        token_end = token_end_to_index_plus_one.get(char_end)
        if token_start is None or token_end is None or token_start >= token_end:
            continue
        result.append((token_start, token_end))
    return result


def _within_neighborhood(
    token_index: int,
    anchor_spans: list[tuple[int, int]],
    radius: int,
) -> bool:
    """True when ``token_index`` is at most ``radius`` tokens from any anchor."""

    for anchor_start, anchor_end in anchor_spans:
        if anchor_start - radius <= token_index < anchor_end + radius:
            return True
    return False


def _compile_span_expansion_regex(
    span_expansion_regex: SpanExpansionRegex,
) -> Pattern[str]:
    """Return a compiled expansion regex or raise a useful error."""

    if isinstance(span_expansion_regex, str):
        if not span_expansion_regex:
            raise ValueError("span_expansion_regex must not be empty")
        try:
            return re.compile(span_expansion_regex)
        except re.error as exc:
            raise ValueError("span_expansion_regex is invalid") from exc
    return span_expansion_regex


def _expand_to_regex_bounds(
    spans: list[tuple[int, int]],
    chunk_text: str,
    span_expansion_regex: SpanExpansionRegex,
) -> list[tuple[int, int]]:
    """Expand each span to regex match bounds, then merge overlaps.

    The regex defines expansion units. The default unit is one rendered
    line. If a span crosses multiple units, the start expands to the unit
    containing the span start and the end expands to the unit containing
    the span end.

    After expansion two spans in the same unit merge into one; spans on
    adjacent units separated only by newlines (or ``\\r\\n``) also merge
    so consecutive bullet items render as one continuous block.
    """

    if not spans:
        return spans

    expansion_re = _compile_span_expansion_regex(span_expansion_regex)
    units = [
        (match.start(), match.end())
        for match in expansion_re.finditer(chunk_text)
        if match.start() < match.end()
    ]
    if not units:
        return spans

    expanded: list[tuple[int, int]] = []
    for start, end in spans:
        expanded_start = _find_expansion_start(start, units)
        expanded_end = _find_expansion_end(start, end, units)
        if expanded_start is None:
            expanded_start = start
        if expanded_end is None:
            expanded_end = end
        if expanded_start < expanded_end:
            expanded.append((expanded_start, expanded_end))

    if not expanded:
        return []

    expanded.sort()
    merged: list[tuple[int, int]] = [expanded[0]]
    for start, end in expanded[1:]:
        previous_start, previous_end = merged[-1]
        # Merge if ranges overlap or are separated only by a single
        # newline (or \r\n) — this collapses adjacent bulleted lines that
        # both got highlighted into one continuous block.
        gap_text = chunk_text[previous_end:start]
        if start <= previous_end or gap_text.strip("\r\n") == "":
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_expansion_start(
    span_start: int,
    units: list[tuple[int, int]],
) -> Optional[int]:
    """Return the start of the expansion unit containing ``span_start``."""

    for unit_start, unit_end in units:
        if unit_start <= span_start < unit_end:
            return unit_start
    return None


def _find_expansion_end(
    span_start: int,
    span_end: int,
    units: list[tuple[int, int]],
) -> Optional[int]:
    """Return the end of the expansion unit containing the span's final char."""

    probe = max(span_start, span_end - 1)
    for unit_start, unit_end in units:
        if unit_start <= probe < unit_end:
            return unit_end
    return None
