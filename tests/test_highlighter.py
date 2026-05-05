from __future__ import annotations

import pytest

from citeglow import (
    DEFAULT_SPAN_EXPANSION_REGEX,
    HighlightOptions,
    find_answer_highlights,
)


def test_neighbor_highlighter_expands_to_supporting_line() -> None:
    source = "\n".join(
        [
            "Shipping is calculated at checkout.",
            "Unused items can be returned within 30 days of delivery for a refund.",
            "Opened software licenses are not refundable.",
        ]
    )
    answer = "The refund window is 30 days for unused items."

    spans = find_answer_highlights(answer, source)

    assert len(spans) == 1
    assert source[spans[0][0] : spans[0][1]] == (
        "Unused items can be returned within 30 days of delivery for a refund."
    )


def test_neighbor_highlighter_returns_empty_without_lcs_anchor() -> None:
    source = "Refunds are available after manager approval."
    answer = "Customers may receive their money back."

    assert find_answer_highlights(answer, source) == []


def test_neighbor_highlighter_can_keep_multiple_spans() -> None:
    source = "\n".join(
        [
            "The warranty lasts two years.",
            "Shipping is free for replacement parts.",
        ]
    )
    answer = "The warranty lasts two years. Shipping is free for replacement parts."

    spans = find_answer_highlights(answer, source, keep_longest_only=False)

    assert [source[start:end] for start, end in spans] == [source]


def test_neighborhood_tokens_controls_nearby_bow_expansion() -> None:
    source = "alpha beta filler filler filler\nextra evidence"
    answer = "alpha beta extra"

    default_spans = find_answer_highlights(answer, source)
    narrow_spans = find_answer_highlights(answer, source, neighborhood_tokens=1)

    assert [source[start:end] for start, end in default_spans] == [source]
    assert [source[start:end] for start, end in narrow_spans] == [
        "alpha beta filler filler filler"
    ]


def test_min_span_words_can_allow_single_token_evidence() -> None:
    source = "SKU123 ships today."
    answer = "SKU123"

    assert find_answer_highlights(answer, source) == []

    spans = find_answer_highlights(answer, source, min_span_words=1)

    assert [source[start:end] for start, end in spans] == [source]


def test_min_vocab_token_chars_controls_short_bow_terms() -> None:
    source = "alpha beta filler\nz suffix"
    answer = "alpha beta z"
    options = HighlightOptions(lcs_merge_gap_tokens=0, lcs_min_single_token_chars=10)

    default_spans = find_answer_highlights(answer, source, options=options)
    tuned_spans = find_answer_highlights(
        answer,
        source,
        options=options,
        min_vocab_token_chars=1,
    )

    assert [source[start:end] for start, end in default_spans] == [
        "alpha beta filler"
    ]
    assert [source[start:end] for start, end in tuned_spans] == [source]


def test_stop_words_are_configurable() -> None:
    source = "The plan applies."
    answer = "The"
    options = HighlightOptions(
        min_span_words=1,
        stop_words=set(),
        lcs_min_single_token_chars=1,
    )

    assert find_answer_highlights(answer, source) == []

    spans = find_answer_highlights(answer, source, options=options)

    assert [source[start:end] for start, end in spans] == [source]


def test_invalid_options_raise_value_error() -> None:
    with pytest.raises(ValueError, match="neighborhood_tokens"):
        find_answer_highlights("answer", "answer", neighborhood_tokens=-1)


def test_span_expansion_can_be_disabled() -> None:
    source = "Unused items can be returned within 30 days of delivery for a refund."
    answer = "The refund window is 30 days for unused items."

    expanded = find_answer_highlights(answer, source)
    tight = find_answer_highlights(answer, source, expand_spans=False)

    assert [source[start:end] for start, end in expanded] == [source]
    assert [source[start:end] for start, end in tight] == [
        "Unused items can be returned within 30 days of delivery for a refund"
    ]


def test_span_expansion_regex_can_target_custom_blocks() -> None:
    source = "prefix <cite>alpha beta evidence</cite> suffix"
    answer = "alpha beta"

    spans = find_answer_highlights(
        answer,
        source,
        span_expansion_regex=r"<cite>.*?</cite>",
    )

    assert [source[start:end] for start, end in spans] == [
        "<cite>alpha beta evidence</cite>"
    ]


def test_default_span_expansion_regex_is_public() -> None:
    assert DEFAULT_SPAN_EXPANSION_REGEX == r"[^\r\n]+"


def test_invalid_span_expansion_regex_raises_value_error() -> None:
    with pytest.raises(ValueError, match="span_expansion_regex"):
        find_answer_highlights("answer", "answer", span_expansion_regex="")

    with pytest.raises(ValueError, match="span_expansion_regex"):
        find_answer_highlights("answer", "answer", span_expansion_regex="[")
