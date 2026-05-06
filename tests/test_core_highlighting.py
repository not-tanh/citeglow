import importlib.util
import sys
import types
from pathlib import Path

import pytest

from citeglow import find_answer_highlights
from citeglow.highlighter_common import (
    enforce_max_highlight_ratio,
    merge_close_runs,
    tokenize,
)

sys.modules.setdefault("streamlit", types.SimpleNamespace())
STREAMLIT_APP_PATH = Path(__file__).parents[1] / "examples" / "streamlit_app.py"
STREAMLIT_APP_SPEC = importlib.util.spec_from_file_location(
    "streamlit_app",
    STREAMLIT_APP_PATH,
)
assert STREAMLIT_APP_SPEC is not None
streamlit_app = importlib.util.module_from_spec(STREAMLIT_APP_SPEC)
assert STREAMLIT_APP_SPEC.loader is not None
STREAMLIT_APP_SPEC.loader.exec_module(streamlit_app)


def test_public_highlighter_expands_to_supporting_line():
    """The default public API returns the display line that supports the answer."""

    source = "\n".join(
        [
            "Shipping is calculated at checkout.",
            "Unused items can be returned within 30 days of delivery for a refund.",
            "Opened software licenses are not refundable.",
        ]
    )
    answer = "The refund window is 30 days for unused items."

    spans = find_answer_highlights(answer, source)

    assert [source[start:end] for start, end in spans] == [
        "Unused items can be returned within 30 days of delivery for a refund."
    ]


def test_exact_spans_can_be_returned_without_line_expansion():
    """Disabling expansion keeps the offsets tight around matched evidence."""

    source = "\n".join(
        [
            "Shipping is calculated at checkout.",
            "Unused items can be returned within 30 days of delivery for a refund.",
            "Opened software licenses are not refundable.",
        ]
    )
    answer = "The refund window is 30 days for unused items."

    spans = find_answer_highlights(answer, source, expand_spans=False)

    assert [source[start:end] for start, end in spans] == [
        "Unused items can be returned within 30 days of delivery for a refund"
    ]


def test_neighbor_terms_are_absorbed_into_lcs_anchor():
    """Nearby answer vocabulary extends an LCS phrase but distant matches do not."""

    source = "Policy ABC-123 requires manager approval before reimbursement."
    answer = "Manager approval is required for reimbursement under ABC-123."

    neighbor_spans = find_answer_highlights(answer, source, expand_spans=False)
    strict_spans = find_answer_highlights(
        answer,
        source,
        expand_spans=False,
        neighborhood_tokens=0,
    )

    assert [source[start:end] for start, end in neighbor_spans] == [
        "ABC-123 requires manager approval before reimbursement"
    ]
    assert [source[start:end] for start, end in strict_spans] == [
        "manager approval before reimbursement"
    ]


def test_short_single_token_matches_do_not_create_anchor():
    """Short reordered tokens are ignored when they cannot form an LCS anchor."""

    spans = find_answer_highlights(
        "EU US",
        "US and EU markets opened lower.",
        stop_words=set(),
        min_span_words=1,
        expand_spans=False,
    )

    assert spans == []


def test_keep_longest_only_false_preserves_multiple_tight_spans():
    """Callers can keep every supporting span when expansion is disabled."""

    source = "\n".join(
        [
            "Alpha beta evidence is here.",
            "Noise line only.",
            "Gamma delta evidence is there.",
        ]
    )
    answer = "alpha beta and gamma delta"

    spans = find_answer_highlights(
        answer,
        source,
        keep_longest_only=False,
        expand_spans=False,
    )

    assert [source[start:end] for start, end in spans] == [
        "Alpha beta",
        "Gamma delta",
    ]


def test_streamlit_renderer_splits_multiline_spans_by_visible_line():
    """The demo renderer marks each visible line fragment in a multi-line span."""

    source = "alpha\nbeta\ngamma"
    html = streamlit_app.render_highlighted_lines(source, [(2, 13)])

    assert html == (
        '<div class="citeglow-source-line">al<mark>pha</mark></div>'
        '<div class="citeglow-source-line"><mark>beta</mark></div>'
        '<div class="citeglow-source-line"><mark>ga</mark>mma</div>'
    )


def test_tokenize_supports_unicode_words_and_non_whitespace_chars():
    """Core tokenization preserves Unicode word spans and char-mode offsets."""

    word_tokens, word_spans = tokenize("Giá trị A-1")
    char_tokens, char_spans = tokenize("A B", tokenizer="char")

    assert word_tokens == ["giá", "trị", "a", "1"]
    assert word_spans == [(0, 3), (4, 7), (8, 9), (10, 11)]
    assert char_tokens == ["a", "b"]
    assert char_spans == [(0, 1), (2, 3)]


def test_tokenize_rejects_unknown_mode():
    """Unknown tokenizer modes fail early with a clear ValueError."""

    with pytest.raises(ValueError, match="tokenizer must be"):
        tokenize("text", tokenizer="unknown")


def test_merge_close_runs_does_not_cross_sentence_break():
    """Close token runs merge across commas but not across sentence boundaries."""

    chunk = "alpha, beta. gamma delta"
    _, spans = tokenize(chunk)

    merged = merge_close_runs([(0, 1), (1, 2), (2, 3)], chunk, spans, gap_tokens=1)

    assert merged == [(0, 2), (2, 3)]


def test_max_highlight_ratio_collapses_to_longest_span():
    """The safety valve keeps only the longest span when coverage is too broad."""

    spans = enforce_max_highlight_ratio(
        [(0, 4), (10, 19)],
        chunk_length=20,
        max_highlight_ratio=0.5,
    )

    assert spans == [(10, 19)]
