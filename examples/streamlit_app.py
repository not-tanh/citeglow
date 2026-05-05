"""Interactive Streamlit demo for CiteGlow.

Run from the repository root with:

    streamlit run examples/streamlit_app.py
"""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st

from citeglow import (
    DEFAULT_SPAN_EXPANSION_REGEX,
    DEFAULT_STOP_WORDS,
    HighlightOptions,
    find_answer_highlights,
)


SAMPLES = {
    "Refund policy": {
        "answer": "The refund window is 30 days for unused items.",
        "source": "\n".join(
            [
                "Shipping is calculated at checkout.",
                "Unused items can be returned within 30 days of delivery for a refund.",
                "Opened software licenses are not refundable.",
            ]
        ),
    },
    "Technical support log": {
        "answer": "SKU123 shipped today, but ERR42 still needs review.",
        "source": "\n".join(
            [
                "09:00 INFO SKU123 packed in warehouse A.",
                "09:15 INFO SKU123 ships today by ground service.",
                "09:20 WARNING ERR42 retry count exceeded for the billing worker.",
                "09:25 INFO Case owner notified.",
            ]
        ),
    },
    "Japanese return policy": {
        "answer": "未使用の商品は到着後14日以内なら返品できます。",
        "source": "\n".join(
            [
                "送料は注文内容によって異なります。",
                "未使用の商品は到着後14日以内であれば返品できます。",
                "開封済みのソフトウェアは返金対象外です。",
            ]
        ),
        "tokenizer": "char",
    },
}


def main() -> None:
    st.set_page_config(page_title="CiteGlow demo", layout="wide")
    st.title("CiteGlow citation highlighter")
    st.caption("Tune deterministic citation spans and inspect the returned offsets.")

    sample_name = st.sidebar.selectbox("Example", tuple(SAMPLES))
    sample = SAMPLES[sample_name]

    with st.sidebar:
        st.header("Options")
        keep_longest_only = st.checkbox("Keep longest span only", value=True)
        expand_spans = st.checkbox("Expand spans for display", value=True)
        tokenizer = st.selectbox(
            "Tokenizer",
            ("unicode_word", "char"),
            index=1 if sample.get("tokenizer") == "char" else 0,
        )
        neighborhood_tokens = st.slider("Neighbor tokens", 0, 20, 6)
        min_span_words = st.slider("Minimum span words", 1, 8, 2)
        min_vocab_token_chars = st.slider("Minimum vocabulary token chars", 1, 8, 2)
        regex_default = sample.get(
            "span_expansion_regex",
            DEFAULT_SPAN_EXPANSION_REGEX,
        )
        span_expansion_regex = st.text_input(
            "Expansion regex",
            value=regex_default,
            disabled=not expand_spans,
        )
        stop_word_mode = st.radio(
            "Stop words",
            ("Default", "None", "Custom"),
            horizontal=True,
        )
        custom_stop_words = st.text_input(
            "Custom or additional stop words",
            value="",
            help="Comma-separated words. With Default, these extend the built-in list.",
        )

    answer = st.text_area("Answer", value=sample["answer"], height=120)
    source = st.text_area("Source chunk", value=sample["source"], height=220)

    stop_words = resolve_stop_words(stop_word_mode, custom_stop_words)
    try:
        options = HighlightOptions(
            neighborhood_tokens=neighborhood_tokens,
            min_span_words=min_span_words,
            min_vocab_token_chars=min_vocab_token_chars,
            stop_words=stop_words,
            tokenizer=tokenizer,
            expand_spans=expand_spans,
            span_expansion_regex=span_expansion_regex,
        )
        spans = find_answer_highlights(
            answer,
            source,
            keep_longest_only=keep_longest_only,
            options=options,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    highlighted_text = [source[start:end] for start, end in spans]
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Highlighted source")
        st.markdown(render_source(source, spans), unsafe_allow_html=True)

    with right:
        st.subheader("Offsets")
        st.json(
            {
                "spans": [{"start": start, "end": end} for start, end in spans],
                "highlighted_text": highlighted_text,
            }
        )
        st.subheader("Python")
        st.code(
            build_code_preview(
                keep_longest_only,
                options,
                stop_word_mode,
                custom_stop_words,
            ),
            language="python",
        )


def resolve_stop_words(mode: str, raw_words: str) -> Iterable[str]:
    custom = {
        word.strip().lower()
        for word in raw_words.split(",")
        if word.strip()
    }
    if mode == "None":
        return custom
    if mode == "Custom":
        return custom
    return DEFAULT_STOP_WORDS | custom


def render_source(source: str, spans: list[tuple[int, int]]) -> str:
    safe_spans = clamp_spans(spans, len(source))
    parts: list[str] = []
    cursor = 0
    for start, end in safe_spans:
        parts.append(html.escape(source[cursor:start]))
        parts.append(f"<mark>{html.escape(source[start:end])}</mark>")
        cursor = end
    parts.append(html.escape(source[cursor:]))
    body = "".join(parts) or "<span class='placeholder'>No source text.</span>"
    return f"""
<style>
.citeglow-source {{
    border: 1px solid #d5d8df;
    border-radius: 8px;
    background: #fbfbfd;
    color: #1f2328;
    line-height: 1.65;
    padding: 1rem;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    min-height: 220px;
}}
.citeglow-source mark {{
    background: #fff0a8;
    border-bottom: 2px solid #d49b00;
    border-radius: 3px;
    color: inherit;
    padding: 0.05rem 0.12rem;
}}
.citeglow-source .placeholder {{
    color: #6b7280;
}}
</style>
<div class="citeglow-source">{body}</div>
"""


def clamp_spans(spans: list[tuple[int, int]], source_length: int) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    previous_end = 0
    for start, end in sorted(spans):
        bounded_start = max(0, min(start, source_length))
        bounded_end = max(0, min(end, source_length))
        if bounded_start < previous_end:
            bounded_start = previous_end
        if bounded_start < bounded_end:
            normalized.append((bounded_start, bounded_end))
            previous_end = bounded_end
    return normalized


def build_code_preview(
    keep_longest_only: bool,
    options: HighlightOptions,
    stop_word_mode: str,
    raw_stop_words: str,
) -> str:
    import_line = "from citeglow import HighlightOptions, find_answer_highlights"
    stop_words_line = format_stop_words_line(stop_word_mode, raw_stop_words)
    if "DEFAULT_STOP_WORDS" in stop_words_line:
        import_line = (
            "from citeglow import DEFAULT_STOP_WORDS, HighlightOptions, "
            "find_answer_highlights"
        )

    return f"""{import_line}

options = HighlightOptions(
    neighborhood_tokens={options.neighborhood_tokens},
    min_span_words={options.min_span_words},
    min_vocab_token_chars={options.min_vocab_token_chars},
{stop_words_line}    expand_spans={options.expand_spans},
    tokenizer={options.tokenizer!r},
    span_expansion_regex={options.span_expansion_regex!r},
)

spans = find_answer_highlights(
    answer,
    source,
    keep_longest_only={keep_longest_only},
    options=options,
)
highlighted_text = [source[start:end] for start, end in spans]"""


def format_stop_words_line(mode: str, raw_words: str) -> str:
    custom = sorted(
        {
            word.strip().lower()
            for word in raw_words.split(",")
            if word.strip()
        }
    )
    custom_repr = "{" + ", ".join(repr(word) for word in custom) + "}"
    if mode == "Default" and not custom:
        return ""
    if mode == "Default":
        return f"    stop_words=DEFAULT_STOP_WORDS | {custom_repr},\n"
    if custom:
        return f"    stop_words={custom_repr},\n"
    return "    stop_words=set(),\n"


if __name__ == "__main__":
    main()
