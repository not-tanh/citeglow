<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/not-tanh/citeglow/main/citeglow.png" alt="CiteGlow logo" width="300">
</div>

CiteGlow is a tiny Python library for highlighting the supporting passage inside a source document, given a grounded answer.

It is built for RAG systems that already retrieved the right source chunk and generated an answer from it. CiteGlow does **not** call an LLM, create embeddings, run a reranker, or make network requests. It uses deterministic token overlap: LCS phrase anchors plus nearby bag-of-words matches.

## Why use it?

- Highlight source text for citations, answer explanations, and audit views.
- Keep citation UI fast and cheap after answer generation.
- Avoid another model call when the answer is expected to be grounded in the retrieved source.
- Return plain Python character offsets so you can render highlights in any UI.

## Install

From a local checkout:

```bash
python -m pip install -e .
```

After publishing to PyPI:

```bash
python -m pip install citeglow
```

## Quick Start

```python
from citeglow import find_answer_highlights

answer = "The refund window is 30 days for unused items."
source = """
Shipping is calculated at checkout.
Unused items can be returned within 30 days of delivery for a refund.
Opened software licenses are not refundable.
""".strip()

spans = find_answer_highlights(answer, source)
highlighted_text = [source[start:end] for start, end in spans]

print(spans)
print(highlighted_text)
```

`find_answer_highlights` returns sorted, non-overlapping `(start, end)` character offsets into `source`. Offsets are half-open, so `source[start:end]` is the highlighted text.

## API

### `find_answer_highlights(answer, chunk_text, *, keep_longest_only=True, options=None, neighborhood_tokens=None, min_span_words=None, min_vocab_token_chars=None, stop_words=None, tokenizer=None, expand_spans=None, span_expansion_regex=None)`

Recommended highlighter. It:

1. Finds high-precision contiguous token matches with LCS.
2. Treats those LCS spans as anchors.
3. Adds nearby bag-of-words matches from the answer.
4. Drops spans that are too short to be useful.
5. Expands surviving spans to regex-defined display units for cleaner UI rendering.
6. Applies a safety valve so overly broad matches do not highlight most of the source.

Use `keep_longest_only=False` when your UI should display every supporting passage found inside a chunk instead of one best passage.

Tuning options can be passed directly for one call:

```python
spans = find_answer_highlights(
    answer,
    source,
    neighborhood_tokens=10,
    min_span_words=1,
    min_vocab_token_chars=1,
    stop_words={"the", "and", "of"},
    tokenizer="char",
    expand_spans=False,
)
```

Or reuse the same settings with `HighlightOptions`:

```python
from citeglow import DEFAULT_STOP_WORDS, HighlightOptions, find_answer_highlights

options = HighlightOptions(
    neighborhood_tokens=10,
    min_span_words=1,
    min_vocab_token_chars=1,
    stop_words=DEFAULT_STOP_WORDS | {"custom", "domain", "terms"},
    tokenizer="unicode_word",
    expand_spans=True,
    span_expansion_regex=r"[^\r\n]+",
)

spans = find_answer_highlights(answer, source, options=options)
```

Common knobs:

- `neighborhood_tokens`: how far a bag-of-words match may sit from an LCS anchor and still be merged into the highlight. Increase it for looser matching; decrease it for tighter highlights.
- `min_span_words`: minimum token length for final spans before span expansion. Use `1` for short identifiers, product names, error codes, or SKUs.
- `min_vocab_token_chars`: minimum token length for answer words used in bag-of-words expansion. Use `1` only when short tokens are meaningful in your domain.
- `stop_words`: words ignored as low-signal vocabulary. Passing this replaces the defaults for that call; extend `DEFAULT_STOP_WORDS` when you want to keep the built-in English and Vietnamese words.
- `tokenizer`: tokenization mode. Use `"unicode_word"` for the current Unicode word-token behavior, or `"char"` to match each non-whitespace character for languages without spaces between words, such as Thai, Japanese, and Chinese.
- `expand_spans`: whether to expand token-aligned matches to a larger display unit. Defaults to `True`.
- `span_expansion_regex`: regex that defines expansion units when `expand_spans=True`. Defaults to one rendered line: `r"[^\r\n]+"`.

Advanced `HighlightOptions` fields:

- `lcs_merge_gap_tokens`: how many tokens the LCS stage may bridge before the neighbor expansion stage.
- `lcs_min_run_tokens`: minimum non-stop-word token count for the internal LCS anchor stage.
- `lcs_min_single_token_chars`: minimum length for a single-token LCS run.
- `max_highlight_ratio`: collapse to the longest span when highlighted characters exceed this fraction of the source chunk.

## Advanced Guide

CiteGlow is deterministic, so tuning is about choosing how strict or generous the lexical evidence should be for your data.

### Tight vs. Display-Friendly Spans

By default, CiteGlow expands surviving matches to the containing line. This is usually better for citation UI because users see the full supporting sentence or bullet.

```python
from citeglow import find_answer_highlights

source = "Unused items can be returned within 30 days of delivery for a refund."
answer = "The refund window is 30 days for unused items."

display_spans = find_answer_highlights(answer, source)
tight_spans = find_answer_highlights(answer, source, expand_spans=False)

print([source[start:end] for start, end in display_spans])
# ["Unused items can be returned within 30 days of delivery for a refund."]

print([source[start:end] for start, end in tight_spans])
# ["Unused items can be returned within 30 days of delivery for a refund"]
```

Use `expand_spans=False` when you need exact lexical evidence offsets for storage, scoring, or tests. Use the default expansion for user-facing source previews.

### Custom Expansion Regex

`span_expansion_regex` tells CiteGlow what a display unit looks like. The default is `r"[^\r\n]+"`, which means "expand to the containing line."

If your chunks contain tags or structured blocks, expand to those instead:

```python
source = "prefix <cite>alpha beta evidence</cite> suffix"
answer = "alpha beta"

spans = find_answer_highlights(
    answer,
    source,
    span_expansion_regex=r"<cite>.*?</cite>",
)

print([source[start:end] for start, end in spans])
# ["<cite>alpha beta evidence</cite>"]
```

For paragraph expansion, use a regex that matches paragraph blocks:

```python
spans = find_answer_highlights(
    answer,
    source,
    span_expansion_regex=r"(?s)(?:^|\n\n).*?(?=\n\n|$)",
)
```

The regex is applied with Python's `re` engine. CiteGlow expands a span to the regex match that contains it. If a span crosses multiple regex matches, the start expands to the unit containing the first character and the end expands to the unit containing the last character.

### Stop Words

Stop words are removed only from "is this meaningful evidence?" and bag-of-words expansion decisions. They do not remove text from the returned source offsets.

Use the default English and Vietnamese list:

```python
from citeglow import DEFAULT_STOP_WORDS, HighlightOptions

options = HighlightOptions(stop_words=DEFAULT_STOP_WORDS)
```

Extend it for your domain:

```python
options = HighlightOptions(
    stop_words=DEFAULT_STOP_WORDS | {"section", "article", "page"},
)
```

Replace it when your language or domain needs a fully custom list:

```python
options = HighlightOptions(stop_words={"le", "la", "les", "de", "des"})
```

If important terms are currently treated as stop words, remove them from your custom set. If noisy terms repeatedly cause broad highlights, add them.

### Tokenizers

The default `unicode_word` tokenizer keeps CiteGlow's original behavior: contiguous Unicode word characters become one token. That works well for English, Vietnamese, and other text where word boundaries are explicit enough for lexical overlap.

Use `char` when matching languages that do not consistently separate words with spaces:

```python
spans = find_answer_highlights(
    answer,
    source,
    tokenizer="char",
)
```

`char` uses each non-whitespace character as a token and still returns Python character offsets into the original source.

### Practical Presets

For strict citation previews:

```python
strict_options = HighlightOptions(
    neighborhood_tokens=3,
    min_span_words=2,
    min_vocab_token_chars=3,
    expand_spans=True,
)
```

For technical support logs with short identifiers:

```python
log_options = HighlightOptions(
    neighborhood_tokens=8,
    min_span_words=1,
    min_vocab_token_chars=1,
    stop_words=DEFAULT_STOP_WORDS | {"error", "warning", "info"},
    expand_spans=True,
    span_expansion_regex=r"[^\r\n]+",
)
```

For storing exact evidence spans before rendering:

```python
exact_options = HighlightOptions(
    expand_spans=False,
    min_span_words=1,
)
```

## Best Fit

CiteGlow works best when:

- The answer is grounded in the provided source chunk.
- The answer reuses some source wording.
- You run it on the retrieved chunk or document section, not on an entire large corpus at once.
- Your UI needs evidence highlighting, not semantic citation discovery.

It is intentionally conservative. If an answer is mostly paraphrased or unsupported by the source, CiteGlow may return no spans. That is usually preferable to inventing a highlight.

## Limitations

- It is lexical, not semantic. Synonyms and heavy paraphrases may not match.
- It does not decide whether an answer is correct.
- It does not retrieve documents.
- Offsets are Python string character offsets, not byte offsets.
- The built-in stop-word list is intentionally small and currently focused on English and Vietnamese. Tune it for your domain and language mix.

## Rendering Highlights

For HTML, escape the source text before inserting tags:

```python
from html import escape


def render_highlights(text: str, spans: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(escape(text[cursor:start]))
        parts.append(f"<mark>{escape(text[start:end])}</mark>")
        cursor = end
    parts.append(escape(text[cursor:]))
    return "".join(parts)
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
