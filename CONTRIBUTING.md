# Contributing to CiteGlow

Thanks for helping improve CiteGlow. The project is intentionally small: deterministic highlighting, no model calls, no embeddings, and no required runtime dependencies.

## Local Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Contribution Guidelines

- Keep the public API small and stable.
- Prefer deterministic behavior over clever matching.
- Add tests for every behavior change, especially offset boundaries.
- Include examples with short, artificial text instead of private customer or production data.
- Do not add network calls, LLM calls, embeddings, telemetry, or heavyweight dependencies to the core package.
- If you tune matching constants, explain the tradeoff in the pull request.

## Pull Request Checklist

- Tests pass locally with `python -m pytest`.
- New or changed behavior is covered by tests.
- README examples still work.
- Public API changes are documented.
- The change is scoped to highlighting behavior or project maintenance.
