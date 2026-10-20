# Stabilize inference, validate structured output and reduce compatibility heuristics

## Problem and behavior

Pydantic output previously checked JSON syntax without validating returned values,
and custom schema introspection discarded aliases and constraints. Native file
inputs bypassed Chat validation, model-name heuristics rewrote local requests,
and token counts were only exposed with verbose statistics enabled.

This prepares version 2.1.0 with a bounded inference API: Pydantic v2 output is
validated locally, file inputs share one normalization path, explicit sampling
parameters are preserved, usage is always available, and async stream closure
propagates to the underlying Responses stream. Runtime dependencies are unchanged.

## Changes

- Use public OpenAI schema conversion for Pydantic models and validate JSON before
  emitting answers on every sync/async Chat/Responses inference path.
- Remove model-name-based sampling suppression and local token-limit rewrites.
- Consolidate native file validation, preserve Pillow globals and enforce image
  send limits. Harden tool-history conversion and streamed tool-call tracking.
- Report real/unknown token usage independently of verbose chunk statistics.
- Add HTTP/SSE contract tests, executable examples, explicit compatibility limits,
  release notes, Windows and minimum-OpenAI CI coverage, and corrected package URLs.

## Compatibility notes

Answers remain dictionaries/JSON values. Pydantic v2 validation now rejects
invalid values that previously passed. Plain classes and raw schemas retain
JSON parsing without local schema validation. Explicit unsupported sampling
parameters now reach the provider and may be rejected there; omit them as needed.
Native inline files are canonicalized and ambiguous sources fail locally.
See CHANGELOG.md for migration details.

## Validation

Local validation results are recorded with the commit handoff. Automated protocol
tests use real OpenAI/HTTPX clients and in-process mock responses; live model
quality and live provider/version compatibility require the documented manual
checks. No live inference or publication is part of this change.
